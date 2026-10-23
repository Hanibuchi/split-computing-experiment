"""scripts/verify_shiba.py（Shiba et al. の再現検証）の計算部分のテスト。"""

import importlib.util
from pathlib import Path

import pytest
import torch

from splitcomp.temporal import entropy_bits, quantize


def _load():
    path = Path(__file__).resolve().parents[1] / "scripts" / "verify_shiba.py"
    spec = importlib.util.spec_from_file_location("verify_shiba", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


vs = _load()


def _video(T=30, shape=(8, 6, 6), seed=0):
    g = torch.Generator().manual_seed(seed)
    base = torch.randn(shape, generator=g) * 3
    steps = torch.randn((T, *shape), generator=g) * 0.3
    return base + steps.cumsum(0)


@pytest.mark.parametrize("bits", [2, 4, 8])
def test_entropy_matches_library(bits):
    idx, _ = quantize(torch.randn(5000), bits, "dz")
    H, zeros = vs.entropy_and_zeros(idx, bits)
    assert H == pytest.approx(entropy_bits(idx))
    assert zeros == int((idx == 0).sum())


def test_reconstruct_schemes():
    f = _video()
    for scheme in vs.SCHEMES:
        for mode in ("sym", "dz"):
            recon, coded = vs.reconstruct(f, scheme, 8, mode, "p99", 0.0, 30)
            assert recon.shape == f.shape and len(coded) == 30
            kinds = [k for k, _ in coded]
            assert kinds[0] == "I"
            assert set(kinds[1:]) == ({"I"} if scheme == "fd" else {"P"})


def test_dnd_has_no_accumulation_and_dpcm_matches_dnd_without_clipping():
    """DnD の誤差は毎フレーム 1 回分だけ。クリップしない（max）なら 8-bit の DPCM は DnD とほぼ同じ誤差になる。"""
    f = _video()
    dnd, _ = vs.reconstruct(f, "dnd", 8, "sym", "max", 0.0, 30)
    dpcm, _ = vs.reconstruct(f, "dpcm", 8, "sym", "max", 0.0, 30)
    dwd, _ = vs.reconstruct(f, "dwd", 8, "sym", "max", 0.0, 30)
    err = lambda r: (r - f).pow(2).flatten(1).mean(1)  # noqa: E731
    assert err(dpcm)[1:].mean() < 3 * err(dnd)[1:].mean()
    assert err(dwd)[-1] > err(dnd)[-1]  # DwD は誤差がたまる


def test_fdz_is_deadzone_on_every_frame():
    f = _video()
    _, coded = vs.reconstruct(f, "fd", 4, "dz", "p99", 0.0, 30)
    idx, _ = quantize(f[5], 4, "dz")
    assert torch.equal(coded[5][1], idx.flatten())


def test_frame_stats_rho():
    f = _video()
    s = vs.frame_stats(f)
    assert len(s["rho"]) == 29 and len(s["var_f"]) == 30
    # ρ は numpy の相関係数と同じ
    import numpy as np

    expected = np.corrcoef(f[1].flatten().numpy(), f[0].flatten().numpy())[0, 1]
    assert s["rho"][0] == pytest.approx(expected, abs=1e-6)


def test_conditions_cover_paper_grid():
    conds = vs.conditions("resnet50", quick=False)
    assert len(conds) == 4 * 5 * 4 * 2  # 方式 × ビット数 × θ × 量子化方式 = 160（× 12 条件 = 1,920）
    eff = vs.conditions("efficientnet_v2_s", quick=True)
    assert any(c["clip"] == "max" for c in eff) and any(c["theta"] > 0 for c in eff)


def test_comparison_verdicts():
    cmp = vs.Comparison()
    cmp.add("T", "x", "a", 0.90, 0.93, "agreement")  # 差 0.03 ≤ 0.05
    cmp.add("T", "x", "b", 0.90, 0.98, "agreement")  # 0.08 ≤ 0.10
    cmp.add("T", "x", "c", 0.90, 0.70, "agreement")
    cmp.add("T", "x", "d", 10.0, 11.0, "R")  # 相対 10%
    assert [r["verdict"] for r in cmp.rows] == ["✓", "△", "✗", "✓"]


def test_select_targets():
    assert [t[1] for t in vs.select_targets(["resnet50"])] == ["layer1", "layer2", "layer3"]
    assert vs.select_targets(["Eff-S4"])[0][:2] == ("efficientnet_v2_s", "features_4")
    assert vs.select_targets(["11"])[0][:2] == ("mobilenet_v2", "features_17")
    with pytest.raises(SystemExit):
        vs.select_targets(["nope"])

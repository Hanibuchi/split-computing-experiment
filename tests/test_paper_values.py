"""論文の数値の書き写し（splitcomp.temporal.paper_values）に誤りがないかを、論文内の整合性で確かめる。"""

import math

import pytest

from splitcomp.temporal import paper_values as pv


@pytest.mark.parametrize("key", pv.KEYS)
def test_table8_delta_is_dpcm_minus_fd(key):
    # Δ は丸める前の値の差なので、丸めた値の差とは 0.01 までずれうる
    for bits, (fd, _dwd, _dnd, dpcm, delta) in pv.TABLE8[key].items():
        assert dpcm - fd == pytest.approx(delta, abs=0.011), (key, bits)


@pytest.mark.parametrize("key", pv.KEYS)
def test_table9_differences_and_wins(key):
    hf, hsym, hdz, dsym, ddz, wins = pv.TABLE9[key]
    assert hf - hsym == pytest.approx(dsym, abs=0.011)
    assert hf - hdz == pytest.approx(ddz, abs=0.011)
    assert wins == (ddz > 0)


@pytest.mark.parametrize("key", pv.KEYS)
def test_table7_internal(key):
    var_f, var_d, ratio, rf, rd, p99, rho_calc, _rho = pv.TABLE7[key]
    assert var_f / var_d == pytest.approx(ratio, abs=0.06)
    assert rf / rd == pytest.approx(p99, abs=0.06)
    assert 1 - var_d / (2 * var_f) == pytest.approx(rho_calc, abs=0.0006)


@pytest.mark.parametrize("key", pv.KEYS)
def test_table5_from_table7(key):
    _vf, _vd, _r, rf, rd, *_ = pv.TABLE7[key]
    mse_ratio, db = pv.TABLE5[key]
    assert (rf / rd) ** 2 == pytest.approx(mse_ratio, rel=0.01)
    assert math.log2(rf / rd) == pytest.approx(db, abs=0.051)


@pytest.mark.parametrize("key", pv.KEYS)
def test_table6_is_eq27(key):
    _vf, var_d, _r, _rf, rd, *_ = pv.TABLE7[key]
    for bits, alpha in zip(pv.TABLE6_BITS, pv.TABLE6[key]):
        expected = 100 * rd**2 / (3 * 4**bits * var_d)
        if alpha is None:
            assert expected < 0.01
        else:
            assert expected == pytest.approx(alpha, rel=0.02, abs=0.006), (key, bits)


def test_claims_consistent_with_tables():
    rho = [pv.TABLE7[k][7] for k in pv.KEYS]
    assert min(rho) > pv.CLAIMS["rho_min_30fps"]
    ratios = [pv.TABLE7[k][2] for k in pv.KEYS]
    assert (min(ratios), max(ratios)) == pv.CLAIMS["var_ratio_range"]
    assert sum(pv.TABLE9[k][5] for k in pv.KEYS) == 9
    # Table 13 の ρ の平均は Table 7 の ρmeas と同じ
    assert all(pv.TABLE13[k][0] == pv.TABLE7[k][7] for k in pv.KEYS)
    # Fig. 8 の F+Sym は Table 8 の FD 2-bit
    fd, _, _, _, _ = pv.TABLE8[("efficientnet_v2_s", "features_3")][2]
    assert (fd, pv.FIG8_FDZ_READ[("efficientnet_v2_s", "features_3")]) == pv.CLAIMS["fdz_eff_s3_2bit"]


def test_shapes_match_this_framework():
    """論文の分割点名がこの基盤で同じ形状の特徴になること（Table 3）。重みは使わないので学習済みは読まない。"""
    torch = pytest.importorskip("torch")
    from splitcomp.builder import build_split
    from splitcomp.config import ModelConfig

    for model, sp, depth, *_ in pv.TARGETS[:1] + pv.TARGETS[3:4]:  # 時間がかかるので 2 つだけ
        bundle = build_split(ModelConfig(name=model, pretrained=False, trace_depth=depth), sp)
        (f,) = bundle.head.eval()(torch.zeros(1, 3, 224, 224))
        assert tuple(f.shape[1:]) == pv.TABLE3_SHAPE[(model, sp)]

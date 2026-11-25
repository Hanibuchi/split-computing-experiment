"""設定の読み込み（base による継承・コマンドライン上書き）のテスト。"""

from pathlib import Path

import pytest
import yaml

from splitcomp.config import PacketLossConfig, load_config, upgrade_legacy_rps_name, upgrade_legacy_rps_row

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def _write(path: Path, data: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_base_merges_dicts_and_replaces_lists(tmp_path):
    _write(tmp_path / "configs" / "main.yaml", {
        "experiment_name": "main",
        "data": {"name": "ucf101", "kwargs": {"num_clips": None, "size": 224}},
        "bits": [2, 4, 8],
        "seeds": [0, 1, 2],
    })
    # base は子の YAML のフォルダからの相対パス
    child = _write(tmp_path / "configs" / "quick" / "child.yaml", {
        "base": "../main.yaml",
        "experiment_name": "quick",
        "data": {"kwargs": {"num_clips": 2}},
        "bits": [4],
    })
    cfg = load_config(PacketLossConfig, child, ["seeds=[0]"])
    assert cfg.experiment_name == "quick"
    assert cfg.data.name == "ucf101"  # dict は項目ごとにマージ
    assert cfg.data.kwargs == {"num_clips": 2, "size": 224}
    assert cfg.bits == [4]  # list は丸ごと置き換え
    assert cfg.seeds == [0]  # コマンドラインの上書きが最優先


def test_base_cycle(tmp_path):
    _write(tmp_path / "a.yaml", {"base": "b.yaml"})
    _write(tmp_path / "b.yaml", {"base": "a.yaml"})
    with pytest.raises(ValueError, match="循環"):
        load_config(PacketLossConfig, tmp_path / "a.yaml")


def test_packet_loss_quick_config():
    full = load_config(PacketLossConfig, CONFIGS / "packet_loss_ucf101.yaml")
    quick = load_config(PacketLossConfig, CONFIGS / "packet_loss_ucf101_quick.yaml")
    assert quick.experiment_name != full.experiment_name
    assert [t.model.name for t in quick.targets] == ["resnet50", "vgg16"]
    assert {l.name for l in quick.losses if l.p > 0} == {"gilbert_elliott"}
    assert any(l.p == 0 for l in quick.losses)  # agreement_drop の基準
    # 減らしていない条件は本番と同じ
    assert quick.schemes == full.schemes
    assert quick.bits == full.bits
    assert quick.seeds == full.seeds
    assert quick.data == full.data


def test_packet_loss_main_config():
    """メイン実験の設定が方針（論文と同じビット数・Bernoulli と GE だけ・p_step は intra）どおりか。"""
    cfg = load_config(PacketLossConfig, CONFIGS / "packet_loss_ucf101.yaml")
    assert cfg.bits == [2, 4, 6, 8]  # 論文 Table 8 と同じ
    assert cfg.gop == 30
    assert {l.name for l in cfg.losses} == {"bernoulli", "gilbert_elliott"}
    assert sum(l.p == 0 for l in cfg.losses) == 1  # agreement_drop の基準
    assert [s.name for s in cfg.schemes] == [
        "F+Sym", "D+Sym", "D+DZ", "D+DZ+S2", "D+DZ+S5", "D+DZ+RPS", "D+DZ+RPS3", "D+DZ+FEC10", "D+DZ+FEC25",
    ]
    assert all(s.p_step == "intra" for s in cfg.schemes)
    assert all(s.restorer is None for s in cfg.schemes)


def test_legacy_prs_config_is_read_as_rps(tmp_path):
    """旧名（頭文字を取り違えた PRS）の設定も、RPS の設定として読める（以前の実行フォルダを --resume するため）。"""
    path = _write(tmp_path / "old.yaml", {"schemes": [
        {"name": "D+DZ", "scheme": "dpcm", "p_quant": "dz"},
        {"name": "D+DZ+PRS3", "scheme": "dpcm", "p_quant": "dz", "prs": True, "feedback_delay": 3},
        {"name": "D+DZ+QI+PRS", "scheme": "dpcm", "p_quant": "dz", "prs": True},
    ]})
    cfg = load_config(PacketLossConfig, path)
    assert [s.name for s in cfg.schemes] == ["D+DZ", "D+DZ+RPS3", "D+DZ+QI+RPS"]
    assert [s.rps for s in cfg.schemes] == [False, True, True]
    assert cfg.schemes[1].feedback_delay == 3
    with pytest.raises(KeyError):
        load_config(PacketLossConfig, _write(tmp_path / "both.yaml", {"schemes": [{"prs": True, "rps": True}]}))


def test_legacy_prs_result_rows_are_renamed():
    """旧名の結果の行は、列 prs → rps・方式名 PRS → RPS に読み替える（列の順は保つ。ほかの名前は変えない）。"""
    row = {"model": "resnet50", "scheme": "D+DZ+PRS3", "bits": 4, "prs": True, "feedback_delay": 3}
    assert upgrade_legacy_rps_row(row) == {"model": "resnet50", "scheme": "D+DZ+RPS3", "bits": 4, "rps": True, "feedback_delay": 3}
    assert list(upgrade_legacy_rps_row(row)) == ["model", "scheme", "bits", "rps", "feedback_delay"]
    assert upgrade_legacy_rps_row({"scheme": "D+DZ+PRS"}) == {"scheme": "D+DZ+RPS"}  # gop_curve の行（prs 列なし）
    for name in ("D+DZ", "F+Sym", "D+DZ+FEC10", "D+DZ+RPS"):
        assert upgrade_legacy_rps_name(name) == name

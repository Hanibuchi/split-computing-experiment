"""設定の読み込み（base による継承・コマンドライン上書き）のテスト。"""

from pathlib import Path

import pytest
import yaml

from splitcomp.config import PacketLossConfig, load_config

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

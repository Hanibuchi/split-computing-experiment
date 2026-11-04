"""scripts/merge_results.py の --group（Slurm の配列ジョブの結果を 1 つのフォルダにまとめる）のテスト。"""

import importlib.util
import json
from pathlib import Path


def _load():
    path = Path(__file__).resolve().parents[1] / "scripts" / "merge_results.py"
    spec = importlib.util.spec_from_file_location("merge_results", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mr = _load()


def _write_run(run_dir: Path, model: str, n: int) -> None:
    """モデル 1 つ分の実行フォルダ（results / gop_curve の jsonl と meta.json）を作る。"""
    run_dir.mkdir(parents=True)
    (run_dir / "meta.json").write_text(json.dumps({"datetime": "2026-11-02T00:00:00"}), encoding="utf-8")
    for name in ("results", "gop_curve"):
        with open(run_dir / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for i in range(n):
                f.write(json.dumps({"model": model, "split_point": "x", "i": i}) + "\n")


def test_group_members_only_dirs_with_results(tmp_path):
    _write_run(tmp_path / "resnet50", "resnet50", 2)
    _write_run(tmp_path / "vgg16", "vgg16", 3)
    (tmp_path / "figures").mkdir()  # まとめたあとの図のフォルダは含めない
    (tmp_path / "empty").mkdir()  # 開始直後に止まったフォルダも含めない
    assert [d.name for d in mr.group_members(tmp_path)] == ["resnet50", "vgg16"]


def test_merge_into_group_dir(tmp_path):
    _write_run(tmp_path / "resnet50", "resnet50", 2)
    _write_run(tmp_path / "vgg16", "vgg16", 3)
    source = mr.merge_runs(mr.group_members(tmp_path), tmp_path)
    assert set(source) == {("resnet50", "x"), ("vgg16", "x")}
    rows = mr.read_rows(tmp_path, "results")
    assert len(rows) == 5
    # もう一度まとめても（モデルを再開したあとなど）、まとめた結果自体は入力に混ざらず、行が重複しない
    mr.merge_runs(mr.group_members(tmp_path), tmp_path)
    assert len(mr.read_rows(tmp_path, "results")) == 5

"""実験結果の保存。

1 回の実験ごとに ``<output_dir>/<実験名>_<日時>/`` というディレクトリを作り、

    config.yaml    実行時の設定（コマンドライン上書きも反映済み）
    meta.json      実行環境（日時, PyTorch バージョン, デバイス, git コミット等）
    results.jsonl  結果（1 行 = 1 試行）。実行中に逐次追記されるので途中で止めても残る
    results.csv    同じ内容の CSV（終了時に書き出し。Excel / pandas で開きやすい）。
                   既にある results.jsonl に追記した場合（中断した実行の再開など）は前回分の行も含む

を保存する。
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import platform
import subprocess
from pathlib import Path
from typing import Any

import torch
import yaml


def make_run_dir(output_dir: str | Path, name: str) -> Path:
    """``<output_dir>/<name>_<YYYYmmdd-HHMMSS>/`` を作って返す。"""
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(output_dir) / f"{name}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=5
        ).stdout.strip()
    except Exception:
        return None


def save_run_info(run_dir: Path, config: dict, extra: dict | None = None) -> None:
    """設定と実行環境の情報を保存する（あとで「どの条件の結果か」を確認できるように）。"""
    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False)
    meta = {
        "datetime": dt.datetime.now().isoformat(timespec="seconds"),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "git_commit": _git_commit(),
        **(extra or {}),
    }
    with open(run_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


class ResultWriter:
    """結果を 1 行ずつ JSONL に追記し、close() で CSV にもまとめる。

    JSONL が既にあれば、その行も引き継いで CSV に含める（中断した実行を同じフォルダで再開する場合用）。

    with 文で使うと自動で close() される::

        with ResultWriter(run_dir) as w:
            w.write({"split_point": "layer1", "total_ms": 12.3})
    """

    def __init__(self, run_dir: Path, name: str = "results"):
        self.jsonl_path = Path(run_dir) / f"{name}.jsonl"
        self.csv_path = Path(run_dir) / f"{name}.csv"
        self.rows: list[dict[str, Any]] = []
        if self.jsonl_path.exists():  # 前回分の行も CSV に含める
            with open(self.jsonl_path, encoding="utf-8") as f:
                self.rows = [json.loads(line) for line in f if line.strip()]
        self._f = open(self.jsonl_path, "a", encoding="utf-8")

    def write(self, row: dict[str, Any]) -> None:
        self.rows.append(row)
        self._f.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._f.flush()

    def close(self) -> None:
        self._f.close()
        if not self.rows:
            return
        # 行ごとに列が違っても良いように、全ての行に現れた列を出現順に並べる
        columns: list[str] = []
        for r in self.rows:
            columns += [k for k in r if k not in columns]
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=columns)
            w.writeheader()
            w.writerows(self.rows)

    def __enter__(self) -> "ResultWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

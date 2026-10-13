"""モデルごとに別々に実行した結果（実行フォルダ）を 1 つにまとめる。

各フォルダの results / gop_curve を読み（途中で止まった実行でも残る .jsonl を優先し、無ければ .csv）、
連結して出力フォルダに results.csv / gop_curve.csv（と .jsonl）として保存する。
同じ (model, split_point) が複数のフォルダにある場合は、新しい実行（meta.json の日時）のほうを使う。

例:
    python scripts/merge_results.py runs/packet_loss_ucf101_*_2026* -o runs/packet_loss_ucf101_merged
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from splitcomp.metrics import ResultWriter

TABLES = ("results", "gop_curve")


def read_rows(run_dir: Path, name: str) -> list[dict]:
    """run_dir の <name>.jsonl（無ければ <name>.csv）を読む。どちらも無ければ空。"""
    jsonl, csv_path = run_dir / f"{name}.jsonl", run_dir / f"{name}.csv"
    if jsonl.exists():
        with open(jsonl, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    if csv_path.exists():
        with open(csv_path, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    return []


def run_time(run_dir: Path) -> str:
    """実行の日時（meta.json。無ければフォルダ名の末尾の日時）。新旧の比較に使う。"""
    meta = run_dir / "meta.json"
    if meta.exists():
        return json.loads(meta.read_text(encoding="utf-8")).get("datetime", "")
    return run_dir.name.rsplit("_", 1)[-1]


def merge_runs(run_dirs: list[Path], out_dir: Path) -> dict[tuple, Path]:
    """run_dirs の結果をまとめて out_dir に保存し、{(model, split_point): 採用したフォルダ} を返す。"""
    run_dirs = sorted((Path(d) for d in run_dirs), key=run_time)
    source: dict[tuple, Path] = {}  # 新しい実行で上書きされていく
    for d in run_dirs:
        for r in read_rows(d, "results"):
            key = (r["model"], r["split_point"])
            if key in source and source[key] != d:
                print(f"警告: {key[0]}/{key[1]} が {source[key]} と {d} の両方にあります。新しい {d} を使います")
            source[key] = d
    if not source:
        raise SystemExit("結果が見つかりません（results.jsonl / results.csv のあるフォルダを指定してください）")

    out_dir.mkdir(parents=True, exist_ok=True)
    for name in TABLES:
        for ext in ("jsonl", "csv"):
            (out_dir / f"{name}.{ext}").unlink(missing_ok=True)  # ResultWriter は追記なので先に消す
        with ResultWriter(out_dir, name) as w:
            for d in run_dirs:
                for r in read_rows(d, name):
                    if source.get((r["model"], r["split_point"])) == d:
                        w.write(r)
    with open(out_dir / "sources.txt", "w", encoding="utf-8") as f:
        for (model, split_point), d in source.items():
            f.write(f"{model}/{split_point}\t{d}\n")
    return source


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dirs", nargs="+", type=Path, help="まとめる実行フォルダ（runs/<実験名>_<日時>）")
    parser.add_argument("-o", "--out", type=Path, required=True, help="出力フォルダ")
    args = parser.parse_args()

    run_dirs = [d for d in args.run_dirs if d.resolve() != args.out.resolve()]  # glob に出力先が混ざっても無視する
    source = merge_runs(run_dirs, args.out)
    for (model, split_point), d in source.items():
        print(f"  {model}/{split_point}: {d}")
    print(f"まとめた結果: {args.out / 'results.csv'}\nGoP 位置ごと: {args.out / 'gop_curve.csv'}")


if __name__ == "__main__":
    main()

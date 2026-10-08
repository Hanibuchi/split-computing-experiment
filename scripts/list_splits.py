"""分割点の一覧を表示する。

各分割点で切ったときに送るテンソルの形・データ量（float32 換算）と、
head / tail の計算時間を表で表示する。実験でどの分割点を試すか決めるのに使う。

例:
    python scripts/list_splits.py model.name=resnet18
    python scripts/list_splits.py model.name=resnet50 model.trace_depth=2 --kind all
    python scripts/list_splits.py --config configs/sweep_resnet50.yaml --no-timing
"""

from __future__ import annotations

import copy

from splitcomp.cli import parse_cli
from splitcomp.config import SweepConfig
from splitcomp.data import build_dataloader
from splitcomp.models import build_model
from splitcomp.profiling import profile_split_points
from splitcomp.split import Splitter
from splitcomp.utils import format_table, resolve_device, set_seed


def main() -> None:
    def extra(p):
        p.add_argument("--kind", choices=["clean", "all"], default="clean", help="clean: 送信テンソルが1つの分割点のみ / all: 全て")
        p.add_argument("--no-timing", action="store_true", help="計算時間を計測しない（速い）")
        p.add_argument("--csv", default=None, help="結果を CSV に保存するパス")

    cfg, args = parse_cli(SweepConfig, __doc__, extra_args=extra)
    set_seed(cfg.seed)
    head_dev, tail_dev = resolve_device(cfg.devices.head), resolve_device(cfg.devices.tail)

    model = build_model(cfg.model.name, cfg.model.pretrained, **cfg.model.kwargs)
    splitter = Splitter(copy.deepcopy(model), cfg.model.trace_depth)
    x, _ = next(iter(build_dataloader(cfg.data)))

    points = splitter.split_points(args.kind)
    rows = profile_split_points(splitter, points, x, head_dev, tail_dev, measure_time=not args.no_timing)
    for r in rows:
        r["KB_fp32"] = r["bytes_fp32"] / 1024
    cols = ["index", "name", "num_tensors", "shapes", "KB_fp32", "ratio_to_input"]
    if not args.no_timing:
        cols += ["head_ms", "tail_ms"]

    print(f"モデル: {cfg.model.name}  入力: {tuple(x.shape)}  head={head_dev} tail={tail_dev}")
    print(f"分割点 {len(rows)} 個（kind={args.kind}、全体では {len(splitter.split_points('all'))} 個）")
    print("  ratio_to_input: 入力をそのまま送る場合に対する送信データ量の比（1 未満なら入力より小さい）\n")
    print(format_table(rows, cols))

    if args.csv:
        import csv

        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\nCSV に保存しました: {args.csv}")


if __name__ == "__main__":
    main()

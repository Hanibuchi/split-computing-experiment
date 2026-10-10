"""分割点 × 圧縮方式 × 通信路 を総当たりで評価するシミュレーション実験。

1 プロセス内で head と tail を実行し、通信時間は帯域・RTT から計算する。
各組み合わせについて、遅延の内訳・送信データ量・精度・元モデルとの一致度を
runs/<実験名>_<日時>/results.csv に 1 行ずつ保存する。

例:
    python scripts/sweep.py --config configs/sweep_resnet50.yaml
    python scripts/sweep.py --config configs/sweep_resnet50.yaml model.name=resnet18 data.num_samples=8
"""

from __future__ import annotations

import copy
import statistics
from collections import defaultdict

import torch

from splitcomp.builder import build_split
from splitcomp.cli import parse_cli
from splitcomp.compression import build_codec
from splitcomp.config import SweepConfig, to_dict
from splitcomp.data import build_dataloader
from splitcomp.metrics import Evaluator, ResultWriter, make_run_dir, save_run_info
from splitcomp.models import build_model
from splitcomp.split import LocalBackend, SplitPipeline, Splitter
from splitcomp.transport import SimulatedChannel
from splitcomp.utils import format_table, resolve_device, set_seed

# 計測値のうち、バッチごとの平均を結果に載せるもの
_STAT_KEYS = ["head_ms", "encode_ms", "decode_ms", "tail_ms", "payload_bytes", "raw_bytes", "result_bytes"]


def main() -> None:
    cfg, _ = parse_cli(SweepConfig, __doc__)
    set_seed(cfg.seed)
    head_dev, tail_dev = resolve_device(cfg.devices.head), resolve_device(cfg.devices.tail)
    run_dir = make_run_dir(cfg.output_dir, cfg.experiment_name)
    save_run_info(run_dir, to_dict(cfg), {"head_device": str(head_dev), "tail_device": str(tail_dev)})
    print(f"結果の保存先: {run_dir}")

    # --- データは最初に全部読み込んでおく（分割点ごとに読み直すと遅いため。num_samples を大きくしすぎないこと）
    batches = list(build_dataloader(cfg.data))

    # --- 分割点ごとの (名前, index, 送信テンソル数, head, tail) を用意する
    if cfg.bottleneck_ckpt:
        bundle = build_split(cfg.model, "", cfg.bottleneck_ckpt)
        model = bundle.model
        splits = [(bundle.point.name + "+bottleneck", bundle.point.index, 1, bundle.head, bundle.tail)]
    else:
        model = build_model(cfg.model.name, cfg.model.pretrained, **cfg.model.kwargs)
        splitter = Splitter(copy.deepcopy(model), cfg.model.trace_depth)
        points = splitter.resolve(cfg.split_points)
        # head/tail は必要になったときに作る（全分割点分を一度に持つとメモリを食うため）
        splits = [(p.name, p.index, p.num_tensors, None, p) for p in points]

    # --- 比較の基準となる元モデルの出力を先に計算しておく
    model.to(tail_dev).eval()
    with torch.no_grad():
        references = [model(x.to(tail_dev)) for x, _ in batches]

    channels = [SimulatedChannel.from_config(c) for c in cfg.channels]
    codecs = [build_codec(c) for c in cfg.codecs]
    print(f"分割点 {len(splits)} 個 × 圧縮方式 {len(codecs)} 個 × 通信路 {len(channels)} 個 を評価します")

    summary = []
    with ResultWriter(run_dir) as writer:
        for name, index, num_tensors, head, tail_or_point in splits:
            if head is None:
                head, tail = splitter.split(tail_or_point)
            else:
                tail = tail_or_point
            head.to(head_dev).eval()
            tail.to(tail_dev).eval()

            for codec in codecs:
                pipeline = SplitPipeline(head, codec, LocalBackend(tail, codec, tail_dev), head_dev)
                for _ in range(cfg.warmup):  # 空回し（初回の遅い実行を計測から除く）
                    pipeline.run(batches[0][0])

                evaluator = Evaluator()
                per_batch = defaultdict(list)
                for (x, y), ref in zip(batches, references):
                    out, stats = pipeline.run(x)
                    evaluator.update(out, y, ref)
                    for k in _STAT_KEYS:
                        per_batch[k].append(stats[k])
                metrics = evaluator.compute()
                means = {k: statistics.mean(v) for k, v in per_batch.items()}

                # 通信時間はデータ量から計算できるので、推論をやり直さずに全通信路分を求める
                for ch in channels:
                    up = statistics.mean(ch.uplink_ms(b) for b in per_batch["payload_bytes"])
                    down = statistics.mean(ch.downlink_ms(b) for b in per_batch["result_bytes"])
                    compute = means["head_ms"] + means["encode_ms"] + means["decode_ms"] + means["tail_ms"]
                    row = {
                        "model": cfg.model.name,
                        "split_point": name,
                        "split_index": index,
                        "num_tensors": num_tensors,
                        "codec": codec.name,
                        "channel": ch.name,
                        "bandwidth_mbps": ch.bandwidth_mbps,
                        "rtt_ms": ch.rtt_ms,
                        "batch_size": cfg.data.batch_size,
                        "num_samples": sum(len(b[1]) for b in batches),
                        **means,
                        "compression_ratio": means["raw_bytes"] / means["payload_bytes"],
                        "uplink_ms": up,
                        "downlink_ms": down,
                        "total_ms": compute + up + down,
                        **metrics,
                    }
                    writer.write(row)
                    summary.append(row)
                print(
                    f"  {name:<28} {codec.name:<14} payload={means['payload_bytes'] / 1024:9.1f} KB  "
                    + "  ".join(f"{k}={v:.4g}" for k, v in metrics.items())
                )
            # 使い終わった head/tail を解放
            del head, tail, pipeline

    # --- 通信路ごとに、エンドツーエンド遅延が小さい組み合わせ上位 5 件を表示
    for ch in channels:
        best = sorted((r for r in summary if r["channel"] == ch.name), key=lambda r: r["total_ms"])[:5]
        print(f"\n[{ch.name}] 遅延が小さい組み合わせ（上位 5 件）")
        cols = ["split_point", "codec", "payload_bytes", "head_ms", "tail_ms", "uplink_ms", "total_ms"]
        cols += [k for k in ("top1", "out_agree") if k in best[0]]
        print(format_table(best, cols))
    print(f"\n全結果: {run_dir / 'results.csv'}")


if __name__ == "__main__":
    main()

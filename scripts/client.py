"""実ネットワーク実験のクライアント（端末）側: head を実行してサーバに送り、結果を受け取る。

先に scripts/serve.py でサーバを起動しておくこと（同じ設定ファイルを使う）。
バッチごとの遅延内訳を runs/<実験名>_<日時>/results.csv に保存し、最後に平均を表示する。

例（端末側で実行）:
    python scripts/client.py --config configs/network.yaml host=192.168.0.10
"""

from __future__ import annotations

import statistics

import torch

from splitcomp.builder import build_split, config_fingerprint
from splitcomp.cli import parse_cli
from splitcomp.compression import build_codec
from splitcomp.config import NetworkConfig, to_dict
from splitcomp.data import build_dataloader
from splitcomp.metrics import Evaluator, ResultWriter, make_run_dir, save_run_info
from splitcomp.split import SplitPipeline
from splitcomp.transport import TCPBackend
from splitcomp.utils import resolve_device, set_seed


def main() -> None:
    def extra(p):
        p.add_argument("--no-reference", action="store_true", help="元モデルとの一致度（忠実度）を計算しない（端末側の負荷を減らす）")

    cfg, args = parse_cli(NetworkConfig, __doc__, extra_args=extra)
    set_seed(cfg.seed)
    device = resolve_device(cfg.devices.head)
    run_dir = make_run_dir(cfg.output_dir, cfg.experiment_name)
    save_run_info(run_dir, to_dict(cfg), {"head_device": str(device)})

    bundle = build_split(cfg.model, cfg.split_point, cfg.bottleneck_ckpt)
    head = bundle.head.to(device).eval()
    reference_model = None if args.no_reference else bundle.model.to(device).eval()
    codec = build_codec(cfg.codec)
    fp = config_fingerprint(cfg.model, cfg.split_point, cfg.codec, cfg.bottleneck_ckpt)
    batches = list(build_dataloader(cfg.data))

    print(f"[client] {cfg.host}:{cfg.port} に接続します（分割点 {bundle.point.name}, 圧縮方式 {codec.name}）")
    with TCPBackend(cfg.host, cfg.port, fp, output_device=device) as backend, ResultWriter(run_dir) as writer:
        pipeline = SplitPipeline(head, codec, backend, device)
        for _ in range(cfg.warmup):
            pipeline.run(batches[0][0])

        evaluator = Evaluator()
        for i, (x, y) in enumerate(batches):
            out, stats = pipeline.run(x)
            ref = None
            if reference_model is not None:
                with torch.no_grad():
                    ref = reference_model(x.to(device))
            evaluator.update(out, y, ref)
            writer.write({"batch": i, "split_point": bundle.point.name, "codec": codec.name, **stats})

    keys = ["head_ms", "encode_ms", "payload_bytes", "comm_ms", "decode_ms", "tail_ms", "total_ms"]
    print("\n[client] 平均（バッチあたり）")
    for k in keys:
        vals = [r[k] for r in writer.rows]
        print(f"  {k:<14} {statistics.mean(vals):10.3f}  (中央値 {statistics.median(vals):.3f})")
    for k, v in evaluator.compute().items():
        print(f"  {k:<14} {v:10.4f}")
    print(f"\n結果: {run_dir / 'results.csv'}")


if __name__ == "__main__":
    main()

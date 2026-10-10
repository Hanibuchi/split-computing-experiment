"""分割点に学習可能なボトルネック（エンコーダ / デコーダ）を挿入し、元モデルからの知識蒸留で学習する。

学習後は runs/<実験名>_<日時>/bottleneck.pt が保存される。これを
sweep.py / serve.py / client.py の bottleneck_ckpt に指定すると、ボトルネック付きで評価できる。

例:
    python scripts/train_bottleneck.py --config configs/train_bottleneck.yaml
    python scripts/train_bottleneck.py --config configs/train_bottleneck.yaml \\
        split_point=layer3 bottleneck.channels=8 epochs=5
"""

from __future__ import annotations

import copy

import torch

from splitcomp.builder import save_bottleneck
from splitcomp.cli import parse_cli
from splitcomp.compression import build_bottleneck, infer_feature_shape
from splitcomp.config import TrainConfig, to_dict
from splitcomp.data import build_dataloader
from splitcomp.metrics import make_run_dir, save_run_info
from splitcomp.models import build_model
from splitcomp.split import Splitter
from splitcomp.training import BottleneckTrainer
from splitcomp.utils import resolve_device, set_seed


def main() -> None:
    cfg, _ = parse_cli(TrainConfig, __doc__)
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    run_dir = make_run_dir(cfg.output_dir, cfg.experiment_name)
    save_run_info(run_dir, to_dict(cfg), {"device": str(device)})
    print(f"結果の保存先: {run_dir}")

    teacher = build_model(cfg.model.name, cfg.model.pretrained, **cfg.model.kwargs)
    # 生徒側の head/tail は複製から作る（学習で重みを変えても教師に影響しないように）
    splitter = Splitter(copy.deepcopy(teacher), cfg.model.trace_depth)
    head, tail = splitter.split(cfg.split_point)

    train_loader = build_dataloader(cfg.data)
    val_loader = build_dataloader(cfg.val_data) if cfg.val_data else None

    x, _ = next(iter(train_loader))
    feature_shape = infer_feature_shape(head.eval(), x[:1])
    encoder, decoder = build_bottleneck(feature_shape, cfg.bottleneck)
    with torch.no_grad():
        z_shape = list(encoder.eval()(torch.zeros(1, *feature_shape)).shape[1:])
    n_feat, n_z = torch.Size(feature_shape).numel(), torch.Size(z_shape).numel()
    print(f"分割点 {cfg.split_point}: 特徴 {feature_shape} → ボトルネック {z_shape}（要素数 1/{n_feat / n_z:.1f}）")

    trainer = BottleneckTrainer(cfg, teacher, head, tail, encoder, decoder, device)

    def save(epoch: int, log: dict) -> None:
        save_bottleneck(
            run_dir / "bottleneck.pt",
            cfg.model,
            cfg.split_point,
            cfg.bottleneck,
            feature_shape,
            trainer.encoder,
            trainer.decoder,
            # 学習した場合のみ head/tail の重みも保存する
            head=trainer.head if not cfg.freeze_head else None,
            tail=trainer.tail if not cfg.freeze_tail else None,
            extra={"epoch": epoch, "log": log},
        )

    trainer.fit(train_loader, val_loader, run_dir, on_epoch_end=save)
    print(f"チェックポイント: {run_dir / 'bottleneck.pt'}")


if __name__ == "__main__":
    main()

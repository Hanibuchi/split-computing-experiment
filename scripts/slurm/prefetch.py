"""packet_loss_sweep.py で使う動画とモデルの学習済み重みを、ジョブを投げる前にダウンロードしておく。

Slurm の配列ジョブで複数のモデルを同時に動かすと、各ジョブが同じ動画・重みを同時にダウンロードして
書き込みが競合することがある。ログインノード（ログインノード）でこのスクリプトを 1 回実行しておけば、
ジョブは手元のファイルを読むだけになる。計算はしない（モデルを作って捨てるだけ）ので数分で終わる。

保存先:
    動画: 設定の data.root（既定 ./data/UCF-101/<クラス名>/）
    重み: torchvision の既定のキャッシュ（~/.cache/torch/hub/checkpoints/。TORCH_HOME で変えられる）

例:
    python scripts/slurm/prefetch.py
    python scripts/slurm/prefetch.py --config configs/packet_loss_ucf101_quick.yaml
"""

from __future__ import annotations

from splitcomp.cli import parse_cli
from splitcomp.config import PacketLossConfig
from splitcomp.data import build_dataset
from splitcomp.models import build_model


def main() -> None:
    cfg, _ = parse_cli(PacketLossConfig, __doc__, default_config="configs/packet_loss_ucf101.yaml")

    # データセットを作ると、手元に無い動画がダウンロードされる
    ds = build_dataset(cfg.data)
    print(f"動画: {len(ds)} 本（{cfg.data.root}）")

    # 同じモデルが複数の target にあっても 1 回だけ作る
    for name, model_cfg in {t.model.name: t.model for t in cfg.targets}.items():
        build_model(model_cfg.name, model_cfg.pretrained, **model_cfg.kwargs)
        print(f"モデル: {name}")
    print("準備ができました")


if __name__ == "__main__":
    main()

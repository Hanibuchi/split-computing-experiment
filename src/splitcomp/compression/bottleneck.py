"""分割点に挿入する学習可能なボトルネック（エンコーダ / デコーダ）。

    入力 → head → [Encoder] → (送信: 小さい特徴 z) → [Decoder] → tail → 出力

エンコーダは端末側で特徴を小さく（チャネル数削減・空間方向の縮小）し、
デコーダはサーバ側で元の特徴の形に戻す。学習は splitcomp.training で行う。

- 畳み込み特徴（B, C, H, W）: :class:`ConvBottleneck` 系（チャネル削減 + stride による縮小）
- それ以外（B, D）や（B, N, D）: :class:`LinearBottleneck` 系（最後の次元を圧縮）

量子化などの Codec は、エンコーダ出力 z に対してさらに適用される。
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from ..config import BottleneckConfig


def build_bottleneck(feature_shape: list[int] | tuple[int, ...], cfg: BottleneckConfig) -> tuple[nn.Module, nn.Module]:
    """特徴の形（バッチ次元を除く）から適切なエンコーダ / デコーダを作る。

    Args:
        feature_shape: head 出力 1 サンプル分の形。例: (128, 28, 28)。
        cfg: ボトルネック設定。

    Returns:
        (encoder, decoder)
    """
    if len(feature_shape) == 3:
        c, h, w = feature_shape
        return _conv_encoder(c, cfg), _conv_decoder(c, (h, w), cfg)
    d = feature_shape[-1]
    hidden = cfg.hidden_channels or d
    encoder = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, cfg.channels))
    decoder = nn.Sequential(nn.Linear(cfg.channels, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, d))
    return encoder, decoder


def _num_stride2(downsample: int) -> int:
    n = int(math.log2(downsample))
    if 2**n != downsample:
        raise ValueError(f"downsample は 1, 2, 4, 8 ... (2 のべき乗) で指定してください: {downsample}")
    return n


def _conv_encoder(in_ch: int, cfg: BottleneckConfig) -> nn.Module:
    hidden = cfg.hidden_channels or in_ch
    layers: list[nn.Module] = [nn.Conv2d(in_ch, hidden, 3, padding=1), nn.BatchNorm2d(hidden), nn.ReLU(inplace=True)]
    # downsample=2^n のとき stride=2 の畳み込みを n 回重ねて空間方向を縮小する
    for _ in range(_num_stride2(cfg.downsample)):
        layers += [nn.Conv2d(hidden, hidden, 3, stride=2, padding=1), nn.BatchNorm2d(hidden), nn.ReLU(inplace=True)]
    layers.append(nn.Conv2d(hidden, cfg.channels, 1))  # 最後にチャネル数を絞る（ここが送信される）
    return nn.Sequential(*layers)


def _conv_decoder(out_ch: int, size: tuple[int, int], cfg: BottleneckConfig) -> nn.Module:
    hidden = cfg.hidden_channels or out_ch
    layers: list[nn.Module] = [nn.Conv2d(cfg.channels, hidden, 1), nn.BatchNorm2d(hidden), nn.ReLU(inplace=True)]
    if cfg.downsample > 1:
        # 元の空間サイズに正確に戻す（奇数サイズでもずれないよう size を直接指定）
        layers.append(nn.Upsample(size=size, mode="nearest"))
    layers += [nn.Conv2d(hidden, hidden, 3, padding=1), nn.BatchNorm2d(hidden), nn.ReLU(inplace=True)]
    # 出力には活性化関数を付けない（元の特徴が負の値を取る場合もあるため）
    layers.append(nn.Conv2d(hidden, out_ch, 3, padding=1))
    return nn.Sequential(*layers)


class HeadWithEncoder(nn.Module):
    """head の後ろにエンコーダを付けたもの（端末側で実行される）。出力は head と同じく tuple。"""

    def __init__(self, head: nn.Module, encoder: nn.Module):
        super().__init__()
        self.head = head
        self.encoder = encoder

    def forward(self, x):
        (feat,) = self.head(x)  # clean な分割点なので送信テンソルは 1 つ
        return (self.encoder(feat),)


class TailWithDecoder(nn.Module):
    """tail の前にデコーダを付けたもの（サーバ側で実行される）。"""

    def __init__(self, decoder: nn.Module, tail: nn.Module):
        super().__init__()
        self.decoder = decoder
        self.tail = tail

    def forward(self, z):
        return self.tail(self.decoder(z))


def attach_bottleneck(
    head: nn.Module, tail: nn.Module, encoder: nn.Module, decoder: nn.Module
) -> tuple[HeadWithEncoder, TailWithDecoder]:
    """head / tail にエンコーダ / デコーダを付けた新しい (head, tail) を返す。"""
    return HeadWithEncoder(head, encoder), TailWithDecoder(decoder, tail)


@torch.no_grad()
def infer_feature_shape(head: nn.Module, sample: torch.Tensor) -> list[int]:
    """サンプル入力を head に通し、送信テンソルの形（バッチ次元を除く）を調べる。"""
    out = head(sample)
    if len(out) != 1 or not isinstance(out[0], torch.Tensor):
        raise ValueError(
            f"ボトルネックは送信テンソルが 1 つの分割点（clean な分割点）にしか挿入できません（この分割点では {len(out)} 個）"
        )
    return list(out[0].shape[1:])

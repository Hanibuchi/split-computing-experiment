"""時系列符号化で使うスカラー量子化（Shiba et al., IEEE Access 2026 の §III-C）。

- 対称量子化: [-R, R] を 2^b 段階に分け、ステップ幅 q = 2R / 2^b で ``round(x / q)`` する。
- dead-zone 量子化: 0 のビンを 1.5 倍（|x| < 0.75q → 0）に広げ、それ以外のビン幅は q のまま。
  ここでは ``sign(x) · floor(|x| / q + 0.25)`` とし、[0.75q, 1.75q) → 1, [1.75q, 2.75q) → 2 ... とする。
- 範囲 R は |x| の 99 パーセンタイル（P99）。範囲外の値はクリップされる。

インデックスは b ビットで送れるように [-2^(b-1), 2^(b-1) - 1] に収める（送信時は +2^(b-1) して符号なしにする）。
"""

from __future__ import annotations

import torch

#: torch.quantile が扱える要素数の上限（これより大きいテンソルは間引いて P99 を求める）
_QUANTILE_MAX = 1 << 24


def p99_range(x: torch.Tensor) -> float:
    """|x| の 99 パーセンタイル。全て 0 のときは 1e-8 を返す（ステップ幅 0 を避けるため）。"""
    a = x.detach().abs().flatten().float()
    if a.numel() > _QUANTILE_MAX:
        a = a[:: a.numel() // _QUANTILE_MAX + 1]
    return max(float(torch.quantile(a, 0.99)), 1e-8)


def step_size(R: float, bits: int) -> float:
    """b ビット対称量子化のステップ幅 q = 2R / 2^b。"""
    return 2.0 * R / (1 << bits)


def _clamp_index(idx: torch.Tensor, bits: int) -> torch.Tensor:
    half = 1 << (bits - 1)
    return idx.clamp_(-half, half - 1).to(torch.int32)


def symmetric_quantize(x: torch.Tensor, bits: int, q: float) -> torch.Tensor:
    """対称量子化のインデックス ``round(x / q)``（int32）。"""
    return _clamp_index(torch.round(x / q), bits)


def deadzone_quantize(x: torch.Tensor, bits: int, q: float) -> torch.Tensor:
    """dead-zone 量子化のインデックス（|x| < 0.75q は 0）。"""
    return _clamp_index(torch.sign(x) * torch.floor(x.abs() / q + 0.25), bits)


def dequantize(idx: torch.Tensor, q: float) -> torch.Tensor:
    """インデックスを値に戻す（idx · q）。"""
    return idx.float() * q


QUANTIZERS = {"sym": symmetric_quantize, "dz": deadzone_quantize}


def quantize(x: torch.Tensor, bits: int, mode: str = "sym") -> tuple[torch.Tensor, float]:
    """P99 で範囲を決めて量子化し、(インデックス, ステップ幅 q) を返す。"""
    if mode not in QUANTIZERS:
        raise KeyError(f"未知の量子化方式です: {mode!r}（使えるもの: {sorted(QUANTIZERS)}）")
    q = step_size(p99_range(x), bits)
    return QUANTIZERS[mode](x, bits, q), q


def entropy_bits(idx: torch.Tensor) -> float:
    """インデックス列の 0 次エントロピー [bit/要素]（論文の式 (29)）。"""
    counts = torch.unique(idx, return_counts=True)[1].double()
    p = counts / counts.sum()
    return float(-(p * torch.log2(p)).sum())

"""時系列符号化で使うスカラー量子化（Shiba et al., IEEE Access 2026 の §III-C）。

- 対称量子化: [-R, R] を 2^b 段階に分け、ステップ幅 q = 2R / 2^b で ``round(x / q)`` する。
- dead-zone 量子化: 0 のビンを 1.5 倍（|x| < 0.75q → 0）に広げ、それ以外のビン幅は q のまま。
  ここでは ``sign(x) · floor(|x| / q + 0.25)`` とし、[0.75q, 1.75q) → 1, [1.75q, 2.75q) → 2 ... とする。
- 範囲 R は |x| の 99 パーセンタイル（P99）。範囲外の値はクリップされる。
  ``clip="max"`` にすると R = max|x|（クリップしない。論文の Table 11/12 の比較用）。
- スパース化しきい値 θ（論文の §V-D）: θ > 0 なら |x| < θ·R の値を量子化の前に 0 にする。
- ステップ幅 q を外から指定することもできる（``step=``。P フレームの q を I フレームの q に固定する方式で使う）。
  このとき R = q·2^b / 2 とみなし、範囲外の値はクリップされる。

インデックスは b ビットで送れるように [-2^(b-1), 2^(b-1) - 1] に収める（送信時は +2^(b-1) して符号なしにする）。
``full_range=True`` なら論文どおり値を [-R, R] に収め、インデックスは [-2^(b-1), 2^(b-1)]（2^b + 1 段。b ビットには収まらない）。
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


def _clamp_index(idx: torch.Tensor, bits: int, full_range: bool = False) -> torch.Tensor:
    half = 1 << (bits - 1)
    return idx.clamp_(-half, half if full_range else half - 1).to(torch.int32)


def symmetric_quantize(x: torch.Tensor, bits: int, q: float, full_range: bool = False) -> torch.Tensor:
    """対称量子化のインデックス ``round(x / q)``（int32）。"""
    return _clamp_index(torch.round(x / q), bits, full_range)


def deadzone_quantize(x: torch.Tensor, bits: int, q: float, full_range: bool = False) -> torch.Tensor:
    """dead-zone 量子化のインデックス（|x| < 0.75q は 0）。"""
    return _clamp_index(torch.sign(x) * torch.floor(x.abs() / q + 0.25), bits, full_range)


def dequantize(idx: torch.Tensor, q: float) -> torch.Tensor:
    """インデックスを値に戻す（idx · q）。"""
    return idx.float() * q


QUANTIZERS = {"sym": symmetric_quantize, "dz": deadzone_quantize}


CLIPS = ("p99", "max")


def clip_range(x: torch.Tensor, clip: str = "p99") -> float:
    """量子化の範囲 R。"p99": |x| の 99 パーセンタイル / "max": |x| の最大値（全て 0 のときは 1e-8）。"""
    if clip == "p99":
        return p99_range(x)
    if clip == "max":
        return max(float(x.detach().abs().max()), 1e-8)
    raise KeyError(f"未知のクリップ方式です: {clip!r}（使えるもの: {CLIPS}）")


def quantize(
    x: torch.Tensor,
    bits: int,
    mode: str = "sym",
    clip: str = "p99",
    theta: float = 0.0,
    full_range: bool = False,
    step: float | None = None,
) -> tuple[torch.Tensor, float]:
    """範囲 R を決めて量子化し、(インデックス, ステップ幅 q) を返す。

    Args:
        mode: "sym"（対称）/ "dz"（dead-zone）。
        clip: 範囲 R の決め方 "p99" / "max"。
        theta: スパース化しきい値。|x| < theta·R の値は量子化の前に 0 にする（0 なら何もしない）。
        full_range: True なら値を [-R, R] に収める（インデックスは ±2^(b-1)。論文どおり）。
            False なら b ビットに収まる [-2^(b-1), 2^(b-1) - 1]。
        step: ステップ幅 q。指定すると x から範囲を測らず（clip は使わない）、この q で量子化する。
    """
    if mode not in QUANTIZERS:
        raise KeyError(f"未知の量子化方式です: {mode!r}（使えるもの: {sorted(QUANTIZERS)}）")
    if step is None:
        R = clip_range(x, clip)
        q = step_size(R, bits)
    else:
        if step <= 0:
            raise ValueError(f"step は正の値にしてください: {step}")
        q = float(step)
        R = q * (1 << bits) / 2
    if theta > 0:
        x = torch.where(x.abs() < theta * R, torch.zeros_like(x), x)
    return QUANTIZERS[mode](x, bits, q, full_range), q


def entropy_bits(idx: torch.Tensor) -> float:
    """インデックス列の 0 次エントロピー [bit/要素]（論文の式 (29)）。"""
    counts = torch.unique(idx, return_counts=True)[1].double()
    p = counts / counts.sum()
    return float(-(p * torch.log2(p)).sum())

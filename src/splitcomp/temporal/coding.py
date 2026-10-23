"""GoP 構造による中間特徴の時系列符号化（Shiba et al., IEEE Access 2026 の §III-B）。

各 GoP（K フレーム）の先頭は I フレーム（特徴 f_t そのものを対称量子化）、残りは P フレーム。
P フレームで送るものは方式によって違う:

    fd   (Feature Direct)    : 毎フレーム I フレーム（時間方向の冗長性を使わない。ベースライン）
    dwd  (Delta-with-Drift)  : 真の差分 δ_t = f_t − f_{t−1} を送る。クラウドは f̂_t = f̂_{t−1} + Q(δ_t)（誤差が蓄積する）
    dpcm                     : 予測残差 δ'_t = f_t − f̂_{t−1} を送る。端末もクラウドと同じ f̂ を持つのでドリフトしない

（論文の DnD はクラウドが真の f_{t−1} を持つ理想化なので、パケットロスの評価では扱わない。）

パケットロスの扱い:
    端末（:class:`TemporalEncoder`）はロスを知らない（フィードバックなし）。そのため DPCM でロスが起きると
    端末とクラウドの f̂ がずれ、そのずれは次の I フレームまで残る。
    クラウド（:class:`TemporalDecoder`）は届かなかった要素を concealment で補う:

    hold : 直前の f̂ の値を使う（P フレームでは差分 0 とみなすのと同じ）。直前がなければ 0
    zero : 0 にする

    さらに restorer（:mod:`.restore` のロス補修 net など）を渡すと、P フレームでロスが起きたときに
    hold で埋めた結果を restorer で補い、それを出力と復号器の状態（次のフレームの基準 f̂）の両方に使う。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from .quantize import dequantize, quantize

SCHEMES = ("fd", "dwd", "dpcm")
CONCEALMENTS = ("hold", "zero")


@dataclass
class EncodedFrame:
    """1 フレーム分の送信内容。"""

    #: フレーム番号（0 始まり）
    t: int
    #: "I" または "P"
    kind: str
    #: 量子化インデックス（平らにした int32）
    idx: torch.Tensor
    #: ステップ幅 q（パケットのヘッダで送る）
    q: float
    #: 元の特徴の形
    shape: tuple


class TemporalEncoder:
    """端末側の符号化器（状態あり）。

    Args:
        scheme: "fd" / "dwd" / "dpcm"。
        bits: 量子化ビット数（I/P 共通）。
        gop: GoP 長 K。
        p_quant: P フレームの量子化方式 "sym" / "dz"。I フレームは常に対称量子化。
        clip: 量子化の範囲 R の決め方 "p99" / "max"（I/P 共通）。
        theta: スパース化しきい値（|x| < theta·R を量子化前に 0 にする。I/P 共通）。
    """

    def __init__(self, scheme: str, bits: int, gop: int = 30, p_quant: str = "sym", clip: str = "p99", theta: float = 0.0):
        if scheme not in SCHEMES:
            raise KeyError(f"未知の方式です: {scheme!r}（使えるもの: {SCHEMES}）")
        self.scheme = scheme
        self.bits = bits
        self.gop = gop
        self.p_quant = p_quant
        self.clip = clip
        self.theta = theta
        self.reset()

    def reset(self) -> None:
        self.t = 0
        self.prev_true: torch.Tensor | None = None  # f_{t−1}（dwd 用）
        self.prev_hat: torch.Tensor | None = None  # 端末が想定するクラウドの f̂_{t−1}（dpcm 用）

    def is_intra(self, t: int) -> bool:
        return self.scheme == "fd" or t % self.gop == 0

    def encode(self, f: torch.Tensor) -> EncodedFrame:
        t = self.t
        if self.is_intra(t):
            idx, q = quantize(f, self.bits, "sym", self.clip, self.theta)
            recon = dequantize(idx, q)
            kind = "I"
        else:
            base = self.prev_true if self.scheme == "dwd" else self.prev_hat
            idx, q = quantize(f - base, self.bits, self.p_quant, self.clip, self.theta)
            recon = self.prev_hat + dequantize(idx, q)
            kind = "P"
        self.prev_true = f
        self.prev_hat = recon
        self.t += 1
        return EncodedFrame(t, kind, idx.flatten(), q, tuple(f.shape))


class TemporalDecoder:
    """クラウド側の復号器（状態あり）。

    Args:
        concealment: 届かなかった要素の補い方 "hold" / "zero"。
        restorer: P フレームでロスが起きたときに欠けた要素を補う関数
            ``(prev[C,H,W], cur[C,H,W], received[C,H,W]) -> 補った特徴[C,H,W]``
            （例: :meth:`LossRestorer.restore <splitcomp.temporal.restore.LossRestorer.restore>`）。
            cur のロスした要素は concealment によらず hold（= prev）で埋めて渡す。None なら使わない。
    """

    def __init__(self, concealment: str = "hold", restorer: Callable | None = None):
        if concealment not in CONCEALMENTS:
            raise KeyError(f"未知の concealment です: {concealment!r}（使えるもの: {CONCEALMENTS}）")
        self.concealment = concealment
        self.restorer = restorer
        self.reset()

    def reset(self) -> None:
        self.prev_hat: torch.Tensor | None = None

    def decode(self, frame: EncodedFrame, received: torch.Tensor | None = None) -> torch.Tensor:
        """フレームを復元する。

        Args:
            frame: 端末が送ったフレーム。
            received: 各要素が届いたかどうか（平らにした bool, 形は frame.idx と同じ）。None なら全て届いた。
        """
        value = dequantize(frame.idx, frame.q)
        prev = self.prev_hat.flatten() if self.prev_hat is not None else torch.zeros_like(value)
        new = value if frame.kind == "I" else prev + value
        if received is not None:
            if self.restorer is not None and frame.kind == "P" and self.prev_hat is not None:
                cur = torch.where(received, new, prev).view(frame.shape)
                new = self.restorer(self.prev_hat, cur, received.view(frame.shape)).flatten()
            else:
                fill = prev if self.concealment == "hold" else torch.zeros_like(value)
                new = torch.where(received, new, fill)
        self.prev_hat = new.view(frame.shape)
        return self.prev_hat

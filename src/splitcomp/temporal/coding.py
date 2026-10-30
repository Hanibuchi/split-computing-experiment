"""GoP 構造による中間特徴の時系列符号化（Shiba et al., IEEE Access 2026 の §III-B）。

各 GoP（K フレーム）の先頭は I フレーム（特徴 f_t そのものを対称量子化）、残りは P フレーム。
P フレームで送るものは方式によって違う:

    fd   (Feature Direct)    : 毎フレーム I フレーム（時間方向の冗長性を使わない。ベースライン）
    dwd  (Delta-with-Drift)  : 真の差分 δ_t = f_t − f_{t−1} を送る。クラウドは f̂_t = f̂_{t−1} + Q(δ_t)（誤差が蓄積する）
    dpcm                     : 予測残差 δ'_t = f_t − f̂_{t−1} を送る。端末もクラウドと同じ f̂ を持つのでドリフトしない

（論文の DnD はクラウドが真の f_{t−1} を持つ理想化なので、パケットロスの評価では扱わない。）

多系列化（streams = n）:
    P フレームの参照先を直前 f_{t−1} ではなく n フレーム前 f_{t−n} にする。フレームは GoP 内の位置 t mod n で
    n 本の独立な系列に分かれ、DPCM のロスはその系列のフレームにしか伝搬しない（代わりに参照が時間的に遠くなり、
    差分が大きくなるので、同じビット数では量子化誤差が増える）。I フレームは GoP 先頭の 1 枚だけで、GoP 内の位置 1..n−1 は I フレームを
    参照する。つまり参照フレームは max(t − n, GoP 先頭)。n = 1 が通常の DPCM。

    例（n = 2, GoP = 6）:  0(I) ← 1, 0 ← 2 ← 4,  1 ← 3 ← 5

PRS（Reference Picture Selection, prs=True）:
    クラウドは、パケットが 1 つでも落ちたフレームの番号だけを NACK として端末に返す（届いたフレームについては何も返さないので、
    フィードバックはロスが起きたときの数バイトだけで、端末→クラウドの送信量は増えない）。端末は NACK を :meth:`TemporalEncoder.nack` で受け取り、
    「NACK されたフレーム、または参照先が壊れたフレーム」を壊れたとみなす（参照関係は端末が知っているので伝搬は端末で計算できる）。
    NACK がまだ来ていないフレームは無事とみなす。P フレームの参照は

        1. 通常の参照 max(t − n, GoP 先頭) 以前で、無事な最新のフレーム
        2. それが無ければ、t より前で無事な最新のフレーム（多系列で、系列の先頭側がすべて壊れたとき）
        3. それも無ければ（GoP 先頭の I フレームが壊れたとき）、そのフレームを I フレームにする

    とする。ロスがなければ prs=False と全く同じ符号化になる。参照先はパケットヘッダの「種類」u8 の残り 7 ビットに
    t − ref として入れる（GoP ≤ 128 なら足りる）ので、ヘッダは増えない。

P フレームのステップ幅（p_step）:
    adaptive : P フレームごとに、送る差分の範囲 R（既定は |δ| の P99）から q = 2R/2^b を決め直す（論文どおり）。
               差分が小さければ q も同じ割合で小さくなるので、インデックスの分布の形（= エントロピー・送信量）は
               I フレームとあまり変わらない。差分にした効果は量子化誤差が小さくなる形で表れる
    intra    : P フレームの q を、直近の I フレームの q × p_step_scale に固定する。差分が小さいほどインデックスが 0 付近に集まり、
               送信量が減る。差分が ±2^(b−1)·q を超えた要素はクリップ（飽和）される。PRS で P フレームの代わりに I フレームにしたときは、
               そのフレームの q に切り替える。q は毎フレームのパケットのヘッダで送っているので、復号器は変わらない

パケットロスの扱い:
    端末（:class:`TemporalEncoder`）はロスを知らない（フィードバックなし）。そのため DPCM でロスが起きると
    端末とクラウドの f̂ がずれ、そのずれは次の I フレームまで残る。
    クラウド（:class:`TemporalDecoder`）は届かなかった要素を concealment で補う:

    hold : P フレームでは参照フレームの f̂ の値を使う（差分 0 とみなすのと同じ。n = 1 なら直前の f̂）。
           I フレームでは直前の f̂ の値を使う。直前がなければ 0
    zero : 0 にする

    さらに restorer（:mod:`.restore` のロス補修 net など）を渡すと、P フレームでロスが起きたときに
    hold で埋めた結果を restorer で補い（prev には参照フレームの f̂ を渡す）、それを出力と復号器の状態（次のフレームの基準 f̂）の両方に使う。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from .quantize import dequantize, quantize

SCHEMES = ("fd", "dwd", "dpcm")
CONCEALMENTS = ("hold", "zero")
P_STEPS = ("adaptive", "intra")


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
    #: P フレームの参照フレーム番号（I フレームは None）
    ref: int | None = None


class TemporalEncoder:
    """端末側の符号化器（状態あり）。

    Args:
        scheme: "fd" / "dwd" / "dpcm"。
        bits: 量子化ビット数（I/P 共通）。
        gop: GoP 長 K。
        p_quant: P フレームの量子化方式 "sym" / "dz"。I フレームは常に対称量子化。
        clip: 量子化の範囲 R の決め方 "p99" / "max"（I/P 共通）。
        theta: スパース化しきい値（|x| < theta·R を量子化前に 0 にする。I/P 共通）。
        streams: 系列数 n。P フレームは max(t − n, GoP 先頭) のフレームを参照する。1 で通常の DwD / DPCM。
        full_range: True なら量子化の値を [−R, R] に収める（インデックスは ±2^(b−1) で b ビットに収まらない。論文の再現用）。
        prs: True ならクラウドからの NACK（:meth:`nack`）を受けて、クラウド側で壊れていないフレームを参照する（PRS）。fd では使えない。
        p_step: P フレームのステップ幅の決め方 "adaptive"（毎フレーム決め直す）/ "intra"（直近の I フレームの q に固定）。
        p_step_scale: p_step="intra" のときの倍率（P フレームの q = I フレームの q × p_step_scale）。
    """

    def __init__(
        self,
        scheme: str,
        bits: int,
        gop: int = 30,
        p_quant: str = "sym",
        clip: str = "p99",
        theta: float = 0.0,
        streams: int = 1,
        full_range: bool = False,
        prs: bool = False,
        p_step: str = "adaptive",
        p_step_scale: float = 1.0,
    ):
        if scheme not in SCHEMES:
            raise KeyError(f"未知の方式です: {scheme!r}（使えるもの: {SCHEMES}）")
        if streams < 1:
            raise ValueError(f"streams は 1 以上にしてください（{streams}）")
        if prs and scheme == "fd":
            raise ValueError("fd は P フレームが無いので prs は使えません")
        if p_step not in P_STEPS:
            raise KeyError(f"未知の p_step です: {p_step!r}（使えるもの: {P_STEPS}）")
        if p_step_scale <= 0:
            raise ValueError(f"p_step_scale は正の値にしてください（{p_step_scale}）")
        self.scheme = scheme
        self.bits = bits
        self.gop = gop
        self.p_quant = p_quant
        self.clip = clip
        self.theta = theta
        self.streams = streams
        self.full_range = full_range
        self.prs = prs
        self.p_step = p_step
        self.p_step_scale = p_step_scale
        self.reset()

    def reset(self) -> None:
        self.t = 0
        self.prev_true: torch.Tensor | None = None  # f_{t−1}
        self.prev_hat: torch.Tensor | None = None  # 端末が想定するクラウドの f̂_{t−1}
        # 現在の GoP のフレーム番号 → (f, f̂)（参照フレームを取り出す用）。prs=False なら直近 streams フレームだけ残す
        self.history: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        # 現在の GoP のフレーム番号 → 参照フレーム番号（I フレームは None）
        self.refs: dict[int, int | None] = {}
        # 現在の GoP で NACK を受けたフレーム番号
        self.nacked: set[int] = set()
        #: 参照できる無事なフレームが無く、P フレームの代わりに I フレームにした回数（prs=True のとき）
        self.forced_intra = 0
        #: 直近の I フレームのステップ幅 q（p_step="intra" の P フレームで使う）
        self.q_intra: float | None = None

    def nack(self, t: int) -> None:
        """クラウドから「フレーム t が壊れた」という NACK を受け取る（prs=True のときに使う）。

        すでに終わった GoP のフレームの NACK は、今後の参照に関係しないので無視する。
        """
        if t in self.refs:
            self.nacked.add(t)

    def is_broken(self, t: int) -> bool:
        """フレーム t がクラウド側で壊れていると端末が知っているか（NACK 済み、または参照をたどると NACK 済みのフレームがある）。"""
        s: int | None = t
        while s is not None:
            if s in self.nacked:
                return True
            s = self.refs[s]
        return False

    def is_intra(self, t: int) -> bool:
        return self.scheme == "fd" or t % self.gop == 0

    def ref_index(self, t: int) -> int:
        """P フレーム t の参照フレーム番号 max(t − streams, GoP 先頭)。"""
        return max(t - self.streams, t - t % self.gop)

    def select_ref(self, t: int) -> int | None:
        """P フレーム t が実際に参照するフレーム番号。None なら I フレームにする。

        prs=False なら :meth:`ref_index` そのもの。prs=True なら壊れていないフレームを選ぶ（モジュールの説明を参照）。
        """
        ref = self.ref_index(t)
        if not self.prs:
            return ref
        for candidates in (range(ref, -1, -1), range(t - 1, ref, -1)):
            for s in candidates:
                if s in self.history and not self.is_broken(s):
                    return s
        return None

    def encode(self, f: torch.Tensor) -> EncodedFrame:
        t = self.t
        if self.is_intra(t):
            # GoP の先頭: これより前のフレームはもう参照しない
            self.history.clear()
            self.refs.clear()
            self.nacked.clear()
            ref = None
        else:
            ref = self.select_ref(t)
            if ref is None:
                self.forced_intra += 1
        if ref is None:
            idx, q = quantize(f, self.bits, "sym", self.clip, self.theta, self.full_range)
            recon = dequantize(idx, q)
            kind = "I"
            self.q_intra = q
        else:
            ref_true, ref_hat = self.history[ref]
            base = ref_true if self.scheme == "dwd" else ref_hat
            step = self.q_intra * self.p_step_scale if self.p_step == "intra" else None
            idx, q = quantize(f - base, self.bits, self.p_quant, self.clip, self.theta, self.full_range, step)
            recon = ref_hat + dequantize(idx, q)
            kind = "P"
        self.prev_true = f
        self.prev_hat = recon
        self.history[t] = (f, recon)
        self.refs[t] = ref
        if not self.prs:
            self.history.pop(t - self.streams, None)  # 次のフレームの参照は t − streams + 1 以降
        self.t += 1
        return EncodedFrame(t, kind, idx.flatten(), q, tuple(f.shape), ref)


class TemporalDecoder:
    """クラウド側の復号器（状態あり）。

    Args:
        concealment: 届かなかった要素の補い方 "hold" / "zero"。
        restorer: P フレームでロスが起きたときに欠けた要素を補う関数
            ``(prev[C,H,W], cur[C,H,W], received[C,H,W]) -> 補った特徴[C,H,W]``
            （例: :meth:`LossRestorer.restore <splitcomp.temporal.restore.LossRestorer.restore>`）。
            prev は参照フレームの f̂。cur のロスした要素は concealment によらず hold（= prev）で埋めて渡す。
            None なら使わない。

    復号器は各フレームの ``frame.ref`` を見て参照フレームを選ぶので、系列数 n を知らなくてよい。
    """

    def __init__(self, concealment: str = "hold", restorer: Callable | None = None):
        if concealment not in CONCEALMENTS:
            raise KeyError(f"未知の concealment です: {concealment!r}（使えるもの: {CONCEALMENTS}）")
        self.concealment = concealment
        self.restorer = restorer
        self.reset()

    def reset(self) -> None:
        self.prev_hat: torch.Tensor | None = None
        # フレーム番号 → 復元した f̂。これから参照されうるフレームだけ残す
        self.history: dict[int, torch.Tensor] = {}

    def decode(self, frame: EncodedFrame, received: torch.Tensor | None = None) -> torch.Tensor:
        """フレームを復元する。

        Args:
            frame: 端末が送ったフレーム。
            received: 各要素が届いたかどうか（平らにした bool, 形は frame.idx と同じ）。None なら全て届いた。
        """
        value = dequantize(frame.idx, frame.q)
        if frame.kind == "I":
            # I フレームのロスは直前の f̂ で埋める
            base_hat = self.prev_hat
        else:
            # P フレームの参照（ref が無い古い形式のフレームは直前のフレームを参照する）
            base_hat = self.history[frame.ref] if frame.ref is not None else self.prev_hat
        prev = base_hat.flatten() if base_hat is not None else torch.zeros_like(value)
        new = value if frame.kind == "I" else prev + value
        if received is not None:
            if self.restorer is not None and frame.kind == "P" and base_hat is not None:
                cur = torch.where(received, new, prev).view(frame.shape)
                new = self.restorer(base_hat, cur, received.view(frame.shape)).flatten()
            else:
                fill = prev if self.concealment == "hold" else torch.zeros_like(value)
                new = torch.where(received, new, fill)
        self.prev_hat = new.view(frame.shape)
        # I フレームより前のフレームは後で参照されないので捨てる（PRS では今の ref より古いフレームを後で参照することがあるので、
        # P フレームでは捨てない）
        if frame.kind == "I":
            self.history.clear()
        self.history[frame.t] = self.prev_hat
        return self.prev_hat

"""量子化インデックスのパケット分割と、パケットロスのモデル。

パケット分割:
    1 フレームの量子化インデックスを、ペイロードが MTU に収まるように分ける。
    各パケットのヘッダ（:data:`HEADER_BYTES`）にはフレーム番号・種類・連番・オフセット・ステップ幅 q が入り、
    パケット単体で復号できる。そのため、ロスしたパケットの要素だけが欠ける。
    実験（scripts/packet_loss_sweep.py, scripts/train_restorer.py）では常にエントロピー符号化する:

    エントロピー符号化（:func:`entropy_layout`。結果は :class:`PacketLayout`）:
        パケットごとに独立した適応算術符号で符号化する。各パケットの先頭で頻度表を初期化し
        （全 2^b 記号のカウントを α=1 から始める Laplace 推定）、符号化しながら更新するので、
        付加情報なしでパケット単体で復号できる。フレーム全体をまとめて符号化すると 1 パケットのロスで
        フレーム全体が壊れるので、パケットをまたいだ符号化はしない。
        実際のビット列は作らず、符号長を正確に計算する（:func:`adaptive_code_bits`）。
        適応 Laplace モデルの符号長は記号の順序によらず、パケット内の記号のカウントだけで決まる。
        ヘッダには算術符号の復号に必要な要素数 u32 を加え（:data:`AC_HEADER_BYTES`）、
        符号の終端分として :data:`AC_TAIL_BYTES` を足す。
    固定長（:func:`packet_assignment` / :func:`num_packets` / :func:`frame_bytes`）:
        各要素を b ビット固定長で詰める（比較用の参考。ヘッダは :data:`HEADER_BYTES`）。

    interleave=False : 要素を先頭から順に詰める（1 パケット ≒ 数チャネル分の連続した領域）
    interleave=True  : 要素 i をパケット i mod P に入れる（ロスが空間的に分散する）

FEC（:func:`fec_parity` / :func:`fec_recover`）:
    パケット単位の MDS 消失訂正符号（Reed-Solomon 相当）。フレームごとに、データ k パケットの後に
    パリティ m = ceil(r·k) パケットを続けて送る。どのパケットが落ちたかは連番で分かるので、k+m 個のうち
    落ちた数が m 以下ならフレーム全体を完全に復元できる。m を超えたら復元できず、届いたデータパケットだけを使う
    （systematic 符号なので、データパケットはそのまま読める）。実際の符号化はせず、復元できるかどうかと送信量だけを計算する。
    パリティパケットの長さは一番長いデータパケット（:attr:`PacketLayout.max_bytes`）と同じとする
    （ヘッダの要素数 u32 を k, m の u16×2 に置き換えるので、ヘッダの長さも変わらない）。

損失モデル（:func:`build_loss_model` で設定から作る）:
    bernoulli       : 各パケットを独立に確率 p で落とす
    gilbert_elliott : Good / Bad の 2 状態マルコフ連鎖（Bad 中は全て落ちる）。平均損失率 p・平均バースト長 burst_len
    frame           : フレーム単位で、確率 p でそのフレームの全パケットを落とす

損失モデルの状態（Gilbert-Elliott の状態）はフレームをまたいで引き継がれる。

独自の損失モデルの登録例::

    from splitcomp.temporal import LossModel, register_loss_model

    @register_loss_model("my_loss")
    class MyLoss(LossModel):
        def __init__(self, cfg, seed): ...
        def drop(self, num_packets): ...   # True = ロスした、の bool 配列を返す
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch

from ..config import LossConfig

#: パケットヘッダのバイト数: フレーム番号 u32 + 種類 u8 + 連番 u16 + パケット数 u16 + オフセット u32 + q f32
HEADER_BYTES = 4 + 1 + 2 + 2 + 4 + 4
#: エントロピー符号化したパケットのヘッダのバイト数（:data:`HEADER_BYTES` + 要素数 u32）
AC_HEADER_BYTES = HEADER_BYTES + 4
#: 算術符号の終端（最後の区間を確定させる分）のバイト数
AC_TAIL_BYTES = 2
#: 適応モデルの頻度表の初期カウント（Laplace 推定）
AC_ALPHA = 1.0


def elements_per_packet(bits: int, mtu: int = 1400) -> int:
    """1 パケットに入る要素数。"""
    n = (mtu - HEADER_BYTES) * 8 // bits
    if n <= 0:
        raise ValueError(f"MTU {mtu} が小さすぎます（ヘッダ {HEADER_BYTES} バイト）")
    return n


def num_packets(num_elements: int, bits: int, mtu: int = 1400) -> int:
    """1 フレームのパケット数。"""
    return math.ceil(num_elements / elements_per_packet(bits, mtu))


def packet_assignment(num_elements: int, bits: int, mtu: int = 1400, interleave: bool = False) -> torch.Tensor:
    """各要素がどのパケットに入るか（長さ num_elements の int64）。"""
    P = num_packets(num_elements, bits, mtu)
    i = torch.arange(num_elements)
    return i % P if interleave else i // elements_per_packet(bits, mtu)


def packetize(num_elements: int, bits: int, mtu: int = 1400, interleave: bool = False) -> list[torch.Tensor]:
    """各パケットに入る要素の index のリスト。"""
    assign = packet_assignment(num_elements, bits, mtu, interleave)
    P = int(assign.max()) + 1 if num_elements else 0
    order = torch.argsort(assign, stable=True)
    counts = torch.bincount(assign, minlength=P).tolist()
    return list(torch.split(order, counts))


def frame_bytes(num_elements: int, bits: int, mtu: int = 1400) -> int:
    """1 フレームの送信バイト数（ヘッダ込み。各パケットのペイロードはビット詰め）。"""
    per = elements_per_packet(bits, mtu)
    P = num_packets(num_elements, bits, mtu)
    last = num_elements - per * (P - 1)
    return P * HEADER_BYTES + (P - 1) * math.ceil(per * bits / 8) + math.ceil(last * bits / 8)


def received_mask(lost: np.ndarray, assignment: torch.Tensor) -> torch.Tensor:
    """パケットごとのロス（True = ロス）から、要素ごとの「届いた」マスクを作る。"""
    return ~torch.from_numpy(lost)[assignment]


# ---------------------------------------------------------------------------
# エントロピー符号化したパケットの割り当て
# ---------------------------------------------------------------------------


@dataclass
class PacketLayout:
    """1 フレームのパケット分割の結果。"""

    #: 各要素が入るパケットの番号（長さ num_elements の int64）
    assign: torch.Tensor
    #: パケット数
    num_packets: int
    #: 送信バイト数（ヘッダ込み）
    bytes: int
    #: 一番大きいパケットのバイト数（ヘッダ込み。FEC のパリティパケットの長さ）
    max_bytes: int = 0


def adaptive_code_bits(counts: torch.Tensor, alpha: float = AC_ALPHA) -> torch.Tensor:
    """適応 Laplace モデルの算術符号の符号長 [bit]（終端分を除く理想値）。

    記号 s の確率を (それまでの s の出現数 + α) / (それまでの記号数 + αK) とすると、全体の符号長は順序によらず
    ``log2 Γ(n+αK)/Γ(αK) − Σ_s log2 Γ(c_s+α)/Γ(α)`` になる。

    Args:
        counts: 記号ごとの出現数（最後の次元が K 記号）。(P, K) なら各パケットの符号長を返す。
    """
    c = counts.double()
    K = c.shape[-1]
    n = c.sum(-1)
    nats = torch.lgamma(n + alpha * K) - math.lgamma(alpha * K) - (torch.lgamma(c + alpha) - math.lgamma(alpha)).sum(-1)
    return nats / math.log(2)


def _incremental_bits(sym: torch.Tensor, K: int, alpha: float) -> torch.Tensor:
    """sym[0] からパケットを始めたときの、各要素の符号長 [bit]（適応 Laplace モデル）。

    要素 i の符号長は −log2((sym[:i] 中の sym[i] の出現数 + α) / (i + αK))。出現数は記号で安定ソートしたときの
    グループ内の順位なので、K によらない計算量で求まる。
    """
    W = sym.numel()
    pos = torch.arange(W)
    order = torch.argsort(sym, stable=True)
    s = sym[order]
    is_start = torch.ones(W, dtype=torch.bool)
    is_start[1:] = s[1:] != s[:-1]
    group_start = torch.cummax(torch.where(is_start, pos, 0), 0).values
    rank = torch.empty(W, dtype=torch.int64)
    rank[order] = pos - group_start
    return -torch.log2((rank.double() + alpha) / (pos.double() + alpha * K))


def _ac_packet_bytes(code_bits: torch.Tensor) -> torch.Tensor:
    return AC_HEADER_BYTES + torch.ceil(code_bits / 8).long() + AC_TAIL_BYTES


def entropy_layout(idx: torch.Tensor, bits: int, mtu: int = 1400, interleave: bool = False, alpha: float = AC_ALPHA) -> PacketLayout:
    """パケットごとに独立した適応算術符号で符号化したときのパケット分割。

    interleave=False: 先頭から順に、ペイロード（MTU − ヘッダ − 終端）に収まるだけ詰める。
    interleave=True : 要素 i をパケット i mod P に入れ、全てのパケットが MTU に収まる最小に近い P を探す。

    Args:
        idx: 1 フレームの量子化インデックス（平らにした整数, 範囲 [-2^(b-1), 2^(b-1) - 1]）。
    """
    K = 1 << bits
    payload_bits = (mtu - AC_HEADER_BYTES - AC_TAIL_BYTES) * 8
    if payload_bits < bits:
        raise ValueError(f"MTU {mtu} が小さすぎます（ヘッダ {AC_HEADER_BYTES} + 終端 {AC_TAIL_BYTES} バイト）")
    sym = idx.flatten().long() + (K >> 1)
    n = sym.numel()

    if interleave:
        total = float(adaptive_code_bits(torch.bincount(sym, minlength=K)[None], alpha)[0])
        P = max(1, math.ceil(total / payload_bits))
        while True:
            assign = torch.arange(n) % P
            counts = torch.bincount(assign * K + sym, minlength=P * K).view(P, K)
            code = adaptive_code_bits(counts, alpha)
            if bool((code <= payload_bits).all()):
                sizes = _ac_packet_bytes(code)
                return PacketLayout(assign, P, int(sizes.sum()), int(sizes.max()))
            P = min(n, max(P + 1, math.ceil(P * 1.02)))

    sizes, code_bits = [], []
    s, W = 0, max(64, payload_bits // bits)
    while s < n:
        W = min(W, n - s)
        while True:
            cum = torch.cumsum(_incremental_bits(sym[s : s + W], K, alpha), 0)
            m = int(torch.searchsorted(cum, float(payload_bits), right=True))  # 収まる要素数
            if m < W or s + W == n:
                break
            W = min(2 * W, n - s)
        sizes.append(m)
        code_bits.append(float(cum[m - 1]))
        s += m
        W = m + m // 4 + 64  # 次のパケットも同じくらい入ると見込む
    assign = torch.repeat_interleave(torch.arange(len(sizes)), torch.tensor(sizes))
    packet_bytes = _ac_packet_bytes(torch.tensor(code_bits, dtype=torch.float64))
    return PacketLayout(assign, len(sizes), int(packet_bytes.sum()), int(packet_bytes.max()))


# ---------------------------------------------------------------------------
# FEC（パケット単位の消失訂正）
# ---------------------------------------------------------------------------


def fec_parity(num_data: int, rate: float) -> int:
    """データ num_data パケットに付けるパリティのパケット数 m = ceil(rate·num_data)（rate ≤ 0 なら 0）。"""
    return math.ceil(rate * num_data) if rate > 0 and num_data > 0 else 0


def fec_recover(lost: np.ndarray, num_data: int) -> np.ndarray:
    """FEC で復元した後のデータパケットのロス（長さ num_data の bool。True = ロス）。

    Args:
        lost: データ num_data パケット → パリティ m パケットの順に送ったときのロス（長さ num_data + m）。
              落ちた数が m 以下なら全て復元できる（MDS 符号）。超えたらデータパケットのロスがそのまま残る。
    """
    if int(lost.sum()) <= len(lost) - num_data:
        return np.zeros(num_data, dtype=bool)
    return lost[:num_data]


# ---------------------------------------------------------------------------
# 損失モデル
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, Callable[[LossConfig, int], "LossModel"]] = {}


def register_loss_model(name: str) -> Callable:
    """損失モデルのクラス（または ``(cfg, seed) -> LossModel`` の関数）を登録するデコレータ。"""

    def deco(cls):
        if name in _REGISTRY:
            raise KeyError(f"損失モデル名 {name!r} はすでに登録されています")
        _REGISTRY[name] = cls
        return cls

    return deco


def build_loss_model(cfg: LossConfig, seed: int = 0) -> "LossModel":
    """設定から損失モデルを作る。"""
    if cfg.name not in _REGISTRY:
        raise KeyError(f"未知の損失モデルです: {cfg.name!r}（使えるもの: {sorted(_REGISTRY)}）")
    return _REGISTRY[cfg.name](cfg, seed)


class LossModel(ABC):
    """パケットロスのモデル。:meth:`drop` を 1 フレームごとに呼ぶ。"""

    @abstractmethod
    def drop(self, num_packets: int) -> np.ndarray:
        """1 フレーム分（num_packets 個）のパケットのロスを決める。True = ロス。"""


@register_loss_model("bernoulli")
class BernoulliLoss(LossModel):
    def __init__(self, cfg: LossConfig, seed: int = 0):
        self.p = cfg.p
        self.rng = np.random.default_rng(seed)

    def drop(self, num_packets: int) -> np.ndarray:
        return self.rng.random(num_packets) < self.p


@register_loss_model("gilbert_elliott")
class GilbertElliottLoss(LossModel):
    """2 状態の Gilbert モデル。Good では落ちず、Bad では必ず落ちる。

    Bad → Good の遷移確率 r = 1 / burst_len（平均バースト長 = burst_len）、
    Good → Bad の遷移確率 g = p·r / (1 − p)（定常状態の損失率 = p）。
    最初の状態は定常分布から選ぶ。
    """

    def __init__(self, cfg: LossConfig, seed: int = 0):
        if not 0.0 <= cfg.p < 1.0:
            raise ValueError(f"gilbert_elliott の p は [0, 1) で指定してください: {cfg.p}")
        if cfg.burst_len < 1.0:
            raise ValueError(f"burst_len は 1 以上で指定してください: {cfg.burst_len}")
        self.r = 1.0 / cfg.burst_len
        self.g = cfg.p * self.r / (1.0 - cfg.p)
        self.rng = np.random.default_rng(seed)
        self.bad = bool(self.rng.random() < cfg.p)

    def drop(self, num_packets: int) -> np.ndarray:
        u = self.rng.random(num_packets)
        out = np.empty(num_packets, dtype=bool)
        bad, g, r = self.bad, self.g, self.r
        for i in range(num_packets):
            bad = (u[i] >= r) if bad else (u[i] < g)  # 状態を遷移させてから、その状態で送る
            out[i] = bad
        self.bad = bad
        return out


@register_loss_model("frame")
class FrameLoss(LossModel):
    """フレーム単位の全損失。"""

    def __init__(self, cfg: LossConfig, seed: int = 0):
        self.p = cfg.p
        self.rng = np.random.default_rng(seed)

    def drop(self, num_packets: int) -> np.ndarray:
        return np.full(num_packets, bool(self.rng.random() < self.p))

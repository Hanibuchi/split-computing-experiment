"""量子化インデックスのパケット分割と、パケットロスのモデル。

パケット分割:
    1 フレームの量子化インデックス（b ビット固定長）を、ペイロードが MTU に収まるように分ける。
    各パケットのヘッダ（:data:`HEADER_BYTES`）にはフレーム番号・種類・連番・オフセット・ステップ幅 q が入り、
    パケット単体で復号できる。そのため、ロスしたパケットの要素だけが欠ける。
    （エントロピー符号化をパケット間で行うと 1 パケットのロスでフレーム全体が壊れるので、ここでは固定長とする。）

    interleave=False : 要素を先頭から順に詰める（1 パケット ≒ 数チャネル分の連続した領域）
    interleave=True  : 要素 i をパケット i mod P に入れる（ロスが空間的に分散する）

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
from typing import Callable

import numpy as np
import torch

from ..config import LossConfig

#: パケットヘッダのバイト数: フレーム番号 u32 + 種類 u8 + 連番 u16 + パケット数 u16 + オフセット u32 + q f32
HEADER_BYTES = 4 + 1 + 2 + 2 + 4 + 4


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

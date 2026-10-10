"""分割推論パイプライン: head → encode → (通信) → decode → tail を実行し、各段階の時間とデータ量を記録する。

    ┌──────────── 端末側 ────────────┐            ┌──────── サーバ側（Backend）────────┐
    入力 → head → Codec.encode → payload ──通信──▶ Codec.decode → tail → 出力

サーバ側の処理は「Backend」として差し替えられる:
    - :class:`LocalBackend`  : 同じプロセス内で tail を実行（シミュレーション用）
    - :class:`splitcomp.transport.TCPBackend` : TCP で別マシンのサーバに依頼（実ネットワーク用）

使い方::

    pipeline = SplitPipeline(head, codec, LocalBackend(tail, codec, device), head_device=device)
    output, stats = pipeline.run(x)
    print(stats["payload_bytes"], stats["total_ms"])

stats に入る値（単位: ms / bytes）:
    head_ms        端末側で head を実行した時間
    encode_ms      ペイロードへの変換（量子化・圧縮）時間
    payload_bytes  送信データ量（ヘッダ込み）
    raw_bytes      送信テンソルを float32 のまま送った場合のデータ量（圧縮率の基準）
    decode_ms      サーバ側でペイロードを復元した時間
    tail_ms        サーバ側で tail を実行した時間
    result_bytes   サーバから返す出力のデータ量
    comm_ms        通信時間（LocalBackend で channel 未指定なら 0）
    total_ms       上記の時間の合計（エンドツーエンドの遅延）
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

from ..compression.base import Codec
from ..profiling.timer import timed
from ..utils import nbytes_fp32

if TYPE_CHECKING:  # 型ヒント専用（transport → split の循環 import を避けるため実行時には読まない）
    from ..transport.channel import SimulatedChannel


class Backend(ABC):
    """サーバ側の処理（decode → tail）を担当するものの基本クラス。"""

    @abstractmethod
    def infer(self, payload: bytes) -> tuple[Any, dict]:
        """ペイロードを受け取り (tail の出力, 計測値 dict) を返す。

        計測値には少なくとも decode_ms, tail_ms, result_bytes, comm_ms を含めること。
        """


class LocalBackend(Backend):
    """同じプロセス内で tail を実行する backend（シミュレーション用）。

    Args:
        tail: サーバ側のモデル。
        codec: head 側と同じ Codec。
        device: tail を実行するデバイス。
        channel: 通信時間を計算する通信路モデル。None なら通信時間は 0 として、
            あとから scripts/sweep.py のように複数の通信路で計算し直せるようにする。
    """

    def __init__(self, tail: nn.Module, codec: Codec, device: torch.device, channel: SimulatedChannel | None = None):
        self.tail = tail
        self.codec = codec
        self.device = device
        self.channel = channel

    def infer(self, payload: bytes) -> tuple[Any, dict]:
        values, decode_ms = timed(self.codec.decode, payload, self.device, device=self.device)
        out, tail_ms = timed(self.tail, *values, device=self.device)
        result_bytes = nbytes_fp32(out)
        stats = {"decode_ms": decode_ms, "tail_ms": tail_ms, "result_bytes": result_bytes, "comm_ms": 0.0}
        if self.channel is not None:
            stats["uplink_ms"] = self.channel.uplink_ms(len(payload))
            stats["downlink_ms"] = self.channel.downlink_ms(result_bytes)
            stats["comm_ms"] = stats["uplink_ms"] + stats["downlink_ms"]
        return out, stats


class SplitPipeline:
    """分割推論を 1 回分実行し、時間とデータ量を記録するクラス。

    Args:
        head: 端末側のモデル（出力は tuple）。
        codec: 送信時の変換方式。
        backend: サーバ側の処理（LocalBackend / TCPBackend）。
        head_device: head を実行するデバイス。
    """

    def __init__(self, head: nn.Module, codec: Codec, backend: Backend, head_device: torch.device):
        self.head = head
        self.codec = codec
        self.backend = backend
        self.head_device = head_device

    @torch.no_grad()
    def run(self, x: torch.Tensor) -> tuple[Any, dict]:
        """入力 x（バッチ）を分割推論し、(出力, 計測値 dict) を返す。"""
        x = x.to(self.head_device)
        z, head_ms = timed(self.head, x, device=self.head_device)
        payload, encode_ms = timed(self.codec.encode, list(z), device=self.head_device)
        out, bstats = self.backend.infer(payload)
        stats = {
            "head_ms": head_ms,
            "encode_ms": encode_ms,
            "payload_bytes": len(payload),
            "raw_bytes": nbytes_fp32(z),
            **bstats,
        }
        stats["total_ms"] = head_ms + encode_ms + stats["decode_ms"] + stats["tail_ms"] + stats["comm_ms"]
        return out, stats

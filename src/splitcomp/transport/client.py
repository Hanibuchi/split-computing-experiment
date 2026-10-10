"""TCP サーバに tail の実行を依頼するクライアント（端末側の処理）。

:class:`TCPBackend` は :class:`splitcomp.split.SplitPipeline` の backend として使える。
つまりシミュレーション（LocalBackend）と実ネットワーク（TCPBackend）を
同じコードで切り替えられる。

通信時間は「送信開始〜結果受信完了までの時間」から「サーバ内の処理時間」を
引いたもの（上り + 下り + 出力のシリアライズ等のオーバーヘッド）として計測する。
"""

from __future__ import annotations

import socket
import time
from typing import Any

import torch

from ..split.pipeline import Backend
from .protocol import decode_output, recv_msg, send_msg


class TCPBackend(Backend):
    """リモートの TailServer で tail を実行する backend。

    Args:
        host, port: サーバのアドレス。
        fingerprint: 設定の識別子（サーバ側と一致する必要がある）。
        output_device: 受け取った出力を置くデバイス。
        timeout: 接続・受信のタイムアウト [秒]。

    with 文で使うと、終了時に自動で切断する::

        with TCPBackend("192.168.0.10", 50051, fp) as backend:
            pipeline = SplitPipeline(head, codec, backend, head_device)
            ...
    """

    def __init__(
        self,
        host: str,
        port: int,
        fingerprint: str = "",
        output_device: torch.device | str = "cpu",
        timeout: float = 60.0,
    ):
        self.output_device = torch.device(output_device)
        self.sock = socket.create_connection((host, port), timeout=timeout)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        send_msg(self.sock, {"type": "hello", "fingerprint": fingerprint})
        reply = recv_msg(self.sock)
        if reply is None or reply[0].get("type") != "hello_ok":
            self.sock.close()
            reason = reply[0].get("message") if reply else "接続が切れました"
            raise ConnectionError(f"サーバに接続を拒否されました: {reason}")

    def infer(self, payload: bytes) -> tuple[Any, dict]:
        start = time.perf_counter()
        send_msg(self.sock, {"type": "infer"}, payload)
        reply = recv_msg(self.sock)
        roundtrip_ms = (time.perf_counter() - start) * 1e3
        if reply is None:
            raise ConnectionError("サーバとの接続が切れました")
        header, body = reply
        if header.get("type") != "result":
            raise RuntimeError(f"サーバでエラーが発生しました: {header}")
        out = decode_output(header["spec"], body, self.output_device)
        return out, {
            "decode_ms": header["decode_ms"],
            "tail_ms": header["tail_ms"],
            "result_bytes": len(body),
            "roundtrip_ms": roundtrip_ms,
            # 往復時間からサーバ内処理時間を引いたもの = 通信にかかった時間
            "comm_ms": roundtrip_ms - header["server_ms"],
        }

    def close(self) -> None:
        self.sock.close()

    def __enter__(self) -> "TCPBackend":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

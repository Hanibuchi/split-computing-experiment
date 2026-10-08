"""tail を実行する TCP サーバ（サーバ側の処理）。

scripts/serve.py から使う。クライアント（端末側）から送られてきたペイロードを
復元（decode）して tail を実行し、出力と計測値を返す。

接続は 1 クライアントずつ順番に処理する（計測値が他のクライアントの処理で乱れないように）。
"""

from __future__ import annotations

import socket
import socketserver
import time

import torch
import torch.nn as nn

from ..compression.base import Codec
from ..profiling.timer import timed
from .protocol import encode_output, recv_msg, send_msg


class TailServer:
    """tail を動かして推論要求に応える TCP サーバ。

    Args:
        tail: サーバ側で実行するモデル。
        codec: クライアントと同じ設定の Codec。
        device: tail を実行するデバイス。
        host, port: 待ち受けアドレス。port=0 なら空いているポートを自動で選ぶ。
        fingerprint: 設定の識別子。クライアントの設定と一致しない接続は拒否する。
    """

    def __init__(
        self,
        tail: nn.Module,
        codec: Codec,
        device: torch.device,
        host: str = "0.0.0.0",
        port: int = 50051,
        fingerprint: str = "",
        verbose: bool = True,
    ):
        self.tail = tail.to(device).eval()
        self.codec = codec
        self.device = device
        self.fingerprint = fingerprint
        self.verbose = verbose
        outer = self

        class Handler(socketserver.BaseRequestHandler):
            def handle(self):
                outer._handle(self.request, self.client_address)

        socketserver.TCPServer.allow_reuse_address = True
        self._server = socketserver.TCPServer((host, port), Handler)

    @property
    def address(self) -> tuple[str, int]:
        """実際に待ち受けている (host, port)。"""
        return self._server.server_address[:2]

    def serve_forever(self) -> None:
        """Ctrl+C（または shutdown()）されるまで要求を処理し続ける。"""
        self._server.serve_forever()

    def shutdown(self) -> None:
        self._server.shutdown()
        self._server.server_close()

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[server] {msg}", flush=True)

    @torch.no_grad()
    def _handle(self, sock: socket.socket, addr) -> None:
        # 小さいメッセージをすぐ送る（Nagle アルゴリズムによる遅延を避ける）
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        msg = recv_msg(sock)
        if msg is None:
            return
        header, _ = msg
        if header.get("type") != "hello" or header.get("fingerprint") != self.fingerprint:
            send_msg(sock, {"type": "error", "message": "サーバとクライアントの設定（モデル・分割点・圧縮方式）が一致しません"})
            self._log(f"{addr} の接続を拒否しました（設定の不一致）")
            return
        send_msg(sock, {"type": "hello_ok"})
        self._log(f"{addr} が接続しました")
        n = 0
        while (msg := recv_msg(sock)) is not None:
            header, payload = msg
            start = time.perf_counter()
            try:
                values, decode_ms = timed(self.codec.decode, payload, self.device, device=self.device)
                out, tail_ms = timed(self.tail, *values, device=self.device)
                spec, body = encode_output(out)
            except Exception as e:  # サーバを落とさず、エラー内容をクライアントに伝える
                self._log(f"推論中にエラー: {e!r}")
                send_msg(sock, {"type": "error", "message": repr(e)})
                continue
            server_ms = (time.perf_counter() - start) * 1e3
            send_msg(
                sock,
                {"type": "result", "spec": spec, "decode_ms": decode_ms, "tail_ms": tail_ms, "server_ms": server_ms},
                body,
            )
            n += 1
        self._log(f"{addr} が切断しました（{n} 件処理）")

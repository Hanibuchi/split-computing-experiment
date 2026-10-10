"""TCP でやりとりするメッセージの形式。

1 つのメッセージは次の形をしている::

    [4 バイト: ヘッダ長 H][4 バイト: 本体長 B][H バイト: ヘッダ(JSON)][B バイト: 本体]

- ヘッダ: メッセージ種別（hello / infer / result / error）や計測値などの小さな情報。
- 本体  : head 出力のペイロード（Codec.encode の結果）や tail の出力。

やりとりの流れ::

    クライアント                          サーバ
      hello(fingerprint)  ───────────▶   設定が一致するか確認
                          ◀───────────   hello_ok / error
      infer(payload)      ───────────▶   decode → tail
                          ◀───────────   result(計測値, 出力)
      ...（infer を繰り返す）
"""

from __future__ import annotations

import json
import socket
import struct
from typing import Any

import torch
import torch.utils._pytree as pytree

from ..compression.base import Codec, RawTensorCodec

_FRAME = struct.Struct("<II")

# tail の出力を返すときは元の値をそのまま（無劣化で）送る
_OUTPUT_CODEC = Codec(RawTensorCodec("native"), name="native")


def send_msg(sock: socket.socket, header: dict, body: bytes = b"") -> None:
    """メッセージを 1 つ送る。"""
    h = json.dumps(header, separators=(",", ":")).encode()
    sock.sendall(_FRAME.pack(len(h), len(body)) + h + body)


def recv_msg(sock: socket.socket) -> tuple[dict, bytes] | None:
    """メッセージを 1 つ受け取る。相手が接続を閉じていたら None を返す。"""
    frame = _recv_exact(sock, _FRAME.size)
    if frame is None:
        return None
    hlen, blen = _FRAME.unpack(frame)
    h = _recv_exact(sock, hlen)
    body = _recv_exact(sock, blen) if blen else b""
    if h is None or body is None:
        raise ConnectionError("メッセージの途中で接続が切れました")
    return json.loads(h), body


def _recv_exact(sock: socket.socket, n: int) -> bytes | None:
    """ちょうど n バイト受け取る（TCP は一度に全部届くとは限らないため繰り返し読む）。"""
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(min(n - len(buf), 1 << 20))
        if not chunk:
            return None if not buf else bytes(buf)
        buf += chunk
    return bytes(buf)


def encode_output(obj: Any) -> tuple[str, bytes]:
    """tail の出力（テンソル、または dict/list の入れ子）を (構造情報, バイト列) に変換する。

    物体検出モデルのように出力が dict のリストでも送れるよう、pytree で平らにしてから送る。
    """
    leaves, spec = pytree.tree_flatten(obj)
    return pytree.treespec_dumps(spec), _OUTPUT_CODEC.encode(leaves)


def decode_output(spec: str, data: bytes, device: torch.device | str = "cpu") -> Any:
    """encode_output の逆変換。"""
    leaves = _OUTPUT_CODEC.decode(data, device)
    return pytree.tree_unflatten(leaves, pytree.treespec_loads(spec))

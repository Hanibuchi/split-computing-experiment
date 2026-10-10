"""圧縮方式（Codec）の基本クラスと、送信データ（ペイロード）の形式。

圧縮は 2 段構成になっている:

    [テンソル群] --TensorCodec--> [バイト列] --ByteCodec(複数可)--> [ペイロード]

- :class:`TensorCodec`: テンソル 1 つ ⇔ バイト列。量子化などの「非可逆」な変換を担当。
- :class:`ByteCodec`  : バイト列 ⇔ バイト列。zlib などの「可逆」な汎用圧縮を担当。
- :class:`Codec`      : 上の 2 つを組み合わせ、head の出力（値のリスト）全体を 1 つのペイロードにする。

ペイロードの形式（受信側が復元に必要な情報もすべて含むので、実通信にもそのまま使える）::

    [4 バイト: ヘッダ長 N][N バイト: ヘッダ(JSON)][本体: 各テンソルのバイト列を連結し ByteCodec で圧縮]

ヘッダには各値の形状・dtype・量子化方式などが入る。ヘッダも送信データ量に含めて計測する。
head の出力に int やタプルなどテンソル以外の値が含まれる場合は、ヘッダの JSON にそのまま入れる。
"""

from __future__ import annotations

import json
import struct
from abc import ABC, abstractmethod
from typing import Any, Sequence

import torch

_HEADER_LEN = struct.Struct("<I")


class TensorCodec(ABC):
    """テンソル 1 つをバイト列に変換する方式の基本クラス。

    新しい方式を作るときはこれを継承し、``encode`` と ``decode`` を実装して
    ``@register_tensor_codec("名前")`` で登録する（docs/extending.md 参照）。
    """

    #: ヘッダに記録される方式名（受信側で同じ方式を選ぶのに使う）。
    name: str = "base"

    @abstractmethod
    def encode(self, t: torch.Tensor) -> tuple[bytes, dict]:
        """テンソルを (バイト列, 復元に必要なメタ情報 dict) に変換する。

        メタ情報は JSON にできる値（数値・文字列・リスト）だけにすること。
        大きな配列（量子化パラメータなど）はバイト列側に入れる。
        """

    @abstractmethod
    def decode(self, data: bytes, meta: dict, device: torch.device) -> torch.Tensor:
        """encode の逆変換。"""


class ByteCodec(ABC):
    """バイト列 → バイト列 の可逆圧縮の基本クラス。"""

    name: str = "base"

    @abstractmethod
    def compress(self, data: bytes) -> bytes: ...

    @abstractmethod
    def decompress(self, data: bytes) -> bytes: ...


class RawTensorCodec(TensorCodec):
    """量子化せず、指定した dtype のバイト列としてそのまま送る。

    Args:
        dtype: 送信時のデータ型。"float32" / "float16" / "bfloat16"。
               "native" なら元の dtype のまま（整数テンソルなどに使う）。
    """

    name = "raw"

    def __init__(self, dtype: str = "float32"):
        self.dtype = dtype

    def encode(self, t: torch.Tensor) -> tuple[bytes, dict]:
        orig_dtype = t.dtype
        if self.dtype != "native":
            t = t.to(getattr(torch, self.dtype))
        t = t.detach().contiguous().cpu()
        # どの dtype でも uint8 として見ればバイト列にできる
        data = t.reshape(-1).view(torch.uint8).numpy().tobytes()
        meta = {
            "shape": list(t.shape),
            "dtype": str(t.dtype).removeprefix("torch."),
            "orig_dtype": str(orig_dtype).removeprefix("torch."),
        }
        return data, meta

    def decode(self, data: bytes, meta: dict, device: torch.device) -> torch.Tensor:
        buf = torch.frombuffer(bytearray(data), dtype=torch.uint8)
        t = buf.view(getattr(torch, meta["dtype"])).reshape(meta["shape"])
        # 受信側では元の dtype（多くは float32）に戻して tail に渡す
        return t.to(device=device, dtype=getattr(torch, meta["orig_dtype"]))


class Codec:
    """head の出力（値のリスト）⇔ ペイロード（bytes）の変換をまとめて行うクラス。

    Args:
        tensor_codec: 浮動小数点テンソルに使う TensorCodec。
        byte_codecs: 後段で順に適用する ByteCodec のリスト（空なら圧縮なし）。
        name: 表示名。

    Note:
        整数や bool のテンソル（インデックスなど）は量子化すると壊れるため、
        tensor_codec に関わらず常に元の dtype のまま（raw）で送る。
    """

    def __init__(self, tensor_codec: TensorCodec, byte_codecs: Sequence[ByteCodec] = (), name: str = ""):
        self.tensor_codec = tensor_codec
        self.byte_codecs = list(byte_codecs)
        self.name = name or "+".join([tensor_codec.name, *(b.name for b in self.byte_codecs)])
        self._native = RawTensorCodec("native")

    def __repr__(self) -> str:
        return f"Codec({self.name})"

    def _codec_for(self, t: torch.Tensor) -> TensorCodec:
        return self.tensor_codec if t.is_floating_point() else self._native

    def encode(self, values: Sequence[Any]) -> bytes:
        """値のリスト（テンソルや int など）を 1 つのペイロードに変換する。"""
        entries, chunks = [], []
        for v in values:
            if isinstance(v, torch.Tensor):
                codec = self._codec_for(v)
                data, meta = codec.encode(v)
                entries.append({"kind": "tensor", "codec": codec.name, "nbytes": len(data), "meta": meta})
                chunks.append(data)
            else:
                # テンソル以外（int, tuple, None 等）は JSON でヘッダに埋め込む
                entries.append({"kind": "py", "value": _to_jsonable(v)})
        body = b"".join(chunks)
        for bc in self.byte_codecs:
            body = bc.compress(body)
        header = json.dumps({"entries": entries}, separators=(",", ":")).encode()
        return _HEADER_LEN.pack(len(header)) + header + body

    def decode(self, payload: bytes, device: torch.device | str = "cpu") -> list[Any]:
        """ペイロードを値のリストに戻す。テンソルは device 上に置かれる。"""
        device = torch.device(device)
        (hlen,) = _HEADER_LEN.unpack_from(payload, 0)
        header = json.loads(payload[_HEADER_LEN.size : _HEADER_LEN.size + hlen])
        body = payload[_HEADER_LEN.size + hlen :]
        for bc in reversed(self.byte_codecs):
            body = bc.decompress(body)
        values, offset = [], 0
        for e in header["entries"]:
            if e["kind"] == "py":
                values.append(_from_jsonable(e["value"]))
                continue
            codec = self.tensor_codec if e["codec"] == self.tensor_codec.name else self._native
            data = body[offset : offset + e["nbytes"]]
            offset += e["nbytes"]
            values.append(codec.decode(data, e["meta"], device))
        return values


def _to_jsonable(v: Any) -> Any:
    """tuple を区別して復元できるように印を付けて JSON 化可能な形にする。"""
    if isinstance(v, tuple):
        return {"__tuple__": [_to_jsonable(x) for x in v]}
    if isinstance(v, list):
        return [_to_jsonable(x) for x in v]
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    raise TypeError(f"テンソル以外の値 {type(v).__name__} は送信できません（int/float/str/tuple/list/None のみ対応）")


def _from_jsonable(v: Any) -> Any:
    if isinstance(v, dict) and "__tuple__" in v:
        return tuple(_from_jsonable(x) for x in v["__tuple__"])
    if isinstance(v, list):
        return [_from_jsonable(x) for x in v]
    return v

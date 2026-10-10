"""設定（CodecConfig）から Codec を組み立てる。新しい方式の登録もここで行う。

独自の量子化方式を追加する例::

    from splitcomp.compression import TensorCodec, register_tensor_codec

    @register_tensor_codec("my_quant")
    class MyQuant(TensorCodec):
        name = "my_quant"
        def __init__(self, cfg):          # CodecConfig がそのまま渡される
            self.bits = cfg.bits
        def encode(self, t): ...
        def decode(self, data, meta, device): ...

設定ファイルでは ``tensor: my_quant`` と書けば使える。
"""

from __future__ import annotations

from typing import Callable

from ..config import CodecConfig
from .base import ByteCodec, Codec, RawTensorCodec, TensorCodec
from .codecs import Lz4Codec, UniformQuantizer, ZlibCodec

# 名前 → 「CodecConfig を受け取ってインスタンスを返す関数」
_TENSOR: dict[str, Callable[[CodecConfig], TensorCodec]] = {
    "raw": lambda c: RawTensorCodec(c.dtype),
    "quant": lambda c: UniformQuantizer(c.bits, c.per_channel),
}
_BYTE: dict[str, Callable[[CodecConfig], ByteCodec]] = {
    "zlib": lambda c: ZlibCodec(c.level),
    "lz4": lambda c: Lz4Codec(),
}


def register_tensor_codec(name: str) -> Callable:
    """TensorCodec のクラス（または CodecConfig を受け取る関数）を登録するデコレータ。"""

    def deco(factory):
        _TENSOR[name] = factory
        return factory

    return deco


def register_byte_codec(name: str) -> Callable:
    """ByteCodec のクラス（または CodecConfig を受け取る関数）を登録するデコレータ。"""

    def deco(factory):
        _BYTE[name] = factory
        return factory

    return deco


def build_codec(cfg: CodecConfig | dict | None = None) -> Codec:
    """CodecConfig（または同じ内容の dict）から Codec を作る。None なら float32 そのまま。"""
    if cfg is None:
        cfg = CodecConfig()
    elif isinstance(cfg, dict):
        cfg = CodecConfig(**cfg)
    if cfg.tensor not in _TENSOR:
        raise KeyError(f"tensor codec {cfg.tensor!r} は未登録です（登録済み: {sorted(_TENSOR)}）")
    unknown = [b for b in cfg.compress if b not in _BYTE]
    if unknown:
        raise KeyError(f"compress {unknown} は未登録です（登録済み: {sorted(_BYTE)}）")
    return Codec(
        tensor_codec=_TENSOR[cfg.tensor](cfg),
        byte_codecs=[_BYTE[b](cfg) for b in cfg.compress],
        name=cfg.display_name(),
    )

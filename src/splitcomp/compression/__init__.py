"""中間特徴の圧縮まわり。

- :mod:`.base`       : Codec / TensorCodec / ByteCodec とペイロード形式
- :mod:`.codecs`     : 一様量子化・zlib・lz4
- :mod:`.registry`   : 設定から Codec を組み立てる build_codec と登録用デコレータ
- :mod:`.bottleneck` : 学習可能なエンコーダ / デコーダ
"""

from .base import ByteCodec, Codec, RawTensorCodec, TensorCodec
from .bottleneck import (
    HeadWithEncoder,
    TailWithDecoder,
    attach_bottleneck,
    build_bottleneck,
    infer_feature_shape,
)
from .codecs import Lz4Codec, UniformQuantizer, ZlibCodec
from .registry import build_codec, register_byte_codec, register_tensor_codec

__all__ = [
    "Codec",
    "TensorCodec",
    "ByteCodec",
    "RawTensorCodec",
    "UniformQuantizer",
    "ZlibCodec",
    "Lz4Codec",
    "build_codec",
    "register_tensor_codec",
    "register_byte_codec",
    "build_bottleneck",
    "attach_bottleneck",
    "infer_feature_shape",
    "HeadWithEncoder",
    "TailWithDecoder",
]

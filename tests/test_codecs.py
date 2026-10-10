"""圧縮方式のテスト: encode → decode で元に戻ること（量子化は誤差が理論値以内であること）。"""

import numpy as np
import pytest
import torch

from splitcomp.compression import build_codec
from splitcomp.compression.codecs import pack_bits, unpack_bits
from splitcomp.config import CodecConfig


@pytest.mark.parametrize("bits", [1, 2, 3, 4, 5, 7, 8, 12, 16])
def test_pack_unpack_bits(bits):
    rng = np.random.default_rng(0)
    q = rng.integers(0, 2**bits, size=1001)
    packed = pack_bits(q, bits)
    assert len(packed) == (1001 * bits + 7) // 8  # ビット単位で詰められている
    np.testing.assert_array_equal(unpack_bits(packed, bits, 1001), q)


@pytest.mark.parametrize("compress", [[], ["zlib"], ["lz4"]])
def test_raw_fp32_is_lossless(compress):
    codec = build_codec(CodecConfig(tensor="raw", compress=compress))
    values = [torch.randn(2, 8, 5, 5), torch.arange(6).reshape(2, 3), 7, (2, -1), None]
    out = codec.decode(codec.encode(values))
    torch.testing.assert_close(out[0], values[0], rtol=0, atol=0)
    torch.testing.assert_close(out[1], values[1])  # 整数テンソルは dtype もそのまま
    assert out[2:] == [7, (2, -1), None]  # テンソル以外の値も復元される


def test_fp16_halves_size():
    x = [torch.randn(1, 64, 32, 32)]
    n32 = len(build_codec(CodecConfig(dtype="float32")).encode(x))
    c16 = build_codec(CodecConfig(dtype="float16"))
    p16 = c16.encode(x)
    assert len(p16) < n32 * 0.51
    assert c16.decode(p16)[0].dtype == torch.float32  # 受信側では float32 に戻る


@pytest.mark.parametrize("bits", [2, 4, 8])
@pytest.mark.parametrize("per_channel", [False, True])
def test_quant_error_bound(bits, per_channel):
    x = torch.randn(2, 16, 8, 8) * torch.linspace(0.1, 5, 16).reshape(1, 16, 1, 1)
    codec = build_codec(CodecConfig(tensor="quant", bits=bits, per_channel=per_channel))
    payload = codec.encode([x])
    (y,) = codec.decode(payload)
    assert y.shape == x.shape
    # 一様量子化の誤差は最大で scale/2
    if per_channel:
        rng = x.amax(dim=(0, 2, 3), keepdim=True) - x.amin(dim=(0, 2, 3), keepdim=True)
    else:
        rng = x.max() - x.min()
    max_err = rng / (2**bits - 1) / 2
    assert ((y - x).abs() <= max_err + 1e-5).all()
    # データ量はおおよそ bits/32 になる
    assert len(payload) < x.numel() * 4 * bits / 32 * 1.3 + 600


def test_constant_tensor_quant():
    codec = build_codec(CodecConfig(tensor="quant", bits=4))
    x = torch.full((3, 4), 2.5)
    torch.testing.assert_close(codec.decode(codec.encode([x]))[0], x)


def test_unknown_codec_name():
    with pytest.raises(KeyError):
        build_codec(CodecConfig(tensor="nope"))
    with pytest.raises(KeyError):
        build_codec(CodecConfig(compress=["nope"]))

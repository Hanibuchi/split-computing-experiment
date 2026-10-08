"""具体的な圧縮方式: 一様量子化（UniformQuantizer）と、zlib / lz4 による可逆圧縮。"""

from __future__ import annotations

import zlib

import numpy as np
import torch

from .base import ByteCodec, TensorCodec


class UniformQuantizer(TensorCodec):
    """最小値〜最大値の範囲を 2^bits 段階に等分する一様量子化（非対称・min-max 方式）。

        q = round((x - min) / scale),   scale = (max - min) / (2^bits - 1)
        x ≈ q * scale + min                                      （復元）

    1〜16 ビットに対応し、8 ビット未満のときはビット単位で詰めて送る
    （例: 4 ビットなら 1 バイトに 2 値）。

    Args:
        bits: 量子化ビット数（1〜16）。
        per_channel: True ならチャネル（dim=1）ごとに min / scale を求める。
            チャネル間で値の範囲が大きく違う特徴で誤差が減るが、
            チャネル数 × 8 バイトのパラメータが追加で送られる。
    """

    name = "quant"

    def __init__(self, bits: int = 8, per_channel: bool = False):
        if not 1 <= bits <= 16:
            raise ValueError(f"bits は 1〜16 で指定してください: {bits}")
        self.bits = bits
        self.per_channel = per_channel

    def _reduce_dims(self, x: torch.Tensor) -> tuple[int, ...] | None:
        """per-channel のとき、min/max を取る次元（dim=1 以外）を返す。"""
        if self.per_channel and x.dim() >= 2:
            return tuple(d for d in range(x.dim()) if d != 1)
        return None

    def encode(self, t: torch.Tensor) -> tuple[bytes, dict]:
        x = t.detach().float()
        dims = self._reduce_dims(x)
        if dims is None:
            lo, hi = x.min(), x.max()
        else:
            lo, hi = x.amin(dim=dims, keepdim=True), x.amax(dim=dims, keepdim=True)
        levels = (1 << self.bits) - 1
        # 全要素が同じ値のとき scale=0 になって割り算が壊れるのを防ぐ
        scale = (hi - lo).clamp_min(1e-12) / levels
        q = torch.round((x - lo) / scale).clamp_(0, levels).to(torch.int32).cpu().numpy().reshape(-1)

        params = np.concatenate([lo.reshape(-1).cpu().numpy(), scale.reshape(-1).cpu().numpy()]).astype(np.float32)
        data = params.tobytes() + pack_bits(q, self.bits)
        meta = {
            "shape": list(x.shape),
            "bits": self.bits,
            "n_params": int(lo.numel()),
            "per_channel": dims is not None,
            "orig_dtype": str(t.dtype).removeprefix("torch."),
        }
        return data, meta

    def decode(self, data: bytes, meta: dict, device: torch.device) -> torch.Tensor:
        n_params, shape = meta["n_params"], meta["shape"]
        params = np.frombuffer(data, dtype=np.float32, count=2 * n_params)
        numel = int(np.prod(shape)) if shape else 1
        q = unpack_bits(data[params.nbytes :], meta["bits"], numel)
        x = torch.from_numpy(q.astype(np.float32)).reshape(shape)
        lo = torch.from_numpy(params[:n_params].copy())
        scale = torch.from_numpy(params[n_params:].copy())
        if meta["per_channel"]:
            # (C,) → (1, C, 1, 1, ...) に形を合わせて各チャネルに適用する
            view = [1] * len(shape)
            view[1] = n_params
            lo, scale = lo.reshape(view), scale.reshape(view)
        x = x * scale + lo
        return x.to(device=device, dtype=getattr(torch, meta["orig_dtype"]))


def pack_bits(q: np.ndarray, bits: int) -> bytes:
    """0〜2^bits-1 の整数配列を、各値 bits ビットで詰めたバイト列にする。"""
    if bits == 8:
        return q.astype(np.uint8).tobytes()
    if bits == 16:
        return q.astype("<u2").tobytes()
    # 各値を bits 個の 0/1 に展開して（上位ビットから）、8 個ずつ 1 バイトに詰める
    shifts = np.arange(bits - 1, -1, -1, dtype=np.uint32)
    bit_array = ((q.astype(np.uint32)[:, None] >> shifts) & 1).astype(np.uint8)
    return np.packbits(bit_array.reshape(-1)).tobytes()


def unpack_bits(data: bytes, bits: int, count: int) -> np.ndarray:
    """pack_bits の逆変換。"""
    if bits == 8:
        return np.frombuffer(data, dtype=np.uint8, count=count)
    if bits == 16:
        return np.frombuffer(data, dtype="<u2", count=count)
    bit_array = np.unpackbits(np.frombuffer(data, dtype=np.uint8), count=count * bits).reshape(count, bits)
    weights = (1 << np.arange(bits - 1, -1, -1)).astype(np.uint32)
    return bit_array.astype(np.uint32) @ weights


class ZlibCodec(ByteCodec):
    """zlib（DEFLATE）による可逆圧縮。圧縮率は高めだが lz4 より遅い。"""

    name = "zlib"

    def __init__(self, level: int | None = None):
        self.level = -1 if level is None else level

    def compress(self, data: bytes) -> bytes:
        return zlib.compress(data, self.level)

    def decompress(self, data: bytes) -> bytes:
        return zlib.decompress(data)


class Lz4Codec(ByteCodec):
    """lz4 による高速な可逆圧縮（``pip install lz4`` が必要）。"""

    name = "lz4"

    def __init__(self, **kwargs):
        try:
            import lz4.frame
        except ImportError as e:
            raise ImportError("lz4 圧縮を使うには `pip install lz4` が必要です") from e
        self._lz4 = lz4.frame

    def compress(self, data: bytes) -> bytes:
        return self._lz4.compress(data)

    def decompress(self, data: bytes) -> bytes:
        return self._lz4.decompress(data)

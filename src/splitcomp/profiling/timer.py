"""正確な実行時間の計測。

GPU（CUDA / MPS）の処理は非同期に実行されるため、普通に time.perf_counter() で
挟むだけだと「処理の発行にかかった時間」しか測れない。
ここでは計測の前後でデバイスを同期（synchronize）してから時間を取る。
"""

from __future__ import annotations

import time
from typing import Any, Callable

import torch


def synchronize(device: torch.device | str | None) -> None:
    """デバイス上の処理が終わるまで待つ（CPU なら何もしない）。"""
    if device is None:
        return
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def timed(fn: Callable[..., Any], *args, device: torch.device | str | None = None, **kwargs) -> tuple[Any, float]:
    """fn(*args, **kwargs) を実行し、(結果, 実行時間[ms]) を返す。

    Args:
        device: 計測の前後で同期するデバイス。GPU 上の処理を測るときは必ず指定する。
    """
    synchronize(device)
    start = time.perf_counter()
    out = fn(*args, **kwargs)
    synchronize(device)
    return out, (time.perf_counter() - start) * 1e3


class Timer:
    """with 文で使うタイマ。

    例::

        with Timer(device) as t:
            y = model(x)
        print(t.ms)
    """

    def __init__(self, device: torch.device | str | None = None):
        self.device = device
        self.ms = 0.0

    def __enter__(self) -> "Timer":
        synchronize(self.device)
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        synchronize(self.device)
        self.ms = (time.perf_counter() - self._start) * 1e3

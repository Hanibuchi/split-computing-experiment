"""モデル分割と分割推論。

- :class:`Splitter`      : モデルを任意の分割点で head / tail に分ける
- :class:`SplitPipeline` : head → 圧縮 → 通信 → 復元 → tail を計測しながら実行する
"""

from .pipeline import Backend, LocalBackend, SplitPipeline
from .splitter import INPUT_ALIAS, OUTPUT_ALIAS, SplitPoint, Splitter, split_sequential, trace

__all__ = [
    "Splitter",
    "SplitPoint",
    "split_sequential",
    "trace",
    "INPUT_ALIAS",
    "OUTPUT_ALIAS",
    "SplitPipeline",
    "Backend",
    "LocalBackend",
]

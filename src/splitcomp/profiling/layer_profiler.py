"""分割点ごとの「head / tail の計算時間」と「送信データ量」を一覧で調べる。

scripts/list_splits.py から使われる。どこで切るとデータ量が小さくなるか、
端末側の計算がどのくらい重くなるか、を実験前にざっと把握するためのもの。
"""

from __future__ import annotations

import statistics

import torch

from ..split.splitter import SplitPoint, Splitter
from ..utils import nbytes_fp32, tensor_leaves
from .timer import timed


@torch.no_grad()
def profile_split_points(
    splitter: Splitter,
    points: list[SplitPoint],
    sample: torch.Tensor,
    head_device: torch.device,
    tail_device: torch.device,
    repeat: int = 5,
    measure_time: bool = True,
) -> list[dict]:
    """各分割点について送信テンソルの形・サイズ・計算時間を調べる。

    Args:
        splitter: 分割対象の Splitter。
        points: 調べる分割点。
        sample: 入力サンプル（バッチ次元付き）。
        repeat: 時間計測の繰り返し回数（中央値を採用）。
        measure_time: False なら時間計測を省略して高速に一覧だけ出す。

    Returns:
        分割点ごとの dict のリスト。キー: index, name, num_tensors, shapes, bytes_fp32,
        ratio_to_input（入力に対するデータ量比）, head_ms, tail_ms
    """
    input_bytes = nbytes_fp32(sample)
    rows = []
    for p in points:
        head, tail = splitter.split(p)
        head.to(head_device).eval()
        tail.to(tail_device).eval()
        x = sample.to(head_device)
        z = head(x)
        row = {
            "index": p.index,
            "name": p.name,
            "num_tensors": p.num_tensors,
            "shapes": " ".join(str(tuple(t.shape)) for t in tensor_leaves(z)),
            "bytes_fp32": nbytes_fp32(z),
            "ratio_to_input": nbytes_fp32(z) / input_bytes,
        }
        if measure_time:
            zt = [v.to(tail_device) if isinstance(v, torch.Tensor) else v for v in z]
            tail(*zt)  # 1 回目は遅いことが多いので空回し
            row["head_ms"] = statistics.median(timed(head, x, device=head_device)[1] for _ in range(repeat))
            row["tail_ms"] = statistics.median(timed(tail, *zt, device=tail_device)[1] for _ in range(repeat))
        rows.append(row)
    return rows

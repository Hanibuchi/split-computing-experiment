"""デバイス選択・乱数シード・テンソルサイズ計算などの小さな共通関数。"""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch
import torch.utils._pytree as pytree


def resolve_device(name: str = "auto") -> torch.device:
    """デバイス名から torch.device を返す。"auto" なら cuda → mps → cpu の順で使えるものを選ぶ。"""
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def set_seed(seed: int) -> None:
    """Python / NumPy / PyTorch の乱数シードをまとめて設定する（再現性のため）。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def tensor_leaves(obj: Any) -> list[torch.Tensor]:
    """入れ子の tuple/list/dict の中にある全てのテンソルを平らなリストで返す。"""
    return [x for x in pytree.tree_leaves(obj) if isinstance(x, torch.Tensor)]


def nbytes_fp32(obj: Any) -> int:
    """obj に含まれるテンソルを全て float32 で送った場合のバイト数（圧縮率計算の基準）。"""
    return sum(t.numel() * 4 for t in tensor_leaves(obj))


def to_device(obj: Any, device: torch.device) -> Any:
    """入れ子構造の中のテンソルをすべて device に移す。"""
    return pytree.tree_map(lambda x: x.to(device) if isinstance(x, torch.Tensor) else x, obj)


def format_table(rows: list[dict], columns: list[str], floatfmt: str = ".4g") -> str:
    """dict のリストを、ターミナル表示用の簡単な表の文字列にする。"""

    def fmt(v: Any) -> str:
        if isinstance(v, float):
            # 大きな値は指数表記にせず整数で表示（例: 56620 バイト）
            return f"{v:.0f}" if abs(v) >= 1000 else format(v, floatfmt)
        return str(v)

    cells = [[fmt(r.get(c, "")) for c in columns] for r in rows]
    widths = [max(len(c), *(len(row[i]) for row in cells)) if cells else len(c) for i, c in enumerate(columns)]
    lines = ["  ".join(c.ljust(w) for c, w in zip(columns, widths))]
    lines.append("  ".join("-" * w for w in widths))
    lines += ["  ".join(v.ljust(w) for v, w in zip(row, widths)) for row in cells]
    return "\n".join(lines)

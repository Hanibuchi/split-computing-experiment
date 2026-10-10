"""タスク非依存の評価器。

2 種類の指標を集計する:

1. タスク指標（正解ラベルとの比較）
   - task="classification"（または "auto" で出力が (B, クラス数) のとき）: top-1 / top-5 精度
   - 独自指標は metric_fns に関数を渡して追加できる（docs/extending.md 参照）
2. 忠実度（分割前の元モデルの出力との比較） — タスクに依存せず使える
   - out_mse     : 出力の平均二乗誤差
   - out_cosine  : 出力ベクトルのコサイン類似度（サンプルごとの平均）
   - out_agree   : 出力が (B, クラス数) のとき、argmax が元モデルと一致した割合

圧縮で精度がどれだけ落ちたかを、正解ラベルがないデータでも「忠実度」で測れる。
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable

import torch
import torch.nn.functional as F

from ..utils import tensor_leaves

#: 独自指標の関数: (出力, 正解) → そのバッチでの「合計値」（平均ではない）
MetricFn = Callable[[Any, Any], float]


class Evaluator:
    """バッチごとに update() し、最後に compute() で平均を得る。

    Args:
        task: "auto" / "classification" / "none"。
        topk: classification のときに計算する top-k。
        metric_fns: 追加の指標 {名前: 関数}。関数はバッチ内の合計値を返すこと。
    """

    def __init__(self, task: str = "auto", topk: tuple[int, ...] = (1, 5), metric_fns: dict[str, MetricFn] | None = None):
        self.task = task
        self.topk = topk
        self.metric_fns = metric_fns or {}
        self.reset()

    def reset(self) -> None:
        self._sums: dict[str, float] = defaultdict(float)
        self._counts: dict[str, int] = defaultdict(int)

    def _add(self, name: str, total: float, n: int) -> None:
        self._sums[name] += float(total)
        self._counts[name] += n

    @torch.no_grad()
    def update(self, output: Any, target: Any = None, reference: Any = None) -> None:
        """1 バッチ分の結果を追加する。

        Args:
            output: 分割推論の出力。
            target: 正解（なければ None）。
            reference: 元モデルの出力（なければ None）。
        """
        if target is not None:
            if self._is_classification(output, target):
                logits = output.float().cpu()
                target = target.cpu()
                maxk = min(max(self.topk), logits.shape[1])
                pred = logits.topk(maxk, dim=1).indices
                correct = pred.eq(target[:, None])
                for k in self.topk:
                    if k <= logits.shape[1]:
                        self._add(f"top{k}", correct[:, :k].any(dim=1).sum(), len(target))
            for name, fn in self.metric_fns.items():
                self._add(name, fn(output, target), _batch_size(output))

        if reference is not None:
            out = torch.cat([t.float().reshape(t.shape[0], -1).cpu() for t in tensor_leaves(output)], dim=1)
            ref = torch.cat([t.float().reshape(t.shape[0], -1).cpu() for t in tensor_leaves(reference)], dim=1)
            n = out.shape[0]
            self._add("out_mse", ((out - ref) ** 2).mean(dim=1).sum(), n)
            self._add("out_cosine", F.cosine_similarity(out, ref, dim=1).sum(), n)
            if isinstance(output, torch.Tensor) and output.dim() == 2:
                self._add("out_agree", (output.argmax(1).cpu() == reference.argmax(1).cpu()).sum(), n)

    def _is_classification(self, output: Any, target: Any) -> bool:
        if self.task == "classification":
            return True
        if self.task != "auto":
            return False
        return (
            isinstance(output, torch.Tensor)
            and output.dim() == 2
            and isinstance(target, torch.Tensor)
            and target.dim() == 1
            and not target.is_floating_point()
        )

    def compute(self) -> dict[str, float]:
        """これまでの update の平均値を返す。"""
        return {k: self._sums[k] / self._counts[k] for k in self._sums if self._counts[k] > 0}


def _batch_size(output: Any) -> int:
    leaves = tensor_leaves(output)
    return leaves[0].shape[0] if leaves else 1

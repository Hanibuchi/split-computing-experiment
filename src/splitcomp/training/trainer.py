"""ボトルネック（エンコーダ / デコーダ）の学習。

元のモデルを「教師」として、分割点にボトルネックを挿入したモデル（生徒）を学習する:

    教師:  x → head → feat ──────────────────────────→ tail → y_teacher
    生徒:  x → head → feat → Encoder → z → Decoder → feat' → tail → y_student

損失 = distill_weight × 蒸留損失 + task_weight × タスク損失 + rate_weight × mean(|z|)

- 蒸留損失（distill）
    "feature": MSE(feat', feat)            … 特徴を復元できるように学習（タスク非依存・高速）
    "kl"     : KL(y_student ‖ y_teacher)   … 出力分布を合わせる（分類向け）
    "mse"    : MSE(y_student, y_teacher)   … 出力値を合わせる（回帰・検出のスコアなど）
- タスク損失（task_loss）: 正解ラベルとの損失（"cross_entropy" / "mse" / 独自関数）
- レート項（rate_weight）: z を小さく（疎に）して圧縮しやすくする簡易的な正則化

既定では head と tail の重みは固定し、エンコーダ / デコーダだけを学習する。
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..config import TrainConfig
from ..metrics.evaluator import Evaluator

#: タスク損失の関数: (生徒の出力, 正解) → スカラーの損失
TaskLossFn = Callable[[Any, Any], torch.Tensor]

_TASK_LOSSES: dict[str, TaskLossFn] = {
    "cross_entropy": lambda out, y: F.cross_entropy(out, y),
    "mse": lambda out, y: F.mse_loss(out, y),
}


class BottleneckTrainer:
    """ボトルネック付き分割モデルの学習を行うクラス。

    Args:
        cfg: 学習設定。
        teacher: 分割前の元モデル（重みは更新しない）。
        head, tail: 分割後のモデル（cfg.freeze_head / freeze_tail に従って固定 or 学習）。
        encoder, decoder: 学習するボトルネック。
        device: 学習に使うデバイス。
        task_loss_fn: 独自のタスク損失関数（指定すると cfg.task_loss より優先）。
    """

    def __init__(
        self,
        cfg: TrainConfig,
        teacher: nn.Module,
        head: nn.Module,
        tail: nn.Module,
        encoder: nn.Module,
        decoder: nn.Module,
        device: torch.device,
        task_loss_fn: TaskLossFn | None = None,
    ):
        self.cfg = cfg
        self.device = device
        self.teacher = teacher.to(device).eval().requires_grad_(False)
        # 特徴蒸留の目標は「学習前の head の出力」。head を学習する場合に備えて固定コピーを持つ
        self.teacher_head = copy.deepcopy(head).to(device).eval().requires_grad_(False)
        self.head = head.to(device).requires_grad_(not cfg.freeze_head)
        self.tail = tail.to(device).requires_grad_(not cfg.freeze_tail)
        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)

        if task_loss_fn is None and cfg.task_loss:
            if cfg.task_loss not in _TASK_LOSSES:
                raise KeyError(f"task_loss は {sorted(_TASK_LOSSES)} から選んでください: {cfg.task_loss!r}")
            task_loss_fn = _TASK_LOSSES[cfg.task_loss]
        self.task_loss_fn = task_loss_fn
        if cfg.distill not in ("feature", "kl", "mse", "none"):
            raise ValueError(f"distill は feature / kl / mse / none のいずれかです: {cfg.distill!r}")

        params = [p for m in (self.head, self.tail, self.encoder, self.decoder) for p in m.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    def _set_train_mode(self, train: bool) -> None:
        self.encoder.train(train)
        self.decoder.train(train)
        # 固定する部分は BatchNorm の統計量が変わらないよう常に eval モードにする
        self.head.train(train and not self.cfg.freeze_head)
        self.tail.train(train and not self.cfg.freeze_tail)

    def _forward(self, x: torch.Tensor) -> dict[str, Any]:
        """生徒モデルの順伝播。途中の値もまとめて返す。"""
        (feat,) = self.head(x)
        z = self.encoder(feat)
        rec = self.decoder(z)
        return {"feat": feat, "z": z, "rec": rec}

    def compute_loss(self, x: torch.Tensor, y: Any) -> tuple[torch.Tensor, dict[str, float]]:
        """1 バッチ分の損失を計算する。戻り値は (損失, 内訳 dict)。"""
        cfg = self.cfg
        s = self._forward(x)
        need_output = cfg.distill in ("kl", "mse") or (self.task_loss_fn is not None and cfg.task_weight > 0)
        out = self.tail(s["rec"]) if need_output else None

        parts: dict[str, torch.Tensor] = {}
        if cfg.distill == "feature":
            with torch.no_grad():
                (target_feat,) = self.teacher_head(x)
            parts["distill"] = F.mse_loss(s["rec"], target_feat)
        elif cfg.distill in ("kl", "mse"):
            with torch.no_grad():
                t_out = self.teacher(x)
            if cfg.distill == "kl":
                T = cfg.temperature
                parts["distill"] = F.kl_div(
                    F.log_softmax(out / T, dim=1), F.softmax(t_out / T, dim=1), reduction="batchmean"
                ) * (T * T)
            else:
                parts["distill"] = F.mse_loss(out, t_out)
        if self.task_loss_fn is not None and cfg.task_weight > 0:
            parts["task"] = self.task_loss_fn(out, y)
        if cfg.rate_weight > 0:
            parts["rate"] = s["z"].abs().mean()

        if not parts:
            raise ValueError("損失が 1 つもありません。distill か task_loss / task_weight を設定してください")
        weights = {"distill": cfg.distill_weight, "task": cfg.task_weight, "rate": cfg.rate_weight}
        loss = sum(weights[k] * v for k, v in parts.items())
        return loss, {k: float(v.detach()) for k, v in parts.items()}

    def train_epoch(self, loader: DataLoader, epoch: int) -> dict[str, float]:
        """1 エポック学習し、損失の平均を返す。"""
        self._set_train_mode(True)
        sums: dict[str, float] = {}
        n = 0
        pbar = tqdm(loader, desc=f"epoch {epoch}", leave=False)
        for step, (x, y) in enumerate(pbar):
            if self.cfg.max_steps is not None and step >= self.cfg.max_steps:
                break
            x, y = x.to(self.device), y.to(self.device)
            loss, parts = self.compute_loss(x, y)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            parts["loss"] = float(loss.detach())
            for k, v in parts.items():
                sums[k] = sums.get(k, 0.0) + v
            n += 1
            if step % self.cfg.log_interval == 0:
                pbar.set_postfix({k: f"{v:.4g}" for k, v in parts.items()})
        return {k: v / max(n, 1) for k, v in sums.items()}

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> dict[str, float]:
        """検証データで「生徒の出力」を評価する（正解との精度 + 元モデルとの忠実度）。"""
        self._set_train_mode(False)
        ev = Evaluator()
        for x, y in loader:
            x = x.to(self.device)
            out = self.tail(self._forward(x)["rec"])
            ev.update(out, y, reference=self.teacher(x))
        return ev.compute()

    def fit(self, train_loader: DataLoader, val_loader: DataLoader | None, run_dir: Path, on_epoch_end=None) -> None:
        """cfg.epochs 回学習する。各エポックの結果は run_dir/train_log.jsonl に追記される。

        Args:
            on_epoch_end: エポックごとに呼ばれる関数 ``f(epoch, log_dict)``（チェックポイント保存など）。
        """
        log_path = Path(run_dir) / "train_log.jsonl"
        for epoch in range(1, self.cfg.epochs + 1):
            log = {"epoch": epoch, **{f"train_{k}": v for k, v in self.train_epoch(train_loader, epoch).items()}}
            if val_loader is not None:
                log.update({f"val_{k}": v for k, v in self.evaluate(val_loader).items()})
            print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in log.items()), flush=True)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(log) + "\n")
            if on_epoch_end is not None:
                on_epoch_end(epoch, log)

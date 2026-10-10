"""設定から「分割済みモデル」を組み立てる高レベル関数（スクリプトから共通で使う）。

- :func:`build_split`           : モデルを作って分割し、必要ならボトルネックを読み込む
- :func:`save_bottleneck`       : 学習したボトルネックをチェックポイントとして保存
- :func:`config_fingerprint`    : サーバとクライアントの設定一致を確認するための識別子
"""

from __future__ import annotations

import copy
import hashlib
import json
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn

from .compression.bottleneck import attach_bottleneck, build_bottleneck
from .config import BottleneckConfig, CodecConfig, ModelConfig
from .models import build_model
from .split.splitter import SplitPoint, Splitter


@dataclass
class SplitBundle:
    """build_split の結果。

    Attributes:
        model: 分割前の元モデル（精度比較の基準に使う。head/tail とは重みを共有しない）。
        head: 端末側のモデル。
        tail: サーバ側のモデル。
        point: 分割点の情報。
        has_bottleneck: ボトルネックが挿入されているか。
    """

    model: nn.Module
    head: nn.Module
    tail: nn.Module
    point: SplitPoint
    has_bottleneck: bool = False


def build_split(model_cfg: ModelConfig, split_point: str, bottleneck_ckpt: str | None = None) -> SplitBundle:
    """モデルを構築して split_point で分割する。

    bottleneck_ckpt を指定した場合は、チェックポイントに保存された分割点・
    エンコーダ/デコーダ（と、学習で更新されていれば head/tail の重み）を読み込む。
    """
    ckpt = None
    if bottleneck_ckpt:
        ckpt = torch.load(bottleneck_ckpt, map_location="cpu", weights_only=True)
        if ckpt["model"]["name"] != model_cfg.name:
            warnings.warn(f"チェックポイントのモデル {ckpt['model']['name']!r} と設定のモデル {model_cfg.name!r} が異なります")
        if split_point and split_point != ckpt["split_point"]:
            warnings.warn(f"分割点はチェックポイントの {ckpt['split_point']!r} を使います（設定の {split_point!r} は無視）")
        split_point = ckpt["split_point"]
        model_cfg = ModelConfig(**ckpt["model"])

    model = build_model(model_cfg.name, model_cfg.pretrained, **model_cfg.kwargs)
    # head/tail は元モデルの複製から作る（元モデルを比較用に無傷で残すため）
    splitter = Splitter(copy.deepcopy(model), model_cfg.trace_depth)
    point = splitter.get_point(split_point)
    head, tail = splitter.split(point)
    if ckpt is None:
        return SplitBundle(model, head, tail, point)

    if "head" in ckpt:
        head.load_state_dict(ckpt["head"])
    if "tail" in ckpt:
        tail.load_state_dict(ckpt["tail"])
    encoder, decoder = build_bottleneck(ckpt["feature_shape"], BottleneckConfig(**ckpt["bottleneck"]))
    encoder.load_state_dict(ckpt["encoder"])
    decoder.load_state_dict(ckpt["decoder"])
    head, tail = attach_bottleneck(head, tail, encoder, decoder)
    return SplitBundle(model, head.eval(), tail.eval(), point, has_bottleneck=True)


def save_bottleneck(
    path: str | Path,
    model_cfg: ModelConfig,
    split_point: str,
    bottleneck_cfg: BottleneckConfig,
    feature_shape: list[int],
    encoder: nn.Module,
    decoder: nn.Module,
    head: nn.Module | None = None,
    tail: nn.Module | None = None,
    extra: dict | None = None,
) -> None:
    """ボトルネックのチェックポイントを保存する。head/tail を渡すとその重みも保存する（学習で更新した場合）。"""
    ckpt = {
        "model": asdict(model_cfg),
        "split_point": split_point,
        "bottleneck": asdict(bottleneck_cfg),
        "feature_shape": list(feature_shape),
        "encoder": encoder.state_dict(),
        "decoder": decoder.state_dict(),
        **(extra or {}),
    }
    if head is not None:
        ckpt["head"] = head.state_dict()
    if tail is not None:
        ckpt["tail"] = tail.state_dict()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, path)


def config_fingerprint(model_cfg: ModelConfig, split_point: str, codec_cfg: CodecConfig, bottleneck_ckpt: str | None) -> str:
    """サーバとクライアントで設定が一致しているか確認するための短い識別子を作る。"""
    info = {
        "model": asdict(model_cfg),
        "split_point": split_point,
        "codec": asdict(codec_cfg),
        "bottleneck": Path(bottleneck_ckpt).name if bottleneck_ckpt else None,
    }
    return hashlib.sha1(json.dumps(info, sort_keys=True).encode()).hexdigest()[:16]

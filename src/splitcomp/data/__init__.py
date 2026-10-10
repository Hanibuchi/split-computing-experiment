"""入力データセット。

- :mod:`.registry` : build_dataset / build_dataloader / register_dataset と組み込みデータセット
- :mod:`.ucf101`   : UCF-101 の動画クリップ（時系列実験用）
"""

from . import ucf101  # noqa: F401  （"ucf101" を登録するため）
from .registry import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    RandomDataset,
    available_datasets,
    build_dataloader,
    build_dataset,
    register_dataset,
)
from .ucf101 import UCF101Clips

__all__ = [
    "build_dataset",
    "build_dataloader",
    "register_dataset",
    "available_datasets",
    "RandomDataset",
    "UCF101Clips",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
]

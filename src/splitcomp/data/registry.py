"""データセットの構築（登録制）。

データセット名の例:
    - ``"random"``      : 乱数のダミーデータ（モデル・分割点の動作確認や遅延計測用）
    - ``"cifar10"``     : torchvision の CIFAR-10（初回は root にダウンロード）
    - ``"imagefolder"`` : ``root/<split>/<クラス名>/*.jpg`` 形式（ImageNet など）
    - ``"ucf101"``      : UCF-101 の動画クリップ（:mod:`.ucf101` 参照）

独自データセットの登録例::

    from splitcomp.data import register_dataset

    @register_dataset("my_data")
    def build_my_data(root: str, split: str, image_size: int = 224, **kwargs):
        return MyDataset(root=root, train=(split == "train"), image_size=image_size)
"""

from __future__ import annotations

from typing import Callable

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from ..config import DataConfig

_REGISTRY: dict[str, Callable[..., Dataset]] = {}

#: ImageNet の正規化パラメータ（torchvision の学習済みモデル用）
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def register_dataset(name: str) -> Callable:
    """データセット構築関数を登録するデコレータ。構築関数は ``(root, split, **kwargs)`` を受け取ること。"""

    def deco(fn: Callable[..., Dataset]) -> Callable[..., Dataset]:
        if name in _REGISTRY:
            raise KeyError(f"データセット名 {name!r} はすでに登録されています")
        _REGISTRY[name] = fn
        return fn

    return deco


def available_datasets() -> list[str]:
    """登録済みのデータセット名の一覧。"""
    return sorted(_REGISTRY)


def build_dataset(cfg: DataConfig) -> Dataset:
    """設定からデータセットを作る。num_samples を指定した場合は先頭 N 件に絞る。"""
    if cfg.name not in _REGISTRY:
        raise KeyError(f"未知のデータセットです: {cfg.name!r}（使えるもの: {available_datasets()}）")
    ds = _REGISTRY[cfg.name](cfg.root, cfg.split, **cfg.kwargs)
    if cfg.num_samples is not None and cfg.num_samples < len(ds):
        ds = Subset(ds, range(cfg.num_samples))
    return ds


def build_dataloader(cfg: DataConfig) -> DataLoader:
    """設定から DataLoader を作る。各バッチは ``(入力, 正解)``。"""
    return DataLoader(
        build_dataset(cfg),
        batch_size=cfg.batch_size,
        shuffle=cfg.shuffle,
        num_workers=cfg.num_workers,
    )


# ---------------------------------------------------------------------------
# 組み込みのデータセット
# ---------------------------------------------------------------------------


class RandomDataset(Dataset):
    """乱数の入力とラベルを返すダミーデータセット。同じ index には常に同じ値を返す。"""

    def __init__(self, length: int = 64, input_shape=(3, 224, 224), num_classes: int = 1000, seed: int = 0):
        self.length = length
        self.input_shape = tuple(input_shape)
        self.num_classes = num_classes
        self.seed = seed

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, i: int):
        g = torch.Generator().manual_seed(self.seed * 1_000_003 + i)
        x = torch.randn(self.input_shape, generator=g)
        y = int(torch.randint(self.num_classes, (1,), generator=g))
        return x, y


@register_dataset("random")
def _build_random(root: str, split: str, length: int = 1024, input_shape=(3, 224, 224), num_classes: int = 1000, seed: int = 0):
    return RandomDataset(length, input_shape, num_classes, seed)


def _image_transform(image_size: int, resize: int | None):
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.Resize(resize or int(image_size * 256 / 224)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


@register_dataset("cifar10")
def _build_cifar10(root: str, split: str, image_size: int = 32, download: bool = True, normalize: bool = True):
    from torchvision import datasets, transforms

    tf = [transforms.Resize(image_size)] if image_size != 32 else []
    tf.append(transforms.ToTensor())
    if normalize:
        tf.append(transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)))
    return datasets.CIFAR10(root, train=(split == "train"), download=download, transform=transforms.Compose(tf))


@register_dataset("imagefolder")
def _build_imagefolder(root: str, split: str, image_size: int = 224, resize: int | None = None):
    from pathlib import Path

    from torchvision import datasets

    path = Path(root) / split if (Path(root) / split).is_dir() else Path(root)
    return datasets.ImageFolder(str(path), transform=_image_transform(image_size, resize))

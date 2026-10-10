"""モデルの構築（torchvision / timm / 独自モデル）。

モデル名の書き方:
    - ``"resnet18"`` や ``"torchvision:resnet18"`` → torchvision のモデル
    - ``"timm:convnext_tiny"`` → timm のモデル（``pip install timm`` が必要）
    - ``@register_model("名前")`` で登録した独自モデル

独自モデルの登録例::

    from splitcomp.models import register_model

    @register_model("my_net")
    def build_my_net(pretrained: bool = False, num_classes: int = 10):
        return MyNet(num_classes)
"""

from __future__ import annotations

from typing import Callable

import torch.nn as nn

_REGISTRY: dict[str, Callable[..., nn.Module]] = {}


def register_model(name: str) -> Callable:
    """独自モデルの構築関数を登録するデコレータ。構築関数は ``pretrained`` 引数を受け取ること。"""

    def deco(fn: Callable[..., nn.Module]) -> Callable[..., nn.Module]:
        if name in _REGISTRY:
            raise KeyError(f"モデル名 {name!r} はすでに登録されています")
        _REGISTRY[name] = fn
        return fn

    return deco


def available_models() -> list[str]:
    """登録済みの独自モデル名の一覧（torchvision / timm のモデルは含まない）。"""
    return sorted(_REGISTRY)


def build_model(name: str, pretrained: bool = False, **kwargs) -> nn.Module:
    """名前からモデルを作り、評価モード（eval）にして返す。

    Args:
        name: モデル名（モジュール docstring 参照）。
        pretrained: 学習済み重みを使うか。
        **kwargs: 構築関数への追加引数（num_classes など）。
    """
    if name in _REGISTRY:
        model = _REGISTRY[name](pretrained=pretrained, **kwargs)
    elif name.startswith("timm:"):
        try:
            import timm
        except ImportError as e:
            raise ImportError("timm のモデルを使うには `pip install timm` が必要です") from e
        model = timm.create_model(name[len("timm:"):], pretrained=pretrained, **kwargs)
    else:
        import torchvision

        tv_name = name[len("torchvision:"):] if name.startswith("torchvision:") else name
        if tv_name not in torchvision.models.list_models():
            raise KeyError(
                f"モデル {name!r} が見つかりません。torchvision のモデル名、'timm:<名前>'、"
                f"または登録済みモデル {available_models()} を指定してください"
            )
        weights = "DEFAULT" if pretrained else None
        model = torchvision.models.get_model(tv_name, weights=weights, **kwargs)
    return model.eval()


# ---------------------------------------------------------------------------
# 動作確認・テスト用の小さなモデル
# ---------------------------------------------------------------------------


class ToyCNN(nn.Module):
    """テスト用の小さな CNN（skip connection を含むので分割処理の確認に使える）。"""

    def __init__(self, num_classes: int = 10, in_channels: int = 3):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(in_channels, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU())
        self.block = nn.Sequential(nn.Conv2d(16, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU())
        self.down = nn.Sequential(nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU())
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = x + self.block(x)  # 残差接続
        x = self.down(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


class ToyMLP(nn.Module):
    """テスト用の小さな MLP（画像以外の入力も扱えることの確認用）。"""

    def __init__(self, in_features: int = 32, hidden: int = 64, num_classes: int = 10):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden, hidden)
        self.act2 = nn.ReLU()
        self.head = nn.Linear(hidden, num_classes)

    def forward(self, x):
        return self.head(self.act2(self.fc2(self.act1(self.fc1(x)))))


@register_model("toy_cnn")
def _build_toy_cnn(pretrained: bool = False, **kwargs) -> nn.Module:
    return ToyCNN(**kwargs)


@register_model("toy_mlp")
def _build_toy_mlp(pretrained: bool = False, **kwargs) -> nn.Module:
    return ToyMLP(**kwargs)

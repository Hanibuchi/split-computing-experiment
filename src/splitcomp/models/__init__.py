"""モデル構築まわり。詳しくは :mod:`splitcomp.models.registry` を参照。"""

from .registry import ToyCNN, ToyMLP, available_models, build_model, register_model

__all__ = ["build_model", "register_model", "available_models", "ToyCNN", "ToyMLP"]

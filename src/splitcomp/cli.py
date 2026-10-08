"""スクリプト共通のコマンドライン処理。

全てのスクリプトは次の形式で引数を受け取る::

    python scripts/xxx.py --config 設定.yaml [key=value ...]
"""

from __future__ import annotations

import argparse
from typing import TypeVar

from .config import load_config

T = TypeVar("T")


def parse_cli(cls: type[T], description: str, default_config: str | None = None, extra_args=None) -> tuple[T, argparse.Namespace]:
    """--config と key=value 形式の上書きを解釈して、設定オブジェクトを返す。

    Args:
        cls: 設定クラス。
        description: --help に表示する説明文。
        default_config: --config 省略時に読む YAML（None ならクラスの既定値）。
        extra_args: argparse.ArgumentParser を受け取り、スクリプト固有の引数を追加する関数。

    Returns:
        (設定オブジェクト, argparse の結果)
    """
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="設定の上書き例: model.name=resnet50 data.batch_size=8 codecs.0.bits=4",
    )
    parser.add_argument("--config", default=default_config, help="設定ファイル（YAML）のパス")
    parser.add_argument("overrides", nargs="*", help="key=value 形式の設定上書き（複数可）")
    if extra_args is not None:
        extra_args(parser)
    args = parser.parse_args()
    return load_config(cls, args.config, args.overrides), args

"""splitcomp: Split computing 研究のための汎用実験基盤。

任意の PyTorch モデルを任意の層で
「head（エッジ端末側）」と「tail（サーバ側）」に分割し、

    入力 → head → [圧縮(encode)] → 通信 → [伸長(decode)] → tail → 出力

という流れの推論・計測・学習を共通の API で行えるようにする。

よく使う入口:
    - :func:`splitcomp.models.build_model`           モデルの構築
    - :class:`splitcomp.split.Splitter`               モデルの分割
    - :func:`splitcomp.compression.build_codec`       中間特徴の圧縮方式の構築
    - :class:`splitcomp.split.SplitPipeline`          分割推論（時間・データ量の計測つき）
    - :class:`splitcomp.transport.SimulatedChannel`   帯域・遅延を模擬した通信路
"""

__version__ = "0.1.0"

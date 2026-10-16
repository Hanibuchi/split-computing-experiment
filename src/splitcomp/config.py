"""設定（YAML）の読み込みと、コマンドラインからの上書き。

設定は dataclass で定義しており、各フィールドの意味はここにまとめて書いてある。
YAML ファイルを読み込んだあと、コマンドラインの ``key=value`` 形式の引数で
一部だけ上書きできる。

例::

    python scripts/sweep.py --config configs/sweep_resnet50.yaml \\
        model.name=resnet18 data.num_samples=16 codecs.0.bits=4

- ``a.b=値`` でネストした項目を上書きする。
- ``codecs.0.bits=4`` のように数字でリストの要素を指定できる。
- 値は YAML として解釈される（``true`` → bool、``[1,2]`` → list、``null`` → None）。
"""

from __future__ import annotations

import dataclasses
import types
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# 共通の部品
# ---------------------------------------------------------------------------


@dataclass
class ModelConfig:
    """分割対象のモデルの設定。"""

    #: モデル名。"resnet18"（torchvision）, "timm:resnet50"（timm）,
    #: register_model で登録した独自モデル名などを指定する。
    name: str = "resnet18"
    #: 学習済み重みを読み込むかどうか（torchvision / timm のみ）。
    pretrained: bool = False
    #: モデル構築関数に渡す追加引数（例: {num_classes: 10}）。
    kwargs: dict = field(default_factory=dict)
    #: torch.fx でグラフ化するときの「展開の深さ」。
    #: None なら全ての層を細かく展開する（分割点が最も多くなる）。
    #: 1 にすると ResNet なら layer1, layer2 ... といった粗い単位で分割点が並ぶ。
    trace_depth: int | None = None


@dataclass
class DataConfig:
    """入力データセットの設定。"""

    #: データセット名。"random"（ダミーデータ）, "cifar10", "imagefolder" など。
    name: str = "random"
    #: データの置き場所（random 以外で使用）。
    root: str = "./data"
    #: "train" / "val" などの分割。
    split: str = "val"
    #: 使うサンプル数。None なら全件。実験を素早く回すために先頭 N 件に絞る。
    num_samples: int | None = 64
    #: バッチサイズ。レイテンシ計測では 1（=1 枚ずつ送る）が現実的。
    batch_size: int = 1
    #: DataLoader のワーカ数。
    num_workers: int = 0
    #: シャッフルするかどうか（学習時は true 推奨）。
    shuffle: bool = False
    #: データセット構築関数に渡す追加引数
    #: （例: random なら {input_shape: [3, 224, 224], num_classes: 1000}）。
    kwargs: dict = field(default_factory=dict)


@dataclass
class CodecConfig:
    """中間特徴（head の出力）を送信用バイト列に変換する方式の設定。

    処理は 2 段階:
      1. テンソル → バイト列（tensor）: そのまま送る(raw) か量子化(quant)するか
      2. バイト列の可逆圧縮（compress）: zlib / lz4 を順に適用（空なら圧縮なし）
    """

    #: 結果 CSV の codec 列に出す表示名。空なら設定内容から自動生成する。
    name: str = ""
    #: "raw"（量子化しない）または "quant"（一様量子化）。
    tensor: str = "raw"
    #: raw のときの送信データ型。"float32" / "float16" / "bfloat16"。
    dtype: str = "float32"
    #: quant のときのビット数（1〜16）。
    bits: int = 8
    #: quant のとき、チャネル（dim=1）ごとに量子化範囲を決めるか。
    per_channel: bool = False
    #: 後段の可逆圧縮。["zlib"], ["lz4"], [] など。
    compress: list = field(default_factory=list)
    #: zlib の圧縮レベル（0〜9）。None で既定値。
    level: int | None = None

    def display_name(self) -> str:
        """結果表示用の短い名前を返す（例: "quant8+zlib"）。"""
        if self.name:
            return self.name
        if self.tensor == "quant":
            base = f"quant{self.bits}" + ("pc" if self.per_channel else "")
        else:
            base = {"float32": "fp32", "float16": "fp16", "bfloat16": "bf16"}.get(self.dtype, self.dtype)
        return "+".join([base, *self.compress])


@dataclass
class ChannelConfig:
    """シミュレーションする通信路の設定。"""

    #: 結果 CSV の channel 列に出す表示名。空なら "10Mbps/20ms" のように自動生成。
    name: str = ""
    #: 上り（端末→サーバ）の帯域 [Mbit/s]。
    bandwidth_mbps: float = 10.0
    #: 往復遅延時間 RTT [ms]。片道あたり RTT/2 として計算する。
    rtt_ms: float = 20.0
    #: 下り（サーバ→端末）の帯域 [Mbit/s]。None なら上りと同じ。
    downlink_bandwidth_mbps: float | None = None

    def display_name(self) -> str:
        return self.name or f"{self.bandwidth_mbps:g}Mbps/{self.rtt_ms:g}ms"


@dataclass
class DeviceConfig:
    """head / tail を動かすデバイス。"auto" なら cuda → mps → cpu の順に選ぶ。

    端末側の非力さを模擬したい場合は head: cpu, tail: auto のように分けるとよい。
    """

    head: str = "auto"
    tail: str = "auto"


@dataclass
class BottleneckConfig:
    """分割点に挿入する学習可能なボトルネック（エンコーダ/デコーダ）の設定。"""

    #: ボトルネック後のチャネル数（畳み込み特徴の場合）または次元数（ベクトル特徴の場合）。
    channels: int = 16
    #: 空間方向の縮小率（1, 2, 4, 8 ...）。畳み込み特徴の場合のみ有効。
    downsample: int = 1
    #: 中間層のチャネル数。None なら入力チャネル数と同じ。
    hidden_channels: int | None = None


@dataclass
class LossConfig:
    """パケットロスのモデルの設定（splitcomp.temporal.packet 参照）。"""

    #: "bernoulli"（独立）/ "gilbert_elliott"（バースト）/ "frame"（フレーム単位の全損失）。
    name: str = "bernoulli"
    #: 平均損失率（frame ではフレームが落ちる確率）。
    p: float = 0.0
    #: gilbert_elliott の平均バースト長 [パケット]。
    burst_len: float = 4.0
    #: 結果 CSV の loss 列に出す表示名。空なら "bernoulli(p=0.05)" のように自動生成。
    label: str = ""

    def display_name(self) -> str:
        if self.label:
            return self.label
        if self.name == "gilbert_elliott":
            return f"{self.name}(p={self.p:g},L={self.burst_len:g})"
        return f"{self.name}(p={self.p:g})"


@dataclass
class TemporalSchemeConfig:
    """時系列符号化の方式（splitcomp.temporal.coding 参照）。"""

    #: 結果 CSV の scheme 列に出す表示名（例: "D+DZ"）。
    name: str = "D+DZ"
    #: "fd"（毎フレーム独立）/ "dwd"（真の差分・ドリフトあり）/ "dpcm"（予測残差）。
    scheme: str = "dpcm"
    #: P フレームの量子化方式。"sym"（対称）/ "dz"（dead-zone）。I フレームは常に対称。
    p_quant: str = "dz"
    #: ロス補修 net の学習済み重み（scripts/train_restorer.py の出力）。指定すると P フレームでロスが起きたときに
    #: 欠けた要素を補ってから tail に渡す（例: D+DZ+R）。None なら使わない。
    restorer: str | None = None


@dataclass
class SplitTarget:
    """評価するモデルと分割点の組。"""

    model: ModelConfig = field(default_factory=ModelConfig)
    split_point: str = "layer2"


# ---------------------------------------------------------------------------
# スクリプトごとの設定
# ---------------------------------------------------------------------------


@dataclass
class SweepConfig:
    """scripts/sweep.py 用: 分割点 × 圧縮方式 × 通信路 を総当たりで評価する。"""

    #: 実験名。出力ディレクトリ名に使われる。
    experiment_name: str = "sweep"
    #: 結果の出力先ディレクトリ。runs/<実験名>_<日時>/ が作られる。
    output_dir: str = "runs"
    #: 乱数シード。
    seed: int = 0
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    devices: DeviceConfig = field(default_factory=DeviceConfig)
    #: 評価する分割点。
    #:   "clean": 転送テンソルが 1 つだけになる分割点すべて（既定）
    #:   "all"  : 全ての分割点（skip connection の途中なども含む）
    #:   リスト : ["layer1", "layer2"] のように分割点名を列挙
    #: "input"（全部サーバ）/ "output"（全部端末）も指定可能。
    split_points: Any = "clean"
    #: 比較する圧縮方式のリスト。
    codecs: list[CodecConfig] = field(default_factory=lambda: [CodecConfig()])
    #: 比較する通信路のリスト。
    channels: list[ChannelConfig] = field(default_factory=lambda: [ChannelConfig()])
    #: 計測前の空回し回数（初回の遅い実行を計測から除くため）。
    warmup: int = 3
    #: 学習済みボトルネックのチェックポイント。指定すると分割点はチェックポイントのものに固定される。
    bottleneck_ckpt: str | None = None


@dataclass
class TrainConfig:
    """scripts/train_bottleneck.py 用: 分割点にボトルネックを挿入して学習する。"""

    experiment_name: str = "train_bottleneck"
    output_dir: str = "runs"
    seed: int = 0
    model: ModelConfig = field(default_factory=ModelConfig)
    #: 学習データ。
    data: DataConfig = field(default_factory=lambda: DataConfig(split="train", batch_size=32, shuffle=True))
    #: 検証データ。None なら検証しない。
    val_data: DataConfig | None = None
    #: 学習に使うデバイス。
    device: str = "auto"
    #: ボトルネックを挿入する分割点名（scripts/list_splits.py で確認できる）。
    split_point: str = "layer2"
    bottleneck: BottleneckConfig = field(default_factory=BottleneckConfig)
    #: head（分割点より前）の重みを固定するか。
    freeze_head: bool = True
    #: tail（分割点より後）の重みを固定するか。
    freeze_tail: bool = True
    epochs: int = 1
    #: 1 エポックあたりの最大ステップ数（None で全データ）。動作確認用。
    max_steps: int | None = None
    lr: float = 1e-3
    weight_decay: float = 0.0
    #: 蒸留損失の種類。
    #:   "feature": 復元した特徴と元の head 出力の MSE（タスク非依存・既定）
    #:   "kl"     : 元モデルの出力（logits）との KL ダイバージェンス（分類向け）
    #:   "mse"    : 元モデルの出力との MSE（回帰など）
    #:   "none"   : 蒸留しない
    distill: str = "feature"
    distill_weight: float = 1.0
    #: kl のときの温度。
    temperature: float = 1.0
    #: 正解ラベルを使うタスク損失。None / "cross_entropy" / "mse"。
    task_loss: str | None = None
    task_weight: float = 0.0
    #: ボトルネック出力の L1 ノルムに掛ける重み（データ量を減らす方向の正則化の簡易版）。
    rate_weight: float = 0.0
    #: 何ステップごとに損失を表示するか。
    log_interval: int = 10


@dataclass
class NetworkConfig:
    """scripts/serve.py と scripts/client.py 用: 実ネットワークで分割推論する。

    サーバとクライアントは同じ設定ファイルを使う（分割点・圧縮方式が一致している必要がある）。
    """

    experiment_name: str = "network"
    output_dir: str = "runs"
    seed: int = 0
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    devices: DeviceConfig = field(default_factory=DeviceConfig)
    split_point: str = "layer2"
    codec: CodecConfig = field(default_factory=CodecConfig)
    #: サーバのアドレス。サーバ側では待ち受けアドレス（0.0.0.0 で全インタフェース）。
    host: str = "127.0.0.1"
    port: int = 50051
    warmup: int = 3
    bottleneck_ckpt: str | None = None


@dataclass
class PacketLossConfig:
    """scripts/packet_loss_sweep.py 用: 動画の時系列符号化（FD / DwD / DPCM）をパケットロス下で評価する。"""

    experiment_name: str = "packet_loss"
    output_dir: str = "runs"
    seed: int = 0
    #: 評価する (モデル, 分割点) のリスト。分割点は送信テンソルが 1 つの点（clean）であること。
    targets: list[SplitTarget] = field(default_factory=lambda: [SplitTarget()])
    #: 動画データ。各要素が (frames[T, 3, H, W], label) のデータセット（例: ucf101）。
    data: DataConfig = field(default_factory=lambda: DataConfig(name="ucf101", root="./data", num_samples=None))
    devices: DeviceConfig = field(default_factory=DeviceConfig)
    #: 比較する時系列符号化の方式。
    schemes: list[TemporalSchemeConfig] = field(default_factory=lambda: [TemporalSchemeConfig()])
    #: 量子化ビット数のリスト。
    bits: list[int] = field(default_factory=lambda: [4])
    #: GoP 長 K（I フレームの間隔）。
    gop: int = 30
    #: 1 パケットの最大サイズ [バイト]（アプリケーション層のヘッダ込み）。
    mtu: int = 1400
    #: true で要素をパケットに交互に振り分ける（ロスを空間的に分散させる）。
    interleave: bool = False
    #: 届かなかった要素の補い方。"hold"（直前の値）/ "zero"。
    concealment: str = "hold"
    #: 比較する損失モデルのリスト。
    losses: list[LossConfig] = field(default_factory=lambda: [LossConfig()])
    #: 損失パターンの乱数シード（損失率 0 の条件は 1 回だけ実行する）。
    seeds: list[int] = field(default_factory=lambda: [0])


@dataclass
class RestorerTrainConfig:
    """scripts/train_restorer.py 用: モデル共通のロス補修 net（splitcomp.temporal.restore）を学習する。

    head（固定）の特徴を DPCM + dead-zone 量子化で符号化し、ランダムなパケットロスを起こして、
    欠けた要素を「ロスがなかったときの復元特徴」に近づけるよう学習する。tail やラベルは使わない。
    """

    experiment_name: str = "train_restorer"
    output_dir: str = "runs"
    seed: int = 0
    #: 学習に使う (モデル, 分割点)。複数書くと混ぜて学習する。分割点は送信テンソルが 1 つで (C, H, W) の点であること。
    targets: list[SplitTarget] = field(default_factory=lambda: [SplitTarget()])
    #: 学習用の動画（評価用のクリップと重ならないようにする。例: ucf101 の groups: "8-25"）。
    data: DataConfig = field(default_factory=lambda: DataConfig(name="ucf101", root="./data", num_samples=None))
    #: data の末尾から何本を検証用にするか（学習には使わない）。
    val_clips: int = 10
    devices: DeviceConfig = field(default_factory=DeviceConfig)
    #: 符号化の条件（評価と合わせる）。bits は系列ごとにこの中からランダムに選ぶ。
    bits: list[int] = field(default_factory=lambda: [2, 4, 8])
    gop: int = 30
    mtu: int = 1400
    interleave: bool = False
    #: 学習時に起こす損失。系列ごとにこの中からランダムに選ぶ。
    losses: list[LossConfig] = field(default_factory=lambda: [LossConfig("bernoulli", 0.05)])
    #: 補修 net の構成（splitcomp.temporal.restore.LossRestorer の引数）。
    hidden: int = 32
    dilations: list[int] = field(default_factory=lambda: [1, 2, 4, 2, 1])
    epochs: int = 10
    #: 1 エポックで 1 クリップあたり何系列（bits・損失をランダムに選んで符号化 → ロス）を作るか。
    sequences_per_clip: int = 2
    #: 1 回の更新に使うフレーム数（ロスのあった P フレーム）。
    batch_size: int = 4
    #: 1 フレームあたり学習に使うチャネル数の上限（メモリと時間の節約。null で全て）。
    max_channels: int | None = 128
    lr: float = 1e-3
    #: 学習した重みを run フォルダ（restorer.pt）に加えて、ここにも保存する（packet_loss_sweep の restorer に指定する用）。
    save_to: str | None = None


# ---------------------------------------------------------------------------
# 読み込み処理
# ---------------------------------------------------------------------------

T = typing.TypeVar("T")


def load_config(cls: type[T], path: str | Path | None = None, overrides: list[str] | None = None) -> T:
    """YAML ファイルとコマンドライン上書きから設定オブジェクトを作る。

    Args:
        cls: 作りたい設定クラス（SweepConfig など）。
        path: YAML ファイルのパス。None ならクラスの既定値のみ。
        overrides: ``["model.name=resnet50", "data.batch_size=8"]`` のような上書き指定。

    Returns:
        cls のインスタンス。
    """
    raw: dict = {}
    if path is not None:
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"上書き指定は key=value の形式で書いてください: {item!r}")
        key, value = item.split("=", 1)
        _set_by_path(raw, key.split("."), yaml.safe_load(value))
    return from_dict(cls, raw)


def to_dict(cfg: Any) -> dict:
    """設定オブジェクトを（YAML/JSON に保存できる）dict に戻す。"""
    return dataclasses.asdict(cfg)


def from_dict(cls: type[T], data: dict) -> T:
    """dict から dataclass を再帰的に構築する。未知のキーがあればエラーにする（タイプミス検出のため）。"""
    if not isinstance(data, dict):
        raise TypeError(f"{cls.__name__} には dict が必要ですが {type(data).__name__} が渡されました")
    hints = typing.get_type_hints(cls)
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - names
    if unknown:
        raise KeyError(f"{cls.__name__} に存在しない設定項目です: {sorted(unknown)}（使える項目: {sorted(names)}）")
    kwargs = {k: _convert(hints[k], v) for k, v in data.items()}
    return cls(**kwargs)


def _convert(tp: Any, value: Any) -> Any:
    """型ヒント tp に従って value を変換する（dataclass / list[dataclass] / Optional に対応）。"""
    if value is None:
        return None
    origin = typing.get_origin(tp)
    if origin in (typing.Union, types.UnionType):
        # Optional[X] など: None 以外の候補のうち dataclass があればそれで変換を試みる
        for arg in typing.get_args(tp):
            if dataclasses.is_dataclass(arg) and isinstance(value, dict):
                return from_dict(arg, value)
        return value
    if dataclasses.is_dataclass(tp):
        return from_dict(tp, value)
    if origin is list:
        (arg,) = typing.get_args(tp) or (Any,)
        return [_convert(arg, v) for v in value]
    return value


def _set_by_path(d: Any, keys: list[str], value: Any) -> None:
    """``["codecs", "0", "bits"]`` のようなパスで入れ子の dict/list に値を設定する。"""
    key = keys[0]
    last = len(keys) == 1
    if isinstance(d, list):
        idx = int(key)
        while len(d) <= idx:  # 足りない要素は空 dict で埋める
            d.append({})
        if last:
            d[idx] = value
        else:
            _set_by_path(d[idx], keys[1:], value)
        return
    if last:
        d[key] = value
        return
    if key not in d or d[key] is None:
        d[key] = [] if keys[1].isdigit() else {}
    _set_by_path(d[key], keys[1:], value)

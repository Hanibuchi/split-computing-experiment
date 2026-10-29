# 全体構成と処理の流れ

## 処理の流れ

```
                          ┌────────────────── SplitPipeline.run(x) ──────────────────┐
                          │                                                         │
 x ─▶ head ─▶ (z1, z2, …) ─▶ Codec.encode ─▶ payload(bytes) ─▶ Backend.infer(payload) ─▶ 出力, 計測値
       │                        │                                │
   Splitter.split()     TensorCodec(量子化)              LocalBackend: decode → tail（同一プロセス）
   で作られる            + ByteCodec(zlib等)              　　+ SimulatedChannel で通信時間を計算
                                                         TCPBackend : TailServer に送信して実行
```

1. **分割**: `Splitter` がモデルを `torch.fx` でグラフ化し、指定の分割点で `head` と `tail` を作る。
2. **head 実行**: 端末側で `head(x)` を実行。出力は送信するテンソルの tuple。
3. **encode**: `Codec` がテンソル群を 1 つのバイト列（ペイロード）に変換する。
4. **通信**: シミュレーションなら時間を計算、実通信なら TCP で送信。
5. **decode → tail**: サーバ側でペイロードを復元し、`tail(*values)` を実行。
6. 各段階の時間とデータ量を `stats` として返す。

## モジュール構成

```
src/splitcomp/
├── config.py            設定項目の定義（dataclass）と YAML 読み込み・上書き
├── cli.py               スクリプト共通の引数処理（--config と key=value）
├── builder.py           設定から分割済みモデルを作る / ボトルネックの保存・読込
├── utils.py             デバイス選択、乱数シード、表の整形など
├── models/registry.py   build_model / register_model（torchvision・timm・独自モデル）
├── data/registry.py     build_dataset / build_dataloader / register_dataset
├── data/ucf101.py       UCF101Clips（動画クリップ）
├── temporal/            動画の時系列符号化（FD / DwD / DPCM, dead-zone 量子化）とパケットロス
│   ├── coding.py        TemporalEncoder / TemporalDecoder（多系列 DPCM、NACK による参照フレーム選択 PRS を含む）
│   └── restore.py       LossRestorer（ロスで欠けた特徴をクラウド側で補う、モデル共通のロス補修 net）
├── split/
│   ├── splitter.py      Splitter（分割点の列挙と分割）, split_sequential（簡易版）
│   └── pipeline.py      SplitPipeline, Backend, LocalBackend
├── compression/
│   ├── base.py          Codec, TensorCodec, ByteCodec, RawTensorCodec, ペイロード形式
│   ├── codecs.py        UniformQuantizer, ZlibCodec, Lz4Codec, ビットパッキング
│   ├── registry.py      build_codec, register_tensor_codec, register_byte_codec
│   └── bottleneck.py    学習型エンコーダ/デコーダ, attach_bottleneck
├── transport/
│   ├── channel.py       SimulatedChannel（帯域・RTT モデル）
│   ├── protocol.py      TCP メッセージ形式, 出力のシリアライズ
│   ├── server.py        TailServer
│   └── client.py        TCPBackend
├── profiling/
│   ├── timer.py         GPU 同期付きの時間計測（timed, Timer）
│   └── layer_profiler.py 分割点ごとのデータ量・計算時間の一覧
├── metrics/
│   ├── evaluator.py     Evaluator（top-k 精度・元モデルとの一致度・独自指標）
│   └── logger.py        実験ディレクトリ作成、ResultWriter（CSV/JSONL）
└── training/trainer.py  BottleneckTrainer（知識蒸留）
```

## 主要クラス

### `Splitter`（split/splitter.py）

モデルを `torch.fx` で「ノードの列」に変換し、k 番目のノードの直後で切ります。

**どの値を送るか**: 「分割点以前に計算され、分割点より後で使われる値」を全て送ります。
そのため ResNet の残差ブロックの途中で切ると、ブロックの入力（skip 用）と途中の値の
2 つを送ることになります（`num_tensors = 2`）。送る値が 1 つの分割点を **clean** と呼びます。

```
x ─▶ conv ─▶ bn ─▶ relu ─▶ (+) ─▶ ...
│                          ▲
└──────── skip ────────────┘
          ↑ ここ（bn の後）で切ると x と bn の出力の 2 つを送る必要がある
```

- `trace_depth` を指定すると、それより深いモジュールを 1 ノードとして扱うので分割点が粗くなります。
- パラメータ参照（`get_attr`）は送らず、tail 側で参照し直します。
- head / tail は元モデルとパラメータを**共有**します。学習で重みを変える場合は
  `copy.deepcopy(model)` を渡してください（`builder.build_split` や学習スクリプトは自動で行います）。
- `torch.fx` で追跡できないモデル（入力の値に依存する `if` を含むなど）は、
  `split_sequential` で「子モジュールを順に適用するだけ」とみなして分割できます。

### `Codec`（compression/base.py）

2 段構成です:

1. `TensorCodec`（非可逆になりうる）: テンソル ⇔ バイト列。`RawTensorCodec`（dtype 変換のみ）、`UniformQuantizer`
2. `ByteCodec`（可逆）: バイト列 ⇔ バイト列。`ZlibCodec`、`Lz4Codec`

ペイロード形式:

```
[4 バイト: ヘッダ長][ヘッダ(JSON): 各値の形・dtype・方式][本体: 各テンソルのバイト列を連結して ByteCodec で圧縮]
```

受信側が復元に必要な情報は全てペイロードに入っているので、そのまま実通信に使えます。
ヘッダもデータ量（`payload_bytes`）に含めて計測しています。
整数テンソルは量子化すると壊れるため、常に元の dtype のまま送ります。

### `SplitPipeline` と `Backend`（split/pipeline.py）

`SplitPipeline.run(x)` が 1 回分の分割推論を実行し、`(出力, 計測値)` を返します。
サーバ側の処理は `Backend` として差し替え可能です。

| Backend | 用途 | 通信時間 |
|---|---|---|
| `LocalBackend(tail, codec, device, channel)` | シミュレーション | `SimulatedChannel` で計算（`channel=None` なら 0） |
| `TCPBackend(host, port, fingerprint)` | 実ネットワーク | 実測（往復時間 − サーバ処理時間） |

### `SimulatedChannel`（transport/channel.py）

```
上り時間 = RTT/2 + payload_bytes × 8 / 上り帯域
下り時間 = RTT/2 + result_bytes  × 8 / 下り帯域
```

パケット損失や TCP の輻輳制御などは考慮しない単純なモデルです。
より精密なモデルが必要なら継承して `uplink_ms` / `downlink_ms` を上書きしてください。

### `TailServer` / `TCPBackend`（transport/）

最初に設定の識別子（fingerprint: モデル・分割点・圧縮方式から計算）を照合し、
一致した場合のみ推論要求を受け付けます。接続は 1 つずつ順番に処理します
（他のクライアントの処理で計測値が乱れないように）。
tail の出力は pytree で平らにして送るので、dict のリスト（物体検出など）も返せます。

### `Evaluator`（metrics/evaluator.py）

- 分類（出力が `(B, クラス数)`、正解が整数）なら top-1 / top-5 精度を自動で計算
- 元モデルの出力（reference）を渡すと、タスクに関係なく `out_mse` / `out_cosine` / `out_agree` を計算
- `metric_fns` に関数を渡すと独自の指標を追加できる

### `BottleneckTrainer`（training/trainer.py）

```
教師:  x → head → feat ──────────────────────────→ tail → y_teacher
生徒:  x → head → feat → Encoder → z → Decoder → feat' → tail → y_student

損失 = distill_weight × 蒸留損失 + task_weight × タスク損失 + rate_weight × mean(|z|)
```

固定する部分（head / tail）は常に eval モードにしており、BatchNorm の統計量は変化しません。

## 時間計測についての注意

- GPU（CUDA / MPS）の処理は非同期なので、計測の前後でデバイスを同期しています（`profiling/timer.py`）。
- 最初の数回は遅いことが多いので、`warmup` 回の空回しの後に計測します。
- `encode_ms` には GPU → CPU のデータ転送時間も含まれます（実機でも必要な処理のため）。
- シミュレーションでは head と tail を同じマシンで実行するため、`head_ms` は「端末の性能」ではなく
  「実験マシンの `devices.head` の性能」です。端末の非力さを模擬するなら `devices.head: cpu` にするか、
  実機で `client.py` を動かしてください。

# 使い方

## 共通: 設定ファイルとコマンドライン上書き

全てのスクリプトは同じ形式で引数を受け取ります。

```bash
python scripts/<スクリプト>.py --config <設定.yaml> [key=value ...]
```

- `--config` で YAML を読み込み、続く `key=value` でその一部を上書きします。
- ネストした項目は `.` でつなぎます: `model.name=resnet50`、`data.batch_size=8`
- リストの要素は番号で指定します: `codecs.0.bits=4`（1 つ目の codec の bits）
- 値は YAML として解釈されます: `true` / `null` / `[1, 2]` / `{a: 1}`
- **zsh では `[` `]` を含む値は引用符で囲んでください**: `'data.kwargs.input_shape=[3,32,32]'`
- 存在しない項目名を書くとエラーになります（タイプミス防止）。
- 使える項目と意味は `src/splitcomp/config.py` に全て書いてあります。`--help` でも概要を表示します。

実行のたびに `runs/<experiment_name>_<日時>/` が作られ、次のファイルが保存されます。

| ファイル | 内容 |
|---|---|
| `config.yaml` | 実際に使われた設定（上書き反映済み）。この条件を再現するときに使う |
| `meta.json` | 実行日時、ホスト名、PyTorch バージョン、GPU 名、git コミットなど |
| `results.jsonl` / `results.csv` | 結果。jsonl は実行中に 1 行ずつ追記されるので途中で止めても残る |

---

## 1. `scripts/list_splits.py` — 分割点の一覧

```bash
python scripts/list_splits.py model.name=resnet18                     # 細かい分割点（clean なもの）
python scripts/list_splits.py model.name=resnet18 model.trace_depth=1 # layer1, layer2 ... 単位
python scripts/list_splits.py model.name=resnet18 --kind all          # skip connection の途中も含む全て
python scripts/list_splits.py --config configs/sweep_resnet50.yaml --no-timing --csv splits.csv
```

| オプション | 意味 |
|---|---|
| `--kind clean` | 送信テンソルが 1 つの分割点だけ表示（既定） |
| `--kind all` | 全ての分割点を表示（`num_tensors` が 2 以上のものも含む） |
| `--no-timing` | 計算時間を測らない（大きいモデルで素早く一覧したいとき） |
| `--csv パス` | 表を CSV に保存 |

表の列:

| 列 | 意味 |
|---|---|
| `index` | 分割点の番号（この番号のノードの直後で切る） |
| `name` | 分割点名。`split_points` / `split_point` に書く名前 |
| `num_tensors` | 送信するテンソルの数 |
| `shapes` | 送信テンソルの形（バッチ次元込み） |
| `KB_fp32` | float32 のまま送った場合のデータ量 |
| `ratio_to_input` | 入力をそのまま送る場合に対するデータ量の比 |
| `head_ms` / `tail_ms` | head / tail の計算時間（中央値） |

**分割点名について**: `torch.fx` のノード名です。`layer1.0.conv1` のようなモジュールは
`layer1_0_conv1` になります。`input`（入力をそのまま送る = 全てサーバ処理）と
`output`（全て端末で処理して出力を送る）という別名も使えます。

---

## 2. `scripts/sweep.py` — シミュレーションで総当たり評価

```bash
python scripts/sweep.py --config configs/sweep_resnet50.yaml
python scripts/sweep.py --config configs/sweep_resnet50.yaml model.name=resnet18 data.num_samples=8
```

**分割点 × 圧縮方式 × 通信路** の全組み合わせについて評価します。
推論は「分割点 × 圧縮方式」ごとに 1 回だけ行い、通信時間はデータ量から各通信路について計算するので、
通信路の数を増やしても実行時間はほとんど増えません。

主な設定項目（`configs/sweep_resnet50.yaml` にコメント付きの例があります）:

| 項目 | 意味 |
|---|---|
| `model.name` | モデル名。`resnet50`（torchvision）, `timm:convnext_tiny`, 登録した独自モデル |
| `model.trace_depth` | 分割点の粗さ。`null` で全層、`1` で最上位のモジュール単位 |
| `data` | データセット。`random` はダウンロード不要のダミーデータ |
| `devices.head` / `devices.tail` | head / tail を動かすデバイス（`cpu`, `cuda`, `mps`, `auto`） |
| `split_points` | `clean` / `all` / 分割点名のリスト |
| `codecs` | 圧縮方式のリスト（下表） |
| `channels` | 通信路のリスト: `{name, bandwidth_mbps, rtt_ms, downlink_bandwidth_mbps}` |
| `warmup` | 計測前の空回し回数 |
| `bottleneck_ckpt` | 学習済みボトルネック（指定すると分割点はチェックポイントのものに固定） |

圧縮方式（`codecs` の各要素 / `codec`）の書き方:

| 書き方 | 意味 |
|---|---|
| `{tensor: raw, dtype: float32}` | 無圧縮（基準） |
| `{tensor: raw, dtype: float16}` | 半精度で送る（データ量 1/2） |
| `{tensor: quant, bits: 8}` | 8 bit 一様量子化（データ量 約 1/4） |
| `{tensor: quant, bits: 4, per_channel: true}` | チャネルごとに範囲を決める 4 bit 量子化 |
| `{tensor: quant, bits: 8, compress: [zlib]}` | 量子化後に zlib で可逆圧縮 |
| `{..., compress: [lz4]}` | lz4（zlib より高速・圧縮率は低め） |
| `{..., name: 好きな名前}` | 結果の `codec` 列に出す名前 |

> **注意**: `per_channel: true` は dim=1 をチャネルとみなします。`(B, 1000)` のような
> 出力に使うとパラメータ（1000 個 × 8 バイト）が本体より大きくなることがあります。

### 出力 CSV（`results.csv`）の列

時間は**バッチあたりの平均** [ms]、データ量は**バッチあたりの平均** [bytes] です。

| 列 | 意味 |
|---|---|
| `model`, `split_point`, `split_index`, `num_tensors` | モデルと分割点 |
| `codec`, `channel`, `bandwidth_mbps`, `rtt_ms` | 圧縮方式と通信路 |
| `batch_size`, `num_samples` | 評価条件 |
| `head_ms` | 端末側の head の計算時間 |
| `encode_ms` | 量子化・圧縮の時間（GPU→CPU 転送を含む） |
| `payload_bytes` | 実際に送るデータ量（ヘッダ込み） |
| `raw_bytes` | 送信テンソルを float32 で送った場合のデータ量 |
| `compression_ratio` | `raw_bytes / payload_bytes` |
| `uplink_ms` | 上り通信時間 = RTT/2 + payload_bytes×8 / 帯域 |
| `decode_ms` | サーバ側の伸長時間 |
| `tail_ms` | サーバ側の tail の計算時間 |
| `result_bytes`, `downlink_ms` | 結果を端末へ返すデータ量と時間 |
| `total_ms` | エンドツーエンド遅延 = 上記の時間の合計 |
| `top1`, `top5` | 正解ラベルに対する精度（分類のとき。random データでは無意味） |
| `out_mse`, `out_cosine` | 元モデルの出力との平均二乗誤差 / コサイン類似度 |
| `out_agree` | 元モデルと予測クラス（argmax）が一致した割合 |

---

## 3. `scripts/train_bottleneck.py` — ボトルネックの学習

```bash
python scripts/train_bottleneck.py --config configs/train_bottleneck.yaml
python scripts/train_bottleneck.py --config configs/train_bottleneck.yaml \
    split_point=layer3 bottleneck.channels=8 bottleneck.downsample=1 epochs=10 max_steps=null
```

分割点にエンコーダ（端末側）とデコーダ（サーバ側）を挿入し、元のモデルを教師として学習します。
既定では head と tail の重みは固定され、エンコーダ / デコーダだけが学習されます。

| 項目 | 意味 |
|---|---|
| `split_point` | 挿入位置（`num_tensors` が 1 の分割点のみ） |
| `bottleneck.channels` | 送信する特徴のチャネル数（ベクトル特徴なら次元数） |
| `bottleneck.downsample` | 空間方向の縮小率（1, 2, 4, ...） |
| `freeze_head` / `freeze_tail` | head / tail の重みを固定するか |
| `distill` | `feature`（特徴の MSE、既定）/ `kl`（出力分布、分類向け）/ `mse` / `none` |
| `task_loss`, `task_weight` | 正解ラベルとの損失（`cross_entropy` など）とその重み |
| `rate_weight` | ボトルネック出力の L1 ノルムの重み（大きいほど疎になり圧縮しやすい） |
| `epochs`, `max_steps`, `lr` | 学習の長さと学習率 |
| `val_data` | 検証データ（`null` で検証なし） |

出力:

- `bottleneck.pt`: チェックポイント（モデル設定・分割点・エンコーダ/デコーダの重みを含む）。毎エポック上書き
- `train_log.jsonl`: エポックごとの損失と検証指標

`bottleneck.pt` を `sweep.py` / `serve.py` / `client.py` の `bottleneck_ckpt=` に渡すと、
ボトルネック付きのモデルで評価できます。

> 既定の設定は**ダミーデータでの動作確認用**です。実際の研究では
> `model.pretrained: true` にし、`data` を ImageNet（`imagefolder`）などに変えてください。

---

## 4. `scripts/serve.py` / `scripts/client.py` — 実ネットワーク実験

サーバ（tail を実行）とクライアント（head を実行）を**同じ設定ファイル**で起動します。
モデル・分割点・圧縮方式が一致しない場合、サーバは接続を拒否します。

```bash
# サーバ機
python scripts/serve.py --config configs/network.yaml host=0.0.0.0 port=50051

# 端末（クライアント）
python scripts/client.py --config configs/network.yaml host=192.168.0.10 port=50051
```

| 項目 | 意味 |
|---|---|
| `host`, `port` | サーバ: 待ち受けアドレス / クライアント: 接続先 |
| `split_point`, `codec` | 分割点と圧縮方式（`codec` は 1 つだけ） |
| `devices.head` | クライアントで head を動かすデバイス |
| `devices.tail` | サーバで tail を動かすデバイス |
| `--no-reference` | （client のみ）元モデルとの一致度を計算しない（端末の負荷を減らす） |

クライアントの `results.csv` には**バッチごと**の結果が入ります。
`comm_ms` は「送信開始〜結果受信完了」の時間からサーバ内の処理時間を引いたもので、
上り・下りの通信と、出力のシリアライズなどの小さなオーバーヘッドを含みます。

**帯域を制限して実験したい場合**は、OS の機能でネットワークを絞ります。

- Linux: `sudo tc qdisc add dev eth0 root tbf rate 10mbit burst 32kbit latency 400ms`
  （遅延は `netem delay 25ms` を併用）
- macOS: Network Link Conditioner（Xcode の Additional Tools）や `dnctl` + `pfctl`

---

## 5. Python API から使う

スクリプトを使わず、ノートブックなどから直接使うこともできます。
最小例は [`examples/quickstart.py`](../examples/quickstart.py) を参照してください。

```python
from splitcomp.models import build_model
from splitcomp.split import Splitter, SplitPipeline, LocalBackend
from splitcomp.compression import build_codec
from splitcomp.config import CodecConfig
from splitcomp.transport import SimulatedChannel

model = build_model("resnet18")
head, tail = Splitter(model, trace_depth=1).split("layer2")
codec = build_codec(CodecConfig(tensor="quant", bits=8))
backend = LocalBackend(tail, codec, device, SimulatedChannel(10, 50))
output, stats = SplitPipeline(head, codec, backend, device).run(x)
```

自作の `nn.Module` もそのまま `Splitter(my_model)` に渡せます（`build_model` を通す必要はありません）。

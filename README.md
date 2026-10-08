# splitcomp — Split Computing 汎用実験基盤

任意の PyTorch モデルを**任意の層で分割**し、端末（head）とサーバ（tail）に分けて推論する
Split Computing の研究用基盤です。中間特徴の**圧縮**、**通信の模擬／実通信**、
**遅延・データ量・精度の計測**、**学習型ボトルネックの蒸留学習**までを共通の仕組みで扱えます。

```
┌──────────── 端末（エッジ）側 ────────────┐               ┌──────────── サーバ側 ────────────┐
 入力 ─▶ head ─▶ [Encoder] ─▶ 圧縮(量子化+zlib) ─▶ 通信 ─▶ 伸長 ─▶ [Decoder] ─▶ tail ─▶ 出力
         head_ms   (任意)       encode_ms      uplink_ms  decode_ms   (任意)      tail_ms
```

## できること

| 機能 | 内容 |
|---|---|
| 任意の分割点 | `torch.fx` でモデルをグラフ化し、どの層でも分割可能。ResNet の skip connection のように複数のテンソルをまたぐ位置でも正しく分割される |
| 特徴の圧縮 | float32/16/bf16、1〜16 bit 一様量子化（チャネル単位も可）、zlib / lz4。組み合わせ自由・独自方式も追加可能 |
| 通信 | 帯域・RTT から計算するシミュレーション と、TCP による実ネットワーク通信の両方 |
| 計測 | head / encode / 通信 / decode / tail の各時間、送信データ量、圧縮率、精度（top-k）、元モデルとの一致度 |
| 学習 | 分割点にエンコーダ / デコーダ（ボトルネック）を挿入し、元モデルからの知識蒸留で学習 |
| タスク非依存 | 分類以外のモデルでも動作（出力が dict のリストでも可）。指標は関数を渡して追加できる |

## インストール

```bash
cd split-computing-experiment
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"        # lz4 と pytest も入る。timm を使う場合は ".[dev,timm]"
pytest                         # 動作確認（1 分以内に終わる）
```

## 5 分クイックスタート

```bash
# 1) 分割点の一覧を見る（どこで切ると送信データが小さいか、計算時間はどうか）
python scripts/list_splits.py model.name=resnet18 model.trace_depth=1

# 2) 分割点 × 圧縮方式 × 通信路 を総当たりでシミュレーション評価
python scripts/sweep.py --config configs/sweep_resnet50.yaml data.num_samples=8

# 3) ボトルネックを学習し、そのチェックポイントで再評価
python scripts/train_bottleneck.py --config configs/train_bottleneck.yaml
python scripts/sweep.py --config configs/sweep_resnet50.yaml model.name=resnet18 \
    bottleneck_ckpt=runs/<学習の出力ディレクトリ>/bottleneck.pt

# 4) 実ネットワークで分割推論（別々のターミナル / 別々のマシンで）
python scripts/serve.py  --config configs/network.yaml host=0.0.0.0
python scripts/client.py --config configs/network.yaml host=<サーバのIP>
```

結果は `runs/<実験名>_<日時>/` に `results.csv`（1 行 = 1 条件）として保存されます。

Python から直接使う最小例は [`examples/quickstart.py`](examples/quickstart.py) を見てください。

## ドキュメント

| ファイル | 内容 |
|---|---|
| [docs/usage.md](docs/usage.md) | 各スクリプトの使い方、設定ファイルの書き方、出力 CSV の各列の意味 |
| [docs/architecture.md](docs/architecture.md) | 全体の構成と処理の流れ、各モジュール・クラスの役割 |
| [docs/extending.md](docs/extending.md) | 独自モデル・データセット・圧縮方式・評価指標・損失の追加方法 |
| [docs/experiments.md](docs/experiments.md) | 典型的な実験の進め方と結果の見方 |

## ディレクトリ構成

```
configs/            設定ファイルの例（各項目に日本語コメントあり）
docs/               ドキュメント
examples/           Python API の使用例
scripts/            実験用スクリプト（list_splits / sweep / train_bottleneck / serve / client）
src/splitcomp/      ライブラリ本体
  split/              モデル分割（Splitter）と分割推論（SplitPipeline）
  compression/        圧縮方式（Codec）と学習型ボトルネック
  transport/          通信シミュレーションと TCP サーバ / クライアント
  profiling/          時間計測
  metrics/            評価指標と結果の保存
  training/           ボトルネックの蒸留学習
  models/, data/      モデル・データセットの登録と構築
  config.py           設定項目の定義（全項目の説明あり）
tests/              テスト
runs/               実験結果の出力先（git 管理外）
```

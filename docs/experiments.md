# 実験レシピ

典型的な実験の進め方と、結果の見方をまとめます。
どの実験も、まず `data.name=random` と小さい `num_samples` で動作確認してから本番データで実行するのがおすすめです。

---

## レシピ 1: 分割点 × 帯域のトレードオフを測る

**問い**: どこで分割すると、エンドツーエンド遅延が最小になるか？ それは帯域によってどう変わるか？

1. 分割点の候補を確認する

   ```bash
   python scripts/list_splits.py model.name=resnet50 model.trace_depth=1 devices.head=cpu
   ```

   - `KB_fp32` が小さく、`head_ms` が大きすぎない点が候補になります。
   - 一般に、浅い層は特徴が大きく（入力より大きいことも多い）、深い層ほど小さくなります。

2. 候補の分割点・複数の帯域で評価する（`configs/sweep_resnet50.yaml` を編集）

   ```yaml
   split_points: [input, layer1, layer2, layer3, layer4, output]
   codecs:
     - {tensor: raw, dtype: float32}
   channels:
     - {name: 1Mbps,   bandwidth_mbps: 1,   rtt_ms: 50}
     - {name: 10Mbps,  bandwidth_mbps: 10,  rtt_ms: 30}
     - {name: 100Mbps, bandwidth_mbps: 100, rtt_ms: 10}
   ```

3. 結果を見る

   ```python
   import pandas as pd
   df = pd.read_csv("runs/sweep_resnet50_xxxx/results.csv")
   print(df.pivot_table(index="split_point", columns="channel", values="total_ms"))
   ```

**見どころ**: 帯域が狭いと「全て端末で処理（output）」や深い分割点が有利になり、
帯域が広いと「全てサーバで処理（input）」が有利になる、という切り替わりが見えるはずです。
`head_ms` を現実的にするため、端末を模擬する側は `devices.head: cpu` にしてください。

---

## レシピ 2: 量子化ビット数と精度の関係を調べる

**問い**: 何ビットまで落としても精度が保たれるか？

```yaml
model: {name: resnet50, pretrained: true, trace_depth: 1}
data:
  name: imagefolder
  root: /path/to/imagenet
  split: val
  num_samples: 5000
  batch_size: 50          # 精度評価が目的なので大きくてよい
split_points: [layer2, layer3]
codecs:
  - {tensor: raw, dtype: float32}
  - {tensor: quant, bits: 8,  compress: [zlib]}
  - {tensor: quant, bits: 6,  compress: [zlib]}
  - {tensor: quant, bits: 4,  compress: [zlib]}
  - {tensor: quant, bits: 4,  per_channel: true, compress: [zlib]}
  - {tensor: quant, bits: 2,  per_channel: true, compress: [zlib]}
channels:
  - {name: 10Mbps, bandwidth_mbps: 10, rtt_ms: 30}
```

**見る列**: `top1`（精度）、`out_agree`（元モデルとの一致率）、`payload_bytes`、`compression_ratio`。
横軸 `payload_bytes`、縦軸 `top1` でプロットすると、レート-精度曲線になります。

```python
import matplotlib.pyplot as plt
for sp, g in df.groupby("split_point"):
    plt.plot(g.payload_bytes / 1024, g.top1, "o-", label=sp)
plt.xscale("log"); plt.xlabel("送信データ量 [KB/バッチ]"); plt.ylabel("top-1"); plt.legend()
```

> `batch_size` を変えると `payload_bytes` や時間は「バッチあたり」の値になります。
> サンプルあたりにするには `batch_size` で割ってください。

---

## レシピ 3: 学習型ボトルネック → 評価

**問い**: 学習したボトルネックで、量子化だけの場合よりどれだけデータ量を減らせるか？

1. 学習（ImageNet の学習済み ResNet-50 を教師にする例）

   ```bash
   python scripts/train_bottleneck.py --config configs/train_bottleneck.yaml \
       model.name=resnet50 model.pretrained=true split_point=layer2 \
       bottleneck.channels=16 bottleneck.downsample=2 \
       data.name=imagefolder data.root=/path/to/imagenet data.num_samples=null \
       val_data.name=imagefolder val_data.root=/path/to/imagenet val_data.num_samples=5000 \
       distill=kl task_loss=cross_entropy task_weight=0.5 epochs=10 max_steps=null
   ```

   - `distill=feature` は速くタスク非依存。分類なら `kl` の方が最終精度は高くなりやすいです。
   - 精度が足りなければ `freeze_tail=false` で tail も微調整してみてください。

2. ボトルネック付きで、量子化と組み合わせて評価

   ```bash
   python scripts/sweep.py --config configs/sweep_resnet50.yaml \
       bottleneck_ckpt=runs/bottleneck_xxxx/bottleneck.pt \
       data.name=imagefolder data.root=/path/to/imagenet data.num_samples=5000 data.batch_size=50
   ```

3. 同じ分割点の「ボトルネックなし」の結果（レシピ 2）と、`payload_bytes` と `top1` を比較します。

> 学習時には量子化を行っていないため、強い量子化（2〜4 bit）と組み合わせると精度が落ちることがあります。
> 量子化を考慮した学習（ノイズ注入や Straight-Through Estimator）を入れたい場合は
> `BottleneckTrainer._forward` の `z` に処理を加えてください。

---

## レシピ 4: 実機・実ネットワークで測る

1. サーバ機と端末（Raspberry Pi、Jetson、ノート PC など）の両方にこのリポジトリをインストール
2. 同じ設定ファイル（`configs/network.yaml`）を両方に置く（`bottleneck_ckpt` を使う場合はそのファイルも）
3. サーバで `python scripts/serve.py --config configs/network.yaml host=0.0.0.0`
4. 端末で `python scripts/client.py --config configs/network.yaml host=<サーバIP> --no-reference`
5. 分割点や圧縮方式を変えて繰り返す（サーバも同じ設定で再起動が必要）

シミュレーション結果（`sweep.py`）と実測（`client.py`）の `comm_ms` / `total_ms` を比べると、
シミュレーションの通信モデルの妥当性を確認できます。

---

## 再現性のためのチェックリスト

- [ ] 実行ディレクトリの `config.yaml` と `meta.json` を論文の実験記録と紐づけたか
- [ ] `seed` を固定したか（`num_workers > 0` の場合はデータの読み込み順に注意）
- [ ] 時間計測で `warmup` を十分取ったか（特に GPU / MPS）
- [ ] 時間は「バッチあたり」であることを考慮したか（`batch_size`）
- [ ] 計測中に他の重い処理を動かしていないか
- [ ] git でコードをコミットしてから実験したか（`meta.json` に commit が記録される）

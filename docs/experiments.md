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

## レシピ 5: 動画の時系列符号化（Delta coding）をパケットロス下で評価する

Shiba et al., "Delta Coding With Dead-Zone Quantization for Bandwidth-Efficient Split Computing on
Video Streams"（IEEE Access 2026）の方式を、パケットロスのある通信路で評価します。
同論文では、パケットロスは今後の課題とされています（§VII-E-5）。

1. `pip install -e ".[video]"` で PyAV を入れる。UCF-101 は手で用意しなくてよい。初回実行時に、使う 30 本だけ
   （約 15MB）が HuggingFace の ZIP ミラーから `./data/UCF-101/<クラス名>/` に自動でダウンロードされる
   （7GB 全体は落とさない）
2. 動作確認:
   `python scripts/packet_loss_sweep.py --config configs/packet_loss_ucf101.yaml data.kwargs.num_clips=2 'seeds=[0]'`
3. 新手法を試す段階（お試し）: `python scripts/packet_loss_sweep.py --config configs/packet_loss_ucf101_quick.yaml`
   - 本番の設定を `base:` で読み、モデルを resnet50 / vgg16 の 2 つ、損失を Gilbert-Elliott（と基準の no_loss）だけに減らしたもの。
     方式・ビット数・クリップ数・シードは本番と同じ（方式を足すときは本番の設定に書けば、お試し側にも反映される）
   - 損失・シードの組は 46 → 16 通り。結果は `runs/packet_loss_ucf101_quick_<日時>/` に出るので本番と混ざらない
   - `--scheme D+DZ --scheme D+DZ+R` のように方式も絞れる（`--target` と同じく複数回指定できる。本番の設定にも使える）
4. 本番: `python scripts/packet_loss_sweep.py --config configs/packet_loss_ucf101.yaml`
   （時間がかかるので、ssh で動かすなら tmux の中で。Slurm クラスタでは下の「Slurm クラスタで動かす」を参照）

実行時間について:
- 端末側の符号化（I/P の判定・差分の計算・量子化・エントロピー・パケット分割）は損失条件によらないので、
  クリップごとに (方式, ビット数) ごとに 1 回だけ行い、全ての損失条件・シード（既定の設定で 46 通り）で使い回す。
  以前は損失条件ごとに符号化し直していた。結果は以前の版と完全に一致する（`tests/test_temporal.py` で確認している）。
  ResNet50/layer2 相当の特徴で、CPU 側の処理が約 10 倍速くなる
- 符号化・復号は CPU で行い、GPU は tail だけに使う
- torch の CPU スレッド数は、ジョブに割り当てられたコア数（`os.sched_getaffinity`）を超えないように自動で調整する
  （torch の既定はマシン全体のコア数なので、割り当てが一部だけだとスレッドが多すぎて遅くなる）。
  使ったスレッド数は起動時に表示され、`meta.json` の `cpu_threads` に残る。手動で決めたいときは `OMP_NUM_THREADS=8 python ...`
- GPU が 2 枚あるなら、モデルを分けて 2 プロセスを並列に動かせる（あとで `merge_results.py` でまとめる）。
  速い GPU に重いモデル（vgg16, resnet50）を割り当てる。CPU コアは 2 プロセスで分け合うことになる:

  ```bash
  python scripts/packet_loss_sweep.py --config configs/packet_loss_ucf101.yaml --target resnet50 --target vgg16 devices.head=cuda:0 devices.tail=cuda:0
  python scripts/packet_loss_sweep.py --config configs/packet_loss_ucf101.yaml --target efficientnet_v2_s --target mobilenet_v2 devices.head=cuda:1 devices.tail=cuda:1
  ```

  どちらが `cuda:0` かは `nvidia-smi` で確認する（`CUDA_DEVICE_ORDER=PCI_BUS_ID` を付けると `nvidia-smi` と同じ番号になる）

時間制限のある計算機（時間を指定して借りる JupyterLab など）では、モデルごとに別々に実行して、あとでまとめる:

```bash
python scripts/packet_loss_sweep.py --config configs/packet_loss_ucf101.yaml --list-targets   # 番号と名前を確認
python scripts/packet_loss_sweep.py --config configs/packet_loss_ucf101.yaml --target efficientnet_v2_s
python scripts/packet_loss_sweep.py --config configs/packet_loss_ucf101.yaml --target resnet50
python scripts/packet_loss_sweep.py --config configs/packet_loss_ucf101.yaml --target vgg16
python scripts/packet_loss_sweep.py --config configs/packet_loss_ucf101.yaml --target mobilenet_v2
python scripts/merge_results.py runs/packet_loss_ucf101_*_2026* -o runs/packet_loss_ucf101_merged
```

- `--target` にはモデル名・`モデル名/分割点`・番号のどれでも指定できる。複数回付けると、その分だけまとめて実行する
- 結果は `runs/packet_loss_ucf101_<モデル名>_<日時>/` に出る
- `merge_results.py` は、各フォルダの `results` / `gop_curve` を 1 つにまとめる。同じモデルの結果が複数あるときは、新しい実行のほうを使う
- `*.jsonl` はモデル 1 つ分が終わるたびに追記される。途中で止まった実行（CSV が無いもの）でも、終わったモデルの分はまとめられる

途中で止まった（セッションが切れた）実行は、同じフォルダの続きから再開できる:

```bash
python scripts/packet_loss_sweep.py --resume runs/packet_loss_ucf101_resnet50_20261013-120000
```

- 実行中のモデルの途中経過は、クリップ 1 本ごとに `checkpoint.pkl` に保存される（モデルが終わると消える）。
  再開すると、終わったモデルは飛ばし、実行中だったモデルは続きのクリップから計算する。失うのは止まったときに処理中だった 1 本分だけ
- 設定は実行フォルダの `config.yaml` から読むので `--config` / `--target` は要らない（付けても無視される）。
  条件が変わると前回分と混ざるので、上書きできるのは `devices.*` だけ（例: 別マシンで再開するときの `devices.head=cuda`）
- クリップごとの乱数は固定なので、再開しても中断なしで実行したのと同じ結果になる
- 開始時に再開用のコマンドが表示される。何度止まっても同じコマンドで続けられる
- この機能を入れる前の版で作った実行フォルダ（`checkpoint.pkl` なし）も `--resume` できる。終わったモデルは飛ばし、途中だったモデルは最初からやり直す
- 再開したフォルダも、そのまま `merge_results.py` に渡せる

### 研究室の Slurm クラスタで動かす

`scripts/slurm/packet_loss.sbatch` を使うと、配列ジョブで 4 モデルを別々のジョブとして並列に実行できる。
GPU パーティション（`-p GPU`）・12 コア・時間制限 2 日（GPU パーティションの上限）で投げる。方式は既定で F+Sym / D+Sym / D+DZ / D+DZ+S2 / D+DZ+S5 / D+DZ+PRS / D+DZ+PRS3
（D+DZ+R を入れるときは `SCHEMES=` と空にするか、`SCHEMES="... D+DZ+R"` とする）。
以下はすべてログインノード上で、リポジトリのトップで行う（ホームは計算ノードと共有されているので、
ログインノードで作った venv・動画・結果がそのままジョブから見える）。

1. 準備（最初の 1 回だけ）:

   ```bash
   python3 -m venv .venv && source .venv/bin/activate
   pip install -e ".[video,analysis]"
   python scripts/slurm/prefetch.py   # 動画 30 本と 4 モデルの学習済み重みを先にダウンロード（ジョブ同士の競合を防ぐ）
   mkdir -p runs/slurm                # ジョブのログの出力先（Slurm は作ってくれない）
   ```

2. 動作確認（target 1 番 = resnet50 を、クリップ 2 本・シード 1 個で）:

   ```bash
   EXTRA="data.kwargs.num_clips=2 seeds=[0]" sbatch --array=1 scripts/slurm/packet_loss.sbatch
   squeue -u $USER                          # ST が R なら実行中、PD なら待ち
   cat runs/slurm/packet_loss_*_1.out       # 使ったノード・GPU・結果の表
   ```

   `.out` の最初のほうに `CPU スレッド数: 12` と `デバイス: head=cuda, tail=cuda` のように出る。
   スレッド数が 1 なら `-c` が効いていない。
   GRES が未設定なので、GPU は同じノードの他のジョブと共有になることがある（`nvidia-smi` の出力で使用状況が分かる）

3. 本番（4 モデルを並列に）:

   ```bash
   sbatch scripts/slurm/packet_loss.sbatch
   tail -f runs/slurm/packet_loss_<ジョブID>_*.err   # 進捗バー
   ```

   ノードの GPU が PyTorch から使えないと、CPU で黙って何倍も時間をかけないように、ジョブは計算を始める前にエラーで止まる
   （`.err` に `エラー: <ノード名> では PyTorch から GPU が使えません`）。
   例えば NVIDIA ドライバの更新後にノードを再起動していない場合は、`nvidia-smi` が `Driver/library version mismatch` になってこうなる
   （管理者に再起動を頼む）。それまでは、そのノードを外して投げる。各ノードで GPU が使えるかは `srun` で確かめられる:

   ```bash
   srun -p GPU -w <ノード名> nvidia-smi                         # GPU の一覧が出れば使える
   sbatch --exclude=<ノード名> scripts/slurm/packet_loss.sbatch   # 複数外すときは --exclude=<ノード名1>,<ノード名2>
   ```

   CPU で実行してよいときは `ALLOW_CPU=1 sbatch ...` とする

4. 2 日の制限で切れた（`.err` に `DUE TO TIME LIMIT`）ら、`.out` に出ている実行フォルダを指定して、同じ番号で投げ直す:

   ```bash
   RESUME=runs/packet_loss_ucf101_vgg16_20261023-120000 sbatch --array=2 scripts/slurm/packet_loss.sbatch
   ```

   `--array` の番号は何でもよい（RESUME では target は実行フォルダの config.yaml から決まる）が、ログを追いやすいので同じ番号にする

5. 全て終わったらまとめて図にする:

   ```bash
   python scripts/merge_results.py runs/packet_loss_ucf101_*_2026* -o runs/packet_loss_ucf101_merged
   python scripts/packet_loss_plot.py runs/packet_loss_ucf101_merged
   ```

- 設定ファイルは `CONFIG=configs/packet_loss_ucf101_quick.yaml` で変えられる（quick は target が 2 つなので `--array=0-1`）
- 実行されるコマンドは `DRY_RUN=1 SLURM_ARRAY_TASK_ID=1 bash scripts/slurm/packet_loss.sbatch` で確認できる
- 他の環境変数（`EXTRA` / `SCHEMES` / `VENV` / `ALLOW_CPU`）はスクリプトの先頭のコメントを参照

使う動画（`selection: paper`、`src/splitcomp/data/ucf101.py` の `PAPER_CLIPS`）:
- 論文には「UCF-101 の 5 カテゴリにまたがる 30 クリップ・各 30 連続フレーム」としか書かれておらず、
  具体的なリストは公開されていない。そこで次の方針で選んだ（推測）
  - カテゴリのクラス数に比例して配分: Human–Object 6 / Body-Motion 5 / Human–Human 1 / Instruments 3 / Sports 15
  - 論文が例に挙げている Typing と Basketball は必ず含める。残りは seed=0 で選ぶ
  - 各クラスの `v_<クラス名>_g01_c01.avi`（公式 split 1 のテスト側）の先頭 30 フレームを使う
- `selection: random` にすると、`seed` で選んだ別のクリップの組で頑健性を確かめられる

モデル:
- GoP（`gop` フレーム）の先頭が I フレーム（特徴を対称量子化）、残りが P フレーム
- P フレームで送るもの: FD は特徴そのもの、DwD は真の差分、DPCM は予測残差
- 量子化インデックスは、全ての方式で常にパケットごとに独立した適応算術符号でエントロピー符号化し、
  MTU に収まるだけ詰める（`splitcomp.temporal.packet.entropy_layout`。学習用の `train_restorer.py` も同じ）
  - 各パケットの先頭で頻度表を初期化する（全 2^b 記号のカウントを 1 から始める Laplace 推定）ので、
    付加情報なしでパケット単体で復号でき、ロスしたパケットの要素だけが欠ける。
    パケットをまたいで符号化すると 1 パケットのロスでフレーム全体が壊れるので、そうはしない
  - 実際のビット列は作らず、符号長を閉じた式で正確に計算して詰める。
    ヘッダは 21 バイト（フレーム番号・種類・連番・パケット数・オフセット・q・要素数）、符号の終端として 2 バイトを足す
  - パケット数はフレームの中身で変わる（dead-zone 量子化の P フレームなどは少なくなる）。そのため、同じパケット損失率でも
    フレーム内で欠ける要素の割合やロスの分布が方式ごとに違う
  - 以前の版（b ビット固定長で詰めていた）の結果とは、`packets_per_frame` / `bytes_per_frame` とロスのパターンが違うので、
    ロスありの一致率は直接比べられない（ロスなしの一致率は同じ）
- 損失モデル（`losses`）: `bernoulli` / `gilbert_elliott`（`burst_len`）/ `frame`
- 届かなかった要素は `concealment` で補う。`hold` は直前の値（P フレームでは差分 0 と同じ）、`zero` は 0
- 端末はロスを知らない。そのため DPCM では、ロスによる端末とクラウドの状態のずれが次の I フレームまで残る

出力:
- `results.csv`: 条件（モデル × 方式 × ビット数 × 損失）ごとに、`agreement`（量子化・ロスなしの元モデルとの
  top-1 一致率）、`agreement_ci95`、`agreement_drop`（同じ方式のロスなしとの差）、`feature_mse`、
  `loss_rate`（実際のパケット損失率）、`packets_per_frame`、`bytes_per_frame`、`entropy_bpe`、`coding`
  - `bytes_per_frame` は、パケットごとに適応算術符号で符号化したときに実際に送るバイト数（ヘッダ込み）。
    `coding` は符号化の種類（今は常に `adaptive_ac`。`packet_loss_plot.py` が古い固定長の結果と見分けるのに使う）
  - `entropy_bpe` はフレーム全体の 0 次エントロピー [bit/要素] で、理想的なエントロピー符号化の参考値。
    パケットごとに符号化すると、ヘッダ・終端・パケットごとの頻度表の学習の分だけ `entropy_bpe × 要素数 / 8` より大きくなる
- `gop_curve.csv`: GoP 内の位置ごとの `agreement` / `feature_mse`。ロスの伝搬が見える

図と要約表は `packet_loss_plot.py` で作れる（`pip install -e ".[analysis]"` が必要）。上の 2 つから、
損失率ごとの一致率・DPCM と FD の差・GoP 内の伝搬などの PNG と `summary.md` を `<実行フォルダ>/figures/` に書き出す:

```bash
python scripts/packet_loss_plot.py runs/packet_loss_ucf101_merged
```

```python
import pandas as pd
df = pd.read_csv("runs/<実行ディレクトリ>/results.csv")
# 損失モデルごとに分ける（混ぜると、同じ loss_p の Bernoulli と GE の行が平均されてしまう）。loss_p = 0 の行は基準として入れる
for m in ["bernoulli", "gilbert_elliott", "frame"]:
    sub = df[(df.loss_model == m) | (df.loss_p == 0)]
    print(f"\n== {m}")
    print(sub.pivot_table(index=["model", "bits", "loss_p"], columns="scheme", values="agreement").round(3))
```

確認ポイント:
- `loss_p = 0` の行は、論文 Table 8 の FD / DPCM の値とおおむね一致するはず（クリップの選び方の違いで多少ずれる）
- D+DZ（DPCM）が F+Sym（FD）を下回り始める損失率
- `gop_curve.csv` で、ロス後に誤差が GoP の終わりまで残るか
- `gop` を短くしたとき（`gop=10` など）の、ロス耐性と送信量のトレードオフ

### ロス補修 net（D+DZ+R）

Itahara et al. (2021) は、パケットロスを模擬しながら DNN を再学習してロスへの耐性を得ています。
D+DZ+R は再学習をしない代わりに、**クラウド側で復号のあと・tail の前**に小さな CNN（`splitcomp.temporal.restore.LossRestorer`）を挟みます。

- P フレームでロスが起きたときだけ動く。入力は「前フレームの復元特徴 f̂_{t−1}」「今のフレームの復元特徴（欠けた要素は hold で埋めたもの）」「届いたマスク」
- 全てのチャネルで同じ重みを使う（1 チャネル = 1 サンプル）。そのため 1 つのモデルで学習した net を、
  チャネル数・空間サイズの違う他のモデル・分割点にもそのまま使える。チャネル間の情報は、届いた要素の差分を
  チャネル方向に平均した「どこが動いたか」のマップ（チャネル数によらない）として渡す
- head / tail は固定のまま。学習の目的は特徴の MSE（「ロスがなかったときの復元特徴」との差）だけで、tail もラベルも使わない（タスク非依存）
- 補った値は復号器の状態（次のフレームの基準）にもなるので、DPCM のずれの伝搬も減る
- 最後の層は 0 で初期化するので、学習前は D+DZ と全く同じ結果になる。ロスがない条件でも D+DZ と同じ結果になる

手順:

1. 学習する（既定は resnet50/layer2 の特徴で学習。動画は評価用の 30 本（g01）と重ならないよう、公式 split 1 の train 側 g08–g25 から
   101 クラス × 2 本を自動でダウンロードする。約 100MB）

   ```bash
   # 動作確認（クリップ 4 本・1 エポック）
   python scripts/train_restorer.py --config configs/train_restorer_ucf101.yaml data.kwargs.num_clips=4 data.kwargs.clips_per_class=1 val_clips=1 epochs=1 save_to=null
   # 本番。重みは runs/train_restorer_ucf101_<日時>/restorer.pt と checkpoints/restorer_resnet50_layer2.pt に保存される
   python scripts/train_restorer.py --config configs/train_restorer_ucf101.yaml
   ```

   エポックごとに `results.csv` に `train_loss` と、検証クリップでの復号誤差（ロスなしの復元特徴に対する相対誤差）
   `val_rel_err`、hold だけの場合の `val_rel_err_hold`、その比 `val_ratio`（1 未満なら hold より良い）が出る。
   検証の誤差が最も小さかったエポックの重みが残る

2. 評価する。`configs/packet_loss_ucf101.yaml` の `schemes` に D+DZ+R の行がある（`restorer:` に重みのパス）。
   学習していないときはこの行を消すかコメントアウトする

   ```bash
   python scripts/packet_loss_sweep.py --config configs/packet_loss_ucf101.yaml
   ```

   D+DZ と D+DZ+R は符号化が同じなので、符号化は 1 回だけ行って使い回す

確認ポイント:
- 学習に使ったモデル（resnet50）と、使っていないモデル（efficientnet_v2_s / vgg16 / mobilenet_v2）での、D+DZ からの改善の差（汎用性）
- 改善が大きいのはどの損失モデルか（`interleave: false` では 1 パケットのロスで数チャネル分がまるごと欠けるので、
  他のチャネルの動きからどれだけ推定できるかが効く）
- 学習時の損失（`losses`）に入れていない損失率・バースト長への汎化
- 複数のモデルで学習した net（`targets` に複数書く）との比較

### 多系列 DPCM（D+DZ+S2 / D+DZ+S5）

D+DZ は P フレームが直前のフレーム f̂_{t−1} を参照するので、端末がロスを知らない条件では 1 回のロスが
次の I フレームまで GoP 内の全フレームに伝搬します。多系列 DPCM では、P フレームの参照先を **n フレーム前** f̂_{t−n} にします。
フレームは GoP 内の位置 t mod n で n 本の独立な系列に分かれ、ロスはそのフレームが属する系列（GoP の 1/n）にしか伝搬しません。

参照の構造（n = 2, GoP = 6 の例。矢印は「参照される ← 参照する」）:

```
0(I) ← 1
0(I) ← 2 ← 4        系列 0（偶数フレーム）
   1 ← 3 ← 5        系列 1（奇数フレーム）
```

- I フレームは GoP 先頭の 1 枚だけ。GoP 内の位置 1..n−1 は I フレームを参照する（参照フレーム = max(t − n, GoP 先頭)）。
  I フレームを失うと、今まで通り GoP 全体に影響する
- P フレームでロスした要素の hold は参照フレームの f̂_{t−n} で埋める（差分 0 とみなす）。I フレームのロスは直前の f̂_{t−1} で埋める
- 代わりに参照が時間的に遠くなるので差分が大きくなる。量子化の範囲 R はフレームごとに差分から決めるので、
  同じビット数ではステップ q が広がって量子化誤差が増える（dead-zone で 0 になる要素も減るので、エントロピー・送信量も変わりうる）。
  n = 1 が通常の D+DZ（結果も完全に一致する）
- 設定は方式の `streams: n`（`configs/packet_loss_ucf101.yaml` に D+DZ+S2 / D+DZ+S5 がある）。
  `TemporalEncoder(..., streams=n)` を直接使ってもよい（復号器は各フレームの `ref` を見るので n を知らなくてよい）
- 結果の `streams` 列に系列数が入る。restorer（D+DZ+R）と組み合わせると prev に f̂_{t−n} が渡るが、
  学習は n = 1 で行っているので既定では組み合わせていない

```bash
python scripts/packet_loss_sweep.py --config configs/packet_loss_ucf101_quick.yaml --scheme D+DZ --scheme D+DZ+S2 --scheme D+DZ+S5
```

確認ポイント:
- ロスなしでの代償: `no_loss` の `agreement` / `feature_mse`（量子化誤差の増加）と `entropy_bpe` / `bytes_per_frame` を D+DZ と比べる
- ロス耐性: `agreement_drop` が D+DZ よりどれだけ小さいか。F+Sym（ロスが伝搬しない）にどこまで近づくか
- `gop_curve.csv` で GoP 内の位置ごとの劣化の様子（D+DZ のような後半ほどの悪化が緩やかになるか）
- バースト（Gilbert-Elliott）では連続するフレームが別の系列に入るので、1 系列あたりのロスが分散される効果がどれだけあるか

### PRS: 壊れたフレームの番号だけを返す参照フレーム選択（D+DZ+PRS / D+DZ+PRS3）

仕組みの詳しい解説（具体例・実測例・前提）は [d_dz_prs.md](d_dz_prs.md) を参照。

動画符号化の Reference Picture Selection（NACK モード）を応用したもの。D+DZ は端末がロスを知らないので、ロスが次の I フレームまで伝搬します。
PRS では、クラウドが **パケットが 1 つでも落ちたフレームの番号だけ** を NACK として端末に返します（届いたフレームには何も返さない）。
端末は NACK を受け取ったら、それ以降のフレームを **クラウド側で壊れていないフレーム** を参照して符号化するので、
伝搬は NACK が届くまでの数フレームで止まります。返すのは壊れたフレームの番号（数バイト）だけなので、端末→クラウドの送信量は増えません。

例（d = 2, フレーム 3 が壊れた場合。矢印は「参照される ← 参照する」）:

```
0(I) ← 1 ← 2 ← 3(ロス) ← 4          3 の NACK は 5 を符号化する前に届く（3, 4 はずれたまま）
               2 ← 5 ← 6 ← ...      5 は壊れていない最新の 2 を参照する → 5 以降は元に戻る
```

- フィードバック遅延 `feedback_delay: d` [フレーム]: フレーム t の NACK はフレーム t + d を符号化する前に端末に届く
  （30fps なら d = 1 は RTT 33ms 以内、d = 3 は 100ms 以内）。フィードバック路ではロスは起きないとする（NACK は小さいので冗長に送れる前提）
- 端末は「NACK されたフレーム、または参照先が壊れたフレーム」を壊れたとみなす（参照関係は端末が知っているので伝搬も端末で計算できる）。
  NACK がまだ来ていないフレームは無事とみなす（ACK を返さないので楽観的）
- 参照の選び方: 通常の参照（max(t − n, GoP 先頭)）以前で無事な最新のフレーム → 無ければ t より前で無事な最新のフレーム →
  それも無ければ（GoP 先頭の I フレームが壊れたとき）そのフレームを I フレームにする（送信量は実際の I フレーム分で数える）
- 参照先 t − ref はパケットヘッダの「種類」u8 の残り 7 ビットに入れる想定なので、ヘッダは増えない
- ロスがなければ D+DZ と全く同じ符号化・結果になる。多系列（`streams`）や restorer（`restorer`）とも組み合わせられる
- 設定は方式の `prs: true` と `feedback_delay: d`（`configs/packet_loss_ucf101.yaml` に D+DZ+PRS（d=1）/ D+DZ+PRS3（d=3）がある）。
  Python からは `TemporalEncoder(..., prs=True)` を作り、NACK を `enc.nack(t)` で渡す（復号器は各フレームの `ref` を見るので変更不要）
- 端末の符号化がロスに依存するので、PRS だけは損失条件ごとに符号化と復号を交互に行う（`prs_clip`）。
  今の GoP で NACK を受ける前のフレームはフィードバックなしと同じ符号化なので、そのパケット分割を使い回す（結果は完全に一致する）。
  それでもロスの多い条件では D+DZ より時間がかかる
- 結果の列: `prs`, `feedback_delay`, `nacks_per_frame`（1 フレームあたりの NACK 数 = フィードバックの量。1 つ = フレーム番号 u32）,
  `forced_intra_per_frame`（参照できる無事なフレームが無く I フレームにした回数）

```bash
python scripts/packet_loss_sweep.py --config configs/packet_loss_ucf101_quick.yaml --scheme D+DZ --scheme D+DZ+PRS --scheme D+DZ+PRS3
```

確認ポイント:
- `no_loss` で D+DZ と `agreement` / `bytes_per_frame` が一致すること（ロスがなければ代償なし）
- ロス耐性: `agreement_drop` が D+DZ よりどれだけ小さいか、d = 1 と d = 3 の差（フィードバックが遅いほど伝搬が長い）
- 送信量: 参照が古くなると差分が大きくなるので、ロスありでは `bytes_per_frame` が D+DZ より少し増えうる。強制 I フレームの分も含まれる
- `gop_curve.csv` で、ロス後の劣化が d フレームで回復しているか

---

## レシピ 6: Shiba et al. の結果を再現できるか確かめる

**問い**: この基盤の時系列符号化（`splitcomp.temporal`）で、論文の表・図と同じ結果になるか？
（レシピ 5 のパケットロス実験の前提が正しいかの確認）

1. 動作確認（2 条件 × 2 クリップ、数分）:

   ```bash
   python scripts/verify_shiba.py --target RN-L2 --target Eff-S3 --num-clips 2 --quick
   ```

2. 本番（論文と同じ 12 条件 × 30 クリップ × 1,920 通り）:

   ```bash
   python scripts/verify_shiba.py
   ```

   - 時間の大半は tail の計算。M 系の Mac（MPS）で 8 時間ほど、CUDA の GPU ならずっと速い。
     target が終わるたびに `raw/` に保存するので、止まっても `--out <実行フォルダ>` で続きから計算できる
   - モデルごとに別のマシンで計算して、`raw/*.json` を 1 つのフォルダに集め、`--report-only` でまとめてもよい
   - `--quick`（θ = 0 だけ）にすると約 1/4 の時間になる。Table 8・9 と図はほぼ全て作れるが、480 条件の数え上げは「—」になる

   研究室の Slurm クラスタでは、`scripts/slurm/verify_shiba.sbatch` で 12 条件を別々のジョブとして並列に計算できる
   （準備はレシピ 5 の「研究室の Slurm クラスタで動かす」の 1 と同じ。`prefetch.py` で動画 30 本と 4 モデルの重みがそろう）:

   ```bash
   mkdir -p runs/slurm
   EXTRA="--num-clips 2 --quick" sbatch --array=4 scripts/slurm/verify_shiba.sbatch   # 動作確認（RN-L2 だけ）
   sbatch scripts/slurm/verify_shiba.sbatch                                         # 本番（配列 0-11）
   squeue -u $USER
   python scripts/verify_shiba.py --report-only runs/verify_shiba_<配列ジョブ ID>     # 全て終わったら報告書
   ```

   - 全てのジョブが同じ `runs/verify_shiba_<配列ジョブ ID>/raw/` に書く。報告書は最後に 1 回だけ作る（ジョブは `--no-report`）
   - 失敗した条件だけ投げ直すときは、同じフォルダを `OUT=` で指定する:
     `OUT=runs/verify_shiba_<配列ジョブ ID> sbatch --array=3,7 scripts/slurm/verify_shiba.sbatch`
   - 配列の番号: 0-2 EfficientNetV2-S、3-5 ResNet-50、6-8 VGG-16、9-11 MobileNetV2（それぞれ浅い順）

3. `report/report.md` を見る。

   - 「判定のまとめ」: 表・図ごとの ✓（許容幅以内）/ △（2 倍以内）/ ✗ の数。許容幅は `verify_shiba.py` の `TOL`
   - 「本文の主張の検証」: 「472/480 条件で DPCM ≥ FD」「D+DZ が 31/36 で有利」「α < 2%」などが本実装でも成り立つか
   - 各表: 論文 / 本実装 / 差 / 判定。`table_scatter.png` で全体の傾向が分かる

判定の読み方:

- 論文の 30 クリップのリストは公開されていない（`PAPER_CLIPS` は推測）。特徴の統計（Table 7）や一致率は、
  クリップの違いだけで 0.0x 程度はずれる。**個々の数値の一致より、大小関係と傾向（本文の主張）が再現できているか**を重視する
- Fig. 4（15fps）と Fig. 8 の論文の値は図からの目での読み取りなので、許容幅を広くしている
- 論文の書き写し（`paper_values.py`）は `tests/test_paper_values.py` で論文内の整合性を確かめている（Δ 列が差になっているか、Table 5/6 が Table 7 から計算できるか など）

既知の差: 「診断」の節を見る。論文の記述どおり I フレームも P フレームも P99 でクリップすると、
ReLU のモデルでは max|f| が P99 の 10 倍以上あり、I フレームで切られた大きな値の誤差が DPCM では少しずつしか戻らず、
DwD では GoP の終わりまで残る。そのため DPCM / DwD の一致率が論文より低く出ることがある（DnD・FD には効かない）。
`clip="max"` にすると DPCM は DnD とほぼ同じになる（`diag_mse_vs_gop.png` と Table 11）

---

## 再現性のためのチェックリスト

- [ ] 実行ディレクトリの `config.yaml` と `meta.json` を論文の実験記録と紐づけたか
- [ ] `seed` を固定したか（`num_workers > 0` の場合はデータの読み込み順に注意）
- [ ] 時間計測で `warmup` を十分取ったか（特に GPU / MPS）
- [ ] 時間は「バッチあたり」であることを考慮したか（`batch_size`）
- [ ] 計測中に他の重い処理を動かしていないか
- [ ] git でコードをコミットしてから実験したか（`meta.json` に commit が記録される）

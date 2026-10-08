# 拡張方法

研究で独自のモデル・データ・圧縮手法を試すための拡張ポイントです。
登録したものは設定ファイルから名前で指定できるようになります。

> **登録コードを読み込ませる方法**: スクリプトは `splitcomp` しか import しないので、
> 独自の登録コードは `src/splitcomp/` 内のモジュール（例: `src/splitcomp/models/my_models.py`）に置き、
> 対応する `__init__.py` で import してください。
> もしくは scripts/*.py をコピーした自分用スクリプトの先頭で import しても構いません。

---

## 1. 独自モデルを追加する

```python
# src/splitcomp/models/my_models.py
import torch.nn as nn
from splitcomp.models import register_model

class MyNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.classifier = nn.Linear(32 * 16 * 16, num_classes)

    def forward(self, x):
        return self.classifier(self.features(x).flatten(1))

@register_model("my_net")
def build_my_net(pretrained: bool = False, num_classes: int = 10):
    model = MyNet(num_classes)
    if pretrained:
        model.load_state_dict(torch.load("weights/my_net.pt"))
    return model
```

```python
# src/splitcomp/models/__init__.py に追加
from . import my_models  # noqa: F401  （登録のために読み込む）
```

設定ファイルでは `model: {name: my_net, kwargs: {num_classes: 10}}` と書けます。

**torch.fx で追跡できない場合**: forward 内で「テンソルの値」に依存する分岐（`if x.sum() > 0:`）や、
Python の制御構文で形が変わる処理があるとグラフ化できません。対処法:

- その部分をサブモジュールに切り出し、`trace_depth` でそのモジュールを展開しないようにする
- `torch.fx.wrap` で関数を 1 ノードとして扱わせる
- `splitcomp.split.split_sequential(model, index)` で単純分割する（forward が子モジュールの順次適用の場合のみ）

---

## 2. 独自データセットを追加する

`(入力, 正解)` を返す `torch.utils.data.Dataset` を返す関数を登録します。

```python
from splitcomp.data import register_dataset

@register_dataset("my_data")
def build_my_data(root: str, split: str, image_size: int = 224, **kwargs):
    return MyDataset(root=root, train=(split == "train"), image_size=image_size)
```

設定: `data: {name: my_data, root: /path/to/data, split: val, kwargs: {image_size: 256}}`

正解がないタスクでは正解に `0` などのダミーを返してください。
その場合でも元モデルとの一致度（`out_mse`, `out_cosine`）で評価できます。

---

## 3. 独自の圧縮方式（TensorCodec）を追加する

テンソル 1 つ ⇔ バイト列 の変換を実装します。復元に必要な小さな情報は `meta`（JSON にできる値）に、
大きな配列はバイト列に入れてください。

```python
import numpy as np
import torch
from splitcomp.compression import TensorCodec, register_tensor_codec

@register_tensor_codec("topk_sparse")
class TopKSparse(TensorCodec):
    """絶対値の大きい上位 k% の要素だけを (インデックス, 値) で送る例。"""
    name = "topk_sparse"

    def __init__(self, cfg):          # CodecConfig がそのまま渡される
        self.ratio = cfg.bits / 100   # 例として bits 欄を「残す割合[%]」に流用

    def encode(self, t):
        flat = t.detach().float().reshape(-1).cpu()
        k = max(1, int(flat.numel() * self.ratio))
        idx = flat.abs().topk(k).indices.to(torch.int32)
        vals = flat[idx.long()].to(torch.float16)
        data = idx.numpy().tobytes() + vals.numpy().tobytes()
        return data, {"shape": list(t.shape), "k": k}

    def decode(self, data, meta, device):
        k = meta["k"]
        idx = torch.from_numpy(np.frombuffer(data, dtype=np.int32, count=k).copy()).long()
        vals = torch.from_numpy(np.frombuffer(data, dtype=np.float16, offset=4 * k).copy()).float()
        out = torch.zeros(int(np.prod(meta["shape"])))
        out[idx] = vals
        return out.reshape(meta["shape"]).to(device)
```

設定: `codecs: [{tensor: topk_sparse, bits: 10, compress: [zlib], name: top10%}]`

`CodecConfig` に新しい項目が必要なら `src/splitcomp/config.py` の `CodecConfig` にフィールドを追加してください。

## 4. 独自の可逆圧縮（ByteCodec）を追加する

```python
import zstandard
from splitcomp.compression import ByteCodec, register_byte_codec

@register_byte_codec("zstd")
class ZstdCodec(ByteCodec):
    name = "zstd"
    def __init__(self, cfg):
        self.c = zstandard.ZstdCompressor(level=cfg.level or 3)
        self.d = zstandard.ZstdDecompressor()
    def compress(self, data):   return self.c.compress(data)
    def decompress(self, data): return self.d.decompress(data)
```

設定: `compress: [zstd]`

---

## 5. 独自の評価指標を追加する

`Evaluator` に `metric_fns` を渡します。関数は **バッチ内の合計値** を返してください（平均は自動で計算）。

```python
from splitcomp.metrics import Evaluator

def mean_iou_sum(output, target):
    ...  # バッチ内の各サンプルの IoU の合計を返す
    return iou_per_sample.sum().item()

evaluator = Evaluator(task="none", metric_fns={"miou": mean_iou_sum})
```

スクリプトで使う場合は、`scripts/sweep.py` の `Evaluator()` を上のように書き換えたコピーを作ってください。

## 6. 独自の損失で学習する

`BottleneckTrainer` に `task_loss_fn` を渡すと、`cfg.task_loss` の代わりに使われます。

```python
def my_loss(student_output, target):
    return torch.nn.functional.smooth_l1_loss(student_output, target)

trainer = BottleneckTrainer(cfg, teacher, head, tail, encoder, decoder, device, task_loss_fn=my_loss)
```

（`cfg.task_weight` を 0 より大きくするのを忘れずに。）

エンコーダ / デコーダの構造そのものを変えたい場合は、`build_bottleneck` の代わりに
任意の `nn.Module` を `BottleneckTrainer` に渡せます。
チェックポイントから復元できるようにするには `compression/bottleneck.py` の
`build_bottleneck` に分岐を追加するのが簡単です。

## 7. 通信モデルを変える

```python
from splitcomp.transport import SimulatedChannel

class LossyChannel(SimulatedChannel):
    """パケット損失による再送を期待値で考慮する例。"""
    def __init__(self, *args, loss_rate=0.01, **kw):
        super().__init__(*args, **kw)
        self.loss_rate = loss_rate

    def uplink_ms(self, nbytes):
        return super().uplink_ms(nbytes) / (1 - self.loss_rate)
```

## 8. Backend を追加する（gRPC、別プロセス、実機など）

`splitcomp.split.Backend` を継承し、`infer(payload) -> (出力, 計測値 dict)` を実装すれば
`SplitPipeline` からそのまま使えます。計測値には `decode_ms`, `tail_ms`, `result_bytes`, `comm_ms` を含めてください。

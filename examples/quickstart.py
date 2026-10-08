"""Python API から直接使う最小例。

    python examples/quickstart.py

スクリプト（scripts/*.py）を使わず、自分のコードやノートブックから
分割・圧縮・通信シミュレーションを組み合わせる方法を示す。
"""

import torch

from splitcomp.compression import build_codec
from splitcomp.config import CodecConfig
from splitcomp.models import build_model
from splitcomp.split import LocalBackend, SplitPipeline, Splitter
from splitcomp.transport import SimulatedChannel

# 1. モデルを用意する（torchvision のモデル名で指定。独自モデルをそのまま渡してもよい）
model = build_model("resnet18")
x = torch.randn(1, 3, 224, 224)

# 2. 分割点の候補を見る（trace_depth=1 で layer1, layer2 ... の粗い単位になる）
splitter = Splitter(model, trace_depth=1)
for p in splitter.split_points("clean"):
    print(f"  分割点 {p.index:2d}: {p.name}")

# 3. layer2 の直後で head（端末側）と tail（サーバ側）に分割する
head, tail = splitter.split("layer2")
z = head(x)  # 送信するテンソルの tuple
print("送信テンソルの形:", [tuple(t.shape) for t in z])
assert torch.allclose(tail(*z), model(x), atol=1e-5)  # 分割しても出力は変わらない

# 4. 圧縮方式（8bit 量子化 + zlib）と通信路（10 Mbps, RTT 50 ms）を決める
codec = build_codec(CodecConfig(tensor="quant", bits=8, compress=["zlib"]))
channel = SimulatedChannel(bandwidth_mbps=10, rtt_ms=50)

# 5. 分割推論を実行し、時間とデータ量の内訳を見る
device = torch.device("cpu")
pipeline = SplitPipeline(head, codec, LocalBackend(tail, codec, device, channel), head_device=device)
output, stats = pipeline.run(x)

print(f"\n圧縮方式: {codec.name} / 通信路: {channel.name}")
for key in ["head_ms", "encode_ms", "payload_bytes", "raw_bytes", "uplink_ms", "downlink_ms", "decode_ms", "tail_ms", "total_ms"]:
    print(f"  {key:<14} {stats[key]:10.2f}")
print(f"  圧縮率         {stats['raw_bytes'] / stats['payload_bytes']:10.2f} 倍")
print("元モデルと予測クラスが一致:", bool((output.argmax(1) == model(x).argmax(1)).all()))

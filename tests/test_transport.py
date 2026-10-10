"""通信のテスト: シミュレーションの計算式と、localhost での実 TCP 分割推論。"""

import threading

import pytest
import torch

from splitcomp.compression import build_codec
from splitcomp.config import CodecConfig
from splitcomp.models import build_model
from splitcomp.split import LocalBackend, SplitPipeline, Splitter
from splitcomp.transport import SimulatedChannel, TailServer, TCPBackend, decode_output, encode_output


def test_simulated_channel_formula():
    ch = SimulatedChannel(bandwidth_mbps=8, rtt_ms=20)
    # 1,000,000 bytes = 8 Mbit → 8 Mbit/s で 1000 ms + RTT/2 = 10 ms
    assert ch.uplink_ms(1_000_000) == pytest.approx(1010)
    assert SimulatedChannel(8, 20, downlink_bandwidth_mbps=80).downlink_ms(1_000_000) == pytest.approx(110)


def test_output_roundtrip_nested():
    obj = [{"boxes": torch.randn(3, 4), "labels": torch.tensor([1, 2, 3])}, torch.randn(2)]
    spec, data = encode_output(obj)
    back = decode_output(spec, data)
    torch.testing.assert_close(back[0]["boxes"], obj[0]["boxes"])
    torch.testing.assert_close(back[0]["labels"], obj[0]["labels"])
    torch.testing.assert_close(back[1], obj[1])


def _setup(point="add"):
    model = build_model("toy_cnn")
    splitter = Splitter(model)
    head, tail = splitter.split(point)
    return model, head.eval(), tail.eval()


def test_local_pipeline_with_channel():
    model, head, tail = _setup()
    codec = build_codec(CodecConfig())
    ch = SimulatedChannel(10, 20)
    pipe = SplitPipeline(head, codec, LocalBackend(tail, codec, torch.device("cpu"), ch), torch.device("cpu"))
    x = torch.randn(2, 3, 16, 16)
    out, stats = pipe.run(x)
    torch.testing.assert_close(out, model(x))
    assert stats["comm_ms"] == pytest.approx(ch.uplink_ms(stats["payload_bytes"]) + ch.downlink_ms(stats["result_bytes"]))
    expected_total = sum(stats[k] for k in ["head_ms", "encode_ms", "decode_ms", "tail_ms", "comm_ms"])
    assert stats["total_ms"] == pytest.approx(expected_total)


def test_tcp_roundtrip_matches_local():
    model, head, tail = _setup()
    codec = build_codec(CodecConfig(compress=["zlib"]))
    server = TailServer(tail, codec, torch.device("cpu"), "127.0.0.1", 0, fingerprint="abc", verbose=False)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.address
        with pytest.raises(ConnectionError):  # 設定が違うクライアントは拒否される
            TCPBackend(host, port, fingerprint="different")
        with TCPBackend(host, port, fingerprint="abc") as backend:
            pipe = SplitPipeline(head, codec, backend, torch.device("cpu"))
            for _ in range(3):
                x = torch.randn(2, 3, 16, 16)
                out, stats = pipe.run(x)
                torch.testing.assert_close(out, model(x))
                assert stats["comm_ms"] >= 0 and stats["payload_bytes"] > 0
    finally:
        server.shutdown()

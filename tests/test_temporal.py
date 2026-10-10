"""時系列符号化（FD / DwD / DPCM）・パケット分割・損失モデルのテスト。"""

from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest
import torch

from splitcomp.compression.codecs import pack_bits
from splitcomp.config import DataConfig, LossConfig
from splitcomp.data import build_dataloader, build_dataset
from splitcomp.temporal import (
    EncodedFrame,
    TemporalDecoder,
    TemporalEncoder,
    build_loss_model,
    deadzone_quantize,
    frame_bytes,
    num_packets,
    packet_assignment,
    packetize,
    quantize,
    received_mask,
    symmetric_quantize,
)


def _video(T=30, shape=(8, 6, 6), rho=0.95, seed=0):
    """隣接フレームの相関が rho 程度になる疑似的な特徴列。"""
    g = torch.Generator().manual_seed(seed)
    f = [torch.randn(shape, generator=g)]
    for _ in range(T - 1):
        f.append(rho * f[-1] + (1 - rho**2) ** 0.5 * torch.randn(shape, generator=g))
    return torch.stack(f)


def _run(feats, scheme, bits=4, gop=30, p_quant="sym", lost_frames=(), concealment="hold"):
    enc = TemporalEncoder(scheme, bits, gop, p_quant)
    dec = TemporalDecoder(concealment)
    out = []
    for t, f in enumerate(feats):
        frame = enc.encode(f)
        mask = torch.zeros(f.numel(), dtype=torch.bool) if t in lost_frames else None
        out.append(dec.decode(frame, mask))
    return torch.stack(out)


# --- 量子化 -----------------------------------------------------------------


@pytest.mark.parametrize("bits", [2, 4, 8])
def test_index_fits_in_bits(bits):
    x = torch.randn(10000) * 3
    for mode in ("sym", "dz"):
        idx, _ = quantize(x, bits, mode)
        assert idx.min() >= -(1 << (bits - 1)) and idx.max() <= (1 << (bits - 1)) - 1


def test_deadzone_widens_zero_bin():
    q = 1.0
    x = torch.tensor([0.0, 0.5, 0.74, -0.74, 0.76, 1.74, 1.76, -0.76])
    np.testing.assert_array_equal(deadzone_quantize(x, 8, q).numpy(), [0, 0, 0, 0, 1, 1, 2, -1])
    # 対称量子化では 0.5 以上で 1 になる
    np.testing.assert_array_equal(symmetric_quantize(torch.tensor([0.49, 0.51]), 8, q).numpy(), [0, 1])


# --- 符号化（ロスなし）--------------------------------------------------------


def test_dpcm_has_no_drift_but_dwd_drifts():
    feats = _video(T=30)
    err_dpcm = (_run(feats, "dpcm", bits=2) - feats).pow(2).flatten(1).mean(1)
    err_dwd = (_run(feats, "dwd", bits=2) - feats).pow(2).flatten(1).mean(1)
    # DPCM: 各フレームの誤差は 1 回分の量子化誤差（≦ (q/2)^2 程度）で増えない
    assert err_dpcm[20:].mean() < 2 * err_dpcm[1:10].mean()
    # DwD: 誤差が蓄積する（GoP の後半ほど大きい）
    assert err_dwd[20:].mean() > 3 * err_dwd[1:5].mean()
    assert err_dwd[20:].mean() > err_dpcm[20:].mean()


def test_fd_frames_are_all_intra():
    enc = TemporalEncoder("fd", 4, gop=30)
    assert all(enc.encode(f).kind == "I" for f in _video(T=5))
    enc = TemporalEncoder("dpcm", 4, gop=3)
    assert [enc.encode(f).kind for f in _video(T=7)] == list("IPPIPPI")


# --- パケットロスの伝搬 -----------------------------------------------------------


def test_dpcm_loss_propagates_until_next_iframe():
    feats = _video(T=20)
    gop = 10
    clean = _run(feats, "dpcm", bits=8, gop=gop)
    lossy = _run(feats, "dpcm", bits=8, gop=gop, lost_frames={3})
    diff = (lossy - clean).abs().flatten(1).amax(1)
    assert torch.all(diff[:3] == 0)
    assert torch.all(diff[3:gop] > 0)  # ロス以降、GoP の終わりまでずれが残る
    assert torch.all(diff[gop:] == 0)  # 次の I フレームで元に戻る


def test_fd_loss_affects_only_lost_frame():
    feats = _video(T=10)
    clean = _run(feats, "fd", bits=8)
    lossy = _run(feats, "fd", bits=8, lost_frames={4})
    diff = (lossy - clean).abs().flatten(1).amax(1)
    assert diff[4] > 0
    assert torch.all(diff[torch.arange(10) != 4] == 0)
    # hold: ロスしたフレームは直前の復元値になる
    torch.testing.assert_close(lossy[4], clean[3])


def test_zero_concealment():
    dec = TemporalDecoder("zero")
    frame = EncodedFrame(0, "I", torch.tensor([1, 2, 3], dtype=torch.int32), 0.5, (3,))
    out = dec.decode(frame, torch.tensor([True, False, True]))
    torch.testing.assert_close(out, torch.tensor([0.5, 0.0, 1.5]))


# --- パケット分割 -------------------------------------------------------------


@pytest.mark.parametrize("interleave", [False, True])
@pytest.mark.parametrize("bits", [2, 4, 8])
def test_packetize_covers_all_elements(bits, interleave):
    n = 50_000
    packets = packetize(n, bits, 1400, interleave)
    assert len(packets) == num_packets(n, bits, 1400)
    np.testing.assert_array_equal(torch.cat(packets).sort().values.numpy(), np.arange(n))
    assert max(len(p) for p in packets) * bits <= (1400 - 17) * 8


def test_frame_bytes_matches_pack_bits():
    n, bits = 10_000, 3
    payload = sum(len(pack_bits(np.zeros(len(p), dtype=np.int64), bits)) for p in packetize(n, bits))
    assert frame_bytes(n, bits) == payload + 17 * num_packets(n, bits)


def test_received_mask_drops_packet_elements():
    n, bits = 5000, 8
    assign = packet_assignment(n, bits)
    lost = np.zeros(num_packets(n, bits), dtype=bool)
    lost[1] = True
    mask = received_mask(lost, assign)
    assert int((~mask).sum()) == int((assign == 1).sum())


# --- 損失モデル ---------------------------------------------------------------


def test_bernoulli_rate():
    m = build_loss_model(LossConfig("bernoulli", p=0.1), seed=0)
    lost = np.concatenate([m.drop(100) for _ in range(500)])
    assert abs(lost.mean() - 0.1) < 0.01


def _burst_lengths(lost):
    lengths, run = [], 0
    for v in lost:
        if v:
            run += 1
        elif run:
            lengths.append(run)
            run = 0
    return lengths


def test_gilbert_elliott_rate_and_burst():
    m = build_loss_model(LossConfig("gilbert_elliott", p=0.05, burst_len=4), seed=0)
    lost = np.concatenate([m.drop(100) for _ in range(2000)])
    assert abs(lost.mean() - 0.05) < 0.01
    assert abs(np.mean(_burst_lengths(lost)) - 4) < 0.4


def test_frame_loss_is_all_or_nothing():
    m = build_loss_model(LossConfig("frame", p=0.3), seed=0)
    frames = [m.drop(50) for _ in range(1000)]
    assert all(f.all() or not f.any() for f in frames)
    assert abs(np.mean([f[0] for f in frames]) - 0.3) < 0.05


def test_loss_model_is_reproducible():
    a = build_loss_model(LossConfig("gilbert_elliott", p=0.1), seed=3).drop(1000)
    b = build_loss_model(LossConfig("gilbert_elliott", p=0.1), seed=3).drop(1000)
    np.testing.assert_array_equal(a, b)


# --- data パッケージ -----------------------------------------------------------


def test_random_dataloader():
    cfg = DataConfig(name="random", num_samples=4, batch_size=2, kwargs={"input_shape": [3, 8, 8], "num_classes": 5})
    batches = list(build_dataloader(cfg))
    assert len(batches) == 2
    assert batches[0][0].shape == (2, 3, 8, 8)


def test_ucf101_frame_folders(tmp_path):
    from PIL import Image

    for cls in ("ApplyEyeMakeup", "Basketball"):
        d = tmp_path / "UCF-101" / cls / f"v_{cls}_g01_c01"
        d.mkdir(parents=True)
        for i in range(6):
            Image.fromarray(np.full((24, 32, 3), i * 10, dtype=np.uint8)).save(d / f"{i:04d}.jpg")
    cfg = DataConfig(name="ucf101", root=str(tmp_path), num_samples=None, kwargs={"num_clips": 2, "clip_len": 3, "frame_step": 2, "size": 16, "selection": "random", "download": False})
    ds = build_dataset(cfg)
    x, y = ds[0]
    assert x.shape == (3, 3, 16, 16)
    assert y in (0, 1)
    x2, _ = ds[0]  # 2 回目はキャッシュから読む
    torch.testing.assert_close(x, x2)


# --- UCF-101 のクリップ選択とダウンロード ------------------------------------------------


def test_paper_clips_selection():
    from collections import Counter

    from splitcomp.data.ucf101 import PAPER_CLIPS, UCF101_CLASSES, class_of

    assert len(UCF101_CLASSES) == 101
    classes = [class_of(v) for _, v in PAPER_CLIPS]
    assert len(set(classes)) == 30 and all(c in UCF101_CLASSES for c in classes)
    assert {"Typing", "Basketball"} <= set(classes)
    counts = Counter(cat for cat, _ in PAPER_CLIPS)
    assert counts == {"human_object": 6, "body_motion": 5, "human_human": 1, "instruments": 3, "sports": 15}
    assert len({cat for cat, _ in PAPER_CLIPS[:5]}) == 5  # 先頭 5 本で全カテゴリが揃う


class _RangeHandler(SimpleHTTPRequestHandler):
    """Range リクエストに対応した最小限の HTTP サーバ（テスト用）。"""

    requests = 0

    def log_message(self, *args):
        pass

    def do_HEAD(self):
        data = open(self.translate_path(self.path), "rb").read()
        self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()

    def do_GET(self):
        type(self).requests += 1
        data = open(self.translate_path(self.path), "rb").read()
        start, end = self.headers["Range"].split("=")[1].split("-")
        chunk = data[int(start) : int(end) + 1]
        self.send_response(206)
        self.send_header("Content-Length", str(len(chunk)))
        self.end_headers()
        self.wfile.write(chunk)


def test_fetch_zip_members_over_http(tmp_path):
    import functools
    import threading
    import zipfile

    from splitcomp.data.download import fetch_zip_members, list_remote_zip

    srv_dir = tmp_path / "srv"
    srv_dir.mkdir()
    with zipfile.ZipFile(srv_dir / "a.zip", "w", zipfile.ZIP_DEFLATED) as zf:
        for i in range(20):
            zf.writestr(f"UCF-101/C{i}/v_C{i}_g01_c01.avi", bytes([i]) * 100_000)
    server = ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(_RangeHandler, directory=str(srv_dir)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}/a.zip"
        assert len(list_remote_zip(url)) == 20
        members = ["UCF-101/C3/v_C3_g01_c01.avi", "UCF-101/C7/v_C7_g01_c01.avi"]
        # 1 つ目のミラーが壊れていても次のミラーで取得できる
        paths = fetch_zip_members([url + ".missing", url], members, tmp_path / "out", verbose=False)
        assert paths[0].read_bytes() == bytes([3]) * 100_000
        assert paths[1].read_bytes() == bytes([7]) * 100_000
        assert not (tmp_path / "out" / "UCF-101" / "C4").exists()  # 指定したものだけ
        before = _RangeHandler.requests
        fetch_zip_members(url, members, tmp_path / "out", verbose=False)  # 2 回目はダウンロードしない
        assert _RangeHandler.requests == before
    finally:
        server.shutdown()

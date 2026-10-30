"""時系列符号化（FD / DwD / DPCM）・パケット分割・損失モデルのテスト。"""

from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest
import torch

from splitcomp.compression.codecs import pack_bits
from splitcomp.config import DataConfig, LossConfig
from splitcomp.data import build_dataloader, build_dataset
from splitcomp.temporal import (
    AC_HEADER_BYTES,
    AC_TAIL_BYTES,
    EncodedFrame,
    TemporalDecoder,
    TemporalEncoder,
    adaptive_code_bits,
    build_loss_model,
    deadzone_quantize,
    entropy_layout,
    fec_parity,
    fec_recover,
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


def _run(feats, scheme, bits=4, gop=30, p_quant="sym", lost_frames=(), concealment="hold", streams=1):
    enc = TemporalEncoder(scheme, bits, gop, p_quant, streams=streams)
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


def test_full_range_is_symmetric():
    # 論文どおり [−R, R] に収める: 2-bit（q = R/2）で +0.9R は +R、−0.9R は −R に戻る
    R = 1.0
    q = 2 * R / 4
    x = torch.tensor([0.9, -0.9, 5.0, -5.0])
    np.testing.assert_array_equal(symmetric_quantize(x, 2, q, full_range=True).numpy(), [2, -2, 2, -2])
    # 既定（b ビットに収める）では +側は 1 で止まる
    np.testing.assert_array_equal(symmetric_quantize(x, 2, q).numpy(), [1, -2, 1, -2])
    idx, _ = quantize(torch.randn(10000) * 3, 4, "dz", full_range=True)
    assert idx.max() == 8 and idx.min() == -8


def test_deadzone_widens_zero_bin():
    q = 1.0
    x = torch.tensor([0.0, 0.5, 0.74, -0.74, 0.76, 1.74, 1.76, -0.76])
    np.testing.assert_array_equal(deadzone_quantize(x, 8, q).numpy(), [0, 0, 0, 0, 1, 1, 2, -1])
    # 対称量子化では 0.5 以上で 1 になる
    np.testing.assert_array_equal(symmetric_quantize(torch.tensor([0.49, 0.51]), 8, q).numpy(), [0, 1])


def test_quantize_clip_max_and_theta():
    x = torch.linspace(-10, 10, 2001)
    # max: R = max|x| なので、q = 2R / 2^b
    _, q = quantize(x, 4, "sym", clip="max")
    assert q == pytest.approx(2 * 10 / 16)
    _, q99 = quantize(x, 4, "sym")
    assert q99 < q
    # theta: |x| < θR の値は 0 になる。theta=0 は既定と同じ
    idx, q = quantize(x, 8, "sym", clip="max", theta=0.2)
    assert (idx[x.abs() < 2.0] == 0).all() and (idx[x.abs() > 2.1] != 0).all()
    assert torch.equal(quantize(x, 4, "dz", theta=0.0)[0], quantize(x, 4, "dz")[0])


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


# --- 多系列 DPCM（streams = n） ------------------------------------------------------


@pytest.mark.parametrize("streams", [2, 5])
def test_multi_stream_reference_structure(streams):
    """I フレームは GoP 先頭だけで、参照フレームは max(t − n, GoP 先頭)（GoP をまたがない）。"""
    gop = 12
    enc = TemporalEncoder("dpcm", 4, gop, "dz", streams=streams)
    frames = [enc.encode(f) for f in _video(T=2 * gop)]
    for fr in frames:
        pos = fr.t % gop
        if pos == 0:
            assert fr.kind == "I" and fr.ref is None
        else:
            assert fr.kind == "P"
            assert fr.ref == (fr.t - pos if pos < streams else fr.t - streams)


def test_single_stream_matches_previous_frame_reference():
    """streams = 1 は、直前のフレームを参照する通常の DPCM と同じ。"""
    feats = _video(T=20)
    frames = [f for f in (TemporalEncoder("dpcm", 4, 10, "dz").encode(x) for x in feats)]
    assert all(fr.ref == fr.t - 1 for fr in frames if fr.kind == "P")
    # ref を持たない（古い形式の）フレームでも、直前を参照して同じ結果になる
    dec_new, dec_old = TemporalDecoder(), TemporalDecoder()
    for fr in frames:
        old = EncodedFrame(fr.t, fr.kind, fr.idx, fr.q, fr.shape)
        torch.testing.assert_close(dec_new.decode(fr), dec_old.decode(old), rtol=0, atol=0)


@pytest.mark.parametrize("scheme", ["dpcm", "dwd"])
@pytest.mark.parametrize("streams", [2, 5])
def test_multi_stream_lossless_matches_encoder(scheme, streams):
    """ロスがなければ、クラウドの復元は端末が想定する f̂ と一致する（DPCM はドリフトもしない）。"""
    feats = _video(T=30)
    enc = TemporalEncoder(scheme, 4, 15, "dz", streams=streams)
    dec = TemporalDecoder()
    for f in feats:
        frame = enc.encode(f)
        torch.testing.assert_close(dec.decode(frame), enc.prev_hat, rtol=0, atol=0)
    if scheme == "dpcm":
        err = (_run(feats, "dpcm", bits=2, gop=30, p_quant="dz", streams=streams) - feats).pow(2).flatten(1).mean(1)
        assert err[20:].mean() < 2 * err[1:10].mean()


@pytest.mark.parametrize("streams", [2, 5])
def test_multi_stream_loss_stays_in_its_stream(streams):
    """P フレームを 1 枚失うと、同じ GoP の同じ系列（t ≡ 失ったフレーム mod n）の、後ろのフレームだけがずれる。"""
    feats = _video(T=24)
    gop, lost = 12, 3
    clean = _run(feats, "dpcm", bits=8, gop=gop, p_quant="dz", streams=streams)
    lossy = _run(feats, "dpcm", bits=8, gop=gop, p_quant="dz", streams=streams, lost_frames={lost})
    diff = (lossy - clean).abs().flatten(1).amax(1)
    t = torch.arange(len(feats))
    affected = (t >= lost) & (t < gop) & ((t - lost) % streams == 0)
    assert torch.all(diff[affected] > 0)
    assert torch.all(diff[~affected] == 0)
    # hold: 失ったフレームは参照フレームの復元値になる
    torch.testing.assert_close(lossy[lost], clean[max(lost - streams, 0)])


def test_multi_stream_iframe_loss_affects_whole_gop():
    """I フレームは全ての系列の起点なので、失うと GoP 全体がずれる。"""
    feats = _video(T=20)
    clean = _run(feats, "dpcm", bits=8, gop=10, p_quant="dz", streams=2)
    lossy = _run(feats, "dpcm", bits=8, gop=10, p_quant="dz", streams=2, lost_frames={10})
    diff = (lossy - clean).abs().flatten(1).amax(1)
    assert torch.all(diff[:10] == 0) and torch.all(diff[10:] > 0)


def test_multi_stream_widens_quantization_step():
    """参照が時間的に遠くなるので差分が大きくなり、同じビット数では量子化ステップ q（＝量子化誤差）が大きくなる。"""
    feats = _video(T=30, rho=0.95)

    def mean_step(streams):
        enc = TemporalEncoder("dpcm", 4, 30, "dz", streams=streams)
        return np.mean([enc.encode(f).q for f in feats][5:])

    assert mean_step(1) < mean_step(2) < mean_step(5)


def test_streams_must_be_positive():
    with pytest.raises(ValueError):
        TemporalEncoder("dpcm", 4, streams=0)


# --- PRS（NACK による参照フレーム選択） ------------------------------------------------


def _run_prs(feats, lost_frames=(), delay=1, gop=30, streams=1, bits=8, prs=True):
    """PRS で符号化・復号する。lost_frames のフレームは全要素を落とし、delay フレーム後に NACK が端末に届く。

    戻り値: (クラウドの復元, 送ったフレームのリスト, 端末が想定した f̂ のリスト, 符号化器)
    """
    enc = TemporalEncoder("dpcm", bits, gop, "dz", streams=streams, prs=prs)
    dec = TemporalDecoder()
    out, frames, hats = [], [], []
    for t, f in enumerate(feats):
        for s in lost_frames:
            if s + delay == t:
                enc.nack(s)
        frame = enc.encode(f)
        mask = torch.zeros(f.numel(), dtype=torch.bool) if t in lost_frames else None
        out.append(dec.decode(frame, mask))
        frames.append(frame)
        hats.append(enc.prev_hat)
    return torch.stack(out), frames, hats, enc


def test_prs_without_loss_matches_dpcm():
    """NACK が来なければ、PRS は通常の DPCM と全く同じ符号化になる。"""
    feats = _video(T=24)
    for streams in (1, 2):
        _, prs, _, _ = _run_prs(feats, gop=12, streams=streams)
        _, ref, _, _ = _run_prs(feats, gop=12, streams=streams, prs=False)
        for a, b in zip(prs, ref):
            assert (a.kind, a.ref, a.q) == (b.kind, b.ref, b.q)
            assert torch.equal(a.idx, b.idx)


@pytest.mark.parametrize("delay", [1, 3])
def test_prs_stops_propagation_after_feedback(delay):
    """P フレーム lost を失うと、NACK が届くまでの delay フレームだけずれ、その後は壊れていないフレームを参照して元に戻る。"""
    feats = _video(T=20)
    lost = 5
    out, frames, hats, enc = _run_prs(feats, lost_frames={lost}, delay=delay)
    diff = torch.stack([(o - h).abs().amax() for o, h in zip(out, hats)])
    assert torch.all(diff[:lost] == 0)
    assert torch.all(diff[lost : lost + delay] > 0)  # NACK が届くまでは伝搬する
    assert torch.all(diff[lost + delay :] == 0)  # 届いた後は端末とクラウドの f̂ が一致する
    # NACK の後の最初のフレームは、壊れたフレームより前の最新のフレームを参照する
    assert frames[lost + delay].ref == lost - 1
    assert enc.is_broken(lost) and enc.is_broken(lost + delay - 1) and not enc.is_broken(lost + delay)
    assert enc.forced_intra == 0


def test_prs_iframe_loss_forces_intra():
    """GoP 先頭の I フレームを失うと参照できる無事なフレームが無いので、NACK が届いたフレームを I フレームにする。"""
    feats = _video(T=10)
    out, frames, hats, enc = _run_prs(feats, lost_frames={0}, delay=2)
    assert frames[2].kind == "I" and frames[2].ref is None
    assert frames[3].ref == 2
    assert enc.forced_intra == 1
    diff = torch.stack([(o - h).abs().amax() for o, h in zip(out, hats)])
    assert torch.all(diff[:2] > 0) and torch.all(diff[2:] == 0)


def test_prs_multi_stream_falls_back_to_newer_frame():
    """多系列で、通常の参照以前に無事なフレームが無いときは、より新しい無事なフレームを参照する（I フレームにはしない）。"""
    feats = _video(T=12)
    # streams=2: 1 → 0(I)。0 が壊れると 0 以前に無事なフレームが無いので、強制 I フレーム 2 の後は 2 を参照する
    _, frames, _, enc = _run_prs(feats, lost_frames={0}, delay=2, streams=2)
    assert frames[2].kind == "I"
    assert frames[3].ref == 2  # 通常なら 1（壊れている）だが、より新しい 2 を参照する
    assert frames[4].ref == 2
    assert enc.forced_intra == 1


def test_prs_ignores_nack_of_previous_gop():
    feats = _video(T=12)
    out, frames, hats, _ = _run_prs(feats, lost_frames={5}, delay=2, gop=6)
    assert [f.kind for f in frames[6:]] == list("IPPPPP")
    assert all(f.ref == f.t - 1 for f in frames[7:])


def test_prs_requires_p_frames():
    with pytest.raises(ValueError):
        TemporalEncoder("fd", 4, prs=True)


# --- P フレームのステップ幅の固定（p_step="intra"） ------------------------------------------


def test_quantize_with_given_step():
    """step を指定すると、範囲を測らずにその q で量子化する（範囲外は b ビットの端に飽和する）。"""
    x = torch.tensor([0.0, 0.3, -0.6, 1.0, 100.0])
    idx, q = quantize(x, 4, "sym", step=0.5)
    assert q == 0.5
    assert idx.tolist() == [0, 1, -1, 2, 7]
    # dead-zone: |x| < 0.75q は 0
    assert quantize(x, 4, "dz", step=0.5)[0].tolist() == [0, 0, -1, 2, 7]
    # theta は R = q·2^b/2 = 4 を基準にする（|x| < 0.25·4 = 1 は 0）
    assert quantize(x, 4, "sym", theta=0.25, step=0.5)[0].tolist() == [0, 0, 0, 2, 7]
    with pytest.raises(ValueError):
        quantize(x, 4, "sym", step=0.0)


@pytest.mark.parametrize("scale", [0.5, 1.0, 2.0])
def test_intra_step_fixes_p_frame_step(scale):
    """p_step="intra" の P フレームの q は I フレームの q × 倍率で、ロスなしなら端末とクラウドの f̂ が一致する。"""
    feats = _video(T=12)
    enc = TemporalEncoder("dpcm", 4, 6, "dz", p_step="intra", p_step_scale=scale)
    dec = TemporalDecoder()
    frames = []
    for f in feats:
        frame = enc.encode(f)
        assert torch.equal(dec.decode(frame), enc.prev_hat)
        frames.append(frame)
    for i in (0, 6):
        assert frames[i].kind == "I"
        assert all(f.kind == "P" and f.q == frames[i].q * scale for f in frames[i + 1 : i + 6])


def test_intra_step_reduces_entropy_of_small_residuals():
    """差分が小さいとき、q を固定するとインデックスが 0 付近に集まってエントロピーが下がる（adaptive では下がらない）。"""
    from splitcomp.temporal import entropy_bits

    feats = _video(T=10, shape=(16, 8, 8), rho=0.99)
    ent = {}
    for p_step in ("adaptive", "intra"):
        enc = TemporalEncoder("dpcm", 4, 30, "dz", p_step=p_step)
        ent[p_step] = [entropy_bits(enc.encode(f).idx) for f in feats]
    assert ent["intra"][0] == ent["adaptive"][0]  # I フレームは同じ
    assert max(ent["intra"][1:]) < min(ent["adaptive"][1:])


def test_adaptive_step_is_default():
    """p_step の既定（adaptive）は、指定しないときと全く同じ符号化になる。"""
    feats = _video(T=8)
    a = TemporalEncoder("dpcm", 4, 30, "dz")
    b = TemporalEncoder("dpcm", 4, 30, "dz", p_step="adaptive", p_step_scale=3.0)
    for f in feats:
        fa, fb = a.encode(f), b.encode(f)
        assert fa.q == fb.q and torch.equal(fa.idx, fb.idx)


def test_intra_step_follows_forced_intra():
    """PRS で P フレームの代わりに I フレームにしたら、以降の P フレームはそのフレームの q を使う。"""
    feats = _video(T=8)
    enc = TemporalEncoder("dpcm", 4, 30, "dz", prs=True, p_step="intra")
    frames = []
    for t, f in enumerate(feats):
        if t == 2:
            enc.nack(0)  # GoP 先頭の I フレームが壊れた
        frames.append(enc.encode(f))
    assert frames[2].kind == "I" and frames[2].q != frames[0].q
    assert frames[1].q == frames[0].q
    assert all(f.q == frames[2].q for f in frames[3:])


def test_invalid_p_step():
    with pytest.raises(KeyError):
        TemporalEncoder("dpcm", 4, p_step="foo")
    with pytest.raises(ValueError):
        TemporalEncoder("dpcm", 4, p_step="intra", p_step_scale=0.0)


def test_sweep_encoding_key_distinguishes_step_settings():
    """sweep の符号化の使い回しのキーは、clip / p_step / p_step_scale の違いを区別する（PRS・FEC・restorer は区別しない）。"""
    from splitcomp.config import TemporalSchemeConfig

    sweep = _load_script("packet_loss_sweep")
    base = TemporalSchemeConfig("D+DZ", "dpcm", "dz")
    key = sweep.encoding_key(base, 4)
    assert sweep.encoding_key(TemporalSchemeConfig("D+DZ+PRS", "dpcm", "dz", prs=True), 4) == key
    assert sweep.encoding_key(TemporalSchemeConfig("D+DZ+FEC10", "dpcm", "dz", fec=0.1), 4) == key
    others = [
        TemporalSchemeConfig("D+DZ+Cmax", "dpcm", "dz", clip="max"),
        TemporalSchemeConfig("D+DZ+QI", "dpcm", "dz", p_step="intra"),
        TemporalSchemeConfig("D+DZ+QI2", "dpcm", "dz", p_step="intra", p_step_scale=2.0),
    ]
    keys = {sweep.encoding_key(s, 4) for s in others}
    assert len(keys) == 3 and key not in keys
    assert sweep.encoding_key(base, 8) != key


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


def test_adaptive_code_bits_matches_sequential():
    """閉じた式の符号長が、適応 Laplace モデルで 1 記号ずつ −log2 p を足したものと一致する。"""
    K = 8
    sym = torch.randint(0, K, (300,), generator=torch.Generator().manual_seed(0))
    counts, total = np.ones(K), 0.0
    for s in sym.tolist():
        total -= np.log2(counts[s] / counts.sum())
        counts[s] += 1
    got = float(adaptive_code_bits(torch.bincount(sym, minlength=K)[None])[0])
    assert got == pytest.approx(total, rel=1e-9)


@pytest.mark.parametrize("interleave", [False, True])
@pytest.mark.parametrize("bits", [2, 4, 8])
def test_entropy_layout_fits_mtu_and_covers_all(bits, interleave):
    mtu, K = 200, 1 << bits
    g = torch.Generator().manual_seed(bits)
    idx = deadzone_quantize(torch.randn(20_000, generator=g) * 0.3, bits, 0.5)
    layout = entropy_layout(idx, bits, mtu, interleave)
    assert layout.assign.shape == idx.shape
    assert int(layout.assign.max()) + 1 == layout.num_packets
    sym = idx.long() + K // 2
    counts = torch.bincount(layout.assign * K + sym, minlength=layout.num_packets * K).view(-1, K)
    assert bool((counts.sum(1) > 0).all())  # 空のパケットがない
    sizes = AC_HEADER_BYTES + torch.ceil(adaptive_code_bits(counts) / 8).long() + AC_TAIL_BYTES
    assert int(sizes.max()) <= mtu
    assert int(sizes.sum()) == layout.bytes
    assert int(sizes.max()) == layout.max_bytes  # FEC のパリティパケットの長さ
    if not interleave:
        assert bool((layout.assign[1:] >= layout.assign[:-1]).all())  # 先頭から順に詰まっている


@pytest.mark.parametrize("interleave", [False, True])
def test_entropy_layout_shrinks_sparse_frames(interleave):
    """0 ばかりの P フレームは、固定長より大幅に少ないパケット数・バイト数になる。"""
    n, bits = 50_000, 4
    idx = torch.zeros(n, dtype=torch.int32)
    idx[::97] = 1
    ac = entropy_layout(idx, bits, 1400, interleave)
    assert ac.num_packets < num_packets(n, bits) / 5
    assert ac.bytes < frame_bytes(n, bits) / 5


def test_fec_parity_count():
    assert fec_parity(20, 0.1) == 2
    assert fec_parity(20, 0.25) == 5
    assert fec_parity(3, 0.1) == 1  # 切り上げなので少なくとも 1 個
    assert fec_parity(20, 0.0) == 0
    assert fec_parity(0, 0.5) == 0


def test_fec_recover_up_to_parity_count():
    """k+m 個のうち落ちた数が m 以下なら全て復元し、超えたらデータパケットのロスがそのまま残る。"""
    k, m = 6, 2
    lost = np.zeros(k + m, dtype=bool)
    lost[[1, 7]] = True  # データ 1 個 + パリティ 1 個
    assert not fec_recover(lost, k).any()
    lost[[1, 4]] = True  # データ 2 個（パリティが 2 個あるので復元できる）
    lost[7] = False
    assert not fec_recover(lost, k).any()
    lost[6] = True  # 3 個落ちたので復元できない
    np.testing.assert_array_equal(fec_recover(lost, k), lost[:k])
    # パリティなしなら素通し
    lost = np.array([False, True, False])
    np.testing.assert_array_equal(fec_recover(lost, 3), lost)


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


# --- スクリプト: target の選択と結果のマージ ------------------------------------------


def _load_script(name):
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_select_targets():
    from splitcomp.config import ModelConfig, SplitTarget

    sweep = _load_script("packet_loss_sweep")
    targets = [
        SplitTarget(ModelConfig(name="resnet50"), "layer2"),
        SplitTarget(ModelConfig(name="resnet50"), "layer3"),
        SplitTarget(ModelConfig(name="vgg16"), "features_16"),
    ]
    assert sweep.select_targets(targets, None) == targets
    assert sweep.select_targets(targets, ["resnet50"]) == targets[:2]
    assert sweep.select_targets(targets, ["resnet50/layer3"]) == [targets[1]]
    assert sweep.select_targets(targets, ["2", "resnet50/layer2"]) == [targets[0], targets[2]]  # 設定ファイルの順
    with pytest.raises(SystemExit, match="候補"):
        sweep.select_targets(targets, ["mobilenet_v2"])


def test_select_schemes():
    from splitcomp.config import TemporalSchemeConfig

    sweep = _load_script("packet_loss_sweep")
    schemes = [TemporalSchemeConfig(name) for name in ("F+Sym", "D+DZ", "D+DZ+R")]
    assert sweep.select_schemes(schemes, None) == schemes
    assert sweep.select_schemes(schemes, ["D+DZ+R", "F+Sym"]) == [schemes[0], schemes[2]]  # 設定ファイルの順
    with pytest.raises(SystemExit, match="候補"):
        sweep.select_schemes(schemes, ["D+Sym"])


def test_merge_results(tmp_path):
    import csv
    import json

    from splitcomp.metrics import ResultWriter

    merge = _load_script("merge_results")

    def make_run(name, dt, rows, finished=True):
        d = tmp_path / name
        d.mkdir()
        (d / "meta.json").write_text(json.dumps({"datetime": dt}))
        w = ResultWriter(d)
        g = ResultWriter(d, "gop_curve")
        for model, agreement in rows:
            w.write({"model": model, "split_point": "s", "agreement": agreement})
            g.write({"model": model, "split_point": "s", "gop_position": 0, "agreement": agreement})
        for writer in (w, g):
            if finished:
                writer.close()
            else:  # 時間切れで止まった実行は CSV が書かれず jsonl だけが残る
                writer._f.close()
        return d

    old = make_run("run_a", "2026-10-12T10:00:00", [("resnet50", 0.1), ("vgg16", 0.2)], finished=False)
    new = make_run("run_b", "2026-10-13T10:00:00", [("resnet50", 0.9)])
    out = tmp_path / "merged"
    source = merge.merge_runs([new, old], out)
    assert source == {("resnet50", "s"): new, ("vgg16", "s"): old}
    with open(out / "results.csv", newline="") as f:
        rows = {r["model"]: float(r["agreement"]) for r in csv.DictReader(f)}
    assert rows == {"resnet50": 0.9, "vgg16": 0.2}
    with open(out / "gop_curve.csv", newline="") as f:
        assert len(list(csv.DictReader(f))) == 2
    merge.merge_runs([new, old], out)  # 2 回実行しても行が重複しない
    with open(out / "results.csv", newline="") as f:
        assert len(list(csv.DictReader(f))) == 2


def test_result_writer_appends_to_existing(tmp_path):
    import csv

    from splitcomp.metrics import ResultWriter

    with ResultWriter(tmp_path) as w:
        w.write({"a": 1})
    with ResultWriter(tmp_path) as w:  # 再開: 前回分も CSV に入る
        w.write({"a": 2, "b": 3})
    with open(tmp_path / "results.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert [r["a"] for r in rows] == ["1", "2"]
    assert (tmp_path / "results.jsonl").read_text().count("\n") == 2


def test_checkpoint_roundtrip(tmp_path):
    from splitcomp.config import ModelConfig, SplitTarget

    sweep = _load_script("packet_loss_sweep")
    target = SplitTarget(ModelConfig(name="resnet50"), "layer2")
    other = SplitTarget(ModelConfig(name="vgg16"), "features_16")
    acc = sweep._Accumulator(4)
    acc.agree, acc.mse = 5, 0.25
    acc.pos_agree[1] = 3.0
    sweep.save_checkpoint(tmp_path, target, 2, 10, [8, 7, 7], {("D+DZ", 4, "no_loss", 0): acc})

    next_clip, shape, accs = sweep.load_checkpoint(tmp_path, target, 10, 4)
    assert (next_clip, shape) == (2, [8, 7, 7])
    loaded = accs[("D+DZ", 4, "no_loss", 0)]
    assert (loaded.agree, loaded.mse) == (5, 0.25)
    assert loaded.pos_agree.tolist() == [0.0, 3.0, 0.0, 0.0]
    assert sweep.load_checkpoint(tmp_path, other, 10, 4) is None  # 別の target のチェックポイントは使わない
    with pytest.raises(SystemExit, match="クリップ数"):
        sweep.load_checkpoint(tmp_path, target, 30, 4)


def test_finished_targets_and_prune(tmp_path):
    import json

    from splitcomp.config import LossConfig, PacketLossConfig

    sweep = _load_script("packet_loss_sweep")
    cfg = PacketLossConfig(bits=[2, 4], losses=[LossConfig(p=0.0), LossConfig(p=0.1)])  # 1 方式 × 2 × 2 = 4 行
    done_rows = [{"model": "resnet50", "split_point": "layer2", "i": i} for i in range(4)]
    partial_rows = [{"model": "vgg16", "split_point": "features_16", "i": i} for i in range(2)]
    # 旧バージョンの実行フォルダ（チェックポイントなし）が、書き込み途中の行を残して止まった状態
    lines = [json.dumps(r) for r in done_rows + partial_rows]
    (tmp_path / "results.jsonl").write_text("\n".join(lines) + '\n{"model": "vgg1')
    (tmp_path / "gop_curve.jsonl").write_text("\n".join(lines[:1] + lines[4:]) + "\n")

    done = sweep.finished_targets(tmp_path, cfg)
    assert done == {"resnet50/layer2"}
    sweep.prune_unfinished(tmp_path, done)
    assert sweep._read_jsonl(tmp_path / "results.jsonl") == done_rows
    assert sweep._read_jsonl(tmp_path / "gop_curve.jsonl") == done_rows[:1]


def test_resume_config(tmp_path):
    import yaml

    from splitcomp.config import PacketLossConfig, to_dict

    sweep = _load_script("packet_loss_sweep")
    with pytest.raises(SystemExit, match="config.yaml"):
        sweep.resume_config(tmp_path, [])
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(to_dict(PacketLossConfig(bits=[2, 8]))))
    cfg = sweep.resume_config(tmp_path, ["devices.head=cpu"])
    assert cfg.bits == [2, 8] and cfg.devices.head == "cpu"
    with pytest.raises(SystemExit, match="devices"):
        sweep.resume_config(tmp_path, ["bits=[4]"])


def _run_clip_reference(sweep, feats, ref_pred, tail, cfg, scheme, bits, loss_cfg, seed, clip, acc):
    """高速化する前の run_clip（損失条件ごとに符号化し直す版）。新しい encode_clip / decode_clip との一致確認用。"""
    import numpy as np

    from splitcomp.temporal import (
        TemporalDecoder,
        TemporalEncoder,
        build_loss_model,
        entropy_bits,
        entropy_layout,
        received_mask,
    )

    T = feats.shape[0]
    loss = build_loss_model(loss_cfg, sweep._clip_seed(seed, clip))
    enc = TemporalEncoder(scheme.scheme, bits, cfg.gop, scheme.p_quant, streams=scheme.streams)
    dec = TemporalDecoder(cfg.concealment)
    recon = torch.empty_like(feats)
    for t in range(T):
        frame = enc.encode(feats[t])
        layout = entropy_layout(frame.idx, bits, cfg.mtu, cfg.interleave)
        lost = loss.drop(layout.num_packets)
        recon[t] = dec.decode(frame, received_mask(lost, layout.assign) if lost.any() else None)
        acc.lost_packets += int(lost.sum())
        acc.packets += layout.num_packets
        acc.bytes += layout.bytes
        acc.entropy += entropy_bits(frame.idx)
        acc.data_packets += layout.num_packets  # FEC なしでは、データパケットのロスがそのまま残る
        acc.residual_lost += int(lost.sum())
        acc.max_frame_bytes = max(acc.max_frame_bytes, layout.bytes)
        if frame.kind == "P":
            half = 1 << (bits - 1)
            acc.p_elements += frame.idx.numel()
            acc.p_clamped += int(((frame.idx == -half) | (frame.idx == half - 1)).sum())
            acc.p_frames += 1
            acc.p_bytes += layout.bytes
    pred = tail(recon).argmax(-1)
    agree = (pred == ref_pred).double()
    mse = (recon - feats).pow(2).flatten(1).mean(1).double()
    pos = torch.arange(T) % cfg.gop
    acc.agree += int(agree.sum())
    acc.frames += T
    acc.mse += float(mse.sum())
    np.add.at(acc.pos_agree, pos.numpy(), agree.numpy())
    np.add.at(acc.pos_mse, pos.numpy(), mse.numpy())
    np.add.at(acc.pos_count, pos.numpy(), 1)


@pytest.mark.parametrize("interleave", [False, True])
def test_encode_once_matches_reference(interleave):
    """符号化を損失条件で使い回しても、条件ごとに符号化し直す以前の版と集計値が完全に一致する。"""
    import numpy as np

    from splitcomp.config import LossConfig, PacketLossConfig, TemporalSchemeConfig

    sweep = _load_script("packet_loss_sweep")
    schemes = [
        TemporalSchemeConfig("F+Sym", "fd", "sym"),
        TemporalSchemeConfig("DwD+Sym", "dwd", "sym"),
        TemporalSchemeConfig("D+DZ", "dpcm", "dz"),
        TemporalSchemeConfig("D+DZ+S2", "dpcm", "dz", streams=2),
    ]
    losses = [
        LossConfig("bernoulli", 0.0),
        LossConfig("bernoulli", 0.1),
        LossConfig("gilbert_elliott", 0.1, burst_len=3),
        LossConfig("frame", 0.2),
    ]
    # MTU を小さくして 1 フレームを複数パケットに分ける
    cfg = PacketLossConfig(schemes=schemes, bits=[2, 8], gop=5, mtu=60, interleave=interleave, losses=losses, seeds=[0, 1])
    torch.manual_seed(0)
    tail = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(8 * 7 * 7, 10))
    for clip in range(2):
        feats = torch.randn(12, 8, 7, 7).cumsum(0)  # 時間方向に相関のある特徴
        ref_pred = tail(feats).argmax(-1)
        for scheme in schemes:
            for bits in cfg.bits:
                encoded = sweep.encode_clip(feats, cfg, scheme, bits)
                for loss_cfg in losses:
                    for seed in cfg.seeds:
                        new, old = sweep._Accumulator(cfg.gop), sweep._Accumulator(cfg.gop)
                        sweep.decode_clip(encoded, feats, ref_pred, tail, "cpu", cfg, loss_cfg, seed, clip, new)
                        _run_clip_reference(sweep, feats, ref_pred, tail, cfg, scheme, bits, loss_cfg, seed, clip, old)
                        for k, v in vars(old).items():
                            got = vars(new)[k]
                            if isinstance(v, np.ndarray):
                                assert np.array_equal(got, v), k
                            else:
                                assert got == v, k


def test_prs_clip_matches_dpcm_without_loss_and_helps_with_loss():
    """sweep の prs_clip: ロスなしなら D+DZ と集計値が一致し、ロスありでは D+DZ より特徴 MSE が小さく、NACK を数える。"""
    import numpy as np

    from splitcomp.config import LossConfig, PacketLossConfig, TemporalSchemeConfig

    sweep = _load_script("packet_loss_sweep")
    dz = TemporalSchemeConfig("D+DZ", "dpcm", "dz")
    prs = TemporalSchemeConfig("D+DZ+PRS", "dpcm", "dz", prs=True, feedback_delay=1)
    cfg = PacketLossConfig(schemes=[dz, prs], bits=[4], gop=15, mtu=60)
    torch.manual_seed(0)
    tail = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(8 * 7 * 7, 10))
    feats = _video(T=15, shape=(8, 7, 7))
    ref_pred = tail(feats).argmax(-1)

    def run(scheme, loss_cfg, reuse=False):
        acc = sweep._Accumulator(cfg.gop)
        encoded = sweep.encode_clip(feats, cfg, scheme, 4)
        if scheme.prs:
            sweep.prs_clip(feats, ref_pred, tail, "cpu", cfg, scheme, 4, loss_cfg, 0, 0, acc, encoded=encoded if reuse else None)
        else:
            sweep.decode_clip(encoded, feats, ref_pred, tail, "cpu", cfg, loss_cfg, 0, 0, acc)
        return acc

    no_loss = LossConfig("bernoulli", 0.0)
    a, b = run(dz, no_loss), run(prs, no_loss)
    for k, v in vars(a).items():
        got = vars(b)[k]
        assert np.array_equal(got, v) if isinstance(v, np.ndarray) else got == pytest.approx(v), k
    assert b.nacks == 0

    lossy = LossConfig("bernoulli", 0.1)
    a, b = run(dz, lossy), run(prs, lossy)
    assert b.nacks > 0 and a.nacks == 0
    assert b.mse < a.mse
    # NACK を受ける前のフレームでフィードバックなしの符号化を使い回しても、結果は完全に一致する
    for loss_cfg in (no_loss, lossy, LossConfig("gilbert_elliott", 0.2, burst_len=3)):
        c, d = run(prs, loss_cfg), run(prs, loss_cfg, reuse=True)
        for k, v in vars(c).items():
            got = vars(d)[k]
            assert np.array_equal(got, v) if isinstance(v, np.ndarray) else got == v, k


def test_fec_clip_costs_bytes_and_recovers_loss():
    """sweep の FEC: ロスなしなら D+DZ と精度が同じで送信量だけ増え、ロスありでは残るロスと特徴 MSE が減る。PRS とも組み合わせられる。"""
    import numpy as np

    from splitcomp.config import LossConfig, PacketLossConfig, TemporalSchemeConfig

    sweep = _load_script("packet_loss_sweep")
    dz = TemporalSchemeConfig("D+DZ", "dpcm", "dz")
    fec = TemporalSchemeConfig("D+DZ+FEC25", "dpcm", "dz", fec=0.25)
    prs = TemporalSchemeConfig("D+DZ+PRS", "dpcm", "dz", prs=True)
    prs_fec = TemporalSchemeConfig("D+DZ+PRS+FEC", "dpcm", "dz", prs=True, fec=0.25)
    cfg = PacketLossConfig(schemes=[dz, fec], bits=[4], gop=15, mtu=60)
    torch.manual_seed(0)
    tail = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(8 * 7 * 7, 10))
    feats = _video(T=15, shape=(8, 7, 7))
    ref_pred = tail(feats).argmax(-1)
    encoded = sweep.encode_clip(feats, cfg, dz, 4)

    def run(scheme, loss_cfg):
        acc = sweep._Accumulator(cfg.gop)
        if scheme.prs:
            sweep.prs_clip(feats, ref_pred, tail, "cpu", cfg, scheme, 4, loss_cfg, 0, 0, acc)
        else:
            sweep.decode_clip(encoded, feats, ref_pred, tail, "cpu", cfg, loss_cfg, 0, 0, acc, fec=scheme.fec)
        return acc

    no_loss = LossConfig("bernoulli", 0.0)
    a, b = run(dz, no_loss), run(fec, no_loss)
    assert a.agree == b.agree and a.mse == b.mse
    assert a.parity_packets == 0 and b.parity_packets > 0
    assert b.packets == a.packets + b.parity_packets
    assert b.bytes > a.bytes * 1.25 * 0.9  # パリティは一番長いデータパケットと同じ長さ
    assert np.array_equal(a.pos_mse, b.pos_mse)

    lossy = LossConfig("bernoulli", 0.1)
    a, b = run(dz, lossy), run(fec, lossy)
    assert a.residual_lost == a.lost_packets  # FEC なしでは通信路のロスがそのまま残る
    assert b.residual_lost / b.data_packets < a.residual_lost / a.data_packets
    assert b.mse < a.mse

    # PRS と組み合わせると、FEC で復元できなかったフレームだけ NACK を返す
    c, d = run(prs, lossy), run(prs_fec, lossy)
    assert 0 < c.nacks and d.nacks < c.nacks
    for k, v in vars(run(prs, no_loss)).items():
        if k not in ("bytes", "packets", "parity_packets", "p_bytes", "max_frame_bytes"):
            got = vars(run(prs_fec, no_loss))[k]
            assert np.array_equal(got, v) if isinstance(v, np.ndarray) else got == v, k


def _parallel_setup():
    """並列化のテスト用の小さな設定（PRS・FEC・多系列を含む。MTU を小さくして 1 フレームを複数パケットに分ける）。"""
    from splitcomp.config import LossConfig, PacketLossConfig, TemporalSchemeConfig

    schemes = [
        TemporalSchemeConfig("F+Sym", "fd", "sym"),
        TemporalSchemeConfig("D+DZ", "dpcm", "dz"),
        TemporalSchemeConfig("D+DZ+S2", "dpcm", "dz", streams=2),
        TemporalSchemeConfig("D+DZ+PRS", "dpcm", "dz", prs=True, feedback_delay=1),
        TemporalSchemeConfig("D+DZ+PRS3", "dpcm", "dz", prs=True, feedback_delay=3),
        TemporalSchemeConfig("D+DZ+FEC25", "dpcm", "dz", fec=0.25),
    ]
    losses = [
        LossConfig("bernoulli", 0.0),
        LossConfig("bernoulli", 0.1),
        LossConfig("gilbert_elliott", 0.1, burst_len=3),
        LossConfig("gilbert_elliott", 0.3, burst_len=3),
    ]
    cfg = PacketLossConfig(schemes=schemes, bits=[2, 4], gop=5, mtu=60, losses=losses, seeds=[0, 1])
    conditions = [(l, s) for l in losses for s in (cfg.seeds if l.p > 0 else cfg.seeds[:1])]
    torch.manual_seed(0)
    tail = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(8 * 7 * 7, 10)).eval()
    clips = []
    for clip in range(2):
        feats = _video(T=12, shape=(8, 7, 7), seed=clip)
        with torch.no_grad():
            clips.append((feats, tail(feats).argmax(-1)))
    return cfg, conditions, tail, clips


def _assert_accs_equal(got: dict, want: dict):
    import numpy as np

    assert list(got) == list(want)  # 挿入順（= results の行の順）も同じ
    for key, acc in want.items():
        for k, v in vars(acc).items():
            g = vars(got[key])[k]
            assert np.array_equal(g, v) if isinstance(v, np.ndarray) else g == v, (key, k)


def _accs_by_tasks(sweep, cfg, conditions, clips, run):
    """クリップの順・(方式, ビット数, 条件) の順に、run(clip, 方式の番号, bits, feats, ref_pred) の結果を足し込む。"""
    from collections import defaultdict

    accs = defaultdict(lambda: sweep._Accumulator(cfg.gop))
    for clip, (feats, ref_pred) in enumerate(clips):
        for si, scheme in enumerate(cfg.schemes):
            for bits in cfg.bits:
                for (loss_cfg, seed), res in zip(conditions, run(clip, si, bits, feats, ref_pred)):
                    sweep._apply(accs[(scheme.name, bits, loss_cfg.display_name(), seed)], res, cfg)
    return accs


def test_run_task_matches_per_condition():
    """_run_task（tail を全条件まとめて 1 回・PRS のパケット分割キャッシュあり）の結果を足し込むと、
    条件ごとに decode_clip / prs_clip を呼ぶ以前の版とビット単位で一致する。"""
    from collections import defaultdict

    sweep = _load_script("packet_loss_sweep")
    cfg, conditions, tail, clips = _parallel_setup()

    want = defaultdict(lambda: sweep._Accumulator(cfg.gop))
    for clip, (feats, ref_pred) in enumerate(clips):
        for scheme in cfg.schemes:
            for bits in cfg.bits:
                encoded = sweep.encode_clip(feats, cfg, scheme, bits)
                for loss_cfg, seed in conditions:
                    acc = want[(scheme.name, bits, loss_cfg.display_name(), seed)]
                    if scheme.prs:
                        sweep.prs_clip(feats, ref_pred, tail, "cpu", cfg, scheme, bits, loss_cfg, seed, clip, acc, encoded=encoded)
                    else:
                        sweep.decode_clip(encoded, feats, ref_pred, tail, "cpu", cfg, loss_cfg, seed, clip, acc, fec=scheme.fec)

    reused = frozenset({sweep.encoding_key(cfg.schemes[1], b) for b in cfg.bits})  # D+DZ / PRS / FEC は符号化が同じ
    sweep._init_worker(cfg, tail, torch.device("cpu"), {}, conditions, reused_encodings=reused)
    got = _accs_by_tasks(sweep, cfg, conditions, clips, sweep._run_task)
    _assert_accs_equal(got, want)
    assert sum(a.nacks for k, a in got.items() if "PRS" in k[0]) > 0  # PRS で NACK を受けた後のフレームも通っている


def test_layout_cache_reuses_identical_indices():
    """_layout_cached: 同じ量子化インデックスなら計算し直さずに同じ結果を返し、違うインデックスでは計算し直す。"""
    from splitcomp.config import PacketLossConfig

    sweep = _load_script("packet_loss_sweep")
    cfg = PacketLossConfig(mtu=60)
    idx = torch.randint(-8, 8, (400,))
    cache = {}
    layout, entropy = sweep._layout_cached(idx, 4, cfg, cache)
    again, entropy2 = sweep._layout_cached(idx.clone(), 4, cfg, cache)
    assert again is layout and entropy2 == entropy and len(cache) == 1
    other = idx.clone()
    other[0] = (other[0] + 1) % 8
    layout3, _ = sweep._layout_cached(other, 4, cfg, cache)
    assert layout3 is not layout and len(cache) == 2
    ref = sweep.entropy_layout(idx, 4, cfg.mtu, cfg.interleave)
    assert torch.equal(layout.assign, ref.assign) and (layout.num_packets, layout.bytes) == (ref.num_packets, ref.bytes)


def test_run_task_in_worker_processes_matches_in_process(monkeypatch):
    """ワーカープロセス（spawn・共有メモリで特徴を渡す）で _run_task を実行しても、1 プロセスで実行した結果と一致する。"""
    import multiprocessing
    import sys
    from concurrent.futures import ProcessPoolExecutor
    from pathlib import Path

    # spawn のワーカーが関数を import できるよう、scripts/ を sys.path に入れて普通のモジュールとして読む
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    monkeypatch.delitem(sys.modules, "packet_loss_sweep", raising=False)
    import packet_loss_sweep as sweep

    cfg, conditions, tail, clips = _parallel_setup()
    sweep._init_worker(cfg, tail, torch.device("cpu"), {}, conditions)
    want = _accs_by_tasks(sweep, cfg, conditions, clips, sweep._run_task)

    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(2, mp_context=ctx, initializer=sweep._init_worker, initargs=(cfg, tail, torch.device("cpu"), {}, conditions, 1)) as pool:
        for feats, ref_pred in clips:
            feats.share_memory_()
            ref_pred.share_memory_()
        got = _accs_by_tasks(sweep, cfg, conditions, clips, lambda *a: pool.submit(sweep._run_task, *a).result())
    _assert_accs_equal(got, want)


# --- ロス補修 net（D+DZ+R） --------------------------------------------------------


def _run_with_restorer(feats, restorer, lost_packets, bits=4, mtu=60):
    """D+DZ で符号化し、指定したパケットを落として復号する。lost_packets: {フレーム番号: 落とすパケット番号のリスト}"""
    n = feats[0].numel()
    assign = packet_assignment(n, bits, mtu)
    P = num_packets(n, bits, mtu)
    enc = TemporalEncoder("dpcm", bits, 30, "dz")
    dec = TemporalDecoder("hold", restorer=restorer)
    out = []
    for t, f in enumerate(feats):
        lost = np.zeros(P, dtype=bool)
        lost[lost_packets.get(t, [])] = True
        out.append(dec.decode(enc.encode(f), received_mask(lost, assign) if lost.any() else None))
    return torch.stack(out)


def test_untrained_restorer_equals_hold():
    """最後の層が 0 初期化なので、学習前の補修 net は hold と全く同じ結果になる。"""
    from splitcomp.temporal import LossRestorer

    feats = _video(T=8)
    lost = {0: [0], 3: [1], 4: [0, 2]}
    torch.testing.assert_close(_run_with_restorer(feats, LossRestorer().restore, lost), _run_with_restorer(feats, None, lost), rtol=0, atol=0)


@pytest.mark.parametrize("shape", [(8, 6, 6), (16, 7, 7), (3, 14, 10)])
def test_restorer_is_channel_agnostic_and_keeps_received(shape):
    """同じ net がチャネル数・空間サイズの違う特徴に使え、届いた要素は変えない。"""
    from splitcomp.temporal import LossRestorer

    torch.manual_seed(0)
    net = LossRestorer(hidden=8, dilations=(1, 2))
    torch.nn.init.normal_(net.out.weight, std=0.1)  # 0 初期化のままだと何も変わらないので、出力が出るようにする
    prev = torch.randn(shape)
    cur = prev + 0.1 * torch.randn(shape)
    received = torch.rand(shape) > 0.3
    cur = torch.where(received, cur, prev)
    out = net.restore(prev, cur, received)
    assert out.shape == prev.shape
    torch.testing.assert_close(out[received], cur[received], rtol=0, atol=0)
    assert not torch.equal(out[~received], cur[~received])


def test_restorer_called_only_for_lossy_p_frames():
    calls = []

    def restorer(prev, cur, received):
        calls.append(int((~received).sum()))
        return cur

    feats = _video(T=6)
    _run_with_restorer(feats, restorer, {0: [0], 2: [1]})  # フレーム 0 は I フレーム
    assert len(calls) == 1 and calls[0] > 0


def test_restorer_save_load(tmp_path):
    from splitcomp.temporal import LossRestorer, load_restorer, save_restorer

    torch.manual_seed(0)
    net = LossRestorer(hidden=8, dilations=(1, 3))
    torch.nn.init.normal_(net.out.weight, std=0.1)
    save_restorer(tmp_path / "r.pt", net, {"targets": ["x/y"]})
    loaded = load_restorer(tmp_path / "r.pt")
    assert loaded.meta == {"targets": ["x/y"]}
    prev, received = torch.randn(4, 5, 5), torch.rand(4, 5, 5) > 0.5
    torch.testing.assert_close(loaded.restore(prev, prev, received), net.restore(prev, prev, received))


def test_ucf101_groups_and_clips_per_class(tmp_path):
    from PIL import Image

    from splitcomp.data.ucf101 import group_of, parse_groups

    assert parse_groups("8-10,12") == {8, 9, 10, 12} and parse_groups([1, 3]) == {1, 3} and parse_groups(None) is None
    assert group_of("v_ApplyEyeMakeup_g08_c02") == 8
    for cls in ("ApplyEyeMakeup", "Basketball"):
        for g, c in [(1, 1), (8, 1), (9, 2), (10, 1)]:
            d = tmp_path / "UCF-101" / cls / f"v_{cls}_g{g:02d}_c{c:02d}"
            d.mkdir(parents=True)
            for i in range(2):
                Image.fromarray(np.full((8, 8, 3), i, dtype=np.uint8)).save(d / f"{i:04d}.jpg")
    kw = {"num_clips": 2, "clip_len": 2, "size": 8, "selection": "random", "download": False, "groups": "8-25", "clips_per_class": 2}
    ds = build_dataset(DataConfig(name="ucf101", root=str(tmp_path), num_samples=None, kwargs=kw))
    names = [ds.clip_name(i) for i in range(len(ds))]
    assert len(names) == 4 and len(set(names)) == 4
    assert all(group_of(n) >= 8 for n in names)


def test_train_restorer_step_reduces_loss():
    """学習スクリプトの simulate / train_step が動き、同じバッチで繰り返すと損失が下がる。"""
    from splitcomp.config import LossConfig, RestorerTrainConfig
    from splitcomp.temporal import LossRestorer

    script = _load_script("train_restorer")
    cfg = RestorerTrainConfig(mtu=60, gop=30, max_channels=None)
    torch.manual_seed(0)
    net = LossRestorer(hidden=8, dilations=(1, 2))
    feats = _video(T=10, shape=(8, 6, 6))
    samples, err, ref = script.simulate(feats, cfg, 4, LossConfig("bernoulli", 0.3), 0, net)
    assert samples and err > 0 and ref > 0
    _, err_hold, _ = script.simulate(feats, cfg, 4, LossConfig("bernoulli", 0.3), 0, None)
    assert err == pytest.approx(err_hold)  # 学習前は hold と同じ
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    first = script.train_step(net, opt, samples, cfg, "cpu")
    for _ in range(30):
        last = script.train_step(net, opt, samples, cfg, "cpu")
    assert last < first

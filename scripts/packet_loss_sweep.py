"""動画の時系列符号化（FD / DwD / DPCM、dead-zone 量子化）をパケットロス下で評価するシミュレーション実験。

参考: H. Shiba et al., "Delta Coding With Dead-Zone Quantization for Bandwidth-Efficient
Split Computing on Video Streams," IEEE Access, 2026（パケットロスは同論文 §VII-E-5 で今後の課題とされている）。

各 (モデル, 分割点) × 方式 × ビット数 × 損失モデル × シードについて、クリップごとに
    head → 時系列符号化（端末）→ パケット分割 → ロス → 復号・concealment（クラウド）→ tail
を行い、量子化もロスもない元モデルの出力との top-1 一致率（agreement）などを集計する。
端末はロスを知らない（フィードバックなし）ので、DPCM のロスは次の I フレームまで伝搬する。
方式に restorer（scripts/train_restorer.py で学習したロス補修 net）を指定すると、クラウド側で欠けた要素を補ってから
tail に渡す（例: D+DZ+R）。方式に streams: n を指定すると、P フレームが n フレーム前を参照する多系列 DPCM になり、
ロスの伝搬が n 本の系列のうち 1 本に閉じる（例: D+DZ+S2 / D+DZ+S5）。
方式に prs: true を指定すると PRS（Reference Picture Selection）になる: クラウドはパケットが落ちたフレームの番号だけを
NACK として返し（feedback_delay フレーム後に端末に届く）、端末はクラウド側で壊れていないフレームを参照して符号化する
（例: D+DZ+PRS / D+DZ+PRS3）。フィードバックは壊れたフレームの番号だけなので、端末→クラウドの送信量は増えない。
方式に fec: r を指定すると FEC（パケット単位の消失訂正, Reed-Solomon 相当）になる: 各フレームのデータ k パケットに
パリティ m = ceil(r·k) パケットを付けて送り、落ちた数が m 以下ならクラウドでフレーム全体を復元する（例: D+DZ+FEC10 / D+DZ+FEC25）。

端末側の符号化（I/P の判定・差分の計算・量子化・エントロピー・パケット分割）はロスによらないので、
クリップごとに (方式, ビット数) ごとに 1 回だけ行い、その結果を全ての損失条件・シードで使い回す。
ただし PRS は端末の符号化がフィードバック（= ロス）に依存するので、条件ごとに符号化と復号を交互に行う（:func:`prs_clip`）。
PRS でフィードバックによって符号化し直したフレームのパケット分割は、量子化インデックスが同じなら損失条件をまたいで使い回す。

並列化: 1 クリップの (方式, ビット数) 1 組を 1 タスクとして、ワーカープロセス（--workers。既定はジョブに割り当てられた
コア数 = os.sched_getaffinity）で並列に処理する（:func:`_run_task`）。各ワーカーは torch を 1 スレッドで動かし、
符号化 → 全損失条件の復号 → tail（全損失条件をまとめて 1 回のバッチ）までを行う。集計値への足し込みはメインプロセスが
クリップの順・(方式, ビット数, 条件) の順に行うので、結果は 1 プロセスで順に処理した場合と同じになる
（torch のスレッド数が違うと、tail や特徴 MSE の計算が浮動小数点の最後の桁でずれることはある）。
メインプロセスは、ワーカーが処理している間に次のクリップの head を計算して投入しておく（最大 :data:`LOOKAHEAD` クリップ先まで）。
tail のデバイスが cpu 以外（cuda / mps）のときと --workers 1 のときは、ワーカーを使わず 1 プロセスで順に処理する。
メインプロセスの torch の CPU スレッド数（head の計算用）は、ジョブに割り当てられたコア数を超えないように調整する。

出力（runs/<実験名>_<日時>/。--target で絞った場合は runs/<実験名>_<モデル名>_<日時>/）:
    results.csv    条件ごとの集計（agreement, 95% CI, 特徴 MSE, 実際の損失率, 送信量, エントロピー ...）
                   送信量（packets_per_frame, bytes_per_frame）は、量子化インデックスをパケットごとに独立した
                   適応算術符号で符号化したときの値（splitcomp.temporal.packet.entropy_layout）
    gop_curve.csv  GoP 内の位置 t ごとの agreement と特徴 MSE（誤差伝搬の様子）
    *.jsonl        同じ内容。モデル 1 つ分が終わるたびに追記されるので、途中で止まっても終わった分は残る
    checkpoint.pkl 実行中のモデルの途中経過（クリップ 1 本ごとに保存。モデルが終わると消える）

途中で止まった（セッションが切れた）実行は --resume で同じフォルダの続きから再開できる。
設定は実行フォルダの config.yaml から読むので --config は不要（devices.* だけは上書きできる）。
クリップごとの乱数は固定なので、再開しても中断なしで実行したのと同じ結果になる。

例:
    python scripts/packet_loss_sweep.py --config configs/packet_loss_ucf101.yaml
    python scripts/packet_loss_sweep.py --config configs/packet_loss_ucf101.yaml data.kwargs.num_clips=2 'seeds=[0]'

    # 新手法を試す段階: 条件を減らしたお試し設定（2 モデル・Gilbert-Elliott のみ）。--scheme で方式も絞れる
    python scripts/packet_loss_sweep.py --config configs/packet_loss_ucf101_quick.yaml --scheme D+DZ --scheme D+DZ+R

    # 多系列 DPCM（2 系列・5 系列）を通常の D+DZ と比べる
    python scripts/packet_loss_sweep.py --config configs/packet_loss_ucf101_quick.yaml --scheme D+DZ --scheme D+DZ+S2 --scheme D+DZ+S5

    # PRS（フィードバック遅延 1 / 3 フレーム）を通常の D+DZ と比べる
    python scripts/packet_loss_sweep.py --config configs/packet_loss_ucf101_quick.yaml --scheme D+DZ --scheme D+DZ+PRS --scheme D+DZ+PRS3

    # モデルごとに別々に実行する（時間制限のある計算機向け）。結果は scripts/merge_results.py でまとめる
    python scripts/packet_loss_sweep.py --config configs/packet_loss_ucf101.yaml --list-targets
    python scripts/packet_loss_sweep.py --config configs/packet_loss_ucf101.yaml --target resnet50

    # FEC（パリティ 10% / 25%）を通常の D+DZ と比べる
    python scripts/packet_loss_sweep.py --config configs/packet_loss_ucf101_quick.yaml --scheme D+DZ --scheme D+DZ+FEC10 --scheme D+DZ+FEC25

    # 途中で止まった実行を再開する
    python scripts/packet_loss_sweep.py --resume runs/packet_loss_ucf101_resnet50_20261013-120000

    # ワーカー数を指定する（1 にするとワーカーを使わず 1 プロセスで動かす）
    python scripts/packet_loss_sweep.py --config configs/packet_loss_ucf101_quick.yaml --workers 16
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import multiprocessing
import os
import pickle
import statistics
import time
from collections import Counter, defaultdict, deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from splitcomp.builder import build_split
from splitcomp.cli import parse_cli
from splitcomp.config import PacketLossConfig, TemporalSchemeConfig, load_config, to_dict
from splitcomp.data import build_dataset
from splitcomp.metrics import ResultWriter, make_run_dir, save_run_info
from splitcomp.temporal import (
    TemporalDecoder,
    TemporalEncoder,
    build_loss_model,
    PacketLayout,
    entropy_bits,
    entropy_layout,
    fec_parity,
    fec_recover,
    load_restorer,
    received_mask,
)
from splitcomp.utils import format_table, resolve_device, set_seed


class _Accumulator:
    """1 条件（方式 × ビット数 × 損失 × シード）の集計値。"""

    def __init__(self, gop: int):
        self.agree = 0
        self.frames = 0
        self.mse = 0.0
        self.lost_packets = 0
        self.packets = 0
        self.bytes = 0
        self.entropy = 0.0
        #: PRS: クラウドが返した NACK の数（フィードバックの量）
        self.nacks = 0
        #: PRS: 参照できる無事なフレームが無く I フレームにした回数
        self.forced_intra = 0
        #: FEC: 送ったパリティのパケット数
        self.parity_packets = 0
        #: FEC: 送ったデータパケット数と、FEC で復元できずに残ったデータパケットのロス数
        self.data_packets = 0
        self.residual_lost = 0
        #: P フレームの要素数と、そのうちインデックスが範囲の端 ±2^(b−1)（飽和）になった要素数
        self.p_elements = 0
        self.p_clamped = 0
        #: P フレームの数と、その送信バイト数（パリティ込み）
        self.p_frames = 0
        self.p_bytes = 0
        #: 1 フレームの送信バイト数（パリティ込み）の最大値
        self.max_frame_bytes = 0
        self.pos_agree = np.zeros(gop)
        self.pos_mse = np.zeros(gop)
        self.pos_count = np.zeros(gop)


CHECKPOINT = "checkpoint.pkl"


def _clip_seed(seed: int, clip: int) -> int:
    return seed * 100_003 + clip


class EncodedClip:
    """1 クリップを端末側で符号化した結果（損失条件によらないので、全ての損失条件で使い回す）。"""

    def __init__(self, frames: list, entropy: list[float], layouts: list[PacketLayout], bits: int = 0):
        #: フレームごとの送信内容（EncodedFrame）
        self.frames = frames
        #: フレームごとのエントロピー [bit/要素]
        self.entropy = entropy
        #: フレームごとのパケット分割（パケットごとの適応算術符号）
        self.layouts = layouts
        #: 量子化ビット数
        self.bits = bits


def make_encoder(cfg, scheme, bits, prs: bool = False) -> TemporalEncoder:
    """方式の設定から端末側の符号化器を作る（prs=True なら NACK を受けて参照を選ぶ）。"""
    return TemporalEncoder(
        scheme.scheme,
        bits,
        cfg.gop,
        scheme.p_quant,
        clip=scheme.clip,
        streams=scheme.streams,
        prs=prs,
        p_step=scheme.p_step,
        p_step_scale=scheme.p_step_scale,
    )


def encoding_key(scheme, bits) -> tuple:
    """端末側の符号化を決める設定（同じキーの方式は、フィードバックなしの符号化を使い回せる）。"""
    return (scheme.scheme, scheme.p_quant, scheme.streams, scheme.clip, scheme.p_step, scheme.p_step_scale, bits)


def encode_clip(feats, cfg, scheme, bits) -> EncodedClip:
    """1 クリップ（T フレーム）を端末側で符号化する（I/P の判定・差分の計算・量子化・パケット分割）。

    端末はロスを知らないので、結果は損失モデル・シードによらない。
    """
    enc = make_encoder(cfg, scheme, bits)
    frames = [enc.encode(feats[t]) for t in range(feats.shape[0])]
    layouts = [entropy_layout(f.idx, bits, cfg.mtu, cfg.interleave) for f in frames]
    return EncodedClip(frames, entropy=[entropy_bits(f.idx) for f in frames], layouts=layouts, bits=bits)


def _transmit(layout: PacketLayout, loss, fec: float, acc: _Accumulator, frame=None, bits: int = 0) -> np.ndarray:
    """1 フレームのパケット（と FEC のパリティ）を送ってロスを起こし、FEC で復元した後のデータパケットのロスを返す。

    データ k パケットの後にパリティ m = ceil(fec·k) パケットを続けて送る（fec = 0 なら m = 0 で、drop を k 個分呼ぶだけ）。
    パリティパケットの長さは一番長いデータパケットと同じとして送信量に足す。
    frame（EncodedFrame）と bits を渡すと、P フレームの飽和率・送信量と 1 フレームの最大送信量も集計する。
    """
    k = layout.num_packets
    m = fec_parity(k, fec)
    lost = loss.drop(k + m)
    lost_data = fec_recover(lost, k) if m else lost
    acc.lost_packets += int(lost.sum())  # 通信路での損失（パリティを含む）
    acc.packets += k + m
    nbytes = layout.bytes + m * layout.max_bytes
    acc.bytes += nbytes
    acc.max_frame_bytes = max(acc.max_frame_bytes, nbytes)
    if frame is not None and frame.kind == "P":
        half = 1 << (bits - 1)
        acc.p_elements += frame.idx.numel()
        acc.p_clamped += int(((frame.idx == -half) | (frame.idx == half - 1)).sum())
        acc.p_frames += 1
        acc.p_bytes += nbytes
    acc.parity_packets += m
    acc.data_packets += k
    acc.residual_lost += int(lost_data.sum())
    return lost_data


#: :class:`_Accumulator` の整数のフィールドのうち、クリップごとの値を足し込むもの（max_frame_bytes は最大値を取る）
_COUNT_FIELDS = (
    "lost_packets",
    "packets",
    "bytes",
    "nacks",
    "forced_intra",
    "parity_packets",
    "data_packets",
    "residual_lost",
    "p_elements",
    "p_clamped",
    "p_frames",
    "p_bytes",
)


class _CondResult:
    """1 クリップ × 1 条件（方式 × ビット数 × 損失 × シード）の結果。ワーカーが計算し、メインが :func:`_apply` で足し込む。

    浮動小数点の和の順序が 1 プロセスで順に処理した場合と同じになるよう、足す値をそのままの形で持つ。
    """

    def __init__(self, counts: dict, entropies: list, agree: np.ndarray, mse: np.ndarray, agree_sum: int, mse_sum: float):
        #: 整数の集計値（:data:`_COUNT_FIELDS` と max_frame_bytes）
        self.counts = counts
        #: フレームごとのエントロピー [bit/要素]（足す順）
        self.entropies = entropies
        #: フレームごとの top-1 一致（0/1）と特徴 MSE（長さ T の float64）
        self.agree = agree
        self.mse = mse
        #: agree.sum() と mse.sum()（torch で和を取った値。以前の版と同じ計算）
        self.agree_sum = agree_sum
        self.mse_sum = mse_sum


def _apply(acc: _Accumulator, res: _CondResult, cfg) -> None:
    """1 クリップ × 1 条件の結果を集計値に足し込む（以前の版の decode_clip / prs_clip / _score と同じ順序で足す）。"""
    for name in _COUNT_FIELDS:
        setattr(acc, name, getattr(acc, name) + res.counts[name])
    acc.max_frame_bytes = max(acc.max_frame_bytes, res.counts["max_frame_bytes"])
    for e in res.entropies:
        acc.entropy += e
    T = len(res.agree)
    pos = (torch.arange(T) % cfg.gop).numpy()
    acc.agree += res.agree_sum
    acc.frames += T
    acc.mse += res.mse_sum
    np.add.at(acc.pos_agree, pos, res.agree)
    np.add.at(acc.pos_mse, pos, res.mse)
    np.add.at(acc.pos_count, pos, 1)


@torch.no_grad()
def _evaluate(parts: list, feats, ref_pred, tail, tail_dev) -> list[_CondResult]:
    """復元した特徴を tail に通し、条件ごとの結果を返す。

    parts は条件ごとの (送信の集計値 _Accumulator, 復元した特徴 recon[T, ...], エントロピーの list)。
    全ての条件の recon をつなげて、tail を 1 回のバッチで呼ぶ。
    """
    T = feats.shape[0]
    recons = [recon for _, recon, _ in parts]
    batch = torch.cat(recons) if len(recons) > 1 else recons[0]
    preds = tail(batch.to(tail_dev)).argmax(-1).cpu()
    out = []
    for i, (part, recon, entropies) in enumerate(parts):
        agree = (preds[i * T : (i + 1) * T] == ref_pred).double()
        mse = (recon - feats).pow(2).flatten(1).mean(1).double()
        counts = {name: getattr(part, name) for name in _COUNT_FIELDS}
        counts["max_frame_bytes"] = part.max_frame_bytes
        out.append(_CondResult(counts, entropies, agree.numpy(), mse.numpy(), int(agree.sum()), float(mse.sum())))
    return out


@torch.no_grad()
def _decode_recon(encoded: EncodedClip, feats, cfg, loss_cfg, seed, clip, acc: _Accumulator, restorer=None, fec: float = 0.0):
    """符号化済みの 1 クリップにロスを起こして復号し、(復元した特徴, フレームごとのエントロピーの list) を返す。

    送信量などの整数の集計値は acc に加算する。損失の乱数はクリップとシードで決まり、フレームごとに drop を 1 回呼ぶ。
    """
    loss = build_loss_model(loss_cfg, _clip_seed(seed, clip))
    dec = TemporalDecoder(cfg.concealment, restorer=restorer.restore if restorer is not None else None)

    recon = torch.empty_like(feats)
    entropies = []
    for t, (frame, layout) in enumerate(zip(encoded.frames, encoded.layouts)):
        lost = _transmit(layout, loss, fec, acc, frame, encoded.bits)
        recon[t] = dec.decode(frame, received_mask(lost, layout.assign) if lost.any() else None)
        entropies.append(encoded.entropy[t])
    return recon, entropies


def decode_clip(
    encoded: EncodedClip, feats, ref_pred, tail, tail_dev, cfg, loss_cfg, seed, clip, acc: _Accumulator, restorer=None, fec: float = 0.0
) -> None:
    """符号化済みの 1 クリップにロスを起こして復号し、tail を通して acc に加算する。

    損失の乱数はクリップとシードで決まり、フレームごとに drop を 1 回呼ぶので、条件ごとに独立に再現できる。
    restorer（LossRestorer）を渡すと、P フレームでロスが起きたときに欠けた要素を補う。
    fec > 0 なら、各フレームにパリティを付けて送り、落ちた数がパリティ数以下ならクラウドで全て復元する（:func:`_transmit`）。
    """
    part = _Accumulator(cfg.gop)
    recon, entropies = _decode_recon(encoded, feats, cfg, loss_cfg, seed, clip, part, restorer, fec)
    (res,) = _evaluate([(part, recon, entropies)], feats, ref_pred, tail, tail_dev)
    _apply(acc, res, cfg)


def _layout_cached(idx: torch.Tensor, bits: int, cfg, cache: dict | None) -> tuple[PacketLayout, float]:
    """量子化インデックス idx の (パケット分割, エントロピー)。cache（dict）を渡すと、同じ idx の結果を使い回す。

    キーは idx の内容のハッシュで、当たったときは idx が本当に同じかを確かめてから使う（結果は使い回さない場合と完全に一致する）。
    """
    if cache is None:
        return entropy_layout(idx, bits, cfg.mtu, cfg.interleave), entropy_bits(idx)
    key = hashlib.blake2b(idx.numpy().tobytes(), digest_size=16).digest()
    hit = cache.get(key)
    if hit is not None and torch.equal(hit[0], idx):
        return hit[1], hit[2]
    layout, entropy = entropy_layout(idx, bits, cfg.mtu, cfg.interleave), entropy_bits(idx)
    cache[key] = (idx, layout, entropy)
    return layout, entropy


@torch.no_grad()
def _prs_recon(
    feats, cfg, scheme, bits, loss_cfg, seed, clip, acc: _Accumulator, restorer=None, encoded: EncodedClip | None = None, layout_cache: dict | None = None
):
    """PRS で 1 クリップを符号化・ロス・復号し、(復元した特徴, フレームごとのエントロピーの list) を返す（:func:`prs_clip` の本体）。"""
    T = feats.shape[0]
    loss = build_loss_model(loss_cfg, _clip_seed(seed, clip))
    enc = make_encoder(cfg, scheme, bits, prs=True)
    dec = TemporalDecoder(cfg.concealment, restorer=restorer.restore if restorer is not None else None)
    arriving: dict[int, list[int]] = defaultdict(list)  # 端末に届くフレーム番号 → その時に届く NACK（壊れたフレームの番号）

    recon = torch.empty_like(feats)
    entropies = []
    for t in range(T):
        for s in arriving.pop(t, []):
            enc.nack(s)
        frame = enc.encode(feats[t])
        if encoded is not None and not enc.nacked:
            # この GoP でまだ NACK を受けていないので、フィードバックなしの符号化と同じフレーム
            frame, layout, entropy = encoded.frames[t], encoded.layouts[t], encoded.entropy[t]
        else:
            layout, entropy = _layout_cached(frame.idx, bits, cfg, layout_cache)
        lost = _transmit(layout, loss, scheme.fec, acc, frame, bits)
        recon[t] = dec.decode(frame, received_mask(lost, layout.assign) if lost.any() else None)
        if lost.any():
            arriving[t + scheme.feedback_delay].append(t)
            acc.nacks += 1
        entropies.append(entropy)
    acc.forced_intra += enc.forced_intra
    return recon, entropies


def prs_clip(
    feats,
    ref_pred,
    tail,
    tail_dev,
    cfg,
    scheme,
    bits,
    loss_cfg,
    seed,
    clip,
    acc: _Accumulator,
    restorer=None,
    encoded: EncodedClip | None = None,
    layout_cache: dict | None = None,
) -> None:
    """PRS で 1 クリップを符号化・ロス・復号し、tail を通して acc に加算する（端末とクラウドを 1 フレームずつ交互に動かす閉ループ）。

    フレーム t のパケットが 1 つでも落ちると、クラウドは t の NACK を返し、端末はフレーム t + feedback_delay を符号化する前に受け取る。
    フィードバック路ではロスは起きないとする。損失の乱数の使い方は :func:`decode_clip` と同じ（クリップとシードで決まり、
    フレームごとに drop を 1 回呼ぶ）なので、パケット数が同じ間は同じロスのパターンになる。

    encoded に同じ (方式, P の量子化, 系列数, ビット数) をフィードバックなしで符号化した結果（:func:`encode_clip`）を渡すと、
    今の GoP でまだ NACK を受けていない（= フィードバックなしと同じ符号化になる）フレームでは、そのパケット分割とエントロピーを
    使い回す（パケット分割が重いので。結果は使い回さない場合と完全に一致する）。
    layout_cache に dict を渡すと、NACK を受けた後のフレームでも、量子化インデックスが同じならパケット分割を使い回す
    （同じ dict を同じクリップ・方式・ビット数の損失条件の間で共有する。:func:`_layout_cached`）。
    scheme.fec > 0 なら FEC を併用し、FEC で復元できなかったフレームだけ NACK を返す。
    """
    part = _Accumulator(cfg.gop)
    recon, entropies = _prs_recon(feats, cfg, scheme, bits, loss_cfg, seed, clip, part, restorer, encoded, layout_cache)
    (res,) = _evaluate([(part, recon, entropies)], feats, ref_pred, tail, tail_dev)
    _apply(acc, res, cfg)


# ---------------------------------------------------------------------------
# 並列化（1 クリップの (方式, ビット数) 1 組 = 1 タスク）
# ---------------------------------------------------------------------------

#: メインプロセスが先に head を計算してタスクを投入しておくクリップ数（今のクリップの先に何本まで）
LOOKAHEAD = 2

#: タスクの実行に使う状態（:func:`_init_worker` で設定する。ワーカーを使わないときはメインプロセスで設定する）
_WORKER: dict = {}


def _init_worker(cfg, tail, tail_dev, restorers: dict, conditions: list, threads: int | None = None, reused_encodings=frozenset()) -> None:
    """タスクの実行に使う状態を設定する（ワーカープロセスの initializer。threads を渡すと torch のスレッド数をそれにする）。

    reused_encodings（複数の方式で同じになる符号化のキー）を渡すと、同じクリップの符号化をタスクの間で使い回す
    （ワーカーを使わず 1 プロセスで順に処理するとき用。ワーカーではメモリを食うので使い回さない）。
    """
    if threads:
        torch.set_num_threads(threads)
    _WORKER.clear()
    _WORKER.update(
        cfg=cfg,
        tail=tail,
        tail_dev=tail_dev,
        restorers=restorers,
        conditions=conditions,
        reused_encodings=reused_encodings,
        encodings=(None, {}),  # (クリップ番号, 符号化のキー → EncodedClip)
    )


@torch.no_grad()
def _run_task(clip: int, scheme_index: int, bits: int, feats, ref_pred) -> list[_CondResult]:
    """1 クリップの (方式, ビット数) 1 組を、全ての損失条件・シードについて処理し、条件の順に結果を返す。

    符号化（損失条件によらないので 1 回）→ 条件ごとの復号（PRS は符号化と復号を交互に行う閉ループ）→ tail（全条件まとめて 1 回）。
    """
    w = _WORKER
    cfg = w["cfg"]
    scheme = cfg.schemes[scheme_index]
    restorer = w["restorers"].get(scheme.name)

    # 符号化が同じで復号だけ違う方式（D+DZ と D+DZ+R など）では、同じクリップの符号化を使い回す（1 プロセスで動かすときだけ）
    enc_key = encoding_key(scheme, bits)
    enc_clip, encodings = w["encodings"]
    if enc_clip != clip:
        encodings = {}
        w["encodings"] = (clip, encodings)
    encoded = encodings.get(enc_key) or encode_clip(feats, cfg, scheme, bits)
    if enc_key in w["reused_encodings"]:
        encodings[enc_key] = encoded

    # PRS で NACK を受けた後のフレームのパケット分割を、損失条件の間で使い回す
    layout_cache = {} if scheme.prs else None
    parts = []
    for loss_cfg, seed in w["conditions"]:
        part = _Accumulator(cfg.gop)
        if scheme.prs:
            recon, entropies = _prs_recon(feats, cfg, scheme, bits, loss_cfg, seed, clip, part, restorer, encoded, layout_cache)
        else:
            recon, entropies = _decode_recon(encoded, feats, cfg, loss_cfg, seed, clip, part, restorer, scheme.fec)
        parts.append((part, recon, entropies))
    return _evaluate(parts, feats, ref_pred, w["tail"], w["tail_dev"])


def available_cores() -> int:
    """このプロセスに割り当てられたコア数（os.sched_getaffinity が無い環境（macOS など）ではマシンのコア数）。"""
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


def limit_cpu_threads() -> int:
    """torch の CPU スレッド数を、このプロセスに割り当てられたコア数以下にして、使うスレッド数を返す。

    torch の既定はマシン全体の物理コア数なので、ジョブに一部のコアしか割り当てられていないと
    スレッドが多すぎて遅くなる。os.sched_getaffinity が無い環境（macOS など）では何もしない。
    """
    if hasattr(os, "sched_getaffinity"):
        n = len(os.sched_getaffinity(0))
        if torch.get_num_threads() > n:
            torch.set_num_threads(n)
    return torch.get_num_threads()


def target_name(target) -> str:
    return f"{target.model.name}/{target.split_point}"


def select_targets(targets: list, names: list[str] | None) -> list:
    """names（モデル名・「モデル名/分割点」・0 始まりの番号）に一致する target を、設定ファイルの順で返す。

    names が空なら全ての target を返す。一致しない名前があればエラーにする。
    """
    if not names:
        return list(targets)
    chosen = set()
    for name in names:
        if name.isdigit() and int(name) < len(targets):
            hits = [int(name)]
        else:
            hits = [i for i, t in enumerate(targets) if name in (t.model.name, target_name(t))]
        if not hits:
            candidates = ", ".join(f"{i}: {target_name(t)}" for i, t in enumerate(targets))
            raise SystemExit(f"--target {name!r} に一致する target がありません。候補: {candidates}")
        chosen.update(hits)
    return [t for i, t in enumerate(targets) if i in chosen]


def select_schemes(schemes: list, names: list[str] | None) -> list:
    """names（方式名）に一致する方式を、設定ファイルの順で返す。

    names が空なら全ての方式を返す。一致しない名前があればエラーにする。
    """
    if not names:
        return list(schemes)
    known = {s.name for s in schemes}
    unknown = [n for n in names if n not in known]
    if unknown:
        raise SystemExit(f"--scheme {', '.join(unknown)} に一致する方式がありません。候補: {', '.join(s.name for s in schemes)}")
    return [s for s in schemes if s.name in names]


# ---------------------------------------------------------------------------
# 途中保存・再開
# ---------------------------------------------------------------------------


def resume_config(run_dir: Path, overrides: list[str]) -> PacketLossConfig:
    """再開する実行フォルダの config.yaml から設定を読む。

    条件が変わると前回分と結果が混ざるので、上書きはデバイス（devices.*）だけ許す。
    """
    path = run_dir / "config.yaml"
    if not path.exists():
        raise SystemExit(f"--resume: {path} がありません（packet_loss_sweep.py の実行フォルダを指定してください）")
    bad = [o for o in overrides if not o.startswith("devices.")]
    if bad:
        raise SystemExit(f"--resume では devices.* 以外の設定は変更できません: {bad}")
    return load_config(PacketLossConfig, path, overrides)


def _read_jsonl(path: Path) -> list[dict]:
    """JSONL を読む。壊れた行（書き込み途中で止まった最終行など）は読み飛ばす。"""
    if not path.exists():
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def _row_target(row: dict) -> str:
    return f"{row['model']}/{row['split_point']}"


def finished_targets(run_dir: Path, cfg: PacketLossConfig) -> set[str]:
    """results.jsonl に全条件の行がそろっている target の名前（モデル名/分割点）。

    チェックポイントの無い旧バージョンの実行フォルダでも、これで終わったモデルが分かる。
    """
    expected = len(cfg.schemes) * len(cfg.bits) * len(cfg.losses)
    counts = Counter(_row_target(r) for r in _read_jsonl(run_dir / "results.jsonl"))
    return {name for name, n in counts.items() if n >= expected}


def prune_unfinished(run_dir: Path, done: set[str]) -> None:
    """results / gop_curve の jsonl から、終わっていない target の行と壊れた行を取り除く（やり直す分の重複を防ぐ）。"""
    for name in ("results", "gop_curve"):
        path = run_dir / f"{name}.jsonl"
        if not path.exists():
            continue
        rows = [r for r in _read_jsonl(path) if _row_target(r) in done]
        tmp = path.with_suffix(".jsonl.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        os.replace(tmp, path)


def save_checkpoint(run_dir: Path, target, next_clip: int, num_clips: int, shape: list[int], accs: dict) -> None:
    """実行中の target の途中経過（次に処理するクリップ番号と集計値）を保存する。

    書き込み中に止まっても前回のチェックポイントが壊れないよう、一時ファイルに書いてから置き換える。
    """
    state = {
        "target": target_name(target),
        "next_clip": next_clip,
        "num_clips": num_clips,
        "shape": shape,
        "accs": {key: vars(acc) for key, acc in accs.items()},
    }
    path = run_dir / CHECKPOINT
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(state, f)
    os.replace(tmp, path)


def load_checkpoint(run_dir: Path, target, num_clips: int, gop: int) -> tuple[int, list[int], dict] | None:
    """target のチェックポイントがあれば (次のクリップ番号, 特徴の形状, 集計値) を返す。無ければ None。"""
    path = run_dir / CHECKPOINT
    if not path.exists():
        return None
    with open(path, "rb") as f:
        state = pickle.load(f)
    if state["target"] != target_name(target):
        return None
    if state["num_clips"] != num_clips:
        raise SystemExit(f"チェックポイントのクリップ数 {state['num_clips']} が現在のデータセット {num_clips} 本と違います")
    accs = {}
    for key, fields in state["accs"].items():
        acc = _Accumulator(gop)
        vars(acc).update(fields)
        accs[key] = acc
    return state["next_clip"], state["shape"], accs


def main() -> None:
    def extra(p):
        p.add_argument("--target", action="append", metavar="NAME", help="実行する target（モデル名・モデル名/分割点・番号）。複数回指定可。省略時は全て")
        p.add_argument("--scheme", action="append", metavar="NAME", help="実行する方式（設定の schemes の name）。複数回指定可。省略時は全て")
        p.add_argument("--list-targets", action="store_true", help="設定にある target を番号付きで表示して終了する")
        p.add_argument("--resume", type=Path, metavar="RUN_DIR", help="途中で止まった実行フォルダの続きから再開する（設定はそのフォルダの config.yaml を使う）")
        p.add_argument(
            "--workers",
            type=int,
            default=None,
            metavar="N",
            help="(方式, ビット数) を並列に処理するワーカープロセスの数（既定: 割り当てられたコア数）。1 ならワーカーを使わない",
        )

    cfg, args = parse_cli(PacketLossConfig, __doc__, extra_args=extra)
    if args.resume:
        if args.config or args.target or args.scheme:
            print("--resume では実行フォルダの config.yaml を使うので、--config / --target / --scheme は無視します")
        cfg = resume_config(args.resume, args.overrides)
    if args.list_targets:
        for i, t in enumerate(cfg.targets):
            print(f"{i}: {target_name(t)}")
        return
    if not args.resume:
        cfg.targets = select_targets(cfg.targets, args.target)
        cfg.schemes = select_schemes(cfg.schemes, args.scheme)
    for s in cfg.schemes:
        if s.prs and (s.scheme == "fd" or s.feedback_delay < 1):
            raise SystemExit(f"方式 {s.name}: prs は fd 以外で、feedback_delay は 1 以上にしてください")
        if s.fec < 0:
            raise SystemExit(f"方式 {s.name}: fec（パリティの比率）は 0 以上にしてください: {s.fec}")
        if s.restorer and not Path(s.restorer).exists():
            raise SystemExit(
                f"方式 {s.name} のロス補修 net {s.restorer} がありません。先に scripts/train_restorer.py で学習するか、"
                f"設定の schemes からこの方式を外してください"
            )
    set_seed(cfg.seed)
    threads = limit_cpu_threads()
    print(f"CPU スレッド数: {threads}")
    head_dev, tail_dev = resolve_device(cfg.devices.head), resolve_device(cfg.devices.tail)
    print(f"デバイス: head={head_dev}, tail={tail_dev}")
    workers = args.workers if args.workers is not None else available_cores()
    if workers > 1 and tail_dev.type != "cpu":
        print(f"tail が {tail_dev} なので、ワーカープロセスは使わず 1 プロセスで処理します（並列化は tail が cpu のときだけ）")
        workers = 1
    workers = max(1, workers)
    print(f"ワーカープロセス数: {workers}" + ("（使わない）" if workers == 1 else "（各 1 スレッド）"))
    done: set[str] = set()
    if args.resume:
        run_dir = args.resume
        done = finished_targets(run_dir, cfg)
        prune_unfinished(run_dir, done)
        meta_path = run_dir / "meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        meta.setdefault("resumed_at", []).append(dt.datetime.now().isoformat(timespec="seconds"))
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"再開します: {run_dir}（終わっている target: {', '.join(sorted(done)) or 'なし'}）")
    else:
        # 一部の target だけを実行する場合は、どのモデルの結果か分かるようにフォルダ名に入れる
        name = cfg.experiment_name
        if args.target:
            name += "_" + "+".join(dict.fromkeys(t.model.name for t in cfg.targets))
        run_dir = make_run_dir(cfg.output_dir, name)
        save_run_info(
            run_dir, to_dict(cfg), {"head_device": str(head_dev), "tail_device": str(tail_dev), "cpu_threads": threads, "workers": workers}
        )
        print(f"結果の保存先: {run_dir}")
    print(f"途中で止まったら次のコマンドで再開できます: python scripts/packet_loss_sweep.py --resume {run_dir}")

    # ロス補修 net は方式ごとに 1 回だけ読み込む（モデル共通なので全ての target で同じものを使う）
    restorers = {s.name: load_restorer(s.restorer, tail_dev) for s in cfg.schemes if s.restorer}
    for name, r in restorers.items():
        print(f"{name}: ロス補修 net を使います（学習条件: {r.meta}）")

    # 複数の方式で同じになる符号化（方式, P の量子化, 系列数, ビット数）。1 プロセスで動かすときは、クリップごとに 1 回だけ符号化して
    # 使い回す（PRS も、NACK を受けるまではフィードバックなしと同じ符号化なので、その分を使い回す）
    enc_count = Counter(encoding_key(s, b) for s in cfg.schemes for b in cfg.bits)
    reused_encodings = frozenset(key for key, n in enc_count.items() if n > 1)

    dataset = build_dataset(cfg.data)
    print(f"クリップ {len(dataset)} 本を評価します（GoP={cfg.gop}, MTU={cfg.mtu}, concealment={cfg.concealment}）")

    # 損失率 0 の条件はシードによらないので 1 回だけ実行する
    conditions = []
    for loss_cfg in cfg.losses:
        for seed in cfg.seeds if loss_cfg.p > 0 else cfg.seeds[:1]:
            conditions.append((loss_cfg, seed))

    # 1 クリップのタスク = (方式の番号, ビット数)。結果はこの順（以前の版のループの順）に足し込む
    tasks = [(si, bits) for si in range(len(cfg.schemes)) for bits in cfg.bits]
    # 投入は重い PRS（損失条件ごとに符号化し直す）から先にする（待ち時間を減らすため。結果の順には影響しない）
    submit_order = sorted(range(len(tasks)), key=lambda i: not cfg.schemes[tasks[i][0]].prs)

    with ResultWriter(run_dir) as writer, ResultWriter(run_dir, "gop_curve") as curve_writer:
        for target in cfg.targets:
            if target_name(target) in done:
                print(f"\n=== {target.model.name} / {target.split_point}: 前回の実行で終わっているのでスキップ")
                continue
            # pretrained: false のときの初期重みが、前の target を飛ばして再開しても変わらないように target ごとに乱数を戻す
            set_seed(cfg.seed)
            bundle = build_split(target.model, target.split_point)
            if not bundle.point.is_clean:
                raise ValueError(f"{target.model.name} の分割点 {target.split_point!r} は送信テンソルが複数あります（clean な点を指定してください）")
            head = bundle.head.to(head_dev).eval()
            tail = bundle.tail.to(tail_dev).eval()
            model = bundle.model.to(tail_dev).eval()
            print(f"\n=== {target.model.name} / {target.split_point}")

            # acc[(方式名, ビット数, 損失名, シード)]
            accs: dict[tuple, _Accumulator] = defaultdict(lambda: _Accumulator(cfg.gop))
            start_clip, shape = 0, None
            ckpt = load_checkpoint(run_dir, target, len(dataset), cfg.gop)
            if ckpt is not None:
                start_clip, shape, saved = ckpt
                accs.update(saved)
                print(f"  チェックポイントから再開: クリップ {start_clip + 1}/{len(dataset)} から")

            pool = None
            if workers > 1:
                # fork は torch / OpenMP のスレッドを持った状態で固まることがあるので spawn にする。
                # tail・restorer の重みは torch の仕組みで共有メモリに置かれ、ワーカーごとにはコピーされない
                pool = ProcessPoolExecutor(
                    workers,
                    mp_context=multiprocessing.get_context("spawn"),
                    initializer=_init_worker,
                    initargs=(cfg, tail, tail_dev, restorers, conditions, 1),
                )
            else:
                _init_worker(cfg, tail, tail_dev, restorers, conditions, reused_encodings=reused_encodings)

            def submit(clip: int) -> tuple[list[int], dict]:
                """クリップ clip の head を計算してタスクを投入し、(特徴の形状, タスクの番号 → Future) を返す。

                ワーカーを使わないときは、Future の代わりに :func:`_run_task` の引数を返すだけ（足し込むときに実行する）。
                """
                x, _ = dataset[clip]
                with torch.no_grad():
                    ref_pred = model(x.to(tail_dev)).argmax(-1).cpu()
                    (feats,) = head(x.to(head_dev))
                feats = feats.float().cpu()
                if pool is None:
                    return list(feats.shape[1:]), {i: (clip, *tasks[i], feats, ref_pred) for i in range(len(tasks))}
                # タスクごとに特徴（resnet50/layer2 で 1 クリップ 48MB）を pickle しないよう、共有メモリに置いてから渡す
                feats.share_memory_()
                ref_pred.share_memory_()
                return list(feats.shape[1:]), {i: pool.submit(_run_task, clip, *tasks[i], feats, ref_pred) for i in submit_order}

            try:
                started = time.perf_counter()
                pending: deque[tuple[list[int], dict]] = deque()
                next_clip = start_clip
                for clip in range(start_clip, len(dataset)):
                    # ワーカーを使うときは、今のクリップを待つ間に先のクリップのタスクも投入しておく
                    while next_clip < len(dataset) and len(pending) <= (LOOKAHEAD if pool is not None else 0):
                        pending.append(submit(next_clip))
                        next_clip += 1
                    shape, jobs = pending.popleft()
                    for i, (si, bits) in enumerate(tasks):
                        results = jobs[i].result() if pool is not None else _run_task(*jobs[i])
                        for (loss_cfg, seed), res in zip(conditions, results):
                            _apply(accs[(cfg.schemes[si].name, bits, loss_cfg.display_name(), seed)], res, cfg)
                    save_checkpoint(run_dir, target, clip + 1, len(dataset), shape, accs)
                    print(f"  クリップ {clip + 1}/{len(dataset)} 完了（経過 {time.perf_counter() - started:.0f} 秒）", flush=True)
            finally:
                if pool is not None:
                    pool.shutdown(cancel_futures=True)

            rows, curve = _summarize(cfg, target, shape, accs)
            # gop_curve を先に書く（results の行がそろっていれば gop_curve もそろっている、という再開時の判定のため）
            for r in curve:
                curve_writer.write(r)
            for r in rows:
                writer.write(r)
            (run_dir / CHECKPOINT).unlink(missing_ok=True)
        summary = list(writer.rows)  # 再開した場合は前回分も含めて表示する

    cols = ["model", "split_point", "scheme", "bits", "loss", "agreement", "agreement_drop", "feature_mse", "loss_rate", "entropy_bpe"]
    print("\n" + format_table(summary, cols))
    print(f"\n全結果: {run_dir / 'results.csv'}\nGoP 位置ごと: {run_dir / 'gop_curve.csv'}")


def _summarize(cfg: PacketLossConfig, target, shape, accs) -> tuple[list[dict], list[dict]]:
    """シードごとの集計をまとめて、(results の行, gop_curve の行) を返す。"""
    by_cond: dict[tuple, list[_Accumulator]] = defaultdict(list)
    for (scheme, bits, loss, _seed), acc in accs.items():
        by_cond[(scheme, bits, loss)].append(acc)
    loss_cfgs = {l.display_name(): l for l in cfg.losses}
    schemes = {s.name: s for s in cfg.schemes}

    rows, curve = [], []
    for (scheme, bits, loss), group in by_cond.items():
        frames = sum(a.frames for a in group)
        agreement = sum(a.agree for a in group) / frames
        per_seed = [a.agree / a.frames for a in group]
        lc = loss_cfgs[loss]
        sc = schemes.get(scheme, TemporalSchemeConfig(name=scheme))
        rows.append(
            {
                "model": target.model.name,
                "split_point": target.split_point,
                "feature_shape": "x".join(map(str, shape)),
                "scheme": scheme,
                "bits": bits,
                "streams": sc.streams,  # 多系列 DPCM の系列数（1 = 通常）
                "prs": sc.prs,  # NACK による参照フレーム選択（PRS）
                "feedback_delay": sc.feedback_delay if sc.prs else None,  # NACK が届くまでのフレーム数
                "fec_rate": sc.fec,  # FEC のパリティの比率 r（0 = FEC なし）
                "clip": sc.clip,  # 量子化の範囲 R の決め方（p99 / max）
                "p_step": sc.p_step,  # P フレームの q の決め方（adaptive / intra）
                "p_step_scale": sc.p_step_scale if sc.p_step == "intra" else None,  # intra の倍率
                "coding": "adaptive_ac",  # パケットの符号化（packet_loss_plot.py が古い固定長の結果と見分ける用）
                "gop": cfg.gop,
                "loss": loss,
                "loss_model": lc.name,
                "loss_p": lc.p,
                "burst_len": lc.burst_len if lc.name == "gilbert_elliott" else None,
                "num_seeds": len(group),
                "num_frames": frames,
                "agreement": agreement,
                # 95% 信頼区間の半幅（フレームを独立とみなした正規近似。論文と同じ扱い）
                "agreement_ci95": 1.96 * math.sqrt(agreement * (1 - agreement) / frames),
                "agreement_seed_std": statistics.stdev(per_seed) if len(per_seed) > 1 else 0.0,
                "feature_mse": sum(a.mse for a in group) / frames,
                "loss_rate": sum(a.lost_packets for a in group) / sum(a.packets for a in group),
                "packets_per_frame": sum(a.packets for a in group) / frames,
                "bytes_per_frame": sum(a.bytes for a in group) / frames,
                "entropy_bpe": sum(a.entropy for a in group) / frames,
                # PRS のフィードバック: 1 フレームあたりの NACK 数（NACK 1 つ = 壊れたフレームの番号 u32）と強制 I フレーム数
                "nacks_per_frame": sum(a.nacks for a in group) / frames,
                "forced_intra_per_frame": sum(a.forced_intra for a in group) / frames,
                # FEC: 1 フレームあたりのパリティのパケット数と、FEC で復元した後のデータパケットの損失率
                # （loss_rate はパリティを含む通信路での損失率。FEC なしでは両者は一致する）
                "parity_packets_per_frame": sum(a.parity_packets for a in group) / frames,
                "residual_loss_rate": sum(a.residual_lost for a in group) / max(1, sum(a.data_packets for a in group)),
                # P フレームの量子化インデックスのうち、範囲の端 ±2^(b−1) に張り付いた（飽和した）要素の割合
                "clamp_rate": sum(a.p_clamped for a in group) / max(1, sum(a.p_elements for a in group)),
                # P フレーム 1 枚あたりの送信量と、1 フレームの送信量の最大値（I フレームとの差・ピークの大きさを見る用）
                "p_bytes_per_frame": sum(a.p_bytes for a in group) / max(1, sum(a.p_frames for a in group)),
                "max_frame_bytes": max(a.max_frame_bytes for a in group),
            }
        )
        pos_count = sum(a.pos_count for a in group)
        pos_agree = sum(a.pos_agree for a in group)
        pos_mse = sum(a.pos_mse for a in group)
        for t in range(cfg.gop):
            if pos_count[t] == 0:
                continue
            curve.append(
                {
                    "model": target.model.name,
                    "split_point": target.split_point,
                    "scheme": scheme,
                    "bits": bits,
                    "loss": loss,
                    "gop_position": t,
                    "num_frames": int(pos_count[t]),
                    "agreement": pos_agree[t] / pos_count[t],
                    "feature_mse": pos_mse[t] / pos_count[t],
                }
            )

    # 同じ方式・ビット数でロスがない場合（損失率 0 の条件があれば）との差
    no_loss = {(r["scheme"], r["bits"]): r["agreement"] for r in rows if r["loss_p"] == 0}
    for r in rows:
        base = no_loss.get((r["scheme"], r["bits"]))
        r["agreement_drop"] = base - r["agreement"] if base is not None else None
    return rows, curve


if __name__ == "__main__":
    main()

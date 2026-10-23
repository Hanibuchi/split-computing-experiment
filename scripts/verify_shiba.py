"""Shiba et al. (IEEE Access 2026) の表・図を、この基盤の実装で再計算して論文の値と比べる（再現検証）。

H. Shiba et al., "Delta Coding With Dead-Zone Quantization for Bandwidth-Efficient Split Computing on Video Streams,"
IEEE Access, vol. 14, 2026。論文の数値は splitcomp.temporal.paper_values に書き写してある。

論文と同じ条件で計算する:
    - 4 モデル × 3 分割点 = 12 条件（Table 3）、UCF-101 の 30 クリップ × 30 フレーム（selection: paper）、GoP K = 30
    - 4 方式（FD / DwD / DnD / DPCM）× b ∈ {2,3,4,6,8} × θ ∈ {0, 0.05, 0.1, 0.2} × {対称, dead-zone} = 1,920 通り
      （dead-zone のとき、FD は全フレームを dead-zone（F+DZ）、差分の方式は I を対称・P を dead-zone（D+DZ）で量子化）
    - 量子化の範囲 R は P99。EfficientNetV2-S だけ Max も計算する（Table 11/12）
    - 15fps（frame_step=2）の特徴の統計（Fig. 4）、GoP 長 K ∈ {5,10,15,30} の GoP 平均エントロピー（Fig. 11）

FD / DwD / DPCM は splitcomp.temporal の TemporalEncoder / TemporalDecoder（ロスなし）をそのまま使う。
DnD（クラウドが真の f_{t−1} を持つ理想化）と F+DZ は TemporalEncoder に無いので、ここで同じ quantize を使って作る。

出力（runs/verify_shiba_<日時>/）:
    raw/<モデル>_<分割点>.json  target ごとの計算結果（target が終わるたびに保存。同じ --out で再実行すると終わった target は飛ばす）
    report/report.md            表ごとの「論文 / 本実装 / 差 / 判定」と、本文の主張の検証
    report/*.csv                比較の全行（comparisons.csv）、条件ごとの結果（conditions.csv）、特徴の統計（stats.csv）
    report/fig*.png             論文の Fig. 4–11 と同じ形の図（論文の値は白抜きの記号で重ねる）

判定: ✓ 許容幅以内 / △ 許容幅の 2 倍以内 / ✗ それ以上（許容幅は TOL）。
論文のクリップのリストは公開されておらず PAPER_CLIPS は推測なので、数値の完全な一致は期待できない。

例:
    python scripts/verify_shiba.py                                   # 本番（12 条件 × 30 クリップ。数時間）
    python scripts/verify_shiba.py --target resnet50 --num-clips 2 --quick   # 動作確認
    python scripts/verify_shiba.py --out runs/verify_shiba_20261023-120000   # 途中で止まった実行の続き
    python scripts/verify_shiba.py --report-only runs/verify_shiba_20261023-120000
    python scripts/verify_shiba.py --report-only RUN --packet-loss-run runs/packet_loss_ucf101_merged

    # Slurm クラスタで 12 条件を並列に（scripts/slurm/verify_shiba.sbatch）
    sbatch scripts/slurm/verify_shiba.sbatch
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
import zlib
from pathlib import Path

import numpy as np
import torch

from splitcomp.builder import build_split
from splitcomp.config import ModelConfig
from splitcomp.data.ucf101 import UCF101Clips
from splitcomp.metrics import make_run_dir, save_run_info
from splitcomp.temporal import TemporalDecoder, TemporalEncoder, dequantize, p99_range, quantize
from splitcomp.temporal import paper_values as pv
from splitcomp.utils import resolve_device, set_seed

SCHEMES = pv.TABLE8_SCHEMES  # ("fd", "dwd", "dnd", "dpcm")
SCHEME_LABEL = {"fd": "FD", "dwd": "DwD", "dnd": "DnD", "dpcm": "DPCM"}
GOP_LENGTHS = (5, 10, 15, 30)
ZLIB_BITS = (4, 6, 8)  # Fig. 10

#: 許容幅 (種類, 幅)。"abs" は差の絶対値、"rel" は相対誤差で判定する
TOL = {
    "rho": ("abs", 0.02),
    "rho_std": ("abs", 0.02),
    "var": ("rel", 0.25),
    "R": ("rel", 0.20),
    "ratio": ("rel", 0.25),
    "db": ("abs", 0.3),
    "alpha": ("rel", 0.25),
    "agreement": ("abs", 0.05),  # 論文の 95% CI（±0.04）+ クリップの違い
    "entropy": ("abs", 0.3),
    "sparsity": ("abs", 0.05),
    "read": ("abs", 0.08),  # 図からの読み取り値
}


# ---------------------------------------------------------------------------
# 計算
# ---------------------------------------------------------------------------


def limit_cpu_threads() -> int:
    """torch の CPU スレッド数を割り当てられたコア数以下にする（packet_loss_sweep.py と同じ）。"""
    if hasattr(os, "sched_getaffinity"):
        n = len(os.sched_getaffinity(0))
        if torch.get_num_threads() > n:
            torch.set_num_threads(n)
    return torch.get_num_threads()


def entropy_and_zeros(idx: torch.Tensor, bits: int) -> tuple[float, int]:
    """インデックスの 0 次エントロピー [bit/要素]（論文の式 (29)）と 0 の個数。

    splitcomp.temporal.entropy_bits と同じ値を、bincount で速く求める。
    """
    half = 1 << (bits - 1)
    counts = torch.bincount((idx.flatten() + half).long(), minlength=1 << bits).double()
    p = counts[counts > 0] / counts.sum()
    return float(-(p * torch.log2(p)).sum()), int(counts[half])


def zlib_bpe(idx: torch.Tensor, bits: int) -> float:
    """量子化インデックスを 1 要素 1 バイトにして zlib（level 6）で圧縮したときの bit/要素（論文 §V-C-4）。"""
    data = (idx.flatten() + (1 << (bits - 1))).to(torch.uint8).numpy().tobytes()
    return len(zlib.compress(data, 6)) * 8 / idx.numel()


def reconstruct(feats: torch.Tensor, scheme: str, bits: int, mode: str, clip: str, theta: float, gop: int):
    """1 クリップを符号化・復号（ロスなし）して、(復元特徴, フレームごとの (I/P, インデックス)) を返す。"""
    T = feats.shape[0]
    recon = torch.empty_like(feats)
    coded = []
    if scheme in ("dwd", "dpcm") or (scheme == "fd" and mode == "sym"):
        enc = TemporalEncoder(scheme, bits, gop, p_quant=mode, clip=clip, theta=theta)
        dec = TemporalDecoder()
        for t in range(T):
            frame = enc.encode(feats[t])
            recon[t] = dec.decode(frame)
            coded.append((frame.kind, frame.idx))
    elif scheme == "fd":  # F+DZ: 全フレームを dead-zone で量子化（TemporalEncoder の I フレームは常に対称なのでここで作る）
        for t in range(T):
            idx, q = quantize(feats[t], bits, mode, clip, theta)
            recon[t] = dequantize(idx, q)
            coded.append(("I", idx.flatten()))
    elif scheme == "dnd":  # DnD: クラウドが真の f_{t−1} を持つ理想化（式 (5)）
        for t in range(T):
            if t % gop == 0:
                idx, q = quantize(feats[t], bits, "sym", clip, theta)
                recon[t] = dequantize(idx, q)
                coded.append(("I", idx.flatten()))
            else:
                idx, q = quantize(feats[t] - feats[t - 1], bits, mode, clip, theta)
                recon[t] = feats[t - 1] + dequantize(idx, q)
                coded.append(("P", idx.flatten()))
    else:
        raise KeyError(scheme)
    return recon, coded


def conditions(model: str, quick: bool) -> list[dict]:
    """1 target で計算する条件の一覧（scheme, mode, bits, theta, clip）。"""
    out = []
    thetas = (0.0,) if quick else pv.THETAS
    for scheme in SCHEMES:
        for mode in pv.MODES:
            for bits in pv.BITS:
                for theta in thetas:
                    out.append(dict(scheme=scheme, mode=mode, bits=bits, theta=theta, clip="p99"))
    if model == "efficientnet_v2_s":
        if quick:  # Table 10 の θ だけは --quick でも計算する
            for bits in (2, 4, 8):
                for theta in pv.THETAS[1:]:
                    out.append(dict(scheme="dpcm", mode="dz", bits=bits, theta=theta, clip="p99"))
        for bits in pv.TABLE11_BITS:  # Table 11/12
            out.append(dict(scheme="dpcm", mode="dz", bits=bits, theta=0.0, clip="max"))
    return out


def cond_key(c: dict) -> str:
    return f"{c['scheme']}|{c['mode']}|{c['bits']}|{c['theta']}|{c['clip']}"


def frame_stats(feats: torch.Tensor) -> dict:
    """特徴の統計（式 (2), (3)）。フレームごとの平均・分散・P99、隣り合うフレームの差分の統計と ρ。"""
    f = feats.flatten(1).double()
    mean, var = f.mean(1), f.var(1, unbiased=False)
    d = f[1:] - f[:-1]
    cross = (f[1:] * f[:-1]).mean(1)
    rho = (cross - mean[1:] * mean[:-1]) / (var[1:].sqrt() * var[:-1].sqrt())
    return {
        "mean_f": mean.tolist(),
        "var_f": var.tolist(),
        "R_f": [p99_range(x) for x in feats],
        "max_f": feats.flatten(1).abs().max(1).values.tolist(),
        "mean_d": d.mean(1).tolist(),
        "var_d": d.var(1, unbiased=False).tolist(),
        "R_d": [p99_range(x) for x in (feats[1:] - feats[:-1])],
        "rho": rho.tolist(),
    }


@torch.no_grad()
def run_target(target: tuple, args, head_dev, tail_dev, ds30, ds15) -> dict:
    """1 つの (モデル, 分割点) について全クリップ・全条件を計算する。"""
    model_name, sp, depth, *_ = target
    set_seed(0)
    bundle = build_split(ModelConfig(name=model_name, pretrained=True, trace_depth=depth), sp)
    head = bundle.head.to(head_dev).eval()
    tail = bundle.tail.to(tail_dev).eval()
    model = bundle.model.to(tail_dev).eval()
    conds = conditions(model_name, args.quick)
    T = pv.CLIP_LEN
    acc = {cond_key(c): dict(c, pos_agree=[0] * T, pos_mse=[0.0] * T, frames=0, mse=0.0, H_I=0.0, n_I=0, H_P=0.0, n_P=0,
                             Z_I=0.0, Z_P=0.0, zeros_I=0, zeros_P=0, elems_I=0, elems_P=0, alpha=0.0, n_alpha=0) for c in conds}
    gop_acc = {K: {"H": 0.0, "n": 0} for K in GOP_LENGTHS}
    stats30, stats15 = [], []
    started = time.perf_counter()

    for clip in range(len(ds30)):
        x, _ = ds30[clip]
        ref = model(x.to(tail_dev)).argmax(-1).cpu()
        (feats,) = head(x.to(head_dev))
        feats = feats.float().cpu()
        stats30.append(frame_stats(feats))
        if ds15 is not None:
            (f15,) = head(ds15[clip][0].to(head_dev))
            stats15.append(frame_stats(f15.float().cpu()))
        if tuple(feats.shape[1:]) != pv.TABLE3_SHAPE[(model_name, sp)]:
            raise ValueError(f"{model_name}/{sp} の特徴の形 {tuple(feats.shape[1:])} が論文の Table 3 と違います")

        # θ を変えてもインデックスが全く同じなら（量子化の 0 のビンが θ·R より広いとき）、tail の結果も同じなので使い回す
        last: dict[tuple, tuple] = {}
        for c in conds:
            a = acc[cond_key(c)]
            bits = c["bits"]
            recon, coded = reconstruct(feats, c["scheme"], bits, c["mode"], c["clip"], c["theta"], pv.GOP)
            group = (c["scheme"], c["mode"], bits, c["clip"])
            prev = last.get(group)
            if prev is not None and all(torch.equal(i, j) for (_, i), (_, j) in zip(coded, prev[1])):
                pred = prev[0]
            else:
                pred = tail(recon.to(tail_dev)).argmax(-1).cpu()
            last[group] = (pred, coded)
            mse = (recon - feats).pow(2).flatten(1).mean(1)
            for t, (ok, m) in enumerate(zip((pred == ref).tolist(), mse.tolist())):
                a["pos_agree"][t] += int(ok)
                a["pos_mse"][t] += m
            a["frames"] += T
            a["mse"] += float(mse.sum())
            want_zlib = bits in ZLIB_BITS and c["theta"] == 0 and c["clip"] == "p99" and c["scheme"] in ("fd", "dnd")
            for kind, idx in coded:
                H, zeros = entropy_and_zeros(idx, bits)
                a[f"H_{kind}"] += H
                a[f"n_{kind}"] += 1
                a[f"zeros_{kind}"] += zeros
                a[f"elems_{kind}"] += idx.numel()
                if want_zlib:
                    a[f"Z_{kind}"] += zlib_bpe(idx, bits)
            if c["scheme"] == "dpcm" and c["theta"] == 0 and c["clip"] == "p99":
                # 実測の α = Var(δ') / Var(δ) − 1（δ' = f_t − f̂_{t−1}, δ = f_t − f_{t−1}）。式 (26)
                for t in range(1, T):
                    if t % pv.GOP:
                        vd = float((feats[t] - feats[t - 1]).double().var(unbiased=False))
                        vdp = float((feats[t] - recon[t - 1]).double().var(unbiased=False))
                        a["alpha"] += vdp / vd - 1
                        a["n_alpha"] += 1

        # Fig. 11: GoP 長 K を変えたときの D+DZ（DPCM）の GoP 平均エントロピー（8-bit。tail は使わない）
        for K in GOP_LENGTHS:
            _, coded = reconstruct(feats, "dpcm", 8, "dz", "p99", 0.0, K)
            for _, idx in coded:
                gop_acc[K]["H"] += entropy_and_zeros(idx, 8)[0]
                gop_acc[K]["n"] += 1
        print(f"  クリップ {clip + 1}/{len(ds30)} 完了（経過 {time.perf_counter() - started:.0f} 秒）", flush=True)

    return {
        "model": model_name,
        "split_point": sp,
        "shape": list(feats.shape[1:]),
        "num_clips": len(ds30),
        "quick": args.quick,
        "stats30": stats30,
        "stats15": stats15,
        "conditions": list(acc.values()),
        "gop_entropy": {str(K): v["H"] / v["n"] for K, v in gop_acc.items()},
    }


def select_targets(names: list[str] | None) -> list[tuple]:
    """--target（モデル名・モデル名/分割点・0 始まりの番号）に一致する論文の条件。"""
    if not names:
        return list(pv.TARGETS)
    out = []
    for i, t in enumerate(pv.TARGETS):
        if any(n == str(i) or n in (t[0], f"{t[0]}/{t[1]}", t[4]) for n in names):
            out.append(t)
    if not out:
        cands = ", ".join(f"{i}: {t[0]}/{t[1]}" for i, t in enumerate(pv.TARGETS))
        raise SystemExit(f"--target {names} に一致する条件がありません。候補: {cands}")
    return out


def compute(args) -> Path:
    threads = limit_cpu_threads()
    head_dev, tail_dev = resolve_device(args.device), resolve_device(args.device)
    if args.out:
        run_dir = Path(args.out)
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        run_dir = make_run_dir("runs", "verify_shiba")
    if not (run_dir / "config.yaml").exists():
        save_run_info(run_dir, {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
                      {"device": str(tail_dev), "cpu_threads": threads})
    raw_dir = run_dir / "raw"
    raw_dir.mkdir(exist_ok=True)
    print(f"結果の保存先: {run_dir}（CPU スレッド数 {threads}, デバイス {tail_dev}）")
    print(f"途中で止まったら同じコマンドに --out {run_dir} を付けると、終わった target を飛ばして続けられます")

    common = dict(num_clips=args.num_clips, clip_len=pv.CLIP_LEN, selection="paper", size=224, download=True)
    ds30 = UCF101Clips(args.data_root, frame_step=1, **common)
    ds15 = None if args.no_15fps else UCF101Clips(args.data_root, frame_step=2, **common)
    for target in select_targets(args.target):
        path = raw_dir / f"{target[0]}_{target[1]}.json"
        if path.exists():
            print(f"\n=== {target[0]} / {target[1]}: 計算済みなのでスキップ（{path}）")
            continue
        print(f"\n=== {target[0]} / {target[1]}")
        result = run_target(target, args, head_dev, tail_dev, ds30, ds15)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(result), encoding="utf-8")
        os.replace(tmp, path)
    return run_dir


# ---------------------------------------------------------------------------
# 集計
# ---------------------------------------------------------------------------


def load_raw(run_dir: Path):
    """raw/*.json を読み、(特徴の統計の表, 条件ごとの結果の表, GoP 長ごとのエントロピー, GoP 位置ごとの一致率) にする。"""
    import pandas as pd

    stats, conds, gops, pos = [], [], [], []
    for path in sorted((run_dir / "raw").glob("*.json")):
        r = json.loads(path.read_text(encoding="utf-8"))
        key = (r["model"], r["split_point"])
        base = {"model": r["model"], "split_point": r["split_point"], "label": pv.LABEL[key], "num_clips": r["num_clips"]}
        s = _summarize_stats(r["stats30"])
        if r["stats15"]:
            s15 = _summarize_stats(r["stats15"])
            s.update({f"{k}_15": v for k, v in s15.items() if k in ("rho_meas", "var_ratio", "rho_calc")})
        stats.append({**base, **s})
        for c in r["conditions"]:
            n_all = c["n_I"] + c["n_P"]
            row = {**base, **{k: c[k] for k in ("scheme", "mode", "bits", "theta", "clip")}}
            row.update(
                agreement=sum(c["pos_agree"]) / c["frames"],
                frames=c["frames"],
                feature_mse=c["mse"] / c["frames"],
                H_all=(c["H_I"] + c["H_P"]) / n_all,
                H_P=c["H_P"] / c["n_P"] if c["n_P"] else np.nan,
                zlib_all=(c["Z_I"] + c["Z_P"]) / n_all if c["Z_I"] + c["Z_P"] > 0 else np.nan,
                zlib_P=c["Z_P"] / c["n_P"] if c["Z_P"] > 0 else np.nan,
                sparsity=(c["zeros_I"] + c["zeros_P"]) / (c["elems_I"] + c["elems_P"]),
                sparsity_P=c["zeros_P"] / c["elems_P"] if c["elems_P"] else np.nan,
                alpha_meas=100 * c["alpha"] / c["n_alpha"] if c["n_alpha"] else np.nan,
            )
            conds.append(row)
            if c["theta"] == 0 and c["clip"] == "p99":
                n_clips = c["frames"] // len(c["pos_agree"])
                for t, (v, m) in enumerate(zip(c["pos_agree"], c["pos_mse"])):
                    pos.append({**base, "scheme": c["scheme"], "mode": c["mode"], "bits": c["bits"], "t": t,
                                "agreement": v / n_clips, "feature_mse": m / n_clips})
        for K, H in r["gop_entropy"].items():
            gops.append({**base, "K": int(K), "H": H})
    if not stats:
        raise SystemExit(f"{run_dir / 'raw'} に計算結果がありません")
    return pd.DataFrame(stats), pd.DataFrame(conds), pd.DataFrame(gops), pd.DataFrame(pos)


def _summarize_stats(clips: list[dict]) -> dict:
    """クリップごとのフレーム統計を、論文 Table 7 / 13 の値にまとめる（全フレーム・全ペアの平均）。"""
    cat = {k: np.concatenate([c[k] for c in clips]) for k in clips[0]}
    var_f, var_d = cat["var_f"].mean(), cat["var_d"].mean()
    clip_rho = np.array([np.mean(c["rho"]) for c in clips])
    return {
        "var_f": var_f,
        "var_d": var_d,
        "var_ratio": var_f / var_d,
        "R_f": cat["R_f"].mean(),
        "R_d": cat["R_d"].mean(),
        "p99_ratio": cat["R_f"].mean() / cat["R_d"].mean(),
        "rho_calc": 1 - var_d / (2 * var_f),
        "rho_meas": cat["rho"].mean(),
        "rho_pair_std": cat["rho"].std(),
        "rho_clip_mean": clip_rho.mean(),
        "rho_clip_std": clip_rho.std(ddof=1) if len(clip_rho) > 1 else np.nan,
        "mean_f": cat["mean_f"].mean(),
        "max_over_p99": (cat["max_f"] / cat["R_f"]).mean() if "max_f" in cat else np.nan,
        "delta_mean_abs": abs(cat["mean_d"].mean()),
        "delta_mean_abs_max_pair": np.abs(cat["mean_d"]).max(),
        "num_pairs": len(cat["rho"]),
    }


# ---------------------------------------------------------------------------
# 比較
# ---------------------------------------------------------------------------


class Comparison:
    """論文の値と本実装の値の比較を 1 行ずつためる。"""

    def __init__(self):
        self.rows: list[dict] = []

    def add(self, section: str, item: str, metric: str, paper, ours, tol: str) -> None:
        if paper is None or ours is None or (isinstance(ours, float) and math.isnan(ours)):
            return
        kind, width = TOL[tol]
        diff = ours - paper
        err = abs(diff) / abs(paper) if kind == "rel" and paper != 0 else abs(diff)
        verdict = "✓" if err <= width else "△" if err <= 2 * width else "✗"
        self.rows.append(dict(section=section, item=item, metric=metric, paper=paper, ours=float(ours), diff=float(diff),
                              tol=f"{'±' if kind == 'abs' else '±'}{width:g}{'' if kind == 'abs' else ' (相対)'}", tol_kind=tol, verdict=verdict))


def get(conds, key, **where):
    """条件 where に一致する行の列 key の値（無ければ None）。"""
    m = conds
    for k, v in where.items():
        m = m[m[k] == v]
    return None if m.empty else float(m[key].iloc[0])


def compare_all(stats, conds, gops) -> tuple[Comparison, list[dict]]:
    cmp = Comparison()
    have = {(r.model, r.split_point) for r in stats.itertuples()}
    keys = [k for k in pv.KEYS if k in have]
    st = {(r["model"], r["split_point"]): r for r in stats.to_dict("records")}
    base = dict(theta=0.0, clip="p99")

    for key in keys:
        lab, s = pv.LABEL[key], st[key]
        # Table 7
        for col, tol in zip(pv.TABLE7_COLUMNS, ("var", "var", "ratio", "R", "R", "ratio", "rho", "rho")):
            cmp.add("Table 7", lab, col, pv.TABLE7[key][pv.TABLE7_COLUMNS.index(col)], s[col], tol)
        # Table 13
        cmp.add("Table 13", lab, "rho_clip_mean", pv.TABLE13[key][0], s["rho_clip_mean"], "rho")
        cmp.add("Table 13", lab, "rho_clip_std", pv.TABLE13[key][1], s["rho_clip_std"], "rho_std")
        # Fig. 4（15fps）
        cmp.add("Fig. 4", lab, "rho_15fps (図の読み取り)", pv.FIG4_RHO15_READ[key], s.get("rho_meas_15"), "read")
        # Table 5（本実装の Rf/Rδ から計算）
        ratio = s["R_f"] / s["R_d"]
        cmp.add("Table 5", lab, "mse_ratio", pv.TABLE5[key][0], ratio**2, "ratio")
        cmp.add("Table 5", lab, "delta_b", pv.TABLE5[key][1], math.log2(ratio), "db")
        # Table 6（式 (27) と実測）
        for bits, alpha in zip(pv.TABLE6_BITS, pv.TABLE6[key]):
            formula = 100 * s["R_d"] ** 2 / (3 * 4**bits * s["var_d"])
            cmp.add("Table 6", lab, f"alpha_{bits}bit (式27)", alpha, formula, "alpha")
            meas = get(conds[(conds.model == key[0]) & (conds.split_point == key[1])], "alpha_meas", scheme="dpcm", mode="sym", bits=bits, **base)
            cmp.add("Table 6", lab, f"alpha_{bits}bit (実測)", alpha, meas, "alpha")
        c = conds[(conds.model == key[0]) & (conds.split_point == key[1])]
        # Table 8
        for bits, vals in pv.TABLE8[key].items():
            for scheme, paper in zip(SCHEMES, vals):
                cmp.add("Table 8", lab, f"{SCHEME_LABEL[scheme]} {bits}bit", paper, get(c, "agreement", scheme=scheme, mode="sym", bits=bits, **base), "agreement")
        # Table 9（8-bit。D+Sym / D+DZ は DnD の P フレーム）
        hf, hsym, hdz, *_ = pv.TABLE9[key]
        cmp.add("Table 9", lab, "H_F+Sym", hf, get(c, "H_all", scheme="fd", mode="sym", bits=8, **base), "entropy")
        cmp.add("Table 9", lab, "H_D+Sym", hsym, get(c, "H_P", scheme="dnd", mode="sym", bits=8, **base), "entropy")
        cmp.add("Table 9", lab, "H_D+DZ", hdz, get(c, "H_P", scheme="dnd", mode="dz", bits=8, **base), "entropy")
        # Fig. 8（2-bit。読み取り値）
        cmp.add("Fig. 8", lab, "F+DZ 2bit (図の読み取り)", pv.FIG8_FDZ_READ[key], get(c, "agreement", scheme="fd", mode="dz", bits=2, **base), "read")
        cmp.add("Fig. 8", lab, "D+DZ 2bit (図の読み取り)", pv.FIG8_DDZ_READ[key], get(c, "agreement", scheme="dnd", mode="dz", bits=2, **base), "read")

    # Table 10–12（EfficientNetV2-S）
    for (sp, bits), (agree, sparse) in pv.TABLE10.items():
        c = conds[(conds.model == "efficientnet_v2_s") & (conds.split_point == sp)]
        lab = pv.LABEL[("efficientnet_v2_s", sp)]
        for theta, pa, ps in zip(pv.THETAS, agree, sparse):
            w = dict(scheme="dpcm", mode="dz", bits=bits, theta=theta, clip="p99")
            cmp.add("Table 10", lab, f"agreement {bits}bit θ={theta}", pa, get(c, "agreement", **w), "agreement")
            cmp.add("Table 10", lab, f"sparsity {bits}bit θ={theta}", ps, get(c, "sparsity", **w), "sparsity")
    for (sp, bits), (p99, mx) in pv.TABLE11.items():
        c = conds[(conds.model == "efficientnet_v2_s") & (conds.split_point == sp)]
        lab = pv.LABEL[("efficientnet_v2_s", sp)]
        for clip, paper_a, paper_h in (("p99", p99, pv.TABLE12[(sp, bits)][0]), ("max", mx, pv.TABLE12[(sp, bits)][1])):
            w = dict(scheme="dpcm", mode="dz", bits=bits, theta=0.0, clip=clip)
            cmp.add("Table 11", lab, f"agreement {bits}bit {clip}", paper_a, get(c, "agreement", **w), "agreement")
            cmp.add("Table 12", lab, f"entropy {bits}bit {clip}", paper_h, get(c, "H_all", **w), "entropy")

    return cmp, check_claims(stats, conds, gops, keys)


def check_claims(stats, conds, gops, keys) -> list[dict]:
    """本文の主張を本実装の結果で確かめる。各行は (主張, 論文, 本実装, 判定)。"""
    out = []
    partial = len(keys) < len(pv.KEYS)
    # 一致率の数え上げ（論文の表と同じく小数 2 桁に丸めて比べる。全条件がそろっているときだけ）
    piv = conds[conds["clip"] == "p99"].pivot_table(index=["model", "split_point", "mode", "bits", "theta"], columns="scheme", values="agreement")
    piv = piv.round(2)
    n = len(piv)
    full = n == 12 * len(pv.BITS) * len(pv.THETAS) * 2

    def claim(name, paper, ours, ok, needs_all=False):
        """needs_all: 範囲・数え上げなど 12 条件がそろわないと比べられない主張（そろっていなければ判定は「—」）。
        "grid" なら、さらに論文の 480 条件（θ も全て）がそろっている必要がある。"""
        missing = partial or (needs_all == "grid" and not full)
        verdict = "—" if needs_all and missing else "✓" if ok else "✗"
        out.append({"claim": name, "paper": paper, "ours": ours, "verdict": verdict})

    s = stats.set_index(["model", "split_point"]).loc[keys]
    sym0 = conds[(conds.theta == 0) & (conds["clip"] == "p99")]
    C = pv.CLAIMS
    claim("全条件で ρ > 0.9（30fps）", "> 0.9", f"最小 {s.rho_meas.min():.3f}", s.rho_meas.min() > C["rho_min_30fps"])
    lo, hi = C["rho_range_30fps"]
    claim("ρ の範囲（30fps）", f"{lo}–{hi}", f"{s.rho_meas.min():.3f}–{s.rho_meas.max():.3f}",
          abs(s.rho_meas.min() - lo) <= 0.02 and abs(s.rho_meas.max() - hi) <= 0.02, needs_all=True)
    lo, hi = C["var_ratio_range"]
    claim("分散比 σf²/σδ² の範囲", f"{lo}×–{hi}×", f"{s.var_ratio.min():.1f}×–{s.var_ratio.max():.1f}×",
          abs(s.var_ratio.min() / lo - 1) <= 0.25 and abs(s.var_ratio.max() / hi - 1) <= 0.25, needs_all=True)
    err = (s.rho_meas - s.rho_calc).abs()
    claim("|ρ − ρcalc|（Proposition 1 の検証）", f"最大 {C['rho_calc_err_max']}, 9/12 が ≤ {C['rho_calc_err_9of12']}",
          f"最大 {err.max():.4f}, {int((err <= C['rho_calc_err_9of12']).sum())}/{len(err)} が ≤ {C['rho_calc_err_9of12']}",
          err.max() <= C["rho_calc_err_max"] * 2)
    claim("|δ̄| < 0.002（条件ごとの平均）", "< 0.002", f"最大 {s.delta_mean_abs.max():.4f}", s.delta_mean_abs.max() < C["delta_mean_abs_max"])
    if "rho_meas_15" in s:
        lo, hi = C["rho_range_15fps"]
        claim("ρ の範囲（15fps）", f"{lo}–{hi}", f"{s.rho_meas_15.min():.3f}–{s.rho_meas_15.max():.3f}",
              abs(s.rho_meas_15.min() - lo) <= 0.02 and abs(s.rho_meas_15.max() - hi) <= 0.02, needs_all=True)
        drop = s.var_ratio / s.var_ratio_15
        lo, hi = C["var_ratio_15fps_drop"]
        claim("15fps で分散比が何分の 1 になるか", f"{lo}–{hi}×", f"{drop.min():.2f}–{drop.max():.2f}×",
              drop.min() >= lo * 0.85 and drop.max() <= hi * 1.15, needs_all=True)
        claim("15fps でも分散比 > 3×", "> 3×", f"最小 {s.var_ratio_15.min():.1f}×", s.var_ratio_15.min() > C["var_ratio_15fps_min"])

    note = "" if full else f"（{n} 条件のみ。論文は 480）"
    for scheme, (paper_k, paper_n) in (("dpcm", C["dpcm_ge_fd"]), ("dnd", C["dnd_ge_fd"])):
        k = int((piv[scheme] >= piv["fd"]).sum())
        claim(f"{SCHEME_LABEL[scheme]} ≥ FD の条件数", f"{paper_k}/{paper_n} ({paper_k / paper_n:.1%})", f"{k}/{n} ({k / n:.1%}){note}",
              k / n >= paper_k / paper_n - 0.03, needs_all="grid")

    t8 = sym0[sym0["mode"] == "sym"].pivot_table(index=["model", "split_point", "bits"], columns="scheme", values="agreement")
    gap = (t8["dnd"] - t8["dpcm"]).reset_index(name="gap")
    g4, g2 = gap[gap.bits >= 4].gap.max(), gap[gap.bits == 2].gap.max()
    claim("DnD − DPCM の最大差（b ≥ 4 / 2-bit）", f"≤ {C['dnd_dpcm_gap_b4']} / ≤ {C['dnd_dpcm_gap_b2']}", f"{g4:.3f} / {g2:.3f}",
          g4 <= C["dnd_dpcm_gap_b4"] + 0.02 and g2 <= C["dnd_dpcm_gap_b2"] + 0.03, needs_all=True)

    alpha = {b: 100 * s.R_d**2 / (3 * 4**b * s.var_d) for b in (2, 4)}
    lo, hi = C["alpha_range_b2"]
    claim("α < 2%（b ≥ 4, 式 27）", "< 2%", f"最大 {alpha[4].max():.2f}%", alpha[4].max() < C["alpha_max_b4"])
    claim("α の範囲（2-bit, 式 27）", f"{lo}–{hi}%", f"{alpha[2].min():.1f}–{alpha[2].max():.1f}%",
          alpha[2].min() >= lo * 0.75 and alpha[2].max() <= hi * 1.25, needs_all=True)

    # §VI-C: エントロピーの勝ち数（b ∈ {4,6,8}）
    ent = sym0[sym0.bits.isin([4, 6, 8])]
    hf = ent[(ent.scheme == "fd") & (ent["mode"] == "sym")].set_index(["model", "split_point", "bits"]).H_all
    hsym = ent[(ent.scheme == "dnd") & (ent["mode"] == "sym")].set_index(["model", "split_point", "bits"]).H_P
    hdz = ent[(ent.scheme == "dnd") & (ent["mode"] == "dz")].set_index(["model", "split_point", "bits"]).H_P
    n36 = len(hf)
    for name, h, (pk, pn) in (("D+Sym", hsym, C["dsym_wins"]), ("D+DZ", hdz, C["ddz_wins"])):
        k = int((h < hf).sum())
        claim(f"{name} が F+Sym よりエントロピーが小さい条件数（4/6/8 bit）", f"{pk}/{pn}", f"{k}/{n36}", abs(k / max(n36, 1) - pk / pn) <= 0.1, needs_all=True)
    wins8 = {k: bool(hdz.get((*k, 8), np.nan) < hf.get((*k, 8), np.nan)) for k in keys}
    match = sum(wins8[k] == pv.TABLE9[k][5] for k in keys)
    claim("Table 9 の「D+DZ wins」（8-bit）の Yes/No の一致", f"{len(keys)}/{len(keys)}", f"{match}/{len(keys)}", match >= len(keys) - 1)
    saving = (hf - hdz).max()
    claim("D+DZ のエントロピー削減の最大", f"約 {C['ddz_max_saving']} bpe", f"{saving:.2f} bpe", abs(saving - C["ddz_max_saving"]) <= 0.3, needs_all=True)
    # zlib の順序（F+Sym / D+Sym / D+DZ の 3 つの大小がエントロピーと zlib で同じか）
    for bits, (pk, pn) in ((4, C["zlib_order_match_4bit"]), (8, C["zlib_order_match_8bit"])):
        k = tot = 0
        for key in keys:
            c = ent[(ent.model == key[0]) & (ent.split_point == key[1]) & (ent.bits == bits)]
            try:
                hs = [get(c, "H_all", scheme="fd", mode="sym"), get(c, "H_P", scheme="dnd", mode="sym"), get(c, "H_P", scheme="dnd", mode="dz")]
                zs = [get(c, "zlib_all", scheme="fd", mode="sym"), get(c, "zlib_P", scheme="dnd", mode="sym"), get(c, "zlib_P", scheme="dnd", mode="dz")]
            except (KeyError, IndexError):
                continue
            if None in hs or None in zs:
                continue
            tot += 1
            k += int(np.argsort(hs).tolist() == np.argsort(zs).tolist())
        claim(f"zlib の大小がエントロピーの大小と一致（{bits}-bit）", f"{pk}/{pn}", f"{k}/{tot}", tot > 0 and k >= tot - 1)
    z = ent.dropna(subset=["zlib_all"])
    zh = np.concatenate([(z.zlib_all - z.H_all).values, (z.zlib_P - z.H_P).dropna().values])
    claim("zlib のサイズ S はエントロピー H よりわずかに大きい（Fig. 10）", "S ≳ H", f"S − H: {zh.min():.2f}–{zh.max():.2f} bpe", len(zh) > 0 and zh.min() > -0.05)

    # §VII-A: 6-bit DnD ≥ 8-bit FD
    k = sum(int(round(t8.loc[(*key, 6), "dnd"], 2) >= round(t8.loc[(*key, 8), "fd"], 2)) for key in keys if (*key, 6) in t8.index)
    pk, pn = C["dnd6_ge_fd8"]
    claim("6-bit DnD ≥ 8-bit FD の条件数", f"{pk}/{pn}", f"{k}/{len(keys)}", k >= len(keys) - 2)
    # §III-D: F+DZ の崩壊
    c = conds[(conds.model == "efficientnet_v2_s") & (conds.split_point == "features_3") & (conds.theta == 0) & (conds["clip"] == "p99")]
    fs, fz = get(c, "agreement", scheme="fd", mode="sym", bits=2), get(c, "agreement", scheme="fd", mode="dz", bits=2)
    if fs is not None:
        pa, pb = C["fdz_eff_s3_2bit"]
        claim("F+DZ は崩壊する（EfficientNetV2-S S3, 2-bit の F+Sym → F+DZ）", f"{pa} → {pb}", f"{fs:.2f} → {fz:.2f}", fz < fs - 0.3)

    # Proposition 3 / Fig. 7: DwD は 2-bit で GoP 内に単調に悪化し、DPCM はドリフトしない
    pos = _pos_cache.get("pos")
    if pos is not None and not pos.empty:
        p = pos[pos["mode"] == "sym"]
        def drift(scheme, bits):
            g = p[(p.scheme == scheme) & (p.bits == bits)]
            head = g[(g.t >= 1) & (g.t <= 5)].groupby(["model", "split_point"]).agreement.mean()
            tail_ = g[g.t >= 25].groupby(["model", "split_point"]).agreement.mean()
            return (head - tail_)
        d2, d8 = drift("dwd", 2), drift("dwd", 8)
        claim("DwD は 2-bit で GoP 内に悪化する（t=1–5 と t=25–29 の差）", "単調に悪化", f"差の平均 {d2.mean():.2f}（最小 {d2.min():.2f}）", d2.min() > 0.05)
        claim("DwD は 8-bit ではほぼ平ら", "ほぼ平ら", f"差の平均 {d8.mean():.2f}", abs(d8.mean()) < 0.05)
        dp = pd_concat([drift("dpcm", b) for b in (2, 4, 8)])
        claim("DPCM はドリフトしない（2/4/8-bit の t=1–5 と t=25–29 の差）", "ドリフトなし", f"差の平均 {dp.mean():.3f}（最大 {dp.max():.2f}）", dp.mean() < 0.03)

    # §VII-D-1 / Fig. 11: K を大きくすると（有利な条件では）bpe は下がり、K ≈ 15 で飽和する
    if not gops.empty:
        g = gops.pivot_table(index=["model", "split_point"], columns="K", values="H")
        dec = ((g[5] - g[15]).abs() >= 3 * (g[15] - g[30]).abs()).sum()
        claim("GoP 平均 bpe は K ≈ 15 を超えるとほぼ飽和", "K ≈ 15 で飽和", f"{int(dec)}/{len(g)} 条件で |H(5)−H(15)| ≥ 3|H(15)−H(30)|", dec >= len(g) * 0.8)
    return out


_pos_cache: dict = {}


def pd_concat(series):
    import pandas as pd

    return pd.concat(series)


# ---------------------------------------------------------------------------
# 図
# ---------------------------------------------------------------------------

COLORS = {"fd": "#2a78d6", "dwd": "#eda100", "dnd": "#d6336c", "dpcm": "#1baf7a"}
MODEL_COLORS = {"efficientnet_v2_s": "#2a78d6", "resnet50": "#eb6834", "vgg16": "#1baf7a", "mobilenet_v2": "#c2185b"}
PAPER_STYLE = dict(marker="o", mfc="none", mec="#0b0b0b", ls="none", ms=6, label="paper")


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 150, "font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "grid.color": "#e4e3df", "legend.frameon": False})
    return plt


def _save(fig, path: Path) -> None:
    fig.savefig(path, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)
    print(f"  {path}")


def make_figures(stats, conds, gops, pos, out: Path) -> None:
    plt = _plt()
    keys = [k for k in pv.KEYS if ((stats.model == k[0]) & (stats.split_point == k[1])).any()]
    labels = [pv.LABEL[k] for k in keys]
    st = stats.set_index(["model", "split_point"])
    sym0 = conds[(conds.theta == 0) & (conds["clip"] == "p99")]
    x = np.arange(len(keys))

    def sel(c, tk, **w):
        return get(c[(c.model == tk[0]) & (c.split_point == tk[1])], **w)

    # Fig. 4: 30fps と 15fps の ρ
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.bar(x - 0.2, [st.loc[k, "rho_meas"] for k in keys], 0.4, color="#2a78d6", label="30 fps (ours)")
    if "rho_meas_15" in st:
        ax.bar(x + 0.2, [st.loc[k, "rho_meas_15"] for k in keys], 0.4, color="#eb6834", label="15 fps (ours)")
        ax.plot(x + 0.2, [pv.FIG4_RHO15_READ[k] for k in keys], marker="_", ms=14, mew=2, color="#0b0b0b", ls="none", label="paper 15 fps (read from fig.)")
    ax.plot(x - 0.2, [pv.TABLE7[k][7] for k in keys], **{**PAPER_STYLE, "label": "paper 30 fps (Table 7)"})
    ax.axhline(0.83, ls=":", color="#8a8984")
    ax.set_ylim(0.7, 1.0)
    ax.set_xticks(x, labels, rotation=45, ha="right")
    ax.set_ylabel("Pearson correlation ρ")
    ax.set_title("Fig. 4: ρ at 30 fps vs. 15 fps")
    ax.legend(fontsize=7, ncol=2)
    _save(fig, out / "fig4_rho_fps.png")

    # Fig. 5: ビット数と一致率（代表 4 条件）
    reps = [k for k in [("efficientnet_v2_s", "features_4"), ("resnet50", "layer2"), ("vgg16", "features_16"), ("mobilenet_v2", "features_14")] if k in keys]
    if reps:
        fig, axes = plt.subplots(1, len(reps), figsize=(3 * len(reps), 2.8), sharey=True, squeeze=False)
        for ax, key in zip(axes[0], reps):
            for scheme in SCHEMES:
                ys = [sel(sym0, key, key="agreement", scheme=scheme, mode="sym", bits=b) for b in pv.BITS]
                ax.plot(pv.BITS, ys, marker="o", color=COLORS[scheme], label=f"{SCHEME_LABEL[scheme]} (ours)")
                ax.plot(pv.TABLE8_BITS, [pv.TABLE8[key][b][SCHEMES.index(scheme)] for b in pv.TABLE8_BITS],
                        marker="o", mfc="none", mec=COLORS[scheme], ls="none", ms=7)
            ax.set_title(pv.LABEL[key])
            ax.set_xlabel("bit width")
        axes[0, 0].set_ylabel("top-1 agreement")
        axes[0, -1].plot([], [], **{**PAPER_STYLE, "label": "paper (Table 8)"})
        axes[0, -1].legend(fontsize=7)
        fig.suptitle("Fig. 5: agreement vs. bit width (symmetric, θ = 0). filled: ours, hollow: paper", y=1.02)
        _save(fig, out / "fig5_agreement_vs_bits.png")

    # Fig. 6: DnD vs DPCM
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.6), sharex=True, sharey=True)
    markers = {2: "D", 4: "o", 6: "s", 8: "^"}
    for key in keys:
        for b in pv.TABLE8_BITS:
            col = MODEL_COLORS[key[0]]
            axes[0].plot(sel(sym0, key, key="agreement", scheme="dnd", mode="sym", bits=b), sel(sym0, key, key="agreement", scheme="dpcm", mode="sym", bits=b),
                         marker=markers[b], color=col, ls="none")
            axes[1].plot(pv.TABLE8[key][b][2], pv.TABLE8[key][b][3], marker=markers[b], color=col, ls="none")
    for ax, t in zip(axes, ("ours", "paper (Table 8)")):
        ax.plot([0.7, 1], [0.7, 1], ls="--", color="#8a8984")
        ax.set_title(t)
        ax.set_xlabel("DnD agreement")
    axes[0].set_ylabel("DPCM agreement")
    for m, c in MODEL_COLORS.items():
        axes[1].plot([], [], marker="o", color=c, ls="none", label=m)
    for b, mk in markers.items():
        axes[1].plot([], [], marker=mk, color="#8a8984", ls="none", label=f"{b}-bit")
    axes[1].legend(fontsize=7)
    fig.suptitle("Fig. 6: DnD vs. DPCM", y=1.02)
    _save(fig, out / "fig6_dnd_vs_dpcm.png")

    # Fig. 7: DwD の GoP 位置ごとの一致率
    reps7 = [k for k in [("efficientnet_v2_s", "features_4"), ("resnet50", "layer2"), ("vgg16", "features_16"), ("mobilenet_v2", "features_4")] if k in keys]
    if reps7 and not pos.empty:
        fig, axes = plt.subplots(2, 2, figsize=(9, 5.5), squeeze=False)
        bcol = {2: "#eb6834", 4: "#2a78d6", 6: "#1baf7a", 8: "#c2185b"}
        for ax, key in zip(axes.flat, reps7):
            p = pos[(pos.model == key[0]) & (pos.split_point == key[1]) & (pos["mode"] == "sym")]
            for b in pv.TABLE8_BITS:
                g = p[(p.scheme == "dwd") & (p.bits == b)].sort_values("t")
                ax.plot(g.t, g.agreement.rolling(5, center=True).mean(), color=bcol[b], label=f"{b}-bit")
                ax.axhline(sel(sym0, key, key="agreement", scheme="fd", mode="sym", bits=b), ls="--", color=bcol[b], alpha=0.5)
            ax.set_title(pv.LABEL[key])
            ax.set_xlabel("GoP position t")
            ax.set_ylabel("DwD agreement")
        axes[0, 0].legend(fontsize=7)
        fig.suptitle("Fig. 7: DwD agreement vs. GoP position (5-frame moving average; dashed: FD)", y=1.0)
        fig.tight_layout()
        _save(fig, out / "fig7_dwd_gop.png")

    # Fig. 8: dead-zone の非対称な効果（2-bit）
    fig, axes = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
    panels = [("Features (2-bit)", ("fd", "sym", "F+Sym", "#8a8984"), ("fd", "dz", "F+DZ", "#d6336c"),
               [pv.TABLE8[k][2][0] for k in keys], [pv.FIG8_FDZ_READ[k] for k in keys]),
              ("Deltas (2-bit, DnD)", ("dnd", "sym", "D+Sym", "#86c5f0"), ("dnd", "dz", "D+DZ", "#11427a"),
               [pv.TABLE8[k][2][2] for k in keys], [pv.FIG8_DDZ_READ[k] for k in keys])]
    for ax, (title, a, b, pa, pb) in zip(axes, panels):
        for off, (scheme, mode, name, col), paper in ((-0.2, a, pa), (0.2, b, pb)):
            ax.bar(x + off, [sel(sym0, k, key="agreement", scheme=scheme, mode=mode, bits=2) for k in keys], 0.4, color=col, label=name)
            ax.plot(x + off, paper, marker="_", ms=10, mew=2, color="#0b0b0b", ls="none")
        ax.set_title(title)
        ax.set_ylabel("top-1 agreement")
        ax.legend(fontsize=7)
    axes[1].set_xticks(x, labels, rotation=45, ha="right")
    fig.suptitle("Fig. 8: asymmetric effect of dead-zone (bars: ours, black ticks: paper)", y=1.0)
    _save(fig, out / "fig8_deadzone_asymmetry.png")

    # Fig. 9: 8-bit のエントロピー
    fig, ax = plt.subplots(figsize=(7, 3))
    pats = [("F+Sym", "fd", "sym", "H_all", 0, "#8a8984"), ("D+Sym", "dnd", "sym", "H_P", 1, "#86c5f0"), ("D+DZ", "dnd", "dz", "H_P", 2, "#11427a")]
    for i, (name, scheme, mode, col, pi, color) in enumerate(pats):
        off = (i - 1) * 0.27
        ax.bar(x + off, [sel(sym0, k, key=col, scheme=scheme, mode=mode, bits=8) for k in keys], 0.27, color=color, label=name)
        ax.plot(x + off, [pv.TABLE9[k][pi] for k in keys], marker="_", ms=8, mew=2, color="#0b0b0b", ls="none")
    ax.set_xticks(x, labels, rotation=45, ha="right")
    ax.set_ylabel("Shannon entropy (bpe)")
    ax.set_title("Fig. 9: entropy at 8-bit (bars: ours, black ticks: paper Table 9)")
    ax.legend(fontsize=7, ncol=3)
    _save(fig, out / "fig9_entropy.png")

    # Fig. 10: エントロピー vs zlib
    fig, ax = plt.subplots(figsize=(4.2, 4))
    bm = {4: "o", 6: "s", 8: "^"}
    for name, scheme, mode, hcol, zcol, color in (("F+Sym", "fd", "sym", "H_all", "zlib_all", "#8a8984"), ("D+Sym", "dnd", "sym", "H_P", "zlib_P", "#86c5f0"),
                                                  ("D+DZ", "dnd", "dz", "H_P", "zlib_P", "#11427a")):
        for b in ZLIB_BITS:
            g = sym0[(sym0.scheme == scheme) & (sym0["mode"] == mode) & (sym0.bits == b)]
            ax.plot(g[hcol], g[zcol], marker=bm[b], color=color, ls="none", label=name if b == 8 else None)
    lim = [0, 9]
    ax.plot(lim, lim, ls="--", color="#8a8984")
    ax.set_xlabel("Shannon entropy H (bpe)")
    ax.set_ylabel("zlib size S (bpe)")
    ax.set_title("Fig. 10: H vs. zlib (○4 □6 △8 bit)")
    ax.legend(fontsize=7)
    _save(fig, out / "fig10_entropy_vs_zlib.png")

    # Fig. 11: GoP 長 K と GoP 平均 bpe
    if not gops.empty:
        fig, axes = plt.subplots(2, 1, figsize=(5, 6), sharex=True)
        Ks = np.arange(2, 31)
        for key in keys:
            hf = sel(sym0, key, key="H_all", scheme="fd", mode="sym", bits=8)
            hdz = sel(sym0, key, key="H_P", scheme="dnd", mode="dz", bits=8)
            ax = axes[0] if hf > hdz else axes[1]
            col = MODEL_COLORS[key[0]]
            g = gops[(gops.model == key[0]) & (gops.split_point == key[1])].sort_values("K")
            ax.plot(g.K, g.H, marker="o", color=col, ls="none", label=pv.LABEL[key])
            ax.plot(Ks, (hf + (Ks - 1) * hdz) / Ks, ls="--", color=col, lw=1)
            phf, _, phdz, *_ = pv.TABLE9[key]
            ax.plot(GOP_LENGTHS, [(phf + (K - 1) * phdz) / K for K in GOP_LENGTHS], marker="x", color=col, ls="none", ms=5)
        axes[0].set_title("(a) favorable (H_f > H_D+DZ)")
        axes[1].set_title("(b) unfavorable")
        axes[1].set_xlabel("GoP length K")
        for ax in axes:
            ax.set_ylabel("GoP-avg bpe (8-bit)")
            ax.legend(fontsize=6, ncol=2)
        fig.suptitle("Fig. 11: ●ours measured, -- ours theory, × paper theory (Table 9)", y=1.0)
        fig.tight_layout()
        _save(fig, out / "fig11_gop_length.png")


def plot_drift_mse(pos, out: Path) -> None:
    """診断用: GoP 位置ごとの特徴の MSE（8-bit, 対称, θ = 0）。DPCM / DwD に I フレームのクリップ誤差が残るかが見える。"""
    if pos.empty:
        return
    plt = _plt()
    keys = [k for k in pv.KEYS if ((pos.model == k[0]) & (pos.split_point == k[1])).any()]
    cols = 4
    rows = math.ceil(len(keys) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 2.4 * rows), squeeze=False)
    for ax, key in zip(axes.flat, keys):
        p = pos[(pos.model == key[0]) & (pos.split_point == key[1]) & (pos["mode"] == "sym") & (pos.bits == 8)]
        for scheme in SCHEMES:
            g = p[p.scheme == scheme].sort_values("t")
            ax.plot(g.t, g.feature_mse, color=COLORS[scheme], label=SCHEME_LABEL[scheme])
        ax.set_yscale("log")
        ax.set_title(pv.LABEL[key])
    for ax in axes.flat[len(keys):]:
        ax.set_visible(False)
    axes[0, 0].legend(fontsize=7)
    axes[0, 0].set_ylabel("feature MSE")
    fig.suptitle("Diagnosis: feature MSE vs. GoP position (8-bit, symmetric, P99)", y=1.0)
    fig.tight_layout()
    _save(fig, out / "diag_mse_vs_gop.png")


def diagnosis_rows(stats, conds, pos) -> list[dict]:
    """診断用の表: P99 クリップの強さ（max|f| / P99|f|）と、DPCM / DwD の誤差が DnD よりどれだけ大きいか（8-bit）。"""
    rows = []
    sym0 = conds[(conds.theta == 0) & (conds["clip"] == "p99") & (conds["mode"] == "sym") & (conds.bits == 8)]
    for r in stats.itertuples():
        key = (r.model, r.split_point)
        c = sym0[(sym0.model == key[0]) & (sym0.split_point == key[1])].set_index("scheme")
        p = pos[(pos.model == key[0]) & (pos.split_point == key[1]) & (pos["mode"] == "sym") & (pos.bits == 8)]
        def mse_at(scheme, t):
            g = p[(p.scheme == scheme) & (p.t == t)]
            return float(g.feature_mse.iloc[0]) if not g.empty else np.nan
        rows.append({
            "item": pv.LABEL[key],
            "max|f| / P99|f|": r.max_over_p99,
            "MSE DPCM/DnD (t=1)": mse_at("dpcm", 1) / mse_at("dnd", 1),
            "MSE DPCM/DnD (t=29)": mse_at("dpcm", 29) / mse_at("dnd", 29),
            "MSE DwD/DnD (t=29)": mse_at("dwd", 29) / mse_at("dnd", 29),
            "agree DnD−DPCM": c.loc["dnd", "agreement"] - c.loc["dpcm", "agreement"] if {"dnd", "dpcm"} <= set(c.index) else np.nan,
            "agree DnD−DPCM (paper)": pv.TABLE8[key][8][2] - pv.TABLE8[key][8][3],
        })
    return rows


def plot_table_scatter(cmp_df, out: Path) -> None:
    """全ての比較を、指標の種類ごとに 論文の値 vs 本実装の値 の散布図にする。"""
    plt = _plt()
    groups = [("rho", "ρ"), ("agreement", "agreement"), ("entropy", "entropy (bpe)"), ("R", "P99 range R"), ("var", "variance"), ("alpha", "α (%)")]
    groups = [(g, t) for g, t in groups if (cmp_df.tol_kind == g).any()]
    fig, axes = plt.subplots(1, len(groups), figsize=(3 * len(groups), 3), squeeze=False)
    vcol = {"✓": "#1baf7a", "△": "#eda100", "✗": "#d6336c"}
    for ax, (g, title) in zip(axes[0], groups):
        d = cmp_df[cmp_df.tol_kind == g]
        for v, c in vcol.items():
            dd = d[d.verdict == v]
            ax.plot(dd.paper, dd.ours, "o", color=c, ms=3.5, label=v)
        lo, hi = min(d.paper.min(), d.ours.min()), max(d.paper.max(), d.ours.max())
        ax.plot([lo, hi], [lo, hi], ls="--", color="#8a8984")
        if g in ("var", "alpha"):
            ax.set_xscale("log")
            ax.set_yscale("log")
        ax.set_title(title)
        ax.set_xlabel("paper")
    axes[0, 0].set_ylabel("ours")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("All table values: paper vs. ours", y=1.03)
    _save(fig, out / "table_scatter.png")


# ---------------------------------------------------------------------------
# 報告書
# ---------------------------------------------------------------------------


def compare_packet_loss_run(pl_dir: Path) -> list[dict]:
    """packet_loss_sweep の結果のロスなしの行（F+Sym = FD, D+Sym = DPCM）を Table 8 と比べる。"""
    import pandas as pd

    df = pd.read_csv(pl_dir / "results.csv")
    df = df[df.loss_p == 0]
    cmp = Comparison()
    for r in df.itertuples():
        key = (r.model, r.split_point)
        col = {"F+Sym": 0, "D+Sym": 3}.get(r.scheme)
        if key in pv.TABLE8 and col is not None and r.bits in pv.TABLE8[key]:
            cmp.add("packet_loss (no_loss)", pv.LABEL[key], f"{r.scheme} {r.bits}bit", pv.TABLE8[key][r.bits][col], r.agreement, "agreement")
    return cmp.rows


def _md_table(rows: list[dict], cols: list[str], fmt=".3f") -> str:
    def f(v):
        if isinstance(v, float):
            return format(v, fmt) if not math.isnan(v) else ""
        return str(v).replace("|", "\\|")  # セルの中の | は表の区切りと見なされるのでエスケープする

    lines = ["| " + " | ".join(f(c) for c in cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(f(r.get(c, "")) for c in cols) + " |" for r in rows]
    return "\n".join(lines)


SECTION_NOTE = {
    "Table 7": "特徴の統計。σ²・R は全フレーム（ペア）の平均、ρcalc = 1 − σδ²/(2σf²)",
    "Table 13": "クリップごとの ρ の平均の、30 クリップでの平均と標準偏差",
    "Fig. 4": "15fps（1 フレームおき）の ρ。論文の値は棒グラフからの読み取り",
    "Table 5": "本実装の Rf/Rδ から計算した MSE 比 (Rf/Rδ)² と Δb = log2(Rf/Rδ)",
    "Table 6": "α [%]。「式27」は本実装の Rδ, σδ² を式 (27) に入れた値（論文の Table 6 と同じ求め方）、「実測」は Var(δ')/Var(δ) − 1",
    "Table 8": "top-1 一致率（対称量子化, θ = 0, 900 フレーム）",
    "Table 9": "8-bit のエントロピー [bpe]。F+Sym は全フレーム、D+Sym / D+DZ は真の差分（DnD）の P フレームの平均",
    "Fig. 8": "2-bit の一致率。F+DZ は FD を dead-zone で、D+DZ は DnD の P を dead-zone で量子化。論文の値は棒グラフからの読み取り",
    "Table 10": "EfficientNetV2-S, DPCM, dead-zone の θ 感度。スパース率は全フレームの量子化インデックスのうち 0 の割合",
    "Table 11": "EfficientNetV2-S, DPCM, dead-zone のクリップ範囲（P99 / Max）と一致率",
    "Table 12": "同じくエントロピー（全フレームの平均）",
    "packet_loss (no_loss)": "packet_loss_sweep のロスなしの行（F+Sym = FD, D+Sym = DPCM）と Table 8",
}


def write_report(run_dir: Path, packet_loss_run: Path | None) -> None:
    import pandas as pd

    out = run_dir / "report"
    out.mkdir(exist_ok=True)
    stats, conds, gops, pos = load_raw(run_dir)
    _pos_cache["pos"] = pos
    cmp, claims = compare_all(stats, conds, gops)
    rows = cmp.rows
    if packet_loss_run:
        rows += compare_packet_loss_run(packet_loss_run)
    cmp_df = pd.DataFrame(rows)
    cmp_df.to_csv(out / "comparisons.csv", index=False)
    conds.to_csv(out / "conditions.csv", index=False)
    stats.to_csv(out / "stats.csv", index=False)
    print(f"  {out / 'comparisons.csv'}")

    make_figures(stats, conds, gops, pos, out)
    plot_drift_mse(pos, out)
    plot_table_scatter(cmp_df, out)

    n_clips = sorted(stats.num_clips.unique().tolist())
    lines = [
        "# Shiba et al. (IEEE Access 2026) の再現検証",
        "",
        f"- 実行フォルダ: `{run_dir}`",
        f"- 条件: {len(stats)}/12 の (モデル, 分割点)、クリップ数 {n_clips}（論文は 30 本 × 30 フレーム）",
        f"- 判定: ✓ 許容幅以内 / △ 許容幅の 2 倍以内 / ✗ それ以上。許容幅: "
        + ", ".join(f"{k} {'±' + format(w, 'g') if kind == 'abs' else '±' + format(w * 100, 'g') + '%'}" for k, (kind, w) in TOL.items()),
        "- 注意: 論文の 30 クリップのリストは公開されておらず、`PAPER_CLIPS`（src/splitcomp/data/ucf101.py）は推測。"
        "そのため数値の完全な一致は期待できない。大小関係・傾向（本文の主張）が再現できているかを重視する",
        "",
        "## 判定のまとめ",
        "",
    ]
    summary = cmp_df.groupby("section").verdict.value_counts().unstack(fill_value=0)
    summary = summary.reindex(columns=["✓", "△", "✗"], fill_value=0)
    order = [s for s in SECTION_NOTE if s in summary.index]
    lines.append(_md_table([{"表・図": s, **{v: int(summary.loc[s, v]) for v in ["✓", "△", "✗"]}} for s in order], ["表・図", "✓", "△", "✗"]))
    nc = pd.Series([c["verdict"] for c in claims]).value_counts()
    lines += ["", f"本文の主張: ✓ {int(nc.get('✓', 0))} / ✗ {int(nc.get('✗', 0))} / —（12 条件・全条件がそろわず判定なし）{int(nc.get('—', 0))}",
              "", "## 本文の主張の検証", ""]
    lines.append(_md_table(claims, ["claim", "paper", "ours", "verdict"]))

    for section in order:
        d = cmp_df[cmp_df.section == section]
        lines += ["", f"## {section}", "", SECTION_NOTE[section], ""]
        fmt = ".2f" if section in ("Table 8", "Table 10", "Table 11", "Fig. 8", "packet_loss (no_loss)") else ".3f"
        lines.append(_md_table(d.to_dict("records"), ["item", "metric", "paper", "ours", "diff", "tol", "verdict"], fmt))

    lines += [
        "",
        "## 診断: DPCM / DwD の誤差（8-bit, 対称, P99）",
        "",
        "論文どおり I フレームも P フレームも P99 でクリップすると、I フレームで切られた大きな値の誤差は、DPCM では"
        "予測残差もクリップされるため少しずつしか取り戻せず、DwD では GoP の終わりまで残る（DnD は真の f_{t−1} を使うので残らない）。"
        "MSE の比が 1 より大きく、t が進んでも 1 に近づかないなら、この効果が効いている。図は `diag_mse_vs_gop.png`",
        "",
        _md_table(diagnosis_rows(stats, conds, pos), ["item", "max|f| / P99|f|", "MSE DPCM/DnD (t=1)", "MSE DPCM/DnD (t=29)",
                                                       "MSE DwD/DnD (t=29)", "agree DnD−DPCM", "agree DnD−DPCM (paper)"], ".2f"),
    ]
    lines += ["", "## 図", ""]
    for p in sorted(out.glob("*.png")):
        lines.append(f"![{p.stem}]({p.name})")
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  {out / 'report.md'}")
    print("\n判定のまとめ:\n" + summary.reindex(order).to_string())
    print(f"本文の主張: ✓ {int(nc.get('✓', 0))} / ✗ {int(nc.get('✗', 0))} / — {int(nc.get('—', 0))}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--target", action="append", metavar="NAME", help="計算する条件（モデル名・モデル名/分割点・Eff-S3 などの略称・番号）。複数回指定可。省略時は 12 条件全て")
    p.add_argument("--num-clips", type=int, default=None, help="クリップ数（既定: 論文と同じ 30 本）")
    p.add_argument("--quick", action="store_true", help="θ = 0 だけにする（EfficientNetV2-S の Table 10–12 は計算する）。論文の 480 条件の数え上げはできない")
    p.add_argument("--no-15fps", action="store_true", help="15fps の統計（Fig. 4）を計算しない")
    p.add_argument("--data-root", default="./data", help="UCF-101 の置き場所（無い動画は自動でダウンロード）")
    p.add_argument("--device", default="auto", help="head / tail のデバイス（auto / cuda / mps / cpu）")
    p.add_argument("--out", type=Path, help="結果の保存先（既存のフォルダなら、計算済みの target を飛ばして続きを計算する）")
    p.add_argument("--report-only", type=Path, metavar="RUN_DIR", help="計算せず、RUN_DIR の raw/ から報告書と図だけ作り直す")
    p.add_argument("--packet-loss-run", type=Path, metavar="DIR", help="packet_loss_sweep の実行フォルダ。ロスなしの行も Table 8 と比べる")
    p.add_argument("--no-report", action="store_true", help="計算だけして報告書は作らない（Slurm の配列ジョブで、条件ごとに別々に計算するとき）")
    args = p.parse_args()

    run_dir = args.report_only or compute(args)
    if args.no_report and not args.report_only:
        print(f"\n計算が終わりました。報告書は: python scripts/verify_shiba.py --report-only {run_dir}")
        return
    print("\n報告書を作ります")
    write_report(run_dir, args.packet_loss_run)


if __name__ == "__main__":
    main()

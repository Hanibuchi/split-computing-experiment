"""D+DZ と D+DZ+QI の量子化範囲を、実データの中間特徴のヒストグラムで比べる図を作る（解説資料用）。

1 クリップの先頭（I フレーム）と、その次のフレーム（P フレーム）の中間特徴について、
    1. I フレームの特徴 f_I のヒストグラムと、量子化範囲 [−R_I, R_I]（R_I = P99(|f_I|)）
    2. P フレームで送る差分 δ' = f_P − f̂_I のヒストグラムと、D+DZ の量子化範囲 [−R_P, R_P]（R_P = P99(|δ'|)）
    3. 同じ δ' のヒストグラムと、QI の量子化範囲（q を I フレームの q_I に固定するので [−R_I, R_I]）
を描く。3 枚とも同じ大きさ・同じ横軸（I フレームのスケール）にそろえ、並べて比べられるようにする。
q は splitcomp.temporal.TemporalEncoder が実際に使った値（EncodedFrame.q）をそのまま使う。

出力（--out。既定は docs/images/）:
    qi_hist_i_frame.png   図 1
    qi_hist_p_ddz.png     図 2
    qi_hist_p_qi.png      図 3
    qi_summary_table.png  統計（範囲 R、q、0 になった割合、エントロピー、範囲外の割合）の表
同じ表を標準出力にも表示する。

縦軸は線形（範囲外が 1% ほどしかないことが見た目でわかるように）。範囲外の帯には、その外に出た要素の割合を書く。
--log で縦軸を対数にすると、裾の分布（外れ値）が見やすくなる。

既定のモデルは efficientnet_v2_s / features_4（活性化が SiLU で、ブロックの出力は残差の足し算なので負の値も出る。
I フレームのヒストグラムがほぼ左右対称になり、対称量子化の範囲 ±R_I の見え方がわかりやすい）。
resnet50 / layer2 や vgg16 は ReLU の後なので値が 0 以上に偏る。

例:
    python scripts/qi_histogram.py
    python scripts/qi_histogram.py --log
    python scripts/qi_histogram.py --target resnet50 --bits 2 --clip 3 --out figs
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402
import torch  # noqa: E402

from splitcomp.builder import build_split  # noqa: E402
from splitcomp.config import PacketLossConfig, load_config  # noqa: E402
from splitcomp.data import build_dataset  # noqa: E402
from splitcomp.temporal import TemporalEncoder  # noqa: E402
from splitcomp.temporal.quantize import entropy_bits  # noqa: E402

# 色（dataviz スキルの参照パレット: 1 = 青, 2 = 橙, 8 = 赤）
C_HIST = "#2a78d6"  # ヒストグラム
C_RANGE = "#1baf7a"  # 量子化範囲（そのまま段に割り当てられる）
C_CLIP = "#eb6834"  # 範囲外（端の段に飽和 = クリップ）
C_DEAD = "#e34948"  # dead zone（0 になる）
C_INK = "#52514e"  # 補助線・文字

#: 日本語を表示できるフォント（手元にあるものだけを使う。無ければ DejaVu Sans で、日本語は豆腐になる）
_JA_FONTS = [f for f in ("Hiragino Sans", "Noto Sans CJK JP", "IPAexGothic") if f in {x.name for x in font_manager.fontManager.ttflist}]

plt.rcParams.update(
    {
        "font.family": _JA_FONTS + ["DejaVu Sans"],
        "font.size": 13,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": C_INK,
        "savefig.dpi": 200,
    }
)


def load_features(cfg: PacketLossConfig, target_name: str, clip: int, frames: int) -> tuple[torch.Tensor, str]:
    """設定の targets から target_name（モデル名）を選び、クリップ clip の先頭 frames フレームの中間特徴 [frames, C, H, W] を返す。"""
    targets = [t for t in cfg.targets if t.model.name == target_name or f"{t.model.name}/{t.split_point}" == target_name]
    if not targets:
        raise SystemExit(f"target {target_name!r} が設定にありません（{[t.model.name for t in cfg.targets]}）")
    target = targets[0]
    bundle = build_split(target.model, target.split_point)
    head = bundle.head.eval()
    x, _ = build_dataset(cfg.data)[clip]
    with torch.no_grad():
        (feats,) = head(x[:frames])
    return feats.float().cpu(), f"{target.model.name} / {target.split_point}"


def encode_ip(feats: torch.Tensor, bits: int, p_step: str):
    """I フレーム（feats[0]）と P フレーム（feats[1]）を D+DZ（p_step="adaptive"）または QI（"intra"）で符号化する。

    Returns:
        (I フレームの EncodedFrame, P フレームの EncodedFrame, P フレームで量子化した差分 δ' = f_P − f̂_I)
    """
    enc = TemporalEncoder("dpcm", bits, p_quant="dz", p_step=p_step)
    fi = enc.encode(feats[0])
    i_hat = enc.history[0][1]  # 端末が想定するクラウドの f̂_I（P フレームの参照）
    fp = enc.encode(feats[1])
    return fi, fp, feats[1] - i_hat


def index_stats(idx: torch.Tensor, x: np.ndarray, R: float) -> dict:
    """量子化インデックスの統計（0 の割合・エントロピー）と、量子化した値 x のうち範囲 ±R の外に出た割合。

    範囲外の要素は端の段に飽和（クリップ）する。インデックスが端の値になった割合（列 clamp_rate）とは少し違う
    （端の段の内側の値も端のインデックスになるため）が、図の「範囲外」の帯と同じ数え方にしている。
    """
    return {
        "zero": float((idx == 0).float().mean()),
        "entropy": entropy_bits(idx),
        "outside": float((np.abs(x) > R).mean()),
    }


def _hist(ax, values: np.ndarray, edges: np.ndarray, log: bool = False) -> None:
    """度数のヒストグラム（log=True なら縦軸を対数にする）。"""
    ax.hist(values, bins=edges, color=C_HIST, alpha=0.85, linewidth=0)
    if log:
        ax.set_yscale("log")
    ax.set_xlabel("値")
    ax.set_ylabel("度数（対数）" if log else "度数")
    ax.set_xlim(edges[0], edges[-1])


def _outside_share(ax, values: np.ndarray, R: float, lim: float) -> None:
    """範囲外の帯（−lim〜−R と R〜lim）の中に、そこに入る要素の割合を書く（範囲の外に出た要素は全てここに数える）。"""
    for sign in (-1, 1):
        share = float((sign * values > R).mean())
        ax.text(sign * (R + lim) / 2, 0.5, f"範囲外\n{share:.1%}", transform=ax.get_xaxis_transform(), ha="center",
                va="center", fontsize=14, fontweight="bold", color=C_CLIP)


def _shade_range(ax, R: float, xlim: tuple[float, float], label: str) -> None:
    """量子化範囲 [−R, R] を緑、範囲外（クリップされる）を橙で塗る。"""
    ax.axvspan(-R, R, color=C_RANGE, alpha=0.15, lw=0, label=label)
    if xlim[0] < -R:
        ax.axvspan(xlim[0], -R, color=C_CLIP, alpha=0.12, lw=0)
    ax.axvspan(R, xlim[1], color=C_CLIP, alpha=0.12, lw=0, label="範囲外（端の段に飽和）")
    for v in (-R, R):
        ax.axvline(v, color=C_RANGE, lw=1.5)


def _levels(ax, q: float, bits: int, dead_zone: bool) -> None:
    """量子化の段の境界を細線で描く（dead-zone なら |x| < 0.75q が 0 の段）。"""
    half = 1 << (bits - 1)
    if dead_zone:
        # idx = sign(x)·floor(|x|/q + 0.25): 境界は ±(k − 0.25)q（k = 1, 2, ...）
        bounds = [s * (k - 0.25) * q for k in range(1, half + 1) for s in (1, -1)]
    else:
        # idx = round(x/q): 境界は (k + 0.5)q
        bounds = [(k + 0.5) * q for k in range(-half, half - 1)]
    for b in bounds:
        ax.axvline(b, color=C_INK, lw=0.4, alpha=0.5)


def _dead_zone(ax, q: float) -> None:
    ax.axvspan(-0.75 * q, 0.75 * q, color=C_DEAD, alpha=0.25, lw=0, label="dead zone（|x| < 0.75q → 0）")


def _note(ax, text: str) -> None:
    ax.text(0.02, 0.97, text, transform=ax.transAxes, va="top", ha="left", fontsize=10, color="#0b0b0b",
            bbox={"facecolor": "white", "edgecolor": C_INK, "alpha": 0.9, "boxstyle": "round,pad=0.4"})


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/packet_loss_ucf101_qfixed.yaml", help="データとモデルを読む設定ファイル")
    p.add_argument("--target", default="efficientnet_v2_s", help="モデル名（または モデル名/分割点）")
    p.add_argument("--bits", type=int, default=4, help="量子化ビット数")
    p.add_argument("--clip", type=int, default=0, help="使うクリップの番号")
    p.add_argument("--out", type=Path, default=Path("docs/images"), help="図の保存先")
    p.add_argument("--log", action="store_true", help="縦軸を対数にする（既定は線形）")
    args = p.parse_args()

    cfg = load_config(PacketLossConfig, args.config, [])
    feats, title = load_features(cfg, args.target, args.clip, frames=2)
    b = args.bits

    fi, p_ddz, delta = encode_ip(feats, b, "adaptive")
    _, p_qi, delta_qi = encode_ip(feats, b, "intra")
    assert torch.equal(delta, delta_qi)  # I フレームは同じなので、P フレームで送る差分も同じ
    q_i, q_ddz, q_qi = fi.q, p_ddz.q, p_qi.q
    R_i, R_ddz = q_i * (1 << b) / 2, q_ddz * (1 << b) / 2  # q = 2R / 2^b

    f_i = feats[0].flatten().numpy()
    d = delta.flatten().numpy()

    # 横軸は 3 枚とも I フレームのスケール ±1.6·R_I にそろえる（ごく一部の外れ値まで入れると、分布が中央に潰れるため）。
    # 横軸の外に出た要素の数は注記に書く
    lim = 1.6 * R_i
    xlim = (-lim, lim)
    edges = np.linspace(-lim, lim, 241)
    args.out.mkdir(parents=True, exist_ok=True)

    def outside(v: np.ndarray) -> str:
        n = int((np.abs(v) > lim).sum())
        return f"横軸の外: {n} 個（最大 {np.abs(v).max():.1f}）" if n else ""

    # 1. I フレーム
    st_i = index_stats(fi.idx, f_i, R_i)
    fig, ax = plt.subplots(figsize=(14, 5.6))
    _shade_range(ax, R_i, xlim, "量子化範囲 ±R_I（R_I = P99(|f_I|)）")
    _levels(ax, q_i, b, dead_zone=False)
    _hist(ax, f_i, edges, args.log)
    _outside_share(ax, f_i, R_i, lim)
    ax.set_title(f"I フレームの中間特徴 $f_I$（{title}, {b} bit, 対称量子化）")
    _note(ax, f"R_I = P99(|f_I|) = {R_i:.3f}\nq_I = 2R_I / 2^{b} = {q_i:.4f}\n細線 = 量子化の段の境界（{1 << b} 段）\n"
              f"範囲外（端の段に飽和）: {st_i['outside']:.1%}\n{outside(f_i)}".rstrip())
    ax.legend(loc="upper right", fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(args.out / "qi_hist_i_frame.png")
    plt.close(fig)

    def p_figure(path: Path, R: float, q: float, label: str, name: str, note: str) -> None:
        """P フレームの差分 δ' のヒストグラム（I フレームの図と同じ大きさ・同じ横軸）。"""
        fig, ax = plt.subplots(figsize=(14, 5.6))
        _shade_range(ax, R, xlim, label)
        _dead_zone(ax, q)
        _levels(ax, q, b, dead_zone=True)
        _hist(ax, d, edges, args.log)
        _outside_share(ax, d, R, lim)
        ax.set_title(f"P フレームの差分 $\\delta' = f_P - \\hat{{f}}_I$：{name}（横軸は I と同じ）")
        _note(ax, note + (f"\n{outside(d)}" if outside(d) else ""))
        ax.legend(loc="upper right", fontsize=9, frameon=True, framealpha=0.9)

        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)

    # 2. P フレーム（D+DZ: 範囲を差分の P99 に縮める）
    st_d = index_stats(p_ddz.idx, d, R_ddz)
    p_figure(
        args.out / "qi_hist_p_ddz.png", R_ddz, q_ddz, "D+DZ の量子化範囲 ±R_P（R_P = P99(|δ'|)）", "D+DZ",
        f"R_P = P99(|δ'|) = {R_ddz:.3f}（R_I の {R_ddz / R_i:.0%}）\nq_P = 2R_P / 2^{b} = {q_ddz:.4f}\n"
        f"→ 狭い範囲を {1 << b} 段で細かく刻む\n0 になる割合: {st_d['zero']:.0%}",
    )

    # 3. P フレーム（QI: q を I フレームの q_I に固定する）
    st_q = index_stats(p_qi.idx, d, R_i)
    p_figure(
        args.out / "qi_hist_p_qi.png", R_i, q_qi, "QI の量子化範囲 ±R_I（q = q_I に固定）", "QI",
        f"q_P = q_I = {q_qi:.4f}（範囲は I と同じ ±{R_i:.3f}）\ndead zone ±0.75q_I = ±{0.75 * q_qi:.3f}\n"
        f"→ 粗い段のまま。大半が dead zone に入る\n0 になる割合: {st_q['zero']:.0%}",
    )

    # 統計の表
    rows = [("I フレーム", R_i, q_i, st_i), ("P: D+DZ", R_ddz, q_ddz, st_d), ("P: QI", R_i, q_qi, st_q)]
    print(f"{title}, {b} bit, クリップ {args.clip}（要素数 {f_i.size}）")
    print(f"{'':12s} {'範囲R':>8s} {'q':>8s} {'0の割合':>8s} {'エントロピー':>10s} {'範囲外':>8s}")
    for name, R, q, s in rows:
        print(f"{name:12s} {R:8.4f} {q:8.4f} {s['zero']:8.1%} {s['entropy']:10.3f} {s['outside']:8.2%}")
    save_table(args.out / "qi_summary_table.png", rows, f"{title}, {b} bit, UCF-101 クリップ {args.clip}（エントロピーは bit/要素）")
    print(f"図を保存しました: {args.out}/qi_hist_*.png, qi_summary_table.png")


def save_table(path: Path, rows: list, caption: str) -> None:
    """統計の表を画像にする（スライドに貼る用）。QI の行の「0 になる割合」と「エントロピー」を強調する。"""
    cells = [
        [name.replace("P: ", "P："), f"{R:.2f}" + (" (= I)" if name == "P: QI" else " (P99)"),
         f"{q:.3f}" + (" (= q_I)" if name == "P: QI" else ""), f"{s['zero']:.1%}", f"{s['entropy']:.2f}", f"{s['outside']:.2%}"]
        for name, R, q, s in rows
    ]
    fig, ax = plt.subplots(figsize=(14, 3.6))
    ax.axis("off")
    t = ax.table(cellText=cells, colLabels=["", "範囲 R", "ステップ幅 q", "0 になる割合", "エントロピー", "範囲外"],
                 loc="center", cellLoc="center", colWidths=[0.16, 0.17, 0.19, 0.16, 0.16, 0.13])
    t.auto_set_font_size(False)
    t.set_fontsize(20)
    t.scale(1, 3.2)
    for (r, c), cell in t.get_celld().items():
        cell.set_edgecolor("#c3c2b7")
        if r == 0:
            cell.set_facecolor("#eef3fb")
            cell.set_text_props(weight="bold")
        if r == len(cells):  # QI の行
            cell.set_facecolor("#fdecec")
            if c in (3, 4):
                cell.set_text_props(weight="bold", color="#c0392b")
    ax.set_title(caption, fontsize=15, color=C_INK)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


if __name__ == "__main__":
    main()

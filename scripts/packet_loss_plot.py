"""packet_loss_sweep の結果（results.csv / gop_curve.csv）を可視化し、要約表を書き出す。

出力（既定は <run_dir>/figures/）:
    agreement_vs_loss_<損失モデル>.png  損失率ごとの一致率（行=モデル、列=ビット数、線=方式）
    agreement_drop_heatmap.png          ロスなしからの一致率の低下（モデル平均）
    dpcm_vs_fd.png                      D+DZ と F+Sym の一致率の差（0 を下回ると DPCM が負ける）
    gop_curve_<モデル>.png              GoP 内の位置ごとの一致率と特徴 MSE（ロスの伝搬）
    rate_vs_agreement.png               送信量（パケットごとにエントロピー符号化した実際の送信量）と一致率
    loss_rate_check.png                 指定した損失率と実際のパケット損失率
    summary.md                          ロスなしの表、損失モデル別の表、DPCM が FD を下回る損失率

例:
    python scripts/packet_loss_plot.py runs/packet_loss_ucf101_merged
    python scripts/packet_loss_plot.py runs/packet_loss_ucf101_merged -o figs --bits 2
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter  # noqa: E402

# 方式の表示順と色・マーカー（すべての図で共通）。結果に無い方式は main で除く
SCHEMES = ["F+Sym", "D+Sym", "D+DZ", "D+DZ+R", "D+DZ+S2", "D+DZ+S5", "D+DZ+PRS", "D+DZ+PRS3", "D+DZ+FEC10", "D+DZ+FEC25"]
SCHEME_STYLE = {
    "F+Sym": {"color": "#2a78d6", "marker": "o"},
    "D+Sym": {"color": "#1baf7a", "marker": "^"},
    "D+DZ": {"color": "#eda100", "marker": "D"},
    "D+DZ+R": {"color": "#c2185b", "marker": "s"},
    "D+DZ+S2": {"color": "#7b4fd6", "marker": "v"},  # 多系列 DPCM（2 系列）
    "D+DZ+S5": {"color": "#6d4c41", "marker": "P"},  # 多系列 DPCM（5 系列）
    "D+DZ+PRS": {"color": "#00897b", "marker": "X"},  # PRS（NACK が 1 フレーム後に届く）
    "D+DZ+PRS3": {"color": "#5c6bc0", "marker": "*"},  # PRS（NACK が 3 フレーム後に届く）
    "D+DZ+FEC10": {"color": "#e53935", "marker": "h"},  # FEC（パリティ 10%）
    "D+DZ+FEC25": {"color": "#546e7a", "marker": "p"},  # FEC（パリティ 25%）
}
# 3 系列までの色（ビット数・損失モデル用）
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
LOSS_MODELS = ["bernoulli", "gilbert_elliott", "frame"]
LOSS_MODEL_LABEL = {"bernoulli": "Bernoulli", "gilbert_elliott": "Gilbert-Elliott (L=4)", "frame": "Frame loss"}
# GoP 曲線で並べる代表的な損失条件
GOP_LOSSES = ["no_loss", "bernoulli(p=0.05)", "bernoulli(p=0.2)", "gilbert_elliott(p=0.2,L=4)"]

plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 150,
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#8a8984",
    "axes.grid": True,
    "grid.color": "#e4e3df",
    "grid.linewidth": 0.6,
    "lines.linewidth": 1.6,
    "lines.markersize": 4,
    "legend.frameon": False,
})


# ---------------------------------------------------------------- 読み込み

def load(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """results と gop_curve を読む。gop_curve には results の loss_model / loss_p を付ける。"""
    df = pd.read_csv(run_dir / "results.csv")
    df["loss_p"] = df["loss_p"].astype(float)
    df["num_elements"] = df["feature_shape"].map(lambda s: math.prod(int(v) for v in str(s).split("x")))
    # エントロピー符号化した場合の 1 フレームあたりの送信量（参考値）
    df["entropy_kb_per_frame"] = df["entropy_bpe"] * df["num_elements"] / 8 / 1024
    # 図に使う送信量: パケットごとにエントロピー符号化した実際の送信量（ヘッダ込み）。
    #   coding 列の無い古い結果（固定長で送っていた版）は、上のエントロピーからの理想値を使う
    df["rate_kb_per_frame"] = df["bytes_per_frame"] / 1024 if "coding" in df else df["entropy_kb_per_frame"]

    gop_path = run_dir / "gop_curve.csv"
    gop = None
    if gop_path.exists():
        gop = pd.read_csv(gop_path)
        loss_map = df[["loss", "loss_model", "loss_p"]].drop_duplicates("loss")
        gop = gop.merge(loss_map, on="loss", how="left")
    return df, gop


def models_of(df: pd.DataFrame) -> list[str]:
    """モデル名（実行に現れた順）。"""
    return list(dict.fromkeys(df["model"]))


def by_loss_model(df: pd.DataFrame, m: str) -> pd.DataFrame:
    """損失モデル m の行と、基準になるロスなし（loss_p = 0）の行。損失モデルどうしは混ぜない。"""
    return df[(df.loss_model == m) | (df.loss_p == 0)]


def scheme_legend(fig: plt.Figure) -> None:
    handles = [plt.Line2D([], [], **SCHEME_STYLE[s], label=s) for s in SCHEMES]
    fig.legend(handles=handles, loc="upper center", ncol=len(SCHEMES), bbox_to_anchor=(0.5, 1.0))


def save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path}")


# ---------------------------------------------------------------- 図

def plot_agreement_vs_loss(df: pd.DataFrame, out_dir: Path) -> None:
    """損失モデルごとに、行=モデル・列=ビット数のパネルで 損失率 → 一致率 を描く。"""
    models, bits_list = models_of(df), sorted(df.bits.unique())
    for m in LOSS_MODELS:
        sub = by_loss_model(df, m)
        if sub.loss_p.max() == 0:
            continue
        fig, axes = plt.subplots(len(models), len(bits_list), figsize=(3.2 * len(bits_list), 2.4 * len(models)),
                                 sharex=True, sharey="row", squeeze=False)
        for i, model in enumerate(models):
            for j, bits in enumerate(bits_list):
                ax = axes[i, j]
                for s in SCHEMES:
                    g = sub[(sub.model == model) & (sub.bits == bits) & (sub.scheme == s)].sort_values("loss_p")
                    if g.empty:
                        continue
                    ax.plot(g.loss_p, g.agreement, **SCHEME_STYLE[s])
                    ax.fill_between(g.loss_p, g.agreement - g.agreement_ci95, g.agreement + g.agreement_ci95,
                                    color=SCHEME_STYLE[s]["color"], alpha=0.12, linewidth=0)
                if i == 0:
                    ax.set_title(f"{bits} bit")
                if j == 0:
                    ax.set_ylabel(f"{model}\nagreement")
                if i == len(models) - 1:
                    ax.set_xlabel("loss probability p")
        fig.suptitle(f"Top-1 agreement vs. packet loss — {LOSS_MODEL_LABEL[m]} (band: 95% CI)", y=1.03)
        scheme_legend(fig)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        save(fig, out_dir / f"agreement_vs_loss_{m}.png")


def plot_drop_heatmap(df: pd.DataFrame, out_dir: Path) -> None:
    """方式 × 損失条件の agreement_drop（モデル平均）。ビット数ごとにパネルを分ける。"""
    lossy = df[df.loss_p > 0]
    order = (lossy[["loss", "loss_model", "loss_p"]].drop_duplicates()
             .assign(k=lambda x: x.loss_model.map(LOSS_MODELS.index)).sort_values(["k", "loss_p"])["loss"].tolist())
    bits_list = sorted(df.bits.unique())
    tables = [lossy[lossy.bits == b].pivot_table(index="scheme", columns="loss", values="agreement_drop")
              .reindex(index=SCHEMES, columns=order) for b in bits_list]
    vmax = max(np.nanmax(t.values) for t in tables)
    fig, axes = plt.subplots(len(bits_list), 1, figsize=(0.62 * len(order) + 2, 1.5 * len(bits_list) + 1.2),
                             squeeze=False)
    for ax, b, t in zip(axes[:, 0], bits_list, tables):
        im = ax.imshow(t.values, cmap="Blues", vmin=0, vmax=vmax, aspect="auto")
        ax.grid(False)
        ax.set_yticks(range(len(SCHEMES)), SCHEMES)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([])

        ax.set_title(f"{b} bit")
        for (r, c), v in np.ndenumerate(t.values):
            if not np.isnan(v):
                ax.text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if v > vmax * 0.6 else "#0b0b0b")
    axes[-1, 0].set_xticklabels([o.replace("gilbert_elliott", "GE").replace(",L=4", "") for o in order],
                                rotation=45, ha="right")
    fig.colorbar(im, ax=axes[:, 0].tolist(), label="agreement drop vs. no loss (mean over models)", shrink=0.8)
    fig.suptitle("Agreement drop caused by packet loss")
    save(fig, out_dir / "agreement_drop_heatmap.png")


def dpcm_minus_fd(df: pd.DataFrame) -> pd.DataFrame:
    """(model, bits, loss_model, loss_p) ごとの agreement(D+DZ) - agreement(F+Sym)。"""
    rows = []
    for m in LOSS_MODELS:
        sub = by_loss_model(df, m)
        p = sub.pivot_table(index=["model", "bits", "loss_p"], columns="scheme", values="agreement")
        if not {"D+DZ", "F+Sym"} <= set(p.columns):
            continue
        d = (p["D+DZ"] - p["F+Sym"]).rename("diff").reset_index()
        d["loss_model"] = m
        rows.append(d)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def crossover_table(diff: pd.DataFrame) -> pd.DataFrame:
    """D+DZ が F+Sym を下回る最小の loss_p（下回らなければ空欄）。"""
    def first_below(g: pd.DataFrame) -> float:
        below = g[g["diff"] < 0].loss_p
        return below.min() if not below.empty else np.nan

    t = diff.groupby(["model", "bits", "loss_model"]).apply(first_below, include_groups=False).unstack("loss_model")
    return t.reindex(columns=[m for m in LOSS_MODELS if m in t.columns])


def plot_dpcm_vs_fd(diff: pd.DataFrame, models: list[str], out_dir: Path) -> None:
    """行=損失モデル、列=モデルで、D+DZ − F+Sym を損失率に対して描く（線=ビット数）。"""
    if diff.empty:
        return
    bits_list = sorted(diff.bits.unique())
    lms = [m for m in LOSS_MODELS if m in diff.loss_model.unique()]
    fig, axes = plt.subplots(len(lms), len(models), figsize=(2.9 * len(models), 2.3 * len(lms)),
                             sharex=True, sharey=True, squeeze=False)
    for i, m in enumerate(lms):
        for j, model in enumerate(models):
            ax = axes[i, j]
            ax.axhline(0, color="#52514e", linewidth=0.8)
            for k, b in enumerate(bits_list):
                g = diff[(diff.loss_model == m) & (diff.model == model) & (diff.bits == b)].sort_values("loss_p")
                ax.plot(g.loss_p, g["diff"], marker="o", color=SERIES_COLORS[k % 3], label=f"{b} bit")
            if i == 0:
                ax.set_title(model)
            if j == 0:
                ax.set_ylabel(f"{LOSS_MODEL_LABEL[m]}\nD+DZ − F+Sym")
            if i == len(lms) - 1:
                ax.set_xlabel("loss probability p")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(bits_list), bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("Agreement advantage of DPCM (D+DZ) over FD (F+Sym); below 0 = FD wins", y=1.03)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save(fig, out_dir / "dpcm_vs_fd.png")


def plot_gop_curves(gop: pd.DataFrame, out_dir: Path, bits: int) -> None:
    """モデルごとに、GoP 内の位置 → 一致率・特徴 MSE を、代表的な損失条件ごとに描く。"""
    losses = [l for l in GOP_LOSSES if l in set(gop.loss)]
    for model in models_of(gop):
        sub = gop[(gop.model == model) & (gop.bits == bits)]
        if sub.empty:
            continue
        fig, axes = plt.subplots(2, len(losses), figsize=(3.0 * len(losses), 4.6), sharex=True, sharey="row",
                                 squeeze=False)
        for j, loss in enumerate(losses):
            for s in SCHEMES:
                g = sub[(sub.loss == loss) & (sub.scheme == s)].sort_values("gop_position")
                if g.empty:
                    continue
                style = {**SCHEME_STYLE[s], "markersize": 3}
                axes[0, j].plot(g.gop_position, g.agreement, **style)
                axes[1, j].plot(g.gop_position, g.feature_mse, **style)
            axes[0, j].set_title(loss.replace(",L=4", ", L=4"))
            axes[1, j].set_xlabel("position in GoP (0 = I-frame)")
            axes[1, j].set_yscale("log")
        axes[0, 0].set_ylabel("agreement")
        axes[1, 0].set_ylabel("feature MSE (log)")
        fig.suptitle(f"Loss propagation within a GoP — {model}, {bits} bit", y=1.04)
        scheme_legend(fig)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        save(fig, out_dir / f"gop_curve_{model}.png")


def plot_rate_vs_agreement(df: pd.DataFrame, out_dir: Path, lossy: str = "bernoulli(p=0.05)") -> None:
    """送信量（rate_kb_per_frame）→ 一致率。ロスなしと lossy を並べる。点はビット数（2/4/8）。"""
    models = models_of(df)
    conds = [c for c in ["no_loss", lossy] if c in set(df.loss)]
    fig, axes = plt.subplots(len(conds), len(models), figsize=(2.9 * len(models), 2.5 * len(conds)),
                             sharey=True, squeeze=False)
    for i, cond in enumerate(conds):
        for j, model in enumerate(models):
            ax = axes[i, j]
            for s in SCHEMES:
                g = df[(df.model == model) & (df.loss == cond) & (df.scheme == s)].sort_values("bits")
                ax.plot(g.rate_kb_per_frame, g.agreement, **SCHEME_STYLE[s])
            ax.set_xscale("log")
            ax.xaxis.set_major_locator(LogLocator(subs=(1, 2, 5)))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            ax.xaxis.set_minor_formatter(NullFormatter())
            if i == 0:
                ax.set_title(model)
            if j == 0:
                ax.set_ylabel(f"{cond}\nagreement")
            if i == len(conds) - 1:
                ax.set_xlabel("size [KB/frame]")
    fig.suptitle("Rate vs. agreement (points: 2 / 4 / 8 bit)", y=1.04)
    scheme_legend(fig)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, out_dir / "rate_vs_agreement.png")


def plot_loss_rate_check(df: pd.DataFrame, out_dir: Path) -> None:
    """指定した loss_p と、実際に観測したパケット損失率 loss_rate。"""
    lossy = df[df.loss_p > 0]
    fig, ax = plt.subplots(figsize=(4, 3.6))
    lim = max(lossy.loss_p.max(), lossy.loss_rate.max()) * 1.08
    ax.plot([0, lim], [0, lim], color="#8a8984", linewidth=0.8, linestyle="--", label="y = x")
    rng = np.random.default_rng(0)
    for k, m in enumerate(LOSS_MODELS):
        g = lossy[lossy.loss_model == m]
        jitter = rng.uniform(-0.002, 0.002, len(g))  # 同じ loss_p の点が重ならないよう少しずらす
        ax.scatter(g.loss_p + jitter, g.loss_rate, s=14, color=SERIES_COLORS[k], edgecolor="white", linewidth=0.5,
                   label=LOSS_MODEL_LABEL[m])
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("configured loss probability p")
    ax.set_ylabel("observed packet loss rate")
    ax.set_title("Loss model sanity check")
    ax.legend()
    save(fig, out_dir / "loss_rate_check.png")


# ---------------------------------------------------------------- 要約

def md_table(t: pd.DataFrame, digits: int = 3) -> str:
    """DataFrame（多重インデックス可）を Markdown の表にする。"""
    t = t.reset_index()
    cols = [str(c) for c in t.columns]

    def fmt(v: object) -> str:
        if isinstance(v, float):
            return "" if np.isnan(v) else f"{round(v, digits):g}"
        return str(v)

    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in t.itertuples(index=False)]
    return "\n".join(lines)


def write_summary(df: pd.DataFrame, diff: pd.DataFrame, run_dir: Path, path: Path) -> None:
    clean = df[df.loss_p == 0]
    parts = [f"# packet loss 実験の要約: `{run_dir}`", ""]
    parts += ["## ロスなし（loss_p = 0）の一致率", "", "論文 Table 8 の FD（F+Sym）/ DPCM（D+DZ）と比べる。", "",
              md_table(clean.pivot_table(index=["model", "bits"], columns="scheme", values="agreement")
                       .reindex(columns=SCHEMES)), ""]
    parts += ["## ロスなしのエントロピー [bit/要素]", "",
              md_table(clean.pivot_table(index=["model", "bits"], columns="scheme", values="entropy_bpe")
                       .reindex(columns=SCHEMES)), ""]
    parts += ["## D+DZ が F+Sym を下回り始める損失率", "",
              "空欄は、評価した範囲では下回らなかったことを表す。", ""]
    if not diff.empty:
        parts += [md_table(crossover_table(diff)), ""]
    for m in LOSS_MODELS:
        sub = by_loss_model(df, m)
        if sub.loss_p.max() == 0:
            continue
        parts += [f"## 一致率: {LOSS_MODEL_LABEL[m]}", "",
                  md_table(sub.pivot_table(index=["model", "bits", "loss_p"], columns="scheme", values="agreement")
                           .reindex(columns=SCHEMES)), ""]
    path.write_text("\n".join(parts), encoding="utf-8")
    print(f"  {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path, help="results.csv のある実行フォルダ（merge_results.py の出力など）")
    parser.add_argument("-o", "--out", type=Path, help="出力フォルダ（既定: <run_dir>/figures）")
    parser.add_argument("--bits", type=int, default=4, help="GoP 曲線で使うビット数（既定: 4）")
    args = parser.parse_args()

    out_dir = args.out or args.run_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    df, gop = load(args.run_dir)
    print(f"{len(df)} 行（モデル: {', '.join(models_of(df))}）を読みました。出力先: {out_dir}")
    SCHEMES[:] = [s for s in SCHEMES if s in set(df.scheme)]  # 結果に無い方式は凡例・表に出さない

    plot_agreement_vs_loss(df, out_dir)
    plot_drop_heatmap(df, out_dir)
    diff = dpcm_minus_fd(df)
    plot_dpcm_vs_fd(diff, models_of(df), out_dir)
    if gop is not None:
        plot_gop_curves(gop, out_dir, args.bits)
    plot_rate_vs_agreement(df, out_dir)
    plot_loss_rate_check(df, out_dir)
    write_summary(df, diff, args.run_dir, out_dir / "summary.md")


if __name__ == "__main__":
    main()

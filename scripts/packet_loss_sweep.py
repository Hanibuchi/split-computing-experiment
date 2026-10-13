"""動画の時系列符号化（FD / DwD / DPCM、dead-zone 量子化）をパケットロス下で評価するシミュレーション実験。

参考: H. Shiba et al., "Delta Coding With Dead-Zone Quantization for Bandwidth-Efficient
Split Computing on Video Streams," IEEE Access, 2026（パケットロスは同論文 §VII-E-5 で今後の課題とされている）。

各 (モデル, 分割点) × 方式 × ビット数 × 損失モデル × シードについて、クリップごとに
    head → 時系列符号化（端末）→ パケット分割 → ロス → 復号・concealment（クラウド）→ tail
を行い、量子化もロスもない元モデルの出力との top-1 一致率（agreement）などを集計する。
端末はロスを知らない（フィードバックなし）ので、DPCM のロスは次の I フレームまで伝搬する。

出力（runs/<実験名>_<日時>/。--target で絞った場合は runs/<実験名>_<モデル名>_<日時>/）:
    results.csv    条件ごとの集計（agreement, 95% CI, 特徴 MSE, 実際の損失率, 送信量, エントロピー ...）
    gop_curve.csv  GoP 内の位置 t ごとの agreement と特徴 MSE（誤差伝搬の様子）
    *.jsonl        同じ内容。モデル 1 つ分が終わるたびに追記されるので、途中で止まっても終わった分は残る

例:
    python scripts/packet_loss_sweep.py --config configs/packet_loss_ucf101.yaml
    python scripts/packet_loss_sweep.py --config configs/packet_loss_ucf101.yaml data.kwargs.num_clips=2 'seeds=[0]'

    # モデルごとに別々に実行する（時間制限のある計算機向け）。結果は scripts/merge_results.py でまとめる
    python scripts/packet_loss_sweep.py --config configs/packet_loss_ucf101.yaml --list-targets
    python scripts/packet_loss_sweep.py --config configs/packet_loss_ucf101.yaml --target resnet50
"""

from __future__ import annotations

import math
import statistics
import time
from collections import defaultdict

import numpy as np
import torch

from splitcomp.builder import build_split
from splitcomp.cli import parse_cli
from splitcomp.config import PacketLossConfig, to_dict
from splitcomp.data import build_dataset
from splitcomp.metrics import ResultWriter, make_run_dir, save_run_info
from splitcomp.temporal import (
    TemporalDecoder,
    TemporalEncoder,
    build_loss_model,
    entropy_bits,
    frame_bytes,
    num_packets,
    packet_assignment,
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
        self.pos_agree = np.zeros(gop)
        self.pos_mse = np.zeros(gop)
        self.pos_count = np.zeros(gop)


def _clip_seed(seed: int, clip: int) -> int:
    return seed * 100_003 + clip


@torch.no_grad()
def run_clip(feats, ref_pred, tail, tail_dev, cfg, scheme, bits, loss_cfg, seed, clip, acc: _Accumulator) -> None:
    """1 クリップを符号化 → ロス → 復号して tail を通し、acc に加算する。"""
    T = feats.shape[0]
    n = feats[0].numel()
    assign = packet_assignment(n, bits, cfg.mtu, cfg.interleave)
    P = num_packets(n, bits, cfg.mtu)
    loss = build_loss_model(loss_cfg, _clip_seed(seed, clip))
    enc = TemporalEncoder(scheme.scheme, bits, cfg.gop, scheme.p_quant)
    dec = TemporalDecoder(cfg.concealment)

    recon = torch.empty_like(feats)
    for t in range(T):
        frame = enc.encode(feats[t])
        lost = loss.drop(P)
        recon[t] = dec.decode(frame, received_mask(lost, assign) if lost.any() else None)
        acc.lost_packets += int(lost.sum())
        acc.packets += P
        acc.bytes += frame_bytes(n, bits, cfg.mtu)
        acc.entropy += entropy_bits(frame.idx)

    pred = tail(recon.to(tail_dev)).argmax(-1).cpu()
    agree = (pred == ref_pred).double()
    mse = (recon - feats).pow(2).flatten(1).mean(1).double()
    pos = torch.arange(T) % cfg.gop
    acc.agree += int(agree.sum())
    acc.frames += T
    acc.mse += float(mse.sum())
    np.add.at(acc.pos_agree, pos.numpy(), agree.numpy())
    np.add.at(acc.pos_mse, pos.numpy(), mse.numpy())
    np.add.at(acc.pos_count, pos.numpy(), 1)


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


def main() -> None:
    def extra(p):
        p.add_argument("--target", action="append", metavar="NAME", help="実行する target（モデル名・モデル名/分割点・番号）。複数回指定可。省略時は全て")
        p.add_argument("--list-targets", action="store_true", help="設定にある target を番号付きで表示して終了する")

    cfg, args = parse_cli(PacketLossConfig, __doc__, extra_args=extra)
    if args.list_targets:
        for i, t in enumerate(cfg.targets):
            print(f"{i}: {target_name(t)}")
        return
    cfg.targets = select_targets(cfg.targets, args.target)
    set_seed(cfg.seed)
    head_dev, tail_dev = resolve_device(cfg.devices.head), resolve_device(cfg.devices.tail)
    # 一部の target だけを実行する場合は、どのモデルの結果か分かるようにフォルダ名に入れる
    name = cfg.experiment_name
    if args.target:
        name += "_" + "+".join(dict.fromkeys(t.model.name for t in cfg.targets))
    run_dir = make_run_dir(cfg.output_dir, name)
    save_run_info(run_dir, to_dict(cfg), {"head_device": str(head_dev), "tail_device": str(tail_dev)})
    print(f"結果の保存先: {run_dir}")

    dataset = build_dataset(cfg.data)
    print(f"クリップ {len(dataset)} 本を評価します（GoP={cfg.gop}, MTU={cfg.mtu}, concealment={cfg.concealment}）")

    # 損失率 0 の条件はシードによらないので 1 回だけ実行する
    conditions = []
    for loss_cfg in cfg.losses:
        for seed in cfg.seeds if loss_cfg.p > 0 else cfg.seeds[:1]:
            conditions.append((loss_cfg, seed))

    summary = []
    with ResultWriter(run_dir) as writer, ResultWriter(run_dir, "gop_curve") as curve_writer:
        for target in cfg.targets:
            bundle = build_split(target.model, target.split_point)
            if not bundle.point.is_clean:
                raise ValueError(f"{target.model.name} の分割点 {target.split_point!r} は送信テンソルが複数あります（clean な点を指定してください）")
            head = bundle.head.to(head_dev).eval()
            tail = bundle.tail.to(tail_dev).eval()
            model = bundle.model.to(tail_dev).eval()
            print(f"\n=== {target.model.name} / {target.split_point}")

            # acc[(方式名, ビット数, 損失名, シード)]
            accs: dict[tuple, _Accumulator] = defaultdict(lambda: _Accumulator(cfg.gop))
            started = time.perf_counter()
            for clip in range(len(dataset)):
                x, _ = dataset[clip]
                with torch.no_grad():
                    ref_pred = model(x.to(tail_dev)).argmax(-1).cpu()
                    (feats,) = head(x.to(head_dev))
                feats = feats.float().cpu()
                for scheme in cfg.schemes:
                    for bits in cfg.bits:
                        for loss_cfg, seed in conditions:
                            key = (scheme.name, bits, loss_cfg.display_name(), seed)
                            run_clip(feats, ref_pred, tail, tail_dev, cfg, scheme, bits, loss_cfg, seed, clip, accs[key])
                print(f"  クリップ {clip + 1}/{len(dataset)} 完了（経過 {time.perf_counter() - started:.0f} 秒）", flush=True)

            shape = list(feats.shape[1:])
            rows = _summarize(cfg, target, shape, accs, curve_writer)
            for r in rows:
                writer.write(r)
            summary += rows

    cols = ["model", "split_point", "scheme", "bits", "loss", "agreement", "agreement_drop", "feature_mse", "loss_rate", "entropy_bpe"]
    print("\n" + format_table(summary, cols))
    print(f"\n全結果: {run_dir / 'results.csv'}\nGoP 位置ごと: {run_dir / 'gop_curve.csv'}")


def _summarize(cfg: PacketLossConfig, target, shape, accs, curve_writer) -> list[dict]:
    """シードごとの集計をまとめて結果の行にする。"""
    by_cond: dict[tuple, list[_Accumulator]] = defaultdict(list)
    for (scheme, bits, loss, _seed), acc in accs.items():
        by_cond[(scheme, bits, loss)].append(acc)
    loss_cfgs = {l.display_name(): l for l in cfg.losses}

    rows = []
    for (scheme, bits, loss), group in by_cond.items():
        frames = sum(a.frames for a in group)
        agreement = sum(a.agree for a in group) / frames
        per_seed = [a.agree / a.frames for a in group]
        lc = loss_cfgs[loss]
        rows.append(
            {
                "model": target.model.name,
                "split_point": target.split_point,
                "feature_shape": "x".join(map(str, shape)),
                "scheme": scheme,
                "bits": bits,
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
            }
        )
        pos_count = sum(a.pos_count for a in group)
        pos_agree = sum(a.pos_agree for a in group)
        pos_mse = sum(a.pos_mse for a in group)
        for t in range(cfg.gop):
            if pos_count[t] == 0:
                continue
            curve_writer.write(
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
    return rows


if __name__ == "__main__":
    main()

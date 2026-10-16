"""動画の時系列符号化（FD / DwD / DPCM、dead-zone 量子化）をパケットロス下で評価するシミュレーション実験。

参考: H. Shiba et al., "Delta Coding With Dead-Zone Quantization for Bandwidth-Efficient
Split Computing on Video Streams," IEEE Access, 2026（パケットロスは同論文 §VII-E-5 で今後の課題とされている）。

各 (モデル, 分割点) × 方式 × ビット数 × 損失モデル × シードについて、クリップごとに
    head → 時系列符号化（端末）→ パケット分割 → ロス → 復号・concealment（クラウド）→ tail
を行い、量子化もロスもない元モデルの出力との top-1 一致率（agreement）などを集計する。
端末はロスを知らない（フィードバックなし）ので、DPCM のロスは次の I フレームまで伝搬する。
方式に restorer（scripts/train_restorer.py で学習したロス補修 net）を指定すると、クラウド側で欠けた要素を補ってから
tail に渡す（例: D+DZ+R）。

端末側の符号化（I/P の判定・差分の計算・量子化・エントロピー）はロスによらないので、
クリップごとに (方式, ビット数) ごとに 1 回だけ行い、その結果を全ての損失条件・シードで使い回す。
torch の CPU スレッド数は、ジョブに割り当てられたコア数（os.sched_getaffinity）を超えないように調整する。

出力（runs/<実験名>_<日時>/。--target で絞った場合は runs/<実験名>_<モデル名>_<日時>/）:
    results.csv    条件ごとの集計（agreement, 95% CI, 特徴 MSE, 実際の損失率, 送信量, エントロピー ...）
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

    # モデルごとに別々に実行する（時間制限のある計算機向け）。結果は scripts/merge_results.py でまとめる
    python scripts/packet_loss_sweep.py --config configs/packet_loss_ucf101.yaml --list-targets
    python scripts/packet_loss_sweep.py --config configs/packet_loss_ucf101.yaml --target resnet50

    # 途中で止まった実行を再開する
    python scripts/packet_loss_sweep.py --resume runs/packet_loss_ucf101_resnet50_20261013-120000
"""

from __future__ import annotations

import datetime as dt
import json
import math
import os
import pickle
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from splitcomp.builder import build_split
from splitcomp.cli import parse_cli
from splitcomp.config import PacketLossConfig, load_config, to_dict
from splitcomp.data import build_dataset
from splitcomp.metrics import ResultWriter, make_run_dir, save_run_info
from splitcomp.temporal import (
    TemporalDecoder,
    TemporalEncoder,
    build_loss_model,
    entropy_bits,
    frame_bytes,
    load_restorer,
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


CHECKPOINT = "checkpoint.pkl"


def _clip_seed(seed: int, clip: int) -> int:
    return seed * 100_003 + clip


class EncodedClip:
    """1 クリップを端末側で符号化した結果（損失条件によらないので、全ての損失条件で使い回す）。"""

    def __init__(self, frames: list, entropy: list[float], frame_bytes: int, assign: torch.Tensor, num_packets: int):
        #: フレームごとの送信内容（EncodedFrame）
        self.frames = frames
        #: フレームごとのエントロピー [bit/要素]
        self.entropy = entropy
        #: 1 フレームの送信バイト数（ヘッダ込み）
        self.frame_bytes = frame_bytes
        #: 各要素が入るパケットの番号
        self.assign = assign
        #: 1 フレームのパケット数
        self.num_packets = num_packets


def encode_clip(feats, cfg, scheme, bits) -> EncodedClip:
    """1 クリップ（T フレーム）を端末側で符号化する（I/P の判定・差分の計算・量子化）。

    端末はロスを知らないので、結果は損失モデル・シードによらない。
    """
    n = feats[0].numel()
    enc = TemporalEncoder(scheme.scheme, bits, cfg.gop, scheme.p_quant)
    frames = [enc.encode(feats[t]) for t in range(feats.shape[0])]
    return EncodedClip(
        frames,
        entropy=[entropy_bits(f.idx) for f in frames],
        frame_bytes=frame_bytes(n, bits, cfg.mtu),
        assign=packet_assignment(n, bits, cfg.mtu, cfg.interleave),
        num_packets=num_packets(n, bits, cfg.mtu),
    )


@torch.no_grad()
def decode_clip(encoded: EncodedClip, feats, ref_pred, tail, tail_dev, cfg, loss_cfg, seed, clip, acc: _Accumulator, restorer=None) -> None:
    """符号化済みの 1 クリップにロスを起こして復号し、tail を通して acc に加算する。

    損失の乱数はクリップとシードで決まり、フレームごとに drop を 1 回呼ぶので、条件ごとに独立に再現できる。
    restorer（LossRestorer）を渡すと、P フレームでロスが起きたときに欠けた要素を補う。
    """
    T = len(encoded.frames)
    P = encoded.num_packets
    loss = build_loss_model(loss_cfg, _clip_seed(seed, clip))
    dec = TemporalDecoder(cfg.concealment, restorer=restorer.restore if restorer is not None else None)

    recon = torch.empty_like(feats)
    for t, frame in enumerate(encoded.frames):
        lost = loss.drop(P)
        recon[t] = dec.decode(frame, received_mask(lost, encoded.assign) if lost.any() else None)
        acc.lost_packets += int(lost.sum())
        acc.entropy += encoded.entropy[t]  # 以前の版と同じ順に足す（浮動小数点の和を一致させるため）
    acc.packets += P * T
    acc.bytes += encoded.frame_bytes * T

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
        if s.restorer and not Path(s.restorer).exists():
            raise SystemExit(
                f"方式 {s.name} のロス補修 net {s.restorer} がありません。先に scripts/train_restorer.py で学習するか、"
                f"設定の schemes からこの方式を外してください"
            )
    set_seed(cfg.seed)
    threads = limit_cpu_threads()
    print(f"CPU スレッド数: {threads}")
    head_dev, tail_dev = resolve_device(cfg.devices.head), resolve_device(cfg.devices.tail)
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
        save_run_info(run_dir, to_dict(cfg), {"head_device": str(head_dev), "tail_device": str(tail_dev), "cpu_threads": threads})
        print(f"結果の保存先: {run_dir}")
    print(f"途中で止まったら次のコマンドで再開できます: python scripts/packet_loss_sweep.py --resume {run_dir}")

    # ロス補修 net は方式ごとに 1 回だけ読み込む（モデル共通なので全ての target で同じものを使う）
    restorers = {s.name: load_restorer(s.restorer, tail_dev) for s in cfg.schemes if s.restorer}
    for name, r in restorers.items():
        print(f"{name}: ロス補修 net を使います（学習条件: {r.meta}）")

    # 複数の方式で同じになる符号化（方式, P の量子化, ビット数）。クリップごとに 1 回だけ符号化して保持する
    enc_count = Counter((s.scheme, s.p_quant) for s in cfg.schemes)
    reused_encodings = {(sch, pq, b) for (sch, pq), n in enc_count.items() if n > 1 for b in cfg.bits}

    dataset = build_dataset(cfg.data)
    print(f"クリップ {len(dataset)} 本を評価します（GoP={cfg.gop}, MTU={cfg.mtu}, concealment={cfg.concealment}）")

    # 損失率 0 の条件はシードによらないので 1 回だけ実行する
    conditions = []
    for loss_cfg in cfg.losses:
        for seed in cfg.seeds if loss_cfg.p > 0 else cfg.seeds[:1]:
            conditions.append((loss_cfg, seed))

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
            started = time.perf_counter()
            for clip in range(start_clip, len(dataset)):
                x, _ = dataset[clip]
                with torch.no_grad():
                    ref_pred = model(x.to(tail_dev)).argmax(-1).cpu()
                    (feats,) = head(x.to(head_dev))
                feats = feats.float().cpu()
                shared: dict[tuple, EncodedClip] = {}
                for scheme in cfg.schemes:
                    for bits in cfg.bits:
                        # 符号化は損失条件によらないので 1 回だけ行い、全ての損失条件・シードで使い回す。
                        # 符号化が同じで復号だけ違う方式（D+DZ と D+DZ+R など）の間でも使い回す
                        enc_key = (scheme.scheme, scheme.p_quant, bits)
                        encoded = shared.get(enc_key) or encode_clip(feats, cfg, scheme, bits)
                        if enc_key in reused_encodings:
                            shared[enc_key] = encoded
                        for loss_cfg, seed in conditions:
                            key = (scheme.name, bits, loss_cfg.display_name(), seed)
                            decode_clip(encoded, feats, ref_pred, tail, tail_dev, cfg, loss_cfg, seed, clip, accs[key], restorers.get(scheme.name))
                        del encoded
                del shared
                shape = list(feats.shape[1:])
                save_checkpoint(run_dir, target, clip + 1, len(dataset), shape, accs)
                print(f"  クリップ {clip + 1}/{len(dataset)} 完了（経過 {time.perf_counter() - started:.0f} 秒）", flush=True)

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

    rows, curve = [], []
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

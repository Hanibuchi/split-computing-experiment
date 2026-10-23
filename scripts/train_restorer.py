"""モデル共通のロス補修 net（splitcomp.temporal.restore.LossRestorer）を学習する。

packet_loss_sweep.py の方式 "D+DZ+R" で使う。head / tail は固定したまま（再学習しない）、
クラウド側で復号のあと・tail の前に挟む小さな CNN だけを学習する。学習の手順:

    1. 学習クリップの head 特徴を計算する（head は固定）
    2. bits・損失条件をランダムに選び、DPCM + dead-zone 量子化（D+DZ）で符号化してパケットロスを起こす
    3. 現在の補修 net を入れた復号器で時間順に復号し（評価時と同じく、補修した値が次のフレームの基準になる）、
       ロスのあった P フレームの (前フレームの復元特徴, 欠けた今のフレーム, 届いたマスク) を集める
    4. 欠けた要素を「ロスがなかったときの復元特徴」（端末の f̂_t）に近づける（チャネルごとの標準偏差で割った MSE）

tail やラベルは使わないので、タスクによらない。全てのチャネルで同じ重みを使うので、
1 つのモデルで学習した net を、チャネル数の違う他のモデル・分割点にもそのまま使える。

出力（runs/<実験名>_<日時>/）:
    restorer.pt   検証の誤差が最も小さかったエポックの重み（save_to を指定するとそこにもコピーする）
    results.csv   エポックごとの学習損失と検証の誤差（hold との比較）

例:
    python scripts/train_restorer.py --config configs/train_restorer_ucf101.yaml
    # 動作確認（クリップ 4 本・1 エポック）
    python scripts/train_restorer.py --config configs/train_restorer_ucf101.yaml data.kwargs.num_clips=4 val_clips=1 epochs=1
"""

from __future__ import annotations

import random
import shutil
import time
from pathlib import Path

import torch

from splitcomp.builder import build_split
from splitcomp.cli import parse_cli
from splitcomp.config import RestorerTrainConfig, to_dict
from splitcomp.data import build_dataset
from splitcomp.metrics import ResultWriter, make_run_dir, save_run_info
from splitcomp.temporal import (
    LossRestorer,
    TemporalDecoder,
    TemporalEncoder,
    build_loss_model,
    entropy_layout,
    received_mask,
    save_restorer,
)
from splitcomp.utils import resolve_device, set_seed


def target_name(target) -> str:
    return f"{target.model.name}/{target.split_point}"


@torch.no_grad()
def simulate(feats: torch.Tensor, cfg: RestorerTrainConfig, bits: int, loss_cfg, seed: int, model: LossRestorer | None):
    """1 系列（T フレーム）を D+DZ で符号化してロスを起こし、model を入れた復号器で復号する。

    Args:
        feats: head 特徴 (T, C, H, W)（CPU）。
        model: 補修 net。None なら hold だけで復号する（比較用）。

    Returns:
        (samples, err, ref):
            samples: ロスのあった P フレームごとの (prev, cur, received, target)。target はロスがなかったときの復元特徴。
            err: 全フレームの ||復号結果 − ロスなしの復元特徴||² の和。
            ref: 全フレームの ||ロスなしの復元特徴||² の和（err を相対誤差にする用）。
    """
    loss = build_loss_model(loss_cfg, seed)
    enc = TemporalEncoder("dpcm", bits, cfg.gop, "dz")

    record: list[tuple] = []

    def restorer(prev, cur, received):
        record.append((prev.clone(), cur.clone(), received.clone()))
        return model.restore(prev, cur, received)

    dec = TemporalDecoder("hold", restorer=restorer if model is not None else None)
    samples, err, ref = [], 0.0, 0.0
    for f in feats:
        frame = enc.encode(f)
        target = enc.prev_hat.clone()  # ロスがなかったときにクラウドが持つ f̂_t
        layout = entropy_layout(frame.idx, bits, cfg.mtu, cfg.interleave)
        lost = loss.drop(layout.num_packets)
        record.clear()
        out = dec.decode(frame, received_mask(lost, layout.assign) if lost.any() else None)
        if record:
            samples.append((*record[0], target))
        err += float((out - target).pow(2).sum())
        ref += float(target.pow(2).sum())
    return samples, err, ref


def train_step(model: LossRestorer, opt: torch.optim.Optimizer, batch: list[tuple], cfg: RestorerTrainConfig, device) -> float | None:
    """ロスのあった P フレームのまとまりで 1 回更新し、損失を返す（欠けた要素がなければ None）。"""
    prev, cur, received, target = (torch.stack(x).to(device) for x in zip(*batch))
    # 届いた要素は出力が cur のままで勾配が出ないので、欠けた要素を含むチャネルだけを計算する
    lost_ch = (~received).any(dim=(0, 2, 3)).nonzero().flatten()
    if lost_ch.numel() == 0:
        return None
    if cfg.max_channels is not None and lost_ch.numel() > cfg.max_channels:
        lost_ch = lost_ch[torch.randperm(lost_ch.numel(), device=lost_ch.device)[: cfg.max_channels]]
    out = model(prev, cur, received, channels=lost_ch)
    s = model.scale(prev)[:, lost_ch]
    lost = ~received[:, lost_ch]
    loss = ((out - target[:, lost_ch]) / s).pow(2)[lost].mean()
    opt.zero_grad()
    loss.backward()
    # フレーム全損失が続いたあとなどは誤差がとても大きいことがあるので、勾配の大きさを抑える
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    return loss.item()


def main() -> None:
    cfg, _ = parse_cli(RestorerTrainConfig, __doc__)
    set_seed(cfg.seed)
    rng = random.Random(cfg.seed)
    head_dev, dev = resolve_device(cfg.devices.head), resolve_device(cfg.devices.tail)
    run_dir = make_run_dir(cfg.output_dir, cfg.experiment_name)
    save_run_info(run_dir, to_dict(cfg), {"head_device": str(head_dev), "restorer_device": str(dev)})
    print(f"結果の保存先: {run_dir}")

    dataset = build_dataset(cfg.data)
    n_val = min(cfg.val_clips, len(dataset) - 1)
    train_idx = list(range(len(dataset) - n_val))
    val_idx = list(range(len(dataset) - n_val, len(dataset)))
    print(f"学習クリップ {len(train_idx)} 本 / 検証クリップ {len(val_idx)} 本")

    heads = []
    for target in cfg.targets:
        bundle = build_split(target.model, target.split_point)
        if not bundle.point.is_clean:
            raise ValueError(f"{target_name(target)} は送信テンソルが複数あります（clean な点を指定してください）")
        heads.append((target_name(target), bundle.head.to(head_dev).eval()))

    def features(clip: int):
        x, _ = dataset[clip]
        for name, head in heads:
            with torch.no_grad():
                (f,) = head(x.to(head_dev))
            if f.dim() != 4:
                raise ValueError(f"{name} の特徴の形 {tuple(f.shape[1:])} は (C, H, W) ではありません")
            yield name, f.float().cpu()

    model = LossRestorer(cfg.hidden, tuple(cfg.dilations)).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    meta = {
        "targets": [n for n, _ in heads],
        "bits": list(cfg.bits),
        "losses": [l.display_name() for l in cfg.losses],
        "gop": cfg.gop,
        "mtu": cfg.mtu,
        "interleave": cfg.interleave,
    }

    def validate() -> tuple[float, float]:
        """検証クリップで、全ての (bits, 損失) について復号誤差（ロスなしの復元特徴に対する相対誤差）を測る。"""
        model.eval()
        err_r = err_h = ref = 0.0
        for clip in val_idx:
            for _, feats in features(clip):
                for bits in cfg.bits:
                    for k, loss_cfg in enumerate(cfg.losses):
                        seed = 1_000_003 + clip * 101 + k
                        _, e, r = simulate(feats, cfg, bits, loss_cfg, seed, model)
                        _, h, _ = simulate(feats, cfg, bits, loss_cfg, seed, None)
                        err_r, err_h, ref = err_r + e, err_h + h, ref + r
        return err_r / ref, err_h / ref

    best = float("inf")
    with ResultWriter(run_dir) as writer:
        if val_idx:
            v_r, v_h = validate()
            print(f"学習前: 検証の相対誤差 restorer {v_r:.4e} / hold {v_h:.4e}")
        for epoch in range(1, cfg.epochs + 1):
            started = time.perf_counter()
            losses, order = [], train_idx[:]
            rng.shuffle(order)
            for i, clip in enumerate(order):
                for _, feats in features(clip):
                    for _ in range(cfg.sequences_per_clip):
                        bits, loss_cfg = rng.choice(cfg.bits), rng.choice(cfg.losses)
                        model.eval()
                        samples, _, _ = simulate(feats, cfg, bits, loss_cfg, rng.randrange(1 << 31), model)
                        rng.shuffle(samples)
                        model.train()
                        for b in range(0, len(samples), cfg.batch_size):
                            l = train_step(model, opt, samples[b : b + cfg.batch_size], cfg, dev)
                            if l is not None:
                                losses.append(l)
                if (i + 1) % 20 == 0:
                    recent = sum(losses[-50:]) / max(len(losses[-50:]), 1)
                    print(f"  epoch {epoch} クリップ {i + 1}/{len(order)} 損失 {recent:.4f}", flush=True)
            train_loss = sum(losses) / max(len(losses), 1)
            row = {"epoch": epoch, "train_loss": train_loss, "steps": len(losses)}
            if val_idx:
                v_r, v_h = validate()
                row.update(val_rel_err=v_r, val_rel_err_hold=v_h, val_ratio=v_r / v_h)
            row["seconds"] = time.perf_counter() - started
            writer.write(row)
            print(f"epoch {epoch}: " + ", ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()), flush=True)

            score = row.get("val_rel_err", -epoch)  # 検証がなければ最後のエポックを残す
            if score < best:
                best = score
                save_restorer(run_dir / "restorer.pt", model, {**meta, "epoch": epoch, **{k: v for k, v in row.items() if k.startswith("val")}})

    print(f"\n重み: {run_dir / 'restorer.pt'}")
    if cfg.save_to:
        Path(cfg.save_to).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(run_dir / "restorer.pt", cfg.save_to)
        print(f"コピー: {cfg.save_to}（packet_loss_sweep の方式の restorer に指定できます）")


if __name__ == "__main__":
    main()

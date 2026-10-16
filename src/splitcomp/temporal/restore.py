"""パケットロスで欠けた中間特徴を、クラウド側で補うモデル共通の小さな CNN（ロス補修 net）。

Itahara et al. (2021) はロスを模擬しながら DNN を再学習して耐性を得るが、ここでは head / tail を一切変えず、
復号のあと・tail の前に補修 net を挟む（方式 "D+DZ+R"）。P フレームでロスが起きたときだけ使い、
届かなかった要素を「前フレームの復元特徴 f̂_{t−1}」と「今のフレームの、ロスで欠けた復元特徴」から推定する。

モデル共通にするための工夫:
    - **全てのチャネルで同じ重みを使う**。特徴 (C, H, W) を、1 チャネル = 1 サンプルとして CNN に通すので、
      チャネル数・空間サイズが違うモデル（分割点）にも、学習し直さずにそのまま使える。
    - チャネル間の情報は、チャネル数によらない「文脈マップ」として渡す: 届いた要素の差分 (cur − prev)/s の
      チャネル方向の平均・絶対値の平均と、その位置で届いた要素の割合。チャネルがまるごと欠けても、
      他のチャネルで「どこが動いたか」が分かる。
    - スケール s はチャネルごとの f̂_{t−1} の標準偏差（クラウドが知っている値）。モデルごとに大きく違う特徴の大きさを揃える。

1 チャネルあたりの入力（6 枚）:
    prev / s, cur / s, mask（届いた = 1）, 文脈マップ 3 枚（全チャネル共通）
出力:
    out = where(mask, cur, cur + s · Δ)。届いた要素は変えない。cur のロスした要素は hold（= prev）で埋めた値。

最後の層は 0 で初期化するので、学習前は hold と全く同じ出力になる。

使い方::

    from splitcomp.temporal import TemporalDecoder, load_restorer

    restorer = load_restorer("checkpoints/restorer_resnet50_layer2.pt", device="cuda")
    dec = TemporalDecoder("hold", restorer=restorer.restore)
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

#: 1 チャネルあたりの入力枚数（prev, cur, mask + 文脈マップ 3 枚）
IN_CHANNELS = 6


class LossRestorer(nn.Module):
    """モデル共通のロス補修 net。

    Args:
        hidden: 中間層のチャネル数。
        dilations: 3×3 畳み込みの各層の dilation（層の数もこれで決まる）。大きな値で大きな欠けにも届く。
        scale_floor: スケール s の下限（全体の標準偏差に対する割合）。ほぼ一定のチャネルで s が 0 に近づくのを防ぐ。
    """

    def __init__(self, hidden: int = 32, dilations: tuple[int, ...] = (1, 2, 4, 2, 1), scale_floor: float = 0.1):
        super().__init__()
        self.arch = {"hidden": hidden, "dilations": list(dilations), "scale_floor": scale_floor}
        self.scale_floor = scale_floor
        #: 学習条件のメモ（load_restorer で読み込んだときに入る）
        self.meta: dict = {}
        layers: list[nn.Module] = []
        ch = IN_CHANNELS
        for d in dilations:
            layers += [nn.Conv2d(ch, hidden, 3, padding=d, dilation=d), nn.ReLU(inplace=True)]
            ch = hidden
        self.body = nn.Sequential(*layers)
        self.out = nn.Conv2d(ch, 1, 1)
        # 学習前は Δ = 0（hold と同じ出力）から始める
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def scale(self, prev: torch.Tensor) -> torch.Tensor:
        """チャネルごとのスケール s（形 (B, C, 1, 1)）。prev は (B, C, H, W)。"""
        std_c = prev.std(dim=(2, 3), keepdim=True, unbiased=False)
        std_all = prev.flatten(1).std(dim=1, unbiased=False).view(-1, 1, 1, 1)
        return torch.maximum(std_c, self.scale_floor * std_all) + 1e-8

    def forward(
        self,
        prev: torch.Tensor,
        cur: torch.Tensor,
        received: torch.Tensor,
        channels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """欠けた要素を補った特徴を返す。

        Args:
            prev: 前フレームの復元特徴 f̂_{t−1}。(B, C, H, W)。
            cur: 今のフレームの復元特徴（ロスした要素は prev で埋めたもの）。(B, C, H, W)。
            received: 各要素が届いたか（bool）。(B, C, H, W)。
            channels: 指定するとそのチャネルだけを補って (B, len(channels), H, W) を返す（学習時の節約用）。
                文脈マップは常に全チャネルから作る。

        Returns:
            補った特徴。届いた要素は cur のまま。
        """
        if prev.dim() != 4:
            raise ValueError(f"LossRestorer は (B, C, H, W) の特徴だけを扱えます（形 {tuple(prev.shape)}）")
        s = self.scale(prev)
        m = received.to(prev.dtype)
        d = (cur - prev) / s * m
        cnt = m.sum(1, keepdim=True)
        denom = cnt.clamp(min=1.0)
        ctx = torch.cat([d.sum(1, keepdim=True) / denom, d.abs().sum(1, keepdim=True) / denom, cnt / prev.shape[1]], dim=1)

        if channels is not None:
            prev, cur, m, s, received = prev[:, channels], cur[:, channels], m[:, channels], s[:, channels], received[:, channels]
        B, C, H, W = prev.shape
        per_ch = torch.stack([prev / s, cur / s, m], dim=2)  # (B, C, 3, H, W)
        x = torch.cat([per_ch, ctx.unsqueeze(1).expand(B, C, 3, H, W)], dim=2).reshape(B * C, IN_CHANNELS, H, W)
        delta = self.out(self.body(x)).view(B, C, H, W)
        return torch.where(received, cur, cur + s * delta)

    @torch.no_grad()
    def restore(self, prev: torch.Tensor, cur: torch.Tensor, received: torch.Tensor) -> torch.Tensor:
        """1 フレーム (C, H, W) を補う（:class:`~splitcomp.temporal.coding.TemporalDecoder` から呼ぶ用）。

        入力はどのデバイスにあってもよく、このモデルのデバイスで計算して入力と同じデバイスに戻す。
        """
        dev = next(self.parameters()).device
        out = self(prev[None].to(dev), cur[None].to(dev), received[None].to(dev))[0]
        return out.to(cur.device)


def save_restorer(path: str | Path, model: LossRestorer, meta: dict | None = None) -> None:
    """補修 net を保存する（重み・構成・学習条件のメモ）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "arch": model.arch, "meta": meta or {}}, path)


def load_restorer(path: str | Path, device: str | torch.device = "cpu") -> LossRestorer:
    """:func:`save_restorer` で保存した補修 net を読み込み、評価モードで返す。"""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    arch = ckpt["arch"]
    model = LossRestorer(arch["hidden"], tuple(arch["dilations"]), arch["scale_floor"])
    model.load_state_dict(ckpt["state_dict"])
    model.meta = ckpt.get("meta", {})
    return model.to(device).eval()

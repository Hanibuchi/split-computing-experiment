"""通信路のシミュレーション。

実際に通信はせず、データ量から通信時間を計算する:

    片道の通信時間[ms] = RTT/2 + データ量[bit] / 帯域[bit/s] × 1000

単純なモデルだが、分割点・圧縮方式・帯域を変えたときのトレードオフを
素早く比較するには十分なことが多い。より細かいモデル（パケット損失、
TCP スロースタート等）が必要なら SimulatedChannel を継承して上書きする。
"""

from __future__ import annotations

from ..config import ChannelConfig


class SimulatedChannel:
    """帯域と RTT から通信時間を計算する通信路モデル。

    Args:
        bandwidth_mbps: 上り（端末→サーバ）の帯域 [Mbit/s]。
        rtt_ms: 往復遅延 [ms]。
        downlink_bandwidth_mbps: 下り（サーバ→端末）の帯域 [Mbit/s]。None なら上りと同じ。
        name: 表示名。
    """

    def __init__(
        self,
        bandwidth_mbps: float,
        rtt_ms: float = 0.0,
        downlink_bandwidth_mbps: float | None = None,
        name: str = "",
    ):
        if bandwidth_mbps <= 0:
            raise ValueError("bandwidth_mbps は正の値で指定してください")
        self.bandwidth_mbps = bandwidth_mbps
        self.rtt_ms = rtt_ms
        self.downlink_bandwidth_mbps = downlink_bandwidth_mbps or bandwidth_mbps
        self.name = name or f"{bandwidth_mbps:g}Mbps/{rtt_ms:g}ms"

    @classmethod
    def from_config(cls, cfg: ChannelConfig) -> "SimulatedChannel":
        return cls(cfg.bandwidth_mbps, cfg.rtt_ms, cfg.downlink_bandwidth_mbps, cfg.display_name())

    @staticmethod
    def _transfer_ms(nbytes: int, mbps: float) -> float:
        # Mbit/s = 1e6 bit/s。bytes → bit は ×8。秒 → ミリ秒は ×1000。
        return nbytes * 8 / (mbps * 1e6) * 1e3

    def uplink_ms(self, nbytes: int) -> float:
        """端末→サーバへ nbytes 送るのにかかる時間 [ms]。"""
        return self.rtt_ms / 2 + self._transfer_ms(nbytes, self.bandwidth_mbps)

    def downlink_ms(self, nbytes: int) -> float:
        """サーバ→端末へ nbytes 返すのにかかる時間 [ms]。"""
        return self.rtt_ms / 2 + self._transfer_ms(nbytes, self.downlink_bandwidth_mbps)

    def __repr__(self) -> str:
        return f"SimulatedChannel({self.name})"

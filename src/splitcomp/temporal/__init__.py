"""動画の中間特徴の時系列符号化（Delta coding / DPCM / dead-zone 量子化）とパケットロスのシミュレーション。

参考: H. Shiba et al., "Delta Coding With Dead-Zone Quantization for Bandwidth-Efficient
Split Computing on Video Streams," IEEE Access, vol. 14, 2026.

- :mod:`.quantize` : P99 範囲の対称量子化 / dead-zone 量子化、エントロピー
- :mod:`.coding`   : TemporalEncoder（端末）/ TemporalDecoder（クラウド）: FD / DwD / DPCM
- :mod:`.packet`   : パケット分割と損失モデル（Bernoulli / Gilbert-Elliott / フレーム単位）
- :mod:`.restore`  : ロスで欠けた特徴をクラウド側で補う、モデル共通のロス補修 net（D+DZ+R）
"""

from .coding import EncodedFrame, TemporalDecoder, TemporalEncoder
from .packet import (
    HEADER_BYTES,
    BernoulliLoss,
    FrameLoss,
    GilbertElliottLoss,
    LossModel,
    build_loss_model,
    elements_per_packet,
    frame_bytes,
    num_packets,
    packet_assignment,
    packetize,
    received_mask,
    register_loss_model,
)
from .quantize import deadzone_quantize, dequantize, entropy_bits, p99_range, quantize, symmetric_quantize
from .restore import LossRestorer, load_restorer, save_restorer

__all__ = [
    "TemporalEncoder",
    "TemporalDecoder",
    "EncodedFrame",
    "LossModel",
    "BernoulliLoss",
    "GilbertElliottLoss",
    "FrameLoss",
    "build_loss_model",
    "register_loss_model",
    "HEADER_BYTES",
    "elements_per_packet",
    "num_packets",
    "packet_assignment",
    "packetize",
    "frame_bytes",
    "received_mask",
    "quantize",
    "symmetric_quantize",
    "deadzone_quantize",
    "dequantize",
    "p99_range",
    "entropy_bits",
    "LossRestorer",
    "load_restorer",
    "save_restorer",
]

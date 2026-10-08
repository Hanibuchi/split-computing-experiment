"""通信まわり。

- :class:`SimulatedChannel` : 帯域・RTT から通信時間を計算する（シミュレーション）
- :class:`TailServer`       : tail を実行する TCP サーバ
- :class:`TCPBackend`       : TailServer に推論を依頼するクライアント
"""

from .channel import SimulatedChannel
from .client import TCPBackend
from .protocol import decode_output, encode_output, recv_msg, send_msg
from .server import TailServer

__all__ = [
    "SimulatedChannel",
    "TailServer",
    "TCPBackend",
    "send_msg",
    "recv_msg",
    "encode_output",
    "decode_output",
]

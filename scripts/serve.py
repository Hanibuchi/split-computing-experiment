"""実ネットワーク実験のサーバ側: tail を実行する TCP サーバを起動する。

クライアント（scripts/client.py）と同じ設定ファイルを使うこと。
Ctrl+C で終了する。

例（サーバ機で実行）:
    python scripts/serve.py --config configs/network.yaml host=0.0.0.0
"""

from __future__ import annotations

from splitcomp.builder import build_split, config_fingerprint
from splitcomp.cli import parse_cli
from splitcomp.compression import build_codec
from splitcomp.config import NetworkConfig
from splitcomp.transport import TailServer
from splitcomp.utils import resolve_device, set_seed


def main() -> None:
    cfg, _ = parse_cli(NetworkConfig, __doc__)
    set_seed(cfg.seed)
    device = resolve_device(cfg.devices.tail)
    bundle = build_split(cfg.model, cfg.split_point, cfg.bottleneck_ckpt)
    codec = build_codec(cfg.codec)
    fp = config_fingerprint(cfg.model, cfg.split_point, cfg.codec, cfg.bottleneck_ckpt)

    server = TailServer(bundle.tail, codec, device, cfg.host, cfg.port, fingerprint=fp)
    host, port = server.address
    print(f"[server] {cfg.model.name} の分割点 {bundle.point.name} 以降（tail）を {device} で実行します")
    print(f"[server] 圧縮方式: {codec.name}  待ち受け: {host}:{port}  （Ctrl+C で終了）")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[server] 終了します")
    finally:
        server.shutdown()


if __name__ == "__main__":
    main()

"""リモートの ZIP から必要なファイルだけを取り出すダウンロード機能（標準ライブラリのみ）。

HTTP の Range リクエストで ZIP の目次（末尾の central directory）と必要なメンバだけを読むので、
巨大なアーカイブ（UCF-101 は約 7GB）でも、使うファイルの分だけダウンロードすればよい。

例::

    from splitcomp.data.download import fetch_zip_members
    fetch_zip_members(url, ["UCF-101/Typing/v_Typing_g01_c01.avi"], "./data")
"""

from __future__ import annotations

import io
import shutil
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

#: UCF-101 の ZIP ミラー（HuggingFace）。先頭から順に試す。
#: どちらも中身は ``UCF-101/<クラス名>/v_<クラス名>_gXX_cYY.avi``。
UCF101_ZIP_URLS = [
    "https://huggingface.co/datasets/MichiganNLP/ucf-101/resolve/main/UCF101.zip",
    "https://huggingface.co/datasets/quchenyuan/UCF101-ZIP/resolve/main/UCF-101.zip",
]

_USER_AGENT = "splitcomp/0.1"


class HttpRangeFile(io.RawIOBase):
    """HTTP の Range リクエストで読む、シーク可能な読み取り専用ファイル。

    サーバが Range（206 Partial Content）に対応している必要がある。
    リダイレクト後の URL を覚えておき、以降のリクエストはそこへ直接送る。
    """

    def __init__(self, url: str, timeout: float = 60.0, retries: int = 3):
        self.timeout = timeout
        self.retries = retries
        self.pos = 0
        self.num_requests = 0
        req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": _USER_AGENT})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            self.url = r.url
            self.size = int(r.headers["Content-Length"])

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.pos, io.SEEK_END: self.size}[whence]
        self.pos = base + offset
        return self.pos

    def readinto(self, b) -> int:
        if self.pos >= self.size or len(b) == 0:
            return 0
        end = min(self.pos + len(b), self.size) - 1
        data = self._get(self.pos, end)
        b[: len(data)] = data
        self.pos += len(data)
        return len(data)

    def _get(self, start: int, end: int) -> bytes:
        headers = {"Range": f"bytes={start}-{end}", "User-Agent": _USER_AGENT}
        for attempt in range(self.retries):
            try:
                self.num_requests += 1
                with urllib.request.urlopen(urllib.request.Request(self.url, headers=headers), timeout=self.timeout) as r:
                    if r.status != 206:
                        raise OSError(f"サーバが Range リクエストに対応していません（HTTP {r.status}）: {self.url}")
                    return r.read()
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                if attempt == self.retries - 1:
                    raise
                time.sleep(2**attempt)
        raise AssertionError("unreachable")


def open_remote_zip(url: str) -> zipfile.ZipFile:
    """リモートの ZIP を開く（目次だけを読む）。"""
    return zipfile.ZipFile(io.BufferedReader(HttpRangeFile(url), buffer_size=1 << 20))


def fetch_zip_members(url: str | list[str], members: list[str], dest_root: str | Path, verbose: bool = True) -> list[Path]:
    """リモートの ZIP から members だけを ``dest_root/<メンバのパス>`` に展開する。

    展開済みのファイルは飛ばす。url にリストを渡すと先頭から順に試し、失敗したら次のミラーを使う。
    CRC の検査は zipfile が行う（壊れていれば例外になり、途中のファイルは残さない）。

    Returns:
        展開先のパスのリスト（members と同じ順）。
    """
    dest_root = Path(dest_root)
    paths = [dest_root / m for m in members]
    todo = [m for m, p in zip(members, paths) if not p.exists()]
    if not todo:
        return paths

    urls = [url] if isinstance(url, str) else list(url)
    errors = []
    for u in urls:
        try:
            if verbose:
                print(f"ダウンロード: {u} から {len(todo)} ファイルを取得します", flush=True)
            with open_remote_zip(u) as zf:
                for i, m in enumerate(todo):
                    out = dest_root / m
                    out.parent.mkdir(parents=True, exist_ok=True)
                    tmp = out.with_name(out.name + ".part")
                    with zf.open(m) as src, open(tmp, "wb") as dst:
                        shutil.copyfileobj(src, dst, 1 << 20)
                    tmp.replace(out)
                    if verbose:
                        print(f"  [{i + 1}/{len(todo)}] {m} ({out.stat().st_size / 1e6:.1f} MB)", flush=True)
            return paths
        except Exception as e:  # 次のミラーを試す
            errors.append(f"{u}: {e!r}")
            todo = [m for m, p in zip(members, paths) if not p.exists()]
    raise RuntimeError("どのミラーからもダウンロードできませんでした:\n  " + "\n  ".join(errors))


def list_remote_zip(url: str | list[str]) -> list[str]:
    """リモートの ZIP のメンバ名の一覧（url がリストなら最初に開けたもの）。"""
    urls = [url] if isinstance(url, str) else list(url)
    errors = []
    for u in urls:
        try:
            with open_remote_zip(u) as zf:
                return zf.namelist()
        except Exception as e:
            errors.append(f"{u}: {e!r}")
    raise RuntimeError("どのミラーも開けませんでした:\n  " + "\n  ".join(errors))

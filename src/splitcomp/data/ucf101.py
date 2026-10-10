"""UCF-101 の動画クリップ（時系列の特徴量圧縮・パケットロス実験用）。

UCF-101 は 101 種類の動作（クラス）の動画データセットで、動画はクラスごとのディレクトリに入っている::

    <root>/UCF-101/<クラス名>/v_<クラス名>_gXX_cYY.avi    例: ./data/UCF-101/Typing/v_Typing_g01_c01.avi

（gXX は撮影グループ（人・場所）、cYY はそのグループ内のクリップ番号。フレーム抽出済みのディレクトリ
``<root>/UCF-101/<クラス名>/<動画名>/*.jpg`` も読める。）

使う動画が手元になければ、HuggingFace の ZIP ミラーから**必要な動画だけ**を自動でダウンロードする
（:mod:`.download`。7GB 全体は落とさない）。

クリップの選び方（``selection``）:
    "paper"  : :data:`PAPER_CLIPS` の 30 本（既定）。Shiba et al. (IEEE Access 2026) §V-B と同じ条件を狙ったもの
    "random" : seed で num_clips クラスを選び、各クラスから動画を 1 本ずつ選ぶ

1 要素 = 1 クリップで ``(frames[T, 3, size, size], label)`` を返す（label は 101 クラスのアルファベット順の番号）。
各動画の先頭から clip_len フレームを（frame_step おきに）取り出す。frame_step=2 で時間間隔が 2 倍（論文の 15fps 実験）。
デコード結果は ``<root>/UCF-101/.cache/`` に保存して再利用する。.avi の読み込みには PyAV が必要（``pip install av``）。
"""

from __future__ import annotations

import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .download import UCF101_ZIP_URLS, fetch_zip_members, list_remote_zip
from .registry import IMAGENET_MEAN, IMAGENET_STD, register_dataset

_VIDEO_EXTS = {".avi", ".mp4", ".mkv", ".mov", ".webm"}
_IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

#: 論文の実験に合わせた 30 クリップ (UCF-101 のカテゴリ, 動画名)。
#:
#: 論文（§V-B）には「5 カテゴリにまたがって 30 クリップ・各 30 連続フレーム」とあるが、具体的なリストは
#: 公開されていない。そのため以下は次の方針で選んだ**推測**である:
#:   - 30 本を UCF-101 の 5 カテゴリのクラス数に比例して配分
#:     （Human–Object 20 → 6, Body-Motion 16 → 5, Human–Human 5 → 1, Instruments 10 → 3, Sports 50 → 15）
#:   - 論文が例に挙げている Typing（ほぼ静止）と Basketball（激しい動き）は必ず含め、残りは各カテゴリから seed=0 で選択
#:   - 動画は公式 split 1 のテスト側にあたるグループ 1 の最初のクリップ（g01_c01）。全て 60 フレーム以上（15fps 実験にも使える）
#: 並び順はカテゴリを順に回しているので、num_clips で先頭 N 本に絞ってもカテゴリが偏りにくい。
#: なお UCF-101 の大半は 25fps（一部 29.97fps）だが、論文どおり「元動画の連続 30 フレーム」を使う。
PAPER_CLIPS: list[tuple[str, str]] = [
    ("human_object", "v_ApplyLipstick_g01_c01"),
    ("body_motion", "v_HandstandWalking_g01_c01"),
    ("human_human", "v_HeadMassage_g01_c01"),
    ("instruments", "v_PlayingDaf_g01_c01"),
    ("sports", "v_Basketball_g01_c01"),
    ("human_object", "v_JumpRope_g01_c01"),
    ("body_motion", "v_Lunges_g01_c01"),
    ("instruments", "v_PlayingDhol_g01_c01"),
    ("sports", "v_BenchPress_g01_c01"),
    ("human_object", "v_Nunchucks_g01_c01"),
    ("body_motion", "v_PullUps_g01_c01"),
    ("instruments", "v_PlayingViolin_g01_c01"),
    ("sports", "v_Billiards_g01_c01"),
    ("human_object", "v_PizzaTossing_g01_c01"),
    ("body_motion", "v_TaiChi_g01_c01"),
    ("sports", "v_BoxingPunchingBag_g01_c01"),
    ("human_object", "v_SkateBoarding_g01_c01"),
    ("body_motion", "v_WallPushups_g01_c01"),
    ("sports", "v_BoxingSpeedBag_g01_c01"),
    ("human_object", "v_Typing_g01_c01"),
    ("sports", "v_Fencing_g01_c01"),
    ("sports", "v_FloorGymnastics_g01_c01"),
    ("sports", "v_FrisbeeCatch_g01_c01"),
    ("sports", "v_GolfSwing_g01_c01"),
    ("sports", "v_ParallelBars_g01_c01"),
    ("sports", "v_Rafting_g01_c01"),
    ("sports", "v_Rowing_g01_c01"),
    ("sports", "v_Skijet_g01_c01"),
    ("sports", "v_SkyDiving_g01_c01"),
    ("sports", "v_ThrowDiscus_g01_c01"),
]

#: UCF-101 の 101 クラス（アルファベット順。ラベル番号はこの順）
UCF101_CLASSES = sorted(
    """ApplyEyeMakeup ApplyLipstick Archery BabyCrawling BalanceBeam BandMarching BaseballPitch Basketball
    BasketballDunk BenchPress Biking Billiards BlowDryHair BlowingCandles BodyWeightSquats Bowling BoxingPunchingBag
    BoxingSpeedBag BreastStroke BrushingTeeth CleanAndJerk CliffDiving CricketBowling CricketShot CuttingInKitchen
    Diving Drumming Fencing FieldHockeyPenalty FloorGymnastics FrisbeeCatch FrontCrawl GolfSwing Haircut Hammering
    HammerThrow HandstandPushups HandstandWalking HeadMassage HighJump HorseRace HorseRiding HulaHoop IceDancing
    JavelinThrow JugglingBalls JumpingJack JumpRope Kayaking Knitting LongJump Lunges MilitaryParade Mixing
    MoppingFloor Nunchucks ParallelBars PizzaTossing PlayingCello PlayingDaf PlayingDhol PlayingFlute PlayingGuitar
    PlayingPiano PlayingSitar PlayingTabla PlayingViolin PoleVault PommelHorse PullUps Punch PushUps Rafting
    RockClimbingIndoor RopeClimbing Rowing SalsaSpin ShavingBeard Shotput SkateBoarding Skiing Skijet SkyDiving
    SoccerJuggling SoccerPenalty StillRings SumoWrestling Surfing Swing TableTennisShot TaiChi TennisSwing ThrowDiscus
    TrampolineJumping Typing UnevenBars VolleyballSpiking WalkingWithDog WallPushups WritingOnBoard YoYo""".split()
)


def class_of(video: str) -> str:
    """動画名 ``v_<クラス名>_gXX_cYY`` からクラス名を取り出す。"""
    return video[2:].rsplit("_", 2)[0]


class UCF101Clips(Dataset):
    """UCF-101 から固定長のクリップを取り出すデータセット。

    Args:
        root: データの置き場所。``<root>/UCF-101/<クラス名>/*.avi`` を探す（ダウンロード先も同じ）。
        num_clips: 使うクリップ数。selection="paper" では PAPER_CLIPS の先頭 N 本（None で 30 本全て）。
        clip_len: 1 クリップのフレーム数。
        frame_step: 何フレームおきに取り出すか（1: 元の間隔, 2: 間隔 2 倍）。
        size: 入力解像度（size × size にリサイズする。論文と同じくクロップしない）。
        selection: "paper"（PAPER_CLIPS）/ "random"（seed でクラスを選ぶ）。
        seed: selection="random" のときの乱数シード。
        download: 手元にない動画を自動でダウンロードするか。
        cache: デコード結果を ``.cache`` に保存するか。
    """

    def __init__(
        self,
        root: str,
        num_clips: int | None = None,
        clip_len: int = 30,
        frame_step: int = 1,
        size: int = 224,
        selection: str = "paper",
        seed: int = 0,
        download: bool = True,
        cache: bool = True,
    ):
        root = Path(root)
        # root に UCF-101 ディレクトリそのものを指定してもよい
        self.base = root if root.name == "UCF-101" else root / "UCF-101"
        self.clip_len = clip_len
        self.frame_step = frame_step
        self.size = size
        self.cache_dir = self.base / ".cache" if cache else None

        if selection == "paper":
            videos = [v for _, v in PAPER_CLIPS][:num_clips]
        elif selection == "random":
            videos = self._random_videos(num_clips or 30, seed, download)
        else:
            raise KeyError(f"未知の selection です: {selection!r}（\"paper\" / \"random\"）")

        #: (クラス名, 動画のパス, ラベル) のリスト
        self.clips: list[tuple[str, Path, int]] = []
        missing = []
        for v in videos:
            cls = class_of(v)
            path = self._find(cls, v)
            if path is None:
                path = self.base / cls / f"{v}.avi"
                missing.append(path)
            self.clips.append((cls, path, UCF101_CLASSES.index(cls)))

        if missing:
            if not download:
                raise FileNotFoundError(
                    f"UCF-101 の動画が {len(missing)} 本見つかりません（例: {missing[0]}）。\n"
                    f"<root>/UCF-101/<クラス名>/<動画名>.avi の形で置くか、download: true にしてください。"
                )
            members = [f"UCF-101/{p.parent.name}/{p.name}" for p in missing]
            fetch_zip_members(UCF101_ZIP_URLS, members, self.base.parent)

    def _find(self, cls: str, video: str) -> Path | None:
        """手元にある動画（またはフレームのディレクトリ）を探す。"""
        for p in [self.base / cls / video, *(self.base / cls / f"{video}{e}" for e in _VIDEO_EXTS)]:
            if p.exists():
                return p
        return None

    def _random_videos(self, num_clips: int, seed: int, download: bool) -> list[str]:
        local = {}
        if self.base.is_dir():
            for d in self.base.iterdir():
                if d.is_dir() and not d.name.startswith("."):
                    vids = sorted(p.stem if p.is_file() else p.name for p in d.iterdir() if p.suffix.lower() in _VIDEO_EXTS or p.is_dir())
                    if vids:
                        local[d.name] = vids
        if len(local) < num_clips:
            if not download:
                raise FileNotFoundError(f"UCF-101 のクラスが {len(local)} 個しか見つかりません: {self.base}")
            local = {}
            for name in list_remote_zip(UCF101_ZIP_URLS):
                parts = name.split("/")
                if len(parts) == 3 and parts[2].endswith(".avi"):
                    local.setdefault(parts[1], []).append(parts[2][:-4])
            local = {k: sorted(v) for k, v in local.items()}
        rng = random.Random(seed)
        return [rng.choice(local[c]) for c in sorted(rng.sample(sorted(local), num_clips))]

    def __len__(self) -> int:
        return len(self.clips)

    def clip_name(self, i: int) -> str:
        return self.clips[i][1].stem

    def __getitem__(self, i: int):
        _, path, label = self.clips[i]
        cache_path = None
        if self.cache_dir is not None:
            cache_path = self.cache_dir / f"{path.stem}_T{self.clip_len}_s{self.frame_step}_{self.size}_u8.pt"
        if cache_path is not None and cache_path.exists():
            x = torch.load(cache_path, weights_only=True)
        else:
            frames = _read_frames(path, self.clip_len * self.frame_step)[:: self.frame_step]  # (T, H, W, 3) uint8
            if len(frames) < self.clip_len:
                raise ValueError(f"{path} のフレーム数が足りません（{len(frames)} < {self.clip_len}）")
            x = frames.permute(0, 3, 1, 2).float()
            x = F.interpolate(x, size=(self.size, self.size), mode="bilinear", align_corners=False, antialias=True)
            # キャッシュは uint8 で持つ（float32 の 1/4 の容量）。キャッシュの有無で入力が変わらないよう、常にここで丸める
            x = x.round().clamp(0, 255).to(torch.uint8)
            if cache_path is not None:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(x, cache_path)
        mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
        return (x.float() / 255.0 - mean) / std, label

def _read_frames(path: Path, n: int) -> torch.Tensor:
    """動画（またはフレーム画像のディレクトリ）の先頭 n フレームを (n, H, W, 3) uint8 で読む。"""
    import numpy as np

    if path.is_dir():
        from PIL import Image

        files = sorted(p for p in path.iterdir() if p.suffix.lower() in _IMAGE_EXTS)[:n]
        return torch.from_numpy(np.stack([np.asarray(Image.open(p).convert("RGB")) for p in files]))
    try:
        import av
    except ImportError as e:
        raise ImportError("動画の読み込みには `pip install av`（または pip install -e \".[video]\"）が必要です") from e
    frames = []
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            frames.append(frame.to_ndarray(format="rgb24"))
            if len(frames) >= n:
                break
    return torch.from_numpy(np.stack(frames))


@register_dataset("ucf101")
def _build_ucf101(root: str, split: str, **kwargs):
    return UCF101Clips(root, **kwargs)

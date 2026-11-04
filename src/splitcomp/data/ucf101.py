"""UCF-101 の動画クリップ（時系列の特徴量圧縮・パケットロス実験用）。

UCF-101 は 101 種類の動作（クラス）の動画データセットで、動画はクラスごとのディレクトリに入っている::

    <root>/UCF-101/<クラス名>/v_<クラス名>_gXX_cYY.avi    例: ./data/UCF-101/Typing/v_Typing_g01_c01.avi

（gXX は撮影グループ（人・場所）、cYY はそのグループ内のクリップ番号。フレーム抽出済みのディレクトリ
``<root>/UCF-101/<クラス名>/<動画名>/*.jpg`` も読める。）

使う動画が手元になければ、HuggingFace の ZIP ミラーから**必要な動画だけ**を自動でダウンロードする
（:mod:`.download`。7GB 全体は落とさない）。

クリップの選び方（``selection``）:
    "paper"  : :data:`PAPER_CLIPS` の 30 本（既定）。Shiba et al. (IEEE Access 2026) の著者から提供された、
               論文の実験と同じ動画と開始フレーム。clip_len × frame_step フレームに足りない動画は除外する
    "random" : seed で num_clips クラスを選び、各クラスから動画を clips_per_class 本ずつ選ぶ。
               groups で撮影グループを絞れる（例: "8-25" = 公式 split 1 の train 側。評価用の g01 と重ならない）
               clip_len に満たない動画（UCF-101 にまれにある）は読み込み時に除外する。
               PAPER_CLIPS の動画は選ばない（評価用の 30 本と重ならないようにするため）

1 要素 = 1 クリップで ``(frames[T, 3, size, size], label)`` を返す（label は 101 クラスのアルファベット順の番号）。
各動画の開始フレーム（paper は PAPER_CLIPS の値、random は 0）から clip_len フレームを（frame_step おきに）取り出す。
frame_step=2 で時間間隔が 2 倍（論文の 15fps 実験）。

前処理は論文と同じ: RGB で読む → 縦横比を無視して size × size にバイリニア補間でリサイズ（アンチエイリアスなし。
OpenCV の ``cv2.resize(..., INTER_LINEAR)`` 相当）→ 255 で割って [0, 1] → ImageNet の平均・標準偏差で正規化。
デコード結果は ``<root>/UCF-101/.cache/`` に保存して再利用する。.avi の読み込みには PyAV が必要（``pip install av``）。
"""

from __future__ import annotations

import random
import warnings
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .download import UCF101_ZIP_URLS, fetch_zip_members, list_remote_zip
from .registry import IMAGENET_MEAN, IMAGENET_STD, register_dataset

_VIDEO_EXTS = {".avi", ".mp4", ".mkv", ".mov", ".webm"}
_IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

#: 論文の実験で使った 30 クリップ (動画名, 開始フレーム, 総フレーム数)。先行研究（Shiba et al.）の著者から提供されたリスト。
#:
#: - 開始フレームは 0 始まり（動画の先頭が 0）。開始フレームから連続する clip_len フレームを使う（間引きなし）
#: - 総フレーム数は著者が OpenCV で読み取った参考値。デコーダによってはずれることがあり、ずれるとフレームの
#:   対応がずれている可能性があるので、読み込み時に照合して違えば警告を出す
#: - 25fps と 29.97fps の動画が混在するが、どちらも元のフレームレートのまま使う（リサンプリングなし）
#: - frame_step=2（15fps 実験）では開始フレームから 60 フレーム必要で、5 本（Shotput_g12_c06, BabyCrawling_g02_c02,
#:   TennisSwing_g18_c02, Lunges_g15_c02, CuttingInKitchen_g02_c03）は末尾を超えるので除外される（25 本になる）
PAPER_CLIPS: list[tuple[str, int, int]] = [
    ("v_Shotput_g12_c06", 26, 67),
    ("v_BlowingCandles_g08_c03", 28, 101),
    ("v_BabyCrawling_g02_c02", 57, 112),
    ("v_TennisSwing_g20_c04", 71, 171),
    ("v_Haircut_g15_c02", 0, 116),
    ("v_FloorGymnastics_g21_c02", 20, 114),
    ("v_Drumming_g24_c01", 216, 300),
    ("v_BoxingPunchingBag_g20_c01", 87, 237),
    ("v_TennisSwing_g08_c02", 71, 234),
    ("v_BlowDryHair_g05_c03", 39, 188),
    ("v_SoccerJuggling_g15_c03", 220, 599),
    ("v_TennisSwing_g18_c02", 97, 148),
    ("v_PlayingViolin_g20_c01", 86, 167),
    ("v_Biking_g10_c03", 26, 240),
    ("v_Rafting_g18_c04", 11, 128),
    ("v_Lunges_g15_c02", 48, 97),
    ("v_BabyCrawling_g23_c01", 12, 116),
    ("v_BabyCrawling_g17_c03", 45, 108),
    ("v_Billiards_g05_c03", 352, 592),
    ("v_Drumming_g12_c05", 135, 300),
    ("v_FieldHockeyPenalty_g06_c01", 5, 115),
    ("v_PlayingFlute_g17_c06", 117, 203),
    ("v_RopeClimbing_g04_c02", 31, 158),
    ("v_BabyCrawling_g07_c02", 48, 145),
    ("v_PommelHorse_g19_c01", 40, 307),
    ("v_CuttingInKitchen_g02_c03", 70, 123),
    ("v_Swing_g16_c04", 37, 109),
    ("v_SkateBoarding_g22_c05", 92, 166),
    ("v_SumoWrestling_g19_c01", 24, 101),
    ("v_PlayingViolin_g18_c02", 17, 182),
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


#: 小文字のクラス名 → 正式なクラス名。動画名とクラス名で大文字・小文字が違うものがあるため
#: （例: フォルダ HandstandPushups の中の動画は v_HandStandPushups_g01_c01.avi）。
_CLASS_BY_LOWER = {c.lower(): c for c in UCF101_CLASSES}


def class_of(video: str) -> str:
    """動画名 ``v_<クラス名>_gXX_cYY`` からクラス名（UCF101_CLASSES の表記）を取り出す。"""
    name = video[2:].rsplit("_", 2)[0]
    return _CLASS_BY_LOWER.get(name.lower(), name)


def group_of(video: str) -> int:
    """動画名 ``v_<クラス名>_gXX_cYY`` から撮影グループの番号 XX を取り出す。"""
    return int(video.rsplit("_", 2)[1][1:])


def parse_groups(groups: list[int] | str | None) -> set[int] | None:
    """groups の指定（[8, 9] / "8-25" / "1,3,8-10" / None）を番号の集合にする。"""
    if groups is None:
        return None
    if isinstance(groups, str):
        out: set[int] = set()
        for part in groups.split(","):
            a, _, b = part.strip().partition("-")
            out.update(range(int(a), int(b or a) + 1))
        return out
    return {int(g) for g in groups}


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
        groups: selection="random" で使う撮影グループ（gXX の番号）。[8, 9, 10] のようなリストか "8-25" の形の文字列。
            None なら全て。
        clips_per_class: selection="random" で 1 クラスから選ぶ動画の本数（クリップ数は num_clips × clips_per_class）。
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
        groups: list[int] | str | None = None,
        clips_per_class: int = 1,
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

        #: 動画名 → (開始フレーム, 総フレーム数の参考値)
        starts: dict[str, tuple[int, int | None]] = {}
        if selection == "paper":
            videos = [v for v, _, _ in PAPER_CLIPS][:num_clips]
            starts = {v: (start, total) for v, start, total in PAPER_CLIPS}
        elif selection == "random":
            videos = self._random_videos(num_clips or 30, seed, download, parse_groups(groups), clips_per_class)
        else:
            raise KeyError(f"未知の selection です: {selection!r}（\"paper\" / \"random\"）")

        #: (クラス名, 動画のパス, ラベル, 開始フレーム, 総フレーム数の参考値（無ければ None）) のリスト
        self.clips: list[tuple[str, Path, int, int, int | None]] = []
        missing = []
        for v in videos:
            cls = class_of(v)
            path = self._find(cls, v)
            if path is None:
                path = self.base / cls / f"{v}.avi"
                missing.append(path)
            start, total = starts.get(v, (0, None))
            self.clips.append((cls, path, UCF101_CLASSES.index(cls), start, total))

        if missing:
            if not download:
                raise FileNotFoundError(
                    f"UCF-101 の動画が {len(missing)} 本見つかりません（例: {missing[0]}）。\n"
                    f"<root>/UCF-101/<クラス名>/<動画名>.avi の形で置くか、download: true にしてください。"
                )
            members = [f"UCF-101/{p.parent.name}/{p.name}" for p in missing]
            fetch_zip_members(UCF101_ZIP_URLS, members, self.base.parent)

        # 開始フレームから clip_len × frame_step フレームに足りない動画（random でまれにある動画や、paper の 15fps 実験での
        # 5 本）は、ここで 1 回読んで外す（読んだ結果はキャッシュに残る）
        kept = []
        for i, clip in enumerate(self.clips):
            try:
                self._load_u8(i)
            except _TooShortError as e:
                print(f"除外: {e}")
                continue
            kept.append(clip)
        if not kept:
            raise ValueError(f"clip_len={clip_len}（frame_step={frame_step}）のフレーム数がある動画がありません")
        self.clips = kept

    def _find(self, cls: str, video: str) -> Path | None:
        """手元にある動画（またはフレームのディレクトリ）を探す。"""
        for p in [self.base / cls / video, *(self.base / cls / f"{video}{e}" for e in _VIDEO_EXTS)]:
            if p.exists():
                return p
        return None

    def _random_videos(self, num_clips: int, seed: int, download: bool, groups: set[int] | None = None, per_class: int = 1) -> list[str]:
        paper = {v for v, _, _ in PAPER_CLIPS}

        def keep(names: list[str]) -> list[str]:
            # 評価用の PAPER_CLIPS とは重ならないようにする
            return sorted(v for v in names if (groups is None or group_of(v) in groups) and v not in paper)

        local = {}
        if self.base.is_dir():
            for d in self.base.iterdir():
                if d.is_dir() and not d.name.startswith("."):
                    vids = keep([p.stem if p.is_file() else p.name for p in d.iterdir() if p.suffix.lower() in _VIDEO_EXTS or p.is_dir()])
                    if len(vids) >= per_class:
                        local[d.name] = vids
        if len(local) < num_clips:
            if not download:
                raise FileNotFoundError(f"UCF-101 のクラスが {len(local)} 個しか見つかりません: {self.base}")
            local = {}
            for name in list_remote_zip(UCF101_ZIP_URLS):
                parts = name.split("/")
                if len(parts) == 3 and parts[2].endswith(".avi"):
                    local.setdefault(parts[1], []).append(parts[2][:-4])
            local = {k: keep(v) for k, v in local.items()}
            local = {k: v for k, v in local.items() if len(v) >= per_class}
            if len(local) < num_clips:
                raise ValueError(f"条件（groups={sorted(groups) if groups else None}, clips_per_class={per_class}）に合うクラスが {len(local)} 個しかありません")
        rng = random.Random(seed)
        classes = sorted(rng.sample(sorted(local), num_clips))
        if per_class == 1:
            return [rng.choice(local[c]) for c in classes]
        return [v for c in classes for v in sorted(rng.sample(local[c], per_class))]

    def __len__(self) -> int:
        return len(self.clips)

    def clip_name(self, i: int) -> str:
        return self.clips[i][1].stem

    def __getitem__(self, i: int):
        x = self._load_u8(i)
        mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
        return (x.float() / 255.0 - mean) / std, self.clips[i][2]

    def _load_u8(self, i: int) -> torch.Tensor:
        """i 番目のクリップを (T, 3, size, size) uint8 で返す（キャッシュがあればそれを使い、なければ作る）。"""
        _, path, _, start, total = self.clips[i]
        need = self.clip_len * self.frame_step
        cache_path = None
        if self.cache_dir is not None:
            # bl = バイリニア（アンチエイリアスなし）。前処理を変えた古いキャッシュとは名前で区別する
            cache_path = self.cache_dir / f"{path.stem}_f{start}_T{self.clip_len}_s{self.frame_step}_{self.size}_bl_u8.pt"
        if cache_path is not None and cache_path.exists():
            x = torch.load(cache_path, weights_only=True)
        else:
            frames = _read_frames(path, need, start, total)  # (n, H, W, 3) uint8
            if len(frames) < need:
                raise _TooShortError(
                    f"{path.stem} は開始フレーム {start} から {need} フレーム（clip_len={self.clip_len} × frame_step={self.frame_step}）"
                    f"取れません（{len(frames)} フレームしかない）"
                )
            x = frames[:: self.frame_step].permute(0, 3, 1, 2).float()
            # 論文と同じく cv2.resize の INTER_LINEAR 相当（アンチエイリアスなし）
            x = F.interpolate(x, size=(self.size, self.size), mode="bilinear", align_corners=False, antialias=False)
            # キャッシュは uint8 で持つ（float32 の 1/4 の容量）。キャッシュの有無で入力が変わらないよう、常にここで丸める
            x = x.round().clamp(0, 255).to(torch.uint8)
            if cache_path is not None:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(x, cache_path)
        return x


class _TooShortError(ValueError):
    """動画のフレーム数が clip_len に足りない。"""


def _read_frames(path: Path, n: int, start: int = 0, expected_total: int | None = None) -> torch.Tensor:
    """動画（またはフレーム画像のディレクトリ）のフレーム start〜start+n−1（0 始まり）を (≤n, H, W, 3) uint8（RGB）で読む。

    動画の末尾を超える分は返さない（呼び出し側で長さを確認する）。expected_total（総フレーム数の参考値）を渡すと、
    最後まで数えて照合し、違えば警告を出す（デコーダの違いでフレームの対応がずれている可能性があるため）。
    """
    import numpy as np

    if path.is_dir():
        from PIL import Image

        files = sorted(p for p in path.iterdir() if p.suffix.lower() in _IMAGE_EXTS)
        count = len(files)
        frames = [np.asarray(Image.open(p).convert("RGB")) for p in files[start : start + n]]
    else:
        try:
            import av
        except ImportError as e:
            raise ImportError("動画の読み込みには `pip install av`（または pip install -e \".[video]\"）が必要です") from e
        frames = []
        count = 0
        with av.open(str(path)) as container:
            for frame in container.decode(video=0):
                if start <= count < start + n:
                    frames.append(frame.to_ndarray(format="rgb24"))
                count += 1
                if count >= start + n and expected_total is None:
                    break
    if expected_total is not None and count != expected_total:
        warnings.warn(
            f"{path.name} の総フレーム数が {count} で、参考値 {expected_total} と違います。"
            f"デコーダの違いで、開始フレーム {start} の位置が論文とずれている可能性があります",
            stacklevel=2,
        )
    if not frames:
        return torch.zeros(0, 1, 1, 3, dtype=torch.uint8)
    return torch.from_numpy(np.stack(frames))


@register_dataset("ucf101")
def _build_ucf101(root: str, split: str, **kwargs):
    return UCF101Clips(root, **kwargs)

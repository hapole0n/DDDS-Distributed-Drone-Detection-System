"""Convert arbitrary user-supplied media files into normalised mono WAVs.

Uses the `ffmpeg` binary shipped with the `imageio-ffmpeg` wheel, so the
project works on a clean Windows install without a system-wide ffmpeg.
"""

from __future__ import annotations

import subprocess
from collections.abc import Iterable
from pathlib import Path

import imageio_ffmpeg

from drone_detector.config import SUPPORTED_MEDIA_EXTS, settings


def ffmpeg_exe() -> str:
    return imageio_ffmpeg.get_ffmpeg_exe()


def list_media_files(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_MEDIA_EXTS
    )


def media_to_wav(src: Path, dst: Path, sample_rate: int | None = None) -> Path:
    """Decode `src` (any ffmpeg-readable container) into a mono WAV at `dst`.

    Re-encoding parameters are pinned so all downstream features are reproducible.
    Skips the conversion if `dst` is newer than `src`.
    """
    sr = sample_rate or settings.sample_rate
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
        return dst

    cmd = [
        ffmpeg_exe(),
        "-y",                  # overwrite without prompting
        "-loglevel", "error",
        "-i", str(src),
        "-vn",                 # drop video stream if any
        "-ac", "1",            # mono
        "-ar", str(sr),        # target sample rate
        "-sample_fmt", "s16",  # 16-bit PCM
        str(dst),
    ]
    subprocess.run(cmd, check=True)
    return dst


def ingest_label_folder(label: str, src_dir: Path, dst_dir: Path) -> list[Path]:
    """Convert every media file in `src_dir` to a WAV in `dst_dir`.

    Returns the list of produced WAV paths.
    """
    files = list_media_files(src_dir)
    produced: list[Path] = []
    for f in files:
        out = dst_dir / f"{f.stem}.wav"
        media_to_wav(f, out)
        produced.append(out)
    return produced


def ingest_all(labels: Iterable[str] | None = None) -> dict[str, list[Path]]:
    """Walk `audio/<label>/` for every label and produce `data/raw/<label>/*.wav`."""
    settings.ensure_dirs()
    audio_root = settings.audio_dir
    if labels is None:
        labels = [p.name for p in audio_root.iterdir() if p.is_dir()]
    out: dict[str, list[Path]] = {}
    for label in labels:
        src = audio_root / label
        dst = settings.raw_dir / label
        out[label] = ingest_label_folder(label, src, dst)
    return out

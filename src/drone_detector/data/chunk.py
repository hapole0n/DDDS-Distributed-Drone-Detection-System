"""Slice long WAV recordings into fixed-length, overlapping training windows."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from drone_detector.config import settings


def slice_wav(
    wav_path: Path,
    out_dir: Path,
    chunk_duration: float | None = None,
    overlap: float | None = None,
    sample_rate: int | None = None,
    min_rms: float = 1e-4,
) -> list[Path]:
    """Cut `wav_path` into overlapping fixed-length WAV chunks.

    Silent chunks (RMS below `min_rms`) are skipped — they pollute the
    training set with empty negatives.
    """
    chunk_duration = chunk_duration or settings.chunk_duration
    overlap = overlap if overlap is not None else settings.chunk_overlap
    sr_target = sample_rate or settings.sample_rate

    audio, sr = sf.read(wav_path, always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sr != sr_target:
        # downstream code expects exactly `sr_target`; reject mismatched files
        raise ValueError(f"{wav_path} sample rate is {sr}, expected {sr_target}")

    chunk_len = int(chunk_duration * sr)
    hop = max(1, int(chunk_len * (1.0 - overlap)))
    out_dir.mkdir(parents=True, exist_ok=True)

    produced: list[Path] = []
    n = len(audio)
    if n < chunk_len:
        return produced

    idx = 0
    start = 0
    while start + chunk_len <= n:
        window = audio[start : start + chunk_len].astype(np.float32)
        rms = float(np.sqrt(np.mean(window**2)))
        if rms >= min_rms:
            out_path = out_dir / f"{wav_path.stem}__c{idx:04d}.wav"
            sf.write(out_path, window, sr)
            produced.append(out_path)
            idx += 1
        start += hop
    return produced


def chunk_label(label: str) -> list[Path]:
    """Chunk every raw WAV for a given label into `data/chunks/<label>/`."""
    src_dir = settings.raw_dir / label
    dst_dir = settings.chunks_dir / label
    if not src_dir.exists():
        return []
    produced: list[Path] = []
    for wav in sorted(src_dir.glob("*.wav")):
        produced.extend(slice_wav(wav, dst_dir))
    return produced


def chunk_all() -> dict[str, list[Path]]:
    settings.ensure_dirs()
    out: dict[str, list[Path]] = {}
    for sub in sorted(p for p in settings.raw_dir.iterdir() if p.is_dir()):
        out[sub.name] = chunk_label(sub.name)
    return out

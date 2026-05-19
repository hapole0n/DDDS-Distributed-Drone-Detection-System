"""Slicing logic — write a tiny WAV, chunk it, verify output."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from drone_detector.config import settings
from drone_detector.data.chunk import slice_wav


def test_slice_wav_produces_expected_number_of_chunks(tmp_path: Path) -> None:
    sr = settings.sample_rate
    duration_s = 5.0
    audio = (0.5 * np.sin(2 * np.pi * 200.0 * np.arange(int(sr * duration_s)) / sr)).astype(
        np.float32
    )
    src = tmp_path / "in.wav"
    sf.write(src, audio, sr)

    out_dir = tmp_path / "chunks"
    produced = slice_wav(
        src,
        out_dir,
        chunk_duration=2.0,
        overlap=0.5,
        sample_rate=sr,
    )
    # 5 s file, 2 s chunks with 50% overlap → step 1 s → indices 0,1,2,3 fit
    assert len(produced) == 4
    for p in produced:
        a, sr_out = sf.read(p)
        assert sr_out == sr
        assert len(a) == int(2.0 * sr)


def test_slice_drops_silence(tmp_path: Path) -> None:
    sr = settings.sample_rate
    audio = np.zeros(int(3.0 * sr), dtype=np.float32)  # pure silence
    src = tmp_path / "in.wav"
    sf.write(src, audio, sr)
    produced = slice_wav(src, tmp_path / "chunks", chunk_duration=1.0, overlap=0.0)
    assert produced == []

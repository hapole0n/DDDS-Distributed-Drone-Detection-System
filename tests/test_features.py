"""Smoke tests for the feature extractor — no network, no audio device required."""

from __future__ import annotations

import numpy as np

from drone_detector.config import settings
from drone_detector.features.audio import (
    feature_dim,
    log_mel_spectrogram,
    mfcc_feature_vector,
    speed_of_sound,
)


def _synth_tone(freq: float, duration_s: float, sr: int) -> np.ndarray:
    t = np.arange(int(sr * duration_s)) / sr
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_feature_dim_matches_extractor() -> None:
    y = _synth_tone(120.0, settings.chunk_duration, settings.sample_rate)
    vec = mfcc_feature_vector(y, settings.sample_rate)
    assert vec.shape == (feature_dim(),)
    assert np.isfinite(vec).all()


def test_two_different_tones_produce_different_features() -> None:
    sr = settings.sample_rate
    y_low = _synth_tone(120.0, settings.chunk_duration, sr)   # Shahed-ish fundamental
    y_high = _synth_tone(4000.0, settings.chunk_duration, sr)
    v1 = mfcc_feature_vector(y_low, sr)
    v2 = mfcc_feature_vector(y_high, sr)
    assert not np.allclose(v1, v2)


def test_log_mel_shape() -> None:
    sr = settings.sample_rate
    y = _synth_tone(440.0, 1.0, sr)
    S = log_mel_spectrogram(y, sr)
    assert S.ndim == 2
    assert S.shape[0] == settings.n_mels


def test_speed_of_sound_reasonable() -> None:
    c = speed_of_sound(20.0, 50.0)
    assert 340.0 < c < 350.0

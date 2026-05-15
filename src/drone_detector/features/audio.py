"""Audio feature extraction.

Two layers of features are exposed:

1. `mfcc_feature_vector` — a fixed-length 1-D vector (mean + std of MFCC
   coefficients + their first temporal derivatives). Designed for classical
   ML (Random Forest, SVM, MLP).
2. `log_mel_spectrogram` — a 2-D time–frequency representation, used for
   visualisation in the UI and as a future input to CNN models.

Why MFCC mean + std + delta?
  * Mean captures the average timbre of the window.
  * Std captures variability — a Shahed engine has very steady tonal
    components, while wind/noise is highly variable in spectrum.
  * Delta-mean captures temporal evolution of the spectrum, which separates
    a tonal engine from a sustained broadband noise.
"""

from __future__ import annotations

import librosa
import numpy as np

from drone_detector.config import settings


def mfcc_feature_vector(
    y: np.ndarray,
    sr: int,
    n_mfcc: int | None = None,
    n_fft: int | None = None,
    hop_length: int | None = None,
) -> np.ndarray:
    """Return a 1-D feature vector of length 3 * n_mfcc (mean + std + delta-mean)."""
    n_mfcc = n_mfcc or settings.n_mfcc
    n_fft = n_fft or settings.n_fft
    hop_length = hop_length or settings.hop_length

    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length
    )
    mfcc_mean = mfcc.mean(axis=1)
    mfcc_std = mfcc.std(axis=1)
    delta = librosa.feature.delta(mfcc)
    delta_mean = delta.mean(axis=1)
    return np.concatenate([mfcc_mean, mfcc_std, delta_mean]).astype(np.float32)


def feature_dim(n_mfcc: int | None = None) -> int:
    return 3 * (n_mfcc or settings.n_mfcc)


def log_mel_spectrogram(
    y: np.ndarray,
    sr: int,
    n_mels: int | None = None,
    n_fft: int | None = None,
    hop_length: int | None = None,
) -> np.ndarray:
    n_mels = n_mels or settings.n_mels
    n_fft = n_fft or settings.n_fft
    hop_length = hop_length or settings.hop_length

    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length, power=2.0
    )
    return librosa.power_to_db(mel, ref=np.max)


def speed_of_sound(temp_c: float, humidity_pct: float = 50.0) -> float:
    """Approximate speed of sound in air, m/s.

    Cramer (1993) light formulation: c ~ 331.3 + 0.606*T + 0.0124*RH.
    """
    return 331.3 + 0.606 * temp_c + 0.0124 * humidity_pct

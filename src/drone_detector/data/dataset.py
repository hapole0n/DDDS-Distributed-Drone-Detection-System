"""Load chunked WAVs, extract features, materialise as a single `.npz` dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from drone_detector.config import settings
from drone_detector.features.audio import mfcc_feature_vector


@dataclass
class Dataset:
    X: np.ndarray            # shape (n_samples, n_features)
    y: np.ndarray            # shape (n_samples,) — integer class ids
    labels: list[str]        # human-readable class names indexed by class id
    sources: np.ndarray      # original WAV file path per row (for traceability)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            X=self.X,
            y=self.y,
            labels=np.array(self.labels),
            sources=self.sources,
        )

    @classmethod
    def load(cls, path: Path) -> Dataset:
        d = np.load(path, allow_pickle=True)
        return cls(
            X=d["X"],
            y=d["y"],
            labels=list(map(str, d["labels"].tolist())),
            sources=d["sources"],
        )


def build_dataset(chunks_root: Path | None = None) -> Dataset:
    """Walk `data/chunks/<label>/*.wav` and produce a feature/label matrix."""
    chunks_root = chunks_root or settings.chunks_dir
    label_dirs = sorted(p for p in chunks_root.iterdir() if p.is_dir())
    if not label_dirs:
        raise FileNotFoundError(
            f"No class folders found under {chunks_root}. Run chunking first."
        )

    labels = [p.name for p in label_dirs]
    label_to_id = {name: i for i, name in enumerate(labels)}

    X_rows: list[np.ndarray] = []
    y_rows: list[int] = []
    src_rows: list[str] = []

    for label_dir in label_dirs:
        cid = label_to_id[label_dir.name]
        for wav in sorted(label_dir.glob("*.wav")):
            audio, sr = sf.read(wav, always_2d=False)
            if audio.ndim == 2:
                audio = audio.mean(axis=1)
            vec = mfcc_feature_vector(audio.astype(np.float32), sr)
            X_rows.append(vec)
            y_rows.append(cid)
            src_rows.append(str(wav))

    if not X_rows:
        raise FileNotFoundError(f"No WAV chunks found under {chunks_root}.")

    return Dataset(
        X=np.stack(X_rows),
        y=np.asarray(y_rows, dtype=np.int64),
        labels=labels,
        sources=np.array(src_rows),
    )

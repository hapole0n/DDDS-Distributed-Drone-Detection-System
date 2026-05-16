"""Random Forest baseline classifier for UAV acoustic detection."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from drone_detector.config import settings


@dataclass
class ModelBundle:
    """A trained classifier together with the metadata needed for inference."""

    clf: RandomForestClassifier
    labels: list[str]
    feature_dim: int
    sample_rate: int
    chunk_duration: float
    n_mfcc: int
    meta: dict = field(default_factory=dict)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict(X)

    def positive_class_id(self, name: str = "shahed") -> int | None:
        try:
            return self.labels.index(name)
        except ValueError:
            return None

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        meta_path = path.with_suffix(".json")
        meta_path.write_text(
            json.dumps(
                {
                    "labels": self.labels,
                    "feature_dim": self.feature_dim,
                    "sample_rate": self.sample_rate,
                    "chunk_duration": self.chunk_duration,
                    "n_mfcc": self.n_mfcc,
                    "meta": self.meta,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> ModelBundle:
        return joblib.load(path)


def train_random_forest(
    X: np.ndarray,
    y: np.ndarray,
    labels: list[str],
    random_state: int | None = None,
) -> RandomForestClassifier:
    clf = RandomForestClassifier(
        n_estimators=settings.rf_n_estimators,
        max_depth=settings.rf_max_depth,
        class_weight=settings.rf_class_weight,
        random_state=random_state or settings.random_state,
        n_jobs=-1,
    )
    clf.fit(X, y)
    return clf

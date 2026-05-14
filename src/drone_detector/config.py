"""Central configuration: paths, audio parameters, model hyper-parameters.

All values are overridable via environment variables prefixed with `DD_`.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DD_", env_file=".env", extra="ignore")

    # --- Filesystem layout ---
    project_root: Path = PROJECT_ROOT
    audio_dir: Path = PROJECT_ROOT / "audio"
    raw_dir: Path = PROJECT_ROOT / "data" / "raw"
    chunks_dir: Path = PROJECT_ROOT / "data" / "chunks"
    features_dir: Path = PROJECT_ROOT / "data" / "features"
    models_dir: Path = PROJECT_ROOT / "models"
    reports_dir: Path = PROJECT_ROOT / "reports"

    # --- Audio parameters ---
    sample_rate: int = Field(22050, description="Target mono sample rate in Hz")
    chunk_duration: float = Field(2.0, description="Length of one training window in seconds")
    chunk_overlap: float = Field(0.5, description="Overlap fraction between consecutive chunks")

    # --- Feature extraction ---
    n_mfcc: int = 40
    n_fft: int = 2048
    hop_length: int = 512
    n_mels: int = 128

    # --- Model ---
    rf_n_estimators: int = 300
    rf_max_depth: int | None = None
    rf_class_weight: str | None = "balanced"
    random_state: int = 42
    test_size: float = 0.2

    # --- Live detection ---
    live_window_seconds: float = 2.0
    live_hop_seconds: float = 0.5
    detection_threshold: float = 0.5

    def ensure_dirs(self) -> None:
        for d in (
            self.audio_dir,
            self.raw_dir,
            self.chunks_dir,
            self.features_dir,
            self.models_dir,
            self.reports_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)


settings = Settings()


SUPPORTED_MEDIA_EXTS: tuple[str, ...] = (
    ".mp4", ".mkv", ".webm", ".mov", ".avi", ".flv", ".m4a",
    ".mp3", ".wav", ".flac", ".ogg", ".opus", ".aac",
)

"""End-to-end training pipeline: audio/ -> chunks -> dataset -> model -> report."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from rich.console import Console
from rich.table import Table
from sklearn.model_selection import train_test_split

from drone_detector.config import settings
from drone_detector.data.chunk import chunk_all
from drone_detector.data.dataset import Dataset, build_dataset
from drone_detector.data.ingest import ingest_all
from drone_detector.features.audio import feature_dim
from drone_detector.models.baseline import ModelBundle, train_random_forest
from drone_detector.pipeline.evaluate import write_report


console = Console()


def run_full_pipeline(
    skip_ingest: bool = False,
    skip_chunk: bool = False,
    skip_features: bool = False,
) -> ModelBundle:
    """Execute the entire video -> trained model -> report pipeline."""
    settings.ensure_dirs()

    if not skip_ingest:
        console.rule("[bold cyan]1/5  Ingest video -> WAV")
        ingested = ingest_all()
        for label, files in ingested.items():
            console.print(f"  {label}: {len(files)} source file(s)")

    if not skip_chunk:
        console.rule("[bold cyan]2/5  Slice WAV -> fixed-length chunks")
        chunked = chunk_all()
        for label, files in chunked.items():
            console.print(f"  {label}: {len(files)} chunk(s)")

    features_path = settings.features_dir / "dataset.npz"
    if skip_features and features_path.exists():
        console.rule("[bold cyan]3/5  Load cached feature matrix")
        ds = Dataset.load(features_path)
    else:
        console.rule("[bold cyan]3/5  Extract MFCC features")
        ds = build_dataset()
        ds.save(features_path)
    console.print(
        f"  X: {ds.X.shape}, y: {ds.y.shape}, classes: {ds.labels}"
    )

    if len(ds.labels) < 2:
        raise RuntimeError(
            "Need at least 2 labelled classes to train. "
            "Put MP3 (or other audio) files in audio/shahed/ AND audio/noise/."
        )

    console.rule("[bold cyan]4/5  Train Random Forest")
    X_tr, X_te, y_tr, y_te = train_test_split(
        ds.X,
        ds.y,
        test_size=settings.test_size,
        random_state=settings.random_state,
        stratify=ds.y if len(np.unique(ds.y)) > 1 else None,
    )
    clf = train_random_forest(X_tr, y_tr, ds.labels)

    bundle = ModelBundle(
        clf=clf,
        labels=ds.labels,
        feature_dim=feature_dim(),
        sample_rate=settings.sample_rate,
        chunk_duration=settings.chunk_duration,
        n_mfcc=settings.n_mfcc,
        meta={
            "n_train": int(len(y_tr)),
            "n_test": int(len(y_te)),
            "class_counts_train": {
                ds.labels[c]: int((y_tr == c).sum()) for c in range(len(ds.labels))
            },
            "class_counts_test": {
                ds.labels[c]: int((y_te == c).sum()) for c in range(len(ds.labels))
            },
        },
    )

    bundle.save(settings.models_dir / "baseline.pkl")
    console.print(f"  Saved model to {settings.models_dir / 'baseline.pkl'}")

    console.rule("[bold cyan]5/5  Evaluate + write report")
    report_path = write_report(bundle, X_te, y_te, settings.reports_dir)
    console.print(f"  Wrote report to {report_path}")

    _print_summary(bundle)
    return bundle


def _print_summary(bundle: ModelBundle) -> None:
    table = Table(title="Trained model summary", show_header=True)
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("classes", ", ".join(bundle.labels))
    table.add_row("feature_dim", str(bundle.feature_dim))
    table.add_row("sample_rate", f"{bundle.sample_rate} Hz")
    table.add_row("chunk_duration", f"{bundle.chunk_duration} s")
    for k, v in bundle.meta.items():
        table.add_row(k, str(v))
    console.print(table)


def main() -> int:
    run_full_pipeline()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Typer-based CLI: `drone-detector <command>`."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from drone_detector.config import settings
from drone_detector.data.chunk import chunk_all
from drone_detector.data.dataset import build_dataset
from drone_detector.data.ingest import ingest_all
from drone_detector.pipeline.train import run_full_pipeline

app = typer.Typer(add_completion=False, help="Acoustic UAV detector CLI.")
console = Console()


@app.command()
def ingest() -> None:
    """Decode every audio/media file under audio/<label>/ into a mono WAV."""
    result = ingest_all()
    for label, files in result.items():
        console.print(f"{label}: {len(files)} file(s) processed")


@app.command()
def chunk() -> None:
    """Slice every raw WAV into fixed-length training windows."""
    result = chunk_all()
    for label, files in result.items():
        console.print(f"{label}: {len(files)} chunk(s) produced")


@app.command()
def features() -> None:
    """Extract MFCC features and dump dataset.npz."""
    ds = build_dataset()
    out = settings.features_dir / "dataset.npz"
    ds.save(out)
    console.print(f"Saved {ds.X.shape} feature matrix to {out}")


@app.command()
def train(
    skip_ingest: bool = typer.Option(False, help="Skip video -> WAV step"),
    skip_chunk: bool = typer.Option(False, help="Skip WAV -> chunk step"),
    skip_features: bool = typer.Option(
        False, help="Reuse cached dataset.npz if present"
    ),
) -> None:
    """Run the full training pipeline end-to-end."""
    run_full_pipeline(
        skip_ingest=skip_ingest,
        skip_chunk=skip_chunk,
        skip_features=skip_features,
    )


@app.command()
def live(
    device: int | None = typer.Option(None, help="Input device index (sounddevice)"),
    model: Path = typer.Option(
        settings.models_dir / "baseline.pkl",
        help="Path to a trained model bundle",
    ),
    duration: float = typer.Option(0.0, help="Seconds to record (0 = forever)"),
) -> None:
    """Stream from a microphone and print live detection probabilities."""
    from drone_detector.live.mic import run_console_live

    run_console_live(model_path=model, device=device, duration_s=duration)


@app.command()
def ui() -> None:
    """Open the Streamlit dashboard."""
    import subprocess
    import sys

    here = Path(__file__).resolve()
    project_root = here.parents[3]
    app_path = project_root / "app" / "streamlit_app.py"
    cmd = [sys.executable, "-m", "streamlit", "run", str(app_path)]
    subprocess.run(cmd, check=False)


@app.command()
def devices() -> None:
    """List available input audio devices."""
    import sounddevice as sd

    for idx, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] > 0:
            console.print(
                f"[{idx}] {dev['name']}  "
                f"in_ch={dev['max_input_channels']}  "
                f"sr={int(dev['default_samplerate'])}"
            )


if __name__ == "__main__":
    app()

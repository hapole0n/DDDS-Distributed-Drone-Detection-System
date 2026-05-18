"""Convenience wrapper: `python scripts/run_pipeline.py`."""

from drone_detector.pipeline.train import run_full_pipeline


if __name__ == "__main__":
    run_full_pipeline()

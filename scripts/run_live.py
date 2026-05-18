"""Convenience wrapper: `python scripts/run_live.py [--device IDX]`."""

import argparse

from drone_detector.config import settings
from drone_detector.live.mic import run_console_live


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=int, default=None, help="sounddevice input index")
    p.add_argument("--duration", type=float, default=0.0,
                   help="seconds to record (0 = forever)")
    p.add_argument("--model", default=str(settings.models_dir / "baseline.pkl"))
    args = p.parse_args()
    run_console_live(
        model_path=args.model,
        device=args.device,
        duration_s=args.duration,
    )


if __name__ == "__main__":
    main()

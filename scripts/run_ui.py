"""Convenience wrapper: `python scripts/run_ui.py` -> starts the Streamlit app."""

import subprocess
import sys
from pathlib import Path


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    app = project_root / "app" / "streamlit_app.py"
    return subprocess.call(
        [sys.executable, "-m", "streamlit", "run", str(app)]
    )


if __name__ == "__main__":
    raise SystemExit(main())

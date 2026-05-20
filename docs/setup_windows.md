# Setup on a fresh Windows machine

A step-by-step guide to go from "Python 3.10+ and git, nothing else" to a
running dashboard with live microphone detection. All commands assume
**PowerShell**.

## 1. Clone the repository

```powershell
cd C:\Users\cebularz
git clone <your-github-url> drone-detector
cd drone-detector
```

(If you are working from the local copy produced during the project
session, just `cd drone-detector`.)

## 2. Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks the activation script with an execution-policy
error, run once:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

## 3. Install dependencies + the project itself

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

The final command does an **editable install** of the `drone_detector`
package (defined by `pyproject.toml`). Without it, the `python -m
drone_detector.*` command will fail with `ModuleNotFoundError`, because
the source code lives under `src/` (the standard "src-layout"). After the
editable install, two equivalent ways to invoke the CLI become available:

```powershell
python -m drone_detector.pipeline.cli train       # long form
drone-detector train                              # short form (console script)
```

> **Note for Python 3.14 users.** Several scientific wheels (`numba`,
> `librosa`, `scikit-learn`) may not yet publish a 3.14 wheel and will
> attempt a source build that fails without a C/C++ toolchain. If `pip
> install` errors out, install Python 3.11 or 3.12 alongside 3.14 and
> create the venv against the older interpreter:
>
> ```powershell
> Remove-Item -Recurse -Force .venv
> py -3.12 -m venv .venv         # or: py -3.11 -m venv .venv
> .\.venv\Scripts\Activate.ps1
> python -m pip install --upgrade pip
> pip install -r requirements.txt
> pip install -e .
> ```

This pulls in:

* `librosa`, `soundfile`, `sounddevice` (audio I/O + features)
* `scikit-learn`, `joblib` (Random Forest + persistence)
* `imageio-ffmpeg` (bundled `ffmpeg.exe` — no system install needed)
* `yt-dlp` (optional YouTube downloader)
* `streamlit`, `matplotlib`, `pandas`, `pydantic-settings`, `typer`, `rich`

Total install size on a clean machine is approximately 1.0–1.5 GB.

## 4. Verify the USB microphone is visible

Plug in the **Maono** USB microphone, then:

```powershell
python -m drone_detector.pipeline.cli devices
```

You should see a line similar to

```
[2] Maono ...  in_ch=1  sr=48000
```

Remember the index (`2` in this example) — you will pass it to `live`.

## 5. Drop training MP3 files in place

```powershell
mkdir audio\shahed  # already exists if you cloned the repo
mkdir audio\noise
# Move/copy your .mp3 (or .wav / .m4a / .mp4 / ...) files into the
# corresponding folder.
```

Alternatively, prepare two URL lists and let `yt-dlp` do the download:

```powershell
notepad audio\shahed.urls.txt   # one YouTube URL per line
notepad audio\noise.urls.txt
python -m drone_detector.data.youtube audio\shahed.urls.txt audio\shahed
python -m drone_detector.data.youtube audio\noise.urls.txt  audio\noise
```

## 6. Run the full training pipeline

```powershell
python -m drone_detector.pipeline.cli train
```

The first run will spend most of its time on the ingest step (ffmpeg
decoding). Subsequent runs can skip already-decoded files automatically.

When it finishes, look in `reports/`:

* `evaluation.md`            — full classification report + ROC numbers
* `confusion_matrix.png`     — confusion matrix on the test split
* `roc.png`                  — binary ROC, Shahed as positive class

## 7. Launch the dashboard

```powershell
python -m drone_detector.pipeline.cli ui
```

then open <http://localhost:8501> in your browser.

* **Overview** — model status + dataset stats + the latest report.
* **Train pipeline** — re-run training without touching the terminal.
* **Analyse a file** — drag a video/audio onto the page and inspect the
  detection probability timeline.
* **Live microphone** — start streaming from the Maono and watch the
  probability bar move.
* **4-mic TDOA concept** — interactive paper-design page for the
  multi-channel localisation extension.

## 8. Live demo on the defence day

The smoothest demo path is:

1. Open the dashboard.
2. Go to the **Live microphone** page.
3. Pick the Maono device in the dropdown and press *Start*.
4. Play a Shahed-136 flight video on your phone, positioned ~30 cm from
   the Maono.
5. The probability bar should jump within ~1 second.

## 9. Optional: run the test suite

```powershell
pytest -q
```

Smoke tests for the feature extractor and chunking are independent of any
audio device or training data — they should pass on a clean install.

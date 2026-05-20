# QUICKSTART — what to do *right now*

Step-by-step, copy-paste PowerShell. Designed to fit inside the
remaining time before the defence.

## 0. Where you are

```
C:\Users\cebularz\drone-detector\
```

The repository is already `git init`-ed and has an initial commit. You
have not pushed it anywhere yet — that is step 6.

## 1. Create a virtual environment (≈ 1 minute)

```powershell
cd C:\Users\cebularz\drone-detector
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

If `Activate.ps1` is blocked:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

## 2. Install dependencies + the project itself (≈ 5–10 minutes; bandwidth-bound)

```powershell
pip install -r requirements.txt
pip install -e .
```

`pip install -e .` is the **editable install** of this project. Without it
`python -m drone_detector.pipeline.cli` fails with
`ModuleNotFoundError: No module named 'drone_detector'`, because the
package lives under `src/` (the standard "src-layout" used by serious
Python projects). After this step, two things become available from any
shell where the venv is active:

* `python -m drone_detector.pipeline.cli <command>` — the long form.
* `drone-detector <command>` — a console script auto-generated from the
  `[project.scripts]` entry in `pyproject.toml`.

> **Python 3.14 note.** Some scientific wheels (`numba`, `librosa`,
> `scikit-learn`) may not yet ship a 3.14 wheel and will try to build
> from source, which usually fails without a C compiler. If `pip
> install` errors out on 3.14, install Python **3.11** or **3.12** from
> <https://www.python.org/downloads/> and recreate the venv against that
> interpreter:
>
> ```powershell
> Remove-Item -Recurse -Force .venv
> py -3.12 -m venv .venv     # or py -3.11
> .\.venv\Scripts\Activate.ps1
> python -m pip install --upgrade pip
> pip install -r requirements.txt
> pip install -e .
> ```

While this is downloading, move on to step 3.

## 3. Drop your MP3 files (≈ 2 minutes while pip runs)

Copy your prepared MP3 files into the two class folders:

```
audio\shahed\   ← Shahed-136 flight recordings (.mp3)
audio\noise\    ← city traffic, wind, birds, indoor ambience, etc. (.mp3)
```

Aim for at least **5 source clips per class** for the model to be
defensible. The pipeline accepts any `ffmpeg`-readable format
(`.mp3`, `.wav`, `.flac`, `.ogg`, `.opus`, `.m4a`, `.mp4`, …) — but MP3 is
the expected default for this project.

Optional — pull more recordings from YouTube automatically:

```powershell
# create a URL list, one per line, then:
python -m drone_detector.data.youtube audio\shahed.urls.txt audio\shahed
python -m drone_detector.data.youtube audio\noise.urls.txt  audio\noise
```

## 4. Run the full pipeline (≈ 1–5 minutes depending on data)

```powershell
python -m drone_detector.pipeline.cli train
```

This prints a 5-step progress log:
1. Ingest video → WAV
2. Slice WAV → 2-second chunks
3. Extract MFCC features → `data/features/dataset.npz`
4. Train Random Forest → `models/baseline.pkl`
5. Evaluate + write report → `reports/evaluation.md`

If something goes wrong, the error stack trace plus a `rich`-coloured
status block will tell you exactly which stage failed.

## 5. Open the dashboard

```powershell
python -m drone_detector.pipeline.cli ui
```

Browser opens at <http://localhost:8501>. Visit each page in order:

1. **Overview** — confirms model + dataset are loaded.
2. **Train pipeline** — alternative entrypoint to step 4 if you change data later.
3. **Analyse a file** — drag any video onto the page; watch the probability
   timeline.
4. **Live microphone** — pick your Maono in the dropdown, press *Start*,
   play a Shahed clip on a phone next to the mic.
5. **4-mic TDOA concept** — interactive paper design page.

## 6. Push to GitHub (≈ 2 minutes)

Create an empty repo on github.com (e.g. `drone-detector`) — do NOT
initialise it with README or .gitignore.

```powershell
git remote add origin https://github.com/<your-username>/drone-detector.git
git push -u origin main
```

## 7. Run the smoke tests (optional but quick)

```powershell
pytest -q
```

You should see 5 passing tests in under a few seconds.

## 8. Defence ammunition

Read these files in this order — they were written so you can quote them
verbatim if asked:

1. `README.md` — top-level pitch and architecture diagram.
2. `docs/methodology.md` — why MFCC, why Random Forest, why 22050 Hz.
3. `docs/limitations.md` — every honest caveat you might be asked about.
4. `docs/tdoa_concept.md` — full paper design of the 4-mic extension.
5. `docs/defense_talking_points.md` — a 10-minute talk outline plus
   crisp answers to the most likely questions.

## Troubleshooting

| Symptom | Likely fix |
|---|---|
| `ModuleNotFoundError: drone_detector` | Two causes: (1) you are not in the project directory — `cd C:\Users\cebularz\drone-detector`; (2) the editable install was skipped — run `pip install -e .` inside the activated venv. |
| `pip install` fails compiling `numba` / `llvmlite` from source | Your Python is too new (e.g. 3.14). Install Python 3.11 or 3.12 and recreate the venv against it — see step 2 in this file. |
| `ffmpeg` not found | You should not need a system ffmpeg — `imageio-ffmpeg` ships one. `pip show imageio-ffmpeg` should print a version. |
| Maono not in `devices` list | Unplug + replug, then re-run `python -m drone_detector.pipeline.cli devices`. Windows sometimes reserves it for another app — close Skype/Discord. |
| `PortAudioError: Invalid sample rate` | Open Windows sound settings and set Maono input to 44100 or 48000 Hz, or pass `--device-sr` if you add such an option later. |
| Streamlit slow to start | First run compiles bytecode cache; the second is instant. |
| `youtube_dl: Unable to extract uploader id` | `yt-dlp` is intentionally fast-moving; if a video fails just download the audio manually and drop the MP3 into `audio\shahed\`. |
| Random Forest gives `P ≈ 0.5` everywhere | Too few training samples or the two folders are too similar. Add more diverse `noise/` content. |

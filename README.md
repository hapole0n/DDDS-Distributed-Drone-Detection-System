# Drone Acoustic Detector

> project — Politechnika Lubelska, *Inżynierskie Zastosowania
> Informatyki w Elektrotechnice II stopnia*, course *Programowanie w języku
> Python*. Author: **Oleh Kropyva**.

An end-to-end Python pipeline that learns to recognise a **Shahed-136**
attack-drone flight from raw audio, and runs the trained classifier live
on a single USB microphone. A reproducible video → WAV → chunk → MFCC →
Random-Forest training pipeline, a Streamlit dashboard for analysis, and
a paper-design extension to a 4-microphone TDOA localisation array.

> **Safety boundary.** This software is research for **defensive
> early-warning**. It does not provide target assignment, weapon guidance,
> or engagement coordinates.

---

## Why this project

The Shahed-136 is a fixed-wing single-piston-engine loitering munition
deployed against Ukrainian and (with airspace violations) Polish
infrastructure since 2022. Its propeller and ICE create a very distinctive
*moped-like* acoustic signature — a tonal ~80–120 Hz fundamental with rich
harmonics up to ~5 kHz — that **can** be picked up at hundreds of metres by
a cheap microphone, well before optical detection is possible in low-light
or cloud cover. That makes acoustic detection a viable building block of a
low-cost civil early-warning network.

This project demonstrates the **software stack** of such a node: a
reproducible training pipeline driven entirely from user-supplied MP3
recordings,
a real-time inference path running on a single off-the-shelf USB mic, and
a documented extension toward 4-microphone direction-of-arrival
estimation.

---

## Architecture (one screen)

```
audio/<label>/*.mp3    ┐
yt-dlp URL list       ─┴─▶  ingest.py    ──▶  data/raw/<label>/*.wav    (mono 22050 Hz)
                                              │
                                              ▼
                                        chunk.py      ──▶  data/chunks/<label>/*.wav  (2 s, 50 % overlap)
                                              │
                                              ▼
                                        dataset.py    ──▶  data/features/dataset.npz  (MFCC 3·N vector)
                                              │
                                              ▼
                                        train.py      ──▶  models/baseline.pkl
                                              │           +  reports/{confusion_matrix.png, roc.png, evaluation.md}
                                              ▼
USB Maono microphone ──▶ live/mic.py  ──▶  Streamlit live dashboard
                                              │
                                              ▼
                            4-mic TDOA concept page (docs/tdoa_concept.md)
```

Full diagram and rationale: [docs/architecture.md](docs/architecture.md).

---

## Quick start (Windows, PowerShell)

```powershell
# 1. Create a virtual environment and install everything
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .                # editable install — required for `python -m drone_detector...`

# 2. Put some training data in place
#    drop Shahed MP3 files into audio\shahed\
#    drop background / nature / traffic MP3 files into audio\noise\
#    (any .mp3 / .wav / .flac / .m4a / .mp4 etc. works — MP3 is the default)

# 3. Train end-to-end (ingest -> chunk -> features -> train -> evaluate)
python -m drone_detector.pipeline.cli train

# 4. Open the dashboard
python -m drone_detector.pipeline.cli ui

# 5. Or run a console live-detection session against the USB mic
python -m drone_detector.pipeline.cli devices       # find your input index
python -m drone_detector.pipeline.cli live --device 1
```

The dashboard runs on <http://localhost:8501>.

Detailed setup for a fresh Windows machine (Python, virtual env, `ffmpeg`
shipped with the package): [docs/setup_windows.md](docs/setup_windows.md).

---

## Repository layout

```
drone-detector/
├── README.md                   ← you are here
├── LICENSE                     ← MIT
├── pyproject.toml              ← package metadata + entry points
├── requirements.txt
├── audio/                      ← user input drop folder (gitignored content)
│   ├── shahed/                 ← Shahed-136 MP3 files
│   └── noise/                  ← background / negative-class MP3 files
├── data/                       ← all generated artefacts (gitignored)
│   ├── raw/<label>/*.wav
│   ├── chunks/<label>/*.wav
│   └── features/dataset.npz
├── models/                     ← trained model bundles + metadata JSON
├── reports/                    ← confusion matrix, ROC, evaluation.md
├── src/drone_detector/
│   ├── config.py               ← single source of truth for paths + hp
│   ├── data/
│   │   ├── ingest.py           ← video → WAV (imageio-ffmpeg)
│   │   ├── chunk.py            ← WAV → fixed windows (with silence drop)
│   │   ├── dataset.py          ← chunks → feature matrix
│   │   └── youtube.py          ← optional yt-dlp downloader
│   ├── features/audio.py       ← MFCC + delta + mel spectrogram + c(T, RH)
│   ├── models/baseline.py      ← RandomForest bundle + save/load
│   ├── pipeline/
│   │   ├── train.py            ← orchestrator with rich console output
│   │   ├── evaluate.py         ← confusion matrix + ROC + classification report
│   │   └── cli.py              ← typer CLI: ingest, chunk, train, live, ui, devices
│   ├── live/mic.py             ← sounddevice producer/consumer + DetectionEvent
│   └── viz/plots.py            ← matplotlib helpers for the Streamlit UI
├── app/streamlit_app.py        ← 5-page dashboard (overview, train, analyse, live, 4-mic concept)
├── scripts/                    ← thin wrappers for the CLI commands
├── tests/                      ← pytest smoke tests (features + chunking)
└── docs/
    ├── architecture.md
    ├── methodology.md
    ├── tdoa_concept.md
    ├── limitations.md
    └── setup_windows.md
```

---

## What the pipeline actually does

1. **Ingest.** Every audio file under `audio/<label>/` (MP3 by default,
   but any `ffmpeg`-readable container works) is decoded into a mono
   22050 Hz, 16-bit PCM WAV in `data/raw/<label>/`. The conversion uses
   the `ffmpeg` binary bundled with the
   [`imageio-ffmpeg`](https://pypi.org/project/imageio-ffmpeg/) wheel,
   so no system-wide `ffmpeg` install is required.
2. **Chunk.** Each WAV is sliced into 2-second windows with 50 % overlap.
   Chunks below an RMS threshold are dropped to keep silent intervals out
   of the training set.
3. **Feature extraction.** For every chunk we compute 40 MFCC coefficients
   over the 2-second window and pack a 120-dim feature vector consisting of:
   * 40 MFCC means
   * 40 MFCC standard deviations
   * 40 means of the first temporal derivative (Δ-MFCC)

   Rationale and ablations: [docs/methodology.md](docs/methodology.md).
4. **Training.** A stratified 80 / 20 split, then a Random Forest with
   class-weight balancing. Hyper-parameters and seeds are pinned in
   `src/drone_detector/config.py`.
5. **Evaluation.** Confusion matrix, classification report and (binary
   case) ROC curve are written into `reports/evaluation.md` and embedded
   in the Streamlit *Overview* page.
6. **Inference.** Two paths share the same feature extractor:
   * **File analysis.** Slide a window across an uploaded clip, plot the
     detection probability over time.
   * **Live microphone.** A `sounddevice` producer thread feeds a ring
     buffer; a consumer thread emits a `DetectionEvent` every
     `live_hop_seconds` (default 0.5 s).

---

## Tech stack and why

| Concern | Choice | Rationale |
|---|---|---|
| Language | Python 3.10+ | Course requirement; rich audio + ML ecosystem |
| Audio I/O | `soundfile`, `sounddevice` | PortAudio-based, cross-platform, real-time-friendly |
| Media decoding | `imageio-ffmpeg` | Bundles `ffmpeg` as a pip wheel — zero system setup |
| Feature extraction | `librosa` | Reference implementation of MFCC, mel, delta |
| Classifier | `scikit-learn` RandomForest | Robust, no GPU, interpretable feature importance |
| CLI | `typer` + `rich` | Self-documenting, colour console |
| Dashboard | `streamlit` | Single-language UI built in minutes, professional look |
| Settings | `pydantic-settings` | Typed, env-overridable single source of truth |
| Tests | `pytest` | Smoke tests for features and chunking, no audio device needed |

What was **deliberately not** chosen and why:
* **React/TypeScript frontend** — the original ddds prototype used it, but
  the course is *Programowanie w języku Python*. Streamlit keeps the
  entire stack in one language and removes Node/Vite from the defence
  surface.
* **Docker** — adds operational complexity that is not needed to score
  acoustic windows locally on a laptop. `pip install -r requirements.txt`
  is the deployment path.
* **PyTorch CNN** — the Random Forest baseline is enough to demonstrate
  the methodology. A CNN on log-mel-spectrogram is the natural next step
  and is sketched in `docs/methodology.md`.

---

## Defensive use-case + honest limitations

Read [docs/limitations.md](docs/limitations.md) before drawing any
operational conclusion. The most important caveats:

* The classifier is trained on **YouTube audio**, which is lossy AAC and
  contains commentary / background music in many recordings. Real-world
  performance with a clean field-recorded dataset will be different (in
  either direction).
* A **single mono microphone** cannot localise a sound source. The
  *4-mic TDOA concept* page documents the extension path; it is **not
  implemented as a real-hardware feature** in this revision.
* Detection range and SNR were not measured under controlled conditions
  in this revision. Reported numbers come from the YouTube test split,
  not from a calibrated field trial.

---

## Reproducibility

* All random seeds are fixed in `src/drone_detector/config.py`.
* Audio parameters (sample rate, window length, MFCC count) are
  centralised; they are stored in the model bundle metadata and re-read
  at inference time.
* `reports/evaluation.md` is regenerated on every training run.
* The pipeline is **idempotent**: ingest and chunk steps skip files
  whose output is already up-to-date with the input.

```powershell
pytest -q
```

runs the smoke tests without needing an audio device or any input data.

---

## Roadmap

Short-term (post-defence polish):
* CNN on log-mel-spectrogram for comparison with the RF baseline.
* Per-window feature importance overlay on the spectrogram.
* `--profile` flag to swap between Shahed-centric (low-frequency tonal)
  and multirotor-centric (broadband) MFCC parameters.

Medium-term (hardware extension):
* Replace the simulated 4-mic concept with a real 4 × INMP441 I²S array
  on ESP32-S3 (sample-synchronous capture, sample-accurate timestamps).
* BME280 environmental sensor for live $c(T, RH)$ correction.
* GCC-PHAT direction-of-arrival on the real array.
* Bearing-only Extended Kalman Filter for trajectory inference under a
  constant-altitude assumption.

---

## Acknowledgements & related work

* Shahed-136 acoustic signature characterisation: open-source intelligence
  reports + the Ukrainian *Sky Fortress* civil acoustic-sensor network
  (publicly documented, defensive).
* `librosa` and the MFCC pipeline as taught by Müller, *Fundamentals of
  Music Processing*.
* GCC-PHAT algorithm: Knapp & Carter, *The Generalized Correlation Method
  for Estimation of Time Delay*, IEEE TASSP 1976.

---

## License

MIT — see [LICENSE](LICENSE).

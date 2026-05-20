# Architecture

## High-level data flow

```
                ┌────────────────────────────────────────┐
                │            INPUT DROP FOLDER           │
                │  audio/<label>/*.mp3  (or .wav, .m4a)  │
                │  optional: audio/<label>.urls.txt      │
                └──────────────────┬─────────────────────┘
                                   │
                          (data/youtube.py — optional)
                                   ▼
                ┌────────────────────────────────────────┐
                │           data/ingest.py               │
                │  imageio-ffmpeg → mono 22050 Hz WAV    │
                └──────────────────┬─────────────────────┘
                                   ▼
                ┌────────────────────────────────────────┐
                │           data/chunk.py                │
                │  2-s windows, 50 % overlap, RMS gate   │
                └──────────────────┬─────────────────────┘
                                   ▼
                ┌────────────────────────────────────────┐
                │   features/audio.py + data/dataset.py  │
                │  MFCC mean/std/Δ-mean → 120-D vector   │
                │  → data/features/dataset.npz           │
                └──────────────────┬─────────────────────┘
                                   ▼
                ┌────────────────────────────────────────┐
                │         models/baseline.py             │
                │  scikit-learn RandomForestClassifier   │
                │  stratified 80/20 split, balanced cw   │
                └──────────────────┬─────────────────────┘
                                   ▼
                ┌────────────────────────────────────────┐
                │       pipeline/evaluate.py             │
                │  confusion matrix · ROC · per-class F1 │
                │  → reports/evaluation.md               │
                └──────────────────┬─────────────────────┘
                                   ▼
   ┌───────────────────┐    ┌─────────────────────┐    ┌────────────────────────┐
   │  pipeline/cli.py  │    │  app/streamlit_app  │    │   live/mic.py          │
   │  typer commands   │    │  5-page dashboard    │   │   producer/consumer    │
   └───────────────────┘    └─────────────────────┘    │   DetectionEvent stream│
                                                       └────────────────────────┘
                                                                ▲
                                                                │
                                                       USB Maono microphone
```

## Component map

| Module | Responsibility | Key types / functions |
|---|---|---|
| `config.py` | Centralised paths and hyper-parameters | `Settings` (pydantic) |
| `data/ingest.py` | Decode any media file to mono WAV | `media_to_wav`, `ingest_all` |
| `data/chunk.py` | Slice WAV into fixed windows | `slice_wav`, `chunk_all` |
| `data/dataset.py` | Build a feature matrix from chunks | `Dataset`, `build_dataset` |
| `data/youtube.py` | Pull a URL list with `yt-dlp` | `download_urls` |
| `features/audio.py` | MFCC + mel-spec + speed of sound | `mfcc_feature_vector`, `log_mel_spectrogram`, `speed_of_sound` |
| `models/baseline.py` | RF bundle with save/load + metadata | `ModelBundle`, `train_random_forest` |
| `pipeline/train.py` | Orchestrator (5-step pipeline) | `run_full_pipeline` |
| `pipeline/evaluate.py` | Reports and plots | `write_report` |
| `pipeline/cli.py` | `typer` CLI | `ingest`, `chunk`, `train`, `live`, `ui`, `devices` |
| `live/mic.py` | Sounddevice producer + ring buffer | `stream_detections`, `DetectionEvent`, `run_console_live` |
| `viz/plots.py` | Matplotlib figures for Streamlit | `waveform_figure`, `mel_spectrogram_figure`, `microphone_array_figure` |
| `app/streamlit_app.py` | UI router across 5 pages | `main` |

## Threading model (live inference)

```
sounddevice callback  ──put──▶  audio_q (queue.Queue)
                                    │
                                    │ get
                                    ▼
                       consumer in stream_detections()
                                    │
                                    │ append → ring buffer
                                    │ if >= window:
                                    │     extract MFCC
                                    │     predict_proba
                                    │     yield DetectionEvent
                                    ▼
       Streamlit page reads events from another queue, reruns every 0.5 s
```

The producer is the PortAudio callback thread, which **must not block**.
All heavy work (MFCC + sklearn predict) happens in the consumer thread
that drains the queue.

## Where each requirement maps in code

| Requirement | Module |
|---|---|
| Ingest user-supplied MP3/audio to WAV | `data/ingest.py` |
| Auto-create dataset from labelled folders | `data/dataset.py` |
| Train classifier | `models/baseline.py`, `pipeline/train.py` |
| Produce evaluation report | `pipeline/evaluate.py` |
| Live mic detection | `live/mic.py` |
| Dashboard | `app/streamlit_app.py` |
| 4-mic TDOA concept (documented + visualised) | `docs/tdoa_concept.md`, `app/streamlit_app.py` (concept page), `viz/plots.py` (`microphone_array_figure`) |
| Environmental correction sketch ($c(T, RH)$) | `features/audio.py` (`speed_of_sound`) and the concept page slider |

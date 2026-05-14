# `audio/` — input drop folder

The pipeline ingests **any audio file** placed inside the labeled
sub-folders. Each sub-folder name is used as the **class label**.

**Primary expected format: MP3** (the user's pre-extracted recordings).
Anything else `ffmpeg` can decode also works.

## Default layout

```
audio/
├── shahed/   # positive class: clips containing Shahed-136 flight noise
│   ├── shahed_01.mp3
│   ├── shahed_02.mp3
│   └── ...
└── noise/    # negative class: city traffic, wind, birds, music, silence, etc.
    ├── traffic_01.mp3
    ├── birds_01.mp3
    └── ...
```

You may add additional class folders later (e.g. `audio/dji/`,
`audio/helicopter/`); the trainer will pick them up automatically and
produce a multi-class model.

## Accepted input formats

`.mp3`, `.wav`, `.flac`, `.ogg`, `.opus`, `.m4a`, `.aac`, `.mp4`, `.mkv`,
`.webm`, `.mov` — anything `ffmpeg` can decode is supported, because the
pipeline uses [`imageio-ffmpeg`](https://pypi.org/project/imageio-ffmpeg/)
which ships its own `ffmpeg` binary (no system-wide install required).

## Workflow

### 1. Drop your MP3 files

Simply copy/move MP3 files into the relevant sub-folder. File names do
not matter — the parent folder name is the class label.

### 2. Train

```powershell
python -m drone_detector.pipeline.cli train
```

### 3. (Optional) Pull more recordings from YouTube

Create a plain-text URL list (one URL per line) and run:

```powershell
python -m drone_detector.data.youtube audio/shahed.urls.txt audio/shahed
```

`yt-dlp` will download the audio track of each video and place it here
as an MP3/WAV.

## Recommendations for a defensible dataset

| Aspect | Target |
|---|---|
| Number of source clips per class | ≥ 5 |
| Total audio per class | ≥ 5 minutes |
| Diversity in `noise/` | mix of indoor + outdoor, urban + rural, wind + calm |
| Recording quality | clean ambient capture preferred; avoid heavy commentary or music overlay |
| Per-source provenance | keep a `sources.csv` or `sources.md` listing the URL / origin of every file — defensible at the master's defence |

## What happens next

The pipeline will:
1. Decode each MP3/audio file into a mono 22050-Hz WAV → `data/raw/<label>/`.
2. Slice each WAV into overlapping fixed-length chunks → `data/chunks/<label>/`.
3. Extract MFCC features → `data/features/dataset.npz`.
4. Train a Random Forest classifier → `models/baseline.pkl`.
5. Write an evaluation report → `reports/evaluation.md`.

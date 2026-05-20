# Methodology

## Target acoustic signature — Shahed-136

The Shahed-136 is a fixed-wing pusher-propeller loitering munition powered
by a small two-cylinder four-stroke or two-stroke internal-combustion
engine of approximately 50 hp (Mado MD-550 / Limbach L 550 E and Chinese
clones). At cruise the engine runs at roughly 5000–7000 rpm, which produces

* a **tonal fundamental** at ~80–120 Hz (firing rate),
* a rich set of integer harmonics up to roughly 5 kHz,
* a slowly-varying *moped-like* timbre, very different from the broadband
  *whirr* of consumer quadcopters such as the DJI Phantom.

This distinguishing feature is what motivates an acoustic detector: even a
single cheap omnidirectional microphone can capture enough spectral
structure to discriminate Shahed-class flights from common urban / natural
background.

## Dataset construction

1. **Source media.** MP3 audio extracted from YouTube clips containing
   in-flight or impact recordings (or directly downloaded via `yt-dlp`;
   see `audio/README.md`). Class folders are `audio/shahed/` for positive
   samples and `audio/noise/` for negatives (city traffic, wind, birds,
   indoor ambience, etc.).
2. **Decoding.** Each file is decoded to mono 22050 Hz 16-bit PCM with
   `imageio-ffmpeg`. The 22050 Hz target is a compromise between
   compatibility with most pre-trained audio models and the fact that the
   spectral content of interest sits well below 11 kHz.
3. **Chunking.** Each WAV is sliced into 2-second windows with 50 %
   overlap. Windows below RMS = 1e-4 are dropped — they contribute nothing
   but noise to the training set.
4. **Labels.** Chunks inherit the label of their parent folder. The
   pipeline trivially extends to more classes (e.g. `dji/`, `helicopter/`).

## Feature representation

For every 2-second window we compute 40 MFCC coefficients with
`librosa.feature.mfcc(n_fft=2048, hop_length=512)` and aggregate them into
a 120-dimensional vector:

| Block | Length | Purpose |
|---|---|---|
| `mfcc.mean(axis=1)` | 40 | Average timbre of the window |
| `mfcc.std(axis=1)`  | 40 | Spectral variability (tonal vs broadband) |
| `librosa.feature.delta(mfcc).mean(axis=1)` | 40 | Average temporal change |

The intuition: a Shahed engine produces **steady tonal** content (high
mean, **low** std on the relevant MFCC bins, **near-zero** delta), while
wind, traffic and birds produce **variable broadband** content (different
mean, **high** std, non-zero delta).

A log-mel spectrogram (`features/audio.py:log_mel_spectrogram`) is also
exposed and is the natural input for a future CNN. It is currently used
only for visualisation.

## Classifier

`scikit-learn` `RandomForestClassifier` with:

* `n_estimators = 300`
* `max_depth = None`
* `class_weight = "balanced"` (compensates for any class imbalance in
  the YouTube collection)
* `random_state = 42`
* `n_jobs = -1`

Why Random Forest:

* **No GPU required.** Training a few hundred trees on ~120-D features for
  ~10⁴ samples takes seconds on a laptop CPU.
* **Robust to noisy labels.** YouTube clips have a non-trivial label-noise
  rate (commentary, background music, edited audio); ensemble averaging
  smooths this out.
* **Interpretable feature importance.** `clf.feature_importances_` can be
  mapped back to MFCC bin indices to *show* which spectral bands carry the
  Shahed signature.

## Evaluation protocol

* **Split.** Stratified 80 / 20 train / test on the chunk level, fixed
  `random_state = 42`. (Per-source leave-one-video-out is the correct
  protocol if a single source dominates the dataset — easy follow-up.)
* **Metrics.** Confusion matrix, per-class precision / recall / F1 from
  `classification_report`, and binary ROC + AUC with `shahed` as the
  positive class.
* **Reports.** Generated as `reports/evaluation.md` + `confusion_matrix.png`
  + `roc.png` on every training run.

## Limitations

See [limitations.md](limitations.md) for a full discussion. In short:

* YouTube audio is **lossy** and often contains overlaid commentary.
* A single mono microphone cannot estimate direction or range.
* The model has **not** been validated against a calibrated outdoor
  recording set; field SNR-vs-distance characterisation is future work.
* If the noise class is dominated by indoor recordings, the model may
  overfit to "indoor vs outdoor" rather than "shahed vs not shahed".

## Future-work pipeline

| Step | Why | Expected uplift |
|---|---|---|
| Replace RF with a small CNN on log-mel | Captures time-frequency patterns directly | +3–5 pp F1 typically |
| Fine-tune YAMNet (pre-trained AudioSet) | Transfer from 521-class baseline | Better generalisation to unseen environments |
| Add per-source leave-one-out CV | Removes within-video leakage bias | Honest, defensible numbers |
| Add real outdoor recordings + SNR sweep | Field validation | Defensible operational claims |
| Move to 4-mic synchronised array | Direction-of-arrival | New deliverable: bearing(t) |
| EKF / particle filter on bearing | Trajectory inference under motion model | Trajectory plot |

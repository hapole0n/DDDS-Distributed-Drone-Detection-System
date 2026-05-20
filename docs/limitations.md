# Limitations & honest threats to validity

> A short, deliberately negative-sounding chapter. The point is that any
> *positive* number reported in this project comes with the caveats below;
> ignoring them would make the work indefensible at a master's level.

## 1. Dataset realism

* **Source: YouTube.** All training audio is lossy AAC compressed (≈ 128
  kbps for older uploads, ≈ 256 kbps for newer). High-frequency content
  above ~16 kHz is removed entirely, and tonal components are quantised.
  This is *acceptable* because the Shahed band of interest sits well
  below 16 kHz, but it nonetheless biases the spectral statistics the
  classifier learns.
* **Label noise.** Many flight clips contain narrator commentary, music
  overlays, or sirens. Random Forest is moderately robust to this, but
  ROC/F1 numbers are upper bounds on field performance.
* **Class imbalance.** The "noise" class is hand-curated from a much
  wider distribution than the "shahed" class. A model that simply learns
  "indoor vs outdoor" can score deceptively well; per-source LOO-CV is
  the right next step.

## 2. Acoustic propagation

* **Wind ≥ 5 m/s** dominates the microphone signal and lowers usable
  SNR by tens of dB. The current pipeline does not implement
  wind-screen-aware filtering.
* **Atmospheric absorption** is strongly frequency-dependent above
  ~1 kHz, so detection range varies with the harmonic that survives at a
  given distance. The model uses MFCC means / stds, which average across
  these bins — robust, but blind to the dependence.
* **Doppler.** A Shahed at cruise speed (~180 km/h) produces a Doppler
  shift of ~0.15 (head-on) at typical detection ranges. The model is
  not pitch-augmented and may degrade for fast head-on passes.

## 3. Hardware realism

* **Mono USB microphone.** A single mic gives time-domain detection
  only. Direction, range, and trajectory are out of reach with this
  hardware.
* **Maono USB capsule** is a budget condenser, optimised for voice. Its
  low-frequency response below 80 Hz is gentle but **not flat**;
  Shahed fundamentals at ~80–120 Hz are recorded with mild attenuation
  that the YouTube training set partially compensates for, but not
  consistently.
* **No environmental sensor.** Air temperature and humidity are used
  only in the conceptual TDOA page, not in the live detector.

## 4. Algorithmic limitations

* **Random Forest on MFCC** is a 2018-era baseline. A small CNN on a
  log-mel-spectrogram is expected to outperform it by several
  percentage points on the same data; this comparison is future work.
* **MFCC mean / std / Δ-mean** discard the full temporal structure of
  the window. A model that "sees" the periodic blade-passing pattern
  would in principle be more robust at low SNR.
* **Window length 2 s** trades latency for evidence: shorter windows
  detect faster but produce noisier scores; longer windows are more
  certain but lag.

## 5. Evaluation pitfalls

* **Same source in train and test.** Even with stratified split, two
  chunks from the *same* video frequently end up in the train and test
  sets. The honest metric is per-source leave-one-out, not per-chunk
  stratified.
* **Threshold = 0.5 is not principled.** It should be chosen on a
  precision-recall curve against a desired false-alarm budget. This
  project ships a default; do not interpret it as tuned.

## 6. Operational disclaimer

This software does not detect *threats*. It detects an *acoustic
signature*. A passing motorcycle, a generator, or another small piston
aircraft will sometimes trigger it. Any deployment outside a research
context must build a multi-modal sensor fusion stack and a human-in-the-
loop decision process. **No target assignment, no weapon guidance, no
engagement coordinates** are produced by this code, and the safety
boundary is reproduced in the dashboard sidebar.

## 7. What an honest defence sounds like

Examples of phrasing that the project supports:

* "On a stratified YouTube-derived test split, the Random Forest reaches
  F1 = X.XX on the Shahed class. Per-source leave-one-out cross
  validation, real-recording validation, and field SNR-vs-range
  characterisation are the natural next steps."
* "Direction-of-arrival is **not** measured with the current hardware.
  The 4-microphone TDOA path is documented and visualised as a
  conceptual extension; a defensible implementation requires
  sample-synchronous capture and is sketched in `docs/tdoa_concept.md`."
* "Detection latency on a laptop CPU is ~XX ms per 2-second window with
  50 % overlap, which is suitable for an early-warning UI but is not a
  real-time guarantee."

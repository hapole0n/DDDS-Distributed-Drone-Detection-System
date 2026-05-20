# Defence talking points

A 10-minute outline of how to present this project to the master's
commission and which questions to anticipate.

## Suggested 10-minute structure

| Time | Section | Slide-equivalent |
|---|---|---|
| 0:00–0:45  | Motivation: Shahed-136, defensive early warning, why acoustic | Slide 1–2 |
| 0:45–2:00  | Architecture overview (the one-screen diagram) | Slide 3 |
| 2:00–4:00  | Methodology: ingest → chunk → MFCC → RF, why each choice | Slide 4–5 |
| 4:00–5:30  | Live demo: phone plays Shahed clip → Maono → dashboard | Live |
| 5:30–7:00  | Results: confusion matrix, ROC, classification report | Slide 6 |
| 7:00–8:30  | 4-mic TDOA concept page + speed-of-sound interactivity | Live + slide 7 |
| 8:30–9:30  | Limitations and what was deliberately out of scope | Slide 8 |
| 9:30–10:00 | Future work + close | Slide 9 |

## Talking-point cheatsheet

* **Why mono microphone is enough for the classification thesis.** A
  single mic is enough to demonstrate the *recognition* layer, which is
  the first and necessary step. Localisation is a separate sub-problem
  that requires synchronised multi-channel hardware; treating it
  honestly as a documented extension is stronger than faking it with
  software-delayed mono.
* **Why Random Forest over a CNN.** Robustness on a small, noisy
  YouTube dataset; interpretability via feature importance; zero GPU
  requirement. CNN on log-mel is the natural next step and is sketched
  in the methodology file.
* **Why Streamlit over React.** The course is *Programowanie w języku
  Python* — keeping the entire stack in Python is both pedagogically
  appropriate and operationally simpler.
* **Why YouTube for the dataset.** Practical access to real Shahed
  recordings without travelling to a conflict zone. Honestly disclosed
  as a limitation, with field validation listed as future work.
* **What I would do with one more month.** Per-source leave-one-out
  cross-validation, a small CNN baseline for comparison, and SNR vs
  distance characterisation against a calibrated outdoor recording set.

## Anticipated questions and crisp answers

> "Your 4-microphone TDOA is just a paper design — why is that a
> contribution?"

Because the algorithm + geometry + speed-of-sound correction +
angular-resolution analysis are written down end-to-end, with a working
interactive visualisation, and with an explicit hardware specification
(4 × I²S INMP441 on ESP32-S3, baseline 30 cm, ≥ 22 kHz capture). A
follow-up engineer can build this in a week. The reason it is not built
in this revision is that the project is scoped to the recognition
problem.

> "Random Forest on MFCC is not state-of-the-art."

Correct. The methodology document acknowledges this explicitly and
sketches the CNN-on-log-mel and YAMNet-fine-tune paths as future work.
The RF baseline provides interpretability — feature importance maps
back to MFCC bins — which is useful pedagogically even if it leaves
some accuracy on the table.

> "Your test data is from YouTube. How do you know the model
> generalises?"

The honest answer is *I do not*. This is called out in the limitations
document and in the README. The next step is to record an outdoor
calibration set with a known source at known distances and re-evaluate.
The current numbers should be read as an upper bound contingent on the
training distribution.

> "What does your detector do when the Shahed is not present?"

The probability output is a continuous score in [0, 1]. The detection
*threshold* of 0.5 is a default and can be tuned on a precision-recall
curve once a false-alarm budget is defined. A 5-minute control recording
of typical background can be used to set the threshold per environment.

> "Why is this not a weapon?"

Because the output is a classification probability, not a coordinate, a
target identifier, or a fire-control command. The dashboard sidebar
documents the safety boundary explicitly. The system is an
early-warning research tool: it tells a human "the acoustic signature
of a Shahed-class platform is currently present", and nothing more.

> "Why 22050 Hz and not 48000 Hz?"

22050 Hz is enough to fully resolve the Shahed harmonic stack up to
~10 kHz, and it matches the sample rate used by most pre-trained audio
models (YAMNet, VGGish), which preserves the option of transfer
learning later. 48 kHz would be required if we were targeting smaller
multirotors with higher blade-passing harmonics; this is documented.

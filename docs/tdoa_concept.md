# 4-microphone TDOA — paper design

This document explains the **conceptual extension** of the single-mic
detector into a 4-microphone array capable of estimating the
direction-of-arrival (DoA) of an incoming UAV. It is intentionally not
shipped as runnable code in this revision — the project does not have the
synchronised multi-channel hardware that the algorithm requires.

## Geometry

Four microphones are placed at the corners of a horizontal square with
baseline $b$:

$$
\mathbf{m}_1 = \left(-\tfrac{b}{2}, -\tfrac{b}{2}, 0\right),\quad
\mathbf{m}_2 = \left(+\tfrac{b}{2}, -\tfrac{b}{2}, 0\right),\quad
\mathbf{m}_3 = \left(+\tfrac{b}{2}, +\tfrac{b}{2}, 0\right),\quad
\mathbf{m}_4 = \left(-\tfrac{b}{2}, +\tfrac{b}{2}, 0\right).
$$

A square array gives 6 independent microphone pairs and is symmetric in
both planar axes, which simplifies least-squares DoA estimation.

## Speed of sound

The propagation speed depends on air temperature and humidity. The Cramer
lightweight approximation is used to keep the BME280-driven correction
trivial in firmware:

$$
c(T, RH) \approx 331.3 + 0.606\,T + 0.0124\,RH
$$

with $T$ in °C and $RH$ in %. At 20 °C and 50 % RH this gives ≈ 344 m/s.

## Time-difference-of-arrival via GCC-PHAT

For each pair $(i, j)$ we compute the generalised cross-correlation with
phase transform:

$$
R_{ij}(\tau) = \mathcal{F}^{-1}\!\!\left\{\frac{X_i(f)\,X_j^{*}(f)}{|X_i(f)\,X_j^{*}(f)|}\right\}(\tau)
$$

PHAT-whitening removes the magnitude information and keeps only the
phase, which is **robust to reverberation** and to the unknown spectral
shape of the source — both highly relevant for outdoor drone capture.
The TDOA estimate for the pair is

$$
\hat{\tau}_{ij} = \arg\max_\tau R_{ij}(\tau).
$$

Sub-sample refinement is done by parabolic interpolation on the three
samples around the peak.

## Hardware-side requirement: sample-synchronous capture

The maths above is only meaningful if all 4 channels share the **same
sample clock**. This rules out four independent USB microphones, which
drift relative to each other. Workable options:

1. **Single 4-channel ADC** (e.g. PCM1864, MAX9814 × 4 + ESP32 ADC mux is
   *not* good enough — needs a true 4-channel converter).
2. **4 × I²S MEMS microphones** (INMP441, ICS-43434) wired to one MCU
   (ESP32-S3 or RP2040 with PIO) that drives the shared MCLK / WCLK /
   BCLK lines.
3. **A ready-made USB microphone array** (ReSpeaker 4-Mic Array, MATRIX
   Voice, MiniDSP UMA-8). These are firmware-synchronised and present
   themselves to the host OS as a single multi-channel device.

For the present project the simulated 4-mic implementation in the
predecessor `ddds` repository was explicitly labelled as "lab-only" — it
inserted software delays into a mono signal and recovered them, which is
a tautology. A real array is required for any defensible field claim.

## Direction-of-arrival least-squares

Given a far-field plane-wave approximation, every TDOA satisfies

$$
\hat{\tau}_{ij}\,c \;=\; (\mathbf{m}_j - \mathbf{m}_i) \cdot \hat{\mathbf{u}}
$$

where $\hat{\mathbf{u}}$ is the unit vector pointing **away** from the
source. Stacking the 6 pair equations gives an over-determined linear
system $A\hat{\mathbf{u}} = \mathbf{d}$, solved by ordinary least squares
followed by renormalisation $\hat{\mathbf{u}} \leftarrow \hat{\mathbf{u}}/\|\hat{\mathbf{u}}\|$.

The azimuth and elevation follow from

$$
\varphi = \operatorname{atan2}(u_x, u_y), \qquad
\theta = \arcsin(u_z).
$$

## Angular resolution as a function of baseline

For a baseline $b$, the smallest meaningful TDOA difference is one
sample, $\Delta\tau = 1/f_s$. The angular resolution at broadside is

$$
\Delta\varphi \approx \frac{c}{b \cdot f_s}.
$$

| $b$ | $f_s$ | $\Delta\varphi$ |
|---|---|---|
| 5 cm  | 22050 Hz | ~17.9° |
| 10 cm | 22050 Hz | ~8.9° |
| 30 cm | 22050 Hz | ~3.0° |
| 30 cm | 48000 Hz | ~1.4° |

This is the headline reason to go to a 30 cm-class baseline (and ideally
to 48 kHz capture) in the hardware extension.

## Trajectory inference from bearing-only measurements

A single static 4-mic array measures **bearing only** — it cannot
recover range. To produce a trajectory hypothesis we propose a
bearing-only Extended Kalman Filter:

* **State.** $\mathbf{x} = (p_x, p_y, v_x, v_y)^T$ assuming constant altitude.
* **Process model.** Constant-velocity motion with white-noise acceleration.
* **Measurement model.** $z_k = \operatorname{atan2}(p_x, p_y) + n_k$
  with measurement noise variance derived from GCC-PHAT peak sharpness.
* **Initialisation.** A wide prior over range (e.g. 50 m – 1 km) with
  uniform azimuth.

This is a textbook bearing-only tracking problem and produces a
*plausibility cone* rather than a unique trajectory until a second
array, a known speed, or a second sensor (e.g. optical) breaks the
range ambiguity.

## Why this is *not* in scope of the current project

* No 4-microphone hardware is available for the defence.
* A simulated array is mathematically meaningless (see introduction).
* The detection / classification task — which **is** solvable with a
  single mic — is enough to demonstrate the methodology and core DSP
  competency expected from the *Programowanie w języku Python* course
  at the master level.

The corresponding visualisation page in the dashboard (`4-mic TDOA
concept`) lets the reader explore the relationship between baseline,
speed of sound and angular resolution interactively, without making any
claim that the array has been physically built.

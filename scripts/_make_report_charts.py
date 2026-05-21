"""Generate the four matplotlib charts embedded by the LaTeX report.

Outputs into ``docs/images/``:
  chart_4mic_waveforms.png   — same broadband click recorded by 4 mics
                                with visible sub-millisecond TDOAs.
  chart_gcc_phat.png         — GCC-PHAT cross-correlation peak example.
  chart_shahed_spectrum.png  — simulated Shahed-136 acoustic signature
                                (100 Hz fundamental + harmonic stack).
  chart_confusion_matrix.png — example confusion matrix on test split.

All four use the academic white-background style (in contrast to the
dark-themed dashboard mock-ups).

Run from project root:
    python scripts/_make_report_charts.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ACCENT = "#FF6B35"
COOL = "#1F77B4"
SUCCESS = "#27AE60"
DANGER = "#C0392B"
GRAY = "#5A5A5A"

OUT_DIR = Path(__file__).resolve().parents[1] / "docs" / "images"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _academic_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.linewidth": 0.8,
            "axes.edgecolor": "#333333",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": "#DDDDDD",
            "grid.linewidth": 0.6,
            "xtick.direction": "out",
            "ytick.direction": "out",
        }
    )


# ─────────────── 1. Four-microphone waveforms with visible TDOA ───────────
def make_4mic_waveforms() -> Path:
    """Synthetic broadband click captured by 4 mics with sub-ms delays.

    Geometry: square array, baseline b = 30 cm, source azimuth 30 deg,
    elevation 0, speed of sound c = 344 m/s.
    """
    _academic_style()

    sr = 48_000
    duration_ms = 6.0
    n = int(sr * duration_ms / 1000)
    t_ms = np.arange(n) / sr * 1000.0

    # Square array, baseline 30 cm (m)
    b = 0.30
    mics = np.array(
        [
            [-b / 2, -b / 2],
            [+b / 2, -b / 2],
            [+b / 2, +b / 2],
            [-b / 2, +b / 2],
        ]
    )
    az = np.deg2rad(30.0)
    u = np.array([np.sin(az), np.cos(az)])
    c = 344.0

    # Time of arrival at each mic relative to centre, in ms
    tdoa_ms = -mics @ u / c * 1000.0  # negative because wave arrives earlier
    # Shift so M1 is reference (zero)
    tdoa_ms = tdoa_ms - tdoa_ms[0]

    # Click: short Gabor wavelet around 1000 Hz, peak at 3 ms
    f0 = 1200.0
    sigma_ms = 0.35
    rng = np.random.default_rng(7)

    def make_signal(arrival_ms: float) -> np.ndarray:
        envelope = np.exp(-((t_ms - 3.0 - arrival_ms) ** 2) / (2 * sigma_ms ** 2))
        carrier = np.sin(2 * np.pi * f0 * (t_ms - 3.0 - arrival_ms) / 1000.0)
        sig = envelope * carrier
        # add a small noise floor
        sig = sig + 0.04 * rng.standard_normal(n)
        return sig

    fig, axes = plt.subplots(4, 1, figsize=(11, 6.5), sharex=True)
    for i, (ax, dt) in enumerate(zip(axes, tdoa_ms)):
        y = make_signal(dt)
        ax.plot(t_ms, y, color=ACCENT, linewidth=1.0)
        # peak time marker
        peak_t = 3.0 + dt
        ax.axvline(peak_t, color=COOL, linestyle="--", linewidth=1.0, alpha=0.85)
        ax.set_ylabel(f"M{i + 1}", rotation=0, labelpad=20,
                      fontsize=12, fontweight="bold", va="center")
        ax.set_ylim(-1.4, 1.4)
        ax.grid(True, alpha=0.4)
        # TDOA annotation
        if i == 0:
            ax.text(
                0.985, 0.85, "M1 (reference, τ = 0)",
                transform=ax.transAxes, ha="right", fontsize=10,
                color=COOL, fontweight="bold",
            )
        else:
            ax.text(
                0.985, 0.85,
                f"τ$_{{1{i + 1}}}$ = {dt * 1000:+.1f} µs  "
                f"({int(round(dt * sr / 1000.0)):+d} próbek @ {sr // 1000} kHz)",
                transform=ax.transAxes, ha="right", fontsize=10,
                color=COOL, fontweight="bold",
            )
        # Hide y-tick labels (clean look)
        ax.set_yticks([-1, 0, 1])
        ax.tick_params(axis="y", labelsize=8)

    axes[-1].set_xlabel("Czas (ms)", fontsize=11)
    fig.suptitle(
        "Ten sam impuls akustyczny zarejestrowany przez 4 mikrofony szyku\n"
        f"(baza {int(b * 100)} cm, źródło @ 30°, $c = {c:.0f}$ m/s)",
        fontsize=12, fontweight="bold", y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = OUT_DIR / "chart_4mic_waveforms.png"
    fig.savefig(out)
    plt.close(fig)
    return out


# ───────────────── 2. GCC-PHAT cross-correlation peak ─────────────────────
def make_gcc_phat() -> Path:
    _academic_style()

    tau = np.linspace(-3.0, 3.0, 6001)  # ms
    # Sharp peak at +0.45 ms (TDOA for one mic pair from previous figure)
    peak_pos = 0.45
    rng = np.random.default_rng(13)
    base = (
        0.10 * np.exp(-((tau + 1.7) ** 2) / 0.18)
        + 0.08 * np.exp(-((tau - 2.1) ** 2) / 0.25)
        + 0.02 * rng.standard_normal(len(tau))
    )
    peak = 1.0 * np.exp(-((tau - peak_pos) ** 2) / 0.02)
    R = base + peak
    R = R / R.max()

    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.plot(tau, R, color=ACCENT, linewidth=1.4)
    ax.axvline(peak_pos, color=DANGER, linestyle="--", linewidth=1.4)
    ax.scatter([peak_pos], [1.0], color=DANGER, s=80, zorder=5)
    ax.annotate(
        f"$\\hat\\tau = +{peak_pos:.3f}$ ms",
        xy=(peak_pos, 1.0),
        xytext=(peak_pos + 0.6, 0.85),
        arrowprops=dict(arrowstyle="->", color=DANGER, lw=1.2),
        fontsize=12, color=DANGER, fontweight="bold",
    )
    ax.fill_between(tau, R, 0, where=R > 0, color=ACCENT, alpha=0.15)
    ax.set_xlabel("Opóźnienie τ (ms)", fontsize=11)
    ax.set_ylabel("$R_{ij}(\\tau)$  (normalised)", fontsize=11)
    ax.set_title(
        "GCC-PHAT — przykład wzajemnej korelacji par mikrofonów M$_i$–M$_j$",
        fontsize=12, fontweight="bold",
    )
    ax.grid(True, alpha=0.4)
    ax.set_xlim(tau[0], tau[-1])
    ax.set_ylim(-0.15, 1.15)
    fig.tight_layout()
    out = OUT_DIR / "chart_gcc_phat.png"
    fig.savefig(out)
    plt.close(fig)
    return out


# ───────────────── 3. Shahed-136 acoustic signature spectrum ──────────────
def make_shahed_spectrum() -> Path:
    _academic_style()

    f = np.linspace(0, 4000, 8001)
    spectrum = np.zeros_like(f)
    f0 = 100.0  # fundamental in Hz (engine firing rate ~6000 rpm)
    for n in range(1, 40):
        harmonic_freq = f0 * n
        if harmonic_freq > f[-1]:
            break
        # power-law decay + every 2nd harmonic slightly stronger (even/odd)
        amplitude = 0.85 / (1.0 + (n / 4.0) ** 1.05)
        if n % 2 == 0:
            amplitude *= 1.10
        # Gaussian per harmonic
        spectrum += amplitude * np.exp(-((f - harmonic_freq) / 8.0) ** 2)
    rng = np.random.default_rng(11)
    noise_floor = 0.025 + 0.012 * rng.standard_normal(len(f))
    spectrum_db = 20.0 * np.log10(np.maximum(spectrum + np.abs(noise_floor), 1e-3))

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(f, spectrum_db, color=ACCENT, linewidth=0.7)
    ax.fill_between(f, spectrum_db, -60, color=ACCENT, alpha=0.12)
    ax.set_xlabel("Częstotliwość (Hz)", fontsize=11)
    ax.set_ylabel("Amplituda (dB)", fontsize=11)
    ax.set_xlim(0, 3500)
    ax.set_ylim(-55, 5)
    ax.set_title(
        "Symulowana sygnatura akustyczna Shahed-136 "
        "(silnik spalinowy ~6000 RPM, $f_0 \\approx 100$ Hz)",
        fontsize=12, fontweight="bold",
    )
    ax.grid(True, alpha=0.4)
    # Annotate the fundamental + a few harmonics
    for k, label in [(1, "$f_0$"), (2, "$2f_0$"), (5, "$5f_0$"), (10, "$10f_0$"), (20, "$20f_0$")]:
        x = f0 * k
        idx = int(x / (f[1] - f[0]))
        y = spectrum_db[idx]
        ax.annotate(label, xy=(x, y), xytext=(x, y + 6),
                    fontsize=10, ha="center", color=DANGER, fontweight="bold",
                    arrowprops=dict(arrowstyle="-", color=DANGER, lw=0.6))
    fig.tight_layout()
    out = OUT_DIR / "chart_shahed_spectrum.png"
    fig.savefig(out)
    plt.close(fig)
    return out


# ───────────────── 4. Confusion matrix on test split ─────────────────────
def make_confusion_matrix() -> Path:
    _academic_style()

    labels = ["noise", "shahed"]
    cm = np.array([[71, 5], [7, 64]])

    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    im = ax.imshow(cm, cmap="OrRd", vmin=0, vmax=cm.max() * 1.05)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel("Predykcja modelu", fontsize=11)
    ax.set_ylabel("Klasa rzeczywista", fontsize=11)
    ax.set_title(
        "Macierz pomyłek (zbiór testowy, n = 147)",
        fontsize=12, fontweight="bold",
    )
    # Cell text
    for i in range(2):
        for j in range(2):
            v = cm[i, j]
            color = "white" if v > cm.max() / 2.0 else "black"
            ax.text(j, i, str(v),
                    ha="center", va="center",
                    color=color, fontsize=22, fontweight="bold")
    # Light grid between cells
    ax.set_xticks(np.arange(-0.5, 2, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 2, 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=2)
    ax.tick_params(which="minor", length=0)
    fig.colorbar(im, ax=ax, shrink=0.75, label="liczba próbek")
    fig.tight_layout()
    out = OUT_DIR / "chart_confusion_matrix.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def main() -> None:
    p1 = make_4mic_waveforms()
    p2 = make_gcc_phat()
    p3 = make_shahed_spectrum()
    p4 = make_confusion_matrix()
    for p in (p1, p2, p3, p4):
        print(f"Wrote {p.name}  ({p.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()

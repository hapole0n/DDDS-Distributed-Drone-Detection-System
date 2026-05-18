"""Plotting helpers for the Streamlit UI.

Two backends live side by side:

* `matplotlib` — used by `pipeline/evaluate.py` for static PNG reports
  that are embedded in `reports/evaluation.md` and checked into git.
* `plotly` — used by the Streamlit UI for interactive, animated, fluid
  visualisations. Plotly gives native gauges, smooth heatmap updates,
  responsive layout, and a dark theme that matches `.streamlit/config.toml`.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
from plotly.subplots import make_subplots  # noqa: E402

from drone_detector.features.audio import (  # noqa: E402
    log_mel_spectrogram,
    speed_of_sound,
)


# ───────────────────────── Theme constants ─────────────────────────

ACCENT = "#FF6B35"      # orange / shahed
ACCENT_2 = "#FFB23F"
COOL = "#5DADE2"        # blue / neutral
SUCCESS = "#27AE60"
DANGER = "#E74C3C"
WARNING = "#F39C12"
TEXT = "#FAFAFA"
PAPER = "rgba(0,0,0,0)"
GRID = "rgba(255,255,255,0.08)"


def _layout(**kw) -> dict:
    base = dict(
        paper_bgcolor=PAPER,
        plot_bgcolor=PAPER,
        font=dict(color=TEXT, family="Inter, system-ui, sans-serif"),
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis=dict(gridcolor=GRID, zerolinecolor=GRID),
        yaxis=dict(gridcolor=GRID, zerolinecolor=GRID),
    )
    base.update(kw)
    return base


# ═════════════════════════ MATPLOTLIB (static reports) ══════════════════════

def waveform_figure(y: np.ndarray, sr: int):
    fig, ax = plt.subplots(figsize=(7, 1.8))
    t = np.arange(len(y)) / sr
    ax.plot(t, y, color=ACCENT, lw=0.6)
    ax.set_xlim(t[0], t[-1])
    ax.set_xlabel("time (s)")
    ax.set_yticks([])
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def mel_spectrogram_figure(y: np.ndarray, sr: int):
    S = log_mel_spectrogram(y, sr)
    fig, ax = plt.subplots(figsize=(7, 3))
    img = ax.imshow(S, aspect="auto", origin="lower", cmap="magma")
    ax.set_xlabel("frame")
    ax.set_ylabel("mel bin")
    fig.colorbar(img, ax=ax, format="%+2.0f dB", pad=0.01)
    fig.tight_layout()
    return fig


def probability_history_figure(times: np.ndarray, probs: np.ndarray, threshold: float):
    fig, ax = plt.subplots(figsize=(7, 2.4))
    ax.plot(times, probs, color=ACCENT, lw=1.8)
    ax.axhline(threshold, color="white", lw=0.8, ls="--", alpha=0.5)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("P(shahed)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def microphone_array_figure(baseline_m: float = 0.30):
    """Matplotlib version, kept for compatibility with older code paths."""
    fig, ax = plt.subplots(figsize=(5, 5))
    half = baseline_m / 2
    mics = np.array([[-half, -half], [+half, -half], [+half, +half], [-half, +half]])
    ax.scatter(mics[:, 0], mics[:, 1], s=180, c=ACCENT, edgecolors="white", zorder=3)
    for i, (x, y) in enumerate(mics):
        ax.annotate(f"M{i + 1}", (x, y), xytext=(8, 8), textcoords="offset points",
                    fontsize=11, color="white")
    bearing = np.deg2rad(30.0)
    ax.annotate("", xy=(2 * np.sin(bearing), 2 * np.cos(bearing)), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", lw=2, color=COOL))
    ax.set_xlim(-1.5, 2.6)
    ax.set_ylim(-1.5, 2.6)
    ax.set_aspect("equal")
    ax.set_title(f"4-mic square array (baseline = {baseline_m * 100:.0f} cm)")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig


# ═════════════════════════ PLOTLY (interactive UI) ══════════════════════════

# --- Waveform ---------------------------------------------------------------

def plotly_waveform(y: np.ndarray, sr: int, height: int = 160) -> go.Figure:
    t = np.arange(len(y)) / sr
    fig = go.Figure(go.Scatter(x=t, y=y, mode="lines",
                               line=dict(color=ACCENT, width=1)))
    fig.update_layout(
        **_layout(
            height=height,
            xaxis_title="time (s)",
            yaxis_title=None,
            yaxis=dict(gridcolor=GRID, zerolinecolor=GRID,
                       range=[-max(1e-3, np.abs(y).max() * 1.1),
                              +max(1e-3, np.abs(y).max() * 1.1)]),
            showlegend=False,
        )
    )
    return fig


def plotly_mel_spectrogram(y: np.ndarray, sr: int, height: int = 280) -> go.Figure:
    S = log_mel_spectrogram(y, sr)
    fig = go.Figure(go.Heatmap(
        z=S,
        colorscale="Magma",
        colorbar=dict(title="dB", thickness=10),
        zsmooth="best",
    ))
    fig.update_layout(
        **_layout(
            height=height,
            xaxis_title="frame",
            yaxis_title="mel bin",
        )
    )
    return fig


# --- Live gauge -------------------------------------------------------------

def plotly_probability_gauge(prob: float, threshold: float = 0.5,
                             title: str = "P(shahed)") -> go.Figure:
    colour = DANGER if prob >= threshold else SUCCESS if prob < 0.3 else WARNING
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=prob * 100.0,
        number={"suffix": " %", "font": {"size": 36, "color": colour}},
        delta={"reference": threshold * 100.0,
               "increasing": {"color": DANGER}, "decreasing": {"color": SUCCESS}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": TEXT},
            "bar": {"color": colour, "thickness": 0.30},
            "bgcolor": "rgba(255,255,255,0.05)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 30], "color": "rgba(39,174,96,0.25)"},
                {"range": [30, 50], "color": "rgba(243,156,18,0.25)"},
                {"range": [50, 100], "color": "rgba(231,76,60,0.30)"},
            ],
            "threshold": {
                "line": {"color": TEXT, "width": 3},
                "thickness": 0.85,
                "value": threshold * 100.0,
            },
        },
        title={"text": f"<b>{title}</b>", "font": {"size": 16, "color": TEXT}},
    ))
    fig.update_layout(
        paper_bgcolor=PAPER,
        font={"color": TEXT},
        height=260,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


# --- Class probability bars -------------------------------------------------

def plotly_class_probabilities(labels: list[str], probs: np.ndarray) -> go.Figure:
    colours = [DANGER if l.lower() == "shahed" else COOL for l in labels]
    fig = go.Figure(go.Bar(
        x=probs * 100, y=labels, orientation="h",
        marker=dict(color=colours, line=dict(color="white", width=0.5)),
        text=[f"{p * 100:0.1f} %" for p in probs],
        textposition="outside",
    ))
    fig.update_layout(
        **_layout(
            height=max(140, 60 * len(labels)),
            xaxis=dict(range=[0, 110], gridcolor=GRID, title="probability (%)"),
            yaxis=dict(gridcolor="rgba(0,0,0,0)", autorange="reversed"),
            showlegend=False,
        )
    )
    return fig


# --- VU meter ---------------------------------------------------------------

def plotly_vu_meter(rms_db: float) -> go.Figure:
    """Bullet chart showing instantaneous RMS in dBFS."""
    db = float(np.clip(rms_db, -60, 0))
    fig = go.Figure(go.Indicator(
        mode="number+gauge",
        value=db,
        number={"suffix": " dB", "font": {"size": 22, "color": TEXT}},
        gauge={
            "shape": "bullet",
            "axis": {"range": [-60, 0]},
            "bar": {"color": ACCENT, "thickness": 0.6},
            "bgcolor": "rgba(255,255,255,0.05)",
            "borderwidth": 0,
            "steps": [
                {"range": [-60, -30], "color": "rgba(39,174,96,0.4)"},
                {"range": [-30, -10], "color": "rgba(243,156,18,0.4)"},
                {"range": [-10, 0], "color": "rgba(231,76,60,0.5)"},
            ],
            "threshold": {
                "line": {"color": "white", "width": 2},
                "thickness": 0.8, "value": -6,
            },
        },
        title={"text": "<b>input level</b>", "font": {"size": 14}},
    ))
    fig.update_layout(paper_bgcolor=PAPER, font={"color": TEXT},
                      height=110, margin=dict(l=10, r=10, t=30, b=10))
    return fig


# --- Probability history ----------------------------------------------------

def plotly_probability_history(
    times: np.ndarray, probs: np.ndarray, threshold: float = 0.5
) -> go.Figure:
    fig = go.Figure()
    above = probs >= threshold
    fig.add_trace(go.Scatter(
        x=times, y=probs, mode="lines", line=dict(color=ACCENT, width=2.5),
        fill="tozeroy", fillcolor="rgba(255,107,53,0.25)", name="P(shahed)",
    ))
    if above.any():
        fig.add_trace(go.Scatter(
            x=times[above], y=probs[above], mode="markers",
            marker=dict(color=DANGER, size=9, symbol="circle",
                        line=dict(color="white", width=1)),
            name="above threshold", hoverinfo="x+y",
        ))
    fig.add_hline(y=threshold, line=dict(color="white", dash="dash", width=1),
                  annotation_text=f"threshold = {threshold:.2f}",
                  annotation_position="top left",
                  annotation_font_color="white")
    fig.update_layout(
        **_layout(
            height=240,
            yaxis=dict(range=[0, 1.02], gridcolor=GRID, title="P(shahed)"),
            xaxis=dict(gridcolor=GRID, title="time (s)"),
            legend=dict(orientation="h", y=1.15, x=0.0,
                        bgcolor="rgba(0,0,0,0)"),
        )
    )
    return fig


# --- Rolling spectrogram (live mic) -----------------------------------------

def plotly_rolling_spectrogram(windows: list[np.ndarray], sr: int,
                                height: int = 280) -> go.Figure:
    if not windows:
        return go.Figure().update_layout(
            **_layout(height=height,
                      annotations=[dict(text="waiting for audio…",
                                        xref="paper", yref="paper",
                                        x=0.5, y=0.5, showarrow=False,
                                        font=dict(color="rgba(255,255,255,0.5)"))])
        )
    full = np.concatenate(windows)
    return plotly_mel_spectrogram(full, sr, height=height)


# --- Confusion matrix -------------------------------------------------------

def plotly_confusion_matrix(cm: np.ndarray, labels: list[str]) -> go.Figure:
    z = cm.astype(float)
    text = [[str(int(v)) for v in row] for row in cm]
    fig = go.Figure(go.Heatmap(
        z=z, x=labels, y=labels,
        text=text, texttemplate="<b>%{text}</b>",
        textfont=dict(color="white", size=18),
        colorscale="Magma", showscale=False,
    ))
    fig.update_layout(
        **_layout(
            height=320,
            xaxis=dict(title="predicted", side="bottom"),
            yaxis=dict(title="actual", autorange="reversed"),
        )
    )
    return fig


# --- Class distribution -----------------------------------------------------

def plotly_class_distribution(counts: dict[str, int],
                              title: str = "Dataset class distribution") -> go.Figure:
    labels = list(counts)
    values = [counts[k] for k in labels]
    colours = [DANGER if l.lower() == "shahed" else COOL for l in labels]
    fig = go.Figure(go.Bar(
        x=labels, y=values, marker=dict(color=colours),
        text=values, textposition="outside",
    ))
    fig.update_layout(
        **_layout(
            height=260, title=title,
            yaxis=dict(title="files / chunks", gridcolor=GRID),
            xaxis=dict(gridcolor=GRID),
        )
    )
    return fig


# ─────────────────────── 4-mic TDOA visualisations ──────────────────────────

def _square_array(baseline_m: float) -> np.ndarray:
    h = baseline_m / 2
    return np.array([[-h, -h], [+h, -h], [+h, +h], [-h, +h]])


def plotly_tdoa_geometry(
    baseline_m: float, azimuth_deg: float, elevation_deg: float = 0.0,
    sample_rate: int = 22050, c: float = 343.0,
) -> go.Figure:
    mics = _square_array(baseline_m)
    az = np.deg2rad(azimuth_deg)
    el = np.deg2rad(elevation_deg)
    # Source direction unit vector in horizontal plane (project to 2D)
    u = np.array([np.sin(az) * np.cos(el), np.cos(az) * np.cos(el)])

    # Wavefront line — perpendicular to u, passing through a point on the
    # source ray for visual reference
    perp = np.array([-u[1], u[0]])
    p_source = u * 2.0
    wf_a = p_source - perp * 1.5
    wf_b = p_source + perp * 1.5

    fig = go.Figure()

    # Wavefront (animated by user moving slider)
    fig.add_trace(go.Scatter(
        x=[wf_a[0], wf_b[0]], y=[wf_a[1], wf_b[1]],
        mode="lines", line=dict(color=COOL, width=2, dash="dot"),
        name="incoming wavefront",
    ))

    # Source arrow (incoming direction = -u)
    fig.add_annotation(
        ax=p_source[0], ay=p_source[1], x=0, y=0,
        xref="x", yref="y", axref="x", ayref="y",
        showarrow=True, arrowhead=3, arrowsize=1.4, arrowwidth=2.5,
        arrowcolor=COOL,
    )
    fig.add_trace(go.Scatter(
        x=[p_source[0]], y=[p_source[1]], mode="markers+text",
        marker=dict(symbol="diamond", color=COOL, size=18,
                    line=dict(color="white", width=1.5)),
        text=["source"], textposition="top center", textfont=dict(color=COOL),
        name="source",
    ))

    # Microphones
    fig.add_trace(go.Scatter(
        x=mics[:, 0], y=mics[:, 1], mode="markers+text",
        marker=dict(color=ACCENT, size=22, symbol="circle",
                    line=dict(color="white", width=2)),
        text=[f"M{i + 1}" for i in range(4)],
        textposition="bottom center",
        textfont=dict(color=TEXT, size=12, family="monospace"),
        name="microphones",
    ))

    # TDOA per pair as annotations near mics
    pairs = [(i, j) for i in range(4) for j in range(i + 1, 4)]
    for i, j in pairs:
        d = mics[j] - mics[i]
        tdoa_s = float(np.dot(d, u)) / c
        mid = (mics[i] + mics[j]) / 2.0
        fig.add_annotation(
            x=mid[0], y=mid[1], text=f"τ{i + 1}{j + 1} = {tdoa_s * 1000:+.3f} ms",
            showarrow=False,
            font=dict(color="rgba(255,255,255,0.55)", size=9, family="monospace"),
            bgcolor="rgba(0,0,0,0)",
        )

    extent = max(2.6, baseline_m * 1.5)
    fig.update_layout(
        **_layout(
            height=480,
            xaxis=dict(range=[-extent, extent], gridcolor=GRID,
                       title="x (m)", zeroline=True, zerolinecolor=GRID),
            yaxis=dict(range=[-extent, extent], gridcolor=GRID,
                       title="y (m)", scaleanchor="x", scaleratio=1,
                       zeroline=True, zerolinecolor=GRID),
            showlegend=False,
            title=(f"4-mic square array — baseline {baseline_m * 100:.0f} cm · "
                   f"source @ {azimuth_deg:.0f}° · c = {c:.1f} m/s"),
        )
    )
    return fig


def plotly_tdoa_table(baseline_m: float, azimuth_deg: float, c: float) -> go.Figure:
    """Companion table of TDOAs in ms + samples."""
    mics = _square_array(baseline_m)
    az = np.deg2rad(azimuth_deg)
    u = np.array([np.sin(az), np.cos(az)])
    pairs = [(i, j) for i in range(4) for j in range(i + 1, 4)]
    rows = []
    for i, j in pairs:
        d = mics[j] - mics[i]
        tau = float(np.dot(d, u)) / c
        rows.append([f"M{i + 1}–M{j + 1}", f"{tau * 1000:+.3f}",
                     f"{tau * 22050:+.2f}", f"{tau * 48000:+.2f}"])
    header = ["pair", "τ (ms)", "samples @ 22 050 Hz", "samples @ 48 000 Hz"]
    fig = go.Figure(go.Table(
        header=dict(values=[f"<b>{h}</b>" for h in header],
                    fill_color="rgba(255,107,53,0.25)",
                    line_color=GRID, font=dict(color=TEXT, size=12)),
        cells=dict(values=list(zip(*rows)),
                   fill_color="rgba(255,255,255,0.03)",
                   line_color=GRID,
                   font=dict(color=TEXT, family="monospace", size=12),
                   align=["left", "right", "right", "right"], height=26),
    ))
    fig.update_layout(paper_bgcolor=PAPER, font={"color": TEXT},
                      height=240, margin=dict(l=0, r=0, t=10, b=10))
    return fig


def plotly_resolution_curve(sample_rate: int, c: float) -> go.Figure:
    baselines_cm = np.array([5, 10, 20, 30, 50, 80, 100])
    res_deg = np.rad2deg(c / (baselines_cm / 100.0 * sample_rate))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=baselines_cm, y=res_deg,
        mode="lines+markers",
        line=dict(color=ACCENT, width=3, shape="spline"),
        marker=dict(size=10, color=ACCENT, line=dict(color="white", width=1)),
        fill="tozeroy", fillcolor="rgba(255,107,53,0.15)",
        name="Δφ",
    ))
    fig.update_layout(
        **_layout(
            height=280,
            title=f"Angular resolution at broadside  (fs = {sample_rate} Hz, c = {c:.0f} m/s)",
            xaxis=dict(title="baseline (cm)", gridcolor=GRID),
            yaxis=dict(title="Δφ (degrees)", gridcolor=GRID),
            showlegend=False,
        )
    )
    return fig


def plotly_speed_of_sound_curve(rh_pct: float = 50.0) -> go.Figure:
    temps = np.linspace(-10, 40, 51)
    cs = np.array([speed_of_sound(t, rh_pct) for t in temps])
    fig = go.Figure(go.Scatter(
        x=temps, y=cs, mode="lines",
        line=dict(color=COOL, width=3, shape="spline"),
        fill="tozeroy", fillcolor="rgba(93,173,226,0.20)",
        name="c(T)",
    ))
    fig.update_layout(
        **_layout(
            height=240,
            title=f"Speed of sound vs temperature (RH = {rh_pct:.0f} %)",
            xaxis=dict(title="temperature (°C)", gridcolor=GRID),
            yaxis=dict(title="c (m/s)", gridcolor=GRID),
            showlegend=False,
        )
    )
    return fig


def plotly_wavelength_bar(frequencies_hz: list[float], c: float) -> go.Figure:
    wavelengths_m = [c / f for f in frequencies_hz]
    fig = go.Figure(go.Bar(
        x=[f"{int(f)} Hz" for f in frequencies_hz], y=wavelengths_m,
        marker=dict(color=ACCENT_2), text=[f"{w:.2f} m" for w in wavelengths_m],
        textposition="outside",
    ))
    fig.update_layout(
        **_layout(
            height=240,
            title=f"Wavelength λ = c / f  (c = {c:.1f} m/s)",
            xaxis=dict(title="frequency", gridcolor=GRID),
            yaxis=dict(title="λ (m)", gridcolor=GRID),
            showlegend=False,
        )
    )
    return fig


# ─────────────────────── Pipeline / architecture viz ────────────────────────

def plotly_pipeline_diagram() -> go.Figure:
    """Stylised horizontal flow of the 5 pipeline stages."""
    stages = [
        ("1. Ingest", "MP3 → mono WAV", "#5DADE2"),
        ("2. Chunk", "2 s windows, 50% overlap", "#48C9B0"),
        ("3. Features", "40 MFCC × (μ, σ, Δμ)", "#F4D03F"),
        ("4. Train", "Random Forest 300×", "#E67E22"),
        ("5. Evaluate", "F1 · ROC · CM", "#E74C3C"),
    ]
    n = len(stages)
    fig = go.Figure()
    for i, (title, sub, color) in enumerate(stages):
        # rounded rectangle approximation via shape
        x0, x1 = i, i + 0.85
        fig.add_shape(type="rect", x0=x0, x1=x1, y0=0.2, y1=0.8,
                      line=dict(color=color, width=2),
                      fillcolor=f"rgba(255,255,255,0.03)")
        fig.add_annotation(x=(x0 + x1) / 2, y=0.62, text=f"<b>{title}</b>",
                           showarrow=False,
                           font=dict(color=TEXT, size=14))
        fig.add_annotation(x=(x0 + x1) / 2, y=0.38, text=sub,
                           showarrow=False,
                           font=dict(color="rgba(255,255,255,0.65)", size=11))
        if i < n - 1:
            fig.add_annotation(x=i + 0.92, y=0.5, ax=i + 1.0, ay=0.5,
                               xref="x", yref="y", axref="x", ayref="y",
                               showarrow=True, arrowhead=2, arrowsize=1.0,
                               arrowwidth=1.5, arrowcolor=TEXT)
    fig.update_layout(
        **_layout(
            height=140,
            xaxis=dict(visible=False, range=[-0.1, n + 0.05]),
            yaxis=dict(visible=False, range=[0, 1]),
            margin=dict(l=10, r=10, t=10, b=10),
        )
    )
    return fig

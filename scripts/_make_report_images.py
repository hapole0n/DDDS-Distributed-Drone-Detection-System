"""Generate the two illustration images embedded by the LaTeX report.

Outputs (overwritten on every run):
  docs/images/screen_live_mic.png       — Live microphone page mock-up
  docs/images/screen_tdoa_concept.png   — 4-mic TDOA concept page mock-up

The mock-ups deliberately match the colour palette and layout of the real
Streamlit dashboard (``app/streamlit_app.py``) so the report illustrations
look like authentic screenshots.

Run:
    python scripts/_make_report_images.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Wedge  # noqa: E402


# ───────────────────────── Theme palette ─────────────────────────
BG_DARK = "#0E1117"
BG_MID = "#1A1F2B"
SIDEBAR = "#11141B"
ACCENT = "#FF6B35"
ACCENT2 = "#FFB23F"
COOL = "#5DADE2"
SUCCESS = "#27AE60"
SUCCESS_DARK = "#1E8449"
DANGER = "#E74C3C"
WARNING = "#F39C12"
TEXT = "#FAFAFA"
TEXT_MUTED = (1.0, 1.0, 1.0, 0.65)
GRID = (1.0, 1.0, 1.0, 0.08)

OUT_DIR = Path(__file__).resolve().parents[1] / "docs" / "images"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _setup_dark_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": BG_DARK,
            "axes.facecolor": BG_DARK,
            "savefig.facecolor": BG_DARK,
            "axes.edgecolor": "#FFFFFF20",
            "axes.labelcolor": TEXT,
            "axes.titlecolor": TEXT,
            "text.color": TEXT,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
            "grid.color": "#FFFFFF15",
            "grid.linestyle": "-",
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


# ─────────────────────── Helper drawing primitives ───────────────────────
def draw_sidebar(fig, x0=0.0, w=0.16, top=1.0, bottom=0.0):
    """Draw the orange-titled left sidebar that the Streamlit app uses."""
    ax = fig.add_axes([x0, bottom, w, top - bottom], facecolor=SIDEBAR)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    # Brand gradient title
    ax.text(0.07, 0.92, "DDDS", color=ACCENT, fontsize=22, fontweight="bold")
    ax.text(
        0.07,
        0.87,
        "Distributed Drone\nDetection System",
        color=ACCENT2,
        fontsize=8.5,
        fontweight="bold",
        linespacing=1.1,
    )
    ax.text(
        0.07,
        0.81,
        "Politechnika Lubelska\nmaster's project",
        color=TEXT_MUTED,
        fontsize=7,
        linespacing=1.2,
    )
    # Nav items (geometric symbols that DejaVu Sans supports)
    nav = [
        ("◆  Overview", False),
        ("●  Train pipeline", False),
        ("◉  Analyse a file", False),
        ("▶  Live microphone", True),
        ("◇  4-mic TDOA concept", False),
    ]
    y = 0.72
    for label, active in nav:
        if active:
            ax.add_patch(
                mpatches.Circle(
                    (0.06, y + 0.005),
                    0.013,
                    facecolor=ACCENT,
                    edgecolor=ACCENT,
                    transform=ax.transAxes,
                )
            )
        else:
            ax.add_patch(
                mpatches.Circle(
                    (0.06, y + 0.005),
                    0.013,
                    facecolor="none",
                    edgecolor="#444a55",
                    linewidth=1.0,
                    transform=ax.transAxes,
                )
            )
        ax.text(0.13, y, label, color=TEXT if active else TEXT_MUTED, fontsize=8)
        y -= 0.045

    # Safety boundary
    ax.axhline(0.40, color="#FFFFFF15", linewidth=0.8)
    ax.text(
        0.07, 0.37,
        "Safety boundary.",
        color=TEXT, fontsize=8.0, fontweight="bold",
        va="top",
    )
    ax.text(
        0.07, 0.32,
        "Defensive early-warning\nresearch tool. No target\n"
        "assignment, no weapon\nguidance, no engagement\ncoordinates.",
        color=TEXT_MUTED, fontsize=7.0, linespacing=1.35,
        va="top",
    )


def draw_sidebar_with_tdoa_controls(fig, x0=0.0, w=0.16):
    draw_sidebar(fig, x0=x0, w=w, top=1.0, bottom=0.32)
    # TDOA controls panel
    ax = fig.add_axes([x0, 0.0, w, 0.32], facecolor=SIDEBAR)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.axhline(0.95, color="#FFFFFF15", linewidth=0.8)
    ax.text(0.07, 0.86, "TDOA controls", color=TEXT, fontsize=8, fontweight="bold")
    sliders = [
        ("Array baseline (cm)", 100, 0.72),
        ("Sample rate (Hz)", 22050, 0.50),
        ("Temperature (°C)", 20, 0.28),
        ("Humidity (%)", 50, 0.06),
    ]
    for label, value, y in sliders:
        ax.text(0.07, y + 0.08, label, color=TEXT_MUTED, fontsize=7)
        # slider bar
        ax.add_patch(
            mpatches.Rectangle(
                (0.07, y + 0.04),
                0.85,
                0.012,
                color="#FFFFFF18",
                transform=ax.transAxes,
            )
        )
        # filled section
        frac = 1.0 if "baseline" in label.lower() else 0.55
        ax.add_patch(
            mpatches.Rectangle(
                (0.07, y + 0.04),
                0.85 * frac,
                0.012,
                color=ACCENT,
                transform=ax.transAxes,
            )
        )
        # thumb
        ax.add_patch(
            mpatches.Circle(
                (0.07 + 0.85 * frac, y + 0.046),
                0.014,
                color=ACCENT,
                transform=ax.transAxes,
                zorder=5,
            )
        )
        ax.text(
            0.93,
            y + 0.07,
            f"{value:,}".replace(",", " "),
            color=ACCENT,
            fontsize=7.5,
            ha="right",
            fontweight="bold",
        )


def draw_kpi_card(
    fig,
    rect,
    label: str,
    value: str,
    *,
    color=TEXT,
    sub: str | None = None,
):
    """Draw one of the rounded KPI cards used across the dashboard."""
    ax = fig.add_axes(rect, facecolor=BG_DARK)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    box = FancyBboxPatch(
        (0.02, 0.05),
        0.96,
        0.90,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        linewidth=0.8,
        facecolor="#FFFFFF06",
        edgecolor="#FFFFFF18",
        transform=ax.transAxes,
    )
    ax.add_patch(box)
    ax.text(
        0.07,
        0.74,
        label.upper(),
        color=TEXT_MUTED,
        fontsize=8,
        fontweight="bold",
    )
    ax.text(0.07, 0.34, value, color=color, fontsize=22, fontweight="bold")
    if sub:
        ax.text(0.07, 0.12, sub, color=TEXT_MUTED, fontsize=7)


# ──────────────────── 1.  LIVE MICROPHONE MOCK-UP ────────────────────
def make_live_mic_image() -> Path:
    _setup_dark_style()
    fig = plt.figure(figsize=(18, 9), dpi=120, facecolor=BG_DARK)

    draw_sidebar(fig)

    # Main column body
    body_left = 0.18
    body_right = 0.985
    body_w = body_right - body_left

    # --- Input device selector mock ---
    ax_dev = fig.add_axes([body_left, 0.90, body_w, 0.06], facecolor=BG_DARK)
    ax_dev.set_xticks([])
    ax_dev.set_yticks([])
    for s in ax_dev.spines.values():
        s.set_visible(False)
    ax_dev.text(0.0, 0.95, "Input device", color=TEXT_MUTED, fontsize=10)
    ax_dev.add_patch(
        FancyBboxPatch(
            (0.0, 0.05),
            1.0,
            0.65,
            boxstyle="round,pad=0.01,rounding_size=0.04",
            facecolor="#FFFFFF08",
            edgecolor="#FFFFFF20",
            linewidth=0.7,
            transform=ax_dev.transAxes,
        )
    )
    ax_dev.text(
        0.012,
        0.34,
        "[14] Microphone Array 2 (Realtek HD Audio Mic input with SST)  "
        "(in_ch=4, sr=16000)",
        color=TEXT,
        fontsize=10,
    )

    # --- Detection threshold slider ---
    ax_th = fig.add_axes([body_left, 0.79, body_w, 0.07], facecolor=BG_DARK)
    ax_th.set_xticks([])
    ax_th.set_yticks([])
    for s in ax_th.spines.values():
        s.set_visible(False)
    ax_th.text(0.0, 0.95, "Detection threshold", color=TEXT_MUTED, fontsize=10)
    ax_th.add_patch(
        mpatches.Rectangle((0.0, 0.32), 1.0, 0.06, color="#FFFFFF15",
                            transform=ax_th.transAxes)
    )
    ax_th.add_patch(
        mpatches.Rectangle((0.0, 0.32), 0.50, 0.06, color=DANGER,
                            transform=ax_th.transAxes)
    )
    ax_th.add_patch(
        mpatches.Circle((0.50, 0.35), 0.018, color=DANGER,
                         transform=ax_th.transAxes, zorder=5)
    )
    ax_th.text(0.50, 0.62, "0.50", color=DANGER, fontsize=9,
               ha="center", fontweight="bold")

    # --- Start / Stop buttons ---
    for i, (label, enabled) in enumerate([("▶  Start", False), ("■  Stop", True)]):
        ax_b = fig.add_axes(
            [body_left + 0.005 + i * 0.115, 0.71, 0.10, 0.06],
            facecolor=BG_DARK,
        )
        ax_b.set_xticks([])
        ax_b.set_yticks([])
        for s in ax_b.spines.values():
            s.set_visible(False)
        ax_b.add_patch(
            FancyBboxPatch(
                (0.0, 0.05),
                1.0,
                0.9,
                boxstyle="round,pad=0.01,rounding_size=0.12",
                facecolor="#FFFFFF06" if not enabled else "#FFFFFF0E",
                edgecolor="#FFFFFF20",
                linewidth=0.7,
                transform=ax_b.transAxes,
            )
        )
        ax_b.text(
            0.5,
            0.5,
            label,
            color=TEXT_MUTED if not enabled else TEXT,
            fontsize=11,
            ha="center",
            va="center",
        )

    # --- CLEAR banner ---
    ax_ban = fig.add_axes([body_left, 0.56, body_w, 0.13], facecolor=BG_DARK)
    ax_ban.set_xticks([])
    ax_ban.set_yticks([])
    for s in ax_ban.spines.values():
        s.set_visible(False)
    # gradient effect — overlay two boxes
    ax_ban.add_patch(
        FancyBboxPatch(
            (0.0, 0.10),
            1.0,
            0.80,
            boxstyle="round,pad=0.0,rounding_size=0.06",
            facecolor=SUCCESS,
            edgecolor=SUCCESS,
            transform=ax_ban.transAxes,
        )
    )
    ax_ban.add_patch(
        FancyBboxPatch(
            (0.0, 0.10),
            1.0,
            0.80,
            boxstyle="round,pad=0.0,rounding_size=0.06",
            facecolor=SUCCESS_DARK,
            edgecolor="none",
            alpha=0.35,
            transform=ax_ban.transAxes,
        )
    )
    ax_ban.text(
        0.5,
        0.5,
        "✓   C L E A R   ·   P(shahed) = 0.44",
        color="white",
        fontsize=22,
        ha="center",
        va="center",
        fontweight="bold",
    )

    # --- Three bottom panels: gauge, class probs, VU meter ---
    panel_y = 0.06
    panel_h = 0.46
    pad = 0.018
    panel_w = (body_w - 2 * pad) / 3.0

    # Panel 1: Gauge
    ax_g = fig.add_axes([body_left, panel_y, panel_w, panel_h], facecolor=BG_DARK)
    _draw_gauge(ax_g, value_pct=44.0, threshold_pct=50.0)

    # Panel 2: Class probabilities
    ax_p = fig.add_axes(
        [body_left + panel_w + pad, panel_y, panel_w, panel_h], facecolor=BG_DARK
    )
    _draw_class_bars(ax_p, labels=["noise", "shahed"], probs=[0.56, 0.44])

    # Panel 3: VU meter + inference rate card
    ax_v_outer = fig.add_axes(
        [body_left + 2 * (panel_w + pad), panel_y, panel_w, panel_h],
        facecolor=BG_DARK,
    )
    _draw_vu_panel(ax_v_outer, rms_db=-59.5)

    out_path = OUT_DIR / "screen_live_mic.png"
    fig.savefig(out_path, facecolor=BG_DARK, bbox_inches=None)
    plt.close(fig)
    return out_path


def _draw_gauge(ax, value_pct: float, threshold_pct: float):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-0.65, 1.25)
    ax.set_aspect("equal")

    ax.text(0.5, 1.02, "P(shahed)", transform=ax.transAxes, color=TEXT,
            fontsize=13, ha="center", fontweight="bold")

    # gauge background tri-color
    cx, cy = 0.0, 0.0
    r_outer = 1.0
    r_inner = 0.72
    for (a0, a1, col) in [
        (180, 126, "#27AE6033"),
        (126, 90, "#F39C1233"),
        (90, 0, "#E74C3C4D"),
    ]:
        w = Wedge((cx, cy), r_outer, a0, a1, width=r_outer - r_inner,
                  facecolor=col, edgecolor="none")
        ax.add_patch(w)

    # active bar (orange) — value 44%
    # 0% = angle 180°, 100% = 0°
    val_angle = 180 - 1.80 * value_pct
    ax.add_patch(
        Wedge((cx, cy), 0.97, val_angle, 180, width=0.22, facecolor=ACCENT2,
              edgecolor=ACCENT2)
    )

    # threshold marker (white line at 50%)
    th_angle = np.deg2rad(180 - 1.80 * threshold_pct)
    ax.plot(
        [cx + r_inner * np.cos(th_angle), cx + r_outer * np.cos(th_angle)],
        [cy + r_inner * np.sin(th_angle), cy + r_outer * np.sin(th_angle)],
        color="white",
        linewidth=2.5,
    )
    ax.text(0.0, 1.04, f"{threshold_pct:.0f}", color="white", fontsize=9, ha="center")

    # axis labels
    ax.text(-1.05, -0.08, "0", color=TEXT_MUTED, fontsize=10, ha="center")
    ax.text(+1.05, -0.08, "100", color=TEXT_MUTED, fontsize=10, ha="center")

    # value number — centred inside the half-circle (below the arc)
    ax.text(0.0, 0.30, f"{value_pct:.0f} %", color=ACCENT2, fontsize=34,
            ha="center", va="center", fontweight="bold")
    ax.text(0.0, -0.05, f"▼ -{threshold_pct - value_pct:.0f}",
            color=SUCCESS, fontsize=12, ha="center", va="center")


def _draw_class_bars(ax, labels, probs):
    ax.text(0.5, 1.06, "Class probabilities", transform=ax.transAxes,
            color=TEXT, fontsize=13, ha="center", fontweight="bold")
    colors = [DANGER if l == "shahed" else COOL for l in labels]
    y_pos = np.arange(len(labels))
    pct = np.asarray(probs) * 100
    ax.barh(y_pos, pct, color=colors, edgecolor="white", linewidth=0.5,
            height=0.55)
    for yi, p in zip(y_pos, pct):
        ax.text(p + 2, yi, f"{p:.1f} %", color=TEXT, va="center", fontsize=10)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, color=TEXT, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlim(0, 110)
    ax.set_xticks([0, 50, 100])
    ax.set_xlabel("probability (%)", color=TEXT_MUTED, fontsize=9)
    ax.tick_params(axis="x", colors=TEXT_MUTED, labelsize=9)
    ax.tick_params(axis="y", colors=TEXT, labelsize=10)
    ax.grid(axis="x", color="#FFFFFF12", linestyle="-", linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_visible(False)


def _draw_vu_panel(ax, rms_db: float):
    """Two stacked: VU meter on top, inference-rate KPI card on bottom."""
    fig = ax.figure
    pos = ax.get_position()
    ax.set_visible(False)

    # Top: VU meter
    vu_ax = fig.add_axes(
        [pos.x0, pos.y0 + 0.30 * pos.height,
         pos.width, 0.70 * pos.height], facecolor=BG_DARK
    )
    vu_ax.set_xticks([])
    vu_ax.set_yticks([])
    for s in vu_ax.spines.values():
        s.set_visible(False)
    vu_ax.text(0.5, 0.92, "Input level", transform=vu_ax.transAxes, color=TEXT,
               fontsize=13, ha="center", fontweight="bold")
    # three-color bar 0..1 representing -60..0 dB
    bar_y = 0.45
    bar_h = 0.18
    vu_ax.add_patch(
        mpatches.Rectangle((0.07, bar_y), 0.45, bar_h, color="#27AE6066",
                            transform=vu_ax.transAxes)
    )
    vu_ax.add_patch(
        mpatches.Rectangle((0.52, bar_y), 0.30, bar_h, color="#F39C1266",
                            transform=vu_ax.transAxes)
    )
    vu_ax.add_patch(
        mpatches.Rectangle((0.82, bar_y), 0.10, bar_h, color="#E74C3C77",
                            transform=vu_ax.transAxes)
    )
    # value pointer: -59.5 dB → very near the left edge
    db_clamped = max(-60.0, min(0.0, rms_db))
    frac = (db_clamped + 60.0) / 60.0
    x_ptr = 0.07 + (0.92 - 0.07) * frac
    vu_ax.add_patch(
        mpatches.Rectangle((x_ptr - 0.005, bar_y - 0.06),
                            0.01, bar_h + 0.12,
                            color="white",
                            transform=vu_ax.transAxes, zorder=5)
    )
    vu_ax.text(
        0.96,
        bar_y + bar_h + 0.07,
        f"{rms_db:.1f} dB",
        color=TEXT,
        fontsize=16,
        ha="right",
        fontweight="bold",
        transform=vu_ax.transAxes,
    )

    # Bottom: inference-rate KPI
    kpi_ax = fig.add_axes(
        [pos.x0, pos.y0, pos.width, 0.28 * pos.height], facecolor=BG_DARK
    )
    kpi_ax.set_xticks([])
    kpi_ax.set_yticks([])
    for s in kpi_ax.spines.values():
        s.set_visible(False)
    kpi_ax.add_patch(
        FancyBboxPatch(
            (0.02, 0.05),
            0.96,
            0.90,
            boxstyle="round,pad=0.02,rounding_size=0.10",
            facecolor="#FFFFFF06",
            edgecolor="#FFFFFF18",
            linewidth=0.7,
            transform=kpi_ax.transAxes,
        )
    )
    kpi_ax.text(0.07, 0.70, "INFERENCE RATE", color=TEXT_MUTED, fontsize=9,
                fontweight="bold")
    kpi_ax.text(0.07, 0.18, "2.0 fps · 2000 ms window", color=TEXT,
                fontsize=15, fontweight="bold")


# ──────────────────── 2.  4-mic TDOA MOCK-UP ────────────────────
def make_tdoa_image() -> Path:
    _setup_dark_style()
    fig = plt.figure(figsize=(18, 9), dpi=120, facecolor=BG_DARK)
    draw_sidebar_with_tdoa_controls(fig)

    body_left = 0.18
    body_right = 0.985
    body_w = body_right - body_left

    # --- Four KPI cards top row ---
    kpi_y = 0.83
    kpi_h = 0.14
    pad = 0.012
    kpi_w = (body_w - 3 * pad) / 4.0
    kpis = [
        ("speed of sound", "344.0 m/s", TEXT, None),
        ("angular resolution Δφ", "0.89°", SUCCESS, None),
        ("max TDOA (broadside)", "2.907 ms", TEXT, None),
        ("wavelength @ chosen f", "2.87 m", ACCENT, None),
    ]
    for i, (lbl, val, col, sub) in enumerate(kpis):
        x = body_left + i * (kpi_w + pad)
        draw_kpi_card(fig, [x, kpi_y, kpi_w, kpi_h], lbl, val, color=col, sub=sub)

    # --- Title row ---
    ax_t = fig.add_axes([body_left, 0.74, body_w, 0.05], facecolor=BG_DARK)
    ax_t.set_xticks([])
    ax_t.set_yticks([])
    for s in ax_t.spines.values():
        s.set_visible(False)
    ax_t.text(0.0, 0.4, "Array geometry · source direction · wavefront",
              color=TEXT, fontsize=15, fontweight="bold")
    ax_t.text(0.68, 0.4, "TDOA per pair", color=TEXT, fontsize=15,
              fontweight="bold")

    # --- Geometry plot ---
    geo_w = body_w * 0.62
    geo_h = 0.68
    ax_g = fig.add_axes(
        [body_left, 0.05, geo_w, geo_h], facecolor=BG_DARK
    )
    _draw_tdoa_geometry(ax_g)

    # --- TDOA table ---
    tbl_x = body_left + geo_w + 0.01
    tbl_w = body_w - geo_w - 0.01
    ax_tbl = fig.add_axes([tbl_x, 0.30, tbl_w, 0.43], facecolor=BG_DARK)
    _draw_tdoa_table(ax_tbl)

    # --- Info banner under table ---
    ax_info = fig.add_axes([tbl_x, 0.08, tbl_w, 0.18], facecolor=BG_DARK)
    ax_info.set_xticks([])
    ax_info.set_yticks([])
    for s in ax_info.spines.values():
        s.set_visible(False)
    ax_info.add_patch(
        FancyBboxPatch(
            (0.0, 0.05),
            1.0,
            0.9,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            facecolor="#FF6B3310",
            edgecolor="#FF6B3360",
            linewidth=0.9,
            transform=ax_info.transAxes,
        )
    )
    ax_info.text(
        0.04, 0.65,
        "A sample-synchronous 4-mic array is the minimum hardware\n"
        "requirement — independent USB mics drift and cannot resolve\n"
        "sub-millisecond TDOAs.",
        color=TEXT, fontsize=10, linespacing=1.4, transform=ax_info.transAxes,
    )

    out_path = OUT_DIR / "screen_tdoa_concept.png"
    fig.savefig(out_path, facecolor=BG_DARK)
    plt.close(fig)
    return out_path


def _draw_tdoa_geometry(ax):
    ax.set_facecolor(BG_DARK)
    ax.text(
        0.5, 1.02,
        "4-mic square array — baseline 100 cm · source @ 30° · c = 344.0 m/s",
        transform=ax.transAxes, color=TEXT, fontsize=11, ha="center",
        fontweight="bold",
    )
    half = 0.5  # 1 m baseline
    mics = np.array([[-half, -half], [+half, -half], [+half, +half], [-half, +half]])
    az_deg = 30.0
    az = np.deg2rad(az_deg)
    u = np.array([np.sin(az), np.cos(az)])
    p_source = u * 2.0

    # Wavefront (dotted line)
    perp = np.array([-u[1], u[0]])
    wf_a = p_source - perp * 1.4
    wf_b = p_source + perp * 1.4
    ax.plot([wf_a[0], wf_b[0]], [wf_a[1], wf_b[1]], color=COOL, linestyle=":",
            linewidth=2, label="wavefront")

    # Source arrow (from source toward origin)
    ax.annotate(
        "",
        xy=(0.0, 0.0),
        xytext=(p_source[0], p_source[1]),
        arrowprops=dict(arrowstyle="->", color=COOL, lw=2.5),
    )
    ax.scatter(p_source[0], p_source[1], s=260, marker="D",
               facecolor=COOL, edgecolor="white", zorder=5)
    ax.text(p_source[0] + 0.12, p_source[1] + 0.08, "source",
            color=COOL, fontsize=12, fontweight="bold")

    # Microphones — larger and with offset labels (no overlap with mics)
    ax.scatter(mics[:, 0], mics[:, 1], s=600, facecolor=ACCENT,
               edgecolor="white", linewidth=2, zorder=4)
    label_offsets = [(-0.15, -0.18), (+0.15, -0.18), (+0.15, +0.18), (-0.15, +0.18)]
    for i, ((x, y), (dx, dy)) in enumerate(zip(mics, label_offsets)):
        ax.text(x + dx, y + dy, f"M{i + 1}", color=TEXT, fontsize=14,
                ha="center", va="center", fontfamily="monospace",
                fontweight="bold")

    # TDOA annotations OUTSIDE the mic square to avoid overlap
    pairs_with_offsets = [
        ((0, 1), (0.0, -0.55)),    # below
        ((1, 2), (+0.92, 0.0)),    # right
        ((3, 2), (0.0, +0.62)),    # above
        ((0, 3), (-0.92, 0.0)),    # left
    ]
    for (i, j), (off_x, off_y) in pairs_with_offsets:
        d = mics[j] - mics[i]
        tau_s = float(np.dot(d, u)) / 344.0
        mid = (mics[i] + mics[j]) / 2.0
        ax.text(
            mid[0] + off_x, mid[1] + off_y,
            f"τ{i + 1}{j + 1} = {tau_s * 1000:+.3f} ms",
            color=TEXT_MUTED, fontsize=9, ha="center",
            fontfamily="monospace",
            bbox=dict(facecolor=BG_DARK, edgecolor="none", pad=2),
        )

    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-1.6, 2.5)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)", color=TEXT_MUTED, fontsize=10)
    ax.set_ylabel("y (m)", color=TEXT_MUTED, fontsize=10)
    ax.tick_params(colors=TEXT_MUTED, labelsize=9)
    ax.axhline(0, color="#FFFFFF12", linewidth=0.6)
    ax.axvline(0, color="#FFFFFF12", linewidth=0.6)
    ax.grid(True, color="#FFFFFF08", linewidth=0.5)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_visible(False)


def _draw_tdoa_table(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor(BG_DARK)
    for s in ax.spines.values():
        s.set_visible(False)

    header = ["pair", "τ (ms)", "samples @ 22 050 Hz", "samples @ 48 000 Hz"]
    rows = [
        ["M1–M2", "+1.453", "+32.05", "+69.76"],
        ["M1–M3", "+3.971", "+87.55", "+190.59"],
        ["M1–M4", "+2.517", "+55.50", "+120.83"],
        ["M2–M3", "+2.517", "+55.50", "+120.83"],
        ["M2–M4", "+1.064", "+23.46", "+51.07"],
        ["M3–M4", "−1.453", "−32.05", "−69.76"],
    ]
    n_rows = len(rows)
    col_w = [0.14, 0.20, 0.32, 0.34]
    x_starts = np.cumsum([0] + col_w[:-1])
    row_h = 1.0 / (n_rows + 1.5)

    # header strip
    ax.add_patch(
        mpatches.Rectangle(
            (0.0, 1.0 - row_h),
            1.0,
            row_h,
            color="#FF6B3340",
            transform=ax.transAxes,
        )
    )
    for x, h, w in zip(x_starts, header, col_w):
        align = "left" if "pair" in h else "right"
        x_text = x + 0.01 if align == "left" else x + w - 0.01
        ax.text(
            x_text,
            1.0 - row_h / 2,
            h,
            color=TEXT,
            fontsize=9.5,
            fontweight="bold",
            ha=align,
            va="center",
            transform=ax.transAxes,
        )

    # body rows
    for i, row in enumerate(rows):
        y_top = 1.0 - (i + 2) * row_h
        if i % 2 == 0:
            ax.add_patch(
                mpatches.Rectangle(
                    (0.0, y_top),
                    1.0,
                    row_h,
                    color="#FFFFFF06",
                    transform=ax.transAxes,
                )
            )
        for j, (x, cell, w) in enumerate(zip(x_starts, row, col_w)):
            align = "left" if j == 0 else "right"
            x_text = x + 0.01 if align == "left" else x + w - 0.01
            ax.text(
                x_text,
                y_top + row_h / 2,
                cell,
                color=TEXT,
                fontsize=10,
                ha=align,
                va="center",
                fontfamily="monospace",
                transform=ax.transAxes,
            )


# ─────────────────────────── entry point ─────────────────────────
def main() -> None:
    p1 = make_live_mic_image()
    p2 = make_tdoa_image()
    print(f"Wrote {p1}  ({p1.stat().st_size:,} bytes)")
    print(f"Wrote {p2}  ({p2.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()

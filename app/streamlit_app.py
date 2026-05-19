"""Streamlit dashboard for the drone-detector project.

Pages
-----
1. Overview            — KPI cards, class distribution, confusion matrix, pipeline diagram
2. Train pipeline      — animated pipeline, live progress, latest results
3. Analyse a file      — waveform + spectrogram + probability timeline overlay
4. Live microphone     — DETECT/CLEAR banner, gauge, VU meter, rolling spectrogram
5. 4-mic TDOA concept  — 7 interactive sliders driving the array, table, resolution curve
"""

from __future__ import annotations

import queue
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import streamlit as st

from drone_detector.config import settings
from drone_detector.data.dataset import Dataset
from drone_detector.data.ingest import media_to_wav
from drone_detector.features.audio import (
    log_mel_spectrogram,
    mfcc_feature_vector,
    speed_of_sound,
)
from drone_detector.live.mic import (
    DetectionEvent,
    list_input_devices,
    stream_detections,
)
from drone_detector.models.baseline import ModelBundle
from drone_detector.pipeline.train import run_full_pipeline
from drone_detector.viz.plots import (
    plotly_class_distribution,
    plotly_class_probabilities,
    plotly_confusion_matrix,
    plotly_mel_spectrogram,
    plotly_pipeline_diagram,
    plotly_probability_gauge,
    plotly_probability_history,
    plotly_resolution_curve,
    plotly_rolling_spectrogram,
    plotly_speed_of_sound_curve,
    plotly_tdoa_geometry,
    plotly_tdoa_table,
    plotly_vu_meter,
    plotly_waveform,
    plotly_wavelength_bar,
)


# ════════════════════════ Page config + global CSS ═════════════════════════

st.set_page_config(
    page_title="Shahed Acoustic Detector",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)


CUSTOM_CSS = """
<style>
@keyframes pulse_detect {
  0%   { box-shadow: 0 0 0 0 rgba(231, 76, 60, 0.65); }
  70%  { box-shadow: 0 0 0 24px rgba(231, 76, 60, 0); }
  100% { box-shadow: 0 0 0 0 rgba(231, 76, 60, 0); }
}
@keyframes pulse_clear {
  0%   { box-shadow: 0 0 0 0 rgba(39, 174, 96, 0.5); }
  70%  { box-shadow: 0 0 0 18px rgba(39, 174, 96, 0); }
  100% { box-shadow: 0 0 0 0 rgba(39, 174, 96, 0); }
}
@keyframes pulse_idle {
  0%   { opacity: 0.35; }
  50%  { opacity: 0.75; }
  100% { opacity: 0.35; }
}
@keyframes gradient_shift {
  0%   { background-position: 0% 50%; }
  50%  { background-position: 100% 50%; }
  100% { background-position: 0% 50%; }
}
.hero {
  background: linear-gradient(120deg, #1A1F2B 0%, #2C1B14 50%, #1A1F2B 100%);
  background-size: 200% 200%;
  animation: gradient_shift 12s ease infinite;
  border: 1px solid rgba(255, 107, 53, 0.25);
  border-radius: 14px;
  padding: 22px 28px;
  margin-bottom: 18px;
}
.hero h1 {
  margin: 0 0 4px 0;
  font-size: 26px;
  letter-spacing: 0.4px;
  color: #FAFAFA;
}
.hero .sub {
  color: rgba(255, 255, 255, 0.65);
  font-size: 14px;
}
.detect-banner {
  background: linear-gradient(135deg, #E74C3C, #C0392B);
  color: white; padding: 22px; border-radius: 12px;
  text-align: center; font-size: 26px; font-weight: 800;
  letter-spacing: 2px; animation: pulse_detect 1.4s infinite;
  border: 2px solid rgba(255, 255, 255, 0.15);
}
.clear-banner {
  background: linear-gradient(135deg, #27AE60, #1E8449);
  color: white; padding: 22px; border-radius: 12px;
  text-align: center; font-size: 22px; font-weight: 700;
  letter-spacing: 2px; animation: pulse_clear 2s infinite;
  border: 2px solid rgba(255, 255, 255, 0.10);
}
.idle-banner {
  background: linear-gradient(135deg, #34495E, #2C3E50);
  color: rgba(255, 255, 255, 0.75); padding: 22px; border-radius: 12px;
  text-align: center; font-size: 18px; font-weight: 600;
  letter-spacing: 2px; animation: pulse_idle 3s ease-in-out infinite;
  border: 1px dashed rgba(255, 255, 255, 0.18);
}
.kpi-card {
  background: rgba(255, 255, 255, 0.025);
  border: 1px solid rgba(255, 255, 255, 0.08);
  border-radius: 12px; padding: 16px 18px; height: 100%;
}
.kpi-card .label {
  color: rgba(255, 255, 255, 0.6);
  font-size: 11px; letter-spacing: 1.4px; text-transform: uppercase;
}
.kpi-card .value {
  color: #FAFAFA; font-size: 28px; font-weight: 700; margin-top: 4px;
}
.kpi-card .accent { color: #FF6B35; }
.kpi-card .ok { color: #27AE60; }
.kpi-card .warn { color: #F39C12; }
.kpi-card .danger { color: #E74C3C; }
.tip-card {
  background: rgba(255, 107, 53, 0.06);
  border-left: 3px solid #FF6B35;
  border-radius: 6px;
  padding: 12px 16px;
  margin-bottom: 10px;
  color: rgba(255, 255, 255, 0.85);
  font-size: 13px;
}
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #1A1F2B, #11141B);
  border-right: 1px solid rgba(255, 255, 255, 0.05);
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


MODEL_PATH = settings.models_dir / "baseline.pkl"


# ───────────────────────────── helpers ──────────────────────────────


@st.cache_resource(show_spinner=False)
def load_model_cached(path_str: str, mtime: float) -> ModelBundle | None:  # noqa: ARG001
    p = Path(path_str)
    if not p.exists():
        return None
    return ModelBundle.load(p)


def get_model() -> ModelBundle | None:
    if not MODEL_PATH.exists():
        return None
    return load_model_cached(str(MODEL_PATH), MODEL_PATH.stat().st_mtime)


SUPPORTED_EXTS = {
    ".mp3", ".wav", ".flac", ".ogg", ".opus", ".m4a", ".aac",
    ".mp4", ".mkv", ".webm", ".mov", ".avi",
}


def count_audio_files() -> dict[str, int]:
    out: dict[str, int] = {}
    if not settings.audio_dir.exists():
        return out
    for sub in settings.audio_dir.iterdir():
        if sub.is_dir():
            out[sub.name] = sum(
                1 for p in sub.iterdir()
                if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
            )
    return out


def count_chunks() -> dict[str, int]:
    out: dict[str, int] = {}
    if not settings.chunks_dir.exists():
        return out
    for sub in settings.chunks_dir.iterdir():
        if sub.is_dir():
            out[sub.name] = sum(1 for _ in sub.glob("*.wav"))
    return out


def kpi_card(label: str, value: str, *, klass: str = "") -> str:
    return (
        f'<div class="kpi-card"><div class="label">{label}</div>'
        f'<div class="value {klass}">{value}</div></div>'
    )


def hero(title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="hero"><h1>{title}</h1><div class="sub">{subtitle}</div></div>',
        unsafe_allow_html=True,
    )


def rms_db(window: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(window**2)) + 1e-12)
    return 20.0 * np.log10(rms)


# ════════════════════════════ PAGES ═════════════════════════════════


# ─────────────────────────── Overview ───────────────────────────────

def page_overview() -> None:
    hero("📡  Drone Acoustic Detector",
         "Defensive early-warning research · Shahed-136 · Politechnika Lubelska · master's thesis")

    model = get_model()
    audio_counts = count_audio_files()
    chunk_counts = count_chunks()

    n_classes = len(model.labels) if model else 0
    n_train = (model.meta.get("n_train", 0) if model else 0)
    n_test = (model.meta.get("n_test", 0) if model else 0)

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(
        kpi_card("model status",
                 "ready" if model else "not trained",
                 klass="ok" if model else "warn"),
        unsafe_allow_html=True,
    )
    c2.markdown(
        kpi_card("classes",
                 ", ".join(model.labels) if model else "—",
                 klass="accent" if model else ""),
        unsafe_allow_html=True,
    )
    c3.markdown(
        kpi_card("training chunks", f"{n_train + n_test:,}" if model else "0",
                 klass="accent"),
        unsafe_allow_html=True,
    )
    c4.markdown(
        kpi_card("sample rate", f"{settings.sample_rate:,} Hz"),
        unsafe_allow_html=True,
    )

    st.markdown("")
    st.subheader("Processing pipeline")
    st.plotly_chart(plotly_pipeline_diagram(), width="stretch",
                    config={"displayModeBar": False})

    # Class distribution + feature configuration
    colA, colB = st.columns([2, 1])

    with colA:
        st.subheader("Dataset class distribution")
        all_labels = sorted(set(audio_counts) | set(chunk_counts))
        if not all_labels:
            st.markdown(
                '<div class="tip-card">📥 No audio yet. Drop MP3 files into '
                '<code>audio/shahed/</code> and <code>audio/noise/</code>, then '
                'open the <b>Train pipeline</b> page.</div>',
                unsafe_allow_html=True,
            )
        else:
            chunks = {l: chunk_counts.get(l, 0) for l in all_labels}
            if any(chunks.values()):
                st.plotly_chart(
                    plotly_class_distribution(chunks, "Training chunks per class"),
                    width="stretch",
                    config={"displayModeBar": False},
                )
            df = pd.DataFrame({
                "label": all_labels,
                "source files": [audio_counts.get(l, 0) for l in all_labels],
                "training chunks": [chunk_counts.get(l, 0) for l in all_labels],
            })
            st.dataframe(df, hide_index=True, width="stretch")

    with colB:
        st.subheader("Feature configuration")
        feat_md = (
            f"- **window**: {settings.chunk_duration} s "
            f"@ {settings.sample_rate} Hz\n"
            f"- **overlap**: {int(settings.chunk_overlap * 100)} %\n"
            f"- **MFCC**: {settings.n_mfcc} coeffs\n"
            f"- **FFT**: {settings.n_fft} · hop {settings.hop_length}\n"
            f"- **feature dim**: {3 * settings.n_mfcc}D (mean + std + Δ-mean)\n"
            f"- **classifier**: RandomForest "
            f"({settings.rf_n_estimators} trees, balanced)\n"
            f"- **detection threshold**: {settings.detection_threshold:.2f}"
        )
        st.markdown(feat_md)

        st.subheader("Quick start")
        st.markdown(
            '<div class="tip-card">1. Drop MP3 → <code>audio/shahed/</code> + '
            '<code>audio/noise/</code></div>'
            '<div class="tip-card">2. <b>Train pipeline</b> → press <i>Run pipeline</i></div>'
            '<div class="tip-card">3. <b>Live microphone</b> → play a Shahed clip near the mic</div>',
            unsafe_allow_html=True,
        )

    # Latest evaluation
    if model is not None:
        st.markdown("---")
        st.subheader("Latest evaluation")

        cm_path = settings.reports_dir / "confusion_matrix.png"
        report_md_path = settings.reports_dir / "evaluation.md"

        ec1, ec2 = st.columns([1.2, 1])
        with ec1:
            if cm_path.exists():
                st.image(str(cm_path), caption="Confusion matrix (test split)",
                         width="stretch")
            else:
                st.info("Run the pipeline to generate an evaluation report.")
        with ec2:
            st.markdown("**Training metadata**")
            st.json(model.meta, expanded=False)
            if report_md_path.exists():
                with st.expander("Full evaluation report"):
                    st.markdown(report_md_path.read_text(encoding="utf-8"))


# ─────────────────────────── Train ──────────────────────────────────

def page_train() -> None:
    hero("⚙️  Train pipeline",
         "MP3 → mono WAV → 2 s chunks → MFCC → Random Forest → evaluation report")

    st.plotly_chart(plotly_pipeline_diagram(), width="stretch",
                    config={"displayModeBar": False})

    audio_counts = count_audio_files()
    chunk_counts = count_chunks()

    c1, c2, c3 = st.columns(3)
    c1.markdown(
        kpi_card("audio classes", str(len(audio_counts)) if audio_counts else "0",
                 klass="accent" if audio_counts else "warn"),
        unsafe_allow_html=True,
    )
    c2.markdown(
        kpi_card("source files", str(sum(audio_counts.values()) or 0),
                 klass="accent"),
        unsafe_allow_html=True,
    )
    c3.markdown(
        kpi_card("training chunks", str(sum(chunk_counts.values()) or 0),
                 klass="accent"),
        unsafe_allow_html=True,
    )

    if not audio_counts or all(v == 0 for v in audio_counts.values()):
        st.markdown("")
        st.markdown(
            '<div class="tip-card">⚠️  No audio files found in '
            '<code>audio/</code>. Drop MP3s into <code>audio/shahed/</code> and '
            '<code>audio/noise/</code>, then return here.</div>',
            unsafe_allow_html=True,
        )
        return

    st.markdown("")
    st.subheader("Configuration")
    o1, o2, o3 = st.columns(3)
    skip_ingest = o1.toggle("Skip ingest", value=False,
                             help="Re-use the WAVs already in data/raw/")
    skip_chunk = o2.toggle("Skip chunking", value=False,
                            help="Re-use the chunks already in data/chunks/")
    skip_features = o3.toggle("Reuse cached features", value=False,
                              help="Re-use data/features/dataset.npz if present")

    if st.button("▶  Run pipeline", type="primary", width="stretch"):
        with st.status("Training…", expanded=True) as status:
            try:
                bundle = run_full_pipeline(
                    skip_ingest=skip_ingest,
                    skip_chunk=skip_chunk,
                    skip_features=skip_features,
                )
                status.update(label="✅ Pipeline complete", state="complete")
                load_model_cached.clear()

                st.balloons()

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("classes", len(bundle.labels))
                m2.metric("train samples", bundle.meta["n_train"])
                m3.metric("test samples", bundle.meta["n_test"])
                m4.metric("feature dim", bundle.feature_dim)

                # Show resulting confusion matrix immediately
                cm_path = settings.reports_dir / "confusion_matrix.png"
                if cm_path.exists():
                    st.image(str(cm_path), caption="Confusion matrix",
                             width="stretch")

                report_md_path = settings.reports_dir / "evaluation.md"
                if report_md_path.exists():
                    with st.expander("📄  Full evaluation report", expanded=True):
                        st.markdown(report_md_path.read_text(encoding="utf-8"))

            except Exception as exc:  # noqa: BLE001
                status.update(label=f"❌ Failed: {exc}", state="error")
                st.exception(exc)


# ─────────────────────── Analyse a file ─────────────────────────────

def page_analyse_file() -> None:
    hero("🔍  Analyse a file",
         "Upload any audio/video. The detector slides a 2-second window across it and "
         "scores every frame.")

    model = get_model()
    if model is None:
        st.warning("No trained model yet — open the **Train pipeline** page first.")
        return

    uploaded = st.file_uploader(
        "Upload an MP3/WAV/MP4 file",
        type=["mp3", "wav", "m4a", "ogg", "flac", "opus", "mp4", "mkv", "webm", "mov"],
    )
    if uploaded is None:
        st.markdown(
            '<div class="tip-card">💡  Tip — for the live defence demo you can '
            'just drag a Shahed YouTube download here to confirm the model '
            'fires before going to the microphone page.</div>',
            unsafe_allow_html=True,
        )
        return

    tmp_in = settings.reports_dir / f"_upload{Path(uploaded.name).suffix}"
    tmp_in.write_bytes(uploaded.getvalue())
    tmp_wav = settings.reports_dir / "_upload.wav"
    media_to_wav(tmp_in, tmp_wav)
    audio, sr = sf.read(tmp_wav, always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)

    st.audio(uploaded.getvalue())
    st.caption(f"file: **{uploaded.name}** · {len(audio) / sr:0.2f} s · {sr} Hz mono")

    win_n = int(settings.live_window_seconds * sr)
    hop_n = int(settings.live_hop_seconds * sr)
    if len(audio) < win_n:
        st.warning("File is shorter than the analysis window.")
        return

    times: list[float] = []
    probs: list[float] = []
    pos_id = model.positive_class_id("shahed")
    for start in range(0, len(audio) - win_n + 1, hop_n):
        window = audio[start : start + win_n]
        vec = mfcc_feature_vector(window, sr)[None, :]
        p = model.predict_proba(vec)[0]
        probs.append(float(p[pos_id]) if pos_id is not None else float(np.max(p)))
        times.append((start + win_n / 2) / sr)

    probs_arr = np.asarray(probs)
    times_arr = np.asarray(times)
    above = probs_arr >= settings.detection_threshold

    g1, g2 = st.columns([1, 2])
    with g1:
        st.plotly_chart(
            plotly_probability_gauge(float(probs_arr.max()),
                                     settings.detection_threshold,
                                     title="peak P(shahed)"),
            width="stretch",
            config={"displayModeBar": False},
        )
    with g2:
        m1, m2, m3 = st.columns(3)
        m1.markdown(kpi_card("peak", f"{probs_arr.max():.2f}",
                              klass="danger" if probs_arr.max() >= 0.5 else "ok"),
                    unsafe_allow_html=True)
        m2.markdown(kpi_card("mean", f"{probs_arr.mean():.2f}"),
                    unsafe_allow_html=True)
        m3.markdown(kpi_card("frames above threshold",
                              f"{100 * above.mean():.0f} %",
                              klass="danger" if above.mean() > 0.3 else "warn"),
                    unsafe_allow_html=True)
        st.markdown("")
        st.markdown(kpi_card("verdict",
                              "SHAHED LIKELY" if above.mean() > 0.3 else "CLEAR",
                              klass="danger" if above.mean() > 0.3 else "ok"),
                    unsafe_allow_html=True)

    st.subheader("Detection probability over time")
    st.plotly_chart(
        plotly_probability_history(times_arr, probs_arr, settings.detection_threshold),
        width="stretch",
        config={"displayModeBar": False},
    )

    cw1, cw2 = st.columns(2)
    with cw1:
        st.subheader("Waveform")
        st.plotly_chart(
            plotly_waveform(audio[: min(len(audio), sr * 20)], sr, height=220),
            width="stretch",
            config={"displayModeBar": False},
        )
    with cw2:
        st.subheader("Mel spectrogram")
        st.plotly_chart(
            plotly_mel_spectrogram(audio[: min(len(audio), sr * 10)], sr, height=220),
            width="stretch",
            config={"displayModeBar": False},
        )


# ─────────────────────── Live microphone ────────────────────────────


def page_live_mic() -> None:
    hero("🎙️  Live microphone",
         "Real-time inference from the USB Maono input — pulse banner, gauge, VU, rolling spectrogram.")

    model = get_model()
    if model is None:
        st.warning("No trained model yet — open the **Train pipeline** page first.")
        return

    devices = list_input_devices()
    if not devices:
        st.error("No input audio devices detected.")
        return

    device_labels = [
        f"[{d['index']}] {d['name']}  (in_ch={d['max_input_channels']}, sr={int(d['default_samplerate'])})"
        for d in devices
    ]
    # Pick the first device whose name contains "Realtek" or first overall
    default_idx = next(
        (i for i, d in enumerate(devices) if "realtek" in d["name"].lower()),
        0,
    )
    chosen = st.selectbox("Input device", device_labels, index=default_idx)
    device_index = devices[device_labels.index(chosen)]["index"]

    threshold = st.slider(
        "Detection threshold",
        min_value=0.10, max_value=0.95, value=float(settings.detection_threshold),
        step=0.05,
    )

    # --- Session state ---
    ss = st.session_state
    ss.setdefault("live_running", False)
    ss.setdefault("live_history", [])  # list[tuple[float, float]]
    ss.setdefault("live_windows", deque(maxlen=5))
    ss.setdefault("live_stop_event", None)
    ss.setdefault("live_event_queue", None)
    ss.setdefault("live_last_event", None)
    ss.setdefault("live_start_ts", None)

    bcol1, bcol2, _spacer = st.columns([1, 1, 4])
    if bcol1.button("▶  Start", type="primary", disabled=ss.live_running,
                     width="stretch"):
        # IMPORTANT: capture local refs and pass them as thread args.
        # Touching st.session_state from a background thread is unsupported
        # by Streamlit and silently breaks the event flow.
        local_stop = threading.Event()
        local_queue: queue.Queue = queue.Queue()

        ss.live_running = True
        ss.live_history = []
        ss.live_windows = deque(maxlen=5)
        ss.live_stop_event = local_stop
        ss.live_event_queue = local_queue
        ss.live_last_event = None
        ss.live_start_ts = time.time()

        def runner(_model, _device, _stop, _q) -> None:
            try:
                for ev in stream_detections(
                    _model,
                    device=_device,
                    stop_event=_stop,
                ):
                    _q.put(ev)
            except Exception as exc:  # noqa: BLE001
                _q.put(exc)

        t = threading.Thread(
            target=runner,
            args=(model, device_index, local_stop, local_queue),
            daemon=True,
            name="DroneDetectorLiveStream",
        )
        # Best-effort: give the thread the current Streamlit script context
        # so any st.* call inside (e.g. via a callback) wouldn't error.
        try:
            from streamlit.runtime.scriptrunner import add_script_run_ctx
            add_script_run_ctx(t)
        except Exception:  # noqa: BLE001
            pass
        t.start()

    if bcol2.button("■  Stop", disabled=not ss.live_running,
                     width="stretch"):
        if ss.live_stop_event is not None:
            ss.live_stop_event.set()
        ss.live_running = False

    # --- Drain queue once per rerun ---
    drained_error: Exception | None = None
    if ss.live_running and ss.live_event_queue is not None:
        pos_id = model.positive_class_id("shahed")
        while True:
            try:
                ev = ss.live_event_queue.get_nowait()
            except queue.Empty:
                break
            if isinstance(ev, Exception):
                drained_error = ev
                ss.live_running = False
                break
            ss.live_last_event = ev
            p_shahed = (
                float(ev.proba_vector[pos_id]) if pos_id is not None else ev.probability
            )
            t_rel = ev.timestamp - ss.live_start_ts
            ss.live_history.append((t_rel, p_shahed))
            ss.live_windows.append(ev.window)

    if drained_error is not None:
        st.error("Live capture failed — see stack trace below.")
        st.exception(drained_error)

    # --- Status banner ---
    banner = st.empty()
    last = ss.live_last_event
    pos_id = model.positive_class_id("shahed")
    if not ss.live_running:
        banner.markdown(
            '<div class="idle-banner">●  IDLE — press Start to listen</div>',
            unsafe_allow_html=True,
        )
        p_shahed = 0.0
    elif last is None:
        banner.markdown(
            '<div class="idle-banner">●  warming up…</div>',
            unsafe_allow_html=True,
        )
        p_shahed = 0.0
    else:
        p_shahed = (
            float(last.proba_vector[pos_id]) if pos_id is not None else last.probability
        )
        if p_shahed >= threshold:
            banner.markdown(
                f'<div class="detect-banner">▲  SHAHED DETECTED · P = {p_shahed:.2f}</div>',
                unsafe_allow_html=True,
            )
        else:
            banner.markdown(
                f'<div class="clear-banner">✓  CLEAR · P(shahed) = {p_shahed:.2f}</div>',
                unsafe_allow_html=True,
            )

    st.markdown("")

    # --- Row 1: gauge + per-class bars + VU meter ---
    g1, g2, g3 = st.columns([1, 1.2, 1])
    with g1:
        st.plotly_chart(
            plotly_probability_gauge(p_shahed, threshold, title="P(shahed)"),
            width="stretch",
            config={"displayModeBar": False, "staticPlot": False},
        )
    with g2:
        st.markdown("**Class probabilities**")
        if last is not None:
            st.plotly_chart(
                plotly_class_probabilities(model.labels, last.proba_vector),
                width="stretch",
                config={"displayModeBar": False},
            )
        else:
            zeros = np.zeros(len(model.labels))
            st.plotly_chart(
                plotly_class_probabilities(model.labels, zeros),
                width="stretch",
                config={"displayModeBar": False},
            )
    with g3:
        st.markdown("**Input level**")
        db = rms_db(last.window) if last is not None else -60.0
        st.plotly_chart(
            plotly_vu_meter(db),
            width="stretch",
            config={"displayModeBar": False},
        )
        # Frame stats card
        fps_s = (
            f"{1.0 / settings.live_hop_seconds:.1f} fps · "
            f"{int(settings.live_window_seconds * 1000)} ms window"
        )
        st.markdown(kpi_card("inference rate", fps_s), unsafe_allow_html=True)

    # --- Row 2: rolling spectrogram + recent waveform ---
    s1, s2 = st.columns(2)
    with s1:
        st.markdown("**Rolling mel spectrogram (last ~10 s)**")
        st.plotly_chart(
            plotly_rolling_spectrogram(list(ss.live_windows), model.sample_rate,
                                       height=240),
            width="stretch",
            config={"displayModeBar": False},
        )
    with s2:
        st.markdown("**Current window waveform**")
        if last is not None:
            st.plotly_chart(
                plotly_waveform(last.window, model.sample_rate, height=240),
                width="stretch",
                config={"displayModeBar": False},
            )
        else:
            st.empty()

    # --- Row 3: probability history (long view) ---
    st.markdown("**P(shahed) over the last 60 s**")
    if ss.live_history:
        hist = np.array(ss.live_history[-300:])  # keep last 300 events
        st.plotly_chart(
            plotly_probability_history(hist[:, 0], hist[:, 1], threshold),
            width="stretch",
            config={"displayModeBar": False},
        )
    else:
        st.info("History will appear here as soon as the detector fires the first window.")

    # --- Demo tip + auto-rerun ---
    st.markdown(
        '<div class="tip-card">🎯  <b>Defence demo trick:</b> play a Shahed-136 '
        'flight clip from YouTube on your phone, hold it ~30 cm from the '
        'microphone. The banner should turn red within ~1 second.</div>',
        unsafe_allow_html=True,
    )

    if ss.live_running:
        time.sleep(0.4)
        st.rerun()


# ─────────────────────── 4-mic TDOA concept ─────────────────────────


def page_concept_4mic() -> None:
    hero("📐  4-microphone TDOA — interactive paper design",
         "Paper extension of the single-mic detector to a 4-mic square array. "
         "Slide the controls and watch every chart re-derive.")

    with st.sidebar:
        st.markdown("---")
        st.markdown("**TDOA controls**")
        baseline_cm = st.slider("Array baseline (cm)", 5, 100, 30, 1)
        baseline = baseline_cm / 100.0
        sample_rate = st.select_slider("Sample rate (Hz)",
                                        options=[16000, 22050, 32000, 44100, 48000, 96000],
                                        value=22050)
        temp = st.slider("Temperature (°C)", -20, 45, 20)
        rh = st.slider("Relative humidity (%)", 0, 100, 50)
        azimuth = st.slider("Source azimuth (°)", -180, 180, 30, 1)
        elev = st.slider("Source elevation (°)", -10, 60, 0)
        freq = st.select_slider("Source frequency (Hz)",
                                 options=[60, 80, 100, 120, 200, 400, 800,
                                          1600, 3200, 6400],
                                 value=120)

    c = speed_of_sound(temp, rh)
    res_deg = np.rad2deg(c / (baseline * sample_rate)) if baseline > 0 else float("inf")
    max_tdoa_ms = (baseline * 1000.0) / c
    wavelength_m = c / freq

    # --- KPI row ---
    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(kpi_card("speed of sound", f"{c:.1f} m/s"),
                 unsafe_allow_html=True)
    k2.markdown(kpi_card("angular resolution Δφ",
                          f"{res_deg:.2f}°",
                          klass="ok" if res_deg < 5 else "warn" if res_deg < 12 else "danger"),
                 unsafe_allow_html=True)
    k3.markdown(kpi_card("max TDOA (broadside)",
                          f"{max_tdoa_ms:.3f} ms"),
                 unsafe_allow_html=True)
    k4.markdown(kpi_card("wavelength @ chosen f",
                          f"{wavelength_m:.2f} m",
                          klass="accent"),
                 unsafe_allow_html=True)

    st.markdown("")

    # --- Main geometry plot + TDOA table ---
    gc1, gc2 = st.columns([2, 1])
    with gc1:
        st.subheader("Array geometry · source direction · wavefront")
        st.plotly_chart(
            plotly_tdoa_geometry(baseline, azimuth, elev, sample_rate, c),
            width="stretch",
            config={"displayModeBar": False},
        )
    with gc2:
        st.subheader("TDOA per pair")
        st.plotly_chart(
            plotly_tdoa_table(baseline, azimuth, c),
            width="stretch",
            config={"displayModeBar": False},
        )
        st.markdown(
            '<div class="tip-card">A sample-synchronous 4-mic array is the '
            'minimum hardware requirement — independent USB mics drift and '
            'cannot resolve sub-millisecond TDOAs.</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # --- Resolution curve + c(T) ---
    rc1, rc2 = st.columns(2)
    with rc1:
        st.subheader("Angular resolution vs baseline")
        st.plotly_chart(
            plotly_resolution_curve(sample_rate, c),
            width="stretch",
            config={"displayModeBar": False},
        )
    with rc2:
        st.subheader("Speed of sound vs temperature")
        st.plotly_chart(
            plotly_speed_of_sound_curve(rh),
            width="stretch",
            config={"displayModeBar": False},
        )

    st.markdown("---")

    # --- Wavelength comparison ---
    st.subheader("Wavelength across the Shahed harmonic stack")
    st.plotly_chart(
        plotly_wavelength_bar([80, 120, 200, 400, 800, 1600, 3200], c),
        width="stretch",
        config={"displayModeBar": False},
    )

    st.markdown("---")
    st.subheader("Localisation algorithm — sketch")
    st.markdown(
        r"""
1. **Sample-synchronous acquisition** of all 4 channels (single ADC or
   I²S bus with shared MCLK).
2. For each of the 6 pairs $(i,j)$, estimate the TDOA $\tau_{ij}$ via
   GCC-PHAT:
   $$
   R_{ij}(\tau) =
   \mathcal{F}^{-1}\!\!\left\{\frac{X_i(f)\,X_j^{*}(f)}{|X_i(f)\,X_j^{*}(f)|}\right\}(\tau).
   $$
3. Convert TDOAs to path differences using the current $c(T, RH)$.
4. Solve the over-determined linear system
   $A\,\hat{\mathbf{u}} = \mathbf{d}$ for the direction-of-arrival unit
   vector $\hat{\mathbf{u}}$ via least squares.
5. Track $\hat{\mathbf{u}}(t)$ with a bearing-only Extended Kalman Filter
   to derive a trajectory hypothesis under a constant-altitude motion model.

Full derivation in `docs/tdoa_concept.md`.
        """
    )


# ─────────────────────────── Router ─────────────────────────────────


PAGES = {
    "📡  Overview":          page_overview,
    "⚙️  Train pipeline":    page_train,
    "🔍  Analyse a file":    page_analyse_file,
    "🎙️  Live microphone":  page_live_mic,
    "📐  4-mic TDOA concept": page_concept_4mic,
}


def main() -> None:
    with st.sidebar:
        st.markdown(
            '<div style="font-size:22px; font-weight:700; '
            'background: linear-gradient(90deg, #FF6B35, #FFB23F); '
            '-webkit-background-clip: text; -webkit-text-fill-color: transparent;">'
            'Shahed Detector</div>',
            unsafe_allow_html=True,
        )
        st.caption("Politechnika Lubelska · master's project")
        choice = st.radio("Navigation", list(PAGES), label_visibility="collapsed")
        st.markdown("---")
        st.caption(
            "**Safety boundary.** Defensive early-warning research tool. "
            "No target assignment, no weapon guidance, no engagement coordinates."
        )

    PAGES[choice]()


def _is_running_under_streamlit() -> bool:
    """True when the script was launched via `streamlit run`."""
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        return get_script_run_ctx() is not None
    except ImportError:
        return False


if __name__ == "__main__":
    if not _is_running_under_streamlit():
        # Bare-mode launch (`python streamlit_app.py`, PyCharm Run, etc.) —
        # re-exec ourselves under the proper Streamlit server.
        import sys

        from streamlit.web import cli as stcli

        sys.argv = ["streamlit", "run", __file__]
        sys.exit(stcli.main())
    main()

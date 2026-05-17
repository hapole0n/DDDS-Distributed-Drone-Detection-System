"""Live microphone capture + sliding-window classification.

Producer thread = sounddevice callback writes raw audio into a queue.
Consumer thread = drains the queue, maintains a ring buffer, and emits a
detection event every `live_hop_seconds`.

Sample-rate handling: most Windows USB microphones run at 44 100 Hz or
48 000 Hz, but the MFCC pipeline expects 22 050 Hz. This module captures
at the device's native rate and resamples each window down to the model's
training rate via `scipy.signal.resample_poly` before feature extraction.
"""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sounddevice as sd
from scipy.signal import resample_poly

from drone_detector.config import settings
from drone_detector.features.audio import mfcc_feature_vector
from drone_detector.models.baseline import ModelBundle


@dataclass
class DetectionEvent:
    timestamp: float
    label: str
    probability: float
    proba_vector: np.ndarray
    window: np.ndarray  # audio window at the MODEL sample rate (for UI)


def list_input_devices() -> list[dict]:
    return [
        {"index": i, **d}
        for i, d in enumerate(sd.query_devices())
        if d["max_input_channels"] > 0
    ]


def _resolve_capture_params(device: int | None, requested_sr: int) -> tuple[int, int]:
    """Return (capture_sample_rate, capture_channels) for the given device.

    Picks the device's reported `default_samplerate` (PortAudio's safest
    choice on Windows) and uses 1–2 input channels (mono if available,
    otherwise downmix from stereo).
    """
    info = sd.query_devices(device, "input")
    capture_sr = int(info.get("default_samplerate") or requested_sr)
    if capture_sr <= 0:
        capture_sr = requested_sr
    capture_ch = max(1, min(int(info.get("max_input_channels", 1)), 2))
    return capture_sr, capture_ch


def stream_detections(
    model: ModelBundle,
    device: int | None = None,
    sample_rate: int | None = None,
    window_seconds: float | None = None,
    hop_seconds: float | None = None,
    stop_event: threading.Event | None = None,
) -> Iterator[DetectionEvent]:
    """Yield a `DetectionEvent` every `hop_seconds` until `stop_event` is set.

    The audio is captured at the device's native sample rate and resampled
    to `sample_rate` (the model's training rate) before MFCC extraction.
    """
    target_sr = sample_rate or settings.sample_rate
    win_s = window_seconds or settings.live_window_seconds
    hop_s = hop_seconds or settings.live_hop_seconds

    capture_sr, capture_ch = _resolve_capture_params(device, target_sr)

    target_win_n = int(target_sr * win_s)
    capture_win_n = int(capture_sr * win_s)
    capture_hop_n = int(capture_sr * hop_s)
    blocksize = max(256, capture_hop_n // 4)

    audio_q: queue.Queue[np.ndarray] = queue.Queue()

    def callback(indata, frames, time_info, status):  # noqa: ARG001
        # status carries under-/over-flow events; not fatal — keep going.
        if indata.ndim == 2 and indata.shape[1] > 1:
            mono = indata.mean(axis=1)
        else:
            mono = indata[:, 0] if indata.ndim == 2 else indata
        audio_q.put(mono.astype(np.float32, copy=True))

    stop_event = stop_event or threading.Event()
    buffer = np.zeros(0, dtype=np.float32)
    last_emit = 0

    with sd.InputStream(
        samplerate=capture_sr,
        channels=capture_ch,
        dtype="float32",
        device=device,
        callback=callback,
        blocksize=blocksize,
    ):
        while not stop_event.is_set():
            try:
                chunk = audio_q.get(timeout=0.5)
            except queue.Empty:
                continue
            buffer = np.concatenate([buffer, chunk])
            while (
                len(buffer) >= capture_win_n
                and last_emit <= len(buffer) - capture_win_n
            ):
                capture_window = buffer[last_emit : last_emit + capture_win_n]
                # Resample to the model's training rate
                if capture_sr != target_sr:
                    window = resample_poly(
                        capture_window, target_sr, capture_sr
                    ).astype(np.float32)
                    if len(window) >= target_win_n:
                        window = window[:target_win_n]
                    else:
                        window = np.pad(window, (0, target_win_n - len(window)))
                else:
                    window = capture_window.astype(np.float32, copy=False)

                vec = mfcc_feature_vector(window, target_sr)[None, :]
                proba = model.predict_proba(vec)[0]
                cls = int(np.argmax(proba))
                yield DetectionEvent(
                    timestamp=time.time(),
                    label=model.labels[cls],
                    probability=float(proba[cls]),
                    proba_vector=proba.astype(np.float32),
                    window=window.copy(),
                )
                last_emit += capture_hop_n
            # Trim the ring buffer to keep memory bounded
            if last_emit > 4 * capture_win_n:
                buffer = buffer[last_emit:]
                last_emit = 0


def run_console_live(
    model_path: Path,
    device: int | None = None,
    duration_s: float = 0.0,
    on_event: Callable[[DetectionEvent], None] | None = None,
) -> None:
    """Convenience entry-point used by the CLI."""
    bundle = ModelBundle.load(model_path)
    stop_event = threading.Event()
    start = time.time()

    print(f"Listening on device={device}. Press Ctrl+C to stop.")
    try:
        for ev in stream_detections(bundle, device=device, stop_event=stop_event):
            line = (
                f"[{time.strftime('%H:%M:%S')}] "
                f"label={ev.label:<10}  "
                f"p={ev.probability:0.3f}  "
                f"probs={dict(zip(bundle.labels, ev.proba_vector.round(3)))}"
            )
            print(line)
            if on_event is not None:
                on_event(ev)
            if duration_s > 0 and (time.time() - start) >= duration_s:
                stop_event.set()
    except KeyboardInterrupt:
        stop_event.set()
        print("\nStopped.")

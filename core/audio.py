"""Shot detection from microphone audio.

A shot click is a short, sharp transient: very high amplitude over a very
short window. The detector tracks a rolling RMS baseline and fires when a
peak both exceeds an absolute threshold and is several times louder than
the baseline. This rejects steady-state noise (talking, fans) while
accepting genuine clicks.
"""

from __future__ import annotations

import collections
import threading
import time
from typing import Callable, Deque, Optional

import numpy as np

try:
    import sounddevice as sd
    SD_AVAILABLE = True
except ImportError:
    SD_AVAILABLE = False


WAVEFORM_SAMPLES = 300
"""Number of recent RMS values kept for the live waveform display."""


class AudioDetector:
    """Detect gunshot-like transients on a microphone input stream.

    Args:
        threshold: Absolute peak floor in [0, 1]. Quiet rooms can use 0.05;
            louder ones 0.15-0.3.
        transient_ratio: Peak must exceed the rolling baseline RMS by this
            factor before triggering. Higher values reject steadier sounds.
        cooldown_ms: Minimum time between successive triggers.
        sample_rate: Microphone sample rate in Hz.
        chunk_size: Samples per callback. ~12 ms at 44.1 kHz keeps latency
            low while still capturing the full transient.
        device_index: Input device index, or ``None`` for the system default.
        on_shot: Callback invoked with the trigger timestamp when a shot is
            detected.
    """

    def __init__(
        self,
        threshold: float = 0.15,
        transient_ratio: float = 6.0,
        cooldown_ms: int = 800,
        sample_rate: int = 44100,
        chunk_size: int = 512,
        device_index: Optional[int] = None,
        on_shot: Optional[Callable[[float], None]] = None,
    ):
        self.threshold = threshold
        self.transient_ratio = transient_ratio
        self.cooldown_s = cooldown_ms / 1000.0
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.device_index = device_index
        self.on_shot = on_shot

        self._stream = None
        self._last_trigger_time: float = 0.0
        self._running = False
        self._paused = False
        self._lock = threading.Lock()

        # Live display state, read by the UI thread.
        self.current_level: float = 0.0
        self.current_peak: float = 0.0
        self.current_baseline: float = 0.0
        self.last_trigger_level: float = 0.0
        self._waveform: Deque[float] = collections.deque(
            [0.0] * WAVEFORM_SAMPLES, maxlen=WAVEFORM_SAMPLES)

        # Rolling baseline: percentile over recent chunk RMSes.
        self._baseline_buf: Deque[float] = collections.deque(
            [0.001] * 40, maxlen=40)

    def start(self) -> None:
        """Open the input stream and begin processing audio."""
        if not SD_AVAILABLE:
            print("[Audio] sounddevice not available.")
            return
        if self._running:
            return
        self._running = True
        self._paused = False
        try:
            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                device=self.device_index,
                channels=1,
                blocksize=self.chunk_size,
                callback=self._audio_callback,
            )
            self._stream.start()
        except Exception as e:
            print(f"[Audio] Stream error: {e}")
            self._running = False

    def stop(self) -> None:
        """Stop and close the input stream."""
        self._running = False
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

    def pause(self, paused: bool) -> None:
        """Suspend or resume trigger detection without closing the stream."""
        with self._lock:
            self._paused = paused

    def set_threshold(self, value: float) -> None:
        self.threshold = max(0.005, min(1.0, value))

    def set_transient_ratio(self, value: float) -> None:
        self.transient_ratio = max(1.5, min(20.0, value))

    def set_cooldown(self, ms: int) -> None:
        self.cooldown_s = ms / 1000.0

    def get_waveform(self) -> list:
        """Snapshot of the most recent RMS history."""
        return list(self._waveform)

    @staticmethod
    def list_devices() -> list:
        """Available input devices as ``(index, name)`` tuples."""
        if not SD_AVAILABLE:
            return []
        try:
            return [
                (i, d["name"]) for i, d in enumerate(sd.query_devices())
                if d["max_input_channels"] > 0
            ]
        except Exception:
            return []

    def _audio_callback(self, indata, frames, time_info, status) -> None:
        if not any(indata):
            return

        audio = indata[:, 0].astype(np.float32)
        rms = float(np.sqrt(np.mean(audio ** 2)))
        peak = float(np.max(np.abs(audio)))

        self._waveform.append(rms)
        self._baseline_buf.append(rms)
        baseline = float(np.percentile(list(self._baseline_buf), 60))

        self.current_level = rms
        self.current_peak = peak
        self.current_baseline = baseline

        with self._lock:
            paused = self._paused
        if paused:
            return

        ratio = peak / max(baseline, 1e-6)
        if peak < self.threshold or ratio < self.transient_ratio:
            return

        now = time.time()
        if now - self._last_trigger_time < self.cooldown_s:
            return

        self._last_trigger_time = now
        self.last_trigger_level = peak
        if self.on_shot is None:
            return
        try:
            self.on_shot(now)
        except Exception as e:
            print(f"[Audio] callback error: {e}")

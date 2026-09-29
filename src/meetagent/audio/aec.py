"""WebRTC acoustic echo cancellation and speaker-reference buffering."""

from __future__ import annotations

import threading
from collections import deque

import numpy as np
from scipy.signal import resample_poly


class PlaybackReference:
    """Buffer speaker output as 16 kHz reference frames for the microphone AEC."""

    def __init__(self, max_seconds: float = 2.0) -> None:
        self._chunks: deque[np.ndarray] = deque()
        self._offset = 0
        self._samples = 0
        self._max_samples = int(16_000 * max_seconds)
        self._lock = threading.Lock()

    def write_24k(self, playback: np.ndarray) -> None:
        """Convert one 24 kHz output callback block to the 16 kHz reference rate."""
        if playback.size == 0:
            return
        reference = resample_poly(playback.astype(np.float32, copy=False), 2, 3)
        with self._lock:
            self._chunks.append(reference.astype(np.float32, copy=False))
            self._samples += reference.size
            while self._samples > self._max_samples and self._chunks:
                excess = self._samples - self._max_samples
                first = self._chunks[0]
                remaining = len(first) - self._offset
                if excess >= remaining:
                    self._chunks.popleft()
                    self._samples -= remaining
                    self._offset = 0
                else:
                    self._offset += excess
                    self._samples -= excess

    def read(self, frames: int) -> np.ndarray:
        """Return a same-length far-end frame, padding with silence on underrun."""
        reference = np.zeros(frames, dtype=np.float32)
        written = 0
        with self._lock:
            while written < frames and self._chunks:
                current = self._chunks[0]
                count = min(frames - written, len(current) - self._offset)
                reference[written : written + count] = current[self._offset : self._offset + count]
                written += count
                self._offset += count
                self._samples -= count
                if self._offset >= len(current):
                    self._chunks.popleft()
                    self._offset = 0
        return reference


class WebRtcEchoCanceller:
    """Remove playback echo using pywebrtc-audio's WebRTC AEC3 processor."""

    def __init__(self, sample_rate: int = 16_000, stream_delay_ms: int = 50) -> None:
        try:
            from pywebrtc_audio import AudioProcessor
        except (ImportError, OSError) as exc:
            raise RuntimeError(
                "未安装 WebRTC 回声消除依赖；请执行 `uv sync --extra aec`。"
            ) from exc
        self._processor = AudioProcessor(
            sample_rate=sample_rate,
            echo_cancellation=True,
            noise_suppression=False,
            auto_gain_control=False,
            stream_delay_ms=stream_delay_ms,
        )

    def process(self, microphone: np.ndarray, playback_reference: np.ndarray) -> np.ndarray:
        """Return cleaned float32 microphone samples; both inputs must have equal length."""
        if microphone.shape != playback_reference.shape:
            raise ValueError("AEC 麦克风帧和扬声器参考帧必须具有相同形状。")
        cleaned = self._processor.process(
            microphone.astype(np.float32, copy=False),
            playback_reference.astype(np.float32, copy=False),
        )
        return np.asarray(cleaned, dtype=np.float32)

    def reset(self) -> None:
        """Clear adaptive AEC state, for example before a new meeting."""
        self._processor.reset()

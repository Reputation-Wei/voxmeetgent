"""Voice activity detection and speech segment assembly."""

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class VADBackend(Protocol):
    """Mark audio frames that contain speech."""

    def is_speech(self, pcm: bytes, sample_rate: int) -> bool:
        """Return whether a frame contains speech."""
        raise NotImplementedError


@dataclass(frozen=True)
class SpeechSegment:
    """A single detected speech region in a mono 16 kHz signal."""

    start_seconds: float
    audio: np.ndarray


class SileroSpeechSegmenter:
    """Split 16 kHz float audio into speech segments using Silero VAD."""

    def __init__(self, threshold: float = 0.5, min_silence_duration_ms: int = 500) -> None:
        try:
            import torch
            from silero_vad import VADIterator, load_silero_vad
        except ImportError as exc:
            raise RuntimeError("缺少 silero-vad/onnxruntime，请在仓库目录执行 `uv sync`。") from exc
        self._torch = torch
        model = load_silero_vad(onnx=True)
        self._iterator = VADIterator(
            model, sampling_rate=16_000, threshold=threshold,
            min_silence_duration_ms=min_silence_duration_ms,
        )

    def segment(self, audio: np.ndarray, chunk_samples: int = 512) -> list[SpeechSegment]:
        """Feed the signal in Silero-sized frames and return detected utterances."""
        signal = np.asarray(audio, dtype=np.float32).reshape(-1)
        self._iterator.reset_states()
        pending: list[np.ndarray] = []
        start: float | None = None
        segments: list[SpeechSegment] = []
        for offset in range(0, len(signal), chunk_samples):
            frame = signal[offset : offset + chunk_samples]
            if len(frame) != chunk_samples:
                frame = np.pad(frame, (0, chunk_samples - len(frame)))
            event = self.process_frame(frame)
            if event and "start" in event:
                start = max(0.0, float(event["start"]))
                pending = [frame]
            elif start is not None:
                pending.append(frame)
            if event and "end" in event and start is not None:
                combined = np.concatenate(pending)
                segments.append(SpeechSegment(start, combined))
                pending = []
                start = None
        if start is not None and pending:
            segments.append(SpeechSegment(start, np.concatenate(pending)))
        return segments

    def process_frame(self, frame: np.ndarray) -> dict[str, float] | None:
        """Feed one 512-sample microphone frame into the streaming VAD."""
        samples = np.asarray(frame, dtype=np.float32).reshape(-1)
        if len(samples) != 512:
            raise ValueError("Silero VAD 的实时帧必须正好包含 512 个采样点。")
        event = self._iterator(self._torch.from_numpy(samples), return_seconds=True)
        if event is None:
            return None
        return {key: float(value) for key, value in event.items()}

    def reset(self) -> None:
        """Clear model state before starting a new audio stream."""
        self._iterator.reset_states()

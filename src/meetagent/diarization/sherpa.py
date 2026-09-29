"""Local speaker identification and enrollment using sherpa-onnx embeddings."""

from __future__ import annotations

import json
import os
import tempfile
import urllib.request
import wave
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

MODEL_NAME = "3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"
MODEL_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    f"speaker-recongition-models/{MODEL_NAME}"
)
REGISTRY_PATH = Path.home() / ".local" / "share" / "meetagent" / "speakers.json"


def _embedding_model_path() -> Path:
    configured = os.environ.get("MEETAGENT_SPEAKER_MODEL")
    if configured:
        path = Path(configured).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"找不到说话人 embedding 模型：{path}")
        return path

    cache_dir = Path.home() / ".cache" / "meetagent" / "models"
    model_path = cache_dir / MODEL_NAME
    if model_path.is_file():
        return model_path
    cache_dir.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=cache_dir, prefix="speaker-model-", delete=False) as tmp:
            temporary_path = Path(tmp.name)
        urllib.request.urlretrieve(MODEL_URL, temporary_path)
        temporary_path.replace(model_path)
    except Exception:
        if temporary_path:
            temporary_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"下载本地说话人模型失败。可手动下载 {MODEL_URL}，并设置 MEETAGENT_SPEAKER_MODEL。"
        ) from None
    return model_path


def _load_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as wav:
        channels, sample_rate, width = wav.getnchannels(), wav.getframerate(), wav.getsampwidth()
        if width != 2:
            raise ValueError("说话人注册 WAV 必须是 16-bit PCM。")
        samples = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2")
    audio = samples.astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    if sample_rate != 16_000:
        audio = resample_poly(audio, 16_000, sample_rate).astype(np.float32)
    return np.ascontiguousarray(audio)


class SherpaSpeakerIdentifier:
    """Identify known enrollees and cluster unknown speakers within one meeting."""

    def __init__(self, registry_path: Path = REGISTRY_PATH, threshold: float = 0.68) -> None:
        try:
            import sherpa_onnx
        except ImportError as exc:
            raise RuntimeError("缺少 sherpa-onnx，请在仓库目录执行 `uv sync`。") from exc
        self._sherpa = sherpa_onnx
        self._model_path = _embedding_model_path()
        config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=str(self._model_path), num_threads=max(1, min(4, os.cpu_count() or 1)),
            provider="cpu",
        )
        if not config.validate():
            raise RuntimeError(f"无效的说话人 embedding 配置：{config}")
        self._extractor = sherpa_onnx.SpeakerEmbeddingExtractor(config)
        self.registry_path = registry_path
        self.threshold = threshold
        self._known: dict[str, list[float]] = self._read_registry()
        self._session: dict[str, list[float]] = {}
        self._pending: list[dict[str, object]] = []
        self._confirmed_audio: dict[str, np.ndarray] = {}
        self._confirmed_audio_start: dict[str, float] = {}
        self._manager = sherpa_onnx.SpeakerEmbeddingManager(self._extractor.dim)
        self._rebuild_manager()

    def _read_registry(self) -> dict[str, list[float]]:
        if not self.registry_path.exists():
            return {}
        data = json.loads(self.registry_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"说话人注册表格式无效：{self.registry_path}")
        registry: dict[str, list[float]] = {}
        for name, value in data.items():
            # Read both the original single-vector format and the multi-sample format.
            if value and isinstance(value[0], (list, tuple)):
                samples = np.asarray(value, dtype=np.float32)
                center = np.mean(samples, axis=0)
                norm = float(np.linalg.norm(center))
                if norm:
                    center /= norm
                registry[str(name)] = center.tolist()
            else:
                registry[str(name)] = [float(item) for item in value]
        return registry

    def _rebuild_manager(self) -> None:
        self._manager = self._sherpa.SpeakerEmbeddingManager(self._extractor.dim)
        for name, vector in {**self._known, **self._session}.items():
            if not self._manager.add(name, np.asarray(vector, dtype=np.float32)):
                raise RuntimeError(f"无法载入说话人声纹：{name}")

    def _compute_embedding(self, audio: np.ndarray) -> np.ndarray:
        stream = self._extractor.create_stream()
        stream.accept_waveform(sample_rate=16_000, waveform=np.ascontiguousarray(audio, dtype=np.float32))
        stream.input_finished()
        if not self._extractor.is_ready(stream):
            raise ValueError("录音太短，无法提取说话人声纹；请提供至少约 2 秒清晰语音。")
        return np.asarray(self._extractor.compute(stream), dtype=np.float32)

    def enroll(self, display_name: str, wav_path: Path) -> None:
        """Add one clean single-speaker sample to a speaker's voice profile."""
        name = display_name.strip()
        if not name:
            raise ValueError("说话人姓名不能为空。")
        audio = _load_wav(wav_path.expanduser())
        if len(audio) < 32_000:
            raise ValueError("注册样本至少需要 2 秒音频；建议录制 5–10 秒清晰单人语音。")
        embedding = self._compute_embedding(audio)
        existing = np.asarray(self._known[name], dtype=np.float32) if name in self._known else None
        if existing is not None:
            if float(np.dot(embedding, existing)) < 0.60:
                raise ValueError(f"新样本与“{name}”已有声纹差异过大；为避免污染档案，未保存。")
            embedding = existing + embedding
            norm = float(np.linalg.norm(embedding))
            if norm:
                embedding /= norm
        self._known[name] = embedding.tolist()
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.registry_path.with_suffix(".tmp")
        temp_path.write_text(json.dumps(self._known, ensure_ascii=False, indent=2), encoding="utf-8")
        temp_path.replace(self.registry_path)
        self._rebuild_manager()

    def identify(self, audio: np.ndarray, start_seconds: float | None = None) -> str | None:
        """Identify a speaker, returning unknown until an unregistered voice repeats."""
        signal = np.asarray(audio, dtype=np.float32).reshape(-1)
        if len(signal) < 32_000:
            return None
        rms = float(np.sqrt(np.mean(signal * signal)))
        if rms < 0.01:
            return None
        try:
            embedding = self._compute_embedding(signal)
        except ValueError:
            return None
        match = self._manager.search(embedding, threshold=self.threshold)
        if match:
            return str(match)

        # A single unmatched utterance is too weak to establish a new identity.
        # Confirm a candidate only after two clear utterances agree.
        for candidate in self._pending:
            samples = candidate["samples"]
            assert isinstance(samples, list)
            center = np.mean(samples, axis=0)
            center_norm = float(np.linalg.norm(center))
            if center_norm:
                center /= center_norm
            similarity = float(np.dot(embedding, center))
            if similarity >= 0.72:
                samples.append(embedding)
                if len(samples) < 2:
                    return None
                confirmed = np.mean(samples, axis=0)
                norm = float(np.linalg.norm(confirmed))
                if norm > 0:
                    confirmed /= norm
                number = 1
                while f"Speaker {number}" in self._known or f"Speaker {number}" in self._session:
                    number += 1
                name = f"Speaker {number}"
                self._session[name] = confirmed.tolist()
                self._pending.remove(candidate)
                if not self._manager.add(name, confirmed):
                    return None
                candidate_audio = candidate["audio"]
                candidate_starts = candidate["starts"]
                if isinstance(candidate_audio, list) and isinstance(candidate_starts, list):
                    self._confirmed_audio[name] = candidate_audio[0]
                    if candidate_starts:
                        self._confirmed_audio_start[name] = float(candidate_starts[0])
                return name
        if len(self._pending) < 16:
            self._pending.append({
                "samples": [embedding], "audio": [signal.copy()],
                "starts": [start_seconds if start_seconds is not None else -1.0],
            })
        elif not any(
            float(np.dot(np.asarray(item["samples"][-1]), embedding)) >= 0.72
            for item in self._pending
            if isinstance(item.get("samples"), list)
        ):
            self._pending.pop(0)
            self._pending.append({
                "samples": [embedding], "audio": [signal.copy()],
                "starts": [start_seconds if start_seconds is not None else -1.0],
            })
        return None

    def take_confirmed_audio(self, speaker: str) -> tuple[np.ndarray, float] | None:
        """Return the first matching segment and its start time for transcript correction."""
        audio = self._confirmed_audio.pop(speaker, None)
        start = self._confirmed_audio_start.pop(speaker, None)
        if audio is None or start is None or start < 0:
            return None
        return audio, start


def enroll_speaker(display_name: str, wav_path: Path) -> Path:
    """Create or update the persistent voice registry with a single enrollment."""
    identifier = SherpaSpeakerIdentifier()
    identifier.enroll(display_name, wav_path)
    return identifier.registry_path

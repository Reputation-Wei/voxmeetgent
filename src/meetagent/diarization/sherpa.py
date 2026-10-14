"""Local speaker identification and enrollment using sherpa-onnx embeddings."""

from __future__ import annotations

import json
import os
import tempfile
from typing import Any
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

MIN_AUDIO_SECONDS = 0.8
MIN_IDENTIFY_SAMPLES = int(16_000 * MIN_AUDIO_SECONDS)  # 12,800 samples = 0.8s
MIN_IDENTIFY_RMS = 0.003
DEFAULT_SEARCH_THRESHOLD = 0.62
DEFAULT_PENDING_THRESHOLD = 0.65
DEFAULT_MERGE_THRESHOLD = 0.58
EMA_ALPHA = 0.85


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default



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

    def __init__(
        self,
        registry_path: Path = REGISTRY_PATH,
        threshold: float | None = None,
        pending_threshold: float | None = None,
        merge_threshold: float | None = None,
        min_samples: int = MIN_IDENTIFY_SAMPLES,
        min_rms: float = MIN_IDENTIFY_RMS,
    ) -> None:
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
        self.threshold = threshold if threshold is not None else _float_env("MEETAGENT_SPEAKER_THRESHOLD", DEFAULT_SEARCH_THRESHOLD)
        self.pending_threshold = pending_threshold if pending_threshold is not None else _float_env("MEETAGENT_SPEAKER_PENDING", DEFAULT_PENDING_THRESHOLD)
        self.merge_threshold = merge_threshold if merge_threshold is not None else _float_env("MEETAGENT_SPEAKER_MERGE", DEFAULT_MERGE_THRESHOLD)
        self.min_samples = min_samples
        self.min_rms = min_rms

        self._known: dict[str, list[float]] = self._read_registry()
        self._session: dict[str, list[float]] = {}
        self._pending: list[dict[str, object]] = []
        self._confirmed_audio: dict[str, np.ndarray] = {}
        self._confirmed_audio_start: dict[str, float] = {}
        self._utterances: list[dict[str, Any]] = []
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
            raise ValueError("录音太短，无法提取说话人声纹；请提供至少约 0.8 秒清晰语音。")
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
        """Identify a speaker, updating session centroids and merging near-duplicate speakers."""
        signal = np.asarray(audio, dtype=np.float32).reshape(-1)
        if len(signal) < self.min_samples:
            return None
        rms = float(np.sqrt(np.mean(signal * signal))) if signal.size else 0.0
        if rms < self.min_rms:
            return None
        try:
            embedding = self._compute_embedding(signal)
            norm = float(np.linalg.norm(embedding))
            if norm:
                embedding /= norm
        except ValueError:
            return None

        # 1. Search existing known & session speakers
        match = self._manager.search(embedding, threshold=self.threshold)
        if match:
            name = str(match)
            # Update EMA centroid for active session speaker
            if name in self._session:
                old_vec = np.asarray(self._session[name], dtype=np.float32)
                new_vec = EMA_ALPHA * old_vec + (1.0 - EMA_ALPHA) * embedding
                new_norm = float(np.linalg.norm(new_vec))
                if new_norm:
                    new_vec /= new_norm
                self._session[name] = new_vec.tolist()
                self._manager.remove(name)
                self._manager.add(name, new_vec)
            if start_seconds is not None:
                self._utterances.append({
                    "start_seconds": start_seconds,
                    "embedding": embedding,
                    "speaker": name,
                })
            return name

        # 2. Check candidate pool in _pending
        best_cand: dict[str, Any] | None = None
        best_cand_sim = -1.0
        for candidate in self._pending:
            samples = candidate["samples"]
            assert isinstance(samples, list)
            center = np.mean(samples, axis=0)
            center_norm = float(np.linalg.norm(center))
            if center_norm:
                center /= center_norm
            similarity = float(np.dot(embedding, center))
            if similarity >= self.pending_threshold and similarity > best_cand_sim:
                best_cand_sim = similarity
                best_cand = candidate

        if best_cand is not None:
            samples = best_cand["samples"]
            assert isinstance(samples, list)
            samples.append(embedding)
            if len(samples) < 2:
                if start_seconds is not None:
                    self._utterances.append({
                        "start_seconds": start_seconds,
                        "embedding": embedding,
                        "speaker": None,
                    })
                return None

            confirmed = np.mean(samples, axis=0)
            norm = float(np.linalg.norm(confirmed))
            if norm > 0:
                confirmed /= norm

            # Merge Check: Does this confirmed candidate match an existing speaker?
            best_existing_speaker: str | None = None
            best_existing_sim = -1.0
            all_speakers = {**self._known, **self._session}
            for s_name, s_vec in all_speakers.items():
                sim = float(np.dot(confirmed, np.asarray(s_vec, dtype=np.float32)))
                if sim > best_existing_sim:
                    best_existing_sim = sim
                    best_existing_speaker = s_name

            if best_existing_speaker is not None and best_existing_sim >= self.merge_threshold:
                target_name = best_existing_speaker
                if target_name in self._session:
                    old_vec = np.asarray(self._session[target_name], dtype=np.float32)
                    merged_vec = 0.7 * old_vec + 0.3 * confirmed
                    m_norm = float(np.linalg.norm(merged_vec))
                    if m_norm:
                        merged_vec /= m_norm
                    self._session[target_name] = merged_vec.tolist()
                    self._manager.remove(target_name)
                    self._manager.add(target_name, merged_vec)
                self._pending.remove(best_cand)
                cand_audio = best_cand.get("audio")
                cand_starts = best_cand.get("starts")
                if isinstance(cand_audio, list) and isinstance(cand_starts, list):
                    self._confirmed_audio[target_name] = cand_audio[0]
                    if cand_starts:
                        self._confirmed_audio_start[target_name] = float(cand_starts[0])
                if start_seconds is not None:
                    self._utterances.append({
                        "start_seconds": start_seconds,
                        "embedding": embedding,
                        "speaker": target_name,
                    })
                return target_name

            # Genuinely distinct new speaker
            number = 1
            while f"Speaker {number}" in self._known or f"Speaker {number}" in self._session:
                number += 1
            name = f"Speaker {number}"
            self._session[name] = confirmed.tolist()
            self._pending.remove(best_cand)
            if not self._manager.add(name, confirmed):
                return None
            cand_audio = best_cand.get("audio")
            cand_starts = best_cand.get("starts")
            if isinstance(cand_audio, list) and isinstance(cand_starts, list):
                self._confirmed_audio[name] = cand_audio[0]
                if cand_starts:
                    self._confirmed_audio_start[name] = float(cand_starts[0])
            if start_seconds is not None:
                self._utterances.append({
                    "start_seconds": start_seconds,
                    "embedding": embedding,
                    "speaker": name,
                })
            return name

        # 3. Add to pending
        if len(self._pending) < 16:
            self._pending.append({
                "samples": [embedding], "audio": [signal.copy()],
                "starts": [start_seconds if start_seconds is not None else -1.0],
            })
        elif not any(
            float(np.dot(np.asarray(item["samples"][-1]), embedding)) >= self.pending_threshold
            for item in self._pending
            if isinstance(item.get("samples"), list)
        ):
            self._pending.pop(0)
            self._pending.append({
                "samples": [embedding], "audio": [signal.copy()],
                "starts": [start_seconds if start_seconds is not None else -1.0],
            })
        if start_seconds is not None:
            self._utterances.append({
                "start_seconds": start_seconds,
                "embedding": embedding,
                "speaker": None,
            })
        return None

    def recluster_session(
        self, entries: list[dict[str, Any]], meeting_started: float = 0.0,
    ) -> dict[str, Any]:
        """Perform a global offline re-clustering pass over meeting transcript entries.

        1. Merges overlapping/close session speaker clusters.
        2. Resolves '说话人未知' entries using recorded embeddings and context continuity.
        3. Renames surviving session speakers sequentially.
        """
        if not entries:
            return {"merged_speakers": 0, "resolved_unknowns": 0, "final_speakers": []}

        # 1. Merge session speakers with cosine similarity >= merge_threshold
        session_speakers = [s for s in self._session.keys() if s not in self._known]
        parent = {s: s for s in session_speakers}

        def find(item: str) -> str:
            if parent[item] == item:
                return item
            parent[item] = find(parent[item])
            return parent[item]

        def union(i: str, j: str) -> None:
            root_i, root_j = find(i), find(j)
            if root_i != root_j:
                def get_num(s: str) -> int:
                    parts = s.split()
                    return int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 9999
                if get_num(root_i) <= get_num(root_j):
                    parent[root_j] = root_i
                else:
                    parent[root_i] = root_j

        for i, s1 in enumerate(session_speakers):
            v1 = np.asarray(self._session[s1], dtype=np.float32)
            for s2 in session_speakers[i + 1:]:
                v2 = np.asarray(self._session[s2], dtype=np.float32)
                sim = float(np.dot(v1, v2))
                if sim >= self.merge_threshold:
                    union(s1, s2)

        speaker_map = {s: find(s) for s in session_speakers}
        merged_count = sum(1 for old_s, new_s in speaker_map.items() if old_s != new_s)

        merged_centroids: dict[str, np.ndarray] = {}
        for old_s, target in speaker_map.items():
            vec = np.asarray(self._session[old_s], dtype=np.float32)
            if target not in merged_centroids:
                merged_centroids[target] = vec.copy()
            else:
                merged_centroids[target] += vec
        for target, vec in merged_centroids.items():
            norm = float(np.linalg.norm(vec))
            if norm:
                merged_centroids[target] = vec / norm

        for entry in entries:
            current_spk = entry.get("speaker")
            if current_spk in speaker_map and speaker_map[current_spk] != current_spk:
                entry["speaker"] = speaker_map[current_spk]
                entry["speaker_confidence"] = "session_merged"

        # 2. Resolve unknown speakers
        resolved_unknowns = 0
        all_centroids = {
            **{k: np.asarray(v, dtype=np.float32) for k, v in self._known.items()},
            **merged_centroids,
        }

        if all_centroids and self._utterances:
            for entry in entries:
                if entry.get("speaker") == "说话人未知":
                    entry_time = float(entry.get("timestamp", 0.0))
                    best_match_utt = None
                    min_dt = 0.75
                    for utt in self._utterances:
                        dt = abs(entry_time - (meeting_started + utt["start_seconds"]))
                        if dt < min_dt:
                            min_dt = dt
                            best_match_utt = utt

                    if best_match_utt is not None:
                        emb = best_match_utt["embedding"]
                        best_spk: str | None = None
                        best_sim = -1.0
                        for s_name, s_vec in all_centroids.items():
                            sim = float(np.dot(emb, s_vec))
                            if sim > best_sim:
                                best_sim = sim
                                best_spk = s_name
                        if best_spk is not None and best_sim >= 0.55:
                            entry["speaker"] = best_spk
                            entry["speaker_confidence"] = "post_meeting_recluster"
                            resolved_unknowns += 1

        # 3. Context continuity fallback for remaining unknowns
        for i, entry in enumerate(entries):
            if entry.get("speaker") == "说话人未知":
                if i > 0:
                    prev = entries[i - 1]
                    prev_spk = prev.get("speaker")
                    if prev_spk and prev_spk not in ("说话人未知", "Agent"):
                        dt = float(entry.get("timestamp", 0.0)) - float(prev.get("timestamp", 0.0))
                        if 0.0 <= dt <= 1.2:
                            entry["speaker"] = prev_spk
                            entry["speaker_confidence"] = "context_continuity"
                            resolved_unknowns += 1

        # 4. Renumber surviving session speakers sequentially
        seen_speakers: list[str] = []
        for entry in entries:
            spk = entry.get("speaker")
            if spk and spk.startswith("Speaker ") and spk not in seen_speakers:
                seen_speakers.append(spk)

        renumber_map = {
            old_name: f"Speaker {idx + 1}"
            for idx, old_name in enumerate(seen_speakers)
        }
        for entry in entries:
            spk = entry.get("speaker")
            if spk in renumber_map:
                entry["speaker"] = renumber_map[spk]

        final_speakers = sorted(list({entry.get("speaker") for entry in entries if entry.get("speaker")}))
        return {
            "merged_speakers": merged_count,
            "resolved_unknowns": resolved_unknowns,
            "final_speakers": final_speakers,
        }

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

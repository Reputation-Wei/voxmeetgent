"""Tests for Phase 8 P3: refactored speaker identification, online EMA, and post-meeting clustering."""

from pathlib import Path
import json
import numpy as np
import pytest

from meetagent.diarization.sherpa import (
    SherpaSpeakerIdentifier,
    MIN_IDENTIFY_SAMPLES,
    DEFAULT_SEARCH_THRESHOLD,
    DEFAULT_PENDING_THRESHOLD,
    DEFAULT_MERGE_THRESHOLD,
)
from meetagent.storage.jsonl import JsonlTranscriptStore


def _make_unit_vector(dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(dim).astype(np.float32)
    return vec / np.linalg.norm(vec)


def test_default_thresholds_and_environment_overrides(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    registry = tmp_path / "speakers.json"
    ident = SherpaSpeakerIdentifier(registry_path=registry)
    assert ident.min_samples == MIN_IDENTIFY_SAMPLES
    assert ident.threshold == DEFAULT_SEARCH_THRESHOLD
    assert ident.pending_threshold == DEFAULT_PENDING_THRESHOLD
    assert ident.merge_threshold == DEFAULT_MERGE_THRESHOLD

    monkeypatch.setenv("MEETAGENT_SPEAKER_THRESHOLD", "0.75")
    monkeypatch.setenv("MEETAGENT_SPEAKER_PENDING", "0.80")
    monkeypatch.setenv("MEETAGENT_SPEAKER_MERGE", "0.60")
    ident_custom = SherpaSpeakerIdentifier(registry_path=registry)
    assert ident_custom.threshold == 0.75
    assert ident_custom.pending_threshold == 0.80
    assert ident_custom.merge_threshold == 0.60


def test_short_and_quiet_audio_rejected(tmp_path: Path) -> None:
    registry = tmp_path / "speakers.json"
    ident = SherpaSpeakerIdentifier(registry_path=registry)
    # Audio shorter than 0.8s (12,800 samples)
    short_audio = np.full(10_000, 0.1, dtype=np.float32)
    assert ident.identify(short_audio) is None

    # Audio with very low RMS (< 0.003)
    quiet_audio = np.full(16_000, 0.0001, dtype=np.float32)
    assert ident.identify(quiet_audio) is None


def test_online_identify_and_ema_centroid_update(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    registry = tmp_path / "speakers.json"
    ident = SherpaSpeakerIdentifier(registry_path=registry)
    dim = ident._extractor.dim

    v1 = _make_unit_vector(dim, 1)
    # v2 is very close to v1 (similarity ~0.95)
    noise = _make_unit_vector(dim, 2)
    v2 = v1 + 0.1 * noise
    v2 /= np.linalg.norm(v2)

    # v3 is slightly shifted from v1 (similarity ~0.75)
    v3 = v1 + 0.5 * noise
    v3 /= np.linalg.norm(v3)

    embeddings = [v1, v2, v3]
    call_idx = 0

    def mock_compute(audio: np.ndarray) -> np.ndarray:
        nonlocal call_idx
        emb = embeddings[min(call_idx, len(embeddings) - 1)]
        call_idx += 1
        return emb

    monkeypatch.setattr(ident, "_compute_embedding", mock_compute)
    dummy_audio = np.full(16_000, 0.1, dtype=np.float32)

    # First utterance goes to pending -> None
    r1 = ident.identify(dummy_audio, start_seconds=0.0)
    assert r1 is None

    # Second utterance matches pending -> creates Speaker 1
    r2 = ident.identify(dummy_audio, start_seconds=2.0)
    assert r2 == "Speaker 1"
    first_pending = ident.take_confirmed_audio("Speaker 1")
    assert first_pending is not None
    assert abs(first_pending[1] - 0.0) < 1e-4

    # Third utterance matches Speaker 1 directly -> returns Speaker 1 and adapts centroid via EMA
    r3 = ident.identify(dummy_audio, start_seconds=4.0)
    assert r3 == "Speaker 1"
    assert "Speaker 1" in ident._session


def test_candidate_merge_with_existing_speaker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    registry = tmp_path / "speakers.json"
    ident = SherpaSpeakerIdentifier(registry_path=registry, merge_threshold=0.58)
    dim = ident._extractor.dim

    base = _make_unit_vector(dim, 10)
    ident._session["Speaker 1"] = base.tolist()
    ident._manager.add("Speaker 1", base)

    # Candidate vectors that are just below direct search threshold (0.62)
    # but above merge_threshold (0.58), e.g. similarity ~0.59.
    noise = _make_unit_vector(dim, 11)
    noise -= np.dot(noise, base) * base
    noise /= np.linalg.norm(noise)
    cand_vec = 0.59 * base + np.sqrt(1 - 0.59**2) * noise
    cand_vec /= np.linalg.norm(cand_vec)


    embeddings = [cand_vec, cand_vec]
    call_idx = 0

    def mock_compute(audio: np.ndarray) -> np.ndarray:
        nonlocal call_idx
        emb = embeddings[min(call_idx, len(embeddings) - 1)]
        call_idx += 1
        return emb

    monkeypatch.setattr(ident, "_compute_embedding", mock_compute)
    dummy_audio = np.full(16_000, 0.1, dtype=np.float32)

    # First cand utterance goes to pending
    r1 = ident.identify(dummy_audio, start_seconds=10.0)
    assert r1 is None

    # Second cand utterance confirms candidate, passes merge check, merges into Speaker 1!
    r2 = ident.identify(dummy_audio, start_seconds=12.0)
    assert r2 == "Speaker 1"
    assert "Speaker 2" not in ident._session


def test_candidate_distinct_speaker_creates_speaker_2(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    registry = tmp_path / "speakers.json"
    ident = SherpaSpeakerIdentifier(registry_path=registry, merge_threshold=0.58)
    dim = ident._extractor.dim

    spk1_vec = _make_unit_vector(dim, 20)
    ident._session["Speaker 1"] = spk1_vec.tolist()
    ident._manager.add("Speaker 1", spk1_vec)

    # A completely orthogonal speaker (similarity ~0.0 < 0.58)
    spk2_vec = _make_unit_vector(dim, 21)
    embeddings = [spk2_vec, spk2_vec]
    call_idx = 0

    def mock_compute(audio: np.ndarray) -> np.ndarray:
        nonlocal call_idx
        emb = embeddings[min(call_idx, len(embeddings) - 1)]
        call_idx += 1
        return emb

    monkeypatch.setattr(ident, "_compute_embedding", mock_compute)
    dummy_audio = np.full(16_000, 0.1, dtype=np.float32)

    r1 = ident.identify(dummy_audio, start_seconds=10.0)
    assert r1 is None
    r2 = ident.identify(dummy_audio, start_seconds=12.0)
    assert r2 == "Speaker 2"


def test_recluster_session_merges_and_resolves_unknowns(tmp_path: Path) -> None:
    registry = tmp_path / "speakers.json"
    ident = SherpaSpeakerIdentifier(registry_path=registry, merge_threshold=0.58)
    dim = ident._extractor.dim

    v_base = _make_unit_vector(dim, 30)
    noise = _make_unit_vector(dim, 31)
    v_near = 0.70 * v_base + np.sqrt(1 - 0.70**2) * noise
    v_near /= np.linalg.norm(v_near)

    # Simulate an over-split session with Speaker 1 and Speaker 2 that are actually close (sim 0.70)
    ident._session["Speaker 1"] = v_base.tolist()
    ident._session["Speaker 2"] = v_near.tolist()

    # Utterance recorded for an unknown speech segment at start_seconds=5.0
    ident._utterances.append({
        "start_seconds": 5.0,
        "embedding": v_base,
        "speaker": None,
    })

    entries = [
        {"timestamp": 105.0, "time": "00:05", "speaker": "说话人未知", "text": "起初没被认出"},
        {"timestamp": 110.0, "time": "00:10", "speaker": "Speaker 1", "text": "我是第一位"},
        {"timestamp": 120.0, "time": "00:20", "speaker": "Speaker 2", "text": "我也是同一个人"},
        {"timestamp": 121.0, "time": "00:21", "speaker": "说话人未知", "text": "对"}, # 1s gap -> context continuity
    ]

    summary = ident.recluster_session(entries, meeting_started=100.0)
    assert summary["merged_speakers"] >= 1
    assert summary["resolved_unknowns"] == 2

    # All entries should now be unified under Speaker 1!
    speakers = [e["speaker"] for e in entries]
    assert speakers == ["Speaker 1", "Speaker 1", "Speaker 1", "Speaker 1"]


def test_jsonl_rename_speaker(tmp_path: Path) -> None:
    txt = tmp_path / "meeting.txt"
    jsonl = tmp_path / "meeting.jsonl"
    entries = [
        {"timestamp": 1.0, "time": "00:01", "speaker": "Speaker 1", "text": "你好", "language": "zh"},
        {"timestamp": 2.0, "time": "00:02", "speaker": "Speaker 2", "text": "世界", "language": "zh"},
        {"timestamp": 3.0, "time": "00:03", "speaker": "Speaker 1", "text": "再见", "language": "zh"},
    ]
    store = JsonlTranscriptStore(txt)
    store.rewrite(entries)

    # Rename all "Speaker 1" to "张三"
    count = store.rename_speaker("Speaker 1", "张三")
    assert count == 2

    lines = [json.loads(line) for line in jsonl.read_text(encoding="utf-8").splitlines()]
    assert lines[0]["speaker"] == "张三"
    assert lines[1]["speaker"] == "Speaker 2"
    assert lines[2]["speaker"] == "张三"

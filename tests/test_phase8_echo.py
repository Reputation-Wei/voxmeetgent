"""Phase 8 P2: barge-in under speaker echo and echo calibration analysis."""

import numpy as np

from meetagent import demo
from meetagent.audio.calibrate import SAMPLE_RATE, analyze, set_toml_value, speech_like_signal


def test_short_echo_burst_does_not_barge_in() -> None:
    detector = demo._BargeInDetector(threshold=0.05, hold_seconds=0.2, quiet_threshold=0.02)
    # 0.1 s of loud echo at playback onset, then quiet: no interruption.
    events = [detector.update(0.13, t / 100, agent_speaking=True) for t in range(10)]
    events += [detector.update(0.01, 0.1 + t / 100, agent_speaking=True) for t in range(30)]
    assert "barge_in" not in events


def test_sustained_speech_barges_in_and_rearms() -> None:
    detector = demo._BargeInDetector(threshold=0.05, hold_seconds=0.2, quiet_threshold=0.02)
    events = [detector.update(0.2, t / 100, agent_speaking=True) for t in range(30)]
    assert events.count("barge_in") == 1 and detector.latched
    events = [detector.update(0.005, 0.3 + t / 100, agent_speaking=False) for t in range(40)]
    assert events.count("rearm") == 1 and not detector.latched


def test_no_barge_in_while_agent_is_silent() -> None:
    detector = demo._BargeInDetector(threshold=0.05, hold_seconds=0.2, quiet_threshold=0.02)
    assert all(detector.update(0.3, t / 100, agent_speaking=False) is None for t in range(50))


def test_analyze_finds_delay_and_recommends_threshold_above_echo() -> None:
    rng = np.random.default_rng(1)
    played = speech_like_signal(2.0, rms=0.1)
    delay = int(0.06 * SAMPLE_RATE)  # 60 ms speaker -> mic latency
    recorded = np.zeros(len(played) + SAMPLE_RATE // 2, dtype=np.float32)
    recorded[delay:delay + len(played)] += 0.8 * played
    recorded += 0.003 * rng.standard_normal(len(recorded)).astype(np.float32)
    noise = 0.003 * rng.standard_normal(SAMPLE_RATE).astype(np.float32)
    result = analyze(played, recorded, noise, candidate_delays=range(0, 121, 20))
    assert abs(result.measured_delay_ms - 60) <= 2
    assert result.recommended_barge_in >= 0.02
    if result.residual_echo_p95 is not None:  # AEC extra installed
        assert result.residual_echo_p95 < result.raw_echo_rms


def test_set_toml_value_updates_or_adds() -> None:
    text = '[audio]\nmodel_path = "x.bin"\naec_delay_ms = 20\n\n[live]\nreply = "text"\n'
    text = set_toml_value(text, "audio", "aec_delay_ms", "60")
    text = set_toml_value(text, "live", "barge_in_threshold", "0.08")
    text = set_toml_value(text, "agent", "followup_seconds", "30")
    assert 'aec_delay_ms = 60' in text and 'aec_delay_ms = 20' not in text
    assert 'reply = "text"\nbarge_in_threshold = 0.08' in text
    assert text.rstrip().endswith("[agent]\nfollowup_seconds = 30")
    import tomllib
    parsed = tomllib.loads(text)
    assert parsed["audio"]["model_path"] == "x.bin"


def test_noise_glitch_does_not_inflate_threshold() -> None:
    played = speech_like_signal(1.0, rms=0.1)
    recorded = np.concatenate([0.2 * played, np.zeros(SAMPLE_RATE // 4, dtype=np.float32)])
    noise = np.full(SAMPLE_RATE, 0.005, dtype=np.float32)
    noise[:800] = 0.9  # a key press / bad first block at the start
    result = analyze(played, recorded, noise, candidate_delays=range(0, 41, 20))
    assert result.noise_floor_rms < 0.01
    assert result.recommended_barge_in < 0.5 and not result.problems


def test_invalid_recording_is_reported_and_never_written(tmp_path) -> None:
    import pytest

    from meetagent.audio.calibrate import write_config

    played = speech_like_signal(1.0, rms=0.1)
    recorded = np.concatenate([played, np.zeros(SAMPLE_RATE // 4, dtype=np.float32)])
    noise = np.full(SAMPLE_RATE, 1.6, dtype=np.float32)  # the impossible value seen on 2026-10-08
    result = analyze(played, recorded, noise, candidate_delays=range(0, 41, 20))
    assert result.problems
    config = tmp_path / "config.toml"
    config.write_text("[live]\nreply = \"voice\"\n", encoding="utf-8")
    with pytest.raises(ValueError):
        write_config(config, result)
    assert "barge_in_threshold" not in config.read_text(encoding="utf-8")


def test_current_aec_delay_kept_unless_clearly_better() -> None:
    from meetagent.audio import calibrate

    played = speech_like_signal(1.0, rms=0.1)
    recorded = np.concatenate([0.5 * played, np.zeros(SAMPLE_RATE // 4, dtype=np.float32)])
    noise = np.full(SAMPLE_RATE, 0.004, dtype=np.float32)
    fake = {0: 0.0044, 20: 0.0046, 100: 0.0045}
    original = calibrate.residual_after_aec
    calibrate.residual_after_aec = lambda p, r, d: np.full(len(r), fake[d], dtype=np.float32)
    try:
        kept = analyze(played, recorded, noise, candidate_delays=range(0, 101, 100), current_delay_ms=20)
        assert kept.best_aec_delay_ms == 20  # 0.0044 is not >10 % better than 0.0046
        fake[0] = 0.0020
        moved = analyze(played, recorded, noise, candidate_delays=range(0, 101, 100), current_delay_ms=20)
        assert moved.best_aec_delay_ms == 0
    finally:
        calibrate.residual_after_aec = original


def test_sustained_level_ignores_short_spikes() -> None:
    from meetagent.audio.calibrate import sustained_level

    frames = np.full(40, 0.005)
    frames[10:13] = 0.2          # 3 loud frames (~100 ms): below the 200 ms hold
    assert sustained_level(frames) == 0.005
    frames[20:28] = 0.04         # 8 frames (~260 ms) at 0.04
    assert sustained_level(frames) == 0.04


def test_choose_playback_gain_picks_loudest_usable_level() -> None:
    from meetagent.audio import calibrate

    speech = calibrate.calibration_speech()[: SAMPLE_RATE]
    assert 5.0 < len(calibrate.calibration_speech()) / SAMPLE_RATE <= 6.0
    levels = [(g, (speech * g).astype(np.float32), np.zeros(len(speech) + 8000, np.float32)) for g in (1.0, 0.7, 0.5)]
    needed = {1.0: 0.16, 0.7: 0.06, 0.5: 0.03}  # distortion fades as the level drops
    original = calibrate.needed_threshold
    calibrate.needed_threshold = lambda played, rec, d, floor: needed[round(float(np.abs(played).max() / np.abs(speech).max()), 1)]
    try:
        gain, threshold, results = calibrate.choose_playback_gain(levels, 0, 0.02)
        assert (gain, threshold) == (0.7, 0.06) and [g for g, _ in results] == [1.0, 0.7]
        needed[0.7] = 0.08
        gain, threshold, _ = calibrate.choose_playback_gain(levels, 0, 0.02)
        assert (gain, threshold) == (0.5, 0.03)
        needed[0.5] = 0.08
        assert calibrate.choose_playback_gain(levels, 0, 0.02)[0] is None
    finally:
        calibrate.needed_threshold = original


def test_playback_gain_scales_agent_voice_and_its_reference() -> None:
    from meetagent.audio.aec import PlaybackReference

    reference = PlaybackReference()
    playback = demo._PlaybackBuffer(reference, gain=0.5)
    playback.authorize_response()
    playback.add(np.full(480, 16384, dtype="<i2").tobytes())  # 0.5 full scale
    out = np.zeros((480, 1), dtype=np.float32)
    playback.callback(out, 480, None, None)
    assert np.allclose(out[:, 0], 0.25)
    # The AEC reference is what was actually played (scaled), not the raw stream.
    assert np.allclose(reference.read(320)[40:280], 0.25, atol=0.02)


def test_playback_gain_validation(tmp_path) -> None:
    import pytest

    from meetagent.config import AppConfig

    config = tmp_path / "c.toml"
    config.write_text("[live]\nplayback_gain = 0.5\n", encoding="utf-8")
    assert AppConfig.load(config).live.playback_gain == 0.5
    config.write_text("[live]\nplayback_gain = 1.5\n", encoding="utf-8")
    with pytest.raises(ValueError):
        AppConfig.load(config)


def test_repeated_level_uses_worst_recording() -> None:
    from meetagent.audio import calibrate

    speech = calibrate.calibration_speech()[: SAMPLE_RATE]
    good = np.zeros(len(speech) + 8000, np.float32)
    bad = np.ones(len(speech) + 8000, np.float32)
    levels = [(1.0, speech, good), (1.0, speech, bad), (0.5, speech * 0.5, good), (0.5, speech * 0.5, good)]
    original = calibrate.needed_threshold
    calibrate.needed_threshold = lambda played, rec, d, floor: 0.09 if rec[0] == 1 else 0.04
    try:
        gain, threshold, results = calibrate.choose_playback_gain(levels, 0, 0.02)
        assert results[0] == (1.0, 0.09) and (gain, threshold) == (0.5, 0.04)
    finally:
        calibrate.needed_threshold = original

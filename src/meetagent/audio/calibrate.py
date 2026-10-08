"""Measure speaker echo on this machine (PLAN.md Phase 8 P2).

`meetagent calibrate-echo` plays a speech-like test signal through the
speakers while recording the microphone, then:

- estimates the playback -> microphone delay by cross-correlation;
- replays the recording through WebRTC AEC offline for a range of
  `stream_delay_ms` values and keeps the one with the least residual echo;
- recommends a barge-in threshold above the residual echo level, so the
  Agent's own voice no longer counts as an interruption.

Run it at the speaker volume used in meetings; echo scales with volume.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16_000
FRAME = 512  # same block size as the meeting microphone callback
# Agent voice levels tried, loudest first; the first one whose echo the AEC
# cancels becomes [live].playback_gain.
CALIBRATION_GAINS = (1.0, 0.7, 0.5, 0.35, 0.25)
# Barge-in fires only when the AEC output stays above the threshold for
# barge_in_ms (200 ms ~ 7 frames of 32 ms), so a level needs a threshold just
# above the loudest *sustained* residual echo. Normal speech measured 0.10–0.42
# at the mic, so thresholds up to 0.05 still let people interrupt the Agent.
HOLD_FRAMES = 7
THRESHOLD_MARGIN = 1.5  # run-to-run AEC variance; speech (>= 0.10) stays far above
# AEC results vary a lot between runs (same level needed 0.028 once, 0.078 the
# next time), so every level is played twice and the worse result counts, and
# the threshold never goes below MIN_BARGE_IN_THRESHOLD.
MIN_BARGE_IN_THRESHOLD = 0.04
MAX_BARGE_IN_THRESHOLD = 0.06
REPEATS_PER_LEVEL = 2


@dataclass(frozen=True)
class EchoCalibration:
    noise_floor_rms: float
    raw_echo_rms: float
    measured_delay_ms: float
    best_aec_delay_ms: int | None
    residual_echo_p95: float | None
    recommended_barge_in: float
    # Reasons the measurement cannot be trusted; nothing is written if non-empty.
    problems: list[str] = field(default_factory=list)
    playback_gain: float = 1.0
    # (gain, barge-in threshold that level would need) for each level tried
    levels: list[tuple[float, float]] = field(default_factory=list)


def calibration_speech() -> np.ndarray:
    """Six seconds of a Gemini Live voice, the same kind of audio the Agent plays.

    Synthetic noise with a speech-like RMS did not distort the speakers the way
    real speech did (7 % vs. 26–49 % failing frames), so real speech is used.
    """
    import wave

    path = Path(__file__).with_name("calibration_speech.wav")
    with wave.open(str(path), "rb") as handle:
        pcm = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2")
    return pcm.astype(np.float32) / 32768.0


def sustained_level(frames: np.ndarray, hold_frames: int = HOLD_FRAMES) -> float:
    """Highest level the signal keeps for `hold_frames` consecutive frames."""
    if len(frames) < hold_frames:
        return float(frames.min()) if len(frames) else 0.0
    windows = np.lib.stride_tricks.sliding_window_view(frames, hold_frames)
    return float(windows.min(axis=1).max())


def needed_threshold(
    played: np.ndarray, recorded: np.ndarray, delay_ms: int, floor: float,
) -> float:
    """Barge-in threshold that the AEC residual echo of this level never triggers."""
    residual = residual_after_aec(played, recorded, delay_ms)[: len(played)]
    level = sustained_level(frame_rms(residual))
    return round(max(floor, MIN_BARGE_IN_THRESHOLD, THRESHOLD_MARGIN * level), 3)


def choose_playback_gain(
    levels: list[tuple[float, np.ndarray, np.ndarray]], delay_ms: int, floor: float,
) -> tuple[float | None, float | None, list[tuple[float, float]]]:
    """Loudest gain whose echo stays below a usable barge-in threshold.

    Returns (gain, threshold, per-level results); gain is None when even the
    quietest level would need a threshold above MAX_BARGE_IN_THRESHOLD.
    """
    results = []
    for gain in sorted({level[0] for level in levels}, reverse=True):
        # Worst of the repeated recordings at this gain.
        needed = max(
            needed_threshold(played, recorded, delay_ms, floor)
            for level_gain, played, recorded in levels if level_gain == gain
        )
        results.append((gain, needed))
        if needed <= MAX_BARGE_IN_THRESHOLD:
            return gain, needed, results
    return None, None, results


def speech_like_signal(seconds: float = 4.0, rms: float = 0.1, seed: int = 7) -> np.ndarray:
    """Band-limited noise with a ~4 Hz syllable envelope, scaled to `rms`."""
    from scipy.signal import butter, sosfilt

    rng = np.random.default_rng(seed)
    n = int(seconds * SAMPLE_RATE)
    noise = sosfilt(butter(4, [200, 4000], btype="band", fs=SAMPLE_RATE, output="sos"), rng.standard_normal(n))
    t = np.arange(n) / SAMPLE_RATE
    envelope = 0.55 + 0.45 * np.sin(2 * np.pi * 4.0 * t) ** 2
    signal = noise * envelope
    signal *= rms / np.sqrt(np.mean(signal * signal))
    return signal.astype(np.float32)


def frame_rms(audio: np.ndarray, frame: int = FRAME) -> np.ndarray:
    usable = len(audio) // frame * frame
    frames = audio[:usable].reshape(-1, frame)
    return np.sqrt(np.mean(frames * frames, axis=1))


def estimate_delay_ms(played: np.ndarray, recorded: np.ndarray, max_ms: float = 500.0) -> float:
    """Lag of `played` inside `recorded` via FFT cross-correlation."""
    from scipy.signal import correlate

    corr = correlate(recorded, played, mode="full", method="fft")
    lags = np.arange(-len(played) + 1, len(recorded))
    window = (lags >= 0) & (lags <= max_ms * SAMPLE_RATE / 1000)
    best = lags[window][np.argmax(np.abs(corr[window]))]
    return float(best) * 1000 / SAMPLE_RATE


def residual_after_aec(played: np.ndarray, recorded: np.ndarray, delay_ms: int) -> np.ndarray:
    """Run the app's WebRTC AEC frame by frame, reference given at render time."""
    from meetagent.audio.aec import WebRtcEchoCanceller

    aec = WebRtcEchoCanceller(stream_delay_ms=delay_ms)
    reference = np.zeros(len(recorded), dtype=np.float32)
    reference[: len(played)] = played[: len(recorded)]
    out = np.zeros_like(recorded)
    for start in range(0, len(recorded) // FRAME * FRAME, FRAME):
        out[start:start + FRAME] = aec.process(recorded[start:start + FRAME], reference[start:start + FRAME])
    return out


def analyze(
    played: np.ndarray, recorded: np.ndarray, noise: np.ndarray, rms_threshold: float = 0.02,
    candidate_delays: range = range(0, 201, 10), current_delay_ms: int | None = None,
) -> EchoCalibration:
    """Pure analysis of one calibration recording (testable without devices)."""
    problems: list[str] = []
    for label, audio in (("环境噪声录音", noise), ("回声录音", recorded)):
        if not np.all(np.isfinite(audio)):
            problems.append(f"{label}包含无效数值（NaN/Inf）")
        elif np.max(np.abs(audio)) > 1.5:
            problems.append(f"{label}幅度超出正常范围（{np.max(np.abs(audio)):.2f}），可能是设备启动时的异常数据")
    noise = np.nan_to_num(np.clip(noise, -1.0, 1.0))
    recorded = np.nan_to_num(np.clip(recorded, -1.0, 1.0))
    # Median over 32 ms frames: a key press or a glitchy first block during the
    # noise recording must not dominate (one run reported an impossible RMS 1.61).
    noise_floor = float(np.median(frame_rms(noise))) if len(noise) >= FRAME else 0.0
    if noise_floor > 0.05:
        problems.append(f"环境噪声过大（{noise_floor:.3f}），请在安静环境、不碰键盘时重测")
    during = recorded[: len(played)]
    raw_echo = float(np.sqrt(np.mean(during * during)))
    delay = estimate_delay_ms(played, recorded)
    best_delay: int | None = None
    residual_p95: float | None = None
    try:
        scores = {}
        for candidate in candidate_delays:
            residual = residual_after_aec(played, recorded, candidate)[: len(played)]
            # p95 over 32 ms frames includes the AEC's unconverged start, which
            # is exactly when the Agent used to get cut off.
            scores[candidate] = float(np.percentile(frame_rms(residual), 95))
        best_delay = min(scores, key=scores.get)
        # AEC3 estimates the delay itself: three runs on one laptop picked 100,
        # 70 and 0 ms with the same residual (0.0044–0.0046). Keep the current
        # setting unless another delay is clearly (>10 %) better.
        if current_delay_ms is not None:
            current = scores.get(current_delay_ms)
            if current is None:
                current = float(np.percentile(frame_rms(
                    residual_after_aec(played, recorded, current_delay_ms)[: len(played)]), 95))
                scores[current_delay_ms] = current
            if scores[best_delay] > 0.9 * current:
                best_delay = current_delay_ms
        residual_p95 = scores[best_delay]
    except RuntimeError:
        pass  # AEC extra not installed: base the threshold on raw echo
    echo_level = residual_p95 if residual_p95 is not None else float(np.percentile(frame_rms(during), 95))
    recommended = round(max(rms_threshold, 2.0 * echo_level, 3.0 * noise_floor), 3)
    if not 0 < recommended < 1:
        problems.append(f"推荐门限 {recommended} 不在 0–1 之间")
    return EchoCalibration(noise_floor, raw_echo, delay, best_delay, residual_p95, recommended, problems)


def _valid_noise(noise: np.ndarray) -> bool:
    return bool(np.all(np.isfinite(noise)) and np.max(np.abs(noise)) <= 1.5
                and np.median(frame_rms(noise)) <= 0.05)


def record_calibration(
    delay_ms: int = 0, threshold: float = 0.02, segment_seconds: float = 3.0,
) -> tuple[list[tuple[float, np.ndarray, np.ndarray]], np.ndarray]:
    """Play the speech sample at falling levels until the echo is cancelled, then record noise.

    On this machine the first input stream of a process sometimes returns
    garbage (a constant ~2.9 offset) for up to about a second; later streams
    are clean. So: discard a warm-up recording, record the echo first, and
    re-record the noise if it is still implausible.
    """
    import sounddevice as sd

    sd.rec(int(0.5 * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="float32", blocking=True)
    speech = calibration_speech()[: int(segment_seconds * SAMPLE_RATE)]
    tail = np.zeros(int(0.5 * SAMPLE_RATE), dtype=np.float32)  # catch delayed echo
    levels: list[tuple[float, np.ndarray, np.ndarray]] = []
    for gain in CALIBRATION_GAINS:
        played = (speech * gain).astype(np.float32)
        worst = 0.0
        for _ in range(REPEATS_PER_LEVEL):
            recorded = sd.playrec(
                np.concatenate([played, tail]), samplerate=SAMPLE_RATE, channels=1, dtype="float32", blocking=True,
            )[:, 0]
            levels.append((gain, played, recorded))
            worst = max(worst, needed_threshold(played, recorded, delay_ms, threshold))
            if worst > MAX_BARGE_IN_THRESHOLD:
                break  # this level already failed; no need to repeat it
        if worst <= MAX_BARGE_IN_THRESHOLD:
            break
    noise = np.zeros(0, dtype=np.float32)
    for _ in range(3):
        noise = sd.rec(int(1.5 * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="float32", blocking=True)[:, 0]
        if _valid_noise(noise):
            break
    return levels, noise


def set_toml_value(text: str, table: str, key: str, value: str) -> str:
    """Set `key = value` inside `[table]`, adding the key or table if missing."""
    table_re = re.compile(rf"^\[{re.escape(table)}\]\s*$", re.MULTILINE)
    match = table_re.search(text)
    if match is None:
        return text.rstrip("\n") + f"\n\n[{table}]\n{key} = {value}\n"
    next_table = re.compile(r"^\[", re.MULTILINE).search(text, match.end())
    end = next_table.start() if next_table else len(text)
    body = text[match.end():end]
    key_re = re.compile(rf"^{re.escape(key)}\s*=.*$", re.MULTILINE)
    if key_re.search(body):
        body = key_re.sub(f"{key} = {value}", body, count=1)
    else:
        body = body.rstrip("\n") + f"\n{key} = {value}\n" + ("\n" if next_table else "")
    return text[:match.end()] + body + text[end:]


def write_config(path: Path, result: EchoCalibration) -> None:
    if result.problems:
        raise ValueError("测量无效，未写入配置：" + "；".join(result.problems))
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    if result.best_aec_delay_ms is not None:
        text = set_toml_value(text, "audio", "aec_delay_ms", str(result.best_aec_delay_ms))
    text = set_toml_value(text, "live", "playback_gain", str(result.playback_gain))
    text = set_toml_value(text, "live", "barge_in_threshold", str(result.recommended_barge_in))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")

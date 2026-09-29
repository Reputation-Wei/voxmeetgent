"""Short, low-level microphone/speaker check for AEC on the current machine.

The script records no audio to disk and prints only one-second RMS measurements.
It plays a quiet band-limited noise probe so the AEC receives a known far-end
reference signal. Stop it with Ctrl+C at any time.
"""

from __future__ import annotations

import time

import numpy as np
import sounddevice as sd
from scipy.signal import butter, sosfiltfilt

from meetagent.audio.aec import PlaybackReference, WebRtcEchoCanceller


SPEAKER_RATE = 24_000
MIC_RATE = 16_000
BLOCK = 480
DURATION_SECONDS = 12


def main() -> None:
    devices = sd.default.device
    if devices is None or devices[0] is None or devices[1] is None:
        raise RuntimeError("请先在 macOS 选择默认麦克风和扬声器。")

    probe_length = SPEAKER_RATE * DURATION_SECONDS
    rng = np.random.default_rng(17)
    noise = rng.normal(0, 1, probe_length).astype(np.float32)
    band = butter(4, [300, 4_000], btype="bandpass", fs=SPEAKER_RATE, output="sos")
    noise = sosfiltfilt(band, noise).astype(np.float32)
    noise *= 0.018 / max(float(np.max(np.abs(noise))), 1e-6)
    # One second on, one second off, to distinguish acoustic echo from room noise.
    sample_times = np.arange(probe_length) / SPEAKER_RATE
    noise[(sample_times % 2) >= 1] = 0

    reference = PlaybackReference()
    delay_candidates = (20,)
    aec_processors = [WebRtcEchoCanceller(stream_delay_ms=delay) for delay in delay_candidates]
    output_cursor = 0
    raw_windows: list[list[np.ndarray]] = [[] for _ in range(DURATION_SECONDS)]
    clean_windows: list[list[list[np.ndarray]]] = [
        [[] for _ in delay_candidates] for _ in range(DURATION_SECONDS)
    ]
    started = time.monotonic()

    def output_callback(outdata: np.ndarray, frames: int, time_info: object, status: object) -> None:
        nonlocal output_cursor
        end = min(output_cursor + frames, probe_length)
        count = end - output_cursor
        outdata.fill(0)
        if count > 0:
            outdata[:count, 0] = noise[output_cursor:end]
        output_cursor = end
        reference.write_24k(outdata[:, 0])

    def input_callback(indata: np.ndarray, frames: int, time_info: object, status: object) -> None:
        elapsed = time.monotonic() - started
        index = min(int(elapsed), DURATION_SECONDS - 1)
        near = indata[:, 0].astype(np.float32, copy=True)
        far = reference.read(frames)
        raw_windows[index].append(near)
        for delay_index, aec in enumerate(aec_processors):
            clean_windows[index][delay_index].append(aec.process(near, far))

    print(f"麦克风：{sd.query_devices(devices[0])['name']}")
    print(f"扬声器：{sd.query_devices(devices[1])['name']}")
    print(
        f"开始 {DURATION_SECONDS} 秒低音量参考信号测试；比较 delay_ms={delay_candidates}，"
        "不会保存麦克风音频。"
    )
    with (
        sd.InputStream(
            device=devices[0], samplerate=MIC_RATE, channels=1, dtype="float32",
            blocksize=512, callback=input_callback,
        ),
        sd.OutputStream(
            device=devices[1], samplerate=SPEAKER_RATE, channels=1, dtype="float32",
            blocksize=BLOCK, callback=output_callback,
        ),
    ):
        try:
            while output_cursor < probe_length:
                time.sleep(0.1)
            time.sleep(0.5)
        except KeyboardInterrupt:
            print("收到 Ctrl+C，停止设备测试。")

    for second, (raw, clean_by_delay) in enumerate(zip(raw_windows, clean_windows, strict=True)):
        if not raw:
            continue
        raw_rms = float(np.sqrt(np.mean(np.concatenate(raw) ** 2)))
        phase = "probe" if second % 2 == 0 else "silence"
        changes = []
        for delay, clean in zip(delay_candidates, clean_by_delay, strict=True):
            if clean:
                clean_rms = float(np.sqrt(np.mean(np.concatenate(clean) ** 2)))
                change_db = 20 * np.log10(max(clean_rms, 1e-8) / max(raw_rms, 1e-8))
                changes.append(f"{delay}ms:{change_db:+.1f}dB")
        print(f"second={second:02d} phase={phase:7s} raw_rms={raw_rms:.5f} " + " ".join(changes))


if __name__ == "__main__":
    main()

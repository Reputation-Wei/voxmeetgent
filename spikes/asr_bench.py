"""Run one local Whisper backend on an audio fixture and report latency/RTF."""

import argparse
import json
import platform
import statistics
import subprocess
import time
import wave
from pathlib import Path

import numpy as np
import requests


def read_wav(path: Path) -> tuple[np.ndarray, float]:
    with wave.open(str(path), "rb") as wav:
        if wav.getsampwidth() != 2:
            raise ValueError("Input WAV must be 16-bit PCM")
        audio = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2")
        if wav.getnchannels() > 1:
            audio = audio.reshape(-1, wav.getnchannels()).mean(axis=1).astype("<i2")
        return audio, len(audio) / wav.getframerate()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", type=Path)
    parser.add_argument("--language", choices=("auto", "zh", "en", "de"), default="auto")
    parser.add_argument("--backend", choices=("mlx", "whisper-cli", "whisper-server"), default="mlx")
    parser.add_argument("--model", type=Path, help="GGML model path for whispercpp")
    parser.add_argument("--server-url", default="http://127.0.0.1:8080/inference")
    parser.add_argument("--repeat", type=int, default=3, help="Number of measured invocations")
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat must be at least 1")

    _, duration = read_wav(args.audio)
    times: list[float] = []
    if args.backend == "mlx":
        from meetagent.demo import _transcribe

        with wave.open(str(args.audio), "rb") as wav:
            audio = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(np.float32) / 32768
        for _ in range(args.repeat):
            start = time.perf_counter()
            text, detected_language = _transcribe(audio, args.language)
            times.append(time.perf_counter() - start)
    elif args.backend == "whisper-cli":
        if not args.model:
            parser.error("--model is required for --backend whisper-cli")
        command = [
            "whisper-cli", "-m", str(args.model), "-f", str(args.audio), "-np", "-nt",
        ]
        if args.language != "auto":
            command += ["-l", args.language]
        for _ in range(args.repeat):
            start = time.perf_counter()
            result = subprocess.run(command, text=True, capture_output=True, check=True)
            times.append(time.perf_counter() - start)
            text = result.stdout.strip()
            detected_language = args.language
    else:
        if not args.model:
            parser.error("--model is required to identify the whisper-server model in the report")
        for _ in range(args.repeat):
            start = time.perf_counter()
            with args.audio.open("rb") as audio_file:
                result = requests.post(
                    args.server_url,
                    files={"file": (args.audio.name, audio_file, "audio/wav")},
                    data={"temperature": "0", "response_format": "json", "language": args.language},
                    timeout=600,
                )
            result.raise_for_status()
            times.append(time.perf_counter() - start)
            payload = result.json()
            text = payload.get("text", "").strip()
            detected_language = payload.get("language", args.language)

    elapsed = statistics.median(times)

    print(json.dumps({
        "backend": args.backend,
        "model": "large-v3-turbo" if args.backend == "mlx" else str(args.model),
        "machine": platform.machine(),
        "system": platform.platform(),
        "language_setting": args.language,
        "detected_language": detected_language,
        "audio_seconds": round(duration, 3),
        "wall_seconds": round(elapsed, 3),
        "rtf": round(elapsed / duration, 4) if duration else None,
        "repeat_seconds": [round(item, 3) for item in times],
        "text": text,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

"""Local Whisper implementations used by the meeting pipeline."""

from __future__ import annotations

import os
import platform
import re
from functools import lru_cache

import numpy as np

MODEL_ID = "mlx-community/whisper-large-v3-mlx"
MIN_AUDIO_SECONDS = 0.65
MIN_AUDIO_RMS = 0.003
MAX_NO_SPEECH_PROB = 0.55
MIN_AVG_LOGPROB = -1.0


def simplify_chinese(text: str) -> str:
    """Normalize Whisper's traditional Chinese output to simplified Chinese."""
    try:
        from opencc import OpenCC
    except ImportError:
        return text
    return OpenCC("t2s").convert(text)


def _float_setting(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


@lru_cache(maxsize=1)
def _linux_model(model_path: str):
    from pywhispercpp.model import Model

    return Model(
        model_path, n_threads=os.cpu_count() or 4,
        redirect_whispercpp_logs_to=None, print_progress=False,
    )


class LocalWhisperBackend:
    """Run MLX Whisper on Apple Silicon or a cached whisper.cpp model on Linux."""

    def __init__(self, model_path: str | None = None, model_id: str | None = None) -> None:
        if model_path is None:
            from meetagent.config import AppConfig

            # Env var wins over [audio].model_path; AppConfig applies that order.
            configured = AppConfig.load().audio.model_path
            model_path = str(configured) if configured else None
        self.model_path = model_path
        self.model_id = model_id or os.environ.get("MEETAGENT_WHISPER_MODEL_ID") or MODEL_ID

    def transcribe(self, audio: np.ndarray, language: str = "auto") -> tuple[str, str]:
        audio = np.asarray(audio, dtype=np.float32)
        audio = np.nan_to_num(audio).reshape(-1)
        duration = len(audio) / 16_000
        rms = float(np.sqrt(np.mean(audio * audio))) if audio.size else 0.0
        if duration < MIN_AUDIO_SECONDS or rms < MIN_AUDIO_RMS:
            return "", "unknown"
        no_speech_threshold = _float_setting("MEETAGENT_WHISPER_NO_SPEECH", MAX_NO_SPEECH_PROB)
        logprob_threshold = _float_setting("MEETAGENT_WHISPER_LOGPROB", MIN_AVG_LOGPROB)
        compression_threshold = _float_setting("MEETAGENT_WHISPER_COMPRESSION", 2.2)
        if platform.system() == "Darwin" and platform.machine() == "arm64":
            try:
                import mlx_whisper
            except ImportError as exc:
                if "No Metal device" in str(exc):
                    raise RuntimeError("MLX 找不到可用的 Metal 设备；请在本机桌面 macOS 环境运行。") from exc
                raise RuntimeError(f"无法加载 mlx-whisper：{exc}。请在仓库目录执行 `uv sync`。") from exc
            result = mlx_whisper.transcribe(
                audio,
                path_or_hf_repo=self.model_id,
                language=None if language == "auto" else language,
                task="transcribe",
                condition_on_previous_text=False,
                temperature=0.0,
                no_speech_threshold=no_speech_threshold,
                logprob_threshold=logprob_threshold,
                compression_ratio_threshold=compression_threshold,
            )
            segments = result.get("segments", [])
            # MLX's avg_logprob/compression/no_speech metrics describe the decoded
            # 30s window and are copied to subsegments. Use them to flag the whole
            # VAD utterance; don't drop individual subtitle fragments on that basis.
            suspicious_window = bool(segments) and (
                all(float(segment.get("no_speech_prob", 0.0)) > no_speech_threshold for segment in segments)
                or all(float(segment.get("avg_logprob", 0.0)) < logprob_threshold for segment in segments)
                or all(float(segment.get("compression_ratio", 0.0)) > compression_threshold for segment in segments)
            )
            if suspicious_window:
                return "", result.get("language", "unknown")
            text = " ".join(str(segment.get("text", "")).strip() for segment in segments).strip()
            normalized = re.sub(r"\W+", "", text.casefold())
            # Suppress obvious looped fragments produced from music or silence.
            if len(normalized) >= 18:
                for width in range(4, min(20, len(normalized) // 2) + 1):
                    for start in range(0, len(normalized) - 2 * width + 1):
                        left = normalized[start : start + width]
                        right = normalized[start + width : start + 2 * width]
                        if left == right:
                            return "", result.get("language", "unknown")
            return simplify_chinese(text), result.get("language", "unknown")

        if platform.system() == "Linux":
            try:
                from pywhispercpp.model import Model  # noqa: F401
            except ImportError as exc:
                raise RuntimeError("缺少 pywhispercpp，请在 Python 3.12 环境执行 `uv sync`。") from exc
            if not self.model_path:
                raise RuntimeError(
                    "Linux 转写需要设置 MEETAGENT_WHISPER_MODEL 或 config.toml 的 [audio].model_path，"
                    "指向 whisper.cpp GGML 模型文件。"
                )
            segments = _linux_model(self.model_path).transcribe(
                audio, language=None if language == "auto" else language, translate=False,
                no_speech_thold=no_speech_threshold, logprob_thold=logprob_threshold,
                entropy_thold=compression_threshold,
            )
            text = " ".join(segment.text.strip() for segment in segments).strip()
            normalized = re.sub(r"\W+", "", text.casefold())
            if len(normalized) >= 18:
                for width in range(4, min(20, len(normalized) // 2) + 1):
                    for start in range(0, len(normalized) - 2 * width + 1):
                        if normalized[start : start + width] == normalized[start + width : start + 2 * width]:
                            return "", "unknown"
            return simplify_chinese(text), "unknown"

        raise RuntimeError("此 Demo 的本地 ASR 仅支持 Apple Silicon macOS 和 Linux。")


def create_local_backend(model_path: str | None = None, model_id: str | None = None) -> LocalWhisperBackend:
    return LocalWhisperBackend(model_path, model_id)

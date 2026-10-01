"""Environment diagnostics; lists devices but does not start audio capture."""

from __future__ import annotations

import importlib.util
import os
import platform
import tempfile
from pathlib import Path

from meetagent.config import AppConfig


def diagnostics(config: AppConfig | None = None) -> list[tuple[str, bool, str]]:
    """Return labeled checks without opening the microphone or revealing secrets."""
    config = config or AppConfig.load()
    key_is_set = bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
    checks: list[tuple[str, bool, str]] = [
        ("系统", True, f"{platform.system()} {platform.machine()}"),
        ("Gemini API key", key_is_set,
         "已设置" if key_is_set else "未设置（文本/Live 云端功能不可用）"),
    ]

    output_dir = config.output.directory
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(prefix=".meetagent-doctor-", dir=output_dir, delete=True):
            pass
        output_ok, output_detail = True, f"可写：{output_dir}"
    except OSError as exc:
        output_ok, output_detail = False, f"不可写：{output_dir}（{exc}）"
    checks.append(("输出目录", output_ok, output_detail))

    sounddevice_available = importlib.util.find_spec("sounddevice") is not None
    checks.append(("音频依赖", sounddevice_available,
                   "sounddevice 已安装" if sounddevice_available else "缺少 sounddevice"))
    try:
        import sherpa_onnx

        sherpa_available = bool(sherpa_onnx.SpeakerEmbeddingExtractor)
        sherpa_detail = f"sherpa-onnx Python/C++ backend {getattr(sherpa_onnx, '__version__', '已加载')}"
    except Exception as exc:
        sherpa_available, sherpa_detail = False, f"不可用：{exc}"
    checks.append(("说话人识别", sherpa_available, sherpa_detail))
    speaker_model = Path(os.getenv("MEETAGENT_SPEAKER_MODEL", "")).expanduser() if os.getenv(
        "MEETAGENT_SPEAKER_MODEL"
    ) else Path.home() / ".cache" / "meetagent" / "models" / (
        "3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"
    )
    checks.append(("说话人模型", speaker_model.is_file(),
                   str(speaker_model) if speaker_model.is_file() else "首次转写时自动下载"))
    if sounddevice_available:
        try:
            import sounddevice as sd

            input_devices = [device for device in sd.query_devices()
                             if device.get("max_input_channels", 0) > 0]
            device_names = ", ".join(str(device["name"]) for device in input_devices[:3])
            checks.append(("麦克风设备", bool(input_devices),
                           device_names if input_devices else "未发现输入设备；请检查系统权限和音频设备"))
        except Exception as exc:
            checks.append(("麦克风设备", False, f"查询失败：{exc}"))
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        mlx_available = importlib.util.find_spec("mlx_whisper") is not None
        checks.append(("MLX Whisper", mlx_available,
                       "可用" if mlx_available else "请运行 uv sync 安装 mlx-whisper"))
        if mlx_available:
            try:
                import mlx.core as mx

                metal_available = mx.metal.is_available()
                checks.append(("Metal", metal_available,
                               f"可用，默认设备 {mx.default_device()}" if metal_available else "Metal 不可用"))
            except Exception as exc:
                checks.append(("Metal", False, f"检测失败：{exc}"))
    else:
        model = config.audio.model_path or (Path(os.getenv("MEETAGENT_WHISPER_MODEL"))
                                             if os.getenv("MEETAGENT_WHISPER_MODEL") else None)
        checks.append(("whisper.cpp 模型", bool(model and model.is_file()),
                       str(model) if model else "设置 MEETAGENT_WHISPER_MODEL 或 [audio].model_path"))
        checks.append(_whispercpp_gpu_check())
    return checks


def _whispercpp_gpu_check() -> tuple[str, bool, str]:
    """Report whether pywhispercpp was built with the ggml Vulkan backend."""
    spec = importlib.util.find_spec("_pywhispercpp")
    if spec is None or spec.origin is None:
        return ("whisper.cpp GPU", False, "未安装 pywhispercpp")
    if any(Path(spec.origin).parent.glob("libggml-vulkan.so*")):
        return ("whisper.cpp GPU", True, "Vulkan 后端已编译")
    return ("whisper.cpp GPU", False, "PyPI 轮子仅含 CPU；运行 scripts/build-whisper-vulkan.sh")


def exit_code(checks: list[tuple[str, bool, str]]) -> int:
    """Return nonzero only for missing local runtime prerequisites."""
    required = {"音频依赖", "说话人识别", "MLX Whisper", "whisper.cpp 模型"}
    return 1 if any(not ok and name in required for name, ok, _ in checks) else 0

"""Typed application configuration and file loading helpers."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class AudioConfig:
    """Audio capture and local transcription defaults."""

    sample_rate: int = 16_000
    channels: int = 1
    language: Literal["auto", "zh", "en", "de"] = "auto"
    backend: Literal["auto", "mlx", "whispercpp"] = "auto"
    model_path: Path | None = None
    aec_enabled: bool = True
    aec_delay_ms: int = 20


@dataclass(frozen=True)
class OutputConfig:
    """Where transcript and summary files are written."""

    directory: Path = field(default_factory=lambda: Path.home() / "Meetings")
    save_audio: bool = True
    obsidian_vault: Path | None = None
    obsidian_folder: str = "Meetings"


@dataclass(frozen=True)
class LiveConfig:
    """Gemini Live behavior without storing the API key in the config file."""

    model: str = "gemini-3.8-live"
    mode: Literal["realtime", "text_trigger"] = "realtime"
    intervention: Literal["silent", "on_request", "proactive"] = "on_request"
    rms_threshold: float = 0.02


@dataclass(frozen=True)
class AppConfig:
    """Complete immutable application configuration."""

    audio: AudioConfig = field(default_factory=AudioConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    live: LiveConfig = field(default_factory=LiveConfig)

    @classmethod
    def load(cls, path: Path | None = None) -> "AppConfig":
        """Load TOML settings and apply supported environment overrides."""
        config_path = path or Path.home() / ".config" / "meetagent" / "config.toml"
        raw: dict[str, object] = {}
        if config_path.exists():
            with config_path.open("rb") as handle:
                raw = tomllib.load(handle)

        audio_raw = raw.get("audio", {})
        output_raw = raw.get("output", {})
        live_raw = raw.get("live", {})
        if not isinstance(audio_raw, dict) or not isinstance(output_raw, dict) or not isinstance(live_raw, dict):
            raise ValueError("TOML 的 [audio]、[output]、[live] 必须是配置表。")
        intervention = live_raw.get("intervention", "on_request")
        if intervention not in {"silent", "on_request", "proactive"}:
            raise ValueError("[live].intervention 只能是 silent、on_request 或 proactive。")
        rms_threshold = float(live_raw.get("rms_threshold", 0.02))
        if not 0 < rms_threshold < 1:
            raise ValueError("[live].rms_threshold 必须大于 0 且小于 1。")
        audio_aec_delay_ms = int(audio_raw.get("aec_delay_ms", 20))
        if not 0 <= audio_aec_delay_ms <= 500:
            raise ValueError("[audio].aec_delay_ms 必须在 0 到 500 毫秒之间。")

        model_path = os.getenv("MEETAGENT_WHISPER_MODEL") or audio_raw.get("model_path")
        output_dir = os.getenv("MEETAGENT_OUTPUT_DIR") or output_raw.get("directory")
        obsidian_vault = os.getenv("MEETAGENT_OBSIDIAN_VAULT") or output_raw.get("obsidian_vault")
        return cls(
            audio=AudioConfig(
                sample_rate=int(audio_raw.get("sample_rate", 16_000)),
                channels=int(audio_raw.get("channels", 1)),
                language=audio_raw.get("language", "auto"),
                backend=audio_raw.get("backend", "auto"),
                model_path=Path(model_path).expanduser() if model_path else None,
                aec_enabled=bool(audio_raw.get("aec_enabled", True)),
                aec_delay_ms=audio_aec_delay_ms,
            ),
            output=OutputConfig(
                directory=Path(output_dir).expanduser() if output_dir else Path.home() / "Meetings",
                save_audio=bool(output_raw.get("save_audio", True)),
                obsidian_vault=Path(obsidian_vault).expanduser() if obsidian_vault else None,
                obsidian_folder=str(output_raw.get("obsidian_folder", "Meetings")),
            ),
            live=LiveConfig(
                model=live_raw.get("model", "gemini-3.8-live"),
                mode=live_raw.get("mode", "realtime"),
                intervention=intervention,
                rms_threshold=rms_threshold,
            ),
        )

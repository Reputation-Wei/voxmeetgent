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
    # Send speaker-labeled local transcript lines and Analyst conclusions to the
    # Live session as context (PLAN.md Phase 8 P1). Off until the P1 spike shows
    # that mixing send_client_content with realtime audio is safe; the SDK warns
    # that interleaving them "can lead to unexpected results".
    share_transcript: bool = False
    # "text" (default): print and save answers only, no Gemini Live, no meeting
    # audio upload. "voice": speak them; needs headphones or a calibrated
    # barge-in threshold (`meetagent calibrate-echo`) on laptop speakers.
    reply: Literal["voice", "text"] = "text"
    # Barge-in while the Agent is speaking: mic level (after AEC) must stay
    # above this for barge_in_ms. None = use rms_threshold. Laptop speaker
    # echo reaches about the level of real speech, so calibrate it with
    # `meetagent calibrate-echo` when not using headphones.
    barge_in_threshold: float | None = None
    barge_in_ms: int = 200
    # Scales only the Agent's voice. Laptop speakers distort loud speech, and
    # the AEC cannot cancel distorted echo (measured 2026-10-08: residual 0.165
    # at full level vs. 0.014 at 0.4). `meetagent calibrate-echo` picks it.
    playback_gain: float = 1.0


@dataclass(frozen=True)
class AgentConfig:
    """How attendees address the Agent (PLAN.md Phase 8 P2)."""

    # Names attendees use; also given to Whisper as a vocabulary hint, which
    # fixes "Agent" being heard as "A阵盾" in Chinese mode.
    names: tuple[str, ...] = ("Agent", "助手", "MeetAgent")
    # After the Agent answers, follow-up questions need no name for this long.
    followup_seconds: float = 25.0
    # Let Gemini decide whether a name mention is really addressed to the
    # Agent; falls back to local rules without a key or on errors.
    classifier: bool = True


@dataclass(frozen=True)
class AppConfig:
    """Complete immutable application configuration."""

    audio: AudioConfig = field(default_factory=AudioConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    live: LiveConfig = field(default_factory=LiveConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)

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
        agent_raw = raw.get("agent", {})
        if not all(isinstance(table, dict) for table in (audio_raw, output_raw, live_raw, agent_raw)):
            raise ValueError("TOML 的 [audio]、[output]、[live]、[agent] 必须是配置表。")
        names = agent_raw.get("names", list(AgentConfig.names))
        if not isinstance(names, list) or not names or not all(isinstance(n, str) and n.strip() for n in names):
            raise ValueError("[agent].names 必须是非空的名字列表。")
        playback_gain = float(live_raw.get("playback_gain", 1.0))
        if not 0 < playback_gain <= 1:
            raise ValueError("[live].playback_gain 必须大于 0 且不超过 1。")
        barge_in_threshold = live_raw.get("barge_in_threshold")
        if barge_in_threshold is not None and not 0 < float(barge_in_threshold) < 1:
            raise ValueError("[live].barge_in_threshold 必须大于 0 且小于 1。")
        intervention = live_raw.get("intervention", "on_request")
        if intervention not in {"silent", "on_request", "proactive"}:
            raise ValueError("[live].intervention 只能是 silent、on_request 或 proactive。")
        reply = live_raw.get("reply", "text")
        if reply not in {"voice", "text"}:
            raise ValueError("[live].reply 只能是 voice 或 text。")
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
                share_transcript=bool(live_raw.get("share_transcript", False)),
                reply=reply,
                barge_in_threshold=float(barge_in_threshold) if barge_in_threshold is not None else None,
                barge_in_ms=int(live_raw.get("barge_in_ms", 200)),
                playback_gain=playback_gain,
            ),
            agent=AgentConfig(
                names=tuple(name.strip() for name in names),
                followup_seconds=float(agent_raw.get("followup_seconds", 25.0)),
                classifier=bool(agent_raw.get("classifier", True)),
            ),
        )

"""Manage one microphone meeting for CLI-independent local clients."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from meetagent.config import AppConfig
from meetagent.demo import run_demo


@dataclass
class MeetingRun:
    meeting_id: str
    output_dir: Path
    language: str
    summary_language: str
    live: bool
    reply: str
    started_at: float
    stop_event: asyncio.Event
    task: asyncio.Task[None]
    transcript_path: Path


class MeetingService:
    """Own the active meeting task and expose safe read/control operations."""

    def __init__(self) -> None:
        self.config = AppConfig.load()
        self.active: MeetingRun | None = None
        self.last_error: str | None = None
        self.last_meeting: MeetingRun | None = None
        self._lock = asyncio.Lock()
        self._speaker_undo: dict[str, tuple[float, str, str]] = {}

    async def start(
        self, *, output_dir: Path | None = None, language: str = "auto",
        summary_language: str = "zh", live: bool = False,
        consent_to_stream: bool = False,
        consent_to_summary_upload: bool = False,
        reply: str | None = None,
    ) -> MeetingRun:
        async with self._lock:
            if self.active and not self.active.task.done():
                raise RuntimeError("已有会议正在运行。")
            reply = reply or self.config.live.reply
            if reply not in {"voice", "text"}:
                raise ValueError("reply 必须为 voice 或 text。")
            if reply == "text" and live:
                raise ValueError("文字回复模式不使用 Gemini Live；请关闭 Live 或改用语音回复。")
            if live and not consent_to_stream:
                raise PermissionError("启用 Gemini Live 前必须确认与会者同意整场音频上传。")
            if live and not (self._has_api_key()):
                raise RuntimeError("启用 Gemini Live 需要 GEMINI_API_KEY 或 GOOGLE_API_KEY。")
            if language not in {"auto", "zh", "en", "de"}:
                raise ValueError("language 必须为 auto、zh、en 或 de。")
            if summary_language not in {"zh", "en", "de"}:
                raise ValueError("summary_language 必须为 zh、en 或 de。")
            destination = (output_dir or self.config.output.directory).expanduser().resolve()
            if (destination / ".obsidian").is_dir():
                destination = destination / "Meetings"
            destination.mkdir(parents=True, exist_ok=True)
            if not destination.is_dir():
                raise ValueError("输出路径不是文件夹。")
            # run_demo creates a timestamped .txt before returning; reserve the id now.
            meeting_id = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
            transcript_path = destination / f"meeting-{meeting_id}.txt"
            stop_event = asyncio.Event()
            started_at = asyncio.get_running_loop().time()
            task = asyncio.create_task(
                run_demo(
                    destination, None, language=language, summary_language=summary_language,
                    enable_live=live, external_stop=stop_event, meeting_id=meeting_id,
                    allow_gemini_summary=consent_to_summary_upload, reply=reply,
                ),
                name=f"meeting-{meeting_id}",
            )
            run = MeetingRun(
                meeting_id, destination, language, summary_language, live, reply,
                started_at, stop_event, task, transcript_path,
            )
            self.active = run
            self.last_meeting = run
            self.last_error = None
            task.add_done_callback(self._capture_task_error)
            return run

    @staticmethod
    def _has_api_key() -> bool:
        import os

        return bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))

    def _capture_task_error(self, task: asyncio.Task[None]) -> None:
        if task.cancelled():
            return
        error = task.exception()
        if error:
            self.last_error = str(error)

    async def stop(self, meeting_id: str | None = None) -> None:
        async with self._lock:
            run = self.active
            if run is None:
                raise RuntimeError("当前没有正在运行的会议。")
            if meeting_id and meeting_id != run.meeting_id:
                raise ValueError("会议 ID 与当前活动会议不匹配。")
            run.stop_event.set()
            task = run.task
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=45)
        except TimeoutError as exc:
            raise RuntimeError("会议仍在停止；音频/逐字稿收尾超时，请稍候再查询状态。") from exc

    def status(self) -> dict[str, Any]:
        run = self.active
        if run is None:
            return {
                "state": "idle", "meeting_id": None, "elapsed_seconds": 0,
                "output_dir": str(self.config.output.directory), "live": False,
                "reply": self.config.live.reply,
                "gemini_key_available": self._has_api_key(), "error": self.last_error,
            }
        task_done = run.task.done()
        state = "error" if task_done and self.last_error else ("stopped" if task_done else (
            "stopping" if run.stop_event.is_set() else "recording"
        ))
        return {
            "state": state, "meeting_id": run.meeting_id,
            "elapsed_seconds": max(0, int(asyncio.get_running_loop().time() - run.started_at)),
            "output_dir": str(run.output_dir), "transcript": str(run.transcript_path),
            "live": run.live, "reply": run.reply, "live_connected": None,
            "gemini_key_available": self._has_api_key(), "error": self.last_error,
            "last_agent": self._last_agent_entry(run.transcript_path.with_suffix(".jsonl")),
            "audio_path": str(run.transcript_path.with_suffix(".wav")),
            "summary_path": str(run.transcript_path.with_suffix(".md")),
        }

    @staticmethod
    def _last_agent_entry(path: Path, tail_bytes: int = 65_536) -> dict[str, Any] | None:
        """Latest Agent answer, read from the end of the JSONL so polling stays cheap."""
        try:
            with path.open("rb") as handle:
                handle.seek(0, 2)
                size = handle.tell()
                handle.seek(max(0, size - tail_bytes))
                lines = handle.read().decode("utf-8", errors="ignore").splitlines()
        except OSError:
            return None
        for line in reversed(lines):
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue  # the first line of the tail may be cut mid-record
            if entry.get("speaker") == "Agent":
                return {"time": entry.get("time", ""), "text": entry.get("text", ""),
                        "source": entry.get("source", "")}
        return None

    def transcript(self, meeting_id: str) -> list[dict[str, Any]]:
        run = self._find_meeting(meeting_id)
        path = run.transcript_path.with_suffix(".jsonl")
        if not path.is_file():
            return []
        result = []
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                result.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return result

    def correct_speaker(self, meeting_id: str, timestamp: float, name: str) -> dict[str, Any]:
        run = self._find_meeting(meeting_id)
        from meetagent.storage.jsonl import JsonlTranscriptStore

        entry = JsonlTranscriptStore(run.transcript_path).correct_speaker(timestamp, name)
        self._speaker_undo[meeting_id] = (
            float(entry["timestamp"]), str(entry.get("speaker_original", "说话人未知")),
            str(entry["speaker"]),
        )
        return entry

    def undo_speaker_correction(self, meeting_id: str) -> dict[str, Any]:
        run = self._find_meeting(meeting_id)
        previous = self._speaker_undo.pop(meeting_id, None)
        if previous is None:
            raise ValueError("没有可撤销的说话人修正。")
        timestamp, old_name, corrected_name = previous
        entries = self.transcript(meeting_id)
        target = next((entry for entry in entries if float(entry["timestamp"]) == timestamp), None)
        if target is None or target.get("speaker") != corrected_name:
            raise ValueError("逐字稿已发生后续修改，不能安全撤销。")
        from meetagent.storage.jsonl import JsonlTranscriptStore

        target["speaker"] = old_name
        target["speaker_corrected"] = False
        target["speaker_undo_of"] = corrected_name
        JsonlTranscriptStore(run.transcript_path).rewrite(entries)
        return target

    def _find_meeting(self, meeting_id: str) -> MeetingRun:
        run = self.last_meeting
        if run is None or run.meeting_id != meeting_id:
            raise KeyError("找不到该会议。")
        return run

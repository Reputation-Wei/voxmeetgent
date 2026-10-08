"""`meetagent follow`: a terminal view of the running meeting.

The meeting itself runs inside the local GUI service (started from the bar
plugin), so its console output only reaches a log file. This command polls the
same loopback API the plugin uses and prints the live transcript and the
Agent's text answers. It keeps running between meetings, so one window can
follow several in a row.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

# Distinct, readable colors for speakers; the Agent has its own panel style.
_SPEAKER_STYLES = ("green", "magenta", "yellow", "blue", "red", "bright_cyan")


class LocalApi:
    """Minimal client for the token-protected loopback API."""

    def __init__(self, port: int = 8765) -> None:
        self.base = f"http://127.0.0.1:{port}"
        self.token = ""

    def _get(self, path: str) -> Any:
        request = urllib.request.Request(self.base + path, headers={"X-MeetAgent-Token": self.token})
        with urllib.request.urlopen(request, timeout=5) as response:
            return json.loads(response.read().decode("utf-8"))

    def get(self, path: str) -> Any:
        if not self.token:
            self.token = self._get("/api/bootstrap")["token"]
        try:
            return self._get(path)
        except urllib.error.HTTPError as exc:
            if exc.code == 401:  # service restarted with a new token
                self.token = self._get("/api/bootstrap")["token"]
                return self._get(path)
            raise


class TranscriptPrinter:
    """Format transcript entries; Agent answers become Markdown panels."""

    def __init__(self, console: Console) -> None:
        self.console = console
        self._styles: dict[str, str] = {}

    def _style(self, speaker: str) -> str:
        if speaker not in self._styles:
            self._styles[speaker] = _SPEAKER_STYLES[len(self._styles) % len(_SPEAKER_STYLES)]
        return self._styles[speaker]

    def print_entry(self, entry: dict[str, Any]) -> None:
        speaker = str(entry.get("speaker") or "说话人未知")
        text = str(entry.get("text") or "").strip()
        stamp = str(entry.get("time") or "--:--")
        if not text:
            return
        if speaker == "Agent":
            self.console.print(Panel(
                Markdown(text), title=f"Agent · {stamp}", title_align="left", border_style="cyan",
            ))
            return
        line = Text()
        line.append(f"[{stamp}] ", style="dim")
        line.append(f"{speaker}: ", style=f"bold {self._style(speaker)}")
        line.append(text)
        self.console.print(line)


def run_follow(
    port: int = 8765, poll_seconds: float = 1.0,
    console: Console | None = None, max_polls: int | None = None,
) -> None:
    """Follow meetings until interrupted (`max_polls` only bounds tests)."""
    console = console or Console()
    api = LocalApi(port)
    printer = TranscriptPrinter(console)
    meeting_id: str | None = None
    offset = 0
    announced_end = False
    waiting_shown = False
    console.print("[bold]MeetAgent 实时转写[/bold]  ·  Ctrl+C 关闭窗口（不会停止会议）")
    polls = 0
    while max_polls is None or polls < max_polls:
        polls += 1
        try:
            status = api.get("/api/status")
        except (urllib.error.URLError, OSError, KeyError, json.JSONDecodeError):
            if not waiting_shown:
                console.print("[yellow]等待 MeetAgent 服务启动…[/yellow]")
                waiting_shown = True
            time.sleep(2)
            continue
        waiting_shown = False
        current = status.get("meeting_id")
        if current and current != meeting_id:
            meeting_id, offset, announced_end = current, 0, False
            mode = "文字回答" if status.get("reply") == "text" else "语音回答"
            console.rule(f"会议 {meeting_id} · {mode}")
            console.print(f"[dim]保存目录：{status.get('output_dir', '')}[/dim]")
        if meeting_id:
            try:
                page = api.get(f"/api/meetings/{meeting_id}/transcript?offset={offset}&limit=500")
            except urllib.error.HTTPError:
                page = {"entries": [], "total": offset}
            for entry in page.get("entries", []):
                printer.print_entry(entry)
            offset += len(page.get("entries", []))
            # Speaker corrections or late entries can rewrite the file shorter.
            offset = min(offset, int(page.get("total", offset)))
            state = status.get("state")
            if state in {"stopped", "error"} and not announced_end:
                announced_end = True
                if state == "error":
                    console.print(f"[red]会议出错：{status.get('error') or '未知错误'}[/red]")
                transcript = str(status.get("transcript") or "")
                console.rule("会议已结束")
                if status.get("summary_path"):
                    console.print(f"纪要：{status['summary_path']}")
                if transcript:
                    console.print(f"Agent 回答：{transcript.removesuffix('.txt')}.agent.md")
                console.print("[dim]窗口会继续等待下一场会议；Ctrl+C 关闭。[/dim]")
        time.sleep(poll_seconds)

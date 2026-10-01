"""Shared working memory of the running meeting (Phase 8 P1).

Every part of the Agent that needs meeting context reads it from here instead
of slicing the transcript itself:

- the Analyst (fact check, logic check, research, summarize) gets the recent
  transcript with real speaker names plus a rolling digest of everything older;
- the Live voice session receives the same speaker-labeled lines incrementally,
  so its answers know who said what and what the Analyst already found.

The transcript entries are the same dicts the demo pipeline writes to
TXT/JSONL; MeetingState only reads them. The digest is updated lazily, right
before an analysis request, so no transcript is sent to Gemini unless a
trigger already requires a Gemini call.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

# Sources that represent what attendees actually said. Agent output is kept
# separately so the Agent does not fact-check or summarize itself as a speaker.
# Live's own input transcription (gemini_live_fallback) stays in the TXT/JSONL
# but is excluded here: it mishears often (e.g. a whole invented sentence) and
# the local Whisper line for the same speech is usually present.
ATTENDEE_SOURCES = {"local_asr"}

# Updates the digest: (previous digest, transcript lines not yet digested) -> new digest.
DigestFunction = Callable[[str, str], Awaitable[str]]


@dataclass
class AgentConclusion:
    """One Analyst answer the Agent already gave, kept short for context."""

    timestamp: float
    kind: str
    text: str
    sources: tuple[str, ...] = ()


@dataclass
class MeetingState:
    """Rolling, speaker-aware view of one meeting."""

    entries: list[dict[str, Any]]
    digest_function: DigestFunction | None = None
    # Lines newer than this many seconds stay verbatim; older ones get digested.
    recent_seconds: float = 120.0
    digest: str = ""
    conclusions: list[AgentConclusion] = field(default_factory=list)
    _digested_until: float = 0.0
    _digest_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    # ------------------------------------------------------------ transcript

    def attendee_entries(self, since: float = 0.0, until: float | None = None) -> list[dict[str, Any]]:
        """Attendee utterances with since < timestamp <= until, in time order."""
        selected = [
            entry for entry in self.entries
            if entry.get("source") in ATTENDEE_SOURCES
            and float(entry.get("timestamp", 0)) > since
            and (until is None or float(entry.get("timestamp", 0)) <= until)
            and str(entry.get("text", "")).strip()
        ]
        return sorted(selected, key=lambda entry: float(entry["timestamp"]))

    @staticmethod
    def format_line(entry: dict[str, Any]) -> str:
        speaker = entry.get("speaker") or "说话人未知"
        return f"[{entry.get('time', '--:--')}] {speaker}: {str(entry.get('text', '')).strip()}"

    def recent_transcript(self, seconds: float | None = None, now: float | None = None) -> str:
        """Verbatim speaker-labeled transcript of the last `seconds`."""
        now = time.time() if now is None else now
        window = self.recent_seconds if seconds is None else seconds
        return "\n".join(self.format_line(entry) for entry in self.attendee_entries(since=now - window))

    # ---------------------------------------------------------------- digest

    async def refresh_digest(self, now: float | None = None) -> str:
        """Fold transcript older than the recent window into the digest.

        Without a digest function (no Gemini consent/key) the digest stays
        empty and callers fall back to the recent transcript alone.
        """
        if self.digest_function is None:
            return self.digest
        now = time.time() if now is None else now
        cutoff = now - self.recent_seconds
        async with self._digest_lock:
            pending = self.attendee_entries(since=self._digested_until, until=cutoff)
            if not pending:
                return self.digest
            new_lines = "\n".join(self.format_line(entry) for entry in pending)
            self.digest = (await self.digest_function(self.digest, new_lines)).strip()
            self._digested_until = float(pending[-1]["timestamp"])
        return self.digest

    # ----------------------------------------------------------- conclusions

    def add_conclusion(self, kind: str, text: str, sources: list[str] | tuple[str, ...] = ()) -> None:
        self.conclusions.append(AgentConclusion(time.time(), kind, text.strip(), tuple(sources)))

    def conclusions_text(self, limit: int = 6) -> str:
        return "\n".join(
            f"- （{item.kind}）{item.text}" for item in self.conclusions[-limit:]
        )

    # -------------------------------------------------------------- context

    def older_transcript(self, now: float, max_chars: int = 6_000) -> str:
        """Verbatim lines before the recent window, newest kept when over budget."""
        older = self.attendee_entries(until=now - self.recent_seconds)
        kept: list[str] = []
        used = 0
        for entry in reversed(older):
            line = self.format_line(entry)
            if used + len(line) + 1 > max_chars:
                break
            kept.append(line)
            used += len(line) + 1
        return "\n".join(reversed(kept))

    async def analysis_context(self, now: float | None = None) -> str:
        """Context block for one Analyst request: digest, prior conclusions, recent lines.

        If no digest exists (no Gemini key, or the digest call failed), older
        discussion is included verbatim up to a size budget instead of dropped.
        """
        now = time.time() if now is None else now
        try:
            await self.refresh_digest(now)
        except Exception:
            pass  # A digest failure must not block the analysis itself.
        parts: list[str] = []
        if self.digest:
            parts.append("【较早讨论摘要】\n" + self.digest)
        else:
            older = self.older_transcript(now)
            if older:
                parts.append("【较早讨论（原文）】\n" + older)
        if self.conclusions:
            parts.append("【Agent 已给出的结论】\n" + self.conclusions_text())
        recent = self.recent_transcript(now=now)
        parts.append("【最近逐字稿（带说话人）】\n" + (recent or "（最近没有识别到发言）"))
        return "\n\n".join(parts)

    def live_updates_since(self, cursor: float) -> tuple[str, float]:
        """New attendee lines for the Live session, and the cursor to pass next time."""
        fresh = self.attendee_entries(since=cursor)
        if not fresh:
            return "", cursor
        text = "\n".join(self.format_line(entry) for entry in fresh)
        return text, float(fresh[-1]["timestamp"])


async def gemini_digest(previous: str, new_lines: str) -> str:
    """Update the rolling meeting digest with Gemini Flash; never invent decisions."""
    import os

    from google import genai

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    client = genai.Client(api_key=api_key)
    prompt = (
        "你在维护一场会议的工作记忆。把“新逐字稿”合并进“已有摘要”，输出更新后的完整摘要，"
        "用与会者使用的语言，分为：议题；各说话人的主要主张（保留姓名和时间戳）；共识与分歧；"
        "待决问题；值得核查的事实陈述。只依据原文，不推断未说出的决议，控制在 400 字以内。\n\n"
        f"已有摘要：\n{previous or '（无）'}\n\n新逐字稿：\n{new_lines}"
    )
    response = await client.aio.models.generate_content(
        model="gemini-3.8-flash", contents=prompt,
        config={"automatic_function_calling": {"disable": True}},
    )
    return response.text or previous

"""Decide whether an utterance is addressed to the Agent (PLAN.md Phase 8 P2).

Two stages keep this cheap:

1. A local prefilter: the utterance mentions one of the Agent's names, or the
   follow-up window after an Agent answer is still open. Everything else is
   ignored without any API call.
2. For prefiltered utterances, Gemini judges intent ("Agent, what do you think?"
   vs. "did we already ask the agent?"). Without a key, or if the call fails,
   local rules decide instead.
"""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass
from typing import Awaitable, Callable, Literal

AddressKind = Literal["call", "followup", "mention", "none"]

# Question / request cues used by the local fallback rules (zh / en / de).
_QUESTION_CUES = re.compile(
    r"[?？]|你怎么看|你觉得|你认为|吗|呢|能不能|可不可以|请你|帮我|帮我们|告诉我|解释|"
    r"\b(what|why|how|who|when|where|which|can you|could you|do you|tell me|explain)\b|"
    r"\b(was|warum|wie|wer|wann|wo|kannst du|meinst du|erklär)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class AddressDecision:
    addressed: bool
    kind: AddressKind
    source: Literal["rules", "classifier", "prefilter"]


# (names, recent context, utterance, follow-up window open) -> (addressed, kind)
ClassifyFunction = Callable[[tuple[str, ...], str, str, bool], Awaitable[tuple[bool, AddressKind]]]


class AddressDetector:
    """Prefilter by name / follow-up window, then classify intent."""

    def __init__(
        self, names: tuple[str, ...], followup_seconds: float = 25.0,
        classify: ClassifyFunction | None = None,
        classify_timeout: float = 3.0,
    ) -> None:
        self.names = names
        self.followup_seconds = followup_seconds
        self.classify = classify
        # Measured 2–4 s per call with an occasional 29 s outlier; past this
        # the local rules answer instead so a reply is never held up for long.
        self.classify_timeout = classify_timeout
        self._agent_replied_at: float | None = None
        # ASCII names must not sit inside a longer Latin word ("agents" is fine to
        # skip, "Agent你怎么看" must match: CJK counts as \w, so \b fails there).
        parts = [
            rf"(?<![A-Za-z]){re.escape(name)}(?![A-Za-z])" if name.isascii() else re.escape(name)
            for name in sorted(names, key=len, reverse=True)
        ]
        self._name_pattern = re.compile("|".join(parts), re.IGNORECASE)

    def note_agent_reply(self, now: float | None = None) -> None:
        """Open the follow-up window; call when the Agent finishes an answer."""
        self._agent_replied_at = time.time() if now is None else now

    def in_followup_window(self, now: float | None = None) -> bool:
        if self._agent_replied_at is None:
            return False
        now = time.time() if now is None else now
        return now - self._agent_replied_at <= self.followup_seconds

    def mentions_name(self, text: str) -> bool:
        return self._name_pattern.search(text) is not None

    def rules(self, text: str, window_open: bool) -> tuple[bool, AddressKind]:
        """Local fallback: a name plus a question/request, or a question in the window."""
        asks = _QUESTION_CUES.search(text) is not None
        match = self._name_pattern.search(text)
        if match:
            # Vocative use: the name opens the sentence ("Agent, …", "嘿 助手 …")
            # or is directly followed by "you" / a comma ("…, Agent 你怎么看").
            # "我们之前问过 Agent 吗" has a name and a question but is only a mention.
            before = re.sub(r"^\W*(hey|hi|嘿|喂|那)?\W*", "", text[:match.start()], flags=re.IGNORECASE)
            after = text[match.end():].lstrip()
            vocative = before == "" or re.match(r"[,，:：、]|你|您|you\b|du\b|bitte\b", after, re.IGNORECASE)
            return (True, "call") if asks and vocative else (False, "mention")
        if window_open and asks:
            return True, "followup"
        return False, "none"

    async def decide(self, text: str, context: str = "", now: float | None = None) -> AddressDecision:
        window_open = self.in_followup_window(now)
        if not self.mentions_name(text) and not window_open:
            return AddressDecision(False, "none", "prefilter")
        addressed, kind = self.rules(text, window_open)
        # A clear vocative call ("Agent，你怎么看") needs no second opinion; only
        # ambiguous mentions and follow-up candidates pay the classifier latency.
        if kind == "call" or self.classify is None:
            return AddressDecision(addressed, kind, "rules")
        try:
            addressed, kind = await asyncio.wait_for(
                self.classify(self.names, context, text, window_open), self.classify_timeout,
            )
            return AddressDecision(addressed, kind, "classifier")
        except Exception:
            # Timeout, network or quota problems must not block the meeting.
            return AddressDecision(*self.rules(text, window_open), "rules")


async def gemini_classify(
    names: tuple[str, ...], context: str, utterance: str, window_open: bool,
) -> tuple[bool, AddressKind]:
    """Ask Gemini Flash whether the utterance is directed at the Agent."""
    import json
    import os

    from google import genai

    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
    prompt = (
        f"会议里有一个 AI 助手，与会者叫它：{'、'.join(names)}。语音识别可能把名字识别得不完全准确。"
        "判断“当前发言”是不是在直接对这个助手说话、要它回答或做事。\n"
        "- call：直接称呼助手并提问或提出请求（如“Agent，你怎么看”“助手帮我们查一下”）。\n"
        "- followup：没有称呼，但助手刚回答过，这句明显是在接着问助手。\n"
        "- mention：只是提到助手，并不是在问它（如“我们之前问过 Agent 吗”“让助手先别说”）。\n"
        "- none：与助手无关。\n"
        f"助手刚回答过、追问窗口{'开启' if window_open else '未开启'}。\n\n"
        f"最近对话：\n{context or '（无）'}\n\n当前发言：{utterance}"
    )
    schema = {
        "type": "OBJECT",
        "properties": {"kind": {"type": "STRING", "enum": ["call", "followup", "mention", "none"]}},
        "required": ["kind"],
    }
    response = await client.aio.models.generate_content(
        model="gemini-3.8-flash", contents=prompt,
        config={
            "response_mime_type": "application/json", "response_schema": schema,
            "automatic_function_calling": {"disable": True},
        },
    )
    kind = json.loads(response.text or "{}").get("kind", "none")
    if kind not in {"call", "followup", "mention", "none"}:
        kind = "none"
    return kind in {"call", "followup"}, kind

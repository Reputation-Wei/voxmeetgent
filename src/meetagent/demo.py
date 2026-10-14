"""Wiring for the intentionally small Phase D demo."""

import asyncio
import json
import os
import platform
from dataclasses import replace
import re
from difflib import SequenceMatcher
import signal
import shutil
import threading
import time
import wave
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from meetagent.agent.analyst import GeminiAnalyst
from meetagent.agent.intervention import InterventionPolicy
from meetagent.agent.addressing import AddressDetector, gemini_classify
from meetagent.agent.meeting_state import MeetingState, gemini_digest
from meetagent.agent.summarizer import local_fallback_summary, persist_summary, summarize_with_gemini
from rich.console import Console
from rich.markdown import Markdown
from meetagent.asr.local import MODEL_ID, create_local_backend
from meetagent.audio.aec import PlaybackReference, WebRtcEchoCanceller
from meetagent.audio.vad import SileroSpeechSegmenter
from meetagent.config import AppConfig, LiveConfig
from meetagent.diarization.sherpa import SherpaSpeakerIdentifier
from meetagent.storage.jsonl import JsonlTranscriptStore

console = Console()
SAMPLE_RATE = 16_000
LIVE_MODEL_ID = "gemini-3.8-live"
LIVE_SAMPLE_RATE = 16_000
PLAYBACK_SAMPLE_RATE = 24_000
AEC_DEBUG_MAX_BLOCKS = 3 * 60 * SAMPLE_RATE // 512  # about 3 minutes of 512-sample blocks


class _PlaybackBuffer:
    """Thread-safe PCM queue; local speech clears it before the server reply arrives."""

    def __init__(self, echo_reference: PlaybackReference | None = None, gain: float = 1.0) -> None:
        self._gain = gain
        self._chunks: deque[np.ndarray] = deque()
        self._current: np.ndarray | None = None
        self._offset = 0
        self._lock = threading.Lock()
        self._speaking = False
        self._local_speech_latched = False
        self._response_authorized = False
        self._echo_reference = echo_reference
        # PortAudio status flags; an underflow plays silence the AEC
        # reference does not contain, shifting echo vs. reference.
        self.output_underflows = 0

    def add(self, pcm_bytes: bytes) -> None:
        pcm = np.frombuffer(pcm_bytes, dtype="<i2").astype(np.float32) * (self._gain / 32768.0)
        if pcm.size:
            with self._lock:
                if self._response_authorized:
                    self._chunks.append(pcm)

    def authorize_response(self) -> None:
        """Allow playback only after Live heard an explicit direct address."""
        with self._lock:
            self._response_authorized = True

    def begin_response(self) -> None:
        """Start a fresh server turn without carrying authorization forward."""
        with self._lock:
            self._response_authorized = False
            self._clear_locked()

    def finish_response(self) -> None:
        with self._lock:
            self._response_authorized = False

    def response_authorized(self) -> bool:
        with self._lock:
            return self._response_authorized

    def is_speaking(self) -> bool:
        with self._lock:
            return self._speaking or self._current is not None or bool(self._chunks)

    def local_barge_in(self) -> bool:
        """Clear queued audio once for a continuous speech burst."""
        with self._lock:
            if self._local_speech_latched:
                return False
            self._local_speech_latched = True
            self._response_authorized = False
            self._clear_locked()
            return True

    def rearm_after_silence(self) -> None:
        with self._lock:
            self._local_speech_latched = False

    def clear_server_turn(self) -> None:
        with self._lock:
            self._response_authorized = False
            self._clear_locked()

    def _clear_locked(self) -> None:
        self._chunks.clear()
        self._current = None
        self._offset = 0
        self._speaking = False

    def callback(self, outdata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
        if status and status.output_underflow:
            self.output_underflows += 1
        outdata.fill(0)
        written = 0
        with self._lock:
            while written < frames:
                if self._current is None or self._offset >= len(self._current):
                    if not self._chunks:
                        self._current = None
                        break
                    self._current = self._chunks.popleft()
                    self._offset = 0
                count = min(frames - written, len(self._current) - self._offset)
                outdata[written : written + count, 0] = self._current[self._offset : self._offset + count]
                self._offset += count
                written += count
            self._speaking = written > 0
        if self._echo_reference is not None:
            # The AEC reference must match what the output device was given,
            # including silence while no Agent response is queued.
            self._echo_reference.write_24k(outdata[:, 0])


class _BargeInDetector:
    """Decide when mic energy during Agent playback is a real interruption.

    Laptop speaker echo reaches the level of real speech (measured RMS 0.13
    vs. 0.10–0.42 on 2026-10-01), so a single loud frame is not enough: the
    level must stay above `threshold` for `hold_seconds` while the Agent is
    speaking. After firing, it re-arms once the mic is quiet again.
    """

    def __init__(
        self, threshold: float, hold_seconds: float, quiet_threshold: float,
        rearm_seconds: float = 0.35,
    ) -> None:
        self.threshold = threshold
        self.hold_seconds = hold_seconds
        self.quiet_threshold = quiet_threshold
        self.rearm_seconds = rearm_seconds
        self.latched = False
        self._loud_since: float | None = None
        self._quiet_since: float | None = None

    def update(self, rms: float, now: float, agent_speaking: bool) -> str | None:
        """Return "barge_in" when playback should stop, "rearm" when it re-arms."""
        if self.latched:
            if rms >= self.quiet_threshold:
                self._quiet_since = None
                return None
            self._quiet_since = self._quiet_since or now
            if now - self._quiet_since >= self.rearm_seconds:
                self.latched = False
                self._quiet_since = None
                return "rearm"
            return None
        if not agent_speaking or rms < self.threshold:
            self._loud_since = None
            return None
        self._loud_since = self._loud_since or now
        if now - self._loud_since >= self.hold_seconds:
            self.latched = True
            self._loud_since = None
            return "barge_in"
        return None


class _LiveCallGate:
    """Carry one call's playback authorization across Live's turn events.

    Live hears the call in the audio stream too and may already be generating
    an (unauthorized) reply when the locally confirmed call text arrives. The
    text then makes the server interrupt that reply. Clearing the whole
    authorization on that `interrupted` dropped the real answer, so:
    - an `interrupted` within GRACE seconds of sending the call only clears audio;
    - a `turn_complete` before any authorized reply audio does not end the grant;
    - a grant with no reply audio after EXPIRY seconds is revoked, so later
      unrequested speech can never use it.
    """

    GRACE_SECONDS = 3.0
    EXPIRY_SECONDS = 15.0

    def __init__(self, playback: _PlaybackBuffer, clock: Any = time.monotonic) -> None:
        self._playback = playback
        self._clock = clock
        self._sent_at: float | None = None
        self._reply_started = False

    @property
    def waiting(self) -> bool:
        return self._sent_at is not None

    def call_sent(self) -> None:
        self._sent_at = self._clock()
        self._reply_started = False
        self._playback.begin_response()
        self._playback.authorize_response()

    def on_audio(self, pcm: bytes) -> None:
        if (
            self._sent_at is not None and not self._reply_started
            and self._clock() - self._sent_at > self.EXPIRY_SECONDS
        ):
            self._sent_at = None
            self._playback.finish_response()
        self._playback.add(pcm)
        if self._sent_at is not None and self._playback.response_authorized():
            self._reply_started = True

    def on_interrupted(self) -> bool:
        """Return True if this interruption was caused by our own call text."""
        self._playback.clear_server_turn()
        if self._sent_at is not None and self._clock() - self._sent_at < self.GRACE_SECONDS:
            self._playback.authorize_response()
            self._reply_started = False
            return True
        self._sent_at = None
        return False

    def on_turn_complete(self) -> bool:
        """Return True when this turn completed the reply to a call."""
        if self._sent_at is not None and not self._reply_started:
            return False  # an earlier turn ended; the reply to the call is still coming
        replied = self._sent_at is not None
        self._sent_at = None
        self._playback.finish_response()
        return replied


def _contains_authorized_call(text: str) -> bool:
    """Accept the explicit Agent call phrase, not incidental Agent mentions."""
    lower = text.casefold()
    # Whisper in zh mode often mishears "Agent"; 助手 is the reliable Chinese call.
    call_phrases = (
        r"(?:hey\s*)?(?:agent|助手)[\s,，:：、]*你怎么看",
        r"(?:hey\s*)?agent[\s,，:：、]*what\s+do\s+you\s+think",
        r"(?:hey\s*)?agent[\s,，:：、]*was\s+meinst\s+du",
    )
    return any(re.search(phrase, lower) is not None for phrase in call_phrases)


def _load_keywords() -> dict[str, list[str]]:
    path = Path(__file__).with_name("keywords.yaml")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _language_hint(text: str) -> str:
    """Return a lightweight display hint; Whisper owns actual language detection."""
    lower = text.lower()
    if re.search(r"\b(der|die|das|und|ist|gibt|forschung|nicht)\b", lower):
        return "de"
    if re.search(r"\b(the|and|is|are|research|does|not)\b", lower):
        return "en"
    return "zh"


def _trigger(text: str, keywords: dict[str, list[str]]) -> str | None:
    lower = text.lower()
    # Wake words are addressing cues, not analyst requests on their own.
    for kind, phrases in keywords.items():
        if kind == "wake":
            continue
        if any(phrase.lower() in lower for phrase in phrases):
            return kind
    return None


def _transcribe(audio: np.ndarray, language: str = "auto") -> tuple[str, str]:
    """Compatibility wrapper around the reusable local backend."""
    return create_local_backend().transcribe(audio, language)


async def _gemini_client() -> Any:
    try:
        from google import genai
    except ImportError as exc:
        raise RuntimeError("缺少 google-genai，请执行 `uv sync`。") from exc
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("未设置 GEMINI_API_KEY 或 GOOGLE_API_KEY")
    return genai.Client(api_key=api_key)


async def _answer(
    transcript: str, trigger_kind: str, request_text: str = "",
) -> tuple[str, list[str], list[dict[str, str]], dict[str, Any]]:
    """Return an analysis grounded in Google Search and scholarly records."""
    from google.genai import types

    client = await _gemini_client()
    papers: list[dict[str, str]] = []
    paper_context = ""
    if trigger_kind == "research":
        from meetagent.tools.scholar import CompositeScholarSearch

        query = request_text.strip() or transcript[-500:]
        try:
            papers = await CompositeScholarSearch().search(query, limit=8)
        except Exception as exc:
            console.print(f"[yellow]论文索引暂不可用：{exc}[/yellow]")
        paper_context = "\n真实论文候选（只能引用以下确实返回的论文，不能补造标题或链接）：\n" + "\n".join(
            f"- {item['title']} ({item['year'] or '年份未知'}), {item['source']}, {item['url']}"
            for item in papers
        )
    if trigger_kind == "fact_check":
        task = (
            "核查最近发言中最重要、可验证的事实陈述。按 Claim / Verdict / Confidence / Evidence 输出。"
            "Verdict 使用 supported、contradicted、mixed、unverified 之一；区分证据和推断，"
            "Confidence 是 0 到 1 的数值；只有搜索结果实际支持时才下结论，说明反证或不确定性。"
            "数字类陈述（数量、规模、日期、比例、金额等）按以下规则判定："
            "1) 在 stated_value 写出发言者说的具体数值，在 reference_value 写出可靠来源给出的数值及其口径；"
            "2) 只把发言者的数值与来源数值直接比较，不得用“同一数量级”“千万级别”“大致相当”等放宽口径来判定属实；"
            "3) 发言者用“大约/左右”时，与来源相差在 10% 以内才算 supported；未加限定词时按原数值比较；"
            "4) 相差超过 10% 判 contradicted；对象有多个常见口径（如完整数据集与常用子集）且发言者数值与任何一个口径都不相符时，也判 contradicted，并在 evidence 中列出各口径的数值；"
            "只有发言者数值符合其中一个口径而另一口径不符时，才判 mixed；"
            "5) spoken 中要说出正确数值。"
            "逐字稿中明显的语音识别错误或与讨论无关的句子不要当作发言者陈述来核查。"
        )
    elif trigger_kind == "logic_check":
        task = (
            "检查最近发言中的前后矛盾、因果倒置、以偏概全或概念变化。只报告有原话依据的问题，"
            "逐条引用相关原句及时间戳，解释推理缺口；若没有足够证据，明确说未发现可确认的问题。"
        )
    elif trigger_kind == "research":
        task = (
            "根据讨论内容提出最多 3 个有排序的研究方向。每条写研究问题、已有工作与差异、可行性、风险和排序理由。"
            "相关工作只能引用所给的真实论文候选，并附准确链接；没有候选论文时明确说明检索未找到依据，不能编论文。"
        )
    elif trigger_kind == "claim_extract":
        task = (
            "从会议窗口中抽取最多 8 条可验证的事实陈述；排除意见、计划和纯粹假设。"
            "保留原话含义，并给出发言者、时间戳和为什么可以核查。不要判断真伪，不要给建议。"
        )
    elif trigger_kind == "summarize":
        task = (
            "总结到目前为止的讨论：议题、每位说话人的主要观点（写出姓名）、共识、分歧和待决问题。"
            "只依据会议上下文，不推断未明确说出的决议或负责人。"
        )
    else:
        task = "结合会议上下文简洁回答触发者的问题。"
    prompt = (
        "你是参与会议讨论的谨慎研究助理。使用触发者原本的语言。只依据会议上下文、Google Search grounding 和论文候选作答；"
        "保留事实与推断的区别，不确定时明确说明。指出问题时说明是谁在什么时间说的。"
        "spoken 字段写一到三句适合在会议中直接说出口的结论，不要念链接或 Markdown。\n"
        f"任务：{task}\n触发类型：{trigger_kind}\n触发者原话：{request_text}\n"
        f"会议上下文：\n{transcript}{paper_context}"
    )
    if trigger_kind == "fact_check":
        schema: dict[str, Any] = {
            "type": "OBJECT", "properties": {
                "findings": {"type": "ARRAY", "items": {"type": "OBJECT", "properties": {
                    "claim": {"type": "STRING"},
                    "verdict": {"type": "STRING", "enum": ["supported", "contradicted", "mixed", "unverified"]},
                    "confidence": {"type": "NUMBER"},
                    "evidence": {"type": "STRING"},
                    # Numeric claims: what was said vs. what sources say.
                    "stated_value": {"type": "STRING"},
                    "reference_value": {"type": "STRING"},
                }, "required": ["claim", "verdict", "confidence", "evidence"]}},
            }, "required": ["findings"],
        }
    elif trigger_kind == "logic_check":
        schema = {
            "type": "OBJECT", "properties": {
                "findings": {"type": "ARRAY", "items": {"type": "OBJECT", "properties": {
                    "quote": {"type": "STRING"}, "timestamps": {"type": "ARRAY", "items": {"type": "STRING"}},
                    "issue_type": {"type": "STRING"}, "explanation": {"type": "STRING"},
                }, "required": ["quote", "timestamps", "issue_type", "explanation"]}},
            }, "required": ["findings"],
        }
    elif trigger_kind == "research":
        schema = {
            "type": "OBJECT", "properties": {
                "directions": {"type": "ARRAY", "items": {"type": "OBJECT", "properties": {
                    "research_question": {"type": "STRING"},
                    "novelty_basis": {"type": "STRING"},
                    "paper_titles": {"type": "ARRAY", "items": {"type": "STRING"}},
                    "feasibility": {"type": "STRING"}, "risks": {"type": "STRING"},
                    "ranking_reason": {"type": "STRING"},
                }, "required": ["research_question", "novelty_basis", "paper_titles", "feasibility", "risks", "ranking_reason"]}},
            }, "required": ["directions"],
        }
    elif trigger_kind == "claim_extract":
        schema = {
            "type": "OBJECT", "properties": {
                "claims": {"type": "ARRAY", "items": {"type": "OBJECT", "properties": {
                    "claim": {"type": "STRING"}, "speaker": {"type": "STRING"},
                    "timestamp": {"type": "STRING"}, "why_verifiable": {"type": "STRING"},
                }, "required": ["claim", "speaker", "timestamp", "why_verifiable"]}},
            }, "required": ["claims"],
        }
    elif trigger_kind == "summarize":
        schema = {
            "type": "OBJECT", "properties": {
                "topics": {"type": "ARRAY", "items": {"type": "STRING"}},
                "positions": {"type": "ARRAY", "items": {"type": "OBJECT", "properties": {
                    "speaker": {"type": "STRING"}, "position": {"type": "STRING"},
                }, "required": ["speaker", "position"]}},
                "agreements": {"type": "ARRAY", "items": {"type": "STRING"}},
                "disagreements": {"type": "ARRAY", "items": {"type": "STRING"}},
                "open_questions": {"type": "ARRAY", "items": {"type": "STRING"}},
            }, "required": ["topics", "positions", "agreements", "disagreements", "open_questions"],
        }
    else:
        schema = {"type": "OBJECT", "properties": {"answer": {"type": "STRING"}}, "required": ["answer"]}
    # Every spoken analysis also returns a short conversational version to say aloud.
    if trigger_kind != "claim_extract":
        schema["properties"]["spoken"] = {"type": "STRING"}
        schema["required"] = [*schema["required"], "spoken"]

    config_values: dict[str, Any] = {
        "response_mime_type": "application/json",
        "response_schema": schema,
    }
    if trigger_kind in {"fact_check", "research", "ask"}:
        config_values["tools"] = [types.Tool(google_search=types.GoogleSearch())]
    # No client-side function calling (search grounding is a built-in tool).
    # Disabling it also silences the SDK's AFC warning on every call.
    config_values["automatic_function_calling"] = types.AutomaticFunctionCallingConfig(disable=True)
    response = await client.aio.models.generate_content(
        model="gemini-3.8-flash", contents=prompt,
        config=types.GenerateContentConfig(**config_values),
    )
    raw_response = response.text or "{}"
    try:
        details = json.loads(raw_response)
    except json.JSONDecodeError:
        details = {}
    links: list[str] = []
    titles: list[str] = []
    candidates = getattr(response, "candidates", None) or []
    for candidate in candidates:
        metadata = getattr(candidate, "grounding_metadata", None)
        for chunk in (getattr(metadata, "grounding_chunks", None) or []):
            web = getattr(chunk, "web", None)
            uri = getattr(web, "uri", None)
            if uri and uri not in links:
                links.append(uri)
                # Grounding URIs are long redirects; the title is the site name.
                titles.append(str(getattr(web, "title", "") or ""))
    if trigger_kind == "fact_check":
        lines = ["## 事实核查"]
        for finding in details.get("findings", []):
            try:
                confidence = float(finding.get("confidence", 0))
                confidence = confidence / 100 if confidence > 1 else confidence
            except (TypeError, ValueError):
                confidence = 0.0
            lines.extend([
                f"- **Claim**：{finding.get('claim', '')}",
                f"  - Verdict：{finding.get('verdict', 'unverified')}",
                f"  - Confidence：{confidence:.0%}",
                f"  - Evidence：{finding.get('evidence', '')}",
            ])
            if finding.get("stated_value") or finding.get("reference_value"):
                lines.append(
                    f"  - 数值：发言 {finding.get('stated_value') or '—'}；"
                    f"来源 {finding.get('reference_value') or '—'}"
                )
        answer = "\n".join(lines) if len(lines) > 1 else (response.text or "没有可核查的事实陈述。")
    elif trigger_kind == "logic_check":
        lines = ["## 逻辑检查"]
        for finding in details.get("findings", []):
            timestamps = "、".join(finding.get("timestamps", [])) or "时间戳未提供"
            lines.extend([
                f"- **{finding.get('issue_type', '待核实')}**（{timestamps}）：{finding.get('quote', '')}",
                f"  - {finding.get('explanation', '')}",
            ])
        answer = "\n".join(lines) if len(lines) > 1 else (response.text or "未发现可确认的逻辑问题。")
    elif trigger_kind == "research":
        by_title = {re.sub(r"\W+", "", item["title"].casefold()): item for item in papers}
        lines = ["## 排序后的研究方向"]
        for index, direction in enumerate(details.get("directions", []), start=1):
            lines.extend([
                f"### {index}. {direction.get('research_question', '')}",
                f"- 新颖性依据：{direction.get('novelty_basis', '')}",
                f"- 可行性：{direction.get('feasibility', '')}",
                f"- 风险：{direction.get('risks', '')}",
                f"- 排序理由：{direction.get('ranking_reason', '')}",
                "- 可核对的相关工作：",
            ])
            matched = False
            for title in direction.get("paper_titles", []):
                item = by_title.get(re.sub(r"\W+", "", title.casefold()))
                if item:
                    lines.append(f"  - [{item['title']}]({item['url']}) ({item['year'] or '年份未知'})")
                    direction.setdefault("paper_sources", []).append(item["url"])
                    if item["url"] not in links:
                        links.append(item["url"])
                    matched = True
            if not matched:
                lines.append("  - 检索候选中没有与本方向匹配的论文；不列出未核实文献。")
        answer = "\n".join(lines) if len(lines) > 1 else (response.text or "没有生成研究方向。")
    elif trigger_kind == "claim_extract":
        claims = details.get("claims", [])
        lines = ["## 可核查陈述（后台静默分析）"]
        lines.extend(
            f"- [{item.get('timestamp', '时间未提供')}] {item.get('speaker', 'Speaker')}: "
            f"{item.get('claim', '')}（核查点：{item.get('why_verifiable', '')}）"
            for item in claims
        )
        answer = "\n".join(lines) if claims else "当前窗口没有抽取到可核查陈述。"
    elif trigger_kind == "summarize":
        lines = ["## 讨论总结"]
        sections = (
            ("议题", details.get("topics", [])),
            ("各方观点", [f"{item.get('speaker', '说话人未知')}：{item.get('position', '')}"
                       for item in details.get("positions", [])]),
            ("共识", details.get("agreements", [])),
            ("分歧", details.get("disagreements", [])),
            ("待决问题", details.get("open_questions", [])),
        )
        for title, items in sections:
            if items:
                lines.append(f"### {title}")
                lines.extend(f"- {item}" for item in items)
        answer = "\n".join(lines) if len(lines) > 1 else (response.text or "目前还没有可总结的讨论。")
    else:
        answer = str(details.get("answer") or response.text or "没有生成文本回答。")
    details["source_urls"] = links.copy()
    details["source_titles"] = titles + [""] * (len(links) - len(titles))
    return answer, links, papers, details


async def _speak(answer: str, playback: _PlaybackBuffer | None = None) -> None:
    """Ask Gemini Live to speak a supplied answer, and play returned PCM."""
    from google.genai import types

    client = await _gemini_client()
    try:
        import sounddevice as sd
    except ImportError as exc:
        raise RuntimeError("缺少 sounddevice，请执行 `uv sync`。") from exc

    audio_parts: list[bytes] = []
    async with client.aio.live.connect(
        model="gemini-3.8-live",
        config={"response_modalities": ["AUDIO"]},
    ) as session:
        # role="user" is required; without it Live rejects the turn with 1007.
        await session.send_client_content(
            turns=types.Content(role="user", parts=[types.Part(
                text="请用自然、简短的口语朗读下面这段回答，不要增加内容：\n" + answer,
            )]),
            turn_complete=True,
        )
        streaming = False
        try:
            async for response in session.receive():
                if response.data:
                    if playback is not None:
                        # Stream into the speaker as chunks arrive instead of
                        # waiting for the whole reply (~1 s vs ~10 s to first sound).
                        if not streaming:
                            playback.authorize_response()
                            streaming = True
                        playback.add(response.data)
                    else:
                        audio_parts.append(response.data)
                content = response.server_content
                if content and content.turn_complete:
                    break
        finally:
            if streaming:
                playback.finish_response()
    # Without a playback buffer (file mode), play the collected reply at once.
    pcm = np.frombuffer(b"".join(audio_parts), dtype=np.int16)
    if pcm.size:
        await asyncio.to_thread(sd.play, pcm, samplerate=PLAYBACK_SAMPLE_RATE, blocking=True)


async def _run_live_voice(
    microphone: asyncio.Queue[bytes], live_calls: asyncio.Queue[str], playback: _PlaybackBuffer,
    stop_requested: asyncio.Event, ready: asyncio.Event,
    entries: list[dict[str, Any]], transcript_path: Path, meeting_started: float,
    policy: InterventionPolicy,
    meeting_state: MeetingState | None = None,
    addressing: AddressDetector | None = None,
) -> None:
    """Keep Gemini Live listening and playing only explicitly requested replies."""
    from google.genai import types

    client = await _gemini_client()
    instruction = (
        "你是会议中的中英德语音助手。整场会议都认真倾听并记住上下文。除非听到明确呼叫短语‘Agent，你怎么看？’（或对应语言的‘Hey Agent, what do you think?’、‘Agent, was meinst du?’），否则必须保持静默，不要生成语音、提示音或主动总结。提及 Agent、提出普通问题、事实核查关键词都不构成呼叫。"
        "如果用户在会议中提到 related work、相关研究、fact check、事实核查或要求来源，保持安静；应用中的研究核查服务会单独处理并播报带来源的结果。"
        "回答使用提问者的中文、英语或德语，尽量简短。有人插话时立即停下并听对方说完。"
        "你会不时收到以“[会议逐字稿更新]”或“[Agent 分析结论]”开头的文字：前者是本地高精度识别的带说话人发言，"
        "后者是你已通过搜索得出的核查/研究结论。它们只是背景信息，不是提问，收到时不要回应；"
        "被呼叫回答时，用它们说明是谁提出了什么观点。"
    )
    if policy.config.intervention == "silent":
        instruction += "本次介入模式为 silent：所有发言都保持静默，只接收上下文，不要输出语音或答案。"
    resumption_handle: str | None = None
    # Shared across reconnects: resumed sessions keep earlier context.
    transcript_cursor = 0.0
    conclusions_sent = 0
    reconnects = 0
    call_gate = _LiveCallGate(playback)
    agent_text_parts: list[str] = []
    pending_fallbacks: list[asyncio.Task[None]] = []

    def save_agent_turn() -> None:
        if agent_text_parts:
            _record_agent_utterance(
                "".join(agent_text_parts), entries, transcript_path, meeting_started,
            )
            agent_text_parts.clear()

    async def one_connection() -> bool:
        """Return True when server asks us to rotate/resume the session."""
        nonlocal resumption_handle
        rotate = asyncio.Event()
        config = types.LiveConnectConfig(
            response_modalities=[types.Modality.AUDIO],
            system_instruction=types.Content(parts=[types.Part(text=instruction)]),
            input_audio_transcription=types.AudioTranscriptionConfig(),
            output_audio_transcription=types.AudioTranscriptionConfig(),
            context_window_compression=types.ContextWindowCompressionConfig(
                sliding_window=types.SlidingWindow(),
            ),
            session_resumption=types.SessionResumptionConfig(handle=resumption_handle),
        )
        async with client.aio.live.connect(model=LIVE_MODEL_ID, config=config) as session:
            ready.set()

            async def share_meeting_context() -> None:
                """Inject new speaker-labeled lines and conclusions without asking for a reply."""
                nonlocal transcript_cursor, conclusions_sent
                if meeting_state is None:
                    return
                blocks: list[str] = []
                lines, transcript_cursor = meeting_state.live_updates_since(transcript_cursor)
                if lines:
                    blocks.append("[会议逐字稿更新]\n" + lines)
                fresh = meeting_state.conclusions[conclusions_sent:]
                if fresh:
                    blocks.append("[Agent 分析结论]\n" + "\n".join(
                        f"- （{item.kind}）{item.text}" for item in fresh
                    ))
                    conclusions_sent += len(fresh)
                if blocks:
                    # turn_complete=False adds context without triggering a model turn.
                    await session.send_client_content(
                        turns=[{"role": "user", "parts": [{"text": "\n\n".join(blocks)}]}],
                        turn_complete=False,
                    )

            async def send_microphone() -> None:
                last_context_push = 0.0
                while not stop_requested.is_set() and not rotate.is_set():
                    if time.monotonic() - last_context_push >= 3.0:
                        last_context_push = time.monotonic()
                        await share_meeting_context()
                    try:
                        direct_call = live_calls.get_nowait()
                    except asyncio.QueueEmpty:
                        direct_call = None
                    if direct_call is not None:
                        # Make sure the reply sees the latest lines, not a 3 s old view.
                        await share_meeting_context()
                        last_context_push = time.monotonic()
                        call_gate.call_sent()
                        await session.send_realtime_input(
                            text=(
                                "本地高精度 Whisper 已确认与会者在对 Agent 说话（点名或追问）。"
                                f"原话是：{direct_call}。请结合当前会议上下文，用原语言简短回答。"
                            ),
                        )
                    try:
                        pcm = await asyncio.wait_for(microphone.get(), timeout=0.2)
                    except TimeoutError:
                        continue
                    await session.send_realtime_input(
                        audio=types.Blob(data=pcm, mime_type=f"audio/pcm;rate={LIVE_SAMPLE_RATE}"),
                    )

            async def receive_agent() -> None:
                while not stop_requested.is_set() and not rotate.is_set():
                    received = 0
                    async for response in session.receive():
                        received += 1
                        update = response.session_resumption_update
                        if update and update.resumable and update.new_handle:
                            resumption_handle = update.new_handle
                        if response.go_away:
                            console.print(f"[dim]Live 会话即将轮换：{response.go_away.time_left}[/dim]")
                            rotate.set()
                            return

                        content = response.server_content
                        if content:
                            if content.interrupted:
                                save_agent_turn()
                                if not call_gate.on_interrupted():
                                    console.print("[dim]检测到插话，已清空 Agent 待播语音。[/dim]")
                            if content.input_transcription and content.input_transcription.text:
                                heard = content.input_transcription.text
                                console.print(f"[dim]Live 听到：{heard}[/dim]")
                                captured_at = time.time()
                                fallback_task = asyncio.create_task(
                                    _save_live_input_fallback(
                                        heard, entries, transcript_path, meeting_started, captured_at,
                                    ),
                                    name="live-input-transcript-fallback",
                                )
                                pending_fallbacks.append(fallback_task)
                            if content.output_transcription and content.output_transcription.text:
                                if playback.response_authorized():
                                    text = content.output_transcription.text
                                    agent_text_parts.append(text)
                                    console.print(f"[cyan]Agent：{text}[/cyan]", end="")
                            if content.model_turn:
                                for part in content.model_turn.parts or []:
                                    audio = getattr(getattr(part, "inline_data", None), "data", None)
                                    if audio:
                                        call_gate.on_audio(audio)
                            if content.turn_complete:
                                save_agent_turn()
                                console.print()
                                if call_gate.on_turn_complete() and addressing is not None:
                                    addressing.note_agent_reply()
                    if received == 0:
                        await asyncio.sleep(0.05)

            sender = asyncio.create_task(send_microphone(), name="live-microphone-upload")
            receiver = asyncio.create_task(receive_agent(), name="live-agent-receive")
            stop_waiter = asyncio.create_task(stop_requested.wait(), name="live-stop-waiter")
            done, pending = await asyncio.wait(
                {sender, receiver, stop_waiter}, return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            for task in done:
                if task is not stop_waiter and not task.cancelled():
                    error = task.exception()
                    if error:
                        raise error
        return rotate.is_set()

    retry_attempt = 0
    try:
        while not stop_requested.is_set():
            try:
                rotate = await one_connection()
            except Exception as exc:
                # A rejected key never fixes itself; retrying would only delay
                # local transcription while the caller waits for Live.
                if "API key not valid" in str(exc) or "PERMISSION_DENIED" in str(exc):
                    raise RuntimeError(
                        "Gemini API key 无效或无权限，请检查 GEMINI_API_KEY（运行 meetagent doctor）。"
                    ) from exc
                reconnects += 1
                retry_attempt += 1
                delay = min(30, 2 ** min(retry_attempt - 1, 5))
                console.print(
                    f"[yellow]Live 连接中断：{exc}；{delay} 秒后尝试恢复上下文（第 {reconnects} 次）…[/yellow]"
                )
                try:
                    await asyncio.wait_for(stop_requested.wait(), timeout=delay)
                except TimeoutError:
                    pass
                continue
            if not rotate or stop_requested.is_set():
                break
            reconnects += 1
            retry_attempt = 0
            console.print(f"[dim]正在恢复 Live 上下文（第 {reconnects} 次）…[/dim]")
    finally:
        if pending_fallbacks:
            await asyncio.gather(*pending_fallbacks, return_exceptions=True)
        save_agent_turn()
        playback.clear_server_turn()


def _new_transcript(out: Path, meeting_started: datetime, meeting_id: str | None = None) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    stamp = meeting_id or meeting_started.strftime("%Y%m%d-%H%M%S-%f")
    text_path = out / f"meeting-{stamp}.txt"
    text_path.touch(exist_ok=False)
    text_path.with_suffix(".jsonl").touch(exist_ok=False)
    return text_path


def _append_transcript(text_path: Path, entry: dict[str, Any]) -> None:
    """Persist each recognized utterance immediately so interruption keeps it."""
    JsonlTranscriptStore(text_path).append(entry)


def _rewrite_transcript_files(text_path: Path, entries: list[dict[str, Any]]) -> None:
    """Rewrite transcript exports after replacing a Live fallback with local ASR."""
    JsonlTranscriptStore(text_path).rewrite(entries)


def _store_ordered_entry(
    entry: dict[str, Any], entries: list[dict[str, Any]], transcript_path: Path,
) -> None:
    """Append an entry or rewrite exports if an asynchronous source arrived late."""
    last_timestamp = entries[-1]["timestamp"] if entries else None
    entries.append(entry)
    if last_timestamp is not None and entry["timestamp"] < last_timestamp:
        _rewrite_transcript_files(transcript_path, entries)
    else:
        _append_transcript(transcript_path, entry)


def _normalized_text(text: str) -> str:
    return re.sub(r"[^\w]+", "", text.casefold())


def _similar_text(left: str, right: str) -> float:
    left_normalized, right_normalized = _normalized_text(left), _normalized_text(right)
    if not left_normalized or not right_normalized:
        return 0.0
    return SequenceMatcher(None, left_normalized, right_normalized).ratio()


def _matches_recent_agent_output(text: str, entries: list[dict[str, Any]], timestamp: float) -> bool:
    """Filter likely speaker-to-microphone echo of Agent's recently recorded response."""
    normalized = _normalized_text(text)
    if len(normalized) < 8:
        return False
    for entry in reversed(entries):
        if entry.get("speaker") != "Agent" or abs(timestamp - entry.get("timestamp", 0)) > 30:
            continue
        agent_text = _normalized_text(str(entry.get("text", "")))
        if not agent_text:
            continue
        if normalized in agent_text or _similar_text(normalized, agent_text) >= 0.78:
            return True
    return False


def _store_local_utterance(
    entry: dict[str, Any], entries: list[dict[str, Any]], transcript_path: Path,
) -> None:
    """Prefer local Whisper text, replacing a matching Live-only fallback if present."""
    for existing in entries:
        if existing.get("source") != "gemini_live_fallback":
            continue
        if abs(existing["timestamp"] - entry["timestamp"]) <= 60 and _similar_text(
            existing["text"], entry["text"],
        ) >= 0.65:
            existing.update(entry)
            _rewrite_transcript_files(transcript_path, entries)
            return
    _store_ordered_entry(entry, entries, transcript_path)


async def _save_live_input_fallback(
    text: str, entries: list[dict[str, Any]], transcript_path: Path,
    meeting_started: float, captured_at: float,
) -> None:
    """Preserve Live's heard words only when local Whisper has not supplied them."""
    await asyncio.sleep(1.25)
    timestamp = captured_at
    if _matches_recent_agent_output(text, entries, timestamp):
        return
    if any(
        entry.get("source") in {"local_asr", "gemini_live_fallback"}
        and abs(entry["timestamp"] - timestamp) <= 60
        and _similar_text(entry["text"], text) >= 0.72
        for entry in entries
    ):
        return
    elapsed = max(0, int(timestamp - meeting_started))
    entry = {
        "timestamp": timestamp,
        "recorded_at": datetime.now().isoformat(timespec="seconds"),
        "time": f"{elapsed // 60:02d}:{elapsed % 60:02d}",
        "speaker": "Speaker",
        "text": text.strip(),
        "language": _language_hint(text),
        "language_mode": "live_fallback",
        "source": "gemini_live_fallback",
    }
    _store_ordered_entry(entry, entries, transcript_path)


def _record_agent_utterance(
    text: str, entries: list[dict[str, Any]], transcript_path: Path,
    meeting_started: float, source: str = "gemini_live",
) -> None:
    """Persist an Agent response in the same timestamped transcript as attendees."""
    text = text.strip()
    if not text:
        return
    timestamp = time.time()
    elapsed = max(0, int(timestamp - meeting_started))
    entry = {
        "timestamp": timestamp,
        "recorded_at": datetime.now().isoformat(timespec="seconds"),
        "time": f"{elapsed // 60:02d}:{elapsed % 60:02d}",
        "speaker": "Agent",
        "text": text,
        "language": _language_hint(text),
        "language_mode": "agent",
        "source": source,
    }
    _store_ordered_entry(entry, entries, transcript_path)


async def _write_summary(
    entries: list[dict[str, Any]], answers: list[dict[str, Any]], transcript_path: Path,
    summary_language: str = "zh", obsidian_vault: Path | None = None,
    obsidian_folder: str = "Meetings", allow_gemini: bool = True,
) -> tuple[Path, bool]:
    generated_by_gemini = False
    findings_path = transcript_path.with_suffix(".findings.jsonl")
    findings: list[dict[str, Any]] = []
    if findings_path.is_file():
        try:
            findings = [
                json.loads(line) for line in findings_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        except (OSError, json.JSONDecodeError) as exc:
            console.print(f"[yellow]读取 Finding 文件失败；摘要将不包含它：{exc}[/yellow]")
    try:
        if not allow_gemini:
            raise RuntimeError("用户未同意将逐字稿发送到 Gemini；生成本地纪要。")
        body = await summarize_with_gemini(entries, answers, summary_language, findings)
        generated_by_gemini = True
    except Exception as exc:
        console.print(f"[yellow]Gemini 纪要生成不可用（{exc}），改写入本地逐字稿纪要。[/yellow]")
        body = local_fallback_summary(entries, answers, summary_language, findings)
    path = persist_summary(
        body, transcript_path, vault=obsidian_vault, vault_folder=obsidian_folder,
    )
    return path, generated_by_gemini


_TRIGGER_TITLES = {
    "ask": "回答提问", "fact_check": "事实核查", "logic_check": "逻辑检查",
    "research": "相关研究", "summarize": "讨论总结",
}


def _append_agent_markdown(
    path: Path, elapsed_seconds: float, trigger_kind: str, request_text: str,
    answer: str, links: list[str], titles: list[str],
) -> None:
    """Append one Agent answer to <meeting>.agent.md as soon as it exists."""
    minutes, seconds = divmod(max(0, int(elapsed_seconds)), 60)
    lines = []
    if not path.exists():
        lines += [f"# Agent 回答 · {path.stem.removesuffix('.agent')}", ""]
    lines += [
        f"## [{minutes:02d}:{seconds:02d}] {_TRIGGER_TITLES.get(trigger_kind, trigger_kind)}",
        "", f"> {request_text.strip()}", "", answer.strip(), "",
    ]
    if links:
        lines.append("来源：")
        lines += [
            f"- [{(titles[i] if i < len(titles) else '') or url}]({url})" for i, url in enumerate(links)
        ]
        lines.append("")
    with path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


# One spoken answer at a time; concurrent analyses queue here before playback.
_speech_lock = asyncio.Lock()


async def _handle_trigger(
    trigger_kind: str, entries: list[dict[str, Any]], answers: list[dict[str, Any]],
    playback: _PlaybackBuffer | None = None,
    transcript_path: Path | None = None, meeting_started: float | None = None,
    request_text: str = "",
    analyst: GeminiAnalyst | None = None,
    policy: InterventionPolicy | None = None,
    directly_addressed: bool = False,
    meeting_state: MeetingState | None = None,
) -> None:
    """Analyze with full meeting context, record the answer, and speak it if allowed."""
    state = meeting_state or MeetingState(entries)
    console.print(f"\n[bold cyan]触发：{trigger_kind}，正在搜索并核查…[/bold cyan]")
    try:
        transcript = await state.analysis_context()
        if analyst is None:
            answer, links, papers, details = await _answer(transcript, trigger_kind, request_text)
            findings = []
        else:
            analysis = await analyst.respond(transcript, trigger_kind, request_text)
            answer, links, papers, details = (
                analysis.answer, analysis.sources, analysis.papers, analysis.details,
            )
            findings = analysis.findings
        console.print(Markdown(answer))
        if links:
            # Short clickable site names (OSC 8 links); full URLs stay in the transcript.
            titles = details.get("source_titles") or []
            labels = [
                f"[link={url}]{(titles[i - 1] if i - 1 < len(titles) else '') or f'[{i}]'}[/link]"
                for i, url in enumerate(links, 1)
            ]
            console.print("来源：" + "  ".join(labels))
        spoken = str(details.get("spoken") or "").strip()
        answers.append({
            "trigger": trigger_kind, "request": request_text,
            "answer": answer, "sources": links, "papers": papers, "details": details,
            "finding_count": len(findings),
        })
        state.add_conclusion(trigger_kind, spoken or answer, links)
        if transcript_path is not None and meeting_started is not None:
            _record_agent_utterance(
                answer, entries, transcript_path, meeting_started, source="analyst",
            )
            agent_markdown = transcript_path.with_suffix(".agent.md")
            _append_agent_markdown(
                agent_markdown, time.time() - meeting_started, trigger_kind, request_text,
                answer, links, list(details.get("source_titles") or []),
            )
            console.print(f"[dim]已追加到 {agent_markdown}[/dim]")
        # Keyword requests speak without a name call (PLAN.md Phase 8, decision 5).
        reason = "call" if directly_addressed else "keyword"
        may_speak = policy.may_speak(reason) if policy is not None else True
        if may_speak:
            console.print("[dim]Agent 正在语音回答…[/dim]")
            async with _speech_lock:
                await _speak(spoken or answer, playback=playback)
        if state.addressing is not None:
            state.addressing.note_agent_reply()
        elif policy is not None and policy.config.reply == "text":
            console.print("[dim]文字回复模式：回答已显示并写入逐字稿，不播报。[/dim]")
        else:
            console.print("[dim]当前介入策略为静默；回答已显示并写入逐字稿。[/dim]")
    except Exception as exc:
        console.print(f"[red]Agent 请求失败：{exc}[/red]")


async def _process_audio(
    audio: np.ndarray, start_seconds: float, entries: list[dict[str, Any]],
    answers: list[dict[str, Any]], keywords: dict[str, list[str]], meeting_started: float,
    transcript_path: Path, language: str,
    playback: _PlaybackBuffer | None = None,
    run_triggers: bool = True,
    speaker_identifier: SherpaSpeakerIdentifier | None = None,
    analyst: GeminiAnalyst | None = None,
    policy: InterventionPolicy | None = None,
    live_calls: asyncio.Queue[str] | None = None,
    meeting_state: MeetingState | None = None,
    trigger_tasks: set[asyncio.Task[None]] | None = None,
) -> None:
    """Transcribe one VAD segment, store it, and dispatch any Agent trigger.

    With `trigger_tasks`, analyses run in the background so local ASR keeps up
    during web searches; without it (offline files) they run inline.
    """
    try:
        text, detected_language = await asyncio.to_thread(_transcribe, audio, language)
    except Exception as exc:
        console.print(f"[yellow]本段 Whisper 转写失败；原始音频仍保留在 WAV：{exc}[/yellow]")
        return
    if not text:
        return
    now = meeting_started + start_seconds
    if _matches_recent_agent_output(text, entries, now):
        console.print("[dim]检测到 Agent 播报回声，未重复记为与会者发言。[/dim]")
        return
    offset = int(start_seconds)
    speaker = "说话人未知"
    speaker_confirmed = False
    if speaker_identifier is not None:
        try:
            identified = await asyncio.to_thread(speaker_identifier.identify, audio, start_seconds)
            if identified:
                speaker = identified
                speaker_confirmed = True
                take_confirmed_audio = getattr(speaker_identifier, "take_confirmed_audio", None)
                if take_confirmed_audio is not None:
                    first_pending = take_confirmed_audio(identified)
                    if first_pending is not None:
                        first_audio, first_start = first_pending
                        for old_entry in entries:
                            if (
                                old_entry.get("source") == "local_asr"
                                and old_entry.get("speaker") == "说话人未知"
                                and abs(float(old_entry.get("timestamp", 0)) - (meeting_started + first_start)) <= 0.75
                            ):
                                old_entry["speaker"] = identified
                                old_entry["speaker_confidence"] = "retrospectively_confirmed"
                                _rewrite_transcript_files(transcript_path, entries)
                                break
            elif len(audio) < 16_000 and entries:
                # Fast context continuity for short conversational turns (< 1.0s)
                last_entry = entries[-1]
                last_spk = last_entry.get("speaker")
                if last_spk and last_spk not in ("说话人未知", "Agent"):
                    if 0.0 <= (now - float(last_entry.get("timestamp", 0.0))) <= 1.2:
                        speaker = last_spk
                        speaker_confirmed = False
        except Exception as exc:
            console.print(f"[yellow]本段说话人识别失败，先标记为 Speaker：{exc}[/yellow]")
    entry = {
        "timestamp": now,
        "recorded_at": datetime.now().isoformat(timespec="seconds"),
        "time": f"{offset // 60:02d}:{offset % 60:02d}",
        "speaker": speaker,
        "text": text,
        "language": detected_language,
        "language_mode": language,
        "source": "local_asr",
        "asr_confidence": "filtered_whisper_segment",
        "speaker_confidence": (
            "voice_embedding_confirmed" if speaker_confirmed
            else ("context_continuity" if speaker != "说话人未知" else "unknown")
        ),
    }

    _store_local_utterance(entry, entries, transcript_path)
    shown_language = detected_language if language == "auto" else f"{language}, detected {detected_language}"
    console.print(
        f"[{entry['time']}] {entry['speaker']}: {text} [dim]({shown_language})[/dim]"
    )
    if not run_triggers:
        return
    kind = _trigger(text, keywords)
    detector = meeting_state.addressing if meeting_state is not None else None
    if kind is not None:
        # Keywords trigger on their own; no classifier call needed.
        addressed = detector.mentions_name(text) if detector else _contains_authorized_call(text)
    elif detector is not None:
        decision = await detector.decide(text, context=meeting_state.recent_transcript(seconds=60))
        addressed = decision.addressed
        if decision.kind != "none":
            console.print(f"[dim]称呼判断：{decision.kind}（{decision.source}）[/dim]")
    else:
        addressed = _contains_authorized_call(text)
    if addressed and kind is None:
        if live_calls is not None:
            if policy is not None and policy.authorize_live_turn(True):
                live_calls.put_nowait(text)
            return
        # No Live session (text reply mode, or Live unavailable): answer the
        # call with the Analyst, which can also search the web.
        kind = "ask"
    if not kind:
        return
    analysis = _handle_trigger(
        kind, entries, answers, playback=playback,
        transcript_path=transcript_path, meeting_started=meeting_started,
        request_text=text, analyst=analyst, policy=policy,
        directly_addressed=addressed, meeting_state=meeting_state,
    )
    if trigger_tasks is None:
        await analysis
        return
    task_name = f"trigger-{kind}"
    if any(task.get_name() == task_name and not task.done() for task in trigger_tasks):
        analysis.close()
        console.print(f"[dim]{kind} 分析仍在进行，本次重复请求已合并。[/dim]")
        return
    task = asyncio.create_task(analysis, name=task_name)
    trigger_tasks.add(task)
    task.add_done_callback(trigger_tasks.discard)


async def _run_file(
    file: Path, entries: list[dict[str, Any]], answers: list[dict[str, Any]],
    keywords: dict[str, list[str]], transcript_path: Path, language: str,
    run_triggers: bool = True,
    speaker_identifier: SherpaSpeakerIdentifier | None = None,
    analyst: GeminiAnalyst | None = None,
    policy: InterventionPolicy | None = None,
    live_calls: asyncio.Queue[str] | None = None,
    meeting_state: MeetingState | None = None,
) -> None:
    """Transcribe WAV speech segments locally using Silero VAD."""
    with wave.open(str(file), "rb") as wav:
        channels, rate, width = wav.getnchannels(), wav.getframerate(), wav.getsampwidth()
        if width != 2:
            raise RuntimeError("离线 WAV 必须是 16-bit PCM。请先用 ffmpeg 转换。")
        raw = wav.readframes(wav.getnframes())
    audio_copy = transcript_path.with_suffix(".wav")
    if file.resolve() != audio_copy.resolve():
        shutil.copy2(file, audio_copy)
    console.print(f"音频已保存：{audio_copy}")
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    if rate != SAMPLE_RATE:
        import scipy.signal

        audio = scipy.signal.resample_poly(audio, SAMPLE_RATE, rate).astype(np.float32)
    segmenter = SileroSpeechSegmenter()
    segments = segmenter.segment(audio)
    meeting_started = time.time()
    if not segments:
        console.print("[yellow]VAD 没有检测到语音片段。[/yellow]")
    for segment in segments:
        await _process_audio(
            segment.audio, segment.start_seconds, entries, answers, keywords,
            meeting_started, transcript_path, language, run_triggers=run_triggers,
            speaker_identifier=speaker_identifier, analyst=analyst, policy=policy,
            live_calls=live_calls, meeting_state=meeting_state,
        )
    if speaker_identifier is not None and hasattr(speaker_identifier, "recluster_session"):
        recluster_info = speaker_identifier.recluster_session(entries, meeting_started=meeting_started)
        if recluster_info.get("merged_speakers") or recluster_info.get("resolved_unknowns"):
            _rewrite_transcript_files(transcript_path, entries)
            console.print(
                f"[dim]会后说话人优化完成：合并 {recluster_info.get('merged_speakers')} 个重叠说话人，"
                f"补全 {recluster_info.get('resolved_unknowns')} 条短句，"
                f"最终共 {len(recluster_info.get('final_speakers', []))} 位发言人。[/dim]"
            )



async def run_local_transcription(file: Path, out: Path, language: str = "auto") -> tuple[Path, Path]:
    """Run the Phase 2 local-only pipeline and export TXT, JSONL, Markdown, and WAV."""
    out.mkdir(parents=True, exist_ok=True)
    transcript_path = _new_transcript(out, datetime.now())
    entries: list[dict[str, Any]] = []
    speaker_identifier = SherpaSpeakerIdentifier()
    await _run_file(
        file, entries, [], {}, transcript_path, language,
        run_triggers=False, speaker_identifier=speaker_identifier,
    )
    markdown_path = transcript_path.with_suffix(".md")
    lines = ["# 本地转写", "", f"音频：{file.name}", "", "## 逐字稿", ""]
    lines.extend(
        f"- [{entry['time']}] {entry['speaker']}: {entry['text']}（{entry['language']}）"
        for entry in entries
    )
    if not entries:
        lines.append("- 没有检测到可转写的语音。")
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    console.print(f"本地转写已保存：{transcript_path}")
    console.print(f"JSONL 已保存：{transcript_path.with_suffix('.jsonl')}")
    console.print(f"Markdown 已保存：{markdown_path}")
    return transcript_path, markdown_path


async def _run_microphone(
    entries: list[dict[str, Any]], answers: list[dict[str, Any]],
    keywords: dict[str, list[str]], transcript_path: Path, language: str,
    speaker_identifier: SherpaSpeakerIdentifier | None, analyst: GeminiAnalyst,
    policy: InterventionPolicy, analyze_claims: bool = False,
    live_enabled: bool = True,
    aec_enabled: bool = True, aec_delay_ms: int = 20,
    external_stop: asyncio.Event | None = None,
    meeting_state: MeetingState | None = None,
    share_transcript_with_live: bool = True,
) -> None:
    """Capture once, then fan microphone frames out to local ASR and Live."""
    meeting_state = meeting_state or MeetingState(entries)
    trigger_tasks: set[asyncio.Task[None]] = set()
    try:
        import sounddevice as sd
    except ImportError as exc:
        raise RuntimeError("缺少音频依赖，请在 Python 3.12 环境执行 `uv sync`。") from exc

    try:
        segmenter: SileroSpeechSegmenter | None = SileroSpeechSegmenter()
    except Exception as exc:
        segmenter = None
        console.print(f"[yellow]本地 VAD 不可用；仍会保存会议音频：{exc}[/yellow]")
    # Keep captured frames queued while a complete utterance is being transcribed.
    # The original WAV is the durable source if the machine cannot keep up.
    chunks: asyncio.Queue[np.ndarray | None] = asyncio.Queue()
    live_chunks: asyncio.Queue[bytes] = asyncio.Queue(maxsize=100)
    live_calls: asyncio.Queue[str] = asyncio.Queue(maxsize=10)
    echo_canceller: WebRtcEchoCanceller | None = None
    echo_reference: PlaybackReference | None = None
    if aec_enabled:
        try:
            echo_canceller = WebRtcEchoCanceller(stream_delay_ms=aec_delay_ms)
            echo_reference = PlaybackReference()
            console.print(f"[green]WebRTC AEC 已启用（参考延迟 {aec_delay_ms} ms）。[/green]")
        except RuntimeError as exc:
            console.print(f"[yellow]{exc} 本次会议将不启用声学回声消除。[/yellow]")
    playback = _PlaybackBuffer(echo_reference, gain=policy.config.playback_gain)
    input_overflows = 0
    # MEETAGENT_AEC_DEBUG=1 keeps raw mic, AEC reference and AEC output for the
    # first minutes in <meeting>.aec.npz to diagnose echo cancellation.
    aec_debug: list[tuple[np.ndarray, np.ndarray, np.ndarray, bool]] | None = (
        [] if os.environ.get("MEETAGENT_AEC_DEBUG") == "1" else None
    )
    loop = asyncio.get_running_loop()
    started = time.monotonic()
    meeting_started = time.time()
    pending: list[np.ndarray] = []
    speech_start: float | None = None
    audio_path = transcript_path.with_suffix(".wav")
    audio_writer = wave.open(str(audio_path), "wb")
    audio_writer.setnchannels(1)
    audio_writer.setsampwidth(2)
    audio_writer.setframerate(SAMPLE_RATE)
    last_audio_flush = time.monotonic()
    barge_in = _BargeInDetector(
        threshold=policy.config.barge_in_threshold or policy.config.rms_threshold,
        hold_seconds=policy.config.barge_in_ms / 1000,
        quiet_threshold=policy.config.rms_threshold,
    )
    dropped_live_chunks = 0
    claim_analysis_task: asyncio.Task[None] | None = None

    if analyze_claims:
        console.print(
            "[yellow]已启用后台静默 claim extraction：最近语音逐字稿每分钟会额外发送到 Gemini Flash。[/yellow]"
        )

    async def analyze_claim_windows() -> None:
        """Analyze recent transcript windows on request, without speaking aloud."""
        last_analyzed_timestamp = 0.0
        while not stop_requested.is_set():
            try:
                await asyncio.wait_for(stop_requested.wait(), timeout=60)
                return
            except TimeoutError:
                pass
            recent = [
                item for item in entries
                if item["source"] in {"local_asr", "gemini_live_fallback"}
                and item["timestamp"] > last_analyzed_timestamp
                and item["timestamp"] >= time.time() - 120
            ]
            if not recent:
                continue
            last_analyzed_timestamp = max(item["timestamp"] for item in recent)
            transcript = "\n".join(
                f"[{item['time']}] {item['speaker']}: {item['text']}" for item in recent
            )
            try:
                result = await analyst.respond(transcript, "claim_extract")
                if result.findings:
                    console.print(Markdown(result.answer))
            except Exception as exc:
                console.print(f"[yellow]后台 claim extraction 暂不可用：{exc}[/yellow]")

    def callback(indata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
        nonlocal echo_canceller
        nonlocal last_audio_flush
        nonlocal dropped_live_chunks, input_overflows
        if status:
            if status.input_overflow:
                input_overflows += 1
            console.print(f"[yellow]音频状态：{status}[/yellow]")
        microphone = indata[:, 0].copy()
        # Input streams sometimes start with garbage frames far outside [-1, 1]
        # (a constant ~2.9 was measured). Treat them as silence so they cannot
        # throw off the AEC, VAD or Live; real microphone audio never gets there.
        if not np.all(np.isfinite(microphone)) or np.max(np.abs(microphone)) > 1.5:
            microphone = np.zeros_like(microphone)
        data = microphone.copy()
        raw_pcm = (np.clip(microphone, -1.0, 1.0) * 32767).astype("<i2")
        if echo_canceller is not None and echo_reference is not None:
            try:
                reference = echo_reference.read(frames)
                data = echo_canceller.process(data, reference)
                if aec_debug is not None and len(aec_debug) < AEC_DEBUG_MAX_BLOCKS:
                    aec_debug.append((microphone.copy(), reference, data.copy(), playback.is_speaking()))
            except Exception as exc:
                console.print(f"[yellow]AEC 处理失败，后续音频改用原始麦克风输入：{exc}[/yellow]")
                echo_canceller = None
        pcm = (np.clip(data, -1.0, 1.0) * 32767).astype("<i2")
        now = time.monotonic()
        rms = float(np.sqrt(np.mean(data * data)))
        agent_speaking = playback.is_speaking()
        event = barge_in.update(rms, now, agent_speaking)
        if event == "barge_in" and playback.local_barge_in():
            console.print("[dim]检测到你在说话，已立即停止 Agent 播放。[/dim]")
        elif event == "rearm":
            playback.rearm_after_silence()
        # Commit the valid WAV header and original mic data on every callback;
        # fsync below bounds unsaved audio if the process is interrupted.
        audio_writer.writeframes(raw_pcm.tobytes())
        if now - last_audio_flush >= 1.0:
            audio_writer._file.flush()
            os.fsync(audio_writer._file.fileno())
            last_audio_flush = now
        # Half duplex towards Live: while the Agent is speaking and nobody has
        # barged in, send silence so Live's own VAD does not take the speaker
        # echo for an interruption (it did on 2026-10-01).
        if agent_speaking and not barge_in.latched:
            live_pcm = bytes(len(pcm) * 2)
        else:
            live_pcm = pcm.tobytes()

        def enqueue() -> None:
            nonlocal dropped_live_chunks
            # Local ASR receives the untouched mic signal; AEC output is only
            # for Live upload and playback-interruption energy checks.
            chunks.put_nowait(microphone)
            if live_chunks.full():
                try:
                    live_chunks.get_nowait()
                    dropped_live_chunks += 1
                except asyncio.QueueEmpty:
                    pass
            live_chunks.put_nowait(live_pcm)

        loop.call_soon_threadsafe(enqueue)

    async def process_microphone_frame(data: np.ndarray) -> None:
        nonlocal segmenter, pending, speech_start
        if segmenter is None:
            return
        try:
            event = segmenter.process_frame(data)
        except Exception as exc:
            console.print(f"[yellow]VAD 处理失败；继续保留音频 WAV：{exc}[/yellow]")
            segmenter = None
            pending = []
            speech_start = None
            return
        if event and "start" in event:
            speech_start = max(0.0, float(event["start"]))
            pending = [data]
        elif speech_start is not None:
            pending.append(data)
        if event and "end" in event and speech_start is not None:
            await _process_audio(
                np.concatenate(pending), speech_start, entries, answers, keywords,
                meeting_started, transcript_path, language,
                playback=playback,
                speaker_identifier=speaker_identifier, analyst=analyst,
                policy=policy,
                # Calls go to Live only while a Live session is running.
                live_calls=live_calls if live_task is not None else None,
                meeting_state=meeting_state, trigger_tasks=trigger_tasks,
            )
            pending = []
            speech_start = None

    if live_enabled:
        console.print("[bold]本会议正在转写。麦克风音频会持续发送到 Gemini Live；只有本地 Whisper 识别到明确呼叫后才允许 Agent 回答。请确保已获得与会者同意。按 Ctrl+C 结束。[/bold]")
    else:
        console.print("[bold]本会议正在本地录音和转写。当前未启用 Gemini Live；按 Ctrl+C 结束。[/bold]")
    stop_requested = external_stop or asyncio.Event()
    live_ready = asyncio.Event()
    previous_sigint = signal.getsignal(signal.SIGINT)
    owns_signal_handler = external_stop is None
    if owns_signal_handler:
        loop.add_signal_handler(signal.SIGINT, stop_requested.set)
    try:
        with (
            sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", blocksize=512,
                           callback=callback) as microphone_stream,
            sd.OutputStream(samplerate=PLAYBACK_SAMPLE_RATE, channels=1, dtype="float32", blocksize=480,
                            callback=playback.callback) as speaker_stream,
        ):
            if analyze_claims:
                claim_analysis_task = asyncio.create_task(
                    analyze_claim_windows(), name="silent-claim-extraction",
                )
            live_task: asyncio.Task[None] | None = None
            if live_enabled:
                live_task = asyncio.create_task(
                    _run_live_voice(
                        live_chunks, live_calls, playback, stop_requested, live_ready,
                        entries, transcript_path, meeting_started,
                        policy,
                        meeting_state if share_transcript_with_live else None,
                        meeting_state.addressing,
                    ),
                    name="continuous-live-voice",
                )
            try:
                if live_task is not None:
                    ready_waiter = asyncio.create_task(live_ready.wait())
                    done, _ = await asyncio.wait(
                        {live_task, ready_waiter}, timeout=60,
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    if not ready_waiter.done():
                        ready_waiter.cancel()
                    await asyncio.gather(ready_waiter, return_exceptions=True)
                    if live_task in done:
                        try:
                            await live_task
                        except Exception as exc:
                            console.print(f"[yellow]Gemini Live 未能连接；继续本地录音与转写：{exc}[/yellow]")
                            live_task = None
                    elif not live_ready.is_set():
                        live_task.cancel()
                        await asyncio.gather(live_task, return_exceptions=True)
                        live_task = None
                        console.print("[yellow]Gemini Live 连接超时；继续本地录音与转写。[/yellow]")
                while not stop_requested.is_set():
                    if live_task is not None and live_task.done():
                        try:
                            await live_task
                        except Exception as exc:
                            console.print(f"[yellow]Live 已断开；继续保存本地录音与转写：{exc}[/yellow]")
                        live_task = None
                    try:
                        data = await asyncio.wait_for(chunks.get(), timeout=0.25)
                    except TimeoutError:
                        continue
                    if data is None:
                        break
                    await process_microphone_frame(data)
            finally:
                stop_requested.set()
                microphone_stream.stop()
                speaker_stream.stop()
                # The callback schedules its final frames thread-safely. Yield
                # once, then finish VAD/ASR over everything captured before stop.
                await asyncio.sleep(0)
                while not chunks.empty():
                    await process_microphone_frame(chunks.get_nowait())
                if speech_start is not None and pending:
                    console.print("[dim]正在识别最后一段发言…[/dim]")
                    await _process_audio(
                        np.concatenate(pending), speech_start, entries, answers, keywords,
                        meeting_started, transcript_path, language,
                        playback=playback,
                        speaker_identifier=speaker_identifier, analyst=analyst,
                        policy=policy, live_calls=None,
                        meeting_state=meeting_state, trigger_tasks=trigger_tasks,
                    )
                    pending = []
                    speech_start = None
                if speaker_identifier is not None and hasattr(speaker_identifier, "recluster_session"):
                    recluster_info = speaker_identifier.recluster_session(entries, meeting_started=meeting_started)
                    if recluster_info.get("merged_speakers") or recluster_info.get("resolved_unknowns"):
                        _rewrite_transcript_files(transcript_path, entries)
                        console.print(
                            f"[dim]会后说话人优化完成：合并 {recluster_info.get('merged_speakers')} 个重叠说话人，"
                            f"补全 {recluster_info.get('resolved_unknowns')} 条短句，"
                            f"最终共 {len(recluster_info.get('final_speakers', []))} 位发言人。[/dim]"
                        )

                if claim_analysis_task is not None:
                    claim_analysis_task.cancel()
                    await asyncio.gather(claim_analysis_task, return_exceptions=True)
                if trigger_tasks:
                    console.print(f"[dim]等待 {len(trigger_tasks)} 个进行中的 Agent 分析完成…[/dim]")
                    await asyncio.gather(*trigger_tasks, return_exceptions=True)
                if live_task is not None:
                    try:
                        await asyncio.wait_for(live_task, timeout=8)
                    except TimeoutError:
                        live_task.cancel()
                        await asyncio.gather(live_task, return_exceptions=True)
                    except Exception as exc:
                        if not live_ready.is_set():
                            raise
                        console.print(f"[yellow]Live 会话结束时出错：{exc}[/yellow]")
        if dropped_live_chunks:
            console.print(f"[yellow]Live 上行过载，丢弃了 {dropped_live_chunks} 个最旧音频块。[/yellow]")
        console.print(
            f"[dim]音频回调统计：输入溢出 {input_overflows} 次，播放欠载 {playback.output_underflows} 次。[/dim]"
        )
        if aec_debug:
            debug_path = transcript_path.with_suffix(".aec.npz")
            np.savez_compressed(
                debug_path,
                raw=np.concatenate([block[0] for block in aec_debug]),
                reference=np.concatenate([block[1] for block in aec_debug]),
                output=np.concatenate([block[2] for block in aec_debug]),
                agent_speaking=np.array([block[3] for block in aec_debug]),
            )
            console.print(f"[dim]AEC 诊断数据已保存：{debug_path}[/dim]")
    finally:
        audio_writer.close()
        if owns_signal_handler:
            loop.remove_signal_handler(signal.SIGINT)
            signal.signal(signal.SIGINT, previous_sigint)
        console.print(f"录音音频已保存：{audio_path}")


async def run_demo(
    out: Path, file: Path | None, language: str = "auto", analyze_claims: bool = False,
    intervention: str | None = None, aec_enabled: bool | None = None,
    summary_language: str = "zh", obsidian_vault: Path | None = None,
    obsidian_folder: str = "Meetings",
    enable_live: bool | None = None, external_stop: asyncio.Event | None = None,
    meeting_id: str | None = None,
    allow_gemini_summary: bool = True,
    reply: str | None = None,
) -> None:
    """Run the selected audio mode and persist the meeting outputs."""
    if (out / ".obsidian").is_dir():
        out = out / "Meetings"
    out.mkdir(parents=True, exist_ok=True)
    keywords = _load_keywords()
    entries: list[dict[str, Any]] = []
    answers: list[dict[str, Any]] = []
    app_config = AppConfig.load()
    if obsidian_vault is None:
        obsidian_vault = app_config.output.obsidian_vault
    if obsidian_folder == "Meetings":
        obsidian_folder = app_config.output.obsidian_folder
    live_config: LiveConfig = app_config.live
    use_aec = app_config.audio.aec_enabled if aec_enabled is None else aec_enabled
    if intervention is not None:
        live_config = replace(live_config, intervention=intervention)
    if reply is not None:
        live_config = replace(live_config, reply=reply)
    if live_config.reply == "text":
        # Text replies never need Gemini Live, so meeting audio is not uploaded.
        enable_live = False
        console.print("[bold]文字回复模式：不连接 Gemini Live、不上传会议音频；点名或关键词触发时，"
                      "相关逐字稿会发送到 Gemini Flash 分析，回答只显示在终端并写入逐字稿。[/bold]")
    policy = InterventionPolicy(live_config)
    transcript_path = _new_transcript(out, datetime.now(), meeting_id)
    console.print("正在加载本地说话人识别模型（首次运行会下载到缓存）…")
    try:
        speaker_identifier: SherpaSpeakerIdentifier | None = SherpaSpeakerIdentifier()
    except Exception as exc:
        speaker_identifier = None
        console.print(f"[yellow]说话人识别不可用；仍会保存音频和逐字稿：{exc}[/yellow]")
    analyst = GeminiAnalyst(
        _answer, transcript_path.stem,
        transcript_path.with_suffix(".findings.jsonl"),
    )
    # Shared working memory for Analyst and Live. The digest only calls Gemini
    # right before an analysis that already sends transcript text to Gemini.
    has_gemini_key = bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
    meeting_state = MeetingState(
        entries, digest_function=gemini_digest if has_gemini_key else None,
        addressing=AddressDetector(
            app_config.agent.names, app_config.agent.followup_seconds,
            classify=gemini_classify if has_gemini_key and app_config.agent.classifier else None,
        ),
    )
    should_save = False
    console.print(f"文字转写将实时保存到：{transcript_path}")
    if platform.system() == "Darwin":
        whisper_model = os.environ.get("MEETAGENT_WHISPER_MODEL_ID", MODEL_ID)
    else:
        whisper_model = str(app_config.audio.model_path or "未设置（MEETAGENT_WHISPER_MODEL）")
    console.print(f"本地 Whisper 模型：{whisper_model}")
    language_labels = {"auto": "自动检测", "zh": "中文", "en": "英语", "de": "德语"}
    console.print(f"Whisper 识别语言：{language_labels.get(language, language)}")
    console.print("[bold]本会议正在转写。请确保已获得与会者同意。[/bold]")
    try:
        if file:
            await _run_file(
                file, entries, answers, keywords, transcript_path, language,
                speaker_identifier=speaker_identifier, analyst=analyst, policy=policy,
                meeting_state=meeting_state,
            )
        else:
            await _run_microphone(
                entries, answers, keywords, transcript_path, language, speaker_identifier, analyst,
                policy=policy, analyze_claims=analyze_claims, aec_enabled=use_aec,
                aec_delay_ms=app_config.audio.aec_delay_ms,
                live_enabled=(
                    bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
                    if enable_live is None else enable_live
                ),
                external_stop=external_stop,
                meeting_state=meeting_state,
                share_transcript_with_live=live_config.share_transcript,
            )
        should_save = True
    except KeyboardInterrupt:
        console.print("\n[dim]正在保存会议记录…[/dim]")
        should_save = True
    finally:
        if should_save:
            console.print(f"完整文字转写已保存：{transcript_path}")
            if entries or answers:
                summary_path, generated_by_gemini = await _write_summary(
                    entries, answers, transcript_path, summary_language,
                    obsidian_vault, obsidian_folder, allow_gemini_summary,
                )
                if generated_by_gemini:
                    console.print(f"Gemini 会议纪要已保存：{summary_path}")
                else:
                    console.print(f"本地 Markdown 会议纪要已保存：{summary_path}")
            else:
                console.print("[yellow]没有识别到文字内容；WAV 与空转写已保存，跳过空会议纪要。[/yellow]")

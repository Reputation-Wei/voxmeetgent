"""Reusable Analyst service and finding event stream."""

import asyncio
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable
from typing import Protocol

from meetagent.events import EventBus, Finding, Utterance


class Analyst(Protocol):
    """Check claims and find research with verifiable source links."""

    async def analyze(self, utterances: list[Utterance], trigger: str) -> list[Finding]:
        """Analyze recent transcript context and return structured findings."""
        raise NotImplementedError


@dataclass(frozen=True)
class AnalysisResult:
    """User-facing answer plus source records and structured analyst findings."""

    answer: str
    sources: list[str]
    papers: list[dict[str, str]]
    details: dict[str, Any]
    findings: list[Finding]


AnswerFunction = Callable[
    [str, str, str], Awaitable[tuple[str, list[str], list[dict[str, str]], dict[str, Any]]]
]


class GeminiAnalyst:
    """Run a configured answer function and publish findings through EventBus."""

    def __init__(
        self,
        answer_function: AnswerFunction,
        meeting_id: str,
        findings_path: Path,
        bus: EventBus | None = None,
    ) -> None:
        self._answer_function = answer_function
        self.meeting_id = meeting_id
        self.findings_path = findings_path
        self.bus = bus or EventBus()
        self._sink = self.bus.subscribe(Finding)
        self._write_lock = asyncio.Lock()

    async def respond(self, transcript: str, trigger: str, request_text: str = "") -> AnalysisResult:
        """Generate one triggered analysis and durably record every Finding event."""
        answer, sources, papers, details = await self._answer_function(
            transcript, trigger, request_text,
        )
        findings = self._make_findings(trigger, details, sources)
        async with self._write_lock:
            for finding in findings:
                published = await self.bus.publish(finding)
                event = self._sink.get_nowait()
                self._append_event(event)
                if published.sequence != event.sequence:
                    raise RuntimeError("Finding event sink received an unexpected event.")
        return AnalysisResult(answer, sources, papers, details, findings)

    async def analyze(self, utterances: list[Utterance], trigger: str) -> list[Finding]:
        """Protocol adapter for callers that already hold typed utterance events."""
        transcript = "\n".join(
            f"[{utterance.start_seconds:.1f}s] {utterance.speaker_id or 'Speaker'}: {utterance.text}"
            for utterance in utterances
        )
        result = await self.respond(transcript, trigger)
        return result.findings

    def _make_findings(self, trigger: str, details: dict[str, Any], sources: list[str]) -> list[Finding]:
        findings: list[Finding] = []
        if trigger == "fact_check":
            for item in details.get("findings", []):
                try:
                    confidence = float(item.get("confidence"))
                    if confidence > 1:
                        confidence /= 100
                except (TypeError, ValueError):
                    confidence = None
                findings.append(Finding(
                    meeting_id=self.meeting_id, kind="fact_check", claim=str(item.get("claim", "")),
                    verdict=str(item.get("verdict", "unverified")),
                    confidence=confidence, sources=tuple(sources),
                ))
        elif trigger == "logic_check":
            for item in details.get("findings", []):
                quote = str(item.get("quote", ""))
                timestamps = ", ".join(item.get("timestamps", []))
                findings.append(Finding(
                    meeting_id=self.meeting_id, kind=f"logic:{item.get('issue_type', 'unknown')}",
                    claim=f"{quote} (时间：{timestamps or '未提供'})",
                    verdict="needs_review", sources=tuple(sources),
                ))
        elif trigger == "research":
            for item in details.get("directions", []):
                findings.append(Finding(
                    meeting_id=self.meeting_id, kind="research_direction",
                    claim=str(item.get("research_question", "")), verdict="candidate",
                    sources=tuple(item.get("paper_sources", [])),
                ))
        elif trigger == "claim_extract":
            for item in details.get("claims", []):
                findings.append(Finding(
                    meeting_id=self.meeting_id, kind="claim_extraction",
                    claim=(
                        f"[{item.get('timestamp', '时间未提供')}] "
                        f"{item.get('speaker', 'Speaker')}: {item.get('claim', '')}"
                    ),
                    verdict="unverified",
                ))
        return findings

    def _append_event(self, finding: Finding) -> None:
        self.findings_path.parent.mkdir(parents=True, exist_ok=True)
        record = asdict(finding)
        record["occurred_at"] = finding.occurred_at.isoformat()
        with self.findings_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

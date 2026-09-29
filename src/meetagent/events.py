"""Immutable event models and a small asyncio event bus."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, TypeVar


@dataclass(frozen=True, kw_only=True)
class Event:
    """Base for timestamped meeting events."""

    meeting_id: str
    occurred_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    sequence: int = 0


@dataclass(frozen=True, kw_only=True)
class Utterance(Event):
    """A recognized speech segment and its provenance."""

    start_seconds: float
    end_seconds: float
    text: str
    language: str = "unknown"
    speaker_id: str | None = None
    source: str = "local_asr"
    confidence: float | None = None


@dataclass(frozen=True, kw_only=True)
class Trigger(Event):
    """A phrase or intent that may request agent attention."""

    kind: str
    utterance: Utterance


@dataclass(frozen=True, kw_only=True)
class Finding(Event):
    """An analyst result with traceable evidence links."""

    kind: str
    claim: str
    verdict: str
    confidence: float | None = None
    sources: tuple[str, ...] = ()


@dataclass(frozen=True, kw_only=True)
class AgentAudio(Event):
    """A generated speech chunk; PCM bytes are transient, not persisted here."""

    pcm: bytes
    sample_rate: int = 24_000
    interrupted: bool = False


TEvent = TypeVar("TEvent", bound=Event)


class EventBus:
    """Typed in-process fan-out bus with bounded subscriber queues."""

    def __init__(self, queue_size: int = 256) -> None:
        if queue_size <= 0:
            raise ValueError("queue_size 必须大于 0。")
        self._queue_size = queue_size
        self._subscribers: dict[type[Event], list[asyncio.Queue[Event]]] = {}
        self._sequence = 0

    def subscribe(self, event_type: type[TEvent]) -> asyncio.Queue[TEvent]:
        """Subscribe to one event type (not its subclasses)."""
        queue: asyncio.Queue[TEvent] = asyncio.Queue(maxsize=self._queue_size)
        self._subscribers.setdefault(event_type, []).append(queue)  # type: ignore[arg-type]
        return queue

    async def publish(self, event: TEvent) -> TEvent:
        """Publish to all subscribers, failing loudly on a full queue."""
        self._sequence += 1
        if event.sequence == 0:
            from dataclasses import replace

            event = replace(event, sequence=self._sequence)
        for queue in self._subscribers.get(type(event), []):
            queue.put_nowait(event)
        return event


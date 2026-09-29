"""Transcript storage interface."""

from typing import Protocol

from meetagent.events import Event, Utterance


class TranscriptStore(Protocol):
    """Persist source events and export readable meeting transcripts."""

    async def append(self, event: Event) -> None:
        """Persist one event; implementation arrives in Phase 2."""
        raise NotImplementedError

    async def utterances(self, meeting_id: str) -> list[Utterance]:
        """Read a meeting's utterances in sequence order."""
        raise NotImplementedError

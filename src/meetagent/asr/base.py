"""Speech recognition backend interface."""

from typing import Protocol

from meetagent.events import Utterance


class ASRBackend(Protocol):
    """Convert a speech segment into one or more timestamped utterances."""

    async def transcribe(self, pcm: bytes, language: str = "auto") -> list[Utterance]:
        """Transcribe locally or via an explicitly configured backend."""
        raise NotImplementedError

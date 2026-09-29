"""Gemini Live voice agent interface."""

from typing import Protocol


class VoiceAgent(Protocol):
    """Stream audio, receive responses, and honor local/server barge-in."""

    async def run(self) -> None:
        """Run concurrent Live send/receive loops; implement in Phase 4."""
        raise NotImplementedError

    async def speak_text(self, text: str) -> None:
        """Speak a supplied analysis result; implement in Phase 4."""
        raise NotImplementedError

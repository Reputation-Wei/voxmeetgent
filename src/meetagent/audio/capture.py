"""Microphone and system-audio capture interface."""

from typing import Protocol


class AudioCapture(Protocol):
    """Yield fixed-rate PCM frames from a selected input source."""

    async def frames(self):
        """Return an async iterator of PCM frames; implemented in Phase 2."""
        raise NotImplementedError

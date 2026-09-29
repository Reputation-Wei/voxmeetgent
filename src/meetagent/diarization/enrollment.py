"""Speaker enrollment interface."""

from pathlib import Path
from typing import Protocol


class SpeakerEnrollment(Protocol):
    """Map anonymous diarization identifiers to user-approved names."""

    def enroll(self, display_name: str, wav_path: Path) -> None:
        """Extract and store a name's embedding from a single-speaker WAV."""
        raise NotImplementedError

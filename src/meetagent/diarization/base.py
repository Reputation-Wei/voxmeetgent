"""Speaker identification interface for one detected utterance."""

from typing import Protocol

import numpy as np


class DiarizationBackend(Protocol):
    """Assign a stable identity to one mono 16 kHz speech segment."""

    def identify(self, audio: np.ndarray) -> str | None:
        """Return a registered or meeting-local speaker label."""
        raise NotImplementedError

"""Multilingual keyword and intent detection interface."""

from typing import Protocol

from meetagent.events import Trigger, Utterance


class TriggerDetector(Protocol):
    """Detect user requests in Chinese, English, and German utterances."""

    def detect(self, utterance: Utterance) -> list[Trigger]:
        """Return matching triggers; implement in Phase 3."""
        raise NotImplementedError

"""Client-side policy for authorizing spoken Live and Analyst responses."""

from __future__ import annotations

from typing import Literal

from meetagent.config import LiveConfig

# Why the Agent wants to speak (PLAN.md Phase 8, user decision 2026-10-01):
#   call        - an attendee addressed the Agent with a question
#   keyword     - research / fact check / logic check / summarize keyword
#   hand_raise  - the Agent itself has something to add (needs approval, P3)
SpeechReason = Literal["call", "keyword", "hand_raise"]


class InterventionPolicy:
    """Decide whether a response may be played aloud."""

    def __init__(self, config: LiveConfig) -> None:
        self.config = config

    def may_speak(self, reason: SpeechReason, approved: bool = False) -> bool:
        """Calls and keywords speak directly; a raised hand only after approval.

        Text reply mode never speaks; answers are only printed and saved.
        """
        if self.config.intervention == "silent" or self.config.reply == "text":
            return False
        if reason == "hand_raise":
            return approved
        return True

    def authorize_live_turn(self, directly_called: bool) -> bool:
        """Authorize a Live voice turn only for a detected direct call."""
        return directly_called and self.may_speak("call")

"""Client-side policy for authorizing spoken Live and Analyst responses."""

from __future__ import annotations

from meetagent.config import LiveConfig


class InterventionPolicy:
    """Require an explicit call before every audible response."""

    def __init__(self, config: LiveConfig) -> None:
        self.config = config

    def authorize_live_turn(self, directly_called: bool) -> bool:
        """Authorize speech only for a detected direct call; silent always wins."""
        return self.config.intervention != "silent" and directly_called

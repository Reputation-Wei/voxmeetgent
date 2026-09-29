"""Console UI interface."""

from typing import Protocol

from meetagent.events import Event


class MeetingUI(Protocol):
    """Render transcript, findings, and meeting state to the user."""

    async def show_event(self, event: Event) -> None:
        """Render one event; implementation is planned for later phases."""
        raise NotImplementedError

"""Academic search tool interface."""

from typing import Protocol


class ScholarSearch(Protocol):
    """Return real research records with resolvable identifiers and URLs."""

    async def search(self, query: str, limit: int = 10) -> list[dict[str, str]]:
        """Search scholarly indexes; implement in Phase 3."""
        raise NotImplementedError

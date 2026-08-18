"""Repository contract plus an in-memory implementation for node tests."""

from typing import Iterable, Protocol, Sequence

from src.infrastructure.memory.models import (
    MemoryCandidate,
    MemoryItem,
    MemoryStatus,
    OutboxEvent,
)


class MemoryRepository(Protocol):
    def save_candidate(self, candidate: MemoryCandidate) -> MemoryItem: ...

    def search_authoritative(
        self,
        *,
        user_id: str,
        project_id: str | None,
        memory_types: Sequence[str],
        statuses: Sequence[MemoryStatus] = ("confirmed", "approved"),
        limit: int = 8,
    ) -> list[MemoryItem]: ...

    def get_by_ids(self, memory_ids: Sequence[str]) -> list[MemoryItem]: ...

    def set_status(
        self,
        *,
        memory_id: str,
        status: MemoryStatus,
        actor: str,
    ) -> MemoryItem: ...


class InMemoryMemoryRepository:
    """Behavioral reference implementation; never used as production fallback."""

    def __init__(self) -> None:
        self.items: list[MemoryItem] = []
        self.outbox_events: list[OutboxEvent] = []
        self._items_by_idempotency_key: dict[str, MemoryItem] = {}

    def save_candidate(self, candidate: MemoryCandidate) -> MemoryItem:
        existing = self._items_by_idempotency_key.get(candidate.idempotency_key)
        if existing is not None:
            return existing
        item = MemoryItem(**candidate.model_dump())
        event = OutboxEvent(
            memory_id=item.memory_id,
            payload=item.model_dump(mode="json"),
        )
        self.items.append(item)
        self.outbox_events.append(event)
        self._items_by_idempotency_key[item.idempotency_key] = item
        return item

    def search_authoritative(
        self,
        *,
        user_id: str,
        project_id: str | None,
        memory_types: Sequence[str],
        statuses: Sequence[MemoryStatus] = ("confirmed", "approved"),
        limit: int = 8,
    ) -> list[MemoryItem]:
        allowed_types = set(memory_types)
        allowed_statuses = set(statuses)
        matches = [
            item
            for item in self.items
            if item.user_id == user_id
            and item.project_id == project_id
            and item.memory_type in allowed_types
            and item.status in allowed_statuses
        ]
        matches.sort(key=lambda item: item.updated_at, reverse=True)
        return matches[: max(int(limit), 0)]

    def get_by_ids(self, memory_ids: Sequence[str]) -> list[MemoryItem]:
        items_by_id = {item.memory_id: item for item in self.items}
        return [
            items_by_id[memory_id]
            for memory_id in dict.fromkeys(memory_ids)
            if memory_id in items_by_id
        ]

    def set_status(
        self,
        *,
        memory_id: str,
        status: MemoryStatus,
        actor: str,
    ) -> MemoryItem:
        for index, item in enumerate(self.items):
            if item.memory_id != memory_id:
                continue
            updated = item.model_copy(
                update={"status": status, "version": item.version + 1}
            )
            self.items[index] = updated
            self._items_by_idempotency_key[item.idempotency_key] = updated
            self.outbox_events.append(
                OutboxEvent(
                    memory_id=updated.memory_id,
                    payload=updated.model_dump(mode="json"),
                )
            )
            return updated
        raise KeyError(f"memory not found: {memory_id}")

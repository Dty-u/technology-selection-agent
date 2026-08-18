"""Bounded outbox worker that keeps Milvus off the response path."""

from typing import Any, Protocol

from src.observability.logging import log


class ProjectionRepository(Protocol):
    def claim_outbox_batch(self, *, limit: int): ...

    def mark_outbox_projected(self, **kwargs): ...

    def mark_outbox_failed(self, **kwargs): ...


class MemoryProjector(Protocol):
    provider: str

    def upsert(self, payload: dict[str, Any]) -> str: ...

    def delete(self, payload: dict[str, Any]) -> str: ...


class MemoryProjectionWorker:
    def __init__(
        self,
        repository: ProjectionRepository,
        projector: MemoryProjector,
    ) -> None:
        self.repository = repository
        self.projector = projector

    def run_once(self, *, limit: int = 20) -> dict[str, int]:
        events = self.repository.claim_outbox_batch(limit=limit)
        projected = 0
        failed = 0
        for event in events:
            try:
                if event.event_type == "memory.delete":
                    external_id = self.projector.delete(event.payload)
                else:
                    external_id = self.projector.upsert(event.payload)
                self.repository.mark_outbox_projected(
                    event_id=event.event_id,
                    memory_id=event.memory_id,
                    external_id=external_id,
                    provider=self.projector.provider,
                )
                projected += 1
            except Exception as exc:
                log.exception("---Milvus记忆投影失败---")
                self.repository.mark_outbox_failed(
                    event_id=event.event_id,
                    error=str(exc),
                )
                failed += 1
        return {
            "claimed": len(events),
            "projected": projected,
            "failed": failed,
        }

import unittest
from unittest.mock import MagicMock


class _FakeOutboxRepository:
    def __init__(self, events):
        self.events = list(events)
        self.projected = []
        self.failed = []

    def claim_outbox_batch(self, *, limit):
        return self.events[:limit]

    def mark_outbox_projected(
        self, *, event_id, memory_id, external_id, provider
    ):
        self.projected.append((event_id, memory_id, external_id, provider))

    def mark_outbox_failed(self, *, event_id, error):
        self.failed.append((event_id, error))


class MemoryProjectionWorkerTest(unittest.TestCase):
    def _event(self, event_type="memory.upsert"):
        from src.infrastructure.memory.models import OutboxEvent

        return OutboxEvent(
            event_id="event-1",
            memory_id="memory-1",
            event_type=event_type,
            payload={
                "memory_id": "memory-1",
                "user_id": "user-a",
                "thread_id": "thread-a",
                "project_id": None,
                "memory_type": "user_preference",
                "content": "用户偏好中文回答",
                "status": "confirmed",
                "metadata": {},
                "version": 1,
            },
        )

    def test_successful_projection_marks_outbox_and_projection_mapping(self):
        from src.infrastructure.memory.worker import MemoryProjectionWorker

        repository = _FakeOutboxRepository([self._event()])
        projector = MagicMock()
        projector.upsert.return_value = "memory-1"
        projector.provider = "milvus"
        worker = MemoryProjectionWorker(repository, projector)

        result = worker.run_once(limit=10)

        self.assertEqual(result, {"claimed": 1, "projected": 1, "failed": 0})
        projector.upsert.assert_called_once_with(self._event().payload)
        self.assertEqual(
            repository.projected,
            [("event-1", "memory-1", "memory-1", "milvus")],
        )

    def test_delete_event_removes_projection_instead_of_upserting(self):
        from src.infrastructure.memory.worker import MemoryProjectionWorker

        event = self._event("memory.delete")
        repository = _FakeOutboxRepository([event])
        projector = MagicMock()
        projector.delete.return_value = "memory-1"
        projector.provider = "milvus"

        result = MemoryProjectionWorker(repository, projector).run_once(limit=1)

        projector.delete.assert_called_once_with(event.payload)
        projector.upsert.assert_not_called()
        self.assertEqual(result["projected"], 1)

    def test_projection_failure_is_retried_via_outbox_not_raised(self):
        from src.infrastructure.memory.worker import MemoryProjectionWorker

        repository = _FakeOutboxRepository([self._event()])
        projector = MagicMock()
        projector.upsert.side_effect = RuntimeError("milvus unavailable")
        worker = MemoryProjectionWorker(repository, projector)

        result = worker.run_once(limit=10)

        self.assertEqual(result, {"claimed": 1, "projected": 0, "failed": 1})
        self.assertEqual(repository.failed[0][0], "event-1")
        self.assertIn("milvus unavailable", repository.failed[0][1])

if __name__ == "__main__":
    unittest.main()

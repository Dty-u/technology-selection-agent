import unittest


class MemoryRepositoryContractTest(unittest.TestCase):
    def _candidate(self, **overrides):
        from src.infrastructure.memory.models import MemoryCandidate

        values = {
            "idempotency_key": "turn-1:preference:0",
            "user_id": "user-a",
            "thread_id": "thread-a",
            "project_id": None,
            "memory_type": "user_preference",
            "content": "用户偏好中文回答",
            "status": "confirmed",
            "source_kind": "explicit_user_statement",
            "confidence": 1.0,
            "metadata": {"language": "zh-CN"},
        }
        values.update(overrides)
        return MemoryCandidate(**values)

    def test_scope_ids_and_status_are_validated(self):
        from pydantic import ValidationError

        with self.assertRaises(ValidationError):
            self._candidate(user_id=" ")
        with self.assertRaises(ValidationError):
            self._candidate(status="invented")
        with self.assertRaises(ValidationError):
            self._candidate(memory_type="project_fact", project_id=None)

    def test_save_candidate_is_idempotent_and_enqueues_one_projection(self):
        from src.infrastructure.memory.repository import (
            InMemoryMemoryRepository,
        )

        repository = InMemoryMemoryRepository()
        first = repository.save_candidate(self._candidate())
        second = repository.save_candidate(self._candidate(content="重复请求"))

        self.assertEqual(first.memory_id, second.memory_id)
        self.assertEqual(second.content, "用户偏好中文回答")
        self.assertEqual(len(repository.outbox_events), 1)
        self.assertEqual(repository.outbox_events[0].memory_id, first.memory_id)

    def test_recall_returns_only_usable_memories_in_the_requested_scope(self):
        from src.infrastructure.memory.repository import (
            InMemoryMemoryRepository,
        )

        repository = InMemoryMemoryRepository()
        repository.save_candidate(self._candidate())
        repository.save_candidate(
            self._candidate(
                idempotency_key="turn-2:inferred:0",
                content="可能偏好简短回答",
                status="pending",
                source_kind="model_inference",
            )
        )
        repository.save_candidate(
            self._candidate(
                idempotency_key="turn-3:other:0",
                user_id="user-b",
                content="其他用户的偏好",
            )
        )

        recalled = repository.search_authoritative(
            user_id="user-a",
            project_id=None,
            memory_types=("user_preference",),
            limit=10,
        )

        self.assertEqual([item.content for item in recalled], ["用户偏好中文回答"])

    def test_get_by_ids_preserves_requested_order_and_ignores_missing_ids(self):
        from src.infrastructure.memory.repository import InMemoryMemoryRepository

        repository = InMemoryMemoryRepository()
        first = repository.save_candidate(self._candidate(idempotency_key="first"))
        second = repository.save_candidate(
            self._candidate(idempotency_key="second", content="Second memory")
        )

        recalled = repository.get_by_ids(
            [second.memory_id, "missing", first.memory_id]
        )

        self.assertEqual(
            [item.memory_id for item in recalled],
            [second.memory_id, first.memory_id],
        )

    def test_project_memory_requires_matching_project_and_approved_status(self):
        from src.infrastructure.memory.repository import (
            InMemoryMemoryRepository,
        )

        repository = InMemoryMemoryRepository()
        repository.save_candidate(
            self._candidate(
                idempotency_key="selection-1",
                project_id="project-a",
                memory_type="selection_decision",
                content="项目A选择GAA",
                status="approved",
            )
        )
        repository.save_candidate(
            self._candidate(
                idempotency_key="selection-2",
                project_id="project-b",
                memory_type="selection_decision",
                content="项目B选择FinFET",
                status="approved",
            )
        )

        recalled = repository.search_authoritative(
            user_id="user-a",
            project_id="project-a",
            memory_types=("selection_decision",),
            statuses=("approved",),
            limit=2,
        )

        self.assertEqual([item.content for item in recalled], ["项目A选择GAA"])

    def test_repository_contract_has_no_raw_conversation_message_api(self):
        from src.infrastructure.memory.repository import (
            InMemoryMemoryRepository,
            MemoryRepository,
        )

        repository = InMemoryMemoryRepository()
        for method_name in (
            "record_message",
            "list_recent_messages",
            "delete_thread_messages",
        ):
            self.assertFalse(hasattr(MemoryRepository, method_name))
            self.assertFalse(hasattr(repository, method_name))

    def test_pending_selection_memory_requires_explicit_approval(self):
        from src.infrastructure.memory.repository import InMemoryMemoryRepository

        repository = InMemoryMemoryRepository()
        item = repository.save_candidate(
            self._candidate(
                idempotency_key="selection-pending",
                project_id="project-a",
                memory_type="selection_case",
                content="建议选择GAA",
                status="pending",
                source_kind="workflow_recommendation",
            )
        )
        self.assertEqual(
            repository.search_authoritative(
                user_id="user-a",
                project_id="project-a",
                memory_types=("selection_case",),
                statuses=("approved",),
            ),
            [],
        )

        approved = repository.set_status(
            memory_id=item.memory_id,
            status="approved",
            actor="user-a",
        )

        self.assertEqual(approved.status, "approved")
        recalled = repository.search_authoritative(
            user_id="user-a",
            project_id="project-a",
            memory_types=("selection_case",),
            statuses=("approved",),
        )
        self.assertEqual([memory.memory_id for memory in recalled], [item.memory_id])
        self.assertEqual(len(repository.outbox_events), 2)


if __name__ == "__main__":
    unittest.main()

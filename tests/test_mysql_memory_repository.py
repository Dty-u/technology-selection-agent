import unittest
from datetime import datetime
from unittest.mock import MagicMock


class _CursorContext:
    def __init__(self, cursor):
        self.cursor = cursor

    def __enter__(self):
        return self.cursor

    def __exit__(self, *args):
        return False


class MySQLMemoryRepositoryTest(unittest.TestCase):
    @staticmethod
    def _item_row(memory_id, content):
        return {
            "memory_id": memory_id,
            "idempotency_key": f"key-{memory_id}",
            "user_id": "user-a",
            "thread_id": "thread-a",
            "project_id": None,
            "memory_type": "user_fact",
            "content": content,
            "status": "confirmed",
            "source_kind": "explicit_user_statement",
            "confidence": 1.0,
            "metadata_json": "{}",
            "version": 1,
            "created_at": datetime(2026, 1, 1),
            "updated_at": datetime(2026, 1, 1),
        }

    def test_mysql_uri_is_parsed_without_logging_credentials(self):
        from src.infrastructure.memory.mysql import parse_mysql_uri

        parsed = parse_mysql_uri(
            "mysql+pymysql://memory_user:p%40ss@127.0.0.1:3307/rag_memory"
        )

        self.assertEqual(parsed["host"], "127.0.0.1")
        self.assertEqual(parsed["port"], 3307)
        self.assertEqual(parsed["user"], "memory_user")
        self.assertEqual(parsed["password"], "p@ss")
        self.assertEqual(parsed["database"], "rag_memory")
        self.assertFalse(parsed["autocommit"])

    def test_mysql_uri_requires_database_and_supported_scheme(self):
        from src.infrastructure.memory.mysql import parse_mysql_uri

        with self.assertRaisesRegex(ValueError, "mysql"):
            parse_mysql_uri("postgresql://user:pass@localhost/rag_memory")
        with self.assertRaisesRegex(ValueError, "database"):
            parse_mysql_uri("mysql://user:pass@localhost")

    def test_candidate_and_outbox_are_committed_in_one_transaction(self):
        from src.infrastructure.memory.models import MemoryCandidate
        from src.infrastructure.memory.mysql import MySQLMemoryRepository

        cursor = MagicMock()
        cursor.fetchone.return_value = None
        connection = MagicMock()
        connection.cursor.return_value = _CursorContext(cursor)
        repository = MySQLMemoryRepository(
            "mysql://user:pass@localhost/rag_memory",
            connection_factory=lambda: connection,
        )
        candidate = MemoryCandidate(
            idempotency_key="turn-1:fact:0",
            user_id="user-a",
            thread_id="thread-a",
            memory_type="user_fact",
            content="用户正在学习LangGraph",
            status="confirmed",
            source_kind="explicit_user_statement",
            confidence=1.0,
        )

        result = repository.save_candidate(candidate)

        statements = "\n".join(
            call.args[0] for call in cursor.execute.call_args_list
        )
        self.assertIn("INSERT INTO memory_items", statements)
        self.assertIn("INSERT INTO memory_versions", statements)
        self.assertIn("INSERT INTO memory_outbox", statements)
        self.assertIn("INSERT INTO memory_audit_events", statements)
        self.assertEqual(result.content, candidate.content)
        connection.commit.assert_called_once_with()
        connection.rollback.assert_not_called()
        connection.close.assert_called_once_with()

    def test_transaction_rolls_back_when_outbox_write_fails(self):
        from src.infrastructure.memory.models import MemoryCandidate
        from src.infrastructure.memory.mysql import MySQLMemoryRepository

        cursor = MagicMock()
        cursor.fetchone.return_value = None

        def execute(statement, _params=None):
            if "INSERT INTO memory_outbox" in statement:
                raise RuntimeError("outbox unavailable")

        cursor.execute.side_effect = execute
        connection = MagicMock()
        connection.cursor.return_value = _CursorContext(cursor)
        repository = MySQLMemoryRepository(
            "mysql://user:pass@localhost/rag_memory",
            connection_factory=lambda: connection,
        )
        candidate = MemoryCandidate(
            idempotency_key="turn-2:fact:0",
            user_id="user-a",
            thread_id="thread-a",
            memory_type="user_fact",
            content="事务必须回滚",
            status="confirmed",
            source_kind="explicit_user_statement",
            confidence=1.0,
        )

        with self.assertRaisesRegex(RuntimeError, "outbox unavailable"):
            repository.save_candidate(candidate)

        connection.rollback.assert_called_once_with()
        connection.commit.assert_not_called()
        connection.close.assert_called_once_with()

    def test_get_by_ids_preserves_semantic_hit_order(self):
        from src.infrastructure.memory.mysql import MySQLMemoryRepository

        cursor = MagicMock()
        cursor.fetchall.return_value = [
            self._item_row("memory-1", "First"),
            self._item_row("memory-2", "Second"),
        ]
        connection = MagicMock()
        connection.cursor.return_value = _CursorContext(cursor)
        repository = MySQLMemoryRepository(
            "mysql://user:pass@localhost/rag_memory",
            connection_factory=lambda: connection,
        )

        items = repository.get_by_ids(["memory-2", "memory-1"])

        self.assertEqual(
            [item.memory_id for item in items],
            ["memory-2", "memory-1"],
        )
        statement, params = cursor.execute.call_args.args
        self.assertIn("memory_id IN", statement)
        self.assertEqual(params, ("memory-2", "memory-1"))

    def test_projection_mapping_records_the_explicit_provider(self):
        from src.infrastructure.memory.mysql import MySQLMemoryRepository

        cursor = MagicMock()
        cursor.fetchone.return_value = {"version": 3}
        connection = MagicMock()
        connection.cursor.return_value = _CursorContext(cursor)
        repository = MySQLMemoryRepository(
            "mysql://user:pass@localhost/rag_memory",
            connection_factory=lambda: connection,
        )

        repository.mark_outbox_projected(
            event_id="event-1",
            memory_id="memory-1",
            external_id="memory-1",
            provider="milvus",
        )

        projection_call = next(
            call
            for call in cursor.execute.call_args_list
            if "INSERT INTO memory_projections" in call.args[0]
        )
        self.assertEqual(projection_call.args[1][1], "milvus")

    def test_projection_provider_count_supports_safe_cutover_preflight(self):
        from src.infrastructure.memory.mysql import MySQLMemoryRepository

        cursor = MagicMock()
        cursor.fetchone.return_value = {"projection_count": 2}
        connection = MagicMock()
        connection.cursor.return_value = _CursorContext(cursor)
        repository = MySQLMemoryRepository(
            "mysql://user:pass@localhost/rag_memory",
            connection_factory=lambda: connection,
        )

        count = repository.count_projections(provider="mem0")

        self.assertEqual(count, 2)
        statement, params = cursor.execute.call_args.args
        self.assertIn("provider = %s", statement)
        self.assertEqual(params, ("mem0",))

if __name__ == "__main__":
    unittest.main()

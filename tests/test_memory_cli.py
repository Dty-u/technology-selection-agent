import unittest
from unittest.mock import MagicMock, patch


class MemoryCliTest(unittest.TestCase):
    def test_worker_uses_direct_milvus_without_a_mem0_config(self):
        import src.infrastructure.memory.cli as cli

        repository = MagicMock()
        repository.count_projections.return_value = 0
        projector = MagicMock()
        worker = MagicMock()
        worker.run_once.return_value = {"claimed": 0, "projected": 0, "failed": 0}

        with patch.object(
            cli, "MySQLMemoryRepository", return_value=repository
        ), patch.object(
            cli, "MilvusMemoryIndex", return_value=projector
        ), patch.object(
            cli, "MemoryProjectionWorker", return_value=worker
        ):
            result = cli.run_worker_once(limit=7)

        self.assertEqual(result["claimed"], 0)
        worker.run_once.assert_called_once_with(limit=7)

    def test_worker_stops_when_legacy_mem0_projection_rows_exist(self):
        import src.infrastructure.memory.cli as cli

        repository = MagicMock()
        repository.count_projections.return_value = 2

        with patch.object(
            cli, "MySQLMemoryRepository", return_value=repository
        ), patch.object(cli, "MilvusMemoryIndex") as create_projector:
            with self.assertRaisesRegex(RuntimeError, "legacy Mem0"):
                cli.run_worker_once(limit=20)

        create_projector.assert_not_called()

    def test_setup_milvus_memory_is_explicit_and_non_destructive(self):
        import src.infrastructure.memory.cli as cli

        index = MagicMock()
        index.ensure_collection.return_value = True
        with patch.object(cli, "MilvusMemoryIndex", return_value=index):
            result = cli.setup_databases(
                checkpoint=False,
                memory=False,
                milvus_memory=True,
            )

        self.assertEqual(result, {"milvus_memory_created": True})
        index.ensure_collection.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()

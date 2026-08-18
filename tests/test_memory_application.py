import unittest
from contextlib import contextmanager
from unittest.mock import MagicMock, patch


class MemoryApplicationTest(unittest.TestCase):
    def test_production_lifecycle_warms_read_only_resources_once(self):
        import src.workflows.application as application

        repository = MagicMock()
        reader = MagicMock()
        checkpointer = MagicMock()

        @contextmanager
        def open_checkpointer(*args, **kwargs):
            yield checkpointer

        with patch.object(
            application,
            "WARMUP_ON_START",
            True,
        ), patch.object(
            application,
            "MySQLMemoryRepository",
            return_value=repository,
        ), patch.object(
            application,
            "MilvusMemoryIndex",
            return_value=reader,
        ), patch.object(
            application,
            "open_mysql_checkpointer",
            side_effect=open_checkpointer,
        ), patch.object(
            application,
            "build_query_router_graph",
            return_value=MagicMock(),
        ), patch.object(
            application,
            "warm_technology_selection_resources",
        ) as warm:
            with application.open_memory_application(
                checkpoint_uri="mysql://checkpoint",
                memory_uri="mysql://memory",
            ):
                pass

        warm.assert_called_once_with(reader)

    def test_default_composition_injects_direct_milvus_reader(self):
        import src.workflows.application as application

        repository = MagicMock()
        reader = MagicMock()
        checkpointer = MagicMock()
        graph = MagicMock()

        @contextmanager
        def open_checkpointer(*args, **kwargs):
            yield checkpointer

        with patch.object(
            application,
            "WARMUP_ON_START",
            False,
        ), patch.object(
            application,
            "MySQLMemoryRepository",
            return_value=repository,
        ), patch.object(
            application,
            "MilvusMemoryIndex",
            return_value=reader,
        ) as create_reader, patch.object(
            application,
            "open_mysql_checkpointer",
            side_effect=open_checkpointer,
        ), patch.object(
            application,
            "build_query_router_graph",
            return_value=graph,
        ) as build_graph:
            with application.open_memory_application(
                checkpoint_uri="mysql://checkpoint",
                memory_uri="mysql://memory",
            ) as runner:
                self.assertIs(runner.graph, graph)
                self.assertIs(runner.checkpointer, checkpointer)
                self.assertIsNone(runner.repository)

        create_reader.assert_called_once_with()
        build_graph.assert_called_once_with(
            checkpointer,
            memory_repository=repository,
            semantic_memory_reader=reader,
        )

    def test_explicit_reader_injection_is_preserved_for_tests(self):
        import src.workflows.application as application

        repository = MagicMock()
        reader = MagicMock()
        checkpointer = MagicMock()

        @contextmanager
        def open_checkpointer(*args, **kwargs):
            yield checkpointer

        with patch.object(
            application,
            "WARMUP_ON_START",
            False,
        ), patch.object(
            application,
            "MySQLMemoryRepository",
            return_value=repository,
        ), patch.object(
            application,
            "MilvusMemoryIndex",
        ) as create_reader, patch.object(
            application,
            "open_mysql_checkpointer",
            side_effect=open_checkpointer,
        ), patch.object(
            application,
            "build_query_router_graph",
            return_value=MagicMock(),
        ) as build_graph:
            with application.open_memory_application(
                checkpoint_uri="mysql://checkpoint",
                memory_uri="mysql://memory",
                semantic_memory_reader=reader,
            ):
                pass

        create_reader.assert_not_called()
        self.assertIs(
            build_graph.call_args.kwargs["semantic_memory_reader"], reader
        )


if __name__ == "__main__":
    unittest.main()

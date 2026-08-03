import os
import importlib
import unittest
import asyncio
import pickle
from hashlib import sha256
from pathlib import Path
from queue import Queue
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document


def _markdown_directory() -> Path:
    return Path(__file__).resolve().parents[1] / "datas" / "md"


class Neo4jConnectionTest(unittest.TestCase):
    def _connection_module(self):
        try:
            return importlib.import_module("src.infrastructure.graphdatabase.neo4j")
        except ModuleNotFoundError:
            return None

    def test_missing_credentials_prevent_connection(self):
        """A missing database credential must fail before creating a driver."""
        connection = self._connection_module()
        self.assertIsNotNone(connection, "Neo4j connection module is missing.")
        if connection is None:
            return

        with patch.object(connection, "NEO4J_URI", None):
            with self.assertRaisesRegex(RuntimeError, "NEO4J_URI"):
                connection.Neo4jDatabase().connect()


class GraphDocumentPreparationTest(unittest.TestCase):
    def _ingestion_module(self):
        try:
            return importlib.import_module(
                "src.infrastructure.document_processing.graph_ingestion"
            )
        except ModuleNotFoundError:
            return None

    def test_documents_receive_stable_source_ids(self):
        """Changing a chunk's content or index must change its source identity."""
        ingestion = self._ingestion_module()
        self.assertIsNotNone(ingestion, "Graph ingestion module is missing.")
        if ingestion is None:
            return

        source_path = Path("datas/md/tech_report_0tfhhamx.md").resolve()
        documents = [
            Document(page_content="first chunk"),
            Document(page_content="second chunk"),
        ]

        prepared_documents = ingestion.prepare_documents_for_graph_ingestion(
            documents,
            source_path,
        )

        expected_first_id = sha256(
            f"{source_path}:0:first chunk".encode("utf-8")
        ).hexdigest()
        self.assertEqual(prepared_documents[0].metadata["id"], expected_first_id)
        self.assertEqual(prepared_documents[0].metadata["source"], str(source_path))
        self.assertNotEqual(
            prepared_documents[0].metadata["id"],
            prepared_documents[1].metadata["id"],
        )

    def test_graph_documents_are_written_without_apoc(self):
        ingestion = self._ingestion_module()
        self.assertIsNotNone(ingestion, "Graph ingestion module is missing.")
        if ingestion is None:
            return

        class RecordingGraph:
            def __init__(self):
                self.calls = []

            def query(self, query, params):
                self.calls.append((query, params))

        source = Document(
            page_content="Plasma treats silicon.",
            metadata={"id": "source-1", "source": "report.md"},
        )
        material = SimpleNamespace(
            id="Silicon",
            type="Material",
            properties={"name": "Silicon"},
        )
        process = SimpleNamespace(
            id="Plasma",
            type="Process",
            properties={"name": "Plasma"},
        )
        relationship = SimpleNamespace(
            source=process,
            target=material,
            type="TREATS",
            properties={},
        )
        graph_document = SimpleNamespace(
            source=source,
            nodes=[material, process],
            relationships=[relationship],
        )
        graph = RecordingGraph()

        writer = getattr(ingestion, "write_graph_documents", None)
        self.assertIsNotNone(writer, "APOC-free graph writer is missing.")
        if writer is None:
            return
        writer(graph, [graph_document])

        queries = "\n".join(query for query, _ in graph.calls)
        self.assertNotIn("apoc.", queries.lower())
        self.assertIn("MERGE (entity:Entity", queries)
        self.assertIn("MERGE (document)-[:MENTIONS]->(entity)", queries)
        self.assertIn("RELATED_TO", queries)


class GraphDirectoryIngestionTest(unittest.TestCase):
    def _ingestion_module(self):
        return importlib.import_module(
            "src.infrastructure.document_processing.graph_ingestion"
        )

    def test_directory_import_scans_only_current_markdown_files(self):
        ingestion = self._ingestion_module()
        source_directory = _markdown_directory()
        expected_files = sorted(
            path
            for path in source_directory.iterdir()
            if path.is_file() and path.suffix.lower() == ".md"
        )
        successes = {
            str(path): {"chunk_count": 2, "entity_count": 3, "document_ids": [path.name]}
            for path in expected_files
        }

        with patch.object(
            ingestion,
            "_run_graph_ingestion_pipeline",
            return_value=(successes, {}),
        ) as pipeline:
            result = ingestion.inject_markdown_directory(source_directory)

        pipeline.assert_called_once_with(expected_files, 3)
        self.assertEqual(result["total_files"], len(expected_files))
        self.assertEqual(result["success_count"], len(expected_files))
        self.assertEqual(result["failure_count"], 0)
        self.assertEqual(result["total_chunk_count"], len(expected_files) * 2)
        self.assertEqual(result["total_entity_count"], len(expected_files) * 3)

    def test_directory_import_keeps_failures_from_pipeline(self):
        ingestion = self._ingestion_module()
        source_directory = _markdown_directory()
        failed_path = (source_directory / "tech_report_0tfhhamx.md").resolve()
        successes = {
            str((source_directory / "overview.md").resolve()): {
                "chunk_count": 1,
                "entity_count": 2,
                "document_ids": ["overview"],
            }
        }
        failures = {str(failed_path): "extract: simulated import failure"}
        with patch.object(
            ingestion,
            "_run_graph_ingestion_pipeline",
            return_value=(successes, failures),
        ):
            result = ingestion.inject_markdown_directory(
                source_directory,
                max_concurrency=2,
            )

        self.assertEqual(result["success_count"], 1)
        self.assertEqual(result["failure_count"], 1)
        self.assertEqual(result["total_chunk_count"], 1)
        self.assertEqual(result["total_entity_count"], 2)
        self.assertEqual(result["failures"], failures)

    def test_directory_import_rejects_invalid_or_empty_directory(self):
        ingestion = self._ingestion_module()

        with self.assertRaisesRegex(NotADirectoryError, "Markdown directory does not exist"):
            ingestion.inject_markdown_directory("missing-markdown-directory")

        source_directory = _markdown_directory()
        with patch.object(ingestion.Path, "iterdir", return_value=iter(())):
            with self.assertRaisesRegex(ValueError, "No Markdown files found"):
                ingestion.inject_markdown_directory(source_directory)
        with self.assertRaisesRegex(ValueError, "max_concurrency"):
            ingestion.inject_markdown_directory(source_directory, max_concurrency=0)

    def test_parser_process_continues_after_file_failure(self):
        ingestion = self._ingestion_module()
        parsed_queue = Queue()
        result_queue = Queue()
        documents = [Document(page_content="parsed", metadata={"id": "first"})]

        with patch.object(
            ingestion,
            "_parse_markdown_documents",
            side_effect=[documents, RuntimeError("simulated parse failure")],
        ):
            ingestion._parse_markdown_directory_process(
                parsed_queue,
                result_queue,
                ["first.md", "broken.md"],
            )

        self.assertEqual(parsed_queue.get(), ("first.md", documents))
        self.assertIsNone(parsed_queue.get())
        self.assertEqual(
            result_queue.get(),
            ("failure", "broken.md", "parse", "simulated parse failure"),
        )

    def test_writer_process_reuses_one_connection(self):
        ingestion = self._ingestion_module()
        graph_queue = Queue()
        result_queue = Queue()
        graph_queue.put(("report.md", ["graph-document"], {"entity_count": 1}))
        graph_queue.put(None)
        graph = MagicMock()
        database = MagicMock()
        database.connect.return_value = graph

        with patch.object(ingestion, "Neo4jDatabase", return_value=database):
            with patch.object(ingestion, "write_graph_documents") as writer:
                ingestion._neo4j_writer_process(graph_queue, result_queue)

        database.connect.assert_called_once_with()
        database.close.assert_called_once_with()
        writer.assert_called_once_with(graph, ["graph-document"])
        self.assertEqual(
            result_queue.get(),
            ("success", "report.md", {"entity_count": 1}),
        )

    def test_async_coordinator_limits_llm_requests(self):
        ingestion = self._ingestion_module()
        parsed_queue = Queue()
        graph_queue = Queue()
        result_queue = Queue()
        source_path = _markdown_directory() / "overview.md"
        documents = [
            Document(page_content=f"chunk {index}", metadata={"id": str(index), "source": str(source_path)})
            for index in range(5)
        ]
        parsed_queue.put((str(source_path), documents))
        parsed_queue.put(None)
        active_requests = 0
        max_active_requests = 0

        class FinishedProcess:
            name = "test-process"

            def is_alive(self):
                return False

            def join(self, _timeout=None):
                return None

        class FakeTransformer:
            async def aprocess_response(self, _document):
                nonlocal active_requests, max_active_requests
                active_requests += 1
                max_active_requests = max(max_active_requests, active_requests)
                try:
                    await asyncio.sleep(0.01)
                    return SimpleNamespace(nodes=[SimpleNamespace(id="Entity")])
                finally:
                    active_requests -= 1

        with patch.object(ingestion, "_create_graph_transformer", return_value=FakeTransformer()):
            successes, failures = asyncio.run(
                ingestion._coordinate_graph_ingestion(
                    parsed_queue,
                    graph_queue,
                    result_queue,
                    FinishedProcess(),
                    FinishedProcess(),
                    max_concurrency=2,
                )
            )

        queued_source, graph_documents, result = graph_queue.get()
        self.assertEqual(queued_source, str(source_path))
        self.assertEqual(len(graph_documents), 5)
        self.assertEqual(result["entity_count"], 5)
        self.assertIsNone(graph_queue.get())
        self.assertFalse(successes)
        self.assertFalse(failures)
        self.assertLessEqual(max_active_requests, 2)

    def test_graph_document_can_cross_process_queue(self):
        from langchain_neo4j.graphs.graph_document import GraphDocument, Node

        source = Document(page_content="source", metadata={"id": "source-1"})
        graph_document = GraphDocument(
            nodes=[Node(id="Silicon", type="Material")],
            relationships=[],
            source=source,
        )

        restored_document = pickle.loads(pickle.dumps(graph_document))
        self.assertEqual(restored_document.source.metadata["id"], "source-1")
        self.assertEqual(restored_document.nodes[0].id, "Silicon")

    def test_single_file_import_keeps_synchronous_public_interface(self):
        ingestion = self._ingestion_module()
        source_path = _markdown_directory() / "tech_report_0tfhhamx.md"
        documents = [
            Document(
                page_content="Plasma treats silicon.",
                metadata={"id": "source-1", "source": str(source_path.resolve())},
            )
        ]
        graph_document = SimpleNamespace(nodes=[SimpleNamespace(id="Plasma")])
        transformer = SimpleNamespace(
            convert_to_graph_documents=MagicMock(return_value=[graph_document])
        )

        with patch.object(ingestion, "_parse_markdown_documents", return_value=documents):
            with patch.object(ingestion, "_create_graph_transformer", return_value=transformer):
                with patch.object(ingestion, "_write_graph_documents_to_neo4j") as writer:
                    result = ingestion.inject_markdown_file(source_path)

        transformer.convert_to_graph_documents.assert_called_once_with(documents)
        writer.assert_called_once_with([graph_document])
        self.assertEqual(result["chunk_count"], 1)
        self.assertEqual(result["entity_count"], 1)


class Neo4jKnowledgeGraphIntegrationTest(unittest.TestCase):
    """Runs only when explicitly enabled because it calls the LLM and writes Neo4j."""

    @unittest.skipUnless(
        os.getenv("RUN_NEO4J_INTEGRATION") == "1",
        "Set RUN_NEO4J_INTEGRATION=1 to run the Neo4j integration test.",
    )
    def test_markdown_entities_are_extracted_and_imported(self):
        try:
            import langchain_neo4j
        except ImportError:
            self.fail(
                "Install neo4j and langchain-neo4j before running this test: "
                "pip install neo4j langchain-neo4j"
            )

        required_settings = (
            "NEO4J_PASSWORD",
            "DEEPSEEK_API_KEY",
            "ZHIPU_API_KEY",
        )
        missing_settings = [name for name in required_settings if not os.getenv(name)]
        self.assertFalse(
            missing_settings,
            f"Missing environment variables: {', '.join(missing_settings)}",
        )

        from src.infrastructure.document_processing.graph_ingestion import (
            inject_markdown_file,
        )

        project_root = Path(__file__).resolve().parents[1]
        source_path = project_root / "datas" / "md" / "tech_report_0tfhhamx.md"
        result = inject_markdown_file(source_path)

        self.assertGreater(
            result["entity_count"],
            0,
            "The LLM did not extract any entities from the Markdown file.",
        )
        self.assertGreater(result["chunk_count"], 0)
        self.assertTrue(result["document_ids"])


if __name__ == "__main__":
    unittest.main()

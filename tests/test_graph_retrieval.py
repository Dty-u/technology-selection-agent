import unittest
from unittest.mock import patch

from langchain_core.documents import Document


class RecordingGraph:
    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    def query(self, query, params):
        self.calls.append((query, params))
        return self.rows


class GraphRetrieverTest(unittest.TestCase):
    def test_graph_rows_become_traceable_documents_with_limits(self):
        from src.infrastructure.graphdatabase.retriever import Neo4jGraphRetriever

        rows = [
            {
                "document_id": f"doc-{index}",
                "text": f"source text {index}",
                "source": "report.md",
                "entity_names": ["GAA", f"Benefit {index}"],
                "relationships": [
                    {
                        "source": "GAA",
                        "type": "IMPROVES",
                        "target": f"Benefit {index}",
                    }
                ],
            }
            for index in range(5)
        ]
        graph = RecordingGraph(rows)
        retriever = Neo4jGraphRetriever(
            graph=graph,
            max_hops=2,
            seed_limit=5,
            path_limit=20,
            max_documents=4,
        )

        documents = retriever.invoke(
            {
                "entities": ["GAA", "FinFET"],
                "relation_intent": "compare power efficiency",
            }
        )

        self.assertEqual(len(documents), 4)
        self.assertIn("GAA -[IMPROVES]-> Benefit 0", documents[0].page_content)
        self.assertIn("source text 0", documents[0].page_content)
        self.assertEqual(documents[0].metadata["retrieval_source"], "graph")
        self.assertEqual(documents[0].metadata["id"], "doc-0")
        self.assertEqual(documents[0].metadata["source"], "report.md")
        self.assertEqual(
            graph.calls[0][1],
            {
                "entity_names": ["GAA", "FinFET"],
                "seed_limit": 5,
                "path_limit": 20,
                "document_limit": 4,
                "relation_types": [],
            },
        )
        self.assertIn("RELATED_TO*1..2", graph.calls[0][0])

    def test_entity_matching_uses_name_and_type_with_exact_match_priority(self):
        from src.infrastructure.graphdatabase.retriever import Neo4jGraphRetriever

        graph = RecordingGraph([])
        Neo4jGraphRetriever(graph=graph).invoke(
            {"entities": ["GAA"], "relation_intent": "structure"}
        )

        query = graph.calls[0][0]
        self.assertIn("coalesce(seed.type", query)
        self.assertIn("AS match_rank", query)
        self.assertIn("collect(seed)[0]", query)
        self.assertIn("ORDER BY fallback_rank, entity_coverage DESC", query)
        self.assertIn("WHERE all(path_entity IN path_entities", query)

    def test_relation_types_rank_paths_before_limit(self):
        from src.infrastructure.graphdatabase.retriever import Neo4jGraphRetriever

        graph = RecordingGraph([])
        Neo4jGraphRetriever(graph=graph).invoke(
            {"entities": ["GAA", "Power"], "relation_types": ["improves"]}
        )

        query, params = graph.calls[0]
        self.assertEqual(params["relation_types"], ["IMPROVES"])
        self.assertIn("relation_coverage DESC", query)
        self.assertLess(query.index("ORDER BY fallback_rank"), query.index("LIMIT $path_limit"))

    def test_empty_entity_list_does_not_query_neo4j(self):
        from src.infrastructure.graphdatabase.retriever import Neo4jGraphRetriever

        graph = RecordingGraph([])
        retriever = Neo4jGraphRetriever(graph=graph)

        self.assertEqual(
            retriever.invoke({"entities": [], "relation_intent": ""}),
            [],
        )
        self.assertEqual(graph.calls, [])

    def test_duplicate_paths_do_not_consume_document_limit(self):
        from src.infrastructure.graphdatabase.retriever import Neo4jGraphRetriever

        duplicate_path = {
            "document_id": "doc-1",
            "text": "same source text",
            "source": "report.md",
            "entity_names": ["GAA", "Power"],
            "relationships": [
                {"source": "GAA", "type": "IMPROVES", "target": "Power"}
            ],
        }
        second_document = {
            "document_id": "doc-2",
            "text": "different source text",
            "source": "report.md",
            "entity_names": ["GAA", "Performance"],
            "relationships": [
                {
                    "source": "GAA",
                    "type": "IMPROVES",
                    "target": "Performance",
                }
            ],
        }
        retriever = Neo4jGraphRetriever(
            graph=RecordingGraph(
                [duplicate_path, duplicate_path, second_document]
            ),
            max_documents=2,
        )

        documents = retriever.invoke(
            {"entities": ["GAA"], "relation_intent": "improves"}
        )

        self.assertEqual(
            [document.metadata["id"] for document in documents],
            ["doc-1", "doc-2"],
        )


class RetrievalNodeTest(unittest.TestCase):
    def test_graph_failure_returns_empty_branch_for_vector_fallback(self):
        from src.workflows.nodes.graph_retrieve import graph_retrieve

        with patch(
            "src.workflows.nodes.graph_retrieve.get_graph_retriever",
            side_effect=RuntimeError("neo4j unavailable"),
        ):
            result = graph_retrieve(
                {
                    "question": "What improves GAA power efficiency?",
                    "query_entities": ["GAA"],
                    "relation_intent": "improves",
                }
            )

        self.assertEqual(result["graph_documents"], [])
        self.assertEqual(result["graph_error"], "neo4j unavailable")

    def test_graph_node_passes_relation_types(self):
        from src.workflows.nodes.graph_retrieve import graph_retrieve

        retriever = patch(
            "src.workflows.nodes.graph_retrieve.get_graph_retriever"
        )
        with retriever as get_graph_retriever:
            get_graph_retriever.return_value.invoke.return_value = []
            graph_retrieve(
                {
                    "query_entities": ["GAA"],
                    "relation_intent": "improves power",
                    "relation_types": ["IMPROVES"],
                }
            )

        get_graph_retriever.return_value.invoke.assert_called_once_with(
            {
                "entities": ["GAA"],
                "relation_intent": "improves power",
                "relation_types": ["IMPROVES"],
            }
        )

    def test_merge_limits_each_branch_and_deduplicates_documents(self):
        from src.workflows.nodes.merge_retrieval_results import (
            merge_retrieval_results,
        )

        vector_documents = [
            Document(
                page_content=f"vector {index}",
                metadata={"id": f"vector-{index}", "retrieval_source": "vector"},
            )
            for index in range(5)
        ]
        graph_documents = [
            Document(
                page_content="same evidence",
                metadata={"id": "vector-0", "retrieval_source": "graph"},
            ),
            *[
                Document(
                    page_content=f"graph {index}",
                    metadata={"id": f"graph-{index}", "retrieval_source": "graph"},
                )
                for index in range(4)
            ],
        ]

        result = merge_retrieval_results(
            {
                "question": "Compare GAA and FinFET",
                "vector_documents": vector_documents,
                "graph_documents": graph_documents,
            }
        )

        self.assertEqual(
            [document.metadata["id"] for document in result["documents"]],
            [
                "vector-0",
                "vector-1",
                "vector-2",
                "vector-3",
                "graph-0",
                "graph-1",
                "graph-2",
            ],
        )
        self.assertEqual(result["retrieval_source"], "local_hybrid")

    def test_merge_keeps_graph_relation_for_duplicate_vector_document(self):
        from src.workflows.nodes.merge_retrieval_results import (
            merge_retrieval_results,
        )

        result = merge_retrieval_results(
            {
                "question": "What improves GAA power?",
                "vector_documents": [
                    Document(page_content="vector source", metadata={"id": "doc-1"})
                ],
                "graph_documents": [
                    Document(
                        page_content="图谱关系：GAA -[IMPROVES]-> Power\n来源内容：vector source",
                        metadata={"id": "doc-1"},
                    )
                ],
            }
        )

        document = result["documents"][0]
        self.assertEqual(len(result["documents"]), 1)
        self.assertEqual(document.metadata["retrieval_sources"], ["vector", "graph"])
        self.assertEqual(document.page_content.count("图谱关系："), 1)


if __name__ == "__main__":
    unittest.main()

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document
from langchain_core.messages import AIMessage


class RetrievalGraphStructureTest(unittest.TestCase):
    def test_graph_reuses_retrieval_nodes_without_answer_generation(self):
        from src.workflows.retrieval_graph import retrieval_graph

        graph_view = retrieval_graph.get_graph()

        self.assertEqual(
            set(graph_view.nodes),
            {
                "__start__",
                "analyze_question",
                "web_search",
                "vector_retrieve",
                "graph_retrieve",
                "merge_retrieval_results",
                "grade_documents",
                "transform_query",
                "extract_retrieval_intent",
                "__end__",
            },
        )
        self.assertNotIn("generate", graph_view.nodes)
        edges = {
            (edge.source, edge.target, edge.conditional)
            for edge in graph_view.edges
        }
        self.assertIn(("web_search", "__end__", False), edges)
        self.assertIn(("grade_documents", "__end__", True), edges)
        self.assertIn(
            ("transform_query", "extract_retrieval_intent", False),
            edges,
        )
        self.assertIn(
            ("extract_retrieval_intent", "vector_retrieve", False),
            edges,
        )
        self.assertIn(
            ("extract_retrieval_intent", "graph_retrieve", False),
            edges,
        )
        self.assertNotIn(
            ("transform_query", "analyze_question", False),
            edges,
        )


class RetrievalGraphBehaviorTest(unittest.TestCase):
    def test_explicit_web_preference_bypasses_local_analysis(self):
        import src.workflows.nodes.analyze_question as analysis_module
        import src.workflows.nodes.web_search as web_module
        from src.workflows.retrieval_graph import retrieval_graph

        analysis_chain = MagicMock()
        with patch.object(
            analysis_module,
            "question_analysis_chain",
            analysis_chain,
        ), patch.object(
            web_module.web_search_tool,
            "invoke",
            return_value=[
                {
                    "title": "Current roadmap",
                    "url": "https://example.com/roadmap",
                    "content": "roadmap evidence",
                }
            ],
        ):
            result = retrieval_graph.invoke(
                {
                    "question": "latest semiconductor roadmap",
                    "source_preference": "web",
                    "transform_count": 0,
                }
            )

        analysis_chain.invoke.assert_not_called()
        self.assertEqual(result["retrieval_source"], "web_search")
        self.assertEqual(
            result["documents"][0].metadata["source"],
            "https://example.com/roadmap",
        )

    def test_rewritten_query_retries_both_local_retrievers_before_web(self):
        import src.workflows.nodes.analyze_question as analysis_module
        import src.workflows.nodes.grade_documents as grade_module
        import src.workflows.nodes.graph_retrieve as graph_module
        import src.workflows.nodes.retrieve as retrieve_module
        import src.workflows.nodes.transform_query as transform_module
        import src.workflows.nodes.web_search as web_module
        from src.workflows.retrieval_graph import retrieval_graph

        analysis_chain = MagicMock()
        analysis_chain.invoke.side_effect = [
            AIMessage(
                content=(
                    '{"route":"local_hybrid","query_entities":["GAA"],'
                    '"relation_intent":"power"}'
                )
            ),
            AIMessage(
                content=(
                    '{"route":"web_search","query_entities":["GAA"],'
                    '"relation_intent":"current power"}'
                )
            ),
        ]
        intent_chain = MagicMock()
        intent_chain.invoke.return_value = AIMessage(
            content=(
                '{"query_entities":["GAA"],'
                '"relation_intent":"current power",'
                '"relation_types":["IMPROVES"]}'
            )
        )
        rewriter = MagicMock()
        rewriter.invoke.return_value = "改写后的GAA功耗证据查询"
        vector_retriever = MagicMock()
        vector_retriever.invoke.side_effect = [
            [],
            [
                Document(
                    page_content="vector evidence",
                    metadata={"id": "vector-1"},
                )
            ],
        ]
        graph_retriever = MagicMock()
        graph_retriever.invoke.side_effect = [
            [],
            [
                Document(
                    page_content="graph evidence",
                    metadata={"id": "graph-1"},
                )
            ],
        ]
        relevance_grader = MagicMock()
        relevance_grader.invoke.return_value = SimpleNamespace(
            relevant_indices=[0, 1]
        )
        web_search_tool = MagicMock()
        web_search_tool.invoke.return_value = [
            {
                "title": "Web source",
                "url": "https://example.com/web",
                "content": "web evidence",
            }
        ]

        with patch.object(
            analysis_module,
            "question_analysis_chain",
            analysis_chain,
        ), patch.object(
            analysis_module,
            "retrieval_intent_chain",
            intent_chain,
        ), patch.object(
            transform_module,
            "question_rewriter_chain",
            rewriter,
        ), patch.object(
            retrieve_module,
            "get_retriever",
            return_value=vector_retriever,
        ), patch.object(
            graph_module,
            "get_graph_retriever",
            return_value=graph_retriever,
        ), patch.object(
            grade_module,
            "retrieval_grader_chain",
            relevance_grader,
        ), patch.object(
            web_module,
            "web_search_tool",
            web_search_tool,
        ):
            result = retrieval_graph.invoke(
                {
                    "question": "GAA功耗证据",
                    "source_preference": "auto",
                    "transform_count": 0,
                }
            )

        self.assertEqual(result["retrieval_source"], "local_hybrid")
        self.assertEqual(result["transform_count"], 1)
        self.assertEqual(vector_retriever.invoke.call_count, 2)
        self.assertEqual(graph_retriever.invoke.call_count, 2)
        web_search_tool.invoke.assert_not_called()

    def test_retrieval_retry_limit_still_falls_back_to_web(self):
        from src.config.settings import MAX_TRANSFORM_COUNT
        from src.workflows.routes.technology_selection import (
            route_retrieval_result,
        )

        route = route_retrieval_result(
            {
                "documents": [],
                "transform_count": MAX_TRANSFORM_COUNT,
            }
        )

        self.assertEqual(route, "web_search")

    def test_retrieval_result_honors_per_request_transform_limit(self):
        from src.workflows.routes.technology_selection import (
            route_retrieval_result,
        )

        route = route_retrieval_result(
            {
                "documents": [],
                "transform_count": 0,
                "max_transform_count": 0,
            }
        )

        self.assertEqual(route, "web_search")


if __name__ == "__main__":
    unittest.main()

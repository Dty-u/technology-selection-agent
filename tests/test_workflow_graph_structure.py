import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda


class HybridWorkflowStructureTest(unittest.TestCase):
    def test_explicit_local_preference_overrides_a_web_analysis_decision(self):
        from src.workflows.routes.route_after_analysis import route_after_analysis

        result = route_after_analysis(
            {
                "route": "web_search",
                "source_preference": "local",
            }
        )

        self.assertEqual(result, ["vector_retrieve", "graph_retrieve"])

    def test_compiled_graph_contains_parallel_retrieval_and_merge_barrier(self):
        from src.workflows.rag_qa_graph import graph

        graph_view = graph.get_graph()
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
                "generate",
                "transform_query",
                "extract_retrieval_intent",
                "__end__",
            },
        )
        edges = {
            (edge.source, edge.target, edge.conditional)
            for edge in graph_view.edges
        }
        expected_edges = {
            ("__start__", "analyze_question", True),
            ("__start__", "web_search", True),
            ("analyze_question", "vector_retrieve", True),
            ("analyze_question", "graph_retrieve", True),
            ("analyze_question", "web_search", True),
            ("vector_retrieve", "merge_retrieval_results", False),
            ("graph_retrieve", "merge_retrieval_results", False),
            ("merge_retrieval_results", "grade_documents", False),
            ("transform_query", "extract_retrieval_intent", False),
            ("extract_retrieval_intent", "vector_retrieve", False),
            ("extract_retrieval_intent", "graph_retrieve", False),
            ("web_search", "generate", False),
        }
        self.assertTrue(expected_edges.issubset(edges), edges)
        self.assertNotIn(
            ("transform_query", "analyze_question", False),
            edges,
        )

    def test_local_question_runs_both_retrievers_before_generation(self):
        import src.workflows.nodes.analyze_question as analysis_module
        import src.workflows.nodes.generate as generate_module
        import src.workflows.nodes.grade_documents as grade_documents_module
        import src.workflows.nodes.graph_retrieve as graph_retrieve_module
        import src.workflows.nodes.retrieve as retrieve_module
        import src.workflows.routes.evaluate_generation as evaluation_module
        from src.workflows.rag_qa_graph import graph

        analysis_chain = MagicMock()
        analysis_chain.invoke.return_value = AIMessage(
            content=(
                '{"route":"local_hybrid","query_entities":["GAA"],'
                '"relation_intent":"improves power"}'
            )
        )
        vector_retriever = MagicMock()
        vector_retriever.invoke.return_value = [
            Document(
                page_content="vector evidence",
                metadata={"id": "vector-1"},
            )
        ]
        graph_retriever = MagicMock()
        graph_retriever.invoke.return_value = [
            Document(
                page_content="graph relationship evidence",
                metadata={"id": "graph-1", "retrieval_source": "graph"},
            )
        ]
        relevance_grader = MagicMock()
        relevance_grader.invoke.return_value = SimpleNamespace(
            relevant_indices=[0, 1]
        )
        groundedness_grader = MagicMock()
        groundedness_grader.invoke.return_value = SimpleNamespace(binary_score="yes")
        answer_grader = MagicMock()
        answer_grader.invoke.return_value = SimpleNamespace(binary_score="yes")
        fake_llm = RunnableLambda(lambda _prompt: AIMessage(content="final answer"))

        with ExitStack() as stack:
            stack.enter_context(
                patch.object(
                    analysis_module,
                    "question_analysis_chain",
                    analysis_chain,
                )
            )
            stack.enter_context(
                patch.object(
                    retrieve_module,
                    "get_retriever",
                    return_value=vector_retriever,
                )
            )
            stack.enter_context(
                patch.object(
                    graph_retrieve_module,
                    "get_graph_retriever",
                    return_value=graph_retriever,
                )
            )
            stack.enter_context(
                patch.object(
                    grade_documents_module,
                    "retrieval_grader_chain",
                    relevance_grader,
                )
            )
            stack.enter_context(patch.object(generate_module, "llm", fake_llm))
            stack.enter_context(
                patch.object(
                    evaluation_module,
                    "hallucination_grader_chain",
                    groundedness_grader,
                )
            )
            stack.enter_context(
                patch.object(
                    evaluation_module,
                    "answer_grader_chain",
                    answer_grader,
                )
            )
            result = graph.invoke(
                {
                    "question": "How does GAA improve power?",
                    "transform_count": 0,
                    "generation_attempt_count": 0,
                }
            )

        self.assertEqual(result["generation"], "final answer")
        self.assertEqual(
            [document.metadata["id"] for document in result["documents"]],
            ["vector-1", "graph-1"],
        )
        self.assertEqual(result["retrieval_source"], "local_hybrid")

    def test_explicit_web_preference_bypasses_local_analysis(self):
        import src.workflows.nodes.analyze_question as analysis_module
        import src.workflows.nodes.generate as generate_module
        import src.workflows.nodes.web_search as web_search_module
        import src.workflows.routes.evaluate_generation as evaluation_module
        from src.workflows.rag_qa_graph import graph

        analysis_chain = MagicMock()
        web_search_tool = MagicMock()
        web_search_tool.invoke.return_value = [
            {
                "title": "Market report",
                "url": "https://example.com/report",
                "content": "market evidence",
            }
        ]
        fake_llm = RunnableLambda(
            lambda _prompt: AIMessage(content="web answer")
        )
        yes_score = SimpleNamespace(binary_score="yes")
        grader = MagicMock()
        grader.invoke.return_value = yes_score

        with patch.object(
            analysis_module,
            "question_analysis_chain",
            analysis_chain,
        ), patch.object(
            web_search_module,
            "web_search_tool",
            web_search_tool,
        ), patch.object(
            web_search_module,
            "retrieval_grader_chain",
            MagicMock(
                invoke=MagicMock(
                    return_value=SimpleNamespace(relevant_indices=[0])
                )
            ),
            create=True,
        ), patch.object(
            generate_module,
            "llm",
            fake_llm,
        ), patch.object(
            evaluation_module,
            "hallucination_grader_chain",
            grader,
        ), patch.object(
            evaluation_module,
            "answer_grader_chain",
            grader,
        ):
            result = graph.invoke(
                {
                    "question": "生成新能源汽车行业报告",
                    "source_preference": "web",
                    "response_type": "answer",
                    "transform_count": 0,
                    "generation_attempt_count": 0,
                }
            )

        analysis_chain.invoke.assert_not_called()
        self.assertEqual(result["generation"], "web answer")
        self.assertEqual(result["retrieval_source"], "web_search")

    def test_unhelpful_web_answer_triggers_one_precise_search_retry(self):
        import src.workflows.nodes.generate as generate_module
        import src.workflows.nodes.web_search as web_search_module
        import src.workflows.routes.evaluate_generation as evaluation_module
        from src.workflows.rag_qa_graph import graph

        first_results = [
            {
                "title": "GAA industry overview",
                "url": "https://example.com/overview",
                "content": "GAA is a transistor architecture.",
            }
        ]
        second_results = [
            {
                "title": "GAA mass production update",
                "url": "https://example.com/production",
                "content": "Gate-All-Around mass production progress.",
            }
        ]
        web_search_tool = MagicMock()
        web_search_tool.invoke.side_effect = [first_results, second_results]
        relevance_grader = MagicMock()
        relevance_grader.invoke.return_value = SimpleNamespace(
            relevant_indices=[0]
        )
        query_rewriter = MagicMock()
        query_rewriter.invoke.return_value = (
            "Gate-All-Around transistor mass production progress"
        )
        answers = iter(["GAA概述", "GAA量产进展回答"])
        fake_llm = RunnableLambda(
            lambda _prompt: AIMessage(content=next(answers))
        )
        groundedness_grader = MagicMock()
        groundedness_grader.invoke.return_value = SimpleNamespace(
            binary_score="yes"
        )
        answer_grader = MagicMock()
        answer_grader.invoke.side_effect = [
            SimpleNamespace(binary_score="no"),
            SimpleNamespace(binary_score="yes"),
        ]

        with patch.object(
            web_search_module,
            "web_search_tool",
            web_search_tool,
        ), patch.object(
            web_search_module,
            "retrieval_grader_chain",
            relevance_grader,
        ), patch.object(
            web_search_module,
            "web_query_rewriter_chain",
            query_rewriter,
        ), patch.object(
            generate_module,
            "llm",
            fake_llm,
        ), patch.object(
            evaluation_module,
            "hallucination_grader_chain",
            groundedness_grader,
        ), patch.object(
            evaluation_module,
            "answer_grader_chain",
            answer_grader,
        ):
            result = graph.invoke(
                {
                    "question": "目前GAA技术的量产进展如何？",
                    "original_question": "目前GAA技术的量产进展如何？",
                    "source_preference": "web",
                    "response_type": "answer",
                    "transform_count": 0,
                    "generation_attempt_count": 0,
                    "web_search_attempt_count": 0,
                    "web_search_queries": [],
                }
            )

        self.assertEqual(web_search_tool.invoke.call_count, 2)
        self.assertEqual(result["generation"], "GAA量产进展回答")
        self.assertEqual(result["web_search_attempt_count"], 2)
        self.assertEqual(len(result["web_search_queries"]), 2)
        self.assertIsNone(result["retrieval_error"])

    def test_two_unhelpful_web_answers_stop_without_a_third_provider_call(self):
        import src.workflows.nodes.generate as generate_module
        import src.workflows.nodes.web_search as web_search_module
        import src.workflows.routes.evaluate_generation as evaluation_module
        from src.workflows.rag_qa_graph import graph

        results = [
            {
                "title": "GAA production update",
                "url": "https://example.com/gaa",
                "content": "Gate-All-Around production information.",
            }
        ]
        web_search_tool = MagicMock()
        web_search_tool.invoke.side_effect = [results, results]
        relevance_grader = MagicMock()
        relevance_grader.invoke.return_value = SimpleNamespace(
            relevant_indices=[0]
        )
        query_rewriter = MagicMock()
        query_rewriter.invoke.return_value = (
            "Gate-All-Around transistor verified mass production status"
        )
        fake_llm = RunnableLambda(
            lambda _prompt: AIMessage(content="仍未解决用户问题")
        )
        groundedness_grader = MagicMock()
        groundedness_grader.invoke.return_value = SimpleNamespace(
            binary_score="yes"
        )
        answer_grader = MagicMock()
        answer_grader.invoke.return_value = SimpleNamespace(binary_score="no")

        with patch.object(
            web_search_module,
            "web_search_tool",
            web_search_tool,
        ), patch.object(
            web_search_module,
            "retrieval_grader_chain",
            relevance_grader,
        ), patch.object(
            web_search_module,
            "web_query_rewriter_chain",
            query_rewriter,
        ), patch.object(
            generate_module,
            "llm",
            fake_llm,
        ), patch.object(
            evaluation_module,
            "hallucination_grader_chain",
            groundedness_grader,
        ), patch.object(
            evaluation_module,
            "answer_grader_chain",
            answer_grader,
        ):
            result = graph.invoke(
                {
                    "question": "目前GAA技术的量产进展如何？",
                    "original_question": "目前GAA技术的量产进展如何？",
                    "source_preference": "web",
                    "response_type": "answer",
                    "transform_count": 0,
                    "generation_attempt_count": 0,
                    "web_search_attempt_count": 0,
                    "web_search_queries": [],
                }
            )

        self.assertEqual(web_search_tool.invoke.call_count, 2)
        self.assertEqual(result["retrieval_error"], "web_search_retry_exhausted")
        self.assertEqual(result["generation"], "")
        self.assertEqual(result["web_search_attempt_count"], 2)

    def test_rewritten_query_retries_both_local_retrievers_before_web(self):
        import src.workflows.nodes.analyze_question as analysis_module
        import src.workflows.nodes.generate as generate_module
        import src.workflows.nodes.grade_documents as grade_documents_module
        import src.workflows.nodes.graph_retrieve as graph_retrieve_module
        import src.workflows.nodes.retrieve as retrieve_module
        import src.workflows.nodes.transform_query as transform_module
        import src.workflows.nodes.web_search as web_search_module
        import src.workflows.routes.evaluate_generation as evaluation_module
        from src.workflows.rag_qa_graph import graph

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
        rewriter.invoke.return_value = "改写后的GAA功耗问题"
        vector_retriever = MagicMock()
        vector_retriever.invoke.return_value = [
            Document(
                page_content="vector evidence",
                metadata={"id": "vector-1"},
            )
        ]
        graph_retriever = MagicMock()
        graph_retriever.invoke.return_value = [
            Document(
                page_content="graph evidence",
                metadata={"id": "graph-1"},
            )
        ]
        relevance_grader = MagicMock()
        relevance_grader.invoke.return_value = SimpleNamespace(
            relevant_indices=[0, 1]
        )
        yes_score = SimpleNamespace(binary_score="yes")
        groundedness_grader = MagicMock()
        groundedness_grader.invoke.return_value = yes_score
        answer_grader = MagicMock()
        answer_grader.invoke.side_effect = [
            SimpleNamespace(binary_score="no"),
            yes_score,
        ]
        fake_llm = RunnableLambda(
            lambda _prompt: AIMessage(content="local answer")
        )
        web_search_tool = MagicMock()
        web_search_tool.invoke.return_value = [
            {
                "title": "Web source",
                "url": "https://example.com/web",
                "content": "web evidence",
            }
        ]

        with ExitStack() as stack:
            stack.enter_context(
                patch.object(
                    analysis_module,
                    "question_analysis_chain",
                    analysis_chain,
                )
            )
            stack.enter_context(
                patch.object(
                    analysis_module,
                    "retrieval_intent_chain",
                    intent_chain,
                )
            )
            stack.enter_context(
                patch.object(
                    transform_module,
                    "question_rewriter_chain",
                    rewriter,
                )
            )
            stack.enter_context(
                patch.object(
                    retrieve_module,
                    "get_retriever",
                    return_value=vector_retriever,
                )
            )
            stack.enter_context(
                patch.object(
                    graph_retrieve_module,
                    "get_graph_retriever",
                    return_value=graph_retriever,
                )
            )
            stack.enter_context(
                patch.object(
                    grade_documents_module,
                    "retrieval_grader_chain",
                    relevance_grader,
                )
            )
            stack.enter_context(
                patch.object(generate_module, "llm", fake_llm)
            )
            stack.enter_context(
                patch.object(
                    evaluation_module,
                    "hallucination_grader_chain",
                    groundedness_grader,
                )
            )
            stack.enter_context(
                patch.object(
                    evaluation_module,
                    "answer_grader_chain",
                    answer_grader,
                )
            )
            stack.enter_context(
                patch.object(
                    web_search_module,
                    "web_search_tool",
                    web_search_tool,
                )
            )
            result = graph.invoke(
                {
                    "question": "GAA功耗如何？",
                    "transform_count": 0,
                    "generation_attempt_count": 0,
                }
            )

        self.assertEqual(result["retrieval_source"], "local_hybrid")
        self.assertEqual(result["transform_count"], 1)
        self.assertEqual(vector_retriever.invoke.call_count, 2)
        self.assertEqual(graph_retriever.invoke.call_count, 2)
        web_search_tool.invoke.assert_not_called()


if __name__ == "__main__":
    unittest.main()

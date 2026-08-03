import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda


class QuestionAnalysisTest(unittest.TestCase):
    def test_analysis_prompt_accepts_only_question_and_keeps_json_example_literal(self):
        from src.workflows.nodes.analyze_question import analysis_prompt

        try:
            messages = analysis_prompt.format_messages(
                question="EUV 如何影响图形精度？"
            )
        except KeyError as error:
            self.fail(f"JSON example was parsed as a prompt variable: {error}")

        self.assertIn(
            '{"route":"local_hybrid","query_entities":["实体"],'
            '"relation_intent":"关系意图","relation_types":["AFFECTS"]}',
            messages[0].content,
        )

    def test_analysis_parses_route_entities_and_relation_intent_from_one_response(self):
        from src.workflows.nodes.analyze_question import parse_question_analysis

        analysis = parse_question_analysis(
            """```json
            {
              "route": "local_hybrid",
              "query_entities": ["GAA", "FinFET"],
              "relation_intent": "compare power and performance",
              "relation_types": ["improves", "invalid"]
            }
            ```"""
        )

        self.assertEqual(analysis.route, "local_hybrid")
        self.assertEqual(analysis.query_entities, ["GAA", "FinFET"])
        self.assertEqual(
            analysis.relation_intent,
            "compare power and performance",
        )
        self.assertEqual(analysis.relation_types, ["IMPROVES"])

    def test_analysis_node_invokes_one_chain_call(self):
        import src.workflows.nodes.analyze_question as analysis_module

        response = AIMessage(
            content=(
                '{"route":"local_hybrid","query_entities":["GAA"],'
                '"relation_intent":"improves power","relation_types":["IMPROVES"]}'
            )
        )
        chain = MagicMock()
        chain.invoke.return_value = response

        with patch.object(analysis_module, "question_analysis_chain", chain):
            result = analysis_module.analyze_question(
                {"question": "How does GAA improve power?"}
            )

        self.assertEqual(
            result,
            {
                "route": "local_hybrid",
                "query_entities": ["GAA"],
                "relation_intent": "improves power",
                "relation_types": ["IMPROVES"],
            },
        )
        chain.invoke.assert_called_once_with(
            {"question": "How does GAA improve power?"}
        )

    def test_analysis_uses_fast_chain_only_for_technology_selection(self):
        import src.workflows.nodes.analyze_question as analysis_module

        regular_chain = MagicMock()
        fast_chain = MagicMock()
        response = AIMessage(
            content=(
                '{"route":"local_hybrid","query_entities":["ALD"],'
                '"relation_intent":"compare uniformity",'
                '"relation_types":["AFFECTS"]}'
            )
        )
        regular_chain.invoke.return_value = response
        fast_chain.invoke.return_value = response

        with patch.object(
            analysis_module,
            "question_analysis_chain",
            regular_chain,
        ), patch.object(
            analysis_module,
            "selection_question_analysis_chain",
            fast_chain,
        ):
            analysis_module.analyze_question(
                {"question": "ALD vs CVD", "compact_relevance": True}
            )
            analysis_module.analyze_question({"question": "ALD vs CVD"})

        fast_chain.invoke.assert_called_once()
        regular_chain.invoke.assert_called_once()

    def test_retrieval_intent_prompt_only_requires_question(self):
        from src.workflows.nodes.analyze_question import (
            retrieval_intent_prompt,
        )

        messages = retrieval_intent_prompt.format_messages(
            question="改写后的GAA功耗检索问题"
        )

        self.assertEqual(
            set(retrieval_intent_prompt.input_variables),
            {"question"},
        )
        self.assertIn(
            '"query_entities":["实体"]',
            messages[0].content,
        )

    def test_retrieval_intent_node_extracts_graph_fields_without_route(self):
        import src.workflows.nodes.analyze_question as analysis_module

        chain = MagicMock()
        chain.invoke.return_value = AIMessage(
            content=(
                '{"query_entities":["GAA"],'
                '"relation_intent":"improves power",'
                '"relation_types":["improves","invalid"],'
                '"route":"web_search"}'
            )
        )

        with patch.object(
            analysis_module,
            "retrieval_intent_chain",
            chain,
        ):
            result = analysis_module.extract_retrieval_intent(
                {"question": "改写后的GAA功耗检索问题"}
            )

        self.assertEqual(
            result,
            {
                "query_entities": ["GAA"],
                "relation_intent": "improves power",
                "relation_types": ["IMPROVES"],
            },
        )

    def test_route_fans_out_local_queries_and_keeps_web_separate(self):
        from src.workflows.routes.route_after_analysis import route_after_analysis

        self.assertEqual(
            route_after_analysis({"route": "local_hybrid"}),
            ["vector_retrieve", "graph_retrieve"],
        )
        self.assertEqual(
            route_after_analysis({"route": "web_search"}),
            "web_search",
        )


class RetrievalStateTest(unittest.TestCase):
    def test_vector_retrieve_writes_only_vector_branch(self):
        import src.workflows.nodes.retrieve as retrieve_module

        expected_documents = [Document(page_content="vector evidence")]
        retriever = MagicMock()
        retriever.invoke.return_value = expected_documents

        with patch.object(retrieve_module, "get_retriever", return_value=retriever):
            result = retrieve_module.retrieve({"question": "What is GAA?"})

        self.assertEqual(
            result,
            {
                "vector_documents": expected_documents,
                "question": "What is GAA?",
            },
        )

    def test_transform_query_clears_stale_hybrid_evidence(self):
        import src.workflows.nodes.transform_query as transform_module

        rewriter = MagicMock()
        rewriter.invoke.return_value = "Compare GAA and FinFET power efficiency"

        with patch.object(transform_module, "question_rewriter_chain", rewriter):
            result = transform_module.transform_query(
                {
                    "question": "compare them",
                    "documents": [Document(page_content="old")],
                    "vector_documents": [Document(page_content="old vector")],
                    "graph_documents": [Document(page_content="old graph")],
                    "query_entities": ["Old Entity"],
                    "relation_intent": "old relation",
                    "relation_types": ["CAUSES"],
                    "transform_count": 1,
                    "generation_attempt_count": 2,
                }
            )

        self.assertEqual(
            result,
            {
                "question": "Compare GAA and FinFET power efficiency",
                "documents": [],
                "vector_documents": [],
                "graph_documents": [],
                "query_entities": [],
                "relation_intent": "",
                "relation_types": [],
                "transform_count": 2,
                "generation_attempt_count": 0,
                "graph_error": None,
            },
        )


class DocumentRelevancePromptTest(unittest.TestCase):
    def test_prompt_requires_only_question_and_documents(self):
        from src.workflows.graders.document_relevance import grade_prompt

        self.assertEqual(
            set(grade_prompt.input_variables),
            {"question", "documents"},
        )

    def test_prompt_renders_json_example_as_literal_text(self):
        from src.workflows.graders.document_relevance import grade_prompt

        messages = grade_prompt.format_messages(
            question="EUV光刻机是什么？",
            documents="[0] EUV光刻证据",
        )

        self.assertIn(
            '{"relevant_indices":[0,2]}',
            messages[0].content,
        )


class BatchDocumentGradeTest(unittest.TestCase):
    def test_compact_relevance_only_shortens_technology_selection_input(self):
        import src.workflows.nodes.grade_documents as grade_module

        fast_grader = MagicMock()
        regular_grader = MagicMock()
        fast_grader.invoke.return_value = SimpleNamespace(relevant_indices=[0])
        document = Document(page_content="x" * 1200)
        with patch.object(
            grade_module,
            "selection_retrieval_grader_chain",
            fast_grader,
        ), patch.object(
            grade_module,
            "retrieval_grader_chain",
            regular_grader,
        ):
            grade_module.grade_documents(
                {
                    "question": "ALD uniformity",
                    "documents": [document],
                    "compact_relevance": True,
                }
            )

        formatted = fast_grader.invoke.call_args.args[0]["documents"]
        self.assertIn("x" * 400, formatted)
        self.assertNotIn("x" * 401, formatted)
        regular_grader.invoke.assert_not_called()

    def test_batch_grader_invokes_once_and_keeps_selected_order(self):
        import src.workflows.nodes.grade_documents as grade_module

        grader = MagicMock()
        grader.invoke.return_value = SimpleNamespace(
            relevant_indices=[2, 0, 99, -1]
        )
        documents = [Document(page_content=f"evidence {index}") for index in range(3)]

        with patch.object(grade_module, "retrieval_grader_chain", grader):
            result = grade_module.grade_documents(
                {"question": "What is GAA?", "documents": documents}
            )

        grader.invoke.assert_called_once()
        self.assertEqual(result["documents"], [documents[0], documents[2]])

    def test_batch_grader_failure_keeps_vector_evidence(self):
        import src.workflows.nodes.grade_documents as grade_module

        grader = MagicMock()
        grader.invoke.side_effect = ValueError("bad JSON")
        vector = Document(
            page_content="vector evidence",
            metadata={"retrieval_sources": ["vector"]},
        )
        graph = Document(
            page_content="graph evidence",
            metadata={"retrieval_sources": ["graph"]},
        )

        with patch.object(grade_module, "retrieval_grader_chain", grader):
            result = grade_module.grade_documents(
                {"question": "What is GAA?", "documents": [vector, graph]}
            )

        self.assertEqual(result["documents"], [vector])

    def test_empty_candidates_do_not_invoke_grader(self):
        import src.workflows.nodes.grade_documents as grade_module

        grader = MagicMock()
        with patch.object(grade_module, "retrieval_grader_chain", grader):
            result = grade_module.grade_documents(
                {"question": "What is GAA?", "documents": []}
            )

        grader.invoke.assert_not_called()
        self.assertEqual(result["documents"], [])


class RetryGuardTest(unittest.TestCase):
    def test_answer_quality_is_checked_against_the_original_question(self):
        import src.workflows.routes.evaluate_generation as evaluation_module

        supported = SimpleNamespace(binary_score="yes")
        useful = SimpleNamespace(binary_score="yes")
        hallucination_grader = MagicMock()
        hallucination_grader.invoke.return_value = supported
        answer_grader = MagicMock()
        answer_grader.invoke.return_value = useful

        with patch.object(
            evaluation_module,
            "hallucination_grader_chain",
            hallucination_grader,
        ), patch.object(
            evaluation_module,
            "answer_grader_chain",
            answer_grader,
        ):
            route = evaluation_module.grade_generation_v_documents_and_question(
                {
                    "original_question": "半导体现在发展的速度快不快？",
                    "question": "2026年半导体市场增长率",
                    "documents": [Document(page_content="evidence")],
                    "generation": "answer",
                }
            )

        self.assertEqual(route, "useful")
        answer_grader.invoke.assert_called_once_with(
            {
                "question": "半导体现在发展的速度快不快？",
                "generation": "answer",
            }
        )

    def test_accepted_generation_logs_evaluation_in_chinese(self):
        import src.workflows.routes.evaluate_generation as evaluation_module

        yes_score = SimpleNamespace(binary_score="yes")
        hallucination_grader = MagicMock()
        hallucination_grader.invoke.return_value = yes_score
        answer_grader = MagicMock()
        answer_grader.invoke.return_value = yes_score
        messages = []
        sink_id = evaluation_module.log.add(
            lambda message: messages.append(str(message)),
            format="{message}",
        )

        try:
            with patch.object(
                evaluation_module,
                "hallucination_grader_chain",
                hallucination_grader,
            ), patch.object(
                evaluation_module,
                "answer_grader_chain",
                answer_grader,
            ):
                route = (
                    evaluation_module.grade_generation_v_documents_and_question(
                        {
                            "question": "What is GAA?",
                            "documents": [Document(page_content="evidence")],
                            "generation": "answer",
                            "generation_attempt_count": 1,
                            "retrieval_source": "local_hybrid",
                        }
                    )
                )
        finally:
            evaluation_module.log.remove(sink_id)

        output = "".join(messages)
        self.assertEqual(route, "useful")
        self.assertIn("---幻觉评估：通过---", output)
        self.assertIn("---问题解决评估：通过---", output)
        self.assertIn("---评估决策：接受回答---", output)
        self.assertNotIn("CHECK", output)
        self.assertNotIn("DECISION", output)

    def test_generate_increments_attempt_count(self):
        import src.workflows.nodes.generate as generate_module

        fake_llm = RunnableLambda(lambda _prompt: AIMessage(content="answer"))
        with patch.object(generate_module, "llm", fake_llm):
            result = generate_module.generate(
                {
                    "question": "What is GAA?",
                    "documents": [Document(page_content="evidence")],
                    "generation_attempt_count": 1,
                }
            )

        self.assertEqual(result["generation"], "answer")
        self.assertEqual(result["generation_attempt_count"], 2)

    def test_generate_report_mode_passes_source_linked_report_context(self):
        import src.workflows.nodes.generate as generate_module

        captured_prompts = []

        def report_llm(prompt):
            captured_prompts.append(prompt.to_string())
            return AIMessage(content="# 行业报告")

        fake_llm = RunnableLambda(report_llm)
        with patch.object(generate_module, "llm", fake_llm):
            result = generate_module.generate(
                {
                    "question": "生成新能源汽车行业报告",
                    "documents": [
                        Document(
                            page_content="market evidence",
                            metadata={
                                "title": "Market source",
                                "url": "https://example.com/market",
                            },
                        )
                    ],
                    "response_type": "report",
                    "generation_attempt_count": 0,
                }
            )

        self.assertEqual(result["generation"], "# 行业报告")
        self.assertIn("Markdown", captured_prompts[0])
        self.assertIn("https://example.com/market", captured_prompts[0])

    def test_unsupported_generation_retries_then_falls_back(self):
        import src.workflows.routes.evaluate_generation as evaluation_module

        unsupported = SimpleNamespace(binary_score="no")
        hallucination_grader = MagicMock()
        hallucination_grader.invoke.return_value = unsupported
        with patch.object(
            evaluation_module,
            "hallucination_grader_chain",
            hallucination_grader,
        ):
            retry_route = (
                evaluation_module.grade_generation_v_documents_and_question(
                    {
                        "question": "What is GAA?",
                        "documents": [Document(page_content="evidence")],
                        "generation": "answer",
                        "generation_attempt_count": 1,
                        "retrieval_source": "local_hybrid",
                    }
                )
            )
            web_route = (
                evaluation_module.grade_generation_v_documents_and_question(
                    {
                        "question": "What is GAA?",
                        "documents": [Document(page_content="evidence")],
                        "generation": "answer",
                        "generation_attempt_count": 2,
                        "retrieval_source": "local_hybrid",
                    }
                )
            )
            end_route = (
                evaluation_module.grade_generation_v_documents_and_question(
                    {
                        "question": "What is GAA?",
                        "documents": [Document(page_content="web evidence")],
                        "generation": "answer",
                        "generation_attempt_count": 2,
                        "retrieval_source": "web_search",
                    }
                )
            )

        self.assertEqual(retry_route, "generate")
        self.assertEqual(web_route, "web_search")
        self.assertEqual(end_route, "web_search")

    def test_not_useful_generation_rewrites_then_falls_back(self):
        import src.workflows.routes.evaluate_generation as evaluation_module

        supported = SimpleNamespace(binary_score="yes")
        not_useful = SimpleNamespace(binary_score="no")
        hallucination_grader = MagicMock()
        hallucination_grader.invoke.return_value = supported
        answer_grader = MagicMock()
        answer_grader.invoke.return_value = not_useful
        with patch.object(
            evaluation_module,
            "hallucination_grader_chain",
            hallucination_grader,
        ):
            with patch.object(
                evaluation_module,
                "answer_grader_chain",
                answer_grader,
            ):
                rewrite_route = (
                    evaluation_module.grade_generation_v_documents_and_question(
                        {
                            "question": "What is GAA?",
                            "documents": [Document(page_content="evidence")],
                            "generation": "answer",
                            "generation_attempt_count": 1,
                            "transform_count": 0,
                            "retrieval_source": "local_hybrid",
                        }
                    )
                )
                web_route = (
                    evaluation_module.grade_generation_v_documents_and_question(
                        {
                            "question": "What is GAA?",
                            "documents": [Document(page_content="evidence")],
                            "generation": "answer",
                            "generation_attempt_count": 1,
                            "transform_count": 2,
                            "retrieval_source": "local_hybrid",
                        }
                    )
                )
                exhausted_route = (
                    evaluation_module.grade_generation_v_documents_and_question(
                        {
                            "question": "What is GAA?",
                            "documents": [Document(page_content="web evidence")],
                            "generation": "answer",
                            "generation_attempt_count": 1,
                            "transform_count": 2,
                            "retrieval_source": "web_search",
                            "web_search_attempt_count": 2,
                        }
                    )
                )

        self.assertEqual(rewrite_route, "transform_query")
        self.assertEqual(web_route, "web_search")
        self.assertEqual(exhausted_route, "web_search")

    def test_retrieval_error_ends_without_running_answer_graders(self):
        import src.workflows.routes.evaluate_generation as evaluation_module

        with patch.object(
            evaluation_module,
            "hallucination_grader_chain",
        ) as hallucination_grader, patch.object(
            evaluation_module,
            "answer_grader_chain",
        ) as answer_grader:
            route = evaluation_module.grade_generation_v_documents_and_question(
                {
                    "question": "目前GAA技术的量产进展如何？",
                    "documents": [],
                    "generation": "",
                    "retrieval_error": "web_search_retry_exhausted",
                }
            )

        self.assertEqual(route, "end")
        hallucination_grader.invoke.assert_not_called()
        answer_grader.invoke.assert_not_called()

    def test_generation_skips_model_when_retrieval_has_failed(self):
        from langchain_core.messages import AIMessage
        from langchain_core.runnables import RunnableLambda

        import src.workflows.nodes.generate as generate_module

        model_calls = []

        def fake_model(prompt):
            model_calls.append(prompt)
            return AIMessage(content="should not be generated")

        with patch.object(
            generate_module,
            "llm",
            RunnableLambda(fake_model),
        ):
            result = generate_module.generate(
                {
                    "question": "目前GAA技术的量产进展如何？",
                    "documents": [],
                    "retrieval_error": "web_search_retry_exhausted",
                    "generation_attempt_count": 1,
                }
            )

        self.assertEqual(result["generation"], "")
        self.assertEqual(result["generation_attempt_count"], 1)
        self.assertEqual(model_calls, [])


if __name__ == "__main__":
    unittest.main()

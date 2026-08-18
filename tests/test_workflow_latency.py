import unittest
import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document
from langchain_core.messages import AIMessage


class DirectResponseLatencyTest(unittest.TestCase):
    def test_direct_response_uses_one_bounded_llm_call(self):
        from src.workflows.nodes import query_router

        chain = MagicMock()
        chain.invoke.return_value = AIMessage(content="你好，请问有什么半导体问题？")
        with patch.object(query_router, "direct_response_chain", chain):
            result = query_router.respond_directly(
                {
                    "query": "你好",
                    "direct_response_kind": "greeting",
                }
            )

        chain.invoke.assert_called_once_with(
            {"direct_response_kind": "greeting", "query": "你好"}
        )
        self.assertEqual(result["direct_answer"], "你好，请问有什么半导体问题？")
        self.assertEqual(result["status"], "completed")

    def test_obvious_out_of_scope_uses_only_the_bounded_response_model(self):
        from src.workflows.nodes import query_router
        from src.workflows.query_router_graph import query_router_graph

        rag_graph = MagicMock()
        selection_graph = MagicMock()
        direct_chain = MagicMock()
        direct_chain.invoke.return_value = AIMessage(
            content="这个请求超出半导体服务范围。"
        )
        with patch.object(
            query_router,
            "query_classification_chain",
        ) as classifier, patch.object(
            query_router,
            "direct_response_chain",
            direct_chain,
        ), patch.object(
            query_router,
            "rag_qa_graph",
            rag_graph,
        ), patch.object(
            query_router,
            "technology_selection_graph",
            selection_graph,
        ):
            result = query_router_graph.invoke(
                {"query": "帮我写一首关于海边的诗。"}
            )

        self.assertEqual(result["query_type"], "direct_response")
        self.assertEqual(result["details"]["direct_response_kind"], "out_of_scope")
        classifier.invoke.assert_not_called()
        direct_chain.invoke.assert_called_once_with(
            {
                "direct_response_kind": "out_of_scope",
                "query": "帮我写一首关于海边的诗。",
            }
        )
        rag_graph.invoke.assert_not_called()
        selection_graph.invoke.assert_not_called()


class RetrievalAnalysisReuseTest(unittest.TestCase):
    def test_classifier_returns_validated_retrieval_analysis(self):
        from src.workflows.nodes import query_router

        response = AIMessage(
            content=json.dumps(
                {
                    "query_type": "rag_qa",
                    "source_preference": "auto",
                    "response_type": "answer",
                    "direct_response_kind": None,
                    "reason": "stable semiconductor question",
                    "retrieval_analysis": {
                        "route": "local_hybrid",
                        "query_entities": ["GAA", "FinFET"],
                        "relation_intent": "compare gate control",
                        "relation_types": ["AFFECTS"],
                    },
                }
            )
        )
        with patch.object(query_router, "query_classification_chain") as chain:
            chain.invoke.return_value = response
            result = query_router.classify_query(
                {"query": "Explain GAA versus FinFET"}
            )

        self.assertEqual(
            result["retrieval_analysis"]["route"],
            "local_hybrid",
        )
        self.assertEqual(
            result["retrieval_analysis"]["query_entities"],
            ["GAA", "FinFET"],
        )

    def test_analyze_question_reuses_valid_precomputed_analysis(self):
        from src.workflows.nodes import analyze_question

        state = {
            "question": "GAA and FinFET",
            "analysis_ready": True,
            "route": "local_hybrid",
            "query_entities": ["GAA", "FinFET"],
            "relation_intent": "compare gate control",
            "relation_types": ["AFFECTS"],
        }
        with patch.object(
            analyze_question,
            "question_analysis_chain",
        ) as chain:
            result = analyze_question.analyze_question(state)

        chain.invoke.assert_not_called()
        self.assertEqual(result["route"], "local_hybrid")
        self.assertEqual(result["query_entities"], ["GAA", "FinFET"])
        self.assertEqual(result["relation_types"], ["AFFECTS"])

    def test_invalid_precomputed_analysis_calls_model_once(self):
        from src.workflows.nodes import analyze_question

        with patch.object(analyze_question, "question_analysis_chain") as chain:
            chain.invoke.return_value = AIMessage(
                content=json.dumps(
                    {
                        "route": "local_hybrid",
                        "query_entities": ["EUV"],
                        "relation_intent": "defect causes",
                        "relation_types": ["CAUSES"],
                    }
                )
            )
            result = analyze_question.analyze_question(
                {
                    "question": "EUV defects",
                    "analysis_ready": True,
                    "route": "invalid",
                }
            )

        chain.invoke.assert_called_once()
        self.assertEqual(result["route"], "local_hybrid")

    def test_invoke_rag_passes_router_analysis_to_child_graph(self):
        from src.workflows.nodes import query_router

        child_graph = MagicMock()
        child_graph.invoke.return_value = {"generation": "answer"}
        analysis = {
            "route": "local_hybrid",
            "query_entities": ["EUV"],
            "relation_intent": "defect causes",
            "relation_types": ["CAUSES"],
        }
        with patch.object(query_router, "rag_qa_graph", child_graph):
            query_router.invoke_rag_qa(
                {
                    "query": "EUV defects",
                    "retrieval_analysis": analysis,
                    "source_preference": "auto",
                    "response_type": "answer",
                }
            )

        payload = child_graph.invoke.call_args.args[0]
        self.assertTrue(payload["analysis_ready"])
        self.assertEqual(payload["route"], "local_hybrid")
        self.assertEqual(payload["query_entities"], ["EUV"])
        self.assertEqual(payload["relation_types"], ["CAUSES"])


class CombinedGenerationAssessmentTest(unittest.TestCase):
    @staticmethod
    def _state():
        return {
            "question": "What is GAA?",
            "original_question": "What is GAA?",
            "documents": [Document(page_content="GAA surrounds the channel.")],
            "generation": "GAA surrounds the channel.",
            "generation_attempt_count": 1,
            "transform_count": 0,
            "retrieval_source": "local_hybrid",
        }

    def test_valid_combined_assessment_uses_one_model_call(self):
        from src.workflows.routes import evaluate_generation

        assessment = SimpleNamespace(
            grounded="yes",
            answers_question="yes",
        )
        with patch.object(
            evaluate_generation,
            "generation_assessment_chain",
        ) as combined, patch.object(
            evaluate_generation,
            "hallucination_grader_chain",
        ) as legacy_grounded, patch.object(
            evaluate_generation,
            "answer_grader_chain",
        ) as legacy_answer:
            combined.invoke.return_value = assessment
            route = (
                evaluate_generation.grade_generation_v_documents_and_question(
                    self._state()
                )
            )

        self.assertEqual(route, "useful")
        combined.invoke.assert_called_once()
        legacy_grounded.invoke.assert_not_called()
        legacy_answer.invoke.assert_not_called()

    def test_combined_assessment_prompt_keeps_json_schema_literal(self):
        from src.workflows.graders.generation import assessment_prompt

        messages = assessment_prompt.format_messages(
            documents="evidence",
            question="question",
            generation="answer",
        )
        self.assertIn('"grounded"', messages[0].content)

    def test_invalid_combined_assessment_falls_back_to_legacy_graders(self):
        from src.workflows.routes import evaluate_generation

        with patch.object(
            evaluate_generation,
            "generation_assessment_chain",
        ) as combined, patch.object(
            evaluate_generation,
            "hallucination_grader_chain",
        ) as legacy_grounded, patch.object(
            evaluate_generation,
            "answer_grader_chain",
        ) as legacy_answer:
            combined.invoke.side_effect = ValueError("invalid JSON")
            legacy_grounded.invoke.return_value = SimpleNamespace(
                binary_score="yes"
            )
            legacy_answer.invoke.return_value = SimpleNamespace(
                binary_score="yes"
            )
            route = (
                evaluate_generation.grade_generation_v_documents_and_question(
                    self._state()
                )
            )

        self.assertEqual(route, "useful")
        legacy_grounded.invoke.assert_called_once()
        legacy_answer.invoke.assert_called_once()


class UtilityModelConfigurationTest(unittest.TestCase):
    def test_chat_models_have_bounded_timeouts_and_retries(self):
        from src.infrastructure.llm import chat_model
        from src.config.settings import WEB_SEARCH_TIMEOUT_SECONDS

        self.assertEqual(chat_model.llm.request_timeout, 60)
        self.assertEqual(chat_model.llm.max_retries, 1)
        self.assertEqual(chat_model.utility_llm.request_timeout, 20)
        self.assertEqual(chat_model.utility_llm.max_retries, 1)
        self.assertEqual(chat_model.utility_llm.temperature, 0.0)
        self.assertEqual(
            chat_model.utility_llm.extra_body["thinking"],
            {"type": "disabled"},
        )
        self.assertEqual(WEB_SEARCH_TIMEOUT_SECONDS, 15)


class MemoryLatencyTest(unittest.TestCase):
    def test_empty_authoritative_scope_skips_semantic_projection(self):
        from src.workflows.nodes.memory import make_recall_memory

        repository = MagicMock()
        repository.search_authoritative.return_value = []
        semantic_reader = MagicMock()
        node = make_recall_memory(repository, semantic_reader)

        result = node(
            {"query": "Explain GAA", "query_type": "rag_qa"},
            {
                "configurable": {
                    "user_id": "user-a",
                    "thread_id": "thread-a",
                    "request_id": "request-a",
                }
            },
        )

        semantic_reader.search_hits.assert_not_called()
        repository.get_by_ids.assert_not_called()
        self.assertEqual(result["recalled_memory_ids"], [])

    def test_persist_node_clears_only_ephemeral_large_fields(self):
        from src.workflows.nodes.memory import make_persist_memory_candidates

        repository = MagicMock()
        node = make_persist_memory_candidates(repository)
        result = node(
            {
                "memory_candidates": [],
                "child_graph_result": {"large": "payload"},
                "memory_context": "derived long-term context",
                "historical_memory_context": "derived history",
                "short_term_context": "derived conversation context",
                "messages": ["must remain in checkpoint"],
                "session_user_id": "user-a",
            }
        )

        self.assertEqual(result["memory_candidates"], [])
        self.assertEqual(result["child_graph_result"], {})
        self.assertEqual(result["memory_context"], "")
        self.assertEqual(result["historical_memory_context"], "")
        self.assertEqual(result["short_term_context"], "")
        self.assertNotIn("messages", result)
        self.assertNotIn("session_user_id", result)


class ResourceWarmupAndCooldownTest(unittest.TestCase):
    def test_failure_cooldown_uses_injected_clock_and_recovers(self):
        from src.config.bootstrap import ResourceFailureCooldown

        now = [100.0]
        cooldown = ResourceFailureCooldown(30, clock=lambda: now[0])

        self.assertTrue(cooldown.allow("vector"))
        cooldown.mark_failed("vector")
        self.assertFalse(cooldown.allow("vector"))
        now[0] += 30
        self.assertTrue(cooldown.allow("vector"))
        cooldown.mark_ready("vector")
        self.assertTrue(cooldown.allow("vector"))

    def test_memory_milvus_warmup_only_checks_collection(self):
        from src.infrastructure.memory.milvus import MilvusMemoryIndex

        client = MagicMock()
        client.has_collection.return_value = True
        embedding = MagicMock()
        index = MilvusMemoryIndex(client=client, embedding=embedding)

        self.assertTrue(index.warm())
        client.has_collection.assert_called_once_with(
            collection_name=index.collection_name
        )
        client.create_collection.assert_not_called()
        embedding.embed_query.assert_not_called()

    def test_both_local_retrievers_unavailable_route_directly_to_web(self):
        from src.workflows.routes.decide_to_generate import decide_to_generate
        from src.workflows.routes.technology_selection import (
            route_retrieval_result,
        )

        failed_state = {
            "documents": [],
            "vector_error": "cooldown",
            "graph_error": "cooldown",
            "transform_count": 0,
            "allow_web_fallback": True,
        }
        self.assertEqual(decide_to_generate(failed_state), "web_search")
        self.assertEqual(route_retrieval_result(failed_state), "web_search")

        one_path_available = {**failed_state, "vector_error": None}
        self.assertEqual(decide_to_generate(one_path_available), "transform_query")
        self.assertEqual(
            route_retrieval_result(one_path_available),
            "transform_query",
        )


class BenchmarkContractTest(unittest.TestCase):
    def test_benchmark_has_three_samples_per_path_and_median_summary(self):
        from scripts.benchmark_workflow_latency import SAMPLES, summarize

        self.assertEqual(set(SAMPLES), {"direct", "local", "web", "selection"})
        self.assertTrue(all(len(samples) == 3 for samples in SAMPLES.values()))
        summary = summarize(
            [
                {"path": "local", "elapsed_seconds": value}
                for value in (3.0, 1.0, 2.0)
            ]
        )
        self.assertEqual(summary["local"]["median_seconds"], 2.0)
        self.assertEqual(summary["local"]["max_seconds"], 3.0)

    def test_benchmark_marks_incomplete_or_error_samples_not_comparable(self):
        from scripts.benchmark_workflow_latency import summarize

        summary = summarize(
            [
                {
                    "path": "web",
                    "elapsed_seconds": 1.0,
                    "status": "completed",
                },
                {
                    "path": "web",
                    "elapsed_seconds": 2.0,
                    "status": "error",
                },
            ]
        )

        self.assertEqual(summary["web"]["terminal_sample_count"], 1)
        self.assertFalse(summary["web"]["comparison_eligible"])


class GraphTopologyRegressionTest(unittest.TestCase):
    def test_router_node_and_edge_topology_is_unchanged(self):
        from src.workflows.query_router_graph import query_router_graph

        graph_view = query_router_graph.get_graph()
        self.assertEqual(
            set(graph_view.nodes),
            {
                "__start__",
                "__end__",
                "prepare_short_term_memory",
                "classify_query",
                "invoke_rag_qa",
                "invoke_technology_selection",
                "respond_directly",
                "build_query_response",
                "finalize_short_term_memory",
            },
        )
        self.assertEqual(
            {
                (edge.source, edge.target, edge.conditional)
                for edge in graph_view.edges
            },
            {
                ("__start__", "prepare_short_term_memory", False),
                ("prepare_short_term_memory", "classify_query", True),
                ("prepare_short_term_memory", "build_query_response", True),
                ("classify_query", "invoke_rag_qa", True),
                ("classify_query", "invoke_technology_selection", True),
                ("classify_query", "respond_directly", True),
                ("classify_query", "build_query_response", True),
                ("invoke_rag_qa", "build_query_response", False),
                ("invoke_technology_selection", "build_query_response", False),
                ("respond_directly", "build_query_response", False),
                ("build_query_response", "finalize_short_term_memory", False),
                ("finalize_short_term_memory", "__end__", False),
            },
        )

    def test_rag_node_and_edge_topology_is_unchanged(self):
        from src.workflows.rag_qa_graph import graph

        graph_view = graph.get_graph()
        self.assertEqual(
            set(graph_view.nodes),
            {
                "__start__",
                "__end__",
                "analyze_question",
                "extract_retrieval_intent",
                "web_search",
                "vector_retrieve",
                "graph_retrieve",
                "merge_retrieval_results",
                "grade_documents",
                "generate",
                "transform_query",
            },
        )
        self.assertEqual(
            {
                (edge.source, edge.target, edge.conditional)
                for edge in graph_view.edges
            },
            {
                ("__start__", "analyze_question", True),
                ("__start__", "web_search", True),
                ("analyze_question", "graph_retrieve", True),
                ("analyze_question", "vector_retrieve", True),
                ("analyze_question", "web_search", True),
                ("extract_retrieval_intent", "graph_retrieve", False),
                ("extract_retrieval_intent", "vector_retrieve", False),
                ("vector_retrieve", "merge_retrieval_results", False),
                ("graph_retrieve", "merge_retrieval_results", False),
                ("merge_retrieval_results", "grade_documents", False),
                ("grade_documents", "generate", True),
                ("grade_documents", "transform_query", True),
                ("grade_documents", "web_search", True),
                ("transform_query", "extract_retrieval_intent", False),
                ("web_search", "generate", False),
                ("generate", "__end__", True),
                ("generate", "generate", True),
                ("generate", "transform_query", True),
                ("generate", "web_search", True),
            },
        )


class TechnologySelectionLatencyTest(unittest.TestCase):
    def test_planned_local_task_supplies_precomputed_retrieval_intent(self):
        from src.workflows.nodes.selection_evidence import plan_evidence_tasks

        result = plan_evidence_tasks(
            {
                "user_request": "Select ALD or CVD",
                "requirements": {
                    "business_goal": "Improve conformality",
                    "application_scenario": "High aspect ratio trench",
                    "active_criteria": ["performance", "power"],
                    "hard_constraints": [],
                    "scenario_criteria": [],
                },
                "feasible_candidates": [
                    {"candidate_id": "ald", "name": "ALD"},
                    {"candidate_id": "cvd", "name": "CVD"},
                ],
            }
        )

        local_task = next(
            task
            for task in result["evidence_tasks"]
            if task["source_preference"] == "local"
        )
        self.assertEqual(local_task["query_entities"], ["ALD", "CVD"])
        self.assertTrue(local_task["relation_intent"])
        self.assertIn("AFFECTS", local_task["relation_types"])

    def test_local_evidence_retrieval_passes_precomputed_intent_to_subgraph(self):
        import src.workflows.nodes.selection_evidence as evidence_module
        from src.workflows.technology_selection_state import EvidenceTask

        child_graph = MagicMock()
        child_graph.invoke.return_value = {"documents": []}
        task = EvidenceTask(
            task_id="technical",
            candidate_ids=["ald", "cvd"],
            query="Compare ALD and CVD",
            criteria=["performance"],
            source_preference="local",
            query_entities=["ALD", "CVD"],
            relation_intent="compare conformality",
            relation_types=["AFFECTS"],
        )
        with patch.object(
            evidence_module,
            "retrieval_graph",
            child_graph,
        ), patch.object(
            evidence_module,
            "get_selection_cache",
            return_value=None,
        ):
            evidence_module._retrieve_task(task)

        payload = child_graph.invoke.call_args.args[0]
        self.assertTrue(payload["analysis_ready"])
        self.assertEqual(payload["query_entities"], ["ALD", "CVD"])
        self.assertEqual(payload["relation_types"], ["AFFECTS"])

    def test_evaluation_precomputes_risks_for_risk_node_reuse(self):
        import src.workflows.nodes.selection_decision as decision_module
        from src.workflows.selection_cache import clear_selection_cache

        clear_selection_cache()
        evaluation_chain = MagicMock()
        evaluation_chain.invoke.return_value = AIMessage(
            content=json.dumps(
                {
                    "evaluations": [
                        {
                            "candidate_id": "ald",
                            "criterion_scores": [
                                {
                                    "criterion": "performance",
                                    "score": 4,
                                    "rationale": "supported",
                                    "evidence_ids": ["E1"],
                                    "confidence": "high",
                                }
                            ],
                        }
                    ],
                    "risk_items": [
                        {
                            "candidate_id": "ald",
                            "category": "integration",
                            "level": "medium",
                            "impact": "Integration effort",
                            "mitigation": "Pilot validation",
                            "evidence_ids": ["E1"],
                            "basis": "evidence",
                        }
                    ],
                }
            )
        )
        state = {
            "requirements": {
                "active_criteria": ["performance"],
                "priorities": {"performance": 1},
            },
            "feasible_candidates": [
                {"candidate_id": "ald", "name": "ALD"}
            ],
            "evidence_records": [
                {
                    "evidence_id": "E1",
                    "candidate_ids": ["ald"],
                    "supports": [
                        {
                            "candidate_id": "ald",
                            "criterion": "performance",
                            "level": "direct",
                            "quote": "performance",
                        }
                    ],
                    "content": "performance integration evidence",
                }
            ],
            "evidence_gaps": [],
        }
        with patch.object(
            decision_module,
            "candidate_evaluation_chain",
            evaluation_chain,
        ):
            evaluated = decision_module.evaluate_candidates(state)

        risk_chain = MagicMock()
        with patch.object(decision_module, "risk_analysis_chain", risk_chain):
            risks = decision_module.analyze_risks(
                {
                    **state,
                    **evaluated,
                }
            )

        risk_chain.invoke.assert_not_called()
        self.assertEqual(risks["risk_items"][0]["candidate_id"], "ald")
        self.assertEqual(risks["risk_items"][0]["evidence_ids"], ["E1"])

    def test_invalid_precomputed_risks_fall_back_to_dedicated_chain(self):
        import src.workflows.nodes.selection_decision as decision_module
        from src.workflows.selection_cache import clear_selection_cache

        clear_selection_cache()
        risk_chain = MagicMock()
        risk_chain.invoke.return_value = AIMessage(
            content=json.dumps(
                {
                    "risk_items": [
                        {
                            "candidate_id": "ald",
                            "category": "integration",
                            "level": "medium",
                            "impact": "Integration effort",
                            "mitigation": "Pilot validation",
                            "evidence_ids": ["E1"],
                            "basis": "evidence",
                        }
                    ]
                }
            )
        )
        with patch.object(decision_module, "risk_analysis_chain", risk_chain):
            result = decision_module.analyze_risks(
                {
                    "requirements": {},
                    "evaluations": [],
                    "precomputed_risk_items": {"invalid": True},
                    "evidence_records": [
                        {"evidence_id": "E1", "candidate_ids": ["ald"]}
                    ],
                }
            )

        risk_chain.invoke.assert_called_once()
        self.assertEqual(result["risk_items"][0]["evidence_ids"], ["E1"])

    def test_invalid_precomputed_risk_basis_is_repaired_before_fallback(self):
        import src.workflows.nodes.selection_decision as decision_module
        from src.workflows.selection_cache import clear_selection_cache

        clear_selection_cache()
        repair_chain = MagicMock()
        repair_chain.invoke.return_value = AIMessage(
            content=json.dumps(
                {
                    "risk_items": [
                        {
                            "candidate_id": "chiplet",
                            "category": "integration",
                            "level": "high",
                            "impact": "Chiplet integration has uncertainty.",
                            "mitigation": "Validate the package and interconnect.",
                            "evidence_ids": ["E1"],
                            "basis": "evidence",
                        }
                    ]
                }
            )
        )
        risk_chain = MagicMock()
        with patch.object(
            decision_module,
            "risk_repair_chain",
            repair_chain,
        ), patch.object(decision_module, "risk_analysis_chain", risk_chain):
            result = decision_module.analyze_risks(
                {
                    "requirements": {},
                    "evaluations": [],
                    "precomputed_risk_items": [
                        {
                            "candidate_id": "chiplet",
                            "category": "integration",
                            "level": "high",
                            "impact": "Chiplet integration has uncertainty.",
                            "mitigation": "Validate the package and interconnect.",
                            "evidence_ids": ["E1"],
                            "basis": "证据显示Chiplet技术存在不确定性。",
                        }
                    ],
                    "evidence_records": [
                        {"evidence_id": "E1", "candidate_ids": ["chiplet"]}
                    ],
                }
            )

        repair_chain.invoke.assert_called_once()
        risk_chain.invoke.assert_not_called()
        self.assertEqual(result["risk_items"][0]["basis"], "evidence")

    def test_invalid_precomputed_risk_repair_falls_back_to_risk_chain(self):
        import src.workflows.nodes.selection_decision as decision_module
        from src.workflows.selection_cache import clear_selection_cache

        clear_selection_cache()
        invalid_precomputed_risk = {
            "candidate_id": "chiplet",
            "category": "integration",
            "level": "high",
            "impact": "Chiplet integration has uncertainty.",
            "mitigation": "Validate the package and interconnect.",
            "evidence_ids": ["E1"],
            "basis": "证据显示Chiplet技术存在不确定性。",
        }
        repair_chain = MagicMock()
        repair_chain.invoke.return_value = AIMessage(
            content=json.dumps({"risk_items": [invalid_precomputed_risk]})
        )
        risk_chain = MagicMock()
        risk_chain.invoke.return_value = AIMessage(
            content=json.dumps(
                {
                    "risk_items": [
                        {
                            **invalid_precomputed_risk,
                            "basis": "evidence",
                        }
                    ]
                }
            )
        )
        with patch.object(
            decision_module,
            "risk_repair_chain",
            repair_chain,
        ), patch.object(decision_module, "risk_analysis_chain", risk_chain):
            result = decision_module.analyze_risks(
                {
                    "requirements": {},
                    "evaluations": [],
                    "precomputed_risk_items": [invalid_precomputed_risk],
                    "evidence_records": [
                        {"evidence_id": "E1", "candidate_ids": ["chiplet"]}
                    ],
                }
            )

        repair_chain.invoke.assert_called_once()
        risk_chain.invoke.assert_called_once()
        self.assertEqual(result["risk_items"][0]["basis"], "evidence")


if __name__ == "__main__":
    unittest.main()

import json
import threading
import time
import unittest
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document
from langchain_core.messages import AIMessage


class TechnologySelectionStateTest(unittest.TestCase):
    def test_requirements_default_to_equal_weights_and_no_hard_constraints(self):
        from src.workflows.technology_selection_state import (
            DEFAULT_CRITERIA_WEIGHTS,
            SelectionRequirements,
        )

        requirements = SelectionRequirements(
            business_goal="降低功耗",
            application_scenario="移动芯片",
        )

        self.assertEqual(
            requirements.priorities,
            DEFAULT_CRITERIA_WEIGHTS,
        )
        self.assertEqual(requirements.hard_constraints, [])

    def test_requirements_include_independent_power_weight(self):
        from src.workflows.technology_selection_state import SelectionRequirements

        requirements = SelectionRequirements(
            business_goal="降低功耗",
            application_scenario="高端手机SoC",
        )

        self.assertEqual(
            requirements.priorities,
            {
                "performance": 1 / 7,
                "power": 1 / 7,
                "thermal": 1 / 7,
                "cost": 1 / 7,
                "maturity": 1 / 7,
                "risk": 1 / 7,
                "scalability": 1 / 7,
            },
        )

        legacy_requirements = SelectionRequirements(
            business_goal="降低成本",
            application_scenario="成熟制程量产",
            priorities={
                "performance": 1,
                "cost": 3,
                "maturity": 0,
                "risk": 0,
                "scalability": 0,
            },
        )
        self.assertEqual(legacy_requirements.priorities["power"], 0.0)
    def test_requirement_weights_are_normalized_and_unknown_criteria_are_removed(self):
        from src.workflows.technology_selection_state import SelectionRequirements

        requirements = SelectionRequirements(
            business_goal="降低成本",
            application_scenario="成熟制程量产",
            priorities={
                "performance": 1,
                "cost": 3,
                "unknown": 100,
            },
        )

        self.assertEqual(
            requirements.priorities,
            {
                "performance": 0.25,
                "power": 0.0,
                "thermal": 0.0,
                "cost": 0.75,
                "maturity": 0.0,
                "risk": 0.0,
                "scalability": 0.0,
            },
        )

    def test_scenario_criteria_are_bounded_and_receive_stable_ids(self):
        from src.workflows.technology_selection_state import SelectionRequirements

        raw_criteria = [
            {"name": "薄膜均匀性与共形性", "priority": "highest"},
            {"name": "台阶覆盖率", "priority": "high"},
            {"name": "缺陷密度", "priority": "medium"},
            {"name": "界面质量", "priority": "low"},
        ]

        first = SelectionRequirements(scenario_criteria=raw_criteria)
        second = SelectionRequirements(scenario_criteria=raw_criteria)

        self.assertEqual(len(first.scenario_criteria), 3)
        self.assertTrue(
            all(item.criterion_id for item in first.scenario_criteria)
        )
        self.assertEqual(
            [item.criterion_id for item in first.scenario_criteria],
            [item.criterion_id for item in second.scenario_criteria],
        )
        self.assertEqual(first.scenario_criteria[0].priority, "highest")

    def test_active_criteria_are_bounded_deduplicated_and_validated(self):
        from src.workflows.technology_selection_state import SelectionRequirements

        requirements = SelectionRequirements(
            active_criteria=[
                "performance",
                "cost",
                "performance",
                "unknown",
                "maturity",
                "power",
                "thermal",
                "risk",
            ]
        )

        self.assertEqual(
            requirements.active_criteria,
            [
                "performance",
                "cost",
                "maturity",
                "power",
                "thermal",
                "risk",
            ],
        )


class SelectionLLMConfigurationTest(unittest.TestCase):
    def test_selection_llm_disables_thinking_without_changing_shared_llm(self):
        from src.infrastructure.llm.chat_model import llm, selection_llm

        self.assertEqual(
            selection_llm.kwargs,
            {
                "temperature": 0.0,
                "extra_body": {"thinking": {"type": "disabled"}},
            },
        )
        self.assertNotIn("extra_body", llm.model_kwargs)

    def test_selection_json_llm_requests_json_object_mode(self):
        from src.infrastructure.llm.chat_model import (
            llm,
            selection_json_llm,
            selection_llm,
        )

        self.assertEqual(
            selection_json_llm.kwargs["response_format"],
            {"type": "json_object"},
        )
        self.assertNotIn("response_format", selection_llm.kwargs)
        self.assertNotIn("response_format", llm.model_kwargs)


class RequirementAndCandidateNodeTest(unittest.TestCase):
    def test_explicit_request_recovers_empty_model_fields_and_scenario_priority(self):
        import src.workflows.nodes.selection_requirements as requirements_module
        from src.workflows.selection_cache import clear_selection_cache

        clear_selection_cache()
        chain = MagicMock()
        chain.invoke.return_value = AIMessage(
            content=json.dumps(
                {
                    "business_goal": "",
                    "application_scenario": "",
                    "priorities": {},
                    "hard_constraints": [],
                    "requested_candidates": [],
                    "scenario_criteria": [],
                    "missing_information": [
                        "business_goal",
                        "application_scenario",
                        "requested_candidates",
                    ],
                },
                ensure_ascii=False,
            )
        )
        request = (
            "业务目标：优先保证薄膜均匀性与共形性；"
            "应用场景：高深宽比结构薄膜沉积；"
            "候选方案：ALD、CVD；"
            "薄膜均匀性与共形性为最高优先级，同时评估生产效率、成本和成熟度。"
        )

        with patch.object(
            requirements_module,
            "requirement_analysis_chain",
            chain,
        ):
            result = requirements_module.parse_requirements(
                {"user_request": request, "clarification_answers": []}
            )

        requirements = result["requirements"]
        self.assertTrue(result["requirements_complete"])
        self.assertEqual(requirements["business_goal"], "优先保证薄膜均匀性与共形性")
        self.assertEqual(requirements["application_scenario"], "高深宽比结构薄膜沉积")
        self.assertEqual(requirements["requested_candidates"], ["ALD", "CVD"])
        self.assertEqual(len(requirements["scenario_criteria"]), 1)
        self.assertIn("均匀性", requirements["scenario_criteria"][0]["name"])
        self.assertEqual(requirements["scenario_criteria"][0]["priority"], "highest")
        self.assertNotIn(
            "生产效率",
            [item["name"] for item in requirements["scenario_criteria"]],
        )
        self.assertEqual(
            requirements["active_criteria"],
            ["performance", "cost", "maturity"],
        )
        self.assertEqual(
            {
                criterion
                for criterion, weight in requirements["priorities"].items()
                if weight > 0
            },
            {"performance", "cost", "maturity"},
        )
        self.assertEqual(requirements["missing_information"], [])

    def test_unlabeled_ald_cvd_request_is_complete_even_when_model_returns_empty(self):
        import src.workflows.nodes.selection_requirements as requirements_module
        from src.workflows.selection_cache import clear_selection_cache

        clear_selection_cache()
        chain = MagicMock()
        chain.invoke.return_value = AIMessage(
            content=json.dumps(
                {
                    "business_goal": "",
                    "application_scenario": "",
                    "requested_candidates": [],
                    "scenario_criteria": [],
                    "missing_information": ["business_goal", "application_scenario"],
                }
            )
        )
        request = (
            "面向高深宽比结构薄膜沉积，比较ALD和CVD。"
            "要求薄膜均匀性优先，同时评估生产效率、成本和成熟度。"
        )

        with patch.object(requirements_module, "requirement_analysis_chain", chain):
            result = requirements_module.parse_requirements(
                {"user_request": request, "clarification_answers": []}
            )

        requirements = result["requirements"]
        self.assertTrue(result["requirements_complete"])
        self.assertEqual(requirements["application_scenario"], "高深宽比结构薄膜沉积")
        self.assertEqual(requirements["requested_candidates"], ["ALD", "CVD"])
        self.assertEqual(requirements["scenario_criteria"][0]["priority"], "highest")
        self.assertEqual(
            requirements["scenario_criteria"][0]["name"],
            "薄膜均匀性与共形性",
        )
        self.assertTrue(requirements["scenario_criteria"][0]["criterion_id"])

    def test_explicit_five_criterion_weights_keep_power_at_zero(self):
        import src.workflows.nodes.selection_requirements as requirements_module

        chain = MagicMock()
        chain.invoke.return_value = AIMessage(
            content=json.dumps(
                {
                    "business_goal": "低功耗量产",
                    "application_scenario": "移动芯片",
                    "priorities": {"performance": 3, "cost": 1},
                    "requested_candidates": ["FinFET", "GAA"],
                    "missing_information": [],
                }
            )
        )
        with patch.object(
            requirements_module,
            "requirement_analysis_chain",
            chain,
        ):
            result = requirements_module.parse_requirements(
                {
                    "user_request": "低功耗目标；性能权重75%，成本权重25%",
                    "clarification_answers": [],
                }
            )

        priorities = result["requirements"]["priorities"]
        self.assertEqual(priorities["performance"], 0.75)
        self.assertEqual(priorities["cost"], 0.25)
        self.assertEqual(priorities["power"], 0.0)

    def test_parse_requirements_returns_complete_normalized_state(self):
        import src.workflows.nodes.selection_requirements as requirements_module

        chain = MagicMock()
        chain.invoke.return_value = AIMessage(
            content="""{
                "business_goal": "降低功耗",
                "application_scenario": "移动芯片",
                "priorities": {"performance": 1, "cost": 1},
                "hard_constraints": ["两年内量产"],
                "requested_candidates": ["FinFET", "GAA"],
                "missing_information": []
            }"""
        )
        with patch.object(
            requirements_module,
            "requirement_analysis_chain",
            chain,
        ):
            result = requirements_module.parse_requirements(
                {
                    "user_request": (
                        "希望移动芯片降低功耗并在两年内量产，"
                        "比较FinFET和GAA"
                    ),
                    "clarification_answers": [],
                }
            )

        self.assertTrue(result["requirements_complete"])
        self.assertEqual(result["status"], "running")
        self.assertEqual(
            result["requirements"]["priorities"],
            {
                "performance": 1 / 3,
                "power": 1 / 3,
                "thermal": 0.0,
                "cost": 1 / 3,
                "maturity": 0.0,
                "risk": 0.0,
                "scalability": 0.0,
            },
        )
        self.assertEqual(
            result["requirements"]["requested_candidates"],
            ["FinFET", "GAA"],
        )

    def test_parse_requirements_strict_cache_reuses_only_identical_input(self):
        import src.workflows.nodes.selection_requirements as requirements_module
        from src.workflows.selection_cache import clear_selection_cache

        clear_selection_cache()
        chain = MagicMock()
        chain.invoke.return_value = AIMessage(
            content=json.dumps(
                {
                    "business_goal": "比较工艺",
                    "application_scenario": "高深宽比沉积",
                    "requested_candidates": ["ALD", "CVD"],
                }
            )
        )
        state = {
            "user_request": "比较ALD和CVD",
            "clarification_answers": [],
        }
        with patch.object(
            requirements_module,
            "requirement_analysis_chain",
            chain,
        ):
            first = requirements_module.parse_requirements(state)
            second = requirements_module.parse_requirements(state)
            requirements_module.parse_requirements(
                {**state, "user_request": "比较ALD和PVD"}
            )

        self.assertEqual(first, second)
        self.assertEqual(chain.invoke.call_count, 2)

    def test_clarify_requirements_returns_at_most_three_questions(self):
        from src.workflows.nodes.selection_requirements import (
            _build_clarification_questions,
        )

        result = _build_clarification_questions(
            {
                "requirements": {
                    "business_goal": "",
                    "application_scenario": "",
                    "missing_information": [
                        "business_goal",
                        "application_scenario",
                        "cost_limit",
                        "timeline",
                    ],
                },
                "excluded_candidates": [],
            }
        )

        self.assertLessEqual(len(result), 3)
        self.assertTrue(result)

    def test_generate_candidates_applies_configured_limit(self):
        import src.workflows.nodes.selection_requirements as requirements_module

        chain = MagicMock()
        chain.invoke.return_value = AIMessage(
            content="""{"candidates": [
                {"candidate_id": "a", "name": "A", "summary": "A"},
                {"candidate_id": "b", "name": "B", "summary": "B"},
                {"candidate_id": "c", "name": "C", "summary": "C"},
                {"candidate_id": "d", "name": "D", "summary": "D"}
            ]}"""
        )
        with patch.object(
            requirements_module,
            "candidate_generation_chain",
            chain,
        ):
            result = requirements_module.generate_candidates(
                {"requirements": {"business_goal": "test"}}
            )

        self.assertEqual(
            [candidate["candidate_id"] for candidate in result["candidates"]],
            ["a", "b", "c"],
        )
        self.assertTrue(result["warnings"])

    def test_generate_candidates_uses_explicit_candidates_without_an_llm_call(self):
        import src.workflows.nodes.selection_requirements as requirements_module

        chain = MagicMock()
        chain.invoke.side_effect = AssertionError(
            "explicit candidates should not require another LLM call"
        )
        with patch.object(
            requirements_module,
            "candidate_generation_chain",
            chain,
        ):
            result = requirements_module.generate_candidates(
                {
                    "requirements": {
                        "requested_candidates": ["FinFET", "GAA"],
                    }
                }
            )

        self.assertEqual(
            result["candidates"],
            [
                {
                    "candidate_id": "finfet",
                    "name": "FinFET",
                    "summary": "用户指定候选方案",
                    "feasible": True,
                    "exclusion_reasons": [],
                },
                {
                    "candidate_id": "gaa",
                    "name": "GAA",
                    "summary": "用户指定候选方案",
                    "feasible": True,
                    "exclusion_reasons": [],
                },
            ],
        )

    def test_candidate_generation_fallback_receives_the_original_request(self):
        import src.workflows.nodes.selection_requirements as requirements_module

        chain = MagicMock()
        chain.invoke.return_value = AIMessage(
            content=(
                '{"candidates": ['
                '{"candidate_id": "finfet", "name": "FinFET"}'
                "]}"
            )
        )
        with patch.object(
            requirements_module,
            "candidate_generation_chain",
            chain,
        ):
            requirements_module.generate_candidates(
                {
                    "user_request": "比较FinFET和GAA",
                    "requirements": {"requested_candidates": []},
                }
            )

        generation_input = chain.invoke.call_args.args[0]
        self.assertEqual(
            generation_input["user_request"],
            "比较FinFET和GAA",
        )

    def test_screen_hard_constraints_separates_excluded_candidates(self):
        import src.workflows.nodes.selection_requirements as requirements_module

        chain = MagicMock()
        chain.invoke.return_value = AIMessage(
            content="""{"excluded": [
                {"candidate_id": "gaa", "reasons": ["不满足一年内量产约束"]}
            ]}"""
        )
        candidates = [
            {"candidate_id": "finfet", "name": "FinFET", "summary": ""},
            {"candidate_id": "gaa", "name": "GAA", "summary": ""},
        ]
        with patch.object(
            requirements_module,
            "constraint_screening_chain",
            chain,
        ):
            result = requirements_module.screen_hard_constraints(
                {
                    "requirements": {
                        "hard_constraints": ["一年内稳定量产"],
                    },
                    "candidates": candidates,
                }
            )

        self.assertEqual(
            [candidate["candidate_id"] for candidate in result["feasible_candidates"]],
            ["finfet"],
        )
        self.assertEqual(
            result["excluded_candidates"][0]["exclusion_reasons"],
            ["不满足一年内量产约束"],
        )


class EvidenceNodeTest(unittest.TestCase):
    def test_relevant_support_counts_without_requiring_an_attributed_web_url(self):
        import src.workflows.nodes.selection_evidence as evidence_module
        from src.workflows.selection_cache import clear_selection_cache

        clear_selection_cache()
        documents = [
            Document(
                page_content=(
                    "ALD achieves uniform conformal coverage in deep trenches "
                    "and high production throughput."
                ),
                metadata={
                    "id": "local-direct",
                    "source": "knowledge.md",
                    "retrieval_sources": ["vector"],
                },
            ),
            Document(
                page_content="CVD can provide high production throughput.",
                metadata={
                    "id": "web-without-url",
                    "source": "web_search",
                    "retrieval_sources": ["web"],
                },
            ),
            Document(
                page_content="General deposition background only.",
                metadata={
                    "id": "invalid-quote",
                    "source": "knowledge.md",
                    "retrieval_sources": ["vector"],
                },
            ),
        ]
        fake_graph = MagicMock()
        fake_graph.invoke.return_value = {
            "documents": documents,
            "retrieval_source": "local_hybrid",
        }
        evidence_ids = [evidence_module._evidence_id(document) for document in documents]
        assessment_chain = MagicMock()
        assessment_chain.invoke.return_value = AIMessage(
            content=json.dumps(
                {
                    "evidence_map": [
                        {
                            "evidence_id": evidence_ids[0],
                            "supports": [
                                {
                                    "candidate_id": "ald",
                                    "criterion": "performance",
                                    "level": "direct",
                                    "quote": "uniform conformal coverage",
                                },
                                {
                                    "candidate_id": "ald",
                                    "criterion": "performance",
                                    "level": "direct",
                                    "quote": "high production throughput",
                                },
                                {
                                    "candidate_id": "ald",
                                    "criterion": "scenario_uniformity",
                                    "level": "direct",
                                    "quote": "uniform conformal coverage",
                                }
                            ],
                        },
                        {
                            "evidence_id": evidence_ids[1],
                            "supports": [
                                {
                                    "candidate_id": "cvd",
                                    "criterion": "performance",
                                    "level": "direct",
                                    "quote": "high production throughput",
                                }
                            ],
                        },
                        {
                            "evidence_id": evidence_ids[2],
                            "supports": [
                                {
                                    "candidate_id": "cvd",
                                    "criterion": "performance",
                                    "level": "direct",
                                    "quote": "not present in the document",
                                }
                            ],
                        },
                    ]
                }
            )
        )

        with patch.object(evidence_module, "retrieval_graph", fake_graph), patch.object(
            evidence_module,
            "evidence_assessment_chain",
            assessment_chain,
        ):
            result = evidence_module.retrieve_evidence(
                {
                    "requirements": {
                        "scenario_criteria": [
                            {
                                "name": "薄膜均匀性与共形性",
                                "criterion_id": "scenario_uniformity",
                                "priority": "highest",
                            }
                        ]
                    },
                    "feasible_candidates": [
                        {"candidate_id": "ald", "name": "ALD"},
                        {"candidate_id": "cvd", "name": "CVD"},
                    ],
                    "evidence_tasks": [
                        {
                            "task_id": "directness",
                            "candidate_ids": ["ald", "cvd"],
                            "query": "unique directness query",
                            "criteria": ["performance"],
                            "source_preference": "auto",
                        }
                    ],
                    "evidence_records": [],
                    "evidence_round": 0,
                }
            )

        records = {record["evidence_id"]: record for record in result["evidence_records"]}
        self.assertEqual(records[evidence_ids[0]]["candidate_ids"], ["ald"])
        self.assertEqual(
            records[evidence_ids[0]]["criteria"],
            ["performance", "scenario_uniformity"],
        )
        support_levels = {
            (support["criterion"], support["quote"]): support["level"]
            for support in records[evidence_ids[0]]["supports"]
        }
        self.assertEqual(
            support_levels[("performance", "uniform conformal coverage")],
            "indirect",
        )
        self.assertEqual(
            support_levels[("performance", "high production throughput")],
            "direct",
        )
        self.assertEqual(
            support_levels[("scenario_uniformity", "uniform conformal coverage")],
            "direct",
        )
        self.assertEqual(records[evidence_ids[1]]["source_reliability"], "web_unattributed")
        self.assertEqual(records[evidence_ids[1]]["candidate_ids"], ["cvd"])
        self.assertEqual(records[evidence_ids[2]]["candidate_ids"], [])
        self.assertFalse(any("CVD" in gap and "performance" in gap for gap in result["evidence_gaps"]))

    def test_vector_and_graph_wrappers_of_same_document_form_one_evidence_record(self):
        import src.workflows.nodes.selection_evidence as evidence_module
        from src.workflows.selection_cache import clear_selection_cache

        clear_selection_cache()
        original = "ALD has conformal step coverage above 50:1 aspect ratio."
        vector_document = Document(
            page_content=original,
            metadata={
                "id": "vector-copy",
                "source": "deposition.md",
                "retrieval_sources": ["vector"],
            },
        )
        graph_document = Document(
            page_content=f"图谱路径：ALD -> step coverage\n来源内容：{original}",
            metadata={
                "id": "graph-copy",
                "source": "deposition.md",
                "retrieval_sources": ["graph"],
            },
        )
        fake_graph = MagicMock()
        fake_graph.invoke.side_effect = [
            {"documents": [vector_document], "retrieval_source": "vector"},
            {"documents": [graph_document], "retrieval_source": "graph"},
        ]
        evidence_id = evidence_module._evidence_id(vector_document)
        assessment_chain = MagicMock()
        assessment_chain.invoke.return_value = AIMessage(
            content=json.dumps(
                {
                    "evidence_map": [
                        {
                            "evidence_id": evidence_id,
                            "supports": [
                                {
                                    "candidate_id": "ald",
                                    "criterion": "thermal",
                                    "level": "direct",
                                    "quote": "conformal step coverage above 50:1",
                                }
                            ],
                        }
                    ]
                }
            )
        )

        with patch.object(evidence_module, "retrieval_graph", fake_graph), patch.object(
            evidence_module,
            "evidence_assessment_chain",
            assessment_chain,
        ):
            result = evidence_module.retrieve_evidence(
                {
                    "requirements": {},
                    "feasible_candidates": [{"candidate_id": "ald", "name": "ALD"}],
                    "evidence_tasks": [
                        {
                            "task_id": "vector-copy",
                            "candidate_ids": ["ald"],
                            "query": "unique vector copy query",
                            "criteria": ["thermal"],
                            "source_preference": "auto",
                        },
                        {
                            "task_id": "graph-copy",
                            "candidate_ids": ["ald"],
                            "query": "unique graph copy query",
                            "criteria": ["thermal"],
                            "source_preference": "auto",
                        },
                    ],
                    "evidence_records": [],
                    "evidence_round": 0,
                }
            )

        self.assertEqual(len(result["evidence_records"]), 1)
        self.assertEqual(
            set(result["evidence_records"][0]["retrieval_sources"]),
            {"vector", "graph"},
        )

    def test_scenario_criterion_is_retrieved_separately_from_performance(self):
        import src.workflows.nodes.selection_evidence as evidence_module

        result = evidence_module.plan_evidence_tasks(
            {
                "requirements": {
                    "business_goal": "优先保证薄膜均匀性",
                    "application_scenario": "高深宽比薄膜沉积",
                    "scenario_criteria": [
                        {
                            "name": "薄膜均匀性与共形性",
                            "criterion_id": "scenario_uniformity",
                            "priority": "highest",
                        }
                    ],
                },
                "feasible_candidates": [
                    {"candidate_id": "ald", "name": "ALD"},
                    {"candidate_id": "cvd", "name": "CVD"},
                ],
            }
        )

        technical_criteria = result["evidence_tasks"][0]["criteria"]
        self.assertIn("thermal", technical_criteria)
        self.assertIn("scenario_uniformity", technical_criteria)
        self.assertIn("performance", technical_criteria)

    def test_retrieval_cache_reuses_only_successful_identical_task(self):
        import src.workflows.nodes.selection_evidence as evidence_module
        from src.workflows.selection_cache import clear_selection_cache
        from src.workflows.technology_selection_state import EvidenceTask

        clear_selection_cache()
        retrieval_graph = MagicMock()
        retrieval_graph.invoke.return_value = {
            "documents": [
                Document(
                    page_content="ALD evidence",
                    metadata={"id": "ald-evidence"},
                )
            ],
            "retrieval_source": "local_hybrid",
        }
        task = EvidenceTask(
            task_id="technical",
            candidate_ids=["ald"],
            query="ALD conformality",
            criteria=["performance"],
            source_preference="auto",
        )
        changed_task = task.model_copy(update={"query": "CVD throughput"})
        with patch.object(evidence_module, "retrieval_graph", retrieval_graph):
            first = evidence_module._retrieve_task(task)
            second = evidence_module._retrieve_task(task)
            evidence_module._retrieve_task(changed_task)

        self.assertEqual(first[1], second[1])
        self.assertEqual(retrieval_graph.invoke.call_count, 2)

    def test_retrieval_cache_does_not_store_empty_results(self):
        import src.workflows.nodes.selection_evidence as evidence_module
        from src.workflows.selection_cache import clear_selection_cache
        from src.workflows.technology_selection_state import EvidenceTask

        clear_selection_cache()
        retrieval_graph = MagicMock()
        retrieval_graph.invoke.return_value = {"documents": []}
        task = EvidenceTask(
            task_id="empty",
            candidate_ids=["ald"],
            query="missing evidence",
            criteria=["maturity"],
            source_preference="web",
        )
        with patch.object(evidence_module, "retrieval_graph", retrieval_graph):
            evidence_module._retrieve_task(task)
            evidence_module._retrieve_task(task)

        self.assertEqual(retrieval_graph.invoke.call_count, 2)

    def test_evidence_assessment_cache_invalidates_when_content_changes(self):
        import src.workflows.nodes.selection_evidence as evidence_module
        from src.workflows.selection_cache import clear_selection_cache

        clear_selection_cache()
        chain = MagicMock()
        chain.invoke.return_value = AIMessage(
            content=json.dumps(
                {
                    "evidence_map": [
                        {
                            "evidence_id": "E1",
                            "candidate_ids": ["ald"],
                            "criteria": ["thermal"],
                        }
                    ]
                }
            )
        )
        candidates = [{"candidate_id": "ald", "name": "ALD"}]
        records = [{"evidence_id": "E1", "content": "first"}]
        with patch.object(evidence_module, "evidence_assessment_chain", chain):
            first = evidence_module._assess_evidence_batch(candidates, records)
            second = evidence_module._assess_evidence_batch(candidates, records)
            evidence_module._assess_evidence_batch(
                candidates,
                [{"evidence_id": "E1", "content": "changed"}],
            )

        self.assertEqual(first, second)
        self.assertEqual(chain.invoke.call_count, 2)

    def test_evidence_assessment_retries_once_after_invalid_json(self):
        import src.workflows.nodes.selection_evidence as evidence_module
        from src.workflows.selection_cache import clear_selection_cache

        clear_selection_cache()
        chain = MagicMock()
        chain.invoke.side_effect = [
            AIMessage(content='{"evidence_map":[}'),
            AIMessage(
                content=json.dumps(
                    {
                        "evidence_map": [
                            {
                                "evidence_id": "E1",
                                "supports": [],
                            }
                        ]
                    }
                )
            ),
        ]

        with patch.object(evidence_module, "evidence_assessment_chain", chain):
            result = evidence_module._assess_evidence_batch(
                [{"candidate_id": "ald", "name": "ALD"}],
                [{"evidence_id": "E1", "content": "ALD evidence"}],
            )

        self.assertEqual(result, {"E1": {"evidence_id": "E1", "supports": []}})
        self.assertEqual(chain.invoke.call_count, 2)

    def test_one_failed_assessment_batch_does_not_discard_successful_batch(self):
        import src.workflows.nodes.selection_evidence as evidence_module
        from src.workflows.selection_cache import clear_selection_cache

        clear_selection_cache()
        documents = [
            Document(
                page_content=f"ALD performance evidence good-{index}",
                metadata={"id": f"good-{index}", "source": "knowledge.md"},
            )
            for index in range(4)
        ] + [
            Document(
                page_content=f"ALD performance evidence failed-{index}",
                metadata={"id": f"failed-{index}", "source": "knowledge.md"},
            )
            for index in range(4)
        ]
        good_evidence_ids = {
            evidence_module._evidence_id(document) for document in documents[:4]
        }
        failed_evidence_ids = {
            evidence_module._evidence_id(document) for document in documents[4:]
        }
        retrieval_graph = MagicMock()
        retrieval_graph.invoke.return_value = {
            "documents": documents,
            "retrieval_source": "local_hybrid",
        }
        assessment_chain = MagicMock()

        def assess_batch(inputs):
            cards = json.loads(inputs["evidence"])
            if any("failed-" in card["content"] for card in cards):
                return AIMessage(content='{"evidence_map":[}')
            return AIMessage(
                content=json.dumps(
                    {
                        "evidence_map": [
                            {
                                "evidence_id": card["evidence_id"],
                                "supports": [
                                    {
                                        "candidate_id": "ald",
                                        "criterion": "performance",
                                        "level": "direct",
                                        "quote": card["content"],
                                    }
                                ],
                            }
                            for card in cards
                        ]
                    }
                )
            )

        assessment_chain.invoke.side_effect = assess_batch
        with patch.object(
            evidence_module,
            "retrieval_graph",
            retrieval_graph,
        ), patch.object(
            evidence_module,
            "evidence_assessment_chain",
            assessment_chain,
        ):
            result = evidence_module.retrieve_evidence(
                {
                    "feasible_candidates": [
                        {"candidate_id": "ald", "name": "ALD"},
                    ],
                    "evidence_tasks": [
                        {
                            "task_id": "ald-evidence",
                            "candidate_ids": ["ald"],
                            "query": "ALD performance",
                            "criteria": ["performance"],
                            "source_preference": "auto",
                        }
                    ],
                    "warnings": [],
                }
            )

        records = {
            record["evidence_id"]: record
            for record in result["evidence_records"]
        }
        for evidence_id in good_evidence_ids:
            self.assertEqual(records[evidence_id]["candidate_ids"], ["ald"])
            self.assertEqual(records[evidence_id]["criteria"], ["performance"])
        for evidence_id in failed_evidence_ids:
            self.assertEqual(records[evidence_id]["candidate_ids"], [])
            self.assertEqual(records[evidence_id]["criteria"], [])
        self.assertEqual(assessment_chain.invoke.call_count, 3)
        self.assertEqual(
            sum("证据归属评估失败" in warning for warning in result["warnings"]),
            1,
        )

    def test_assessment_uses_compact_evidence_cards(self):
        import src.workflows.nodes.selection_evidence as evidence_module

        document = Document(
            page_content="x" * 1200,
            metadata={"id": "compact-card", "source": "knowledge.md"},
        )
        evidence_id = evidence_module._evidence_id(document)
        retrieval_graph = MagicMock()
        retrieval_graph.invoke.return_value = {
            "documents": [document], "retrieval_source": "local_hybrid"
        }
        assessment = MagicMock()
        assessment.invoke.return_value = AIMessage(content=json.dumps({
            "evidence_map": [{"evidence_id": evidence_id, "candidate_ids": ["gaa"], "criteria": list(evidence_module.CRITERIA)}]
        }))
        with patch.object(evidence_module, "retrieval_graph", retrieval_graph), patch.object(
            evidence_module, "evidence_assessment_chain", assessment
        ):
            evidence_module.retrieve_evidence({
                "feasible_candidates": [{"candidate_id": "gaa", "name": "GAA"}],
                "evidence_tasks": [{"task_id": "pack", "candidate_ids": ["gaa"], "query": "q", "criteria": list(evidence_module.CRITERIA), "source_preference": "auto"}],
            })

        cards = json.loads(assessment.invoke.call_args.args[0]["evidence"])
        self.assertEqual(len(cards[0]["content"]), 800)

    def test_retrieve_evidence_reuses_prior_assessment_without_model_call(self):
        import src.workflows.nodes.selection_evidence as evidence_module

        assessment = MagicMock()
        with patch.object(evidence_module, "evidence_assessment_chain", assessment):
            result = evidence_module.retrieve_evidence(
                {
                    "feasible_candidates": [{"candidate_id": "gaa", "name": "GAA"}],
                    "evidence_tasks": [],
                    "evidence_records": [
                        {
                            "evidence_id": "E1",
                            "candidate_ids": ["gaa"],
                            "criteria": list(evidence_module.CRITERIA),
                            "supports": [
                                {
                                    "candidate_id": "gaa",
                                    "criterion": criterion,
                                    "level": "direct",
                                    "quote": criterion,
                                }
                                for criterion in evidence_module.CRITERIA
                            ],
                            "content": (
                                "performance power thermal cost maturity risk "
                                "scalability already assessed"
                            ),
                        }
                    ],
                }
            )

        self.assertTrue(result["evidence_sufficient"])
        assessment.assert_not_called()

    def test_retrieve_evidence_splits_new_records_into_small_assessment_batches(self):
        import src.workflows.nodes.selection_evidence as evidence_module

        documents = [
            Document(
                page_content=(
                    "performance power thermal cost maturity risk scalability "
                    f"evidence {index}"
                ),
                metadata={"id": f"batch-{index}", "source": "knowledge.md"},
            )
            for index in range(9)
        ]
        retrieval_graph = MagicMock()
        retrieval_graph.invoke.return_value = {
            "documents": documents, "retrieval_source": "local_hybrid"
        }
        assessment = MagicMock()

        def assess(payload):
            cards = json.loads(payload["evidence"])
            return AIMessage(content=json.dumps({"evidence_map": [
                {
                    "evidence_id": card["evidence_id"],
                    "supports": [
                        {
                            "candidate_id": "gaa",
                            "criterion": criterion,
                            "level": "direct",
                            "quote": card["content"],
                        }
                        for criterion in evidence_module.CRITERIA
                    ],
                }
                for card in cards
            ]}))

        assessment.invoke.side_effect = assess
        with patch.object(evidence_module, "retrieval_graph", retrieval_graph), patch.object(
            evidence_module, "evidence_assessment_chain", assessment
        ):
            result = evidence_module.retrieve_evidence(
                {
                    "feasible_candidates": [{"candidate_id": "gaa", "name": "GAA"}],
                    "evidence_tasks": [{"task_id": "pack", "candidate_ids": ["gaa"], "query": "q", "criteria": list(evidence_module.CRITERIA), "source_preference": "auto"}],
                }
            )

        self.assertTrue(result["evidence_sufficient"])
        self.assertEqual(assessment.invoke.call_count, 3)
        self.assertEqual(
            sorted(len(json.loads(call.args[0]["evidence"])) for call in assessment.invoke.call_args_list),
            [1, 4, 4],
        )

    def test_retrieve_evidence_limits_task_concurrency_and_preserves_order(self):
        import src.workflows.nodes.selection_evidence as evidence_module

        class ControlledRetrievalGraph:
            def __init__(self):
                self.active = 0
                self.max_active = 0
                self.lock = threading.Lock()

            def invoke(self, payload):
                task_id = payload["question"]
                with self.lock:
                    self.active += 1
                    self.max_active = max(self.max_active, self.active)
                time.sleep({"task-1": 0.08, "task-2": 0.06}.get(task_id, 0.02))
                with self.lock:
                    self.active -= 1
                return {
                    "documents": [
                        Document(
                            page_content=f"evidence for {task_id}",
                            metadata={"id": task_id, "source": f"{task_id}.md"},
                        )
                    ],
                    "retrieval_source": "local_hybrid",
                }

        task_ids = ["task-1", "task-2", "task-3", "task-4"]
        fake_graph = ControlledRetrievalGraph()
        evidence_map = []
        for task_id in task_ids:
            document = Document(
                page_content=f"evidence for {task_id}",
                metadata={"id": task_id},
            )
            evidence_map.append(
                {
                    "evidence_id": evidence_module._evidence_id(document),
                    "candidate_ids": ["finfet"],
                    "criteria": ["performance"],
                }
            )
        assessment_chain = MagicMock()
        assessment_chain.invoke.return_value = AIMessage(
            content=json.dumps(
                {"evidence_map": evidence_map, "missing_queries": []}
            )
        )
        state = {
            "feasible_candidates": [
                {"candidate_id": "finfet", "name": "FinFET"},
            ],
            "evidence_tasks": [
                {
                    "task_id": task_id,
                    "candidate_ids": ["finfet"],
                    "query": task_id,
                    "criteria": ["performance"],
                    "source_preference": "auto",
                }
                for task_id in task_ids
            ],
            "evidence_records": [],
        }

        with patch.object(evidence_module, "retrieval_graph", fake_graph), patch.object(
            evidence_module,
            "evidence_assessment_chain",
            assessment_chain,
        ):
            result = evidence_module.retrieve_evidence(state)

        self.assertEqual(fake_graph.max_active, 2)
        self.assertEqual(
            [record["source"] for record in result["evidence_records"]],
            [f"{task_id}.md" for task_id in task_ids],
        )

    def test_retrieve_evidence_isolates_one_task_failure(self):
        import src.workflows.nodes.selection_evidence as evidence_module
        from src.observability.logging import log

        messages = []
        sink_id = log.add(messages.append, format="{message}")
        self.addCleanup(log.remove, sink_id)

        class PartiallyFailingGraph:
            def invoke(self, payload):
                if payload["question"] == "failed-query":
                    raise RuntimeError("provider timeout")
                return {
                    "documents": [
                        Document(
                            page_content="surviving evidence",
                            metadata={"id": "survivor", "source": "local.md"},
                        )
                    ],
                    "retrieval_source": "local_hybrid",
                }

        document = Document(
            page_content="surviving evidence",
            metadata={"id": "survivor"},
        )
        assessment_chain = MagicMock()
        assessment_chain.invoke.return_value = AIMessage(
            content=json.dumps(
                {
                    "evidence_map": [
                        {
                            "evidence_id": evidence_module._evidence_id(document),
                            "candidate_ids": ["finfet"],
                            "criteria": ["performance"],
                        }
                    ],
                    "missing_queries": [],
                }
            )
        )
        with patch.object(
            evidence_module,
            "retrieval_graph",
            PartiallyFailingGraph(),
        ), patch.object(
            evidence_module,
            "evidence_assessment_chain",
            assessment_chain,
        ):
            result = evidence_module.retrieve_evidence(
                {
                    "feasible_candidates": [
                        {"candidate_id": "finfet", "name": "FinFET"},
                    ],
                    "evidence_tasks": [
                        {
                            "task_id": "failed-task",
                            "candidate_ids": ["finfet"],
                            "query": "failed-query",
                            "criteria": ["power"],
                            "source_preference": "web",
                        },
                        {
                            "task_id": "surviving-task",
                            "candidate_ids": ["finfet"],
                            "query": "surviving-query",
                            "criteria": ["performance"],
                            "source_preference": "auto",
                        },
                    ],
                    "evidence_records": [],
                    "warnings": [],
                }
            )

        self.assertEqual(len(result["evidence_records"]), 1)
        self.assertTrue(
            any("failed-task" in warning for warning in result["warnings"])
        )
        self.assertTrue(
            any("power" in query for query in result["evidence_gaps"])
        )
        rendered = "\n".join(str(message) for message in messages)
        self.assertIn("[证据任务][failed-task] 失败：检索证据", rendered)
        self.assertIn("异常类型=RuntimeError", rendered)
        self.assertNotIn("provider timeout", rendered)
        self.assertFalse(
            any("provider timeout" in warning for warning in result["warnings"])
        )

    def test_plan_evidence_tasks_uses_deterministic_pack_sources(self):
        import src.workflows.nodes.selection_evidence as evidence_module

        source_chain = MagicMock()
        source_chain.invoke.return_value = AIMessage(
            content=json.dumps(
                {
                    "task_sources": [
                        {
                            "task_id": "criterion-performance",
                            "source": "local",
                        },
                        {
                            "task_id": "criterion-power",
                            "source": "local",
                        },
                        {
                            "task_id": "criterion-cost",
                            "source": "web",
                        },
                        {
                            "task_id": "criterion-maturity",
                            "source": "web",
                        },
                        {
                            "task_id": "criterion-risk",
                            "source": "web",
                        },
                        {
                            "task_id": "criterion-scalability",
                            "source": "local",
                        },
                    ]
                }
            )
        )
        with patch.object(
            evidence_module,
            "evidence_source_planning_chain",
            source_chain,
            create=True,
        ):
            result = evidence_module.plan_evidence_tasks(
                {
                    "user_request": (
                        "比较FinFET和GAA的性能、成本、成熟度、"
                        "供应链风险和扩展性"
                    ),
                    "requirements": {
                        "business_goal": "降低功耗",
                        "application_scenario": "高端手机SoC",
                    },
                    "feasible_candidates": [
                        {"candidate_id": "finfet", "name": "FinFET"},
                        {"candidate_id": "gaa", "name": "GAA"},
                    ],
                }
            )

        self.assertEqual(
            [(task["criteria"], task["source_preference"]) for task in result["evidence_tasks"]],
            [
                (["performance", "power", "thermal", "scalability"], "local"),
                (["cost", "maturity", "risk"], "web"),
            ],
        )
        self.assertEqual(source_chain.invoke.call_count, 0)

    def test_plan_evidence_tasks_does_not_depend_on_source_planning_model(self):
        import src.workflows.nodes.selection_evidence as evidence_module

        source_chain = MagicMock()
        source_chain.invoke.return_value = AIMessage(content="not-json")
        with patch.object(
            evidence_module,
            "evidence_source_planning_chain",
            source_chain,
            create=True,
        ):
            result = evidence_module.plan_evidence_tasks(
                {
                    "user_request": "比较FinFET和GAA的供应链风险",
                    "requirements": {
                        "business_goal": "降低风险",
                        "application_scenario": "高端手机SoC",
                    },
                    "feasible_candidates": [
                        {"candidate_id": "finfet", "name": "FinFET"},
                        {"candidate_id": "gaa", "name": "GAA"},
                    ],
                }
            )

        self.assertEqual(
            [task["source_preference"] for task in result["evidence_tasks"]],
            ["local", "web"],
        )
        self.assertEqual(source_chain.invoke.call_count, 0)

    def test_plan_evidence_tasks_routes_current_supply_chain_risk_to_web(self):
        from src.workflows.nodes.selection_evidence import plan_evidence_tasks

        result = plan_evidence_tasks(
            {
                "user_request": "面向汽车功率器件比较SiC和GaN当前供应链风险",
                "requirements": {
                    "business_goal": "降低供应链中断风险",
                    "application_scenario": "汽车功率器件",
                    "active_criteria": ["performance", "risk"],
                    "scenario_criteria": [
                        {
                            "criterion_id": "scenario_supplier_availability",
                            "name": "供应商可用性",
                            "description": "当前供应商覆盖和交付能力",
                        }
                    ],
                },
                "feasible_candidates": [
                    {"candidate_id": "sic", "name": "SiC"},
                    {"candidate_id": "gan", "name": "GaN"},
                ],
            }
        )

        self.assertEqual(
            [
                (task["criteria"], task["source_preference"])
                for task in result["evidence_tasks"]
            ],
            [
                (["performance"], "local"),
                (["risk", "scenario_supplier_availability"], "web"),
            ],
        )

    def test_plan_evidence_tasks_keeps_technical_risk_local_for_cost_goal(self):
        from src.workflows.nodes.selection_evidence import plan_evidence_tasks

        result = plan_evidence_tasks(
            {
                "user_request": "面向手机SoC比较FinFET和GAA，成本优先并评估技术集成风险",
                "requirements": {
                    "business_goal": "降低成本并控制技术集成风险",
                    "application_scenario": "手机SoC",
                    "active_criteria": ["cost", "risk"],
                    "scenario_criteria": [],
                },
                "feasible_candidates": [
                    {"candidate_id": "finfet", "name": "FinFET"},
                    {"candidate_id": "gaa", "name": "GAA"},
                ],
            }
        )

        self.assertEqual(
            [
                (task["criteria"], task["source_preference"])
                for task in result["evidence_tasks"]
            ],
            [
                (["risk"], "local"),
                (["cost"], "web"),
            ],
        )

    def test_initial_deposition_tasks_use_search_labels_not_internal_ids(self):
        from src.workflows.nodes.selection_evidence import plan_evidence_tasks

        result = plan_evidence_tasks(
            {
                "user_request": "面向高深宽比薄膜沉积比较ALD和CVD",
                "requirements": {
                    "business_goal": "比较生产效率、成本和成熟度",
                    "application_scenario": "高深宽比薄膜沉积",
                    "active_criteria": ["performance", "cost", "maturity"],
                    "scenario_criteria": [],
                },
                "feasible_candidates": [
                    {"candidate_id": "ald", "name": "ALD"},
                    {"candidate_id": "cvd", "name": "CVD"},
                ],
            }
        )

        manufacturing_query = result["evidence_tasks"][1]["query"]
        self.assertIn("cost of ownership", manufacturing_query)
        self.assertIn("high volume manufacturing", manufacturing_query)
        self.assertNotIn("cost,maturity", manufacturing_query)

    def test_plan_evidence_tasks_creates_one_cross_candidate_task_per_criterion(self):
        import src.workflows.nodes.selection_evidence as evidence_module

        source_chain = MagicMock()
        source_chain.invoke.return_value = AIMessage(
            content='{"task_sources":[]}'
        )
        with patch.object(
            evidence_module,
            "evidence_source_planning_chain",
            source_chain,
        ):
            result = evidence_module.plan_evidence_tasks(
                {
                    "user_request": "比较适合移动芯片的器件架构",
                    "requirements": {
                        "business_goal": "降低功耗",
                        "application_scenario": "移动芯片",
                    },
                    "feasible_candidates": [
                        {"candidate_id": "finfet", "name": "FinFET"},
                        {"candidate_id": "gaa", "name": "GAA"},
                    ],
                }
            )

        self.assertEqual(len(result["evidence_tasks"]), 2)
        self.assertEqual(
            [task["task_id"] for task in result["evidence_tasks"]],
            ["initial-technical", "initial-manufacturing"],
        )
        self.assertEqual(
            result["evidence_tasks"][0]["candidate_ids"],
            ["finfet", "gaa"],
        )
        self.assertEqual(
            result["evidence_tasks"][0]["criteria"],
            ["performance", "power", "thermal", "scalability", "risk"],
        )
        self.assertEqual(result["evidence_tasks"][1]["criteria"], ["cost", "maturity"])

    def test_plan_evidence_tasks_only_queries_active_fixed_criteria(self):
        import src.workflows.nodes.selection_evidence as evidence_module

        result = evidence_module.plan_evidence_tasks(
            {
                "requirements": {
                    "business_goal": "优先保证薄膜均匀性",
                    "application_scenario": "高深宽比薄膜沉积",
                    "active_criteria": ["performance", "cost", "maturity"],
                    "scenario_criteria": [
                        {
                            "name": "薄膜均匀性与共形性",
                            "criterion_id": "scenario_uniformity",
                            "priority": "highest",
                        }
                    ],
                },
                "feasible_candidates": [
                    {"candidate_id": "ald", "name": "ALD"},
                    {"candidate_id": "cvd", "name": "CVD"},
                ],
            }
        )

        self.assertEqual(
            [task["criteria"] for task in result["evidence_tasks"]],
            [
                ["performance", "scenario_uniformity"],
                ["cost", "maturity"],
            ],
        )
        queried = {
            criterion
            for task in result["evidence_tasks"]
            for criterion in task["criteria"]
        }
        self.assertTrue(
            {"power", "thermal", "risk", "scalability"}.isdisjoint(queried)
        )

    def test_retrieve_evidence_deduplicates_documents_and_tracks_sources(self):
        import src.workflows.nodes.selection_evidence as evidence_module

        fake_graph = MagicMock()
        retrieval_results = [
            {
                "documents": [
                    Document(
                        page_content=(
                            "performance power thermal cost maturity risk "
                            "scalability shared evidence"
                        ),
                        metadata={
                            "id": "doc-1",
                            "source": "knowledge.md",
                            "retrieval_sources": ["vector"],
                        },
                    )
                ],
                "retrieval_source": "local_hybrid",
            },
            {
                "documents": [
                    Document(
                        page_content=(
                            "performance power thermal cost maturity risk "
                            "scalability shared evidence"
                        ),
                        metadata={
                            "id": "doc-1",
                            "source": "knowledge.md",
                            "retrieval_sources": ["vector"],
                        },
                    )
                ],
                "retrieval_source": "local_hybrid",
            },
        ]
        fake_graph.invoke.side_effect = retrieval_results
        evidence_id = evidence_module._evidence_id(
            retrieval_results[0]["documents"][0]
        )
        assessment_chain = MagicMock()
        assessment_chain.invoke.return_value = AIMessage(
            content=json.dumps(
                {
                    "evidence_map": [
                        {
                            "evidence_id": evidence_id,
                            "supports": [
                                {
                                    "candidate_id": candidate_id,
                                    "criterion": criterion,
                                    "level": "direct",
                                    "quote": criterion,
                                }
                                for candidate_id in ("finfet", "gaa")
                                for criterion in evidence_module.CRITERIA
                            ],
                        }
                    ],
                    "missing_queries": [],
                }
            )
        )
        state = {
            "feasible_candidates": [
                {"candidate_id": "finfet", "name": "FinFET"},
                {"candidate_id": "gaa", "name": "GAA"},
            ],
            "evidence_tasks": [
                {
                    "task_id": "task-finfet",
                    "candidate_ids": ["finfet"],
                    "query": "FinFET evidence",
                    "criteria": ["performance"],
                    "source_preference": "auto",
                },
                {
                    "task_id": "task-gaa",
                    "candidate_ids": ["gaa"],
                    "query": "GAA evidence",
                    "criteria": ["cost"],
                    "source_preference": "auto",
                },
            ],
            "evidence_records": [],
            "evidence_round": 0,
        }
        with patch.object(
            evidence_module,
            "retrieval_graph",
            fake_graph,
        ), patch.object(
            evidence_module,
            "evidence_assessment_chain",
            assessment_chain,
        ):
            result = evidence_module.retrieve_evidence(state)

        self.assertEqual(len(result["evidence_records"]), 1)
        self.assertEqual(
            result["evidence_records"][0]["candidate_ids"],
            ["finfet", "gaa"],
        )
        self.assertEqual(
            result["evidence_records"][0]["source"],
            "knowledge.md",
        )
        self.assertTrue(result["evidence_sufficient"])
        self.assertEqual(result["evidence_round"], 1)

    def test_retrieve_evidence_ignores_error_documents_and_reports_gap(self):
        import src.workflows.nodes.selection_evidence as evidence_module

        fake_graph = MagicMock()
        fake_graph.invoke.return_value = {
            "documents": [
                Document(
                    page_content="search unavailable",
                    metadata={"is_error": True},
                )
            ],
            "retrieval_source": "web_search",
        }
        with patch.object(evidence_module, "retrieval_graph", fake_graph):
            result = evidence_module.retrieve_evidence(
                {
                    "feasible_candidates": [
                        {"candidate_id": "gaa", "name": "GAA"},
                    ],
                    "evidence_tasks": [
                        {
                            "task_id": "task-gaa",
                            "candidate_ids": ["gaa"],
                            "query": "GAA evidence",
                            "criteria": ["maturity"],
                            "source_preference": "web",
                        }
                    ],
                    "evidence_records": [],
                    "evidence_round": 0,
                }
            )

        self.assertFalse(result["evidence_sufficient"])
        self.assertEqual(result["evidence_records"], [])
        self.assertIn("GAA", result["missing_evidence_queries"][0])

    def test_web_supplement_failure_is_reported_as_a_controlled_warning(self):
        import src.workflows.nodes.selection_evidence as evidence_module
        from src.workflows.selection_cache import clear_selection_cache

        clear_selection_cache()
        fake_graph = MagicMock()
        fake_graph.invoke.return_value = {
            "documents": [],
            "retrieval_source": "web_search",
            "retrieval_error": "web_search_unavailable",
        }
        with patch.object(evidence_module, "retrieval_graph", fake_graph):
            result = evidence_module.retrieve_evidence(
                {
                    "requirements": {
                        "priorities": {"maturity": 1},
                        "hard_constraints": [],
                        "scenario_criteria": [],
                    },
                    "feasible_candidates": [
                        {"candidate_id": "gaa", "name": "GAA"},
                    ],
                    "evidence_tasks": [
                        {
                            "task_id": "web-gaa-maturity",
                            "candidate_ids": ["gaa"],
                            "query": "unique web GAA maturity evidence",
                            "criteria": ["maturity"],
                            "source_preference": "web",
                        }
                    ],
                    "evidence_records": [],
                    "evidence_round": 2,
                    "warnings": [],
                }
            )

        self.assertTrue(
            any(
                "Web补检任务 web-gaa-maturity" in warning
                and "web_search_unavailable" in warning
                for warning in result["warnings"]
            )
        )

    def test_evidence_progress_reports_web_url_and_relevant_support_counts(self):
        import src.workflows.nodes.selection_evidence as evidence_module
        from src.workflows.selection_cache import clear_selection_cache

        class StageRecorder:
            def __init__(self):
                self.results = []

            def __call__(self, *args, **kwargs):
                return self

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                return False

            def set_result(self, result):
                self.results.append(result)

        clear_selection_cache()
        document = Document(
            page_content="GAA maturity is supported by commercial production.",
            metadata={
                "id": "web-maturity-counts",
                "title": "GAA production",
                "source": "https://example.com/gaa",
                "url": "https://example.com/gaa",
                "retrieval_source": "web_search",
            },
        )
        evidence_id = evidence_module._evidence_id(document)
        fake_graph = MagicMock()
        fake_graph.invoke.return_value = {
            "documents": [document],
            "retrieval_source": "web_search",
            "retrieval_error": None,
        }
        assessment_chain = MagicMock()
        assessment_chain.invoke.return_value = AIMessage(
            content=json.dumps(
                {
                    "evidence_map": [
                        {
                            "evidence_id": evidence_id,
                            "supports": [
                                {
                                    "candidate_id": "gaa",
                                    "criterion": "maturity",
                                    "level": "direct",
                                    "quote": "commercial production",
                                }
                            ],
                        }
                    ]
                }
            )
        )
        recorder = StageRecorder()
        with patch.object(evidence_module, "retrieval_graph", fake_graph), patch.object(
            evidence_module,
            "evidence_assessment_chain",
            assessment_chain,
        ), patch.object(evidence_module, "progress_stage", recorder):
            evidence_module.retrieve_evidence(
                {
                    "requirements": {
                        "priorities": {"maturity": 1},
                        "hard_constraints": [],
                        "scenario_criteria": [],
                    },
                    "feasible_candidates": [
                        {"candidate_id": "gaa", "name": "GAA"},
                    ],
                    "evidence_tasks": [
                        {
                            "task_id": "web-gaa-counts",
                            "candidate_ids": ["gaa"],
                            "query": "unique web GAA maturity counts",
                            "criteria": ["maturity"],
                            "source_preference": "web",
                        }
                    ],
                    "evidence_records": [],
                    "evidence_round": 2,
                }
            )

        self.assertTrue(
            any(
                result.get("Web证据") == 1
                and result.get("Web有效URL") == 1
                and result.get("相关支持") == 1
                for result in recorder.results
            )
        )

    def test_retrieve_evidence_keeps_records_when_assessment_fails(self):
        import src.workflows.nodes.selection_evidence as evidence_module

        document = Document(
            page_content="GAA evidence",
            metadata={"id": "e1", "source": "knowledge.md"},
        )
        fake_graph = MagicMock()
        fake_graph.invoke.return_value = {
            "documents": [document],
            "retrieval_source": "local_hybrid",
        }
        failing_assessment = MagicMock()
        failing_assessment.invoke.side_effect = ValueError("bad assessment")
        state = {
            "feasible_candidates": [{"candidate_id": "gaa", "name": "GAA"}],
            "evidence_tasks": [{
                "task_id": "task-gaa",
                "candidate_ids": ["gaa"],
                "query": "GAA performance",
                "criteria": ["performance"],
                "source_preference": "auto",
            }],
            "evidence_records": [],
            "evidence_round": 0,
        }
        with patch.object(evidence_module, "retrieval_graph", fake_graph):
            with patch.object(
                evidence_module,
                "evidence_assessment_chain",
                failing_assessment,
            ):
                result = evidence_module.retrieve_evidence(state)

        self.assertEqual(len(result["evidence_records"]), 1)
        self.assertFalse(result["evidence_sufficient"])
        self.assertTrue(result["warnings"])
        self.assertEqual(len(result["missing_evidence_queries"]), 7)

    def test_retrieve_evidence_allows_one_local_query_rewrite(self):
        import src.workflows.nodes.selection_evidence as evidence_module

        fake_graph = MagicMock()
        fake_graph.invoke.return_value = {
            "documents": [],
            "retrieval_source": "local_hybrid",
        }
        with patch.object(
            evidence_module,
            "retrieval_graph",
            fake_graph,
        ), patch.object(evidence_module, "ThreadPoolExecutor") as executor:
            evidence_module.retrieve_evidence(
                {
                    "feasible_candidates": [
                        {"candidate_id": "gaa", "name": "GAA"},
                    ],
                    "evidence_tasks": [
                        {
                            "task_id": "task-gaa",
                            "candidate_ids": ["gaa"],
                            "query": "GAA evidence",
                            "criteria": ["maturity"],
                            "source_preference": "auto",
                        }
                    ],
                    "evidence_records": [],
                    "evidence_round": 0,
                }
            )

        retrieval_input = fake_graph.invoke.call_args.args[0]
        self.assertEqual(retrieval_input["max_transform_count"], 1)
        self.assertFalse(retrieval_input["allow_web_fallback"])
        executor.assert_not_called()

    def test_web_evidence_uses_selection_budget_without_requiring_a_url(self):
        import src.workflows.nodes.selection_evidence as evidence_module
        from src.workflows.selection_cache import clear_selection_cache

        clear_selection_cache()
        fake_graph = MagicMock()
        fake_graph.invoke.return_value = {
            "documents": [],
            "retrieval_source": "web_search",
            "retrieval_error": "web_search_no_usable_results",
        }
        with patch.object(evidence_module, "retrieval_graph", fake_graph):
            evidence_module.retrieve_evidence(
                {
                    "requirements": {"priorities": {"maturity": 1}},
                    "feasible_candidates": [
                        {"candidate_id": "sic", "name": "SiC"},
                    ],
                    "evidence_tasks": [
                        {
                            "task_id": "web-sic-maturity-budget",
                            "candidate_ids": ["sic"],
                            "query": "unique SiC maturity Web query",
                            "criteria": ["maturity"],
                            "source_preference": "web",
                        }
                    ],
                    "evidence_records": [],
                    "evidence_round": 1,
                }
            )

        retrieval_input = fake_graph.invoke.call_args.args[0]
        self.assertEqual(retrieval_input["web_search_max_results"], 8)
        self.assertEqual(retrieval_input["web_search_engine"], "search_pro")
        self.assertFalse(retrieval_input["require_source_url"])

    def test_scenario_metric_synonyms_can_form_direct_web_support(self):
        import src.workflows.nodes.selection_evidence as evidence_module

        cases = [
            (
                "scenario_voltage",
                "耐压能力",
                "该器件的击穿电压达到1200V。",
                "击穿电压达到1200V",
            ),
            (
                "scenario_production",
                "量产成熟度",
                "该产品已进入大规模量产阶段。",
                "大规模量产",
            ),
            (
                "scenario_supply_chain",
                "供应链风险",
                "供应商集中度较高，可能造成交付风险。",
                "供应商集中度较高",
            ),
        ]
        criteria_catalog = [
            {"criterion_id": criterion_id, "name": name}
            for criterion_id, name, _, _ in cases
        ]

        for criterion_id, _, content, quote in cases:
            with self.subTest(criterion_id=criterion_id):
                supports = evidence_module._validated_supports(
                    {
                        "content": content,
                        "source_reliability": "web_identified",
                    },
                    {
                        "supports": [
                            {
                                "candidate_id": "sic",
                                "criterion": criterion_id,
                                "level": "direct",
                                "quote": quote,
                            }
                        ]
                    },
                    {"sic"},
                    {item["criterion_id"] for item in criteria_catalog},
                    criteria_catalog,
                )

                self.assertEqual(supports[0]["level"], "direct")

    def test_evidence_is_insufficient_when_some_criteria_are_not_covered(self):
        import src.workflows.nodes.selection_evidence as evidence_module

        document = Document(
            page_content="Only performance evidence",
            metadata={"id": "performance-only", "source": "knowledge.md"},
        )
        evidence_id = evidence_module._evidence_id(document)
        fake_graph = MagicMock()
        fake_graph.invoke.return_value = {
            "documents": [document],
            "retrieval_source": "local_hybrid",
        }
        assessment_chain = MagicMock()
        assessment_chain.invoke.return_value = AIMessage(
            content=json.dumps(
                {
                    "evidence_map": [
                        {
                            "evidence_id": evidence_id,
                            "supports": [
                                {
                                    "candidate_id": "gaa",
                                    "criterion": "performance",
                                    "level": "direct",
                                    "quote": "performance evidence",
                                }
                            ],
                        }
                    ],
                    "missing_queries": ["补充GAA成本和成熟度证据"],
                }
            )
        )
        with patch.object(
            evidence_module,
            "retrieval_graph",
            fake_graph,
        ), patch.object(
            evidence_module,
            "evidence_assessment_chain",
            assessment_chain,
        ):
            result = evidence_module.retrieve_evidence(
                {
                    "feasible_candidates": [
                        {"candidate_id": "gaa", "name": "GAA"},
                    ],
                    "evidence_tasks": [
                        {
                            "task_id": "task-gaa",
                            "candidate_ids": ["gaa"],
                            "query": "GAA evidence",
                            "criteria": [
                                "performance",
                                "cost",
                                "maturity",
                                "risk",
                                "scalability",
                            ],
                            "source_preference": "auto",
                        }
                    ],
                    "evidence_records": [],
                    "evidence_round": 0,
                }
            )

        self.assertFalse(result["evidence_sufficient"])
        self.assertEqual(
            result["evidence_records"][0]["criteria"],
            ["performance"],
        )
        for criterion in ("power", "thermal", "cost", "maturity", "risk", "scalability"):
            self.assertIn(
                f"补充GAA在{criterion}方面的可引用证据。",
                result["missing_evidence_queries"],
            )

    def test_zero_weight_fixed_criteria_do_not_create_supplement_gaps(self):
        import src.workflows.nodes.selection_evidence as evidence_module

        result = evidence_module.retrieve_evidence(
            {
                "requirements": {
                    "priorities": {
                        "performance": 1,
                        "power": 0,
                        "thermal": 0,
                        "cost": 0,
                        "maturity": 0,
                        "risk": 0,
                        "scalability": 0,
                    },
                    "hard_constraints": [],
                    "scenario_criteria": [],
                },
                "feasible_candidates": [
                    {"candidate_id": "gaa", "name": "GAA"},
                ],
                "evidence_tasks": [],
                "evidence_records": [
                    {
                        "evidence_id": "E-performance",
                        "candidate_ids": ["gaa"],
                        "criteria": ["performance"],
                        "supports": [
                            {
                                "candidate_id": "gaa",
                                "criterion": "performance",
                                "level": "direct",
                                "quote": "performance",
                            }
                        ],
                        "content": "GAA performance evidence",
                    }
                ],
                "evidence_round": 0,
            }
        )

        self.assertTrue(result["evidence_sufficient"])
        self.assertEqual(result["missing_evidence_queries"], [])

    def test_hard_constraint_criterion_remains_a_gap_with_zero_weight(self):
        import src.workflows.nodes.selection_evidence as evidence_module

        result = evidence_module.retrieve_evidence(
            {
                "requirements": {
                    "priorities": {
                        "performance": 1,
                        "power": 0,
                        "thermal": 0,
                        "cost": 0,
                        "maturity": 0,
                        "risk": 0,
                        "scalability": 0,
                    },
                    "hard_constraints": ["功耗不得超过既定预算"],
                    "scenario_criteria": [],
                },
                "feasible_candidates": [
                    {"candidate_id": "gaa", "name": "GAA"},
                ],
                "evidence_tasks": [],
                "evidence_records": [
                    {
                        "evidence_id": "E-performance",
                        "candidate_ids": ["gaa"],
                        "criteria": ["performance"],
                        "supports": [
                            {
                                "candidate_id": "gaa",
                                "criterion": "performance",
                                "level": "direct",
                                "quote": "performance",
                            }
                        ],
                        "content": "GAA performance evidence",
                    }
                ],
                "evidence_round": 0,
            }
        )

        self.assertEqual(len(result["missing_evidence_queries"]), 1)
        self.assertIn("power", result["missing_evidence_queries"][0])

    def test_first_supplement_retries_verified_gaps_locally(self):
        import src.workflows.nodes.selection_evidence as evidence_module

        source_chain = MagicMock()
        source_chain.invoke.return_value = AIMessage(
            content=json.dumps(
                {
                    "task_sources": [
                        {
                            "task_id": "retry-1-1",
                            "source_preference": "web",
                        }
                    ]
                }
            )
        )
        with patch.object(
            evidence_module,
            "evidence_source_planning_chain",
            source_chain,
            create=True,
        ):
            result = evidence_module.optimize_evidence_queries(
                {
                    "user_request": "比较GAA和FinFET",
                    "feasible_candidates": [
                        {"candidate_id": "gaa", "name": "GAA"},
                    ],
                    "missing_evidence_queries": ["补充GAA在power方面的可引用证据。"],
                    "evidence_round": 1,
                }
            )

        self.assertEqual(len(result["evidence_tasks"]), 1)
        self.assertIn("功耗", result["evidence_tasks"][0]["query"])
        self.assertEqual(result["evidence_tasks"][0]["criteria"], ["power"])
        self.assertEqual(
            result["evidence_tasks"][0]["source_preference"],
            "local",
        )
        self.assertEqual(source_chain.invoke.call_count, 0)

    def test_first_supplement_keeps_cost_and_maturity_on_web(self):
        from src.workflows.nodes.selection_evidence import optimize_evidence_queries

        result = optimize_evidence_queries(
            {
                "user_request": "面向高深宽比薄膜沉积比较ALD和CVD",
                "requirements": {
                    "application_scenario": "高深宽比薄膜沉积",
                    "scenario_criteria": [],
                },
                "feasible_candidates": [
                    {"candidate_id": "ald", "name": "ALD"},
                    {"candidate_id": "cvd", "name": "CVD"},
                ],
                "missing_evidence_queries": [
                    "补充ALD在cost方面的可引用证据。",
                    "补充CVD在maturity方面的可引用证据。",
                ],
                "evidence_round": 1,
            }
        )

        self.assertEqual(
            [
                (task["candidate_ids"], task["criteria"], task["source_preference"])
                for task in result["evidence_tasks"]
            ],
            [
                (["ald"], ["cost"], "web"),
                (["cvd"], ["maturity"], "web"),
            ],
        )

    def test_deposition_supplement_uses_process_cost_and_maturity_terms(self):
        from src.workflows.nodes.selection_evidence import optimize_evidence_queries

        result = optimize_evidence_queries(
            {
                "requirements": {
                    "application_scenario": "高深宽比薄膜沉积",
                    "scenario_criteria": [],
                },
                "feasible_candidates": [
                    {"candidate_id": "ald", "name": "ALD"},
                    {"candidate_id": "cvd", "name": "CVD"},
                ],
                "missing_evidence_queries": [
                    "补充ALD在cost方面的可引用证据。",
                    "补充CVD在maturity方面的可引用证据。",
                ],
                "evidence_round": 1,
            }
        )

        cost_query = result["evidence_tasks"][0]["query"]
        maturity_query = result["evidence_tasks"][1]["query"]
        self.assertIn("precursor consumption", cost_query)
        self.assertIn("wafer throughput", cost_query)
        self.assertNotIn("device substrate", cost_query)
        self.assertIn("fab adoption", maturity_query)
        self.assertIn("production-proven", maturity_query)
        self.assertNotIn("automotive qualification", maturity_query)

    def test_supplement_query_uses_candidate_and_scenario_names_instead_of_internal_ids(self):
        from src.workflows.nodes.selection_evidence import optimize_evidence_queries

        result = optimize_evidence_queries(
            {
                "user_request": "面向新能源汽车功率器件比较SiC和GaN",
                "requirements": {
                    "scenario_criteria": [
                        {
                            "criterion_id": "scenario_voltage",
                            "name": "耐压能力",
                            "description": "器件可承受的工作电压和击穿电压",
                            "priority": "highest",
                        }
                    ]
                },
                "feasible_candidates": [
                    {"candidate_id": "sic", "name": "SiC"},
                ],
                "missing_evidence_queries": [
                    "补充SiC在power方面的可引用证据。",
                    "补充SiC在scenario_voltage方面的可引用证据。",
                ],
                "evidence_round": 1,
            }
        )

        self.assertEqual(len(result["evidence_tasks"]), 1)
        task = result["evidence_tasks"][0]
        self.assertEqual(task["source_preference"], "local")
        self.assertEqual(task["candidate_ids"], ["sic"])
        self.assertIn("SiC", task["query"])
        self.assertIn("功耗", task["query"])
        self.assertIn("耐压能力", task["query"])
        self.assertNotIn("scenario_voltage", task["query"])

    def test_web_query_disambiguates_common_semiconductor_candidate_abbreviations(self):
        from src.workflows.nodes.selection_evidence import optimize_evidence_queries

        result = optimize_evidence_queries(
            {
                "user_request": "面向新能源汽车功率器件比较SiC和GaN",
                "requirements": {
                    "application_scenario": "新能源汽车功率器件",
                    "scenario_criteria": [],
                },
                "feasible_candidates": [
                    {"candidate_id": "sic", "name": "SiC"},
                ],
                "missing_evidence_queries": [
                    "补充SiC在thermal方面的可引用证据。",
                ],
                "evidence_round": 2,
            }
        )

        query = result["evidence_tasks"][0]["query"]
        self.assertIn("silicon carbide", query)
        self.assertIn("thermal conductivity", query)
        self.assertNotIn("原始选型需求", query)

    def test_full_candidate_name_can_attribute_direct_web_evidence(self):
        import src.workflows.nodes.selection_evidence as evidence_module

        record = {
            "source_reliability": "web_identified",
            "title": "Silicon Carbide power devices",
            "content": (
                "Silicon carbide power devices support a 200 C junction "
                "temperature for automotive applications."
            ),
        }
        supports = evidence_module._deterministic_web_supports(
            record,
            ["sic"],
            ["thermal"],
            {"sic": "SiC"},
            [
                {"criterion_id": criterion, "name": criterion}
                for criterion in evidence_module.CRITERIA
            ],
        )

        self.assertEqual(len(supports), 1)
        self.assertEqual(supports[0]["candidate_id"], "sic")
        self.assertEqual(supports[0]["criterion"], "thermal")

    def test_second_web_supplement_creates_one_task_per_candidate_metric_gap(self):
        from src.workflows.nodes.selection_evidence import optimize_evidence_queries

        result = optimize_evidence_queries(
            {
                "user_request": "面向新能源汽车功率器件比较SiC和GaN",
                "requirements": {
                    "scenario_criteria": [
                        {
                            "criterion_id": "scenario_voltage",
                            "name": "耐压能力",
                            "description": "器件可承受的工作电压和击穿电压",
                            "priority": "highest",
                        }
                    ]
                },
                "feasible_candidates": [
                    {"candidate_id": "sic", "name": "SiC"},
                ],
                "missing_evidence_queries": [
                    "补充SiC在power方面的可引用证据。",
                    "补充SiC在cost方面的可引用证据。",
                    "补充SiC在scenario_voltage方面的可引用证据。",
                ],
                "evidence_round": 2,
            }
        )

        self.assertEqual(len(result["evidence_tasks"]), 3)
        criteria_groups = {
            tuple(task["criteria"])
            for task in result["evidence_tasks"]
        }
        self.assertEqual(
            criteria_groups,
            {("power",), ("cost",), ("scenario_voltage",)},
        )
        self.assertTrue(
            all(task["source_preference"] == "web" for task in result["evidence_tasks"])
        )

    def test_attributed_web_text_can_fill_direct_support_missed_by_the_model(self):
        import src.workflows.nodes.selection_evidence as evidence_module
        from src.workflows.selection_cache import clear_selection_cache

        clear_selection_cache()
        document = Document(
            page_content=(
                "SiC器件的击穿场强高，可支持更高额定电压。"
                "SiC器件的开关损耗和导通损耗更低。"
                "SiC器件具有更高热导率，可在高结温下工作。"
            ),
            metadata={
                "id": "sic-direct-web-fallback",
                "title": "SiC automotive power device data",
                "source": "https://example.com/sic-data",
                "url": "https://example.com/sic-data",
                "retrieval_source": "web_search",
            },
        )
        fake_graph = MagicMock()
        fake_graph.invoke.return_value = {
            "documents": [document],
            "retrieval_source": "web_search",
            "retrieval_error": None,
        }
        assessment_chain = MagicMock()
        assessment_chain.invoke.return_value = AIMessage(
            content=json.dumps({"evidence_map": []})
        )

        with patch.object(evidence_module, "retrieval_graph", fake_graph), patch.object(
            evidence_module,
            "evidence_assessment_chain",
            assessment_chain,
        ):
            result = evidence_module.retrieve_evidence(
                {
                    "requirements": {
                        "priorities": {
                            "performance": 1,
                            "power": 1,
                            "thermal": 1,
                        },
                        "hard_constraints": [],
                        "scenario_criteria": [],
                    },
                    "feasible_candidates": [
                        {"candidate_id": "sic", "name": "SiC"},
                    ],
                    "evidence_tasks": [
                        {
                            "task_id": "web-sic-direct-fallback",
                            "candidate_ids": ["sic"],
                            "query": "unique attributed SiC data query",
                            "criteria": ["performance", "power", "thermal"],
                            "source_preference": "web",
                        }
                    ],
                    "evidence_records": [],
                    "evidence_round": 2,
                }
            )

        direct_criteria = {
            support["criterion"]
            for support in result["evidence_records"][0]["supports"]
            if support["level"] == "direct"
        }
        self.assertEqual(
            direct_criteria,
            {"performance", "power", "thermal"},
        )
        self.assertTrue(result["evidence_sufficient"])

    def test_candidate_specific_web_title_can_attribute_metric_sentence_safely(self):
        import src.workflows.nodes.selection_evidence as evidence_module

        criteria_catalog = [{"criterion_id": "risk", "name": "risk"}]
        candidate_names = {"sic": "SiC", "gan": "GaN"}
        attributed = evidence_module._deterministic_web_supports(
            {
                "title": "GaN车规功率器件可靠性进展",
                "content": "动态导通电阻退化可能带来长期可靠性风险。",
                "source_reliability": "web_identified",
            },
            {"gan"},
            {"risk"},
            candidate_names,
            criteria_catalog,
        )
        ambiguous = evidence_module._deterministic_web_supports(
            {
                "title": "SiC与GaN功率器件对比",
                "content": "动态导通电阻退化可能带来长期可靠性风险。",
                "source_reliability": "web_identified",
            },
            {"gan"},
            {"risk"},
            candidate_names,
            criteria_catalog,
        )

        self.assertEqual(len(attributed), 1)
        self.assertEqual(attributed[0]["candidate_id"], "gan")
        self.assertEqual(attributed[0]["criterion"], "risk")
        self.assertEqual(ambiguous, [])

    def test_second_retry_forces_remaining_gaps_to_web(self):
        from src.workflows.nodes.selection_evidence import optimize_evidence_queries

        result = optimize_evidence_queries(
            {
                "user_request": "compare ALD and CVD",
                "feasible_candidates": [
                    {"candidate_id": "ald", "name": "ALD"},
                    {"candidate_id": "cvd", "name": "CVD"},
                ],
                "missing_evidence_queries": [
                    "补充ALD在power方面的可引用证据。",
                    "补充CVD在thermal方面的可引用证据。",
                ],
                "evidence_round": 2,
            }
        )

        self.assertEqual(len(result["evidence_tasks"]), 2)
        tasks_by_candidate = {
            task["candidate_ids"][0]: task
            for task in result["evidence_tasks"]
        }
        self.assertEqual(set(tasks_by_candidate), {"ald", "cvd"})
        self.assertEqual(tasks_by_candidate["ald"]["source_preference"], "web")
        self.assertEqual(tasks_by_candidate["ald"]["criteria"], ["power"])
        self.assertEqual(tasks_by_candidate["cvd"]["source_preference"], "web")
        self.assertEqual(tasks_by_candidate["cvd"]["criteria"], ["thermal"])


class DecisionNodeTest(unittest.TestCase):
    def test_evaluation_scores_and_gaps_only_use_active_fixed_criteria(self):
        import src.workflows.nodes.selection_decision as decision_module
        from src.workflows.selection_cache import clear_selection_cache

        clear_selection_cache()
        active = ["performance", "cost", "maturity"]
        chain = MagicMock()
        chain.invoke.return_value = AIMessage(
            content=json.dumps(
                {
                    "evaluations": [
                        {
                            "candidate_id": "ald",
                            "criterion_scores": [
                                {
                                    "criterion": criterion,
                                    "score": 4,
                                    "rationale": "relevant evidence",
                                    "evidence_ids": ["E1"],
                                    "confidence": "high",
                                }
                                for criterion in active
                            ],
                        }
                    ]
                }
            )
        )
        result = None
        with patch.object(decision_module, "candidate_evaluation_chain", chain):
            result = decision_module.evaluate_candidates(
                {
                    "requirements": {
                        "active_criteria": active,
                        "priorities": {
                            "performance": 1,
                            "cost": 1,
                            "maturity": 1,
                        },
                    },
                    "feasible_candidates": [
                        {"candidate_id": "ald", "name": "ALD"}
                    ],
                    "evidence_records": [
                        {
                            "evidence_id": "E1",
                            "supports": [
                                {
                                    "candidate_id": "ald",
                                    "criterion": criterion,
                                    "level": "direct",
                                    "quote": criterion,
                                }
                                for criterion in active
                            ],
                            "content": "performance cost maturity",
                        }
                    ],
                    "evidence_gaps": [],
                }
            )

        evaluation = result["evaluations"][0]
        scores = {
            item["criterion"]: item
            for item in evaluation["criterion_scores"]
        }
        self.assertEqual(evaluation["weighted_score"], 4.0)
        self.assertEqual(evaluation["evidence_coverage"], 1.0)
        self.assertEqual(scores["power"]["rationale"], "本次未纳入评估")
        self.assertIsNone(scores["power"]["score"])
        self.assertFalse(
            any(
                criterion in gap
                for gap in result["evidence_gaps"]
                for criterion in ("power", "thermal", "risk", "scalability")
            )
        )

    def test_evaluation_separates_scenario_scores_and_accepts_relevant_indirect_evidence(self):
        import src.workflows.nodes.selection_decision as decision_module
        from src.workflows.selection_cache import clear_selection_cache

        clear_selection_cache()
        chain = MagicMock()
        chain.invoke.return_value = AIMessage(
            content=json.dumps(
                {
                    "evaluations": [
                        {
                            "candidate_id": "ald",
                            "criterion_scores": [
                                {
                                    "criterion": "performance",
                                    "score": 3,
                                    "rationale": "direct throughput evidence",
                                    "evidence_ids": ["E1"],
                                    "confidence": "high",
                                },
                                {
                                    "criterion": "power",
                                    "score": 4,
                                    "rationale": "background inference",
                                    "evidence_ids": ["E2"],
                                    "confidence": "medium",
                                },
                                {
                                    "criterion": "thermal",
                                    "score": 4,
                                    "rationale": "wrong criterion citation",
                                    "evidence_ids": ["E1"],
                                    "confidence": "medium",
                                },
                            ],
                            "scenario_scores": [
                                {
                                    "criterion": "scenario_uniformity",
                                    "score": 5,
                                    "rationale": "direct conformality evidence",
                                    "evidence_ids": ["E1"],
                                    "confidence": "high",
                                }
                            ],
                        }
                    ]
                }
            )
        )
        state = {
            "requirements": {
                "business_goal": "优先保证薄膜均匀性",
                "application_scenario": "高深宽比沉积",
                "scenario_criteria": [
                    {
                        "name": "薄膜均匀性与共形性",
                        "criterion_id": "scenario_uniformity",
                        "priority": "highest",
                    }
                ],
            },
            "feasible_candidates": [{"candidate_id": "ald", "name": "ALD"}],
            "evidence_records": [
                {
                    "evidence_id": "E1",
                    "candidate_ids": ["ald"],
                    "criteria": ["performance", "scenario_uniformity"],
                    "supports": [
                        {
                            "candidate_id": "ald",
                            "criterion": "performance",
                            "level": "direct",
                            "quote": "high throughput",
                        },
                        {
                            "candidate_id": "ald",
                            "criterion": "scenario_uniformity",
                            "level": "direct",
                            "quote": "conformal coverage",
                        },
                    ],
                    "content": "ALD provides high throughput and conformal coverage.",
                },
                {
                    "evidence_id": "E2",
                    "candidate_ids": [],
                    "criteria": [],
                    "supports": [
                        {
                            "candidate_id": "ald",
                            "criterion": "power",
                            "level": "indirect",
                            "quote": "",
                        }
                    ],
                    "content": "General process background.",
                },
            ],
            "evidence_gaps": [],
        }

        with patch.object(decision_module, "candidate_evaluation_chain", chain):
            result = decision_module.evaluate_candidates(state)

        evaluation = result["evaluations"][0]
        fixed = {
            item["criterion"]: item
            for item in evaluation["criterion_scores"]
        }
        self.assertEqual(fixed["performance"]["score"], 3)
        self.assertEqual(fixed["power"]["score"], 4)
        self.assertIsNone(fixed["thermal"]["score"])
        self.assertIsNone(evaluation["weighted_score"])
        self.assertEqual(evaluation["scenario_scores"][0]["score"], 5)
        self.assertTrue(any("thermal" in gap for gap in result["evidence_gaps"]))
        self.assertFalse(any("power" in gap for gap in result["evidence_gaps"]))

    def test_highest_priority_scenario_evidence_produces_conditional_primary(self):
        import src.workflows.nodes.selection_decision as decision_module
        from src.workflows.selection_cache import clear_selection_cache

        clear_selection_cache()
        chain = MagicMock()
        chain.invoke.return_value = AIMessage(
            content=json.dumps(
                {
                    "decision": "conditional",
                    "recommended_candidate_id": None,
                    "alternative_candidate_ids": [],
                    "rationale": ["ALD has stronger conformality evidence."],
                    "prerequisites": ["verify secondary metrics"],
                    "unresolved_items": [],
                }
            )
        )
        state = {
            "requirements": {
                "scenario_criteria": [
                    {
                        "name": "薄膜均匀性与共形性",
                        "criterion_id": "scenario_uniformity",
                        "priority": "highest",
                    }
                ]
            },
            "feasible_candidates": [
                {"candidate_id": "ald", "name": "ALD"},
                {"candidate_id": "cvd", "name": "CVD"},
            ],
            "evaluations": [
                {
                    "candidate_id": "ald",
                    "criterion_scores": [],
                    "scenario_scores": [
                        {
                            "criterion": "scenario_uniformity",
                            "score": 5,
                            "evidence_ids": ["E1"],
                        }
                    ],
                    "weighted_score": None,
                    "evidence_coverage": 0,
                },
                {
                    "candidate_id": "cvd",
                    "criterion_scores": [],
                    "scenario_scores": [
                        {
                            "criterion": "scenario_uniformity",
                            "score": None,
                            "evidence_ids": [],
                        }
                    ],
                    "weighted_score": None,
                    "evidence_coverage": 0,
                },
            ],
            "risk_items": [],
            "evidence_gaps": ["ALD power evidence missing"],
        }

        with patch.object(decision_module, "recommendation_chain", chain):
            result = decision_module.generate_recommendation(state)

        recommendation = result["recommendation"]
        self.assertEqual(recommendation["decision"], "conditional")
        self.assertEqual(recommendation["recommended_candidate_id"], "ald")
        self.assertEqual(recommendation["decision_basis"], "scenario_criteria")

    def test_review_vetoes_model_pass_for_conditional_without_primary(self):
        import src.workflows.nodes.selection_decision as decision_module
        from src.workflows.selection_cache import clear_selection_cache

        clear_selection_cache()
        chain = MagicMock()
        chain.invoke.return_value = AIMessage(
            content='{"verdict":"pass","reasons":[],"requested_queries":[]}'
        )

        with patch.object(decision_module, "recommendation_review_chain", chain):
            result = decision_module.review_recommendation(
                {
                    "requirements": {},
                    "feasible_candidates": [
                        {"candidate_id": "ald", "name": "ALD"},
                        {"candidate_id": "cvd", "name": "CVD"},
                    ],
                    "evaluations": [],
                    "risk_items": [],
                    "recommendation": {
                        "decision": "conditional",
                        "recommended_candidate_id": None,
                        "decision_basis": "scenario_criteria",
                        "rationale": ["ALD is preferred"],
                    },
                    "evidence_records": [],
                    "review_round": 0,
                }
            )

        self.assertEqual(result["recommendation_review"]["verdict"], "reevaluate")
        self.assertTrue(result["recommendation_review"]["reasons"])

    def test_review_allows_scores_that_reference_relevant_indirect_evidence(self):
        from src.workflows.nodes.selection_decision import _deterministic_review_reasons

        reasons = _deterministic_review_reasons(
            {
                "requirements": {"priorities": {"performance": 1}},
                "feasible_candidates": [{"candidate_id": "ald", "name": "ALD"}],
                "evaluations": [
                    {
                        "candidate_id": "ald",
                        "weighted_score": 5.0,
                        "criterion_scores": [
                            {
                                "criterion": "performance",
                                "score": 5,
                                "evidence_ids": ["E1"],
                            }
                        ],
                    }
                ],
                "risk_items": [],
                "recommendation": {
                    "decision": "recommend",
                    "recommended_candidate_id": "ald",
                    "decision_basis": "weighted_score",
                    "rationale": ["ALD综合分最高。"],
                },
                "evidence_records": [
                    {
                        "evidence_id": "E1",
                        "supports": [
                            {
                                "candidate_id": "ald",
                                "criterion": "performance",
                                "level": "indirect",
                                "quote": "",
                            }
                        ],
                        "content": "Background only.",
                    }
                ],
                "review_round": 0,
            }
        )

        self.assertFalse(any("非直接证据" in reason for reason in reasons))

    def test_review_vetoes_recommendation_text_that_conflicts_with_primary(self):
        from src.workflows.nodes.selection_decision import review_recommendation

        result = review_recommendation(
            {
                "requirements": {},
                "feasible_candidates": [
                    {"candidate_id": "ald", "name": "ALD"},
                    {"candidate_id": "cvd", "name": "CVD"},
                ],
                "evaluations": [
                    {"candidate_id": "ald", "weighted_score": 5.0},
                    {"candidate_id": "cvd", "weighted_score": 4.0},
                ],
                "risk_items": [],
                "recommendation": {
                    "decision": "recommend",
                    "recommended_candidate_id": "ald",
                    "decision_basis": "weighted_score",
                    "rationale": ["CVD更优，应当首选CVD。"],
                },
                "evidence_records": [],
                "review_round": 0,
            }
        )

        self.assertEqual(result["recommendation_review"]["verdict"], "reevaluate")
        self.assertTrue(
            any(
                "推荐文字" in reason
                for reason in result["recommendation_review"]["reasons"]
            )
        )

    def test_report_separates_scenario_and_fixed_metrics_without_none_values(self):
        from src.workflows.nodes.selection_decision import generate_report

        result = generate_report(
            {
                "status": "running",
                "requirements": {
                    "business_goal": "优先保证薄膜均匀性",
                    "application_scenario": "高深宽比薄膜沉积",
                    "scenario_criteria": [
                        {
                            "name": "薄膜均匀性与共形性",
                            "criterion_id": "scenario_uniformity",
                            "priority": "highest",
                        }
                    ],
                },
                "feasible_candidates": [
                    {"candidate_id": "ald", "name": "ALD", "summary": "原子层沉积"}
                ],
                "excluded_candidates": [],
                "evaluations": [
                    {
                        "candidate_id": "ald",
                        "criterion_scores": [
                            {
                                "criterion": "performance",
                                "score": 3,
                                "rationale": "有直接证据",
                                "evidence_ids": ["E1"],
                            },
                            {
                                "criterion": "thermal",
                                "score": None,
                                "rationale": "",
                                "evidence_ids": [],
                            },
                        ],
                        "scenario_scores": [
                            {
                                "criterion": "scenario_uniformity",
                                "score": 5,
                                "rationale": "高共形覆盖",
                                "evidence_ids": ["E1"],
                            }
                        ],
                        "weighted_score": None,
                    }
                ],
                "risk_items": [
                    {
                        "candidate_id": "ald",
                        "category": "throughput",
                        "level": "medium",
                        "impact": "产能可能较低",
                        "mitigation": "实测节拍",
                        "basis": "inference",
                        "evidence_ids": [],
                    }
                ],
                "recommendation": {
                    "decision": "conditional",
                    "recommended_candidate_id": "ald",
                    "decision_basis": "scenario_criteria",
                    "rationale": ["ALD在最高优先级场景指标评分领先。"],
                    "prerequisites": ["验证次要固定指标"],
                    "unresolved_items": ["thermal缺少直接证据"],
                },
                "evidence_records": [
                    {
                        "evidence_id": "E1",
                        "source": "web_search",
                        "url": "",
                        "title": "",
                        "source_reliability": "web_unattributed",
                    }
                ],
                "warnings": [],
            }
        )

        report = result["report"]
        self.assertIn("## 3. 方案对比总表", report)
        self.assertIn("## 4. 关键指标分析", report)
        self.assertIn("薄膜均匀性与共形性", report)
        self.assertIn("散热与热管理缺少直接证据", report)
        self.assertNotIn("None", report)
        self.assertIn("| 推断 |", report)
        self.assertIn("来源未验证", report)

    def test_report_includes_original_request_and_clarification_rounds(self):
        from src.workflows.nodes.selection_decision import (
            build_report_data,
            generate_report,
        )

        state = {
            "user_request": "帮我做半导体技术选型",
            "clarification_answers": [
                "应用于手机SoC，优先功耗",
                "两年内量产",
            ],
            "clarification_history": [
                {
                    "questions": [
                        "这次技术选型最希望解决的业务目标是什么？"
                    ],
                    "answer": "应用于手机SoC，优先功耗",
                },
                {
                    "questions": [
                        "该技术将用于什么产品、芯片或制造场景？",
                        "是否有明确的成本或设备投入限制？",
                    ],
                    "answer": "两年内量产",
                },
            ],
            "status": "completed",
            "requirements": {
                "business_goal": "降低功耗",
                "application_scenario": "手机SoC",
                "active_criteria": ["performance", "cost"],
                "priorities": {
                    "performance": 0.7,
                    "cost": 0.3,
                },
                "hard_constraints": [],
            },
            "feasible_candidates": [
                {"candidate_id": "finfet", "name": "FinFET", "summary": ""},
                {"candidate_id": "gaa", "name": "GAA", "summary": ""},
            ],
            "excluded_candidates": [],
            "evaluations": [
                {
                    "candidate_id": "finfet",
                    "criterion_scores": [
                        {
                            "criterion": "performance",
                            "score": 4,
                            "rationale": "证据",
                            "evidence_ids": ["E1"],
                        }
                    ],
                    "weighted_score": None,
                },
                {
                    "candidate_id": "gaa",
                    "criterion_scores": [
                        {
                            "criterion": "performance",
                            "score": 5,
                            "rationale": "证据",
                            "evidence_ids": ["E1"],
                        }
                    ],
                    "weighted_score": None,
                },
            ],
            "risk_items": [],
            "recommendation": {
                "decision": "conditional",
                "recommended_candidate_id": "gaa",
                "decision_basis": "scenario_criteria",
                "rationale": ["低功耗潜力"],
                "prerequisites": [],
                "unresolved_items": [],
            },
            "evidence_records": [],
            "warnings": [],
        }

        result = generate_report(state)
        report_data = build_report_data(state)

        self.assertIn("| 原始需求 | 帮我做半导体技术选型 |", result["report"])
        self.assertIn("### 澄清记录", result["report"])
        self.assertIn("第1轮问题：这次技术选型最希望解决的业务目标是什么？", result["report"])
        self.assertIn("回答：应用于手机SoC，优先功耗", result["report"])
        self.assertIn("第2轮问题：该技术将用于什么产品、芯片或制造场景？；是否有明确的成本或设备投入限制？", result["report"])
        self.assertIn("回答：两年内量产", result["report"])
        self.assertEqual(
            report_data["conversation"]["user_request"],
            "帮我做半导体技术选型",
        )
        self.assertEqual(
            report_data["conversation"]["clarification_answers"],
            ["应用于手机SoC，优先功耗", "两年内量产"],
        )
        self.assertEqual(
            len(report_data["conversation"]["clarification_history"]),
            2,
        )

    def test_report_without_clarification_shows_no_rounds(self):
        from src.workflows.nodes.selection_decision import (
            build_report_data,
            generate_report,
        )

        state = {
            "status": "completed",
            "requirements": {
                "business_goal": "降低功耗",
                "application_scenario": "手机SoC",
            },
            "feasible_candidates": [
                {"candidate_id": "gaa", "name": "GAA", "summary": ""}
            ],
            "excluded_candidates": [],
            "evaluations": [],
            "risk_items": [],
            "recommendation": {"decision": "insufficient_evidence"},
            "evidence_records": [],
            "warnings": [],
        }
        result = generate_report(state)
        self.assertIn("本轮未发起需求澄清。", result["report"])
        self.assertEqual(
            build_report_data(state)["conversation"]["clarification_answers"],
            [],
        )

    def test_report_marks_inactive_fixed_criteria_without_creating_gaps(self):
        from src.workflows.nodes.selection_decision import generate_report

        result = generate_report(
            {
                "status": "running",
                "requirements": {
                    "business_goal": "比较沉积工艺",
                    "application_scenario": "高深宽比沉积",
                    "active_criteria": ["performance", "cost", "maturity"],
                },
                "feasible_candidates": [
                    {"candidate_id": "ald", "name": "ALD", "summary": ""}
                ],
                "excluded_candidates": [],
                "evaluations": [
                    {
                        "candidate_id": "ald",
                        "criterion_scores": [
                            {
                                "criterion": "performance",
                                "score": 4,
                                "rationale": "有相关证据",
                                "evidence_ids": ["E1"],
                            },
                            {
                                "criterion": "cost",
                                "score": None,
                                "rationale": "",
                                "evidence_ids": [],
                            },
                            {
                                "criterion": "power",
                                "score": None,
                                "rationale": "本次未纳入评估",
                                "evidence_ids": [],
                            },
                        ],
                        "weighted_score": None,
                    }
                ],
                "risk_items": [],
                "recommendation": {"decision": "insufficient_evidence"},
                "evidence_records": [],
                "warnings": [],
            }
        )

        report = result["report"]
        self.assertIn("本次启用指标：生产效率/性能、成本、量产成熟度", report)
        self.assertIn("本次未纳入指标：功耗与能效", report)
        self.assertNotIn("功耗与能效：本次未纳入评估", report)

    def test_report_semantically_deduplicates_candidate_metric_gaps(self):
        from src.workflows.nodes.selection_decision import build_report_data

        report_data = build_report_data(
            {
                "status": "needs_human_review",
                "requirements": {
                    "business_goal": "比较ALD和CVD",
                    "application_scenario": "高深宽比薄膜沉积",
                    "active_criteria": ["cost", "maturity"],
                },
                "feasible_candidates": [
                    {"candidate_id": "ald", "name": "ALD"},
                    {"candidate_id": "cvd", "name": "CVD"},
                ],
                "evaluations": [],
                "risk_items": [],
                "evidence_records": [],
                "evidence_gaps": [
                    "补充ALD在cost方面的可引用证据。",
                    "补充CVD在cost方面的可引用证据。",
                    "补充CVD在maturity方面的可引用证据。",
                ],
                "recommendation": {
                    "decision": "conditional",
                    "unresolved_items": [
                        "ALD和CVD的成本对比未量化，需进一步调研。",
                        "CVD的成熟度未知，需收集证据。",
                        "ALD量产中的具体挑战未明确，需深入调研。",
                    ],
                },
            }
        )

        descriptions = [
            item["description"] for item in report_data["evidence_gaps"]
        ]
        self.assertEqual(len(descriptions), 4)
        self.assertFalse(any("成本对比未量化" in item for item in descriptions))
        self.assertTrue(any("ALD量产中的具体挑战" in item for item in descriptions))

    def test_markdown_v2_is_a_standalone_user_facing_decision_report(self):
        from src.workflows.nodes.selection_decision import generate_report

        state = {
            "status": "needs_human_review",
            "requirements": {
                "business_goal": "为新能源汽车功率器件选型",
                "application_scenario": "800V主驱逆变器",
                "active_criteria": ["performance", "cost", "maturity"],
                "priorities": {
                    "performance": 0.5,
                    "cost": 0.2,
                    "maturity": 0.3,
                },
                "hard_constraints": ["需满足800V平台"],
                "scenario_criteria": [
                    {
                        "name": "耐压能力",
                        "criterion_id": "scenario_voltage",
                        "priority": "highest",
                    }
                ],
            },
            "feasible_candidates": [
                {"candidate_id": "sic", "name": "SiC", "summary": "碳化硅器件"},
                {"candidate_id": "gan", "name": "GaN", "summary": "氮化镓器件"},
            ],
            "excluded_candidates": [],
            "evaluations": [
                {
                    "candidate_id": "sic",
                    "criterion_scores": [
                        {"criterion": "performance", "score": 5, "rationale": "目标工况效率较高", "evidence_ids": ["E1"], "confidence": "high"},
                        {"criterion": "cost", "score": 2, "rationale": "衬底成本较高", "evidence_ids": ["E1"], "confidence": "medium"},
                        {"criterion": "maturity", "score": None, "rationale": "缺少目标平台量产数据", "evidence_ids": [], "confidence": "low"},
                        {"criterion": "power", "score": None, "rationale": "本次未纳入评估", "evidence_ids": [], "confidence": "low"},
                    ],
                    "scenario_scores": [
                        {"criterion": "scenario_voltage", "score": 5, "rationale": "适合高压场景", "evidence_ids": ["E1"], "confidence": "high"}
                    ],
                    "weighted_score": None,
                    "evidence_coverage": 2 / 3,
                    "scenario_coverage": 1.0,
                },
                {
                    "candidate_id": "gan",
                    "criterion_scores": [
                        {"criterion": "performance", "score": 4, "rationale": "高频性能较好", "evidence_ids": ["E2"], "confidence": "medium"},
                        {"criterion": "cost", "score": 3, "rationale": "成本中等", "evidence_ids": ["E2"], "confidence": "medium"},
                        {"criterion": "maturity", "score": None, "rationale": "缺少目标平台量产数据", "evidence_ids": [], "confidence": "low"},
                        {"criterion": "power", "score": None, "rationale": "本次未纳入评估", "evidence_ids": [], "confidence": "low"},
                    ],
                    "scenario_scores": [
                        {"criterion": "scenario_voltage", "score": 3, "rationale": "目标电压证据较少", "evidence_ids": ["E2"], "confidence": "medium"}
                    ],
                    "weighted_score": None,
                    "evidence_coverage": 2 / 3,
                    "scenario_coverage": 1.0,
                },
            ],
            "risk_items": [],
            "evidence_gaps": ["补充SiC在maturity方面的可引用证据。"],
            "recommendation": {
                "decision": "conditional",
                "recommended_candidate_id": "sic",
                "decision_basis": "scenario_criteria",
                "rationale": ["在最高优先级耐压指标上SiC评分领先。"],
                "prerequisites": ["补充量产成熟度数据"],
                "unresolved_items": ["补充SiC在maturity方面的可引用证据。"],
            },
            "recommendation_review": {
                "verdict": "evidence_gap",
                "reasons": ["量产数据不完整"],
            },
            "evidence_round": 2,
            "review_round": 1,
            "evidence_records": [
                {
                    "evidence_id": "E1",
                    "title": "SiC高压性能资料",
                    "source": r"D:\\Mydocs\\rag_md\\sic.md",
                    "url": "",
                    "source_reliability": "internal",
                    "retrieval_sources": ["vector", "local_hybrid"],
                    "content": "SiC在高压工况下具有较好的耐压能力。",
                    "supports": [
                        {"candidate_id": "sic", "criterion": "scenario_voltage", "level": "direct", "quote": "较好的耐压能力"}
                    ],
                },
                {
                    "evidence_id": "E2",
                    "title": "GaN应用资料",
                    "source": "https://example.com/gan",
                    "url": "https://example.com/gan",
                    "source_reliability": "web_identified",
                    "retrieval_sources": ["web_search"],
                    "content": "GaN在高频应用中具有优势。",
                    "supports": [
                        {"candidate_id": "gan", "criterion": "performance", "level": "indirect", "quote": "GaN在高频应用中具有优势"}
                    ],
                },
            ],
            "warnings": ["证据需要人工复核。"],
        }

        report = generate_report(state)["report"]

        self.assertLess(report.index("## 1. 执行摘要"), report.index("## 2. 需求与评估范围"))
        self.assertIn("| 指标 | 权重 | SiC | GaN | 当前判断 | 证据状态 |", report)
        self.assertIn("生产效率/性能", report)
        self.assertIn("耐压能力", report)
        self.assertNotIn("scenario_voltage", report)
        self.assertNotIn(r"D:\\Mydocs", report)
        self.assertIn("[查看来源](https://example.com/gan)", report)
        self.assertIn("本次未纳入指标：功耗", report)
        self.assertNotIn("power：本次未纳入评估", report)
        self.assertEqual(report.count("补充SiC在量产成熟度方面的可引用证据"), 1)

    def test_generate_report_returns_compact_structured_report_data(self):
        from src.workflows.nodes.selection_decision import generate_report

        result = generate_report(
            {
                "status": "completed",
                "requirements": {
                    "business_goal": "比较SiC和GaN",
                    "application_scenario": "新能源汽车功率器件",
                    "active_criteria": ["performance", "cost"],
                    "priorities": {"performance": 0.7, "cost": 0.3},
                    "hard_constraints": [],
                },
                "feasible_candidates": [
                    {"candidate_id": "sic", "name": "SiC", "summary": "碳化硅"},
                    {"candidate_id": "gan", "name": "GaN", "summary": "氮化镓"},
                ],
                "excluded_candidates": [],
                "evaluations": [
                    {
                        "candidate_id": "sic",
                        "criterion_scores": [
                            {"criterion": "performance", "score": 5, "rationale": "性能证据", "evidence_ids": ["E1"], "confidence": "high"},
                            {"criterion": "cost", "score": 2, "rationale": "成本证据", "evidence_ids": ["E1"], "confidence": "medium"},
                        ],
                        "scenario_scores": [],
                        "weighted_score": 4.1,
                        "evidence_coverage": 1.0,
                    },
                    {
                        "candidate_id": "gan",
                        "criterion_scores": [
                            {"criterion": "performance", "score": 4, "rationale": "性能证据", "evidence_ids": ["E2"], "confidence": "medium"},
                            {"criterion": "cost", "score": 3, "rationale": "成本证据", "evidence_ids": ["E2"], "confidence": "medium"},
                        ],
                        "scenario_scores": [],
                        "weighted_score": 3.7,
                        "evidence_coverage": 1.0,
                    },
                ],
                "risk_items": [],
                "evidence_gaps": [],
                "recommendation": {
                    "decision": "recommend",
                    "recommended_candidate_id": "sic",
                    "decision_basis": "weighted_score",
                    "rationale": ["SiC综合分最高。"],
                    "prerequisites": [],
                    "unresolved_items": [],
                },
                "recommendation_review": {"verdict": "pass", "reasons": []},
                "evidence_records": [
                    {
                        "evidence_id": "E1",
                        "title": "SiC测试资料",
                        "source": r"D:\\Mydocs\\private\\sic.md",
                        "url": "",
                        "source_reliability": "internal",
                        "content": "x" * 1000,
                        "supports": [
                            {"candidate_id": "sic", "criterion": "performance", "level": "direct", "quote": "SiC性能支持片段"}
                        ],
                    }
                ],
                "warnings": [],
            }
        )

        report_data = result["report_data"]
        self.assertEqual(report_data["schema_version"], "1.0")
        self.assertEqual(report_data["report_version"], "2.0")
        self.assertEqual(report_data["decision"]["primary_candidate_id"], "sic")
        self.assertEqual(report_data["decision"]["primary_candidate_name"], "SiC")
        self.assertEqual(report_data["requirements"]["active_criteria"][0]["label"], "生产效率/性能")
        self.assertEqual(report_data["comparison"]["criteria"][0]["cells"][0]["score"], 5)
        self.assertEqual(report_data["evidence"][0]["display_source"], "内部知识库")
        self.assertNotIn("Mydocs", json.dumps(report_data, ensure_ascii=False))
        self.assertIn(report_data["decision"]["summary"], result["report"])

    def test_evaluation_cache_reuses_only_identical_decision_input(self):
        import src.workflows.nodes.selection_decision as decision_module
        from src.workflows.selection_cache import clear_selection_cache

        clear_selection_cache()
        chain = MagicMock()
        chain.invoke.return_value = AIMessage(content='{"evaluations": []}')
        state = {
            "requirements": {
                "business_goal": "比较工艺",
                "application_scenario": "高深宽比沉积",
            },
            "feasible_candidates": [],
            "evidence_records": [],
        }
        with patch.object(decision_module, "candidate_evaluation_chain", chain):
            first = decision_module.evaluate_candidates(state)
            second = decision_module.evaluate_candidates(state)
            decision_module.evaluate_candidates(
                {
                    **state,
                    "requirements": {
                        **state["requirements"],
                        "application_scenario": "平面薄膜沉积",
                    },
                }
            )

        self.assertEqual(first, second)
        self.assertEqual(chain.invoke.call_count, 2)

    def test_decision_evidence_cards_are_bounded_per_candidate_criterion(self):
        from src.workflows.nodes.selection_decision import _decision_evidence_cards

        cards = _decision_evidence_cards(
            [
                {
                    "evidence_id": f"E{index}",
                    "candidate_ids": ["ald"],
                    "criteria": ["thermal"],
                    "supports": [
                        {
                            "candidate_id": "ald",
                            "criterion": "thermal",
                            "level": "direct",
                            "quote": "x",
                        }
                    ],
                    "content": "x" * 600,
                }
                for index in range(3)
            ]
        )

        self.assertEqual([card["evidence_id"] for card in cards], ["E0", "E1"])
        self.assertTrue(all(len(card["content"]) == 400 for card in cards))

    def test_review_reuses_identical_audit_input_and_invalidates_on_evidence_change(self):
        import src.workflows.nodes.selection_decision as decision_module

        decision_module.clear_review_cache()
        chain = MagicMock()
        chain.invoke.return_value = AIMessage(
            content='{"verdict":"pass","reasons":[],"requested_queries":[]}'
        )
        state = {
            "requirements": {},
            "feasible_candidates": [{"candidate_id": "gaa", "name": "GAA"}],
            "evaluations": [{"candidate_id": "gaa", "weighted_score": 4.0}],
            "risk_items": [],
            "recommendation": {
                "decision": "recommend",
                "recommended_candidate_id": "gaa",
                "decision_basis": "weighted_score",
            },
            "evidence_records": [{
                "evidence_id": "E1",
                "candidate_ids": ["gaa"],
                "criteria": ["cost"],
                "supports": [
                    {
                        "candidate_id": "gaa",
                        "criterion": "cost",
                        "level": "direct",
                        "quote": "first",
                    }
                ],
                "content": "first",
            }],
        }
        with patch.object(decision_module, "recommendation_review_chain", chain):
            decision_module.review_recommendation(state)
            decision_module.review_recommendation(state)
            changed_state = {**state, "evidence_records": [{
                "evidence_id": "E1",
                "candidate_ids": ["gaa"],
                "criteria": ["cost"],
                "supports": [
                    {
                        "candidate_id": "gaa",
                        "criterion": "cost",
                        "level": "direct",
                        "quote": "changed",
                    }
                ],
                "content": "changed",
            }]}
            decision_module.review_recommendation(changed_state)

        self.assertEqual(chain.invoke.call_count, 2)

    def test_missing_power_evidence_forces_a_null_power_score(self):
        import src.workflows.nodes.selection_decision as decision_module

        score_items = []
        for criterion in (
            "performance",
            "power",
            "cost",
            "maturity",
            "risk",
            "scalability",
        ):
            score_items.append(
                {
                    "criterion": criterion,
                    "score": 4,
                    "rationale": "model score",
                    "evidence_ids": ["E1"],
                    "confidence": "high",
                }
            )
        chain = MagicMock()
        chain.invoke.return_value = AIMessage(
            content=json.dumps(
                {
                    "evaluations": [
                        {
                            "candidate_id": "gaa",
                            "criterion_scores": score_items,
                        }
                    ]
                }
            )
        )
        with patch.object(decision_module, "candidate_evaluation_chain", chain):
            result = decision_module.evaluate_candidates(
                {
                    "requirements": {"priorities": {}},
                    "feasible_candidates": [
                        {"candidate_id": "gaa", "name": "GAA"},
                    ],
                    "evidence_records": [
                        {
                            "evidence_id": "E1",
                            "candidate_ids": ["gaa"],
                            "criteria": [
                                "performance",
                                "cost",
                                "maturity",
                                "risk",
                                "scalability",
                            ],
                            "content": "No valid power evidence",
                        }
                    ],
                }
            )

        evaluation = result["evaluations"][0]
        power = next(
            item
            for item in evaluation["criterion_scores"]
            if item["criterion"] == "power"
        )
        self.assertIsNone(power["score"])
        self.assertIsNone(evaluation["weighted_score"])

    def test_evaluate_candidates_calculates_weighted_score_in_python(self):
        import src.workflows.nodes.selection_decision as decision_module

        chain = MagicMock()
        chain.invoke.return_value = AIMessage(
            content="""{"evaluations": [{
                "candidate_id": "finfet",
                "criterion_scores": [
                    {"criterion": "performance", "score": 4, "rationale": "r", "evidence_ids": ["E1"], "confidence": "high"},
                    {"criterion": "power", "score": 5, "rationale": "r", "evidence_ids": ["E1"], "confidence": "high"},
                    {"criterion": "thermal", "score": 4, "rationale": "r", "evidence_ids": ["E1"], "confidence": "high"},
                    {"criterion": "cost", "score": 3, "rationale": "r", "evidence_ids": ["E1"], "confidence": "high"},
                    {"criterion": "maturity", "score": 5, "rationale": "r", "evidence_ids": ["E1"], "confidence": "high"},
                    {"criterion": "risk", "score": 2, "rationale": "r", "evidence_ids": ["E1"], "confidence": "medium"},
                    {"criterion": "scalability", "score": 4, "rationale": "r", "evidence_ids": ["E1"], "confidence": "high"}
                ]
            }]}"""
        )
        with patch.object(decision_module, "candidate_evaluation_chain", chain):
            result = decision_module.evaluate_candidates(
                {
                    "requirements": {
                        "priorities": {
                            "performance": 0.2,
                            "power": 0.0,
                            "thermal": 0.0,
                            "cost": 0.2,
                            "maturity": 0.2,
                            "risk": 0.2,
                            "scalability": 0.2,
                        }
                    },
                    "feasible_candidates": [
                        {"candidate_id": "finfet", "name": "FinFET"},
                    ],
                    "evidence_records": [
                        {
                            "evidence_id": "E1",
                            "candidate_ids": ["finfet"],
                            "criteria": [
                                "performance",
                                "power",
                                "thermal",
                                "cost",
                                "maturity",
                                "risk",
                                "scalability",
                            ],
                            "content": "evidence",
                            "supports": [
                                {
                                    "candidate_id": "finfet",
                                    "criterion": criterion,
                                    "level": "direct",
                                    "quote": "evidence",
                                }
                                for criterion in (
                                    "performance",
                                    "power",
                                    "thermal",
                                    "cost",
                                    "maturity",
                                    "risk",
                                    "scalability",
                                )
                            ],
                        }
                    ],
                }
            )

        self.assertEqual(result["evaluations"][0]["weighted_score"], 3.6)
        self.assertEqual(result["evaluations"][0]["evidence_coverage"], 1.0)

    def test_evaluate_candidates_does_not_score_without_valid_evidence(self):
        import src.workflows.nodes.selection_decision as decision_module

        chain = MagicMock()
        chain.invoke.return_value = AIMessage(
            content="""{"evaluations": [{
                "candidate_id": "gaa",
                "criterion_scores": [{
                    "criterion": "performance",
                    "score": 5,
                    "rationale": "unsupported",
                    "evidence_ids": ["UNKNOWN"],
                    "confidence": "high"
                }]
            }]}"""
        )
        with patch.object(decision_module, "candidate_evaluation_chain", chain):
            result = decision_module.evaluate_candidates(
                {
                    "requirements": {"priorities": {}},
                    "feasible_candidates": [
                        {"candidate_id": "gaa", "name": "GAA"},
                    ],
                    "evidence_records": [],
                }
            )

        evaluation = result["evaluations"][0]
        performance = next(
            score
            for score in evaluation["criterion_scores"]
            if score["criterion"] == "performance"
        )
        self.assertIsNone(performance["score"])
        self.assertEqual(evaluation["weighted_score"], None)

    def test_risk_evidence_must_belong_to_the_same_candidate(self):
        import src.workflows.nodes.selection_decision as decision_module

        chain = MagicMock()
        chain.invoke.return_value = AIMessage(
            content="""{"risk_items": [{
                "candidate_id": "gaa",
                "category": "maturity",
                "level": "high",
                "impact": "量产验证不足",
                "mitigation": "分阶段验证",
                "evidence_ids": ["E_FINFET"],
                "basis": "evidence"
            }]}"""
        )
        with patch.object(decision_module, "risk_analysis_chain", chain):
            result = decision_module.analyze_risks(
                {
                    "requirements": {},
                    "evaluations": [],
                    "evidence_records": [
                        {
                            "evidence_id": "E_FINFET",
                            "candidate_ids": ["finfet"],
                            "criteria": ["maturity"],
                            "content": "FinFET maturity evidence",
                        }
                    ],
                }
            )

        risk = result["risk_items"][0]
        self.assertEqual(risk["evidence_ids"], [])
        self.assertEqual(risk["basis"], "to_verify")

    def test_unknown_risk_level_is_preserved_for_unverified_risk(self):
        import src.workflows.nodes.selection_decision as decision_module

        chain = MagicMock()
        chain.invoke.return_value = AIMessage(
            content="""{"risk_items": [{
                "candidate_id": "gaa",
                "category": "reliability",
                "level": "unknown",
                "impact": "缺乏可靠性数据，需验证",
                "mitigation": "补充可靠性验证",
                "evidence_ids": [],
                "basis": "to_verify"
            }]}"""
        )
        with patch.object(decision_module, "risk_analysis_chain", chain):
            result = decision_module.analyze_risks(
                {
                    "requirements": {},
                    "evaluations": [],
                    "evidence_records": [],
                }
            )

        self.assertEqual(result["risk_items"][0]["level"], "unknown")
        self.assertEqual(result["risk_items"][0]["basis"], "to_verify")

    def test_incomplete_evaluation_cannot_become_unconditional_recommendation(self):
        import src.workflows.nodes.selection_decision as decision_module

        chain = MagicMock()
        chain.invoke.return_value = AIMessage(
            content="""{
                "decision": "recommend",
                "recommended_candidate_id": "gaa",
                "alternative_candidate_ids": [],
                "rationale": ["best"],
                "prerequisites": [],
                "unresolved_items": []
            }"""
        )
        with patch.object(decision_module, "recommendation_chain", chain):
            result = decision_module.generate_recommendation(
                {
                    "requirements": {},
                    "feasible_candidates": [
                        {"candidate_id": "gaa", "name": "GAA"},
                    ],
                    "evaluations": [
                        {
                            "candidate_id": "gaa",
                            "criterion_scores": [],
                            "weighted_score": None,
                            "evidence_coverage": 0.4,
                        }
                    ],
                    "risk_items": [],
                    "evidence_gaps": ["成本证据不足"],
                }
            )

        self.assertEqual(
            result["recommendation"]["decision"],
            "insufficient_evidence",
        )
        self.assertIn(
            "成本证据不足",
            result["recommendation"]["unresolved_items"],
        )
        self.assertIsNone(result["recommendation"]["recommended_candidate_id"])
        self.assertEqual(
            result["recommendation"]["decision_basis"],
            "insufficient_evidence",
        )

    def test_malformed_recommendation_json_uses_evidence_score_fallback(self):
        """A transient model format error must not discard verified scoring."""
        import src.workflows.nodes.selection_decision as decision_module

        chain = MagicMock()
        chain.invoke.return_value = AIMessage(content="暂时无法生成 JSON")
        with patch.object(decision_module, "recommendation_chain", chain):
            result = decision_module.generate_recommendation(
                {
                    "requirements": {},
                    "feasible_candidates": [
                        {"candidate_id": "hbm3", "name": "HBM3"},
                        {"candidate_id": "hbm3e", "name": "HBM3E"},
                    ],
                    "evaluations": [
                        {
                            "candidate_id": "hbm3",
                            "weighted_score": 4.1,
                            "evidence_coverage": 1.0,
                        },
                        {
                            "candidate_id": "hbm3e",
                            "weighted_score": 4.6,
                            "evidence_coverage": 1.0,
                        },
                    ],
                    "risk_items": [],
                    "evidence_gaps": [],
                }
            )

        recommendation = result["recommendation"]
        self.assertEqual(recommendation["decision"], "recommend")
        self.assertEqual(recommendation["recommended_candidate_id"], "hbm3e")
        self.assertEqual(recommendation["decision_basis"], "weighted_score")
        self.assertIn("推荐结论模型返回格式异常", result["warnings"][0])

    def test_unconditional_recommendation_cannot_conflict_with_score_ranking(self):
        import src.workflows.nodes.selection_decision as decision_module

        chain = MagicMock()
        chain.invoke.return_value = AIMessage(
            content="""{
                "decision": "recommend",
                "recommended_candidate_id": "gaa",
                "alternative_candidate_ids": ["finfet"],
                "rationale": ["模型偏好GAA"],
                "prerequisites": [],
                "unresolved_items": []
            }"""
        )
        with patch.object(decision_module, "recommendation_chain", chain):
            result = decision_module.generate_recommendation(
                {
                    "requirements": {},
                    "feasible_candidates": [
                        {"candidate_id": "finfet", "name": "FinFET"},
                        {"candidate_id": "gaa", "name": "GAA"},
                    ],
                    "evaluations": [
                        {
                            "candidate_id": "finfet",
                            "weighted_score": 4.8,
                            "evidence_coverage": 1.0,
                        },
                        {
                            "candidate_id": "gaa",
                            "weighted_score": 2.0,
                            "evidence_coverage": 1.0,
                        },
                    ],
                    "risk_items": [],
                    "evidence_gaps": [],
                }
            )

        recommendation = result["recommendation"]
        self.assertEqual(recommendation["decision"], "recommend")
        self.assertEqual(recommendation["recommended_candidate_id"], "finfet")
        self.assertEqual(recommendation["decision_basis"], "weighted_score")

    def test_recommendation_filters_invalid_alternative_ids(self):
        import src.workflows.nodes.selection_decision as decision_module

        chain = MagicMock()
        chain.invoke.return_value = AIMessage(
            content="""{
                "decision": "recommend",
                "recommended_candidate_id": "finfet",
                "alternative_candidate_ids": ["finfet", "gaa", "unknown", "gaa"],
                "rationale": ["综合评分最高"],
                "prerequisites": [],
                "unresolved_items": []
            }"""
        )
        with patch.object(decision_module, "recommendation_chain", chain):
            result = decision_module.generate_recommendation(
                {
                    "requirements": {},
                    "feasible_candidates": [
                        {"candidate_id": "finfet", "name": "FinFET"},
                        {"candidate_id": "gaa", "name": "GAA"},
                    ],
                    "evaluations": [
                        {
                            "candidate_id": "finfet",
                            "weighted_score": 4.8,
                            "evidence_coverage": 1.0,
                        },
                        {
                            "candidate_id": "gaa",
                            "weighted_score": 4.0,
                            "evidence_coverage": 1.0,
                        },
                    ],
                    "risk_items": [],
                    "evidence_gaps": [],
                }
            )

        self.assertEqual(
            result["recommendation"]["alternative_candidate_ids"],
            ["gaa"],
        )

    def test_review_receives_evidence_and_persists_requested_queries(self):
        import src.workflows.nodes.selection_decision as decision_module

        chain = MagicMock()
        chain.invoke.return_value = AIMessage(
            content="""{
                "verdict": "evidence_gap",
                "reasons": ["成本证据不足"],
                "requested_queries": [
                    "补充GAA成本证据",
                    "补充GAA成本证据",
                    ""
                ]
            }"""
        )
        with patch.object(
            decision_module,
            "recommendation_review_chain",
            chain,
        ):
            result = decision_module.review_recommendation(
                {
                    "requirements": {},
                    "feasible_candidates": [
                        {"candidate_id": "gaa", "name": "GAA"},
                    ],
                    "evaluations": [
                        {"candidate_id": "gaa", "weighted_score": 4.0},
                    ],
                    "risk_items": [],
                    "recommendation": {
                        "decision": "recommend",
                        "recommended_candidate_id": "gaa",
                        "decision_basis": "weighted_score",
                    },
                    "evidence_records": [
                        {
                            "evidence_id": "E1",
                            "candidate_ids": ["gaa"],
                            "criteria": ["cost"],
                            "supports": [
                                {
                                    "candidate_id": "gaa",
                                    "criterion": "cost",
                                    "level": "direct",
                                    "quote": "xxxx",
                                }
                            ],
                            "content": "x" * 2500,
                        }
                    ],
                    "evidence_gaps": ["原有缺口"],
                    "review_round": 0,
                }
            )

        review_input = chain.invoke.call_args.args[0]
        evidence = json.loads(review_input["evidence"])
        self.assertEqual(len(evidence[0]["content"]), 400)
        self.assertEqual(
            result["missing_evidence_queries"],
            ["补充GAA成本证据"],
        )
        self.assertEqual(
            result["evidence_gaps"],
            ["原有缺口", "补充GAA成本证据"],
        )

    def test_review_reevaluate_with_queries_is_repaired_as_evidence_gap(self):
        import src.workflows.nodes.selection_decision as decision_module
        from src.workflows.routes.technology_selection import (
            route_recommendation_review,
        )

        decision_module.clear_review_cache()
        chain = MagicMock()
        chain.invoke.return_value = AIMessage(
            content=json.dumps(
                {
                    "verdict": "reevaluate",
                    "reasons": ["800V场景证据不足"],
                    "requested_queries": [
                        "补充GaN 1200V车规认证证据",
                        "补充GaN 1200V车规认证证据",
                        "",
                    ],
                },
                ensure_ascii=False,
            )
        )
        with patch.object(
            decision_module,
            "recommendation_review_chain",
            chain,
        ):
            result = decision_module.review_recommendation(
                {
                    "requirements": {},
                    "feasible_candidates": [
                        {"candidate_id": "gan", "name": "GaN"},
                    ],
                    "evaluations": [
                        {"candidate_id": "gan", "weighted_score": 3.5},
                    ],
                    "risk_items": [],
                    "recommendation": {
                        "decision": "recommend",
                        "recommended_candidate_id": "gan",
                        "decision_basis": "weighted_score",
                    },
                    "evidence_records": [],
                    "evidence_gaps": [],
                    "evidence_round": 1,
                    "review_round": 0,
                }
            )

        self.assertEqual(
            result["recommendation_review"]["verdict"],
            "evidence_gap",
        )
        self.assertEqual(
            result["missing_evidence_queries"],
            ["补充GaN 1200V车规认证证据"],
        )
        self.assertEqual(
            route_recommendation_review(result),
            "optimize_evidence_queries",
        )

    def test_reevaluation_consumes_review_feedback_instead_of_reusing_cache(self):
        import src.workflows.nodes.selection_decision as decision_module
        from src.workflows.selection_cache import clear_selection_cache

        clear_selection_cache()
        chain = MagicMock()
        chain.invoke.return_value = AIMessage(content='{"evaluations": []}')
        state = {
            "requirements": {
                "business_goal": "比较SiC和GaN",
                "application_scenario": "新能源汽车功率器件",
            },
            "feasible_candidates": [],
            "evidence_records": [],
            "evidence_gaps": [],
        }
        with patch.object(decision_module, "candidate_evaluation_chain", chain):
            decision_module.evaluate_candidates(state)
            decision_module.evaluate_candidates(
                {
                    **state,
                    "recommendation_review": {
                        "verdict": "reevaluate",
                        "reasons": ["需要重新核对耐压场景评分"],
                        "requested_queries": [],
                    },
                }
            )

        self.assertEqual(chain.invoke.call_count, 2)
        feedback = json.loads(chain.invoke.call_args.args[0]["review_feedback"])
        self.assertEqual(feedback["reasons"], ["需要重新核对耐压场景评分"])

    def test_high_priority_scenario_conflict_becomes_auditable_conditional_choice(self):
        import src.workflows.nodes.selection_decision as decision_module
        from src.workflows.selection_cache import clear_selection_cache

        clear_selection_cache()
        chain = MagicMock()
        chain.invoke.return_value = AIMessage(
            content=json.dumps(
                {
                    "decision": "recommend",
                    "recommended_candidate_id": "gan",
                    "alternative_candidate_ids": ["gan"],
                    "rationale": ["GaN综合分最高"],
                    "prerequisites": [],
                    "unresolved_items": ["补充CVD在量产成熟度方面的可引用证据"],
                },
                ensure_ascii=False,
            )
        )
        state = {
            "requirements": {
                "scenario_criteria": [
                    {
                        "name": "耐压能力",
                        "criterion_id": "scenario_voltage",
                        "priority": "high",
                    }
                ]
            },
            "feasible_candidates": [
                {"candidate_id": "sic", "name": "SiC"},
                {"candidate_id": "gan", "name": "GaN"},
            ],
            "evaluations": [
                {
                    "candidate_id": "sic",
                    "weighted_score": 3.25,
                    "scenario_scores": [
                        {"criterion": "scenario_voltage", "score": 5}
                    ],
                },
                {
                    "candidate_id": "gan",
                    "weighted_score": 3.5,
                    "scenario_scores": [
                        {"criterion": "scenario_voltage", "score": 3}
                    ],
                },
            ],
            "risk_items": [
                {
                    "candidate_id": "sic",
                    "category": "cost",
                    "level": "medium",
                    "impact": "衬底成本较高",
                    "mitigation": "按子场景核算全生命周期成本",
                    "basis": "evidence",
                }
            ],
            "evidence_gaps": [],
            "recommendation_review": {
                "verdict": "reevaluate",
                "reasons": ["高优先级耐压要求未体现在推荐中"],
                "requested_queries": [],
            },
        }

        with patch.object(decision_module, "recommendation_chain", chain):
            result = decision_module.generate_recommendation(state)

        recommendation = result["recommendation"]
        self.assertEqual(recommendation["decision"], "conditional")
        self.assertEqual(recommendation["recommended_candidate_id"], "sic")
        self.assertEqual(recommendation["alternative_candidate_ids"], ["gan"])
        rationale = " ".join(recommendation["rationale"])
        self.assertIn("耐压能力", rationale)
        self.assertIn("3.25", rationale)
        self.assertIn("3.50", rationale)
        self.assertIn("衬底成本较高", rationale)
        self.assertNotIn(
            "CVD",
            " ".join(recommendation["unresolved_items"]),
        )
        feedback = json.loads(chain.invoke.call_args.args[0]["review_feedback"])
        self.assertEqual(
            feedback["reasons"],
            ["高优先级耐压要求未体现在推荐中"],
        )

    def test_review_marks_unresolved_gap_when_evidence_limit_is_reached(self):
        import src.workflows.nodes.selection_decision as decision_module

        chain = MagicMock()
        chain.invoke.return_value = AIMessage(
            content='{"verdict":"evidence_gap","reasons":["成本证据不足"],'
            '"requested_queries":["补充GAA成本证据"]}'
        )
        with patch.object(
            decision_module,
            "recommendation_review_chain",
            chain,
        ):
            result = decision_module.review_recommendation(
                {
                    "requirements": {},
                    "feasible_candidates": [
                        {"candidate_id": "gaa", "name": "GAA"},
                    ],
                    "evaluations": [
                        {"candidate_id": "gaa", "weighted_score": 4.0},
                    ],
                    "risk_items": [],
                    "recommendation": {
                        "decision": "recommend",
                        "recommended_candidate_id": "gaa",
                        "decision_basis": "weighted_score",
                    },
                    "evidence_records": [],
                    "evidence_gaps": [],
                    "evidence_round": 3,
                    "review_round": 0,
                    "warnings": [],
                }
            )

        self.assertEqual(result["status"], "needs_human_review")
        self.assertTrue(
            any("证据补检达到上限" in warning for warning in result["warnings"])
        )

    def test_generate_report_contains_fixed_sections_and_evidence_references(self):
        from src.workflows.nodes.selection_decision import generate_report

        result = generate_report(
            {
                "status": "running",
                "requirements": {
                    "business_goal": "降低功耗",
                    "application_scenario": "移动芯片",
                    "hard_constraints": [],
                },
                "feasible_candidates": [
                    {"candidate_id": "gaa", "name": "GAA", "summary": "方案"},
                ],
                "excluded_candidates": [],
                "evaluations": [
                    {
                        "candidate_id": "gaa",
                        "criterion_scores": [
                            {
                                "criterion": "power",
                                "score": 5,
                                "rationale": "目标负载下能效更优",
                                "evidence_ids": ["E1"],
                            }
                        ],
                        "weighted_score": None,
                    }
                ],
                "risk_items": [],
                "recommendation": {
                    "decision": "conditional",
                    "recommended_candidate_id": "gaa",
                    "alternative_candidate_ids": [],
                    "rationale": ["低功耗潜力"],
                    "prerequisites": ["验证良率"],
                    "unresolved_items": ["成本待验证"],
                },
                "evidence_records": [
                    {
                        "evidence_id": "E1",
                        "candidate_ids": ["gaa"],
                        "criteria": ["performance"],
                        "content": "evidence",
                        "source": "knowledge.md",
                        "url": "",
                        "title": "",
                        "retrieval_source": "vector",
                    }
                ],
                "warnings": [],
            }
        )

        self.assertEqual(result["status"], "completed")
        self.assertIn("# 半导体技术选型报告", result["report"])
        self.assertIn("## 6. 风险与应对措施", result["report"])
        self.assertIn("功耗与能效", result["report"])
        self.assertIn("| 5 |", result["report"])
        self.assertIn("[E1]", result["report"])


class TechnologySelectionRouteTest(unittest.TestCase):
    def test_routes_log_evidence_and_review_decisions(self):
        from src.observability.logging import log
        from src.workflows.routes.technology_selection import (
            route_evidence,
            route_recommendation_review,
        )

        messages = []
        sink_id = log.add(messages.append, format="{message}")
        self.addCleanup(log.remove, sink_id)

        self.assertEqual(
            route_evidence(
                {
                    "evidence_sufficient": False,
                    "evidence_round": 1,
                    "evidence_gaps": ["gap-1", "gap-2"],
                }
            ),
            "optimize_evidence_queries",
        )
        self.assertEqual(
            route_recommendation_review(
                {
                    "recommendation_review": {"verdict": "pass"},
                    "review_round": 1,
                }
            ),
            "generate_report",
        )
        rendered = "\n".join(str(message) for message in messages)

        self.assertIn("[技术选型][route_evidence]", rendered)
        self.assertIn("进入第2轮补检", rendered)
        self.assertIn("缺口=2", rendered)
        self.assertIn("[技术选型][route_recommendation_review]", rendered)
        self.assertIn("审查通过，进入报告生成", rendered)

    def test_routes_read_state_without_calling_models(self):
        from langgraph.constants import END

        from src.workflows.routes.technology_selection import (
            route_evidence,
            route_feasible_candidates,
            route_recommendation_review,
            route_retrieval_result,
            route_requirements,
        )

        self.assertEqual(
            route_requirements({"requirements_complete": False}),
            "clarify_requirements",
        )
        self.assertEqual(
            route_requirements({"requirements_complete": True}),
            "generate_candidates",
        )
        self.assertEqual(
            route_feasible_candidates({"feasible_candidates": []}),
            "clarify_requirements",
        )
        self.assertEqual(
            route_feasible_candidates(
                {"feasible_candidates": [{"candidate_id": "gaa"}]}
            ),
            "plan_evidence_tasks",
        )
        self.assertEqual(
            route_evidence(
                {"evidence_sufficient": False, "evidence_round": 1}
            ),
            "optimize_evidence_queries",
        )
        self.assertEqual(
            route_evidence(
                {"evidence_sufficient": False, "evidence_round": 2}
            ),
            "optimize_evidence_queries",
        )
        self.assertEqual(
            route_evidence(
                {"evidence_sufficient": False, "evidence_round": 3}
            ),
            "evaluate_candidates",
        )
        self.assertEqual(
            route_retrieval_result(
                {
                    "documents": [],
                    "transform_count": 1,
                    "max_transform_count": 1,
                    "allow_web_fallback": False,
                }
            ),
            END,
        )
        self.assertEqual(
            route_retrieval_result(
                {
                    "documents": [],
                    "transform_count": 1,
                    "max_transform_count": 1,
                    "allow_web_fallback": True,
                }
            ),
            "web_search",
        )
        self.assertEqual(
            route_recommendation_review(
                {
                    "recommendation_review": {"verdict": "evidence_gap"},
                    "review_round": 1,
                }
            ),
            "optimize_evidence_queries",
        )
        self.assertEqual(
            route_recommendation_review(
                {
                    "recommendation_review": {"verdict": "evidence_gap"},
                    "review_round": 1,
                    "evidence_round": 3,
                }
            ),
            "generate_report",
        )
        self.assertEqual(
            route_recommendation_review(
                {
                    "recommendation_review": {"verdict": "reevaluate"},
                    "review_round": 1,
                }
            ),
            "evaluate_candidates",
        )
        self.assertEqual(
            route_recommendation_review(
                {
                    "recommendation_review": {"verdict": "evidence_gap"},
                    "review_round": 2,
                }
            ),
            "generate_report",
        )


class TechnologySelectionGraphStructureTest(unittest.TestCase):
    def test_compiled_graph_contains_only_planned_outer_nodes(self):
        from src.workflows.technology_selection_graph import (
            technology_selection_graph,
        )

        graph_view = technology_selection_graph.get_graph()

        self.assertEqual(
            set(graph_view.nodes),
            {
                "__start__",
                "parse_requirements",
                "clarify_requirements",
                "generate_candidates",
                "screen_hard_constraints",
                "plan_evidence_tasks",
                "retrieve_evidence",
                "optimize_evidence_queries",
                "evaluate_candidates",
                "analyze_risks",
                "generate_recommendation",
                "review_recommendation",
                "generate_report",
                "__end__",
            },
        )
        edges = {
            (edge.source, edge.target, edge.conditional)
            for edge in graph_view.edges
        }
        expected_edges = {
            ("__start__", "parse_requirements", False),
            ("parse_requirements", "clarify_requirements", True),
            ("parse_requirements", "generate_candidates", True),
            ("generate_candidates", "screen_hard_constraints", False),
            ("screen_hard_constraints", "plan_evidence_tasks", True),
            ("plan_evidence_tasks", "retrieve_evidence", False),
            ("retrieve_evidence", "optimize_evidence_queries", True),
            ("retrieve_evidence", "evaluate_candidates", True),
            ("evaluate_candidates", "analyze_risks", False),
            ("analyze_risks", "generate_recommendation", False),
            ("generate_recommendation", "review_recommendation", False),
            ("review_recommendation", "optimize_evidence_queries", True),
            ("review_recommendation", "generate_report", True),
            ("generate_report", "__end__", False),
            ("clarify_requirements", "parse_requirements", False),
        }
        self.assertTrue(expected_edges.issubset(edges), edges)


class TechnologySelectionEndToEndTest(unittest.TestCase):
    def test_incomplete_request_returns_clarification_and_stops(self):
        import src.workflows.nodes.selection_requirements as requirements_module
        from src.workflows.technology_selection_graph import (
            technology_selection_graph,
        )

        chain = MagicMock()
        chain.invoke.return_value = AIMessage(
            content="""{
                "business_goal": "降低功耗",
                "application_scenario": "",
                "priorities": {},
                "hard_constraints": [],
                "missing_information": ["application_scenario"]
            }"""
        )
        with patch.object(
            requirements_module,
            "requirement_analysis_chain",
            chain,
        ):
            result = technology_selection_graph.invoke(
                {"user_request": "我想降低功耗"}
            )

        self.assertIn("__interrupt__", result)
        payload = result["__interrupt__"][0].value
        self.assertEqual(payload["status"], "awaiting_user")
        self.assertTrue(payload["clarification_questions"])
        self.assertNotIn("report", result)

    def test_complete_request_produces_source_linked_report_with_mocks(self):
        from src.observability.logging import log
        import src.workflows.nodes.selection_decision as decision_module
        import src.workflows.nodes.selection_evidence as evidence_module
        import src.workflows.nodes.selection_requirements as requirements_module
        from src.workflows.technology_selection_graph import (
            technology_selection_graph,
        )

        requirement_chain = MagicMock()
        requirement_chain.invoke.return_value = AIMessage(
            content="""{
                "business_goal": "降低功耗",
                "application_scenario": "移动芯片",
                "priorities": {},
                "hard_constraints": [],
                "requested_candidates": ["FinFET", "GAA"],
                "missing_information": []
            }"""
        )
        evidence_document = Document(
            page_content=(
                "FinFET and GAA performance power thermal cost maturity risk "
                "scalability comparison evidence"
            ),
            metadata={
                "id": "comparison-doc",
                "source": "knowledge.md",
                "retrieval_sources": ["vector"],
            },
        )
        evidence_id = evidence_module._evidence_id(evidence_document)
        fake_retrieval_graph = MagicMock()
        fake_retrieval_graph.invoke.return_value = {
            "documents": [evidence_document],
            "retrieval_source": "local_hybrid",
        }
        finfet_score_items = ",".join(
            (
                '{"criterion":"'
                + criterion
                + '","score":4,"rationale":"supported",'
                + f'"evidence_ids":["{evidence_id}"],"confidence":"high"}}'
            )
            for criterion in (
                "performance",
                "power",
                "thermal",
                "cost",
                "maturity",
                "risk",
                "scalability",
            )
        )
        gaa_score_items = finfet_score_items.replace('"score":4', '"score":5')
        evaluation_chain = MagicMock()
        evaluation_chain.invoke.return_value = AIMessage(
            content=(
                '{"evaluations":['
                '{"candidate_id":"finfet","criterion_scores":['
                + finfet_score_items
                + "]},"
                '{"candidate_id":"gaa","criterion_scores":['
                + gaa_score_items
                + "]}]}"
            )
        )
        risk_chain = MagicMock()
        risk_chain.invoke.return_value = AIMessage(
            content=(
                '{"risk_items":[{"candidate_id":"gaa",'
                '"category":"maturity","level":"medium",'
                '"impact":"量产验证不足","mitigation":"分阶段验证",'
                f'"evidence_ids":["{evidence_id}"],"basis":"evidence"}}]}}'
            )
        )
        recommendation_chain = MagicMock()
        recommendation_chain.invoke.return_value = AIMessage(
            content="""{
                "decision": "recommend",
                "recommended_candidate_id": "gaa",
                "alternative_candidate_ids": ["finfet"],
                "rationale": ["满足低功耗目标"],
                "prerequisites": ["完成量产验证"],
                "unresolved_items": []
            }"""
        )
        review_chain = MagicMock()
        review_chain.invoke.return_value = AIMessage(
            content='{"verdict":"pass","reasons":[],"requested_queries":[]}'
        )
        assessment_chain = MagicMock()
        assessment_chain.invoke.return_value = AIMessage(
            content=json.dumps(
                {
                    "evidence_map": [
                        {
                            "evidence_id": evidence_id,
                            "supports": [
                                {
                                    "candidate_id": candidate_id,
                                    "criterion": criterion,
                                    "level": "direct",
                                    "quote": criterion,
                                }
                                for candidate_id in ("finfet", "gaa")
                                for criterion in evidence_module.CRITERIA
                            ],
                        }
                    ],
                    "missing_queries": [],
                }
            )
        )
        source_chain = MagicMock()
        source_chain.invoke.return_value = AIMessage(
            content=json.dumps(
                {
                    "task_sources": [
                        {
                            "task_id": f"criterion-{criterion}",
                            "source": "local",
                        }
                        for criterion in (
                            "performance",
                            "power",
                            "thermal",
                            "cost",
                            "maturity",
                            "risk",
                            "scalability",
                        )
                    ]
                }
            )
        )
        messages = []
        sink_id = log.add(messages.append, format="{message}")
        self.addCleanup(log.remove, sink_id)

        with patch.object(
            requirements_module,
            "requirement_analysis_chain",
            requirement_chain,
        ), patch.object(
            evidence_module,
            "retrieval_graph",
            fake_retrieval_graph,
        ), patch.object(
            evidence_module,
            "evidence_assessment_chain",
            assessment_chain,
        ), patch.object(
            evidence_module,
            "evidence_source_planning_chain",
            source_chain,
        ), patch.object(
            decision_module,
            "candidate_evaluation_chain",
            evaluation_chain,
        ), patch.object(
            decision_module,
            "risk_analysis_chain",
            risk_chain,
        ), patch.object(
            decision_module,
            "recommendation_chain",
            recommendation_chain,
        ), patch.object(
            decision_module,
            "recommendation_review_chain",
            review_chain,
        ):
            result = technology_selection_graph.invoke(
                {
                    "user_request": "为移动芯片比较低功耗FinFET和GAA器件架构",
                    "clarification_answers": [],
                }
            )

        self.assertEqual(result["status"], "completed")
        self.assertEqual(
            result["recommendation"]["recommended_candidate_id"],
            "gaa",
        )
        self.assertIn(f"[{evidence_id}]", result["report"])
        self.assertEqual(fake_retrieval_graph.invoke.call_count, 2)
        rendered = "\n".join(str(message) for message in messages)
        for node_name in (
            "parse_requirements",
            "generate_candidates",
            "screen_hard_constraints",
            "plan_evidence_tasks",
            "retrieve_evidence",
            "evaluate_candidates",
            "analyze_risks",
            "generate_recommendation",
            "review_recommendation",
            "generate_report",
        ):
            self.assertIn(f"[技术选型][{node_name}] 开始", rendered)
            self.assertIn(f"[技术选型][{node_name}] 完成", rendered)
        self.assertIn(
            "[技术选型][source_planning] 开始：确定证据包来源",
            rendered,
        )
        self.assertIn("自动/本地=1", rendered)
        self.assertIn("联网=1", rendered)
        self.assertIn(
            "[技术选型][evidence_assessment] 开始：评估证据归属",
            rendered,
        )
        self.assertIn("执行轮次=1/3", rendered)
        self.assertIn("并发上限=2", rendered)
        self.assertIn(
            "[证据任务][initial-technical] 开始：检索证据",
            rendered,
        )


class TechnologySelectionInterruptTest(unittest.TestCase):
    """Checkpointed technology-selection session with LangGraph interrupts."""

    def _incomplete_payload(self, user_request, clarification_answers):
        return {
            "business_goal": "降低功耗",
            "application_scenario": "",
            "priorities": {},
            "hard_constraints": [],
            "requested_candidates": ["FinFET", "GAA"],
            "scenario_criteria": [],
            "missing_information": ["application_scenario"],
        }

    def _complete_payload(self, user_request, clarification_answers):
        return {
            "business_goal": "降低功耗",
            "application_scenario": "移动芯片",
            "priorities": {},
            "hard_constraints": [],
            "requested_candidates": ["FinFET", "GAA"],
            "scenario_criteria": [],
            "missing_information": [],
        }

    def _parse_chain(self, complete_payload=True):
        import src.workflows.nodes.selection_requirements as requirements_module
        from src.workflows.selection_cache import clear_selection_cache

        clear_selection_cache()
        chain = MagicMock()
        chain.invoke.side_effect = lambda payload: AIMessage(
            content=json.dumps(
                (
                    self._complete_payload(
                        payload.get("user_request", ""),
                        payload.get("clarification_answers", []),
                    )
                    if complete_payload
                    else self._incomplete_payload(
                        payload.get("user_request", ""),
                        payload.get("clarification_answers", []),
                    )
                ),
                ensure_ascii=False,
            )
        )
        return requirements_module, chain

    def test_first_incomplete_request_produces_interrupt_and_preserves_user_request(self):
        import src.workflows.nodes.selection_requirements as requirements_module
        from src.infrastructure.checkpointing.sqlite import open_sqlite_checkpointer
        from src.workflows.technology_selection_graph import (
            build_technology_selection_graph,
        )
        from pathlib import Path
        from tempfile import TemporaryDirectory

        _, chain = self._parse_chain(complete_payload=False)
        with TemporaryDirectory(dir=Path.cwd()) as directory:
            database_path = Path(directory) / "cp.sqlite3"
            with open_sqlite_checkpointer(database_path) as checkpointer:
                graph = build_technology_selection_graph(checkpointer)
                config = {"configurable": {"thread_id": "interrupt-t1"}}
                with patch.object(
                    requirements_module,
                    "requirement_analysis_chain",
                    chain,
                ):
                    result = graph.invoke(
                        {"user_request": "我想降低功耗"},
                        config=config,
                    )
                    self.assertEqual(
                        graph.get_state(config).values["user_request"],
                        "我想降低功耗",
                    )

        self.assertIn("__interrupt__", result)
        payload = result["__interrupt__"][0].value
        self.assertEqual(payload["status"], "awaiting_user")
        self.assertTrue(payload["clarification_questions"])
        self.assertLessEqual(len(payload["clarification_questions"]), 3)

    def test_resume_reparses_with_original_request_and_latest_answer(self):
        import src.workflows.nodes.selection_requirements as requirements_module
        from src.infrastructure.checkpointing.sqlite import open_sqlite_checkpointer
        from src.workflows.technology_selection_graph import (
            build_technology_selection_graph,
        )
        from langgraph.types import Command
        from pathlib import Path
        from tempfile import TemporaryDirectory

        _, chain = self._parse_chain(complete_payload=False)
        with TemporaryDirectory(dir=Path.cwd()) as directory:
            database_path = Path(directory) / "cp.sqlite3"
            with open_sqlite_checkpointer(database_path) as checkpointer:
                graph = build_technology_selection_graph(checkpointer)
                config = {"configurable": {"thread_id": "interrupt-t2"}}
                with patch.object(
                    requirements_module,
                    "requirement_analysis_chain",
                    chain,
                ):
                    graph.invoke(
                        {"user_request": "为移动芯片比较FinFET和GAA"},
                        config=config,
                    )
                    resumed = graph.invoke(
                        Command(resume="应用于手机SoC，优先功耗"),
                        config=config,
                    )

        self.assertIn("__interrupt__", resumed)
        reparsed = chain.invoke.call_args_list[1].args[0]
        self.assertEqual(reparsed["user_request"], "为移动芯片比较FinFET和GAA")
        self.assertIn("应用于手机SoC，优先功耗", reparsed["clarification_answers"])

    def test_answers_and_history_are_appended_exactly_once(self):
        import src.workflows.nodes.selection_requirements as requirements_module
        from src.infrastructure.checkpointing.sqlite import open_sqlite_checkpointer
        from src.workflows.technology_selection_graph import (
            build_technology_selection_graph,
        )
        from langgraph.types import Command
        from pathlib import Path
        from tempfile import TemporaryDirectory

        _, chain = self._parse_chain(complete_payload=False)
        with TemporaryDirectory(dir=Path.cwd()) as directory:
            database_path = Path(directory) / "cp.sqlite3"
            with open_sqlite_checkpointer(database_path) as checkpointer:
                graph = build_technology_selection_graph(checkpointer)
                config = {"configurable": {"thread_id": "interrupt-t3"}}
                with patch.object(
                    requirements_module,
                    "requirement_analysis_chain",
                    chain,
                ):
                    graph.invoke(
                        {"user_request": "比较FinFET和GAA"},
                        config=config,
                    )
                    graph.invoke(
                        Command(resume="应用于手机SoC"),
                        config=config,
                    )
                    values = graph.get_state(config).values

        self.assertEqual(values["clarification_answers"], ["应用于手机SoC"])
        self.assertEqual(len(values["clarification_history"]), 1)
        self.assertEqual(
            values["clarification_history"][0]["answer"],
            "应用于手机SoC",
        )
        self.assertTrue(values["clarification_history"][0]["questions"])

    def test_incomplete_answer_enters_interrupt_again_without_duplication(self):
        import src.workflows.nodes.selection_requirements as requirements_module
        from src.infrastructure.checkpointing.sqlite import open_sqlite_checkpointer
        from src.workflows.technology_selection_graph import (
            build_technology_selection_graph,
        )
        from langgraph.types import Command
        from pathlib import Path
        from tempfile import TemporaryDirectory

        _, chain = self._parse_chain(complete_payload=False)
        with TemporaryDirectory(dir=Path.cwd()) as directory:
            database_path = Path(directory) / "cp.sqlite3"
            with open_sqlite_checkpointer(database_path) as checkpointer:
                graph = build_technology_selection_graph(checkpointer)
                config = {"configurable": {"thread_id": "interrupt-t4"}}
                with patch.object(
                    requirements_module,
                    "requirement_analysis_chain",
                    chain,
                ):
                    graph.invoke(
                        {"user_request": "比较FinFET和GAA"},
                        config=config,
                    )
                    resumed = graph.invoke(
                        Command(resume="回答仍不完整"),
                        config=config,
                    )
                    values = graph.get_state(config).values

        self.assertIn("__interrupt__", resumed)
        self.assertEqual(values["clarification_answers"], ["回答仍不完整"])
        self.assertEqual(len(values["clarification_history"]), 1)


if __name__ == "__main__":
    unittest.main()

import json
import io
import subprocess
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document
from langchain_core.messages import AIMessage


class QueryClassificationTest(unittest.TestCase):
    def test_common_direct_requests_skip_the_classification_model(self):
        import src.workflows.nodes.query_router as router_module

        cases = (
            ("你是谁？", "identity"),
            ("你好", "greeting"),
            ("你能做什么？", "capabilities"),
        )
        for query, expected_kind in cases:
            with self.subTest(query=query):
                chain = MagicMock()
                with patch.object(
                    router_module,
                    "query_classification_chain",
                    chain,
                ):
                    result = router_module.classify_query({"query": query})

                self.assertEqual(result["query_type"], "direct_response")
                self.assertEqual(result["source_preference"], "none")
                self.assertEqual(result["response_type"], "answer")
                self.assertEqual(
                    result["direct_response_kind"],
                    expected_kind,
                )
                self.assertEqual(result["status"], "running")
                chain.invoke.assert_not_called()

    def test_contextual_semiconductor_comparison_skips_classification_model(self):
        import src.workflows.nodes.query_router as router_module

        chain = MagicMock()

        with patch.object(router_module, "query_classification_chain", chain):
            result = router_module.classify_query(
                {"query": "面向低功耗手机SoC，请比较FinFET和GAA并推荐技术路线"}
            )

        self.assertEqual(result["query_type"], "technology_selection")
        self.assertEqual(result["response_type"], "report")
        chain.invoke.assert_not_called()

    def test_explanatory_semiconductor_comparison_uses_rag_classification(self):
        import src.workflows.nodes.query_router as router_module

        chain = MagicMock()
        chain.invoke.return_value = AIMessage(
            content=(
                '{"query_type":"rag_qa",'
                '"source_preference":"auto","response_type":"answer",'
                '"reason":"解释两种器件架构的工作原理差异"}'
            )
        )

        with patch.object(router_module, "query_classification_chain", chain):
            result = router_module.classify_query(
                {"query": "对比FinFET和GAA的工作原理"}
            )

        self.assertEqual(result["query_type"], "rag_qa")
        self.assertEqual(result["source_preference"], "auto")
        self.assertEqual(result["response_type"], "answer")
        chain.invoke.assert_called_once()

    def test_non_semiconductor_comparison_still_uses_classification_model(self):
        import src.workflows.nodes.query_router as router_module

        chain = MagicMock()
        chain.invoke.return_value = AIMessage(
            content=(
                '{"query_type":"direct_response",'
                '"source_preference":"none","response_type":"answer",'
                '"direct_response_kind":"out_of_scope",'
                '"reason":"非半导体比较"}'
            )
        )
        with patch.object(router_module, "query_classification_chain", chain):
            result = router_module.classify_query(
                {"query": "比较两款旅游背包"}
            )

        self.assertEqual(result["query_type"], "direct_response")
        chain.invoke.assert_called_once()

    def test_ambiguous_packaging_process_does_not_fast_route_to_selection(self):
        import src.workflows.nodes.query_router as router_module

        chain = MagicMock()
        chain.invoke.return_value = AIMessage(
            content=(
                '{"query_type":"direct_response",'
                '"source_preference":"none","response_type":"answer",'
                '"direct_response_kind":"out_of_scope",'
                '"reason":"非半导体包装问题"}'
            )
        )

        with patch.object(router_module, "query_classification_chain", chain):
            result = router_module.classify_query(
                {"query": "面向礼品运输，推荐一种封装工艺"}
            )

        self.assertEqual(result["query_type"], "direct_response")
        self.assertEqual(result["direct_response_kind"], "out_of_scope")
        chain.invoke.assert_called_once()

    def test_out_of_scope_decision_is_validated_as_a_direct_response(self):
        import src.workflows.nodes.query_router as router_module

        chain = MagicMock()
        chain.invoke.return_value = AIMessage(
            content=(
                '{"query_type":"direct_response",'
                '"source_preference":"none","response_type":"answer",'
                '"direct_response_kind":"out_of_scope",'
                '"reason":"非半导体问题"}'
            )
        )

        with patch.object(router_module, "query_classification_chain", chain):
            result = router_module.classify_query({"query": "今天天气怎么样"})

        self.assertEqual(result["query_type"], "direct_response")
        self.assertEqual(result["source_preference"], "none")
        self.assertEqual(result["direct_response_kind"], "out_of_scope")
        self.assertEqual(result["status"], "running")

    def test_time_sensitive_rag_question_overrides_model_auto_source_to_web(self):
        import src.workflows.nodes.query_router as router_module

        chain = MagicMock()
        chain.invoke.return_value = AIMessage(
            content=(
                '{"query_type":"rag_qa",'
                '"source_preference":"auto","response_type":"answer",'
                '"reason":"半导体技术问题"}'
            )
        )

        with patch.object(router_module, "query_classification_chain", chain):
            result = router_module.classify_query(
                {"query": "目前GAA技术的量产进展如何？"}
            )

        self.assertEqual(result["query_type"], "rag_qa")
        self.assertEqual(result["source_preference"], "web")

    def test_current_stable_technical_question_does_not_force_web(self):
        import src.workflows.nodes.query_router as router_module

        chain = MagicMock()
        chain.invoke.return_value = AIMessage(
            content=(
                '{"query_type":"rag_qa",'
                '"source_preference":"auto","response_type":"answer",'
                '"reason":"半导体稳定技术知识"}'
            )
        )

        with patch.object(router_module, "query_classification_chain", chain):
            result = router_module.classify_query(
                {"query": "目前EUV有哪些主要技术难点？"}
            )

        self.assertEqual(result["source_preference"], "auto")

    def test_historical_progress_question_does_not_force_web(self):
        import src.workflows.nodes.query_router as router_module

        chain = MagicMock()
        chain.invoke.return_value = AIMessage(
            content=(
                '{"query_type":"rag_qa",'
                '"source_preference":"auto","response_type":"answer",'
                '"reason":"半导体技术发展历史"}'
            )
        )

        with patch.object(router_module, "query_classification_chain", chain):
            result = router_module.classify_query(
                {"query": "EUV技术进展的历史"}
            )

        self.assertEqual(result["source_preference"], "auto")

    def test_classification_prompt_separates_knowledge_reports_from_selection(self):
        import src.workflows.nodes.query_router as router_module

        system_prompt = router_module.query_classification_prompt.format_messages(
            query="生成一份EUV工作原理报告"
        )[0].content

        self.assertIn("知识型报告", system_prompt)
        self.assertIn("报告形式本身不代表技术选型", system_prompt)

    def test_direct_response_without_a_kind_is_a_controlled_error(self):
        import src.workflows.nodes.query_router as router_module

        chain = MagicMock()
        chain.invoke.return_value = AIMessage(
            content=(
                '{"query_type":"direct_response",'
                '"source_preference":"none","response_type":"answer",'
                '"reason":"缺少直接响应类型"}'
            )
        )

        with patch.object(
            router_module,
            "query_classification_chain",
            chain,
        ), patch.object(router_module.log, "exception"):
            result = router_module.classify_query({"query": "一个越界问题"})

        self.assertIsNone(result["query_type"])
        self.assertEqual(result["status"], "error")
        self.assertTrue(result["warnings"])

    def test_classify_query_returns_validated_routing_decision(self):
        import src.workflows.nodes.query_router as router_module

        chain = MagicMock()
        chain.invoke.return_value = AIMessage(
            content=(
                '{"query_type":"technology_selection",'
                '"source_preference":"auto","response_type":"report",'
                '"reason":"用户要求比较半导体方案"}'
            )
        )

        with patch.object(router_module, "query_classification_chain", chain):
            result = router_module.classify_query(
                {"query": "我需要处理一个半导体技术决策问题"}
            )

        self.assertEqual(
            result,
            {
                "query_type": "technology_selection",
                "source_preference": "auto",
                "response_type": "report",
                "routing_reason": "用户要求比较半导体方案",
                "report_format": "markdown",
                "status": "running",
                "warnings": [],
            },
        )

    def test_classify_query_returns_friendly_error_for_invalid_response(self):
        import src.workflows.nodes.query_router as router_module

        chain = MagicMock()
        chain.invoke.return_value = AIMessage(content="not json")

        with patch.object(
            router_module,
            "query_classification_chain",
            chain,
        ), patch.object(router_module.log, "exception"):
            result = router_module.classify_query({"query": "什么是GAA"})

        self.assertIsNone(result["query_type"])
        self.assertEqual(result["status"], "error")
        self.assertTrue(result["warnings"])

    def test_classify_query_accepts_an_extra_closing_brace(self):
        import src.workflows.nodes.query_router as router_module

        chain = MagicMock()
        chain.invoke.return_value = AIMessage(
            content=(
                '{"query_type":"rag_qa","source_preference":"auto",'
                '"response_type":"answer","reason":"半导体稳定知识"}}'
            )
        )

        with patch.object(router_module, "query_classification_chain", chain):
            result = router_module.classify_query({"query": "请解释芯片封装"})

        self.assertEqual(result["query_type"], "rag_qa")
        self.assertEqual(result["source_preference"], "auto")
        self.assertEqual(result["response_type"], "answer")
        self.assertEqual(result["status"], "running")
        self.assertEqual(result["warnings"], [])

    def test_classify_query_rejects_an_invalid_report_format_without_llm_call(self):
        import src.workflows.nodes.query_router as router_module

        chain = MagicMock()
        with patch.object(router_module, "query_classification_chain", chain):
            result = router_module.classify_query(
                {"query": "生成报告", "report_format": "pdf"}
            )

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["report_format"], "markdown")
        self.assertTrue(result["warnings"])
        chain.invoke.assert_not_called()


class QueryRouteTest(unittest.TestCase):
    def test_route_by_query_type_selects_only_supported_destinations(self):
        from src.observability.logging import log
        from src.workflows.routes.query_router import route_by_query_type

        messages = []
        sink_id = log.add(messages.append, format="{message}")
        self.addCleanup(log.remove, sink_id)

        self.assertEqual(
            route_by_query_type(
                {"query_type": "rag_qa", "status": "running"}
            ),
            "invoke_rag_qa",
        )
        self.assertEqual(
            route_by_query_type(
                {
                    "query_type": "technology_selection",
                    "status": "running",
                }
            ),
            "invoke_technology_selection",
        )
        self.assertEqual(
            route_by_query_type(
                {"query_type": "direct_response", "status": "running"}
            ),
            "respond_directly",
        )
        self.assertEqual(
            route_by_query_type({"query_type": None, "status": "error"}),
            "build_query_response",
        )
        rendered = "\n".join(str(message) for message in messages)
        self.assertIn(
            "[入口][route_by_query_type] 路由：进入普通问答工作流",
            rendered,
        )
        self.assertIn(
            "[入口][route_by_query_type] 路由：进入技术选型工作流",
            rendered,
        )
        self.assertIn(
            "[入口][route_by_query_type] 路由：进入直接响应",
            rendered,
        )
        self.assertIn(
            "[入口][route_by_query_type] 路由：分类失败，整理错误响应",
            rendered,
        )


class QueryDirectResponseTest(unittest.TestCase):
    def test_direct_response_uses_fixed_templates(self):
        from src.workflows.nodes.query_router import respond_directly

        expected_fragments = {
            "identity": "半导体知识问答和技术选型",
            "greeting": "你好",
            "capabilities": "技术选型报告",
            "out_of_scope": "超出了当前半导体技术顾问的服务范围",
        }
        for kind, expected_fragment in expected_fragments.items():
            with self.subTest(kind=kind):
                result = respond_directly(
                    {
                        "query": "private user content",
                        "direct_response_kind": kind,
                    }
                )

                self.assertEqual(result["status"], "completed")
                self.assertIn(expected_fragment, result["direct_answer"])
                self.assertNotIn("private user content", result["direct_answer"])


class QuerySubgraphInvocationTest(unittest.TestCase):
    def test_invoke_rag_qa_maps_the_unified_input(self):
        import src.workflows.nodes.query_router as router_module

        child_graph = MagicMock()
        child_graph.invoke.return_value = {
            "generation": "联网报告",
            "documents": [],
            "retrieval_source": "web_search",
        }

        with patch.object(
            router_module,
            "rag_qa_graph",
            child_graph,
        ):
            result = router_module.invoke_rag_qa(
                {
                    "query": "生成新能源汽车行业报告",
                    "source_preference": "web",
                    "response_type": "report",
                }
            )

        child_graph.invoke.assert_called_once_with(
            {
                "question": "生成新能源汽车行业报告",
                "original_question": "生成新能源汽车行业报告",
                "source_preference": "web",
                "response_type": "report",
                "transform_count": 0,
                "generation_attempt_count": 0,
                "web_search_attempt_count": 0,
                "web_search_queries": [],
            }
        )
        self.assertEqual(result["child_graph_result"]["generation"], "联网报告")

    def test_invoke_technology_selection_maps_the_unified_input(self):
        import src.workflows.nodes.query_router as router_module

        child_graph = MagicMock()
        child_graph.invoke.return_value = {
            "status": "completed",
            "report": "# 半导体技术选型报告",
        }

        with patch.object(
            router_module,
            "technology_selection_graph",
            child_graph,
        ):
            result = router_module.invoke_technology_selection(
                {"query": "为移动芯片比较GAA和FinFET"}
            )

        child_graph.invoke.assert_called_once_with(
            {
                "user_request": "为移动芯片比较GAA和FinFET",
                "clarification_answers": [],
            }
        )
        self.assertEqual(
            result["child_graph_result"]["report"],
            "# 半导体技术选型报告",
        )

    def test_subgraph_failure_becomes_a_friendly_error_state(self):
        import src.workflows.nodes.query_router as router_module

        child_graph = MagicMock()
        child_graph.invoke.side_effect = RuntimeError("secret provider error")

        with patch.object(
            router_module,
            "rag_qa_graph",
            child_graph,
        ), patch.object(router_module.log, "exception"):
            result = router_module.invoke_rag_qa(
                {
                    "query": "解释一个问题",
                    "source_preference": "auto",
                    "response_type": "answer",
                }
            )

        self.assertEqual(result["status"], "error")
        self.assertTrue(result["warnings"])
        self.assertNotIn("secret provider error", result["warnings"][0])

    def test_selection_subgraph_failure_becomes_a_friendly_error_state(self):
        import src.workflows.nodes.query_router as router_module

        child_graph = MagicMock()
        child_graph.invoke.side_effect = RuntimeError("secret provider error")

        with patch.object(
            router_module,
            "technology_selection_graph",
            child_graph,
        ), patch.object(router_module.log, "exception"):
            result = router_module.invoke_technology_selection(
                {"query": "比较GAA和FinFET"}
            )

        self.assertEqual(result["status"], "error")
        self.assertTrue(result["warnings"])
        self.assertNotIn("secret provider error", result["warnings"][0])


class QueryResponseTest(unittest.TestCase):
    def test_direct_response_is_normalized_without_sources_or_docx(self):
        import src.workflows.nodes.query_router as router_module

        with patch(
            "src.infrastructure.reporting.docx.export_report_docx"
        ) as exporter:
            result = router_module.build_query_response(
                {
                    "query": "你是谁",
                    "query_type": "direct_response",
                    "source_preference": "none",
                    "response_type": "answer",
                    "direct_response_kind": "identity",
                    "direct_answer": "我是半导体技术顾问。",
                    "report_format": "docx",
                    "routing_reason": "身份询问",
                    "status": "completed",
                    "warnings": [],
                }
            )

        self.assertEqual(result["query_type"], "direct_response")
        self.assertEqual(result["source_preference"], "none")
        self.assertEqual(result["answer"], "我是半导体技术顾问。")
        self.assertEqual(result["report"], "")
        self.assertEqual(result["sources"], [])
        self.assertEqual(
            result["details"],
            {"direct_response_kind": "identity"},
        )
        self.assertIsNone(result["report_path"])
        exporter.assert_not_called()

    def test_markdown_is_the_default_and_does_not_write_a_file(self):
        import src.workflows.nodes.query_router as router_module

        with patch(
            "src.infrastructure.reporting.docx.export_report_docx"
        ) as exporter:
            result = router_module.build_query_response(
                {
                    "query": "生成报告",
                    "query_type": "rag_qa",
                    "source_preference": "web",
                    "response_type": "report",
                    "status": "running",
                    "child_graph_result": {"generation": "# 报告"},
                }
            )

        self.assertEqual(result["report_format"], "markdown")
        self.assertIsNone(result["report_path"])
        exporter.assert_not_called()

    def test_selection_response_exposes_structured_report_data(self):
        import src.workflows.nodes.query_router as router_module

        report_data = {
            "schema_version": "1.0",
            "report_version": "2.0",
            "decision": {"status": "conditional"},
        }
        result = router_module.build_query_response(
            {
                "query": "比较SiC和GaN",
                "query_type": "technology_selection",
                "source_preference": "auto",
                "response_type": "report",
                "status": "running",
                "child_graph_result": {
                    "status": "completed",
                    "report": "# 半导体技术选型报告",
                    "report_data": report_data,
                    "evidence_records": [
                        {
                            "evidence_id": "E1",
                            "content": "不应在公共details中返回的完整证据正文",
                            "source": r"D:\\private\\evidence.md",
                        }
                    ],
                },
            }
        )

        self.assertEqual(result["report_data"], report_data)
        self.assertNotIn("evidence_records", result["details"])

    def test_docx_request_exports_the_normalized_report(self):
        import src.workflows.nodes.query_router as router_module

        with patch(
            "src.infrastructure.reporting.docx.export_report_docx",
            return_value="D:/reports/report.docx",
        ) as exporter:
            result = router_module.build_query_response(
                {
                    "query": "生成技术选型报告",
                    "query_type": "technology_selection",
                    "source_preference": "auto",
                    "response_type": "report",
                    "report_format": "docx",
                    "status": "running",
                    "child_graph_result": {
                        "status": "completed",
                        "report": "# 技术选型报告\n\n## 功耗评价",
                    },
                }
            )

        exporter.assert_called_once_with(
            "# 技术选型报告\n\n## 功耗评价",
            "生成技术选型报告",
        )
        self.assertEqual(result["report_format"], "docx")
        self.assertEqual(result["report_path"], "D:/reports/report.docx")

    def test_docx_export_failure_preserves_markdown_and_adds_warning(self):
        import src.workflows.nodes.query_router as router_module

        with patch(
            "src.infrastructure.reporting.docx.export_report_docx",
            side_effect=OSError("disk unavailable"),
        ):
            result = router_module.build_query_response(
                {
                    "query": "生成技术选型报告",
                    "query_type": "technology_selection",
                    "source_preference": "auto",
                    "response_type": "report",
                    "report_format": "docx",
                    "status": "running",
                    "child_graph_result": {
                        "status": "completed",
                        "report": "# 技术选型报告",
                    },
                }
            )

        self.assertEqual(result["report"], "# 技术选型报告")
        self.assertIsNone(result["report_path"])
        self.assertTrue(any("DOCX" in item for item in result["warnings"]))

    def test_rag_answer_is_normalized_and_sources_are_deduplicated(self):
        from src.workflows.nodes.query_router import build_query_response

        document = Document(
            page_content="evidence",
            metadata={
                "title": "Source A",
                "source": "https://example.com/a",
                "url": "https://example.com/a",
                "retrieval_source": "web_search",
            },
        )
        result = build_query_response(
            {
                "query": "解释问题",
                "query_type": "rag_qa",
                "source_preference": "web",
                "response_type": "answer",
                "routing_reason": "非半导体问题",
                "status": "running",
                "warnings": [],
                "child_graph_result": {
                    "generation": "普通回答",
                    "documents": [document, document],
                    "retrieval_source": "web_search",
                    "transform_count": 0,
                    "generation_attempt_count": 1,
                    "graph_error": None,
                },
            }
        )

        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["answer"], "普通回答")
        self.assertEqual(result["report"], "")
        self.assertEqual(len(result["sources"]), 1)
        self.assertEqual(result["sources"][0]["url"], "https://example.com/a")
        json.dumps(result, ensure_ascii=False)

    def test_rag_retrieval_error_is_not_reported_as_completed(self):
        from src.workflows.nodes.query_router import build_query_response

        result = build_query_response(
            {
                "query": "latest GAA cost",
                "query_type": "rag_qa",
                "source_preference": "web",
                "response_type": "answer",
                "status": "running",
                "warnings": [],
                "child_graph_result": {
                    "documents": [],
                    "retrieval_source": "web_search",
                    "retrieval_error": "web_search_retry_exhausted",
                    "web_search_attempt_count": 2,
                    "web_search_queries": [
                        "GAA semiconductor production",
                        "Gate-All-Around transistor mass production progress",
                    ],
                    "web_relevant_result_count": 0,
                },
            }
        )

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["answer"], "")
        self.assertIn("未取得足以回答问题的相关资料。", result["warnings"])
        self.assertEqual(result["details"].get("web_search_attempt_count"), 2)
        self.assertEqual(len(result["details"].get("web_search_queries", [])), 2)
        self.assertEqual(result["details"].get("web_relevant_result_count"), 0)

    def test_web_report_is_written_to_report_field(self):
        from src.workflows.nodes.query_router import build_query_response

        result = build_query_response(
            {
                "query": "生成新能源汽车行业报告",
                "query_type": "rag_qa",
                "source_preference": "web",
                "response_type": "report",
                "routing_reason": "非半导体报告",
                "status": "running",
                "warnings": [],
                "child_graph_result": {
                    "generation": "# 新能源汽车行业报告",
                    "documents": [],
                    "retrieval_source": "web_search",
                },
            }
        )

        self.assertEqual(result["answer"], "")
        self.assertEqual(result["report"], "# 新能源汽车行业报告")

    def test_selection_clarification_preserves_awaiting_user_status(self):
        from src.workflows.nodes.query_router import build_query_response

        with patch(
            "src.infrastructure.reporting.docx.export_report_docx"
        ) as exporter:
            result = build_query_response(
                {
                    "query": "帮我选一个半导体技术",
                    "query_type": "technology_selection",
                    "source_preference": "auto",
                    "response_type": "report",
                    "report_format": "docx",
                    "routing_reason": "半导体选型",
                    "status": "running",
                    "warnings": [],
                    "child_graph_result": {
                        "status": "awaiting_user",
                        "clarification_questions": ["应用场景是什么？"],
                        "requirements": {"business_goal": ""},
                    },
                }
            )

        self.assertEqual(result["status"], "awaiting_user")
        self.assertEqual(
            result["clarification_questions"],
            ["应用场景是什么？"],
        )
        self.assertEqual(result["report"], "")
        self.assertIsNone(result["report_path"])
        exporter.assert_not_called()
        self.assertEqual(
            result["details"]["requirements"],
            {"business_goal": ""},
        )

    def test_selection_human_review_status_and_report_are_preserved(self):
        from src.workflows.nodes.query_router import build_query_response

        result = build_query_response(
            {
                "query": "比较GAA和FinFET并给出报告",
                "query_type": "technology_selection",
                "source_preference": "auto",
                "response_type": "report",
                "routing_reason": "半导体技术选型报告",
                "status": "running",
                "warnings": [],
                "child_graph_result": {
                    "status": "needs_human_review",
                    "report": "# 条件性技术选型报告",
                    "warnings": ["仍有一个证据缺口"],
                    "evidence_records": [],
                },
            }
        )

        self.assertEqual(result["status"], "needs_human_review")
        self.assertEqual(result["report"], "# 条件性技术选型报告")
        self.assertEqual(result["warnings"], ["仍有一个证据缺口"])


class QueryRouterGraphTest(unittest.TestCase):
    def test_invalid_report_format_returns_a_controlled_public_error(self):
        import src.workflows.nodes.query_router as router_module
        from src.workflows.query_router_graph import query_router_graph

        classification_chain = MagicMock()
        with patch.object(
            router_module,
            "query_classification_chain",
            classification_chain,
        ):
            result = query_router_graph.invoke(
                {"query": "生成报告", "report_format": "pdf"}
            )

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["report_format"], "markdown")
        self.assertIsNone(result["report_path"])
        classification_chain.invoke.assert_not_called()

    def test_docx_format_flows_through_the_public_graph_and_exports_once(self):
        import src.workflows.nodes.query_router as router_module
        from src.workflows.query_router_graph import query_router_graph

        classification_chain = MagicMock()
        classification_chain.invoke.return_value = AIMessage(
            content=(
                '{"query_type":"technology_selection",'
                '"source_preference":"auto","response_type":"report",'
                '"reason":"半导体技术选型"}'
            )
        )
        selection_graph = MagicMock()
        selection_graph.invoke.return_value = {
            "status": "completed",
            "report": "# 技术选型报告\n\n## 功耗评价",
            "evidence_records": [],
        }
        with patch.object(
            router_module,
            "query_classification_chain",
            classification_chain,
        ), patch.object(
            router_module,
            "technology_selection_graph",
            selection_graph,
        ), patch(
            "src.infrastructure.reporting.docx.export_report_docx",
            return_value="D:/reports/selection.docx",
        ) as exporter:
            result = query_router_graph.invoke(
                {
                    "query": "比较GAA和FinFET",
                    "report_format": "docx",
                }
            )

        self.assertEqual(result["report_format"], "docx")
        self.assertEqual(result["report_path"], "D:/reports/selection.docx")
        exporter.assert_called_once()

    def test_importing_router_graph_does_not_open_network_connections(self):
        project_root = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import socket; "
                "blocked = lambda *args, **kwargs: "
                "(_ for _ in ()).throw(RuntimeError('network disabled')); "
                "socket.create_connection = blocked; "
                "socket.socket.connect = blocked; "
                "from src.workflows.query_router_graph import "
                "query_router_graph; "
                "print(type(query_router_graph).__name__)",
            ],
            cwd=project_root,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "CompiledStateGraph")

    def test_compiled_graph_contains_only_query_router_nodes(self):
        from src.workflows.query_router_graph import query_router_graph

        graph_view = query_router_graph.get_graph()
        self.assertEqual(
            set(graph_view.nodes),
            {
                "__start__",
                "classify_query",
                "respond_directly",
                "invoke_rag_qa",
                "invoke_technology_selection",
                "build_query_response",
                "__end__",
            },
        )
        edges = {
            (edge.source, edge.target, edge.conditional)
            for edge in graph_view.edges
        }
        self.assertIn(
            ("classify_query", "respond_directly", True),
            edges,
        )
        self.assertIn(
            ("respond_directly", "build_query_response", False),
            edges,
        )

    def test_identity_request_uses_only_the_direct_response_path(self):
        import src.workflows.nodes.query_router as router_module
        from src.observability.logging import log
        from src.workflows.query_router_graph import query_router_graph

        classification_chain = MagicMock()
        rag_graph = MagicMock()
        selection_graph = MagicMock()
        messages = []
        sink_id = log.add(messages.append, format="{message}")
        self.addCleanup(log.remove, sink_id)

        with patch.object(
            router_module,
            "query_classification_chain",
            classification_chain,
        ), patch.object(
            router_module,
            "rag_qa_graph",
            rag_graph,
        ), patch.object(
            router_module,
            "technology_selection_graph",
            selection_graph,
        ):
            result = query_router_graph.invoke(
                {"query": "你是谁", "report_format": "docx"}
            )

        self.assertEqual(result["query_type"], "direct_response")
        self.assertEqual(result["source_preference"], "none")
        self.assertEqual(result["status"], "completed")
        self.assertIn("半导体知识问答和技术选型", result["answer"])
        self.assertEqual(result["sources"], [])
        self.assertEqual(result["report"], "")
        self.assertIsNone(result["report_path"])
        classification_chain.invoke.assert_not_called()
        rag_graph.invoke.assert_not_called()
        selection_graph.invoke.assert_not_called()
        rendered = "\n".join(str(message) for message in messages)
        self.assertIn(
            "[入口][respond_directly] 开始：生成直接响应",
            rendered,
        )
        self.assertIn("响应类型=identity", rendered)
        self.assertNotIn("你是谁", rendered)

    def test_out_of_scope_request_does_not_invoke_child_graphs(self):
        import src.workflows.nodes.query_router as router_module
        from src.workflows.query_router_graph import query_router_graph

        classification_chain = MagicMock()
        classification_chain.invoke.return_value = AIMessage(
            content=(
                '{"query_type":"direct_response",'
                '"source_preference":"none","response_type":"answer",'
                '"direct_response_kind":"out_of_scope",'
                '"reason":"非半导体问题"}'
            )
        )
        rag_graph = MagicMock()
        selection_graph = MagicMock()

        with patch.object(
            router_module,
            "query_classification_chain",
            classification_chain,
        ), patch.object(
            router_module,
            "rag_qa_graph",
            rag_graph,
        ), patch.object(
            router_module,
            "technology_selection_graph",
            selection_graph,
        ):
            result = query_router_graph.invoke({"query": "今天天气怎么样"})

        self.assertEqual(result["query_type"], "direct_response")
        self.assertEqual(result["source_preference"], "none")
        self.assertIn("服务范围", result["answer"])
        self.assertEqual(result["sources"], [])
        classification_chain.invoke.assert_called_once()
        rag_graph.invoke.assert_not_called()
        selection_graph.invoke.assert_not_called()

    def test_rag_route_invokes_only_the_rag_graph(self):
        import src.workflows.nodes.query_router as router_module
        from src.workflows.query_router_graph import query_router_graph

        classification_chain = MagicMock()
        classification_chain.invoke.return_value = AIMessage(
            content=(
                '{"query_type":"rag_qa","source_preference":"web",'
                '"response_type":"report","reason":"半导体时效信息"}'
            )
        )
        rag_graph = MagicMock()
        rag_graph.invoke.return_value = {
            "generation": "# 台积电 2nm 量产进展",
            "documents": [],
            "retrieval_source": "web_search",
        }
        selection_graph = MagicMock()

        with patch.object(
            router_module,
            "query_classification_chain",
            classification_chain,
        ), patch.object(
            router_module,
            "rag_qa_graph",
            rag_graph,
        ), patch.object(
            router_module,
            "technology_selection_graph",
            selection_graph,
        ):
            result = query_router_graph.invoke(
                {"query": "生成最新台积电 2nm 量产进展报告"}
            )

        self.assertEqual(result["query_type"], "rag_qa")
        self.assertEqual(result["source_preference"], "web")
        self.assertEqual(result["report"], "# 台积电 2nm 量产进展")
        self.assertEqual(rag_graph.invoke.call_count, 1)
        self.assertEqual(selection_graph.invoke.call_count, 0)

    def test_selection_route_invokes_only_the_selection_graph(self):
        import src.workflows.nodes.query_router as router_module
        from src.observability.logging import log
        from src.workflows.query_router_graph import query_router_graph

        classification_chain = MagicMock()
        classification_chain.invoke.return_value = AIMessage(
            content=(
                '{"query_type":"technology_selection",'
                '"source_preference":"auto","response_type":"report",'
                '"reason":"半导体技术选型"}'
            )
        )
        rag_graph = MagicMock()
        selection_graph = MagicMock()
        selection_graph.invoke.return_value = {
            "status": "completed",
            "report": "# GAA 与 FinFET 技术选型报告",
            "evidence_records": [],
        }
        messages = []
        sink_id = log.add(messages.append, format="{message}")
        self.addCleanup(log.remove, sink_id)

        with patch.object(
            router_module,
            "query_classification_chain",
            classification_chain,
        ), patch.object(
            router_module,
            "rag_qa_graph",
            rag_graph,
        ), patch.object(
            router_module,
            "technology_selection_graph",
            selection_graph,
        ):
            result = query_router_graph.invoke(
                {"query": "比较GAA和FinFET并给出选型报告"}
            )

        self.assertEqual(result["query_type"], "technology_selection")
        self.assertEqual(
            result["report"],
            "# GAA 与 FinFET 技术选型报告",
        )
        self.assertEqual(rag_graph.invoke.call_count, 0)
        self.assertEqual(selection_graph.invoke.call_count, 1)
        rendered = "\n".join(str(message) for message in messages)
        self.assertIn(
            "[入口][classify_query] 开始：识别请求类型",
            rendered,
        )
        self.assertIn(
            "[入口][invoke_technology_selection] 开始：执行技术选型工作流",
            rendered,
        )
        self.assertIn(
            "[入口][invoke_technology_selection] 完成：执行技术选型工作流",
            rendered,
        )
        self.assertIn("状态=completed", rendered)
        self.assertIn(
            "[入口][build_query_response] 完成：整理统一响应",
            rendered,
        )


class QueryRouterCommandLineTest(unittest.TestCase):
    def test_cli_prints_top_level_node_trace_and_final_result(self):
        import src.workflows.query_router_graph as graph_module

        class FakeGraph:
            def __init__(self):
                self.inputs = []

            def stream(self, inputs):
                self.inputs.append(inputs)
                yield {
                    "classify_query": {
                        "query_type": "rag_qa",
                        "status": "running",
                    }
                }
                yield {
                    "invoke_rag_qa": {
                        "child_graph_result": {
                            "generation": "GAA回答",
                        }
                    }
                }
                yield {
                    "build_query_response": {
                        "query": "什么是GAA",
                        "query_type": "rag_qa",
                        "status": "completed",
                        "answer": "GAA回答",
                        "report": "",
                    }
                }

        fake_graph = FakeGraph()
        output = io.StringIO()

        with patch.object(
            graph_module,
            "query_router_graph",
            fake_graph,
        ), patch.object(
            graph_module,
            "warm_technology_selection_resources",
            return_value={"vector": "ready", "graph": "ready"},
        ) as warm_resources, patch(
            "builtins.input",
            side_effect=["什么是GAA", "q"],
        ), redirect_stdout(output):
            graph_module.run_interactive_cli()

        rendered = output.getvalue()
        warm_resources.assert_called_once_with()
        self.assertEqual(fake_graph.inputs, [{"query": "什么是GAA"}])
        self.assertIn("classify_query", rendered)
        self.assertIn("invoke_rag_qa", rendered)
        self.assertIn("build_query_response", rendered)
        self.assertIn("'status': 'completed'", rendered)
        self.assertIn("'answer': 'GAA回答'", rendered)
        self.assertIn("对话结束。", rendered)

    def test_cli_exit_command_does_not_execute_graph(self):
        import src.workflows.query_router_graph as graph_module

        class GraphThatMustNotRun:
            def stream(self, _inputs):
                raise AssertionError("退出命令不应执行工作流")

        output = io.StringIO()

        with patch.object(
            graph_module,
            "query_router_graph",
            GraphThatMustNotRun(),
        ), patch.object(
            graph_module,
            "warm_technology_selection_resources",
            return_value={"vector": "ready", "graph": "ready"},
        ) as warm_resources, patch(
            "builtins.input",
            return_value="EXIT",
        ), redirect_stdout(output):
            graph_module.run_interactive_cli()

        warm_resources.assert_called_once_with()
        self.assertIn("对话结束。", output.getvalue())


if __name__ == "__main__":
    unittest.main()

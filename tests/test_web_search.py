import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


class ZhipuSearchTest(unittest.TestCase):
    def test_adapter_normalizes_zhipu_search_results(self):
        from src.infrastructure.search.zhipu import ZhipuSearch

        request = MagicMock()
        request.return_value = {
            "search_result": [
                {
                    "title": "GAA manufacturing update",
                    "content": "search result",
                    "link": "https://example.com/gaa",
                    "media": "Example",
                    "publish_date": "2026-07-30",
                }
            ],
        }

        results = ZhipuSearch(
            request=request,
            api_key="test-key",
        ).invoke("latest semiconductor news")

        self.assertEqual(
            results,
            [
                {
                    "title": "GAA manufacturing update",
                    "content": "search result",
                    "url": "https://example.com/gaa",
                }
            ],
        )
        request.assert_called_once_with(
            "latest semiconductor news",
            "test-key",
            5,
        )

    def test_adapter_raises_controlled_error_for_provider_failure(self):
        from src.infrastructure.search.zhipu import WebSearchError, ZhipuSearch

        request = MagicMock()
        request.return_value = {
            "error": {
                "code": "provider_unavailable",
                "message": "provider unavailable",
            }
        }

        with self.assertRaises(WebSearchError):
            ZhipuSearch(
                request=request,
                api_key="test-key",
            ).invoke("latest semiconductor news")

    def test_adapter_accepts_a_selection_specific_result_budget(self):
        from src.infrastructure.search.zhipu import ZhipuSearch

        request = MagicMock(return_value={"search_result": []})

        results = ZhipuSearch(
            request=request,
            api_key="test-key",
        ).invoke(
            "SiC automotive evidence",
            max_results=5,
            search_engine="search_pro",
        )

        self.assertEqual(results, [])
        request.assert_called_once_with(
            "SiC automotive evidence",
            "test-key",
            5,
            "search_pro",
        )


class WebSearchNodeTest(unittest.TestCase):
    def test_search_keeps_the_contextual_question_for_a_pronoun_follow_up(self):
        from src.workflows.nodes.web_search import web_search

        contextual_question = (
            "会话摘要：已讨论光刻机及其半导体制造用途。\n"
            "当前用户问题：它主要应用在哪些领域？"
        )
        results = [
            {
                "title": "光刻机应用",
                "url": "https://example.com/lithography",
                "content": "光刻机用于半导体晶圆制造。",
            }
        ]
        grader = MagicMock()
        grader.invoke.return_value = SimpleNamespace(relevant_indices=[0])
        with patch(
            "src.workflows.nodes.web_search.web_search_tool.invoke",
            return_value=results,
        ) as search, patch(
            "src.workflows.nodes.web_search.retrieval_grader_chain",
            grader,
        ):
            result = web_search(
                {
                    "question": contextual_question,
                    "original_question": "它主要应用在哪些领域？",
                }
            )

        self.assertIn("光刻机", search.call_args.args[0])
        self.assertIn(
            "光刻机",
            grader.invoke.call_args.args[0]["question"],
        )
        self.assertEqual(result["question"], contextual_question)

    def test_ordinary_search_expands_semiconductor_acronym_and_uses_pro_budget(self):
        from src.workflows.nodes.web_search import web_search

        results = [
            {
                "title": "GAA transistor production update",
                "url": "https://example.com/gaa-production",
                "content": "Gate-All-Around transistor mass production progress.",
            }
        ]
        relevance_grader = MagicMock()
        relevance_grader.invoke.return_value = SimpleNamespace(
            relevant_indices=[0]
        )
        with patch(
            "src.workflows.nodes.web_search.web_search_tool.invoke",
            return_value=results,
        ) as search, patch(
            "src.workflows.nodes.web_search.retrieval_grader_chain",
            relevance_grader,
            create=True,
        ):
            result = web_search(
                {"question": "目前GAA技术的量产进展如何？"}
            )

        search_query = search.call_args.args[0]
        self.assertIn("Gate-All-Around", search_query)
        self.assertIn("semiconductor", search_query)
        search.assert_called_once_with(
            search_query,
            max_results=5,
            search_engine="search_pro",
        )
        self.assertEqual(result["question"], "目前GAA技术的量产进展如何？")
        self.assertEqual(result["web_search_attempt_count"], 1)
        self.assertEqual(result["web_relevant_result_count"], 1)

    def test_irrelevant_first_search_is_rewritten_and_retried_once(self):
        from src.workflows.nodes.web_search import web_search

        irrelevant = [
            {
                "title": "固态电池量产进展",
                "url": "https://example.com/battery",
                "content": "固态电池已经进入量产验证阶段。",
            }
        ]
        relevant = [
            {
                "title": "GAA transistor production update",
                "url": "https://example.com/gaa",
                "content": "Gate-All-Around transistor mass production evidence.",
            }
        ]
        grader = MagicMock()
        grader.invoke.side_effect = [
            SimpleNamespace(relevant_indices=[]),
            SimpleNamespace(relevant_indices=[0]),
        ]
        rewriter = MagicMock()
        rewriter.invoke.return_value = (
            "Gate-All-Around semiconductor transistor mass production progress"
        )
        with patch(
            "src.workflows.nodes.web_search.web_search_tool.invoke",
            side_effect=[irrelevant, relevant],
        ) as search, patch(
            "src.workflows.nodes.web_search.retrieval_grader_chain",
            grader,
            create=True,
        ), patch(
            "src.workflows.nodes.web_search.web_query_rewriter_chain",
            rewriter,
            create=True,
        ):
            result = web_search(
                {"question": "目前GAA技术的量产进展如何？"}
            )

        self.assertEqual(search.call_count, 2)
        self.assertIn("Gate-All-Around", search.call_args_list[1].args[0])
        self.assertEqual(result["web_search_attempt_count"], 2)
        self.assertEqual(len(result["web_search_queries"]), 2)
        self.assertEqual(result["documents"][0].metadata["url"], "https://example.com/gaa")
        self.assertIsNone(result["retrieval_error"])

    def test_relevance_failure_keeps_only_documents_matching_core_entity(self):
        from src.workflows.nodes.web_search import web_search

        results = [
            {
                "title": "GAA production",
                "url": "",
                "content": "GAA transistor manufacturing evidence.",
            },
            {
                "title": "Solid-state battery production",
                "url": "https://example.com/battery",
                "content": "Battery manufacturing evidence.",
            },
        ]
        failed_grader = MagicMock()
        failed_grader.invoke.side_effect = ValueError("invalid grader output")
        with patch(
            "src.workflows.nodes.web_search.web_search_tool.invoke",
            return_value=results,
        ), patch(
            "src.workflows.nodes.web_search.retrieval_grader_chain",
            failed_grader,
            create=True,
        ):
            result = web_search({"question": "GAA量产进展"})

        self.assertEqual(len(result["documents"]), 1)
        self.assertEqual(result["documents"][0].metadata["title"], "GAA production")
        self.assertEqual(result["documents"][0].metadata["url"], "")

    def test_results_are_returned_as_individual_source_documents(self):
        from src.workflows.nodes.web_search import web_search

        results = [
            {
                "title": "GAA manufacturing update",
                "url": "https://example.com/gaa",
                "content": "GAA evidence",
            },
            {
                "title": "FinFET cost review",
                "url": "https://example.com/finfet",
                "content": "FinFET evidence",
            },
        ]
        relevance_grader = MagicMock()
        relevance_grader.invoke.return_value = SimpleNamespace(
            relevant_indices=[0, 1]
        )
        with patch(
            "src.workflows.nodes.web_search.web_search_tool.invoke",
            return_value=results,
        ), patch(
            "src.workflows.nodes.web_search.retrieval_grader_chain",
            relevance_grader,
        ):
            result = web_search({"question": "Compare GAA and FinFET"})

        self.assertEqual(len(result["documents"]), 2)
        self.assertEqual(
            result["documents"][0].metadata,
            {
                "title": "GAA manufacturing update",
                "source": "https://example.com/gaa",
                "url": "https://example.com/gaa",
                "retrieval_source": "web_search",
            },
        )
        self.assertEqual(
            result["documents"][1].page_content,
            "FinFET evidence",
        )

    def test_provider_failure_returns_error_without_generatable_document(self):
        from src.infrastructure.search.zhipu import WebSearchError
        from src.workflows.nodes.web_search import web_search

        with patch(
            "src.workflows.nodes.web_search.web_search_tool.invoke",
            side_effect=WebSearchError("provider unavailable"),
        ):
            result = web_search({"question": "今天有什么科技新闻？"})

        self.assertEqual(result["retrieval_source"], "web_search")
        self.assertEqual(result["documents"], [])
        self.assertEqual(result["retrieval_error"], "web_search_unavailable")
        self.assertEqual(result["web_search_attempt_count"], 1)
        self.assertEqual(len(result["web_search_queries"]), 1)

    def test_selection_search_filters_results_without_a_source_url(self):
        from src.workflows.nodes.web_search import web_search

        results = [
            {
                "title": "Unattributed snippet",
                "url": "",
                "content": "SiC snippet without a source URL",
            },
            {
                "title": "Attributed source",
                "url": "https://example.com/sic",
                "content": "SiC breakdown voltage is 1200V",
            },
        ]
        with patch(
            "src.workflows.nodes.web_search.web_search_tool.invoke",
            return_value=results,
        ) as search:
            result = web_search(
                {
                    "question": "SiC breakdown voltage",
                    "web_search_max_results": 5,
                    "web_search_engine": "search_pro",
                    "require_source_url": True,
                    "compact_relevance": True,
                }
            )

        self.assertEqual(len(result["documents"]), 1)
        self.assertEqual(
            result["documents"][0].metadata["url"],
            "https://example.com/sic",
        )
        search.assert_called_once_with(
            "SiC breakdown voltage",
            max_results=5,
            search_engine="search_pro",
        )


if __name__ == "__main__":
    unittest.main()

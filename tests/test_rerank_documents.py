import json
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document


def _candidates():
    vector = [
        Document(
            page_content=f"vector {index}",
            metadata={"id": f"vector-{index}", "retrieval_sources": ["vector"]},
        )
        for index in range(6)
    ]
    graph = [
        Document(
            page_content=f"graph {index}",
            metadata={"id": f"graph-{index}", "retrieval_sources": ["graph"]},
        )
        for index in range(2)
    ]
    return vector + graph


class RerankDocumentsTest(unittest.TestCase):
    def test_reranker_scores_all_candidates_and_keeps_top_n(self):
        import src.workflows.nodes.rerank_documents as rerank_module

        documents = _candidates()
        scores = {
            "vector 0": 0.2,
            "vector 1": 0.9,
            "vector 2": 0.1,
            "vector 3": 0.3,
            "vector 4": 0.05,
            "vector 5": 0.4,
            "graph 0": 0.95,
            "graph 1": 0.5,
        }
        reranker = MagicMock()
        reranker.score.side_effect = lambda _query, passages: [
            scores[passage] for passage in passages
        ]

        with patch.object(rerank_module, "RERANKER_TOP_N", 5), patch.object(
            rerank_module, "get_reranker", return_value=reranker
        ):
            result = rerank_module.rerank_documents(
                {"question": "GAA power", "documents": documents}
            )

        reranker.score.assert_called_once_with(
            "GAA power", [document.page_content for document in documents]
        )
        self.assertEqual(
            [document.metadata["id"] for document in result["documents"]],
            ["graph-0", "vector-1", "graph-1", "vector-5", "vector-3"],
        )
        self.assertEqual(result["documents"][0].metadata["rerank_score"], 0.95)
        self.assertNotIn("rerank_score", documents[6].metadata)

    def test_small_candidate_set_skips_model(self):
        import src.workflows.nodes.rerank_documents as rerank_module

        documents = _candidates()[:3]
        with patch.object(rerank_module, "RERANKER_TOP_N", 5), patch.object(
            rerank_module, "get_reranker"
        ) as get_reranker:
            result = rerank_module.rerank_documents(
                {"question": "GAA", "documents": documents}
            )

        get_reranker.assert_not_called()
        self.assertEqual(result["documents"], documents)

    def test_reranker_failure_interleaves_sources_so_graph_survives(self):
        import src.workflows.nodes.rerank_documents as rerank_module

        reranker = MagicMock()
        reranker.score.side_effect = RuntimeError("model unavailable")
        with patch.object(rerank_module, "RERANKER_TOP_N", 5), patch.object(
            rerank_module, "get_reranker", return_value=reranker
        ), patch.object(rerank_module, "resource_retry_allowed", return_value=True), patch.object(
            rerank_module, "mark_resource_failed"
        ) as mark_failed:
            result = rerank_module.rerank_documents(
                {"question": "GAA", "documents": _candidates()}
            )

        mark_failed.assert_called_once_with("reranker")
        self.assertEqual(
            [document.metadata["id"] for document in result["documents"]],
            ["vector-0", "graph-0", "vector-1", "graph-1", "vector-2"],
        )

    def test_disabled_reranker_never_loads_model(self):
        import src.workflows.nodes.rerank_documents as rerank_module

        with patch.object(rerank_module, "RERANKER_TOP_N", 5), patch.object(
            rerank_module, "RERANKER_ENABLED", False
        ), patch.object(rerank_module, "get_reranker") as get_reranker:
            result = rerank_module.rerank_documents(
                {"question": "GAA", "documents": _candidates()}
            )

        get_reranker.assert_not_called()
        self.assertEqual(len(result["documents"]), 5)


class SiliconFlowRerankerTest(unittest.TestCase):
    def _reranker(self, handler, **kwargs):
        import httpx

        from src.infrastructure.llm.reranker import SiliconFlowReranker

        options = {
            "api_key": "test-key",
            "base_url": "https://rerank.test/v1/",
            "timeout_seconds": 5,
            "max_retries": 1,
        }
        options.update(kwargs)
        return SiliconFlowReranker(
            "Qwen/Qwen3-VL-Reranker-8B",
            client=httpx.Client(transport=httpx.MockTransport(handler)),
            **options,
        )

    def test_scores_are_mapped_back_to_passage_order(self):
        import httpx

        requests = []

        def handler(request):
            requests.append(request)
            return httpx.Response(
                200,
                json={
                    "results": [
                        {"index": 1, "relevance_score": 0.9},
                        {"index": 0, "relevance_score": 0.2},
                    ]
                },
            )

        scores = self._reranker(handler).score("GAA", ["a", "b", "c"])

        self.assertEqual(scores, [0.2, 0.9, float("-inf")])
        request = requests[0]
        self.assertEqual(str(request.url), "https://rerank.test/v1/rerank")
        self.assertEqual(request.headers["authorization"], "Bearer test-key")
        self.assertEqual(
            json.loads(request.content),
            {
                "model": "Qwen/Qwen3-VL-Reranker-8B",
                "query": "GAA",
                "documents": ["a", "b", "c"],
                "top_n": 3,
                "return_documents": False,
            },
        )

    def test_overloaded_service_is_retried_once(self):
        import httpx

        responses = [
            httpx.Response(503, json={"code": 50505}),
            httpx.Response(
                200, json={"results": [{"index": 0, "relevance_score": 0.5}]}
            ),
        ]

        scores = self._reranker(lambda _request: responses.pop(0)).score(
            "GAA", ["a"]
        )

        self.assertEqual(scores, [0.5])
        self.assertEqual(responses, [])

    def test_missing_api_key_fails_without_network(self):
        handler = MagicMock()
        reranker = self._reranker(handler, api_key=None)

        with self.assertRaises(RuntimeError):
            reranker.score("GAA", ["a"])
        handler.assert_not_called()

    def test_warm_opens_connection_with_unbilled_models_request(self):
        import httpx

        requests = []

        def handler(request):
            requests.append(request)
            return httpx.Response(200, json={"data": []})

        self.assertTrue(self._reranker(handler).warm())

        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0].method, "GET")
        self.assertEqual(requests[0].url.path, "/v1/models")
        self.assertEqual(requests[0].headers["authorization"], "Bearer test-key")

    def test_warm_network_failure_does_not_disable_reranking(self):
        import httpx

        def handler(request):
            raise httpx.ConnectError("offline", request=request)

        self.assertTrue(self._reranker(handler).warm())

    def test_warm_without_api_key_fails_without_network(self):
        handler = MagicMock()

        with self.assertRaises(RuntimeError):
            self._reranker(handler, api_key=None).warm()
        handler.assert_not_called()


class GradeDocumentsLimitTest(unittest.TestCase):
    def test_grader_limits_come_from_configuration(self):
        import src.workflows.nodes.grade_documents as grade_module

        grader = MagicMock()
        grader.invoke.return_value = SimpleNamespace(relevant_indices=[0, 1, 2])
        documents = [Document(page_content="y" * 50) for _ in range(4)]

        with patch.object(grade_module, "RELEVANCE_MAX_DOCUMENTS", 2), patch.object(
            grade_module, "RELEVANCE_DOCUMENT_MAX_CHARACTERS", 20
        ), patch.object(grade_module, "retrieval_grader_chain", grader):
            result = grade_module.grade_documents(
                {"question": "GAA", "documents": documents}
            )

        formatted = grader.invoke.call_args.args[0]["documents"]
        self.assertIn("[1]", formatted)
        self.assertNotIn("[2]", formatted)
        self.assertIn("y" * 20, formatted)
        self.assertNotIn("y" * 21, formatted)
        self.assertEqual(result["documents"], documents[:2])


if __name__ == "__main__":
    unittest.main()

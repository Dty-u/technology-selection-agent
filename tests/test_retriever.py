import importlib
import logging
import sys
import types
import unittest
from unittest.mock import MagicMock, patch


class _FakeVectorStore:
    def __init__(self, captured):
        self._captured = captured

    def as_retriever(self, **kwargs):
        self._captured.update(kwargs)
        return object()


class _FakeMilvusVectorSave:
    captured = {}

    def __init__(self):
        self.vector_store_saved = _FakeVectorStore(self.captured)

    def create_connection(self):
        return None


def _import_milvus_with_fake_embeddings():
    module_name = "src.infrastructure.llm.embeddings"
    previous_module = sys.modules.get(module_name)
    fake_embeddings = types.ModuleType(module_name)
    fake_embeddings.bge_embedding = object()
    sys.modules[module_name] = fake_embeddings
    sys.modules.pop("src.infrastructure.vectorstores.milvus", None)
    try:
        return importlib.import_module("src.infrastructure.vectorstores.milvus")
    finally:
        if previous_module is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous_module


class RetrieverConfigurationTest(unittest.TestCase):
    def test_sync_milvus_connection_suppresses_only_missing_event_loop_warning(self):
        milvus = _import_milvus_with_fake_embeddings()

        upstream_logger = logging.getLogger(
            "langchain_milvus.vectorstores.milvus"
        )
        records = []
        handler = logging.Handler()
        handler.emit = records.append
        upstream_logger.addHandler(handler)

        class WarningMilvus:
            def __init__(self, **kwargs):
                upstream_logger.warning(
                    "Failed to initialize AsyncMilvusClient during Milvus "
                    "initialization: Cannot create async connection: "
                    "no running event loop. Async operations will be unavailable."
                )
                upstream_logger.warning("important Milvus warning")

        try:
            with patch.object(milvus, "Milvus", WarningMilvus):
                milvus.MilvusVectorSave().create_connection()
        finally:
            upstream_logger.removeHandler(handler)

        self.assertEqual(
            [record.getMessage() for record in records],
            ["important Milvus warning"],
        )

    def test_milvus_filter_is_an_expression_string(self):
        milvus = _import_milvus_with_fake_embeddings()
        _FakeMilvusVectorSave.captured.clear()

        with patch.object(milvus, "MilvusVectorSave", _FakeMilvusVectorSave):
            milvus.create_retriever()

        self.assertEqual(
            _FakeMilvusVectorSave.captured["search_kwargs"]["expr"],
            'category == "content"',
        )

    def test_hybrid_search_fetches_more_per_branch_than_it_returns(self):
        milvus = _import_milvus_with_fake_embeddings()
        _FakeMilvusVectorSave.captured.clear()

        with patch.object(milvus, "MilvusVectorSave", _FakeMilvusVectorSave):
            milvus.create_retriever()

        search_kwargs = _FakeMilvusVectorSave.captured["search_kwargs"]
        self.assertEqual(search_kwargs["k"], milvus.RETRIEVER_K)
        self.assertEqual(search_kwargs["fetch_k"], milvus.RETRIEVER_FETCH_K)
        self.assertGreaterEqual(search_kwargs["fetch_k"], search_kwargs["k"])

    def test_hnsw_ef_is_never_smaller_than_fetch_k(self):
        milvus = _import_milvus_with_fake_embeddings()
        _FakeMilvusVectorSave.captured.clear()

        with patch.object(milvus, "MilvusVectorSave", _FakeMilvusVectorSave), patch.object(
            milvus, "RETRIEVER_HNSW_EF", 10
        ), patch.object(milvus, "RETRIEVER_FETCH_K", 40):
            milvus.create_retriever()

        dense, sparse = _FakeMilvusVectorSave.captured["search_kwargs"]["param"]
        self.assertEqual(dense, {"metric_type": "IP", "params": {"ef": 40}})
        self.assertEqual(sparse["metric_type"], "BM25")

    def test_similarity_retriever_uses_rrf_function_without_legacy_ranker_keys(self):
        milvus = _import_milvus_with_fake_embeddings()
        _FakeMilvusVectorSave.captured.clear()

        with patch.object(milvus, "MilvusVectorSave", _FakeMilvusVectorSave):
            milvus.create_retriever()

        self.assertEqual(_FakeMilvusVectorSave.captured["search_type"], "similarity")
        self.assertNotIn(
            "score_threshold", _FakeMilvusVectorSave.captured["search_kwargs"]
        )
        self.assertNotIn(
            "ranker_type", _FakeMilvusVectorSave.captured["search_kwargs"]
        )
        self.assertNotIn(
            "ranker_params", _FakeMilvusVectorSave.captured["search_kwargs"]
        )
        reranker = _FakeMilvusVectorSave.captured["search_kwargs"]["reranker"]
        self.assertEqual(reranker.name, "rrf")
        self.assertEqual(reranker.type, milvus.FunctionType.RERANK)
        self.assertEqual(reranker.input_field_names, [])
        self.assertEqual(reranker.params, {"reranker": "rrf", "k": 100})

    def test_bootstrap_does_not_connect_until_retriever_is_requested(self):
        fake_milvus_module = types.ModuleType(
            "src.infrastructure.vectorstores.milvus"
        )
        create_retriever = MagicMock(return_value=object())
        fake_milvus_module.create_retriever = create_retriever

        with patch.dict(
            sys.modules,
            {"src.infrastructure.vectorstores.milvus": fake_milvus_module},
        ):
            sys.modules.pop("src.config.bootstrap", None)
            bootstrap = importlib.import_module("src.config.bootstrap")

            create_retriever.assert_not_called()
            first = bootstrap.get_retriever()
            second = bootstrap.get_retriever()

            self.assertIs(first, second)
            create_retriever.assert_called_once_with()

    def test_bootstrap_does_not_create_graph_retriever_until_requested(self):
        fake_graph_module = types.ModuleType(
            "src.infrastructure.graphdatabase.retriever"
        )
        create_graph_retriever = MagicMock(return_value=object())
        fake_graph_module.create_graph_retriever = create_graph_retriever

        with patch.dict(
            sys.modules,
            {
                "src.infrastructure.graphdatabase.retriever": fake_graph_module,
            },
        ):
            sys.modules.pop("src.config.bootstrap", None)
            bootstrap = importlib.import_module("src.config.bootstrap")

            create_graph_retriever.assert_not_called()
            first = bootstrap.get_graph_retriever()
            second = bootstrap.get_graph_retriever()

            self.assertIs(first, second)
            create_graph_retriever.assert_called_once_with()

    def test_bootstrap_warms_vector_and_graph_resources_without_mutation(self):
        import src.config.bootstrap as bootstrap

        graph_retriever = MagicMock()
        with patch.object(
            bootstrap,
            "get_retriever",
            return_value=object(),
        ) as get_retriever, patch.object(
            bootstrap,
            "get_graph_retriever",
            return_value=graph_retriever,
        ) as get_graph_retriever, patch.object(
            bootstrap,
            "get_reranker",
            return_value=MagicMock(),
        ) as get_reranker:
            result = bootstrap.warm_technology_selection_resources()

        self.assertEqual(
            result, {"vector": "ready", "graph": "ready", "reranker": "ready"}
        )
        get_reranker.return_value.warm.assert_called_once_with()
        get_retriever.assert_called_once_with()
        get_graph_retriever.assert_called_once_with()
        graph_retriever._get_graph.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()

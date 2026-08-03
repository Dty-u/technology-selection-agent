import importlib
import sys
import types
import unittest
from unittest.mock import MagicMock, patch


class _FakeEmbedding:
    def embed_query(self, text):
        return [0.1, 0.2, 0.3]


class MilvusSchemaTest(unittest.TestCase):
    def test_dense_dimension_matches_embedding_output(self):
        fake_embeddings = types.ModuleType("src.infrastructure.llm.embeddings")
        fake_embeddings.bge_embedding = _FakeEmbedding()

        with patch.dict(
            sys.modules,
            {"src.infrastructure.llm.embeddings": fake_embeddings},
        ):
            sys.modules.pop("src.infrastructure.vectorstores.milvus", None)
            milvus_db = importlib.import_module(
                "src.infrastructure.vectorstores.milvus"
            )

        client = MagicMock()
        schema = MagicMock()
        client.create_schema.return_value = schema
        client.prepare_index_params.return_value = MagicMock()
        client.list_collections.return_value = []

        with patch.object(milvus_db, "MilvusClient", return_value=client):
            milvus_db.MilvusVectorSave().create_collection()

        dense_field = next(
            call.kwargs
            for call in schema.add_field.call_args_list
            if call.kwargs.get("field_name") == "dense"
        )
        self.assertEqual(dense_field["dim"], 3)


if __name__ == "__main__":
    unittest.main()

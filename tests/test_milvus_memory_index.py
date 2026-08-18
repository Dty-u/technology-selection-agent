import unittest
from unittest.mock import patch


class _FakeEmbedding:
    def __init__(self):
        self.queries = []

    def embed_query(self, text):
        self.queries.append(text)
        return [0.1, 0.2, 0.3]


class _FakeSchema:
    def __init__(self, options):
        self.options = options
        self.fields = []

    def add_field(self, **kwargs):
        self.fields.append(kwargs)


class _FakeIndexParams:
    def __init__(self):
        self.indexes = []

    def add_index(self, **kwargs):
        self.indexes.append(kwargs)


class _FakeMilvusClient:
    def __init__(
        self,
        *,
        exists=True,
        description=None,
        index_description=None,
        search_result=None,
    ):
        self.exists = exists
        self.description = description
        self.index_description = index_description or {
            "index_type": "HNSW",
            "metric_type": "IP",
        }
        self.search_result = search_result or [[]]
        self.created = []
        self.upserted = []
        self.deleted = []
        self.searches = []

    def has_collection(self, *, collection_name):
        return self.exists

    def describe_collection(self, *, collection_name):
        return self.description

    def describe_index(self, *, collection_name, index_name):
        return self.index_description

    def create_schema(self, **kwargs):
        return _FakeSchema(kwargs)

    def prepare_index_params(self):
        return _FakeIndexParams()

    def create_collection(self, **kwargs):
        self.created.append(kwargs)

    def upsert(self, **kwargs):
        self.upserted.append(kwargs)
        return {"upsert_count": 1}

    def delete(self, **kwargs):
        self.deleted.append(kwargs)
        return {"delete_count": 1}

    def search(self, **kwargs):
        self.searches.append(kwargs)
        return self.search_result


def _compatible_description():
    return {
        "auto_id": False,
        "enable_dynamic_field": False,
        "fields": [
            {
                "name": "memory_id",
                "type": 21,
                "is_primary": True,
                "params": {"max_length": 64},
            },
            {"name": "dense_vector", "type": 101, "params": {"dim": 3}},
            {"name": "user_id", "type": 21, "params": {"max_length": 128}},
            {"name": "project_id", "type": 21, "params": {"max_length": 128}},
            {"name": "memory_type", "type": 21, "params": {"max_length": 64}},
            {"name": "status", "type": 21, "params": {"max_length": 32}},
            {"name": "version", "type": 5, "params": {}},
        ],
    }


class MilvusMemoryIndexTest(unittest.TestCase):
    def test_default_dependencies_are_lazy_until_an_operation_runs(self):
        import src.infrastructure.memory.milvus as milvus_module

        with patch.object(milvus_module, "MilvusClient") as create_client, patch.object(
            milvus_module, "_default_embedding"
        ) as create_embedding:
            milvus_module.MilvusMemoryIndex()

        create_client.assert_not_called()
        create_embedding.assert_not_called()

    def test_setup_creates_only_the_dedicated_collection_when_missing(self):
        from src.infrastructure.memory.milvus import MilvusMemoryIndex

        client = _FakeMilvusClient(exists=False)
        index = MilvusMemoryIndex(
            client=client,
            embedding=_FakeEmbedding(),
            collection_name="rag_agent_memory_v1",
        )

        created = index.ensure_collection()

        self.assertTrue(created)
        self.assertEqual(len(client.created), 1)
        request = client.created[0]
        self.assertEqual(request["collection_name"], "rag_agent_memory_v1")
        self.assertFalse(request["schema"].options["auto_id"])
        self.assertFalse(request["schema"].options["enable_dynamic_field"])
        fields = {field["field_name"]: field for field in request["schema"].fields}
        self.assertTrue(fields["memory_id"]["is_primary"])
        self.assertEqual(fields["dense_vector"]["dim"], 3)
        self.assertEqual(request["index_params"].indexes[0]["metric_type"], "IP")

    def test_setup_validates_existing_collection_without_recreating_it(self):
        from src.infrastructure.memory.milvus import MilvusMemoryIndex

        client = _FakeMilvusClient(description=_compatible_description())
        index = MilvusMemoryIndex(client=client, embedding=_FakeEmbedding())

        created = index.ensure_collection()

        self.assertFalse(created)
        self.assertEqual(client.created, [])
        self.assertFalse(hasattr(client, "drop_collection"))

    def test_setup_rejects_an_incompatible_existing_collection(self):
        from src.infrastructure.memory.milvus import MilvusMemoryIndex

        description = _compatible_description()
        description["fields"][1]["params"]["dim"] = 4
        client = _FakeMilvusClient(description=description)
        index = MilvusMemoryIndex(client=client, embedding=_FakeEmbedding())

        with self.assertRaisesRegex(RuntimeError, "incompatible"):
            index.ensure_collection()

        self.assertEqual(client.created, [])

    def test_setup_rejects_an_incompatible_existing_index(self):
        from src.infrastructure.memory.milvus import MilvusMemoryIndex

        client = _FakeMilvusClient(
            description=_compatible_description(),
            index_description={"index_type": "HNSW", "metric_type": "COSINE"},
        )
        index = MilvusMemoryIndex(client=client, embedding=_FakeEmbedding())

        with self.assertRaisesRegex(RuntimeError, "incompatible"):
            index.ensure_collection()

        self.assertEqual(client.created, [])

    def test_upsert_and_delete_use_authoritative_memory_id(self):
        from src.infrastructure.memory.milvus import MilvusMemoryIndex

        embedding = _FakeEmbedding()
        client = _FakeMilvusClient(description=_compatible_description())
        index = MilvusMemoryIndex(client=client, embedding=embedding)
        payload = {
            "memory_id": "memory-1",
            "user_id": "user-a",
            "project_id": None,
            "memory_type": "user_preference",
            "status": "confirmed",
            "version": 2,
            "content": "Prefer concise Chinese answers",
        }

        external_id = index.upsert(payload)
        deleted_id = index.delete(payload)

        self.assertEqual(external_id, "memory-1")
        self.assertEqual(deleted_id, "memory-1")
        self.assertEqual(embedding.queries, [payload["content"]])
        entity = client.upserted[0]["data"][0]
        self.assertEqual(entity["memory_id"], "memory-1")
        self.assertEqual(entity["project_id"], "")
        self.assertNotIn("content", entity)
        self.assertEqual(client.deleted[0]["ids"], ["memory-1"])

    def test_search_uses_filter_parameters_and_returns_only_hits(self):
        from src.infrastructure.memory.milvus import MilvusMemoryIndex

        client = _FakeMilvusClient(
            description=_compatible_description(),
            search_result=[
                [
                    {
                        "id": "memory-1",
                        "distance": 0.91,
                        "entity": {"memory_id": "memory-1", "version": 2},
                    }
                ]
            ],
        )
        index = MilvusMemoryIndex(client=client, embedding=_FakeEmbedding())

        hits = index.search_hits(
            query="checkpoint",
            user_id='user" OR status == "pending',
            project_id=None,
            memory_types=("user_preference", "user_fact"),
            statuses=("confirmed", "approved"),
            limit=5,
        )

        self.assertEqual(hits[0].memory_id, "memory-1")
        self.assertEqual(hits[0].version, 2)
        self.assertEqual(hits[0].score, 0.91)
        request = client.searches[0]
        self.assertNotIn('user" OR status', request["filter"])
        self.assertEqual(
            request["filter_params"]["user_id"],
            'user" OR status == "pending',
        )
        self.assertEqual(request["filter_params"]["project_id"], "")
        self.assertEqual(request["anns_field"], "dense_vector")
        self.assertEqual(request["output_fields"], ["memory_id", "version"])


if __name__ == "__main__":
    unittest.main()

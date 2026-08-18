"""Direct Milvus projection for long-term-memory semantic recall."""

from __future__ import annotations

from typing import Any, Sequence

from pymilvus import DataType, MilvusClient

from src.config.settings import MEMORY_COLLECTION_NAME, MILVUS_URI
from src.infrastructure.memory.models import SemanticMemoryHit


_FIELD_SPECS = {
    "memory_id": DataType.VARCHAR,
    "dense_vector": DataType.FLOAT_VECTOR,
    "user_id": DataType.VARCHAR,
    "project_id": DataType.VARCHAR,
    "memory_type": DataType.VARCHAR,
    "status": DataType.VARCHAR,
    "version": DataType.INT64,
}


def _default_embedding():
    from src.infrastructure.llm.embeddings import bge_embedding

    return bge_embedding


class MilvusMemoryIndex:
    """Disposable semantic projection backed by an authoritative MySQL record."""

    provider = "milvus"

    def __init__(
        self,
        *,
        client: Any | None = None,
        embedding: Any | None = None,
        collection_name: str = MEMORY_COLLECTION_NAME,
        uri: str = MILVUS_URI,
    ) -> None:
        self._client = client
        self._embedding = embedding
        self._uri = str(uri)
        self.collection_name = str(collection_name)
        self._dimension: int | None = None

    @property
    def client(self):
        if self._client is None:
            self._client = MilvusClient(uri=self._uri)
        return self._client

    @property
    def embedding(self):
        if self._embedding is None:
            self._embedding = _default_embedding()
        return self._embedding

    def warm(self) -> bool:
        """Read-only connectivity and collection-existence probe."""
        return bool(
            self.client.has_collection(collection_name=self.collection_name)
        )

    def _embedding_dimension(self) -> int:
        if self._dimension is None:
            self._dimension = len(self.embedding.embed_query("dimension probe"))
        if self._dimension <= 1:
            raise RuntimeError("memory embedding dimension must be greater than one")
        return self._dimension

    def ensure_collection(self) -> bool:
        """Create only when absent; validate and preserve any existing collection."""
        dimension = self._embedding_dimension()
        if self.client.has_collection(collection_name=self.collection_name):
            self._validate_collection(
                self.client.describe_collection(
                    collection_name=self.collection_name
                ),
                dimension=dimension,
            )
            return False

        schema = self.client.create_schema(
            auto_id=False,
            enable_dynamic_field=False,
        )
        schema.add_field(
            field_name="memory_id",
            datatype=DataType.VARCHAR,
            is_primary=True,
            max_length=64,
        )
        schema.add_field(
            field_name="dense_vector",
            datatype=DataType.FLOAT_VECTOR,
            dim=dimension,
        )
        schema.add_field(
            field_name="user_id", datatype=DataType.VARCHAR, max_length=128
        )
        schema.add_field(
            field_name="project_id", datatype=DataType.VARCHAR, max_length=128
        )
        schema.add_field(
            field_name="memory_type", datatype=DataType.VARCHAR, max_length=64
        )
        schema.add_field(
            field_name="status", datatype=DataType.VARCHAR, max_length=32
        )
        schema.add_field(field_name="version", datatype=DataType.INT64)

        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="dense_vector",
            index_name="memory_dense_hnsw",
            index_type="HNSW",
            metric_type="IP",
            params={"M": 16, "efConstruction": 64},
        )
        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params,
            consistency_level="Strong",
        )
        return True

    def _validate_collection(
        self, description: dict[str, Any], *, dimension: int
    ) -> None:
        fields = {
            str(field.get("name")): field
            for field in description.get("fields", [])
        }
        problems = []
        if bool(description.get("auto_id")):
            problems.append("auto_id must be false")
        if bool(description.get("enable_dynamic_field")):
            problems.append("dynamic fields must be disabled")
        for name, expected_type in _FIELD_SPECS.items():
            field = fields.get(name)
            if field is None:
                problems.append(f"missing field {name}")
                continue
            if int(field.get("type")) != int(expected_type):
                problems.append(f"wrong type for {name}")
        primary = fields.get("memory_id", {})
        if not primary.get("is_primary"):
            problems.append("memory_id must be primary")
        dense_params = fields.get("dense_vector", {}).get("params", {})
        if int(dense_params.get("dim", 0)) != dimension:
            problems.append("dense_vector dimension mismatch")
        for name, expected_length in {
            "memory_id": 64,
            "user_id": 128,
            "project_id": 128,
            "memory_type": 64,
            "status": 32,
        }.items():
            actual = fields.get(name, {}).get("params", {}).get("max_length")
            if int(actual or 0) != expected_length:
                problems.append(f"wrong max_length for {name}")
        try:
            index = self.client.describe_index(
                collection_name=self.collection_name,
                index_name="memory_dense_hnsw",
            )
        except Exception as exc:
            raise RuntimeError(
                "existing memory collection is incompatible: missing memory index"
            ) from exc
        if str(index.get("index_type", "")).upper() != "HNSW":
            problems.append("memory index must use HNSW")
        if str(index.get("metric_type", "")).upper() != "IP":
            problems.append("memory index must use IP")
        if problems:
            detail = "; ".join(problems)
            raise RuntimeError(
                f"existing memory collection is incompatible: {detail}"
            )

    def upsert(self, payload: dict[str, Any]) -> str:
        memory_id = str(payload["memory_id"])
        vector = self.embedding.embed_query(str(payload["content"]))
        self.client.upsert(
            collection_name=self.collection_name,
            data=[
                {
                    "memory_id": memory_id,
                    "dense_vector": vector,
                    "user_id": str(payload["user_id"]),
                    "project_id": str(payload.get("project_id") or ""),
                    "memory_type": str(payload["memory_type"]),
                    "status": str(payload["status"]),
                    "version": int(payload["version"]),
                }
            ],
        )
        return memory_id

    def delete(self, payload: dict[str, Any]) -> str:
        memory_id = str(payload["memory_id"])
        self.client.delete(
            collection_name=self.collection_name,
            ids=[memory_id],
        )
        return memory_id

    def search_hits(
        self,
        *,
        query: str,
        user_id: str,
        project_id: str | None,
        memory_types: Sequence[str],
        statuses: Sequence[str],
        limit: int,
    ) -> list[SemanticMemoryHit]:
        if not memory_types or not statuses or int(limit) <= 0:
            return []
        raw = self.client.search(
            collection_name=self.collection_name,
            data=[self.embedding.embed_query(str(query))],
            anns_field="dense_vector",
            filter=(
                "user_id == {user_id} AND project_id == {project_id} "
                "AND memory_type in {memory_types} AND status in {statuses}"
            ),
            filter_params={
                "user_id": str(user_id),
                "project_id": str(project_id or ""),
                "memory_types": list(memory_types),
                "statuses": list(statuses),
            },
            limit=int(limit),
            output_fields=["memory_id", "version"],
            search_params={"metric_type": "IP", "params": {"ef": 64}},
            consistency_level="Strong",
        )
        hits = raw[0] if raw else []
        results = []
        for hit in hits:
            entity = dict(hit.get("entity", {}) or {})
            memory_id = entity.get("memory_id") or hit.get("id")
            version = entity.get("version", hit.get("version"))
            if memory_id is None or version is None:
                continue
            results.append(
                SemanticMemoryHit(
                    memory_id=str(memory_id),
                    score=float(hit.get("distance", hit.get("score", 0))),
                    version=int(version),
                )
            )
        return results

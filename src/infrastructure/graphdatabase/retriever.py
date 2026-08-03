"""Retrieve source-backed relationship evidence from Neo4j."""

from typing import Any, Dict, List, Optional

from langchain_core.documents import Document

from src.config.settings import (
    GRAPH_MAX_DOCUMENTS,
    GRAPH_MAX_HOPS,
    GRAPH_PATH_LIMIT,
    GRAPH_SEED_LIMIT,
)
from src.infrastructure.graphdatabase.neo4j import Neo4jDatabase


class Neo4jGraphRetriever:
    """Find entity paths and return their supporting source documents."""

    def __init__(
        self,
        graph=None,
        max_hops: int = GRAPH_MAX_HOPS,
        seed_limit: int = GRAPH_SEED_LIMIT,
        path_limit: int = GRAPH_PATH_LIMIT,
        max_documents: int = GRAPH_MAX_DOCUMENTS,
    ):
        if max_hops not in (1, 2):
            raise ValueError("max_hops must be 1 or 2")
        self._graph = graph
        self._database: Optional[Neo4jDatabase] = None
        self.max_hops = max_hops
        self.seed_limit = seed_limit
        self.path_limit = path_limit
        self.max_documents = max_documents

    def _get_graph(self):
        if self._graph is None:
            self._database = Neo4jDatabase()
            self._graph = self._database.connect()
        return self._graph

    def invoke(self, query: Dict[str, Any]) -> List[Document]:
        """Retrieve graph paths for analyzed query entities."""
        entity_names = []
        seen_names = set()
        for entity in query.get("entities", []):
            normalized_name = str(entity).strip()
            name_key = normalized_name.casefold()
            if normalized_name and name_key not in seen_names:
                entity_names.append(normalized_name)
                seen_names.add(name_key)
        entity_names = entity_names[: self.seed_limit]
        if not entity_names:
            return []
        relation_types = [
            str(relation_type).strip().upper()
            for relation_type in query.get("relation_types", [])
            if str(relation_type).strip()
        ]

        cypher = f"""
        UNWIND $entity_names AS query_name
        MATCH (seed:Entity)
        WITH seed, query_name,
             CASE
                 WHEN toLower(coalesce(seed.name, '')) = toLower(query_name)
                   OR toLower(coalesce(seed.type, '')) = toLower(query_name)
                 THEN 0
                 WHEN toLower(coalesce(seed.name, '')) CONTAINS toLower(query_name)
                   OR toLower(query_name) CONTAINS toLower(coalesce(seed.name, ''))
                   OR toLower(coalesce(seed.type, '')) CONTAINS toLower(query_name)
                   OR toLower(query_name) CONTAINS toLower(coalesce(seed.type, ''))
                 THEN 1
                 ELSE 2
             END AS match_rank
        WHERE match_rank < 2
        WITH query_name, seed, match_rank
        ORDER BY query_name, match_rank, toLower(coalesce(seed.name, ''))
        WITH query_name, collect(seed)[0] AS seed
        WITH collect(seed) AS seeds
        UNWIND seeds AS seed
        MATCH path=(seed)-[:RELATED_TO*1..{self.max_hops}]-(related:Entity)
        WITH DISTINCT path, nodes(path) AS path_entities,
             relationships(path) AS path_relationships
        WITH path, path_entities, path_relationships,
             size([query_name IN $entity_names WHERE any(path_entity IN path_entities
                 WHERE toLower(coalesce(path_entity.name, '')) = toLower(query_name)
                    OR toLower(coalesce(path_entity.name, '')) CONTAINS toLower(query_name)
                    OR toLower(query_name) CONTAINS toLower(coalesce(path_entity.name, ''))
             )]) AS entity_coverage,
             size([relationship IN path_relationships
                 WHERE relationship.type IN $relation_types]) AS relation_coverage,
             length(path) AS hop_count
        WITH path, path_entities, path_relationships, entity_coverage,
             relation_coverage, hop_count,
             CASE WHEN entity_coverage >= 2 THEN 0 ELSE 1 END AS fallback_rank
        ORDER BY fallback_rank, entity_coverage DESC, relation_coverage DESC,
                 hop_count ASC,
                 [entity IN path_entities | toLower(coalesce(entity.name, ''))]
        LIMIT $path_limit
        MATCH (document:Document)
        WHERE all(path_entity IN path_entities WHERE EXISTS {{
            MATCH (document)-[:MENTIONS]->(path_entity)
        }})
        WITH DISTINCT document, path_entities, path_relationships
        WITH document, collect({{
            entity_names: [entity IN path_entities | entity.name],
            relationships: [relationship IN path_relationships | {{
                source: startNode(relationship).name,
                type: relationship.type,
                target: endNode(relationship).name
            }}]
        }}) AS graph_paths
        RETURN document.id AS document_id,
               document.text AS text,
               document.source AS source,
               graph_paths[0].entity_names AS entity_names,
               graph_paths[0].relationships AS relationships
        LIMIT $document_limit
        """
        rows = self._get_graph().query(
            cypher,
            {
                "entity_names": entity_names,
                "seed_limit": self.seed_limit,
                "path_limit": self.path_limit,
                "document_limit": self.max_documents,
                "relation_types": relation_types,
            },
        )
        relation_intent = str(query.get("relation_intent", "")).strip()
        documents = []
        seen_documents = set()
        for row in rows:
            document_key = row.get("document_id") or (
                row.get("source"),
                row.get("text"),
            )
            if document_key in seen_documents:
                continue
            seen_documents.add(document_key)
            relationships = row.get("relationships") or []
            relationship_text = "；".join(
                f"{relationship['source']} -[{relationship['type']}]-> "
                f"{relationship['target']}"
                for relationship in relationships
            )
            relationship_types = [
                relationship["type"] for relationship in relationships
            ]
            documents.append(
                Document(
                    page_content=(
                        f"图谱关系：{relationship_text}\n"
                        f"来源内容：{row.get('text') or ''}"
                    ),
                    metadata={
                        "id": row.get("document_id"),
                        "source": row.get("source"),
                        "retrieval_source": "graph",
                        "entities": row.get("entity_names") or [],
                        "relations": relationship_types,
                        "relation_intent": relation_intent,
                    },
                )
            )
            if len(documents) >= self.max_documents:
                break
        return documents

    def close(self) -> None:
        """Close the lazily created Neo4j connection."""
        if self._database is not None:
            self._database.close()
            self._database = None
            self._graph = None


def create_graph_retriever() -> Neo4jGraphRetriever:
    """Create a lazy graph retriever without connecting to Neo4j."""
    return Neo4jGraphRetriever()

from src.config.bootstrap import get_graph_retriever
from src.observability.logging import log


def graph_retrieve(state):
    """Retrieve Neo4j relationship evidence without blocking vector results."""
    try:
        documents = get_graph_retriever().invoke(
            {
                "entities": state.get("query_entities", []),
                "relation_intent": state.get("relation_intent", ""),
                "relation_types": state.get("relation_types", []),
            }
        )
        log.info(f"---图谱检索完成：候选关系证据 {len(documents)} 条---")
        return {"graph_documents": documents, "graph_error": None}
    except Exception as error:
        log.error("---图谱检索失败：将继续使用向量检索结果---")
        return {"graph_documents": [], "graph_error": str(error)}

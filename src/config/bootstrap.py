from functools import lru_cache


@lru_cache(maxsize=1)
def get_retriever():
    """Create the Milvus retriever on first use and reuse it afterwards."""
    from src.infrastructure.vectorstores.milvus import create_retriever

    return create_retriever()


@lru_cache(maxsize=1)
def get_graph_retriever():
    """Create the Neo4j graph retriever on first use and reuse it afterwards."""
    from src.infrastructure.graphdatabase.retriever import create_graph_retriever

    return create_graph_retriever()


def warm_technology_selection_resources():
    """Warm reusable retrieval resources during host startup without mutation."""
    result = {}
    try:
        get_retriever()
        result["vector"] = "ready"
    except Exception as error:
        result["vector"] = type(error).__name__
    try:
        get_graph_retriever()._get_graph()
        result["graph"] = "ready"
    except Exception as error:
        result["graph"] = type(error).__name__
    return result

from hashlib import sha256

from langchain_core.documents import Document

from src.config.settings import GRAPH_MAX_DOCUMENTS, RETRIEVER_K


def _document_key(document) -> str:
    document_id = document.metadata.get("id")
    if document_id:
        return f"id:{document_id}"
    source = document.metadata.get("source", "")
    digest = sha256(document.page_content.encode("utf-8")).hexdigest()
    return f"content:{source}:{digest}"


def merge_retrieval_results(state):
    """Limit, deduplicate, and merge vector and graph evidence."""
    documents = []
    documents_by_key = {}
    vector_documents = state.get("vector_documents", [])[:RETRIEVER_K]
    graph_documents = state.get("graph_documents", [])[:GRAPH_MAX_DOCUMENTS]

    for source, branch_documents in (
        ("vector", vector_documents),
        ("graph", graph_documents),
    ):
        for document in branch_documents:
            key = _document_key(document)
            existing = documents_by_key.get(key)
            if existing is None:
                metadata = dict(document.metadata)
                metadata["retrieval_sources"] = [source]
                documents_by_key[key] = Document(
                    page_content=document.page_content,
                    metadata=metadata,
                )
                documents.append(documents_by_key[key])
                continue

            if source != "graph" or "graph" in existing.metadata["retrieval_sources"]:
                continue
            graph_text = document.page_content.split("\n来源内容：", 1)[0]
            if graph_text and graph_text not in existing.page_content:
                existing.page_content = f"{existing.page_content}\n{graph_text}"
            existing.metadata["retrieval_sources"].append("graph")

    log_message = (
        f"---混合检索融合完成：向量 {len(vector_documents)} 条，"
        f"图谱 {len(graph_documents)} 条，保留证据 {len(documents)} 条---"
    )
    from src.observability.logging import log

    log.info(log_message)

    return {
        "question": state["question"],
        "documents": documents,
        "retrieval_source": "local_hybrid",
    }

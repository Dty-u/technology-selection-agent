from langchain_core.documents import Document

from src.config.bootstrap import (
    get_reranker,
    mark_resource_failed,
    mark_resource_ready,
    resource_retry_allowed,
)
from src.config.settings import RERANKER_ENABLED, RERANKER_TOP_N
from src.observability.logging import log


def _primary_source(document) -> str:
    sources = document.metadata.get("retrieval_sources") or [
        document.metadata.get("retrieval_source", "vector")
    ]
    return sources[0]


def _interleave_by_source(documents):
    """Fallback order that keeps graph evidence from being crowded out."""
    queues = {}
    for document in documents:
        queues.setdefault(_primary_source(document), []).append(document)
    ordered = []
    while any(queues.values()):
        for queue in queues.values():
            if queue:
                ordered.append(queue.pop(0))
    return ordered


def rerank_documents(state):
    """Rerank merged coarse candidates with a cross-encoder and keep the top N."""
    question = state["question"]
    documents = state.get("documents", [])
    if len(documents) <= RERANKER_TOP_N:
        log.info(f"---候选证据 {len(documents)} 条，无需精排---")
        return {"documents": documents}

    if not RERANKER_ENABLED or not resource_retry_allowed("reranker"):
        log.info("---精排不可用：按来源交替截取---")
        return {"documents": _interleave_by_source(documents)[:RERANKER_TOP_N]}

    try:
        scores = get_reranker().score(
            question, [document.page_content for document in documents]
        )
        mark_resource_ready("reranker")
    except Exception:
        mark_resource_failed("reranker")
        log.exception("---精排失败：按来源交替截取---")
        return {"documents": _interleave_by_source(documents)[:RERANKER_TOP_N]}

    ranked = sorted(
        zip(scores, range(len(documents)), documents),
        key=lambda item: (-item[0], item[1]),
    )
    reranked = []
    for score, _, document in ranked[:RERANKER_TOP_N]:
        metadata = dict(document.metadata)
        metadata["rerank_score"] = score
        reranked.append(
            Document(page_content=document.page_content, metadata=metadata)
        )
    log.info(
        f"---精排完成：候选 {len(documents)} 条，保留 {len(reranked)} 条---"
    )
    return {"documents": reranked}

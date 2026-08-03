from src.observability.logging import log
from src.config.settings import SELECTION_RELEVANCE_DOCUMENT_MAX_CHARACTERS
from src.workflows.graders.document_relevance import (
    retrieval_grader_chain,
    selection_retrieval_grader_chain,
)


def grade_documents(state):
    """
    评估检索到的文档与问题的相关性
    过滤不相关的doc

    Args:
        state (dict): 当前图状态，包含问题和检索结果

    Returns:
        state (dict): 更新后的状态，documents字段仅保留相关文档
    """
    log.info("---开始相关性筛选---")
    question = state["question"]  # 获取用户问题
    documents = state["documents"]  # 获取待评估文档

    candidates = documents[:6]
    if not candidates:
        log.info("---相关性筛选完成：保留证据 0 条---")
        return {"documents": [], "question": question}

    document_limit = (
        SELECTION_RELEVANCE_DOCUMENT_MAX_CHARACTERS
        if state.get("compact_relevance")
        else 1000
    )
    formatted_documents = "\n\n".join(
        f"[{index}] 来源：{document.metadata.get('source', '')}\n"
        f"{document.page_content[:document_limit]}"
        for index, document in enumerate(candidates)
    )
    try:
        grader_chain = (
            selection_retrieval_grader_chain
            if state.get("compact_relevance")
            else retrieval_grader_chain
        )
        score = grader_chain.invoke(
            {"question": question, "documents": formatted_documents}
        )
        selected_indices = {
            index for index in score.relevant_indices if 0 <= index < len(candidates)
        }
        filtered_docs = [
            document for index, document in enumerate(candidates)
            if index in selected_indices
        ]
    except Exception:
        filtered_docs = [
            document for document in candidates
            if "vector" in document.metadata.get("retrieval_sources", [
                document.metadata.get("retrieval_source", "vector")
            ])
        ]
        log.warning("---相关性筛选异常：保留向量证据---")
    log.info(f"---相关性筛选完成：保留证据 {len(filtered_docs)} 条---")
    return {"documents": filtered_docs, "question": question}

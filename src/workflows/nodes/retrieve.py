from src.config.bootstrap import (
    get_retriever,
    mark_resource_failed,
    mark_resource_ready,
    resource_retry_allowed,
)
from src.observability.logging import log


def retrieve(state):
    """
    检索相关文档
    Args:
        state (dict): 当前图状态，包含用户问题

    Returns:
        state (dict): 更新后的状态，新增包含检索结果的documents字段
    """
    log.info("---开始向量检索---")
    question = state["question"]  # 从状态中获取用户问题
    if not resource_retry_allowed("vector"):
        return {
            "vector_documents": [],
            "vector_error": "resource_failure_cooldown",
            "question": question,
        }
    try:
        documents = get_retriever().invoke(question)
        mark_resource_ready("vector")
    except Exception as error:
        mark_resource_failed("vector")
        log.error("---向量检索失败：将在冷却期后重试---")
        return {
            "vector_documents": [],
            "vector_error": str(error),
            "question": question,
        }
    log.info(f"---向量检索完成：候选文档 {len(documents)} 条---")
    return {
        "vector_documents": documents,
        "vector_error": None,
        "question": question,
    }

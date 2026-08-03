from src.config.bootstrap import get_retriever
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
    # 文档检索
    documents = get_retriever().invoke(question)  # 调用检索器获取相关文档
    log.info(f"---向量检索完成：候选文档 {len(documents)} 条---")
    return {"vector_documents": documents, "question": question}

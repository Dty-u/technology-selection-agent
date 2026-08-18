from src.config.settings import MAX_TRANSFORM_COUNT
from src.observability.logging import log


def decide_to_generate(state):
    """
    决定是生成回答还是重新优化问题

    Args:
        state (dict): 当前图状态，包含问题和过滤后的文档

    Returns:
        str: 下一节点的名称（transform_query或generate）
    """
    log.info("---ASSESS GRADED DOCUMENTS---")  # 阶段标识

    filtered_documents = state["documents"]  # 获取已过滤文档
    transform_count = state.get("transform_count", 0)

    if not filtered_documents:  # 如果没有相关文档
        if state.get("vector_error") and state.get("graph_error"):
            log.info("---本地检索器均不可用，直接转为 Web 检索---")
            return "web_search"
        if transform_count >= MAX_TRANSFORM_COUNT:
            log.info(f"---决策：所有文档都与问题无关,并且已经循环了{MAX_TRANSFORM_COUNT}次，转为web查询问题---")
            return "web_search"  # 返回问题优化节点
        log.info("---决策：所有文档都与问题无关，将转换查询问题---")
        return "transform_query"  # 返回问题优化节点
    else:  # 如果有相关文档
        log.info("---决策：生成最终回答---")
        return "generate"  # 返回回答生成节点

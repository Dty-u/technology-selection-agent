"""Conditional route for the single-query entry workflow."""

from src.observability.logging import log


def route_by_query_type(state):
    """Select one child graph, or finish a failed classification."""
    if state.get("status") == "error":
        log.info(
            "[入口][route_by_query_type] "
            "路由：分类失败，整理错误响应"
        )
        return "build_query_response"
    if state.get("query_type") == "rag_qa":
        log.info(
            "[入口][route_by_query_type] 路由：进入普通问答工作流"
        )
        return "invoke_rag_qa"
    if state.get("query_type") == "technology_selection":
        log.info(
            "[入口][route_by_query_type] 路由：进入技术选型工作流"
        )
        return "invoke_technology_selection"
    if state.get("query_type") == "direct_response":
        log.info(
            "[入口][route_by_query_type] 路由：进入直接响应"
        )
        return "respond_directly"
    log.info(
        "[入口][route_by_query_type] "
        "路由：类型未识别，整理统一响应"
    )
    return "build_query_response"

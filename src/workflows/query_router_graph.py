"""Compiled single-query entry workflow."""

from pprint import pprint

from langgraph.constants import END, START
from langgraph.graph import StateGraph

from src.config.bootstrap import warm_technology_selection_resources
from src.observability.logging import log
from src.observability.workflow_progress import logged_node
from src.workflows.nodes.query_router import (
    build_query_response,
    classify_query,
    invoke_rag_qa,
    invoke_technology_selection,
    respond_directly,
)
from src.workflows.query_router_state import (
    QueryRouterInput,
    QueryRouterOutput,
    QueryRouterState,
)
from src.workflows.routes.query_router import route_by_query_type


workflow = StateGraph(
    QueryRouterState,
    input_schema=QueryRouterInput,
    output_schema=QueryRouterOutput,
)


def _child_status(result):
    child_result = result.get("child_graph_result", {})
    return {"状态": child_result.get("status", "completed")}


workflow.add_node(
    "classify_query",
    logged_node("入口", "classify_query", "识别请求类型", classify_query),
)
workflow.add_node(
    "invoke_rag_qa",
    logged_node(
        "入口",
        "invoke_rag_qa",
        "执行普通问答工作流",
        invoke_rag_qa,
        result_details=_child_status,
    ),
)
workflow.add_node(
    "invoke_technology_selection",
    logged_node(
        "入口",
        "invoke_technology_selection",
        "执行技术选型工作流",
        invoke_technology_selection,
        result_details=_child_status,
    ),
)
workflow.add_node(
    "respond_directly",
    logged_node(
        "入口",
        "respond_directly",
        "生成直接响应",
        respond_directly,
    ),
)
workflow.add_node(
    "build_query_response",
    logged_node(
        "入口",
        "build_query_response",
        "整理统一响应",
        build_query_response,
    ),
)

workflow.add_edge(START, "classify_query")
workflow.add_conditional_edges(
    "classify_query",
    route_by_query_type,
    {
        "invoke_rag_qa": "invoke_rag_qa",
        "invoke_technology_selection": "invoke_technology_selection",
        "respond_directly": "respond_directly",
        "build_query_response": "build_query_response",
    },
)
workflow.add_edge("invoke_rag_qa", "build_query_response")
workflow.add_edge(
    "invoke_technology_selection",
    "build_query_response",
)
workflow.add_edge("respond_directly", "build_query_response")
workflow.add_edge("build_query_response", END)

query_router_graph = workflow.compile()


def run_interactive_cli():
    """Run the top-level query router and print its node trace."""
    warm_result = warm_technology_selection_resources()
    log.info(
        "[启动][技术选型预热] 完成 | 向量={} | 图={}",
        warm_result.get("vector", "unknown"),
        warm_result.get("graph", "unknown"),
    )
    while True:
        query = input("用户：").strip()
        if query.lower() in {"q", "exit", "quit"}:
            print("对话结束。")
            break
        if not query:
            continue

        final_state = None
        for output in query_router_graph.stream({"query": query}):
            for node_name, state in output.items():
                print(f"\n节点：{node_name}")
                final_state = state

        print("\n最终结果：")
        pprint(final_state, sort_dicts=False)


if __name__ == "__main__":
    run_interactive_cli()

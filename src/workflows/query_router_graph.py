"""Compiled single-query entry workflow."""

import argparse
from pprint import pprint
from uuid import uuid4

from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.types import Command

from src.observability.logging import log
from src.observability.workflow_progress import logged_node
from src.workflows.nodes.query_router import (
    build_query_response,
    classify_query,
    invoke_rag_qa,
    invoke_technology_selection,
    make_invoke_technology_selection,
    respond_directly,
)
from src.workflows.nodes.memory import (
    make_persist_memory_candidates,
    make_recall_memory,
    stage_memory_candidates,
)
from src.workflows.nodes.short_term_memory import (
    finalize_short_term_memory,
    prepare_short_term_memory,
)
from src.workflows.query_router_state import (
    QueryRouterInput,
    QueryRouterOutput,
    QueryRouterState,
)
from src.workflows.routes.query_router import route_by_query_type
from src.workflows.technology_selection_graph import (
    build_technology_selection_graph,
)


def _child_status(result):
    child_result = result.get("child_graph_result", {})
    return {"状态": child_result.get("status", "completed")}


def _route_to_memory_or_response(state):
    if state.get("status") == "error":
        return "build_query_response"
    if state.get("query_type") == "direct_response":
        return "respond_directly"
    return "recall_memory"


def _route_after_memory(state):
    if state.get("status") == "error":
        return "build_query_response"
    if state.get("query_type") == "technology_selection":
        return "invoke_technology_selection"
    return "invoke_rag_qa"


def _route_after_short_term_prepare(state):
    if state.get("status") == "error":
        return "build_query_response"
    return "classify_query"


def _compile_router_graph(
    invoke_selection_node,
    checkpointer=None,
    memory_repository=None,
    semantic_memory_reader=None,
):
    """Compile the router workflow with one technology-selection node."""
    workflow = StateGraph(
        QueryRouterState,
        input_schema=QueryRouterInput,
        output_schema=QueryRouterOutput,
    )
    workflow.add_node(
        "prepare_short_term_memory",
        prepare_short_term_memory,
    )
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
            invoke_selection_node,
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
    workflow.add_node(
        "finalize_short_term_memory",
        finalize_short_term_memory,
    )

    workflow.add_edge(START, "prepare_short_term_memory")
    workflow.add_conditional_edges(
        "prepare_short_term_memory",
        _route_after_short_term_prepare,
        {
            "classify_query": "classify_query",
            "build_query_response": "build_query_response",
        },
    )
    if memory_repository is None:
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
    else:
        workflow.add_node(
            "recall_memory",
            make_recall_memory(memory_repository, semantic_memory_reader),
        )
        workflow.add_node(
            "stage_memory_candidates",
            stage_memory_candidates,
        )
        workflow.add_node(
            "persist_memory_candidates",
            make_persist_memory_candidates(memory_repository),
        )
        workflow.add_conditional_edges(
            "classify_query",
            _route_to_memory_or_response,
            {
                "recall_memory": "recall_memory",
                "respond_directly": "respond_directly",
                "build_query_response": "build_query_response",
            },
        )
        workflow.add_conditional_edges(
            "recall_memory",
            _route_after_memory,
            {
                "invoke_rag_qa": "invoke_rag_qa",
                "invoke_technology_selection": "invoke_technology_selection",
                "build_query_response": "build_query_response",
            },
        )
    workflow.add_edge("invoke_rag_qa", "build_query_response")
    workflow.add_edge(
        "invoke_technology_selection",
        "build_query_response",
    )
    workflow.add_edge("respond_directly", "build_query_response")
    workflow.add_edge("build_query_response", "finalize_short_term_memory")
    if memory_repository is None:
        workflow.add_edge("finalize_short_term_memory", END)
    else:
        workflow.add_edge("finalize_short_term_memory", "stage_memory_candidates")
        workflow.add_edge("stage_memory_candidates", "persist_memory_candidates")
        workflow.add_edge("persist_memory_candidates", END)
    return workflow.compile(checkpointer=checkpointer)


query_router_graph = _compile_router_graph(invoke_technology_selection)


def build_query_router_graph(
    checkpointer=None,
    *,
    memory_repository=None,
    semantic_memory_reader=None,
):
    """Compile the router with optional checkpoint and long-term memory adapters."""
    if checkpointer is None:
        return _compile_router_graph(
            invoke_technology_selection,
            memory_repository=memory_repository,
            semantic_memory_reader=semantic_memory_reader,
        )
    selection_graph = build_technology_selection_graph(checkpointer)
    return _compile_router_graph(
        make_invoke_technology_selection(selection_graph),
        checkpointer=checkpointer,
        memory_repository=memory_repository,
        semantic_memory_reader=semantic_memory_reader,
    )


def run_interactive_cli(
    thread_id=None,
    *,
    user_id=None,
    project_id=None,
    application_factory=None,
):
    """Run the MySQL-checkpointed, long-term-memory application.

    A new UUID is generated and printed when no ``--thread-id`` is supplied.
    ``/exit`` keeps the checkpoint; ``/close`` deletes the thread's checkpoint
    before exiting. Plain ``q``/``quit`` never deletes a checkpoint.
    """
    normalized_user_id = str(user_id or "").strip()
    if not normalized_user_id:
        raise ValueError("user_id must not be empty")
    normalized_project_id = str(project_id or "").strip()
    if application_factory is None:
        from src.workflows.application import open_memory_application

        application_factory = open_memory_application

    if not thread_id:
        thread_id = str(uuid4())
        print(f"新建会话 thread_id={thread_id}")
    configurable = {
        "user_id": normalized_user_id,
        "thread_id": thread_id,
    }
    if normalized_project_id:
        configurable["project_id"] = normalized_project_id
    session_config = {"configurable": configurable}
    with application_factory() as application:
        resuming = bool(application.graph.get_state(session_config).next)
        if resuming:
            print(f"检测到待恢复会话 thread_id={thread_id}")

        while True:
            user_input = input("用户：").strip()
            if not user_input:
                continue
            command = user_input.lower()
            if command == "/exit":
                print("对话结束，已保留会话状态。")
                break
            if command == "/close":
                application.close_session(thread_id, user_id=normalized_user_id)
                print("对话结束，会话已关闭。")
                break
            if command in {"q", "quit", "exit"}:
                print("对话结束。")
                break

            turn_config = {
                "configurable": {
                    **configurable,
                    "request_id": str(uuid4()),
                }
            }
            if resuming:
                final_state = application.invoke(
                    Command(resume=user_input),
                    config=turn_config,
                )
            else:
                final_state = application.invoke(
                    {"query": user_input},
                    config=turn_config,
                )
            resuming = bool(application.graph.get_state(session_config).next)

            print("\n最终结果：")
            pprint(final_state, sort_dicts=False)

    return thread_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="半导体技术问答与技术选型会话 CLI")
    parser.add_argument(
        "--user-id",
        required=True,
        help="长期记忆所属的用户ID（必填）",
    )
    parser.add_argument(
        "--thread-id",
        help="恢复或新建的会话 thread_id；不提供时自动生成",
    )
    parser.add_argument(
        "--project-id",
        help="技术选型项目ID；执行技术选型时必填",
    )
    args = parser.parse_args()
    run_interactive_cli(
        user_id=args.user_id,
        thread_id=args.thread_id,
        project_id=args.project_id,
    )

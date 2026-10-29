"""Shared local retrieval chain for the RAG and evidence-retrieval graphs.

Both graphs run the same analyze -> parallel vector/graph retrieval -> merge ->
rerank -> grade chain, with query rewriting looping back into retrieval. Each
graph only adds its own entry route and what happens after grading.
"""

from langgraph.constants import START

from src.observability.workflow_progress import logged_node
from src.workflows.nodes.analyze_question import (
    analyze_question,
    extract_retrieval_intent,
)
from src.workflows.nodes.grade_documents import grade_documents
from src.workflows.nodes.graph_retrieve import graph_retrieve
from src.workflows.nodes.merge_retrieval_results import merge_retrieval_results
from src.workflows.nodes.rerank_documents import rerank_documents
from src.workflows.nodes.retrieve import retrieve
from src.workflows.nodes.transform_query import transform_query
from src.workflows.nodes.web_search import web_search
from src.workflows.routes.route_after_analysis import route_after_analysis


def retrieval_metrics(result):
    metrics = {}
    for field in (
        "documents",
        "vector_documents",
        "graph_documents",
    ):
        value = result.get(field)
        if isinstance(value, list):
            metrics[f"{field}_count"] = len(value)
    if "cache_hit" in result:
        metrics["cache_hit"] = bool(result["cache_hit"])
    return metrics


def make_timed_node(stage):
    """Wrap nodes with progress logging under one workflow stage label."""

    def timed_node(name, action, node, *, model_calls=0):
        return logged_node(
            stage,
            name,
            action,
            node,
            heartbeat_seconds=None,
            start_details=lambda state: {
                "model_calls": (
                    0
                    if name == "analyze_question" and state.get("analysis_ready")
                    else model_calls
                )
            },
            result_details=retrieval_metrics,
        )

    return timed_node


def add_local_retrieval(workflow, timed_node, *, entry_route):
    """Register shared retrieval nodes and edges; grading exits stay per graph."""
    workflow.add_node(
        "analyze_question",
        timed_node("analyze_question", "分析问题", analyze_question, model_calls=1),
    )
    workflow.add_node(
        "extract_retrieval_intent",
        timed_node(
            "extract_retrieval_intent",
            "抽取检索意图",
            extract_retrieval_intent,
            model_calls=1,
        ),
    )
    workflow.add_node(
        "web_search",
        timed_node("web_search", "联网检索", web_search, model_calls=1),
    )
    workflow.add_node(
        "vector_retrieve",
        timed_node("vector_retrieve", "向量检索", retrieve),
    )
    workflow.add_node(
        "graph_retrieve",
        timed_node("graph_retrieve", "图谱检索", graph_retrieve),
    )
    workflow.add_node(
        "merge_retrieval_results",
        timed_node(
            "merge_retrieval_results", "融合检索结果", merge_retrieval_results
        ),
    )
    workflow.add_node(
        "rerank_documents",
        timed_node("rerank_documents", "精排证据", rerank_documents),
    )
    workflow.add_node(
        "grade_documents",
        timed_node("grade_documents", "筛选相关证据", grade_documents, model_calls=1),
    )
    workflow.add_node(
        "transform_query",
        timed_node("transform_query", "改写查询", transform_query, model_calls=1),
    )

    workflow.add_conditional_edges(
        START,
        entry_route,
        {
            "analyze_question": "analyze_question",
            "web_search": "web_search",
        },
    )
    workflow.add_conditional_edges(
        "analyze_question",
        route_after_analysis,
        {
            "web_search": "web_search",
            "vector_retrieve": "vector_retrieve",
            "graph_retrieve": "graph_retrieve",
        },
    )
    workflow.add_edge(
        ["vector_retrieve", "graph_retrieve"],
        "merge_retrieval_results",
    )
    workflow.add_edge("merge_retrieval_results", "rerank_documents")
    workflow.add_edge("rerank_documents", "grade_documents")
    workflow.add_edge("transform_query", "extract_retrieval_intent")
    workflow.add_edge("extract_retrieval_intent", "vector_retrieve")
    workflow.add_edge("extract_retrieval_intent", "graph_retrieve")

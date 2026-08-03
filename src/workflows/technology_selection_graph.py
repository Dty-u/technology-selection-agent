"""Compiled semiconductor technology-selection workflow."""

from langgraph.constants import END, START
from langgraph.graph import StateGraph

from src.config.settings import (
    MAX_EVIDENCE_ROUNDS,
    MAX_EVIDENCE_TASK_CONCURRENCY,
    MAX_REVIEW_ROUNDS,
)
from src.observability.workflow_progress import logged_node
from src.workflows.nodes.selection_decision import (
    analyze_risks,
    evaluate_candidates,
    generate_recommendation,
    generate_report,
    review_recommendation,
)
from src.workflows.nodes.selection_evidence import (
    optimize_evidence_queries,
    plan_evidence_tasks,
    retrieve_evidence,
)
from src.workflows.nodes.selection_requirements import (
    clarify_requirements,
    generate_candidates,
    parse_requirements,
    screen_hard_constraints,
)
from src.workflows.routes.technology_selection import (
    route_evidence,
    route_feasible_candidates,
    route_recommendation_review,
    route_requirements,
)
from src.workflows.technology_selection_state import TechnologySelectionState


workflow = StateGraph(TechnologySelectionState)


def _evidence_round_details(state):
    return {
        "执行轮次": (
            f"{state.get('evidence_round', 0) + 1}/{MAX_EVIDENCE_ROUNDS}"
        ),
        "并发上限": MAX_EVIDENCE_TASK_CONCURRENCY,
    }


def _review_round_details(state):
    return {
        "执行轮次": f"{state.get('review_round', 0) + 1}/{MAX_REVIEW_ROUNDS}"
    }


_NODES = (
    ("parse_requirements", "解析选型需求", parse_requirements, {}),
    ("clarify_requirements", "生成需求澄清问题", clarify_requirements, {}),
    ("generate_candidates", "生成候选方案", generate_candidates, {}),
    ("screen_hard_constraints", "筛选硬约束", screen_hard_constraints, {}),
    ("plan_evidence_tasks", "规划证据任务", plan_evidence_tasks, {}),
    (
        "retrieve_evidence",
        "检索并评估证据",
        retrieve_evidence,
        {"start_details": _evidence_round_details},
    ),
    (
        "optimize_evidence_queries",
        "生成定向补检任务",
        optimize_evidence_queries,
        {},
    ),
    ("evaluate_candidates", "计算候选评分", evaluate_candidates, {}),
    ("analyze_risks", "分析方案风险", analyze_risks, {}),
    (
        "generate_recommendation",
        "生成推荐结论",
        generate_recommendation,
        {},
    ),
    (
        "review_recommendation",
        "审查推荐结论",
        review_recommendation,
        {"start_details": _review_round_details},
    ),
    ("generate_report", "生成Markdown技术选型报告", generate_report, {}),
)

for node_name, action, node, options in _NODES:
    workflow.add_node(
        node_name,
        logged_node("技术选型", node_name, action, node, **options),
    )

workflow.add_edge(START, "parse_requirements")
workflow.add_conditional_edges(
    "parse_requirements",
    route_requirements,
    {
        "generate_candidates": "generate_candidates",
        "clarify_requirements": "clarify_requirements",
    },
)
workflow.add_edge("clarify_requirements", END)
workflow.add_edge("generate_candidates", "screen_hard_constraints")
workflow.add_conditional_edges(
    "screen_hard_constraints",
    route_feasible_candidates,
    {
        "plan_evidence_tasks": "plan_evidence_tasks",
        "clarify_requirements": "clarify_requirements",
    },
)
workflow.add_edge("plan_evidence_tasks", "retrieve_evidence")
workflow.add_conditional_edges(
    "retrieve_evidence",
    route_evidence,
    {
        "optimize_evidence_queries": "optimize_evidence_queries",
        "evaluate_candidates": "evaluate_candidates",
    },
)
workflow.add_edge("optimize_evidence_queries", "retrieve_evidence")
workflow.add_edge("evaluate_candidates", "analyze_risks")
workflow.add_edge("analyze_risks", "generate_recommendation")
workflow.add_edge("generate_recommendation", "review_recommendation")
workflow.add_conditional_edges(
    "review_recommendation",
    route_recommendation_review,
    {
        "optimize_evidence_queries": "optimize_evidence_queries",
        "evaluate_candidates": "evaluate_candidates",
        "generate_report": "generate_report",
    },
)
workflow.add_edge("generate_report", END)

technology_selection_graph = workflow.compile()

from langgraph.constants import END

from src.config.settings import (
    MAX_EVIDENCE_ROUNDS,
    MAX_REVIEW_ROUNDS,
    MAX_TRANSFORM_COUNT,
)
from src.observability.logging import log


def route_retrieval_entry(state):
    """Select direct Web search only for explicitly fresh evidence tasks."""
    if state.get("source_preference", "auto") == "web":
        return "web_search"
    return "analyze_question"


def route_retrieval_result(state):
    """Finish with evidence, otherwise rewrite locally before Web fallback."""
    if state.get("documents"):
        return END
    max_transform_count = state.get(
        "max_transform_count",
        MAX_TRANSFORM_COUNT,
    )
    if state.get("transform_count", 0) >= max_transform_count:
        if state.get("allow_web_fallback", True):
            return "web_search"
        return END
    return "transform_query"


def route_requirements(state):
    """Continue only when the two minimum requirement fields are present."""
    if state.get("requirements_complete"):
        log.info(
            "[技术选型][route_requirements] 路由：需求完整，进入候选生成"
        )
        return "generate_candidates"
    log.info(
        "[技术选型][route_requirements] 路由：需求信息不足，进入需求澄清"
    )
    return "clarify_requirements"


def route_feasible_candidates(state):
    """Retrieve evidence only when at least one candidate remains."""
    if state.get("feasible_candidates"):
        log.info(
            "[技术选型][route_feasible_candidates] "
            "路由：存在可行候选，进入证据规划 | 可行候选={}",
            len(state["feasible_candidates"]),
        )
        return "plan_evidence_tasks"
    log.info(
        "[技术选型][route_feasible_candidates] "
        "路由：没有可行候选，进入需求澄清"
    )
    return "clarify_requirements"


def route_evidence(state):
    """Bound targeted evidence retrieval before evaluation."""
    gap_count = len(state.get("evidence_gaps", []))
    if state.get("evidence_sufficient"):
        log.info(
            "[技术选型][route_evidence] 路由：证据充分，进入候选评分"
        )
        return "evaluate_candidates"
    if state.get("evidence_round", 0) >= MAX_EVIDENCE_ROUNDS:
        log.info(
            "[技术选型][route_evidence] "
            "路由：补检达到上限，带缺口进入候选评分 | 缺口={}",
            gap_count,
        )
        return "evaluate_candidates"
    next_round = state.get("evidence_round", 0) + 1
    log.info(
        "[技术选型][route_evidence] "
        "路由：证据不足，进入第{}轮补检 | 缺口={}",
        next_round,
        gap_count,
    )
    return "optimize_evidence_queries"


def route_recommendation_review(state):
    """Route an audited recommendation with a bounded repair loop."""
    review = state.get("recommendation_review", {})
    verdict = review.get("verdict", "reevaluate")
    if verdict == "pass":
        log.info(
            "[技术选型][route_recommendation_review] "
            "路由：审查通过，进入报告生成"
        )
        return "generate_report"
    if state.get("review_round", 0) >= MAX_REVIEW_ROUNDS:
        log.info(
            "[技术选型][route_recommendation_review] "
            "路由：审查达到上限，进入报告生成 | 审查结果={}",
            verdict,
        )
        return "generate_report"
    if (
        verdict == "evidence_gap"
        and state.get("evidence_round", 0) >= MAX_EVIDENCE_ROUNDS
    ):
        log.info(
            "[技术选型][route_recommendation_review] "
            "路由：证据补检达到上限，进入报告生成 | 缺口={}",
            len(state.get("evidence_gaps", [])),
        )
        return "generate_report"
    if verdict == "evidence_gap":
        log.info(
            "[技术选型][route_recommendation_review] "
            "路由：发现证据缺口，进入定向补检 | 缺口={}",
            len(state.get("evidence_gaps", [])),
        )
        return "optimize_evidence_queries"
    log.info(
        "[技术选型][route_recommendation_review] "
        "路由：审查要求重新评分 | 审查结果={}",
        verdict,
    )
    return "evaluate_candidates"

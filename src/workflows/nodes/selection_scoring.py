"""Deterministic scoring, winner selection, and review checks for selection decisions.

These helpers never call a model; the decision nodes use them to validate and
explain model output.
"""

from src.config.settings import (
    DECISION_EVIDENCE_CARD_MAX_CHARACTERS,
    MAX_DECISION_EVIDENCE_CARDS_PER_CRITERION,
)
from src.workflows.technology_selection_state import (
    CRITERIA,
    CriterionScore,
    RiskItem,
    SelectionRequirements,
)


def _response_content(response) -> str:
    return response.content if hasattr(response, "content") else str(response)


def _decision_evidence_cards(records):
    """Build bounded evidence context without changing evidence attribution."""
    counts = {}
    cards = []
    for record in records:
        pairs = [
            (support.get("candidate_id"), support.get("criterion"))
            for support in record.get("supports", [])
            if support.get("level") in {"direct", "indirect"}
        ]
        if not pairs or not any(
            counts.get(pair, 0) < MAX_DECISION_EVIDENCE_CARDS_PER_CRITERION
            for pair in pairs
        ):
            continue
        for pair in pairs:
            if counts.get(pair, 0) < MAX_DECISION_EVIDENCE_CARDS_PER_CRITERION:
                counts[pair] = counts.get(pair, 0) + 1
        cards.append(
            {
                "evidence_id": record.get("evidence_id", ""),
                "candidate_ids": record.get("candidate_ids", []),
                "criteria": record.get("criteria", []),
                "supports": record.get("supports", []),
                "source": record.get("source", ""),
                "title": record.get("title", ""),
                "url": record.get("url", ""),
                "source_reliability": record.get("source_reliability", "internal"),
                "content": str(record.get("content", ""))[
                    :DECISION_EVIDENCE_CARD_MAX_CHARACTERS
                ],
            }
        )
    return cards


def _has_scoreable_support(records, candidate_id, criterion, evidence_id):
    for record in records:
        if record.get("evidence_id") != evidence_id:
            continue
        for support in record.get("supports", []):
            if (
                support.get("candidate_id") == candidate_id
                and support.get("criterion") == criterion
                and support.get("level") in {"direct", "indirect"}
            ):
                return True
    return False


def _validated_score(raw_score, candidate_id, criterion, evidence_records):
    if raw_score is None:
        return CriterionScore(criterion=criterion)
    score = CriterionScore.model_validate(raw_score)
    score.criterion = criterion
    score.evidence_ids = [
        evidence_id
        for evidence_id in score.evidence_ids
        if _has_scoreable_support(
            evidence_records,
            candidate_id,
            criterion,
            evidence_id,
        )
    ]
    if score.score is not None and not score.evidence_ids:
        score.score = None
        score.confidence = "low"
    return score


def _active_fixed_criteria(requirements):
    active = [
        criterion
        for criterion in requirements.active_criteria
        if criterion in CRITERIA
    ]
    if active:
        return active
    weighted = [
        criterion
        for criterion in CRITERIA
        if requirements.priorities.get(criterion, 0) > 0
    ]
    return weighted or list(CRITERIA)


def _review_feedback(state):
    """Return only actionable feedback from the preceding review round."""
    review = state.get("recommendation_review", {})
    if review.get("verdict") not in {"reevaluate", "evidence_gap"}:
        return {}
    return {
        "verdict": review.get("verdict"),
        "reasons": list(review.get("reasons", [])),
        "requested_queries": list(review.get("requested_queries", [])),
    }


def _validated_risk_items(raw_items, evidence_records):
    if not isinstance(raw_items, list):
        raise ValueError("risk_items must be a list")
    evidence_candidates = {
        record["evidence_id"]: set(record.get("candidate_ids", []))
        for record in evidence_records
    }
    risks = []
    for item in raw_items:
        risk = RiskItem.model_validate(item)
        risk.evidence_ids = [
            evidence_id
            for evidence_id in risk.evidence_ids
            if risk.candidate_id in evidence_candidates.get(evidence_id, set())
        ]
        if risk.basis == "evidence" and not risk.evidence_ids:
            risk.basis = "to_verify"
        risks.append(risk.model_dump())
    return risks


def _unique_weighted_winner(evaluations, candidate_ids):
    if not evaluations or len(evaluations) != len(candidate_ids):
        return None
    scores = {
        evaluation.get("candidate_id"): evaluation.get("weighted_score")
        for evaluation in evaluations
        if evaluation.get("candidate_id") in candidate_ids
    }
    if len(scores) != len(candidate_ids) or any(
        score is None for score in scores.values()
    ):
        return None
    highest = max(scores.values())
    winners = [candidate_id for candidate_id, score in scores.items() if score == highest]
    return winners[0] if len(winners) == 1 else None


def _highest_priority_scenario_winner(requirements, evaluations, candidate_ids):
    highest_ids = [
        item.criterion_id
        for item in requirements.scenario_criteria
        if item.priority == "highest"
    ]
    if not highest_ids:
        highest_ids = [
            item.criterion_id
            for item in requirements.scenario_criteria
            if item.priority == "high"
        ]
    if not highest_ids or not evaluations or len(evaluations) != len(candidate_ids):
        return None, []
    totals = {}
    for evaluation in evaluations:
        candidate_id = evaluation.get("candidate_id")
        if candidate_id not in candidate_ids:
            continue
        scores = {
            item.get("criterion"): item.get("score")
            for item in evaluation.get("scenario_scores", [])
        }
        if any(scores.get(criterion) is None for criterion in highest_ids):
            continue
        totals[candidate_id] = sum(scores[criterion] for criterion in highest_ids)
    if not totals:
        return None, highest_ids
    highest = max(totals.values())
    winners = [candidate_id for candidate_id, score in totals.items() if score == highest]
    return (winners[0] if len(winners) == 1 else None), highest_ids


def _ranked_alternative_ids(evaluations, candidate_ids, primary_id):
    scores = {
        evaluation.get("candidate_id"): evaluation.get("weighted_score")
        for evaluation in evaluations
    }
    original_order = {
        candidate_id: index
        for index, candidate_id in enumerate(candidate_ids)
    }
    return sorted(
        (
            candidate_id
            for candidate_id in candidate_ids
            if candidate_id != primary_id
        ),
        key=lambda candidate_id: (
            scores.get(candidate_id) is None,
            -(scores.get(candidate_id) or 0),
            original_order[candidate_id],
        ),
    )


def _weighted_score_rationale(evaluations, candidate_names):
    ranked = sorted(
        (
            (
                evaluation.get("candidate_id"),
                evaluation.get("weighted_score"),
            )
            for evaluation in evaluations
            if evaluation.get("weighted_score") is not None
        ),
        key=lambda item: -item[1],
    )
    if not ranked:
        return "固定指标综合分尚未完整计算。"
    summary = "，".join(
        f"{candidate_names.get(candidate_id, candidate_id)} {score:.2f}"
        for candidate_id, score in ranked
    )
    return f"固定指标综合分排序：{summary}。"


def _scenario_rationales(requirements, evaluations, candidate_names):
    score_maps = {
        evaluation.get("candidate_id"): {
            score.get("criterion"): score.get("score")
            for score in evaluation.get("scenario_scores", [])
        }
        for evaluation in evaluations
    }
    rationales = []
    for criterion in requirements.scenario_criteria:
        if criterion.priority not in {"highest", "high"}:
            continue
        values = [
            (
                candidate_names.get(candidate_id, candidate_id),
                scores.get(criterion.criterion_id),
            )
            for candidate_id, scores in score_maps.items()
            if scores.get(criterion.criterion_id) is not None
        ]
        if values:
            comparison = "，".join(
                f"{name} {score:.2f}"
                for name, score in values
            )
            rationales.append(
                f"{criterion.priority} 场景指标“{criterion.name}”：{comparison}。"
            )
        else:
            rationales.append(
                f"{criterion.priority} 场景指标“{criterion.name}”尚无完整评分。"
            )
    return rationales


def _risk_rationales(risk_items, candidate_id):
    rationales = []
    for risk in risk_items:
        if risk.get("candidate_id") != candidate_id:
            continue
        if risk.get("level") not in {"medium", "high", "unknown"}:
            continue
        basis = {
            "evidence": "证据支持",
            "inference": "模型推断",
            "to_verify": "待验证",
        }.get(risk.get("basis"), "待核实")
        detail = risk.get("impact") or "影响未明确"
        mitigation = risk.get("mitigation")
        text = (
            f"关键风险（{risk.get('category', 'unknown')}/{risk.get('level')}，"
            f"{basis}）：{detail}"
        )
        if mitigation:
            text += f"；缓解措施：{mitigation}"
        rationales.append(text + "。")
        if len(rationales) >= 3:
            break
    return rationales


def _score_references_are_relevant(state):
    evidence_records = state.get("evidence_records", [])
    for evaluation in state.get("evaluations", []):
        candidate_id = evaluation.get("candidate_id")
        for score in evaluation.get("criterion_scores", []) + evaluation.get(
            "scenario_scores",
            [],
        ):
            if score.get("score") is None:
                continue
            evidence_ids = score.get("evidence_ids", [])
            if not evidence_ids or any(
                not _has_scoreable_support(
                    evidence_records,
                    candidate_id,
                    score.get("criterion"),
                    evidence_id,
                )
                for evidence_id in evidence_ids
            ):
                return False
    return True


def _deterministic_review_reasons(state):
    recommendation = state.get("recommendation", {})
    decision = recommendation.get("decision")
    recommended_id = recommendation.get("recommended_candidate_id")
    candidate_ids = {
        candidate.get("candidate_id")
        for candidate in state.get("feasible_candidates", [])
    }
    reasons = []
    if recommended_id and recommended_id not in candidate_ids:
        reasons.append("推荐方案不在可行候选列表中。")
    if decision == "conditional" and not recommended_id:
        reasons.append("条件性结论缺少结构化首选候选。")
    if decision == "conditional" and recommendation.get("decision_basis") != "scenario_criteria":
        reasons.append("条件性结论缺少最高优先级场景指标决策依据。")
    if decision == "conditional":
        requirements = SelectionRequirements.model_validate(
            state.get("requirements", {})
        )
        scenario_winner, _ = _highest_priority_scenario_winner(
            requirements,
            state.get("evaluations", []),
            candidate_ids,
        )
        if not scenario_winner or recommended_id != scenario_winner:
            reasons.append("综合分未计算，且没有一致的最高优先级场景指标依据。")
    weighted_winner = _unique_weighted_winner(
        state.get("evaluations", []),
        candidate_ids,
    )
    if decision == "recommend" and (
        not weighted_winner or recommended_id != weighted_winner
    ):
        reasons.append("无条件推荐与完整固定指标综合分排名不一致。")
    if decision in {"recommend", "conditional"} and not recommended_id:
        reasons.append("推荐文字没有可审计的结构化首选对象。")
    if not _score_references_are_relevant(state):
        reasons.append("存在缺少相关证据支持的数值评分。")

    rationale = " ".join(recommendation.get("rationale", []))
    if recommended_id:
        for candidate in state.get("feasible_candidates", []):
            other_id = candidate.get("candidate_id")
            other_name = candidate.get("name", "")
            if other_id == recommended_id:
                continue
            if (
                (other_id and other_id.casefold() in rationale.casefold())
                or (other_name and other_name.casefold() in rationale.casefold())
            ) and any(
                marker in rationale.casefold()
                for marker in ("推荐", "首选", "更优", "优于", "recommend", "preferred")
            ):
                reasons.append("结构化首选与推荐文字可能冲突。")
                break
    if decision == "insufficient_evidence":
        candidate_terms = [
            str(value).casefold()
            for candidate in state.get("feasible_candidates", [])
            for value in (candidate.get("candidate_id", ""), candidate.get("name", ""))
            if str(value).strip()
        ]
        candidate_text = " ".join(
            recommendation.get("rationale", [])
            + recommendation.get("prerequisites", [])
        ).casefold()
        if recommended_id or recommendation.get("prerequisites") or any(
            term in candidate_text for term in candidate_terms
        ):
            reasons.append("证据不足结论仍暗示了候选方案或采用前提。")
    return list(dict.fromkeys(reasons))

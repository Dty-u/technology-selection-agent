"""Evaluation, risk, recommendation, review, and report nodes."""

import json

from langchain_core.prompts import ChatPromptTemplate

from src.config.settings import (
    DECISION_EVIDENCE_CARD_MAX_CHARACTERS,
    MAX_EVIDENCE_ROUNDS,
    MAX_DECISION_EVIDENCE_CARDS_PER_CRITERION,
    MAX_REVIEW_ROUNDS,
)
from src.infrastructure.llm.chat_model import selection_llm
from src.workflows.nodes.selection_requirements import parse_json_object
from src.workflows.selection_cache import (
    clear_selection_cache,
    get_selection_cache,
    set_selection_cache,
)
from src.workflows.technology_selection_state import (
    CRITERIA,
    CandidateEvaluation,
    CriterionScore,
    RecommendationResult,
    RecommendationReview,
    RiskItem,
    SelectionRequirements,
)


evaluation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """你是半导体技术候选评价器。固定指标库包含 performance、power、thermal、cost、maturity、risk、scalability；只评价需求 active_criteria 中启用的固定指标，未启用项不要输出分数。场景指标来自需求中的 scenario_criteria，必须放在 scenario_scores 中，不能混入固定指标。
performance 只评价带宽、吞吐、频率、延迟或生产效率；thermal 单独评价散热、温度和热管理。每项按1到5分评价，5分始终代表更适合。
每项数值分必须引用对该候选和该指标具有 direct 或 indirect 相关支持的 evidence_id。direct 可给中高置信度，indirect 应给低或中置信度；无相关证据时 score 必须为 null。
只返回JSON：{{"evaluations":[{{"candidate_id":"id","criterion_scores":[{{"criterion":"performance","score":4,"rationale":"依据","evidence_ids":["E1"],"confidence":"medium"}}],"scenario_scores":[{{"criterion":"scenario_id","score":5,"rationale":"依据","evidence_ids":["E2"],"confidence":"high"}}]}}]}}。""",
        ),
        (
            "human",
            "需求：{requirements}\n候选：{candidates}\n证据：{evidence}\n"
            "上一轮审查反馈：{review_feedback}",
        ),
    ]
)
candidate_evaluation_chain = evaluation_prompt | selection_llm


risk_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """分析候选方案的技术集成、量产成熟度、可靠性、供应链和扩展风险。
level 只能是 low、medium、high 或 unknown；无法判断风险等级时用 unknown。
证据支持的风险填写 basis=evidence；推断填写 inference；缺信息填写 to_verify。
只返回JSON：{{"risk_items":[{{"candidate_id":"id","category":"integration","level":"medium","impact":"影响","mitigation":"缓解建议","evidence_ids":["E1"],"basis":"evidence"}}]}}。""",
        ),
        (
            "human",
            "需求：{requirements}\n候选评价：{evaluations}\n证据：{evidence}",
        ),
    ]
)
risk_analysis_chain = risk_prompt | selection_llm


recommendation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """根据需求、候选评价和风险生成技术选型结论。允许 recommend、conditional、insufficient_evidence。
理由必须分别说明综合分、high/highest 场景指标和首选方案的关键风险；不要把推断写成事实。
上一轮审查反馈不为空时，必须逐项修正反馈指出的不一致。
只返回JSON：{{"decision":"conditional","recommended_candidate_id":"id","alternative_candidate_ids":[],"rationale":[],"prerequisites":[],"unresolved_items":[]}}。""",
        ),
        (
            "human",
            "需求：{requirements}\n候选：{candidates}\n评价：{evaluations}\n风险：{risks}\n"
            "证据缺口：{evidence_gaps}\n上一轮审查反馈：{review_feedback}",
        ),
    ]
)
recommendation_chain = recommendation_prompt | selection_llm


review_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """审查技术推荐是否违反硬约束、是否与评分和风险一致、是否把推断写成事实。
综合分以输入中的 weighted_score 为准；单项分较低不代表综合分排名矛盾。
需要新证据并给出 requested_queries 时使用 evidence_gap；只有现有证据需要重新评分或改写结论时才使用 reevaluate。
只返回JSON：{{"verdict":"pass","reasons":[],"requested_queries":[]}}。
verdict只能是 pass、evidence_gap 或 reevaluate。""",
        ),
        (
            "human",
            "需求：{requirements}\n评价：{evaluations}\n风险：{risks}\n"
            "推荐：{recommendation}\n证据：{evidence}",
        ),
    ]
)
recommendation_review_chain = review_prompt | selection_llm

def clear_review_cache():
    """Clear process-local audit memoization for tests or host lifecycle hooks."""
    clear_selection_cache("review_recommendation")


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


def evaluate_candidates(state):
    """Validate evidence-backed criterion scores and calculate totals in Python."""
    cache_inputs = {
        "requirements": state.get("requirements", {}),
        "feasible_candidates": state.get("feasible_candidates", []),
        "evidence_records": state.get("evidence_records", []),
        "evidence_gaps": state.get("evidence_gaps", []),
        "review_feedback": _review_feedback(state),
    }
    cached = get_selection_cache(
        "evaluate_candidates",
        candidate_evaluation_chain,
        evaluation_prompt,
        cache_inputs,
    )
    if cached is not None:
        return cached
    response = candidate_evaluation_chain.invoke(
        {
            "requirements": json.dumps(
                cache_inputs["requirements"], ensure_ascii=False
            ),
            "candidates": json.dumps(
                cache_inputs["feasible_candidates"], ensure_ascii=False
            ),
            "evidence": json.dumps(
                _decision_evidence_cards(cache_inputs["evidence_records"]),
                ensure_ascii=False,
            ),
            "review_feedback": json.dumps(
                cache_inputs["review_feedback"],
                ensure_ascii=False,
            ),
        }
    )
    payload = parse_json_object(_response_content(response))
    raw_evaluations = {
        str(item.get("candidate_id", "")): item
        for item in payload.get("evaluations", [])
    }
    evidence_records = state.get("evidence_records", [])
    requirements = SelectionRequirements.model_validate(
        state.get("requirements", {})
    )
    active_criteria = _active_fixed_criteria(requirements)
    active_criterion_set = set(active_criteria)
    evaluations = []
    for candidate in state.get("feasible_candidates", []):
        candidate_id = candidate["candidate_id"]
        raw_scores = {
            str(item.get("criterion", "")): item
            for item in raw_evaluations.get(
                candidate_id,
                {},
            ).get("criterion_scores", [])
        }
        scores = []
        for criterion in CRITERIA:
            if criterion in active_criterion_set:
                scores.append(
                    _validated_score(
                        raw_scores.get(criterion),
                        candidate_id,
                        criterion,
                        evidence_records,
                    )
                )
            else:
                scores.append(
                    CriterionScore(
                        criterion=criterion,
                        rationale="本次未纳入评估",
                    )
                )
        raw_scenario_scores = {
            str(item.get("criterion", "")): item
            for item in raw_evaluations.get(
                candidate_id,
                {},
            ).get("scenario_scores", [])
        }
        scenario_scores = [
            _validated_score(
                raw_scenario_scores.get(scenario.criterion_id),
                candidate_id,
                scenario.criterion_id,
                evidence_records,
            )
            for scenario in requirements.scenario_criteria
        ]
        positive_criteria = {
            criterion
            for criterion, weight in requirements.priorities.items()
            if weight > 0 and criterion in active_criterion_set
        }
        complete = all(
            score.score is not None
            for score in scores
            if score.criterion in positive_criteria
        )
        weighted_score = None
        if complete and positive_criteria:
            weighted_score = round(
                sum(
                    score.score * requirements.priorities[score.criterion]
                    for score in scores
                    if score.criterion in positive_criteria
                ),
                2,
            )
        coverage = sum(
            score.score is not None
            for score in scores
            if score.criterion in active_criterion_set
        ) / len(active_criteria)
        scenario_coverage = (
            sum(score.score is not None for score in scenario_scores)
            / len(scenario_scores)
            if scenario_scores
            else 0
        )
        evaluations.append(
            CandidateEvaluation(
                candidate_id=candidate_id,
                criterion_scores=scores,
                scenario_scores=scenario_scores,
                weighted_score=weighted_score,
                evidence_coverage=coverage,
                scenario_coverage=scenario_coverage,
            ).model_dump()
        )
    candidate_names = {
        candidate["candidate_id"]: candidate.get("name", candidate["candidate_id"])
        for candidate in state.get("feasible_candidates", [])
    }
    gaps = list(state.get("evidence_gaps", []))
    for evaluation in evaluations:
        candidate_name = candidate_names.get(
            evaluation["candidate_id"],
            evaluation["candidate_id"],
        )
        score_items = [
            score
            for score in evaluation["criterion_scores"]
            if score["criterion"] in active_criterion_set
        ] + evaluation["scenario_scores"]
        for score in score_items:
            if score["score"] is None:
                gaps.append(
                    f"补充{candidate_name}在{score['criterion']}方面的可引用证据。"
                )
    result = {
        "evaluations": evaluations,
        "evidence_gaps": list(dict.fromkeys(gaps)),
    }
    set_selection_cache(
        "evaluate_candidates",
        candidate_evaluation_chain,
        evaluation_prompt,
        cache_inputs,
        result,
    )
    return result


def analyze_risks(state):
    """Return source-linked risk items and downgrade unsupported assertions."""
    cache_inputs = {
        "requirements": state.get("requirements", {}),
        "evaluations": state.get("evaluations", []),
        "evidence_records": state.get("evidence_records", []),
    }
    cached = get_selection_cache(
        "analyze_risks",
        risk_analysis_chain,
        risk_prompt,
        cache_inputs,
    )
    if cached is not None:
        return cached
    response = risk_analysis_chain.invoke(
        {
            "requirements": json.dumps(
                cache_inputs["requirements"],
                ensure_ascii=False,
            ),
            "evaluations": json.dumps(
                cache_inputs["evaluations"],
                ensure_ascii=False,
            ),
            "evidence": json.dumps(
                _decision_evidence_cards(cache_inputs["evidence_records"]),
                ensure_ascii=False,
            ),
        }
    )
    payload = parse_json_object(_response_content(response))
    evidence_candidates = {
        record["evidence_id"]: set(record.get("candidate_ids", []))
        for record in state.get("evidence_records", [])
    }
    risks = []
    for item in payload.get("risk_items", []):
        risk = RiskItem.model_validate(item)
        risk.evidence_ids = [
            evidence_id
            for evidence_id in risk.evidence_ids
            if risk.candidate_id in evidence_candidates.get(evidence_id, set())
        ]
        if risk.basis == "evidence" and not risk.evidence_ids:
            risk.basis = "to_verify"
        risks.append(risk.model_dump())
    result = {"risk_items": risks}
    set_selection_cache(
        "analyze_risks",
        risk_analysis_chain,
        risk_prompt,
        cache_inputs,
        result,
    )
    return result


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


def generate_recommendation(state):
    """Generate a conclusion while enforcing incomplete-evidence policy."""
    cache_inputs = {
        "requirements": state.get("requirements", {}),
        "feasible_candidates": state.get("feasible_candidates", []),
        "evaluations": state.get("evaluations", []),
        "risk_items": state.get("risk_items", []),
        "evidence_gaps": state.get("evidence_gaps", []),
        "review_feedback": _review_feedback(state),
    }
    cached = get_selection_cache(
        "generate_recommendation",
        recommendation_chain,
        recommendation_prompt,
        cache_inputs,
    )
    if cached is not None:
        return cached
    response = recommendation_chain.invoke(
        {
            "requirements": json.dumps(
                cache_inputs["requirements"],
                ensure_ascii=False,
            ),
            "candidates": json.dumps(
                cache_inputs["feasible_candidates"],
                ensure_ascii=False,
            ),
            "evaluations": json.dumps(
                cache_inputs["evaluations"],
                ensure_ascii=False,
            ),
            "risks": json.dumps(
                cache_inputs["risk_items"],
                ensure_ascii=False,
            ),
            "evidence_gaps": cache_inputs["evidence_gaps"],
            "review_feedback": json.dumps(
                cache_inputs["review_feedback"],
                ensure_ascii=False,
            ),
        }
    )
    recommendation = RecommendationResult.model_validate(
        parse_json_object(_response_content(response))
    )
    candidate_id_order = [
        candidate["candidate_id"]
        for candidate in state.get("feasible_candidates", [])
    ]
    candidate_ids = set(candidate_id_order)
    if recommendation.recommended_candidate_id not in candidate_ids:
        recommendation.recommended_candidate_id = None
        recommendation.decision = "insufficient_evidence"
    recommendation.alternative_candidate_ids = list(
        dict.fromkeys(
            candidate_id
            for candidate_id in recommendation.alternative_candidate_ids
            if candidate_id in candidate_ids
            and candidate_id != recommendation.recommended_candidate_id
        )
    )

    evaluations = state.get("evaluations", [])
    requirements = SelectionRequirements.model_validate(
        state.get("requirements", {})
    )
    weighted_winner = _unique_weighted_winner(evaluations, candidate_ids)
    scenario_winner, _ = _highest_priority_scenario_winner(
        requirements,
        evaluations,
        candidate_ids,
    )
    candidate_names = {
        candidate["candidate_id"]: candidate.get("name", candidate["candidate_id"])
        for candidate in state.get("feasible_candidates", [])
    }
    scenario_conflict = (
        scenario_winner
        and weighted_winner
        and scenario_winner != weighted_winner
    )
    if scenario_winner and (not weighted_winner or scenario_conflict):
        recommendation.decision = "conditional"
        recommendation.recommended_candidate_id = scenario_winner
        recommendation.decision_basis = "scenario_criteria"
        recommendation.rationale = [
            _weighted_score_rationale(evaluations, candidate_names),
            *_scenario_rationales(requirements, evaluations, candidate_names),
            *_risk_rationales(
                state.get("risk_items", []),
                scenario_winner,
            ),
        ]
        if not any(recommendation.rationale):
            recommendation.rationale = [
                "高优先级场景指标具有一致的相关证据评分。"
            ]
    elif weighted_winner:
        recommendation.decision = "recommend"
        recommendation.recommended_candidate_id = weighted_winner
        recommendation.decision_basis = "weighted_score"
        recommendation.rationale = [
            _weighted_score_rationale(evaluations, candidate_names),
            *_scenario_rationales(requirements, evaluations, candidate_names),
            *_risk_rationales(
                state.get("risk_items", []),
                weighted_winner,
            ),
        ]
    else:
        recommendation.decision = "insufficient_evidence"
        recommendation.recommended_candidate_id = None
        recommendation.alternative_candidate_ids = []
        recommendation.decision_basis = "insufficient_evidence"
        recommendation.rationale = [
            "现有相关证据不足以形成首选候选。"
        ]
        recommendation.prerequisites = []
    if recommendation.recommended_candidate_id:
        recommendation.alternative_candidate_ids = _ranked_alternative_ids(
            evaluations,
            candidate_id_order,
            recommendation.recommended_candidate_id,
        )
    # Verification gaps are derived from assessed evidence, not free-form model
    # prose. This prevents unrelated candidates or stale topics from leaking
    # into the final report.
    recommendation.unresolved_items = list(
        dict.fromkeys(state.get("evidence_gaps", []))
    )
    result = {"recommendation": recommendation.model_dump()}
    set_selection_cache(
        "generate_recommendation",
        recommendation_chain,
        recommendation_prompt,
        cache_inputs,
        result,
    )
    return result


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


def review_recommendation(state):
    """Review the recommendation and persist a pure-routing verdict."""
    recommendation = state.get("recommendation", {})
    deterministic_reasons = _deterministic_review_reasons(state)
    if deterministic_reasons:
        review = RecommendationReview(
            verdict="reevaluate",
            reasons=deterministic_reasons,
        )
    else:
        review_input = {
                "requirements": json.dumps(
                    state.get("requirements", {}),
                    ensure_ascii=False,
                ),
                "evaluations": json.dumps(
                    state.get("evaluations", []),
                    ensure_ascii=False,
                ),
                "risks": json.dumps(
                    state.get("risk_items", []),
                    ensure_ascii=False,
                ),
                "recommendation": json.dumps(
                    recommendation,
                    ensure_ascii=False,
                ),
                "evidence": json.dumps(
                    _decision_evidence_cards(
                        state.get("evidence_records", [])
                    ),
                    ensure_ascii=False,
                ),
        }
        cached = get_selection_cache(
            "review_recommendation",
            recommendation_review_chain,
            review_prompt,
            review_input,
        )
        if cached is not None:
            review = RecommendationReview.model_validate(cached)
        else:
            response = recommendation_review_chain.invoke(review_input)
            review = RecommendationReview.model_validate(
                parse_json_object(_response_content(response))
            )
            set_selection_cache(
                "review_recommendation",
                recommendation_review_chain,
                review_prompt,
                review_input,
                review.model_dump(),
            )
    review_round = state.get("review_round", 0) + 1
    result = {
        "recommendation_review": review.model_dump(),
        "review_round": review_round,
    }
    requested_queries = list(
        dict.fromkeys(
            query.strip()
            for query in review.requested_queries
            if query.strip()
        )
    )
    if review.verdict == "reevaluate" and requested_queries:
        review = review.model_copy(
            update={
                "verdict": "evidence_gap",
                "requested_queries": requested_queries,
            }
        )
        result["recommendation_review"] = review.model_dump()
    if review.verdict == "evidence_gap":
        result["missing_evidence_queries"] = requested_queries
        result["evidence_gaps"] = list(
            dict.fromkeys(state.get("evidence_gaps", []) + requested_queries)
        )
    warnings_to_add = []
    if (
        review.verdict == "evidence_gap"
        and state.get("evidence_round", 0) >= MAX_EVIDENCE_ROUNDS
    ):
        warnings_to_add.append(
            "证据补检达到上限，未解决缺口需要人工复核。"
        )
    if review.verdict != "pass" and review_round >= MAX_REVIEW_ROUNDS:
        warnings_to_add.append(
            "推荐审查达到上限，报告需要人工复核。"
        )
    if warnings_to_add:
        result["status"] = "needs_human_review"
        result["warnings"] = list(
            dict.fromkeys(
                state.get("warnings", [])
                + warnings_to_add
            )
        )
    return result


def _candidate_name(state, candidate_id):
    for candidate in state.get("feasible_candidates", []):
        if candidate.get("candidate_id") == candidate_id:
            return candidate.get("name", candidate_id)
    return candidate_id or "无"


CRITERION_LABELS = {
    "performance": "生产效率/性能",
    "power": "功耗与能效",
    "thermal": "散热与热管理",
    "cost": "成本",
    "maturity": "量产成熟度",
    "risk": "风险",
    "scalability": "扩展性",
}
DECISION_LABELS = {
    "recommend": "推荐",
    "conditional": "条件性推荐",
    "insufficient_evidence": "暂无足够证据形成推荐",
}
REVIEW_LABELS = {
    "pass": "审核通过",
    "evidence_gap": "需要补充证据",
    "reevaluate": "需要重新评估",
}
CONFIDENCE_LABELS = {
    "high": "高",
    "medium": "中",
    "low": "低",
}
PRIORITY_LABELS = {
    "highest": "最高",
    "high": "高",
    "medium": "中",
    "low": "低",
}


def _markdown_cell(value):
    return str(value or "").replace("|", "\\|").replace("\n", " ").strip()


def _report_criterion_labels(requirements):
    labels = dict(CRITERION_LABELS)
    labels.update(
        {
            str(item.get("criterion_id", "")): str(item.get("name", ""))
            for item in requirements.get("scenario_criteria", [])
            if str(item.get("criterion_id", "")).strip()
        }
    )
    return labels


def _humanize_report_text(value, criterion_labels):
    text = str(value or "").strip()
    for criterion in sorted(criterion_labels, key=len, reverse=True):
        text = text.replace(criterion, criterion_labels[criterion])
    return text


def _report_coverage(state):
    evaluations = state.get("evaluations", [])
    if not evaluations:
        return 0.0
    values = [float(item.get("evidence_coverage", 0) or 0) for item in evaluations]
    return round(sum(values) / len(values), 4)


def _report_confidence(state, coverage):
    recommendation = state.get("recommendation", {})
    review = state.get("recommendation_review", {})
    if (
        recommendation.get("decision") == "insufficient_evidence"
        or review.get("verdict") not in {None, "pass"}
        or state.get("status") == "needs_human_review"
    ):
        return "low"
    if coverage >= 0.8:
        return "high"
    return "medium" if coverage >= 0.5 else "low"


def _score_maps(state):
    result = {}
    for evaluation in state.get("evaluations", []):
        result[evaluation.get("candidate_id")] = {
            score.get("criterion"): score
            for score in (
                evaluation.get("criterion_scores", [])
                + evaluation.get("scenario_scores", [])
            )
        }
    return result


def _report_weights(requirements, active_criteria):
    weights = {}
    for criterion in active_criteria:
        try:
            weights[criterion] = max(
                float(requirements.get("priorities", {}).get(criterion, 0)),
                0,
            )
        except (TypeError, ValueError):
            weights[criterion] = 0
    total = sum(weights.values())
    if total <= 0 and active_criteria:
        return {
            criterion: 1 / len(active_criteria)
            for criterion in active_criteria
        }
    return {
        criterion: weight / total
        for criterion, weight in weights.items()
    }


def _comparison_rows(state, criterion_labels):
    requirements = state.get("requirements", {})
    candidates = state.get("feasible_candidates", [])
    active = list(requirements.get("active_criteria", []) or list(CRITERIA))
    normalized_weights = _report_weights(requirements, active)
    scenarios = [
        str(item.get("criterion_id", ""))
        for item in requirements.get("scenario_criteria", [])
        if str(item.get("criterion_id", "")).strip()
    ]
    scenario_priorities = {
        str(item.get("criterion_id", "")): PRIORITY_LABELS.get(
            item.get("priority"), "中"
        )
        for item in requirements.get("scenario_criteria", [])
    }
    score_maps = _score_maps(state)
    rows = []
    for criterion in active + scenarios:
        cells = []
        numeric_scores = []
        for candidate in candidates:
            score = score_maps.get(candidate.get("candidate_id"), {}).get(
                criterion, {}
            )
            value = score.get("score")
            if value is not None:
                numeric_scores.append((candidate.get("candidate_id"), value))
            cells.append(
                {
                    "candidate_id": candidate.get("candidate_id"),
                    "candidate_name": candidate.get("name", ""),
                    "score": value,
                    "rationale": _humanize_report_text(
                        score.get("rationale", ""), criterion_labels
                    ),
                    "evidence_ids": list(score.get("evidence_ids", [])),
                    "confidence": score.get("confidence", "low"),
                    "confidence_label": CONFIDENCE_LABELS.get(
                        score.get("confidence", "low"), "低"
                    ),
                }
            )
        if criterion in CRITERIA:
            weight = normalized_weights.get(criterion, 0)
            weight_label = f"{weight:.0%}" if weight > 0 else "0%"
        else:
            weight_label = scenario_priorities.get(criterion, "中")
        complete = len(numeric_scores) == len(candidates) and bool(candidates)
        if complete:
            best = max(value for _, value in numeric_scores)
            leaders = [candidate_id for candidate_id, value in numeric_scores if value == best]
            judgment = (
                _candidate_name(state, leaders[0]) + "领先"
                if len(leaders) == 1
                else "评分并列"
            )
            evidence_status = "证据充分"
        elif numeric_scores:
            judgment = "暂无法完整比较"
            evidence_status = "部分证据"
        else:
            judgment = "暂无法判断"
            evidence_status = "待补证"
        rows.append(
            {
                "criterion_id": criterion,
                "label": criterion_labels.get(criterion, criterion),
                "weight": weight_label,
                "cells": cells,
                "judgment": judgment,
                "evidence_status": evidence_status,
            }
        )
    return rows


_GAP_CRITERION_ALIASES = {
    "performance": ("性能", "效率", "吞吐", "均匀性", "保形性"),
    "power": ("功耗", "能效", "功率"),
    "thermal": ("热", "散热", "结温"),
    "cost": ("成本", "费用", "价格", "总拥有成本", "tco"),
    "maturity": ("成熟", "量产", "商业化", "产业化", "工艺认证"),
    "risk": ("风险", "可靠性", "供应链", "失效"),
    "scalability": ("扩展", "规模化", "产能", "大面积", "批量"),
}


def _gap_signature(value, state, criterion_labels):
    normalized = str(value or "").casefold()
    candidates = set()
    for candidate in state.get("feasible_candidates", []) or []:
        candidate_id = str(candidate.get("candidate_id", "")).strip()
        candidate_name = str(candidate.get("name", "")).strip()
        if any(
            token and token.casefold() in normalized
            for token in (candidate_id, candidate_name)
        ):
            candidates.add(candidate_id or candidate_name)

    criteria = set()
    for criterion, label in criterion_labels.items():
        aliases = [criterion, str(label)]
        aliases.extend(_GAP_CRITERION_ALIASES.get(criterion, ()))
        if any(alias and alias.casefold() in normalized for alias in aliases):
            criteria.add(criterion)
    return frozenset(candidates), frozenset(criteria)


def _is_semantic_gap_duplicate(signature, existing_signatures):
    candidates, criteria = signature
    if not candidates or not criteria:
        return False
    requested_pairs = {
        (candidate, criterion)
        for candidate in candidates
        for criterion in criteria
    }
    existing_pairs = {
        (candidate, criterion)
        for existing_candidates, existing_criteria in existing_signatures
        for candidate in existing_candidates
        for criterion in existing_criteria
    }
    return requested_pairs.issubset(existing_pairs)


def _deduplicated_gaps(state, criterion_labels):
    recommendation = state.get("recommendation", {})
    prioritized = []
    existing_descriptions = set()
    existing_signatures = []

    # Verified scoring gaps are the authoritative list. Recommendation prose may
    # restate several of these cells as one sentence, so merge it semantically.
    for priority, values, suppress_semantic_duplicates in (
        ("高", state.get("evidence_gaps", []), False),
        ("高", recommendation.get("unresolved_items", []), True),
    ):
        for value in values or []:
            rendered = _humanize_report_text(value, criterion_labels).rstrip(
                " 。；;"
            )
            if not rendered or rendered in existing_descriptions:
                continue
            signature = _gap_signature(rendered, state, criterion_labels)
            if (
                suppress_semantic_duplicates
                and _is_semantic_gap_duplicate(signature, existing_signatures)
            ):
                continue
            prioritized.append((priority, rendered))
            existing_descriptions.add(rendered)
            existing_signatures.append(signature)

    return [
        {"priority": priority, "description": description}
        for priority, description in prioritized
    ]


def _structured_evidence(state, criterion_labels):
    records = []
    for record in state.get("evidence_records", []):
        source_reliability = record.get("source_reliability", "internal")
        source_label = {
            "internal": "内部知识库",
            "web_identified": "有来源网页",
            "web_unattributed": "来源未验证",
        }.get(source_reliability, "内部知识库")
        url = str(record.get("url", "")).strip()
        supports = []
        for support in record.get("supports", []):
            supports.append(
                {
                    "candidate_id": support.get("candidate_id", ""),
                    "candidate_name": _candidate_name(
                        state, support.get("candidate_id")
                    ),
                    "criterion_id": support.get("criterion", ""),
                    "criterion_label": criterion_labels.get(
                        support.get("criterion"), support.get("criterion", "")
                    ),
                    "level": support.get("level", "none"),
                    "level_label": {
                        "direct": "直接证据",
                        "indirect": "间接证据",
                        "none": "无支持关系",
                    }.get(support.get("level"), "无支持关系"),
                    "quote": str(support.get("quote", "")).strip(),
                }
            )
        excerpt = next(
            (item["quote"] for item in supports if item["quote"]),
            str(record.get("content", "")).strip()[:240],
        )
        records.append(
            {
                "evidence_id": record.get("evidence_id", ""),
                "title": record.get("title") or "未命名证据",
                "source_type": source_reliability,
                "source_label": source_label,
                "display_source": url if url else source_label,
                "url": url,
                "excerpt": excerpt,
                "supports": supports,
            }
        )
    return records


def build_report_data(state):
    """Build the compact public view model shared by UI and Markdown."""
    requirements = state.get("requirements", {})
    recommendation = state.get("recommendation", {})
    review = state.get("recommendation_review", {})
    criterion_labels = _report_criterion_labels(requirements)
    coverage = _report_coverage(state)
    confidence = _report_confidence(state, coverage)
    active_ids = list(requirements.get("active_criteria", []) or list(CRITERIA))
    normalized_weights = _report_weights(requirements, active_ids)
    inactive_ids = [criterion for criterion in CRITERIA if criterion not in active_ids]
    primary_candidate_id = recommendation.get("recommended_candidate_id")
    summary = _humanize_report_text(
        next(
            iter(recommendation.get("rationale", []) or []),
            "现有证据不足以形成可审计的首选方案。",
        ),
        criterion_labels,
    )
    comparison_rows = _comparison_rows(state, criterion_labels)
    risks = [
        {
            **risk,
            "candidate_name": _candidate_name(state, risk.get("candidate_id")),
            "basis_label": {
                "evidence": "相关证据",
                "inference": "推断",
                "to_verify": "待验证",
            }.get(risk.get("basis"), "待验证"),
        }
        for risk in state.get("risk_items", [])
    ]
    return {
        "schema_version": "1.0",
        "report_version": "2.0",
        "decision": {
            "status": recommendation.get("decision", "insufficient_evidence"),
            "status_label": DECISION_LABELS.get(
                recommendation.get("decision"), "待判断"
            ),
            "primary_candidate_id": primary_candidate_id,
            "primary_candidate_name": _candidate_name(
                state, primary_candidate_id
            ),
            "summary": summary,
            "decision_basis": recommendation.get(
                "decision_basis", "insufficient_evidence"
            ),
            "confidence": confidence,
            "confidence_label": CONFIDENCE_LABELS[confidence],
            "evidence_coverage": coverage,
            "requires_human_review": state.get("status") == "needs_human_review",
            "rationale": [
                _humanize_report_text(item, criterion_labels)
                for item in recommendation.get("rationale", [])
            ],
            "prerequisites": list(recommendation.get("prerequisites", [])),
            "unresolved_items": [
                _humanize_report_text(item, criterion_labels)
                for item in recommendation.get("unresolved_items", [])
            ],
        },
        "requirements": {
            "business_goal": requirements.get("business_goal", ""),
            "application_scenario": requirements.get(
                "application_scenario", ""
            ),
            "hard_constraints": list(requirements.get("hard_constraints", [])),
            "active_criteria": [
                {
                    "criterion_id": criterion,
                    "label": criterion_labels.get(criterion, criterion),
                    "weight": float(
                        normalized_weights.get(criterion, 0)
                    ),
                }
                for criterion in active_ids
            ],
            "inactive_criteria": [
                {
                    "criterion_id": criterion,
                    "label": criterion_labels.get(criterion, criterion),
                }
                for criterion in inactive_ids
            ],
            "scenario_criteria": [
                {
                    "criterion_id": item.get("criterion_id", ""),
                    "label": item.get("name", ""),
                    "priority": item.get("priority", "medium"),
                    "priority_label": PRIORITY_LABELS.get(
                        item.get("priority"), "中"
                    ),
                }
                for item in requirements.get("scenario_criteria", [])
            ],
        },
        "candidates": [
            {
                "candidate_id": candidate.get("candidate_id", ""),
                "name": candidate.get("name", ""),
                "summary": candidate.get("summary", ""),
            }
            for candidate in state.get("feasible_candidates", [])
        ],
        "comparison": {
            "candidates": [
                {
                    "candidate_id": candidate.get("candidate_id", ""),
                    "name": candidate.get("name", ""),
                }
                for candidate in state.get("feasible_candidates", [])
            ],
            "criteria": comparison_rows,
        },
        "risks": risks,
        "evidence_gaps": _deduplicated_gaps(state, criterion_labels),
        "evidence": _structured_evidence(state, criterion_labels),
        "audit": {
            "status": state.get("status", "completed"),
            "review_verdict": review.get("verdict"),
            "review_label": REVIEW_LABELS.get(
                review.get("verdict"), "未审核"
            ),
            "review_reasons": list(review.get("reasons", [])),
            "evidence_round": state.get("evidence_round", 0),
            "review_round": state.get("review_round", 0),
            "warnings": list(state.get("warnings", [])),
        },
    }


def generate_report(state):
    """Render a standalone, user-facing Markdown decision report."""
    report_data = build_report_data(state)
    requirements = state.get("requirements", {})
    recommendation = state.get("recommendation", {})
    review = state.get("recommendation_review", {})
    criterion_labels = _report_criterion_labels(requirements)
    coverage = report_data["decision"]["evidence_coverage"]
    confidence = report_data["decision"]["confidence"]
    candidates = state.get("feasible_candidates", [])
    candidate_names = [candidate["name"] for candidate in report_data["candidates"]]
    comparison_rows = report_data["comparison"]["criteria"]
    primary_name = report_data["decision"]["primary_candidate_name"]
    summary = report_data["decision"]["summary"]
    active = [
        item["criterion_id"]
        for item in report_data["requirements"]["active_criteria"]
    ]
    inactive = [
        item["criterion_id"]
        for item in report_data["requirements"]["inactive_criteria"]
    ]
    lines = [
        "# 半导体技术选型报告",
        "",
        "## 报告信息",
        "",
        "| 项目 | 内容 |",
        "|---|---|",
        f"| 选型主题 | {_markdown_cell(requirements.get('business_goal', ''))} |",
        f"| 结论状态 | {DECISION_LABELS.get(recommendation.get('decision'), '待判断')} |",
        f"| 审核状态 | {REVIEW_LABELS.get(review.get('verdict'), '未审核')} |",
        f"| 结论可信度 | {CONFIDENCE_LABELS[confidence]} |",
        f"| 固定指标证据覆盖率 | {coverage:.0%} |",
        "| 报告版本 | V2.0 |",
        "",
        "## 1. 执行摘要",
        "",
        f"- 结论：{DECISION_LABELS.get(recommendation.get('decision'), '待判断')}",
        f"- 首选方案：{primary_name}",
        f"- 一句话原因：{summary}",
        "- 关键限制："
        + (
            f"存在{len(recommendation.get('unresolved_items', []))}"
            "项可能影响结论的待验证事项，详见第7节。"
            if recommendation.get("unresolved_items")
            else "无"
        ),
        "- 是否需要人工复核："
        + ("是" if state.get("status") == "needs_human_review" else "否"),
        "",
        "## 2. 需求与评估范围",
        "",
        f"- 业务目标：{requirements.get('business_goal', '')}",
        f"- 应用场景：{requirements.get('application_scenario', '')}",
        "- 候选方案：" + ("、".join(candidate_names) or "无"),
        "- 本次启用指标："
        + "、".join(criterion_labels.get(item, item) for item in active),
        "- 本次未纳入指标："
        + (
            "、".join(criterion_labels.get(item, item) for item in inactive)
            or "无"
        ),
        "- 硬约束："
        + ("；".join(requirements.get("hard_constraints", [])) or "未提供"),
        "",
        "### 评分说明",
        "",
        "- 评分范围为1至5分，5分表示在当前场景下更适合。",
        "- `—`表示因证据不足而未评分，不代表0分。",
        "- “本次未纳入”表示该指标权重为0，未参与检索、评分和综合分计算。",
        "",
        "## 3. 方案对比总表",
        "",
    ]
    lines.append(
        "| 指标 | 权重 | "
        + " | ".join(_markdown_cell(name) for name in candidate_names)
        + " | 当前判断 | 证据状态 |"
    )
    lines.append("|---|---:" + "|---:" * len(candidate_names) + "|---|---|")
    for row in comparison_rows:
        values = [
            str(score.get("score")) if score.get("score") is not None else "—"
            for score in row["cells"]
        ]
        lines.append(
            f"| {_markdown_cell(row['label'])} | {row['weight']} | "
            + " | ".join(values)
            + f" | {row['judgment']} | {row['evidence_status']} |"
        )

    lines.extend(["", "## 4. 关键指标分析"])
    for row in comparison_rows:
        lines.extend(["", f"### {row['label']}"])
        for candidate, score in zip(candidates, row["cells"]):
            value = score.get("score")
            references = " ".join(
                f"[{evidence_id}]" for evidence_id in score.get("evidence_ids", [])
            )
            rationale = _humanize_report_text(
                score.get("rationale", "暂无可引用证据"),
                criterion_labels,
            )
            lines.append(
                f"- {candidate.get('name', '')}："
                f"{value if value is not None else '未评分'}；"
                f"{rationale} {references}".rstrip()
            )
        lines.append(f"- 对比结论：{row['judgment']}。")

    lines.extend(
        [
            "",
            "## 5. 推荐结论与适用边界",
            "",
            f"- 结论：{DECISION_LABELS.get(recommendation.get('decision'), '待判断')}",
            f"- 首选方案：{primary_name}",
            "- 推荐依据："
            + (
                "；".join(
                    _humanize_report_text(item, criterion_labels)
                    for item in recommendation.get("rationale", [])
                )
                or "无"
            ),
            "- 推荐成立条件："
            + ("；".join(recommendation.get("prerequisites", [])) or "无"),
        ]
    )

    lines.extend(["", "## 6. 风险与应对措施", ""])
    if report_data["risks"]:
        lines.extend(
            [
                "| 方案 | 风险 | 等级 | 依据类型 | 影响 | 建议措施 | 证据 |",
                "|---|---|---|---|---|---|---|",
            ]
        )
        for risk in report_data["risks"]:
            references = " ".join(
                f"[{evidence_id}]"
                for evidence_id in risk.get("evidence_ids", [])
            )
            lines.append(
                f"| {_markdown_cell(risk.get('candidate_name'))} "
                f"| {_markdown_cell(risk.get('category'))} "
                f"| {_markdown_cell(risk.get('level'))} | {risk.get('basis_label')} "
                f"| {_markdown_cell(risk.get('impact'))} "
                f"| {_markdown_cell(risk.get('mitigation'))} "
                f"| {references or '无'} |"
            )
    else:
        lines.append("- 暂无可验证的风险条目。")

    lines.extend(["", "## 7. 待验证事项", ""])
    gaps = report_data["evidence_gaps"]
    if gaps:
        for gap in gaps:
            lines.append(f"- [{gap['priority']}优先级] {gap['description']}。")
    else:
        lines.append("- 无。")

    lines.extend(["", "## 8. 证据索引", ""])
    for record in report_data["evidence"]:
        support_labels = []
        for support in record.get("supports", []):
            support_labels.append(
                f"{support.get('candidate_name')}/"
                f"{support.get('criterion_label')}/"
                f"{support.get('level_label')}"
            )
        source = (
            f"[查看来源]({record['url']})"
            if record.get("url")
            else record.get("display_source", "来源未验证")
        )
        lines.extend(
            [
                f"### [{record.get('evidence_id')}] {record.get('title') or '未命名证据'}",
                f"- 来源类型：{record.get('source_label')}",
                f"- 来源：{source}",
                "- 支持关系：" + ("；".join(support_labels) or "未标注"),
                f"- 原文片段：{record.get('excerpt') or '未提供'}",
                "",
            ]
        )

    lines.extend(["## 9. 审核说明", ""])
    lines.extend(
        [
            f"- 审核结果：{REVIEW_LABELS.get(review.get('verdict'), '未审核')}",
            f"- 证据检索轮次：{state.get('evidence_round', 0)}",
            f"- 推荐审核轮次：{state.get('review_round', 0)}",
            "- 审核原因：" + ("；".join(review.get("reasons", [])) or "无"),
            "- 警告：" + ("；".join(state.get("warnings", [])) or "无"),
        ]
    )

    status = (
        "needs_human_review"
        if state.get("status") == "needs_human_review"
        else "completed"
    )
    return {
        "report": "\n".join(lines),
        "report_data": report_data,
        "status": status,
    }

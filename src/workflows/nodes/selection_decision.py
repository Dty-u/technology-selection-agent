"""Evaluation, risk, recommendation, and review nodes."""

import json
import logging

from langchain_core.prompts import ChatPromptTemplate

from src.config.settings import MAX_EVIDENCE_ROUNDS, MAX_REVIEW_ROUNDS
from src.infrastructure.llm.chat_model import selection_llm
from src.workflows.nodes.selection_requirements import parse_json_object
from src.workflows.nodes.selection_scoring import (
    _active_fixed_criteria,
    _decision_evidence_cards,
    _deterministic_review_reasons,
    _highest_priority_scenario_winner,
    _ranked_alternative_ids,
    _response_content,
    _review_feedback,
    _risk_rationales,
    _scenario_rationales,
    _unique_weighted_winner,
    _validated_risk_items,
    _validated_score,
    _weighted_score_rationale,
)
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
    SelectionRequirements,
)


logger = logging.getLogger(__name__)


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
            "system",
            "在同一次响应中同时返回 risk_items。risk_items 每项必须包含 "
            "candidate_id、category、level、impact、mitigation、evidence_ids、basis；"
            "basis 只能是 evidence、inference 或 to_verify 之一；不要把风险说明、"
            "证据摘要或其他自然语言填入 basis，应放入 impact 或 mitigation。"
            "最终 JSON 顶层结构为 evaluations 和 risk_items。",
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


risk_repair_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Repair the JSON risk_items array to satisfy the required schema. "
            "Return JSON only in the form {\"risk_items\":[...]}. Preserve the "
            "candidate_id, category, level, impact, mitigation, and evidence_ids "
            "when they are already valid. The basis field must be exactly one of "
            "evidence, inference, or to_verify. Do not put explanations in basis; "
            "place them in impact or mitigation instead.",
        ),
        (
            "human",
            "Validation error: {validation_error}\n"
            "Risk JSON to repair: {risk_items}",
        ),
    ]
)
risk_repair_chain = risk_repair_prompt | selection_llm


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
        "precomputed_risk_items": payload.get("risk_items"),
    }
    set_selection_cache(
        "evaluate_candidates",
        candidate_evaluation_chain,
        evaluation_prompt,
        cache_inputs,
        result,
    )
    return result


def _repair_precomputed_risk_items(raw_items, validation_error):
    """Ask once for a schema-only repair before recomputing risk analysis."""
    response = risk_repair_chain.invoke(
        {
            "validation_error": str(validation_error),
            "risk_items": json.dumps(raw_items, ensure_ascii=False),
        }
    )
    payload = parse_json_object(_response_content(response))
    return payload.get("risk_items")


def analyze_risks(state):
    """Return source-linked risk items and downgrade unsupported assertions."""
    cache_inputs = {
        "requirements": state.get("requirements", {}),
        "evaluations": state.get("evaluations", []),
        "evidence_records": state.get("evidence_records", []),
        "precomputed_risk_items": state.get("precomputed_risk_items"),
    }
    cached = get_selection_cache(
        "analyze_risks",
        risk_analysis_chain,
        risk_prompt,
        cache_inputs,
    )
    if cached is not None:
        return cached
    if "precomputed_risk_items" in state:
        try:
            risks = _validated_risk_items(
                state.get("precomputed_risk_items"),
                state.get("evidence_records", []),
            )
            result = {"risk_items": risks}
            set_selection_cache(
                "analyze_risks",
                risk_analysis_chain,
                risk_prompt,
                cache_inputs,
                result,
            )
            return result
        except (TypeError, ValueError) as exc:
            if isinstance(state.get("precomputed_risk_items"), list):
                logger.info(
                    "Precomputed risk assessment is invalid; attempting repair: %s",
                    exc,
                )
                try:
                    repaired_items = _repair_precomputed_risk_items(
                        state.get("precomputed_risk_items"),
                        exc,
                    )
                    risks = _validated_risk_items(
                        repaired_items,
                        state.get("evidence_records", []),
                    )
                    result = {"risk_items": risks}
                    set_selection_cache(
                        "analyze_risks",
                        risk_analysis_chain,
                        risk_prompt,
                        cache_inputs,
                        result,
                    )
                    return result
                except (TypeError, ValueError) as repair_exc:
                    logger.warning(
                        "Precomputed risk repair is invalid; using risk chain: %s",
                        repair_exc,
                    )
            else:
                logger.warning(
                    "Precomputed risk assessment is invalid; using risk chain: %s",
                    exc,
                )
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
    risks = _validated_risk_items(
        payload.get("risk_items"),
        state.get("evidence_records", []),
    )
    result = {"risk_items": risks}
    set_selection_cache(
        "analyze_risks",
        risk_analysis_chain,
        risk_prompt,
        cache_inputs,
        result,
    )
    return result


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
    fallback_warning = ""
    try:
        recommendation = RecommendationResult.model_validate(
            parse_json_object(_response_content(response))
        )
    except ValueError as exc:
        # The model only supplies explanatory text here. The final candidate is
        # derived below from assessed evidence, so malformed JSON must not
        # discard a completed evidence and scoring workflow.
        logger.warning(
            "Recommendation JSON was invalid; using evidence-score fallback: %s",
            type(exc).__name__,
        )
        recommendation = RecommendationResult(
            decision="insufficient_evidence",
        )
        fallback_warning = (
            "推荐结论模型返回格式异常，已根据已验证评分生成保守结论。"
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
    if fallback_warning:
        result["warnings"] = list(
            dict.fromkeys(
                [*state.get("warnings", []), fallback_warning]
            )
        )
    set_selection_cache(
        "generate_recommendation",
        recommendation_chain,
        recommendation_prompt,
        cache_inputs,
        result,
    )
    return result


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

"""Build structured report data and Markdown for technology selection results."""

from src.workflows.technology_selection_state import CRITERIA


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
        "conversation": {
            "user_request": state.get("user_request", ""),
            "clarification_answers": list(
                state.get("clarification_answers", []) or []
            ),
            "clarification_history": list(
                state.get("clarification_history", []) or []
            ),
        },
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
        f"| 原始需求 | {_markdown_cell(state.get('user_request', ''))} |",
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
            "项可能影响结论的待验证事项。"
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
        "### 澄清记录",
        "",
    ]
    clarification_history = list(state.get("clarification_history", []) or [])
    if clarification_history:
        for round_index, round_item in enumerate(
            clarification_history,
            start=1,
        ):
            questions = "；".join(
                str(question)
                for question in round_item.get("questions", [])
                if str(question).strip()
            )
            answer = _markdown_cell(round_item.get("answer", ""))
            lines.append(f"- 第{round_index}轮问题：{_markdown_cell(questions)}")
            lines.append(f"  回答：{answer}")
    else:
        lines.append("- 本轮未发起需求澄清。")
    lines.append("")
    lines.append("### 评分说明")
    lines.append("")
    lines.append("- 评分范围为1至5分，5分表示在当前场景下更适合。")
    lines.append("- `—`表示因证据不足而未评分，不代表0分。")
    lines.append("- “本次未纳入”表示该指标权重为0，未参与检索、评分和综合分计算。")
    lines.append("")
    lines.append("## 3. 方案对比总表")
    lines.append("")
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

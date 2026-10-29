"""Evidence planning and retrieval nodes for technology selection."""

import json
import re
from concurrent.futures import ThreadPoolExecutor

from langchain_core.prompts import ChatPromptTemplate

from src.config.settings import (
    MAX_EVIDENCE_ASSESSMENT_CONCURRENCY,
    MAX_EVIDENCE_TASK_CONCURRENCY,
    SELECTION_RETRIEVAL_CACHE_TTL_SECONDS,
    SELECTION_WEB_SEARCH_ENGINE,
    SELECTION_WEB_SEARCH_MAX_RESULTS,
)
from src.infrastructure.llm.chat_model import selection_json_llm, selection_llm
from src.observability.logging import log
from src.observability.workflow_progress import progress_stage
from src.workflows.nodes.selection_evidence_rules import (
    MANUFACTURING_CRITERIA,
    TECHNICAL_CRITERIA,
    _EXTERNAL_RISK_HINT_PATTERN,
    _WEB_EVIDENCE_HINT_PATTERN,
    _assessment_batches,
    _candidate_ids_from_query,
    _candidate_search_label,
    _compact_evidence_card,
    _criteria_from_query,
    _criterion_search_label,
    _deterministic_web_supports,
    _evidence_id,
    _required_criterion_ids,
    _retrieval_sources,
    _source_reliability,
    _supplement_source,
    _unwrap_graph_content,
    _validated_supports,
)
from src.workflows.nodes.selection_requirements import parse_json_object
from src.workflows.retrieval_graph import retrieval_graph
from src.workflows.selection_cache import (
    get_selection_cache,
    set_selection_cache,
)
from src.workflows.technology_selection_state import (
    CRITERIA,
    EvidenceRecord,
    EvidenceTask,
)


evidence_source_planning_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """为每个技术选型证据任务独立选择检索来源。
source 只能是 local 或 web：
- local：半导体技术原理、性能机制、器件结构、适用条件和长期扩展性等稳定知识。
- web：当前或近期的成本、量产状态、良率、供应商路线图和供应链状态等时效信息。
按每个任务的主要证据需求判断，不要让其他任务或原始问题中的单个关键词影响当前任务。
宽泛的技术比较默认选择 local；只有主要依赖当前外部事实时才选择 web。
只返回JSON：{{"task_sources":[{{"task_id":"任务ID","source":"local"}}]}}。""",
        ),
        ("human", "证据任务：{tasks}"),
    ]
)
evidence_source_planning_chain = evidence_source_planning_prompt | selection_llm


evidence_assessment_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """判断每条证据对“候选方案 × 指标”的逐项相关关系。指标只能来自给定指标目录。
direct 表示正文有可逐字核验的明确支持；indirect 表示正文与该候选和指标明确相关。direct 和 indirect 都可用于后续评分；不相关才标为 none。
不要因为检索问题提到了某个指标，就认定文档支持该指标。只返回JSON：
{{"evidence_map":[{{"evidence_id":"E1","supports":[{{"candidate_id":"id","criterion":"performance","level":"direct","quote":"正文原文片段"}}]}}],"missing_queries":[]}}。""",
        ),
        (
            "human",
            "候选方案：{candidates}\n指标目录：{criteria_catalog}\n证据：{evidence}",
        ),
    ]
)
evidence_assessment_chain = evidence_assessment_prompt | selection_json_llm


def _assess_evidence_batch(candidates, records, criteria_catalog=None):
    criteria_catalog = criteria_catalog or [
        {"criterion_id": criterion, "name": criterion}
        for criterion in CRITERIA
    ]
    cache_inputs = {
        "candidates": json.dumps(candidates, ensure_ascii=False),
        "criteria_catalog": json.dumps(criteria_catalog, ensure_ascii=False),
        "evidence": json.dumps(
            [_compact_evidence_card(record) for record in records],
            ensure_ascii=False,
        ),
    }
    cached = get_selection_cache(
        "assess_evidence_batch",
        evidence_assessment_chain,
        evidence_assessment_prompt,
        cache_inputs,
    )
    if cached is not None:
        return cached
    payload = None
    for attempt in range(1, 3):
        response = evidence_assessment_chain.invoke(cache_inputs)
        content = response.content if hasattr(response, "content") else str(response)
        try:
            payload = parse_json_object(content)
            break
        except ValueError as error:
            details = {
                "尝试": f"{attempt}/2",
                "异常类型": type(error).__name__,
                "响应字符数": len(content),
            }
            if isinstance(error, json.JSONDecodeError):
                details.update({"行": error.lineno, "列": error.colno})
            log.warning(
                "[技术选型][evidence_assessment] 模型JSON解析失败 | {}",
                " | ".join(f"{key}={value}" for key, value in details.items()),
            )
            if attempt == 2:
                raise
    result = {
        str(item.get("evidence_id", "")): item
        for item in payload.get("evidence_map", [])
        if isinstance(item, dict)
    }
    set_selection_cache(
        "assess_evidence_batch",
        evidence_assessment_chain,
        evidence_assessment_prompt,
        cache_inputs,
        result,
    )
    return result


def _assess_evidence_batch_safely(candidates, records, criteria_catalog):
    try:
        return records, _assess_evidence_batch(
            candidates,
            records,
            criteria_catalog,
        ), None
    except Exception as error:
        return records, {}, error


def _assign_source_preferences(tasks):
    """Route stable technical evidence locally and current facts to Web."""
    for task in tasks:
        if task.get("source_preference") in {"local", "web"}:
            continue
        criteria = {
            str(criterion).strip()
            for criterion in task.get("criteria", [])
            if str(criterion).strip()
        }
        task["source_preference"] = (
            "web"
            if criteria and criteria.issubset(MANUFACTURING_CRITERIA)
            else "local"
        )
    return tasks

    # Kept below temporarily for source-compatible patch locality.
    """Use one bounded model call to route all evidence tasks."""
    if not tasks:
        return tasks
    try:
        with progress_stage(
            "技术选型",
            "source_planning",
            "规划证据检索来源",
            details={"任务": len(tasks)},
        ) as stage:
            response = evidence_source_planning_chain.invoke(
                {
                    "tasks": json.dumps(
                        [
                            {
                                "task_id": task["task_id"],
                                "query": task["query"],
                                "criteria": task["criteria"],
                            }
                            for task in tasks
                        ],
                        ensure_ascii=False,
                    )
                }
            )
            content = (
                response.content
                if hasattr(response, "content")
                else str(response)
            )
            payload = parse_json_object(content)
            source_by_task_id = {}
            for item in payload.get("task_sources", []):
                if not isinstance(item, dict):
                    continue
                source = str(
                    item.get("source", item.get("source_preference", ""))
                ).strip().lower()
                if source in {"local", "auto"}:
                    source_by_task_id[str(item.get("task_id", ""))] = "auto"
                elif source == "web":
                    source_by_task_id[str(item.get("task_id", ""))] = "web"
            for task in tasks:
                task["source_preference"] = source_by_task_id.get(
                    task["task_id"],
                    "auto",
                )
            stage.set_result(
                {
                    "自动/本地": sum(
                        task["source_preference"] == "auto" for task in tasks
                    ),
                    "联网": sum(
                        task["source_preference"] == "web" for task in tasks
                    ),
                }
            )
    except Exception as exc:
        log.warning(
            "[技术选型][source_planning] "
            "来源规划失败，回退到自动检索 | 异常类型={}",
            type(exc).__name__,
        )
        return tasks
    return tasks


def plan_evidence_tasks(state):
    """Create two auditable, cross-candidate evidence packs."""
    requirements = state.get("requirements", {})
    candidates = state.get("feasible_candidates", [])
    candidate_ids = [candidate["candidate_id"] for candidate in candidates]
    names = "、".join(candidate.get("name", "") for candidate in candidates)

    def build_task(task_id, criteria, source_preference):
        criterion_labels = "；".join(
            _criterion_search_label(criterion, requirements)
            for criterion in criteria
        )
        return EvidenceTask(
            task_id=task_id,
            candidate_ids=candidate_ids,
            query=(
                f"针对{requirements.get('application_scenario', '')}场景和"
                f"{requirements.get('business_goal', '')}目标，评估{names}在"
                f"{criterion_labels}方面的可比证据、适用前提和局限。"
            ),
            criteria=list(criteria),
            source_preference=source_preference,
            query_entities=[
                candidate.get("name", "")
                for candidate in candidates
                if candidate.get("name")
            ],
            relation_intent=f"比较候选技术在 {criterion_labels} 方面的影响关系",
            relation_types=["AFFECTS", "IMPROVES", "REDUCES", "REQUIRES"],
        ).model_dump()

    scenario_criteria = [
        item
        for item in requirements.get("scenario_criteria", [])
        if str(item.get("criterion_id", "")).strip()
    ][:3]
    fixed_active = requirements.get("active_criteria", []) or list(CRITERIA)
    technical = [
        criterion
        for criterion in TECHNICAL_CRITERIA
        if criterion in fixed_active and criterion != "risk"
    ]
    manufacturing = [
        criterion
        for criterion in MANUFACTURING_CRITERIA
        if criterion in fixed_active
    ]
    routing_context = " ".join(
        [
            str(state.get("user_request", "")),
            str(requirements.get("business_goal", "")),
            " ".join(
                str(value)
                for value in requirements.get("hard_constraints", [])
            ),
        ]
    )
    if "risk" in fixed_active:
        target = (
            manufacturing
            if _EXTERNAL_RISK_HINT_PATTERN.search(routing_context)
            else technical
        )
        target.append("risk")
    for item in scenario_criteria:
        criterion_id = str(item.get("criterion_id", "")).strip()
        criterion_context = " ".join(
            str(item.get(key, ""))
            for key in ("criterion_id", "name", "description")
        )
        target = (
            manufacturing
            if _WEB_EVIDENCE_HINT_PATTERN.search(criterion_context)
            else technical
        )
        target.append(criterion_id)
    tasks = []
    if technical:
        tasks.append(build_task("initial-technical", technical, "local"))
    if manufacturing:
        tasks.append(build_task("initial-manufacturing", manufacturing, "web"))
    with progress_stage(
        "技术选型",
        "source_planning",
        "确定证据包来源",
        details={"任务": len(tasks)},
    ) as stage:
        tasks = _assign_source_preferences(tasks)
        stage.set_result(
            {
                "自动/本地": sum(
                    task["source_preference"] in {"auto", "local"}
                    for task in tasks
                ),
                "联网": sum(task["source_preference"] == "web" for task in tasks),
            }
        )
    return {"evidence_tasks": tasks}

    # Kept below temporarily for source-compatible patch locality.
    """Create one cross-candidate evidence task for each criterion."""
    requirements = state.get("requirements", {})
    candidates = state.get("feasible_candidates", [])
    candidate_ids = [candidate["candidate_id"] for candidate in candidates]
    names = "、".join(candidate.get("name", "") for candidate in candidates)
    tasks = []
    for criterion in CRITERIA:
        query = (
            f"针对{requirements.get('application_scenario', '')}场景和"
            f"{requirements.get('business_goal', '')}目标，评估"
            f"{names}在{criterion}方面的可比证据、"
            "适用前提和局限。"
        )
        tasks.append(
            EvidenceTask(
                task_id=f"criterion-{criterion}",
                candidate_ids=candidate_ids,
                query=query,
                criteria=[criterion],
            ).model_dump()
        )
    return {"evidence_tasks": _assign_source_preferences(tasks)}


def _retrieve_task(task):
    try:
        with progress_stage(
            "证据任务",
            task.task_id,
            "检索证据",
            details={"来源": task.source_preference},
        ) as stage:
            retrieval_inputs = {
                "question": task.query,
                "source_preference": task.source_preference,
                "transform_count": 0,
                "max_transform_count": 1,
                "compact_relevance": True,
                "allow_web_fallback": task.source_preference == "web",
                "generation_attempt_count": 0,
            }
            if task.query_entities and task.relation_intent:
                retrieval_inputs.update(
                    {
                        "analysis_ready": True,
                        "route": "local_hybrid",
                        "query_entities": list(task.query_entities),
                        "relation_intent": task.relation_intent,
                        "relation_types": list(task.relation_types),
                    }
                )
            if task.source_preference == "web":
                retrieval_inputs.update(
                    {
                        "web_search_max_results": SELECTION_WEB_SEARCH_MAX_RESULTS,
                        "web_search_engine": SELECTION_WEB_SEARCH_ENGINE,
                        "require_source_url": False,
                    }
                )
            result = get_selection_cache(
                "retrieve_evidence",
                retrieval_graph,
                "technology_selection_retrieval_v1",
                retrieval_inputs,
            )
            cache_hit = result is not None
            if not cache_hit:
                result = retrieval_graph.invoke(retrieval_inputs)
                documents = result.get("documents", [])
                if documents and not any(
                    document.metadata.get("is_error")
                    for document in documents
                ):
                    set_selection_cache(
                        "retrieve_evidence",
                        retrieval_graph,
                        "technology_selection_retrieval_v1",
                        retrieval_inputs,
                        result,
                        ttl_seconds=SELECTION_RETRIEVAL_CACHE_TTL_SECONDS,
                    )
            stage.set_result(
                {
                    "结果": len(result.get("documents", [])),
                    "缓存命中": cache_hit,
                }
            )
    except Exception as exc:
        warning = (
            f"证据任务 {task.task_id} 检索失败"
            f"（来源 {task.source_preference}）：{type(exc).__name__}"
        )
        return task, {}, warning
    retrieval_error = str(result.get("retrieval_error", "") or "").strip()
    usable_documents = [
        document
        for document in result.get("documents", [])
        if not document.metadata.get("is_error")
        and document.page_content.strip()
    ]
    if task.source_preference == "web" and retrieval_error and not usable_documents:
        return (
            task,
            result,
            f"Web补检任务 {task.task_id} 未取得可用结果（{retrieval_error}）。",
        )
    return task, result, None


def retrieve_evidence(state):
    """Invoke the retrieval subgraph for each task and merge source records."""
    feasible_candidates = state.get("feasible_candidates", [])
    candidate_names = {
        candidate["candidate_id"]: candidate.get("name", candidate["candidate_id"])
        for candidate in feasible_candidates
    }
    known_candidate_ids = {
        candidate["candidate_id"]
        for candidate in feasible_candidates
    }
    scenario_criteria = state.get("requirements", {}).get(
        "scenario_criteria",
        [],
    )
    criteria_catalog = [
        {"criterion_id": criterion, "name": criterion}
        for criterion in CRITERIA
    ] + [
        {
            "criterion_id": str(item.get("criterion_id", "")).strip(),
            "name": str(item.get("name", "")).strip(),
            "description": str(item.get("description", "")).strip(),
        }
        for item in scenario_criteria
        if str(item.get("criterion_id", "")).strip()
    ]
    criterion_id_list = list(
        dict.fromkeys(item["criterion_id"] for item in criteria_catalog)
    )
    criterion_ids = set(criterion_id_list)
    required_criterion_id_list = _required_criterion_ids(
        state.get("requirements", {}),
        criterion_id_list,
    )
    records_by_id = {}
    for raw_record in state.get("evidence_records", []):
        record_data = EvidenceRecord.model_validate(raw_record).model_dump()
        record_data["supports"] = _validated_supports(
            record_data,
            raw_record,
            known_candidate_ids,
            criterion_ids,
            criteria_catalog,
        )
        relevant_supports = [
            support
            for support in record_data["supports"]
            if support["level"] in {"direct", "indirect"}
        ]
        record_data["candidate_ids"] = list(
            dict.fromkeys(support["candidate_id"] for support in relevant_supports)
        )
        record_data["criteria"] = list(
            dict.fromkeys(support["criterion"] for support in relevant_supports)
        )
        record = EvidenceRecord.model_validate(record_data)
        records_by_id[record.evidence_id] = record
    new_evidence_ids = set()
    retrieval_hints_by_id = {}
    tasks = [
        EvidenceTask.model_validate(raw_task)
        for raw_task in state.get("evidence_tasks", [])
    ]
    if not tasks:
        task_results = []
    elif len(tasks) == 1:
        task_results = [_retrieve_task(tasks[0])]
    else:
        worker_count = min(MAX_EVIDENCE_TASK_CONCURRENCY, len(tasks))
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            task_results = list(executor.map(_retrieve_task, tasks))

    warnings = list(state.get("warnings", []))
    for task, result, warning in task_results:
        if warning:
            warnings.append(warning)
            continue
        for document in result.get("documents", []):
            if document.metadata.get("is_error") or not document.page_content.strip():
                continue
            evidence_id = _evidence_id(document)
            retrieval_hints = retrieval_hints_by_id.setdefault(
                evidence_id,
                {"candidate_ids": set(), "criteria": set()},
            )
            retrieval_hints["candidate_ids"].update(task.candidate_ids)
            retrieval_hints["criteria"].update(task.criteria)
            existing = records_by_id.get(evidence_id)
            metadata = document.metadata
            retrieval_sources = _retrieval_sources(metadata, result, task)
            if existing is not None:
                existing.retrieval_sources = list(
                    dict.fromkeys(existing.retrieval_sources + retrieval_sources)
                )
                existing.retrieval_source = "+".join(existing.retrieval_sources)
                existing.source_reliability = _source_reliability(
                    existing.retrieval_sources,
                    existing.url,
                )
                continue
            source = str(
                metadata.get("source", metadata.get("url", ""))
            ).strip()
            url = str(metadata.get("url", "")).strip()
            if not url and re.match(r"^https?://", source, re.IGNORECASE):
                url = source
            records_by_id[evidence_id] = EvidenceRecord(
                evidence_id=evidence_id,
                content=_unwrap_graph_content(document.page_content),
                source=source,
                url=url,
                title=str(metadata.get("title", "")).strip(),
                retrieval_source="+".join(retrieval_sources),
                retrieval_sources=retrieval_sources,
                source_reliability=_source_reliability(retrieval_sources, url),
            )
            new_evidence_ids.add(evidence_id)

    records = [record.model_dump() for record in records_by_id.values()]
    new_records = [
        record for record in records if record["evidence_id"] in new_evidence_ids
    ]
    candidate_ids = known_candidate_ids
    assessment_missing_queries = []
    try:
        with progress_stage(
            "技术选型",
            "evidence_assessment",
            "评估证据归属",
            details={
                "证据": len(records),
                "候选": len(feasible_candidates),
            },
        ) as stage:
            batches = _assessment_batches(new_records)
            if batches:
                worker_count = min(
                    MAX_EVIDENCE_ASSESSMENT_CONCURRENCY, len(batches)
                )
                if worker_count == 1:
                    batch_results = [
                        _assess_evidence_batch_safely(
                            feasible_candidates,
                            batches[0],
                            criteria_catalog,
                        )
                    ]
                else:
                    with ThreadPoolExecutor(max_workers=worker_count) as executor:
                        batch_results = list(
                            executor.map(
                                lambda batch: _assess_evidence_batch_safely(
                                    feasible_candidates,
                                    batch,
                                    criteria_catalog,
                                ),
                                batches,
                            )
                        )
                failed_batch_results = [
                    (batch, error)
                    for batch, _, error in batch_results
                    if error is not None
                ]
                evidence_map = {
                    evidence_id: mapped
                    for _, batch_map, error in batch_results
                    if error is None
                    for evidence_id, mapped in batch_map.items()
                }
                for record in records:
                    if record["evidence_id"] not in new_evidence_ids:
                        continue
                    mapped = dict(evidence_map.get(record["evidence_id"], {}))
                    mapped_supports = list(mapped.get("supports", []))
                    retrieval_hints = retrieval_hints_by_id.get(
                        record["evidence_id"],
                        {"candidate_ids": set(), "criteria": set()},
                    )
                    mapped_supports.extend(
                        _deterministic_web_supports(
                            record,
                            retrieval_hints["candidate_ids"],
                            retrieval_hints["criteria"],
                            candidate_names,
                            criteria_catalog,
                        )
                    )
                    mapped["supports"] = mapped_supports
                    record["supports"] = _validated_supports(
                        record,
                        mapped,
                        candidate_ids,
                        criterion_ids,
                        criteria_catalog,
                    )
                    relevant_supports = [
                        support
                        for support in record["supports"]
                        if support["level"] in {"direct", "indirect"}
                    ]
                    record["candidate_ids"] = list(
                        dict.fromkeys(
                            support["candidate_id"]
                            for support in relevant_supports
                        )
                    )
                    record["criteria"] = list(
                        dict.fromkeys(
                            support["criterion"]
                            for support in relevant_supports
                        )
                    )
                if failed_batch_results:
                    error_types = ",".join(
                        dict.fromkeys(
                            type(error).__name__
                            for _, error in failed_batch_results
                        )
                    )
                    warnings.append(
                        "证据归属评估失败"
                        f"（{error_types}，{len(failed_batch_results)}/{len(batches)}批），"
                        "失败批次仅保留可确定性识别的网页相关关系。"
                    )
            # Coverage is derived from verified candidate/criterion maps,
            # not from free-form model suggestions.
            assessment_missing_queries = []

            derived_missing_queries = []
            for candidate in feasible_candidates:
                candidate_id = candidate["candidate_id"]
                covered_criteria = {
                    criterion
                    for record in records
                    if candidate_id in record["candidate_ids"]
                    for criterion in record["criteria"]
                }
                for criterion in required_criterion_id_list:
                    if criterion in covered_criteria:
                        continue
                    derived_missing_queries.append(
                        f"补充{candidate.get('name', candidate_id)}在"
                        f"{criterion}方面的可引用证据。"
                    )
            missing_queries = list(
                dict.fromkeys(
                    assessment_missing_queries + derived_missing_queries
                )
            )
            web_records = [
                record
                for record in new_records
                if any(
                    "web" in str(source).casefold()
                    for source in record.get("retrieval_sources", [])
                )
            ]
            web_valid_url_count = sum(
                bool(re.match(r"^https?://", str(record.get("url", "")), re.IGNORECASE))
                for record in web_records
            )
            web_relevant_support_count = len({
                (support.get("candidate_id"), support.get("criterion"))
                for record in web_records
                for support in record.get("supports", [])
                if support.get("level") in {"direct", "indirect"}
            })
            stage.set_result(
                {
                    "有效证据": len(records),
                    "新增证据": len(new_records),
                    "Web证据": len(web_records),
                    "Web有效URL": web_valid_url_count,
                    "相关支持": web_relevant_support_count,
                    "评估批次": len(_assessment_batches(new_records)),
                    "失败批次": len(failed_batch_results) if batches else 0,
                    "缺口": len(missing_queries),
                }
            )
    except Exception as error:
        for record in records:
            if record["evidence_id"] in new_evidence_ids:
                record["candidate_ids"] = []
                record["criteria"] = []
                record["supports"] = []
        missing_queries = [
            f"补充{candidate.get('name', candidate['candidate_id'])}在{criterion}方面的可引用证据。"
            for candidate in feasible_candidates
            for criterion in required_criterion_id_list
        ]
        warnings.append(
            f"证据归属评估失败（{type(error).__name__}），"
            "已保留原始证据但不将其用于评分。"
        )
    sufficient = bool(feasible_candidates) and not missing_queries
    return {
        "evidence_records": records,
        "evidence_sufficient": sufficient,
        "evidence_gaps": missing_queries,
        "missing_evidence_queries": missing_queries,
        "evidence_round": state.get("evidence_round", 0) + 1,
        "warnings": warnings,
    }


def optimize_evidence_queries(state):
    """Replace tasks with source-grouped retries for verified gaps only."""
    candidates = state.get("feasible_candidates", [])
    requirements = state.get("requirements", {})
    evidence_round = state.get("evidence_round", 0)
    grouped = {}
    for missing_query in state.get("missing_evidence_queries", []):
        criteria = _criteria_from_query(
            missing_query,
            requirements,
        )
        candidate_ids = _candidate_ids_from_query(missing_query, candidates)
        if not criteria or not candidate_ids:
            continue
        for criterion in criteria:
            source = _supplement_source(
                criterion,
                state,
                requirements,
                evidence_round,
            )
            if source == "web":
                for candidate_id in candidate_ids:
                    focus = (
                        criterion
                        if evidence_round >= 2
                        else "all"
                    )
                    group_key = (source, candidate_id, focus)
                    group = grouped.setdefault(
                        group_key,
                        {"candidate_ids": [candidate_id], "criteria": set()},
                    )
                    group["criteria"].add(criterion)
            else:
                group_key = (source, "all", "all")
                group = grouped.setdefault(
                    group_key,
                    {"candidate_ids": [], "criteria": set()},
                )
                group["candidate_ids"] = list(
                    dict.fromkeys(group["candidate_ids"] + candidate_ids)
                )
                group["criteria"].add(criterion)

    tasks = []
    candidate_names = {
        candidate["candidate_id"]: candidate.get("name", candidate["candidate_id"])
        for candidate in candidates
    }
    for index, ((source, _, focus), group) in enumerate(grouped.items(), start=1):
        candidate_ids = group["candidate_ids"]
        criteria = sorted(group["criteria"])
        names = " ".join(
            _candidate_search_label(
                candidate_names.get(candidate_id, candidate_id)
            )
            for candidate_id in candidate_ids
        )
        criterion_labels = "；".join(
            _criterion_search_label(criterion, requirements)
            for criterion in criteria
        )
        scenario = str(requirements.get("application_scenario", "")).strip()
        source_terms = (
            "official datasheet technical paper manufacturer report"
            if evidence_round >= 2
            else "datasheet paper manufacturer technical report"
        )
        task = EvidenceTask(
            task_id=f"retry-{evidence_round}-{index}",
            candidate_ids=list(candidate_ids),
            query=(
                f"{names} semiconductor technology {scenario} "
                f"{criterion_labels} {source_terms}"
            ),
            criteria=criteria,
            source_preference=source,
        ).model_dump()
        tasks.append(task)
    return {"evidence_tasks": tasks}

    # Kept below temporarily for source-compatible patch locality.
    """Replace the task list with targeted queries for unresolved evidence."""
    candidates = state.get("feasible_candidates", [])
    candidate_ids = [candidate["candidate_id"] for candidate in candidates]
    tasks = []
    for index, missing_query in enumerate(
        state.get("missing_evidence_queries", []),
        start=1,
    ):
        matched_criteria = [
            criterion
            for criterion in CRITERIA
            if criterion in missing_query
        ]
        query = (
            f"原始选型需求：{state.get('user_request', '')}。"
            f"{missing_query}"
        )
        tasks.append(
            EvidenceTask(
                task_id=f"retry-{state.get('evidence_round', 0)}-{index}",
                candidate_ids=candidate_ids,
                query=query,
                criteria=matched_criteria or list(CRITERIA),
            ).model_dump()
        )
    return {"evidence_tasks": _assign_source_preferences(tasks)}

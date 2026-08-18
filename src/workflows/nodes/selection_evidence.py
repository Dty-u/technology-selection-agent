"""Evidence planning and retrieval nodes for technology selection."""

import json
import re
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from urllib.parse import urlsplit, urlunsplit

from langchain_core.prompts import ChatPromptTemplate

from src.config.settings import (
    EVIDENCE_ASSESSMENT_BATCH_SIZE,
    EVIDENCE_ASSESSMENT_SINGLE_BATCH_LIMIT,
    MAX_EVIDENCE_CARD_CHARACTERS,
    MAX_EVIDENCE_ASSESSMENT_CONCURRENCY,
    MAX_EVIDENCE_TASK_CONCURRENCY,
    SELECTION_RETRIEVAL_CACHE_TTL_SECONDS,
    SELECTION_WEB_SEARCH_ENGINE,
    SELECTION_WEB_SEARCH_MAX_RESULTS,
)
from src.infrastructure.llm.chat_model import selection_json_llm, selection_llm
from src.observability.logging import log
from src.observability.workflow_progress import progress_stage
from src.workflows.nodes.selection_requirements import parse_json_object
from src.workflows.retrieval_graph import retrieval_graph
from src.workflows.selection_cache import (
    get_selection_cache,
    set_selection_cache,
)
from src.workflows.technology_selection_state import (
    CRITERIA,
    EvidenceRecord,
    EvidenceSupport,
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


TECHNICAL_CRITERIA = (
    "performance",
    "power",
    "thermal",
    "scalability",
    "risk",
)
MANUFACTURING_CRITERIA = ("cost", "maturity")
_WEB_EVIDENCE_HINT_PATTERN = re.compile(
    r"供应链|供应商|市场|价格|成本|报价|量产|良率|产能|交付|"
    r"路线图|政策|地缘|出口管制|短缺|可用性|supply\s*chain|"
    r"supplier|market|price|cost|production|yield|capacity|delivery|"
    r"roadmap|policy|geopolit|export\s+control|shortage|availability",
    re.IGNORECASE,
)
_EXTERNAL_RISK_HINT_PATTERN = re.compile(
    r"供应链|供应商|市场供应|量产|良率|产能|交付|政策|地缘|"
    r"出口管制|短缺|可用性|supply\s*chain|supplier|market\s+supply|"
    r"production|yield|capacity|delivery|policy|geopolit|"
    r"export\s+control|shortage|availability",
    re.IGNORECASE,
)
CRITERION_ALIASES = {
    "performance": (
        "performance", "throughput", "bandwidth", "latency", "frequency",
        "deposition rate", "growth rate", "cycle time", "wph", "wafer",
        "on-resistance", "switching frequency", "breakdown field",
        "性能", "带宽", "吞吐", "频率", "延迟", "生产效率", "沉积速率",
        "生长速率", "产能", "节拍", "速度", "导通电阻", "开关频率",
        "击穿场强", "额定电压", "转换效率",
    ),
    "power": (
        "power", "energy", "watt", "power density", "power loss",
        "功耗", "能耗", "能效", "电力", "功率", "功率密度",
        "导通损耗", "开关损耗",
    ),
    "thermal": (
        "thermal", "temperature", "heat", "cooling", "thermal conductivity",
        "散热", "温度", "热管理", "冷却", "热导率", "结温", "耐高温",
    ),
    "cost": ("cost", "price", "expense", "成本", "价格", "费用", "投资"),
    "maturity": (
        "maturity", "mass production", "commercial", "widely used",
        "成熟度", "量产", "商业化", "产业化", "广泛应用", "车规",
        "装车", "批量应用",
    ),
    "risk": (
        "risk", "failure", "defect", "toxicity", "hazard", "limitation",
        "reliability", "风险", "失效", "缺陷", "毒性", "危害", "局限",
        "可靠性", "退化", "供应链",
    ),
    "scalability": (
        "scalability", "scale-up", "large area", "batch processing",
        "production capacity", "wafer size", "扩展性", "可扩展", "规模化",
        "大面积", "批量处理", "产能", "晶圆尺寸", "大批量",
    ),
}
CRITERION_SEARCH_LABELS = {
    "performance": (
        "性能 效率 开关频率 吞吐能力 performance efficiency "
        "switching frequency throughput"
    ),
    "power": (
        "功耗 能效 功率密度 开关损耗 power efficiency power density "
        "switching loss conduction loss"
    ),
    "thermal": (
        "热导率 结温 散热能力 thermal conductivity junction temperature "
        "heat dissipation"
    ),
    "cost": "器件 衬底 制造成本 device substrate manufacturing cost price",
    "maturity": (
        "量产成熟度 商业化 车规应用 automotive qualification "
        "mass production commercial adoption maturity"
    ),
    "risk": (
        "可靠性 供应链 技术风险 reliability failure supply chain risk"
    ),
    "scalability": (
        "产能 晶圆尺寸 规模化能力 production capacity wafer size "
        "manufacturing scale scalability"
    ),
}
DEPOSITION_CRITERION_SEARCH_LABELS = {
    "performance": (
        "薄膜均匀性 台阶覆盖率 保形性 沉积速率 film uniformity "
        "step coverage conformality deposition rate"
    ),
    "cost": (
        "总拥有成本 设备投资 前驱体消耗 气体利用率 沉积速率 晶圆吞吐量 "
        "维护成本 cost of ownership total cost of ownership equipment CAPEX "
        "precursor consumption gas utilization deposition rate wafer throughput "
        "maintenance cost"
    ),
    "maturity": (
        "量产应用 晶圆厂采用 装机基础 工艺认证 大批量制造 "
        "production-proven fab adoption installed base process qualification "
        "high volume manufacturing"
    ),
    "scalability": (
        "批量处理 晶圆吞吐量 沉积速率 产线扩展 batch processing wafer throughput "
        "deposition rate production scalability"
    ),
}
_DEPOSITION_CONTEXT_PATTERN = re.compile(
    r"沉积|薄膜|原子层|气相|\bALD\b|\bCVD\b|deposition|thin[ -]?film",
    re.IGNORECASE,
)
CANDIDATE_SEARCH_ALIASES = {
    "sic": ("silicon carbide", "SiC", "碳化硅"),
    "gan": ("gallium nitride", "GaN", "氮化镓"),
    "ald": ("atomic layer deposition", "ALD", "原子层沉积"),
    "cvd": ("chemical vapor deposition", "CVD", "化学气相沉积"),
    "2.5d": ("2.5D semiconductor packaging", "2.5D"),
    "3d ic": ("3D integrated circuit", "3D IC"),
    "chiplet": ("chiplet semiconductor packaging", "chiplet"),
}


def _candidate_aliases(candidate_name):
    name = str(candidate_name or "").strip()
    aliases = CANDIDATE_SEARCH_ALIASES.get(name.casefold(), (name,))
    return tuple(dict.fromkeys(alias for alias in aliases if str(alias).strip()))


def _candidate_search_label(candidate_name):
    return " ".join(_candidate_aliases(candidate_name))


def _contains_candidate_alias(value, aliases):
    normalized = _normalize_quote(value)
    for alias in aliases:
        normalized_alias = _normalize_quote(alias)
        if not normalized_alias:
            continue
        if re.fullmatch(r"[a-z0-9.+-]+", normalized_alias):
            if re.search(
                rf"(?<![a-z0-9]){re.escape(normalized_alias)}(?![a-z0-9])",
                normalized,
            ):
                return True
        elif normalized_alias in normalized:
            return True
    return False


def _compact_evidence_card(record):
    """Keep criterion-relevant excerpts without asserting evidence support."""
    content = str(record.get("content", "")).strip()
    if len(content) <= MAX_EVIDENCE_CARD_CHARACTERS:
        excerpt = content
    else:
        aliases = [
            alias.lower()
            for criterion in record.get("criteria", [])
            for alias in CRITERION_ALIASES.get(criterion, ())
        ]
        sentences = re.split(r"(?<=[。！？.!?])\s*", content)
        relevant = [
            sentence
            for sentence in sentences
            if any(alias in sentence.lower() for alias in aliases)
        ]
        excerpt = " ".join(relevant)[:MAX_EVIDENCE_CARD_CHARACTERS]
        if not excerpt:
            excerpt = content[:MAX_EVIDENCE_CARD_CHARACTERS]
    return {
        "evidence_id": record["evidence_id"],
        "title": record.get("title", ""),
        "source": record.get("source", ""),
        "url": record.get("url", ""),
        "content": excerpt,
    }


def _assessment_batches(records):
    if len(records) <= EVIDENCE_ASSESSMENT_SINGLE_BATCH_LIMIT:
        return [records] if records else []
    return [
        records[index : index + EVIDENCE_ASSESSMENT_BATCH_SIZE]
        for index in range(0, len(records), EVIDENCE_ASSESSMENT_BATCH_SIZE)
    ]


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


def _unwrap_graph_content(content):
    normalized = str(content or "").strip()
    for marker in ("来源内容：", "来源内容:"):
        if marker in normalized:
            normalized = normalized.split(marker, 1)[1].strip()
            break
    return normalized


def _canonical_source(metadata):
    source = str(metadata.get("url") or metadata.get("source") or "").strip()
    if re.match(r"^https?://", source, re.IGNORECASE):
        parsed = urlsplit(source)
        source = urlunsplit(
            (
                parsed.scheme.casefold(),
                parsed.netloc.casefold(),
                parsed.path.rstrip("/"),
                parsed.query,
                "",
            )
        )
    return source.replace("\\", "/").strip().casefold()


def _normalized_evidence_content(content):
    return " ".join(_unwrap_graph_content(content).split()).casefold()


def _evidence_id(document) -> str:
    metadata = document.metadata
    identity = (
        f"{_canonical_source(metadata)}:"
        f"{_normalized_evidence_content(document.page_content)}"
    )
    digest = sha256(str(identity).encode("utf-8")).hexdigest()[:12].upper()
    return f"E{digest}"


def _retrieval_sources(metadata, result, task):
    values = metadata.get("retrieval_sources", [])
    if isinstance(values, str):
        values = [values]
    values = list(values or [])
    if metadata.get("retrieval_source"):
        values.append(metadata["retrieval_source"])
    if result.get("retrieval_source"):
        values.append(result["retrieval_source"])
    return list(dict.fromkeys(str(value).strip() for value in values if str(value).strip()))


def _source_reliability(retrieval_sources, url):
    is_web = any("web" in source.casefold() for source in retrieval_sources)
    if not is_web:
        return "internal"
    return "web_identified" if url else "web_unattributed"


def _normalize_quote(value):
    return " ".join(str(value or "").split()).casefold()


def _criterion_aliases(criterion, criteria_catalog):
    if criterion in CRITERION_ALIASES:
        return CRITERION_ALIASES[criterion]
    name = next(
        (
            str(item.get("name", "")).strip()
            for item in criteria_catalog
            if item.get("criterion_id") == criterion
        ),
        "",
    )
    description = next(
        (
            str(item.get("description", "")).strip()
            for item in criteria_catalog
            if item.get("criterion_id") == criterion
        ),
        "",
    )
    aliases = [name]
    aliases.extend(
        part.strip()
        for part in re.split(r"与|和|及|/|、", f"{name}、{description}")
        if len(part.strip()) >= 2
    )
    if "均匀" in name or "共形" in name:
        aliases.extend(
            (
                "uniformity",
                "uniform",
                "conformality",
                "conformal",
                "step coverage",
                "均匀",
                "共形",
                "台阶覆盖",
            )
        )
    scenario_text = f"{name} {description}".casefold()
    if "耐压" in scenario_text or "击穿电压" in scenario_text:
        aliases.extend(
            (
                "耐压",
                "击穿电压",
                "工作电压",
                "blocking voltage",
                "breakdown voltage",
                "voltage rating",
            )
        )
    if "量产" in scenario_text or "成熟" in scenario_text:
        aliases.extend(
            (
                "量产",
                "商业化",
                "规模生产",
                "mass production",
                "commercial production",
            )
        )
    if "供应链" in scenario_text or "供应商" in scenario_text:
        aliases.extend(
            (
                "供应链",
                "供应商",
                "产能",
                "supply chain",
                "supplier",
                "capacity",
            )
        )
    return tuple(dict.fromkeys(alias for alias in aliases if alias))


def _validated_supports(
    record,
    mapped,
    candidate_ids,
    criterion_ids,
    criteria_catalog,
):
    supports = []
    raw_supports = mapped.get("supports", []) if isinstance(mapped, dict) else []
    if not raw_supports and isinstance(mapped, dict):
        raw_supports = [
            {
                "candidate_id": candidate_id,
                "criterion": criterion,
                "level": "indirect",
                "quote": "",
            }
            for candidate_id in mapped.get("candidate_ids", [])
            for criterion in mapped.get("criteria", [])
        ]
    normalized_content = _normalize_quote(record.get("content", ""))
    seen = set()
    for raw_support in raw_supports:
        if not isinstance(raw_support, dict):
            continue
        candidate_id = str(raw_support.get("candidate_id", "")).strip()
        criterion = str(raw_support.get("criterion", "")).strip()
        if candidate_id not in candidate_ids or criterion not in criterion_ids:
            continue
        level = str(raw_support.get("level", "none")).strip().casefold()
        if level not in {"direct", "indirect", "none"}:
            level = "none"
        quote = str(raw_support.get("quote", "")).strip()
        normalized_quote = _normalize_quote(quote)
        if level == "direct" and (
            not normalized_quote
            or normalized_quote not in normalized_content
        ):
            level = "none"
        elif level == "direct" and not any(
                _normalize_quote(alias) in normalized_quote
                for alias in _criterion_aliases(criterion, criteria_catalog)
            ):
            level = "indirect"
        key = (candidate_id, criterion, level, normalized_quote)
        if key in seen:
            continue
        seen.add(key)
        supports.append(
            EvidenceSupport(
                candidate_id=candidate_id,
                criterion=criterion,
                level=level,
                quote=quote,
            ).model_dump()
        )
    return supports


def _deterministic_web_supports(
    record,
    hinted_candidate_ids,
    hinted_criteria,
    candidate_names,
    criteria_catalog,
):
    """Extract exact Web sentences only when candidate and metric co-occur."""
    if record.get("source_reliability") != "web_identified":
        return []
    sentences = [
        sentence.strip()
        for sentence in re.split(
            r"(?<=[。！？.!?])\s*|\r?\n+",
            str(record.get("content", "")),
        )
        if sentence.strip()
    ]
    supports = []
    context = f"{record.get('title', '')} {record.get('content', '')}"
    candidate_aliases = {
        candidate_id: _candidate_aliases(candidate_name)
        for candidate_id, candidate_name in candidate_names.items()
    }
    mentioned_candidate_ids = {
        candidate_id
        for candidate_id, aliases in candidate_aliases.items()
        if _contains_candidate_alias(context, aliases)
    }
    for candidate_id in hinted_candidate_ids:
        aliases = candidate_aliases.get(candidate_id, ())
        if not aliases:
            continue
        for criterion in hinted_criteria:
            normalized_aliases = [
                _normalize_quote(alias)
                for alias in _criterion_aliases(criterion, criteria_catalog)
                if _normalize_quote(alias)
            ]
            quote = next(
                (
                    sentence
                    for sentence in sentences
                    if (
                        _contains_candidate_alias(sentence, aliases)
                        or mentioned_candidate_ids == {candidate_id}
                    )
                    and any(
                        alias in _normalize_quote(sentence)
                        for alias in normalized_aliases
                    )
                ),
                "",
            )
            if quote:
                supports.append(
                    {
                        "candidate_id": candidate_id,
                        "criterion": criterion,
                        "level": "direct",
                        "quote": quote,
                    }
                )
    return supports


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


def _required_criterion_ids(requirements, criterion_id_list):
    """Return criteria that can affect this request's decision."""
    requirements = requirements or {}
    active_criteria = requirements.get("active_criteria")
    if isinstance(active_criteria, list) and active_criteria:
        required = [
            criterion
            for criterion in active_criteria
            if criterion in CRITERIA
        ]
    else:
        priorities = requirements.get("priorities")
        if isinstance(priorities, dict) and priorities:
            required = []
            for criterion in CRITERIA:
                try:
                    weight = float(priorities.get(criterion, 0))
                except (TypeError, ValueError):
                    weight = 0
                if weight > 0:
                    required.append(criterion)
        else:
            required = list(CRITERIA)

    for constraint in requirements.get("hard_constraints", []):
        required.extend(_criteria_from_query(constraint, requirements))
    required.extend(
        str(item.get("criterion_id", "")).strip()
        for item in requirements.get("scenario_criteria", [])
        if str(item.get("criterion_id", "")).strip()
    )
    valid_ids = set(criterion_id_list)
    return [
        criterion
        for criterion in dict.fromkeys(required)
        if criterion in valid_ids
    ]


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


def _criteria_from_query(query, requirements=None):
    normalized = str(query).lower()
    matched = [
        criterion
        for criterion, aliases in CRITERION_ALIASES.items()
        if any(alias.lower() in normalized for alias in aliases)
    ]
    for item in (requirements or {}).get("scenario_criteria", []):
        criterion_id = str(item.get("criterion_id", "")).strip()
        name = str(item.get("name", "")).strip().lower()
        if criterion_id and (
            criterion_id.lower() in normalized
            or (name and name in normalized)
        ):
            matched.append(criterion_id)
    return list(dict.fromkeys(matched))


def _candidate_ids_from_query(query, candidates):
    normalized = str(query).lower()
    return [
        candidate["candidate_id"]
        for candidate in candidates
        if candidate["candidate_id"].lower() in normalized
        or candidate.get("name", "").lower() in normalized
    ]


def _criterion_search_label(criterion, requirements):
    context = " ".join(
        str((requirements or {}).get(key, ""))
        for key in ("business_goal", "application_scenario")
    )
    if (
        criterion in DEPOSITION_CRITERION_SEARCH_LABELS
        and _DEPOSITION_CONTEXT_PATTERN.search(context)
    ):
        return DEPOSITION_CRITERION_SEARCH_LABELS[criterion]
    if criterion in CRITERION_SEARCH_LABELS:
        return CRITERION_SEARCH_LABELS[criterion]
    for item in (requirements or {}).get("scenario_criteria", []):
        if str(item.get("criterion_id", "")).strip() != criterion:
            continue
        name = str(item.get("name", "")).strip()
        description = str(item.get("description", "")).strip()
        return "：".join(part for part in (name, description) if part)
    return criterion


def _supplement_source(criterion, state, requirements, evidence_round):
    """Keep time-sensitive gaps on Web while retrying stable facts locally."""
    if evidence_round >= 2 or criterion in MANUFACTURING_CRITERIA:
        return "web"
    if criterion == "risk":
        risk_context = " ".join(
            [
                str(state.get("user_request", "")),
                str((requirements or {}).get("business_goal", "")),
                " ".join(
                    str(value)
                    for value in (requirements or {}).get("hard_constraints", [])
                ),
            ]
        )
        if _EXTERNAL_RISK_HINT_PATTERN.search(risk_context):
            return "web"
    for item in (requirements or {}).get("scenario_criteria", []):
        if str(item.get("criterion_id", "")).strip() != criterion:
            continue
        criterion_context = " ".join(
            str(item.get(key, ""))
            for key in ("criterion_id", "name", "description")
        )
        if _WEB_EVIDENCE_HINT_PATTERN.search(criterion_context):
            return "web"
    return "local"


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

"""Deterministic evidence rules: criterion and candidate aliases, evidence
normalization and identity, support validation, and query helpers.

These helpers never call a model or retrieval backend.
"""

import re
from hashlib import sha256
from urllib.parse import urlsplit, urlunsplit

from src.config.settings import (
    EVIDENCE_ASSESSMENT_BATCH_SIZE,
    EVIDENCE_ASSESSMENT_SINGLE_BATCH_LIMIT,
    MAX_EVIDENCE_CARD_CHARACTERS,
)
from src.workflows.technology_selection_state import CRITERIA, EvidenceSupport


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

"""Deterministic routing rules: phrase, keyword, and regex checks.

These helpers never call a model; classify_query uses them to answer
obvious cases before (or instead of) the LLM classifier.
"""

import re


_DIRECT_RESPONSE_PHRASES = {
    "identity": {
        "你是谁",
        "你是什么",
        "你叫什么",
        "你叫什么名字",
        "请介绍一下你自己",
        "介绍一下你自己",
        "whoareyou",
    },
    "greeting": {
        "你好",
        "您好",
        "嗨",
        "哈喽",
        "hello",
        "hi",
        "谢谢",
        "谢谢你",
        "感谢",
        "再见",
    },
    "acknowledgement": {
        "好",
        "好的",
        "好吧",
        "好滴",
        "嗯",
        "嗯嗯",
        "哦",
        "噢",
        "行",
        "ok",
        "okay",
        "明白",
        "明白了",
        "知道了",
        "了解",
        "了解了",
        "收到",
        "懂了",
        "原来如此",
    },
    "capabilities": {
        "你能做什么",
        "你可以做什么",
        "你会做什么",
        "怎么使用",
        "如何使用",
        "帮助",
        "你支持哪些功能",
        "有哪些功能",
    },
}

_DIRECT_RESPONSE_REASONS = {
    "identity": "身份询问",
    "greeting": "问候或礼貌表达",
    "acknowledgement": "简短确认或语气回应",
    "capabilities": "能力或使用帮助",
    "out_of_scope": "明显超出半导体服务范围",
}

_OBVIOUS_OUT_OF_SCOPE_PATTERN = re.compile(
    r"天气|气温|星座|菜谱|食谱|写(?:一首|首)?.{0,12}诗|写(?:一篇|篇)?.{0,12}小说|"
    r"歌词|旅游攻略|poem|weather|recipe|horoscope",
    re.IGNORECASE,
)

_STRONG_FRESH_INFORMATION_PATTERN = re.compile(
    r"最新|近期|最近|今日|截至|今年|latest|recent|today|as\s+of",
    re.IGNORECASE,
)

_WEAK_FRESH_INFORMATION_PATTERN = re.compile(
    r"目前|当前|现状|current|now",
    re.IGNORECASE,
)

_DYNAMIC_INFORMATION_PATTERN = re.compile(
    r"价格|成本|报价|量产|产业化|良率|出货|交付|产能|市场份额|"
    r"供应商|路线图|供应链|政策|营收|投资|price|cost|quote|"
    r"production|yield|shipment|delivery|capacity|market\s+share|"
    r"supplier|roadmap|supply\s+chain|policy|revenue|investment",
    re.IGNORECASE,
)

_HISTORICAL_INFORMATION_PATTERN = re.compile(
    r"历史|发展历程|演变|回顾|history|historical|timeline",
    re.IGNORECASE,
)

_SEMICONDUCTOR_SELECTION_TERMS = (
    "半导体",
    "芯片",
    "晶圆",
    "晶体管",
    "集成电路",
    "soc",
    "fab",
    "wafer",
    "制程",
    "finfet",
    "gaa",
    "chiplet",
    "2.5d",
    "3d ic",
    "euv",
    "duv",
    "sic",
    "gan",
)

_SEMICONDUCTOR_CONTEXT_TERMS = _SEMICONDUCTOR_SELECTION_TERMS + (
    "光刻",
    "刻蚀",
    "沉积",
    "封装",
    "掩模",
    "eda",
)

_SEMICONDUCTOR_KNOWLEDGE_TERMS = _SEMICONDUCTOR_SELECTION_TERMS + (
    "mosfet",
    "cmos",
    "igbt",
    "dram",
    "sram",
    "nand",
    "hbm",
    "asic",
    "fpga",
    "光刻机",
    "光刻胶",
    "刻蚀机",
    "二极管",
    "存储器",
    "掩模版",
    "掩膜版",
    "eda",
)

_KNOWLEDGE_INTENT_PATTERN = re.compile(
    r"是什么|是啥|啥是|什么意思|定义|概念|原理|工作机制|"
    r"如何工作|怎么工作|有什么用|做什么用|有何用途|用来干什么|用途|"
    r"区别|差异|优缺点|为什么|如何制造|怎么制造|有哪些|"
    r"包括哪些|解释|介绍",
    re.IGNORECASE,
)

_META_CAPABILITY_PATTERN = re.compile(
    r"(?:你|您|助手|系统).{0,8}(?:能|会|可以|支持|擅长).{0,8}"
    r"(?:回答|解答|处理|提供|帮|做)|"
    r"(?:你|您|助手|系统).{0,8}(?:能力|功能|服务范围)",
    re.IGNORECASE,
)

_KNOWLEDGE_REPORT_PATTERN = re.compile(
    r"报告|综述|调研|白皮书|report|review|survey",
    re.IGNORECASE,
)

_FOLLOW_UP_REFERENCE_PREFIXES = (
    "它",
    "他",
    "她",
    "这个",
    "这项",
    "这类",
    "这种",
    "那个",
    "该",
    "上述",
    "前述",
    "其",
)

_FOLLOW_UP_LEAD_IN_PREFIXES = (
    "那么",
    "那",
    "所以",
    "另外",
    "还有",
    "然后",
    "请问",
)

_DIRECT_RESPONSE_TEMPLATES = {
    "identity": (
        "我是一个面向半导体知识问答和技术选型的 AI 助手。"
        "我可以回答半导体材料、器件、制造和封装问题，"
        "也可以比较候选技术并生成有证据支持的选型报告。"
    ),
    "greeting": (
        "你好，我可以帮助你查询半导体知识，"
        "或者完成技术比较、风险分析和选型报告。"
    ),
    "capabilities": (
        "我主要支持半导体知识问答，FinFET、GAA、SiC、GaN "
        "等技术比较，性能、功耗、成本、成熟度、风险和扩展性评估，"
        "以及技术选型报告生成。"
    ),
    "acknowledgement": (
        "好的。还想继续了解的话，可以接着问我半导体知识、工艺细节或技术选型问题。"
    ),
    "out_of_scope": (
        "这个我不太擅长，我主要专注于半导体领域。"
        "你可以问我器件原理、制造工艺，或者让我帮你比较技术方案。"
    ),
}


def _normalize_direct_request(query):
    normalized = "".join(str(query).strip().lower().split())
    return normalized.strip("，,。.!！?？~～")


def _fast_direct_response_kind(query):
    normalized = _normalize_direct_request(query)
    for kind, phrases in _DIRECT_RESPONSE_PHRASES.items():
        if normalized in phrases:
            return kind
    if (
        _OBVIOUS_OUT_OF_SCOPE_PATTERN.search(normalized)
        and not any(term in normalized for term in _SEMICONDUCTOR_CONTEXT_TERMS)
    ):
        return "out_of_scope"
    return None


def _is_explicit_technology_selection(query):
    normalized = " ".join(str(query).strip().lower().split())
    has_domain_term = any(
        term in normalized for term in _SEMICONDUCTOR_SELECTION_TERMS
    )
    has_explicit_decision_intent = bool(
        re.search(
            r"选型|技术路线|方案选择|推荐|采用|如何选择|怎么选|"
            r"选哪个|哪种更适合|哪个更适合|哪个更好",
            normalized,
        )
    )
    has_comparison_intent = bool(
        re.search(r"比较|对比|\bvs\.?\b|\bversus\b", normalized)
    )
    has_decision_context = bool(
        re.search(
            r"面向|用于|应用场景|业务目标|优先|约束|预算|成本上限|"
            r"功耗上限|导入|落地",
            normalized,
        )
    )
    return has_domain_term and (
        has_explicit_decision_intent
        or (has_comparison_intent and has_decision_context)
    )


def _is_explicit_semiconductor_knowledge_query(query):
    """Identify clear domain questions that cannot be an entry-only reply."""
    normalized = " ".join(str(query or "").strip().lower().split())
    if not normalized:
        return False
    if _OBVIOUS_OUT_OF_SCOPE_PATTERN.search(normalized):
        return False
    if _META_CAPABILITY_PATTERN.search(normalized):
        return False
    has_domain_term = any(
        term in normalized for term in _SEMICONDUCTOR_KNOWLEDGE_TERMS
    )
    return has_domain_term and bool(_KNOWLEDGE_INTENT_PATTERN.search(normalized))


def _requires_web_source(query):
    """Force Web only for clearly current facts, not historical wording."""
    normalized = str(query).strip()
    if _HISTORICAL_INFORMATION_PATTERN.search(normalized):
        return False
    if _STRONG_FRESH_INFORMATION_PATTERN.search(normalized):
        return True
    return bool(
        _WEAK_FRESH_INFORMATION_PATTERN.search(normalized)
        and _DYNAMIC_INFORMATION_PATTERN.search(normalized)
    )


def _fallback_semiconductor_rag_classification(query, report_format):
    """Return a conservative RAG route when the classifier omits JSON."""
    normalized = str(query).strip().lower()
    if not any(term in normalized for term in _SEMICONDUCTOR_SELECTION_TERMS):
        return None
    return {
        "query_type": "rag_qa",
        "source_preference": "web" if _requires_web_source(query) else "auto",
        "response_type": "answer",
        "report_format": report_format,
        "routing_reason": "分类模型未返回有效 JSON，按半导体知识问答降级处理",
        "status": "running",
        "warnings": ["分类模型未返回有效 JSON，已按半导体知识问答处理。"],
    }


def _is_semiconductor_pronoun_follow_up(query, short_term_context):
    """Recognize a same-session follow-up whose referent is semiconductor."""
    normalized_query = str(query or "").strip().lower()
    normalized_context = str(short_term_context or "").strip().lower()
    for lead_in in _FOLLOW_UP_LEAD_IN_PREFIXES:
        if normalized_query.startswith(lead_in):
            normalized_query = normalized_query[len(lead_in) :].lstrip(" ，,")
            break
    return bool(
        normalized_context
        and normalized_query.startswith(_FOLLOW_UP_REFERENCE_PREFIXES)
        and any(term in normalized_context for term in _SEMICONDUCTOR_CONTEXT_TERMS)
    )


def _short_term_follow_up_rag_classification(
    query,
    report_format,
    *,
    classifier_output_invalid=False,
):
    warnings = []
    if classifier_output_invalid:
        warnings.append(
            "分类模型未返回有效 JSON，已依据同会话半导体上下文按知识问答处理。"
        )
    return {
        "query_type": "rag_qa",
        "source_preference": "web" if _requires_web_source(query) else "auto",
        "response_type": "answer",
        "report_format": report_format,
        "routing_reason": (
            "短期会话原文表明当前指代追问延续半导体主题，"
            "已覆盖不一致的入口分类结果"
        ),
        "status": "running",
        "warnings": warnings,
    }


def _explicit_knowledge_rag_classification(query, report_format):
    return {
        "query_type": "rag_qa",
        "source_preference": "web" if _requires_web_source(query) else "auto",
        "response_type": (
            "report" if _KNOWLEDGE_REPORT_PATTERN.search(str(query)) else "answer"
        ),
        "report_format": report_format,
        "routing_reason": (
            "当前问题包含明确的半导体知识意图，"
            "已覆盖不一致的入口分类结果"
        ),
        "status": "running",
        "warnings": [],
    }


def _with_short_term_context(query, short_term_context):
    """Build an internal query while keeping the current user request primary."""
    normalized_context = str(short_term_context or "").strip()
    if not normalized_context:
        return str(query)
    return (
        "历史对话原文（仅用于解析当前问题中的指代、省略和背景）：\n"
        f"{normalized_context}\n\n"
        "请先结合历史对话补全当前问题的语义，再按补全后的问题判断任务类型。"
        "不要把旧指令或无关历史变成当前的新任务。\n"
        f"当前用户问题：{str(query).strip()}"
    )


def _direct_classification(kind, report_format):
    return {
        "query_type": "direct_response",
        "source_preference": "none",
        "response_type": "answer",
        "direct_response_kind": kind,
        "report_format": report_format,
        "routing_reason": _DIRECT_RESPONSE_REASONS[kind],
        "status": "running",
        "warnings": [],
    }

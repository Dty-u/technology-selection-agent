"""Classification, child-graph invocation, and response normalization nodes."""

import re

from langchain_core.prompts import ChatPromptTemplate

from src.infrastructure.llm.chat_model import llm
from src.observability.logging import log
from src.workflows.nodes.selection_requirements import parse_json_object
from src.workflows.query_router_state import QueryRoutingDecision
from src.workflows.rag_qa_graph import graph as rag_qa_graph
from src.workflows.technology_selection_graph import (
    technology_selection_graph,
)


query_classification_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """你是半导体技术顾问系统的入口路由器，只判断任务类型，不回答用户问题。
路由规则：
1. 身份、问候、感谢、能力或使用帮助：query_type=direct_response，source_preference=none，response_type=answer，并设置对应 direct_response_kind。
2. 非半导体问题：query_type=direct_response，source_preference=none，response_type=answer，direct_response_kind=out_of_scope。
3. 半导体概念、原理、稳定事实、单点解释、区别或知识性比较：query_type=rag_qa，source_preference=auto。普通回答使用response_type=answer；用户明确要求知识型报告时使用response_type=report。
4. 半导体最新价格、量产状态、良率、供应商、路线图和供应链等时效信息：query_type=rag_qa，source_preference=web；用户明确要求报告时使用response_type=report。
5. 只有用户需要在具体业务目标、应用场景、优先级或约束下选择、推荐或采用半导体技术方案时，才使用query_type=technology_selection，source_preference=auto，response_type=report。
6. 仅询问两个方案的原理、区别或一般优缺点，不属于技术选型；报告形式本身不代表技术选型。
7. 只有决策对象属于半导体材料、器件、工艺或技术方案时，才能进入technology_selection；其他领域的比较属于out_of_scope。
只返回JSON：
{{"query_type":"rag_qa","source_preference":"auto","response_type":"answer","direct_response_kind":null,"reason":"简短路由理由"}}。
当 query_type=direct_response 时，direct_response_kind 只能是 identity、greeting、capabilities 或 out_of_scope。
""",
        ),
        ("human", "{query}"),
    ]
)
query_classification_chain = query_classification_prompt | llm


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
    "capabilities": "能力或使用帮助",
}

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
    "out_of_scope": (
        "这个问题超出了当前半导体技术顾问的服务范围。"
        "你可以向我询问半导体知识、器件技术、制造工艺或技术选型问题。"
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


def _friendly_error(state, message):
    warnings = list(dict.fromkeys(state.get("warnings", []) + [message]))
    return {
        "status": "error",
        "warnings": warnings,
        "child_graph_result": {},
    }


def classify_query(state):
    """Classify one query into direct, RAG, or technology selection."""
    report_format = state.get("report_format", "markdown")
    if report_format not in {"markdown", "docx"}:
        return {
            "query_type": None,
            "source_preference": "auto",
            "response_type": "answer",
            "report_format": "markdown",
            "routing_reason": "",
            "status": "error",
            "warnings": ["report_format 仅支持 markdown 或 docx。"],
        }
    direct_kind = _fast_direct_response_kind(state["query"])
    if direct_kind is not None:
        return _direct_classification(direct_kind, report_format)
    if _is_explicit_technology_selection(state["query"]):
        return {
            "query_type": "technology_selection",
            "source_preference": "auto",
            "response_type": "report",
            "report_format": report_format,
            "routing_reason": "明确的半导体技术比较或选型请求",
            "status": "running",
            "warnings": [],
        }
    try:
        response = query_classification_chain.invoke({"query": state["query"]})
        content = response.content if hasattr(response, "content") else str(response)
        decision = QueryRoutingDecision.model_validate(
            parse_json_object(content)
        )
    except Exception:
        log.exception("---查询类型识别失败---")
        return {
            "query_type": None,
            "source_preference": "auto",
            "response_type": "answer",
            "report_format": report_format,
            "routing_reason": "",
            "status": "error",
            "warnings": ["暂时无法识别请求类型，请稍后重试。"],
        }
    result = {
        "query_type": decision.query_type,
        "source_preference": (
            "web"
            if decision.query_type == "rag_qa"
            and _requires_web_source(state["query"])
            else decision.source_preference
        ),
        "response_type": decision.response_type,
        "report_format": report_format,
        "routing_reason": decision.reason,
        "status": "running",
        "warnings": [],
    }
    if decision.query_type == "direct_response":
        result["direct_response_kind"] = decision.direct_response_kind
    return result


def respond_directly(state):
    """Return one fixed domain-safe response without external calls."""
    kind = state.get("direct_response_kind", "out_of_scope")
    answer = _DIRECT_RESPONSE_TEMPLATES.get(
        kind,
        _DIRECT_RESPONSE_TEMPLATES["out_of_scope"],
    )
    return {
        "direct_answer": answer,
        "status": "completed",
    }


def invoke_rag_qa(state):
    """Adapt the unified query state to the ordinary RAG graph."""
    try:
        result = rag_qa_graph.invoke(
            {
                "question": state["query"],
                "original_question": state["query"],
                "source_preference": state.get("source_preference", "auto"),
                "response_type": state.get("response_type", "answer"),
                "transform_count": 0,
                "generation_attempt_count": 0,
                "web_search_attempt_count": 0,
                "web_search_queries": [],
            }
        )
    except Exception:
        log.exception("---普通RAG工作流执行失败---")
        return _friendly_error(
            state,
            "普通问答暂时不可用，请稍后重试。",
        )
    return {"child_graph_result": result}


def invoke_technology_selection(state):
    """Adapt the unified query state to the technology-selection graph."""
    try:
        result = technology_selection_graph.invoke(
            {
                "user_request": state["query"],
                "clarification_answers": [],
            }
        )
    except Exception:
        log.exception("---技术选型工作流执行失败---")
        return _friendly_error(
            state,
            "技术选型服务暂时不可用，请稍后重试。",
        )
    return {"child_graph_result": result}


def _rag_sources(documents):
    sources = []
    seen = set()
    for document in documents:
        metadata = getattr(document, "metadata", {})
        if metadata.get("is_error"):
            continue
        source = str(metadata.get("source", metadata.get("url", ""))).strip()
        url = str(metadata.get("url", source)).strip()
        title = str(metadata.get("title", "")).strip()
        key = url or source or title
        if not key or key in seen:
            continue
        seen.add(key)
        sources.append(
            {
                "evidence_id": "",
                "title": title,
                "url": url,
                "source": source,
                "retrieval_source": str(
                    metadata.get("retrieval_source", "")
                ),
            }
        )
    return sources


def _selection_sources(records):
    sources = []
    seen = set()
    for record in records:
        evidence_id = str(record.get("evidence_id", "")).strip()
        source = str(record.get("source", record.get("url", ""))).strip()
        url = str(record.get("url", source)).strip()
        key = evidence_id or url or source
        if not key or key in seen:
            continue
        seen.add(key)
        sources.append(
            {
                "evidence_id": evidence_id,
                "title": str(record.get("title", "")).strip(),
                "url": url,
                "source": source,
                "retrieval_source": str(
                    record.get("retrieval_source", "")
                ),
            }
        )
    return sources


def _selection_details(result):
    keys = (
        "requirements",
        "candidates",
        "feasible_candidates",
        "excluded_candidates",
        "evidence_gaps",
        "evidence_round",
        "evaluations",
        "risk_items",
        "recommendation",
        "recommendation_review",
        "review_round",
    )
    details = {key: result[key] for key in keys if key in result}
    if "evidence_records" in result:
        details["evidence_record_count"] = len(
            result.get("evidence_records", [])
        )
    return details


def _export_report_if_requested(response):
    if response["report_format"] != "docx":
        return response
    if response["status"] in {"awaiting_user", "error"} or not response["report"]:
        return response
    try:
        from src.infrastructure.reporting.docx import export_report_docx

        response["report_path"] = export_report_docx(
            response["report"],
            response["query"],
        )
    except Exception:
        log.exception("---DOCX报告导出失败---")
        response["warnings"] = list(
            dict.fromkeys(
                response["warnings"]
                + ["DOCX 报告导出失败，已保留 Markdown 报告。"]
            )
        )
    return response


def build_query_response(state):
    """Convert either child graph result into one JSON-safe response."""
    response = {
        "query": state.get("query", ""),
        "query_type": state.get("query_type"),
        "source_preference": state.get("source_preference", "auto"),
        "response_type": state.get("response_type", "answer"),
        "routing_reason": state.get("routing_reason", ""),
        "status": state.get("status", "error"),
        "answer": "",
        "report": "",
        "report_data": {},
        "report_format": state.get("report_format", "markdown"),
        "report_path": None,
        "clarification_questions": [],
        "sources": [],
        "details": {},
        "warnings": list(state.get("warnings", [])),
    }
    if state.get("status") == "error":
        return _export_report_if_requested(response)

    if state.get("query_type") == "direct_response":
        response["answer"] = str(state.get("direct_answer", ""))
        response["status"] = "completed"
        response["details"] = {
            "direct_response_kind": state.get(
                "direct_response_kind",
                "out_of_scope",
            )
        }
        return _export_report_if_requested(response)

    result = state.get("child_graph_result", {})
    if state.get("query_type") == "rag_qa":
        retrieval_error = result.get("retrieval_error")
        if retrieval_error:
            response["status"] = "error"
            warning = (
                "未取得足以回答问题的相关资料。"
                if retrieval_error in {
                    "web_search_no_relevant_results",
                    "web_search_retry_exhausted",
                }
                else "未获取到可用检索证据，无法生成可验证的回答。"
            )
            response["warnings"] = list(
                dict.fromkeys(
                    response["warnings"]
                    + [warning]
                )
            )
            response["details"] = {
                key: result.get(key)
                for key in (
                    "retrieval_source",
                    "retrieval_error",
                    "web_search_attempt_count",
                    "web_search_query",
                    "web_search_queries",
                    "web_relevant_result_count",
                )
                if key in result
            }
            return _export_report_if_requested(response)
        generation = str(result.get("generation", ""))
        if state.get("response_type") == "report":
            response["report"] = generation
        else:
            response["answer"] = generation
        response["status"] = "completed"
        response["sources"] = _rag_sources(result.get("documents", []))
        response["details"] = {
            key: result.get(key)
            for key in (
                "retrieval_source",
                "transform_count",
                "generation_attempt_count",
                "graph_error",
                "web_search_attempt_count",
                "web_search_query",
                "web_search_queries",
                "web_relevant_result_count",
            )
            if key in result
        }
        return _export_report_if_requested(response)

    response["status"] = result.get("status", "completed")
    response["report"] = str(result.get("report", ""))
    response["report_data"] = dict(result.get("report_data", {}) or {})
    response["clarification_questions"] = list(
        result.get("clarification_questions", [])
    )
    response["sources"] = _selection_sources(
        result.get("evidence_records", [])
    )
    response["details"] = _selection_details(result)
    response["warnings"] = list(
        dict.fromkeys(response["warnings"] + result.get("warnings", []))
    )
    return _export_report_if_requested(response)

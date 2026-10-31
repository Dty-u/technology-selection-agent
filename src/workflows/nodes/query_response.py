"""Normalize child-graph results into the public query response."""

import re
from datetime import datetime, timezone
from uuid import uuid4

from src.observability.logging import log


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


_NO_RELEVANT_EVIDENCE_ANSWER = (
    "我暂时没有找到足以回答这个问题的可靠资料，所以先不给出猜测的答案。"
    "你可以补充更具体的对象或背景，比如具体的芯片、器件或工艺名称，我再帮你查。"
)


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


def _append_selection_report(response, report_id=None):
    """Archive one completed selection report in checkpointed session state."""
    if (
        response.get("query_type") != "technology_selection"
        or response.get("response_type") != "report"
        or response.get("status") not in {"completed", "needs_human_review"}
        or not str(response.get("report", "")).strip()
    ):
        return response
    requirements = response.get("report_data", {}).get("requirements", {})
    business_goal = ""
    if isinstance(requirements, dict):
        business_goal = str(requirements.get("business_goal", "")).strip()
    title = re.sub(r"\s+", " ", business_goal)
    if title and not title.endswith("报告"):
        title = (
            f"{title}报告"
            if title.endswith("技术选型")
            else f"{title}技术选型报告"
        )
    if len(title) > 48:
        title = f"{title[:48]}…"
    normalized_report_id = str(report_id or "").strip() or uuid4().hex
    history = list(response.get("report_history", []))
    if any(
        str(item.get("report_id", "") or "") == normalized_report_id
        for item in history
        if isinstance(item, dict)
    ):
        return response
    response["report_history"] = [
        *history,
        {
            "report_id": normalized_report_id,
            "title": title or "技术选型报告",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "format": "docx" if response.get("report_path") else "markdown",
            "report": response["report"],
            "report_path": response.get("report_path"),
        },
    ]
    return response


def build_query_response(state, config=None):
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
        "report_history": list(state.get("report_history", []) or []),
        "clarification_questions": [],
        "clarification_answers": [],
        "clarification_history": [],
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
            # Finding nothing relevant is a normal outcome the user can act on;
            # only an unavailable retrieval service is an error.
            no_relevant_evidence = retrieval_error in {
                "web_search_no_relevant_results",
                "web_search_retry_exhausted",
            }
            if no_relevant_evidence:
                response["status"] = "completed"
                response["answer"] = _NO_RELEVANT_EVIDENCE_ANSWER
            else:
                response["status"] = "error"
            warning = (
                "未取得足以回答问题的相关资料。"
                if no_relevant_evidence
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
    response["clarification_answers"] = list(
        result.get("clarification_answers", [])
    )
    response["clarification_history"] = list(
        result.get("clarification_history", [])
    )
    response["sources"] = _selection_sources(
        result.get("evidence_records", [])
    )
    response["details"] = _selection_details(result)
    response["warnings"] = list(
        dict.fromkeys(response["warnings"] + result.get("warnings", []))
    )
    configurable = dict((config or {}).get("configurable", {}) or {})
    return _append_selection_report(
        _export_report_if_requested(response),
        report_id=configurable.get("request_id"),
    )

"""Long-term-memory recall and write nodes around the existing child graphs."""

import json
import re
from typing import Any, Protocol, Sequence

from src.config.settings import (
    MEMORY_EXACT_RECALL_LIMIT,
    MEMORY_SEMANTIC_RECALL_LIMIT,
)
from src.config.bootstrap import (
    mark_resource_failed,
    mark_resource_ready,
    resource_retry_allowed,
)
from src.infrastructure.memory.models import (
    MemoryCandidate,
    MemoryItem,
    SemanticMemoryHit,
)
from src.infrastructure.memory.repository import MemoryRepository
from src.observability.logging import log


class SemanticMemoryReader(Protocol):
    def search_hits(self, **kwargs) -> Sequence[SemanticMemoryHit]: ...


def _scope(config: dict[str, Any] | None) -> dict[str, str]:
    configurable = dict((config or {}).get("configurable", {}) or {})
    return {
        key: str(configurable.get(key, "") or "").strip()
        for key in ("user_id", "thread_id", "project_id", "request_id")
    }


def _scope_error(scope: dict[str, str], *, require_project: bool) -> str | None:
    required = ["user_id", "thread_id", "request_id"]
    if require_project:
        required.append("project_id")
    missing = [key for key in required if not scope.get(key)]
    if missing:
        return "长期记忆需要显式配置：" + ", ".join(missing)
    return None


def _format_memory_context(items: Sequence[MemoryItem]) -> str:
    if not items:
        return ""
    lines = [
        "以下是已确认的用户/项目记忆，用于个性化与补全上下文；"
        "它不是知识库证据，不能替代本轮检索证据："
    ]
    lines.extend(f"- [{item.memory_type}] {item.content}" for item in items)
    return "\n".join(lines)


def make_recall_memory(
    repository: MemoryRepository,
    semantic_reader: SemanticMemoryReader | None = None,
):
    """Build a node that reads MySQL first, then the semantic projection."""

    def recall_memory(state, config=None):
        query_type = state.get("query_type")
        if query_type == "direct_response":
            return {
                "memory_context": "",
                "recalled_memory_ids": [],
                "historical_memory_context": "",
            }
        require_project = query_type == "technology_selection"
        scope = _scope(config)
        error = _scope_error(scope, require_project=require_project)
        if error:
            return {
                "status": "error",
                "warnings": list(state.get("warnings", [])) + [error],
                "memory_context": "",
                "recalled_memory_ids": [],
            }

        if query_type == "technology_selection":
            exact_types = (
                "project_fact",
                "selection_constraint",
            )
            historical_types = ("selection_decision", "selection_case")
            project_id = scope["project_id"]
        else:
            exact_types = (
                "user_preference",
                "user_fact",
                "conversation_summary",
            )
            historical_types = ()
            project_id = None

        warnings = list(state.get("warnings", []))
        try:
            exact_items = repository.search_authoritative(
                user_id=scope["user_id"],
                project_id=project_id,
                memory_types=exact_types,
                limit=MEMORY_EXACT_RECALL_LIMIT,
            )
            historical_items = (
                repository.search_authoritative(
                    user_id=scope["user_id"],
                    project_id=project_id,
                    memory_types=historical_types,
                    statuses=("approved",),
                    limit=2,
                )
                if historical_types
                else []
            )
        except Exception:
            log.exception("---MySQL长期记忆检索失败---")
            exact_items = []
            historical_items = []
            warnings.append("长期记忆暂时不可用，本轮仅使用当前输入与检索证据。")

        semantic_items: Sequence[MemoryItem] = []
        if (
            semantic_reader is not None
            and exact_items
            and resource_retry_allowed("memory")
        ):
            try:
                semantic_hits = semantic_reader.search_hits(
                    query=state.get("query", ""),
                    user_id=scope["user_id"],
                    project_id=project_id,
                    memory_types=exact_types,
                    statuses=("confirmed", "approved"),
                    limit=MEMORY_SEMANTIC_RECALL_LIMIT,
                )
                hydrated = repository.get_by_ids(
                    [hit.memory_id for hit in semantic_hits]
                )
                hits_by_id = {hit.memory_id: hit for hit in semantic_hits}
                semantic_items = [
                    item
                    for item in hydrated
                    if item.user_id == scope["user_id"]
                    and item.project_id == project_id
                    and item.memory_type in exact_types
                    and item.status in {"confirmed", "approved"}
                    and item.version == hits_by_id[item.memory_id].version
                ]
                mark_resource_ready("memory")
            except Exception:
                mark_resource_failed("memory")
                log.exception("---Milvus语义记忆检索失败---")
                warnings.append("语义记忆投影暂时不可用，已继续使用MySQL确认记忆。")

        combined: list[MemoryItem] = []
        seen = set()
        for item in [*exact_items, *semantic_items]:
            if item.memory_id in seen:
                continue
            seen.add(item.memory_id)
            combined.append(item)
        return {
            "memory_context": _format_memory_context(combined),
            "historical_memory_context": _format_memory_context(historical_items),
            "recalled_memory_ids": [item.memory_id for item in combined],
            "warnings": warnings,
            "memory_scope_valid": True,
        }

    return recall_memory


_REMEMBER_PATTERN = re.compile(r"(?:请|帮我)?记住[，,:：\s]*(.+)", re.IGNORECASE)
_PREFERENCE_PATTERN = re.compile(r"偏好|喜欢|希望|习惯|请用|使用.+回答")


def stage_memory_candidates(state, config=None):
    """Stage only explicit facts plus a pending, unapproved selection case."""
    if state.get("status") != "completed" or state.get("query_type") == "direct_response":
        return {"memory_candidates": []}
    scope = _scope(config)
    error = _scope_error(
        scope,
        require_project=state.get("query_type") == "technology_selection",
    )
    if error:
        return {
            "memory_candidates": [],
            "warnings": list(state.get("warnings", [])) + [error],
        }

    candidates: list[MemoryCandidate] = []
    query = str(state.get("query", "")).strip()
    explicit_match = _REMEMBER_PATTERN.search(query)
    if explicit_match:
        content = explicit_match.group(1).strip(" ，,。.!！")
        memory_type = (
            "user_preference"
            if _PREFERENCE_PATTERN.search(content)
            else "user_fact"
        )
        candidates.append(
            MemoryCandidate(
                idempotency_key=f"{scope['request_id']}:{memory_type}:0",
                user_id=scope["user_id"],
                thread_id=scope["thread_id"],
                memory_type=memory_type,
                content=content,
                status="confirmed",
                source_kind="explicit_user_statement",
                confidence=1,
                metadata={"request_id": scope["request_id"]},
            )
        )

    if state.get("query_type") == "technology_selection":
        recommendation = dict(
            (state.get("report_data", {}) or {}).get("recommendation", {}) or {}
        )
        if recommendation:
            candidates.append(
                MemoryCandidate(
                    idempotency_key=(
                        f"{scope['request_id']}:selection_case:0"
                    ),
                    user_id=scope["user_id"],
                    thread_id=scope["thread_id"],
                    project_id=scope["project_id"],
                    memory_type="selection_case",
                    content=json.dumps(recommendation, ensure_ascii=False),
                    status="pending",
                    source_kind="workflow_recommendation",
                    confidence=1,
                    metadata={
                        "request_id": scope["request_id"],
                        "requires_explicit_approval": True,
                    },
                )
            )
    return {
        "memory_candidates": [
            candidate.model_dump(mode="json") for candidate in candidates
        ]
    }


def make_persist_memory_candidates(repository: MemoryRepository):
    """Build the post-response MySQL authority/outbox write node."""

    def persist_memory_candidates(state):
        count = 0
        warnings = list(state.get("warnings", []))
        for raw_candidate in state.get("memory_candidates", []) or []:
            try:
                repository.save_candidate(MemoryCandidate.model_validate(raw_candidate))
                count += 1
            except Exception:
                log.exception("---长期记忆候选写入失败---")
                warnings.append("本轮回答已完成，但长期记忆写入失败，可稍后重试。")
                break
        return {
            "memory_write_count": count,
            "warnings": warnings,
            "memory_candidates": [],
            "child_graph_result": {},
            "memory_context": "",
            "historical_memory_context": "",
            "short_term_context": "",
        }

    return persist_memory_candidates

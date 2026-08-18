"""Thread-scoped short-term memory stored in LangGraph checkpoint state."""

from typing import Any, Sequence
from uuid import NAMESPACE_URL, uuid5

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    RemoveMessage,
)
from langgraph.graph.message import add_messages

from src.config.settings import SHORT_TERM_TURN_LIMIT


_MESSAGES_PER_TURN = 2
_MAX_SHORT_TERM_MESSAGES = SHORT_TERM_TURN_LIMIT * _MESSAGES_PER_TURN
_ROLE_LABELS = {
    "human": "用户",
    "user": "用户",
    "ai": "助手",
    "assistant": "助手",
    "system": "系统",
    "tool": "工具",
    "function": "工具",
}


def checkpoint_message_id(
    scope: dict[str, str],
    kind: str,
) -> str:
    """Return a stable UUID for one request-scoped checkpoint message."""
    return str(
        uuid5(
            NAMESPACE_URL,
            ":".join(
                (
                    "rag-agent-checkpoint-message",
                    scope.get("user_id", ""),
                    scope.get("thread_id", ""),
                    scope.get("request_id", ""),
                    kind,
                )
            ),
        )
    )


def checkpoint_scope(config: dict[str, Any] | None) -> dict[str, str]:
    configurable = dict((config or {}).get("configurable", {}) or {})
    return {
        key: str(configurable.get(key, "") or "").strip()
        for key in ("user_id", "thread_id", "request_id")
    }


def format_short_term_messages(messages: Sequence[BaseMessage]) -> str:
    """Render complete message contents without model summarization."""
    lines: list[str] = []
    for message in messages:
        content = message.content
        text = content if isinstance(content, str) else str(content)
        if not text.strip():
            continue
        message_type = str(getattr(message, "type", "") or "").lower()
        role = str(getattr(message, "role", "") or "").lower()
        label = _ROLE_LABELS.get(message_type) or _ROLE_LABELS.get(role) or "消息"
        lines.append(f"[{label}] {text}")
    return "\n".join(lines)


def message_updates_with_trim(
    current_messages: Sequence[BaseMessage],
    additions: Sequence[BaseMessage],
) -> list[BaseMessage]:
    """Append or replace messages, then remove messages before the window."""
    merged = add_messages(list(current_messages), list(additions))
    if _MAX_SHORT_TERM_MESSAGES <= 0:
        expired = merged
    else:
        expired = merged[:-_MAX_SHORT_TERM_MESSAGES]
    removals = [
        RemoveMessage(id=message.id)
        for message in expired
        if getattr(message, "id", None)
    ]
    return [*additions, *removals]


def prepare_short_term_memory(state, config=None):
    """Derive prior context and append the current query to checkpoint state."""
    scope = checkpoint_scope(config)
    current_user_id = scope["user_id"]
    configurable = dict((config or {}).get("configurable", {}) or {})
    current_project_id = str(configurable.get("project_id", "") or "").strip()
    session_user_id = str(state.get("session_user_id", "") or "").strip()
    session_project_id = str(state.get("session_project_id", "") or "").strip()
    if session_user_id and current_user_id and session_user_id != current_user_id:
        return {
            "status": "error",
            "query_type": None,
            "short_term_context": "",
            "warnings": [
                "当前 thread_id 已属于其他用户，已拒绝读取该会话的短期记忆。"
            ],
        }

    if session_project_id and session_project_id != current_project_id:
        return {
            "status": "error",
            "query_type": None,
            "short_term_context": "",
            "warnings": [
                "current thread_id belongs to another project_id; request rejected"
            ],
        }

    messages = list(state.get("messages", []) or [])
    retry_message_ids = {
        checkpoint_message_id(scope, kind)
        for kind in (
            "user-query",
            "user-resume",
            "assistant-response",
            "assistant-clarification",
        )
    }
    prior_messages = [
        message
        for message in messages
        if getattr(message, "id", None) not in retry_message_ids
    ]
    if _MAX_SHORT_TERM_MESSAGES > 0:
        prior_messages = prior_messages[-_MAX_SHORT_TERM_MESSAGES:]
    else:
        prior_messages = []

    update: dict[str, Any] = {
        "status": "running",
        "warnings": [],
        "short_term_context": format_short_term_messages(prior_messages),
    }
    if not session_user_id and current_user_id:
        update["session_user_id"] = current_user_id
    if not session_project_id and current_project_id:
        update["session_project_id"] = current_project_id

    query = str(state.get("query", "") or "")
    if query.strip():
        update["messages"] = [
            HumanMessage(
                content=query,
                id=checkpoint_message_id(scope, "user-query"),
            )
        ]
    return update


def finalize_short_term_memory(state, config=None):
    """Append the stable assistant response and keep only the latest six turns."""
    content = str(state.get("answer") or state.get("report") or "")
    if not content.strip():
        return {}
    scope = checkpoint_scope(config)
    assistant_message = AIMessage(
        content=content,
        id=checkpoint_message_id(scope, "assistant-response"),
    )
    return {
        "messages": message_updates_with_trim(
            state.get("messages", []) or [],
            [assistant_message],
        )
    }

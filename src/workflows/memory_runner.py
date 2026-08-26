"""Public runner for checkpointed, memory-enabled LangGraph workflows."""

import asyncio
from collections.abc import Mapping
from time import perf_counter
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.errors import GraphInterrupt
from langgraph.types import Command

from src.infrastructure.memory.repository import MemoryRepository
from src.observability.logging import log
from src.workflows.nodes.short_term_memory import (
    checkpoint_message_id,
    checkpoint_scope,
    message_updates_with_trim,
)


class MemoryAwareRunner:
    """Enforce explicit scope IDs around a checkpoint-backed workflow."""

    def __init__(
        self,
        *,
        graph: Any,
        repository: MemoryRepository | None = None,
        checkpointer: Any | None = None,
    ) -> None:
        self.graph = graph
        self.repository = repository
        self.checkpointer = checkpointer

    @staticmethod
    def _scope(config: dict[str, Any]) -> dict[str, str]:
        scope = checkpoint_scope(config)
        missing = [key for key, value in scope.items() if not value]
        if missing:
            raise ValueError(
                "memory-enabled invocation requires explicit " + ", ".join(missing)
            )
        return scope

    @staticmethod
    def _user_content(payload: Any) -> str:
        if isinstance(payload, Command):
            return str(payload.resume or "")
        if isinstance(payload, dict):
            return str(payload.get("query", "") or "")
        return ""

    @staticmethod
    def _merge_resume_command(
        payload: Command,
        scope: dict[str, str],
        content: str,
    ) -> Command:
        if payload.update is None:
            update: dict[str, Any] = {}
        elif isinstance(payload.update, Mapping):
            update = dict(payload.update)
        else:
            raise ValueError("Command.update must be a mapping when resuming this graph")

        existing = update.get("messages", [])
        if isinstance(existing, BaseMessage):
            messages = [existing]
        else:
            messages = list(existing or [])
        messages.append(
            HumanMessage(
                content=content,
                id=checkpoint_message_id(scope, "user-resume"),
            )
        )
        update["messages"] = messages
        return Command(
            graph=payload.graph,
            update=update,
            resume=payload.resume,
            goto=payload.goto,
        )

    @staticmethod
    def _clarification_questions(result: Any) -> list[str]:
        if not isinstance(result, Mapping):
            return []
        questions: list[str] = []
        for interrupt_value in result.get("__interrupt__", ()) or ():
            payload = getattr(interrupt_value, "value", interrupt_value)
            if not isinstance(payload, Mapping):
                continue
            for question in payload.get("clarification_questions", []) or []:
                normalized = str(question or "").strip()
                if normalized and normalized not in questions:
                    questions.append(normalized)
        return questions

    def _record_clarification_questions(
        self,
        result: Any,
        *,
        config: dict[str, Any],
        scope: dict[str, str],
    ) -> None:
        questions = self._clarification_questions(result)
        if not questions:
            return
        snapshot = self.graph.get_state(config)
        current_messages = list(snapshot.values.get("messages", []) or [])
        message = AIMessage(
            content="\n".join(questions),
            id=checkpoint_message_id(scope, "assistant-clarification"),
        )
        self.graph.update_state(
            config,
            {
                "messages": message_updates_with_trim(
                    current_messages,
                    [message],
                )
            },
        )

    def _assert_resume_owner(
        self,
        *,
        config: dict[str, Any],
        user_id: str,
    ) -> None:
        """Protect native resume paths, which do not re-enter the START node."""
        snapshot = self.graph.get_state(config)
        values = getattr(snapshot, "values", None)
        if not isinstance(values, Mapping):
            return
        session_user_id = str(values.get("session_user_id", "") or "").strip()
        if session_user_id and session_user_id != user_id:
            raise PermissionError(
                "current thread_id belongs to another user; resume rejected"
            )
        configurable = dict(config.get("configurable", {}) or {})
        project_id = str(configurable.get("project_id", "") or "").strip()
        session_project_id = str(
            values.get("session_project_id", "") or ""
        ).strip()
        if session_project_id and session_project_id != project_id:
            raise PermissionError(
                "current thread_id belongs to another project_id; resume rejected"
            )

    async def aget_state(self, config: dict[str, Any]):
        """Read checkpoint state through the graph's async saver contract."""
        return await self.graph.aget_state(config)

    async def _record_clarification_questions_async(
        self,
        result: Any,
        *,
        config: dict[str, Any],
        scope: dict[str, str],
    ) -> None:
        questions = self._clarification_questions(result)
        if not questions:
            return
        snapshot = await self.graph.aget_state(config)
        current_messages = list(snapshot.values.get("messages", []) or [])
        message = AIMessage(
            content="\n".join(questions),
            id=checkpoint_message_id(scope, "assistant-clarification"),
        )
        await self.graph.aupdate_state(
            config,
            {
                "messages": message_updates_with_trim(
                    current_messages,
                    [message],
                )
            },
        )

    async def _assert_resume_owner_async(
        self,
        *,
        config: dict[str, Any],
        user_id: str,
    ) -> None:
        snapshot = await self.graph.aget_state(config)
        values = getattr(snapshot, "values", None)
        if not isinstance(values, Mapping):
            return
        session_user_id = str(values.get("session_user_id", "") or "").strip()
        if session_user_id and session_user_id != user_id:
            raise PermissionError(
                "current thread_id belongs to another user; resume rejected"
            )
        configurable = dict(config.get("configurable", {}) or {})
        project_id = str(configurable.get("project_id", "") or "").strip()
        session_project_id = str(
            values.get("session_project_id", "") or ""
        ).strip()
        if session_project_id and session_project_id != project_id:
            raise PermissionError(
                "current thread_id belongs to another project_id; resume rejected"
            )

    def invoke(self, payload: Any, *, config: dict[str, Any]):
        started_at = perf_counter()
        scope = self._scope(config)
        content = self._user_content(payload)
        if not content.strip():
            raise ValueError("query or Command(resume=...) must not be empty")
        if isinstance(payload, Command):
            self._assert_resume_owner(
                config=config,
                user_id=scope["user_id"],
            )
            graph_payload = self._merge_resume_command(payload, scope, content)
        else:
            graph_payload = dict(payload)
            graph_payload.pop("short_term_context", None)
        try:
            result = self.graph.invoke(graph_payload, config=config)
        except GraphInterrupt:
            log.info(
                "[工作流][invoke] 暂停 | request_id={} | path={} | 耗时={:.2f}s",
                scope["request_id"],
                "resume" if isinstance(payload, Command) else "request",
                perf_counter() - started_at,
            )
            raise
        except BaseException as exc:
            log.error(
                "[工作流][invoke] 失败 | request_id={} | path={} | error={} | 耗时={:.2f}s",
                scope["request_id"],
                "resume" if isinstance(payload, Command) else "request",
                type(exc).__name__,
                perf_counter() - started_at,
            )
            raise
        self._record_clarification_questions(
            result,
            config=config,
            scope=scope,
        )
        query_type = (
            result.get("query_type", "unknown")
            if isinstance(result, Mapping)
            else "unknown"
        )
        log.info(
            "[工作流][invoke] 完成 | request_id={} | path={} | 耗时={:.2f}s",
            scope["request_id"],
            query_type,
            perf_counter() - started_at,
        )
        return result

    async def astream(self, payload: Any, *, config: dict[str, Any]):
        """Yield graph state values without blocking the API event loop.

        The graph remains responsible for deciding each workflow step.  This
        adapter only validates the explicit memory scope, preserves resume
        ownership checks, and records clarification messages after the stream
        reaches its final value.
        """
        started_at = perf_counter()
        scope = self._scope(config)
        content = self._user_content(payload)
        if not content.strip():
            raise ValueError("query or Command(resume=...) must not be empty")
        if isinstance(payload, Command):
            await self._assert_resume_owner_async(
                config=config,
                user_id=scope["user_id"],
            )
            graph_payload = self._merge_resume_command(payload, scope, content)
        else:
            graph_payload = dict(payload)
            graph_payload.pop("short_term_context", None)

        result: Any = None
        try:
            async for result in self.graph.astream(
                graph_payload,
                config=config,
                stream_mode="values",
            ):
                yield result
        except GraphInterrupt:
            log.info(
                "[工作流][astream] 暂停 | request_id={} | path={} | 耗时={:.2f}s",
                scope["request_id"],
                "resume" if isinstance(payload, Command) else "request",
                perf_counter() - started_at,
            )
            raise
        except BaseException as exc:
            log.error(
                "[工作流][astream] 失败 | request_id={} | path={} | error={} | 耗时={:.2f}s",
                scope["request_id"],
                "resume" if isinstance(payload, Command) else "request",
                type(exc).__name__,
                perf_counter() - started_at,
            )
            raise

        await self._record_clarification_questions_async(
            result,
            config=config,
            scope=scope,
        )

    async def aclose_session(self, thread_id: str, *, user_id: str) -> None:
        """Delete one checkpoint thread through the async saver."""
        normalized_thread_id = str(thread_id or "").strip()
        if not normalized_thread_id:
            raise ValueError("thread_id must not be empty")
        if not str(user_id or "").strip():
            raise ValueError("user_id must not be empty")
        if self.checkpointer is None:
            raise RuntimeError("this runner has no checkpoint saver")
        await self.checkpointer.adelete_thread(normalized_thread_id)
        query_type = (
            result.get("query_type", "unknown")
            if isinstance(result, Mapping)
            else "unknown"
        )
        log.info(
            "[工作流][astream] 完成 | request_id={} | path={} | 耗时={:.2f}s",
            scope["request_id"],
            query_type,
            perf_counter() - started_at,
        )

    def close_session(self, thread_id: str, *, user_id: str) -> None:
        """Delete checkpoint state for one explicit thread."""
        normalized_thread_id = str(thread_id or "").strip()
        if not normalized_thread_id:
            raise ValueError("thread_id must not be empty")
        normalized_user_id = str(user_id or "").strip()
        if not normalized_user_id:
            raise ValueError("user_id must not be empty")
        if self.checkpointer is None:
            raise RuntimeError("this runner has no checkpoint saver")
        self.checkpointer.delete_thread(normalized_thread_id)

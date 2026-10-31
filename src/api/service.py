"""Application service adapting HTTP sessions to the LangGraph runner."""

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import AsyncExitStack, asynccontextmanager
import inspect
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any
from uuid import NAMESPACE_URL, UUID, uuid4, uuid5

from langgraph.errors import GraphInterrupt
from langgraph.types import Command

from src.config.settings import REPORT_OUTPUT_DIR
from src.api.progress import WorkflowProgressRegistry
from src.observability.answer_stream import observe_answer_stream
from src.observability.logging import log
from src.api.schemas import (
    CancelSessionResponse,
    ImportSessionRequest,
    ImportSessionResponse,
    MessageListResponse,
    MessageRequest,
    ProjectListResponse,
    ProjectResponse,
    ReportContentResponse,
    ReportLibraryItem,
    ReportListResponse,
    ReportSummary,
    SessionListResponse,
    SessionResponse,
    SessionSummary,
    StoredMessage,
)

SESSION_TITLE_MAX_CHARACTERS = 32
MESSAGE_HISTORY_LIMIT = 1000
# Reports live in checkpoints, so listing them reads one checkpoint per session.
REPORT_LIBRARY_SESSION_LIMIT = 500
REPORT_LIBRARY_CONCURRENCY = 8


def _session_title(message: str) -> str:
    normalized = " ".join(message.split())
    if len(normalized) <= SESSION_TITLE_MAX_CHARACTERS:
        return normalized
    return f"{normalized[:SESSION_TITLE_MAX_CHARACTERS]}…"


def _timestamp(value: datetime) -> str:
    """Render stored UTC times like the browser's Date.toISOString()."""
    return f"{value.strftime('%Y-%m-%dT%H:%M:%S')}.{value.microsecond // 1000:03d}Z"


def _utc_naive(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value
    return value.astimezone(timezone.utc).replace(tzinfo=None)


def _import_request_id(thread_id: str, message_id: str, request_id: str | None) -> str:
    try:
        return str(UUID(str(request_id)))
    except ValueError:
        return str(uuid5(NAMESPACE_URL, f"{thread_id}/{message_id}"))


@dataclass(frozen=True)
class ReportDownload:
    filename: str
    content: str | None = None
    path: Path | None = None


class SessionNotFoundError(Exception):
    """Hide missing sessions and scope mismatches behind one public error."""


class SessionBusyError(Exception):
    """Reject overlapping writes to one checkpoint thread."""


class SessionConflictError(Exception):
    """Reject creation over an existing client-preallocated thread."""


class WorkflowCancelled(Exception):
    """End one workflow between graph steps after a scoped cancellation request."""


STREAM_KEEPALIVE_SECONDS = 15.0
_STREAM_DONE = object()


def _sse(event: str, data: Any) -> str:
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    return f"event: {event}\ndata: {payload}\n\n"


class SessionService:
    def __init__(
        self,
        runner: Any,
        *,
        report_output_dir: Path = REPORT_OUTPUT_DIR,
        progress_registry: WorkflowProgressRegistry | None = None,
        session_repository: Any = None,
    ):
        self.runner = runner
        self.session_repository = session_repository
        self.report_output_dir = Path(report_output_dir).resolve()
        self.progress_registry = progress_registry or WorkflowProgressRegistry()
        self._locks: dict[str, asyncio.Lock] = {}
        self._locks_guard = Lock()
        self._stream_tasks: set[asyncio.Task] = set()

    @asynccontextmanager
    async def _session_slot(self, thread_id: str):
        with self._locks_guard:
            lock = self._locks.setdefault(thread_id, asyncio.Lock())
        if lock.locked():
            raise SessionBusyError
        await lock.acquire()
        try:
            yield
        finally:
            lock.release()
            with self._locks_guard:
                if self._locks.get(thread_id) is lock and not lock.locked():
                    self._locks.pop(thread_id, None)

    async def _history(self, method: str, **kwargs: Any) -> Any:
        """Call the blocking history store; any store failure means it is unavailable."""
        if self.session_repository is None:
            raise ConnectionError("session history is not configured")
        try:
            return await asyncio.to_thread(
                getattr(self.session_repository, method),
                **kwargs,
            )
        except Exception as exc:
            raise ConnectionError(
                f"session history {method} failed: {type(exc).__name__}"
            ) from exc

    async def _begin_turn(
        self,
        *,
        user_id: str,
        project_id: str,
        thread_id: str,
        request_id: str,
        message: MessageRequest,
    ) -> None:
        if self.session_repository is None:
            return
        await self._history(
            "begin_turn",
            user_id=user_id,
            project_id=project_id,
            thread_id=thread_id,
            request_id=request_id,
            content=message.message,
            title=_session_title(message.message),
        )

    async def _finish_turn(
        self,
        *,
        thread_id: str,
        request_id: str,
        response: SessionResponse,
    ) -> None:
        """Keep the finished answer usable even if its history write fails."""
        if self.session_repository is None:
            return
        content = ""
        kind = "message"
        if response.status == "awaiting_user":
            content = "\n".join(response.clarification_questions)
            kind = "clarification"
        elif response.status != "cancelled":
            content = response.answer.strip() or response.report.strip()
        for attempt in (1, 2):
            try:
                await self._history(
                    "finish_turn",
                    thread_id=thread_id,
                    request_id=request_id,
                    status=response.status,
                    delivery="cancelled" if response.status == "cancelled" else "sent",
                    assistant_content=content,
                    kind=kind,
                    metadata={
                        "query_type": response.query_type,
                        "sources": response.sources,
                    },
                )
                return
            except ConnectionError as exc:
                log.error(
                    "[API] session history write failed | attempt={} | error={}",
                    attempt,
                    exc,
                )

    async def _fail_turn(self, *, thread_id: str, request_id: str) -> None:
        if self.session_repository is None:
            return
        try:
            await self._history(
                "fail_turn",
                thread_id=thread_id,
                request_id=request_id,
            )
        except ConnectionError as exc:
            log.error("[API] session history cleanup failed | error={}", exc)

    async def list_projects(self, *, user_id: str) -> ProjectListResponse:
        rows = await self._history("list_projects", user_id=user_id)
        return ProjectListResponse(projects=[self._project(row) for row in rows])

    async def ensure_project(
        self,
        *,
        user_id: str,
        project_id: str,
        name: str | None,
    ) -> ProjectResponse:
        row = await self._history(
            "ensure_project",
            user_id=user_id,
            project_id=project_id,
            name=str(name or "").strip() or None,
        )
        return self._project(row)

    @staticmethod
    def _project(row: Mapping[str, Any]) -> ProjectResponse:
        return ProjectResponse(
            project_id=row["project_id"],
            name=row["name"],
            created_at=_timestamp(row["created_at"]),
            updated_at=_timestamp(row["updated_at"]),
        )

    async def _current_status(
        self,
        row: Mapping[str, Any],
        *,
        user_id: str,
        project_id: str,
    ) -> str:
        """A stored 'running' only holds while this process is running the turn."""
        thread_id = row["thread_id"]
        if (
            row["status"] != "running"
            or self.progress_registry.snapshot(thread_id)["active"]
        ):
            return row["status"]
        config = self._config(
            user_id=user_id,
            project_id=project_id,
            thread_id=thread_id,
        )
        try:
            snapshot = await self._get_state(config)
            response = await self._response(
                snapshot,
                user_id=user_id,
                project_id=project_id,
                thread_id=thread_id,
                request_id=None,
            )
            status = response.status
        except SessionNotFoundError:
            status = "error"
        await self._history("update_status", thread_id=thread_id, status=status)
        return status

    async def list_sessions(
        self,
        *,
        user_id: str,
        project_id: str,
        limit: int,
    ) -> SessionListResponse:
        rows = await self._history(
            "list_sessions",
            user_id=user_id,
            project_id=project_id,
            limit=limit,
        )
        sessions = []
        for row in rows:
            sessions.append(
                SessionSummary(
                    thread_id=row["thread_id"],
                    project_id=row["project_id"],
                    title=row["title"],
                    status=await self._current_status(
                        row,
                        user_id=user_id,
                        project_id=project_id,
                    ),
                    created_at=_timestamp(row["created_at"]),
                    updated_at=_timestamp(row["updated_at"]),
                )
            )
        return SessionListResponse(sessions=sessions)

    async def list_messages(
        self,
        *,
        user_id: str,
        project_id: str,
        thread_id: str,
    ) -> MessageListResponse:
        session = await self._history(
            "get_session",
            user_id=user_id,
            project_id=project_id,
            thread_id=thread_id,
        )
        if session is None:
            raise SessionNotFoundError
        rows = await self._history(
            "list_messages",
            thread_id=thread_id,
            limit=MESSAGE_HISTORY_LIMIT,
        )
        return MessageListResponse(
            messages=[
                StoredMessage(
                    id=str(row["id"]),
                    role=row["role"],
                    content=row["content"],
                    delivery=row["delivery"],
                    kind=row["kind"],
                    request_id=row["request_id"],
                    created_at=_timestamp(row["created_at"]),
                )
                for row in rows
            ]
        )

    async def import_session(
        self,
        *,
        user_id: str,
        project_id: str,
        session: ImportSessionRequest,
    ) -> ImportSessionResponse:
        """Copy browser history once, refusing threads owned by another scope."""
        thread_id = str(session.thread_id)
        snapshot = await self._get_state(
            self._config(user_id=user_id, project_id=project_id, thread_id=thread_id)
        )
        # A session whose checkpoint is already gone can still keep its history.
        if self._values(snapshot) or getattr(snapshot, "next", ()):
            self._assert_scope(snapshot, user_id=user_id, project_id=project_id)
        imported = await self._history(
            "import_session",
            user_id=user_id,
            project_id=project_id,
            thread_id=thread_id,
            title=session.title,
            status=session.status,
            created_at=_utc_naive(session.created_at),
            updated_at=_utc_naive(session.updated_at),
            messages=[
                {
                    "request_id": _import_request_id(
                        thread_id,
                        message.id,
                        message.request_id,
                    ),
                    "role": message.role,
                    "content": message.content,
                    "delivery": message.delivery,
                    "created_at": _utc_naive(message.created_at),
                }
                for message in session.messages
            ],
        )
        return ImportSessionResponse(thread_id=thread_id, imported=imported)

    @staticmethod
    def _config(
        *,
        user_id: str,
        project_id: str,
        thread_id: str,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        configurable = {
            "user_id": user_id,
            "project_id": project_id,
            "thread_id": thread_id,
        }
        if request_id is not None:
            configurable["request_id"] = request_id
        return {"configurable": configurable}

    @staticmethod
    def _values(snapshot: Any) -> dict[str, Any]:
        values = getattr(snapshot, "values", None)
        return dict(values) if isinstance(values, Mapping) else {}

    async def _get_state(self, config: dict[str, Any]) -> Any:
        getter = getattr(self.runner, "aget_state", None)
        if inspect.iscoroutinefunction(getter):
            return await getter(config)
        return await asyncio.to_thread(self.runner.graph.get_state, config)

    async def _update_state(
        self,
        config: dict[str, Any],
        values: dict[str, Any],
    ) -> None:
        updater = getattr(self.runner, "aupdate_state", None)
        if inspect.iscoroutinefunction(updater):
            await updater(config, values)
            return
        await asyncio.to_thread(self.runner.graph.update_state, config, values)

    @staticmethod
    def _interrupt_payloads(source: Any) -> list[Mapping[str, Any]]:
        payloads: list[Mapping[str, Any]] = []
        if isinstance(source, Mapping):
            interrupts = source.get("__interrupt__", ()) or ()
        else:
            interrupts = source or ()
        for interrupt_value in interrupts:
            payload = getattr(interrupt_value, "value", interrupt_value)
            if isinstance(payload, Mapping):
                payloads.append(payload)
        return payloads

    def _snapshot_interrupt_payloads(self, snapshot: Any) -> list[Mapping[str, Any]]:
        payloads: list[Mapping[str, Any]] = []
        for task in getattr(snapshot, "tasks", ()) or ():
            payloads.extend(
                self._interrupt_payloads(getattr(task, "interrupts", ()) or ())
            )
        return payloads

    def _assert_scope(
        self,
        snapshot: Any,
        *,
        user_id: str,
        project_id: str,
    ) -> dict[str, Any]:
        values = self._values(snapshot)
        if not values and not getattr(snapshot, "next", ()):
            raise SessionNotFoundError
        session_user_id = str(values.get("session_user_id", "") or "").strip()
        session_project_id = str(
            values.get("session_project_id", "") or ""
        ).strip()
        if session_user_id and session_user_id != user_id:
            raise SessionNotFoundError
        if session_project_id and session_project_id != project_id:
            raise SessionNotFoundError
        return values

    async def _response(
        self,
        snapshot: Any,
        *,
        user_id: str,
        project_id: str,
        thread_id: str,
        request_id: str | None,
        invocation_result: Any = None,
    ) -> SessionResponse:
        values = self._assert_scope(
            snapshot,
            user_id=user_id,
            project_id=project_id,
        )
        payloads = self._interrupt_payloads(invocation_result)
        if not payloads:
            payloads = self._snapshot_interrupt_payloads(snapshot)
        questions: list[str] = []
        for payload in payloads:
            for question in payload.get("clarification_questions", []) or []:
                normalized = str(question or "").strip()
                if normalized and normalized not in questions:
                    questions.append(normalized)
        if not questions:
            questions = [
                str(question)
                for question in values.get("clarification_questions", []) or []
                if str(question).strip()
            ]
        awaiting_user = bool(getattr(snapshot, "next", ())) or bool(payloads)
        progress = self.progress_registry.snapshot(thread_id)
        if progress["active"]:
            status = "running"
        elif progress["cancelled"]:
            status = "cancelled"
        else:
            status = "awaiting_user" if awaiting_user else values.get("status", "error")
        report_path = await asyncio.to_thread(self._safe_report_path, values)
        report_url = None
        if report_path is not None:
            report_url = (
                f"/api/v1/projects/{project_id}/sessions/{thread_id}/report"
            )
        reports = self._report_summaries(
            values,
            project_id=project_id,
            thread_id=thread_id,
        )
        return SessionResponse(
            request_id=request_id,
            project_id=project_id,
            thread_id=thread_id,
            status=status,
            query_type=values.get("query_type"),
            response_type=values.get("response_type"),
            answer=str(values.get("answer", "") or ""),
            report=str(values.get("report", "") or ""),
            report_data=dict(values.get("report_data", {}) or {}),
            clarification_questions=questions,
            clarification_answers=list(values.get("clarification_answers", []) or []),
            clarification_history=list(values.get("clarification_history", []) or []),
            sources=list(values.get("sources", []) or []),
            warnings=[str(item) for item in values.get("warnings", []) or []],
            report_url=report_url,
            reports=reports,
            progress=progress,
        )

    def _report_summaries(
        self,
        values: Mapping[str, Any],
        *,
        project_id: str,
        thread_id: str,
    ) -> list[ReportSummary]:
        reports: list[ReportSummary] = []
        for raw_item in values.get("report_history", []) or []:
            if not isinstance(raw_item, Mapping):
                continue
            report_id = str(raw_item.get("report_id", "") or "").strip()
            title = str(raw_item.get("title", "") or "").strip()
            created_at = str(raw_item.get("created_at", "") or "").strip()
            report_format = str(raw_item.get("format", "markdown") or "markdown")
            if not report_id or not title or not created_at:
                continue
            if report_format not in {"markdown", "docx"}:
                report_format = "markdown"
            if (
                report_format == "docx"
                and self._safe_report_path_from_raw(raw_item.get("report_path")) is None
            ):
                report_format = "markdown"
            reports.append(
                ReportSummary(
                    report_id=report_id,
                    title=title,
                    created_at=created_at,
                    format=report_format,
                    download_url=(
                        f"/api/v1/projects/{project_id}/sessions/{thread_id}"
                        f"/reports/{report_id}"
                    ),
                )
            )
        return reports

    def _safe_report_path(self, values: Mapping[str, Any]) -> Path | None:
        return self._safe_report_path_from_raw(values.get("report_path"))

    def _safe_report_path_from_raw(self, value: Any) -> Path | None:
        raw_path = str(value or "").strip()
        if not raw_path:
            return None
        candidate = Path(raw_path).resolve()
        try:
            candidate.relative_to(self.report_output_dir)
        except ValueError:
            return None
        if candidate.suffix.casefold() != ".docx" or not candidate.is_file():
            return None
        return candidate

    @staticmethod
    def _download_filename(title: Any, extension: str) -> str:
        """Return a Windows-safe, user-recognizable report download filename."""
        normalized_title = re.sub(r"\s+", " ", str(title or "").strip())
        safe_title = "".join(
            character
            for character in normalized_title
            if character not in '<>:"/\\|?*'
        ).strip(" .")
        return f"{(safe_title or '技术选型报告')[:48]}.{extension}"

    def _current_report_title(
        self,
        values: Mapping[str, Any],
        report_path: Path,
    ) -> str:
        """Use the archived selection summary title for the current artifact."""
        report_data_title = self._selection_title_from_report_data(
            values.get("report_data", {})
        )
        if report_data_title:
            return report_data_title
        for raw_item in reversed(values.get("report_history", []) or []):
            if not isinstance(raw_item, Mapping):
                continue
            stored_path = self._safe_report_path_from_raw(raw_item.get("report_path"))
            if stored_path != report_path:
                continue
            title = str(raw_item.get("title", "") or "").strip()
            if title:
                return title

        return "技术选型报告"

    @staticmethod
    def _selection_title_from_report_data(report_data: Any) -> str:
        requirements = (
            report_data.get("requirements", {})
            if isinstance(report_data, Mapping)
            else {}
        )
        business_goal = (
            str(requirements.get("business_goal", "") or "").strip()
            if isinstance(requirements, Mapping)
            else ""
        )
        if not business_goal:
            return ""
        if business_goal.endswith("报告"):
            return business_goal
        return (
            f"{business_goal}报告"
            if business_goal.endswith("技术选型")
            else f"{business_goal}技术选型报告"
        )

    @classmethod
    def _historical_report_title(cls, report: Mapping[str, Any]) -> str:
        title = str(report.get("title", "") or "").strip()
        content = str(report.get("report", "") or "")
        for line in content.splitlines():
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            if len(cells) >= 2 and cells[0] == "选型主题" and cells[1]:
                return cls._selection_title_from_report_data(
                    {"requirements": {"business_goal": cells[1]}}
                ) or title
        return title or "技术选型报告"

    async def create_session(
        self,
        *,
        user_id: str,
        project_id: str,
        request_id: str,
        message: MessageRequest,
        thread_id: str | None = None,
    ) -> SessionResponse:
        client_preallocated = bool(str(thread_id or "").strip())
        normalized_thread_id = str(thread_id or uuid4())
        async with self._session_slot(normalized_thread_id):
            if client_preallocated:
                config = self._config(
                    user_id=user_id,
                    project_id=project_id,
                    thread_id=normalized_thread_id,
                )
                snapshot = await self._get_state(config)
                if self._values(snapshot) or getattr(snapshot, "next", ()):
                    raise SessionConflictError
            return await self._submit(
                user_id=user_id,
                project_id=project_id,
                thread_id=normalized_thread_id,
                request_id=request_id,
                message=message,
                create=True,
            )

    async def send_message(
        self,
        *,
        user_id: str,
        project_id: str,
        thread_id: str,
        request_id: str,
        message: MessageRequest,
    ) -> SessionResponse:
        async with self._session_slot(thread_id):
            return await self._submit(
                user_id=user_id,
                project_id=project_id,
                thread_id=thread_id,
                request_id=request_id,
                message=message,
                create=False,
            )

    async def _submit(
        self,
        *,
        user_id: str,
        project_id: str,
        thread_id: str,
        request_id: str,
        message: MessageRequest,
        create: bool,
        answer_observer: Callable[[Mapping[str, Any]], None] | None = None,
    ) -> SessionResponse:
        config = self._config(
            user_id=user_id,
            project_id=project_id,
            thread_id=thread_id,
            request_id=request_id,
        )
        if create:
            payload: Any = {
                "query": message.message,
                "report_format": message.report_format,
                "session_user_id": user_id,
                "session_project_id": project_id,
            }
        else:
            before = await self._get_state(config)
            self._assert_scope(before, user_id=user_id, project_id=project_id)
            if getattr(before, "next", ()):
                payload = Command(resume=message.message)
            else:
                payload = {
                    "query": message.message,
                    "report_format": message.report_format,
                }
        await self._begin_turn(
            user_id=user_id,
            project_id=project_id,
            thread_id=thread_id,
            request_id=request_id,
            message=message,
        )
        result: Any = None
        try:
            with self.progress_registry.track(
                thread_id=thread_id,
                request_id=request_id,
                user_id=user_id,
                project_id=project_id,
            ), observe_answer_stream(answer_observer):
                try:
                    async for result in self.runner.astream(payload, config=config):
                        if self.progress_registry.is_cancel_requested(
                            thread_id=thread_id,
                            request_id=request_id,
                        ):
                            raise WorkflowCancelled
                except WorkflowCancelled:
                    pass
                except PermissionError as exc:
                    raise SessionNotFoundError from exc
                except GraphInterrupt:
                    pass
            snapshot = await self._get_state(config)
            response = await self._response(
                snapshot,
                user_id=user_id,
                project_id=project_id,
                thread_id=thread_id,
                request_id=request_id,
                invocation_result=result,
            )
        except Exception:
            await self._fail_turn(thread_id=thread_id, request_id=request_id)
            raise
        await self._finish_turn(
            thread_id=thread_id,
            request_id=request_id,
            response=response,
        )
        return response

    async def stream_create_session(
        self,
        *,
        user_id: str,
        project_id: str,
        request_id: str,
        message: MessageRequest,
        thread_id: str | None = None,
    ) -> AsyncIterator[str]:
        """Validate a new session, then stream its answer as SSE frames.

        Busy and conflict errors are raised before the first frame so the
        route can still answer them with their usual HTTP status codes.
        """
        client_preallocated = bool(str(thread_id or "").strip())
        normalized_thread_id = str(thread_id or uuid4())
        stack = AsyncExitStack()
        await stack.enter_async_context(self._session_slot(normalized_thread_id))
        try:
            if client_preallocated:
                config = self._config(
                    user_id=user_id,
                    project_id=project_id,
                    thread_id=normalized_thread_id,
                )
                snapshot = await self._get_state(config)
                if self._values(snapshot) or getattr(snapshot, "next", ()):
                    raise SessionConflictError
        except BaseException:
            await stack.aclose()
            raise
        return self._start_event_stream(
            stack,
            lambda observer: self._submit(
                user_id=user_id,
                project_id=project_id,
                thread_id=normalized_thread_id,
                request_id=request_id,
                message=message,
                create=True,
                answer_observer=observer,
            ),
        )

    async def stream_send_message(
        self,
        *,
        user_id: str,
        project_id: str,
        thread_id: str,
        request_id: str,
        message: MessageRequest,
    ) -> AsyncIterator[str]:
        """Validate one follow-up turn, then stream its answer as SSE frames."""
        stack = AsyncExitStack()
        await stack.enter_async_context(self._session_slot(thread_id))
        try:
            config = self._config(
                user_id=user_id,
                project_id=project_id,
                thread_id=thread_id,
            )
            snapshot = await self._get_state(config)
            self._assert_scope(snapshot, user_id=user_id, project_id=project_id)
        except BaseException:
            await stack.aclose()
            raise
        return self._start_event_stream(
            stack,
            lambda observer: self._submit(
                user_id=user_id,
                project_id=project_id,
                thread_id=thread_id,
                request_id=request_id,
                message=message,
                create=False,
                answer_observer=observer,
            ),
        )

    def _start_event_stream(
        self,
        stack: AsyncExitStack,
        submit: Callable[
            [Callable[[Mapping[str, Any]], None]],
            Awaitable[SessionResponse],
        ],
    ) -> AsyncIterator[str]:
        """Run one workflow in the background and relay its answer events.

        The workflow task owns the session slot and keeps running if the
        client disconnects, so the checkpoint still receives the final state
        that ``get_session`` later recovers.
        """
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[Any] = asyncio.Queue()

        def observer(event: Mapping[str, Any]) -> None:
            # Workflow nodes run in worker threads; hand events to the loop.
            loop.call_soon_threadsafe(queue.put_nowait, dict(event))

        async def run() -> SessionResponse:
            try:
                return await submit(observer)
            finally:
                await stack.aclose()

        def finished(task: asyncio.Task) -> None:
            self._stream_tasks.discard(task)
            if not task.cancelled() and task.exception() is not None:
                log.error(
                    "[API] streamed workflow failed | error={}",
                    type(task.exception()).__name__,
                )
            queue.put_nowait(_STREAM_DONE)

        task = asyncio.create_task(run())
        self._stream_tasks.add(task)
        task.add_done_callback(finished)

        async def frames() -> AsyncIterator[str]:
            while True:
                try:
                    event = await asyncio.wait_for(
                        queue.get(),
                        timeout=STREAM_KEEPALIVE_SECONDS,
                    )
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
                    continue
                if event is _STREAM_DONE:
                    break
                event_type = str(event.pop("type", "") or "")
                if event_type in {"reset", "delta"}:
                    yield _sse(event_type, event)
            if task.cancelled():
                yield _sse("error", self._stream_error(asyncio.CancelledError()))
            elif task.exception() is not None:
                yield _sse("error", self._stream_error(task.exception()))
            else:
                yield _sse("final", task.result().model_dump(mode="json"))

        return frames()

    @staticmethod
    def _stream_error(exc: BaseException) -> dict[str, str]:
        """Mirror the sanitized error bodies of the JSON exception handlers."""
        if isinstance(exc, SessionNotFoundError):
            return {"code": "session_not_found", "message": "session not found"}
        if isinstance(exc, (ConnectionError, TimeoutError)):
            return {
                "code": "service_unavailable",
                "message": "a required service is unavailable",
            }
        return {"code": "internal_error", "message": "internal server error"}

    async def cancel_session(
        self,
        *,
        user_id: str,
        project_id: str,
        thread_id: str,
    ) -> CancelSessionResponse:
        """Ask the active workflow to end at its next safe graph boundary."""
        cancellation = self.progress_registry.request_cancel(
            thread_id=thread_id,
            user_id=user_id,
            project_id=project_id,
        )
        if cancellation is not None:
            return CancelSessionResponse(thread_id=thread_id, **cancellation)

        config = self._config(
            user_id=user_id,
            project_id=project_id,
            thread_id=thread_id,
        )
        snapshot = await self._get_state(config)
        self._assert_scope(snapshot, user_id=user_id, project_id=project_id)
        return CancelSessionResponse(thread_id=thread_id, status="not_running")

    async def get_session(
        self,
        *,
        user_id: str,
        project_id: str,
        thread_id: str,
    ) -> SessionResponse:
        config = self._config(
            user_id=user_id,
            project_id=project_id,
            thread_id=thread_id,
        )
        snapshot = await self._get_state(config)
        return await self._response(
            snapshot,
            user_id=user_id,
            project_id=project_id,
            thread_id=thread_id,
            request_id=None,
        )

    async def get_report_path(
        self,
        *,
        user_id: str,
        project_id: str,
        thread_id: str,
    ) -> Path:
        config = self._config(
            user_id=user_id,
            project_id=project_id,
            thread_id=thread_id,
        )
        snapshot = await self._get_state(config)
        values = self._assert_scope(
            snapshot,
            user_id=user_id,
            project_id=project_id,
        )
        report_path = await asyncio.to_thread(self._safe_report_path, values)
        if report_path is None:
            raise SessionNotFoundError
        return report_path

    async def get_current_report_download(
        self,
        *,
        user_id: str,
        project_id: str,
        thread_id: str,
    ) -> ReportDownload:
        config = self._config(
            user_id=user_id,
            project_id=project_id,
            thread_id=thread_id,
        )
        snapshot = await self._get_state(config)
        values = self._assert_scope(
            snapshot,
            user_id=user_id,
            project_id=project_id,
        )
        report_path = await asyncio.to_thread(self._safe_report_path, values)
        if report_path is None:
            raise SessionNotFoundError
        return ReportDownload(
            filename=self._download_filename(
                self._current_report_title(values, report_path),
                "docx",
            ),
            path=report_path,
        )

    async def get_report_download(
        self,
        *,
        user_id: str,
        project_id: str,
        thread_id: str,
        report_id: str,
    ) -> ReportDownload:
        config = self._config(
            user_id=user_id,
            project_id=project_id,
            thread_id=thread_id,
        )
        snapshot = await self._get_state(config)
        values = self._assert_scope(snapshot, user_id=user_id, project_id=project_id)
        for raw_item in values.get("report_history", []) or []:
            if not isinstance(raw_item, Mapping):
                continue
            if str(raw_item.get("report_id", "") or "") != report_id:
                continue
            title = self._historical_report_title(raw_item)
            path = await asyncio.to_thread(
                self._safe_report_path_from_raw,
                raw_item.get("report_path"),
            )
            if path is not None:
                return ReportDownload(
                    filename=self._download_filename(title, "docx"),
                    path=path,
                )
            content = str(raw_item.get("report", "") or "").strip()
            if content:
                return ReportDownload(
                    filename=self._download_filename(title, "md"),
                    content=content,
                )
            raise SessionNotFoundError
        raise SessionNotFoundError

    async def list_reports(self, *, user_id: str) -> ReportListResponse:
        """List one user's reports across projects, newest first."""
        rows = await self._history(
            "list_user_sessions",
            user_id=user_id,
            limit=REPORT_LIBRARY_SESSION_LIMIT,
        )
        slots = asyncio.Semaphore(REPORT_LIBRARY_CONCURRENCY)

        async def session_reports(row: Mapping[str, Any]) -> list[ReportLibraryItem]:
            project_id = row["project_id"]
            thread_id = row["thread_id"]
            async with slots:
                snapshot = await self._get_state(
                    self._config(
                        user_id=user_id,
                        project_id=project_id,
                        thread_id=thread_id,
                    )
                )
            try:
                values = self._assert_scope(
                    snapshot,
                    user_id=user_id,
                    project_id=project_id,
                )
            except SessionNotFoundError:
                # History can outlive its checkpoint, and reports live in the checkpoint.
                return []
            summaries = await asyncio.to_thread(
                self._report_summaries,
                values,
                project_id=project_id,
                thread_id=thread_id,
            )
            return [
                ReportLibraryItem(
                    **summary.model_dump(),
                    project_id=project_id,
                    thread_id=thread_id,
                    session_title=row["title"],
                )
                for summary in summaries
            ]

        grouped = await asyncio.gather(*[session_reports(row) for row in rows])
        reports = [report for group in grouped for report in group]
        reports.sort(key=lambda report: report.created_at, reverse=True)
        return ReportListResponse(reports=reports)

    @staticmethod
    def _report_entry(values: Mapping[str, Any], report_id: str) -> Mapping[str, Any]:
        for raw_item in values.get("report_history", []) or []:
            if (
                isinstance(raw_item, Mapping)
                and str(raw_item.get("report_id", "") or "") == report_id
            ):
                return raw_item
        raise SessionNotFoundError

    async def get_report_content(
        self,
        *,
        user_id: str,
        project_id: str,
        thread_id: str,
        report_id: str,
    ) -> ReportContentResponse:
        snapshot = await self._get_state(
            self._config(user_id=user_id, project_id=project_id, thread_id=thread_id)
        )
        values = self._assert_scope(snapshot, user_id=user_id, project_id=project_id)
        report = self._report_entry(values, report_id)
        summaries = await asyncio.to_thread(
            self._report_summaries,
            {"report_history": [report]},
            project_id=project_id,
            thread_id=thread_id,
        )
        if not summaries:
            raise SessionNotFoundError
        return ReportContentResponse(
            report_id=report_id,
            title=summaries[0].title,
            created_at=summaries[0].created_at,
            format=summaries[0].format,
            content=str(report.get("report", "") or ""),
        )

    async def delete_report(
        self,
        *,
        user_id: str,
        project_id: str,
        thread_id: str,
        report_id: str,
    ) -> None:
        """Remove one archived report and its file; the session itself stays."""
        async with self._session_slot(thread_id):
            config = self._config(
                user_id=user_id,
                project_id=project_id,
                thread_id=thread_id,
            )
            snapshot = await self._get_state(config)
            values = self._assert_scope(
                snapshot,
                user_id=user_id,
                project_id=project_id,
            )
            report = self._report_entry(values, report_id)
            # Writing state into a paused graph could disturb its pending interrupt.
            if getattr(snapshot, "next", ()):
                raise SessionBusyError
            remaining = [
                item
                for item in values.get("report_history", []) or []
                if item is not report
            ]
            update: dict[str, Any] = {"report_history": remaining}
            is_current = (
                str(report.get("report", "") or "").strip()
                == str(values.get("report", "") or "").strip()
            )
            if is_current:
                update.update(
                    report="",
                    report_data={},
                    report_path=None,
                    child_graph_result={},
                )
            await self._update_state(config, update)
            # A kept report may still point at the same exported file.
            kept_paths = {
                str(item.get("report_path") or "")
                for item in remaining
                if isinstance(item, Mapping)
            }
            if not is_current:
                kept_paths.add(str(values.get("report_path") or ""))
            report_path = report.get("report_path")
            if str(report_path or "") in kept_paths:
                return
            path = await asyncio.to_thread(self._safe_report_path_from_raw, report_path)
            if path is not None:
                await asyncio.to_thread(path.unlink, missing_ok=True)

    async def close_session(
        self,
        *,
        user_id: str,
        project_id: str,
        thread_id: str,
    ) -> None:
        async with self._session_slot(thread_id):
            config = self._config(
                user_id=user_id,
                project_id=project_id,
                thread_id=thread_id,
            )
            snapshot = await self._get_state(config)
            try:
                values = self._assert_scope(
                    snapshot,
                    user_id=user_id,
                    project_id=project_id,
                )
            except SessionNotFoundError:
                await self._delete_orphaned_history(
                    user_id=user_id,
                    project_id=project_id,
                    thread_id=thread_id,
                    snapshot=snapshot,
                )
                return
            report_paths = {
                path
                for path in [
                    await asyncio.to_thread(self._safe_report_path, values),
                    *[
                        await asyncio.to_thread(
                            self._safe_report_path_from_raw,
                            item.get("report_path"),
                        )
                        for item in values.get("report_history", []) or []
                        if isinstance(item, Mapping)
                    ],
                ]
                if path is not None
            }
            for report_path in report_paths:
                await asyncio.to_thread(report_path.unlink)
            close_session = getattr(self.runner, "aclose_session", None)
            if inspect.iscoroutinefunction(close_session):
                await close_session(thread_id, user_id=user_id)
            else:
                await asyncio.to_thread(
                    self.runner.close_session,
                    thread_id,
                    user_id=user_id,
                )
            self.progress_registry.remove(thread_id)
            if self.session_repository is not None:
                await self._history("delete_session", thread_id=thread_id)

    async def _delete_orphaned_history(
        self,
        *,
        user_id: str,
        project_id: str,
        thread_id: str,
        snapshot: Any,
    ) -> None:
        """Delete owned history whose checkpoint is gone; anything else stays a 404."""
        if (
            self.session_repository is None
            or self._values(snapshot)
            or await self._history(
                "get_session",
                user_id=user_id,
                project_id=project_id,
                thread_id=thread_id,
            )
            is None
        ):
            raise SessionNotFoundError
        await self._history("delete_session", thread_id=thread_id)

"""Application service adapting HTTP sessions to the LangGraph runner."""

import asyncio
from collections.abc import Mapping
from contextlib import asynccontextmanager
import inspect
import re
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any
from uuid import uuid4

from langgraph.errors import GraphInterrupt
from langgraph.types import Command

from src.config.settings import REPORT_OUTPUT_DIR
from src.api.progress import WorkflowProgressRegistry
from src.api.schemas import (
    CancelSessionResponse,
    MessageRequest,
    ReportSummary,
    SessionResponse,
)


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


class SessionService:
    def __init__(
        self,
        runner: Any,
        *,
        report_output_dir: Path = REPORT_OUTPUT_DIR,
        progress_registry: WorkflowProgressRegistry | None = None,
    ):
        self.runner = runner
        self.report_output_dir = Path(report_output_dir).resolve()
        self.progress_registry = progress_registry or WorkflowProgressRegistry()
        self._locks: dict[str, asyncio.Lock] = {}
        self._locks_guard = Lock()

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
        result: Any = None
        with self.progress_registry.track(
            thread_id=thread_id,
            request_id=request_id,
            user_id=user_id,
            project_id=project_id,
        ):
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
        return await self._response(
            snapshot,
            user_id=user_id,
            project_id=project_id,
            thread_id=thread_id,
            request_id=request_id,
            invocation_result=result,
        )

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
            values = self._assert_scope(
                snapshot,
                user_id=user_id,
                project_id=project_id,
            )
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

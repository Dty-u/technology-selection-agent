"""Application service adapting HTTP sessions to the LangGraph runner."""

from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path
from threading import Lock
from typing import Any
from uuid import uuid4

from langgraph.errors import GraphInterrupt
from langgraph.types import Command

from src.config.settings import REPORT_OUTPUT_DIR
from src.api.schemas import MessageRequest, SessionResponse


class SessionNotFoundError(Exception):
    """Hide missing sessions and scope mismatches behind one public error."""


class SessionBusyError(Exception):
    """Reject overlapping writes to one checkpoint thread."""


class SessionService:
    def __init__(self, runner: Any, *, report_output_dir: Path = REPORT_OUTPUT_DIR):
        self.runner = runner
        self.report_output_dir = Path(report_output_dir).resolve()
        self._locks: dict[str, Lock] = {}
        self._locks_guard = Lock()

    @contextmanager
    def _session_slot(self, thread_id: str):
        with self._locks_guard:
            lock = self._locks.setdefault(thread_id, Lock())
        if not lock.acquire(blocking=False):
            raise SessionBusyError
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

    def _response(
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
        status = "awaiting_user" if awaiting_user else values.get("status", "error")
        report_path = self._safe_report_path(values)
        report_url = None
        if report_path is not None:
            report_url = (
                f"/api/v1/projects/{project_id}/sessions/{thread_id}/report"
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
        )

    def _safe_report_path(self, values: Mapping[str, Any]) -> Path | None:
        raw_path = str(values.get("report_path", "") or "").strip()
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

    def create_session(
        self,
        *,
        user_id: str,
        project_id: str,
        request_id: str,
        message: MessageRequest,
    ) -> SessionResponse:
        thread_id = str(uuid4())
        return self._submit(
            user_id=user_id,
            project_id=project_id,
            thread_id=thread_id,
            request_id=request_id,
            message=message,
            create=True,
        )

    def send_message(
        self,
        *,
        user_id: str,
        project_id: str,
        thread_id: str,
        request_id: str,
        message: MessageRequest,
    ) -> SessionResponse:
        with self._session_slot(thread_id):
            return self._submit(
                user_id=user_id,
                project_id=project_id,
                thread_id=thread_id,
                request_id=request_id,
                message=message,
                create=False,
            )

    def _submit(
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
            }
        else:
            before = self.runner.graph.get_state(config)
            self._assert_scope(before, user_id=user_id, project_id=project_id)
            if getattr(before, "next", ()):
                payload = Command(resume=message.message)
            else:
                payload = {
                    "query": message.message,
                    "report_format": message.report_format,
                }
        result: Any = None
        try:
            result = self.runner.invoke(payload, config=config)
        except PermissionError as exc:
            raise SessionNotFoundError from exc
        except GraphInterrupt:
            pass
        snapshot = self.runner.graph.get_state(config)
        return self._response(
            snapshot,
            user_id=user_id,
            project_id=project_id,
            thread_id=thread_id,
            request_id=request_id,
            invocation_result=result,
        )

    def get_session(
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
        snapshot = self.runner.graph.get_state(config)
        return self._response(
            snapshot,
            user_id=user_id,
            project_id=project_id,
            thread_id=thread_id,
            request_id=None,
        )

    def get_report_path(
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
        snapshot = self.runner.graph.get_state(config)
        values = self._assert_scope(
            snapshot,
            user_id=user_id,
            project_id=project_id,
        )
        report_path = self._safe_report_path(values)
        if report_path is None:
            raise SessionNotFoundError
        return report_path

    def close_session(
        self,
        *,
        user_id: str,
        project_id: str,
        thread_id: str,
    ) -> None:
        with self._session_slot(thread_id):
            config = self._config(
                user_id=user_id,
                project_id=project_id,
                thread_id=thread_id,
            )
            snapshot = self.runner.graph.get_state(config)
            values = self._assert_scope(
                snapshot,
                user_id=user_id,
                project_id=project_id,
            )
            report_path = self._safe_report_path(values)
            if report_path is not None:
                report_path.unlink()
            self.runner.close_session(thread_id, user_id=user_id)

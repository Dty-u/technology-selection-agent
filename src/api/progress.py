"""Request-scoped, process-local workflow progress snapshots for the REST API."""

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
from threading import Lock
from typing import Any

from src.observability.workflow_progress import observe_progress


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class WorkflowProgressRegistry:
    """Keep a bounded public trace without exposing console log contents."""

    def __init__(self, *, max_sessions: int = 100, max_events: int = 100) -> None:
        self.max_sessions = max_sessions
        self.max_events = max_events
        self._lock = Lock()
        self._requests: dict[str, dict[str, Any]] = {}

    @contextmanager
    def track(
        self,
        *,
        thread_id: str,
        request_id: str,
        user_id: str,
        project_id: str,
    ) -> Iterator[None]:
        with self._lock:
            self._requests[thread_id] = {
                "request_id": request_id,
                "user_id": user_id,
                "project_id": project_id,
                "active": True,
                "cancel_requested": False,
                "cancelled": False,
                "started_at": _now_iso(),
                "events": [],
            }
            self._trim_sessions()

        def observer(event: Mapping[str, Any]) -> None:
            self._record(thread_id, request_id, event)

        try:
            with observe_progress(observer):
                yield
        finally:
            with self._lock:
                request = self._requests.get(thread_id)
                if request is not None and request["request_id"] == request_id:
                    request["active"] = False
                    request["cancelled"] = bool(request["cancel_requested"])
                    request["finished_at"] = _now_iso()

    def request_cancel(
        self,
        *,
        thread_id: str,
        user_id: str,
        project_id: str,
    ) -> dict[str, Any] | None:
        """Request cancellation for the active, scoped workflow without deleting it."""
        with self._lock:
            request = self._requests.get(thread_id)
            if request is None:
                return None
            if (
                request.get("user_id") != user_id
                or request.get("project_id") != project_id
            ):
                return None
            if request["active"]:
                request["cancel_requested"] = True
                return {
                    "request_id": request["request_id"],
                    "status": "cancelling",
                }
            return {
                "request_id": request["request_id"],
                "status": "cancelled" if request["cancelled"] else "not_running",
            }

    def is_cancel_requested(self, *, thread_id: str, request_id: str) -> bool:
        with self._lock:
            request = self._requests.get(thread_id)
            return bool(
                request is not None
                and request["request_id"] == request_id
                and request["cancel_requested"]
            )

    def _record(
        self,
        thread_id: str,
        request_id: str,
        event: Mapping[str, Any],
    ) -> None:
        status = str(event.get("status", "") or "")
        scope = str(event.get("scope", "") or "")
        stage_id = str(event.get("stage_id", "") or "")
        label = str(event.get("label", "") or "")
        if status not in {"running", "completed", "paused", "error"}:
            return
        if not scope or not stage_id or not label:
            return

        with self._lock:
            request = self._requests.get(thread_id)
            if request is None or request["request_id"] != request_id:
                return
            events = request["events"]
            if status == "running":
                events.append(
                    {
                        "scope": scope,
                        "stage_id": stage_id,
                        "label": label,
                        "status": status,
                        "started_at": _now_iso(),
                        "finished_at": None,
                        "duration_ms": None,
                    }
                )
                if len(events) > self.max_events:
                    del events[: len(events) - self.max_events]
                return

            for item in reversed(events):
                if (
                    item["scope"] == scope
                    and item["stage_id"] == stage_id
                    and item["status"] == "running"
                ):
                    item["status"] = status
                    item["finished_at"] = _now_iso()
                    duration_ms = event.get("duration_ms")
                    item["duration_ms"] = (
                        int(duration_ms) if isinstance(duration_ms, (int, float)) else None
                    )
                    return

    def snapshot(self, thread_id: str) -> dict[str, Any]:
        with self._lock:
            request = deepcopy(self._requests.get(thread_id))
        if request is None:
            return {
                "request_id": None,
                "active": False,
                "cancel_requested": False,
                "cancelled": False,
                "current_stage": None,
                "events": [],
            }
        current_stage = next(
            (
                {
                    "scope": item["scope"],
                    "stage_id": item["stage_id"],
                    "label": item["label"],
                }
                for item in reversed(request["events"])
                if item["status"] == "running"
            ),
            None,
        )
        return {
            "request_id": request["request_id"],
            "active": request["active"],
            "cancel_requested": request["cancel_requested"],
            "cancelled": request["cancelled"],
            "current_stage": current_stage,
            "events": request["events"],
        }

    def remove(self, thread_id: str) -> None:
        with self._lock:
            self._requests.pop(thread_id, None)

    def _trim_sessions(self) -> None:
        overflow = len(self._requests) - self.max_sessions
        if overflow <= 0:
            return
        inactive = [
            thread_id
            for thread_id, request in self._requests.items()
            if not request["active"]
        ]
        for thread_id in inactive[:overflow]:
            self._requests.pop(thread_id, None)

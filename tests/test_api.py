import asyncio
import unittest
from urllib.parse import quote
import os
import warnings
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
import shutil
from threading import Event, Lock
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from uuid import uuid4

from starlette.exceptions import StarletteDeprecationWarning

warnings.filterwarnings(
    "ignore",
    message="Using `httpx` with `starlette.testclient` is deprecated.*",
    category=StarletteDeprecationWarning,
)

from fastapi.testclient import TestClient


def _stream_chunks(*chunks):
    async def stream(*args, **kwargs):
        for chunk in chunks:
            yield chunk

    return stream


def _stream_error(error):
    async def stream(*args, **kwargs):
        raise error
        yield None

    return stream


class ApiLifecycleTest(unittest.TestCase):
    def test_lifespan_opens_one_runner_and_health_is_available(self):
        from src.api.main import create_app

        events = []
        runner = MagicMock()

        @contextmanager
        def application_factory():
            events.append("open")
            yield runner
            events.append("close")

        app = create_app(application_factory=application_factory)

        with TestClient(app) as client:
            self.assertEqual(events, ["open"])
            response = client.get("/health/live")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json(), {"status": "ok"})

        self.assertEqual(events, ["open", "close"])


class ApiSessionTest(unittest.TestCase):
    def _client(self, runner):
        from src.api.main import create_app

        @contextmanager
        def application_factory():
            yield runner

        return TestClient(create_app(application_factory=application_factory))

    @staticmethod
    def _snapshot(*, values=None, next_nodes=(), tasks=()):
        return SimpleNamespace(
            values=dict(values or {}),
            next=tuple(next_nodes),
            tasks=tuple(tasks),
        )

    def test_create_session_invokes_runner_with_all_scope_ids(self):
        completed = {
            "session_user_id": "user-a",
            "session_project_id": "project-a",
            "status": "completed",
            "query_type": "rag_qa",
            "response_type": "answer",
            "answer": "FinFET uses a fin-shaped channel.",
            "report": "",
            "report_data": {},
            "clarification_questions": [],
            "clarification_answers": [],
            "clarification_history": [],
            "sources": [{"title": "source-a"}],
            "warnings": [],
            "report_path": "D:/private/report.docx",
            "child_graph_result": {"secret": "internal"},
        }
        runner = MagicMock()
        runner.astream.side_effect = _stream_chunks(completed)
        runner.graph.get_state.side_effect = [
            self._snapshot(),
            self._snapshot(values=completed),
        ]
        thread_id = "22222222-2222-4222-8222-222222222222"

        with self._client(runner) as client:
            response = client.post(
                "/api/v1/projects/project-a/sessions",
                headers={
                    "X-User-ID": "user-a",
                    "X-Request-ID": "11111111-1111-4111-8111-111111111111",
                    "X-Thread-ID": thread_id,
                },
                json={"message": "What is FinFET?", "report_format": "markdown"},
            )

        self.assertEqual(response.status_code, 201)
        body = response.json()
        self.assertEqual(body["thread_id"], thread_id)
        self.assertEqual(body["request_id"], "11111111-1111-4111-8111-111111111111")
        self.assertEqual(body["project_id"], "project-a")
        self.assertEqual(body["status"], "completed")
        self.assertEqual(body["answer"], "FinFET uses a fin-shaped channel.")
        self.assertNotIn("report_path", body)
        self.assertNotIn("child_graph_result", body)

        payload = runner.astream.call_args.args[0]
        config = runner.astream.call_args.kwargs["config"]["configurable"]
        self.assertEqual(
            payload,
            {
                "query": "What is FinFET?",
                "report_format": "markdown",
                "session_user_id": "user-a",
                "session_project_id": "project-a",
            },
        )
        self.assertEqual(config["user_id"], "user-a")
        self.assertEqual(config["project_id"], "project-a")
        self.assertEqual(config["request_id"], body["request_id"])
        self.assertEqual(config["thread_id"], body["thread_id"])

    def test_create_session_rejects_an_existing_preallocated_thread(self):
        existing = {
            "session_user_id": "user-a",
            "session_project_id": "project-a",
            "status": "completed",
        }
        runner = MagicMock()
        runner.graph.get_state.return_value = self._snapshot(values=existing)
        runner.astream.side_effect = _stream_chunks(existing)
        thread_id = "23232323-2323-4232-8232-232323232323"

        with self._client(runner) as client:
            response = client.post(
                "/api/v1/projects/project-a/sessions",
                headers={
                    "X-User-ID": "user-a",
                    "X-Request-ID": "24242424-2424-4242-8242-242424242424",
                    "X-Thread-ID": thread_id,
                },
                json={"message": "What is FinFET?"},
            )

        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json()["detail"]["code"], "session_conflict")
        runner.astream.assert_not_called()

    def test_message_resumes_interrupt_and_returns_clarification_questions(self):
        from langgraph.types import Command

        interrupt = SimpleNamespace(
            value={
                "status": "awaiting_user",
                "clarification_questions": ["Which application?", "Power priority?"],
            }
        )
        values = {
            "session_user_id": "user-a",
            "session_project_id": "project-a",
            "status": "running",
            "query_type": "technology_selection",
            "response_type": "report",
            "clarification_answers": [],
            "clarification_history": [],
        }
        waiting = self._snapshot(
            values=values,
            next_nodes=("clarify_requirements",),
            tasks=(SimpleNamespace(interrupts=(interrupt,)),),
        )
        runner = MagicMock()
        runner.graph.get_state.side_effect = [waiting, waiting]
        runner.astream.side_effect = _stream_chunks({"__interrupt__": (interrupt,)})
        thread_id = "22222222-2222-4222-8222-222222222222"

        with self._client(runner) as client:
            response = client.post(
                f"/api/v1/projects/project-a/sessions/{thread_id}/messages",
                headers={"X-User-ID": "user-a"},
                json={"message": "Mobile SoC, power first"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "awaiting_user")
        self.assertEqual(
            response.json()["clarification_questions"],
            ["Which application?", "Power priority?"],
        )
        command = runner.astream.call_args.args[0]
        self.assertIsInstance(command, Command)
        self.assertEqual(command.resume, "Mobile SoC, power first")

    def test_graph_interrupt_is_returned_as_awaiting_user_not_server_error(self):
        from langgraph.errors import GraphInterrupt

        interrupt = SimpleNamespace(
            value={
                "status": "awaiting_user",
                "clarification_questions": ["Please add the target process node."],
            }
        )
        waiting = self._snapshot(
            values={
                "session_user_id": "user-a",
                "session_project_id": "project-a",
                "status": "running",
                "query_type": "technology_selection",
                "response_type": "report",
            },
            next_nodes=("clarify_requirements",),
            tasks=(SimpleNamespace(interrupts=(interrupt,)),),
        )
        runner = MagicMock()
        runner.graph.get_state.side_effect = [waiting, waiting]
        runner.astream.side_effect = _stream_error(GraphInterrupt("paused"))
        thread_id = "12121212-1212-4212-8212-121212121212"

        with self._client(runner) as client:
            response = client.post(
                f"/api/v1/projects/project-a/sessions/{thread_id}/messages",
                headers={"X-User-ID": "user-a"},
                json={"message": "Additional requirement"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "awaiting_user")
        self.assertEqual(
            response.json()["clarification_questions"],
            ["Please add the target process node."],
        )

    def test_get_session_recovers_interrupt_and_hides_scope_mismatches(self):
        interrupt = SimpleNamespace(
            value={
                "status": "awaiting_user",
                "clarification_questions": ["Which application?"],
            }
        )
        waiting = self._snapshot(
            values={
                "session_user_id": "user-a",
                "session_project_id": "project-a",
                "query_type": "technology_selection",
                "response_type": "report",
            },
            next_nodes=("clarify_requirements",),
            tasks=(SimpleNamespace(interrupts=(interrupt,)),),
        )
        runner = MagicMock()
        runner.graph.get_state.return_value = waiting
        thread_id = "33333333-3333-4333-8333-333333333333"

        with self._client(runner) as client:
            visible = client.get(
                f"/api/v1/projects/project-a/sessions/{thread_id}",
                headers={"X-User-ID": "user-a"},
            )
            hidden = client.get(
                f"/api/v1/projects/project-b/sessions/{thread_id}",
                headers={"X-User-ID": "user-a"},
            )

        self.assertEqual(visible.status_code, 200)
        self.assertEqual(visible.json()["status"], "awaiting_user")
        self.assertEqual(visible.json()["clarification_questions"], ["Which application?"])
        self.assertEqual(hidden.status_code, 404)
        self.assertEqual(hidden.json()["detail"]["code"], "session_not_found")

    def test_request_contract_rejects_missing_identity_and_invalid_message(self):
        runner = MagicMock()

        with self._client(runner) as client:
            missing_identity = client.post(
                "/api/v1/projects/project-a/sessions",
                json={"message": "hello"},
            )
            blank_message = client.post(
                "/api/v1/projects/project-a/sessions",
                headers={"X-User-ID": "user-a"},
                json={"message": "   "},
            )
            invalid_format = client.post(
                "/api/v1/projects/project-a/sessions",
                headers={"X-User-ID": "user-a"},
                json={"message": "hello", "report_format": "pdf"},
            )

        self.assertEqual(missing_identity.status_code, 422)
        self.assertEqual(blank_message.status_code, 422)
        self.assertEqual(invalid_format.status_code, 422)
        runner.astream.assert_not_called()


class ApiReportAndConcurrencyTest(unittest.TestCase):
    def setUp(self):
        self.test_directory = Path.cwd() / "runtime" / f"api-test-{uuid4().hex}"
        self.report_directory = self.test_directory / "reports"
        self.report_directory.mkdir(parents=True)

    def tearDown(self):
        if self.test_directory.exists():
            shutil.rmtree(self.test_directory)

    @staticmethod
    def _snapshot(report_path="", report_history=None):
        return SimpleNamespace(
            values={
                "session_user_id": "user-a",
                "session_project_id": "project-a",
                "status": "completed",
                "query_type": "technology_selection",
                "response_type": "report",
                "report": "# Selection report",
                "report_data": {"requirements": {"business_goal": "低功耗移动芯片"}},
                "report_path": str(report_path),
                "report_history": report_history or [],
            },
            next=(),
            tasks=(),
        )

    def _client(self, runner, *, raise_server_exceptions=True):
        from src.api.main import create_app
        from src.api.service import SessionService

        @contextmanager
        def application_factory():
            yield runner

        app = create_app(application_factory=application_factory)
        client = TestClient(
            app,
            raise_server_exceptions=raise_server_exceptions,
        )
        client.__enter__()
        client.app.state.session_service = SessionService(
            runner,
            report_output_dir=self.report_directory,
        )
        self.addCleanup(client.__exit__, None, None, None)
        return client

    def test_report_download_and_session_response_never_expose_server_path(self):
        report_path = self.report_directory / "selection.docx"
        report_path.write_bytes(b"docx-content")
        runner = MagicMock()
        runner.graph.get_state.return_value = self._snapshot(report_path)
        thread_id = "44444444-4444-4444-8444-444444444444"
        client = self._client(runner)

        state = client.get(
            f"/api/v1/projects/project-a/sessions/{thread_id}",
            headers={"X-User-ID": "user-a"},
        )
        download = client.get(
            f"/api/v1/projects/project-a/sessions/{thread_id}/report",
            headers={"X-User-ID": "user-a", "Origin": "http://localhost:5173"},
        )

        self.assertEqual(state.status_code, 200)
        self.assertEqual(
            state.json()["report_url"],
            f"/api/v1/projects/project-a/sessions/{thread_id}/report",
        )
        self.assertNotIn(str(report_path), state.text)
        self.assertEqual(download.status_code, 200)
        self.assertEqual(download.content, b"docx-content")
        self.assertEqual(
            download.headers["content-type"],
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )
        self.assertEqual(
            download.headers["access-control-expose-headers"],
            "Content-Disposition",
        )
        self.assertIn(
            quote("低功耗移动芯片技术选型报告.docx"),
            download.headers["content-disposition"],
        )

    def test_session_lists_all_selection_reports_and_downloads_each_artifact(self):
        docx_path = self.report_directory / "second-selection.docx"
        docx_path.write_bytes(b"second-docx")
        report_history = [
            {
                "report_id": "report-markdown",
                "title": "第一份芯片选型",
                "created_at": "2026-08-23T08:00:00+00:00",
                "format": "markdown",
                "report": "# 第一份芯片选型\n\n推荐方案 A。",
                "report_path": None,
            },
            {
                "report_id": "report-docx",
                "title": "用户原始提问",
                "created_at": "2026-08-23T09:00:00+00:00",
                "format": "docx",
                "report": "| 选型主题 | 车规级功率器件 |",
                "report_path": str(docx_path),
            },
        ]
        runner = MagicMock()
        runner.graph.get_state.return_value = self._snapshot(
            docx_path,
            report_history,
        )
        thread_id = "45454545-4545-4454-8454-454545454545"
        client = self._client(runner)

        state = client.get(
            f"/api/v1/projects/project-a/sessions/{thread_id}",
            headers={"X-User-ID": "user-a"},
        )
        markdown = client.get(
            f"/api/v1/projects/project-a/sessions/{thread_id}/reports/report-markdown",
            headers={"X-User-ID": "user-a"},
        )
        docx = client.get(
            f"/api/v1/projects/project-a/sessions/{thread_id}/reports/report-docx",
            headers={"X-User-ID": "user-a"},
        )

        self.assertEqual(state.status_code, 200)
        reports = state.json()["reports"]
        self.assertEqual([item["title"] for item in reports], [
            "第一份芯片选型",
            "用户原始提问",
        ])
        self.assertNotIn(str(docx_path), state.text)
        self.assertEqual(markdown.status_code, 200)
        self.assertIn("推荐方案 A", markdown.text)
        self.assertIn(".md", markdown.headers["content-disposition"])
        self.assertEqual(docx.content, b"second-docx")
        self.assertIn(
            quote("车规级功率器件技术选型报告.docx"),
            docx.headers["content-disposition"],
        )

    def test_close_deletes_only_safe_report_and_checkpoint(self):
        report_path = self.report_directory / "selection.docx"
        report_path.write_bytes(b"docx-content")
        runner = MagicMock()
        runner.graph.get_state.return_value = self._snapshot(report_path)
        thread_id = "55555555-5555-4555-8555-555555555555"
        client = self._client(runner)

        response = client.delete(
            f"/api/v1/projects/project-a/sessions/{thread_id}",
            headers={"X-User-ID": "user-a"},
        )

        self.assertEqual(response.status_code, 204)
        self.assertFalse(report_path.exists())
        runner.close_session.assert_called_once_with(thread_id, user_id="user-a")

    def test_cross_project_close_has_no_side_effects(self):
        report_path = self.report_directory / "selection.docx"
        report_path.write_bytes(b"docx-content")
        runner = MagicMock()
        runner.graph.get_state.return_value = self._snapshot(report_path)
        thread_id = "66666666-6666-4666-8666-666666666666"
        client = self._client(runner)

        response = client.delete(
            f"/api/v1/projects/project-b/sessions/{thread_id}",
            headers={"X-User-ID": "user-a"},
        )

        self.assertEqual(response.status_code, 404)
        self.assertTrue(report_path.exists())
        runner.close_session.assert_not_called()

    def test_report_path_outside_output_directory_is_never_read_or_deleted(self):
        outside_path = self.test_directory / "outside.docx"
        outside_path.write_bytes(b"private")
        runner = MagicMock()
        runner.graph.get_state.return_value = self._snapshot(outside_path)
        thread_id = "77777777-7777-4777-8777-777777777777"
        client = self._client(runner)

        download = client.get(
            f"/api/v1/projects/project-a/sessions/{thread_id}/report",
            headers={"X-User-ID": "user-a"},
        )
        closed = client.delete(
            f"/api/v1/projects/project-a/sessions/{thread_id}",
            headers={"X-User-ID": "user-a"},
        )

        self.assertEqual(download.status_code, 404)
        self.assertEqual(closed.status_code, 204)
        self.assertTrue(outside_path.exists())

    def test_locked_report_delete_is_internal_error_not_session_not_found(self):
        report_path = self.report_directory / "locked.docx"
        report_path.write_bytes(b"docx-content")
        runner = MagicMock()
        runner.graph.get_state.return_value = self._snapshot(report_path)
        thread_id = "78787878-7878-4878-8878-787878787878"
        client = self._client(runner, raise_server_exceptions=False)

        with patch.object(
            Path,
            "unlink",
            side_effect=PermissionError("locked-private-path"),
        ):
            response = client.delete(
                f"/api/v1/projects/project-a/sessions/{thread_id}",
                headers={"X-User-ID": "user-a"},
            )

        self.assertEqual(response.status_code, 500)
        self.assertNotIn("locked-private-path", response.text)
        self.assertTrue(report_path.exists())
        runner.close_session.assert_not_called()

    def test_second_submission_to_same_thread_is_rejected_while_first_runs(self):
        from src.api.schemas import MessageRequest
        from src.api.service import SessionBusyError, SessionService

        snapshot = self._snapshot()
        started = Event()
        release = Event()
        call_lock = Lock()
        call_count = 0

        async def stream(payload, *, config):
            nonlocal call_count
            with call_lock:
                call_count += 1
                current_call = call_count
            if current_call == 1:
                started.set()
                await asyncio.to_thread(release.wait, 5)
            yield snapshot.values

        runner = MagicMock()
        runner.graph.get_state.return_value = snapshot
        runner.astream.side_effect = stream
        service = SessionService(runner, report_output_dir=self.report_directory)
        kwargs = {
            "user_id": "user-a",
            "project_id": "project-a",
            "thread_id": "88888888-8888-4888-8888-888888888888",
            "message": MessageRequest(message="next question"),
        }

        with ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(
                lambda: asyncio.run(
                    service.send_message(
                        request_id="99999999-9999-4999-8999-999999999991",
                        **kwargs,
                    )
                ),
            )
            self.assertTrue(started.wait(timeout=5))
            try:
                with self.assertRaises(SessionBusyError):
                    asyncio.run(
                        service.send_message(
                            request_id="99999999-9999-4999-8999-999999999992",
                            **kwargs,
                        )
                    )
            finally:
                release.set()
            first.result(timeout=5)

        self.assertEqual(call_count, 1)

    def test_cancel_stops_at_next_stream_boundary_and_preserves_session(self):
        from src.api.schemas import MessageRequest
        from src.api.service import SessionService

        values = {
            "session_user_id": "user-a",
            "session_project_id": "project-a",
            "status": "completed",
            "query_type": "rag_qa",
            "response_type": "answer",
            "answer": "answer",
        }
        snapshot = SimpleNamespace(values=values, next=(), tasks=())
        started = Event()
        release = Event()
        call_count = 0

        async def stream(_payload, *, config):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                yield {"status": "running"}
                started.set()
                await asyncio.to_thread(release.wait, 5)
            yield values

        runner = MagicMock()
        runner.graph.get_state.return_value = snapshot
        runner.astream.side_effect = stream
        service = SessionService(runner, report_output_dir=self.report_directory)
        thread_id = "87878787-8787-4787-8787-878787878787"
        first_request_id = "98989898-9898-4989-8989-989898989891"

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                lambda: asyncio.run(
                    service.send_message(
                        user_id="user-a",
                        project_id="project-a",
                        thread_id=thread_id,
                        request_id=first_request_id,
                        message=MessageRequest(message="停止前的请求"),
                    )
                ),
            )
            self.assertTrue(started.wait(timeout=5))
            cancellation = asyncio.run(
                service.cancel_session(
                    user_id="user-a",
                    project_id="project-a",
                    thread_id=thread_id,
                )
            )
            self.assertEqual(cancellation.status, "cancelling")
            self.assertEqual(cancellation.request_id, first_request_id)
            release.set()
            cancelled = future.result(timeout=5)

        self.assertEqual(cancelled.status, "cancelled")
        self.assertFalse(cancelled.progress.active)
        self.assertTrue(cancelled.progress.cancelled)
        self.assertEqual(call_count, 1)

        next_response = asyncio.run(
            service.send_message(
                user_id="user-a",
                project_id="project-a",
                thread_id=thread_id,
                request_id="98989898-9898-4989-8989-989898989892",
                message=MessageRequest(message="取消后继续提问"),
            )
        )
        self.assertEqual(next_response.status, "completed")
        self.assertEqual(call_count, 2)

    def test_cancel_is_scoped_and_idempotent_for_a_completed_request(self):
        from src.api.service import SessionNotFoundError, SessionService

        values = {
            "session_user_id": "user-a",
            "session_project_id": "project-a",
            "status": "completed",
        }
        runner = MagicMock()
        runner.graph.get_state.return_value = SimpleNamespace(
            values=values,
            next=(),
            tasks=(),
        )
        service = SessionService(runner, report_output_dir=self.report_directory)
        thread_id = "86868686-8686-4686-8686-868686868686"

        first = asyncio.run(
            service.cancel_session(
                user_id="user-a",
                project_id="project-a",
                thread_id=thread_id,
            )
        )
        second = asyncio.run(
            service.cancel_session(
                user_id="user-a",
                project_id="project-a",
                thread_id=thread_id,
            )
        )

        self.assertEqual(first.status, "not_running")
        self.assertEqual(second.status, "not_running")
        with self.assertRaises(SessionNotFoundError):
            asyncio.run(
                service.cancel_session(
                    user_id="user-a",
                    project_id="project-b",
                    thread_id=thread_id,
                )
            )

    def test_get_session_exposes_actual_logged_stage_while_request_runs(self):
        from src.api.schemas import MessageRequest
        from src.api.service import SessionService
        from src.observability.workflow_progress import logged_node

        values = {
            "session_user_id": "user-a",
            "session_project_id": "project-a",
            "status": "completed",
            "query_type": "rag_qa",
            "response_type": "answer",
            "answer": "answer",
        }
        snapshot = SimpleNamespace(values=values, next=(), tasks=())
        started = Event()
        release = Event()

        def retrieve(_state):
            started.set()
            release.wait(timeout=5)
            return {"documents": []}

        tracked_retrieve = logged_node(
            "普通RAG",
            "vector_retrieve",
            "向量检索",
            retrieve,
            heartbeat_seconds=None,
        )

        async def stream(_payload, *, config):
            tracked_retrieve({}, config)
            yield values

        runner = MagicMock()
        runner.graph.get_state.return_value = snapshot
        runner.astream.side_effect = stream
        service = SessionService(runner, report_output_dir=self.report_directory)
        thread_id = "89898989-8989-4989-8989-898989898981"

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                lambda: asyncio.run(
                    service.send_message(
                        user_id="user-a",
                        project_id="project-a",
                        thread_id=thread_id,
                        request_id="99999999-9999-4999-8999-999999999981",
                        message=MessageRequest(message="继续检索"),
                    )
                ),
            )
            self.assertTrue(started.wait(timeout=5))
            live = asyncio.run(
                service.get_session(
                    user_id="user-a",
                    project_id="project-a",
                    thread_id=thread_id,
                )
            )
            release.set()
            completed = future.result(timeout=5)

        self.assertEqual(live.status, "running")
        self.assertTrue(live.progress.active)
        self.assertEqual(live.progress.current_stage.label, "向量检索")
        self.assertEqual(live.progress.events[0].stage_id, "vector_retrieve")
        self.assertEqual(live.progress.events[0].status, "running")
        self.assertFalse(completed.progress.active)
        self.assertIsNone(completed.progress.current_stage)
        self.assertEqual(completed.progress.events[0].status, "completed")
        self.assertIsNotNone(completed.progress.events[0].duration_ms)


class ApiErrorMappingTest(unittest.TestCase):
    def _client_with_service(self, service):
        from src.api.main import create_app

        @contextmanager
        def application_factory():
            yield MagicMock()

        app = create_app(application_factory=application_factory)
        client = TestClient(app, raise_server_exceptions=False)
        client.__enter__()
        client.app.state.session_service = service
        self.addCleanup(client.__exit__, None, None, None)
        return client

    def test_busy_dependency_and_unknown_errors_have_sanitized_statuses(self):
        from src.api.service import SessionBusyError

        thread_id = "99999999-9999-4999-8999-999999999999"
        path = f"/api/v1/projects/project-a/sessions/{thread_id}/messages"
        headers = {"X-User-ID": "user-a"}

        busy_service = MagicMock()
        busy_service.send_message.side_effect = SessionBusyError
        busy = self._client_with_service(busy_service).post(
            path,
            headers=headers,
            json={"message": "hello"},
        )

        unavailable_service = MagicMock()
        unavailable_service.send_message.side_effect = ConnectionError(
            "mysql://secret@host"
        )
        unavailable = self._client_with_service(unavailable_service).post(
            path,
            headers=headers,
            json={"message": "hello"},
        )

        broken_service = MagicMock()
        broken_service.send_message.side_effect = RuntimeError("private-stack-detail")
        broken = self._client_with_service(broken_service).post(
            path,
            headers=headers,
            json={"message": "hello"},
        )

        self.assertEqual(busy.status_code, 409)
        self.assertEqual(busy.json()["detail"]["code"], "session_busy")
        self.assertEqual(unavailable.status_code, 503)
        self.assertNotIn("secret", unavailable.text)
        self.assertEqual(broken.status_code, 500)
        self.assertNotIn("private-stack-detail", broken.text)

    def test_runner_scope_permission_error_is_hidden_as_session_not_found(self):
        from src.api.service import SessionService

        thread_id = "89898989-8989-4989-8989-898989898989"
        path = f"/api/v1/projects/project-a/sessions/{thread_id}/messages"
        runner = MagicMock()
        runner.graph.get_state.return_value = SimpleNamespace(
            values={
                "session_user_id": "user-a",
                "session_project_id": "project-a",
                "status": "completed",
            },
            next=(),
            tasks=(),
        )
        runner.astream.side_effect = _stream_error(PermissionError("another project"))
        service = SessionService(runner)

        response = self._client_with_service(service).post(
            path,
            headers={"X-User-ID": "user-a"},
            json={"message": "hello"},
        )

        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["detail"]["code"], "session_not_found")
        self.assertNotIn("another project", response.text)


class ApiContractTest(unittest.TestCase):
    @staticmethod
    @contextmanager
    def _application_factory():
        yield MagicMock()

    def test_default_cors_allowlist_accepts_loopback_frontend_origin(self):
        from src.api.main import create_app

        with patch.dict(os.environ):
            os.environ.pop("API_CORS_ORIGINS", None)
            app = create_app(application_factory=self._application_factory)
        with TestClient(app) as client:
            response = client.get(
                "/health/live",
                headers={"Origin": "http://127.0.0.1:3000"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.headers.get("access-control-allow-origin"),
            "http://127.0.0.1:3000",
        )

    def test_cors_allowlist_and_openapi_include_the_public_contract(self):
        from src.api.main import create_app

        with patch.dict(
            os.environ,
            {"API_CORS_ORIGINS": "https://frontend.example"},
        ):
            app = create_app(application_factory=self._application_factory)
        with TestClient(app) as client:
            preflight = client.options(
                "/api/v1/projects/project-a/sessions",
                headers={
                    "Origin": "https://frontend.example",
                    "Access-Control-Request-Method": "POST",
                    "Access-Control-Request-Headers": (
                        "X-User-ID,X-Request-ID,X-Thread-ID"
                    ),
                },
            )
            openapi = client.get("/openapi.json").json()

        self.assertEqual(preflight.status_code, 200)
        self.assertEqual(
            preflight.headers["access-control-allow-origin"],
            "https://frontend.example",
        )
        self.assertEqual(preflight.status_code, 200)
        self.assertIn(
            "x-thread-id",
            preflight.headers["access-control-allow-headers"].lower(),
        )
        self.assertEqual(
            set(openapi["paths"]),
            {
                "/health/live",
                "/api/v1/projects",
                "/api/v1/projects/{project_id}",
                "/api/v1/projects/{project_id}/sessions",
                "/api/v1/projects/{project_id}/sessions/import",
                "/api/v1/projects/{project_id}/sessions/stream",
                "/api/v1/projects/{project_id}/sessions/{thread_id}/messages",
                "/api/v1/projects/{project_id}/sessions/{thread_id}/messages/stream",
                "/api/v1/projects/{project_id}/sessions/{thread_id}/cancel",
                "/api/v1/projects/{project_id}/sessions/{thread_id}",
                "/api/v1/projects/{project_id}/sessions/{thread_id}/report",
                "/api/v1/projects/{project_id}/sessions/{thread_id}/reports/{report_id}",
                "/api/v1/projects/{project_id}/sessions/{thread_id}/reports/{report_id}/content",
                "/api/v1/reports",
            },
        )

    def test_blank_project_invalid_request_id_and_long_message_return_422(self):
        from src.api.main import create_app

        service = MagicMock()
        service.create_session.return_value = {
            "request_id": "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
            "project_id": "project-a",
            "thread_id": "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb",
            "status": "completed",
        }
        app = create_app(application_factory=self._application_factory)
        with TestClient(app) as client:
            client.app.state.session_service = service
            blank_project = client.post(
                "/api/v1/projects/%20%20%20/sessions",
                headers={"X-User-ID": "user-a"},
                json={"message": "hello"},
            )
            invalid_request_id = client.post(
                "/api/v1/projects/project-a/sessions",
                headers={
                    "X-User-ID": "user-a",
                    "X-Request-ID": "not-a-uuid",
                },
                json={"message": "hello"},
            )
            long_message = client.post(
                "/api/v1/projects/project-a/sessions",
                headers={"X-User-ID": "user-a"},
                json={"message": "x" * 10_001},
            )

        self.assertEqual(blank_project.status_code, 422)
        self.assertEqual(invalid_request_id.status_code, 422)
        self.assertEqual(long_message.status_code, 422)
        service.create_session.assert_not_called()


if __name__ == "__main__":
    unittest.main()

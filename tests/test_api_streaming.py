import asyncio
import json
import shutil
import unittest
import warnings
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock
from uuid import uuid4

from starlette.exceptions import StarletteDeprecationWarning

warnings.filterwarnings(
    "ignore",
    message="Using `httpx` with `starlette.testclient` is deprecated.*",
    category=StarletteDeprecationWarning,
)

from fastapi.testclient import TestClient


THREAD_ID = "33333333-3333-4333-8333-333333333333"
HEADERS = {
    "X-User-ID": "user-a",
    "X-Request-ID": "44444444-4444-4444-8444-444444444444",
}
COMPLETED = {
    "session_user_id": "user-a",
    "session_project_id": "project-a",
    "status": "completed",
    "query_type": "rag_qa",
    "response_type": "answer",
    "answer": "FinFET 是鳍式晶体管。",
}


def _snapshot(values=None):
    return SimpleNamespace(values=dict(values or {}), next=(), tasks=())


def _parse_sse(text):
    events = []
    for frame in text.split("\n\n"):
        lines = [line for line in frame.splitlines() if not line.startswith(":")]
        if not lines:
            continue
        name = next(line[7:] for line in lines if line.startswith("event: "))
        data = next(line[6:] for line in lines if line.startswith("data: "))
        events.append((name, json.loads(data)))
    return events


def _streaming_runner(*deltas, error=None):
    from src.observability.answer_stream import emit_answer_delta, emit_answer_reset

    async def stream(_payload, *, config):
        # Nodes run in worker threads, exactly like async_node.
        await asyncio.to_thread(emit_answer_reset)
        for delta in deltas:
            await asyncio.to_thread(emit_answer_delta, delta)
        if error is not None:
            raise error
        yield COMPLETED

    runner = MagicMock()
    runner.astream.side_effect = stream
    return runner


class ApiStreamingTest(unittest.TestCase):
    def setUp(self):
        self.report_directory = Path("runtime") / "test-stream-reports" / uuid4().hex
        self.report_directory.mkdir(parents=True)
        self.addCleanup(shutil.rmtree, self.report_directory, ignore_errors=True)

    @staticmethod
    def _client(runner):
        from src.api.main import create_app

        @contextmanager
        def application_factory():
            yield runner

        return TestClient(create_app(application_factory=application_factory))

    def test_create_stream_relays_deltas_then_final_response(self):
        runner = _streaming_runner("FinFET ", "是鳍式晶体管。")
        runner.graph.get_state.side_effect = [_snapshot(), _snapshot(COMPLETED)]

        with self._client(runner) as client:
            response = client.post(
                "/api/v1/projects/project-a/sessions/stream",
                headers={**HEADERS, "X-Thread-ID": THREAD_ID},
                json={"message": "什么是FinFET？"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.headers["content-type"].startswith("text/event-stream"))
        events = _parse_sse(response.text)
        self.assertEqual(
            [name for name, _ in events],
            ["reset", "delta", "delta", "final"],
        )
        self.assertEqual(events[1][1], {"text": "FinFET "})
        final = events[-1][1]
        self.assertEqual(final["thread_id"], THREAD_ID)
        self.assertEqual(final["status"], "completed")
        self.assertEqual(final["answer"], COMPLETED["answer"])

    def test_message_stream_to_another_users_thread_is_plain_404(self):
        runner = _streaming_runner("never")
        runner.graph.get_state.return_value = _snapshot(
            {**COMPLETED, "session_user_id": "user-b"}
        )

        with self._client(runner) as client:
            response = client.post(
                f"/api/v1/projects/project-a/sessions/{THREAD_ID}/messages/stream",
                headers=HEADERS,
                json={"message": "继续"},
            )

        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["detail"]["code"], "session_not_found")
        runner.astream.assert_not_called()

    def test_dependency_failure_mid_stream_becomes_sanitized_error_event(self):
        runner = _streaming_runner("部分", error=ConnectionError("mysql down"))
        runner.graph.get_state.return_value = _snapshot(COMPLETED)

        with self._client(runner) as client:
            response = client.post(
                f"/api/v1/projects/project-a/sessions/{THREAD_ID}/messages/stream",
                headers=HEADERS,
                json={"message": "继续"},
            )

        events = _parse_sse(response.text)
        self.assertEqual(events[-1][0], "error")
        self.assertEqual(events[-1][1]["code"], "service_unavailable")
        self.assertNotIn("mysql", response.text)

    def test_busy_thread_is_rejected_before_streaming(self):
        from src.api.schemas import MessageRequest
        from src.api.service import SessionBusyError, SessionService

        release = asyncio.Event()

        async def stream(_payload, *, config):
            await release.wait()
            yield COMPLETED

        runner = MagicMock()
        runner.graph.get_state.return_value = _snapshot(COMPLETED)
        runner.astream.side_effect = stream
        service = SessionService(runner, report_output_dir=self.report_directory)
        kwargs = {
            "user_id": "user-a",
            "project_id": "project-a",
            "thread_id": THREAD_ID,
            "message": MessageRequest(message="继续"),
        }

        async def scenario():
            frames = await service.stream_send_message(request_id="r-1", **kwargs)
            with self.assertRaises(SessionBusyError):
                await service.stream_send_message(request_id="r-2", **kwargs)
            release.set()
            return [frame async for frame in frames]

        frames = asyncio.run(scenario())
        self.assertTrue(frames[-1].startswith("event: final"))
        self.assertEqual(runner.astream.call_count, 1)

    def test_workflow_finishes_and_releases_slot_after_client_disconnects(self):
        from src.api.schemas import MessageRequest
        from src.api.service import SessionService

        finished = []

        async def stream(_payload, *, config):
            await asyncio.sleep(0.01)
            finished.append(config["configurable"]["request_id"])
            yield COMPLETED

        runner = MagicMock()
        runner.graph.get_state.return_value = _snapshot(COMPLETED)
        runner.astream.side_effect = stream
        service = SessionService(runner, report_output_dir=self.report_directory)
        kwargs = {
            "user_id": "user-a",
            "project_id": "project-a",
            "thread_id": THREAD_ID,
            "message": MessageRequest(message="继续"),
        }

        async def scenario():
            frames = await service.stream_send_message(request_id="r-1", **kwargs)
            await frames.aclose()
            await asyncio.gather(*service._stream_tasks)
            follow_up = await service.stream_send_message(request_id="r-2", **kwargs)
            return [frame async for frame in follow_up]

        frames = asyncio.run(scenario())
        self.assertEqual(finished, ["r-1", "r-2"])
        self.assertTrue(frames[-1].startswith("event: final"))


if __name__ == "__main__":
    unittest.main()

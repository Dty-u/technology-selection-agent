"""Session history is scoped as user_id -> project_id -> thread_id."""

from datetime import datetime, timezone
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock
from uuid import uuid4

from fastapi.testclient import TestClient

from src.api.main import create_app
from src.infrastructure.sessions.repository import MySQLSessionRepository


class FakeHistory:
    """In-memory stand-in with the repository's scoping rules."""

    def __init__(self):
        self.projects = {}
        self.sessions = {}
        self.messages = []
        self.fail_writes = False
        self.statuses = []

    def _now(self):
        return datetime(2026, 1, 1, 8, 0, 0, 123456)

    def ensure_project(self, *, user_id, project_id, name=None):
        row = self.projects.setdefault(
            (user_id, project_id),
            {
                "project_id": project_id,
                "name": name or project_id,
                "created_at": self._now(),
                "updated_at": self._now(),
            },
        )
        if name:
            row["name"] = name
        return row

    def list_projects(self, *, user_id):
        return [row for key, row in self.projects.items() if key[0] == user_id]

    def begin_turn(self, *, user_id, project_id, thread_id, request_id, content, title):
        if self.fail_writes:
            raise RuntimeError("mysql down")
        self.ensure_project(user_id=user_id, project_id=project_id)
        session = self.sessions.setdefault(
            thread_id,
            {
                "thread_id": thread_id,
                "user_id": user_id,
                "project_id": project_id,
                "title": title,
                "created_at": self._now(),
                "updated_at": self._now(),
            },
        )
        session["status"] = "running"
        self.messages.append(
            {
                "id": len(self.messages) + 1,
                "thread_id": thread_id,
                "request_id": request_id,
                "role": "user",
                "content": content,
                "delivery": "pending",
                "kind": "message",
                "created_at": self._now(),
            }
        )

    def finish_turn(
        self,
        *,
        thread_id,
        request_id,
        status,
        delivery,
        assistant_content="",
        kind="message",
        metadata=None,
    ):
        self.statuses.append(status)
        self.sessions[thread_id]["status"] = status
        for message in self.messages:
            if message["request_id"] == request_id and message["role"] == "user":
                message["delivery"] = delivery
        if assistant_content:
            self.messages.append(
                {
                    "id": len(self.messages) + 1,
                    "thread_id": thread_id,
                    "request_id": request_id,
                    "role": "assistant",
                    "content": assistant_content,
                    "delivery": "sent",
                    "kind": kind,
                    "created_at": self._now(),
                }
            )

    def fail_turn(self, *, thread_id, request_id):
        self.messages = [
            message
            for message in self.messages
            if not (message["request_id"] == request_id and message["role"] == "user")
        ]
        if not any(message["thread_id"] == thread_id for message in self.messages):
            self.sessions.pop(thread_id, None)

    def update_status(self, *, thread_id, status):
        self.sessions[thread_id]["status"] = status

    def list_sessions(self, *, user_id, project_id, limit):
        return [
            row
            for row in self.sessions.values()
            if row["user_id"] == user_id and row["project_id"] == project_id
        ][:limit]

    def get_session(self, *, user_id, project_id, thread_id):
        row = self.sessions.get(thread_id)
        if row and row["user_id"] == user_id and row["project_id"] == project_id:
            return row
        return None

    def list_messages(self, *, thread_id, limit):
        return [m for m in self.messages if m["thread_id"] == thread_id][-limit:]

    def delete_session(self, *, thread_id):
        self.sessions.pop(thread_id, None)
        self.messages = [m for m in self.messages if m["thread_id"] != thread_id]

    def import_session(
        self,
        *,
        user_id,
        project_id,
        thread_id,
        title,
        status,
        created_at,
        updated_at,
        messages,
    ):
        if thread_id in self.sessions:
            return False
        self.sessions[thread_id] = {
            "thread_id": thread_id,
            "user_id": user_id,
            "project_id": project_id,
            "title": title,
            "status": status,
            "created_at": created_at,
            "updated_at": updated_at,
        }
        for message in messages:
            self.messages.append(
                {
                    "id": len(self.messages) + 1,
                    "thread_id": thread_id,
                    "kind": "message",
                    **message,
                }
            )
        return True


class SessionHistoryApiTest(unittest.TestCase):
    def setUp(self):
        self.history = FakeHistory()
        self.state = {}
        self.next = ()
        self.stream_error = None
        self.runner = MagicMock()

        async def astream(payload, *, config):
            if self.stream_error:
                raise self.stream_error
            self.state = {
                "status": "completed",
                "answer": self.answer,
                "session_user_id": config["configurable"]["user_id"],
                "session_project_id": config["configurable"]["project_id"],
            }
            yield {}

        self.answer = "回答"
        self.runner.astream.side_effect = astream
        self.runner.graph.get_state.side_effect = lambda config: SimpleNamespace(
            values=self.state,
            next=self.next,
            tasks=(),
        )
        self.runner.close_session = MagicMock()
        del self.runner.aget_state
        del self.runner.aclose_session
        app = create_app(
            application_factory=self._application_factory,
            session_repository_factory=lambda: self.history,
        )
        self.client = TestClient(app, raise_server_exceptions=False)
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)

    def _application_factory(self):
        runner = self.runner

        class Context:
            def __enter__(self):
                return runner

            def __exit__(self, *_):
                return False

        return Context()

    def _send(self, user="user-a", project="project-a", message="问题"):
        thread_id = str(uuid4())
        response = self.client.post(
            f"/api/v1/projects/{project}/sessions",
            json={"message": message},
            headers={"X-User-ID": user, "X-Thread-ID": thread_id},
        )
        return thread_id, response

    def test_turn_is_stored_with_user_message_and_answer(self):
        thread_id, response = self._send()

        self.assertEqual(response.status_code, 201)
        stored = self.client.get(
            f"/api/v1/projects/project-a/sessions/{thread_id}/messages",
            headers={"X-User-ID": "user-a"},
        ).json()["messages"]
        self.assertEqual(
            [(m["role"], m["content"], m["delivery"]) for m in stored],
            [("user", "问题", "sent"), ("assistant", "回答", "sent")],
        )

    def test_streamed_turn_is_stored_too(self):
        thread_id = str(uuid4())

        response = self.client.post(
            "/api/v1/projects/project-a/sessions/stream",
            json={"message": "问题"},
            headers={"X-User-ID": "user-a", "X-Thread-ID": thread_id},
        )

        self.assertIn("event: final", response.text)
        self.assertEqual(
            [m["role"] for m in self.history.list_messages(thread_id=thread_id, limit=10)],
            ["user", "assistant"],
        )

    def test_same_project_id_is_separate_for_each_user(self):
        thread_a, _ = self._send(user="user-a", project="shared-name")
        self.state = {}
        thread_b, _ = self._send(user="user-b", project="shared-name")

        def visible(user):
            return [
                row["thread_id"]
                for row in self.client.get(
                    "/api/v1/projects/shared-name/sessions",
                    headers={"X-User-ID": user},
                ).json()["sessions"]
            ]

        self.assertEqual(visible("user-a"), [thread_a])
        self.assertEqual(visible("user-b"), [thread_b])

    def test_projects_are_listed_per_user(self):
        self._send(user="user-a", project="project-a")
        self.state = {}
        self._send(user="user-b", project="project-b")
        self.client.put(
            "/api/v1/projects/project-c",
            json={"name": "第三个"},
            headers={"X-User-ID": "user-a"},
        )

        projects = self.client.get(
            "/api/v1/projects",
            headers={"X-User-ID": "user-a"},
        ).json()["projects"]

        self.assertEqual(
            {(row["project_id"], row["name"]) for row in projects},
            {("project-a", "project-a"), ("project-c", "第三个")},
        )

    def test_another_user_cannot_read_messages(self):
        thread_id, _ = self._send(user="user-a")

        response = self.client.get(
            f"/api/v1/projects/project-a/sessions/{thread_id}/messages",
            headers={"X-User-ID": "user-b"},
        )

        self.assertEqual(response.status_code, 404)

    def test_another_project_of_the_same_user_cannot_read_messages(self):
        thread_id, _ = self._send(project="project-a")

        response = self.client.get(
            f"/api/v1/projects/project-b/sessions/{thread_id}/messages",
            headers={"X-User-ID": "user-a"},
        )

        self.assertEqual(response.status_code, 404)

    def test_history_outage_rejects_new_turn_before_running_workflow(self):
        self.history.fail_writes = True

        _, response = self._send()

        self.assertEqual(response.status_code, 503)
        self.runner.astream.assert_not_called()

    def test_failed_workflow_leaves_no_undelivered_turn(self):
        self.stream_error = ConnectionError("provider down")

        _, response = self._send()

        self.assertEqual(response.status_code, 503)
        self.assertEqual(self.history.sessions, {})
        self.assertEqual(self.history.messages, [])

    def test_answer_survives_history_write_failure_after_workflow(self):
        self.history.finish_turn = MagicMock(side_effect=RuntimeError("mysql down"))

        _, response = self._send()

        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.json()["answer"], "回答")
        self.assertEqual(self.history.finish_turn.call_count, 2)

    def test_clarification_is_marked_so_clients_can_keep_their_card(self):
        async def astream(payload, *, config):
            self.state = {
                "status": "completed",
                "answer": "上一轮回答",
                "clarification_questions": ["预算是多少？"],
                "session_user_id": "user-a",
                "session_project_id": "project-a",
            }
            self.next = ("clarify",)
            yield {}

        self.runner.astream.side_effect = astream

        thread_id, response = self._send()

        self.assertEqual(response.json()["status"], "awaiting_user")
        stored = self.history.list_messages(thread_id=thread_id, limit=10)
        self.assertEqual(
            [(m["role"], m["content"], m["kind"]) for m in stored],
            [
                ("user", "问题", "message"),
                ("assistant", "预算是多少？", "clarification"),
            ],
        )

    def test_stale_running_status_is_corrected_from_the_checkpoint(self):
        thread_id, _ = self._send()
        self.history.sessions[thread_id]["status"] = "running"

        sessions = self.client.get(
            "/api/v1/projects/project-a/sessions",
            headers={"X-User-ID": "user-a"},
        ).json()["sessions"]

        self.assertEqual(sessions[0]["status"], "completed")
        self.assertEqual(self.history.sessions[thread_id]["status"], "completed")

    def test_close_removes_history(self):
        thread_id, _ = self._send()

        response = self.client.delete(
            f"/api/v1/projects/project-a/sessions/{thread_id}",
            headers={"X-User-ID": "user-a"},
        )

        self.assertEqual(response.status_code, 204)
        self.assertEqual(self.history.sessions, {})
        self.assertEqual(self.history.messages, [])

    def _import_body(self, thread_id):
        return {
            "thread_id": thread_id,
            "title": "旧会话",
            "status": "completed",
            "created_at": "2025-01-01T00:00:00.000Z",
            "updated_at": "2025-01-02T00:00:00.000Z",
            "messages": [
                {
                    "id": "local-1",
                    "role": "user",
                    "content": "旧问题",
                    "delivery": "sent",
                    "created_at": "2025-01-01T00:00:00.000Z",
                    "request_id": str(uuid4()),
                },
                {
                    "id": "local-2",
                    "role": "assistant",
                    "content": "旧回答",
                    "delivery": "sent",
                    "created_at": "2025-01-01T00:00:01.000Z",
                },
            ],
        }

    def test_import_copies_browser_history_once(self):
        thread_id = str(uuid4())
        url = "/api/v1/projects/project-a/sessions/import"
        headers = {"X-User-ID": "user-a"}

        first = self.client.post(url, json=self._import_body(thread_id), headers=headers)
        second = self.client.post(url, json=self._import_body(thread_id), headers=headers)

        self.assertTrue(first.json()["imported"])
        self.assertFalse(second.json()["imported"])
        self.assertEqual(len(self.history.messages), 2)
        self.assertEqual(
            self.history.sessions[thread_id]["created_at"],
            datetime(2025, 1, 1, 0, 0),
        )

    def test_import_rejects_a_thread_owned_by_another_user(self):
        self.state = {"session_user_id": "user-b", "session_project_id": "project-a"}

        response = self.client.post(
            "/api/v1/projects/project-a/sessions/import",
            json=self._import_body(str(uuid4())),
            headers={"X-User-ID": "user-a"},
        )

        self.assertEqual(response.status_code, 404)
        self.assertEqual(self.history.sessions, {})

    def test_history_without_a_checkpoint_can_still_be_closed(self):
        thread_id = str(uuid4())
        headers = {"X-User-ID": "user-a"}
        self.client.post(
            "/api/v1/projects/project-a/sessions/import",
            json=self._import_body(thread_id),
            headers=headers,
        )

        response = self.client.delete(
            f"/api/v1/projects/project-a/sessions/{thread_id}",
            headers=headers,
        )

        self.assertEqual(response.status_code, 204)
        self.assertEqual(self.history.sessions, {})


class MySQLSessionRepositoryTest(unittest.TestCase):
    def _repository(self):
        self.cursor = MagicMock()
        connection = MagicMock()
        connection.cursor.return_value.__enter__.return_value = self.cursor
        self.connection = connection
        return MySQLSessionRepository(
            "mysql://user:secret@localhost/db",
            connection_factory=lambda: connection,
        )

    def test_listing_sessions_filters_by_user_and_project(self):
        repository = self._repository()
        self.cursor.fetchall.return_value = []

        repository.list_sessions(user_id="user-a", project_id="project-a", limit=5)

        sql, parameters = self.cursor.execute.call_args.args
        self.assertIn("user_id = %s AND project_id = %s", sql)
        self.assertEqual(parameters, ("user-a", "project-a", 5))

    def test_listing_a_users_sessions_spans_projects_but_not_users(self):
        repository = self._repository()
        self.cursor.fetchall.return_value = []

        repository.list_user_sessions(user_id="user-a", limit=5)

        sql, parameters = self.cursor.execute.call_args.args
        self.assertIn("WHERE user_id = %s ", sql)
        self.assertNotIn("project_id", sql)
        self.assertEqual(parameters, ("user-a", 5))

    def test_repeated_answer_is_not_stored_twice(self):
        repository = self._repository()
        self.cursor.fetchone.return_value = {"content": "相同回答"}

        repository.finish_turn(
            thread_id="thread",
            request_id="request",
            status="completed",
            delivery="sent",
            assistant_content="相同回答",
        )

        self.assertFalse(
            any(
                "INSERT" in call.args[0]
                for call in self.cursor.execute.call_args_list
            )
        )
        self.connection.commit.assert_called_once_with()

    def test_failed_write_rolls_back(self):
        repository = self._repository()
        self.cursor.execute.side_effect = RuntimeError("boom")

        with self.assertRaises(RuntimeError):
            repository.begin_turn(
                user_id="user-a",
                project_id="project-a",
                thread_id="thread",
                request_id="request",
                content="问题",
                title="问题",
            )

        self.connection.rollback.assert_called_once_with()
        self.connection.close.assert_called_once_with()

    def test_timestamps_can_be_built_without_a_timezone(self):
        from src.api.service import _timestamp, _utc_naive

        aware = datetime(2026, 1, 1, 8, 0, 0, 123456, tzinfo=timezone.utc)

        self.assertEqual(_timestamp(_utc_naive(aware)), "2026-01-01T08:00:00.123Z")


if __name__ == "__main__":
    unittest.main()

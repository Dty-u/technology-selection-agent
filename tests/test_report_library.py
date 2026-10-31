"""The report library lists, reads and deletes one user's reports across projects."""

from contextlib import contextmanager
from pathlib import Path
import shutil
from types import SimpleNamespace
from typing import TypedDict
import unittest
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from src.api.main import create_app
from src.api.service import SessionService
from src.workflows.memory_runner import MemoryAwareRunner


THREAD_A = "11111111-1111-4111-8111-111111111111"
THREAD_B = "22222222-2222-4222-8222-222222222222"
THREAD_OTHER_USER = "33333333-3333-4333-8333-333333333333"
THREAD_NO_CHECKPOINT = "44444444-4444-4444-8444-444444444444"


def _report(report_id, created_at, *, path=None, text=None):
    return {
        "report_id": report_id,
        "title": f"{report_id} 报告",
        "created_at": created_at,
        "format": "docx" if path else "markdown",
        "report": text or f"# {report_id}",
        "report_path": str(path) if path else None,
    }


class FakeGraph:
    """Checkpoints by thread with LangGraph's whole-channel update semantics."""

    def __init__(self):
        self.states = {}
        self.paused = set()

    def add(self, thread_id, *, user, project, reports=(), current=None):
        self.states[thread_id] = {
            "session_user_id": user,
            "session_project_id": project,
            "status": "completed",
            "query_type": "technology_selection",
            "response_type": "report",
            "report": (current or {}).get("report", ""),
            "report_data": {"requirements": {"business_goal": "x"}} if current else {},
            "report_path": (current or {}).get("report_path"),
            "report_history": list(reports),
        }

    def get_state(self, config):
        thread_id = config["configurable"]["thread_id"]
        return SimpleNamespace(
            values=dict(self.states.get(thread_id, {})),
            next=("clarify",) if thread_id in self.paused else (),
            tasks=(),
        )

    def update_state(self, config, values):
        self.states[config["configurable"]["thread_id"]].update(values)


class FakeHistory:
    def __init__(self, rows):
        self.rows = rows

    def list_user_sessions(self, *, user_id, limit):
        return [row for row in self.rows if row["user_id"] == user_id][:limit]


def _row(thread_id, user, project, title="会话标题"):
    return {
        "thread_id": thread_id,
        "user_id": user,
        "project_id": project,
        "title": title,
    }


class ReportLibraryTest(unittest.TestCase):
    def setUp(self):
        self.root = Path.cwd() / "runtime" / f"api-test-{uuid4().hex}"
        self.reports_dir = self.root / "reports"
        self.reports_dir.mkdir(parents=True)
        self.addCleanup(shutil.rmtree, self.root, True)
        self.graph = FakeGraph()
        self.history = FakeHistory(
            [
                _row(THREAD_A, "user-a", "project-a", "第一个会话"),
                _row(THREAD_B, "user-a", "project-b", "第二个会话"),
                _row(THREAD_OTHER_USER, "user-b", "project-a"),
                _row(THREAD_NO_CHECKPOINT, "user-a", "project-a"),
            ]
        )
        runner = MagicMock()
        runner.graph = self.graph
        # A MagicMock attribute is not a coroutine function, so the service uses graph.*.

        @contextmanager
        def application_factory():
            yield runner

        app = create_app(application_factory=application_factory)
        self.client = TestClient(app)
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)
        self.client.app.state.session_service = SessionService(
            runner,
            report_output_dir=self.reports_dir,
            session_repository=self.history,
        )

    def _docx(self, name="selection.docx"):
        path = self.reports_dir / name
        path.write_bytes(b"docx")
        return path

    def _url(self, project, thread, report_id, suffix=""):
        return (
            f"/api/v1/projects/{project}/sessions/{thread}/reports/{report_id}{suffix}"
        )

    def test_lists_only_this_users_reports_across_projects_newest_first(self):
        docx = self._docx()
        self.graph.add(
            THREAD_A,
            user="user-a",
            project="project-a",
            reports=[_report("old", "2026-01-01T00:00:00+00:00", path=docx)],
        )
        self.graph.add(
            THREAD_B,
            user="user-a",
            project="project-b",
            reports=[_report("new", "2026-02-01T00:00:00+00:00")],
        )
        self.graph.add(
            THREAD_OTHER_USER,
            user="user-b",
            project="project-a",
            reports=[_report("foreign", "2026-03-01T00:00:00+00:00")],
        )

        response = self.client.get("/api/v1/reports", headers={"X-User-ID": "user-a"})

        self.assertEqual(response.status_code, 200)
        reports = response.json()["reports"]
        self.assertEqual([item["report_id"] for item in reports], ["new", "old"])
        self.assertEqual(reports[0]["project_id"], "project-b")
        self.assertEqual(reports[0]["session_title"], "第二个会话")
        self.assertEqual(reports[1]["format"], "docx")
        self.assertEqual(reports[1]["thread_id"], THREAD_A)
        self.assertEqual(
            reports[1]["download_url"],
            self._url("project-a", THREAD_A, "old"),
        )
        self.assertNotIn(str(docx), response.text)

    def test_listing_needs_a_user(self):
        self.assertEqual(self.client.get("/api/v1/reports").status_code, 422)

    def test_content_returns_report_text_only_to_its_owner(self):
        docx = self._docx()
        self.graph.add(
            THREAD_A,
            user="user-a",
            project="project-a",
            reports=[
                _report("r1", "2026-01-01T00:00:00+00:00", path=docx, text="# 正文"),
            ],
        )
        url = self._url("project-a", THREAD_A, "r1", "/content")

        owned = self.client.get(url, headers={"X-User-ID": "user-a"})
        foreign = self.client.get(url, headers={"X-User-ID": "user-b"})
        missing = self.client.get(
            self._url("project-a", THREAD_A, "nope", "/content"),
            headers={"X-User-ID": "user-a"},
        )

        self.assertEqual(owned.status_code, 200)
        self.assertEqual(owned.json()["content"], "# 正文")
        self.assertEqual(owned.json()["format"], "docx")
        self.assertNotIn(str(docx), owned.text)
        self.assertEqual(foreign.status_code, 404)
        self.assertEqual(missing.status_code, 404)

    def test_deleting_an_older_report_keeps_the_current_one(self):
        old_docx = self._docx("old.docx")
        current_docx = self._docx("current.docx")
        current = _report(
            "current",
            "2026-02-01T00:00:00+00:00",
            path=current_docx,
            text="# 当前",
        )
        self.graph.add(
            THREAD_A,
            user="user-a",
            project="project-a",
            reports=[
                _report("old", "2026-01-01T00:00:00+00:00", path=old_docx),
                current,
            ],
            current=current,
        )

        response = self.client.delete(
            self._url("project-a", THREAD_A, "old"),
            headers={"X-User-ID": "user-a"},
        )

        self.assertEqual(response.status_code, 204)
        state = self.graph.states[THREAD_A]
        self.assertEqual([r["report_id"] for r in state["report_history"]], ["current"])
        self.assertEqual(state["report"], "# 当前")
        self.assertEqual(state["report_path"], str(current_docx))
        self.assertFalse(old_docx.exists())
        self.assertTrue(current_docx.exists())

    def test_deleting_the_current_report_clears_it_from_the_session(self):
        docx = self._docx()
        current = _report(
            "current",
            "2026-02-01T00:00:00+00:00",
            path=docx,
            text="# 当前",
        )
        self.graph.add(
            THREAD_A,
            user="user-a",
            project="project-a",
            reports=[current],
            current=current,
        )

        response = self.client.delete(
            self._url("project-a", THREAD_A, "current"),
            headers={"X-User-ID": "user-a"},
        )
        session = self.client.get(
            f"/api/v1/projects/project-a/sessions/{THREAD_A}",
            headers={"X-User-ID": "user-a"},
        )

        self.assertEqual(response.status_code, 204)
        self.assertFalse(docx.exists())
        self.assertEqual(session.status_code, 200)
        self.assertEqual(session.json()["report"], "")
        self.assertEqual(session.json()["report_data"], {})
        self.assertIsNone(session.json()["report_url"])
        self.assertEqual(session.json()["reports"], [])

    def test_a_file_shared_with_a_kept_report_is_not_removed(self):
        docx = self._docx()
        self.graph.add(
            THREAD_A,
            user="user-a",
            project="project-a",
            reports=[
                _report("one", "2026-01-01T00:00:00+00:00", path=docx),
                _report("two", "2026-02-01T00:00:00+00:00", path=docx),
            ],
        )

        response = self.client.delete(
            self._url("project-a", THREAD_A, "one"),
            headers={"X-User-ID": "user-a"},
        )

        self.assertEqual(response.status_code, 204)
        self.assertTrue(docx.exists())

    def test_another_user_cannot_delete_a_report(self):
        docx = self._docx()
        self.graph.add(
            THREAD_A,
            user="user-a",
            project="project-a",
            reports=[_report("r1", "2026-01-01T00:00:00+00:00", path=docx)],
        )

        response = self.client.delete(
            self._url("project-a", THREAD_A, "r1"),
            headers={"X-User-ID": "user-b"},
        )

        self.assertEqual(response.status_code, 404)
        self.assertTrue(docx.exists())
        self.assertEqual(len(self.graph.states[THREAD_A]["report_history"]), 1)

    def test_a_paused_session_refuses_deletion_without_changing_state(self):
        docx = self._docx()
        self.graph.add(
            THREAD_A,
            user="user-a",
            project="project-a",
            reports=[_report("r1", "2026-01-01T00:00:00+00:00", path=docx)],
        )
        self.graph.paused.add(THREAD_A)

        response = self.client.delete(
            self._url("project-a", THREAD_A, "r1"),
            headers={"X-User-ID": "user-a"},
        )

        self.assertEqual(response.status_code, 409)
        self.assertTrue(docx.exists())
        self.assertEqual(len(self.graph.states[THREAD_A]["report_history"]), 1)

    def test_deleting_an_unknown_report_is_not_found(self):
        self.graph.add(THREAD_A, user="user-a", project="project-a")

        response = self.client.delete(
            self._url("project-a", THREAD_A, "nope"),
            headers={"X-User-ID": "user-a"},
        )

        self.assertEqual(response.status_code, 404)


class RunnerStateUpdateTest(unittest.IsolatedAsyncioTestCase):
    async def _assert_update_finishes_at(self, terminal_node):
        class State(TypedDict, total=False):
            report: str
            report_history: list

        graph = StateGraph(State)
        graph.add_node("start", lambda state: {"report": "text"})
        graph.add_node(terminal_node, lambda state: {})
        graph.add_edge(START, "start")
        graph.add_edge("start", terminal_node)
        graph.add_edge(terminal_node, END)
        compiled = graph.compile(checkpointer=InMemorySaver())
        runner = MemoryAwareRunner(graph=compiled, checkpointer=InMemorySaver())
        config = {"configurable": {"thread_id": "t"}}
        await compiled.ainvoke({}, config)

        await runner.aupdate_state(config, {"report": "", "report_history": []})

        state = await compiled.aget_state(config)
        self.assertEqual(state.next, ())
        self.assertEqual(state.values["report"], "")

    async def test_update_on_a_finished_memory_graph_schedules_nothing(self):
        await self._assert_update_finishes_at("persist_memory_candidates")

    async def test_update_on_a_finished_memoryless_graph_schedules_nothing(self):
        await self._assert_update_finishes_at("finalize_short_term_memory")

    async def test_update_names_its_writer_so_langgraph_need_not_infer_one(self):
        graph = MagicMock()
        graph.nodes = {"finalize_short_term_memory": object()}
        graph.aupdate_state = AsyncMock()
        runner = MemoryAwareRunner(graph=graph, checkpointer=MagicMock())

        await runner.aupdate_state({"configurable": {}}, {"report": ""})

        graph.aupdate_state.assert_awaited_once_with(
            {"configurable": {}},
            {"report": ""},
            as_node="finalize_short_term_memory",
        )


if __name__ == "__main__":
    unittest.main()

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, TypedDict
from unittest.mock import Mock

_WORKSPACE_TMP_DIR = Path.cwd()

from langchain_core.documents import Document
from langgraph.constants import START
from langgraph.graph import StateGraph


class _CheckpointState(TypedDict, total=False):
    payload: Any


def _build_checkpoint_test_graph(checkpointer):
    workflow = StateGraph(_CheckpointState)
    workflow.add_node("finish", lambda _state: {})
    workflow.add_edge(START, "finish")
    return workflow.compile(checkpointer=checkpointer)


class SqliteCheckpointingTest(unittest.TestCase):
    def test_open_is_lazy_and_strict_serializer_restores_document(self):
        from src.infrastructure.checkpointing.sqlite import (
            open_sqlite_checkpointer,
        )

        with TemporaryDirectory(dir=_WORKSPACE_TMP_DIR) as directory:
            database_path = Path(directory) / "nested" / "workflow.sqlite3"
            manager = open_sqlite_checkpointer(database_path)

            self.assertFalse(database_path.exists())
            with manager as checkpointer:
                graph = _build_checkpoint_test_graph(checkpointer)
                config = {"configurable": {"thread_id": "document-thread"}}
                document = Document(
                    page_content="checkpoint body",
                    metadata={"source": "unit-test"},
                )

                graph.invoke({"payload": document}, config=config)
                restored = graph.get_state(config).values["payload"]

                self.assertEqual(restored, document)
                self.assertTrue(database_path.exists())

    def test_close_session_deletes_only_the_requested_thread(self):
        from src.infrastructure.checkpointing.sqlite import (
            close_session,
            open_sqlite_checkpointer,
        )

        with TemporaryDirectory(dir=_WORKSPACE_TMP_DIR) as directory:
            database_path = Path(directory) / "workflow.sqlite3"
            with open_sqlite_checkpointer(database_path) as checkpointer:
                graph = _build_checkpoint_test_graph(checkpointer)
                first = {"configurable": {"thread_id": "thread-a"}}
                second = {"configurable": {"thread_id": "thread-b"}}
                graph.invoke({"payload": "A"}, config=first)
                graph.invoke({"payload": "B"}, config=second)

                close_session(checkpointer, "thread-a")

                self.assertIsNone(checkpointer.get_tuple(first))
                self.assertIsNotNone(checkpointer.get_tuple(second))
                self.assertEqual(graph.get_state(second).values["payload"], "B")

    def test_close_session_rejects_an_empty_thread_id(self):
        from src.infrastructure.checkpointing.sqlite import close_session

        checkpointer = Mock()

        with self.assertRaisesRegex(ValueError, "thread_id"):
            close_session(checkpointer, "   ")

        checkpointer.delete_thread.assert_not_called()


class SqliteSessionResumeTest(unittest.TestCase):
    """Checkpointed technology-selection sessions across SQLite lifecycle."""

    def _parse_chain(self, complete_after_answer=False):
        import json

        from langchain_core.messages import AIMessage
        from unittest.mock import MagicMock

        chain = MagicMock()

        def invoke(payload):
            answers = payload.get("clarification_answers", [])
            complete = bool(answers) if complete_after_answer else False
            content = json.dumps(
                {
                    "business_goal": "降低功耗",
                    "application_scenario": "移动芯片" if complete else "",
                    "priorities": {},
                    "hard_constraints": [],
                    "requested_candidates": ["FinFET", "GAA"],
                    "scenario_criteria": [],
                    "missing_information": [] if complete else ["application_scenario"],
                },
                ensure_ascii=False,
            )
            return AIMessage(content=content)

        chain.invoke.side_effect = invoke
        return chain

    def test_reopened_database_resumes_the_same_thread(self):
        import src.workflows.nodes.selection_requirements as requirements_module
        from langgraph.types import Command
        from unittest.mock import patch

        from src.infrastructure.checkpointing.sqlite import open_sqlite_checkpointer
        from src.workflows.technology_selection_graph import (
            build_technology_selection_graph,
        )

        chain = self._parse_chain(complete_after_answer=False)
        with TemporaryDirectory(dir=_WORKSPACE_TMP_DIR) as directory:
            database_path = Path(directory) / "session.sqlite3"
            config = {"configurable": {"thread_id": "session-t1"}}
            with open_sqlite_checkpointer(database_path) as checkpointer:
                graph = build_technology_selection_graph(checkpointer)
                with patch.object(
                    requirements_module,
                    "requirement_analysis_chain",
                    chain,
                ):
                    first = graph.invoke(
                        {"user_request": "比较FinFET和GAA"},
                        config=config,
                    )
            self.assertIn("__interrupt__", first)

            with open_sqlite_checkpointer(database_path) as reopened:
                graph = build_technology_selection_graph(reopened)
                with patch.object(
                    requirements_module,
                    "requirement_analysis_chain",
                    chain,
                ):
                    resumed = graph.invoke(
                        Command(resume="应用于手机SoC"),
                        config=config,
                    )
                    values = graph.get_state(config).values

        self.assertIn("__interrupt__", resumed)
        self.assertEqual(values["clarification_answers"], ["应用于手机SoC"])

    def test_two_thread_ids_are_isolated(self):
        import src.workflows.nodes.selection_requirements as requirements_module
        from langgraph.types import Command
        from unittest.mock import patch

        from src.infrastructure.checkpointing.sqlite import open_sqlite_checkpointer
        from src.workflows.technology_selection_graph import (
            build_technology_selection_graph,
        )

        chain = self._parse_chain(complete_after_answer=False)
        with TemporaryDirectory(dir=_WORKSPACE_TMP_DIR) as directory:
            database_path = Path(directory) / "isolated.sqlite3"
            with open_sqlite_checkpointer(database_path) as checkpointer:
                graph = build_technology_selection_graph(checkpointer)
                config_a = {"configurable": {"thread_id": "iso-a"}}
                config_b = {"configurable": {"thread_id": "iso-b"}}
                with patch.object(
                    requirements_module,
                    "requirement_analysis_chain",
                    chain,
                ):
                    graph.invoke(
                        {"user_request": "A的选型需求"},
                        config=config_a,
                    )
                    graph.invoke(
                        {"user_request": "B的选型需求"},
                        config=config_b,
                    )
                    self.assertEqual(
                        graph.get_state(config_a).values["user_request"],
                        "A的选型需求",
                    )
                    graph.invoke(
                        Command(resume="A的回答"),
                        config=config_a,
                    )
                    values_a = graph.get_state(config_a).values
                    values_b = graph.get_state(config_b).values

        self.assertEqual(values_a["clarification_answers"], ["A的回答"])
        self.assertEqual(values_b.get("clarification_answers", []), [])
        self.assertEqual(values_b["user_request"], "B的选型需求")

    def test_close_session_on_one_thread_does_not_affect_another(self):
        import src.workflows.nodes.selection_requirements as requirements_module
        from langgraph.types import Command
        from unittest.mock import patch

        from src.infrastructure.checkpointing.sqlite import (
            close_session,
            open_sqlite_checkpointer,
        )
        from src.workflows.technology_selection_graph import (
            build_technology_selection_graph,
        )

        chain = self._parse_chain(complete_after_answer=False)
        with TemporaryDirectory(dir=_WORKSPACE_TMP_DIR) as directory:
            database_path = Path(directory) / "isolated.sqlite3"
            with open_sqlite_checkpointer(database_path) as checkpointer:
                graph = build_technology_selection_graph(checkpointer)
                config_a = {"configurable": {"thread_id": "iso-a"}}
                config_b = {"configurable": {"thread_id": "iso-b"}}
                with patch.object(
                    requirements_module,
                    "requirement_analysis_chain",
                    chain,
                ):
                    graph.invoke(
                        {"user_request": "A的选型需求"},
                        config=config_a,
                    )
                    graph.invoke(
                        {"user_request": "B的选型需求"},
                        config=config_b,
                    )
                    close_session(checkpointer, "iso-a")
                    self.assertIsNone(checkpointer.get_tuple(config_a))
                    self.assertIsNotNone(checkpointer.get_tuple(config_b))
                    resumed_b = graph.invoke(
                        Command(resume="B的回答"),
                        config=config_b,
                    )
                    values_b = graph.get_state(config_b).values

        self.assertEqual(values_b["clarification_answers"], ["B的回答"])
        self.assertIn("__interrupt__", resumed_b)


if __name__ == "__main__":
    unittest.main()

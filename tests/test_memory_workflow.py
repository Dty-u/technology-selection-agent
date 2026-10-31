import unittest
from unittest.mock import AsyncMock, MagicMock, patch


class _SemanticReader:
    def __init__(self, results=()):
        self.results = list(results)
        self.calls = []

    def search_hits(self, **kwargs):
        self.calls.append(kwargs)
        return list(self.results)


class _RecallRepository:
    """Expose controlled exact results while keeping real ID hydration."""

    def __init__(self, repository, exact_results=()):
        self.repository = repository
        self.exact_results = list(exact_results)

    def search_authoritative(self, **kwargs):
        if tuple(kwargs.get("statuses", ())) == ("approved",):
            return []
        return list(self.exact_results)

    def get_by_ids(self, memory_ids):
        return self.repository.get_by_ids(memory_ids)


class MemoryWorkflowNodeTest(unittest.TestCase):
    def _config(self, **overrides):
        configurable = {
            "thread_id": "thread-a",
            "user_id": "user-a",
            "request_id": "request-a",
        }
        configurable.update(overrides)
        return {"configurable": configurable}

    def test_rag_recall_combines_exact_then_semantic_without_treating_as_evidence(self):
        from src.infrastructure.memory.models import MemoryCandidate, SemanticMemoryHit
        from src.infrastructure.memory.repository import InMemoryMemoryRepository
        from src.workflows.nodes.memory import make_recall_memory

        repository = InMemoryMemoryRepository()
        exact = repository.save_candidate(
            MemoryCandidate(
                idempotency_key="pref-1",
                user_id="user-a",
                thread_id="thread-a",
                memory_type="user_preference",
                content="用户偏好中文回答",
                status="confirmed",
                source_kind="explicit_user_statement",
                confidence=1,
            )
        )
        semantic = repository.save_candidate(
            MemoryCandidate(
                idempotency_key="fact-1",
                user_id="user-a",
                thread_id="thread-a",
                memory_type="user_fact",
                content="用户正在学习LangGraph",
                status="confirmed",
                source_kind="explicit_user_statement",
                confidence=1,
            )
        )
        reader = _SemanticReader(
            [
                SemanticMemoryHit(
                    memory_id=semantic.memory_id,
                    score=0.92,
                    version=semantic.version,
                )
            ]
        )

        result = make_recall_memory(_RecallRepository(repository, [exact]), reader)(
            {"query": "解释checkpoint", "query_type": "rag_qa"},
            self._config(),
        )

        self.assertEqual(
            result["recalled_memory_ids"],
            [exact.memory_id, semantic.memory_id],
        )
        self.assertIn("用户偏好中文回答", result["memory_context"])
        self.assertIn("用户正在学习LangGraph", result["memory_context"])
        self.assertIn("不是知识库证据", result["memory_context"])
        self.assertEqual(reader.calls[0]["user_id"], "user-a")
        self.assertIsNone(reader.calls[0]["project_id"])

    def test_semantic_recall_rejects_stale_or_out_of_scope_hits(self):
        from src.infrastructure.memory.models import MemoryCandidate, SemanticMemoryHit
        from src.infrastructure.memory.repository import InMemoryMemoryRepository
        from src.workflows.nodes.memory import make_recall_memory

        repository = InMemoryMemoryRepository()
        current = repository.save_candidate(
            MemoryCandidate(
                idempotency_key="fact-current",
                user_id="user-a",
                thread_id="thread-a",
                memory_type="user_fact",
                content="Current authoritative content",
                status="confirmed",
                source_kind="explicit_user_statement",
                confidence=1,
            )
        )
        other_user = repository.save_candidate(
            MemoryCandidate(
                idempotency_key="fact-other",
                user_id="user-b",
                thread_id="thread-b",
                memory_type="user_fact",
                content="Must never leak",
                status="confirmed",
                source_kind="explicit_user_statement",
                confidence=1,
            )
        )
        reader = _SemanticReader(
            [
                SemanticMemoryHit(
                    memory_id=current.memory_id,
                    score=0.99,
                    version=current.version + 1,
                ),
                SemanticMemoryHit(
                    memory_id=other_user.memory_id,
                    score=0.98,
                    version=other_user.version,
                ),
            ]
        )

        result = make_recall_memory(_RecallRepository(repository), reader)(
            {"query": "facts", "query_type": "rag_qa"},
            self._config(),
        )

        self.assertNotIn(current.memory_id, result["recalled_memory_ids"])
        self.assertNotIn(other_user.memory_id, result["recalled_memory_ids"])
        self.assertNotIn("Must never leak", result["memory_context"])

    def test_selection_recall_requires_explicit_project_id(self):
        from src.infrastructure.memory.repository import InMemoryMemoryRepository
        from src.workflows.nodes.memory import make_recall_memory

        node = make_recall_memory(InMemoryMemoryRepository())
        result = node(
            {"query": "为手机SoC选择工艺", "query_type": "technology_selection"},
            self._config(),
        )

        self.assertEqual(result["status"], "error")
        self.assertIn("project_id", result["warnings"][0])

    def test_explicit_remember_request_becomes_confirmed_candidate(self):
        from src.workflows.nodes.memory import stage_memory_candidates

        result = stage_memory_candidates(
            {
                "query": "请记住，我偏好使用中文回答",
                "query_type": "rag_qa",
                "status": "completed",
            },
            self._config(),
        )

        candidate = result["memory_candidates"][0]
        self.assertEqual(candidate["status"], "confirmed")
        self.assertEqual(candidate["memory_type"], "user_preference")
        self.assertEqual(candidate["user_id"], "user-a")
        self.assertEqual(candidate["idempotency_key"], "request-a:user_preference:0")

    def test_selection_result_is_pending_until_explicit_approval(self):
        from src.workflows.nodes.memory import stage_memory_candidates

        result = stage_memory_candidates(
            {
                "query": "为手机SoC选择FinFET或GAA",
                "query_type": "technology_selection",
                "status": "completed",
                "report_data": {
                    "recommendation": {
                        "decision": "recommend",
                        "recommended_candidate_id": "gaa",
                    }
                },
            },
            self._config(project_id="project-a"),
        )

        candidate = result["memory_candidates"][0]
        self.assertEqual(candidate["memory_type"], "selection_case")
        self.assertEqual(candidate["status"], "pending")
        self.assertEqual(candidate["project_id"], "project-a")

    def test_persist_node_writes_each_candidate_through_repository(self):
        from src.infrastructure.memory.repository import InMemoryMemoryRepository
        from src.workflows.nodes.memory import make_persist_memory_candidates

        repository = InMemoryMemoryRepository()
        node = make_persist_memory_candidates(repository)
        staged = {
            "memory_candidates": [
                {
                    "idempotency_key": "request-a:user_fact:0",
                    "user_id": "user-a",
                    "thread_id": "thread-a",
                    "project_id": None,
                    "memory_type": "user_fact",
                    "content": "用户正在学习LangGraph",
                    "status": "confirmed",
                    "source_kind": "explicit_user_statement",
                    "confidence": 1.0,
                    "metadata": {},
                }
            ],
            "warnings": [],
        }

        result = node(staged)

        self.assertEqual(result["memory_write_count"], 1)
        self.assertEqual(len(repository.items), 1)
        self.assertEqual(len(repository.outbox_events), 1)


class MemoryAwareGraphTest(unittest.TestCase):
    def _config(self, **overrides):
        configurable = {
            "thread_id": "thread-a",
            "user_id": "user-a",
            "request_id": "request-a",
        }
        configurable.update(overrides)
        return {"configurable": configurable}

    def test_memory_enabled_graph_places_recall_before_child_and_write_after_response(self):
        from src.infrastructure.memory.repository import InMemoryMemoryRepository
        from src.workflows.query_router_graph import build_query_router_graph

        graph = build_query_router_graph(
            memory_repository=InMemoryMemoryRepository()
        )
        node_names = set(graph.get_graph().nodes)

        self.assertTrue(
            {
                "prepare_short_term_memory",
                "finalize_short_term_memory",
                "recall_memory",
                "stage_memory_candidates",
                "persist_memory_candidates",
            }.issubset(node_names)
        )

    def test_checkpoint_messages_supply_the_next_turn_context_verbatim(self):
        from langchain_core.messages import AIMessage
        from langgraph.checkpoint.memory import InMemorySaver

        import src.workflows.nodes.query_router as router_module
        from src.workflows.query_router_graph import build_query_router_graph

        graph = build_query_router_graph(InMemorySaver())
        direct_chain = MagicMock()
        direct_chain.invoke.side_effect = [
            AIMessage(content="第一轮完整回答"),
            AIMessage(content="第二轮完整回答"),
        ]

        with patch.object(router_module, "direct_response_chain", direct_chain):
            graph.invoke(
                {"query": "你好"},
                config=self._config(request_id="request-1"),
            )
            result = graph.invoke(
                {"query": "你是谁"},
                config=self._config(request_id="request-2"),
            )

        values = graph.get_state(self._config(request_id="request-2")).values
        self.assertEqual(
            [(message.type, message.content) for message in values.get("messages", [])],
            [
                ("human", "你好"),
                ("ai", "第一轮完整回答"),
                ("human", "你是谁"),
                ("ai", "第二轮完整回答"),
            ],
        )
        self.assertEqual(
            values.get("short_term_context"),
            "[用户] 你好\n[助手] 第一轮完整回答",
        )
        self.assertEqual(direct_chain.invoke.call_count, 2)
        self.assertNotIn("messages", result)

    def test_checkpoint_keeps_only_six_turns_and_same_request_is_idempotent(self):
        from langchain_core.messages import AIMessage
        from langgraph.checkpoint.memory import InMemorySaver

        import src.workflows.nodes.query_router as router_module
        from src.workflows.query_router_graph import build_query_router_graph

        graph = build_query_router_graph(InMemorySaver())
        questions = [
            "你好",
            "您好",
            "嗨",
            "哈喽",
            "hello",
            "你是谁",
            "你能做什么",
        ]
        answers = [
            router_module._DIRECT_RESPONSE_TEMPLATES["greeting"],
            router_module._DIRECT_RESPONSE_TEMPLATES["greeting"],
            router_module._DIRECT_RESPONSE_TEMPLATES["greeting"],
            router_module._DIRECT_RESPONSE_TEMPLATES["greeting"],
            router_module._DIRECT_RESPONSE_TEMPLATES["greeting"],
            router_module._DIRECT_RESPONSE_TEMPLATES["identity"],
            router_module._DIRECT_RESPONSE_TEMPLATES["capabilities"],
        ]
        direct_chain = MagicMock()
        direct_chain.invoke.side_effect = [
            *(AIMessage(content=answer) for answer in answers),
            AIMessage(content=answers[-1]),
        ]

        with patch.object(router_module, "direct_response_chain", direct_chain):
            for turn, question in enumerate(questions, start=1):
                graph.invoke(
                    {"query": question},
                    config=self._config(request_id=f"request-{turn}"),
                )
            graph.invoke(
                {"query": questions[-1]},
                config=self._config(request_id="request-7"),
            )

        values = graph.get_state(self._config(request_id="request-7")).values
        self.assertEqual(
            [message.content for message in values.get("messages", [])],
            [
                item
                for pair in zip(questions[1:], answers[1:])
                for item in pair
            ],
        )

    def test_same_thread_rejects_a_different_user_without_exposing_messages(self):
        from langchain_core.messages import AIMessage
        from langgraph.checkpoint.memory import InMemorySaver

        import src.workflows.nodes.query_router as router_module
        from src.workflows.query_router_graph import build_query_router_graph

        graph = build_query_router_graph(InMemorySaver())
        direct_chain = MagicMock()
        direct_chain.invoke.return_value = AIMessage(content="仅属于用户A的回答")

        with patch.object(router_module, "direct_response_chain", direct_chain):
            graph.invoke(
                {"query": "你好"},
                config=self._config(request_id="request-a"),
            )
            rejected = graph.invoke(
                {"query": "你是谁"},
                config=self._config(
                    user_id="user-b",
                    request_id="request-b",
                ),
            )

        self.assertEqual(rejected["status"], "error")
        self.assertEqual(rejected["answer"], "")
        self.assertTrue(any("thread_id" in warning for warning in rejected["warnings"]))
        self.assertNotIn("messages", rejected)
        direct_chain.invoke.assert_called_once_with(
            {
                "direct_response_kind": "greeting",
                "query": "你好",
                "short_term_context": "（无）",
            }
        )

    def test_same_thread_rejects_a_different_project_without_exposing_messages(self):
        from langchain_core.messages import AIMessage
        from langgraph.checkpoint.memory import InMemorySaver

        import src.workflows.nodes.query_router as router_module
        from src.workflows.query_router_graph import build_query_router_graph

        graph = build_query_router_graph(InMemorySaver())
        direct_chain = MagicMock()
        direct_chain.invoke.return_value = AIMessage(content="project-a response")

        with patch.object(router_module, "direct_response_chain", direct_chain):
            graph.invoke(
                {"query": "hello"},
                config=self._config(
                    project_id="project-a",
                    request_id="project-request-a",
                ),
            )
            rejected = graph.invoke(
                {"query": "who are you"},
                config=self._config(
                    project_id="project-b",
                    request_id="project-request-b",
                ),
            )

        self.assertEqual(rejected["status"], "error")
        self.assertEqual(rejected["answer"], "")
        self.assertTrue(
            any("project_id" in warning for warning in rejected["warnings"])
        )
        self.assertNotIn("messages", rejected)
        direct_chain.invoke.assert_called_once()

    def test_runner_merges_resume_message_without_overwriting_command_fields(self):
        from langchain_core.messages import AIMessage
        from langgraph.errors import GraphInterrupt
        from langgraph.types import Command

        from src.workflows.memory_runner import MemoryAwareRunner

        graph = MagicMock()
        graph.invoke.side_effect = GraphInterrupt("needs clarification")
        runner = MemoryAwareRunner(graph=graph, repository=MagicMock())
        original = Command(
            resume="应用于手机SoC",
            update={
                "marker": "keep-me",
                "messages": [AIMessage(content="已有更新", id="existing")],
            },
            goto=("resume-node",),
            graph=Command.PARENT,
        )

        with self.assertRaises(GraphInterrupt):
            runner.invoke(
                original,
                config=self._config(request_id="clarification-1"),
            )

        forwarded = graph.invoke.call_args.args[0]
        self.assertIsInstance(forwarded, Command)
        self.assertEqual(forwarded.resume, original.resume)
        self.assertEqual(forwarded.goto, original.goto)
        self.assertEqual(forwarded.graph, original.graph)
        self.assertEqual(forwarded.update["marker"], "keep-me")
        self.assertEqual(
            [(message.type, message.content) for message in forwarded.update["messages"]],
            [("ai", "已有更新"), ("human", "应用于手机SoC")],
        )

    def test_runner_rejects_resume_when_checkpoint_belongs_to_another_user(self):
        from types import SimpleNamespace

        from langgraph.types import Command

        from src.workflows.memory_runner import MemoryAwareRunner

        graph = MagicMock()
        graph.get_state.return_value = SimpleNamespace(
            values={"session_user_id": "user-a"}
        )
        runner = MemoryAwareRunner(graph=graph)

        with self.assertRaisesRegex(PermissionError, "thread_id"):
            runner.invoke(
                Command(resume="窃取恢复"),
                config=self._config(
                    user_id="user-b",
                    request_id="unauthorized-resume",
                ),
            )

        graph.invoke.assert_not_called()

    def test_runner_rejects_resume_when_checkpoint_belongs_to_another_project(self):
        from types import SimpleNamespace

        from langgraph.types import Command

        from src.workflows.memory_runner import MemoryAwareRunner

        graph = MagicMock()
        graph.get_state.return_value = SimpleNamespace(
            values={
                "session_user_id": "user-a",
                "session_project_id": "project-a",
            }
        )
        runner = MemoryAwareRunner(graph=graph)

        with self.assertRaisesRegex(PermissionError, "project_id"):
            runner.invoke(
                Command(resume="resume another project"),
                config=self._config(
                    project_id="project-b",
                    request_id="unauthorized-project-resume",
                ),
            )

        graph.invoke.assert_not_called()

    def test_runner_records_interrupt_questions_and_trims_checkpoint_messages(self):
        from types import SimpleNamespace

        from langchain_core.messages import AIMessage, HumanMessage

        from src.workflows.memory_runner import MemoryAwareRunner

        existing_messages = []
        for turn in range(1, 7):
            existing_messages.extend(
                [
                    HumanMessage(content=f"问题{turn}", id=f"h-{turn}"),
                    AIMessage(content=f"回答{turn}", id=f"a-{turn}"),
                ]
            )
        graph = MagicMock()
        graph.invoke.return_value = {
            "__interrupt__": (
                SimpleNamespace(
                    value={
                        "status": "awaiting_user",
                        "clarification_questions": ["应用场景是什么？", "功耗优先吗？"],
                    }
                ),
            )
        }
        graph.get_state.return_value = SimpleNamespace(
            values={"messages": existing_messages}
        )
        runner = MemoryAwareRunner(graph=graph, repository=MagicMock())

        runner.invoke(
            {"query": "为芯片选择FinFET或GAA"},
            config=self._config(request_id="selection-1"),
        )

        graph.update_state.assert_called_once()
        update = graph.update_state.call_args.args[1]
        updates = update["messages"]
        self.assertTrue(
            any(
                message.type == "ai"
                and message.content == "应用场景是什么？\n功耗优先吗？"
                for message in updates
            )
        )
        self.assertTrue(any(message.type == "remove" for message in updates))

    def test_runner_close_only_deletes_the_checkpoint_thread(self):
        from src.workflows.memory_runner import MemoryAwareRunner

        repository = MagicMock()
        checkpointer = MagicMock()
        runner = MemoryAwareRunner(
            graph=MagicMock(),
            repository=repository,
            checkpointer=checkpointer,
        )

        runner.close_session("thread-a", user_id="user-a")

        self.assertEqual(repository.method_calls, [])
        checkpointer.delete_thread.assert_called_once_with("thread-a")

    def test_direct_response_bypasses_long_term_memory_writes(self):
        from langchain_core.messages import AIMessage

        import src.workflows.nodes.query_router as router_module
        from src.infrastructure.memory.repository import InMemoryMemoryRepository
        from src.workflows.query_router_graph import build_query_router_graph

        repository = InMemoryMemoryRepository()
        graph = build_query_router_graph(memory_repository=repository)
        direct_chain = MagicMock()
        direct_chain.invoke.return_value = AIMessage(
            content="你好，我可以协助半导体知识问答与技术选型。"
        )

        with patch.object(router_module, "direct_response_chain", direct_chain):
            result = graph.invoke({"query": "你好"}, config=self._config())

        self.assertEqual(result["query_type"], "direct_response")
        direct_chain.invoke.assert_called_once_with(
            {
                "direct_response_kind": "greeting",
                "query": "你好",
                "short_term_context": "（无）",
            }
        )
        self.assertEqual(repository.items, [])
        self.assertEqual(repository.outbox_events, [])

    def test_runner_requires_explicit_user_thread_and_request_ids(self):
        from src.infrastructure.memory.repository import InMemoryMemoryRepository
        from src.workflows.memory_runner import MemoryAwareRunner

        runner = MemoryAwareRunner(
            graph=MagicMock(),
            repository=InMemoryMemoryRepository(),
        )

        with self.assertRaisesRegex(ValueError, "user_id"):
            runner.invoke(
                {"query": "什么是GAA"},
                config={
                    "configurable": {
                        "thread_id": "thread-a",
                        "request_id": "request-a",
                    }
                },
            )

    def test_follow_up_question_uses_same_thread_raw_context_and_routes_to_rag(self):
        import json

        from langchain_core.messages import AIMessage
        from langgraph.checkpoint.memory import InMemorySaver

        import src.workflows.nodes.query_router as router_module
        from src.infrastructure.memory.repository import InMemoryMemoryRepository
        from src.workflows.memory_runner import MemoryAwareRunner
        from src.workflows.query_router_graph import build_query_router_graph

        repository = InMemoryMemoryRepository()
        graph = build_query_router_graph(
            InMemorySaver(),
            memory_repository=repository,
        )
        runner = MemoryAwareRunner(graph=graph, repository=repository)
        classifier = MagicMock()
        classifier.invoke.return_value = AIMessage(
            content=json.dumps(
                {
                    "query_type": "rag_qa",
                    "source_preference": "auto",
                    "response_type": "answer",
                    "reason": "半导体知识问答",
                },
                ensure_ascii=False,
            )
        )
        rag_graph = MagicMock()
        rag_graph.invoke.return_value = {
            "generation": "EUV主要用于先进制程。",
            "documents": [],
            "retrieval_source": "local_hybrid",
        }

        with patch.object(
            router_module,
            "query_classification_chain",
            classifier,
        ), patch.object(router_module, "rag_qa_graph", rag_graph):
            runner.invoke(
                {"query": "EUV光刻机是什么"},
                config=self._config(request_id="request-1"),
            )
            follow_up = runner.invoke(
                {"query": "它在哪里应用较多"},
                config=self._config(request_id="request-2"),
            )

        self.assertEqual(follow_up["query_type"], "rag_qa")
        second_payload = rag_graph.invoke.call_args_list[1].args[0]
        self.assertIn("EUV光刻机是什么", second_payload["question"])
        self.assertIn("它在哪里应用较多", second_payload["question"])

    def test_same_thread_context_repairs_colloquial_follow_up_misclassified_as_capabilities(self):
        import json

        from langchain_core.messages import AIMessage
        from langgraph.checkpoint.memory import InMemorySaver

        import src.workflows.nodes.query_router as router_module
        from src.infrastructure.memory.repository import InMemoryMemoryRepository
        from src.workflows.memory_runner import MemoryAwareRunner
        from src.workflows.query_router_graph import build_query_router_graph

        repository = InMemoryMemoryRepository()
        graph = build_query_router_graph(
            InMemorySaver(),
            memory_repository=repository,
        )
        runner = MemoryAwareRunner(graph=graph, repository=repository)
        classifier = MagicMock()
        classifier.invoke.side_effect = [
            AIMessage(
                content=json.dumps(
                    {
                        "query_type": "rag_qa",
                        "source_preference": "auto",
                        "response_type": "answer",
                        "reason": "半导体知识问答",
                    },
                    ensure_ascii=False,
                )
            ),
            AIMessage(
                content=json.dumps(
                    {
                        "query_type": "direct_response",
                        "source_preference": "none",
                        "response_type": "answer",
                        "direct_response_kind": "capabilities",
                        "reason": "误认为用户在询问助手能力",
                    },
                    ensure_ascii=False,
                )
            ),
        ]
        rag_graph = MagicMock()
        rag_graph.invoke.side_effect = [
            {
                "generation": "光刻机用于把电路图形转移到晶圆上。",
                "documents": [],
                "retrieval_source": "local_hybrid",
            },
            {
                "generation": "它用于芯片制造中的曝光和图形转移。",
                "documents": [],
                "retrieval_source": "local_hybrid",
            },
        ]

        rewriter = MagicMock()
        rewriter.invoke.return_value = AIMessage(
            content='{"standalone_query":"光刻机有什么用"}'
        )

        with patch.object(
            router_module,
            "query_classification_chain",
            classifier,
        ), patch.object(router_module, "standalone_query_chain", rewriter), patch.object(
            router_module,
            "rag_qa_graph",
            rag_graph,
        ):
            runner.invoke(
                {"query": "光刻机是什么"},
                config=self._config(request_id="request-1"),
            )
            follow_up = runner.invoke(
                {"query": "他有什么用"},
                config=self._config(request_id="request-2"),
            )

        self.assertEqual(follow_up["query_type"], "rag_qa")
        self.assertEqual(rag_graph.invoke.call_count, 2)
        rewrite_input = rewriter.invoke.call_args.args[0]
        self.assertEqual(rewrite_input["query"], "他有什么用")
        self.assertIn("光刻机是什么", rewrite_input["short_term_context"])
        second_payload = rag_graph.invoke.call_args_list[1].args[0]
        self.assertEqual(second_payload["question"], "光刻机有什么用")

    def test_current_semiconductor_question_overrides_prior_out_of_scope_turn(self):
        import json

        from langchain_core.messages import AIMessage
        from langgraph.checkpoint.memory import InMemorySaver

        import src.workflows.nodes.query_router as router_module
        from src.infrastructure.memory.repository import InMemoryMemoryRepository
        from src.workflows.memory_runner import MemoryAwareRunner
        from src.workflows.query_router_graph import build_query_router_graph

        repository = InMemoryMemoryRepository()
        graph = build_query_router_graph(
            InMemorySaver(),
            memory_repository=repository,
        )
        runner = MemoryAwareRunner(graph=graph, repository=repository)
        classifier = MagicMock()
        mistaken_direct = AIMessage(
            content=json.dumps(
                {
                    "query_type": "direct_response",
                    "source_preference": "none",
                    "response_type": "answer",
                    "direct_response_kind": "out_of_scope",
                    "reason": "误受上一轮越界问题影响",
                },
                ensure_ascii=False,
            )
        )
        classifier.invoke.side_effect = [mistaken_direct, mistaken_direct]
        direct_chain = MagicMock()
        direct_chain.invoke.return_value = AIMessage(content="超出半导体服务范围。")
        rag_graph = MagicMock()
        rag_graph.invoke.return_value = {
            "generation": "半导体是导电能力介于导体与绝缘体之间的材料。",
            "documents": [],
            "retrieval_source": "local_hybrid",
        }

        with patch.object(
            router_module,
            "query_classification_chain",
            classifier,
        ), patch.object(
            router_module,
            "direct_response_chain",
            direct_chain,
        ), patch.object(router_module, "rag_qa_graph", rag_graph):
            runner.invoke(
                {"query": "李白是谁"},
                config=self._config(request_id="request-1"),
            )
            result = runner.invoke(
                {"query": "半导体是啥"},
                config=self._config(request_id="request-2"),
            )

        self.assertEqual(result["query_type"], "rag_qa")
        self.assertEqual(rag_graph.invoke.call_count, 1)
        self.assertIn("半导体是啥", rag_graph.invoke.call_args.args[0]["question"])


class MemoryAwareRunnerAsyncTest(unittest.IsolatedAsyncioTestCase):
    async def test_astream_yields_values_and_records_interrupt_questions(self):
        from types import SimpleNamespace

        from src.workflows.memory_runner import MemoryAwareRunner

        interrupt = SimpleNamespace(
            value={
                "status": "awaiting_user",
                "clarification_questions": ["目标工艺节点是什么？"],
            }
        )

        async def stream_values(*args, **kwargs):
            yield {"query_type": "technology_selection"}
            yield {"__interrupt__": (interrupt,)}

        graph = MagicMock()
        graph.astream.side_effect = stream_values
        graph.aget_state = AsyncMock(
            return_value=SimpleNamespace(values={"messages": []})
        )
        graph.aupdate_state = AsyncMock()
        runner = MemoryAwareRunner(graph=graph)

        chunks = [
            chunk
            async for chunk in runner.astream(
                {"query": "帮我选择FinFET或GAA"},
                config={
                    "configurable": {
                        "user_id": "user-a",
                        "thread_id": "thread-a",
                        "request_id": "request-a",
                    }
                },
            )
        ]

        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[-1]["__interrupt__"], (interrupt,))
        graph.astream.assert_called_once()
        self.assertEqual(
            graph.astream.call_args.kwargs["stream_mode"],
            "values",
        )
        graph.aupdate_state.assert_awaited_once()

    async def test_astream_logs_completion_with_final_query_type(self):
        from types import SimpleNamespace

        import src.workflows.memory_runner as runner_module

        async def stream_values(*args, **kwargs):
            yield {"query_type": "rag_qa"}

        graph = MagicMock()
        graph.astream.side_effect = stream_values
        graph.aget_state = AsyncMock(return_value=SimpleNamespace(values={}))
        runner = runner_module.MemoryAwareRunner(graph=graph)

        with patch.object(runner_module, "log") as log:
            chunks = [
                chunk
                async for chunk in runner.astream(
                    {"query": "什么是FinFET"},
                    config={
                        "configurable": {
                            "user_id": "user-a",
                            "thread_id": "thread-a",
                            "request_id": "request-a",
                        }
                    },
                )
            ]

        self.assertEqual(chunks, [{"query_type": "rag_qa"}])
        message, request_id, query_type, _elapsed = log.info.call_args.args
        self.assertIn("[工作流][astream] 完成", message)
        self.assertEqual((request_id, query_type), ("request-a", "rag_qa"))

    async def test_aclose_session_deletes_thread_without_error(self):
        from src.workflows.memory_runner import MemoryAwareRunner

        checkpointer = MagicMock()
        checkpointer.adelete_thread = AsyncMock()
        runner = MemoryAwareRunner(graph=MagicMock(), checkpointer=checkpointer)

        await runner.aclose_session(" thread-a ", user_id="user-a")

        checkpointer.adelete_thread.assert_awaited_once_with("thread-a")


if __name__ == "__main__":
    unittest.main()

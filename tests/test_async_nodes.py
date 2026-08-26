import asyncio
import unittest
from unittest.mock import patch


class AsyncNodeTest(unittest.IsolatedAsyncioTestCase):
    async def test_sync_node_runs_without_blocking_event_loop(self):
        from src.workflows.async_node import async_node

        def blocking_node(state):
            import time

            time.sleep(0.05)
            return {"answer": state["query"]}

        task = asyncio.create_task(async_node(blocking_node)({"query": "ok"}))
        await asyncio.sleep(0)
        self.assertFalse(task.done())
        self.assertEqual(await task, {"answer": "ok"})

    async def test_async_router_graph_supports_astream(self):
        import src.workflows.query_router_graph as router_graph_module

        def classify(_state, _config=None):
            return {
                "query_type": "direct_response",
                "source_preference": "none",
                "response_type": "answer",
                "direct_response_kind": "greeting",
                "routing_reason": "test",
            }

        def respond(_state, _config=None):
            return {"status": "completed", "direct_answer": "你好"}

        def build_response(state, _config=None):
            return {
                "query": state["query"],
                "query_type": state["query_type"],
                "source_preference": state["source_preference"],
                "response_type": state["response_type"],
                "routing_reason": state["routing_reason"],
                "status": state["status"],
                "answer": state["direct_answer"],
                "report": "",
                "report_data": {},
                "report_format": "markdown",
                "report_path": None,
                "clarification_questions": [],
                "clarification_answers": [],
                "clarification_history": [],
                "sources": [],
                "details": {},
                "warnings": [],
            }

        with (
            patch.object(router_graph_module, "prepare_short_term_memory", return_value={}),
            patch.object(router_graph_module, "classify_query", side_effect=classify),
            patch.object(router_graph_module, "respond_directly", side_effect=respond),
            patch.object(router_graph_module, "build_query_response", side_effect=build_response),
            patch.object(router_graph_module, "finalize_short_term_memory", return_value={}),
        ):
            graph = router_graph_module.build_query_router_graph(async_nodes=True)
            states = [
                state
                async for state in graph.astream(
                    {"query": "你好"}, stream_mode="values"
                )
            ]

        self.assertEqual(states[-1]["answer"], "你好")


if __name__ == "__main__":
    unittest.main()

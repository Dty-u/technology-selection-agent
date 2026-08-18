import time
import unittest

from src.observability.logging import log


def _load_progress_api(test_case):
    try:
        from src.observability.workflow_progress import logged_node
    except ModuleNotFoundError:
        test_case.fail("缺少统一工作流进度日志封装")
    return logged_node


class WorkflowProgressTest(unittest.TestCase):
    def _capture_messages(self):
        messages = []
        sink_id = log.add(messages.append, format="{message}")
        self.addCleanup(log.remove, sink_id)
        return messages

    def test_logged_node_reports_safe_start_completion_and_duration(self):
        logged_node = _load_progress_api(self)
        messages = self._capture_messages()

        def node(_state):
            return {
                "status": "completed",
                "evidence_records": [
                    {"content": "private evidence body"},
                    {"content": "another private body"},
                ],
                "report": "private report body",
            }

        wrapped = logged_node(
            "技术选型",
            "example_node",
            "执行示例节点",
            node,
            heartbeat_seconds=None,
        )
        result = wrapped({"query": "private user query"})
        rendered = "\n".join(str(message) for message in messages)

        self.assertEqual(result["status"], "completed")
        self.assertIn("[技术选型][example_node] 开始：执行示例节点", rendered)
        self.assertIn("[技术选型][example_node] 完成：执行示例节点", rendered)
        self.assertIn("状态=completed", rendered)
        self.assertIn("证据=2", rendered)
        self.assertIn("报告字符数=19", rendered)
        self.assertIn("耗时=", rendered)
        self.assertNotIn("private user query", rendered)
        self.assertNotIn("private evidence body", rendered)
        self.assertNotIn("private report body", rendered)

    def test_logged_node_emits_repeated_heartbeats_and_stops_after_completion(self):
        logged_node = _load_progress_api(self)
        messages = self._capture_messages()

        def slow_node(_state):
            time.sleep(0.045)
            return {"status": "completed"}

        wrapped = logged_node(
            "技术选型",
            "slow_node",
            "等待模型响应",
            slow_node,
            heartbeat_seconds=0.01,
        )
        wrapped({})
        heartbeat_count = sum(
            "仍在执行：等待模型响应" in str(message)
            for message in messages
        )
        time.sleep(0.03)
        final_heartbeat_count = sum(
            "仍在执行：等待模型响应" in str(message)
            for message in messages
        )

        self.assertGreaterEqual(heartbeat_count, 2)
        self.assertEqual(final_heartbeat_count, heartbeat_count)

    def test_logged_node_reports_exception_type_without_sensitive_message(self):
        logged_node = _load_progress_api(self)
        messages = self._capture_messages()

        def failing_node(_state):
            raise RuntimeError("private provider response")

        wrapped = logged_node(
            "技术选型",
            "failing_node",
            "调用外部模型",
            failing_node,
            heartbeat_seconds=None,
        )
        with self.assertRaisesRegex(RuntimeError, "private provider response"):
            wrapped({})
        rendered = "\n".join(str(message) for message in messages)

        self.assertIn("[技术选型][failing_node] 失败：调用外部模型", rendered)
        self.assertIn("异常类型=RuntimeError", rendered)
        self.assertNotIn("private provider response", rendered)

    def test_logged_node_reports_graph_interrupt_as_a_pause_not_a_failure(self):
        from langgraph.errors import GraphInterrupt

        logged_node = _load_progress_api(self)
        messages = self._capture_messages()

        def pausing_node(_state):
            raise GraphInterrupt("awaiting user")

        wrapped = logged_node(
            "技术选型",
            "pausing_node",
            "等待用户回答",
            pausing_node,
            heartbeat_seconds=None,
        )
        with self.assertRaises(GraphInterrupt):
            wrapped({})
        rendered = "\n".join(str(message) for message in messages)

        self.assertIn("[技术选型][pausing_node] 暂停：等待用户回答", rendered)
        self.assertIn("等待用户输入/暂停", rendered)
        self.assertNotIn("[技术选型][pausing_node] 失败", rendered)
        self.assertNotIn("异常类型=GraphInterrupt", rendered)


if __name__ == "__main__":
    unittest.main()

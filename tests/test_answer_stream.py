import unittest
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.prompts import ChatPromptTemplate


class AnswerStreamChannelTest(unittest.TestCase):
    def test_events_reach_only_the_active_observer(self):
        from src.observability.answer_stream import (
            emit_answer_delta,
            emit_answer_reset,
            has_answer_observer,
            observe_answer_stream,
        )

        events = []
        emit_answer_delta("ignored")
        self.assertFalse(has_answer_observer())
        with observe_answer_stream(events.append):
            self.assertTrue(has_answer_observer())
            emit_answer_reset()
            emit_answer_delta("硅")
            emit_answer_delta("")
        emit_answer_delta("ignored")

        self.assertEqual(events, [{"type": "reset"}, {"type": "delta", "text": "硅"}])
        self.assertFalse(has_answer_observer())

    def test_failing_observer_does_not_break_the_chain(self):
        from src.observability.answer_stream import (
            observe_answer_stream,
            stream_chain_text,
        )

        chain = MagicMock()
        chain.stream.return_value = iter(["a", "b"])
        observer = MagicMock(side_effect=RuntimeError("client gone"))

        with observe_answer_stream(observer):
            text = stream_chain_text(chain, {"q": 1})

        self.assertEqual(text, "ab")
        chain.invoke.assert_not_called()


class GenerateStreamingTest(unittest.TestCase):
    @staticmethod
    def _state():
        return {
            "question": "什么是FinFET？",
            "documents": [Document(page_content="FinFET 是鳍式场效应晶体管。")],
            "generation_attempt_count": 0,
        }

    def test_generate_streams_reset_then_deltas_matching_generation(self):
        from src.observability.answer_stream import observe_answer_stream
        import src.workflows.nodes.generate as generate_module

        events = []
        fake_llm = FakeListChatModel(responses=["鳍式晶体管"])
        with patch.object(generate_module, "llm", fake_llm):
            with observe_answer_stream(events.append):
                result = generate_module.generate(self._state())

        self.assertEqual(result["generation"], "鳍式晶体管")
        self.assertEqual(result["generation_attempt_count"], 1)
        self.assertEqual(events[0], {"type": "reset"})
        deltas = [event["text"] for event in events[1:]]
        self.assertGreater(len(deltas), 1)
        self.assertEqual("".join(deltas), "鳍式晶体管")

    def test_generate_without_observer_keeps_invoke_path(self):
        import src.workflows.nodes.generate as generate_module

        fake_llm = FakeListChatModel(responses=["答案"])
        with patch.object(generate_module, "llm", fake_llm), patch.object(
            FakeListChatModel,
            "stream",
            side_effect=AssertionError("stream must not be used"),
        ):
            result = generate_module.generate(self._state())

        self.assertEqual(result["generation"], "答案")


class DirectResponseStreamingTest(unittest.TestCase):
    def test_direct_response_streams_model_text(self):
        from src.observability.answer_stream import observe_answer_stream
        import src.workflows.nodes.query_router as router_module

        events = []
        with patch.object(
            router_module,
            "direct_response_chain",
            ChatPromptTemplate.from_messages([("human", "{query}")])
            | FakeListChatModel(responses=["你好！"]),
        ):
            with observe_answer_stream(events.append):
                result = router_module.respond_directly(
                    {"query": "你好", "direct_response_kind": "greeting"}
                )

        self.assertEqual(result["direct_answer"], "你好！")
        self.assertEqual(events[0], {"type": "reset"})
        self.assertEqual(
            "".join(event.get("text", "") for event in events),
            "你好！",
        )

    def test_direct_response_failure_streams_the_template(self):
        from src.observability.answer_stream import observe_answer_stream
        import src.workflows.nodes.query_router as router_module

        chain = MagicMock()
        chain.stream.side_effect = RuntimeError("provider down")
        events = []
        with patch.object(router_module, "direct_response_chain", chain):
            with observe_answer_stream(events.append):
                result = router_module.respond_directly(
                    {"query": "你好", "direct_response_kind": "greeting"}
                )

        last_reset = max(
            index for index, event in enumerate(events) if event["type"] == "reset"
        )
        streamed = "".join(event.get("text", "") for event in events[last_reset:])
        self.assertEqual(streamed, result["direct_answer"])


if __name__ == "__main__":
    unittest.main()

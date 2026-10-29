"""Compiled workflow for ordinary retrieval-augmented question answering."""

from pprint import pprint

from langgraph.constants import END
from langgraph.graph import StateGraph

from src.workflows.local_retrieval import add_local_retrieval, make_timed_node
from src.workflows.nodes.assess_generation import assess_generation
from src.workflows.nodes.generate import generate
from src.workflows.routes.decide_to_generate import decide_to_generate
from src.workflows.routes.evaluate_generation import (
    grade_generation_v_documents_and_question,
)
from src.workflows.routes.rag_qa_entry import route_rag_qa_entry
from src.workflows.rag_qa_state import GraphState


_timed_node = make_timed_node("普通RAG")

workflow = StateGraph(GraphState)
add_local_retrieval(workflow, _timed_node, entry_route=route_rag_qa_entry)

workflow.add_node(
    "generate",
    _timed_node("generate", "生成回答", generate, model_calls=1),
)
workflow.add_node(
    "assess_generation",
    _timed_node(
        "assess_generation", "评估回答", assess_generation, model_calls=1
    ),
)

workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "generate": "generate",
        "transform_query": "transform_query",
        "web_search": "web_search",
    },
)
workflow.add_edge("web_search", "generate")
workflow.add_edge("generate", "assess_generation")
workflow.add_conditional_edges(
    "assess_generation",
    grade_generation_v_documents_and_question,
    {
        "useful": END,
        "generate": "generate",
        "transform_query": "transform_query",
        "web_search": "web_search",
        "end": END,
    },
)

graph = workflow.compile()


if __name__ == "__main__":
    while True:
        question = input("用户：")
        if question.lower() in {"q", "exit", "quit"}:
            print("对话结束。")
            break

        inputs = {
            "question": question,
            "transform_count": 0,
            "generation_attempt_count": 0,
        }
        final_state = None
        for output in graph.stream(inputs):
            for node_name, state in output.items():
                pprint(f"Node '{node_name}':")
                final_state = state
            pprint("\n---\n")
        if final_state and "generation" in final_state:
            pprint(final_state["generation"])

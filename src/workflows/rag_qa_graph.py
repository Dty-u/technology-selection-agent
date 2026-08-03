"""Compiled workflow for ordinary retrieval-augmented question answering."""

from pprint import pprint

from langgraph.constants import END, START
from langgraph.graph import StateGraph

from src.workflows.nodes.analyze_question import (
    analyze_question,
    extract_retrieval_intent,
)
from src.workflows.nodes.generate import generate
from src.workflows.nodes.grade_documents import grade_documents
from src.workflows.nodes.graph_retrieve import graph_retrieve
from src.workflows.nodes.merge_retrieval_results import merge_retrieval_results
from src.workflows.nodes.retrieve import retrieve
from src.workflows.nodes.transform_query import transform_query
from src.workflows.nodes.web_search import web_search
from src.workflows.routes.decide_to_generate import decide_to_generate
from src.workflows.routes.evaluate_generation import (
    grade_generation_v_documents_and_question,
)
from src.workflows.routes.rag_qa_entry import route_rag_qa_entry
from src.workflows.routes.route_after_analysis import route_after_analysis
from src.workflows.rag_qa_state import GraphState


workflow = StateGraph(GraphState)

workflow.add_node("analyze_question", analyze_question)
workflow.add_node("extract_retrieval_intent", extract_retrieval_intent)
workflow.add_node("web_search", web_search)
workflow.add_node("vector_retrieve", retrieve)
workflow.add_node("graph_retrieve", graph_retrieve)
workflow.add_node("merge_retrieval_results", merge_retrieval_results)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate", generate)
workflow.add_node("transform_query", transform_query)

workflow.add_conditional_edges(
    START,
    route_rag_qa_entry,
    {
        "analyze_question": "analyze_question",
        "web_search": "web_search",
    },
)
workflow.add_conditional_edges(
    "analyze_question",
    route_after_analysis,
    {
        "web_search": "web_search",
        "vector_retrieve": "vector_retrieve",
        "graph_retrieve": "graph_retrieve",
    },
)

workflow.add_edge(
    ["vector_retrieve", "graph_retrieve"],
    "merge_retrieval_results",
)
workflow.add_edge("merge_retrieval_results", "grade_documents")
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
workflow.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question,
    {
        "useful": END,
        "generate": "generate",
        "transform_query": "transform_query",
        "web_search": "web_search",
        "end": END,
    },
)
workflow.add_edge("transform_query", "extract_retrieval_intent")
workflow.add_edge("extract_retrieval_intent", "vector_retrieve")
workflow.add_edge("extract_retrieval_intent", "graph_retrieve")

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

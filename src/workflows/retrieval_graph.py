from langgraph.constants import END, START
from langgraph.graph import StateGraph

from src.workflows.nodes.analyze_question import (
    analyze_question,
    extract_retrieval_intent,
)
from src.workflows.nodes.grade_documents import grade_documents
from src.workflows.nodes.graph_retrieve import graph_retrieve
from src.workflows.nodes.merge_retrieval_results import merge_retrieval_results
from src.workflows.nodes.retrieve import retrieve
from src.workflows.nodes.transform_query import transform_query
from src.workflows.nodes.web_search import web_search
from src.workflows.routes.route_after_analysis import route_after_analysis
from src.workflows.routes.technology_selection import (
    route_retrieval_entry,
    route_retrieval_result,
)
from src.workflows.technology_selection_state import EvidenceRetrievalState


workflow = StateGraph(EvidenceRetrievalState)

workflow.add_node("analyze_question", analyze_question)
workflow.add_node("extract_retrieval_intent", extract_retrieval_intent)
workflow.add_node("web_search", web_search)
workflow.add_node("vector_retrieve", retrieve)
workflow.add_node("graph_retrieve", graph_retrieve)
workflow.add_node("merge_retrieval_results", merge_retrieval_results)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("transform_query", transform_query)

workflow.add_conditional_edges(
    START,
    route_retrieval_entry,
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
    route_retrieval_result,
    {
        END: END,
        "transform_query": "transform_query",
        "web_search": "web_search",
    },
)
workflow.add_edge("transform_query", "extract_retrieval_intent")
workflow.add_edge("extract_retrieval_intent", "vector_retrieve")
workflow.add_edge("extract_retrieval_intent", "graph_retrieve")
workflow.add_edge("web_search", END)

retrieval_graph = workflow.compile()

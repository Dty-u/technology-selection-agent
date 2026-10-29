from langgraph.constants import END
from langgraph.graph import StateGraph

from src.workflows.local_retrieval import add_local_retrieval, make_timed_node
from src.workflows.routes.technology_selection import (
    route_retrieval_entry,
    route_retrieval_result,
)
from src.workflows.technology_selection_state import EvidenceRetrievalState


_timed_retrieval_node = make_timed_node("证据检索")

workflow = StateGraph(EvidenceRetrievalState)
add_local_retrieval(
    workflow,
    _timed_retrieval_node,
    entry_route=route_retrieval_entry,
)

workflow.add_conditional_edges(
    "grade_documents",
    route_retrieval_result,
    {
        END: END,
        "transform_query": "transform_query",
        "web_search": "web_search",
    },
)
workflow.add_edge("web_search", END)

retrieval_graph = workflow.compile()

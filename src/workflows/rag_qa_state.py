from typing import List, Literal, Optional, TypedDict

from langchain_core.documents import Document


class GraphState(TypedDict, total=False):
    """Shared state for routing, hybrid retrieval, and answer evaluation."""

    question: str
    original_question: str
    transform_count: int
    generation: str
    documents: List[Document]
    route: Literal["local_hybrid", "web_search"]
    query_entities: List[str]
    relation_intent: str
    relation_types: List[str]
    vector_documents: List[Document]
    graph_documents: List[Document]
    retrieval_source: Literal["local_hybrid", "web_search"]
    graph_error: Optional[str]
    vector_error: Optional[str]
    retrieval_error: Optional[str]
    generation_attempt_count: int
    web_search_query: str
    web_search_queries: List[str]
    web_search_attempt_count: int
    web_relevant_result_count: int
    source_preference: Literal["auto", "web"]
    response_type: Literal["answer", "report"]
    memory_context: str
    short_term_context: str
    analysis_ready: bool

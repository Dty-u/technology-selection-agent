"""State and validated decisions for the single-query router workflow."""

from typing import Annotated, Any, Dict, List, Literal, NotRequired, Optional, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

from pydantic import BaseModel, Field, model_validator


class RetrievalAnalysis(BaseModel):
    """Internal retrieval fields optionally produced by the entry classifier."""

    route: Literal["local_hybrid", "web_search"]
    query_entities: List[str] = Field(default_factory=list)
    relation_intent: str = ""
    relation_types: List[str] = Field(default_factory=list)


class QueryRoutingDecision(BaseModel):
    """Validated semantic routing decision for one user query."""

    query_type: Literal[
        "direct_response",
        "rag_qa",
        "technology_selection",
    ]
    source_preference: Literal["none", "auto", "web"]
    response_type: Literal["answer", "report"]
    direct_response_kind: Optional[
        Literal["identity", "greeting", "capabilities", "out_of_scope"]
    ] = None
    reason: str
    retrieval_analysis: Optional[RetrievalAnalysis] = None

    @model_validator(mode="after")
    def validate_direct_response_contract(self):
        """Keep direct responses isolated from retrieval source choices."""
        if self.query_type == "direct_response":
            if self.direct_response_kind is None:
                raise ValueError("direct_response requires a response kind")
            if self.source_preference != "none":
                raise ValueError("direct_response requires source none")
            if self.response_type != "answer":
                raise ValueError("direct_response requires answer output")
            return self
        if self.source_preference == "none":
            raise ValueError("retrieval workflows require auto or web")
        self.direct_response_kind = None
        if self.query_type != "rag_qa":
            self.retrieval_analysis = None
        return self

class QueryRouterInput(TypedDict):
    """Public input accepted by the query router."""

    query: str
    report_format: NotRequired[Literal["markdown", "docx"]]


class QueryRouterOutput(TypedDict, total=False):
    """Stable public result returned by the query router."""

    query: str
    query_type: Optional[
        Literal["direct_response", "rag_qa", "technology_selection"]
    ]
    source_preference: Literal["none", "auto", "web"]
    response_type: Literal["answer", "report"]
    routing_reason: str
    status: Literal[
        "running",
        "completed",
        "awaiting_user",
        "needs_human_review",
        "error",
    ]
    answer: str
    report: str
    report_data: Dict[str, Any]
    report_format: Literal["markdown", "docx"]
    report_path: Optional[str]
    report_history: List[Dict[str, Any]]
    clarification_questions: List[str]
    clarification_answers: List[str]
    clarification_history: List[Dict[str, Any]]
    sources: List[Dict[str, Any]]
    details: Dict[str, Any]
    warnings: List[str]


class QueryRouterState(QueryRouterOutput, total=False):
    """Internal state shared by query-router nodes."""

    direct_response_kind: Literal[
        "identity",
        "greeting",
        "capabilities",
        "out_of_scope",
    ]
    direct_answer: str
    child_graph_result: Dict[str, Any]
    messages: Annotated[List[AnyMessage], add_messages]
    session_user_id: str
    session_project_id: str
    memory_context: str
    short_term_context: str
    historical_memory_context: str
    recalled_memory_ids: List[str]
    memory_scope_valid: bool
    memory_candidates: List[Dict[str, Any]]
    memory_write_count: int
    retrieval_analysis: Dict[str, Any]

"""State and validated decisions for the single-query router workflow."""

from typing import Any, Dict, List, Literal, NotRequired, Optional, TypedDict

from pydantic import BaseModel, model_validator


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
    clarification_questions: List[str]
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

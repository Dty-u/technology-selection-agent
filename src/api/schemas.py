"""Stable HTTP request and response models for the agent API."""

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class MessageRequest(BaseModel):
    """One user turn submitted to a session."""

    message: str = Field(min_length=1, max_length=10_000)
    report_format: Literal["markdown", "docx"] = "markdown"

    @field_validator("message")
    @classmethod
    def normalize_message(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("message must not be blank")
        return normalized


class WorkflowStage(BaseModel):
    scope: str
    stage_id: str
    label: str


class WorkflowProgressEvent(WorkflowStage):
    status: Literal["running", "completed", "paused", "error"]
    started_at: str
    finished_at: str | None = None
    duration_ms: int | None = None


class WorkflowProgress(BaseModel):
    request_id: str | None = None
    active: bool = False
    cancel_requested: bool = False
    cancelled: bool = False
    current_stage: WorkflowStage | None = None
    events: list[WorkflowProgressEvent] = Field(default_factory=list)


class CancelSessionResponse(BaseModel):
    """Acknowledgement for a cooperative cancellation request."""

    thread_id: str
    request_id: str | None = None
    status: Literal["cancelling", "cancelled", "not_running"]


class ReportSummary(BaseModel):
    report_id: str
    title: str
    created_at: str
    format: Literal["markdown", "docx"]
    download_url: str


class SessionResponse(BaseModel):
    """Frontend-safe view of one checkpointed agent session."""

    request_id: str | None = None
    project_id: str
    thread_id: str
    status: Literal[
        "running",
        "completed",
        "awaiting_user",
        "needs_human_review",
        "cancelled",
        "error",
    ]
    query_type: Literal[
        "direct_response",
        "rag_qa",
        "technology_selection",
    ] | None = None
    response_type: Literal["answer", "report"] | None = None
    answer: str = ""
    report: str = ""
    report_data: dict[str, Any] = Field(default_factory=dict)
    clarification_questions: list[str] = Field(default_factory=list)
    clarification_answers: list[str] = Field(default_factory=list)
    clarification_history: list[dict[str, Any]] = Field(default_factory=list)
    sources: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    report_url: str | None = None
    reports: list[ReportSummary] = Field(default_factory=list)
    progress: WorkflowProgress = Field(default_factory=WorkflowProgress)


class ErrorDetail(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    detail: ErrorDetail

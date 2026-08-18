"""Validated domain records shared by memory adapters and workflow nodes."""

from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator, model_validator


MemoryStatus = Literal[
    "pending",
    "confirmed",
    "approved",
    "superseded",
    "deleted",
]
MemoryType = Literal[
    "user_preference",
    "user_fact",
    "conversation_summary",
    "project_fact",
    "selection_constraint",
    "selection_decision",
    "selection_case",
]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class MemoryCandidate(BaseModel):
    """One candidate fact awaiting authoritative persistence."""

    idempotency_key: str
    user_id: str
    thread_id: str
    project_id: str | None = None
    memory_type: MemoryType
    content: str
    status: MemoryStatus = "pending"
    source_kind: str
    confidence: float = Field(default=0.5, ge=0, le=1)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator(
        "idempotency_key",
        "user_id",
        "thread_id",
        "content",
        "source_kind",
    )
    @classmethod
    def require_non_empty(cls, value: str) -> str:
        normalized = str(value or "").strip()
        if not normalized:
            raise ValueError("value must not be empty")
        return normalized

    @field_validator("project_id")
    @classmethod
    def normalize_project_id(cls, value: str | None) -> str | None:
        normalized = str(value or "").strip()
        return normalized or None

    @model_validator(mode="after")
    def require_project_scope_for_project_memory(self):
        if self.memory_type.startswith("selection_") or self.memory_type == "project_fact":
            if not self.project_id:
                raise ValueError(
                    f"project_id is required for {self.memory_type} memory"
                )
        return self


class MemoryItem(BaseModel):
    """Authoritative, versioned memory record stored in MySQL."""

    memory_id: str = Field(default_factory=lambda: str(uuid4()))
    idempotency_key: str
    user_id: str
    thread_id: str
    project_id: str | None = None
    memory_type: MemoryType
    content: str
    status: MemoryStatus
    source_kind: str
    confidence: float
    metadata: dict[str, Any] = Field(default_factory=dict)
    version: int = 1
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class OutboxEvent(BaseModel):
    """Projection request committed atomically with its authoritative item."""

    event_id: str = Field(default_factory=lambda: str(uuid4()))
    memory_id: str
    event_type: Literal["memory.upsert", "memory.delete"] = "memory.upsert"
    status: Literal["pending", "processing", "projected", "failed"] = "pending"
    attempts: int = 0
    payload: dict[str, Any] = Field(default_factory=dict)
    available_at: datetime = Field(default_factory=utc_now)
    created_at: datetime = Field(default_factory=utc_now)


class SemanticMemoryHit(BaseModel):
    """A disposable Milvus match that must be hydrated from MySQL."""

    memory_id: str
    score: float
    version: int = Field(ge=1)

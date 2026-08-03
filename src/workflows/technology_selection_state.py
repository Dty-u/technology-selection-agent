from hashlib import sha256
from typing import Any, Dict, List, Literal, Optional, TypedDict

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.workflows.rag_qa_state import GraphState


CRITERIA = (
    "performance",
    "power",
    "thermal",
    "cost",
    "maturity",
    "risk",
    "scalability",
)
DEFAULT_CRITERIA_WEIGHTS = {
    criterion: 1 / len(CRITERIA)
    for criterion in CRITERIA
}


class ScenarioCriterion(BaseModel):
    model_config = ConfigDict(validate_default=True)

    name: str
    criterion_id: str = ""
    description: str = ""
    priority: Literal["highest", "high", "medium", "low"] = "medium"

    @field_validator("name", mode="before")
    @classmethod
    def normalize_name(cls, value):
        name = str(value or "").strip()
        if name in {"薄膜均匀性", "膜厚均匀性"}:
            return "薄膜均匀性与共形性"
        return name

    @field_validator("criterion_id", mode="before")
    @classmethod
    def normalize_criterion_id(cls, value, info):
        if str(value or "").strip():
            normalized = "".join(
                character if character.isalnum() else "_"
                for character in str(value).strip().casefold()
            ).strip("_")
            if normalized and normalized not in CRITERIA:
                return normalized
        name = str(info.data.get("name", "")).strip().casefold()
        digest = sha256(name.encode("utf-8")).hexdigest()[:10]
        return f"scenario_{digest}"


class SelectionRequirements(BaseModel):
    business_goal: str = ""
    application_scenario: str = ""
    active_criteria: List[str] = Field(default_factory=list)
    priorities: Dict[str, float] = Field(
        default_factory=lambda: dict(DEFAULT_CRITERIA_WEIGHTS)
    )
    hard_constraints: List[str] = Field(default_factory=list)
    requested_candidates: List[str] = Field(default_factory=list)
    scenario_criteria: List[ScenarioCriterion] = Field(default_factory=list)
    missing_information: List[str] = Field(default_factory=list)

    @field_validator("active_criteria", mode="before")
    @classmethod
    def normalize_active_criteria(cls, value):
        if not isinstance(value, list):
            return []
        return list(
            dict.fromkeys(
                str(criterion).strip()
                for criterion in value
                if str(criterion).strip() in CRITERIA
            )
        )

    @field_validator("priorities", mode="before")
    @classmethod
    def normalize_priorities(cls, value):
        if not isinstance(value, dict) or not value:
            return dict(DEFAULT_CRITERIA_WEIGHTS)
        cleaned = {}
        for criterion in CRITERIA:
            try:
                cleaned[criterion] = max(float(value.get(criterion, 0)), 0)
            except (TypeError, ValueError):
                cleaned[criterion] = 0
        total = sum(cleaned.values())
        if total <= 0:
            return dict(DEFAULT_CRITERIA_WEIGHTS)
        return {
            criterion: weight / total
            for criterion, weight in cleaned.items()
        }

    @field_validator("scenario_criteria", mode="before")
    @classmethod
    def normalize_scenario_criteria(cls, value):
        if not isinstance(value, list):
            return []
        normalized = []
        seen = set()
        for raw_item in value:
            if not isinstance(raw_item, dict):
                continue
            item = dict(raw_item)
            name = str(item.get("name", "")).strip()
            if not name:
                continue
            key = name.casefold()
            if key in seen:
                continue
            seen.add(key)
            normalized.append(item)
            if len(normalized) >= 3:
                break
        return normalized


class CandidateOption(BaseModel):
    candidate_id: str
    name: str
    summary: str = ""
    feasible: bool = True
    exclusion_reasons: List[str] = Field(default_factory=list)


class EvidenceTask(BaseModel):
    task_id: str
    candidate_ids: List[str]
    query: str
    criteria: List[str] = Field(default_factory=lambda: list(CRITERIA))
    source_preference: Literal["auto", "local", "web"] = "auto"


class EvidenceSupport(BaseModel):
    candidate_id: str
    criterion: str
    level: Literal["direct", "indirect", "none"] = "none"
    quote: str = ""


class EvidenceRecord(BaseModel):
    evidence_id: str
    candidate_ids: List[str] = Field(default_factory=list)
    criteria: List[str] = Field(default_factory=list)
    supports: List[EvidenceSupport] = Field(default_factory=list)
    content: str
    source: str = ""
    url: str = ""
    title: str = ""
    retrieval_source: str = ""
    retrieval_sources: List[str] = Field(default_factory=list)
    source_reliability: Literal[
        "internal",
        "web_identified",
        "web_unattributed",
    ] = "internal"


class CriterionScore(BaseModel):
    criterion: str
    score: Optional[int] = Field(default=None, ge=1, le=5)
    rationale: str = ""
    evidence_ids: List[str] = Field(default_factory=list)
    confidence: Literal["low", "medium", "high"] = "low"


class CandidateEvaluation(BaseModel):
    candidate_id: str
    criterion_scores: List[CriterionScore] = Field(default_factory=list)
    scenario_scores: List[CriterionScore] = Field(default_factory=list)
    weighted_score: Optional[float] = None
    evidence_coverage: float = Field(default=0, ge=0, le=1)
    scenario_coverage: float = Field(default=0, ge=0, le=1)


class RiskItem(BaseModel):
    candidate_id: str
    category: str
    level: Literal["low", "medium", "high", "unknown"]
    impact: str
    mitigation: str
    evidence_ids: List[str] = Field(default_factory=list)
    basis: Literal["evidence", "inference", "to_verify"] = "to_verify"


class RecommendationResult(BaseModel):
    decision: Literal["recommend", "conditional", "insufficient_evidence"]
    recommended_candidate_id: Optional[str] = None
    alternative_candidate_ids: List[str] = Field(default_factory=list)
    rationale: List[str] = Field(default_factory=list)
    prerequisites: List[str] = Field(default_factory=list)
    unresolved_items: List[str] = Field(default_factory=list)
    decision_basis: Literal[
        "weighted_score",
        "scenario_criteria",
        "insufficient_evidence",
    ] = "insufficient_evidence"


class RecommendationReview(BaseModel):
    verdict: Literal["pass", "evidence_gap", "reevaluate"]
    reasons: List[str] = Field(default_factory=list)
    requested_queries: List[str] = Field(default_factory=list)


class EvidenceRetrievalState(GraphState, total=False):
    """State used by the retrieval-only workflow."""

    source_preference: Literal["auto", "local", "web"]
    max_transform_count: int
    compact_relevance: bool
    allow_web_fallback: bool
    web_search_max_results: int
    web_search_engine: str
    require_source_url: bool


class TechnologySelectionState(TypedDict, total=False):
    """Shared state for the semiconductor technology-selection workflow."""

    user_request: str
    clarification_answers: List[str]
    status: Literal[
        "running",
        "awaiting_user",
        "completed",
        "needs_human_review",
    ]
    requirements: Dict[str, Any]
    requirements_complete: bool
    clarification_questions: List[str]
    candidates: List[Dict[str, Any]]
    feasible_candidates: List[Dict[str, Any]]
    excluded_candidates: List[Dict[str, Any]]
    evidence_tasks: List[Dict[str, Any]]
    evidence_records: List[Dict[str, Any]]
    evidence_sufficient: bool
    evidence_gaps: List[str]
    missing_evidence_queries: List[str]
    evidence_round: int
    evaluations: List[Dict[str, Any]]
    risk_items: List[Dict[str, Any]]
    recommendation: Dict[str, Any]
    recommendation_review: Dict[str, Any]
    review_round: int
    report: str
    report_data: Dict[str, Any]
    warnings: List[str]

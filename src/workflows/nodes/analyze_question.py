"""Analyze one question for routing and graph retrieval in one LLM call."""

import json
from typing import List, Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from src.infrastructure.llm.chat_model import (
    selection_json_llm,
    selection_llm,
    utility_json_llm,
)
from src.observability.logging import log

ALLOWED_RELATION_TYPES = {
    "AFFECTS",
    "CAUSES",
    "LEADS_TO",
    "IMPROVES",
    "REDUCES",
    "REQUIRES",
    "ENABLES",
    "PRODUCES",
    "HAS_CHALLENGE",
}


class QuestionAnalysis(BaseModel):
    route: Literal["local_hybrid", "web_search"]
    query_entities: List[str] = Field(default_factory=list)
    relation_intent: str = ""
    relation_types: List[str] = Field(default_factory=list)


class RetrievalIntent(BaseModel):
    query_entities: List[str] = Field(default_factory=list)
    relation_intent: str = ""
    relation_types: List[str] = Field(default_factory=list)


analysis_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """你负责分析用户问题并选择知识来源。
本地知识库覆盖半导体材料、芯片制造、器件结构、光刻、沉积、刻蚀和封装。
属于这些主题时 route 使用 local_hybrid，否则使用 web_search。
同时抽取适合在知识图谱中匹配的实体名称，以及问题关注的关系意图。
只返回 JSON，不要返回 Markdown：
{{"route":"local_hybrid","query_entities":["实体"],"relation_intent":"关系意图","relation_types":["AFFECTS"]}}""",
        ),
        ("human", "{question}"),
    ]
)
question_analysis_llm = utility_json_llm
question_analysis_chain = analysis_prompt | question_analysis_llm
selection_question_analysis_chain = analysis_prompt | selection_json_llm


retrieval_intent_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """你负责为本地知识库检索抽取问题中的实体和关系意图。
只抽取适合知识图谱匹配的实体名称、关系意图和关系类型，
不要判断使用本地知识库还是网络搜索。
只返回 JSON，不要返回 Markdown：
{{"query_entities":["实体"],"relation_intent":"关系意图","relation_types":["AFFECTS"]}}""",
        ),
        ("human", "{question}"),
    ]
)
retrieval_intent_chain = retrieval_intent_prompt | utility_json_llm
selection_retrieval_intent_chain = retrieval_intent_prompt | selection_llm


def _parse_json_object(content: str) -> dict:
    """Parse one JSON object while tolerating an accidental code fence."""
    normalized = content.strip()
    if normalized.startswith("```"):
        normalized = normalized.split("\n", 1)[-1]
        if normalized.endswith("```"):
            normalized = normalized[:-3]
    object_start = normalized.find("{")
    object_end = normalized.rfind("}")
    if object_start < 0 or object_end < object_start:
        raise ValueError("Question analysis did not return a JSON object")
    return json.loads(normalized[object_start : object_end + 1])


def _normalize_relation_types(payload: dict) -> dict:
    """Filter and normalize relation types accepted by graph retrieval."""
    payload["relation_types"] = list(
        dict.fromkeys(
            relation_type
            for value in payload.get("relation_types", [])
            if (relation_type := str(value).strip().upper())
            in ALLOWED_RELATION_TYPES
        )
    )
    return payload


def parse_question_analysis(content: str) -> QuestionAnalysis:
    """Parse routing and graph retrieval fields from one model response."""
    payload = _normalize_relation_types(_parse_json_object(content))
    return QuestionAnalysis.model_validate(payload)


def parse_retrieval_intent(content: str) -> RetrievalIntent:
    """Parse graph retrieval fields without accepting a source route."""
    payload = _normalize_relation_types(_parse_json_object(content))
    return RetrievalIntent.model_validate(payload)


def analyze_question(state):
    """Return route, graph entities, and relation intent from one model call."""
    if state.get("analysis_ready"):
        try:
            return QuestionAnalysis.model_validate(
                {
                    "route": state.get("route"),
                    "query_entities": state.get("query_entities", []),
                    "relation_intent": state.get("relation_intent", ""),
                    "relation_types": state.get("relation_types", []),
                }
            ).model_dump()
        except (TypeError, ValueError):
            log.warning(
                "---预计算检索意图无效，回退到问题分析模型---"
            )
    question = state["question"]
    log.info("---ANALYZE QUESTION---")
    chain = (
        selection_question_analysis_chain
        if state.get("compact_relevance")
        else question_analysis_chain
    )
    response = chain.invoke({"question": question})
    content = response.content if hasattr(response, "content") else str(response)
    try:
        return parse_question_analysis(content).model_dump()
    except (TypeError, ValueError) as error:
        log.warning(
            "---问题分析未返回有效 JSON：错误类型={}，降级为本地混合检索---",
            type(error).__name__,
        )
        return QuestionAnalysis(route="local_hybrid").model_dump()


def extract_retrieval_intent(state):
    """Extract graph retrieval fields without choosing a knowledge source."""
    question = state["question"]
    log.info("---EXTRACT RETRIEVAL INTENT---")
    chain = (
        selection_retrieval_intent_chain
        if state.get("compact_relevance")
        else retrieval_intent_chain
    )
    response = chain.invoke({"question": question})
    content = response.content if hasattr(response, "content") else str(response)
    return parse_retrieval_intent(content).model_dump()

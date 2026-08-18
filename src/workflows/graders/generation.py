"""Single-call assessment for grounding and answer completeness."""

import json
from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel

from src.infrastructure.llm.chat_model import utility_json_llm


class GenerationAssessment(BaseModel):
    grounded: Literal["yes", "no"]
    answers_question: Literal["yes", "no"]


assessment_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Evaluate the generated answer using only the supplied evidence. "
            "Return JSON only: "
            '{{"grounded":"yes|no","answers_question":"yes|no"}}.',
        ),
        (
            "human",
            "Evidence:\n{documents}\n\nQuestion:\n{question}\n\n"
            "Generated answer:\n{generation}",
        ),
    ]
)


def _parse_assessment(message):
    content = message.content if hasattr(message, "content") else str(message)
    normalized = str(content).strip().removeprefix("```json").removesuffix("```")
    return GenerationAssessment.model_validate(json.loads(normalized.strip()))


generation_assessment_chain = (
    assessment_prompt
    | utility_json_llm
    | RunnableLambda(_parse_assessment)
)

from langchain_openai import ChatOpenAI

from src.config.settings import (
    DEEPSEEK_API_KEY,
    LLM_BASE_URL,
    LLM_MAX_RETRIES,
    LLM_MODEL,
    LLM_REQUEST_TIMEOUT_SECONDS,
    LLM_TEMPERATURE,
    UTILITY_LLM_MAX_RETRIES,
    UTILITY_LLM_TIMEOUT_SECONDS,
)

llm = ChatOpenAI(
    temperature=LLM_TEMPERATURE,
    model=LLM_MODEL,
    api_key=DEEPSEEK_API_KEY,
    base_url=LLM_BASE_URL,
    timeout=LLM_REQUEST_TIMEOUT_SECONDS,
    max_retries=LLM_MAX_RETRIES,
)

utility_llm = ChatOpenAI(
    temperature=0.0,
    model=LLM_MODEL,
    api_key=DEEPSEEK_API_KEY,
    base_url=LLM_BASE_URL,
    timeout=UTILITY_LLM_TIMEOUT_SECONDS,
    max_retries=UTILITY_LLM_MAX_RETRIES,
    extra_body={"thinking": {"type": "disabled"}},
)

utility_json_llm = utility_llm.bind(
    response_format={"type": "json_object"},
)

# Preserve the pre-existing technology-selection model behavior and its
# production timeout while utility chains use the shorter utility budget.
selection_llm = llm.bind(
    temperature=0.0,
    extra_body={"thinking": {"type": "disabled"}},
)

# Use provider-side JSON mode only for chains whose public contract is JSON.
# Keep selection_llm available for any non-JSON technology-selection call.
selection_json_llm = selection_llm.bind(
    response_format={"type": "json_object"},
)

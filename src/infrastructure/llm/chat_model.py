from langchain_openai import ChatOpenAI

from src.config.settings import (
    DEEPSEEK_API_KEY,
    LLM_BASE_URL,
    LLM_MODEL,
    LLM_TEMPERATURE,
)

llm = ChatOpenAI(
    temperature=LLM_TEMPERATURE,
    model=LLM_MODEL,
    api_key=DEEPSEEK_API_KEY,
    base_url=LLM_BASE_URL)

# Technology-selection nodes mostly produce bounded JSON. DeepSeek V4 enables
# thinking by default, which adds large latency without helping these schemas.
# Keep the shared model unchanged so ordinary RAG behavior is unaffected.
selection_llm = llm.bind(
    temperature=0.0,
    extra_body={"thinking": {"type": "disabled"}},
)

# Use provider-side JSON mode only for chains whose public contract is JSON.
# Keep selection_llm available for any non-JSON technology-selection call.
selection_json_llm = selection_llm.bind(
    response_format={"type": "json_object"},
)

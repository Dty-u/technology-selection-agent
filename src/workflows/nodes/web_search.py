import re
from datetime import datetime
from time import perf_counter

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from src.config.settings import (
    WEB_SEARCH_ENGINE,
    WEB_SEARCH_MAX_ATTEMPTS,
    WEB_SEARCH_MAX_RESULTS,
)
from src.infrastructure.llm.chat_model import utility_llm
from src.infrastructure.search.zhipu import WebSearchError, web_search_tool
from src.observability.logging import log
from src.workflows.graders.document_relevance import retrieval_grader_chain


_SEMICONDUCTOR_EXPANSIONS = {
    "GAA": "Gate-All-Around transistor",
    "HBM": "High Bandwidth Memory",
    "EUV": "Extreme Ultraviolet lithography",
    "TSV": "Through-Silicon Via",
    "ALD": "Atomic Layer Deposition",
    "CVD": "Chemical Vapor Deposition",
    "SIC": "Silicon Carbide SiC",
    "GAN": "Gallium Nitride GaN",
}
_TIME_SENSITIVE_PATTERN = re.compile(
    r"目前|当前|最新|近期|进展|量产|现状|latest|current|progress|production",
    re.IGNORECASE,
)
_WEB_REWRITE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是半导体联网检索查询改写器。保持用户原意，补全缩写和技术语境，"
            "针对未解决的问题生成一条更精准的搜索词。只返回搜索词。",
        ),
        (
            "human",
            "原始问题：{question}\n已搜索：{previous_queries}\n"
            "未解决的回答：{previous_answer}\n请生成新的搜索词。",
        ),
    ]
)
web_query_rewriter_chain = _WEB_REWRITE_PROMPT | utility_llm | StrOutputParser()


def _contains_ascii_token(text, token):
    return bool(
        re.search(
            rf"(?<![A-Za-z0-9]){re.escape(token)}(?![A-Za-z0-9])",
            text,
            re.IGNORECASE,
        )
    )


def _expanded_terms(question):
    terms = []
    for token, expansion in _SEMICONDUCTOR_EXPANSIONS.items():
        if _contains_ascii_token(question, token):
            terms.extend((token, expansion))
    return terms


def _initial_search_query(question):
    parts = [question.strip()]
    expansions = _expanded_terms(question)
    if expansions:
        parts.extend(expansions[1::2])
        parts.append("semiconductor")
    if _TIME_SENSITIVE_PATTERN.search(question):
        parts.append(f"{datetime.now().year} latest production status")
    return " ".join(dict.fromkeys(part for part in parts if part))


def _rewrite_search_query(question, previous_queries, previous_answer):
    try:
        rewritten = web_query_rewriter_chain.invoke(
            {
                "question": question,
                "previous_queries": "\n".join(previous_queries),
                "previous_answer": previous_answer,
            }
        ).strip()
    except Exception:
        log.warning("---联网补检查询改写失败：使用确定性改写---")
        rewritten = ""
    if rewritten and rewritten not in previous_queries:
        return rewritten
    return (
        f"{_initial_search_query(question)} official semiconductor "
        "manufacturing update"
    )


def _documents_from_results(results, require_source_url):
    documents = [
        Document(
            page_content=str(result.get("content", "")).strip(),
            metadata={
                "title": str(result.get("title", "")).strip(),
                "source": str(result.get("url", "")).strip(),
                "url": str(result.get("url", "")).strip(),
                "retrieval_source": "web_search",
            },
        )
        for result in results
        if str(result.get("content", "")).strip()
        and (
            not require_source_url
            or str(result.get("url", "")).strip()
        )
    ]
    return sorted(
        documents,
        key=lambda document: not bool(document.metadata.get("url")),
    )


def _fallback_relevant_documents(question, documents):
    terms = _expanded_terms(question)
    if not terms:
        latin_terms = re.findall(r"[A-Za-z][A-Za-z0-9.+-]{2,}", question)
        terms = [term for term in latin_terms if term.lower() not in {
            "the", "and", "how", "what", "latest", "current", "technology"
        }]
    if not terms:
        return documents
    return [
        document
        for document in documents
        if any(
            term.lower() in (
                f"{document.metadata.get('title', '')} {document.page_content}"
            ).lower()
            for term in terms
        )
    ]


def _filter_relevant_documents(question, documents):
    if not documents:
        return []
    formatted = "\n\n".join(
        f"[{index}] 标题：{document.metadata.get('title', '')}\n"
        f"来源：{document.metadata.get('url', '')}\n"
        f"{document.page_content[:1000]}"
        for index, document in enumerate(documents[:WEB_SEARCH_MAX_RESULTS])
    )
    try:
        score = retrieval_grader_chain.invoke(
            {"question": question, "documents": formatted}
        )
        indices = {
            index
            for index in score.relevant_indices
            if 0 <= index < len(documents)
        }
        return [
            document
            for index, document in enumerate(documents)
            if index in indices
        ]
    except Exception:
        log.warning("---联网结果相关性评估失败：使用实体匹配兜底---")
        return _fallback_relevant_documents(question, documents)


def _search_once(query, max_results, search_engine):
    return web_search_tool.invoke(
        query,
        max_results=max_results,
        search_engine=search_engine,
    )


def web_search(state):
    """
    基于优化后的问题进行网络搜索

    Args:
        state (dict): 当前图状态，包含优化后的问题

    Returns:
        state (dict): 更新后的状态，documents字段替换为网络搜索结果
    """
    started_at = perf_counter()
    log.info("---开始联网检索---")
    question = state["question"]
    original_question = state.get("original_question", question)
    compact_relevance = bool(state.get("compact_relevance", False))
    max_results = state.get("web_search_max_results", WEB_SEARCH_MAX_RESULTS)
    search_engine = state.get("web_search_engine", WEB_SEARCH_ENGINE)
    require_source_url = bool(state.get("require_source_url", False))
    max_attempts = 1 if compact_relevance else WEB_SEARCH_MAX_ATTEMPTS
    attempt_count = int(state.get("web_search_attempt_count", 0))
    queries = list(state.get("web_search_queries", []))

    if attempt_count >= max_attempts:
        log.info("---联网补检达到上限：不再调用搜索服务---")
        return {
            "documents": list(state.get("documents", [])),
            "question": question,
            "original_question": original_question,
            "retrieval_source": "web_search",
            "retrieval_error": "web_search_retry_exhausted",
            "generation_attempt_count": state.get("generation_attempt_count", 0),
            "web_search_attempt_count": attempt_count,
            "web_search_queries": queries,
            "web_relevant_result_count": len(state.get("documents", [])),
        }

    search_query = (
        question
        if compact_relevance
        else (
            _initial_search_query(question)
            if attempt_count == 0
            else _rewrite_search_query(
                question,
                queries,
                state.get("generation", ""),
            )
        )
    )

    while attempt_count < max_attempts:
        query_started_at = perf_counter()
        attempt_count += 1
        queries.append(search_query)
        try:
            results = _search_once(search_query, max_results, search_engine)
        except WebSearchError:
            log.warning("---联网检索暂时不可用：不产生可生成证据---")
            return {
                "documents": [],
                "question": question,
                "original_question": original_question,
                "retrieval_source": "web_search",
                "retrieval_error": "web_search_unavailable",
                "generation_attempt_count": state.get("generation_attempt_count", 0),
                "web_search_attempt_count": attempt_count,
                "web_search_query": search_query,
                "web_search_queries": queries,
                "web_relevant_result_count": 0,
            }

        documents = _documents_from_results(results, require_source_url)
        relevant_documents = (
            documents
            if compact_relevance
            else _filter_relevant_documents(question, documents)
        )
        log.info(
            f"---联网检索批次完成：轮次={attempt_count}，"
            f"结果={len(documents)}，相关={len(relevant_documents)}，"
            f"耗时={perf_counter() - query_started_at:.2f}s---"
        )
        if relevant_documents:
            log.info(
                f"---联网检索完成：总耗时={perf_counter() - started_at:.2f}s---"
            )
            return {
                "documents": relevant_documents,
                "question": question,
                "original_question": original_question,
                "retrieval_source": "web_search",
                "retrieval_error": None,
                "generation_attempt_count": 0,
                "web_search_attempt_count": attempt_count,
                "web_search_query": search_query,
                "web_search_queries": queries,
                "web_relevant_result_count": len(relevant_documents),
            }
        if compact_relevance or attempt_count >= max_attempts:
            break
        log.info("---首次联网结果无关：执行一次精准补检---")
        search_query = _rewrite_search_query(
            question,
            queries,
            state.get("generation", ""),
        )

    error = (
        "web_search_no_usable_results"
        if compact_relevance
        else "web_search_no_relevant_results"
    )
    log.info(
        f"---联网检索未取得相关结果：总耗时={perf_counter() - started_at:.2f}s---"
    )
    return {
        "documents": [],
        "question": question,
        "original_question": original_question,
        "retrieval_source": "web_search",
        "retrieval_error": error,
        "generation_attempt_count": state.get("generation_attempt_count", 0),
        "web_search_attempt_count": attempt_count,
        "web_search_query": search_query,
        "web_search_queries": queries,
        "web_relevant_result_count": 0,
    }

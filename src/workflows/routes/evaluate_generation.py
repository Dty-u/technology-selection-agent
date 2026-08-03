from time import perf_counter

from src.config.settings import MAX_GENERATION_ATTEMPTS, MAX_TRANSFORM_COUNT
from src.observability.logging import log
from src.workflows.graders.answer_quality import answer_grader_chain
from src.workflows.graders.hallucination import hallucination_grader_chain


def grade_generation_v_documents_and_question(state):
    """Route an evaluated answer with bounded generation and rewrite attempts."""
    started_at = perf_counter()
    log.info("---检查生成回答是否基于检索证据---")
    if state.get("retrieval_error"):
        log.info("---评估决策：检索失败，结束工作流---")
        return "end"
    question = state["question"]
    evaluation_question = state.get("original_question") or question
    documents = state["documents"]
    generation = state["generation"]

    grounded_score = hallucination_grader_chain.invoke(
        {"documents": documents, "generation": generation}
    )
    grounded = grounded_score.binary_score.strip().lower()
    log.info(f"---幻觉评估：{'通过' if grounded == 'yes' else '未通过'}---")
    if grounded == "yes":
        answer_score = answer_grader_chain.invoke(
            {"question": evaluation_question, "generation": generation}
        )
        answer_quality = answer_score.binary_score.strip().lower()
        log.info(
            f"---问题解决评估：{'通过' if answer_quality == 'yes' else '未通过'}---"
        )
        if answer_quality == "yes":
            log.info("---评估决策：接受回答---")
            log.info(
                f"---回答审核完成：耗时={perf_counter() - started_at:.2f}s---"
            )
            return "useful"
        if state.get("retrieval_source") == "web_search":
            log.info("---评估决策：联网回答未解决问题，执行一次精准补检---")
            return "web_search"
        if state.get("transform_count", 0) >= MAX_TRANSFORM_COUNT:
            log.info("---评估决策：转为联网检索，查询改写已达上限---")
            return "web_search"
        log.info("---评估决策：改写查询，回答未解决问题---")
        return "transform_query"

    if state.get("generation_attempt_count", 0) < MAX_GENERATION_ATTEMPTS:
        log.info("---评估决策：重新生成，回答缺少证据支持---")
        return "generate"
    if state.get("retrieval_source") == "web_search":
        log.info("---评估决策：联网回答仍缺少支持，转入有界补检---")
        return "web_search"
    log.info("---评估决策：转为联网检索，生成重试已达上限---")
    return "web_search"

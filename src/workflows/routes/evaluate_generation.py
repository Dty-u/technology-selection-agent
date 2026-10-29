from src.config.settings import MAX_GENERATION_ATTEMPTS, MAX_TRANSFORM_COUNT
from src.observability.logging import log


def grade_generation_v_documents_and_question(state):
    """Route on the stored answer assessment with bounded retry budgets."""
    if state.get("retrieval_error"):
        log.info("---评估决策：检索失败，结束工作流---")
        return "end"

    if state.get("generation_grounded") == "yes":
        if state.get("generation_answers_question") == "yes":
            log.info("---评估决策：接受回答---")
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

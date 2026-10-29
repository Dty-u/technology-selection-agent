from time import perf_counter

from src.observability.logging import log
from src.workflows.graders.answer_quality import answer_grader_chain
from src.workflows.graders.generation import generation_assessment_chain
from src.workflows.graders.hallucination import hallucination_grader_chain


def _legacy_assessment(*, documents, question, generation):
    grounded_score = hallucination_grader_chain.invoke(
        {"documents": documents, "generation": generation}
    )
    grounded = grounded_score.binary_score.strip().lower()
    answers_question = "no"
    if grounded == "yes":
        answer_score = answer_grader_chain.invoke(
            {"question": question, "generation": generation}
        )
        answers_question = answer_score.binary_score.strip().lower()
    return grounded, answers_question


def assess_generation(state):
    """Grade the latest answer once and store the verdict in workflow state."""
    if state.get("retrieval_error"):
        log.info("---检索失败：跳过回答评估---")
        return {
            "generation_grounded": None,
            "generation_answers_question": None,
        }

    started_at = perf_counter()
    log.info("---检查生成回答是否基于检索证据---")
    evaluation_question = state.get("original_question") or state["question"]
    documents = state["documents"]
    generation = state["generation"]
    try:
        assessment = generation_assessment_chain.invoke(
            {
                "documents": documents,
                "question": evaluation_question,
                "generation": generation,
            }
        )
        grounded = assessment.grounded.strip().lower()
        answers_question = assessment.answers_question.strip().lower()
        if grounded not in {"yes", "no"} or answers_question not in {"yes", "no"}:
            raise ValueError("invalid combined generation assessment")
    except Exception:
        log.warning("---Combined answer assessment failed; using legacy graders---")
        grounded, answers_question = _legacy_assessment(
            documents=documents,
            question=evaluation_question,
            generation=generation,
        )

    log.info(f"---幻觉评估：{'通过' if grounded == 'yes' else '未通过'}---")
    if grounded == "yes":
        log.info(
            f"---问题解决评估：{'通过' if answers_question == 'yes' else '未通过'}---"
        )
    log.info(f"---回答审核完成：耗时={perf_counter() - started_at:.2f}s---")
    return {
        "generation_grounded": grounded,
        "generation_answers_question": answers_question,
    }

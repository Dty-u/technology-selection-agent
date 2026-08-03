from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from src.infrastructure.llm.chat_model import llm
from src.observability.logging import log


rewrite_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """作为问题重写器，请将输入问题转换为更适合本地知识库检索的版本。
理解问题背后的语义意图，保留实体名称、比较维度和关系要求。""",
        ),
        (
            "human",
            "这是初始问题：\n\n{question}\n请生成一个优化后的检索问题。",
        ),
    ]
)
question_rewriter_chain = rewrite_prompt | llm | StrOutputParser()


def transform_query(state):
    """Rewrite the query and clear evidence derived from the previous query."""
    log.info("---TRANSFORM QUERY---")
    better_question = question_rewriter_chain.invoke(
        {"question": state["question"]}
    )
    return {
        "question": better_question,
        "documents": [],
        "vector_documents": [],
        "graph_documents": [],
        "query_entities": [],
        "relation_intent": "",
        "relation_types": [],
        "transform_count": state.get("transform_count", 0) + 1,
        "generation_attempt_count": 0,
        "graph_error": None,
    }

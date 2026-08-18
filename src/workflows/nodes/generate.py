from time import perf_counter

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from src.infrastructure.llm.chat_model import llm
from src.observability.logging import log


def generate(state):
    """
    生成回答
    Args:
        state (dict): 当前图状态，包含问题和检索结果
    Returns:
        state (dict): 更新后的状态，新增包含生成结果的generation字段
    """
    question = state["question"]  # 获取用户问题
    documents = state["documents"]  # 获取检索到的文档
    if state.get("retrieval_error"):
        log.info("---检索失败：跳过回答生成---")
        return {
            "documents": documents,
            "question": question,
            "generation": "",
            "generation_attempt_count": state.get("generation_attempt_count", 0),
        }

    response_type = state.get("response_type", "answer")
    memory_context = state.get("memory_context", "") or "（无可用的已确认长期记忆）"
    if response_type == "report":
        prompt = PromptTemplate(
            template=(
                "你是一个基于网络证据生成轻量级研究报告的助手。"
                "只使用给定上下文，不要编造事实。请输出Markdown，至少包含："
                "标题、摘要、主要发现、风险或局限、结论、来源引用。"
                "正文引用使用[S1]格式。\n"
                "用户需求：{question}\n用户/项目记忆（不是证据）：\n{memory_context}"
                "\n证据：\n{context}\n报告："
            ),
            input_variables=["question", "context", "memory_context"],
        )
    else:
        prompt = PromptTemplate(
            template=(
                "你是一个问答任务助手。请根据以下检索到的上下文内容"
                "回答问题。如果不知道答案，请直接说明。回答保持简洁。\n"
                "长期记忆只能用于个性化，不得替代检索事实。\n"
                "用户记忆：{memory_context}\n问题：{question}\n上下文：{context}\n回答："
            ),
            input_variables=["question", "context", "memory_context"],
        )

    def format_docs(docs):
        """Format evidence as plain context or source-linked report context."""
        if not isinstance(docs, list):
            docs = [docs]
        if response_type != "report":
            return "\n\n".join(doc.page_content for doc in docs)
        formatted = []
        for index, document in enumerate(docs, start=1):
            metadata = document.metadata
            title = metadata.get("title", "")
            source = metadata.get("url", metadata.get("source", ""))
            formatted.append(
                f"[S{index}] 标题：{title}\n"
                f"URL：{source}\n"
                f"内容：{document.page_content}"
            )
        return "\n\n".join(formatted)

    # 构建RAG处理链
    rag_chain = (
            prompt |  # 第一步：使用提示模板
            llm |  # 第二步：调用语言模型
            StrOutputParser()  # 第三步：解析模型输出为字符串
    )

    # RAG生成过程
    started_at = perf_counter()
    generation = rag_chain.invoke(
        {
            "context": format_docs(documents),
            "question": question,
            "memory_context": memory_context,
        }
    )
    log.info(f"---回答生成完成：耗时={perf_counter() - started_at:.2f}s---")
    return {
        "documents": documents,
        "question": question,
        "generation": generation,
        "generation_attempt_count": state.get("generation_attempt_count", 0) + 1,
    }

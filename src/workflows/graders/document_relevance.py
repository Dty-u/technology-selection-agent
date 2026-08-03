import json

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel, Field

from src.infrastructure.llm.chat_model import llm, selection_llm


# 数据模型 - 文档相关性评分
class GradeDocuments(BaseModel):
    """一次选择相关文档的下标。"""

    relevant_indices: list[int] = Field(
        default_factory=list,
        description="相关文档的下标列表",
    )


# 带函数调用的LLM初始化
structured_llm_grader = llm | RunnableLambda(
    lambda message: GradeDocuments.model_validate(
        json.loads(message.content.strip().removeprefix("```json").removesuffix("```").strip())
    )
)
structured_selection_llm_grader = selection_llm | RunnableLambda(
    lambda message: GradeDocuments.model_validate(
        json.loads(message.content.strip().removeprefix("```json").removesuffix("```").strip())
    )
)

# 提示词模板
system = """你是检索证据相关性评分器。根据用户问题，从候选文档中选择相关下标。
只返回 JSON，不要返回 Markdown：{{"relevant_indices":[0,2]}}。"""
grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),  # 系统角色提示
        ("human", "候选文档：\n\n{documents}\n\n用户问题：{question}"),
    ]
)

# 构建检索评分器工作流
retrieval_grader_chain = grade_prompt | structured_llm_grader  # 组合提示模板和LLM评分器
selection_retrieval_grader_chain = grade_prompt | structured_selection_llm_grader

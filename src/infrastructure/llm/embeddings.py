from src.config.settings import (
    BGE_DEVICE,
    BGE_MODEL_NAME,
    ZHIPU_API_KEY,
    ZHIPU_EMBEDDING_BASE_URL,
    ZHIPU_EMBEDDING_MODEL,
)
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

openai_embeddings = OpenAIEmbeddings(
    model=ZHIPU_EMBEDDING_MODEL,                     # 智谱模型名
    openai_api_key=ZHIPU_API_KEY,
    openai_api_base=ZHIPU_EMBEDDING_BASE_URL
)

model_name = BGE_MODEL_NAME
model_kwargs = {"device": BGE_DEVICE}
encode_kwargs = {"normalize_embeddings": True}

bge_embedding = HuggingFaceEmbeddings(
    model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
)

import logging
from typing import List

from langchain_core.documents import Document
from langchain_milvus import Milvus, BM25BuiltInFunction
from pymilvus import IndexType, MilvusClient, Function
from pymilvus.client.types import MetricType, DataType, FunctionType

from src.config.settings import (
    COLLECTION_NAME,
    MILVUS_URI,
    RETRIEVER_K,
)
from src.infrastructure.llm.embeddings import bge_embedding


def _keep_milvus_warning(record: logging.LogRecord) -> bool:
    message = record.getMessage()
    return not (
        "Failed to initialize AsyncMilvusClient during Milvus initialization"
        in message
        and "no running event loop" in message
    )


"""
使用langchain_milvus的Milvus 调用add_documents()时候
里面传document后,会自动把document里的page_content和metadata的key注入到数据库表的字段
当然也可以提前创建好表结构,结构里设置自己想注入的字段
"""
class MilvusVectorSave:
    """把新的document数据插入到数据库中"""

    def __init__(self) -> object:
        """自定义collection的索引"""
        self.vector_store_saved: Milvus = None

    def create_collection(self):
        client = MilvusClient(uri=MILVUS_URI)
        dense_dimension = len(bge_embedding.embed_query("dimension probe"))
        schema = client.create_schema()
        schema.add_field(field_name='id', datatype=DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field(field_name='text', datatype=DataType.VARCHAR, max_length=6000, enable_analyzer=True,
                         analyzer_params={"tokenizer": "jieba", "filter": ["cnalphanumonly"]})
        schema.add_field(field_name='category', datatype=DataType.VARCHAR, max_length=1000)
        schema.add_field(field_name='source', datatype=DataType.VARCHAR, max_length=1000)
        schema.add_field(field_name='filename', datatype=DataType.VARCHAR, max_length=1000)
        schema.add_field(field_name='filetype', datatype=DataType.VARCHAR, max_length=1000)
        schema.add_field(field_name='title', datatype=DataType.VARCHAR, max_length=1000)
        schema.add_field(field_name='category_depth', datatype=DataType.INT64)
        schema.add_field(field_name='sparse', datatype=DataType.SPARSE_FLOAT_VECTOR)
        schema.add_field(field_name='dense', datatype=DataType.FLOAT_VECTOR, dim=dense_dimension)

        bm25_function = Function(
            name="text_bm25_emb",  # Function name
            input_field_names=["text"],  # Name of the VARCHAR field containing raw text data
            output_field_names=["sparse"],
            # Name of the SPARSE_FLOAT_VECTOR field reserved to store generated embeddings
            function_type=FunctionType.BM25,  # Set to `BM25`
        )
        schema.add_function(bm25_function)
        index_params = client.prepare_index_params()

        index_params.add_index(
            field_name="sparse",
            index_name="sparse_inverted_index",
            index_type="SPARSE_INVERTED_INDEX",  # Inverted index type for sparse vectors
            metric_type="BM25",
            params={
                "inverted_index_algo": "DAAT_MAXSCORE",
                # Algorithm for building and querying the index. Valid values: DAAT_MAXSCORE, DAAT_WAND, TAAT_NAIVE.
                "bm25_k1": 1.2,
                "bm25_b": 0.75
            },
        )
        index_params.add_index(
            field_name="dense",
            index_name="dense_inverted_index",
            index_type=IndexType.HNSW,  # Inverted index type for sparse vectors
            metric_type=MetricType.IP,
            params={"M": 16, "efConstruction": 64}  # M :邻接节点数, efConstruction: 搜索范围
        )

        if COLLECTION_NAME in client.list_collections():
            # 先释放， 再删除索引，再删除collection
            client.release_collection(collection_name=COLLECTION_NAME)
            client.drop_index(collection_name=COLLECTION_NAME, index_name='sparse_inverted_index')
            client.drop_index(collection_name=COLLECTION_NAME, index_name='dense_inverted_index')
            client.drop_collection(collection_name=COLLECTION_NAME)

        client.create_collection(
            collection_name=COLLECTION_NAME,
            schema=schema,
            index_params=index_params
        )

    def create_connection(self):
        """创建一个Connection： milvus + langchain。pip install  langchain-milvus"""
        upstream_logger = logging.getLogger(
            "langchain_milvus.vectorstores.milvus"
        )
        upstream_logger.addFilter(_keep_milvus_warning)
        try:
            self.vector_store_saved = Milvus(
                embedding_function=bge_embedding,
                collection_name=COLLECTION_NAME,
                builtin_function=BM25BuiltInFunction(),
                vector_field=['dense', 'sparse'],
                consistency_level="Strong",
                auto_id=True,
                connection_args={"uri": MILVUS_URI}
            )
        finally:
            upstream_logger.removeFilter(_keep_milvus_warning)

    def add_documents(self, datas: List[Document]):
        """把新的document保存到Milvus中"""
        self.vector_store_saved.add_documents(datas)


def create_retriever():
    """Create the configured Milvus retriever."""
    vector_store = MilvusVectorSave()
    vector_store.create_connection()
    rrf_reranker = Function(
        name="rrf",
        input_field_names=[],
        function_type=FunctionType.RERANK,
        params={"reranker": "rrf", "k": 100},
    )
    return vector_store.vector_store_saved.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": RETRIEVER_K,
            "reranker": rrf_reranker,
            "expr": 'category == "content"',
        },
    )

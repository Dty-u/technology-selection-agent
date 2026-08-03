from typing import List
from langchain_experimental.text_splitter import SemanticChunker

from src.config.settings import DOCUMENT_CHUNK_LENGTH_THRESHOLD
from src.infrastructure.llm.embeddings import openai_embeddings
from src.observability.logging import log
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_core.documents import Document

"""

"""
class MarkdownParser:
    """
    涓撻棬璐熻矗markdown鏂囦欢鐨勮В鏋愬拰鍒囩墖
    """
    def __init__(self):
        self.text_splitter = SemanticChunker(
            openai_embeddings, breakpoint_threshold_type="percentile"
        )

    def text_chunker(self, datas: List[Document]) -> List[Document]:
        new_docs = []
        for d in datas:
            if len(d.page_content) > DOCUMENT_CHUNK_LENGTH_THRESHOLD:  # 鍐呭瓒呭嚭浜嗛槇鍊硷紝鍒欐寜鐓ц涔夊啀鍒囧壊
                new_docs.extend(self.text_splitter.split_documents([d]))
                continue
            new_docs.append(d)
        return new_docs

    def parse_markdown_to_documents(self, md_file: str, encoding='utf-8') -> List[Document]:
        documents = self.parse_markdown(md_file)
        log.info(f'文件解析后的docs长度: {len(documents)}')

        merged_documents = self.merge_title_content(documents)

        log.info(f'文件合并后的长度: {len(merged_documents)}')

        chunk_documents = self.text_chunker(merged_documents)
        log.info(f'语义切割后的长度: {len(chunk_documents)}')
        return chunk_documents

    def parse_markdown(self, md_file: str) -> List[Document]:
        loader = UnstructuredMarkdownLoader(
            file_path=md_file,
            mode='elements',
            strategy='fast'
        )
        docs = []
        for doc in loader.lazy_load():
            docs.append(doc)

        return docs

    def merge_title_content(self, datas: List[Document]) -> List[Document]:
        merged_data = []
        parent_dict = {}  # 鏄竴涓瓧鍏革紝淇濆瓨鎵€鏈夌殑鐖禿ocument锛?key涓哄綋鍓嶇埗document鐨処D
        for document in datas:
            metadata = document.metadata
            if 'languages' in metadata:
                metadata.pop('languages')

            parent_id = metadata.get('parent_id', None)
            category = metadata.get('category', None)
            element_id = metadata.get('element_id', None)

            if category == 'NarrativeText' and parent_id is None:  # 鏄惁涓猴細鍐呭document
                merged_data.append(document)
            if category == 'Title':
                document.metadata['title'] = document.page_content
                if parent_id in parent_dict:
                    document.page_content = parent_dict[parent_id].page_content + ' -> ' + document.page_content
                parent_dict[element_id] = document
            if category != 'Title' and parent_id:
                parent_dict[parent_id].page_content = parent_dict[parent_id].page_content + ' ' + document.page_content
                parent_dict[parent_id].metadata['category'] = 'content'

        # 澶勭悊瀛楀吀
        if parent_dict is not None:
            merged_data.extend(parent_dict.values())

        return merged_data

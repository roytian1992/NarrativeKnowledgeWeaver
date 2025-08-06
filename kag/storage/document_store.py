# kag/memory/bm25_store.py
import os
import json
from typing import List, Optional, Dict
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from typing import List, Union


def to_documents(data: Union[List[str], List[Document]]) -> List[Document]:
    """将输入转换为标准 Document 列表。
    
    Args:
        data: 文本列表或 Document 列表
    
    Returns:
        List[Document]
    """
    if not data:
        return []

    # 如果是纯文本列表
    if isinstance(data[0], str):
        return [Document(page_content=t, metadata={}) for t in data]
    
    # 如果本身就是 Document 列表
    if isinstance(data[0], Document):
        return data

    raise ValueError("输入类型不支持，必须是 List[str] 或 List[Document]")



class DocumentStore:
    """BM25 文档检索管理器，支持加载/保存文档，执行关键词检索。"""

    def __init__(self, config: Dict):
        """
        Args:
            storage_path: 保存 JSON 文档数据的路径
        """
        self.storage_path = config.storage.document_store_path
        self.docs: List[Document] = []
        self.retriever: Optional[BM25Retriever] = None
        os.makedirs(self.storage_path, exist_ok=True)

        # # 自动加载已有文档（如果存在）
        # if os.path.exists(self.storage_path):
        #     self.load()
        #     self._build_retriever()

    def add_documents(self, docs: List[Document], save: bool = True) -> None:
        """添加文档并重建检索器"""
        self.docs.extend(to_documents(docs))
        # print(self.docs)
        # self._build_retriever()
        if save:
            self.save()

    def _build_retriever(self) -> None:
        """重建 BM25 检索器"""
        self.retriever = BM25Retriever.from_documents(self.docs)
        self.save()

    def save(self) -> None:
        """保存文档到本地 JSON"""
        filename = os.path.join(self.storage_path, "document_store.json")
        with open(filename, "w", encoding="utf-8") as f:
            json.dump([
                {"page_content": d.page_content, "metadata": d.metadata}
                for d in self.docs
            ], f, ensure_ascii=False, indent=2)

    def load(self) -> None:
        """从 JSON 文件加载文档"""
        filename = os.path.join(self.storage_path, "document_store.json")
        with open(filename, "r", encoding="utf-8") as f:
            raw = json.load(f)
        self.docs = [
            Document(page_content=d["page_content"], metadata=d.get("metadata", {}))
            for d in raw
        ]

    def search(self, query: str, k: int = None, filters: Optional[Dict] = None) -> List[Document]:
        """
        执行关键词检索（可选 metadata 筛选）

        Args:
            query: 查询文本
            k: 返回前 k 个结果
            filters: 元信息过滤条件（如 {"source": "doc1"}）
        """
        if not self.retriever:
            raise ValueError("尚未初始化检索器")

        results = self.retriever.get_relevant_documents(query)

        if filters:
            def match(doc: Document) -> bool:
                return all(doc.metadata.get(key) == value for key, value in filters.items())
            results = list(filter(match, results))
        if k:
            return results[:k]
        else:
            return results

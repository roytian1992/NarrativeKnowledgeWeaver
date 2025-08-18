from typing import Dict, Any
import json
from qwen_agent.tools.base import BaseTool, register_tool
from qwen_agent.utils.utils import logger
from retriever.vectordb_retriever import ParentChildRetriever


@register_tool("vdb_search_hierdocs")
class VDBHierdocsSearchTool(BaseTool):
    """父子文档（Parent-Child）检索：句子召回 + 父级聚合 + 重排"""

    name = "vdb_search_hierdocs"
    description = "使用父子文档检索器（Parent-Child Retriever）从向量数据库中获取与查询相关的内容（句子级召回，聚合到父段落/文档后重排）。"
    parameters = [
        {
            "name": "query",
            "type": "string",
            "description": "检索的查询文本",
            "required": True
        },
        {
            "name": "limit",
            "type": "integer",
            "description": "返回条数，默认 5",
            "required": False
        }
    ]

    def __init__(self, document_vector_store, sentence_vector_store, reranker):
        self.retriever = ParentChildRetriever(
            doc_vs=document_vector_store,
            sent_vs=sentence_vector_store,
            reranker=reranker
        )

    def call(self, params: str, **kwargs) -> str:
        logger.info("🔍 调用父子文档检索器 vdb_search_hierdocs")
        params_dict: Dict[str, Any] = json.loads(params)
        query = params_dict.get("query", "")
        limit = int(params_dict.get("limit", 5))

        hits = self.retriever.retrieve(
            query=query,
            ks=20,                 # 子级候选数
            kp=limit * 2,          # 父级候选数
            window=1,              # 句窗拼接
            topn=limit             # 最终返回
        )
        texts = [h.content for h in hits]
        return "相关结果：\n" + "\n--\n".join(texts)


@register_tool("vdb_search_docs")
class VDBDocsSearchTool(BaseTool):
    """文档级检索：粗粒度召回"""

    name = "vdb_search_docs"
    description = "使用文档级检索器从向量数据库中获取粗粒度的相关文档内容。"
    parameters = [
        {
            "name": "query",
            "type": "string",
            "description": "检索的查询文本",
            "required": True
        },
        {
            "name": "limit",
            "type": "integer",
            "description": "返回条数，默认 5",
            "required": False
        }
    ]

    def __init__(self, document_vector_store):
        self.doc_vs = document_vector_store

    def call(self, params: str, **kwargs) -> str:
        logger.info("🔍 调用文档级检索 vdb_search_docs")
        params_dict: Dict[str, Any] = json.loads(params)
        query = params_dict.get("query", "")
        limit = int(params_dict.get("limit", 5))

        results = self.doc_vs.search(query=query, limit=limit)
        texts = [r.content for r in results]
        return "相关结果：\n" + "\n--\n".join(texts)


@register_tool("vdb_get_docs_by_chunk_ids")
class VDBGetDocsByChunkIDsTool(BaseTool):
    """按 chunk_id 直接获取段落/片段"""

    name = "vdb_get_docs_by_chunk_ids"
    description = "根据提供的 chunk_id 列表，从向量数据库中直接获取对应的段落/片段内容。"
    parameters = [
        {
            "name": "ids",
            "type": "array",
            "description": "待获取的 chunk_id 列表",
            "required": True
        }
    ]

    def __init__(self, document_vector_store):
        self.doc_vs = document_vector_store

    def call(self, params: str, **kwargs) -> str:
        logger.info("🔍 按 chunk_id 获取内容 vdb_get_docs_by_chunk_ids")
        params_dict: Dict[str, Any] = json.loads(params)
        ids = params_dict.get("ids") or []

        results = self.doc_vs.search_by_ids(ids)
        texts = [r.content for r in results]
        return "相关结果：\n" + "\n--\n".join(texts)


@register_tool("vdb_search_sentences")
class VDBSentencesSearchTool(BaseTool):
    """句子级检索：细粒度相关语句"""

    name = "vdb_search_sentences"
    description = "使用句子级检索器从向量数据库中获取细粒度的相关语句。"
    parameters = [
        {
            "name": "query",
            "type": "string",
            "description": "检索的查询文本",
            "required": True
        },
        {
            "name": "limit",
            "type": "integer",
            "description": "返回条数，默认 5",
            "required": False
        }
    ]

    def __init__(self, sentence_vector_store):
        self.sent_vs = sentence_vector_store

    def call(self, params: str, **kwargs) -> str:
        logger.info("🔍 调用句子级检索 vdb_search_sentences")
        params_dict: Dict[str, Any] = json.loads(params)
        query = params_dict.get("query", "")
        limit = int(params_dict.get("limit", 5))

        results = self.sent_vs.search(query=query, limit=limit)
        texts = [r.content for r in results]
        return "相关结果：\n" + "\n--\n".join(texts)
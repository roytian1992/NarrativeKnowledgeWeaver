from typing import Dict, Any
import json
from qwen_agent.tools.base import BaseTool, register_tool
from qwen_agent.utils.utils import logger


def _sort_by_similarity_desc(docs):
    """
    Ensure deterministic descending order by metadata.similarity_score when available.
    """
    rows = list(docs or [])
    rows.sort(
        key=lambda d: float(((getattr(d, "metadata", {}) or {}).get("similarity_score", 0.0) or 0.0)),
        reverse=True,
    )
    return rows

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

        results = _sort_by_similarity_desc(self.doc_vs.search(query=query, limit=limit))
        texts = [r.content for r in results]
        if not texts:
            return "No results."
        return "\n--\n".join(texts)


@register_tool("vdb_get_docs_by_document_ids")
class VDBGetDocsByDocumentIDsTool(BaseTool):
    """按 document_id（即 source_documents）直接获取文档片段。"""

    name = "vdb_get_docs_by_document_ids"
    description = (
        "根据提供的 document_id 列表（即图中实体的 source_documents），"
        "从向量库按 metadata.document_id 获取对应原文内容。"
        "一个 document_id 可能对应多个 chunk，工具会返回全部命中的 chunk 文本。"
    )
    parameters = [
        {
            "name": "document_ids",
            "type": "array",
            "description": "待获取的 document_id/source_documents 列表",
            "required": True
        }
    ]

    def __init__(self, document_vector_store):
        self.doc_vs = document_vector_store

    @staticmethod
    def _dedup_ids(items: Any) -> list:
        if not isinstance(items, list):
            return []
        out = []
        seen = set()
        for x in items:
            s = str(x).strip()
            if not s or s in seen:
                continue
            seen.add(s)
            out.append(s)
        return out

    def call(self, params: str, **kwargs) -> str:
        logger.info("🔍 按 document_id 获取内容 vdb_get_docs_by_document_ids")
        params_dict: Dict[str, Any] = json.loads(params)
        ids = params_dict.get("document_ids") or []
        ids = self._dedup_ids(ids)
        if not ids:
            return "请提供非空的 document_ids 列表。"

        results = self.doc_vs.search_by_document_ids(ids)
        texts = [r.content for r in results]
        if not texts:
            return "No results."
        return "\n--\n".join(texts)


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

        results = _sort_by_similarity_desc(self.sent_vs.search(query=query, limit=limit))
        texts = [r.content for r in results]
        if not texts:
            return "No results."
        return "\n--\n".join(texts)

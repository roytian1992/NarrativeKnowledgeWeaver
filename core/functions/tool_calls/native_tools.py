# core/functions/tool_calls/native_tools.py
import json
from typing import Any, Dict, List, Optional

from qwen_agent.tools.base import BaseTool, register_tool
from qwen_agent.utils.utils import logger

from langchain_core.documents import Document
from retriever.sparse_retriever import KeywordBM25Retriever  # ← 前面给的轻封装BM25

# 可选：中文分词
def zh_preprocess(text: str):
    try:
        import re, jieba
        text = (text or "").strip()
        text = re.sub(r"\s+", " ", text)
        return list(jieba.cut(text, cut_all=False))
    except Exception:
        return (text or "").split()

def _meta_clean(m: Dict[str, Any]) -> Dict[str, Any]:
    # 只剔除 None 和 ""；保留 0 / False
    return {k: v for k, v in (m or {}).items() if v is not None and v != ""}

def _format_hits_mandatory(docs: List[Document]) -> str:
    lines = ["检索到以下文档："]
    for i, d in enumerate(docs, 1):
        lines.append(f"序号：{i}")
        lines.append("内容：")
        lines.append((d.page_content or "").strip())
        lines.append("元数据为：")  # ★ 必须打印
        md = _meta_clean(d.metadata or {})
        if md:
            for k, v in md.items():
                lines.append(f"- {k}: {v}")
        # 如果你希望空元数据也标注，可取消下一行注释
        # else:
        #     lines.append("- （空）")
        lines.append("")  # 空行分隔
    return "\n".join(lines)

@register_tool("bm25_search_docs")
class BM25SearchDocsTool(BaseTool):
    """
    基于 LangChain BM25 的稀疏检索（可选 rerank）
    - 初始化：documents(List[Document])、reranker(Optional)
    - 调用：params={"query": str, "k": int?}
    - 输出：严格包含“检索到以下文档 / 内容 / 元数据为”字段
    """

    name = "bm25_search_docs"
    description = "关键词驱动的 BM25 检索工具：对给定文档集进行关键词匹配与相关度排序（可选 rerank 精排）。"
    parameters = [
        {"name": "query", "type": "string", "description": "检索查询文本", "required": True},
        {"name": "k", "type": "integer", "description": "返回条数（默认 10）", "required": False},
    ]

    def __init__(
        self,
        documents: List[Document],
        reranker: Optional[Any] = None,
        *,
        use_zh_preprocess: bool = True,
        overfetch_mult: int = 2,
        k_default: int = 10,
    ):
        self.retriever = KeywordBM25Retriever(
            documents=documents,
            reranker=reranker,
            zh_preprocess=(zh_preprocess if use_zh_preprocess else None),
            overfetch_mult=overfetch_mult,
            k_default=k_default
        )

    def call(self, params: str, **kwargs) -> str:
        logger.info("🔎 调用 bm25_search_docs")
        try:
            p: Dict[str, Any] = json.loads(params or "{}")
        except Exception as e:
            return f"参数解析失败：{e}"

        query = str(p.get("query", "")).strip()
        if not query:
            return "query 不能为空"
        k = int(p.get("k", 10))

        try:
            hits = self.retriever.retrieve(query=query, k=k)
        except Exception as e:
            logger.exception("BM25 检索失败")
            return f"检索失败：{e}"

        if not hits:
            # 即使无结果，也保持“检索到以下文档：”抬头，保持输出结构一致
            return "检索到以下文档：\n（无）"

        return _format_hits_mandatory(hits)

# retriever/bm25_simple.py
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever as LCBM25Retriever

class KeywordBM25Retriever:
    """
    基于 LangChain 的 BM25Retriever 的轻封装：
    - 输入/输出统一使用 langchain_core.documents.Document（page_content + metadata）
    - 可选 reranker：需实现 .rerank(query, documents: List[str], top_n: int, return_documents: bool=False)
      返回 [{"index": i, "relevance_score": float}, ...]
    - 不依赖 rank_bm25.BM25Okapi；简单、稳定
    """

    def __init__(
        self,
        documents: List[Document],
        reranker: Optional[Any] = None,
        *,
        zh_preprocess=None,       # 可传中文分词函数；如不传则用默认分词
        overfetch_mult: int = 2,  # 启用 reranker 时的预取倍数
        k_default: int = 10
    ):
        self.reranker = reranker
        self.overfetch_mult = max(1, int(overfetch_mult))
        self.k_default = k_default

        self._bm25 = LCBM25Retriever.from_documents(
            documents=documents,
            preprocess_func=zh_preprocess
        )
        self._bm25.k = k_default  # 只是默认值；真正检索时会覆盖

    def retrieve(self, query: str, *, k: Optional[int] = None) -> List[Document]:
        """BM25 -> (可选) rerank -> 返回 List[Document]；分数写入 metadata.similarity_score"""
        k = int(k or self.k_default)
        # 1) 先用 BM25 拿候选
        bm25_k = k * self.overfetch_mult if self.reranker else k
        self._bm25.k = bm25_k
        cands: List[Document] = self._bm25.get_relevant_documents(query) or []

        # 2) 如果没 reranker，直接返回（附上来源标签）
        if not self.reranker or not cands:
            out = []
            for d in cands[:k]:
                md = dict(d.metadata or {})
                md["source"] = "bm25"
                # LangChain BM25 不提供分值，这里不填 similarity_score（或你也可以设为 None）
                out.append(Document(page_content=d.page_content, metadata=md))
            return out

        # 3) rerank：对候选文本重排，取前 k
        docs_for_rerank = [d.page_content for d in cands]
        res = self.reranker.rerank(
            query=query,
            documents=docs_for_rerank,
            top_n=min(k, len(docs_for_rerank)),
            return_documents=False
        ) or []
        # 合法过滤 + 按分数排序
        res = sorted(
            (r for r in res if isinstance(r.get("index"), int)),
            key=lambda x: (x.get("relevance_score") or 0.0),
            reverse=True
        )[:k]

        out: List[Document] = []
        for r in res:
            idx = int(r["index"])
            score = float(r.get("relevance_score") or 0.0)
            d = cands[idx]
            md = dict(d.metadata or {})
            md["similarity_score"] = score
            md["source"] = "bm25+rerank"
            out.append(Document(page_content=d.page_content, metadata=md))
        return out

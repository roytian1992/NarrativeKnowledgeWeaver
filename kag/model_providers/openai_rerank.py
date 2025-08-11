from typing import List, Dict, Any, Optional
import requests
from kag.utils.config import KAGConfig

class OpenAIRerankModel:
    """
    OpenAI 兼容的 Rerank 封装（base_url 指到 vLLM，根或 /v1 都可）
    依赖：vLLM 提供的 /v1/rerank（Cohere 风格）
    """

    def __init__(self, config: KAGConfig):
        if not getattr(config, "rerank", None):
            raise ValueError("config.rerank 未配置")

        rr = config.rerank
        self.model_name: str = rr.model_name
        self.base_url: str = rr.base_url              # 例如 http://<IP>:8012  或  http://<IP>:8012/v1
        self.api_key: str = getattr(rr, "api_key", None) or "not-needed"
        self.timeout: int = getattr(rr, "timeout", 60)

        base = (self.base_url or "").rstrip("/")
        self.endpoint = f"{base}/rerank" if base.endswith("/v1") else f"{base}/v1/rerank"

        self.headers = {"Content-Type": "application/json"}
        if self.api_key:
            self.headers["Authorization"] = f"Bearer {self.api_key}"

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_n: Optional[int] = None,
        return_documents: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        返回格式（标准化）：
        [
          {"index": int, "relevance_score": float, "document": str}, ...
        ]
        """
        if not documents:
            return []
        
        payload: Dict[str, Any] = {
            "model": self.model_name,
            "query": query,
            "documents": documents,
            "return_documents": return_documents,
        }
        if top_n is not None:
            payload["top_n"] = int(top_n)

        resp = requests.post(self.endpoint, json=payload, headers=self.headers, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()

        # vLLM/Cohere 风格字段：results
        results = data.get("results") or data.get("data") or []
        out: List[Dict[str, Any]] = []
        for r in results:
            idx = r.get("index")
            score = r.get("relevance_score") or r.get("relevanceScore") or r.get("score")
            doc = r.get("document") if return_documents else None
            if doc is None and isinstance(idx, int) and 0 <= idx < len(documents):
                doc = documents[idx]
            out.append(
                {"index": idx, "relevance_score": float(score) if score is not None else None, "document": doc}
            )
        return out

    def top_indices(self, query: str, documents: List[str], top_n: int) -> List[int]:
        """只要索引，按分数降序返回前 top_n。"""
        res = self.rerank(query, documents, top_n=top_n, return_documents=False)
        res = sorted(res, key=lambda x: (x["relevance_score"] or 0.0), reverse=True)
        return [r["index"] for r in res[:top_n]]

    def score_pair(self, query: str, document: str) -> float:
        """单文档配对打分。"""
        res = self.rerank(query, [document], top_n=1, return_documents=False)
        return float(res[0]["relevance_score"]) if res else 0.0

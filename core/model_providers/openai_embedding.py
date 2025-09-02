from typing import List, Dict, Any, Optional, Union
import chromadb
from chromadb.config import Settings
from core.models.data import Document
from core.utils.config import KAGConfig, EmbeddingConfig
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI
import time, random, string
import math

def _l2_normalize(vecs: List[List[float]]) -> List[List[float]]:
    eps = 1e-12
    out: List[List[float]] = []
    for v in vecs:
        s = math.sqrt(sum(x * x for x in v)) + eps
        inv = 1.0 / s
        out.append([x * inv for x in v])
    return out

def _l2_normalize_1d(v: List[float]) -> List[float]:
    eps = 1e-12
    s = math.sqrt(sum(x * x for x in v)) + eps
    return [x / s for x in v]


def _char_clip(text: str, max_tokens: Optional[int]) -> str:
    """用字符长度近似截断：裁到 max_tokens - 20 个字符；未配置则不截断。"""
    if not text:
        return ""
    if not max_tokens or max_tokens <= 0:
        return text
    limit = int(max_tokens) - 10
    if limit <= 0:
        return ""
    return text[:limit]

class OpenAICompatEmbeddings:
    """
    适配 OpenAI SDK v1 的 embeddings 客户端为 LangChain 风格接口：
    - embed_documents(List[str]) -> List[List[float]]
    - embed_query(str) -> List[float]
    - __call__(List[str]) -> List[List[float]]  （便于 chromadb 直接调用）
    """
    def __init__(
        self,
        client,                # openai.OpenAI 实例
        model: str,
        *,
        dimensions: Optional[int] = None,
        batch_size: int = 128,
        normalize: bool = True,
        doc_prefix: str = "",  # 对 BGE：通常 'passage: '
        qry_prefix: str = "",  # 对 BGE：通常 'query: '
    ):
        self.client = client
        self.model = model
        # self.dimensions = dimensions
        self.batch_size = batch_size
        self.normalize = normalize
        self.doc_prefix = doc_prefix
        self.qry_prefix = qry_prefix
        # max_tokens 不改构造器签名，由外层在实例化后 setattr 传入

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        # 直连：只传 model 和 input（不传 dimensions）
        kwargs = dict(model=self.model, input=texts)
        resp = self.client.embeddings.create(**kwargs)
        return [d.embedding for d in resp.data]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        # 与自测脚本对齐：doc 前缀 + 简单字符截断（max_tokens-20）
        max_tokens = getattr(self, "max_tokens", None)
        inputs = [self.doc_prefix + _char_clip((t or ""), max_tokens) for t in texts]

        all_vecs: List[List[float]] = []
        for i in range(0, len(inputs), self.batch_size):
            batch = inputs[i:i + self.batch_size]
            # 直连调用（不传 dimensions）
            resp = self.client.embeddings.create(model=self.model, input=batch)
            all_vecs.extend([d.embedding for d in resp.data])

        # 与脚本一致的 eps 归一化：norm + 1e-12
        return _l2_normalize(all_vecs) if self.normalize else all_vecs


    def embed_query(self, text: str) -> List[float]:
        # 与你的脚本保持同一路径：构造 query 输入 -> 直接请求 -> 用同样的 eps 归一化
        max_tokens = getattr(self, "max_tokens", None)
        # 如果你之前加了 _char_clip，就保留；没有则去掉下一行的 _char_clip 调用
        try:
            inp = self.qry_prefix + (text or "")
            if ' _char_clip' in globals():
                inp = _char_clip(inp, max_tokens)
        except NameError:
            inp = self.qry_prefix + (text or "")

        # 直连调用（不传 dimensions）
        resp = self.client.embeddings.create(model=self.model, input=[inp])
        vec = resp.data[0].embedding
        return _l2_normalize_1d(vec) if self.normalize else vec

    # 让 chromadb 可以把它当 callable 使用：emb_fn(list[str]) -> list[list[float]]
    def __call__(self, texts: List[str]) -> List[List[float]]:
        return self.embed_documents(texts)



# --- 统一的嵌入封装 ---
# --- 统一的嵌入封装 ---
class OpenAIEmbeddingModel:
    """
    兼容 LangChain Embeddings 接口与原生 chromadb 的 callable：
    - embed_documents(List[str]) -> List[List[float]]
    - embed_query(str) -> List[float]
    - __call__(List[str]) -> List[List[float]]  # 原生 chromadb 可直接当函数用
    """
    def __init__(self, config):
        self.base_url   = config.base_url
        self.api_key    = getattr(config, "api_key", "x")
        self.model      = config.model_name
        self.max_tokens = getattr(config, "max_tokens", None)
        self.dimensions = getattr(config, "dimensions", None)  # 不主动传给 API
        self.normalize  = True
        self.batch_size = getattr(config, "batch_size", 128)

        from openai import OpenAI
        self.cli = OpenAI(base_url=self.base_url, api_key=self.api_key)

        name = (self.model or "").lower()
        # 对带双头前缀的模型使用 'passage:' / 'query:'；Qwen 之类用空前缀
        self._is_dual   = any(k in name for k in ["bge", "gte", "m3"])
        self.doc_prefix = "passage: " if self._is_dual else ""
        self.qry_prefix = "query: "   if self._is_dual else ""

    # 统一的预处理（可选字符级截断，默认足够宽松）
    def _prep(self, texts, prefix):
        outs = []
        for t in texts:
            s = prefix + (t or "")
            if self.max_tokens and self.max_tokens > 0:
                s = s[: self.max_tokens - 10]  # 近似安全截断；通常对 Qwen 不需要
            outs.append(s)
        return outs

    def _l2_norm(self, vec):
        s = (sum(x * x for x in vec) ** 0.5)
        if s < 1e-12:
            return vec[:]  # 零向量直接返回；也可以返回原值避免 NaN
        inv = 1.0 / (s + 1e-12)
        return [x * inv for x in vec]

    def _embed(self, texts: List[str]) -> List[List[float]]:
        # 不传 dimensions，避免部分后端不支持
        resp = self.cli.embeddings.create(model=self.model, input=texts)
        vecs = [d.embedding for d in resp.data]
        if self.normalize:
            vecs = [self._l2_norm(v) for v in vecs]
        return vecs

    # =============== 核心公开接口 ===============

    # 与 SentenceTransformer.encode([...]) 语义一致（文档分支）
    def encode(self, text_or_texts: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        if isinstance(text_or_texts, str):
            inputs = self._prep([text_or_texts], self.doc_prefix)
            return self._embed(inputs)[0]
        else:
            inputs = self._prep(list(text_or_texts), self.doc_prefix)
            # 批处理
            out: List[List[float]] = []
            for i in range(0, len(inputs), self.batch_size):
                out.extend(self._embed(inputs[i:i + self.batch_size]))
            return out

    # 专门的查询分支（双头模型用 query 前缀；其它沿用 doc 前缀）
    def encode_query(self, text_or_texts: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        prefix = self.qry_prefix if self._is_dual else self.doc_prefix
        if isinstance(text_or_texts, str):
            inputs = self._prep([text_or_texts], prefix)
            return self._embed(inputs)[0]
        else:
            inputs = self._prep(list(text_or_texts), prefix)
            out: List[List[float]] = []
            for i in range(0, len(inputs), self.batch_size):
                out.extend(self._embed(inputs[i:i + self.batch_size]))
            return out

    # -------- LangChain Embeddings 接口所需 --------
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """给向量库写入文档用。"""
        if not texts:
            return []
        return self.encode(texts)  # 文档分支

    def embed_query(self, text: str) -> List[float]:
        """给检索时的查询用。"""
        return self.encode_query(text)

    # -------- 原生 chromadb 友好：可当 callable 使用 --------
    def __call__(self, texts: List[str]) -> List[List[float]]:
        return self.embed_documents(texts)



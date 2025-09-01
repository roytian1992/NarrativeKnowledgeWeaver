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
class OpenAIEmbeddingModel:
    def __init__(self, config):
        self.base_url   = config.base_url
        self.api_key    = getattr(config, "api_key", "x")
        self.model      = config.model_name
        self.max_tokens = getattr(config, "max_tokens", None)
        self.dimensions = getattr(config, "dimensions", None)
        self.normalize  = True

        from openai import OpenAI
        self.cli = OpenAI(base_url=self.base_url, api_key=self.api_key)

        name = (self.model or "").lower()
        # 仅对“确实有双头/指令前缀”的模型启用前缀，其余（含 Qwen）一律空前缀
        self._is_dual = any(k in name for k in ["bge", "gte", "m3"])
        self.doc_prefix = "passage: " if self._is_dual else ""
        self.qry_prefix = "query: "   if self._is_dual else ""

    # 统一的预处理（不做奇怪的 input_type；Qwen 走空前缀）
    def _prep(self, texts, prefix):
        outs = []
        for t in texts:
            s = prefix + (t or "")
            # 一般不建议做字符截断；如需保险也让阈值足够大
            if self.max_tokens and self.max_tokens > 0:
                # 可选：保留，但 Qwen 上通常没必要
                s = s  # 不截断或你自己的 tokenizer 截断
            outs.append(s)
        return outs

    def _embed(self, texts):
        resp = self.cli.embeddings.create(model=self.model, input=texts)
        vecs = [d.embedding for d in resp.data]
        if self.normalize:
            # L2 归一
            out = []
            for v in vecs:
                s = (sum(x*x for x in v) ** 0.5) or 1e-12
                out.append([x/s for x in v])
            return out
        return vecs

    # --- 关键：encode() 走“文档分支”，与 ST.encode([...]) 对齐 ---
    def encode(self, text_or_texts):
        if isinstance(text_or_texts, str):
            texts = [text_or_texts]
            inputs = self._prep(texts, self.doc_prefix)  # 文档前缀（Qwen为空）
            vecs = self._embed(inputs)
            return vecs[0]
        else:
            inputs = self._prep(list(text_or_texts), self.doc_prefix)
            return self._embed(inputs)

    # 分离出的查询接口（只有明确需要时才用）
    def encode_query(self, text_or_texts):
        if isinstance(text_or_texts, str):
            texts = [text_or_texts]
            inputs = self._prep(texts, self.qry_prefix if self._is_dual else self.doc_prefix)
            vecs = self._embed(inputs)
            return vecs[0]
        else:
            inputs = self._prep(list(text_or_texts), self.qry_prefix if self._is_dual else self.doc_prefix)
            return self._embed(inputs)


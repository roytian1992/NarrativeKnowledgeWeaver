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
    out = []
    for v in vecs:
        s = math.sqrt(sum(x*x for x in v)) or 1.0
        out.append([x / s for x in v])
    return out

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

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        # 注意：部分服务端不支持 dimensions；支持时再传
        kwargs = dict(model=self.model, input=texts)
        # if self.dimensions is not None:
        #     kwargs["dimensions"] = self.dimensions
        resp = self.client.embeddings.create(**kwargs)
        return [d.embedding for d in resp.data]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        # 前缀（BGE 推荐：docs 用 'passage: '）
        inputs = [self.doc_prefix + (t or "") for t in texts]
        all_vecs: List[List[float]] = []
        for i in range(0, len(inputs), self.batch_size):
            batch = inputs[i:i+self.batch_size]
            all_vecs.extend(self._embed_batch(batch))
        return _l2_normalize(all_vecs) if self.normalize else all_vecs

    def embed_query(self, text: str) -> List[float]:
        # 前缀（BGE 推荐：query 用 'query: '）
        vecs = self.embed_documents([self.qry_prefix + (text or "")])
        return vecs[0]

    # 让 chromadb 可以把它当 callable 使用：emb_fn(list[str]) -> list[list[float]]
    def __call__(self, texts: List[str]) -> List[List[float]]:
        return self.embed_documents(texts)



# --- 统一的嵌入封装 ---
class OpenAIEmbeddingModel:
    """
    统一封装两类后端：
    - local(HF): sentence-transformers
    - openai: langchain_openai.OpenAIEmbeddings（base_url 指向 vLLM /v1）
    """
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.provider = (self.config.provider or "local").lower()
        self.model_name = self.config.model_name
        self.base_url = self.config.base_url
        self.dimensions = self.config.dimensions
        self.api_key = self.config.api_key

        # OpenAI 兼容 API
        kwargs = {"model": self.model_name}
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.base_url:
            kwargs["base_url"] = self.base_url  # 必须包含 /v1
        # if self.dimensions is not None:
        #     kwargs["dimensions"] = self.dimensions  # 仅当服务端支持才会生效
        if "bge-large" in self.model_name.lower():
            client = OpenAI(base_url=self.base_url, api_key=self.api_key)
            self.model = OpenAICompatEmbeddings(
                client=client,
                model=self.model_name,
                # dimensions=self.dimensions,   # 服务端支持才会生效
                batch_size=128,
                normalize=True,               # BGE 推荐归一化
                doc_prefix="passage: ",       # BGE 推荐前缀
                qry_prefix="query: ",
            )
            # self.model = client.embeddings
        else:
            self.model = OpenAIEmbeddings(**kwargs)
        


    def encode(self, x: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        # if "bge" in self.model_name.lower():
        #     if isinstance(x, list):
        #         result =  self.model.create(model=self.model_name, input=x)
        #         return [doc.embedding for doc in result.data]
                
        #     else:
        #         result =  self.model.create(model=self.model_name, input=x)
        #         return [doc.embedding for doc in result.data][0]              # List[float]
            
        # else:
        # 输入单条 -> 返回 1D 向量；输入多条 -> 返回 2D 列表
        if isinstance(x, list):
            return self.model.embed_documents(x)            # List[List[float]]
        else:
            return self.model.embed_query(x)                # List[float]
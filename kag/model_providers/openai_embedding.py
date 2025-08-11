from typing import List, Dict, Any, Optional, Union
import chromadb
from chromadb.config import Settings
from kag.models.data import Document
from kag.utils.config import KAGConfig
from langchain_openai import OpenAIEmbeddings

# --- 统一的嵌入封装 ---
class OpenAIEmbeddingModel:
    """
    统一封装两类后端：
    - local(HF): sentence-transformers
    - openai: langchain_openai.OpenAIEmbeddings（base_url 指向 vLLM /v1）
    """
    def __init__(self, config: KAGConfig):
        self.config = config
        self.provider = (self.config.embedding.provider or "local").lower()
        self.model_name = self.config.embedding.model_name
        self.base_url = self.config.embedding.base_url
        self.dimensions = self.config.embedding.dimensions
        self.api_key = self.config.embedding.api_key

        # OpenAI 兼容 API
        kwargs = {"model": self.model_name}
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.base_url:
            kwargs["base_url"] = self.base_url  # 必须包含 /v1
        # if self.dimensions is not None:
        #     kwargs["dimensions"] = self.dimensions  # 仅当服务端支持才会生效
        self.model = OpenAIEmbeddings(**kwargs)

    def encode(self, x: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        # 输入单条 -> 返回 1D 向量；输入多条 -> 返回 2D 列表
        if isinstance(x, list):
            return self.model.embed_documents(x)            # List[List[float]]
        else:
            return self.model.embed_query(x)                # List[float]

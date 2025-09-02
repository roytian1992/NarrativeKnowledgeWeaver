# core/memory/vector_memory.py

import os
import json
import time
from typing import Dict, Any, List, Optional

from core.memory.base_memory import BaseMemory
from langchain_community.vectorstores import Chroma
from core.utils.config import KAGConfig

class VectorMemory(BaseMemory):
    """向量记忆
    
    使用向量数据库存储记忆，支持语义检索。
    """
    
    def __init__(self, config: KAGConfig, category: str = None):
        """初始化向量记忆
        
        Args:
            config: 记忆配置
        """
        super().__init__(config)
        self.config = config
        self.memory_path = os.path.join(self.config.memory.memory_path, category)
        
        # 确保目录存在
        os.makedirs(self.memory_path, exist_ok=True)
        
        # 初始化向量数据库
        self._init_vector_db()
        
    def _init_vector_db(self) -> None:
        """初始化向量数据库"""
        try:
            if self.config.graph_embedding.provider != "local":
                from core.model_providers.openai_embedding import OpenAIEmbeddingModel
                self.embedding_model = OpenAIEmbeddingModel(self.config.graph_embedding)
                # 初始化向量数据库
                self.vector_db = Chroma(
                    collection_name="memory",
                    embedding_function=self.embedding_model,
                    persist_directory=self.memory_path
                )
            else:
                from sentence_transformers import SentenceTransformer
                self.embedding_model = SentenceTransformer(self.config.graph_embedding.model_name)
                self.vector_db = Chroma(
                    collection_name="memory",
                    embedding_function=self.embedding_model,
                    persist_directory=self.memory_path
                )
            
            # print("向量记忆初始化成功")
        except Exception as e:
            print(f"初始化向量数据库失败: {str(e)}")
            # 创建一个空的记忆列表作为备用
            self.fallback_memory = []
            self.use_fallback = True
        else:
            self.use_fallback = False
    
    def add(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """添加记忆项（显式传入文本和元信息）

        Args:
            text: 用于语义检索的文本（将被向量化）
            metadata: 可选的结构化信息（用于筛选和展示）
        """
        metadata = metadata or {}

        # 添加时间戳（如未提供）
        if "timestamp" not in metadata:
            metadata["timestamp"] = time.time()

        if self.use_fallback:
            metadata["text"] = text  # 保证 fallback 中也有原文
            self.fallback_memory.append(metadata)
            self._save_fallback()
        else:
            try:
                self.vector_db.add_texts(
                    texts=[text],
                    metadatas=[metadata]
                )
                self.vector_db.persist()
            except Exception as e:
                print(f"添加记忆到向量数据库失败: {str(e)}")
                self.use_fallback = True
                self.fallback_memory = []
                self.add(text, metadata)
    
    def get(self, query: Optional[str] = None, k: int = 5) -> List[Dict[str, Any]]:
        """获取记忆项
        
        Args:
            query: 查询字符串，用于检索相关记忆
            k: 返回的记忆项数量
            
        Returns:
            记忆项列表
        """
        if self.use_fallback:
            # 使用备用记忆，直接返回最近的k个记忆
            return self.fallback_memory[-k:]
        else:
            # 使用向量数据库
            try:
                if query:
                    # 相似度检索
                    results = self.vector_db.similarity_search(query, k=k)
                    # return [doc.page_content for doc in results]
                else:
                    # 无查询，返回最近的k个记忆
                    # 注意：这里简化处理，实际上应该按时间戳排序
                    results = self.vector_db.similarity_search("recent memories", k=k)
                    # return [doc.page_content for doc in results]
                return results
            except Exception as e:
                print(f"从向量数据库检索记忆失败: {str(e)}")
                return []
    
    def clear(self) -> None:
        """清空记忆"""
        if self.use_fallback:
            self.fallback_memory = []
            self._save_fallback()
        else:
            try:
                # 删除集合
                self.vector_db.delete_collection()
                # 重新初始化
                self._init_vector_db()
            except Exception as e:
                print(f"清空向量数据库失败: {str(e)}")
    
    def save(self) -> None:
        """保存记忆到磁盘"""
        if self.use_fallback:
            self._save_fallback()
        else:
            try:
                self.vector_db.persist()
            except Exception as e:
                print(f"保存向量数据库失败: {str(e)}")
    
    def load(self) -> None:
        """从磁盘加载记忆"""
        if self.use_fallback:
            self._load_fallback()
        # 向量数据库在初始化时已经加载
    
    def _save_fallback(self) -> None:
        """保存备用记忆到JSON文件"""
        fallback_path = os.path.join(self.memory_path, "fallback_memory.json")
        try:
            with open(fallback_path, "w", encoding="utf-8") as f:
                json.dump(self.fallback_memory, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存备用记忆失败: {str(e)}")
    
    def _load_fallback(self) -> None:
        """从JSON文件加载备用记忆"""
        fallback_path = os.path.join(self.memory_path, "fallback_memory.json")
        if os.path.exists(fallback_path):
            try:
                with open(fallback_path, "r", encoding="utf-8") as f:
                    self.fallback_memory = json.load(f)
            except Exception as e:
                print(f"加载备用记忆失败: {str(e)}")
                self.fallback_memory = []
        else:
            self.fallback_memory = []


"""
配置管理模块
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path
import yaml
from dotenv import load_dotenv


@dataclass
class LLMConfig:
    """LLM配置"""
    provider: str = "openai"
    
    # OpenAI专用字段
    model_name: str = "gpt-3.5-turbo"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.1
    max_tokens: int = 2000
    timeout: int = 60
    
    # Local LLM字段
    model_path: Optional[str] = None
    device: str = "auto"
    max_new_tokens: int = 2000


@dataclass
class ExtractionConfig:
    """信息抽取配置"""
    chunk_size: int = 1000
    chunk_overlap: int = 200
    score_threshold: int = 7
    max_retries: int = 3
    enable_parallel: bool = False
    max_workers: int = 1



@dataclass
class StorageConfig:
    """存储配置"""
    # Neo4j配置
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_username: str = "neo4j"
    neo4j_password: str = "password"
    
    # 向量数据库配置
    vector_store_type: str = "chroma"
    vector_store_path: str = "./data/vector_store"
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

     # 知识图谱抽取结果存储路径
    knowledge_graph_path: str = "./data/knowledge_graph"
    sql_database_path: str = "./data/sql"
    document_store_path: str = "data/document_store"


@dataclass
class ProcessingConfig:
    """处理配置"""
    batch_size: int = 10
    max_workers: int = 4
    enable_parallel: bool = True
    cache_enabled: bool = True
    cache_dir: str = "./data/cache"


@dataclass
class MemoryConfig:
    """记忆配置"""
    enabled: bool = False
    memory_type: str = "vector"  # buffer, vector, summary
    max_token_limit: int = 4000
    memory_path: str = "./data/memory"
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    

@dataclass
class ReflectionConfig:
    """反思配置"""
    enabled: bool = False
    reflection_interval: int = 5  # 每处理多少个文档进行一次反思
    reflection_path: str = "./data/reflection"
    max_reflections: int = 10  # 最多保存多少条反思


@dataclass
class KAGConfig:
    """KAG主配置类"""
    llm: LLMConfig = field(default_factory=LLMConfig)
    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    reflection: ReflectionConfig = field(default_factory=ReflectionConfig)
    
    @classmethod
    def from_env(cls) -> "KAGConfig":
        """从环境变量创建配置"""
        load_dotenv()
        
        config = cls()
        
        # LLM配置
        config.llm.api_key = os.getenv("OPENAI_API_KEY")
        config.llm.base_url = os.getenv("OPENAI_BASE_URL")
        config.llm.model = os.getenv("OPENAI_MODEL", config.llm.model)
        
        # Neo4j配置
        config.storage.neo4j_uri = os.getenv("NEO4J_URI", config.storage.neo4j_uri)
        config.storage.neo4j_username = os.getenv("NEO4J_USERNAME", config.storage.neo4j_username)
        config.storage.neo4j_password = os.getenv("NEO4J_PASSWORD", config.storage.neo4j_password)
        
        # 向量存储配置
        config.storage.vector_store_path = os.getenv("VECTOR_STORE_PATH", config.storage.vector_store_path)
        config.storage.embedding_model = os.getenv("EMBEDDING_MODEL", config.storage.embedding_model)
        
        return config
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "KAGConfig":
        """从YAML文件创建配置"""
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        config = cls()
        
        # 更新LLM配置
        if 'llm' in data:
            for key, value in data['llm'].items():
                if hasattr(config.llm, key):
                    setattr(config.llm, key, value)
        
        # 更新抽取配置
        if 'extraction' in data:
            for key, value in data['extraction'].items():
                if hasattr(config.extraction, key):
                    setattr(config.extraction, key, value)
        
        # 更新存储配置
        if 'storage' in data:
            for key, value in data['storage'].items():
                if hasattr(config.storage, key):
                    setattr(config.storage, key, value)
        
        # 更新处理配置
        if 'processing' in data:
            for key, value in data['processing'].items():
                if hasattr(config.processing, key):
                    setattr(config.processing, key, value)
                    
        # 更新记忆配置
        if 'memory' in data:
            for key, value in data['memory'].items():
                if hasattr(config.memory, key):
                    setattr(config.memory, key, value)
                    
        # 更新反思配置
        if 'reflection' in data:
            for key, value in data['reflection'].items():
                if hasattr(config.reflection, key):
                    setattr(config.reflection, key, value)
        
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'llm': {
                'provider': self.llm.provider,
                'model': self.llm.model,
                'temperature': self.llm.temperature,
                'max_tokens': self.llm.max_tokens,
                'timeout': self.llm.timeout,
                'model_name': self.llm.model_name,
                'model_path': self.llm.model_path,
                'device': self.llm.device,
                'max_new_tokens': self.llm.max_new_tokens,
            },
            'extraction': {
                'chunk_size': self.extraction.chunk_size,
                'chunk_overlap': self.extraction.chunk_overlap,
                'score_threshold': self.extraction.score_threshold,
                'max_retries': self.extraction.max_retries,
                'enable_parallel': self.extraction.enable_parallel,
                'max_workers': self.extraction.max_workers
            },
            'storage': {
                'neo4j_uri': self.storage.neo4j_uri,
                'neo4j_username': self.storage.neo4j_username,
                'vector_store_type': self.storage.vector_store_type,
                'vector_store_path': self.storage.vector_store_path,
                'embedding_model_name': self.storage.embedding_model_name,
                'knowledge_graph_path': self.storage.knowledge_graph_path,
                'document_store_path': self.storage.document_store_path
            },
            'processing': {
                'batch_size': self.processing.batch_size,
                'max_workers': self.processing.max_workers,
                'enable_parallel': self.processing.enable_parallel,
                'cache_enabled': self.processing.cache_enabled,
                'cache_dir': self.processing.cache_dir,
            },
            'memory': {
                'enabled': self.memory.enabled,
                'memory_type': self.memory.memory_type,
                'max_token_limit': self.memory.max_token_limit,
                'memory_path': self.memory.memory_path,
                'embedding_model_name': self.memory.embedding_model_name,
                
            },
            'reflection': {
                'enabled': self.reflection.enabled,
                'reflection_interval': self.reflection.reflection_interval,
                'reflection_path': self.reflection.reflection_path,
                'max_reflections': self.reflection.max_reflections,
            }
        }
    
    def save_yaml(self, yaml_path: str) -> None:
        """保存为YAML文件"""
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, allow_unicode=True)


def get_config() -> KAGConfig:
    """获取默认配置"""
    return KAGConfig.from_env()


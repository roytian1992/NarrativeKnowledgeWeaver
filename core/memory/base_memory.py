# kag/memory/base_memory.py

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional

from core.utils.config import KAGConfig, MemoryConfig

class BaseMemory(ABC):
    """记忆模块基类
    
    所有记忆实现都应该继承这个基类，提供统一的接口。
    """
    
    def __init__(self, config: MemoryConfig):
        """初始化记忆模块
        
        Args:
            config: 记忆配置
        """
        self.config = config
    
    @abstractmethod
    def add(self, item: Dict[str, Any]) -> None:
        """添加记忆项
        
        Args:
            item: 记忆项，包含任意键值对
        """
        pass
    
    @abstractmethod
    def get(self, query: Optional[str] = None, k: int = 5) -> List[Dict[str, Any]]:
        """获取记忆项
        
        Args:
            query: 查询字符串，用于检索相关记忆
            k: 返回的记忆项数量
            
        Returns:
            记忆项列表
        """
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """清空记忆"""
        pass
    
    @abstractmethod
    def save(self) -> None:
        """保存记忆到磁盘"""
        pass
    
    @abstractmethod
    def load(self) -> None:
        """从磁盘加载记忆"""
        pass


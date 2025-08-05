# kag/memory/buffer_memory.py

import os
import json
import time
from typing import Dict, Any, List, Optional

from kag.memory.base_memory import BaseMemory
from kag.utils.config import MemoryConfig

class BufferMemory(BaseMemory):
    """缓冲记忆
    
    简单的列表记忆，保存最近的记忆项。
    """
    
    def __init__(self, config: MemoryConfig):
        """初始化缓冲记忆
        
        Args:
            config: 记忆配置
        """
        super().__init__(config)
        self.buffer: List[Dict[str, Any]] = []
        self.max_token_limit = config.max_token_limit
        self.memory_path = os.path.join(config.memory_path, "buffer_memory.json")
        
        # 确保目录存在
        os.makedirs(os.path.dirname(self.memory_path), exist_ok=True)
        
        # 尝试加载记忆
        self.load()
    
    def add(self, item: Dict[str, Any]) -> None:
        """添加记忆项
        
        Args:
            item: 记忆项，包含任意键值对
        """
        # 添加时间戳
        if "timestamp" not in item:
            item["timestamp"] = time.time()
            
        # 添加到缓冲区
        self.buffer.append(item)
        
        # 如果超过最大长度，移除最旧的记忆
        while self._estimate_tokens() > self.max_token_limit and len(self.buffer) > 1:
            self.buffer.pop(0)
            
        # 保存记忆
        self.save()
    
    def get(self, query: Optional[str] = None, k: int = 5) -> List[Dict[str, Any]]:
        """获取记忆项
        
        Args:
            query: 查询字符串，在缓冲记忆中被忽略
            k: 返回的记忆项数量
            
        Returns:
            记忆项列表
        """
        # 缓冲记忆直接返回最近的k个记忆
        return self.buffer[-k:]
    
    def clear(self) -> None:
        """清空记忆"""
        self.buffer = []
        self.save()
    
    def save(self) -> None:
        """保存记忆到磁盘"""
        try:
            with open(self.memory_path, "w", encoding="utf-8") as f:
                json.dump(self.buffer, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存记忆失败: {str(e)}")
    
    def load(self) -> None:
        """从磁盘加载记忆"""
        if os.path.exists(self.memory_path):
            try:
                with open(self.memory_path, "r", encoding="utf-8") as f:
                    self.buffer = json.load(f)
            except Exception as e:
                print(f"加载记忆失败: {str(e)}")
                self.buffer = []
        else:
            self.buffer = []
    
    def _estimate_tokens(self) -> int:
        """估计当前缓冲区的token数量
        
        Returns:
            估计的token数量
        """
        # 简单估计：每4个字符约为1个token
        buffer_str = json.dumps(self.buffer, ensure_ascii=False)
        return len(buffer_str) // 4


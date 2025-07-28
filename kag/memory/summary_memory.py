# kag/memory/summary_memory.py

import os
import json
import time
from typing import Dict, Any, List, Optional

from kag.memory.base_memory import BaseMemory
from kag.utils.config import MemoryConfig

class SummaryMemory(BaseMemory):
    """摘要记忆
    
    定期对记忆进行摘要，保存摘要和最近的记忆项。
    """
    
    def __init__(self, config: MemoryConfig):
        """初始化摘要记忆
        
        Args:
            config: 记忆配置
        """
        super().__init__(config)
        self.buffer: List[Dict[str, Any]] = []
        self.summaries: List[Dict[str, Any]] = []
        self.max_token_limit = config.max_token_limit
        self.memory_path = os.path.join(config.memory_path, "summary_memory.json")
        
        # 确保目录存在
        os.makedirs(os.path.dirname(self.memory_path), exist_ok=True)
        
        # 尝试加载记忆
        self.load()
        
        # 记录上次摘要的时间
        self.last_summary_time = time.time()
        
        # 摘要间隔（秒）
        self.summary_interval = 3600  # 1小时
        
        # 摘要阈值（记忆项数量）
        self.summary_threshold = 20
    
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
        
        # 检查是否需要生成摘要
        current_time = time.time()
        if (len(self.buffer) >= self.summary_threshold or 
            current_time - self.last_summary_time >= self.summary_interval):
            self._generate_summary()
            
        # 保存记忆
        self.save()
    
    def get(self, query: Optional[str] = None, k: int = 5) -> List[Dict[str, Any]]:
        """获取记忆项
        
        Args:
            query: 查询字符串，在摘要记忆中被忽略
            k: 返回的记忆项数量
            
        Returns:
            记忆项列表，包括摘要和最近的记忆
        """
        # 返回所有摘要和最近的记忆
        result = self.summaries.copy()
        
        # 添加最近的记忆，但不超过k个
        recent_memories = self.buffer[-k:]
        result.extend(recent_memories)
        
        return result
    
    def clear(self) -> None:
        """清空记忆"""
        self.buffer = []
        self.summaries = []
        self.last_summary_time = time.time()
        self.save()
    
    def save(self) -> None:
        """保存记忆到磁盘"""
        try:
            data = {
                "buffer": self.buffer,
                "summaries": self.summaries,
                "last_summary_time": self.last_summary_time
            }
            with open(self.memory_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存记忆失败: {str(e)}")
    
    def load(self) -> None:
        """从磁盘加载记忆"""
        if os.path.exists(self.memory_path):
            try:
                with open(self.memory_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.buffer = data.get("buffer", [])
                    self.summaries = data.get("summaries", [])
                    self.last_summary_time = data.get("last_summary_time", time.time())
            except Exception as e:
                print(f"加载记忆失败: {str(e)}")
                self.buffer = []
                self.summaries = []
                self.last_summary_time = time.time()
        else:
            self.buffer = []
            self.summaries = []
            self.last_summary_time = time.time()
    
    def _generate_summary(self) -> None:
        """生成记忆摘要"""
        if not self.buffer:
            return
            
        try:
            # 在实际实现中，这里应该调用LLM生成摘要
            # 这里简化处理，直接创建一个摘要记录
            summary = {
                "type": "summary",
                "timestamp": time.time(),
                "content": f"记忆摘要 ({len(self.buffer)} 项)",
                "count": len(self.buffer)
            }
            
            # 添加摘要
            self.summaries.append(summary)
            
            # 清空缓冲区
            self.buffer = []
            
            # 更新上次摘要时间
            self.last_summary_time = time.time()
            
            print(f"生成记忆摘要: {summary['content']}")
        except Exception as e:
            print(f"生成摘要失败: {str(e)}")
    
    def summarize_with_llm(self, llm: Any) -> None:
        """使用LLM生成摘要
        
        Args:
            llm: 语言模型
        """
        if not self.buffer:
            return
            
        try:
            # 将缓冲区记忆转换为文本
            memories_text = "\n".join([
                f"- {json.dumps(item, ensure_ascii=False)}"
                for item in self.buffer
            ])
            
            # 构建提示词
            prompt = f"""请对以下记忆项进行摘要，提取关键信息：

{memories_text}

请生成一个简洁的摘要，包含这些记忆中的关键信息和模式。
"""
            
            # 调用LLM生成摘要
            if hasattr(llm, "invoke"):
                # LangChain风格
                response = llm.invoke(prompt)
                summary_text = response.content if hasattr(response, "content") else str(response)
            elif hasattr(llm, "chat"):
                # Qwen-Agent风格
                messages = [
                    {"role": "system", "content": "你是一个记忆摘要助手，擅长提取关键信息并生成简洁的摘要。"},
                    {"role": "user", "content": prompt}
                ]
                summary_text = llm.chat(messages)
            else:
                # 备用方案
                summary_text = f"记忆摘要 ({len(self.buffer)} 项)"
            
            # 创建摘要记录
            summary = {
                "type": "summary",
                "timestamp": time.time(),
                "content": summary_text,
                "count": len(self.buffer)
            }
            
            # 添加摘要
            self.summaries.append(summary)
            
            # 清空缓冲区
            self.buffer = []
            
            # 更新上次摘要时间
            self.last_summary_time = time.time()
            
            print(f"使用LLM生成记忆摘要成功")
        except Exception as e:
            print(f"使用LLM生成摘要失败: {str(e)}")
            # 回退到简单摘要
            self._generate_summary()


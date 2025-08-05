# kag/llm/llm_manager.py
from typing import Tuple, Any, Optional
from langchain_openai import ChatOpenAI
from ..utils.config import KAGConfig
from .openai_llm import OpenAIQwenLLM


class LLMManager:
    """LLM管理器
    
    根据配置创建不同的LLM实例，支持：
    - OpenAI API (GPT-3.5, GPT-4)
    - 本地Qwen3模型
    """
    
    def __init__(self, config: KAGConfig):
        self.config = config
        self.llm = None
        self.is_chat_model = False
    

    def get_llm(self, force_device=None):
        if self.llm is None:
            provider = self.config.llm.provider.lower()
            # if provider == "qwen3":
            #     from .qwen3_llm import QwenFnCallLLM
            #     self.llm = QwenFnCallLLM(self.config, force_device)
            if provider == "openai":
                self.llm = OpenAIQwenLLM(self.config)
                self.is_chat_model = True
            else:
                raise ValueError(f"Unsupported LLM provider: {provider}")

        return self.llm

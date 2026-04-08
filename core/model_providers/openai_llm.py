import asyncio
import re
from typing import Any, Dict, List, Optional

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from openai import OpenAI
from pydantic import PrivateAttr

ROLE_MAP = {
    "system": SystemMessage,
    "user": HumanMessage,
    "assistant": AIMessage,
}


def _message_get(message: Any, key: str, default: Any = None) -> Any:
    if isinstance(message, dict):
        return message.get(key, default)
    getter = getattr(message, "get", None)
    if callable(getter):
        try:
            return getter(key, default)
        except Exception:
            pass
    return getattr(message, key, default)



class OpenAILLM(ChatOpenAI):
    """仅接收 cfg 的 LLM 封装（<think> 清洗）"""

    _client: OpenAI = PrivateAttr()

    # ---------- init ----------
    def __init__(self, cfg, *, llm_profile: str = "llm", llm_config=None, **kwargs):
        if llm_config is None and hasattr(cfg, "get_llm_profile"):
            llm_cfg = cfg.get_llm_profile(llm_profile)
        else:
            llm_cfg = llm_config or cfg.llm

        # 1) 初始化 ChatOpenAI（父类）
        super().__init__(
            openai_api_base=llm_cfg.base_url,
            openai_api_key=llm_cfg.api_key or "EMPTY",
            model=llm_cfg.model_name,
            temperature=llm_cfg.temperature,
            max_tokens=llm_cfg.max_tokens,
            **kwargs,
        )

        # 2) 原生 OpenAI client（给 run() 用）
        self._client = OpenAI(
            base_url=llm_cfg.base_url,
            api_key=llm_cfg.api_key or "EMPTY"
        )

    # ---------- LangChain 组件走这里 ----------
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager=None,
        **kwargs,
    ):
        # 真正请求
        gens = super()._generate(messages=messages, stop=stop, run_manager=run_manager, **kwargs)

        # 去掉 <think> 标签（兼容偶发残留）
        for g in gens.generations:
            g.message.content = self._remove_think_tags(g.message.content)

        return gens

    def run(
        self,
        messages: List[Any],
        enable_thinking: Optional[bool] = None,
        **kwargs,
    ) -> List[Dict[str, str]]:
        """接受消息列表，返回与旧逻辑兼容的 assistant 消息列表。"""
        del enable_thinking
        lc_messages: List[BaseMessage] = []
        for m in messages:
            role = _message_get(m, "role")
            content = _message_get(m, "content", "")
            if role not in ROLE_MAP:
                raise ValueError(f"Unsupported role: {role}")
            lc_messages.append(ROLE_MAP[role](content=content))

        output = self._generate(messages=lc_messages, **kwargs)
        content = self._remove_think_tags(output.generations[0].message.content)
        return [{"role": "assistant", "content": content}]

    async def arun(self, messages, enable_thinking: Optional[bool] = None, **kwargs):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,  # 默认线程池
            lambda: self.run(messages, enable_thinking=enable_thinking, **kwargs)
        )

    # ---------- 工具 ----------
    @staticmethod
    def _remove_think_tags(txt: str) -> str:
        return re.sub(r"<think>.*?</think>", "", txt, flags=re.DOTALL).strip()

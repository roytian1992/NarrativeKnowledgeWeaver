# kag/llm/openai_llm.py
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from openai import OpenAI
from pydantic import PrivateAttr
from typing import List, Optional
import copy, re
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from openai import OpenAI
from pydantic import PrivateAttr
from typing import List, Optional
import copy, re
from qwen_agent.llm.schema import Message, ASSISTANT, USER, SYSTEM
import asyncio
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

ROLE_MAP = {
    "system": SystemMessage,
    "user": HumanMessage,
    "assistant": AIMessage,
}



class OpenAILLM(ChatOpenAI):
    """仅接收 cfg 的 Qwen 封装（/no_think 注入 & <think> 清洗）"""

    _default_thinking: bool = PrivateAttr(default=True)
    _client: OpenAI = PrivateAttr()

    # ---------- init ----------
    def __init__(self, cfg, **kwargs):
        llm_cfg = cfg.llm

        # 1) 初始化 ChatOpenAI（父类）
        super().__init__(
            openai_api_base=llm_cfg.base_url,
            openai_api_key=llm_cfg.api_key or "EMPTY",
            model=llm_cfg.model_name,
            temperature=llm_cfg.temperature,
            max_tokens=llm_cfg.max_tokens,
            **kwargs,
        )

        # 2) 默认思考开关
        self._default_thinking = getattr(llm_cfg, "enable_thinking", True)
        # self._default_thinking = False
        # 3) 原生 OpenAI client（给 run() 用）
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
        thinking = kwargs.pop("enable_thinking", self._default_thinking)

        # 注入 /no_think
        if not thinking and messages:
            new_msgs, injected = [], False
            for msg in messages:
                if not injected and msg.type == "human":
                    new_msgs.append(
                        HumanMessage(content=self._inject_no_think(msg.content))
                    )
                    injected = True
                else:
                    new_msgs.append(msg)
            messages = new_msgs

        # 真正请求
        gens = super()._generate(messages=messages, stop=stop, run_manager=run_manager, **kwargs)

        # 可选：如需去掉 <think> 标签，取消下一行注释
        for g in gens.generations:
            g.message.content = self._remove_think_tags(g.message.content)

        return gens

    def run(self, messages: List[dict], enable_thinking: Optional[bool] = None, **kwargs) -> str:
        """高层封装的调用方法，接受 list-of-dict 格式消息，返回纯文本内容"""
        lc_messages = []
        for m in messages:
            role = m.get("role")
            content = m.get("content", "")
            if role not in ROLE_MAP:
                raise ValueError(f"Unsupported role: {role}")
            lc_messages.append(ROLE_MAP[role](content=content))

        # 使用 enable_thinking 软控制（覆盖默认值）
        kwargs["enable_thinking"] = (
            self._default_thinking if enable_thinking is None else enable_thinking
        )

        # 执行生成（调用 _generate）
        output = self._generate(messages=lc_messages, **kwargs)
        # print("LLM output:", output)
        content = output.generations[0].message.content.split("</think>")
        if len(content) == 1:
            result = [Message(role=ASSISTANT, content=content[0])]
        else:
            result = [Message(role=ASSISTANT, content=content[1])] 
            
        return result
    
    async def arun(self, messages, enable_thinking=None, **kwargs):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,  # 默认线程池
            lambda: self.run(messages, enable_thinking=enable_thinking, **kwargs)
        )


    # ---------- 工具 ----------
    @staticmethod
    def _inject_no_think(text: str) -> str:
        text = text.lstrip()
        return text if text.startswith("/no_think") else "/no_think " + text

    @staticmethod
    def _remove_think_tags(txt: str) -> str:
        return re.sub(r"<think>.*?</think>", "", txt, flags=re.DOTALL).strip()


# kag/llm/qwen3_llm.py
import os
from typing import List, Dict, Any, Iterator, Literal, Optional
from langchain_core.language_models import LLM
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage

from qwen_agent.llm.base import BaseChatModel as QwenBaseChatModel
from qwen_agent.llm.schema import Message, ASSISTANT, USER, SYSTEM
from qwen_agent.llm.function_calling import BaseFnCallModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from transformers import TextIteratorStreamer
import threading
from qwen_agent.agents.fncall_agent import FnCallAgent


class WrappedQwenLLM(LLM):
    # 给SQL Agent使用的
    def __init__(self, qwen_fncall_llm):
        super().__init__()
        object.__setattr__(self, "_qwen", qwen_fncall_llm)

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        messages = [Message(role=USER, content=prompt)]
        responses = self._qwen._chat_no_stream(messages, generate_cfg={
            "temperature": 0.7,
            "max_new_tokens": 1024,
            "enable_thinking": False  # ✅ 禁用思考过程
        })
        return responses[0].content

    def run(self, messages: List[Dict], enable_thinking=False) -> str:
        # messages = [Message(role=USER, content=prompt)]
        responses = self._qwen._chat_no_stream(messages, generate_cfg={
            "temperature": 0.2,
            "max_new_tokens": 32768,
            "enable_thinking": enable_thinking # ✅ 思考过程
        })
        return responses[0].content

    @property
    def _llm_type(self) -> str:
        return "wrapped-qwen-fncall"


class NonStreamingFnCallAgent(FnCallAgent):
    # 工具调用Agent使用的
    def _run(self, messages: List[Message], lang: Literal['en', 'zh'] = 'zh', **kwargs) -> List[Message]:
        messages = copy.deepcopy(messages)
        response = []

        extra_generate_cfg = {'lang': lang, "max_new_tokens": 2048}
        if kwargs.get('seed') is not None:
            extra_generate_cfg['seed'] = kwargs['seed']

        output_batches = self._call_llm(
            messages=messages,
            functions=[func.function for func in self.function_map.values()],
            stream=False,
            extra_generate_cfg=extra_generate_cfg
        )
        # 非流式，直接取一批消息（通常是 assistant 回复）
        output = output_batches[0] if isinstance(output_batches, list) and output_batches else []

        response.append(output)
        used_any_tool = False

        use_tool, tool_name, tool_args, _ = self._detect_tool(output)
        if use_tool:
            tool_result = self._call_tool(tool_name, tool_args, messages=messages, **kwargs)
            fn_msg = Message(
                role=FUNCTION,
                name=tool_name,
                content=tool_result,
            )
            response.append(fn_msg)
            used_any_tool = True

        return response


class QwenFnCallLLM(BaseFnCallModel):
    """
    支持 Function Calling 的本地 Qwen 模型（基于 qwen_agent 的 BaseFnCallModel）
    """

    def __init__(self, config, force_device: Optional[str] = None):
        super().__init__({
            "model": config.llm.model_name,
            "model_type": "qwen_local",
            "generate_cfg": {
                "max_input_tokens": config.llm.max_tokens,
                "temperature": config.llm.temperature,
                "fncall_prompt_type": "qwen"
            }
        })

        model_path = config.llm.model_path or config.llm.model_name

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if force_device:
            self.hf_model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map=None,  # 👈 禁用 auto
                torch_dtype=torch.float16,
                trust_remote_code=True
            ).to(force_device).eval()
        else:
            self.hf_model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map=config.llm.device or "auto",
                torch_dtype=torch.float16,
                trust_remote_code=True
            ).eval()

    def _chat_no_stream(self, messages: List[Message], enable_thinking=False, generate_cfg: Dict = {}) -> List[Message]:
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.hf_model.device)
        outputs = self.hf_model.generate(
            **inputs,
            temperature=generate_cfg.get("temperature", 0.2),
            max_new_tokens=generate_cfg.get("max_new_tokens", 2048),
        )
        output_ids = outputs[0][inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        return [Message(role=ASSISTANT, content=text)]

    def _chat_stream(self, messages: List[Message], delta_stream: bool, generate_cfg: Dict = {}) -> Iterator[List[Message]]:
        raise self._chat_no_stream(messages, generate_cfg)

        
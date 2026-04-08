import asyncio
import re
from typing import Any, Dict, List, Optional, Tuple

from agents import AsyncOpenAI, ModelSettings, OpenAIChatCompletionsModel
from agents.models.interface import ModelTracing
from langchain_core.messages import AIMessage, BaseMessage


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


def _message_role_and_text(message: Any) -> Tuple[str, str]:
    role = str(_message_get(message, "role", "") or "").strip().lower()
    content = _message_get(message, "content", "")
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text_part = str(item.get("text") or item.get("content") or "").strip()
                if text_part:
                    parts.append(text_part)
            else:
                parts.append(str(item))
        text = "\n".join(part for part in parts if part)
    elif content is None:
        text = ""
    else:
        text = str(content)
    return role, text


class OpenAIAgentsLLM:
    """
    Lightweight adapter that uses OpenAI Agents SDK as the chat-completions caller.

    Important:
    - We do not rely on SDK-native tool calling because the current local endpoint
      rejects `tool_choice=auto`.
    - We also avoid `Runner.run_sync()` per turn because this project already has
      its own manual tool loop; going through Runner for every LLM call adds heavy
      orchestration overhead.
    - Tool execution remains in the existing manual JSON tool loop runtime.
    """

    def __init__(self, cfg, *, llm_profile: str = "llm", llm_config=None, **kwargs) -> None:
        del kwargs
        if llm_config is None and hasattr(cfg, "get_llm_profile"):
            llm_cfg = cfg.get_llm_profile(llm_profile)
        else:
            llm_cfg = llm_config or cfg.llm

        self.model_name = str(llm_cfg.model_name or "").strip()
        self.temperature = float(getattr(llm_cfg, "temperature", 0.0) or 0.0)
        self.max_tokens = int(getattr(llm_cfg, "max_tokens", 0) or 0) or None
        self.timeout = float(getattr(llm_cfg, "timeout", 60) or 60)
        self.base_url = str(llm_cfg.base_url or "").strip()
        self.api_key = str(llm_cfg.api_key or "EMPTY").strip() or "EMPTY"

        self._client = AsyncOpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.timeout,
        )
        self._model = OpenAIChatCompletionsModel(
            model=self.model_name,
            openai_client=self._client,
        )
        self._model_settings = ModelSettings(
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            parallel_tool_calls=False,
        )

    def _prepare_agent_io(self, messages: List[Any]) -> Tuple[str, str]:
        system_parts: List[str] = []
        convo_parts: List[str] = []
        for message in list(messages or []):
            role, text = _message_role_and_text(message)
            text = str(text or "").strip()
            if not text:
                continue
            if role == "system":
                system_parts.append(text)
            elif role == "user":
                convo_parts.append(f"User:\n{text}")
            elif role == "assistant":
                convo_parts.append(f"Assistant:\n{text}")
            elif role in {"tool", "function"}:
                name = str(_message_get(message, "name", "") or "tool").strip()
                convo_parts.append(f"Tool result ({name}):\n{text}")
            else:
                convo_parts.append(f"{role or 'Message'}:\n{text}")

        instructions = "\n\n".join(part for part in system_parts if part).strip()
        if not instructions:
            instructions = "You are a concise assistant."
        prompt = "\n\n".join(part for part in convo_parts if part).strip()
        if not prompt:
            prompt = "Respond briefly."
        return instructions, prompt

    async def _invoke_async(self, messages: List[BaseMessage], **kwargs) -> AIMessage:
        kwargs.pop("max_turns", None)
        instructions, prompt = self._prepare_agent_io(messages)
        response = await self._model.get_response(
            system_instructions=instructions,
            input=prompt,
            model_settings=self._model_settings,
            tools=[],
            output_schema=None,
            handoffs=[],
            tracing=ModelTracing.DISABLED,
            previous_response_id=None,
            conversation_id=None,
            prompt=None,
        )
        content_parts: List[str] = []
        for item in list(getattr(response, "output", []) or []):
            if str(getattr(item, "type", "") or "").strip() != "message":
                continue
            for content in list(getattr(item, "content", []) or []):
                text = str(getattr(content, "text", "") or "").strip()
                if text:
                    content_parts.append(text)
        content = self._remove_think_tags("\n".join(content_parts)).strip()
        return AIMessage(content=content)

    def invoke(self, messages: List[BaseMessage], **kwargs) -> AIMessage:
        return asyncio.run(self._invoke_async(messages, **kwargs))

    def run(
        self,
        messages: List[Any],
        enable_thinking: Optional[bool] = None,
        **kwargs,
    ) -> List[Dict[str, str]]:
        del enable_thinking
        response = self.invoke(messages, **kwargs)
        return [{"role": "assistant", "content": str(response.content or "")}]

    async def arun(self, messages, enable_thinking: Optional[bool] = None, **kwargs):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.run(messages, enable_thinking=enable_thinking, **kwargs),
        )

    @staticmethod
    def _remove_think_tags(txt: str) -> str:
        return re.sub(r"<think>.*?</think>", "", str(txt or ""), flags=re.DOTALL).strip()

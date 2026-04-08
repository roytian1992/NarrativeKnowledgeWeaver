from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, Optional, Sequence

from core.agent.retrieval.langgraph_runtime import LangGraphAssistantRuntime
from core.model_providers.openai_llm import OpenAILLM

from . import fncall_agent


def _coerce_llm_config(llm: Any) -> Any:
    if hasattr(llm, "invoke") and hasattr(llm, "bind_tools"):
        return llm
    if not isinstance(llm, dict):
        raise TypeError(f"Unsupported llm config type for qwen-agent compatibility: {type(llm)!r}")

    llm_cfg = SimpleNamespace(
        model_name=str(llm.get("model") or llm.get("model_name") or "").strip(),
        base_url=str(llm.get("model_server") or llm.get("base_url") or "").strip() or None,
        api_key=str(llm.get("api_key") or "").strip() or None,
        temperature=float(llm.get("temperature", 0.0) or 0.0),
        max_tokens=int(llm.get("max_tokens", 8192) or 8192),
    )
    cfg = SimpleNamespace(llm=llm_cfg)
    return OpenAILLM(cfg, llm_config=llm_cfg)


class Assistant:
    """
    Minimal compatibility wrapper for the legacy qwen-agent Assistant API.

    It preserves the old constructor surface while delegating execution to the
    project's current LangGraph-based tool-calling runtime.
    """

    def __init__(
        self,
        *,
        function_list: Sequence[Any],
        llm: Any,
        system_message: str = "",
        rag_cfg: Optional[Dict[str, Any]] = None,
        **_: Any,
    ) -> None:
        self.function_list = list(function_list or [])
        self.system_message = str(system_message or "")
        self.rag_cfg = dict(rag_cfg or {})
        self.llm = _coerce_llm_config(llm)
        self._runtime = LangGraphAssistantRuntime(
            function_list=self.function_list,
            llm=self.llm,
            system_message=self.system_message,
            rag_cfg=self.rag_cfg,
        )

    def run_nonstream(self, messages, lang: str = "zh", **kwargs):
        if "max_llm_calls_per_run" not in kwargs:
            kwargs["max_llm_calls_per_run"] = int(getattr(fncall_agent, "MAX_LLM_CALL_PER_RUN", 8) or 8)
        return self._runtime.run_nonstream(messages=messages, lang=lang, **kwargs)


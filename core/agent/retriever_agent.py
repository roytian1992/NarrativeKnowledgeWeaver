from __future__ import annotations

import importlib
import os
from typing import Any, Literal, cast

from core.utils.config import KAGConfig

AggregationMode = Literal["narrative", "community", "full"]

DEFAULT_BACKEND = "langchain"
SUPPORTED_BACKENDS = {"langchain", "qwen", "openai_agents"}
BACKEND_ENV_VAR = "NKW_RETRIEVER_AGENT_BACKEND"


def _normalize_backend_name(name: Any) -> str:
    value = str(name or "").strip().lower()
    if not value:
        return DEFAULT_BACKEND
    if value not in SUPPORTED_BACKENDS:
        raise ValueError(
            f"Unsupported retriever agent backend: {value!r}. "
            f"Expected one of: {sorted(SUPPORTED_BACKENDS)}"
        )
    return value


def _resolve_backend_name(config: KAGConfig | None = None) -> str:
    env_value = os.getenv(BACKEND_ENV_VAR, "")
    if env_value.strip():
        return _normalize_backend_name(env_value)
    if config is None:
        return DEFAULT_BACKEND
    global_cfg = getattr(config, "global_config", None) or getattr(config, "global_", None)
    configured = getattr(global_cfg, "retriever_agent_backend", "") if global_cfg is not None else ""
    return _normalize_backend_name(configured)


def _load_backend_module(backend: str):
    backend_name = _normalize_backend_name(backend)
    module_name = f"core.agent.retriever_agent_{backend_name}"
    try:
        return importlib.import_module(module_name)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load retriever agent backend {backend_name!r} from {module_name}. "
            f"This backend may require legacy dependencies that are not available in the current repo."
        ) from exc


_langchain_module = _load_backend_module("langchain")
DEFAULT_SYSTEM_MESSAGE = cast(str, _langchain_module.DEFAULT_SYSTEM_MESSAGE)
_split_bm25_documents = _langchain_module._split_bm25_documents


class QuestionAnsweringAgent:
    """
    Compatibility facade.

    Default backend remains `langchain`.
    Set `global.retriever_agent_backend: qwen|openai_agents` in config, or export
    `NKW_RETRIEVER_AGENT_BACKEND=qwen|openai_agents`, to switch backend.
    """

    def __init__(self, config: KAGConfig, *args: Any, **kwargs: Any) -> None:
        backend_name = _resolve_backend_name(config)
        backend_module = _load_backend_module(backend_name)
        backend_cls = getattr(backend_module, "QuestionAnsweringAgent")
        object.__setattr__(self, "_backend_name", backend_name)
        object.__setattr__(self, "_impl", backend_cls(config, *args, **kwargs))

    @property
    def backend_name(self) -> str:
        return cast(str, object.__getattribute__(self, "_backend_name"))

    def __getattr__(self, item: str) -> Any:
        return getattr(object.__getattribute__(self, "_impl"), item)

    def __setattr__(self, key: str, value: Any) -> None:
        if key in {"_impl", "_backend_name"}:
            object.__setattr__(self, key, value)
            return
        setattr(object.__getattribute__(self, "_impl"), key, value)

    def __repr__(self) -> str:
        impl = object.__getattribute__(self, "_impl")
        backend_name = object.__getattribute__(self, "_backend_name")
        return f"QuestionAnsweringAgent(backend={backend_name!r}, impl={impl!r})"


LangchainQuestionAnsweringAgent = _langchain_module.QuestionAnsweringAgent


def load_qwen_question_answering_agent():
    module = _load_backend_module("qwen")
    return module.QuestionAnsweringAgent


__all__ = [
    "AggregationMode",
    "BACKEND_ENV_VAR",
    "DEFAULT_BACKEND",
    "DEFAULT_SYSTEM_MESSAGE",
    "LangchainQuestionAnsweringAgent",
    "QuestionAnsweringAgent",
    "SUPPORTED_BACKENDS",
    "_split_bm25_documents",
    "load_qwen_question_answering_agent",
]

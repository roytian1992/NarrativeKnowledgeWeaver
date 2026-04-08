from __future__ import annotations

from core.agent.retriever_agent_langchain import (
    DEFAULT_SYSTEM_MESSAGE,
    QuestionAnsweringAgent as _LangchainQuestionAnsweringAgent,
    _split_bm25_documents,
)
from core.model_providers.openai_agents_llm import OpenAIAgentsLLM


class QuestionAnsweringAgent(_LangchainQuestionAnsweringAgent):
    """
    Compatibility backend that keeps the existing retrieval/tool runtime,
    but swaps the answering LLM to OpenAI Agents SDK chat-completions calls.

    Note:
    Native tool calling is intentionally not used here because the current
    local OpenAI-compatible endpoint rejects `tool_choice=auto`. Tool routing
    and execution still happen through the existing manual JSON tool loop.
    """

    def __init__(self, config, *args, **kwargs) -> None:
        super().__init__(config, *args, **kwargs)
        self.retriever_llm = OpenAIAgentsLLM(config, llm_profile="retriever")
        self._assistant_cache = {}
        self._rebuild_assistant()


__all__ = [
    "DEFAULT_SYSTEM_MESSAGE",
    "QuestionAnsweringAgent",
    "_split_bm25_documents",
]

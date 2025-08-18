# qa_agent.py
from __future__ import annotations
from typing import List, Dict, Any, Optional

from core.utils.config import KAGConfig
from core.storage.graph_store import GraphStore
from core.storage.vector_store import VectorStore
from core.utils.neo4j_utils import Neo4jUtils
from qwen_agent.agents import Assistant


from core.functions.tool_calls import (
    EntityRetrieverName,
    EntityRetrieverID,
    SearchRelatedEntities,
    GetRelationSummary,
    GetCommonNeighbors,
    QuerySimilarEntities,
    FindEventChain,
    CheckNodesReachable,
)

from core.functions.tool_calls import (
    VDBHierdocsSearchTool,
    VDBDocsSearchTool,
    VDBGetDocsByChunkIDsTool,
    VDBSentencesSearchTool,
)

DEFAULT_SYSTEM_PROMPT = (
    "你是一名知识图谱 + 向量数据库驱动的问答智能体。\n"
    "当问题包含明确的实体时，优先调用图谱检索工具；当问题更偏段落/背景/长文内容时，优先调用向量检索工具（层级检索优先）。\n"
    "回答请简洁、准确、中文，必要时分点展示；若信息来自检索结果，请用自己的话归纳。"
)

class QuestionAnsweringAgent:
    """
    基于 KAGConfig + Qwen Assistant 的问答智能体：
      - 自动装配 GraphStore / Neo4jUtils（含 embedding）
      - 自动创建向量库 VectorStore(config, "documents"/"sentences")
      - 默认注册 8 个图谱查询工具 + 4 个 VDB 检索工具
      - 提供 ask() 非流式问答与 extract_final_text() 提取最终文本
    """

    def __init__(
        self,
        config: KAGConfig,
        *,
        doc_type: str = "novel",
        system_prompt: Optional[str] = None,
        # 可选：传入自定义重排器；不传则 ParentChildRetriever 按你的实现可用 None
        reranker: Optional[Any] = None,
        # 可选：追加自定义工具
        extra_tools: Optional[List[Any]] = None,
    ):
        self.config = config
        self.doc_type = doc_type
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT

        # ---- Graph / Neo4j ----
        self.graph_store = GraphStore(config)
        self.neo4j_utils = Neo4jUtils(self.graph_store.driver, doc_type=doc_type)
        self.neo4j_utils.load_embedding_model(config.graph_embedding)

        # ---- 向量库（自动基于 KAGConfig 构建）----
        self.document_vector_store = VectorStore(config, "documents")
        self.sentence_vector_store = VectorStore(config, "sentences")
        self.reranker = reranker

        # ---- LLM ----
        self.llm_cfg = {
            "model": config.llm.model_name,
            "model_server": config.llm.base_url,
            "api_key": config.llm.api_key,
        }

        # ---- 工具列表 ----
        self.tools = self._build_default_tools()
        if extra_tools:
            self.tools.extend(extra_tools)

        self.assistant = self._build_assistant()

    # ---------- Public API ----------
    def ask(self, user_text: str, history: Optional[List[Dict[str, str]]] = None) -> List[Dict[str, Any]]:
        messages = list(history) if history else []
        messages.append({"role": "user", "content": user_text})
        return self.assistant.run_nonstream(messages=messages)

    def extract_final_text(self, responses: List[Dict[str, Any]]) -> str:
        final_text = ""
        fallback_chunks: List[str] = []
        for msg in responses:
            role = msg.get("role")
            content = msg.get("content") or ""
            if role == "assistant" and content.strip():
                final_text = content
            elif role == "function" and content.strip():
                fallback_chunks.append(content.strip())
        if final_text:
            return final_text.strip()
        if fallback_chunks:
            return "\n\n".join(fallback_chunks).strip()
        return ""

    # ---------- Internal ----------
    def _build_default_tools(self) -> List[Any]:
        emb_cfg = self.config.graph_embedding

        tools: List[Any] = [
            # 图谱工具 8 个
            EntityRetrieverName(self.neo4j_utils, emb_cfg),
            EntityRetrieverID(self.neo4j_utils, emb_cfg),
            SearchRelatedEntities(self.neo4j_utils, emb_cfg),
            GetRelationSummary(self.neo4j_utils),
            GetCommonNeighbors(self.neo4j_utils),
            QuerySimilarEntities(self.neo4j_utils, emb_cfg),
            FindEventChain(self.neo4j_utils),
            CheckNodesReachable(self.neo4j_utils),
            # VDB 工具 4 个（全部自动启用）
            VDBDocsSearchTool(self.document_vector_store),
            VDBGetDocsByChunkIDsTool(self.document_vector_store),
            VDBSentencesSearchTool(self.sentence_vector_store),
            VDBHierdocsSearchTool(
                document_vector_store=self.document_vector_store,
                sentence_vector_store=self.sentence_vector_store,
                reranker=self.reranker,
            ),
        ]
        return tools

    def _build_assistant(self) -> Assistant:
        return Assistant(
            llm=self.llm_cfg,
            function_list=self.tools,
            # system_prompt=self.system_prompt,
        )

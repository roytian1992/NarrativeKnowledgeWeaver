# retriever_agent.py 

# -*- coding: utf-8 -*-
"""
QA Agent (Retriever-First)
==========================

This module implements a retrieval-first QA agent that can:
- query a Neo4j knowledge graph via tool calls,
- retrieve passages from a vector database (documents & sentences),
- optionally query a CMP (Costume/Makeup/Props) SQLite database via SQL tools,
- perform BM25 keyword search on raw chunks.

Key design:
-----------
- Modes:
    * "graph_only"  : graph tools (+ optional SQL/CMP + BM25)
    * "vector_only" : vector tools (+ optional SQL/CMP + BM25)
    * "hybrid"      : both graph and vector tools (+ optional SQL/CMP + BM25)
- Single-turn `ask()` (no dialogue history is kept/updated).
- Schema knowledge and system message are injected by mode.
- CMP/SQL tools are optional via `enable_cmp_sql` (default False).

Public API:
-----------
- set_mode(mode: Literal["hybrid","graph_only","vector_only"]) -> None
- ask(user_text: str, *, lang: Literal["zh","en"] = "zh", **kwargs) -> List[Dict[str, Any]]
- extract_final_text(responses) -> str
- extract_tool_uses(responses) -> List[Dict[str, Any]]
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional, Literal
import os
import json

from core.utils.config import KAGConfig
from core.storage.graph_store import GraphStore
from core.storage.vector_store import VectorStore
from core.utils.neo4j_utils import Neo4jUtils
from core.model_providers.openai_rerank import OpenAIRerankModel
from qwen_agent.agents import Assistant
from core.model_providers.openai_llm import OpenAILLM
from langchain_core.documents import Document

from core.utils.format import DOC_TYPE_DESCRIPTION

from core.functions.tool_calls import (
    # Graph tools
    EntityRetrieverName,
    EntityRetrieverID,
    SearchRelatedEntities,
    GetRelationSummary,
    GetCommonNeighbors,
    QuerySimilarEntities,
    FindPathsBetweenNodes,
    TopKByCentrality,
    FindRelatedEventsAndPlots,
    GetKHopSubgraph,
    GetCoSectionEntities,  # graph-side helper (not SQL)

    # Vector DB tools
    VDBHierdocsSearchTool,
    VDBDocsSearchTool,
    VDBGetDocsByChunkIDsTool,
    VDBSentencesSearchTool,

    # Keyword/BM25
    BM25SearchDocsTool,
)

# NOTE: SQL/CMP tools are imported *lazily* only when enable_cmp_sql=True:
# from core.functions.sqldb_tools import (
#     Search_By_Character, Search_By_Scene, Chunk_To_Scene, Scene_To_Chunks, NLP2SQL_Query
# )

DEFAULT_SYSTEM_MESSAGE = (
    "You are a retrieval-style assistant powered by a knowledge graph and a vector database.\n"
    "When the question targets explicit entities, prefer graph-based tools. "
    "When the question asks for paragraph/background/long-form content, prefer vector retrieval (hierarchical first).\n"
    "Do not omit necessary information in the final answer. Your role is an information retriever—avoid giving advice.\n"
)

Mode = Literal["hybrid", "graph_only", "vector_only"]


def _prepare_knowledge(schema: Dict[str, Any], doc_type: str, *, include_cmp_note: bool = False) -> str:
    """Build a human-readable knowledge block from graph schema and doc_type metadata."""
    entity_types = (schema or {}).get("entities", [])
    relation_type_groups = (schema or {}).get("relations", {})

    # Entity types
    entity_type_description_text = "\n".join(
        f"- {e.get('type')}: {e.get('description', '')}" for e in entity_types
    )
    entity_type_description_text += DOC_TYPE_DESCRIPTION.get(doc_type, {}).get("entity", "")
    entity_type_description_text += "\n- Plot: High-level narrative segment representing a major storyline."

    # Relation types
    relation_type_description_text = "\n".join(
        f"- {r.get('type')}: {r.get('description', '')}"
        for group in relation_type_groups.values()
        for r in group
    )
    relation_type_description_text += DOC_TYPE_DESCRIPTION.get(doc_type, {}).get("relation", "")
    relation_type_description_text += """
[Plot & Event related relation types]
- HAS_EVENT: Plot contains Event (Plot → Event)
- EVENT_CAUSES: Event A causally leads to Event B
- EVENT_INDIRECT_CAUSES: Event A indirectly triggers Event B
- EVENT_PART_OF: Event A is a part of a larger Event B
- PLOT_PREREQUISITE_FOR: Plot A is a prerequisite for Plot B
- PLOT_ADVANCES: Plot A advances Plot B
- PLOT_BLOCKS: Plot A blocks or delays Plot B
- PLOT_RESOLVES: Plot A resolves the conflict/mystery of Plot B
- PLOT_CONFLICTS_WITH: Plot A conflicts with Plot B
- PLOT_PARALLELS: Plot A runs parallel to Plot B with echoes or contrasts
"""

    full_knowledge = (
        "The Neo4j knowledge graph contains:\n"
        f"## Entity Types\n{entity_type_description_text}\n\n"
        f"## Base Relation Types\n{relation_type_description_text}\n"
        "Entity attribute `source_chunks` lists chunk_ids of source fragments; "
        "use `vdb_get_docs_by_chunk_ids` to fetch raw text. "
        "Use `bm25_search_docs` for keyword-based lookup."
    )

    if include_cmp_note and doc_type == "screenplay":
        # Unified EN column names consistent with CMP_info / Scene_info
        sql_cols = [
            "name", "category", "subcategory", "appearance", "status", "character",
            "evidence", "notes", "chunk_id", "scene_id", "scene_title", "subscene_title"
        ]
        col_txt = ", ".join(sql_cols)
        col_help = (
            "- name: item name (wardrobe piece, styling element, or prop).\n"
            "- category: high-level type, e.g., 'wardrobe', 'styling', or 'prop'.\n"
            "- subcategory: fine-grained subtype, e.g., 'uniform', 'hairstyle', 'weapon'.\n"
            "- appearance: visual description (color, material, distinctive features).\n"
            "- status: state/condition, e.g., 'damaged', 'bloodstained', 'open/closed'.\n"
            "- character: associated on-screen character(s) in the scene.\n"
            "- evidence: supporting script snippet(s) or line references.\n"
            "- notes: extra remarks, disambiguation, or continuity notes.\n"
            "- chunk_id: internal text-chunk identifier mentioning the item.\n"
            "- scene_id: stable scene identifier for cross-referencing.\n"
            "- scene_title: human-readable scene title.\n"
            "- subscene_title: optional subscene title if the scene is subdivided."
        )
        full_knowledge += (
            "\n\nAdditionally, costume/makeup/props (CMP) information is stored in the SQL table "
            "`CMP_info` with columns:\n"
            f"{col_txt}\n\n"
            "Column meanings:\n"
            f"{col_help}"
        )

    return full_knowledge



class QuestionAnsweringAgent:
    """
    Retrieval-first agent.

    Modes:
        - graph_only  : graph tools (+ optional SQL/CMP + BM25)
        - vector_only : vector tools (+ optional SQL/CMP + BM25)
        - hybrid      : both graph and vector tools (+ optional SQL/CMP + BM25)

    `enable_cmp_sql`:
        - If False (default): do NOT load/register SQL/CMP tools or add CMP notes to knowledge.
        - If True: import and register SQL/CMP tools, and append CMP knowledge note.
    """

    def __init__(
        self,
        config: KAGConfig,
        *,
        doc_type: Optional[str] = None,
        system_message: Optional[str] = None,
        rag_cfg: Optional[Dict[str, Any]] = None,
        reranker: Optional[Any] = None,
        extra_tools: Optional[List[Any]] = None,
        mode: Mode = "hybrid",
        doc_path: Optional[str] = None,
        enable_cmp_sql: bool = False,
    ):
        self.config = config
        self.doc_type = doc_type or config.knowledge_graph_builder.doc_type
        self._base_system_message = system_message or DEFAULT_SYSTEM_MESSAGE
        self.rag_cfg = rag_cfg or {}
        self.mode: Mode = mode
        self.enable_cmp_sql = bool(enable_cmp_sql)

        # Graph / Neo4j
        self.graph_store = GraphStore(config)
        self.neo4j_utils = Neo4jUtils(self.graph_store.driver, doc_type=self.doc_type)
        self.neo4j_utils.load_embedding_model(config.graph_embedding)
        self.llm = OpenAILLM(config)

        # Vector DBs
        self.document_vector_store = VectorStore(config, "documents")
        self.sentence_vector_store = VectorStore(config, "sentences")
        self.reranker = reranker or OpenAIRerankModel(config)

        # LLM config for Assistant
        self.llm_cfg = {
            "model": config.llm.model_name,
            "model_server": config.llm.base_url,
            "api_key": config.llm.api_key,
        }

        # (Optional) SQL/CMP DB path
        self.db_path: Optional[str] = None
        if self.enable_cmp_sql:
            self.db_path = os.path.join(self.config.storage.sql_database_path, "CMP.db")

        # Tool groups
        self._graph_tools = self._build_graph_tools()
        self._vdb_tools = self._build_vdb_tools(reranker=self.reranker)
        self._native_tools = self._build_native_tools(doc_path=doc_path)  # BM25 + (optional) SQL/CMP
        self._extra_tools = extra_tools or []

        # Knowledge & system prompt
        self._current_knowledge = self._build_knowledge(self.mode)
        self._current_system_message = self._build_system_message(self.mode)

        # Build assistant
        self._rebuild_assistant()

    # ---------------- Knowledge & System Message ----------------

    def _build_system_message(self, mode: Mode) -> str:
        """Compose system message per mode while preserving 'retriever-only' principle."""
        if mode == "graph_only":
            suffix = "(Mode: graph_only — use graph tools, keyword search, and (optional) SQL tools only.)"
        elif mode == "vector_only":
            suffix = "(Mode: vector_only — use vector tools, keyword search, and (optional) SQL tools only.)"
        else:
            suffix = "(Mode: hybrid — adaptively use both graph and vector tools; still retrieval-only.)"

        if self._current_knowledge:
            suffix += f"\n\nKnowledge:\n{self._current_knowledge}"
        return f"{self._base_system_message}\n\n{suffix}"

    def _load_graph_schema(self) -> Optional[Dict[str, Any]]:
        schema_path = os.path.join(self.config.storage.graph_schema_path, "graph_schema.json")
        if os.path.exists(schema_path):
            with open(schema_path, "r", encoding="utf-8") as f:
                return json.load(f)

        default_path = getattr(self.config.probing, "default_graph_schema_path", None)
        if default_path and os.path.exists(default_path):
            with open(default_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    def _build_knowledge(self, mode: Mode) -> str:
        """
        - graph_only / hybrid: inject graph schema knowledge.
        - vector_only: keep empty for now (can be extended later).
        """
        include_cmp = self.enable_cmp_sql
        if mode == "vector_only":
            return ""
        schema = self._load_graph_schema()
        return _prepare_knowledge(schema, self.doc_type, include_cmp_note=include_cmp) if schema else ""

    # ---------------- Public: Mode ----------------

    def set_mode(self, mode: Mode) -> None:
        """Switch mode and rebuild system message / knowledge / assistant."""
        if mode not in ("hybrid", "graph_only", "vector_only"):
            raise ValueError(f"Unsupported mode: {mode}")
        if mode != self.mode:
            self.mode = mode
            self._current_knowledge = self._build_knowledge(mode)
            self._current_system_message = self._build_system_message(mode)
            self._rebuild_assistant()

    # ---------------- Public: Ask (single turn) ----------------

    def ask(
        self,
        user_text: str,
        *,
        lang: Literal["zh", "en"] = "zh",
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Single-turn retrieval:
        - No history state is maintained.
        - Returns the full Assistant message sequence including tool calls.
        """
        # Keep the instruction short; the tool routing is learned from examples/usages.
        system_prompt = (
            "You are a tool-using assistant. If one retrieval fails, try another: "
            "e.g., if `retrieve_entity_by_name` fails, try `query_similar_entities` or vector tools. "
            "Prefer BM25 keyword search to verify sensitive details (appearances, model numbers, timelines, item provenance)."
        )
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ]
        resp = self.assistant.run_nonstream(messages=messages, lang=lang, **kwargs)
        return resp

    # ---------------- Public: Helpers ----------------

    def extract_final_text(self, responses: List[Dict[str, Any]]) -> str:
        """Pick the last assistant text; fallback to concatenated function outputs."""
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

    def extract_tool_uses(self, responses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract intermediate tool invocations.

        Returns:
            [
              {"tool_name": ..., "tool_arguments": ..., "tool_output": ...},
              ...
            ]
        """
        tool_uses: List[Dict[str, Any]] = []
        pending: Optional[Dict[str, Any]] = None

        for msg in responses:
            role = msg.get("role")
            if role == "assistant" and "function_call" in msg:
                fc = msg["function_call"] or {}
                pending = {
                    "tool_name": fc.get("name") or "unknown_tool",
                    "tool_arguments": fc.get("arguments") or "",
                    "tool_output": None,
                }
            elif role == "function":
                tool_name = msg.get("name") or (pending.get("tool_name") if pending else "unknown_tool")
                tool_output = msg.get("content") or ""
                if pending and pending["tool_output"] is None:
                    pending["tool_output"] = tool_output
                    if msg.get("name"):
                        pending["tool_name"] = msg["name"]
                    tool_uses.append(pending)
                    pending = None
                else:
                    tool_uses.append({
                        "tool_name": tool_name,
                        "tool_arguments": None,
                        "tool_output": tool_output,
                    })
        return tool_uses

    # ---------------- Internal: Tool Wiring ----------------

    def _build_graph_tools(self) -> List[Any]:
        emb_cfg = self.config.graph_embedding
        return [
            EntityRetrieverName(self.neo4j_utils, emb_cfg),
            EntityRetrieverID(self.neo4j_utils, emb_cfg),
            SearchRelatedEntities(self.neo4j_utils, emb_cfg),
            GetRelationSummary(self.neo4j_utils),
            GetCommonNeighbors(self.neo4j_utils),
            QuerySimilarEntities(self.neo4j_utils, emb_cfg),
            FindPathsBetweenNodes(self.neo4j_utils),
            TopKByCentrality(self.neo4j_utils),
            GetCoSectionEntities(self.neo4j_utils),
            FindRelatedEventsAndPlots(self.neo4j_utils),
            GetKHopSubgraph(self.neo4j_utils),
        ]

    def _build_vdb_tools(self, *, reranker: Any) -> List[Any]:
        return [
            VDBDocsSearchTool(self.document_vector_store),
            VDBGetDocsByChunkIDsTool(self.document_vector_store),
            VDBSentencesSearchTool(self.sentence_vector_store),
            VDBHierdocsSearchTool(
                document_vector_store=self.document_vector_store,
                sentence_vector_store=self.sentence_vector_store,
                reranker=reranker,
            ),
        ]

    def _build_native_tools(self, *, doc_path: Optional[str]) -> List[Any]:
        """
        Build native tools:
        - BM25SearchDocsTool (always)
        - (optional) SQL/CMP tools when `enable_cmp_sql=True`
        """
        # Load raw chunks for BM25
        base = doc_path or os.path.join(self.config.storage.knowledge_graph_path, "all_document_chunks.json")
        with open(base, "r", encoding="utf-8") as f:
            data = json.load(f)

        keys_to_drop = {"chunk_index", "chunk_type", "doc_title", "order", "total_doc_chunks"}
        documents: List[Document] = []
        for item in data:
            chunk_id = item.get("id")
            content = (item.get("content") or "").strip()
            if not chunk_id or not content:
                continue
            meta: Dict[str, Any] = dict(item.get("metadata") or {})
            meta["chunk_id"] = chunk_id
            for key in list(keys_to_drop):
                if key in meta:
                    del meta[key]
            documents.append(Document(page_content=content, metadata=meta))

        native_tools: List[Any] = [BM25SearchDocsTool(documents)]

        # Attach SQL/CMP tools ONLY when enabled
        if self.enable_cmp_sql:
            try:
                from core.functions.sqldb_tools import (
                    Search_By_Character,
                    Search_By_Scene,
                    Chunk_To_Scene,
                    Scene_To_Chunks,
                    NLP2SQL_Query,
                )
                if not self.db_path:
                    self.db_path = os.path.join(self.config.storage.sql_database_path, "CMP.db")

                native_tools.extend([
                    Search_By_Character(self.db_path),
                    Search_By_Scene(self.db_path),
                    Chunk_To_Scene(self.db_path),
                    Scene_To_Chunks(self.db_path),
                    NLP2SQL_Query(self.db_path, self.llm),
                ])
            except Exception as e:
                # If SQL tools are missing or DB not present, we silently continue without them.
                # You can switch to logging if you have a logger facility available.
                pass

        return native_tools

    def _select_tools(self, mode: Mode) -> List[Any]:
        if mode == "graph_only":
            return [*self._graph_tools, *self._native_tools, *self._extra_tools]
        elif mode == "vector_only":
            return [*self._vdb_tools, *self._native_tools, *self._extra_tools]
        else:  # hybrid
            return [*self._graph_tools, *self._vdb_tools, *self._native_tools, *self._extra_tools]

    def _rebuild_assistant(self) -> None:
        tools = self._select_tools(self.mode)
        self.assistant = Assistant(
            function_list=tools,
            llm=self.llm_cfg,
            system_message=self._current_system_message,
            rag_cfg=self.rag_cfg,
        )

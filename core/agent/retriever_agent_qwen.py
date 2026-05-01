# retriever_agent.py 

# -*- coding: utf-8 -*-
"""
QA Agent (Retriever-First)
==========================

This module implements a retrieval-first QA agent that can:
- query the base knowledge graph runtime,
- use aggregation-specific tools (narrative or community),
- retrieve passages from vector stores,
- optionally query an Interaction SQLite database,
- perform BM25 keyword search on raw chunks.

Key design:
-----------
- Aggregation modes:
    * "narrative" : event -> episode -> storyline aggregation tools
    * "community" : leiden community report aggregation tools
    * "full" : load both narrative and community aggregation tools
- Single-turn `ask()` (no dialogue history is kept/updated).
- Schema knowledge and system message are injected by mode.
- SQL tools are optional via `enable_sql_tools` (default False).

Public API:
-----------
- set_aggregation_mode(mode: Literal["narrative","community","full"]) -> None
- ask(user_text: str, *, lang: Literal["zh","en"] = "zh", **kwargs) -> List[Dict[str, Any]]
- extract_final_text(responses) -> str
- extract_tool_uses(responses) -> List[Dict[str, Any]]
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional, Literal, Tuple
import os
import json
import logging
import copy
import importlib
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from core.utils.config import KAGConfig
from core.storage.graph_store import GraphStore
from core.storage.vector_store import VectorStore
from core.utils.graph_query_utils import GraphQueryUtils
from core.model_providers.openai_rerank import OpenAIRerankModel
from core.utils.prompt_loader import YAMLPromptLoader


def _prefer_external_qwen_agent_package() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    original_sys_path = list(sys.path)
    try:
        filtered_path: List[str] = []
        for raw in original_sys_path:
            try:
                if Path(raw).resolve() == repo_root:
                    continue
            except Exception:
                pass
            filtered_path.append(raw)
        sys.path = filtered_path
        for name in list(sys.modules.keys()):
            if name == "qwen_agent" or name.startswith("qwen_agent."):
                sys.modules.pop(name, None)
        importlib.import_module("qwen_agent")
    finally:
        sys.path = original_sys_path


_prefer_external_qwen_agent_package()

from qwen_agent.agents import Assistant
import qwen_agent.agents.fncall_agent as qwen_fncall_agent
from qwen_agent.tools.base import BaseTool
from core.model_providers.openai_llm import OpenAILLM
from core.builder.manager.document_manager import DocumentParser
from core.agent.retrieval import RetrievalToolRouter
from core.agent.retrieval.strategy_subagent import StrategySubagentCandidate
from core.memory.online_strategy_buffer import OnlineStrategyBuffer
from core.memory.retrieval_strategy_memory import RetrievalStrategyMemory
from core.memory.self_bootstrap_manager import SelfBootstrapManager
from core.functions.memory_management.candidate_answer_selector import CandidateAnswerSelector
from core.functions.memory_management.failure_retry import FailedAnswerReflector, RetryInstructionBuilder
from core.functions.memory_management.online_answer_judge import OnlineAnswerJudge
from core.functions.memory_management.online_sampling_branch_planner import OnlineSamplingBranchPlanner
from core.functions.memory_management.self_bootstrap_qa_generator import SelfBootstrapQAGenerator
from core.functions.memory_management.strategy_query_pattern import StrategyQueryPatternExtractor
from core.functions.memory_management.effective_tool_chain_extractor import EffectiveToolChainExtractor
from core.functions.memory_management.strategy_template_distiller import StrategyTemplateDistiller
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from retriever import NarrativeTreeRetriever, SectionTreeRetriever

from core.utils.format import DOC_TYPE_META
from core.utils.general_utils import load_json, now_ms, word_len

from core.functions.tool_calls import (
    # Graph tools
    EntityRetrieverName,
    EntityRetrieverID,
    SearchCommunities,
    SearchSections,
    SearchRelatedEntities,
    GetEntitySections,
    GetRelationsBetweenEntities,
    GetCommonNeighbors,
    QuerySimilarFacts,
    FindPathsBetweenNodes,
    TopKByCentrality,
    GetKHopSubgraph,
    GetCoSectionEntities,  # graph-side helper (not SQL)

    # Vector DB tools
    VDBDocsSearchTool,
    VDBGetDocsByDocumentIDsTool,
    VDBSentencesSearchTool,

    # Keyword/BM25
    BM25SearchDocsTool,
    LookupTitlesByDocumentIDsTool,
    LookupDocumentIDsByTitleTool,
    SearchRelatedContentTool,
    CommunityGraphRAGSearch,
    NarrativeHierarchicalSearch,
    NarrativeCausalTraceSearch,
    HybridEvidenceSearch,
    SectionEvidenceSearch,
    ChoiceGroundedEvidenceSearch,
    EntityEventTraceSearch,
)
from core.functions.tool_calls.tool_metadata import ToolMetadataProvider

# NOTE: SQL tools are imported *lazily* only when enable_sql_tools=True.

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_MESSAGE = (
    "You are a retrieval-style assistant powered by a knowledge graph, aggregation summaries, vector retrieval, and optional SQL data.\n"
    "Use the available tools to retrieve evidence before answering.\n"
    "Do not omit necessary information in the final answer. Your role is an information retriever—avoid giving advice.\n"
)

AggregationMode = Literal["narrative", "community", "full"]


TOOL_OUTPUT_CHAR_LIMITS: Dict[str, int] = {
    "retrieve_entity_by_name": 1800,
    "search_sections": 2200,
    "get_entity_sections": 1600,
    "bm25_search_docs": 2200,
    "section_evidence_search": 2200,
    "choice_grounded_evidence_search": 2400,
    "entity_event_trace_search": 2600,
    "lookup_titles_by_document_ids": 1200,
    "lookup_document_ids_by_title": 1200,
    "search_related_content": 1600,
    "search_related_entities": 1800,
    "vdb_get_docs_by_document_ids": 2200,
    "vdb_search_docs": 2200,
    "vdb_search_sentences": 1800,
}
DEFAULT_TOOL_OUTPUT_CHAR_LIMIT = 2400


def _load_json_if_exists(path: str) -> Any:
    if not path or not os.path.exists(path):
        return None
    try:
        return load_json(path)
    except Exception as e:
        logger.warning("Failed to load schema json: %s error=%s", path, e)
        return None


def _clip_tool_output_text(text: Any, *, limit: int) -> str:
    raw = str(text or "")
    size = max(0, int(limit or 0))
    if size <= 0 or len(raw) <= size:
        return raw
    if size < 160:
        return raw[:size]
    head = max(80, int(size * 0.68))
    tail = max(40, size - head - 48)
    if head + tail >= len(raw):
        return raw[:size]
    omitted = len(raw) - head - tail
    marker = f"\n...[truncated {omitted} chars to fit context]...\n"
    return raw[:head].rstrip() + marker + raw[-tail:].lstrip()


class _ContextBudgetToolWrapper(BaseTool):
    def __init__(self, tool: Any, *, char_limit: int) -> None:
        self._tool = tool
        self.name = str(getattr(tool, "name", "") or "")
        self.description = str(getattr(tool, "description", "") or "")
        params = getattr(tool, "parameters", []) or []
        self.parameters = copy.deepcopy(params)
        self.cfg = copy.deepcopy(getattr(tool, "cfg", {}) or {})
        self._char_limit = max(0, int(char_limit or 0))

    def __getattr__(self, item: str) -> Any:
        return getattr(self._tool, item)

    @property
    def file_access(self) -> bool:
        try:
            return bool(getattr(self._tool, "file_access", False))
        except Exception:
            return False

    def call(self, params: Any, **kwargs) -> Any:
        result = self._tool.call(params, **kwargs)
        if self._char_limit <= 0 or not isinstance(result, str):
            return result
        clipped = _clip_tool_output_text(result, limit=self._char_limit)
        if clipped != result:
            logger.debug(
                "Tool output clipped for context budget: tool=%s raw_chars=%d clipped_chars=%d",
                self.name,
                len(result),
                len(clipped),
            )
        return clipped


def _split_bm25_documents(documents: List[Document], *, chunk_size: int) -> List[Document]:
    size = int(chunk_size or 0)
    if size <= 0:
        return list(documents or [])

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max(10, size),
        chunk_overlap=0,
        length_function=lambda t: word_len(t, lang="auto"),
        separators=[
            "\n\n", "### ", "## ", "# ",
            "\n",
            "。", "！", "？", "；", "：", "、", "，",
            ". ", "? ", "! ", "; ", ": ", ", ",
            " ",
            "",
        ],
        keep_separator=True,
    )

    out: List[Document] = []
    for doc in documents or []:
        text = str(getattr(doc, "page_content", "") or "").strip()
        if not text:
            continue
        metadata = dict(getattr(doc, "metadata", None) or {})
        if word_len(text, lang="auto") <= size:
            out.append(Document(page_content=text, metadata=metadata))
            continue

        parts = splitter.split_text(text) or [text]
        subchunk_index = 0
        for part in parts:
            seg = str(part or "").strip()
            if not seg:
                continue
            subchunk_index += 1
            seg_metadata = dict(metadata)
            seg_metadata["bm25_subchunk_index"] = subchunk_index
            out.append(Document(page_content=seg, metadata=seg_metadata))

    return out


def _extract_named_descriptions_from_list(items: Any) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    if not isinstance(items, list):
        return rows
    for item in items:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or item.get("type") or "").strip()
        description = str(item.get("description") or "").strip()
        if not name:
            continue
        rows.append({"name": name, "description": description})
    return rows


def _extract_named_descriptions_from_group_dict(groups: Any) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    if not isinstance(groups, dict):
        return rows
    for group_items in groups.values():
        rows.extend(_extract_named_descriptions_from_list(group_items))
    return rows


def _dedup_named_descriptions(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    seen: set[str] = set()
    for row in rows:
        name = str(row.get("name") or "").strip()
        if not name or name in seen:
            continue
        seen.add(name)
        out.append({"name": name, "description": str(row.get("description") or "").strip()})
    return out


def _format_named_descriptions(rows: List[Dict[str, str]]) -> str:
    if not rows:
        return "- (none)"
    lines: List[str] = []
    for row in rows:
        name = row["name"]
        description = row["description"]
        lines.append(f"- {name}: {description}" if description else f"- {name}")
    return "\n".join(lines)


def _prepare_knowledge(
    schema_dir: str,
    doc_type: str,
    *,
    aggregation_mode: str,
    include_sql_note: bool = False,
    available_entity_types: Optional[List[str]] = None,
) -> str:
    """Build a concise knowledge block directly from task-spec JSON schemas."""
    entity_schema = _load_json_if_exists(os.path.join(schema_dir, "default_entity_schema.json"))
    relation_schema = _load_json_if_exists(os.path.join(schema_dir, "default_relation_schema.json"))

    extra_entity_schema = None
    extra_relation_schema = None
    if aggregation_mode in {"community", "full"}:
        community_entity_schema = _load_json_if_exists(os.path.join(schema_dir, "default_community_entity_schema.json"))
        community_relation_schema = _load_json_if_exists(os.path.join(schema_dir, "default_community_relation_schema.json"))
    else:
        community_entity_schema = None
        community_relation_schema = None
    if aggregation_mode in {"narrative", "full"}:
        narrative_entity_schema = _load_json_if_exists(os.path.join(schema_dir, "default_narrative_entity_schema.json"))
        narrative_relation_schema = _load_json_if_exists(os.path.join(schema_dir, "default_narrative_relation_schema.json"))
    else:
        narrative_entity_schema = None
        narrative_relation_schema = None

    entity_rows = _dedup_named_descriptions(
        _extract_named_descriptions_from_list(entity_schema)
        + _extract_named_descriptions_from_list(narrative_entity_schema)
        + _extract_named_descriptions_from_list(community_entity_schema)
    )
    relation_rows = _dedup_named_descriptions(
        _extract_named_descriptions_from_group_dict(relation_schema if isinstance(relation_schema, dict) else {})
        + _extract_named_descriptions_from_group_dict(narrative_relation_schema if isinstance(narrative_relation_schema, dict) else {})
        + _extract_named_descriptions_from_group_dict(community_relation_schema if isinstance(community_relation_schema, dict) else {})
    )

    meta = DOC_TYPE_META.get(doc_type, DOC_TYPE_META["general"])
    section_label = str(meta.get("section_label", "Document")).strip() or "Document"
    section_line = (
        f"- {section_label}: document-level structural node used to organize mentions and section retrieval."
    )
    current_entity_type_lines = _format_named_descriptions(
        [{"name": item, "description": "available in the current article graph"} for item in (available_entity_types or [])]
    )

    full_knowledge = (
        "The retrieval system uses these schema-defined node and relation types:\n"
        f"## Entity Types\n{_format_named_descriptions(entity_rows)}\n\n"
        f"## Current Graph Entity Types\n{current_entity_type_lines}\n\n"
        f"## Section Node\n{section_line}\n\n"
        f"## Relation Types\n{_format_named_descriptions(relation_rows)}\n\n"
        "Entity attribute `source_documents` lists document-level source ids; "
        "use `vdb_get_docs_by_document_ids` to fetch raw text. "
        "Use `bm25_search_docs` for keyword-based lookup. "
        "When calling `retrieve_entity_by_name`, prefer the exact labels from `Current Graph Entity Types`."
    )

    if include_sql_note:
        title_col = str(meta.get("title", "title")).strip() or "title"
        subtitle_col = str(meta.get("subtitle", "subtitle")).strip() or "subtitle"
        sql_cols = [
            "id", "document_id", title_col, subtitle_col,
            "subject_name", "object_name", "interaction_type", "polarity", "content"
        ]
        col_txt = ", ".join(sql_cols)
        col_help = (
            "- interaction_type: dialogue or non-dialogue interaction.\n"
            "- polarity: sentiment polarity for non-dialogue interactions.\n"
            "- content: extracted interaction/dialogue text."
        )
        full_knowledge += (
            "\n\nAdditionally, interaction information is stored in SQL table "
            "`Interaction_info` with columns:\n"
            f"{col_txt}\n\n"
            "Column meanings:\n"
            f"{col_help}"
        )

    return full_knowledge



class QuestionAnsweringAgent:
    """
    Retrieval-first agent.

    `enable_sql_tools`:
        - If False (default): do NOT load/register SQL tools or add SQL notes to knowledge.
        - If True: import and register SQL tools for Interaction_info.
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
        aggregation_mode: Optional[AggregationMode] = None,
        mode: Optional[str] = None,
        doc_path: Optional[str] = None,
        enable_sql_tools: bool = False,
    ):
        self.config = config
        self.doc_type = doc_type or config.global_config.doc_type
        self.knowledge_graph_path = str(
            getattr(config.knowledge_graph_builder, "file_path", "") or "data/knowledge_graph"
        )
        self._base_system_message = system_message or DEFAULT_SYSTEM_MESSAGE
        self.rag_cfg = rag_cfg or {}
        configured_mode = self._normalize_aggregation_mode(
            aggregation_mode=str(getattr(self.config.global_config, "aggregation_mode", "") or "").strip(),
            legacy_mode=None,
            default_mode="narrative",
        )
        requested_mode = self._normalize_aggregation_mode(
            aggregation_mode=aggregation_mode,
            legacy_mode=mode,
            default_mode=configured_mode,
        )
        self.aggregation_mode: AggregationMode = requested_mode
        self.mode = self.aggregation_mode  # compatibility with existing memory code
        self.enable_sql_tools = bool(enable_sql_tools)
        self.strategy_memory: Optional[RetrievalStrategyMemory] = None
        self.candidate_answer_selector: Optional[CandidateAnswerSelector] = None
        self.online_buffer: Optional[OnlineStrategyBuffer] = None
        self.online_answer_judge: Optional[OnlineAnswerJudge] = None
        self.branch_answer_judge: Optional[OnlineAnswerJudge] = None
        self.sampling_branch_planner: Optional[OnlineSamplingBranchPlanner] = None
        self.failed_answer_reflector: Optional[FailedAnswerReflector] = None
        self.retry_instruction_builder: Optional[RetryInstructionBuilder] = None
        self.effective_tool_chain_extractor: Optional[EffectiveToolChainExtractor] = None
        self.strategy_template_distiller: Optional[StrategyTemplateDistiller] = None
        self.self_bootstrap_generator: Optional[SelfBootstrapQAGenerator] = None
        self.self_bootstrap_manager: Optional[SelfBootstrapManager] = None
        self.online_query_pattern_extractor: Optional[StrategyQueryPatternExtractor] = None
        self._last_strategy_context: Dict[str, Any] = {"query_abstract": "", "routing_hint": ""}
        native_routing_raw = self.rag_cfg.get(
            "qwen_native_tool_routing_only",
            os.getenv("NKW_QWEN_NATIVE_TOOL_ROUTING_ONLY", "1"),
        )
        native_routing_flag = str(native_routing_raw).strip().lower()
        self.qwen_native_tool_routing_only = native_routing_flag not in {"0", "false", "no", "off"}
        self.require_tool_use_by_default = bool(
            getattr(getattr(config, "strategy_memory", None), "require_tool_use", False)
        )
        strategy_enabled = bool(getattr(getattr(config, "strategy_memory", None), "enabled", False))
        self.tool_metadata_provider = ToolMetadataProvider.from_config(
            config,
            include_runtime_overrides=strategy_enabled,
        )

        # Graph runtime
        self.graph_store = GraphStore(config)
        self.graph_query_utils = GraphQueryUtils(self.graph_store, doc_type=self.doc_type)
        self.graph_query_utils.load_embedding_model(config.embedding)
        self.retriever_llm = OpenAILLM(config, llm_profile="retriever")
        self.router_llm = OpenAILLM(config, llm_profile="router")
        self.memory_llm = OpenAILLM(config, llm_profile="memory")
        self.document_parser = DocumentParser(config, self.retriever_llm)
        self.aggregation_mode = self._resolve_available_aggregation_mode(self.aggregation_mode)
        self.mode = self.aggregation_mode

        # Vector DBs
        self.document_vector_store = VectorStore(config, "document")
        self.sentence_vector_store = VectorStore(config, "sentence")
        self.community_vector_store = VectorStore(config, "community")
        self.reranker = reranker or OpenAIRerankModel(config)
        self.section_tree_retriever = SectionTreeRetriever(
            graph_query_utils=self.graph_query_utils,
            document_vector_store=self.document_vector_store,
            sentence_vector_store=self.sentence_vector_store,
            document_parser=self.document_parser,
            section_label=str(DOC_TYPE_META.get(self.doc_type, DOC_TYPE_META["general"]).get("section_label", "Document")),
            max_workers=getattr(self.config.document_processing, "max_workers", 8),
            tree_search_config=getattr(self.config, "tree_search", None),
        )
        self.narrative_tree_retriever = NarrativeTreeRetriever(
            graph_query_utils=self.graph_query_utils,
            document_vector_store=self.document_vector_store,
            sentence_vector_store=self.sentence_vector_store,
            document_parser=self.document_parser,
            max_workers=getattr(self.config.document_processing, "max_workers", 8),
            tree_search_config=getattr(self.config, "tree_search", None),
        )

        # Optional strategy memory for generic tool-routing patterns.
        try:
            strategy_memory_should_init = bool(
                getattr(self.config.strategy_memory, "enabled", False)
                or getattr(self.config.strategy_memory, "read_enabled", False)
                or getattr(self.config.strategy_memory, "runtime_routing_note_enabled", False)
            )
            if strategy_memory_should_init:
                self.strategy_memory = RetrievalStrategyMemory(
                    config=self.config,
                    llm=self.memory_llm,
                    embedding_model=getattr(self.document_vector_store, "embedding_model", None),
                )
                self.candidate_answer_selector = CandidateAnswerSelector(
                    prompt_loader=getattr(self.strategy_memory, "prompt_loader", None),
                    llm=self.memory_llm,
                )
        except Exception as e:
            logger.warning("Strategy memory init failed; continue without it. error=%s", e)
            self.strategy_memory = None
            self.candidate_answer_selector = None

        shared_prompt_loader = (
            getattr(self.strategy_memory, "prompt_loader", None)
            if self.strategy_memory is not None
            else None
        )
        if shared_prompt_loader is None:
            try:
                shared_prompt_loader = YAMLPromptLoader(self.config.global_config.prompt_dir)
            except Exception:
                shared_prompt_loader = None
        if shared_prompt_loader is not None:
            try:
                self.branch_answer_judge = OnlineAnswerJudge(
                    prompt_loader=shared_prompt_loader,
                    llm=self.memory_llm,
                    prompt_id=str(
                        getattr(getattr(self.config, "strategy_memory", None), "online_judge_prompt_id", "memory/judge_online_answer")
                        or "memory/judge_online_answer"
                    ).strip(),
                )
            except Exception as e:
                logger.warning("Branch answer judge init failed; continue without it. error=%s", e)
                self.branch_answer_judge = None
            try:
                self.sampling_branch_planner = OnlineSamplingBranchPlanner(
                    prompt_loader=shared_prompt_loader,
                    llm=self.memory_llm,
                    prompt_id=str(
                        getattr(
                            getattr(self.config, "strategy_memory", None),
                            "sampling_branch_planner_prompt_id",
                            "memory/plan_trajectory_direction",
                        )
                        or "memory/plan_trajectory_direction"
                    ).strip(),
                )
            except Exception as e:
                logger.warning("Sampling branch planner init failed; continue without it. error=%s", e)
                self.sampling_branch_planner = None

        try:
            sm_cfg = getattr(self.config, "strategy_memory", None)
            online_enabled = bool(getattr(sm_cfg, "online_enabled", False))
            bootstrap_enabled = bool(getattr(sm_cfg, "self_bootstrap_enabled", False))
            if online_enabled or bootstrap_enabled:
                prompt_loader = (
                    shared_prompt_loader
                    if shared_prompt_loader is not None
                    else YAMLPromptLoader(self.config.global_config.prompt_dir)
                )
                self.online_buffer = OnlineStrategyBuffer.from_config(self.config)
                self.online_answer_judge = OnlineAnswerJudge(
                    prompt_loader=prompt_loader,
                    llm=self.memory_llm,
                    prompt_id=str(getattr(sm_cfg, "online_judge_prompt_id", "memory/judge_online_answer") or "memory/judge_online_answer").strip(),
                )
                self.failed_answer_reflector = FailedAnswerReflector(
                    prompt_loader=prompt_loader,
                    llm=self.memory_llm,
                    prompt_id=str(getattr(sm_cfg, "failed_answer_reflection_prompt_id", "memory/reflect_failed_answer") or "memory/reflect_failed_answer").strip(),
                )
                self.retry_instruction_builder = RetryInstructionBuilder(
                    prompt_loader=prompt_loader,
                    llm=self.memory_llm,
                    prompt_id=str(getattr(sm_cfg, "retry_instruction_prompt_id", "memory/build_retry_instruction") or "memory/build_retry_instruction").strip(),
                )
                self.effective_tool_chain_extractor = EffectiveToolChainExtractor(
                    prompt_loader=prompt_loader,
                    llm=self.memory_llm,
                    prompt_id="memory/extract_effective_tool_chain",
                )
                self.strategy_template_distiller = StrategyTemplateDistiller(
                    prompt_loader=prompt_loader,
                    llm=self.memory_llm,
                    prompt_id="memory/distill_strategy_template",
                )
                self.online_query_pattern_extractor = (
                    getattr(self.strategy_memory, "query_pattern_extractor", None)
                    if self.strategy_memory is not None
                    else StrategyQueryPatternExtractor(
                        prompt_loader=prompt_loader,
                        llm=self.memory_llm,
                        prompt_id=str(getattr(sm_cfg, "pattern_extractor_prompt_id", "memory/extract_strategy_query_pattern") or "memory/extract_strategy_query_pattern").strip(),
                        abstraction_mode=str(getattr(sm_cfg, "abstraction_mode", "hybrid") or "hybrid").strip(),
                    )
                )
                self.self_bootstrap_generator = SelfBootstrapQAGenerator(
                    prompt_loader=prompt_loader,
                    llm=self.memory_llm,
                    prompt_id=str(getattr(sm_cfg, "self_bootstrap_prompt_id", "memory/generate_self_bootstrap_qa") or "memory/generate_self_bootstrap_qa").strip(),
                )
                self.self_bootstrap_manager = SelfBootstrapManager(
                    generator=self.self_bootstrap_generator,
                    judge=self.online_answer_judge,
                    min_accept_score=float(getattr(sm_cfg, "self_bootstrap_min_accept_score", 0.9) or 0.9),
                )
        except Exception as e:
            logger.warning("Online learning init failed; continue without it. error=%s", e)
            self.online_buffer = None
            self.online_answer_judge = None
            self.failed_answer_reflector = None
            self.retry_instruction_builder = None
            self.effective_tool_chain_extractor = None
            self.strategy_template_distiller = None
            self.self_bootstrap_generator = None
            self.self_bootstrap_manager = None
            self.online_query_pattern_extractor = None

        # LLM config for Assistant
        retriever_llm_cfg = self.config.get_llm_profile("retriever")
        self.llm_cfg = {
            "model": retriever_llm_cfg.model_name,
            "model_server": retriever_llm_cfg.base_url,
            "api_key": retriever_llm_cfg.api_key,
        }
        self.tool_router = RetrievalToolRouter(
            config=self.config,
            tool_metadata_provider=self.tool_metadata_provider,
            router_llm=self.router_llm,
        )

        # (Optional) SQL DB path
        self.db_path: Optional[str] = None
        if self.enable_sql_tools:
            self.db_path = os.path.join(self.config.storage.sql_database_path, "Interaction.db")

        # Tool groups
        self._base_tools = self._build_base_tools(doc_path=doc_path, reranker=self.reranker)
        self._aggregation_tools = self._build_aggregation_tools()
        self._extra_tools = extra_tools or []
        self._assistant_cache: Dict[Tuple[str, Tuple[str, ...]], Assistant] = {}

        # Knowledge & system prompt
        self._current_knowledge = self._build_knowledge(self.aggregation_mode)
        self._current_system_message = self._build_system_message(self.aggregation_mode)

        # Build assistant
        self._rebuild_assistant()

    # ---------------- Knowledge & System Message ----------------

    @staticmethod
    def _normalize_aggregation_mode(
        *,
        aggregation_mode: Optional[str],
        legacy_mode: Optional[str],
        default_mode: str = "narrative",
    ) -> AggregationMode:
        fallback = str(default_mode or "narrative").strip().lower()
        if fallback not in {"narrative", "community", "full"}:
            fallback = "narrative"
        raw = str(aggregation_mode or legacy_mode or fallback).strip().lower()
        if raw in {"narrative", "community", "full"}:
            return raw  # type: ignore[return-value]
        if raw in {"hybrid", "graph_only", "vector_only"}:
            logger.warning("Legacy retriever mode '%s' is deprecated; falling back to global.aggregation_mode.", raw)
        if fallback == "community":
            return "community"
        if fallback == "full":
            return "full"
        return "narrative"

    def _count_nodes_by_label(self, label: str) -> int:
        safe_label = str(label or "").strip()
        if not safe_label:
            return 0
        return self.graph_query_utils.count_nodes_with_label(safe_label)

    def _resolve_available_aggregation_mode(self, preferred_mode: AggregationMode) -> AggregationMode:
        community_count = self._count_nodes_by_label("Community")
        episode_count = self._count_nodes_by_label("Episode")
        storyline_count = self._count_nodes_by_label("Storyline")

        available = {
            "community": community_count > 0,
            "narrative": (episode_count + storyline_count) > 0,
        }
        if preferred_mode == "full":
            if available["community"] and available["narrative"]:
                return "full"
            if available["narrative"]:
                logger.warning(
                    "Preferred aggregation mode 'full' is only partially available; switching to 'narrative' "
                    "(Community=%d Episode=%d Storyline=%d).",
                    community_count,
                    episode_count,
                    storyline_count,
                )
                return "narrative"
            if available["community"]:
                logger.warning(
                    "Preferred aggregation mode 'full' is only partially available; switching to 'community' "
                    "(Community=%d Episode=%d Storyline=%d).",
                    community_count,
                    episode_count,
                    storyline_count,
                )
                return "community"
            logger.warning(
                "No aggregation artifacts detected for preferred mode 'full' "
                "(Community=%d Episode=%d Storyline=%d). Keep preferred mode.",
                community_count,
                episode_count,
                storyline_count,
            )
            return "full"
        if available.get(preferred_mode):
            return preferred_mode

        fallback_mode: Optional[AggregationMode] = None
        if preferred_mode == "community" and available["narrative"]:
            fallback_mode = "narrative"
        elif preferred_mode == "narrative" and available["community"]:
            fallback_mode = "community"

        if fallback_mode:
            logger.warning(
                "Preferred aggregation mode '%s' has no available artifacts; switching to '%s' "
                "(Community=%d Episode=%d Storyline=%d).",
                preferred_mode,
                fallback_mode,
                community_count,
                episode_count,
                storyline_count,
            )
            return fallback_mode

        logger.warning(
            "No aggregation artifacts detected for preferred mode '%s' "
            "(Community=%d Episode=%d Storyline=%d). Keep preferred mode.",
            preferred_mode,
            community_count,
            episode_count,
            storyline_count,
        )
        return preferred_mode

    def _build_system_message(self, aggregation_mode: AggregationMode) -> str:
        """Compose system message per aggregation mode."""
        if aggregation_mode == "community":
            suffix = "(Aggregation mode: community — use Community report tools together with base retrieval tools.)"
        elif aggregation_mode == "full":
            suffix = "(Aggregation mode: full — use both Community report tools and Episode/Storyline tools together with base retrieval tools.)"
        else:
            suffix = "(Aggregation mode: narrative — use Episode/Storyline tools together with base retrieval tools.)"

        if self._current_knowledge:
            suffix += f"\n\nKnowledge:\n{self._current_knowledge}"
        return f"{self._base_system_message}\n\n{suffix}"

    def _build_knowledge(self, aggregation_mode: AggregationMode) -> str:
        include_sql = self.enable_sql_tools
        schema_dir = str(getattr(self.config.global_config, "schema_dir", "") or "").strip()
        if not schema_dir:
            logger.warning("config.global_config.schema_dir is empty; skip schema knowledge injection.")
            return ""
        available_entity_types: List[str] = []
        try:
            section_label = str(self.graph_query_utils.meta.get("section_label", "Document")).strip() or "Document"
            available_entity_types = [
                item for item in (self.graph_query_utils.list_entity_types() or [])
                if str(item or "").strip() and str(item or "").strip() != section_label
            ]
        except Exception as exc:
            logger.warning("Failed to collect current graph entity types for knowledge injection: %s", exc)
        return _prepare_knowledge(
            schema_dir,
            self.doc_type,
            aggregation_mode=aggregation_mode,
            include_sql_note=include_sql,
            available_entity_types=available_entity_types,
        )

    # ---------------- Public: Aggregation Mode ----------------

    def set_aggregation_mode(self, aggregation_mode: AggregationMode) -> None:
        """Switch aggregation mode and rebuild knowledge / tools / assistant."""
        if aggregation_mode not in ("narrative", "community", "full"):
            raise ValueError(f"Unsupported aggregation_mode: {aggregation_mode}")
        target_mode = aggregation_mode
        target_mode = self._resolve_available_aggregation_mode(aggregation_mode)
        if target_mode != self.aggregation_mode:
            self.aggregation_mode = target_mode
            self.mode = target_mode
            self._aggregation_tools = self._build_aggregation_tools()
            self._current_knowledge = self._build_knowledge(target_mode)
            self._current_system_message = self._build_system_message(target_mode)
            self._assistant_cache = {}
            self._rebuild_assistant()

    def set_mode(self, mode: str) -> None:
        """Backward-compatible alias."""
        self.set_aggregation_mode(
            self._normalize_aggregation_mode(
                aggregation_mode=mode,
                legacy_mode=None,
                default_mode=self.aggregation_mode,
            )
        )

    def set_require_tool_use_mode(self, enabled: bool) -> None:
        self.require_tool_use_by_default = bool(enabled)

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
        reference_answer = str(kwargs.pop("reference_answer", "") or "").strip()
        online_learning = bool(kwargs.pop("online_learning", True))
        require_tool_use = bool(kwargs.pop("require_tool_use", self.require_tool_use_by_default))
        online_source = str(kwargs.pop("_online_source", "real") or "real").strip() or "real"
        online_depth = int(kwargs.pop("_online_depth", 0) or 0)
        forced_memory_ctx = kwargs.pop("_forced_memory_ctx", None)
        session_id = str(kwargs.get("session_id", "") or "").strip()
        memory_ctx: Dict[str, Any] = {"query_abstract": "", "routing_hint": ""}
        if isinstance(forced_memory_ctx, dict):
            memory_ctx = copy.deepcopy(forced_memory_ctx)
        elif self.strategy_memory is not None:
            try:
                memory_ctx = self.strategy_memory.prepare_read_context(
                    query=user_text,
                    doc_type=self.doc_type,
                    mode=self.aggregation_mode,
                )
            except Exception as e:
                logger.warning("Strategy memory read failed: %s", e)
        self._last_strategy_context = copy.deepcopy(memory_ctx)

        sm_cfg = getattr(self.config, "strategy_memory", None)
        online_runtime_mode = bool(getattr(sm_cfg, "online_runtime_mode", False))
        forced_sampling = self._should_force_sampling_branches(memory_ctx=memory_ctx)
        should_use_template_subagents = (
            self.strategy_memory is not None
            and self.strategy_memory.should_use_subagents(
                candidate_patterns=memory_ctx.get("candidate_patterns") or []
            )
        )

        if online_runtime_mode and self.strategy_memory is not None:
            query_pattern = memory_ctx.get("query_pattern") if isinstance(memory_ctx.get("query_pattern"), dict) else {}
            matched_patterns = self.strategy_memory.collect_runtime_matched_patterns(
                query=user_text,
                query_pattern=query_pattern,
                candidate_patterns=memory_ctx.get("candidate_patterns") or [],
            )
            deduped_subagent_patterns = self.strategy_memory.deduplicate_patterns_for_subagents(
                candidate_patterns=matched_patterns,
            )
            matched_pattern_count = len(matched_patterns)
            subagent_pattern_count = len(deduped_subagent_patterns)

            memory_ctx["runtime_matched_patterns"] = [copy.deepcopy(item) for item in matched_patterns]
            memory_ctx["runtime_matched_pattern_count"] = matched_pattern_count
            memory_ctx["runtime_subagent_pattern_count"] = subagent_pattern_count
            self._last_strategy_context = copy.deepcopy(memory_ctx)

            if matched_pattern_count > 0:
                forced_sampling = False
                should_use_template_subagents = False

            if matched_pattern_count <= 0:
                branch_contexts = self._build_sampling_branch_contexts(
                    query=user_text,
                    memory_ctx=memory_ctx,
                    min_branches=self._min_sampling_branches(),
                    pattern_pool=[],
                )
                if len(branch_contexts) >= 2:
                    branch_results = self._run_subagent_candidates(
                        user_text=user_text,
                        lang=lang,
                        branch_contexts=branch_contexts,
                        run_kwargs={**kwargs, "require_tool_use": require_tool_use},
                    )
                    selected = self._select_branch_answer_with_online_judge(
                        query=user_text,
                        memory_ctx=memory_ctx,
                        candidates=branch_results,
                    )
                    if selected is not None:
                        enriched_ctx = copy.deepcopy(memory_ctx)
                        enriched_ctx["subagent_mode_triggered"] = True
                        enriched_ctx["sampling_mode_triggered"] = True
                        enriched_ctx["subagent_candidates"] = [
                            item.to_selector_payload() for item in branch_results
                        ]
                        enriched_ctx["subagent_selected_candidate_ids"] = list(selected.get("selected_candidate_ids") or [])
                        enriched_ctx["subagent_selection_reason"] = str(selected.get("reason", "") or "").strip()
                        self._last_strategy_context = enriched_ctx
                        responses = self._compose_subagent_responses(
                            selected_candidates=selected.get("selected_candidates") or [],
                            final_answer=str(selected.get("final_answer", "") or ""),
                        )
                        return self._finalize_responses(
                            question=user_text,
                            responses=responses,
                            memory_ctx=enriched_ctx,
                            lang=lang,
                            session_id=session_id,
                            reference_answer=reference_answer,
                            online_learning=online_learning,
                            online_source=online_source,
                            online_depth=online_depth,
                        )
            elif bool(getattr(self.strategy_memory, "subagent_enabled", False)) and subagent_pattern_count >= 2:
                branch_contexts = self._build_subagent_branch_contexts(
                    query=user_text,
                    memory_ctx=memory_ctx,
                    candidate_patterns=deduped_subagent_patterns,
                    include_no_strategy_baseline=False,
                    include_free_exploration=True,
                )
                if len(branch_contexts) >= 2:
                    branch_results = self._run_subagent_candidates(
                        user_text=user_text,
                        lang=lang,
                        branch_contexts=branch_contexts,
                        run_kwargs={**kwargs, "require_tool_use": require_tool_use},
                    )
                    selected = self._select_branch_answer_with_online_judge(
                        query=user_text,
                        memory_ctx=memory_ctx,
                        candidates=branch_results,
                    )
                    if selected is not None:
                        enriched_ctx = copy.deepcopy(memory_ctx)
                        enriched_ctx["subagent_mode_triggered"] = True
                        enriched_ctx["sampling_mode_triggered"] = False
                        enriched_ctx["subagent_candidates"] = [
                            item.to_selector_payload() for item in branch_results
                        ]
                        enriched_ctx["subagent_selected_candidate_ids"] = list(selected.get("selected_candidate_ids") or [])
                        enriched_ctx["subagent_selection_reason"] = str(selected.get("reason", "") or "").strip()
                        self._last_strategy_context = enriched_ctx
                        responses = self._compose_subagent_responses(
                            selected_candidates=selected.get("selected_candidates") or [],
                            final_answer=str(selected.get("final_answer", "") or ""),
                        )
                        return self._finalize_responses(
                            question=user_text,
                            responses=responses,
                            memory_ctx=enriched_ctx,
                            lang=lang,
                            session_id=session_id,
                            reference_answer=reference_answer,
                            online_learning=online_learning,
                            online_source=online_source,
                            online_depth=online_depth,
                        )

        if forced_sampling or should_use_template_subagents:
            branch_contexts = (
                self._build_sampling_branch_contexts(
                    query=user_text,
                    memory_ctx=memory_ctx,
                    min_branches=self._min_sampling_branches(),
                )
                if forced_sampling
                else self._build_subagent_branch_contexts(
                    query=user_text,
                    memory_ctx=memory_ctx,
                )
            )
            if len(branch_contexts) >= 2:
                branch_results = self._run_subagent_candidates(
                    user_text=user_text,
                    lang=lang,
                    branch_contexts=branch_contexts,
                    run_kwargs={**kwargs, "require_tool_use": require_tool_use},
                )
                selected = (
                    self._select_branch_answer_with_online_judge(
                        query=user_text,
                        memory_ctx=memory_ctx,
                        candidates=branch_results,
                    )
                    if forced_sampling
                    else self._select_subagent_answer(
                        query=user_text,
                        memory_ctx=memory_ctx,
                        candidates=branch_results,
                    )
                )
                if selected is not None:
                    enriched_ctx = copy.deepcopy(memory_ctx)
                    enriched_ctx["subagent_mode_triggered"] = True
                    enriched_ctx["sampling_mode_triggered"] = forced_sampling
                    enriched_ctx["subagent_candidates"] = [
                        item.to_selector_payload() for item in branch_results
                    ]
                    enriched_ctx["subagent_selected_candidate_ids"] = list(selected.get("selected_candidate_ids") or [])
                    enriched_ctx["subagent_selection_reason"] = str(selected.get("reason", "") or "").strip()
                    self._last_strategy_context = enriched_ctx
                    responses = self._compose_subagent_responses(
                        selected_candidates=selected.get("selected_candidates") or [],
                        final_answer=str(selected.get("final_answer", "") or ""),
                    )
                    return self._finalize_responses(
                        question=user_text,
                        responses=responses,
                        memory_ctx=enriched_ctx,
                        lang=lang,
                        session_id=session_id,
                        reference_answer=reference_answer,
                        online_learning=online_learning,
                        online_source=online_source,
                        online_depth=online_depth,
                    )

        responses = self._run_single_route(
            user_text=user_text,
            lang=lang,
            memory_ctx=memory_ctx,
            use_assistant_cache=True,
            require_tool_use=require_tool_use,
            **kwargs,
        )
        return self._finalize_responses(
            question=user_text,
            responses=responses,
            memory_ctx=memory_ctx,
            lang=lang,
            session_id=session_id,
            reference_answer=reference_answer,
            online_learning=online_learning,
            online_source=online_source,
            online_depth=online_depth,
        )

    def _min_sampling_branches(self) -> int:
        sm_cfg = getattr(self.config, "strategy_memory", None)
        return max(1, int(getattr(sm_cfg, "min_sampling_branches", 5) or 5))

    def _should_force_sampling_branches(self, *, memory_ctx: Dict[str, Any]) -> bool:
        min_branches = self._min_sampling_branches()
        if min_branches <= 1:
            return False
        candidate_patterns = [
            row for row in (memory_ctx.get("candidate_patterns") or [])
            if isinstance(row, dict)
        ]
        return len(candidate_patterns) < min_branches

    def _build_sampling_branch_contexts(
        self,
        *,
        query: str,
        memory_ctx: Dict[str, Any],
        min_branches: int,
        pattern_pool: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        query_pattern = memory_ctx.get("query_pattern") if isinstance(memory_ctx.get("query_pattern"), dict) else {}
        out: List[Dict[str, Any]] = []
        seen_template_ids: set[str] = set()
        target_total = max(2, int(min_branches))
        target_non_free = max(1, target_total - 1)

        def add_pattern_branch(pattern: Dict[str, Any]) -> None:
            template_id = str(pattern.get("template_id", "") or "").strip()
            if template_id and template_id in seen_template_ids:
                return
            if self.strategy_memory is not None:
                branch_memory_ctx = self.strategy_memory.build_context_for_patterns(
                    query=query,
                    query_pattern=query_pattern,
                    candidate_patterns=[pattern],
                    patterns=[pattern],
                )
            else:
                branch_memory_ctx = {
                    "query_abstract": str(memory_ctx.get("query_abstract", "") or "").strip(),
                    "query_pattern": copy.deepcopy(query_pattern),
                    "candidate_patterns": [copy.deepcopy(pattern)],
                    "patterns": [copy.deepcopy(pattern)],
                    "routing_hint": "",
                }
            branch_memory_ctx["subagent_candidate_id"] = f"candidate_{len(out) + 1}"
            branch_memory_ctx["subagent_template_id"] = template_id or f"template_hint_{len(out) + 1}"
            branch_memory_ctx["subagent_pattern_name"] = str(pattern.get("pattern_name", "") or "").strip()
            branch_memory_ctx["subagent_similarity"] = float(pattern.get("similarity", 0.0) or 0.0)
            branch_memory_ctx["subagent_recommended_chain"] = [
                str(x).strip() for x in (pattern.get("recommended_chain") or []) if str(x).strip()
            ]
            out.append(branch_memory_ctx)
            if template_id:
                seen_template_ids.add(template_id)

        pattern_rows = pattern_pool if pattern_pool is not None else (memory_ctx.get("candidate_patterns") or [])
        for row in pattern_rows:
            if isinstance(row, dict) and len(out) < target_non_free:
                add_pattern_branch(row)

        planner_needed = max(0, target_non_free - len(out))
        if planner_needed > 0:
            for spec in self._plan_sampling_branch_specs(
                question=query,
                planned_branch_count=planner_needed,
            ):
                out.append(
                    self._build_planned_sampling_branch_context(
                        memory_ctx=memory_ctx,
                        query_pattern=query_pattern,
                        candidate_index=len(out) + 1,
                        spec=spec,
                    )
                )
                if len(out) >= target_non_free:
                    break

        out.append(
            self._build_free_exploration_branch_context(
                memory_ctx=memory_ctx,
                query_pattern=query_pattern,
                candidate_index=len(out) + 1,
            )
        )
        return out[:target_total]

    def _plan_sampling_branch_specs(
        self,
        *,
        question: str,
        planned_branch_count: int,
    ) -> List[Dict[str, str]]:
        planner = getattr(self, "sampling_branch_planner", None)
        if planner is None:
            planner = OnlineSamplingBranchPlanner(prompt_loader=None, llm=None)
        strategy_prior_knowledge = ""
        if self.strategy_memory is not None:
            try:
                strategy_prior_knowledge = self.strategy_memory.build_sampling_branch_prior_knowledge()
            except Exception as exc:
                logger.warning("sampling planner prior knowledge build failed: %s", exc)
                strategy_prior_knowledge = ""
        return planner.plan(
            question=question,
            available_tools=self._build_sampling_available_tools(),
            planned_branch_count=planned_branch_count,
            strategy_prior_knowledge=strategy_prior_knowledge,
        )

    def _build_sampling_available_tools(self) -> List[Dict[str, str]]:
        provider = getattr(self, "tool_metadata_provider", None)
        out: List[Dict[str, str]] = []
        seen: set[str] = set()
        for tool in self._all_tools():
            tool_name = str(getattr(tool, "name", "") or "").strip()
            if not tool_name or tool_name in seen:
                continue
            seen.add(tool_name)
            fallback_description = str(getattr(tool, "description", "") or "").strip()
            meta = (
                provider.resolve_tool_metadata(tool_name, fallback_description=fallback_description)
                if provider is not None
                else {"name": tool_name, "description": fallback_description}
            )
            out.append(
                {
                    "name": str(meta.get("name", "") or tool_name).strip(),
                    "description": str(meta.get("description", "") or fallback_description).strip(),
                }
            )
        return out

    @staticmethod
    def _build_sampling_branch_routing_hint(spec: Dict[str, str]) -> str:
        name = str(spec.get("name", "") or "").strip()
        focus = str(spec.get("focus", "") or "").strip()
        tool_hint = str(spec.get("tool_hint", "") or "").strip()
        constraint = str(spec.get("constraint", "") or "").strip()
        lines = [
            "Sampling Branch Plan:",
            f"- name: {name}" if name else "- name: sampling_branch",
            f"- focus: {focus}" if focus else "- focus: gather evidence that can directly resolve the question",
            f"- tool_hint: {tool_hint}" if tool_hint else "- tool_hint: adaptive retrieval",
            f"- constraint: {constraint}" if constraint else "- constraint: avoid weak or redundant evidence",
            "Treat this as a soft branch bias only. Final decisions must follow retrieved evidence.",
        ]
        return "\n".join(lines)

    def _build_planned_sampling_branch_context(
        self,
        *,
        memory_ctx: Dict[str, Any],
        query_pattern: Dict[str, Any],
        candidate_index: int,
        spec: Dict[str, str],
    ) -> Dict[str, Any]:
        branch_name = str(spec.get("name", "") or f"planned_branch_{candidate_index}").strip()
        tool_hint = str(spec.get("tool_hint", "") or "").strip()
        return {
            "query_abstract": str(memory_ctx.get("query_abstract", "") or "").strip(),
            "query_pattern": copy.deepcopy(query_pattern),
            "candidate_patterns": [],
            "patterns": [],
            "routing_hint": self._build_sampling_branch_routing_hint(spec),
            "subagent_candidate_id": f"candidate_{candidate_index}",
            "subagent_template_id": f"planned_sampling_{candidate_index}",
            "subagent_pattern_name": branch_name,
            "subagent_similarity": 0.0,
            "subagent_recommended_chain": [tool_hint] if tool_hint else [],
            "sampling_branch_spec": copy.deepcopy(spec),
        }

    def _build_free_exploration_branch_context(
        self,
        *,
        memory_ctx: Dict[str, Any],
        query_pattern: Dict[str, Any],
        candidate_index: int,
    ) -> Dict[str, Any]:
        spec = {
            "name": "free_exploration",
            "focus": "explore the question directly without committing to a fixed retrieval bias too early",
            "tool_hint": "adaptive retrieval based on current evidence",
            "constraint": "do not anchor on the first plausible clue; verify before concluding",
        }
        return {
            "query_abstract": str(memory_ctx.get("query_abstract", "") or "").strip(),
            "query_pattern": copy.deepcopy(query_pattern),
            "candidate_patterns": [],
            "patterns": [],
            "routing_hint": self._build_sampling_branch_routing_hint(spec),
            "subagent_candidate_id": f"candidate_{candidate_index}",
            "subagent_template_id": "free_exploration",
            "subagent_pattern_name": "free_exploration",
            "subagent_similarity": 0.0,
            "subagent_recommended_chain": [],
            "sampling_branch_spec": copy.deepcopy(spec),
        }

    @staticmethod
    def _looks_like_manual_tool_request(text: str) -> bool:
        raw = str(text or "").strip()
        if not raw:
            return False
        try:
            payload = json.loads(raw)
            if isinstance(payload, dict):
                return bool(payload.get("tool_calls") or payload.get("function_call"))
        except Exception:
            pass
        return '"tool_calls"' in raw or '"function_call"' in raw

    def _run_single_route(
        self,
        *,
        user_text: str,
        lang: Literal["zh", "en"],
        memory_ctx: Dict[str, Any],
        use_assistant_cache: bool,
        require_tool_use: bool,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        messages: List[Dict[str, Any]] = [{"role": "user", "content": user_text}]
        max_llm_calls_per_run = kwargs.pop("max_llm_calls_per_run", None)
        runtime_system_message = self._build_runtime_system_message(memory_ctx=memory_ctx)
        tool_subsets = self._build_tool_execution_plan(query=user_text, memory_ctx=memory_ctx)
        original_max_llm_calls = getattr(qwen_fncall_agent, "MAX_LLM_CALL_PER_RUN", None)
        last_error: Optional[Exception] = None

        try:
            if isinstance(max_llm_calls_per_run, (int, float)) and int(max_llm_calls_per_run) > 0:
                qwen_fncall_agent.MAX_LLM_CALL_PER_RUN = int(max_llm_calls_per_run)
            for stage_idx, subset in enumerate(tool_subsets):
                assistant = self._get_or_create_assistant_for_subset(
                    tools=subset,
                    system_message=runtime_system_message,
                    use_cache=use_assistant_cache,
                )
                try:
                    responses = assistant.run_nonstream(messages=messages, lang=lang, **kwargs)
                except Exception as e:
                    last_error = e
                    if self._is_context_length_error(e):
                        logger.warning(
                            "Assistant run failed due to context length: stage=%s tools=%s err=%s",
                            stage_idx,
                            [getattr(t, "name", "") for t in subset],
                            e,
                        )
                        raise
                    logger.warning(
                        "Assistant run failed: stage=%s tools=%s err=%s",
                        stage_idx,
                        [getattr(t, "name", "") for t in subset],
                        e,
                    )
                    continue

                tool_uses = self.extract_tool_uses(responses)
                final_text = self.extract_final_text(responses)
                final_text_is_tool_request = self._looks_like_manual_tool_request(final_text)
                if require_tool_use:
                    if tool_uses and not final_text_is_tool_request:
                        return responses
                    if tool_uses and final_text_is_tool_request and stage_idx < (len(tool_subsets) - 1):
                        continue
                    if tool_uses and final_text_is_tool_request:
                        return responses
                    if stage_idx == (len(tool_subsets) - 1):
                        return self._compose_missing_tool_use_response(lang=lang)
                    continue
                if (
                    (tool_uses or final_text.strip())
                    and not final_text_is_tool_request
                ) or stage_idx == (len(tool_subsets) - 1):
                    return responses
            if last_error is not None:
                raise last_error
            return []
        finally:
            if original_max_llm_calls is not None:
                qwen_fncall_agent.MAX_LLM_CALL_PER_RUN = original_max_llm_calls

    @staticmethod
    def _compose_missing_tool_use_response(*, lang: Literal["zh", "en"]) -> List[Dict[str, Any]]:
        if lang == "zh":
            message = "启用了强制检索模式，但模型在本轮中没有调用任何工具。"
        else:
            message = "Mandatory retrieval mode is enabled, but no retrieval tool was called in this run."
        return [
            {
                "role": "assistant",
                "content": json.dumps(
                    {
                        "error": {
                            "code": "missing_tool_use",
                            "message": message,
                        }
                    },
                    ensure_ascii=False,
                ),
            }
        ]

    def _build_subagent_branch_contexts(
        self,
        *,
        query: str,
        memory_ctx: Dict[str, Any],
        candidate_patterns: Optional[List[Dict[str, Any]]] = None,
        include_no_strategy_baseline: bool = True,
        include_free_exploration: bool = False,
    ) -> List[Dict[str, Any]]:
        if self.strategy_memory is None:
            return []
        query_pattern = memory_ctx.get("query_pattern") if isinstance(memory_ctx.get("query_pattern"), dict) else {}
        if candidate_patterns is None:
            candidate_patterns = self.strategy_memory.deduplicate_patterns_for_subagents(
                candidate_patterns=memory_ctx.get("candidate_patterns") or [],
            )
        out: List[Dict[str, Any]] = []
        for idx, pattern in enumerate(candidate_patterns, start=1):
            branch_memory_ctx = self.strategy_memory.build_context_for_patterns(
                query=query,
                query_pattern=query_pattern,
                candidate_patterns=[pattern],
                patterns=[pattern],
            )
            branch_memory_ctx["subagent_candidate_id"] = f"candidate_{idx}"
            branch_memory_ctx["subagent_template_id"] = str(pattern.get("template_id", "") or "").strip()
            branch_memory_ctx["subagent_pattern_name"] = str(pattern.get("pattern_name", "") or "").strip()
            branch_memory_ctx["subagent_similarity"] = float(pattern.get("similarity", 0.0) or 0.0)
            branch_memory_ctx["subagent_recommended_chain"] = [
                str(x).strip() for x in (pattern.get("recommended_chain") or []) if str(x).strip()
            ]
            out.append(branch_memory_ctx)
        if out and include_no_strategy_baseline:
            out.append(
                {
                    "query_abstract": str(memory_ctx.get("query_abstract", "") or "").strip(),
                    "query_pattern": copy.deepcopy(query_pattern),
                    "candidate_patterns": [],
                    "patterns": [],
                    "routing_hint": "",
                    "subagent_candidate_id": f"candidate_{len(out) + 1}",
                    "subagent_template_id": "no_strategy_baseline",
                    "subagent_pattern_name": "no_strategy_baseline",
                    "subagent_similarity": 0.0,
                    "subagent_recommended_chain": [],
                }
            )
        if out and include_free_exploration:
            out.append(
                self._build_free_exploration_branch_context(
                    memory_ctx=memory_ctx,
                    query_pattern=query_pattern,
                    candidate_index=len(out) + 1,
                )
            )
        return out

    def _run_subagent_candidates(
        self,
        *,
        user_text: str,
        lang: Literal["zh", "en"],
        branch_contexts: List[Dict[str, Any]],
        run_kwargs: Dict[str, Any],
    ) -> List[StrategySubagentCandidate]:
        if not branch_contexts:
            return []
        max_workers = len(branch_contexts)
        candidates: List[StrategySubagentCandidate] = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {
                executor.submit(
                    self._run_single_route,
                    user_text=user_text,
                    lang=lang,
                    memory_ctx=copy.deepcopy(branch_memory_ctx),
                    use_assistant_cache=False,
                    **dict(run_kwargs or {}),
                ): branch_memory_ctx
                for branch_memory_ctx in branch_contexts
            }
            for future in as_completed(future_map):
                branch_memory_ctx = future_map[future]
                candidate_id = str(branch_memory_ctx.get("subagent_candidate_id", "") or "").strip()
                try:
                    responses = future.result()
                    tool_uses = self.extract_tool_uses(responses)
                    final_answer = self.extract_final_text(responses)
                    candidates.append(
                        StrategySubagentCandidate(
                            candidate_id=candidate_id,
                            template_id=str(branch_memory_ctx.get("subagent_template_id", "") or "").strip(),
                            pattern_name=str(branch_memory_ctx.get("subagent_pattern_name", "") or "").strip(),
                            similarity=float(branch_memory_ctx.get("subagent_similarity", 0.0) or 0.0),
                            recommended_chain=list(branch_memory_ctx.get("subagent_recommended_chain") or []),
                            query_abstract=str(branch_memory_ctx.get("query_abstract", "") or "").strip(),
                            routing_hint=str(branch_memory_ctx.get("routing_hint", "") or "").strip(),
                            answer=final_answer,
                            tool_uses=tool_uses,
                            responses=responses,
                        )
                    )
                except Exception as exc:
                    logger.warning("Subagent branch failed: candidate=%s err=%s", candidate_id, exc)
                    candidates.append(
                        StrategySubagentCandidate(
                            candidate_id=candidate_id,
                            template_id=str(branch_memory_ctx.get("subagent_template_id", "") or "").strip(),
                            pattern_name=str(branch_memory_ctx.get("subagent_pattern_name", "") or "").strip(),
                            similarity=float(branch_memory_ctx.get("subagent_similarity", 0.0) or 0.0),
                            recommended_chain=list(branch_memory_ctx.get("subagent_recommended_chain") or []),
                            query_abstract=str(branch_memory_ctx.get("query_abstract", "") or "").strip(),
                            routing_hint=str(branch_memory_ctx.get("routing_hint", "") or "").strip(),
                            answer="",
                            tool_uses=[],
                            responses=[],
                            error=str(exc).strip(),
                        )
                    )
        candidates.sort(key=lambda item: (item.similarity, len(item.answer or "")), reverse=True)
        return candidates

    def _select_subagent_answer(
        self,
        *,
        query: str,
        memory_ctx: Dict[str, Any],
        candidates: List[StrategySubagentCandidate],
    ) -> Optional[Dict[str, Any]]:
        usable = [
            item for item in (candidates or [])
            if item.answer.strip() or item.tool_uses or item.responses
        ]
        if not usable:
            return None
        if len(usable) == 1 or self.candidate_answer_selector is None:
            picked = usable[0]
            return {
                "selected_candidate_ids": [picked.candidate_id],
                "selected_candidates": [picked],
                "final_answer": picked.answer,
                "reason": "single_available_candidate",
            }
        result = self.candidate_answer_selector.select(
            question=query,
            query_abstract=str(memory_ctx.get("query_abstract", "") or "").strip(),
            candidate_answers=[item.to_selector_payload() for item in usable],
        )
        selected_ids = [
            str(x).strip()
            for x in (result.get("selected_candidate_ids") or [])
            if str(x).strip()
        ]
        selected_candidates = [item for item in usable if item.candidate_id in selected_ids]
        if not selected_candidates:
            selected_candidates = [usable[0]]
            selected_ids = [usable[0].candidate_id]
        final_answer = str(result.get("final_answer", "") or "").strip() or selected_candidates[0].answer
        return {
            "selected_candidate_ids": selected_ids,
            "selected_candidates": selected_candidates,
            "final_answer": final_answer,
            "reason": str(result.get("reason", "") or "").strip(),
        }

    def _select_branch_answer_with_online_judge(
        self,
        *,
        query: str,
        memory_ctx: Optional[Dict[str, Any]],
        candidates: List[StrategySubagentCandidate],
    ) -> Optional[Dict[str, Any]]:
        usable = [
            item for item in (candidates or [])
            if item.answer.strip() or item.tool_uses or item.responses
        ]
        if not usable:
            return None
        judge = self.branch_answer_judge or self.online_answer_judge
        if judge is None:
            picked = usable[0]
            return {
                "selected_candidate_ids": [picked.candidate_id],
                "selected_candidates": [picked],
                "final_answer": picked.answer,
                "reason": "branch_judge_unavailable",
            }

        scored: List[tuple[tuple[float, float, float, float, int, int], StrategySubagentCandidate]] = []
        for item in usable:
            evidence_summary = self._build_online_evidence_summary(item.tool_uses, max_chars=3200)
            tool_summary_json = self._build_online_tool_summary_json(item.tool_uses)
            selection_judge = judge.evaluate(
                question=query,
                candidate_answer=item.answer,
                reference_answer="",
                evidence_summary=evidence_summary,
                tool_summary_json=tool_summary_json,
            )
            item.selection_judge = dict(selection_judge or {})
            rank_key = (
                1.0 if bool(selection_judge.get("is_success", False)) else 0.0,
                float(selection_judge.get("score", 0.0) or 0.0),
                float(selection_judge.get("answer_support_score", 0.0) or 0.0),
                float(selection_judge.get("intermediate_value_score", 0.0) or 0.0),
                -len(str(item.answer or "").strip()),
                -len(item.tool_uses or []),
            )
            scored.append((rank_key, item))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        ranked_candidates = [item for _, item in scored]
        picked = ranked_candidates[0]
        fallback_reason = str((picked.selection_judge or {}).get("reason", "") or "").strip() or "branch_score_selected"
        if len(ranked_candidates) == 1 or self.candidate_answer_selector is None:
            return {
                "selected_candidate_ids": [picked.candidate_id],
                "selected_candidates": [picked],
                "final_answer": picked.answer,
                "reason": fallback_reason if len(ranked_candidates) == 1 else "branch_score_selected_without_selector",
            }

        selector_result = self.candidate_answer_selector.select(
            question=query,
            query_abstract=str((memory_ctx or {}).get("query_abstract", "") or "").strip(),
            candidate_answers=[item.to_selector_payload() for item in ranked_candidates],
        )
        selected_ids = [
            str(x).strip()
            for x in (selector_result.get("selected_candidate_ids") or [])
            if str(x).strip()
        ]
        selected_candidates = [
            item for item in ranked_candidates
            if item.candidate_id in selected_ids
        ]
        if not selected_candidates:
            selected_candidates = [picked]
            selected_ids = [picked.candidate_id]
        final_answer = str(selector_result.get("final_answer", "") or "").strip() or picked.answer
        reason = str(selector_result.get("reason", "") or "").strip() or fallback_reason
        return {
            "selected_candidate_ids": selected_ids,
            "selected_candidates": selected_candidates,
            "final_answer": final_answer,
            "reason": reason,
        }

    @staticmethod
    def _compose_subagent_responses(
        *,
        selected_candidates: List[StrategySubagentCandidate],
        final_answer: str,
    ) -> List[Dict[str, Any]]:
        responses: List[Dict[str, Any]] = []
        for item in selected_candidates or []:
            responses.extend(item.responses or [])
        responses.append({"role": "assistant", "content": str(final_answer or "").strip()})
        return responses

    def _finalize_responses(
        self,
        *,
        question: str,
        responses: List[Dict[str, Any]],
        memory_ctx: Dict[str, Any],
        lang: Literal["zh", "en"],
        session_id: str,
        reference_answer: str,
        online_learning: bool,
        online_source: str,
        online_depth: int,
    ) -> List[Dict[str, Any]]:
        if online_learning:
            try:
                self._run_online_learning_cycle(
                    question=question,
                    responses=responses,
                    memory_ctx=memory_ctx,
                    lang=lang,
                    session_id=session_id,
                    reference_answer=reference_answer,
                    online_source=online_source,
                    online_depth=online_depth,
                )
            except Exception as e:
                logger.warning("Online learning cycle failed: %s", e)
        return responses

    @staticmethod
    def _clip_runtime_text(value: Any, *, limit: int) -> str:
        text = str(value or "").strip()
        if limit <= 0 or len(text) <= limit:
            return text
        marker = "\n...[truncated]...\n"
        head = max(0, int(limit * 0.7))
        tail = max(0, limit - head - len(marker))
        if tail <= 0:
            return text[:limit]
        return text[:head] + marker + text[-tail:]

    def _summarize_tool_uses_for_online(self, tool_uses: List[Dict[str, Any]], *, max_items: int = 8) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for item in (tool_uses or [])[: max(1, int(max_items or 1))]:
            if not isinstance(item, dict):
                continue
            rows.append(
                {
                    "tool_name": str(item.get("tool_name", "") or "").strip(),
                    "tool_arguments": self._clip_runtime_text(item.get("tool_arguments", ""), limit=300),
                    "tool_output_summary": self._clip_runtime_text(item.get("tool_output", ""), limit=700),
                }
            )
        return rows

    def _build_online_evidence_summary(self, tool_uses: List[Dict[str, Any]], *, max_chars: int) -> str:
        lines: List[str] = []
        for idx, item in enumerate(tool_uses or [], start=1):
            if not isinstance(item, dict):
                continue
            tool_name = str(item.get("tool_name", "") or "").strip()
            tool_output = self._clip_runtime_text(item.get("tool_output", ""), limit=900)
            if not tool_name and not tool_output:
                continue
            lines.append(f"[{idx}] {tool_name}\n{tool_output}".strip())
            text = "\n\n".join(lines)
            if len(text) >= max_chars:
                return self._clip_runtime_text(text, limit=max_chars)
        return self._clip_runtime_text("\n\n".join(lines), limit=max_chars)

    def _build_online_tool_summary_json(self, tool_uses: List[Dict[str, Any]]) -> str:
        max_items = int(getattr(self.config.strategy_memory, "online_max_tool_items", 8) or 8)
        payload = self._summarize_tool_uses_for_online(tool_uses, max_items=max_items)
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def _run_online_learning_cycle(
        self,
        *,
        question: str,
        responses: List[Dict[str, Any]],
        memory_ctx: Dict[str, Any],
        lang: Literal["zh", "en"],
        session_id: str,
        reference_answer: str,
        online_source: str,
        online_depth: int,
    ) -> None:
        sm_cfg = getattr(self.config, "strategy_memory", None)
        if not bool(getattr(sm_cfg, "online_enabled", False)):
            return
        if self.online_buffer is None or self.online_answer_judge is None:
            return

        final_answer = self.extract_final_text(responses)
        tool_uses = self.extract_tool_uses(responses)
        tool_summary_json = self._build_online_tool_summary_json(tool_uses)
        max_evidence_chars = int(getattr(sm_cfg, "online_max_evidence_chars", 4000) or 4000)
        evidence_summary = self._build_online_evidence_summary(tool_uses, max_chars=max_evidence_chars)
        judge = self.online_answer_judge.evaluate(
            question=question,
            candidate_answer=final_answer,
            reference_answer=reference_answer,
            evidence_summary=evidence_summary,
            tool_summary_json=tool_summary_json,
        )

        trace_record = {
            "timestamp_ms": now_ms(),
            "question": question,
            "final_answer": final_answer,
            "reference_answer": reference_answer,
            "lang": lang,
            "doc_type": self.doc_type,
            "aggregation_mode": self.aggregation_mode,
            "session_id": session_id,
            "online_source": online_source,
            "online_depth": online_depth,
            "memory_context": {
                "query_abstract": str(memory_ctx.get("query_abstract", "") or "").strip(),
                "routing_hint": self._clip_runtime_text(memory_ctx.get("routing_hint", ""), limit=1200),
                "matched_template_ids": [
                    str(x.get("template_id", "") or "").strip()
                    for x in (memory_ctx.get("patterns") or [])
                    if isinstance(x, dict) and str(x.get("template_id", "") or "").strip()
                ],
            },
            "tool_uses": tool_uses,
            "tool_summary": json.loads(tool_summary_json or "[]"),
            "evidence_summary": evidence_summary,
            "judge": judge,
        }
        self.online_buffer.append_real_trace(trace_record)

        success = bool(judge.get("is_success", False))
        score = float(judge.get("score", 0.0) or 0.0)
        has_tool_use = bool(tool_uses)
        if has_tool_use and success and score >= float(getattr(sm_cfg, "online_strategy_min_score", 0.85) or 0.85):
            self._record_online_strategy_candidate(
                question=question,
                final_answer=final_answer,
                reference_answer=reference_answer,
                tool_uses=tool_uses,
                tool_summary_json=tool_summary_json,
                evidence_summary=evidence_summary,
                judge=judge,
                memory_ctx=memory_ctx,
                session_id=session_id,
                online_source=online_source,
            )
        else:
            self._record_online_failure(
                question=question,
                final_answer=final_answer,
                reference_answer=reference_answer,
                tool_summary_json=tool_summary_json,
                evidence_summary=evidence_summary,
                judge=judge,
                session_id=session_id,
                online_source=online_source,
            )

        if (
            online_source == "real"
            and online_depth <= 0
            and bool(getattr(sm_cfg, "self_bootstrap_enabled", False))
            and self.self_bootstrap_manager is not None
            and score >= float(getattr(sm_cfg, "self_bootstrap_min_source_score", 0.8) or 0.8)
        ):
            self._run_self_bootstrap_cycle(
                question=question,
                final_answer=final_answer,
                evidence_summary=evidence_summary,
                tool_summary_json=tool_summary_json,
                lang=lang,
            )

    def _record_online_strategy_candidate(
        self,
        *,
        question: str,
        final_answer: str,
        reference_answer: str,
        tool_uses: List[Dict[str, Any]],
        tool_summary_json: str,
        evidence_summary: str,
        judge: Dict[str, Any],
        memory_ctx: Dict[str, Any],
        session_id: str,
        online_source: str,
    ) -> None:
        if (
            self.online_buffer is None
            or self.online_query_pattern_extractor is None
            or self.effective_tool_chain_extractor is None
            or self.strategy_template_distiller is None
        ):
            return
        analysis_reference = reference_answer or final_answer
        query_pattern = self.online_query_pattern_extractor.extract(question)
        effective_chain = self.effective_tool_chain_extractor.extract(
            question=question,
            reference_answer=analysis_reference,
            candidate_answer=final_answer,
            tool_uses=tool_uses,
        )
        raw_tool_chain = [str(item.get("tool_name", "") or "").strip() for item in (tool_uses or []) if str(item.get("tool_name", "") or "").strip()]
        best_attempt = {
            "question": question,
            "query_pattern": query_pattern,
            "candidate_answer": final_answer,
            "judge": judge,
            "raw_tool_chain": raw_tool_chain,
            "minimal_effective_chain": effective_chain.get("minimal_effective_chain") or effective_chain.get("effective_tool_chain") or raw_tool_chain,
            "effective_tool_chain": effective_chain.get("effective_tool_chain") or raw_tool_chain,
            "step_attributions": effective_chain.get("step_attributions") or [],
            "effective_chain_reason": str(effective_chain.get("reason", "") or "").strip(),
            "tool_summary_json": tool_summary_json,
            "evidence_summary": evidence_summary,
        }
        distilled = self.strategy_template_distiller.distill(
            question=question,
            query_pattern=query_pattern,
            best_attempt=best_attempt,
            failed_attempts=[],
            retry_instruction="",
        )
        self.online_buffer.append_strategy_candidate(
            {
                "timestamp_ms": now_ms(),
                "question": question,
                "query_pattern": query_pattern,
                "pattern_name": str(distilled.get("pattern_name", "") or "").strip(),
                "pattern_description": str(distilled.get("pattern_description", "") or "").strip(),
                "minimal_effective_chain": effective_chain.get("minimal_effective_chain") or effective_chain.get("effective_tool_chain") or [],
                "recommended_chain": list(distilled.get("recommended_chain") or []),
                "anti_patterns": list(distilled.get("anti_patterns") or []),
                "chain_rationale": str(distilled.get("chain_rationale", "") or "").strip(),
                "chain_constraints": list(distilled.get("chain_constraints") or []),
                "judge": judge,
                "final_answer": final_answer,
                "reference_answer": reference_answer,
                "tool_uses": tool_uses,
                "raw_tool_chain": raw_tool_chain,
                "effective_tool_chain": effective_chain.get("effective_tool_chain") or [],
                "effective_step_indices": effective_chain.get("effective_step_indices") or [],
                "discarded_step_indices": effective_chain.get("discarded_step_indices") or [],
                "step_attributions": effective_chain.get("step_attributions") or [],
                "session_id": session_id,
                "online_source": online_source,
                "matched_template_ids": [
                    str(x.get("template_id", "") or "").strip()
                    for x in (memory_ctx.get("patterns") or [])
                    if isinstance(x, dict) and str(x.get("template_id", "") or "").strip()
                ],
            }
        )

    def _record_online_failure(
        self,
        *,
        question: str,
        final_answer: str,
        reference_answer: str,
        tool_summary_json: str,
        evidence_summary: str,
        judge: Dict[str, Any],
        session_id: str,
        online_source: str,
    ) -> None:
        if self.online_buffer is None:
            return
        reflection: Dict[str, Any] = {}
        retry_instruction = ""
        reference_like = reference_answer or evidence_summary
        if self.failed_answer_reflector is not None:
            reflection = self.failed_answer_reflector.reflect(
                question=question,
                reference_answer=reference_like,
                candidate_answer=final_answer,
                tool_summary_json=tool_summary_json,
            )
        if reflection.get("need_retry") and self.retry_instruction_builder is not None:
            retry_payload = self.retry_instruction_builder.build(
                question=question,
                missed_fact=str(reflection.get("missed_fact", "") or "").strip(),
                next_action=str(reflection.get("next_action", "") or "").strip(),
            )
            retry_instruction = str(retry_payload.get("retry_instruction", "") or "").strip()
        self.online_buffer.append_failure_reflection(
            {
                "timestamp_ms": now_ms(),
                "question": question,
                "final_answer": final_answer,
                "reference_answer": reference_answer,
                "evidence_summary": evidence_summary,
                "tool_summary": json.loads(tool_summary_json or "[]"),
                "judge": judge,
                "reflection": reflection,
                "retry_instruction": retry_instruction,
                "session_id": session_id,
                "online_source": online_source,
            }
        )

    def _run_self_bootstrap_cycle(
        self,
        *,
        question: str,
        final_answer: str,
        evidence_summary: str,
        tool_summary_json: str,
        lang: Literal["zh", "en"],
    ) -> None:
        if self.online_buffer is None or self.self_bootstrap_manager is None:
            return
        sm_cfg = getattr(self.config, "strategy_memory", None)
        max_questions = int(getattr(sm_cfg, "self_bootstrap_max_questions", 3) or 3)
        if max_questions <= 0:
            return

        def _answer_fn(bootstrap_question: str) -> Dict[str, Any]:
            bootstrap_responses = self.ask(
                bootstrap_question,
                lang=lang,
                online_learning=False,
                _online_source="synthetic",
                _online_depth=1,
            )
            bootstrap_tool_uses = self.extract_tool_uses(bootstrap_responses)
            return {
                "answer": self.extract_final_text(bootstrap_responses),
                "tool_uses": bootstrap_tool_uses,
                "tool_summary_json": self._build_online_tool_summary_json(bootstrap_tool_uses),
            }

        sampling_attempts = int(getattr(sm_cfg, "self_bootstrap_sampling_attempts", 3) or 3)
        records = self.self_bootstrap_manager.run(
            question=question,
            final_answer=final_answer,
            evidence_summary=evidence_summary,
            tool_summary_json=tool_summary_json,
            answer_fn=_answer_fn,
            max_questions=max_questions,
            sampling_attempts=sampling_attempts,
        )
        for row in records:
            if not isinstance(row, dict):
                continue
            payload = dict(row)
            payload["timestamp_ms"] = now_ms()
            payload["source_question"] = question
            payload["source_answer"] = final_answer
            self.online_buffer.append_synthetic_qa(payload)

    def get_last_strategy_context(self) -> Dict[str, Any]:
        return copy.deepcopy(self._last_strategy_context or {})

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

    def _build_aggregation_tools(self) -> List[Any]:
        emb_cfg = self.config.embedding
        community_tools = [
            SearchCommunities(
                self.graph_query_utils,
                emb_cfg,
                summary_vector_store=self.community_vector_store,
                document_parser=self.document_parser,
                max_workers=getattr(self.config.document_processing, "max_workers", 8),
            ),
            CommunityGraphRAGSearch(
                self.graph_query_utils,
                self.document_vector_store,
                self.document_parser,
                community_summary_vector_store=self.community_vector_store,
                embedding_config=emb_cfg,
                max_workers=getattr(self.config.document_processing, "max_workers", 8),
            ),
        ]
        narrative_tools = [
            NarrativeHierarchicalSearch(
                self.graph_query_utils,
                self.document_vector_store,
                self.document_parser,
                sentence_vector_store=self.sentence_vector_store,
                embedding_config=emb_cfg,
                max_workers=getattr(self.config.document_processing, "max_workers", 8),
                narrative_retriever=self.narrative_tree_retriever,
            ),
        ]
        if self.aggregation_mode == "community":
            return community_tools
        if self.aggregation_mode == "full":
            return community_tools + narrative_tools
        return narrative_tools

    def _build_base_tools(self, *, doc_path: Optional[str], reranker: Any) -> List[Any]:
        emb_cfg = self.config.embedding
        tools: List[Any] = [
            EntityRetrieverName(self.graph_query_utils, emb_cfg),
            EntityRetrieverID(self.graph_query_utils, emb_cfg),
            SearchSections(
                self.graph_query_utils,
                emb_cfg,
                self.doc_type,
                document_parser=self.document_parser,
                max_workers=getattr(self.config.document_processing, "max_workers", 8),
            ),
            SearchRelatedEntities(self.graph_query_utils, emb_cfg),
            GetEntitySections(self.graph_query_utils),
            GetRelationsBetweenEntities(self.graph_query_utils),
            GetCommonNeighbors(self.graph_query_utils),
            QuerySimilarFacts(self.graph_query_utils, emb_cfg),
            FindPathsBetweenNodes(self.graph_query_utils),
            TopKByCentrality(self.graph_query_utils),
            GetCoSectionEntities(self.graph_query_utils),
            GetKHopSubgraph(self.graph_query_utils),
            VDBDocsSearchTool(self.document_vector_store),
            VDBGetDocsByDocumentIDsTool(self.document_vector_store),
            VDBSentencesSearchTool(self.sentence_vector_store),
            SectionEvidenceSearch(
                self.graph_query_utils,
                self.document_vector_store,
                self.document_parser,
                sentence_vector_store=self.sentence_vector_store,
                doc_type=self.doc_type,
                embedding_config=emb_cfg,
                max_workers=getattr(self.config.document_processing, "max_workers", 8),
                section_retriever=self.section_tree_retriever,
            ),
            ChoiceGroundedEvidenceSearch(
                self.graph_query_utils,
                self.document_vector_store,
                self.document_parser,
                sentence_vector_store=self.sentence_vector_store,
                max_workers=getattr(self.config.document_processing, "max_workers", 8),
                section_retriever=self.section_tree_retriever,
                narrative_retriever=self.narrative_tree_retriever,
            ),
            EntityEventTraceSearch(
                self.graph_query_utils,
                self.document_vector_store,
                self.document_parser,
                sentence_vector_store=self.sentence_vector_store,
                max_workers=getattr(self.config.document_processing, "max_workers", 8),
                section_retriever=self.section_tree_retriever,
                narrative_retriever=self.narrative_tree_retriever,
                interaction_db_path=self.db_path or "",
                doc_type=self.doc_type,
            ),
            NarrativeCausalTraceSearch(
                self.graph_query_utils,
                self.document_vector_store,
                self.document_parser,
                sentence_vector_store=self.sentence_vector_store,
                embedding_config=emb_cfg,
                max_workers=getattr(self.config.document_processing, "max_workers", 8),
                narrative_retriever=self.narrative_tree_retriever,
                llm=self.retriever_llm,
            ),
        ]
        native_tools = self._build_native_tools(doc_path=doc_path)
        bm25_tool = next((tool for tool in native_tools if getattr(tool, "name", "") == "bm25_search_docs"), None)
        tools.append(
            HybridEvidenceSearch(
                self.graph_query_utils,
                self.document_vector_store,
                self.document_parser,
                sentence_vector_store=self.sentence_vector_store,
                doc_type=self.doc_type,
                embedding_config=emb_cfg,
                max_workers=getattr(self.config.document_processing, "max_workers", 8),
                section_retriever=self.section_tree_retriever,
                bm25_tool=bm25_tool,
                llm=self.retriever_llm,
            )
        )
        tools.extend(native_tools)
        return tools

    def _build_native_tools(self, *, doc_path: Optional[str]) -> List[Any]:
        """
        Build native tools:
        - BM25SearchDocsTool (always)
        - (optional) SQL tools when `enable_sql_tools=True`
        """
        # Load raw chunks for BM25
        base = doc_path or os.path.join(self.knowledge_graph_path, "all_document_chunks.json")
        with open(base, "r", encoding="utf-8") as f:
            data = json.load(f)

        keys_to_drop = {"chunk_index", "chunk_type", "doc_title", "order", "total_doc_chunks"}
        base_documents: List[Document] = []
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
            base_documents.append(Document(page_content=content, metadata=meta))

        bm25_chunk_size = int(getattr(self.config.document_processing, "bm25_chunk_size", 250) or 0)
        documents = _split_bm25_documents(base_documents, chunk_size=bm25_chunk_size)
        logger.info(
            "BM25 corpus prepared: raw_chunks=%d bm25_docs=%d bm25_chunk_size=%d",
            len(base_documents),
            len(documents),
            bm25_chunk_size,
        )

        native_tools: List[Any] = [BM25SearchDocsTool(documents)]
        native_tools.append(
            LookupTitlesByDocumentIDsTool(
                index_path=os.path.join(self.knowledge_graph_path, "doc2chunks_index.json"),
                doc2chunks_path=os.path.join(self.knowledge_graph_path, "doc2chunks.json"),
                doc_type=self.doc_type,
            )
        )
        native_tools.append(
            LookupDocumentIDsByTitleTool(
                index_path=os.path.join(self.knowledge_graph_path, "doc2chunks_index.json"),
                doc2chunks_path=os.path.join(self.knowledge_graph_path, "doc2chunks.json"),
                doc_type=self.doc_type,
            )
        )
        native_tools.append(
            SearchRelatedContentTool(
                document_vector_store=self.document_vector_store,
                document_parser=self.document_parser,
                max_workers=getattr(self.config.document_processing, "max_workers", 8),
            )
        )

        # Attach SQL tools ONLY when enabled
        if self.enable_sql_tools:
            try:
                from core.functions.tool_calls.sqldb_tools import (
                    SQLSearchDialogues,
                    SQLSearchInteractions,
                    SQLGetInteractionsByDocumentIDs,
                )
                if not self.db_path:
                    self.db_path = os.path.join(self.config.storage.sql_database_path, "Interaction.db")

                native_tools.extend([
                    SQLSearchDialogues(self.db_path, doc_type=self.doc_type),
                    SQLSearchInteractions(self.db_path, doc_type=self.doc_type),
                    SQLGetInteractionsByDocumentIDs(self.db_path, doc_type=self.doc_type),
                ])
            except Exception as e:
                # If SQL tools are missing or DB not present, we silently continue without them.
                # You can switch to logging if you have a logger facility available.
                pass

        return native_tools

    def _all_tools(self) -> List[Any]:
        tools = [*self._base_tools, *self._aggregation_tools, *self._extra_tools]
        hidden = {
            str(name or "").strip()
            for name in getattr(self, "hidden_tool_names", set()) or set()
            if str(name or "").strip()
        }
        if not hidden:
            return tools
        return [
            tool
            for tool in tools
            if str(getattr(tool, "name", "") or "").strip() not in hidden
        ]

    def _build_runtime_system_message(self, *, memory_ctx: Dict[str, Any]) -> str:
        hint = str(memory_ctx.get("routing_hint", "") or "").strip()
        if bool(getattr(self, "qwen_native_tool_routing_only", False)):
            if hint:
                return f"{self._current_system_message}\n\n{hint}".strip()
            return self._current_system_message.strip()
        if not bool(getattr(self, "runtime_routing_note_enabled", False)):
            return self._current_system_message.strip()
        routing_note = (
            "You are a tool-using assistant. Start with the smallest relevant tool set. "
            "If one retrieval fails, try a different retrieval path instead of repeating the same failed call. "
            "Prefer BM25 keyword search to verify sensitive factual details such as appearances, model numbers, years, ages, and timelines. "
            "For scene-level evidence or fine-grained factual details, use `section_evidence_search` as a fallback."
        )
        if hint:
            routing_note = routing_note + "\n\n" + hint
        return f"{self._current_system_message}\n\n{routing_note}".strip()

    def _build_tool_execution_plan(self, *, query: str, memory_ctx: Dict[str, Any]) -> List[List[Any]]:
        if bool(getattr(self, "qwen_native_tool_routing_only", False)):
            return [self._all_tools()]
        return self.tool_router.build_tool_execution_plan(
            query=query,
            memory_ctx=memory_ctx,
            tools=self._all_tools(),
        )

    @staticmethod
    def _is_context_length_error(exc: Exception) -> bool:
        text = str(exc or "").lower()
        return ("maximum context length" in text) or ("max_tokens" in text and "too large" in text)

    def _get_or_create_assistant_for_subset(
        self,
        *,
        tools: List[Any],
        system_message: str,
        use_cache: bool = True,
    ) -> Assistant:
        key = (system_message, tuple(str(getattr(t, "name", "") or "").strip() for t in tools))
        if use_cache:
            cached = self._assistant_cache.get(key)
            if cached is not None:
                return cached
        assistant = Assistant(
            function_list=self._apply_tool_metadata(list(tools)),
            llm=self.llm_cfg,
            system_message=system_message,
            rag_cfg=self.rag_cfg,
        )
        if use_cache:
            self._assistant_cache[key] = assistant
        return assistant

    def _rebuild_assistant(self) -> None:
        tools = [*self._base_tools, *self._aggregation_tools, *self._extra_tools]
        tools = self._apply_tool_metadata(tools)
        self.assistant = Assistant(
            function_list=tools,
            llm=self.llm_cfg,
            system_message=self._current_system_message,
            rag_cfg=self.rag_cfg,
        )

    def _apply_tool_metadata(self, tools: List[Any]) -> List[Any]:
        provider = getattr(self, "tool_metadata_provider", None)
        out: List[Any] = []
        for tool in tools or []:
            current = tool
            if provider is not None:
                current = provider.apply_to_tool(current)
            tool_name = str(getattr(current, "name", "") or "").strip()
            char_limit = TOOL_OUTPUT_CHAR_LIMITS.get(tool_name, DEFAULT_TOOL_OUTPUT_CHAR_LIMIT)
            out.append(_ContextBudgetToolWrapper(current, char_limit=char_limit))
        return out

    def close(self) -> None:
        try:
            self.graph_store.close()
        except Exception:
            pass

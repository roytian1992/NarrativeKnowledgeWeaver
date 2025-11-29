# core/builder/graph_builder.py
from __future__ import annotations

from collections import defaultdict
import networkx as nx
import json
import os
import sqlite3
import pickle
import multiprocessing as mp
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError, wait, FIRST_COMPLETED
from collections import defaultdict
from copy import deepcopy
from typing import Any, Awaitable, Callable, Dict, Hashable, Iterable, List, Tuple, Optional
import hashlib
import asyncio
import pandas as pd
import random
import re
import glob
import logging
from tqdm import tqdm
from core.utils.function_manager import run_async_with_retries
from core.utils.prompt_loader import PromptLoader
from core.utils.format import correct_json_format
from core.models.data import Entity, KnowledgeGraph, Relation, TextChunk, Document
from ..storage.graph_store import GraphStore
from ..storage.vector_store import VectorStore
from ..utils.config import KAGConfig
from ..utils.neo4j_utils import Neo4jUtils
from core.model_providers.openai_llm import OpenAILLM
from core.agent.knowledge_extraction_agent import InformationExtractionAgent
from core.agent.attribute_extraction_agent import AttributeExtractionAgent
from core.agent.graph_probing_agent import GraphProbingAgent
from .document_processor import DocumentProcessor
from core.builder.graph_preprocessor import GraphPreprocessor
from core.utils.format import DOC_TYPE_META
from core.builder.reflection import DynamicReflector
from langchain.text_splitter import RecursiveCharacterTextSplitter
from core.utils.function_manager import run_async_with_retries
import unicodedata
import re
import shutil

_BRACKET_NUM_SUFFIX_RE = re.compile(r'[\s\u3000]*[\(\[\{（【]\s*\d+\s*[\)\]\}）】]$')
# 任意内容的末尾括注（用于人物类），限长避免把整句吃掉：1~12 字符
_BRACKET_ANY_SUFFIX_RE = re.compile(r'[\s\u3000]*[\(\[\{（【]\s*([^\s()[\]{}（）【】]{1,12})\s*[\)\]\}）】]$')

_ZERO_WIDTH_RE = re.compile(r'[\u200b-\u200f\u202a-\u202e\u2060\uFEFF]')
_MULTISPACE_RE = re.compile(r'\s+')


logger = logging.getLogger(__name__)


def _normalize_type(t):
    """
    Normalize entity type.

    - Input can be str or list.
    - If contains 'Event', put 'Event' at the first position.
    - De-duplicate while preserving original order.
    - If empty -> return 'Concept'.
    """
    if not t:
        return "Concept"
    if isinstance(t, str):
        return t

    # list case
    seen = set()
    ordered = []
    for x in t:
        if x and x not in seen:
            seen.add(x)
            ordered.append(x)

    if "Event" in seen:
        ordered = ["Event"] + [x for x in ordered if x != "Event"]

    return ordered if len(ordered) > 1 else (ordered[0] if ordered else "Concept")


# ═════════════════════════════════════════════════════════════════════════════
#                               Builder
# ═════════════════════════════════════════════════════════════════════════════
class KnowledgeGraphBuilder:
    """Knowledge graph builder (supports multiple document formats)."""

    def __init__(self, config: KAGConfig, doc_type: str = None):
        self.doc_type = doc_type if doc_type else config.knowledge_graph_builder.doc_type
        if self.doc_type not in DOC_TYPE_META:
            raise ValueError(f"Unsupported doc_type: {self.doc_type}")

        self.config = config
        self.max_workers = config.knowledge_graph_builder.max_workers
        self.meta = DOC_TYPE_META[self.doc_type]
        self.section_chunk_ids = defaultdict(set)

        prompt_dir = config.knowledge_graph_builder.prompt_dir
        self.prompt_loader = PromptLoader(prompt_dir)
        self.llm = OpenAILLM(config)
        self.processor = DocumentProcessor(config, self.llm, self.doc_type)

        # Stores / Databases
        self.graph_store = GraphStore(config)
        self.neo4j_utils = Neo4jUtils(self.graph_store.driver, doc_type=self.doc_type)
        self.document_vector_store = VectorStore(config, "documents")
        self.sentence_vector_store = VectorStore(config, "sentences")

        # Memory / Reflection
        self.reflector = DynamicReflector(config)

        # Runtime data
        self.kg = KnowledgeGraph()
        self.probing_mode = self.config.probing.probing_mode
        self.graph_probing_agent = GraphProbingAgent(self.config, self.llm, self.reflector)

    def clear_directory(self, path: str):
        """
        Delete ALL contents under `path` (files + subdirectories),
        but keep the directory itself.
        """
        if not os.path.exists(path):
            return

        for entry in os.scandir(path):
            try:
                if entry.is_file() or entry.is_symlink():
                    os.remove(entry.path)
                elif entry.is_dir():
                    shutil.rmtree(entry.path)
            except Exception as e:
                logger.warning(f"Failed to delete {entry.path} -> {e}")

    def construct_system_prompt(self, background, abbreviations):
        background_info = self.get_background_info(background, abbreviations)
        system_prompt_id = "agent_prompt_screenplay" if self.doc_type == "screenplay" else "agent_prompt_novel"
        system_prompt_text = self.prompt_loader.render_prompt(system_prompt_id, {"background_info": background_info})
        return system_prompt_text

    def get_background_info(self, background, abbreviations):
        bg_block = f"**Background**: {background}\n" if background else ""

        def fmt(item: dict) -> str:
            """
            Turn an abbreviation item into a Markdown bullet.
            Any field is optional. Title preference: abbr > full > any non-empty field > N/A.
            """
            if not isinstance(item, dict):
                return ""

            title = (
                item.get("abbr")
                or item.get("full")
                or next((v for k, v in item.items() if isinstance(v, str) and v.strip()), "N/A")
            )

            parts = []
            for k, v in item.items():
                if k in ("abbr", "full"):
                    continue
                if isinstance(v, str) and v.strip():
                    parts.append(v.strip())

            return f"- **{title}**: " + " - ".join(parts) if parts else f"- **{title}**"

        abbr_block = "\n".join(fmt(item) for item in abbreviations if isinstance(item, dict)) if abbreviations else ""

        if bg_block and abbr_block:
            background_info = f"{bg_block}\n{abbr_block}"
        else:
            background_info = bg_block or abbr_block

        return background_info

    def prepare_chunks(
        self,
        json_file_path: str,
        verbose: bool = True,
        retries: int = 2,
        per_task_timeout: float = 600.0,
        retry_backoff: float = 1.0,  # kept for signature compatibility; not used by the unified runner
        *,
        use_semantic_split: bool = True,
        extract_summary: bool = True,
        extract_metadata: bool = True,
        extract_timelines: bool = True,
        summary_max_words: int = 200,
    ):
        """
        Build TextChunks from a JSON corpus using a single-pass pipeline:

        1) base_splitter.split_text(document) -> (optional) sliding_semantic_split(...)
        2) (optional) sliding-window summaries on the split chunks
        3) (optional) one-shot document-level metadata from (Title + Aggregated Summary),
           copied to each chunk as `doc_metadata`

        Concurrency, soft timeouts and retries are handled by
        `core.utils.function_manager.run_concurrent_with_retries`.

        Args:
            json_file_path: input JSON path
            verbose: whether to log progress info
            retries: total rounds (round 1 + (retries-1) retries)
            per_task_timeout: soft timeout per *document* (seconds), measured after task start
            retry_backoff: kept for compatibility (not used)
            use_semantic_split / extract_summary / extract_metadata / summary_max_words:
                control the single-pass behavior (see DocumentProcessor.prepare_chunk)
        """
        # Init / cleanup
        self.reflector.clear()
        base = self.config.storage.knowledge_graph_path
        self.clear_directory(base)

        if verbose:
            logger.info(f"Starting knowledge graph build from: {json_file_path}")
            logger.info("Loading documents...")

        # Load items; each item becomes ONE document
        documents = self.processor.load_from_json(json_file_path)
        n_docs = len(documents)

        if verbose:
            logger.info(f"Loaded {n_docs} documents")

        # Per-document task
        def _task(doc: Dict) -> Dict[str, List[TextChunk]]:
            # Expect a dict: {"document_chunks": List[TextChunk]}
            return self.processor.prepare_chunk(
                doc,
                use_semantic_split=use_semantic_split,
                extract_summary=extract_summary,
                extract_metadata=extract_metadata,
                extract_timelines=extract_timelines,
                summary_max_words=summary_max_words,
            )

        # Run concurrent with retries
        from core.utils.function_manager import run_concurrent_with_retries

        max_workers = getattr(self.processor, "max_workers", getattr(self, "max_workers", 4))
        results_map, failed_indices = run_concurrent_with_retries(
            items=documents,
            task_fn=_task,
            per_task_timeout=per_task_timeout,
            max_retry_rounds=max(1, retries),
            max_in_flight=max_workers,
            max_workers=max_workers,
            thread_name_prefix="chunk",
            desc_prefix="Chunking documents",
            treat_empty_as_failure=True,
            is_empty_fn=lambda r: (not r) or (not isinstance(r, dict)) or (not r.get("document_chunks")),
        )

        # Flatten results (keep only successful chunks)
        all_chunks: List[TextChunk] = []
        for i in range(n_docs):
            res = results_map.get(i)
            if not res:
                continue
            chunks = res.get("document_chunks", [])
            if isinstance(chunks, list):
                all_chunks.extend(chunks)

        # Persist successful chunks
        os.makedirs(base, exist_ok=True)
        out_path = os.path.join(base, "all_document_chunks.json")
        with open(out_path, "w", encoding="utf-8") as fw:
            json.dump([c.dict() for c in all_chunks], fw, ensure_ascii=False, indent=2)

        # Final report
        if verbose:
            logger.info(f"Generated {len(all_chunks)} chunks, written to: {out_path}")
            if failed_indices:
                logger.warning(f"{len(failed_indices)} documents failed after {retries} round(s): {sorted(failed_indices)}")

    # ═════════════════════════════════════════════════════════════════════
    #  2) Store chunks (RDB + VDB)
    # ═════════════════════════════════════════════════════════════════════
    def store_chunks(self, verbose: bool = True):
        base = self.config.storage.knowledge_graph_path

        # Description chunks
        doc_chunks = [TextChunk(**o) for o in
                      json.load(open(os.path.join(base, "all_document_chunks.json"), "r", encoding="utf-8"))]

        # Write into KG (Document + Chunk)
        for ch in doc_chunks:
            self.kg.add_document(self.processor.prepare_document(ch))
            self.kg.add_chunk(ch)

        # Vector DB
        if verbose:
            logger.info("Storing to vector databases...")
        self._store_vectordb(verbose)

    def run_graph_probing(self, verbose: bool = True, sample_ratio: float = None):
        """Run (or skip) schema probing and persist schema/settings."""
        self.reflector.clear()

        base = self.config.storage.graph_schema_path
        os.makedirs(base, exist_ok=True)
        self.clear_directory(base)

        if self.probing_mode in ("fixed", "adjust"):
            schema = json.load(open(self.config.probing.default_graph_schema_path, "r", encoding="utf-8"))
            if os.path.exists(self.config.probing.default_background_path):
                settings = json.load(open(self.config.probing.default_background_path, "r", encoding="utf-8"))
                if "abbreviations" not in settings:
                    settings["abbreviations"] = []
            else:
                settings = {"background": "", "abbreviations": []}
        else:
            schema = {}
            settings = {"background": "", "abbreviations": []}

        if self.probing_mode != "fixed":
            schema, settings = self.update_schema(
                schema,
                background=settings.get("background", ""),
                abbreviations=settings.get("abbreviations", []),
                verbose=verbose,
                sample_ratio=sample_ratio,
            )
        else:
            if verbose:
                logger.info("Skipping probing step (fixed schema mode).")

        with open(os.path.join(base, "graph_schema.json"), "w", encoding="utf-8") as f:
            json.dump(schema, f, ensure_ascii=False, indent=2)

        with open(os.path.join(base, "settings.json"), "w", encoding="utf-8") as f:
            json.dump(settings, f, ensure_ascii=False, indent=2)

    def initialize_agents(self):
        """Initialize extraction agents and graph preprocessor using persisted schema/settings."""
        base = self.config.storage.graph_schema_path
        schema_path = os.path.join(base, "graph_schema.json")
        settings_path = os.path.join(base, "settings.json")

        if os.path.exists(schema_path):
            schema = json.load(open(schema_path, "r", encoding="utf-8"))
        elif os.path.exists(self.config.probing.default_graph_schema_path):
            schema = json.load(open(self.config.probing.default_graph_schema_path, "r", encoding="utf-8"))
        else:
            raise FileNotFoundError("Graph schema not found. Provide a valid path or run probing first.")

        if os.path.exists(settings_path):
            settings = json.load(open(settings_path, "r", encoding="utf-8"))
        elif os.path.exists(self.config.probing.default_background_path):
            settings = json.load(open(self.config.probing.default_background_path, "r", encoding="utf-8"))
        else:
            settings = {"background": "", "abbreviations": []}

        self.system_prompt_text = self.construct_system_prompt(
            background=settings.get("background", ""),
            abbreviations=settings.get("abbreviations", ""),
        )

        # Agents
        self.information_extraction_agent = InformationExtractionAgent(
            self.config, self.llm, self.system_prompt_text, schema, self.reflector
        )
        self.attribute_extraction_agent = AttributeExtractionAgent(
            self.config, self.llm, self.system_prompt_text, schema
        )
        self.graph_preprocessor = GraphPreprocessor(self.config, self.llm, system_prompt=self.system_prompt_text)

    def update_schema(
        self,
        schema: Dict = {},
        background: str = "",
        abbreviations: List = [],
        verbose: bool = True,
        sample_ratio: float = None,
        documents: List[Document] = None,
    ):
        """
        Update schema by probing on sampled chunks and saving insights into reflection memory.
        Returns (schema, settings).
        """
        if documents:
            doc_chunks = documents
        else:
            base = self.config.storage.knowledge_graph_path
            doc_chunks = [TextChunk(**o) for o in
                          json.load(open(os.path.join(base, "all_document_chunks.json"), "r", encoding="utf-8"))]

        # Collect one unique summary per document_id, sorted by numeric suffix if present
        summaries = [
            (ch.document_id, ch.metadata.get("summary", ""))
            for ch in doc_chunks if ch.metadata.get("summary", "")
        ]
        unique_summaries = {}
        for doc_id, summary in summaries:
            if doc_id not in unique_summaries:
                unique_summaries[doc_id] = summary

        summaries_sorted = sorted(
            unique_summaries.items(),
            key=lambda x: int(str(x[0]).split("_")[-1]) if str(x[0]).split("_")[-1].isdigit() else float("inf")
        )
        summaries = [s for _, s in summaries_sorted]

        if not sample_ratio:
            sample_ratio = 0.35
        k = int(len(doc_chunks) * sample_ratio)
        sampled_chunks = random.sample(doc_chunks, k=k)

        # Save insights into memory for later retrieval
        sampled_chunks = self.processor.extract_insights(sampled_chunks)
        for chunk in tqdm(sampled_chunks, desc="Saving insights", total=len(sampled_chunks)):
            insights = chunk.metadata.get("insights", [])
            for item in insights:
                self.reflector.insight_memory.add(text=item, metadata={})
        if verbose:
            logger.info("Insight saving completed.")

        params = dict()
        params["documents"] = sampled_chunks
        params["schema"] = schema
        params["background"] = background
        params["abbreviations"] = abbreviations
        params["summaries"] = summaries

        result = self.graph_probing_agent.run(params)
        return result["schema"], result["settings"]

    # ═════════════════════════════════════════════════════════════════════
    #  3) Entity / Relation extraction
    # ═════════════════════════════════════════════════════════════════════
    def _get_tmp_dir(self):
        base = self.config.storage.knowledge_graph_path
        tmp_dir = os.path.join(base, "extraction_tmp")
        os.makedirs(tmp_dir, exist_ok=True)
        return tmp_dir

    def _save_chunk_result(self, chunk_id: str, payload: dict):
        """保存单个 chunk 抽取结果，路径：extraction_tmp/<chunk_id>.json"""
        tmp_dir = self._get_tmp_dir()
        path = os.path.join(tmp_dir, f"{chunk_id}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def _load_existing_chunk_ids(self):
        """读取 extraction_tmp 目录下已有的 chunk 文件，返回 set(chunk_id)。"""
        tmp_dir = self._get_tmp_dir()
        if not os.path.exists(tmp_dir):
            return set()
        files = os.listdir(tmp_dir)
        done_ids = set()
        for name in files:
            if name.endswith(".json"):
                done_ids.add(name[:-5])  # 去掉 .json
        return done_ids

    def _load_all_tmp_results(self):
        """加载全部 extraction_tmp/<id>.json，返回 list of payload。"""
        tmp_dir = self._get_tmp_dir()
        if not os.path.exists(tmp_dir):
            return []
        results = []
        for name in os.listdir(tmp_dir):
            if not name.endswith(".json"):
                continue
            path = os.path.join(tmp_dir, name)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    results.append(json.load(f))
            except Exception as e:
                logger.warning(f"Failed to load tmp result {path}: {e}")
        return results



    def extract_entity_and_relation(self, verbose: bool = True):
        return asyncio.run(self.extract_entity_and_relation_async(verbose=verbose))

    
    async def extract_entity_and_relation_async(self, verbose: bool = True):
        """
        多轮重试 + 并发 + 真正断点续跑（集中重试版）：

        - 每个 chunk 的抽取结果独立保存为:  knowledge_graph_path/extraction_tmp/<chunk_id>.json
        - 再次运行时，会跳过已有 tmp 文件的 chunk（断点续跑）。
        - 失败的 chunk 不在本地 while 重试，而是交给 run_async_with_retries 做「按轮集中重试」。
        - 超时 / CancelledError 都视为该轮失败，不中断全局，只在最后几轮仍失败的 chunk 上打标记。
        """
        base = self.config.storage.knowledge_graph_path
        desc_chunks: List[TextChunk] = [
            TextChunk(**o)
            for o in json.load(open(os.path.join(base, "all_document_chunks.json"), "r", encoding="utf-8"))
        ]

        # ---- 1) 基于 tmp 文件做断点续跑：已有文件的 id 视为已完成 ----
        done_ids = self._load_existing_chunk_ids()
        if verbose:
            logger.info(f"Found {len(done_ids)} completed chunks in extraction_tmp/.")

        pending_chunks: List[TextChunk] = [ch for ch in desc_chunks if ch.id not in done_ids]
        if not pending_chunks:
            if verbose:
                logger.info("All chunks already processed. Skipping extraction.")
            # 合并一次结果，确保 extraction_results.json 存在
            all_results = self._load_all_tmp_results()
            out_path = os.path.join(base, "extraction_results.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(all_results, f, ensure_ascii=False, indent=2)
            return

        if verbose:
            logger.info(f"Pending chunks to process: {len(pending_chunks)}")

        from core.utils.function_manager import run_async_with_retries

        # 为了方便最后给 still_failed 写 stub，做个快速索引
        chunk_by_id: Dict[str, TextChunk] = {ch.id: ch for ch in desc_chunks}

        async def _work(ch: TextChunk) -> Dict[str, Any]:
            """
            对单个 chunk 做一次尝试（不在内部 while 重试）。
            返回结构：
                {
                    "ok": bool,
                    "error_kind": str,   # "ok"/"empty"/"timeout"/"cancelled"/"downstream"/"other"
                    "error": str,
                    "payload": {...}     # chunk 结果（即将写入 tmp 的内容）
                }
            """
            # 默认 payload（失败兜底）
            def _empty_payload(kind: str, msg: str) -> Dict[str, Any]:
                return {
                    "ok": False,
                    "error_kind": kind,
                    "error": msg,
                    "payload": {
                        "chunk_id": ch.id,
                        "chunk_metadata": ch.metadata,
                        "entities": [],
                        "relations": [],
                    },
                }

            try:
                if not (ch.content or "").strip():
                    # 无内容：视为「非重试型失败」，后续不重试
                    return _empty_payload("empty", "no content")

                res = await self.information_extraction_agent.arun(
                    ch.content,
                    timeout=self.config.agent.async_timeout,
                    max_attempts=self.config.agent.async_max_attempts,
                    backoff_seconds=self.config.agent.async_backoff_seconds,
                )

                # 下游主动返回 error
                if isinstance(res, dict) and res.get("error"):
                    msg = str(res["error"])
                    # 这里不区分 retryable / non-retryable，统一交给 run_async_with_retries 做多轮重试
                    return _empty_payload("downstream", msg)

                if isinstance(res, dict):
                    res = dict(res)
                else:
                    res = {}

                res.update(chunk_id=ch.id, chunk_metadata=ch.metadata)
                if "entities" not in res:
                    res["entities"] = []
                if "relations" not in res:
                    res["relations"] = []

                payload = res

                # ⭐ 成功时，直接写 tmp，避免后面异常丢失结果
                self._save_chunk_result(ch.id, payload)

                return {
                    "ok": True,
                    "error_kind": "ok",
                    "error": "",
                    "payload": payload,
                }

            # Python 3.11+ 下 CancelledError 是 BaseException，要单独抓
            except asyncio.CancelledError as e:
                # 把 CancelledError 当作一次「可重试失败」，不要让它向外冒
                return _empty_payload("cancelled", f"cancelled: {e}")

            except asyncio.TimeoutError:
                # 超时：这轮失败，之后由 run_async_with_retries 在下一轮集中重试
                return _empty_payload("timeout", "timeout")

            except Exception as e:
                return _empty_payload("other", f"{type(e).__name__}: {e}")

        def _key_fn(ch: TextChunk) -> str:
            return ch.id

        def _is_success(res: Dict[str, Any]) -> bool:
            """
            控制哪些结果视为「完成，不再重试」：
            - ok == True → 完成
            - error_kind == "empty" → 视为完成（无内容没必要重试）
            其它（timeout/cancelled/downstream/other）都视为失败，会在后续轮次重试。
            """
            if not isinstance(res, dict):
                return False
            if res.get("ok"):
                return True
            if res.get("error_kind") == "empty":
                return True
            return False

        max_rounds = getattr(self.config.agent, "async_max_attempts", 3)
        concurrency = getattr(self, "max_workers", 16)
        backoff = getattr(self.config.agent, "async_backoff_seconds", 1.0)
        timeout = getattr(self.config.agent, "async_timeout", 600.0)

        # ---- 2) 交给 run_async_with_retries 做多轮并发调度 ----
        final_map, still_failed_ids = await run_async_with_retries(
            items=pending_chunks,
            work_fn=_work,
            key_fn=_key_fn,
            is_success_fn=_is_success,
            max_rounds=max_rounds,
            concurrency=concurrency,
            desc_label="Entity/Relation extraction",
            retry_backoff_seconds=backoff,
            per_task_timeout=timeout,
            decay_per_round=0.7,
            use_exponential_backoff=True,
        )

        # ---- 3) 对仍然失败的 chunk 写入 stub 结果（空实体 + 错误信息），保证每个 chunk 都有文件 ----
        if still_failed_ids:
            for cid in still_failed_ids:
                ch = chunk_by_id.get(cid)
                md = ch.metadata if ch is not None else {}
                payload = {
                    "chunk_id": cid,
                    "chunk_metadata": md,
                    "entities": [],
                    "relations": [],
                    "error": f"extraction failed after {max_rounds} rounds",
                }
                self._save_chunk_result(cid, payload)

        # ---- 4) 合并所有 tmp 结果为 extraction_results.json ----
        all_results = self._load_all_tmp_results()
        out_path = os.path.join(base, "extraction_results.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)

        if verbose:
            logger.info(
                f"Entity & relation extraction finished. "
                f"Total chunks: {len(desc_chunks)}, results written: {len(all_results)}"
            )
            if still_failed_ids:
                logger.warning(
                    f"{len(still_failed_ids)} chunks still failed after {max_rounds} rounds. "
                    f"Examples: {list(still_failed_ids)[:10]} ..."
                )

    # ═════════════════════════════════════════════════════════════════════
    #  4) Attribute extraction & refinement
    # ═════════════════════════════════════════════════════════════════════
    def run_extraction_refinement(self, verbose=False):
        KW = ("闪回", "一组蒙太奇")
        base = self.config.storage.knowledge_graph_path
        extraction_results = json.load(open(os.path.join(base, "extraction_results.json"), "r", encoding="utf-8"))

        # ★ 新增：别名/编号归一化（提前去掉诸如 “周喆直[73]” 的编号后缀）
        extraction_results = self._pre_refine_alias_canonicalize(extraction_results)

        # 下面保持你原有的清洗与 refine 流程
        for doc in extraction_results:
            ents = doc.get("entities", [])
            removed_names = {e.get("name", "") for e in ents if any(k in e.get("name", "") for k in KW)}
            doc["entities"] = [e for e in ents if e.get("name", "") not in removed_names]

            rels = doc.get("relations", [])
            doc["relations"] = [
                r for r in rels
                if not any(k in r.get("subject", "") or k in r.get("object", "") for k in KW)
                and r.get("subject", "") not in removed_names
                and r.get("object", "") not in removed_names
            ]

        if verbose:
            logger.info("Refining entity types...")
        extraction_results = self.graph_preprocessor.refine_entity_types(extraction_results)
        # with open(os.path.join(base, "extraction_results_1.json"), "w", encoding="utf-8") as f:
        #     json.dump(extraction_results, f, ensure_ascii=False, indent=2)

        if verbose:
            logger.info("Refining entity scope...")
        extraction_results = self.graph_preprocessor.refine_entity_scope(extraction_results)

        if verbose:
            logger.info("Running entity disambiguation...")
        extraction_results = self.graph_preprocessor.run_entity_disambiguation(extraction_results)

        with open(os.path.join(base, "extraction_results_refined.json"), "w", encoding="utf-8") as f:
            json.dump(extraction_results, f, ensure_ascii=False, indent=2)



    def extract_entity_attributes(self, verbose: bool = True) -> Dict[str, Entity]:
        return asyncio.run(self.extract_entity_attributes_async(verbose=verbose))

    async def extract_entity_attributes_async(
        self,
        verbose: bool = True,
        degree_threshold: int = 2,
    ) -> Dict[str, Any]:
        """
        基于 entity_basic_info.json 的实体属性抽取（多轮重试 + 断点续跑版）

        只对 total_degree > degree_threshold 的实体调用 LLM 抽属性；
        其余实体直接写回，properties 设为空字典。
        """

        base = self.config.storage.knowledge_graph_path
        # 先重新跑一遍 merge_entities_info，确保 total_degree 已经写入 entity_basic_info.json
        with open(os.path.join(base, "extraction_results_refined.json"), "r", encoding="utf-8") as f:
            extraction_results = json.load(f)
        self.merge_entities_info(extraction_results)

        basic_path = os.path.join(base, "entity_basic_info.json")
        if not os.path.exists(basic_path):
            raise FileNotFoundError(
                f"entity_basic_info.json not found at {basic_path}. "
                f"Please make sure merge_entities_info() has been called before attribute extraction."
            )

        # 1) 读取 basic info（id -> Entity）
        ent_raw = json.load(open(basic_path, "r", encoding="utf-8"))
        entities_by_id: Dict[str, Entity] = {
            eid: Entity(**d) for eid, d in ent_raw.items()
        }

        # ---------- 断点续跑相关工具 ----------
        def _get_attr_tmp_dir() -> str:
            tmp_dir = os.path.join(base, "entity_attr_tmp")
            os.makedirs(tmp_dir, exist_ok=True)
            return tmp_dir

        def _load_existing_attr_ids() -> Set[str]:
            tmp_dir = _get_attr_tmp_dir()
            if not os.path.exists(tmp_dir):
                return set()
            ids = set()
            for name in os.listdir(tmp_dir):
                if name.endswith(".json"):
                    ids.add(name[:-5])
            return ids

        def _save_attr_result(entity_id: str, payload: Dict[str, Any]) -> None:
            tmp_dir = _get_attr_tmp_dir()
            path = os.path.join(tmp_dir, f"{entity_id}.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)

        # ---------- 2) 按 total_degree 分组：低度数跳过， 高度数跑 LLM ----------
        done_ids = _load_existing_attr_ids()
        all_ids = list(entities_by_id.keys())

        # 注意：有些旧数据可能没有 total_degree 字段，这里默认 0
        high_degree_ids: List[str] = []
        low_degree_ids: List[str] = []
        for eid, ent in entities_by_id.items():
            td = getattr(ent, "total_degree", 0)
            if td > degree_threshold or ent.type == "Event":
                high_degree_ids.append(eid)
            else:
                low_degree_ids.append(eid)

        # 对 total_degree <= threshold 且尚未有 tmp 的实体：直接写 stub，properties 设为空
        low_pending_ids = [eid for eid in low_degree_ids if eid not in done_ids]
        for eid in low_pending_ids:
            ent = deepcopy(entities_by_id[eid])
            # 明确将 properties 清空
            ent.properties = {}
            payload = {
                "entity_id": eid,
                "entity": ent.dict(),
                "error": f"skipped: total_degree={getattr(ent, 'total_degree', 0)} <= threshold={degree_threshold}",
            }
            _save_attr_result(eid, payload)

        # 这些低度数实体已经“完成”，加入 done 集合
        done_ids |= set(low_pending_ids)

        # 只对 high_degree_ids 里尚未处理的实体跑 LLM
        pending_ids = [eid for eid in high_degree_ids if eid not in done_ids]

        if verbose:
            logger.info(
                "Starting async attribute extraction with checkpointing (degree-filtered). "
                f"#entities(total): {len(all_ids)}, "
                f"high_degree(>{degree_threshold}): {len(high_degree_ids)}, "
                f"low_degree(<= {degree_threshold}): {len(low_degree_ids)}, "
                f"already_done(tmp exists): {len(done_ids)}, "
                f"pending_high_degree_for_LLM: {len(pending_ids)}"
            )

        # 如果没有任何高 degree pending 实体：直接合并 tmp -> entity_info.json
        if not pending_ids:
            if verbose:
                logger.info("No high-degree entities pending. Skipping LLM attribute extraction.")
            tmp_dir = _get_attr_tmp_dir()
            merged_entities: Dict[str, Dict[str, Any]] = {}

            for name in os.listdir(tmp_dir):
                if not name.endswith(".json"):
                    continue
                path = os.path.join(tmp_dir, name)
                try:
                    data = json.load(open(path, "r", encoding="utf-8"))
                except Exception as e:
                    logger.warning(f"Failed to load attr tmp {path}: {e}")
                    continue

                eid = data.get("entity_id")
                ent_data = data.get("entity")
                if eid and isinstance(ent_data, dict):
                    merged_entities[eid] = ent_data

            output_path = os.path.join(base, "entity_info.json")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(merged_entities, f, ensure_ascii=False, indent=2)

            return {
                "status": "success",
                "processed_entities": len(merged_entities),
                "output_path": output_path,
                "failed_entities": [],
            }

        # 为仍然失败的情况准备一个索引，方便最后写 stub
        def _get_orig_entity(eid: str) -> Entity:
            return entities_by_id[eid]

        async def _work(entity_id: str) -> Dict[str, Any]:
            """
            对单个实体做一次属性抽取尝试（不在内部 while 重试）。
            返回：
                {
                    "ok": bool,
                    "error_kind": str,   # "ok"/"empty"/"timeout"/"cancelled"/"downstream"/"other"
                    "error": str,
                    "payload": (entity_id, Entity or None)
                }
            """
            ent = entities_by_id[entity_id]

            def _empty_ret(kind: str, msg: str) -> Dict[str, Any]:
                return {
                    "ok": False,
                    "error_kind": kind,
                    "error": msg,
                    "payload": (entity_id, None),
                }

            # 描述为空：视为完成，不调 LLM，直接保持原实体
            txt = ent.description or ""
            if not txt.strip():
                payload_ent = deepcopy(ent)
                payload = {
                    "entity_id": entity_id,
                    "entity": payload_ent.dict(),
                    "error": "no description",
                }
                _save_attr_result(entity_id, payload)
                return {
                    "ok": True,
                    "error_kind": "empty",
                    "error": "no description",
                    "payload": (entity_id, payload_ent),
                }

            try:
                res = await self.attribute_extraction_agent.arun(
                    text=txt,
                    entity_name=ent.name,
                    entity_type=ent.type,
                    version=ent.version,
                    source_chunks=ent.source_chunks,
                    additional_chunks=ent.additional_chunks,
                    timeout=1200,
                    max_attempts=self.config.agent.async_max_attempts,
                    backoff_seconds=self.config.agent.async_backoff_seconds,
                )

                if isinstance(res, dict) and res.get("error"):
                    msg = str(res["error"])
                    return _empty_ret("downstream", msg)

                attrs = res.get("attributes", {}) if isinstance(res, dict) else {}
                if isinstance(attrs, str):
                    try:
                        attrs = json.loads(attrs)
                    except json.JSONDecodeError:
                        return _empty_ret("validation", "attr JSON decode failed")

                new_ent = deepcopy(ent)
                new_ent.properties = attrs or {}

                nd = (res.get("new_description", "") if isinstance(res, dict) else "")
                if nd:
                    new_ent.description = nd

                # 成功：写 tmp
                payload = {
                    "entity_id": entity_id,
                    "entity": new_ent.dict(),
                    "error": "",
                }
                _save_attr_result(entity_id, payload)

                return {
                    "ok": True,
                    "error_kind": "ok",
                    "error": "",
                    "payload": (entity_id, new_ent),
                }

            except asyncio.CancelledError as e:
                # 把 CancelledError 当作一次可重试失败，不向外冒
                return _empty_ret("cancelled", f"cancelled: {e}")

            except asyncio.TimeoutError:
                return _empty_ret("timeout", "timeout")

            except Exception as e:
                return _empty_ret("other", f"{type(e).__name__}: {e}")

        def _key_fn(entity_id: str) -> str:
            return entity_id

        def _is_success(res: Dict[str, Any]) -> bool:
            """
            认为“完成、不再重试”的条件：
            - ok == True → 完成
            - error_kind == "empty" → 描述为空，不必重试
            """
            if not isinstance(res, dict):
                return False
            if res.get("ok"):
                return True
            if res.get("error_kind") == "empty":
                return True
            return False

        max_rounds = getattr(self.config.agent, "async_max_attempts", 3)
        concurrency = getattr(self, "max_workers", 16)
        backoff = getattr(self.config.agent, "async_backoff_seconds", 1.0)
        timeout = getattr(self.config.agent, "async_timeout", 600.0)

        final_map, still_failed = await run_async_with_retries(
            items=pending_ids,
            work_fn=_work,
            key_fn=_key_fn,
            is_success_fn=_is_success,
            max_rounds=max_rounds,
            concurrency=concurrency,
            desc_label="Attribute extraction",
            retry_backoff_seconds=backoff,
            per_task_timeout=timeout,
            decay_per_round=0.7,
            use_exponential_backoff=True,
        )

        # ---------- 对仍失败的实体写 stub 结果 ----------
        failed_entities = sorted(list(still_failed))
        if failed_entities:
            for eid in failed_entities:
                orig_ent = _get_orig_entity(eid)
                payload = {
                    "entity_id": eid,
                    "entity": orig_ent.dict(),
                    "error": f"attribute extraction failed after {max_rounds} rounds",
                }
                _save_attr_result(eid, payload)

        # ---------- 合并 tmp -> entity_info.json ----------
        tmp_dir = _get_attr_tmp_dir()
        merged_entities: Dict[str, Dict[str, Any]] = {}

        for name in os.listdir(tmp_dir):
            if not name.endswith(".json"):
                continue
            path = os.path.join(tmp_dir, name)
            try:
                data = json.load(open(path, "r", encoding="utf-8"))
            except Exception as e:
                logger.warning(f"Failed to load attr tmp {path}: {e}")
                continue

            eid = data.get("entity_id")
            ent_data = data.get("entity")
            if eid and isinstance(ent_data, dict):
                merged_entities[eid] = ent_data

        output_path = os.path.join(base, "entity_info.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(merged_entities, f, ensure_ascii=False, indent=2)

        if verbose:
            logger.info(
                f"Attribute extraction finished. "
                f"Total entities: {len(all_ids)}, "
                f"high_degree(>{degree_threshold}) with LLM: {len(high_degree_ids)}, "
                f"written to entity_info.json: {len(merged_entities)}"
            )
            if failed_entities:
                logger.warning(
                    f"{len(failed_entities)} high-degree entities still failed after {max_rounds} rounds. "
                    f"Examples: {failed_entities[:10]} ..."
                )

        return {
            "status": "success",
            "processed_entities": len(merged_entities),
            "output_path": output_path,
            "failed_entities": failed_entities,
        }

    # ═════════════════════════════════════════════════════════════════════
    #  5) Build and store the graph
    # ═════════════════════════════════════════════════════════════════════
    def build_graph_from_results(self, verbose: bool = True) -> KnowledgeGraph:
        """
        从 refined 抽取结果和实体信息构建内存 KnowledgeGraph，
        然后落到存储，并在 Neo4j 上做补充与中心性计算。

        关键点：
        - 使用 entity_info.json 中的实体（id 已经是 stable id）
        - 用 (name/alias + version) → id 的映射解析关系端点
        - 场景/章节节点来自 chunk_metadata，通过 contains 关系连接到实体
        """
        if verbose:
            logger.info("Loading refined extraction results and entity info...")

        base = self.config.storage.knowledge_graph_path
        results_path = os.path.join(base, "extraction_results_refined.json")
        ent_info_path = os.path.join(base, "entity_info.json")
        sec_coll_path = os.path.join(base, "section_entities_collection.pkl")

        # chunk 级 refined 结果
        results = json.load(open(results_path, "r", encoding="utf-8"))

        # id -> Entity（注意 entity_info.json 的 key 当前是 id，我们用 values 更稳）
        ent_raw = json.load(open(ent_info_path, "r", encoding="utf-8"))
        entities_by_id: Dict[str, Entity] = {
            d["id"]: Entity(**d) for d in ent_raw.values()
        }

        # section_entities_collection: {section_label -> List[Entity]}
        with open(sec_coll_path, "rb") as f:
            self.section_entities_collection = pickle.load(f)

        # name+version -> id（含别名），用于从关系中的“名字”解析到实体 id
        namever2id: Dict[str, str] = {}
        for e in entities_by_id.values():
            ver = e.version or "default"
            key = f"{e.name}||{ver}"
            namever2id[key] = e.id
            for al in e.aliases:
                namever2id.setdefault(f"{al}||{ver}", e.id)

        # 先把实体装入内存图
        for e in entities_by_id.values():
            self.kg.add_entity(e)

        if verbose:
            logger.info("Building knowledge graph (sections, contains, relations)...")

        self.section_names: List[str] = []

        for res in results:
            md = res.get("chunk_metadata", {}) or {}
            chunk_id = res.get("chunk_id", "")
            version = md.get("version", "default")

            # --------- 场景/章节实体 ---------
            secs = self._create_section_entities(md, chunk_id)
            for se in secs:
                if se.name not in self.section_names and se.id not in self.kg.entities:
                    self.kg.add_entity(se)
                    self.section_names.append(se.name)
                else:
                    # 合并章节的 source_chunks（避免重复）
                    exist = self.kg.entities.get(se.id)
                    if exist:
                        merged = list(dict.fromkeys(list(exist.source_chunks) + list(se.source_chunks)))
                        exist.source_chunks = merged

            # --------- contains 关系（section -> 实体）---------
            label = md.get("doc_title", md.get("subtitle", md.get("title", "")))
            inner_entities = self.section_entities_collection.get(label, [])
            for se in secs:
                # 当前版本的 _link_section_to_entities 已经不依赖 inner_entities，
                # 但是参数保留不动，以兼容老签名。
                self._link_section_to_entities(se, inner_entities, chunk_id)

            # --------- 普通关系（实体间） ---------
            for rdata in res.get("relations", []) or []:
                rel = self._create_relation_from_data(
                    rdata,
                    chunk_id,
                    entities_by_id,   # 目前签名里没用到，可以后面再删参数
                    namever2id,
                    version=version,
                )
                if rel:
                    self.kg.add_relation(rel)

        # --------- 落库与图增强 ---------
        if verbose:
            logger.info("Persisting graph to databases...")
        self._store_knowledge_graph(verbose)

        if verbose:
            logger.info("Enriching event nodes and computing graph metrics...")
        try:
            self.neo4j_utils.enrich_event_nodes_with_context()
            self.neo4j_utils.compute_centrality(exclude_rel_types=[self.meta["contains_pred"]])
        except Exception as e:
            if verbose:
                logger.warning(f"Neo4j enrichment/metrics step encountered an issue: {e}")

        if verbose:
            st = self.kg.stats()
            graph_stats = self.graph_store.get_stats()
            logger.info("Knowledge graph construction completed.")
            logger.info(f"  - Entities: {graph_stats.get('entities')}")
            logger.info(f"  - Relations: {graph_stats.get('relations')}")
            logger.info(f"  - Documents: {st.get('documents')}")
            logger.info(f"  - Chunks: {st.get('chunks')}")

        return self.kg

    # ═════════════════════════════════════════════════════════════════════
    #  Internal utilities
    # ═════════════════════════════════════════════════════════════════════
    def _ekey(self, name: str, version: str = None, entity_type: str = None) -> str:
        """
        生成实体在合并映射中的唯一键。
        - name: 实体名称
        - version: 文档版本（缺省视为 "default"）
        """
        output_str = f"{name}"
        if version:
            output_str += f"||{version}"
        if entity_type:
            output_str += f"||{entity_type}"

        return output_str

    
    def merge_entities_info(self, extraction_results):
        """
        Merge/deduplicate entities across chunks, **within the same version**.

        - 用 (name, version, primary_type) 作为内部合并 key
        - Event 不做跨 chunk 合并（每个事件节点独立）
        - 对 scope=local 和 Action/Emotion/Goal 这类“局部/动作类”：
            * 同一 version 内，如已存在同名同主类型实体且出现在不同 section，
            则自动重命名（name_1, name_2, ...），避免误合并
        - 合并完成后：
            * 为 Part_2 实体回填 additional_chunks = 同名同主类型 Part_1 实体的 source_chunks
            * 持久化 self.section_entities_collection 到 section_entities_collection.pkl
            * 基于 relations 构建有向图，写回 total_degree（入度+出度）

        返回：
            Dict[Tuple[name, version, primary_type], Entity]
        """

        # --- 主类型归一：list/空 → 字符串（Event 优先，其次首个，否则 Concept） ---
        def _primary_type_local(t) -> str:
            if isinstance(t, list):
                return "Event" if "Event" in t else (t[0] if t else "Concept")
            return t or "Concept"

        # key: (name, version, primary_type) -> Entity
        entity_map: Dict[tuple, Entity] = {}

        # chunk_id -> section(doc_title) 映射，用于“局部/动作类跨 section 重命名”
        self.chunk2section_map = {
            result["chunk_id"]: (result.get("chunk_metadata", {}) or {}).get("doc_title", "")
            for result in extraction_results
        }

        # section_label -> List[Entity]，用于后续 contains 关系构建
        self.section_entities_collection = {}

        base = self.config.storage.knowledge_graph_path

        # ------------------------- 第一阶段：按 chunk 汇总 & 合并实体 -------------------------
        for result in extraction_results:
            md = result.get("chunk_metadata", {}) or {}
            chunk_id = result.get("chunk_id", "")
            # section label：优先 doc_title，其次 subtitle/title
            label = md.get("doc_title", md.get("subtitle", md.get("title", "")))
            if label not in self.section_entities_collection:
                self.section_entities_collection[label] = []

            version = md.get("version", "default")

            # 记录本 chunk 内「旧名 → 新名」的映射，用于同步更新 relations
            rename_map: Dict[str, str] = {}

            for ent_data in result.get("entities", []) or []:
                # 原始类型 & 主类型
                t_raw = ent_data.get("type", "Concept")
                t_primary = _primary_type_local(t_raw)

                name = ent_data["name"]
                key = (name, version, t_primary)

                # ---------- 局部/动作类实体：跨 section 重名时自动改名 ----------
                is_event = (t_primary == "Event")
                is_action_like = False
                if not is_event:
                    if isinstance(t_raw, str):
                        is_action_like = t_raw in ["Action", "Emotion", "Goal"]
                    elif isinstance(t_raw, list):
                        is_action_like = any(x in ["Action", "Emotion", "Goal"] for x in t_raw)

                if (ent_data.get("scope", "").lower() == "local" or is_action_like) and key in entity_map:
                    # 已存在一个同 name/version/primary_type 的局部/动作类实体
                    existing_entity = entity_map[key]
                    existing_chunk_id = existing_entity.source_chunks[0] if existing_entity.source_chunks else ""
                    existing_section_name = self.chunk2section_map.get(existing_chunk_id, "")
                    current_section_name = md.get("doc_title", "")

                    # 出现在不同 section → 自动重命名
                    if current_section_name and current_section_name != existing_section_name:
                        base_name = name
                        suffix = 1
                        new_name = f"{base_name}_{suffix}"
                        while (new_name, version, t_primary) in entity_map:
                            suffix += 1
                            new_name = f"{base_name}_{suffix}"

                        # 1) 维护别名：旧名作为 alias 保留下来
                        aliases = ent_data.get("aliases") or []
                        if base_name not in aliases:
                            aliases.append(base_name)
                        ent_data["aliases"] = aliases

                        # 2) 写回新名字
                        ent_data["name"] = new_name
                        name = new_name
                        key = (name, version, t_primary)

                        # 3) 记录本 chunk 内的重命名映射（用于后面同步 relations）
                        rename_map[base_name] = new_name

                # ---------- 创建实体对象 ----------
                ent_obj = self._create_entity_from_data(ent_data, chunk_id, version)

                # ---------- 仅对非 Event 类型尝试跨 chunk 合并 ----------
                existing = None
                if t_primary != "Event":
                    # 1) 先看是否有同 key（name, version, primary_type）的实体
                    if key in entity_map:
                        existing = entity_map[key]
                    else:
                        # 2) 再用 alias 在同 version 内做一次模糊合并
                        for (n_k, ver_k, t_k), candidate in entity_map.items():
                            if ver_k != version:
                                continue  # 版本必须一致
                            # Event 不参与此类合并
                            if _primary_type_local(candidate.type) == "Event":
                                continue
                            # name/alias 交集判断
                            if ent_obj.name in candidate.aliases or candidate.name in ent_obj.aliases:
                                existing = candidate
                                break
                            if ent_obj.aliases and candidate.aliases:
                                if any(a in candidate.aliases for a in ent_obj.aliases):
                                    existing = candidate
                                    break

                # ---------- 合并或登记为新实体 ----------
                if existing:
                    self._merge_entities(existing, ent_obj)
                    final_ent = existing
                    # 注意：key 依然使用原始 (name, version, primary_type)。
                    # 如果 existing 是通过 alias 匹配到的，它对应的 key 已经在 entity_map 里了，
                    # 不需要再写一次。
                else:
                    entity_map[key] = ent_obj
                    final_ent = ent_obj

                # 记录到对应 section（注意：同一个实体对象可以出现在多个 section 的列表中）
                self.section_entities_collection[label].append(final_ent)

            # --- 本 chunk 的实体全部处理完之后：用 rename_map 同步更新 relations ---
            if rename_map:
                for rel in result.get("relations", []) or []:
                    s = rel.get("subject")
                    o = rel.get("object")
                    if s in rename_map:
                        rel["subject"] = rename_map[s]
                    if o in rename_map:
                        rel["object"] = rename_map[o]

        # ---------------------- 第二阶段：Part_2 回填 additional_chunks ----------------------
        # 建立 (name, primary_type, version) -> Entity 索引（只是换个字段顺序，方便阅读）
        idx = {}
        for (name, version, primary_type), ent in entity_map.items():
            idx[(name, primary_type, version)] = ent

        # 先清空 additional_chunks
        for ent in entity_map.values():
            ent.additional_chunks = []

        # 对 Part_2 的实体：找到 Part_1 同名同主类型实体，复制其 source_chunks
        for (name, version, primary_type), ent in entity_map.items():
            if version == "Part_2":
                p1_key = (name, primary_type, "Part_1")
                if p1_key in idx:
                    ent.additional_chunks = list(idx[p1_key].source_chunks)
                else:
                    ent.additional_chunks = []

        # ---------------------- 第三阶段：基于 relations 计算 total_degree ----------------------
        # 先构建 name+version -> {entity_id} 的索引，用于把 relation 的 subject/object 映射到实体
        name_ver2ids: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
        for (name, version, _primary_type), ent in entity_map.items():
            # 主名
            name_ver2ids[(name, version)].add(ent.id)
            # 所有 alias 也映射到同一个实体（兜底）
            for alias in getattr(ent, "aliases", []) or []:
                name_ver2ids[(alias, version)].add(ent.id)

        # 对所有实体初始化 total_degree = 0（确保无边的节点也有字段）
        for ent in entity_map.values():
            # 假定 Entity 允许动态加字段；如果你在 Entity 里已经声明了 total_degree，这里就是正常赋值
            ent.total_degree = 0

        # 使用 entity id 构图
        G = nx.DiGraph()

        for result in extraction_results:
            md = result.get("chunk_metadata", {}) or {}
            version = md.get("version", "default")

            for rel in result.get("relations", []) or []:
                s_name = rel.get("subject")
                o_name = rel.get("object")
                if not s_name or not o_name:
                    continue

                # 在同一个 version 下按 name 找实体（支持 alias）
                s_ids = name_ver2ids.get((s_name, version)) or set()
                o_ids = name_ver2ids.get((o_name, version)) or set()
                if not s_ids or not o_ids:
                    continue

                # 多个实体同名时，简单地对所有组合连边
                for sid in s_ids:
                    for oid in o_ids:
                        G.add_edge(sid, oid)

        # 把图中的度数写回到实体（simple DiGraph：不区分多重边）
        if G.number_of_nodes() > 0:
            # in_degree/out_degree 默认权重为 1，以边计数
            in_deg_dict = dict(G.in_degree())
            out_deg_dict = dict(G.out_degree())
            for ent in entity_map.values():
                d_in = in_deg_dict.get(ent.id, 0)
                d_out = out_deg_dict.get(ent.id, 0)
                ent.total_degree = int(d_in + d_out)

        # ---------------------- 第四阶段：持久化 section_entities_collection ----------------------
        output_path = os.path.join(base, "section_entities_collection.pkl")
        with open(output_path, "wb") as f:
            pickle.dump(self.section_entities_collection, f)

        # 2) 对外统一用 id 作为 key
        id_entity_map: Dict[str, Entity] = {}
        for ent in entity_map.values():
            # 后面如果有需要，可以在这里检查 id 冲突（同 id 不同内容），目前假定 _create_entity_from_data 已保证稳定唯一
            id_entity_map[ent.id] = ent

        # entity_basic_info.json 也用 id 作为 key（这里会带上 total_degree 字段）
        basic_out_path = os.path.join(base, "entity_basic_info.json")
        with open(basic_out_path, "w", encoding="utf-8") as f:
            json.dump({eid: e.dict() for eid, e in id_entity_map.items()}, f, ensure_ascii=False, indent=2)

        # return entity_map



    def _find_existing_entity(self, entity: Entity, entity_map: Dict[str, Entity], version: str) -> Optional[Entity]:
        """Find an existing entity to merge with, **only within the same version** (non-Event)."""
        if (entity.type == "Event") or (isinstance(entity.type, list) and "Event" in entity.type):
            return None
        key = self._ekey(entity.name, version, entity.type)
        if key in entity_map:
            return entity_map[key]
        # alias 命中也需在同一 version 的键空间查找
        for ekey, existing_entity in entity_map.items():
            # 仅同版本比较
            if not ekey.endswith(f"||{version}"):
                continue
            if entity.name in existing_entity.aliases:
                return existing_entity
            if any(alias in existing_entity.aliases for alias in entity.aliases):
                return existing_entity
        return None


    def _merge_types(self, a, b):
        """
        Union of types from a and b, normalized (Event first + de-dup, preserve order).
        """
        a_list = a if isinstance(a, list) else ([a] if a else [])
        b_list = b if isinstance(b, list) else ([b] if b else [])
        merged = []
        seen = set()
        for x in a_list + b_list:
            if x and x not in seen:
                seen.add(x)
                merged.append(x)
        if "Event" in seen:
            merged = ["Event"] + [x for x in merged if x != "Event"]
        return merged if len(merged) > 1 else (merged[0] if merged else "Concept")

    def _merge_entities(self, existing: Entity, new: Entity) -> None:
        """Merge incoming entity into an existing one."""
        for alias in new.aliases:
            if alias not in existing.aliases:
                existing.aliases.append(alias)
        existing.properties.update(new.properties)
        for chunk_id in new.source_chunks:
            if chunk_id not in existing.source_chunks:
                existing.source_chunks.append(chunk_id)
        if new.description:
            if not existing.description:
                existing.description = new.description
            elif new.description not in existing.description:
                existing.description = existing.description + "\n" + new.description

        existing.type = self._merge_types(existing.type, new.type)

    def _ensure_entity_exists(self, entity_id: str, entity_map: Dict[str, Entity]) -> Optional[Entity]:
        return entity_map.get(entity_id, None)

    def _canonicalize_person_name(self, name: str, ent_type=None) -> str:
        """
        名称基准化：
        - NFKC 全半角统一；去零宽字符；收敛空白
        - 若为人物类（Character/Person），移除任意“末尾括注”（如：王强（父亲）、刘洋[青年]）
        - 否则仅移除“纯数字编号括注”（如：周喆直[73]）
        """
        if not name:
            return name
        s = unicodedata.normalize("NFKC", name)
        s = _ZERO_WIDTH_RE.sub("", s).strip()

        is_person = False
        if isinstance(ent_type, str):
            is_person = ent_type.lower() in ("character", "person")
        elif isinstance(ent_type, list):
            is_person = any((str(t).lower() in ("character", "person")) for t in ent_type)

        if is_person:
            s = _BRACKET_ANY_SUFFIX_RE.sub("", s).strip()
        else:
            s = _BRACKET_NUM_SUFFIX_RE.sub("", s).strip()

        s = _MULTISPACE_RE.sub(" ", s)
        return s

    def _pre_refine_alias_canonicalize(self, extraction_results: List[Dict]) -> List[Dict]:
        out = []
        for res in extraction_results:
            ents = res.get("entities", []) or []
            rels = res.get("relations", []) or []

            name_map = {}
            canon_ents = []
            for e in ents:
                e = dict(e)
                etype = e.get("type")
                orig = e.get("name", "")
                base = self._canonicalize_person_name(orig, etype)

                if base and base != orig:
                    aliases = list(dict.fromkeys((e.get("aliases") or []) + [orig, base]))
                    e["aliases"] = aliases
                    e["name"] = base
                    name_map[orig] = base

                if e.get("aliases"):
                    new_aliases = []
                    for a in e["aliases"]:
                        ab = self._canonicalize_person_name(a, etype)
                        new_aliases.append(a)
                        if ab and ab != a:
                            new_aliases.append(ab)
                    e["aliases"] = list(dict.fromkeys([x for x in new_aliases if x]))

                canon_ents.append(e)

            # 合并同一 chunk 内同名同类实体（同之前给你的合并逻辑不变）
            merged_by_key = {}
            for e in canon_ents:
                t = e.get("type")
                if isinstance(t, list):
                    primary_t = "Event" if "Event" in t else (t[0] if t else "Concept")
                else:
                    primary_t = t or "Concept"

                key = (e.get("name", ""), primary_t)
                if key not in merged_by_key:
                    merged_by_key[key] = dict(e)
                else:
                    me = merged_by_key[key]
                    me["aliases"] = list(dict.fromkeys((me.get("aliases") or []) + (e.get("aliases") or [])))
                    da, db = (me.get("description") or "").strip(), (e.get("description") or "").strip()
                    if db and db not in da:
                        me["description"] = (da + ("\n" if da and db else "") + db).strip()
                    sa, sb = (me.get("scope") or "").lower(), (e.get("scope") or "").lower()
                    if sa != "global" and sb == "global":
                        me["scope"] = "global"
                    ta, tb = me.get("types"), e.get("types")
                    if ta or tb:
                        la = ta if isinstance(ta, list) else ([ta] if ta else [])
                        lb = tb if isinstance(tb, list) else ([tb] if tb else [])
                        merged_types, seen = [], set()
                        for x in la + lb:
                            if x and x not in seen:
                                seen.add(x); merged_types.append(x)
                        me["types"] = merged_types

            new_entities = list(merged_by_key.values())

            # 关系端点同步（对 subject/object 同样跑一遍规则；如果端点恰是人物类名字，前一步已在 name_map 中）
            new_rels = []
            for r in (rels or []):
                r = dict(r)
                subj = r.get("subject") or r.get("source") or r.get("head") or r.get("relation_subject")
                obj  = r.get("object")  or r.get("target") or r.get("tail") or r.get("relation_object")

                if subj:
                    r["subject"] = name_map.get(subj) or self._canonicalize_person_name(subj, ent_type="Character")
                if obj:
                    r["object"] = name_map.get(obj) or self._canonicalize_person_name(obj, ent_type="Character")

                if "predicate" not in r and "relation" in r:
                    r["predicate"] = r.get("relation")
                new_rels.append(r)

            res_new = dict(res)
            res_new["entities"]  = new_entities
            res_new["relations"] = new_rels
            out.append(res_new)

        return out

    
    # -------- Section / Contains --------
    def _create_section_entities(self, md: Dict[str, Any], chunk_id: str) -> List[Entity]:
        """
        Create section/scene entities.

        - 'title'/'subtitle' are read directly.
        - Entity.properties will contain mapped fields (e.g., scene_name) + other metadata fields.
        """
        raw_title = md.get("title", "").strip()
        raw_subtitle = md.get("subtitle", "").strip()
        order = md.get("order", None)

        if not raw_title:
            return []

        label = self.meta["section_label"]
        full_name = md.get("doc_title", f"{label}{raw_title}-{raw_subtitle}" if raw_subtitle else f"{label}{raw_title}")
        eid = f"{label.lower()}_{order}" if order is not None else f"{label.lower()}_{hash(full_name) % 1_000_000}"

        title_field = self.meta["title"]
        subtitle_field = self.meta["subtitle"]

        excluded = {"chunk_index", "chunk_type", "doc_title", "title", "subtitle", "total_description_chunks", "total_doc_chunks"}
        properties = {
            title_field: raw_title,
            subtitle_field: raw_subtitle,
        }

        if order is not None:
            properties["order"] = order

        for k, v in md.items():
            if k not in excluded:
                properties[k] = v

        self.section_chunk_ids[eid].add(chunk_id)
        agg_chunks = sorted(self.section_chunk_ids[eid])

        return [
            Entity(
                id=eid,
                name=full_name,
                type=label,
                scope="local",
                description=md.get("summary", ""),
                properties=properties,
                source_chunks=agg_chunks,
            )
        ]

    # def _link_section_to_entities(self, section: Entity, inners: List[Entity], chunk_id: str):
    #     pred = self.meta["contains_pred"]
    #     for tgt in inners:
    #         rid = f"rel_{hash(f'{section.id}_{pred}_{tgt.id}') % 1_000_000}"
    #         self.kg.add_relation(
    #             Relation(id=rid, subject_id=section.id, predicate=pred,
    #                      object_id=tgt.id, properties={}, source_chunks=[chunk_id])
    #         )

    def _link_section_to_entities(self, section: Entity, inners: List[Entity], chunk_id: str):
        """
        修复：确保所有“在该 section 出现过”的实体都与 section 建立 contains 关系。
        判定准则：
        - 取该 section 的 chunk 集合（优先 self.section_chunk_ids[section.id]，否则回退到 section.source_chunks）
        - 遍历图中所有实体（排除 section 自身 & 其他 section 实体）
        - 若实体的 source_chunks 与 section 的 chunk 集合有交集 → 建立 contains 边
        说明：
        - 忽略传入的 inners（可能不完整），以避免漏连
        - 使用确定性 rid，自动去重（若已存在同样的边，不会重复添加）
        """
        pred = self.meta["contains_pred"]

        # 该 section 对应的 chunk 集
        sec_chunk_ids = set(self.section_chunk_ids.get(section.id, set()))
        if not sec_chunk_ids:
            sec_chunk_ids = set(getattr(section, "source_chunks", []) or [])

        # 备用：基于 doc_title 的回退匹配（当 sec_chunk_ids 仍为空时）
        doc_title_key = ""
        if not sec_chunk_ids:
            # section.properties 在 _create_section_entities 时会包含原始 metadata（含 doc_title）
            props = getattr(section, "properties", {}) or {}
            doc_title_key = (props.get("doc_title") or "").strip()

        # 为了避免重复添加：构造一个已存在的三元组集合
        existing = set()
        # 若你的 KnowledgeGraph 实现是 self.kg.relations: Dict[id, Relation]
        for rel in getattr(self.kg, "relations", {}).values():
            if rel.subject_id == section.id and rel.predicate == pred:
                existing.add((rel.subject_id, rel.predicate, rel.object_id))

        # 遍历当前图中所有实体，筛选真正出现于该 section 的实体
        for ent in list(getattr(self.kg, "entities", {}).values()):
            # 跳过：section 自身 以及 其他 section 实体（按类型名等于 section_label 判断）
            if ent.id == section.id:
                continue
            ent_type = getattr(ent, "type", "")
            if (
                ent_type == self.meta["section_label"] or
                (isinstance(ent_type, list) and self.meta["section_label"] in ent_type)
            ):
                continue

            e_chunks = set(getattr(ent, "source_chunks", []) or [])
            if not e_chunks:
                continue

            # 核心命中逻辑：chunk 交集 或 doc_title 映射相等（作为兜底）
            hit = False
            if sec_chunk_ids and (e_chunks & sec_chunk_ids):
                hit = True
            elif doc_title_key:
                # 基于 chunk -> doc_title 的映射兜底（避免某些 section 未记录 chunk 集时漏连）
                for cid in e_chunks:
                    if self.chunk2section_map.get(cid, "") == doc_title_key:
                        hit = True
                        break

            if not hit:
                continue

            key = (section.id, pred, ent.id)
            if key in existing:
                # 已有同样的 contains 边，跳过
                continue

            rid = f"rel_{hash(f'{section.id}_{pred}_{ent.id}') % 1_000_000}"

            # 关系的 source_chunks：尽量标注真实的交集；若没有，则落回当前 chunk_id
            rel_chunks = sorted((e_chunks & sec_chunk_ids)) if sec_chunk_ids else ([chunk_id] if chunk_id else [])

            self.kg.add_relation(
                Relation(
                    id=rid,
                    subject_id=section.id,
                    predicate=pred,
                    object_id=ent.id,
                    properties={},
                    source_chunks=rel_chunks if rel_chunks else ([chunk_id] if chunk_id else []),
                )
            )
            existing.add(key)


    # -------- Entity / Relation creation --------
    @staticmethod
    def _create_entity_from_data(data: Dict, chunk_id: str, version: str = "default") -> Entity:
        """
        基于 (name, version, primary_type) 生成稳定 id：
        - primary_type: Event 优先，其次 type[0]，否则 Concept
        - id = "ent_" + md5(f"{name}||{version}||{primary_type}")[:12]
        """
        name = data["name"]

        # 先把原始 type 拿出来
        t_raw = data.get("type", "Concept")
        if isinstance(t_raw, list):
            primary_type = "Event" if "Event" in t_raw else (t_raw[0] if t_raw else "Concept")
        else:
            primary_type = t_raw or "Concept"

        # 为 id 构造稳定 key
        id_key = f"{name}||{version}||{primary_type}"
        ent_id = "ent_" + hashlib.md5(id_key.encode("utf-8")).hexdigest()[:12]

        return Entity(
            id=ent_id,
            name=name,
            type=_normalize_type(t_raw),
            scope=data.get("scope", "local"),
            description=data.get("description", ""),
            aliases=data.get("aliases", []),
            version=version,
            source_chunks=[chunk_id],
        )


    @staticmethod
    def _create_relation_from_data(
        d: Dict, chunk_id: str, entity_map: Dict[str, Entity], namever2id: Dict[str, str], version: str = "default"
    ) -> Optional[Relation]:
        subj = d.get("subject") or d.get("source") or d.get("head") or d.get("relation_subject")
        obj  = d.get("object")  or d.get("target") or d.get("tail") or d.get("relation_object")
        pred = d.get("predicate") or d.get("relation") or d.get("relation_type")
        if not subj or not obj or not pred:
            return None

        sid = namever2id.get(f"{subj}||{version}")
        oid = namever2id.get(f"{obj}||{version}")
        if not sid or not oid:
            return None

        rid = f"rel_{hash(f'{sid}_{pred}_{oid}_{version}') % 1_000_000}"

        return Relation(
            id=rid,
            subject_id=sid,
            predicate=pred,
            object_id=oid,
            version=version,
            properties={
                "description": d.get("description", ""),
                "relation_name": d.get("relation_name", ""),
            },
            source_chunks=[chunk_id],
        )


    def _store_vectordb(self, verbose: bool):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=200,
            chunk_overlap=50,
            separators=[
                "\n\n", "### ", "## ", "# ",
                "\n",
                "。", "！", "？", "；", "：", "、", "，",
                ". ", "? ", "! ",
                " ",
                ""
            ],
            keep_separator=True,
        )

        try:
            all_documents = list(self.kg.documents.values())
            self.document_vector_store.delete_collection()
            self.document_vector_store._initialize()
            self.document_vector_store.store_documents(all_documents)

            self.sentence_vector_store.delete_collection()
            self.sentence_vector_store._initialize()

            all_sentences = []
            for doc in tqdm(all_documents, desc="Persisting split sentences", total=len(all_documents)):
                content = doc.content
                if len(content) > 200:
                    sentences = splitter.split_text(content)
                else:
                    sentences = [content]
                for i, sentence in enumerate(sentences):
                    sentence = sentence.replace("\\n", "").strip()
                    all_sentences.append(
                        Document(id=f"{doc.id}-{i+1}", content=sentence, metadata=doc.metadata)
                    )

            self.sentence_vector_store.store_documents(all_sentences)

        except Exception as e:
            if verbose:
                logger.error(f"Vector store persistence failed: {e}")

    def _store_knowledge_graph(self, verbose: bool):
        try:
            self.graph_store.reset_knowledge_graph()
            self.graph_store.store_knowledge_graph(self.kg)
        except Exception as e:
            if verbose:
                logger.error(f"Graph store persistence failed: {e}")

    # ═════════════════════════════════════════════════════════════════════
    #  Embedding & Stats
    # ═════════════════════════════════════════════════════════════════════
    def prepare_graph_embeddings(self):
        """Compute and persist graph embeddings as configured."""
        self.neo4j_utils.load_embedding_model(self.config.graph_embedding)
        self.neo4j_utils.create_vector_index()
        self.neo4j_utils.process_all_embeddings()
        self.neo4j_utils.ensure_entity_superlabel()
        logger.info("Graph embeddings prepared successfully.")

    def get_stats(self) -> Dict[str, Any]:
        """Return a consolidated stats dictionary for current components."""
        return {
            "knowledge_graph": self.kg.stats(),
            "graph_store": self.graph_store.get_stats(),
            "document_vector_store": self.document_vector_store.get_stats(),
            "sentence_vector_store": self.sentence_vector_store.get_stats(),
        }

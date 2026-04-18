# core/builder/graph_builder.py
from __future__ import annotations

import json
import os
import shutil
import hashlib
import logging
import re
import pickle
import threading
from typing import Any, Dict, List, Optional, Tuple, Set
from collections import defaultdict

import networkx as nx
from langchain.text_splitter import RecursiveCharacterTextSplitter

from core.models.data import Entity, KnowledgeGraph, Relation, TextChunk, Document
from core.utils.format import DOC_TYPE_META
from core.utils.config import KAGConfig
from core.model_providers.openai_llm import OpenAILLM
from core.builder.document_processor import DocumentProcessor
from core.agent.knowledge_extraction_agent import InformationExtractionAgent
from core.agent.property_extraction_agent import PropertyExtractionAgent
from core.agent.interaction_extraction_agent import InteractionExtractionAgent
from core.builder.graph_refiner import GraphRefiner
from ..storage.graph_store import GraphStore
from ..storage.sql_store import SQLStore
from ..storage.vector_store import VectorStore
from ..utils.graph_query_utils import GraphQueryUtils
from core.utils.function_manager import run_concurrent_with_retries

# unified utils (moved out from this file)
from core.utils.general_utils import (
    load_json,
    dump_json,
    dump_pickle,
    load_pickle,
    safe_title,
    safe_str,
    is_nonempty_str,
    order_key,
    document_metadata_from_first_chunk,
    strip_part_suffix,
    word_len,
    compute_centrality,
    filter_nodes_by_centrality,
)

logger = logging.getLogger(__name__)


class KnowledgeGraphBuilder:
    """
    Lite Builder (non-async):
      1) prepare_chunks -> doc2chunks.json keyed by title (with collision handling)
      2) document-level entity/relation extraction (threaded concurrency, checkpoint, multi-round retries)
      3) refinement (GraphRefiner)
      4) build entity_basic_info.json and relation_basic_info.json
      5) postprocess relations, build graph, compute degrees
      6) extract properties from graph
      7) extract interactions to JSON
      8) save interaction JSON-list to SQL (optional, explicit step-2)
      9) save to the local graph store (optional)

    Notes:
    - Uses new YAML layout:
        global: prompt_dir, task_dir, schema_dir, doc_type
        knowledge_graph_builder: file_path, max_workers, max_retries, per_task_timeout
        llm / embedding / rerank / storage follow the new config.py
    - ENTITY_SCHEMA is single-source-of-truth loaded from entity_schema_path
    """

    def __init__(
        self,
        config: KAGConfig,
        *,
        doc_type: Optional[str] = None,
        use_memory: bool = True,
    ):
        self.config = config

        # doc_type and meta
        self.doc_type = doc_type or config.global_config.doc_type
        self.meta = DOC_TYPE_META[self.doc_type]

        # dirs from global
        self.prompt_dir = config.global_config.prompt_dir
        self.task_dir = config.global_config.task_dir
        self.schema_dir = config.global_config.schema_dir

        # builder configs
        self.max_workers = int(getattr(config.knowledge_graph_builder, "max_workers", 8))
        self.max_retries = int(getattr(config.knowledge_graph_builder, "max_retries", 2))
        self.per_task_timeout = float(getattr(config.knowledge_graph_builder, "per_task_timeout", 2400))

        # output base path from knowledge_graph_builder.file_path
        self.file_path = safe_str(getattr(config.knowledge_graph_builder, "file_path", "")) or "data/knowledge_graph"

        # schema/task paths
        self.entity_schema_path = os.path.join(self.schema_dir, "default_entity_schema.json")
        self.relation_schema_path = os.path.join(self.schema_dir, "default_relation_schema.json")
        self.interaction_schema_path = os.path.join(self.schema_dir, "default_interaction_schema.json")
        self.entity_extraction_task_path = os.path.join(self.task_dir, "entity_extraction_task.json")
        self.relation_extraction_task_path = os.path.join(self.task_dir, "relation_extraction_task.json")
        self.interaction_extraction_task_path = os.path.join(self.task_dir, "interaction_extraction_task.json")

        # validate paths
        if not os.path.exists(self.entity_schema_path):
            raise FileNotFoundError(f"Entity schema not found: {self.entity_schema_path}")
        if not os.path.exists(self.relation_schema_path):
            raise FileNotFoundError(f"Relation schema not found: {self.relation_schema_path}")
        if not os.path.exists(self.interaction_schema_path):
            raise FileNotFoundError(f"Interaction schema not found: {self.interaction_schema_path}")
        if not os.path.exists(self.entity_extraction_task_path):
            raise FileNotFoundError(f"Entity extraction task not found: {self.entity_extraction_task_path}")
        if not os.path.exists(self.relation_extraction_task_path):
            raise FileNotFoundError(f"Relation extraction task not found: {self.relation_extraction_task_path}")
        if not os.path.exists(self.interaction_extraction_task_path):
            raise FileNotFoundError(f"Interaction extraction task not found: {self.interaction_extraction_task_path}")

        # shared LLM for the whole pipeline
        self.llm = OpenAILLM(config)

        # document processor
        self.processor = DocumentProcessor(config, self.llm, self.doc_type)

        # storage helpers
        self.graph_store = GraphStore(self.config)
        self.graph_query_utils = GraphQueryUtils(self.graph_store, doc_type=self.doc_type)

        # load schemas/tasks and build derived runtime structures
        self.load_entity_schema_and_tasks()

        # load relation schema once
        with open(self.relation_schema_path, "r", encoding="utf-8") as f:
            self.RELATION_SCHEMA = json.load(f)
        with open(self.interaction_schema_path, "r", encoding="utf-8") as f:
            self.INTERACTION_SCHEMA = json.load(f)

        # initialize extraction memory store (if enabled)
        self._memory_store = None
        self._memory_distiller = None
        self._memory_distill_lock = threading.Lock()
        mem_cfg = getattr(self.config, "extraction_memory", None)
        if use_memory and mem_cfg is not None and getattr(mem_cfg, "enabled", True):
            try:
                from core.memory.extraction_store import ExtractionMemoryStore
                from core.memory.memory_distiller import MemoryDistiller

                raw_store_path = str(getattr(mem_cfg, "raw_store_path", "") or "data/memory/raw_memory")
                distilled_store_path = str(
                    getattr(mem_cfg, "distilled_store_path", "") or "data/memory/distilled_memory"
                )
                realtime_store_path = str(
                    getattr(mem_cfg, "realtime_store_path", "") or "data/memory/realtime_memory"
                )
                for p in [raw_store_path, distilled_store_path, realtime_store_path]:
                    os.makedirs(p, exist_ok=True)

                self._memory_store = ExtractionMemoryStore.from_config(self.config)
                logger.info("ExtractionMemoryStore initialized at %s", raw_store_path)
                self._memory_distiller = MemoryDistiller(
                    llm=self.llm,
                    memory_store=self._memory_store,
                    problem_solver=None,
                    store_path=distilled_store_path,
                    prompt_dir=self.prompt_dir,
                    enabled=bool(getattr(mem_cfg, "distill_enabled", True)),
                    cycle_every_n_docs=int(getattr(mem_cfg, "distill_every_n_docs", 20)),
                    min_new_entries=int(getattr(mem_cfg, "distill_min_new_entries", 100)),
                    max_source_entries=int(getattr(mem_cfg, "distill_max_source_entries", 200)),
                    max_new_memories=int(getattr(mem_cfg, "distill_max_new_memories", 30)),
                )
            except Exception as _e:
                logger.warning("ExtractionMemoryStore init failed, running without memory: %s", _e)

        # initialize extraction agent once
        self.extract_agent = InformationExtractionAgent(
            config=self.config,
            llm=self.llm,
            entity_schema=self.ENTITY_SCHEMA,
            relation_schema=self.RELATION_SCHEMA,
            memory_store=self._memory_store,
        )
        if self._memory_distiller is not None:
            try:
                self._memory_distiller.problem_solver = getattr(self.extract_agent, "problem_solver", None)
            except Exception:
                pass

        self.graph_refiner: Optional[GraphRefiner] = None
        self.interaction_agent = InteractionExtractionAgent(
            config=self.config,
            llm=self.llm,
            interaction_schema=self.INTERACTION_SCHEMA,
        )

    # -------------------------
    # Schema/task loading
    # -------------------------
    def load_entity_schema_and_tasks(self) -> None:
        """
        Load entity schema + entity extraction task config, and derive
        runtime structures.

        Single source of truth:
          self.ENTITY_SCHEMA is exactly what's loaded from entity_schema_path
        """

        with open(self.entity_schema_path, "r", encoding="utf-8") as f:
            entity_schema = json.load(f)

        with open(self.entity_extraction_task_path, "r", encoding="utf-8") as f:
            extraction_tasks = json.load(f)

        if not isinstance(entity_schema, list):
            raise ValueError("default_entity_schema.json must be a list of entity type definitions")
        if not isinstance(extraction_tasks, list):
            raise ValueError("entity_extraction_task.json must be a list")

        # category priority (used for stable type ordering)
        self.CATEGORY_PRIORITY: Dict[str, int] = {
            "induced": 0,
            "anchor": 1,
            "referential": 2,
            "general_semantic": 3,
        }

        type2category: Dict[str, str] = {}
        category2types: Dict[str, List[str]] = defaultdict(list)

        induced_types: Set[str] = set()
        general_semantic_types: List[str] = []

        for ent in entity_schema:
            if not isinstance(ent, dict):
                continue

            t = ent.get("type")
            cat = ent.get("category")

            if not isinstance(t, str) or not t.strip():
                continue
            if not isinstance(cat, str) or not cat.strip():
                continue

            t = t.strip()
            cat = cat.strip()

            type2category[t] = cat
            category2types[cat].append(t)

            if cat == "induced":
                induced_types.add(t)
            if cat == "general_semantic":
                general_semantic_types.append(t)

        if not general_semantic_types:
            raise ValueError("No general_semantic entity type found in entity schema")

        if len(general_semantic_types) > 1:
            raise ValueError(
                f"Multiple general_semantic types found: {general_semantic_types}. "
                f"Schema should define exactly one."
            )

        general_semantic_type = general_semantic_types[0]
        allowed_types = set(type2category.keys())

        # parse extraction task config for default scopes
        type2default_scope: Dict[str, str] = {}

        for task in extraction_tasks:
            if not isinstance(task, dict):
                continue

            for tdef in task.get("types", []) or []:
                if not isinstance(tdef, dict):
                    continue

                t = tdef.get("type")
                if not isinstance(t, str) or not t.strip():
                    continue
                t = t.strip()

                if "default_scope" in tdef:
                    ds = tdef.get("default_scope")
                    if isinstance(ds, str) and ds in ("global", "local"):
                        type2default_scope[t] = ds
                        continue

                scope_rules = tdef.get("scope_rules")
                if isinstance(scope_rules, dict) and len(scope_rules) == 1:
                    only_scope = next(iter(scope_rules.keys()))
                    if only_scope in ("global", "local"):
                        type2default_scope[t] = only_scope

        # build TYPE_PRIORITY by category priority only
        cat_prio = self.CATEGORY_PRIORITY

        def _type_rank(t: str) -> int:
            cat = type2category.get(t, "general_semantic")
            return int(cat_prio.get(cat, 9999))

        type_priority = tuple(sorted(allowed_types, key=_type_rank))

        # write back
        self.ENTITY_SCHEMA = entity_schema
        self.ENTITY_EXTRACTION_TASKS = extraction_tasks

        self.TYPE2CATEGORY = type2category
        self.CATEGORY2TYPES = dict(category2types)

        self.INDUCED_TYPES = induced_types
        self.GENERAL_SEMANTIC_TYPE = general_semantic_type

        self.TYPE2DEFAULT_SCOPE = type2default_scope
        self.ALLOWED_TYPES = allowed_types
        self.TYPE_PRIORITY = type_priority

    # -------------------------
    # Paths / tmp
    # -------------------------
    def _base_dir(self) -> str:
        base = safe_str(getattr(self, "file_path", "")) or "data/knowledge_graph"
        os.makedirs(base, exist_ok=True)
        return base

    def _interaction_base_dir(self) -> str:
        base = "data/narrative_interactions"
        os.makedirs(base, exist_ok=True)
        return base

    def _tmp_dir(self) -> str:
        tmp = os.path.join(self._base_dir(), "document_extraction_tmp")
        os.makedirs(tmp, exist_ok=True)
        return tmp

    def _tmp_path(self, key: str) -> str:
        h = hashlib.md5(key.encode("utf-8")).hexdigest()[:12]
        return os.path.join(self._tmp_dir(), f"document_{h}.json")

    def _tmp_exists(self, key: str) -> bool:
        return os.path.exists(self._tmp_path(key))

    def _save_tmp(self, key: str, payload: Dict[str, Any]) -> None:
        dump_json(self._tmp_path(key), payload)

    def _load_tmp(self, key: str) -> Optional[Dict[str, Any]]:
        p = self._tmp_path(key)
        if not os.path.exists(p):
            return None
        try:
            return load_json(p)
        except Exception as e:
            logger.warning(f"Failed to load tmp {p}: {e}")
            return None

    def clear_directory_keep_root(self, path: str) -> None:
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

    def maybe_distill_memory(
        self,
        *,
        docs_processed: int = 0,
        force: bool = False,
        reason: str = "",
        extra_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if self._memory_distiller is None:
            return {"ran": False, "trigger": "distiller_unavailable"}
        if not self._memory_distill_lock.acquire(blocking=False):
            return {"ran": False, "trigger": "distiller_busy"}
        try:
            return self._memory_distiller.maybe_run(
                docs_processed=docs_processed,
                force=force,
                reason=reason,
                extra_context=extra_context,
            )
        except Exception as e:
            logger.warning("Memory distillation failed: %s", e)
            return {"ran": False, "trigger": "distiller_exception", "error": str(e)}
        finally:
            self._memory_distill_lock.release()

    def clear_all_memories(self) -> None:
        """
        Clear raw memory, distilled memory artifacts, and memory dump snapshots.
        """
        mem_cfg = getattr(self.config, "extraction_memory", None)
        raw_store_path = (
            str(getattr(mem_cfg, "raw_store_path", "") or "")
            if mem_cfg is not None
            else ""
        ) or "data/memory/raw_memory"
        distilled_store_path = (
            str(getattr(mem_cfg, "distilled_store_path", "") or "")
            if mem_cfg is not None
            else ""
        ) or "data/memory/distilled_memory"
        realtime_store_path = (
            str(getattr(mem_cfg, "realtime_store_path", "") or "")
            if mem_cfg is not None
            else ""
        ) or "data/memory/realtime_memory"

        if self._memory_store is not None:
            try:
                self._memory_store.clear()
            except Exception as e:
                logger.warning("clear_all_memories: memory store clear failed: %s", e)

        if self._memory_distiller is not None:
            try:
                self._memory_distiller.clear_artifacts()
            except Exception as e:
                logger.warning("clear_all_memories: distiller artifact cleanup failed: %s", e)

        for fn in [
            "memory_dump_realtime.json",
            "raw_memories_realtime.jsonl",
            "distilled_memories_realtime.jsonl",
        ]:
            p = os.path.join(realtime_store_path, fn)
            if os.path.exists(p):
                try:
                    os.remove(p)
                except Exception as e:
                    logger.warning("clear_all_memories: failed to remove %s: %s", p, e)

        for fn in [
            "distill_state.json",
            "distilled_memories_latest.json",
            "distilled_memory_summaries.json",
        ]:
            p = os.path.join(distilled_store_path, fn)
            if os.path.exists(p):
                try:
                    os.remove(p)
                except Exception as e:
                    logger.warning("clear_all_memories: failed to remove %s: %s", p, e)

        logger.info(
            "All extraction memories cleared: raw=%s distilled=%s realtime=%s",
            raw_store_path,
            distilled_store_path,
            realtime_store_path,
        )

    def _store_vectordb_from_doc2chunks(
        self,
        *,
        doc2chunks: Dict[str, Dict[str, Any]],
        reset_collections: bool = True,
        sentence_chunk_size: int = 200,
        sentence_chunk_overlap: int = 50,
        verbose: bool = True,
    ) -> None:
        """
        Persist chunked raw texts into two vector stores:
        - document level: one vector per extraction document_id (must match source_documents IDs)
        - sentence level: finer segments with metadata pointing to parent document_id
        """
        if not isinstance(doc2chunks, dict) or not doc2chunks:
            if verbose:
                logger.warning("Skip vector persistence: doc2chunks is empty.")
            return

        document_vector_store = VectorStore(self.config, category="document")
        sentence_vector_store = VectorStore(self.config, category="sentence")

        if reset_collections:
            document_vector_store.delete_collection()
            document_vector_store._initialize()
            sentence_vector_store.delete_collection()
            sentence_vector_store._initialize()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=max(10, int(sentence_chunk_size)),
            chunk_overlap=max(0, int(sentence_chunk_overlap)),
            length_function=lambda t: word_len(t, lang="auto"),
            separators=[
                "\n\n", "### ", "## ", "# ",
                "\n",
                "。", "！", "？", "；", "：", "、", "，",
                ". ", "? ", "! ",
                " ",
                "",
            ],
            keep_separator=True,
        )

        document_items: List[Document] = []
        sentence_items: List[Document] = []

        for part_id, pack in doc2chunks.items():
            if not isinstance(part_id, str) or not part_id.strip():
                continue
            if not isinstance(pack, dict):
                continue
            chunks = pack.get("chunks") or []
            if not isinstance(chunks, list):
                chunks = []
            base_md = dict(pack.get("document_metadata") or {})
            base_md["document_id"] = part_id

            # Store vector "document" collection at chunk granularity:
            # - vector id = chunk_id
            # - metadata.document_id = logical document id (many chunks can share one document_id)
            for cidx, c in enumerate(chunks, start=1):
                if not isinstance(c, dict):
                    continue
                chunk_id = safe_str(c.get("id")).strip() or f"{part_id}_chunk_{cidx}"
                chunk_text = str(c.get("content", "") or "").strip()
                if not chunk_text:
                    continue

                chunk_md = dict(base_md)
                cmeta = c.get("metadata") or {}
                if isinstance(cmeta, dict):
                    for mk, mv in cmeta.items():
                        if mk not in chunk_md:
                            chunk_md[mk] = mv
                chunk_md["chunk_id"] = chunk_id
                chunk_md["vector_granularity"] = "document"
                chunk_md["parent_document_id"] = part_id

                document_items.append(Document(id=chunk_id, content=chunk_text, metadata=chunk_md))

                if word_len(chunk_text, lang="auto") > int(sentence_chunk_size):
                    segments = splitter.split_text(chunk_text)
                else:
                    segments = [chunk_text]

                for sidx, seg in enumerate(segments, start=1):
                    seg_txt = str(seg or "").replace("\\n", "\n").strip()
                    if not seg_txt:
                        continue
                    seg_md = dict(chunk_md)
                    seg_md["vector_granularity"] = "sentence"
                    seg_md["parent_document_id"] = chunk_id
                    seg_md["sentence_order"] = sidx
                    sentence_items.append(Document(id=f"{chunk_id}<->{sidx}", content=seg_txt, metadata=seg_md))

        document_vector_store.store_documents(document_items)
        sentence_vector_store.store_documents(sentence_items)

        if verbose:
            logger.info(
                "Vector stores persisted: document=%d sentence=%d (collections=document,sentence)",
                len(document_items),
                len(sentence_items),
            )

    # -------------------------
    # 1) prepare_chunks
    # -------------------------
    def prepare_chunks(
        self,
        json_file_path: str,
        *,
        verbose: bool = True,
        retries: int = 2,
        per_task_timeout: float = 600.0,
        use_semantic_split: bool = True,
        extract_summary: bool = True,
        extract_metadata: bool = True,
        summary_max_words: int = 200,
        reset_output_dir: bool = True,
        store_vector_chunks: Optional[bool] = None,
        sentence_chunk_size: Optional[int] = None,
        sentence_chunk_overlap: Optional[int] = None,
        reset_vector_collections: Optional[bool] = None,
    ) -> None:
        """
        Naming conventions (unified):
        - raw_doc_id: original source doc id from input.
        - doc_segment_id: pre-split segment id, e.g. "{raw_doc_id}_seg_1".
        - document_id: retrieval/extraction unit id, e.g. "{section_label}_{x}_part_1".
        - chunk_id: "{doc_segment_id}_chunk_{k}" (stored in chunk["id"]).

        Outputs:
        - all_document_chunks.json : flat list with stable chunk_id + metadata ids.
        - doc2chunks.json         : {document_id -> {"document_metadata": {...}, "chunks":[chunk_dicts...]}}
        - doc2chunks_index.json   : legacy index maps (kept for convenience)
        """
        base = self._base_dir()
        if reset_output_dir:
            self.clear_directory_keep_root(base)

        if verbose:
            logger.info(f"Chunking from: {json_file_path}")

        documents = self.processor.load_from_json(json_file_path)
        n_docs = len(documents)

        def _task(doc: Dict[str, Any]) -> Dict[str, List[Any]]:
            return self.processor.prepare_chunk(
                doc,
                use_semantic_split=use_semantic_split,
                extract_summary=extract_summary,
                extract_metadata=extract_metadata,
                summary_max_words=summary_max_words,
            )

        max_workers = int(getattr(self.processor, "max_workers", self.max_workers))
        rounds = max(1, int(retries))

        results_map, failed_indices = run_concurrent_with_retries(
            items=documents,
            task_fn=_task,
            per_task_timeout=per_task_timeout,
            max_retry_rounds=rounds,
            max_in_flight=max_workers,
            max_workers=max_workers,
            thread_name_prefix="chunk",
            desc_prefix="Chunking documents",
            treat_empty_as_failure=True,
            is_empty_fn=lambda r: (not r) or (not isinstance(r, dict)) or (not r.get("document_chunks")),
        )

        def _split_chunks_into_parts(chunks: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
            n = len(chunks)
            if n <= 2:
                return [chunks]

            parts: List[List[Dict[str, Any]]] = []
            start = 0
            if n % 2 == 1:
                parts.append(chunks[0:1])
                start = 1

            while start < n:
                parts.append(chunks[start : start + 2])
                start += 2

            return parts

        section_prefix = re.sub(
            r"[^A-Za-z0-9_]+",
            "_",
            safe_str(self.meta.get("section_label", "document")).strip().lower(),
        ).strip("_") or "document"
        section_id_key = f"{section_prefix}_id"

        def _build_base_key(md0: Dict[str, Any], doc_index: int, doc_segment_id: str) -> str:
            # Prefer doc-type specific section id key, e.g. scene_id/chapter_id/document_id.
            section_id_val = md0.get(section_id_key, None)
            # Backward compatibility for older screenplay metadata.
            if section_id_val is None and section_id_key != "scene_id":
                section_id_val = md0.get("scene_id", None)

            if section_id_val is not None:
                s = str(section_id_val).strip()
                if s:
                    return f"{section_prefix}_{s}"

            order = md0.get("order", None)
            if order is not None:
                try:
                    return f"{section_prefix}_{int(order)}"
                except Exception:
                    pass

            source_id = safe_str(doc_segment_id).strip()
            if source_id:
                return source_id

            return f"doc_{doc_index}"

        def _ensure_unique_base_key(base_key: str, used_base_keys: Dict[str, int]) -> str:
            if base_key not in used_base_keys:
                used_base_keys[base_key] = 1
                return base_key
            used_base_keys[base_key] += 1
            return f"{base_key}_dup_{used_base_keys[base_key]}"

        def _sanitize_id_component(x: str) -> str:
            s = safe_str(x).strip()
            if not s:
                return ""
            s = re.sub(r"[^A-Za-z0-9_]+", "_", s)
            s = s.strip("_")
            return s

        all_chunks: List[Dict[str, Any]] = []
        doc2chunks: Dict[str, Dict[str, Any]] = {}

        # legacy index maps
        title2keys: Dict[str, List[str]] = {}
        key2title: Dict[str, str] = {}
        key2doc_index: Dict[str, int] = {}

        used_base_keys: Dict[str, int] = {}

        for i in range(n_docs):
            res = results_map.get(i)
            if not res:
                continue

            chunks = res.get("document_chunks", []) or []
            if not chunks:
                continue

            chunk_dicts = [c.dict() for c in chunks]

            try:
                chunk_dicts.sort(key=order_key)
            except Exception as e:
                if verbose:
                    logger.warning(f"Sort failed for doc {i}: {e}")

            md0 = (chunk_dicts[0].get("metadata", {}) or {})
            title = safe_title(md0.get("doc_title") or md0.get("title") or f"doc_{i}")

            raw_doc = documents[i] if (0 <= i < len(documents)) else {}
            raw_id = None
            if isinstance(raw_doc, dict):
                raw_id = raw_doc.get("_id", None)
                if raw_id is None:
                    raw_id = raw_doc.get("id", None)

            doc_segment_id = str(raw_id).strip() if raw_id is not None else f"doc_{i}_seg_1"
            doc_segment_id = _sanitize_id_component(doc_segment_id) or f"doc_{i}_seg_1"
            raw_doc_id = ""
            if isinstance(raw_doc, dict):
                raw_doc_id = safe_str(
                    raw_doc.get("raw_doc_id")
                    or (raw_doc.get("metadata") or {}).get("raw_doc_id")
                ).strip()
            if not raw_doc_id:
                m = re.match(r"^(.*)_seg_\d+$", doc_segment_id)
                raw_doc_id = m.group(1) if m else doc_segment_id

            # stable chunk ids + metadata
            for cidx in range(len(chunk_dicts)):
                cd = chunk_dicts[cidx]
                if not isinstance(cd, dict):
                    continue
                chunk_id = f"{doc_segment_id}_chunk_{cidx}"
                cd["id"] = chunk_id
                md = cd.get("metadata") or {}
                if not isinstance(md, dict):
                    md = {}
                md["raw_doc_id"] = raw_doc_id
                md["doc_segment_id"] = doc_segment_id
                md["source_title"] = title
                cd["metadata"] = md

            base_key_raw = _build_base_key(md0, i, doc_segment_id)
            base_key = _ensure_unique_base_key(base_key_raw, used_base_keys)

            parts = _split_chunks_into_parts(chunk_dicts)

            for pidx, part_chunks in enumerate(parts, start=1):
                document_id = f"{base_key}_part_{pidx}"

                if document_id in doc2chunks:
                    kk = 2
                    while f"{document_id}_dup_{kk}" in doc2chunks:
                        kk += 1
                    document_id = f"{document_id}_dup_{kk}"

                document_md = document_metadata_from_first_chunk(part_chunks)
                document_md["raw_doc_id"] = raw_doc_id
                document_md["doc_segment_id"] = doc_segment_id
                document_md["source_title"] = title

                for cd in part_chunks:
                    if not isinstance(cd, dict):
                        continue
                    md = cd.get("metadata") or {}
                    if not isinstance(md, dict):
                        md = {}
                    md["document_id"] = document_id
                    md["raw_doc_id"] = raw_doc_id
                    md["doc_segment_id"] = doc_segment_id
                    cd["metadata"] = md

                doc2chunks[document_id] = {
                    "document_metadata": document_md,
                    "chunks": part_chunks,
                }

                title2keys.setdefault(title, []).append(document_id)
                key2title[document_id] = title
                key2doc_index[document_id] = i

            all_chunks.extend(chunk_dicts)

        dump_json(os.path.join(base, "all_document_chunks.json"), all_chunks)
        dump_json(os.path.join(base, "doc2chunks.json"), doc2chunks)
        dump_json(
            os.path.join(base, "doc2chunks_index.json"),
            {
                "title2keys": title2keys,
                "key2title": key2title,
                "key2doc_index": key2doc_index,
            },
        )

        dp_cfg = getattr(self.config, "document_processing", None)
        if store_vector_chunks is None:
            store_vector_chunks = bool(getattr(dp_cfg, "store_vector_chunks", True))
        if sentence_chunk_size is None:
            sentence_chunk_size = int(getattr(dp_cfg, "sentence_chunk_size", 200))
        if sentence_chunk_overlap is None:
            sentence_chunk_overlap = int(getattr(dp_cfg, "sentence_chunk_overlap", 50))
        if reset_vector_collections is None:
            reset_vector_collections = bool(getattr(dp_cfg, "reset_vector_collections", True))

        if store_vector_chunks:
            self._store_vectordb_from_doc2chunks(
                doc2chunks=doc2chunks,
                reset_collections=reset_vector_collections,
                sentence_chunk_size=sentence_chunk_size,
                sentence_chunk_overlap=sentence_chunk_overlap,
                verbose=verbose,
            )

        if verbose:
            logger.info(f"Chunking done. documents={len(doc2chunks)} chunks={len(all_chunks)}")
            if failed_indices:
                try:
                    failed_show = sorted(failed_indices)
                except Exception:
                    failed_show = failed_indices
                logger.warning(f"{len(failed_indices)} docs failed after {rounds} round(s): {failed_show}")

    # -------------------------
    # 2) document extraction (threaded)
    # -------------------------
    def _extract_one_document(
        self,
        document_id: str,
        document_pack: Dict[str, Any],
        *,
        aggressive_clean: bool = True,
    ) -> Dict[str, Any]:
        chunks = document_pack.get("chunks") or []
        chunk_ids = [c.get("id") for c in chunks if isinstance(c, dict) and c.get("id")]
        packed_chunks = self._pack_document_chunks_for_extraction(document_id=document_id, chunks=chunks)

        out = self.extract_agent.run_document(
            packed_chunks,
            aggressive_clean=aggressive_clean,
            document_rid_namespace=f"document:{document_id}",
        )

        return {
            "ok": True,
            "document_id": document_id,
            "document_metadata": document_pack.get("document_metadata") or {},
            "chunk_ids": chunk_ids,
            "entities": out.get("entities", []) or [],
            "relations": out.get("relations", []) or [],
            "error": "",
        }

    def _pack_document_chunks_for_extraction(
        self,
        *,
        document_id: str,
        chunks: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        cfg = getattr(self.config, "knowledge_graph_builder", None)
        short_part_word_threshold = max(
            1,
            int(getattr(cfg, "extraction_pack_short_chunk_word_threshold", 220) or 220),
        )
        max_pack_words = max(
            short_part_word_threshold,
            int(getattr(cfg, "extraction_pack_max_words", 900) or 900),
        )

        packed: List[Dict[str, Any]] = []
        current_group: List[Dict[str, Any]] = []
        current_words = 0
        pack_index = 0

        def _flush_group() -> None:
            nonlocal current_group, current_words, pack_index
            if not current_group:
                return
            if len(current_group) == 1:
                packed.append(current_group[0])
            else:
                first = current_group[0]
                text_parts: List[str] = []
                source_chunk_ids: List[str] = []
                for item in current_group:
                    text = safe_str(item.get("content")).strip()
                    if text:
                        text_parts.append(text)
                    cid = safe_str(item.get("id")).strip()
                    if cid:
                        source_chunk_ids.append(cid)

                merged = dict(first)
                merged["id"] = f"{safe_str(document_id)}__pack_{pack_index:04d}"
                merged["content"] = "\n\n".join(text_parts).strip()
                merged["source_chunk_ids"] = source_chunk_ids
                packed.append(merged)
                pack_index += 1

            current_group = []
            current_words = 0

        ordered_chunks = [c for c in (chunks or []) if isinstance(c, dict)]
        try:
            ordered_chunks.sort(key=lambda c: (c.get("metadata", {}) or {}).get("order", 0))
        except Exception:
            pass

        for chunk in ordered_chunks:
            text = safe_str(chunk.get("content")).strip()
            words = word_len(text, lang="auto")
            if words <= 0:
                continue

            if words > short_part_word_threshold:
                _flush_group()
                packed.append(chunk)
                continue

            if current_group and current_words + words > max_pack_words:
                _flush_group()

            current_group.append(chunk)
            current_words += words

        _flush_group()

        return packed or ordered_chunks

    def extract_entity_and_relation(
        self,
        *,
        verbose: bool = True,
        retries: int = 3,
        per_task_timeout: float = 1200.0,
        concurrency: Optional[int] = None,
        reset_outputs: bool = False,
        aggressive_clean: bool = True,
    ) -> None:
        """
        document-level extraction (threaded, like property_extraction):
        - Parallel by documents using run_concurrent_with_retries
        - Resume via tmp checkpoint
        - Multi-round retries
        """
        base = self._base_dir()

        doc2chunks_path = os.path.join(base, "doc2chunks.json")
        if not os.path.exists(doc2chunks_path):
            raise FileNotFoundError(f"doc2chunks.json not found at {doc2chunks_path}. Run prepare_chunks() first.")

        doc2chunks: Dict[str, Dict[str, Any]] = load_json(doc2chunks_path)
        document_ids = list(doc2chunks.keys())

        if not document_ids:
            if verbose:
                logger.warning("doc2chunks.json is empty. Nothing to extract.")
            return

        if concurrency is None:
            concurrency = max(1, int(getattr(self, "max_workers", 8)))

        tmp_dir = self._tmp_dir()
        out_results = os.path.join(base, "extraction_results.json")
        out_ent_jsonl = os.path.join(base, "document_entities.jsonl")
        out_rel_jsonl = os.path.join(base, "document_relations.jsonl")

        if reset_outputs:
            self.clear_directory_keep_root(tmp_dir)
            for p in [out_results, out_ent_jsonl, out_rel_jsonl]:
                if os.path.exists(p):
                    try:
                        os.remove(p)
                    except Exception:
                        pass

        done: List[str] = []
        pending: List[str] = []

        for doc_id in document_ids:
            if not self._tmp_exists(doc_id):
                pending.append(doc_id)
                continue
            payload = self._load_tmp(doc_id) or {}
            if isinstance(payload, dict) and payload.get("ok") is True:
                done.append(doc_id)
            else:
                pending.append(doc_id)

        if verbose:
            logger.info(
                f"document extraction total={len(document_ids)} done={len(done)} pending={len(pending)} "
                f"concurrency={concurrency} timeout={per_task_timeout}s retries={retries}"
            )

        if not pending:
            merged: Dict[str, Any] = {}
            for doc_id in document_ids:
                p = self._load_tmp(doc_id)
                if isinstance(p, dict):
                    merged[doc_id] = p
            dump_json(out_results, merged)

            if verbose:
                ok_cnt = sum(1 for v in merged.values() if isinstance(v, dict) and v.get("ok"))
                fail_cnt = sum(1 for v in merged.values() if isinstance(v, dict) and (not v.get("ok")))
                logger.info(f"All documents already processed. ok={ok_cnt} failed={fail_cnt} output={out_results}")
            return

        items: List[str] = pending

        def _task(doc_id: str) -> Dict[str, Any]:
            pack = doc2chunks.get(doc_id, {}) or {}
            try:
                return self._extract_one_document(doc_id, pack, aggressive_clean=aggressive_clean)
            except Exception as e:
                chunks = pack.get("chunks") or []
                chunk_ids = [c.get("id") for c in chunks if isinstance(c, dict) and c.get("id")]
                return {
                    "ok": False,
                    "document_id": doc_id,
                    "document_metadata": pack.get("document_metadata") or {},
                    "chunk_ids": chunk_ids,
                    "entities": [],
                    "relations": [],
                    "error": f"{type(e).__name__}: {e}",
                }

        def _is_empty_fn(res: Any) -> bool:
            if not isinstance(res, dict):
                return True
            if res.get("ok") is not True:
                return True
            return False

        mem_cfg = getattr(self.config, "extraction_memory", None)
        mem_store_path = (
            str(getattr(mem_cfg, "realtime_store_path", "") or "")
            if mem_cfg is not None
            else ""
        ) or "data/memory/realtime_memory"
        os.makedirs(mem_store_path, exist_ok=True)
        mem_dump_path = os.path.join(mem_store_path, "memory_dump_realtime.json")
        mem_realtime_every = max(1, int(getattr(mem_cfg, "flush_every_n_docs", 10) or 10))
        mem_realtime_ok_count = 0
        distill_every = max(1, int(getattr(mem_cfg, "distill_every_n_docs", 20) or 20))
        distill_docs_pending = 0

        def _attempt_checkpoint(idx: int, res: Any, ok: bool, attempt_idx: int) -> None:
            nonlocal mem_realtime_ok_count, distill_docs_pending
            if not (0 <= idx < len(items)):
                return
            doc_id = items[idx]
            if ok and isinstance(res, dict):
                self._save_tmp(doc_id, res)
                if self._memory_store is not None:
                    try:
                        self._memory_store.mark_doc_processed()
                    except Exception:
                        pass
                    mem_realtime_ok_count += 1
                    if mem_realtime_ok_count >= mem_realtime_every:
                        try:
                            self._memory_store.flush()
                            self._memory_store.dump_to_json(mem_dump_path, silent=True)
                        except Exception as e:
                            logger.warning("Realtime memory checkpoint failed: %s", e)
                        mem_realtime_ok_count = 0
                if self._memory_distiller is not None:
                    distill_docs_pending += 1
                    if distill_docs_pending >= distill_every:
                        self.maybe_distill_memory(
                            docs_processed=distill_docs_pending,
                            force=False,
                            reason="during_extract",
                            extra_context={"checkpoint_doc_id": doc_id},
                        )
                        distill_docs_pending = 0
                return
            pack = doc2chunks.get(doc_id, {}) or {}
            chunks = pack.get("chunks") or []
            chunk_ids = [c.get("id") for c in chunks if isinstance(c, dict) and c.get("id")]
            payload = {
                "ok": False,
                "document_id": doc_id,
                "document_metadata": pack.get("document_metadata") or {},
                "chunk_ids": chunk_ids,
                "entities": [],
                "relations": [],
                "error": f"attempt_{attempt_idx+1}_failed",
            }
            self._save_tmp(doc_id, payload)

        results_map, failed_indices = run_concurrent_with_retries(
            items=items,
            task_fn=_task,
            per_task_timeout=per_task_timeout,
            max_retry_rounds=max(1, int(retries)),
            max_in_flight=concurrency,
            max_workers=concurrency,
            thread_name_prefix="extract",
            desc_prefix="Extracting documents",
            treat_empty_as_failure=True,
            is_empty_fn=_is_empty_fn,
            on_attempt_result=_attempt_checkpoint,
        )

        # persist
        for idx, doc_id in enumerate(items):
            res = results_map.get(idx)
            if not isinstance(res, dict):
                pack = doc2chunks.get(doc_id, {}) or {}
                res = {
                    "ok": False,
                    "document_id": doc_id,
                    "document_metadata": pack.get("document_metadata") or {},
                    "chunk_ids": [],
                    "entities": [],
                    "relations": [],
                    "error": "unknown failure",
                }

            self._save_tmp(doc_id, res)

            if res.get("ok") is True:
                try:
                    with open(out_ent_jsonl, "a", encoding="utf-8") as f:
                        f.write(
                            json.dumps(
                                {
                                    "document_id": doc_id,
                                    "document_metadata": res.get("document_metadata") or {},
                                    "entities": res.get("entities") or [],
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                    with open(out_rel_jsonl, "a", encoding="utf-8") as f:
                        f.write(
                            json.dumps(
                                {
                                    "document_id": doc_id,
                                    "document_metadata": res.get("document_metadata") or {},
                                    "relations": res.get("relations") or [],
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                except Exception:
                    pass

        # merge all
        merged: Dict[str, Any] = {}
        for doc_id in document_ids:
            p = self._load_tmp(doc_id)
            if isinstance(p, dict):
                merged[doc_id] = p
        dump_json(out_results, merged)

        if verbose:
            ok_cnt = sum(1 for v in merged.values() if isinstance(v, dict) and v.get("ok"))
            fail_cnt = sum(1 for v in merged.values() if isinstance(v, dict) and (not v.get("ok")))
            logger.info(f"document extraction finished. ok={ok_cnt} failed={fail_cnt} output={out_results}")

            if failed_indices:
                failed_docs = [items[i] for i in failed_indices if 0 <= i < len(items)]
                logger.warning(f"Failed documents after retries: {failed_docs[:10]} (total={len(failed_docs)})")

        # Flush memory store at batch end
        if self._memory_store is not None:
            try:
                self._memory_store.flush()
            except Exception as _e:
                logger.warning("Memory store flush failed: %s", _e)

        if self._memory_distiller is not None and distill_docs_pending > 0:
            self.maybe_distill_memory(
                docs_processed=distill_docs_pending,
                force=False,
                reason="post_extract_batch_remainder",
                extra_context={
                    "ok_documents": sum(1 for v in merged.values() if isinstance(v, dict) and v.get("ok") is True),
                    "failed_documents": sum(1 for v in merged.values() if isinstance(v, dict) and v.get("ok") is not True),
                },
            )

    # -------------------------
    # 3) Extraction refinement
    # -------------------------
    def run_extraction_refinement(
        self,
        *,
        verbose: bool = True,
        extraction_results_path: Optional[str] = None,
        refined_results_path: Optional[str] = None,
        reset_outputs: bool = False,
        enable_scope_refine_llm: bool = False,
        type_timeout: float = 120.0,
        scope_timeout: float = 180.0,
        disambig_timeout_summary: float = 180.0,
        disambig_timeout_merge: float = 120.0,
    ) -> None:
        base = self._base_dir()

        if extraction_results_path is None:
            extraction_results_path = os.path.join(base, "extraction_results.json")

        if refined_results_path is None:
            refined_results_path = os.path.join(base, "extraction_results_refined.json")

        if not os.path.exists(extraction_results_path):
            raise FileNotFoundError(f"Extraction results not found: {extraction_results_path}")

        if reset_outputs and os.path.exists(refined_results_path):
            try:
                os.remove(refined_results_path)
            except Exception:
                pass

        with open(extraction_results_path, "r", encoding="utf-8") as f:
            document_results: Dict[str, Dict[str, Any]] = json.load(f)

        if not isinstance(document_results, dict) or not document_results:
            if verbose:
                logger.warning("extraction_results is empty or invalid. Nothing to refine.")
            with open(refined_results_path, "w", encoding="utf-8") as f:
                json.dump({}, f, ensure_ascii=False, indent=2)
            return

        self.graph_refiner = GraphRefiner(self.config, self.llm, memory_store=self._memory_store)
        refined = self.graph_refiner.run_all(
            document_results,
            verbose=verbose,
            type_timeout=type_timeout,
            scope_timeout=scope_timeout,
            disambig_timeout_summary=disambig_timeout_summary,
            disambig_timeout_merge=disambig_timeout_merge,
            enable_scope_refine_llm=enable_scope_refine_llm,
        )

        os.makedirs(os.path.dirname(refined_results_path), exist_ok=True)
        with open(refined_results_path, "w", encoding="utf-8") as f:
            json.dump(refined, f, ensure_ascii=False, indent=2)

        if self._memory_store is not None:
            try:
                self._memory_store.flush()
            except Exception as _e:
                logger.warning("Memory store flush after refinement failed: %s", _e)

        # Distillation is intentionally triggered in extraction stage only.

    # -------------------------
    # 4) Build entity_basic_info.json + relation_basic_info.json
    # -------------------------
    def build_entity_and_relation_basic_info(
        self,
        *,
        verbose: bool = True,
        refined_results_path: Optional[str] = None,
        entity_output_path: Optional[str] = None,
        relation_output_path: Optional[str] = None,
        merge_induced_across_documents: bool = False,
    ) -> None:
        base = self._base_dir()

        if refined_results_path is None:
            refined_results_path = os.path.join(base, "extraction_results_refined.json")
        if entity_output_path is None:
            entity_output_path = os.path.join(base, "entity_basic_info.json")
        if relation_output_path is None:
            relation_output_path = os.path.join(base, "relation_basic_info.json")

        if not os.path.exists(refined_results_path):
            raise FileNotFoundError(f"Refined results not found: {refined_results_path}")

        document_results: Dict[str, Dict[str, Any]] = load_json(refined_results_path)
        if not isinstance(document_results, dict) or not document_results:
            dump_json(entity_output_path, {})
            dump_json(relation_output_path, [])
            if verbose:
                logger.warning("Refined results empty. Wrote empty outputs.")
            return

        def _stable_id(key: str, prefix: str) -> str:
            return prefix + hashlib.md5(key.encode("utf-8")).hexdigest()[:12]

        def _as_type_list(t: Any) -> List[str]:
            if isinstance(t, str) and t.strip():
                return [t.strip()]
            if isinstance(t, list):
                out: List[str] = []
                for x in t:
                    if isinstance(x, str) and x.strip():
                        out.append(x.strip())
                return out
            return []

        def _norm_type_value(t: Any) -> Any:
            tl = _as_type_list(t)
            if not tl:
                return self.GENERAL_SEMANTIC_TYPE

            allowed = set(getattr(self, "ALLOWED_TYPES", []) or [])
            if allowed:
                tl = [x for x in tl if x in allowed] or [self.GENERAL_SEMANTIC_TYPE]

            seen = set()
            tl2: List[str] = []
            for x in tl:
                if x not in seen:
                    seen.add(x)
                    tl2.append(x)

            t2c = getattr(self, "TYPE2CATEGORY", {}) or {}
            cprio = getattr(self, "CATEGORY_PRIORITY", {}) or {}

            def _rank(tp: str) -> Tuple[int, str]:
                cat = t2c.get(tp, "general_semantic")
                return (int(cprio.get(cat, 9999)), tp)

            tl2.sort(key=_rank)
            return tl2[0] if len(tl2) == 1 else tl2

        def _merge_types(a: Any, b: Any) -> Any:
            al = _as_type_list(a)
            bl = _as_type_list(b)
            merged = al + bl
            return _norm_type_value(merged)

        def _primary_type(t: Any) -> str:
            tl = _as_type_list(t)
            if not tl:
                return self.GENERAL_SEMANTIC_TYPE

            allowed = set(getattr(self, "ALLOWED_TYPES", []) or [])
            if allowed:
                tl = [x for x in tl if x in allowed] or [self.GENERAL_SEMANTIC_TYPE]

            t2c = getattr(self, "TYPE2CATEGORY", {}) or {}
            cprio = getattr(self, "CATEGORY_PRIORITY", {}) or {}

            def _rank(tp: str) -> Tuple[int, str]:
                cat = t2c.get(tp, "general_semantic")
                return (int(cprio.get(cat, 9999)), tp)

            tl.sort(key=_rank)
            return tl[0] if tl else self.GENERAL_SEMANTIC_TYPE

        def _is_induced_primary(primary_type: str) -> bool:
            induced = set(getattr(self, "INDUCED_TYPES", set()) or set())
            return primary_type in induced

        def _merge_text(a: str, b: str) -> str:
            a, b = (a or "").strip(), (b or "").strip()
            if not b:
                return a
            if not a or b in a:
                return b if not a else a
            return a + "\n" + b

        # 1) bucket entities
        buckets: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
        bucket_source_documents: Dict[Tuple[Any, ...], set] = defaultdict(set)

        for document_id, payload in document_results.items():
            if not isinstance(payload, dict) or payload.get("ok") is False:
                continue

            base_document = strip_part_suffix(document_id)

            for e in (payload.get("entities") or []):
                if not isinstance(e, dict):
                    continue

                raw_name = safe_str(e.get("name"))
                if not raw_name:
                    continue

                t_raw = e.get("type")
                t_norm = _norm_type_value(t_raw)
                t_primary = _primary_type(t_raw)

                scope = safe_str(e.get("scope") or "local").lower() or "local"
                desc = safe_str(e.get("description"))
                summ = safe_str(e.get("summary"))

                aliases = e.get("aliases") or []
                if not isinstance(aliases, list):
                    aliases = []
                aliases = [a.strip() for a in aliases if isinstance(a, str) and a.strip()]

                force_local = _is_induced_primary(t_primary) and (not merge_induced_across_documents)

                if scope == "local" or force_local:
                    bkey = ("local", base_document, t_primary, raw_name)
                    out_scope = "local"
                else:
                    bkey = ("global", t_primary, raw_name)
                    out_scope = "global"

                if bkey not in buckets:
                    buckets[bkey] = {
                        "raw_name": raw_name,
                        "type": t_norm,
                        "scope": out_scope,
                        "description": desc,
                        "summary": summ,
                        "aliases": set(aliases),
                    }
                else:
                    ex = buckets[bkey]
                    ex["type"] = _merge_types(ex.get("type"), t_norm)
                    ex["description"] = _merge_text(ex.get("description", ""), desc)
                    ex["summary"] = _merge_text(ex.get("summary", ""), summ)
                    ex.setdefault("aliases", set()).update(aliases)

                bucket_source_documents[bkey].add(document_id)

        # 2) materialize entities
        entities_by_id: Dict[str, Dict[str, Any]] = {}
        local_name_map: Dict[Tuple[str, str], List[str]] = defaultdict(list)
        global_name_map: Dict[str, List[str]] = defaultdict(list)

        for bkey, info in buckets.items():
            kind = bkey[0]
            raw_name = info.get("raw_name")
            aliases = sorted(a for a in info.get("aliases", []) if a != raw_name)

            if kind == "local":
                _, base_document, t_primary, raw_name = bkey
                # Keep local entity display name unchanged; uniqueness is handled by id + scope.
                final_name = raw_name
                # Legacy behavior (kept for reference):
                # final_name = f"{raw_name} ({base_document})"
                ent_id = _stable_id(f"{raw_name}||{t_primary}||{base_document}||local", "ent_")
                src_docs = sorted(bucket_source_documents[bkey])

                entities_by_id[ent_id] = {
                    "id": ent_id,
                    "name": final_name,
                    "type": info.get("type", self.GENERAL_SEMANTIC_TYPE),
                    "scope": "local",
                    "description": info.get("description", ""),
                    "summary": info.get("summary", ""),
                    "aliases": aliases,
                    "source_documents": src_docs,
                }
                local_name_map[(base_document, raw_name)].append(ent_id)
            else:
                _, t_primary, raw_name = bkey
                final_name = raw_name
                ent_id = _stable_id(f"{raw_name}||{t_primary}||global", "ent_")
                src_docs = sorted(bucket_source_documents[bkey])

                entities_by_id[ent_id] = {
                    "id": ent_id,
                    "name": final_name,
                    "type": info.get("type", self.GENERAL_SEMANTIC_TYPE),
                    "scope": "global",
                    "description": info.get("description", ""),
                    "summary": info.get("summary", ""),
                    "aliases": aliases,
                    "source_documents": src_docs,
                }
                global_name_map[raw_name].append(ent_id)

        # 3) build relations
        def _resolve_ids(base_document: str, name: str) -> List[str]:
            if not name:
                return []
            if (base_document, name) in local_name_map:
                return local_name_map[(base_document, name)]
            return global_name_map.get(name, [])

        def _choose_one(ids: List[str]) -> str:
            return ids[0] if ids else ""

        relations_out: List[Dict[str, Any]] = []

        for document_id, payload in document_results.items():
            if not isinstance(payload, dict) or payload.get("ok") is False:
                continue

            base_document = strip_part_suffix(document_id)
            for idx, r in enumerate(payload.get("relations") or []):
                if not isinstance(r, dict):
                    continue

                subj = safe_str(r.get("subject"))
                obj = safe_str(r.get("object"))
                pred = safe_str(r.get("relation_type") or r.get("predicate"))
                persist = r.get("persistence") or "phase"

                s_id = _choose_one(_resolve_ids(base_document, subj))
                o_id = _choose_one(_resolve_ids(base_document, obj))

                if s_id and o_id and pred:
                    rid_key = f"{s_id}||{pred}||{o_id}||{base_document}"
                else:
                    rid_key = f"{subj}||{pred}||{obj}||{base_document}||{idx}"

                rel_id = _stable_id(rid_key, "rel_")

                relations_out.append(
                    {
                        "id": rel_id,
                        "relation_type": pred,
                        "subject": subj,
                        "object": obj,
                        "subject_id": s_id,
                        "object_id": o_id,
                        "relation_name": safe_str(r.get("relation_name")),
                        "description": safe_str(r.get("description")),
                        "persistence": persist,
                        "base_document": base_document,
                        "document_id": document_id,
                    }
                )

        dump_json(entity_output_path, entities_by_id)
        dump_json(relation_output_path, relations_out)

        if verbose:
            logger.info(
                "Built basic infos. "
                f"entity_out={entity_output_path} relation_out={relation_output_path}"
            )

        # Write extraction stats after entity/relation info is ready
        try:
            self.write_extraction_stats(
                entity_output_path=entity_output_path,
                relation_output_path=relation_output_path,
                verbose=verbose,
            )
        except Exception as _e:
            logger.warning("write_extraction_stats failed: %s", _e)

    # -------------------------
    # 4b) Extraction statistics
    # -------------------------
    def write_extraction_stats(
        self,
        *,
        extraction_results_path: Optional[str] = None,
        entity_output_path: Optional[str] = None,
        relation_output_path: Optional[str] = None,
        output_path: Optional[str] = None,
        verbose: bool = True,
    ) -> None:
        """
        Compile and write extraction statistics to audit_logs/extraction_stats.json.

        Reads from (whichever files exist):
          - extraction_results.json          : raw entity/relation counts per document
          - entity_basic_info.json           : final entity counts by type and scope
          - relation_basic_info.json         : final relation counts by relation_type
          - audit_logs/refine_entity_types_audit.json        : type-conflict repair counts
          - audit_logs/run_entity_disambiguation_audit.json  : disambiguation merge counts
        """
        import time as _time

        base = self._base_dir()
        audit_dir = os.path.join(base, "audit_logs")
        os.makedirs(audit_dir, exist_ok=True)

        if extraction_results_path is None:
            extraction_results_path = os.path.join(base, "extraction_results.json")
        if entity_output_path is None:
            entity_output_path = os.path.join(base, "entity_basic_info.json")
        if relation_output_path is None:
            relation_output_path = os.path.join(base, "relation_basic_info.json")
        if output_path is None:
            output_path = os.path.join(audit_dir, "extraction_stats.json")

        stats: Dict[str, Any] = {
            "created_at": _time.strftime("%Y%m%d_%H%M%S", _time.localtime()),
        }

        # ------------------------------------------------------------------
        # Raw extraction counts (before refinement)
        # ------------------------------------------------------------------
        if os.path.exists(extraction_results_path):
            try:
                doc_results = load_json(extraction_results_path)
                if isinstance(doc_results, dict):
                    ok_docs = [v for v in doc_results.values() if isinstance(v, dict) and v.get("ok")]
                    fail_docs = [v for v in doc_results.values() if isinstance(v, dict) and not v.get("ok")]
                    raw_ents = sum(len(v.get("entities") or []) for v in ok_docs)
                    raw_rels = sum(len(v.get("relations") or []) for v in ok_docs)
                    stats["extraction_raw"] = {
                        "num_documents": len(doc_results),
                        "num_documents_ok": len(ok_docs),
                        "num_documents_failed": len(fail_docs),
                        "entities_total": raw_ents,
                        "relations_total": raw_rels,
                    }
            except Exception as _e:
                logger.warning("write_extraction_stats: extraction_results read error: %s", _e)

        # ------------------------------------------------------------------
        # Final entity counts (by type and scope)
        # ------------------------------------------------------------------
        if os.path.exists(entity_output_path):
            try:
                entities_by_id = load_json(entity_output_path)
                if isinstance(entities_by_id, dict):
                    by_type: Dict[str, int] = defaultdict(int)
                    by_scope: Dict[str, int] = defaultdict(int)
                    for ent in entities_by_id.values():
                        if not isinstance(ent, dict):
                            continue
                        t = ent.get("type")
                        t_str = t[0] if isinstance(t, list) and t else (safe_str(t) or "unknown")
                        by_type[t_str] += 1
                        sc = safe_str(ent.get("scope") or "unknown")
                        by_scope[sc] += 1
                    stats["entity_counts"] = {
                        "total": len(entities_by_id),
                        "by_type": dict(sorted(by_type.items(), key=lambda x: -x[1])),
                        "by_scope": dict(by_scope),
                    }
            except Exception as _e:
                logger.warning("write_extraction_stats: entity_basic_info read error: %s", _e)

        # ------------------------------------------------------------------
        # Final relation counts (by relation_type)
        # ------------------------------------------------------------------
        if os.path.exists(relation_output_path):
            try:
                relations = load_json(relation_output_path)
                if isinstance(relations, list):
                    by_rtype: Dict[str, int] = defaultdict(int)
                    for rel in relations:
                        if not isinstance(rel, dict):
                            continue
                        rt = safe_str(rel.get("relation_type") or "unknown")
                        by_rtype[rt] += 1
                    stats["relation_counts"] = {
                        "total": len(relations),
                        "by_type": dict(sorted(by_rtype.items(), key=lambda x: -x[1])),
                    }
            except Exception as _e:
                logger.warning("write_extraction_stats: relation_basic_info read error: %s", _e)

        # ------------------------------------------------------------------
        # Repair counts from audit logs
        # ------------------------------------------------------------------
        repairs: Dict[str, Any] = {}

        # — Entity type conflict repairs —
        type_audit_path = os.path.join(audit_dir, "refine_entity_types_audit.json")
        if os.path.exists(type_audit_path):
            try:
                type_audit = load_json(type_audit_path)
                if isinstance(type_audit, dict):
                    audits = type_audit.get("audits") or []
                    drop_count = sum(1 for a in audits if isinstance(a, dict) and a.get("decision") == "drop")
                    keep_count = sum(1 for a in audits if isinstance(a, dict) and a.get("decision") == "keep")
                    rename_count = sum(
                        len([
                            act for act in (a.get("actions") or [])
                            if isinstance(act, dict) and act.get("new_name")
                        ])
                        for a in audits if isinstance(a, dict) and a.get("decision") == "keep"
                    )
                    relation_fix_count = sum(
                        len(a.get("relation_fixes") or [])
                        for a in audits if isinstance(a, dict)
                    )
                    repairs["entity_type_conflicts"] = {
                        "num_entities_checked": int(type_audit.get("num_entities_checked", 0)),
                        "num_conflict_cases": int(type_audit.get("num_audit_items", 0)),
                        "drop": drop_count,
                        "keep": keep_count,
                        "entity_renames_from_keep": rename_count,
                        "relation_fixes": relation_fix_count,
                    }
            except Exception as _e:
                logger.warning("write_extraction_stats: refine_entity_types audit read error: %s", _e)

        # — Entity disambiguation —
        disambig_audit_path = os.path.join(audit_dir, "run_entity_disambiguation_audit.json")
        if os.path.exists(disambig_audit_path):
            try:
                disambig_audit = load_json(disambig_audit_path)
                if isinstance(disambig_audit, dict):
                    audits = disambig_audit.get("audits") or []
                    meta_entry = next((a for a in audits if isinstance(a, dict) and a.get("kind") == "meta"), {})
                    meta_stats = meta_entry.get("stats", {}) or {}
                    merge_groups = [a for a in audits if isinstance(a, dict) and a.get("kind") == "merge_group"]
                    rename_map_entry = next(
                        (a for a in audits if isinstance(a, dict) and a.get("kind") == "apply_rename_map"), {}
                    )
                    rename_stats = rename_map_entry.get("stats", {}) or {}
                    repairs["entity_disambiguation"] = {
                        "entities_total": int(meta_stats.get("entities_total", 0)),
                        "entities_considered": int(meta_stats.get("entities_considered", 0)),
                        "entities_skipped_scope": int(meta_stats.get("entities_skipped_scope", 0)),
                        "merge_groups_detected": len(merge_groups),
                        "entities_merged_aliases": int(rename_stats.get("entity_renames", 0)),
                        "relation_endpoint_renames": int(rename_stats.get("relation_endpoint_renames", 0)),
                    }
            except Exception as _e:
                logger.warning("write_extraction_stats: disambiguation audit read error: %s", _e)

        if repairs:
            # Compute total repair actions across all repair types
            total_actions = 0
            tc = repairs.get("entity_type_conflicts", {})
            total_actions += tc.get("drop", 0) + tc.get("entity_renames_from_keep", 0) + tc.get("relation_fixes", 0)
            dg = repairs.get("entity_disambiguation", {})
            total_actions += dg.get("entities_merged_aliases", 0)
            repairs["total_repair_actions"] = total_actions
            stats["repairs"] = repairs

        # ------------------------------------------------------------------
        # Write
        # ------------------------------------------------------------------
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

        if verbose:
            ent_total = (stats.get("entity_counts") or {}).get("total", "?")
            rel_total = (stats.get("relation_counts") or {}).get("total", "?")
            repair_total = (stats.get("repairs") or {}).get("total_repair_actions", "?")
            logger.info(
                "Extraction stats written to %s "
                "(entities=%s relations=%s total_repair_actions=%s)",
                output_path, ent_total, rel_total, repair_total,
            )

    # -------------------------
    # 5) Postprocess relations, build graph, compute degrees
    # -------------------------
    def postprocess_and_save(
        self,
        *,
        relation_basic_info_path: Optional[str] = None,
        entity_basic_info_path: Optional[str] = None,
        relation_output_path: Optional[str] = None,
        graph_output_path: Optional[str] = None,
        enable_llm: bool = True,
        verbose: bool = True,
    ) -> None:
        base = self._base_dir()

        if relation_basic_info_path is None:
            relation_basic_info_path = os.path.join(base, "relation_basic_info.json")
        if entity_basic_info_path is None:
            entity_basic_info_path = os.path.join(base, "entity_basic_info.json")

        if relation_output_path is None:
            relation_output_path = os.path.join(base, "relation_info_refined.json")

        if graph_output_path is None:
            graph_output_path = os.path.join(base, "graph_nx.pkl")

        entities_by_id = load_json(entity_basic_info_path)
        if not isinstance(entities_by_id, dict):
            entities_by_id = {}

        self.graph_refiner = GraphRefiner(self.config, self.llm, memory_store=self._memory_store)

        merged_relations = self.graph_refiner.merge_relations(
            input_path=relation_basic_info_path,
            output_path=None,
            enable_llm=enable_llm,
        )

        G = nx.MultiDiGraph()

        for eid, ent in entities_by_id.items():
            if not isinstance(ent, dict):
                continue
            node_id = safe_str(ent.get("id")) or safe_str(eid)
            if not node_id:
                continue
            G.add_node(node_id, **ent)

        skipped_missing_ids = 0
        for r in merged_relations:
            if not isinstance(r, dict):
                continue

            sid = r.get("subject_id")
            oid = r.get("object_id")
            if not is_nonempty_str(sid) or not is_nonempty_str(oid):
                skipped_missing_ids += 1
                continue

            if sid not in G:
                G.add_node(sid)
            if oid not in G:
                G.add_node(oid)

            edge_key = safe_str(r.get("id")) or None
            if edge_key is None:
                G.add_edge(sid, oid, **r)
            else:
                G.add_edge(sid, oid, key=edge_key, **r)

        dump_pickle(graph_output_path, G)
        dump_json(relation_output_path, merged_relations)

        if verbose:
            logger.info(
                "Postprocess done. "
                f"relation_out={relation_output_path} graph_out={graph_output_path} "
                f"(entities={len(entities_by_id)} relations={len(merged_relations)} "
                f"graph_nodes={G.number_of_nodes()} graph_edges={G.number_of_edges()} "
                f"skipped_edges_missing_ids={skipped_missing_ids})"
            )

    # -------------------------
    # 6) Graph-based property extraction
    # -------------------------
    def extract_properties(
        self,
        *,
        graph_path: Optional[str] = None,
        entity_schema_path: Optional[str] = None,
        entity_info_path: Optional[str] = None,
        output_entity_info_path: Optional[str] = None,
        verbose: bool = True,
        # selection
        scope: str = "global",
        metric: str = "total_degree",
        threshold: float = 2.0,
        num_top: Optional[int] = None,
        node_type: Optional[str] = None,
        exclude_relation_types: Optional[Set[str]] = None,
        include_relation_types: Optional[Set[str]] = None,
        # chunking
        chunk_size: int = 1200,
        chunk_overlap: int = 200,
        max_edge_descriptions_per_node: Optional[int] = None,
        max_context_words_per_node: Optional[int] = None,
        dedupe_edge_descriptions: Optional[bool] = None,
        # concurrency
        max_workers: Optional[int] = None,
        per_task_timeout: float = 120.0,
        retries: int = 2,
        # merge
        num_properties: int = 10,
        merge_max_workers: Optional[int] = None,
        merge_timeout: float = 120.0,
        merge_retries: int = 2,
    ) -> None:
        base = self._base_dir()

        if graph_path is None:
            graph_path = os.path.join(base, "graph_nx.pkl")
        if entity_info_path is None:
            entity_info_path = os.path.join(base, "entity_basic_info.json")
        if output_entity_info_path is None:
            output_entity_info_path = os.path.join(base, "entity_info_refined.json")
        if entity_schema_path is None:
            entity_schema_path = self.entity_schema_path

        if exclude_relation_types is None:
            exclude_relation_types = set()

        if not os.path.exists(graph_path):
            raise FileNotFoundError(f"Graph not found: {graph_path}")
        if not os.path.exists(entity_schema_path):
            raise FileNotFoundError(f"Entity schema not found: {entity_schema_path}")

        with open(graph_path, "rb") as f:
            G: nx.Graph = pickle.load(f)

        entities_by_id: Dict[str, Dict[str, Any]] = {}
        if os.path.exists(entity_info_path):
            try:
                entities_by_id = load_json(entity_info_path)
                if not isinstance(entities_by_id, dict):
                    entities_by_id = {}
            except Exception:
                entities_by_id = {}

        with open(entity_schema_path, "r", encoding="utf-8") as f:
            entity_schema = json.load(f)
        if not isinstance(entity_schema, list):
            entity_schema = []

        if max_workers is None:
            max_workers = int(getattr(self, "max_workers", 8))
        kg_cfg = getattr(self.config, "knowledge_graph_builder", None)
        if max_edge_descriptions_per_node is None:
            max_edge_descriptions_per_node = int(
                getattr(kg_cfg, "property_context_max_edge_descriptions", 80) or 80
            )
        if max_context_words_per_node is None:
            max_context_words_per_node = int(
                getattr(kg_cfg, "property_context_max_total_words", 2400) or 2400
            )
        if dedupe_edge_descriptions is None:
            raw_dedupe = getattr(kg_cfg, "property_context_dedupe_descriptions", True)
            if isinstance(raw_dedupe, bool):
                dedupe_edge_descriptions = raw_dedupe
            else:
                dedupe_edge_descriptions = safe_str(raw_dedupe).strip().lower() not in {"", "0", "false", "no", "off"}

        agent = PropertyExtractionAgent(self.config, self.llm)

        res = agent.run(
            G=G,
            entities_by_id=entities_by_id,
            entity_schema=entity_schema,
            general_semantic_type=self.GENERAL_SEMANTIC_TYPE,
            scope=scope,
            metric=metric,
            threshold=float(threshold),
            num_top=num_top,
            node_type=node_type,
            exclude_relation_types=set(exclude_relation_types) if exclude_relation_types else None,
            include_relation_types=set(include_relation_types) if include_relation_types else None,
            chunk_size=int(chunk_size),
            chunk_overlap=int(chunk_overlap),
            max_edge_descriptions_per_node=max_edge_descriptions_per_node,
            max_context_words_per_node=max_context_words_per_node,
            dedupe_edge_descriptions=bool(dedupe_edge_descriptions),
            max_workers=int(max_workers),
            per_task_timeout=float(per_task_timeout),
            retries=int(retries),
            num_properties=int(num_properties),
            merge_max_workers=merge_max_workers,
            merge_timeout=float(merge_timeout),
            merge_retries=int(merge_retries),
            run_concurrent_with_retries_fn=run_concurrent_with_retries,
            verbose=verbose,
        )

        dump_json(output_entity_info_path, res.entity_info)

        if verbose:
            logger.info(
                f"[PropertyExtraction] done. selected_nodes={res.selected_nodes} "
                f"flattened_tasks={res.flattened_tasks} updated_nodes={res.updated_nodes} "
                f"failed_extract_tasks={res.failed_extract_tasks} failed_merge_nodes={res.failed_merge_nodes} "
                f"entity_out={output_entity_info_path}"
            )

    # -------------------------
    # 7) Interaction extraction (JSON + SQL)
    # -------------------------
    @staticmethod
    def _interaction_primary_type(x: Any) -> str:
        if isinstance(x, str):
            return x.strip()
        if isinstance(x, list) and x:
            for t in x:
                s = safe_str(t).strip()
                if s:
                    return s
        return ""

    def _build_interaction_entity_candidates_by_document(
        self,
        entities_by_id: Dict[str, Dict[str, Any]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        by_doc: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        seen: Dict[str, Set[str]] = defaultdict(set)

        for ent in entities_by_id.values():
            if not isinstance(ent, dict):
                continue

            ent_id = safe_str(ent.get("id")).strip()
            name = safe_str(ent.get("name")).strip()
            etype = self._interaction_primary_type(ent.get("type"))
            if not ent_id or not name or not etype:
                continue
            if etype not in {"Character", "Object"}:
                continue

            aliases = ent.get("aliases") or []
            if not isinstance(aliases, list):
                aliases = []
            aliases = [safe_str(a).strip() for a in aliases if safe_str(a).strip()]

            src_docs = ent.get("source_documents") or []
            if not isinstance(src_docs, list):
                src_docs = []

            cand = {
                "id": ent_id,
                "name": name,
                "type": etype,
                "aliases": aliases,
            }

            for d in src_docs:
                doc_id = safe_str(d).strip()
                if not doc_id:
                    continue
                if ent_id in seen[doc_id]:
                    continue
                seen[doc_id].add(ent_id)
                by_doc[doc_id].append(cand)

        for doc_id, arr in by_doc.items():
            arr.sort(key=lambda x: (safe_str(x.get("name")), safe_str(x.get("id"))))
            by_doc[doc_id] = arr
        return by_doc

    def _interaction_doc_type_meta(self) -> Dict[str, str]:
        section_label = safe_str(self.meta.get("section_label", "Document")).strip() or "Document"
        section_id = re.sub(
            r"[^A-Za-z0-9_]+",
            "_",
            section_label,
        ).strip("_").lower() + "_id"
        return {
            "section_label": section_label,
            "section_id": section_id,
            "title": safe_str(self.meta.get("title", "title")).strip() or "title",
            "subtitle": safe_str(self.meta.get("subtitle", "subtitle")).strip() or "subtitle",
        }

    def extract_interactions(
        self,
        *,
        refined_results_path: Optional[str] = None,
        all_chunks_path: Optional[str] = None,
        entity_info_path: Optional[str] = None,
        interaction_json_path: Optional[str] = None,
        interaction_list_json_path: Optional[str] = None,
        retries: int = 3,
        per_task_timeout: float = 1200.0,
        concurrency: Optional[int] = None,
        show_progress: bool = True,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Extract interaction records and persist JSON only.

        Placement in pipeline:
        - after extract_properties() (entity dedup/refinement complete)
        """
        base = self._base_dir()
        interaction_base = self._interaction_base_dir()

        if refined_results_path is None:
            refined_results_path = os.path.join(base, "extraction_results_refined.json")
        if all_chunks_path is None:
            all_chunks_path = os.path.join(base, "all_document_chunks.json")
        if entity_info_path is None:
            p_refined = os.path.join(base, "entity_info_refined.json")
            entity_info_path = p_refined if os.path.exists(p_refined) else os.path.join(base, "entity_basic_info.json")
        if interaction_json_path is None:
            interaction_json_path = os.path.join(interaction_base, "interaction_results.json")
        if interaction_list_json_path is None:
            interaction_list_json_path = os.path.join(
                os.path.dirname(interaction_json_path),
                "interaction_records_list.json",
            )

        if not os.path.exists(refined_results_path):
            raise FileNotFoundError(f"Refined results not found: {refined_results_path}")
        if not os.path.exists(all_chunks_path):
            raise FileNotFoundError(f"all_document_chunks.json not found: {all_chunks_path}")
        if not os.path.exists(entity_info_path):
            raise FileNotFoundError(f"Entity info not found: {entity_info_path}")

        document_results = load_json(refined_results_path)
        all_chunks = load_json(all_chunks_path)
        entities_by_id = load_json(entity_info_path)

        if not isinstance(document_results, dict):
            document_results = {}
        if not isinstance(all_chunks, list):
            all_chunks = []
        if not isinstance(entities_by_id, dict):
            entities_by_id = {}

        chunk_map: Dict[str, Dict[str, Any]] = {}
        for c in all_chunks:
            if not isinstance(c, dict):
                continue
            cid = safe_str(c.get("id")).strip()
            if cid:
                chunk_map[cid] = c

        entities_by_doc = self._build_interaction_entity_candidates_by_document(entities_by_id)
        kg_cfg = getattr(self.config, "knowledge_graph_builder", None)
        min_entity_candidates = max(1, int(getattr(kg_cfg, "interaction_min_entity_candidates", 2) or 2))

        if concurrency is None:
            concurrency = max(1, int(getattr(self, "max_workers", 8)))

        per_doc: Dict[str, Dict[str, Any]] = {}
        flat: List[Dict[str, Any]] = []
        pending_doc_ids: List[str] = []

        for doc_id, payload in document_results.items():
            if not isinstance(payload, dict):
                continue

            if payload.get("ok") is False:
                per_doc[doc_id] = {
                    "ok": False,
                    "document_id": doc_id,
                    "chunk_ids": payload.get("chunk_ids") or [],
                    "interactions": [],
                    "feedback_count": 0,
                    "feedbacks": {},
                    "error": safe_str(payload.get("error")),
                }
                continue

            doc_candidates = entities_by_doc.get(doc_id, [])
            chunk_ids = payload.get("chunk_ids") or []
            if not doc_candidates or not isinstance(chunk_ids, list) or not chunk_ids:
                per_doc[doc_id] = {
                    "ok": True,
                    "document_id": doc_id,
                    "chunk_ids": chunk_ids if isinstance(chunk_ids, list) else [],
                    "interactions": [],
                    "feedback_count": 0,
                    "feedbacks": {},
                    "error": "",
                }
                continue
            if len(doc_candidates) < min_entity_candidates:
                per_doc[doc_id] = {
                    "ok": True,
                    "document_id": doc_id,
                    "chunk_ids": chunk_ids,
                    "interactions": [],
                    "feedback_count": 0,
                    "feedbacks": {},
                    "error": "",
                }
                continue

            pending_doc_ids.append(doc_id)

        if verbose:
            logger.info(
                "[InteractionExtraction] total=%s pending=%s skipped=%s concurrency=%s timeout=%ss retries=%s",
                len(document_results),
                len(pending_doc_ids),
                max(0, len(document_results) - len(pending_doc_ids)),
                concurrency,
                per_task_timeout,
                retries,
            )

        if pending_doc_ids:
            items = pending_doc_ids

            def _task(doc_id: str) -> Dict[str, Any]:
                payload = document_results.get(doc_id) or {}
                doc_candidates = entities_by_doc.get(doc_id, [])
                try:
                    return self.interaction_agent.run_document(
                        document_id=doc_id,
                        payload=payload,
                        chunk_map=chunk_map,
                        entity_candidates=doc_candidates,
                        document_rid_namespace=f"interaction:{doc_id}",
                    )
                except Exception as e:
                    chunk_ids = payload.get("chunk_ids") or []
                    if not isinstance(chunk_ids, list):
                        chunk_ids = []
                    return {
                        "ok": False,
                        "document_id": doc_id,
                        "chunk_ids": chunk_ids,
                        "interactions": [],
                        "feedback_count": 0,
                        "feedbacks": {},
                        "error": f"{type(e).__name__}: {e}",
                    }

            def _is_empty_fn(res: Any) -> bool:
                return (not isinstance(res, dict)) or (res.get("ok") is not True)

            results_map, failed_indices = run_concurrent_with_retries(
                items=items,
                task_fn=_task,
                per_task_timeout=per_task_timeout,
                max_retry_rounds=max(1, int(retries)),
                max_in_flight=concurrency,
                max_workers=concurrency,
                thread_name_prefix="interaction_extract",
                desc_prefix="Extracting interactions",
                treat_empty_as_failure=True,
                is_empty_fn=_is_empty_fn,
                show_progress=bool(show_progress),
            )

            for idx, doc_id in enumerate(items):
                res = results_map.get(idx)
                if not isinstance(res, dict):
                    payload = document_results.get(doc_id) or {}
                    chunk_ids = payload.get("chunk_ids") or []
                    if not isinstance(chunk_ids, list):
                        chunk_ids = []
                    res = {
                        "ok": False,
                        "document_id": doc_id,
                        "chunk_ids": chunk_ids,
                        "interactions": [],
                        "feedback_count": 0,
                        "feedbacks": {},
                        "error": "unknown failure",
                    }
                per_doc[doc_id] = res
                if res.get("ok") is True and isinstance(res.get("interactions"), list):
                    flat.extend(res["interactions"])

            if failed_indices and verbose:
                try:
                    failed_doc_ids = [items[i] for i in sorted(failed_indices) if 0 <= i < len(items)]
                except Exception:
                    failed_doc_ids = []
                logger.warning(
                    "[InteractionExtraction] failed after retries: count=%s docs=%s",
                    len(failed_indices),
                    failed_doc_ids,
                )

        flat.sort(
            key=lambda x: (
                safe_str(x.get("document_id")),
                safe_str(x.get("chunk_id")),
                safe_str(x.get("subject_name")),
                safe_str(x.get("object_name")),
                safe_str(x.get("interaction_type")),
            )
        )

        dump_json(
            interaction_json_path,
            {
                "doc_type_meta": self._interaction_doc_type_meta(),
                "documents": per_doc,
                "interactions": flat,
            },
        )
        dump_json(interaction_list_json_path, flat)

        if verbose:
            ok_docs = sum(1 for v in per_doc.values() if isinstance(v, dict) and v.get("ok") is True)
            fail_docs = sum(1 for v in per_doc.values() if isinstance(v, dict) and v.get("ok") is not True)
            logger.info(
                "[InteractionExtraction] done. docs_ok=%s docs_failed=%s interactions=%s json_out=%s list_out=%s",
                ok_docs,
                fail_docs,
                len(flat),
                interaction_json_path,
                interaction_list_json_path,
            )
        return {
            "interaction_json_path": interaction_json_path,
            "interaction_list_json_path": interaction_list_json_path,
            "documents": per_doc,
            "interactions": flat,
        }

    def store_interactions_to_sql(
        self,
        *,
        interaction_list_json_path: Optional[str] = None,
        sql_db_path: Optional[str] = None,
        sql_table_name: str = "Interaction_info",
        include_auto_id: bool = True,
        reset_database: bool = True,
        reset_table: bool = True,
        verbose: bool = True,
    ) -> int:
        """
        Step-2 only: load interaction list JSON from disk and persist to SQL.

        This method does not run extraction; it only consumes the dumped list file.
        """
        base = self._interaction_base_dir()
        if interaction_list_json_path is None:
            interaction_list_json_path = os.path.join(base, "interaction_records_list.json")
        if sql_db_path is None:
            store = SQLStore(self.config)
        else:
            store = SQLStore(self.config, db_path=sql_db_path)

        # Requirement: SQL table should not include chunk_id / section_id columns.
        records = store.load_json_list(json_path=interaction_list_json_path)
        normalized: List[Dict[str, Any]] = []
        for r in records:
            if not isinstance(r, dict):
                continue
            rr = dict(r)
            rr.pop("chunk_id", None)
            rr.pop("section_id", None)
            normalized.append(rr)
        inserted = store.insert_records(
            table_name=sql_table_name,
            records=normalized,
            include_auto_id=bool(include_auto_id),
            reset_database=bool(reset_database),
            reset_table=bool(reset_table),
        )
        if verbose:
            logger.info(
                "[InteractionSQL] saved rows=%s from=%s table=%s db=%s",
                inserted,
                interaction_list_json_path,
                sql_table_name,
                store.db_path,
            )
        return inserted

    # -------------------------
    # Local graph saving helpers
    # -------------------------
    def _sanitize_doc_node_id(self, source_doc_id: str) -> str:
        s = safe_str(source_doc_id).strip()
        if not s:
            return ""
        s = re.sub(r"[^A-Za-z0-9_]+", "_", s).strip("_")
        return s

    def _create_document_entities_from_index(self) -> Dict[str, Dict[str, Any]]:
        base = self._base_dir()
        doc2chunks_path = os.path.join(base, "doc2chunks.json")
        if not os.path.exists(doc2chunks_path):
            raise FileNotFoundError(f"doc2chunks.json not found at {doc2chunks_path}. Run prepare_chunks() first.")

        doc2chunks = load_json(doc2chunks_path)
        if not isinstance(doc2chunks, dict):
            doc2chunks = {}

        meta = self.meta
        groups: Dict[str, Dict[str, Any]] = {}

        for part_id, pack in doc2chunks.items():
            if not isinstance(part_id, str) or not part_id.strip():
                continue
            if not isinstance(pack, dict):
                continue

            md = pack.get("document_metadata") or {}
            if not isinstance(md, dict):
                md = {}

            raw_doc_id = safe_str(md.get("raw_doc_id")).strip()
            doc_segment_id = safe_str(md.get("doc_segment_id")).strip()
            source_doc_id = safe_str(md.get("source_doc_id")).strip()
            generated_segment_title = bool(md.get("generated_segment_title"))
            original_title = safe_str(md.get("original_title")).strip()
            current_title = safe_str(md.get(meta.get("title", "title"))).strip()
            use_segment_anchor = bool(
                doc_segment_id
                and (
                    generated_segment_title
                    or (
                        raw_doc_id
                        and doc_segment_id != raw_doc_id
                        and original_title
                        and current_title
                        and current_title != original_title
                    )
                )
            )
            anchor_id = (doc_segment_id if use_segment_anchor else raw_doc_id) or doc_segment_id or source_doc_id
            if not anchor_id:
                continue

            title_key = meta.get("title", "title")
            subtitle_key = meta.get("subtitle", "subtitle")

            main_title = safe_str(md.get(title_key)).strip()
            sub_title = safe_str(md.get(subtitle_key)).strip()

            fallback_title = safe_str(md.get("source_title") or md.get("doc_title") or md.get("title")).strip()
            if not main_title:
                main_title = fallback_title

            g = groups.setdefault(
                anchor_id,
                {
                    "parts": [],
                    "title": main_title,
                    "subtitle": sub_title,
                    "metadata": md,
                    "doc_segment_id": doc_segment_id,
                    "source_doc_id": source_doc_id,
                },
            )

            g["parts"].append(part_id)

            if (not g.get("title")) and main_title:
                g["title"] = main_title
            if (not g.get("subtitle")) and sub_title:
                g["subtitle"] = sub_title
            if (not safe_str(g.get("doc_segment_id")).strip()) and doc_segment_id:
                g["doc_segment_id"] = doc_segment_id
            if (not safe_str(g.get("source_doc_id")).strip()) and source_doc_id:
                g["source_doc_id"] = source_doc_id

        documents: Dict[str, Dict[str, Any]] = {}

        for anchor_id, g in groups.items():
            doc_id = self._sanitize_doc_node_id(anchor_id)
            if not doc_id:
                continue

            parts = g.get("parts") or []
            if not isinstance(parts, list) or not parts:
                continue

            documents[doc_id] = {
                "id": doc_id,
                "type": meta.get("section_label", "Document"),
                "document_type": safe_str(self.doc_type) or "document",
                "doc_segment_id": safe_str(g.get("doc_segment_id")).strip() or anchor_id,
                "source_doc_id": (
                    safe_str(g.get("source_doc_id")).strip()
                    or safe_str(g.get("doc_segment_id")).strip()
                    or anchor_id
                ),
                "title": safe_str(g.get("title")).strip(),
                "subtitle": safe_str(g.get("subtitle")).strip(),
                "metadata": g.get("metadata") or {},
                "parts": sorted([p for p in parts if isinstance(p, str) and p.strip()]),
            }

        return documents

    def build_doc_entities(
        self,
        *,
        entity_info_path: Optional[str] = None,
        doc2chunks_path: Optional[str] = None,
        verbose: bool = True,
    ) -> None:
        """
        Build document-level supernode entities and their edges to regular entities,
        then persist both to JSON files:
          - doc_entities.json      : list of doc Entity dicts
          - doc_entity_edges.json  : list of CONTAINS Relation dicts

        Must be called after postprocess_and_save() so that entity_info_refined.json exists.
        """
        base = self._base_dir()

        if entity_info_path is None:
            p_refined = os.path.join(base, "entity_info_refined.json")
            entity_info_path = p_refined if os.path.exists(p_refined) else os.path.join(base, "entity_basic_info.json")

        if doc2chunks_path is None:
            doc2chunks_path = os.path.join(base, "doc2chunks.json")

        entities_by_id = load_json(entity_info_path)
        doc2chunks = load_json(doc2chunks_path)

        if not isinstance(entities_by_id, dict):
            entities_by_id = {}
        if not isinstance(doc2chunks, dict):
            doc2chunks = {}

        meta = self.meta
        section_label = meta.get("section_label", "Document")
        contains_pred = meta.get("contains_pred", "CONTAINS")
        title_key = meta.get("title", "title")
        subtitle_key = meta.get("subtitle", "subtitle")

        documents_map = self._create_document_entities_from_index()

        part2doc: Dict[str, str] = {}
        doc_entity_dicts: List[Dict[str, Any]] = []

        def _first_part_metadata(part_id: str) -> Dict[str, Any]:
            pack = doc2chunks.get(part_id, {}) or {}
            md = pack.get("document_metadata") or {}
            return md if isinstance(md, dict) else {}

        def _guess_description(md: Dict[str, Any]) -> str:
            for k in ["summary", "description", "logline", "synopsis"]:
                v = md.get(k)
                if isinstance(v, str) and v.strip():
                    return v.strip()
            return ""

        def _clean_section_metadata(md: Dict[str, Any]) -> Dict[str, Any]:
            """
            Keep section metadata concise for section supernodes.
            Remove preprocessing/internal fields that are noisy for retrieval.
            """
            if not isinstance(md, dict):
                return {}
            drop_keys = {
                "doc_title",
                "source_doc_id",
                "raw_doc_id",
            }
            out: Dict[str, Any] = {}
            for k, v in md.items():
                kk = str(k).strip()
                if not kk or kk in drop_keys:
                    continue
                if v is None:
                    continue
                if isinstance(v, str) and not v.strip():
                    continue
                out[kk] = v
            return out

        for doc_group_id, d in documents_map.items():
            if not isinstance(d, dict):
                continue

            parts = d.get("parts") or []
            if not isinstance(parts, list) or not parts:
                continue

            parts_clean = [p.strip() for p in parts if isinstance(p, str) and p.strip()]
            if not parts_clean:
                continue

            for p in parts_clean:
                part2doc[p] = doc_group_id

            md = _first_part_metadata(parts_clean[0])

            main_title = safe_str(d.get("title")).strip()
            sub_title = safe_str(d.get("subtitle")).strip()

            properties: Dict[str, Any] = {
                "document_type": safe_str(self.doc_type) or "document"
            }

            cleaned_md = _clean_section_metadata(md) if isinstance(md, dict) and md else {}
            md_title = safe_str(cleaned_md.get("title")).strip()
            md_subtitle = safe_str(cleaned_md.get("subtitle")).strip()

            resolved_title = main_title or md_title
            resolved_subtitle = sub_title or md_subtitle
            if resolved_title:
                properties[title_key] = resolved_title
            if resolved_subtitle:
                properties[subtitle_key] = resolved_subtitle

            # flatten cleaned metadata into properties (no nested "metadata")
            # keep doc_type-mapped title/subtitle keys as canonical keys
            for mk, mv in cleaned_md.items():
                if mk in {"title", "subtitle", "document_type"}:
                    continue
                if mk in {title_key, subtitle_key}:
                    continue
                properties[mk] = mv

            doc_name = main_title or properties.get(title_key) or doc_group_id
            doc_description = _guess_description(md)

            doc_entity_dicts.append(
                {
                    "id": doc_group_id,
                    "name": doc_name,
                    "type": [section_label],
                    "aliases": [],
                    "description": doc_description,
                    "scope": "global",
                    "source_documents": sorted(set(parts_clean)),
                    "properties": properties,
                }
            )

        # build Document -> Entity edges via part intersection
        doc2parts: Dict[str, Set[str]] = {
            d["id"]: set(d["source_documents"])
            for d in doc_entity_dicts
            if d.get("id") and isinstance(d.get("source_documents"), list)
        }

        def _stable_rel_id(key: str, prefix: str = "rel_") -> str:
            return prefix + hashlib.md5(key.encode("utf-8")).hexdigest()[:12]

        doc_entity_edges: List[Dict[str, Any]] = []

        for eid, ent in entities_by_id.items():
            if not isinstance(ent, dict):
                continue
            ent_id = safe_str(ent.get("id")) or safe_str(eid)
            if not ent_id:
                continue

            src_docs = ent.get("source_documents") or []
            if not isinstance(src_docs, list):
                src_docs = []
            src_docs_set = set([x for x in src_docs if isinstance(x, str) and x.strip()])
            if not src_docs_set:
                continue

            touched_docs: Set[str] = set()
            for p in src_docs_set:
                dg = part2doc.get(p)
                if dg:
                    touched_docs.add(dg)

            for dg in sorted(touched_docs):
                parts = doc2parts.get(dg, set())
                if not parts:
                    continue
                inter = parts.intersection(src_docs_set)
                if not inter:
                    continue

                rid = _stable_rel_id(f"{dg}||{contains_pred}||{ent_id}", "rel_doc_ent_")
                doc_entity_edges.append(
                    {
                        "id": rid,
                        "subject_id": dg,
                        "object_id": ent_id,
                        "predicate": contains_pred,
                        "relation_name": "contains",
                        "persistence": "static",
                        "confidence": 1.0,
                        "description": f"{section_label} contains mentions of the entity.",
                        "source_documents": sorted(list(inter)),
                        "properties": {},
                    }
                )

        # persist to JSON
        doc_entities_path = os.path.join(base, "doc_entities.json")
        doc_entity_edges_path = os.path.join(base, "doc_entity_edges.json")
        dump_json(doc_entities_path, doc_entity_dicts)
        dump_json(doc_entity_edges_path, doc_entity_edges)

        if verbose:
            logger.info(
                f"[DocEntities] doc_label={section_label} contains_pred={contains_pred} "
                f"doc_entities={len(doc_entity_dicts)} doc_entity_edges={len(doc_entity_edges)} "
                f"saved to {doc_entities_path} and {doc_entity_edges_path}"
            )

    def load_json_to_graph_store(
        self,
        *,
        verbose: bool = True,
        entity_info_path: Optional[str] = None,
        relation_info_path: Optional[str] = None,
        doc_entities_path: Optional[str] = None,
        doc_entity_edges_path: Optional[str] = None,
    ) -> None:
        """
        Load all JSON files into the local runtime graph store,
        then build embeddings and derived graph metrics.

        Must be called after build_doc_entities() so that doc_entities.json and
        doc_entity_edges.json exist.
        """
        base = self._base_dir()

        if entity_info_path is None:
            p_refined = os.path.join(base, "entity_info_refined.json")
            entity_info_path = p_refined if os.path.exists(p_refined) else os.path.join(base, "entity_basic_info.json")

        if relation_info_path is None:
            relation_info_path = os.path.join(base, "relation_info_refined.json")

        if doc_entities_path is None:
            doc_entities_path = os.path.join(base, "doc_entities.json")

        if doc_entity_edges_path is None:
            doc_entity_edges_path = os.path.join(base, "doc_entity_edges.json")

        entities_by_id = load_json(entity_info_path)
        relations = load_json(relation_info_path)
        doc_entity_dicts = load_json(doc_entities_path)
        doc_entity_edges = load_json(doc_entity_edges_path)

        if not isinstance(entities_by_id, dict):
            entities_by_id = {}
        if not isinstance(relations, list):
            relations = []
        if not isinstance(doc_entity_dicts, list):
            doc_entity_dicts = []
        if not isinstance(doc_entity_edges, list):
            doc_entity_edges = []

        if verbose:
            logger.info(
                f"[GraphLoad] entities={len(entities_by_id)} relations={len(relations)} "
                f"doc_entities={len(doc_entity_dicts)} doc_entity_edges={len(doc_entity_edges)}"
            )

        self.graph_store.reset_knowledge_graph()

        entity_objs: List[Entity] = []
        relation_objs: List[Relation] = []

        for d in doc_entity_dicts:
            if not isinstance(d, dict):
                continue
            try:
                entity_objs.append(Entity(**d))
            except Exception:
                continue

        for ent in entities_by_id.values():
            if not isinstance(ent, dict):
                continue
            try:
                entity_objs.append(Entity(**ent))
            except Exception:
                continue

        for r in relations:
            if not isinstance(r, dict):
                continue
            row = dict(r)
            if "predicate" not in row and "relation_type" in row:
                row["predicate"] = row.get("relation_type")
            if "source_documents" not in row:
                did = row.get("document_id")
                row["source_documents"] = [did.strip()] if isinstance(did, str) and did.strip() else []
            if "confidence" not in row or row.get("confidence") is None:
                row["confidence"] = 1.0
            if "properties" not in row or not isinstance(row.get("properties"), dict):
                row["properties"] = {}
            try:
                relation_objs.append(Relation(**row))
            except Exception:
                continue

        for r in doc_entity_edges:
            if not isinstance(r, dict):
                continue
            try:
                relation_objs.append(Relation(**r))
            except Exception:
                continue

        self.graph_store.upsert_entities(entity_objs)
        self.graph_store.upsert_relations(relation_objs)

        self.graph_query_utils.load_embedding_model(self.config.embedding)
        self.graph_query_utils.create_vector_index()
        self.graph_query_utils.process_all_embeddings()
        self.graph_query_utils.ensure_entity_superlabel()
        try:
            cstats = self.graph_query_utils.compute_centrality(
                graph_name="centrality_graph",
                force_refresh=True,
                include_betweenness=True,
            )
            if verbose:
                logger.info("[GraphCentrality] %s", cstats)
        except Exception as e:
            logger.warning("Graph centrality computation failed (skipped): %s", e)
        logger.info("Local graph runtime prepared successfully.")

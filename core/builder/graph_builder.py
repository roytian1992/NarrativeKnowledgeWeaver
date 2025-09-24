# core/builder/graph_builder.py
from __future__ import annotations

import json
import os
import sqlite3
import pickle
import multiprocessing as mp
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError, wait, FIRST_COMPLETED
from collections import defaultdict
from copy import deepcopy
from typing import Any, Dict, List, Optional
import asyncio
import pandas as pd
import random
import re
import glob
import logging
from tqdm import tqdm

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
        """Delete all .json files under a directory."""
        for file in glob.glob(os.path.join(path, "*.json")):
            try:
                os.remove(file)
            except Exception as e:
                logger.warning(f"Failed to delete: {file} -> {e}")

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
            else:
                settings = {"background": "", "abbreviations": []}
        else:
            schema = {}
            settings = {"background": "", "abbreviations": []}

        if self.probing_mode != "fixed":
            schema, settings = self.update_schema(
                schema,
                background=settings["background"],
                abbreviations=settings["abbreviations"],
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
    def extract_entity_and_relation(self, verbose: bool = True):
        return asyncio.run(self.extract_entity_and_relation_async(verbose=verbose))

    async def extract_entity_and_relation_async(self, verbose: bool = True):
        """
        Concurrent extraction with a unified retry round; results persisted to disk.
        """
        base = self.config.storage.knowledge_graph_path
        desc_chunks = [TextChunk(**o) for o in
                       json.load(open(os.path.join(base, "all_document_chunks.json"), "r", encoding="utf-8"))]

        if verbose:
            logger.info("Asynchronously extracting entities and relations...")

        sem = asyncio.Semaphore(self.max_workers)

        async def _arun_once(ch: TextChunk):
            async with sem:
                try:
                    if not ch.content.strip():
                        result = {"entities": [], "relations": []}
                    else:
                        result = await self.information_extraction_agent.arun(
                            ch.content,
                            timeout=self.config.agent.async_timeout,
                            max_attempts=self.config.agent.async_max_attempts,
                            backoff_seconds=self.config.agent.async_backoff_seconds,
                        )
                    result.update(chunk_id=ch.id, chunk_metadata=ch.metadata)
                    return result
                except Exception as e:
                    if verbose:
                        logger.error(f"Extraction failed: chunk_id={ch.id} | {e.__class__.__name__}: {e}")
                    return {
                        "chunk_id": ch.id,
                        "chunk_metadata": ch.metadata,
                        "entities": [],
                        "relations": [],
                        "error": f"{e.__class__.__name__}: {e}",
                    }

        async def _arun_with_ch(ch: TextChunk):
            """Return (chunk, result) to record failures precisely."""
            res = await _arun_once(ch)
            return ch, res

        # First round
        tasks = [_arun_with_ch(ch) for ch in desc_chunks]
        first_round_pairs = []
        failed_chs = []

        for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Async extraction"):
            ch, res = await coro
            first_round_pairs.append((ch, res))
            if res.get("error"):
                failed_chs.append(ch)

        # Unified single retry
        retry_pairs = []
        if failed_chs:
            if verbose:
                logger.info(f"Retrying {len(failed_chs)} failed chunks...")
            retry_tasks = [_arun_with_ch(ch) for ch in failed_chs]
            for coro in tqdm(asyncio.as_completed(retry_tasks), total=len(retry_tasks), desc="Retry extraction"):
                ch, res = await coro
                retry_pairs.append((ch, res))

        # Merge results (replace failed with retries)
        failed_ids = {ch.id for ch in failed_chs}
        final_results = [res for ch, res in first_round_pairs if ch.id not in failed_ids]
        final_results += [res for _, res in retry_pairs]

        # Persist
        output_path = os.path.join(base, "extraction_results.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(final_results, f, ensure_ascii=False, indent=2)

        if verbose:
            still_failed = sum(1 for r in final_results if r.get("error"))
            logger.info(f"Entity & relation extraction finished. Total chunks: {len(final_results)}")
            if still_failed:
                logger.warning(f"{still_failed} chunks still failed after retry (kept 'error' for debugging).")
            logger.info(f"Saved to: {output_path}")

    # ═════════════════════════════════════════════════════════════════════
    #  4) Attribute extraction & refinement
    # ═════════════════════════════════════════════════════════════════════
    def run_extraction_refinement(self, verbose=False):
        """
        Refinement pipeline:
          - Remove noisy entities/relations by keyword
          - Refine entity types
          - Refine entity scope
          - Entity disambiguation
        """
        # Chinese keywords commonly appearing as non-entities in screenplays
        KW = ("闪回", "一组蒙太奇")

        base = self.config.storage.knowledge_graph_path
        extraction_results = json.load(open(os.path.join(base, "extraction_results.json"), "r", encoding="utf-8"))
        for doc in extraction_results:
            # 1) Remove entities with keyword in name
            ents = doc.get("entities", [])
            removed_names = {e.get("name", "") for e in ents if any(k in e.get("name", "") for k in KW)}
            doc["entities"] = [e for e in ents if e.get("name", "") not in removed_names]

            # 2) Remove relations whose subject/object include keywords or point to removed entities
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

    async def extract_entity_attributes_async(self, verbose: bool = True) -> Dict[str, Entity]:
        """
        Asynchronous attribute extraction for merged entities.

        - Merge entities from extraction results
        - For each entity, call AttributeExtractionAgent.arun()
        - arun() has built-in timeout and retry; won't hang indefinitely
        """
        base = self.config.storage.knowledge_graph_path
        results = json.load(open(os.path.join(base, "extraction_results_refined.json"), "r", encoding="utf-8"))

        # Merge / deduplicate entities
        entity_map = self.merge_entities_info(results)  # {name: Entity}

        if verbose:
            logger.info(f"Starting async attribute extraction, #entities: {len(entity_map)}")

        sem = asyncio.Semaphore(self.max_workers)
        updated_entities: Dict[str, Entity] = {}

        async def _arun_attr(name: str, ent: Entity):
            async with sem:
                try:
                    txt = ent.description or ""
                    if not txt.strip():
                        return name, None

                    # AttributeExtractionAgent.arun has timeout+retry internally
                    res = await self.attribute_extraction_agent.arun(
                        text=txt,
                        entity_name=name,
                        entity_type=ent.type,
                        source_chunks=ent.source_chunks,
                        original_text="",
                        timeout=self.config.agent.async_timeout,
                        max_attempts=self.config.agent.async_max_attempts,
                        backoff_seconds=self.config.agent.async_backoff_seconds,
                    )

                    if res.get("error"):
                        return name, None

                    attrs = res.get("attributes", {}) or {}
                    if isinstance(attrs, str):
                        try:
                            attrs = json.loads(attrs)
                        except json.JSONDecodeError:
                            attrs = {}

                    new_ent = deepcopy(ent)
                    new_ent.properties = attrs

                    nd = res.get("new_description", "")
                    if nd:
                        new_ent.description = nd

                    return name, new_ent
                except Exception as e:
                    if verbose:
                        logger.error(f"Attribute extraction failed (async): {name}: {e}")
                    return name, None

        # Concurrent execution
        tasks = [_arun_attr(n, e) for n, e in entity_map.items()]
        for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Attribute extraction (async)"):
            n, e2 = await coro
            if e2:
                updated_entities[n] = e2

        # Persist
        output_path = os.path.join(base, "entity_info.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({k: v.dict() for k, v in updated_entities.items()}, f, ensure_ascii=False, indent=2)

        if verbose:
            logger.info(f"Attribute extraction finished. Processed entities: {len(updated_entities)}")
            logger.info(f"Saved to: {output_path}")

        return {
            "status": "success",
            "processed_entities": len(updated_entities),
            "output_path": output_path,
        }

    # ═════════════════════════════════════════════════════════════════════
    #  5) Build and store the graph
    # ═════════════════════════════════════════════════════════════════════
    def build_graph_from_results(self, verbose: bool = True) -> KnowledgeGraph:
        if verbose:
            logger.info("Loading refined extraction results and entity info...")

        base = self.config.storage.knowledge_graph_path
        results = json.load(open(os.path.join(base, "extraction_results_refined.json"), "r", encoding="utf-8"))
        ent_raw = json.load(open(os.path.join(base, "entity_info.json"), "r", encoding="utf-8"))

        with open(os.path.join(base, "section_entities_collection.pkl"), "rb") as f:
            self.section_entities_collection = pickle.load(f)

        # id -> Entity
        entity_map = {d["id"]: Entity(**d) for d in ent_raw.values()}
        name2id: Dict[str, str] = {e.name: e.id for e in entity_map.values()}

        for e in entity_map.values():
            for al in e.aliases:
                name2id.setdefault(al, e.id)
            self.kg.add_entity(e)

        if verbose:
            logger.info("Building knowledge graph...")

        self.section_names = []
        for res in results:
            md = res.get("chunk_metadata", {})

            # Section entities
            secs = self._create_section_entities(md, res["chunk_id"])
            for se in secs:
                if se.name not in self.section_names and se.id not in self.kg.entities:
                    self.kg.add_entity(se)
                    self.section_names.append(se.name)
                else:
                    exist = self.kg.entities.get(se.id)
                    if exist:
                        merged = list(dict.fromkeys(list(exist.source_chunks) + list(se.source_chunks)))
                        exist.source_chunks = merged

            inner = self.section_entities_collection[se.name]
            for se in secs:
                self._link_section_to_entities(se, inner, res["chunk_id"])

            # Ordinary relations
            for rdata in res.get("relations", []):
                rel = self._create_relation_from_data(rdata, res["chunk_id"], entity_map, name2id)
                if rel:
                    self.kg.add_relation(rel)

        # Persist into DBs
        if verbose:
            logger.info("Persisting graph to databases...")
        self._store_knowledge_graph(verbose)
        if verbose:
            logger.info("Enriching event nodes and computing graph metrics...")
        self.neo4j_utils.enrich_event_nodes_with_context()
        self.neo4j_utils.compute_centrality(exclude_rel_types=[self.meta['contains_pred']])

        if verbose:
            st = self.kg.stats()
            graph_stats = self.graph_store.get_stats()
            logger.info("Knowledge graph construction completed.")
            logger.info(f"  - Entities: {graph_stats['entities']}")
            logger.info(f"  - Relations: {graph_stats['relations']}")
            logger.info(f"  - Documents: {st['documents']}")
            logger.info(f"  - Chunks: {st['chunks']}")

        return self.kg

    # ═════════════════════════════════════════════════════════════════════
    #  Internal utilities
    # ═════════════════════════════════════════════════════════════════════
    def merge_entities_info(self, extraction_results):
        """
        Merge/deduplicate entities across chunks.

        - Entities with local scope that collide in name may be renamed with a suffix
          when they appear in different sections.
        - Section numbering prefers chunk_metadata.order; otherwise falls back to title.
        """
        entity_map: Dict[str, Entity] = {}
        self.chunk2section_map = {result["chunk_id"]: result["chunk_metadata"]["doc_title"] for result in extraction_results}
        self.section_entities_collection = dict()

        base = self.config.storage.knowledge_graph_path

        for i, result in enumerate(extraction_results):
            md = result.get("chunk_metadata", {}) or {}
            label = md.get("doc_title", md.get("subtitle", md.get("title", "")))
            if label not in self.section_entities_collection:
                self.section_entities_collection[label] = []

            # Entities from current chunk
            for ent_data in result.get("entities", []):
                t = ent_data.get("type", "")
                is_event = (t == "Event") or (isinstance(t, list) and "Event" in t)
                is_action_like = (
                    (isinstance(t, str) and t in ["Action", "Emotion", "Goal"]) or
                    (isinstance(t, list) and any(x in ["Action", "Emotion", "Goal"] for x in t))
                )
                if is_event:
                    is_action_like = False

                # Rename local/action-like entities if they collide across different sections
                if (ent_data.get("scope", "").lower() == "local" or is_action_like) and ent_data["name"] in entity_map:
                    existing_entity = entity_map[ent_data["name"]]
                    existing_chunk_id = existing_entity.source_chunks[0]
                    existing_section_name = self.chunk2section_map[existing_chunk_id]
                    current_section_name = md["doc_title"]
                    suffix = 1
                    if current_section_name != existing_section_name:
                        new_name = f"{ent_data['name']}_in_{suffix}"
                        while new_name in entity_map:
                            suffix += 1
                            new_name = f"{ent_data['name']}_{suffix}"
                        ent_data["name"] = new_name

                # Create / merge
                ent_obj = self._create_entity_from_data(ent_data, result["chunk_id"])
                existing = self._find_existing_entity(ent_obj, entity_map)
                if existing:
                    self._merge_entities(existing, ent_obj)
                else:
                    entity_map[ent_obj.name] = ent_obj
                self.section_entities_collection[label].append(ent_obj)

        output_path = os.path.join(base, "section_entities_collection.pkl")
        with open(output_path, "wb") as f:
            pickle.dump(self.section_entities_collection, f)

        return entity_map

    def _find_existing_entity(self, entity: Entity, entity_map: Dict[str, Entity]) -> Optional[Entity]:
        """Find an existing entity that should be merged with the incoming one (non-Event only)."""
        if (entity.type == "Event") or (isinstance(entity.type, list) and "Event" in entity.type):
            return None
        if entity.name in entity_map:
            return entity_map[entity.name]
        for existing_entity in entity_map.values():
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

    def _link_section_to_entities(self, section: Entity, inners: List[Entity], chunk_id: str):
        pred = self.meta["contains_pred"]
        for tgt in inners:
            rid = f"rel_{hash(f'{section.id}_{pred}_{tgt.id}') % 1_000_000}"
            self.kg.add_relation(
                Relation(id=rid, subject_id=section.id, predicate=pred,
                         object_id=tgt.id, properties={}, source_chunks=[chunk_id])
            )

    # -------- Entity / Relation creation --------
    @staticmethod
    def _create_entity_from_data(data: Dict, chunk_id: str) -> Entity:
        return Entity(
            id=f"ent_{hash(data['name']) % 1_000_000}",
            name=data["name"],
            type=_normalize_type(data.get("type", "Concept")),
            scope=data.get("scope", "local"),
            description=data.get("description", ""),
            aliases=data.get("aliases", []),
            source_chunks=[chunk_id],
        )

    @staticmethod
    def _create_relation_from_data(
        d: Dict, chunk_id: str, entity_map: Dict[str, Entity], name2id: Dict[str, str]
    ) -> Optional[Relation]:
        subj = d.get("subject") or d.get("source") or d.get("head") or d.get("relation_subject")
        obj = d.get("object") or d.get("target") or d.get("tail") or d.get("relation_object")
        pred = d.get("predicate") or d.get("relation") or d.get("relation_type")
        if not subj or not obj or not pred:
            return None
        sid, oid = name2id.get(subj), name2id.get(obj)
        if not sid or not oid:
            return None
        rid = f"rel_{hash(f'{sid}_{pred}_{oid}') % 1_000_000}"

        return Relation(
            id=rid,
            subject_id=sid,
            predicate=pred,
            object_id=oid,
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

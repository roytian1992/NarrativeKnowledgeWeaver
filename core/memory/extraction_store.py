"""
core/memory/extraction_store.py

Persistent extraction memory store for knowledge-graph extraction pipeline.

Architecture
------------
- In-memory hot cache (dict[id → entry])
- Keyword name index (dict[surface_form_lower → set[id]])
- ChromaDB collection for persistent vector storage

Memory types
------------
  alias       : "'陈所长' is an alias for '陈建国' (Character)"
  naming      : "'北京市公安局' should be referred to as '公安局'"
  type_rule   : "'东方红' typed Organization, not Location: <reason>"
  scope_rule  : "'大决战' scope=global: recurring across multiple scenes"
  dedup_rule  : "A→B: 'precedes' is redundant given 'causes'"
  term        : "'三产' = '第三产业' in this domain"

Retrieval strategy
------------------
For each incoming text_chunk:
  Pass A (precise): extract surface keywords via YAKE/jieba → keyword index
  Pass B (semantic, conditional): if Pass-A hits < min_kw_hits:
      embed " ".join(extracted_keywords) → ChromaDB query
  Merge → rank by confidence × usage_count(hit_count) → budget cap
"""
from __future__ import annotations

import json
import logging
import math
import os
import time
import uuid
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

from core.memory.base_memory import BaseMemoryStore
from core.utils.general_utils import extract_keywords

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MEMORY_TYPES = frozenset(
    {
        "alias",
        "naming",
        "type_rule",
        "scope_rule",
        "dedup_rule",
        "term",
        "guideline",
        "anti_pattern",
        "canonical_map",
        "guideline_summary",
        "anti_pattern_summary",
        "canonical_map_summary",
    }
)
DERIVED_MEMORY_TYPES = frozenset(
    {
        "guideline",
        "anti_pattern",
        "canonical_map",
        "guideline_summary",
        "anti_pattern_summary",
        "canonical_map_summary",
    }
)
SUMMARY_MEMORY_TYPES = frozenset({"guideline_summary", "anti_pattern_summary", "canonical_map_summary"})
DEFAULT_MEMORY_SCOPE = "shared"
ALLOWED_MEMORY_SCOPES = frozenset({"entity_extraction", "relation_extraction", "shared"})

# ---------------------------------------------------------------------------
# Ranking helper
# ---------------------------------------------------------------------------

def _rank_score(entry: Dict[str, Any], now: float) -> float:
    conf = float(entry.get("confidence", 0.5))
    hits = int(entry.get("hit_count", 0))
    # Count-driven ranking: avoid wall-clock decay for distilled memories.
    # More frequently useful memories are promoted via hit_count.
    return conf * (1.0 + 0.35 * math.log1p(max(0, hits)))


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class ExtractionMemoryStore(BaseMemoryStore):
    """
    Persistent extraction memory for knowledge-graph extraction pipeline.

    Parameters
    ----------
    store_path : str
        Path to the ChromaDB persist directory (e.g. data/memory/raw_memory).
    embedding_model :
        An object that implements embed_query(str) → List[float] and
        embed_documents(List[str]) → List[List[float]].
        Typically OpenAIEmbeddingModel(config.embedding).
    max_context_entries : int
        Maximum number of memory entries returned by query(). Default 12.
    max_chars_per_entry : int
        Maximum characters per entry string in the formatted output. Default 120.
    similarity_threshold : float
        Cosine similarity above which two entries are considered duplicates
        and merged instead of inserted. Default 0.92.
    flush_every_n_entries : int
        Auto-flush when dirty entries accumulate to this count. Default 20.
    flush_every_n_docs : int
        Auto-flush every N documents processed. Default 10.
    min_kw_hits : int
        Minimum keyword hits before skipping vector search. Default 3.
    """

    COLLECTION_NAME = "extraction_memories"
    BASE_META_KEYS = {"type", "confidence", "hit_count", "source", "created_at", "updated_at", "keywords", "memory_scope"}

    def __init__(
        self,
        store_path: str,
        embedding_model: Any,
        *,
        max_context_entries: int = 12,
        max_chars_per_entry: int = 120,
        similarity_threshold: float = 0.92,
        flush_every_n_entries: int = 20,
        flush_every_n_docs: int = 10,
        min_kw_hits: int = 3,
        max_raw_context_entries: int = 5,
        realtime_store_path: str = "",
    ) -> None:
        self.store_path = store_path
        self.embedding_model = embedding_model
        self.max_context_entries = max_context_entries
        self.max_chars_per_entry = max_chars_per_entry
        self.similarity_threshold = similarity_threshold
        self.flush_every_n_entries = flush_every_n_entries
        self.flush_every_n_docs = flush_every_n_docs
        self.min_kw_hits = min_kw_hits
        self.max_raw_context_entries = max(0, int(max_raw_context_entries))
        self.realtime_store_path = str(realtime_store_path or "").strip()
        self.realtime_raw_path = ""
        self.realtime_distilled_path = ""
        if self.realtime_store_path:
            os.makedirs(self.realtime_store_path, exist_ok=True)
            self.realtime_raw_path = os.path.join(self.realtime_store_path, "raw_memories_realtime.jsonl")
            self.realtime_distilled_path = os.path.join(
                self.realtime_store_path, "distilled_memories_realtime.jsonl"
            )

        # In-memory cache
        self._cache: Dict[str, Dict[str, Any]] = {}
        # surface_form_lower → set of entry ids
        self._name_index: Dict[str, Set[str]] = defaultdict(set)
        # dirty entry ids waiting to be flushed
        self._dirty: Set[str] = set()
        self._docs_since_flush: int = 0

        # Init ChromaDB
        self._collection = None
        self._init_chroma()
        self.load_from_disk()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_chroma(self) -> None:
        try:
            import chromadb
            os.makedirs(self.store_path, exist_ok=True)
            client = chromadb.PersistentClient(path=self.store_path)
            self._collection = client.get_or_create_collection(
                name=self.COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
            logger.debug("ExtractionMemoryStore: ChromaDB collection ready at %s", self.store_path)
        except Exception as e:
            logger.warning("ExtractionMemoryStore: ChromaDB init failed (%s); running in-memory only.", e)
            self._collection = None

    def load_from_disk(self) -> None:
        """Populate hot cache and keyword index from ChromaDB."""
        if self._collection is None:
            return
        try:
            result = self._collection.get(include=["documents", "metadatas"])
            ids = result.get("ids") or []
            docs = result.get("documents") or []
            metas = result.get("metadatas") or []
            for eid, content, meta in zip(ids, docs, metas):
                if not isinstance(meta, dict):
                    meta = {}
                entry = dict(meta)
                entry["id"] = eid
                entry["content"] = content
                # restore list field (stored as comma-separated string)
                raw_kw = entry.get("keywords", "")
                if isinstance(raw_kw, str):
                    entry["keywords"] = [k for k in raw_kw.split("|||") if k]
                extra_json = entry.get("extra_json", "")
                if isinstance(extra_json, str) and extra_json.strip():
                    try:
                        extra = json.loads(extra_json)
                        if isinstance(extra, dict):
                            for k, v in extra.items():
                                if k not in entry:
                                    entry[k] = v
                    except Exception:
                        pass
                self._cache[eid] = entry
                self._index_entry(entry)
            logger.info("ExtractionMemoryStore: loaded %d entries from disk.", len(ids))
        except Exception as e:
            logger.warning("ExtractionMemoryStore: load_from_disk failed: %s", e)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _index_entry(self, entry: Dict[str, Any]) -> None:
        """Add entry's keywords to the name index."""
        eid = entry.get("id", "")
        if not eid:
            return
        for kw in entry.get("keywords") or []:
            if isinstance(kw, str) and kw.strip():
                self._name_index[kw.strip().lower()].add(eid)

    def _unindex_entry(self, entry: Dict[str, Any]) -> None:
        """Remove entry's keywords from the name index."""
        eid = entry.get("id", "")
        for kw in entry.get("keywords") or []:
            if isinstance(kw, str) and kw.strip():
                bucket = self._name_index.get(kw.strip().lower())
                if bucket:
                    bucket.discard(eid)

    def _embed(self, text: str) -> Optional[List[float]]:
        try:
            return self.embedding_model.embed_query(text)
        except Exception as e:
            logger.warning("ExtractionMemoryStore: embed failed: %s", e)
            return None

    def _cosine(self, a: List[float], b: List[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a)) + 1e-12
        nb = math.sqrt(sum(x * x for x in b)) + 1e-12
        return dot / (na * nb)

    def _find_duplicate(
        self, content: str, mem_type: str, vec: List[float]
    ) -> Optional[str]:
        """
        Check ChromaDB for a near-duplicate entry of the same type.
        Returns the id of the best match if similarity > threshold, else None.
        """
        if self._collection is None or vec is None:
            return None
        try:
            res = self._collection.query(
                query_embeddings=[vec],
                n_results=3,
                where={"type": mem_type},
                include=["distances", "metadatas"],
            )
            ids_list = (res.get("ids") or [[]])[0]
            dists = (res.get("distances") or [[]])[0]
            # ChromaDB cosine distance = 1 - cosine_similarity
            for eid, dist in zip(ids_list, dists):
                sim = 1.0 - float(dist)
                if sim >= self.similarity_threshold:
                    return eid
        except Exception as e:
            logger.debug("ExtractionMemoryStore._find_duplicate: %s", e)
        return None

    def _auto_flush(self) -> None:
        if (
            len(self._dirty) >= self.flush_every_n_entries
            or self._docs_since_flush >= self.flush_every_n_docs
        ):
            self.flush()

    def _append_realtime_event(self, entry: Dict[str, Any], *, action: str) -> None:
        if not self.realtime_store_path:
            return
        mem_type = str(entry.get("type", "")).strip()
        path = self.realtime_distilled_path if mem_type in DERIVED_MEMORY_TYPES else self.realtime_raw_path
        if not path:
            return
        event = {
            "event_time": time.time(),
            "action": str(action).strip() or "insert",
            "id": str(entry.get("id", "")),
            "type": mem_type,
            "memory_scope": str(entry.get("memory_scope", DEFAULT_MEMORY_SCOPE)),
            "source": str(entry.get("source", "")),
            "confidence": float(entry.get("confidence", 0.7)),
            "content": str(entry.get("content", "")).strip(),
            "keywords": list(entry.get("keywords") or []),
        }
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.debug("ExtractionMemoryStore._append_realtime_event failed: %s", e)

    # ------------------------------------------------------------------
    # Public API: CRUD
    # ------------------------------------------------------------------

    def add(self, entry: Dict[str, Any]) -> str:
        """
        Insert or merge a memory entry.

        Required fields in entry:
          type        : one of MEMORY_TYPES
          content     : natural-language description (will be embedded)
          keywords    : List[str] – surface forms for keyword index

        Optional:
          confidence  : float in [0, 1], default 0.7
          source      : str, e.g. "refine_entity_types"

        Returns the assigned entry id.
        """
        mem_type = str(entry.get("type", ""))
        if mem_type not in MEMORY_TYPES:
            logger.warning("ExtractionMemoryStore.add: unknown type %r, using 'term'", mem_type)
            mem_type = "term"

        content = str(entry.get("content", "")).strip()
        if not content:
            logger.debug("ExtractionMemoryStore.add: empty content, skipped.")
            return ""

        keywords: List[str] = [
            str(k).strip()
            for k in (entry.get("keywords") or [])
            if str(k).strip()
        ]
        confidence = float(entry.get("confidence", 0.7))
        source = str(entry.get("source", ""))
        memory_scope = str(entry.get("memory_scope", DEFAULT_MEMORY_SCOPE)).strip().lower()
        if memory_scope not in ALLOWED_MEMORY_SCOPES:
            memory_scope = DEFAULT_MEMORY_SCOPE
        now = time.time()

        # Embed content for dedup check
        vec = self._embed(content)

        # Dedup check
        dup_id = self._find_duplicate(content, mem_type, vec) if vec else None
        if dup_id and dup_id in self._cache:
            # Merge: update existing entry
            existing = self._cache[dup_id]
            # Keep longer content
            if len(content) > len(existing.get("content", "")):
                existing["content"] = content
            # Merge keywords
            existing_kw_set = set(existing.get("keywords") or [])
            for kw in keywords:
                if kw not in existing_kw_set:
                    existing_kw_set.add(kw)
                    self._name_index[kw.lower()].add(dup_id)
            existing["keywords"] = list(existing_kw_set)
            # Update confidence (take max)
            existing["confidence"] = max(float(existing.get("confidence", 0.0)), confidence)
            existing["memory_scope"] = memory_scope or str(existing.get("memory_scope", DEFAULT_MEMORY_SCOPE))
            existing["updated_at"] = now
            self._dirty.add(dup_id)
            self._append_realtime_event(existing, action="merge")
            self._auto_flush()
            logger.debug("ExtractionMemoryStore.add: merged with existing %s", dup_id)
            return dup_id

        # New entry
        eid = str(uuid.uuid4())
        new_entry: Dict[str, Any] = {
            "id": eid,
            "type": mem_type,
            "content": content,
            "keywords": keywords,
            "confidence": confidence,
            "hit_count": 0,
            "created_at": now,
            "updated_at": now,
            "source": source,
            "memory_scope": memory_scope,
        }
        self._cache[eid] = new_entry
        self._index_entry(new_entry)
        self._dirty.add(eid)
        self._append_realtime_event(new_entry, action="insert")
        self._auto_flush()
        logger.debug("ExtractionMemoryStore.add: inserted %s (type=%s)", eid, mem_type)
        return eid

    def update(self, entry_id: str, **fields: Any) -> None:
        """Update fields of an existing entry and mark as dirty."""
        entry = self._cache.get(entry_id)
        if not entry:
            logger.debug("ExtractionMemoryStore.update: id %s not found", entry_id)
            return
        # Handle keyword re-indexing
        if "keywords" in fields:
            self._unindex_entry(entry)
        entry.update(fields)
        entry["updated_at"] = time.time()
        if "keywords" in fields:
            self._index_entry(entry)
        self._dirty.add(entry_id)

    def delete(self, entry_id: str) -> None:
        """Remove an entry from cache, keyword index, and ChromaDB."""
        entry = self._cache.pop(entry_id, None)
        if entry:
            self._unindex_entry(entry)
        self._dirty.discard(entry_id)
        if self._collection is not None:
            try:
                self._collection.delete(ids=[entry_id])
            except Exception as e:
                logger.debug("ExtractionMemoryStore.delete ChromaDB error: %s", e)

    def flush(self) -> None:
        """Write all dirty entries to ChromaDB."""
        if not self._dirty or self._collection is None:
            self._dirty.clear()
            self._docs_since_flush = 0
            return

        dirty_ids = list(self._dirty)
        entries_to_flush = [self._cache[eid] for eid in dirty_ids if eid in self._cache]

        if not entries_to_flush:
            self._dirty.clear()
            self._docs_since_flush = 0
            return

        contents = [e["content"] for e in entries_to_flush]
        try:
            vecs = self.embedding_model.embed_documents(contents)
        except Exception as e:
            logger.warning("ExtractionMemoryStore.flush: embed_documents failed: %s", e)
            vecs = [None] * len(entries_to_flush)

        ids_to_upsert: List[str] = []
        docs_to_upsert: List[str] = []
        metas_to_upsert: List[Dict[str, Any]] = []
        embs_to_upsert: List[List[float]] = []

        for entry, vec in zip(entries_to_flush, vecs):
            if vec is None:
                continue
            eid = entry["id"]
            meta: Dict[str, Any] = {
                "type": entry.get("type", ""),
                "confidence": float(entry.get("confidence", 0.7)),
                "hit_count": int(entry.get("hit_count", 0)),
                "source": str(entry.get("source", "")),
                "memory_scope": str(entry.get("memory_scope", DEFAULT_MEMORY_SCOPE)),
                "created_at": float(entry.get("created_at", 0.0)),
                "updated_at": float(entry.get("updated_at", 0.0)),
                # Store keywords as pipe-separated string (ChromaDB metadata must be scalar)
                "keywords": "|||".join(entry.get("keywords") or []),
            }
            extra = {k: v for k, v in entry.items() if k not in self.BASE_META_KEYS and k not in {"id", "content"}}
            if extra:
                meta["extra_json"] = json.dumps(extra, ensure_ascii=False)
            ids_to_upsert.append(eid)
            docs_to_upsert.append(entry["content"])
            metas_to_upsert.append(meta)
            embs_to_upsert.append(vec)

        if ids_to_upsert:
            try:
                self._collection.upsert(
                    ids=ids_to_upsert,
                    documents=docs_to_upsert,
                    metadatas=metas_to_upsert,
                    embeddings=embs_to_upsert,
                )
                logger.debug("ExtractionMemoryStore.flush: upserted %d entries.", len(ids_to_upsert))
            except Exception as e:
                logger.warning("ExtractionMemoryStore.flush: upsert failed: %s", e)

        self._dirty.clear()
        self._docs_since_flush = 0

    def mark_doc_processed(self) -> None:
        """Call after finishing a document to trigger auto-flush check."""
        self._docs_since_flush += 1
        self._auto_flush()

    def get_entries(
        self,
        *,
        updated_after: Optional[float] = None,
        include_types: Optional[Set[str]] = None,
        exclude_types: Optional[Set[str]] = None,
        limit: Optional[int] = None,
        newest_first: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Return cache entries with optional time/type filters.

        This is used by memory distillation and debugging paths.
        """
        rows: List[Dict[str, Any]] = []
        for entry in self._cache.values():
            if not isinstance(entry, dict):
                continue
            et = str(entry.get("type", "")).strip()
            if include_types and et not in include_types:
                continue
            if exclude_types and et in exclude_types:
                continue
            if updated_after is not None:
                try:
                    upd = float(entry.get("updated_at", 0.0))
                except Exception:
                    upd = 0.0
                if upd <= float(updated_after):
                    continue
            rows.append(dict(entry))

        rows.sort(key=lambda x: float(x.get("updated_at", 0.0)), reverse=bool(newest_first))
        if limit is not None and int(limit) > 0:
            rows = rows[: int(limit)]
        return rows

    def has_scope_entries(self, memory_scope: str) -> bool:
        """
        Return whether cache contains at least one entry in the given scope.
        """
        scope = str(memory_scope or "").strip().lower()
        if not scope:
            return False
        for entry in self._cache.values():
            if not isinstance(entry, dict):
                continue
            entry_scope = str(entry.get("memory_scope", DEFAULT_MEMORY_SCOPE)).strip().lower() or DEFAULT_MEMORY_SCOPE
            if entry_scope == scope:
                return True
        return False

    # ------------------------------------------------------------------
    # Public API: Query
    # ------------------------------------------------------------------

    def query(
        self,
        text: str,
        *,
        max_entries: Optional[int] = None,
        max_chars_per_entry: Optional[int] = None,
        min_kw_hits: Optional[int] = None,
        task_context: Optional[Dict[str, Any]] = None,
        memory_scopes: Optional[Set[str]] = None,
    ) -> str:
        """
        Retrieve relevant memory entries for a text chunk.

        Returns a formatted string for prompt injection, or "" if nothing found.
        """
        max_entries = max_entries if max_entries is not None else self.max_context_entries
        max_chars = max_chars_per_entry if max_chars_per_entry is not None else self.max_chars_per_entry
        min_kw = min_kw_hits if min_kw_hits is not None else self.min_kw_hits

        if not text or not text.strip():
            return ""

        if not self._cache:
            return ""

        now = time.time()
        hit_ids: Set[str] = set()
        allowed_scopes = {str(x).strip().lower() for x in (memory_scopes or set()) if str(x).strip()}

        def _scope_ok(entry: Dict[str, Any]) -> bool:
            if not allowed_scopes:
                return True
            entry_scope = str(entry.get("memory_scope", DEFAULT_MEMORY_SCOPE)).strip().lower() or DEFAULT_MEMORY_SCOPE
            return entry_scope in allowed_scopes

        # ------------------------------------------------------------------
        # Pass A: keyword extraction + name index lookup
        # ------------------------------------------------------------------
        kw_list = extract_keywords(text, top_k=10)
        kw_hits: List[Dict[str, Any]] = []

        for kw in kw_list:
            kw_lower = kw.lower()
            # Exact match
            for eid in (self._name_index.get(kw_lower) or set()):
                if eid not in hit_ids and eid in self._cache:
                    ent = self._cache[eid]
                    if _scope_ok(ent):
                        hit_ids.add(eid)
                        kw_hits.append(ent)

            # Substring match (keyword is substring of index term, or vice versa)
            if len(kw_lower) >= 2:
                for surface, eids in self._name_index.items():
                    if surface == kw_lower:
                        continue
                    if kw_lower in surface or surface in kw_lower:
                        for eid in eids:
                            if eid not in hit_ids and eid in self._cache:
                                ent = self._cache[eid]
                                if _scope_ok(ent):
                                    hit_ids.add(eid)
                                    kw_hits.append(ent)

        # ------------------------------------------------------------------
        # Pass B: vector search (only if keyword hits are insufficient)
        # ------------------------------------------------------------------
        vec_hits: List[Dict[str, Any]] = []
        if len(kw_hits) < min_kw and kw_list and self._collection is not None:
            query_str = " ".join(kw_list)
            try:
                n_results = min(max_entries * 2, max(8, len(self._cache)))
                res = self._collection.query(
                    query_embeddings=[self._embed(query_str)],
                    n_results=n_results,
                    include=["distances", "metadatas", "documents"],
                )
                ids_list = (res.get("ids") or [[]])[0]
                dists = (res.get("distances") or [[]])[0]
                for eid, dist in zip(ids_list, dists):
                    sim = 1.0 - float(dist)
                    if sim < 0.5:  # low-quality match, skip
                        continue
                    if eid not in hit_ids and eid in self._cache:
                        ent = self._cache[eid]
                        if _scope_ok(ent):
                            hit_ids.add(eid)
                            vec_hits.append(ent)
            except Exception as e:
                logger.debug("ExtractionMemoryStore.query vector search failed: %s", e)

        # ------------------------------------------------------------------
        # Merge, rank, budget cap
        # ------------------------------------------------------------------
        all_hits = kw_hits + vec_hits
        if not all_hits:
            return ""

        # Sort by relevance score descending
        def _context_bonus(e: Dict[str, Any]) -> float:
            if not task_context:
                return 0.0
            bonus = 0.0
            if str(e.get("type", "")) in DERIVED_MEMORY_TYPES:
                bonus += 0.02
            tc_task = str(task_context.get("task_name", "")).strip().lower()
            tc_doc = str(task_context.get("doc_type", "")).strip().lower()
            tc_lang = str(task_context.get("language", "")).strip().lower()
            apps_task = e.get("applicable_tasks")
            if isinstance(apps_task, list):
                apps_task_txt = " ".join([str(x).strip().lower() for x in apps_task if str(x).strip()])
            else:
                apps_task_txt = str(apps_task or "").strip().lower()
            if tc_task and apps_task_txt.find(tc_task) >= 0:
                bonus += 0.06
            apps_doc = e.get("applicable_doc_types")
            if isinstance(apps_doc, list):
                apps_doc_txt = " ".join([str(x).strip().lower() for x in apps_doc if str(x).strip()])
            else:
                apps_doc_txt = str(apps_doc or "").strip().lower()
            if tc_doc and apps_doc_txt.find(tc_doc) >= 0:
                bonus += 0.05
            if tc_lang and str(e.get("language", "")).strip().lower() == tc_lang:
                bonus += 0.04
            reuse_level = str(e.get("reuse_level", "")).strip().lower()
            if reuse_level == "global_reusable":
                bonus += 0.03
            elif reuse_level == "domain_reusable":
                bonus += 0.02
            return bonus

        all_hits.sort(key=lambda e: (_rank_score(e, now) + _context_bonus(e)), reverse=True)

        # Deduplicate (preserve order)
        seen_ids: Set[str] = set()
        ranked: List[Dict[str, Any]] = []
        for e in all_hits:
            eid = e.get("id", "")
            if eid and eid not in seen_ids:
                seen_ids.add(eid)
                ranked.append(e)

        ranked = ranked[: max(max_entries * 3, max_entries)]

        # Dynamic composition:
        # 1) summary distilled memories first
        # 2) raw memories capped by max_raw_context_entries
        # 3) remaining slots filled with non-summary distilled memories
        summaries: List[Dict[str, Any]] = []
        raw_rows: List[Dict[str, Any]] = []
        derived_rows: List[Dict[str, Any]] = []
        for e in ranked:
            t = str(e.get("type", ""))
            if t in SUMMARY_MEMORY_TYPES:
                summaries.append(e)
            elif t in DERIVED_MEMORY_TYPES:
                derived_rows.append(e)
            else:
                raw_rows.append(e)

        final_ranked: List[Dict[str, Any]] = []
        used_ids: Set[str] = set()

        for e in summaries:
            eid = str(e.get("id", ""))
            if not eid or eid in used_ids:
                continue
            final_ranked.append(e)
            used_ids.add(eid)
            if len(final_ranked) >= max_entries:
                break

        if len(final_ranked) < max_entries:
            raw_cap = min(self.max_raw_context_entries, max_entries - len(final_ranked))
            raw_count = 0
            for e in raw_rows:
                eid = str(e.get("id", ""))
                if not eid or eid in used_ids:
                    continue
                final_ranked.append(e)
                used_ids.add(eid)
                raw_count += 1
                if raw_count >= raw_cap or len(final_ranked) >= max_entries:
                    break

        if len(final_ranked) < max_entries:
            for e in derived_rows:
                eid = str(e.get("id", ""))
                if not eid or eid in used_ids:
                    continue
                final_ranked.append(e)
                used_ids.add(eid)
                if len(final_ranked) >= max_entries:
                    break

        ranked = final_ranked

        # Update hit_counts for matched entries
        for e in ranked:
            eid = e["id"]
            self._cache[eid]["hit_count"] = self._cache[eid].get("hit_count", 0) + 1
            self._dirty.add(eid)

        # Format output
        lines: List[str] = []
        for e in ranked:
            content = str(e.get("content", "")).strip()
            if len(content) > max_chars:
                content = content[: max_chars - 3] + "..."
            if content:
                lines.append(f"- [{e.get('type', '')}] {content}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # JSON dump
    # ------------------------------------------------------------------

    def dump_to_json(self, path: str, *, silent: bool = False) -> None:
        """
        Serialize all in-memory cache entries to a JSON file for human inspection.

        The output is:
          {
            "total": <int>,
            "entries": [ { id, type, content, keywords, confidence, hit_count,
                           created_at, updated_at, source }, ... ]
          }
        """
        import json as _json

        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        serializable = []
        for e in self._cache.values():
            ec = dict(e)
            # Ensure keywords is a list (not a set)
            kw = ec.get("keywords")
            if isinstance(kw, (set, frozenset)):
                ec["keywords"] = sorted(kw)
            serializable.append(ec)

        # Sort by created_at for stable output
        serializable.sort(key=lambda x: float(x.get("created_at", 0.0)))

        payload = {
            "total": len(serializable),
            "entries": serializable,
        }
        with open(path, "w", encoding="utf-8") as f:
            _json.dump(payload, f, ensure_ascii=False, indent=2)

        if not silent:
            logger.info("ExtractionMemoryStore: dumped %d entries to %s", len(serializable), path)

    def clear(self) -> None:
        """
        Clear all extraction memories from cache and persistent collection.
        """
        ids = list(self._cache.keys())
        if self._collection is not None and not ids:
            try:
                res = self._collection.get()
                ids = list(res.get("ids") or [])
            except Exception:
                ids = []
        if self._collection is not None and ids:
            try:
                self._collection.delete(ids=ids)
            except Exception as e:
                logger.warning("ExtractionMemoryStore.clear: delete failed: %s", e)

        self._cache.clear()
        self._name_index.clear()
        self._dirty.clear()
        self._docs_since_flush = 0
        logger.info("ExtractionMemoryStore: all entries cleared.")

    # ------------------------------------------------------------------
    # Convenience factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config: Any) -> "ExtractionMemoryStore":
        """
        Build an ExtractionMemoryStore from a KAGConfig object.

        Reads config.extraction_memory (ExtractionMemoryConfig).
        Uses config.embedding for the embedding model.
        """
        from core.model_providers.openai_embedding import OpenAIEmbeddingModel

        emb_model = OpenAIEmbeddingModel(config.embedding)

        mem_cfg = getattr(config, "extraction_memory", None)
        store_path = "data/memory/raw_memory"
        max_context_entries = 12
        max_chars_per_entry = 120
        similarity_threshold = 0.92
        flush_every_n_entries = 20
        flush_every_n_docs = 10
        min_kw_hits = 3
        max_raw_context_entries = 5
        realtime_store_path = "data/memory/realtime_memory"

        if mem_cfg is not None:
            store_path = str(getattr(mem_cfg, "raw_store_path", store_path) or store_path)
            max_context_entries = int(getattr(mem_cfg, "max_context_entries", max_context_entries))
            max_chars_per_entry = int(getattr(mem_cfg, "max_chars_per_entry", max_chars_per_entry))
            similarity_threshold = float(getattr(mem_cfg, "similarity_threshold", similarity_threshold))
            flush_every_n_entries = int(getattr(mem_cfg, "flush_every_n_entries", flush_every_n_entries))
            flush_every_n_docs = int(getattr(mem_cfg, "flush_every_n_docs", flush_every_n_docs))
            min_kw_hits = int(getattr(mem_cfg, "min_kw_hits", min_kw_hits))
            max_raw_context_entries = int(
                getattr(mem_cfg, "max_raw_context_entries", max_raw_context_entries)
            )
            realtime_store_path = str(getattr(mem_cfg, "realtime_store_path", realtime_store_path) or realtime_store_path)

        return cls(
            store_path=store_path,
            embedding_model=emb_model,
            max_context_entries=max_context_entries,
            max_chars_per_entry=max_chars_per_entry,
            similarity_threshold=similarity_threshold,
            flush_every_n_entries=flush_every_n_entries,
            flush_every_n_docs=flush_every_n_docs,
            min_kw_hits=min_kw_hits,
            max_raw_context_entries=max_raw_context_entries,
            realtime_store_path=realtime_store_path,
        )

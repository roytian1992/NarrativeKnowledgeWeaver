# retriever/parent_child_retriever.py

# -*- coding: utf-8 -*-
"""
Parent-Child Retriever
======================

This module implements a **two-stage retrieval strategy** that combines
sentence-level and parent-document-level retrieval:

Workflow:
---------
1. **Sentence-level retrieval**  
   - Retrieve candidate sentences from a sentence-level vector store.
   - Optionally apply an *early rerank* using a cross-encoder or reranker.

2. **Sentence → Parent aggregation**  
   - Map retrieved sentences back to their parent documents.
   - Collect evidence windows around the hit sentences.

3. **Parent-level retrieval and rerank**  
   - Supplement with direct parent-document retrieval.
   - Aggregate evidence and run an optional reranker over parent documents.

4. **Final output**  
   - Return parent-level `Document` objects with metadata including:
     - `"similarity_score"` (from reranker or sentence prior)
     - `"source"` (e.g., `"sentence_window"`, `"parent_direct"`, `"parent_only"`)
     - `"evidence"` (if available)

Dependencies:
-------------
- Requires two vector stores:
  - `doc_vs`: document-level store with `search(query, limit)` and `search_by_ids(ids)`.
  - `sent_vs`: sentence-level store with the same API.
- Optionally integrates a reranker (e.g., `OpenAIRerankModel`).

Class:
------
    ParentChildRetriever
        Combines sentence-level and parent-level retrieval with reranking.

Usage example:
--------------
    >>> retriever = ParentChildRetriever(doc_vs, sent_vs, reranker=my_reranker)
    >>> results = retriever.retrieve("nuclear reactor shutdown", ks=20, topn=5)
    >>> for doc in results:
    ...     print(doc.id, doc.metadata["source"], doc.metadata["similarity_score"])
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from core.models.data import Document

_SENT_ID_RE = re.compile(r"^(?P<parent>.+?)<->(?P<idx>\d+)$")


class ParentChildRetriever:
    """
    Two-stage retriever: **sentence-level → parent-document-level**.

    Features:
    - Sentence-level recall with optional *early rerank*.
    - Evidence window expansion around hit sentences.
    - Aggregation and rerank at parent-document level.
    - Metadata enrichment (source, score, evidence).
    """

    def __init__(
        self,
        doc_vs,  # Document-level VectorStore
        sent_vs,  # Sentence-level VectorStore
        reranker=None,
        rerank_with_evidence: bool = True,
        evidence_sep: str = "\n\n--- evidence ---\n\n",
        *,
        child_first_rerank: bool = True,
        child_rerank_multiplier: int = 2,
        child_rerank_use_window: bool = True,
    ):
        """
        Initialize the parent-child retriever.

        Args:
            doc_vs: Document-level vector store (must implement search, search_by_ids).
            sent_vs: Sentence-level vector store (must implement search, search_by_ids).
            reranker: Optional reranker with `.rerank(query, documents, top_n, return_documents)`.
            rerank_with_evidence: Whether to include evidence in parent rerank input.
            evidence_sep: Separator string between evidence and parent text.
            child_first_rerank: Whether to rerank sentence-level hits before aggregation.
            child_rerank_multiplier: Expansion factor for candidate sentences before rerank.
            child_rerank_use_window: Whether to use sentence windows in rerank input.
        """
        self.doc_vs = doc_vs
        self.sent_vs = sent_vs
        self.reranker = reranker
        self.rerank_with_evidence = rerank_with_evidence
        self.evidence_sep = evidence_sep

        self.child_first_rerank = bool(child_first_rerank)
        self.child_rerank_multiplier = max(1, int(child_rerank_multiplier))
        self.child_rerank_use_window = bool(child_rerank_use_window)

    # ---------------- Internal helpers ----------------

    @staticmethod
    def _parse_sentence_id(sid: str) -> Optional[Tuple[str, int]]:
        """Parse sentence ID of the form 'parentId<->12' into (parent_id, sent_idx)."""
        m = _SENT_ID_RE.match(str(sid))
        return (m.group("parent"), int(m.group("idx"))) if m else None

    def _expand_window_text(self, parent_id: str, sent_idx: int, window: int) -> str:
        """
        Expand evidence text around a sentence index by ±window.

        Args:
            parent_id: ID of the parent document.
            sent_idx: Index of the hit sentence.
            window: Number of neighboring sentences to include.

        Returns:
            Concatenated text from neighboring sentences.
        """
        if window <= 0:
            return ""
        start = max(1, sent_idx - window)
        end = sent_idx + window
        ids = [f"{parent_id}-{i}" for i in range(start, end + 1)]
        neighbors: List[Document] = self.sent_vs.search_by_ids(ids)
        pairs = []
        for d in neighbors or []:
            parsed = self._parse_sentence_id(d.id)
            if parsed:
                pairs.append((parsed[1], d.content or ""))
        pairs.sort(key=lambda x: x[0])
        return " ".join([p[1] for p in pairs]) if pairs else ""

    # ---------------- Retrieval pipeline ----------------

    def retrieve(
        self,
        query: str,
        *,
        ks: int = 20,
        kp: int = 6,
        window: int = 1,
        topn: int = 8,
        parent_only_fallback: bool = True,
    ) -> List[Document]:
        """
        Run the full parent-child retrieval pipeline.

        Args:
            query: Query text.
            ks: Number of sentence-level hits to retain.
            kp: Number of parent-level hits to supplement.
            window: Evidence window size (± sentences).
            topn: Number of final parent documents to return.
            parent_only_fallback: Fallback to parent-only retrieval if no hits.

        Returns:
            List of parent-level `Document` objects with enriched metadata.
        """
        # 1) Sentence-level retrieval (with optional early rerank)
        raw_k = ks * (self.child_rerank_multiplier if (self.reranker and self.child_first_rerank) else 1)
        raw_sent_hits: List[Document] = self.sent_vs.search(query, limit=raw_k) or []

        if self.reranker and self.child_first_rerank and raw_sent_hits:
            child_docs_for_rerank: List[str] = []
            for h in raw_sent_hits:
                parsed = self._parse_sentence_id(h.id)
                if parsed and self.child_rerank_use_window and window > 0:
                    parent_id, idx = parsed
                    ev = self._expand_window_text(parent_id, idx, window)
                    child_docs_for_rerank.append(ev or (h.content or ""))
                else:
                    child_docs_for_rerank.append(h.content or "")

            child_res = self.reranker.rerank(
                query=query,
                documents=child_docs_for_rerank,
                top_n=min(ks, len(child_docs_for_rerank)),
                return_documents=False,
            ) or []

            child_res = sorted(
                (r for r in child_res if isinstance(r.get("index"), int)),
                key=lambda x: (x.get("relevance_score") or 0.0),
                reverse=True,
            )[:ks]
            chosen_child_idx = [r["index"] for r in child_res]
            sent_hits: List[Document] = [raw_sent_hits[i] for i in chosen_child_idx]
        else:
            sent_hits: List[Document] = raw_sent_hits[:ks]

        # 2) Aggregate parent hits from sentence hits
        agg: Dict[str, Dict[str, Any]] = {}
        evidences: Dict[str, List[str]] = {}

        for h in sent_hits:
            parsed = self._parse_sentence_id(h.id)
            if not parsed:
                continue
            parent_id, idx = parsed
            ev = self._expand_window_text(parent_id, idx, window) or (h.content or "")
            score = float((h.metadata or {}).get("similarity_score", 0.0))
            cur = agg.get(parent_id)
            if cur is None or score > cur.get("best_sent_score", 0.0):
                agg[parent_id] = {"best_sent_score": score, "source": "sentence_window"}
            evidences.setdefault(parent_id, []).append(ev)

        # 3) Retrieve parent documents
        parent_ids = list(agg.keys())
        parents: List[Document] = self.doc_vs.search_by_ids(parent_ids) if parent_ids else []

        # 4) Supplement with parent-level retrieval
        parent_hits: List[Document] = self.doc_vs.search(query, limit=kp) or []
        for h in parent_hits:
            pid = str(h.id)
            pscore = float((h.metadata or {}).get("similarity_score", 0.0))
            cur = agg.get(pid)
            if cur is None or pscore > cur.get("best_sent_score", 0.0):
                agg.setdefault(pid, {})
                agg[pid]["best_sent_score"] = max(pscore, agg[pid].get("best_sent_score", 0.0))
                agg[pid]["source"] = agg[pid].get("source", "parent_direct")

        # 5) Fallback: parent-only search
        if not agg and parent_only_fallback:
            fallback = self.doc_vs.search(query, limit=topn) or []
            for d in fallback:
                (d.metadata or {}).setdefault("source", "parent_only")
            return fallback

        # 6) Fetch missing parent documents
        have = {str(p.id) for p in parents}
        need = set(agg.keys()) - have
        if need:
            more = self.doc_vs.search_by_ids(list(need)) or []
            parents += more

        pid2doc: Dict[str, Document] = {str(p.id): p for p in (parents or [])}

        # 7) Prepare parent rerank input
        docs_for_rerank: List[str] = []
        items: List[Dict[str, Any]] = []
        for pid, info in agg.items():
            pd = pid2doc.get(pid)
            if not pd:
                continue
            ev = " ".join(evidences.get(pid, [])).strip()
            parent_text = pd.content or ""
            text_for_rerank = (
                f"{ev}{self.evidence_sep}{parent_text}"
                if (self.rerank_with_evidence and ev)
                else parent_text
            )
            docs_for_rerank.append(text_for_rerank)
            items.append(
                {
                    "pid": pid,
                    "parent_text": parent_text,
                    "evidence": ev,
                    "source": info.get("source", "mixed"),
                    "prior": float(info.get("best_sent_score", 0.0)),
                    "meta": pd.metadata or {},
                }
            )

        # 8) Parent rerank (or fallback to prior scores)
        if self.reranker and docs_for_rerank:
            res = self.reranker.rerank(
                query=query,
                documents=docs_for_rerank,
                top_n=min(topn, len(docs_for_rerank)),
                return_documents=False,
            ) or []
            res = sorted(
                (r for r in res if isinstance(r.get("index"), int)),
                key=lambda x: (x.get("relevance_score") or 0.0),
                reverse=True,
            )[:topn]
            chosen = [r["index"] for r in res]
            score_map = {r["index"]: float(r.get("relevance_score") or 0.0) for r in res}
        else:
            ranks = sorted(enumerate(items), key=lambda kv: kv[1]["prior"], reverse=True)[:topn]
            chosen = [i for i, _ in ranks]
            score_map = {i: items[i]["prior"] for i in chosen}

        # 9) Assemble final parent documents
        out: List[Document] = []
        for i in chosen:
            it = items[i]
            pid = it["pid"]
            pd = pid2doc.get(pid)
            if not pd:
                continue
            meta = dict(it["meta"])
            meta["similarity_score"] = score_map.get(i, it["prior"])
            meta["source"] = it["source"]
            if it["evidence"]:
                meta["evidence"] = it["evidence"]
            out.append(Document(id=pid, content=it["parent_text"], metadata=meta))

        return out

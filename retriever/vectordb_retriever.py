import re
from typing import List, Dict, Any, Optional, Tuple
from core.models.data import Document

_SENT_ID_RE = re.compile(r"^(?P<parent>.+?)<->(?P<idx>\d+)$")

class ParentChildRetriever:
    def __init__(
        self,
        doc_vs,                     # 文档级 VectorStore
        sent_vs,                    # 句子级 VectorStore
        reranker=None,              # OpenAIRerankModel 或 None
        rerank_with_evidence: bool = True,
        evidence_sep: str = "\n\n--- evidence ---\n\n",
    ):
        self.doc_vs = doc_vs
        self.sent_vs = sent_vs
        self.reranker = reranker
        self.rerank_with_evidence = rerank_with_evidence
        self.evidence_sep = evidence_sep

    @staticmethod
    def _parse_sentence_id(sid: str) -> Optional[Tuple[str, int]]:
        m = _SENT_ID_RE.match(str(sid))
        return (m.group("parent"), int(m.group("idx"))) if m else None

    def _expand_window_text(self, parent_id: str, sent_idx: int, window: int) -> str:
        if window <= 0:
            return ""
        ids = [f"{parent_id}-{i}" for i in range(max(1, sent_idx - window), sent_idx + window + 1)]
        neighbors: List[Document] = self.sent_vs.search_by_ids(ids)
        pairs = []
        for d in neighbors:
            parsed = self._parse_sentence_id(d.id)
            if parsed:
                pairs.append((parsed[1], d.content))
        pairs.sort(key=lambda x: x[0])
        return " ".join([p[1] for p in pairs]) if pairs else ""

    def retrieve(
        self,
        query: str,
        *,
        ks: int = 20,
        kp: int = 6,
        window: int = 1,
        topn: int = 8,
        parent_only_fallback: bool = True
    ) -> List[Document]:
        # 1) 句子库精召回
        sent_hits: List[Document] = self.sent_vs.search(query, limit=ks)

        # parent_id -> {best_sent_score, evidence_texts[], source}
        agg: Dict[str, Dict[str, Any]] = {}
        evidences: Dict[str, List[str]] = {}

        for h in sent_hits:
            parsed = self._parse_sentence_id(h.id)
            if not parsed:
                continue
            parent_id, idx = parsed
            ev = self._expand_window_text(parent_id, idx, window) or h.content
            score = float((h.metadata or {}).get("similarity_score", 0.0))

            cur = agg.get(parent_id)
            if cur is None or score > cur["best_sent_score"]:
                agg[parent_id] = {"best_sent_score": score, "source": "sentence_window"}
            evidences.setdefault(parent_id, []).append(ev)

        # 2) 子→父拿正文
        parent_ids = list(agg.keys())
        parents = self.doc_vs.search_by_ids(parent_ids) if parent_ids else []

        # 3) 文档库补召回
        parent_hits: List[Document] = self.doc_vs.search(query, limit=kp)
        for h in parent_hits:
            pid = str(h.id)
            pscore = float((h.metadata or {}).get("similarity_score", 0.0))
            cur = agg.get(pid)
            if cur is None or pscore > cur.get("best_sent_score", 0.0):
                agg.setdefault(pid, {})
                agg[pid]["best_sent_score"] = max(pscore, agg[pid].get("best_sent_score", 0.0))
                agg[pid]["source"] = agg[pid].get("source", "parent_direct")

        if not agg and parent_only_fallback:
            fallback = self.doc_vs.search(query, limit=topn)
            for d in fallback:
                (d.metadata or {}).setdefault("source", "parent_only")
            return fallback

        # 4) 补齐父正文
        need = set(agg.keys()) - {str(p.id) for p in parents}
        if need:
            parents += self.doc_vs.search_by_ids(list(need))
        pid2doc: Dict[str, Document] = {str(p.id): p for p in parents}

        # 5) 组装重排输入 —— 严格使用 List[str]
        docs_for_rerank: List[str] = []
        items: List[Dict[str, Any]] = []
        for pid, info in agg.items():
            pd = pid2doc.get(pid)
            if not pd:
                continue
            ev = " ".join(evidences.get(pid, [])).strip()
            parent_text = pd.content
            text_for_rerank = f"{ev}{self.evidence_sep}{parent_text}" if (self.rerank_with_evidence and ev) else parent_text
            docs_for_rerank.append(text_for_rerank)
            items.append({
                "pid": pid,
                "parent_text": parent_text,
                "evidence": ev,
                "source": info.get("source", "mixed"),
                "prior": float(info.get("best_sent_score", 0.0)),
                "meta": pd.metadata or {}
            })

        # 6) 重排（documents: List[str]）
        if self.reranker and docs_for_rerank:
            res = self.reranker.rerank(
                query=query,
                documents=docs_for_rerank,     # ✅ 只传 List[str]
                top_n=topn,
                return_documents=False
            )
            res = sorted(
                (r for r in res if isinstance(r.get("index"), int)),
                key=lambda x: (x.get("relevance_score") or 0.0),
                reverse=True
            )[:topn]
            chosen = [r["index"] for r in res]
            score_map = {r["index"]: float(r.get("relevance_score") or 0.0) for r in res}
        else:
            # 无重排：按先验分
            ranks = sorted(enumerate(items), key=lambda kv: kv[1]["prior"], reverse=True)[:topn]
            chosen = [i for i, _ in ranks]
            score_map = {i: items[i]["prior"] for i in chosen}

        # 7) 组装为父文档列表
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

import re
from typing import List, Dict, Any, Optional, Tuple
from core.models.data import Document

_SENT_ID_RE = re.compile(r"^(?P<parent>.+?)<->(?P<idx>\d+)$")


class ParentChildRetriever:
    """
    基于“句子级→父文档级”的两段式检索器。
    支持在子文档（句子）召回后先进行一次 early rerank（扩倍率×ks），
    再进入既有的父文档聚合与（可选）父级重排流程。
    """

    def __init__(
        self,
        doc_vs,                     # 文档级 VectorStore：需实现 search(query, limit), search_by_ids(ids)
        sent_vs,                    # 句子级 VectorStore：需实现 search(query, limit), search_by_ids(ids)
        reranker=None,              # OpenAIRerankModel 或兼容接口（.rerank(query, documents, top_n, return_documents)）
        rerank_with_evidence: bool = True,
        evidence_sep: str = "\n\n--- evidence ---\n\n",
        *,
        # ---- 子级 early rerank 开关与参数 ----
        child_first_rerank: bool = True,
        child_rerank_multiplier: int = 2,
        child_rerank_use_window: bool = True,
    ):
        self.doc_vs = doc_vs
        self.sent_vs = sent_vs
        self.reranker = reranker
        self.rerank_with_evidence = rerank_with_evidence
        self.evidence_sep = evidence_sep

        self.child_first_rerank = bool(child_first_rerank)
        self.child_rerank_multiplier = max(1, int(child_rerank_multiplier))
        self.child_rerank_use_window = bool(child_rerank_use_window)

    @staticmethod
    def _parse_sentence_id(sid: str) -> Optional[Tuple[str, int]]:
        """
        解析句子 ID（形如 'parentId<->12'），返回 (parent_id, sent_idx)。
        """
        m = _SENT_ID_RE.match(str(sid))
        return (m.group("parent"), int(m.group("idx"))) if m else None

    def _expand_window_text(self, parent_id: str, sent_idx: int, window: int) -> str:
        """
        基于句子索引，向左右扩展 window，拼接邻居句子的文本作为证据。
        假设句子向量库中的条目 id 为 'parentId-<num>'，其中 <num> 从 1 开始递增。
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

    def retrieve(
        self,
        query: str,
        *,
        ks: int = 20,               # 子级（句子）最终保留数量
        kp: int = 6,                # 父文档直接召回数量（补召回）
        window: int = 1,            # 证据窗口大小
        topn: int = 8,              # 最终返回父文档数量
        parent_only_fallback: bool = True
    ) -> List[Document]:
        # ---------------------------
        # 1) 子级召回（可选：early rerank）
        # ---------------------------
        raw_k = ks * (self.child_rerank_multiplier if (self.reranker and self.child_first_rerank) else 1)
        raw_sent_hits: List[Document] = self.sent_vs.search(query, limit=raw_k) or []

        if self.reranker and self.child_first_rerank and raw_sent_hits:
            # 为子级重排准备文本：可选使用窗口文本以增强鲁棒性
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
                return_documents=False
            ) or []

            child_res = sorted(
                (r for r in child_res if isinstance(r.get("index"), int)),
                key=lambda x: (x.get("relevance_score") or 0.0),
                reverse=True
            )[:ks]
            chosen_child_idx = [r["index"] for r in child_res]
            sent_hits: List[Document] = [raw_sent_hits[i] for i in chosen_child_idx]
        else:
            # 不启用子级重排：直接使用前 ks 条
            sent_hits: List[Document] = raw_sent_hits[:ks]

        # -----------------------------------------
        # 2) 基于句子命中聚合父文档 + 收集证据与先验分
        # -----------------------------------------
        agg: Dict[str, Dict[str, Any]] = {}      # parent_id -> {"best_sent_score": float, "source": str}
        evidences: Dict[str, List[str]] = {}     # parent_id -> [evidence_texts]

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

        # ---------------------------
        # 3) 子→父拿正文
        # ---------------------------
        parent_ids = list(agg.keys())
        parents: List[Document] = self.doc_vs.search_by_ids(parent_ids) if parent_ids else []

        # ---------------------------
        # 4) 父文档库补召回（直接检索）
        # ---------------------------
        parent_hits: List[Document] = self.doc_vs.search(query, limit=kp) or []
        for h in parent_hits:
            pid = str(h.id)
            pscore = float((h.metadata or {}).get("similarity_score", 0.0))
            cur = agg.get(pid)
            if cur is None or pscore > cur.get("best_sent_score", 0.0):
                agg.setdefault(pid, {})
                agg[pid]["best_sent_score"] = max(pscore, agg[pid].get("best_sent_score", 0.0))
                # 仅在未被句子命中时标为 parent_direct；否则保留原有 source
                agg[pid]["source"] = agg[pid].get("source", "parent_direct")

        # ---------------------------
        # 5) 兜底：如果彻底没有父候选
        # ---------------------------
        if not agg and parent_only_fallback:
            fallback = self.doc_vs.search(query, limit=topn) or []
            for d in fallback:
                (d.metadata or {}).setdefault("source", "parent_only")
            return fallback

        # ---------------------------
        # 6) 补齐父正文
        # ---------------------------
        have = {str(p.id) for p in parents}
        need = set(agg.keys()) - have
        if need:
            more = self.doc_vs.search_by_ids(list(need)) or []
            parents += more

        pid2doc: Dict[str, Document] = {str(p.id): p for p in (parents or [])}

        # ---------------------------
        # 7) 组装父级重排输入（证据 + 分隔符 + 正文）
        # ---------------------------
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
            items.append({
                "pid": pid,
                "parent_text": parent_text,
                "evidence": ev,
                "source": info.get("source", "mixed"),
                "prior": float(info.get("best_sent_score", 0.0)),
                "meta": pd.metadata or {}
            })

        # ---------------------------
        # 8) 父级重排（或用先验分）
        # ---------------------------
        if self.reranker and docs_for_rerank:
            res = self.reranker.rerank(
                query=query,
                documents=docs_for_rerank,  # ✅ 只传 List[str]
                top_n=min(topn, len(docs_for_rerank)),
                return_documents=False
            ) or []
            res = sorted(
                (r for r in res if isinstance(r.get("index"), int)),
                key=lambda x: (x.get("relevance_score") or 0.0),
                reverse=True
            )[:topn]
            chosen = [r["index"] for r in res]
            score_map = {r["index"]: float(r.get("relevance_score") or 0.0) for r in res}
        else:
            ranks = sorted(enumerate(items), key=lambda kv: kv[1]["prior"], reverse=True)[:topn]
            chosen = [i for i, _ in ranks]
            score_map = {i: items[i]["prior"] for i in chosen}

        # ---------------------------
        # 9) 组装为父文档列表（带打分/来源/证据）
        # ---------------------------
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

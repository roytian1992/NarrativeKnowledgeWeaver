from __future__ import annotations

import json
import math
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from langchain_core.documents import Document as LCDocument

from core.utils.format import correct_json_format
from retriever.sparse_retriever import KeywordBM25Retriever


def _safe_str(value: Any) -> str:
    return str(value or "").strip()


def _as_list(value: Any) -> List[str]:
    if isinstance(value, list):
        out: List[str] = []
        seen = set()
        for item in value:
            text = _safe_str(item)
            if not text or text in seen:
                continue
            seen.add(text)
            out.append(text)
        return out
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    return []


def _props_to_text(properties: Any) -> str:
    if not isinstance(properties, dict):
        return ""
    bits: List[str] = []
    for key in [
        "title",
        "original_title",
        "source_title",
        "summary",
        "topic",
        "domain",
        "related_characters",
        "related_occasions",
        "related_events",
        "top_members",
    ]:
        value = properties.get(key)
        if isinstance(value, list):
            value = ", ".join(_as_list(value))
        value_text = _safe_str(value)
        if value_text:
            bits.append(f"{key}: {value_text}")
    return "\n".join(bits)


def _join_nonempty(parts: Sequence[str], sep: str = "\n") -> str:
    return sep.join([part for part in (_safe_str(p) for p in parts) if part])


def _vector_norm(vec: Sequence[float]) -> float:
    return math.sqrt(sum(float(x) * float(x) for x in vec))


def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    denom = _vector_norm(left) * _vector_norm(right)
    if denom <= 0:
        return 0.0
    score = sum(float(a) * float(b) for a, b in zip(left, right)) / denom
    return float(score)


def _rrf(rank: int, *, k: int = 60) -> float:
    return 1.0 / float(k + max(0, rank) + 1)


def _clamp01(value: Any, default: float = 0.0) -> float:
    try:
        num = float(value)
    except Exception:
        return float(default)
    return max(0.0, min(1.0, num))


def _looks_like_mcq(query: str) -> bool:
    text = _safe_str(query)
    lower = text.lower()
    if "choices:" in lower or "options:" in lower:
        return True
    markers = ["(a)", "(b)", "(c)", "(d)"]
    return sum(1 for marker in markers if marker in lower) >= 2


def _keyword_hits(text: str, keywords: Sequence[str]) -> int:
    lower = _safe_str(text).lower()
    return sum(1 for kw in keywords if kw in lower)


def _query_prefers_narrative_tree(query: str) -> bool:
    narrative_keywords = [
        "why", "how", "cause", "because", "reason", "motivation", "plan", "goal",
        "before", "after", "then", "finally", "relationship", "plot", "storyline",
        "episode", "event", "turning point", "led to", "result", "consequence",
        "为什么", "如何", "原因", "导致", "结果", "之前", "之后", "情节", "剧情", "主线", "事件",
    ]
    return _keyword_hits(query, narrative_keywords) > 0


def _query_prefers_direct_evidence(query: str) -> bool:
    direct_keywords = [
        "who", "what", "when", "where", "which", "name", "title", "object", "item",
        "upon", "greeted", "said", "told", "wearing", "holding", "arrive", "landing",
        "谁", "什么", "何时", "哪里", "哪一个", "名字", "标题", "台词", "细节", "落地", "到达",
    ]
    return _keyword_hits(query, direct_keywords) > 0 or _looks_like_mcq(query)


def _encode_text(embedding_model: Any, text: str) -> Optional[List[float]]:
    if embedding_model is None:
        return None
    try:
        if hasattr(embedding_model, "encode"):
            result = embedding_model.encode([text])
            if hasattr(result, "tolist"):
                result = result.tolist()
            if isinstance(result, list) and result and isinstance(result[0], list):
                return [float(x) for x in result[0]]
            if isinstance(result, list):
                return [float(x) for x in result]
        if hasattr(embedding_model, "embed_query"):
            result = embedding_model.embed_query(text)
            if isinstance(result, list):
                return [float(x) for x in result]
    except Exception:
        return None
    return None


@dataclass
class IndexedNode:
    node_id: str
    name: str
    description: str
    text: str
    source_documents: List[str]
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None


@dataclass
class TreeSearchMCTSConfig:
    enabled: bool = False
    simulations: int = 24
    exploration_weight: float = 1.25
    branching_factor: int = 3
    max_rollout_depth: int = 4
    min_candidates_to_trigger: int = 6
    value_llm_weight: float = 0.35
    value_semantic_weight: float = 0.40
    value_prior_weight: float = 0.25
    document_limit_per_leaf: int = 3


@dataclass
class _MCTSNode:
    key: str
    level_name: str
    row: Optional[Dict[str, Any]]
    parent: Optional["_MCTSNode"] = None
    prior: float = 0.0
    terminal: bool = False
    depth: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    children: List["_MCTSNode"] = field(default_factory=list)
    visits: int = 0
    value_sum: float = 0.0
    expanded: bool = False

    @property
    def mean_value(self) -> float:
        if self.visits <= 0:
            return 0.0
        return float(self.value_sum) / float(self.visits)


class HybridNodeIndex:
    def __init__(self, nodes: List[IndexedNode], *, embedding_model: Any = None):
        self.embedding_model = embedding_model
        self.nodes = nodes
        self.node_map: Dict[str, IndexedNode] = {node.node_id: node for node in nodes}
        self.bm25_docs: List[LCDocument] = [
            LCDocument(
                page_content=node.text,
                metadata={"node_id": node.node_id, "name": node.name},
            )
            for node in nodes
            if node.text
        ]
        self._bm25 = KeywordBM25Retriever(self.bm25_docs, k_default=min(20, max(1, len(self.bm25_docs)))) if self.bm25_docs else None

    def _vector_search(self, query: str, *, limit: int, candidate_ids: Optional[Iterable[str]] = None) -> List[Tuple[str, float]]:
        query_vec = _encode_text(self.embedding_model, query)
        if not query_vec:
            return []
        allow = set(candidate_ids) if candidate_ids is not None else None
        rows: List[Tuple[str, float]] = []
        for node in self.nodes:
            if allow is not None and node.node_id not in allow:
                continue
            if not node.embedding:
                continue
            score = _cosine_similarity(query_vec, node.embedding)
            if score <= 0:
                continue
            rows.append((node.node_id, float(score)))
        rows.sort(key=lambda item: item[1], reverse=True)
        return rows[: max(1, int(limit or 5))]

    def _keyword_search(self, query: str, *, limit: int, candidate_ids: Optional[Iterable[str]] = None) -> List[str]:
        if self._bm25 is None:
            return []
        allow = set(candidate_ids) if candidate_ids is not None else None
        if allow is None:
            docs = self._bm25.retrieve(query, k=max(1, int(limit or 5)))
            return [_safe_str((doc.metadata or {}).get("node_id")) for doc in docs if _safe_str((doc.metadata or {}).get("node_id"))]

        filtered_docs = [doc for doc in self.bm25_docs if _safe_str((doc.metadata or {}).get("node_id")) in allow]
        if not filtered_docs:
            return []
        retriever = KeywordBM25Retriever(filtered_docs, k_default=min(max(1, int(limit or 5)), len(filtered_docs)))
        docs = retriever.retrieve(query, k=max(1, int(limit or 5)))
        return [_safe_str((doc.metadata or {}).get("node_id")) for doc in docs if _safe_str((doc.metadata or {}).get("node_id"))]

    def search(
        self,
        query: str,
        *,
        top_k: int,
        vector_top_k: int,
        keyword_top_k: int,
        candidate_ids: Optional[Iterable[str]] = None,
    ) -> List[Dict[str, Any]]:
        vector_hits = self._vector_search(query, limit=max(1, int(vector_top_k or top_k or 5)), candidate_ids=candidate_ids)
        keyword_hits = self._keyword_search(query, limit=max(1, int(keyword_top_k or top_k or 5)), candidate_ids=candidate_ids)

        fused: Dict[str, Dict[str, Any]] = {}
        for rank, (node_id, score) in enumerate(vector_hits):
            fused.setdefault(node_id, {"vector_score": 0.0, "keyword_score": 0.0, "rrf_score": 0.0})
            fused[node_id]["vector_score"] = max(fused[node_id]["vector_score"], float(score))
            fused[node_id]["rrf_score"] += _rrf(rank)
        for rank, node_id in enumerate(keyword_hits):
            if not node_id:
                continue
            fused.setdefault(node_id, {"vector_score": 0.0, "keyword_score": 0.0, "rrf_score": 0.0})
            fused[node_id]["keyword_score"] = max(fused[node_id]["keyword_score"], 1.0 - (rank / max(1, len(keyword_hits))))
            fused[node_id]["rrf_score"] += _rrf(rank)

        rows: List[Dict[str, Any]] = []
        for node_id, score_pack in fused.items():
            node = self.node_map.get(node_id)
            if node is None:
                continue
            rows.append(
                {
                    "id": node.node_id,
                    "name": node.name,
                    "description": node.description,
                    "source_documents": list(node.source_documents),
                    "properties": dict(node.metadata),
                    "vector_score": float(score_pack["vector_score"]),
                    "keyword_score": float(score_pack["keyword_score"]),
                    "score": float(score_pack["rrf_score"]),
                }
            )
        rows.sort(
            key=lambda row: (
                float(row.get("score", 0.0)),
                float(row.get("vector_score", 0.0)),
                float(row.get("keyword_score", 0.0)),
                row.get("name", ""),
            ),
            reverse=True,
        )
        return rows[: max(1, int(top_k or 5))]


class TreeSearchRetrieverBackbone:
    """
    Shared tree-search retrieval backbone.

    This class owns the common mechanics used by both concrete shapes:
    - section-shaped retrieval
    - narrative-shaped retrieval

    The public tools still expose section / narrative interfaces, but both now
    rely on the same tree-search backbone instead of independent ad hoc logic.
    """
    def __init__(
        self,
        *,
        graph_query_utils: Any,
        document_vector_store: Any,
        sentence_vector_store: Any,
        document_parser: Any,
        max_workers: int = 4,
        tree_search_config: Optional[Any] = None,
    ) -> None:
        self.graph_query_utils = graph_query_utils
        self.document_vector_store = document_vector_store
        self.sentence_vector_store = sentence_vector_store
        self.document_parser = document_parser
        self.max_workers = max(1, int(max_workers or 4))
        self.embedding_model = (
            getattr(self.sentence_vector_store, "embedding_model", None)
            or getattr(self.document_vector_store, "embedding_model", None)
            or getattr(self.graph_query_utils, "model", None)
        )
        self.graph = self.graph_query_utils._graph()
        self.tree_search_config = self._normalize_tree_search_config(tree_search_config)
        self._candidate_relevance_cache: Dict[Tuple[str, str, str], float] = {}
        self._document_eval_cache: Dict[Tuple[str, str, int], Dict[str, Any]] = {}
        self._document_text_cache: Dict[str, str] = {}

    @staticmethod
    def _normalize_tree_search_config(cfg: Optional[Any]) -> TreeSearchMCTSConfig:
        if isinstance(cfg, TreeSearchMCTSConfig):
            return cfg
        if cfg is None:
            return TreeSearchMCTSConfig()
        out = TreeSearchMCTSConfig()
        for field_name in out.__dataclass_fields__.keys():
            if hasattr(cfg, field_name):
                setattr(out, field_name, getattr(cfg, field_name))
            elif isinstance(cfg, dict) and field_name in cfg:
                setattr(out, field_name, cfg.get(field_name))
        out.simulations = max(1, int(out.simulations or 1))
        out.branching_factor = max(1, int(out.branching_factor or 1))
        out.max_rollout_depth = max(1, int(out.max_rollout_depth or 1))
        out.min_candidates_to_trigger = max(1, int(out.min_candidates_to_trigger or 1))
        out.document_limit_per_leaf = max(1, int(out.document_limit_per_leaf or 1))
        out.exploration_weight = max(0.1, float(out.exploration_weight or 0.1))
        return out

    def _should_use_mcts(self, *, candidate_count: int) -> bool:
        cfg = self.tree_search_config
        return bool(cfg.enabled) and int(candidate_count or 0) >= int(cfg.min_candidates_to_trigger or 1)

    @staticmethod
    def _candidate_key(level_name: str, row: Dict[str, Any]) -> str:
        rid = _safe_str(row.get("id") or row.get("document_id") or row.get("name"))
        return f"{_safe_str(level_name).lower()}::{rid}"

    @staticmethod
    def _candidate_prior(row: Dict[str, Any]) -> float:
        parts = [
            _clamp01(row.get("score", 0.0)),
            _clamp01(row.get("vector_score", 0.0)),
            _clamp01(row.get("keyword_score", 0.0)),
            _clamp01(row.get("doc_score", 0.0)),
            _clamp01(row.get("best_sentence_score", 0.0)),
            _clamp01(row.get("parent_score", 0.0)),
        ]
        parts = [p for p in parts if p > 0]
        if not parts:
            return 0.05
        return max(0.05, min(1.0, sum(parts) / len(parts)))

    def _score_candidate_probability(
        self,
        *,
        query: str,
        row: Dict[str, Any],
        level_name: str,
        branch_context: str = "",
    ) -> float:
        row_key = self._candidate_key(level_name, row)
        cache_key = (_safe_str(level_name), row_key, _safe_str(query))
        cached = self._candidate_relevance_cache.get(cache_key)
        if cached is not None:
            return float(cached)
        prob = 0.0
        if self.document_parser is not None:
            try:
                raw = self.document_parser.score_candidate_relevance(
                    text=self._build_candidate_text(row),
                    goal=_join_nonempty([query, branch_context], sep="\n\n"),
                )
                parsed = json.loads(correct_json_format(raw))
                prob = _clamp01(parsed.get("probability", 0.0))
            except Exception:
                prob = 0.0
        if prob <= 0.0:
            prob = self._candidate_prior(row)
        self._candidate_relevance_cache[cache_key] = float(prob)
        return float(prob)

    def _get_document_text(self, document_id: str) -> str:
        doc_id = _safe_str(document_id)
        if not doc_id:
            return ""
        cached = self._document_text_cache.get(doc_id)
        if cached is not None:
            return cached
        texts = self._collect_document_texts([doc_id])
        text = _safe_str(texts.get(doc_id))
        self._document_text_cache[doc_id] = text
        return text

    def _evaluate_document_row(
        self,
        *,
        query: str,
        row: Dict[str, Any],
        max_length: int,
    ) -> Dict[str, Any]:
        document_id = _safe_str(row.get("document_id"))
        cache_key = (_safe_str(query), document_id, max(80, int(max_length or 240)))
        cached = self._document_eval_cache.get(cache_key)
        if cached is not None:
            return dict(cached)

        snippet = ""
        sentence_hits = list(row.get("sentence_hits") or [])
        if sentence_hits:
            snippet = " ".join([_safe_str(text) for _, text in sentence_hits[:4]]).strip()
        full_text = self._get_document_text(document_id)
        llm_prob = 0.0
        if self.document_parser is not None:
            try:
                parser_input = snippet if len(snippet) >= 120 else (full_text or snippet)
                if parser_input:
                    raw = self.document_parser.search_related_content(
                        text=parser_input,
                        goal=query,
                        max_length=max_length,
                    )
                    parsed = _safe_str(raw)
                    if parsed.startswith("{") and "related_content" in parsed:
                        obj = json.loads(parsed)
                        parsed = _safe_str(obj.get("related_content"))
                    if parsed:
                        snippet = parsed
                score_raw = self.document_parser.score_candidate_relevance(
                    text=snippet or full_text[: max(80, int(max_length or 240))],
                    goal=query,
                )
                score_data = json.loads(correct_json_format(score_raw))
                llm_prob = _clamp01(score_data.get("probability", 0.0))
            except Exception:
                llm_prob = 0.0
        semantic_score = max(
            _clamp01(row.get("doc_score", 0.0)),
            _clamp01(row.get("best_sentence_score", 0.0)),
        )
        prior_score = max(
            _clamp01(row.get("parent_score", 0.0)),
            _clamp01(row.get("score", 0.0)),
        )
        cfg = self.tree_search_config
        value = (
            float(cfg.value_llm_weight) * llm_prob
            + float(cfg.value_semantic_weight) * semantic_score
            + float(cfg.value_prior_weight) * prior_score
        )
        payload = {
            "document_id": document_id,
            "snippet": _safe_str(snippet) or _safe_str(full_text[: max(80, int(max_length or 240))]),
            "value": float(value),
            "llm_prob": float(llm_prob),
            "semantic_score": float(semantic_score),
            "prior_score": float(prior_score),
        }
        self._document_eval_cache[cache_key] = dict(payload)
        return payload

    def _select_mcts_child(self, node: _MCTSNode) -> Optional[_MCTSNode]:
        if not node.children:
            return None
        parent_visits = max(1, int(node.visits))
        explore = float(self.tree_search_config.exploration_weight)
        best: Optional[_MCTSNode] = None
        best_score = float("-inf")
        for child in node.children:
            q = child.mean_value
            u = explore * max(0.05, float(child.prior)) * math.sqrt(parent_visits) / (1.0 + float(child.visits))
            score = q + u
            if score > best_score:
                best = child
                best_score = score
        return best

    def _attach_children(
        self,
        *,
        parent: _MCTSNode,
        level_name: str,
        rows: List[Dict[str, Any]],
        terminal: bool = False,
        metadata_builder: Optional[Any] = None,
    ) -> None:
        existing = {child.key for child in parent.children}
        for row in rows:
            key = self._candidate_key(level_name, row)
            if key in existing:
                continue
            metadata = metadata_builder(row) if callable(metadata_builder) else {}
            parent.children.append(
                _MCTSNode(
                    key=key,
                    level_name=level_name,
                    row=dict(row),
                    parent=parent,
                    prior=self._candidate_prior(row),
                    terminal=bool(terminal),
                    depth=parent.depth + 1,
                    metadata=dict(metadata or {}),
                )
            )
        parent.expanded = True

    def _backpropagate(self, path: List[_MCTSNode], value: float) -> None:
        reward = float(value)
        for node in path:
            node.visits += 1
            node.value_sum += reward

    def _collect_terminal_nodes(self, root: _MCTSNode) -> List[_MCTSNode]:
        out: List[_MCTSNode] = []
        stack = [root]
        while stack:
            node = stack.pop()
            if node.terminal:
                out.append(node)
            stack.extend(node.children)
        out.sort(
            key=lambda item: (
                float(item.mean_value),
                int(item.visits),
                float(item.prior),
            ),
            reverse=True,
        )
        return out

    @staticmethod
    def _build_candidate_text(row: Dict[str, Any]) -> str:
        parts: List[str] = []
        name = _safe_str(row.get("name"))
        if name:
            parts.append(f"title: {name}")
        description = _safe_str(row.get("description"))
        if description:
            parts.append(f"description: {description}")
        if row.get("storyline_names"):
            parts.append(f"storylines: {', '.join(_as_list(row.get('storyline_names')))}")
        if row.get("episode_names"):
            parts.append(f"episodes: {', '.join(_as_list(row.get('episode_names')))}")
        if row.get("source_documents"):
            parts.append(f"source_documents: {', '.join(_as_list(row.get('source_documents')))}")
        properties = row.get("properties")
        prop_text = _props_to_text(properties)
        if prop_text:
            parts.append(prop_text)
        return _join_nonempty(parts)

    def _score_rows_by_llm(
        self,
        *,
        rows: List[Dict[str, Any]],
        query: str,
        threshold: float,
        top_k: int,
        branch_context: str = "",
    ) -> List[Dict[str, Any]]:
        if not rows or self.document_parser is None:
            return rows[: max(1, int(top_k or 5))]

        goal = _safe_str(query)
        if branch_context:
            goal = f"{goal}\n\nPath hints:\n{branch_context}".strip()

        def _score(row: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
            rid = _safe_str(row.get("id"))
            try:
                raw = self.document_parser.score_candidate_relevance(
                    text=self._build_candidate_text(row),
                    goal=goal,
                )
                data = json.loads(correct_json_format(raw))
                probability = _clamp01(data.get("probability", 0.0))
                is_relevant = bool(data.get("is_relevant", probability >= threshold))
                reason = _safe_str(data.get("reason"))
            except Exception:
                probability = 0.0
                is_relevant = False
                reason = ""
            return rid, {
                "probability": probability,
                "is_relevant": is_relevant,
                "reason": reason,
            }

        outputs: Dict[str, Dict[str, Any]] = {}
        worker_count = max(1, min(self.max_workers, len(rows)))
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [executor.submit(_score, row) for row in rows]
            for future in as_completed(futures):
                rid, parsed = future.result()
                if rid:
                    outputs[rid] = parsed

        ranked: List[Dict[str, Any]] = []
        for row in rows:
            parsed = outputs.get(_safe_str(row.get("id")), {})
            new_row = dict(row)
            prob = _clamp01(parsed.get("probability", 0.0))
            is_rel = bool(parsed.get("is_relevant", prob >= threshold))
            new_row["llm_probability"] = prob
            new_row["llm_is_relevant"] = is_rel
            new_row["llm_reason"] = _safe_str(parsed.get("reason"))
            new_row["score"] = 0.6 * prob + 0.4 * _clamp01(row.get("score", 0.0))
            ranked.append(new_row)

        kept = [
            row for row in ranked
            if row.get("llm_is_relevant") or float(row.get("llm_probability", 0.0)) >= threshold
        ]
        ranked.sort(
            key=lambda row: (
                float(row.get("score", 0.0)),
                float(row.get("llm_probability", 0.0)),
                float(row.get("vector_score", 0.0)),
                float(row.get("keyword_score", 0.0)),
            ),
            reverse=True,
        )
        kept.sort(
            key=lambda row: (
                float(row.get("score", 0.0)),
                float(row.get("llm_probability", 0.0)),
                float(row.get("vector_score", 0.0)),
                float(row.get("keyword_score", 0.0)),
            ),
            reverse=True,
        )
        return (kept or ranked[:1])[: max(1, int(top_k or 5))]

    def _render_tree_prompt(
        self,
        *,
        level_name: str,
        query: str,
        rows: List[Dict[str, Any]],
        top_k: int,
        branch_context: str = "",
    ) -> str:
        level_key = _safe_str(level_name).lower()
        if level_key == "storyline":
            selection_focus = (
                "Pick the broad narrative arc that most likely contains the answer. "
                "Prefer the main storyline that explains the question over side arcs."
            )
        elif level_key == "episode":
            selection_focus = (
                "Pick the episode or stage where the answering scene most likely occurs. "
                "Prefer the stage that directly contains the decisive evidence."
            )
        elif level_key == "event":
            selection_focus = (
                "Pick the concrete event that most directly supports the final answer. "
                "Prefer explicit answer-bearing actions or interactions over broad summaries."
            )
        else:
            selection_focus = (
                "Pick the section that is most likely to contain explicit answer evidence. "
                "Prefer sections with direct wording, local details, or decisive clues."
            )
        candidate_lines: List[str] = []
        for idx, row in enumerate(rows, 1):
            rid = _safe_str(row.get("id")) or f"{level_name.lower()}_{idx}"
            name = _safe_str(row.get("name")) or "(unnamed)"
            desc = _safe_str(row.get("description"))
            docs = ", ".join(_as_list(row.get("source_documents")))
            parent_storylines = ", ".join(_as_list(row.get("storyline_names")))
            parent_episodes = ", ".join(_as_list(row.get("episode_names")))
            candidate_lines.append(f"- id: {rid}")
            candidate_lines.append(f"  title: {name}")
            if desc:
                candidate_lines.append(f"  description: {desc[:700]}")
            if parent_storylines:
                candidate_lines.append(f"  storylines: {parent_storylines}")
            if parent_episodes:
                candidate_lines.append(f"  episodes: {parent_episodes}")
            if docs:
                candidate_lines.append(f"  source_documents: {docs}")

        branch_text = branch_context or "(none)"
        return (
            f"You are doing hierarchical tree search over candidate {level_name} nodes.\n"
            f"Select up to {max(1, int(top_k or 1))} node ids that are most likely to contain direct evidence for the query.\n"
            f"{selection_focus}\n"
            "Prefer nodes that can answer the question directly or sharply constrain the answer.\n"
            "Avoid broad but weakly related nodes.\n"
            "For multiple-choice questions, prefer nodes that best discriminate between options.\n\n"
            f"Query:\n{query}\n\n"
            f"Current tree path hints:\n{branch_text}\n\n"
            f"Candidate {level_name} nodes:\n" + "\n".join(candidate_lines) + "\n\n"
            "Return strict JSON only:\n"
            '{\n'
            '  "thinking": "brief reasoning",\n'
            '  "node_list": ["node_id_1", "node_id_2"]\n'
            '}\n'
        )

    def _should_run_tree_planner(
        self,
        *,
        level_name: str,
        query: str,
        rows: List[Dict[str, Any]],
        top_k: int,
    ) -> bool:
        shortlist = list(rows[: max(1, int(top_k or 5) * 3)])
        if len(shortlist) <= max(1, int(top_k or 5)):
            return False

        top1 = float(shortlist[0].get("score", 0.0) or 0.0)
        top2 = float(shortlist[1].get("score", 0.0) or 0.0) if len(shortlist) > 1 else 0.0
        strong_leader = top1 >= 0.72 and (top1 - top2) >= 0.12

        level_key = _safe_str(level_name).lower()
        is_mcq = _looks_like_mcq(query)
        prefers_narrative = _query_prefers_narrative_tree(query)
        prefers_direct = _query_prefers_direct_evidence(query)

        if level_key in {"storyline", "episode", "event"}:
            if is_mcq or prefers_narrative:
                return True
            return len(shortlist) >= max(4, top_k + 1) and not (level_key != "event" and strong_leader)

        if prefers_direct or is_mcq:
            return not strong_leader
        return not strong_leader and len(shortlist) >= max(5, top_k + 2)

    def _tree_search_select_rows(
        self,
        *,
        level_name: str,
        query: str,
        rows: List[Dict[str, Any]],
        top_k: int,
        threshold: float = 0.35,
        branch_context: str = "",
        candidate_limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        shortlist = list(rows[: max(1, int(candidate_limit or max(top_k * 3, 8)))])
        if not shortlist:
            return []
        if not self._should_run_tree_planner(
            level_name=level_name,
            query=query,
            rows=shortlist,
            top_k=top_k,
        ):
            return shortlist[: max(1, int(top_k or 5))]
        if self.document_parser is None or getattr(self.document_parser, "llm", None) is None:
            return shortlist[: max(1, int(top_k or 5))]

        selected_ids: List[str] = []
        prompt = self._render_tree_prompt(
            level_name=level_name,
            query=query,
            rows=shortlist,
            top_k=top_k,
            branch_context=branch_context,
        )
        try:
            responses = self.document_parser.llm.run([{"role": "user", "content": prompt}])
            raw = ""
            if isinstance(responses, list) and responses:
                raw = _safe_str(responses[-1].get("content"))
            data = json.loads(correct_json_format(raw))
            selected_ids = _as_list(data.get("node_list"))
        except Exception:
            selected_ids = []

        row_map = {_safe_str(row.get("id")): row for row in shortlist if _safe_str(row.get("id"))}
        selected_rows: List[Dict[str, Any]] = []
        for idx, node_id in enumerate(selected_ids):
            row = row_map.get(node_id)
            if row is None:
                continue
            new_row = dict(row)
            new_row["tree_selected"] = True
            new_row["tree_rank"] = idx + 1
            new_row["score"] = max(float(new_row.get("score", 0.0)), 1.0 - (idx / max(1, len(selected_ids))))
            selected_rows.append(new_row)

        if not selected_rows:
            return self._score_rows_by_llm(
                rows=shortlist,
                query=query,
                threshold=threshold,
                top_k=top_k,
                branch_context=branch_context,
            )

        rescored_selected = self._score_rows_by_llm(
            rows=selected_rows,
            query=query,
            threshold=threshold,
            top_k=top_k,
            branch_context=branch_context,
        )
        if len(rescored_selected) >= max(1, int(top_k or 5)):
            return rescored_selected[: max(1, int(top_k or 5))]

        used_ids = {_safe_str(row.get("id")) for row in rescored_selected}
        remainder = [row for row in shortlist if _safe_str(row.get("id")) not in used_ids]
        if not remainder:
            return rescored_selected[: max(1, int(top_k or 5))]
        filler = self._score_rows_by_llm(
            rows=remainder,
            query=query,
            threshold=threshold,
            top_k=max(1, int(top_k or 5) - len(rescored_selected)),
            branch_context=branch_context,
        )
        merged = rescored_selected + [row for row in filler if _safe_str(row.get("id")) not in used_ids]
        return merged[: max(1, int(top_k or 5))]

    def _collect_document_texts(self, document_ids: List[str]) -> Dict[str, str]:
        docs = self.document_vector_store.search_by_document_ids(document_ids)
        doc_texts: Dict[str, str] = {}
        for doc in docs or []:
            metadata = getattr(doc, "metadata", {}) or {}
            document_id = _safe_str(metadata.get("document_id"))
            text = _safe_str(getattr(doc, "content", ""))
            if not document_id or not text:
                continue
            doc_texts[document_id] = _join_nonempty([doc_texts.get(document_id, ""), text], sep="\n")
        return doc_texts

    def _collect_sentence_hits(
        self,
        *,
        query: str,
        document_ids: List[str],
        limit: int,
    ) -> Dict[str, List[Tuple[float, str]]]:
        wanted = _as_list(document_ids)
        if not wanted:
            return {}

        query_vec = _encode_text(getattr(self.sentence_vector_store, "embedding_model", None), query)
        sentence_rows: List[Tuple[str, str, float]] = []
        if query_vec:
            try:
                hits = self.sentence_vector_store.search_by_embedding(
                    query_vec,
                    limit=max(limit * 8, len(wanted) * 6),
                    metadata_filter={"document_id": {"$in": wanted}},
                ) or []
                for hit in hits:
                    metadata = getattr(hit, "metadata", {}) or {}
                    document_id = _safe_str(metadata.get("document_id"))
                    text = _safe_str(getattr(hit, "content", ""))
                    score = float(metadata.get("similarity_score", 0.0) or 0.0)
                    if document_id and text:
                        sentence_rows.append((document_id, text, score))
            except Exception:
                sentence_rows = []

        grouped_sentences: Dict[str, List[Tuple[float, str]]] = defaultdict(list)
        for document_id, text, score in sentence_rows:
            grouped_sentences[document_id].append((score, text))
        for document_id, rows in list(grouped_sentences.items()):
            deduped: List[Tuple[float, str]] = []
            seen = set()
            for score, text in sorted(rows, key=lambda item: item[0], reverse=True):
                if text in seen:
                    continue
                seen.add(text)
                deduped.append((float(score), text))
            grouped_sentences[document_id] = deduped
        return dict(grouped_sentences)

    def _rank_documents(
        self,
        *,
        query: str,
        document_ids: List[str],
        limit: int,
        parent_scores: Optional[Dict[str, float]] = None,
    ) -> List[Dict[str, Any]]:
        wanted = _as_list(document_ids)
        if not wanted:
            return []

        sentence_hits = self._collect_sentence_hits(
            query=query,
            document_ids=wanted,
            limit=max(4, int(limit or 5)),
        )
        parent_scores = {str(k): float(v or 0.0) for k, v in (parent_scores or {}).items()}

        doc_order = {document_id: idx for idx, document_id in enumerate(wanted)}
        rows: List[Dict[str, Any]] = []
        for document_id in wanted:
            hits = sentence_hits.get(document_id, [])
            sentence_score_sum = sum(max(0.0, float(score)) for score, _ in hits)
            hit_count = len(hits)
            doc_score = sentence_score_sum / math.sqrt(hit_count + 1.0) if hit_count else 0.0
            best_sentence_score = max((float(score) for score, _ in hits), default=0.0)
            parent_score = float(parent_scores.get(document_id, 0.0))
            final_score = max(parent_score, doc_score * 0.8, best_sentence_score * 0.7, parent_score + doc_score * 0.15)
            rows.append(
                {
                    "document_id": document_id,
                    "score": float(final_score),
                    "doc_score": float(doc_score),
                    "best_sentence_score": float(best_sentence_score),
                    "parent_score": float(parent_score),
                    "hit_count": hit_count,
                    "sentence_hits": hits,
                    "original_rank": int(doc_order.get(document_id, 10**9)),
                }
            )
        rows.sort(
            key=lambda row: (
                float(row.get("parent_score", 0.0)),
                float(row.get("score", 0.0)),
                float(row.get("doc_score", 0.0)),
                float(row.get("best_sentence_score", 0.0)),
                -int(row.get("original_rank", 10**9)),
            ),
            reverse=True,
        )
        return rows

    def _search_document_evidence(
        self,
        *,
        query: str,
        document_ids: List[str],
        max_length: int,
        limit: int,
        parent_scores: Optional[Dict[str, float]] = None,
    ) -> Tuple[List[Dict[str, Any]], List[Tuple[str, str]]]:
        wanted = _as_list(document_ids)
        if not wanted:
            return [], []

        ranked_docs = self._rank_documents(
            query=query,
            document_ids=wanted,
            limit=limit,
            parent_scores=parent_scores,
        )
        expanded_limit = max(max(1, int(limit or 5)), min(len(wanted), max(2, int(limit or 5)) * 2))
        parent_sorted_rows = sorted(
            ranked_docs,
            key=lambda row: (
                float(row.get("parent_score", 0.0)),
                -int(row.get("original_rank", 10**9)),
            ),
            reverse=True,
        )
        semantic_sorted_rows = sorted(
            ranked_docs,
            key=lambda row: (
                float(row.get("doc_score", 0.0)),
                float(row.get("best_sentence_score", 0.0)),
                float(row.get("score", 0.0)),
            ),
            reverse=True,
        )
        selected_document_ids: List[str] = []
        for row in parent_sorted_rows[:expanded_limit]:
            document_id = _safe_str(row.get("document_id"))
            if document_id and document_id not in selected_document_ids:
                selected_document_ids.append(document_id)
        for row in semantic_sorted_rows[:expanded_limit]:
            document_id = _safe_str(row.get("document_id"))
            if document_id and document_id not in selected_document_ids:
                selected_document_ids.append(document_id)
        for document_id in wanted:
            document_id = _safe_str(document_id)
            if document_id and document_id not in selected_document_ids:
                selected_document_ids.append(document_id)
            if len(selected_document_ids) >= expanded_limit:
                break
        selected_ranked_docs = [
            row for row in ranked_docs if _safe_str(row.get("document_id")) in set(selected_document_ids)
        ]
        selected_ranked_docs.sort(
            key=lambda row: (
                selected_document_ids.index(_safe_str(row.get("document_id"))),
                -float(row.get("score", 0.0)),
            )
        )
        doc_texts = self._collect_document_texts(selected_document_ids)

        outputs: List[Tuple[str, str, float]] = []
        for row in selected_ranked_docs:
            document_id = _safe_str(row.get("document_id"))
            snippet = ""
            best_score = float(row.get("score", 0.0) or 0.0)
            hits = list(row.get("sentence_hits") or [])
            if hits:
                hit_text = " ".join([text for _, text in hits[:4]]).strip()
                if hit_text:
                    snippet = hit_text
            full_text = _safe_str(doc_texts.get(document_id))
            if full_text:
                try:
                    parser_input = snippet if snippet and len(snippet) >= 120 else full_text
                    raw = self.document_parser.search_related_content(
                        text=parser_input,
                        goal=query,
                        max_length=max_length,
                    )
                    parsed = _safe_str(raw)
                    if parsed.startswith("{") and "related_content" in parsed:
                        import json

                        obj = json.loads(parsed)
                        parsed = _safe_str(obj.get("related_content"))
                    if parsed:
                        snippet = parsed
                except Exception:
                    if not snippet:
                        snippet = full_text[: max(80, int(max_length or 240))]
            snippet = _safe_str(snippet)
            if snippet:
                outputs.append((document_id, snippet[: max(80, int(max_length or 240))], best_score))
        outputs.sort(key=lambda row: (row[2], row[0]), reverse=True)
        return selected_ranked_docs, [(document_id, snippet) for document_id, snippet, _ in outputs[: max(1, expanded_limit)]]


class SectionTreeRetriever(TreeSearchRetrieverBackbone):
    def __init__(
        self,
        *,
        graph_query_utils: Any,
        document_vector_store: Any,
        sentence_vector_store: Any,
        document_parser: Any,
        section_label: str,
        max_workers: int = 4,
        tree_search_config: Optional[Any] = None,
    ) -> None:
        super().__init__(
            graph_query_utils=graph_query_utils,
            document_vector_store=document_vector_store,
            sentence_vector_store=sentence_vector_store,
            document_parser=document_parser,
            max_workers=max_workers,
            tree_search_config=tree_search_config,
        )
        self.section_label = _safe_str(section_label) or "Document"
        self.section_index = HybridNodeIndex(self._build_section_nodes(), embedding_model=self.embedding_model)

    def _build_section_nodes(self) -> List[IndexedNode]:
        nodes: List[IndexedNode] = []
        for node_id, data in self.graph.nodes(data=True):
            types = data.get("type") or []
            if isinstance(types, str):
                types = [types]
            if self.section_label not in types:
                continue
            name = _safe_str(data.get("name")) or node_id
            description = _safe_str(data.get("description"))
            properties = dict(data.get("properties") or {})
            text = _join_nonempty([name, description, _props_to_text(properties)])
            embedding = data.get("embedding")
            if isinstance(embedding, list):
                try:
                    embedding = [float(x) for x in embedding]
                except Exception:
                    embedding = None
            if embedding is None and text:
                embedding = _encode_text(self.embedding_model, text)
            nodes.append(
                IndexedNode(
                    node_id=str(node_id),
                    name=name,
                    description=description,
                    text=text,
                    source_documents=_as_list(data.get("source_documents")),
                    metadata=properties,
                    embedding=embedding,
                )
            )
        return nodes

    def _search_with_mcts(
        self,
        *,
        query: str,
        section_candidates: List[Dict[str, Any]],
        section_top_k: int,
        max_length: int,
    ) -> Dict[str, Any]:
        root = _MCTSNode(key="root", level_name="Root", row=None)
        shortlist = list(section_candidates[: max(len(section_candidates), section_top_k)])
        self._attach_children(
            parent=root,
            level_name=self.section_label,
            rows=shortlist,
            terminal=False,
        )
        simulations = max(self.tree_search_config.simulations, len(root.children) * 2)
        for _ in range(simulations):
            path = [root]
            node = root
            while True:
                if node.terminal or node.depth >= self.tree_search_config.max_rollout_depth:
                    break
                if not node.children:
                    if node.row is None:
                        break
                    source_documents = _as_list((node.row or {}).get("source_documents"))
                    parent_scores = {
                        document_id: float((node.row or {}).get("score", 0.0) or 0.0)
                        for document_id in source_documents
                    }
                    doc_rows = self._rank_documents(
                        query=query,
                        document_ids=source_documents,
                        limit=max(
                            self.tree_search_config.document_limit_per_leaf,
                            self.tree_search_config.branching_factor,
                        ),
                        parent_scores=parent_scores,
                    )
                    self._attach_children(
                        parent=node,
                        level_name="Document",
                        rows=doc_rows[: self.tree_search_config.branching_factor],
                        terminal=True,
                    )
                    if not node.children:
                        break
                next_node = self._select_mcts_child(node)
                if next_node is None:
                    break
                node = next_node
                path.append(node)
            value = 0.0
            if node.row is not None and _safe_str(node.level_name).lower() == "document":
                payload = self._evaluate_document_row(
                    query=query,
                    row=node.row,
                    max_length=max_length,
                )
                node.metadata.update(payload)
                value = float(payload.get("value", 0.0))
            elif node.row is not None:
                value = self._score_candidate_probability(
                    query=query,
                    row=node.row,
                    level_name=node.level_name,
                )
            self._backpropagate(path, value)

        section_rows: List[Dict[str, Any]] = []
        for child in root.children:
            row = dict(child.row or {})
            row["mcts_visits"] = child.visits
            row["mcts_value"] = round(child.mean_value, 4)
            row["score"] = max(float(row.get("score", 0.0)), child.mean_value)
            section_rows.append(row)
        section_rows.sort(
            key=lambda row: (
                float(row.get("mcts_value", 0.0)),
                int(row.get("mcts_visits", 0)),
                float(row.get("score", 0.0)),
            ),
            reverse=True,
        )
        section_rows = section_rows[: max(1, int(section_top_k or 8))]

        selected_section_ids = {_safe_str(row.get("id")) for row in section_rows}
        terminal_nodes = self._collect_terminal_nodes(root)
        document_map: Dict[str, Dict[str, Any]] = {}
        evidence: List[Tuple[str, str]] = []
        for leaf in terminal_nodes:
            if leaf.parent is None or _safe_str((leaf.parent.row or {}).get("id")) not in selected_section_ids:
                continue
            payload = leaf.metadata or self._evaluate_document_row(query=query, row=leaf.row or {}, max_length=max_length)
            document_id = _safe_str(payload.get("document_id") or (leaf.row or {}).get("document_id"))
            if not document_id:
                continue
            row = dict(leaf.row or {})
            row["score"] = max(float(row.get("score", 0.0)), float(payload.get("value", 0.0)))
            row["mcts_value"] = float(payload.get("value", 0.0))
            row["mcts_visits"] = int(leaf.visits)
            current = document_map.get(document_id)
            if current is None or float(row.get("score", 0.0)) > float(current.get("score", 0.0)):
                document_map[document_id] = row
            snippet = _safe_str(payload.get("snippet"))
            if snippet:
                evidence.append((document_id, snippet))
        ranked_docs = sorted(
            document_map.values(),
            key=lambda row: (
                float(row.get("mcts_value", 0.0)),
                int(row.get("mcts_visits", 0)),
                float(row.get("score", 0.0)),
            ),
            reverse=True,
        )
        dedup_evidence: List[Tuple[str, str]] = []
        seen_docs = set()
        for document_id, snippet in evidence:
            if document_id in seen_docs:
                continue
            seen_docs.add(document_id)
            dedup_evidence.append((document_id, snippet))
        return {
            "sections": section_rows,
            "documents": ranked_docs,
            "evidence": dedup_evidence[: max(1, len(ranked_docs) or 1)],
        }

    def search(
        self,
        query: str,
        *,
        section_top_k: int = 8,
        llm_filter_top_k: Optional[int] = None,
        llm_filter_threshold: float = 0.35,
        max_length: int = 320,
    ) -> Dict[str, Any]:
        candidate_k = max(int(section_top_k or 8), int(llm_filter_top_k or 0) or 0, 8)
        section_candidates = self.section_index.search(
            query,
            top_k=candidate_k,
            vector_top_k=max(5, candidate_k),
            keyword_top_k=max(5, min(10, candidate_k)),
        )
        section_rows = self._tree_search_select_rows(
            level_name=self.section_label,
            query=query,
            rows=section_candidates,
            top_k=max(1, int(section_top_k or 8)),
            threshold=float(llm_filter_threshold or 0.35),
            candidate_limit=candidate_k,
        )
        document_ids: List[str] = []
        parent_scores: Dict[str, float] = {}
        for row in section_rows:
            for document_id in _as_list(row.get("source_documents")):
                if document_id not in document_ids:
                    document_ids.append(document_id)
                parent_scores[document_id] = max(parent_scores.get(document_id, 0.0), float(row.get("score", 0.0) or 0.0))
        ranked_docs, evidence = self._search_document_evidence(
            query=query,
            document_ids=document_ids,
            max_length=max_length,
            limit=max(3, min(8, len(document_ids) or 1)),
            parent_scores=parent_scores,
        )
        return {
            "sections": section_rows[: max(1, int(section_top_k or 8))],
            "documents": ranked_docs,
            "evidence": evidence,
        }


class NarrativeTreeRetriever(TreeSearchRetrieverBackbone):
    def __init__(
        self,
        *,
        graph_query_utils: Any,
        document_vector_store: Any,
        sentence_vector_store: Any,
        document_parser: Any,
        max_workers: int = 4,
        tree_search_config: Optional[Any] = None,
    ) -> None:
        super().__init__(
            graph_query_utils=graph_query_utils,
            document_vector_store=document_vector_store,
            sentence_vector_store=sentence_vector_store,
            document_parser=document_parser,
            max_workers=max_workers,
            tree_search_config=tree_search_config,
        )
        self.storyline_to_episodes: Dict[str, List[str]] = defaultdict(list)
        self.episode_to_events: Dict[str, List[str]] = defaultdict(list)
        self.event_score_cache: Dict[str, float] = {}
        self.storyline_index = HybridNodeIndex(self._build_nodes("Storyline"), embedding_model=self.embedding_model)
        self.episode_index = HybridNodeIndex(self._build_nodes("Episode"), embedding_model=self.embedding_model)
        self.event_nodes = {node.node_id: node for node in self._build_nodes("Event")}
        self._build_hierarchy_maps()

    def _build_nodes(self, label: str) -> List[IndexedNode]:
        rows: List[IndexedNode] = []
        for node_id, data in self.graph.nodes(data=True):
            types = data.get("type") or []
            if isinstance(types, str):
                types = [types]
            if label not in types:
                continue
            properties = dict(data.get("properties") or {})
            name = _safe_str(data.get("name")) or node_id
            description = _safe_str(data.get("description"))
            text = _join_nonempty([name, description, _props_to_text(properties)])
            embedding = data.get("embedding")
            if isinstance(embedding, list):
                try:
                    embedding = [float(x) for x in embedding]
                except Exception:
                    embedding = None
            if embedding is None and text:
                embedding = _encode_text(self.embedding_model, text)
            rows.append(
                IndexedNode(
                    node_id=str(node_id),
                    name=name,
                    description=description,
                    text=text,
                    source_documents=_as_list(data.get("source_documents")),
                    metadata=properties,
                    embedding=embedding,
                )
            )
        return rows

    def _build_hierarchy_maps(self) -> None:
        for src, dst, key, data in self.graph.edges(keys=True, data=True):
            predicate = _safe_str(data.get("predicate") or data.get("relation_type"))
            if predicate == "STORYLINE_CONTAINS":
                if src in self.storyline_index.node_map and dst in self.episode_index.node_map:
                    self.storyline_to_episodes[src].append(dst)
            elif predicate == "EPISODE_CONTAINS":
                if src in self.episode_index.node_map and dst in self.event_nodes:
                    self.episode_to_events[src].append(dst)

    def _rank_events(self, query: str, *, episode_rows: List[Dict[str, Any]], final_top_k: int) -> List[Dict[str, Any]]:
        query_vec = _encode_text(self.embedding_model, query)
        merged: Dict[str, Dict[str, Any]] = {}
        for episode in episode_rows:
            parent_score = float(episode.get("score", 0.0) or 0.0)
            episode_name = _safe_str(episode.get("name"))
            for event_id in self.episode_to_events.get(_safe_str(episode.get("id")), []):
                event = self.event_nodes.get(event_id)
                if event is None:
                    continue
                local_score = 0.0
                if query_vec and event.embedding:
                    local_score = max(0.0, _cosine_similarity(query_vec, event.embedding))
                score = max(parent_score * 0.8, local_score, parent_score + local_score * 0.25)
                row = merged.setdefault(
                    event_id,
                    {
                        "id": event.node_id,
                        "name": event.name,
                        "description": event.description,
                        "source_documents": list(event.source_documents),
                        "properties": dict(event.metadata),
                        "episode_names": [],
                        "score": 0.0,
                    },
                )
                row["score"] = max(float(row.get("score", 0.0)), float(score))
                if episode_name and episode_name not in row["episode_names"]:
                    row["episode_names"].append(episode_name)
        rows = list(merged.values())
        rows.sort(key=lambda row: (float(row.get("score", 0.0)), row.get("name", "")), reverse=True)
        return rows[: max(1, int(final_top_k or 8))]

    def _narrative_expand_rows(
        self,
        *,
        query: str,
        node: _MCTSNode,
        episode_top_k: int,
        event_top_k: int,
    ) -> Tuple[str, List[Dict[str, Any]], bool]:
        level = _safe_str(node.level_name).lower()
        row = dict(node.row or {})
        if level == "storyline":
            candidate_episode_ids = list(self.storyline_to_episodes.get(_safe_str(row.get("id")), []))
            rows = self.episode_index.search(
                query,
                top_k=max(max(1, int(episode_top_k or 5)) * 2, 8),
                vector_top_k=max(6, int(episode_top_k or 5) * 2),
                keyword_top_k=max(4, int(episode_top_k or 5)),
                candidate_ids=candidate_episode_ids or None,
            )
            for child in rows:
                child.setdefault("storyline_names", [])
                s_name = _safe_str(row.get("name"))
                if s_name and s_name not in child["storyline_names"]:
                    child["storyline_names"].append(s_name)
                child["score"] = max(float(child.get("score", 0.0)), float(row.get("score", 0.0) or 0.0) * 0.85)
            return "Episode", rows[: max(4, self.tree_search_config.branching_factor * 2)], False
        if level == "episode":
            rows = self._rank_events(
                query,
                episode_rows=[row],
                final_top_k=max(max(1, int(event_top_k or 8)) * 2, 10),
            )
            if rows:
                return "Event", rows[: max(4, self.tree_search_config.branching_factor * 2)], False
            source_documents = _as_list(row.get("source_documents"))
            parent_scores = {document_id: float(row.get("score", 0.0) or 0.0) for document_id in source_documents}
            doc_rows = self._rank_documents(
                query=query,
                document_ids=source_documents,
                limit=max(
                    self.tree_search_config.document_limit_per_leaf,
                    self.tree_search_config.branching_factor,
                ),
                parent_scores=parent_scores,
            )
            return "Document", doc_rows[: self.tree_search_config.branching_factor], True
        if level == "event":
            source_documents = _as_list(row.get("source_documents"))
            parent_scores = {document_id: float(row.get("score", 0.0) or 0.0) for document_id in source_documents}
            doc_rows = self._rank_documents(
                query=query,
                document_ids=source_documents,
                limit=max(
                    self.tree_search_config.document_limit_per_leaf,
                    self.tree_search_config.branching_factor,
                ),
                parent_scores=parent_scores,
            )
            return "Document", doc_rows[: self.tree_search_config.branching_factor], True
        return "", [], False

    def _search_with_mcts(
        self,
        *,
        query: str,
        storyline_candidates: List[Dict[str, Any]],
        storyline_top_k: int,
        episode_top_k: int,
        event_top_k: int,
        document_top_k: int,
        max_evidence_length: int,
    ) -> Dict[str, Any]:
        fallback_mode = not storyline_candidates
        root_level = "Episode" if fallback_mode else "Storyline"
        root_candidates = storyline_candidates if storyline_candidates else self.episode_index.search(
            query,
            top_k=max(max(1, int(episode_top_k or 5)) * 2, 8),
            vector_top_k=max(6, int(episode_top_k or 5) * 2),
            keyword_top_k=max(4, int(episode_top_k or 5)),
        )
        root = _MCTSNode(key="root", level_name="Root", row=None)
        self._attach_children(parent=root, level_name=root_level, rows=root_candidates, terminal=False)

        simulations = max(self.tree_search_config.simulations, len(root.children) * 3)
        for _ in range(simulations):
            path = [root]
            node = root
            while True:
                if node.terminal or node.depth >= self.tree_search_config.max_rollout_depth:
                    break
                if not node.children:
                    next_level, rows, terminal = self._narrative_expand_rows(
                        query=query,
                        node=node,
                        episode_top_k=episode_top_k,
                        event_top_k=event_top_k,
                    )
                    if rows:
                        self._attach_children(
                            parent=node,
                            level_name=next_level,
                            rows=rows,
                            terminal=terminal,
                        )
                    if not node.children:
                        break
                next_node = self._select_mcts_child(node)
                if next_node is None:
                    break
                node = next_node
                path.append(node)
            value = 0.0
            if node.row is not None and _safe_str(node.level_name).lower() == "document":
                payload = self._evaluate_document_row(
                    query=query,
                    row=node.row,
                    max_length=max_evidence_length,
                )
                node.metadata.update(payload)
                value = float(payload.get("value", 0.0))
            elif node.row is not None:
                branch_bits: List[str] = []
                if node.parent is not None and node.parent.row is not None:
                    branch_bits.append(_safe_str(node.parent.row.get("name")))
                value = self._score_candidate_probability(
                    query=query,
                    row=node.row,
                    level_name=node.level_name,
                    branch_context=", ".join([b for b in branch_bits if b]),
                )
            self._backpropagate(path, value)

        selected_root_nodes = sorted(
            root.children,
            key=lambda child: (float(child.mean_value), int(child.visits), float(child.prior)),
            reverse=True,
        )
        if fallback_mode:
            selected_storyline_ids: set[str] = set()
            selected_episode_ids = {
                _safe_str(node.row.get("id"))
                for node in selected_root_nodes[: max(1, int(episode_top_k or 5))]
                if node.row is not None
            }
        else:
            selected_storyline_ids = {
                _safe_str(node.row.get("id"))
                for node in selected_root_nodes[: max(1, int(storyline_top_k or 4))]
                if node.row is not None
            }
            selected_episode_ids: set[str] = set()

        storyline_rows: List[Dict[str, Any]] = []
        episode_map: Dict[str, Dict[str, Any]] = {}
        event_map: Dict[str, Dict[str, Any]] = {}
        document_map: Dict[str, Dict[str, Any]] = {}
        evidence: List[Tuple[str, str]] = []

        def _node_to_row(node: _MCTSNode) -> Dict[str, Any]:
            row = dict(node.row or {})
            row["mcts_visits"] = node.visits
            row["mcts_value"] = round(node.mean_value, 4)
            row["score"] = max(float(row.get("score", 0.0)), node.mean_value)
            return row

        def _find_ancestor(node: Optional[_MCTSNode], level_name: str) -> Optional[_MCTSNode]:
            want = _safe_str(level_name).lower()
            cur = node
            while cur is not None:
                if _safe_str(cur.level_name).lower() == want:
                    return cur
                cur = cur.parent
            return None

        for child in root.children:
            if child.row is None:
                continue
            if _safe_str(child.level_name).lower() == "storyline":
                storyline_rows.append(_node_to_row(child))
            elif fallback_mode:
                row = _node_to_row(child)
                selected_episode_ids.add(_safe_str(row.get("id")))
                episode_map[_safe_str(row.get("id"))] = row

        all_nodes: List[_MCTSNode] = []
        stack = list(root.children)
        while stack:
            node = stack.pop()
            all_nodes.append(node)
            stack.extend(node.children)

        for node in all_nodes:
            if node.row is None:
                continue
            level = _safe_str(node.level_name).lower()
            row = _node_to_row(node)
            if level == "episode":
                story_ancestor = _find_ancestor(node.parent, "Storyline")
                story_id = _safe_str((story_ancestor.row or {}).get("id")) if story_ancestor is not None else ""
                if fallback_mode or story_id in selected_storyline_ids:
                    selected_episode_ids.add(_safe_str(row.get("id")))
                    current = episode_map.get(_safe_str(row.get("id")))
                    if current is None or float(row.get("score", 0.0)) > float(current.get("score", 0.0)):
                        episode_map[_safe_str(row.get("id"))] = row
            elif level == "event":
                episode_ancestor = _find_ancestor(node.parent, "Episode")
                episode_id = _safe_str((episode_ancestor.row or {}).get("id")) if episode_ancestor is not None else ""
                if episode_id in selected_episode_ids:
                    current = event_map.get(_safe_str(row.get("id")))
                    if current is None or float(row.get("score", 0.0)) > float(current.get("score", 0.0)):
                        event_map[_safe_str(row.get("id"))] = row
            elif level == "document":
                episode_ancestor = _find_ancestor(node.parent, "Episode")
                episode_id = _safe_str((episode_ancestor.row or {}).get("id")) if episode_ancestor is not None else ""
                story_ancestor = _find_ancestor(node.parent, "Storyline")
                story_id = _safe_str((story_ancestor.row or {}).get("id")) if story_ancestor is not None else ""
                if not fallback_mode and story_id and story_id not in selected_storyline_ids:
                    continue
                if episode_id and not fallback_mode and episode_id not in selected_episode_ids:
                    continue
                payload = node.metadata or self._evaluate_document_row(query=query, row=node.row, max_length=max_evidence_length)
                document_id = _safe_str(payload.get("document_id") or row.get("document_id"))
                if not document_id:
                    continue
                row["mcts_value"] = float(payload.get("value", 0.0))
                row["mcts_visits"] = node.visits
                row["score"] = max(float(row.get("score", 0.0)), float(payload.get("value", 0.0)))
                current = document_map.get(document_id)
                if current is None or float(row.get("score", 0.0)) > float(current.get("score", 0.0)):
                    document_map[document_id] = row
                snippet = _safe_str(payload.get("snippet"))
                if snippet:
                    evidence.append((document_id, snippet))

        storyline_rows.sort(key=lambda row: (float(row.get("mcts_value", 0.0)), int(row.get("mcts_visits", 0))), reverse=True)
        episode_rows = sorted(episode_map.values(), key=lambda row: (float(row.get("mcts_value", 0.0)), int(row.get("mcts_visits", 0))), reverse=True)
        event_rows = sorted(event_map.values(), key=lambda row: (float(row.get("mcts_value", 0.0)), int(row.get("mcts_visits", 0))), reverse=True)
        document_rows = sorted(document_map.values(), key=lambda row: (float(row.get("mcts_value", 0.0)), int(row.get("mcts_visits", 0))), reverse=True)

        dedup_evidence: List[Tuple[str, str]] = []
        seen_docs = set()
        for document_id, snippet in evidence:
            if document_id in seen_docs:
                continue
            seen_docs.add(document_id)
            dedup_evidence.append((document_id, snippet))

        return {
            "fallback_mode": fallback_mode,
            "storylines": storyline_rows[: max(1, int(storyline_top_k or 4))] if not fallback_mode else [],
            "episodes": episode_rows[: max(1, int(episode_top_k or 5))],
            "events": event_rows[: max(1, int(event_top_k or 8))],
            "documents": [_safe_str(row.get("document_id")) for row in document_rows[: max(1, int(document_top_k or 6))]],
            "document_rows": document_rows[: max(1, int(document_top_k or 6))],
            "evidence": dedup_evidence[: max(1, int(document_top_k or 6))],
        }

    def search(
        self,
        query: str,
        *,
        storyline_top_k: int = 4,
        episode_top_k: int = 5,
        event_top_k: int = 8,
        document_top_k: int = 6,
        llm_filter_threshold: float = 0.35,
        max_evidence_length: int = 240,
    ) -> Dict[str, Any]:
        storyline_candidates = self.storyline_index.search(
            query,
            top_k=max(1, int(storyline_top_k or 4)),
            vector_top_k=max(4, int(storyline_top_k or 4) * 2),
            keyword_top_k=max(3, int(storyline_top_k or 4)),
        ) if self.storyline_index.nodes else []
        storyline_rows = self._tree_search_select_rows(
            level_name="Storyline",
            query=query,
            rows=storyline_candidates,
            top_k=max(1, int(storyline_top_k or 4)),
            threshold=float(llm_filter_threshold or 0.35),
            candidate_limit=max(8, int(storyline_top_k or 4) * 2),
        ) if storyline_candidates else []

        fallback_mode = not storyline_rows
        candidate_episode_ids: Optional[List[str]] = None
        if not fallback_mode:
            ids: List[str] = []
            for storyline in storyline_rows:
                for episode_id in self.storyline_to_episodes.get(_safe_str(storyline.get("id")), []):
                    if episode_id not in ids:
                        ids.append(episode_id)
            candidate_episode_ids = ids or None

        episode_candidates = self.episode_index.search(
            query,
            top_k=max(max(1, int(episode_top_k or 5)) * 2, 8),
            vector_top_k=max(6, int(episode_top_k or 5) * 2),
            keyword_top_k=max(4, int(episode_top_k or 5)),
            candidate_ids=candidate_episode_ids,
        )
        branch_storylines = ", ".join([_safe_str(row.get("name")) for row in storyline_rows if _safe_str(row.get("name"))])
        episode_rows = self._tree_search_select_rows(
            level_name="Episode",
            query=query,
            rows=episode_candidates,
            top_k=max(1, int(episode_top_k or 5)),
            threshold=float(llm_filter_threshold or 0.35),
            branch_context=f"Selected storylines: {branch_storylines}" if branch_storylines else "",
            candidate_limit=max(8, int(episode_top_k or 5) * 2),
        ) if episode_candidates else []

        storyline_score_map = {row["id"]: float(row.get("score", 0.0) or 0.0) for row in storyline_rows}
        if not fallback_mode:
            episode_storyline_names: Dict[str, List[str]] = defaultdict(list)
            episode_storyline_scores: Dict[str, float] = defaultdict(float)
            for storyline_id, episode_ids in self.storyline_to_episodes.items():
                if storyline_id not in storyline_score_map:
                    continue
                storyline_name = self.storyline_index.node_map.get(storyline_id).name if self.storyline_index.node_map.get(storyline_id) else storyline_id
                for episode_id in episode_ids:
                    episode_storyline_names[episode_id].append(storyline_name)
                    episode_storyline_scores[episode_id] = max(episode_storyline_scores[episode_id], storyline_score_map[storyline_id])
            for row in episode_rows:
                row_id = _safe_str(row.get("id"))
                if row_id in episode_storyline_names:
                    row["storyline_names"] = episode_storyline_names[row_id]
                    row["score"] = max(float(row.get("score", 0.0)), episode_storyline_scores[row_id] * 0.85)

        event_candidates = self._rank_events(
            query,
            episode_rows=episode_rows,
            final_top_k=max(max(1, int(event_top_k or 8)) * 2, 10),
        )
        branch_episodes = ", ".join([_safe_str(row.get("name")) for row in episode_rows if _safe_str(row.get("name"))])
        event_rows = self._tree_search_select_rows(
            level_name="Event",
            query=query,
            rows=event_candidates,
            top_k=max(1, int(event_top_k or 8)),
            threshold=float(llm_filter_threshold or 0.35),
            branch_context=f"Selected episodes: {branch_episodes}" if branch_episodes else "",
            candidate_limit=max(10, int(event_top_k or 8) * 2),
        ) if event_candidates else []

        document_ids: List[str] = []
        parent_scores: Dict[str, float] = {}
        for row in episode_rows:
            for document_id in _as_list(row.get("source_documents")):
                if document_id not in document_ids:
                    document_ids.append(document_id)
                parent_scores[document_id] = max(parent_scores.get(document_id, 0.0), float(row.get("score", 0.0) or 0.0))
        for row in event_rows:
            for document_id in _as_list(row.get("source_documents")):
                if document_id not in document_ids:
                    document_ids.append(document_id)
                parent_scores[document_id] = max(parent_scores.get(document_id, 0.0), float(row.get("score", 0.0) or 0.0))
        ranked_docs, evidence = self._search_document_evidence(
            query=query,
            document_ids=document_ids,
            max_length=max_evidence_length,
            limit=max(3, min(8, int(document_top_k or 6))),
            parent_scores=parent_scores,
        )

        return {
            "fallback_mode": fallback_mode,
            "storylines": storyline_rows,
            "episodes": episode_rows,
            "events": event_rows,
            "documents": [_safe_str(row.get("document_id")) for row in ranked_docs[: max(1, int(document_top_k or 6))]],
            "document_rows": ranked_docs,
            "evidence": evidence,
        }

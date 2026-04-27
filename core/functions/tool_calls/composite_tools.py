from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
from typing import Any, Dict, List, Optional, Tuple

from qwen_agent.tools.base import BaseTool, register_tool
from qwen_agent.utils.utils import logger

from core.utils.format import DOC_TYPE_META
from core.utils.format import correct_json_format
from .native_tools import SearchRelatedContentTool
from .sqldb_tools import SQLGetInteractionsByDocumentIDs
from .graphdb_tools import (
    _apply_llm_filter_to_rows,
    _as_props_dict,
    _clamp01,
    _hydrate_community_rows_from_summary_store,
    _format_section_related_entities,
    _to_bool,
    search_community_candidates,
    search_episode_storyline_candidates,
)


_GROUNDING_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "than", "that", "this", "those", "these",
    "to", "of", "in", "on", "at", "for", "from", "by", "with", "about", "into", "over", "after",
    "before", "under", "without", "within", "between", "through", "during", "because", "while",
    "is", "are", "was", "were", "be", "been", "being", "do", "does", "did", "done", "have", "has",
    "had", "can", "could", "would", "should", "may", "might", "must", "will", "shall", "not",
    "it", "its", "he", "she", "they", "them", "their", "his", "her", "hers", "him", "you", "your",
    "i", "we", "our", "ours", "me", "my", "mine", "who", "what", "when", "where", "why", "how",
    "which", "whom", "whose", "one", "two", "three", "four", "all", "some", "any", "each", "every",
    "more", "most", "less", "least", "very", "just", "only", "own", "same", "such", "other",
    "than", "too", "also", "again", "still", "even", "ever", "much", "many", "few", "little",
    "there", "here", "out", "off", "up", "down", "as", "so", "no", "nor",
}


def _dedup_strings(items: Any) -> List[str]:
    if isinstance(items, str):
        items = [items]
    if not isinstance(items, list):
        return []
    out: List[str] = []
    seen = set()
    for item in items:
        s = str(item or "").strip()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def _parse_related_content(raw: str) -> str:
    try:
        data = json.loads(correct_json_format(raw))
    except Exception:
        return ""
    val = data.get("related_content", "")
    return val.strip() if isinstance(val, str) else ""


def _clip_text(text: Any, limit: int = 320) -> str:
    raw = str(text or "").strip()
    if len(raw) <= limit:
        return raw
    return raw[: max(0, limit - 3)].rstrip() + "..."


def _tokenize_grounding_text(text: Any) -> List[str]:
    raw = str(text or "").strip().lower()
    if not raw:
        return []
    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9_-]*", raw)
    out: List[str] = []
    seen = set()
    for token in tokens:
        token = token.strip("_-")
        if len(token) <= 2 or token in _GROUNDING_STOPWORDS:
            continue
        if token.endswith("ing") and len(token) > 5:
            token = token[:-3]
        elif token.endswith("ed") and len(token) > 4:
            token = token[:-2]
        elif token.endswith("es") and len(token) > 4:
            token = token[:-2]
        elif token.endswith("s") and len(token) > 4:
            token = token[:-1]
        if len(token) <= 2 or token in _GROUNDING_STOPWORDS or token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


def _token_overlap_ratio(source_terms: List[str], target_text: Any) -> float:
    if not source_terms:
        return 0.0
    target_tokens = set(_tokenize_grounding_text(target_text))
    if not target_tokens:
        return 0.0
    hit_count = sum(1 for term in source_terms if term in target_tokens)
    return float(hit_count) / float(max(1, len(source_terms)))


def _is_structural_label(labels: Any) -> bool:
    clean = {str(x or "").strip() for x in (labels if isinstance(labels, list) else [labels]) if str(x or "").strip()}
    blocked = {"Event", "Episode", "Storyline", "Document", "Community", "Scene", "Chapter"}
    return bool(clean & blocked)


def _parse_mcq_query(query: str, explicit_options: Optional[List[Dict[str, Any]]] = None) -> Tuple[str, List[Dict[str, str]]]:
    text = str(query or "").strip()
    if explicit_options:
        options: List[Dict[str, str]] = []
        for item in explicit_options:
            if not isinstance(item, dict):
                continue
            label = str(item.get("label", "") or "").strip().upper()
            option_text = str(item.get("text", "") or "").strip()
            if label and option_text:
                options.append({"label": label, "text": option_text})
        return text, options

    normalized = text.replace("\r\n", "\n")
    stem = normalized
    options: List[Dict[str, str]] = []

    choices_match = re.search(r"\b(?:Choices|Options)\s*:\s*", normalized, flags=re.IGNORECASE)
    if choices_match:
        stem = normalized[: choices_match.start()].strip()
        options_block = normalized[choices_match.end() :].strip()
    else:
        inline_match = re.match(r"^(.*?)(\(\s*A\s*\)|A\.)", normalized, flags=re.IGNORECASE | re.DOTALL)
        if inline_match:
            stem = inline_match.group(1).strip()
            options_block = normalized[inline_match.start(2) :].strip()
        else:
            options_block = ""

    if options_block:
        patterns = [
            re.compile(r"(?:^|\n)\s*([A-D])[\.\):]\s*(.*?)(?=(?:\n\s*[A-D][\.\):]\s*)|\Z)", re.IGNORECASE | re.DOTALL),
            re.compile(r"\(\s*([A-D])\s*\)\s*(.*?)(?=(?:\(\s*[A-D]\s*\))|\Z)", re.IGNORECASE | re.DOTALL),
        ]
        for pattern in patterns:
            matches = list(pattern.finditer(options_block))
            if matches:
                seen = set()
                for m in matches:
                    label = str(m.group(1) or "").strip().upper()
                    option_text = " ".join(str(m.group(2) or "").split())
                    if label and option_text and label not in seen:
                        seen.add(label)
                        options.append({"label": label, "text": option_text})
                if options:
                    break

    return stem.strip(), options


def _format_option_rows(rows: List[Dict[str, Any]]) -> List[str]:
    if not rows:
        return ["None"]
    lines: List[str] = []
    for idx, row in enumerate(rows, 1):
        label = str(row.get("label", "") or "").strip()
        option_text = str(row.get("option_text", "") or "").strip()
        score = float(row.get("score", 0.0) or 0.0)
        support = float(row.get("support_probability", 0.0) or 0.0)
        lines.append(f"{idx}. {label}. {option_text} score={score:.4f} support={support:.4f}")
        rationale = str(row.get("rationale", "") or "").strip()
        if rationale:
            lines.append(f"   rationale: {rationale}")
        if row.get("section_name"):
            lines.append(f"   best_section: {row['section_name']}")
        if row.get("best_document_id"):
            lines.append(f"   best_document: {row['best_document_id']}")
        evidence = str(row.get("evidence", "") or "").strip()
        if evidence:
            lines.append(f"   evidence: {_clip_text(evidence, 260)}")
    return lines


def _entity_to_row(entity: Any, *, seed_score: float = 0.0) -> Dict[str, Any]:
    labels = getattr(entity, "type", []) or []
    if not isinstance(labels, list):
        labels = [labels]
    return {
        "id": getattr(entity, "id", "") or "",
        "name": getattr(entity, "name", "") or "(未命名)",
        "labels": labels,
        "description": getattr(entity, "description", "") or "",
        "source_documents": getattr(entity, "source_documents", []) or [],
        "properties": _as_props_dict(getattr(entity, "properties", {}) or {}),
        "keyword_score": 0.0,
        "matched_keyword_count": 0,
        "vector_score": 0.0,
        "score": _clamp01(seed_score),
    }


def _format_ranked_rows(title: str, rows: List[Dict[str, Any]], *, include_docs: bool = True) -> List[str]:
    if not rows:
        return [f"[{title}]", "None"]
    lines: List[str] = [f"[{title}]"]
    for idx, row in enumerate(rows, 1):
        rid = row.get("id") or "UNKNOWN_ID"
        name = row.get("name") or "(unnamed)"
        score = float(row.get("score", 0.0))
        lines.append(f"{idx}. {name} [ID: {rid}] score={score:.4f}")
        desc = str(row.get("description", "") or "").strip()
        if desc:
            lines.append(f"   description: {desc}")
        llm_prob = row.get("llm_probability")
        if llm_prob is not None:
            lines.append(f"   llm_relevance: {float(llm_prob):.4f}")
        if row.get("storyline_names"):
            lines.append(f"   storylines: {', '.join(row['storyline_names'])}")
        if row.get("episode_names"):
            lines.append(f"   episodes: {', '.join(row['episode_names'])}")
        if include_docs:
            source_docs = _dedup_strings(row.get("source_documents") or [])
            if source_docs:
                lines.append(f"   source_documents: {', '.join(source_docs)}")
    return lines


@register_tool("community_graphrag_search")
class CommunityGraphRAGSearch(BaseTool):
    name = "community_graphrag_search"
    description = (
        "按高层 Community report 逐层下钻到更细社区，再回到原文抽取证据，"
        "适合 GraphRAG 风格的主题级检索、跨事件聚合问答和高层语义导航。"
    )
    parameters = [
        {"name": "query", "type": "string", "description": "需要检索的自然语言问题。", "required": True},
        {"name": "community_top_k", "type": "integer", "description": "每层保留多少个 community，默认 3。", "required": False},
        {"name": "start_level", "type": "integer", "description": "从指定 community 层级开始检索；默认自动使用最高层。", "required": False},
        {"name": "max_depth", "type": "integer", "description": "最多向下细化多少层，默认 2。", "required": False},
        {"name": "child_limit_per_parent", "type": "integer", "description": "每个父 community 最多展开多少个子 community，默认 8。", "required": False},
        {"name": "use_llm_filter", "type": "bool", "description": "是否对 community 候选做 LLM 相关性过滤。", "required": False},
        {"name": "llm_filter_threshold", "type": "number", "description": "LLM 相关性保留阈值，默认 0.35。", "required": False},
        {"name": "document_top_k", "type": "integer", "description": "最多抽取多少个 document_id 的证据，默认 6。", "required": False},
        {"name": "max_evidence_length", "type": "integer", "description": "每个文档证据片段最大长度，默认 240。", "required": False},
        {"name": "member_preview_limit", "type": "integer", "description": "每个最终 community 展示多少个代表成员，默认 5。", "required": False},
    ]

    def __init__(
        self,
        graph_query_utils,
        document_vector_store,
        document_parser,
        *,
        community_summary_vector_store=None,
        embedding_config=None,
        max_workers: int = 4,
    ):
        self.graph_query_utils = graph_query_utils
        self.doc_vs = document_vector_store
        self.document_parser = document_parser
        self.community_summary_vector_store = community_summary_vector_store
        self.max_workers = max(1, int(max_workers or 4))
        self.search_related_content_tool = SearchRelatedContentTool(
            document_vector_store=self.doc_vs,
            document_parser=self.document_parser,
            max_workers=self.max_workers,
        )
        if embedding_config:
            self.graph_query_utils.load_embedding_model(embedding_config)

    def _expand_child_rows(self, parent_rows: List[Dict[str, Any]], *, child_limit_per_parent: int) -> List[Dict[str, Any]]:
        merged: Dict[str, Dict[str, Any]] = {}
        for parent in parent_rows:
            children = self.graph_query_utils.get_child_communities(
                str(parent.get("id") or ""),
                limit=child_limit_per_parent,
            ) or []
            for child in children:
                row = _entity_to_row(child, seed_score=float(parent.get("score", 0.0)) * 0.95)
                cid = row.get("id")
                if not cid:
                    continue
                merged.setdefault(cid, row)
                merged[cid]["score"] = max(float(merged[cid].get("score", 0.0)), float(row.get("score", 0.0)))
                merged[cid].setdefault("parent_communities", [])
                parent_name = str(parent.get("name") or "").strip()
                if parent_name and parent_name not in merged[cid]["parent_communities"]:
                    merged[cid]["parent_communities"].append(parent_name)
        rows = list(merged.values())
        return _hydrate_community_rows_from_summary_store(self.community_summary_vector_store, rows)

    def _collect_member_preview(self, community_rows: List[Dict[str, Any]], *, limit: int) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for row in community_rows:
            members = self.graph_query_utils.get_community_member_entities(
                str(row.get("id") or ""),
                limit=limit,
            ) or []
            names = []
            for member in members:
                name = str(getattr(member, "name", "") or "").strip()
                if name:
                    names.append(name)
            if not names:
                props = _as_props_dict(row.get("properties", {}) or {})
                names = [x for x in _dedup_strings(props.get("top_members") or []) if x][: max(1, int(limit or 5))]
            out.append(
                {
                    "community_id": str(row.get("id") or ""),
                    "community_name": str(row.get("name") or "").strip(),
                    "members": names[: max(1, int(limit or 5))],
                }
            )
        return out

    @staticmethod
    def _parse_evidence_output(raw: str) -> List[Tuple[str, str]]:
        text = str(raw or "").strip()
        if not text or text == "（无）":
            return []
        out: List[Tuple[str, str]] = []
        for line in text.splitlines():
            line = line.strip()
            if not line or ": " not in line:
                continue
            document_id, _, content = line.partition(": ")
            document_id = document_id.strip()
            content = content.strip()
            if document_id and content:
                out.append((document_id, content))
        return out

    def call(self, params: str, **kwargs) -> str:
        logger.info("🔎 调用 community_graphrag_search")
        try:
            data = json.loads(params) if isinstance(params, str) else dict(params or {})
        except Exception as e:
            return f"参数解析失败: {e}"

        query = str(data.get("query", "") or "").strip()
        if not query:
            return "query 不能为空。"

        community_top_k = max(1, int(data.get("community_top_k", 3) or 3))
        max_depth = max(1, int(data.get("max_depth", 2) or 2))
        child_limit_per_parent = max(1, int(data.get("child_limit_per_parent", 8) or 8))
        use_llm_filter = _to_bool(data.get("use_llm_filter"), True)
        llm_filter_threshold = float(data.get("llm_filter_threshold", 0.35) or 0.35)
        document_top_k = max(1, int(data.get("document_top_k", 6) or 6))
        max_evidence_length = max(80, int(data.get("max_evidence_length", 240) or 240))
        member_preview_limit = max(1, int(data.get("member_preview_limit", 5) or 5))

        levels = self.graph_query_utils.list_community_levels()
        if not levels:
            return "未找到Community。"
        start_level_raw = data.get("start_level")
        if isinstance(start_level_raw, (int, float, str)) and str(start_level_raw).strip():
            start_level = int(start_level_raw)
        else:
            start_level = max(levels)

        current_rows = search_community_candidates(
            self.graph_query_utils,
            summary_vector_store=self.community_summary_vector_store,
            query=query,
            top_k=community_top_k,
            level=start_level,
            vector_top_k=max(3, community_top_k),
            keyword_top_k=max(2, min(4, community_top_k)),
            use_llm_filter=use_llm_filter,
            llm_filter_top_k=max(8, community_top_k * 2),
            llm_filter_threshold=llm_filter_threshold,
            document_parser=self.document_parser,
            max_workers=self.max_workers,
        )
        if not current_rows:
            return "未找到相关 Community。"

        traversal: List[Tuple[int, List[Dict[str, Any]]]] = [(start_level, current_rows)]
        level_now = start_level
        while level_now > 1 and len(traversal) < max_depth:
            child_candidates = self._expand_child_rows(
                traversal[-1][1],
                child_limit_per_parent=child_limit_per_parent,
            )
            if not child_candidates:
                break
            if use_llm_filter:
                child_rows = _apply_llm_filter_to_rows(
                    child_candidates,
                    query=query,
                    document_parser=self.document_parser,
                    threshold=llm_filter_threshold,
                    top_k=community_top_k,
                    max_workers=self.max_workers,
                )
            else:
                child_rows = child_candidates[:community_top_k]
            if not child_rows:
                break
            level_now -= 1
            traversal.append((level_now, child_rows))

        final_rows = traversal[-1][1]
        member_preview = self._collect_member_preview(final_rows, limit=member_preview_limit)

        doc_score_map: Dict[str, float] = {}
        for row in final_rows:
            score = float(row.get("score", 0.0))
            for document_id in _dedup_strings(row.get("source_documents") or []):
                doc_score_map[document_id] = max(doc_score_map.get(document_id, 0.0), score)
        document_ids = [
            document_id
            for document_id, _ in sorted(doc_score_map.items(), key=lambda x: (-x[1], x[0]))[:document_top_k]
        ]

        evidence_rows: List[Tuple[str, str]] = []
        if document_ids:
            raw = self.search_related_content_tool.call(
                json.dumps(
                    {
                        "document_ids": document_ids,
                        "query": query,
                        "max_length": max_evidence_length,
                    },
                    ensure_ascii=False,
                )
            )
            evidence_rows = self._parse_evidence_output(raw)

        lines: List[str] = ["[Community GraphRAG]"]
        for level, rows in traversal:
            lines.extend(_format_ranked_rows(f"Communities@L{level}", rows))
        lines.append("[Member Preview]")
        if member_preview:
            for idx, item in enumerate(member_preview, 1):
                if item["members"]:
                    lines.append(f"{idx}. {item['community_name']} [{item['community_id']}]: {', '.join(item['members'])}")
                else:
                    lines.append(f"{idx}. {item['community_name']} [{item['community_id']}]: （无成员预览）")
        else:
            lines.append("None")

        lines.append("[Documents]")
        if document_ids:
            for idx, document_id in enumerate(document_ids, 1):
                lines.append(f"{idx}. {document_id}")
        else:
            lines.append("None")

        lines.append("[Evidence]")
        if evidence_rows:
            for idx, (document_id, content) in enumerate(evidence_rows, 1):
                lines.append(f"{idx}. {document_id}: {content}")
        else:
            lines.append("（无）")
        return "\n".join(lines)


@register_tool("section_evidence_search")
class SectionEvidenceSearch(BaseTool):
    name = "section_evidence_search"
    description = (
        "章节级保底证据检索：先找与问题最相关的场次或章节，再从原文抽取可直接回答问题的关键片段，"
        "适合场次定位、出现位置枚举、台词/道具/细节核对，以及其他检索路线没有直接给出答案时的兜底查证。"
    )
    parameters = [
        {"name": "query", "type": "string", "description": "需要定位章节与证据的自然语言问题。", "required": True},
        {"name": "section_top_k", "type": "integer", "description": "保留多少个相关章节，默认 8。", "required": False},
        {"name": "use_llm_filter", "type": "bool", "description": "是否对章节候选做 LLM 相关性过滤。", "required": False},
        {"name": "llm_filter_top_k", "type": "integer", "description": "进入 LLM 判断的章节候选数量，默认 12。", "required": False},
        {"name": "llm_filter_threshold", "type": "number", "description": "LLM 相关性保留阈值，默认 0.35。", "required": False},
        {"name": "max_length", "type": "integer", "description": "每个章节抽取片段的最大长度，默认 320。", "required": False},
        {"name": "related_entity_limit", "type": "integer", "description": "每种实体类型最多展示多少个章节关联实体，默认 3。", "required": False},
    ]

    def __init__(
        self,
        graph_query_utils,
        document_vector_store,
        document_parser,
        *,
        sentence_vector_store=None,
        doc_type: str = "general",
        embedding_config=None,
        max_workers: int = 4,
        section_retriever=None,
    ):
        self.graph_query_utils = graph_query_utils
        self.doc_vs = document_vector_store
        self.sent_vs = sentence_vector_store
        self.document_parser = document_parser
        self.doc_type = str(doc_type or "general").strip().lower() or "general"
        meta = DOC_TYPE_META.get(self.doc_type, DOC_TYPE_META["general"])
        self.section_label = str(meta.get("section_label", "Document")).strip() or "Document"
        self.max_workers = max(1, int(max_workers or 4))
        self.section_retriever = section_retriever
        self.search_related_content_tool = SearchRelatedContentTool(
            document_vector_store=self.doc_vs,
            document_parser=self.document_parser,
            max_workers=self.max_workers,
        )
        if embedding_config:
            self.graph_query_utils.load_embedding_model(embedding_config)

    @staticmethod
    def _parse_search_related_content_output(raw: str) -> List[Tuple[str, str]]:
        text = str(raw or "").strip()
        if not text or text == "（无）":
            return []
        rows: List[Tuple[str, str]] = []
        for line in text.splitlines():
            line = line.strip()
            if not line or ": " not in line:
                continue
            document_id, _, content = line.partition(": ")
            document_id = document_id.strip()
            content = content.strip()
            if document_id and content:
                rows.append((document_id, content))
        return rows

    def call(self, params: str, **kwargs) -> str:
        logger.info("🔎 调用 section_evidence_search")
        try:
            data = json.loads(params) if isinstance(params, str) else dict(params or {})
        except Exception as e:
            return f"参数解析失败: {e}"

        query = str(data.get("query", "") or "").strip()
        if not query:
            return "query 不能为空。"

        section_top_k = max(1, int(data.get("section_top_k", 8) or 8))
        use_llm_filter = _to_bool(data.get("use_llm_filter"), False)
        llm_filter_top_k = max(1, int(data.get("llm_filter_top_k", max(12, section_top_k)) or max(12, section_top_k)))
        llm_filter_threshold = float(data.get("llm_filter_threshold", 0.35) or 0.35)
        max_length = max(80, int(data.get("max_length", 320) or 320))
        related_entity_limit = max(1, int(data.get("related_entity_limit", 3) or 3))

        if self.section_retriever is not None:
            result = self.section_retriever.search(
                query,
                section_top_k=section_top_k,
                llm_filter_top_k=llm_filter_top_k,
                llm_filter_threshold=llm_filter_threshold,
                max_length=max_length,
            )
            section_rows = list(result.get("sections") or [])
            document_rows = list(result.get("documents") or [])
            evidence_rows = list(result.get("evidence") or [])
        else:
            section_rows = search_episode_storyline_candidates(
                self.graph_query_utils,
                label=self.section_label,
                query=query,
                top_k=section_top_k,
                vector_top_k=max(5, section_top_k),
                keyword_top_k=max(4, min(8, section_top_k)),
                use_llm_filter=use_llm_filter,
                llm_filter_top_k=llm_filter_top_k,
                llm_filter_threshold=llm_filter_threshold,
                document_parser=self.document_parser,
                max_workers=self.max_workers,
            )
            if not section_rows:
                return f"未找到相关{self.section_label}。"

            document_ids: List[str] = []
            for row in section_rows:
                document_ids.extend(_dedup_strings(row.get("source_documents") or []))
            document_ids = _dedup_strings(document_ids)

            evidence_rows = []
            document_rows = []
            if document_ids:
                raw = self.search_related_content_tool.call(
                    json.dumps(
                        {
                            "document_ids": document_ids,
                            "query": query,
                            "max_length": max_length,
                        },
                        ensure_ascii=False,
                    )
                )
                evidence_rows = self._parse_search_related_content_output(raw)

        if not section_rows:
            return f"未找到相关{self.section_label}。"

        lines: List[str] = [f"[Matched {self.section_label}]"]
        for idx, row in enumerate(section_rows, 1):
            section_id = str(row.get("id") or "").strip()
            section_name = str(row.get("name") or "").strip() or "(unnamed)"
            score = float(row.get("score", 0.0))
            lines.append(f"{idx}. {section_name} [ID: {section_id}] score={score:.4f}")
            desc = str(row.get("description", "") or "").strip()
            if desc:
                lines.append(f"   description: {desc}")
            llm_prob = row.get("llm_probability")
            if llm_prob is not None:
                lines.append(f"   llm_relevance: {float(llm_prob):.4f}")
            source_documents = _dedup_strings(row.get("source_documents") or [])
            if source_documents:
                lines.append(f"   source_documents: {', '.join(source_documents)}")
            lines.extend(
                _format_section_related_entities(
                    self.graph_query_utils,
                    section_id,
                    related_entity_limit=related_entity_limit,
                )
            )

        lines.append("[Documents]")
        if document_rows:
            for idx, row in enumerate(document_rows, 1):
                document_id = str(row.get("document_id") or "").strip()
                score = float(row.get("score", 0.0) or 0.0)
                doc_score = float(row.get("doc_score", 0.0) or 0.0)
                parent_score = float(row.get("parent_score", 0.0) or 0.0)
                hit_count = int(row.get("hit_count", 0) or 0)
                lines.append(
                    f"{idx}. {document_id} score={score:.4f} doc_score={doc_score:.4f} parent_score={parent_score:.4f} hits={hit_count}"
                )
        else:
            lines.append("None")
        lines.append("[Evidence]")
        if evidence_rows:
            for idx, (document_id, content) in enumerate(evidence_rows, 1):
                lines.append(f"{idx}. {document_id}: {content}")
        else:
            lines.append("None")
        return "\n".join(lines)


@register_tool("narrative_hierarchical_search")
class NarrativeHierarchicalSearch(BaseTool):
    name = "narrative_hierarchical_search"
    description = (
        "按 Storyline -> Episode -> Event 的层级先定位相关叙事节点，"
        "再回收到 source_documents 抽取原文证据。"
        "更适合回答需要定位相关情节阶段、场景衔接、事件链或跨段对应关系的问题；"
        "如果只是找一句直接事实，优先用句子或章节检索工具。"
        "当没有 Storyline 时，会自动降级到 Episode -> Event / source_documents。"
    )
    parameters = [
        {"name": "query", "type": "string", "description": "需要检索的剧情问题或自然语言查询。", "required": True},
        {"name": "storyline_top_k", "type": "integer", "description": "保留多少个相关 Storyline，默认 4。", "required": False},
        {"name": "episode_top_k", "type": "integer", "description": "保留多少个相关 Episode，默认 5。", "required": False},
        {"name": "event_top_k", "type": "integer", "description": "返回多少个相关 Event，默认 8。", "required": False},
        {"name": "document_top_k", "type": "integer", "description": "最多抽取多少个 document_id 的证据，默认 6。", "required": False},
        {"name": "llm_filter_threshold", "type": "number", "description": "层级过滤使用的相关性阈值，默认 0.35。", "required": False},
        {"name": "max_evidence_length", "type": "integer", "description": "每个文档证据片段的最大长度，默认 240。", "required": False},
    ]

    def __init__(
        self,
        graph_query_utils,
        document_vector_store,
        document_parser,
        *,
        sentence_vector_store=None,
        embedding_config=None,
        max_workers: int = 4,
        narrative_retriever=None,
    ):
        self.graph_query_utils = graph_query_utils
        self.doc_vs = document_vector_store
        self.sent_vs = sentence_vector_store
        self.document_parser = document_parser
        self.max_workers = max(1, int(max_workers or 4))
        self.narrative_retriever = narrative_retriever
        if embedding_config:
            self.graph_query_utils.load_embedding_model(embedding_config)

    def _collect_episode_rows(
        self,
        storyline_rows: List[Dict[str, Any]],
        *,
        limit_per_storyline: int,
    ) -> List[Dict[str, Any]]:
        merged: Dict[str, Dict[str, Any]] = {}
        for storyline in storyline_rows:
            related = self.graph_query_utils.search_related_entities(
                source_id=str(storyline.get("id") or ""),
                relation_types=["STORYLINE_CONTAINS"],
                entity_types=["Episode"],
                limit=limit_per_storyline,
                return_relations=False,
            ) or []
            for entity in related:
                row = _entity_to_row(entity, seed_score=float(storyline.get("score", 0.0)) * 0.9)
                eid = row["id"]
                if not eid:
                    continue
                merged.setdefault(eid, row)
                merged[eid]["score"] = max(float(merged[eid].get("score", 0.0)), float(row.get("score", 0.0)))
                merged[eid].setdefault("storyline_names", [])
                merged[eid].setdefault("storyline_ids", [])
                s_name = str(storyline.get("name") or "").strip()
                s_id = str(storyline.get("id") or "").strip()
                if s_name and s_name not in merged[eid]["storyline_names"]:
                    merged[eid]["storyline_names"].append(s_name)
                if s_id and s_id not in merged[eid]["storyline_ids"]:
                    merged[eid]["storyline_ids"].append(s_id)
        return list(merged.values())

    def _collect_event_rows(
        self,
        episode_rows: List[Dict[str, Any]],
        *,
        limit_per_episode: int,
        final_top_k: int,
    ) -> List[Dict[str, Any]]:
        merged: Dict[str, Dict[str, Any]] = {}
        for episode in episode_rows:
            related = self.graph_query_utils.search_related_entities(
                source_id=str(episode.get("id") or ""),
                relation_types=["EPISODE_CONTAINS"],
                entity_types=["Event"],
                limit=limit_per_episode,
                return_relations=False,
            ) or []
            for entity in related:
                row = _entity_to_row(entity, seed_score=float(episode.get("score", 0.0)))
                eid = row["id"]
                if not eid:
                    continue
                merged.setdefault(eid, row)
                merged[eid]["score"] = max(float(merged[eid].get("score", 0.0)), float(row.get("score", 0.0)))
                merged[eid].setdefault("episode_names", [])
                e_name = str(episode.get("name") or "").strip()
                if e_name and e_name not in merged[eid]["episode_names"]:
                    merged[eid]["episode_names"].append(e_name)
        rows = list(merged.values())
        rows.sort(key=lambda x: (x.get("score", 0.0), x.get("name", "")), reverse=True)
        return rows[: max(1, int(final_top_k or 8))]

    def _extract_document_evidence(
        self,
        *,
        document_ids: List[str],
        query: str,
        max_evidence_length: int,
    ) -> List[Tuple[str, str]]:
        docs = self.doc_vs.search_by_document_ids(document_ids, limit_per_document=6)
        grouped: Dict[str, List[str]] = {}
        for doc in docs or []:
            metadata = getattr(doc, "metadata", {}) or {}
            document_id = str(metadata.get("document_id") or "").strip()
            text = str(getattr(doc, "content", "") or "").strip()
            if document_id and text:
                grouped.setdefault(document_id, []).append(text)

        if not grouped:
            return []

        def _work(item: Tuple[str, List[str]]) -> Tuple[str, str]:
            document_id, parts = item
            raw = self.document_parser.search_related_content(
                text="\n".join(parts),
                goal=query,
                max_length=max_evidence_length,
            )
            return document_id, _parse_related_content(raw)

        ordered_items = [(document_id, grouped[document_id]) for document_id in document_ids if document_id in grouped]
        outputs: Dict[str, str] = {}
        worker_count = max(1, min(self.max_workers, len(ordered_items)))
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [executor.submit(_work, item) for item in ordered_items]
            for future in as_completed(futures):
                document_id, related = future.result()
                if related:
                    outputs[document_id] = related

        return [(document_id, outputs[document_id]) for document_id in document_ids if document_id in outputs]

    def call(self, params: str, **kwargs) -> str:
        logger.info("🔎 调用 narrative_hierarchical_search")
        try:
            data = json.loads(params) if isinstance(params, str) else dict(params or {})
        except Exception as e:
            return f"参数解析失败: {e}"

        query = str(data.get("query", "") or "").strip()
        if not query:
            return "query 不能为空。"

        storyline_top_k = max(1, int(data.get("storyline_top_k", 4) or 4))
        episode_top_k = max(1, int(data.get("episode_top_k", 5) or 5))
        event_top_k = max(1, int(data.get("event_top_k", 8) or 8))
        document_top_k = max(1, int(data.get("document_top_k", 6) or 6))
        max_evidence_length = max(80, int(data.get("max_evidence_length", 240) or 240))
        llm_filter_threshold = float(data.get("llm_filter_threshold", 0.35) or 0.35)

        if self.narrative_retriever is not None:
            result = self.narrative_retriever.search(
                query,
                storyline_top_k=storyline_top_k,
                episode_top_k=episode_top_k,
                event_top_k=event_top_k,
                document_top_k=document_top_k,
                llm_filter_threshold=llm_filter_threshold,
                max_evidence_length=max_evidence_length,
            )
            fallback_mode = bool(result.get("fallback_mode", False))
            storyline_rows = list(result.get("storylines") or [])
            episode_rows = list(result.get("episodes") or [])
            event_rows = list(result.get("events") or [])
            selected_document_ids = list(result.get("documents") or [])
            document_rows = list(result.get("document_rows") or [])
            evidences = list(result.get("evidence") or [])
            tree_paths = list(result.get("tree_paths") or [])
            search_mode = str(result.get("search_mode") or "").strip()
        else:
            storyline_rows = search_episode_storyline_candidates(
                self.graph_query_utils,
                label="Storyline",
                query=query,
                top_k=storyline_top_k,
                vector_top_k=max(3, storyline_top_k),
                keyword_top_k=max(2, min(4, storyline_top_k)),
                use_llm_filter=True,
                llm_filter_top_k=max(8, storyline_top_k * 2),
                llm_filter_threshold=llm_filter_threshold,
                document_parser=self.document_parser,
                max_workers=self.max_workers,
            )
            fallback_mode = not storyline_rows

            if fallback_mode:
                episode_rows = search_episode_storyline_candidates(
                    self.graph_query_utils,
                    label="Episode",
                    query=query,
                    top_k=episode_top_k,
                    vector_top_k=max(4, episode_top_k),
                    keyword_top_k=max(3, min(6, episode_top_k)),
                    use_llm_filter=True,
                    llm_filter_top_k=max(10, episode_top_k * 2),
                    llm_filter_threshold=llm_filter_threshold,
                    document_parser=self.document_parser,
                    max_workers=self.max_workers,
                )
            else:
                episode_candidates = self._collect_episode_rows(
                    storyline_rows,
                    limit_per_storyline=max(episode_top_k * 2, 8),
                )
                episode_rows = _apply_llm_filter_to_rows(
                    episode_candidates,
                    query=query,
                    document_parser=self.document_parser,
                    threshold=llm_filter_threshold,
                    top_k=episode_top_k,
                    max_workers=self.max_workers,
                ) if episode_candidates else []

            event_rows = self._collect_event_rows(
                episode_rows,
                limit_per_episode=max(event_top_k, 6),
                final_top_k=event_top_k,
            ) if episode_rows else []

            scored_document_ids: List[Tuple[str, float]] = []
            doc_score_map: Dict[str, float] = {}
            for row in episode_rows:
                row_score = float(row.get("score", 0.0))
                for document_id in _dedup_strings(row.get("source_documents") or []):
                    doc_score_map[document_id] = max(doc_score_map.get(document_id, 0.0), row_score)
            scored_document_ids = sorted(doc_score_map.items(), key=lambda x: (-x[1], x[0]))
            selected_document_ids = [document_id for document_id, _ in scored_document_ids[:document_top_k]]
            document_rows = [
                {
                    "document_id": document_id,
                    "score": score,
                    "doc_score": score,
                    "parent_score": score,
                    "hit_count": 0,
                }
                for document_id, score in scored_document_ids[:document_top_k]
            ]
            evidences = self._extract_document_evidence(
                document_ids=selected_document_ids,
                query=query,
                max_evidence_length=max_evidence_length,
            ) if selected_document_ids else []
            tree_paths = []
            search_mode = ""

        lines: List[str] = ["[Hierarchical Narrative Search]"]
        if search_mode:
            lines.append(f"mode: {search_mode}")
        if fallback_mode:
            lines.append("mode: no storyline found; downgraded to episode-level retrieval")
        lines.extend(_format_ranked_rows("Storylines", storyline_rows, include_docs=False))
        lines.extend(_format_ranked_rows("Episodes", episode_rows))
        lines.extend(_format_ranked_rows("Events", event_rows, include_docs=False))
        if tree_paths:
            lines.append("[Tree Paths]")
            for idx, path in enumerate(tree_paths[: max(1, min(8, document_top_k * 2))], 1):
                storyline = str(path.get("storyline_name") or path.get("storyline_id") or "").strip()
                episode = str(path.get("episode_name") or path.get("episode_id") or "").strip()
                event = str(path.get("event_name") or path.get("event_id") or "").strip()
                path_score = float(path.get("path_score", 0.0) or 0.0)
                docs = _dedup_strings(path.get("document_ids") or [])
                path_bits = [bit for bit in [storyline, episode, event] if bit]
                lines.append(f"{idx}. {' > '.join(path_bits) if path_bits else '(path)'} score={path_score:.4f}")
                if docs:
                    lines.append(f"   source_documents: {', '.join(docs[:6])}")
        lines.append("[Documents]")
        if document_rows:
            for idx, row in enumerate(document_rows, 1):
                document_id = str(row.get("document_id") or "").strip()
                score = float(row.get("score", 0.0) or 0.0)
                doc_score = float(row.get("doc_score", 0.0) or 0.0)
                parent_score = float(row.get("parent_score", 0.0) or 0.0)
                hit_count = int(row.get("hit_count", 0) or 0)
                lines.append(
                    f"{idx}. {document_id} score={score:.4f} doc_score={doc_score:.4f} parent_score={parent_score:.4f} hits={hit_count}"
                )
        elif selected_document_ids:
            for idx, document_id in enumerate(selected_document_ids, 1):
                lines.append(f"{idx}. {document_id}")
        else:
            lines.append("None")

        lines.append("[Evidence]")
        if evidences:
            for idx, (document_id, text) in enumerate(evidences, 1):
                lines.append(f"{idx}. {document_id}: {text}")
        else:
            lines.append("None")
        return "\n".join(lines)


@register_tool("narrative_causal_trace_search")
class NarrativeCausalTraceSearch(NarrativeHierarchicalSearch):
    name = "narrative_causal_trace_search"
    description = (
        "因果链专用检索：先定位相关 Episode/Event，再显式拉取 EPISODE_CAUSAL_LINK 的一跳上下游，"
        "回到原文证据，并用一次 LLM 调用压缩出候选原因、触发点、结果和证据。"
        "适合 why/how/what caused/what led to/动机/因果/策略变化/心理转变类开放问答；"
        "不适合只查一句事实或简单场景定位。"
    )
    parameters = [
        {"name": "query", "type": "string", "description": "需要追踪因果链的问题。", "required": True},
        {"name": "episode_top_k", "type": "integer", "description": "最多保留多少个相关 Episode，默认 8。", "required": False},
        {"name": "event_top_k", "type": "integer", "description": "最多展示多少个相关 Event，默认 8。", "required": False},
        {"name": "causal_link_top_k", "type": "integer", "description": "最多展示多少条候选因果 Episode 边，默认 8。", "required": False},
        {"name": "document_top_k", "type": "integer", "description": "最多抽取多少个 document_id 的证据，默认 6。", "required": False},
        {"name": "max_evidence_length", "type": "integer", "description": "每个文档证据片段的最大长度，默认 280。", "required": False},
        {"name": "use_llm_distill", "type": "bool", "description": "是否用 LLM 对因果证据做结构化压缩，默认 true。", "required": False},
    ]

    def __init__(
        self,
        graph_query_utils,
        document_vector_store,
        document_parser,
        *,
        sentence_vector_store=None,
        embedding_config=None,
        max_workers: int = 4,
        narrative_retriever=None,
        llm=None,
    ):
        super().__init__(
            graph_query_utils,
            document_vector_store,
            document_parser,
            sentence_vector_store=sentence_vector_store,
            embedding_config=embedding_config,
            max_workers=max_workers,
            narrative_retriever=narrative_retriever,
        )
        self.llm = llm or (getattr(document_parser, "llm", None) if document_parser is not None else None)

    def _fetch_causal_edges(self, seed_episode_ids: List[str], *, query: str, limit: int) -> List[Dict[str, Any]]:
        seed_set = {str(x or "").strip() for x in seed_episode_ids if str(x or "").strip()}
        query_terms = _tokenize_grounding_text(query)
        edge_rows: List[Dict[str, Any]] = []
        iter_edges = getattr(self.graph_query_utils, "_iter_edges", None)
        if iter_edges is None:
            return []

        for src, dst, key, data in iter_edges():
            pred = str((data or {}).get("predicate") or (data or {}).get("relation_type") or "").strip()
            if pred != "EPISODE_CAUSAL_LINK":
                continue
            src_id = str((data or {}).get("subject_id") or src or "").strip()
            dst_id = str((data or {}).get("object_id") or dst or "").strip()
            if seed_set and src_id not in seed_set and dst_id not in seed_set:
                continue
            src_ent = self.graph_query_utils.get_entity_by_id(src_id)
            dst_ent = self.graph_query_utils.get_entity_by_id(dst_id)
            src_name = str(getattr(src_ent, "name", "") or src_id)
            dst_name = str(getattr(dst_ent, "name", "") or dst_id)
            src_desc = str(getattr(src_ent, "description", "") or "")
            dst_desc = str(getattr(dst_ent, "description", "") or "")
            desc = str((data or {}).get("description") or "")
            props = _as_props_dict((data or {}).get("properties", {}) or {})
            relation_type = str(props.get("relation_type") or (data or {}).get("relation_type") or pred)
            overlap_text = " ".join([src_name, dst_name, src_desc, dst_desc, desc, relation_type])
            overlap = _token_overlap_ratio(query_terms, overlap_text)
            seed_bonus = 0.0
            if src_id in seed_set:
                seed_bonus += 0.35
            if dst_id in seed_set:
                seed_bonus += 0.35
            try:
                confidence = float((data or {}).get("confidence", 1.0) or 1.0)
            except Exception:
                confidence = 1.0
            score = min(1.0, seed_bonus + 0.45 * overlap + 0.20 * max(0.0, min(1.0, confidence)))
            edge_rows.append(
                {
                    "id": str((data or {}).get("id") or key or ""),
                    "source_id": src_id,
                    "target_id": dst_id,
                    "source_name": src_name,
                    "target_name": dst_name,
                    "relation_type": relation_type,
                    "description": desc,
                    "confidence": confidence,
                    "score": score,
                    "source_documents": _dedup_strings((data or {}).get("source_documents") or props.get("source_documents") or []),
                }
            )

        edge_rows.sort(key=lambda row: (float(row.get("score", 0.0) or 0.0), float(row.get("confidence", 0.0) or 0.0)), reverse=True)
        return edge_rows[: max(1, int(limit or 8))]

    def _distill_causal_trace(
        self,
        *,
        query: str,
        episode_rows: List[Dict[str, Any]],
        causal_edges: List[Dict[str, Any]],
        evidences: List[Tuple[str, str]],
    ) -> List[Dict[str, Any]]:
        if self.llm is None or not (causal_edges or evidences or episode_rows):
            return []

        episode_block = "\n".join(
            f"- {row.get('name') or row.get('id')} [{row.get('id')}]: {_clip_text(row.get('description', ''), 240)}"
            for row in episode_rows[:8]
        )
        edge_block = "\n".join(
            (
                f"- {row.get('source_name')} -> {row.get('target_name')} "
                f"type={row.get('relation_type')} confidence={float(row.get('confidence', 0.0) or 0.0):.2f}; "
                f"{_clip_text(row.get('description', ''), 260)}"
            )
            for row in causal_edges[:10]
        )
        evidence_block = "\n".join(
            f"- {document_id}: {_clip_text(text, 420)}"
            for document_id, text in evidences[:8]
        )
        prompt = (
            "You are a screenplay QA retrieval tool. Distill causal evidence for the user question.\n"
            "Prefer local triggers, explicit motivations, immediate consequences, and later revelations that explain earlier behavior.\n"
            "Do not invent facts beyond the supplied episodes, causal links, and source evidence.\n\n"
            f"Question:\n{query}\n\n"
            f"Candidate episodes:\n{episode_block or '(none)'}\n\n"
            f"Episode causal links:\n{edge_block or '(none)'}\n\n"
            f"Source evidence:\n{evidence_block or '(none)'}\n\n"
            "Return strict JSON only:\n"
            "{\n"
            '  "causal_traces": [\n'
            "    {\n"
            '      "candidate_cause": "brief cause or motivation",\n'
            '      "anchor_or_effect": "effect, decision, behavior, or reveal being explained",\n'
            '      "causal_relation_type": "cause|motivation|trigger|consequence|reveal|contrast|unknown",\n'
            '      "source_document_id": "document id if known",\n'
            '      "local_quote": "short evidence quote or paraphrase",\n'
            '      "why_relevant": "why this helps answer the question"\n'
            "    }\n"
            "  ]\n"
            "}\n"
        )
        try:
            responses = self.llm.run([{"role": "user", "content": prompt}])
            raw = ""
            if isinstance(responses, list) and responses:
                raw = str((responses[-1] or {}).get("content", "") or "").strip()
            parsed = json.loads(correct_json_format(raw))
        except Exception:
            return []
        traces = parsed.get("causal_traces", [])
        return traces if isinstance(traces, list) else []

    def call(self, params: str, **kwargs) -> str:
        logger.info("🔎 调用 narrative_causal_trace_search")
        try:
            data = json.loads(params) if isinstance(params, str) else dict(params or {})
        except Exception as e:
            return f"参数解析失败: {e}"

        query = str(data.get("query", "") or "").strip()
        if not query:
            return "query 不能为空。"

        episode_top_k = max(1, int(data.get("episode_top_k", 8) or 8))
        event_top_k = max(1, int(data.get("event_top_k", 8) or 8))
        causal_link_top_k = max(1, int(data.get("causal_link_top_k", 8) or 8))
        document_top_k = max(1, int(data.get("document_top_k", 6) or 6))
        max_evidence_length = max(100, int(data.get("max_evidence_length", 280) or 280))
        use_llm_distill = _to_bool(data.get("use_llm_distill"), True)

        if self.narrative_retriever is not None:
            result = self.narrative_retriever.search(
                query,
                storyline_top_k=2,
                episode_top_k=episode_top_k,
                event_top_k=event_top_k,
                document_top_k=document_top_k,
                llm_filter_threshold=0.25,
                max_evidence_length=max_evidence_length,
            )
            episode_rows = list(result.get("episodes") or [])
            event_rows = list(result.get("events") or [])
            selected_document_ids = _dedup_strings(result.get("documents") or [])
            evidences = list(result.get("evidence") or [])
        else:
            episode_rows = search_episode_storyline_candidates(
                self.graph_query_utils,
                label="Episode",
                query=query,
                top_k=episode_top_k,
                vector_top_k=max(episode_top_k, 10),
                keyword_top_k=max(6, min(12, episode_top_k * 2)),
                use_llm_filter=True,
                llm_filter_top_k=max(12, episode_top_k * 2),
                llm_filter_threshold=0.25,
                document_parser=self.document_parser,
                max_workers=self.max_workers,
            )
            event_rows = self._collect_event_rows(
                episode_rows,
                limit_per_episode=max(event_top_k, 6),
                final_top_k=event_top_k,
            ) if episode_rows else []
            selected_document_ids = []
            evidences = []

        seed_episode_ids = [str(row.get("id") or "").strip() for row in episode_rows if str(row.get("id") or "").strip()]
        causal_edges = self._fetch_causal_edges(seed_episode_ids, query=query, limit=causal_link_top_k)

        doc_ids = []
        for row in episode_rows:
            doc_ids.extend(_dedup_strings(row.get("source_documents") or []))
        for row in causal_edges:
            doc_ids.extend(_dedup_strings(row.get("source_documents") or []))
        doc_ids.extend(selected_document_ids)
        doc_ids = _dedup_strings(doc_ids)[:document_top_k]

        if doc_ids:
            focused_query = (
                f"{query}\n"
                "Find the concrete cause, motivation, trigger, consequence, and local evidence relevant to this question."
            )
            evidences = self._extract_document_evidence(
                document_ids=doc_ids,
                query=focused_query,
                max_evidence_length=max_evidence_length,
            ) or evidences

        traces = self._distill_causal_trace(
            query=query,
            episode_rows=episode_rows,
            causal_edges=causal_edges,
            evidences=evidences,
        ) if use_llm_distill else []

        lines: List[str] = ["[Narrative Causal Trace Search]"]
        lines.extend(_format_ranked_rows("Candidate Episodes", episode_rows[:episode_top_k]))
        lines.extend(_format_ranked_rows("Candidate Events", event_rows[:event_top_k], include_docs=False))
        lines.append("[Episode Causal Links]")
        if causal_edges:
            for idx, row in enumerate(causal_edges, 1):
                lines.append(
                    f"{idx}. {row.get('source_name')} [{row.get('source_id')}] -> "
                    f"{row.get('target_name')} [{row.get('target_id')}] "
                    f"type={row.get('relation_type')} score={float(row.get('score', 0.0) or 0.0):.4f}"
                )
                desc = str(row.get("description") or "").strip()
                if desc:
                    lines.append(f"   description: {desc}")
                docs = _dedup_strings(row.get("source_documents") or [])
                if docs:
                    lines.append(f"   source_documents: {', '.join(docs)}")
        else:
            lines.append("None")

        lines.append("[Documents]")
        if doc_ids:
            for idx, document_id in enumerate(doc_ids, 1):
                lines.append(f"{idx}. {document_id}")
        else:
            lines.append("None")

        lines.append("[Evidence]")
        if evidences:
            for idx, (document_id, text) in enumerate(evidences, 1):
                lines.append(f"{idx}. {document_id}: {text}")
        else:
            lines.append("None")

        lines.append("[LLM Causal Distillation]")
        if traces:
            for idx, item in enumerate(traces[:6], 1):
                if not isinstance(item, dict):
                    continue
                lines.append(f"{idx}. cause: {str(item.get('candidate_cause') or '').strip()}")
                lines.append(f"   effect: {str(item.get('anchor_or_effect') or '').strip()}")
                lines.append(f"   type: {str(item.get('causal_relation_type') or '').strip()}")
                source_document_id = str(item.get("source_document_id") or "").strip()
                if source_document_id:
                    lines.append(f"   source_document_id: {source_document_id}")
                quote = str(item.get("local_quote") or "").strip()
                if quote:
                    lines.append(f"   evidence: {quote}")
                why_relevant = str(item.get("why_relevant") or "").strip()
                if why_relevant:
                    lines.append(f"   why_relevant: {why_relevant}")
        else:
            lines.append("None")
        return "\n".join(lines)


@register_tool("hybrid_evidence_search")
class HybridEvidenceSearch(BaseTool):
    name = "hybrid_evidence_search"
    description = (
        "Hybrid evidence search: combines BM25, section evidence, sentence vectors, document vectors, "
        "and raw parent document expansion into one local evidence panel. Use it for direct passage "
        "grounding when lexical and semantic recall should be combined before answering."
    )
    parameters = [
        {"name": "query", "type": "string", "description": "需要检索证据的问题或查询。", "required": True},
        {"name": "top_k", "type": "integer", "description": "每路检索保留多少条候选，默认 5。", "required": False},
        {"name": "max_candidates", "type": "integer", "description": "进入最终 LLM 判别的候选片段数，默认 20。", "required": False},
        {"name": "max_candidate_chars", "type": "integer", "description": "每个候选片段最大字符数，默认 1800。", "required": False},
        {"name": "use_llm_plan", "type": "bool", "description": "是否先用 LLM 生成局部检索计划，默认 true。", "required": False},
        {"name": "use_llm_judge", "type": "bool", "description": "是否让工具内部选择最终答案，默认 false。", "required": False},
        {"name": "use_related_content", "type": "bool", "description": "是否额外用 LLM 从召回文档中摘取 related_content，默认 false。", "required": False},
    ]

    def __init__(
        self,
        graph_query_utils,
        document_vector_store,
        document_parser,
        *,
        sentence_vector_store=None,
        doc_type: str = "general",
        embedding_config=None,
        max_workers: int = 4,
        section_retriever=None,
        bm25_tool=None,
        llm=None,
    ):
        self.graph_query_utils = graph_query_utils
        self.doc_vs = document_vector_store
        self.sent_vs = sentence_vector_store
        self.document_parser = document_parser
        self.doc_type = str(doc_type or "general").strip().lower() or "general"
        self.max_workers = max(1, int(max_workers or 4))
        self.section_retriever = section_retriever
        self.bm25_tool = bm25_tool
        self.llm = llm or (getattr(document_parser, "llm", None) if document_parser is not None else None)
        self.section_tool = SectionEvidenceSearch(
            graph_query_utils,
            document_vector_store,
            document_parser,
            sentence_vector_store=sentence_vector_store,
            doc_type=doc_type,
            embedding_config=embedding_config,
            max_workers=max_workers,
            section_retriever=section_retriever,
        )
        self.search_related_content_tool = SearchRelatedContentTool(
            document_vector_store=document_vector_store,
            document_parser=document_parser,
            max_workers=max_workers,
        )

    @staticmethod
    def _extract_capitalized_phrases(text: str, *, limit: int = 8) -> List[str]:
        phrases: List[str] = []
        for match in re.finditer(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}\b", str(text or "")):
            phrase = match.group(0).strip()
            if phrase in {"What", "Why", "Who", "Where", "When", "How", "Which"}:
                continue
            if phrase not in phrases:
                phrases.append(phrase)
            if len(phrases) >= limit:
                break
        return phrases

    def _build_query_plan(self, query: str, *, use_llm_plan: bool) -> Dict[str, Any]:
        entities = self._extract_capitalized_phrases(query)
        default_queries = _dedup_strings(
            [
                query,
                f"{query} exact local dialogue action evidence",
                f"{' '.join(entities)} evidence context".strip(),
                f"{' '.join(entities)} conversation action response detail".strip(),
            ]
        )[:4]
        plan = {
            "information_need": query,
            "must_keep_anchors": entities,
            "reject_if": ["evidence describes a similar but different scene or later plot point"],
            "retrieval_queries": default_queries,
        }
        if not use_llm_plan or self.llm is None:
            return plan
        prompt = (
            "Create a hybrid local evidence retrieval plan for a screenplay QA tool.\n"
            "The task is to find evidence that directly answers the question.\n"
            "Depending on the question, useful evidence may be a direct fact, a scene-local action, "
            "a stated motivation, a causal trigger, a preceding event, a response, or a contrastive near miss.\n"
            "Do not guess the answer. Extract constraints that retrieved evidence must satisfy.\n"
            "If the question names a person, conversation, place, or outcome, keep those as hard anchors.\n"
            "Return strict JSON only:\n"
            "{\n"
            '  "information_need": "what information needs evidence",\n'
            '  "must_keep_anchors": ["names, places, conversation anchors, or outcome words"],\n'
            '  "reject_if": ["conditions that mean a candidate is a similar but wrong scene"],\n'
            '  "retrieval_queries": ["3-5 short lexical/semantic searches"]\n'
            "}\n\n"
            f"Question:\n{query}\n"
        )
        try:
            responses = self.llm.run([{"role": "user", "content": prompt}])
            raw = ""
            if isinstance(responses, list) and responses:
                raw = str((responses[-1] or {}).get("content", "") or "").strip()
            parsed = json.loads(correct_json_format(raw))
            if isinstance(parsed, dict):
                queries = _dedup_strings(parsed.get("retrieval_queries") or [])
                if queries:
                    parsed["retrieval_queries"] = _dedup_strings([query, *queries])[:3]
                    return {**plan, **parsed}
        except Exception:
            return plan
        return plan

    def _augment_retrieval_queries(self, query: str, queries: List[str]) -> List[str]:
        return _dedup_strings([query, *queries])[:6]

    @staticmethod
    def _doc_to_candidate(doc: Any, *, source: str, query: str = "") -> Dict[str, Any]:
        metadata = getattr(doc, "metadata", {}) or {}
        content = str(getattr(doc, "content", "") or getattr(doc, "page_content", "") or "").strip()
        document_id = str(metadata.get("document_id") or metadata.get("chunk_id") or getattr(doc, "id", "") or "").strip()
        try:
            score = float(metadata.get("similarity_score", 0.0) or 0.0)
        except Exception:
            score = 0.0
        if query:
            score += 0.35 * _token_overlap_ratio(_tokenize_grounding_text(query), content)
        return {
            "source": source,
            "document_id": document_id,
            "content": content,
            "score": score,
            "metadata": metadata,
        }

    @staticmethod
    def _parse_bm25_output(raw: str) -> List[Dict[str, Any]]:
        text = str(raw or "").strip()
        if not text or text == "No results.":
            return []
        blocks = re.split(r"\n---\n", text)
        out: List[Dict[str, Any]] = []
        for block in blocks:
            cleaned = block.strip()
            if not cleaned:
                continue
            document_id = ""
            m = re.search(r"^\s*(?:-\s*)?document_id:\s*(\S+)\s*$", cleaned, flags=re.MULTILINE)
            if m:
                document_id = m.group(1).strip()
            out.append(
                {
                    "source": "bm25",
                    "document_id": document_id,
                    "content": cleaned,
                    "score": 0.85,
                    "metadata": {},
                }
            )
        return out

    @staticmethod
    def _parse_section_output(raw: str) -> List[Dict[str, Any]]:
        text = str(raw or "").strip()
        if not text:
            return []
        out: List[Dict[str, Any]] = []
        evidence_idx = text.find("[Evidence]")
        if evidence_idx >= 0:
            for line in text[evidence_idx:].splitlines():
                m = re.match(r"\s*\d+\.\s*([^:]+):\s*(.+)$", line)
                if not m:
                    continue
                out.append(
                    {
                        "source": "section_evidence",
                        "document_id": m.group(1).strip(),
                        "content": m.group(2).strip(),
                        "score": 0.75,
                        "metadata": {},
                    }
                )
        doc_ids: List[str] = []
        for line in text.splitlines():
            if "source_documents:" not in line:
                continue
            _, _, rest = line.partition("source_documents:")
            doc_ids.extend(_dedup_strings([x.strip() for x in rest.split(",")]))
        for document_id in doc_ids[:8]:
            out.append(
                {
                    "source": "section_document",
                    "document_id": document_id,
                    "content": "",
                    "score": 0.45,
                    "metadata": {},
                }
            )
        return out

    def _collect_candidates(self, *, query: str, queries: List[str], top_k: int, use_related_content: bool) -> List[Dict[str, Any]]:
        candidates: List[Dict[str, Any]] = []

        def run_one(search_query: str, *, use_section: bool) -> List[Dict[str, Any]]:
            rows: List[Dict[str, Any]] = []
            if self.bm25_tool is not None:
                try:
                    raw = self.bm25_tool.call(json.dumps({"query": search_query, "k": top_k}, ensure_ascii=False))
                    rows.extend(self._parse_bm25_output(raw))
                except Exception:
                    pass
            if use_section:
                try:
                    raw = self.section_tool.call(
                        json.dumps(
                            {
                                "query": search_query,
                                "section_top_k": max(3, min(8, top_k)),
                                "llm_filter_top_k": max(8, top_k * 2),
                                "max_length": 420,
                            },
                            ensure_ascii=False,
                        )
                    )
                    rows.extend(self._parse_section_output(raw))
                except Exception:
                    pass
            if self.sent_vs is not None:
                try:
                    rows.extend(
                        self._doc_to_candidate(doc, source="sentence_vector", query=search_query)
                        for doc in (self.sent_vs.search(query=search_query, limit=top_k) or [])
                    )
                except Exception:
                    pass
            if self.doc_vs is not None:
                try:
                    rows.extend(
                        self._doc_to_candidate(doc, source="document_vector", query=search_query)
                        for doc in (self.doc_vs.search(query=search_query, limit=max(2, min(4, top_k))) or [])
                    )
                except Exception:
                    pass
            return rows

        worker_count = max(1, min(self.max_workers, len(queries)))
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [executor.submit(run_one, q, use_section=(idx == 0)) for idx, q in enumerate(queries)]
            for future in as_completed(futures):
                candidates.extend(future.result())

        doc_ids = _dedup_strings([row.get("document_id") for row in candidates if row.get("document_id")])
        if doc_ids:
            try:
                for doc in self.doc_vs.search_by_document_ids(doc_ids[:24], limit_per_document=2) or []:
                    row = self._doc_to_candidate(doc, source="raw_document", query=query)
                    if row.get("content"):
                        row["score"] = float(row.get("score", 0.0) or 0.0) + 1.75
                        candidates.append(row)
            except Exception:
                pass
            if use_related_content:
                try:
                    raw = self.search_related_content_tool.call(
                        json.dumps(
                            {
                                "document_ids": doc_ids[:12],
                                "query": query,
                                "max_length": 520,
                            },
                            ensure_ascii=False,
                        )
                    )
                    for did, content in SectionEvidenceSearch._parse_search_related_content_output(raw):
                        candidates.append(
                            {
                                "source": "related_content",
                                "document_id": did,
                                "content": content,
                                "score": 0.80,
                                "metadata": {},
                            }
                        )
                except Exception:
                    pass
        elif use_related_content:
            try:
                raw = self.search_related_content_tool.call(
                    json.dumps(
                        {
                            "document_ids": doc_ids[:12],
                            "query": query,
                            "max_length": 520,
                        },
                        ensure_ascii=False,
                    )
                )
                for did, content in SectionEvidenceSearch._parse_search_related_content_output(raw):
                    candidates.append(
                        {
                            "source": "related_content",
                            "document_id": did,
                            "content": content,
                            "score": 0.80,
                            "metadata": {},
                        }
                    )
            except Exception:
                pass

        merged: Dict[Tuple[str, str], Dict[str, Any]] = {}
        query_terms = _tokenize_grounding_text(query)
        for row in candidates:
            content = str(row.get("content") or "").strip()
            document_id = str(row.get("document_id") or "").strip()
            if not content and document_id:
                continue
            if not content:
                continue
            key = (document_id, content[:180])
            score = float(row.get("score", 0.0) or 0.0) + _token_overlap_ratio(query_terms, content)
            if key not in merged or score > float(merged[key].get("score", 0.0) or 0.0):
                row["score"] = score
                merged[key] = row
        rows = list(merged.values())
        rows.sort(key=lambda x: float(x.get("score", 0.0) or 0.0), reverse=True)
        return rows

    def _judge_candidates(
        self,
        *,
        query: str,
        plan: Dict[str, Any],
        candidates: List[Dict[str, Any]],
        max_candidate_chars: int,
    ) -> Dict[str, Any]:
        if self.llm is None:
            return {}
        blocks: List[str] = []
        for idx, row in enumerate(candidates, 1):
            blocks.append(
                "\n".join(
                    [
                        f"[{idx}] source={row.get('source')} document_id={row.get('document_id')}",
                        _clip_text(row.get("content", ""), max_candidate_chars),
                    ]
                )
            )
        prompt = (
            "You are a strict hybrid evidence judge for screenplay QA.\n"
            "Answer only from the candidate evidence below.\n"
            "Choose evidence that satisfies ALL hard anchors in the question. Reject similar but different scenes.\n"
            "Prefer raw_document or bm25 candidates when they contain concrete dialogue/action; related_content and section_evidence may be summaries.\n"
            "Prefer evidence that directly answers the user's query over broad summaries or adjacent-but-different scenes.\n"
            "For explanatory questions, the best evidence may be a concrete cause, trigger, motivation, immediately preceding event, or response; infer this from the question as a whole, not from keyword matching.\n"
            "If multiple candidates are plausible, report the strongest direct support and the most important near misses.\n\n"
            f"Question:\n{query}\n\n"
            f"Information need:\n{plan.get('information_need') or plan.get('target_effect')}\n"
            f"Hard anchors:\n{json.dumps(plan.get('must_keep_anchors') or [], ensure_ascii=False)}\n"
            f"Reject if:\n{json.dumps(plan.get('reject_if') or [], ensure_ascii=False)}\n\n"
            "Candidate evidence:\n"
            + "\n\n".join(blocks)
            + "\n\nReturn strict JSON only:\n"
            "{\n"
            '  "answer_candidate": "short direct answer",\n'
            '  "confidence": 0.0,\n'
            '  "source_document_ids": ["..."],\n'
            '  "supporting_evidence": [{"candidate_index": 1, "quote": "short quote/paraphrase"}],\n'
            '  "plausible_answers": [{"candidate_index": 1, "answer": "alternative answer candidate", "why_plausible": "brief"}],\n'
            '  "why_this_matches": "why it satisfies the question anchors",\n'
            '  "rejected_near_misses": [{"candidate_index": 2, "reason": "why similar but wrong"}]\n'
            "}\n"
        )
        try:
            responses = self.llm.run([{"role": "user", "content": prompt}])
            raw = ""
            if isinstance(responses, list) and responses:
                raw = str((responses[-1] or {}).get("content", "") or "").strip()
            parsed = json.loads(correct_json_format(raw))
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}

    def call(self, params: str, **kwargs) -> str:
        logger.info("🔎 调用 hybrid_evidence_search")
        try:
            data = json.loads(params) if isinstance(params, str) else dict(params or {})
        except Exception as e:
            return f"参数解析失败: {e}"

        query = str(data.get("query", "") or "").strip()
        if not query:
            return "query 不能为空。"
        top_k = max(2, int(data.get("top_k", 5) or 5))
        max_candidates = max(4, int(data.get("max_candidates", 20) or 20))
        max_candidate_chars = max(240, int(data.get("max_candidate_chars", 1800) or 1800))
        use_llm_plan = _to_bool(data.get("use_llm_plan"), True)
        use_llm_judge = _to_bool(data.get("use_llm_judge"), False)
        use_related_content = _to_bool(data.get("use_related_content"), False)

        plan = self._build_query_plan(query, use_llm_plan=use_llm_plan)
        queries = self._augment_retrieval_queries(query, _dedup_strings(plan.get("retrieval_queries") or [query]))
        candidates = self._collect_candidates(
            query=query,
            queries=queries,
            top_k=top_k,
            use_related_content=use_related_content,
        )
        selected = candidates[:max_candidates]
        judged = self._judge_candidates(
            query=query,
            plan=plan,
            candidates=selected,
            max_candidate_chars=max_candidate_chars,
        ) if selected and use_llm_judge else {}

        lines: List[str] = ["[Hybrid Evidence Search]"]
        lines.append(f"information_need: {plan.get('information_need') or plan.get('target_effect') or query}")
        anchors = _dedup_strings(plan.get("must_keep_anchors") or [])
        if anchors:
            lines.append(f"hard_anchors: {', '.join(anchors)}")
        reject_if = _dedup_strings(plan.get("reject_if") or [])
        if reject_if:
            lines.append("reject_if: " + " | ".join(reject_if))
        lines.append("[Retrieval Queries]")
        for idx, q in enumerate(queries, 1):
            lines.append(f"{idx}. {q}")
        lines.append("[Candidate Evidence]")
        if selected:
            for idx, row in enumerate(selected, 1):
                lines.append(
                    f"{idx}. source={row.get('source')} document_id={row.get('document_id')} "
                    f"score={float(row.get('score', 0.0) or 0.0):.4f}"
                )
                lines.append(f"   {_clip_text(row.get('content', ''), max_candidate_chars)}")
        else:
            lines.append("None")
        lines.append("[Hybrid Evidence Judgment]")
        if judged:
            lines.append(f"answer_candidate: {str(judged.get('answer_candidate') or '').strip()}")
            lines.append(f"confidence: {float(judged.get('confidence', 0.0) or 0.0):.3f}")
            doc_ids = _dedup_strings(judged.get("source_document_ids") or [])
            if doc_ids:
                lines.append(f"source_document_ids: {', '.join(doc_ids)}")
            support = judged.get("supporting_evidence") or []
            if isinstance(support, list) and support:
                lines.append("supporting_evidence:")
                for item in support[:4]:
                    if not isinstance(item, dict):
                        continue
                    lines.append(
                        f"  - candidate={item.get('candidate_index')} quote={str(item.get('quote') or '').strip()}"
                    )
            plausible = judged.get("plausible_answers") or judged.get("plausible_causes") or []
            if isinstance(plausible, list) and plausible:
                lines.append("plausible_answers:")
                for item in plausible[:5]:
                    if not isinstance(item, dict):
                        continue
                    lines.append(
                        f"  - candidate={item.get('candidate_index')} answer={str(item.get('answer') or item.get('cause') or '').strip()} why={str(item.get('why_plausible') or '').strip()}"
                    )
            why = str(judged.get("why_this_matches") or "").strip()
            if why:
                lines.append(f"why_this_matches: {why}")
            rejected = judged.get("rejected_near_misses") or []
            if isinstance(rejected, list) and rejected:
                lines.append("rejected_near_misses:")
                for item in rejected[:4]:
                    if not isinstance(item, dict):
                        continue
                    lines.append(
                        f"  - candidate={item.get('candidate_index')} reason={str(item.get('reason') or '').strip()}"
                    )
        else:
            lines.append("disabled; use the candidate evidence panel above and answer with the closest local trigger.")
        return "\n".join(lines)


@register_tool("choice_grounded_evidence_search")
class ChoiceGroundedEvidenceSearch(BaseTool):
    name = "choice_grounded_evidence_search"
    description = (
        "面向多选题的逐选项证据比较工具。它会把每个选项分别当成候选答案，并发检索章节、叙事、句子和文档证据，"
        "输出每个选项的支持度、区分度和核心证据，再给出推荐选项。"
        "特别适合 why / implication / attitude / except / not / least-supported 这类必须逐项比较的选择题。"
    )
    parameters = [
        {"name": "query", "type": "string", "description": "完整问题文本；推荐带 Choices/Options 或 (A)(B)(C)(D) 选项。", "required": True},
        {"name": "options", "type": "array", "description": "可选；显式传入选项列表，每项为 {label, text}。", "required": False},
        {"name": "section_top_k", "type": "integer", "description": "每个选项保留多少个相关章节，默认 4。", "required": False},
        {"name": "document_top_k", "type": "integer", "description": "每个选项最多保留多少个候选文档，默认 4。", "required": False},
        {"name": "sentence_top_k", "type": "integer", "description": "每个选项最多保留多少条句子证据，默认 4。", "required": False},
        {"name": "max_length", "type": "integer", "description": "每个选项抽取证据片段的最大长度，默认 220。", "required": False},
        {"name": "use_llm_judge", "type": "bool", "description": "是否在本地排序后再做一次全选项 LLM 裁决；默认 false。", "required": False},
    ]

    def __init__(
        self,
        graph_query_utils,
        document_vector_store,
        document_parser,
        *,
        sentence_vector_store=None,
        section_retriever=None,
        narrative_retriever=None,
        max_workers: int = 4,
    ):
        self.graph_query_utils = graph_query_utils
        self.doc_vs = document_vector_store
        self.sent_vs = sentence_vector_store
        self.document_parser = document_parser
        self.section_retriever = section_retriever
        self.narrative_retriever = narrative_retriever
        self.max_workers = max(1, int(max_workers or 4))
        self.search_related_content_tool = SearchRelatedContentTool(
            document_vector_store=self.doc_vs,
            document_parser=self.document_parser,
            max_workers=self.max_workers,
        )

    @staticmethod
    def _dedup_document_ids(items: Any) -> List[str]:
        return _dedup_strings(items)

    @staticmethod
    def _collect_sentence_rows(raw_docs: List[Any], *, limit: int) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for doc in raw_docs[: max(1, int(limit or 4))]:
            text = str(getattr(doc, "content", "") or "").strip()
            if not text:
                continue
            md = getattr(doc, "metadata", {}) or {}
            rows.append(
                {
                    "text": text,
                    "score": float(md.get("similarity_score", 0.0) or 0.0),
                    "document_id": str(md.get("document_id", "") or "").strip(),
                }
            )
        return rows

    @staticmethod
    def _selection_mode(question_stem: str) -> str:
        lowered = str(question_stem or "").strip().lower()
        negative_cues = (
            "which is not",
            "which was not",
            "which did not",
            "which does not",
            "which cannot",
            "which isn't",
            "which wasn't",
            "except",
            "least likely",
            "least supported",
            "least accurate",
            "least consistent",
            "incorrect",
            "false",
            "not true",
            "not supported",
            "not mentioned",
            "not the reason",
            "doesn't belong",
            "does not belong",
        )
        return "least_supported" if any(cue in lowered for cue in negative_cues) else "most_supported"

    @staticmethod
    def _sorted_option_rows(option_rows: List[Dict[str, Any]], *, selection_mode: str) -> List[Dict[str, Any]]:
        rows = list(option_rows or [])
        if selection_mode == "least_supported":
            rows.sort(
                key=lambda row: (
                    float(row.get("selection_score", 0.0) or 0.0),
                    -float(row.get("support_probability", 0.0) or 0.0),
                    str(row.get("label", "") or ""),
                ),
                reverse=True,
            )
            return rows
        rows.sort(
            key=lambda row: (
                float(row.get("selection_score", 0.0) or 0.0),
                float(row.get("support_probability", 0.0) or 0.0),
                str(row.get("label", "") or ""),
            ),
            reverse=True,
        )
        return rows

    def _judge_best_choice(self, *, question_stem: str, option_rows: List[Dict[str, Any]], selection_mode: str) -> Dict[str, Any]:
        llm = getattr(self.document_parser, "llm", None) if self.document_parser is not None else None
        if llm is None or not option_rows:
            return {}

        choice_blocks: List[str] = []
        for row in option_rows:
            label = str(row.get("label", "") or "").strip().upper()
            option_text = str(row.get("option_text", "") or "").strip()
            evidence = _clip_text(row.get("evidence", "") or "", 420)
            rationale = _clip_text(row.get("rationale", "") or "", 220)
            choice_blocks.append(
                "\n".join(
                    [
                        f"{label}. {option_text}",
                        f"support_score={float(row.get('support_probability', 0.0) or 0.0):.4f}",
                        f"grounding_score={float(row.get('grounding_score', 0.0) or 0.0):.4f}",
                        f"rival_penalty={float(row.get('rival_overlap_penalty', 0.0) or 0.0):.4f}",
                        f"evidence={evidence or '(none)'}",
                        f"rationale={rationale or '(none)'}",
                    ]
                )
            )

        prompt = (
            "You are comparing candidate answers for a multiple-choice QA task.\n"
            + (
                "Choose the option with the weakest support or strongest contradiction in the evidence.\n"
                if selection_mode == "least_supported"
                else "Choose the option with the strongest direct support in the evidence and the fewest unsupported assumptions.\n"
            )
            + "Penalize options whose evidence is only loosely related, or whose evidence also fits rival options better.\n"
            + "Do not reward dramatic implications unless they are explicitly supported.\n\n"
            f"Question:\n{question_stem}\n\n"
            "Candidate options with retrieved evidence:\n"
            + "\n\n".join(choice_blocks)
            + "\n\nReturn strict JSON only:\n"
            "{\n"
            '  "selected_label": "A",\n'
            '  "confidence": 0.0,\n'
            '  "reason": "brief explanation"\n'
            "}\n"
        )
        try:
            responses = llm.run([{"role": "user", "content": prompt}])
            raw = ""
            if isinstance(responses, list) and responses:
                raw = str((responses[-1] or {}).get("content", "") or "").strip()
            parsed = json.loads(correct_json_format(raw))
        except Exception:
            return {}

        selected_label = str(parsed.get("selected_label", "") or "").strip().upper()
        if not selected_label:
            return {}
        return {
            "selected_label": selected_label,
            "confidence": _clamp01(parsed.get("confidence", 0.0)),
            "reason": str(parsed.get("reason", "") or "").strip(),
        }

    def _score_option(self, *, question_stem: str, option: Dict[str, str], section_top_k: int, document_top_k: int, sentence_top_k: int, max_length: int) -> Dict[str, Any]:
        label = str(option.get("label", "") or "").strip().upper()
        option_text = str(option.get("text", "") or "").strip()
        focused_query = (
            f"Question: {question_stem}\n"
            f"Candidate answer {label}: {option_text}\n"
            "Find evidence that directly supports this candidate and helps distinguish it from competing options."
        ).strip()

        section_rows: List[Dict[str, Any]] = []
        narrative_rows: List[Dict[str, Any]] = []
        document_rows: List[Dict[str, Any]] = []
        evidence_rows: List[Tuple[str, str]] = []
        sentence_rows: List[Dict[str, Any]] = []
        retrieval_tasks: Dict[str, Any] = {}
        with ThreadPoolExecutor(max_workers=3) as executor:
            if self.section_retriever is not None:
                retrieval_tasks["section"] = executor.submit(
                    self.section_retriever.search,
                    focused_query,
                    section_top_k=section_top_k,
                    llm_filter_top_k=max(8, section_top_k * 2),
                    llm_filter_threshold=0.3,
                    max_length=max_length,
                )
            if self.narrative_retriever is not None:
                retrieval_tasks["narrative"] = executor.submit(
                    self.narrative_retriever.search,
                    focused_query,
                    storyline_top_k=2,
                    episode_top_k=3,
                    event_top_k=4,
                    document_top_k=document_top_k,
                    llm_filter_threshold=0.3,
                    max_evidence_length=max_length,
                )
            if self.sent_vs is not None:
                retrieval_tasks["sentence"] = executor.submit(
                    self.sent_vs.search,
                    query=focused_query,
                    limit=max(1, int(sentence_top_k or 4)),
                )
            for key, future in retrieval_tasks.items():
                try:
                    result = future.result()
                except Exception:
                    continue
                if key == "section":
                    section_rows = list((result or {}).get("sections") or [])
                    document_rows.extend(list((result or {}).get("documents") or []))
                    evidence_rows.extend(list((result or {}).get("evidence") or []))
                elif key == "narrative":
                    narrative_rows = list((result or {}).get("episodes") or [])
                    document_rows.extend(list((result or {}).get("document_rows") or []))
                    evidence_rows.extend(list((result or {}).get("evidence") or []))
                elif key == "sentence":
                    sentence_rows = self._collect_sentence_rows(result or [], limit=sentence_top_k)

        document_ids: List[str] = []
        for row in document_rows:
            did = str(row.get("document_id", "") or "").strip()
            if did and did not in document_ids:
                document_ids.append(did)
        for row in section_rows:
            for did in _dedup_strings(row.get("source_documents") or []):
                if did not in document_ids:
                    document_ids.append(did)
        document_ids = document_ids[: max(1, int(document_top_k or 4))]

        provisional_section_score = max([float(row.get("score", 0.0) or 0.0) for row in section_rows] or [0.0])
        provisional_narrative_score = max([float(row.get("score", 0.0) or 0.0) for row in narrative_rows] or [0.0])
        provisional_sentence_score = max([float(row.get("score", 0.0) or 0.0) for row in sentence_rows] or [0.0])
        local_evidence_count = len(evidence_rows) + len(sentence_rows)
        need_doc_refinement = (
            bool(document_ids)
            and (
                local_evidence_count < 2
                or max(provisional_section_score, provisional_narrative_score, provisional_sentence_score) < 0.58
            )
        )

        if need_doc_refinement:
            try:
                raw = self.search_related_content_tool.call(
                    json.dumps(
                        {
                            "document_ids": document_ids[:2],
                            "query": focused_query,
                            "max_length": max_length,
                        },
                        ensure_ascii=False,
                    )
                )
                parsed_rows = SectionEvidenceSearch._parse_search_related_content_output(raw)
                if parsed_rows:
                    if evidence_rows:
                        evidence_rows.extend(parsed_rows[: max(0, 2 - len(evidence_rows))])
                    else:
                        evidence_rows = parsed_rows
            except Exception:
                pass

        evidence_parts: List[str] = []
        if section_rows:
            best_section = section_rows[0]
            section_name = str(best_section.get("name", "") or "").strip()
        else:
            section_name = ""
        if evidence_rows:
            evidence_parts.extend([content for _, content in evidence_rows[:2]])
        if sentence_rows:
            evidence_parts.extend([row["text"] for row in sentence_rows[:2]])
        if narrative_rows and not evidence_parts:
            evidence_parts.extend([str(row.get("description", "") or "").strip() for row in narrative_rows[:2] if str(row.get("description", "") or "").strip()])
        evidence_text = "\n".join([part for part in evidence_parts if part]).strip()

        section_score = max([float(row.get("score", 0.0) or 0.0) for row in section_rows] or [0.0])
        narrative_score = max([float(row.get("score", 0.0) or 0.0) for row in narrative_rows] or [0.0])
        sentence_score = max([float(row.get("score", 0.0) or 0.0) for row in sentence_rows] or [0.0])
        document_score = max([float(row.get("score", 0.0) or 0.0) for row in document_rows] or [0.0])

        support_probability = max(section_score, narrative_score, sentence_score, document_score)
        rationale = ""
        if evidence_text and self.document_parser is not None:
            try:
                raw = self.document_parser.score_candidate_relevance(
                    text=evidence_text,
                    goal=(
                        f"Question: {question_stem}\n"
                        f"Candidate answer {label}: {option_text}\n"
                        "Does the evidence directly support this candidate compared with the competing options?"
                    ),
                )
                parsed = json.loads(correct_json_format(raw))
                support_probability = max(
                    support_probability,
                    _clamp01(parsed.get("probability", 0.0)),
                )
                rationale = str(parsed.get("reason", "") or "").strip()
            except Exception:
                pass

        score = (
            0.35 * float(support_probability)
            + 0.25 * float(section_score)
            + 0.20 * float(narrative_score)
            + 0.10 * float(sentence_score)
            + 0.10 * float(document_score)
        )
        option_terms = _tokenize_grounding_text(option_text)
        grounding_score = _token_overlap_ratio(option_terms, evidence_text)
        return {
            "label": label,
            "option_text": option_text,
            "score": float(score),
            "selection_score": float(score),
            "support_probability": float(support_probability),
            "grounding_score": float(grounding_score),
            "section_name": section_name,
            "best_document_id": document_ids[0] if document_ids else "",
            "evidence": evidence_text,
            "rationale": rationale,
            "document_ids": list(document_ids),
        }

    def call(self, params: str, **kwargs) -> str:
        logger.info("🔎 调用 choice_grounded_evidence_search")
        try:
            data = json.loads(params) if isinstance(params, str) else dict(params or {})
        except Exception as e:
            return f"参数解析失败: {e}"

        query = str(data.get("query", "") or "").strip()
        if not query:
            return "query 不能为空。"

        explicit_options = data.get("options")
        if not isinstance(explicit_options, list):
            explicit_options = None
        question_stem, options = _parse_mcq_query(query, explicit_options=explicit_options)
        if not options:
            return "未能从 query 中解析出选项；请提供带 Choices/Options 的完整多选题，或显式传入 options。"

        section_top_k = max(1, int(data.get("section_top_k", 4) or 4))
        document_top_k = max(1, int(data.get("document_top_k", 4) or 4))
        sentence_top_k = max(1, int(data.get("sentence_top_k", 4) or 4))
        max_length = max(80, int(data.get("max_length", 220) or 220))
        use_llm_judge = _to_bool(data.get("use_llm_judge"), False)
        selection_mode = self._selection_mode(question_stem)

        option_rows: List[Dict[str, Any]] = []
        worker_count = max(1, min(self.max_workers, len(options)))
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [
                executor.submit(
                    self._score_option,
                    question_stem=question_stem,
                    option=option,
                    section_top_k=section_top_k,
                    document_top_k=document_top_k,
                    sentence_top_k=sentence_top_k,
                    max_length=max_length,
                )
                for option in options
            ]
            for future in as_completed(futures):
                option_rows.append(future.result())

        for row in option_rows:
            evidence_text = str(row.get("evidence", "") or "").strip()
            rival_overlap_penalty = 0.0
            for rival in option_rows:
                if rival is row:
                    continue
                rival_terms = _tokenize_grounding_text(rival.get("option_text", "") or "")
                rival_overlap_penalty = max(rival_overlap_penalty, _token_overlap_ratio(rival_terms, evidence_text))
            row["rival_overlap_penalty"] = float(rival_overlap_penalty)
            row["discriminative_score"] = max(0.0, float(row.get("grounding_score", 0.0) or 0.0) - 0.65 * rival_overlap_penalty)
            row["score"] = float(row.get("score", 0.0) or 0.0) + (
                0.12 * float(row.get("grounding_score", 0.0) or 0.0)
                + 0.08 * float(row.get("discriminative_score", 0.0) or 0.0)
                - 0.14 * rival_overlap_penalty
            )
            if selection_mode == "least_supported":
                row["selection_score"] = (
                    0.62 * (1.0 - float(row.get("support_probability", 0.0) or 0.0))
                    + 0.20 * float(rival_overlap_penalty)
                    + 0.18 * (1.0 - float(row.get("grounding_score", 0.0) or 0.0))
                )
            else:
                row["selection_score"] = float(row.get("score", 0.0) or 0.0)

        option_rows = self._sorted_option_rows(option_rows, selection_mode=selection_mode)

        selected_label = ""
        selected_reason = ""
        selected_confidence = 0.0
        if use_llm_judge:
            judge_result = self._judge_best_choice(
                question_stem=question_stem,
                option_rows=option_rows,
                selection_mode=selection_mode,
            )
            selected_label = str(judge_result.get("selected_label", "") or "").strip().upper()
            selected_reason = str(judge_result.get("reason", "") or "").strip()
            selected_confidence = float(judge_result.get("confidence", 0.0) or 0.0)
            top_before_judge = option_rows[0] if option_rows else {}
            selected_row = None
            for row in option_rows:
                if str(row.get("label", "") or "").strip().upper() == selected_label:
                    selected_row = row
                    break
            use_judge_override = False
            if selected_label and selected_row is not None:
                top_label = str(top_before_judge.get("label", "") or "").strip().upper()
                top_support = float(top_before_judge.get("support_probability", 0.0) or 0.0)
                top_grounding = float(top_before_judge.get("grounding_score", 0.0) or 0.0)
                top_rival_penalty = float(top_before_judge.get("rival_overlap_penalty", 0.0) or 0.0)
                selected_support = float(selected_row.get("support_probability", 0.0) or 0.0)
                selected_grounding = float(selected_row.get("grounding_score", 0.0) or 0.0)
                selected_rival_penalty = float(selected_row.get("rival_overlap_penalty", 0.0) or 0.0)
                use_judge_override = (
                    selected_label == top_label
                    or selected_support >= top_support + 0.04
                    or (
                        selected_support >= top_support - 0.02
                        and (
                            selected_grounding >= top_grounding + 0.08
                            or top_rival_penalty >= selected_rival_penalty + 0.12
                            or top_support < 0.82
                        )
                    )
                )
            if selected_label:
                for row in option_rows:
                    if str(row.get("label", "") or "").strip().upper() == selected_label:
                        row["judge_selected"] = True
                        row["judge_reason"] = selected_reason
                        row["judge_confidence"] = selected_confidence
                        if use_judge_override:
                            row["selection_score"] = float(row.get("selection_score", 0.0) or 0.0) + 0.22 + 0.12 * selected_confidence
                    else:
                        row["judge_selected"] = False
                option_rows = self._sorted_option_rows(option_rows, selection_mode=selection_mode)

        lines: List[str] = ["[Choice Grounded Evidence Search]"]
        if question_stem:
            lines.append(f"question: {question_stem}")
        lines.append(f"selection_mode: {selection_mode}")
        if selected_label:
            lines.append("[Judge]")
            lines.append(
                f"selected={selected_label} confidence={selected_confidence:.4f}"
                + (f" reason={selected_reason}" if selected_reason else "")
            )
        lines.append("[Ranked Choices]")
        lines.extend(_format_option_rows(option_rows))
        if option_rows:
            best = option_rows[0]
            lines.append("[Recommended Choice]")
            lines.append(
                f"{best['label']}. {best['option_text']} "
                f"selection_score={float(best.get('selection_score', 0.0) or 0.0):.4f}"
            )
        return "\n".join(lines)


@register_tool("entity_event_trace_search")
class EntityEventTraceSearch(BaseTool):
    name = "entity_event_trace_search"
    description = (
        "沿着 Storyline / Episode / Event / Section / Interaction 串联问题相关的证据轨迹，"
        "特别适合回答叙事作用、隐含因果、真实关系、身份反转、以及“前面像A但后面揭示其实是B”这类问题。"
        "如果输入是多选题，它会基于整条证据轨迹给出一个建议选项。"
    )
    parameters = [
        {"name": "query", "type": "string", "description": "问题文本；可以直接带 Choices/Options。", "required": True},
        {"name": "focus_entities", "type": "array", "description": "可选；显式指定要追踪的实体名列表。", "required": False},
        {"name": "storyline_top_k", "type": "integer", "description": "最多保留多少条 Storyline，默认 3。", "required": False},
        {"name": "episode_top_k", "type": "integer", "description": "最多保留多少个 Episode，默认 4。", "required": False},
        {"name": "event_top_k", "type": "integer", "description": "最多保留多少个 Event，默认 6。", "required": False},
        {"name": "section_top_k", "type": "integer", "description": "最多保留多少个 Section，默认 4。", "required": False},
        {"name": "document_top_k", "type": "integer", "description": "最多追踪多少个 document_id，默认 6。", "required": False},
        {"name": "related_entity_limit", "type": "integer", "description": "最多展示多少个关键相关实体，默认 6。", "required": False},
        {"name": "interaction_limit", "type": "integer", "description": "最多展示多少条 interactions/dialogues，默认 8。", "required": False},
        {"name": "max_length", "type": "integer", "description": "每条证据片段的最大长度，默认 220。", "required": False},
        {"name": "use_option_probes", "type": "bool", "description": "多选题时是否额外对每个选项做 section probe；默认 false。", "required": False},
        {"name": "use_llm_choice_judge", "type": "bool", "description": "多选题时，是否基于整条证据轨迹做一次最终选项判断；默认 true。", "required": False},
    ]

    def __init__(
        self,
        graph_query_utils,
        document_vector_store,
        document_parser,
        *,
        sentence_vector_store=None,
        section_retriever=None,
        narrative_retriever=None,
        interaction_db_path: str = "",
        doc_type: str = "general",
        max_workers: int = 4,
    ):
        self.graph_query_utils = graph_query_utils
        self.doc_vs = document_vector_store
        self.sent_vs = sentence_vector_store
        self.document_parser = document_parser
        self.section_retriever = section_retriever
        self.narrative_retriever = narrative_retriever
        self.doc_type = str(doc_type or "general").strip().lower() or "general"
        self.max_workers = max(1, int(max_workers or 4))
        self.search_related_content_tool = SearchRelatedContentTool(
            document_vector_store=self.doc_vs,
            document_parser=self.document_parser,
            max_workers=self.max_workers,
        )
        self.interaction_tool = None
        if str(interaction_db_path or "").strip():
            try:
                self.interaction_tool = SQLGetInteractionsByDocumentIDs(
                    str(interaction_db_path).strip(),
                    doc_type=self.doc_type,
                )
            except Exception:
                self.interaction_tool = None

    @staticmethod
    def _collect_document_ids(*sources: Any, limit: int) -> List[str]:
        out: List[str] = []
        seen = set()
        for source in sources:
            if isinstance(source, list):
                for item in source:
                    if isinstance(item, dict):
                        did = str(item.get("document_id", "") or "").strip()
                        if did and did not in seen:
                            seen.add(did)
                            out.append(did)
                        for sub_did in _dedup_strings(item.get("source_documents") or []):
                            if sub_did not in seen:
                                seen.add(sub_did)
                                out.append(sub_did)
                    elif isinstance(item, tuple) and len(item) >= 1:
                        did = str(item[0] or "").strip()
                        if did and did not in seen:
                            seen.add(did)
                            out.append(did)
                    else:
                        did = str(item or "").strip()
                        if did and did not in seen:
                            seen.add(did)
                            out.append(did)
        return out[: max(1, int(limit or 6))]

    def _resolve_focus_entities(self, *, query: str, explicit_names: List[str], limit: int) -> List[Dict[str, Any]]:
        merged: Dict[str, Dict[str, Any]] = {}
        candidate_names = [name for name in explicit_names if str(name or "").strip()]
        if not candidate_names:
            ranked_rows = self.graph_query_utils.search_entities_by_type_ranked("Entity", keyword=query, limit=max(8, limit * 3)) or []
            for row in ranked_rows:
                ent = row.get("entity")
                if ent is None:
                    continue
                labels = getattr(ent, "type", [])
                if _is_structural_label(labels):
                    continue
                eid = str(getattr(ent, "id", "") or "").strip()
                if not eid:
                    continue
                merged[eid] = {
                    "id": eid,
                    "name": str(getattr(ent, "name", "") or "").strip(),
                    "labels": labels if isinstance(labels, list) else [labels],
                    "description": str(getattr(ent, "description", "") or "").strip(),
                    "score": float(row.get("keyword_score_raw", 0.0) or 0.0),
                    "source_documents": list(getattr(ent, "source_documents", []) or []),
                }
        else:
            for name in candidate_names:
                ranked_rows = self.graph_query_utils.search_entities_by_type_ranked("Entity", keyword=name, limit=4) or []
                for row in ranked_rows:
                    ent = row.get("entity")
                    if ent is None:
                        continue
                    labels = getattr(ent, "type", [])
                    if _is_structural_label(labels):
                        continue
                    eid = str(getattr(ent, "id", "") or "").strip()
                    if not eid:
                        continue
                    prev = merged.get(eid)
                    score = float(row.get("keyword_score_raw", 0.0) or 0.0)
                    if prev is None or score > float(prev.get("score", 0.0) or 0.0):
                        merged[eid] = {
                            "id": eid,
                            "name": str(getattr(ent, "name", "") or "").strip(),
                            "labels": labels if isinstance(labels, list) else [labels],
                            "description": str(getattr(ent, "description", "") or "").strip(),
                            "score": score,
                            "source_documents": list(getattr(ent, "source_documents", []) or []),
                        }
        rows = list(merged.values())
        rows.sort(key=lambda row: (float(row.get("score", 0.0) or 0.0), str(row.get("name", "") or "")), reverse=True)
        return rows[: max(1, int(limit or 6))]

    def _collect_related_entity_rows(self, *, source_rows: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
        merged: Dict[str, Dict[str, Any]] = {}
        for row in source_rows[: max(1, min(6, len(source_rows)))]:
            source_id = str(row.get("id", "") or "").strip()
            if not source_id:
                continue
            related = self.graph_query_utils.search_related_entities(
                source_id=source_id,
                limit=max(6, limit * 2),
                return_relations=True,
            ) or []
            seed_score = float(row.get("score", 0.0) or 0.0)
            for ent, rel in related:
                labels = getattr(ent, "type", [])
                if _is_structural_label(labels):
                    continue
                eid = str(getattr(ent, "id", "") or "").strip()
                if not eid:
                    continue
                relation_bits = []
                predicate = str(getattr(rel, "predicate", "") or "").strip()
                desc = str(getattr(rel, "description", "") or "").strip()
                if predicate:
                    relation_bits.append(predicate)
                if desc:
                    relation_bits.append(desc)
                relation_text = " | ".join(relation_bits)
                current = merged.get(eid)
                if current is None or seed_score > float(current.get("score", 0.0) or 0.0):
                    merged[eid] = {
                        "id": eid,
                        "name": str(getattr(ent, "name", "") or "").strip(),
                        "labels": labels if isinstance(labels, list) else [labels],
                        "description": str(getattr(ent, "description", "") or "").strip(),
                        "score": seed_score,
                        "relation_text": relation_text,
                        "anchor_name": str(row.get("name", "") or "").strip(),
                        "source_documents": list(getattr(ent, "source_documents", []) or []),
                    }
        rows = list(merged.values())
        rows.sort(key=lambda row: (float(row.get("score", 0.0) or 0.0), str(row.get("name", "") or "")), reverse=True)
        return rows[: max(1, int(limit or 6))]

    def _collect_interaction_lines(self, *, document_ids: List[str], focus_entity_names: List[str], limit: int) -> List[str]:
        if self.interaction_tool is None or not document_ids:
            return []
        try:
            raw = self.interaction_tool.call(
                json.dumps(
                    {
                        "document_ids": document_ids,
                        "limit": max(1, int(limit or 8)),
                    },
                    ensure_ascii=False,
                )
            )
        except Exception:
            return []
        lines: List[str] = []
        for line in str(raw or "").splitlines():
            line = line.strip()
            if not line or line.startswith("按 document_id 查询结果"):
                continue
            if focus_entity_names:
                lowered = line.lower()
                if not any(name.lower() in lowered for name in focus_entity_names if name):
                    continue
            lines.append(line)
        return lines[: max(1, int(limit or 8))]

    def _judge_choice_from_trace(self, *, question_stem: str, options: List[Dict[str, str]], trace_text: str) -> Dict[str, Any]:
        llm = getattr(self.document_parser, "llm", None) if self.document_parser is not None else None
        if llm is None or not options:
            return {}
        option_text = "\n".join([f"{opt['label']}. {opt['text']}" for opt in options])
        prompt = (
            "You are answering a multiple-choice narrative reasoning question from a traced evidence chain.\n"
            "Use the whole trace, not just the most local sentence.\n"
            "Prefer later reveals, real causal consequences, true relationships, and explicit resolutions over misleading early appearances.\n"
            "For 'why important / significance / what does this reveal' questions, choose the option best supported by the downstream narrative consequence.\n"
            "For identity or role questions, prefer the final story-level reality over early surface impressions.\n\n"
            f"Question:\n{question_stem}\n\n"
            f"Choices:\n{option_text}\n\n"
            f"Trace:\n{_clip_text(trace_text, 5000)}\n\n"
            "Return strict JSON only:\n"
            "{\n"
            '  "selected_label": "A",\n'
            '  "confidence": 0.0,\n'
            '  "reason": "brief explanation"\n'
            "}\n"
        )
        try:
            responses = llm.run([{"role": "user", "content": prompt}])
            raw = ""
            if isinstance(responses, list) and responses:
                raw = str((responses[-1] or {}).get("content", "") or "").strip()
            parsed = json.loads(correct_json_format(raw))
        except Exception:
            return {}
        label = str(parsed.get("selected_label", "") or "").strip().upper()
        if not label:
            return {}
        return {
            "selected_label": label,
            "confidence": _clamp01(parsed.get("confidence", 0.0)),
            "reason": str(parsed.get("reason", "") or "").strip(),
        }

    def _probe_option_sections(
        self,
        *,
        question_stem: str,
        options: List[Dict[str, str]],
        max_length: int,
    ) -> List[Dict[str, Any]]:
        if self.section_retriever is None or not options:
            return []
        rows: List[Dict[str, Any]] = []
        for option in options[:4]:
            label = str(option.get("label", "") or "").strip().upper()
            option_text = str(option.get("text", "") or "").strip()
            focused_query = (
                f"Question: {question_stem}\n"
                f"Candidate answer {label}: {option_text}\n"
                "Find the section most relevant to deciding whether this candidate is true, especially any later reveal or decisive consequence."
            ).strip()
            try:
                result = self.section_retriever.search(
                    focused_query,
                    section_top_k=1,
                    llm_filter_top_k=4,
                    llm_filter_threshold=0.3,
                    max_length=max_length,
                ) or {}
            except Exception:
                continue
            sections = list(result.get("sections") or [])
            evidences = list(result.get("evidence") or [])
            documents = list(result.get("documents") or [])
            best_section = sections[0] if sections else {}
            rows.append(
                {
                    "label": label,
                    "option_text": option_text,
                    "section_name": str(best_section.get("name", "") or "").strip(),
                    "score": float(best_section.get("score", 0.0) or 0.0),
                    "document_ids": self._collect_document_ids(documents, evidences, sections, limit=2),
                    "evidence": _clip_text((evidences[0][1] if evidences and isinstance(evidences[0], tuple) and len(evidences[0]) >= 2 else ""), 260),
                }
            )
        rows.sort(key=lambda row: (float(row.get("score", 0.0) or 0.0), str(row.get("label", "") or "")), reverse=True)
        return rows

    def call(self, params: str, **kwargs) -> str:
        logger.info("🔎 调用 entity_event_trace_search")
        try:
            data = json.loads(params) if isinstance(params, str) else dict(params or {})
        except Exception as e:
            return f"参数解析失败: {e}"

        query = str(data.get("query", "") or "").strip()
        if not query:
            return "query 不能为空。"

        explicit_focus_entities = [str(x or "").strip() for x in (data.get("focus_entities") or []) if str(x or "").strip()]
        storyline_top_k = max(1, int(data.get("storyline_top_k", 3) or 3))
        episode_top_k = max(1, int(data.get("episode_top_k", 4) or 4))
        event_top_k = max(1, int(data.get("event_top_k", 6) or 6))
        section_top_k = max(1, int(data.get("section_top_k", 4) or 4))
        document_top_k = max(1, int(data.get("document_top_k", 6) or 6))
        related_entity_limit = max(1, int(data.get("related_entity_limit", 6) or 6))
        interaction_limit = max(1, int(data.get("interaction_limit", 8) or 8))
        max_length = max(80, int(data.get("max_length", 220) or 220))
        use_option_probes = _to_bool(data.get("use_option_probes"), False)
        use_llm_choice_judge = _to_bool(data.get("use_llm_choice_judge"), True)

        question_stem, options = _parse_mcq_query(query)
        base_query = question_stem or query
        retrieval_query = query if options else base_query

        narrative_rows: Dict[str, Any] = {}
        if self.narrative_retriever is not None:
            try:
                narrative_rows = self.narrative_retriever.search(
                    retrieval_query,
                    storyline_top_k=storyline_top_k,
                    episode_top_k=episode_top_k,
                    event_top_k=event_top_k,
                    document_top_k=document_top_k,
                    llm_filter_threshold=0.35,
                    max_evidence_length=max_length,
                ) or {}
            except Exception:
                narrative_rows = {}

        section_rows: Dict[str, Any] = {}
        if self.section_retriever is not None:
            try:
                section_rows = self.section_retriever.search(
                    retrieval_query,
                    section_top_k=section_top_k,
                    llm_filter_top_k=max(8, section_top_k * 2),
                    llm_filter_threshold=0.35,
                    max_length=max_length,
                ) or {}
            except Exception:
                section_rows = {}

        storylines = list(narrative_rows.get("storylines") or [])
        episodes = list(narrative_rows.get("episodes") or [])
        events = list(narrative_rows.get("events") or [])
        narrative_documents = list(narrative_rows.get("document_rows") or [])
        narrative_evidence = list(narrative_rows.get("evidence") or [])
        sections = list(section_rows.get("sections") or [])
        section_documents = list(section_rows.get("documents") or [])
        section_evidence = list(section_rows.get("evidence") or [])

        document_ids = self._collect_document_ids(
            narrative_documents,
            section_documents,
            narrative_evidence,
            section_evidence,
            sections,
            events,
            limit=document_top_k,
        )
        if document_ids:
            try:
                raw = self.search_related_content_tool.call(
                    json.dumps(
                        {
                            "document_ids": document_ids,
                            "query": retrieval_query,
                            "max_length": max_length,
                        },
                        ensure_ascii=False,
                    )
                )
                parsed_related = SectionEvidenceSearch._parse_search_related_content_output(raw)
            except Exception:
                parsed_related = []
        else:
            parsed_related = []

        resolved_focus_entities = self._resolve_focus_entities(
            query=retrieval_query,
            explicit_names=explicit_focus_entities,
            limit=related_entity_limit,
        )
        option_probe_rows = (
            self._probe_option_sections(
                question_stem=base_query,
                options=options,
                max_length=max_length,
            )
            if use_option_probes
            else []
        )
        structural_related_entities = self._collect_related_entity_rows(
            source_rows=events or episodes or sections,
            limit=related_entity_limit,
        )
        merged_focus: Dict[str, Dict[str, Any]] = {}
        for row in resolved_focus_entities + structural_related_entities:
            eid = str(row.get("id", "") or "").strip()
            if not eid:
                continue
            prev = merged_focus.get(eid)
            if prev is None or float(row.get("score", 0.0) or 0.0) > float(prev.get("score", 0.0) or 0.0):
                merged_focus[eid] = dict(row)
        focus_rows = list(merged_focus.values())
        focus_rows.sort(key=lambda row: (float(row.get("score", 0.0) or 0.0), str(row.get("name", "") or "")), reverse=True)
        focus_rows = focus_rows[: max(1, related_entity_limit)]

        interaction_lines = self._collect_interaction_lines(
            document_ids=document_ids,
            focus_entity_names=[str(row.get("name", "") or "").strip() for row in focus_rows],
            limit=interaction_limit,
        )

        lines: List[str] = ["[Entity Event Trace Search]"]
        lines.append(f"question: {base_query}")
        if storylines:
            lines.extend(_format_ranked_rows("Storylines", storylines[:storyline_top_k], include_docs=False))
        if episodes:
            lines.extend(_format_ranked_rows("Episodes", episodes[:episode_top_k]))
        if events:
            lines.extend(_format_ranked_rows("Events", events[:event_top_k], include_docs=False))
        if sections:
            lines.extend(_format_ranked_rows("Sections", sections[:section_top_k]))

        lines.append("[Focus Entities]")
        if focus_rows:
            for idx, row in enumerate(focus_rows, 1):
                labels = ", ".join([str(x or "").strip() for x in (row.get("labels") or []) if str(x or "").strip()])
                header = f"{idx}. {str(row.get('name', '') or '').strip() or '(unnamed)'}"
                if labels:
                    header += f" [{labels}]"
                header += f" score={float(row.get('score', 0.0) or 0.0):.4f}"
                lines.append(header)
                relation_text = str(row.get("relation_text", "") or "").strip()
                if relation_text:
                    anchor_name = str(row.get("anchor_name", "") or "").strip()
                    if anchor_name:
                        lines.append(f"   linked_from: {anchor_name}")
                    lines.append(f"   relation: {_clip_text(relation_text, 220)}")
                desc = str(row.get("description", "") or "").strip()
                if desc:
                    lines.append(f"   description: {_clip_text(desc, 220)}")
        else:
            lines.append("None")

        lines.append("[Tracked Documents]")
        if document_ids:
            for idx, document_id in enumerate(document_ids, 1):
                lines.append(f"{idx}. {document_id}")
        else:
            lines.append("None")

        if option_probe_rows:
            lines.append("[Option Probes]")
            for row in option_probe_rows:
                line = (
                    f"{row['label']}. {row['option_text']} score={float(row.get('score', 0.0) or 0.0):.4f}"
                )
                if row.get("section_name"):
                    line += f" section={row['section_name']}"
                lines.append(line)
                if row.get("document_ids"):
                    lines.append(f"   docs: {', '.join(row['document_ids'])}")
                if row.get("evidence"):
                    lines.append(f"   evidence: {row['evidence']}")

        lines.append("[Interactions]")
        if interaction_lines:
            lines.extend(interaction_lines)
        else:
            lines.append("None")

        lines.append("[Evidence]")
        evidence_rows = parsed_related or narrative_evidence or section_evidence
        if evidence_rows:
            for idx, item in enumerate(evidence_rows[: max(3, document_top_k)], 1):
                if isinstance(item, tuple):
                    document_id, content = item
                elif isinstance(item, dict):
                    document_id = str(item.get("document_id", "") or "").strip()
                    content = str(item.get("content", "") or item.get("text", "") or "").strip()
                else:
                    document_id = ""
                    content = str(item or "").strip()
                if content:
                    prefix = f"{document_id}: " if document_id else ""
                    lines.append(f"{idx}. {prefix}{_clip_text(content, 320)}")
        else:
            lines.append("None")

        trace_text = "\n".join(lines)
        if options and use_llm_choice_judge:
            judge_result = self._judge_choice_from_trace(
                question_stem=base_query,
                options=options,
                trace_text=trace_text,
            )
            selected_label = str(judge_result.get("selected_label", "") or "").strip().upper()
            if selected_label:
                selected_reason = str(judge_result.get("reason", "") or "").strip()
                selected_confidence = float(judge_result.get("confidence", 0.0) or 0.0)
                lines.append("[Suggested Choice]")
                lines.append(f"{selected_label} confidence={selected_confidence:.4f}")
                if selected_reason:
                    lines.append(f"reason: {selected_reason}")

        return "\n".join(lines)


@register_tool("implication_constrained_inference_search")
class ImplicationConstrainedInferenceSearch(BaseTool):
    name = "implication_constrained_inference_search"
    description = (
        "面向隐含结论、未明说警告、可能为真、梦境意义等问题的约束推断工具。"
        "它会先汇总局部章节、句子和事件证据，再抽取被文本强约束的结论，"
        "避免把表层字面信息或过度脑补当成答案。"
    )
    parameters = [
        {"name": "query", "type": "string", "description": "问题文本；可以直接带 Choices/Options。", "required": True},
        {"name": "section_top_k", "type": "integer", "description": "最多保留多少个章节候选，默认 4。", "required": False},
        {"name": "sentence_top_k", "type": "integer", "description": "最多保留多少条句子证据，默认 6。", "required": False},
        {"name": "event_top_k", "type": "integer", "description": "最多保留多少个事件候选，默认 4。", "required": False},
        {"name": "document_top_k", "type": "integer", "description": "最多保留多少个 document_id，默认 5。", "required": False},
        {"name": "max_length", "type": "integer", "description": "每条证据片段最大长度，默认 220。", "required": False},
        {"name": "use_llm_choice_judge", "type": "bool", "description": "多选题时是否做最终选项判断；默认 true。", "required": False},
    ]

    def __init__(
        self,
        graph_query_utils,
        document_vector_store,
        document_parser,
        *,
        sentence_vector_store=None,
        section_retriever=None,
        narrative_retriever=None,
        max_workers: int = 4,
    ):
        self.graph_query_utils = graph_query_utils
        self.doc_vs = document_vector_store
        self.sent_vs = sentence_vector_store
        self.document_parser = document_parser
        self.section_retriever = section_retriever
        self.narrative_retriever = narrative_retriever
        self.max_workers = max(1, int(max_workers or 4))
        self.search_related_content_tool = SearchRelatedContentTool(
            document_vector_store=self.doc_vs,
            document_parser=self.document_parser,
            max_workers=self.max_workers,
        )

    @staticmethod
    def _collect_sentence_rows(raw_docs: List[Any], *, limit: int) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for doc in raw_docs[: max(1, int(limit or 6))]:
            text = str(getattr(doc, "content", "") or "").strip()
            if not text:
                continue
            md = getattr(doc, "metadata", {}) or {}
            rows.append(
                {
                    "text": text,
                    "score": float(md.get("similarity_score", 0.0) or 0.0),
                    "document_id": str(md.get("document_id", "") or "").strip(),
                }
            )
        return rows

    def _judge_implication(
        self,
        *,
        question_stem: str,
        options: List[Dict[str, str]],
        evidence_text: str,
    ) -> Dict[str, Any]:
        llm = getattr(self.document_parser, "llm", None) if self.document_parser is not None else None
        if llm is None:
            return {}
        option_block = "\n".join([f"{opt['label']}. {opt['text']}" for opt in options]) if options else ""
        prompt = (
            "You are answering a constrained inference question from local narrative evidence.\n"
            "Your task is not to summarize the whole story. Your task is to infer only what is tightly forced by the evidence.\n"
            "Separate these carefully:\n"
            "1. explicit observations\n"
            "2. character beliefs or excuses\n"
            "3. conclusions strongly implied by the observations\n"
            "4. conclusions that are merely plausible but not forced\n"
            "For unspoken warning questions, identify the deeper risk or adjustment being hinted at, not just the surface symptom.\n"
            "For likely true questions, choose the option best constrained by the evidence and reject personality speculation without support.\n"
            "For dream significance questions, identify what later action the dream specifically motivates.\n"
            "Reject answers that are too literal, too narrow, or add unsupported motives.\n\n"
            f"Question:\n{question_stem}\n\n"
        )
        if option_block:
            prompt += f"Choices:\n{option_block}\n\n"
        prompt += (
            f"Evidence:\n{_clip_text(evidence_text, 5200)}\n\n"
            "Return strict JSON only:\n"
            "{\n"
            '  "explicit_observations": ["fact 1", "fact 2"],\n'
            '  "forced_implications": ["implication 1"],\n'
            '  "rejected_plausible_but_unsupported": ["rejected guess"],\n'
        )
        if option_block:
            prompt += '  "selected_label": "A",\n'
        prompt += (
            '  "confidence": 0.0,\n'
            '  "reason": "brief explanation"\n'
            "}\n"
        )
        try:
            responses = llm.run([{"role": "user", "content": prompt}])
            raw = ""
            if isinstance(responses, list) and responses:
                raw = str((responses[-1] or {}).get("content", "") or "").strip()
            parsed = json.loads(correct_json_format(raw))
        except Exception:
            return {}
        return {
            "explicit_observations": [str(x or "").strip() for x in (parsed.get("explicit_observations") if isinstance(parsed.get("explicit_observations"), list) else []) if str(x or "").strip()],
            "forced_implications": [str(x or "").strip() for x in (parsed.get("forced_implications") if isinstance(parsed.get("forced_implications"), list) else []) if str(x or "").strip()],
            "rejected_plausible_but_unsupported": [str(x or "").strip() for x in (parsed.get("rejected_plausible_but_unsupported") if isinstance(parsed.get("rejected_plausible_but_unsupported"), list) else []) if str(x or "").strip()],
            "selected_label": str(parsed.get("selected_label", "") or "").strip().upper(),
            "confidence": _clamp01(parsed.get("confidence", 0.0)),
            "reason": str(parsed.get("reason", "") or "").strip(),
        }

    def call(self, params: str, **kwargs) -> str:
        logger.info("🔎 调用 implication_constrained_inference_search")
        try:
            data = json.loads(params) if isinstance(params, str) else dict(params or {})
        except Exception as e:
            return f"参数解析失败: {e}"

        query = str(data.get("query", "") or "").strip()
        if not query:
            return "query 不能为空。"

        section_top_k = max(1, int(data.get("section_top_k", 4) or 4))
        sentence_top_k = max(1, int(data.get("sentence_top_k", 6) or 6))
        event_top_k = max(1, int(data.get("event_top_k", 4) or 4))
        document_top_k = max(1, int(data.get("document_top_k", 5) or 5))
        max_length = max(80, int(data.get("max_length", 220) or 220))
        use_llm_choice_judge = _to_bool(data.get("use_llm_choice_judge"), True)

        question_stem, options = _parse_mcq_query(query)
        base_query = question_stem or query
        retrieval_query = query if options else base_query

        section_rows: Dict[str, Any] = {}
        if self.section_retriever is not None:
            try:
                section_rows = self.section_retriever.search(
                    retrieval_query,
                    section_top_k=section_top_k,
                    llm_filter_top_k=max(8, section_top_k * 2),
                    llm_filter_threshold=0.28,
                    max_length=max_length,
                ) or {}
            except Exception:
                section_rows = {}

        narrative_rows: Dict[str, Any] = {}
        if self.narrative_retriever is not None:
            try:
                narrative_rows = self.narrative_retriever.search(
                    retrieval_query,
                    storyline_top_k=1,
                    episode_top_k=2,
                    event_top_k=event_top_k,
                    document_top_k=document_top_k,
                    llm_filter_threshold=0.28,
                    max_evidence_length=max_length,
                ) or {}
            except Exception:
                narrative_rows = {}

        sentence_rows: List[Dict[str, Any]] = []
        if self.sent_vs is not None:
            try:
                sentence_hits = self.sent_vs.search(query=retrieval_query, limit=max(1, int(sentence_top_k or 6)))
                sentence_rows = self._collect_sentence_rows(sentence_hits, limit=sentence_top_k)
            except Exception:
                sentence_rows = []

        sections = list(section_rows.get("sections") or [])
        section_documents = list(section_rows.get("documents") or [])
        section_evidence = list(section_rows.get("evidence") or [])
        episodes = list(narrative_rows.get("episodes") or [])
        events = list(narrative_rows.get("events") or [])
        narrative_documents = list(narrative_rows.get("document_rows") or [])
        narrative_evidence = list(narrative_rows.get("evidence") or [])

        evidence_blocks: List[str] = []
        for idx, row in enumerate(sections[:section_top_k], 1):
            name = str(row.get("name", "") or "").strip()
            desc = str(row.get("description", "") or "").strip()
            if name or desc:
                evidence_blocks.append(f"[Section {idx}] {name} score={float(row.get('score', 0.0) or 0.0):.4f}\n{_clip_text(desc, 260)}")
        for idx, row in enumerate(events[:event_top_k], 1):
            name = str(row.get("name", "") or "").strip()
            desc = str(row.get("description", "") or "").strip()
            if name or desc:
                evidence_blocks.append(f"[Event {idx}] {name} score={float(row.get('score', 0.0) or 0.0):.4f}\n{_clip_text(desc, 260)}")
        for idx, row in enumerate(sentence_rows[:sentence_top_k], 1):
            text = str(row.get("text", "") or "").strip()
            if text:
                evidence_blocks.append(f"[Sentence {idx}] score={float(row.get('score', 0.0) or 0.0):.4f}\n{_clip_text(text, 260)}")
        lines: List[str] = ["[Implication Constrained Inference Search]"]
        lines.append(f"question: {base_query}")
        if sections:
            lines.extend(_format_ranked_rows("Sections", sections[:section_top_k]))
        if episodes:
            lines.extend(_format_ranked_rows("Episodes", episodes[:2]))
        if events:
            lines.extend(_format_ranked_rows("Events", events[:event_top_k], include_docs=False))
        lines.append("[Evidence]")
        if evidence_blocks:
            lines.extend(evidence_blocks)
        else:
            lines.append("None")

        if evidence_blocks:
            judged = self._judge_implication(
                question_stem=base_query,
                options=options if use_llm_choice_judge else [],
                evidence_text="\n\n".join(evidence_blocks),
            )
            lines.append("[Explicit Observations]")
            observations = judged.get("explicit_observations") or []
            if observations:
                for idx, item in enumerate(observations[:6], 1):
                    lines.append(f"{idx}. {item}")
            else:
                lines.append("None")
            lines.append("[Forced Implications]")
            implications = judged.get("forced_implications") or []
            if implications:
                for idx, item in enumerate(implications[:4], 1):
                    lines.append(f"{idx}. {item}")
            else:
                lines.append("None")
            lines.append("[Rejected Unsupported Guesses]")
            rejected = judged.get("rejected_plausible_but_unsupported") or []
            if rejected:
                for idx, item in enumerate(rejected[:4], 1):
                    lines.append(f"{idx}. {item}")
            else:
                lines.append("None")

            selected_label = str(judged.get("selected_label", "") or "").strip().upper()
            if options and selected_label:
                lines.append("[Suggested Choice]")
                lines.append(f"{selected_label} confidence={float(judged.get('confidence', 0.0) or 0.0):.4f}")
                reason = str(judged.get("reason", "") or "").strip()
                if reason:
                    lines.append(f"reason: {reason}")
        return "\n".join(lines)

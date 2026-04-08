from typing import Dict, Any, List, Optional, Tuple
import json
import difflib
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from qwen_agent.tools.base import BaseTool, register_tool

from qwen_agent.utils.utils import logger
from core.utils.format import DOC_TYPE_META
from core.utils.format import correct_json_format
from core.utils.general_utils import compress_query_for_vector_search, safe_list, safe_str

# =========================
# 公共格式化/工具函数
# =========================

def _as_props_dict(v: Any) -> Dict[str, Any]:
    if isinstance(v, dict):
        return v
    if isinstance(v, str):
        s = v.strip()
        if s.startswith("{") and s.endswith("}"):
            try:
                obj = json.loads(s)
                return obj if isinstance(obj, dict) else {}
            except Exception:
                return {}
    return {}

def format_entity_results(
    results,
    *,
    graph_query_utils=None,
    resolve_source_documents: bool = False,
    include_document_ids_in_source_sections: bool = True,
):
    lines: List[str] = []
    for entity in results:
        eid = getattr(entity, "id", None)
        name = getattr(entity, "name", "") or "(unnamed)"

        etype = getattr(entity, "type", None)
        if isinstance(etype, (list, tuple, set)):
            type_text = ", ".join(map(str, etype))
        elif isinstance(etype, str):
            type_text = etype
        else:
            type_text = ""

        aliases_list = getattr(entity, "aliases", []) or []
        aliases_text = ", ".join(map(str, aliases_list)) if aliases_list else ""

        desc = getattr(entity, "description", None)
        props = _as_props_dict(getattr(entity, "properties", {}) or {})
        source_docs_list = getattr(entity, "source_documents", []) or []
        source_docs_text = ", ".join(map(str, source_docs_list)) if source_docs_list else ""

        if lines:
            lines.append("")
            lines.append("---")
            lines.append("")
        lines.append(f"name: {name}")
        if eid:
            lines.append(f"id: {eid}")
        if type_text:
            lines.append(f"type: {type_text}")
        if desc:
            lines.append(f"description: {desc}")
        if aliases_text:
            lines.append(f"aliases: {aliases_text}")
        if source_docs_text:
            if resolve_source_documents and graph_query_utils is not None:
                section_lines = graph_query_utils.format_source_document_sections(
                    source_docs_list,
                    include_document_ids=include_document_ids_in_source_sections,
                )
                section_label = graph_query_utils.meta.get('section_label', 'Document')
                if section_lines:
                    lines.append(f"{section_label.lower()}_count: {len(section_lines)}")
                    lines.append(f"{section_label.lower()}_details:")
                    lines.extend(section_lines)
                elif source_docs_list:
                    lines.append(f"source_document_count: {len(source_docs_list)}")
            else:
                if len(source_docs_list) <= 12:
                    lines.append(f"source_documents: {source_docs_text}")
                else:
                    lines.append(f"source_document_count: {len(source_docs_list)}")

        if isinstance(props, dict) and props:
            prop_lines = []
            for key, val in props.items():
                if val in (None, "", [], {}, ()):
                    continue
                if key == "name" and (val == name or val in aliases_list):
                    continue
                prop_lines.append(f"- {key}: {val}")
            if prop_lines:
                lines.append("properties:")
                lines.extend(prop_lines)

    return "\n".join(lines) if lines else "No entities found."


def _clamp01(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
    except Exception:
        v = default
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


def _normalize_entity_type_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").strip().lower())


def _resolve_entity_type_name(requested: Any, available_entity_types: List[str]) -> tuple[str, bool]:
    available = [str(item or "").strip() for item in (available_entity_types or []) if str(item or "").strip()]
    if not available:
        return "Entity", False

    raw = str(requested or "").strip()
    if not raw:
        return ("Entity", True) if "Entity" in available else (available[0], False)
    if raw in available:
        return raw, True

    normalized_available: Dict[str, str] = {}
    for item in available:
        normalized_available.setdefault(_normalize_entity_type_key(item), item)

    normalized_requested = _normalize_entity_type_key(raw)
    if normalized_requested in normalized_available:
        return normalized_available[normalized_requested], True

    alias_targets = {
        "person": "Character",
        "people": "Character",
        "human": "Character",
        "humans": "Character",
        "character": "Character",
        "characters": "Character",
        "event": "Event",
        "events": "Event",
        "object": "Object",
        "objects": "Object",
        "item": "Object",
        "items": "Object",
        "artifact": "Object",
        "artifacts": "Object",
        "concept": "Concept",
        "concepts": "Concept",
        "idea": "Concept",
        "ideas": "Concept",
        "theme": "Concept",
        "themes": "Concept",
        "occasion": "Occasion",
        "occasions": "Occasion",
        "scene": "Occasion",
        "scenes": "Occasion",
        "moment": "Occasion",
        "moments": "Occasion",
        "location": "Location",
        "locations": "Location",
        "place": "Location",
        "places": "Location",
        "setting": "Location",
        "settings": "Location",
        "time": "TimePoint",
        "times": "TimePoint",
        "timepoint": "TimePoint",
        "timepoints": "TimePoint",
        "date": "TimePoint",
        "dates": "TimePoint",
        "episode": "Episode",
        "episodes": "Episode",
        "storyline": "Storyline",
        "storylines": "Storyline",
        "plotline": "Storyline",
        "plotlines": "Storyline",
        "plot": "Storyline",
        "document": "Document",
        "documents": "Document",
        "entity": "Entity",
        "entities": "Entity",
    }
    alias_target = alias_targets.get(normalized_requested)
    if alias_target and alias_target in available:
        return alias_target, True

    normalized_keys = list(normalized_available.keys())
    close_matches = difflib.get_close_matches(normalized_requested, normalized_keys, n=1, cutoff=0.86)
    if close_matches:
        return normalized_available[close_matches[0]], True

    if "Entity" in available:
        return "Entity", False
    return available[0], False


def _has_label(labels: Any, target: str) -> bool:
    if not isinstance(labels, list):
        return False
    t = str(target or "").strip()
    if not t:
        return False
    return any(str(x) == t for x in labels)


def _hybrid_search_episode_storyline(
    graph_query_utils,
    *,
    label: str,
    query: str,
    vector_top_k: int = 3,
    keyword_top_k: int = 2,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """
    Default hybrid strategy:
    - keyword top2 as primary retrieval channel
    - semantic vector top3 as supplemental channel
    - dedupe by id and rerank by merged score
    """
    raw_query = (query or "").strip()
    if not raw_query:
        return []
    vector_query = compress_query_for_vector_search(raw_query, top_k=8) or raw_query

    v_k = max(1, int(vector_top_k or 3))
    k_k = max(1, int(keyword_top_k or 2))
    out_k = max(1, int(top_k or 5))

    merged: Dict[str, Dict[str, Any]] = {}

    # A) keyword hits
    kw_hits = graph_query_utils.search_entities_by_type_ranked(label, keyword=raw_query, limit=max(10, k_k * 4)) or []
    kw_hits = kw_hits[:k_k]
    max_kw_score = max((float(row.get("keyword_score_raw", 0.0) or 0.0) for row in kw_hits), default=0.0)
    for idx, row in enumerate(kw_hits):
        ent = row.get("entity")
        eid = getattr(ent, "id", None)
        if not eid:
            continue
        raw_kw_score = float(row.get("keyword_score_raw", 0.0) or 0.0)
        rank_bonus = float((len(kw_hits) - idx) / max(1, len(kw_hits)))
        normalized_kw = raw_kw_score / max_kw_score if max_kw_score > 0 else 0.0
        kw_score = 0.8 * normalized_kw + 0.2 * rank_bonus
        merged[eid] = {
            "id": eid,
            "name": getattr(ent, "name", "") or "(未命名)",
            "labels": getattr(ent, "type", []) if isinstance(getattr(ent, "type", []), list) else [getattr(ent, "type", "")],
            "description": getattr(ent, "description", "") or "",
            "source_documents": getattr(ent, "source_documents", []) or [],
            "properties": _as_props_dict(getattr(ent, "properties", {}) or {}),
            "keyword_score": _clamp01(kw_score),
            "matched_keyword_count": int(row.get("matched_keyword_count", 0) or 0),
            "vector_score": 0.0,
        }

    # B) vector hits
    vec_hits: List[Dict[str, Any]] = []
    try:
        vec_hits = graph_query_utils.query_similar_entities(
            text=vector_query,
            top_k=max(10, v_k * 3),
            normalize=False,
            label_filter=label,
        ) or []
    except Exception:
        vec_hits = []
    vec_hits = vec_hits[:v_k]
    for r in vec_hits:
        eid = r.get("id")
        if not eid:
            continue
        raw = float(r.get("score", 0.0))  # cosine in [-1,1]
        vec_score = _clamp01((raw + 1.0) / 2.0)
        if eid not in merged:
            merged[eid] = {
                "id": eid,
                "name": r.get("name", "") or "(未命名)",
                "labels": r.get("labels", []) or [label],
                "description": r.get("description", "") or "",
                "source_documents": [],
                "properties": {},
                "keyword_score": 0.0,
                "matched_keyword_count": 0,
                "vector_score": vec_score,
            }
        else:
            merged[eid]["vector_score"] = max(float(merged[eid].get("vector_score", 0.0)), vec_score)

    if not merged:
        return []

    # hydrate stable fields (especially source_documents) by ids
    ent_map = graph_query_utils.get_entities_by_ids(list(merged.keys())) or {}
    for eid, item in merged.items():
        ent = ent_map.get(eid)
        if not ent:
            continue
        item["name"] = getattr(ent, "name", "") or item.get("name", "")
        item["description"] = getattr(ent, "description", "") or item.get("description", "")
        item["source_documents"] = getattr(ent, "source_documents", []) or item.get("source_documents", [])
        item["properties"] = _as_props_dict(getattr(ent, "properties", {}) or item.get("properties", {}))
        et = getattr(ent, "type", [])
        item["labels"] = et if isinstance(et, list) else ([et] if et else item.get("labels", []))

    # merged rank score: vector-prioritized hybrid
    rows = list(merged.values())
    for x in rows:
        ks = _clamp01(x.get("keyword_score", 0.0))
        vs = _clamp01(x.get("vector_score", 0.0))
        mk = max(0, int(x.get("matched_keyword_count", 0) or 0))
        keyword_coverage = _clamp01(mk / 3.0)
        x["score"] = 0.55 * ks + 0.15 * keyword_coverage + 0.30 * vs

    rows.sort(
        key=lambda x: (
            x.get("score", 0.0),
            x.get("keyword_score", 0.0),
            x.get("matched_keyword_count", 0),
            x.get("vector_score", 0.0),
        ),
        reverse=True,
    )
    return rows[:out_k]


def _hybrid_search_entities(
    graph_query_utils,
    *,
    entity_type: str,
    query: str,
    vector_top_k: int = 3,
    keyword_top_k: int = 5,
    top_k: int = 8,
) -> List[Dict[str, Any]]:
    raw_query = (query or "").strip()
    target_label = str(entity_type or "").strip() or "Entity"
    if not raw_query:
        ranked_rows = graph_query_utils.search_entities_by_type_ranked(target_label, keyword="", limit=max(1, int(top_k or 8))) or []
        out: List[Dict[str, Any]] = []
        for row in ranked_rows[: max(1, int(top_k or 8))]:
            ent = row.get("entity")
            if ent is None:
                continue
            out.append(
                {
                    "entity": ent,
                    "id": getattr(ent, "id", "") or "",
                    "name": getattr(ent, "name", "") or "(未命名)",
                    "labels": getattr(ent, "type", []) if isinstance(getattr(ent, "type", []), list) else [getattr(ent, "type", "")],
                    "description": getattr(ent, "description", "") or "",
                    "source_documents": getattr(ent, "source_documents", []) or [],
                    "properties": _as_props_dict(getattr(ent, "properties", {}) or {}),
                    "keyword_score": 0.0,
                    "matched_keyword_count": 0,
                    "vector_score": 0.0,
                    "score": 0.0,
                    "match_sources": ["list_all"],
                }
            )
        return out

    vector_query = compress_query_for_vector_search(raw_query, top_k=8) or raw_query
    v_k = max(1, int(vector_top_k or 3))
    k_k = max(1, int(keyword_top_k or 5))
    out_k = max(1, int(top_k or 8))
    merged: Dict[str, Dict[str, Any]] = {}

    kw_hits = graph_query_utils.search_entities_by_type_ranked(target_label, keyword=raw_query, limit=max(12, k_k * 4)) or []
    kw_hits = kw_hits[:k_k]
    max_kw_score = max((float(row.get("keyword_score_raw", 0.0) or 0.0) for row in kw_hits), default=0.0)
    for idx, row in enumerate(kw_hits):
        ent = row.get("entity")
        eid = getattr(ent, "id", None)
        if not eid:
            continue
        raw_kw_score = float(row.get("keyword_score_raw", 0.0) or 0.0)
        rank_bonus = float((len(kw_hits) - idx) / max(1, len(kw_hits)))
        normalized_kw = raw_kw_score / max_kw_score if max_kw_score > 0 else 0.0
        kw_score = 0.8 * normalized_kw + 0.2 * rank_bonus
        merged[eid] = {
            "entity": ent,
            "id": eid,
            "name": getattr(ent, "name", "") or "(未命名)",
            "labels": getattr(ent, "type", []) if isinstance(getattr(ent, "type", []), list) else [getattr(ent, "type", "")],
            "description": getattr(ent, "description", "") or "",
            "source_documents": getattr(ent, "source_documents", []) or [],
            "properties": _as_props_dict(getattr(ent, "properties", {}) or {}),
            "keyword_score": _clamp01(kw_score),
            "matched_keyword_count": int(row.get("matched_keyword_count", 0) or 0),
            "vector_score": 0.0,
            "match_sources": ["keyword"],
        }

    vec_hits: List[Dict[str, Any]] = []
    try:
        label_filter = None if target_label == "Entity" else target_label
        vec_hits = graph_query_utils.query_similar_entities(
            text=vector_query,
            top_k=max(12, v_k * 4),
            normalize=False,
            label_filter=label_filter,
        ) or []
    except Exception:
        vec_hits = []
    vec_hits = vec_hits[:v_k]
    for r in vec_hits:
        eid = r.get("id")
        if not eid:
            continue
        raw_vec = float(r.get("score", 0.0) or 0.0)
        vec_score = _clamp01((raw_vec + 1.0) / 2.0)
        if eid not in merged:
            merged[eid] = {
                "entity": None,
                "id": eid,
                "name": r.get("name", "") or "(未命名)",
                "labels": r.get("labels", []) or ([target_label] if target_label else []),
                "description": r.get("description", "") or "",
                "source_documents": [],
                "properties": {},
                "keyword_score": 0.0,
                "matched_keyword_count": 0,
                "vector_score": vec_score,
                "match_sources": ["vector"],
            }
        else:
            merged[eid]["vector_score"] = max(float(merged[eid].get("vector_score", 0.0) or 0.0), vec_score)
            sources = list(merged[eid].get("match_sources") or [])
            if "vector" not in sources:
                sources.append("vector")
            merged[eid]["match_sources"] = sources

    if not merged:
        return []

    ent_map = graph_query_utils.get_entities_by_ids(list(merged.keys())) or {}
    for eid, item in merged.items():
        ent = ent_map.get(eid)
        if not ent:
            continue
        item["entity"] = ent
        item["name"] = getattr(ent, "name", "") or item.get("name", "")
        item["description"] = getattr(ent, "description", "") or item.get("description", "")
        item["source_documents"] = getattr(ent, "source_documents", []) or item.get("source_documents", [])
        item["properties"] = _as_props_dict(getattr(ent, "properties", {}) or item.get("properties", {}))
        et = getattr(ent, "type", [])
        item["labels"] = et if isinstance(et, list) else ([et] if et else item.get("labels", []))

    rows = list(merged.values())
    for x in rows:
        ks = _clamp01(x.get("keyword_score", 0.0))
        vs = _clamp01(x.get("vector_score", 0.0))
        mk = max(0, int(x.get("matched_keyword_count", 0) or 0))
        keyword_coverage = _clamp01(mk / 3.0)
        x["score"] = 0.52 * ks + 0.13 * keyword_coverage + 0.35 * vs
        sources = list(x.get("match_sources") or [])
        if "keyword" in sources and "vector" in sources:
            x["match_mode"] = "hybrid"
        elif "vector" in sources:
            x["match_mode"] = "vector"
        elif "keyword" in sources:
            x["match_mode"] = "fuzzy"
        else:
            x["match_mode"] = "unknown"

    rows.sort(
        key=lambda x: (
            x.get("score", 0.0),
            x.get("keyword_score", 0.0),
            x.get("matched_keyword_count", 0),
            x.get("vector_score", 0.0),
        ),
        reverse=True,
    )
    return rows[:out_k]


def _build_candidate_text(row: Dict[str, Any]) -> str:
    name = str(row.get("name", "") or "").strip()
    description = str(row.get("description", "") or "").strip()
    labels = row.get("labels") or []
    if not isinstance(labels, list):
        labels = [labels]
    label_text = ", ".join(str(x) for x in labels if str(x).strip())
    parts: List[str] = []
    if label_text:
        parts.append(f"type: {label_text}")
    if name:
        parts.append(f"name: {name}")
    if description:
        parts.append(f"description: {description}")
    props = _as_props_dict(row.get("properties", {}) or {})
    if props:
        prop_lines = []
        for key, value in props.items():
            if value in (None, "", [], {}, ()):
                continue
            prop_lines.append(f"{key}: {value}")
        if prop_lines:
            parts.append("properties:\n" + "\n".join(prop_lines))
    return "\n".join(parts).strip()


def _parse_relevance_result(raw: str) -> Dict[str, Any]:
    try:
        data = json.loads(correct_json_format(raw))
    except Exception:
        data = {}
    probability = _clamp01(data.get("probability", 0.0))
    is_relevant = _to_bool(data.get("is_relevant"), probability >= 0.5)
    reason = str(data.get("reason", "") or "").strip()
    return {
        "probability": probability,
        "is_relevant": is_relevant,
        "reason": reason,
    }


def _apply_llm_filter_to_rows(
    rows: List[Dict[str, Any]],
    *,
    query: str,
    document_parser: Any,
    threshold: float = 0.35,
    top_k: int = 5,
    max_workers: int = 4,
) -> List[Dict[str, Any]]:
    if not rows or document_parser is None:
        return rows[: max(1, int(top_k or 5))]

    thr = _clamp01(threshold, default=0.35)

    def _score(row: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        rid = str(row.get("id") or "")
        try:
            candidate_text = _build_candidate_text(row)
            raw = document_parser.score_candidate_relevance(
                text=candidate_text,
                goal=query,
            )
            parsed = _parse_relevance_result(raw)
        except Exception:
            parsed = {"probability": 0.0, "is_relevant": False, "reason": ""}
        return rid, parsed

    outputs: Dict[str, Dict[str, Any]] = {}
    worker_count = max(1, min(int(max_workers or 4), len(rows)))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(_score, row) for row in rows]
        for future in as_completed(futures):
            rid, parsed = future.result()
            if rid:
                outputs[rid] = parsed

    ranked: List[Dict[str, Any]] = []
    for row in rows:
        parsed = outputs.get(str(row.get("id") or ""), {})
        new_row = dict(row)
        prob = _clamp01(parsed.get("probability", 0.0))
        is_rel = _to_bool(parsed.get("is_relevant"), prob >= thr)
        new_row["llm_probability"] = prob
        new_row["llm_is_relevant"] = is_rel
        new_row["llm_reason"] = str(parsed.get("reason", "") or "").strip()
        new_row["score"] = 0.55 * prob + 0.45 * _clamp01(row.get("score", 0.0))
        ranked.append(new_row)

    kept = [
        row for row in ranked
        if row.get("llm_is_relevant") or float(row.get("llm_probability", 0.0)) >= thr
    ]

    ranked.sort(
        key=lambda x: (
            x.get("score", 0.0),
            x.get("llm_probability", 0.0),
            x.get("keyword_score", 0.0),
            x.get("vector_score", 0.0),
        ),
        reverse=True,
    )
    kept.sort(
        key=lambda x: (
            x.get("score", 0.0),
            x.get("llm_probability", 0.0),
            x.get("keyword_score", 0.0),
            x.get("vector_score", 0.0),
        ),
        reverse=True,
    )

    if kept:
        return kept[: max(1, int(top_k or 5))]

    fallback = ranked[:1]
    for row in fallback:
        row["llm_fallback"] = True
    return fallback


def search_episode_storyline_candidates(
    graph_query_utils,
    *,
    label: str,
    query: str,
    vector_top_k: int = 3,
    keyword_top_k: int = 2,
    top_k: int = 5,
    use_llm_filter: bool = False,
    llm_filter_top_k: int = 8,
    llm_filter_threshold: float = 0.35,
    document_parser: Any = None,
    max_workers: int = 4,
) -> List[Dict[str, Any]]:
    base_top_k = max(int(top_k or 5), int(llm_filter_top_k or 8)) if use_llm_filter else int(top_k or 5)
    rows = _hybrid_search_episode_storyline(
        graph_query_utils,
        label=label,
        query=query,
        vector_top_k=vector_top_k,
        keyword_top_k=keyword_top_k,
        top_k=base_top_k,
    )
    if use_llm_filter:
        return _apply_llm_filter_to_rows(
            rows,
            query=query,
            document_parser=document_parser,
            threshold=llm_filter_threshold,
            top_k=top_k,
            max_workers=max_workers,
    )
    return rows[: max(1, int(top_k or 5))]


def _pick_primary_type(labels: Any) -> str:
    if isinstance(labels, str):
        labels = [labels]
    if not isinstance(labels, list):
        return "Unknown"
    clean = [str(x).strip() for x in labels if str(x).strip() and str(x).strip() != "Entity"]
    if clean:
        return clean[0]
    return "Entity" if labels else "Unknown"


def _parse_json_listish(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    if not isinstance(value, str):
        return []
    raw = value.strip()
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if str(x).strip()]
    except Exception:
        pass
    return [x.strip() for x in raw.split(",") if x.strip()]


def _parse_int_maybe(value: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        if value is None or str(value).strip() == "":
            return default
        return int(value)
    except Exception:
        return default


def _count_nodes_with_label(graph_query_utils, label: str) -> int:
    return int(graph_query_utils.count_nodes_with_label(str(label or "").strip()) or 0)


def _community_level(row: Dict[str, Any]) -> Optional[int]:
    props = _as_props_dict(row.get("properties", {}) or {})
    val = props.get("level")
    try:
        return int(val) if val is not None else None
    except Exception:
        return None


def _hydrate_community_rows_from_summary_store(
    summary_vector_store,
    rows: List[Dict[str, Any]],
    *,
    prefer_vector_text: bool = True,
) -> List[Dict[str, Any]]:
    if summary_vector_store is None or not rows:
        return rows

    report_ids: List[str] = []
    for row in rows:
        props = _as_props_dict(row.get("properties", {}) or {})
        report_id = str(props.get("report_id") or "").strip()
        community_id = str(row.get("id") or "").strip()
        if not report_id and community_id:
            report_id = f"{community_id}_report"
        if report_id:
            report_ids.append(report_id)

    if not report_ids:
        return rows

    docs = summary_vector_store.search_by_ids(report_ids) or []
    report_by_community: Dict[str, Any] = {}
    for doc in docs:
        meta = getattr(doc, "metadata", {}) or {}
        community_id = str(meta.get("community_id") or "").strip()
        if not community_id:
            doc_id = str(getattr(doc, "id", "") or "").strip()
            community_id = doc_id[:-7] if doc_id.endswith("_report") else ""
        if community_id:
            report_by_community[community_id] = doc

    for row in rows:
        community_id = str(row.get("id") or "").strip()
        doc = report_by_community.get(community_id)
        if doc is None:
            continue
        meta = getattr(doc, "metadata", {}) or {}
        props = dict(_as_props_dict(row.get("properties", {}) or {}))
        props.setdefault("report_id", str(getattr(doc, "id", "") or "").strip())
        if "level" not in props:
            level = _parse_int_maybe(meta.get("level"))
            if level is not None:
                props["level"] = level
        if "member_count" not in props:
            member_count = _parse_int_maybe(meta.get("member_count"))
            if member_count is not None:
                props["member_count"] = member_count
        if "rating" not in props:
            rating = meta.get("rating")
            try:
                props["rating"] = float(rating)
            except Exception:
                pass
        top_members = _parse_json_listish(meta.get("top_members"))
        if top_members and "top_members" not in props:
            props["top_members"] = top_members
        row["properties"] = props
        summary_text = str(meta.get("summary") or "").strip()
        if prefer_vector_text or not str(row.get("description", "") or "").strip():
            row["description"] = summary_text or str(getattr(doc, "content", "") or "").strip()
        source_documents = []
        for item in list(row.get("source_documents") or []) + _parse_json_listish(meta.get("source_documents")):
            s = str(item or "").strip()
            if s and s not in source_documents:
                source_documents.append(s)
        row["source_documents"] = source_documents
    return rows


def _vector_search_community_candidates(
    graph_query_utils,
    summary_vector_store,
    *,
    query: str,
    limit: int,
    level: Optional[int] = None,
) -> List[Dict[str, Any]]:
    if summary_vector_store is None:
        return []

    docs = summary_vector_store.search(query=query, limit=max(1, int(limit or 5))) or []
    merged: Dict[str, Dict[str, Any]] = {}
    for doc in docs:
        meta = getattr(doc, "metadata", {}) or {}
        community_id = str(meta.get("community_id") or "").strip()
        if not community_id:
            doc_id = str(getattr(doc, "id", "") or "").strip()
            community_id = doc_id[:-7] if doc_id.endswith("_report") else ""
        if not community_id:
            continue
        row_level = _parse_int_maybe(meta.get("level"))
        if level is not None and row_level != int(level):
            continue
        vector_score = _clamp01(meta.get("similarity_score", 0.0))
        row = {
            "id": community_id,
            "name": str(meta.get("title") or meta.get("community_name") or community_id).strip() or community_id,
            "labels": ["Community"],
            "description": str(meta.get("summary") or getattr(doc, "content", "") or "").strip(),
            "source_documents": _parse_json_listish(meta.get("source_documents")),
            "properties": {
                "report_id": str(getattr(doc, "id", "") or "").strip(),
                "level": row_level,
                "member_count": _parse_int_maybe(meta.get("member_count")),
                "top_members": _parse_json_listish(meta.get("top_members")),
                "rating": meta.get("rating"),
                "rating_explanation": str(meta.get("rating_explanation") or "").strip(),
            },
            "keyword_score": 0.0,
            "matched_keyword_count": 0,
            "vector_score": vector_score,
            "score": vector_score,
        }
        if community_id not in merged or float(merged[community_id].get("score", 0.0)) < vector_score:
            merged[community_id] = row

    if not merged:
        return []

    ent_map = graph_query_utils.get_entities_by_ids(list(merged.keys())) or {}
    rows: List[Dict[str, Any]] = []
    for community_id, row in merged.items():
        ent = ent_map.get(community_id)
        if ent is not None:
            props = _as_props_dict(getattr(ent, "properties", {}) or {})
            vector_props = _as_props_dict(row.get("properties", {}) or {})
            props.update({k: v for k, v in vector_props.items() if v not in (None, "", [], {}, ())})
            row["properties"] = props
            row["name"] = getattr(ent, "name", "") or row.get("name", community_id)
            row["labels"] = getattr(ent, "type", []) or row.get("labels", ["Community"])
            if not row.get("source_documents"):
                row["source_documents"] = getattr(ent, "source_documents", []) or []
        rows.append(row)

    rows.sort(
        key=lambda x: (
            x.get("score", 0.0),
            x.get("vector_score", 0.0),
            x.get("name", ""),
        ),
        reverse=True,
    )
    return rows


def search_community_candidates(
    graph_query_utils,
    *,
    summary_vector_store=None,
    query: str,
    top_k: int = 5,
    level: Optional[int] = None,
    vector_top_k: int = 3,
    keyword_top_k: int = 2,
    use_llm_filter: bool = False,
    llm_filter_top_k: int = 8,
    llm_filter_threshold: float = 0.35,
    document_parser: Any = None,
    max_workers: int = 4,
) -> List[Dict[str, Any]]:
    base_top_k = max(int(top_k or 5), int(llm_filter_top_k or 8), 80 if level is not None else 20)
    summary_doc_count = 0
    if summary_vector_store is not None:
        try:
            summary_doc_count = int((summary_vector_store.get_stats() or {}).get("document_count", 0) or 0)
        except Exception:
            summary_doc_count = 0
    if summary_doc_count <= 0 and _count_nodes_with_label(graph_query_utils, "Community") <= 0:
        return []
    rows = _vector_search_community_candidates(
        graph_query_utils,
        summary_vector_store,
        query=query,
        limit=base_top_k,
        level=level,
    )
    if not rows:
        rows = _hybrid_search_episode_storyline(
            graph_query_utils,
            label="Community",
            query=query,
            vector_top_k=max(vector_top_k, 3),
            keyword_top_k=max(keyword_top_k, 2),
            top_k=base_top_k,
        )
        if level is not None:
            rows = [row for row in rows if _community_level(row) == int(level)]
        rows = _hydrate_community_rows_from_summary_store(summary_vector_store, rows)
    if use_llm_filter:
        rows = _apply_llm_filter_to_rows(
            rows,
            query=query,
            document_parser=document_parser,
            threshold=llm_filter_threshold,
            top_k=top_k,
            max_workers=max_workers,
        )
    return rows[: max(1, int(top_k or 5))]


def _format_episode_storyline_hits(rows: List[Dict[str, Any]], *, title: str, include_meta: bool = False) -> str:
    if not rows:
        return f"未找到{title}。"
    lines: List[str] = [f"{title}检索结果（默认 hybrid：关键词Top2 + 向量Top3，已去重）："]
    for i, r in enumerate(rows, 1):
        rid = r.get("id") or "UNKNOWN_ID"
        name = r.get("name") or "(unnamed)"
        score = float(r.get("score", 0.0))
        kscore = float(r.get("keyword_score", 0.0))
        vscore = float(r.get("vector_score", 0.0))
        lines.append(f"{i}. {name} [ID: {rid}] score={score:.4f} (vec={vscore:.4f}, kw={kscore:.4f})")
        if "llm_probability" in r:
            llm_prob = float(r.get("llm_probability", 0.0))
            extra = " [fallback]" if r.get("llm_fallback") else ""
            reason = str(r.get("llm_reason", "") or "").strip()
            lines.append(f"   llm_relevance: {llm_prob:.4f}{extra}")
            if reason and include_meta:
                lines.append(f"   llm_reason: {reason}")
        desc = (r.get("description") or "").strip()
        if desc:
            lines.append(f"   description: {desc}")
        src_docs = r.get("source_documents") or []
        if src_docs:
            lines.append(f"   source_documents(document_id): {', '.join(str(x) for x in src_docs)}")
        if include_meta:
            props = _as_props_dict(r.get("properties", {}) or {})
            prop_lines = []
            for k, v in props.items():
                if v in (None, "", [], {}, ()):
                    continue
                prop_lines.append(f"{k}={v}")
            if prop_lines:
                lines.append(f"   properties: {'; '.join(prop_lines)}")
    return "\n".join(lines)


def _format_section_related_entities(
    graph_query_utils,
    section_id: str,
    *,
    related_entity_limit: int = 3,
) -> List[str]:
    if not graph_query_utils or not section_id:
        return []
    related = graph_query_utils.search_related_entities(
        source_id=section_id,
        limit=max(related_entity_limit * 8, 12),
        return_relations=False,
    ) or []
    if not related:
        return []

    grouped: Dict[str, List[str]] = {}
    for entity in related:
        label = _pick_primary_type(getattr(entity, "type", []) or [])
        grouped.setdefault(label, [])
        if len(grouped[label]) >= related_entity_limit:
            continue
        entity_id = str(getattr(entity, "id", "") or "").strip()
        entity_name = str(getattr(entity, "name", "") or "").strip() or "(未命名)"
        grouped[label].append(f"{entity_name} [{entity_id}]")

    if not grouped:
        return []

    lines = ["   相关实体:"]
    for label in sorted(grouped.keys()):
        values = grouped.get(label) or []
        if values:
            lines.append(f"   - {label}: {', '.join(values)}")
    return lines


def _format_community_hits(
    rows: List[Dict[str, Any]],
    *,
    include_meta: bool = False,
    include_member_preview: bool = True,
    member_preview_limit: int = 5,
) -> str:
    if not rows:
        return "未找到Community。"
    lines: List[str] = ["Community检索结果："]
    for idx, row in enumerate(rows, 1):
        rid = row.get("id") or "UNKNOWN_ID"
        name = row.get("name") or "(未命名)"
        score = float(row.get("score", 0.0))
        level = _community_level(row)
        props = _as_props_dict(row.get("properties", {}) or {})
        member_count = props.get("member_count")
        lines.append(f"{idx}. {name} [ID: {rid}] score={score:.4f}")
        if level is not None:
            lines.append(f"   level: {level}")
        if member_count is not None:
            lines.append(f"   member_count: {member_count}")
        rating = props.get("rating")
        if rating not in (None, ""):
            lines.append(f"   rating: {rating}")
        llm_prob = row.get("llm_probability")
        if llm_prob is not None:
            lines.append(f"   LLM相关性: {float(llm_prob):.4f}")
            reason = str(row.get("llm_reason", "") or "").strip()
            if reason and include_meta:
                lines.append(f"   LLM判断: {reason}")
        desc = safe_str(row.get("description")).strip()
        if desc:
            lines.append(f"   summary: {desc}")
        source_docs = row.get("source_documents") or []
        if source_docs:
            lines.append(f"   source_documents(document_id): {', '.join(str(x) for x in source_docs)}")
        if include_member_preview:
            preview = [safe_str(x) for x in safe_list(props.get("top_members")) if safe_str(x)]
            if preview:
                lines.append(f"   top_members: {', '.join(preview[: max(1, int(member_preview_limit or 5))])}")
        if include_meta:
            extra = []
            for key, value in props.items():
                if key in {"top_members", "member_count", "level"}:
                    continue
                if value in (None, "", [], {}, ()):
                    continue
                extra.append(f"{key}={value}")
            if extra:
                lines.append(f"   properties: {'; '.join(extra)}")
    return "\n".join(lines)


def _to_bool(val, default=False):
    if isinstance(val, bool):
        return val
    if val is None:
        return default
    if isinstance(val, (int, float)):
        return bool(val)
    if isinstance(val, str):
        return val.strip().lower() in {"1", "true", "yes", "y", "on"}
    return default

def _as_list(val):
    if val is None:
        return None
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        s = val.strip()
        if not s:
            return []
        return [x.strip() for x in s.split(",")]
    return [val]

def _fmt_entity_line(e) -> str:
    _labels = e.type if isinstance(e.type, list) else ([e.type] if e.type else [])
    etype = "/".join(_labels) if _labels else "未知类型"
    return f"- {e.name}  [ID: {e.id}]  <{etype}>"

def _fmt_relation_line(rel) -> str:
    pred = getattr(rel, "predicate", "") or "UNKNOWN_REL"
    rid = getattr(rel, "id", "")
    rn = getattr(rel, "relation_name", "") or ""
    props = _as_props_dict(getattr(rel, "properties", {}) or {})
    if not rn:
        rn = props.get("relation_name") or ""
    desc = props.get("description") or ""
    return f"  ↳ {pred}{('('+rn+')' if rn else '')} [rel_id: {rid}]  {('description: '+desc) if desc else ''}"

def _fmt_chain(ids: List[str], graph_query_utils) -> str:
    parts = []
    for _id in ids:
        node = graph_query_utils.get_entity_by_id(_id)
        if node:
            parts.append(f"{node.name}({_id})")
        else:
            parts.append(_id)
    return " -> ".join(parts)


# =========================
# 工具类
# =========================

@register_tool("retrieve_entity_by_name")
class EntityRetrieverName(BaseTool):
    name = "retrieve_entity_by_name"
    description = (
        "按指定实体类型执行混合实体检索。"
        "同时结合名称/别名模糊匹配与语义向量匹配，并对结果去重重排。"
        "当 entity_type 无效或未提供时回退为 'Entity'；"
        "当 query 为空字符串时返回该类型下的实体列表。"
        "可选把 source_documents 去重映射成可读的场次/章节标题。"
    )
    parameters = [
        {"name": "query", "type": "string", "description": "实体名称、别名或自然语言描述；可为空以列出该类型实体。", "required": True},
        {"name": "entity_type", "type": "string", "description": "目标实体类型；若无效将安全回退为 'Entity'。", "required": False},
        {"name": "top_k", "type": "number", "description": "返回结果数量上限，默认 8。", "required": False},
        {"name": "resolve_source_documents", "type": "bool", "description": "是否把实体的 source_documents 去重映射成可读的场次/章节标题，默认 False。", "required": False},
    ]

    def __init__(self, graph_query_utils, embedding_config):
        self.graph_query_utils = graph_query_utils
        self.graph_query_utils.load_embedding_model(embedding_config)

    def call(self, params: str, **kwargs) -> str:
        logger.info("🔎 调用 retrieve_entity_by_name")
        params_dict = json.loads(params)
        query = params_dict.get("query", "")
        requested_entity_type = params_dict.get("entity_type", "Entity")
        top_k = max(1, int(params_dict.get("top_k", 8) or 8))
        resolve_source_documents = _to_bool(params_dict.get("resolve_source_documents"), False)
        available_entity_types = self.graph_query_utils.list_entity_types()
        entity_type, matched = _resolve_entity_type_name(requested_entity_type, available_entity_types)
        if not matched:
            logger.info(
                "❗ 未找到实体类型 '%s'，回退为 '%s'。当前图可用类型: %s",
                requested_entity_type,
                entity_type,
                ", ".join(available_entity_types) if available_entity_types else "(none)",
            )

        rows = _hybrid_search_entities(
            self.graph_query_utils,
            entity_type=entity_type,
            query=query,
            vector_top_k=max(4, top_k),
            keyword_top_k=max(6, top_k),
            top_k=top_k,
        )
        results = [row.get("entity") for row in rows if row.get("entity") is not None]
        return format_entity_results(
            results,
            graph_query_utils=self.graph_query_utils,
            resolve_source_documents=resolve_source_documents,
            include_document_ids_in_source_sections=resolve_source_documents,
        )

@register_tool("retrieve_entity_by_id")
class EntityRetrieverID(BaseTool):
    name = "retrieve_entity_by_id"
    description = (
        "根据实体 ID 返回实体信息。可选返回属性、关系，以及由 source_documents 去重映射出的场次/章节标题。"
    )
    parameters = [
        {"name": "entity_id", "type": "string", "description": "实体唯一 ID。", "required": True},
        {"name": "contain_properties", "type": "bool", "description": "是否包含属性字段，默认 False。", "required": False},
        {"name": "contain_relations", "type": "bool", "description": "是否包含与其它实体的关系列表，默认 False。", "required": False},
        {"name": "resolve_source_documents", "type": "bool", "description": "是否把 source_documents 去重映射成可读的场次/章节标题，默认 False。", "required": False},
    ]

    def __init__(self, graph_query_utils, embedding_config=None):
        self.graph_query_utils = graph_query_utils
        if embedding_config:
            self.graph_query_utils.load_embedding_model(embedding_config)

    def call(self, params: str, **kwargs) -> str:
        logger.info("🔎 调用 retrieve_entity_by_id")
        params_dict = json.loads(params) if isinstance(params, str) else dict(params or {})
        entity_id = params_dict.get("entity_id")
        contain_properties = _to_bool(params_dict.get("contain_properties"), False)
        contain_relations = _to_bool(params_dict.get("contain_relations"), False)
        resolve_source_documents = _to_bool(params_dict.get("resolve_source_documents"), False)
        return self.graph_query_utils.get_entity_info(
            entity_id,
            contain_properties,
            contain_relations,
            resolve_source_documents,
        )


@register_tool("get_entity_sections")
class GetEntitySections(BaseTool):
    name = "get_entity_sections"
    description = "根据实体 ID 查看该实体出现过哪些场次/章节。默认同时显示真实 document_id，并自动补充明显同指的名称变体实体。"
    parameters = [
        {"name": "entity_id", "type": "string", "description": "实体唯一 ID。", "required": True},
        {"name": "include_document_ids", "type": "bool", "description": "是否同时显示每个场次/章节对应的 document_id，默认 True。", "required": False},
    ]

    def __init__(self, graph_query_utils, embedding_config=None):
        self.graph_query_utils = graph_query_utils
        if embedding_config:
            self.graph_query_utils.load_embedding_model(embedding_config)

    def call(self, params: str, **kwargs) -> str:
        logger.info("🔎 调用 get_entity_sections")
        params_dict = json.loads(params) if isinstance(params, str) else dict(params or {})
        entity_id = params_dict.get("entity_id")
        include_document_ids = _to_bool(params_dict.get("include_document_ids"), True)
        return self.graph_query_utils.get_entity_section_info(
            entity_id,
            include_document_ids=include_document_ids,
        )

@register_tool("search_related_entities")
class SearchRelatedEntities(BaseTool):
    name = "search_related_entities"
    description = (
        "给定实体 ID，检索与之相连的相关实体。"
        "可按谓词、关系类型与实体类型过滤；"
        "支持返回 (实体, 关系) 的详细模式或仅返回实体。"
    )
    parameters = [
        {"name": "source_id", "type": "string", "description": "起点实体 ID。", "required": True},
        {"name": "predicate", "type": "string", "description": "关系谓词过滤（可选）。", "required": False},
        {"name": "relation_types", "type": "array", "description": "关系类型过滤（字符串数组，可选）。", "required": False},
        {"name": "entity_types", "type": "array", "description": "目标实体类型过滤（字符串数组，可选）。", "required": False},
        {"name": "limit", "type": "number", "description": "返回条数上限（可选）。", "required": False},
        {"name": "return_relations", "type": "bool", "description": "是否返回 (实体, 关系) 对而非仅实体，默认 False。", "required": False},
    ]

    def __init__(self, graph_query_utils, embedding_config=None):
        self.graph_query_utils = graph_query_utils
        if embedding_config:
            self.graph_query_utils.load_embedding_model(embedding_config)

    def call(self, params: str, **kwargs) -> str:
        logger.info("🔎 调用 search_related_entities")
        data = json.loads(params) if isinstance(params, str) else dict(params or {})
        source_id = data.get("source_id")
        if not source_id:
            raise ValueError("Missing required parameter: source_id")

        predicate = data.get("predicate") or None
        relation_types = _as_list(data.get("relation_types"))
        entity_types = _as_list(data.get("entity_types"))
        limit = data.get("limit")
        return_relations = _to_bool(data.get("return_relations"), default=False)

        results = self.graph_query_utils.search_related_entities(
            source_id=source_id,
            predicate=predicate,
            relation_types=relation_types,
            entity_types=entity_types,
            limit=int(limit) if isinstance(limit, (int, float, str)) and str(limit).isdigit() else None,
            return_relations=return_relations
        )

        if not results:
            return "未找到相关实体。"

        lines = []
        if return_relations:
            lines.append("Related entities (with relations):")
            for ent, rel in results:
                lines.append(_fmt_entity_line(ent))
                lines.append(_fmt_relation_line(rel))
        else:
            lines.append("Related entities:")
            for ent in results:
                lines.append(_fmt_entity_line(ent))

        return "\n".join(lines)

@register_tool("get_relations_between_entities")
class GetRelationsBetweenEntities(BaseTool):
    name = "get_relations_between_entities"
    description = (
        "给定两个实体 ID 与关系类型，返回这两个实体之间该关系的可读说明。"
        "适合核对两个实体之间是否存在某种明确关系，以及该关系的描述内容。"
    )
    parameters = [
        {"name": "src_id", "type": "string", "description": "源实体ID（必填）", "required": True},
        {"name": "tgt_id", "type": "string", "description": "目标实体ID（必填）", "required": True},
        {"name": "relation_type", "type": "string", "description": "关系类型（如 'EVENT_CAUSES'）", "required": True},
    ]

    def __init__(self, graph_query_utils, embedding_config=None):
        self.graph_query_utils = graph_query_utils

    def call(self, params: str, **kwargs) -> str:
        logger.info("🔎 调用 get_relations_between_entities")
        data = json.loads(params) if isinstance(params, str) else dict(params or {})
        src_id = data.get("src_id")
        tgt_id = data.get("tgt_id")
        relation_type = data.get("relation_type")
        if not (src_id and tgt_id and relation_type):
            raise ValueError("缺少必要参数：src_id / tgt_id / relation_type")

        txt = self.graph_query_utils.get_relation_summary(src_id, tgt_id, relation_type)
        return txt or "未找到指定关系。"

@register_tool("get_common_neighbors")
class GetCommonNeighbors(BaseTool):
    name = "get_common_neighbors"
    description = (
        "返回两个实体的共同邻居。支持限定关系类型与方向；"
        "可选择是否附带从A/B到该邻居的关系类型列表。"
    )
    parameters = [
        {"name": "id1", "type": "string", "description": "第一个实体ID（必填）", "required": True},
        {"name": "id2", "type": "string", "description": "第二个实体ID（必填）", "required": True},
        {"name": "rel_types", "type": "array", "description": "关系类型白名单（如 ['RELATED_TO']）", "required": False},
        {"name": "direction", "type": "string", "description": "方向：any/out/in（默认 any）", "required": False},
        {"name": "limit", "type": "number", "description": "返回上限", "required": False},
        {"name": "include_rel_types", "type": "bool", "description": "是否附带从A/B出发的关系类型（默认 False）", "required": False},
    ]

    def __init__(self, graph_query_utils, embedding_config=None):
        self.graph_query_utils = graph_query_utils

    def call(self, params: str, **kwargs) -> str:
        logger.info("🔎 get_common_neighbors")
        data = json.loads(params) if isinstance(params, str) else dict(params or {})
        id1 = data.get("id1")
        id2 = data.get("id2")
        if not (id1 and id2):
            raise ValueError("缺少必要参数：id1 / id2")

        rel_types = _as_list(data.get("rel_types"))
        direction = (data.get("direction") or "any").lower()
        if direction not in {"any", "out", "in"}:
            direction = "any"
        limit_raw = data.get("limit")
        limit = int(limit_raw) if isinstance(limit_raw, (int, float, str)) and str(limit_raw).isdigit() else None
        include_rel_types = _to_bool(data.get("include_rel_types"), default=False)

        if include_rel_types:
            items = self.graph_query_utils.get_common_neighbors_with_rels(
                id1=id1, id2=id2, rel_types=rel_types, direction=direction, limit=limit
            )
            if not items:
                return "无共同邻居。"
            lines = ["共同邻居（含从A/B的边类型）:"]
            for it in items:
                ent = it["entity"]
                lines.append(_fmt_entity_line(ent))
                lines.append(f"  ←A: {', '.join(it.get('rels_from_a', []) or [])}")
                lines.append(f"  ←B: {', '.join(it.get('rels_from_b', []) or [])}")
            return "\n".join(lines)
        else:
            ents = self.graph_query_utils.get_common_neighbors(
                id1=id1, id2=id2, rel_types=rel_types, direction=direction, limit=limit
            )
            if not ents:
                return "无共同邻居。"
            lines = ["共同邻居："]
            for e in ents:
                lines.append(_fmt_entity_line(e))
            return "\n".join(lines)

@register_tool("find_paths_between_nodes")
class FindPathsBetweenNodes(BaseTool):
    """
    在图中抽取两个节点之间的无向路径，并以自然语言格式返回。
    - 节点展示: name, id, labels, description
    - 关系展示: relation_name/predicate(type), description   # ← 已去掉 confidence
    """
    name = "find_paths_between_nodes"
    description = "在图中抽取两个节点之间的无向路径（证据链），返回自然语言描述。"
    parameters = [
        {"name": "src_id", "type": "string", "description": "起点节点的 id", "required": True},
        {"name": "dst_id", "type": "string", "description": "终点节点的 id", "required": True},
        {"name": "max_depth", "type": "integer", "description": "路径最大边数（默认 4）", "required": False},
        {"name": "limit", "type": "integer", "description": "返回路径条数上限（默认 5）", "required": False},
    ]

    def __init__(self, graph_query_utils):
        self.graph_query_utils = graph_query_utils

    def _shorten(self, text: str, max_len: int = 120) -> str:
        if not text:
            return ""
        text = text.replace("\n", " ")
        return text if len(text) <= max_len else text[:max_len] + "…"

    def _format_node(self, node: Dict[str, Any]) -> str:
        name = node.get("name") or "(未命名)"
        eid = node.get("id") or "N/A"
        labels = node.get("labels", [])
        if "Entity" in labels:
            labels.remove("Entity")
        labels = ",".join(labels)
        desc = self._shorten(node.get("description", ""))
        return f"**{name}** (id={eid}, labels=[{labels}]) — {desc}"

    # ↓↓↓ 只改这个方法：不再显示 confidence ↓↓↓
    def _format_rel(self, rel: Dict[str, Any]) -> str:
        rname = rel.get("relation_name") or rel.get("predicate") or rel.get("type") or "RELATED"
        desc = (rel.get("properties") or {}).get("description") or ""
        desc_txt = f" — {self._shorten(desc)}" if desc else ""
        return f"── {rname}{desc_txt} ──>"
    # ↑↑↑

    def _render_path(self, path: Dict[str, Any]) -> str:
        nodes = path.get("nodes", [])
        rels = path.get("relationships", [])
        parts = []
        for i, node in enumerate(nodes):
            parts.append(self._format_node(node))
            if i < len(rels):
                parts.append(self._format_rel(rels[i]))
        return "\n".join(parts)

    def call(self, params: Any, **kwargs) -> str:
        logger.info("🔎 调用 find_paths_between_nodes")
        try:
            data: Dict[str, Any] = json.loads(params) if isinstance(params, str) else dict(params or {})
        except Exception as e:
            return f"参数解析失败: {e}"

        src_id = data.get("src_id")
        dst_id = data.get("dst_id")
        if not src_id or not dst_id:
            return "❌ 必须提供 src_id 和 dst_id"

        max_depth = int(data.get("max_depth", 4))
        limit = int(data.get("limit", 5))

        try:
            paths = self.graph_query_utils.find_paths_between_nodes(
                src_id=src_id,
                dst_id=dst_id,
                max_depth=max_depth,
                limit=limit
            )
            if not paths:
                return f"⚠️ 在 {max_depth} 跳内，没有找到 {src_id} 与 {dst_id} 之间的路径。"

            lines = [f"找到 {len(paths)} 条路径："]
            for i, p in enumerate(paths, 1):
                lines.append(f"\n**路径 {i} (长度={p['length']})**\n{self._render_path(p)}")
            return "\n".join(lines)
        except Exception as e:
            logger.exception("find_paths_between_nodes 执行失败")
            return f"执行失败: {str(e)}"


@register_tool("top_k_by_centrality")
class TopKByCentrality(BaseTool):
    name = "top_k_by_centrality"
    description = (
        "按中心度指标返回 Top-K 节点排名（已写回到节点属性的中心度）。"
        "支持的指标：pagerank/pr、degree/deg、betweenness/btw。"
        "可选按节点标签过滤（如 ['Plot','Event']）。"
    )
    parameters = [
        {
            "name": "metric",
            "type": "string",
            "description": "中心度指标：pagerank、degree、betweenness三选一。",
            "required": True,
        },
        {
            "name": "top_k",
            "type": "number",
            "description": "返回数量，默认 50；<=0 表示不限制（大图不建议）。",
            "required": False,
        },
        {
            "name": "node_labels",
            "type": "array",
            "description": "可选的节点标签过滤（如 ['Episode', 'Scene']）；不传表示全图。",
            "required": False,
        },
    ]

    def __init__(self, graph_query_utils):
        self.graph_query_utils = graph_query_utils  # 依赖 graph_query_utils.top_k_by_centrality()

    def call(self, params: str, **kwargs) -> str:
        logger.info("🔎 top_k_by_centrality")
        data = json.loads(params) if isinstance(params, str) else dict(params or {})
        metric_in = (data.get("metric") or "").strip().lower()
        metric_map = {
            "pagerank": "pagerank", "pr": "pagerank",
            "degree": "degree", "deg": "degree",
            "betweenness": "betweenness", "btw": "betweenness",
        }
        if metric_in not in metric_map:
            raise ValueError("metric 仅支持：pagerank/pr、degree/deg、betweenness/btw（不支持 closeness）")

        metric = metric_map[metric_in]
        top_k_raw = data.get("top_k", 50)
        top_k = int(top_k_raw) if isinstance(top_k_raw, (int, float, str)) and str(top_k_raw).lstrip("-").isdigit() else 50
        node_labels = data.get("node_labels")
        if isinstance(node_labels, str):
            node_labels = [s.strip() for s in node_labels.split(",") if s.strip()]
        elif node_labels is not None and not isinstance(node_labels, list):
            node_labels = [node_labels]

        # 调用底层工具方法，直接读取本地图中的中心性结果
        rows: List[Dict[str, Any]] = self.graph_query_utils.top_k_by_centrality(
            metric=metric,
            top_k=top_k,
            node_labels=node_labels,
        )

        if not rows:
            scope = f"{node_labels}" if node_labels else "全图"
            return f"{scope} 未发现含有该中心度属性的节点（请先运行中心度写回过程）。"

        # 格式化输出
        header = f"Top-{top_k if top_k and top_k > 0 else 'ALL'} by {metric.upper()}" + (f" @labels={node_labels}" if node_labels else "")
        lines = [header + ":"]
        for i, r in enumerate(rows, 1):
            name = r.get("name") or "(无名)"
            nid = r.get("id") or ""
            labs = r.get("labels") or []
            if "Entity" in labs:
                labs.remove("Entity")
            score = r.get("score")
            labs_txt = "/".join(labs) if labs else "Unknown"
            score_txt = f"{score:.6f}" if isinstance(score, (int, float)) else str(score)
            lines.append(f"{i:>2}. {name}  [ID: {nid}]  <{labs_txt}>  {metric}={score_txt}")
        return "\n".join(lines)

@register_tool("get_co_section_entities")
class GetCoSectionEntities(BaseTool):
    name = "get_co_section_entities"
    description = "输入实体 id，按章节/场次分组返回与其共同出现的实体列表。"
    parameters = [
        {"name": "entity_id", "type": "string", "description": "实体ID", "required": True},
        {"name": "include_types", "type": "array", "description": "可选的实体类型过滤，如 ['Event','Character']", "required": False},
        {"name": "document_ids", "type": "array", "description": "可选的 document_id 列表；只返回这些章节/场次中的共同出现实体。", "required": False},
        {"name": "max_sections", "type": "number", "description": "可选；最多返回多少个章节/场次分组。", "required": False},
        {"name": "max_entities_per_section", "type": "number", "description": "可选；每个章节/场次最多返回多少个实体。", "required": False},
    ]

    def __init__(self, graph_query_utils):
        self.graph_query_utils = graph_query_utils
        dt = str(getattr(graph_query_utils, "doc_type", "") or "").strip().lower()
        self.doc_type = dt if dt in DOC_TYPE_META else "general"
        self.section_label = str(DOC_TYPE_META.get(self.doc_type, DOC_TYPE_META["general"]).get("section_label", "Document"))

    def call(self, params: str, **kwargs) -> str:
        logger.info("🔎 调用 get_co_section_entities")
        data = json.loads(params) if isinstance(params, str) else dict(params or {})
        entity_id = data.get("entity_id")
        if not entity_id:
            raise ValueError("缺少必要参数：entity_id")

        include_types = _as_list(data.get("include_types"))
        document_ids = _as_list(data.get("document_ids"))
        max_sections_raw = data.get("max_sections")
        max_entities_raw = data.get("max_entities_per_section")
        max_sections = (
            max(1, int(max_sections_raw))
            if isinstance(max_sections_raw, (int, float, str)) and str(max_sections_raw).strip()
            else None
        )
        max_entities_per_section = (
            max(1, int(max_entities_raw))
            if isinstance(max_entities_raw, (int, float, str)) and str(max_entities_raw).strip()
            else None
        )
        grouped = self.graph_query_utils.find_co_section_entities_grouped(
            entity_id=entity_id,
            include_types=include_types,
            document_ids=document_ids,
            max_sections=max_sections,
            max_entities_per_section=max_entities_per_section,
        )

        if not grouped:
            scope = f" (type filter: {include_types})" if include_types else ""
            return f"No co-occurring entities found in the same {self.section_label.lower()}{scope}."

        source_entity = self.graph_query_utils.get_entity_by_id(entity_id)
        source_name = getattr(source_entity, "name", "") or entity_id

        lines = [f"Co-occurring {self.section_label.lower()} entities for `{source_name}` ({entity_id}):"]
        for idx, item in enumerate(grouped, 1):
            document_ids = [str(x).strip() for x in (item.get("document_ids") or []) if str(x).strip()]
            section_name = str(item.get("section_name") or "").strip() or (document_ids[0] if document_ids else "") or f"{self.section_label} {idx}"
            entities = item.get("entities") or []
            names: List[str] = []
            seen = set()
            for ent in entities:
                name = getattr(ent, "name", "") or getattr(ent, "id", "")
                etype = getattr(ent, "type", []) or []
                if isinstance(etype, list):
                    etype_text = ",".join(str(x) for x in etype if str(x).strip() and str(x) != "Entity")
                else:
                    etype_text = str(etype).strip()
                display = f"{name} [{etype_text}]" if etype_text else str(name)
                if display and display not in seen:
                    seen.add(display)
                    names.append(display)
            lines.append(f"{idx}. {self.section_label}: {section_name}")
            if document_ids:
                lines.append(f"   document_ids: {', '.join(document_ids)}")
            if names:
                lines.append(f"   entities: {', '.join(names)}")
            else:
                lines.append("   entities: （无）")
        return "\n".join(lines)


@register_tool("search_episodes")
class SearchEpisodes(BaseTool):
    name = "search_episodes"
    description = (
        "检索 Episode（不依赖ID）。默认 hybrid：关键词Top2主召回，向量Top3补充。"
    )
    parameters = [
        {"name": "query", "type": "string", "description": "检索文本（关键词或语义查询）。", "required": True},
        {"name": "top_k", "type": "number", "description": "最终返回条数，默认 5。", "required": False},
        {"name": "vector_top_k", "type": "number", "description": "向量候选数量，默认 3。", "required": False},
        {"name": "keyword_top_k", "type": "number", "description": "关键词候选数量，默认 2。", "required": False},
        {"name": "use_llm_filter", "type": "bool", "description": "是否对召回候选做 LLM 相关性过滤。", "required": False},
        {"name": "llm_filter_top_k", "type": "number", "description": "进入 LLM 判断的候选数量，默认 8。", "required": False},
        {"name": "llm_filter_threshold", "type": "number", "description": "LLM 相关性保留阈值，默认 0.35。", "required": False},
        {"name": "include_meta", "type": "bool", "description": "是否输出描述等详细信息，默认 False。", "required": False},
    ]

    def __init__(self, graph_query_utils, embedding_config=None, document_parser=None, max_workers: int = 4):
        self.graph_query_utils = graph_query_utils
        self.document_parser = document_parser
        self.max_workers = max(1, int(max_workers or 4))
        if embedding_config:
            self.graph_query_utils.load_embedding_model(embedding_config)

    def call(self, params: str, **kwargs) -> str:
        logger.info("🔎 调用 search_episodes")
        data = json.loads(params) if isinstance(params, str) else dict(params or {})
        query = (data.get("query") or "").strip()
        if not query:
            return "query 不能为空。"
        top_k = int(data.get("top_k", 5) or 5)
        vector_top_k = int(data.get("vector_top_k", 3) or 3)
        keyword_top_k = int(data.get("keyword_top_k", 2) or 2)
        use_llm_filter = _to_bool(data.get("use_llm_filter"), False)
        llm_filter_top_k = int(data.get("llm_filter_top_k", max(8, top_k)) or max(8, top_k))
        llm_filter_threshold = float(data.get("llm_filter_threshold", 0.35) or 0.35)
        include_meta = _to_bool(data.get("include_meta"), False)

        rows = search_episode_storyline_candidates(
            self.graph_query_utils,
            label="Episode",
            query=query,
            vector_top_k=vector_top_k,
            keyword_top_k=keyword_top_k,
            top_k=top_k,
            use_llm_filter=use_llm_filter,
            llm_filter_top_k=llm_filter_top_k,
            llm_filter_threshold=llm_filter_threshold,
            document_parser=self.document_parser,
            max_workers=self.max_workers,
        )
        return _format_episode_storyline_hits(rows, title="Episode", include_meta=include_meta)


@register_tool("search_storylines")
class SearchStorylines(BaseTool):
    name = "search_storylines"
    description = (
        "检索 Storyline（不依赖ID）。默认 hybrid：关键词Top2主召回，向量Top3补充。"
    )
    parameters = [
        {"name": "query", "type": "string", "description": "检索文本（关键词或语义查询）。", "required": True},
        {"name": "top_k", "type": "number", "description": "最终返回条数，默认 5。", "required": False},
        {"name": "vector_top_k", "type": "number", "description": "向量候选数量，默认 3。", "required": False},
        {"name": "keyword_top_k", "type": "number", "description": "关键词候选数量，默认 2。", "required": False},
        {"name": "use_llm_filter", "type": "bool", "description": "是否对召回候选做 LLM 相关性过滤。", "required": False},
        {"name": "llm_filter_top_k", "type": "number", "description": "进入 LLM 判断的候选数量，默认 8。", "required": False},
        {"name": "llm_filter_threshold", "type": "number", "description": "LLM 相关性保留阈值，默认 0.35。", "required": False},
        {"name": "include_meta", "type": "bool", "description": "是否输出描述等详细信息，默认 False。", "required": False},
    ]

    def __init__(self, graph_query_utils, embedding_config=None, document_parser=None, max_workers: int = 4):
        self.graph_query_utils = graph_query_utils
        self.document_parser = document_parser
        self.max_workers = max(1, int(max_workers or 4))
        if embedding_config:
            self.graph_query_utils.load_embedding_model(embedding_config)

    def call(self, params: str, **kwargs) -> str:
        logger.info("🔎 调用 search_storylines")
        data = json.loads(params) if isinstance(params, str) else dict(params or {})
        query = (data.get("query") or "").strip()
        if not query:
            return "query 不能为空。"
        top_k = int(data.get("top_k", 5) or 5)
        vector_top_k = int(data.get("vector_top_k", 3) or 3)
        keyword_top_k = int(data.get("keyword_top_k", 2) or 2)
        use_llm_filter = _to_bool(data.get("use_llm_filter"), False)
        llm_filter_top_k = int(data.get("llm_filter_top_k", max(8, top_k)) or max(8, top_k))
        llm_filter_threshold = float(data.get("llm_filter_threshold", 0.35) or 0.35)
        include_meta = _to_bool(data.get("include_meta"), False)

        rows = search_episode_storyline_candidates(
            self.graph_query_utils,
            label="Storyline",
            query=query,
            vector_top_k=vector_top_k,
            keyword_top_k=keyword_top_k,
            top_k=top_k,
            use_llm_filter=use_llm_filter,
            llm_filter_top_k=llm_filter_top_k,
            llm_filter_threshold=llm_filter_threshold,
            document_parser=self.document_parser,
            max_workers=self.max_workers,
        )
        return _format_episode_storyline_hits(rows, title="Storyline", include_meta=include_meta)


@register_tool("search_sections")
class SearchSections(BaseTool):
    name = "search_sections"
    description = (
        "检索文档分节超节点（按 doc_type 自动映射为 Scene/Chapter/Document）。"
        "默认 hybrid：关键词Top2主召回，向量Top3补充。"
    )
    parameters = [
        {"name": "query", "type": "string", "description": "检索文本（关键词或语义查询）。", "required": True},
        {"name": "top_k", "type": "number", "description": "最终返回条数，默认 5。", "required": False},
        {"name": "vector_top_k", "type": "number", "description": "向量候选数量，默认 3。", "required": False},
        {"name": "keyword_top_k", "type": "number", "description": "关键词候选数量，默认 2。", "required": False},
        {"name": "use_llm_filter", "type": "bool", "description": "是否对召回候选做 LLM 相关性过滤。", "required": False},
        {"name": "llm_filter_top_k", "type": "number", "description": "进入 LLM 判断的候选数量，默认 8。", "required": False},
        {"name": "llm_filter_threshold", "type": "number", "description": "LLM 相关性保留阈值，默认 0.35。", "required": False},
        {"name": "include_related_entities", "type": "bool", "description": "是否展示该章节关联的实体，默认 True。", "required": False},
        {"name": "related_entity_limit", "type": "number", "description": "每种实体类型最多展示多少个关联实体，默认 3。", "required": False},
        {"name": "include_meta", "type": "bool", "description": "是否输出 properties 等详细信息，默认 False。", "required": False},
    ]

    def __init__(self, graph_query_utils, embedding_config=None, doc_type: str = "general", document_parser=None, max_workers: int = 4):
        self.graph_query_utils = graph_query_utils
        self.document_parser = document_parser
        self.max_workers = max(1, int(max_workers or 4))
        if embedding_config:
            self.graph_query_utils.load_embedding_model(embedding_config)
        dt = str(doc_type or "general").strip().lower() or "general"
        self.doc_type = dt if dt in DOC_TYPE_META else "general"
        self.section_label = str(DOC_TYPE_META.get(self.doc_type, DOC_TYPE_META["general"]).get("section_label", "Document"))

    def call(self, params: str, **kwargs) -> str:
        logger.info("🔎 调用 search_sections")
        data = json.loads(params) if isinstance(params, str) else dict(params or {})
        query = (data.get("query") or "").strip()
        if not query:
            return "query 不能为空。"
        top_k = int(data.get("top_k", 5) or 5)
        vector_top_k = int(data.get("vector_top_k", 3) or 3)
        keyword_top_k = int(data.get("keyword_top_k", 2) or 2)
        use_llm_filter = _to_bool(data.get("use_llm_filter"), False)
        llm_filter_top_k = int(data.get("llm_filter_top_k", max(8, top_k)) or max(8, top_k))
        llm_filter_threshold = float(data.get("llm_filter_threshold", 0.35) or 0.35)
        include_related_entities = _to_bool(data.get("include_related_entities"), True)
        related_entity_limit = int(data.get("related_entity_limit", 3) or 3)
        include_meta = _to_bool(data.get("include_meta"), False)

        rows = search_episode_storyline_candidates(
            self.graph_query_utils,
            label=self.section_label,
            query=query,
            vector_top_k=vector_top_k,
            keyword_top_k=keyword_top_k,
            top_k=top_k,
            use_llm_filter=use_llm_filter,
            llm_filter_top_k=llm_filter_top_k,
            llm_filter_threshold=llm_filter_threshold,
            document_parser=self.document_parser,
            max_workers=self.max_workers,
        )
        text = _format_episode_storyline_hits(rows, title=self.section_label, include_meta=include_meta)
        if not rows or not include_related_entities:
            return text

        lines = text.splitlines()
        enhanced: List[str] = []
        current_row = None
        row_index = 0
        for line in lines:
            enhanced.append(line)
            if line.startswith(tuple(f"{i}. " for i in range(1, 10))) or (". " in line and line.split(". ", 1)[0].isdigit()):
                try:
                    current_row = rows[row_index]
                    row_index += 1
                except Exception:
                    current_row = None
            if current_row is not None and line.startswith("   source_documents(document_id):"):
                extra_lines = _format_section_related_entities(
                    self.graph_query_utils,
                    str(current_row.get("id") or ""),
                    related_entity_limit=max(1, related_entity_limit),
                )
                enhanced.extend(extra_lines)
        return "\n".join(enhanced)


@register_tool("search_communities")
class SearchCommunities(BaseTool):
    name = "search_communities"
    description = "检索 Community report 向量索引，并回填图结构信息，适合按社区主题和高层语义簇定位候选证据。"
    parameters = [
        {"name": "query", "type": "string", "description": "检索文本（关键词或语义查询）。", "required": True},
        {"name": "top_k", "type": "number", "description": "最终返回条数，默认 5。", "required": False},
        {"name": "level", "type": "number", "description": "仅检索指定层级的 Community。", "required": False},
        {"name": "vector_top_k", "type": "number", "description": "向量候选数量，默认 3。", "required": False},
        {"name": "keyword_top_k", "type": "number", "description": "关键词候选数量，默认 2。", "required": False},
        {"name": "use_llm_filter", "type": "bool", "description": "是否对召回候选做 LLM 相关性过滤。", "required": False},
        {"name": "llm_filter_top_k", "type": "number", "description": "进入 LLM 判断的候选数量，默认 8。", "required": False},
        {"name": "llm_filter_threshold", "type": "number", "description": "LLM 相关性保留阈值，默认 0.35。", "required": False},
        {"name": "include_meta", "type": "bool", "description": "是否输出更多 community properties。", "required": False},
        {"name": "include_member_preview", "type": "bool", "description": "是否展示社区代表成员预览。", "required": False},
        {"name": "member_preview_limit", "type": "number", "description": "代表成员预览条数上限，默认 5。", "required": False},
    ]

    def __init__(self, graph_query_utils, embedding_config=None, summary_vector_store=None, document_parser=None, max_workers: int = 4):
        self.graph_query_utils = graph_query_utils
        self.summary_vector_store = summary_vector_store
        self.document_parser = document_parser
        self.max_workers = max(1, int(max_workers or 4))
        if embedding_config:
            self.graph_query_utils.load_embedding_model(embedding_config)

    def call(self, params: str, **kwargs) -> str:
        logger.info("🔎 调用 search_communities")
        data = json.loads(params) if isinstance(params, str) else dict(params or {})
        query = safe_str(data.get("query")).strip()
        if not query:
            return "query 不能为空。"
        top_k = int(data.get("top_k", 5) or 5)
        level_raw = data.get("level")
        level = int(level_raw) if isinstance(level_raw, (int, float, str)) and str(level_raw).strip() else None
        vector_top_k = int(data.get("vector_top_k", 3) or 3)
        keyword_top_k = int(data.get("keyword_top_k", 2) or 2)
        use_llm_filter = _to_bool(data.get("use_llm_filter"), False)
        llm_filter_top_k = int(data.get("llm_filter_top_k", max(8, top_k)) or max(8, top_k))
        llm_filter_threshold = float(data.get("llm_filter_threshold", 0.35) or 0.35)
        include_meta = _to_bool(data.get("include_meta"), False)
        include_member_preview = _to_bool(data.get("include_member_preview"), True)
        member_preview_limit = int(data.get("member_preview_limit", 5) or 5)

        rows = search_community_candidates(
            self.graph_query_utils,
            summary_vector_store=self.summary_vector_store,
            query=query,
            top_k=top_k,
            level=level,
            vector_top_k=vector_top_k,
            keyword_top_k=keyword_top_k,
            use_llm_filter=use_llm_filter,
            llm_filter_top_k=llm_filter_top_k,
            llm_filter_threshold=llm_filter_threshold,
            document_parser=self.document_parser,
            max_workers=self.max_workers,
        )
        return _format_community_hits(
            rows,
            include_meta=include_meta,
            include_member_preview=include_member_preview,
            member_preview_limit=member_preview_limit,
        )


class QuerySimilarEntities(BaseTool):
    """
    基于向量索引的语义检索工具：输入自然语言文本，返回最相似的实体节点。
    内部使用本地图中的实体 embedding 进行最近邻搜索，
    默认关闭 embedding 归一化（normalize=False），并在预处理时轻度清理中文标点。

    特点：
    - 支持 Top-K 控制；
    - 可按实体类型过滤（如 Character、Event 等），自动校验类型合法性；
    - 自动过滤低质量结果（score < min_score 默认阈值 0.0）；
    - 输出可选为紧凑列表或详细信息。
    """
    name = "query_similar_entities"
    description = "兼容性保留：根据输入文本进行语义相似度检索。优先使用 retrieve_entity_by_name 进行混合实体检索。"
    parameters = [
        {"name": "text", "type": "string", "required": True},
        {"name": "top_k", "type": "number", "required": False},
        {"name": "entity_types", "type": "array", "required": False},
        {"name": "include_meta", "type": "bool", "required": False},
    ]

    def __init__(self, graph_query_utils, embedding_config):
        self.graph_query_utils = graph_query_utils
        self.graph_query_utils.load_embedding_model(embedding_config)

        # 默认参数
        self._default_min_score = 0.0
        self._default_normalize = False
        self._default_strip = True

    # ---- 内部辅助 ----
    @staticmethod
    def _strip_zh_punct(text: str) -> str:
        if not isinstance(text, str):
            return text
        return text.replace("“", "").replace("”", "").replace("‘", "").replace("’", "") \
                   .replace("，", ",").replace("。", ".").replace("？", "?").replace("！", "!").strip()

    @staticmethod
    def _labels_match(row_labels, wanted_types: Optional[List[str]]) -> bool:
        if not wanted_types:
            return True
        if not row_labels:
            return False
        return bool(set(map(str, row_labels)) & set(map(str, wanted_types)))

    @staticmethod
    def _fmt_compact(rows: List[dict]) -> str:
        if not rows:
            return "No similar entities found."
        lines = ["Similar entities:"]
        for r in rows:
            name = r.get("name") or "(unnamed)"
            rid = r.get("id") or "UNKNOWN_ID"
            labels = r.get("labels") or []
            if "Entity" in labels:
                labels.remove("Entity")
            score = float(r.get("score", 0.0) or 0.0)
            lab = "/".join(map(str, labels)) if labels else "unknown"
            lines.append(f"- {name}  [ID: {rid}]  <{lab}>  score={score:.6f}")
        return "\n".join(lines)

    @staticmethod
    def _fmt_verbose(rows: List[dict]) -> str:
        if not rows:
            return "No similar entities found."
        out: List[str] = []
        for r in rows:
            if out:
                out.append("")
                out.append("---")
                out.append("")
            out.append(f"name: {r.get('name') or '(unnamed)'}")
            out.append(f"id: {r.get('id') or 'UNKNOWN_ID'}")
            if r.get("description"):
                out.append(f"description: {r.get('description')}")
            labels_raw = r.get("labels") or []
            labels = [str(x) for x in labels_raw if str(x) != "Entity"]  # 过滤掉 "Entity"
            if labels:
                out.append(f"type: {', '.join(labels)}")

            if r.get("score") is not None:
                out.append(f"score: {float(r['score']):.6f}")
        return "\n".join(out)

    def call(self, params: str, **kwargs) -> str:
        logger.info("🔎 调用 query_similar_entities")
        data = json.loads(params) if isinstance(params, str) else dict(params or {})

        text: str = data.get("text", "")
        if not text:
            return "text 不能为空。"

        top_k: int = int(data.get("top_k", 5) or 5)
        wanted_types: Optional[List[str]] = _as_list(data.get("entity_types"))
        include_meta: bool = _to_bool(data.get("include_meta"), False)

        # ---- 安全校验实体类型 ----
        if wanted_types:
            available_entity_types = self.graph_query_utils.list_entity_types()
            safe_types = []
            for t in wanted_types:
                if t not in available_entity_types:
                    logger.info(f"❗ 未找到实体类型 {t}，使用 Entity 代替")
                    safe_types.append("Entity")
                else:
                    safe_types.append(t)
            wanted_types = list(set(safe_types))  # 去重

        # 默认清洗中文符号
        if self._default_strip:
            text = self._strip_zh_punct(text)

        # 检索
        rows = self.graph_query_utils.query_similar_entities(
            text=text,
            top_k=top_k,
            normalize=self._default_normalize,
        ) or []

        # 阈值过滤 + 类型过滤
        filtered = [
            r for r in rows
            if r.get("score", 0.0) >= self._default_min_score
            and self._labels_match(r.get("labels"), wanted_types)
        ]
        filtered.sort(key=lambda x: x.get("score", 0.0), reverse=True)

        return self._fmt_verbose(filtered) if include_meta else self._fmt_compact(filtered)


@register_tool("retrieve_triple_facts")
class QuerySimilarFacts(BaseTool):
    """
    基于关系 embedding 的事实检索：
    - 仅检索有 embedding 的关系
    - 仅返回 description 非空的关系
    """
    name = "retrieve_triple_facts"
    description = "根据输入文本检索语义相关的事实三元组关系，仅返回包含 description 的关系。"
    parameters = [
        {"name": "text", "type": "string", "required": True},
        {"name": "top_k", "type": "number", "required": False},
        {"name": "min_score", "type": "number", "required": False},
    ]

    def __init__(self, graph_query_utils, embedding_config):
        self.graph_query_utils = graph_query_utils
        self.graph_query_utils.load_embedding_model(embedding_config)

    @staticmethod
    def _fmt(rows: List[Dict[str, Any]]) -> str:
        if not rows:
            return "未找到包含关系 embedding 且有 description 的相关事实。"
        lines: List[str] = ["相关事实关系："]
        for i, r in enumerate(rows, 1):
            sid = r.get("subject_id") or "UNKNOWN_SUBJECT"
            sname = r.get("subject_name") or sid
            oid = r.get("object_id") or "UNKNOWN_OBJECT"
            oname = r.get("object_name") or oid
            rel_name = (r.get("relation_name") or "").strip()
            pred = r.get("predicate") or "RELATED"
            edge_label = rel_name or pred
            rid = r.get("relation_id") or "UNKNOWN_REL"
            score = float(r.get("score", 0.0))
            desc = (r.get("description") or "").strip()
            if not desc:
                # hard guarantee: result must include description
                continue
            lines.append(f"{i}. ({sname})-[{edge_label}]->({oname}) [relation_id: {rid}] score={score:.6f}")
            lines.append(f"   description: {desc}")
            src_docs = r.get("source_documents") or []
            if src_docs:
                lines.append(f"   source_documents: {', '.join(str(x) for x in src_docs)}")
        return "\n".join(lines) if len(lines) > 1 else "未找到包含关系 embedding 且有 description 的相关事实。"

    def call(self, params: str, **kwargs) -> str:
        logger.info("🔎 调用 retrieve_triple_facts")
        data = json.loads(params) if isinstance(params, str) else dict(params or {})
        text = (data.get("text") or "").strip()
        if not text:
            return "text 不能为空。"
        top_k = int(data.get("top_k", 5) or 5)
        min_score = float(data.get("min_score", -1.0) or -1.0)
        rows = self.graph_query_utils.query_similar_relations(
            text=text,
            top_k=top_k,
            min_score=min_score,
            normalize=False,
        ) or []
        # enforce non-empty description one more time
        rows = [r for r in rows if (r.get("description") or "").strip()]
        return self._fmt(rows)


@register_tool("get_k_hop_subgraph")
class GetKHopSubgraph(BaseTool):
    """
    从一个或多个中心节点出发，抽取其 k-hop 邻居子图（简洁&稳健版）
    - 兼容 nodes/relationships 的 properties 为 JSON 字符串或 dict
    - 不展示 confidence
    """
    name = "get_k_hop_subgraph"
    description = (
        "输入一个或多个中心节点 ID，返回其 k-hop 邻居子图（包含节点与关系）。\n"
        "⚠️ 注意：k 不能太大，建议 1–3 跳，否则图会爆炸性增长。"
    )
    parameters = [
        {"name": "center_ids", "type": "array", "description": "中心节点 ID 列表", "required": True},
        {"name": "k", "type": "integer", "description": "邻居跳数，建议 1–3（默认 2）", "required": False},
        {"name": "limit_nodes", "type": "integer", "description": "返回的最大节点数上限（默认 200）", "required": False},
    ]

    def __init__(self, graph_query_utils):
        self.graph_query_utils = graph_query_utils

    # ------------ 小工具函数 ------------
    def _shorten(self, text: Any, max_len: int = 120) -> str:
        if not text:
            return ""
        s = str(text).replace("\n", " ")
        return s if len(s) <= max_len else s[:max_len] + "…"

    def _ensure_map(self, maybe_json: Any) -> Dict[str, Any]:
        """把 JSON 字符串安全转为 dict；否则给空 dict。"""
        if isinstance(maybe_json, dict):
            return maybe_json
        if isinstance(maybe_json, str):
            s = maybe_json.strip()
            if s.startswith("{") and s.endswith("}"):
                try:
                    return json.loads(s)
                except Exception:
                    return {}
        return {}

    def _fmt_node(self, n: Dict[str, Any]) -> str:
        name = n.get("name") or "(未命名)"
        nid = n.get("id") or "N/A"
        labels = n.get("labels", [])
        if "Entity" in labels:
            labels.remove("Entity")
        labels = ",".join(labels)
        desc = self._shorten(n.get("description", ""))
        return f"- **{name}** [ID: {nid}, Labels: {labels}] — {desc}"

    def _fmt_rel(self, r: Dict[str, Any], node_map: Dict[str, str]) -> Optional[str]:
        # 必要字段
        rtype = r.get("relation_name") or r.get("predicate") or r.get("type") or "RELATED"
        start_id = r.get("start") or r.get("start_id") or r.get("source") or r.get("from")
        end_id   = r.get("end")   or r.get("end_id")   or r.get("target") or r.get("to")
        if not (start_id and end_id):
            return None

        sname = node_map.get(str(start_id), str(start_id))
        tname = node_map.get(str(end_id), str(end_id))

        props = self._ensure_map(r.get("properties"))
        # 描述优先级：properties.description -> r.description
        desc = props.get("description") or r.get("description") or ""
        desc_txt = f" — {self._shorten(desc)}" if desc else ""
        return f"- {sname} ({start_id}) -[{rtype}]-> {tname} ({end_id}){desc_txt}"

    # ------------ 主逻辑 ------------
    def call(self, params: Any, **kwargs) -> str:
        logger.info("🔎 调用 get_k_hop_subgraph")
        try:
            data = json.loads(params) if isinstance(params, str) else dict(params or {})
        except Exception as e:
            return f"参数解析失败: {e}"

        center_ids = data.get("center_ids")
        if not center_ids:
            return "❌ 必须提供至少一个 center_id"
        if isinstance(center_ids, str):
            center_ids = [center_ids]

        k = int(data.get("k", 2))
        limit_nodes = int(data.get("limit_nodes", 200))

        try:
            subgraph = self.graph_query_utils.get_k_hop_subgraph(center_ids, k, limit_nodes) or {}
            nodes = subgraph.get("nodes") or []
            rels = subgraph.get("relationships") or []

            if not nodes:
                return f"⚠️ 在 {k}-hop 内未找到子图。"

            # id -> name
            node_map: Dict[str, str] = {}
            for n in nodes:
                if isinstance(n, dict):
                    nid = str(n.get("id") or "")
                    if nid:
                        node_map[nid] = n.get("name") or nid

            lines = [
                f"抽取到 {len(nodes)} 个节点和 {len(rels)} 条关系 (中心节点: {', '.join(map(str, center_ids))}，跳数={k})",
                "",
                "节点："
            ]
            for n in nodes:
                if isinstance(n, dict):
                    lines.append(self._fmt_node(n))

            rel_lines: List[str] = []
            for r in rels:
                if not isinstance(r, dict):
                    # 保险：如果整条关系也是 JSON 字符串（目前你这边是 dict），解一下
                    if isinstance(r, str) and r.strip().startswith("{") and r.strip().endswith("}"):
                        try:
                            r = json.loads(r)
                        except Exception:
                            continue
                    else:
                        continue
                line = self._fmt_rel(r, node_map)
                if line:
                    rel_lines.append(line)

            if rel_lines:
                lines.append("\n关系：")
                lines.extend(rel_lines)

            return "\n".join(lines)

        except Exception as e:
            logger.exception("get_k_hop_subgraph 执行失败")
            return f"执行失败: {str(e)}"


# Backward-compatible aliases for direct imports in tests/scripts.
GetRelationSummary = GetRelationsBetweenEntities

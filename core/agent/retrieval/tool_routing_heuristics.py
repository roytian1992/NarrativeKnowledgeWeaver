from __future__ import annotations

import re
from typing import Any, Dict, List


def tool_parameters_text(tool: Any) -> str:
    params = getattr(tool, "parameters", []) or []
    if not isinstance(params, list):
        return str(params)
    parts: List[str] = []
    for item in params:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        desc = str(item.get("description") or "").strip()
        if name and desc:
            parts.append(f"{name} {desc}")
        elif name:
            parts.append(name)
    return " ".join(parts)


def _contains_any(text: str, needles: List[str]) -> bool:
    lowered = str(text or "").lower()
    return any(needle in lowered for needle in needles)


def _question_stem(query: str) -> str:
    text = str(query or "").strip()
    if not text:
        return ""
    for marker in ["Choices:", "choices:"]:
        if marker in text:
            text = text.split(marker, 1)[0]
    match = re.search(r"\([A-H]\)", text)
    if match:
        text = text[: match.start()]
    return text.strip()


def _has_named_entity_anchor(query: str) -> bool:
    stem = _question_stem(query)
    tokens = re.findall(r"\b[A-Z][A-Za-z'’\-]*\b", stem)
    ignore = {
        "A",
        "An",
        "And",
        "How",
        "If",
        "In",
        "The",
        "What",
        "When",
        "Where",
        "Which",
        "Who",
        "Why",
    }
    anchored = [token for token in tokens if token not in ignore and len(token) > 1]
    return len(anchored) >= 1


def _is_relationship_query(query: str) -> bool:
    return _contains_any(
        query,
        [
            "relationship between",
            "relationship to",
            "what is x's relationship",
            "what is the relationship",
            "opinion about",
            "feel about",
            "feels about",
        ],
    )


def _is_exact_fact_query(query: str) -> bool:
    return _contains_any(
        query,
        [
            "where ",
            "when ",
            "how often",
            "what caused",
            "what did ",
            "what were ",
            "which word",
            "what kind of",
            "what is the name of",
            "how are ",
            "how were ",
            "what will happen",
            "what was an error",
        ],
    )


def _is_attribute_disambiguation_query(query: str) -> bool:
    return _contains_any(
        query,
        [
            "doesn't describe",
            "does not describe",
            "best describes",
            "best describe",
            "most likely:",
            "most likely ",
        ],
    ) and _has_named_entity_anchor(query)


def _is_subject_sensitive_reasoning_query(query: str) -> bool:
    if not _has_named_entity_anchor(query):
        return False
    return _contains_any(
        query,
        [
            "why ",
            "reason",
            "opinion",
            "attitude",
            "feel about",
            "feels about",
            "surprised that",
            "actually true",
        ],
    ) or _is_attribute_disambiguation_query(query)


def _is_broad_semantic_query(query: str) -> bool:
    if _is_relationship_query(query) or _is_exact_fact_query(query) or _is_subject_sensitive_reasoning_query(query):
        return False
    return _contains_any(
        query,
        [
            "infer",
            "inference",
            "imply",
            "implied",
            "attitude",
            "theme",
            "main message",
            "significance",
            "tone",
            "most likely",
            "unlikely",
            "conclude",
            "conclusion",
            "warning",
            "mission",
        ],
    )


def tool_stage(tool_name: str) -> str:
    name = str(tool_name or "").strip()
    core = {
        "vdb_search_hierdocs",
        "vdb_search_sentences",
        "bm25_search_docs",
        "section_evidence_search",
        "search_sections",
    }
    extended = {
        "retrieve_entity_by_name",
        "get_entity_sections",
        "vdb_get_docs_by_document_ids",
        "retrieve_entity_by_id",
        "search_related_entities",
        "get_relations_between_entities",
        "narrative_hierarchical_search",
        "entity_event_trace_search",
        "fact_timeline_resolution_search",
        "community_graphrag_search",
        "search_communities",
        "search_related_content",
        "vdb_search_docs",
        "lookup_titles_by_document_ids",
        "lookup_document_ids_by_title",
    }
    exploratory = {
        "get_common_neighbors",
        "find_paths_between_nodes",
        "get_k_hop_subgraph",
        "top_k_by_centrality",
        "get_co_section_entities",
        "retrieve_triple_facts",
    }
    internal_only = {"search_episodes", "search_storylines"}
    if name in internal_only:
        return "internal_only"
    if name in core:
        return "core"
    if name in extended:
        return "extended"
    if name in exploratory:
        return "exploratory"
    return "extended"


def heuristic_tool_boosts(query: str) -> Dict[str, float]:
    q = str(query or "").strip().lower()
    boosts: Dict[str, float] = {}

    def add(names: List[str], score: float) -> None:
        for name in names:
            boosts[name] = max(score, boosts.get(name, 0.0))

    if re.search(r"\bent_[a-z0-9]+\b", q):
        add(["retrieve_entity_by_id"], 2.5)
    if re.search(r"\bscene_[a-z0-9_\-]+\b", q) or "document_id" in q or "_part_" in q:
        add(["lookup_titles_by_document_ids", "vdb_get_docs_by_document_ids"], 2.0)

    if any(x in q for x in ["场次", "章节", "scene", "chapter", "片段", "证据", "原文", "哪一场", "哪几场"]):
        add(["search_sections", "section_evidence_search", "vdb_search_hierdocs", "lookup_titles_by_document_ids"], 2.6)
    if any(x in q for x in ["几岁", "年龄", "几年", "年份", "何时", "时间", "when", "age", "year", "timeline"]):
        add(["fact_timeline_resolution_search", "vdb_search_sentences", "section_evidence_search", "search_sections", "bm25_search_docs", "get_entity_sections"], 2.8)
    if any(x in q for x in ["全称", "简称", "缩写", "formal name", "full name", "型号", "机型", "代号"]):
        add(["bm25_search_docs", "section_evidence_search", "retrieve_entity_by_name"], 2.8)
    if _is_relationship_query(query):
        add(
            [
                "bm25_search_docs",
                "section_evidence_search",
                "search_sections",
                "vdb_search_sentences",
                "retrieve_entity_by_name",
                "get_entity_sections",
            ],
            3.0,
        )
        add(["vdb_get_docs_by_document_ids"], 1.8)
        add(["fact_timeline_resolution_search"], 1.6)
        add(["entity_event_trace_search"], 1.1)
    if _is_subject_sensitive_reasoning_query(query):
        add(
            [
                "bm25_search_docs",
                "section_evidence_search",
                "vdb_search_sentences",
                "search_sections",
                "vdb_get_docs_by_document_ids",
                "retrieve_entity_by_name",
            ],
            2.9,
        )
        add(["entity_event_trace_search"], 1.2)
    elif any(x in q for x in ["伤疤", "伤痕", "受伤", "创伤", "原因", "导致", "车祸", "事故", "cause"]):
        add(["vdb_search_sentences", "vdb_search_hierdocs", "bm25_search_docs", "section_evidence_search", "retrieve_entity_by_name"], 2.9)
    if any(x in q for x in ["经常", "总是", "常常", "动作", "互动", "frequent", "often"]):
        add(["vdb_search_hierdocs", "vdb_search_sentences", "bm25_search_docs", "section_evidence_search", "retrieve_entity_by_name", "get_entity_sections"], 2.7)
    if any(x in q for x in ["西装", "制服", "穿着", "着装", "外观", "appearance", "wearing"]):
        add(["section_evidence_search", "vdb_search_sentences", "bm25_search_docs", "retrieve_entity_by_name"], 2.6)
    if any(x in q for x in ["坏了", "损坏", "变化", "开始", "持续多久", "持续了多久", "state change", "duration"]):
        add(["vdb_search_hierdocs", "bm25_search_docs", "section_evidence_search", "retrieve_entity_by_name", "get_entity_sections"], 2.8)
    if _is_exact_fact_query(query):
        add(["bm25_search_docs", "section_evidence_search", "search_sections", "vdb_search_sentences"], 3.0)
        add(["fact_timeline_resolution_search"], 1.8)
    if any(x in q for x in ["how many", "how long", "how much", "difference between", "first and second", "years since", "times did", "what happened", "who is ", "role of "]):
        add(["fact_timeline_resolution_search", "section_evidence_search", "vdb_search_sentences", "bm25_search_docs"], 3.1)
    if any(x in q for x in ["warning", "unspoken", "dream", "likely true", "what happened", "happened to", "significant", "significance", "what does this mean", "implied", "implies"]):
        add(["entity_event_trace_search", "section_evidence_search", "vdb_search_sentences", "bm25_search_docs"], 3.0)
    if any(x in q for x in ["系列", "型号", "版本", "哪几个型号", "哪些型号", "model", "series"]):
        add(["bm25_search_docs", "retrieve_entity_by_name", "get_entity_sections", "lookup_titles_by_document_ids"], 2.7)
    if any(x in q for x in ["剧情", "主线", "episode", "storyline", "因果", "推进", "阶段"]):
        add(["narrative_hierarchical_search", "entity_event_trace_search", "vdb_search_hierdocs"], 2.2)
    if any(x in q for x in ["significance", "important for", "warning", "reveal", "actually true", "真正", "其实", "暗示", "意味着", "重要", "作用"]):
        add(["entity_event_trace_search", "narrative_hierarchical_search", "section_evidence_search"], 2.9)
    if any(x in q for x in ["社区", "community", "摘要"]):
        add(["search_communities", "community_graphrag_search"], 2.1)
    if any(x in q for x in ["对话", "台词", "dialogue", "说了什么"]):
        add(["search_dialogues", "vdb_search_sentences", "section_evidence_search", "bm25_search_docs"], 2.0)
    if any(x in q for x in ["互动", "interaction", "关系", "polarity", "冲突", "合作"]):
        add(["search_interactions", "search_related_entities", "get_relations_between_entities"], 1.8)
    if any(x in q for x in ["谁", "人物", "角色", "实体", "entity", "关系"]):
        add(["retrieve_entity_by_name", "search_related_entities"], 1.4)
    if _is_broad_semantic_query(query):
        add(["vdb_search_hierdocs", "vdb_search_sentences", "section_evidence_search"], 3.0)
    return boosts


def preferred_core_tools(*, query: str, query_pattern: Dict[str, Any]) -> List[str]:
    q = str(query or "").strip().lower()
    problem_type = str((query_pattern or {}).get("problem_type", "") or "").strip().lower()
    answer_shape = str((query_pattern or {}).get("answer_shape", "") or "").strip().lower()

    if any(x in q for x in ["系列", "型号", "机型", "版本", "代号", "model", "series"]):
        return [
            "bm25_search_docs",
            "vdb_search_hierdocs",
            "retrieve_entity_by_name",
            "get_entity_sections",
            "lookup_titles_by_document_ids",
            "section_evidence_search",
            "vdb_get_docs_by_document_ids",
        ]
    if any(x in q for x in ["经常", "总是", "常常", "动作", "互动", "frequent", "often"]):
        return [
            "vdb_search_hierdocs",
            "vdb_search_sentences",
            "bm25_search_docs",
            "section_evidence_search",
            "retrieve_entity_by_name",
            "get_entity_sections",
            "vdb_get_docs_by_document_ids",
            "search_sections",
        ]
    if _is_relationship_query(query):
        return [
            "bm25_search_docs",
            "section_evidence_search",
            "search_sections",
            "vdb_search_sentences",
            "vdb_get_docs_by_document_ids",
            "retrieve_entity_by_name",
            "get_entity_sections",
            "fact_timeline_resolution_search",
            "entity_event_trace_search",
            "vdb_search_hierdocs",
        ]
    if _is_exact_fact_query(query):
        return [
            "bm25_search_docs",
            "section_evidence_search",
            "search_sections",
            "vdb_search_sentences",
            "vdb_get_docs_by_document_ids",
            "retrieve_entity_by_name",
            "get_entity_sections",
            "fact_timeline_resolution_search",
            "vdb_search_hierdocs",
        ]
    if _is_subject_sensitive_reasoning_query(query):
        return [
            "bm25_search_docs",
            "section_evidence_search",
            "vdb_search_sentences",
            "search_sections",
            "vdb_get_docs_by_document_ids",
            "retrieve_entity_by_name",
            "get_entity_sections",
            "entity_event_trace_search",
            "vdb_search_hierdocs",
        ]
    if any(x in q for x in ["坏了", "损坏", "变化", "开始", "持续多久", "持续了多久", "state change", "duration"]):
        return [
            "fact_timeline_resolution_search",
            "vdb_search_hierdocs",
            "bm25_search_docs",
            "section_evidence_search",
            "retrieve_entity_by_name",
            "get_entity_sections",
            "vdb_get_docs_by_document_ids",
            "search_sections",
        ]
    if problem_type == "causal_or_explanatory_lookup" or any(
        x in q for x in ["伤疤", "伤痕", "受伤", "创伤", "原因", "导致", "车祸", "事故", "cause"]
    ):
        return [
            "bm25_search_docs",
            "section_evidence_search",
            "vdb_search_sentences",
            "search_sections",
            "retrieve_entity_by_name",
            "vdb_get_docs_by_document_ids",
            "get_entity_sections",
            "fact_timeline_resolution_search",
            "entity_event_trace_search",
            "vdb_search_hierdocs",
        ]
    if _is_broad_semantic_query(query):
        return [
            "bm25_search_docs",
            "section_evidence_search",
            "vdb_search_sentences",
            "search_sections",
            "vdb_get_docs_by_document_ids",
            "vdb_search_hierdocs",
            "fact_timeline_resolution_search",
            "entity_event_trace_search",
        ]
    if problem_type in {"content_span_lookup", "section_localization"} or any(
        x in q for x in ["西装", "制服", "穿着", "着装", "外观", "appearance", "wearing"]
    ):
        return [
            "section_evidence_search",
            "search_sections",
            "vdb_search_sentences",
            "bm25_search_docs",
            "retrieve_entity_by_name",
            "get_entity_sections",
            "vdb_get_docs_by_document_ids",
        ]
    if answer_shape == "section_names" or any(x in q for x in ["场次", "场景", "章节", "scene", "chapter", "section"]):
        return [
            "search_sections",
            "section_evidence_search",
            "vdb_search_hierdocs",
            "lookup_titles_by_document_ids",
            "bm25_search_docs",
            "retrieve_entity_by_name",
            "get_entity_sections",
        ]
    return [
        "bm25_search_docs",
        "section_evidence_search",
        "vdb_search_sentences",
        "search_sections",
        "vdb_get_docs_by_document_ids",
        "retrieve_entity_by_name",
        "vdb_search_hierdocs",
        "fact_timeline_resolution_search",
    ]


def build_query_routing_hint(query: str) -> str:
    if _is_relationship_query(query) or _is_subject_sensitive_reasoning_query(query):
        return (
            "Routing Hint:\n"
            "- Start with `bm25_search_docs`, `section_evidence_search`, or `vdb_search_sentences`.\n"
            "- Prefer local textual evidence that directly distinguishes the answer choices.\n"
            "- Use `entity_event_trace_search` only if local evidence is still ambiguous after checking concrete passages."
        )
    if _is_exact_fact_query(query):
        return (
            "Routing Hint:\n"
            "- Start with `bm25_search_docs`, `section_evidence_search`, or `search_sections`.\n"
            "- Use `fact_timeline_resolution_search` only when the answer depends on chronology, ordering, or time comparison."
        )
    if _is_broad_semantic_query(query):
        return (
            "Routing Hint:\n"
            "- Start with `bm25_search_docs` plus `section_evidence_search` or `vdb_search_sentences`.\n"
            "- Prefer evidence from the most relevant local passages before relying on higher-level event traces."
        )
    return (
        "Routing Hint:\n"
        "- Start with `bm25_search_docs`, `section_evidence_search`, or `vdb_search_sentences`.\n"
        "- Escalate to higher-level tracing tools only when direct passage evidence is insufficient."
    )

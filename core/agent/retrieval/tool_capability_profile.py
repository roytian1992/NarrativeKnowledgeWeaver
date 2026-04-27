from __future__ import annotations

from typing import Any, Dict, Iterable, List


CAPABILITY_KEYWORDS: Dict[str, List[str]] = {
    "entity_grounding": [
        "entity lookup",
        "find or disambiguate an entity",
        "entity id",
        "entity-discovery",
        "entity_type",
    ],
    "entity_relation": [
        "relation",
        "related entities",
        "interaction",
        "graph-anchored",
        "common neighbors",
        "paths between",
    ],
    "passage_evidence": [
        "evidence search",
        "extract answerable evidence",
        "direct evidence extraction",
        "raw text chunks",
        "original text",
        "dialogue",
        "actions",
        "objects",
        "short factual",
    ],
    "sentence_retrieval": [
        "sentence-level",
        "specific sentence",
        "fine-grained evidence",
        "short fact",
        "local detail",
        "limit",
    ],
    "document_fetch": [
        "document_ids",
        "source documents",
        "promising document_ids",
        "fetch",
        "full text",
        "original text of those documents",
    ],
    "lexical_match": [
        "bm25",
        "exact wording",
        "surface-form",
        "quoted phrases",
        "verify literally",
        "lexical",
    ],
    "option_comparison": [
        "answer options",
        "competing options",
        "distinguishes the competing options",
        "choice",
        "multiple-choice",
        "compare options",
    ],
    "narrative_aggregation": [
        "storyline",
        "episode",
        "event chains",
        "plot progression",
        "narrative hierarchy",
        "narrative-structure",
        "causal development",
    ],
    "chronology_reasoning": [
        "timeline",
        "temporal",
        "chronology",
        "time travel",
        "time-based",
    ],
    "localization": [
        "section",
        "scene",
        "chapter",
        "document ids",
        "title",
        "locates relevant sections",
        "localization",
    ],
    "structural_ranking": [
        "deterministically rank",
        "centrality",
        "pagerank",
        "degree",
        "betweenness",
        "top-k ranking",
        "most central",
        "narratively key",
    ],
    "followup_refinement": [
        "follow-up",
        "after another tool",
        "promising document_ids",
        "refine",
        "already identified",
    ],
}

CAPABILITY_PRIORITY: List[str] = [
    "entity_grounding",
    "entity_relation",
    "option_comparison",
    "narrative_aggregation",
    "chronology_reasoning",
    "structural_ranking",
    "localization",
    "passage_evidence",
    "sentence_retrieval",
    "document_fetch",
    "lexical_match",
    "followup_refinement",
]

CAPABILITY_LABELS: Dict[str, str] = {
    "entity_grounding": "entity grounding",
    "entity_relation": "entity relation / interaction",
    "passage_evidence": "passage evidence extraction",
    "sentence_retrieval": "sentence-level retrieval",
    "document_fetch": "document fetch / follow-up reading",
    "lexical_match": "lexical matching",
    "option_comparison": "option comparison",
    "narrative_aggregation": "narrative aggregation",
    "chronology_reasoning": "chronology reasoning",
    "structural_ranking": "deterministic graph ranking",
    "localization": "section / document localization",
    "followup_refinement": "follow-up refinement",
}


def _normalize_text(parts: Iterable[Any]) -> str:
    chunks: List[str] = []
    for part in parts:
        text = str(part or "").strip().lower()
        if text:
            chunks.append(text)
    return "\n".join(chunks)


def infer_capabilities(
    *,
    tool_name: str,
    description: str,
    parameter_names: List[str],
) -> List[str]:
    text = _normalize_text([tool_name, description, " ".join(parameter_names or [])])
    out: List[str] = []
    for capability in CAPABILITY_PRIORITY:
        needles = CAPABILITY_KEYWORDS.get(capability) or []
        if any(needle in text for needle in needles):
            out.append(capability)

    # Minimal name-based fallback only when descriptions are sparse.
    lowered_name = str(tool_name or "").strip().lower()
    if "entity" in lowered_name and "entity_grounding" not in out:
        out.append("entity_grounding")
    if "interaction" in lowered_name and "entity_relation" not in out:
        out.append("entity_relation")
    if "section" in lowered_name and "localization" not in out:
        out.append("localization")
    if "narrative" in lowered_name and "narrative_aggregation" not in out:
        out.append("narrative_aggregation")
    if "timeline" in lowered_name and "chronology_reasoning" not in out:
        out.append("chronology_reasoning")
    if ("centrality" in lowered_name or "top_k" in lowered_name) and "structural_ranking" not in out:
        out.append("structural_ranking")
    if "sentence" in lowered_name and "sentence_retrieval" not in out:
        out.append("sentence_retrieval")
    if "bm25" in lowered_name and "lexical_match" not in out:
        out.append("lexical_match")
    if "choice" in lowered_name and "option_comparison" not in out:
        out.append("option_comparison")
    if "doc" in lowered_name and "document_fetch" not in out:
        out.append("document_fetch")
    if "search" in lowered_name and "passage_evidence" not in out and not out:
        out.append("passage_evidence")
    return out


def capability_labels(capabilities: Iterable[str]) -> List[str]:
    labels: List[str] = []
    for capability in capabilities or []:
        cap = str(capability or "").strip()
        if not cap:
            continue
        labels.append(CAPABILITY_LABELS.get(cap, cap))
    return labels


def family_from_capabilities(capabilities: List[str], *, fallback_family: str) -> str:
    caps = set(str(x or "").strip() for x in (capabilities or []))
    if {"entity_grounding", "entity_relation"} & caps:
        return "entity_relation"
    if {"narrative_aggregation", "chronology_reasoning"} & caps:
        return "narrative_reasoning"
    if {"structural_ranking"} & caps:
        return "structural_lookup"
    if {"passage_evidence", "sentence_retrieval", "document_fetch", "lexical_match", "option_comparison"} & caps:
        return "local_evidence"
    if {"localization", "followup_refinement"} & caps:
        return "structural_lookup"
    return str(fallback_family or "local_evidence").strip() or "local_evidence"

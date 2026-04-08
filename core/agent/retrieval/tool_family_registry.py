from __future__ import annotations

from typing import Dict, List

from core.agent.retrieval.tool_routing_heuristics import tool_stage
from core.agent.retrieval.tool_capability_profile import capability_labels


TOOL_FAMILY_ORDER: List[str] = [
    "local_evidence",
    "narrative_reasoning",
    "entity_relation",
    "structural_lookup",
    "exploratory",
]

TOOL_FAMILY_LABELS: Dict[str, str] = {
    "local_evidence": "Local textual evidence",
    "narrative_reasoning": "Narrative aggregation and event trace",
    "entity_relation": "Entity lookup and relation grounding",
    "structural_lookup": "Section and document lookup",
    "exploratory": "Graph exploration",
}

TOOL_FAMILY_CAPABILITY_CARDS: Dict[str, str] = {
    "local_evidence": (
        "Use for direct passage lookup, sentence evidence, document recall, and option grounding. "
        "Prefer this family for exact facts and option comparison."
    ),
    "narrative_reasoning": (
        "Use for storyline or event-level aggregation, causal narrative understanding, and temporal resolution. "
        "Prefer this family for motives, attitudes, dilemmas, implications, and chronology."
    ),
    "entity_relation": (
        "Use for named entity lookup, relation grounding, interaction tracing, and graph-anchored evidence."
    ),
    "structural_lookup": (
        "Use for section/document localization and resolving document ids or titles before deeper retrieval."
    ),
    "exploratory": (
        "Use only when direct evidence, narrative aggregation, and entity grounding are insufficient."
    ),
}

TOOL_FAMILY_MEMBERS: Dict[str, List[str]] = {
    "local_evidence": [
        "bm25_search_docs",
        "section_evidence_search",
        "vdb_search_sentences",
        "vdb_get_docs_by_document_ids",
        "vdb_search_hierdocs",
        "vdb_search_docs",
        "choice_grounded_evidence_search",
    ],
    "narrative_reasoning": [
        "narrative_hierarchical_search",
        "entity_event_trace_search",
        "fact_timeline_resolution_search",
    ],
    "entity_relation": [
        "retrieve_entity_by_name",
        "retrieve_entity_by_id",
        "get_entity_sections",
        "search_related_entities",
        "get_relations_between_entities",
        "search_interactions",
        "get_interactions_by_document_ids",
    ],
    "structural_lookup": [
        "search_sections",
        "lookup_titles_by_document_ids",
        "lookup_document_ids_by_title",
        "search_related_content",
    ],
    "exploratory": [
        "get_common_neighbors",
        "find_paths_between_nodes",
        "get_k_hop_subgraph",
        "top_k_by_centrality",
        "get_co_section_entities",
        "query_similar_facts",
        "search_communities",
        "community_graphrag_search",
    ],
}


def family_for_tool(tool_name: str) -> str:
    name = str(tool_name or "").strip()
    if not name:
        return "local_evidence"
    for family, members in TOOL_FAMILY_MEMBERS.items():
        if name in members:
            return family
    stage = tool_stage(name)
    if stage == "extended":
        return "entity_relation"
    if stage == "exploratory":
        return "exploratory"
    return "local_evidence"


def family_cards_text(cards: List[Dict[str, str]]) -> str:
    grouped: Dict[str, List[str]] = {}
    for card in cards:
        family = str(card.get("family", "") or "local_evidence").strip() or "local_evidence"
        desc = str(card.get("description", "") or "").strip()
        caps = capability_labels(card.get("capabilities") or [])
        cap_text = f" [capabilities: {', '.join(caps)}]" if caps else ""
        readiness = str(card.get("readiness", "") or "").strip()
        readiness_text = f" [readiness: {readiness}]" if readiness else ""
        line = f"- {card['name']}: {desc}{cap_text}{readiness_text}" if desc else f"- {card['name']}{cap_text}{readiness_text}"
        grouped.setdefault(family, []).append(line)

    blocks: List[str] = []
    for family in TOOL_FAMILY_ORDER:
        tools = grouped.get(family) or []
        if not tools:
            continue
        label = TOOL_FAMILY_LABELS.get(family, family)
        capability = TOOL_FAMILY_CAPABILITY_CARDS.get(family, "")
        header = f"[{family}] {label}"
        if capability:
            header = f"{header}\nCapability: {capability}"
        blocks.append(header + "\n" + "\n".join(tools))
    return "\n\n".join(blocks).strip()

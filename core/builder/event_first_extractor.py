from __future__ import annotations

import copy
import json
import time
from typing import Any, Dict, List, Tuple

from core.agent.knowledge_extraction_agent import (
    InformationExtractionAgent,
    apply_canonicalization_to_relations,
    build_name_canonicalizer,
    clean_screenplay_text,
)
from core.functions.regular_functions.entity_schema_grounding import EntitySchemaGrounder
from core.functions.regular_functions.event_occasion_frame_extraction import EventOccasionFrameExtractor
from core.functions.regular_functions.external_entity_candidates import ExternalEntityCandidateExtractor
from core.functions.regular_functions.open_entity_extraction import OpenEntityExtractor
from core.utils.prompt_loader import YAMLPromptLoader


def _norm_name(text: Any) -> str:
    return str(text or "").strip().lower()


def _dedup_open_entities(items: List[Dict[str, str]]) -> List[Dict[str, str]]:
    merged: Dict[str, Dict[str, str]] = {}
    for item in items or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        desc = str(item.get("description", "")).strip()
        if not name:
            continue
        key = name.lower()
        cur = merged.get(key)
        if cur is None or len(desc) > len(str(cur.get("description", ""))):
            merged[key] = {"name": name, "description": desc}
    return list(merged.values())


def _dedup_typed_entities(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for item in items or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        etype = str(item.get("type", "")).strip()
        desc = str(item.get("description", "")).strip()
        if not name or not etype:
            continue
        key = (name.lower(), etype)
        cur = merged.get(key)
        if cur is None or len(desc) > len(str(cur.get("description", ""))):
            merged[key] = {
                "name": name,
                "type": etype,
                "description": desc,
                "scope": str(item.get("scope", "")).strip() or "local",
            }
    return list(merged.values())


def _is_low_value_related_entity(name: str, description: str) -> bool:
    key = _norm_name(name)
    desc = _norm_name(description)
    weak_markers = [
        "消息",
        "news of",
        "report of",
        "rumor",
        "传闻",
        "流言",
    ]
    return any(marker in key or marker in desc for marker in weak_markers)


def _is_communicative_event(name: str, description: str) -> bool:
    text = f"{_norm_name(name)} {_norm_name(description)}"
    markers = [
        "tell",
        "telling",
        "inform",
        "discuss",
        "discussion",
        "talk",
        "talking",
        "read",
        "reading",
        "mention",
        "listing",
        "list",
        "conversation",
        "dialogue",
    ]
    return any(marker in text for marker in markers)


def _collect_frame_entity_candidates(
    frames: Dict[str, Any],
    *,
    frame_node_names: set[str],
) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    fields = [
        "agents",
        "experiencers",
        "patients",
        "participants",
        "initiators",
        "locations",
        "times",
        "related_entities",
    ]
    for group_key in ["events", "occasions"]:
        for frame in frames.get(group_key, []):
            for field in fields:
                for item in frame.get(field, []):
                    if not isinstance(item, dict) or not item.get("name"):
                        continue
                    name = str(item.get("name", "")).strip()
                    description = str(item.get("description", "")).strip()
                    if _norm_name(name) in frame_node_names:
                        continue
                    if field == "related_entities" and _is_low_value_related_entity(name, description):
                        continue
                    out.append({"name": name, "description": description})
    return _dedup_open_entities(out)


def _collect_direct_typed_frame_entities(
    frames: Dict[str, Any],
    *,
    frame_node_names: set[str],
    scope_rules: Dict[str, str],
) -> List[Dict[str, Any]]:
    role_to_type = {
        "agents": "Character",
        "experiencers": "Character",
        "participants": "Character",
        "initiators": "Character",
        "locations": "Location",
        "times": "TimePoint",
    }
    out: List[Dict[str, Any]] = []
    for group_key in ["events", "occasions"]:
        for frame in frames.get(group_key, []):
            for field, entity_type in role_to_type.items():
                for item in frame.get(field, []):
                    if not isinstance(item, dict) or not item.get("name"):
                        continue
                    name = str(item.get("name", "")).strip()
                    description = str(item.get("description", "")).strip()
                    if not name or _norm_name(name) in frame_node_names:
                        continue
                    out.append(
                        {
                            "name": name,
                            "description": description,
                            "type": entity_type,
                            "scope": scope_rules.get(entity_type, "local"),
                            "source_kind": "frame_slot",
                        }
                    )
    return _dedup_typed_entities(out)


def _collect_related_entity_candidates(
    frames: Dict[str, Any],
    *,
    frame_node_names: set[str],
) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for group_key in ["events", "occasions"]:
        for frame in frames.get(group_key, []):
            for item in frame.get("patients", []):
                if not isinstance(item, dict) or not item.get("name"):
                    continue
                name = str(item.get("name", "")).strip()
                description = str(item.get("description", "")).strip()
                if not name or _norm_name(name) in frame_node_names:
                    continue
                out.append({"name": name, "description": description})
            for item in frame.get("related_entities", []):
                if not isinstance(item, dict) or not item.get("name"):
                    continue
                name = str(item.get("name", "")).strip()
                description = str(item.get("description", "")).strip()
                if not name or _norm_name(name) in frame_node_names:
                    continue
                if _is_low_value_related_entity(name, description):
                    continue
                out.append({"name": name, "description": description})
    return _dedup_open_entities(out)


def _frame_nodes(frames: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    events = []
    for item in frames.get("events", []):
        events.append({"type": "Event", "name": item["name"], "description": item["description"], "scope": "local"})
    occasions = []
    for item in frames.get("occasions", []):
        occasions.append({"type": "Occasion", "name": item["name"], "description": item["description"], "scope": "global"})
    return events, occasions


def _known_entities_text(
    event_nodes: List[Dict[str, Any]],
    occasion_nodes: List[Dict[str, Any]],
    typed_frame_entities: List[Dict[str, Any]],
    open_candidates: List[Dict[str, str]],
) -> str:
    lines = []
    for item in event_nodes + occasion_nodes + typed_frame_entities:
        lines.append(item["name"])
    for item in open_candidates:
        lines.append(item["name"])
    return "\n".join(sorted(set(x for x in lines if x)))


def _dedup_grounded_entities(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    for item in items or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        etype = str(item.get("type", "")).strip()
        if not name or not etype:
            continue
        key = (name.lower(), etype)
        cur = merged.get(key)
        if cur is None or len(str(item.get("description", ""))) > len(str(cur.get("description", ""))):
            merged[key] = item
    return list(merged.values())


def _merge_entities(base: List[Dict[str, Any]], new_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for item in (base or []) + (new_items or []):
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        etype = str(item.get("type", "")).strip()
        if not name or not etype:
            continue
        key = (name.lower(), etype)
        cur = merged.get(key)
        if cur is None:
            merged[key] = dict(item)
            continue
        if len(str(item.get("description", ""))) > len(str(cur.get("description", ""))):
            merged[key] = dict(item)
    return list(merged.values())


def _entity_index(entities: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {str(e.get("name", "")).strip(): e for e in entities if isinstance(e, dict) and str(e.get("name", "")).strip()}


def _make_relation(rid: str, subject: str, object_: str, relation_type: str, relation_name: str, description: str) -> Dict[str, Any]:
    return {
        "rid": rid,
        "subject": subject,
        "object": object_,
        "relation_type": relation_type,
        "relation_name": relation_name,
        "description": description,
        "conf": 0.8,
    }


def _json_load_list(payload: str) -> List[Dict[str, Any]]:
    try:
        parsed = json.loads(payload) if payload else []
    except Exception:
        parsed = []
    if not isinstance(parsed, list):
        return []
    return [item for item in parsed if isinstance(item, dict)]


def _merge_open_entity_candidates(*groups: List[Dict[str, str]]) -> List[Dict[str, str]]:
    merged: List[Dict[str, str]] = []
    for items in groups:
        merged.extend(items or [])
    return _dedup_open_entities(merged)


def _collect_names(*groups: List[Dict[str, Any]]) -> set[str]:
    names: set[str] = set()
    for items in groups:
        for item in items or []:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "")).strip()
            if name:
                names.add(_norm_name(name))
    return names


def _collect_relation_brought_entity_candidates(
    proposals: List[Dict[str, Any]],
    *,
    known_entity_names: set[str],
) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for proposal in proposals or []:
        if not isinstance(proposal, dict):
            continue
        for item in proposal.get("new_entities", []) or []:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "")).strip()
            description = str(item.get("description", "")).strip()
            if not name or _norm_name(name) in known_entity_names:
                continue
            out.append({"name": name, "description": description})
    return _dedup_open_entities(out)


def _canonicalize_open_relation_proposals(
    proposals: List[Dict[str, Any]],
    *,
    entities: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    name_map = {
        _norm_name(item.get("name", "")): str(item.get("name", "")).strip()
        for item in entities or []
        if isinstance(item, dict) and str(item.get("name", "")).strip()
    }
    out: List[Dict[str, Any]] = []
    for proposal in proposals or []:
        if not isinstance(proposal, dict):
            continue
        subject = name_map.get(_norm_name(proposal.get("subject", "")), "")
        object_ = name_map.get(_norm_name(proposal.get("object", "")), "")
        relation_phrase = str(proposal.get("relation_phrase", "")).strip()
        description = str(proposal.get("description", "")).strip()
        evidence = str(proposal.get("evidence", "")).strip()
        if not subject or not object_ or not relation_phrase:
            continue
        item = dict(proposal)
        item["subject"] = subject
        item["object"] = object_
        if description:
            item["description"] = description
        if evidence:
            item["evidence"] = evidence
        out.append(item)
    return out


_HIGH_VALUE_RELATION_TYPES = {
    "performs",
    "undergoes",
    "experiences",
    "occurs_at",
    "occurs_during",
    "participates_in",
    "initiates",
    "hostility_with",
    "affinity_with",
    "kinship_with",
    "member_of",
    "part_of",
}

_WEAK_UTILITY_RELATION_TYPES = {"located_at", "possesses", "is_a", "occurs_on"}
_FAST_OPEN_GROUNDED_RELATION_TYPES = {"hostility_with", "affinity_with", "kinship_with", "member_of"}
_GENERIC_CHARACTER_PHRASES = {
    "a couple",
    "boy",
    "girl",
    "man",
    "woman",
    "young man",
    "young woman",
    "lady",
    "guest",
    "guests",
    "crowd",
    "reporters",
    "people",
}
_GENERIC_OBJECT_TERMS = {"book", "box", "bag", "coat", "cups", "cup", "cars", "car", "desk", "door", "echo", "cigarette"}
_GENERIC_TIME_TERMS = {"day", "night", "morning", "afternoon", "evening", "later", "now", "then", "soon"}


def _normalize_space(text: Any) -> str:
    return " ".join(str(text or "").strip().split())


def _has_non_ascii(text: str) -> bool:
    return any(ord(ch) > 127 for ch in text or "")


def _is_generic_character_name(name: str) -> bool:
    norm = _norm_name(name)
    if not norm:
        return True
    if norm in _GENERIC_CHARACTER_PHRASES:
        return True
    if any(marker in norm for marker in ["a lady", "a man", "a woman", "young man", "young woman", "in the group"]):
        return True
    if any(ch.isdigit() for ch in norm) and any(token in norm for token in ["men", "women", "男女", "人"]):
        return True
    return False


def _is_low_value_timepoint(name: str) -> bool:
    norm = _norm_name(name)
    if not norm:
        return True
    if norm in _GENERIC_TIME_TERMS:
        return True
    return any(marker in norm for marker in ["after ", "before ", "during ", "when ", "while ", "as ", "shortly ", "silence"])


def _is_descriptive_location_name(name: str) -> bool:
    raw = _normalize_space(name)
    norm = raw.lower()
    if not raw:
        return True
    if _has_non_ascii(raw):
        return any(marker in raw for marker in ["其中", "附近", "中间", "旁边", "入口", "门口"])
    if norm in {"room", "house", "office", "desk", "door", "hall"}:
        return True
    if any(marker in norm for marker in ["where ", "with ", "between ", "around ", "center of", "entrance of", "of the room", "connecting door"]):
        return True
    return norm.startswith(("a room", "the room", "a sunny room", "the area", "the place"))


def _is_low_value_object_name(name: str) -> bool:
    norm = _norm_name(name)
    if not norm:
        return True
    if norm in _GENERIC_OBJECT_TERMS:
        return True
    if norm.startswith(("a ", "an ", "the ")) and len(norm.split()) <= 3:
        return True
    return False


def _is_low_value_concept_name(name: str) -> bool:
    norm = _norm_name(name)
    if not norm:
        return True
    return any(marker in norm for marker in ["eye color", "face", "friends", "message", "news", "report", "speech"])


def _is_high_value_concept_name(name: str) -> bool:
    norm = _norm_name(name)
    return any(marker in norm for marker in ["family", "marriage", "divorce", "alimony", "play", "newspaper", "post", "committee", "harvard", "estate"])


def _entity_relation_type_map(relations: Dict[str, Dict[str, Any]]) -> Dict[str, set[str]]:
    out: Dict[str, set[str]] = {}
    for rel in (relations or {}).values():
        if not isinstance(rel, dict):
            continue
        rtype = str(rel.get("relation_type", "")).strip()
        subject = str(rel.get("subject", "")).strip()
        object_ = str(rel.get("object", "")).strip()
        if not rtype:
            continue
        if subject:
            out.setdefault(subject, set()).add(rtype)
        if object_:
            out.setdefault(object_, set()).add(rtype)
    return out


def _should_keep_entity_by_value(entity: Dict[str, Any], rel_types: set[str]) -> bool:
    etype = str(entity.get("type", "")).strip()
    name = str(entity.get("name", "")).strip()
    source_kind = str(entity.get("source_kind", "")).strip()
    if not etype or not name:
        return False
    if etype in {"Event", "Occasion"}:
        return True
    if source_kind == "frame_slot":
        return True

    has_high_value_relation = bool(rel_types.intersection(_HIGH_VALUE_RELATION_TYPES))
    only_weak_relations = bool(rel_types) and rel_types.issubset(_WEAK_UTILITY_RELATION_TYPES)

    if etype == "Character":
        if _is_generic_character_name(name):
            return has_high_value_relation
        return True
    if etype == "TimePoint":
        return has_high_value_relation and not _is_low_value_timepoint(name)
    if etype == "Location":
        if _is_descriptive_location_name(name):
            return bool(rel_types.intersection({"occurs_at", "part_of", "member_of"}))
        return True
    if etype == "Object":
        if _is_low_value_object_name(name):
            return bool(rel_types.intersection({"undergoes", "part_of", "member_of"}))
        return not only_weak_relations or has_high_value_relation
    if etype == "Concept":
        if _is_high_value_concept_name(name):
            return True
        if _is_low_value_concept_name(name):
            return bool(rel_types.intersection({"member_of", "kinship_with", "hostility_with", "affinity_with", "part_of"}))
        return has_high_value_relation or not only_weak_relations
    return True


def _prune_low_value_entities_and_relations(
    *,
    entities: List[Dict[str, Any]],
    relations: Dict[str, Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]], Dict[str, int]]:
    rel_type_map = _entity_relation_type_map(relations)
    kept_entities: List[Dict[str, Any]] = []
    dropped_names: set[str] = set()
    dropped_counts: Dict[str, int] = {}

    for entity in entities or []:
        if not isinstance(entity, dict):
            continue
        name = str(entity.get("name", "")).strip()
        if not name:
            continue
        rel_types = rel_type_map.get(name, set())
        if _should_keep_entity_by_value(entity, rel_types):
            kept_entities.append(entity)
            continue
        dropped_names.add(name)
        etype = str(entity.get("type", "")).strip() or "Unknown"
        dropped_counts[etype] = dropped_counts.get(etype, 0) + 1

    if not dropped_names:
        return kept_entities, relations, dropped_counts

    kept_relations: Dict[str, Dict[str, Any]] = {}
    for rid, rel in (relations or {}).items():
        if not isinstance(rel, dict):
            continue
        subject = str(rel.get("subject", "")).strip()
        object_ = str(rel.get("object", "")).strip()
        if subject in dropped_names or object_ in dropped_names:
            continue
        kept_relations[rid] = rel
    return kept_entities, kept_relations, dropped_counts


def _build_fast_open_relation_hints(agent: InformationExtractionAgent, *, entities: List[Dict[str, Any]]) -> str:
    lines: List[str] = [
        "[FAST PROFILE] Only extract social or affiliation relations that are not already built from Event/Occasion frames.",
        "[FAST PROFILE] Skip spatial, possession, temporal, and event-participant relations here; those should come from frame-native construction.",
    ]
    for rtype in ["hostility_with", "affinity_with", "kinship_with", "member_of"]:
        info = (agent.relation_type_info or {}).get(rtype)
        if not isinstance(info, dict):
            continue
        from_types = [str(x).strip() for x in (info.get("from") or []) if str(x).strip()]
        to_types = [str(x).strip() for x in (info.get("to") or []) if str(x).strip()]
        desc = str(info.get("description", "")).strip()
        direction = str(info.get("direction", "directed")).strip() or "directed"
        lines.append(
            f"{rtype}: {desc}        allowed_types: {','.join(from_types)} -> {','.join(to_types)}        direction: {direction}"
        )
    return "\n".join(lines)


def _has_fast_open_relation_candidates(
    *,
    agent: InformationExtractionAgent,
    entities: List[Dict[str, Any]],
) -> bool:
    names = [
        str(item.get("name", "")).strip()
        for item in entities or []
        if isinstance(item, dict) and str(item.get("name", "")).strip() and str(item.get("type", "")).strip()
    ]
    for i, subject in enumerate(names):
        for j, object_ in enumerate(names):
            if i == j:
                continue
            candidate_specs = agent._candidate_relation_specs_for_seed(
                subject_name=subject,
                object_name=object_,
                entities=entities,
            )
            if any(str(spec.get("relation_type", "")).strip() in _FAST_OPEN_GROUNDED_RELATION_TYPES for spec in candidate_specs):
                return True
    return False


def _filter_fast_open_relation_proposals(
    proposals: List[Dict[str, Any]],
    *,
    agent: InformationExtractionAgent,
    entities: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    filtered: List[Dict[str, Any]] = []
    for proposal in proposals or []:
        if not isinstance(proposal, dict):
            continue
        subject = str(proposal.get("subject", "")).strip()
        object_ = str(proposal.get("object", "")).strip()
        if not subject or not object_:
            continue
        candidate_specs = [
            spec for spec in agent._candidate_relation_specs_for_seed(
                subject_name=subject,
                object_name=object_,
                entities=entities,
            )
            if str(spec.get("relation_type", "")).strip() in _FAST_OPEN_GROUNDED_RELATION_TYPES
        ]
        if not candidate_specs:
            continue
        item = dict(proposal)
        item["candidate_relations"] = candidate_specs
        filtered.append(item)
    return filtered


def _should_run_fast_coverage_repair(
    *,
    frames: Dict[str, Any],
    grounded_entities: List[Dict[str, Any]],
    current_relations: Dict[str, Dict[str, Any]],
) -> bool:
    rel_types = {
        str(rel.get("relation_type", "")).strip()
        for rel in (current_relations or {}).values()
        if isinstance(rel, dict) and str(rel.get("relation_type", "")).strip()
    }
    type_counts: Dict[str, int] = {}
    for ent in grounded_entities or []:
        if not isinstance(ent, dict):
            continue
        etype = str(ent.get("type", "")).strip()
        if not etype:
            continue
        type_counts[etype] = type_counts.get(etype, 0) + 1

    social_missing = (
        int(type_counts.get("Character", 0)) >= 2
        and not any(r in rel_types for r in {"affinity_with", "hostility_with", "kinship_with", "member_of"})
    )

    return social_missing


def _auto_relations_from_frames(
    *,
    frames: Dict[str, Any],
    grounded_entities: List[Dict[str, Any]],
    event_nodes: List[Dict[str, Any]],
    occasion_nodes: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    idx = _entity_index(grounded_entities + event_nodes + occasion_nodes)
    rid_i = 1
    out: Dict[str, Dict[str, Any]] = {}
    seen_triples = set()

    def add_if_valid(subject: str, object_: str, relation_type: str, relation_name: str, description: str) -> None:
        nonlocal rid_i
        triple = (subject, relation_type, object_)
        if subject not in idx or object_ not in idx or subject == object_ or triple in seen_triples:
            return
        seen_triples.add(triple)
        key = f"auto#{rid_i}"
        rid_i += 1
        out[key] = _make_relation(key, subject, object_, relation_type, relation_name, description)

    for event in frames.get("events", []):
        ev_name = event["name"]
        is_communicative = _is_communicative_event(ev_name, str(event.get("description", "")))
        event_locations = []
        for item in event.get("agents", []):
            add_if_valid(item["name"], ev_name, "performs", "performs", f"{item['name']} performs event [{ev_name}].")
        for item in event.get("experiencers", []):
            ent = idx.get(item["name"], {})
            if ent.get("type") == "Character":
                add_if_valid(item["name"], ev_name, "experiences", "experiences", f"{item['name']} experiences event [{ev_name}].")
        if not is_communicative:
            for item in event.get("patients", []):
                ent = idx.get(item["name"], {})
                if ent.get("type") in {"Character", "Object", "Concept"}:
                    add_if_valid(item["name"], ev_name, "undergoes", "undergoes", f"{item['name']} undergoes event [{ev_name}].")
        for item in event.get("locations", []):
            ent = idx.get(item["name"], {})
            if ent.get("type") == "Location":
                event_locations.append(item["name"])
                add_if_valid(ev_name, item["name"], "occurs_at", "occurs at", f"Event [{ev_name}] occurs at [{item['name']}].")
        for item in event.get("times", []):
            ent = idx.get(item["name"], {})
            if ent.get("type") == "TimePoint":
                add_if_valid(ev_name, item["name"], "occurs_on", "occurs on", f"Event [{ev_name}] occurs on [{item['name']}].")
        for item in event.get("occasion", []):
            ent = idx.get(item["name"], {})
            if ent.get("type") == "Occasion":
                add_if_valid(ev_name, item["name"], "occurs_during", "occurs during", f"Event [{ev_name}] occurs during [{item['name']}].")
        if len(event_locations) == 1:
            event_location = event_locations[0]
            for field in ["agents", "experiencers"]:
                for item in event.get(field, []):
                    ent = idx.get(item["name"], {})
                    if ent.get("type") in {"Character", "Object"}:
                        add_if_valid(item["name"], event_location, "located_at", "located at", f"[{item['name']}] is located at [{event_location}] during the event [{ev_name}].")
            for item in event.get("related_entities", []):
                ent = idx.get(item["name"], {})
                if ent.get("type") == "Location":
                    add_if_valid(item["name"], event_location, "part_of", "part of", f"[{item['name']}] is a sub-location within [{event_location}].")
                elif ent.get("type") == "Object":
                    add_if_valid(item["name"], event_location, "located_at", "located at", f"[{item['name']}] is located at [{event_location}].")

    for occasion in frames.get("occasions", []):
        oc_name = occasion["name"]
        occasion_locations = []
        for item in occasion.get("participants", []):
            ent = idx.get(item["name"], {})
            if ent.get("type") in {"Character", "Concept"}:
                add_if_valid(item["name"], oc_name, "participates_in", "participates in", f"{item['name']} participates in occasion [{oc_name}].")
        for item in occasion.get("initiators", []):
            ent = idx.get(item["name"], {})
            if ent.get("type") in {"Character", "Concept"}:
                add_if_valid(item["name"], oc_name, "initiates", "initiates", f"{item['name']} initiates occasion [{oc_name}].")
        for item in occasion.get("locations", []):
            ent = idx.get(item["name"], {})
            if ent.get("type") == "Location":
                occasion_locations.append(item["name"])
                add_if_valid(oc_name, item["name"], "occurs_at", "occurs at", f"Occasion [{oc_name}] occurs at [{item['name']}].")
        for item in occasion.get("times", []):
            ent = idx.get(item["name"], {})
            if ent.get("type") == "TimePoint":
                add_if_valid(oc_name, item["name"], "occurs_on", "occurs on", f"Occasion [{oc_name}] occurs on [{item['name']}].")
        if len(occasion_locations) == 1:
            occasion_location = occasion_locations[0]
            add_if_valid(occasion_location, oc_name, "part_of", "part of", f"[{occasion_location}] is a location that forms part of occasion [{oc_name}].")
            for item in occasion.get("participants", []):
                ent = idx.get(item["name"], {})
                if ent.get("type") in {"Character", "Object"}:
                    add_if_valid(item["name"], occasion_location, "located_at", "located at", f"[{item['name']}] is located at [{occasion_location}] during occasion [{oc_name}].")
            for item in occasion.get("related_entities", []):
                ent = idx.get(item["name"], {})
                if ent.get("type") == "Location":
                    add_if_valid(item["name"], occasion_location, "part_of", "part of", f"[{item['name']}] is a sub-location within [{occasion_location}].")
                elif ent.get("type") == "Object":
                    add_if_valid(item["name"], occasion_location, "located_at", "located at", f"[{item['name']}] is located at [{occasion_location}] during occasion [{oc_name}].")

    return out


class EventFirstExtractor:
    def __init__(self, *, config: Any, llm: Any, agent: InformationExtractionAgent) -> None:
        self.config = config
        self.llm = llm
        self.agent = agent
        prompt_loader = YAMLPromptLoader(config.global_config.prompt_dir)
        self.frame_extractor = EventOccasionFrameExtractor(
            prompt_loader=prompt_loader,
            llm=llm,
            task_schema_path=str(config.global_config.task_dir) + "/event_occasion_frame_extraction_task.json",
            max_retries=1,
        )
        self.open_entity_extractor = OpenEntityExtractor(
            prompt_loader=prompt_loader,
            llm=llm,
            task_schema_path=str(config.global_config.task_dir) + "/open_entity_extraction_task.json",
            max_retries=1,
        )
        self.entity_grounder = EntitySchemaGrounder(
            prompt_loader=prompt_loader,
            llm=llm,
            task_schema_path=str(config.global_config.task_dir) + "/entity_schema_grounding_task.json",
            entity_schema_path=str(config.global_config.schema_dir) + "/default_entity_schema.json",
            max_retries=1,
        )
        self.external_entity_extractor = ExternalEntityCandidateExtractor(config, llm=llm)
        if hasattr(self.agent, "extractor") and self.agent.extractor is not None:
            if hasattr(self.agent.extractor, "open_relation_extraction"):
                self.agent.extractor.open_relation_extraction.max_retries = 1
                self.agent.extractor.open_relation_extraction.max_few_shot_word_len = 220
                self.agent.extractor.open_relation_extraction.max_entities_word_len = 180
            if hasattr(self.agent.extractor, "schema_relation_grounding"):
                self.agent.extractor.schema_relation_grounding.max_retries = 1
                self.agent.extractor.schema_relation_grounding.max_candidates_word_len = 260

    def run_document(
        self,
        *,
        ordered_chunks: List[Dict[str, Any]],
        aggressive_clean: bool = True,
        document_rid_namespace: str = "document0",
    ) -> Dict[str, Any]:
        entities_all: List[Dict[str, Any]] = []
        all_relations: Dict[str, Dict[str, Any]] = {}
        stats = {
            "processed_chunks": 0,
            "skipped_chunks": 0,
            "auto_relation_count": 0,
            "open_relation_proposal_count": 0,
            "open_relation_count_after_fix": 0,
            "initial_grounded_entity_count": 0,
            "relation_brought_entity_candidate_count": 0,
            "relation_brought_grounded_entity_count": 0,
            "resolved_entity_count": 0,
            "pruned_low_value_entity_count": 0,
            "coverage_repair_triggered_chunks": 0,
            "external_entity_raw_count": 0,
            "external_entity_typed_count": 0,
            "external_entity_open_candidate_count": 0,
            "open_entity_llm_skipped_chunks": 0,
            "frame_extraction_sec": 0.0,
            "external_entity_sec": 0.0,
            "open_entity_extraction_sec": 0.0,
            "entity_grounding_sec": 0.0,
            "auto_relation_build_sec": 0.0,
            "open_relation_extraction_sec": 0.0,
            "relation_brought_entity_grounding_sec": 0.0,
            "relation_grounding_sec": 0.0,
            "resolve_errors_sec": 0.0,
            "coverage_repair_sec": 0.0,
            "total_chunk_sec": 0.0,
        }

        chunks = list(ordered_chunks or [])
        try:
            chunks.sort(key=lambda c: (c.get("metadata", {}) or {}).get("order", 0))
        except Exception:
            pass

        for i, ch in enumerate(chunks):
            content = str(ch.get("content", "") or "").strip()
            cleaned = clean_screenplay_text(content, aggressive=aggressive_clean)
            if not cleaned:
                stats["skipped_chunks"] += 1
                continue

            out = self.run_chunk(cleaned_text=cleaned, rid_namespace=f"{document_rid_namespace}.chunk{i}")
            chunk_entities = out["entities"]
            chunk_relations = out["relations"]
            entities_all = _merge_entities(entities_all, chunk_entities)
            for rid, rel in chunk_relations.items():
                all_relations[rid] = rel
            stats["processed_chunks"] += 1
            for key, value in (out.get("stats") or {}).items():
                if key == "coverage_repair_triggered":
                    key = "coverage_repair_triggered_chunks"
                if key not in stats:
                    continue
                if isinstance(stats[key], float):
                    stats[key] += float(value or 0.0)
                else:
                    stats[key] += int(value or 0)

        return {"entities": entities_all, "relations": list(all_relations.values()), "stats": stats}

    def run_document_doc_grounded_fast(
        self,
        *,
        ordered_chunks: List[Dict[str, Any]],
        aggressive_clean: bool = True,
        document_rid_namespace: str = "document0",
    ) -> Dict[str, Any]:
        entities_all: List[Dict[str, Any]] = []
        all_relations: Dict[str, Dict[str, Any]] = {}
        stats = {
            "processed_chunks": 0,
            "skipped_chunks": 0,
            "auto_relation_count": 0,
            "open_relation_proposal_count": 0,
            "open_relation_count_after_fix": 0,
            "initial_grounded_entity_count": 0,
            "relation_brought_entity_candidate_count": 0,
            "relation_brought_grounded_entity_count": 0,
            "resolved_entity_count": 0,
            "pruned_low_value_entity_count": 0,
            "coverage_repair_triggered_chunks": 0,
            "external_entity_raw_count": 0,
            "external_entity_typed_count": 0,
            "external_entity_open_candidate_count": 0,
            "open_entity_llm_skipped_chunks": 0,
            "frame_extraction_sec": 0.0,
            "external_entity_sec": 0.0,
            "open_entity_extraction_sec": 0.0,
            "entity_grounding_sec": 0.0,
            "auto_relation_build_sec": 0.0,
            "open_relation_extraction_sec": 0.0,
            "relation_brought_entity_grounding_sec": 0.0,
            "relation_grounding_sec": 0.0,
            "resolve_errors_sec": 0.0,
            "coverage_repair_sec": 0.0,
            "total_chunk_sec": 0.0,
        }

        chunks = list(ordered_chunks or [])
        try:
            chunks.sort(key=lambda c: (c.get("metadata", {}) or {}).get("order", 0))
        except Exception:
            pass

        chunk_states: List[Dict[str, Any]] = []
        doc_cleaned_parts: List[str] = []
        doc_typed_entities: List[Dict[str, Any]] = []
        doc_open_candidates: List[Dict[str, str]] = []

        for i, ch in enumerate(chunks):
            content = str(ch.get("content", "") or "").strip()
            cleaned = clean_screenplay_text(content, aggressive=aggressive_clean)
            if not cleaned:
                stats["skipped_chunks"] += 1
                continue

            pass1 = self._collect_chunk_pass1(cleaned_text=cleaned)
            pass1["rid_namespace"] = f"{document_rid_namespace}.chunk{i}"
            chunk_states.append(pass1)
            doc_cleaned_parts.append(cleaned)
            doc_typed_entities.extend(pass1["typed_frame_entities"])
            doc_typed_entities.extend(pass1["external_typed_entities"])
            doc_open_candidates.extend(pass1["grounding_candidates"])
            stats["processed_chunks"] += 1
            for key, value in (pass1.get("stats") or {}).items():
                if key not in stats:
                    continue
                if isinstance(stats[key], float):
                    stats[key] += float(value or 0.0)
                else:
                    stats[key] += int(value or 0)

        document_text = "\n\n".join(doc_cleaned_parts).strip()
        grounded_open_entities: List[Dict[str, Any]] = []
        doc_grounding_candidates = _dedup_open_entities(doc_open_candidates)
        entity_grounding_start = time.perf_counter()
        if document_text and doc_grounding_candidates:
            grounded_open_entities = _json_load_list(
                self.entity_grounder.call(
                    json.dumps(
                        {
                            "text": document_text,
                            "open_entities": doc_grounding_candidates,
                            "memory_context": "",
                        }
                    )
                )
            )
        for item in grounded_open_entities:
            item["source_kind"] = "open_grounded"
        stats["entity_grounding_sec"] += time.perf_counter() - entity_grounding_start

        doc_grounded_entities = _dedup_grounded_entities(doc_typed_entities + grounded_open_entities)
        stats["initial_grounded_entity_count"] = len(doc_grounded_entities)

        relation_pass: List[Dict[str, Any]] = []
        relation_brought_candidates_all: List[Dict[str, str]] = []

        self.agent.config.knowledge_graph_builder.relation_extraction_mode = "open_then_ground"

        for chunk_state in chunk_states:
            chunk_entities = self._select_relevant_doc_entities(
                cleaned_text=chunk_state["cleaned_text"],
                doc_grounded_entities=doc_grounded_entities,
                chunk_state=chunk_state,
            )
            auto_relation_start = time.perf_counter()
            auto_relations = _auto_relations_from_frames(
                frames=chunk_state["frames"],
                grounded_entities=chunk_entities,
                event_nodes=chunk_state["event_nodes"],
                occasion_nodes=chunk_state["occasion_nodes"],
            )
            stats["auto_relation_build_sec"] += time.perf_counter() - auto_relation_start

            open_relation_extract_start = time.perf_counter()
            open_relation_proposals: List[Dict[str, Any]] = []
            if _has_fast_open_relation_candidates(agent=self.agent, entities=chunk_entities):
                relation_hints = _build_fast_open_relation_hints(self.agent, entities=chunk_entities)
                raw_open_relations = self.agent.extractor.extract_open_relations(
                    text=chunk_state["cleaned_text"],
                    extracted_entities=self.agent._entities_text_for_extractor(chunk_entities),
                    previous_results=None,
                    feedbacks=None,
                    memory_context="",
                    relation_hints=relation_hints,
                    focus_entities="",
                )
                open_relation_proposals = _json_load_list(raw_open_relations)
            stats["open_relation_extraction_sec"] += time.perf_counter() - open_relation_extract_start
            stats["open_relation_proposal_count"] += len(open_relation_proposals)

            relation_brought_candidates = _collect_relation_brought_entity_candidates(
                open_relation_proposals,
                known_entity_names={_norm_name(item.get("name", "")) for item in doc_grounded_entities},
            )
            relation_brought_candidates_all.extend(relation_brought_candidates)
            relation_pass.append(
                {
                    "chunk_state": chunk_state,
                    "chunk_entities": chunk_entities,
                    "auto_relations": auto_relations,
                    "open_relation_proposals": open_relation_proposals,
                }
            )

        relation_brought_candidates_all = _dedup_open_entities(relation_brought_candidates_all)
        stats["relation_brought_entity_candidate_count"] = len(relation_brought_candidates_all)

        relation_brought_grounding_start = time.perf_counter()
        relation_brought_grounded_entities: List[Dict[str, Any]] = []
        if document_text and relation_brought_candidates_all:
            relation_brought_grounded_entities = _json_load_list(
                self.entity_grounder.call(
                    json.dumps(
                        {
                            "text": document_text,
                            "open_entities": relation_brought_candidates_all,
                            "memory_context": "",
                        }
                    )
                )
            )
        for item in relation_brought_grounded_entities:
            item["source_kind"] = "relation_brought_grounded"
        stats["relation_brought_entity_grounding_sec"] += time.perf_counter() - relation_brought_grounding_start
        stats["relation_brought_grounded_entity_count"] = len(relation_brought_grounded_entities)

        final_doc_grounded_entities = _dedup_grounded_entities(doc_grounded_entities + relation_brought_grounded_entities)

        for item in relation_pass:
            chunk_state = item["chunk_state"]
            proposal_names = self._proposal_entity_names(item["open_relation_proposals"])
            chunk_entities = self._select_relevant_doc_entities(
                cleaned_text=chunk_state["cleaned_text"],
                doc_grounded_entities=final_doc_grounded_entities,
                chunk_state=chunk_state,
                extra_names=proposal_names,
            )

            relation_grounding_start = time.perf_counter()
            open_relation_proposals = _canonicalize_open_relation_proposals(
                item["open_relation_proposals"],
                entities=chunk_entities,
            )
            open_relation_proposals = _filter_fast_open_relation_proposals(
                open_relation_proposals,
                agent=self.agent,
                entities=chunk_entities,
            )
            seeded_relations = self.agent._prepare_grounded_open_relation_seeds(
                cleaned_text=chunk_state["cleaned_text"],
                proposals=copy.deepcopy(open_relation_proposals),
                entities=copy.deepcopy(chunk_entities),
                rid_prefix=f"{chunk_state['rid_namespace']}:event_first.open_rel",
                memory_context="",
            )
            seeded_relations = apply_canonicalization_to_relations(seeded_relations, build_name_canonicalizer(chunk_entities))
            seeded_relations = self.agent._filter_illegal_is_a_relations(seeded_relations, entities=chunk_entities)

            open_rel_map: Dict[str, Dict[str, Any]] = {}
            open_feedbacks: Dict[str, List[Dict[str, Any]]] = {}
            for rel in seeded_relations:
                rid = str(rel.get("rid", "")).strip()
                if not rid:
                    continue
                rel2, feedback = self.agent._revalidate_one_relation_global(rel, entities=chunk_entities)
                open_rel_map[rid] = rel2
                if feedback is not None:
                    self.agent._append_relation_feedback(
                        open_feedbacks,
                        str(feedback.get("error_type", "")).strip() or "unknown validation error",
                        rid=rid,
                        feedback=str(feedback.get("feedback", "")).strip(),
                        subject=str(feedback.get("subject", "")).strip() or str(rel2.get("subject", "")).strip(),
                        object=str(feedback.get("object", "")).strip() or str(rel2.get("object", "")).strip(),
                        relation_type=str(feedback.get("relation_type", "")).strip() or str(rel2.get("relation_type", "")).strip(),
                        relation_name=str(feedback.get("relation_name", "")).strip() or str(rel2.get("relation_name", "")).strip(),
                    )
            stats["relation_grounding_sec"] += time.perf_counter() - relation_grounding_start

            resolve_errors_start = time.perf_counter()
            resolved_entities, fixed_open_rel_map = self.agent._resolve_errors(
                entities=copy.deepcopy(chunk_entities),
                all_relations=copy.deepcopy(open_rel_map),
                all_feedbacks=copy.deepcopy(open_feedbacks),
                content=chunk_state["cleaned_text"],
            )
            stats["resolve_errors_sec"] += time.perf_counter() - resolve_errors_start
            resolved_entities = _dedup_grounded_entities(resolved_entities)

            coverage_repair_triggered = 0
            coverage_probe_rel_map = dict(item["auto_relations"])
            coverage_probe_rel_map.update(fixed_open_rel_map)
            if _should_run_fast_coverage_repair(
                frames=chunk_state["frames"],
                grounded_entities=resolved_entities,
                current_relations=coverage_probe_rel_map,
            ):
                coverage_repair_triggered = 1
            coverage_repair_start = time.perf_counter()
            if coverage_repair_triggered:
                fixed_open_rel_map = self.agent._repair_relation_coverage(
                    cleaned_text=chunk_state["cleaned_text"],
                    entities=copy.deepcopy(resolved_entities),
                    all_relations={
                        rid: rel for rid, rel in copy.deepcopy(fixed_open_rel_map).items()
                        if str(rel.get("relation_type", "")).strip() in _FAST_OPEN_GROUNDED_RELATION_TYPES
                    },
                    rid_namespace=f"{chunk_state['rid_namespace']}:event_first.coverage",
                    memory_context="",
                )
            stats["coverage_repair_sec"] += time.perf_counter() - coverage_repair_start

            combined_rel_map = dict(item["auto_relations"])
            combined_rel_map.update(fixed_open_rel_map)
            pruned_entities, pruned_rel_map, dropped_counts = _prune_low_value_entities_and_relations(
                entities=chunk_state["event_nodes"] + chunk_state["occasion_nodes"] + resolved_entities,
                relations=combined_rel_map,
            )
            entities_all = _merge_entities(entities_all, pruned_entities)
            for rid, rel in pruned_rel_map.items():
                all_relations[rid] = rel

            stats["coverage_repair_triggered_chunks"] += coverage_repair_triggered
            stats["resolved_entity_count"] += sum(
                1 for ent in pruned_entities if str(ent.get("type", "")).strip() not in {"Event", "Occasion"}
            )
            stats["pruned_low_value_entity_count"] += sum(dropped_counts.values())
            stats["open_relation_count_after_fix"] += sum(
                1 for rid in pruned_rel_map if ":event_first.open_rel" in rid or ":coverage#" in rid
            )
            stats["auto_relation_count"] += sum(1 for rid in pruned_rel_map if rid.startswith("auto#"))

        return {"entities": entities_all, "relations": list(all_relations.values()), "stats": stats}

    def _proposal_entity_names(self, proposals: List[Dict[str, Any]]) -> set[str]:
        names: set[str] = set()
        for proposal in proposals or []:
            if not isinstance(proposal, dict):
                continue
            for key in ["subject", "object"]:
                value = str(proposal.get(key, "")).strip()
                if value:
                    names.add(_norm_name(value))
            for item in proposal.get("new_entities", []) or []:
                if not isinstance(item, dict):
                    continue
                value = str(item.get("name", "")).strip()
                if value:
                    names.add(_norm_name(value))
        return names

    def _select_relevant_doc_entities(
        self,
        *,
        cleaned_text: str,
        doc_grounded_entities: List[Dict[str, Any]],
        chunk_state: Dict[str, Any],
        extra_names: set[str] | None = None,
    ) -> List[Dict[str, Any]]:
        text_lower = cleaned_text.lower()
        seed_names = set(extra_names or set())
        seed_names.update(
            _collect_names(
                chunk_state.get("typed_frame_entities") or [],
                chunk_state.get("external_typed_entities") or [],
                chunk_state.get("related_entity_candidates") or [],
                chunk_state.get("open_entities") or [],
                chunk_state.get("grounding_candidates") or [],
            )
        )
        selected: List[Dict[str, Any]] = []
        for entity in doc_grounded_entities or []:
            if not isinstance(entity, dict):
                continue
            name = str(entity.get("name", "")).strip()
            if not name:
                continue
            norm = _norm_name(name)
            if norm in seed_names or norm in text_lower:
                selected.append(entity)
        return _dedup_grounded_entities(selected)

    def _collect_chunk_pass1(self, *, cleaned_text: str) -> Dict[str, Any]:
        frame_start = time.perf_counter()
        frames = json.loads(self.frame_extractor.call(json.dumps({"text": cleaned_text, "memory_context": ""})))
        frame_extraction_sec = time.perf_counter() - frame_start

        event_nodes, occasion_nodes = _frame_nodes(frames)
        frame_node_names = {_norm_name(item["name"]) for item in event_nodes + occasion_nodes}
        typed_frame_entities = _collect_direct_typed_frame_entities(
            frames,
            frame_node_names=frame_node_names,
            scope_rules=getattr(self.agent, "scope_rules", {}) or {},
        )
        related_entity_candidates = _collect_related_entity_candidates(frames, frame_node_names=frame_node_names)

        external_entity_start = time.perf_counter()
        external_entity_payload = self.external_entity_extractor.extract(
            text=cleaned_text,
            known_names=[item.get("name", "") for item in (event_nodes + occasion_nodes + typed_frame_entities)],
            scope_rules=getattr(self.agent, "scope_rules", {}) or {},
        )
        external_entity_sec = time.perf_counter() - external_entity_start
        external_typed_entities = _dedup_typed_entities(external_entity_payload.get("typed_entities") or [])
        external_open_candidates = _dedup_open_entities(external_entity_payload.get("open_candidates") or [])
        external_stats = external_entity_payload.get("stats") or {}

        kg_cfg = getattr(self.config, "knowledge_graph_builder", None)
        skip_llm_min_typed = int(getattr(kg_cfg, "fast_external_entity_skip_llm_min_typed", 3) or 3)
        open_entity_llm_skipped = 0
        open_entity_start = time.perf_counter()
        open_entities: List[Dict[str, Any]] = []
        external_backend = str(external_stats.get("backend", "")).strip().lower()
        should_skip_open_entity_llm = (
            (external_backend == "qwen" and (len(external_typed_entities) + len(external_open_candidates)) > 0)
            or len(external_typed_entities) >= skip_llm_min_typed
        )
        if should_skip_open_entity_llm:
            open_entity_llm_skipped = 1
        else:
            open_entities = _json_load_list(
                self.open_entity_extractor.call(
                    json.dumps(
                        {
                            "text": cleaned_text,
                            "known_entities": _known_entities_text(
                                event_nodes,
                                occasion_nodes,
                                typed_frame_entities + external_typed_entities,
                                _merge_open_entity_candidates(related_entity_candidates, external_open_candidates),
                            ),
                            "memory_context": "",
                        }
                    )
                )
            )
        open_entity_extraction_sec = time.perf_counter() - open_entity_start

        grounding_candidates = _merge_open_entity_candidates(
            related_entity_candidates,
            external_open_candidates,
            open_entities or [],
        )

        return {
            "cleaned_text": cleaned_text,
            "frames": frames,
            "event_nodes": event_nodes,
            "occasion_nodes": occasion_nodes,
            "typed_frame_entities": typed_frame_entities,
            "related_entity_candidates": related_entity_candidates,
            "external_typed_entities": external_typed_entities,
            "external_open_candidates": external_open_candidates,
            "open_entities": open_entities,
            "grounding_candidates": grounding_candidates,
            "stats": {
                "frame_extraction_sec": frame_extraction_sec,
                "external_entity_sec": external_entity_sec,
                "open_entity_extraction_sec": open_entity_extraction_sec,
                "external_entity_raw_count": int(external_stats.get("raw_count", 0) or 0),
                "external_entity_typed_count": len(external_typed_entities),
                "external_entity_open_candidate_count": len(external_open_candidates),
                "open_entity_llm_skipped_chunks": open_entity_llm_skipped,
            },
        }

    def run_chunk(self, *, cleaned_text: str, rid_namespace: str) -> Dict[str, Any]:
        chunk_start = time.perf_counter()

        frame_start = time.perf_counter()
        frames = json.loads(self.frame_extractor.call(json.dumps({"text": cleaned_text, "memory_context": ""})))
        frame_extraction_sec = time.perf_counter() - frame_start

        event_nodes, occasion_nodes = _frame_nodes(frames)
        frame_node_names = {_norm_name(item["name"]) for item in event_nodes + occasion_nodes}
        typed_frame_entities = _collect_direct_typed_frame_entities(
            frames,
            frame_node_names=frame_node_names,
            scope_rules=getattr(self.agent, "scope_rules", {}) or {},
        )
        related_entity_candidates = _collect_related_entity_candidates(frames, frame_node_names=frame_node_names)

        external_entity_start = time.perf_counter()
        external_entity_payload = self.external_entity_extractor.extract(
            text=cleaned_text,
            known_names=[item.get("name", "") for item in (event_nodes + occasion_nodes + typed_frame_entities)],
            scope_rules=getattr(self.agent, "scope_rules", {}) or {},
        )
        external_entity_sec = time.perf_counter() - external_entity_start
        external_typed_entities = _dedup_typed_entities(external_entity_payload.get("typed_entities") or [])
        external_open_candidates = _dedup_open_entities(external_entity_payload.get("open_candidates") or [])
        external_stats = external_entity_payload.get("stats") or {}

        kg_cfg = getattr(self.config, "knowledge_graph_builder", None)
        skip_llm_min_typed = int(getattr(kg_cfg, "fast_external_entity_skip_llm_min_typed", 3) or 3)
        open_entity_llm_skipped = 0
        open_entity_start = time.perf_counter()
        open_entities: List[Dict[str, Any]] = []
        external_backend = str(external_stats.get("backend", "")).strip().lower()
        should_skip_open_entity_llm = (
            (external_backend == "qwen" and (len(external_typed_entities) + len(external_open_candidates)) > 0)
            or len(external_typed_entities) >= skip_llm_min_typed
        )
        if should_skip_open_entity_llm:
            open_entity_llm_skipped = 1
        else:
            open_entities = _json_load_list(
                self.open_entity_extractor.call(
                    json.dumps(
                        {
                            "text": cleaned_text,
                            "known_entities": _known_entities_text(
                                event_nodes,
                                occasion_nodes,
                                typed_frame_entities + external_typed_entities,
                                _merge_open_entity_candidates(related_entity_candidates, external_open_candidates),
                            ),
                            "memory_context": "",
                        }
                    )
                )
            )
        open_entity_extraction_sec = time.perf_counter() - open_entity_start

        grounding_candidates = _merge_open_entity_candidates(
            related_entity_candidates,
            external_open_candidates,
            open_entities or [],
        )
        entity_grounding_start = time.perf_counter()
        grounded_open_entities = _json_load_list(
            self.entity_grounder.call(
                json.dumps(
                    {
                        "text": cleaned_text,
                        "open_entities": grounding_candidates,
                        "memory_context": "",
                    }
                )
            )
        )
        for item in grounded_open_entities:
            item["source_kind"] = "open_grounded"
        entity_grounding_sec = time.perf_counter() - entity_grounding_start

        grounded_entities = _dedup_grounded_entities(typed_frame_entities + external_typed_entities + grounded_open_entities)
        initial_grounded_entity_count = len(grounded_entities)

        auto_relation_start = time.perf_counter()
        auto_relations = _auto_relations_from_frames(
            frames=frames,
            grounded_entities=grounded_entities,
            event_nodes=event_nodes,
            occasion_nodes=occasion_nodes,
        )
        auto_relation_build_sec = time.perf_counter() - auto_relation_start

        self.agent.config.knowledge_graph_builder.relation_extraction_mode = "open_then_ground"

        open_relation_extract_start = time.perf_counter()
        open_relation_proposals: List[Dict[str, Any]] = []
        if _has_fast_open_relation_candidates(agent=self.agent, entities=grounded_entities):
            relation_hints = _build_fast_open_relation_hints(self.agent, entities=grounded_entities)
            raw_open_relations = self.agent.extractor.extract_open_relations(
                text=cleaned_text,
                extracted_entities=self.agent._entities_text_for_extractor(grounded_entities),
                previous_results=None,
                feedbacks=None,
                memory_context="",
                relation_hints=relation_hints,
                focus_entities="",
            )
            open_relation_proposals = _json_load_list(raw_open_relations)
        open_relation_extraction_sec = time.perf_counter() - open_relation_extract_start

        relation_brought_candidates = _collect_relation_brought_entity_candidates(
            open_relation_proposals,
            known_entity_names={_norm_name(item.get("name", "")) for item in grounded_entities},
        )
        relation_brought_entity_candidate_count = len(relation_brought_candidates)

        relation_brought_entity_grounding_start = time.perf_counter()
        relation_brought_grounded_entities: List[Dict[str, Any]] = []
        if relation_brought_candidates:
            relation_brought_grounded_entities = _json_load_list(
                self.entity_grounder.call(
                    json.dumps(
                        {
                            "text": cleaned_text,
                            "open_entities": relation_brought_candidates,
                            "memory_context": "",
                        }
                    )
                )
            )
            for item in relation_brought_grounded_entities:
                item["source_kind"] = "relation_brought_grounded"
            grounded_entities = _dedup_grounded_entities(grounded_entities + relation_brought_grounded_entities)
        relation_brought_entity_grounding_sec = time.perf_counter() - relation_brought_entity_grounding_start

        relation_grounding_start = time.perf_counter()
        open_relation_proposals = _canonicalize_open_relation_proposals(
            open_relation_proposals,
            entities=grounded_entities,
        )
        open_relation_proposals = _filter_fast_open_relation_proposals(
            open_relation_proposals,
            agent=self.agent,
            entities=grounded_entities,
        )
        seeded_relations = self.agent._prepare_grounded_open_relation_seeds(
            cleaned_text=cleaned_text,
            proposals=copy.deepcopy(open_relation_proposals),
            entities=copy.deepcopy(grounded_entities),
            rid_prefix=f"{rid_namespace}:event_first.open_rel",
            memory_context="",
        )
        seeded_relations = apply_canonicalization_to_relations(seeded_relations, build_name_canonicalizer(grounded_entities))
        seeded_relations = self.agent._filter_illegal_is_a_relations(seeded_relations, entities=grounded_entities)

        open_rel_map: Dict[str, Dict[str, Any]] = {}
        open_feedbacks: Dict[str, List[Dict[str, Any]]] = {}
        for rel in seeded_relations:
            rid = str(rel.get("rid", "")).strip()
            if not rid:
                continue
            rel2, feedback = self.agent._revalidate_one_relation_global(rel, entities=grounded_entities)
            open_rel_map[rid] = rel2
            if feedback is not None:
                self.agent._append_relation_feedback(
                    open_feedbacks,
                    str(feedback.get("error_type", "")).strip() or "unknown validation error",
                    rid=rid,
                    feedback=str(feedback.get("feedback", "")).strip(),
                    subject=str(feedback.get("subject", "")).strip() or str(rel2.get("subject", "")).strip(),
                    object=str(feedback.get("object", "")).strip() or str(rel2.get("object", "")).strip(),
                    relation_type=str(feedback.get("relation_type", "")).strip() or str(rel2.get("relation_type", "")).strip(),
                    relation_name=str(feedback.get("relation_name", "")).strip() or str(rel2.get("relation_name", "")).strip(),
                )
        relation_grounding_sec = time.perf_counter() - relation_grounding_start

        resolve_errors_start = time.perf_counter()
        resolved_entities, fixed_open_rel_map = self.agent._resolve_errors(
            entities=copy.deepcopy(grounded_entities),
            all_relations=copy.deepcopy(open_rel_map),
            all_feedbacks=copy.deepcopy(open_feedbacks),
            content=cleaned_text,
        )
        resolve_errors_sec = time.perf_counter() - resolve_errors_start
        resolved_entities = _dedup_grounded_entities(resolved_entities)

        coverage_repair_triggered = 0
        coverage_probe_rel_map = dict(auto_relations)
        coverage_probe_rel_map.update(fixed_open_rel_map)
        if _should_run_fast_coverage_repair(
            frames=frames,
            grounded_entities=resolved_entities,
            current_relations=coverage_probe_rel_map,
        ):
            coverage_repair_triggered = 1
        coverage_repair_start = time.perf_counter()
        if coverage_repair_triggered:
            fixed_open_rel_map = self.agent._repair_relation_coverage(
                cleaned_text=cleaned_text,
                entities=copy.deepcopy(resolved_entities),
                all_relations={
                    rid: rel for rid, rel in copy.deepcopy(fixed_open_rel_map).items()
                    if str(rel.get("relation_type", "")).strip() in _FAST_OPEN_GROUNDED_RELATION_TYPES
                },
                rid_namespace=f"{rid_namespace}:event_first.coverage",
                memory_context="",
            )
        coverage_repair_sec = time.perf_counter() - coverage_repair_start

        combined_rel_map = dict(auto_relations)
        combined_rel_map.update(fixed_open_rel_map)
        pruned_entities, pruned_rel_map, dropped_counts = _prune_low_value_entities_and_relations(
            entities=event_nodes + occasion_nodes + resolved_entities,
            relations=combined_rel_map,
        )
        total_chunk_sec = time.perf_counter() - chunk_start

        return {
            "entities": pruned_entities,
            "relations": pruned_rel_map,
            "stats": {
                "auto_relation_count": sum(1 for rid in pruned_rel_map if rid.startswith("auto#")),
                "open_relation_proposal_count": len(open_relation_proposals),
                "open_relation_count_after_fix": sum(1 for rid in pruned_rel_map if ":event_first.open_rel" in rid or ":coverage#" in rid),
                "initial_grounded_entity_count": initial_grounded_entity_count,
                "relation_brought_entity_candidate_count": relation_brought_entity_candidate_count,
                "relation_brought_grounded_entity_count": len(relation_brought_grounded_entities),
                "resolved_entity_count": sum(
                    1 for item in pruned_entities if str(item.get("type", "")).strip() not in {"Event", "Occasion"}
                ),
                "pruned_low_value_entity_count": sum(dropped_counts.values()),
                "coverage_repair_triggered": coverage_repair_triggered,
                "external_entity_raw_count": int(external_stats.get("raw_count", 0) or 0),
                "external_entity_typed_count": len(external_typed_entities),
                "external_entity_open_candidate_count": len(external_open_candidates),
                "open_entity_llm_skipped_chunks": open_entity_llm_skipped,
                "frame_extraction_sec": frame_extraction_sec,
                "external_entity_sec": external_entity_sec,
                "open_entity_extraction_sec": open_entity_extraction_sec,
                "entity_grounding_sec": entity_grounding_sec,
                "auto_relation_build_sec": auto_relation_build_sec,
                "open_relation_extraction_sec": open_relation_extraction_sec,
                "relation_brought_entity_grounding_sec": relation_brought_entity_grounding_sec,
                "relation_grounding_sec": relation_grounding_sec,
                "resolve_errors_sec": resolve_errors_sec,
                "coverage_repair_sec": coverage_repair_sec,
                "total_chunk_sec": total_chunk_sec,
            },
        }

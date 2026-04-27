from __future__ import annotations

import copy
import json
import re
import time
from collections import Counter
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
from core.utils.general_utils import word_len
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
            row = dict(item)
            row["name"] = name
            row["type"] = etype
            row["description"] = desc
            row["scope"] = str(item.get("scope", "")).strip() or "local"
            merged[key] = row
    return list(merged.values())


def _typed_entities_from_metadata_ner(
    metadata: Dict[str, Any] | None,
    *,
    known_names: set[str],
    scope_rules: Dict[str, str],
) -> List[Dict[str, Any]]:
    allowed = {"Character", "Location", "Object", "Concept"}
    out: List[Dict[str, Any]] = []
    md = metadata if isinstance(metadata, dict) else {}
    raw_items = md.get("ner_entities") or []
    if not isinstance(raw_items, list):
        raw_items = []

    for item in raw_items:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        etype = str(item.get("type", "")).strip()
        if not name or etype not in allowed:
            continue
        if _norm_name(name) in known_names:
            continue
        if etype == "Character" and _is_generic_character_name(name):
            continue
        description = str(item.get("description", "")).strip() or f"Detected by metadata NER as {etype}."
        try:
            confidence = float(item.get("confidence", 0.0) or 0.0)
        except Exception:
            confidence = 0.0
        out.append(
            {
                "name": name,
                "type": etype,
                "description": description,
                "scope": scope_rules.get(etype, "local"),
                "source_kind": "metadata_ner",
                "confidence": round(confidence, 4),
            }
        )
    return _dedup_typed_entities(out)


def _typed_entities_to_open_candidates(items: List[Dict[str, Any]], *, source_hint: str = "") -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for item in items or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        etype = str(item.get("type", "")).strip()
        desc = str(item.get("description", "")).strip()
        if not name:
            continue
        if desc:
            candidate_desc = desc
        elif etype:
            candidate_desc = f"Candidate {etype} entity from {source_hint or 'NER'}."
        else:
            candidate_desc = f"Candidate entity from {source_hint or 'NER'}."
        if etype:
            candidate_desc = f"[NER candidate type: {etype}] {candidate_desc}"
        out.append({"name": name, "description": candidate_desc})
    return _dedup_open_entities(out)


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
        if cur is None:
            merged[key] = dict(item)
            continue
        merged[key] = _merge_entity_records(cur, item)
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
        merged[key] = _merge_entity_records(cur, item)
    return list(merged.values())


def _entity_index(entities: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for e in entities if isinstance(entities, list) else []:
        if not isinstance(e, dict):
            continue
        name = str(e.get("name", "")).strip()
        if name:
            out[name] = e
        for alias in e.get("aliases") or []:
            alias_text = str(alias or "").strip()
            if alias_text and alias_text not in out:
                out[alias_text] = e
    return out


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


def _is_auto_relation_rid(rid: Any) -> bool:
    text = str(rid or "").strip()
    return text.startswith("auto#") or ":auto#" in text


def _grounding_mode_counts(relations: List[Dict[str, Any]]) -> Dict[str, int]:
    counter: Counter[str] = Counter()
    for rel in relations or []:
        if not isinstance(rel, dict):
            continue
        props = rel.get("properties") or {}
        if not isinstance(props, dict):
            continue
        mode = str(props.get("grounding_mode", "") or "").strip()
        if mode:
            counter[mode] += 1
    return dict(counter)


def _merge_count_maps(base: Dict[str, int], incoming: Dict[str, int]) -> Dict[str, int]:
    merged: Counter[str] = Counter()
    for mapping in [base or {}, incoming or {}]:
        for key, value in mapping.items():
            key_text = str(key or "").strip()
            if not key_text:
                continue
            merged[key_text] += int(value or 0)
    return dict(merged)


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


def _should_ground_relation_brought_candidate(name: str, description: str) -> bool:
    raw_name = _normalize_space(name)
    raw_desc = _normalize_space(description)
    if not raw_name:
        return False
    if _looks_named_character(raw_name):
        return True
    if _is_generic_character_name(raw_name):
        return False
    if _has_non_ascii(raw_name):
        if len(raw_name) >= 3 and not any(marker in raw_name for marker in ["消息", "东西", "东西们"]):
            return True
    else:
        words = raw_name.split()
        if len(words) >= 2 and any(word[:1].isupper() for word in words):
            return True
    combined = _norm_name(f"{raw_name} {raw_desc}")
    high_value_markers = [
        "family",
        "committee",
        "newspaper",
        "estate",
        "play",
        "marriage",
        "divorce",
        "alimony",
        "organization",
        "club",
        "company",
        "office",
        "household",
        "报社",
        "委员会",
        "家族",
        "婚姻",
        "离婚",
        "财产",
        "庄园",
        "剧本",
    ]
    if any(marker in combined for marker in high_value_markers):
        return True
    return False


def _filter_relation_brought_grounding_candidates(
    candidates: List[Dict[str, str]],
) -> List[Dict[str, str]]:
    return [
        item
        for item in _dedup_open_entities(candidates)
        if isinstance(item, dict)
        and _should_ground_relation_brought_candidate(
            str(item.get("name", "")).strip(),
            str(item.get("description", "")).strip(),
        )
    ]


def _infer_relation_brought_rule_entities(
    candidates: List[Dict[str, str]],
    *,
    scope_rules: Dict[str, str],
) -> List[Dict[str, Any]]:
    """
    Fast-path replacement for a second LLM grounding pass.
    Only promote relation-mentioned entities when their type is conservative.
    """
    out: List[Dict[str, Any]] = []
    concept_markers = {
        "family",
        "committee",
        "organization",
        "organisation",
        "club",
        "company",
        "office",
        "agency",
        "government",
        "project",
        "plan",
        "program",
        "marriage",
        "divorce",
        "estate",
        "newspaper",
        "newspaper office",
        "家族",
        "家庭",
        "委员会",
        "组织",
        "机构",
        "公司",
        "政府",
        "计划",
        "工程",
        "项目",
        "婚姻",
        "离婚",
        "报社",
    }
    object_markers = {
        "letter",
        "document",
        "contract",
        "money",
        "newspaper",
        "paper",
        "phone",
        "car",
        "ship",
        "weapon",
        "信",
        "文件",
        "合同",
        "钱",
        "报纸",
        "电话",
        "车",
        "武器",
    }

    for item in _dedup_open_entities(candidates):
        name = _normalize_space(item.get("name", ""))
        desc = _normalize_space(item.get("description", ""))
        if not name:
            continue
        combined = _norm_name(f"{name} {desc}")
        if any(marker in combined for marker in ["news of", "消息", "传闻", "rumor"]):
            continue
        etype = ""
        if any(marker in combined for marker in concept_markers):
            etype = "Concept"
        elif _looks_named_character(name):
            etype = "Character"
        elif any(marker in combined for marker in object_markers) and not _is_low_value_object_name(name):
            etype = "Object"
        if not etype:
            continue
        out.append(
            {
                "name": name,
                "description": desc,
                "type": etype,
                "scope": scope_rules.get(etype, "local"),
                "source_kind": "relation_brought_rule",
            }
        )
    return _dedup_typed_entities(out)


def _character_alias_score(
    entity: Dict[str, Any],
    target: Dict[str, Any],
    *,
    global_characters: List[Dict[str, Any]],
) -> int:
    source_name = str(entity.get("name", "")).strip()
    target_name = str(target.get("name", "")).strip()
    if not source_name or not target_name or _norm_name(source_name) == _norm_name(target_name):
        return 0
    if str(target.get("scope", "")).strip() != "global":
        return 0

    source_desc = str(entity.get("description", "")).strip()
    target_text = f"{target_name} {str(target.get('description', '')).strip()} {' '.join(target.get('aliases') or [])}"
    source_tokens = [tok for tok in _character_tokens(source_name) if tok not in _CHARACTER_TITLE_WORDS]
    target_tokens = [tok for tok in _character_tokens(target_name) if tok not in _CHARACTER_TITLE_WORDS]
    if not source_tokens or not target_tokens:
        return 0

    score = 0
    if _looks_named_character(source_name):
        if not _looks_named_character(target_name):
            return 0
        src_first = source_tokens[0]
        tgt_first = target_tokens[0]
        if src_first == tgt_first:
            score += 10
        elif len(src_first) >= 4 and tgt_first.startswith(src_first):
            score += 8
        elif len(tgt_first) >= 4 and src_first.startswith(tgt_first):
            score += 6

        if len(source_tokens) > 1 and set(source_tokens).issubset(set(target_tokens)):
            score += 6
        elif len(source_tokens) == 1 and src_first in set(target_tokens[1:]):
            score -= 4
    else:
        role_terms = _character_role_terms(source_name)
        if not role_terms:
            return 0
        target_text_norm = _norm_name(target_text)
        for role in role_terms:
            if role in target_text_norm:
                score += 3
        anchors = _character_anchor_names(source_desc, candidates=global_characters)
        if anchors and any(anchor in target_text_norm for anchor in anchors):
            score += 4
        if _looks_named_character(target_name):
            score += 2

    return score


def _normalize_character_entities(
    entities: List[Dict[str, Any]],
    *,
    mention_counts: Counter[str] | None = None,
) -> List[Dict[str, Any]]:
    mention_counts = mention_counts or Counter()
    normalized = [dict(item) for item in (entities or []) if isinstance(item, dict)]
    global_characters = [
        item
        for item in normalized
        if str(item.get("type", "")).strip() == "Character"
        and str(item.get("scope", "")).strip() == "global"
        and _looks_named_character(str(item.get("name", "")).strip())
    ]

    for item in normalized:
        if str(item.get("type", "")).strip() != "Character":
            continue
        original_name = str(item.get("name", "")).strip()
        if not original_name:
            continue

        best_target = None
        best_score = 0
        second_score = 0
        for target in global_characters:
            score = _character_alias_score(item, target, global_characters=global_characters)
            if score > best_score:
                second_score = best_score
                best_score = score
                best_target = target
            elif score > second_score:
                second_score = score

        if best_target is not None and best_score >= 8 and best_score >= second_score + 2:
            target_name = str(best_target.get("name", "")).strip()
            if target_name and _norm_name(target_name) != _norm_name(original_name):
                existing_aliases = item.get("aliases") or []
                item["aliases"] = _sanitize_entity_aliases(
                    target_name,
                    "Character",
                    _merge_unique_strings(existing_aliases, [original_name]),
                )
                item["name"] = target_name
                item["scope"] = "global"
                item["alias_of"] = target_name
                continue

        if not _looks_named_character(original_name):
            item["scope"] = "local"
            continue

        if (
            str(item.get("scope", "")).strip() != "global"
            and _looks_named_character(original_name)
            and int(mention_counts.get(_norm_name(original_name), 0)) >= 2
        ):
            item["scope"] = "global"

    return _dedup_grounded_entities(normalized)


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
_CHARACTER_PRONOUN_PREFIXES = (
    "his ",
    "her ",
    "their ",
    "my ",
    "our ",
    "your ",
    "this ",
    "that ",
    "these ",
    "those ",
    "another ",
    "some ",
    "a ",
    "an ",
    "the ",
)
_CHARACTER_ROLE_WORDS = {
    "husband",
    "wife",
    "lawyer",
    "friend",
    "butler",
    "reporter",
    "reporters",
    "mother",
    "father",
    "daughter",
    "son",
    "brother",
    "sister",
    "guest",
    "guests",
    "valet",
    "chauffeur",
    "doctor",
    "nurse",
    "captain",
    "driver",
    "bartender",
    "maid",
    "servant",
    "footman",
    "host",
    "hostess",
    "girl",
    "boy",
    "man",
    "woman",
    "lady",
    "丈夫",
    "妻子",
    "律师",
    "朋友",
    "管家",
    "记者",
    "母亲",
    "父亲",
    "女儿",
    "儿子",
    "兄弟",
    "姐妹",
}
_CHARACTER_TITLE_WORDS = {"mr", "mrs", "miss", "ms", "dr", "sir", "lady", "captain", "capt", "prof", "professor"}
_SOURCE_KIND_PRIORITY = {
    "frame_slot": 4,
    "external_typed": 3,
    "open_grounded": 2,
    "relation_brought_grounded": 1,
    "relation_brought_rule": 1,
    "": 0,
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
    if norm in _CHARACTER_ROLE_WORDS:
        return True
    if norm.startswith(_CHARACTER_PRONOUN_PREFIXES):
        return True
    if any(marker in norm for marker in ["a lady", "a man", "a woman", "young man", "young woman", "in the group"]):
        return True
    if any(ch.isdigit() for ch in norm) and any(token in norm for token in ["men", "women", "男女", "人"]):
        return True
    return False


def _clean_character_token(token: str) -> str:
    return re.sub(r"^[^A-Za-z\u4e00-\u9fff]+|[^A-Za-z\u4e00-\u9fff]+$", "", str(token or "").strip()).lower()


def _character_tokens(name: str) -> List[str]:
    return [tok for tok in (_clean_character_token(x) for x in str(name or "").split()) if tok]


def _character_anchor_names(text: str, *, candidates: List[Dict[str, Any]]) -> set[str]:
    norm_text = _norm_name(text)
    anchors: set[str] = set()
    if not norm_text:
        return anchors
    for item in candidates or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        if not name:
            continue
        if _norm_name(name) in norm_text:
            anchors.add(_norm_name(name))
        for alias in item.get("aliases") or []:
            alias_norm = _norm_name(alias)
            if alias_norm and alias_norm in norm_text:
                anchors.add(alias_norm)
    return anchors


def _character_role_terms(name: str) -> set[str]:
    tokens = set(_character_tokens(name))
    roles = {tok for tok in tokens if tok in _CHARACTER_ROLE_WORDS}
    norm = _norm_name(name)
    if "lawyer friend" in norm:
        roles.update({"lawyer", "friend"})
    return roles


def _looks_named_character(name: str) -> bool:
    raw = _normalize_space(name)
    if not raw or _is_generic_character_name(raw):
        return False
    if _has_non_ascii(raw):
        return not any(marker in raw for marker in ["某", "一个", "一些", "他的", "她的", "他们的"])
    words = raw.split()
    if not words:
        return False
    if raw.lower().startswith(_CHARACTER_PRONOUN_PREFIXES):
        return False
    sig_tokens = [tok for tok in _character_tokens(raw) if tok not in _CHARACTER_TITLE_WORDS]
    if not sig_tokens:
        return False
    if len(sig_tokens) == 1 and sig_tokens[0] in _CHARACTER_ROLE_WORDS:
        return False
    non_title_words = [w for w in words if _clean_character_token(w) not in _CHARACTER_TITLE_WORDS]
    if not non_title_words:
        return False
    if len(non_title_words) == 1:
        word = non_title_words[0]
        return bool(word[:1].isupper() or word.isupper())
    return any(word[:1].isupper() or word.isupper() for word in non_title_words)


def _merge_unique_strings(*groups: Any) -> List[str]:
    out: List[str] = []
    seen = set()
    for group in groups:
        if isinstance(group, str):
            group = [group]
        if not isinstance(group, list):
            continue
        for item in group:
            text = str(item or "").strip()
            if not text:
                continue
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(text)
    return out


def _is_explicit_character_alias(alias: str, canonical_name: str) -> bool:
    alias_text = _normalize_space(alias)
    canonical_text = _normalize_space(canonical_name)
    if not alias_text or not canonical_text:
        return False
    if _norm_name(alias_text) == _norm_name(canonical_text):
        return False
    if not _looks_named_character(alias_text) or not _looks_named_character(canonical_text):
        return False

    alias_tokens = [tok for tok in _character_tokens(alias_text) if tok not in _CHARACTER_TITLE_WORDS and tok != "s"]
    canonical_tokens = [tok for tok in _character_tokens(canonical_text) if tok not in _CHARACTER_TITLE_WORDS and tok != "s"]
    if not alias_tokens or not canonical_tokens:
        return False
    if any(tok in _CHARACTER_ROLE_WORDS for tok in alias_tokens):
        return False

    alias_set = set(alias_tokens)
    canonical_set = set(canonical_tokens)
    if alias_set.issubset(canonical_set) or canonical_set.issubset(alias_set):
        return True

    alias_head = alias_tokens[0]
    canonical_head = canonical_tokens[0]
    if alias_head == canonical_head:
        return True
    if len(alias_head) >= 4 and canonical_head.startswith(alias_head):
        return True
    if len(canonical_head) >= 4 and alias_head.startswith(canonical_head):
        return True
    return False


def _sanitize_entity_aliases(name: str, etype: str, aliases: Any) -> List[str]:
    merged = _merge_unique_strings(aliases or [])
    if str(etype).strip() != "Character":
        return merged
    return [alias for alias in merged if _is_explicit_character_alias(alias, name)]


def _merge_entity_records(current: Dict[str, Any], incoming: Dict[str, Any]) -> Dict[str, Any]:
    cur = dict(current or {})
    new = dict(incoming or {})
    cur_scope = str(cur.get("scope", "")).strip()
    new_scope = str(new.get("scope", "")).strip()
    cur_desc = str(cur.get("description", "")).strip()
    new_desc = str(new.get("description", "")).strip()
    cur_kind = str(cur.get("source_kind", "")).strip()
    new_kind = str(new.get("source_kind", "")).strip()

    take_new = False
    if cur_scope != "global" and new_scope == "global":
        take_new = True
    elif _SOURCE_KIND_PRIORITY.get(new_kind, 0) > _SOURCE_KIND_PRIORITY.get(cur_kind, 0):
        take_new = True
    elif len(new_desc) > len(cur_desc):
        take_new = True

    base = dict(new if take_new else cur)
    other = cur if take_new else new

    base["description"] = new_desc if len(new_desc) >= len(cur_desc) else cur_desc
    base["aliases"] = _sanitize_entity_aliases(
        str(base.get("name", "")).strip(),
        str(base.get("type", "")).strip(),
        _merge_unique_strings(base.get("aliases") or [], other.get("aliases") or []),
    )
    alias_of = str(base.get("alias_of") or "").strip() or str(other.get("alias_of") or "").strip()
    if alias_of:
        base["alias_of"] = alias_of
    source_kinds = _merge_unique_strings(
        base.get("source_kinds") or [],
        other.get("source_kinds") or [],
        base.get("source_kind") or "",
        other.get("source_kind") or "",
    )
    if source_kinds:
        base["source_kinds"] = source_kinds
    best_kind = cur_kind
    if _SOURCE_KIND_PRIORITY.get(new_kind, 0) > _SOURCE_KIND_PRIORITY.get(cur_kind, 0):
        best_kind = new_kind
    if best_kind:
        base["source_kind"] = best_kind
    return base


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
    rid_prefix: str = "auto",
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
        key = f"{rid_prefix}#{rid_i}" if rid_prefix else f"auto#{rid_i}"
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

    def _ground_open_candidates(
        self,
        *,
        text: str,
        candidates: List[Dict[str, str]],
        source_kind: str,
    ) -> List[Dict[str, Any]]:
        grounded_items: List[Dict[str, Any]] = []
        deduped = _dedup_open_entities(candidates)
        if not text or not deduped:
            return grounded_items
        grounded_items = _json_load_list(
            self.entity_grounder.call(
                json.dumps(
                    {
                        "text": text,
                        "open_entities": deduped,
                        "memory_context": "",
                    }
                )
            )
        )
        for item in grounded_items:
            item["source_kind"] = source_kind
        return grounded_items

    def _relation_window_limits(self) -> Tuple[int, int]:
        kg_cfg = getattr(self.config, "knowledge_graph_builder", None)
        target_words = int(getattr(kg_cfg, "fast_relation_window_target_words", 850) or 850)
        max_words = int(getattr(kg_cfg, "fast_relation_window_max_words", 1000) or 1000)
        max_words = max(300, max_words)
        target_words = max(200, min(target_words, max_words))
        return target_words, max_words

    def _split_relation_text(self, text: str, *, max_words: int) -> List[str]:
        raw = str(text or "").strip()
        if not raw:
            return []
        if word_len(raw, lang="auto") <= max_words:
            return [raw]

        sentence_parts = [
            part.strip()
            for part in re.split(r"(?<=[。！？!?；;])\s+|(?<=[.!?;])\s+|\n+", raw)
            if str(part or "").strip()
        ]
        if not sentence_parts:
            sentence_parts = [raw]

        out: List[str] = []
        current: List[str] = []
        current_words = 0

        def flush() -> None:
            nonlocal current, current_words
            if current:
                out.append(" ".join(current).strip())
            current = []
            current_words = 0

        for part in sentence_parts:
            part_words = word_len(part, lang="auto")
            if part_words <= 0:
                continue
            if part_words > max_words:
                flush()
                tokens = part.split()
                if len(tokens) > 1:
                    for start in range(0, len(tokens), max_words):
                        out.append(" ".join(tokens[start : start + max_words]).strip())
                else:
                    out.append(part)
                continue
            if current and current_words + part_words > max_words:
                flush()
            current.append(part)
            current_words += part_words
        flush()
        return [part for part in out if part]

    def _build_relation_windows(self, chunk_states: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        target_words, max_words = self._relation_window_limits()
        windows: List[Dict[str, Any]] = []
        current_texts: List[str] = []
        current_indices: List[int] = []
        current_words = 0

        def flush() -> None:
            nonlocal current_texts, current_indices, current_words
            if current_texts:
                windows.append(
                    {
                        "text": "\n\n".join(current_texts).strip(),
                        "chunk_indices": list(current_indices),
                    }
                )
            current_texts = []
            current_indices = []
            current_words = 0

        for idx, state in enumerate(chunk_states or []):
            text = str((state or {}).get("cleaned_text", "") or "").strip()
            if not text:
                continue
            text_words = word_len(text, lang="auto")
            if text_words > max_words:
                flush()
                for part in self._split_relation_text(text, max_words=max_words):
                    windows.append({"text": part, "chunk_indices": [idx]})
                continue
            if current_texts and current_words + text_words > max_words:
                flush()
            current_texts.append(text)
            current_indices.append(idx)
            current_words += text_words
            if current_words >= target_words:
                flush()
        flush()
        return windows

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
            "relation_grounding_mode_counts": {},
            "initial_grounded_entity_count": 0,
            "relation_brought_entity_candidate_count": 0,
            "relation_brought_grounded_entity_count": 0,
            "resolved_entity_count": 0,
            "pruned_low_value_entity_count": 0,
            "coverage_repair_triggered_chunks": 0,
            "external_entity_raw_count": 0,
            "external_entity_typed_count": 0,
            "external_entity_open_candidate_count": 0,
            "metadata_ner_typed_count": 0,
            "ner_auxiliary_candidate_count": 0,
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

            out = self.run_chunk(
                cleaned_text=cleaned,
                rid_namespace=f"{document_rid_namespace}.chunk{i}",
                metadata=ch.get("metadata") or {},
            )
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
            "relation_grounding_mode_counts": {},
            "initial_grounded_entity_count": 0,
            "relation_brought_entity_candidate_count": 0,
            "relation_brought_grounded_entity_count": 0,
            "resolved_entity_count": 0,
            "pruned_low_value_entity_count": 0,
            "coverage_repair_triggered_chunks": 0,
            "external_entity_raw_count": 0,
            "external_entity_typed_count": 0,
            "external_entity_open_candidate_count": 0,
            "metadata_ner_typed_count": 0,
            "ner_auxiliary_candidate_count": 0,
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

            pass1 = self._collect_chunk_pass1(cleaned_text=cleaned, metadata=ch.get("metadata") or {})
            pass1["rid_namespace"] = f"{document_rid_namespace}.chunk{i}"
            chunk_states.append(pass1)
            doc_cleaned_parts.append(cleaned)
            doc_typed_entities.extend(pass1["typed_frame_entities"])
            doc_typed_entities.extend(pass1["metadata_typed_entities"])
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
            grounded_open_entities = self._ground_open_candidates(
                text=document_text,
                candidates=doc_grounding_candidates,
                source_kind="open_grounded",
            )
        stats["entity_grounding_sec"] += time.perf_counter() - entity_grounding_start

        character_mentions = Counter(
            _norm_name(item.get("name", ""))
            for item in (doc_typed_entities + grounded_open_entities)
            if isinstance(item, dict) and str(item.get("type", "")).strip() == "Character" and str(item.get("name", "")).strip()
        )
        doc_grounded_entities = _normalize_character_entities(
            _dedup_grounded_entities(doc_typed_entities + grounded_open_entities),
            mention_counts=character_mentions,
        )
        stats["initial_grounded_entity_count"] = len(doc_grounded_entities)

        self.agent.config.knowledge_graph_builder.relation_extraction_mode = "open_then_ground"

        auto_relations_by_chunk: Dict[int, Dict[str, Dict[str, Any]]] = {}
        auto_relations_all: Dict[str, Dict[str, Any]] = {}
        for chunk_idx, chunk_state in enumerate(chunk_states):
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
                rid_prefix=f"{chunk_state['rid_namespace']}:auto",
            )
            stats["auto_relation_build_sec"] += time.perf_counter() - auto_relation_start
            auto_relations_by_chunk[chunk_idx] = auto_relations
            auto_relations_all.update(auto_relations)

        relation_windows = self._build_relation_windows(chunk_states)
        relation_brought_candidates_all: List[Dict[str, str]] = []
        relation_brought_rule_entity_total = 0
        open_relations_all: Dict[str, Dict[str, Any]] = {}
        resolved_doc_entities = list(doc_grounded_entities)

        for window_idx, window in enumerate(relation_windows, start=1):
            window_text = str(window.get("text", "") or "").strip()
            chunk_indices = [int(x) for x in (window.get("chunk_indices") or []) if isinstance(x, int) or str(x).isdigit()]
            if not window_text or not chunk_indices:
                continue
            window_entities = self._select_relevant_window_entities(
                cleaned_text=window_text,
                doc_grounded_entities=resolved_doc_entities,
                chunk_states=chunk_states,
                chunk_indices=chunk_indices,
            )
            open_relation_extract_start = time.perf_counter()
            open_relation_proposals: List[Dict[str, Any]] = []
            if _has_fast_open_relation_candidates(agent=self.agent, entities=window_entities):
                relation_hints = _build_fast_open_relation_hints(self.agent, entities=window_entities)
                raw_open_relations = self.agent.extractor.extract_open_relations(
                    text=window_text,
                    extracted_entities=self.agent._entities_text_for_extractor(window_entities),
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
                known_entity_names={_norm_name(item.get("name", "")) for item in resolved_doc_entities},
            )
            relation_brought_candidates_all.extend(relation_brought_candidates)
            rule_entities = _infer_relation_brought_rule_entities(
                relation_brought_candidates,
                scope_rules=getattr(self.agent, "scope_rules", {}) or {},
            )
            relation_brought_rule_entity_total += len(rule_entities)
            if rule_entities:
                character_mentions.update(
                    _norm_name(item.get("name", ""))
                    for item in rule_entities
                    if isinstance(item, dict)
                    and str(item.get("type", "")).strip() == "Character"
                    and str(item.get("name", "")).strip()
                )
                resolved_doc_entities = _normalize_character_entities(
                    _dedup_grounded_entities(resolved_doc_entities + rule_entities),
                    mention_counts=character_mentions,
                )

            proposal_names = self._proposal_entity_names(open_relation_proposals)
            proposal_names.update(_norm_name(item.get("name", "")) for item in rule_entities if isinstance(item, dict))
            window_entities = self._select_relevant_window_entities(
                cleaned_text=window_text,
                doc_grounded_entities=resolved_doc_entities,
                chunk_states=chunk_states,
                chunk_indices=chunk_indices,
                extra_names=proposal_names,
            )

            relation_grounding_start = time.perf_counter()
            open_relation_proposals = _canonicalize_open_relation_proposals(
                open_relation_proposals,
                entities=window_entities,
            )
            open_relation_proposals = _filter_fast_open_relation_proposals(
                open_relation_proposals,
                agent=self.agent,
                entities=window_entities,
            )
            seeded_relations = self.agent._prepare_grounded_open_relation_seeds(
                cleaned_text=window_text,
                proposals=copy.deepcopy(open_relation_proposals),
                entities=copy.deepcopy(window_entities),
                rid_prefix=f"{document_rid_namespace}.window{window_idx}:event_first.open_rel",
                memory_context="",
            )
            seeded_relations = apply_canonicalization_to_relations(seeded_relations, build_name_canonicalizer(window_entities))
            seeded_relations = self.agent._filter_illegal_is_a_relations(seeded_relations, entities=window_entities)

            open_rel_map: Dict[str, Dict[str, Any]] = {}
            open_feedbacks: Dict[str, List[Dict[str, Any]]] = {}
            for rel in seeded_relations:
                rid = str(rel.get("rid", "")).strip()
                if not rid:
                    continue
                rel2, feedback = self.agent._revalidate_one_relation_global(rel, entities=window_entities)
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
            stats["relation_grounding_mode_counts"] = _merge_count_maps(
                stats.get("relation_grounding_mode_counts") or {},
                _grounding_mode_counts(list(open_rel_map.values())),
            )
            stats["relation_grounding_sec"] += time.perf_counter() - relation_grounding_start

            resolve_errors_start = time.perf_counter()
            resolved_entities, fixed_open_rel_map = self.agent._resolve_errors(
                entities=copy.deepcopy(window_entities),
                all_relations=copy.deepcopy(open_rel_map),
                all_feedbacks=copy.deepcopy(open_feedbacks),
                content=window_text,
            )
            stats["resolve_errors_sec"] += time.perf_counter() - resolve_errors_start
            resolved_entities = _normalize_character_entities(
                _dedup_grounded_entities(resolved_entities),
                mention_counts=character_mentions,
            )
            resolved_doc_entities = _normalize_character_entities(
                _dedup_grounded_entities(resolved_doc_entities + resolved_entities),
                mention_counts=character_mentions,
            )

            coverage_repair_triggered = 0
            coverage_probe_rel_map: Dict[str, Dict[str, Any]] = {}
            for idx in chunk_indices:
                coverage_probe_rel_map.update(auto_relations_by_chunk.get(idx, {}))
            coverage_probe_rel_map.update(fixed_open_rel_map)
            if _should_run_fast_coverage_repair(
                frames={},
                grounded_entities=resolved_entities,
                current_relations=coverage_probe_rel_map,
            ):
                coverage_repair_triggered = 1
            coverage_repair_start = time.perf_counter()
            if coverage_repair_triggered:
                repair_scope = {
                    str(ent.get("name", "")).strip()
                    for ent in resolved_entities
                    if isinstance(ent, dict)
                    and str(ent.get("name", "")).strip()
                    and str(ent.get("type", "")).strip() in {"Character", "Concept"}
                }
                fixed_open_rel_map = self.agent._repair_relation_coverage_by_rules(
                    cleaned_text=window_text,
                    entities=copy.deepcopy(resolved_entities),
                    all_relations={
                        rid: rel for rid, rel in copy.deepcopy(fixed_open_rel_map).items()
                        if str(rel.get("relation_type", "")).strip() in _FAST_OPEN_GROUNDED_RELATION_TYPES
                    },
                    rid_namespace=f"{document_rid_namespace}.window{window_idx}:event_first.coverage",
                    focus_entity_names=repair_scope,
                )
                coverage_probe_after_rules = dict(coverage_probe_rel_map)
                coverage_probe_after_rules.update(fixed_open_rel_map)
                if _should_run_fast_coverage_repair(
                    frames={},
                    grounded_entities=resolved_entities,
                    current_relations=coverage_probe_after_rules,
                ):
                    fixed_open_rel_map = self.agent._repair_relation_coverage(
                        cleaned_text=window_text,
                        entities=copy.deepcopy(resolved_entities),
                        all_relations={
                            rid: rel for rid, rel in copy.deepcopy(fixed_open_rel_map).items()
                            if str(rel.get("relation_type", "")).strip() in _FAST_OPEN_GROUNDED_RELATION_TYPES
                        },
                        rid_namespace=f"{document_rid_namespace}.window{window_idx}:event_first.coverage",
                        memory_context="",
                    )
            stats["coverage_repair_sec"] += time.perf_counter() - coverage_repair_start
            open_relations_all.update(fixed_open_rel_map)
            stats["coverage_repair_triggered_chunks"] += coverage_repair_triggered

        relation_brought_candidates_all = _dedup_open_entities(relation_brought_candidates_all)
        stats["relation_brought_entity_candidate_count"] = len(relation_brought_candidates_all)
        stats["relation_brought_grounded_entity_count"] = relation_brought_rule_entity_total

        all_event_nodes: List[Dict[str, Any]] = []
        all_occasion_nodes: List[Dict[str, Any]] = []
        for chunk_state in chunk_states:
            all_event_nodes.extend(chunk_state.get("event_nodes") or [])
            all_occasion_nodes.extend(chunk_state.get("occasion_nodes") or [])

        combined_rel_map = dict(auto_relations_all)
        combined_rel_map.update(open_relations_all)
        pruned_entities, pruned_rel_map, dropped_counts = _prune_low_value_entities_and_relations(
            entities=all_event_nodes + all_occasion_nodes + resolved_doc_entities,
            relations=combined_rel_map,
        )
        entities_all = _merge_entities(entities_all, pruned_entities)
        for rid, rel in pruned_rel_map.items():
            all_relations[rid] = rel

        stats["resolved_entity_count"] += sum(
            1 for ent in pruned_entities if str(ent.get("type", "")).strip() not in {"Event", "Occasion"}
        )
        stats["pruned_low_value_entity_count"] += sum(dropped_counts.values())
        stats["open_relation_count_after_fix"] += sum(
            1
            for rid in pruned_rel_map
            if ":event_first.open_rel" in rid or ":event_first.coverage" in rid or ":coverage#" in rid
        )
        stats["auto_relation_count"] += sum(1 for rid in pruned_rel_map if _is_auto_relation_rid(rid))

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
            candidate_names = {_norm_name(name)}
            candidate_names.update(
                _norm_name(alias)
                for alias in (entity.get("aliases") or [])
                if str(alias or "").strip()
            )
            if any(norm in seed_names or norm in text_lower for norm in candidate_names if norm):
                selected.append(entity)
        return _dedup_grounded_entities(selected)

    def _select_relevant_window_entities(
        self,
        *,
        cleaned_text: str,
        doc_grounded_entities: List[Dict[str, Any]],
        chunk_states: List[Dict[str, Any]],
        chunk_indices: List[int],
        extra_names: set[str] | None = None,
    ) -> List[Dict[str, Any]]:
        text_lower = cleaned_text.lower()
        seed_names = set(extra_names or set())
        for idx in chunk_indices or []:
            if not (0 <= int(idx) < len(chunk_states)):
                continue
            chunk_state = chunk_states[int(idx)] or {}
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
            candidate_names = {_norm_name(name)}
            candidate_names.update(
                _norm_name(alias)
                for alias in (entity.get("aliases") or [])
                if str(alias or "").strip()
            )
            if any(norm in seed_names or norm in text_lower for norm in candidate_names if norm):
                selected.append(entity)
        return _dedup_grounded_entities(selected)

    def _collect_chunk_pass1(self, *, cleaned_text: str, metadata: Dict[str, Any] | None = None) -> Dict[str, Any]:
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
        metadata_typed_entities = _typed_entities_from_metadata_ner(
            metadata,
            known_names={_norm_name(item.get("name", "")) for item in (event_nodes + occasion_nodes + typed_frame_entities)},
            scope_rules=getattr(self.agent, "scope_rules", {}) or {},
        )
        related_entity_candidates = _collect_related_entity_candidates(frames, frame_node_names=frame_node_names)

        kg_cfg = getattr(self.config, "knowledge_graph_builder", None)
        doc_cfg = getattr(self.config, "document_processing", None)
        entity_mode = str(getattr(kg_cfg, "entity_extraction_mode", "llm") or "llm").strip().lower()
        metadata_entity_mode = str(getattr(doc_cfg, "metadata_entity_mode", "llm") or "llm").strip().lower()
        reuse_metadata_ner = entity_mode == "ner" and metadata_entity_mode == "ner"
        ner_auxiliary_only = bool(getattr(kg_cfg, "ner_auxiliary_only", True))
        ner_skip_open_entity_llm = bool(getattr(kg_cfg, "ner_skip_open_entity_llm", False))

        external_entity_start = time.perf_counter()
        if reuse_metadata_ner:
            external_entity_payload = {
                "typed_entities": [],
                "open_candidates": [],
                "stats": {
                    "backend": "metadata_reuse",
                    "raw_count": 0,
                    "typed_count": 0,
                    "open_count": 0,
                },
            }
        else:
            external_entity_payload = self.external_entity_extractor.extract(
                text=cleaned_text,
                known_names=[item.get("name", "") for item in (event_nodes + occasion_nodes + typed_frame_entities + metadata_typed_entities)],
                scope_rules=getattr(self.agent, "scope_rules", {}) or {},
            )
        external_entity_sec = time.perf_counter() - external_entity_start
        external_typed_entities = _dedup_typed_entities(external_entity_payload.get("typed_entities") or [])
        external_open_candidates = _dedup_open_entities(external_entity_payload.get("open_candidates") or [])
        external_stats = external_entity_payload.get("stats") or {}
        ner_auxiliary_candidates: List[Dict[str, str]] = []
        if entity_mode == "ner" and ner_auxiliary_only:
            ner_auxiliary_candidates = _dedup_open_entities(
                _typed_entities_to_open_candidates(metadata_typed_entities, source_hint="metadata NER")
                + _typed_entities_to_open_candidates(external_typed_entities, source_hint="external NER")
            )
            metadata_typed_entities = []
            external_typed_entities = []

        skip_llm_min_typed = int(getattr(kg_cfg, "ner_skip_llm_min_typed", 3) or 3)
        open_entity_llm_skipped = 0
        open_entity_start = time.perf_counter()
        open_entities: List[Dict[str, Any]] = []
        should_skip_open_entity_llm = (
            ner_skip_open_entity_llm
            and (
                reuse_metadata_ner
                or (len(metadata_typed_entities) + len(external_typed_entities)) >= skip_llm_min_typed
            )
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
                                typed_frame_entities + metadata_typed_entities + external_typed_entities,
                                _merge_open_entity_candidates(
                                    related_entity_candidates,
                                    external_open_candidates,
                                    ner_auxiliary_candidates,
                                ),
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
            ner_auxiliary_candidates,
            open_entities or [],
        )

        return {
            "cleaned_text": cleaned_text,
            "frames": frames,
            "event_nodes": event_nodes,
            "occasion_nodes": occasion_nodes,
            "typed_frame_entities": typed_frame_entities,
            "metadata_typed_entities": metadata_typed_entities,
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
                "metadata_ner_typed_count": len(metadata_typed_entities),
                "ner_auxiliary_candidate_count": len(ner_auxiliary_candidates),
                "open_entity_llm_skipped_chunks": open_entity_llm_skipped,
            },
        }

    def run_chunk(self, *, cleaned_text: str, rid_namespace: str, metadata: Dict[str, Any] | None = None) -> Dict[str, Any]:
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
        metadata_typed_entities = _typed_entities_from_metadata_ner(
            metadata,
            known_names={_norm_name(item.get("name", "")) for item in (event_nodes + occasion_nodes + typed_frame_entities)},
            scope_rules=getattr(self.agent, "scope_rules", {}) or {},
        )
        related_entity_candidates = _collect_related_entity_candidates(frames, frame_node_names=frame_node_names)

        kg_cfg = getattr(self.config, "knowledge_graph_builder", None)
        doc_cfg = getattr(self.config, "document_processing", None)
        entity_mode = str(getattr(kg_cfg, "entity_extraction_mode", "llm") or "llm").strip().lower()
        metadata_entity_mode = str(getattr(doc_cfg, "metadata_entity_mode", "llm") or "llm").strip().lower()
        reuse_metadata_ner = entity_mode == "ner" and metadata_entity_mode == "ner"
        ner_auxiliary_only = bool(getattr(kg_cfg, "ner_auxiliary_only", True))
        ner_skip_open_entity_llm = bool(getattr(kg_cfg, "ner_skip_open_entity_llm", False))

        external_entity_start = time.perf_counter()
        if reuse_metadata_ner:
            external_entity_payload = {
                "typed_entities": [],
                "open_candidates": [],
                "stats": {
                    "backend": "metadata_reuse",
                    "raw_count": 0,
                    "typed_count": 0,
                    "open_count": 0,
                },
            }
        else:
            external_entity_payload = self.external_entity_extractor.extract(
                text=cleaned_text,
                known_names=[item.get("name", "") for item in (event_nodes + occasion_nodes + typed_frame_entities + metadata_typed_entities)],
                scope_rules=getattr(self.agent, "scope_rules", {}) or {},
            )
        external_entity_sec = time.perf_counter() - external_entity_start
        external_typed_entities = _dedup_typed_entities(external_entity_payload.get("typed_entities") or [])
        external_open_candidates = _dedup_open_entities(external_entity_payload.get("open_candidates") or [])
        external_stats = external_entity_payload.get("stats") or {}
        ner_auxiliary_candidates: List[Dict[str, str]] = []
        if entity_mode == "ner" and ner_auxiliary_only:
            ner_auxiliary_candidates = _dedup_open_entities(
                _typed_entities_to_open_candidates(metadata_typed_entities, source_hint="metadata NER")
                + _typed_entities_to_open_candidates(external_typed_entities, source_hint="external NER")
            )
            metadata_typed_entities = []
            external_typed_entities = []

        skip_llm_min_typed = int(getattr(kg_cfg, "ner_skip_llm_min_typed", 3) or 3)
        open_entity_llm_skipped = 0
        open_entity_start = time.perf_counter()
        open_entities: List[Dict[str, Any]] = []
        should_skip_open_entity_llm = (
            ner_skip_open_entity_llm
            and (
                reuse_metadata_ner
                or (len(metadata_typed_entities) + len(external_typed_entities)) >= skip_llm_min_typed
            )
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
                                typed_frame_entities + metadata_typed_entities + external_typed_entities,
                                _merge_open_entity_candidates(
                                    related_entity_candidates,
                                    external_open_candidates,
                                    ner_auxiliary_candidates,
                                ),
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
            ner_auxiliary_candidates,
            open_entities or [],
        )
        entity_grounding_start = time.perf_counter()
        grounded_open_entities = self._ground_open_candidates(
            text=cleaned_text,
            candidates=grounding_candidates,
            source_kind="open_grounded",
        )
        entity_grounding_sec = time.perf_counter() - entity_grounding_start

        character_mentions = Counter(
            _norm_name(item.get("name", ""))
            for item in (typed_frame_entities + metadata_typed_entities + external_typed_entities + grounded_open_entities)
            if isinstance(item, dict) and str(item.get("type", "")).strip() == "Character" and str(item.get("name", "")).strip()
        )
        grounded_entities = _normalize_character_entities(
            _dedup_grounded_entities(typed_frame_entities + metadata_typed_entities + external_typed_entities + grounded_open_entities),
            mention_counts=character_mentions,
        )
        initial_grounded_entity_count = len(grounded_entities)

        auto_relation_start = time.perf_counter()
        auto_relations = _auto_relations_from_frames(
            frames=frames,
            grounded_entities=grounded_entities,
            event_nodes=event_nodes,
            occasion_nodes=occasion_nodes,
            rid_prefix=f"{rid_namespace}:auto",
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
        relation_brought_candidates_for_grounding = _filter_relation_brought_grounding_candidates(
            relation_brought_candidates
        )

        relation_brought_entity_grounding_start = time.perf_counter()
        relation_brought_grounded_entities: List[Dict[str, Any]] = []
        if relation_brought_candidates_for_grounding:
            relation_brought_grounded_entities = self._ground_open_candidates(
                text=cleaned_text,
                candidates=relation_brought_candidates_for_grounding,
                source_kind="relation_brought_grounded",
            )
            character_mentions.update(
                _norm_name(item.get("name", ""))
                for item in relation_brought_grounded_entities
                if isinstance(item, dict) and str(item.get("type", "")).strip() == "Character" and str(item.get("name", "")).strip()
            )
            grounded_entities = _normalize_character_entities(
                _dedup_grounded_entities(grounded_entities + relation_brought_grounded_entities),
                mention_counts=character_mentions,
            )
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
        relation_grounding_mode_counts = _grounding_mode_counts(list(open_rel_map.values()))

        resolve_errors_start = time.perf_counter()
        resolved_entities, fixed_open_rel_map = self.agent._resolve_errors(
            entities=copy.deepcopy(grounded_entities),
            all_relations=copy.deepcopy(open_rel_map),
            all_feedbacks=copy.deepcopy(open_feedbacks),
            content=cleaned_text,
        )
        resolve_errors_sec = time.perf_counter() - resolve_errors_start
        resolved_entities = _normalize_character_entities(
            _dedup_grounded_entities(resolved_entities),
            mention_counts=character_mentions,
        )

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
                "auto_relation_count": sum(1 for rid in pruned_rel_map if _is_auto_relation_rid(rid)),
                "open_relation_proposal_count": len(open_relation_proposals),
                "open_relation_count_after_fix": sum(
                    1
                    for rid in pruned_rel_map
                    if ":event_first.open_rel" in rid or ":event_first.coverage" in rid or ":coverage#" in rid
                ),
                "relation_grounding_mode_counts": relation_grounding_mode_counts,
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
                "metadata_ner_typed_count": len(metadata_typed_entities),
                "ner_auxiliary_candidate_count": len(ner_auxiliary_candidates),
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

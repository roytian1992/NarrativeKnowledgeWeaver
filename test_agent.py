# %%
"""
Sequential narrative KG extraction over ordered chunks.

Goal:
- desc_chunks["0"] is a list of ordered text blocks within one chapter.
- Process blocks in order.
- Carry forward the latest entity_extraction and all_relations into the next block:
  - feed them into extractor as extracted_entities / extracted_relations (text or json).

This script:
1) Loads config + LLM + extractor
2) Loads entity/relation schema
3) Defines an InformationExtractionAgent that:
   - cleans text
   - extracts entities (event, time_and_location, general) with previous entities injected
   - canonicalizes entity names (global vs local)
   - extracts relations by schema group, injecting previously extracted relations
   - validates relations and accumulates feedback
   - resolves errors (relation fixes, entity fixes) using your extractor tools
   - deduplicates multiple relations between same pair
4) Runs sequentially on desc_chunks["0"] list, carrying state forward

Notes:
- You may need to adjust the extractor method names if yours differ.
- This is designed to be readable and robust, not micro-optimized.
"""

import json
import re
from collections import defaultdict
from typing import Any, Dict, List, Tuple, Iterable, Set, Optional

from core.model_providers.openai_llm import OpenAILLM
from core import KAGConfig
from core.builder.manager.information_manager import InformationExtractor


# -----------------------------
# Text cleaning
# -----------------------------
_whitespace_re = re.compile(r"\s+")

def clean_screenplay_text(content: str) -> str:
    s = content or ""

    # 1) word-\nword -> wordword
    s = re.sub(r"([A-Za-z])-\s*\n\s*([A-Za-z])", r"\1\2", s)

    # 2) word- word (aggressive, enable if your data has lots of this)
    s = re.sub(r"([A-Za-z])-\s+([a-z])", r"\1\2", s)

    # 3) compress whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s


# -----------------------------
# Canonicalization utilities
# -----------------------------
def normalize_name_basic(name: str) -> str:
    if name is None:
        return ""
    name = str(name)
    name = _whitespace_re.sub(" ", name).strip()
    return name

def titlecase_global_name(name: str) -> str:
    name = normalize_name_basic(name)
    return name.title()

def canonicalize_entity_name(name: str, scope: str) -> str:
    name = normalize_name_basic(name)
    if (scope or "local") == "global":
        name = titlecase_global_name(name)
    return name

def apply_canonicalization_to_entities(entities: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for ent in entities or []:
        e = dict(ent)
        e["name"] = canonicalize_entity_name(e.get("name", ""), e.get("scope", "local"))
        out.append(e)
    return out

def build_name_canonicalizer(entities: Iterable[Dict[str, Any]]) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    raw2canon:
      - key is normalized raw
      - value is canonical name
    This is more robust than keying on exact raw string.
    """
    raw2canon: Dict[str, str] = {}
    canon2raw: Dict[str, str] = {}
    for ent in entities or []:
        raw = ent.get("name", "")
        scope = ent.get("scope", "local")
        raw_norm = normalize_name_basic(raw)
        canon = canonicalize_entity_name(raw, scope)
        raw2canon[raw_norm] = canon
        canon2raw[canon] = raw
    return raw2canon, canon2raw

def apply_canonicalization_to_relations(relations: Iterable[Dict[str, Any]], raw2canon: Dict[str, str]) -> List[Dict[str, Any]]:
    out = []
    for rel in relations or []:
        r = dict(rel)
        subj_raw = normalize_name_basic(r.get("subject", ""))
        obj_raw = normalize_name_basic(r.get("object", ""))
        r["subject"] = raw2canon.get(subj_raw, subj_raw)
        r["object"] = raw2canon.get(obj_raw, obj_raw)
        out.append(r)
    return out


# -----------------------------
# Schema text helpers
# -----------------------------
def generate_schema_text(group: List[Dict[str, Any]]) -> str:
    lines = []
    for rel in group:
        if rel.get("direction") == "directed":
            symbol = "-->"
        else:
            symbol = "<-->"
        rule = "/".join(rel.get("from", [])) + symbol + "/".join(rel.get("to", []))
        line = f"{rel['type']}: {rel['description']}         constraint: {rule}"
        lines.append(line)
    return "\n".join(lines)

def prepare_few_shot_examples(group: List[Dict[str, Any]]) -> str:
    samples = []
    for rel in group:
        samples.extend(rel.get("samples", []))
    return json.dumps(samples, ensure_ascii=False, indent=2)


# -----------------------------
# Relation typepair index for constraint fixes
# -----------------------------
def build_typepair_relation_index(global_rule_finder: Dict[str, Dict[str, Any]]) -> Dict[Tuple[str, str], Set[str]]:
    """
    (type_a, type_b) (unordered) -> {relation_type, ...}
    """
    index: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
    for relation_type, rule in global_rule_finder.items():
        from_types = rule.get("from", []) or []
        to_types = rule.get("to", []) or []
        for t1 in from_types:
            for t2 in to_types:
                key = tuple(sorted((t1, t2)))
                index[key].add(relation_type)
    return dict(index)

def get_allowed_relations_between_types(
    typepair_index: Dict[Tuple[str, str], Set[str]],
    subject_type: str,
    object_type: str,
) -> List[str]:
    key = tuple(sorted((subject_type, object_type)))
    allowed = typepair_index.get(key, set())
    return sorted(list(allowed))


# -----------------------------
# Agent
# -----------------------------
class InformationExtractionAgent:
    def __init__(
        self,
        *,
        extractor: InformationExtractor,
        relation_schema: Dict[str, List[Dict[str, Any]]],
        entity_schema: List[Dict[str, Any]],
        enable_aggressive_text_fix: bool = True,
    ):
        self.extractor = extractor
        self.relation_schema = relation_schema

        # entity schema descriptions for entity type fix prompt
        self.entity_type_description = {ent["type"]: ent.get("description", "") for ent in (entity_schema or [])}

        # relation type info (global)
        self.relation_type_info: Dict[str, Dict[str, Any]] = {}
        for group in relation_schema.values():
            for r in group:
                rtype = r["type"]
                if rtype in self.relation_type_info:
                    raise ValueError(f"Duplicate relation_type detected across groups: {rtype}")
                self.relation_type_info[rtype] = {
                    "from": r.get("from", []),
                    "to": r.get("to", []),
                    "direction": r.get("direction", "directed"),
                    "description": r.get("description", ""),
                }

        self.typepair_index = build_typepair_relation_index(self.relation_type_info)

        self.enable_aggressive_text_fix = enable_aggressive_text_fix

    # -------------------------
    # Entity extraction
    # -------------------------
    def _entities_text_for_extractor(self, entity_extraction: List[Dict[str, Any]]) -> str:
        lines = []
        for ent in entity_extraction or []:
            name = ent.get("name", "")
            etype = ent.get("type", "")
            if not name or not etype:
                continue
            lines.append(f"entity name: {name}        entity type {etype}")
        return "\n".join(lines)

    def extract_entities_for_chunk(
        self,
        *,
        cleaned_text: str,
        previous_entities: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        entity_extraction: List[Dict[str, Any]] = list(previous_entities or [])

        for entity_type in ["event", "time_and_location", "general"]:
            existing_txt = self._entities_text_for_extractor(entity_extraction)
            raw = self.extractor.extract_entities_enhanced(
                text=cleaned_text,
                entity_type=entity_type,
                extracted_entities=existing_txt,
            )
            result = json.loads(raw)

            if entity_type.lower() == "event":
                for res in result:
                    if res.get("type") == "Event":
                        res["scope"] = "local"
                    elif res.get("type") == "Occasion":
                        res["scope"] = "global"
                    else:
                        # ignore unexpected
                        continue

            entity_extraction.extend(result)

        # Canonicalize entity names
        entity_extraction = apply_canonicalization_to_entities(entity_extraction)

        # Optional: dedup exact same (name,type) pairs
        seen = set()
        deduped = []
        for e in entity_extraction:
            k = (e.get("name", ""), e.get("type", ""))
            if k in seen:
                continue
            seen.add(k)
            deduped.append(e)

        return deduped

    # -------------------------
    # Relation extraction helpers
    # -------------------------
    def _build_entities_text_for_group(self, entity_extraction: List[Dict[str, Any]], group: List[Dict[str, Any]]) -> str:
        entity_type_selector = {t for rel in group for t in (rel.get("from", []) + rel.get("to", []))}
        lines = []
        for ent in entity_extraction or []:
            if ent.get("type") in entity_type_selector:
                lines.append(f"entity_name: {ent.get('name','')}        type {ent.get('type','')}")
        return "\n".join(lines)

    def _assign_rids(self, relations: List[Dict[str, Any]], rid_prefix: str) -> None:
        for i, rel in enumerate(relations, start=1):
            rel["rid"] = f"{rid_prefix}#{i}"

    def _filter_illegal_is_a_relations(
        self,
        relations: List[Dict[str, Any]],
        *,
        entity_extraction: List[Dict[str, Any]],
        entity_type_names: Optional[Set[str]] = None,
    ) -> List[Dict[str, Any]]:
        if not relations:
            return relations
        if entity_type_names is None:
            entity_type_names = {"Event", "Occasion", "Location", "TimePoint", "Character", "Object", "Concept", "Action"}

        type_name_norm = {t.strip().lower() for t in entity_type_names}
        entity_names = {normalize_name_basic(e.get("name", "")) for e in (entity_extraction or []) if e.get("name")}
        entity_names_norm = {n.lower() for n in entity_names}

        kept = []
        for rel in relations:
            rtype = (rel.get("relation_type") or rel.get("type") or "").strip()
            if rtype != "is_a":
                kept.append(rel)
                continue

            subj = normalize_name_basic(rel.get("subject", ""))
            obj = normalize_name_basic(rel.get("object", ""))

            if subj.lower() not in entity_names_norm:
                continue
            if subj.lower() in type_name_norm:
                continue
            if obj.lower() in type_name_norm:
                continue

            kept.append(rel)
        return kept

    def _validate_and_fix_relations(
        self,
        relations: List[Dict[str, Any]],
        *,
        entity_name2type: Dict[str, str],
        entity_name_selector: Set[str],
        group: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
        relation_type_selector = {r["type"] for r in group}
        rule_finder = {r["type"]: {"from": r.get("from", []), "to": r.get("to", []), "direction": r.get("direction")} for r in group}

        feedbacks: Dict[str, List[Dict[str, Any]]] = {}

        def add_feedback(
            key: str,
            rid: str,
            feedback: str,
            subj: str,
            obj: str,
            rtype: str,
        ) -> None:
            payload = {
                "rid": rid,
                "feedback": feedback,
                "subject": subj,
                "object": obj,
                "relation_type": rtype,
            }
            feedbacks.setdefault(key, []).append(payload)

        for rel in relations:
            rid = rel.get("rid", "")
            subj = rel.get("subject", "")
            obj = rel.get("object", "")
            rtype = rel.get("relation_type", "")
            rname = rel.get("relation_name", "")

            if subj not in entity_name_selector:
                add_feedback(
                    "subject error",
                    rid,
                    f"Subject [{subj}] is not previously extracted. Consider adding the entity or dropping the relation.",
                    subj,
                    obj,
                    rtype,
                )
                rel["conf"] = 0.0
                continue

            if obj not in entity_name_selector:
                add_feedback(
                    "object error",
                    rid,
                    f"Object [{obj}] is not previously extracted. Consider adding the entity or dropping the relation.",
                    subj,
                    obj,
                    rtype,
                )
                rel["conf"] = 0.0
                continue

            if rtype not in relation_type_selector:
                add_feedback(
                    "undefined relation error",
                    rid,
                    f"Relation type [{rtype}] (name [{rname}]) is not defined in schema. Consider modifying or dropping the relation.",
                    subj,
                    obj,
                    rtype,
                )
                rel["conf"] = 0.0
                continue

            subject_type = entity_name2type.get(subj)
            object_type = entity_name2type.get(obj)
            if subject_type is None:
                add_feedback("subject error", rid, f"Subject [{subj}] exists but its type is missing. Fix entity typing or drop the relation.", subj, obj, rtype)
                rel["conf"] = 0.0
                continue
            if object_type is None:
                add_feedback("object error", rid, f"Object [{obj}] exists but its type is missing. Fix entity typing or drop the relation.", subj, obj, rtype)
                rel["conf"] = 0.0
                continue

            rule = rule_finder.get(rtype)
            if not rule:
                add_feedback("missing rule error", rid, f"No rule defined for relation type [{rtype}].", subj, obj, rtype)
                rel["conf"] = 0.0
                continue

            direction = rule.get("direction")
            if direction == "symmetric":
                ok = ((subject_type in rule["from"] and object_type in rule["to"]) or (subject_type in rule["to"] and object_type in rule["from"]))
            else:
                ok = (subject_type in rule["from"] and object_type in rule["to"])

            if not ok:
                # minimal auto-fix: swap if directed and swapped satisfies
                if direction == "directed":
                    flipped_ok = (object_type in rule["from"] and subject_type in rule["to"])
                    if flipped_ok:
                        rel["subject"], rel["object"] = obj, subj
                        rel["auto_fixed"] = True
                        rel["fix_reason"] = "swap_subject_object_to_satisfy_type_constraint"
                        rel["conf"] = 0.5
                        continue

                add_feedback(
                    "constraint violation error",
                    rid,
                    f"Type constraint violated: subject type [{subject_type}] and object type [{object_type}] do not satisfy {rule['from']} -> {rule['to']} for the relation type [{rtype}].",
                    subj,
                    obj,
                    rtype,
                )
                rel["conf"] = 0.0
            else:
                rel["conf"] = 0.8

        return relations, feedbacks

    # -------------------------
    # Error fixing
    # -------------------------
    def _format_allowed_list(self, allowed: List[str], reverse_hint: bool) -> Tuple[Optional[str], str]:
        """
        Returns:
          allowed_list_text, action in {"select","reverse_select","drop"}
        """
        if allowed:
            lines = []
            for rel in allowed:
                info = self.relation_type_info.get(rel)
                if not info:
                    continue
                lines.append(f"{rel}: {info.get('description','')}")
            return "\n".join(lines), ("reverse_select" if reverse_hint else "select")
        return None, "drop"

    def _fix_relation_error(
        self,
        *,
        error: Dict[str, Any],
        content: str,
        all_relations: Dict[str, Dict[str, Any]],
        entity_name2type: Dict[str, str],
    ) -> Tuple[str, Dict[str, Any]]:
        rid = error["rid"]
        extracted_relation = all_relations[rid]
        feedback = error.get("feedback", "")

        subject = extracted_relation.get("subject", error.get("subject", ""))
        obj = extracted_relation.get("object", error.get("object", ""))

        subject_type = entity_name2type.get(subject, "")
        object_type = entity_name2type.get(obj, "")

        allowed_forward = get_allowed_relations_between_types(self.typepair_index, subject_type, object_type)
        allowed_backward = get_allowed_relations_between_types(self.typepair_index, object_type, subject_type)

        if allowed_forward:
            allowed_list, action = self._format_allowed_list(allowed_forward, reverse_hint=False)
        elif allowed_backward:
            allowed_list, action = self._format_allowed_list(allowed_backward, reverse_hint=True)
            allowed_list = (allowed_list or "") + "\nPlease consider reversing the subject and object to match the allowed relation types above."
        else:
            return "drop", {}

        raw = self.extractor.fix_relation_error(
            text=content,
            extracted_relation=json.dumps(extracted_relation, ensure_ascii=False, indent=2),
            allowed_relation_types=allowed_list,
            feedback=feedback,
        )
        out = json.loads(raw)
        decision = out.get("decision", "drop")
        output = out.get("output", {}) or {}

        if action == "select":
            return decision, output

        # reverse_select
        if decision == "rewrite":
            output = dict(output)
            output["subject"] = obj
            output["object"] = subject
            return decision, output

        return "drop", {}

    def _fix_entity_error(
        self,
        *,
        error: Dict[str, Any],
        error_type: str,
        content: str,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Assumption:
          extractor.fix_entity_error returns:
            {"decision": "add"|"drop", "output": { "name": "...", "type": "..." , ... }}
        If your tool returns a different format, adjust here.
        """
        feedback = error.get("feedback", "")
        relation_type = error.get("relation_type", "")

        if relation_type not in self.relation_type_info:
            return "drop", {}

        if error_type == "subject error":
            candidate_list = self.relation_type_info[relation_type]["from"]
        else:
            candidate_list = self.relation_type_info[relation_type]["to"]

        # Build descriptions text
        desc_lines = []
        for t in candidate_list:
            desc_lines.append(f"{t}: {self.entity_type_description.get(t,'')}")
        entity_type_description_text = "\n".join(desc_lines)

        raw = self.extractor.fix_entity_error(
            text=content,
            candidate_entity_types=", ".join(candidate_list),
            candidate_entity_descriptions=entity_type_description_text,
            feedback=feedback,
        )
        out = json.loads(raw)
        decision = out.get("decision", "drop")
        output = out.get("output", {}) or {}
        return decision, output

    def _resolve_errors(
        self,
        *,
        entity_extraction: List[Dict[str, Any]],
        all_relations: Dict[str, Dict[str, Any]],
        all_feedbacks: Dict[str, List[Dict[str, Any]]],
        content: str,
        entity_name2type: Dict[str, str],
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
        # iterate snapshot of keys to allow mutation
        for error_type in list(all_feedbacks.keys()):
            error_list = all_feedbacks.get(error_type, [])
            if not error_list:
                continue

            idx = 0
            while idx < len(error_list):
                error = error_list[idx]
                rid = error.get("rid")

                if (not rid) or (rid not in all_relations):
                    error_list.pop(idx)
                    continue

                if error_type in ["undefined relation error", "missing rule error", "constraint violation error"]:
                    decision, output = self._fix_relation_error(
                        error=error,
                        content=content,
                        all_relations=all_relations,
                        entity_name2type=entity_name2type,
                    )
                    if decision == "drop":
                        all_relations.pop(rid, None)
                        error_list.pop(idx)
                        continue

                    output = dict(output)
                    output.setdefault("rid", rid)
                    all_relations[rid] = output
                    error_list.pop(idx)
                    continue

                if error_type in ["subject error", "object error"]:
                    decision, output = self._fix_entity_error(
                        error=error,
                        error_type=error_type,
                        content=content,
                    )
                    if decision == "drop":
                        all_relations.pop(rid, None)
                        error_list.pop(idx)
                        continue

                    # Add entity and continue; then future chunks will use it
                    entity_extraction.append(output)
                    error_list.pop(idx)
                    continue

                # default: drop unknown error types
                all_relations.pop(rid, None)
                error_list.pop(idx)
                continue

        return entity_extraction, all_relations, all_feedbacks

    # -------------------------
    # Dedup relations
    # -------------------------
    def _dedup_multi_relations(
        self,
        *,
        all_relations: Dict[str, Dict[str, Any]],
        content: str,
    ) -> Dict[str, Dict[str, Any]]:
        buckets: Dict[Tuple[str, str], List[str]] = defaultdict(list)
        for rid, rel in all_relations.items():
            if not isinstance(rel, dict):
                continue
            s = normalize_name_basic(rel.get("subject", ""))
            o = normalize_name_basic(rel.get("object", ""))
            if not s or not o:
                continue
            buckets[(s, o)].append(rid)

        for pair, rids in list(buckets.items()):
            if len(rids) < 2:
                continue

            rels = []
            for rid in rids:
                rel = all_relations.get(rid)
                if isinstance(rel, dict):
                    rels.append({**rel, "rid": rid})
            if len(rels) < 2:
                continue

            # Rule 1: same type keep first
            seen_types = set()
            drop_rids_same_type = []
            kept_rels = []
            for rel in rels:
                rid = rel.get("rid")
                rtype = (rel.get("relation_type") or rel.get("type") or "").strip()
                if (not rid) or (not rtype):
                    kept_rels.append(rel)
                    continue
                if rtype in seen_types:
                    drop_rids_same_type.append(rid)
                else:
                    seen_types.add(rtype)
                    kept_rels.append(rel)

            for rid in drop_rids_same_type:
                all_relations.pop(rid, None)

            kept_rels = [r for r in kept_rels if r.get("rid") in all_relations]
            types_left = {(r.get("relation_type") or r.get("type") or "").strip() for r in kept_rels}
            types_left.discard("")
            if len(kept_rels) < 2 or len(types_left) < 2:
                continue

            # Rule 2: different types -> ask LLM
            relations_json = json.dumps(kept_rels, ensure_ascii=False, indent=2)
            raw = self.extractor.dedup_relations(text=content, relations=relations_json)

            try:
                out = json.loads(raw)
            except Exception:
                continue

            decision = (out.get("decision") or "").strip().lower()
            output = out.get("output") or {}
            if decision != "drop":
                continue

            drop_types = output.get("drop_relation_types") or []
            if not isinstance(drop_types, list) or not drop_types:
                continue

            drop_types = [t for t in drop_types if isinstance(t, str) and t.strip() in types_left]
            if not drop_types:
                continue

            drop_set = set(drop_types)
            for r in kept_rels:
                rid = r.get("rid")
                if not rid:
                    continue
                rtype = (r.get("relation_type") or r.get("type") or "").strip()
                if rtype in drop_set:
                    all_relations.pop(rid, None)

        return all_relations

    # -------------------------
    # One chunk step
    # -------------------------
    def run_one_chunk(
        self,
        *,
        chunk_text: str,
        prev_entity_extraction: List[Dict[str, Any]],
        prev_all_relations: Dict[str, Dict[str, Any]],
        chunk_rid_namespace: str,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
        """
        Process one chunk with previous state injected.
        Returns updated (entity_extraction, all_relations).
        """
        cleaned = clean_screenplay_text(chunk_text)

        # 1) Entities (carry forward)
        entity_extraction = self.extract_entities_for_chunk(
            cleaned_text=cleaned,
            previous_entities=prev_entity_extraction,
        )

        # Build canonicalizer and selectors
        raw2canon, _ = build_name_canonicalizer(entity_extraction)
        entity_name_selector = {e.get("name", "") for e in entity_extraction if e.get("name")}
        entity_name2type = {e.get("name", ""): e.get("type", "") for e in entity_extraction if e.get("name") and e.get("type")}

        # 2) Relations (carry forward)
        all_relations: Dict[str, Dict[str, Any]] = dict(prev_all_relations or {})

        # Provide extracted_relations context as list (without internal rid/conf noise if you want)
        def relations_context_list() -> List[Dict[str, Any]]:
            ctx = []
            for rel in all_relations.values():
                if not isinstance(rel, dict):
                    continue
                # keep only the core fields for context
                ctx.append(
                    {
                        "subject": rel.get("subject", ""),
                        "object": rel.get("object", ""),
                        "relation_type": rel.get("relation_type", rel.get("type", "")),
                        "relation_name": rel.get("relation_name", ""),
                        "description": rel.get("description", ""),
                    }
                )
            return ctx

        all_feedbacks: Dict[str, List[Dict[str, Any]]] = {}
        all_extracted_relations_for_llm: List[Dict[str, Any]] = relations_context_list()

        # Extract per relation group
        for relation_group, group in self.relation_schema.items():
            entities_text = self._build_entities_text_for_group(entity_extraction, group)
            relation_schema_text = generate_schema_text(group)
            few_shot_samples = prepare_few_shot_examples(group)

            raw = self.extractor.extract_relations_enhanced(
                text=cleaned,
                extracted_entities=entities_text,
                extracted_relations=json.dumps(all_extracted_relations_for_llm, ensure_ascii=False, indent=2) if all_extracted_relations_for_llm else "",
                relation_schema_text=relation_schema_text,
                relation_few_shot_examples=few_shot_samples,
                previous_results=None,
                feedbacks=None,
            )
            relation_extraction = json.loads(raw) if raw else []

            # extend LLM context for the next groups (as your code did)
            all_extracted_relations_for_llm.extend(relation_extraction)

            # filter illegal is_a
            relation_extraction = self._filter_illegal_is_a_relations(
                relation_extraction,
                entity_extraction=entity_extraction,
            )

            # canonicalize subj/obj and assign rids with global-unique namespace
            relation_extraction = apply_canonicalization_to_relations(relation_extraction, raw2canon)
            rid_prefix = f"{chunk_rid_namespace}:{relation_group}"
            self._assign_rids(relation_extraction, rid_prefix=rid_prefix)

            # validate + minimal auto-fix
            relation_extraction, feedbacks = self._validate_and_fix_relations(
                relation_extraction,
                entity_name2type=entity_name2type,
                entity_name_selector=entity_name_selector,
                group=group,
            )

            # accumulate relations
            for rel in relation_extraction or []:
                rid = rel.get("rid")
                if not rid:
                    raise ValueError(f"Missing rid in relation: {rel}")
                if rid in all_relations:
                    raise ValueError(f"Duplicate rid detected: {rid}")
                all_relations[rid] = rel

            # accumulate feedbacks
            for k, items in (feedbacks or {}).items():
                all_feedbacks.setdefault(k, [])
                if isinstance(items, list):
                    all_feedbacks[k].extend(items)
                else:
                    all_feedbacks[k].append(items)

        # 3) Resolve errors (may add entities, may rewrite/drop relations)
        entity_extraction, all_relations, all_feedbacks = self._resolve_errors(
            entity_extraction=entity_extraction,
            all_relations=all_relations,
            all_feedbacks=all_feedbacks,
            content=cleaned,
            entity_name2type=entity_name2type,
        )

        # Rebuild selectors because entities may have changed
        entity_extraction = apply_canonicalization_to_entities(entity_extraction)
        entity_name2type = {e.get("name", ""): e.get("type", "") for e in entity_extraction if e.get("name") and e.get("type")}

        # 4) Dedup relations within this chunk text (conservative)
        all_relations = self._dedup_multi_relations(
            all_relations=all_relations,
            content=cleaned,
        )

        return entity_extraction, all_relations

    # -------------------------
    # Full chapter run
    # -------------------------
    def run_chapter(
        self,
        *,
        ordered_chunks: List[Dict[str, Any]],
        init_entity_extraction: Optional[List[Dict[str, Any]]] = None,
        init_all_relations: Optional[Dict[str, Dict[str, Any]]] = None,
        chapter_id: str = "chapter0",
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
        """
        ordered_chunks: list of chunk dicts. Each should have 'content' (your doc2chunks.json style).
        """
        entity_extraction = list(init_entity_extraction or [])
        all_relations = dict(init_all_relations or {})

        for idx, chunk in enumerate(ordered_chunks):
            chunk_text = chunk.get("content", "")
            chunk_ns = f"{chapter_id}.chunk{idx}"
            entity_extraction, all_relations = self.run_one_chunk(
                chunk_text=chunk_text,
                prev_entity_extraction=entity_extraction,
                prev_all_relations=all_relations,
                chunk_rid_namespace=chunk_ns,
            )

        return entity_extraction, all_relations


# %%
if __name__ == "__main__":
    # 1) Load config + LLM + extractor
    config = KAGConfig.from_yaml("configs/config_openai.yaml")
    llm = OpenAILLM(config)
    extractor = InformationExtractor(config, llm, prompt_loader=None)

    # 2) Load data
    with open("data/knowledge_graph/doc2chunks.json", "r") as f:
        desc_chunks = json.load(f)

    # 3) Load schemas
    with open("../NarrativeWeaver/core/schema/default_relation_schema.json", "r") as f:
        relation_schema = json.load(f)

    with open("../NarrativeWeaver/core/schema/default_entity_schema.json", "r") as f:
        entity_schema = json.load(f)

    # 4) Build agent
    agent = InformationExtractionAgent(
        extractor=extractor,
        relation_schema=relation_schema,  # dict: group_name -> list[rules]
        entity_schema=entity_schema,      # list of entity type dicts
    )

    # 5) Run one chapter sequentially (desc_chunks["0"] is ordered list of blocks)
    chapter_chunks = desc_chunks["0"]  # list of chunk dicts
    final_entities, final_relations = agent.run_chapter(
        ordered_chunks=chapter_chunks,
        chapter_id="doc0_ch0",
    )

    # 6) Save outputs
    with open("out_entities_doc0_ch0.json", "w") as f:
        json.dump(final_entities, f, ensure_ascii=False, indent=2)

    with open("out_relations_doc0_ch0.json", "w") as f:
        # store as list, but keep rid for traceability
        json.dump(list(final_relations.values()), f, ensure_ascii=False, indent=2)

    print(f"Done. entities={len(final_entities)} relations={len(final_relations)}")

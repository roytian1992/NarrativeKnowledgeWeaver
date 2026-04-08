# core/agent/knowledge_extraction_agent.py
from __future__ import annotations

import json
import re
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union
from core.builder.manager.information_manager import InformationExtractor  # keep same dependency
from core.builder.manager.error_manager import ProblemSolver
from tqdm import tqdm

_WHITESPACE_RE = re.compile(r"\s+")
_WORD_HYPHEN_LINEBREAK_RE = re.compile(r"([A-Za-z])-\s*\n\s*([A-Za-z])")
_WORD_HYPHEN_SPACE_RE = re.compile(r"([A-Za-z])-\s+([a-z])")


def clean_screenplay_text(content: str, *, aggressive: bool = True) -> str:
    s = content or ""
    # word-\nword -> wordword
    s = _WORD_HYPHEN_LINEBREAK_RE.sub(r"\1\2", s)
    # bur- ied -> buried (aggressive)
    if aggressive:
        s = _WORD_HYPHEN_SPACE_RE.sub(r"\1\2", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_name_basic(name: str) -> str:
    if name is None:
        return ""
    s = str(name)
    s = _WHITESPACE_RE.sub(" ", s).strip()
    return s


def titlecase_global_name(name: str) -> str:
    return normalize_name_basic(name).title()


def canonicalize_entity_name(name: str, scope: str) -> str:
    n = normalize_name_basic(name)
    if (scope or "local") == "global":
        return titlecase_global_name(n)
    return n


def apply_canonicalization_to_entities(entities: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for ent in entities or []:
        e = dict(ent)
        e["name"] = canonicalize_entity_name(e.get("name", ""), e.get("scope", "local"))
        out.append(e)
    return out


def build_name_canonicalizer(entities: Iterable[Dict[str, Any]]) -> Dict[str, str]:
    """
    Map normalized raw name -> canonical name, based on current entity list.
    """
    raw2canon: Dict[str, str] = {}
    for ent in entities or []:
        raw = normalize_name_basic(ent.get("name", ""))
        scope = ent.get("scope", "local")
        raw2canon[raw] = canonicalize_entity_name(raw, scope)
    return raw2canon


def apply_canonicalization_to_relations(relations: Iterable[Dict[str, Any]], raw2canon: Dict[str, str]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for rel in relations or []:
        r = dict(rel)
        s = normalize_name_basic(r.get("subject", ""))
        o = normalize_name_basic(r.get("object", ""))
        r["subject"] = raw2canon.get(s, s)
        r["object"] = raw2canon.get(o, o)
        out.append(r)
    return out


def generate_schema_text(group: List[Dict[str, Any]]) -> str:
    lines = []
    for rel in group:
        direction = rel.get("direction", "directed")
        symbol = "-->" if direction == "directed" else "<-->"
        rule = "/".join(rel.get("from", [])) + symbol + "/".join(rel.get("to", []))
        lines.append(f"{rel['type']}: {rel.get('description','')}         constraint: {rule}")
    return "\n".join(lines)


def prepare_few_shot_examples(group: List[Dict[str, Any]]) -> str:
    samples: List[Dict[str, Any]] = []
    for rel in group:
        samples.extend(rel.get("samples", []))
    return json.dumps(samples, ensure_ascii=False, indent=2)


def build_typepair_relation_index(relation_type_info: Dict[str, Dict[str, Any]]) -> Dict[Tuple[str, str], Set[str]]:
    """
    Unordered (type_a, type_b) -> {relation_type}
    """
    idx: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
    for rtype, info in relation_type_info.items():
        from_types = info.get("from", []) or []
        to_types = info.get("to", []) or []
        for a in from_types:
            for b in to_types:
                idx[tuple(sorted((a, b)))].add(rtype)
    return dict(idx)


def get_allowed_relations_between_types(
    typepair_index: Dict[Tuple[str, str], Set[str]],
    t1: str,
    t2: str,
) -> List[str]:
    return sorted(list(typepair_index.get(tuple(sorted((t1, t2))), set())))


class InformationExtractionAgent:
    """
    document-sequential extraction agent.

    Inputs:
      - entity_schema: list[dict] or dict
      - relation_schema: dict[group_name -> list[relation_rule_dict]]

    Output (document-level):
      {
        "entities": list[dict],
        "relations": list[dict]
      }

    Notes:
    - The agent carries forward entity_extraction and all_relations across chunks in order.
    - It injects previous entities and previous relations into the extractor calls.
    """

    def __init__(
        self,
        config: Any,
        llm: Any,
        entity_schema: List[Dict[str, Any]],
        relation_schema: Dict[str, List[Dict[str, Any]]],
        category_priority: Optional[Dict[str, int]] = None,
        system_prompt_text: str = None,
        reflector: Any = None,
        memory_store: Any = None,
    ):

        self.config = config
        self.llm = llm
        self.system_prompt_text = system_prompt_text or ""
        self.reflector = reflector
        self.memory_store = memory_store  # Optional[ExtractionMemoryStore]

        self.extractor = InformationExtractor(config, llm)
        self.problem_solver = ProblemSolver(config, llm)

        self.category_priority = category_priority or {"induced": 0, "anchor": 1, "referential": 2, "general_semantic": 3}
        
        self.entity_schema_list = entity_schema

        self.scope_rules = {ent["type"]: ent["default_scope"] for ent in self.entity_schema_list if "default_scope" in ent and ent["default_scope"] is not None}
        
        self.entity_type_description = {
            t.get("type"): t.get("description", "")
            for t in (self.entity_schema_list or [])
            if isinstance(t, dict) and t.get("type")
        }

        self.relation_schema = relation_schema or {}

        # Build relation_type_info globally
        self.relation_type_info: Dict[str, Dict[str, Any]] = {}
        for group in self.relation_schema.values():
            for r in group:
                rtype = r["type"]
                if rtype in self.relation_type_info:
                    raise ValueError(f"Duplicate relation_type detected across groups: {rtype}")
                self.relation_type_info[rtype] = {
                    "from": r.get("from", []),
                    "to": r.get("to", []),
                    "direction": r.get("direction", "directed"),
                    "description": r.get("description", ""),
                    "persistence": r.get("persistence", None),
                    "allow_self_loop": bool(r.get("allow_self_loop", False)),
                }

        self.typepair_index = build_typepair_relation_index(self.relation_type_info)
        self._global_rule_finder: Dict[str, Dict[str, Any]] = {}
        for rtype, info in (self.relation_type_info or {}).items():
            self._global_rule_finder[rtype] = {
                "from": info.get("from", []) or [],
                "to": info.get("to", []) or [],
                "direction": info.get("direction", "directed"),
                "persistence": info.get("persistence", None),
                "description": info.get("description", ""),
                "allow_self_loop": bool(info.get("allow_self_loop", False)),
            }

    # -------------------------
    # public APIs
    # -------------------------
    def run_document(
        self,
        ordered_chunks: List[Dict[str, Any]],
        *,
        aggressive_clean: bool = True,
        document_rid_namespace: str = "document0",
    ) -> Dict[str, Any]:
        entities, relations = self._run_document_sequential(
            ordered_chunks=ordered_chunks,
            aggressive_clean=aggressive_clean,
            document_rid_namespace=document_rid_namespace,
        )
        return {"entities": entities, "relations": relations}

    async def arun_document(
        self,
        ordered_chunks: List[Dict[str, Any]],
        *,
        aggressive_clean: bool = True,
        document_rid_namespace: str = "document0",
    ) -> Dict[str, Any]:
        # current implementation is synchronous; keep signature compatible with builder
        return self.run_document(
            ordered_chunks,
            aggressive_clean=aggressive_clean,
            document_rid_namespace=document_rid_namespace,
        )

    # Backward compatible: single-chunk extraction
    def run(
        self,
        text: str,
        *,
        aggressive_clean: bool = True,
        rid_namespace: str = "chunk0",
    ) -> Dict[str, Any]:
        chunk = {"id": rid_namespace, "content": text, "metadata": {"order": 0}}
        return self.run_document([chunk], aggressive_clean=aggressive_clean, document_rid_namespace=rid_namespace)

    async def arun(
        self,
        text: str,
        *,
        aggressive_clean: bool = True,
        rid_namespace: str = "chunk0",
    ) -> Dict[str, Any]:
        return self.run(text, aggressive_clean=aggressive_clean, rid_namespace=rid_namespace)

    # -------------------------
    # core logic
    # -------------------------
    def _entities_text_for_extractor(self, entities: List[Dict[str, Any]]) -> str:
        lines = []
        for ent in entities or []:
            name = ent.get("name", "")
            etype = ent.get("type", "")
            if not name or not etype:
                continue
            lines.append(f"entity name: {name}        entity type {etype}")
        return "\n".join(lines)

    def _relations_text_for_extractor(self, all_relations: Dict[str, Dict[str, Any]]) -> str:
        # Use a light context schema to reduce token noise
        ctx = []
        for rel in all_relations.values():
            if not isinstance(rel, dict):
                continue
            ctx.append(
                {
                    "subject": rel.get("subject", ""),
                    "object": rel.get("object", ""),
                    "relation_type": rel.get("relation_type", rel.get("type", "")),
                    "relation_name": rel.get("relation_name", ""),
                    # "description": rel.get("description", ""),
                }
            )
        return json.dumps(ctx, ensure_ascii=False, indent=2) if ctx else ""

    def _build_entities_text_for_group(self, entities: List[Dict[str, Any]], group: List[Dict[str, Any]]) -> str:
        selector = {t for rel in group for t in (rel.get("from", []) + rel.get("to", []))}
        lines = []
        for ent in entities or []:
            if ent.get("type") in selector:
                lines.append(f"entity_name: {ent.get('name','')}        type {ent.get('type','')}")
        return "\n".join(lines)

    def _assign_rids(self, relations: List[Dict[str, Any]], rid_prefix: str) -> None:
        for i, rel in enumerate(relations, start=1):
            rel["rid"] = f"{rid_prefix}#{i}"

    def _filter_illegal_is_a_relations(
        self,
        relations: List[Dict[str, Any]],
        *,
        entities: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if not relations:
            return relations
        
        type_names = set(self.entity_type_description.keys())
        type_name_norm = {x.lower() for x in type_names}

        entity_names = {normalize_name_basic(e.get("name", "")) for e in (entities or []) if e.get("name")}
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

    def _revalidate_one_relation_global(
        self,
        rel: Dict[str, Any],
        *,
        entities: List[Dict[str, Any]],
    ) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        """
        Revalidate one relation using global schema (self._global_rule_finder).
        Returns (relation, feedback_or_none).
        If invalid, relation["conf"] will be 0.0 and feedback will be returned.
        """
        r = dict(rel or {})
        rid = r.get("rid", "")
        subj = r.get("subject", "") or ""
        obj = r.get("object", "") or ""
        rtype = (r.get("relation_type") or r.get("type") or "").strip()
        rname = r.get("relation_name", "") or ""

        entity_name_selector = {e.get("name", "") for e in (entities or []) if e.get("name")}
        entity_name2type = {e.get("name", ""): e.get("type", "") for e in (entities or []) if e.get("name") and e.get("type")}

        def _fb(key: str, msg: str) -> Dict[str, Any]:
            return {
                "rid": rid,
                "feedback": msg,
                "subject": subj,
                "object": obj,
                "relation_type": rtype,
                "relation_name": rname,
                "error_type": key,
            }

        # Subject/object exist
        if subj not in entity_name_selector:
            r["conf"] = 0.0
            return r, _fb("subject error", f"Subject [{subj}] is not previously extracted.")

        if obj not in entity_name_selector:
            r["conf"] = 0.0
            return r, _fb("object error", f"Object [{obj}] is not previously extracted.")

        # Relation type exists in schema
        rule = self._global_rule_finder.get(rtype)
        if not rule:
            r["conf"] = 0.0
            return r, _fb("undefined relation error", f"Relation type [{rtype}] (name [{rname}]) is not defined in schema.")

        st = entity_name2type.get(subj, "")
        ot = entity_name2type.get(obj, "")
        if not st:
            r["conf"] = 0.0
            return r, _fb("subject error", f"Subject [{subj}] exists but its type is missing.")
        if not ot:
            r["conf"] = 0.0
            return r, _fb("object error", f"Object [{obj}] exists but its type is missing.")

        direction = rule.get("direction", "directed")
        from_types = rule.get("from", []) or []
        to_types = rule.get("to", []) or []
        allow_self_loop = bool(rule.get("allow_self_loop", False))

        if subj == obj and not allow_self_loop:
            r["conf"] = 0.0
            return r, _fb(
                "self loop violation error",
                f"Self-loop relation is not allowed for relation type [{rtype}] with subject/object [{subj}].",
            )

        if direction == "symmetric":
            ok = ((st in from_types and ot in to_types) or (st in to_types and ot in from_types))
        else:
            ok = (st in from_types and ot in to_types)

        if not ok:
            # Try auto swap for directed
            if direction == "directed":
                flipped_ok = (ot in from_types and st in to_types)
                if flipped_ok:
                    r["subject"], r["object"] = obj, subj
                    r["auto_fixed"] = True
                    r["fix_reason"] = "swap_subject_object_to_satisfy_type_constraint"
                    r["conf"] = 0.5
                    persistence = rule.get("persistence")
                    if persistence:
                        r["persistence"] = persistence
                    return r, None

            r["conf"] = 0.0
            return r, _fb(
                "constraint violation error",
                f"Type constraint violated for relation [{rtype}] with subject type [{st}] and object type [{ot}].",
            )

        # Valid
        r["conf"] = float(r.get("conf", 0.8) or 0.8)
        persistence = rule.get("persistence")
        if persistence:
            r["persistence"] = persistence
        return r, None


    def _validate_and_fix_relations(
        self,
        relations: List[Dict[str, Any]],
        *,
        entities: List[Dict[str, Any]],
        group: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
        """
        Validate relations against:
        1) subject/object must exist in extracted entities
        2) relation_type must be defined in the current group schema
        3) subject/object types must satisfy schema constraints (direction-aware)
        4) if directed and reversed types are valid, auto-swap subject/object

        Guarantees:
        - Any relation with conf == 0.0 will have a corresponding feedback entry.
        """
        entity_name_selector = {e.get("name", "") for e in (entities or []) if e.get("name")}
        entity_name2type = {
            e.get("name", ""): e.get("type", "")
            for e in (entities or [])
            if e.get("name") and e.get("type")
        }

        relation_type_selector = {r["type"] for r in (group or []) if isinstance(r, dict) and r.get("type")}
        rule_finder: Dict[str, Dict[str, Any]] = {}
        for r in (group or []):
            if not isinstance(r, dict) or not r.get("type"):
                continue
            rule_finder[r["type"]] = {
                "from": r.get("from", []) or [],
                "to": r.get("to", []) or [],
                "direction": r.get("direction", "directed"),
                "persistence": r.get("persistence", None),
                "allow_self_loop": bool(r.get("allow_self_loop", False)),
            }

        feedbacks: Dict[str, List[Dict[str, Any]]] = {}

        def add_feedback(key: str, rid: str, feedback: str, subj: str, obj: str, rtype: str, rname: str) -> None:
            feedbacks.setdefault(key, []).append(
                {
                    "rid": rid,
                    "feedback": feedback,
                    "subject": subj,
                    "object": obj,
                    "relation_type": rtype,
                    "relation_name": rname,
                }
            )

        # Validate each relation
        for rel in relations or []:
            rid = rel.get("rid", "")
            subj = rel.get("subject", "") or ""
            obj = rel.get("object", "") or ""
            rtype = (rel.get("relation_type") or rel.get("type") or "").strip()
            rname = rel.get("relation_name", "") or ""

            # Default conf unless proven invalid
            if "conf" not in rel:
                rel["conf"] = 0.8

            # 1) subject exists
            if subj not in entity_name_selector:
                add_feedback("subject error", rid, f"Subject [{subj}] is not previously extracted.", subj, obj, rtype, rname)
                rel["conf"] = 0.0
                continue

            # 2) object exists
            if obj not in entity_name_selector:
                add_feedback("object error", rid, f"Object [{obj}] is not previously extracted.", subj, obj, rtype, rname)
                rel["conf"] = 0.0
                continue

            # 3) relation type defined in this group schema
            if rtype not in relation_type_selector:
                add_feedback(
                    "undefined relation error",
                    rid,
                    f"Relation type [{rtype}] (name [{rname}]) is not defined in schema.",
                    subj,
                    obj,
                    rtype,
                    rname,
                )
                rel["conf"] = 0.0
                continue

            # 4) subject/object types exist
            st = entity_name2type.get(subj, "")
            ot = entity_name2type.get(obj, "")
            if not st:
                add_feedback("subject error", rid, f"Subject [{subj}] exists but its type is missing.", subj, obj, rtype, rname)
                rel["conf"] = 0.0
                continue
            if not ot:
                add_feedback("object error", rid, f"Object [{obj}] exists but its type is missing.", subj, obj, rtype, rname)
                rel["conf"] = 0.0
                continue

            # 5) type constraints
            rule = rule_finder.get(rtype)
            if not rule:
                add_feedback("missing rule error", rid, f"No rule defined for relation type [{rtype}].", subj, obj, rtype, rname)
                rel["conf"] = 0.0
                continue

            direction = rule.get("direction", "directed")
            from_types = rule.get("from", []) or []
            to_types = rule.get("to", []) or []
            allow_self_loop = bool(rule.get("allow_self_loop", False))

            if subj == obj and not allow_self_loop:
                add_feedback(
                    "self loop violation error",
                    rid,
                    f"Self-loop relation is not allowed for relation type [{rtype}] with subject/object [{subj}].",
                    subj,
                    obj,
                    rtype,
                    rname,
                )
                rel["conf"] = 0.0
                continue

            if direction == "symmetric":
                ok = ((st in from_types and ot in to_types) or (st in to_types and ot in from_types))
            else:
                ok = (st in from_types and ot in to_types)

            if not ok:
                # If directed, try swapping subject/object
                if direction == "directed":
                    flipped_ok = (ot in from_types and st in to_types)
                    if flipped_ok:
                        rel["subject"], rel["object"] = obj, subj
                        rel["auto_fixed"] = True
                        rel["fix_reason"] = "swap_subject_object_to_satisfy_type_constraint"
                        rel["conf"] = 0.5

                        persistence = rule.get("persistence")
                        if persistence:
                            rel["persistence"] = persistence
                        continue

                add_feedback(
                    "constraint violation error",
                    rid,
                    f"Type constraint violated for relation [{rtype}] with subject type [{st}] and object type [{ot}].",
                    subj,
                    obj,
                    rtype,
                    rname,
                )
                rel["conf"] = 0.0
                continue

            # Valid: set conf and persistence if configured
            rel["conf"] = float(rel.get("conf", 0.8) or 0.8)
            persistence = rule.get("persistence")
            if persistence:
                rel["persistence"] = persistence

        # ---- Guarantee: every conf==0.0 relation is recorded in feedbacks ----
        recorded_rids: Set[str] = set()
        for items in (feedbacks or {}).values():
            for it in (items or []):
                if isinstance(it, dict) and it.get("rid"):
                    recorded_rids.add(it["rid"])

        for rel in relations or []:
            rid = rel.get("rid", "")
            if not rid:
                continue
            conf = float(rel.get("conf", 0.8) or 0.8)
            if conf == 0.0 and rid not in recorded_rids:
                feedbacks.setdefault("unknown validation error", []).append(
                    {
                        "rid": rid,
                        "feedback": "Relation has conf=0.0 but was not captured by any validator feedback bucket.",
                        "subject": rel.get("subject", ""),
                        "object": rel.get("object", ""),
                        "relation_type": rel.get("relation_type", rel.get("type", "")),
                        "relation_name": rel.get("relation_name", ""),
                    }
                )

        return relations, feedbacks

    def _format_allowed_list(self, allowed: List[str], *, reverse_hint: bool) -> Tuple[Optional[str], str]:
        if not allowed:
            return None, "drop"

        lines = []
        for rtype in allowed:
            info = self.relation_type_info.get(rtype)
            if info:
                lines.append(f"{rtype}: {info.get('description','')}")
        text = "\n".join(lines)
        return text, ("reverse_select" if reverse_hint else "select")

    def _fix_relation_error(
        self,
        *,
        error: Dict[str, Any],
        content: str,
        all_relations: Dict[str, Dict[str, Any]],
        entities: List[Dict[str, Any]],
    ) -> Tuple[str, Dict[str, Any]]:
        rid = error["rid"]
        extracted_relation = all_relations[rid]
        feedback = error.get("feedback", "")

        entity_name2type = {e.get("name", ""): e.get("type", "") for e in entities if e.get("name") and e.get("type")}

        subj = extracted_relation.get("subject", error.get("subject", ""))
        obj = extracted_relation.get("object", error.get("object", ""))

        st = entity_name2type.get(subj, "")
        ot = entity_name2type.get(obj, "")

        allowed_forward = get_allowed_relations_between_types(self.typepair_index, st, ot)
        allowed_backward = get_allowed_relations_between_types(self.typepair_index, ot, st)

        if allowed_forward:
            allowed_list, action = self._format_allowed_list(allowed_forward, reverse_hint=False)
        elif allowed_backward:
            allowed_list, action = self._format_allowed_list(allowed_backward, reverse_hint=True)
            allowed_list = (allowed_list or "") + "\nPlease consider reversing the subject and object to match the allowed relation types above."
        else:
            return "drop", {}

        raw = self.problem_solver.fix_relation_error(
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
            output["object"] = subj
            return decision, output

        return "drop", {}

    def _fix_entity_error(
        self,
        *,
        error: Dict[str, Any],
        error_type: str,
        content: str,
    ) -> Tuple[str, Dict[str, Any]]:
        feedback = error.get("feedback", "")
        relation_type = error.get("relation_type", "")

        rinfo = self.relation_type_info.get(relation_type)
        if not rinfo:
            return "drop", {}

        candidate_list = rinfo["from"] if error_type == "subject error" else rinfo["to"]

        raw = self.problem_solver.fix_entity_error(
            text=content,
            candidate_entity_types=candidate_list,
            feedback=feedback,
        )
        out = json.loads(raw)
        return out.get("decision", "drop"), (out.get("output", {}) or {})

    def _resolve_errors(
        self,
        *,
        entities: List[Dict[str, Any]],
        all_relations: Dict[str, Dict[str, Any]],
        all_feedbacks: Dict[str, List[Dict[str, Any]]],
        content: str,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
        def _add_memory(entry: Dict[str, Any]) -> None:
            if self.memory_store is None:
                return
            try:
                self.memory_store.add(entry)
            except Exception:
                pass

        def _record_relation_refine_memory(
            *,
            rid: str,
            relation_before: Dict[str, Any],
            relation_after: Dict[str, Any],
            error_type: str,
            feedback: str,
            decision: str,
        ) -> None:
            subj_before = str((relation_before or {}).get("subject", "")).strip()
            obj_before = str((relation_before or {}).get("object", "")).strip()
            rel_before = str((relation_before or {}).get("relation_type", (relation_before or {}).get("type", ""))).strip()
            subj_after = str((relation_after or {}).get("subject", "")).strip()
            obj_after = str((relation_after or {}).get("object", "")).strip()
            rel_after = str((relation_after or {}).get("relation_type", (relation_after or {}).get("type", ""))).strip()
            if not any([subj_before, obj_before, rel_before, subj_after, obj_after, rel_after]):
                return
            content = (
                f"Relation refine [{error_type}]: "
                f"({subj_before})-[{rel_before}]->({obj_before}) => "
                f"({subj_after})-[{rel_after}]->({obj_after}); decision={decision}. "
                f"{feedback}"
            ).strip()
            kws = [x for x in [subj_before, obj_before, rel_before, subj_after, obj_after, rel_after] if x]
            _add_memory(
                {
                    "type": "term",
                    "content": content,
                    "keywords": kws[:8],
                    "confidence": 0.72,
                    "source": "agent_relation_refine",
                    "memory_scope": "relation_extraction",
                }
            )

        def _record_entity_refine_memory(
            *,
            error_type: str,
            relation_type: str,
            feedback: str,
            output: Dict[str, Any],
        ) -> None:
            name = str((output or {}).get("name", "")).strip()
            etype = str((output or {}).get("type", "")).strip()
            if not name or not etype:
                return
            content = (
                f"Entity refine [{error_type}]: '{name}' should be typed as {etype} "
                f"for relation [{relation_type}]. {feedback}"
            ).strip()
            _add_memory(
                {
                    "type": "type_rule",
                    "content": content,
                    "keywords": [name, etype, relation_type],
                    "confidence": 0.8,
                    "source": "agent_entity_refine",
                    "memory_scope": "entity_extraction",
                }
            )

        for error_type in list(all_feedbacks.keys()):
            errors = all_feedbacks.get(error_type, [])
            if not errors:
                continue

            idx = 0
            while idx < len(errors):
                err = errors[idx]
                rid = err.get("rid")
                if not rid or rid not in all_relations:
                    errors.pop(idx)
                    continue

                # Relation-level errors: attempt rewrite, then revalidate, else drop
                if error_type in [
                    "undefined relation error",
                    "missing rule error",
                    "constraint violation error",
                    "self loop violation error",
                ]:
                    rel_before = dict(all_relations.get(rid, {}) or {})
                    decision, output = self._fix_relation_error(
                        error=err,
                        content=content,
                        all_relations=all_relations,
                        entities=entities,
                    )

                    if decision != "rewrite" or not isinstance(output, dict) or not output:
                        all_relations.pop(rid, None)
                        errors.pop(idx)
                        continue

                    output = dict(output)
                    output.setdefault("rid", rid)

                    # Revalidate rewritten relation; if still invalid -> drop
                    output2, fb = self._revalidate_one_relation_global(output, entities=entities)
                    if fb is not None or float(output2.get("conf", 0.8) or 0.8) == 0.0:
                        all_relations.pop(rid, None)
                        errors.pop(idx)
                        continue

                    all_relations[rid] = output2
                    _record_relation_refine_memory(
                        rid=rid,
                        relation_before=rel_before,
                        relation_after=output2,
                        error_type=error_type,
                        feedback=str(err.get("feedback", "")).strip(),
                        decision=decision,
                    )
                    errors.pop(idx)
                    continue

                # Entity missing errors: add entity, then revalidate original relation; if still invalid -> drop
                if error_type in ["subject error", "object error"]:
                    decision, output = self._fix_entity_error(
                        error=err,
                        error_type=error_type,
                        content=content,
                    )

                    if decision != "rewrite" or not isinstance(output, dict) or not output:
                        all_relations.pop(rid, None)
                        errors.pop(idx)
                        continue

                    _record_entity_refine_memory(
                        error_type=error_type,
                        relation_type=str(err.get("relation_type", "")).strip(),
                        feedback=str(err.get("feedback", "")).strip(),
                        output=output,
                    )

                    # Add entity
                    entities.append(output)

                    # Immediately revalidate the relation that triggered this error
                    rel0 = all_relations.get(rid, {})
                    rel2, fb = self._revalidate_one_relation_global(rel0, entities=entities)
                    if fb is not None or float(rel2.get("conf", 0.8) or 0.8) == 0.0:
                        all_relations.pop(rid, None)
                        errors.pop(idx)
                        continue

                    all_relations[rid] = rel2
                    errors.pop(idx)
                    continue

                # Unknown error type: drop relation
                all_relations.pop(rid, None)
                errors.pop(idx)

        return entities, all_relations


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
            if s and o:
                buckets[(s, o)].append(rid)

        for pair, rids in list(buckets.items()):
            if len(rids) < 2:
                continue

            rels = []
            for rid in rids:
                rel = all_relations.get(rid)
                if isinstance(rel, dict):
                    rels.append({**rel, "rid": rid})

            # Rule 1: same relation_type keep first
            seen: Set[str] = set()
            drop_same: List[str] = []
            kept: List[Dict[str, Any]] = []
            for r in rels:
                rid = r.get("rid")
                rtype = (r.get("relation_type") or r.get("type") or "").strip()
                if not rid or not rtype:
                    kept.append(r)
                    continue
                if rtype in seen:
                    drop_same.append(rid)
                else:
                    seen.add(rtype)
                    kept.append(r)

            for rid in drop_same:
                all_relations.pop(rid, None)

            kept = [r for r in kept if r.get("rid") in all_relations]
            types_left = {(r.get("relation_type") or r.get("type") or "").strip() for r in kept}
            types_left.discard("")
            if len(kept) < 2 or len(types_left) < 2:
                continue

            # Rule 2: different types -> ask LLM
            raw = self.problem_solver.dedup_relations(
                # text=content,
                relations=json.dumps(kept, ensure_ascii=False, indent=2),
            )

            try:
                out = json.loads(raw)
            except Exception:
                continue

            decision = (out.get("decision") or "").strip().lower()
            if decision != "drop":
                continue

            drop_types = out.get("output", {}).get("drop_relation_types") or []
            if not isinstance(drop_types, list) or not drop_types:
                continue

            drop_set = {t.strip() for t in drop_types if isinstance(t, str) and t.strip() in types_left}
            if not drop_set:
                continue

            for r in kept:
                rid = r.get("rid")
                rtype = (r.get("relation_type") or r.get("type") or "").strip()
                if rid and rtype in drop_set:
                    if self.memory_store is not None:
                        try:
                            self.memory_store.add(
                                {
                                    "type": "dedup_rule",
                                    "content": (
                                        f"For pair ({r.get('subject','')}, {r.get('object','')}), "
                                        f"drop relation type [{rtype}] in multi-relation dedup."
                                    ),
                                    "keywords": [str(r.get("subject", "")).strip(), str(r.get("object", "")).strip(), rtype],
                                    "confidence": 0.75,
                                    "source": "agent_relation_dedup",
                                    "memory_scope": "relation_extraction",
                                }
                            )
                        except Exception:
                            pass
                    all_relations.pop(rid, None)

        return all_relations

    def _extract_entities_one_chunk(
        self,
        *,
        cleaned_text: str,
        prev_entities: List[Dict[str, Any]],
        memory_context: str = "",
    ) -> List[Dict[str, Any]]:
        """
        Entity extraction in 3 passes with hard type gating:
        - event:            only Event / Occasion
        - time_and_location only TimePoint / Location
        - general:          only Character / Object / Concept

        Rationale (logic, not bug-fix):
        - Prevent cross-pass type drift (e.g., Character re-extracted as Concept/Event).
        - Keep each pass semantically focused and reduce duplicates.
        """
        entity_extraction: List[Dict[str, Any]] = list(prev_entities or []) # carry forward previous entities for this chunk

        # self.category_priority

        # # Hard allow-lists per pass
        # allowed_by_pass: Dict[str, Set[str]] = {
        #     "event": {"Event", "Occasion"},
        #     "time_and_location": {"TimePoint", "Location"},
        #     "general": {"Character", "Object", "Concept"},
        # }

        allowed_by_pass = {
            "induced": set([ent["type"] for ent in self.entity_schema_list if ent["category"] == "induced"]),
            "anchor": set([ent["type"] for ent in self.entity_schema_list if ent["category"] == "anchor"]),
            "general": set([ent["type"] for ent in self.entity_schema_list if ent["category"] == "referential" or ent["category"] == "general_semantic"]),
        }

        def _filter_by_allowed_types(items: Any, allowed: Set[str]) -> List[Dict[str, Any]]:
            if not isinstance(items, list):
                return []
            out: List[Dict[str, Any]] = []
            for x in items:
                if not isinstance(x, dict):
                    continue
                t = (x.get("type") or "").strip()
                if t in allowed:
                    out.append(x)
            return out

        for entity_pass in ["induced", "anchor", "general"]:
            existing_txt = self._entities_text_for_extractor(entity_extraction)

            raw = self.extractor.extract_entities(
                text=cleaned_text,
                entity_group=entity_pass,
                extracted_entities=existing_txt,
                memory_context=memory_context,
            )

            try:
                result = json.loads(raw) if raw else []
            except Exception:
                result = []

            # Pass-level hard gating
            allowed = allowed_by_pass[entity_pass]
            result = _filter_by_allowed_types(result, allowed)

            # apply scope rules
            for res in result:
                # keep your original scope convention
                if res.get("type") in self.scope_rules:
                    res["scope"] = self.scope_rules[res["type"]]

            entity_extraction.extend(result)

        # Canonicalize and dedup by (name,type)
        entity_extraction = apply_canonicalization_to_entities(entity_extraction)

        seen: Set[Tuple[str, str]] = set()
        out: List[Dict[str, Any]] = []
        for e in entity_extraction:
            name = e.get("name", "")
            etype = e.get("type", "")
            if not name or not etype:
                continue
            k = (name, etype)
            if k in seen:
                continue
            seen.add(k)
            out.append(e)

        return out


    def _extract_relations_one_chunk(
        self,
        *,
        cleaned_text: str,
        entities: List[Dict[str, Any]],
        prev_all_relations: Dict[str, Dict[str, Any]],
        rid_namespace: str,
        memory_context: str = "",
    ) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
        raw2canon = build_name_canonicalizer(entities)

        all_relations: Dict[str, Dict[str, Any]] = dict(prev_all_relations or {})
        all_feedbacks: Dict[str, List[Dict[str, Any]]] = {}

        extracted_relations_ctx = self._relations_text_for_extractor(all_relations)

        for relation_group, group in self.relation_schema.items():
            # if relation_group in ["inter_event_relations", "interactions"]:
            #     continue
            entities_text = self._build_entities_text_for_group(entities, group)

            raw = self.extractor.extract_relations(
                text=cleaned_text,
                relation_group=relation_group,
                extracted_entities=entities_text,
                extracted_relations=extracted_relations_ctx,
                previous_results=None,
                feedbacks=None,
                memory_context=memory_context,
            )
            rels = json.loads(raw) if raw else []

            rels = self._filter_illegal_is_a_relations(rels, entities=entities)
            rels = apply_canonicalization_to_relations(rels, raw2canon)

            rid_prefix = f"{rid_namespace}:{relation_group}"
            self._assign_rids(rels, rid_prefix=rid_prefix)

            rels, fbs = self._validate_and_fix_relations(rels, entities=entities, group=group)

            for rel in rels:
                rid = rel.get("rid")
                if not rid:
                    continue
                if rid in all_relations:
                    raise ValueError(f"Duplicate rid detected: {rid}")
                all_relations[rid] = rel

            for k, items in (fbs or {}).items():
                all_feedbacks.setdefault(k, [])
                all_feedbacks[k].extend(items)

        return all_relations, all_feedbacks

    def _run_document_sequential(
        self,
        *,
        ordered_chunks: List[Dict[str, Any]],
        aggressive_clean: bool,
        document_rid_namespace: str,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        entities_all: List[Dict[str, Any]] = []
        all_relations: Dict[str, Dict[str, Any]] = {}

        def _norm_key(name: str) -> str:
            return normalize_name_basic(name).lower()

        def _merge_entities_inplace(base: List[Dict[str, Any]], new_ents: List[Dict[str, Any]]) -> None:
            """
            Merge new_ents into base with rules:
            - Same normalized name: do NOT append a new entity
            - Prefer richer description (non-empty and longer)
            - Do NOT overwrite type unless base type is missing
            - Always keep scope if base missing; otherwise keep base
            """
            name2idx: Dict[str, int] = {}
            for i, e in enumerate(base):
                if not isinstance(e, dict):
                    continue
                nm = e.get("name")
                if isinstance(nm, str) and nm.strip():
                    name2idx[_norm_key(nm)] = i

            for e in new_ents or []:
                if not isinstance(e, dict):
                    continue
                nm = e.get("name")
                if not isinstance(nm, str) or not nm.strip():
                    continue
                k = _norm_key(nm)

                if k not in name2idx:
                    base.append(e)
                    name2idx[k] = len(base) - 1
                    continue

                # update existing
                idx = name2idx[k]
                cur = base[idx]
                if not isinstance(cur, dict):
                    base[idx] = e
                    continue

                # description: prefer new if cur empty or new longer
                cur_desc = (cur.get("description") or "").strip() if isinstance(cur.get("description"), str) else ""
                new_desc = (e.get("description") or "").strip() if isinstance(e.get("description"), str) else ""
                if (not cur_desc and new_desc) or (new_desc and len(new_desc) > len(cur_desc)):
                    cur["description"] = new_desc
                # summary (optional): same policy
                cur_sum = (cur.get("summary") or "").strip() if isinstance(cur.get("summary"), str) else ""
                new_sum = (e.get("summary") or "").strip() if isinstance(e.get("summary"), str) else ""
                if (not cur_sum and new_sum) or (new_sum and len(new_sum) > len(cur_sum)):
                    cur["summary"] = new_sum

                # type: only fill if missing in base
                cur_type = cur.get("type")
                new_type = e.get("type")
                if (not isinstance(cur_type, str) or not cur_type.strip()) and isinstance(new_type, str) and new_type.strip():
                    cur["type"] = new_type

                # scope: only fill if missing in base
                cur_scope = cur.get("scope")
                new_scope = e.get("scope")
                if (not isinstance(cur_scope, str) or not cur_scope.strip()) and isinstance(new_scope, str) and new_scope.strip():
                    cur["scope"] = new_scope

                # carry other fields conservatively if base missing
                for field in ["properties", "aliases", "source_chunks", "source_documents"]:
                    if field in e and field not in cur:
                        cur[field] = e[field]

                base[idx] = cur

        # Ensure deterministic order
        chunks = list(ordered_chunks or [])
        try:
            chunks.sort(key=lambda c: (c.get("metadata", {}) or {}).get("order", 0))
        except Exception:
            pass

        for i, ch in enumerate(chunks):
            content = ch.get("content", "") or ""
            cleaned = clean_screenplay_text(content, aggressive=aggressive_clean)

            # Query memory by task scope to avoid cross-task memory pollution.
            entity_memory_context = ""
            relation_memory_context = ""
            if self.memory_store is not None:
                gc = getattr(self.config, "global_config", None)
                lang = str(getattr(gc, "language", "") or "").strip().lower()
                if not lang:
                    lang = str(getattr(gc, "locale", "") or "").strip().lower()
                doc_type = str(getattr(getattr(self.config, "global_config", None), "doc_type", "") or "general")
                try:
                    has_entity_memories = bool(getattr(self.memory_store, "has_scope_entries", lambda _s: False)("entity_extraction"))
                    if has_entity_memories:
                        entity_memory_context = self.memory_store.query(
                            cleaned,
                            memory_scopes={"entity_extraction", "shared"},
                            task_context={"task_name": "entity_extraction", "doc_type": doc_type, "language": lang},
                        ) or ""
                    else:
                        entity_memory_context = ""
                except Exception:
                    entity_memory_context = ""
                try:
                    relation_memory_context = self.memory_store.query(
                        cleaned,
                        memory_scopes={"relation_extraction", "shared"},
                        task_context={"task_name": "relation_extraction", "doc_type": doc_type, "language": lang},
                    ) or ""
                except Exception:
                    relation_memory_context = ""

            # 1) entities: extract, then merge into cumulative store
            # IMPORTANT: pass entities_all as prev so extractor can avoid re-adding semantically
            entities_after = self._extract_entities_one_chunk(
                cleaned_text=cleaned,
                prev_entities=entities_all,
                memory_context=entity_memory_context,
            )

            # entities_after contains prev + new, but we want to identify new deltas robustly.
            # We'll compute delta by name key relative to entities_all snapshot.
            before_keys = {_norm_key(e.get("name", "")) for e in (entities_all or []) if isinstance(e, dict) and isinstance(e.get("name"), str)}
            new_candidates = []
            for e in entities_after or []:
                if not isinstance(e, dict):
                    continue
                nm = e.get("name", "")
                if not isinstance(nm, str) or not nm.strip():
                    continue
                if _norm_key(nm) not in before_keys:
                    new_candidates.append(e)
                else:
                    # also treat as "update candidate" in case desc got better
                    new_candidates.append(e)

            # Canonicalize before merging so key stability improves
            new_candidates = apply_canonicalization_to_entities(new_candidates)
            entities_all = apply_canonicalization_to_entities(entities_all)
            _merge_entities_inplace(entities_all, new_candidates)

            # 2) relations: use ALL accumulated entities, not just this chunk's new ones
            rid_ns = f"{document_rid_namespace}.chunk{i}"
            all_relations, feedbacks = self._extract_relations_one_chunk(
                cleaned_text=cleaned,
                entities=entities_all,
                prev_all_relations=all_relations,
                rid_namespace=rid_ns,
                memory_context=relation_memory_context,
            )

            # 3) resolve errors (may add entities, may rewrite relations)
            entities_all, all_relations = self._resolve_errors(
                entities=entities_all,
                all_relations=all_relations,
                all_feedbacks=feedbacks,
                content=cleaned,
            )

            # 4) canonicalize entities again (after potential additions)
            entities_all = apply_canonicalization_to_entities(entities_all)

            # 5) dedup relations within this chunk context
            all_relations = self._dedup_multi_relations(all_relations=all_relations, content=cleaned)

        return entities_all, list(all_relations.values())

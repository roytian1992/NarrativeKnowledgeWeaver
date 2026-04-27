# core/agent/knowledge_extraction_agent.py
from __future__ import annotations

import json
import re
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union
from core.builder.manager.information_manager import InformationExtractor  # keep same dependency
from core.builder.manager.error_manager import ProblemSolver
from core.utils.general_utils import safe_dict, safe_list, safe_str
from core.utils.task_specs import load_task_spec_json
from tqdm import tqdm

_WHITESPACE_RE = re.compile(r"\s+")
_WORD_HYPHEN_LINEBREAK_RE = re.compile(r"([A-Za-z])-\s*\n\s*([A-Za-z])")
_WORD_HYPHEN_SPACE_RE = re.compile(r"([A-Za-z])-\s+([a-z])")
_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")

_RELATION_GROUNDING_KEYWORDS: Dict[str, Dict[str, List[str]]] = {
    "zh": {
        "kinship_with": [
            "父亲", "母亲", "爸爸", "妈妈", "儿子", "女儿", "兄弟", "姐妹", "哥哥", "姐姐", "弟弟", "妹妹",
            "丈夫", "妻子", "配偶", "父母", "孩子", "家人", "祖父", "祖母", "爷爷", "奶奶", "外公", "外婆",
            "叔叔", "阿姨", "舅舅", "姑妈", "姑姑", "侄子", "侄女", "外甥", "外甥女", "表亲",
        ],
        "affinity_with": [
            "朋友", "友人", "伙伴", "搭档", "盟友", "战友", "同伴", "知己", "信任", "亲近", "站在一边",
        ],
        "hostility_with": [
            "敌人", "敌对", "仇人", "宿敌", "死敌", "死对头", "对头", "仇敌", "敌视", "对立",
        ],
        "allied_with": [
            "结盟", "联盟", "盟友", "同盟", "联合阵线",
        ],
        "member_of": [
            "成员", "属于", "隶属", "加入", "效力于", "任职于", "就职于", "服役于", "在职于", "是其中一员",
        ],
        "part_of": [
            "一部分", "组成部分", "构成部分", "隶属于", "附属", "下属", "从属于",
        ],
        "possesses": [
            "拥有", "持有", "带着", "携带", "拿着", "握着", "掌握",
        ],
    },
    "en": {
        "kinship_with": [
            "father", "mother", "dad", "mom", "son", "daughter", "brother", "sister", "husband", "wife",
            "spouse", "parent", "child", "children", "sibling", "grandfather", "grandmother", "uncle", "aunt",
            "cousin", "nephew", "niece",
        ],
        "affinity_with": [
            "friend", "friends", "partner", "partners", "ally", "allies", "trusted partner", "close ally",
            "companion", "teammate", "confidant", "loyal to",
        ],
        "hostility_with": [
            "enemy", "enemies", "adversary", "adversaries", "rival", "rivals", "nemesis", "hostile to", "at war with",
        ],
        "allied_with": [
            "allied with", "ally of", "allies with", "in alliance with",
        ],
        "member_of": [
            "member of", "belongs to", "affiliated with", "works for", "serves in", "serves with", "part of the",
        ],
        "part_of": [
            "part of", "component of", "section of", "subset of", "belongs within",
        ],
        "possesses": [
            "has", "have", "holds", "holding", "carries", "carrying", "owns", "possesses",
        ],
    },
}


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


def _contains_cjk(text: Any) -> bool:
    return bool(_CJK_RE.search(str(text or "")))


def _normalize_relation_text(text: Any) -> str:
    return _WHITESPACE_RE.sub(" ", safe_str(text)).strip().lower()


def _relation_text_tokens(text: Any) -> List[str]:
    norm = _normalize_relation_text(text)
    if not norm:
        return []
    return [tok for tok in re.split(r"[^a-z0-9]+", norm) if tok]


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
        canon_name = canonicalize_entity_name(raw, scope)
        raw2canon[raw] = canon_name
        aliases = ent.get("aliases") or []
        if isinstance(aliases, list):
            for alias in aliases:
                alias_norm = normalize_name_basic(alias)
                if alias_norm:
                    raw2canon[alias_norm] = canon_name
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
        self._relation_type_to_group: Dict[str, str] = {}
        for rtype, info in (self.relation_type_info or {}).items():
            self._global_rule_finder[rtype] = {
                "from": info.get("from", []) or [],
                "to": info.get("to", []) or [],
                "direction": info.get("direction", "directed"),
                "persistence": info.get("persistence", None),
                "description": info.get("description", ""),
                "allow_self_loop": bool(info.get("allow_self_loop", False)),
            }
        for group_name, group_items in (self.relation_schema or {}).items():
            for item in group_items or []:
                if not isinstance(item, dict):
                    continue
                rtype = safe_str(item.get("type")).strip()
                if rtype and rtype not in self._relation_type_to_group:
                    self._relation_type_to_group[rtype] = safe_str(group_name).strip()

        self._relation_grounding_lexicon = self._load_relation_grounding_lexicon()
        self._relation_repair_rules = self._load_relation_repair_rules()
        self._relation_grounding_keywords = self._build_relation_grounding_keyword_index()

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

    def _focus_entities_text(self, entities: List[Dict[str, Any]]) -> str:
        lines = []
        for ent in entities or []:
            if not isinstance(ent, dict):
                continue
            name = safe_str(ent.get("name")).strip()
            etype = safe_str(ent.get("type")).strip()
            if not name:
                continue
            line = f"entity_name: {name}"
            if etype:
                line += f"        type: {etype}"
            lines.append(line)
        return "\n".join(lines)

    def _build_open_relation_hints(
        self,
        *,
        entities: List[Dict[str, Any]],
        focus_entity_names: Optional[Set[str]] = None,
    ) -> str:
        entity_name2type = {
            safe_str(e.get("name")).strip(): safe_str(e.get("type")).strip()
            for e in (entities or [])
            if isinstance(e, dict) and safe_str(e.get("name")).strip() and safe_str(e.get("type")).strip()
        }
        focus_types: Set[str] = set()
        for name in focus_entity_names or set():
            etype = entity_name2type.get(name)
            if etype:
                focus_types.add(etype)

        lines: List[str] = []
        for rtype, info in sorted((self.relation_type_info or {}).items()):
            from_types = [safe_str(x).strip() for x in (info.get("from") or []) if safe_str(x).strip()]
            to_types = [safe_str(x).strip() for x in (info.get("to") or []) if safe_str(x).strip()]
            if focus_types and not (focus_types.intersection(from_types) or focus_types.intersection(to_types)):
                continue
            desc = safe_str(info.get("description")).strip()
            direction = safe_str(info.get("direction", "directed")).strip() or "directed"
            lines.append(
                f"{rtype}: {desc}        allowed_types: {','.join(from_types)} -> {','.join(to_types)}        direction: {direction}"
            )

        return "\n".join(lines)

    def _relation_grounding_language(self, *texts: Any) -> str:
        global_cfg = getattr(self.config, "global_config", None)
        lang = safe_str(getattr(global_cfg, "language", "") or getattr(global_cfg, "locale", "")).strip().lower()
        if lang in {"zh", "en"}:
            return lang
        merged = " ".join(safe_str(text) for text in texts if safe_str(text).strip())
        return "zh" if _contains_cjk(merged) else "en"

    def _load_relation_grounding_lexicon(self) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for lang, rel_path in {
            "zh": "text_resources/relation_grounding_lexicon.json",
            "en": "text_resources_en/relation_grounding_lexicon.json",
        }.items():
            out[lang] = load_task_spec_json(
                self.config,
                relative_path=rel_path,
                default={"keywords": {}, "experience_bank": []},
            )
        return out

    def _load_relation_repair_rules(self) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for lang, rel_path in {
            "zh": "text_resources/relation_repair_rules.json",
            "en": "text_resources_en/relation_repair_rules.json",
        }.items():
            out[lang] = load_task_spec_json(
                self.config,
                relative_path=rel_path,
                default={"rules": []},
            )
        return out

    def _build_relation_grounding_keyword_index(self) -> Dict[str, Dict[str, List[str]]]:
        out: Dict[str, Dict[str, List[str]]] = {
            lang: {rtype: list(items) for rtype, items in mapping.items()}
            for lang, mapping in _RELATION_GROUNDING_KEYWORDS.items()
        }
        for lang in ["zh", "en"]:
            lang_map = out.setdefault(lang, {})
            for group_items in (self.relation_schema or {}).values():
                for item in group_items or []:
                    if not isinstance(item, dict):
                        continue
                    rtype = safe_str(item.get("type")).strip()
                    if not rtype:
                        continue
                    bucket = lang_map.setdefault(rtype, [])
                    for sample in safe_list(item.get("samples")):
                        if not isinstance(sample, dict):
                            continue
                        relation_name = _normalize_relation_text(sample.get("relation_name"))
                        if relation_name and relation_name not in bucket:
                            bucket.append(relation_name)
            lexicon = safe_dict((self._relation_grounding_lexicon or {}).get(lang))
            for rtype, items in safe_dict(lexicon.get("keywords")).items():
                bucket = lang_map.setdefault(safe_str(rtype).strip(), [])
                for item in safe_list(items):
                    keyword = _normalize_relation_text(item)
                    if keyword and keyword not in bucket:
                        bucket.append(keyword)
        return out

    def _relation_repair_rule_set(self, lang: str) -> List[Dict[str, Any]]:
        payload = safe_dict((self._relation_repair_rules or {}).get(lang))
        rules = payload.get("rules") or []
        return [item for item in rules if isinstance(item, dict)]

    def _schema_relation_grounding_experience_bank(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        seen: Set[Tuple[str, str, str, str]] = set()
        for group_items in (self.relation_schema or {}).values():
            for item in group_items or []:
                if not isinstance(item, dict):
                    continue
                relation_type = safe_str(item.get("type")).strip()
                if not relation_type:
                    continue
                from_types = [safe_str(x).strip() for x in safe_list(item.get("from")) if safe_str(x).strip()]
                to_types = [safe_str(x).strip() for x in safe_list(item.get("to")) if safe_str(x).strip()]
                subject_type_candidates = from_types or [""]
                object_type_candidates = to_types or [""]
                for sample in safe_list(item.get("samples")):
                    if not isinstance(sample, dict):
                        continue
                    phrase = safe_str(sample.get("relation_name")).strip()
                    if not phrase:
                        continue
                    phrase_norm = _normalize_relation_text(phrase)
                    if not phrase_norm:
                        continue
                    for subject_type in subject_type_candidates:
                        for object_type in object_type_candidates:
                            key = (relation_type, phrase_norm, subject_type, object_type)
                            if key in seen:
                                continue
                            seen.add(key)
                            out.append(
                                {
                                    "phrase": phrase,
                                    "relation_type": relation_type,
                                    "subject_type": subject_type,
                                    "object_type": object_type,
                                    "source": "schema_sample",
                                }
                            )
        return out

    def _relation_grounding_experience_bank(self, lang: str) -> List[Dict[str, Any]]:
        payload = safe_dict((self._relation_grounding_lexicon or {}).get(lang))
        out: List[Dict[str, Any]] = []
        seen: Set[Tuple[str, str, str, str]] = set()
        for item in payload.get("experience_bank") or []:
            if not isinstance(item, dict):
                continue
            relation_type = safe_str(item.get("relation_type")).strip()
            phrase = safe_str(item.get("phrase")).strip()
            subject_type = safe_str(item.get("subject_type")).strip()
            object_type = safe_str(item.get("object_type")).strip()
            phrase_norm = _normalize_relation_text(phrase)
            if not relation_type or not phrase_norm:
                continue
            key = (relation_type, phrase_norm, subject_type, object_type)
            if key in seen:
                continue
            seen.add(key)
            out.append(item)
        for item in self._schema_relation_grounding_experience_bank():
            relation_type = safe_str(item.get("relation_type")).strip()
            phrase = safe_str(item.get("phrase")).strip()
            subject_type = safe_str(item.get("subject_type")).strip()
            object_type = safe_str(item.get("object_type")).strip()
            phrase_norm = _normalize_relation_text(phrase)
            if not relation_type or not phrase_norm:
                continue
            key = (relation_type, phrase_norm, subject_type, object_type)
            if key in seen:
                continue
            seen.add(key)
            out.append(item)
        return out

    def _score_relation_keyword_match(self, text: str, keyword: str) -> int:
        norm_text = _normalize_relation_text(text)
        norm_keyword = _normalize_relation_text(keyword)
        if not norm_text or not norm_keyword:
            return 0
        if norm_text == norm_keyword:
            return 4
        if _contains_cjk(norm_text) or _contains_cjk(norm_keyword):
            return 3 if norm_keyword in norm_text else 0
        if re.search(rf"(?<![a-z0-9]){re.escape(norm_keyword)}(?![a-z0-9])", norm_text):
            return 3
        return 0

    def _score_relation_phrase_similarity(self, text: str, phrase: str) -> int:
        norm_text = _normalize_relation_text(text)
        norm_phrase = _normalize_relation_text(phrase)
        if not norm_text or not norm_phrase:
            return 0
        keyword_score = self._score_relation_keyword_match(norm_text, norm_phrase)
        if keyword_score > 0:
            return keyword_score + 1
        if _contains_cjk(norm_text) or _contains_cjk(norm_phrase):
            return 0
        text_tokens = set(_relation_text_tokens(norm_text))
        phrase_tokens = set(_relation_text_tokens(norm_phrase))
        if not text_tokens or not phrase_tokens:
            return 0
        overlap = len(text_tokens & phrase_tokens)
        if overlap == 0:
            return 0
        if overlap == len(phrase_tokens):
            return 3
        if overlap >= max(1, min(len(phrase_tokens), 2)):
            return 2
        return 0

    def _experience_bank_ground_open_relation(
        self,
        *,
        proposal_id: str,
        proposal: Dict[str, Any],
        candidate_specs: List[Dict[str, Any]],
        lang: str,
    ) -> Optional[Dict[str, Any]]:
        bank = self._relation_grounding_experience_bank(lang)
        if not bank:
            return None

        relation_phrase = safe_str(proposal.get("relation_phrase")).strip()
        description = safe_str(proposal.get("description")).strip()
        evidence = safe_str(proposal.get("evidence")).strip()

        spec_by_type = {
            safe_str(item.get("relation_type")).strip(): item
            for item in candidate_specs
            if isinstance(item, dict) and safe_str(item.get("relation_type")).strip()
        }
        if not spec_by_type:
            return None

        best: Optional[Tuple[int, str, str]] = None
        for item in bank:
            rtype = safe_str(item.get("relation_type")).strip()
            phrase = safe_str(item.get("phrase")).strip()
            if not rtype or not phrase or rtype not in spec_by_type:
                continue

            spec = spec_by_type[rtype]
            bank_subject_type = safe_str(item.get("subject_type")).strip()
            bank_object_type = safe_str(item.get("object_type")).strip()
            spec_subject_type = safe_str(spec.get("subject_type")).strip()
            spec_object_type = safe_str(spec.get("object_type")).strip()
            if bank_subject_type and spec_subject_type and bank_subject_type != spec_subject_type:
                continue
            if bank_object_type and spec_object_type and bank_object_type != spec_object_type:
                continue

            phrase_score = self._score_relation_phrase_similarity(relation_phrase, phrase)
            desc_score = max(self._score_relation_phrase_similarity(description, phrase) - 1, 0)
            evidence_score = max(self._score_relation_phrase_similarity(evidence, phrase) - 1, 0)
            total_score = max(phrase_score, desc_score, evidence_score)
            if total_score <= 0:
                continue
            candidate = (total_score, rtype, phrase)
            if best is None or candidate > best:
                best = candidate

        if best is None:
            return None

        top_score, top_rtype, top_phrase = best
        competing_scores = []
        for item in bank:
            rtype = safe_str(item.get("relation_type")).strip()
            phrase = safe_str(item.get("phrase")).strip()
            if not rtype or not phrase or rtype == top_rtype or rtype not in spec_by_type:
                continue
            score = max(
                self._score_relation_phrase_similarity(relation_phrase, phrase),
                max(self._score_relation_phrase_similarity(description, phrase) - 1, 0),
                max(self._score_relation_phrase_similarity(evidence, phrase) - 1, 0),
            )
            if score > 0:
                competing_scores.append(score)

        second_score = max(competing_scores) if competing_scores else 0
        if top_score < 3 or top_score <= second_score:
            return None

        chosen_spec = spec_by_type.get(top_rtype)
        if not isinstance(chosen_spec, dict):
            return None

        return {
            "proposal_id": proposal_id,
            "decision": "ground",
            "relation_type": top_rtype,
            "swap_subject_object": chosen_spec.get("endpoint_mapping") == "object_to_subject",
            "relation_name": relation_phrase or top_phrase or top_rtype,
            "description": description or evidence,
            "grounding_mode": "experience_bank_shortcut",
            "matched_keyword": top_phrase,
            "matched_language": lang,
            "matched_score": top_score,
        }

    def _exact_experience_bank_ground_open_relation(
        self,
        *,
        proposal_id: str,
        proposal: Dict[str, Any],
        candidate_specs: List[Dict[str, Any]],
        lang: str,
    ) -> Optional[Dict[str, Any]]:
        bank = self._relation_grounding_experience_bank(lang)
        if not bank:
            return None

        relation_phrase = safe_str(proposal.get("relation_phrase")).strip()
        relation_phrase_norm = _normalize_relation_text(relation_phrase)
        description = safe_str(proposal.get("description")).strip()
        evidence = safe_str(proposal.get("evidence")).strip()
        if not relation_phrase_norm:
            return None

        spec_by_type = {
            safe_str(item.get("relation_type")).strip(): item
            for item in candidate_specs
            if isinstance(item, dict) and safe_str(item.get("relation_type")).strip()
        }
        if not spec_by_type:
            return None

        matches: List[Dict[str, Any]] = []
        for item in bank:
            rtype = safe_str(item.get("relation_type")).strip()
            phrase = safe_str(item.get("phrase")).strip()
            if not rtype or not phrase or rtype not in spec_by_type:
                continue
            if _normalize_relation_text(phrase) != relation_phrase_norm:
                continue

            spec = spec_by_type[rtype]
            bank_subject_type = safe_str(item.get("subject_type")).strip()
            bank_object_type = safe_str(item.get("object_type")).strip()
            spec_subject_type = safe_str(spec.get("subject_type")).strip()
            spec_object_type = safe_str(spec.get("object_type")).strip()
            if bank_subject_type and spec_subject_type and bank_subject_type != spec_subject_type:
                continue
            if bank_object_type and spec_object_type and bank_object_type != spec_object_type:
                continue
            matches.append(item)

        matched_types = {safe_str(item.get("relation_type")).strip() for item in matches if safe_str(item.get("relation_type")).strip()}
        if len(matched_types) != 1:
            return None

        chosen = next(
            (
                item
                for item in matches
                if safe_str(item.get("source")).strip() == "schema_sample"
            ),
            matches[0] if matches else None,
        )
        if not isinstance(chosen, dict):
            return None

        chosen_type = safe_str(chosen.get("relation_type")).strip()
        chosen_spec = spec_by_type.get(chosen_type)
        if not isinstance(chosen_spec, dict):
            return None

        source = safe_str(chosen.get("source")).strip() or "experience_bank"
        if source == "schema_sample":
            grounding_mode = "schema_sample_exact_shortcut"
        else:
            grounding_mode = "experience_bank_exact_shortcut"

        return {
            "proposal_id": proposal_id,
            "decision": "ground",
            "relation_type": chosen_type,
            "swap_subject_object": chosen_spec.get("endpoint_mapping") == "object_to_subject",
            "relation_name": relation_phrase or safe_str(chosen.get("phrase")).strip() or chosen_type,
            "description": description or evidence,
            "grounding_mode": grounding_mode,
            "matched_keyword": safe_str(chosen.get("phrase")).strip(),
            "matched_language": lang,
            "matched_score": 5,
            "matched_source": source,
        }

    def _heuristic_ground_open_relation(
        self,
        *,
        proposal_id: str,
        proposal: Dict[str, Any],
        candidate_specs: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        if len(candidate_specs) <= 1:
            return None

        relation_phrase = safe_str(proposal.get("relation_phrase")).strip()
        description = safe_str(proposal.get("description")).strip()
        evidence = safe_str(proposal.get("evidence")).strip()
        lang = self._relation_grounding_language(relation_phrase, description, evidence)
        exact_grounded = self._exact_experience_bank_ground_open_relation(
            proposal_id=proposal_id,
            proposal=proposal,
            candidate_specs=candidate_specs,
            lang=lang,
        )
        if exact_grounded is not None:
            return exact_grounded
        keyword_map = self._relation_grounding_keywords.get(lang, {})
        if keyword_map:
            scores: List[Tuple[int, str, str]] = []
            for spec in candidate_specs:
                rtype = safe_str(spec.get("relation_type")).strip()
                if not rtype:
                    continue
                best_score = 0
                best_keyword = ""
                for keyword in keyword_map.get(rtype, []):
                    phrase_score = self._score_relation_keyword_match(relation_phrase, keyword)
                    desc_score = self._score_relation_keyword_match(description, keyword)
                    evidence_score = self._score_relation_keyword_match(evidence, keyword)
                    score = max(phrase_score, max(desc_score - 1, 0), max(evidence_score - 1, 0))
                    if score > best_score:
                        best_score = score
                        best_keyword = keyword
                if best_score > 0:
                    scores.append((best_score, rtype, best_keyword))

            if scores:
                scores.sort(key=lambda item: (-item[0], item[1]))
                top_score, top_rtype, top_keyword = scores[0]
                second_score = scores[1][0] if len(scores) > 1 else 0
                if top_score >= 3 and top_score > second_score:
                    chosen_spec = next(
                        (item for item in candidate_specs if safe_str(item.get("relation_type")).strip() == top_rtype),
                        None,
                    )
                    if isinstance(chosen_spec, dict):
                        return {
                            "proposal_id": proposal_id,
                            "decision": "ground",
                            "relation_type": top_rtype,
                            "swap_subject_object": chosen_spec.get("endpoint_mapping") == "object_to_subject",
                            "relation_name": relation_phrase,
                            "description": description or evidence,
                            "grounding_mode": "keyword_shortcut",
                            "matched_keyword": top_keyword,
                            "matched_language": lang,
                            "matched_score": top_score,
                        }

        return self._experience_bank_ground_open_relation(
            proposal_id=proposal_id,
            proposal=proposal,
            candidate_specs=candidate_specs,
            lang=lang,
        )

    def _match_repair_rule_window(
        self,
        *,
        text: str,
        subject_name: str,
        object_name: str,
        markers: List[str],
        window_chars: int,
    ) -> Tuple[str, str]:
        norm_text = _normalize_relation_text(text)
        subject = _normalize_relation_text(subject_name)
        object_ = _normalize_relation_text(object_name)
        if not norm_text or not subject or not object_:
            return "", ""

        subject_positions = [m.start() for m in re.finditer(re.escape(subject), norm_text)]
        object_positions = [m.start() for m in re.finditer(re.escape(object_), norm_text)]
        if not subject_positions or not object_positions:
            return "", ""

        best_marker = ""
        best_window = ""
        best_score = 0
        max_window = max(20, int(window_chars or 120))

        for s_pos in subject_positions:
            for o_pos in object_positions:
                if s_pos == o_pos and subject == object_:
                    continue
                start = min(s_pos, o_pos)
                end = max(s_pos + len(subject), o_pos + len(object_))
                if end - start > max_window:
                    continue
                window = norm_text[max(0, start - 24):min(len(norm_text), end + 24)]
                for marker in markers:
                    score = self._score_relation_keyword_match(window, marker)
                    if score > best_score:
                        best_score = score
                        best_marker = safe_str(marker).strip()
                        best_window = window
        if best_score < 3:
            return "", ""
        return best_marker, best_window

    def _build_rule_repair_relation(
        self,
        *,
        rid: str,
        subject: str,
        object_: str,
        relation_type: str,
        relation_name: str,
        description: str,
        properties: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "rid": rid,
            "subject": subject,
            "object": object_,
            "relation_type": relation_type,
            "relation_name": relation_name,
            "description": description,
            "conf": 0.85,
        }
        if properties:
            payload["properties"] = dict(properties)
        return payload

    def _repair_relation_coverage_by_rules(
        self,
        *,
        cleaned_text: str,
        entities: List[Dict[str, Any]],
        all_relations: Dict[str, Dict[str, Any]],
        rid_namespace: str,
        focus_entity_names: Set[str],
    ) -> Dict[str, Dict[str, Any]]:
        lang = self._relation_grounding_language(cleaned_text)
        rules = self._relation_repair_rule_set(lang)
        if not rules:
            return all_relations

        entity_list = [
            ent for ent in (entities or [])
            if isinstance(ent, dict) and safe_str(ent.get("name")).strip() and safe_str(ent.get("type")).strip()
        ]
        if len(entity_list) < 2:
            return all_relations

        existing_keys = {
            (
                safe_str(rel.get("subject")).strip(),
                safe_str(rel.get("relation_type")).strip(),
                safe_str(rel.get("object")).strip(),
            )
            for rel in (all_relations or {}).values()
            if isinstance(rel, dict)
        }
        next_idx = 1 + sum(1 for rid in (all_relations or {}) if safe_str(rid).startswith(f"{rid_namespace}:rule#"))

        for rule in rules:
            relation_type = safe_str(rule.get("relation_type")).strip()
            markers = [safe_str(x).strip() for x in safe_list(rule.get("markers")) if safe_str(x).strip()]
            subject_types = {safe_str(x).strip() for x in safe_list(rule.get("subject_types")) if safe_str(x).strip()}
            object_types = {safe_str(x).strip() for x in safe_list(rule.get("object_types")) if safe_str(x).strip()}
            window_chars = int(rule.get("window_chars", 120) or 120)
            symmetric = bool(rule.get("symmetric", False))
            if not relation_type or not markers:
                continue

            for subject_ent in entity_list:
                subject_name = safe_str(subject_ent.get("name")).strip()
                subject_type = safe_str(subject_ent.get("type")).strip()
                if subject_types and subject_type not in subject_types:
                    continue
                for object_ent in entity_list:
                    object_name = safe_str(object_ent.get("name")).strip()
                    object_type = safe_str(object_ent.get("type")).strip()
                    if not object_name or subject_name == object_name:
                        continue
                    if object_types and object_type not in object_types:
                        continue
                    if focus_entity_names and subject_name not in focus_entity_names and object_name not in focus_entity_names:
                        continue

                    candidate_specs = self._candidate_relation_specs_for_seed(
                        subject_name=subject_name,
                        object_name=object_name,
                        entities=entity_list,
                    )
                    chosen_spec = next(
                        (
                            spec for spec in candidate_specs
                            if safe_str(spec.get("relation_type")).strip() == relation_type
                        ),
                        None,
                    )
                    if not isinstance(chosen_spec, dict):
                        continue

                    matched_marker, matched_window = self._match_repair_rule_window(
                        text=cleaned_text,
                        subject_name=subject_name,
                        object_name=object_name,
                        markers=markers,
                        window_chars=window_chars,
                    )
                    if not matched_marker:
                        continue

                    final_subject = subject_name
                    final_object = object_name
                    if chosen_spec.get("endpoint_mapping") == "object_to_subject":
                        final_subject, final_object = final_object, final_subject
                    if symmetric and final_subject > final_object:
                        final_subject, final_object = final_object, final_subject

                    rel_key = (final_subject, relation_type, final_object)
                    if rel_key in existing_keys:
                        continue

                    description_template = safe_str(rule.get("description_template")).strip()
                    description = description_template.format(
                        subject=final_subject,
                        object=final_object,
                        marker=matched_marker,
                    ) if description_template else safe_str(matched_window).strip()
                    rel = self._build_rule_repair_relation(
                        rid=f"{rid_namespace}:rule#{next_idx}",
                        subject=final_subject,
                        object_=final_object,
                        relation_type=relation_type,
                        relation_name=safe_str(rule.get("relation_name")).strip() or matched_marker or relation_type,
                        description=description,
                        properties={
                            "repair_mode": "rule_based",
                            "rule_id": safe_str(rule.get("id")).strip(),
                            "matched_marker": matched_marker,
                            "matched_language": lang,
                        },
                    )
                    rel2, fb = self._revalidate_one_relation_global(rel, entities=entity_list)
                    if fb is not None or float(rel2.get("conf", 0.85) or 0.85) == 0.0:
                        continue
                    all_relations[safe_str(rel2.get("rid")).strip()] = rel2
                    existing_keys.add(rel_key)
                    next_idx += 1

        return all_relations

    def _relation_incident_entity_names(self, all_relations: Dict[str, Dict[str, Any]]) -> Set[str]:
        names: Set[str] = set()
        for rel in (all_relations or {}).values():
            if not isinstance(rel, dict):
                continue
            subj = safe_str(rel.get("subject")).strip()
            obj = safe_str(rel.get("object")).strip()
            if subj:
                names.add(subj)
            if obj:
                names.add(obj)
        return names

    def _entities_mentioned_in_text(self, text: str, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        lower_text = f" {safe_str(text).lower()} "
        mentioned: List[Dict[str, Any]] = []
        for ent in entities or []:
            if not isinstance(ent, dict):
                continue
            name = safe_str(ent.get("name")).strip()
            if not name:
                continue
            pattern = r"\b" + re.escape(name.lower()) + r"\b"
            if re.search(pattern, lower_text):
                mentioned.append(ent)
                continue
            if name.lower() in lower_text:
                mentioned.append(ent)
        return mentioned

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
            if not isinstance(rel, dict):
                continue
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

    def _relation_extraction_mode(self) -> str:
        kg_cfg = getattr(self.config, "knowledge_graph_builder", None)
        raw = safe_str(getattr(kg_cfg, "relation_extraction_mode", "schema_direct") if kg_cfg is not None else "schema_direct")
        mode = raw.strip().lower() or "schema_direct"
        aliases = {
            "schema_direct": "schema_direct",
            "schema": "schema_direct",
            "legacy": "schema_direct",
            "open_then_ground": "open_then_ground",
            "open": "open_then_ground",
            "free_then_ground": "open_then_ground",
        }
        return aliases.get(mode, "schema_direct")

    def _append_relation_feedback(
        self,
        all_feedbacks: Dict[str, List[Dict[str, Any]]],
        key: str,
        *,
        rid: str,
        feedback: str,
        subject: str,
        object: str,
        relation_type: str,
        relation_name: str,
    ) -> None:
        all_feedbacks.setdefault(key, []).append(
            {
                "rid": rid,
                "feedback": feedback,
                "subject": subject,
                "object": object,
                "relation_type": relation_type,
                "relation_name": relation_name,
            }
        )

    def _candidate_relation_types_for_seed(
        self,
        *,
        subject_name: str,
        object_name: str,
        entities: List[Dict[str, Any]],
    ) -> List[str]:
        entity_name2type = {
            e.get("name", ""): e.get("type", "")
            for e in (entities or [])
            if isinstance(e, dict) and e.get("name") and e.get("type")
        }
        st = safe_str(entity_name2type.get(subject_name)).strip()
        ot = safe_str(entity_name2type.get(object_name)).strip()
        if not st or not ot:
            return []
        forward = get_allowed_relations_between_types(self.typepair_index, st, ot)
        backward = get_allowed_relations_between_types(self.typepair_index, ot, st)
        merged: List[str] = []
        for rtype in forward + backward:
            if rtype not in merged:
                merged.append(rtype)
        return merged

    def _candidate_relation_specs_for_seed(
        self,
        *,
        subject_name: str,
        object_name: str,
        entities: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        entity_name2type = {
            e.get("name", ""): e.get("type", "")
            for e in (entities or [])
            if isinstance(e, dict) and e.get("name") and e.get("type")
        }
        st = safe_str(entity_name2type.get(subject_name)).strip()
        ot = safe_str(entity_name2type.get(object_name)).strip()
        if not st or not ot:
            return []

        specs: List[Dict[str, Any]] = []
        seen: Set[str] = set()
        for rtype, info in (self.relation_type_info or {}).items():
            from_types = info.get("from", []) or []
            to_types = info.get("to", []) or []
            direction = safe_str(info.get("direction", "directed")).strip() or "directed"

            forward_ok = st in from_types and ot in to_types
            reverse_ok = ot in from_types and st in to_types
            if direction == "symmetric":
                forward_ok = ((st in from_types and ot in to_types) or (st in to_types and ot in from_types))
                reverse_ok = forward_ok

            if not forward_ok and not reverse_ok:
                continue
            if rtype in seen:
                continue
            seen.add(rtype)

            if direction == "symmetric":
                endpoint_mapping = "either"
            elif forward_ok and reverse_ok:
                endpoint_mapping = "either"
            elif forward_ok:
                endpoint_mapping = "subject_to_object"
            else:
                endpoint_mapping = "object_to_subject"

            specs.append(
                {
                    "relation_type": rtype,
                    "description": safe_str(info.get("description")).strip(),
                    "endpoint_mapping": endpoint_mapping,
                    "subject_type": st,
                    "object_type": ot,
                }
            )

        return specs

    def _build_open_relation_seed(
        self,
        *,
        proposal: Dict[str, Any],
        entities: List[Dict[str, Any]],
        rid_prefix: str,
        idx: int,
        grounded: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        subject = safe_str(proposal.get("subject")).strip()
        object_ = safe_str(proposal.get("object")).strip()
        relation_phrase = safe_str(proposal.get("relation_phrase")).strip()
        description = safe_str(proposal.get("description")).strip()
        evidence = safe_str(proposal.get("evidence")).strip()
        candidate_specs = [
            dict(item)
            for item in safe_list(proposal.get("candidate_relations"))
            if isinstance(item, dict) and safe_str(item.get("relation_type")).strip()
        ]
        if not candidate_specs:
            candidate_specs = self._candidate_relation_specs_for_seed(
                subject_name=subject,
                object_name=object_,
                entities=entities,
            )
        candidate_types = [safe_str(x.get("relation_type")).strip() for x in candidate_specs if safe_str(x.get("relation_type")).strip()]

        if grounded and bool(grounded.get("swap_subject_object", False)):
            subject, object_ = object_, subject

        if grounded and safe_str(grounded.get("relation_type")).strip():
            relation_type = safe_str(grounded.get("relation_type")).strip()
            grounding_mode = safe_str(grounded.get("grounding_mode")).strip() or "llm_grounded"
        elif len(candidate_specs) == 1:
            relation_type = candidate_specs[0]["relation_type"]
            grounding_mode = "unique_candidate_shortcut"
            if candidate_specs[0].get("endpoint_mapping") == "object_to_subject":
                subject, object_ = object_, subject
        else:
            relation_type = relation_phrase
            grounding_mode = "defer_to_fix"

        props = safe_dict(proposal.get("properties"))
        props = dict(props) if isinstance(props, dict) else {}
        props["open_relation_phrase"] = relation_phrase
        if evidence:
            props["open_relation_evidence"] = evidence
        if description:
            props["open_relation_description"] = description
        if candidate_types:
            props["grounding_candidate_types"] = candidate_types
        if candidate_specs:
            props["grounding_candidate_specs"] = candidate_specs
        props["grounding_mode"] = grounding_mode
        if grounded:
            props["grounding_decision"] = safe_str(grounded.get("decision")).strip()
            if safe_str(grounded.get("description")).strip():
                props["grounding_description"] = safe_str(grounded.get("description")).strip()
            if safe_str(grounded.get("matched_keyword")).strip():
                props["grounding_matched_keyword"] = safe_str(grounded.get("matched_keyword")).strip()
            if safe_str(grounded.get("matched_source")).strip():
                props["grounding_matched_source"] = safe_str(grounded.get("matched_source")).strip()
            if safe_str(grounded.get("matched_language")).strip():
                props["grounding_matched_language"] = safe_str(grounded.get("matched_language")).strip()
            if grounded.get("matched_score") is not None:
                props["grounding_matched_score"] = grounded.get("matched_score")

        return {
            "rid": f"{rid_prefix}#{idx}",
            "subject": subject,
            "object": object_,
            "relation_type": relation_type,
            "relation_name": safe_str((grounded or {}).get("relation_name")).strip() or relation_phrase or relation_type,
            "description": safe_str((grounded or {}).get("description")).strip() or description or evidence,
            "properties": props,
        }

    def _prepare_grounded_open_relation_seeds(
        self,
        *,
        cleaned_text: str,
        proposals: List[Dict[str, Any]],
        entities: List[Dict[str, Any]],
        rid_prefix: str,
        memory_context: str = "",
    ) -> List[Dict[str, Any]]:
        ambiguous_proposals: List[Dict[str, Any]] = []
        grounded_by_id: Dict[str, Dict[str, Any]] = {}
        seeded_relations: List[Dict[str, Any]] = []

        for idx, proposal in enumerate(proposals, start=1):
            if not isinstance(proposal, dict):
                continue
            proposal = dict(proposal)
            proposal_id = f"{rid_prefix}:proposal#{idx}"
            proposal["proposal_id"] = proposal_id
            candidate_specs = [
                dict(item)
                for item in safe_list(proposal.get("candidate_relations"))
                if isinstance(item, dict) and safe_str(item.get("relation_type")).strip()
            ]
            if not candidate_specs:
                candidate_specs = self._candidate_relation_specs_for_seed(
                    subject_name=safe_str(proposal.get("subject")).strip(),
                    object_name=safe_str(proposal.get("object")).strip(),
                    entities=entities,
                )
            proposal["candidate_relations"] = candidate_specs
            if not candidate_specs:
                continue
            grounded = self._heuristic_ground_open_relation(
                proposal_id=proposal_id,
                proposal=proposal,
                candidate_specs=candidate_specs,
            )
            if len(candidate_specs) > 1:
                if grounded is None:
                    ambiguous_proposals.append(
                        {
                            "proposal_id": proposal_id,
                            "subject": safe_str(proposal.get("subject")).strip(),
                            "subject_type": next((safe_str(e.get("type")).strip() for e in entities if safe_str(e.get("name")).strip() == safe_str(proposal.get("subject")).strip()), ""),
                            "object": safe_str(proposal.get("object")).strip(),
                            "object_type": next((safe_str(e.get("type")).strip() for e in entities if safe_str(e.get("name")).strip() == safe_str(proposal.get("object")).strip()), ""),
                            "relation_phrase": safe_str(proposal.get("relation_phrase")).strip(),
                            "description": safe_str(proposal.get("description")).strip(),
                            "evidence": safe_str(proposal.get("evidence")).strip(),
                            "candidate_relations": candidate_specs,
                        }
                    )
            if grounded is None and len(candidate_specs) == 1:
                grounded = {
                    "proposal_id": proposal_id,
                    "decision": "ground",
                    "relation_type": candidate_specs[0]["relation_type"],
                    "swap_subject_object": candidate_specs[0].get("endpoint_mapping") == "object_to_subject",
                    "relation_name": safe_str(proposal.get("relation_phrase")).strip(),
                    "description": safe_str(proposal.get("description")).strip() or safe_str(proposal.get("evidence")).strip(),
                    "grounding_mode": "unique_candidate_shortcut",
                }
            if grounded is not None:
                grounded_by_id[proposal_id] = grounded

            proposal["_grounded"] = grounded
            proposals[idx - 1] = proposal

        if ambiguous_proposals:
            raw_grounded = self.extractor.ground_open_relations(
                text=cleaned_text,
                proposals=ambiguous_proposals,
                memory_context=memory_context,
            )
            try:
                grounded_items = json.loads(raw_grounded) if raw_grounded else []
            except Exception:
                grounded_items = []
            if not isinstance(grounded_items, list):
                grounded_items = []
            for item in grounded_items:
                if not isinstance(item, dict):
                    continue
                proposal_id = safe_str(item.get("proposal_id")).strip()
                if proposal_id:
                    grounded_by_id[proposal_id] = item

        for idx, proposal in enumerate(proposals, start=1):
            if not isinstance(proposal, dict):
                continue
            proposal_id = safe_str(proposal.get("proposal_id")).strip()
            grounded = proposal.get("_grounded")
            if proposal_id:
                grounded = grounded_by_id.get(proposal_id, grounded)
            if len(safe_list(proposal.get("candidate_relations"))) > 1:
                if not grounded or safe_str(grounded.get("decision")).strip().lower() != "ground":
                    continue
            seed = self._build_open_relation_seed(
                proposal=proposal,
                entities=entities,
                rid_prefix=rid_prefix,
                idx=idx,
                grounded=grounded if isinstance(grounded, dict) else None,
            )
            seeded_relations.append(seed)

        return seeded_relations

    def _repair_relation_coverage(
        self,
        *,
        cleaned_text: str,
        entities: List[Dict[str, Any]],
        all_relations: Dict[str, Dict[str, Any]],
        rid_namespace: str,
        memory_context: str = "",
    ) -> Dict[str, Dict[str, Any]]:
        if self._relation_extraction_mode() != "open_then_ground":
            return all_relations

        mentioned_entities = self._entities_mentioned_in_text(cleaned_text, entities)
        if not mentioned_entities:
            return all_relations

        incident_names = self._relation_incident_entity_names(all_relations)
        uncovered_entities = [
            ent for ent in mentioned_entities
            if safe_str(ent.get("name")).strip() and safe_str(ent.get("name")).strip() not in incident_names
        ]
        if not uncovered_entities:
            return all_relations

        uncovered_names = {safe_str(ent.get("name")).strip() for ent in uncovered_entities if safe_str(ent.get("name")).strip()}
        all_relations = self._repair_relation_coverage_by_rules(
            cleaned_text=cleaned_text,
            entities=entities,
            all_relations=all_relations,
            rid_namespace=rid_namespace,
            focus_entity_names=uncovered_names,
        )
        incident_names = self._relation_incident_entity_names(all_relations)
        uncovered_entities = [
            ent for ent in mentioned_entities
            if safe_str(ent.get("name")).strip() and safe_str(ent.get("name")).strip() not in incident_names
        ]
        if not uncovered_entities:
            return all_relations

        uncovered_names = {safe_str(ent.get("name")).strip() for ent in uncovered_entities if safe_str(ent.get("name")).strip()}
        relation_hints = self._build_open_relation_hints(entities=entities, focus_entity_names=uncovered_names)
        focus_entities_text = self._focus_entities_text(uncovered_entities)
        entities_text = self._entities_text_for_extractor(entities)
        extracted_relations_ctx = self._relations_text_for_extractor(all_relations)

        raw = self.extractor.extract_open_relations(
            text=cleaned_text,
            extracted_entities=entities_text,
            previous_results=extracted_relations_ctx or None,
            feedbacks=None,
            memory_context=memory_context,
            relation_hints=relation_hints,
            focus_entities=focus_entities_text,
        )
        try:
            proposals = json.loads(raw) if raw else []
        except Exception:
            proposals = []
        if not isinstance(proposals, list):
            proposals = []

        proposals = [
            proposal for proposal in proposals
            if isinstance(proposal, dict)
            and (
                safe_str(proposal.get("subject")).strip() in uncovered_names
                or safe_str(proposal.get("object")).strip() in uncovered_names
            )
        ]
        if not proposals:
            return all_relations

        seed_prefix = f"{rid_namespace}:coverage"
        seeded_relations = self._prepare_grounded_open_relation_seeds(
            cleaned_text=cleaned_text,
            proposals=proposals,
            entities=entities,
            rid_prefix=seed_prefix,
            memory_context=memory_context,
        )
        seeded_relations = apply_canonicalization_to_relations(seeded_relations, build_name_canonicalizer(entities))
        seeded_relations = self._filter_illegal_is_a_relations(seeded_relations, entities=entities)

        existing_keys = {
            (
                safe_str(rel.get("subject")).strip(),
                safe_str(rel.get("relation_type")).strip(),
                safe_str(rel.get("object")).strip(),
            )
            for rel in (all_relations or {}).values()
            if isinstance(rel, dict)
        }
        for rel in seeded_relations:
            rel2, fb = self._revalidate_one_relation_global(rel, entities=entities)
            if fb is not None or float(rel2.get("conf", 0.8) or 0.8) == 0.0:
                continue
            rel_key = (
                safe_str(rel2.get("subject")).strip(),
                safe_str(rel2.get("relation_type")).strip(),
                safe_str(rel2.get("object")).strip(),
            )
            if rel_key in existing_keys:
                continue
            existing_keys.add(rel_key)
            all_relations[safe_str(rel2.get("rid")).strip()] = rel2

        return all_relations

    def _extract_relations_one_chunk_open_then_ground(
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
        entities_text = self._entities_text_for_extractor(entities)
        relation_hints = self._build_open_relation_hints(entities=entities)
        raw = self.extractor.extract_open_relations(
            text=cleaned_text,
            extracted_entities=entities_text,
            previous_results=extracted_relations_ctx or None,
            feedbacks=None,
            memory_context=memory_context,
            relation_hints=relation_hints,
            focus_entities="",
        )

        try:
            proposals = json.loads(raw) if raw else []
        except Exception:
            proposals = []
        if not isinstance(proposals, list):
            proposals = []

        seed_prefix = f"{rid_namespace}:open"
        seeded_relations = self._prepare_grounded_open_relation_seeds(
            cleaned_text=cleaned_text,
            proposals=[proposal for proposal in proposals if isinstance(proposal, dict)],
            entities=entities,
            rid_prefix=seed_prefix,
            memory_context=memory_context,
        )

        seeded_relations = apply_canonicalization_to_relations(seeded_relations, raw2canon)
        seeded_relations = self._filter_illegal_is_a_relations(seeded_relations, entities=entities)

        for rel in seeded_relations:
            rid = safe_str(rel.get("rid")).strip()
            if not rid:
                continue
            rel2, fb = self._revalidate_one_relation_global(rel, entities=entities)
            all_relations[rid] = rel2
            if fb is not None:
                self._append_relation_feedback(
                    all_feedbacks,
                    safe_str(fb.get("error_type")).strip() or "unknown validation error",
                    rid=rid,
                    feedback=safe_str(fb.get("feedback")).strip(),
                    subject=safe_str(fb.get("subject")).strip() or safe_str(rel2.get("subject")).strip(),
                    object=safe_str(fb.get("object")).strip() or safe_str(rel2.get("object")).strip(),
                    relation_type=safe_str(fb.get("relation_type")).strip() or safe_str(rel2.get("relation_type")).strip(),
                    relation_name=safe_str(fb.get("relation_name")).strip() or safe_str(rel2.get("relation_name")).strip(),
                )

        return all_relations, all_feedbacks


    def _extract_relations_one_chunk(
        self,
        *,
        cleaned_text: str,
        entities: List[Dict[str, Any]],
        prev_all_relations: Dict[str, Dict[str, Any]],
        rid_namespace: str,
        memory_context: str = "",
    ) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
        if self._relation_extraction_mode() == "open_then_ground":
            return self._extract_relations_one_chunk_open_then_ground(
                cleaned_text=cleaned_text,
                entities=entities,
                prev_all_relations=prev_all_relations,
                rid_namespace=rid_namespace,
                memory_context=memory_context,
            )

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
            try:
                rels = json.loads(raw) if raw else []
            except Exception:
                rels = []
            if not isinstance(rels, list):
                rels = []
            rels = [rel for rel in rels if isinstance(rel, dict)]

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

            all_relations = self._repair_relation_coverage(
                cleaned_text=cleaned,
                entities=entities_all,
                all_relations=all_relations,
                rid_namespace=rid_ns,
                memory_context=relation_memory_context,
            )

            # 4) canonicalize entities again (after potential additions)
            entities_all = apply_canonicalization_to_entities(entities_all)

            # 5) dedup relations within this chunk context
            all_relations = self._dedup_multi_relations(all_relations=all_relations, content=cleaned)

        return entities_all, list(all_relations.values())

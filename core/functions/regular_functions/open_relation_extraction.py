from __future__ import annotations

import json
import logging
import re
import os
from typing import Any, Dict, List, Optional, Tuple

from core.utils.function_manager import process_with_format_guarantee
from core.utils.general_text import general_rules, general_repair_template
from core.utils.format import correct_json_format
from core.utils.general_utils import (
    join_bullets,
    load_json,
    pretty_json,
    safe_dict,
    safe_list,
    safe_str,
    truncate_by_word_len,
)

logger = logging.getLogger(__name__)

TASK_SCHEMA_PATH = "task_specs/task_settings/open_relation_extraction_task.json"
RELATION_SCHEMA_PATH = "task_specs/schema/default_relation_schema.json"

_WS_RE = re.compile(r"\s+")
_ENTITY_NAME_PATTERNS = [
    re.compile(r"entity[_ ]name:\s*(.*?)\s+entity[_ ]type\b", re.IGNORECASE),
    re.compile(r"entity[_ ]name:\s*(.*?)\s+type\b", re.IGNORECASE),
    re.compile(r"name:\s*(.*?)\s+type\b", re.IGNORECASE),
]


def _normalize_text(s: Any) -> str:
    return _WS_RE.sub(" ", safe_str(s)).strip()


def _normalize_key(s: Any) -> str:
    return _normalize_text(s).lower()


def _normalize_new_entities(items: Any) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    seen = set()
    for item in safe_list(items):
        if not isinstance(item, dict):
            continue
        name = _normalize_text(item.get("name"))
        description = _normalize_text(item.get("description"))
        if not name or not description:
            continue
        key = _normalize_key(name)
        if key in seen:
            continue
        seen.add(key)
        out.append({"name": name, "description": description})
    return out


class OpenRelationExtractor:
    """
    YAML-driven open relation extractor.

    This extractor proposes free-form relations between already-extracted entities
    without assigning schema relation types. Schema grounding is expected to happen
    in a later stage.
    """

    def __init__(
        self,
        prompt_loader: Any,
        llm: Any,
        task_schema_path: str = TASK_SCHEMA_PATH,
        relation_schema_path: str = RELATION_SCHEMA_PATH,
        max_few_shot_word_len: Optional[int] = None,
        max_entities_word_len: Optional[int] = None,
        max_retries: int = 3,
    ) -> None:
        if llm is None:
            raise ValueError("llm must be provided")
        if prompt_loader is None:
            raise ValueError("prompt_loader must be provided")

        self.llm = llm
        self.prompt_loader = prompt_loader
        self.task_schema_path = task_schema_path
        self.relation_schema_path = relation_schema_path
        self.max_few_shot_word_len = max_few_shot_word_len
        self.max_entities_word_len = max_entities_word_len

        self.system_prompt_id = "knowledge_extraction/system_prompt"
        self.open_relation_prompt_id = "knowledge_extraction/extract_open_relations"
        self.previous_round_prompt_id = "knowledge_extraction/context_previous_round"
        self.already_extracted_prompt_id = "knowledge_extraction/context_already_extracted"

        self.required_fields: List[str] = []
        self.field_validators: Dict[str, Any] = {}
        self.repair_template = general_repair_template
        self.max_retries = max(0, int(max_retries))

        task_schema = load_json(self.task_schema_path)
        if not isinstance(task_schema, list):
            raise ValueError(f"Invalid task schema: expected list at {self.task_schema_path}")
        self.task_schema: List[Dict[str, Any]] = [x for x in task_schema if isinstance(x, dict)]
        self.task_schema_map: Dict[str, Dict[str, Any]] = {}
        for tb in self.task_schema:
            task = safe_str(tb.get("task"))
            if task:
                self.task_schema_map[task] = tb

        relation_schema = load_json(self.relation_schema_path)
        if not isinstance(relation_schema, dict):
            raise ValueError(f"Invalid relation schema: expected dict at {self.relation_schema_path}")
        self.relation_schema: Dict[str, List[Dict[str, Any]]] = {
            safe_str(k): [x for x in safe_list(v) if isinstance(x, dict)]
            for k, v in relation_schema.items()
            if safe_str(k)
        }

    def _get_task_block(self) -> Dict[str, Any]:
        tb = self.task_schema_map.get("open_relation_extraction")
        if not isinstance(tb, dict):
            raise RuntimeError(
                f"Task schema not found for task='open_relation_extraction' in {self.task_schema_path}"
            )
        return tb

    def _extract_relation_types_from_hints(self, relation_hints: str) -> List[str]:
        out: List[str] = []
        for line in safe_str(relation_hints).splitlines():
            line = line.strip()
            if not line or ":" not in line:
                continue
            rtype = safe_str(line.split(":", 1)[0]).strip()
            if rtype and rtype not in out:
                out.append(rtype)
        return out

    def _schema_few_shot_examples(self, relation_hints: str, max_total: int = 12) -> List[Dict[str, Any]]:
        wanted = self._extract_relation_types_from_hints(relation_hints)
        examples: List[Dict[str, Any]] = []
        seen = set()
        # Prefer hinted relation types first.
        ordered_defs: List[Dict[str, Any]] = []
        type_to_def: Dict[str, Dict[str, Any]] = {}
        for group_items in self.relation_schema.values():
            for item in group_items:
                rtype = safe_str(item.get("type")).strip()
                if rtype:
                    type_to_def[rtype] = item
        for rtype in wanted:
            item = type_to_def.get(rtype)
            if item is not None:
                ordered_defs.append(item)
        for group_items in self.relation_schema.values():
            for item in group_items:
                if item not in ordered_defs:
                    ordered_defs.append(item)
        for item in ordered_defs:
            rtype = safe_str(item.get("type")).strip()
            for sample in safe_list(item.get("samples")):
                if not isinstance(sample, dict):
                    continue
                key = (
                    safe_str(sample.get("subject")).strip(),
                    safe_str(sample.get("object")).strip(),
                    safe_str(sample.get("relation_type") or rtype).strip(),
                )
                if not all(key) or key in seen:
                    continue
                seen.add(key)
                examples.append(
                    {
                        "subject": key[0],
                        "object": key[1],
                        "relation_phrase": safe_str(sample.get("relation_name")).strip()
                        or safe_str(sample.get("relation_type") or rtype).strip(),
                        "description": safe_str(sample.get("description")).strip(),
                        "evidence": safe_str(sample.get("description")).strip(),
                    }
                )
                if len(examples) >= max_total:
                    return examples
        return examples

    def _build_static_vars(self, task_block: Dict[str, Any], *, relation_hints: str = "") -> Dict[str, Any]:
        task_name = safe_str(task_block.get("task_name"))
        task_goal = safe_str(task_block.get("task_goal"))
        global_constraints_block = join_bullets(task_block.get("global_constraints"))
        extraction_rules_block = join_bullets(task_block.get("extraction_rules"))

        output_fields = [safe_str(x) for x in safe_list(task_block.get("output_fields")) if safe_str(x)]
        if not output_fields:
            output_fields = ["subject", "object", "relation_phrase", "description", "evidence"]
        output_fields_block = "\n".join([f'- "{k}"' for k in output_fields]).strip()

        schema_examples = self._schema_few_shot_examples(relation_hints)
        if schema_examples:
            few_shot_samples = pretty_json(schema_examples)
        else:
            few_shot_cfg = safe_dict(task_block.get("few_shot"))
            examples = safe_list(few_shot_cfg.get("examples"))
            few_shot_samples = pretty_json([x for x in examples if isinstance(x, dict)])
        if isinstance(self.max_few_shot_word_len, int) and self.max_few_shot_word_len > 0:
            few_shot_samples = truncate_by_word_len(few_shot_samples, self.max_few_shot_word_len, lang="auto")

        return {
            "task_name": task_name,
            "task_goal": task_goal,
            "global_constraints": global_constraints_block,
            "output_fields_block": output_fields_block,
            "extraction_rules_block": extraction_rules_block,
            "few_shot_samples": few_shot_samples,
        }

    def _parse_entity_objects(self, extracted_entities: Any) -> List[Dict[str, Any]]:
        if isinstance(extracted_entities, list):
            return [x for x in extracted_entities if isinstance(x, dict)]

        if isinstance(extracted_entities, dict):
            if isinstance(extracted_entities.get("entities"), list):
                return [x for x in extracted_entities.get("entities", []) if isinstance(x, dict)]
            return []

        text = safe_str(extracted_entities).strip()
        if not text:
            return []

        try:
            obj = json.loads(text)
        except Exception:
            obj = None

        if isinstance(obj, list):
            return [x for x in obj if isinstance(x, dict)]
        if isinstance(obj, dict):
            if isinstance(obj.get("entities"), list):
                return [x for x in obj.get("entities", []) if isinstance(x, dict)]

        out: List[Dict[str, Any]] = []
        for line in text.splitlines():
            raw = line.strip()
            if not raw:
                continue
            name = ""
            for pat in _ENTITY_NAME_PATTERNS:
                m = pat.search(raw)
                if m:
                    name = _normalize_text(m.group(1))
                    break
            if not name:
                continue
            out.append({"name": name})
        return out

    def _build_allowed_entity_map(self, extracted_entities: Any) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
        ents = self._parse_entity_objects(extracted_entities)
        alias_map: Dict[str, str] = {}
        for ent in ents:
            name = _normalize_text(ent.get("name"))
            if not name:
                continue
            alias_map.setdefault(_normalize_key(name), name)
            for alias in safe_list(ent.get("aliases")):
                alias_norm = _normalize_text(alias)
                if alias_norm:
                    alias_map.setdefault(_normalize_key(alias_norm), name)
        return ents, alias_map

    def _normalize_entities_text(self, extracted_entities: Any) -> str:
        ent_objs, _ = self._build_allowed_entity_map(extracted_entities)
        if not ent_objs:
            text = safe_str(extracted_entities).strip()
            if isinstance(self.max_entities_word_len, int) and self.max_entities_word_len > 0 and text:
                text = truncate_by_word_len(text, self.max_entities_word_len, lang="auto")
            return text

        lines: List[str] = []
        for ent in ent_objs:
            name = _normalize_text(ent.get("name"))
            etype = _normalize_text(ent.get("type"))
            desc = _normalize_text(ent.get("description"))
            if not name:
                continue
            line = f"entity_name: {name}"
            if etype:
                line += f"        type: {etype}"
            if desc:
                line += f"        description: {desc}"
            lines.append(line)

        text = "\n".join(lines).strip()
        if isinstance(self.max_entities_word_len, int) and self.max_entities_word_len > 0 and text:
            text = truncate_by_word_len(text, self.max_entities_word_len, lang="auto")
        return text

    def _canonicalize_endpoint(self, value: Any, alias_map: Dict[str, str]) -> str:
        key = _normalize_key(value)
        if not key:
            return ""
        return alias_map.get(key, "")

    def _postprocess_open_relations(self, items: Any, extracted_entities: Any) -> List[Dict[str, Any]]:
        _, alias_map = self._build_allowed_entity_map(extracted_entities)
        raw_items = items if isinstance(items, list) else []

        seen = set()
        out: List[Dict[str, Any]] = []
        for item in raw_items:
            if not isinstance(item, dict):
                continue

            new_entities = _normalize_new_entities(item.get("new_entities"))
            new_entity_map = {_normalize_key(x.get("name")): x.get("name") for x in new_entities}

            subject = self._canonicalize_endpoint(item.get("subject"), alias_map) or new_entity_map.get(_normalize_key(item.get("subject")))
            object_ = self._canonicalize_endpoint(item.get("object"), alias_map) or new_entity_map.get(_normalize_key(item.get("object")))
            relation_phrase = _normalize_text(item.get("relation_phrase"))
            description = _normalize_text(item.get("description"))
            evidence = _normalize_text(item.get("evidence"))

            if not subject or not object_ or not relation_phrase:
                continue
            if not description and evidence:
                description = evidence
            if not evidence and description:
                evidence = description
            if not description and not evidence:
                continue

            dedup_key = (
                _normalize_key(subject),
                _normalize_key(object_),
                _normalize_key(relation_phrase),
            )
            if dedup_key in seen:
                continue
            seen.add(dedup_key)

            out.append(
                {
                    "subject": subject,
                    "object": object_,
                    "relation_phrase": relation_phrase,
                    "description": description,
                    "evidence": evidence,
                    "new_entities": new_entities,
                }
            )

        out.sort(
            key=lambda x: (
                _normalize_key(x.get("subject")),
                _normalize_key(x.get("object")),
                _normalize_key(x.get("relation_phrase")),
            )
        )
        return out

    def call(self, params: str, **kwargs) -> str:
        try:
            params_dict = json.loads(params)
        except Exception as e:
            logger.error(f"Params parse failed: {e}")
            return correct_json_format(json.dumps({"error": f"Params parse failed: {str(e)}"}, ensure_ascii=False))

        text = params_dict.get("text", "") or ""
        extracted_entities = params_dict.get("extracted_entities", "") or ""
        memory_context = str(params_dict.get("memory_context") or "").strip()
        relation_hints = str(params_dict.get("relation_hints") or "").strip()
        focus_entities = str(params_dict.get("focus_entities") or "").strip()

        previous_results = params_dict.get("previous_results", [])
        feedbacks = params_dict.get("feedbacks", [])

        if not isinstance(text, str):
            text = str(text)

        if isinstance(previous_results, list):
            previous_results = json.dumps(previous_results, ensure_ascii=False)
        elif previous_results is None:
            previous_results = ""

        if isinstance(feedbacks, list):
            feedbacks = "\n".join([str(x) for x in feedbacks])
        elif feedbacks is None:
            feedbacks = ""

        try:
            task_block = self._get_task_block()
            static_vars = self._build_static_vars(task_block, relation_hints=relation_hints)
            entities_text = self._normalize_entities_text(extracted_entities)

            messages: List[Dict[str, str]] = []
            system_prompt = self.prompt_loader.render(
                self.system_prompt_id,
                static_values={"rules": general_rules, "json_only": ""},
                task_values={},
                strict=False,
            )
            messages.append({"role": "system", "content": system_prompt})

            if previous_results and feedbacks:
                prev_msg = self.prompt_loader.render(
                    self.previous_round_prompt_id,
                    static_values={},
                    task_values={
                        "previous_results": str(previous_results),
                        "feedbacks": str(feedbacks),
                    },
                    strict=False,
                )
                messages.append({"role": "user", "content": prev_msg})

            if previous_results:
                dedup_msg = self.prompt_loader.render(
                    self.already_extracted_prompt_id,
                    static_values={"default_item_type": "open relations"},
                    task_values={
                        "item_type": "open relations",
                        "extracted_items": str(previous_results),
                    },
                    strict=False,
                )
                messages.append({"role": "user", "content": dedup_msg})

            user_prompt = self.prompt_loader.render(
                self.open_relation_prompt_id,
                static_values=static_vars,
                task_values={
                    "text": text,
                    "entities": entities_text,
                    "memory_context": memory_context,
                    "relation_hints": relation_hints,
                    "focus_entities": focus_entities,
                },
                strict=False,
            )
            messages.append({"role": "user", "content": user_prompt})

            corrected_json, status = process_with_format_guarantee(
                llm_client=self.llm,
                messages=messages,
                required_fields=self.required_fields,
                field_validators=self.field_validators,
                max_retries=self.max_retries,
                repair_template=self.repair_template,
            )

            if status != "success":
                return correct_json_format(json.dumps({"error": "Open relation extraction failed"}, ensure_ascii=False))

            try:
                parsed = json.loads(corrected_json)
            except Exception:
                parsed = []

            normalized = self._postprocess_open_relations(parsed, extracted_entities=extracted_entities)
            return correct_json_format(json.dumps(normalized, ensure_ascii=False))

        except Exception as e:
            logger.error(f"Open relation extraction exception: {e}")
            return correct_json_format(json.dumps({"error": f"Open relation extraction failed: {str(e)}"}, ensure_ascii=False))

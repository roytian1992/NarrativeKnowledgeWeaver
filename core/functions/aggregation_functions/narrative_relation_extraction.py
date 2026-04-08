from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Set

from core.utils.general_utils import load_json, safe_str, pretty_json
from core.utils.function_manager import process_with_format_guarantee
from core.utils.general_text import general_repair_template
from core.utils.format import correct_json_format

logger = logging.getLogger(__name__)

DEFAULT_REL_SCHEMA_PATH = "task_specs/schema_en/default_narrative_relation_schema.json"


def _load_entity_spec_map(schema_path: str) -> dict[str, dict[str, Any]]:
    raw = load_json(schema_path)
    if not isinstance(raw, list):
        raise ValueError(f"Invalid narrative/entity schema: expected list at {schema_path}")
    spec_map: dict[str, dict[str, Any]] = {}
    for item in raw:
        if not isinstance(item, dict):
            continue
        entity_type = safe_str(item.get("type")).strip()
        if entity_type:
            spec_map[entity_type] = item
    return spec_map


def _render_entity_definitions(spec_map: dict[str, dict[str, Any]], types: list[str]) -> str:
    lines: list[str] = []
    for entity_type in types:
        spec = spec_map.get(safe_str(entity_type).strip())
        if not isinstance(spec, dict):
            raise ValueError(f"Schema definition missing for entity type: {entity_type}")
        desc = safe_str(spec.get("description")).strip()
        lines.append(f"- {entity_type}: {desc}" if desc else f"- {entity_type}")
    return "\n".join(lines).strip()


def _render_relation_schema(rel_items: List[Dict[str, Any]]) -> str:
    """
    Render relation schema into a readable block for the model.
    """
    if not rel_items:
        return "- (No relation schema found.)"

    lines: List[str] = []
    lines.append("Allowed relation types (choose exactly one, or NONE):")
    for r in rel_items:
        rtype = safe_str(r.get("type", "")).strip()
        desc = safe_str(r.get("description", "")).strip()
        direction = safe_str(r.get("direction", "")).strip()  # directed/symmetric/undirected
        persistence = safe_str(r.get("persistence", "")).strip()

        if not rtype:
            continue

        parts = [f"- {rtype}"]
        meta_bits: List[str] = []
        if direction:
            meta_bits.append(f"direction={direction}")
        if persistence:
            meta_bits.append(f"persistence={persistence}")
        if meta_bits:
            parts.append(f"({', '.join(meta_bits)})")

        if desc:
            parts.append(f": {desc}")

        lines.append(" ".join(parts))

    return "\n".join(lines)


def _render_few_shot_samples(rel_items: List[Dict[str, Any]], max_samples_total: int = 6) -> str:
    """
    Render a small set of samples across relations to guide the classifier.
    """
    samples_out: List[Dict[str, Any]] = []
    for r in rel_items:
        rtype = safe_str(r.get("type", "")).strip()
        samples = r.get("samples", [])
        if not rtype or not isinstance(samples, list) or not samples:
            continue
        for s in samples:
            if not isinstance(s, dict):
                continue
            # keep only a compact subset of fields that exist in your examples
            item: Dict[str, Any] = {}
            subj = s.get("subject")
            obj = s.get("object")
            if isinstance(subj, str):
                item["subject"] = subj
            if isinstance(obj, str):
                item["object"] = obj
            item["relation_type"] = s.get("relation_type", rtype)
            if "direction" in s:
                item["direction"] = s.get("direction")
            if "confidence" in s:
                item["confidence"] = s.get("confidence")
            if "description" in s:
                item["description"] = s.get("description")
            samples_out.append(item)

            if len(samples_out) >= max_samples_total:
                break
        if len(samples_out) >= max_samples_total:
            break

    if not samples_out:
        return "[]"

    return pretty_json(samples_out)


def _allowed_relation_types(rel_items: List[Dict[str, Any]]) -> Set[str]:
    out: Set[str] = set()
    for r in rel_items:
        t = safe_str(r.get("type", "")).strip()
        if t:
            out.add(t)
    return out


def _validate_output_obj(
    obj: Any,
    allowed_types: Set[str],
    subject_id: str,
    object_id: str,
) -> bool:
    if not isinstance(obj, dict):
        return False

    # whitelist keys
    allowed_keys = {"subject_id", "object_id", "relation_type", "confidence", "description"}
    if set(obj.keys()) != allowed_keys:
        return False

    if obj.get("subject_id") != subject_id:
        return False
    if obj.get("object_id") != object_id:
        return False

    rt = obj.get("relation_type")
    if not isinstance(rt, str) or not rt.strip():
        return False
    rt = rt.strip()
    if rt != "NONE" and rt not in allowed_types:
        return False

    conf = obj.get("confidence")
    if not isinstance(conf, (int, float)):
        return False
    if conf < 0.0 or conf > 1.0:
        return False

    desc = obj.get("description")
    if not isinstance(desc, str):
        return False

    return True


class NarrativeRelationExtractor:
    """
    YAML-driven relation classifier for Episode-Episode and Storyline-Storyline pairs.

    Prompt YAML:
      - aggregation/extract_narrative_relation

    Params JSON:
      {
        "entity_type": "Episode" | "Storyline",
        "subject": {... must include id ...},
        "object": {... must include id ...}
      }

    Output JSON:
      {
        "subject_id": "...",
        "object_id": "...",
        "relation_type": "... | NONE",
        "confidence": 0.0-1.0,
        "description": "..."
      }
    """

    def __init__(
        self,
        prompt_loader,
        llm,
        schema_path: str = DEFAULT_REL_SCHEMA_PATH,
        narrative_entity_schema_path: str = "task_specs/schema_en/default_narrative_entity_schema.json",
        prompt_id: str = "aggregation/extract_narrative_relations",
    ):
        if llm is None:
            raise ValueError("llm must be provided")
        if prompt_loader is None:
            raise ValueError("prompt_loader must be provided")

        self.llm = llm
        self.prompt_loader = prompt_loader
        self.schema_path = schema_path
        self.prompt_id = prompt_id
        self.repair_template = general_repair_template

        schema = load_json(schema_path)
        if not isinstance(schema, dict):
            raise ValueError(f"Invalid relation schema: expected dict at {schema_path}")
        self.schema = schema
        self.entity_spec_map = _load_entity_spec_map(narrative_entity_schema_path)
        self.entity_definitions_text = _render_entity_definitions(self.entity_spec_map, ["Episode", "Storyline"])

    def _get_relation_items_for_type(self, entity_type: str) -> List[Dict[str, Any]]:
        et = entity_type.strip().lower()
        if et == "episode":
            items = self.schema.get("inter_episode_relations", [])
        elif et == "storyline":
            items = self.schema.get("inter_storyline_relations", [])
        else:
            items = []
        return items if isinstance(items, list) else []

    def call(self, params: str, **kwargs) -> str:
        # 1) parse params
        try:
            params_dict = json.loads(params) if isinstance(params, str) else (params or {})
            entity_type = safe_str(params_dict.get("entity_type", "Episode")).strip()
            subject_entity = params_dict.get("subject_entity_info", None)
            object_entity = params_dict.get("object_entity_info", None)
        except Exception as e:
            logger.error(f"params parse failed: {e}")
            return correct_json_format(
                json.dumps({"error": f"params parse failed: {str(e)}"}, ensure_ascii=False)
            )

        if not isinstance(subject_entity, dict) or not isinstance(object_entity, dict):
            return correct_json_format(
                json.dumps({"error": "subject_entity_info/object_entity_info must be objects"}, ensure_ascii=False)
            )

        subject_id = safe_str(subject_entity.get("id", "")).strip()
        object_id = safe_str(object_entity.get("id", "")).strip()
        if not subject_id or not object_id:
            return correct_json_format(
                json.dumps({"error": "subject.id and object.id are required"}, ensure_ascii=False)
            )

        pair_ids = {subject_id, object_id}

        # 2) build static vars from schema
        rel_items = self._get_relation_items_for_type(entity_type)
        allowed_types = _allowed_relation_types(rel_items)

        relation_schema_text = _render_relation_schema(rel_items)
        few_shot_samples_text = _render_few_shot_samples(rel_items, max_samples_total=6)

        # 3) render prompt
        try:
            user_prompt = self.prompt_loader.render(
                self.prompt_id,
                static_values={
                    "entity_definitions_text": self.entity_definitions_text,
                    "relation_schema_text": relation_schema_text,
                    "few_shot_samples_text": few_shot_samples_text,
                },
                task_values={
                    "entity_type": entity_type,
                    "subject_json": pretty_json(subject_entity),
                    "object_json": pretty_json(object_entity),
                },
                strict=True,
            )
        except Exception as e:
            logger.error(f"relation prompt render failed: {e}")
            return correct_json_format(
                json.dumps({"error": f"prompt render failed: {str(e)}"}, ensure_ascii=False)
            )

        messages = [{"role": "user", "content": user_prompt}]

        # 4) validators (order-insensitive subject/object)
        required_fields = ["subject_id", "object_id", "relation_type", "confidence", "description"]

        field_validators: Dict[str, Any] = {
            "subject_id": lambda x: isinstance(x, str) and x in pair_ids,
            "object_id": lambda x: isinstance(x, str) and x in pair_ids,
            "relation_type": lambda x: isinstance(x, str) and (x == "NONE" or x in allowed_types),
            "confidence": lambda x: isinstance(x, (int, float)) and 0.0 <= float(x) <= 1.0,
            "description": lambda x: isinstance(x, str),
        }

        corrected_json, status = process_with_format_guarantee(
            llm_client=self.llm,
            messages=messages,
            required_fields=required_fields,
            field_validators=field_validators,
            max_retries=3,
            repair_template=self.repair_template,
        )

        if status == "success":
            try:
                obj = json.loads(corrected_json)

                # subject_id and object_id must not be identical
                if obj.get("subject_id") == obj.get("object_id"):
                    return correct_json_format(
                        json.dumps(
                            {
                                "subject_id": subject_id,
                                "object_id": object_id,
                                "relation_type": "NONE",
                                "confidence": 0.0,
                                "description": "Invalid output: subject_id equals object_id.",
                            },
                            ensure_ascii=False,
                        )
                    )

                # optional strict whole-object validation
                if _validate_output_obj(obj, allowed_types, subject_id, object_id):
                    return correct_json_format(corrected_json)

                # even if whole-object validation fails, return corrected output
                return correct_json_format(corrected_json)

            except Exception:
                pass

        return correct_json_format(
            json.dumps({"error": "narrative relation classification failed"}, ensure_ascii=False)
        )

from __future__ import annotations

import json
import logging
from typing import Any

from core.utils.general_utils import load_json, safe_list, safe_str
from core.utils.function_manager import process_with_format_guarantee
from core.utils.general_text import general_repair_template
from core.utils.format import correct_json_format

logger = logging.getLogger(__name__)


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


def _get_required_specs(spec_map: dict[str, dict[str, Any]], types: list[str]) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for entity_type in types:
        spec = spec_map.get(safe_str(entity_type).strip())
        if not isinstance(spec, dict):
            raise ValueError(f"Schema definition missing for entity type: {entity_type}")
        specs.append(spec)
    return specs


def _render_entity_definitions(specs: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for spec in specs:
        entity_type = safe_str(spec.get("type")).strip()
        desc = safe_str(spec.get("description")).strip()
        if entity_type:
            lines.append(f"- {entity_type}: {desc}" if desc else f"- {entity_type}")
    return "\n".join(lines).strip()


def _render_output_fields(spec: dict[str, Any]) -> str:
    lines: list[str] = []
    for field in safe_list(spec.get("output_fields")):
        if not isinstance(field, dict):
            continue
        name = safe_str(field.get("name")).strip()
        field_type = safe_str(field.get("type")).strip()
        desc = safe_str(field.get("description")).strip()
        if not name:
            continue
        meta = [x for x in [field_type, desc] if x]
        line = f'- "{name}"'
        if meta:
            line += f" ({'; '.join(meta)})"
        lines.append(line)
    return "\n".join(lines).strip()


def _render_rule_block(spec: dict[str, Any], key: str) -> str:
    lines: list[str] = []
    for item in safe_list(spec.get(key)):
        text = safe_str(item).strip()
        if text:
            lines.append(f"- {text}")
    return "\n".join(lines).strip()


def _render_sample_json(spec: dict[str, Any]) -> str:
    for sample in safe_list(spec.get("samples")):
        if isinstance(sample, dict):
            return json.dumps(sample, ensure_ascii=False, indent=2)
    return "{}"


def _valid_storyline_obj(x: Any) -> bool:
    """
    Relaxed validation for Storyline card.

    Requirements:
      - Must be a dict
      - name, description: non-empty strings
      - impact: string (may be empty)
      - key_characters, key_locations: list[str]
    """
    if not isinstance(x, dict):
        return False

    # name & description are required and must be non-empty
    for k in ["name", "description"]:
        v = x.get(k)
        if not isinstance(v, str) or not v.strip():
            return False

    if "impact" not in x or not isinstance(x.get("impact"), str):
        return False

    for k in ["key_characters", "key_locations"]:
        v = x.get(k)
        if not isinstance(v, list):
            return False
        if not all(isinstance(i, str) for i in v):
            return False

    return True



class StorylineExtractor:
    """
    YAML-driven Storyline extractor.

    Prompt YAML:
      - aggregation/extract_storyline_entities

    Input params JSON:
      {
        "chain_information": "..."  (required)
      }

    Output JSON:
      {
        "name": "...",
        "description": "...",
        "impact": "...",
        "key_characters": ["..."],
        "key_locations": ["..."]
      }
    """

    def __init__(
        self,
        prompt_loader,
        llm,
        narrative_entity_schema_path: str,
        prompt_id: str = "aggregation/extract_storyline_entities",
    ):
        if llm is None:
            raise ValueError("llm must be provided")
        if prompt_loader is None:
            raise ValueError("prompt_loader must be provided")

        self.llm = llm
        self.prompt_loader = prompt_loader
        self.prompt_id = prompt_id
        self.entity_spec_map = _load_entity_spec_map(narrative_entity_schema_path)

        self.required_fields = ["name", "description", "impact", "key_characters", "key_locations"]
        self.field_validators = {
            "name": lambda v: isinstance(v, str) and bool(v.strip()),
            "description": lambda v: isinstance(v, str) and bool(v.strip()),
            "impact": lambda v: isinstance(v, str),
            "key_characters": lambda v: isinstance(v, list) and all(isinstance(i, str) for i in v),
            "key_locations": lambda v: isinstance(v, list) and all(isinstance(i, str) for i in v),
        }
        self.repair_template = general_repair_template

        defs = _get_required_specs(self.entity_spec_map, ["Episode", "Storyline"])
        storyline_spec = _get_required_specs(self.entity_spec_map, ["Storyline"])[0]
        self.static_values = {
            "entity_definitions_text": _render_entity_definitions(defs),
            "storyline_output_fields_text": _render_output_fields(storyline_spec),
            "storyline_construction_constraints_text": _render_rule_block(
                storyline_spec,
                "construction_constraints",
            ),
            "storyline_writing_constraints_text": _render_rule_block(
                storyline_spec,
                "writing_constraints",
            ),
            "storyline_sample_json_text": _render_sample_json(storyline_spec),
        }

    def call(self, params: str, **kwargs) -> str:
        # 1) parse params
        try:
            params_dict = json.loads(params) if isinstance(params, str) else (params or {})
            chain_information = safe_str(params_dict.get("chain_information", "")).strip()
        except Exception as e:
            logger.error(f"参数解析失败: {e}")
            return correct_json_format(json.dumps({"error": f"参数解析失败: {str(e)}"}, ensure_ascii=False))

        if not chain_information:
            return correct_json_format(
                json.dumps({"error": "missing required field: chain_information"}, ensure_ascii=False)
            )

        # 2) render prompt
        try:
            user_prompt = self.prompt_loader.render(
                self.prompt_id,
                static_values=self.static_values,
                task_values={"chain_information": chain_information},
                strict=True,
            )
        except Exception as e:
            logger.error(f"storyline prompt render failed: {e}")
            return correct_json_format(json.dumps({"error": f"prompt render failed: {str(e)}"}, ensure_ascii=False))

        messages = [{"role": "user", "content": user_prompt}]

        # 3) LLM call with format guarantee
        corrected_json, status = process_with_format_guarantee(
            llm_client=self.llm,
            messages=messages,
            required_fields=self.required_fields,
            field_validators=self.field_validators,
            max_retries=3,
            repair_template=self.repair_template,
        )

        if status == "success":
            # Optional: a stricter whole-object validation pass
            try:
                obj = json.loads(corrected_json)
                if not _valid_storyline_obj(obj):
                    return correct_json_format(
                        json.dumps({"error": "storyline extraction produced invalid schema"}, ensure_ascii=False)
                    )
            except Exception:
                return correct_json_format(
                    json.dumps({"error": "storyline extraction produced invalid JSON"}, ensure_ascii=False)
                )

            return correct_json_format(corrected_json)

        return correct_json_format(
            json.dumps(
                {
                    "error": "storyline extraction failed",
                    "name": "",
                    "description": "",
                    "impact": "",
                    "key_characters": [],
                    "key_locations": [],
                },
                ensure_ascii=False,
            )
        )

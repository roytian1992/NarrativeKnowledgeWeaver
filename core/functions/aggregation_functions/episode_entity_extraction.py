from __future__ import annotations

import json
import logging
from typing import Any

from core.utils.general_utils import load_json, safe_list, safe_str
from core.utils.function_manager import process_with_format_guarantee
from core.utils.general_text import general_repair_template
from core.utils.format import correct_json_format

logger = logging.getLogger(__name__)


def _load_entity_spec_map(*schema_paths: str) -> dict[str, dict[str, Any]]:
    spec_map: dict[str, dict[str, Any]] = {}
    for path in schema_paths:
        raw = load_json(path)
        if not isinstance(raw, list):
            raise ValueError(f"Invalid narrative/entity schema: expected list at {path}")
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


def _valid_episodes_obj(x: Any) -> bool:
    if not isinstance(x, list):
        return False

    for ep in x:
        if not isinstance(ep, dict):
            return False

        name = ep.get("name")
        desc = ep.get("description")
        related_events = ep.get("related_events")
        related_occasions = ep.get("related_occasions")

        # name and description must be non-empty strings
        if not isinstance(name, str) or not name.strip():
            return False
        if not isinstance(desc, str) or not desc.strip():
            return False

        # related_events: optional list of strings (empty strings allowed)
        if related_events is not None:
            if not isinstance(related_events, list):
                return False
            for eid in related_events:
                if not isinstance(eid, str):
                    return False

        # related_occasions: optional list of strings (empty strings allowed)
        if related_occasions is not None:
            if not isinstance(related_occasions, list):
                return False
            for oid in related_occasions:
                if not isinstance(oid, str):
                    return False

    return True



class EpisodeExtractor:
    """
    YAML-driven Episode extractor/updater.

    Prompt YAML:
      - aggregation/extract_episode_entities
      - aggregation/update_episode_entities

    Input params JSON:
      {
        "text": "...",
        "entities": "... (JSON string or list)",
        "mode": "extract|update" (or "goal" for backward compatibility),
        "existing_episodes": "... (required for update)"
      }

    Output JSON:
      { "episodes": [ ... ] }
    """

    def __init__(
        self,
        prompt_loader,
        llm,
        entity_schema_path: str,
        narrative_entity_schema_path: str,
        extract_prompt_id: str = "aggregation/extract_episode_entities",
        update_prompt_id: str = "aggregation/update_episode_entities",
    ):
        if llm is None:
            raise ValueError("llm must be provided")
        if prompt_loader is None:
            raise ValueError("prompt_loader must be provided")

        self.llm = llm
        self.prompt_loader = prompt_loader
        self.extract_prompt_id = extract_prompt_id
        self.update_prompt_id = update_prompt_id
        self.entity_spec_map = _load_entity_spec_map(entity_schema_path, narrative_entity_schema_path)

        self.required_fields = ["episodes"]
        self.field_validators = {"episodes": _valid_episodes_obj}
        self.repair_template = general_repair_template

        defs = _get_required_specs(self.entity_spec_map, ["Event", "Occasion", "Episode", "Storyline"])
        episode_spec = _get_required_specs(self.entity_spec_map, ["Episode"])[0]
        self.common_static_values = {
            "entity_definitions_text": _render_entity_definitions(defs),
            "episode_output_fields_text": _render_output_fields(episode_spec),
            "episode_construction_constraints_text": _render_rule_block(episode_spec, "construction_constraints"),
            "episode_writing_constraints_text": _render_rule_block(episode_spec, "writing_constraints"),
            "episode_sample_json_text": _render_sample_json(episode_spec),
        }

    def call(self, params: str, **kwargs) -> str:
        # 1) parse params
        try:
            params_dict = json.loads(params) if isinstance(params, str) else (params or {})
            text = safe_str(params_dict.get("text", "")).strip()
            entities = params_dict.get("entities", "")
            existing_episodes = params_dict.get("existing_episodes", "")

            # backward compatibility: goal or mode
            mode = safe_str(params_dict.get("goal", "extract")).strip().lower()
        except Exception as e:
            logger.error(f"参数解析失败: {e}")
            return correct_json_format(json.dumps({"error": f"参数解析失败: {str(e)}"}, ensure_ascii=False))

        if not text:
            return correct_json_format(json.dumps({"error": "missing required field: text"}, ensure_ascii=False))

        # 2) normalize entities/existing_episodes to pretty JSON strings for prompt
        def _to_pretty_json(v: Any) -> str:
            if isinstance(v, str):
                s = v.strip()
                if not s:
                    return "[]"
                # if it's already json-ish, keep as-is; else wrap as string
                try:
                    obj = json.loads(s)
                    return json.dumps(obj, ensure_ascii=False, indent=2)
                except Exception:
                    return json.dumps(s, ensure_ascii=False)
            return json.dumps(v, ensure_ascii=False, indent=2)

        entities_str = _to_pretty_json(entities)

        if mode == "update":
            if not existing_episodes:
                return correct_json_format(
                    json.dumps({"error": "missing required field: existing_episodes for update"}, ensure_ascii=False)
                )
            existing_episodes_str = _to_pretty_json(existing_episodes)
            prompt_id = self.update_prompt_id
            task_values = {
                "text": text,
                "entities": entities_str,
                "existing_episodes": existing_episodes_str,
            }
        else:
            prompt_id = self.extract_prompt_id
            task_values = {
                "text": text,
                "entities": entities_str,
            }

        # 3) render prompt
        try:
            user_prompt = self.prompt_loader.render(
                prompt_id,
                static_values=self.common_static_values,
                task_values=task_values,
                strict=True,
            )
        except Exception as e:
            logger.error(f"episode prompt render failed: {e}")
            return correct_json_format(json.dumps({"error": f"prompt render failed: {str(e)}"}, ensure_ascii=False))

        messages = [{"role": "user", "content": user_prompt}]

        # 4) LLM call with format guarantee
        corrected_json, status = process_with_format_guarantee(
            llm_client=self.llm,
            messages=messages,
            required_fields=self.required_fields,
            field_validators=self.field_validators,
            max_retries=3,
            repair_template=self.repair_template,
        )

        if status == "success":
            return correct_json_format(corrected_json)

        return correct_json_format(json.dumps({"error": "episode extraction failed", "episodes": []}, ensure_ascii=False))

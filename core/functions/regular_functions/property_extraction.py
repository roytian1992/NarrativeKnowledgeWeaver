from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional
from core.utils.general_utils import load_json, safe_str
from core.utils.function_manager import process_with_format_guarantee
from core.utils.general_text import general_repair_template
from core.utils.format import correct_json_format

logger = logging.getLogger(__name__)

SCHEMA_PATH = "core/schema/default_entity_schema.json"

def _render_property_schema(props: Any) -> str:
    """
    Render properties dict into a readable schema block:
      - key: meaning
    """
    if not isinstance(props, dict) or not props:
        return "- (No property schema found.)"
    lines: List[str] = []
    for k, v in props.items():
        if not isinstance(k, str):
            continue
        kk = k.strip()
        if not kk:
            continue
        vv = safe_str(v).strip()
        if vv:
            lines.append(f"- {kk}: {vv}")
        else:
            lines.append(f"- {kk}:")
    return "\n".join(lines) if lines else "- (No property schema found.)"


class PropertyExtractor:
    """
    YAML-driven chunk-level property extractor (extraction only, no merging).

    Prompt YAML:
      - problem_solving/extract_entity_properties

    Input params JSON:
      {
        "text": "...",
        "entity_name": "...",
        "entity_type": "Character|Event|Location|Occasion|TimePoint|Object|Concept"
      }

    Output JSON:
      {
        "new_description": "... (non-empty) ...",
        "properties": { "k": "v", ... }
      }
    """

    def __init__(
        self,
        prompt_loader,
        llm,
        schema_path: str = SCHEMA_PATH,
        prompt_id: str = "knowledge_extraction/extract_entity_properties",
    ):
        if llm is None:
            raise ValueError("llm must be provided")
        if prompt_loader is None:
            raise ValueError("prompt_loader must be provided")

        self.llm = llm
        self.prompt_loader = prompt_loader
        self.schema_path = schema_path
        self.prompt_id = prompt_id

        self.required_fields = ["new_description", "properties"]

        def _valid_new_description(x: Any) -> bool:
            return isinstance(x, str) and bool(x.strip())

        def _valid_properties(x: Any) -> bool:
            if not isinstance(x, dict):
                return False
            for k, v in x.items():
                if not isinstance(k, str) or not k.strip():
                    return False
                if not isinstance(v, str):
                    return False
            return True

        self.field_validators = {
            "new_description": _valid_new_description,
            "properties": _valid_properties,
        }
        self.repair_template = general_repair_template

        schema = load_json(schema_path)
        if not isinstance(schema, list):
            raise ValueError(f"Invalid schema: expected list at {schema_path}")
        self.schema: List[Dict[str, Any]] = schema

        # type -> properties dict
        self.type_to_properties: Dict[str, Dict[str, Any]] = {}
        self._index_schema()

    def _index_schema(self) -> None:
        out: Dict[str, Dict[str, Any]] = {}
        for item in self.schema:
            if not isinstance(item, dict):
                continue
            t = safe_str(item.get("type", "")).strip()
            props = item.get("properties")
            if t and isinstance(props, dict):
                out[t] = props
        self.type_to_properties = out

    def _build_static_vars(self, entity_type: str) -> Dict[str, str]:
        props = self.type_to_properties.get(entity_type, {})
        return {"property_schema": _render_property_schema(props)}

    def call(self, params: str, **kwargs) -> str:
        # 1) parse params
        try:
            params_dict = json.loads(params) if isinstance(params, str) else (params or {})
            text = safe_str(params_dict.get("text", "")).strip()
            entity_name = safe_str(params_dict.get("entity_name", "")).strip()
            entity_type = safe_str(params_dict.get("entity_type", "")).strip()
        except Exception as e:
            logger.error(f"参数解析失败: {e}")
            return correct_json_format(json.dumps({"error": f"参数解析失败: {str(e)}"}, ensure_ascii=False))

        if not text or not entity_name or not entity_type:
            return correct_json_format(
                json.dumps(
                    {"error": "missing required fields: text/entity_name/entity_type"},
                    ensure_ascii=False,
                )
            )

        # 2) static vars from default_entity_schema.json
        static_vars = self._build_static_vars(entity_type)

        # 3) render YAML prompt
        try:
            user_prompt = self.prompt_loader.render(
                self.prompt_id,
                static_values=static_vars,
                task_values={
                    "text": text,
                    "entity_name": entity_name,
                    "entity_type": entity_type,
                },
                strict=True,
            )
        except Exception as e:
            logger.error(f"extract_entity_properties prompt render failed: {e}")
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

        return correct_json_format(json.dumps({"error": "属性抽取失败"}, ensure_ascii=False))

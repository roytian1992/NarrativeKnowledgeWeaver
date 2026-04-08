from __future__ import annotations

import json
import logging
from typing import Any, Dict, List
from core.utils.general_utils import safe_str
from core.utils.function_manager import process_with_format_guarantee
from core.utils.general_text import general_repair_template
from core.utils.format import correct_json_format

logger = logging.getLogger(__name__)


class PropertyFinalizer:
    """
    YAML-driven property finalizer (canonical abstraction + consolidation).

    Prompt YAML:
      - problem_solving/finalize_entity_properties

    Input params JSON:
      {
        "entity_name": "string",
        "properties": "string OR object (candidate properties pool)",
        "full_description": "string",
        "num_properties": int
      }

    Output JSON:
      {
        "new_description": "...",
        "properties": { "k": "v", ... }    # <= num_properties
      }
    """

    def __init__(
        self,
        prompt_loader,
        llm,
        prompt_id: str = "knowledge_extraction/finalize_entity_properties",
    ):
        if llm is None:
            raise ValueError("llm must be provided")
        if prompt_loader is None:
            raise ValueError("prompt_loader must be provided")

        self.llm = llm
        self.prompt_loader = prompt_loader
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

    @staticmethod
    def _normalize_num_properties(x: Any, default: int = 5) -> int:
        try:
            n = int(x)
            if n <= 0:
                return default
            return n
        except Exception:
            return default

    def call(self, params: str, **kwargs) -> str:
        # 1) parse params
        try:
            params_dict = json.loads(params) if isinstance(params, str) else (params or {})

            entity_name = safe_str(params_dict.get("entity_name", "")).strip()
            props_in = params_dict.get("properties", {})
            full_description = safe_str(params_dict.get("full_description", "")).strip()
            num_properties = self._normalize_num_properties(params_dict.get("num_properties", 5), default=5)
        except Exception as e:
            logger.error(f"参数解析失败: {e}")
            return correct_json_format(json.dumps({"error": f"参数解析失败: {str(e)}"}, ensure_ascii=False))

        if not entity_name or not full_description:
            return correct_json_format(
                json.dumps({"error": "missing required fields: entity_name/full_description"}, ensure_ascii=False)
            )

        # 2) parse candidate properties pool into dict, then pretty json
        try:
            if isinstance(props_in, str):
                candidate_props = json.loads(props_in) if props_in.strip() else {}
            else:
                candidate_props = props_in
        except Exception as e:
            logger.error(f"properties 不是合法 JSON: {e}")
            return correct_json_format(json.dumps({"error": f"properties 不是合法 JSON: {str(e)}"}, ensure_ascii=False))

        if not isinstance(candidate_props, dict):
            return correct_json_format(json.dumps({"error": "properties must be a JSON object (dict)"}, ensure_ascii=False))

        candidate_properties_json = json.dumps(candidate_props, ensure_ascii=False, indent=2)

        # 3) render YAML prompt
        try:
            user_prompt = self.prompt_loader.render(
                self.prompt_id,
                static_values={},
                task_values={
                    "entity_name": entity_name,
                    "candidate_properties_json": candidate_properties_json,
                    "full_description_text": full_description,
                    "num_properties": num_properties,
                },
                strict=True,
            )
        except Exception as e:
            logger.error(f"finalize_entity_properties prompt render failed: {e}")
            return correct_json_format(json.dumps({"error": f"prompt render failed: {str(e)}"}, ensure_ascii=False))

        messages: List[Dict[str, str]] = [{"role": "user", "content": user_prompt}]

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

        return correct_json_format(json.dumps({"error": "属性合并失败"}, ensure_ascii=False))

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from core.utils.function_manager import process_with_format_guarantee
from core.utils.general_text import general_repair_template
from core.utils.format import correct_json_format
from core.utils.general_utils import load_json, safe_str, join_bullets

logger = logging.getLogger(__name__)

TASK_PATH = "core/task_settings/entity_extraction_task.json"


class EntityRenamer:
    """
    YAML-driven Entity Renamer (problem_solving).

    Prompt YAML:
      - problem_solving/rename_entity

    Params JSON:
      {
        "text": "...",
        "entity_name": "...",
        "entity_type": "Character|Concept|Object|Location|TimePoint|Event|Occasion|..."
      }

    Static vars are built from:
      core/schema/default_task_descriptions.json
        - type_description
        - naming_rules
        - exclusions
    """

    def __init__(
        self,
        prompt_loader,
        llm,
        task_path: str = TASK_PATH,
        prompt_id: str = "problem_solving/rename_entity",
    ):
        if llm is None:
            raise ValueError("llm must be provided")
        if prompt_loader is None:
            raise ValueError("prompt_loader must be provided")

        self.llm = llm
        self.prompt_loader = prompt_loader
        self.task_path = task_path
        self.prompt_id = prompt_id

        self.required_fields: List[str] = ["new_name", "reason"]
        self.field_validators = {
            "new_name": lambda x: isinstance(x, str) and bool(x.strip()),
            "reason": lambda x: isinstance(x, str),
        }
        self.repair_template = general_repair_template

        schema = load_json(task_path)
        if not isinstance(schema, list):
            raise ValueError(f"Invalid schema: expected list at {task_path}")
        self.schema: List[Dict[str, Any]] = schema

        # type -> {"description": str, "naming_rules": [str], "exclusions": [str]}
        self.type_meta: Dict[str, Dict[str, Any]] = {}
        self._index_type_meta()

    def _index_type_meta(self) -> None:
        """
        Collect metadata across all task blocks.
        If the same type appears multiple times, keep the first non-empty description
        and merge lists for naming_rules/exclusions (preserving order, de-duplicated).
        """
        out: Dict[str, Dict[str, Any]] = {}

        def _merge_list(dst: List[str], src: Any) -> List[str]:
            if not isinstance(src, list):
                return dst
            for x in src:
                if not isinstance(x, str):
                    continue
                s = x.strip()
                if not s:
                    continue
                if s not in dst:
                    dst.append(s)
            return dst

        for task_block in self.schema:
            if not isinstance(task_block, dict):
                continue
            for t in (task_block.get("types") or []):
                if not isinstance(t, dict):
                    continue
                type_name = safe_str(t.get("type")).strip()
                if not type_name:
                    continue

                if type_name not in out:
                    out[type_name] = {
                        "description": "",
                        "naming_rules": [],
                        "exclusions": [],
                    }

                desc = safe_str(t.get("description")).strip()
                if desc and not out[type_name]["description"]:
                    out[type_name]["description"] = desc

                out[type_name]["naming_rules"] = _merge_list(out[type_name]["naming_rules"], t.get("naming_rules"))
                out[type_name]["exclusions"] = _merge_list(out[type_name]["exclusions"], t.get("exclusions"))

        self.type_meta = out

    @staticmethod
    def _normalize_name(name: str) -> str:
        # Conservative normalization only: trim whitespace, keep casing/punctuation.
        return (name or "").strip()

    def _build_static_vars(self, entity_type: str) -> Dict[str, str]:
        meta = self.type_meta.get(entity_type, {}) if isinstance(self.type_meta, dict) else {}

        type_description = safe_str(meta.get("description", "")).strip()

        naming_rules = join_bullets(meta.get("naming_rules", []))
        exclusions = join_bullets(meta.get("exclusions", []))

        # Keep these blocks non-empty to reduce prompt brittleness.
        if not type_description:
            type_description = "(No type description found in schema.)"
        if not naming_rules:
            naming_rules = "- (No naming rules found in schema.)"
        if not exclusions:
            exclusions = "- (No exclusions found in schema.)"

        return {
            "type_description": type_description,
            "naming_rules": naming_rules,
            "exclusions": exclusions,
        }

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
                    {
                        "error": "missing required fields: text/entity_name/entity_type",
                        "new_name": self._normalize_name(entity_name),
                        "reason": "",
                    },
                    ensure_ascii=False,
                )
            )

        # 2) build static vars from schema
        static_vars = self._build_static_vars(entity_type)

        # 3) render prompt from YAML
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
            logger.error(f"rename_entity prompt render failed: {e}")
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

        if status != "success":
            return correct_json_format(json.dumps({"error": "entity rename failed"}, ensure_ascii=False))

        # 5) post-validate and normalize
        try:
            obj = json.loads(corrected_json)
            if not isinstance(obj, dict):
                obj = {}

            new_name = safe_str(obj.get("new_name", "")).strip()
            reason = safe_str(obj.get("reason", "")).strip()

            new_name_out = self._normalize_name(new_name) or self._normalize_name(entity_name)
            safe = {"new_name": new_name_out, "reason": reason}
            return correct_json_format(json.dumps(safe, ensure_ascii=False))

        except Exception as e:
            logger.error(f"entity rename output validation failed: {e}")
            safe = {"new_name": self._normalize_name(entity_name), "reason": ""}
            return correct_json_format(json.dumps(safe, ensure_ascii=False))

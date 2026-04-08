from __future__ import annotations

import json
import logging
from typing import Any, Dict, List
from core.utils.general_utils import safe_str    
from core.utils.function_manager import process_with_format_guarantee
from core.utils.general_text import general_repair_template
from core.utils.format import correct_json_format

logger = logging.getLogger(__name__)


class EntityScopeValidator:
    """
    YAML-driven entity scope validator.

    Prompt YAML:
      - problem_solving/check_entity_scope

    Input params JSON:
      {
        "entity_name": "string",
        "context": "string"
      }

    Output JSON:
      {
        "scope": "global" | "local",
        "reason": "string"
      }
    """

    def __init__(
        self,
        prompt_loader,
        llm,
        prompt_id: str = "problem_solving/check_entity_scope",
    ):
        if llm is None:
            raise ValueError("llm must be provided")
        if prompt_loader is None:
            raise ValueError("prompt_loader must be provided")

        self.llm = llm
        self.prompt_loader = prompt_loader
        self.prompt_id = prompt_id

        self.required_fields: List[str] = ["scope", "reason"]
        self.field_validators = {
            "scope": lambda x: isinstance(x, str) and x in ("global", "local"),
            "reason": lambda x: isinstance(x, str),
        }
        self.repair_template = general_repair_template

    def call(self, params: str, **kwargs) -> str:
        # 1) parse params
        try:
            params_dict = json.loads(params) if isinstance(params, str) else (params or {})
            entity_name = safe_str(params_dict.get("entity_name", "")).strip()
            context = safe_str(params_dict.get("text", "")).strip()
        except Exception as e:
            logger.error(f"参数解析失败: {e}")
            return correct_json_format(json.dumps({"error": f"参数解析失败: {str(e)}"}, ensure_ascii=False))

        if not entity_name or not context:
            return correct_json_format(
                json.dumps(
                    {
                        "error": "missing required fields: entity_name and/or context",
                        "scope": "global",
                        "reason": "",
                    },
                    ensure_ascii=False,
                )
            )

        # 2) render YAML prompt
        try:
            user_prompt = self.prompt_loader.render(
                self.prompt_id,
                static_values={},
                task_values={
                    "entity_name": entity_name,
                    "context": context,
                },
                strict=True,
            )
        except Exception as e:
            logger.error(f"check_entity_scope prompt render failed: {e}")
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

        if status != "success":
            return correct_json_format(json.dumps({"error": "entity scope validation failed"}, ensure_ascii=False))

        # 4) post-validate and safe-normalize
        try:
            obj = json.loads(corrected_json)
            if not isinstance(obj, dict):
                obj = {}

            scope = obj.get("scope", "global")
            reason = obj.get("reason", "")

            scope = scope if scope in ("global", "local") else "global"
            reason = reason if isinstance(reason, str) else ""

            safe = {"scope": scope, "reason": reason.strip()}
            return correct_json_format(json.dumps(safe, ensure_ascii=False))
        except Exception as e:
            logger.error(f"scope validator output validation failed: {e}")
            return correct_json_format(json.dumps({"scope": "global", "reason": ""}, ensure_ascii=False))

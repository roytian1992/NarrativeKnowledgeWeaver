from __future__ import annotations

import json
import logging
from typing import Any, Dict, List
from core.utils.general_utils import safe_str
from core.utils.function_manager import process_with_format_guarantee
from core.utils.general_text import general_repair_template
from core.utils.format import correct_json_format

logger = logging.getLogger(__name__)


class RelationErrorFixer:
    """
    YAML-driven relation error fixer.

    Prompt YAML:
      - problem_solving/fix_relation_error

    Input params JSON:
      {
        "content": "string",
        "extracted_relation": "string (optional)",
        "allowed_relation_types": "string",
        "feedback": "string (optional)"
      }

    Output JSON (decision-based):
      - rewrite:
        {
          "decision": "rewrite",
          "output": {
            "subject": "...",
            "relation_type": "...",
            "object": "...",
            "relation_name": "...",
            "persistence": "stable|phase|momentary",
            "description": "..."
          }
        }
      - drop:
        { "decision": "drop", "output": {} }
    """

    def __init__(
        self,
        prompt_loader,
        llm,
        prompt_id: str = "problem_solving/fix_relation_error",
    ):
        if llm is None:
            raise ValueError("llm must be provided")
        if prompt_loader is None:
            raise ValueError("prompt_loader must be provided")

        self.llm = llm
        self.prompt_loader = prompt_loader
        self.prompt_id = prompt_id

        # keep empty unless you want strict validation
        self.required_fields: List[str] = []
        self.field_validators: Dict[str, Any] = {}
        self.repair_template = general_repair_template

    def call(self, params: str, **kwargs) -> str:
        # 1) parse params
        try:
            params_dict = json.loads(params) if isinstance(params, str) else (params or {})

            content = safe_str(params_dict.get("text", "")).strip()
            extracted_relation = safe_str(params_dict.get("extracted_relation", "")).strip()
            allowed_relation_types = safe_str(params_dict.get("allowed_relation_types", "")).strip()
            feedback = safe_str(params_dict.get("feedback", "")).strip()
        except Exception as e:
            logger.error(f"参数解析失败: {e}")
            return correct_json_format(json.dumps({"error": f"参数解析失败: {str(e)}"}, ensure_ascii=False))

        if not content or not allowed_relation_types:
            return correct_json_format(
                json.dumps(
                    {"error": "missing required fields: content and allowed_relation_types are required"},
                    ensure_ascii=False,
                )
            )

        # 2) render YAML prompt
        try:
            user_prompt = self.prompt_loader.render(
                self.prompt_id,
                static_values={},
                task_values={
                    "content": content,
                    "extracted_relation": extracted_relation,
                    "allowed_relation_types": allowed_relation_types,
                    "feedback": feedback,
                },
                strict=True,
            )
        except Exception as e:
            logger.error(f"fix_relation_error prompt render failed: {e}")
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
            return correct_json_format(corrected_json)

        return correct_json_format(json.dumps({"error": "约束违规修复失败"}, ensure_ascii=False))

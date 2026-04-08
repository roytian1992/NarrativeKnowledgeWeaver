from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional
from core.utils.general_utils import safe_str
from core.utils.function_manager import process_with_format_guarantee
from core.utils.general_text import general_repair_template
from core.utils.format import correct_json_format

logger = logging.getLogger(__name__)



def _normalize_percentage(p: Any) -> str:
    """
    Accepts: 98, 98.0, "98", "98%", "98.0%".
    Returns a string like "98" or "98.5". Falls back to "0".
    """
    if p is None:
        return "0"
    if isinstance(p, (int, float)):
        return f"{float(p):.1f}".rstrip("0").rstrip(".")
    if isinstance(p, str):
        s = p.strip()
        if s.endswith("%"):
            s = s[:-1].strip()
        try:
            v = float(s)
            return f"{v:.1f}".rstrip("0").rstrip(".")
        except Exception:
            return "0"
    return "0"


class EntityTypeOverrideValidator:
    """
    YAML-driven entity type override validator.

    Prompt YAML:
      - problem_solving/check_entity_type_override

    Input params JSON:
      {
        "text": "...",
        "entity_name": "...",
        "minority_type": "...",
        "majority_type": "...",
        "majority_percentage": 98.0 | "98" | "98%"
      }

    Output JSON:
      {
        "decision": "keep" | "drop",
        "reason": "...",
        "new_name": "..." | null
      }

    Rule:
      - if decision == "keep": new_name is REQUIRED (non-empty string)
      - if decision == "drop": new_name MUST be null (we normalize to null)
    """

    def __init__(
        self,
        prompt_loader,
        llm,
        prompt_id: str = "problem_solving/check_entity_type",
        keep_name_fallback_suffix: str = " (local usage)",
    ):
        if llm is None:
            raise ValueError("llm must be provided")
        if prompt_loader is None:
            raise ValueError("prompt_loader must be provided")

        self.llm = llm
        self.prompt_loader = prompt_loader
        self.prompt_id = prompt_id
        self.keep_name_fallback_suffix = keep_name_fallback_suffix

        self.required_fields: List[str] = ["decision", "reason"]
        self.field_validators = {
            "decision": lambda x: isinstance(x, str) and x.strip().lower() in ("keep", "drop"),
            "reason": lambda x: isinstance(x, str),
        }
        self.repair_template = general_repair_template

    def call(self, params: str, **kwargs) -> str:
        # 1) parse params
        try:
            params_dict = json.loads(params) if isinstance(params, str) else (params or {})
            text = safe_str(params_dict.get("text", "")).strip()
            entity_name = safe_str(params_dict.get("entity_name", "")).strip()
            minority_type = safe_str(params_dict.get("minority_type", "")).strip()
            majority_type = safe_str(params_dict.get("majority_type", "")).strip()
            percentage = _normalize_percentage(params_dict.get("majority_percentage", None))
        except Exception as e:
            logger.error(f"参数解析失败: {e}")
            return correct_json_format(json.dumps({"error": f"参数解析失败: {str(e)}"}, ensure_ascii=False))

        if not text or not entity_name or not minority_type or not majority_type:
            return correct_json_format(
                json.dumps(
                    {
                        "error": "missing required fields: text/entity_name/minority_type/majority_type",
                        "decision": "drop",
                        "reason": "",
                        "new_name": None,
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
                    "text": text,
                    "entity_name": entity_name,
                    "minority_type": minority_type,
                    "majority_type": majority_type,
                    "majority_percentage": percentage,
                },
                strict=True,
            )
        except Exception as e:
            logger.error(f"check_entity_type_override prompt render failed: {e}")
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
            return correct_json_format(json.dumps({"error": "entity type override validation failed"}, ensure_ascii=False))

        # 4) post-validate and normalize (conditional new_name rule)
        try:
            obj = json.loads(corrected_json)
            if not isinstance(obj, dict):
                obj = {}

            decision = obj.get("decision", "drop")
            reason = obj.get("reason", "")
            new_name = obj.get("new_name", None)

            # decision normalization
            if not isinstance(decision, str):
                decision = "drop"
            decision = decision.strip().lower()
            if decision not in ("keep", "drop"):
                decision = "drop"

            # reason normalization
            if not isinstance(reason, str):
                reason = safe_str(reason)
            reason = reason.strip()

            # new_name conditional
            if decision == "keep":
                nn = safe_str(new_name).strip() if new_name is not None else ""
                new_name_out = nn if nn else None
                if not new_name_out:
                    # safe fallback to keep pipeline moving
                    new_name_out = f"{entity_name}{self.keep_name_fallback_suffix}"
            else:
                new_name_out = None

            safe = {"decision": decision, "reason": reason, "new_name": new_name_out}
            return correct_json_format(json.dumps(safe, ensure_ascii=False))

        except Exception as e:
            logger.error(f"override validator output validation failed: {e}")
            return correct_json_format(json.dumps({"decision": "drop", "reason": "", "new_name": None}, ensure_ascii=False))

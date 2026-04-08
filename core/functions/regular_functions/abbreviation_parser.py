from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

from core.utils.general_utils import safe_str
from core.utils.function_manager import process_with_format_guarantee
from core.utils.general_text import general_repair_template
from core.utils.format import correct_json_format

logger = logging.getLogger(__name__)


def _valid_abbreviations(x: Any) -> bool:
    if not isinstance(x, list):
        return False
    # allow empty list
    for it in x:
        if not isinstance(it, dict):
            return False
        # name is required and must be non-empty
        name = it.get("name")
        if not isinstance(name, str) or not name.strip():
            return False
        # description must be a string (empty is allowed)
        desc = it.get("description")
        if desc is not None and not isinstance(desc, str):
            return False
        # optional fields if present must be strings (empty allowed)
        for opt in ("abbr", "full", "zh"):
            if opt in it:
                v = it.get(opt)
                if v is not None and not isinstance(v, str):
                    return False
    return True


class AbbreviationParser:
    """
    YAML-driven document-specific term extractor.

    Prompt YAML:
      - text_processing/parse_abbreviations

    Input params JSON:
      {
        "text": "...",
        "current_background": "... (optional) ..."
      }

    Output JSON:
      {
        "abbreviations": [
          { "name": "...", "description": "...", "abbr": "...", "full": "...", "zh": "..." }
        ]
      }
    """

    def __init__(
        self,
        prompt_loader,
        llm,
        prompt_id: str = "text_processing/parse_abbreviations",
    ):
        if llm is None:
            raise ValueError("llm must be provided")
        if prompt_loader is None:
            raise ValueError("prompt_loader must be provided")

        self.prompt_loader = prompt_loader
        self.llm = llm
        self.prompt_id = prompt_id

        self.required_fields = ["abbreviations"]
        self.field_validators = {"abbreviations": _valid_abbreviations}
        self.repair_template = general_repair_template

    def call(self, params: str, **kwargs) -> str:
        # 1) parse params
        try:
            params_dict = json.loads(params) if isinstance(params, str) else (params or {})
            text = safe_str(params_dict.get("text", "")).strip()
            current_background = safe_str(params_dict.get("current_background", "")).strip()
        except Exception as e:
            logger.error(f"参数解析失败: {e}")
            return correct_json_format(
                json.dumps({"error": f"参数解析失败: {str(e)}", "abbreviations": []}, ensure_ascii=False)
            )

        if not text:
            return correct_json_format(
                json.dumps({"error": "missing required field: text", "abbreviations": []}, ensure_ascii=False)
            )

        if not current_background:
            current_background = "None"

        # 2) render YAML prompt (text is part of the prompt, no extra messages)
        try:
            user_prompt = self.prompt_loader.render(
                self.prompt_id,
                static_values={},
                task_values={
                    "text": text,
                    "current_background": current_background,
                },
                strict=True,
            )
        except Exception as e:
            logger.error(f"parse_abbreviations prompt render failed: {e}")
            return correct_json_format(
                json.dumps({"error": f"prompt render failed: {str(e)}", "abbreviations": []}, ensure_ascii=False)
            )

        messages = [{"role": "user", "content": user_prompt}]

        # 3) LLM call with format guarantee
        corrected_json, status = process_with_format_guarantee(
            llm_client=self.llm,
            messages=messages,
            required_fields=self.required_fields,
            field_validators=self.field_validators,
            max_retries=2,
            repair_template=self.repair_template,
        )

        if status == "success":
            return correct_json_format(corrected_json)

        return correct_json_format(
            json.dumps({"error": "术语信息提取失败", "abbreviations": []}, ensure_ascii=False)
        )

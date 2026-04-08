from __future__ import annotations

import json
import logging
from typing import Any

from core.utils.general_utils import safe_str
from core.utils.function_manager import process_with_format_guarantee
from core.utils.general_text import general_repair_template
from core.utils.format import correct_json_format

logger = logging.getLogger(__name__)


def _valid_related_content(x: Any) -> bool:
    # Empty string is valid when no related content exists.
    return isinstance(x, str)


class RelatedContentExtractor:
    """
    YAML-driven related-content extractor.

    Prompt YAML:
      - text_processing/extract_related_content

    Input params JSON:
      {
        "text": "...",
        "goal": "...",
        "max_length": 300
      }

    Output JSON:
      {
        "related_content": "..."
      }
    """

    def __init__(
        self,
        prompt_loader,
        llm,
        prompt_id: str = "text_processing/extract_related_content",
    ):
        if llm is None:
            raise ValueError("llm must be provided")
        if prompt_loader is None:
            raise ValueError("prompt_loader must be provided")

        self.prompt_loader = prompt_loader
        self.llm = llm
        self.prompt_id = prompt_id

        self.required_fields = ["related_content"]
        self.field_validators = {"related_content": _valid_related_content}
        self.repair_template = general_repair_template

    def call(self, params: str, **kwargs) -> str:
        try:
            params_dict = json.loads(params) if isinstance(params, str) else (params or {})
            text = safe_str(params_dict.get("text", "")).strip()
            goal = safe_str(params_dict.get("goal", "")).strip()
            max_length_raw = params_dict.get("max_length", 300)
        except Exception as e:
            logger.error(f"参数解析失败: {e}")
            return correct_json_format(
                json.dumps(
                    {"error": f"参数解析失败: {str(e)}", "related_content": ""},
                    ensure_ascii=False,
                )
            )

        if not text:
            return correct_json_format(
                json.dumps(
                    {"error": "missing required field: text", "related_content": ""},
                    ensure_ascii=False,
                )
            )
        if not goal:
            return correct_json_format(
                json.dumps(
                    {"error": "missing required field: goal", "related_content": ""},
                    ensure_ascii=False,
                )
            )

        try:
            max_length = int(max_length_raw)
            if max_length <= 0:
                max_length = 300
        except Exception:
            max_length = 300

        try:
            user_prompt = self.prompt_loader.render(
                self.prompt_id,
                static_values={},
                task_values={
                    "text": text,
                    "goal": goal,
                    "max_length": max_length,
                },
                strict=True,
            )
        except Exception as e:
            logger.error(f"extract_related_content prompt render failed: {e}")
            return correct_json_format(
                json.dumps({"error": f"prompt render failed: {str(e)}", "related_content": ""}, ensure_ascii=False)
            )

        messages = [{"role": "user", "content": user_prompt}]

        corrected_json, status = process_with_format_guarantee(
            llm_client=self.llm,
            messages=messages,
            required_fields=self.required_fields,
            field_validators=self.field_validators,
            max_retries=1,
            repair_template=self.repair_template,
        )

        if status == "success":
            return correct_json_format(corrected_json)

        fallback = {"error": "提取相关内容失败", "related_content": ""}
        return correct_json_format(json.dumps(fallback, ensure_ascii=False))

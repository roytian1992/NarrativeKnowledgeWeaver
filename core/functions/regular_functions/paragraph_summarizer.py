from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

from core.utils.general_utils import safe_str
from core.utils.function_manager import process_with_format_guarantee
from core.utils.general_text import general_repair_template
from core.utils.format import correct_json_format

logger = logging.getLogger(__name__)


def _valid_summary(x: Any) -> bool:
    return isinstance(x, str) and bool(x.strip())


class ParagraphSummarizer:
    """
    YAML-driven paragraph summarizer.

    Prompt YAML:
      - text_processing/summarize_text

    Input params JSON:
      {
        "text": "...",
        "max_length": 200,
        "previous_summary": "... (optional) ...",
        "goal": "... (optional) ..."
      }

    Output JSON:
      {
        "summary": "... (non-empty) ..."
      }
    """

    def __init__(
        self,
        prompt_loader,
        llm,
        prompt_id: str = "text_processing/summarize_text",
    ):
        if llm is None:
            raise ValueError("llm must be provided")
        if prompt_loader is None:
            raise ValueError("prompt_loader must be provided")

        self.prompt_loader = prompt_loader
        self.llm = llm
        self.prompt_id = prompt_id

        self.required_fields = ["summary"]
        self.field_validators = {"summary": _valid_summary}
        self.repair_template = general_repair_template

    def call(self, params: str, **kwargs) -> str:
        # 1) parse params
        try:
            params_dict = json.loads(params) if isinstance(params, str) else (params or {})
            text = safe_str(params_dict.get("text", "")).strip()
            max_length_raw = params_dict.get("max_length", 200)
            previous_summary = safe_str(params_dict.get("previous_summary", "")).strip()
            goal = safe_str(params_dict.get("goal", "")).strip()
        except Exception as e:
            logger.error(f"参数解析失败: {e}")
            return correct_json_format(
                json.dumps(
                    {"error": f"参数解析失败: {str(e)}", "summary": ""},
                    ensure_ascii=False,
                )
            )

        if not text:
            return correct_json_format(
                json.dumps(
                    {"error": "missing required field: text", "summary": ""},
                    ensure_ascii=False,
                )
            )

        # 2) normalize max_length
        try:
            max_length = int(max_length_raw)
            if max_length <= 0:
                max_length = 200
        except Exception:
            max_length = 200

        # 3) render YAML prompt
        try:
            user_prompt = self.prompt_loader.render(
                self.prompt_id,
                static_values={},
                task_values={
                    "text": text,
                    "max_length": max_length,
                    # keep keys always present for strict rendering
                    "previous_summary": f"[Previous Summary]\n{previous_summary}\n" if previous_summary else "",
                    "goal": goal,
                },
                strict=True,
            )
        except Exception as e:
            logger.error(f"summarize_text prompt render failed: {e}")
            return correct_json_format(
                json.dumps({"error": f"prompt render failed: {str(e)}", "summary": ""}, ensure_ascii=False)
            )

        messages = [{"role": "user", "content": user_prompt}]

        # 4) LLM call with format guarantee
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

        # 5) fallback
        fallback = {"error": "提取摘要失败，返回整段文本", "summary": text}
        return correct_json_format(json.dumps(fallback, ensure_ascii=False))

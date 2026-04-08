from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from core.utils.general_utils import safe_str
from core.utils.function_manager import process_with_format_guarantee
from core.utils.general_text import general_repair_template
from core.utils.format import correct_json_format

logger = logging.getLogger(__name__)


def _valid_segments(x: Any) -> bool:
    if not isinstance(x, list) or len(x) == 0:
        return False
    for seg in x:
        if not isinstance(seg, str):
            return False
    return True


class SemanticSplitter:
    """
    YAML-driven semantic splitter.

    Prompt YAML:
      - text_processing/split_text

    Input params JSON:
      {
        "text": "...",
        "max_segments": 3,
        "min_length": 120
      }

    Output JSON:
      {
        "segments": ["...", "...", ""]
      }
    """

    def __init__(
        self,
        prompt_loader,
        llm,
        prompt_id: str = "text_processing/split_text",
    ):
        if llm is None:
            raise ValueError("llm must be provided")
        if prompt_loader is None:
            raise ValueError("prompt_loader must be provided")

        self.llm = llm
        self.prompt_loader = prompt_loader
        self.prompt_id = prompt_id

        self.required_fields = ["segments"]
        self.field_validators = {"segments": _valid_segments}
        self.repair_template = general_repair_template

    def call(self, params: str, **kwargs) -> str:
        # 1) parse params
        try:
            params_dict = json.loads(params) if isinstance(params, str) else (params or {})
            text = safe_str(params_dict.get("text", "")).strip()

            max_segments_raw = params_dict.get("max_segments", 3)
            min_length_raw = params_dict.get("min_length", None)
        except Exception as e:
            logger.error(f"参数解析失败: {e}")
            return correct_json_format(
                json.dumps(
                    {"error": f"参数解析失败: {str(e)}", "segments": []},
                    ensure_ascii=False,
                )
            )

        if not text:
            return correct_json_format(
                json.dumps(
                    {"error": "missing required field: text", "segments": []},
                    ensure_ascii=False,
                )
            )

        # 2) normalize numeric fields
        try:
            max_segments = int(max_segments_raw)
            if max_segments <= 0:
                max_segments = 3
        except Exception:
            max_segments = 3

        try:
            if min_length_raw is None:
                # default heuristic: 40% of words, at least 30
                word_count = len(text.split())
                min_length = max(30, int(word_count * 0.4))
            else:
                min_length = int(min_length_raw)
                if min_length < 0:
                    min_length = 0
        except Exception:
            word_count = len(text.split())
            min_length = max(30, int(word_count * 0.4))

        # 3) render YAML prompt
        try:
            user_prompt = self.prompt_loader.render(
                self.prompt_id,
                static_values={},
                task_values={
                    "text": text,
                    "max_segments": max_segments,
                    "min_length": min_length,
                },
                strict=True,
            )
        except Exception as e:
            logger.error(f"split_text prompt render failed: {e}")
            return correct_json_format(
                json.dumps({"error": f"prompt render failed: {str(e)}", "segments": []}, ensure_ascii=False)
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
        fallback = {
            "error": "语义分割失败，返回默认分割方式",
            "segments": [text, ""],
        }
        return correct_json_format(json.dumps(fallback, ensure_ascii=False))

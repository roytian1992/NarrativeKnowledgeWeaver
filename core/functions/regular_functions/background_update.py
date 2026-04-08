from __future__ import annotations

import json
import logging
from typing import Any, Dict

from core.utils.general_utils import safe_str
from core.utils.function_manager import process_with_format_guarantee
from core.utils.general_text import general_repair_template
from core.utils.format import correct_json_format

logger = logging.getLogger(__name__)


def _valid_background(x: Any) -> bool:
    return isinstance(x, str) and bool(x.strip())


def _normalize_prev(s: str) -> str:
    t = (s or "").strip()
    if not t:
        return ""
    low = t.lower()
    if low in {"none", "null", "nil", "n/a"}:
        return ""
    return t


class BackgroundUpdater:
    """
    YAML-driven background updater.

    Prompt YAML:
      - text_processing/update_background

    Input params JSON:
      {
        "text": "...",
        "current_background": "... (optional)",
        "goal": "... (optional)"
      }

    Output JSON:
      {
        "background": "..."
      }
    """

    def __init__(
        self,
        prompt_loader,
        llm,
        prompt_id: str = "text_processing/update_background",
    ):
        if llm is None:
            raise ValueError("llm must be provided")
        if prompt_loader is None:
            raise ValueError("prompt_loader must be provided")

        self.prompt_loader = prompt_loader
        self.llm = llm
        self.prompt_id = prompt_id

        self.required_fields = ["background"]
        self.field_validators = {"background": _valid_background}
        self.repair_template = general_repair_template

    def call(self, params: str, **kwargs) -> str:
        # 1) parse params
        try:
            params_dict = json.loads(params) if isinstance(params, str) else (params or {})
            text = safe_str(params_dict.get("text", "")).strip()
            current_background = safe_str(params_dict.get("current_background", "")).strip()
            goal = safe_str(params_dict.get("goal", "")).strip()
        except Exception as e:
            logger.error(f"参数解析失败: {e}")
            return correct_json_format(
                json.dumps({"error": f"参数解析失败: {str(e)}", "background": ""}, ensure_ascii=False)
            )

        if not text:
            return correct_json_format(
                json.dumps({"error": "missing required field: text", "background": ""}, ensure_ascii=False)
            )

        # 2) optional blocks
        prev = _normalize_prev(current_background)
        previous_background_block = f"[Previous Background]\n{prev}\n" if prev else ""

        goal = goal.strip()
        goal_block = f"[Goal]\n{goal}\n" if goal else ""

        # 3) render prompt
        try:
            user_prompt = self.prompt_loader.render(
                self.prompt_id,
                static_values={},
                task_values={
                    "text": text,
                    "previous_background_block": previous_background_block,
                    "goal_block": goal_block,
                },
                strict=True,
            )
        except Exception as e:
            logger.error(f"update_background prompt render failed: {e}")
            return correct_json_format(
                json.dumps({"error": f"prompt render failed: {str(e)}", "background": ""}, ensure_ascii=False)
            )

        messages = [{"role": "user", "content": user_prompt}]

        # 4) LLM call
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
            json.dumps({"error": "背景信息生成失败", "background": ""}, ensure_ascii=False)
        )

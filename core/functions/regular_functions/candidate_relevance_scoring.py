from __future__ import annotations

import json
import logging
from typing import Any

from core.utils.format import correct_json_format
from core.utils.function_manager import process_with_format_guarantee
from core.utils.general_text import general_repair_template
from core.utils.general_utils import safe_str

logger = logging.getLogger(__name__)


def _valid_probability(x: Any) -> bool:
    try:
        val = float(x)
    except Exception:
        return False
    return 0.0 <= val <= 1.0


def _valid_bool_like(x: Any) -> bool:
    if isinstance(x, bool):
        return True
    if isinstance(x, str):
        return x.strip().lower() in {"true", "false"}
    return False


def _valid_reason(x: Any) -> bool:
    return isinstance(x, str)


class CandidateRelevanceScorer:
    """
    YAML-driven relevance scorer for generic retrieval candidates.

    Prompt YAML:
      - text_processing/score_candidate_relevance

    Input params JSON:
      {
        "text": "...",
        "goal": "..."
      }

    Output JSON:
      {
        "probability": 0.0,
        "is_relevant": false,
        "reason": ""
      }
    """

    def __init__(
        self,
        prompt_loader,
        llm,
        prompt_id: str = "text_processing/score_candidate_relevance",
    ):
        if llm is None:
            raise ValueError("llm must be provided")
        if prompt_loader is None:
            raise ValueError("prompt_loader must be provided")

        self.prompt_loader = prompt_loader
        self.llm = llm
        self.prompt_id = prompt_id
        self.required_fields = ["probability", "is_relevant", "reason"]
        self.field_validators = {
            "probability": _valid_probability,
            "is_relevant": _valid_bool_like,
            "reason": _valid_reason,
        }
        self.repair_template = general_repair_template

    def call(self, params: str, **kwargs) -> str:
        try:
            params_dict = json.loads(params) if isinstance(params, str) else (params or {})
            text = safe_str(params_dict.get("text", "")).strip()
            goal = safe_str(params_dict.get("goal", "")).strip()
        except Exception as e:
            logger.error("参数解析失败: %s", e)
            return correct_json_format(
                json.dumps(
                    {
                        "error": f"参数解析失败: {str(e)}",
                        "probability": 0.0,
                        "is_relevant": False,
                        "reason": "",
                    },
                    ensure_ascii=False,
                )
            )

        if not text:
            return correct_json_format(
                json.dumps(
                    {
                        "error": "missing required field: text",
                        "probability": 0.0,
                        "is_relevant": False,
                        "reason": "",
                    },
                    ensure_ascii=False,
                )
            )
        if not goal:
            return correct_json_format(
                json.dumps(
                    {
                        "error": "missing required field: goal",
                        "probability": 0.0,
                        "is_relevant": False,
                        "reason": "",
                    },
                    ensure_ascii=False,
                )
            )

        try:
            user_prompt = self.prompt_loader.render(
                self.prompt_id,
                static_values={},
                task_values={
                    "text": text,
                    "goal": goal,
                },
                strict=True,
            )
        except Exception as e:
            logger.error("score_candidate_relevance prompt render failed: %s", e)
            return correct_json_format(
                json.dumps(
                    {
                        "error": f"prompt render failed: {str(e)}",
                        "probability": 0.0,
                        "is_relevant": False,
                        "reason": "",
                    },
                    ensure_ascii=False,
                )
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

        return correct_json_format(
            json.dumps(
                {
                    "error": "候选相关性评分失败",
                    "probability": 0.0,
                    "is_relevant": False,
                    "reason": "",
                },
                ensure_ascii=False,
            )
        )

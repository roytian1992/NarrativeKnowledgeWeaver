from __future__ import annotations

import json
import logging
from typing import Any

from core.utils.format import correct_json_format
from core.utils.function_manager import process_with_format_guarantee
from core.utils.general_text import general_repair_template
from core.utils.general_utils import safe_str

logger = logging.getLogger(__name__)


def _valid_title(x: Any) -> bool:
    return isinstance(x, str) and bool(x.strip())


def _valid_summary(x: Any) -> bool:
    return isinstance(x, str) and bool(x.strip())


def _valid_rating(x: Any) -> bool:
    try:
        val = float(x)
    except Exception:
        return False
    return 0.0 <= val <= 10.0


def _valid_findings(x: Any) -> bool:
    if not isinstance(x, list) or not x:
        return False
    for item in x:
        if not isinstance(item, dict):
            return False
        if not safe_str(item.get("summary")).strip():
            return False
        if not safe_str(item.get("explanation")).strip():
            return False
    return True


class CommunityReportGenerator:
    """
    YAML-driven GraphRAG-style aggregation community report generator.

    Prompt YAML:
      - aggregation/generate_community_report

    Input params JSON:
      {
        "text": "...",
        "max_length": 1800,
        "max_findings": 6,
        "goal": "..."
      }

    Output JSON:
      {
        "title": "...",
        "summary": "...",
        "rating": 0.0,
        "rating_explanation": "...",
        "findings": [{"summary": "...", "explanation": "..."}]
      }
    """

    def __init__(
        self,
        prompt_loader,
        llm,
        prompt_id: str = "aggregation/generate_community_report",
    ):
        if llm is None:
            raise ValueError("llm must be provided")
        if prompt_loader is None:
            raise ValueError("prompt_loader must be provided")

        self.prompt_loader = prompt_loader
        self.llm = llm
        self.prompt_id = prompt_id
        self.required_fields = ["title", "summary", "rating", "rating_explanation", "findings"]
        self.field_validators = {
            "title": _valid_title,
            "summary": _valid_summary,
            "rating": _valid_rating,
            "rating_explanation": _valid_summary,
            "findings": _valid_findings,
        }
        self.repair_template = general_repair_template

    def call(self, params: str, **kwargs) -> str:
        try:
            params_dict = json.loads(params) if isinstance(params, str) else (params or {})
            text = safe_str(params_dict.get("text", "")).strip()
            max_length_raw = params_dict.get("max_length", 1800)
            max_findings_raw = params_dict.get("max_findings", 6)
            goal = safe_str(params_dict.get("goal", "")).strip()
        except Exception as e:
            logger.error("参数解析失败: %s", e)
            return correct_json_format(
                json.dumps(
                    {
                        "error": f"参数解析失败: {str(e)}",
                        "title": "",
                        "summary": "",
                        "rating": 0.0,
                        "rating_explanation": "",
                        "findings": [],
                    },
                    ensure_ascii=False,
                )
            )

        if not text:
            return correct_json_format(
                json.dumps(
                    {
                        "error": "missing required field: text",
                        "title": "",
                        "summary": "",
                        "rating": 0.0,
                        "rating_explanation": "",
                        "findings": [],
                    },
                    ensure_ascii=False,
                )
            )

        try:
            max_length = max(200, int(max_length_raw))
        except Exception:
            max_length = 1800

        try:
            max_findings = max(1, int(max_findings_raw))
        except Exception:
            max_findings = 6

        try:
            user_prompt = self.prompt_loader.render(
                self.prompt_id,
                static_values={},
                task_values={
                    "text": text,
                    "max_length": max_length,
                    "max_findings": max_findings,
                    "goal": goal,
                },
                strict=True,
            )
        except Exception as e:
            logger.error("generate_community_report prompt render failed: %s", e)
            return correct_json_format(
                json.dumps(
                    {
                        "error": f"prompt render failed: {str(e)}",
                        "title": "",
                        "summary": "",
                        "rating": 0.0,
                        "rating_explanation": "",
                        "findings": [],
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
                    "error": "社区报告生成失败",
                    "title": "",
                    "summary": text[: max_length].strip(),
                    "rating": 0.0,
                    "rating_explanation": "",
                    "findings": [
                        {
                            "summary": "上下文回退",
                            "explanation": text[: max_length].strip(),
                        }
                    ],
                },
                ensure_ascii=False,
            )
        )

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

from core.utils.format import correct_json_format
from core.utils.function_manager import process_with_format_guarantee
from core.utils.general_text import general_repair_template
from core.utils.general_utils import parse_json_object_from_text

logger = logging.getLogger(__name__)


class StrategyFailureSummarizer:
    def __init__(self, prompt_loader, llm, prompt_id: str = "memory/summarize_strategy_failures"):
        self.prompt_loader = prompt_loader
        self.llm = llm
        self.prompt_id = prompt_id

    def summarize(
        self,
        *,
        question: str,
        reference_answer: str,
        attempt_summaries: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        fallback = {
            "failure_summary": "all_attempts_failed",
            "likely_causes": ["judge_or_strategy_summarizer_unavailable"],
            "recommended_improvements": [],
        }
        if self.prompt_loader is None or self.llm is None:
            return fallback
        try:
            prompt = self.prompt_loader.render(
                self.prompt_id,
                static_values={},
                task_values={
                    "question": str(question or ""),
                    "reference_answer": str(reference_answer or ""),
                    "attempt_summaries_json": json.dumps(attempt_summaries or [], ensure_ascii=False, indent=2),
                },
                strict=True,
            )
        except Exception as exc:
            logger.warning("failure summary prompt render failed: %s", exc)
            return fallback
        try:
            corrected_json, status = process_with_format_guarantee(
                llm_client=self.llm,
                messages=[{"role": "user", "content": prompt}],
                required_fields=["failure_summary", "likely_causes", "recommended_improvements"],
                field_validators={
                    "failure_summary": lambda v: isinstance(v, str),
                    "likely_causes": lambda v: isinstance(v, list),
                    "recommended_improvements": lambda v: isinstance(v, list),
                },
                max_retries=2,
                repair_template=general_repair_template,
            )
        except Exception as exc:
            logger.warning("failure summary generation failed: %s", exc)
            return {
                "failure_summary": f"all_attempts_failed_due_to_error: {str(exc).strip()[:500]}",
                "likely_causes": ["failure_summarizer_error"],
                "recommended_improvements": [],
            }
        if status != "success":
            return fallback
        payload = parse_json_object_from_text(corrected_json) or {}
        return {
            "failure_summary": str(payload.get("failure_summary", "") or fallback["failure_summary"]),
            "likely_causes": [str(x).strip() for x in (payload.get("likely_causes") or []) if str(x).strip()],
            "recommended_improvements": [str(x).strip() for x in (payload.get("recommended_improvements") or []) if str(x).strip()],
        }


def summarize_strategy_failures_with_guard(
    *,
    llm,
    prompt_loader,
    question: str,
    reference_answer: str,
    attempt_summaries: List[Dict[str, Any]],
    prompt_id: str = "memory/summarize_strategy_failures",
) -> str:
    summarizer = StrategyFailureSummarizer(prompt_loader=prompt_loader, llm=llm, prompt_id=prompt_id)
    result = summarizer.summarize(
        question=question,
        reference_answer=reference_answer,
        attempt_summaries=attempt_summaries,
    )
    return correct_json_format(json.dumps(result, ensure_ascii=False))


class StrategyFailureSummarizerTool:
    def __init__(self, prompt_loader, llm, prompt_id: str = "memory/summarize_strategy_failures"):
        self.prompt_loader = prompt_loader
        self.llm = llm
        self.prompt_id = prompt_id

    def call(self, params: str, **kwargs) -> str:
        try:
            payload = json.loads(params) if isinstance(params, str) else (params or {})
        except Exception as exc:
            return correct_json_format(json.dumps({"error": f"params parse failed: {exc}"}, ensure_ascii=False))
        return summarize_strategy_failures_with_guard(
            llm=self.llm,
            prompt_loader=self.prompt_loader,
            question=str(payload.get("question", "") or ""),
            reference_answer=str(payload.get("reference_answer", "") or ""),
            attempt_summaries=payload.get("attempt_summaries") or [],
            prompt_id=self.prompt_id,
        )

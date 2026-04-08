from __future__ import annotations

import json
import logging
from typing import Any, Dict

from core.utils.format import correct_json_format
from core.utils.function_manager import process_with_format_guarantee
from core.utils.general_text import general_repair_template
from core.utils.general_utils import parse_json_object_from_text

logger = logging.getLogger(__name__)


def _clip_text(value: Any, limit: int) -> str:
    text = str(value or "").strip()
    if limit <= 0 or len(text) <= limit:
        return text
    marker = "\n...[truncated]...\n"
    head = max(0, int(limit * 0.7))
    tail = max(0, limit - head - len(marker))
    if tail <= 0:
        return text[:limit]
    return text[:head] + marker + text[-tail:]


class FailedAnswerReflector:
    def __init__(self, prompt_loader, llm, prompt_id: str = "memory/reflect_failed_answer"):
        self.prompt_loader = prompt_loader
        self.llm = llm
        self.prompt_id = prompt_id

    def reflect(
        self,
        *,
        question: str,
        reference_answer: str,
        candidate_answer: str,
        tool_summary_json: str,
    ) -> Dict[str, Any]:
        fallback = {
            "failure_reason": "reflection_unavailable",
            "missed_fact": "",
            "next_action": "",
            "need_retry": False,
        }
        if self.prompt_loader is None or self.llm is None:
            return fallback
        try:
            prompt = self.prompt_loader.render(
                self.prompt_id,
                static_values={},
                task_values={
                    "question": str(question or ""),
                    "reference_answer": _clip_text(reference_answer, 2400),
                    "candidate_answer": _clip_text(candidate_answer, 5000),
                    "tool_summary_json": _clip_text(tool_summary_json, 2400),
                },
                strict=True,
            )
        except Exception as exc:
            logger.warning("failed answer reflection prompt render failed: %s", exc)
            return fallback

        original_max_tokens = getattr(self.llm, "max_tokens", None)
        try:
            if isinstance(original_max_tokens, int) and original_max_tokens > 768:
                self.llm.max_tokens = 768
            corrected_json, status = process_with_format_guarantee(
                llm_client=self.llm,
                messages=[{"role": "user", "content": prompt}],
                required_fields=["failure_reason", "missed_fact", "next_action", "need_retry"],
                field_validators={
                    "failure_reason": lambda v: isinstance(v, str),
                    "missed_fact": lambda v: isinstance(v, str),
                    "next_action": lambda v: isinstance(v, str),
                    "need_retry": lambda v: isinstance(v, bool),
                },
                max_retries=2,
                repair_template=general_repair_template,
            )
        except Exception as exc:
            logger.warning("failed answer reflection failed: %s", exc)
            return {
                **fallback,
                "failure_reason": f"reflection_error: {str(exc).strip()[:300]}",
            }
        finally:
            if original_max_tokens is not None:
                self.llm.max_tokens = original_max_tokens

        if status != "success":
            return fallback
        payload = parse_json_object_from_text(corrected_json) or {}
        return {
            "failure_reason": str(payload.get("failure_reason", "") or fallback["failure_reason"]).strip(),
            "missed_fact": str(payload.get("missed_fact", "") or "").strip(),
            "next_action": str(payload.get("next_action", "") or "").strip(),
            "need_retry": bool(payload.get("need_retry", False)),
        }


class RetryInstructionBuilder:
    def __init__(self, prompt_loader, llm, prompt_id: str = "memory/build_retry_instruction"):
        self.prompt_loader = prompt_loader
        self.llm = llm
        self.prompt_id = prompt_id

    def build(
        self,
        *,
        question: str,
        missed_fact: str,
        next_action: str,
    ) -> Dict[str, Any]:
        fallback = {"retry_instruction": ""}
        if self.prompt_loader is None or self.llm is None:
            return fallback
        try:
            prompt = self.prompt_loader.render(
                self.prompt_id,
                static_values={},
                task_values={
                    "question": str(question or ""),
                    "missed_fact": _clip_text(missed_fact, 800),
                    "next_action": _clip_text(next_action, 800),
                },
                strict=True,
            )
        except Exception as exc:
            logger.warning("retry instruction prompt render failed: %s", exc)
            return fallback

        original_max_tokens = getattr(self.llm, "max_tokens", None)
        try:
            if isinstance(original_max_tokens, int) and original_max_tokens > 512:
                self.llm.max_tokens = 512
            corrected_json, status = process_with_format_guarantee(
                llm_client=self.llm,
                messages=[{"role": "user", "content": prompt}],
                required_fields=["retry_instruction"],
                field_validators={"retry_instruction": lambda v: isinstance(v, str)},
                max_retries=2,
                repair_template=general_repair_template,
            )
        except Exception as exc:
            logger.warning("retry instruction build failed: %s", exc)
            return fallback
        finally:
            if original_max_tokens is not None:
                self.llm.max_tokens = original_max_tokens

        if status != "success":
            return fallback
        payload = parse_json_object_from_text(corrected_json) or {}
        return {
            "retry_instruction": str(payload.get("retry_instruction", "") or "").strip(),
        }


def reflect_failed_answer_with_guard(
    *,
    llm,
    prompt_loader,
    question: str,
    reference_answer: str,
    candidate_answer: str,
    tool_summary_json: str,
    prompt_id: str = "memory/reflect_failed_answer",
) -> str:
    reflector = FailedAnswerReflector(prompt_loader=prompt_loader, llm=llm, prompt_id=prompt_id)
    result = reflector.reflect(
        question=question,
        reference_answer=reference_answer,
        candidate_answer=candidate_answer,
        tool_summary_json=tool_summary_json,
    )
    return correct_json_format(json.dumps(result, ensure_ascii=False))


def build_retry_instruction_with_guard(
    *,
    llm,
    prompt_loader,
    question: str,
    missed_fact: str,
    next_action: str,
    prompt_id: str = "memory/build_retry_instruction",
) -> str:
    builder = RetryInstructionBuilder(prompt_loader=prompt_loader, llm=llm, prompt_id=prompt_id)
    result = builder.build(
        question=question,
        missed_fact=missed_fact,
        next_action=next_action,
    )
    return correct_json_format(json.dumps(result, ensure_ascii=False))

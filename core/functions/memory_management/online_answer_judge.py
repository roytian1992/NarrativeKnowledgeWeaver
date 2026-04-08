from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

from core.functions.memory_management.judge_retrieval_answer import RetrievalAnswerJudge
from core.utils.format import correct_json_format
from core.utils.function_manager import process_with_format_guarantee
from core.utils.general_text import general_repair_template
from core.utils.general_utils import parse_json_object_from_text

logger = logging.getLogger(__name__)


def _clamp_score(value: Any) -> float:
    try:
        score = float(value)
    except Exception:
        return 0.0
    return max(0.0, min(1.0, score))


def _clip_text(value: Any, limit: int) -> str:
    text = str(value or "").strip()
    if limit <= 0 or len(text) <= limit:
        return text
    marker = "\n...[truncated for online judge]...\n"
    head = max(0, int(limit * 0.7))
    tail = max(0, limit - head - len(marker))
    if tail <= 0:
        return text[:limit]
    return text[:head] + marker + text[-tail:]


class OnlineAnswerJudge:
    def __init__(
        self,
        prompt_loader,
        llm,
        *,
        prompt_id: str = "memory/judge_online_answer",
        reference_prompt_id: str = "memory/judge_retrieval_answer",
    ) -> None:
        self.prompt_loader = prompt_loader
        self.llm = llm
        self.prompt_id = prompt_id
        self.reference_judge = RetrievalAnswerJudge(
            prompt_loader=prompt_loader,
            llm=llm,
            prompt_id=reference_prompt_id,
        )

    def evaluate(
        self,
        *,
        question: str,
        candidate_answer: str,
        reference_answer: str = "",
        evidence_summary: str = "",
        tool_summary_json: str = "",
    ) -> Dict[str, Any]:
        ref = str(reference_answer or "").strip()
        if ref:
            payload = self.reference_judge.evaluate(
                question=question,
                reference_answer=ref,
                candidate_answer=candidate_answer,
            )
            return {
                "mode": "reference",
                "is_success": bool(payload.get("is_correct", False)),
                "score": float(payload.get("score", 0.0) or 0.0),
                "reason": str(payload.get("reason", "") or "").strip(),
                "answer_support_score": float(payload.get("score", 0.0) or 0.0),
                "evidence_specificity_score": float(payload.get("score", 0.0) or 0.0),
                "evidence_coverage_score": float(payload.get("score", 0.0) or 0.0),
                "intermediate_value_score": 1.0 if bool(payload.get("is_correct", False)) else 0.0,
                "trace_efficiency_score": 0.5,
                "matched_points": [str(x).strip() for x in (payload.get("matched_points") or []) if str(x).strip()],
                "missing_points": [str(x).strip() for x in (payload.get("missing_points") or []) if str(x).strip()],
                "hallucination_points": [str(x).strip() for x in (payload.get("hallucination_points") or []) if str(x).strip()],
                "supported_points": [str(x).strip() for x in (payload.get("matched_points") or []) if str(x).strip()],
                "unsupported_points": [str(x).strip() for x in (payload.get("hallucination_points") or []) if str(x).strip()],
                "needs_human_review": False,
            }

        fallback = {
            "mode": "evidence_consistency",
            "is_success": False,
            "score": 0.0,
            "reason": "online_judge_unavailable",
            "answer_support_score": 0.0,
            "evidence_specificity_score": 0.0,
            "evidence_coverage_score": 0.0,
            "intermediate_value_score": 0.0,
            "trace_efficiency_score": 0.0,
            "matched_points": [],
            "missing_points": [],
            "hallucination_points": [],
            "supported_points": [],
            "unsupported_points": [],
            "needs_human_review": True,
        }
        if self.prompt_loader is None or self.llm is None:
            return fallback
        try:
            prompt = self.prompt_loader.render(
                self.prompt_id,
                static_values={},
                task_values={
                    "question": _clip_text(question, 1500),
                    "candidate_answer": _clip_text(candidate_answer, 6000),
                    "evidence_summary": _clip_text(evidence_summary, 5000),
                    "tool_summary_json": _clip_text(tool_summary_json, 5000),
                },
                strict=True,
            )
        except Exception as exc:
            logger.warning("online judge prompt render failed: %s", exc)
            return fallback

        original_max_tokens = getattr(self.llm, "max_tokens", None)
        try:
            if isinstance(original_max_tokens, int) and original_max_tokens > 1024:
                self.llm.max_tokens = 1024
            corrected_json, status = process_with_format_guarantee(
                llm_client=self.llm,
                messages=[{"role": "user", "content": prompt}],
                required_fields=[
                    "is_success",
                    "score",
                    "reason",
                    "answer_support_score",
                    "evidence_specificity_score",
                    "evidence_coverage_score",
                    "intermediate_value_score",
                    "trace_efficiency_score",
                    "supported_points",
                    "unsupported_points",
                    "missing_points",
                    "needs_human_review",
                ],
                field_validators={
                    "is_success": lambda v: isinstance(v, bool),
                    "score": lambda v: isinstance(v, (int, float)),
                    "reason": lambda v: isinstance(v, str),
                    "answer_support_score": lambda v: isinstance(v, (int, float)),
                    "evidence_specificity_score": lambda v: isinstance(v, (int, float)),
                    "evidence_coverage_score": lambda v: isinstance(v, (int, float)),
                    "intermediate_value_score": lambda v: isinstance(v, (int, float)),
                    "trace_efficiency_score": lambda v: isinstance(v, (int, float)),
                    "supported_points": lambda v: isinstance(v, list),
                    "unsupported_points": lambda v: isinstance(v, list),
                    "missing_points": lambda v: isinstance(v, list),
                    "needs_human_review": lambda v: isinstance(v, bool),
                },
                max_retries=2,
                repair_template=general_repair_template,
            )
        except Exception as exc:
            logger.warning("online judge failed: %s", exc)
            return {
                **fallback,
                "reason": f"online_judge_error: {str(exc).strip()[:300]}",
            }
        finally:
            if original_max_tokens is not None:
                self.llm.max_tokens = original_max_tokens
        if status != "success":
            return fallback

        payload = parse_json_object_from_text(corrected_json) or {}
        supported_points = [str(x).strip() for x in (payload.get("supported_points") or []) if str(x).strip()]
        unsupported_points = [str(x).strip() for x in (payload.get("unsupported_points") or []) if str(x).strip()]
        missing_points = [str(x).strip() for x in (payload.get("missing_points") or []) if str(x).strip()]
        answer_support_score = _clamp_score(payload.get("answer_support_score", 0.0))
        evidence_specificity_score = _clamp_score(payload.get("evidence_specificity_score", 0.0))
        evidence_coverage_score = _clamp_score(payload.get("evidence_coverage_score", 0.0))
        intermediate_value_score = _clamp_score(payload.get("intermediate_value_score", 0.0))
        trace_efficiency_score = _clamp_score(payload.get("trace_efficiency_score", 0.0))
        computed_score = round(
            0.35 * answer_support_score
            + 0.20 * evidence_specificity_score
            + 0.20 * evidence_coverage_score
            + 0.15 * intermediate_value_score
            + 0.10 * trace_efficiency_score,
            6,
        )
        is_success = bool(payload.get("is_success", False))
        if answer_support_score < 0.6 or evidence_coverage_score < 0.5:
            is_success = False
        return {
            "mode": "evidence_consistency",
            "is_success": is_success,
            "score": computed_score,
            "reason": str(payload.get("reason", "") or "").strip(),
            "answer_support_score": answer_support_score,
            "evidence_specificity_score": evidence_specificity_score,
            "evidence_coverage_score": evidence_coverage_score,
            "intermediate_value_score": intermediate_value_score,
            "trace_efficiency_score": trace_efficiency_score,
            "matched_points": supported_points,
            "missing_points": missing_points,
            "hallucination_points": unsupported_points,
            "supported_points": supported_points,
            "unsupported_points": unsupported_points,
            "needs_human_review": bool(payload.get("needs_human_review", False)),
        }


def judge_online_answer_with_guard(
    *,
    llm,
    prompt_loader,
    question: str,
    candidate_answer: str,
    reference_answer: str = "",
    evidence_summary: str = "",
    tool_summary_json: str = "",
    prompt_id: str = "memory/judge_online_answer",
) -> str:
    judge = OnlineAnswerJudge(prompt_loader=prompt_loader, llm=llm, prompt_id=prompt_id)
    result = judge.evaluate(
        question=question,
        candidate_answer=candidate_answer,
        reference_answer=reference_answer,
        evidence_summary=evidence_summary,
        tool_summary_json=tool_summary_json,
    )
    return correct_json_format(json.dumps(result, ensure_ascii=False))

from __future__ import annotations

import json
import logging
from typing import Any, Dict

from core.utils.format import correct_json_format
from core.utils.function_manager import process_with_format_guarantee
from core.utils.general_text import general_repair_template
from core.utils.general_utils import parse_json_object_from_text

logger = logging.getLogger(__name__)


def _clip_for_judge(value: Any, limit: int) -> str:
    text = str(value or "").strip()
    if limit <= 0 or len(text) <= limit:
        return text
    marker = "\n...[truncated for judge]...\n"
    head = max(0, int(limit * 0.7))
    tail = max(0, limit - head - len(marker))
    if tail <= 0:
        return text[:limit]
    return text[:head] + marker + text[-tail:]


class RetrievalAnswerJudge:
    def __init__(self, prompt_loader, llm, prompt_id: str = "memory/judge_retrieval_answer"):
        self.prompt_loader = prompt_loader
        self.llm = llm
        self.prompt_id = prompt_id

    def evaluate(
        self,
        *,
        question: str,
        reference_answer: str,
        candidate_answer: str,
    ) -> Dict[str, Any]:
        fallback = {
            "is_correct": False,
            "score": 0.0,
            "reason": "judge_unavailable",
            "matched_points": [],
            "missing_points": [],
            "hallucination_points": [],
        }
        if self.prompt_loader is None or self.llm is None:
            return fallback
        try:
            prompt = self.prompt_loader.render(
                self.prompt_id,
                static_values={},
                task_values={
                    "question": str(question or ""),
                    "reference_answer": _clip_for_judge(reference_answer, limit=3000),
                    "candidate_answer": _clip_for_judge(candidate_answer, limit=7000),
                },
                strict=True,
            )
        except Exception as exc:
            logger.warning("judge prompt render failed: %s", exc)
            return fallback

        original_max_tokens = getattr(self.llm, "max_tokens", None)
        try:
            if isinstance(original_max_tokens, int) and original_max_tokens > 1024:
                self.llm.max_tokens = 1024
            corrected_json, status = process_with_format_guarantee(
                llm_client=self.llm,
                messages=[{"role": "user", "content": prompt}],
                required_fields=["is_correct", "score", "reason", "matched_points", "missing_points", "hallucination_points"],
                field_validators={
                    "is_correct": lambda v: isinstance(v, bool),
                    "score": lambda v: isinstance(v, (int, float)),
                    "reason": lambda v: isinstance(v, str),
                    "matched_points": lambda v: isinstance(v, list),
                    "missing_points": lambda v: isinstance(v, list),
                    "hallucination_points": lambda v: isinstance(v, list),
                },
                max_retries=2,
                repair_template=general_repair_template,
            )
        except Exception as exc:
            logger.warning("judge evaluation failed: %s", exc)
            return {
                **fallback,
                "reason": f"judge_error: {str(exc).strip()[:500]}",
            }
        finally:
            if original_max_tokens is not None:
                self.llm.max_tokens = original_max_tokens
        if status != "success":
            return fallback
        payload = parse_json_object_from_text(corrected_json) or {}
        try:
            return {
                "is_correct": bool(payload.get("is_correct", False)),
                "score": float(payload.get("score", 0.0) or 0.0),
                "reason": str(payload.get("reason", "") or "judge_no_reason"),
                "matched_points": [str(x).strip() for x in (payload.get("matched_points") or []) if str(x).strip()],
                "missing_points": [str(x).strip() for x in (payload.get("missing_points") or []) if str(x).strip()],
                "hallucination_points": [str(x).strip() for x in (payload.get("hallucination_points") or []) if str(x).strip()],
            }
        except Exception:
            return fallback


def judge_retrieval_answer_with_guard(
    *,
    llm,
    prompt_loader,
    question: str,
    reference_answer: str,
    candidate_answer: str,
    prompt_id: str = "memory/judge_retrieval_answer",
) -> str:
    judge = RetrievalAnswerJudge(prompt_loader=prompt_loader, llm=llm, prompt_id=prompt_id)
    result = judge.evaluate(
        question=question,
        reference_answer=reference_answer,
        candidate_answer=candidate_answer,
    )
    return correct_json_format(json.dumps(result, ensure_ascii=False))


class RetrievalAnswerJudgeTool:
    def __init__(self, prompt_loader, llm, prompt_id: str = "memory/judge_retrieval_answer"):
        self.prompt_loader = prompt_loader
        self.llm = llm
        self.prompt_id = prompt_id

    def call(self, params: str, **kwargs) -> str:
        try:
            payload = json.loads(params) if isinstance(params, str) else (params or {})
        except Exception as exc:
            return correct_json_format(json.dumps({"error": f"params parse failed: {exc}"}, ensure_ascii=False))
        return judge_retrieval_answer_with_guard(
            llm=self.llm,
            prompt_loader=self.prompt_loader,
            question=str(payload.get("question", "") or ""),
            reference_answer=str(payload.get("reference_answer", "") or ""),
            candidate_answer=str(payload.get("candidate_answer", "") or ""),
            prompt_id=self.prompt_id,
        )

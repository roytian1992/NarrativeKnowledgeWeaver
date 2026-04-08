from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

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


class CandidateAnswerSelector:
    def __init__(self, prompt_loader, llm, prompt_id: str = "memory/select_candidate_answers"):
        self.prompt_loader = prompt_loader
        self.llm = llm
        self.prompt_id = prompt_id

    def select(
        self,
        *,
        question: str,
        query_abstract: str,
        candidate_answers: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        fallback = {
            "selected_candidate_ids": [str(candidate_answers[0].get("candidate_id", "candidate_1"))] if candidate_answers else [],
            "final_answer": str(candidate_answers[0].get("answer", "") or "").strip() if candidate_answers else "",
            "reason": "selector_unavailable",
        }
        if self.prompt_loader is None or self.llm is None or not candidate_answers:
            return fallback
        try:
            prompt = self.prompt_loader.render(
                self.prompt_id,
                static_values={},
                task_values={
                    "question": str(question or "").strip(),
                    "query_abstract": str(query_abstract or "").strip(),
                    "candidate_answers_json": _clip_text(
                        json.dumps(candidate_answers, ensure_ascii=False, indent=2),
                        12000,
                    ),
                },
                strict=True,
            )
        except Exception as exc:
            logger.warning("candidate answer selector prompt render failed: %s", exc)
            return fallback

        original_max_tokens = getattr(self.llm, "max_tokens", None)
        try:
            if isinstance(original_max_tokens, int) and original_max_tokens > 1536:
                self.llm.max_tokens = 1536
            corrected_json, status = process_with_format_guarantee(
                llm_client=self.llm,
                messages=[{"role": "user", "content": prompt}],
                required_fields=["selected_candidate_ids", "final_answer", "reason"],
                field_validators={
                    "selected_candidate_ids": lambda v: isinstance(v, list),
                    "final_answer": lambda v: isinstance(v, str),
                    "reason": lambda v: isinstance(v, str),
                },
                max_retries=2,
                repair_template=general_repair_template,
            )
        except Exception as exc:
            logger.warning("candidate answer selector failed: %s", exc)
            return {
                **fallback,
                "reason": f"selector_error: {str(exc).strip()[:300]}",
            }
        finally:
            if original_max_tokens is not None:
                self.llm.max_tokens = original_max_tokens

        if status != "success":
            return fallback
        payload = parse_json_object_from_text(corrected_json) or {}
        selected_candidate_ids = [
            str(x).strip()
            for x in (payload.get("selected_candidate_ids") or [])
            if str(x).strip()
        ]
        final_answer = str(payload.get("final_answer", "") or "").strip()
        reason = str(payload.get("reason", "") or "").strip() or fallback["reason"]
        if not selected_candidate_ids:
            selected_candidate_ids = fallback["selected_candidate_ids"]
        if not final_answer:
            final_answer = fallback["final_answer"]
        return {
            "selected_candidate_ids": selected_candidate_ids,
            "final_answer": final_answer,
            "reason": reason,
        }


def select_candidate_answers_with_guard(
    *,
    llm,
    prompt_loader,
    question: str,
    query_abstract: str,
    candidate_answers: List[Dict[str, Any]],
    prompt_id: str = "memory/select_candidate_answers",
) -> str:
    selector = CandidateAnswerSelector(prompt_loader=prompt_loader, llm=llm, prompt_id=prompt_id)
    result = selector.select(
        question=question,
        query_abstract=query_abstract,
        candidate_answers=candidate_answers,
    )
    return correct_json_format(json.dumps(result, ensure_ascii=False))


class CandidateAnswerSelectorTool:
    def __init__(self, prompt_loader, llm, prompt_id: str = "memory/select_candidate_answers"):
        self.prompt_loader = prompt_loader
        self.llm = llm
        self.prompt_id = prompt_id

    def call(self, params: str, **kwargs) -> str:
        try:
            payload = json.loads(params) if isinstance(params, str) else (params or {})
        except Exception as exc:
            return correct_json_format(json.dumps({"error": f"params parse failed: {exc}"}, ensure_ascii=False))
        return select_candidate_answers_with_guard(
            llm=self.llm,
            prompt_loader=self.prompt_loader,
            question=str(payload.get("question", "") or ""),
            query_abstract=str(payload.get("query_abstract", "") or ""),
            candidate_answers=payload.get("candidate_answers") or [],
            prompt_id=self.prompt_id,
        )

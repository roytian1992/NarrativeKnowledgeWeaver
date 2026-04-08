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
    marker = "\n...[truncated for bootstrap]...\n"
    head = max(0, int(limit * 0.7))
    tail = max(0, limit - head - len(marker))
    if tail <= 0:
        return text[:limit]
    return text[:head] + marker + text[-tail:]


class SelfBootstrapQAGenerator:
    def __init__(self, prompt_loader, llm, prompt_id: str = "memory/generate_self_bootstrap_qa") -> None:
        self.prompt_loader = prompt_loader
        self.llm = llm
        self.prompt_id = prompt_id

    def generate(
        self,
        *,
        question: str,
        final_answer: str,
        evidence_summary: str,
        tool_summary_json: str,
        max_questions: int = 3,
    ) -> List[Dict[str, Any]]:
        if max_questions <= 0:
            return []
        if self.prompt_loader is None or self.llm is None:
            return []
        try:
            prompt = self.prompt_loader.render(
                self.prompt_id,
                static_values={},
                task_values={
                    "question": _clip_text(question, 1200),
                    "final_answer": _clip_text(final_answer, 3000),
                    "evidence_summary": _clip_text(evidence_summary, 4500),
                    "tool_summary_json": _clip_text(tool_summary_json, 4000),
                    "max_questions": str(max_questions),
                },
                strict=True,
            )
        except Exception as exc:
            logger.warning("self bootstrap prompt render failed: %s", exc)
            return []

        original_max_tokens = getattr(self.llm, "max_tokens", None)
        try:
            if isinstance(original_max_tokens, int) and original_max_tokens > 1536:
                self.llm.max_tokens = 1536
            corrected_json, status = process_with_format_guarantee(
                llm_client=self.llm,
                messages=[{"role": "user", "content": prompt}],
                required_fields=["qa_pairs"],
                field_validators={"qa_pairs": lambda v: isinstance(v, list)},
                max_retries=2,
                repair_template=general_repair_template,
            )
        except Exception as exc:
            logger.warning("self bootstrap generation failed: %s", exc)
            return []
        finally:
            if original_max_tokens is not None:
                self.llm.max_tokens = original_max_tokens
        if status != "success":
            return []

        payload = parse_json_object_from_text(corrected_json) or {}
        rows: List[Dict[str, Any]] = []
        seen_questions: set[str] = set()
        for item in (payload.get("qa_pairs") or []):
            if not isinstance(item, dict):
                continue
            qa_question = str(item.get("question", "") or "").strip()
            qa_answer = str(item.get("answer", "") or "").strip()
            qa_evidence = str(item.get("evidence", "") or "").strip()
            qa_type = str(item.get("question_type", "") or "").strip()
            difficulty = str(item.get("difficulty", "") or "").strip()
            if not qa_question or not qa_answer:
                continue
            key = qa_question.lower()
            if key in seen_questions:
                continue
            seen_questions.add(key)
            rows.append(
                {
                    "question": qa_question,
                    "answer": qa_answer,
                    "evidence": qa_evidence,
                    "question_type": qa_type,
                    "difficulty": difficulty,
                }
            )
            if len(rows) >= max_questions:
                break
        return rows


def generate_self_bootstrap_qa_with_guard(
    *,
    llm,
    prompt_loader,
    question: str,
    final_answer: str,
    evidence_summary: str,
    tool_summary_json: str,
    max_questions: int = 3,
    prompt_id: str = "memory/generate_self_bootstrap_qa",
) -> str:
    generator = SelfBootstrapQAGenerator(prompt_loader=prompt_loader, llm=llm, prompt_id=prompt_id)
    rows = generator.generate(
        question=question,
        final_answer=final_answer,
        evidence_summary=evidence_summary,
        tool_summary_json=tool_summary_json,
        max_questions=max_questions,
    )
    return correct_json_format(json.dumps({"qa_pairs": rows}, ensure_ascii=False))


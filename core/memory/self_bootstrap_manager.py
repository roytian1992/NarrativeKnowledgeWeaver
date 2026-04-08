from __future__ import annotations

from typing import Any, Callable, Dict, List


class SelfBootstrapManager:
    def __init__(
        self,
        *,
        generator,
        judge,
        min_accept_score: float = 0.9,
    ) -> None:
        self.generator = generator
        self.judge = judge
        self.min_accept_score = float(min_accept_score or 0.9)

    def run(
        self,
        *,
        question: str,
        final_answer: str,
        evidence_summary: str,
        tool_summary_json: str,
        answer_fn: Callable[[str], Dict[str, Any]],
        max_questions: int = 3,
        sampling_attempts: int = 1,
    ) -> List[Dict[str, Any]]:
        rows = self.generator.generate(
            question=question,
            final_answer=final_answer,
            evidence_summary=evidence_summary,
            tool_summary_json=tool_summary_json,
            max_questions=max_questions,
        )
        out: List[Dict[str, Any]] = []
        for idx, row in enumerate(rows, start=1):
            qa_question = str(row.get("question", "") or "").strip()
            qa_answer = str(row.get("answer", "") or "").strip()
            if not qa_question or not qa_answer:
                continue
            attempt_records: List[Dict[str, Any]] = []
            for _ in range(max(1, int(sampling_attempts or 1))):
                attempt = answer_fn(qa_question) or {}
                candidate_answer = str(attempt.get("answer", "") or "").strip()
                judge = self.judge.evaluate(
                    question=qa_question,
                    candidate_answer=candidate_answer,
                    reference_answer=qa_answer,
                    evidence_summary=str(row.get("evidence", "") or "").strip(),
                    tool_summary_json=str(attempt.get("tool_summary_json", "") or ""),
                )
                attempt_records.append(
                    {
                        "candidate_answer": candidate_answer,
                        "tool_uses": attempt.get("tool_uses") or [],
                        "tool_summary_json": str(attempt.get("tool_summary_json", "") or ""),
                        "judge": judge,
                    }
                )
            correct_attempts = [
                item for item in attempt_records
                if bool((item.get("judge") or {}).get("is_success", False))
            ]
            if correct_attempts:
                correct_attempts.sort(
                    key=lambda item: (
                        len(str(item.get("candidate_answer", "") or "").strip()),
                        len(item.get("tool_uses") or []),
                    )
                )
                selected = correct_attempts[0]
            else:
                attempt_records.sort(
                    key=lambda item: (
                        float(((item.get("judge") or {}).get("score", 0.0)) or 0.0),
                        -len(str(item.get("candidate_answer", "") or "").strip()),
                    ),
                    reverse=True,
                )
                selected = attempt_records[0]
            candidate_answer = str(selected.get("candidate_answer", "") or "").strip()
            judge = dict(selected.get("judge") or {})
            out.append(
                {
                    "bootstrap_index": idx,
                    "question": qa_question,
                    "reference_answer": qa_answer,
                    "generated_evidence": str(row.get("evidence", "") or "").strip(),
                    "question_type": str(row.get("question_type", "") or "").strip(),
                    "difficulty": str(row.get("difficulty", "") or "").strip(),
                    "agent_answer": candidate_answer,
                    "judge": judge,
                    "accepted": bool(judge.get("is_success", False))
                    and float(judge.get("score", 0.0) or 0.0) >= self.min_accept_score,
                    "tool_uses": selected.get("tool_uses") or [],
                    "sampling_attempt_count": len(attempt_records),
                    "correct_attempt_count": len(correct_attempts),
                }
            )
        return out

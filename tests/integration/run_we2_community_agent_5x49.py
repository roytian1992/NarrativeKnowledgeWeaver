from __future__ import annotations

import copy
import csv
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

from core.agent.retriever_agent import QuestionAnsweringAgent
from core.functions.memory_management.judge_retrieval_answer import RetrievalAnswerJudge
from core.model_providers.openai_llm import OpenAILLM
from core.utils.config import KAGConfig
from core.utils.prompt_loader import YAMLPromptLoader


CSV_PATH = REPO_ROOT / "examples" / "datasets" / "we2_qa.csv"
REPORT_DIR = REPO_ROOT / "reports"
JSON_PATH = REPORT_DIR / "we2_community_agent_5x49_20260316.json"
MD_PATH = REPORT_DIR / "we2_community_agent_5x49_20260316.md"
REPEATS = 5
MAX_WORKERS = 4


@dataclass
class EvalResult:
    run_index: int
    question_id: str
    question_index: str
    question_type: str
    question: str
    reference_answer: str
    final_answer: str
    judge: Dict[str, Any]
    latency_ms: int
    extra: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "setting": "community_agent_no_strategy",
            "run_index": self.run_index,
            "question_id": self.question_id,
            "question_index": self.question_index,
            "question_type": self.question_type,
            "question": self.question,
            "reference_answer": self.reference_answer,
            "final_answer": self.final_answer,
            "judge": self.judge,
            "latency_ms": self.latency_ms,
            **self.extra,
        }


def _truncate(text: Any, limit: int = 500) -> str:
    raw = str(text or "").strip()
    if len(raw) <= limit:
        return raw
    return raw[: max(0, limit - 3)] + "..."


def _load_rows() -> List[Dict[str, str]]:
    with CSV_PATH.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    out: List[Dict[str, str]] = []
    for idx, row in enumerate(rows):
        question = str((row or {}).get("question", "") or "").strip()
        answer = str((row or {}).get("answer", "") or "").strip()
        if not question or not answer:
            continue
        out.append(
            {
                "question_id": f"q{idx}",
                "question_index": str((row or {}).get("question_index", idx) or idx),
                "question_type": str((row or {}).get("question_type", "") or ""),
                "question": question,
                "answer": answer,
            }
        )
    return out


class AgentThreadLocal:
    def __init__(self, cfg: KAGConfig) -> None:
        self.cfg = cfg
        self.local = threading.local()
        self._states: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

    def _build_state(self) -> Dict[str, Any]:
        cfg = copy.deepcopy(self.cfg)
        cfg.global_config.aggregation_mode = "community"
        if hasattr(cfg, "global_"):
            cfg.global_.aggregation_mode = "community"
        cfg.strategy_memory.enabled = False
        cfg.strategy_memory.read_enabled = False
        prompt_loader = YAMLPromptLoader(cfg.global_config.prompt_dir)
        llm = OpenAILLM(cfg, llm_profile="memory")
        judge = RetrievalAnswerJudge(prompt_loader, llm)
        agent = QuestionAnsweringAgent(
            cfg,
            aggregation_mode="community",
            enable_sql_tools=False,
        )
        state = {
            "cfg": cfg,
            "judge": judge,
            "agent": agent,
        }
        with self._lock:
            self._states.append(state)
        return state

    def state(self) -> Dict[str, Any]:
        state = getattr(self.local, "state", None)
        if state is None:
            state = self._build_state()
            self.local.state = state
        return state

    def close(self) -> None:
        for state in self._states:
            try:
                state["agent"].close()
            except Exception:
                pass


class CommunityAgentEvaluator:
    def __init__(self, cfg: KAGConfig) -> None:
        self.tlocal = AgentThreadLocal(cfg)

    def evaluate_row(self, row: Dict[str, str], run_index: int) -> EvalResult:
        state = self.tlocal.state()
        started = time.time()
        responses: List[Dict[str, Any]] = []
        final_answer = ""
        tool_uses: List[Dict[str, Any]] = []
        error_text = ""

        for max_calls in (10, 6, 4):
            try:
                responses = state["agent"].ask(
                    row["question"],
                    lang=str(state["cfg"].global_config.language or "zh"),
                    session_id=f"community_no_strategy_{row['question_id']}_run_{run_index}_{max_calls}",
                    max_llm_calls_per_run=max_calls,
                )
                final_answer = state["agent"].extract_final_text(responses)
                tool_uses = state["agent"].extract_tool_uses(responses)
                error_text = ""
                break
            except Exception as exc:
                error_text = f"{type(exc).__name__}: {exc}"
                if "maximum context length" not in error_text.lower():
                    break

        if error_text:
            judge = {
                "is_correct": False,
                "score": 0.0,
                "reason": f"agent_error: {error_text}",
                "matched_points": [],
                "missing_points": [],
                "hallucination_points": [],
            }
        else:
            judge = state["judge"].evaluate(
                question=row["question"],
                reference_answer=row["answer"],
                candidate_answer=final_answer,
            )

        return EvalResult(
            run_index=run_index,
            question_id=row["question_id"],
            question_index=row["question_index"],
            question_type=row["question_type"],
            question=row["question"],
            reference_answer=row["answer"],
            final_answer=final_answer,
            judge=judge,
            latency_ms=int((time.time() - started) * 1000),
            extra={
                "tool_call_count": len(tool_uses),
                "tool_names": [
                    str((item or {}).get("tool_name", "") or "").strip()
                    for item in tool_uses
                    if str((item or {}).get("tool_name", "") or "").strip()
                ],
                "tool_uses": tool_uses,
                "strategy_context": state["agent"].get_last_strategy_context(),
                "error": error_text,
            },
        )

    def close(self) -> None:
        self.tlocal.close()


def _load_cached_results() -> List[Dict[str, Any]]:
    if not JSON_PATH.exists():
        return []
    try:
        payload = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    except Exception:
        return []
    rows = payload.get("results")
    return [item for item in rows if isinstance(item, dict)] if isinstance(rows, list) else []


def _summarize(rows: List[Dict[str, Any]], question_rows: List[Dict[str, str]]) -> Dict[str, Any]:
    total_attempts = len(rows)
    correct_attempts = sum(
        1
        for item in rows
        if bool(((item.get("judge") or {}) if isinstance(item.get("judge"), dict) else {}).get("is_correct", False))
    )
    avg_score = round(
        sum(float(((item.get("judge") or {}) if isinstance(item.get("judge"), dict) else {}).get("score", 0.0) or 0.0) for item in rows)
        / float(total_attempts or 1),
        4,
    )
    avg_latency = int(sum(int(item.get("latency_ms", 0) or 0) for item in rows) / float(total_attempts or 1))

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for item in rows:
        grouped.setdefault(str(item.get("question_id", "") or ""), []).append(item)
    pass_questions = 0
    for qrow in question_rows:
        attempts = grouped.get(qrow["question_id"], [])
        if any(
            bool(((item.get("judge") or {}) if isinstance(item.get("judge"), dict) else {}).get("is_correct", False))
            for item in attempts
        ):
            pass_questions += 1

    return {
        "setting": "community_agent_no_strategy",
        "repeat_count": REPEATS,
        "question_count": len(question_rows),
        "total_attempts": total_attempts,
        "correct_attempts": correct_attempts,
        "overall_accuracy": round(correct_attempts / float(total_attempts or 1), 4),
        "five_pass_correct": pass_questions,
        "five_pass_accuracy": round(pass_questions / float(len(question_rows) or 1), 4),
        "avg_judge_score": avg_score,
        "avg_latency_ms": avg_latency,
    }


def _build_question_summary(question_rows: List[Dict[str, str]], rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for item in rows:
        grouped.setdefault(str(item.get("question_id", "") or ""), []).append(item)

    out: List[Dict[str, Any]] = []
    for base in question_rows:
        attempts = sorted(grouped.get(base["question_id"], []), key=lambda item: int(item.get("run_index", 0) or 0))
        correct_count = sum(
            1
            for item in attempts
            if bool(((item.get("judge") or {}) if isinstance(item.get("judge"), dict) else {}).get("is_correct", False))
        )
        out.append(
            {
                "question_id": base["question_id"],
                "question": base["question"],
                "reference_answer": base["answer"],
                "correct_count": correct_count,
                "pass_5": bool(correct_count > 0),
                "sample_answer": _truncate(str((attempts[0] if attempts else {}).get("final_answer", "") or ""), 500),
                "sample_tool_names": list((attempts[0] if attempts else {}).get("tool_names", []) or []),
            }
        )
    return out


def _flush_payload(question_rows: List[Dict[str, str]], rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": str(CSV_PATH),
        "aggregation_mode": "community",
        "strategy_enabled": False,
        "repeats": REPEATS,
        "summary": _summarize(rows, question_rows),
        "question_summary": _build_question_summary(question_rows, rows),
        "results": sorted(rows, key=lambda item: (int(item.get("question_index", 0) or 0), int(item.get("run_index", 0) or 0))),
    }
    JSON_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_report(payload)
    return payload


def _write_report(payload: Dict[str, Any]) -> None:
    summary = payload["summary"]
    lines: List[str] = [
        "# WE2 Community Agent 5x49",
        "",
        f"- Generated at: `{payload['generated_at']}`",
        f"- Dataset: `{payload['dataset']}`",
        f"- Aggregation mode: `{payload['aggregation_mode']}`",
        f"- Strategy memory enabled: `{payload['strategy_enabled']}`",
        f"- Repeats per question: `{payload['repeats']}`",
        "",
        "## Summary",
        "",
        f"- Question count: `{summary['question_count']}`",
        f"- Total attempts: `{summary['total_attempts']}`",
        f"- Correct attempts: `{summary['correct_attempts']}`",
        f"- Overall accuracy: `{summary['overall_accuracy']:.4f}`",
        f"- 5-pass correct questions: `{summary['five_pass_correct']}`",
        f"- 5-pass accuracy: `{summary['five_pass_accuracy']:.4f}`",
        f"- Average judge score: `{summary['avg_judge_score']:.4f}`",
        f"- Average latency: `{summary['avg_latency_ms']} ms`",
        "",
        "## Per Question",
        "",
    ]

    for item in payload["question_summary"]:
        lines.extend(
            [
                f"### {item['question_id']}: {item['question']}",
                "",
                f"- Reference: `{item['reference_answer']}`",
                f"- Correct attempts: `{item['correct_count']}/{payload['repeats']}`",
                f"- 5-pass: `{item['pass_5']}`",
                f"- Sample tools: `{', '.join(item['sample_tool_names'])}`",
                "",
                "Sample answer:",
                "```text",
                item["sample_answer"],
                "```",
                "",
            ]
        )
    MD_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    rows = _load_rows()
    cache = _load_cached_results()
    existing = {(str(item.get("question_id", "") or ""), int(item.get("run_index", 0) or 0)) for item in cache}
    pending = [
        (row, run_index)
        for row in rows
        for run_index in range(1, REPEATS + 1)
        if (row["question_id"], run_index) not in existing
    ]
    if not pending:
        payload = _flush_payload(rows, cache)
        print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
        return

    cfg = KAGConfig.from_yaml("configs/config_openai.yaml")
    cfg.global_config.aggregation_mode = "community"
    if hasattr(cfg, "global_"):
        cfg.global_.aggregation_mode = "community"
    cfg.strategy_memory.enabled = False
    cfg.strategy_memory.read_enabled = False

    evaluator = CommunityAgentEvaluator(cfg)
    try:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS, thread_name_prefix="we2-community") as executor:
            future_map = {
                executor.submit(evaluator.evaluate_row, row, run_index): (row["question_id"], run_index)
                for row, run_index in pending
            }
            for future in as_completed(future_map):
                result = future.result()
                cache.append(result.to_dict())
                _flush_payload(rows, cache)
                print(
                    json.dumps(
                        {
                            "question_id": result.question_id,
                            "run_index": result.run_index,
                            "is_correct": bool((result.judge or {}).get("is_correct", False)),
                            "latency_ms": result.latency_ms,
                        },
                        ensure_ascii=False,
                    )
                )
    finally:
        evaluator.close()

    payload = _flush_payload(rows, cache)
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

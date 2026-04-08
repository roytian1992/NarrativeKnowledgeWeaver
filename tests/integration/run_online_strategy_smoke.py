from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from core.agent.retriever_agent import QuestionAnsweringAgent
from core.utils.config import KAGConfig
from core.utils.general_utils import ensure_dir, load_jsonl


def _load_first_qa_row(csv_path: Path) -> Dict[str, str]:
    with csv_path.open("r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            question = str(row.get("question", "") or "").strip()
            answer = str(row.get("answer", "") or "").strip()
            if question and answer:
                return {
                    "question_index": str(row.get("question_index", "") or "").strip(),
                    "question": question,
                    "answer": answer,
                }
    raise RuntimeError(f"No valid question/answer rows found in {csv_path}")


def _tail_row(path: Path) -> Dict[str, Any]:
    rows = load_jsonl(str(path))
    return rows[-1] if rows else {}


def main() -> None:
    cfg = KAGConfig.from_yaml("configs/config_openai.yaml")
    report_dir = Path("reports/online_smoke")
    ensure_dir(str(report_dir))
    for stale_name in (
        "real_traces.jsonl",
        "online_strategy_buffer.jsonl",
        "failure_reflections.jsonl",
        "synthetic_qa_buffer.jsonl",
        "online_smoke_report.json",
    ):
        stale_path = report_dir / stale_name
        if stale_path.exists():
            stale_path.unlink()

    cfg.strategy_memory.enabled = False
    cfg.strategy_memory.read_enabled = False
    cfg.strategy_memory.online_enabled = True
    cfg.strategy_memory.self_bootstrap_enabled = True
    cfg.strategy_memory.self_bootstrap_max_questions = 1
    cfg.strategy_memory.online_buffer_dir = str(report_dir)
    cfg.strategy_memory.online_real_trace_path = str(report_dir / "real_traces.jsonl")
    cfg.strategy_memory.online_strategy_buffer_path = str(report_dir / "online_strategy_buffer.jsonl")
    cfg.strategy_memory.online_failure_reflection_path = str(report_dir / "failure_reflections.jsonl")
    cfg.strategy_memory.online_synthetic_qa_path = str(report_dir / "synthetic_qa_buffer.jsonl")

    qa_row = _load_first_qa_row(Path("examples/datasets/we2_qa.csv"))

    agent = QuestionAnsweringAgent(config=cfg, aggregation_mode="narrative", enable_sql_tools=False)
    responses = agent.ask(
        qa_row["question"],
        lang="zh",
        reference_answer=qa_row["answer"],
        session_id=f"online_smoke_q{qa_row['question_index'] or '0'}",
        online_learning=True,
    )

    final_answer = agent.extract_final_text(responses)
    tool_uses = agent.extract_tool_uses(responses)

    real_trace_path = report_dir / "real_traces.jsonl"
    strategy_buffer_path = report_dir / "online_strategy_buffer.jsonl"
    failure_path = report_dir / "failure_reflections.jsonl"
    synthetic_path = report_dir / "synthetic_qa_buffer.jsonl"

    report = {
        "question_index": qa_row["question_index"],
        "question": qa_row["question"],
        "reference_answer": qa_row["answer"],
        "final_answer": final_answer,
        "tool_use_count": len(tool_uses),
        "tool_names": [str(x.get("tool_name", "") or "").strip() for x in tool_uses if isinstance(x, dict)],
        "real_trace_path": str(real_trace_path),
        "strategy_buffer_path": str(strategy_buffer_path),
        "failure_reflection_path": str(failure_path),
        "synthetic_qa_path": str(synthetic_path),
        "real_trace_last": _tail_row(real_trace_path),
        "strategy_candidate_last": _tail_row(strategy_buffer_path),
        "failure_reflection_last": _tail_row(failure_path),
        "synthetic_qa_last": _tail_row(synthetic_path),
    }

    report_path = report_dir / "online_smoke_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(json.dumps(
        {
            "report_path": str(report_path),
            "final_answer_preview": final_answer[:200],
            "tool_names": report["tool_names"],
            "has_real_trace": bool(report["real_trace_last"]),
            "has_strategy_candidate": bool(report["strategy_candidate_last"]),
            "has_failure_reflection": bool(report["failure_reflection_last"]),
            "has_synthetic_qa": bool(report["synthetic_qa_last"]),
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()

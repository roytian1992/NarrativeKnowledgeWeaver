from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.agent.retriever_agent import QuestionAnsweringAgent
from core.utils.config import KAGConfig
from experiments.quality.run_quality_benchmark import (
    _format_mcq_for_agent_with_retrieval_guard,
    load_existing_article_workspace_or_raise,
)


def _resolve_repo_path(raw_path: str) -> Path:
    path = Path(str(raw_path or "").strip())
    if path.is_absolute():
        return path.resolve()
    return (REPO_ROOT / path).resolve()


def _load_questions_from_report(report_path: Path) -> List[str]:
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    items: Any = payload.get("results") or payload.get("records") or payload
    if isinstance(items, dict):
        items = items.get("records") or items.get("results") or []
    if not isinstance(items, list):
        raise ValueError(f"Unsupported report payload format: {report_path}")

    questions: List[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        question = str(item.get("question") or item.get("query") or item.get("user_question") or "").strip()
        if question:
            questions.append(question)
    if not questions:
        raise ValueError(f"No questions found in report: {report_path}")
    return questions


def _build_tool_preview(tool_uses: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    preview: List[Dict[str, str]] = []
    for item in tool_uses:
        preview.append(
            {
                "tool_name": str((item or {}).get("tool_name", "") or "").strip(),
                "tool_arguments": str((item or {}).get("tool_arguments", "") or "").strip()[:500],
                "tool_output": str((item or {}).get("tool_output", "") or "").strip()[:1200],
            }
        )
    return preview


def main() -> None:
    parser = argparse.ArgumentParser(description="Load an extracted article workspace and run one QA smoke test.")
    parser.add_argument("--config", default="configs/config_openai.yaml")
    parser.add_argument("--workspace-dir", required=True)
    parser.add_argument("--question", default="")
    parser.add_argument("--report-json", default="")
    parser.add_argument("--question-index", type=int, default=0)
    parser.add_argument("--lang", choices=["zh", "en"], default="en")
    parser.add_argument("--disable-sql-tools", action="store_true")
    args = parser.parse_args()

    question = str(args.question or "").strip()
    report_json = str(args.report_json or "").strip()
    if not question and not report_json:
        parser.error("Provide either --question or --report-json.")

    if not question:
        report_path = _resolve_repo_path(report_json)
        questions = _load_questions_from_report(report_path)
        index = max(0, min(int(args.question_index or 0), len(questions) - 1))
        question = questions[index]

    config_path = _resolve_repo_path(args.config)
    workspace_dir = Path(str(args.workspace_dir or "")).resolve()
    base_cfg = KAGConfig.from_yaml(str(config_path))
    cfg = load_existing_article_workspace_or_raise(
        base_cfg=base_cfg,
        workspace_dir=workspace_dir,
        strategy_enabled=False,
        subagent_enabled=False,
    )

    agent = QuestionAnsweringAgent(
        cfg,
        aggregation_mode="narrative",
        enable_sql_tools=not bool(args.disable_sql_tools),
    )
    try:
        prompt = _format_mcq_for_agent_with_retrieval_guard(question)
        responses = agent.ask(
            prompt,
            lang=args.lang,
            require_tool_use=True,
            online_learning=False,
        )
        final_text = agent.extract_final_text(responses)
        tool_uses = agent.extract_tool_uses(responses)
        result = {
            "workspace_dir": str(workspace_dir),
            "graph_store_path": str(cfg.storage.graph_store_path),
            "aggregation_mode": agent.aggregation_mode,
            "question": question,
            "final_answer": final_text,
            "tool_use_count": len(tool_uses),
            "tool_names": [str((item or {}).get("tool_name", "") or "").strip() for item in tool_uses],
            "tool_uses_preview": _build_tool_preview(tool_uses),
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        try:
            agent.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()

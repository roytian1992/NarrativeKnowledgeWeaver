from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.agent.retriever_agent import QuestionAnsweringAgent
from core.utils.config import KAGConfig


CSV_PATH = REPO_ROOT / "examples" / "datasets" / "we2_qa.csv"
REPORT_DIR = REPO_ROOT / "reports"
JSON_REPORT_PATH = REPORT_DIR / "we2_agent_eval_results.json"
MD_REPORT_PATH = REPORT_DIR / "we2_agent_eval_report.md"


def _load_rows() -> List[Dict[str, str]]:
    with CSV_PATH.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _truncate(text: str, limit: int = 1200) -> str:
    raw = str(text or "")
    if len(raw) <= limit:
        return raw
    return raw[:limit] + "\n... [truncated]"


def _tool_issue_markers(tool_uses: List[Dict[str, Any]]) -> List[str]:
    issues: List[str] = []
    for item in tool_uses:
        output = str(item.get("tool_output") or "")
        if any(x in output for x in ("执行失败", "查询失败", "检索失败", "参数解析失败", "Traceback", "Exception")):
            issues.append("tool_error_output")
            break
    return issues


def _answer_issue_markers(answer: str, tool_uses: List[Dict[str, Any]]) -> List[str]:
    issues: List[str] = []
    text = str(answer or "").strip()
    if not text:
        issues.append("empty_answer")
    if not tool_uses:
        issues.append("no_tool_use")
    if any(x in text for x in ("未找到", "未查询到结果", "无法回答", "信息不足", "（无）")):
        issues.append("weak_or_empty_retrieval")
    issues.extend(_tool_issue_markers(tool_uses))
    seen = set()
    out: List[str] = []
    for x in issues:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _tool_name_chain(tool_uses: List[Dict[str, Any]]) -> List[str]:
    names: List[str] = []
    for item in tool_uses:
        name = str(item.get("tool_name") or "").strip()
        if name:
            names.append(name)
    return names


def _build_markdown(results: List[Dict[str, Any]], meta: Dict[str, Any]) -> str:
    total = len(results)
    pass_count = sum(1 for x in results if not x["issue_markers"])
    flagged_count = total - pass_count
    issue_counts: Dict[str, int] = {}
    for item in results:
        for marker in item.get("issue_markers", []):
            issue_counts[marker] = issue_counts.get(marker, 0) + 1
    lines: List[str] = [
        "# WE2 Agent Eval Report",
        "",
        f"- Dataset: `{CSV_PATH.name}`",
        f"- Total questions: `{total}`",
        f"- Clean runs (no issue marker): `{pass_count}`",
        f"- Flagged runs: `{flagged_count}`",
        f"- Mode: `{meta['mode']}`",
        f"- SQL tools enabled: `{meta['enable_sql_tools']}`",
        f"- Strategy memory enabled during eval: `{meta['strategy_memory_enabled']}`",
        "",
        "## Issue Counts",
        "",
    ]
    if issue_counts:
        for key in sorted(issue_counts):
            lines.append(f"- `{key}`: `{issue_counts[key]}`")
    else:
        lines.append("- `(none)`")
    lines.extend(
        [
            "",
        "## Issue Marker Legend",
        "",
        "- `empty_answer`: final answer empty",
        "- `no_tool_use`: no tool call captured",
        "- `weak_or_empty_retrieval`: answer contains obvious retrieval miss markers",
        "- `tool_error_output`: at least one tool output contains explicit failure/error markers",
        "- `exception`: agent run raised an exception and was captured by the runner",
        "",
        ]
    )

    for item in results:
        lines.extend(
            [
                f"## Q{item['question_index']}: {item['question']}",
                "",
                f"- Question type: `{item['question_type']}`",
                f"- Runtime: `{item['latency_ms']} ms`",
                f"- Tool calls: `{item['tool_call_count']}`",
                f"- Tool chain: `{ ' -> '.join(item['tool_names']) if item['tool_names'] else '(none)' }`",
                f"- Issue markers: `{', '.join(item['issue_markers']) if item['issue_markers'] else 'none'}`",
                "",
                "### Reference Answer",
                "",
                "```text",
                item["refined_answer"] or "",
                "```",
                "",
                "### Agent Answer",
                "",
                "```text",
                item["final_answer"] or "",
                "```",
                "",
            ]
        )
        if item.get("exception"):
            lines.extend(
                [
                    "### Exception",
                    "",
                    "```text",
                    str(item["exception"]),
                    "```",
                    "",
                ]
            )
        lines.extend(
            [
                "### Tool Trace",
                "",
            ]
        )
        if item["tool_uses"]:
            for idx, tool in enumerate(item["tool_uses"], 1):
                lines.extend(
                    [
                        f"#### {idx}. `{tool['tool_name']}`",
                        "",
                        "- Arguments:",
                        "```json",
                        tool["tool_arguments"] or "",
                        "```",
                        "- Output sample:",
                        "```text",
                        _truncate(tool["tool_output"]),
                        "```",
                        "",
                    ]
                )
        else:
            lines.extend(["(none)", ""])

    return "\n".join(lines)


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    rows = _load_rows()

    cfg = KAGConfig.from_yaml("configs/config_openai.yaml")
    if hasattr(cfg, "strategy_memory") and cfg.strategy_memory is not None:
        cfg.strategy_memory.enabled = False
        cfg.strategy_memory.read_enabled = False

    agent = QuestionAnsweringAgent(
        cfg,
        mode="hybrid",
        enable_sql_tools=True,
    )

    results: List[Dict[str, Any]] = []
    started = time.time()
    try:
        for idx, row in enumerate(rows, 1):
            q_started = time.time()
            question = str(row.get("question") or "").strip()
            try:
                responses = agent.ask(question, lang="zh", session_id=f"we2_q_{idx}")
                final_answer = agent.extract_final_text(responses)
                tool_uses = agent.extract_tool_uses(responses)
                tool_names = _tool_name_chain(tool_uses)
                issue_markers = _answer_issue_markers(final_answer, tool_uses)
                result = {
                    "row_number": idx,
                    "question_index": row.get("question_index"),
                    "question_type": row.get("question_type"),
                    "question": question,
                    "refined_answer": row.get("refined_answer") or "",
                    "original_ground_truth": row.get("original_ground_truth") or "",
                    "final_answer": final_answer or "",
                    "latency_ms": int((time.time() - q_started) * 1000),
                    "tool_call_count": len(tool_uses),
                    "tool_names": tool_names,
                    "issue_markers": issue_markers,
                    "tool_uses": [
                        {
                            "tool_name": str(x.get("tool_name") or ""),
                            "tool_arguments": str(x.get("tool_arguments") or ""),
                            "tool_output": str(x.get("tool_output") or ""),
                        }
                        for x in tool_uses
                    ],
                }
            except Exception as exc:
                result = {
                    "row_number": idx,
                    "question_index": row.get("question_index"),
                    "question_type": row.get("question_type"),
                    "question": question,
                    "refined_answer": row.get("refined_answer") or "",
                    "original_ground_truth": row.get("original_ground_truth") or "",
                    "final_answer": "",
                    "latency_ms": int((time.time() - q_started) * 1000),
                    "tool_call_count": 0,
                    "tool_names": [],
                    "issue_markers": ["exception"],
                    "tool_uses": [],
                    "exception": f"{type(exc).__name__}: {exc}",
                }
            results.append(result)
            print(
                json.dumps(
                    {
                        "row": idx,
                        "question_index": row.get("question_index"),
                        "tool_calls": result["tool_call_count"],
                        "issues": result["issue_markers"],
                        "latency_ms": result["latency_ms"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    finally:
        agent.close()

    meta = {
        "dataset": str(CSV_PATH),
        "generated_at": int(time.time()),
        "total_runtime_ms": int((time.time() - started) * 1000),
        "mode": "hybrid",
        "enable_sql_tools": True,
        "strategy_memory_enabled": False,
    }

    JSON_REPORT_PATH.write_text(
        json.dumps({"meta": meta, "results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    MD_REPORT_PATH.write_text(_build_markdown(results, meta), encoding="utf-8")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

from core.agent.retriever_agent import QuestionAnsweringAgent
from core.utils.config import KAGConfig
from experiments.quality.run_quality_benchmark import load_existing_article_workspace_or_raise


DEFAULT_SAMPLE_QUESTIONS: List[Tuple[str, str]] = [
    ("a_2b020f28fca0d8fe", "q0"),
    ("a_2b020f28fca0d8fe", "q1"),
    ("a_2b020f28fca0d8fe", "q6"),
    ("a_d388a00f3ee9b087", "q0"),
    ("a_d388a00f3ee9b087", "q13"),
    ("a_6fb3b00f835658f2", "q3"),
    ("a_6fb3b00f835658f2", "q5"),
    ("a_6fb3b00f835658f2", "q6"),
    ("a_a30ad9d413f4bbc3", "q10"),
    ("a_d8287e7a17d971ca", "q0"),
]

LABEL_RE = re.compile(r"^\s*([A-D])(?:[\.\s]|$)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config_openai.yaml")
    parser.add_argument(
        "--workspace-root",
        default="experiments/quality/assets/article_workspaces",
    )
    parser.add_argument(
        "--source-json",
        default="experiments/quality/artifacts/worst50_zero_pass_analysis_20260403.json",
    )
    parser.add_argument(
        "--output-json",
        default="experiments/quality/artifacts/manual_tool_probe_hard10_zero_pass_eval.json",
    )
    parser.add_argument(
        "--sample-mode",
        choices=["hard10", "balanced"],
        default="hard10",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--tool-names",
        nargs="+",
        default=["choice_grounded_evidence_search", "entity_event_trace_search"],
    )
    return parser.parse_args()


def load_question_pool(path: Path) -> Dict[Tuple[str, str], Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    rows = payload.get("zero_pass_questions") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise ValueError(f"Invalid zero_pass_questions payload: {path}")
    out: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        article = str(row.get("article_name", "") or "").strip()
        question_id = str(row.get("question_id", "") or "").strip()
        if not article or not question_id:
            continue
        out[(article, question_id)] = row
    return out


def extract_predicted_label(tool_name: str, raw_output: str) -> str:
    lines = [line.strip() for line in str(raw_output or "").splitlines() if line.strip()]
    if tool_name == "choice_grounded_evidence_search":
        for idx, line in enumerate(lines):
            if line == "[Recommended Choice]" and idx + 1 < len(lines):
                match = LABEL_RE.match(lines[idx + 1])
                if match:
                    return match.group(1)
    if tool_name == "entity_event_trace_search":
        for idx, line in enumerate(lines):
            if line == "[Suggested Choice]" and idx + 1 < len(lines):
                match = LABEL_RE.match(lines[idx + 1])
                if match:
                    return match.group(1)
                label_match = re.search(r"\b([A-D])\b", lines[idx + 1])
                if label_match:
                    return label_match.group(1)
    if tool_name == "fact_timeline_resolution_search":
        for idx, line in enumerate(lines):
            if line == "[Suggested Choice]" and idx + 1 < len(lines):
                match = LABEL_RE.match(lines[idx + 1])
                if match:
                    return match.group(1)
                label_match = re.search(r"\b([A-D])\b", lines[idx + 1])
                if label_match:
                    return label_match.group(1)
    if tool_name == "implication_constrained_inference_search":
        for idx, line in enumerate(lines):
            if line == "[Suggested Choice]" and idx + 1 < len(lines):
                match = LABEL_RE.match(lines[idx + 1])
                if match:
                    return match.group(1)
                label_match = re.search(r"\b([A-D])\b", lines[idx + 1])
                if label_match:
                    return label_match.group(1)
    return ""


def select_balanced_questions(
    pool: Dict[Tuple[str, str], Dict[str, Any]],
    *,
    sample_size: int,
) -> List[Tuple[str, str]]:
    rows = list(pool.values())
    rows.sort(
        key=lambda row: (
            str(row.get("category", "") or ""),
            str(row.get("article_name", "") or ""),
            str(row.get("question_id", "") or ""),
        )
    )

    selected: List[Tuple[str, str]] = []
    article_counts: Dict[str, int] = defaultdict(int)
    category_counts: Dict[str, int] = defaultdict(int)
    used = set()

    preferred_categories = [
        "causal_intent",
        "entity_relation",
        "attitude_trait",
        "option_discrimination",
        "event_fact",
        "counterfactual_theme",
        "count_time",
        "other",
    ]
    by_category: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_category[str(row.get("category", "") or "unknown")].append(row)

    while len(selected) < sample_size:
        progress = False
        for category in preferred_categories:
            bucket = by_category.get(category) or []
            bucket.sort(
                key=lambda row: (
                    article_counts[str(row.get("article_name", "") or "")],
                    category_counts[str(row.get("category", "") or "")],
                    str(row.get("article_name", "") or ""),
                    str(row.get("question_id", "") or ""),
                )
            )
            picked = None
            for row in bucket:
                key = (str(row.get("article_name", "") or ""), str(row.get("question_id", "") or ""))
                if key in used:
                    continue
                article_name = key[0]
                if article_counts[article_name] >= 2 and len(selected) < sample_size // 2:
                    continue
                picked = row
                break
            if picked is None:
                continue
            key = (str(picked.get("article_name", "") or ""), str(picked.get("question_id", "") or ""))
            used.add(key)
            selected.append(key)
            article_counts[key[0]] += 1
            category_counts[str(picked.get("category", "") or "unknown")] += 1
            progress = True
            if len(selected) >= sample_size:
                break
        if not progress:
            break

    if len(selected) < sample_size:
        remaining = [
            (str(row.get("article_name", "") or ""), str(row.get("question_id", "") or ""))
            for row in rows
            if (str(row.get("article_name", "") or ""), str(row.get("question_id", "") or "")) not in used
        ]
        selected.extend(remaining[: max(0, sample_size - len(selected))])
    return selected[:sample_size]


def main() -> None:
    args = parse_args()
    base_cfg = KAGConfig.from_yaml(args.config)
    workspace_root = REPO_ROOT / args.workspace_root
    source_json = REPO_ROOT / args.source_json
    output_json = REPO_ROOT / args.output_json
    pool = load_question_pool(source_json)

    tool_names = list(dict.fromkeys([str(name or "").strip() for name in args.tool_names if str(name or "").strip()]))
    results: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    article_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    if args.sample_mode == "hard10":
        sample_questions = DEFAULT_SAMPLE_QUESTIONS[: max(1, int(args.sample_size or 10))]
    else:
        sample_questions = select_balanced_questions(
            pool,
            sample_size=max(1, int(args.sample_size or 10)),
        )

    for article_name, question_id in sample_questions:
        row = pool.get((article_name, question_id))
        if row is None:
            raise KeyError(f"Missing sample question: {(article_name, question_id)}")
        article_groups[article_name].append(row)

    for article_name, rows in article_groups.items():
        workspace_dir = workspace_root / article_name
        cfg = load_existing_article_workspace_or_raise(
            base_cfg=base_cfg,
            workspace_dir=workspace_dir,
        )
        agent = QuestionAnsweringAgent(
            cfg,
            aggregation_mode="narrative",
            enable_sql_tools=True,
        )
        try:
            tool_map = {
                getattr(tool, "name", type(tool).__name__): tool
                for tool in list(getattr(agent, "_base_tools", []) or [])
            }
            for row in rows:
                question = str(row.get("question", "") or "").strip()
                gold = str(row.get("reference_choice", "") or "").strip().upper()
                for tool_name in tool_names:
                    tool = tool_map.get(tool_name)
                    if tool is None:
                        raise KeyError(f"Tool not found: {tool_name}")
                    params = {"query": question}
                    if tool_name == "choice_grounded_evidence_search":
                        params["use_llm_judge"] = False
                    elif tool_name == "entity_event_trace_search":
                        params["use_option_probes"] = False
                        params["use_llm_choice_judge"] = True
                    started = time.time()
                    raw_output = tool.call(json.dumps(params, ensure_ascii=False))
                    latency_ms = int((time.time() - started) * 1000)
                    pred = extract_predicted_label(tool_name, raw_output)
                    results[tool_name].append(
                        {
                            "article_name": article_name,
                            "question_id": str(row.get("question_id", "") or "").strip(),
                            "category": str(row.get("category", "") or "").strip(),
                            "question": question,
                            "gold": gold,
                            "pred": pred,
                            "is_correct": bool(pred and pred == gold),
                            "latency_ms": latency_ms,
                            "raw_output": raw_output,
                        }
                    )
        finally:
            agent.close()

    summary: Dict[str, Dict[str, Any]] = {}
    for tool_name, rows in results.items():
        correct = sum(1 for row in rows if row.get("is_correct"))
        total = len(rows)
        category_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {"correct": 0, "total": 0})
        for row in rows:
            category = str(row.get("category", "") or "unknown").strip() or "unknown"
            category_stats[category]["total"] += 1
            if row.get("is_correct"):
                category_stats[category]["correct"] += 1
        summary[tool_name] = {
            "correct": correct,
            "total": total,
            "accuracy": round(correct / total, 4) if total else 0.0,
            "avg_latency_ms": int(sum(int(row.get("latency_ms", 0) or 0) for row in rows) / total) if total else 0,
            "category_breakdown": dict(sorted(category_stats.items())),
        }

    output_payload = {
        "samples": [{"article_name": a, "question_id": q} for a, q in sample_questions],
        "summary": summary,
        "results": dict(results),
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(output_payload, f, ensure_ascii=False, indent=2)

    print(json.dumps(output_payload["summary"], ensure_ascii=False, indent=2))
    print(f"saved_to={output_json}")


if __name__ == "__main__":
    main()

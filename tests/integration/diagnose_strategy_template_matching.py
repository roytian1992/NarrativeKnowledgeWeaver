from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List


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
TRAINING_DETAIL_DIR = REPO_ROOT / "strategy_training" / "we2_49x5_parallel_v11" / "distilled" / "per_question"
REPORT_JSON = REPO_ROOT / "reports" / "strategy_template_diagnostics_20260315.json"
REPORT_MD = REPO_ROOT / "reports" / "strategy_template_diagnostics_20260315.md"
TARGET_QIDS = ["q10", "q17", "q27", "q33", "q38", "q45", "q43"]


def _load_rows() -> List[Dict[str, str]]:
    with CSV_PATH.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows: List[Dict[str, str]] = []
        for idx, row in enumerate(reader):
            question = str((row or {}).get("question", "") or "").strip()
            answer = str((row or {}).get("answer", "") or "").strip()
            if not question or not answer:
                continue
            rows.append(
                {
                    "question_id": f"q{idx}",
                    "question": question,
                    "answer": answer,
                }
            )
        return rows


def _truncate(value: Any, limit: int = 220) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)] + "..."


def _load_training_detail(question_id: str) -> Dict[str, Any]:
    path = TRAINING_DETAIL_DIR / f"{question_id}.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _sanitize_strategy_context(ctx: Dict[str, Any], *, question_id: str) -> Dict[str, Any]:
    if not isinstance(ctx, dict):
        return {}
    out = {
        "query_abstract": str(ctx.get("query_abstract", "") or ""),
        "query_pattern": ctx.get("query_pattern") if isinstance(ctx.get("query_pattern"), dict) else {},
        "routing_hint": str(ctx.get("routing_hint", "") or ""),
        "active_patterns": [],
        "candidate_patterns": [],
    }
    active_ids = {
        str(item.get("template_id", "") or "")
        for item in (ctx.get("patterns") or [])
        if isinstance(item, dict) and str(item.get("template_id", "") or "").strip()
    }
    for item in (ctx.get("candidate_patterns") or []):
        if not isinstance(item, dict):
            continue
        srcs = [str(x).strip() for x in (item.get("source_question_ids") or []) if str(x).strip()]
        out["candidate_patterns"].append(
            {
                "template_id": str(item.get("template_id", "") or ""),
                "pattern_name": str(item.get("pattern_name", "") or ""),
                "query_abstract": str(item.get("query_abstract", "") or ""),
                "similarity": float(item.get("similarity", 0.0) or 0.0),
                "embedding_similarity": item.get("embedding_similarity"),
                "match_text_overlap": float(item.get("match_text_overlap", 0.0) or 0.0),
                "answer_shape_match": float(item.get("answer_shape_match", 0.0) or 0.0),
                "selection_status": str(item.get("selection_status", "") or ""),
                "selection_reason": str(item.get("selection_reason", "") or ""),
                "chain_compatible": item.get("chain_compatible"),
                "has_self_source": question_id in srcs,
                "source_question_ids": srcs,
                "recommended_chain": [str(x).strip() for x in (item.get("recommended_chain") or []) if str(x).strip()],
            }
        )
    for item in (ctx.get("patterns") or []):
        if not isinstance(item, dict):
            continue
        out["active_patterns"].append(
            {
                "template_id": str(item.get("template_id", "") or ""),
                "pattern_name": str(item.get("pattern_name", "") or ""),
                "recommended_chain": [str(x).strip() for x in (item.get("recommended_chain") or []) if str(x).strip()],
                "is_selected": str(item.get("template_id", "") or "") in active_ids,
            }
        )
    return out


def main() -> None:
    cfg = KAGConfig.from_yaml("configs/config_openai.yaml")
    cfg.global_config.aggregation_mode = "narrative"
    if hasattr(cfg, "global_"):
        cfg.global_.aggregation_mode = "narrative"
    cfg.strategy_memory.enabled = True
    cfg.strategy_memory.read_enabled = True

    prompt_loader = YAMLPromptLoader(cfg.global_config.prompt_dir)
    llm = OpenAILLM(cfg)
    judge = RetrievalAnswerJudge(prompt_loader, llm)
    agent = QuestionAnsweringAgent(cfg, aggregation_mode="narrative", enable_sql_tools=False)

    rows = {item["question_id"]: item for item in _load_rows()}
    results: List[Dict[str, Any]] = []
    for qid in TARGET_QIDS:
        row = rows[qid]
        started = time.time()
        responses = agent.ask(
            row["question"],
            lang=str(cfg.global_config.language or "zh"),
            session_id=f"strategy_diag_{qid}",
            max_llm_calls_per_run=10,
        )
        final_answer = agent.extract_final_text(responses)
        tool_uses = agent.extract_tool_uses(responses)
        strategy_context = _sanitize_strategy_context(agent.get_last_strategy_context(), question_id=qid)
        judgment = judge.evaluate(
            question=row["question"],
            reference_answer=row["answer"],
            candidate_answer=final_answer,
        )
        training_detail = _load_training_detail(qid)
        results.append(
            {
                "question_id": qid,
                "question": row["question"],
                "reference_answer": row["answer"],
                "final_answer": final_answer,
                "judge": judgment,
                "latency_ms": int((time.time() - started) * 1000),
                "tool_names": [str((x or {}).get("tool_name", "") or "").strip() for x in tool_uses if str((x or {}).get("tool_name", "") or "").strip()],
                "tool_uses": tool_uses,
                "strategy_context": strategy_context,
                "training_query_pattern": training_detail.get("query_pattern", {}) if isinstance(training_detail.get("query_pattern"), dict) else {},
                "training_best_effective_tool_chain": training_detail.get("best_effective_tool_chain") or [],
                "training_best_raw_tool_chain": training_detail.get("best_raw_tool_chain") or [],
                "training_best_attempt_id": str(training_detail.get("best_attempt_id", "") or ""),
            }
        )

    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "target_qids": TARGET_QIDS,
        "results": results,
    }
    REPORT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines: List[str] = [
        "# Strategy Template Diagnostics",
        "",
        f"- Generated at: `{payload['generated_at']}`",
        f"- Questions: `{', '.join(TARGET_QIDS)}`",
        "",
    ]
    for item in results:
        lines.extend(
            [
                f"## {item['question_id']}",
                "",
                f"- Question: `{item['question']}`",
                f"- Judge correct: `{bool(((item.get('judge') or {}) if isinstance(item.get('judge'), dict) else {}).get('is_correct', False))}`",
                f"- Latency ms: `{item['latency_ms']}`",
                f"- Training best chain: `{item.get('training_best_effective_tool_chain')}`",
                f"- Actual tool chain: `{item.get('tool_names')}`",
                f"- Final answer: `{_truncate(item.get('final_answer'), 500)}`",
                "",
                "### Strategy Context",
                "",
                f"- Query abstract: `{_truncate((item.get('strategy_context') or {}).get('query_abstract', ''), 300)}`",
                f"- Active templates: `{[(x.get('template_id'), x.get('pattern_name')) for x in ((item.get('strategy_context') or {}).get('active_patterns') or [])]}`",
                "",
                "### Candidate Templates",
                "",
            ]
        )
        for cand in (item.get("strategy_context") or {}).get("candidate_patterns") or []:
            lines.extend(
                [
                    f"- `{cand['template_id']}` `{cand['pattern_name']}` "
                    f"`score={cand['similarity']:.4f}` "
                    f"`status={cand['selection_status']}` "
                    f"`reason={cand['selection_reason']}` "
                    f"`has_self_source={cand['has_self_source']}` "
                    f"`chain={cand['recommended_chain']}`",
                ]
            )
        lines.extend(
            [
                "",
                "### Routing Hint",
                "",
                "```text",
                str((item.get("strategy_context") or {}).get("routing_hint", "") or ""),
                "```",
                "",
            ]
        )
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved JSON: {REPORT_JSON}")
    print(f"Saved MD: {REPORT_MD}")


if __name__ == "__main__":
    main()

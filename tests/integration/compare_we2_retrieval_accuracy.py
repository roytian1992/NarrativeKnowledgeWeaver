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
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document as LCDocument


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

from core.agent.retriever_agent import QuestionAnsweringAgent, _split_bm25_documents
from core.functions.memory_management.judge_retrieval_answer import RetrievalAnswerJudge
from core.model_providers.openai_llm import OpenAILLM
from core.storage.vector_store import VectorStore
from core.utils.config import KAGConfig
from core.utils.prompt_loader import YAMLPromptLoader
from retriever.sparse_retriever import KeywordBM25Retriever
from core.functions.tool_calls.native_tools import zh_preprocess


CSV_PATH = REPO_ROOT / "examples" / "datasets" / "we2_qa.csv"
REPORT_DIR = REPO_ROOT / "reports"
JSON_PATH = REPORT_DIR / "we2_retrieval_accuracy_5x_comparison_20260315.json"
MD_PATH = REPORT_DIR / "we2_retrieval_accuracy_5x_comparison_20260315.md"
REPEATS = 5
AGENT_MAX_WORKERS = 4
HYBRID_MAX_WORKERS = 6


@dataclass
class EvalResult:
    setting: str
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
            "setting": self.setting,
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


def _truncate(text: Any, limit: int = 360) -> str:
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


def _extract_text_from_llm_result(result: Any) -> str:
    if isinstance(result, list) and result:
        item = result[0]
        content = getattr(item, "content", None)
        if isinstance(content, str):
            return content.strip()
    if isinstance(result, str):
        return result.strip()
    return str(result or "").strip()


def _format_evidence_item(idx: int, item: Dict[str, Any]) -> str:
    meta = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    lines = [
        f"[{idx}] channel={item.get('channel', '')}",
        f"document_id={meta.get('document_id', '')}",
        f"title={meta.get('title') or meta.get('source_title') or meta.get('doc_title') or ''}",
        f"chunk_id={meta.get('chunk_id') or item.get('id') or ''}",
        "content:",
        str(item.get("content", "") or "").strip(),
    ]
    return "\n".join(lines)


def _rrf_merge(
    doc_hits: List[Any],
    sent_hits: List[Any],
    bm25_hits: List[LCDocument],
    *,
    top_k: int = 8,
) -> List[Dict[str, Any]]:
    fused: Dict[str, Dict[str, Any]] = {}
    k0 = 60.0
    channels: List[Tuple[str, float, List[Any]]] = [
        ("vector_document", 1.0, doc_hits or []),
        ("vector_sentence", 0.9, sent_hits or []),
        ("bm25", 1.1, bm25_hits or []),
    ]

    for channel_name, weight, hits in channels:
        for rank, hit in enumerate(hits, start=1):
            if hasattr(hit, "content"):
                content = str(getattr(hit, "content", "") or "").strip()
                meta = dict(getattr(hit, "metadata", None) or {})
                item_id = str(getattr(hit, "id", "") or meta.get("chunk_id") or "")
            else:
                content = str(getattr(hit, "page_content", "") or "").strip()
                meta = dict(getattr(hit, "metadata", None) or {})
                item_id = str(meta.get("chunk_id") or meta.get("document_id") or "")
            if not content:
                continue
            if not item_id:
                item_id = f"{channel_name}:{rank}:{hash(content)}"
            score = weight / (k0 + float(rank))
            row = fused.get(item_id)
            if row is None:
                fused[item_id] = {
                    "id": item_id,
                    "channel": channel_name,
                    "content": content,
                    "metadata": meta,
                    "score": score,
                    "channels": [channel_name],
                }
            else:
                row["score"] = float(row.get("score", 0.0)) + score
                if channel_name not in row["channels"]:
                    row["channels"].append(channel_name)
                if len(content) > len(str(row.get("content", "") or "")):
                    row["content"] = content
                    row["metadata"] = meta

    rows = sorted(
        fused.values(),
        key=lambda x: (float(x.get("score", 0.0)), len(str(x.get("content", "") or ""))),
        reverse=True,
    )
    return rows[: max(1, int(top_k or 8))]


class AgentThreadLocal:
    def __init__(self, cfg: KAGConfig, *, setting_name: str, strategy_enabled: bool) -> None:
        self.cfg = cfg
        self.setting_name = setting_name
        self.strategy_enabled = bool(strategy_enabled)
        self.local = threading.local()
        self._states: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

    def _build_state(self) -> Dict[str, Any]:
        cfg = copy.deepcopy(self.cfg)
        cfg.global_config.aggregation_mode = "narrative"
        if hasattr(cfg, "global_"):
            cfg.global_.aggregation_mode = "narrative"
        cfg.strategy_memory.enabled = self.strategy_enabled
        cfg.strategy_memory.read_enabled = self.strategy_enabled
        prompt_loader = YAMLPromptLoader(cfg.global_config.prompt_dir)
        llm = OpenAILLM(cfg)
        judge = RetrievalAnswerJudge(prompt_loader, llm)
        agent = QuestionAnsweringAgent(
            cfg,
            aggregation_mode="narrative",
            enable_sql_tools=False,
        )
        state = {
            "cfg": cfg,
            "prompt_loader": prompt_loader,
            "llm": llm,
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


class AgentEvaluator:
    def __init__(self, cfg: KAGConfig, *, setting_name: str, strategy_enabled: bool) -> None:
        self.setting_name = str(setting_name or "agent").strip() or "agent"
        self.tlocal = AgentThreadLocal(cfg, setting_name=self.setting_name, strategy_enabled=strategy_enabled)

    def evaluate_row(self, row: Dict[str, str], run_index: int) -> EvalResult:
        state = self.tlocal.state()
        started = time.time()
        responses: List[Dict[str, Any]] = []
        final_answer = ""
        tool_uses: List[Dict[str, Any]] = []
        strategy_context: Dict[str, Any] = {}
        judge: Dict[str, Any]
        error_text = ""

        for max_calls in (10, 6, 4):
            try:
                responses = state["agent"].ask(
                    row["question"],
                    lang=str(state["cfg"].global_config.language or "zh"),
                    session_id=f"{self.setting_name}_{row['question_id']}_run_{run_index}_{max_calls}",
                    max_llm_calls_per_run=max_calls,
                )
                final_answer = state["agent"].extract_final_text(responses)
                tool_uses = state["agent"].extract_tool_uses(responses)
                strategy_context = state["agent"].get_last_strategy_context()
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
            setting=self.setting_name,
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
                "tool_names": [str((x or {}).get("tool_name", "") or "").strip() for x in tool_uses if str((x or {}).get("tool_name", "") or "").strip()],
                "tool_uses": tool_uses,
                "strategy_context": strategy_context,
                "error": error_text,
            },
        )

    def close(self) -> None:
        self.tlocal.close()


class TraditionalHybridThreadLocal:
    def __init__(self, cfg: KAGConfig) -> None:
        self.cfg = cfg
        self.local = threading.local()

    def _build_state(self) -> Dict[str, Any]:
        prompt_loader = YAMLPromptLoader(self.cfg.global_config.prompt_dir)
        llm = OpenAILLM(self.cfg)
        judge = RetrievalAnswerJudge(prompt_loader, llm)
        doc_vs = VectorStore(self.cfg, "document")
        sent_vs = VectorStore(self.cfg, "sentence")

        base = Path(self.cfg.knowledge_graph_builder.file_path) / "all_document_chunks.json"
        data = json.loads(base.read_text(encoding="utf-8"))
        docs: List[LCDocument] = []
        keys_to_drop = {"chunk_index", "chunk_type", "doc_title", "order", "total_doc_chunks"}
        for item in data:
            chunk_id = item.get("id")
            content = str(item.get("content") or "").strip()
            if not chunk_id or not content:
                continue
            meta = dict(item.get("metadata") or {})
            meta["chunk_id"] = chunk_id
            for key in list(keys_to_drop):
                meta.pop(key, None)
            docs.append(LCDocument(page_content=content, metadata=meta))
        split_docs = _split_bm25_documents(
            docs,
            chunk_size=int(getattr(self.cfg.document_processing, "bm25_chunk_size", 250) or 0),
        )
        bm25 = KeywordBM25Retriever(split_docs, zh_preprocess=zh_preprocess, k_default=10)
        return {
            "prompt_loader": prompt_loader,
            "llm": llm,
            "judge": judge,
            "doc_vs": doc_vs,
            "sent_vs": sent_vs,
            "bm25": bm25,
        }

    def state(self) -> Dict[str, Any]:
        state = getattr(self.local, "state", None)
        if state is None:
            state = self._build_state()
            self.local.state = state
        return state


class TraditionalHybridEvaluator:
    def __init__(self, cfg: KAGConfig) -> None:
        self.cfg = cfg
        self.tlocal = TraditionalHybridThreadLocal(cfg)

    def evaluate_row(self, row: Dict[str, str], run_index: int) -> EvalResult:
        state = self.tlocal.state()
        started = time.time()
        error_text = ""
        doc_hits: List[Any] = []
        sent_hits: List[Any] = []
        bm25_hits: List[LCDocument] = []
        fused: List[Dict[str, Any]] = []
        final_answer = ""
        try:
            doc_hits = state["doc_vs"].search(row["question"], limit=8)
            sent_hits = state["sent_vs"].search(row["question"], limit=8)
            bm25_hits = state["bm25"].retrieve(row["question"], k=8)
            fused = _rrf_merge(doc_hits, sent_hits, bm25_hits, top_k=8)
            evidence_text = "\n\n".join(_format_evidence_item(i, item) for i, item in enumerate(fused, start=1))
            prompt = state["prompt_loader"].render(
                "memory/answer_with_retrieved_evidence",
                task_values={
                    "question": row["question"],
                    "retrieved_evidence": evidence_text or "（无）",
                },
                strict=True,
            )
            llm_result = state["llm"].run([{"role": "user", "content": prompt}])
            final_answer = _extract_text_from_llm_result(llm_result)
            judge = state["judge"].evaluate(
                question=row["question"],
                reference_answer=row["answer"],
                candidate_answer=final_answer,
            )
        except Exception as exc:
            error_text = f"{type(exc).__name__}: {exc}"
            judge = {
                "is_correct": False,
                "score": 0.0,
                "reason": f"traditional_hybrid_error: {error_text}",
                "matched_points": [],
                "missing_points": [],
                "hallucination_points": [],
            }
        return EvalResult(
            setting="traditional_hybrid_rag_bm25",
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
                "retrieval_counts": {
                    "vector_document": len(doc_hits),
                    "vector_sentence": len(sent_hits),
                    "bm25": len(bm25_hits),
                    "fused": len(fused),
                },
                "evidence": fused,
                "error": error_text,
            },
        )


def _result_key(item: EvalResult | Dict[str, Any]) -> Tuple[str, int]:
    if isinstance(item, EvalResult):
        return item.question_id, int(item.run_index)
    return str(item.get("question_id", "") or ""), int(item.get("run_index", 0) or 0)


def _load_cached_results(path: Path) -> Dict[str, List[Dict[str, Any]]]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    out: Dict[str, List[Dict[str, Any]]] = {}
    for setting, rows in payload.items():
        if setting in {"generated_at", "dataset", "summary", "question_summary", "repeats"}:
            continue
        if isinstance(rows, list):
            out[setting] = [item for item in rows if isinstance(item, dict)]
    return out


def _summarize_setting(name: str, rows: List[Dict[str, Any]], question_rows: List[Dict[str, str]]) -> Dict[str, Any]:
    total_attempts = len(rows)
    correct_attempts = sum(1 for x in rows if bool(((x.get("judge") or {}) if isinstance(x.get("judge"), dict) else {}).get("is_correct", False)))
    avg_score = round(
        sum(float(((x.get("judge") or {}) if isinstance(x.get("judge"), dict) else {}).get("score", 0.0) or 0.0) for x in rows)
        / float(total_attempts or 1),
        4,
    )
    avg_latency = int(sum(int(x.get("latency_ms", 0) or 0) for x in rows) / float(total_attempts or 1))
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for item in rows:
        grouped.setdefault(str(item.get("question_id", "") or ""), []).append(item)
    pass_questions = 0
    for qrow in question_rows:
        attempts = grouped.get(qrow["question_id"], [])
        if any(bool(((x.get("judge") or {}) if isinstance(x.get("judge"), dict) else {}).get("is_correct", False)) for x in attempts):
            pass_questions += 1
    return {
        "setting": name,
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


def _build_question_summary(rows: List[Dict[str, str]], by_setting: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for setting, items in by_setting.items():
        qmap: Dict[str, Dict[str, Any]] = {}
        for row in items:
            qid = str(row.get("question_id", "") or "")
            qmap.setdefault(qid, {"attempts": []})["attempts"].append(row)
        grouped[setting] = qmap

    summary_rows: List[Dict[str, Any]] = []
    for base in rows:
        row: Dict[str, Any] = {
            "question_id": base["question_id"],
            "question": base["question"],
            "reference_answer": base["answer"],
        }
        for setting in ["strategy_agent", "no_strategy_agent", "traditional_hybrid_rag_bm25"]:
            attempts = (grouped.get(setting, {}).get(base["question_id"], {}) or {}).get("attempts", [])
            attempts = sorted(attempts, key=lambda x: int(x.get("run_index", 0) or 0))
            correct_count = sum(1 for x in attempts if bool(((x.get("judge") or {}) if isinstance(x.get("judge"), dict) else {}).get("is_correct", False)))
            row[f"{setting}_correct_count"] = correct_count
            row[f"{setting}_pass_5"] = bool(correct_count > 0)
            row[f"{setting}_sample_answer"] = _truncate(str((attempts[0] if attempts else {}).get("final_answer", "") or ""), 500)
        summary_rows.append(row)
    return summary_rows


def _write_report(payload: Dict[str, Any]) -> None:
    lines: List[str] = [
        "# WE2 Retrieval Accuracy 5x Comparison",
        "",
        f"- Generated at: `{payload['generated_at']}`",
        f"- Dataset: `{payload['dataset']}`",
        f"- Repeats per question: `{payload['repeats']}`",
        "",
        "## Summary",
        "",
    ]
    for item in payload["summary"]:
        lines.extend(
            [
                f"### {item['setting']}",
                "",
                f"- Question count: `{item['question_count']}`",
                f"- Total attempts: `{item['total_attempts']}`",
                f"- Correct attempts: `{item['correct_attempts']}`",
                f"- Overall accuracy: `{item['overall_accuracy']:.4f}`",
                f"- 5-pass correct questions: `{item['five_pass_correct']}`",
                f"- 5-pass accuracy: `{item['five_pass_accuracy']:.4f}`",
                f"- Average judge score: `{item['avg_judge_score']:.4f}`",
                f"- Average latency: `{item['avg_latency_ms']} ms`",
                "",
            ]
        )
    lines.extend(["## Per Question", ""])
    for row in payload["question_summary"]:
        lines.extend(
            [
                f"### {row['question_id']}: {row['question']}",
                "",
                f"- Reference: `{row['reference_answer']}`",
                f"- Strategy agent: `{row['strategy_agent_correct_count']}/{payload['repeats']}` attempts correct ; 5-pass=`{row['strategy_agent_pass_5']}`",
                f"- No-strategy agent: `{row['no_strategy_agent_correct_count']}/{payload['repeats']}` attempts correct ; 5-pass=`{row['no_strategy_agent_pass_5']}`",
                f"- Traditional hybrid: `{row['traditional_hybrid_rag_bm25_correct_count']}/{payload['repeats']}` attempts correct ; 5-pass=`{row['traditional_hybrid_rag_bm25_pass_5']}`",
                "",
                "Strategy sample answer:",
                "```text",
                row["strategy_agent_sample_answer"],
                "```",
                "",
                "No-strategy sample answer:",
                "```text",
                row["no_strategy_agent_sample_answer"],
                "```",
                "",
                "Traditional hybrid sample answer:",
                "```text",
                row["traditional_hybrid_rag_bm25_sample_answer"],
                "```",
                "",
            ]
        )
    MD_PATH.write_text("\n".join(lines), encoding="utf-8")


def _flush_payload(rows: List[Dict[str, str]], cache: Dict[str, List[Dict[str, Any]]]) -> None:
    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": str(CSV_PATH),
        "repeats": REPEATS,
        "summary": [
            _summarize_setting("strategy_agent", cache.get("strategy_agent", []), rows),
            _summarize_setting("no_strategy_agent", cache.get("no_strategy_agent", []), rows),
            _summarize_setting("traditional_hybrid_rag_bm25", cache.get("traditional_hybrid_rag_bm25", []), rows),
        ],
        "question_summary": _build_question_summary(rows, cache),
        "strategy_agent": cache.get("strategy_agent", []),
        "no_strategy_agent": cache.get("no_strategy_agent", []),
        "traditional_hybrid_rag_bm25": cache.get("traditional_hybrid_rag_bm25", []),
    }
    JSON_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_report(payload)


def _run_setting(
    *,
    rows: List[Dict[str, str]],
    setting_name: str,
    evaluator: Any,
    max_workers: int,
    cache: Dict[str, List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    existing = {(str(item.get("question_id", "") or ""), int(item.get("run_index", 0) or 0)) for item in cache.get(setting_name, [])}
    tasks: List[Tuple[Dict[str, str], int]] = []
    for row in rows:
        for run_index in range(REPEATS):
            if (row["question_id"], run_index) not in existing:
                tasks.append((row, run_index))

    if not tasks:
        items = sorted(cache.get(setting_name, []), key=lambda x: (int(x.get("question_index", 0) or 0), int(x.get("run_index", 0) or 0)))
        cache[setting_name] = items
        return items

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        future_map = {ex.submit(evaluator.evaluate_row, row, run_index): (row, run_index) for row, run_index in tasks}
        for fut in as_completed(future_map):
            result = fut.result()
            cache.setdefault(setting_name, []).append(result.to_dict())
            cache[setting_name] = sorted(
                cache[setting_name],
                key=lambda x: (int(x.get("question_index", 0) or 0), int(x.get("run_index", 0) or 0)),
            )
            _flush_payload(rows, cache)
            print(
                json.dumps(
                    {
                        "setting": setting_name,
                        "question_id": result.question_id,
                        "run_index": result.run_index,
                        "correct": bool((result.judge or {}).get("is_correct", False)),
                        "latency_ms": result.latency_ms,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    items = sorted(cache.get(setting_name, []), key=lambda x: (int(x.get("question_index", 0) or 0), int(x.get("run_index", 0) or 0)))
    cache[setting_name] = items
    return items


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    rows = _load_rows()
    cache = _load_cached_results(JSON_PATH)
    requested_settings = {
        str(x).strip()
        for x in str(os.environ.get("NKW_EVAL_SETTINGS", "") or "").split(",")
        if str(x).strip()
    }
    if not requested_settings:
        requested_settings = {
            "strategy_agent",
            "no_strategy_agent",
            "traditional_hybrid_rag_bm25",
        }

    strategy_cfg = KAGConfig.from_yaml("configs/config_openai.yaml")
    no_strategy_cfg = KAGConfig.from_yaml("configs/config_openai.yaml")
    hybrid_cfg = KAGConfig.from_yaml("configs/config_openai.yaml")

    strategy_evaluator = AgentEvaluator(
        strategy_cfg,
        setting_name="strategy_agent",
        strategy_enabled=True,
    )
    no_strategy_evaluator = AgentEvaluator(
        no_strategy_cfg,
        setting_name="no_strategy_agent",
        strategy_enabled=False,
    )
    hybrid_evaluator = TraditionalHybridEvaluator(hybrid_cfg)

    try:
        if "strategy_agent" in requested_settings:
            _run_setting(
                rows=rows,
                setting_name="strategy_agent",
                evaluator=strategy_evaluator,
                max_workers=AGENT_MAX_WORKERS,
                cache=cache,
            )
        if "no_strategy_agent" in requested_settings:
            _run_setting(
                rows=rows,
                setting_name="no_strategy_agent",
                evaluator=no_strategy_evaluator,
                max_workers=AGENT_MAX_WORKERS,
                cache=cache,
            )
        if "traditional_hybrid_rag_bm25" in requested_settings:
            _run_setting(
                rows=rows,
                setting_name="traditional_hybrid_rag_bm25",
                evaluator=hybrid_evaluator,
                max_workers=HYBRID_MAX_WORKERS,
                cache=cache,
            )
    finally:
        strategy_evaluator.close()
        no_strategy_evaluator.close()

    _flush_payload(rows, cache)


if __name__ == "__main__":
    main()

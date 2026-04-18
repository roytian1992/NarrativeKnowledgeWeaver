from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

from core.agent.retriever_agent import QuestionAnsweringAgent
from core.functions.memory_management.judge_retrieval_answer import RetrievalAnswerJudge
from core.model_providers.openai_llm import OpenAILLM
from core.utils.config import KAGConfig
from core.utils.general_utils import json_dump_atomic
from core.utils.prompt_loader import YAMLPromptLoader
from experiments.quality.run_quality_benchmark import (
    _extract_json_object,
    _extract_llm_text,
    _extract_semantic_answer_text,
    _format_open_question_for_agent,
    _format_open_question_for_agent_with_retrieval_guard,
    _has_error_payload,
    _looks_like_tool_call_payload,
    _resolve_article_workspace_dir,
    _resolve_cli_path,
    _summarize_setting,
    _summarize_tool_uses_for_finalization,
    _update_workspace_asset_registry,
    _write_setting_progress,
    ensure_article_ready,
    load_existing_article_workspace_or_raise,
)

logger = logging.getLogger(__name__)


FIARY_ASSISTANT_RAG_CFG: Dict[str, Any] = {
    "assistant_runtime_variant": "s4",
    "max_tool_rounds_per_run": 3,
    "first_round_max_tool_calls": 3,
    "followup_round_max_tool_calls": 3,
    "parallel_tool_workers": 4,
}


@dataclass
class OpenEvalResult:
    setting: str
    article_name: str
    run_index: int
    question_id: str
    question: str
    reference_answer: str
    predicted_answer: str
    is_correct: bool
    latency_ms: int
    extra: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "setting": self.setting,
            "article_name": self.article_name,
            "run_index": self.run_index,
            "question_id": self.question_id,
            "question": self.question,
            "reference_answer": self.reference_answer,
            "predicted_answer": self.predicted_answer,
            "is_correct": self.is_correct,
            "latency_ms": self.latency_ms,
            **self.extra,
        }


def _default_workspace_asset_root() -> Path:
    return REPO_ROOT / "experiments" / "fiarytableqa" / "assets" / "article_workspaces"


def _default_converted_article_root() -> Path:
    return REPO_ROOT / "experiments" / "fiarytableqa" / "assets" / "converted_articles"


def _story_title_from_slug(slug: str) -> str:
    raw = str(slug or "").strip().replace("_", " ").replace("-", " ")
    return " ".join(piece for piece in raw.split() if piece).strip() or slug


def _convert_fiarytable_article(*, article_name: str, src_path: Path, dst_path: Path) -> None:
    payload = json.loads(src_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected article JSON list: {src_path}")
    title = _story_title_from_slug(article_name)
    out: List[Dict[str, Any]] = []
    for idx, item in enumerate(payload, start=1):
        if not isinstance(item, dict):
            continue
        content = str(item.get("content", "") or "").strip()
        if not content:
            continue
        raw_id = item.get("id")
        part_id = str(raw_id).strip() if raw_id is not None and str(raw_id).strip() else str(idx)
        out.append(
            {
                "id": f"{article_name}_part_{part_id}",
                "title": title,
                "subtitle": f"Part {part_id}",
                "content": content,
            }
        )
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    json_dump_atomic(str(dst_path), out)


def _format_reference_answers(answers: Sequence[str]) -> str:
    cleaned = []
    seen = set()
    for ans in answers or []:
        text = str(ans or "").strip()
        norm = text.lower()
        if not text or norm in seen:
            continue
        seen.add(norm)
        cleaned.append(text)
    if not cleaned:
        return "(no reference answer)"
    lines = ["The following are acceptable reference answers or answer variants:"]
    lines.extend(f"- {item}" for item in cleaned)
    return "\n".join(lines)


_QUESTION_TYPE_FIELDS: Tuple[str, ...] = (
    "local-or-sum",
    "attribute1",
    "attribute2",
    "ex-or-im1",
    "ex-or-im2",
)


def _extract_question_type_fields(metadata: Dict[str, Any]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for key in _QUESTION_TYPE_FIELDS:
        value = str((metadata or {}).get(key, "") or "").strip()
        if value:
            out[key] = value
    return out


def _extract_question_type_tags(fields: Dict[str, str]) -> List[str]:
    tags = [f"{key}:{value}" for key, value in fields.items() if str(value or "").strip()]
    attr1 = str(fields.get("attribute1", "") or "").strip()
    attr2 = str(fields.get("attribute2", "") or "").strip()
    if attr1:
        combo = attr1 if not attr2 else f"{attr1} + {attr2}"
        tags.append(f"attribute_combo:{combo}")
    ex1 = str(fields.get("ex-or-im1", "") or "").strip()
    ex2 = str(fields.get("ex-or-im2", "") or "").strip()
    if ex1:
        combo = ex1 if not ex2 else f"{ex1} + {ex2}"
        tags.append(f"explicitness_combo:{combo}")
    return tags


def _load_fiarytable_qas(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            question = str((row or {}).get("question", "") or "").strip()
            if not question:
                continue
            answers: List[str] = []
            for key in ("answer1", "answer2", "answer3", "answer4", "answer5", "answer6"):
                value = str((row or {}).get(key, "") or "").strip()
                if value:
                    answers.append(value)
            metadata = dict(row or {})
            question_type_fields = _extract_question_type_fields(metadata)
            rows.append(
                {
                    "question_id": str((row or {}).get("question_id", "") or "").strip() or f"q{len(rows)}",
                    "question": question,
                    "reference_answers": answers,
                    "reference_answer": _format_reference_answers(answers),
                    "metadata": metadata,
                    "question_type_fields": question_type_fields,
                    "question_type_tags": _extract_question_type_tags(question_type_fields),
                }
            )
    return rows


def _ensure_open_answer_payload(answer_text: str, *, default_confidence: float = 0.62) -> str:
    raw = str(answer_text or "").strip()
    if not raw:
        return raw
    payload = _extract_json_object(raw)
    if isinstance(payload, dict):
        answer_value = str(
            payload.get("answer_text")
            or payload.get("answer")
            or payload.get("final_answer")
            or ""
        ).strip()
        if answer_value:
            payload["answer_text"] = answer_value
            if payload.get("confidence") is None:
                payload["confidence"] = float(default_confidence)
            if payload.get("evidence") is None:
                payload["evidence"] = ""
            return json.dumps(payload, ensure_ascii=False)
    semantic = _extract_semantic_answer_text(raw)
    if not semantic:
        semantic = raw
    return json.dumps(
        {
            "answer_text": semantic,
            "evidence": "",
            "confidence": float(default_confidence),
        },
        ensure_ascii=False,
    )


class OpenAgentThreadLocal:
    def __init__(self, cfg: KAGConfig, *, setting_name: str, enable_sql_tools: bool) -> None:
        self.cfg = cfg
        self.setting_name = setting_name
        self.enable_sql_tools = bool(enable_sql_tools)
        self.local = threading.local()
        self._states: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        base_hidden_tool_names: set[str] = {
            "get_co_section_entities",
            "get_k_hop_subgraph",
            "get_common_neighbors",
            "get_relations_between_entities",
            "find_paths_between_nodes",
            "vdb_search_docs",
            "vdb_search_hierdocs",
            "search_related_content",
            "choice_grounded_evidence_search",
            "community_graphrag_search",
            "search_communities",
            "implication_constrained_inference_search",
            "lookup_document_ids_by_title",
            "fact_timeline_resolution_search",
        }
        extra_hidden_tool_names = {
            str(name or "").strip()
            for name in (getattr(getattr(cfg, "strategy_memory", None), "hidden_tool_names", []) or [])
            if str(name or "").strip()
        }
        self.hidden_tool_names = base_hidden_tool_names | extra_hidden_tool_names

    def _build_state(self) -> Dict[str, Any]:
        agent = QuestionAnsweringAgent(
            self.cfg,
            aggregation_mode="narrative",
            enable_sql_tools=self.enable_sql_tools,
            rag_cfg=dict(FIARY_ASSISTANT_RAG_CFG),
        )
        setattr(agent, "hidden_tool_names", set(self.hidden_tool_names))
        rebuild = getattr(agent, "_rebuild_assistant", None)
        if callable(rebuild):
            rebuild()
        prompt_loader = YAMLPromptLoader(self.cfg.global_config.prompt_dir)
        finalizer_llm = OpenAILLM(self.cfg, llm_profile="retriever")
        judge_llm = OpenAILLM(self.cfg, llm_profile="retriever")
        judge = RetrievalAnswerJudge(
            prompt_loader=prompt_loader,
            llm=judge_llm,
            prompt_id="memory/judge_open_retrieval_answer",
        )
        state = {
            "agent": agent,
            "finalizer_llm": finalizer_llm,
            "judge": judge,
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


class OpenAgentEvaluator:
    def __init__(self, cfg: KAGConfig, *, setting_name: str, article_name: str, enable_sql_tools: bool) -> None:
        self.cfg = cfg
        self.setting_name = setting_name
        self.article_name = article_name
        self.tlocal = OpenAgentThreadLocal(cfg, setting_name=setting_name, enable_sql_tools=enable_sql_tools)

    def _repair_open_answer(
        self,
        *,
        state: Dict[str, Any],
        row: Dict[str, Any],
        tool_uses: List[Dict[str, Any]],
        current_answer: str,
    ) -> str:
        if not tool_uses:
            return current_answer
        evidence_block = _summarize_tool_uses_for_finalization(tool_uses, max_items=8)
        if not evidence_block:
            return current_answer
        prompt = "\n".join(
            [
                "You are the final open-question answer adapter.",
                "Your job is to convert retrieved evidence and any incomplete prior answer into one grounded final answer.",
                "Return JSON only:",
                '{"answer_text":"...","evidence":"...","confidence":0.72}',
                "",
                "Rules:",
                "- Answer the question directly in one short sentence or phrase.",
                "- Use only the retrieved evidence below.",
                "- `evidence` must be brief and grounded.",
                "- `confidence` must be a number between 0 and 1.",
                "- If the evidence is ambiguous, answer conservatively instead of inventing details.",
                "- Do not return a tool call, a plan, or an intermediate analysis.",
                "",
                "Question:",
                str(row.get("question", "") or ""),
                "",
                "Retrieved evidence:",
                evidence_block,
                "",
                "Previous answer or unfinished draft:",
                _trim_text(current_answer, limit=700) or "(none)",
            ]
        )
        try:
            result = state["finalizer_llm"].run([{"role": "user", "content": prompt}])
            adapted = _extract_llm_text(result)
            semantic = _extract_semantic_answer_text(adapted)
            if semantic and not _looks_like_tool_call_payload(adapted):
                return _ensure_open_answer_payload(adapted)
        except Exception:
            return current_answer
        return current_answer

    def evaluate_row(self, row: Dict[str, Any], run_index: int) -> OpenEvalResult:
        started = time.time()
        state = self.tlocal.state()
        responses: List[Dict[str, Any]] = []
        final_answer = ""
        raw_agent_answer = ""
        tool_uses: List[Dict[str, Any]] = []
        strategy_context: Dict[str, Any] = {}
        error_text = ""
        prompt_variants = [
            ("base", _format_open_question_for_agent(str(row.get("question", "") or "")), (8, 10, 12)),
            ("guarded", _format_open_question_for_agent_with_retrieval_guard(str(row.get("question", "") or "")), (10, 12, 14)),
        ]
        for prompt_tag, prompt, max_calls_seq in prompt_variants:
            for max_calls in max_calls_seq:
                try:
                    responses = state["agent"].ask(
                        prompt,
                        lang="en",
                        session_id=f"{self.setting_name}_{self.article_name}_{row['question_id']}_{run_index}_{prompt_tag}_{max_calls}",
                        max_llm_calls_per_run=max_calls,
                        require_tool_use=True,
                        _router_branch_index=run_index,
                    )
                    final_answer = state["agent"].extract_final_text(responses)
                    raw_agent_answer = final_answer
                    tool_uses = state["agent"].extract_tool_uses(responses)
                    strategy_context = state["agent"].get_last_strategy_context()
                    error_text = ""
                    if not tool_uses or _has_error_payload(final_answer):
                        error_text = "missing_tool_use" if not tool_uses else "llm_error_payload"
                        continue
                    break
                except Exception as exc:
                    error_text = f"{type(exc).__name__}: {exc}"
                    if "maximum context length" not in error_text.lower():
                        break
            if tool_uses and not _has_error_payload(final_answer):
                break

        if tool_uses and (
            not str(final_answer or "").strip()
            or _looks_like_tool_call_payload(final_answer)
            or _has_error_payload(final_answer)
        ):
            final_answer = self._repair_open_answer(
                state=state,
                row=row,
                tool_uses=tool_uses,
                current_answer=final_answer,
            )

        if tool_uses and final_answer:
            final_answer = _ensure_open_answer_payload(final_answer)

        predicted_answer = _extract_semantic_answer_text(final_answer)
        judge_result = {
            "is_correct": False,
            "score": 0.0,
            "reason": error_text or "evaluation_not_run",
            "matched_points": [],
            "missing_points": [],
            "hallucination_points": [],
        }
        if tool_uses and predicted_answer:
            try:
                judge_result = state["judge"].evaluate(
                    question=str(row.get("question", "") or ""),
                    reference_answer=str(row.get("reference_answer", "") or ""),
                    candidate_answer=predicted_answer,
                )
            except Exception as exc:
                judge_result = {
                    "is_correct": False,
                    "score": 0.0,
                    "reason": f"judge_error: {type(exc).__name__}: {exc}",
                    "matched_points": [],
                    "missing_points": [],
                    "hallucination_points": [],
                }

        return OpenEvalResult(
            setting=self.setting_name,
            article_name=self.article_name,
            run_index=run_index,
            question_id=str(row.get("question_id", "") or ""),
            question=str(row.get("question", "") or ""),
            reference_answer=str(row.get("reference_answer", "") or ""),
            predicted_answer=predicted_answer,
            is_correct=bool(judge_result.get("is_correct", False)),
            latency_ms=int((time.time() - started) * 1000),
            extra={
                "tool_call_count": len(tool_uses),
                "tool_names": [
                    str((item or {}).get("tool_name", "") or "").strip()
                    for item in tool_uses
                    if str((item or {}).get("tool_name", "") or "").strip()
                ],
                "tool_uses": tool_uses,
                "strategy_context": strategy_context,
                "raw_agent_answer": raw_agent_answer,
                "judge": judge_result,
                "reference_answers": list(row.get("reference_answers") or []),
                "question_metadata": dict(row.get("metadata") or {}),
                "question_type_fields": dict(row.get("question_type_fields") or {}),
                "question_type_tags": list(row.get("question_type_tags") or []),
                "error": error_text,
                "required_retrieval_enforced": True,
            },
        )

    def close(self) -> None:
        self.tlocal.close()


def _trim_text(value: Any, *, limit: int = 1200) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def _evaluate_setting(*, rows: List[Dict[str, Any]], evaluator: OpenAgentEvaluator, repeats: int, max_workers: int) -> List[Dict[str, Any]]:
    tasks: List[Tuple[Dict[str, Any], int]] = []
    for row in rows:
        for run_index in range(repeats):
            tasks.append((row, run_index))
    results: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, int(max_workers or 1))) as executor:
        future_map = {executor.submit(evaluator.evaluate_row, row, run_index): (row, run_index) for row, run_index in tasks}
        for future in as_completed(future_map):
            results.append(future.result().to_dict())
    results.sort(key=lambda item: (item["article_name"], item["question_id"], int(item["run_index"])))
    return results


def _summarize_open_setting_base(results: List[Dict[str, Any]], *, question_count: int, repeats: int) -> Dict[str, Any]:
    summary = _summarize_setting(results, question_count=question_count, repeats=repeats)
    scores = [float(((item.get("judge") or {}).get("score", 0.0) or 0.0)) for item in results]
    summary["avg_judge_score"] = round(sum(scores) / float(len(scores) or 1), 4)
    return summary


def _summarize_open_setting(results: List[Dict[str, Any]], *, question_count: int, repeats: int) -> Dict[str, Any]:
    summary = _summarize_open_setting_base(results, question_count=question_count, repeats=repeats)

    type_breakdown: Dict[str, Dict[str, Any]] = {}
    for field in _QUESTION_TYPE_FIELDS:
        buckets: Dict[str, List[Dict[str, Any]]] = {}
        for item in results:
            fields = item.get("question_type_fields") or {}
            if not isinstance(fields, dict):
                continue
            label = str(fields.get(field, "") or "").strip()
            if not label:
                continue
            buckets.setdefault(label, []).append(item)
        if not buckets:
            continue
        field_summary: Dict[str, Any] = {}
        for label, bucket in sorted(buckets.items()):
            bucket_question_count = len({(str(x.get("article_name", "") or ""), str(x.get("question_id", "") or "")) for x in bucket})
            field_summary[label] = _summarize_open_setting_base(
                bucket,
                question_count=bucket_question_count,
                repeats=repeats,
            )
        type_breakdown[field] = field_summary

    if type_breakdown:
        summary["type_breakdown"] = type_breakdown
    return summary


def _collect_article_pairs(benchmark_root: Path) -> List[Tuple[str, Path, Path]]:
    articles_dir = benchmark_root / "articles"
    questions_dir = benchmark_root / "questions"
    article_map = {p.stem: p for p in sorted(articles_dir.glob("*.json"))}
    question_map = {p.stem: p for p in sorted(questions_dir.glob("*.csv"))}
    names = sorted(set(article_map) & set(question_map))
    return [(name, article_map[name], question_map[name]) for name in names]


def _build_base_cfg(config_path: Path) -> KAGConfig:
    cfg = KAGConfig.from_yaml(str(config_path))
    cfg.global_.language = "en"
    cfg.global_.locale = "en"
    cfg.global_config.language = "en"
    cfg.global_config.locale = "en"
    _apply_fiarytable_document_profile(cfg)
    return cfg


def _apply_fiarytable_document_profile(cfg: KAGConfig) -> None:
    """
    FiarytableQA article JSONs are already segmented into short story parts.
    Keep each original `content` as a single document chunk so we avoid the
    extra chunk+merge/split phase while preserving the downstream chunk schema,
    summary generation, and metadata extraction flow.
    """
    dp_cfg = getattr(cfg, "document_processing", None)
    if dp_cfg is None:
        return
    # Corpus scan on FiarytableQA:
    # - max chars per content: 2193
    # - max words per content: 446
    # Set a conservative threshold above the observed maximum so every raw
    # content item passes through as one chunk.
    dp_cfg.chunk_size = max(int(getattr(dp_cfg, "chunk_size", 600) or 600), 2500)
    dp_cfg.max_content_size = max(int(getattr(dp_cfg, "max_content_size", 1200) or 1200), 2500)
    dp_cfg.max_segments = max(1, int(getattr(dp_cfg, "max_segments", 3) or 3))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config_openai_quality_stable.yaml")
    parser.add_argument("--benchmark-root", default="/vepfs-mlp2/c20250513/241404044/users/roytian/benchmarks/FiarytableQA")
    parser.add_argument("--limit-articles", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--eval-max-workers", type=int, default=8)
    parser.add_argument("--enable-sql-tools", action="store_true", default=True)
    parser.add_argument("--disable-sql-tools", action="store_true")
    parser.add_argument("--rebuild-workspaces", action="store_true")
    parser.add_argument("--workspace-source-root", default="")
    parser.add_argument("--workspace-asset-root", default="experiments/fiarytableqa/assets/article_workspaces")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--max-questions-per-article", type=int, default=0)
    parser.add_argument("--reuse-existing-only", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    benchmark_root = _resolve_cli_path(args.benchmark_root)
    config_path = _resolve_cli_path(args.config)
    workspace_source_root = _resolve_cli_path(args.workspace_source_root) if str(args.workspace_source_root or "").strip() else None
    workspace_asset_root = _resolve_cli_path(args.workspace_asset_root) if str(args.workspace_asset_root or "").strip() else _default_workspace_asset_root()
    workspace_asset_root.mkdir(parents=True, exist_ok=True)
    converted_article_root = _default_converted_article_root()
    converted_article_root.mkdir(parents=True, exist_ok=True)

    run_name = str(args.run_name or "").strip() or time.strftime("fiarytableqa_no_strategy_%Y%m%d_%H%M%S")
    run_root = REPO_ROOT / "experiments" / "fiarytableqa" / "runs" / run_name
    report_root = run_root / "reports"
    workspace_root = run_root / "article_workspaces"
    for path in (run_root, report_root, workspace_root):
        path.mkdir(parents=True, exist_ok=True)

    base_cfg = _build_base_cfg(config_path)
    enable_sql_tools = bool(args.enable_sql_tools) and not bool(args.disable_sql_tools)

    article_pairs = _collect_article_pairs(benchmark_root)
    if args.limit_articles and args.limit_articles > 0:
        article_pairs = article_pairs[: int(args.limit_articles)]
    manifest = [
        {
            "article_name": name,
            "article_json": str(article_path),
            "question_csv": str(question_path),
        }
        for name, article_path, question_path in article_pairs
    ]
    json_dump_atomic(str(run_root / "manifest.json"), manifest)

    all_results: List[Dict[str, Any]] = []
    article_summaries: Dict[str, Any] = {}
    total_questions = 0
    progress_path = report_root / "progress.json"

    for article_index, (article_name, article_src_path, question_csv_path) in enumerate(article_pairs, start=1):
        converted_article_path = converted_article_root / f"{article_name}.json"
        if not converted_article_path.exists():
            _convert_fiarytable_article(
                article_name=article_name,
                src_path=article_src_path,
                dst_path=converted_article_path,
            )

        workspace_dir = _resolve_article_workspace_dir(
            article_name=article_name,
            workspace_root=workspace_root,
            workspace_source_root=workspace_source_root,
            workspace_asset_root=workspace_asset_root,
        )
        if bool(args.reuse_existing_only):
            cfg = load_existing_article_workspace_or_raise(
                base_cfg=base_cfg,
                workspace_dir=workspace_dir,
                strategy_enabled=False,
                subagent_enabled=False,
            )
        else:
            cfg = ensure_article_ready(
                base_cfg=base_cfg,
                article_json_path=converted_article_path,
                workspace_dir=workspace_dir,
                strategy_enabled=False,
                subagent_enabled=False,
                rebuild=bool(args.rebuild_workspaces),
            )
        cfg.strategy_memory.hidden_tool_names = list(
            {
                *list(getattr(cfg.strategy_memory, "hidden_tool_names", []) or []),
                "choice_grounded_evidence_search",
                "lookup_document_ids_by_title",
                "fact_timeline_resolution_search",
            }
        )
        _update_workspace_asset_registry(
            workspace_asset_root=workspace_asset_root,
            article_name=article_name,
            workspace_dir=workspace_dir,
            article_json_path=converted_article_path,
        )

        qa_rows = _load_fiarytable_qas(question_csv_path)
        if args.max_questions_per_article and args.max_questions_per_article > 0:
            qa_rows = qa_rows[: int(args.max_questions_per_article)]
        total_questions += len(qa_rows)

        evaluator = OpenAgentEvaluator(
            cfg,
            setting_name="no_strategy_agent",
            article_name=article_name,
            enable_sql_tools=enable_sql_tools,
        )
        try:
            article_results = _evaluate_setting(
                rows=qa_rows,
                evaluator=evaluator,
                repeats=max(1, int(args.repeats or 1)),
                max_workers=max(1, int(args.eval_max_workers or 1)),
            )
        finally:
            evaluator.close()

        summary = _summarize_open_setting(
            article_results,
            question_count=len(qa_rows),
            repeats=max(1, int(args.repeats or 1)),
        )
        article_payload = {
            "setting": "no_strategy_agent",
            "article_name": article_name,
            "question_count": len(qa_rows),
            "repeats": max(1, int(args.repeats or 1)),
            "results": article_results,
            "summary": summary,
        }
        json_dump_atomic(str(report_root / f"{article_name}.json"), article_payload)
        article_summaries[article_name] = summary
        all_results.extend(article_results)

        _write_setting_progress(
            path=progress_path,
            setting_name="no_strategy_agent",
            article_name=article_name,
            repeats=max(1, int(args.repeats or 1)),
            repeat_index=max(1, int(args.repeats or 1)),
            batch_index=article_index,
            batch_total=len(article_pairs),
            phase="completed_article",
            evaluated_attempts_done=len(all_results),
            evaluated_attempts_total=total_questions * max(1, int(args.repeats or 1)),
            batch_question_count=len(qa_rows),
            note=f"completed {article_index}/{len(article_pairs)} articles",
        )
        logger.info(
            "[%d/%d] article=%s overall=%.4f pass=%.4f avg_judge=%.4f",
            article_index,
            len(article_pairs),
            article_name,
            float(summary.get("overall_accuracy", 0.0) or 0.0),
            float(summary.get("pass_accuracy", 0.0) or 0.0),
            float(summary.get("avg_judge_score", 0.0) or 0.0),
        )

    overall_summary = _summarize_open_setting(
        all_results,
        question_count=total_questions,
        repeats=max(1, int(args.repeats or 1)),
    )
    final_payload = {
        "benchmark_root": str(benchmark_root),
        "config": str(config_path),
        "setting": "no_strategy_agent",
        "article_count": len(article_pairs),
        "question_count": total_questions,
        "repeats": max(1, int(args.repeats or 1)),
        "summary": overall_summary,
        "article_summaries": article_summaries,
        "manifest_path": str(run_root / "manifest.json"),
    }
    json_dump_atomic(str(report_root / "summary.json"), final_payload)
    md_lines = [
        "# FiarytableQA No-Strategy Agent Summary",
        "",
        f"- article_count: {len(article_pairs)}",
        f"- question_count: {total_questions}",
        f"- repeats: {max(1, int(args.repeats or 1))}",
        f"- overall_accuracy: {overall_summary['overall_accuracy']}",
        f"- pass_accuracy: {overall_summary['pass_accuracy']}",
        f"- avg_judge_score: {overall_summary['avg_judge_score']}",
        f"- avg_latency_ms: {overall_summary['avg_latency_ms']}",
        "",
        "## Per Article",
        "",
    ]
    for article_name, summary in sorted(article_summaries.items()):
        md_lines.append(
            f"- {article_name}: overall={summary['overall_accuracy']} pass={summary['pass_accuracy']} avg_judge={summary['avg_judge_score']}"
        )

    type_breakdown = overall_summary.get("type_breakdown") if isinstance(overall_summary.get("type_breakdown"), dict) else {}
    if type_breakdown:
        md_lines.extend(["", "## Type Breakdown", ""])
        for field, field_payload in sorted(type_breakdown.items()):
            md_lines.append(f"### {field}")
            md_lines.append("")
            for label, bucket_summary in sorted((field_payload or {}).items()):
                md_lines.append(
                    f"- {label}: overall={bucket_summary['overall_accuracy']} pass={bucket_summary['pass_accuracy']} avg_judge={bucket_summary['avg_judge_score']}"
                )
            md_lines.append("")

    (report_root / "summary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    logger.info("Finished FiarytableQA benchmark. summary=%s", report_root / "summary.json")


if __name__ == "__main__":
    main()

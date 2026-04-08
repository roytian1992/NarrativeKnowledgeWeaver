from __future__ import annotations

import copy
import logging
import csv
import json
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from tqdm import tqdm

from core.agent.retriever_agent import QuestionAnsweringAgent
from core.functions.tool_calls.tool_metadata import ToolMetadataProvider
from core.functions.memory_management.effective_tool_chain_extractor import EffectiveToolChainExtractor
from core.functions.memory_management.failure_retry import FailedAnswerReflector, RetryInstructionBuilder
from core.functions.memory_management.online_sampling_branch_planner import OnlineSamplingBranchPlanner
from core.functions.memory_management.judge_retrieval_answer import RetrievalAnswerJudge
from core.functions.memory_management.strategy_failure_summarizer import StrategyFailureSummarizer
from core.functions.memory_management.strategy_query_pattern import StrategyQueryPatternExtractor
from core.functions.memory_management.strategy_template_distiller import StrategyTemplateDistiller
from core.functions.memory_management.tool_description_reflector import ToolDescriptionReflector
from core.strategy_training.strategy_cluster_manager import StrategyTemplateClusterManager
from core.strategy_training.strategy_runtime_assets import StrategyRuntimeAssetManager
from core.model_providers.openai_llm import OpenAILLM
from core.storage.vector_store import VectorStore
from core.utils.config import KAGConfig
from core.utils.general_utils import json_dump_atomic
from core.utils.prompt_loader import YAMLPromptLoader


logger = logging.getLogger(__name__)


class StrategyMemoryTrainingRunner:
    def __init__(
        self,
        *,
        config: KAGConfig,
        csv_path: str,
        dataset_name: str = "default_dataset",
        attempts_per_question: int = 5,
        question_limit: Optional[int] = None,
        output_root: str = "strategy_training",
        runtime_library_path: Optional[str] = None,
        answer_temperature: Optional[float] = None,
        enable_sql_tools: bool = True,
        max_llm_calls_per_attempt: int = 10,
    ) -> None:
        self.repo_root = Path(__file__).resolve().parents[2]
        self.config = copy.deepcopy(config)
        self.csv_path = Path(csv_path)
        self.dataset_name = str(dataset_name or self.csv_path.stem or "default_dataset").strip()
        self.attempts_per_question = max(1, int(attempts_per_question or 1))
        self.question_limit = question_limit if question_limit is None else max(1, int(question_limit))
        self.output_root = self.repo_root / str(output_root or "strategy_training") / self.dataset_name
        self.runtime_library_path = Path(runtime_library_path) if runtime_library_path else self.repo_root / str(
            getattr(self.config.strategy_memory, "library_path", "data/memory/strategy/strategy_library.json")
        )
        self.runtime_tool_metadata_dir = self.repo_root / str(
            getattr(self.config.strategy_memory, "tool_metadata_runtime_dir", "data/memory/strategy/tool_metadata")
        )
        self.answer_temperature = answer_temperature
        self.enable_sql_tools = bool(enable_sql_tools)
        self.max_llm_calls_per_attempt = max(1, int(max_llm_calls_per_attempt or 1))

        self.manifest_path = self.output_root / "manifests" / "training_manifest.json"
        self.attempts_path = self.output_root / "attempts" / "attempt_runs.jsonl"
        self.judgments_path = self.output_root / "judged" / "answer_judgments.jsonl"
        self.effective_chain_path = self.output_root / "judged" / "effective_tool_chains.jsonl"
        self.question_summary_path = self.output_root / "distilled" / "question_training_summaries.json"
        self.question_detail_dir = self.output_root / "distilled" / "per_question"
        self.progress_path = self.output_root / "progress" / "training_progress.json"
        self.live_attempts_path = self.output_root / "progress" / "attempt_runs_live.jsonl"
        self.live_retry_attempts_path = self.output_root / "progress" / "retry_attempts_live.jsonl"
        self.raw_template_path = self.output_root / "distilled" / "raw_templates.json"
        self.failure_summary_path = self.output_root / "failures" / "failed_question_summaries.json"
        self.reflection_path = self.output_root / "failures" / "attempt_reflections.jsonl"
        self.retry_attempts_path = self.output_root / "attempts" / "retry_attempts.jsonl"
        self.tool_reflection_record_path = self.output_root / "tool_metadata" / "tool_description_reflection_records.jsonl"
        self.tool_description_candidate_path = self.output_root / "tool_metadata" / "tool_description_candidates.json"
        self.cluster_path = self.output_root / "clusters" / "template_clusters.json"
        self.merge_decision_path = self.output_root / "clusters" / "merge_decisions.jsonl"
        self.library_output_path = self.output_root / "library" / "strategy_library.json"
        self.template_source_index_path = self.output_root / "library" / "template_source_index.json"
        self.report_path = self.output_root / "report.md"
        self.runtime_asset_manager = StrategyRuntimeAssetManager(
            library_path=str(self.runtime_library_path),
            tool_metadata_runtime_dir=str(self.runtime_tool_metadata_dir),
        )
        self.training_max_workers = max(1, int(getattr(self.config.strategy_memory, "training_max_workers", 16) or 16))

        for path in [
            self.manifest_path,
            self.attempts_path,
            self.judgments_path,
            self.effective_chain_path,
            self.question_summary_path,
            self.raw_template_path,
            self.failure_summary_path,
            self.reflection_path,
            self.retry_attempts_path,
            self.progress_path,
            self.live_attempts_path,
            self.live_retry_attempts_path,
            self.tool_reflection_record_path,
            self.tool_description_candidate_path,
            self.cluster_path,
            self.merge_decision_path,
            self.library_output_path,
            self.template_source_index_path,
            self.report_path,
        ]:
            path.parent.mkdir(parents=True, exist_ok=True)
        self.question_detail_dir.mkdir(parents=True, exist_ok=True)
        for path in [self.live_attempts_path, self.live_retry_attempts_path]:
            if not path.exists():
                path.touch()

        self.prompt_loader = YAMLPromptLoader(self.config.global_config.prompt_dir)
        self.llm = OpenAILLM(self.config, llm_profile="memory")
        self.query_pattern_extractor = StrategyQueryPatternExtractor(
            prompt_loader=self.prompt_loader,
            llm=self.llm,
            prompt_id=getattr(self.config.strategy_memory, "pattern_extractor_prompt_id", "memory/extract_strategy_query_pattern"),
            abstraction_mode=getattr(self.config.strategy_memory, "abstraction_mode", "hybrid"),
        )
        self.answer_judge = RetrievalAnswerJudge(self.prompt_loader, self.llm)
        self.failed_answer_reflector = FailedAnswerReflector(
            self.prompt_loader,
            self.llm,
            prompt_id=getattr(self.config.strategy_memory, "failed_answer_reflection_prompt_id", "memory/reflect_failed_answer"),
        )
        self.retry_instruction_builder = RetryInstructionBuilder(
            self.prompt_loader,
            self.llm,
            prompt_id=getattr(self.config.strategy_memory, "retry_instruction_prompt_id", "memory/build_retry_instruction"),
        )
        self.chain_extractor = EffectiveToolChainExtractor(self.prompt_loader, self.llm)
        self.failure_summarizer = StrategyFailureSummarizer(self.prompt_loader, self.llm)
        self.template_distiller = StrategyTemplateDistiller(self.prompt_loader, self.llm)
        self.tool_description_reflector = ToolDescriptionReflector(
            self.prompt_loader,
            self.llm,
            prompt_id=getattr(self.config.strategy_memory, "tool_description_prompt_id", "memory/optimize_tool_description"),
        )
        self.tool_metadata_provider = ToolMetadataProvider.from_config(self.config)

        self.embedding_store = VectorStore(self.config, "document")
        self.embedding_model = getattr(self.embedding_store, "embedding_model", None)
        self.cluster_manager = StrategyTemplateClusterManager(
            prompt_loader=self.prompt_loader,
            llm=self.llm,
            embedding_model=self.embedding_model,
            candidate_top_k=getattr(self.config.strategy_memory, "merge_candidate_top_k", 3),
            min_candidate_score=getattr(self.config.strategy_memory, "merge_min_candidate_score", 0.28),
            consolidation_rounds=getattr(self.config.strategy_memory, "consolidation_rounds", 1),
            max_members_for_distill_prompt=getattr(self.config.strategy_memory, "cluster_distill_max_members", 12),
        )
        self.max_retry_per_attempt = max(0, int(getattr(self.config.strategy_memory, "max_retry_per_attempt", 3) or 0))
        self.question_timeout_sec = max(0, int(getattr(self.config.strategy_memory, "question_timeout_sec", 900) or 0))
        self.question_timeout_retry_rounds = max(
            0,
            int(getattr(self.config.strategy_memory, "question_timeout_retry_rounds", 2) or 0),
        )
        self.question_timeout_retry_workers = max(
            1,
            int(getattr(self.config.strategy_memory, "question_timeout_retry_workers", 4) or 1),
        )
        self.timeout_retry_max_retry_per_attempt = max(
            0,
            int(
                getattr(
                    self.config.strategy_memory,
                    "timeout_retry_max_retry_per_attempt",
                    min(1, self.max_retry_per_attempt),
                )
                or 0
            ),
        )
        self.agent = self._build_agent()
        self.tool_inventory = self._build_tool_inventory()
        self._thread_local = threading.local()
        self._worker_context_lock = threading.Lock()
        self._worker_contexts: List[Dict[str, Any]] = []
        self._progress_lock = threading.Lock()
        self._progress_state: Dict[str, Any] = {
            "dataset_name": self.dataset_name,
            "question_count_total": 0,
            "questions_completed": 0,
            "questions_running": 0,
            "questions_failed": 0,
            "attempt_records_completed": 0,
            "attempt_records_expected_upper_bound": 0,
            "updated_at": "",
            "current_questions": {},
        }

    def _build_agent(self) -> QuestionAnsweringAgent:
        agent_cfg = copy.deepcopy(self.config)
        agent_cfg.global_config.aggregation_mode = "narrative"
        agent_cfg.global_.aggregation_mode = "narrative"
        agent_cfg.strategy_memory.enabled = False
        agent_cfg.strategy_memory.read_enabled = False
        if self.answer_temperature is not None:
            agent_cfg.llm.temperature = float(self.answer_temperature)
            agent_cfg.retriever_llm.temperature = float(self.answer_temperature)
        return QuestionAnsweringAgent(
            agent_cfg,
            aggregation_mode="narrative",
            enable_sql_tools=self.enable_sql_tools,
        )

    def _create_worker_context(self) -> Dict[str, Any]:
        prompt_loader = YAMLPromptLoader(self.config.global_config.prompt_dir)
        llm = OpenAILLM(self.config, llm_profile="memory")
        ctx = {
            "prompt_loader": prompt_loader,
            "llm": llm,
            "query_pattern_extractor": StrategyQueryPatternExtractor(
                prompt_loader=prompt_loader,
                llm=llm,
                prompt_id=getattr(self.config.strategy_memory, "pattern_extractor_prompt_id", "memory/extract_strategy_query_pattern"),
                abstraction_mode=getattr(self.config.strategy_memory, "abstraction_mode", "hybrid"),
            ),
            "answer_judge": RetrievalAnswerJudge(prompt_loader, llm),
            "failed_answer_reflector": FailedAnswerReflector(
                prompt_loader,
                llm,
                prompt_id=getattr(self.config.strategy_memory, "failed_answer_reflection_prompt_id", "memory/reflect_failed_answer"),
            ),
            "retry_instruction_builder": RetryInstructionBuilder(
                prompt_loader,
                llm,
                prompt_id=getattr(self.config.strategy_memory, "retry_instruction_prompt_id", "memory/build_retry_instruction"),
            ),
            "chain_extractor": EffectiveToolChainExtractor(prompt_loader, llm),
            "failure_summarizer": StrategyFailureSummarizer(prompt_loader, llm),
            "template_distiller": StrategyTemplateDistiller(prompt_loader, llm),
            "tool_description_reflector": ToolDescriptionReflector(
                prompt_loader,
                llm,
                prompt_id=getattr(self.config.strategy_memory, "tool_description_prompt_id", "memory/optimize_tool_description"),
            ),
            "sampling_branch_planner": OnlineSamplingBranchPlanner(
                prompt_loader=prompt_loader,
                llm=llm,
                prompt_id=str(
                    getattr(
                        self.config.strategy_memory,
                        "sampling_branch_planner_prompt_id",
                        "memory/plan_trajectory_direction",
                    )
                    or "memory/plan_trajectory_direction"
                ).strip(),
            ),
            "agent": self._build_agent(),
        }
        with self._worker_context_lock:
            self._worker_contexts.append(ctx)
        return ctx

    def _get_worker_context(self) -> Dict[str, Any]:
        ctx = getattr(self._thread_local, "strategy_training_context", None)
        if ctx is None:
            ctx = self._create_worker_context()
            self._thread_local.strategy_training_context = ctx
        return ctx

    @staticmethod
    def _append_jsonl(path: Path, row: Dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    @staticmethod
    def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _initialize_progress(self, *, question_count: int) -> None:
        with self._progress_lock:
            self._progress_state = {
                "dataset_name": self.dataset_name,
                "question_count_total": int(question_count),
                "questions_completed": 0,
                "questions_running": 0,
                "questions_failed": 0,
                "attempt_records_completed": 0,
                "attempt_records_expected_upper_bound": int(question_count) * int(self.attempts_per_question),
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "current_questions": {},
            }
            self._write_progress_locked()

    def _write_progress_locked(self) -> None:
        current_questions = self._progress_state.get("current_questions", {})
        completed = int(self._progress_state.get("questions_completed", 0) or 0)
        running = int(self._progress_state.get("questions_running", 0) or 0)
        failed = int(self._progress_state.get("questions_failed", 0) or 0)
        total = int(self._progress_state.get("question_count_total", 0) or 0)
        snapshot = {
            **self._progress_state,
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "questions_pending": max(0, total - completed - running),
            "current_questions": current_questions,
        }
        json_dump_atomic(str(self.progress_path), snapshot)
        self._progress_state = snapshot

    def _mark_question_started(self, *, row: Dict[str, Any]) -> None:
        question_id = str(row.get("question_id", "") or "")
        question_text = str(row.get("question", "") or "")
        with self._progress_lock:
            current_questions = self._progress_state.setdefault("current_questions", {})
            current_questions[question_id] = {
                "question_id": question_id,
                "question": self._clip_text(question_text, limit=240),
                "status": "running",
                "attempt_records_completed": 0,
                "attempt_records_expected_upper_bound": int(self.attempts_per_question) * (self.max_retry_per_attempt + 1),
                "last_attempt_index": -1,
                "last_retry_index": -1,
                "last_attempt_correct": False,
                "last_error": "",
                "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            self._progress_state["questions_running"] = int(self._progress_state.get("questions_running", 0) or 0) + 1
            self._write_progress_locked()

    def _mark_attempt_completed(self, *, attempt_record: Dict[str, Any]) -> None:
        question_id = str(attempt_record.get("question_id", "") or "")
        with self._progress_lock:
            current_questions = self._progress_state.setdefault("current_questions", {})
            question_state = current_questions.setdefault(
                question_id,
                {
                    "question_id": question_id,
                    "question": self._clip_text(attempt_record.get("question", ""), limit=240),
                    "status": "running",
                    "attempt_records_completed": 0,
                    "attempt_records_expected_upper_bound": int(self.attempts_per_question) * (self.max_retry_per_attempt + 1),
                    "last_attempt_index": -1,
                    "last_retry_index": -1,
                    "last_attempt_correct": False,
                    "last_error": "",
                    "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                },
            )
            question_state["attempt_records_completed"] = int(question_state.get("attempt_records_completed", 0) or 0) + 1
            question_state["last_attempt_index"] = int(attempt_record.get("attempt_index", -1) or -1)
            question_state["last_retry_index"] = int(attempt_record.get("retry_index", -1) or -1)
            question_state["last_attempt_correct"] = bool((attempt_record.get("judge") or {}).get("is_correct", False))
            question_state["last_error"] = self._clip_text(attempt_record.get("error", ""), limit=240)
            question_state["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            self._progress_state["attempt_records_completed"] = int(
                self._progress_state.get("attempt_records_completed", 0) or 0
            ) + 1
            self._write_progress_locked()

    def _mark_question_finished(self, *, result: Dict[str, Any]) -> None:
        question_id = str(result.get("question_id", "") or "")
        with self._progress_lock:
            current_questions = self._progress_state.setdefault("current_questions", {})
            question_state = current_questions.get(question_id, {})
            question_state.update(
                {
                    "question_id": question_id,
                    "question": self._clip_text(result.get("question", ""), limit=240),
                    "status": "completed" if bool(result.get("successful")) else "failed",
                    "successful": bool(result.get("successful")),
                    "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
            )
            current_questions[question_id] = question_state
            self._progress_state["questions_running"] = max(
                0,
                int(self._progress_state.get("questions_running", 0) or 0) - 1,
            )
            self._progress_state["questions_completed"] = int(
                self._progress_state.get("questions_completed", 0) or 0
            ) + 1
            if not bool(result.get("successful")):
                self._progress_state["questions_failed"] = int(
                    self._progress_state.get("questions_failed", 0) or 0
                ) + 1
            self._write_progress_locked()

    @staticmethod
    def _load_rows(path: Path) -> List[Dict[str, Any]]:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            rows: List[Dict[str, Any]] = []
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

    @staticmethod
    def _clip_text(value: Any, limit: int = 1200) -> str:
        text = str(value or "").strip()
        if limit <= 0:
            return ""
        if len(text) <= limit:
            return text
        return text[: max(0, limit - 3)] + "..."

    def _build_sampling_available_tools(self) -> List[Dict[str, str]]:
        rows: List[Dict[str, str]] = []
        for tool_name, meta in sorted((self.tool_inventory or {}).items()):
            name = str(tool_name or "").strip()
            if not name:
                continue
            fallback_description = str(meta.get("description", "") or "").strip()
            resolved = self.tool_metadata_provider.resolve_tool_metadata(
                name,
                fallback_description=fallback_description,
                fallback_parameters=meta.get("parameters") or [],
            )
            rows.append(
                {
                    "name": str(resolved.get("name", "") or name).strip(),
                    "description": str(resolved.get("description", "") or fallback_description).strip(),
                }
            )
        return rows

    @staticmethod
    def _build_sampling_branch_routing_hint(spec: Dict[str, str]) -> str:
        name = str(spec.get("name", "") or "").strip()
        focus = str(spec.get("focus", "") or "").strip()
        tool_hint = str(spec.get("tool_hint", "") or "").strip()
        constraint = str(spec.get("constraint", "") or "").strip()
        lines = [
            "Sampling Branch Plan:",
            f"- name: {name}" if name else "- name: sampling_branch",
            f"- focus: {focus}" if focus else "- focus: gather evidence that can directly resolve the question",
            f"- tool_hint: {tool_hint}" if tool_hint else "- tool_hint: adaptive retrieval",
            f"- constraint: {constraint}" if constraint else "- constraint: avoid weak or redundant evidence",
            "Treat this as a soft branch bias only. Final decisions must follow retrieved evidence.",
        ]
        return "\n".join(lines)

    @staticmethod
    def _free_exploration_branch_spec() -> Dict[str, str]:
        return {
            "name": "free_exploration",
            "focus": "explore the question directly without committing to a fixed retrieval bias too early",
            "tool_hint": "adaptive retrieval based on current evidence",
            "constraint": "do not anchor on the first plausible clue; verify before concluding",
        }

    def _build_attempt_branch_specs(
        self,
        *,
        row: Dict[str, Any],
        ctx: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, str]]:
        total = max(1, int(self.attempts_per_question or 1))
        if total == 1:
            return [self._free_exploration_branch_spec()]

        ctx = ctx or self._get_worker_context()
        planned: List[Dict[str, str]] = []
        try:
            planned = ctx["sampling_branch_planner"].plan(
                question=str(row.get("question", "") or "").strip(),
                available_tools=self._build_sampling_available_tools(),
                planned_branch_count=max(0, total - 1),
            )
        except Exception as exc:
            logger.warning(
                "offline sampling branch planning failed: question_id=%s err=%s",
                row.get("question_id", ""),
                exc,
            )
            planned = []

        specs = [dict(item) for item in planned[: max(0, total - 1)] if isinstance(item, dict)]
        while len(specs) < max(0, total - 1):
            fallback_idx = len(specs) + 1
            specs.append(
                {
                    "name": f"planned_branch_{fallback_idx}",
                    "focus": "gather evidence that can directly resolve the question",
                    "tool_hint": "adaptive retrieval",
                    "constraint": "avoid weak or redundant evidence",
                }
            )
        specs.append(self._free_exploration_branch_spec())
        return specs[:total]

    @staticmethod
    def _complexity_key(item: Dict[str, Any]) -> tuple:
        effective_len = len(item.get("minimal_effective_chain") or item.get("effective_tool_chain") or [])
        raw_len = len(item.get("raw_tool_chain") or [])
        latency_ms = int(item.get("latency_ms", 0) or 0)
        output_chars = int(item.get("tool_output_chars", 0) or 0)
        return (
            effective_len if effective_len > 0 else 999,
            raw_len,
            latency_ms,
            output_chars,
        )

    @staticmethod
    def _make_attempt_id(question_id: str, attempt_index: int, retry_index: int) -> str:
        return f"{question_id or 'q'}_run_{attempt_index}_retry_{retry_index}"

    @staticmethod
    def _build_retry_query(question: str, retry_instruction: str) -> str:
        base = str(question or "").strip()
        hint = str(retry_instruction or "").strip()
        if not hint:
            return base
        return base + "\n\n补充要求：\n" + hint

    def _question_deadline_exceeded(self, started_at: float) -> bool:
        timeout_sec = int(self.question_timeout_sec or 0)
        if timeout_sec <= 0:
            return False
        return (time.time() - float(started_at)) >= float(timeout_sec)

    def _summarize_tool_uses_for_retry(
        self,
        *,
        tool_uses: List[Dict[str, Any]],
        effective_step_indices: List[int],
    ) -> Dict[str, Any]:
        effective_idx = {int(x) for x in (effective_step_indices or []) if isinstance(x, (int, float))}
        items: List[Dict[str, Any]] = []
        for idx, item in enumerate(tool_uses or []):
            tool_name = str((item or {}).get("tool_name", "") or "").strip()
            if not tool_name:
                continue
            output = self._clip_text((item or {}).get("tool_output", ""), limit=220)
            items.append(
                {
                    "step_index": idx,
                    "tool_name": tool_name,
                    "kept_in_effective_chain": idx in effective_idx,
                    "output_summary": output,
                }
            )
            if len(items) >= 8:
                break
        return {
            "raw_tool_chain": [
                str((item or {}).get("tool_name", "") or "").strip()
                for item in (tool_uses or [])
                if str((item or {}).get("tool_name", "") or "").strip()
            ][:12],
            "effective_step_indices": sorted(effective_idx),
            "tool_observations": items,
        }

    def _reflect_and_build_retry(
        self,
        *,
        row: Dict[str, Any],
        attempt_record: Dict[str, Any],
        ctx: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        ctx = ctx or self._get_worker_context()
        tool_summary = self._summarize_tool_uses_for_retry(
            tool_uses=attempt_record.get("tool_uses") or [],
            effective_step_indices=attempt_record.get("effective_step_indices") or [],
        )
        reflection = ctx["failed_answer_reflector"].reflect(
            question=str(row.get("question", "") or ""),
            reference_answer=str(row.get("answer", "") or ""),
            candidate_answer=str(attempt_record.get("final_answer", "") or ""),
            tool_summary_json=json.dumps(tool_summary, ensure_ascii=False, indent=2),
        )
        retry_payload = {"retry_instruction": ""}
        if bool(reflection.get("need_retry")):
            retry_payload = ctx["retry_instruction_builder"].build(
                question=str(row.get("question", "") or ""),
                missed_fact=str(reflection.get("missed_fact", "") or ""),
                next_action=str(reflection.get("next_action", "") or ""),
            )
        return {
            "reflection": reflection,
            "retry_instruction": str(retry_payload.get("retry_instruction", "") or "").strip(),
            "tool_summary": tool_summary,
        }

    def _run_single_attempt(
        self,
        *,
        row: Dict[str, Any],
        attempt_index: int,
        retry_index: int,
        query_pattern: Dict[str, Any],
        branch_spec: Optional[Dict[str, str]] = None,
        retry_instruction: str = "",
        parent_attempt_id: str = "",
        ctx: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        ctx = ctx or self._get_worker_context()
        start_ts = time.time()
        responses: List[Dict[str, Any]] = []
        attempt_error = ""
        query_text = self._build_retry_query(str(row.get("question", "") or ""), retry_instruction)
        branch_payload = dict(branch_spec or {})
        branch_routing_hint = self._build_sampling_branch_routing_hint(branch_payload) if branch_payload else ""
        question_id = str(row.get("question_id", "") or "")
        logger.info(
            "Strategy training attempt start: question_id=%s attempt=%d retry=%d branch=%s",
            question_id,
            attempt_index,
            retry_index,
            str(branch_payload.get("name", "") or "").strip() or "default",
        )
        forced_memory_ctx = {
            "query_abstract": str(query_pattern.get("query_abstract", "") or "").strip(),
            "query_pattern": copy.deepcopy(query_pattern),
            "candidate_patterns": [],
            "patterns": [],
            "routing_hint": branch_routing_hint,
            "sampling_branch_spec": copy.deepcopy(branch_payload),
        }
        try:
            responses = ctx["agent"].ask(
                query_text,
                lang=str(self.config.global_config.language or "zh"),
                session_id=f"strategy_training_{row.get('question_id', '')}_{attempt_index}_{retry_index}",
                max_llm_calls_per_run=self.max_llm_calls_per_attempt,
                _forced_memory_ctx=forced_memory_ctx,
            )
        except Exception as exc:
            attempt_error = str(exc or "").strip()
        latency_ms = int((time.time() - start_ts) * 1000)
        if attempt_error:
            logger.warning(
                "Strategy training attempt failed: question_id=%s attempt=%d retry=%d latency_ms=%d err=%s",
                question_id,
                attempt_index,
                retry_index,
                latency_ms,
                attempt_error[:400],
            )
            return self._build_error_attempt_record(
                row=row,
                attempt_index=attempt_index,
                retry_index=retry_index,
                query_pattern=query_pattern,
                branch_spec=branch_payload,
                routing_hint=branch_routing_hint,
                latency_ms=latency_ms,
                error=attempt_error,
                retry_instruction=retry_instruction,
                parent_attempt_id=parent_attempt_id,
            )
        try:
            record = self._build_attempt_record(
                row=row,
                attempt_index=attempt_index,
                retry_index=retry_index,
                query_pattern=query_pattern,
                branch_spec=branch_payload,
                routing_hint=branch_routing_hint,
                responses=responses,
                latency_ms=latency_ms,
                retry_instruction=retry_instruction,
                parent_attempt_id=parent_attempt_id,
            )
            logger.info(
                "Strategy training attempt done: question_id=%s attempt=%d retry=%d latency_ms=%d correct=%s tool_calls=%d",
                question_id,
                attempt_index,
                retry_index,
                latency_ms,
                bool((record.get("judge") or {}).get("is_correct", False)),
                len(record.get("tool_uses") or []),
            )
            return record
        except Exception as exc:
            logger.exception(
                "Strategy training attempt postprocess failed: question_id=%s attempt=%d retry=%d",
                question_id,
                attempt_index,
                retry_index,
            )
            return self._build_error_attempt_record(
                row=row,
                attempt_index=attempt_index,
                retry_index=retry_index,
                query_pattern=query_pattern,
                branch_spec=branch_payload,
                routing_hint=branch_routing_hint,
                latency_ms=latency_ms,
                error=f"attempt_postprocess_error: {exc}",
                retry_instruction=retry_instruction,
                parent_attempt_id=parent_attempt_id,
            )

    def _build_attempt_record(
        self,
        *,
        row: Dict[str, Any],
        attempt_index: int,
        retry_index: int,
        query_pattern: Dict[str, Any],
        branch_spec: Optional[Dict[str, Any]],
        routing_hint: str,
        responses: List[Dict[str, Any]],
        latency_ms: int,
        retry_instruction: str = "",
        parent_attempt_id: str = "",
        ctx: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        ctx = ctx or self._get_worker_context()
        final_answer = ctx["agent"].extract_final_text(responses)
        tool_uses = ctx["agent"].extract_tool_uses(responses)
        judge = ctx["answer_judge"].evaluate(
            question=str(row.get("question", "") or ""),
            reference_answer=str(row.get("answer", "") or ""),
            candidate_answer=final_answer,
        )
        effective_chain = (
            ctx["chain_extractor"].extract(
                question=str(row.get("question", "") or ""),
                reference_answer=str(row.get("answer", "") or ""),
                candidate_answer=final_answer,
                tool_uses=tool_uses,
            )
            if judge.get("is_correct")
            else {
                "raw_tool_chain": [],
                "minimal_effective_chain": [],
                "effective_tool_chain": [],
                "effective_step_indices": [],
                "discarded_step_indices": list(range(len(tool_uses))),
                "step_attributions": [],
                "reason": "answer_not_correct",
            }
        )
        raw_chain = [
            str((item or {}).get("tool_name", "") or "").strip()
            for item in tool_uses
            if str((item or {}).get("tool_name", "") or "").strip()
        ]
        tool_output_chars = sum(len(str((item or {}).get("tool_output", "") or "")) for item in tool_uses)
        question_id = str(row.get("question_id", "") or "")
        return {
            "attempt_id": self._make_attempt_id(question_id, attempt_index, retry_index),
            "attempt_group_id": f"{question_id or 'q'}_run_{attempt_index}",
            "parent_attempt_id": str(parent_attempt_id or ""),
            "question_id": question_id,
            "question": str(row.get("question", "") or ""),
            "reference_answer": str(row.get("answer", "") or ""),
            "attempt_index": attempt_index,
            "retry_index": retry_index,
            "retry_instruction": str(retry_instruction or ""),
            "query_pattern": query_pattern,
            "sampling_branch_spec": copy.deepcopy(branch_spec or {}),
            "routing_hint": str(routing_hint or ""),
            "final_answer": final_answer,
            "latency_ms": latency_ms,
            "tool_uses": tool_uses,
            "raw_tool_chain": raw_chain,
            "tool_output_chars": tool_output_chars,
            "judge": judge,
            "minimal_effective_chain": effective_chain.get("minimal_effective_chain") or effective_chain.get("effective_tool_chain") or [],
            "effective_tool_chain": effective_chain.get("effective_tool_chain") or [],
            "effective_step_indices": effective_chain.get("effective_step_indices") or [],
            "discarded_step_indices": effective_chain.get("discarded_step_indices") or [],
            "step_attributions": effective_chain.get("step_attributions") or [],
            "effective_chain_reason": effective_chain.get("reason", ""),
        }

    def _build_error_attempt_record(
        self,
        *,
        row: Dict[str, Any],
        attempt_index: int,
        retry_index: int,
        query_pattern: Dict[str, Any],
        branch_spec: Optional[Dict[str, Any]],
        routing_hint: str,
        latency_ms: int,
        error: str,
        retry_instruction: str = "",
        parent_attempt_id: str = "",
    ) -> Dict[str, Any]:
        error_text = str(error or "").strip()
        judge = {
            "is_correct": False,
            "score": 0.0,
            "reason": f"attempt_error: {error_text[:1200]}",
            "matched_points": [],
            "missing_points": [],
            "hallucination_points": [],
        }
        question_id = str(row.get("question_id", "") or "")
        return {
            "attempt_id": self._make_attempt_id(question_id, attempt_index, retry_index),
            "attempt_group_id": f"{question_id or 'q'}_run_{attempt_index}",
            "parent_attempt_id": str(parent_attempt_id or ""),
            "question_id": question_id,
            "question": str(row.get("question", "") or ""),
            "reference_answer": str(row.get("answer", "") or ""),
            "attempt_index": attempt_index,
            "retry_index": retry_index,
            "retry_instruction": str(retry_instruction or ""),
            "query_pattern": query_pattern,
            "sampling_branch_spec": copy.deepcopy(branch_spec or {}),
            "routing_hint": str(routing_hint or ""),
            "final_answer": "",
            "latency_ms": latency_ms,
            "tool_uses": [],
            "raw_tool_chain": [],
            "tool_output_chars": 0,
            "judge": judge,
            "minimal_effective_chain": [],
            "effective_tool_chain": [],
            "effective_step_indices": [],
            "discarded_step_indices": [],
            "step_attributions": [],
            "effective_chain_reason": "attempt_execution_error",
            "error": error_text,
        }

    def _build_failure_summary_input(self, attempts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for item in attempts:
            judge = item.get("judge") if isinstance(item.get("judge"), dict) else {}
            rows.append(
                {
                    "attempt_id": item.get("attempt_id"),
                    "final_answer": self._clip_text(item.get("final_answer", ""), limit=1200),
                    "raw_tool_chain": list(item.get("raw_tool_chain", []) or [])[:12],
                    "judge": {
                        "is_correct": bool(judge.get("is_correct", False)),
                        "score": float(judge.get("score", 0.0) or 0.0),
                        "reason": self._clip_text(judge.get("reason", ""), limit=600),
                        "matched_points": [self._clip_text(x, limit=120) for x in (judge.get("matched_points") or [])[:6]],
                        "missing_points": [self._clip_text(x, limit=120) for x in (judge.get("missing_points") or [])[:6]],
                        "hallucination_points": [self._clip_text(x, limit=120) for x in (judge.get("hallucination_points") or [])[:6]],
                    },
                    "error": self._clip_text(item.get("error", ""), limit=800),
                }
            )
        return rows

    def _build_failed_attempt_summaries_for_template(self, attempts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        failed = [item for item in attempts if not bool((item.get("judge") or {}).get("is_correct", False))]
        failed.sort(
            key=lambda item: (
                float(((item.get("judge") or {}).get("score", 0.0) or 0.0)),
                int(item.get("retry_index", 0) or 0),
            )
        )
        summaries: List[Dict[str, Any]] = []
        for item in failed[:3]:
            judge = item.get("judge") if isinstance(item.get("judge"), dict) else {}
            summaries.append(
                {
                    "attempt_id": str(item.get("attempt_id", "") or ""),
                    "retry_index": int(item.get("retry_index", 0) or 0),
                    "retry_instruction": self._clip_text(item.get("retry_instruction", ""), limit=220),
                    "raw_tool_chain": list(item.get("raw_tool_chain", []) or [])[:8],
                    "effective_tool_chain": list(item.get("effective_tool_chain", []) or [])[:8],
                    "judge": {
                        "score": float(judge.get("score", 0.0) or 0.0),
                        "reason": self._clip_text(judge.get("reason", ""), limit=320),
                        "missing_points": [self._clip_text(x, limit=120) for x in (judge.get("missing_points") or [])[:4]],
                        "hallucination_points": [self._clip_text(x, limit=120) for x in (judge.get("hallucination_points") or [])[:4]],
                    },
                    "error": self._clip_text(item.get("error", ""), limit=240),
                }
            )
        return summaries

    def _build_tool_inventory(self) -> Dict[str, Dict[str, Any]]:
        inventory: Dict[str, Dict[str, Any]] = {}
        for tool in [*(getattr(self.agent, "_base_tools", []) or []), *(getattr(self.agent, "_aggregation_tools", []) or []), *(getattr(self.agent, "_extra_tools", []) or [])]:
            tool_name = str(getattr(tool, "name", "") or "").strip()
            if not tool_name:
                continue
            inventory[tool_name] = {
                "name": tool_name,
                "description": str(getattr(tool, "description", "") or "").strip(),
                "parameters": copy.deepcopy(getattr(tool, "parameters", []) or []),
            }
        return inventory

    @staticmethod
    def _comparison_chain(attempt: Dict[str, Any]) -> List[str]:
        chain = attempt.get("effective_tool_chain") or attempt.get("raw_tool_chain") or []
        return [str(x).strip() for x in chain if str(x).strip()]

    def _get_tool_metadata_for_reflection(self, tool_name: str) -> Dict[str, Any]:
        inventory_meta = self.tool_inventory.get(str(tool_name or "").strip(), {})
        return self.tool_metadata_provider.resolve_tool_metadata(
            str(tool_name or "").strip(),
            fallback_description=str(inventory_meta.get("description", "") or ""),
            fallback_parameters=inventory_meta.get("parameters") or [],
        )

    def _summarize_attempt_for_reflection(self, attempt: Dict[str, Any]) -> Dict[str, Any]:
        judge = attempt.get("judge") if isinstance(attempt.get("judge"), dict) else {}
        return {
            "attempt_id": str(attempt.get("attempt_id", "") or ""),
            "raw_tool_chain": list(attempt.get("raw_tool_chain", []) or [])[:12],
            "effective_tool_chain": list(attempt.get("effective_tool_chain", []) or [])[:12],
            "final_answer": self._clip_text(attempt.get("final_answer", ""), limit=800),
            "judge": {
                "is_correct": bool(judge.get("is_correct", False)),
                "score": float(judge.get("score", 0.0) or 0.0),
                "reason": self._clip_text(judge.get("reason", ""), limit=500),
            },
            "error": self._clip_text(attempt.get("error", ""), limit=500),
        }

    def _select_reflection_tool_names(
        self,
        *,
        best_attempt: Dict[str, Any],
        failed_attempts: List[Dict[str, Any]],
    ) -> List[str]:
        best_chain = self._comparison_chain(best_attempt)
        if not best_chain or not failed_attempts:
            return []
        failed_chains = [set(self._comparison_chain(item)) for item in failed_attempts]
        selected: List[str] = []
        for tool_name in best_chain:
            if any(tool_name not in chain for chain in failed_chains) and tool_name not in selected:
                selected.append(tool_name)
        return selected

    def _run_tool_description_reflection(
        self,
        *,
        row: Dict[str, Any],
        query_pattern: Dict[str, Any],
        best_attempt: Dict[str, Any],
        failed_attempts: List[Dict[str, Any]],
        ctx: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        ctx = ctx or self._get_worker_context()
        tool_names = self._select_reflection_tool_names(best_attempt=best_attempt, failed_attempts=failed_attempts)
        if not tool_names:
            return []

        successful_attempt_summary = self._summarize_attempt_for_reflection(best_attempt)
        records: List[Dict[str, Any]] = []
        for tool_name in tool_names:
            relevant_failed = [item for item in failed_attempts if tool_name not in set(self._comparison_chain(item))]
            if not relevant_failed:
                continue
            failed_summaries = [self._summarize_attempt_for_reflection(item) for item in relevant_failed[:3]]
            tool_meta = self._get_tool_metadata_for_reflection(tool_name)
            reflection = ctx["tool_description_reflector"].reflect(
                tool_name=tool_name,
                current_description=str(tool_meta.get("description", "") or ""),
                parameters=tool_meta.get("parameters") or [],
                question=str(row.get("question", "") or ""),
                query_pattern=query_pattern,
                successful_attempt=successful_attempt_summary,
                failed_attempts=failed_summaries,
            )
            record = {
                "question_id": str(row.get("question_id", "") or ""),
                "question": str(row.get("question", "") or ""),
                "language": str(self.config.global_config.language or "zh"),
                "tool_name": tool_name,
                "current_description": str(tool_meta.get("description", "") or ""),
                "decision": str(reflection.get("decision", "keep") or "keep"),
                "proposed_description": str(reflection.get("proposed_description", tool_meta.get("description", "")) or ""),
                "reason": self._clip_text(reflection.get("reason", ""), limit=1000),
                "best_attempt_id": str(best_attempt.get("attempt_id", "") or ""),
                "failed_attempt_ids": [str(item.get("attempt_id", "") or "") for item in relevant_failed[:3]],
                "missing_from_failed_attempt_count": len(relevant_failed),
            }
            records.append(record)
        return records

    @staticmethod
    def _aggregate_tool_description_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        grouped: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
        for record in records:
            key = (
                str(record.get("tool_name", "") or ""),
                str(record.get("language", "") or ""),
            )
            if key[0]:
                grouped[key].append(record)

        aggregated: List[Dict[str, Any]] = []
        for (tool_name, language), items in sorted(grouped.items()):
            current_description = str(items[0].get("current_description", "") or "")
            keep_items = [item for item in items if str(item.get("decision", "") or "").strip().lower() == "keep"]
            revise_items = [
                item
                for item in items
                if str(item.get("decision", "") or "").strip().lower() == "revise"
                and str(item.get("proposed_description", "") or "").strip()
            ]
            if revise_items and len(revise_items) > len(keep_items):
                proposal_counter = Counter(str(item.get("proposed_description", "") or "").strip() for item in revise_items)
                proposed_description, support_count = proposal_counter.most_common(1)[0]
                supporting_items = [item for item in revise_items if str(item.get("proposed_description", "") or "").strip() == proposed_description]
                reason = Counter(str(item.get("reason", "") or "").strip() for item in supporting_items).most_common(1)[0][0]
                decision = "revise"
            else:
                proposed_description = current_description
                support_count = len(keep_items) if keep_items else len(items)
                base_items = keep_items if keep_items else items
                reason = Counter(str(item.get("reason", "") or "").strip() for item in base_items).most_common(1)[0][0]
                decision = "keep"
            aggregated.append(
                {
                    "tool_name": tool_name,
                    "language": language,
                    "current_description": current_description,
                    "decision": decision,
                    "proposed_description": proposed_description,
                    "reason": reason,
                    "support_count": int(support_count),
                    "keep_count": len(keep_items),
                    "revise_count": len(revise_items),
                }
            )
        return aggregated

    def _embed_template(self, template: Dict[str, Any], query_pattern: Dict[str, Any]) -> Dict[str, Any]:
        pattern_text = StrategyQueryPatternExtractor.pattern_to_text(query_pattern)
        embedding = None
        if self.embedding_model is not None and hasattr(self.embedding_model, "embed_query"):
            try:
                embedding = [float(x) for x in self.embedding_model.embed_query(pattern_text)]
            except Exception:
                embedding = None
        out = dict(template)
        out["query_pattern"] = query_pattern
        out["query_abstract"] = str(query_pattern.get("query_abstract", "") or "")
        out["query_pattern_text"] = pattern_text
        if embedding is not None:
            out["pattern_embedding"] = embedding
        return out

    def _build_markdown_report(
        self,
        *,
        manifest: Dict[str, Any],
        question_summaries: List[Dict[str, Any]],
        failed_summaries: List[Dict[str, Any]],
        tool_description_candidates: List[Dict[str, Any]],
        raw_template_count: int,
        cluster_payload: Dict[str, Any],
        consolidation_merge_count: int,
        runtime_library_payload: Dict[str, Any],
    ) -> str:
        success_count = len(question_summaries)
        failure_count = len(failed_summaries)
        clusters = cluster_payload.get("clusters") or []
        lines = [
            "# Strategy Memory Training Report",
            "",
            f"- Dataset: `{manifest['dataset_name']}`",
            f"- CSV: `{manifest['csv_path']}`",
            f"- Questions processed: `{manifest['question_count']}`",
            f"- Attempts per question: `{manifest['attempts_per_question']}`",
            f"- Question-level workers: `{manifest.get('training_max_workers', 1)}`",
            f"- Timeout retry rounds: `{manifest.get('question_timeout_retry_rounds', 0)}`",
            f"- Timeout retry workers: `{manifest.get('question_timeout_retry_workers', 1)}`",
            f"- Timeout retry worker plan: `{manifest.get('timeout_retry_worker_plan', [])}`",
            f"- Successful questions: `{success_count}`",
            f"- Failed questions: `{failure_count}`",
            f"- Tool description candidates: `{len(tool_description_candidates)}`",
            f"- Raw template count: `{raw_template_count}`",
            f"- Cluster count: `{cluster_payload.get('cluster_count', 0)}`",
            f"- Consolidation merges: `{consolidation_merge_count}`",
            f"- Runtime library: `{manifest['runtime_library_path']}`",
            "",
            "## Successful Questions",
            "",
        ]
        if not question_summaries:
            lines.append("- `(none)`")
        for item in question_summaries:
            template = item.get("template", {})
            lines.extend(
                [
                    f"### {item.get('question_id', '')}: {item.get('question', '')}",
                    "",
                    f"- Successful attempts: `{item.get('successful_attempt_count', 0)}`",
                    f"- Assigned cluster: `{item.get('cluster_id', '')}`",
                    f"- Recommended chain: `{ ' -> '.join(template.get('recommended_chain', [])) if template.get('recommended_chain') else '(none)' }`",
                    f"- Pattern: `{template.get('pattern_name', '')}`",
                    "",
                    "```text",
                    template.get("pattern_description", ""),
                    "```",
                    "",
                ]
            )
        lines.extend(["## Clusters", ""])
        if not clusters:
            lines.append("- `(none)`")
        for cluster in clusters[:12]:
            lines.extend(
                [
                    f"### {cluster.get('cluster_id', '')}: {cluster.get('pattern_name', '')}",
                    "",
                    f"- Template count: `{cluster.get('template_count', 0)}`",
                    f"- Support count: `{cluster.get('support_count', 0)}`",
                    f"- Success rate: `{cluster.get('success_rate', 0.0)}`",
                    f"- Recommended chain: `{ ' -> '.join(cluster.get('recommended_chain', [])) if cluster.get('recommended_chain') else '(none)' }`",
                    "",
                    "```text",
                    cluster.get("pattern_description", ""),
                    "```",
                    "",
                ]
            )
        lines.extend(["## Failed Questions", ""])
        if not failed_summaries:
            lines.append("- `(none)`")
        for item in failed_summaries:
            lines.extend(
                [
                    f"### {item.get('question_id', '')}: {item.get('question', '')}",
                    "",
                    "```text",
                    str(item.get("failure_summary", "")),
                    "```",
                    "",
                ]
            )
        lines.extend([
            "## Runtime Library Summary",
            "",
            f"- Template count: `{len(runtime_library_payload.get('templates', []))}`",
        ])
        return "\n".join(lines)

    def _make_failed_question_result(
        self,
        *,
        row: Dict[str, Any],
        failure_summary: str,
        likely_causes: Optional[List[str]] = None,
        recommended_improvements: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        return {
            "question_id": str(row.get("question_id", "") or ""),
            "question": str(row.get("question", "") or ""),
            "successful": False,
            "attempts": [],
            "reflection_records": [],
            "tool_description_records": [],
            "query_pattern": {},
            "failure": {
                "question_id": str(row.get("question_id", "") or ""),
                "question": str(row.get("question", "") or ""),
                "failure_summary": self._clip_text(failure_summary, limit=1200),
                "likely_causes": list(likely_causes or []),
                "recommended_improvements": list(recommended_improvements or []),
            },
        }

    def _process_question(
        self,
        row: Dict[str, Any],
        *,
        max_retry_per_attempt_override: Optional[int] = None,
    ) -> Dict[str, Any]:
        ctx = self._get_worker_context()
        started_at = time.time()
        question_id = str(row.get("question_id", "") or "")
        logger.info("Strategy training question start: question_id=%s", question_id)
        self._mark_question_started(row=row)
        query_pattern = ctx["query_pattern_extractor"].extract(str(row.get("question", "") or ""))
        attempts: List[Dict[str, Any]] = []
        reflection_records: List[Dict[str, Any]] = []
        tool_description_records: List[Dict[str, Any]] = []
        retry_limit = self.max_retry_per_attempt if max_retry_per_attempt_override is None else max(
            0, int(max_retry_per_attempt_override or 0)
        )
        attempt_branch_specs = self._build_attempt_branch_specs(row=row, ctx=ctx)

        for attempt_index in range(self.attempts_per_question):
            if self._question_deadline_exceeded(started_at):
                raise TimeoutError(f"question_timeout: {row.get('question_id', '')} exceeded {self.question_timeout_sec}s")
            retry_instruction = ""
            parent_attempt_id = ""
            branch_spec = attempt_branch_specs[min(attempt_index, len(attempt_branch_specs) - 1)]
            for retry_index in range(retry_limit + 1):
                if self._question_deadline_exceeded(started_at):
                    raise TimeoutError(f"question_timeout: {row.get('question_id', '')} exceeded {self.question_timeout_sec}s")
                attempt_record = self._run_single_attempt(
                    row=row,
                    attempt_index=attempt_index,
                    retry_index=retry_index,
                    query_pattern=query_pattern,
                    branch_spec=branch_spec,
                    retry_instruction=retry_instruction,
                    parent_attempt_id=parent_attempt_id,
                    ctx=ctx,
                )
                attempts.append(attempt_record)
                self._append_jsonl(self.live_attempts_path, attempt_record)
                if int(attempt_record.get("retry_index", 0) or 0) > 0:
                    self._append_jsonl(self.live_retry_attempts_path, attempt_record)
                self._mark_attempt_completed(attempt_record=attempt_record)
                if bool(attempt_record.get("judge", {}).get("is_correct", False)):
                    break
                if retry_index >= retry_limit:
                    break
                retry_payload = self._reflect_and_build_retry(
                    row=row,
                    attempt_record=attempt_record,
                    ctx=ctx,
                )
                reflection_records.append(
                    {
                        "attempt_id": attempt_record["attempt_id"],
                        "attempt_group_id": attempt_record["attempt_group_id"],
                        "question_id": attempt_record["question_id"],
                        "retry_index": attempt_record.get("retry_index", 0),
                        "reflection": retry_payload.get("reflection", {}),
                        "retry_instruction": retry_payload.get("retry_instruction", ""),
                        "tool_summary": retry_payload.get("tool_summary", {}),
                    }
                )
                retry_instruction = str(retry_payload.get("retry_instruction", "") or "").strip()
                parent_attempt_id = str(attempt_record.get("attempt_id", "") or "")
                if not retry_instruction or not bool((retry_payload.get("reflection") or {}).get("need_retry", False)):
                    break

        successful = [item for item in attempts if bool(item.get("judge", {}).get("is_correct", False))]
        if successful:
            best_attempt = sorted(successful, key=self._complexity_key)[0]
            failed_attempts = [item for item in attempts if not bool(item.get("judge", {}).get("is_correct", False))]
            if failed_attempts:
                try:
                    tool_description_records = self._run_tool_description_reflection(
                        row=row,
                        query_pattern=query_pattern,
                        best_attempt=best_attempt,
                        failed_attempts=failed_attempts,
                        ctx=ctx,
                    )
                except Exception:
                    logger.exception("tool description reflection failed: question_id=%s", row.get("question_id", ""))
            template = ctx["template_distiller"].distill(
                question=str(row.get("question", "") or ""),
                query_pattern=query_pattern,
                best_attempt={
                    "minimal_effective_chain": best_attempt.get("minimal_effective_chain", []),
                    "effective_tool_chain": best_attempt.get("effective_tool_chain", []),
                    "raw_tool_chain": best_attempt.get("raw_tool_chain", []),
                    "step_attributions": best_attempt.get("step_attributions", []),
                    "judge": best_attempt.get("judge", {}),
                    "latency_ms": best_attempt.get("latency_ms", 0),
                    "retry_index": best_attempt.get("retry_index", 0),
                    "retry_instruction": best_attempt.get("retry_instruction", ""),
                },
                failed_attempts=self._build_failed_attempt_summaries_for_template(attempts),
                retry_instruction=str(best_attempt.get("retry_instruction", "") or ""),
            )
            template_seed = {
                "template_id": f"stm_{row.get('question_id', '')}",
                "question_id": str(row.get("question_id", "") or ""),
                "question": str(row.get("question", "") or ""),
                "pattern_name": template.get("pattern_name", ""),
                "pattern_description": template.get("pattern_description", ""),
                "raw_tool_chain": best_attempt.get("raw_tool_chain", []),
                "minimal_effective_chain": best_attempt.get("minimal_effective_chain", []) or template.get("recommended_chain", []),
                "recommended_chain": template.get("recommended_chain", []),
                "anti_patterns": template.get("anti_patterns", []),
                "chain_rationale": template.get("chain_rationale", ""),
                "chain_constraints": template.get("chain_constraints", []),
                "avg_latency_ms": best_attempt.get("latency_ms", 0),
                "source_attempt_ids": [str(best_attempt.get("attempt_id", "") or "")] if str(best_attempt.get("attempt_id", "") or "") else [],
                "support_count": len(successful),
                "successful_attempts": len(successful),
                "attempt_count": self.attempts_per_question,
                "success_rate": round(float(len(successful)) / float(self.attempts_per_question), 4),
                "template_sources": [
                    {
                        "question_id": str(row.get("question_id", "") or ""),
                        "question": str(row.get("question", "") or ""),
                        "source_type": "direct_distillation",
                    }
                ],
            }
            result = {
                "question_id": str(row.get("question_id", "") or ""),
                "question": str(row.get("question", "") or ""),
                "query_pattern": query_pattern,
                "attempts": attempts,
                "reflection_records": reflection_records,
                "tool_description_records": tool_description_records,
                "successful": True,
                "template_seed": template_seed,
                "best_attempt_id": best_attempt.get("attempt_id", ""),
                "best_retry_index": best_attempt.get("retry_index", 0),
                "best_retry_instruction": best_attempt.get("retry_instruction", ""),
                "best_effective_tool_chain": best_attempt.get("minimal_effective_chain", []) or best_attempt.get("effective_tool_chain", []),
                "best_raw_tool_chain": best_attempt.get("raw_tool_chain", []),
                "best_judge": best_attempt.get("judge", {}),
            }
            logger.info(
                "Strategy training question done: question_id=%s successful=%s attempts=%d elapsed_sec=%.2f",
                question_id,
                True,
                len(attempts),
                time.time() - started_at,
            )
            self._mark_question_finished(result=result)
            return result

        try:
            failure = ctx["failure_summarizer"].summarize(
                question=str(row.get("question", "") or ""),
                reference_answer=str(row.get("answer", "") or ""),
                attempt_summaries=self._build_failure_summary_input(attempts),
            )
        except Exception as exc:
            failure = {
                "failure_summary": self._clip_text(f"failure_summarizer_error: {exc}", limit=1200),
                "likely_causes": ["failure_summarizer_error"],
                "recommended_improvements": [],
            }
        result = {
            "question_id": str(row.get("question_id", "") or ""),
            "question": str(row.get("question", "") or ""),
            "query_pattern": query_pattern,
            "attempts": attempts,
            "reflection_records": reflection_records,
            "tool_description_records": tool_description_records,
            "successful": False,
            "failure": {
                "question_id": str(row.get("question_id", "") or ""),
                "question": str(row.get("question", "") or ""),
                **failure,
            },
        }
        logger.info(
            "Strategy training question done: question_id=%s successful=%s attempts=%d elapsed_sec=%.2f",
            question_id,
            False,
            len(attempts),
            time.time() - started_at,
        )
        self._mark_question_finished(result=result)
        return result

    def _build_timeout_retry_worker_plan(self, question_count: int) -> List[int]:
        total = max(1, int(question_count or 1))
        plan: List[int] = [max(1, min(self.training_max_workers, total))]
        if self.question_timeout_retry_rounds <= 0:
            return plan
        for round_index in range(self.question_timeout_retry_rounds):
            if round_index == 0:
                workers = max(1, min(self.question_timeout_retry_workers, total))
            else:
                workers = 1
            if workers != plan[-1]:
                plan.append(workers)
        return plan

    def _run_question_round(
        self,
        *,
        rows: List[Dict[str, Any]],
        max_workers: int,
        max_retry_per_attempt_override: Optional[int],
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        if not rows:
            return [], []

        future_to_row: Dict[Any, Dict[str, Any]] = {}
        pending: Set[Any] = set()
        timed_out_futures: Set[Any] = set()
        started_at_by_future: Dict[Any, float] = {}
        results: List[Dict[str, Any]] = []

        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="strategy-q") as executor:
            for row in rows:
                future = executor.submit(
                    self._process_question,
                    row,
                    max_retry_per_attempt_override=max_retry_per_attempt_override,
                )
                future_to_row[future] = row
                pending.add(future)
                started_at_by_future[future] = time.time()

            while pending:
                done, not_done = wait(pending, timeout=1.0, return_when=FIRST_COMPLETED)
                for future in list(done):
                    pending.discard(future)
                    row = future_to_row[future]
                    try:
                        result = future.result()
                    except Exception as exc:
                        logger.exception(
                            "strategy training question failed: question_id=%s",
                            row.get("question_id", ""),
                        )
                        result = self._make_failed_question_result(
                            row=row,
                            failure_summary=f"question_processing_error: {exc}",
                            likely_causes=["question_processing_error"],
                        )
                        self._mark_question_finished(result=result)
                    results.append(result)

                timeout_sec = int(self.question_timeout_sec or 0)
                if timeout_sec <= 0:
                    continue
                now_ts = time.time()
                for future in list(not_done):
                    if future in timed_out_futures:
                        continue
                    if (now_ts - started_at_by_future.get(future, now_ts)) < float(timeout_sec):
                        continue
                    timed_out_futures.add(future)
                    pending.discard(future)
                    future.cancel()

        timeout_rows: List[Dict[str, Any]] = []
        for future in list(timed_out_futures):
            row = future_to_row[future]
            if future.cancelled():
                timeout_rows.append(row)
                continue
            try:
                result = future.result()
            except TimeoutError:
                timeout_rows.append(row)
                continue
            except Exception as exc:
                logger.exception(
                    "strategy training late question failed after timeout: question_id=%s",
                    row.get("question_id", ""),
                )
                result = self._make_failed_question_result(
                    row=row,
                    failure_summary=f"question_processing_error: {exc}",
                    likely_causes=["question_processing_error"],
                )
                self._mark_question_finished(result=result)
            else:
                logger.warning(
                    "Late question result salvaged after timeout window: question_id=%s",
                    row.get("question_id", ""),
                )
            results.append(result)

        return results, timeout_rows

    def _persist_question_result(self, result: Dict[str, Any]) -> None:
        for attempt_record in result.get("attempts") or []:
            self._append_jsonl(self.attempts_path, attempt_record)
            if int(attempt_record.get("retry_index", 0) or 0) > 0:
                self._append_jsonl(self.retry_attempts_path, attempt_record)
            self._append_jsonl(
                self.judgments_path,
                {
                    "attempt_id": attempt_record["attempt_id"],
                    "question_id": attempt_record["question_id"],
                    "judge": attempt_record["judge"],
                    "retry_index": attempt_record.get("retry_index", 0),
                },
            )
            self._append_jsonl(
                self.effective_chain_path,
                {
                    "attempt_id": attempt_record["attempt_id"],
                    "question_id": attempt_record["question_id"],
                    "retry_index": attempt_record.get("retry_index", 0),
                    "raw_tool_chain": attempt_record["raw_tool_chain"],
                    "minimal_effective_chain": attempt_record.get("minimal_effective_chain", []),
                    "effective_tool_chain": attempt_record["effective_tool_chain"],
                    "effective_step_indices": attempt_record["effective_step_indices"],
                    "discarded_step_indices": attempt_record["discarded_step_indices"],
                    "step_attributions": attempt_record.get("step_attributions", []),
                    "reason": attempt_record["effective_chain_reason"],
                },
            )
        for record in result.get("reflection_records") or []:
            self._append_jsonl(self.reflection_path, record)
        for record in result.get("tool_description_records") or []:
            self._append_jsonl(self.tool_reflection_record_path, record)

        out_path = self.question_detail_dir / f"{str(result.get('question_id', '') or 'question')}.json"
        json_dump_atomic(str(out_path), result)

    def clear_runtime_library(self) -> Dict[str, Any]:
        return self.runtime_asset_manager.clear_all(
            aggregation_mode="narrative",
            dataset_name="",
        )

    def run(self, *, reset_runtime_library: bool = False) -> Dict[str, Any]:
        if reset_runtime_library:
            self.clear_runtime_library()
        rows = self._load_rows(self.csv_path)
        if self.question_limit is not None:
            rows = rows[: self.question_limit]
        self._initialize_progress(question_count=len(rows))

        manifest = {
            "dataset_name": self.dataset_name,
            "csv_path": str(self.csv_path),
            "question_count": len(rows),
            "attempts_per_question": self.attempts_per_question,
            "max_retry_per_attempt": self.max_retry_per_attempt,
            "training_max_workers": self.training_max_workers,
            "question_timeout_retry_rounds": self.question_timeout_retry_rounds,
            "question_timeout_retry_workers": self.question_timeout_retry_workers,
            "timeout_retry_max_retry_per_attempt": self.timeout_retry_max_retry_per_attempt,
            "timeout_retry_worker_plan": self._build_timeout_retry_worker_plan(len(rows)),
            "aggregation_mode": "narrative",
            "answer_temperature": self.answer_temperature,
            "runtime_library_path": str(self.runtime_library_path),
            "runtime_tool_metadata_dir": str(self.runtime_tool_metadata_dir),
            "config_snapshot": asdict(self.config),
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        json_dump_atomic(str(self.manifest_path), manifest)

        question_summaries: List[Dict[str, Any]] = []
        failed_summaries: List[Dict[str, Any]] = []
        tool_description_records: List[Dict[str, Any]] = []
        question_results: List[Dict[str, Any]] = []
        timeout_retry_plan = self._build_timeout_retry_worker_plan(len(rows))
        pending_rows: List[Dict[str, Any]] = list(rows)
        with tqdm(total=len(rows), desc="Training strategy memory", dynamic_ncols=True) as pbar:
            for round_index, workers in enumerate(timeout_retry_plan):
                if not pending_rows:
                    break
                retry_override = None if round_index == 0 else self.timeout_retry_max_retry_per_attempt
                pbar.set_description(f"Training strategy memory [round {round_index + 1}/{len(timeout_retry_plan)}]")
                pbar.set_postfix(
                    workers=workers,
                    queue=len(pending_rows),
                    retries=("full" if retry_override is None else retry_override),
                    refresh=True,
                )
                logger.info(
                    "Strategy training round start: round=%d workers=%d questions=%d retry_override=%s",
                    round_index,
                    workers,
                    len(pending_rows),
                    retry_override,
                )
                round_results, timeout_rows = self._run_question_round(
                    rows=pending_rows,
                    max_workers=max(1, min(workers, len(pending_rows) or 1)),
                    max_retry_per_attempt_override=retry_override,
                )
                round_results.sort(key=lambda item: int(str(item.get("question_id", "q0")).lstrip("q") or 0))
                for result in round_results:
                    self._persist_question_result(result)
                    question_results.append(result)
                    pbar.update(1)
                pending_rows = list(timeout_rows)
                pbar.set_postfix(
                    workers=workers,
                    queue=len(pending_rows),
                    retries=("full" if retry_override is None else retry_override),
                    refresh=True,
                )
                if pending_rows:
                    logger.warning(
                        "Strategy training round timed out questions: round=%d remaining=%d next_workers=%s",
                        round_index,
                        len(pending_rows),
                        timeout_retry_plan[round_index + 1] if (round_index + 1) < len(timeout_retry_plan) else "none",
                    )

            for row in pending_rows:
                result = self._make_failed_question_result(
                    row=row,
                    failure_summary=f"question_timeout: exceeded {self.question_timeout_sec}s across all retry rounds",
                    likely_causes=["question_timeout"],
                    recommended_improvements=["reduce concurrency or simplify retry workload for this question"],
                )
                self._mark_question_finished(result=result)
                self._persist_question_result(result)
                question_results.append(result)
                pbar.update(1)
            pbar.set_postfix(workers=0, queue=0, retries=0, refresh=True)

        question_results.sort(key=lambda item: int(str(item.get("question_id", "q0")).lstrip("q") or 0))
        pending_raw_templates: List[Dict[str, Any]] = []
        pending_summary_links: List[Dict[str, Any]] = []
        for result in question_results:
            for record in result.get("tool_description_records") or []:
                tool_description_records.append(record)
            if bool(result.get("successful")):
                query_pattern = result.get("query_pattern") if isinstance(result.get("query_pattern"), dict) else {}
                raw_template = self._embed_template(result.get("template_seed") or {}, query_pattern)
                pending_raw_templates.append(raw_template)
                pending_summary_links.append(
                    {
                        "question_id": str(result.get("question_id", "") or ""),
                        "question": str(result.get("question", "") or ""),
                        "successful_attempt_count": len(
                            [item for item in (result.get("attempts") or []) if bool((item.get("judge") or {}).get("is_correct", False))]
                        ),
                        "best_attempt_id": result.get("best_attempt_id", ""),
                        "best_retry_index": result.get("best_retry_index", 0),
                        "best_retry_instruction": result.get("best_retry_instruction", ""),
                        "best_effective_tool_chain": result.get("best_effective_tool_chain", []),
                        "best_raw_tool_chain": result.get("best_raw_tool_chain", []),
                        "best_judge": result.get("best_judge", {}),
                        "query_pattern": query_pattern,
                        "cluster_id": "",
                        "template": raw_template,
                    }
                )
            else:
                failure = result.get("failure") if isinstance(result.get("failure"), dict) else {}
                failed_summaries.append(
                    {
                        "question_id": str(failure.get("question_id", result.get("question_id", "")) or ""),
                        "question": str(failure.get("question", result.get("question", "")) or ""),
                        "failure_summary": str(failure.get("failure_summary", "") or ""),
                        "likely_causes": list(failure.get("likely_causes") or []),
                        "recommended_improvements": list(failure.get("recommended_improvements") or []),
                    }
                )

        cluster_ids = self.cluster_manager.add_templates(pending_raw_templates)
        for item, cluster_id in zip(pending_summary_links, cluster_ids):
            item["cluster_id"] = cluster_id
        question_summaries.extend(pending_summary_links)

        try:
            consolidation_merge_count = self.cluster_manager.consolidate()
        except Exception as exc:
            logger.exception("strategy cluster consolidation failed")
            consolidation_merge_count = 0
        tool_description_candidates = self._aggregate_tool_description_records(tool_description_records)
        raw_templates = list(self.cluster_manager.raw_templates)
        try:
            cluster_payload = self.cluster_manager.export_training_payload()
        except Exception as exc:
            logger.exception("strategy cluster export failed")
            cluster_payload = {"cluster_count": len(getattr(self.cluster_manager, "clusters", []) or []), "clusters": []}
        template_to_cluster: Dict[str, str] = {}
        for cluster in cluster_payload.get("clusters", []):
            cluster_id = str(cluster.get("cluster_id", "") or "")
            for template_id in cluster.get("member_template_ids", []) or []:
                template_to_cluster[str(template_id or "")] = cluster_id
        for item in question_summaries:
            template_id = str((item.get("template") or {}).get("template_id", "") or "")
            if template_id and template_id in template_to_cluster:
                item["cluster_id"] = template_to_cluster[template_id]
        template_source_index = {
            str(cluster.get("cluster_id", "") or ""): {
                "template_id": str(cluster.get("cluster_id", "") or ""),
                "pattern_name": str(cluster.get("pattern_name", "") or ""),
                "source_question_ids": list(cluster.get("source_question_ids") or []),
                "source_questions": list(cluster.get("source_questions") or []),
                "member_template_ids": list(cluster.get("member_template_ids") or []),
            }
            for cluster in cluster_payload.get("clusters", [])
            if str(cluster.get("cluster_id", "") or "")
        }

        runtime_patterns = self.cluster_manager.runtime_templates() if hasattr(self.cluster_manager, "runtime_templates") else []
        training_library_payload = {
            "library_version": 3,
            "aggregation_mode": "narrative",
            "dataset_name": self.dataset_name,
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "raw_template_count": len(raw_templates),
            "cluster_count": cluster_payload.get("cluster_count", 0),
            "clusters": cluster_payload.get("clusters", []),
            "pattern_count": len(runtime_patterns),
            "patterns": runtime_patterns,
            "templates": runtime_patterns,
        }
        runtime_library_payload = {
            "library_version": 3,
            "aggregation_mode": "narrative",
            "dataset_name": self.dataset_name,
            "generated_at": training_library_payload["generated_at"],
            "pattern_count": len(runtime_patterns),
            "patterns": runtime_patterns,
            "template_count": len(runtime_patterns),
            "templates": runtime_patterns,
        }

        json_dump_atomic(str(self.question_summary_path), question_summaries)
        json_dump_atomic(str(self.raw_template_path), raw_templates)
        json_dump_atomic(str(self.failure_summary_path), failed_summaries)
        json_dump_atomic(str(self.tool_description_candidate_path), tool_description_candidates)
        json_dump_atomic(str(self.cluster_path), cluster_payload)
        self._write_jsonl(self.merge_decision_path, self.cluster_manager.merge_decisions)
        json_dump_atomic(str(self.library_output_path), training_library_payload)
        json_dump_atomic(str(self.template_source_index_path), template_source_index)
        json_dump_atomic(str(self.runtime_library_path), runtime_library_payload)
        runtime_tool_metadata_paths = self.runtime_asset_manager.export_tool_metadata_overrides(tool_description_candidates)
        runtime_template_source_index_path = self.runtime_asset_manager.export_template_source_index(template_source_index)
        self.report_path.write_text(
            self._build_markdown_report(
                manifest=manifest,
                question_summaries=question_summaries,
                failed_summaries=failed_summaries,
                tool_description_candidates=tool_description_candidates,
                raw_template_count=len(raw_templates),
                cluster_payload=cluster_payload,
                consolidation_merge_count=consolidation_merge_count,
                runtime_library_payload=runtime_library_payload,
            ),
            encoding="utf-8",
        )
        return {
            "manifest_path": str(self.manifest_path),
            "question_summary_path": str(self.question_summary_path),
            "raw_template_path": str(self.raw_template_path),
            "failure_summary_path": str(self.failure_summary_path),
            "reflection_path": str(self.reflection_path),
            "retry_attempts_path": str(self.retry_attempts_path),
            "tool_reflection_record_path": str(self.tool_reflection_record_path),
            "tool_description_candidate_path": str(self.tool_description_candidate_path),
            "cluster_path": str(self.cluster_path),
            "merge_decision_path": str(self.merge_decision_path),
            "library_output_path": str(self.library_output_path),
            "template_source_index_path": str(self.template_source_index_path),
            "runtime_library_path": str(self.runtime_library_path),
            "runtime_tool_metadata_dir": str(self.runtime_tool_metadata_dir),
            "runtime_tool_metadata_paths": runtime_tool_metadata_paths,
            "runtime_template_source_index_path": runtime_template_source_index_path,
            "report_path": str(self.report_path),
            "successful_question_count": len(question_summaries),
            "failed_question_count": len(failed_summaries),
            "tool_description_reflection_count": len(tool_description_records),
            "tool_description_candidate_count": len(tool_description_candidates),
            "cluster_count": cluster_payload.get("cluster_count", 0),
            "consolidation_merge_count": consolidation_merge_count,
        }

    def close(self) -> None:
        try:
            self.agent.close()
        except Exception:
            pass
        for ctx in list(self._worker_contexts):
            try:
                ctx.get("agent").close()
            except Exception:
                pass

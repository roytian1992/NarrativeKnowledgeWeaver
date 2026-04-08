from __future__ import annotations

import copy
import csv
import json
import logging
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from core.agent.retriever_agent import QuestionAnsweringAgent
from core.functions.memory_management.effective_tool_chain_extractor import EffectiveToolChainExtractor
from core.functions.memory_management.strategy_query_pattern import StrategyQueryPatternExtractor
from core.functions.memory_management.strategy_template_distiller import StrategyTemplateDistiller
from core.functions.memory_management.tool_description_reflector import ToolDescriptionReflector
from core.functions.tool_calls.tool_metadata import ToolMetadataProvider
from core.memory.online_strategy_buffer import OnlineStrategyBuffer
from core.model_providers.openai_llm import OpenAILLM
from core.storage.vector_store import VectorStore
from core.strategy_training.strategy_cluster_manager import StrategyTemplateClusterManager
from core.strategy_training.strategy_runtime_assets import StrategyRuntimeAssetManager
from core.utils.config import KAGConfig
from core.utils.general_utils import json_dump_atomic
from core.utils.prompt_loader import YAMLPromptLoader


logger = logging.getLogger(__name__)


class OnlineStrategyTrainingRunner:
    def __init__(
        self,
        *,
        config: KAGConfig,
        csv_path: str,
        dataset_name: str = "we2_online",
        attempts_per_question: int = 5,
        question_limit: Optional[int] = None,
        output_root: str = "strategy_training",
        runtime_library_path: Optional[str] = None,
        enable_sql_tools: bool = True,
        max_llm_calls_per_attempt: int = 8,
        self_bootstrap_max_questions: int = 1,
    ) -> None:
        self.repo_root = Path(__file__).resolve().parents[2]
        self.config = copy.deepcopy(config)
        self.csv_path = Path(csv_path)
        self.dataset_name = str(dataset_name or self.csv_path.stem or "we2_online").strip()
        self.attempts_per_question = max(1, int(attempts_per_question or 1))
        self.question_limit = question_limit if question_limit is None else max(1, int(question_limit))
        self.output_root = self.repo_root / str(output_root or "strategy_training") / self.dataset_name
        self.runtime_library_path = Path(runtime_library_path) if runtime_library_path else self.repo_root / str(
            getattr(self.config.strategy_memory, "library_path", "data/memory/strategy/strategy_library.json")
        )
        self.runtime_tool_metadata_dir = self.repo_root / str(
            getattr(self.config.strategy_memory, "tool_metadata_runtime_dir", "data/memory/strategy/tool_metadata")
        )
        self.enable_sql_tools = bool(enable_sql_tools)
        self.max_llm_calls_per_attempt = max(1, int(max_llm_calls_per_attempt or 1))
        self.self_bootstrap_max_questions = max(0, int(self_bootstrap_max_questions or 0))
        self.training_max_workers = max(1, int(getattr(self.config.strategy_memory, "training_max_workers", 16) or 16))

        self.online_buffer_dir = self.output_root / "online_buffers"
        self.manifest_path = self.output_root / "manifests" / "online_training_manifest.json"
        self.attempts_path = self.output_root / "attempts" / "online_attempt_runs.jsonl"
        self.raw_template_path = self.output_root / "distilled" / "raw_templates.json"
        self.question_summary_path = self.output_root / "distilled" / "question_training_summaries.json"
        self.failure_summary_path = self.output_root / "failures" / "failed_question_summaries.json"
        self.tool_reflection_record_path = self.output_root / "tool_metadata" / "tool_description_reflection_records.jsonl"
        self.tool_description_candidate_path = self.output_root / "tool_metadata" / "tool_description_candidates.json"
        self.cluster_path = self.output_root / "clusters" / "template_clusters.json"
        self.merge_decision_path = self.output_root / "clusters" / "merge_decisions.jsonl"
        self.library_output_path = self.output_root / "library" / "strategy_library.json"
        self.template_source_index_path = self.output_root / "library" / "template_source_index.json"
        self.source_question_hints_path = self.output_root / "library" / "source_question_hints.json"
        self.progress_path = self.output_root / "progress.json"
        self.report_path = self.output_root / "report.md"

        for path in [
            self.manifest_path,
            self.attempts_path,
            self.raw_template_path,
            self.question_summary_path,
            self.failure_summary_path,
            self.tool_reflection_record_path,
            self.tool_description_candidate_path,
            self.cluster_path,
            self.merge_decision_path,
            self.library_output_path,
            self.template_source_index_path,
            self.source_question_hints_path,
            self.progress_path,
            self.report_path,
        ]:
            path.parent.mkdir(parents=True, exist_ok=True)

        self.prompt_loader = YAMLPromptLoader(self.config.global_config.prompt_dir)
        self.llm = OpenAILLM(self.config, llm_profile="memory")
        self.query_pattern_extractor = StrategyQueryPatternExtractor(
            prompt_loader=self.prompt_loader,
            llm=self.llm,
            prompt_id=getattr(self.config.strategy_memory, "pattern_extractor_prompt_id", "memory/extract_strategy_query_pattern"),
            abstraction_mode=getattr(self.config.strategy_memory, "abstraction_mode", "hybrid"),
        )
        self.chain_extractor = EffectiveToolChainExtractor(self.prompt_loader, self.llm)
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
        self.runtime_asset_manager = StrategyRuntimeAssetManager(
            library_path=str(self.runtime_library_path),
            tool_metadata_runtime_dir=str(self.runtime_tool_metadata_dir),
        )
        self.tool_inventory = self._build_tool_inventory()
        self._thread_local = threading.local()
        self._worker_context_lock = threading.Lock()
        self._worker_contexts: List[Dict[str, Any]] = []

    def _build_agent_config(self) -> KAGConfig:
        cfg = copy.deepcopy(self.config)
        cfg.global_config.aggregation_mode = "narrative"
        if hasattr(cfg, "global_"):
            cfg.global_.aggregation_mode = "narrative"
        cfg.strategy_memory.enabled = False
        cfg.strategy_memory.read_enabled = False
        cfg.strategy_memory.online_enabled = True
        cfg.strategy_memory.self_bootstrap_enabled = self.self_bootstrap_max_questions > 0
        cfg.strategy_memory.self_bootstrap_max_questions = self.self_bootstrap_max_questions
        cfg.strategy_memory.online_buffer_dir = str(self.online_buffer_dir)
        cfg.strategy_memory.online_real_trace_path = str(self.online_buffer_dir / "real_traces.jsonl")
        cfg.strategy_memory.online_strategy_buffer_path = str(self.online_buffer_dir / "online_strategy_buffer.jsonl")
        cfg.strategy_memory.online_failure_reflection_path = str(self.online_buffer_dir / "failure_reflections.jsonl")
        cfg.strategy_memory.online_synthetic_qa_path = str(self.online_buffer_dir / "synthetic_qa_buffer.jsonl")
        cfg.strategy_memory.online_strategy_min_score = 0.78
        cfg.strategy_memory.min_sampling_branches = max(5, int(getattr(cfg.strategy_memory, "min_sampling_branches", 5) or 5))
        cfg.strategy_memory.self_bootstrap_min_source_score = 0.78
        cfg.strategy_memory.self_bootstrap_min_accept_score = 0.88
        cfg.strategy_memory.self_bootstrap_sampling_attempts = max(
            3,
            int(getattr(cfg.strategy_memory, "self_bootstrap_sampling_attempts", 3) or 3),
        )
        return cfg

    def _create_worker_context(self) -> Dict[str, Any]:
        cfg = self._build_agent_config()
        agent = QuestionAnsweringAgent(
            cfg,
            aggregation_mode="narrative",
            enable_sql_tools=self.enable_sql_tools,
        )
        ctx = {"cfg": cfg, "agent": agent}
        with self._worker_context_lock:
            self._worker_contexts.append(ctx)
        return ctx

    def _get_worker_context(self) -> Dict[str, Any]:
        ctx = getattr(self._thread_local, "online_strategy_training_context", None)
        if ctx is None:
            ctx = self._create_worker_context()
            self._thread_local.online_strategy_training_context = ctx
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

    @staticmethod
    def _clip_text(value: Any, limit: int = 1200) -> str:
        text = str(value or "").strip()
        if limit <= 0:
            return ""
        if len(text) <= limit:
            return text
        return text[: max(0, limit - 3)] + "..."

    def _build_tool_inventory(self) -> Dict[str, Dict[str, Any]]:
        cfg = copy.deepcopy(self.config)
        cfg.global_config.aggregation_mode = "narrative"
        if hasattr(cfg, "global_"):
            cfg.global_.aggregation_mode = "narrative"
        cfg.strategy_memory.enabled = False
        cfg.strategy_memory.read_enabled = False
        cfg.strategy_memory.online_enabled = False
        cfg.strategy_memory.self_bootstrap_enabled = False
        agent = QuestionAnsweringAgent(
            cfg,
            aggregation_mode="narrative",
            enable_sql_tools=self.enable_sql_tools,
        )
        try:
            inventory: Dict[str, Dict[str, Any]] = {}
            for tool in [
                *(getattr(agent, "_base_tools", []) or []),
                *(getattr(agent, "_aggregation_tools", []) or []),
                *(getattr(agent, "_extra_tools", []) or []),
            ]:
                tool_name = str(getattr(tool, "name", "") or "").strip()
                if not tool_name:
                    continue
                inventory[tool_name] = {
                    "name": tool_name,
                    "description": str(getattr(tool, "description", "") or "").strip(),
                    "parameters": copy.deepcopy(getattr(tool, "parameters", []) or []),
                }
            return inventory
        finally:
            try:
                agent.close()
            except Exception:
                pass

    def _load_rows(self) -> List[Dict[str, Any]]:
        with self.csv_path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            rows: List[Dict[str, Any]] = []
            for idx, row in enumerate(reader):
                question = str((row or {}).get("question", "") or "").strip()
                if not question:
                    continue
                rows.append(
                    {
                        "question_id": f"q{idx}",
                        "question_index": str((row or {}).get("question_index", idx) or idx),
                        "question_type": str((row or {}).get("question_type", "") or "").strip(),
                        "question": question,
                    }
                )
                if self.question_limit is not None and len(rows) >= self.question_limit:
                    break
        return rows

    @staticmethod
    def _has_error_payload(text: Any) -> bool:
        raw = str(text or "").strip()
        if not raw:
            return False
        try:
            payload = json.loads(raw)
        except Exception:
            return False
        return isinstance(payload, dict) and bool(payload.get("error"))

    @staticmethod
    def _guarded_prompt(question: str) -> str:
        raw = str(question or "").strip()
        return "\n".join(
            [
                "Use retrieval tools before answering.",
                "You must call at least one tool and ground the answer in retrieved evidence.",
                "",
                raw,
            ]
        ).strip()

    def _write_progress(
        self,
        *,
        status: str,
        question_count: int,
        total_attempts: int,
        completed_attempts: int,
        completed_questions: int,
        current_question_id: str = "",
        current_attempt_index: int = -1,
        note: str = "",
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        payload: Dict[str, Any] = {
            "status": str(status or "").strip(),
            "dataset_name": self.dataset_name,
            "question_count": int(question_count or 0),
            "attempts_per_question": int(self.attempts_per_question or 0),
            "total_attempts": int(total_attempts or 0),
            "completed_attempts": int(completed_attempts or 0),
            "completed_questions": int(completed_questions or 0),
            "current_question_id": str(current_question_id or "").strip(),
            "current_attempt_index": int(current_attempt_index if current_attempt_index >= 0 else -1),
            "note": str(note or "").strip(),
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        if total_attempts > 0:
            payload["attempt_progress"] = round(float(completed_attempts) / float(total_attempts), 4)
        if question_count > 0:
            payload["question_progress"] = round(float(completed_questions) / float(question_count), 4)
        if extra:
            payload.update(extra)
        json_dump_atomic(str(self.progress_path), payload)

    def _run_attempt(self, row: Dict[str, Any], attempt_index: int) -> Dict[str, Any]:
        state = self._get_worker_context()
        agent: QuestionAnsweringAgent = state["agent"]
        started = time.time()
        base_session_id = f"online_{row['question_id']}_run_{attempt_index}"
        responses: List[Dict[str, Any]] = []
        final_answer = ""
        tool_uses: List[Dict[str, Any]] = []
        memory_context: Dict[str, Any] = {}
        prompt_variant = ""
        error_text = ""
        selected_prompt = ""
        selected_max_calls = max(6, int(self.max_llm_calls_per_attempt or 1))
        for variant_name, prompt in (
            ("guarded", self._guarded_prompt(str(row.get("question", "") or ""))),
            ("base", str(row.get("question", "") or "").strip()),
        ):
            if not prompt:
                continue
            for max_calls in (
                max(6, int(self.max_llm_calls_per_attempt or 1)),
                max(10, int(self.max_llm_calls_per_attempt or 1) + 4),
            ):
                responses = agent.ask(
                    prompt,
                    lang=str(state["cfg"].global_config.language or "zh"),
                    session_id=f"{base_session_id}_{variant_name}_{max_calls}",
                    max_llm_calls_per_run=max_calls,
                    online_learning=False,
                    require_tool_use=True,
                )
                final_answer = agent.extract_final_text(responses)
                tool_uses = agent.extract_tool_uses(responses)
                memory_context = agent.get_last_strategy_context()
                prompt_variant = variant_name
                error_text = ""
                if not tool_uses:
                    error_text = "missing_tool_use"
                    continue
                if self._has_error_payload(final_answer):
                    error_text = "llm_error_payload"
                    continue
                selected_prompt = prompt
                selected_max_calls = max_calls
                break
            if tool_uses and not self._has_error_payload(final_answer):
                break
        if selected_prompt:
            responses = agent.ask(
                selected_prompt,
                lang=str(state["cfg"].global_config.language or "zh"),
                session_id=f"{base_session_id}_{prompt_variant}_{selected_max_calls}_final",
                max_llm_calls_per_run=selected_max_calls,
                online_learning=True,
                require_tool_use=True,
            )
            final_answer = agent.extract_final_text(responses)
            tool_uses = agent.extract_tool_uses(responses)
            memory_context = agent.get_last_strategy_context()
            if not tool_uses:
                error_text = "missing_tool_use"
            elif self._has_error_payload(final_answer):
                error_text = "llm_error_payload"
            else:
                error_text = ""
        else:
            responses = agent.ask(
                self._guarded_prompt(str(row.get("question", "") or "")),
                lang=str(state["cfg"].global_config.language or "zh"),
                session_id=f"{base_session_id}_final_fallback",
                max_llm_calls_per_run=max(10, int(self.max_llm_calls_per_attempt or 1) + 4),
                online_learning=True,
                require_tool_use=True,
            )
            final_answer = agent.extract_final_text(responses)
            tool_uses = agent.extract_tool_uses(responses)
            memory_context = agent.get_last_strategy_context()
            if not tool_uses:
                error_text = "missing_tool_use"
            elif self._has_error_payload(final_answer):
                error_text = "llm_error_payload"
        latency_ms = int(round((time.time() - started) * 1000.0))
        payload = {
            "question_id": row["question_id"],
            "question_index": row["question_index"],
            "question_type": row["question_type"],
            "question": row["question"],
            "attempt_index": int(attempt_index),
            "session_id": base_session_id,
            "final_answer": final_answer,
            "prompt_variant": prompt_variant,
            "tool_names": [
                str(item.get("tool_name", "") or "").strip()
                for item in (tool_uses or [])
                if isinstance(item, dict) and str(item.get("tool_name", "") or "").strip()
            ],
            "tool_use_count": len(tool_uses or []),
            "latency_ms": latency_ms,
            "memory_context": memory_context,
            "error": error_text,
            "required_retrieval_enforced": True,
        }
        self._append_jsonl(self.attempts_path, payload)
        return payload

    def _clear_online_buffers(self) -> None:
        for name in (
            "real_traces.jsonl",
            "online_strategy_buffer.jsonl",
            "failure_reflections.jsonl",
            "synthetic_qa_buffer.jsonl",
        ):
            path = self.online_buffer_dir / name
            if path.exists():
                path.unlink()

    @staticmethod
    def _raw_tool_chain(tool_uses: List[Dict[str, Any]]) -> List[str]:
        return [
            str(item.get("tool_name", "") or "").strip()
            for item in (tool_uses or [])
            if isinstance(item, dict) and str(item.get("tool_name", "") or "").strip()
        ]

    @staticmethod
    def _comparison_chain(attempt: Dict[str, Any]) -> List[str]:
        chain = attempt.get("minimal_effective_chain") or attempt.get("effective_tool_chain") or attempt.get("raw_tool_chain") or []
        return [str(x).strip() for x in chain if str(x).strip()]

    @staticmethod
    def _online_pattern_key(row: Dict[str, Any]) -> str:
        query_pattern = row.get("query_pattern") if isinstance(row.get("query_pattern"), dict) else {}
        return " | ".join(
            [
                str(query_pattern.get("problem_type", "") or "").strip().lower(),
                str(query_pattern.get("answer_shape", "") or "").strip().lower(),
                str(query_pattern.get("query_abstract", "") or "").strip().lower(),
            ]
        ).strip()

    def _select_online_strategy_candidates_for_library(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not rows:
            return []
        grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[self._online_pattern_key(row)].append(row)
        selected: List[Dict[str, Any]] = []
        for row in rows:
            key = self._online_pattern_key(row)
            group = grouped.get(key, [])
            judge_score = float(((row.get("judge") or {}).get("score", 0.0)) or 0.0)
            matched_template_ids = [str(x).strip() for x in (row.get("matched_template_ids") or []) if str(x).strip()]
            clearly_distinct = not matched_template_ids
            if len(group) >= 2 or judge_score >= 0.9 or clearly_distinct:
                selected.append(row)
        return selected

    def _get_tool_metadata_for_reflection(self, tool_name: str) -> Dict[str, Any]:
        inventory_meta = self.tool_inventory.get(str(tool_name or "").strip(), {})
        return self.tool_metadata_provider.resolve_tool_metadata(
            str(tool_name or "").strip(),
            fallback_description=str(inventory_meta.get("description", "") or ""),
            fallback_parameters=inventory_meta.get("parameters") or [],
        )

    def _question_lookup(self, rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        return {str(item.get("question", "") or ""): item for item in rows}

    def _template_from_online_candidate(
        self,
        *,
        row: Dict[str, Any],
        idx: int,
        question_lookup: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        question = str(row.get("question", "") or "").strip()
        question_meta = question_lookup.get(question, {})
        score = float(((row.get("judge") or {}).get("score", 0.0)) or 0.0)
        return {
            "template_id": f"ost_{idx:04d}",
            "question_id": str(question_meta.get("question_id", "") or ""),
            "question": question,
            "pattern_name": str(row.get("pattern_name", "") or "").strip(),
            "pattern_description": str(row.get("pattern_description", "") or "").strip(),
            "raw_tool_chain": list(row.get("raw_tool_chain") or []),
            "minimal_effective_chain": list(row.get("minimal_effective_chain") or row.get("effective_tool_chain") or row.get("recommended_chain") or []),
            "recommended_chain": list(row.get("recommended_chain") or []),
            "anti_patterns": list(row.get("anti_patterns") or []),
            "chain_rationale": str(row.get("chain_rationale", "") or "").strip(),
            "chain_constraints": list(row.get("chain_constraints") or []),
            "support_count": 1,
            "successful_attempts": 1,
            "attempt_count": 1,
            "success_rate": score if score > 0.0 else 0.8,
            "score_supervision": "none",
            "query_pattern": dict(row.get("query_pattern") or {}),
            "query_abstract": str((row.get("query_pattern") or {}).get("query_abstract", "") or "").strip(),
            "template_sources": [
                {
                    "question_id": str(question_meta.get("question_id", "") or ""),
                    "question": question,
                    "source_type": str(row.get("online_source", "online_real") or "online_real"),
                }
            ],
        }

    def _template_from_synthetic_record(
        self,
        *,
        row: Dict[str, Any],
        idx: int,
    ) -> Optional[Dict[str, Any]]:
        if not bool(row.get("accepted", False)):
            return None
        question = str(row.get("question", "") or "").strip()
        reference_answer = str(row.get("reference_answer", "") or "").strip()
        agent_answer = str(row.get("agent_answer", "") or "").strip()
        tool_uses = list(row.get("tool_uses") or [])
        if not question or not reference_answer or not agent_answer or not tool_uses:
            return None
        query_pattern = self.query_pattern_extractor.extract(question)
        effective_chain = self.chain_extractor.extract(
            question=question,
            reference_answer=reference_answer,
            candidate_answer=agent_answer,
            tool_uses=tool_uses,
        )
        raw_tool_chain = self._raw_tool_chain(tool_uses)
        best_attempt = {
            "question": question,
            "query_pattern": query_pattern,
            "candidate_answer": agent_answer,
            "judge": row.get("judge") or {},
            "raw_tool_chain": raw_tool_chain,
            "minimal_effective_chain": effective_chain.get("minimal_effective_chain") or effective_chain.get("effective_tool_chain") or raw_tool_chain,
            "effective_tool_chain": effective_chain.get("effective_tool_chain") or raw_tool_chain,
            "step_attributions": effective_chain.get("step_attributions") or [],
            "effective_chain_reason": str(effective_chain.get("reason", "") or "").strip(),
            "tool_summary_json": json.dumps(tool_uses, ensure_ascii=False),
            "evidence_summary": str(row.get("generated_evidence", "") or "").strip(),
        }
        distilled = self.template_distiller.distill(
            question=question,
            query_pattern=query_pattern,
            best_attempt=best_attempt,
            failed_attempts=[],
            retry_instruction="",
        )
        return {
            "template_id": f"syn_{idx:04d}",
            "question_id": "",
            "question": question,
            "pattern_name": str(distilled.get("pattern_name", "") or "").strip(),
            "pattern_description": str(distilled.get("pattern_description", "") or "").strip(),
            "raw_tool_chain": raw_tool_chain,
            "minimal_effective_chain": effective_chain.get("minimal_effective_chain") or effective_chain.get("effective_tool_chain") or raw_tool_chain,
            "recommended_chain": list(distilled.get("recommended_chain") or []),
            "anti_patterns": list(distilled.get("anti_patterns") or []),
            "chain_rationale": str(distilled.get("chain_rationale", "") or "").strip(),
            "chain_constraints": list(distilled.get("chain_constraints") or []),
            "support_count": 1,
            "successful_attempts": 1,
            "attempt_count": 1,
            "success_rate": float(((row.get("judge") or {}).get("score", 0.0)) or 0.9),
            "score_supervision": "none",
            "query_pattern": query_pattern,
            "query_abstract": str(query_pattern.get("query_abstract", "") or "").strip(),
            "template_sources": [
                {
                    "question_id": "",
                    "question": question,
                    "source_type": "synthetic_bootstrap",
                    "source_question": str(row.get("source_question", "") or "").strip(),
                }
            ],
        }

    def _build_question_summaries(
        self,
        *,
        rows: List[Dict[str, Any]],
        real_traces: List[Dict[str, Any]],
        strategy_candidates: List[Dict[str, Any]],
        synthetic_rows: List[Dict[str, Any]],
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        question_by_text = self._question_lookup(rows)
        traces_by_question: Dict[str, List[Dict[str, Any]]] = {}
        for trace in real_traces:
            question = str(trace.get("question", "") or "").strip()
            if question:
                traces_by_question.setdefault(question, []).append(trace)
        candidates_by_question: Dict[str, List[Dict[str, Any]]] = {}
        for item in strategy_candidates:
            question = str(item.get("question", "") or "").strip()
            if question:
                candidates_by_question.setdefault(question, []).append(item)
        synthetic_by_source: Dict[str, List[Dict[str, Any]]] = {}
        for item in synthetic_rows:
            source_question = str(item.get("source_question", "") or "").strip()
            if source_question:
                synthetic_by_source.setdefault(source_question, []).append(item)

        summaries: List[Dict[str, Any]] = []
        failures: List[Dict[str, Any]] = []
        for row in rows:
            question = str(row.get("question", "") or "").strip()
            traces = traces_by_question.get(question, [])
            candidates = candidates_by_question.get(question, [])
            accepted_synthetic = [item for item in synthetic_by_source.get(question, []) if bool(item.get("accepted", False))]
            avg_score = 0.0
            if traces:
                avg_score = round(
                    sum(float(((item.get("judge") or {}).get("score", 0.0)) or 0.0) for item in traces) / float(len(traces)),
                    4,
                )
            summary = {
                "question_id": row["question_id"],
                "question_index": row["question_index"],
                "question_type": row["question_type"],
                "question": question,
                "attempt_count": len(traces),
                "strategy_candidate_count": len(candidates),
                "accepted_synthetic_count": len(accepted_synthetic),
                "avg_judge_score": avg_score,
                "best_judge_score": max(
                    [float(((item.get("judge") or {}).get("score", 0.0)) or 0.0) for item in traces] or [0.0]
                ),
            }
            summaries.append(summary)
            if not candidates:
                failures.append(
                    {
                        "question_id": row["question_id"],
                        "question": question,
                        "failure_summary": "No accepted online strategy candidate was produced from evidence-only judging.",
                        "likely_causes": [
                            "The answer was not sufficiently supported by retrieved evidence.",
                            "The evidence-consistency judge score stayed below the online strategy threshold.",
                        ],
                        "recommended_improvements": [
                            "Increase evidence recall or allow more precise lexical retrieval.",
                            "Inspect online traces to see whether the tool chain found direct supporting text.",
                        ],
                    }
                )
        return summaries, failures

    def _summarize_attempt_for_reflection(self, attempt: Dict[str, Any]) -> Dict[str, Any]:
        judge = attempt.get("judge") if isinstance(attempt.get("judge"), dict) else {}
        return {
            "attempt_id": str(attempt.get("attempt_id", "") or attempt.get("session_id", "") or ""),
            "raw_tool_chain": list(attempt.get("raw_tool_chain", []) or [])[:12],
            "effective_tool_chain": list(attempt.get("effective_tool_chain", []) or [])[:12],
            "final_answer": self._clip_text(attempt.get("final_answer", ""), limit=800),
            "judge": {
                "is_success": bool(judge.get("is_success", False)),
                "score": float(judge.get("score", 0.0) or 0.0),
                "reason": self._clip_text(judge.get("reason", ""), limit=500),
                "intermediate_value_score": float(judge.get("intermediate_value_score", 0.0) or 0.0),
                "answer_support_score": float(judge.get("answer_support_score", 0.0) or 0.0),
            },
            "retry_instruction": self._clip_text(attempt.get("retry_instruction", ""), limit=220),
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

    def _build_failed_attempts_for_reflection(
        self,
        *,
        question: str,
        real_traces: List[Dict[str, Any]],
        failure_reflections: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        reflection_by_session: Dict[str, Dict[str, Any]] = {}
        for item in failure_reflections:
            session_id = str(item.get("session_id", "") or "").strip()
            item_question = str(item.get("question", "") or "").strip()
            if session_id and item_question == question:
                reflection_by_session[session_id] = item

        failed: List[Dict[str, Any]] = []
        for trace in real_traces:
            if str(trace.get("question", "") or "").strip() != question:
                continue
            judge = trace.get("judge") if isinstance(trace.get("judge"), dict) else {}
            if bool(judge.get("is_success", False)):
                continue
            session_id = str(trace.get("session_id", "") or "").strip()
            reflection_row = reflection_by_session.get(session_id, {})
            reflection_payload = reflection_row.get("reflection") if isinstance(reflection_row.get("reflection"), dict) else {}
            failed.append(
                {
                    "attempt_id": session_id,
                    "session_id": session_id,
                    "raw_tool_chain": self._raw_tool_chain(list(trace.get("tool_uses") or [])),
                    "effective_tool_chain": [],
                    "final_answer": str(trace.get("final_answer", "") or "").strip(),
                    "judge": judge,
                    "retry_instruction": str(reflection_row.get("retry_instruction", "") or "").strip(),
                    "error": str(reflection_payload.get("next_action", "") or reflection_payload.get("missed_fact", "") or "").strip(),
                }
            )
        failed.sort(
            key=lambda item: (
                float(((item.get("judge") or {}).get("score", 0.0)) or 0.0),
                float(((item.get("judge") or {}).get("intermediate_value_score", 0.0)) or 0.0),
            )
        )
        return failed[:3]

    def _run_tool_description_reflection(
        self,
        *,
        rows: List[Dict[str, Any]],
        strategy_candidates: List[Dict[str, Any]],
        real_traces: List[Dict[str, Any]],
        failure_reflections: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        question_lookup = self._question_lookup(rows)
        candidates_by_question: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for item in strategy_candidates:
            question = str(item.get("question", "") or "").strip()
            if question:
                candidates_by_question[question].append(item)

        records: List[Dict[str, Any]] = []
        for question, candidate_rows in candidates_by_question.items():
            if not candidate_rows:
                continue
            best_attempt = max(
                candidate_rows,
                key=lambda item: (
                    float(((item.get("judge") or {}).get("score", 0.0)) or 0.0),
                    float(((item.get("judge") or {}).get("intermediate_value_score", 0.0)) or 0.0),
                    len(item.get("effective_tool_chain") or []),
                ),
            )
            failed_attempts = self._build_failed_attempts_for_reflection(
                question=question,
                real_traces=real_traces,
                failure_reflections=failure_reflections,
            )
            tool_names = self._select_reflection_tool_names(best_attempt=best_attempt, failed_attempts=failed_attempts)
            if not tool_names:
                continue

            query_pattern = dict(best_attempt.get("query_pattern") or {})
            if not query_pattern:
                query_pattern = self.query_pattern_extractor.extract(question)
            successful_attempt_summary = self._summarize_attempt_for_reflection(best_attempt)
            question_meta = question_lookup.get(question, {})

            for tool_name in tool_names:
                relevant_failed = [
                    item for item in failed_attempts
                    if tool_name not in set(self._comparison_chain(item))
                ]
                if not relevant_failed:
                    continue
                failed_summaries = [self._summarize_attempt_for_reflection(item) for item in relevant_failed[:3]]
                tool_meta = self._get_tool_metadata_for_reflection(tool_name)
                reflection = self.tool_description_reflector.reflect(
                    tool_name=tool_name,
                    current_description=str(tool_meta.get("description", "") or ""),
                    parameters=tool_meta.get("parameters") or [],
                    question=question,
                    query_pattern=query_pattern,
                    successful_attempt=successful_attempt_summary,
                    failed_attempts=failed_summaries,
                )
                records.append(
                    {
                        "question_id": str(question_meta.get("question_id", "") or ""),
                        "question": question,
                        "language": str(self.config.global_config.language or "zh"),
                        "tool_name": tool_name,
                        "current_description": str(tool_meta.get("description", "") or ""),
                        "decision": str(reflection.get("decision", "keep") or "keep"),
                        "proposed_description": str(reflection.get("proposed_description", tool_meta.get("description", "")) or ""),
                        "reason": self._clip_text(reflection.get("reason", ""), limit=1000),
                        "best_attempt_id": str(best_attempt.get("session_id", "") or ""),
                        "failed_attempt_ids": [str(item.get("attempt_id", "") or "") for item in relevant_failed[:3]],
                        "missing_from_failed_attempt_count": len(relevant_failed),
                    }
                )
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
                supporting_items = [
                    item
                    for item in revise_items
                    if str(item.get("proposed_description", "") or "").strip() == proposed_description
                ]
                reason = Counter(str(item.get("reason", "") or "").strip() for item in supporting_items).most_common(1)[0][0]
                decision = "revise"
            else:
                proposed_description = current_description
                support_count = len(keep_items) if keep_items else len(items)
                reason = Counter(str(item.get("reason", "") or "").strip() for item in keep_items).most_common(1)[0][0] if keep_items else "keep_current_description_by_majority"
                decision = "keep"
            aggregated.append(
                {
                    "tool_name": tool_name,
                    "language": language,
                    "decision": decision,
                    "current_description": current_description,
                    "proposed_description": proposed_description,
                    "support_count": support_count,
                    "record_count": len(items),
                    "reason": reason,
                }
            )
        return aggregated

    def _build_markdown_report(
        self,
        *,
        manifest: Dict[str, Any],
        question_summaries: List[Dict[str, Any]],
        raw_template_count: int,
        cluster_payload: Dict[str, Any],
        real_trace_count: int,
        strategy_candidate_count: int,
        synthetic_qa_count: int,
        tool_reflection_record_count: int,
        tool_description_candidate_count: int,
    ) -> str:
        lines = [
            "# Online Strategy Training Report",
            "",
            f"- Dataset: `{self.dataset_name}`",
            f"- Source CSV: `{self.csv_path}`",
            f"- Question count: `{manifest.get('question_count', 0)}`",
            f"- Attempts per question: `{manifest.get('attempts_per_question', 0)}`",
            f"- Real trace count: `{real_trace_count}`",
            f"- Online strategy candidate count: `{strategy_candidate_count}`",
            f"- Accepted synthetic QA count: `{synthetic_qa_count}`",
            f"- Tool description reflection records: `{tool_reflection_record_count}`",
            f"- Tool description candidate count: `{tool_description_candidate_count}`",
            f"- Raw template count: `{raw_template_count}`",
            f"- Cluster count: `{cluster_payload.get('cluster_count', 0)}`",
            f"- Runtime library: `{self.runtime_library_path}`",
            "",
            "## Per-Question Summary",
            "",
        ]
        for item in question_summaries:
            lines.append(
                f"- {item['question_id']}: attempts={item['attempt_count']}, "
                f"templates={item['strategy_candidate_count']}, synthetic={item['accepted_synthetic_count']}, "
                f"avg_score={item['avg_judge_score']}"
            )
        return "\n".join(lines) + "\n"

    def run(self) -> Dict[str, Any]:
        rows = self._load_rows()
        if not rows:
            raise RuntimeError(f"No valid questions found in {self.csv_path}")
        self._clear_online_buffers()
        if self.attempts_path.exists():
            self.attempts_path.unlink()

        manifest = {
            "dataset_name": self.dataset_name,
            "csv_path": str(self.csv_path),
            "question_count": len(rows),
            "attempts_per_question": self.attempts_per_question,
            "training_max_workers": self.training_max_workers,
            "self_bootstrap_max_questions": self.self_bootstrap_max_questions,
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "mode": "online_strategy_training_without_reference_answers",
        }
        json_dump_atomic(str(self.manifest_path), manifest)
        total_attempts = len(rows) * self.attempts_per_question
        self._write_progress(
            status="running",
            question_count=len(rows),
            total_attempts=total_attempts,
            completed_attempts=0,
            completed_questions=0,
            note="starting_online_strategy_training",
        )

        jobs: List[tuple[Dict[str, Any], int]] = []
        for row in rows:
            for attempt_index in range(self.attempts_per_question):
                jobs.append((row, attempt_index))

        completed_attempts = 0
        question_attempt_counts: Dict[str, int] = {}
        with ThreadPoolExecutor(max_workers=self.training_max_workers) as executor:
            future_map = {
                executor.submit(self._run_attempt, row, attempt_index): (row, attempt_index)
                for row, attempt_index in jobs
            }
            for future in tqdm(as_completed(future_map), total=len(future_map), desc="online_strategy_training", ncols=100):
                row, attempt_index = future_map[future]
                try:
                    future.result()
                except Exception as exc:
                    logger.warning(
                        "online strategy attempt failed: question_id=%s attempt=%s err=%s",
                        row.get("question_id", ""),
                        attempt_index,
                        exc,
                    )
                completed_attempts += 1
                qid = str(row.get("question_id", "") or "").strip()
                if qid:
                    question_attempt_counts[qid] = question_attempt_counts.get(qid, 0) + 1
                completed_questions = sum(
                    1
                    for count in question_attempt_counts.values()
                    if count >= int(self.attempts_per_question or 1)
                )
                self._write_progress(
                    status="running",
                    question_count=len(rows),
                    total_attempts=total_attempts,
                    completed_attempts=completed_attempts,
                    completed_questions=completed_questions,
                    current_question_id=qid,
                    current_attempt_index=attempt_index,
                    note="running_online_attempts",
                )

        for ctx in self._worker_contexts:
            try:
                ctx["agent"].wait_for_online_learning()
            except Exception as exc:
                logger.warning("wait_for_online_learning failed: %s", exc)

        buffer = OnlineStrategyBuffer.from_config(self._build_agent_config())
        real_traces = buffer.load_real_traces()
        buffered_strategy_candidates = [
            row for row in buffer.load_strategy_candidates()
            if list(row.get("tool_uses") or [])
            or list(row.get("raw_tool_chain") or [])
            or list(row.get("effective_tool_chain") or [])
        ]
        strategy_candidates = self._select_online_strategy_candidates_for_library(buffered_strategy_candidates)
        failure_reflections = buffer.load_failure_reflections()
        synthetic_rows = buffer.load_synthetic_qas()
        accepted_synthetic_rows = [row for row in synthetic_rows if bool(row.get("accepted", False))]

        question_lookup = self._question_lookup(rows)
        raw_templates: List[Dict[str, Any]] = []
        for idx, row in enumerate(strategy_candidates, start=1):
            raw_templates.append(
                self._template_from_online_candidate(
                    row=row,
                    idx=idx,
                    question_lookup=question_lookup,
                )
            )
        synthetic_template_count = 0
        for idx, row in enumerate(accepted_synthetic_rows, start=1):
            template = self._template_from_synthetic_record(row=row, idx=idx)
            if template is None:
                continue
            raw_templates.append(template)
            synthetic_template_count += 1

        question_summaries, failed_summaries = self._build_question_summaries(
            rows=rows,
            real_traces=real_traces,
            strategy_candidates=strategy_candidates,
            synthetic_rows=accepted_synthetic_rows,
        )
        tool_description_records = self._run_tool_description_reflection(
            rows=rows,
            strategy_candidates=strategy_candidates,
            real_traces=real_traces,
            failure_reflections=failure_reflections,
        )
        tool_description_candidates = self._aggregate_tool_description_records(tool_description_records)

        self.cluster_manager.add_templates(raw_templates)
        consolidation_merge_count = self.cluster_manager.consolidate()
        cluster_payload = self.cluster_manager.export_training_payload()
        runtime_templates = self.cluster_manager.runtime_templates()
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

        generated_at = time.strftime("%Y-%m-%d %H:%M:%S")
        training_library_payload = {
            "library_version": 3,
            "aggregation_mode": "narrative",
            "dataset_name": self.dataset_name,
            "generated_at": generated_at,
            "raw_template_count": len(raw_templates),
            "cluster_count": cluster_payload.get("cluster_count", 0),
            "clusters": cluster_payload.get("clusters", []),
            "pattern_count": len(runtime_templates),
            "patterns": runtime_templates,
            "templates": runtime_templates,
        }
        runtime_library_payload = {
            "library_version": 3,
            "aggregation_mode": "narrative",
            "dataset_name": self.dataset_name,
            "generated_at": generated_at,
            "pattern_count": len(runtime_templates),
            "patterns": runtime_templates,
            "template_count": len(runtime_templates),
            "templates": runtime_templates,
        }

        json_dump_atomic(str(self.question_summary_path), question_summaries)
        json_dump_atomic(str(self.raw_template_path), raw_templates)
        json_dump_atomic(str(self.failure_summary_path), failed_summaries)
        self._write_jsonl(self.tool_reflection_record_path, tool_description_records)
        json_dump_atomic(str(self.tool_description_candidate_path), tool_description_candidates)
        json_dump_atomic(str(self.cluster_path), cluster_payload)
        self._write_jsonl(self.merge_decision_path, self.cluster_manager.merge_decisions)
        json_dump_atomic(str(self.library_output_path), training_library_payload)
        json_dump_atomic(str(self.template_source_index_path), template_source_index)
        json_dump_atomic(str(self.source_question_hints_path), {})
        json_dump_atomic(str(self.runtime_library_path), runtime_library_payload)
        runtime_tool_metadata_paths = self.runtime_asset_manager.export_tool_metadata_overrides(tool_description_candidates)
        runtime_template_source_index_path = self.runtime_asset_manager.export_template_source_index(template_source_index)
        runtime_source_hint_path = self.runtime_library_path.parent / "source_question_hints.json"
        json_dump_atomic(str(runtime_source_hint_path), {})
        self.report_path.write_text(
            self._build_markdown_report(
                manifest=manifest,
                question_summaries=question_summaries,
                raw_template_count=len(raw_templates),
                cluster_payload=cluster_payload,
                real_trace_count=len(real_traces),
                strategy_candidate_count=len(strategy_candidates),
                synthetic_qa_count=len(accepted_synthetic_rows),
                tool_reflection_record_count=len(tool_description_records),
                tool_description_candidate_count=len(tool_description_candidates),
            ),
            encoding="utf-8",
        )
        self._write_progress(
            status="completed",
            question_count=len(rows),
            total_attempts=total_attempts,
            completed_attempts=total_attempts,
            completed_questions=len(rows),
            note="completed_online_strategy_training",
            extra={
                "real_trace_count": len(real_traces),
                "online_strategy_candidate_count": len(strategy_candidates),
                "accepted_synthetic_qa_count": len(accepted_synthetic_rows),
                "tool_description_reflection_record_count": len(tool_description_records),
                "tool_description_candidate_count": len(tool_description_candidates),
                "raw_template_count": len(raw_templates),
                "cluster_count": cluster_payload.get("cluster_count", 0),
            },
        )

        return {
            "manifest_path": str(self.manifest_path),
            "attempts_path": str(self.attempts_path),
            "progress_path": str(self.progress_path),
            "raw_template_path": str(self.raw_template_path),
            "question_summary_path": str(self.question_summary_path),
            "failure_summary_path": str(self.failure_summary_path),
            "tool_reflection_record_path": str(self.tool_reflection_record_path),
            "tool_description_candidate_path": str(self.tool_description_candidate_path),
            "cluster_path": str(self.cluster_path),
            "merge_decision_path": str(self.merge_decision_path),
            "library_output_path": str(self.library_output_path),
            "template_source_index_path": str(self.template_source_index_path),
            "source_question_hints_path": str(self.source_question_hints_path),
            "runtime_library_path": str(self.runtime_library_path),
            "runtime_tool_metadata_dir": str(self.runtime_tool_metadata_dir),
            "runtime_tool_metadata_paths": runtime_tool_metadata_paths,
            "runtime_template_source_index_path": runtime_template_source_index_path,
            "runtime_source_question_hints_path": str(runtime_source_hint_path),
            "report_path": str(self.report_path),
            "question_count": len(rows),
            "real_trace_count": len(real_traces),
            "online_strategy_candidate_count": len(strategy_candidates),
            "failure_reflection_count": len(failure_reflections),
            "accepted_synthetic_qa_count": len(accepted_synthetic_rows),
            "tool_description_reflection_record_count": len(tool_description_records),
            "tool_description_candidate_count": len(tool_description_candidates),
            "synthetic_template_count": synthetic_template_count,
            "raw_template_count": len(raw_templates),
            "cluster_count": cluster_payload.get("cluster_count", 0),
            "consolidation_merge_count": consolidation_merge_count,
        }

    def close(self) -> None:
        for ctx in self._worker_contexts:
            try:
                ctx["agent"].close()
            except Exception:
                pass

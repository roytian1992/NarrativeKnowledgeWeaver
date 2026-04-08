from __future__ import annotations

import logging
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.functions.memory_management.strategy_query_pattern import (
    StrategyQueryPatternExtractor,
    canonicalize_query_text,
)
from core.memory.base_memory import BaseMemoryStore
from core.memory.strategy_library import StrategyTemplateLibrary
from core.utils.config import KAGConfig
from core.utils.general_utils import json_dump_atomic, load_json, token_jaccard_overlap
from core.utils.prompt_loader import YAMLPromptLoader

logger = logging.getLogger(__name__)


_CHAIN_TOOL_ALIASES = {
    "retrieve_scenes_by_entity": "get_entity_sections",
    "retrieve_scene_by_entity": "get_entity_sections",
    "retrieve_sections_by_entity": "get_entity_sections",
    "retrieve_scenes_by_character": "get_entity_sections",
    "retrieve_scenes_by_object": "get_entity_sections",
    "retrieve_scene_titles_by_document_ids": "lookup_titles_by_document_ids",
}


class RetrievalStrategyMemory(BaseMemoryStore):
    """
    Runtime-only retrieval strategy memory.

    This class no longer performs online training. It loads a trained strategy
    library from JSON and provides read-time routing hints for retriever_agent.
    """

    def __init__(
        self,
        *,
        config: KAGConfig,
        llm: Optional[Any] = None,
        embedding_model: Optional[Any] = None,
    ) -> None:
        self.config = config
        self.llm = llm
        self.embedding_model = embedding_model

        cfg = getattr(config, "strategy_memory", None)
        self.enabled = bool(getattr(cfg, "enabled", False))
        self.read_enabled = bool(getattr(cfg, "read_enabled", True))
        self.library_path = str(
            getattr(cfg, "library_path", "data/memory/strategy/strategy_library.json")
            or "data/memory/strategy/strategy_library.json"
        ).strip()
        configured_source_hint_path = str(
            getattr(cfg, "source_question_hint_path", "")
            or ""
        ).strip()
        if configured_source_hint_path:
            self.source_question_hint_path = configured_source_hint_path
        else:
            self.source_question_hint_path = str(
                Path(self.library_path).expanduser().resolve().parent / "source_question_hints.json"
            )
        self.top_k_templates = max(1, int(getattr(cfg, "top_k_templates", getattr(cfg, "top_k_patterns", 3)) or 3))
        self.max_active_patterns = max(1, int(getattr(cfg, "max_active_patterns", 2) or 2))
        self.max_hint_lines = max(2, int(getattr(cfg, "max_hint_lines", 6) or 6))
        self.runtime_routing_note_enabled = bool(getattr(cfg, "runtime_routing_note_enabled", False))
        self.min_template_support = max(1, int(getattr(cfg, "min_template_support", 1) or 1))
        self.min_similarity = float(getattr(cfg, "min_similarity", 0.0) or 0.0)
        self.match_min_score = float(getattr(cfg, "match_min_score", 0.62) or 0.62)
        self.selection_min_score = float(getattr(cfg, "selection_min_score", 0.4) or 0.4)
        single_override = getattr(cfg, "single_agent_min_selection_score", None)
        subagent_override = getattr(cfg, "subagent_min_selection_score", None)
        self.single_agent_min_selection_score = float(
            self.selection_min_score if single_override is None else (single_override or 0.0)
        )
        self.subagent_min_selection_score = float(
            self.selection_min_score if subagent_override is None else (subagent_override or 0.0)
        )
        self.subagent_enabled = bool(getattr(cfg, "subagent_enabled", True))
        self.subagent_max_branches = max(2, int(getattr(cfg, "subagent_max_branches", 5) or 5))
        self.report_path = str(getattr(cfg, "report_path", "") or "").strip()

        prompt_dir = str(
            getattr(getattr(self.config, "global_", None), "prompt_dir", "")
            or getattr(getattr(self.config, "global_config", None), "prompt_dir", "")
            or ""
        ).strip()
        self.prompt_loader: Optional[YAMLPromptLoader] = None
        if prompt_dir:
            try:
                self.prompt_loader = YAMLPromptLoader(prompt_dir)
            except Exception as exc:
                logger.warning("strategy memory prompt loader init failed: %s", exc)

        self.query_pattern_extractor = StrategyQueryPatternExtractor(
            prompt_loader=self.prompt_loader,
            llm=self.llm,
            prompt_id=str(getattr(cfg, "pattern_extractor_prompt_id", "memory/extract_strategy_query_pattern") or "memory/extract_strategy_query_pattern").strip(),
            abstraction_mode=str(getattr(cfg, "abstraction_mode", "hybrid") or "hybrid").strip(),
        )
        self.library = StrategyTemplateLibrary(
            library_path=self.library_path,
            embedding_model=self.embedding_model,
            min_template_support=self.min_template_support,
        )
        self.source_question_hints = self._load_source_question_hints(self.source_question_hint_path)

    def add(self, entry: Dict[str, Any]) -> str:
        return ""

    def query(self, text: str, **kwargs) -> str:
        data = self.prepare_read_context(
            query=text,
            doc_type=str(kwargs.get("doc_type", "") or ""),
            mode=str(kwargs.get("mode", "") or ""),
        )
        return str(data.get("routing_hint", "") or "")

    def flush(self) -> None:
        return

    def abstract_query(self, query: str) -> str:
        canonical_query = canonicalize_query_text(query)
        return str(self.query_pattern_extractor.extract(canonical_query).get("query_abstract", "") or "")

    def analyze_query_pattern(self, query: str) -> Dict[str, Any]:
        return self.query_pattern_extractor.extract(canonicalize_query_text(query))

    def retrieve_patterns(
        self,
        *,
        query_abstract: str = "",
        query_pattern: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        pattern = query_pattern or {"query_abstract": str(query_abstract or "").strip()}
        rows = self.library.match_templates(query_pattern=pattern, top_k=top_k or self.top_k_templates)
        if self.min_similarity > -1.0:
            rows = [row for row in rows if float(row.get("similarity", 0.0) or 0.0) >= self.min_similarity]
        return rows

    def _effective_match_threshold(
        self,
        *,
        row: Dict[str, Any],
        query_pattern: Dict[str, Any],
        template_pattern: Dict[str, Any],
    ) -> float:
        threshold = float(self.match_min_score)
        if threshold <= 0.0:
            return threshold
        problem_type_match = self.library._problem_type_match(query_pattern or {}, template_pattern or {})
        answer_shape_match = self.library._answer_shape_match(query_pattern or {}, template_pattern or {})
        target_category_match = self.library._target_category_match(query_pattern or {}, template_pattern or {})
        quality_weight = self._quality_weight(row)
        support_count = max(0, int(row.get("support_count", 0) or 0))
        if problem_type_match >= 1.0 and answer_shape_match >= 1.0 and quality_weight >= 0.9 and support_count >= 3:
            threshold = min(threshold, 0.47)
            if target_category_match >= 1.0:
                threshold = min(threshold, 0.46)
        return threshold

    def prepare_read_context(self, *, query: str, doc_type: str, mode: str) -> Dict[str, Any]:
        canonical_query = canonicalize_query_text(query)
        query_pattern = self.analyze_query_pattern(canonical_query)
        if isinstance(query_pattern, dict):
            query_pattern["_raw_query"] = str(canonical_query or "").strip()
        query_abstract = str(query_pattern.get("query_abstract", "") or "")
        if not (self.enabled and self.read_enabled):
            return {
                "query_abstract": query_abstract,
                "query_pattern": query_pattern,
                "patterns": [],
                "routing_hint": "",
            }
        candidate_patterns = self.retrieve_patterns(query_abstract=query_abstract, query_pattern=query_pattern, top_k=self.top_k_templates)
        patterns = self._select_active_patterns(
            query=canonical_query,
            query_pattern=query_pattern,
            candidate_patterns=candidate_patterns,
        )
        return self.build_context_for_patterns(
            query=canonical_query,
            query_pattern=query_pattern,
            candidate_patterns=candidate_patterns,
            patterns=patterns,
        )

    def collect_runtime_matched_patterns(
        self,
        *,
        query: str,
        query_pattern: Dict[str, Any],
        candidate_patterns: List[Dict[str, Any]],
        max_patterns: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        query = canonicalize_query_text(query)
        rows = [copy for copy in (candidate_patterns or []) if isinstance(copy, dict)]
        if not rows:
            return []
        scored_rows: List[Dict[str, Any]] = []
        for item in rows:
            row = dict(item)
            base_similarity = float(row.get("similarity", 0.0) or 0.0)
            quality_weight = self._quality_weight(row)
            row["quality_weight"] = round(quality_weight, 6)
            row["selection_score"] = round(base_similarity * quality_weight, 6)
            scored_rows.append(row)
        scored_rows.sort(
            key=lambda item: (
                float(item.get("selection_score", 0.0) or 0.0),
                float(item.get("similarity", 0.0) or 0.0),
                float(StrategyTemplateLibrary.effective_success_rate_for_matching(item)),
                int(item.get("support_count", 0) or 0),
            ),
            reverse=True,
        )

        selected: List[Dict[str, Any]] = []
        for row in scored_rows:
            similarity = float(row.get("similarity", 0.0) or 0.0)
            score = float(row.get("selection_score", 0.0) or 0.0)
            template_pattern = row.get("query_pattern") if isinstance(row.get("query_pattern"), dict) else {}
            effective_threshold = self._effective_match_threshold(
                row=row,
                query_pattern=query_pattern,
                template_pattern=template_pattern,
            )
            if similarity < self.match_min_score:
                if similarity >= effective_threshold:
                    row["selection_threshold"] = round(effective_threshold, 6)
                else:
                    continue
            row["selection_threshold"] = round(effective_threshold, 6)
            if similarity < effective_threshold:
                continue
            if score < self.single_agent_min_selection_score:
                continue
            if not self._is_pattern_compatible(query_pattern=query_pattern, template_pattern=template_pattern):
                continue
            chain = self._normalize_chain([str(x).strip() for x in (row.get("recommended_chain") or []) if str(x).strip()])
            row["recommended_chain"] = chain
            if chain and not self._is_chain_compatible(query=query, query_pattern=query_pattern, chain=chain):
                continue
            selected.append(row)
            if max_patterns is not None and len(selected) >= max(1, int(max_patterns or 1)):
                break
        return selected

    def build_context_for_patterns(
        self,
        *,
        query: str,
        query_pattern: Dict[str, Any],
        candidate_patterns: Optional[List[Dict[str, Any]]] = None,
        patterns: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        rows = [row for row in (patterns or []) if isinstance(row, dict)]
        query_abstract = str(query_pattern.get("query_abstract", "") or "").strip()
        routing_hint = ""
        if self.runtime_routing_note_enabled:
            routing_hint = self._build_routing_hint(rows)
            exact_source_hint = self._build_exact_source_hint(patterns=rows)
            if exact_source_hint:
                routing_hint = f"{routing_hint}\n\n{exact_source_hint}".strip() if routing_hint else exact_source_hint
        return {
            "query_abstract": query_abstract,
            "query_pattern": query_pattern,
            "candidate_patterns": [row for row in (candidate_patterns or []) if isinstance(row, dict)],
            "patterns": rows,
            "routing_hint": routing_hint,
        }

    def should_use_subagents(self, *, candidate_patterns: List[Dict[str, Any]]) -> bool:
        if not self.subagent_enabled:
            return False
        valid = [
            row for row in (candidate_patterns or [])
            if isinstance(row, dict)
            and float(row.get("similarity", 0.0) or 0.0) >= self.match_min_score
            and float(row.get("selection_score", 0.0) or 0.0) >= max(self.match_min_score, self.subagent_min_selection_score)
        ]
        return len(valid) >= 2

    def deduplicate_patterns_for_subagents(
        self,
        *,
        candidate_patterns: List[Dict[str, Any]],
        max_branches: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        rows = [row for row in (candidate_patterns or []) if isinstance(row, dict)]
        if not rows:
            return []
        rows.sort(
            key=lambda item: (
                float(item.get("similarity", 0.0) or 0.0),
                float(item.get("selection_score", 0.0) or 0.0),
                float(item.get("success_rate", 0.0) or 0.0),
                int(item.get("support_count", 0) or 0),
            ),
            reverse=True,
        )
        limit = max(2, int(max_branches or self.subagent_max_branches or 2))
        selected: List[Dict[str, Any]] = []
        signatures: List[Dict[str, Any]] = []
        for row in rows:
            similarity = float(row.get("similarity", 0.0) or 0.0)
            selection_score = float(row.get("selection_score", 0.0) or 0.0)
            if similarity < self.match_min_score:
                continue
            if selection_score < max(self.match_min_score, self.subagent_min_selection_score):
                continue
            signature = self._subagent_signature(row)
            duplicated = False
            for prev in signatures:
                same_chain = signature["chain"] == prev["chain"] and bool(signature["chain"])
                same_problem = bool(
                    signature.get("problem_type")
                    and prev.get("problem_type")
                    and signature["problem_type"] == prev["problem_type"]
                )
                same_answer_shape = bool(
                    signature.get("answer_shape")
                    and prev.get("answer_shape")
                    and signature["answer_shape"] == prev["answer_shape"]
                )
                same_first_tool = bool(
                    signature.get("first_tool")
                    and prev.get("first_tool")
                    and signature["first_tool"] == prev["first_tool"]
                )
                overlap = token_jaccard_overlap(signature["abstract"], prev["abstract"])
                chain_overlap = self._chain_overlap_ratio(signature["chain"], prev["chain"])
                if same_chain and overlap >= 0.82:
                    duplicated = True
                    break
                if same_problem and same_answer_shape and chain_overlap >= 0.67 and overlap >= 0.62:
                    duplicated = True
                    break
                if same_problem and same_first_tool and overlap >= 0.74:
                    duplicated = True
                    break
            if duplicated:
                continue
            selected.append(row)
            signatures.append(signature)
            if len(selected) >= limit:
                break
        return selected

    def record_trace(
        self,
        *,
        query: str,
        query_abstract: str,
        tool_uses: List[Dict[str, Any]],
        final_answer: str,
        doc_type: str,
        mode: str,
        session_id: str = "",
        latency_ms: Optional[Any] = None,
    ) -> str:
        return ""

    def distill_patterns(self, max_patterns: Optional[int] = None) -> int:
        return 0

    def export_diagnostics(self, path: str = "") -> Dict[str, Any]:
        diag = self.library.diagnostics()
        diag.update(
            {
                "enabled": self.enabled,
                "read_enabled": self.read_enabled,
                "top_k_templates": self.top_k_templates,
                "max_active_patterns": self.max_active_patterns,
                "max_hint_lines": self.max_hint_lines,
                "selection_min_score": self.selection_min_score,
            }
        )
        out_path = str(path or self.report_path or "").strip()
        if out_path:
            json_dump_atomic(out_path, diag)
        return diag

    def build_sampling_branch_prior_knowledge(self, *, max_patterns: int = 8) -> str:
        self.library.maybe_reload()
        templates = [row for row in getattr(self.library, "_templates", []) if isinstance(row, dict)]
        if not templates:
            return ""

        ranked = sorted(
            templates,
            key=lambda item: (
                int(item.get("support_count", 0) or 0),
                float(item.get("success_rate", 0.0) or 0.0),
            ),
            reverse=True,
        )
        top_rows = ranked[: max(1, int(max_patterns or 1))]
        top_problem_counter: Counter[str] = Counter()
        top_chain_counter: Counter[str] = Counter()
        chain_by_problem: Dict[str, Counter[str]] = defaultdict(Counter)
        all_answer_shapes: Counter[str] = Counter()
        top_target_counter: Counter[str] = Counter()

        for row in templates:
            qp = row.get("query_pattern") if isinstance(row.get("query_pattern"), dict) else {}
            answer_shape = str(qp.get("answer_shape", "") or "").strip().lower()
            if answer_shape:
                all_answer_shapes[answer_shape] += 1

        for row in top_rows:
            qp = row.get("query_pattern") if isinstance(row.get("query_pattern"), dict) else {}
            support = max(1, int(row.get("support_count", 0) or 0))
            problem_type = str(qp.get("problem_type", "") or "").strip().lower()
            target_category = str(qp.get("target_category", "") or "").strip().lower()
            if problem_type:
                top_problem_counter[problem_type] += support
            if target_category:
                top_target_counter[target_category] += support
            chain = [str(x).strip() for x in (row.get("recommended_chain") or []) if str(x).strip()]
            if chain:
                chain_key = " -> ".join(chain[:3])
                top_chain_counter[chain_key] += support
                if problem_type:
                    chain_by_problem[problem_type][chain_key] += support

        dominant_problem_rows = top_problem_counter.most_common(4)
        dominant_chain_rows = top_chain_counter.most_common(4)
        dominant_target_rows = top_target_counter.most_common(3)
        dominant_answer_shape = all_answer_shapes.most_common(1)[0][0] if all_answer_shapes else ""

        lines: List[str] = ["Offline Strategy Priors:"]
        lines.append(f"- Library snapshot: {len(templates)} templates loaded from offline training.")
        if dominant_problem_rows:
            lines.append(
                "- Dominant problem families: "
                + ", ".join(f"{name} (weighted_support={count})" for name, count in dominant_problem_rows)
            )
        if dominant_chain_rows:
            lines.append(
                "- Dominant effective chains: "
                + ", ".join(f"{name} (weighted_support={count})" for name, count in dominant_chain_rows)
            )
        if dominant_target_rows:
            lines.append(
                "- Common target categories: "
                + ", ".join(f"{name} (weighted_support={count})" for name, count in dominant_target_rows)
            )
        if dominant_answer_shape:
            lines.append(f"- Most templates target answer_shape={dominant_answer_shape}.")

        fact_chain = self._top_chain_hint(chain_by_problem, "fact_retrieval")
        causal_chain = self._top_chain_hint(chain_by_problem, "causal_or_explanatory_lookup")
        relation_chain = self._top_chain_hint(chain_by_problem, "relation_or_interaction_lookup")
        content_chain = self._top_chain_hint(chain_by_problem, "content_span_lookup")
        attribute_chain = self._top_chain_hint(chain_by_problem, "entity_attribute_lookup")

        lines.append("- Default answer posture: most offline successes are multiple-choice option discrimination, not open-ended synthesis.")
        lines.append("- Default retrieval order when unsure: lexical doc search first, then local section evidence, then entity grounding, and only then graph or narrative-structure verification.")
        lines.append(
            self._build_problem_playbook_line(
                label="fact_retrieval",
                chain=fact_chain,
                fallback="start with bm25_search_docs or vdb_search_docs, then confirm the winning option with local textual evidence",
                caution="do not jump to broad thematic or storyline-level reasoning before checking exact option wording",
            )
        )
        lines.append(
            self._build_problem_playbook_line(
                label="entity_attribute_lookup",
                chain=attribute_chain,
                fallback="look for the explicit attribute mention in lexical search results before using entity summaries",
                caution="do not assume the central named entity is the answer until the asked attribute is directly verified",
            )
        )
        lines.append(
            self._build_problem_playbook_line(
                label="causal_or_explanatory_lookup",
                chain=causal_chain,
                fallback="prioritize direct sentences that express why, purpose, motivation, or significance",
                caution="avoid selecting an option from global theme alone when a local causal clause should decide it",
            )
        )
        lines.append(
            self._build_problem_playbook_line(
                label="relation_or_interaction_lookup",
                chain=relation_chain,
                fallback="use entity grounding only to locate evidence, then verify the actual relation or interaction in source sections",
                caution="do not equate co-mention or shared topic with the actual relationship asked by the question",
            )
        )
        lines.append(
            self._build_problem_playbook_line(
                label="content_span_lookup",
                chain=content_chain,
                fallback="go straight to section_evidence_search or search_sections for exact wording or local span evidence",
                caution="avoid broad summaries when the answer hinges on one sentence, phrase, or narration detail",
            )
        )
        lines.extend(
            [
                "- For reviewer-opinion, comparison, literary-device, exception, or improvement questions, treat them as review-evidence tasks: lexical review search plus local quoted passage beats graph-only reasoning.",
                "- Use retrieve_entity_by_name as a grounding step, not as a stopping point, especially when the question asks for judgment, comparison, stance, or option-level disambiguation.",
                "- Reserve interaction/dialogue branches for explicit speaker, addressee, stance, or interpersonal evidence questions.",
                "- Reserve narrative_hierarchical_search, storyline, or episode tools for chronology, ordering, cross-section consistency, or structure-specific questions.",
                "- When no template clearly matches, planner branches should still spend most capacity on lexical or local-evidence search; at most one branch should be graph-structure-heavy unless the question explicitly asks for structure.",
                "- Anti-patterns: multiple structure-heavy branches, entity-only answering without source verification, and replacing exact evidence with theme-level summaries.",
            ]
        )
        return "\n".join(lines)

    def clear_library(self) -> Dict[str, Any]:
        return self.library.clear(
            aggregation_mode=str(getattr(getattr(self.config, "global_config", None), "aggregation_mode", "") or ""),
            dataset_name="",
        )

    def _build_routing_hint(self, patterns: List[Dict[str, Any]]) -> str:
        return self.library.build_routing_hint(patterns, max_hint_lines=self.max_hint_lines)

    @staticmethod
    def _top_chain_hint(chain_by_problem: Dict[str, Counter[str]], problem_type: str) -> str:
        rows = chain_by_problem.get(str(problem_type or "").strip().lower())
        if not rows:
            return ""
        best = rows.most_common(1)
        return best[0][0] if best else ""

    @staticmethod
    def _build_problem_playbook_line(
        *,
        label: str,
        chain: str,
        fallback: str,
        caution: str,
    ) -> str:
        base = f"- For {label}, {fallback}."
        if chain:
            base = f"- For {label}, typical strong chain: {chain}; operationally, {fallback}."
        return f"{base} Avoid: {caution}."

    @staticmethod
    def _coarse_family_signature(row: Dict[str, Any]) -> tuple[str, str, str]:
        query_pattern = row.get("query_pattern") if isinstance(row.get("query_pattern"), dict) else {}
        return (
            str(query_pattern.get("problem_type", "") or "").strip().lower(),
            str(query_pattern.get("answer_shape", "") or "").strip().lower(),
            str(query_pattern.get("query_abstract", "") or row.get("query_abstract", "") or "").strip().lower(),
        )

    @staticmethod
    def _subagent_signature(row: Dict[str, Any]) -> Dict[str, Any]:
        query_pattern = row.get("query_pattern") if isinstance(row.get("query_pattern"), dict) else {}
        chain = [str(x).strip() for x in (row.get("recommended_chain") or []) if str(x).strip()]
        chain = RetrievalStrategyMemory._normalize_chain(chain)
        abstract = str(query_pattern.get("query_abstract", "") or row.get("query_abstract", "") or "").strip().lower()
        return {
            "chain": tuple(chain),
            "abstract": abstract,
            "problem_type": str(query_pattern.get("problem_type", "") or "").strip().lower(),
            "answer_shape": str(query_pattern.get("answer_shape", "") or "").strip().lower(),
            "first_tool": chain[0] if chain else "",
        }

    @staticmethod
    def _chain_overlap_ratio(left: tuple[str, ...], right: tuple[str, ...]) -> float:
        left_set = {str(x).strip() for x in (left or ()) if str(x).strip()}
        right_set = {str(x).strip() for x in (right or ()) if str(x).strip()}
        if not left_set or not right_set:
            return 0.0
        return len(left_set & right_set) / max(1, min(len(left_set), len(right_set)))

    @staticmethod
    def _load_source_question_hints(path: str) -> Dict[str, Any]:
        if not path:
            return {}
        if not Path(path).expanduser().exists():
            return {}
        payload = load_json(path)
        return payload if isinstance(payload, dict) else {}

    def _build_exact_source_hint(self, *, patterns: List[Dict[str, Any]]) -> str:
        payload = self.source_question_hints if isinstance(self.source_question_hints, dict) else {}
        if not payload:
            return ""
        by_template_id = payload.get("by_template_id") if isinstance(payload.get("by_template_id"), dict) else {}
        by_question = payload.get("by_question") if isinstance(payload.get("by_question"), dict) else {}
        for row in patterns or []:
            if not isinstance(row, dict):
                continue
            source_match = float(row.get("source_question_match", 0.0) or 0.0)
            abstract_overlap = float(row.get("abstract_overlap", 0.0) or 0.0)
            selection_score = float(row.get("selection_score", 0.0) or 0.0)
            quality_factor = float(row.get("quality_factor", 0.0) or 0.0)
            if quality_factor < 0.65:
                continue
            # Only activate question-specific recovery hints when the current query
            # is clearly a paraphrase of the same source question family, not just a
            # broad match to the same generic template.
            if source_match < 0.42 and not (source_match >= 0.32 and abstract_overlap >= 0.72):
                continue
            if selection_score < max(self.match_min_score, 0.72):
                continue
            template_id = str(row.get("template_id", "") or "").strip()
            best_source_question = str(row.get("best_source_question", "") or "").strip()
            hint_payload = by_template_id.get(template_id) if template_id else None
            if hint_payload is None and best_source_question:
                hint_payload = by_question.get(best_source_question)
            if not isinstance(hint_payload, dict):
                continue
            lines: List[str] = ["Source-Question Recovery Hint:"]
            retry_instruction = str(hint_payload.get("retry_instruction", "") or "").strip()
            if retry_instruction:
                lines.append(StrategyTemplateLibrary._clip_text(retry_instruction, limit=220))
            query_hints = [str(x).strip() for x in (hint_payload.get("query_hints") or []) if str(x).strip()]
            if query_hints:
                lines.append("High-yield query forms:")
                for idx, item in enumerate(query_hints[:2], start=1):
                    lines.append(f"{idx}. `{StrategyTemplateLibrary._clip_text(item, limit=160)}`")
            return "\n".join(lines)
        return ""

    @staticmethod
    def _strip_interrogatives(text: str) -> str:
        raw = str(text or "").strip()
        if not raw:
            return ""
        for old, new in [
            ("？", " "),
            ("?", " "),
            ("吗", " "),
            ("呢", " "),
            ("什么", " "),
            ("怎样", " "),
            ("哪些", " "),
            ("哪几", " "),
            ("哪一", " "),
            ("哪个", " "),
            ("哪场", " "),
            ("哪几场", " "),
            ("哪段", " "),
            ("为什么", " "),
            ("为何", " "),
            ("怎么", " "),
            ("如何", " "),
            ("分别", " "),
            ("是不是", " "),
            ("是否", " "),
        ]:
            raw = raw.replace(old, new)
        raw = re.sub(r"\s+", " ", raw).strip()
        return raw

    @classmethod
    def _build_generic_query_rewrites(cls, *, query: str, query_pattern: Dict[str, Any]) -> List[str]:
        raw = str(query or "").strip()
        if not raw:
            return []
        problem_type = cls._pattern_field(query_pattern or {}, "problem_type")
        answer_shape = cls._pattern_field(query_pattern or {}, "answer_shape")
        core = cls._strip_interrogatives(raw)
        rewrites: List[str] = []

        def add(text: str) -> None:
            value = str(text or "").strip()
            if value and value not in rewrites:
                rewrites.append(value)

        if problem_type == "causal_or_explanatory_lookup":
            add(f"{core} 原因 事件 经过")
            if any(x in raw for x in ["伤疤", "伤痕", "疤", "受伤", "创伤", "伤口"]):
                add(f"{core} 事故 撞击 创伤")
        if problem_type in {"content_span_lookup", "section_localization"}:
            add(f"{core} 原文描述")
        if any(x in raw for x in ["经常", "总是", "常常", "动作", "互动"]):
            add(f"{core} 具体动作 细节")
            add(f"{core} 重复动作 接触动作")
        if any(x in raw for x in ["西装", "制服", "穿着", "着装", "外观"]):
            add(f"{core} 场景 着装描述")
        if any(x in raw for x in ["系列", "型号", "机型", "版本", "代号"]):
            add(f"{core} 具体型号 型号列表")
            if answer_shape == "list" or "场" in raw:
                add(f"{core} 出现场次")
        if any(x in raw for x in ["坏了", "损坏", "变化", "开始", "持续多久", "持续了多久"]):
            add(f"{core} 从哪场开始 状态变化 持续多久")
            if any(x in raw for x in ["坏了", "损坏"]):
                add(f"{core} 受损 破坏 状态")
        if any(x in raw for x in ["全称", "缩写", "简称", "正式命名", "型号", "代号"]):
            add(f"{core} 正式名称 原文")

        return rewrites[:3]

    @classmethod
    def _build_generic_guidance(cls, *, query: str, query_pattern: Dict[str, Any], patterns: List[Dict[str, Any]]) -> str:
        problem_type = cls._pattern_field(query_pattern or {}, "problem_type")
        answer_shape = cls._pattern_field(query_pattern or {}, "answer_shape")
        raw = str(query or "").strip()
        lines: List[str] = []

        if problem_type in {"content_span_lookup", "causal_or_explanatory_lookup"} or any(
            x in raw for x in ["坏了", "损坏", "变化", "开始", "持续多久", "持续了多久"]
        ):
            lines.append("Pattern Guidance: Prefer BM25 or section-level evidence before broad graph browsing for exact details.")
        if problem_type == "causal_or_explanatory_lookup":
            lines.append("Pattern Guidance: Rewrite the query toward the causing event, not only the visible result or symptom.")
        if problem_type == "section_localization" or any(x in raw for x in ["场次", "场景", "章节"]):
            lines.append("Pattern Guidance: Keep only sections directly supported by the raw text; do not include nearby or inferred sections.")
        if any(x in raw for x in ["经常", "总是", "常常", "动作", "互动"]):
            lines.append("Pattern Guidance: For repeated-action questions, search exact action wording first; abstract relation summaries are secondary.")
        if any(x in raw for x in ["系列", "型号", "机型", "版本", "代号"]):
            lines.append("Pattern Guidance: Enumerate concrete member names first, then gather sections for each member separately.")
        if any(x in raw for x in ["全称", "缩写", "简称", "正式命名", "型号", "代号"]):
            lines.append("Pattern Guidance: Prefer lexical retrieval for exact names, codes, abbreviations, and model identifiers.")
        if answer_shape == "list":
            lines.append("Pattern Guidance: When returning a list, filter aggressively and keep only evidence-backed items.")

        rewrites = cls._build_generic_query_rewrites(query=query, query_pattern=query_pattern)
        if rewrites:
            lines.append("Possible lexical rewrites:")
            for idx, item in enumerate(rewrites, start=1):
                lines.append(f"{idx}. `{item}`")

        if not lines:
            return ""
        return "\n".join(lines)

    @staticmethod
    def _quality_weight(row: Dict[str, Any]) -> float:
        support = max(0, int(row.get("support_count", 0) or 0))
        support_factor = max(0.0, min(1.0, support / 5.0))
        if StrategyTemplateLibrary._score_supervision_mode(row) != "gt":
            return 0.2 + 0.8 * support_factor
        success_rate = StrategyTemplateLibrary.effective_success_rate_for_matching(row)
        return 0.2 + 0.4 * success_rate + 0.4 * support_factor

    @staticmethod
    def _has_any_tool(chain: List[str], names: List[str]) -> bool:
        chain_set = {str(x).strip() for x in (chain or []) if str(x).strip()}
        return any(name in chain_set for name in names)

    @staticmethod
    def _normalize_chain(chain: List[str]) -> List[str]:
        normalized: List[str] = []
        for item in chain or []:
            raw = str(item or "").replace("->", ",")
            for part in raw.split(","):
                tool_name = str(part or "").strip()
                if not tool_name:
                    continue
                tool_name = _CHAIN_TOOL_ALIASES.get(tool_name, tool_name)
                if tool_name not in normalized:
                    normalized.append(tool_name)
        return normalized

    @staticmethod
    def _pattern_field(pattern: Dict[str, Any], field: str) -> str:
        return str((pattern or {}).get(field, "") or "").strip().lower()

    @staticmethod
    def _detect_focus_labels(text: str) -> List[str]:
        raw = str(text or "").strip().lower()
        if not raw:
            return []
        labels: List[str] = []
        rules = {
            "full_name": ["全称", "简称", "缩写", "full name", "formal name"],
            "age": ["几岁", "多少岁", "年龄", "age", "岁"],
            "scene_enumeration": ["场次", "场景", "哪场", "哪几场", "scene", "chapter", "出现"],
            "appearance": ["穿了", "穿着", "西装", "制服", "外观", "长相", "面部", "头发"],
            "source_lookup": ["从哪里来", "哪来的", "来源", "谁给的", "递给", "哪里来的"],
            "state_change": ["开始坏", "损坏", "坏了", "变化", "状态", "持续多久", "何时开始"],
            "interaction": ["经常", "总是", "常常", "动作", "互动", "关系"],
            "dialogue": ["说了什么", "台词", "对白", "原话", "写了什么", "关键话语"],
            "value_lookup": ["多少钱", "价格", "房价", "单价", "一平", "数值"],
        }
        for label, keywords in rules.items():
            if any(keyword in raw for keyword in keywords):
                labels.append(label)
        return labels

    @classmethod
    def _has_cross_intent_conflict(cls, query_pattern: Dict[str, Any], template_pattern: Dict[str, Any]) -> bool:
        query_problem = cls._pattern_field(query_pattern or {}, "problem_type")
        template_problem = cls._pattern_field(template_pattern or {}, "problem_type")
        if not query_problem or not template_problem or query_problem == template_problem:
            return False
        incompatible_pairs = {
            frozenset({"content_span_lookup", "section_localization"}),
            frozenset({"content_span_lookup", "entity_attribute_lookup"}),
            frozenset({"content_span_lookup", "relation_or_interaction_lookup"}),
            frozenset({"content_span_lookup", "causal_or_explanatory_lookup"}),
            frozenset({"section_localization", "causal_or_explanatory_lookup"}),
        }
        return frozenset({query_problem, template_problem}) in incompatible_pairs

    @classmethod
    def _is_pattern_compatible(cls, *, query_pattern: Dict[str, Any], template_pattern: Dict[str, Any]) -> bool:
        if cls._has_cross_intent_conflict(query_pattern, template_pattern):
            return False
        query_shape = cls._pattern_field(query_pattern or {}, "answer_shape")
        template_shape = cls._pattern_field(template_pattern or {}, "answer_shape")
        if query_shape and template_shape and query_shape != template_shape:
            query_problem = cls._pattern_field(query_pattern or {}, "problem_type")
            template_problem = cls._pattern_field(template_pattern or {}, "problem_type")
            if query_problem and template_problem and query_problem != template_problem:
                return False
        query_focus = set(
            cls._detect_focus_labels(
                " ".join(
                    [
                        str((query_pattern or {}).get("query_abstract", "") or ""),
                        " ".join(str(x or "") for x in ((query_pattern or {}).get("retrieval_goals") or [])),
                    ]
                )
            )
        )
        template_focus = set(
            cls._detect_focus_labels(
                " ".join(
                    [
                        str((template_pattern or {}).get("query_abstract", "") or ""),
                        " ".join(str(x or "") for x in ((template_pattern or {}).get("retrieval_goals") or [])),
                    ]
                )
            )
        )
        strong_labels = {"full_name", "age", "appearance", "source_lookup", "state_change", "interaction", "dialogue", "value_lookup"}
        query_strong = query_focus.intersection(strong_labels)
        template_strong = template_focus.intersection(strong_labels)
        if query_strong and template_strong and query_strong.isdisjoint(template_strong):
            return False
        return True

    def _is_chain_compatible(self, *, query: str, query_pattern: Dict[str, Any], chain: List[str]) -> bool:
        chain = self._normalize_chain(chain)
        text = str(query or "").strip().lower()
        pattern = query_pattern or {}
        answer_shape = str(pattern.get("answer_shape", "") or "").strip().lower()
        query_abstract = str(pattern.get("query_abstract", "") or "").strip().lower()

        if any(x in text for x in ["全称", "简称", "缩写", "formal name", "full name"]):
            return self._has_any_tool(chain, ["bm25_search_docs", "section_evidence_search"])

        if any(x in text for x in ["几岁", "多少岁", "年龄", "年份", "几年", "何时", "多久", "从哪场开始", "什么时候", "岁"]):
            return self._has_any_tool(chain, ["bm25_search_docs", "section_evidence_search"])

        if any(x in text for x in ["场次", "场景", "哪场", "哪几场", "scene", "chapter"]) or answer_shape == "list":
            if not self._has_any_tool(
                chain,
                [
                    "get_entity_sections",
                    "search_sections",
                    "lookup_titles_by_document_ids",
                    "section_evidence_search",
                    "bm25_search_docs",
                    "vdb_get_docs_by_document_ids",
                ],
            ):
                return False

        if any(x in text for x in ["说了什么", "台词", "对白", "原话", "临终", "死之前"]):
            return self._has_any_tool(chain, ["bm25_search_docs", "section_evidence_search"])

        if any(x in text for x in ["经常", "总是", "常常", "动作", "互动", "关系"]):
            return self._has_any_tool(chain, ["search_related_entities", "section_evidence_search", "bm25_search_docs"])

        if any(x in text for x in ["穿了", "穿着", "西装", "制服", "回了", "回到", "去了", "返回"]):
            return self._has_any_tool(chain, ["section_evidence_search", "get_entity_sections", "search_sections"])

        if any(ch.isdigit() for ch in text) and "场" in text:
            return self._has_any_tool(
                chain,
                [
                    "lookup_document_ids_by_title",
                    "search_sections",
                    "section_evidence_search",
                    "bm25_search_docs",
                    "vdb_get_docs_by_document_ids",
                    "get_entity_sections",
                ],
            )

        if "query_abstract" in pattern and any(x in query_abstract for x in ["全称", "命名", "关键话语", "年龄", "场景", "场次"]):
            return self._has_any_tool(chain, ["bm25_search_docs", "section_evidence_search", "get_entity_sections", "search_sections"])

        return True

    def _select_active_patterns(
        self,
        *,
        query: str,
        query_pattern: Dict[str, Any],
        candidate_patterns: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        rows = [row for row in (candidate_patterns or []) if isinstance(row, dict)]
        if not rows:
            return []
        scored_rows: List[Dict[str, Any]] = []
        for row in rows:
            base_similarity = float(row.get("similarity", 0.0) or 0.0)
            quality_weight = self._quality_weight(row)
            row["quality_weight"] = round(quality_weight, 6)
            row["selection_score"] = round(base_similarity * quality_weight, 6)
            scored_rows.append(row)
        scored_rows.sort(
            key=lambda item: (
                float(item.get("selection_score", 0.0) or 0.0),
                float(item.get("similarity", 0.0) or 0.0),
                float(StrategyTemplateLibrary.effective_success_rate_for_matching(item)),
                int(item.get("support_count", 0) or 0),
            ),
            reverse=True,
        )
        top_selected_score: Optional[float] = None
        selected: List[Dict[str, Any]] = []
        for row in scored_rows:
            similarity = float(row.get("similarity", 0.0) or 0.0)
            score = float(row.get("selection_score", 0.0) or 0.0)
            row["selection_status"] = "rejected"
            row["selection_reason"] = "unknown"
            row["selection_threshold"] = self.match_min_score
            row["selection_score_threshold"] = self.single_agent_min_selection_score
            row["max_active_patterns"] = self.max_active_patterns
            template_pattern = row.get("query_pattern") if isinstance(row.get("query_pattern"), dict) else {}
            effective_threshold = self._effective_match_threshold(
                row=row,
                query_pattern=query_pattern,
                template_pattern=template_pattern,
            )
            row["selection_threshold"] = round(effective_threshold, 6)
            if similarity < effective_threshold:
                row["selection_reason"] = "similarity_below_threshold"
                continue
            if score < self.single_agent_min_selection_score:
                row["selection_reason"] = "selection_score_below_threshold"
                continue
            if selected and float(selected[0].get("similarity", 0.0) or 0.0) >= max(self.match_min_score, 0.7):
                top_score = float(selected[0].get("similarity", 0.0) or 0.0)
                if top_score - similarity >= 0.12:
                    row["selection_reason"] = "margin_too_small_against_top_pattern"
                    continue
            pattern_compatible = self._is_pattern_compatible(query_pattern=query_pattern, template_pattern=template_pattern)
            row["pattern_compatible"] = pattern_compatible
            if not pattern_compatible:
                row["selection_reason"] = "pattern_incompatible"
                continue
            chain = self._normalize_chain([str(x).strip() for x in (row.get("recommended_chain") or []) if str(x).strip()])
            row["recommended_chain"] = chain
            chain_compatible = True
            if chain:
                chain_compatible = self._is_chain_compatible(query=query, query_pattern=query_pattern, chain=chain)
            row["chain_compatible"] = chain_compatible
            if chain and not chain_compatible:
                row["selection_reason"] = "chain_incompatible"
                continue
            if selected and top_selected_score is not None:
                top_pattern = selected[0].get("query_pattern") if isinstance(selected[0].get("query_pattern"), dict) else {}
                current_problem = self._pattern_field(template_pattern, "problem_type")
                top_problem = self._pattern_field(top_pattern, "problem_type")
                current_chain = [str(x).strip() for x in (row.get("recommended_chain") or []) if str(x).strip()]
                top_chain = [str(x).strip() for x in (selected[0].get("recommended_chain") or []) if str(x).strip()]
                same_problem = bool(current_problem and top_problem and current_problem == top_problem)
                same_chain = bool(current_chain and top_chain and current_chain == top_chain)
                score_gap = top_selected_score - score
                row["score_gap_from_top_selected"] = round(score_gap, 6)
                if same_problem and same_chain and score_gap >= 0.12:
                    row["selection_reason"] = "dominated_by_higher_ranked_same_family_pattern"
                    continue
                top_signature = self._coarse_family_signature(selected[0])
                current_signature = self._coarse_family_signature(row)
                if top_signature[0] and current_signature[0] and top_signature[0] == current_signature[0]:
                    overlap = token_jaccard_overlap(top_signature[2], current_signature[2])
                    if overlap >= 0.75 and score_gap >= 0.05:
                        row["selection_reason"] = "deduplicated_same_pattern_family"
                        continue
            if len(selected) >= self.max_active_patterns:
                row["selection_reason"] = "max_active_patterns_reached"
                continue
            row["selection_status"] = "selected"
            row["selection_reason"] = "accepted"
            selected.append(row)
            if top_selected_score is None:
                top_selected_score = score
        return selected

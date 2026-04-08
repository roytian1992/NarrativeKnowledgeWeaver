from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

from core.functions.memory_management.strategy_cluster_distiller import StrategyClusterDistiller
from core.functions.memory_management.strategy_query_pattern import (
    StrategyQueryPatternExtractor,
    normalize_answer_shape,
    normalize_problem_type,
    normalize_target_category,
    sanitize_retrieval_goals,
)
from core.functions.memory_management.strategy_template_merge_decider import StrategyTemplateMergeDecider
from core.utils.general_utils import clamp_float, cosine_sim, token_jaccard_overlap


class StrategyTemplateClusterManager:
    def __init__(
        self,
        *,
        prompt_loader,
        llm,
        embedding_model: Any = None,
        candidate_top_k: int = 3,
        min_candidate_score: float = 0.45,
        consolidation_rounds: int = 1,
        max_members_for_distill_prompt: int = 12,
    ) -> None:
        self.prompt_loader = prompt_loader
        self.llm = llm
        self.embedding_model = embedding_model
        self.candidate_top_k = max(1, int(candidate_top_k or 1))
        self.min_candidate_score = float(min_candidate_score or 0.0)
        self.consolidation_rounds = max(0, int(consolidation_rounds or 0))
        self.max_members_for_distill_prompt = max(1, int(max_members_for_distill_prompt or 1))

        self.merge_decider = StrategyTemplateMergeDecider(prompt_loader=self.prompt_loader, llm=self.llm)
        self.cluster_distiller = StrategyClusterDistiller(prompt_loader=self.prompt_loader, llm=self.llm)

        self.raw_templates: List[Dict[str, Any]] = []
        self.clusters: List[Dict[str, Any]] = []
        self.merge_decisions: List[Dict[str, Any]] = []
        self._cluster_seq = 0
        self._decision_seq = 0
        self._dirty_cluster_ids: set[str] = set()

    @staticmethod
    def _sanitize_text(value: Any) -> str:
        return str(value or "").strip()

    @staticmethod
    def _sanitize_list(items: Any, *, limit: int = 12) -> List[str]:
        if not isinstance(items, list):
            return []
        out: List[str] = []
        for item in items:
            value = str(item or "").strip()
            if value and value not in out:
                out.append(value)
            if len(out) >= limit:
                break
        return out

    @staticmethod
    def _text_overlap_score(a: str, b: str) -> float:
        return token_jaccard_overlap(a, b)

    @staticmethod
    def _pattern_match_text(pattern: Dict[str, Any], *, query_abstract: str = "") -> str:
        pattern = pattern or {}
        abstract_text = str(query_abstract or pattern.get("query_abstract", "") or "").strip()
        goals = [str(x).strip() for x in (pattern.get("retrieval_goals") or []) if str(x).strip()]
        parts = [abstract_text]
        if goals:
            parts.append(" ; ".join(goals))
        return " ; ".join(x for x in parts if x)

    @staticmethod
    def _safe_embedding(value: Any) -> Optional[List[float]]:
        if not isinstance(value, list):
            return None
        out: List[float] = []
        for item in value:
            try:
                out.append(float(item))
            except Exception:
                return None
        return out or None

    def _embed_text(self, text: str) -> Optional[List[float]]:
        if self.embedding_model is None or not hasattr(self.embedding_model, "embed_query"):
            return None
        try:
            emb = self.embedding_model.embed_query(text)
        except Exception:
            return None
        return self._safe_embedding(emb)

    @staticmethod
    def _goal_overlap(a: List[str], b: List[str]) -> float:
        sa = {str(x).strip() for x in (a or []) if str(x).strip()}
        sb = {str(x).strip() for x in (b or []) if str(x).strip()}
        if not sa or not sb:
            return 0.0
        return float(len(sa & sb)) / float(len(sa | sb))

    @staticmethod
    def _normalize_chain(chain: List[str]) -> List[str]:
        return [str(x).strip() for x in (chain or []) if str(x).strip()]

    @classmethod
    def _chain_overlap(cls, a: List[str], b: List[str]) -> float:
        seq_a = cls._normalize_chain(a)
        seq_b = cls._normalize_chain(b)
        if not seq_a or not seq_b:
            return 0.0
        if seq_a == seq_b:
            return 1.0

        set_score = float(len(set(seq_a) & set(seq_b))) / float(len(set(seq_a) | set(seq_b)))
        prefix_len = 0
        for left, right in zip(seq_a, seq_b):
            if left != right:
                break
            prefix_len += 1
        prefix_score = float(prefix_len) / float(max(len(seq_a), len(seq_b)))

        lcs = [[0] * (len(seq_b) + 1) for _ in range(len(seq_a) + 1)]
        for i, left in enumerate(seq_a, start=1):
            for j, right in enumerate(seq_b, start=1):
                if left == right:
                    lcs[i][j] = lcs[i - 1][j - 1] + 1
                else:
                    lcs[i][j] = max(lcs[i - 1][j], lcs[i][j - 1])
        lcs_score = float(lcs[-1][-1]) / float(max(len(seq_a), len(seq_b)))

        boundary_score = 0.0
        if seq_a[0] == seq_b[0]:
            boundary_score += 0.5
        if seq_a[-1] == seq_b[-1]:
            boundary_score += 0.5

        shorter, longer = (seq_a, seq_b) if len(seq_a) <= len(seq_b) else (seq_b, seq_a)
        subseq_index = 0
        for step in longer:
            if subseq_index < len(shorter) and step == shorter[subseq_index]:
                subseq_index += 1
        subseq_score = 1.0 if subseq_index == len(shorter) else 0.0

        score = (
            0.28 * set_score
            + 0.24 * prefix_score
            + 0.24 * lcs_score
            + 0.14 * boundary_score
            + 0.10 * subseq_score
        )
        return round(clamp_float(score, low=0.0, high=1.0, default=0.0), 6)

    @staticmethod
    def _pattern_field(pattern_container: Dict[str, Any], field: str) -> str:
        pattern = pattern_container.get("query_pattern") if isinstance(pattern_container.get("query_pattern"), dict) else {}
        if not pattern:
            pattern = pattern_container.get("query_pattern_prototype") if isinstance(pattern_container.get("query_pattern_prototype"), dict) else {}
        return str(pattern.get(field, "") or "").strip().lower()

    @classmethod
    def _problem_type_match(cls, a: Dict[str, Any], b: Dict[str, Any]) -> float:
        a_problem = cls._pattern_field(a, "problem_type")
        b_problem = cls._pattern_field(b, "problem_type")
        if not a_problem or not b_problem:
            return 0.0
        return 1.0 if a_problem == b_problem else 0.0

    @classmethod
    def _has_cross_intent_conflict(cls, a: Dict[str, Any], b: Dict[str, Any]) -> bool:
        a_problem = cls._pattern_field(a, "problem_type")
        b_problem = cls._pattern_field(b, "problem_type")
        if not a_problem or not b_problem or a_problem == b_problem:
            return False
        incompatible_pairs = {
            frozenset({"content_span_lookup", "section_localization"}),
            frozenset({"content_span_lookup", "entity_attribute_lookup"}),
            frozenset({"content_span_lookup", "relation_or_interaction_lookup"}),
            frozenset({"content_span_lookup", "causal_explanation"}),
            frozenset({"section_localization", "causal_explanation"}),
        }
        return frozenset({a_problem, b_problem}) in incompatible_pairs

    @classmethod
    def _merge_compatibility(
        cls,
        incoming: Dict[str, Any],
        candidate: Dict[str, Any],
        *,
        stage: str,
    ) -> tuple[bool, str]:
        incoming_problem = cls._pattern_field(incoming, "problem_type")
        candidate_problem = cls._pattern_field(candidate, "problem_type")
        incoming_shape = cls._pattern_field(incoming, "answer_shape")
        candidate_shape = cls._pattern_field(candidate, "answer_shape")
        incoming_target = cls._pattern_field(incoming, "target_category")
        candidate_target = cls._pattern_field(candidate, "target_category")
        problem_match = 1.0 if incoming_problem and candidate_problem and incoming_problem == candidate_problem else 0.0
        shape_match = 1.0 if incoming_shape and candidate_shape and incoming_shape == candidate_shape else 0.0
        target_match = 1.0 if incoming_target and candidate_target and incoming_target == candidate_target else 0.0

        if cls._has_cross_intent_conflict(incoming, candidate):
            return False, "cross_intent_problem_type_conflict"
        if incoming_problem and candidate_problem and incoming_problem != candidate_problem:
            return False, "requires_same_problem_type"
        if incoming_shape and candidate_shape and incoming_shape != candidate_shape:
            return False, "requires_same_answer_shape"
        if problem_match < 1.0:
            return False, "requires_same_problem_type"
        if shape_match < 1.0:
            return False, "requires_same_answer_shape"
        if stage == "incremental" and incoming_target and candidate_target and target_match == 0.0:
            return True, "soft_target_mismatch_allowed"
        if stage == "global_consolidation" and incoming_target and candidate_target and target_match == 0.0:
            return True, "soft_target_mismatch_allowed"
        return True, ""

    @classmethod
    def _candidate_eligible(
        cls,
        incoming: Dict[str, Any],
        candidate: Dict[str, Any],
        *,
        stage: str,
    ) -> bool:
        compatible, _ = cls._merge_compatibility(incoming, candidate, stage=stage)
        return compatible

    @staticmethod
    def _answer_shape_match(a: Dict[str, Any], b: Dict[str, Any]) -> float:
        a_pattern = a.get("query_pattern") if isinstance(a.get("query_pattern"), dict) else {}
        b_pattern = (
            b.get("query_pattern_prototype")
            if isinstance(b.get("query_pattern_prototype"), dict)
            else (b.get("query_pattern") if isinstance(b.get("query_pattern"), dict) else {})
        )
        a_shape = str(a_pattern.get("answer_shape", "") or "").strip().lower()
        b_shape = str(b_pattern.get("answer_shape", "") or "").strip().lower()
        if not a_shape or not b_shape:
            return 0.0
        return 1.0 if a_shape == b_shape else 0.0

    def _new_cluster_id(self) -> str:
        self._cluster_seq += 1
        return f"stc_{self._cluster_seq:04d}"

    def _new_decision_id(self) -> str:
        self._decision_seq += 1
        return f"smd_{self._decision_seq:05d}"

    def _estimate_attempt_count(self, template: Dict[str, Any]) -> int:
        if int(template.get("attempt_count", 0) or 0) > 0:
            return int(template.get("attempt_count", 0) or 0)
        support_count = max(0, int(template.get("support_count", 0) or 0))
        success_rate = float(template.get("success_rate", 0.0) or 0.0)
        if support_count > 0 and success_rate > 0.0:
            try:
                estimated = int(round(float(support_count) / float(success_rate)))
                if estimated > 0:
                    return estimated
            except Exception:
                pass
        return max(1, support_count)

    @staticmethod
    def _score_supervision_mode(item: Dict[str, Any]) -> str:
        mode = str(item.get("score_supervision", "gt") or "gt").strip().lower()
        return "none" if mode in {"none", "no_gt", "unsupervised", "online_unsupervised"} else "gt"

    @classmethod
    def _matching_success_rate(cls, item: Dict[str, Any]) -> float:
        if cls._score_supervision_mode(item) != "gt":
            return 0.0
        return clamp_float(item.get("success_rate", 0.0), low=0.0, high=1.0, default=0.0)

    def sanitize_template(self, item: Dict[str, Any], *, default_id: str = "") -> Dict[str, Any]:
        query_pattern = item.get("query_pattern") if isinstance(item.get("query_pattern"), dict) else {}
        query_abstract = self._sanitize_text(item.get("query_abstract") or query_pattern.get("query_abstract"))
        query_pattern_text = self._sanitize_text(item.get("query_pattern_text"))
        if not query_pattern_text:
            query_pattern_text = StrategyQueryPatternExtractor.pattern_to_text({**query_pattern, "query_abstract": query_abstract})
        raw_tool_chain = self._sanitize_list(item.get("raw_tool_chain"), limit=12)
        minimal_effective_chain = self._sanitize_list(
            item.get("minimal_effective_chain") or item.get("recommended_chain"),
            limit=8,
        )
        if not minimal_effective_chain:
            minimal_effective_chain = self._sanitize_list(item.get("effective_tool_chain"), limit=8)
        recommended_chain = minimal_effective_chain or self._sanitize_list(item.get("recommended_chain"), limit=8)
        support_count = max(0, int(item.get("support_count", item.get("successful_attempts", 0)) or 0))
        attempt_count = self._estimate_attempt_count({**item, "support_count": support_count})
        success_rate = clamp_float(
            item.get("success_rate", float(support_count) / float(attempt_count) if attempt_count > 0 else 0.0),
            low=0.0,
            high=1.0,
            default=0.0,
        )
        sanitized = {
            "template_id": self._sanitize_text(item.get("template_id") or default_id or f"template_{len(self.raw_templates) + 1}"),
            "question_id": self._sanitize_text(item.get("question_id")),
            "question": self._sanitize_text(item.get("question")),
            "pattern_name": self._sanitize_text(item.get("pattern_name")) or "Generic Retrieval Pattern",
            "pattern_description": self._sanitize_text(item.get("pattern_description")),
            "recommended_chain": recommended_chain,
            "raw_tool_chain": raw_tool_chain,
            "minimal_effective_chain": minimal_effective_chain or recommended_chain,
            "anti_patterns": self._sanitize_list(item.get("anti_patterns"), limit=12),
            "chain_rationale": self._sanitize_text(item.get("chain_rationale")),
            "chain_constraints": self._sanitize_list(item.get("chain_constraints"), limit=8),
            "support_count": support_count,
            "successful_attempts": max(0, int(item.get("successful_attempts", support_count) or support_count)),
            "attempt_count": attempt_count,
            "success_rate": success_rate,
            "score_supervision": self._score_supervision_mode(item),
            "query_pattern": {
                "query_abstract": query_abstract,
                "problem_type": normalize_problem_type(query_pattern.get("problem_type")),
                "target_category": normalize_target_category(query_pattern.get("target_category")),
                "answer_shape": normalize_answer_shape(query_pattern.get("answer_shape")),
                "retrieval_goals": sanitize_retrieval_goals(
                    query_pattern.get("retrieval_goals"),
                    query=str(item.get("question", "") or ""),
                    problem_type=normalize_problem_type(query_pattern.get("problem_type")),
                    answer_shape=normalize_answer_shape(query_pattern.get("answer_shape")),
                ),
                "notes": self._sanitize_text(query_pattern.get("notes")),
            },
            "query_abstract": query_abstract,
            "query_pattern_text": query_pattern_text,
            "pattern_embedding": self._safe_embedding(item.get("pattern_embedding")),
            "template_sources": list(item.get("template_sources") or []),
            "chain_token_cost": max(0, int(item.get("chain_token_cost", 0) or 0)),
            "avg_latency_ms": max(0.0, float(item.get("avg_latency_ms", item.get("latency_ms", 0.0)) or 0.0)),
            "intermediate_value_score": clamp_float(item.get("intermediate_value_score", success_rate), low=0.0, high=1.0, default=success_rate),
            "source_attempt_ids": [self._sanitize_text(x) for x in (item.get("source_attempt_ids") or []) if self._sanitize_text(x)],
        }
        if sanitized["pattern_embedding"] is None:
            sanitized["pattern_embedding"] = self._embed_text(query_pattern_text)
        return sanitized

    def _template_summary_for_llm(self, template: Dict[str, Any]) -> Dict[str, Any]:
        query_pattern = template.get("query_pattern") if isinstance(template.get("query_pattern"), dict) else {}
        return {
            "template_id": self._sanitize_text(template.get("template_id")),
            "pattern_name": self._sanitize_text(template.get("pattern_name")),
            "pattern_description": self._sanitize_text(template.get("pattern_description")),
            "recommended_chain": self._sanitize_list(template.get("recommended_chain"), limit=8),
            "minimal_effective_chain": self._sanitize_list(template.get("minimal_effective_chain"), limit=8),
            "anti_patterns": self._sanitize_list(template.get("anti_patterns"), limit=8),
            "chain_rationale": self._sanitize_text(template.get("chain_rationale")),
            "chain_constraints": self._sanitize_list(template.get("chain_constraints"), limit=8),
            "support_count": max(0, int(template.get("support_count", 0) or 0)),
            "success_rate": clamp_float(template.get("success_rate", 0.0), low=0.0, high=1.0, default=0.0),
            "score_supervision": self._score_supervision_mode(template),
            "query_pattern": {
                "query_abstract": self._sanitize_text(query_pattern.get("query_abstract")),
                "problem_type": self._sanitize_text(query_pattern.get("problem_type")),
                "target_category": self._sanitize_text(query_pattern.get("target_category")),
                "answer_shape": self._sanitize_text(query_pattern.get("answer_shape")),
                "retrieval_goals": self._sanitize_list(query_pattern.get("retrieval_goals"), limit=6),
                "notes": self._sanitize_text(query_pattern.get("notes")),
            },
        }

    def _cluster_template_view(self, cluster: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "template_id": self._sanitize_text(cluster.get("cluster_id")),
            "pattern_name": self._sanitize_text(cluster.get("pattern_name")),
            "pattern_description": self._sanitize_text(cluster.get("pattern_description")),
            "recommended_chain": self._sanitize_list(cluster.get("recommended_chain"), limit=8),
            "chain_variants": list(cluster.get("chain_variants") or [])[:4],
            "anti_patterns": self._sanitize_list(cluster.get("anti_patterns"), limit=12),
            "support_count": max(0, int(cluster.get("support_count", 0) or 0)),
            "successful_attempts": max(0, int(cluster.get("successful_attempts", 0) or 0)),
            "attempt_count": max(1, int(cluster.get("attempt_count", 1) or 1)),
            "success_rate": clamp_float(cluster.get("success_rate", 0.0), low=0.0, high=1.0, default=0.0),
            "query_pattern": cluster.get("query_pattern_prototype") if isinstance(cluster.get("query_pattern_prototype"), dict) else {},
            "query_abstract": self._sanitize_text(cluster.get("query_abstract")),
            "query_pattern_text": self._sanitize_text(cluster.get("query_pattern_text")),
            "pattern_embedding": self._safe_embedding(cluster.get("pattern_embedding")),
        }

    def _template_cluster_view(self, template: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "cluster_id": self._sanitize_text(template.get("template_id")),
            "pattern_name": self._sanitize_text(template.get("pattern_name")),
            "pattern_description": self._sanitize_text(template.get("pattern_description")),
            "recommended_chain": self._sanitize_list(template.get("recommended_chain"), limit=8),
            "minimal_effective_chain": self._sanitize_list(template.get("minimal_effective_chain"), limit=8),
            "anti_patterns": self._sanitize_list(template.get("anti_patterns"), limit=12),
            "query_pattern_prototype": template.get("query_pattern") if isinstance(template.get("query_pattern"), dict) else {},
            "query_abstract": self._sanitize_text(template.get("query_abstract")),
            "query_pattern_text": self._sanitize_text(template.get("query_pattern_text")),
            "pattern_embedding": self._safe_embedding(template.get("pattern_embedding")),
            "support_count": max(0, int(template.get("support_count", 0) or 0)),
            "successful_attempts": max(0, int(template.get("successful_attempts", 0) or 0)),
            "attempt_count": max(1, int(template.get("attempt_count", 1) or 1)),
            "success_rate": clamp_float(template.get("success_rate", 0.0), low=0.0, high=1.0, default=0.0),
            "score_supervision": self._score_supervision_mode(template),
        }

    def _cluster_statistics(self, member_templates: List[Dict[str, Any]]) -> Dict[str, Any]:
        chain_weights: Dict[Tuple[str, ...], int] = {}
        chain_variant_stats: Dict[Tuple[str, ...], Dict[str, Any]] = {}
        problem_type_weights: Dict[str, int] = {}
        target_weights: Dict[str, int] = {}
        answer_shape_weights: Dict[str, int] = {}
        retrieval_goal_weights: Dict[str, int] = {}
        support_total = 0
        attempts_total = 0
        for member in member_templates:
            weight = max(1, int(member.get("support_count", 1) or 1))
            support_total += max(0, int(member.get("successful_attempts", member.get("support_count", 0)) or 0))
            attempts_total += max(1, int(member.get("attempt_count", 1) or 1))
            chain_key = tuple(
                self._sanitize_list(member.get("minimal_effective_chain") or member.get("recommended_chain"), limit=8)
            )
            if chain_key:
                chain_weights[chain_key] = chain_weights.get(chain_key, 0) + weight
                variant = chain_variant_stats.setdefault(
                    chain_key,
                    {
                        "recommended_chain": list(chain_key),
                        "tool_count": len(chain_key),
                        "chain_token_cost_total": 0,
                        "support_count": 0,
                        "attempt_count": 0,
                        "success_total": 0.0,
                        "intermediate_value_score_total": 0.0,
                        "latency_total_ms": 0.0,
                        "source_attempt_ids": [],
                        "source_template_ids": [],
                    },
                )
                variant["support_count"] += weight
                variant["attempt_count"] += max(1, int(member.get("attempt_count", 1) or 1))
                variant["success_total"] += float(member.get("success_rate", 0.0) or 0.0) * float(weight)
                variant["intermediate_value_score_total"] += float(member.get("intermediate_value_score", 0.0) or 0.0) * float(weight)
                variant["latency_total_ms"] += float(member.get("avg_latency_ms", 0.0) or 0.0) * float(weight)
                variant["chain_token_cost_total"] += max(0, int(member.get("chain_token_cost", 0) or 0)) * max(1, weight)
                template_id = self._sanitize_text(member.get("template_id"))
                if template_id and template_id not in variant["source_template_ids"]:
                    variant["source_template_ids"].append(template_id)
                for attempt_id in member.get("source_attempt_ids") or []:
                    safe_attempt_id = self._sanitize_text(attempt_id)
                    if safe_attempt_id and safe_attempt_id not in variant["source_attempt_ids"]:
                        variant["source_attempt_ids"].append(safe_attempt_id)
            qp = member.get("query_pattern") if isinstance(member.get("query_pattern"), dict) else {}
            for field, bucket in (("problem_type", problem_type_weights), ("target_category", target_weights), ("answer_shape", answer_shape_weights)):
                value = self._sanitize_text(qp.get(field))
                if value:
                    bucket[value] = bucket.get(value, 0) + weight
            for goal in self._sanitize_list(qp.get("retrieval_goals"), limit=6):
                retrieval_goal_weights[goal] = retrieval_goal_weights.get(goal, 0) + weight
        top_chains = [
            {"chain": list(chain), "count": count}
            for chain, count in sorted(chain_weights.items(), key=lambda item: (item[1], -len(item[0]), item[0]), reverse=True)[:6]
        ]
        chain_variants: List[Dict[str, Any]] = []
        for idx, (chain_key, stats) in enumerate(
            sorted(
                chain_variant_stats.items(),
                key=lambda item: (
                    float(item[1].get("success_total", 0.0) or 0.0) / float(max(1, int(item[1].get("support_count", 1) or 1))),
                    int(item[1].get("support_count", 0) or 0),
                    -len(item[0]),
                ),
                reverse=True,
            ),
            start=1,
        ):
            support_count = max(1, int(stats.get("support_count", 1) or 1))
            chain_variants.append(
                {
                    "variant_id": f"variant_{idx:03d}",
                    "recommended_chain": list(chain_key),
                    "tool_count": len(chain_key),
                    "chain_token_cost": round(float(stats.get("chain_token_cost_total", 0.0) or 0.0) / float(support_count), 2),
                    "support_count": support_count,
                    "success_rate": round(float(stats.get("success_total", 0.0) or 0.0) / float(support_count), 4),
                    "intermediate_value_score": round(float(stats.get("intermediate_value_score_total", 0.0) or 0.0) / float(support_count), 4),
                    "avg_latency_ms": round(float(stats.get("latency_total_ms", 0.0) or 0.0) / float(support_count), 2),
                    "source_attempt_ids": list(stats.get("source_attempt_ids") or [])[:32],
                    "source_template_ids": list(stats.get("source_template_ids") or [])[:32],
                }
            )
        answer_shape_votes = sorted(answer_shape_weights.items(), key=lambda item: item[1], reverse=True)
        return {
            "template_count": len(member_templates),
            "support_count": support_total,
            "attempt_count": attempts_total,
            "success_rate": round(float(support_total) / float(attempts_total), 4) if attempts_total > 0 else 0.0,
            "top_chains": top_chains,
            "chain_variants": chain_variants[:6],
            "problem_type_votes": sorted(problem_type_weights.items(), key=lambda item: item[1], reverse=True),
            "target_category_votes": sorted(target_weights.items(), key=lambda item: item[1], reverse=True),
            "answer_shape_votes": answer_shape_votes,
            "supported_answer_shapes": [shape for shape, _ in answer_shape_votes],
            "top_retrieval_goals": [goal for goal, _ in sorted(retrieval_goal_weights.items(), key=lambda item: item[1], reverse=True)[:6]],
        }

    def _recompute_cluster_summary(self, cluster: Dict[str, Any]) -> Dict[str, Any]:
        return self._recompute_cluster_summary_with_mode(cluster, use_llm=True)

    @staticmethod
    def _select_representative_variant(chain_variants: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not chain_variants:
            return {}
        ranked = sorted(
            [item for item in chain_variants if isinstance(item, dict)],
            key=lambda item: (
                float(item.get("success_rate", 0.0) or 0.0),
                float(item.get("intermediate_value_score", 0.0) or 0.0),
                -int(item.get("tool_count", 999) or 999),
                -float(item.get("chain_token_cost", 0.0) or 0.0),
                -float(item.get("avg_latency_ms", 0.0) or 0.0),
                int(item.get("support_count", 0) or 0),
            ),
            reverse=True,
        )
        return ranked[0] if ranked else {}

    def _mark_cluster_dirty(self, cluster_id: str) -> None:
        cluster_id = self._sanitize_text(cluster_id)
        if cluster_id:
            self._dirty_cluster_ids.add(cluster_id)

    def _mark_cluster_clean(self, cluster_id: str) -> None:
        cluster_id = self._sanitize_text(cluster_id)
        if cluster_id:
            self._dirty_cluster_ids.discard(cluster_id)

    def _recompute_cluster_summary_with_mode(self, cluster: Dict[str, Any], *, use_llm: bool) -> Dict[str, Any]:
        members = list(cluster.get("member_templates") or [])
        members.sort(key=lambda item: (int(item.get("support_count", 0) or 0), self._matching_success_rate(item)), reverse=True)
        statistics = self._cluster_statistics(members)
        score_supervision = "gt" if all(self._score_supervision_mode(item) == "gt" for item in members) else "none"
        llm_members = [self._template_summary_for_llm(item) for item in members[: self.max_members_for_distill_prompt]]
        distilled = self.cluster_distiller._fallback(member_templates=llm_members, cluster_statistics=statistics)
        if use_llm and len(llm_members) > 1:
            distilled = self.cluster_distiller.distill(
                cluster_id=self._sanitize_text(cluster.get("cluster_id")),
                member_templates=llm_members,
                cluster_statistics=statistics,
            )
        source_questions: List[Dict[str, str]] = []
        seen_source_ids = set()
        for member in members:
            for source in list(member.get("template_sources") or []):
                if not isinstance(source, dict):
                    continue
                question_id = self._sanitize_text(source.get("question_id"))
                question = self._sanitize_text(source.get("question"))
                if not question_id or question_id in seen_source_ids:
                    continue
                source_questions.append({"question_id": question_id, "question": question})
                seen_source_ids.add(question_id)
        query_pattern = distilled.get("query_pattern_prototype") if isinstance(distilled.get("query_pattern_prototype"), dict) else {}
        query_abstract = self._sanitize_text(query_pattern.get("query_abstract"))
        normalized_problem_type = normalize_problem_type(query_pattern.get("problem_type"))
        normalized_target = normalize_target_category(query_pattern.get("target_category"))
        normalized_answer_shape = normalize_answer_shape(query_pattern.get("answer_shape"))
        normalized_goals = sanitize_retrieval_goals(
            query_pattern.get("retrieval_goals"),
            problem_type=normalized_problem_type,
            answer_shape=normalized_answer_shape,
        )
        chain_variants = list(statistics.get("chain_variants") or [])
        representative_variant = self._select_representative_variant(chain_variants)
        recommended_chain = self._sanitize_list(distilled.get("recommended_chain"), limit=8)
        if representative_variant.get("recommended_chain"):
            recommended_chain = self._sanitize_list(representative_variant.get("recommended_chain"), limit=8) or recommended_chain
        query_pattern_text = StrategyQueryPatternExtractor.pattern_to_text({**query_pattern, "query_abstract": query_abstract})
        pattern_embedding = self._embed_text(query_pattern_text)
        cluster.update(
            {
                "pattern_id": self._sanitize_text(cluster.get("cluster_id")),
                "pattern_name": self._sanitize_text(distilled.get("pattern_name")) or self._sanitize_text(cluster.get("pattern_name")) or "Generic Retrieval Pattern",
                "pattern_description": self._sanitize_text(distilled.get("pattern_description")) or self._sanitize_text(cluster.get("pattern_description")),
                "recommended_chain": recommended_chain,
                "anti_patterns": self._sanitize_list(distilled.get("anti_patterns"), limit=12),
                "chain_rationale": self._sanitize_text(distilled.get("chain_rationale")),
                "chain_constraints": self._sanitize_list(distilled.get("chain_constraints"), limit=8),
                "query_pattern_prototype": {
                    "query_abstract": query_abstract,
                    "problem_type": normalized_problem_type,
                    "target_category": normalized_target,
                    "answer_shape": normalized_answer_shape,
                    "retrieval_goals": normalized_goals,
                    "notes": self._sanitize_text(query_pattern.get("notes")),
                },
                "query_abstract": query_abstract,
                "query_pattern_text": query_pattern_text,
                "pattern_embedding": pattern_embedding,
                "support_count": statistics["support_count"],
                "successful_attempts": statistics["support_count"],
                "attempt_count": statistics["attempt_count"],
                "success_rate": statistics["success_rate"],
                "score_supervision": score_supervision,
                "template_count": statistics["template_count"],
                "alternative_chains": statistics["top_chains"],
                "chain_variants": chain_variants,
                "representative_variant_id": self._sanitize_text(representative_variant.get("variant_id")),
                "member_template_ids": [self._sanitize_text(item.get("template_id")) for item in members if self._sanitize_text(item.get("template_id"))],
                "supported_answer_shapes": list(statistics.get("supported_answer_shapes") or []),
                "source_question_ids": [item["question_id"] for item in source_questions],
                "source_questions": source_questions,
                "success_summary": {
                    "support_count": statistics["support_count"],
                    "attempt_count": statistics["attempt_count"],
                    "success_rate": statistics["success_rate"],
                },
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
        )
        cluster_id = self._sanitize_text(cluster.get("cluster_id"))
        if use_llm or len(members) <= 1:
            self._mark_cluster_clean(cluster_id)
        else:
            self._mark_cluster_dirty(cluster_id)
        return cluster

    def _new_cluster_from_template(self, template: Dict[str, Any]) -> Dict[str, Any]:
        return self._new_cluster_from_templates([template])

    def _new_cluster_from_templates(self, templates: List[Dict[str, Any]]) -> Dict[str, Any]:
        first = templates[0] if templates else {}
        cluster = {
            "cluster_id": self._new_cluster_id(),
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "member_templates": list(templates or []),
            "pattern_name": first.get("pattern_name", "Generic Retrieval Pattern"),
            "pattern_description": first.get("pattern_description", ""),
            "recommended_chain": first.get("recommended_chain", []),
            "anti_patterns": first.get("anti_patterns", []),
            "query_pattern_prototype": first.get("query_pattern", {}),
            "query_abstract": first.get("query_abstract", ""),
            "query_pattern_text": first.get("query_pattern_text", ""),
            "pattern_embedding": first.get("pattern_embedding"),
        }
        return self._recompute_cluster_summary_with_mode(cluster, use_llm=False)

    def _incoming_group_min_score(self) -> float:
        return max(0.66, float(self.min_candidate_score or 0.0) + 0.10)

    def _group_incoming_templates(self, templates: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        if len(templates) <= 1:
            return [[item] for item in templates]

        parent = list(range(len(templates)))

        def find(idx: int) -> int:
            while parent[idx] != idx:
                parent[idx] = parent[parent[idx]]
                idx = parent[idx]
            return idx

        def union(left: int, right: int) -> None:
            root_left = find(left)
            root_right = find(right)
            if root_left != root_right:
                parent[root_right] = root_left

        threshold = self._incoming_group_min_score()
        views = [self._template_cluster_view(item) for item in templates]
        for left in range(len(templates)):
            for right in range(left + 1, len(templates)):
                if not self._candidate_eligible(templates[left], views[right], stage="incremental"):
                    continue
                pair_score = self._candidate_score(templates[left], views[right])
                if pair_score >= threshold:
                    union(left, right)

        grouped: Dict[int, List[Dict[str, Any]]] = {}
        for idx, item in enumerate(templates):
            root = find(idx)
            grouped.setdefault(root, []).append(item)
        ordered_groups = list(grouped.values())
        ordered_groups.sort(
            key=lambda group: (
                max(int(item.get("support_count", 0) or 0) for item in group),
                len(group),
            ),
            reverse=True,
        )
        for group in ordered_groups:
            group.sort(key=lambda item: (int(item.get("support_count", 0) or 0), self._matching_success_rate(item)), reverse=True)
        return ordered_groups

    def _record_group_decision(
        self,
        *,
        group: List[Dict[str, Any]],
        stage: str,
        decision: str,
        result_cluster_id: str,
        candidate_cluster_ids: List[str],
        reason: str,
    ) -> None:
        source_ids = [self._sanitize_text(item.get("template_id")) for item in group if self._sanitize_text(item.get("template_id"))]
        for source_template_id in source_ids or [""]:
            self._record_decision(
                {
                    "stage": stage,
                    "source_template_id": source_template_id,
                    "source_template_ids": source_ids,
                    "decision": decision,
                    "result_cluster_id": result_cluster_id,
                    "candidate_cluster_ids": list(candidate_cluster_ids or []),
                    "reason": reason,
                }
            )

    def _commit_sanitized_group(self, group: List[Dict[str, Any]]) -> str:
        if not group:
            return ""
        if not self.clusters:
            new_cluster = self._new_cluster_from_templates(group)
            self.clusters.append(new_cluster)
            self._record_group_decision(
                group=group,
                stage="incremental",
                decision="new_cluster",
                result_cluster_id=self._sanitize_text(new_cluster.get("cluster_id")),
                candidate_cluster_ids=[],
                reason="first_group_creates_cluster",
            )
            return self._sanitize_text(new_cluster.get("cluster_id"))

        incoming_view = self._cluster_template_view(self._new_cluster_from_templates(group))
        candidates = self._match_candidate_clusters(incoming_view, stage="incremental")
        decision = self.merge_decider.decide(
            incoming_template=self._template_summary_for_llm(incoming_view),
            candidate_clusters=candidates,
        )
        matched_cluster_id = self._sanitize_text(decision.get("matched_cluster_id"))
        candidate_cluster_ids = [self._sanitize_text(item.get("cluster_id")) for item in candidates]
        if decision.get("decision") in {"duplicate", "mergeable_variant"} and matched_cluster_id:
            target = self._find_cluster(matched_cluster_id)
            if target is not None:
                target_members = target.get("member_templates") if isinstance(target.get("member_templates"), list) else []
                target_members.extend(group)
                target["member_templates"] = target_members
                self._recompute_cluster_summary_with_mode(target, use_llm=False)
                self._record_group_decision(
                    group=group,
                    stage="incremental",
                    decision=decision.get("decision", ""),
                    result_cluster_id=matched_cluster_id,
                    candidate_cluster_ids=candidate_cluster_ids,
                    reason=self._sanitize_text(decision.get("reason")),
                )
                return matched_cluster_id

        new_cluster = self._new_cluster_from_templates(group)
        self.clusters.append(new_cluster)
        self._record_group_decision(
            group=group,
            stage="incremental",
            decision=decision.get("decision", "distinct") or "distinct",
            result_cluster_id=self._sanitize_text(new_cluster.get("cluster_id")),
            candidate_cluster_ids=candidate_cluster_ids,
            reason=self._sanitize_text(decision.get("reason")) or "create_new_cluster",
        )
        return self._sanitize_text(new_cluster.get("cluster_id"))

    def _candidate_score(self, template: Dict[str, Any], cluster: Dict[str, Any]) -> float:
        template_pattern = template.get("query_pattern") if isinstance(template.get("query_pattern"), dict) else {}
        cluster_pattern = cluster.get("query_pattern_prototype") if isinstance(cluster.get("query_pattern_prototype"), dict) else {}
        sim = cosine_sim(template.get("pattern_embedding"), cluster.get("pattern_embedding")) if template.get("pattern_embedding") and cluster.get("pattern_embedding") else None
        template_match_text = self._pattern_match_text(template_pattern, query_abstract=self._sanitize_text(template.get("query_abstract")))
        cluster_match_text = self._pattern_match_text(cluster_pattern, query_abstract=self._sanitize_text(cluster.get("query_abstract")))
        match_overlap = self._text_overlap_score(template_match_text, cluster_match_text)
        chain_overlap = self._chain_overlap(template.get("recommended_chain") or [], cluster.get("recommended_chain") or [])
        target_match = 1.0 if self._pattern_field(template, "target_category") == self._pattern_field(cluster, "target_category") and self._pattern_field(template, "target_category") else 0.0
        score = 0.0
        shape_match = self._answer_shape_match(template, cluster)
        problem_match = self._problem_type_match(template, cluster)
        if sim is not None:
            score += 0.40 * clamp_float(sim, low=0.0, high=1.0, default=0.0)
            score += 0.20 * match_overlap
            score += 0.16 * chain_overlap
            score += 0.16 * problem_match
            score += 0.06 * shape_match
            score += 0.02 * target_match
        else:
            score += 0.34 * match_overlap
            score += 0.22 * chain_overlap
            score += 0.24 * problem_match
            score += 0.14 * shape_match
            score += 0.06 * target_match
        if self._has_cross_intent_conflict(template, cluster):
            score *= 0.35
        return round(score, 6)

    def _cluster_summary_for_llm(self, cluster: Dict[str, Any], *, candidate_score: float) -> Dict[str, Any]:
        return {
            "cluster_id": self._sanitize_text(cluster.get("cluster_id")),
            "pattern_name": self._sanitize_text(cluster.get("pattern_name")),
            "pattern_description": self._sanitize_text(cluster.get("pattern_description")),
            "recommended_chain": self._sanitize_list(cluster.get("recommended_chain"), limit=8),
            "alternative_chains": list(cluster.get("alternative_chains") or [])[:4],
            "anti_patterns": self._sanitize_list(cluster.get("anti_patterns"), limit=8),
            "query_pattern_prototype": cluster.get("query_pattern_prototype") if isinstance(cluster.get("query_pattern_prototype"), dict) else {},
            "support_count": max(0, int(cluster.get("support_count", 0) or 0)),
            "success_rate": clamp_float(cluster.get("success_rate", 0.0), low=0.0, high=1.0, default=0.0),
            "score_supervision": self._score_supervision_mode(cluster),
            "template_count": max(0, int(cluster.get("template_count", 0) or 0)),
            "supported_answer_shapes": list(cluster.get("supported_answer_shapes") or []),
            "source_questions": list(cluster.get("source_questions") or [])[:3],
            "candidate_score": candidate_score,
        }

    def _match_candidate_clusters(
        self,
        template: Dict[str, Any],
        *,
        exclude_cluster_id: str = "",
        stage: str = "incremental",
    ) -> List[Dict[str, Any]]:
        ranked: List[Tuple[float, Dict[str, Any]]] = []
        for cluster in self.clusters:
            cluster_id = self._sanitize_text(cluster.get("cluster_id"))
            if exclude_cluster_id and cluster_id == exclude_cluster_id:
                continue
            if not self._candidate_eligible(template, cluster, stage=stage):
                continue
            score = self._candidate_score(template, cluster)
            ranked.append((score, cluster))
        ranked.sort(key=lambda item: (item[0], self._matching_success_rate(item[1]), int(item[1].get("support_count", 0) or 0)), reverse=True)
        selected: List[Dict[str, Any]] = []
        for score, cluster in ranked:
            if score >= self.min_candidate_score:
                selected.append(self._cluster_summary_for_llm(cluster, candidate_score=score))
            if len(selected) >= self.candidate_top_k:
                break
        return selected

    def _find_cluster(self, cluster_id: str) -> Optional[Dict[str, Any]]:
        cluster_id = self._sanitize_text(cluster_id)
        for cluster in self.clusters:
            if self._sanitize_text(cluster.get("cluster_id")) == cluster_id:
                return cluster
        return None

    def _record_decision(self, row: Dict[str, Any]) -> None:
        self.merge_decisions.append({
            "decision_id": self._new_decision_id(),
            "recorded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            **row,
        })

    def add_template(self, template: Dict[str, Any]) -> str:
        results = self.add_templates([template])
        return results[0] if results else ""

    def add_templates(self, templates: List[Dict[str, Any]]) -> List[str]:
        sanitized_rows: List[Dict[str, Any]] = []
        for template in templates or []:
            if not isinstance(template, dict):
                continue
            sanitized = self.sanitize_template(template, default_id=self._sanitize_text(template.get("template_id")))
            self.raw_templates.append(sanitized)
            sanitized_rows.append(sanitized)
        if not sanitized_rows:
            return []

        grouped = self._group_incoming_templates(sanitized_rows)
        result_by_template_id: Dict[str, str] = {}
        for group in grouped:
            cluster_id = self._commit_sanitized_group(group)
            for item in group:
                template_id = self._sanitize_text(item.get("template_id"))
                if template_id:
                    result_by_template_id[template_id] = cluster_id
        return [result_by_template_id.get(self._sanitize_text(item.get("template_id")), "") for item in sanitized_rows]

    def _merge_clusters(self, *, source_cluster_id: str, target_cluster_id: str, reason: str, stage: str) -> bool:
        source = self._find_cluster(source_cluster_id)
        target = self._find_cluster(target_cluster_id)
        if source is None or target is None or source is target:
            return False
        target_members = target.get("member_templates") if isinstance(target.get("member_templates"), list) else []
        source_members = source.get("member_templates") if isinstance(source.get("member_templates"), list) else []
        seen_ids = {self._sanitize_text(item.get("template_id")) for item in target_members if self._sanitize_text(item.get("template_id"))}
        for item in source_members:
            template_id = self._sanitize_text(item.get("template_id"))
            if template_id and template_id in seen_ids:
                continue
            target_members.append(item)
            if template_id:
                seen_ids.add(template_id)
        target["member_templates"] = target_members
        self._recompute_cluster_summary_with_mode(target, use_llm=False)
        self.clusters = [cluster for cluster in self.clusters if self._sanitize_text(cluster.get("cluster_id")) != source_cluster_id]
        self._mark_cluster_clean(source_cluster_id)
        self._record_decision(
            {
                "stage": stage,
                "source_cluster_id": source_cluster_id,
                "decision": "cluster_merge",
                "result_cluster_id": target_cluster_id,
                "candidate_cluster_ids": [target_cluster_id],
                "reason": reason,
            }
        )
        return True

    def consolidate(self, rounds: Optional[int] = None) -> int:
        total_merges = 0
        max_rounds = self.consolidation_rounds if rounds is None else max(0, int(rounds or 0))
        for _ in range(max_rounds):
            changed = False
            ordered_cluster_ids = [
                self._sanitize_text(item.get("cluster_id"))
                for item in sorted(
                    self.clusters,
                    key=lambda cluster: (int(cluster.get("support_count", 0) or 0), int(cluster.get("template_count", 0) or 0)),
                )
            ]
            for source_cluster_id in ordered_cluster_ids:
                source = self._find_cluster(source_cluster_id)
                if source is None:
                    continue
                source_template_view = self._cluster_template_view(source)
                candidates = self._match_candidate_clusters(
                    source_template_view,
                    exclude_cluster_id=source_cluster_id,
                    stage="global_consolidation",
                )
                if not candidates:
                    continue
                decision = self.merge_decider.decide(incoming_template=self._template_summary_for_llm(source_template_view), candidate_clusters=candidates)
                target_cluster_id = self._sanitize_text(decision.get("matched_cluster_id"))
                if decision.get("decision") not in {"duplicate", "mergeable_variant"} or not target_cluster_id:
                    continue
                if self._merge_clusters(
                    source_cluster_id=source_cluster_id,
                    target_cluster_id=target_cluster_id,
                    reason=self._sanitize_text(decision.get("reason")) or "global_consolidation",
                    stage="global_consolidation",
                ):
                    total_merges += 1
                    changed = True
            if not changed:
                break
        for cluster in self.clusters:
            cluster_id = self._sanitize_text(cluster.get("cluster_id"))
            if cluster_id and cluster_id in self._dirty_cluster_ids:
                self._recompute_cluster_summary_with_mode(cluster, use_llm=True)
        return total_merges

    def export_training_payload(self) -> Dict[str, Any]:
        return {
            "cluster_count": len(self.clusters),
            "clusters": [self._cluster_training_view(cluster) for cluster in self.clusters],
        }

    def _cluster_training_view(self, cluster: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "cluster_id": self._sanitize_text(cluster.get("cluster_id")),
            "pattern_id": self._sanitize_text(cluster.get("pattern_id") or cluster.get("cluster_id")),
            "pattern_name": self._sanitize_text(cluster.get("pattern_name")),
            "pattern_description": self._sanitize_text(cluster.get("pattern_description")),
            "recommended_chain": self._sanitize_list(cluster.get("recommended_chain"), limit=8),
            "alternative_chains": list(cluster.get("alternative_chains") or []),
            "chain_variants": list(cluster.get("chain_variants") or []),
            "representative_variant_id": self._sanitize_text(cluster.get("representative_variant_id")),
            "anti_patterns": self._sanitize_list(cluster.get("anti_patterns"), limit=12),
            "chain_rationale": self._sanitize_text(cluster.get("chain_rationale")),
            "chain_constraints": self._sanitize_list(cluster.get("chain_constraints"), limit=8),
            "query_pattern_prototype": cluster.get("query_pattern_prototype") if isinstance(cluster.get("query_pattern_prototype"), dict) else {},
            "query_abstract": self._sanitize_text(cluster.get("query_abstract")),
            "query_pattern_text": self._sanitize_text(cluster.get("query_pattern_text")),
            "pattern_embedding": cluster.get("pattern_embedding") or [],
            "support_count": max(0, int(cluster.get("support_count", 0) or 0)),
            "successful_attempts": max(0, int(cluster.get("successful_attempts", 0) or 0)),
            "attempt_count": max(1, int(cluster.get("attempt_count", 1) or 1)),
            "success_rate": clamp_float(cluster.get("success_rate", 0.0), low=0.0, high=1.0, default=0.0),
            "score_supervision": self._score_supervision_mode(cluster),
            "template_count": max(0, int(cluster.get("template_count", 0) or 0)),
            "supported_answer_shapes": list(cluster.get("supported_answer_shapes") or []),
            "member_template_ids": list(cluster.get("member_template_ids") or []),
            "member_templates": list(cluster.get("member_templates") or []),
            "source_question_ids": list(cluster.get("source_question_ids") or []),
            "source_questions": list(cluster.get("source_questions") or []),
            "success_summary": dict(cluster.get("success_summary") or {}),
            "created_at": self._sanitize_text(cluster.get("created_at")),
            "updated_at": self._sanitize_text(cluster.get("updated_at")),
        }

    def runtime_templates(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for cluster in self.clusters:
            rows.append(
                {
                    "template_id": self._sanitize_text(cluster.get("pattern_id") or cluster.get("cluster_id")),
                    "pattern_id": self._sanitize_text(cluster.get("pattern_id") or cluster.get("cluster_id")),
                    "cluster_id": self._sanitize_text(cluster.get("cluster_id")),
                    "pattern_name": self._sanitize_text(cluster.get("pattern_name")),
                    "pattern_description": self._sanitize_text(cluster.get("pattern_description")),
                    "recommended_chain": self._sanitize_list(cluster.get("recommended_chain"), limit=8),
                    "chain_variants": list(cluster.get("chain_variants") or []),
                    "representative_variant_id": self._sanitize_text(cluster.get("representative_variant_id")),
                    "anti_patterns": self._sanitize_list(cluster.get("anti_patterns"), limit=12),
                    "chain_rationale": self._sanitize_text(cluster.get("chain_rationale")),
                    "chain_constraints": self._sanitize_list(cluster.get("chain_constraints"), limit=8),
                    "support_count": max(0, int(cluster.get("support_count", 0) or 0)),
                    "success_rate": clamp_float(cluster.get("success_rate", 0.0), low=0.0, high=1.0, default=0.0),
                    "score_supervision": self._score_supervision_mode(cluster),
                    "supported_answer_shapes": list(cluster.get("supported_answer_shapes") or []),
                    "query_pattern": cluster.get("query_pattern_prototype") if isinstance(cluster.get("query_pattern_prototype"), dict) else {},
                    "query_abstract": self._sanitize_text(cluster.get("query_abstract")),
                    "query_pattern_text": self._sanitize_text(cluster.get("query_pattern_text")),
                    "pattern_embedding": cluster.get("pattern_embedding") or [],
                    "source_question_ids": list(cluster.get("source_question_ids") or []),
                    "source_questions": list(cluster.get("source_questions") or []),
                    "success_summary": dict(cluster.get("success_summary") or {}),
                }
            )
        return rows

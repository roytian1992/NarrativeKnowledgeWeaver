from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

from core.utils.format import correct_json_format
from core.utils.function_manager import process_with_format_guarantee
from core.utils.general_text import general_repair_template
from core.utils.general_utils import parse_json_object_from_text

logger = logging.getLogger(__name__)


def _sanitize_text(value: Any) -> str:
    return str(value or "").strip()


def _sanitize_list(items: Any, *, limit: int = 10) -> List[str]:
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


def _problem_type(payload: Dict[str, Any], *, field_name: str) -> str:
    pattern = payload.get(field_name) if isinstance(payload.get(field_name), dict) else {}
    return _sanitize_text(pattern.get("problem_type")).lower()


def _answer_shape(payload: Dict[str, Any], *, field_name: str) -> str:
    pattern = payload.get(field_name) if isinstance(payload.get(field_name), dict) else {}
    return _sanitize_text(pattern.get("answer_shape")).lower()


def _normalize_chain(chain: Any) -> List[str]:
    return [x for x in _sanitize_list(chain) if x]


def _chain_similarity(a: List[str], b: List[str]) -> float:
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    set_score = float(len(set(a) & set(b))) / float(len(set(a) | set(b)))
    prefix_len = 0
    for left, right in zip(a, b):
        if left != right:
            break
        prefix_len += 1
    prefix_score = float(prefix_len) / float(max(len(a), len(b)))
    boundary_score = 0.0
    if a[0] == b[0]:
        boundary_score += 0.5
    if a[-1] == b[-1]:
        boundary_score += 0.5
    return round(0.45 * set_score + 0.35 * prefix_score + 0.20 * boundary_score, 6)


def _cross_intent_conflict(incoming_problem: str, candidate_problem: str) -> bool:
    if not incoming_problem or not candidate_problem or incoming_problem == candidate_problem:
        return False
    incompatible_pairs = {
        frozenset({"content_span_lookup", "section_localization"}),
        frozenset({"content_span_lookup", "entity_attribute_lookup"}),
        frozenset({"content_span_lookup", "relation_or_interaction_lookup"}),
        frozenset({"content_span_lookup", "causal_or_explanatory_lookup"}),
        frozenset({"section_localization", "causal_or_explanatory_lookup"}),
    }
    return frozenset({incoming_problem, candidate_problem}) in incompatible_pairs


class StrategyTemplateMergeDecider:
    def __init__(self, prompt_loader, llm, prompt_id: str = "memory/decide_strategy_template_merge"):
        self.prompt_loader = prompt_loader
        self.llm = llm
        self.prompt_id = prompt_id

    @staticmethod
    def _heuristic_decision(incoming_template: Dict[str, Any], candidate_clusters: List[Dict[str, Any]]) -> Dict[str, Any]:
        incoming_pattern = incoming_template.get("query_pattern") if isinstance(incoming_template.get("query_pattern"), dict) else {}
        incoming_problem_type = _sanitize_text(incoming_pattern.get("problem_type")).lower()
        incoming_target = _sanitize_text(incoming_pattern.get("target_category")).lower()
        incoming_answer_shape = _sanitize_text(incoming_pattern.get("answer_shape")).lower()
        incoming_chain = _normalize_chain(incoming_template.get("recommended_chain"))

        best = None
        best_score = -1.0
        for item in candidate_clusters or []:
            score = float(item.get("candidate_score", 0.0) or 0.0)
            candidate_pattern = item.get("query_pattern_prototype") if isinstance(item.get("query_pattern_prototype"), dict) else {}
            candidate_problem_type = _sanitize_text(candidate_pattern.get("problem_type")).lower()
            if _cross_intent_conflict(incoming_problem_type, candidate_problem_type):
                continue
            bonus = 0.0
            if incoming_problem_type and incoming_problem_type == candidate_problem_type:
                bonus += 0.20
            if incoming_target and incoming_target == _sanitize_text(candidate_pattern.get("target_category")).lower():
                bonus += 0.12
            if incoming_answer_shape and incoming_answer_shape == _sanitize_text(candidate_pattern.get("answer_shape")).lower():
                bonus += 0.12
            candidate_chain = _normalize_chain(item.get("recommended_chain"))
            chain_score = _chain_similarity(incoming_chain, candidate_chain)
            bonus += 0.18 * chain_score
            if incoming_chain and candidate_chain and incoming_chain == candidate_chain:
                bonus += 0.10
            total = score + bonus
            if total > best_score:
                best = item
                best_score = total

        if best is None:
            return {
                "decision": "distinct",
                "matched_cluster_id": "",
                "reason": "no_candidate_clusters",
            }

        candidate_pattern = best.get("query_pattern_prototype") if isinstance(best.get("query_pattern_prototype"), dict) else {}
        same_problem = incoming_problem_type and incoming_problem_type == _sanitize_text(candidate_pattern.get("problem_type")).lower()
        same_target = incoming_target and incoming_target == _sanitize_text(candidate_pattern.get("target_category")).lower()
        same_shape = incoming_answer_shape and incoming_answer_shape == _sanitize_text(candidate_pattern.get("answer_shape")).lower()
        candidate_chain = _normalize_chain(best.get("recommended_chain"))
        same_chain = bool(incoming_chain and candidate_chain and incoming_chain == candidate_chain)
        chain_score = _chain_similarity(incoming_chain, candidate_chain)
        if _cross_intent_conflict(incoming_problem_type, _sanitize_text(candidate_pattern.get("problem_type")).lower()):
            return {
                "decision": "distinct",
                "matched_cluster_id": "",
                "reason": "cross_intent_problem_type_conflict",
            }
        if incoming_answer_shape and _answer_shape(best, field_name="query_pattern_prototype") and incoming_answer_shape != _answer_shape(best, field_name="query_pattern_prototype") and chain_score < 0.9:
            return {
                "decision": "distinct",
                "matched_cluster_id": "",
                "reason": "answer_shape_conflict_without_near_identical_chain",
            }
        if not same_problem or not same_shape:
            return {
                "decision": "distinct",
                "matched_cluster_id": "",
                "reason": "requires_same_problem_type_and_answer_shape",
            }
        mergeable = best_score >= 0.84 or (best_score >= 0.76 and same_problem and same_shape and (same_target or chain_score >= 0.42))
        if not mergeable:
            return {
                "decision": "distinct",
                "matched_cluster_id": "",
                "reason": "heuristic_similarity_below_merge_threshold",
            }
        return {
            "decision": "duplicate" if same_problem and same_target and same_shape and same_chain and chain_score >= 0.82 else "mergeable_variant",
            "matched_cluster_id": _sanitize_text(best.get("cluster_id")),
            "reason": "heuristic_match",
        }

    def _sanitize_result(self, payload: Dict[str, Any], allowed_cluster_ids: List[str], fallback: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(payload, dict):
            return fallback
        decision = _sanitize_text(payload.get("decision")).lower()
        if decision not in {"duplicate", "mergeable_variant", "distinct"}:
            decision = fallback["decision"]
        matched_cluster_id = _sanitize_text(payload.get("matched_cluster_id"))
        if matched_cluster_id not in allowed_cluster_ids:
            matched_cluster_id = ""
        if decision in {"duplicate", "mergeable_variant"} and not matched_cluster_id:
            decision = "distinct"
        return {
            "decision": decision,
            "matched_cluster_id": matched_cluster_id,
            "reason": _sanitize_text(payload.get("reason")) or fallback.get("reason", ""),
        }

    def decide(
        self,
        *,
        incoming_template: Dict[str, Any],
        candidate_clusters: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        fallback = self._heuristic_decision(incoming_template=incoming_template, candidate_clusters=candidate_clusters)
        if not candidate_clusters:
            return fallback
        if self.prompt_loader is None or self.llm is None:
            return fallback
        allowed_cluster_ids = [_sanitize_text(item.get("cluster_id")) for item in candidate_clusters if _sanitize_text(item.get("cluster_id"))]
        try:
            prompt = self.prompt_loader.render(
                self.prompt_id,
                static_values={},
                task_values={
                    "incoming_template_json": json.dumps(incoming_template or {}, ensure_ascii=False, indent=2),
                    "candidate_clusters_json": json.dumps(candidate_clusters or [], ensure_ascii=False, indent=2),
                },
                strict=True,
            )
        except Exception as exc:
            logger.warning("strategy merge decision prompt render failed: %s", exc)
            return fallback

        try:
            corrected_json, status = process_with_format_guarantee(
                llm_client=self.llm,
                messages=[{"role": "user", "content": prompt}],
                required_fields=["decision", "matched_cluster_id", "reason"],
                field_validators={
                    "decision": lambda v: isinstance(v, str),
                    "matched_cluster_id": lambda v: isinstance(v, str),
                    "reason": lambda v: isinstance(v, str),
                },
                max_retries=2,
                repair_template=general_repair_template,
            )
        except Exception as exc:
            logger.warning("strategy merge decision failed: %s", exc)
            return {
                **fallback,
                "reason": f"merge_decision_error: {str(exc).strip()[:500]}",
            }
        if status != "success":
            return fallback
        payload = parse_json_object_from_text(corrected_json) or {}
        return self._sanitize_result(payload, allowed_cluster_ids, fallback)


def decide_strategy_template_merge_with_guard(
    *,
    llm,
    prompt_loader,
    incoming_template: Dict[str, Any],
    candidate_clusters: List[Dict[str, Any]],
    prompt_id: str = "memory/decide_strategy_template_merge",
) -> str:
    decider = StrategyTemplateMergeDecider(prompt_loader=prompt_loader, llm=llm, prompt_id=prompt_id)
    result = decider.decide(
        incoming_template=incoming_template,
        candidate_clusters=candidate_clusters,
    )
    return correct_json_format(json.dumps(result, ensure_ascii=False))


class StrategyTemplateMergeDeciderTool:
    def __init__(self, prompt_loader, llm, prompt_id: str = "memory/decide_strategy_template_merge"):
        self.prompt_loader = prompt_loader
        self.llm = llm
        self.prompt_id = prompt_id

    def call(self, params: str, **kwargs) -> str:
        try:
            payload = json.loads(params) if isinstance(params, str) else (params or {})
        except Exception as exc:
            return correct_json_format(json.dumps({"error": f"params parse failed: {exc}"}, ensure_ascii=False))
        return decide_strategy_template_merge_with_guard(
            llm=self.llm,
            prompt_loader=self.prompt_loader,
            incoming_template=payload.get("incoming_template") or {},
            candidate_clusters=payload.get("candidate_clusters") or [],
            prompt_id=self.prompt_id,
        )

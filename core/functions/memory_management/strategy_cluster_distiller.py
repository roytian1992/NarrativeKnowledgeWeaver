from __future__ import annotations

import json
import logging
from collections import Counter
from typing import Any, Dict, List

from core.utils.format import correct_json_format
from core.utils.function_manager import process_with_format_guarantee
from core.utils.general_text import general_repair_template
from core.utils.general_utils import parse_json_object_from_text

logger = logging.getLogger(__name__)


def _sanitize_text(value: Any) -> str:
    return str(value or "").strip()


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


class StrategyClusterDistiller:
    def __init__(self, prompt_loader, llm, prompt_id: str = "memory/distill_strategy_cluster"):
        self.prompt_loader = prompt_loader
        self.llm = llm
        self.prompt_id = prompt_id

    @staticmethod
    def _fallback_query_pattern(member_templates: List[Dict[str, Any]]) -> Dict[str, Any]:
        query_patterns = [item.get("query_pattern") for item in member_templates if isinstance(item.get("query_pattern"), dict)]
        if not query_patterns:
            return {
                "query_abstract": "",
                "problem_type": "fact_retrieval",
                "target_category": "narrative_fact",
                "answer_shape": "short_fact",
                "retrieval_goals": ["retrieve evidence before answering"],
                "notes": "fallback_cluster_pattern",
            }

        def _weighted_pick(field: str, default: str = "") -> str:
            counts: Counter[str] = Counter()
            for member in member_templates:
                qp = member.get("query_pattern") if isinstance(member.get("query_pattern"), dict) else {}
                key = _sanitize_text(qp.get(field))
                if key:
                    counts[key] += max(1, int(member.get("support_count", 1) or 1))
            if counts:
                return counts.most_common(1)[0][0]
            return default

        retrieval_goal_counts: Counter[str] = Counter()
        query_abstract_counts: Counter[str] = Counter()
        for member in member_templates:
            weight = max(1, int(member.get("support_count", 1) or 1))
            qp = member.get("query_pattern") if isinstance(member.get("query_pattern"), dict) else {}
            query_abstract = _sanitize_text(qp.get("query_abstract"))
            if query_abstract:
                query_abstract_counts[query_abstract] += weight
            for goal in _sanitize_list(qp.get("retrieval_goals"), limit=8):
                retrieval_goal_counts[goal] += weight

        query_abstract = query_abstract_counts.most_common(1)[0][0] if query_abstract_counts else _sanitize_text(query_patterns[0].get("query_abstract"))
        retrieval_goals = [goal for goal, _ in retrieval_goal_counts.most_common(4)] or _sanitize_list(query_patterns[0].get("retrieval_goals"), limit=4)
        return {
            "query_abstract": query_abstract,
            "problem_type": _weighted_pick("problem_type", "fact_retrieval"),
            "target_category": _weighted_pick("target_category", "narrative_fact"),
            "answer_shape": _weighted_pick("answer_shape", "short_fact"),
            "retrieval_goals": retrieval_goals,
            "notes": "cluster_fallback_prototype",
        }

    @staticmethod
    def _fallback(member_templates: List[Dict[str, Any]], cluster_statistics: Dict[str, Any]) -> Dict[str, Any]:
        fallback_pattern = StrategyClusterDistiller._fallback_query_pattern(member_templates)
        top_chains = cluster_statistics.get("top_chains") or []
        recommended_chain = _sanitize_list(top_chains[0].get("chain") if top_chains and isinstance(top_chains[0], dict) else [], limit=8)
        if not recommended_chain:
            for member in member_templates:
                recommended_chain = _sanitize_list(member.get("recommended_chain"), limit=8)
                if recommended_chain:
                    break
        anti_patterns: List[str] = []
        for member in member_templates:
            for item in _sanitize_list(member.get("anti_patterns"), limit=12):
                if item not in anti_patterns:
                    anti_patterns.append(item)
                if len(anti_patterns) >= 8:
                    break
            if len(anti_patterns) >= 8:
                break
        pattern_name = _sanitize_text(member_templates[0].get("pattern_name")) if member_templates else "Generic Retrieval Pattern"
        if len(member_templates) > 1:
            pattern_name = _sanitize_text(fallback_pattern.get("problem_type")).replace("_", " ") or pattern_name
        if not pattern_name:
            pattern_name = "Generic Retrieval Pattern"
        pattern_description = _sanitize_text(member_templates[0].get("pattern_description")) if member_templates else ""
        if not pattern_description:
            chain_text = " -> ".join(recommended_chain) if recommended_chain else "adaptive evidence retrieval"
            pattern_description = (
                f"Generic strategy for {fallback_pattern.get('problem_type', 'fact retrieval')} "
                f"targeting {fallback_pattern.get('target_category', 'narrative facts')} with preferred chain {chain_text}."
            )
        return {
            "pattern_name": pattern_name,
            "pattern_description": pattern_description,
            "recommended_chain": recommended_chain,
            "anti_patterns": anti_patterns,
            "query_pattern_prototype": fallback_pattern,
        }

    def _sanitize_result(self, payload: Dict[str, Any], fallback: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(payload, dict):
            return fallback
        qp = payload.get("query_pattern_prototype") if isinstance(payload.get("query_pattern_prototype"), dict) else {}
        sanitized_pattern = {
            "query_abstract": _sanitize_text(qp.get("query_abstract")) or fallback["query_pattern_prototype"]["query_abstract"],
            "problem_type": _sanitize_text(qp.get("problem_type")) or fallback["query_pattern_prototype"]["problem_type"],
            "target_category": _sanitize_text(qp.get("target_category")) or fallback["query_pattern_prototype"]["target_category"],
            "answer_shape": _sanitize_text(qp.get("answer_shape")) or fallback["query_pattern_prototype"]["answer_shape"],
            "retrieval_goals": _sanitize_list(qp.get("retrieval_goals"), limit=4) or fallback["query_pattern_prototype"]["retrieval_goals"],
            "notes": _sanitize_text(qp.get("notes")) or fallback["query_pattern_prototype"].get("notes", ""),
        }
        return {
            "pattern_name": _sanitize_text(payload.get("pattern_name")) or fallback["pattern_name"],
            "pattern_description": _sanitize_text(payload.get("pattern_description")) or fallback["pattern_description"],
            "recommended_chain": _sanitize_list(payload.get("recommended_chain"), limit=8) or fallback["recommended_chain"],
            "anti_patterns": _sanitize_list(payload.get("anti_patterns"), limit=12) or fallback["anti_patterns"],
            "query_pattern_prototype": sanitized_pattern,
        }

    def distill(
        self,
        *,
        cluster_id: str,
        member_templates: List[Dict[str, Any]],
        cluster_statistics: Dict[str, Any],
    ) -> Dict[str, Any]:
        fallback = self._fallback(member_templates=member_templates, cluster_statistics=cluster_statistics)
        if self.prompt_loader is None or self.llm is None:
            return fallback
        try:
            prompt = self.prompt_loader.render(
                self.prompt_id,
                static_values={},
                task_values={
                    "cluster_id": _sanitize_text(cluster_id),
                    "member_templates_json": json.dumps(member_templates or [], ensure_ascii=False, indent=2),
                    "cluster_statistics_json": json.dumps(cluster_statistics or {}, ensure_ascii=False, indent=2),
                },
                strict=True,
            )
        except Exception as exc:
            logger.warning("strategy cluster distill prompt render failed: %s", exc)
            return fallback
        try:
            corrected_json, status = process_with_format_guarantee(
                llm_client=self.llm,
                messages=[{"role": "user", "content": prompt}],
                required_fields=["pattern_name", "pattern_description", "recommended_chain", "anti_patterns", "query_pattern_prototype"],
                field_validators={
                    "pattern_name": lambda v: isinstance(v, str),
                    "pattern_description": lambda v: isinstance(v, str),
                    "recommended_chain": lambda v: isinstance(v, list),
                    "anti_patterns": lambda v: isinstance(v, list),
                    "query_pattern_prototype": lambda v: isinstance(v, dict),
                },
                max_retries=2,
                repair_template=general_repair_template,
            )
        except Exception as exc:
            logger.warning("strategy cluster distill failed: %s", exc)
            return fallback
        if status != "success":
            return fallback
        payload = parse_json_object_from_text(corrected_json) or {}
        return self._sanitize_result(payload, fallback)


def distill_strategy_cluster_with_guard(
    *,
    llm,
    prompt_loader,
    cluster_id: str,
    member_templates: List[Dict[str, Any]],
    cluster_statistics: Dict[str, Any],
    prompt_id: str = "memory/distill_strategy_cluster",
) -> str:
    distiller = StrategyClusterDistiller(prompt_loader=prompt_loader, llm=llm, prompt_id=prompt_id)
    result = distiller.distill(
        cluster_id=cluster_id,
        member_templates=member_templates,
        cluster_statistics=cluster_statistics,
    )
    return correct_json_format(json.dumps(result, ensure_ascii=False))


class StrategyClusterDistillerTool:
    def __init__(self, prompt_loader, llm, prompt_id: str = "memory/distill_strategy_cluster"):
        self.prompt_loader = prompt_loader
        self.llm = llm
        self.prompt_id = prompt_id

    def call(self, params: str, **kwargs) -> str:
        try:
            payload = json.loads(params) if isinstance(params, str) else (params or {})
        except Exception as exc:
            return correct_json_format(json.dumps({"error": f"params parse failed: {exc}"}, ensure_ascii=False))
        return distill_strategy_cluster_with_guard(
            llm=self.llm,
            prompt_loader=self.prompt_loader,
            cluster_id=_sanitize_text(payload.get("cluster_id")),
            member_templates=payload.get("member_templates") or [],
            cluster_statistics=payload.get("cluster_statistics") or {},
            prompt_id=self.prompt_id,
        )

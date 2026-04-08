from __future__ import annotations

import difflib
import os
import time
from typing import Any, Dict, List, Optional

from core.functions.memory_management.strategy_query_pattern import (
    StrategyQueryPatternExtractor,
    normalize_answer_shape,
    normalize_problem_type,
    normalize_target_category,
    sanitize_retrieval_goals,
)
from core.utils.general_utils import (
    clamp_float,
    cosine_sim,
    json_dump_atomic,
    load_json,
    token_jaccard_overlap,
)


class StrategyTemplateLibrary:
    _CHAIN_DEPENDENCY_DEFAULTS = {
        "get_entity_sections": ["retrieve_entity_by_name"],
        "search_related_entities": ["retrieve_entity_by_name"],
        "get_relations_between_entities": ["retrieve_entity_by_name"],
        "get_common_neighbors": ["retrieve_entity_by_name"],
        "find_paths_between_nodes": ["retrieve_entity_by_name"],
        "get_k_hop_subgraph": ["retrieve_entity_by_name"],
    }

    @staticmethod
    def _score_supervision_mode(item: Dict[str, Any]) -> str:
        mode = str(item.get("score_supervision", "gt") or "gt").strip().lower()
        return "none" if mode in {"none", "no_gt", "unsupervised", "online_unsupervised"} else "gt"

    @classmethod
    def effective_success_rate_for_matching(cls, item: Dict[str, Any]) -> float:
        if cls._score_supervision_mode(item) != "gt":
            return 0.0
        return clamp_float(item.get("success_rate", 0.0), low=0.0, high=1.0, default=0.0)

    @classmethod
    def quality_factor_for_matching(cls, item: Dict[str, Any]) -> float:
        support_count = max(0, int(item.get("support_count", 0) or 0))
        support_factor = clamp_float(float(support_count) / 5.0, low=0.0, high=1.0, default=0.0)
        if cls._score_supervision_mode(item) != "gt":
            return support_factor
        success_rate = cls.effective_success_rate_for_matching(item)
        return clamp_float(0.65 * success_rate + 0.35 * support_factor, low=0.0, high=1.0, default=0.0)

    @staticmethod
    def _normalize_question_text(text: Any) -> str:
        raw = str(text or "").strip().lower()
        if not raw:
            return ""
        out: List[str] = []
        for ch in raw:
            if ch.isascii():
                if ch.isalnum():
                    out.append(ch)
                continue
            if "\u4e00" <= ch <= "\u9fff":
                out.append(ch)
        return "".join(out)

    @staticmethod
    def _pattern_field(pattern: Dict[str, Any], field: str) -> str:
        return str((pattern or {}).get(field, "") or "").strip().lower()

    @classmethod
    def _problem_type_match(cls, query_pattern: Dict[str, Any], template_pattern: Dict[str, Any]) -> float:
        query_problem = cls._pattern_field(query_pattern, "problem_type")
        template_problem = cls._pattern_field(template_pattern, "problem_type")
        if not query_problem or not template_problem:
            return 0.0
        return 1.0 if query_problem == template_problem else 0.0

    @classmethod
    def _target_category_match(cls, query_pattern: Dict[str, Any], template_pattern: Dict[str, Any]) -> float:
        query_target = cls._pattern_field(query_pattern, "target_category")
        template_target = cls._pattern_field(template_pattern, "target_category")
        if not query_target or not template_target:
            return 0.0
        return 1.0 if query_target == template_target else 0.0

    @classmethod
    def _has_cross_intent_conflict(cls, query_pattern: Dict[str, Any], template_pattern: Dict[str, Any]) -> bool:
        query_problem = cls._pattern_field(query_pattern, "problem_type")
        template_problem = cls._pattern_field(template_pattern, "problem_type")
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

    @staticmethod
    def _clip_text(text: Any, limit: int = 140) -> str:
        raw = str(text or "").strip()
        if len(raw) <= limit:
            return raw
        return raw[: max(0, limit - 3)] + "..."

    def __init__(
        self,
        *,
        library_path: str,
        embedding_model: Any = None,
        min_template_support: int = 1,
    ) -> None:
        self.library_path = str(library_path or "").strip()
        self.embedding_model = embedding_model
        self.min_template_support = max(1, int(min_template_support or 1))
        self._mtime: Optional[float] = None
        self._payload: Dict[str, Any] = {}
        self._templates: List[Dict[str, Any]] = []
        self.reload(force=True)

    @staticmethod
    def _text_overlap_score(a: str, b: str) -> float:
        return token_jaccard_overlap(a, b)

    @classmethod
    def _question_similarity(cls, query: str, candidate: str) -> float:
        q_norm = cls._normalize_question_text(query)
        c_norm = cls._normalize_question_text(candidate)
        if not q_norm or not c_norm:
            return 0.0
        if q_norm == c_norm:
            return 1.0
        seq_score = difflib.SequenceMatcher(None, q_norm, c_norm).ratio()
        overlap_score = cls._text_overlap_score(query, candidate)
        return clamp_float(0.65 * seq_score + 0.35 * overlap_score, low=0.0, high=1.0, default=0.0)

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
    def _answer_shape_match(query_pattern: Dict[str, Any], template_pattern: Dict[str, Any]) -> float:
        q_shape = str((query_pattern or {}).get("answer_shape", "") or "").strip().lower()
        t_shape = str((template_pattern or {}).get("answer_shape", "") or "").strip().lower()
        if not q_shape or not t_shape:
            return 0.0
        return 1.0 if q_shape == t_shape else 0.0

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

    @staticmethod
    def _sanitize_chain(chain: Any, *, limit: int = 8) -> List[str]:
        if not isinstance(chain, list):
            return []
        out: List[str] = []
        for item in chain:
            value = str(item or "").strip()
            if value and value not in out:
                out.append(value)
            if len(out) >= limit:
                break
        return out

    @classmethod
    def _repair_chain_dependencies(cls, chain: List[str], *, query_pattern: Optional[Dict[str, Any]] = None) -> List[str]:
        repaired = [str(item).strip() for item in (chain or []) if str(item).strip()]
        if not repaired:
            return []

        query_pattern = query_pattern or {}
        target_category = str(query_pattern.get("target_category", "") or "").strip().lower()
        for tool_name, defaults in cls._CHAIN_DEPENDENCY_DEFAULTS.items():
            if tool_name in repaired and "retrieve_entity_by_name" not in repaired:
                repaired = [*defaults, *repaired]

        if "vdb_get_docs_by_document_ids" in repaired:
            has_doc_id_producer = any(
                name in repaired
                for name in (
                    "bm25_search_docs",
                    "section_evidence_search",
                    "search_sections",
                    "retrieve_entity_by_name",
                    "vdb_search_docs",
                    "lookup_document_ids_by_title",
                    "search_related_content",
                )
            )
            if not has_doc_id_producer:
                seed_tool = (
                    "retrieve_entity_by_name"
                    if target_category in {"character", "object_or_device", "place", "organization"}
                    else "bm25_search_docs"
                )
                repaired = [seed_tool, *repaired]

        deduped: List[str] = []
        for item in repaired:
            if item not in deduped:
                deduped.append(item)
        return deduped[:8]

    def _embed_text(self, text: str) -> Optional[List[float]]:
        if self.embedding_model is None or not hasattr(self.embedding_model, "embed_query"):
            return None
        try:
            emb = self.embedding_model.embed_query(text)
        except Exception:
            return None
        return self._safe_embedding(emb)

    @staticmethod
    def _select_representative_variant(chain_variants: List[Dict[str, Any]]) -> Dict[str, Any]:
        valid = [item for item in chain_variants if isinstance(item, dict) and list(item.get("recommended_chain") or [])]
        if not valid:
            return {}
        ranked = sorted(
            valid,
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
        return ranked[0]

    def _sanitize_chain_variant(self, item: Dict[str, Any], idx: int) -> Optional[Dict[str, Any]]:
        if not isinstance(item, dict):
            return None
        query_pattern = item.get("query_pattern") if isinstance(item.get("query_pattern"), dict) else {}
        chain = self._repair_chain_dependencies(
            self._sanitize_chain(item.get("recommended_chain") or item.get("chain")),
            query_pattern=query_pattern,
        )
        if not chain:
            return None
        return {
            "variant_id": str(item.get("variant_id", f"variant_{idx:03d}") or f"variant_{idx:03d}").strip(),
            "recommended_chain": chain,
            "tool_count": max(1, int(item.get("tool_count", len(chain)) or len(chain))),
            "chain_token_cost": float(item.get("chain_token_cost", 0.0) or 0.0),
            "support_count": max(0, int(item.get("support_count", 0) or 0)),
            "success_rate": clamp_float(item.get("success_rate", 0.0), low=0.0, high=1.0, default=0.0),
            "intermediate_value_score": clamp_float(item.get("intermediate_value_score", 0.0), low=0.0, high=1.0, default=0.0),
            "avg_latency_ms": max(0.0, float(item.get("avg_latency_ms", 0.0) or 0.0)),
            "source_attempt_ids": [str(x).strip() for x in (item.get("source_attempt_ids") or []) if str(x).strip()],
            "source_template_ids": [str(x).strip() for x in (item.get("source_template_ids") or []) if str(x).strip()],
        }

    def _sanitize_template(self, item: Dict[str, Any], idx: int) -> Optional[Dict[str, Any]]:
        if not isinstance(item, dict):
            return None
        query_pattern = item.get("query_pattern") if isinstance(item.get("query_pattern"), dict) else {}
        if not query_pattern:
            query_pattern = item.get("query_pattern_prototype") if isinstance(item.get("query_pattern_prototype"), dict) else {}
        query_abstract = str(item.get("query_abstract", "") or query_pattern.get("query_abstract", "")).strip()
        normalized_problem_type = normalize_problem_type(query_pattern.get("problem_type") or item.get("problem_type"))
        normalized_target = normalize_target_category(query_pattern.get("target_category") or item.get("target_category"))
        normalized_answer_shape = normalize_answer_shape(query_pattern.get("answer_shape") or item.get("answer_shape"))
        normalized_goals = sanitize_retrieval_goals(
            query_pattern.get("retrieval_goals") or item.get("retrieval_goals"),
            problem_type=normalized_problem_type,
            answer_shape=normalized_answer_shape,
        )
        normalized_query_pattern = {
            "query_abstract": query_abstract,
            "problem_type": normalized_problem_type,
            "target_category": normalized_target,
            "answer_shape": normalized_answer_shape,
            "retrieval_goals": normalized_goals,
            "notes": str(query_pattern.get("notes", "") or item.get("notes", "")).strip(),
        }
        query_pattern_text = str(item.get("query_pattern_text", "") or "").strip()
        if not query_pattern_text:
            query_pattern_text = StrategyQueryPatternExtractor.pattern_to_text(normalized_query_pattern)

        raw_variants = item.get("chain_variants")
        chain_variants: List[Dict[str, Any]] = []
        if isinstance(raw_variants, list):
            for v_idx, variant in enumerate(raw_variants, start=1):
                sanitized_variant = self._sanitize_chain_variant(variant, v_idx)
                if sanitized_variant is not None:
                    chain_variants.append(sanitized_variant)
        if not chain_variants:
            alternative_chains = item.get("alternative_chains") if isinstance(item.get("alternative_chains"), list) else []
            for v_idx, variant in enumerate(alternative_chains, start=1):
                chain_variant = self._sanitize_chain_variant(variant, v_idx)
                if chain_variant is not None:
                    chain_variants.append(chain_variant)
        if not chain_variants:
            fallback_chain = self._repair_chain_dependencies(
                self._sanitize_chain(item.get("recommended_chain")),
                query_pattern=normalized_query_pattern,
            )
            if fallback_chain:
                chain_variants.append(
                    {
                        "variant_id": "variant_001",
                        "recommended_chain": fallback_chain,
                        "tool_count": len(fallback_chain),
                        "chain_token_cost": 0.0,
                        "support_count": max(0, int(item.get("support_count", 0) or 0)),
                        "success_rate": clamp_float(item.get("success_rate", 0.0), low=0.0, high=1.0, default=0.0),
                        "intermediate_value_score": clamp_float(item.get("success_rate", 0.0), low=0.0, high=1.0, default=0.0),
                        "avg_latency_ms": 0.0,
                        "source_attempt_ids": [],
                        "source_template_ids": [],
                    }
                )

        representative_variant = self._select_representative_variant(chain_variants)
        recommended_chain = self._repair_chain_dependencies(
            self._sanitize_chain(
                item.get("recommended_chain") or representative_variant.get("recommended_chain"),
                limit=8,
            ),
            query_pattern=normalized_query_pattern,
        )
        if not recommended_chain and chain_variants:
            recommended_chain = self._repair_chain_dependencies(
                self._sanitize_chain(chain_variants[0].get("recommended_chain"), limit=8),
                query_pattern=normalized_query_pattern,
            )

        support_count = max(0, int(item.get("support_count", item.get("successful_attempt_count", 0)) or 0))
        template = {
            "template_id": str(item.get("pattern_id", item.get("template_id", f"pattern_{idx}"))).strip() or f"pattern_{idx}",
            "pattern_id": str(item.get("pattern_id", item.get("template_id", f"pattern_{idx}"))).strip() or f"pattern_{idx}",
            "pattern_name": str(item.get("pattern_name", "Generic Retrieval Pattern") or "Generic Retrieval Pattern").strip(),
            "pattern_description": str(item.get("pattern_description", "") or "").strip(),
            "query_abstract": query_abstract,
            "query_pattern": normalized_query_pattern,
            "query_pattern_text": query_pattern_text,
            "recommended_chain": recommended_chain,
            "chain_variants": chain_variants,
            "representative_variant_id": str(
                item.get("representative_variant_id", representative_variant.get("variant_id", ""))
            ).strip(),
            "anti_patterns": [str(x).strip() for x in (item.get("anti_patterns") or []) if str(x).strip()],
            "chain_rationale": str(item.get("chain_rationale", "") or "").strip(),
            "chain_constraints": [str(x).strip() for x in (item.get("chain_constraints") or []) if str(x).strip()],
            "supported_answer_shapes": [str(x).strip() for x in (item.get("supported_answer_shapes") or []) if str(x).strip()],
            "source_question_ids": [str(x).strip() for x in (item.get("source_question_ids") or []) if str(x).strip()],
            "source_questions": list(item.get("source_questions") or []),
            "support_count": support_count,
            "success_rate": clamp_float(item.get("success_rate", 1.0), low=0.0, high=1.0, default=1.0),
            "score_supervision": self._score_supervision_mode(item),
            "success_summary": dict(item.get("success_summary") or {}),
        }
        template["_embedding"] = self._safe_embedding(item.get("pattern_embedding")) or self._embed_text(query_pattern_text)
        if template["support_count"] < self.min_template_support:
            return None
        return template

    def reload(self, *, force: bool = False) -> None:
        if not self.library_path or not os.path.exists(self.library_path):
            self._payload = {}
            self._templates = []
            self._mtime = None
            return
        mtime = os.path.getmtime(self.library_path)
        if not force and self._mtime is not None and abs(self._mtime - mtime) < 1e-6:
            return
        payload = load_json(self.library_path)
        if isinstance(payload, list):
            payload = {"templates": payload}
        if not isinstance(payload, dict):
            payload = {}
        templates_raw = payload.get("patterns") or payload.get("templates") or payload.get("clusters") or []
        templates: List[Dict[str, Any]] = []
        for idx, item in enumerate(templates_raw):
            row = self._sanitize_template(item, idx)
            if row is not None:
                templates.append(row)
        self._payload = payload
        self._templates = templates
        self._mtime = mtime

    def clear(self, *, aggregation_mode: str = "", dataset_name: str = "") -> Dict[str, Any]:
        payload = {
            "library_version": 3,
            "aggregation_mode": str(aggregation_mode or self._payload.get("aggregation_mode", "") or "").strip(),
            "dataset_name": str(dataset_name or self._payload.get("dataset_name", "") or "").strip(),
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "pattern_count": 0,
            "patterns": [],
            "template_count": 0,
            "templates": [],
        }
        if self.library_path:
            parent_dir = os.path.dirname(self.library_path)
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True)
            json_dump_atomic(self.library_path, payload)
        self._payload = payload
        self._templates = []
        self._mtime = os.path.getmtime(self.library_path) if self.library_path and os.path.exists(self.library_path) else None
        return payload

    def maybe_reload(self) -> None:
        if not self.library_path or not os.path.exists(self.library_path):
            self.reload(force=True)
            return
        mtime = os.path.getmtime(self.library_path)
        if self._mtime is None or abs(self._mtime - mtime) > 1e-6:
            self.reload(force=True)

    @staticmethod
    def _select_variant_for_query(row: Dict[str, Any], query_pattern: Dict[str, Any]) -> Dict[str, Any]:
        chain_variants = [item for item in (row.get("chain_variants") or []) if isinstance(item, dict)]
        if not chain_variants:
            return {}
        preferred_shape = str((query_pattern or {}).get("answer_shape", "") or "").strip().lower()
        ranked = sorted(
            chain_variants,
            key=lambda item: (
                float(item.get("success_rate", 0.0) or 0.0),
                float(item.get("intermediate_value_score", 0.0) or 0.0),
                -int(item.get("tool_count", 999) or 999),
                -float(item.get("chain_token_cost", 0.0) or 0.0),
                -float(item.get("avg_latency_ms", 0.0) or 0.0),
                int(item.get("support_count", 0) or 0),
                1 if preferred_shape == "list" and len(item.get("recommended_chain") or []) <= 2 else 0,
            ),
            reverse=True,
        )
        return ranked[0]

    def match_templates(self, *, query_pattern: Dict[str, Any], top_k: int = 3) -> List[Dict[str, Any]]:
        self.maybe_reload()
        if not self._templates:
            return []
        raw_query = str((query_pattern or {}).get("_raw_query", "") or (query_pattern or {}).get("raw_query", "") or "").strip()
        q_abstract = str((query_pattern or {}).get("query_abstract", "") or "").strip()
        q_match_text = q_abstract or StrategyQueryPatternExtractor.pattern_to_text(query_pattern or {})
        q_emb = self._embed_text(q_match_text)
        ranked: List[Dict[str, Any]] = []
        now_ts = time.time()
        for template in self._templates:
            sim = cosine_sim(q_emb, template.get("_embedding")) if q_emb and template.get("_embedding") else None
            tpl_pattern = template.get("query_pattern") if isinstance(template.get("query_pattern"), dict) else {}
            tpl_abstract = str(tpl_pattern.get("query_abstract", "") or template.get("query_abstract", "") or "").strip()
            tpl_match_text = self._pattern_match_text(tpl_pattern, query_abstract=tpl_abstract)
            match_overlap = self._text_overlap_score(q_match_text, tpl_match_text)
            abstract_overlap = self._text_overlap_score(q_abstract, tpl_abstract) if q_abstract and tpl_abstract else 0.0
            answer_shape_match = self._answer_shape_match(query_pattern or {}, tpl_pattern)
            problem_type_match = self._problem_type_match(query_pattern or {}, tpl_pattern)
            target_category_match = self._target_category_match(query_pattern or {}, tpl_pattern)
            emb_score = clamp_float(sim, low=0.0, high=1.0, default=0.0) if sim is not None else None
            raw_success_rate = clamp_float(template.get("success_rate", 0.0), low=0.0, high=1.0, default=0.0)
            success_rate = self.effective_success_rate_for_matching(template)
            support_count = max(0, int(template.get("support_count", 0) or 0))
            support_factor = clamp_float(float(support_count) / 5.0, low=0.0, high=1.0, default=0.0)
            quality_factor = self.quality_factor_for_matching(template)
            source_questions = template.get("source_questions") or []
            source_question_match = 0.0
            best_source_question = ""
            if raw_query and isinstance(source_questions, list):
                for item in source_questions:
                    if not isinstance(item, dict):
                        continue
                    candidate_question = str(item.get("question", "") or "").strip()
                    if not candidate_question:
                        continue
                    score_q = self._question_similarity(raw_query, candidate_question)
                    if score_q > source_question_match:
                        source_question_match = score_q
                        best_source_question = candidate_question
            trusted_source_match = source_question_match * (0.25 + 0.75 * quality_factor)

            score = 0.0
            if emb_score is not None:
                score += 0.38 * emb_score
                score += 0.16 * match_overlap
                score += 0.10 * abstract_overlap
                score += 0.10 * problem_type_match
                score += 0.03 * target_category_match
                score += 0.06 * answer_shape_match
                score += 0.09 * trusted_source_match
                score += 0.05 * success_rate
                score += 0.03 * support_factor
            else:
                score += 0.30 * match_overlap
                score += 0.18 * abstract_overlap
                score += 0.18 * problem_type_match
                score += 0.06 * target_category_match
                score += 0.10 * answer_shape_match
                score += 0.10 * trusted_source_match
                score += 0.05 * success_rate
                score += 0.03 * support_factor
            if source_question_match >= 0.92 and quality_factor >= 0.6:
                score = max(score, 0.93 + 0.04 * success_rate + 0.03 * support_factor)
            elif source_question_match >= 0.82:
                score += 0.08 * trusted_source_match
            if self._has_cross_intent_conflict(query_pattern or {}, tpl_pattern):
                score *= 0.45

            chosen_variant = self._select_variant_for_query(template, query_pattern or {})
            recommended_chain = self._sanitize_chain(chosen_variant.get("recommended_chain") or template.get("recommended_chain"), limit=8)

            row = {k: v for k, v in template.items() if not str(k).startswith("_")}
            row["recommended_chain"] = recommended_chain
            row["selected_variant"] = chosen_variant
            row["similarity"] = round(score, 6)
            row["embedding_similarity"] = round(emb_score, 6) if emb_score is not None else None
            row["match_text_overlap"] = round(match_overlap, 6)
            row["abstract_overlap"] = round(abstract_overlap, 6)
            row["answer_shape_match"] = round(answer_shape_match, 6)
            row["problem_type_match"] = round(problem_type_match, 6)
            row["target_category_match"] = round(target_category_match, 6)
            row["source_question_match"] = round(source_question_match, 6)
            row["trusted_source_match"] = round(trusted_source_match, 6)
            row["best_source_question"] = best_source_question
            row["support_factor"] = round(support_factor, 6)
            row["quality_factor"] = round(quality_factor, 6)
            row["raw_success_rate"] = round(raw_success_rate, 6)
            row["effective_success_rate"] = round(success_rate, 6)
            row["score_supervision"] = self._score_supervision_mode(template)
            row["cross_intent_conflict"] = self._has_cross_intent_conflict(query_pattern or {}, tpl_pattern)
            row["query_match_text"] = q_match_text
            row["template_match_text"] = tpl_match_text
            row["retrieved_ts"] = now_ts
            ranked.append(row)
        ranked.sort(
            key=lambda item: (
                float(item.get("similarity", 0.0)),
                float(item.get("trusted_source_match", 0.0)),
                float(item.get("effective_success_rate", 0.0)),
                int(item.get("support_count", 0)),
            ),
            reverse=True,
        )
        return ranked[: max(1, int(top_k or 1))]

    @staticmethod
    def build_routing_hint(patterns: List[Dict[str, Any]], *, max_hint_lines: int = 6) -> str:
        if not patterns:
            return ""
        lines = [
            "Routing Memory (candidate retrieval strategies):",
            "Use at most one template as a soft prior.",
            "If none of the templates clearly fits the current question, ignore all of them.",
            "Do not combine multiple weak templates into one plan.",
        ]
        budget = max(2, int(max_hint_lines or 6) - 1)
        for idx, item in enumerate(patterns, start=1):
            if len(lines) >= budget:
                break
            chain = [str(x).strip() for x in (item.get("recommended_chain") or []) if str(x).strip()]
            chain_text = " -> ".join(chain) if chain else "adaptive retrieval chain"
            name = str(item.get("pattern_name", "Generic Retrieval Pattern") or "Generic Retrieval Pattern").strip()
            desc = str(item.get("pattern_description", "") or "").strip()
            support = int(item.get("support_count", 0) or 0)
            if desc:
                lines.append(
                    f"{idx}. {name}: initial guess `{chain_text}`. "
                    f"{StrategyTemplateLibrary._clip_text(desc, limit=180)} (support={support})."
                )
            else:
                lines.append(f"{idx}. {name}: initial guess `{chain_text}` (support={support}).")
            anti_patterns = [StrategyTemplateLibrary._clip_text(x, limit=120) for x in (item.get("anti_patterns") or []) if str(x).strip()]
            if anti_patterns and len(lines) < budget:
                lines.append(f"   Avoid: {' ; '.join(anti_patterns[:2])}")
        lines.append(
            "Treat these as optional hypotheses only. Final tool choice must be verified by the current question and retrieved evidence."
        )
        return "\n".join(lines)

    def diagnostics(self) -> Dict[str, Any]:
        self.maybe_reload()
        return {
            "library_path": self.library_path,
            "pattern_count": len(self._templates),
            "template_count": len(self._templates),
            "dataset_name": self._payload.get("dataset_name", ""),
            "aggregation_mode": self._payload.get("aggregation_mode", ""),
            "pattern_ids": [str(item.get("pattern_id", "")) for item in self._templates],
            "template_ids": [str(item.get("template_id", "")) for item in self._templates],
        }

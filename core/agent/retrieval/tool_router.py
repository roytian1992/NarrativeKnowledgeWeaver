from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from core.agent.retrieval.llm_router import StableLLMRetrievalRouter
from core.agent.retrieval.router_schema import (
    deterministic_parse,
    normalize_answer_shape,
    normalize_evidence_need,
    normalize_question_type,
    unique_names,
)
from core.agent.retrieval.tool_capability_profile import (
    family_from_capabilities,
    infer_capabilities,
)
from core.agent.retrieval.tool_aliases import normalize_memory_tool_name
from core.agent.retrieval.tool_family_registry import (
    TOOL_FAMILY_LABELS,
    TOOL_FAMILY_MEMBERS,
    family_cards_text,
    family_for_tool,
)
from core.agent.retrieval.tool_routing_heuristics import (
    heuristic_tool_boosts,
    preferred_core_tools,
    tool_parameters_text,
    tool_stage,
)
from core.utils.general_utils import token_jaccard_overlap


logger = logging.getLogger(__name__)

QUESTION_TYPE_KEYWORDS: Dict[str, List[str]] = {
    "exact_fact": ["what was", "what is", "who is", "where", "when", "which", "name of"],
    "attitude_or_state": ["attitude", "feel", "thinks", "think about", "internal dilemma", "why does", "why is", "seem"],
    "implication_or_inference": ["imply", "implied", "most likely", "warning", "suggest", "significance", "why", "dilemma"],
    "relationship": ["relationship", "feel about", "opinion about", "between"],
    "chronology": ["before", "after", "timeline", "first", "second", "year", "age", "how long"],
    "section_localization": ["which section", "which scene", "which chapter", "where in the story", "section"],
}


class RetrievalToolRouter:
    def __init__(
        self,
        *,
        config: Any,
        tool_metadata_provider: Any = None,
        router_llm: Any = None,
    ) -> None:
        self.config = config
        self.tool_metadata_provider = tool_metadata_provider
        self.router_llm = router_llm
        sm_cfg = getattr(config, "strategy_memory", None)
        self.router_mode = str(getattr(sm_cfg, "runtime_router_mode", "branching") or "branching").strip().lower()
        if self.router_mode == "qwen_like":
            logger.warning("runtime_router_mode=qwen_like is deprecated; falling back to branching")
            self.router_mode = "branching"
        self.initial_tool_limit = max(2, int(getattr(sm_cfg, "runtime_router_initial_tool_limit", 6) or 6))
        self.escalation_tool_limit = max(self.initial_tool_limit, int(getattr(sm_cfg, "runtime_router_escalation_tool_limit", 10) or 10))
        self.disable_heuristic_router = (
            str(os.environ.get("NKW_DISABLE_HEURISTIC_ROUTER", "") or "").strip().lower()
            in {"1", "true", "yes", "on"}
        )
        self._last_route_plan: Dict[str, Any] = {}
        self._llm_route_cache: Dict[str, Dict[str, Any]] = {}
        self._stable_router = StableLLMRetrievalRouter(router_llm=router_llm)

    @staticmethod
    def _rotate_names(items: List[str], offset: int) -> List[str]:
        names = unique_names(items, limit=None)
        if len(names) <= 1:
            return names
        idx = max(0, int(offset or 0)) % len(names)
        return names[idx:] + names[:idx]

    @staticmethod
    def _memory_query_pattern(memory_ctx: Dict[str, Any]) -> Dict[str, Any]:
        pattern = memory_ctx.get("query_pattern") if isinstance(memory_ctx, dict) else {}
        return pattern if isinstance(pattern, dict) else {}

    def _resolve_tool_profile(self, tool: Any) -> Tuple[str, str, str]:
        name = str(getattr(tool, "name", "") or "").strip()
        description = str(getattr(tool, "description", "") or "").strip()
        params_text = tool_parameters_text(tool)
        provider = self.tool_metadata_provider
        if provider is None or not name:
            return name, description, params_text
        meta = provider.resolve_tool_metadata(
            name,
            fallback_description=description,
            fallback_parameters=getattr(tool, "parameters", []) or [],
        )
        meta_params = meta.get("parameters") if isinstance(meta.get("parameters"), list) else []
        meta_tool = type("MetaTool", (), {})()
        setattr(meta_tool, "parameters", meta_params)
        return (
            str(meta.get("name") or name).strip(),
            str(meta.get("description") or description).strip(),
            tool_parameters_text(meta_tool),
        )

    def _resolve_parameter_names(self, tool: Any) -> List[str]:
        name = str(getattr(tool, "name", "") or "").strip()
        provider = self.tool_metadata_provider
        params = None
        if provider is not None and name:
            meta = provider.resolve_tool_metadata(
                name,
                fallback_description=str(getattr(tool, "description", "") or "").strip(),
                fallback_parameters=getattr(tool, "parameters", []) or [],
            )
            if isinstance(meta.get("parameters"), list):
                params = meta.get("parameters") or []
        if params is None:
            params = getattr(tool, "parameters", []) or []
        return [
            str((item or {}).get("name") or "").strip()
            for item in params
            if isinstance(item, dict) and str((item or {}).get("name") or "").strip()
        ]

    @staticmethod
    def preferred_tools_from_memory(memory_ctx: Dict[str, Any]) -> List[str]:
        preferred: List[str] = []
        for row in memory_ctx.get("patterns") or []:
            if not isinstance(row, dict):
                continue
            for name in row.get("recommended_chain") or []:
                tool_name = normalize_memory_tool_name(name)
                if tool_name and tool_name not in preferred:
                    preferred.append(tool_name)
        return preferred

    def get_last_route_plan(self) -> Dict[str, Any]:
        return dict(self._last_route_plan or {})

    @staticmethod
    def _active_tool_names(tools: List[Any]) -> List[str]:
        names: List[str] = []
        for tool in tools or []:
            name = str(getattr(tool, "name", "") or "").strip()
            if name and name not in names:
                names.append(name)
        return names

    def _sanitize_last_route_plan(self, *, tools: List[Any]) -> None:
        plan = self._last_route_plan if isinstance(self._last_route_plan, dict) else {}
        if not plan:
            return
        active_names = set(self._active_tool_names(tools))
        if not active_names:
            self._last_route_plan = {}
            return

        def filter_names(values: Any) -> List[str]:
            out: List[str] = []
            for raw in values or []:
                name = str(raw or "").strip()
                if name and name in active_names and name not in out:
                    out.append(name)
            return out

        sanitized = dict(plan)
        for key in ("initial_tools", "escalation_tools", "shared_core_tools"):
            if key in sanitized:
                sanitized[key] = filter_names(sanitized.get(key))

        candidate_branches = []
        for branch in plan.get("candidate_branches") or []:
            if not isinstance(branch, dict):
                continue
            branch_tools = filter_names(branch.get("tools"))
            if not branch_tools:
                continue
            row = dict(branch)
            row["tools"] = branch_tools
            candidate_branches.append(row)
        if "candidate_branches" in sanitized:
            sanitized["candidate_branches"] = candidate_branches

        selected_branch = sanitized.get("selected_branch")
        if isinstance(selected_branch, dict):
            branch_tools = filter_names(selected_branch.get("tools"))
            if branch_tools:
                row = dict(selected_branch)
                row["tools"] = branch_tools
                sanitized["selected_branch"] = row
            else:
                sanitized.pop("selected_branch", None)

        stages = []
        for stage in plan.get("stages") or []:
            stage_names = filter_names(stage)
            if stage_names:
                stages.append(stage_names)
        if "stages" in sanitized:
            sanitized["stages"] = stages

        self._last_route_plan = sanitized

    @staticmethod
    def _normalized_query(query: str) -> str:
        text = str(query or "").strip()
        if not text:
            return ""
        if "Mandatory retrieval guard:" in text:
            text = text.split("Mandatory retrieval guard:", 1)[0].strip()
        if "Question:" in text:
            text = text.split("Question:", 1)[1].strip()
        return text

    @classmethod
    def _question_type_hint(cls, query: str) -> str:
        lowered = cls._normalized_query(query).lower()
        for label, needles in QUESTION_TYPE_KEYWORDS.items():
            if any(needle in lowered for needle in needles):
                return label
        return "unknown"

    @classmethod
    def _mcq_profile(
        cls,
        *,
        query: str,
        parse: Dict[str, Any],
        question_type: str,
        answer_shape: str,
        evidence_need: str,
    ) -> Dict[str, Any]:
        normalized = cls._normalized_query(query)
        lowered = normalized.lower()
        is_mcq = (
            bool(parse.get("is_mcq"))
            or answer_shape == "mcq_option"
            or "choices:" in lowered
            or "options:" in lowered
            or "(a)" in lowered
        )
        has_negation = any(
            cue in lowered
            for cue in (
                "which is not",
                "which was not",
                "which did not",
                "which does not",
                "which cannot",
                "except",
                "least likely",
                "least supported",
                "least accurate",
                "incorrect",
                "false",
                "not true",
                "not supported",
                "not mentioned",
            )
        )
        needs_narrative_compare = (
            question_type in {"attitude_or_state", "implication_or_inference", "relationship", "chronology"}
            or evidence_need in {"local_plus_narrative", "chronology"}
            or any(
                cue in lowered
                for cue in ("why", "attitude", "feel", "imply", "most likely", "suggest", "significance")
            )
        )
        needs_choice_tool_initial = bool(is_mcq and has_negation)
        needs_choice_tool_stage1 = bool(is_mcq and (has_negation or needs_narrative_compare))
        return {
            "is_mcq": bool(is_mcq),
            "has_negation": bool(has_negation),
            "needs_narrative_compare": bool(needs_narrative_compare),
            "needs_choice_tool_initial": bool(needs_choice_tool_initial),
            "needs_choice_tool_stage1": bool(needs_choice_tool_stage1),
        }

    @staticmethod
    def _family_for_tool(tool_name: str) -> str:
        return family_for_tool(tool_name)

    @staticmethod
    def _normalize_capability_names(values: List[Any]) -> List[str]:
        alias_map = {
            "entity grounding": "entity_grounding",
            "entity_grounding": "entity_grounding",
            "entity relation": "entity_relation",
            "entity relation / interaction": "entity_relation",
            "entity_relation": "entity_relation",
            "passage evidence": "passage_evidence",
            "passage evidence extraction": "passage_evidence",
            "passage_evidence": "passage_evidence",
            "sentence retrieval": "sentence_retrieval",
            "sentence-level retrieval": "sentence_retrieval",
            "sentence_retrieval": "sentence_retrieval",
            "document fetch": "document_fetch",
            "document fetch / follow-up reading": "document_fetch",
            "document_fetch": "document_fetch",
            "lexical match": "lexical_match",
            "lexical matching": "lexical_match",
            "lexical_match": "lexical_match",
            "option comparison": "option_comparison",
            "option_comparison": "option_comparison",
            "narrative aggregation": "narrative_aggregation",
            "narrative_aggregation": "narrative_aggregation",
            "chronology reasoning": "chronology_reasoning",
            "chronology_reasoning": "chronology_reasoning",
            "section / document localization": "localization",
            "localization": "localization",
            "follow-up refinement": "followup_refinement",
            "followup refinement": "followup_refinement",
            "followup_refinement": "followup_refinement",
        }
        normalized: List[str] = []
        for raw in values or []:
            key = str(raw or "").strip().lower()
            if not key:
                continue
            item = alias_map.get(key, key.replace(" ", "_").replace("-", "_"))
            if item not in normalized:
                normalized.append(item)
        return normalized

    @staticmethod
    def _extract_json_object(raw: str) -> Optional[Dict[str, Any]]:
        text = str(raw or "").strip()
        if not text:
            return None
        candidates = [text]
        fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
        candidates.extend(fenced)
        start = text.find("{")
        end = text.rfind("}")
        if 0 <= start < end:
            candidates.append(text[start : end + 1])
        for item in candidates:
            try:
                payload = json.loads(item)
            except Exception:
                continue
            if isinstance(payload, dict):
                return payload
        return None

    def _tool_cards(self, *, tools: List[Any]) -> List[Dict[str, str]]:
        rows: List[Dict[str, str]] = []
        seen: set[str] = set()
        for tool in tools:
            raw_name = str(getattr(tool, "name", "") or "").strip()
            if not raw_name or raw_name in seen:
                continue
            name, description, params_text = self._resolve_tool_profile(tool)
            stage = tool_stage(name or raw_name)
            if stage == "internal_only":
                continue
            param_names = self._resolve_parameter_names(tool)
            capabilities = infer_capabilities(
                tool_name=raw_name,
                description=description,
                parameter_names=param_names,
            )
            readiness = "followup" if "document_ids" in {str(x or "").strip() for x in param_names} else "direct"
            seen.add(raw_name)
            rows.append(
                {
                    "name": raw_name,
                    "family": family_from_capabilities(capabilities, fallback_family=self._family_for_tool(raw_name)),
                    "description": description,
                    "parameters": params_text,
                    "capabilities": capabilities,
                    "readiness": readiness,
                    "stage": stage,
                }
            )
        return rows

    @staticmethod
    def _tool_cards_text(cards: List[Dict[str, str]]) -> str:
        return family_cards_text(cards)

    def _llm_route_plan(
        self,
        *,
        query: str,
        memory_ctx: Dict[str, Any],
        tools: List[Any],
    ) -> Optional[Dict[str, Any]]:
        if self.router_llm is None:
            return None
        normalized_query = self._normalized_query(query)
        cached = self._llm_route_cache.get(normalized_query)
        if isinstance(cached, dict) and cached:
            return dict(cached)
        cards = self._tool_cards(tools=tools)
        if not cards:
            return None
        preferred_tools = self.preferred_tools_from_memory(memory_ctx)
        query_pattern = self._memory_query_pattern(memory_ctx)
        prompt = "\n".join(
            [
                "You are a retrieval routing planner. Do not answer the question.",
                "Return JSON only.",
                "",
                "Goal:",
                "- Choose a SMALL but SUFFICIENT retrieval tool set.",
                "- Propose multiple plausible branches for diversified repeated runs.",
                "- For attitude, implication, dilemma, motive, or internal-state questions, include at least one narrative_reasoning branch if such tools are available.",
                "- Prefer local textual evidence tools for exact facts, but do not suppress narrative tools when narrative aggregation is needed.",
                "- Avoid redundant branches that differ only trivially.",
                "",
                "JSON schema:",
                '{',
                '  "question_type": "exact_fact|relationship|attitude_or_state|implication_or_inference|chronology|section_localization|unknown",',
                '  "evidence_need": "local|local_plus_narrative|entity_plus_local|chronology|mixed",',
                '  "shared_core_tools": ["tool_a", "tool_b"],',
                '  "candidate_branches": [',
                '    {"branch_id": "branch_name", "intent": "short text", "tools": ["tool_a", "tool_b"], "score": 0.0}',
                "  ],",
                '  "escalation_tools": ["tool_x", "tool_y"]',
                '}',
                "",
                "Rules:",
                "- Use only tool names from the provided list.",
                "- shared_core_tools: 1-3 tools.",
                "- candidate_branches: 2-4 branches when possible; otherwise 1 branch.",
                "- Each branch should usually contain 2-5 tools.",
                "- Branches should represent meaningfully different retrieval routes.",
                "- score should be between 0 and 1 and reflect expected usefulness.",
                "",
                f"Question:\n{normalized_query}",
                "",
                f"Question type hint: {self._question_type_hint(normalized_query)}",
                f"Query pattern hint: {json.dumps(query_pattern, ensure_ascii=False)}",
                f"Preferred tools from memory: {json.dumps(preferred_tools, ensure_ascii=False)}",
                "",
                "Available tools:",
                self._tool_cards_text(cards),
            ]
        ).strip()
        try:
            result = self.router_llm.run([{"role": "user", "content": prompt}])
            content = ""
            if isinstance(result, list) and result:
                content = str((result[0] or {}).get("content", "") or "").strip()
            payload = self._extract_json_object(content)
        except Exception as exc:
            logger.warning("LLM tool routing failed: query=%s err=%s", normalized_query, exc)
            return None
        if not isinstance(payload, dict):
            return None
        normalized_plan = self._normalize_route_plan(payload=payload, cards=cards, query=normalized_query)
        if normalized_plan:
            self._llm_route_cache[normalized_query] = dict(normalized_plan)
        return normalized_plan

    def _normalize_route_plan(
        self,
        *,
        payload: Dict[str, Any],
        cards: List[Dict[str, str]],
        query: str,
    ) -> Optional[Dict[str, Any]]:
        valid_tools = {str(card.get("name", "") or "").strip() for card in cards if str(card.get("name", "") or "").strip()}
        shared_core_tools = [
            name for name in (payload.get("shared_core_tools") or [])
            if isinstance(name, str) and name.strip() in valid_tools
        ]
        candidate_branches_raw = payload.get("candidate_branches") if isinstance(payload.get("candidate_branches"), list) else []
        candidate_branches: List[Dict[str, Any]] = []
        for idx, branch in enumerate(candidate_branches_raw, start=1):
            if not isinstance(branch, dict):
                continue
            branch_tools: List[str] = []
            for raw_name in branch.get("tools") or []:
                name = str(raw_name or "").strip()
                if name and name in valid_tools and name not in branch_tools:
                    branch_tools.append(name)
            if not branch_tools:
                continue
            branch_id = str(branch.get("branch_id", "") or f"branch_{idx}").strip() or f"branch_{idx}"
            candidate_branches.append(
                {
                    "branch_id": branch_id,
                    "intent": str(branch.get("intent", "") or "").strip(),
                    "tools": branch_tools[:5],
                    "score": max(0.0, min(1.0, float(branch.get("score", 0.0) or 0.0))),
                }
            )
        escalation_tools = [
            name for name in (payload.get("escalation_tools") or [])
            if isinstance(name, str) and name.strip() in valid_tools
        ]
        if not candidate_branches:
            return None
        candidate_branches.sort(key=lambda item: (float(item.get("score", 0.0) or 0.0), -len(item.get("tools") or []), item.get("branch_id", "")), reverse=True)
        if self._question_type_hint(query) in {"attitude_or_state", "implication_or_inference"}:
            has_narrative = any(
                any(self._family_for_tool(tool_name) == "narrative_reasoning" for tool_name in (branch.get("tools") or []))
                for branch in candidate_branches
            )
            if not has_narrative:
                fallback_tools = [name for name in ["section_evidence_search", "narrative_hierarchical_search", "entity_event_trace_search"] if name in valid_tools]
                if fallback_tools:
                    candidate_branches.append(
                        {
                            "branch_id": "narrative_fallback",
                            "intent": "use storyline or event aggregation for inference-heavy questions",
                            "tools": fallback_tools[:4],
                            "score": 0.66,
                        }
                    )
        if not shared_core_tools:
            for name in ["bm25_search_docs", "section_evidence_search", "vdb_search_sentences"]:
                if name in valid_tools and name not in shared_core_tools:
                    shared_core_tools.append(name)
                if len(shared_core_tools) >= 2:
                    break
        if not escalation_tools:
            for name in ["vdb_get_docs_by_document_ids", "narrative_hierarchical_search", "entity_event_trace_search", "fact_timeline_resolution_search"]:
                if name in valid_tools and name not in escalation_tools:
                    escalation_tools.append(name)
        return {
            "question_type": str(payload.get("question_type", "") or self._question_type_hint(query)).strip() or "unknown",
            "evidence_need": str(payload.get("evidence_need", "") or "mixed").strip() or "mixed",
            "shared_core_tools": shared_core_tools[:3],
            "candidate_branches": candidate_branches[:4],
            "escalation_tools": escalation_tools[:4],
            "router_mode": "llm",
        }

    @staticmethod
    def _desired_capabilities(
        *,
        question_type: str,
        evidence_need: str,
        has_entity_anchor: bool,
        answer_shape: str,
    ) -> Tuple[List[str], List[str]]:
        initial_caps: List[str] = []
        escalation_caps: List[str] = []
        if has_entity_anchor:
            initial_caps.append("entity_grounding")
        if answer_shape == "mcq_option":
            initial_caps.append("option_comparison")
        initial_caps.append("passage_evidence")
        if question_type == "section_localization":
            initial_caps.append("localization")
        if question_type in {"exact_fact", "unknown"}:
            initial_caps.append("sentence_retrieval")
            initial_caps.append("lexical_match")
        if evidence_need in {"local_plus_narrative", "chronology"} or question_type in {
            "attitude_or_state",
            "implication_or_inference",
            "chronology",
        }:
            initial_caps.append("narrative_aggregation")
        if question_type == "chronology":
            initial_caps.append("chronology_reasoning")
        if evidence_need == "entity_plus_local":
            initial_caps.append("entity_relation")
        if not has_entity_anchor:
            escalation_caps.append("entity_grounding")
        escalation_caps.extend(
            [
                "entity_relation",
                "narrative_aggregation",
                "chronology_reasoning",
                "localization",
                "document_fetch",
                "sentence_retrieval",
                "lexical_match",
                "followup_refinement",
            ]
        )
        return unique_names(initial_caps), unique_names(escalation_caps)

    def _ranked_candidates(
        self,
        *,
        ranked_names: List[str],
        valid_tools: List[str],
        card_by_name: Dict[str, Dict[str, Any]],
        family: Optional[str] = None,
        capability: Optional[str] = None,
        readiness: Optional[str] = None,
    ) -> List[str]:
        valid_set = {str(name).strip() for name in valid_tools if str(name).strip()}
        candidates: List[str] = []
        for name in ranked_names:
            normalized = str(name or "").strip()
            if not normalized or normalized not in valid_set:
                continue
            card = card_by_name.get(normalized) or {}
            card_family = str(card.get("family", "") or self._family_for_tool(normalized)).strip()
            caps = {str(item or "").strip() for item in (card.get("capabilities") or []) if str(item or "").strip()}
            card_readiness = str(card.get("readiness", "direct") or "direct").strip() or "direct"
            if family and card_family != family:
                continue
            if capability and capability not in caps:
                continue
            if readiness and card_readiness != readiness:
                continue
            if normalized not in candidates:
                candidates.append(normalized)
        return candidates

    def _stable_priority_tool_order(
        self,
        *,
        question_type: str,
        evidence_need: str,
        answer_shape: str,
        has_entity_anchor: bool,
    ) -> List[str]:
        priority: List[str] = []

        def add(*names: str) -> None:
            for raw_name in names:
                name = str(raw_name or "").strip()
                if name and name not in priority:
                    priority.append(name)

        # Old strong baseline behavior: start from section evidence, then keep
        # a narrative path visible for most multiple-choice questions.
        if answer_shape == "mcq_option":
            add("section_evidence_search")
            add("narrative_hierarchical_search")
            add("bm25_search_docs")
        else:
            add("section_evidence_search", "bm25_search_docs")

        if question_type in {"attitude_or_state", "implication_or_inference", "chronology"}:
            add("narrative_hierarchical_search", "entity_event_trace_search")
        if evidence_need in {"local_plus_narrative", "chronology"}:
            add("narrative_hierarchical_search")
        if question_type == "chronology" or evidence_need == "chronology":
            add("fact_timeline_resolution_search")

        add("vdb_search_sentences")

        if has_entity_anchor:
            add("retrieve_entity_by_name")

        add("search_sections")

        return priority

    def _stable_priority_escalation_order(
        self,
        *,
        question_type: str,
        evidence_need: str,
        answer_shape: str,
        has_entity_anchor: bool,
    ) -> List[str]:
        priority: List[str] = []

        def add(*names: str) -> None:
            for raw_name in names:
                name = str(raw_name or "").strip()
                if name and name not in priority:
                    priority.append(name)

        add("vdb_get_docs_by_document_ids")

        if answer_shape == "mcq_option":
            add("choice_grounded_evidence_search")

        if question_type in {"attitude_or_state", "implication_or_inference"} or evidence_need == "local_plus_narrative":
            add("entity_event_trace_search", "narrative_hierarchical_search")
        if question_type == "chronology" or evidence_need == "chronology":
            add("fact_timeline_resolution_search")

        if has_entity_anchor:
            add("get_entity_sections")

        add("search_related_content", "vdb_search_hierdocs")
        return priority

    def _augment_stable_initial_tools(
        self,
        *,
        initial_tools: List[str],
        desired_capabilities: List[str],
        question_type: str,
        evidence_need: str,
        answer_shape: str,
        has_entity_anchor: bool,
        valid_tools: List[str],
        ranked_names: List[str],
        card_by_name: Dict[str, Dict[str, Any]],
    ) -> List[str]:
        valid_set = {str(name).strip() for name in valid_tools if str(name).strip()}
        desired = {str(item or "").strip() for item in (desired_capabilities or []) if str(item or "").strip()}
        ranked_direct = self._ranked_candidates(
            ranked_names=ranked_names,
            valid_tools=valid_tools,
            card_by_name=card_by_name,
            readiness="direct",
        )
        ranked_any = self._ranked_candidates(
            ranked_names=ranked_names,
            valid_tools=valid_tools,
            card_by_name=card_by_name,
            readiness=None,
        )

        initial: List[str] = []

        def add(name: str) -> None:
            normalized = str(name or "").strip()
            if not normalized or normalized not in valid_set or normalized in initial:
                return
            if tool_stage(normalized) in {"exploratory", "internal_only"}:
                return
            initial.append(normalized)

        def add_first(candidates: List[str]) -> None:
            for candidate in candidates:
                if len(initial) >= self.initial_tool_limit:
                    return
                add(candidate)
                if candidate in initial:
                    return

        for name in self._stable_priority_tool_order(
            question_type=question_type,
            evidence_need=evidence_need,
            answer_shape=answer_shape,
            has_entity_anchor=has_entity_anchor,
        ):
            add(name)

        for name in unique_names(initial_tools, valid=valid_tools, limit=None):
            readiness = str((card_by_name.get(name) or {}).get("readiness", "direct") or "direct").strip() or "direct"
            if readiness == "followup":
                continue
            add(name)

        # For anchored questions, early entity grounding is usually useful.
        if has_entity_anchor:
            if "retrieve_entity_by_name" in valid_set:
                add("retrieve_entity_by_name")
            else:
                add_first(
                    self._ranked_candidates(
                        ranked_names=ranked_names,
                        valid_tools=valid_tools,
                        card_by_name=card_by_name,
                        capability="entity_grounding",
                        readiness="direct",
                    )
                )

        # Keep one direct local evidence tool in the first wave.
        add_first(
            self._ranked_candidates(
                ranked_names=ranked_names,
                valid_tools=valid_tools,
                card_by_name=card_by_name,
                family="local_evidence",
                readiness="direct",
            )
        )

        need_narrative = (
            evidence_need in {"local_plus_narrative", "chronology"}
            or question_type in {"attitude_or_state", "implication_or_inference", "chronology"}
            or "narrative_aggregation" in desired
            or "chronology_reasoning" in desired
        )
        if need_narrative:
            add_first(
                self._ranked_candidates(
                    ranked_names=ranked_names,
                    valid_tools=valid_tools,
                    card_by_name=card_by_name,
                    family="narrative_reasoning",
                    readiness="direct",
                )
            )

        if question_type == "section_localization" or "localization" in desired:
            add_first(
                self._ranked_candidates(
                    ranked_names=ranked_names,
                    valid_tools=valid_tools,
                    card_by_name=card_by_name,
                    family="structural_lookup",
                    readiness="direct",
                )
            )

        if "entity_grounding" in desired and not any(
            str((card_by_name.get(name) or {}).get("family", "") or self._family_for_tool(name)).strip() == "entity_relation"
            for name in initial
        ):
            add_first(
                self._ranked_candidates(
                    ranked_names=ranked_names,
                    valid_tools=valid_tools,
                    card_by_name=card_by_name,
                    family="entity_relation",
                    readiness="direct",
                )
            )

        for name in ranked_direct:
            if len(initial) >= self.initial_tool_limit:
                break
            add(name)

        for name in ranked_any:
            if len(initial) >= self.initial_tool_limit:
                break
            add(name)

        return initial[: self.initial_tool_limit]

    def _augment_stable_escalation_tools(
        self,
        *,
        initial_tools: List[str],
        escalation_tools: List[str],
        desired_capabilities: List[str],
        question_type: str,
        evidence_need: str,
        answer_shape: str,
        has_entity_anchor: bool,
        valid_tools: List[str],
        ranked_names: List[str],
        card_by_name: Dict[str, Dict[str, Any]],
    ) -> List[str]:
        limit = max(1, self.escalation_tool_limit - len(initial_tools))
        valid_set = {str(name).strip() for name in valid_tools if str(name).strip()}
        desired = {str(item or "").strip() for item in (desired_capabilities or []) if str(item or "").strip()}
        escalation: List[str] = []

        def add(name: str) -> None:
            normalized = str(name or "").strip()
            if not normalized or normalized not in valid_set or normalized in escalation or normalized in initial_tools:
                return
            escalation.append(normalized)

        for name in self._stable_priority_escalation_order(
            question_type=question_type,
            evidence_need=evidence_need,
            answer_shape=answer_shape,
            has_entity_anchor=has_entity_anchor,
        ):
            add(name)

        for name in unique_names(escalation_tools, valid=valid_tools, limit=None):
            add(name)

        # Follow-up tools become more valuable after an initial search has produced anchors or document ids.
        for name in self._ranked_candidates(
            ranked_names=ranked_names,
            valid_tools=valid_tools,
            card_by_name=card_by_name,
            readiness="followup",
        ):
            if len(escalation) >= limit:
                break
            add(name)

        if question_type in {"attitude_or_state", "implication_or_inference"} or evidence_need == "local_plus_narrative":
            for name in ["narrative_hierarchical_search", "entity_event_trace_search"]:
                if len(escalation) >= limit:
                    break
                add(name)
        if question_type == "chronology" or evidence_need == "chronology" or "chronology_reasoning" in desired:
            add("fact_timeline_resolution_search")

        for family in ["entity_relation", "narrative_reasoning", "structural_lookup", "local_evidence"]:
            family_missing = all(
                str((card_by_name.get(name) or {}).get("family", "") or self._family_for_tool(name)).strip() != family
                for name in [*initial_tools, *escalation]
            )
            if not family_missing:
                continue
            for name in self._ranked_candidates(
                ranked_names=ranked_names,
                valid_tools=valid_tools,
                card_by_name=card_by_name,
                family=family,
                readiness=None,
            ):
                if len(escalation) >= limit:
                    break
                add(name)
                break

        for name in self._ranked_candidates(
            ranked_names=ranked_names,
            valid_tools=valid_tools,
            card_by_name=card_by_name,
            readiness=None,
        ):
            if len(escalation) >= limit:
                break
            add(name)
        return escalation[:limit]

    def _default_stable_tools(
        self,
        *,
        question_type: str,
        evidence_need: str,
        has_entity_anchor: bool,
        answer_shape: str,
        desired_capabilities: List[str],
        valid_tools: List[str],
        ranked_names: List[str],
        card_by_name: Dict[str, Dict[str, Any]],
    ) -> Tuple[List[str], List[str]]:
        initial_caps, escalation_caps = self._desired_capabilities(
            question_type=question_type,
            evidence_need=evidence_need,
            has_entity_anchor=has_entity_anchor,
            answer_shape=answer_shape,
        )
        if desired_capabilities:
            initial_caps = unique_names([*desired_capabilities, *initial_caps])
        ranked_unique = unique_names(ranked_names, valid=valid_tools, limit=None)
        capability_ranked: Dict[str, List[str]] = {}
        readiness_by_name: Dict[str, str] = {}
        for name in ranked_unique:
            card = card_by_name.get(name) or {}
            readiness_by_name[name] = str(card.get("readiness", "direct") or "direct").strip() or "direct"
            for capability in card.get("capabilities") or []:
                capability_ranked.setdefault(str(capability or "").strip(), []).append(name)

        initial: List[str] = []
        escalation: List[str] = []

        def add_candidates(target: List[str], candidates: List[str], *, limit: int) -> None:
            for name in unique_names(candidates, valid=valid_tools, limit=None):
                if name in target:
                    continue
                if target is initial and readiness_by_name.get(name) == "followup":
                    continue
                target.append(name)
                if len(target) >= limit:
                    return

        for capability in initial_caps:
            add_candidates(initial, capability_ranked.get(capability, []), limit=self.initial_tool_limit)
        add_candidates(initial, ranked_unique, limit=self.initial_tool_limit)

        remaining_ranked = [name for name in ranked_unique if name not in initial]
        for capability in escalation_caps:
            add_candidates(
                escalation,
                [name for name in capability_ranked.get(capability, []) if name not in initial],
                limit=max(1, self.escalation_tool_limit - len(initial)),
            )
        add_candidates(
            escalation,
            remaining_ranked,
            limit=max(1, self.escalation_tool_limit - len(initial)),
        )
        return initial, escalation

    def _normalize_stable_route_plan(
        self,
        *,
        payload: Optional[Dict[str, Any]],
        cards: List[Dict[str, str]],
        query: str,
        ranked_names: List[str],
    ) -> Dict[str, Any]:
        card_by_name: Dict[str, Dict[str, Any]] = {
            str(card.get("name", "") or "").strip(): dict(card)
            for card in cards
            if str(card.get("name", "") or "").strip()
        }
        valid_tools = [
            str(card.get("name", "") or "").strip()
            for card in cards
            if str(card.get("name", "") or "").strip()
        ]
        parse = deterministic_parse(query)
        payload = payload if isinstance(payload, dict) else {}
        question_type = normalize_question_type(payload.get("question_type"), fallback=parse["question_type"])
        answer_shape = normalize_answer_shape(
            payload.get("answer_shape"),
            fallback=parse["answer_shape"],
            query=query,
        )
        evidence_need = normalize_evidence_need(payload.get("evidence_need"), fallback=parse["evidence_need"])
        desired_capabilities = self._normalize_capability_names(
            unique_names(payload.get("desired_capabilities") or [], limit=6)
        )
        default_desired_capabilities, _ = self._desired_capabilities(
            question_type=question_type,
            evidence_need=evidence_need,
            has_entity_anchor=bool(parse.get("has_entity_anchor")),
            answer_shape=answer_shape,
        )
        default_initial, default_escalation = self._default_stable_tools(
            question_type=question_type,
            evidence_need=evidence_need,
            has_entity_anchor=bool(parse.get("has_entity_anchor")),
            answer_shape=answer_shape,
            desired_capabilities=desired_capabilities,
            valid_tools=valid_tools,
            ranked_names=ranked_names,
            card_by_name=card_by_name,
        )
        initial_tools = unique_names(
            payload.get("initial_tools") or default_initial,
            valid=valid_tools,
            limit=self.initial_tool_limit,
        )
        initial_tools = self._augment_stable_initial_tools(
            initial_tools=initial_tools,
            desired_capabilities=desired_capabilities or default_desired_capabilities,
            question_type=question_type,
            evidence_need=evidence_need,
            answer_shape=answer_shape,
            has_entity_anchor=bool(parse.get("has_entity_anchor")),
            valid_tools=valid_tools,
            ranked_names=ranked_names,
            card_by_name=card_by_name,
        )
        escalation_tools = unique_names(
            payload.get("escalation_tools") or default_escalation,
            valid=valid_tools,
            limit=max(1, self.escalation_tool_limit - len(initial_tools)),
        )
        escalation_tools = self._augment_stable_escalation_tools(
            initial_tools=initial_tools,
            escalation_tools=escalation_tools,
            desired_capabilities=desired_capabilities or default_desired_capabilities,
            question_type=question_type,
            evidence_need=evidence_need,
            answer_shape=answer_shape,
            has_entity_anchor=bool(parse.get("has_entity_anchor")),
            valid_tools=valid_tools,
            ranked_names=ranked_names,
            card_by_name=card_by_name,
        )
        if question_type in {"attitude_or_state", "implication_or_inference"}:
            if all(self._family_for_tool(name) != "narrative_reasoning" for name in initial_tools):
                initial_tools = unique_names(
                    [*initial_tools, "narrative_hierarchical_search"],
                    valid=valid_tools,
                    limit=self.initial_tool_limit,
                )
        if not initial_tools:
            initial_tools = unique_names(default_initial, valid=valid_tools, limit=self.initial_tool_limit)
        if not escalation_tools:
            escalation_tools = unique_names(default_escalation, valid=valid_tools, limit=max(1, self.escalation_tool_limit - len(initial_tools)))
        initial_clusters = unique_names(
            payload.get("initial_clusters") or [self._family_for_tool(name) for name in initial_tools],
            limit=4,
        )
        raw_parallel = payload.get("parallel_tools") if isinstance(payload.get("parallel_tools"), list) else []
        parallel_tools: List[List[str]] = []
        for group in raw_parallel:
            if not isinstance(group, list):
                continue
            names = unique_names(group, valid=valid_tools, limit=3)
            if len(names) >= 2:
                parallel_tools.append(names)
        if not parallel_tools:
            fallback_parallel = unique_names(
                (
                    ["retrieve_entity_by_name", "section_evidence_search"]
                    if bool(parse.get("has_entity_anchor"))
                    else ["bm25_search_docs", "section_evidence_search"]
                ),
                valid=valid_tools,
                limit=2,
            )
            if len(fallback_parallel) >= 2:
                parallel_tools.append(fallback_parallel)
        return {
            "router_mode": "stable",
            "question_type": question_type,
            "answer_shape": answer_shape,
            "evidence_need": evidence_need,
            "desired_capabilities": desired_capabilities or default_desired_capabilities,
            "initial_clusters": initial_clusters,
            "initial_tools": initial_tools,
            "escalation_tools": escalation_tools,
            "parallel_tools": parallel_tools[:2],
            "must_compare_options": bool(payload.get("must_compare_options", answer_shape == "mcq_option")),
            "reason": str(payload.get("reason", "") or "").strip(),
            "deterministic_parse": parse,
        }

    def _build_stable_execution_plan(
        self,
        *,
        query: str,
        memory_ctx: Dict[str, Any],
        tools: List[Any],
    ) -> List[List[Any]]:
        normalized_query = self._normalized_query(query)
        registry: Dict[str, Any] = {}
        for tool in tools:
            name = str(getattr(tool, "name", "") or "").strip()
            if name and name not in registry:
                registry[name] = tool
        cards = self._tool_cards(tools=tools)
        ranked_names = [
            name
            for _, name in self.score_tools_for_query(
                query=normalized_query,
                memory_ctx=memory_ctx,
                tools=tools,
            )
        ]
        payload = self._stable_router.route(query=normalized_query, tool_cards=cards)
        route_plan = self._normalize_stable_route_plan(
            payload=payload,
            cards=cards,
            query=normalized_query,
            ranked_names=ranked_names,
        )

        def names_to_tools(names: List[str]) -> List[Any]:
            return [registry[name] for name in names if name in registry]

        stage0_names = unique_names(route_plan.get("initial_tools") or [], valid=registry.keys(), limit=self.initial_tool_limit)
        stage1_names = unique_names(
            [*stage0_names, *(route_plan.get("escalation_tools") or [])],
            valid=registry.keys(),
            limit=self.escalation_tool_limit,
        )
        full_names = unique_names(
            [*stage1_names, *[name for name in registry.keys() if tool_stage(name) != "internal_only"]],
            valid=registry.keys(),
            limit=None,
        )
        plan: List[List[Any]] = []
        for subset in [names_to_tools(stage0_names), names_to_tools(stage1_names), names_to_tools(full_names)]:
            names = tuple(getattr(t, "name", "") for t in subset)
            if subset and all(tuple(getattr(x, "name", "") for x in s) != names for s in plan):
                plan.append(subset)
        self._last_route_plan = {
            **route_plan,
            "selected_branch_index": 0,
            "selected_branch": {
                "branch_id": "stable_initial",
                "intent": route_plan.get("reason", "") or "start from the structured initial tool set",
                "tools": stage0_names,
                "score": 1.0,
            },
            "stages": [[getattr(t, "name", "") for t in stage] for stage in plan],
            "query": normalized_query,
        }
        logger.info(
            "Stable tool routing plan built for query=%s initial=%s escalation=%s",
            normalized_query,
            stage0_names,
            route_plan.get("escalation_tools") or [],
        )
        return plan

    def _build_qwen_like_execution_plan(
        self,
        *,
        query: str,
        memory_ctx: Dict[str, Any],
        tools: List[Any],
    ) -> List[List[Any]]:
        normalized_query = self._normalized_query(query)
        registry: Dict[str, Any] = {}
        for tool in tools:
            name = str(getattr(tool, "name", "") or "").strip()
            if name and name not in registry:
                registry[name] = tool
        if not registry:
            return []

        cards = self._tool_cards(tools=tools)
        card_by_name: Dict[str, Dict[str, Any]] = {
            str(card.get("name", "") or "").strip(): dict(card)
            for card in cards
            if str(card.get("name", "") or "").strip()
        }
        valid_tools = list(registry.keys())
        ranked_names = [
            name
            for _, name in self.score_tools_for_query(
                query=normalized_query,
                memory_ctx=memory_ctx,
                tools=tools,
            )
            if name in registry
        ]
        parse = deterministic_parse(normalized_query)
        question_type = normalize_question_type(parse.get("question_type"), fallback="unknown")
        answer_shape = normalize_answer_shape(parse.get("answer_shape"), fallback="short_fact", query=normalized_query)
        evidence_need = normalize_evidence_need(parse.get("evidence_need"), fallback="mixed")
        has_entity_anchor = bool(parse.get("has_entity_anchor"))
        mcq_profile = self._mcq_profile(
            query=normalized_query,
            parse=parse,
            question_type=question_type,
            answer_shape=answer_shape,
            evidence_need=evidence_need,
        )
        is_mcq = bool(mcq_profile.get("is_mcq"))
        branch_index = max(0, int(memory_ctx.get("router_branch_index", 0) or 0))

        initial_limit = max(6, int(self.initial_tool_limit or 6))
        stage1_limit = max(initial_limit + 3, int(self.escalation_tool_limit or (initial_limit + 3)))
        initial_blocklist = {
            "lookup_document_ids_by_title",
            "lookup_titles_by_document_ids",
            "search_related_content",
            "search_dialogues",
            "search_interactions",
            "get_interactions_by_document_ids",
            "vdb_get_docs_by_document_ids",
        }
        stage1_blocklist = {
            "lookup_document_ids_by_title",
            "lookup_titles_by_document_ids",
        }

        ranked_direct = self._rotate_names(
            self._ranked_candidates(
                ranked_names=ranked_names,
                valid_tools=valid_tools,
                card_by_name=card_by_name,
                readiness="direct",
            ),
            branch_index,
        )
        ranked_followup = self._rotate_names(
            self._ranked_candidates(
                ranked_names=ranked_names,
                valid_tools=valid_tools,
                card_by_name=card_by_name,
                readiness="followup",
            ),
            branch_index,
        )
        local_direct = self._rotate_names(
            self._ranked_candidates(
                ranked_names=ranked_names,
                valid_tools=valid_tools,
                card_by_name=card_by_name,
                family="local_evidence",
                readiness="direct",
            ),
            branch_index,
        )
        narrative_direct = self._rotate_names(
            self._ranked_candidates(
                ranked_names=ranked_names,
                valid_tools=valid_tools,
                card_by_name=card_by_name,
                family="narrative_reasoning",
                readiness="direct",
            ),
            branch_index // 2,
        )
        entity_direct = self._rotate_names(
            self._ranked_candidates(
                ranked_names=ranked_names,
                valid_tools=valid_tools,
                card_by_name=card_by_name,
                family="entity_relation",
                readiness="direct",
            ),
            branch_index,
        )
        structural_direct = self._rotate_names(
            self._ranked_candidates(
                ranked_names=ranked_names,
                valid_tools=valid_tools,
                card_by_name=card_by_name,
                family="structural_lookup",
                readiness="direct",
            ),
            branch_index,
        )

        initial_names: List[str] = []
        need_narrative = (
            answer_shape == "mcq_option"
            or evidence_need in {"local_plus_narrative", "chronology"}
            or question_type in {"attitude_or_state", "implication_or_inference", "chronology"}
        )

        def add(name: str) -> None:
            normalized = str(name or "").strip()
            if not normalized or normalized not in registry or normalized in initial_names:
                return
            if normalized in initial_blocklist:
                return
            if tool_stage(normalized) in {"internal_only", "exploratory"}:
                return
            initial_names.append(normalized)

        def add_first(candidates: List[str], *, count: int = 1) -> None:
            added = 0
            for candidate in candidates:
                if len(initial_names) >= initial_limit or added >= max(1, int(count or 1)):
                    return
                before = len(initial_names)
                add(candidate)
                if len(initial_names) > before:
                    added += 1

        # Qwen-like behavior: weak routing, broad but not full tool subset.
        # Keep canonical evidence tools visible, and let the model self-decide inside the subset.
        for preferred in ["section_evidence_search", "bm25_search_docs", "vdb_search_sentences"]:
            add(preferred)
        if has_entity_anchor:
            add("retrieve_entity_by_name")
        if is_mcq and has_entity_anchor and question_type in {"exact_fact", "relationship"}:
            add("get_entity_sections")
        if need_narrative:
            add("narrative_hierarchical_search")
        if question_type == "chronology" or evidence_need == "chronology":
            add("fact_timeline_resolution_search")
        if mcq_profile.get("needs_choice_tool_initial"):
            add("choice_grounded_evidence_search")
        elif is_mcq and question_type == "section_localization":
            add("choice_grounded_evidence_search")
        if is_mcq and question_type == "relationship":
            add("entity_event_trace_search")
        if question_type == "section_localization":
            add("search_sections")

        add_first(local_direct, count=2)
        if has_entity_anchor:
            add_first(entity_direct, count=1)
        if need_narrative:
            add_first(narrative_direct, count=1)
        if is_mcq and not mcq_profile.get("needs_choice_tool_initial"):
            add_first(
                [name for name in local_direct if name not in {"section_evidence_search", "bm25_search_docs", "vdb_search_sentences"}],
                count=1,
            )

        if question_type == "section_localization":
            add_first(structural_direct, count=1)

        family_counts: Dict[str, int] = {}
        for name in list(initial_names):
            family_counts[self._family_for_tool(name)] = family_counts.get(self._family_for_tool(name), 0) + 1

        for name in ranked_direct:
            if len(initial_names) >= initial_limit:
                break
            family = self._family_for_tool(name)
            # Prefer broader family coverage first; only then add same-family redundancy.
            if family_counts.get(family, 0) >= 2 and family != "local_evidence":
                continue
            before = len(initial_names)
            add(name)
            if len(initial_names) > before:
                family_counts[family] = family_counts.get(family, 0) + 1

        for name in ranked_direct:
            if len(initial_names) >= initial_limit:
                break
            add(name)

        stage1_names = list(initial_names)

        def add_stage1(name: str) -> None:
            normalized = str(name or "").strip()
            if (
                not normalized
                or normalized not in registry
                or normalized in stage1_names
                or normalized in stage1_blocklist
                or tool_stage(normalized) == "internal_only"
            ):
                return
            stage1_names.append(normalized)

        for preferred in [
            "fact_timeline_resolution_search",
            "choice_grounded_evidence_search" if mcq_profile.get("needs_choice_tool_stage1") else "",
            "entity_event_trace_search",
            "vdb_get_docs_by_document_ids",
            "search_related_content",
            "get_entity_sections",
            "search_sections",
        ]:
            if len(stage1_names) >= stage1_limit:
                break
            add_stage1(preferred)

        for name in ranked_followup:
            if len(stage1_names) >= stage1_limit:
                break
            add_stage1(name)

        if need_narrative:
            for name in narrative_direct:
                if len(stage1_names) >= stage1_limit:
                    break
                add_stage1(name)

        if has_entity_anchor:
            for name in entity_direct:
                if len(stage1_names) >= stage1_limit:
                    break
                add_stage1(name)

        for name in structural_direct:
            if len(stage1_names) >= stage1_limit:
                break
            add_stage1(name)

        for name in ranked_names:
            if len(stage1_names) >= stage1_limit:
                break
            add_stage1(name)

        full_names = unique_names(
            [*stage1_names, *[name for name in registry.keys() if tool_stage(name) != "internal_only"]],
            valid=registry.keys(),
            limit=None,
        )

        def names_to_tools(names: List[str]) -> List[Any]:
            return [registry[name] for name in names if name in registry]

        plan: List[List[Any]] = []
        for subset in [
            names_to_tools(initial_names),
            names_to_tools(stage1_names),
            names_to_tools(full_names),
        ]:
            names = tuple(getattr(t, "name", "") for t in subset)
            if subset and all(tuple(getattr(x, "name", "") for x in s) != names for s in plan):
                plan.append(subset)

        self._last_route_plan = {
            "router_mode": "qwen_like",
            "question_type": question_type,
            "answer_shape": answer_shape,
            "evidence_need": evidence_need,
            "deterministic_parse": parse,
            "mcq_profile": mcq_profile,
            "initial_tools": list(initial_names),
            "escalation_tools": [name for name in stage1_names if name not in initial_names],
            "selected_branch_index": branch_index,
            "selected_branch": {
                "branch_id": f"qwen_like_{branch_index}",
                "intent": "weak routing with a broad direct-evidence subset and additive expansion",
                "tools": list(initial_names),
                "score": 1.0,
            },
            "stages": [[getattr(t, "name", "") for t in stage] for stage in plan],
            "query": normalized_query,
        }
        logger.info(
            "Qwen-like tool routing plan built for query=%s initial=%s stage1=%s",
            normalized_query,
            initial_names,
            stage1_names,
        )
        return plan

    def score_tools_for_query(
        self,
        *,
        query: str,
        memory_ctx: Dict[str, Any],
        tools: List[Any],
    ) -> List[Tuple[float, str]]:
        preferred_tools = set(self.preferred_tools_from_memory(memory_ctx))
        normalized_query = self._normalized_query(query)
        heuristic_boost_map = {} if self.disable_heuristic_router else heuristic_tool_boosts(normalized_query)
        preferred_tool_boost = float(
            getattr(getattr(self.config, "strategy_memory", None), "preferred_tool_boost", 1.1) or 1.1
        )
        ranked: List[Tuple[float, str]] = []

        for tool in tools:
            raw_name = str(getattr(tool, "name", "") or "").strip()
            if not raw_name:
                continue
            name, description, params_text = self._resolve_tool_profile(tool)
            stage = tool_stage(name or raw_name)
            if stage == "internal_only":
                continue
            profile_text = f"{name} {description} {params_text}".strip()
            score = 0.0
            score += 3.0 * token_jaccard_overlap(normalized_query, (name or raw_name).replace("_", " "))
            score += 2.0 * token_jaccard_overlap(normalized_query, description)
            score += 1.0 * token_jaccard_overlap(normalized_query, profile_text)
            if raw_name in preferred_tools or name in preferred_tools:
                score += preferred_tool_boost
            score += heuristic_boost_map.get(raw_name, 0.0)
            score += heuristic_boost_map.get(name, 0.0)
            if stage == "extended":
                score -= 0.35
            elif stage == "exploratory":
                score -= 1.35
            ranked.append((score, raw_name))

        ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return ranked

    def build_tool_execution_plan(
        self,
        *,
        query: str,
        memory_ctx: Dict[str, Any],
        tools: List[Any],
    ) -> List[List[Any]]:
        if self.router_mode == "legacy_full":
            self._last_route_plan = {}
            plan = [
                [
                    tool
                    for tool in tools
                    if tool_stage(str(getattr(tool, "name", "") or "").strip()) != "internal_only"
                ]
            ]
            self._sanitize_last_route_plan(tools=tools)
            return plan
        if self.router_mode == "stable":
            plan = self._build_stable_execution_plan(query=query, memory_ctx=memory_ctx, tools=tools)
            if plan:
                self._sanitize_last_route_plan(tools=tools)
                return plan
        llm_plan = self._llm_route_plan(query=query, memory_ctx=memory_ctx, tools=tools)
        if llm_plan is not None:
            plan = self._build_llm_execution_plan(
                query=query,
                memory_ctx=memory_ctx,
                tools=tools,
                route_plan=llm_plan,
            )
            if plan:
                self._sanitize_last_route_plan(tools=tools)
                return plan

        plan = self._build_heuristic_execution_plan(query=query, memory_ctx=memory_ctx, tools=tools)
        self._sanitize_last_route_plan(tools=tools)
        return plan

    def _build_llm_execution_plan(
        self,
        *,
        query: str,
        memory_ctx: Dict[str, Any],
        tools: List[Any],
        route_plan: Dict[str, Any],
    ) -> List[List[Any]]:
        normalized_query = self._normalized_query(query)
        registry: Dict[str, Any] = {}
        for tool in tools:
            name = str(getattr(tool, "name", "") or "").strip()
            if name and name not in registry:
                registry[name] = tool
        branch_index = max(0, int(memory_ctx.get("router_branch_index", 0) or 0))
        candidate_branches = list(route_plan.get("candidate_branches") or [])
        if not candidate_branches:
            return []
        selected_branch = candidate_branches[branch_index % len(candidate_branches)]
        shared_core = [name for name in route_plan.get("shared_core_tools") or [] if name in registry]
        selected_tools = [name for name in (selected_branch.get("tools") or []) if name in registry]
        escalation = [name for name in route_plan.get("escalation_tools") or [] if name in registry]
        all_branch_tools: List[str] = []
        for branch in candidate_branches:
            for name in branch.get("tools") or []:
                if name in registry and name not in all_branch_tools:
                    all_branch_tools.append(name)

        def merge_names(base: List[str], extra: List[str], *, limit: Optional[int]) -> List[Any]:
            names: List[str] = []
            for name in [*base, *extra]:
                if name in registry and name not in names:
                    names.append(name)
            if limit is not None:
                names = names[: max(1, int(limit))]
            return [registry[name] for name in names if name in registry]

        stage0 = merge_names(selected_tools, [], limit=5)
        if not stage0:
            stage0 = merge_names(shared_core, [], limit=5)
        stage1 = merge_names([getattr(t, "name", "") for t in stage0], [*shared_core, *escalation], limit=10)
        stage2 = merge_names([getattr(t, "name", "") for t in stage1], all_branch_tools, limit=14)
        full = merge_names([getattr(t, "name", "") for t in stage2], list(registry.keys()), limit=None)

        plan: List[List[Any]] = []
        for subset in [stage0, stage1, stage2, full]:
            names = tuple(getattr(t, "name", "") for t in subset)
            if subset and all(tuple(getattr(x, "name", "") for x in s) != names for s in plan):
                plan.append(subset)
        self._last_route_plan = {
            **dict(route_plan),
            "selected_branch_index": branch_index % len(candidate_branches),
            "selected_branch": dict(selected_branch),
            "stages": [[getattr(t, "name", "") for t in stage] for stage in plan],
            "query": normalized_query,
        }
        logger.info(
            "LLM tool routing plan built for query=%s selected_branch=%s stages=%s",
            normalized_query,
            selected_branch.get("branch_id", ""),
            self._last_route_plan["stages"],
        )
        return plan

    def _build_heuristic_execution_plan(
        self,
        *,
        query: str,
        memory_ctx: Dict[str, Any],
        tools: List[Any],
    ) -> List[List[Any]]:
        normalized_query = self._normalized_query(query)
        registry: Dict[str, Any] = {}
        for tool in tools:
            name = str(getattr(tool, "name", "") or "").strip()
            if name and name not in registry:
                registry[name] = tool

        ranked = self.score_tools_for_query(query=normalized_query, memory_ctx=memory_ctx, tools=tools)
        ranked_names = [name for _, name in ranked if name in registry]
        stage_names: Dict[str, List[str]] = {"core": [], "extended": [], "exploratory": []}
        for name in ranked_names:
            stage = tool_stage(name)
            if stage in stage_names and name not in stage_names[stage]:
                stage_names[stage].append(name)

        query_pattern = self._memory_query_pattern(memory_ctx)
        preferred_core = (
            []
            if self.disable_heuristic_router
            else preferred_core_tools(query=normalized_query, query_pattern=query_pattern)
        )

        def merge_names(base: List[str], extra: List[str], *, limit: Optional[int]) -> List[Any]:
            names: List[str] = []
            for name in [*base, *extra]:
                if name in registry and name not in names:
                    names.append(name)
            if limit is not None:
                names = names[: max(1, int(limit))]
            return [registry[name] for name in names if name in registry]

        stage0 = merge_names(
            [name for name in preferred_core if name in registry],
            stage_names["core"],
            limit=6,
        )
        stage1 = merge_names(
            [getattr(t, "name", "") for t in stage0],
            [*stage_names["extended"], *stage_names["core"]],
            limit=10,
        )
        stage2 = merge_names(
            [getattr(t, "name", "") for t in stage1],
            [*stage_names["exploratory"], *stage_names["extended"], *stage_names["core"]],
            limit=14,
        )
        full = merge_names(
            [getattr(t, "name", "") for t in stage2],
            [name for name in registry.keys() if tool_stage(name) != "internal_only"],
            limit=None,
        )

        plan: List[List[Any]] = []
        for subset in [stage0, stage1, stage2, full]:
            names = tuple(getattr(t, "name", "") for t in subset)
            if subset and all(tuple(getattr(x, "name", "") for x in s) != names for s in plan):
                plan.append(subset)
        self._last_route_plan = {
            "router_mode": "heuristic_disabled" if self.disable_heuristic_router else "heuristic",
            "question_type": self._question_type_hint(query),
            "candidate_branches": [],
            "selected_branch_index": 0,
            "selected_branch": {},
            "stages": [[getattr(t, "name", "") for t in stage] for stage in plan],
            "query": normalized_query,
        }
        logger.info(
            "Tool routing plan built for query=%s stages=%s",
            normalized_query,
            [[getattr(t, "name", "") for t in stage] for stage in plan],
        )
        return plan

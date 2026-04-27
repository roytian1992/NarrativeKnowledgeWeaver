from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from core.agent.retrieval.router_schema import deterministic_parse
from core.agent.retrieval.tool_family_registry import family_cards_text


class StableLLMRetrievalRouter:
    def __init__(self, *, router_llm: Any = None) -> None:
        self.router_llm = router_llm
        self._cache: Dict[str, Dict[str, Any]] = {}

    def route(
        self,
        *,
        query: str,
        tool_cards: List[Dict[str, str]],
    ) -> Optional[Dict[str, Any]]:
        if self.router_llm is None:
            return None
        normalized_query = str(query or "").strip()
        if not normalized_query:
            return None
        cached = self._cache.get(normalized_query)
        if isinstance(cached, dict) and cached:
            return dict(cached)

        parse = deterministic_parse(normalized_query)
        prompt = "\n".join(
            [
                "You are a retrieval planner. Do not answer the question.",
                "Return JSON only.",
                "",
                "You must classify the question and choose a SMALL but FLEXIBLE initial tool set.",
                "Do not emit natural-language advice for the downstream agent.",
                "The deterministic parse below is only a soft prior, not a hard rule.",
                "Choose tools from their descriptions, capabilities, and parameter schema, not from a memorized rigid question-type-to-tool table.",
                "If the question is anchored by named entities, characters, places, organizations, or titles, early entity grounding is often valuable.",
                "If a tool clearly resolves a surface name to a stable entity or node identifier, that tool is often a good early choice for anchored questions.",
                "If the question asks for a global ranking by importance, centrality, prominence, or key narrative role, prefer deterministic graph-ranking tools over ordinary entity lookup.",
                "For attitude, implication, motive, dilemma, or internal-state questions, keep at least one narrative_reasoning option in the initial set if available.",
                "Do not overfit to a single retrieval family when multiple evidence routes look plausible.",
                "When both local textual evidence and narrative/event aggregation look useful, keep both in the initial set instead of committing too early to one path.",
                "If likely uncertainty remains after the initial pass, provide escalation tools.",
                "",
                "JSON schema:",
                "{",
                '  "question_type": "exact_fact|relationship|attitude_or_state|implication_or_inference|chronology|section_localization|unknown",',
                '  "answer_shape": "mcq_option|short_fact|explanation|list",',
                '  "evidence_need": "local|local_plus_narrative|entity_plus_local|chronology|mixed",',
                '  "anchor_strategy": "entity_first|local_first|narrative_first|mixed",',
                '  "desired_capabilities": ["entity_grounding", "passage_evidence"],',
                '  "initial_clusters": ["local_evidence"],',
                '  "initial_tools": ["tool_a", "tool_b"],',
                '  "escalation_tools": ["tool_c"],',
                '  "parallel_tools": [["tool_a", "tool_b"]],',
                '  "tool_query_hints": [{"tool": "retrieve_entity_by_name", "query": "entity or constraint anchor", "params": {"entity_type": "Character", "resolve_source_documents": true}, "purpose": "why this query helps"}],',
                '  "must_compare_options": true,',
                '  "reason": "short justification"',
                "}",
                "",
                "Rules:",
                "- Use only tool names from the provided list.",
                "- initial_tools should usually contain 4-6 tools.",
                "- escalation_tools should usually contain 1-4 tools.",
                "- parallel_tools should only group tools that complement each other.",
                "- Do not repeat the same capability unnecessarily, but do keep complementary routes when useful.",
                "- If the question has entity anchors, prefer including at least one entity grounding tool unless the anchors are obviously irrelevant.",
                "- For multiple-choice questions, keep at least one tool that helps compare options against evidence.",
                "- Prefer direct-ready tools in the initial set. Tools marked as followup usually require outputs such as document_ids from earlier tools and should usually be escalation tools.",
                "- Choose desired_capabilities based on the tool descriptions and parameter schema shown below, not on memorized tool names.",
                "- Favor an additive first pass: if 2-3 families look complementary, include them together rather than outputting a narrow single-family set.",
                "- Avoid brittle hard commitments. The downstream agent should still have room to explore within the chosen subset.",
                "- tool_query_hints are optional query-expansion hints for the downstream agent. For scene-list questions, separate named entity anchors from semantic constraints and prefer resolve_source_documents=true for entity lookup.",
                "",
                f"Deterministic parse:\n{json.dumps(parse, ensure_ascii=False)}",
                "",
                f"Question:\n{normalized_query}",
                "",
                "Available tool families:",
                family_cards_text(tool_cards),
            ]
        ).strip()
        try:
            result = self.router_llm.run([{"role": "user", "content": prompt}])
        except Exception:
            return None
        content = ""
        if isinstance(result, list) and result:
            content = str((result[0] or {}).get("content", "") or "").strip()
        if not content:
            return None
        payload = self._extract_json_object(content)
        if not isinstance(payload, dict):
            return None
        self._cache[normalized_query] = dict(payload)
        return dict(payload)

    @staticmethod
    def _extract_json_object(raw: str) -> Optional[Dict[str, Any]]:
        text = str(raw or "").strip()
        if not text:
            return None
        candidates = [text]
        start = text.find("{")
        end = text.rfind("}")
        if 0 <= start < end:
            candidates.append(text[start : end + 1])
        for candidate in candidates:
            try:
                payload = json.loads(candidate)
            except Exception:
                continue
            if isinstance(payload, dict):
                return payload
        return None

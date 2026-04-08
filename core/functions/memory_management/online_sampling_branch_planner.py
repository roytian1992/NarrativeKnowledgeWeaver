from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

from core.utils.function_manager import process_with_format_guarantee
from core.utils.general_text import general_repair_template
from core.utils.general_utils import parse_json_object_from_text


logger = logging.getLogger(__name__)


def _clip_text(value: Any, limit: int) -> str:
    text = str(value or "").strip()
    if limit <= 0 or len(text) <= limit:
        return text
    marker = "\n...[truncated]...\n"
    head = max(0, int(limit * 0.7))
    tail = max(0, limit - head - len(marker))
    if tail <= 0:
        return text[:limit]
    return text[:head] + marker + text[-tail:]


def _normalize_branch_spec(raw: Any) -> Dict[str, str]:
    row = raw if isinstance(raw, dict) else {}
    return {
        "name": str(row.get("name", "") or "").strip(),
        "focus": str(row.get("focus", "") or "").strip(),
        "tool_hint": str(row.get("tool_hint", "") or "").strip(),
        "constraint": str(row.get("constraint", "") or "").strip(),
    }


class OnlineSamplingBranchPlanner:
    def __init__(
        self,
        prompt_loader,
        llm,
        *,
        prompt_id: str = "memory/plan_trajectory_direction",
    ) -> None:
        self.prompt_loader = prompt_loader
        self.llm = llm
        self.prompt_id = prompt_id

    def plan(
        self,
        *,
        question: str,
        available_tools: List[Dict[str, str]],
        planned_branch_count: int,
        strategy_prior_knowledge: str = "",
    ) -> List[Dict[str, str]]:
        target = max(0, int(planned_branch_count or 0))
        if target <= 0:
            return []

        fallback = self._fallback_branches(
            available_tools=available_tools,
            planned_branch_count=target,
        )
        if self.prompt_loader is None or self.llm is None:
            return fallback

        try:
            prompt = self.prompt_loader.render(
                self.prompt_id,
                static_values={},
                task_values={
                    "question": _clip_text(question, 2000),
                    "available_tools_json": _clip_text(
                        json.dumps(available_tools, ensure_ascii=False, indent=2),
                        8000,
                    ),
                    "planned_branch_count": str(target),
                    "strategy_prior_knowledge": _clip_text(strategy_prior_knowledge, 4000),
                },
                strict=True,
            )
        except Exception as exc:
            logger.warning("sampling branch planner prompt render failed: %s", exc)
            return fallback

        original_max_tokens = getattr(self.llm, "max_tokens", None)
        try:
            if isinstance(original_max_tokens, int) and original_max_tokens > 1024:
                self.llm.max_tokens = 1024
            corrected_json, status = process_with_format_guarantee(
                llm_client=self.llm,
                messages=[{"role": "user", "content": prompt}],
                required_fields=["branches"],
                field_validators={"branches": lambda v: isinstance(v, list)},
                max_retries=2,
                repair_template=general_repair_template,
            )
        except Exception as exc:
            logger.warning("sampling branch planner failed: %s", exc)
            return fallback
        finally:
            if original_max_tokens is not None:
                self.llm.max_tokens = original_max_tokens

        if status != "success":
            return fallback

        payload = parse_json_object_from_text(corrected_json) or {}
        raw_rows = payload.get("branches") if isinstance(payload.get("branches"), list) else []
        parsed: List[Dict[str, str]] = []
        seen: set[tuple[str, str, str, str]] = set()
        for raw in raw_rows:
            item = _normalize_branch_spec(raw)
            if not item["name"] or not item["focus"] or not item["tool_hint"] or not item["constraint"]:
                continue
            sig = (
                item["name"].lower(),
                item["focus"].lower(),
                item["tool_hint"].lower(),
                item["constraint"].lower(),
            )
            if sig in seen:
                continue
            seen.add(sig)
            parsed.append(item)
            if len(parsed) >= target:
                break

        if len(parsed) < target:
            for item in fallback:
                sig = (
                    item["name"].lower(),
                    item["focus"].lower(),
                    item["tool_hint"].lower(),
                    item["constraint"].lower(),
                )
                if sig in seen:
                    continue
                seen.add(sig)
                parsed.append(item)
                if len(parsed) >= target:
                    break
        return parsed[:target]

    @staticmethod
    def _pick_tool_hint(preferred_names: List[str], available_names: set[str], default_hint: str) -> str:
        for name in preferred_names:
            if name in available_names:
                return name
        return default_hint

    def _fallback_branches(
        self,
        *,
        available_tools: List[Dict[str, str]],
        planned_branch_count: int,
    ) -> List[Dict[str, str]]:
        available_names = {
            str(item.get("name", "") or "").strip()
            for item in (available_tools or [])
            if str(item.get("name", "") or "").strip()
        }
        defaults: List[Dict[str, str]] = [
            {
                "name": "lexical_grounding",
                "focus": "anchor exact wording, local constraints, and option-level evidence",
                "tool_hint": self._pick_tool_hint(
                    ["bm25_search_docs", "search_related_content"],
                    available_names,
                    "keyword-oriented retrieval",
                ),
                "constraint": "do not stop at broad thematic matches; verify the exact wording required by the question",
            },
            {
                "name": "section_evidence_first",
                "focus": "prioritize local passage evidence that can directly support or reject candidate answers",
                "tool_hint": self._pick_tool_hint(
                    ["section_evidence_search", "search_sections", "vdb_search_sentences"],
                    available_names,
                    "section-level evidence tools",
                ),
                "constraint": "avoid relying on broad summaries before finding direct local support",
            },
            {
                "name": "entity_relation_crosscheck",
                "focus": "identify central entities first, then verify relations, roles, or supporting sections",
                "tool_hint": self._pick_tool_hint(
                    ["retrieve_entity_by_name", "search_related_entities", "get_entity_sections"],
                    available_names,
                    "entity-centric retrieval",
                ),
                "constraint": "do not equate a nearby entity mention with the answer; verify the actual relation or role",
            },
            {
                "name": "interaction_dialogue_check",
                "focus": "inspect interaction or dialogue evidence when speaker, target, stance, or interpersonal evidence matters",
                "tool_hint": self._pick_tool_hint(
                    ["search_dialogues", "search_interactions", "get_interactions_by_document_ids"],
                    available_names,
                    "interaction-oriented retrieval",
                ),
                "constraint": "use this branch only for grounded interaction evidence, not vague character impressions",
            },
            {
                "name": "narrative_structure_check",
                "focus": "use episode or storyline structure to verify ordering, context, or cross-section consistency",
                "tool_hint": self._pick_tool_hint(
                    ["search_episodes", "search_storylines", "narrative_hierarchical_search"],
                    available_names,
                    "narrative-structure retrieval",
                ),
                "constraint": "do not answer from graph structure alone without checking supporting content",
            },
        ]
        out: List[Dict[str, str]] = []
        for item in defaults:
            hint = str(item.get("tool_hint", "") or "").strip()
            if "retrieval" not in hint and hint not in available_names:
                continue
            out.append(item)
            if len(out) >= planned_branch_count:
                break
        if len(out) < planned_branch_count:
            for item in defaults:
                if item in out:
                    continue
                out.append(item)
                if len(out) >= planned_branch_count:
                    break
        return out[:planned_branch_count]

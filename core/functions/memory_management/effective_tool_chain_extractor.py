from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Set

from core.utils.format import correct_json_format
from core.utils.function_manager import process_with_format_guarantee
from core.utils.general_text import general_repair_template
from core.utils.general_utils import parse_json_object_from_text

logger = logging.getLogger(__name__)


_REDUNDANT_RETRIEVAL_TOOLS = {
    "bm25_search_docs",
    "vdb_search_sentences",
    "search_sections",
    "search_related_content",
    "section_evidence_search",
    "vdb_get_docs_by_document_ids",
}

_TOOL_DEPENDENCY_GROUPS = {
    "get_entity_sections": [["retrieve_entity_by_name"]],
    "search_related_entities": [["retrieve_entity_by_name"]],
    "get_relations_between_entities": [["retrieve_entity_by_name"]],
    "get_common_neighbors": [["retrieve_entity_by_name"]],
    "find_paths_between_nodes": [["retrieve_entity_by_name"]],
    "get_k_hop_subgraph": [["retrieve_entity_by_name"]],
    "vdb_get_docs_by_document_ids": [[
        "bm25_search_docs",
        "section_evidence_search",
        "search_sections",
        "retrieve_entity_by_name",
        "vdb_search_docs",
        "lookup_document_ids_by_title",
        "search_related_content",
    ]],
}


def _clip_text(value: Any, limit: int) -> str:
    text = str(value or "").strip()
    if limit <= 0 or len(text) <= limit:
        return text
    marker = "\n...[truncated for chain extraction]...\n"
    head = max(0, int(limit * 0.7))
    tail = max(0, limit - head - len(marker))
    if tail <= 0:
        return text[:limit]
    return text[:head] + marker + text[-tail:]


def _tool_name(item: Dict[str, Any]) -> str:
    return str((item or {}).get("tool_name", "") or "").strip()


def _safe_bool(value: Any, *, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "1"}:
            return True
        if lowered in {"false", "no", "0"}:
            return False
    return default


def _safe_int_list(items: Any, *, limit: int = 32) -> List[int]:
    out: List[int] = []
    if not isinstance(items, list):
        return out
    for item in items:
        if isinstance(item, bool):
            continue
        if isinstance(item, (int, float)):
            value = int(item)
        elif isinstance(item, str) and item.strip().lstrip("-").isdigit():
            value = int(item.strip())
        else:
            continue
        if value not in out:
            out.append(value)
        if len(out) >= limit:
            break
    return out


def _build_default_step_attribution(
    *,
    step_index: int,
    tool_name: str,
    keep: bool,
    used_by_step_indices: List[int] | None = None,
    reason: str = "",
) -> Dict[str, Any]:
    used_by_step_indices = used_by_step_indices or []
    return {
        "step_index": int(step_index),
        "tool_name": str(tool_name or "").strip(),
        "is_question_relevant": bool(keep),
        "provides_final_evidence": False,
        "is_used_by_later_step": bool(used_by_step_indices),
        "used_by_step_indices": list(used_by_step_indices),
        "extracted_insights": "",
        "keep": bool(keep),
        "reason": str(reason or "").strip(),
    }


def _build_fallback_step_attributions(tool_uses: List[Dict[str, Any]], keep_indices: List[int]) -> List[Dict[str, Any]]:
    keep_set = set(keep_indices)
    out: List[Dict[str, Any]] = []
    for idx, item in enumerate(tool_uses or []):
        out.append(
            _build_default_step_attribution(
                step_index=idx,
                tool_name=_tool_name(item),
                keep=idx in keep_set,
                reason="fallback_keep_raw_chain" if idx in keep_set else "fallback_discard",
            )
        )
    return out


def _normalize_step_attributions(tool_uses: List[Dict[str, Any]], payload: Dict[str, Any], keep_indices: List[int]) -> List[Dict[str, Any]]:
    raw_items = payload.get("step_attributions")
    attr_by_index: Dict[int, Dict[str, Any]] = {}
    if isinstance(raw_items, list):
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            index_candidates = _safe_int_list([item.get("step_index")], limit=1)
            if not index_candidates:
                continue
            step_index = index_candidates[0]
            if step_index < 0 or step_index >= len(tool_uses):
                continue
            tool_name = str(item.get("tool_name", "") or "").strip() or _tool_name(tool_uses[step_index])
            used_by = [idx for idx in _safe_int_list(item.get("used_by_step_indices"), limit=16) if idx > step_index]
            attr_by_index[step_index] = {
                "step_index": step_index,
                "tool_name": tool_name,
                "is_question_relevant": _safe_bool(item.get("is_question_relevant"), default=step_index in keep_indices),
                "provides_final_evidence": _safe_bool(item.get("provides_final_evidence"), default=False),
                "is_used_by_later_step": _safe_bool(item.get("is_used_by_later_step"), default=bool(used_by)),
                "used_by_step_indices": used_by,
                "extracted_insights": str(item.get("extracted_insights", "") or "").strip(),
                "keep": _safe_bool(item.get("keep"), default=step_index in keep_indices),
                "reason": str(item.get("reason", "") or "").strip(),
            }
    if not attr_by_index:
        return _build_fallback_step_attributions(tool_uses, keep_indices)
    normalized: List[Dict[str, Any]] = []
    keep_set = set(keep_indices)
    for idx, item in enumerate(tool_uses or []):
        tool_name = _tool_name(item)
        attr = attr_by_index.get(idx)
        if attr is None:
            attr = _build_default_step_attribution(
                step_index=idx,
                tool_name=tool_name,
                keep=idx in keep_set,
                reason="missing_from_llm_attribution",
            )
        else:
            attr["tool_name"] = attr["tool_name"] or tool_name
            if idx in keep_set:
                attr["keep"] = True
        normalized.append(attr)
    return normalized


def _used_tools_ahead(keep_indices: List[int], tool_uses: List[Dict[str, Any]], current_idx: int) -> Set[str]:
    out: Set[str] = set()
    for idx in keep_indices:
        if idx <= current_idx or idx >= len(tool_uses):
            continue
        name = _tool_name(tool_uses[idx])
        if name:
            out.add(name)
    return out


def _deterministic_minimize(
    tool_uses: List[Dict[str, Any]],
    step_attributions: List[Dict[str, Any]],
    keep_indices: List[int],
) -> List[int]:
    if not keep_indices:
        return []
    keep_set = set(idx for idx in keep_indices if 0 <= idx < len(tool_uses))
    if not keep_set:
        return []
    attr_by_index = {
        int(item.get("step_index", -1)): item
        for item in (step_attributions or [])
        if isinstance(item, dict) and isinstance(item.get("step_index"), int)
    }
    ordered = sorted(keep_set)
    trimmed: List[int] = []
    for idx in ordered:
        attr = attr_by_index.get(idx, {})
        tool_name = _tool_name(tool_uses[idx])
        if not tool_name:
            continue
        provides_final_evidence = bool(attr.get("provides_final_evidence", False))
        is_used_by_later = bool(attr.get("is_used_by_later_step", False) or list(attr.get("used_by_step_indices") or []))
        future_tools = _used_tools_ahead(ordered, tool_uses, idx)

        if tool_name == "retrieve_entity_by_name" and not provides_final_evidence and not is_used_by_later:
            continue
        if (
            tool_name == "bm25_search_docs"
            and not provides_final_evidence
            and not is_used_by_later
            and ("section_evidence_search" in future_tools or "vdb_get_docs_by_document_ids" in future_tools)
        ):
            continue
        if (
            tool_name in _REDUNDANT_RETRIEVAL_TOOLS
            and not provides_final_evidence
            and not is_used_by_later
            and any(_tool_name(tool_uses[next_idx]) == tool_name for next_idx in ordered if next_idx > idx)
        ):
            continue
        trimmed.append(idx)

    if not trimmed:
        trimmed = [ordered[-1]]

    while len(trimmed) > 1:
        first_idx = trimmed[0]
        first_attr = attr_by_index.get(first_idx, {})
        first_tool = _tool_name(tool_uses[first_idx])
        if bool(first_attr.get("provides_final_evidence", False)):
            break
        if bool(first_attr.get("is_used_by_later_step", False) or list(first_attr.get("used_by_step_indices") or [])):
            break
        if first_tool not in {"retrieve_entity_by_name", "bm25_search_docs", "vdb_search_sentences", "search_sections", "search_related_content"}:
            break
        trimmed.pop(0)

    return trimmed or [ordered[-1]]


def _repair_dependency_closure(
    tool_uses: List[Dict[str, Any]],
    keep_indices: List[int],
) -> List[int]:
    if not keep_indices:
        return []
    keep_set = {idx for idx in keep_indices if 0 <= idx < len(tool_uses)}
    if not keep_set:
        return []

    changed = True
    while changed:
        changed = False
        ordered = sorted(keep_set)
        for idx in list(ordered):
            tool_name = _tool_name(tool_uses[idx])
            dependency_groups = _TOOL_DEPENDENCY_GROUPS.get(tool_name) or []
            if not dependency_groups:
                continue
            has_dependency = False
            for prev_idx in ordered:
                if prev_idx >= idx:
                    break
                prev_name = _tool_name(tool_uses[prev_idx])
                if any(prev_name in group for group in dependency_groups):
                    has_dependency = True
                    break
            if has_dependency:
                continue

            chosen_idx = None
            for group in dependency_groups:
                for prev_idx in range(idx - 1, -1, -1):
                    prev_name = _tool_name(tool_uses[prev_idx])
                    if prev_name in group:
                        chosen_idx = prev_idx
                        break
                if chosen_idx is not None:
                    break
            if chosen_idx is None or chosen_idx in keep_set:
                continue
            keep_set.add(chosen_idx)
            changed = True

    return sorted(keep_set)


class EffectiveToolChainExtractor:
    def __init__(self, prompt_loader, llm, prompt_id: str = "memory/extract_effective_tool_chain"):
        self.prompt_loader = prompt_loader
        self.llm = llm
        self.prompt_id = prompt_id

    def extract(
        self,
        *,
        question: str,
        reference_answer: str,
        candidate_answer: str,
        tool_uses: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        raw_chain = [_tool_name(item) for item in (tool_uses or []) if _tool_name(item)]
        raw_keep_indices = [idx for idx, item in enumerate(tool_uses or []) if _tool_name(item)]
        fallback = {
            "raw_tool_chain": raw_chain,
            "minimal_effective_chain": raw_chain,
            "effective_tool_chain": raw_chain,
            "effective_step_indices": raw_keep_indices,
            "discarded_step_indices": [],
            "step_attributions": _build_fallback_step_attributions(tool_uses or [], raw_keep_indices),
            "effective_step_attribution": _build_fallback_step_attributions(tool_uses or [], raw_keep_indices),
            "reason": "llm_unavailable_fallback_to_raw_chain",
        }
        if not raw_chain:
            return fallback
        if self.prompt_loader is None or self.llm is None:
            return fallback

        safe_tool_uses: List[Dict[str, Any]] = []
        for idx, item in enumerate(tool_uses or []):
            safe_tool_uses.append(
                {
                    "step_index": idx,
                    "tool_name": _tool_name(item),
                    "tool_arguments": _clip_text((item or {}).get("tool_arguments", ""), 600),
                    "tool_output": _clip_text((item or {}).get("tool_output", ""), 900),
                }
            )
        try:
            prompt = self.prompt_loader.render(
                self.prompt_id,
                static_values={},
                task_values={
                    "question": _clip_text(question, 1200),
                    "reference_answer": _clip_text(reference_answer, 2200),
                    "candidate_answer": _clip_text(candidate_answer, 2600),
                    "raw_tool_uses_json": json.dumps(safe_tool_uses, ensure_ascii=False, indent=2),
                },
                strict=True,
            )
        except Exception as exc:
            logger.warning("effective chain prompt render failed: %s", exc)
            return fallback

        original_max_tokens = getattr(self.llm, "max_tokens", None)
        try:
            if isinstance(original_max_tokens, int) and original_max_tokens > 900:
                self.llm.max_tokens = 900
            corrected_json, status = process_with_format_guarantee(
                llm_client=self.llm,
                messages=[{"role": "user", "content": prompt}],
                required_fields=["step_attributions", "effective_step_indices", "discarded_step_indices", "reason"],
                field_validators={
                    "step_attributions": lambda v: isinstance(v, list),
                    "effective_step_indices": lambda v: isinstance(v, list),
                    "discarded_step_indices": lambda v: isinstance(v, list),
                    "reason": lambda v: isinstance(v, str),
                },
                max_retries=2,
                repair_template=general_repair_template,
            )
        except Exception as exc:
            logger.warning("effective chain extraction failed: %s", exc)
            return {
                **fallback,
                "reason": f"effective_chain_error: {str(exc).strip()[:500]}",
            }
        finally:
            if original_max_tokens is not None:
                self.llm.max_tokens = original_max_tokens

        if status != "success":
            return fallback

        payload = parse_json_object_from_text(corrected_json) or {}
        llm_keep_indices = _safe_int_list(payload.get("effective_step_indices"), limit=len(tool_uses) + 4)
        step_attributions = _normalize_step_attributions(tool_uses or [], payload, llm_keep_indices)
        attributed_keep = [
            int(item.get("step_index"))
            for item in step_attributions
            if isinstance(item.get("step_index"), int) and bool(item.get("keep", False))
        ]
        keep_indices = llm_keep_indices or attributed_keep or raw_keep_indices
        keep_indices = _deterministic_minimize(tool_uses or [], step_attributions, keep_indices)
        repaired_keep_indices = _repair_dependency_closure(tool_uses or [], keep_indices)
        dependency_repair_applied = repaired_keep_indices != keep_indices
        keep_indices = repaired_keep_indices
        keep_set = set(keep_indices)
        minimal_chain = [_tool_name(tool_uses[idx]) for idx in keep_indices if 0 <= idx < len(tool_uses) and _tool_name(tool_uses[idx])]
        if not minimal_chain:
            keep_indices = raw_keep_indices
            keep_set = set(keep_indices)
            minimal_chain = raw_chain

        discarded_step_indices = [idx for idx, item in enumerate(tool_uses or []) if _tool_name(item) and idx not in keep_set]
        finalized_attributions: List[Dict[str, Any]] = []
        for idx, attr in enumerate(step_attributions):
            row = dict(attr)
            row["keep"] = idx in keep_set
            if idx in keep_set and not str(row.get("reason", "") or "").strip():
                row["reason"] = "retained_after_llm_and_deterministic_minimization"
            elif idx not in keep_set and not str(row.get("reason", "") or "").strip():
                row["reason"] = "discarded_after_llm_and_deterministic_minimization"
            finalized_attributions.append(row)

        reason = str(payload.get("reason", "") or "").strip() or fallback["reason"]
        if keep_indices != llm_keep_indices and llm_keep_indices:
            reason = f"{reason} | deterministic_trim_applied".strip()
        if dependency_repair_applied:
            reason = f"{reason} | dependency_closure_repaired".strip()

        return {
            "raw_tool_chain": raw_chain,
            "minimal_effective_chain": minimal_chain,
            "effective_tool_chain": minimal_chain,
            "effective_step_indices": keep_indices,
            "discarded_step_indices": discarded_step_indices,
            "step_attributions": finalized_attributions,
            "effective_step_attribution": finalized_attributions,
            "reason": reason,
        }


def extract_effective_tool_chain_with_guard(
    *,
    llm,
    prompt_loader,
    question: str,
    reference_answer: str,
    candidate_answer: str,
    tool_uses: List[Dict[str, Any]],
    prompt_id: str = "memory/extract_effective_tool_chain",
) -> str:
    extractor = EffectiveToolChainExtractor(prompt_loader=prompt_loader, llm=llm, prompt_id=prompt_id)
    result = extractor.extract(
        question=question,
        reference_answer=reference_answer,
        candidate_answer=candidate_answer,
        tool_uses=tool_uses,
    )
    return correct_json_format(json.dumps(result, ensure_ascii=False))


class EffectiveToolChainExtractorTool:
    def __init__(self, prompt_loader, llm, prompt_id: str = "memory/extract_effective_tool_chain"):
        self.prompt_loader = prompt_loader
        self.llm = llm
        self.prompt_id = prompt_id

    def call(self, params: str, **kwargs) -> str:
        try:
            payload = json.loads(params) if isinstance(params, str) else (params or {})
        except Exception as exc:
            return correct_json_format(json.dumps({"error": f"params parse failed: {exc}"}, ensure_ascii=False))
        return extract_effective_tool_chain_with_guard(
            llm=self.llm,
            prompt_loader=self.prompt_loader,
            question=str(payload.get("question", "") or ""),
            reference_answer=str(payload.get("reference_answer", "") or ""),
            candidate_answer=str(payload.get("candidate_answer", "") or ""),
            tool_uses=payload.get("tool_uses") or [],
            prompt_id=self.prompt_id,
        )

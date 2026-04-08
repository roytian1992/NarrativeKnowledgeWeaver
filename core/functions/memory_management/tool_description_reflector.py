from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

from core.utils.format import correct_json_format
from core.utils.function_manager import process_with_format_guarantee
from core.utils.general_text import general_repair_template
from core.utils.general_utils import parse_json_object_from_text


logger = logging.getLogger(__name__)


def _format_parameters(parameters: Any) -> str:
    if not isinstance(parameters, list) or not parameters:
        return "- (none)"
    lines: List[str] = []
    for item in parameters:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "") or "").strip()
        ptype = str(item.get("type", "") or "").strip()
        description = str(item.get("description", "") or "").strip()
        required = bool(item.get("required", False))
        if not name:
            continue
        label = f"- {name}"
        if ptype:
            label += f" ({ptype})"
        label += " required" if required else " optional"
        if description:
            label += f": {description}"
        lines.append(label)
    return "\n".join(lines) if lines else "- (none)"


class ToolDescriptionReflector:
    def __init__(self, prompt_loader, llm, prompt_id: str = "memory/optimize_tool_description"):
        self.prompt_loader = prompt_loader
        self.llm = llm
        self.prompt_id = prompt_id

    def reflect(
        self,
        *,
        tool_name: str,
        current_description: str,
        parameters: List[Dict[str, Any]],
        question: str,
        query_pattern: Dict[str, Any],
        successful_attempt: Dict[str, Any],
        failed_attempts: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        fallback_description = str(current_description or "").strip()
        fallback = {
            "decision": "keep",
            "proposed_description": fallback_description,
            "reason": "keep_current_description_by_default",
        }
        if self.prompt_loader is None or self.llm is None:
            return fallback
        try:
            prompt = self.prompt_loader.render(
                self.prompt_id,
                static_values={},
                task_values={
                    "tool_name": str(tool_name or "").strip(),
                    "current_description": fallback_description,
                    "parameter_descriptions": _format_parameters(parameters),
                    "question": str(question or "").strip(),
                    "query_pattern_json": json.dumps(query_pattern or {}, ensure_ascii=False, indent=2),
                    "successful_attempt_json": json.dumps(successful_attempt or {}, ensure_ascii=False, indent=2),
                    "failed_attempts_json": json.dumps(failed_attempts or [], ensure_ascii=False, indent=2),
                },
                strict=True,
            )
        except Exception as exc:
            logger.warning("tool description reflection prompt render failed: %s", exc)
            return fallback
        try:
            corrected_json, status = process_with_format_guarantee(
                llm_client=self.llm,
                messages=[{"role": "user", "content": prompt}],
                required_fields=["decision", "proposed_description", "reason"],
                field_validators={
                    "decision": lambda v: isinstance(v, str) and str(v).strip().lower() in {"keep", "revise"},
                    "proposed_description": lambda v: isinstance(v, str),
                    "reason": lambda v: isinstance(v, str),
                },
                max_retries=2,
                repair_template=general_repair_template,
            )
        except Exception as exc:
            logger.warning("tool description reflection failed: %s", exc)
            return {
                **fallback,
                "reason": f"reflection_error: {str(exc).strip()[:500]}",
            }
        if status != "success":
            return fallback

        payload = parse_json_object_from_text(corrected_json) or {}
        decision = str(payload.get("decision", "keep") or "keep").strip().lower()
        proposed_description = str(payload.get("proposed_description", "") or "").strip()
        reason = str(payload.get("reason", "") or "").strip() or fallback["reason"]

        if decision not in {"keep", "revise"}:
            decision = "keep"
        if not proposed_description:
            proposed_description = fallback_description
        if decision == "keep" or proposed_description == fallback_description:
            return {
                "decision": "keep",
                "proposed_description": fallback_description,
                "reason": reason or fallback["reason"],
            }
        return {
            "decision": "revise",
            "proposed_description": proposed_description,
            "reason": reason,
        }


def reflect_tool_description_with_guard(
    *,
    llm,
    prompt_loader,
    tool_name: str,
    current_description: str,
    parameters: List[Dict[str, Any]],
    question: str,
    query_pattern: Dict[str, Any],
    successful_attempt: Dict[str, Any],
    failed_attempts: List[Dict[str, Any]],
    prompt_id: str = "memory/optimize_tool_description",
) -> str:
    reflector = ToolDescriptionReflector(prompt_loader=prompt_loader, llm=llm, prompt_id=prompt_id)
    result = reflector.reflect(
        tool_name=tool_name,
        current_description=current_description,
        parameters=parameters,
        question=question,
        query_pattern=query_pattern,
        successful_attempt=successful_attempt,
        failed_attempts=failed_attempts,
    )
    return correct_json_format(json.dumps(result, ensure_ascii=False))


class ToolDescriptionReflectorTool:
    def __init__(self, prompt_loader, llm, prompt_id: str = "memory/optimize_tool_description"):
        self.prompt_loader = prompt_loader
        self.llm = llm
        self.prompt_id = prompt_id

    def call(self, params: str, **kwargs) -> str:
        try:
            payload = json.loads(params) if isinstance(params, str) else (params or {})
        except Exception as exc:
            return correct_json_format(json.dumps({"error": f"params parse failed: {exc}"}, ensure_ascii=False))
        return reflect_tool_description_with_guard(
            llm=self.llm,
            prompt_loader=self.prompt_loader,
            tool_name=str(payload.get("tool_name", "") or ""),
            current_description=str(payload.get("current_description", "") or ""),
            parameters=payload.get("parameters") or [],
            question=str(payload.get("question", "") or ""),
            query_pattern=payload.get("query_pattern") or {},
            successful_attempt=payload.get("successful_attempt") or {},
            failed_attempts=payload.get("failed_attempts") or [],
            prompt_id=self.prompt_id,
        )

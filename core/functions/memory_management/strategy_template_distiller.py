from __future__ import annotations

import json
import logging
from typing import Any, Dict

from core.utils.format import correct_json_format
from core.utils.function_manager import process_with_format_guarantee
from core.utils.general_text import general_repair_template
from core.utils.general_utils import parse_json_object_from_text

logger = logging.getLogger(__name__)


class StrategyTemplateDistiller:
    def __init__(self, prompt_loader, llm, prompt_id: str = "memory/distill_strategy_template"):
        self.prompt_loader = prompt_loader
        self.llm = llm
        self.prompt_id = prompt_id

    def distill(
        self,
        *,
        question: str,
        query_pattern: Dict[str, Any],
        best_attempt: Dict[str, Any],
        failed_attempts: list[Dict[str, Any]] | None = None,
        retry_instruction: str = "",
    ) -> Dict[str, Any]:
        fallback_chain = [
            str(x).strip()
            for x in (
                best_attempt.get("minimal_effective_chain")
                or best_attempt.get("effective_tool_chain")
                or best_attempt.get("raw_tool_chain")
                or []
            )
            if str(x).strip()
        ]
        fallback = {
            "pattern_name": str(query_pattern.get("problem_type", "generic_retrieval_pattern") or "generic_retrieval_pattern"),
            "pattern_description": str(query_pattern.get("query_abstract", "Generic retrieval pattern") or "Generic retrieval pattern"),
            "recommended_chain": fallback_chain,
            "anti_patterns": [],
            "chain_rationale": "",
            "chain_constraints": [],
        }
        if self.prompt_loader is None or self.llm is None:
            return fallback
        try:
            prompt = self.prompt_loader.render(
                self.prompt_id,
                static_values={},
                task_values={
                    "question": str(question or ""),
                    "query_pattern_json": json.dumps(query_pattern or {}, ensure_ascii=False, indent=2),
                    "best_attempt_json": json.dumps(best_attempt or {}, ensure_ascii=False, indent=2),
                    "failed_attempts_json": json.dumps(failed_attempts or [], ensure_ascii=False, indent=2),
                    "retry_instruction": str(retry_instruction or ""),
                },
                strict=True,
            )
        except Exception as exc:
            logger.warning("strategy template prompt render failed: %s", exc)
            return fallback
        try:
            corrected_json, status = process_with_format_guarantee(
                llm_client=self.llm,
                messages=[{"role": "user", "content": prompt}],
                required_fields=["pattern_name", "pattern_description", "recommended_chain", "anti_patterns"],
                field_validators={
                    "pattern_name": lambda v: isinstance(v, str),
                    "pattern_description": lambda v: isinstance(v, str),
                    "recommended_chain": lambda v: isinstance(v, list),
                    "anti_patterns": lambda v: isinstance(v, list),
                },
                max_retries=2,
                repair_template=general_repair_template,
            )
        except Exception as exc:
            logger.warning("strategy template distill failed: %s", exc)
            return fallback
        if status != "success":
            return fallback
        payload = parse_json_object_from_text(corrected_json) or {}
        chain = [str(x).strip() for x in (payload.get("recommended_chain") or []) if str(x).strip()] or fallback_chain
        return {
            "pattern_name": str(payload.get("pattern_name", "") or fallback["pattern_name"]),
            "pattern_description": str(payload.get("pattern_description", "") or fallback["pattern_description"]),
            "recommended_chain": chain,
            "anti_patterns": [str(x).strip() for x in (payload.get("anti_patterns") or []) if str(x).strip()],
            "chain_rationale": str(payload.get("chain_rationale", "") or fallback["chain_rationale"]).strip(),
            "chain_constraints": [str(x).strip() for x in (payload.get("chain_constraints") or []) if str(x).strip()],
        }


def distill_strategy_template_with_guard(
    *,
    llm,
    prompt_loader,
    question: str,
    query_pattern: Dict[str, Any],
    best_attempt: Dict[str, Any],
    failed_attempts: list[Dict[str, Any]] | None = None,
    retry_instruction: str = "",
    prompt_id: str = "memory/distill_strategy_template",
) -> str:
    distiller = StrategyTemplateDistiller(prompt_loader=prompt_loader, llm=llm, prompt_id=prompt_id)
    result = distiller.distill(
        question=question,
        query_pattern=query_pattern,
        best_attempt=best_attempt,
        failed_attempts=failed_attempts,
        retry_instruction=retry_instruction,
    )
    return correct_json_format(json.dumps(result, ensure_ascii=False))


class StrategyTemplateDistillerTool:
    def __init__(self, prompt_loader, llm, prompt_id: str = "memory/distill_strategy_template"):
        self.prompt_loader = prompt_loader
        self.llm = llm
        self.prompt_id = prompt_id

    def call(self, params: str, **kwargs) -> str:
        try:
            payload = json.loads(params) if isinstance(params, str) else (params or {})
        except Exception as exc:
            return correct_json_format(json.dumps({"error": f"params parse failed: {exc}"}, ensure_ascii=False))
        return distill_strategy_template_with_guard(
            llm=self.llm,
            prompt_loader=self.prompt_loader,
            question=str(payload.get("question", "") or ""),
            query_pattern=payload.get("query_pattern") or {},
            best_attempt=payload.get("best_attempt") or {},
            failed_attempts=payload.get("failed_attempts") or [],
            retry_instruction=str(payload.get("retry_instruction", "") or ""),
            prompt_id=self.prompt_id,
        )

from __future__ import annotations

import json
import logging
from typing import Any

from core.utils.function_manager import process_with_format_guarantee
from core.utils.general_text import general_repair_template
from core.utils.format import correct_json_format

logger = logging.getLogger(__name__)


def summarize_distilled_memories_with_guard(
    *,
    llm,
    prompt_loader,
    grouped_memories_json: str,
    existing_summaries_json: str,
    prompt_id: str = "memory/summarize_distilled_memories",
) -> str:
    grouped_memories_json = str(grouped_memories_json or "").strip()
    existing_summaries_json = str(existing_summaries_json or "[]").strip()
    if not grouped_memories_json:
        return correct_json_format(json.dumps({"error": "missing grouped_memories_json"}, ensure_ascii=False))

    try:
        user_prompt = prompt_loader.render(
            prompt_id,
            static_values={},
            task_values={
                "grouped_memories_json": grouped_memories_json,
                "existing_summaries_json": existing_summaries_json,
            },
            strict=True,
        )
    except Exception as e:
        logger.error("summarize_distilled_memories prompt render failed: %s", e)
        return correct_json_format(json.dumps({"error": f"prompt render failed: {e}"}, ensure_ascii=False))

    messages = [{"role": "user", "content": user_prompt}]
    corrected_json, status = process_with_format_guarantee(
        llm_client=llm,
        messages=messages,
        required_fields=["summaries"],
        field_validators={"summaries": lambda v: isinstance(v, list)},
        max_retries=2,
        repair_template=general_repair_template,
    )
    if status == "success":
        return correct_json_format(corrected_json)
    return correct_json_format(json.dumps({"error": "distilled memory summary failed"}, ensure_ascii=False))


class DistilledMemorySummarizer:
    def __init__(self, prompt_loader, llm, prompt_id: str = "memory/summarize_distilled_memories"):
        self.prompt_loader = prompt_loader
        self.llm = llm
        self.prompt_id = prompt_id

    def call(self, params: str, **kwargs) -> str:
        try:
            p = json.loads(params) if isinstance(params, str) else (params or {})
        except Exception as e:
            return correct_json_format(json.dumps({"error": f"params parse failed: {e}"}, ensure_ascii=False))
        return summarize_distilled_memories_with_guard(
            llm=self.llm,
            prompt_loader=self.prompt_loader,
            grouped_memories_json=str(p.get("grouped_memories_json", "") or ""),
            existing_summaries_json=str(p.get("existing_summaries_json", "[]") or "[]"),
            prompt_id=self.prompt_id,
        )

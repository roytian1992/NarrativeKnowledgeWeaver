from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

from core.utils.function_manager import process_with_format_guarantee
from core.utils.general_text import general_repair_template
from core.utils.format import correct_json_format

logger = logging.getLogger(__name__)


def _normalize_json_unicode_text(s: str) -> str:
    """
    Normalize JSON text to UTF-8 visible chars (avoid \\uXXXX in persisted output).
    """
    try:
        obj = json.loads(s)
        return json.dumps(obj, ensure_ascii=False)
    except Exception:
        return correct_json_format(s)


def extract_distilled_memories_with_guard(
    *,
    llm,
    prompt_loader,
    source_payload: str,
    requested_max: int = 30,
    prompt_id: str = "memory/distill_extraction_memory",
) -> str:
    """
    Functional API for distilled memory extraction with JSON format guarantee.
    """
    source_payload = str(source_payload or "").strip()
    requested_max = int(requested_max or 30)
    if not source_payload:
        return correct_json_format(json.dumps({"error": "missing required field: source_payload"}, ensure_ascii=False))

    try:
        user_prompt = prompt_loader.render(
            prompt_id,
            static_values={},
            task_values={"source_payload": source_payload, "requested_max": str(requested_max)},
            strict=True,
        )
    except Exception as e:
        logger.error("distill_extraction_memory prompt render failed: %s", e)
        return correct_json_format(json.dumps({"error": f"prompt render failed: {e}"}, ensure_ascii=False))

    required_fields = ["guidelines", "anti_patterns", "canonical_maps"]
    field_validators = {
        "guidelines": lambda v: isinstance(v, list),
        "anti_patterns": lambda v: isinstance(v, list),
        "canonical_maps": lambda v: isinstance(v, list),
    }
    messages = [{"role": "user", "content": user_prompt}]
    corrected_json, status = process_with_format_guarantee(
        llm_client=llm,
        messages=messages,
        required_fields=required_fields,
        field_validators=field_validators,
        max_retries=2,
        repair_template=general_repair_template,
    )
    if status == "success":
        return _normalize_json_unicode_text(corrected_json)
    return correct_json_format(json.dumps({"error": "distilled memory extraction failed"}, ensure_ascii=False))


class DistilledMemoryExtractor:
    """
    YAML-driven distilled memory extractor.

    Prompt YAML:
      - memory/distill_extraction_memory

    Input params JSON:
      {
        "source_payload": "{...json string...}",
        "requested_max": "30"
      }
    """

    def __init__(self, prompt_loader, llm, prompt_id: str = "memory/distill_extraction_memory"):
        if llm is None:
            raise ValueError("llm must be provided")
        if prompt_loader is None:
            raise ValueError("prompt_loader must be provided")
        self.llm = llm
        self.prompt_loader = prompt_loader
        self.prompt_id = prompt_id
        self.repair_template = general_repair_template

        self.required_fields = ["guidelines", "anti_patterns", "canonical_maps"]
        self.field_validators = {
            "guidelines": lambda v: isinstance(v, list),
            "anti_patterns": lambda v: isinstance(v, list),
            "canonical_maps": lambda v: isinstance(v, list),
        }

    def call(self, params: str, **kwargs) -> str:
        try:
            p = json.loads(params) if isinstance(params, str) else (params or {})
            source_payload = str(p.get("source_payload", "") or "").strip()
            requested_max = str(p.get("requested_max", "") or "").strip()
        except Exception as e:
            logger.error("DistilledMemoryExtractor params parse failed: %s", e)
            return correct_json_format(json.dumps({"error": f"params parse failed: {e}"}, ensure_ascii=False))

        if not source_payload:
            return correct_json_format(json.dumps({"error": "missing required field: source_payload"}, ensure_ascii=False))
        if not requested_max:
            requested_max = "30"

        return extract_distilled_memories_with_guard(
            llm=self.llm,
            prompt_loader=self.prompt_loader,
            source_payload=source_payload,
            requested_max=int(requested_max),
            prompt_id=self.prompt_id,
        )

from __future__ import annotations

import json
import logging
from typing import Any, List

from core.utils.general_utils import load_json, safe_str
from core.utils.function_manager import process_with_format_guarantee
from core.utils.general_text import general_repair_template
from core.utils.format import correct_json_format

logger = logging.getLogger(__name__)

TASK_SCHEMA_PATH = "core/task_settings/insight_extraction_task.json"


def render_insight_task_spec(task: dict) -> str:
    lines = []
    desc = safe_str(task.get("description", "")).strip()
    if desc:
        lines.append(desc)
        lines.append("")

    guidelines = task.get("guidelines", [])
    if isinstance(guidelines, list) and guidelines:
        lines.append("Guidelines:")
        for g in guidelines:
            gg = safe_str(g).strip()
            if gg:
                lines.append(f"- {gg}")

    return "\n".join(lines).strip()


def _valid_facts(x: Any) -> bool:
    if not isinstance(x, list):
        return False
    for s in x:
        if not isinstance(s, str):
            return False
    return True


class InsightExtractor:
    """
    Extract explicit factual points from text.

    Prompt YAML:
      - text_processing/extract_insights

    Params JSON:
      {
        "doc_type": "narrative|general",
        "text": "..."
      }

    Output JSON:
      {
        "facts": ["...", "..."]
      }
    """

    def __init__(
        self,
        prompt_loader,
        llm,
        task_schema_path: str = TASK_SCHEMA_PATH,
        prompt_id: str = "text_processing/extract_insights",
    ):
        if llm is None:
            raise ValueError("llm must be provided")
        if prompt_loader is None:
            raise ValueError("prompt_loader must be provided")

        self.prompt_loader = prompt_loader
        self.llm = llm
        self.prompt_id = prompt_id
        self.repair_template = general_repair_template

        schema = load_json(task_schema_path)
        if not isinstance(schema, dict):
            raise ValueError("insight_extraction_task.json must be a dict")
        self.task_schema = schema

        self.required_fields = ["facts"]
        self.field_validators = {"facts": _valid_facts}

    def call(self, params: str, **kwargs) -> str:
        try:
            params_dict = json.loads(params) if isinstance(params, str) else (params or {})
            doc_type = safe_str(params_dict.get("doc_type", "general")).strip()
            text = safe_str(params_dict.get("text", "")).strip()
        except Exception as e:
            logger.error(f"参数解析失败: {e}")
            return correct_json_format(
                json.dumps({"error": f"参数解析失败: {str(e)}", "facts": []}, ensure_ascii=False)
            )

        if not text:
            return correct_json_format(
                json.dumps({"error": "missing required field: text", "facts": []}, ensure_ascii=False)
            )

        # narrative covers novel + screenplay
        if doc_type not in self.task_schema:
            doc_type = "general"

        task = self.task_schema.get(doc_type, {})
        task_spec = render_insight_task_spec(task)

        try:
            user_prompt = self.prompt_loader.render(
                self.prompt_id,
                static_values={"task_spec": task_spec},
                task_values={
                    "doc_type": doc_type,
                    "text": text,
                },
                strict=True,
            )
        except Exception as e:
            logger.error(f"extract_insights prompt render failed: {e}")
            return correct_json_format(
                json.dumps({"error": f"prompt render failed: {str(e)}", "facts": []}, ensure_ascii=False)
            )

        messages = [{"role": "user", "content": user_prompt}]

        corrected_json, status = process_with_format_guarantee(
            llm_client=self.llm,
            messages=messages,
            required_fields=self.required_fields,
            field_validators=self.field_validators,
            max_retries=2,
            repair_template=self.repair_template,
        )

        if status == "success":
            return correct_json_format(corrected_json)

        return correct_json_format(
            json.dumps({"error": "fact extraction failed", "facts": []}, ensure_ascii=False)
        )

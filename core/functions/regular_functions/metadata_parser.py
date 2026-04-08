from __future__ import annotations

import json
import logging
from typing import Any, Dict, Tuple

from core.utils.general_utils import safe_str, load_json
from core.utils.function_manager import process_with_format_guarantee
from core.utils.general_text import general_repair_template
from core.utils.format import correct_json_format

logger = logging.getLogger(__name__)


def render_task_spec(task: Dict[str, Any]) -> str:
    """
    Render a metadata parsing task into a readable instruction block.
    """
    lines = []

    desc = task.get("description")
    if desc:
        lines.append(f"Task: {desc}")
        lines.append("")

    fields = task.get("fields", [])
    if fields:
        lines.append("Fields to extract:")
        for f in fields:
            key = f.get("key")
            d = f.get("description", "")
            if key:
                lines.append(f"- {key}: {d}")

    return "\n".join(lines)


def _safe_json_response(payload: Dict[str, Any]) -> str:
    return correct_json_format(json.dumps(payload, ensure_ascii=False))


class MetadataParser:
    """
    YAML-driven metadata parser.

    Prompt YAML:
      - text_processing/parse_metadata

    Input params JSON:
      {
        "doc_type": "novel | screenplay | general | ...",
        "title": "...",
        "subtitle": "... (optional)",
        "text": "... (optional)"
      }

    Output JSON:
      {
        "metadata": {
          "field": "value or null"
        }
      }
    """

    def __init__(
        self,
        prompt_loader,
        llm,
        task_schema_path: str = "core/task_settings/metadata_parsing_task.json",
        prompt_id: str = "text_processing/parse_metadata",
    ):
        if llm is None:
            raise ValueError("llm must be provided")
        if prompt_loader is None:
            raise ValueError("prompt_loader must be provided")

        self.llm = llm
        self.prompt_loader = prompt_loader
        self.prompt_id = prompt_id
        self.repair_template = general_repair_template

        # no hard required fields, schema varies by doc_type
        self.required_fields = ["metadata"]
        self.field_validators = {
            "metadata": lambda x: isinstance(x, dict)
        }
        self.title_metadata_required_fields = ["title", "metadata"]
        self.title_metadata_field_validators = {
            "title": lambda x: isinstance(x, str) and bool(safe_str(x).strip()),
            "metadata": lambda x: isinstance(x, dict),
        }

        self.task_schema = load_json(task_schema_path)
        if not isinstance(self.task_schema, dict):
            raise ValueError("metadata_parsing_task.json must be a dict")

    def _parse_common_params(self, params: Any) -> Tuple[str, str, str, str]:
        params_dict = json.loads(params) if isinstance(params, str) else (params or {})
        doc_type = safe_str(params_dict.get("doc_type", "general")).strip()
        title = safe_str(params_dict.get("title", "")).strip()
        subtitle = safe_str(params_dict.get("subtitle", "")).strip()
        text = safe_str(params_dict.get("text", "")).strip()
        return doc_type, title, subtitle, text

    def _get_task(self, doc_type: str) -> Dict[str, Any] | None:
        task = self.task_schema.get(doc_type)
        if task is None:
            logger.warning(f"Unknown doc_type '{doc_type}', fallback to 'general'")
            task = self.task_schema.get("general")
        return task

    def _run_prompt(
        self,
        *,
        prompt_id: str,
        task_spec_text: str,
        doc_type: str,
        title: str,
        subtitle: str,
        text: str,
        required_fields,
        field_validators,
        fallback_payload: Dict[str, Any],
        error_prefix: str,
    ) -> str:
        try:
            user_prompt = self.prompt_loader.render(
                prompt_id,
                static_values={
                    "task_spec": task_spec_text
                },
                task_values={
                    "doc_type": doc_type,
                    "title": title,
                    "subtitle": f"[Subtitle]\n{subtitle}\n" if subtitle else "",
                    "text": text,
                },
                strict=True,
            )
        except Exception as e:
            logger.error(f"{error_prefix} prompt render failed: {e}")
            fallback = dict(fallback_payload)
            fallback["error"] = f"prompt render failed: {str(e)}"
            return _safe_json_response(fallback)

        corrected_json, status = process_with_format_guarantee(
            llm_client=self.llm,
            messages=[{"role": "user", "content": user_prompt}],
            required_fields=required_fields,
            field_validators=field_validators,
            max_retries=2,
            repair_template=self.repair_template,
        )
        if status == "success":
            return correct_json_format(corrected_json)

        fallback = dict(fallback_payload)
        fallback["error"] = f"{error_prefix} failed"
        return _safe_json_response(fallback)

    def call(self, params: str, **kwargs) -> str:
        try:
            doc_type, title, subtitle, text = self._parse_common_params(params)
        except Exception as e:
            logger.error(f"参数解析失败: {e}")
            return _safe_json_response({"error": f"参数解析失败: {str(e)}", "metadata": {}})

        if not title:
            return _safe_json_response({"error": "missing required field: title", "metadata": {}})

        task = self._get_task(doc_type)
        if task is None:
            return _safe_json_response({"error": "No metadata task spec available", "metadata": {}})

        task_spec_text = render_task_spec(task)
        return self._run_prompt(
            prompt_id=self.prompt_id,
            task_spec_text=task_spec_text,
            doc_type=doc_type,
            title=title,
            subtitle=subtitle,
            text=text,
            required_fields=self.required_fields,
            field_validators=self.field_validators,
            fallback_payload={"metadata": {}},
            error_prefix="metadata parsing",
        )

    def generate_title_and_metadata(self, params: str, **kwargs) -> str:
        try:
            doc_type, title, subtitle, text = self._parse_common_params(params)
        except Exception as e:
            logger.error(f"参数解析失败: {e}")
            return _safe_json_response(
                {
                    "error": f"参数解析失败: {str(e)}",
                    "title": safe_str((params or {}).get("title", "")) if isinstance(params, dict) else "",
                    "metadata": {},
                }
            )

        if not text:
            return _safe_json_response(
                {"error": "missing required field: text", "title": title, "metadata": {}}
            )

        task = self._get_task(doc_type)
        if task is None:
            return _safe_json_response(
                {"error": "No metadata task spec available", "title": title, "metadata": {}}
            )

        task_spec_text = render_task_spec(task)
        fallback_title = title or "Untitled Segment"
        return self._run_prompt(
            prompt_id="text_processing/generate_title_and_metadata",
            task_spec_text=task_spec_text,
            doc_type=doc_type,
            title=title,
            subtitle=subtitle,
            text=text,
            required_fields=self.title_metadata_required_fields,
            field_validators=self.title_metadata_field_validators,
            fallback_payload={"title": fallback_title, "metadata": {}},
            error_prefix="title and metadata generation",
        )

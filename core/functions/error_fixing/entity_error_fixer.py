from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

from core.utils.function_manager import process_with_format_guarantee
from core.utils.general_text import general_repair_template
from core.utils.format import correct_json_format
from core.utils.general_utils import load_json


logger = logging.getLogger(__name__)

TASK_PATH = "core/task_settings/entity_extraction_task.json"


def _parse_candidate_types(raw: Any) -> List[str]:
    """
    Compatible input forms:
      - list[str]
      - JSON array string: '["Character","Object"]'
      - comma/newline separated string: 'Character,Object' or 'Character\nObject'

    Return: de-duplicated, cleaned list preserving order.
    """
    if raw is None:
        return []

    items: List[Any]
    if isinstance(raw, list):
        items = raw
    elif isinstance(raw, str):
        s = raw.strip()
        if not s:
            return []
        # Try JSON list first
        try:
            v = json.loads(s)
            if isinstance(v, list):
                items = v
            else:
                items = [s]
        except Exception:
            # Split by commas/newlines
            parts: List[str] = []
            normalized = s.replace("\r\n", "\n").replace("\r", "\n")
            for line in normalized.split("\n"):
                parts.extend([p.strip() for p in line.split(",")])
            items = [p for p in parts if p]
    else:
        items = [str(raw)]

    seen = set()
    out: List[str] = []
    for x in items:
        if not isinstance(x, str):
            x = str(x)
        x = x.strip()
        if not x or x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


class EntityErrorFixer:
    """
    YAML-driven Entity Error Fixer (problem_solving).

    - Prompt YAML: problem_solving/entity_error_fix
    - No system prompt. No previous-round context.

    Params JSON:
      - content: str
      - candidate_entity_types: list[str] | str
      - feedback: str

    Note:
      - candidate_entity_descriptions is built as a STATIC var by reading
        core/schema/default_task_descriptions.json (renamed from full_entity_schema.json).
    """

    def __init__(
        self,
        prompt_loader,
        llm,
        task_path: str = TASK_PATH,
        prompt_id: str = "problem_solving/fix_entity_error",
    ):
        if llm is None:
            raise ValueError("llm must be provided")
        if prompt_loader is None:
            raise ValueError("prompt_loader must be provided")

        self.llm = llm
        self.prompt_loader = prompt_loader
        self.task_path = task_path
        self.prompt_id = prompt_id

        # format guarantee config
        self.required_fields: List[str] = []
        self.field_validators: Dict[str, Any] = {}
        self.repair_template = general_repair_template

        schema = load_json(task_path)
        if not isinstance(schema, list):
            raise ValueError(f"Invalid schema: expected list at {task_path}")
        self.schema: List[Dict[str, Any]] = schema

        # entity type -> description
        self.entity_type_description: Dict[str, str] = {}
        self._index_type_descriptions()

    def _index_type_descriptions(self) -> None:
        """
        Build map: TypeName -> type.description from ALL task blocks.
        If duplicates exist, keep the first non-empty description encountered.
        """
        out: Dict[str, str] = {}
        for task_block in self.schema:
            if not isinstance(task_block, dict):
                continue
            for t in (task_block.get("types") or []):
                if not isinstance(t, dict):
                    continue
                type_name = (t.get("type") or "").strip()
                if not type_name:
                    continue
                desc = (t.get("description") or "").strip()
                if type_name not in out and desc:
                    out[type_name] = desc
        self.entity_type_description = out

    def _build_candidate_descriptions_block(self, candidate_types: List[str]) -> str:
        """
        Render:
          Character: ...
          Object: ...
        """
        lines: List[str] = []
        for t in candidate_types:
            desc = (self.entity_type_description.get(t) or "").strip()
            if desc:
                lines.append(f"{t}: {desc}")
            else:
                lines.append(f"{t}:")
        return "\n".join(lines).strip()

    def call(self, params: str, **kwargs) -> str:
        try:
            params_dict = json.loads(params)
        except Exception as e:
            logger.error(f"参数解析失败: {e}")
            return correct_json_format(json.dumps({"error": f"参数解析失败: {str(e)}"}, ensure_ascii=False))

        content = params_dict.get("text", "")
        candidate_entity_types_raw = params_dict.get("candidate_entity_types", "")
        feedback = params_dict.get("feedback", "")

        if not isinstance(content, str):
            content = str(content or "")
        if not isinstance(feedback, str):
            feedback = str(feedback or "")

        # compat: list or str
        candidate_list = _parse_candidate_types(candidate_entity_types_raw)

        # static var built from schema
        candidate_desc_text = self._build_candidate_descriptions_block(candidate_list)

        # keep the prompt's candidate_entity_types readable and unambiguous
        candidate_types_for_prompt: str
        if candidate_list:
            candidate_types_for_prompt = json.dumps(candidate_list, ensure_ascii=False)
        else:
            candidate_types_for_prompt = str(candidate_entity_types_raw or "")

        try:
            user_prompt = self.prompt_loader.render(
                self.prompt_id,
                static_values={
                    "candidate_entity_descriptions": candidate_desc_text,
                },
                task_values={
                    "content": content,
                    "candidate_entity_types": candidate_types_for_prompt,
                    "feedback": feedback,
                },
                strict=True,
            )

            messages = [{"role": "user", "content": user_prompt}]

            corrected_json, status = process_with_format_guarantee(
                llm_client=self.llm,
                messages=messages,
                required_fields=self.required_fields,
                field_validators=self.field_validators,
                max_retries=3,
                repair_template=self.repair_template,
            )

            if status == "success":
                return correct_json_format(corrected_json)

            return correct_json_format(json.dumps({"error": "实体缺失修复失败"}, ensure_ascii=False))

        except Exception as e:
            logger.error(f"实体缺失修复过程中出现异常: {e}")
            return correct_json_format(json.dumps({"error": f"实体缺失修复失败: {str(e)}"}, ensure_ascii=False))

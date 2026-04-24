from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

from core.utils.function_manager import process_with_format_guarantee
from core.utils.general_text import general_rules, general_repair_template
from core.utils.format import correct_json_format
from core.utils.general_utils import join_bullets, load_json, pretty_json, safe_list, safe_str

logger = logging.getLogger(__name__)

TASK_SCHEMA_PATH = "task_specs/task_settings/open_entity_extraction_task.json"
_LOW_VALUE_OPEN_ENTITY_NAMES = {
    "book",
    "box",
    "bag",
    "coat",
    "cup",
    "cups",
    "car",
    "cars",
    "desk",
    "door",
    "day",
    "night",
    "morning",
    "afternoon",
    "evening",
}


def _looks_low_value_open_entity(name: str) -> bool:
    norm = safe_str(name).strip().lower()
    if not norm:
        return True
    if norm in _LOW_VALUE_OPEN_ENTITY_NAMES:
        return True
    if any(marker in norm for marker in ["a room where", "the room where", "the area around", "center of the room", "entrance of"]):
        return True
    if norm.startswith(("a ", "an ", "the ")) and len(norm.split()) <= 3:
        return True
    return False


def _parse_known_names(raw: str) -> set[str]:
    names = set()
    for line in safe_str(raw).splitlines():
        name = line.strip()
        if not name:
            continue
        names.add(name.lower())
    return names


class OpenEntityExtractor:
    def __init__(
        self,
        prompt_loader: Any,
        llm: Any,
        task_schema_path: str = TASK_SCHEMA_PATH,
        max_retries: int = 3,
    ) -> None:
        if llm is None:
            raise ValueError("llm must be provided")
        if prompt_loader is None:
            raise ValueError("prompt_loader must be provided")
        self.llm = llm
        self.prompt_loader = prompt_loader
        self.task_schema_path = task_schema_path
        self.system_prompt_id = "knowledge_extraction/system_prompt"
        self.prompt_id = "knowledge_extraction/extract_open_entities"
        self.required_fields: List[str] = []
        self.field_validators: Dict[str, Any] = {}
        self.repair_template = general_repair_template
        self.max_retries = max(0, int(max_retries))

        task_schema = load_json(self.task_schema_path)
        if not isinstance(task_schema, list):
            raise ValueError(f"Invalid task schema: expected list at {self.task_schema_path}")
        self.task_schema = [x for x in task_schema if isinstance(x, dict)]
        self.task_schema_map = {safe_str(x.get("task")): x for x in self.task_schema if safe_str(x.get("task"))}

    def _get_task_block(self) -> Dict[str, Any]:
        tb = self.task_schema_map.get("open_entity_extraction")
        if not isinstance(tb, dict):
            raise RuntimeError("Task block 'open_entity_extraction' not found")
        return tb

    def _build_static_vars(self, task_block: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "task_name": safe_str(task_block.get("task_name")).strip(),
            "task_goal": safe_str(task_block.get("task_goal")).strip(),
            "global_constraints": join_bullets(task_block.get("global_constraints")),
            "extraction_rules_block": join_bullets(task_block.get("extraction_rules")),
            "output_fields_block": "\n".join([f'- "{x}"' for x in safe_list(task_block.get("output_fields")) if safe_str(x)]),
            "few_shot_samples": pretty_json(task_block.get("few_shot", {}).get("examples", [])),
        }

    def _postprocess(self, payload: Any, known_names: set[str] | None = None) -> List[Dict[str, str]]:
        out: List[Dict[str, str]] = []
        seen = set()
        known_names = known_names or set()
        for item in safe_list(payload):
            if not isinstance(item, dict):
                continue
            name = safe_str(item.get("name")).strip()
            description = safe_str(item.get("description")).strip()
            if not name or not description:
                continue
            if _looks_low_value_open_entity(name):
                continue
            key = name.lower()
            if key in seen or key in known_names:
                continue
            seen.add(key)
            out.append({"name": name, "description": description})
        return out

    def call(self, params: str, **kwargs) -> str:
        try:
            params_dict = json.loads(params)
        except Exception as e:
            logger.error(f"Params parse failed: {e}")
            return correct_json_format(json.dumps({"error": f"Params parse failed: {str(e)}"}, ensure_ascii=False))

        text = safe_str(params_dict.get("text"))
        known_entities = safe_str(params_dict.get("known_entities"))
        memory_context = safe_str(params_dict.get("memory_context"))

        try:
            task_block = self._get_task_block()
            static_vars = self._build_static_vars(task_block)

            messages: List[Dict[str, str]] = []
            system_prompt = self.prompt_loader.render(
                self.system_prompt_id,
                static_values={"rules": general_rules, "json_only": ""},
                task_values={},
                strict=False,
            )
            messages.append({"role": "system", "content": system_prompt})
            user_prompt = self.prompt_loader.render(
                self.prompt_id,
                static_values=static_vars,
                task_values={"text": text, "known_entities": known_entities, "memory_context": memory_context},
                strict=False,
            )
            messages.append({"role": "user", "content": user_prompt})
            corrected_json, status = process_with_format_guarantee(
                llm_client=self.llm,
                messages=messages,
                required_fields=self.required_fields,
                field_validators=self.field_validators,
                max_retries=self.max_retries,
                repair_template=self.repair_template,
            )
            if status != "success":
                return correct_json_format(json.dumps({"error": "Open entity extraction failed"}, ensure_ascii=False))
            known_names = _parse_known_names(known_entities)
            try:
                parsed = json.loads(corrected_json)
            except Exception:
                parsed = []
            return correct_json_format(json.dumps(self._postprocess(parsed, known_names=known_names), ensure_ascii=False))
        except Exception as e:
            logger.error(f"Open entity extraction exception: {e}")
            return correct_json_format(json.dumps({"error": f"Open entity extraction failed: {str(e)}"}, ensure_ascii=False))

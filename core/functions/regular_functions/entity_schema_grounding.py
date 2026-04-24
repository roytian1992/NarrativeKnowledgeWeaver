from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

from core.utils.function_manager import process_with_format_guarantee
from core.utils.general_text import general_rules, general_repair_template
from core.utils.format import correct_json_format
from core.utils.general_utils import join_bullets, load_json, pretty_json, safe_list, safe_str

logger = logging.getLogger(__name__)

TASK_SCHEMA_PATH = "task_specs/task_settings/entity_schema_grounding_task.json"
ENTITY_SCHEMA_PATH = "task_specs/schema/default_entity_schema.json"


class EntitySchemaGrounder:
    def __init__(
        self,
        prompt_loader: Any,
        llm: Any,
        task_schema_path: str = TASK_SCHEMA_PATH,
        entity_schema_path: str = ENTITY_SCHEMA_PATH,
        max_retries: int = 3,
    ) -> None:
        if llm is None:
            raise ValueError("llm must be provided")
        if prompt_loader is None:
            raise ValueError("prompt_loader must be provided")
        self.llm = llm
        self.prompt_loader = prompt_loader
        self.task_schema_path = task_schema_path
        self.entity_schema_path = entity_schema_path
        self.system_prompt_id = "knowledge_extraction/system_prompt"
        self.prompt_id = "knowledge_extraction/ground_open_entities"
        self.required_fields: List[str] = []
        self.field_validators: Dict[str, Any] = {}
        self.repair_template = general_repair_template
        self.max_retries = max(0, int(max_retries))

        task_schema = load_json(self.task_schema_path)
        if not isinstance(task_schema, list):
            raise ValueError(f"Invalid task schema: expected list at {self.task_schema_path}")
        self.task_schema = [x for x in task_schema if isinstance(x, dict)]
        self.task_schema_map = {safe_str(x.get("task")): x for x in self.task_schema if safe_str(x.get("task"))}

        entity_schema = load_json(self.entity_schema_path)
        if not isinstance(entity_schema, list):
            raise ValueError(f"Invalid entity schema: expected list at {self.entity_schema_path}")
        self.entity_schema = [x for x in entity_schema if isinstance(x, dict)]

    def _get_task_block(self) -> Dict[str, Any]:
        tb = self.task_schema_map.get("entity_schema_grounding")
        if not isinstance(tb, dict):
            raise RuntimeError("Task block 'entity_schema_grounding' not found")
        return tb

    def _build_type_definitions(self) -> str:
        allowed = {"Character", "Object", "Concept", "Location", "TimePoint"}
        lines = []
        for item in self.entity_schema:
            t = safe_str(item.get("type")).strip()
            if t not in allowed:
                continue
            desc = safe_str(item.get("description")).strip()
            lines.append(f"{t}: {desc}")
        return "\n".join(lines)

    def _build_static_vars(self, task_block: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "task_name": safe_str(task_block.get("task_name")).strip(),
            "task_goal": safe_str(task_block.get("task_goal")).strip(),
            "global_constraints": join_bullets(task_block.get("global_constraints")),
            "grounding_rules_block": join_bullets(task_block.get("grounding_rules")),
            "type_definitions": self._build_type_definitions(),
            "output_fields_block": "\n".join([f'- "{x}"' for x in safe_list(task_block.get("output_fields")) if safe_str(x)]),
            "few_shot_samples": pretty_json(task_block.get("few_shot", {}).get("examples", [])),
        }

    def _postprocess(self, payload: Any) -> List[Dict[str, str]]:
        out: List[Dict[str, str]] = []
        seen = set()
        allowed = {"Character", "Object", "Concept", "Location", "TimePoint"}
        for item in safe_list(payload):
            if not isinstance(item, dict):
                continue
            name = safe_str(item.get("name")).strip()
            description = safe_str(item.get("description")).strip()
            etype = safe_str(item.get("type")).strip()
            scope = safe_str(item.get("scope")).strip() or "local"
            if not name or not description or etype not in allowed:
                continue
            key = (name.lower(), etype)
            if key in seen:
                continue
            seen.add(key)
            out.append({"name": name, "description": description, "type": etype, "scope": scope})
        return out

    def call(self, params: str, **kwargs) -> str:
        try:
            params_dict = json.loads(params)
        except Exception as e:
            logger.error(f"Params parse failed: {e}")
            return correct_json_format(json.dumps({"error": f"Params parse failed: {str(e)}"}, ensure_ascii=False))

        text = safe_str(params_dict.get("text"))
        open_entities = params_dict.get("open_entities", [])
        memory_context = safe_str(params_dict.get("memory_context"))
        if not isinstance(open_entities, list):
            open_entities = []

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
                task_values={
                    "text": text,
                    "open_entities": pretty_json(open_entities),
                    "memory_context": memory_context,
                },
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
                return correct_json_format(json.dumps({"error": "Entity schema grounding failed"}, ensure_ascii=False))
            try:
                parsed = json.loads(corrected_json)
            except Exception:
                parsed = []
            return correct_json_format(json.dumps(self._postprocess(parsed), ensure_ascii=False))
        except Exception as e:
            logger.error(f"Entity schema grounding exception: {e}")
            return correct_json_format(json.dumps({"error": f"Entity schema grounding failed: {str(e)}"}, ensure_ascii=False))

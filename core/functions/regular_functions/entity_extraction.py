from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from core.utils.function_manager import process_with_format_guarantee
from core.utils.general_text import general_rules, general_repair_template
from core.utils.format import correct_json_format
from core.utils.general_utils import load_json, join_bullets, safe_dict, safe_list

logger = logging.getLogger(__name__)

# Task descriptions no longer include samples
TASK_SCHEMA_PATH = "core/task_settings/entity_extraction_task.json"
# Samples live in the entity schema file
ENTITY_SCHEMA_PATH = "core/schema/default_entity_schema.json"


def _collect_few_shot_samples_from_entity_schema(
    *,
    entity_schema: List[Dict[str, Any]],
    allowed_types: List[str],
    per_type_k: int = 1,
) -> List[Dict[str, Any]]:
    """
    Collect first k samples for each allowed type from default_entity_schema.json.
    """
    allowed = {t.strip() for t in allowed_types if isinstance(t, str) and t.strip()}
    if not allowed or per_type_k <= 0:
        return []

    out: List[Dict[str, Any]] = []
    for type_block in safe_list(entity_schema):
        if not isinstance(type_block, dict):
            continue
        t_name = (type_block.get("type") or "").strip()
        if t_name not in allowed:
            continue
        samples = safe_list(type_block.get("samples"))
        for s in samples[:per_type_k]:
            if isinstance(s, dict):
                out.append(s)
    return out


class EntityExtractor:
    """
    YAML-driven Entity Extractor.

    - Reads task schema: core/schema/default_task_descriptions.json
    - Reads samples schema: core/schema/default_entity_schema.json
    - Renders prompts using injected YAMLPromptLoader:
        - system prompt YAML
        - entity extraction YAML
        - optional context YAML blocks:
            - previous_results + feedbacks
            - extracted_entities
    - Calls LLM with format guarantee and returns JSON-only string.
    """

    def __init__(
        self,
        prompt_loader,
        llm,
        task_schema_path: str = TASK_SCHEMA_PATH,
        entity_schema_path: str = ENTITY_SCHEMA_PATH,
        per_type_k: int = 1,
    ):
        if llm is None:
            raise ValueError("llm must be provided")
        if prompt_loader is None:
            raise ValueError("prompt_loader must be provided")

        self.llm = llm
        self.prompt_loader = prompt_loader

        self.task_schema_path = task_schema_path
        self.entity_schema_path = entity_schema_path

        self.system_prompt_id = "knowledge_extraction/system_prompt"
        self.entity_prompt_id = "knowledge_extraction/extract_entities"
        self.prev_round_prompt_id = "knowledge_extraction/context_previous_round"
        self.already_extracted_prompt_id = "knowledge_extraction/context_already_extracted"

        self.per_type_k = max(0, int(per_type_k))

        # format guarantee config
        self.required_fields: List[str] = []
        self.field_validators: Dict[str, Any] = {}
        self.repair_template = general_repair_template

        # -------------------------
        # load task schema
        # -------------------------
        task_schema = load_json(self.task_schema_path)
        if not isinstance(task_schema, list):
            raise ValueError(f"Invalid task schema: expected list at {self.task_schema_path}")
        self.task_schema: List[Dict[str, Any]] = task_schema

        # -------------------------
        # load entity schema (holds samples)
        # -------------------------
        entity_schema = load_json(self.entity_schema_path)
        if not isinstance(entity_schema, list):
            raise ValueError(f"Invalid entity schema: expected list at {self.entity_schema_path}")
        self.entity_schema: List[Dict[str, Any]] = entity_schema

        # task -> task_schema
        self.task_schema_map: Dict[str, Dict[str, Any]] = {}
        self._index_task_schema()

    def _index_task_schema(self) -> None:
        for task_block in self.task_schema:
            if not isinstance(task_block, dict):
                continue
            task = (task_block.get("task") or "").strip()
            if not task:
                continue
            self.task_schema_map[task] = task_block

    def _entity_type_to_task(self, entity_group: str) -> str: # 
        k = (entity_group or "").lower().strip()
        if k == "induced" or k == "induced_entity_extraction":
            return "induced_entity_extraction"
        if k == "anchor" or k == "anchor_entity_extraction":
            return "anchor_entity_extraction"
        if k == "general" or k == "general_entity_extraction":
            return "general_entity_extraction"
        return entity_group

    def _build_static_vars_from_task_schema(self, task_schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build static variables required by core/prompts/knowledge_extraction/entity_extraction.yaml
        based on a task block inside default_task_descriptions.json.

        Note:
        - task schema has no samples
        - few-shot samples are collected from default_entity_schema.json by matching allowed types
        """
        task_name = (task_schema.get("task_name") or "").strip()
        task_goal = (task_schema.get("task_goal") or "").strip()
        global_constraints = join_bullets(task_schema.get("global_constraints"))

        types = task_schema.get("types") or []
        if not isinstance(types, list):
            types = []

        # allowed_types are the type names inside this task block, used to fetch samples
        allowed_types: List[str] = []
        for t in types:
            if not isinstance(t, dict):
                continue
            tn = (t.get("type") or "").strip()
            if tn:
                allowed_types.append(tn)

        # output_fields: take first declared
        output_fields: List[str] = []
        for t in types:
            if not isinstance(t, dict):
                continue
            of = t.get("output_fields")
            if isinstance(of, list) and of:
                output_fields = [x for x in of if isinstance(x, str) and x.strip()]
                break
        if not output_fields:
            output_fields = ["type", "name", "description"]

        output_fields_block = "\n".join([f'- "{k}"' for k in output_fields]).strip()

        # type_definitions: compact but schema-driven
        type_blocks: List[str] = []
        for t in types:
            if not isinstance(t, dict):
                continue
            t_name = (t.get("type") or "").strip()
            t_desc = (t.get("description") or "").strip()
            if not t_name:
                continue

            parts: List[str] = []
            parts.append(f"{t_name}: {t_desc}" if t_desc else f"{t_name}:")

            naming_rules = t.get("naming_rules")
            if isinstance(naming_rules, list) and naming_rules:
                parts.append("Naming rules:\n" + join_bullets(naming_rules))

            exclusions = t.get("exclusions")
            if isinstance(exclusions, list) and exclusions:
                parts.append("Exclusions:\n" + join_bullets(exclusions))

            scope_rules = t.get("scope_rules")
            if isinstance(scope_rules, dict) and scope_rules:
                sr_lines: List[str] = []
                g = scope_rules.get("global")
                l = scope_rules.get("local")
                if isinstance(g, str) and g.strip():
                    sr_lines.append(f"global: {g.strip()}")
                if isinstance(l, str) and l.strip():
                    sr_lines.append(f"local: {l.strip()}")
                if sr_lines:
                    parts.append("Scope rules:\n" + join_bullets(sr_lines))

            granularity = t.get("granularity_rules")
            if isinstance(granularity, list) and granularity:
                parts.append("Granularity:\n" + join_bullets(granularity))

            type_blocks.append("\n".join([p for p in parts if p]).strip())

        type_definitions = "\n\n".join([tb for tb in type_blocks if tb]).strip()

        # few-shot samples: collected from entity schema file, filtered by allowed_types
        few_shot_list: List[Dict[str, Any]] = []
        if self.per_type_k > 0 and allowed_types:
            few_shot_list = _collect_few_shot_samples_from_entity_schema(
                entity_schema=self.entity_schema,
                allowed_types=allowed_types,
                per_type_k=self.per_type_k,
            )
        few_shot_samples = json.dumps(few_shot_list, ensure_ascii=False, indent=2)

        return {
            "task_name": task_name,
            "task_goal": task_goal,
            "global_constraints": global_constraints,
            "type_definitions": type_definitions,
            "output_fields_block": output_fields_block,
            "few_shot_samples": few_shot_samples,
        }

    def call(self, params: str, **kwargs) -> str:
        """
        params JSON:
          - text: str
          - entity_group: one of [general, event, time_and_location]
          - previous_results (optional): str
          - feedbacks (optional): str
          - extracted_entities (optional): str/json
        """
        try:
            params_dict = json.loads(params)
        except Exception as e:
            logger.error(f"参数解析失败: {e}")
            return correct_json_format(json.dumps({"error": f"参数解析失败: {str(e)}"}, ensure_ascii=False))

        text = params_dict.get("text", "")
        entity_type = params_dict.get("entity_group")
        memory_context = str(params_dict.get("memory_context") or "").strip()
        previous_results = params_dict.get("previous_results", [])
        if isinstance(previous_results, list):
            previous_results = json.dumps(previous_results, ensure_ascii=False)
        feedbacks = params_dict.get("feedbacks", [])
        if isinstance(feedbacks, list):
            feedbacks = "\n".join(feedbacks)
        extracted_entities = params_dict.get("extracted_entities", [])
        if isinstance(extracted_entities, list):
            extracted_entities = "\n".join(extracted_entities)

        if not isinstance(text, str):
            text = str(text or "")

        try:
            task = self._entity_type_to_task(entity_type)
            task_schema = self.task_schema_map.get(task)
            if not task_schema:
                raise RuntimeError(f"Task schema not found for task='{task}' in {self.task_schema_path}")

            # -------------------------
            # 1) system prompt (YAML)
            # -------------------------
            system_prompt = self.prompt_loader.render(
                self.system_prompt_id,
                static_values={
                    "rules": general_rules,
                    "json_only": "",
                },
                task_values={},
                strict=False,
            )

            messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]

            # -------------------------
            # 2) optional context: previous round (YAML)
            # -------------------------
            if previous_results and feedbacks:
                ctx_prev = self.prompt_loader.render(
                    self.prev_round_prompt_id,
                    static_values={},
                    task_values={
                        "previous_results": previous_results,
                        "feedbacks": feedbacks,
                    },
                    strict=False,
                )
                messages.append({"role": "user", "content": ctx_prev})

            # -------------------------
            # 3) optional context: already extracted entities (YAML)
            # -------------------------
            if extracted_entities:
                ctx_extracted = self.prompt_loader.render(
                    self.already_extracted_prompt_id,
                    static_values={
                        "default_item_type": "entities",
                    },
                    task_values={
                        "item_type": "entities",
                        "extracted_items": str(extracted_entities),
                    },
                    strict=False,
                )
                messages.append({"role": "user", "content": ctx_extracted})

            # -------------------------
            # 4) main entity extraction prompt (YAML)
            # -------------------------
            static_vars = self._build_static_vars_from_task_schema(task_schema)

            user_prompt = self.prompt_loader.render(
                self.entity_prompt_id,
                static_values=static_vars,
                task_values={
                    "text": text,
                    "memory_context": memory_context,
                },
                strict=False,
            )
            messages.append({"role": "user", "content": user_prompt})

            # -------------------------
            # 5) format guarantee call
            # -------------------------
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

            return correct_json_format(json.dumps({"error": "实体提取失败"}, ensure_ascii=False))

        except Exception as e:
            logger.error(f"实体提取过程中出现异常: {e}")
            return correct_json_format(json.dumps({"error": f"实体提取失败: {str(e)}"}, ensure_ascii=False))

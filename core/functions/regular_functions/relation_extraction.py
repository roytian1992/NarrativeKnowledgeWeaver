from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from core.utils.function_manager import process_with_format_guarantee
from core.utils.general_text import general_rules, general_repair_template
from core.utils.format import correct_json_format

# Use your general utils
from core.utils.general_utils import (
    load_json,
    join_bullets,
    safe_str,
    safe_list,
    safe_dict,
    pretty_json,
    truncate_by_word_len,
)

logger = logging.getLogger(__name__)

TASK_SCHEMA_PATH = "core/task_settings/relation_extraction_task.json"
RELATION_SCHEMA_PATH = "core/schema/default_relation_schema.json"


def _title_from_group_key(k: str) -> str:
    s = safe_str(k).replace("_", " ").strip()
    return s.title() if s else ""


def _collect_few_shot_samples_from_group_defs(
    *,
    group_defs: List[Dict[str, Any]],
    per_type_k: int,
    max_total: Optional[int],
    strategy: str,
) -> List[Dict[str, Any]]:
    """
    Collect few-shot samples from one relation group defs.

    strategy:
      - per_relation_type_first_k: take first k samples from each relation type block
      - flatten_all: flatten all samples in order, then cap
    """
    k = max(0, int(per_type_k or 0))
    if k <= 0:
        return []

    out: List[Dict[str, Any]] = []
    strat = safe_str(strategy) or "per_relation_type_first_k"

    defs = safe_list(group_defs)

    if strat == "per_relation_type_first_k":
        for d in defs:
            if not isinstance(d, dict):
                continue
            samples = safe_list(d.get("samples"))
            if not samples:
                continue
            for s in samples[:k]:
                if isinstance(s, dict):
                    out.append(s)
    else:
        # flatten_all (deterministic)
        for d in defs:
            if not isinstance(d, dict):
                continue
            for s in safe_list(d.get("samples")):
                if isinstance(s, dict):
                    out.append(s)

    if isinstance(max_total, int) and max_total > 0:
        out = out[:max_total]

    return out


class RelationExtractor:
    """
    YAML-driven Relation Extractor (task-schema driven, EntityExtractor-aligned).

    Task schema:
      - core/schema/relation_extraction_task.json  (task-level goals/constraints/output/few-shot/group meta)

    Relation schema:
      - core/schema/default_relation_schema.json  (group -> list of relation type defs, includes samples)

    Prompts:
      - knowledge_extraction/system_prompt
      - knowledge_extraction/extract_relations
      - knowledge_extraction/context_previous_round
      - knowledge_extraction/context_already_extracted
    """

    def __init__(
        self,
        prompt_loader: Any,
        llm: Any,
        task_schema_path: str = TASK_SCHEMA_PATH,
        relation_schema_path: str = RELATION_SCHEMA_PATH,
        # Optional prompt-size controls (safe defaults)
        max_relation_schema_word_len: Optional[int] = None,
        max_few_shot_word_len: Optional[int] = None,
    ) -> None:
        if llm is None:
            raise ValueError("llm must be provided")
        if prompt_loader is None:
            raise ValueError("prompt_loader must be provided")

        self.llm = llm
        self.prompt_loader = prompt_loader

        self.task_schema_path = task_schema_path
        self.relation_schema_path = relation_schema_path

        self.max_relation_schema_word_len = max_relation_schema_word_len
        self.max_few_shot_word_len = max_few_shot_word_len

        # Keep consistent with EntityExtractor
        self.system_prompt_id = "knowledge_extraction/system_prompt"
        self.relation_prompt_id = "knowledge_extraction/extract_relations"
        self.previous_round_prompt_id = "knowledge_extraction/context_previous_round"
        self.already_extracted_prompt_id = "knowledge_extraction/context_already_extracted"

        # format guarantee config
        self.required_fields: List[str] = []
        self.field_validators: Dict[str, Any] = {}
        self.repair_template = general_repair_template

        # -------------------------
        # load task schema (list)
        # -------------------------
        task_schema = load_json(self.task_schema_path)
        if not isinstance(task_schema, list):
            raise ValueError(f"Invalid task schema: expected list at {self.task_schema_path}")
        self.task_schema: List[Dict[str, Any]] = [x for x in task_schema if isinstance(x, dict)]
        self.task_schema_map: Dict[str, Dict[str, Any]] = {}
        self._index_task_schema()

        # -------------------------
        # load relation schema (dict)
        # -------------------------
        relation_schema = load_json(self.relation_schema_path)
        if not isinstance(relation_schema, dict):
            raise ValueError(f"Invalid relation schema: expected dict at {self.relation_schema_path}")
        self.relation_schema: Dict[str, Any] = relation_schema

    def _index_task_schema(self) -> None:
        for tb in self.task_schema:
            task = safe_str(tb.get("task"))
            if task:
                self.task_schema_map[task] = tb

    # -------------------------------------------------------------------------
    # Schema helpers
    # -------------------------------------------------------------------------

    def _get_task_block(self) -> Dict[str, Any]:
        """
        Current design: single task named 'relation_extraction'.
        """
        tb = self.task_schema_map.get("relation_extraction")
        if not isinstance(tb, dict):
            raise RuntimeError(
                f"Task schema not found for task='relation_extraction' in {self.task_schema_path}"
            )
        return tb

    def _get_group_defs(self, relation_group: str) -> List[Dict[str, Any]]:
        key = safe_str(relation_group)
        if not key:
            raise ValueError("Missing required param: relation_group")

        defs = self.relation_schema.get(key)
        if not isinstance(defs, list):
            raise ValueError(
                f"Invalid relation_group '{key}'. Expected one of: {sorted(list(self.relation_schema.keys()))}"
            )

        return [x for x in defs if isinstance(x, dict)]

    def _find_group_block(self, task_block: Dict[str, Any], relation_group: str) -> Dict[str, Any]:
        groups = safe_list(task_block.get("relation_groups"))
        gkey = safe_str(relation_group)
        for g in groups:
            if not isinstance(g, dict):
                continue
            if safe_str(g.get("group")) == gkey:
                return g
        return {}

    def _build_relation_schema_text(self, group_defs: List[Dict[str, Any]]) -> str:
        """
        Compact, deterministic schema text. Includes only what the model needs to obey:
        type/description/from/to/persistence/direction/opposite.
        """
        blocks: List[str] = []
        for d in safe_list(group_defs):
            if not isinstance(d, dict):
                continue

            rtype = safe_str(d.get("type"))
            if not rtype:
                continue

            desc = safe_str(d.get("description"))
            frm = [safe_str(x) for x in safe_list(d.get("from")) if safe_str(x)]
            to = [safe_str(x) for x in safe_list(d.get("to")) if safe_str(x)]
            persistence = safe_str(d.get("persistence"))
            direction = safe_str(d.get("direction"))
            opposite = safe_str(d.get("opposite"))

            lines: List[str] = []
            lines.append(f"type: {rtype}")
            if desc:
                lines.append(f"description: {desc}")
            if frm:
                lines.append(f"from: [{', '.join(frm)}]")
            if to:
                lines.append(f"to: [{', '.join(to)}]")
            if persistence:
                lines.append(f"persistence: {persistence}")
            if direction:
                lines.append(f"direction: {direction}")
            if opposite:
                lines.append(f"opposite: {opposite}")

            blocks.append("\n".join(lines).strip())

        text = "\n\n".join([b for b in blocks if b]).strip()

        if isinstance(self.max_relation_schema_word_len, int) and self.max_relation_schema_word_len > 0:
            text = truncate_by_word_len(text, self.max_relation_schema_word_len, lang="auto", suffix="...")

        return text

    def _build_static_vars(
        self,
        *,
        task_block: Dict[str, Any],
        group_block: Dict[str, Any],
        group_defs: List[Dict[str, Any]],
        relation_group: str,
    ) -> Dict[str, Any]:
        # task-level
        task_name = safe_str(task_block.get("task_name"))
        task_goal = safe_str(task_block.get("task_goal"))

        global_constraints_block = join_bullets(task_block.get("global_constraints"))

        output_fields = safe_list(task_block.get("output_fields"))
        output_fields = [safe_str(x) for x in output_fields if safe_str(x)]
        if not output_fields:
            output_fields = ["subject", "object", "relation_type", "relation_name", "description"]
        output_fields_block = "\n".join([f'- "{k}"' for k in output_fields]).strip()

        # group-level
        group_name = safe_str(group_block.get("group_name")) or _title_from_group_key(relation_group)
        group_constraints_block = join_bullets(group_block.get("group_constraints"))

        # schema text from relation schema
        relation_schema_text = self._build_relation_schema_text(group_defs)

        # few-shot config from task schema
        few_shot_cfg = safe_dict(task_block.get("few_shot"))
        per_type_k = int(few_shot_cfg.get("per_type_k") or 0)
        max_total = few_shot_cfg.get("max_total")
        max_total_i: Optional[int] = None
        if isinstance(max_total, int):
            max_total_i = max_total
        else:
            # tolerate "10" as string
            try:
                s = safe_str(max_total)
                if s:
                    max_total_i = int(s)
            except Exception:
                max_total_i = None

        strategy = safe_str(few_shot_cfg.get("strategy")) or "per_relation_type_first_k"

        few_shot_list = _collect_few_shot_samples_from_group_defs(
            group_defs=group_defs,
            per_type_k=per_type_k,
            max_total=max_total_i,
            strategy=strategy,
        )
        few_shot_samples = pretty_json(few_shot_list)

        if isinstance(self.max_few_shot_word_len, int) and self.max_few_shot_word_len > 0:
            few_shot_samples = truncate_by_word_len(few_shot_samples, self.max_few_shot_word_len, lang="auto")

        # IMPORTANT:
        # Your YAML has relation_group as task_variable, but some loaders validate static+task strictly.
        # We'll provide relation_group in both (harmless).
        return {
            "task_name": task_name,
            "task_goal": task_goal,
            "global_constraints": global_constraints_block,
            "output_fields_block": output_fields_block,
            "group_name": group_name,
            "group_constraints_block": group_constraints_block,
            "relation_schema": relation_schema_text,
            "few_shot_samples": few_shot_samples,
            "relation_group": safe_str(relation_group),
        }

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def call(self, params: str, **kwargs) -> str:
        """
        params JSON:
          - text: str
          - extracted_entities: str | list
          - relation_group: str
          - previous_results (optional): str | list
          - feedbacks (optional): str | list
          - extracted_relations (optional): str | list
        """
        try:
            params_dict = json.loads(params)
        except Exception as e:
            logger.error(f"Params parse failed: {e}")
            return correct_json_format(json.dumps({"error": f"Params parse failed: {str(e)}"}, ensure_ascii=False))

        text = params_dict.get("text", "") or ""
        extracted_entities = params_dict.get("extracted_entities", "") or ""
        relation_group = safe_str(params_dict.get("relation_group", ""))
        memory_context = str(params_dict.get("memory_context") or "").strip()

        previous_results = params_dict.get("previous_results", [])
        feedbacks = params_dict.get("feedbacks", [])
        extracted_relations = params_dict.get("extracted_relations", [])

        if not isinstance(text, str):
            text = str(text)

        if isinstance(extracted_entities, list):
            extracted_entities = "\n".join([str(x) for x in extracted_entities])

        if isinstance(previous_results, list):
            previous_results = json.dumps(previous_results, ensure_ascii=False)
        elif previous_results is None:
            previous_results = ""

        if isinstance(feedbacks, list):
            feedbacks = "\n".join([str(x) for x in feedbacks])
        elif feedbacks is None:
            feedbacks = ""

        if isinstance(extracted_relations, list):
            extracted_relations = "\n".join([str(x) for x in extracted_relations])
        elif extracted_relations is None:
            extracted_relations = ""

        try:
            if not relation_group:
                raise ValueError("Missing required param: relation_group")

            # 1) schemas
            task_block = self._get_task_block()
            group_defs = self._get_group_defs(relation_group)
            group_block = self._find_group_block(task_block, relation_group)

            static_vars = self._build_static_vars(
                task_block=task_block,
                group_block=group_block,
                group_defs=group_defs,
                relation_group=relation_group,
            )

            messages: List[Dict[str, str]] = []

            # 2) system prompt
            system_prompt = self.prompt_loader.render(
                self.system_prompt_id,
                static_values={
                    "rules": general_rules,
                    "json_only": "",
                },
                task_values={},
                strict=False,
            )
            messages.append({"role": "system", "content": system_prompt})

            # 3) optional context: previous round
            if previous_results and feedbacks:
                prev_msg = self.prompt_loader.render(
                    self.previous_round_prompt_id,
                    static_values={},
                    task_values={
                        "previous_results": str(previous_results),
                        "feedbacks": str(feedbacks),
                    },
                    strict=False,
                )
                messages.append({"role": "user", "content": prev_msg})

            # 4) optional context: already extracted relations (dedup)
            if extracted_relations:
                dedup_msg = self.prompt_loader.render(
                    self.already_extracted_prompt_id,
                    static_values={
                        "default_item_type": "relations",
                    },
                    task_values={
                        "item_type": "relations",
                        "extracted_items": str(extracted_relations),
                    },
                    strict=False,
                )
                messages.append({"role": "user", "content": dedup_msg})

            # 5) main prompt
            user_prompt = self.prompt_loader.render(
                self.relation_prompt_id,
                static_values=static_vars,
                task_values={
                    "text": text,
                    "entities": str(extracted_entities),
                    "relation_group": relation_group,
                    "memory_context": memory_context,
                },
                strict=False,
            )
            messages.append({"role": "user", "content": user_prompt})

            # 6) format guarantee
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

            return correct_json_format(json.dumps({"error": "Relation extraction failed"}, ensure_ascii=False))

        except Exception as e:
            logger.error(f"Relation extraction exception: {e}")
            return correct_json_format(json.dumps({"error": f"Relation extraction failed: {str(e)}"}, ensure_ascii=False))

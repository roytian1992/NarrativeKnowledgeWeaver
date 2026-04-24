from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

from core.utils.function_manager import process_with_format_guarantee
from core.utils.general_text import general_rules, general_repair_template
from core.utils.format import correct_json_format
from core.utils.general_utils import (
    join_bullets,
    load_json,
    pretty_json,
    safe_dict,
    safe_list,
    safe_str,
    truncate_by_word_len,
)

logger = logging.getLogger(__name__)

TASK_SCHEMA_PATH = "task_specs/task_settings/schema_relation_grounding_task.json"
RELATION_SCHEMA_PATH = "task_specs/schema/default_relation_schema.json"


class SchemaRelationGrounder:
    """
    Ground open-form relation proposals to schema relation types.
    """

    def __init__(
        self,
        prompt_loader: Any,
        llm: Any,
        task_schema_path: str = TASK_SCHEMA_PATH,
        relation_schema_path: str = RELATION_SCHEMA_PATH,
        max_candidates_word_len: int | None = None,
        max_retries: int = 3,
    ) -> None:
        if llm is None:
            raise ValueError("llm must be provided")
        if prompt_loader is None:
            raise ValueError("prompt_loader must be provided")

        self.llm = llm
        self.prompt_loader = prompt_loader
        self.task_schema_path = task_schema_path
        self.relation_schema_path = relation_schema_path
        self.max_candidates_word_len = max_candidates_word_len

        self.system_prompt_id = "knowledge_extraction/system_prompt"
        self.ground_prompt_id = "knowledge_extraction/ground_open_relations"

        self.required_fields: List[str] = []
        self.field_validators: Dict[str, Any] = {}
        self.repair_template = general_repair_template
        self.max_retries = max(0, int(max_retries))

        task_schema = load_json(self.task_schema_path)
        if not isinstance(task_schema, list):
            raise ValueError(f"Invalid task schema: expected list at {self.task_schema_path}")
        self.task_schema: List[Dict[str, Any]] = [x for x in task_schema if isinstance(x, dict)]
        self.task_schema_map: Dict[str, Dict[str, Any]] = {}
        for tb in self.task_schema:
            task = safe_str(tb.get("task"))
            if task:
                self.task_schema_map[task] = tb

        relation_schema = load_json(self.relation_schema_path)
        if not isinstance(relation_schema, dict):
            raise ValueError(f"Invalid relation schema: expected dict at {self.relation_schema_path}")
        self.relation_schema: Dict[str, List[Dict[str, Any]]] = {
            safe_str(k): [x for x in safe_list(v) if isinstance(x, dict)]
            for k, v in relation_schema.items()
            if safe_str(k)
        }

    def _get_task_block(self) -> Dict[str, Any]:
        tb = self.task_schema_map.get("schema_relation_grounding")
        if not isinstance(tb, dict):
            raise RuntimeError(
                f"Task schema not found for task='schema_relation_grounding' in {self.task_schema_path}"
            )
        return tb

    def _schema_few_shot_examples(self, proposals: List[Dict[str, Any]], max_total: int = 12) -> List[Dict[str, Any]]:
        wanted: List[str] = []
        for proposal in proposals or []:
            if not isinstance(proposal, dict):
                continue
            for item in safe_list(proposal.get("candidate_relations")):
                if not isinstance(item, dict):
                    continue
                rtype = safe_str(item.get("relation_type")).strip()
                if rtype and rtype not in wanted:
                    wanted.append(rtype)
        examples: List[Dict[str, Any]] = []
        seen = set()
        type_to_def: Dict[str, Dict[str, Any]] = {}
        for group_items in self.relation_schema.values():
            for item in group_items:
                rtype = safe_str(item.get("type")).strip()
                if rtype:
                    type_to_def[rtype] = item
        for rtype in wanted:
            item = type_to_def.get(rtype)
            if item is None:
                continue
            for sample in safe_list(item.get("samples")):
                if not isinstance(sample, dict):
                    continue
                key = (
                    safe_str(sample.get("subject")).strip(),
                    safe_str(sample.get("object")).strip(),
                    safe_str(sample.get("relation_type") or rtype).strip(),
                )
                if not all(key) or key in seen:
                    continue
                seen.add(key)
                examples.append(
                    {
                        "proposal_id": f"demo_{rtype}_{len(examples)+1}",
                        "decision": "ground",
                        "relation_type": key[2],
                        "swap_subject_object": False,
                        "relation_name": safe_str(sample.get("relation_name")).strip() or key[2],
                        "description": safe_str(sample.get("description")).strip(),
                    }
                )
                if len(examples) >= max_total:
                    return examples
        return examples

    def _build_static_vars(self, task_block: Dict[str, Any], *, proposals: List[Dict[str, Any]]) -> Dict[str, Any]:
        task_name = safe_str(task_block.get("task_name"))
        task_goal = safe_str(task_block.get("task_goal"))
        global_constraints_block = join_bullets(task_block.get("global_constraints"))
        grounding_rules_block = join_bullets(task_block.get("grounding_rules"))

        output_fields = [safe_str(x) for x in safe_list(task_block.get("output_fields")) if safe_str(x)]
        if not output_fields:
            output_fields = [
                "proposal_id",
                "decision",
                "relation_type",
                "swap_subject_object",
                "relation_name",
                "description",
            ]
        output_fields_block = "\n".join([f'- "{k}"' for k in output_fields]).strip()

        schema_examples = self._schema_few_shot_examples(proposals)
        if schema_examples:
            few_shot_samples = pretty_json(schema_examples)
        else:
            few_shot_cfg = safe_dict(task_block.get("few_shot"))
            examples = [x for x in safe_list(few_shot_cfg.get("examples")) if isinstance(x, dict)]
            few_shot_samples = pretty_json(examples)

        return {
            "task_name": task_name,
            "task_goal": task_goal,
            "global_constraints": global_constraints_block,
            "grounding_rules_block": grounding_rules_block,
            "output_fields_block": output_fields_block,
            "few_shot_samples": few_shot_samples,
        }

    def _normalize_proposals_text(self, proposals: Any) -> str:
        text = pretty_json(proposals if isinstance(proposals, list) else [])
        if isinstance(self.max_candidates_word_len, int) and self.max_candidates_word_len > 0 and text:
            text = truncate_by_word_len(text, self.max_candidates_word_len, lang="auto")
        return text

    def _postprocess_outputs(self, outputs: Any, proposals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not isinstance(outputs, list):
            return []

        proposal_map: Dict[str, Dict[str, Any]] = {}
        for proposal in proposals or []:
            if not isinstance(proposal, dict):
                continue
            pid = safe_str(proposal.get("proposal_id")).strip()
            if pid:
                proposal_map[pid] = proposal

        cleaned: List[Dict[str, Any]] = []
        for item in outputs:
            if not isinstance(item, dict):
                continue
            proposal_id = safe_str(item.get("proposal_id")).strip()
            decision = safe_str(item.get("decision")).strip().lower()
            proposal = proposal_map.get(proposal_id)
            if not proposal or decision not in {"ground", "drop"}:
                continue

            if decision == "drop":
                cleaned.append({"proposal_id": proposal_id, "decision": "drop"})
                continue

            relation_type = safe_str(item.get("relation_type")).strip()
            allowed = {
                safe_str(x.get("relation_type")).strip()
                for x in safe_list(proposal.get("candidate_relations"))
                if isinstance(x, dict) and safe_str(x.get("relation_type")).strip()
            }
            if relation_type not in allowed:
                cleaned.append({"proposal_id": proposal_id, "decision": "drop"})
                continue

            relation_name = safe_str(item.get("relation_name")).strip() or safe_str(proposal.get("relation_phrase")).strip()
            description = safe_str(item.get("description")).strip() or safe_str(proposal.get("description")).strip() or safe_str(proposal.get("evidence")).strip()
            cleaned.append(
                {
                    "proposal_id": proposal_id,
                    "decision": "ground",
                    "relation_type": relation_type,
                    "swap_subject_object": bool(item.get("swap_subject_object", False)),
                    "relation_name": relation_name,
                    "description": description,
                }
            )

        return cleaned

    def call(self, params: str, **kwargs) -> str:
        try:
            params_dict = json.loads(params)
        except Exception as e:
            logger.error(f"Params parse failed: {e}")
            return correct_json_format(json.dumps({"error": f"Params parse failed: {str(e)}"}, ensure_ascii=False))

        text = safe_str(params_dict.get("text"))
        proposals = params_dict.get("proposals", [])
        memory_context = safe_str(params_dict.get("memory_context"))
        if not isinstance(proposals, list):
            proposals = []

        try:
            task_block = self._get_task_block()
            static_vars = self._build_static_vars(task_block, proposals=proposals)
            proposals_text = self._normalize_proposals_text(proposals)

            messages: List[Dict[str, str]] = []
            system_prompt = self.prompt_loader.render(
                self.system_prompt_id,
                static_values={"rules": general_rules, "json_only": ""},
                task_values={},
                strict=False,
            )
            messages.append({"role": "system", "content": system_prompt})

            user_prompt = self.prompt_loader.render(
                self.ground_prompt_id,
                static_values=static_vars,
                task_values={
                    "text": text,
                    "open_relation_proposals": proposals_text,
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
                return correct_json_format(json.dumps({"error": "Schema relation grounding failed"}, ensure_ascii=False))

            try:
                parsed = json.loads(corrected_json)
            except Exception:
                parsed = []

            normalized = self._postprocess_outputs(parsed, proposals=proposals)
            return correct_json_format(json.dumps(normalized, ensure_ascii=False))

        except Exception as e:
            logger.error(f"Schema relation grounding exception: {e}")
            return correct_json_format(json.dumps({"error": f"Schema relation grounding failed: {str(e)}"}, ensure_ascii=False))

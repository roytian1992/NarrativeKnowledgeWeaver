from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Set
from core.utils.general_utils import safe_str
from core.utils.function_manager import process_with_format_guarantee
from core.utils.general_text import general_repair_template
from core.utils.format import correct_json_format

logger = logging.getLogger(__name__)


class RelationDeduper:
    """
    YAML-driven relation type deduper.

    Prompt YAML:
      - problem_solving/dedup_relations

    Input params JSON:
      {
        "relations": "string (JSON array of relations)"
      }

    Output JSON:
      - keep:
        { "decision": "keep", "output": {} }
      - drop:
        {
          "decision": "drop",
          "output": {
            "drop_relation_types": ["..."],
            "rationale": "..."
          }
        }
    """

    def __init__(
        self,
        prompt_loader,
        llm,
        prompt_id: str = "problem_solving/deduplicate_relations",
    ):
        if llm is None:
            raise ValueError("llm must be provided")
        if prompt_loader is None:
            raise ValueError("prompt_loader must be provided")

        self.llm = llm
        self.prompt_loader = prompt_loader
        self.prompt_id = prompt_id

        self.required_fields = ["decision", "output"]
        self.field_validators = {
            "decision": lambda x: isinstance(x, str) and x in ("keep", "drop"),
            "output": lambda x: isinstance(x, dict),
        }
        self.repair_template = general_repair_template

    @staticmethod
    def _extract_allowed_types(relations: Any) -> List[str]:
        """
        Extract allowed relation types from parsed relations.
        Accepts list[dict] where relation type may be under:
          - relation_type
          - type
        Returns stable-dedup list.
        """
        cands: List[str] = []
        if isinstance(relations, list):
            for r in relations:
                if not isinstance(r, dict):
                    continue
                t = r.get("relation_type", None)
                if t is None:
                    t = r.get("type", None)
                if isinstance(t, str):
                    tt = t.strip()
                    if tt:
                        cands.append(tt)
        seen: Set[str] = set()
        out: List[str] = []
        for t in cands:
            if t not in seen:
                seen.add(t)
                out.append(t)
        return out

    def _post_validate(self, obj: Dict[str, Any], allowed_types: List[str]) -> Dict[str, Any]:
        """
        Enforce:
          - decision in {keep, drop} else keep
          - if keep -> output={}
          - if drop -> output.drop_relation_types non-empty subset of allowed_types
          - if invalid drop list -> fallback keep
        """
        decision = obj.get("decision", "keep")
        output = obj.get("output", {})

        if not isinstance(decision, str):
            decision = "keep"
        decision = decision.strip().lower()
        if decision not in ("keep", "drop"):
            decision = "keep"

        if not isinstance(output, dict):
            output = {}

        allowed_set = set([t for t in allowed_types if isinstance(t, str) and t.strip()])

        if decision == "keep":
            return {"decision": "keep", "output": {}}

        # decision == "drop"
        drop_types = output.get("drop_relation_types", None)
        if not isinstance(drop_types, list) or len(drop_types) == 0:
            return {"decision": "keep", "output": {}}

        filtered: List[str] = []
        for t in drop_types:
            if not isinstance(t, str):
                continue
            tt = t.strip()
            if tt and tt in allowed_set and tt not in filtered:
                filtered.append(tt)

        if not filtered:
            return {"decision": "keep", "output": {}}

        rationale = output.get("rationale", "")
        if not isinstance(rationale, str):
            rationale = safe_str(rationale)

        return {
            "decision": "drop",
            "output": {
                "drop_relation_types": filtered,
                "rationale": rationale.strip(),
            },
        }

    def call(self, params: str, **kwargs) -> str:
        # 1) parse params
        try:
            params_dict = json.loads(params) if isinstance(params, str) else (params or {})
            relations_str = safe_str(params_dict.get("relations", "")).strip()
        except Exception as e:
            logger.error(f"参数解析失败: {e}")
            return correct_json_format(json.dumps({"error": f"参数解析失败: {str(e)}"}, ensure_ascii=False))

        if not relations_str:
            return correct_json_format(
                json.dumps({"error": "missing required field: relations"}, ensure_ascii=False)
            )

        # 2) parse relations for allowed type filtering (best-effort)
        parsed_relations: Any = []
        try:
            parsed_relations = json.loads(relations_str)
        except Exception:
            parsed_relations = []

        allowed_types = self._extract_allowed_types(parsed_relations)
        if not allowed_types:
            # If we cannot infer types, we still run LLM, but post-validation will likely fall back to keep.
            pass

        # 3) render YAML prompt
        try:
            user_prompt = self.prompt_loader.render(
                self.prompt_id,
                static_values={},
                task_values={"relations": relations_str},
                strict=True,
            )
        except Exception as e:
            logger.error(f"dedup_relations prompt render failed: {e}")
            return correct_json_format(json.dumps({"error": f"prompt render failed: {str(e)}"}, ensure_ascii=False))

        messages = [{"role": "user", "content": user_prompt}]

        # 4) LLM call with format guarantee
        corrected_json, status = process_with_format_guarantee(
            llm_client=self.llm,
            messages=messages,
            required_fields=self.required_fields,
            field_validators=self.field_validators,
            max_retries=3,
            repair_template=self.repair_template,
        )

        if status != "success":
            return correct_json_format(json.dumps({"error": "relation dedup failed"}, ensure_ascii=False))

        # 5) post-validate decision-specific constraints
        try:
            obj = json.loads(corrected_json)
            if not isinstance(obj, dict):
                obj = {}

            safe = self._post_validate(obj, allowed_types)
            return correct_json_format(json.dumps(safe, ensure_ascii=False))

        except Exception as e:
            logger.error(f"dedup output validation failed: {e}")
            return correct_json_format(json.dumps({"decision": "keep", "output": {}}, ensure_ascii=False))

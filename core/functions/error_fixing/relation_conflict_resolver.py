from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List
from core.utils.general_utils import safe_str
from core.utils.function_manager import process_with_format_guarantee
from core.utils.general_text import general_repair_template
from core.utils.format import correct_json_format

logger = logging.getLogger(__name__)


class RelationConflictResolver:
    """
    YAML-driven relation conflict resolver.

    Prompt YAML:
      - problem_solving/resolve_relation_conflicts

    Input params JSON:
      {
        "context": "string",
        "subject_entity": "string",
        "object_entity": "string"
      }

    Output JSON:
      {
        "keep_relation_types": ["..."],
        "reason": "..."
      }
    """

    def __init__(
        self,
        prompt_loader,
        llm,
        prompt_id: str = "problem_solving/resolve_relation_conflicts",
    ):
        if llm is None:
            raise ValueError("llm must be provided")
        if prompt_loader is None:
            raise ValueError("prompt_loader must be provided")

        self.llm = llm
        self.prompt_loader = prompt_loader
        self.prompt_id = prompt_id

        self.required_fields = ["keep_relation_types", "reason"]
        self.field_validators = {
            "keep_relation_types": lambda x: isinstance(x, list) and all(isinstance(i, str) for i in x),
            "reason": lambda x: isinstance(x, str),
        }
        self.repair_template = general_repair_template

    @staticmethod
    def _norm(s: Any) -> str:
        return safe_str(s).strip()

    @staticmethod
    def _extract_relation_types(context: str) -> List[str]:
        """
        Best-effort extraction of relation types from the report context.
        Supports lines containing patterns like:
          - "- affinity_with | base=scene_93 ..."
          - "relation_type: affinity_with"
          - "affinity_with: <desc> (count=3)"
        """
        text = context or ""
        cands: List[str] = []

        for m in re.finditer(r"^\s*-\s*([A-Za-z0-9_]+)\s*\|\s*base=", text, flags=re.MULTILINE):
            cands.append(m.group(1).strip())

        for m in re.finditer(r"relation_type\s*[:=]\s*([A-Za-z0-9_]+)", text, flags=re.IGNORECASE):
            cands.append(m.group(1).strip())

        for m in re.finditer(r"^\s*([A-Za-z0-9_]+)\s*:\s+.+$", text, flags=re.MULTILINE):
            t = m.group(1).strip()
            if "_" in t:
                cands.append(t)

        seen = set()
        out: List[str] = []
        for t in cands:
            if t and t not in seen:
                seen.add(t)
                out.append(t)
        return out

    @staticmethod
    def _post_validate(obj: Dict[str, Any], allowed_types: List[str]) -> Dict[str, Any]:
        allowed = [t for t in (allowed_types or []) if isinstance(t, str) and t.strip()]
        allowed_set = set(allowed)

        keep = obj.get("keep_relation_types", [])
        reason = obj.get("reason", "")

        if not isinstance(reason, str):
            reason = str(reason)

        keep_clean: List[str] = []
        if isinstance(keep, list):
            for t in keep:
                if not isinstance(t, str):
                    continue
                tt = t.strip()
                if tt in allowed_set and tt not in keep_clean:
                    keep_clean.append(tt)

        if not keep_clean:
            keep_clean = allowed[:]  # conservative fallback: keep all

        order = {t: i for i, t in enumerate(allowed)}
        keep_clean = sorted(keep_clean, key=lambda x: order.get(x, 10**9))

        return {"keep_relation_types": keep_clean, "reason": reason.strip()}

    def call(self, params: str, **kwargs) -> str:
        # 1) parse params
        try:
            params_dict = json.loads(params) if isinstance(params, str) else (params or {})
            context = safe_str(params_dict.get("text", "")).strip()
            subject_entity = safe_str(params_dict.get("subject_entity", "")).strip()
            object_entity = safe_str(params_dict.get("object_entity", "")).strip()
        except Exception as e:
            logger.error(f"参数解析失败: {e}")
            return correct_json_format(json.dumps({"error": f"参数解析失败: {str(e)}"}, ensure_ascii=False))

        if not context or not subject_entity or not object_entity:
            return correct_json_format(
                json.dumps(
                    {
                        "error": "missing required fields: context, subject_entity, object_entity are required",
                        "keep_relation_types": [],
                        "reason": "",
                    },
                    ensure_ascii=False,
                )
            )

        allowed_types = self._extract_relation_types(context)
        if not allowed_types:
            return correct_json_format(
                json.dumps(
                    {
                        "error": "cannot extract relation types from context",
                        "keep_relation_types": [],
                        "reason": "",
                    },
                    ensure_ascii=False,
                )
            )

        # 2) render YAML prompt
        try:
            user_prompt = self.prompt_loader.render(
                self.prompt_id,
                static_values={},
                task_values={
                    "context": context,
                    "subject_entity": subject_entity,
                    "object_entity": object_entity,
                },
                strict=True,
            )
        except Exception as e:
            logger.error(f"resolve_relation_conflicts prompt render failed: {e}")
            return correct_json_format(json.dumps({"error": f"prompt render failed: {str(e)}"}, ensure_ascii=False))

        messages = [{"role": "user", "content": user_prompt}]

        # 3) LLM call with format guarantee
        corrected_json, status = process_with_format_guarantee(
            llm_client=self.llm,
            messages=messages,
            required_fields=self.required_fields,
            field_validators=self.field_validators,
            max_retries=3,
            repair_template=self.repair_template,
        )

        if status != "success":
            return correct_json_format(json.dumps({"error": "relation conflict decision failed"}, ensure_ascii=False))

        # 4) post-validate and normalize
        try:
            obj = json.loads(corrected_json)
            if not isinstance(obj, dict):
                obj = {}

            safe = self._post_validate(obj, allowed_types)
            return correct_json_format(json.dumps(safe, ensure_ascii=False))
        except Exception as e:
            logger.error(f"relation conflict output validation failed: {e}")
            safe = {"keep_relation_types": allowed_types, "reason": ""}
            return correct_json_format(json.dumps(safe, ensure_ascii=False))

# core/functions/regular_functions/prune_causal_edges.py
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from core.utils.function_manager import process_with_format_guarantee
from core.utils.general_text import general_rules, general_repair_template
from core.utils.format import correct_json_format
from core.utils.general_utils import safe_dict, safe_list

logger = logging.getLogger(__name__)


def _stringify(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    try:
        return json.dumps(v, ensure_ascii=False, indent=2)
    except Exception:
        return str(v)


def _is_bool_like(x: Any) -> bool:
    return isinstance(x, bool)


def _is_nonempty_str(x: Any) -> bool:
    return isinstance(x, str) and bool(x.strip())


class CausalLinkPruner:
    """
    YAML-driven evaluator for triadic causal redundancy.

    Prompt file:
      aggregation/prune_causal_links.yaml

    Expected output JSON:
      {
        "remove_edge": true,
        "reason": "..."
      }
    """

    def __init__(self, prompt_loader, llm):
        if llm is None:
            raise ValueError("llm must be provided")
        if prompt_loader is None:
            raise ValueError("prompt_loader must be provided")

        self.llm = llm
        self.prompt_loader = prompt_loader

        self.prompt_id = "aggregation/prune_causal_links"

        # format guarantee
        self.required_fields: List[str] = ["remove_edge", "reason"]
        self.field_validators: Dict[str, Any] = {
            "remove_edge": _is_bool_like,
            "reason": lambda x: isinstance(x, str),
        }
        self.repair_template = general_repair_template

    def call(self, params: str, **kwargs) -> str:
        """
        params JSON:
          - entity_a: str | dict
          - entity_b: str | dict
          - entity_c: str | dict
          - relation_ab: str | dict
          - relation_bc: str | dict
          - relation_ac: str | dict

        Returns:
          JSON-only string:
            {"remove_edge": bool, "reason": str}
        """
        try:
            params_dict = json.loads(params)
        except Exception as e:
            logger.error(f"Parameter parse failed: {e}")
            return correct_json_format(
                json.dumps({"error": f"Parameter parse failed: {str(e)}"}, ensure_ascii=False)
            )

        p = safe_dict(params_dict)

        entity_a = _stringify(p.get("entity_a"))
        entity_b = _stringify(p.get("entity_b"))
        entity_c = _stringify(p.get("entity_c"))

        relation_ab = _stringify(p.get("relation_ab"))
        relation_bc = _stringify(p.get("relation_bc"))
        relation_ac = _stringify(p.get("relation_ac"))

        missing: List[str] = []
        if not entity_a.strip():
            missing.append("entity_a")
        if not entity_b.strip():
            missing.append("entity_b")
        if not entity_c.strip():
            missing.append("entity_c")
        if not relation_ab.strip():
            missing.append("relation_ab")
        if not relation_bc.strip():
            missing.append("relation_bc")
        if not relation_ac.strip():
            missing.append("relation_ac")

        if missing:
            return correct_json_format(
                json.dumps(
                    {"error": "Missing required fields", "missing": missing},
                    ensure_ascii=False,
                )
            )

        try:

            messages: List[Dict[str, str]] = []

            user_prompt = self.prompt_loader.render(
                self.prompt_id,
                static_values={},  # no static variables for this task
                task_values={
                    "entity_a": entity_a,
                    "entity_b": entity_b,
                    "entity_c": entity_c,
                    "relation_ab": relation_ab,
                    "relation_bc": relation_bc,
                    "relation_ac": relation_ac,
                },
                strict=True,
            )
            messages.append({"role": "user", "content": user_prompt})

            # 3) format guarantee
            corrected_json, status = process_with_format_guarantee(
                llm_client=self.llm,
                messages=messages,
                required_fields=self.required_fields,
                field_validators=self.field_validators,
                max_retries=3,
                repair_template=self.repair_template,
            )

            if status != "success":
                return correct_json_format(json.dumps({"error": "Pruning evaluation failed"}, ensure_ascii=False))

            # normalize to the exact schema and strip extras if the model returned more keys
            try:
                obj = json.loads(corrected_json)
            except Exception:
                # let correct_json_format try to salvage
                return correct_json_format(corrected_json)

            if not isinstance(obj, dict):
                return correct_json_format(
                    json.dumps({"error": "Invalid output type: expected JSON object"}, ensure_ascii=False)
                )

            out = {
                "remove_edge": bool(obj.get("remove_edge")),
                "reason": str(obj.get("reason") or "").strip(),
            }

            # final validation
            if not isinstance(out["reason"], str):
                out["reason"] = ""

            return correct_json_format(json.dumps(out, ensure_ascii=False))

        except Exception as e:
            logger.error(f"Pruning evaluation exception: {e}")
            return correct_json_format(json.dumps({"error": f"Pruning evaluation failed: {str(e)}"}, ensure_ascii=False))

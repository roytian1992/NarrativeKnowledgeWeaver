from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List
from core.utils.general_utils import  safe_str
from core.utils.function_manager import process_with_format_guarantee
from core.utils.general_text import general_repair_template
from core.utils.format import correct_json_format

logger = logging.getLogger(__name__)

def _extract_input_names(entity_descriptions: str) -> List[str]:
    """
    Best-effort parse names out of the formatted candidate list.
    Expects lines like: "Entity 1 name: XXX"
    """
    names: List[str] = []
    for line in (entity_descriptions or "").splitlines():
        line = line.strip()
        m = re.match(r"Entity\s+\d+\s+name:\s*(.+)$", line, flags=re.IGNORECASE)
        if m:
            nm = m.group(1).strip()
            if nm:
                names.append(nm)

    # stable dedup
    seen = set()
    out: List[str] = []
    for nm in names:
        if nm not in seen:
            seen.add(nm)
            out.append(nm)
    return out


class EntityDisambiguationJudger:
    """
    YAML-driven entity merge judger.

    Prompt YAML:
      - problem_solving/merge_entities

    Input params JSON:
      {
        "entity_descriptions": "string"
      }

    Output JSON:
      {
        "merges": [{"canonical_name": "...", "aliases": [...], "reason": "..."}],
        "unmerged": [{"name": "...", "reason": "..."}]
      }

    Notes:
      - Only one input field: entity_descriptions (str).
      - No special-case heuristics for events.
      - Post-validation enforces:
          * canonical_name ∈ input_names
          * aliases ⊂ input_names and exclude canonical_name
          * each input name appears in at most one merge group
          * if merges empty => unmerged covers all input names
          * unmerged items are from input_names and exclude merged aliases
    """

    def __init__(
        self,
        prompt_loader,
        llm,
        prompt_id: str = "problem_solving/merge_entities",
    ):
        if llm is None:
            raise ValueError("llm must be provided")
        if prompt_loader is None:
            raise ValueError("prompt_loader must be provided")

        self.llm = llm
        self.prompt_loader = prompt_loader
        self.prompt_id = prompt_id

        self.required_fields = ["merges", "unmerged"]
        self.field_validators = {
            "merges": lambda x: isinstance(x, list),
            "unmerged": lambda x: isinstance(x, list),
        }
        self.repair_template = general_repair_template

    def _post_validate(self, obj: Dict[str, Any], input_names: List[str]) -> Dict[str, Any]:
        input_set = set(input_names or [])

        merges_raw = obj.get("merges", [])
        unmerged_raw = obj.get("unmerged", [])

        if not isinstance(merges_raw, list):
            merges_raw = []
        if not isinstance(unmerged_raw, list):
            unmerged_raw = []

        used: set[str] = set()
        merges_out: List[Dict[str, Any]] = []

        for m in merges_raw:
            if not isinstance(m, dict):
                continue

            canonical = safe_str(m.get("canonical_name"))
            aliases = m.get("aliases", [])
            reason = m.get("reason", "")

            if not canonical or canonical not in input_set:
                continue

            if not isinstance(aliases, list):
                aliases = []

            aliases_clean: List[str] = []
            for a in aliases:
                a = safe_str(a)
                if not a or a == canonical:
                    continue
                if a not in input_set:
                    continue
                aliases_clean.append(a)

            # must have at least 1 alias
            if not aliases_clean:
                continue

            group_names = [canonical] + aliases_clean
            if any(n in used for n in group_names):
                continue
            for n in group_names:
                used.add(n)

            if not isinstance(reason, str):
                reason = safe_str(reason)

            merges_out.append(
                {
                    "canonical_name": canonical,
                    "aliases": aliases_clean,
                    "reason": reason.strip(),
                }
            )

        merged_aliases: set[str] = set()
        for m in merges_out:
            merged_aliases.update(m["aliases"])

        # aliases are removed; canonicals stay
        remaining = [n for n in input_names if n not in merged_aliases]

        # keep only valid unmerged entries
        unmerged_valid: List[Dict[str, str]] = []
        for u in unmerged_raw:
            if not isinstance(u, dict):
                continue
            nm = safe_str(u.get("name"))
            rs = u.get("reason", "")
            if not nm or nm not in input_set:
                continue
            if nm in merged_aliases:
                continue
            if not isinstance(rs, str):
                rs = safe_str(rs)
            unmerged_valid.append({"name": nm, "reason": rs.strip()})

        # ensure coverage for remaining
        present = {x["name"] for x in unmerged_valid}
        for n in remaining:
            if n not in present:
                unmerged_valid.append({"name": n, "reason": ""})

        return {"merges": merges_out, "unmerged": unmerged_valid}

    def call(self, params: str, **kwargs) -> str:
        # 1) parse params
        try:
            params_dict = json.loads(params) if isinstance(params, str) else (params or {})
            entity_descriptions = safe_str(params_dict.get("entity_descriptions", "")).strip()
        except Exception as e:
            logger.error(f"参数解析失败: {e}")
            return correct_json_format(json.dumps({"error": f"参数解析失败: {str(e)}"}, ensure_ascii=False))

        if not entity_descriptions:
            return correct_json_format(
                json.dumps(
                    {"error": "missing required field: entity_descriptions", "merges": [], "unmerged": []},
                    ensure_ascii=False,
                )
            )

        input_names = _extract_input_names(entity_descriptions)

        # 2) render YAML prompt
        try:
            user_prompt = self.prompt_loader.render(
                self.prompt_id,
                static_values={},
                task_values={"entity_descriptions": entity_descriptions},
                strict=True,
            )
        except Exception as e:
            logger.error(f"merge_entities prompt render failed: {e}")
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
            return correct_json_format(json.dumps({"error": "entity disambiguation failed"}, ensure_ascii=False))

        # 4) post-validate and normalize
        try:
            obj = json.loads(corrected_json)
            if not isinstance(obj, dict):
                obj = {}
            safe = self._post_validate(obj, input_names)
            return correct_json_format(json.dumps(safe, ensure_ascii=False))
        except Exception as e:
            logger.error(f"entity disambiguation output validation failed: {e}")
            safe = {"merges": [], "unmerged": [{"name": n, "reason": ""} for n in input_names]}
            return correct_json_format(json.dumps(safe, ensure_ascii=False))

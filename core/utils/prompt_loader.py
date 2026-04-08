from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union
import yaml
from langchain.prompts import PromptTemplate


# -----------------------------------------------------------------------------
# Shared helpers
# -----------------------------------------------------------------------------

_VAR_PATTERN = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")


def _stringify(v: Any) -> str:
    """Convert values to string for prompt rendering."""
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    if isinstance(v, (dict, list)):
        return json.dumps(v, ensure_ascii=False, indent=2)
    return str(v)


def _declared_names(vars_list: Any) -> Set[str]:
    out: Set[str] = set()
    if not isinstance(vars_list, list):
        return out
    for v in vars_list:
        if not isinstance(v, dict):
            continue
        name = v.get("name")
        if isinstance(name, str) and name.strip():
            out.add(name.strip())
    return out


def _required_names(vars_list: Any) -> Set[str]:
    out: Set[str] = set()
    if not isinstance(vars_list, list):
        return out
    for v in vars_list:
        if not isinstance(v, dict):
            continue
        name = v.get("name")
        req = v.get("required", False)
        if isinstance(name, str) and name.strip() and req is True:
            out.add(name.strip())
    return out


def _safe_replace(template: str, declared: Set[str], values: Dict[str, Any]) -> str:
    """
    Safely replace {var} by regex, only for declared variable names.
    Leaves all other braces untouched (safe for JSON dumps with {}).
    """
    merged: Dict[str, str] = {k: _stringify(v) for k, v in values.items() if k in declared}

    def repl(m: re.Match) -> str:
        name = m.group(1)
        if name in declared:
            return merged.get(name, "")
        return m.group(0)

    return _VAR_PATTERN.sub(repl, template)


# -----------------------------------------------------------------------------
# JSONPromptLoader (your original, renamed)
# -----------------------------------------------------------------------------

class JSONPromptLoader:
    """Prompt template loader (JSON format + LangChain PromptTemplate + variable declarations)."""

    def __init__(self, prompt_dir: str, global_variables: Optional[Dict[str, str]] = None):
        self.prompt_dir = Path(prompt_dir)
        if not self.prompt_dir.exists():
            raise FileNotFoundError(f"Prompt dir not found: {self.prompt_dir}")
        self.global_variables = global_variables or {}

    def load_prompt(self, prompt_id: str) -> Dict[str, Any]:
        prompt_path = self.prompt_dir / f"{prompt_id}.json"
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
        with open(prompt_path, "r", encoding="utf-8") as f:
            prompt_data = json.load(f)
        if not isinstance(prompt_data, dict):
            raise ValueError(f"Invalid prompt JSON: {prompt_path}")
        return prompt_data

    def render_prompt(self, prompt_id: str, variables: Dict[str, Any]) -> str:
        prompt_data = self.load_prompt(prompt_id)
        template_str = prompt_data.get("template", "")
        if not isinstance(template_str, str) or not template_str.strip():
            raise ValueError(f"Missing template in prompt: {prompt_id}")

        # Merge globals + locals
        full_vars = {**self.global_variables, **(variables or {})}

        # Declared variables in JSON prompt
        declared_vars = prompt_data.get("variables", []) or []
        required_vars = [var.get("name") for var in declared_vars if isinstance(var, dict) and var.get("name")]
        missing = set(required_vars) - set(full_vars.keys())
        if missing:
            raise ValueError(f"Missing variables: {sorted(list(missing))}, prompt_id={prompt_id}")

        # Only keep required ones (to avoid accidental injection)
        filtered_vars = {k: full_vars[k] for k in required_vars}

        # Escape braces in variable values if needed
        for var_name, var_value in list(filtered_vars.items()):
            if isinstance(var_value, str) and self._should_escape_variable(var_name):
                filtered_vars[var_name] = self._escape_braces(var_value)

        # Escape braces inside ```json blocks``` in template
        template_str_safe = self._escape_braces_in_json_block(template_str)

        # Render via LangChain PromptTemplate
        prompt_template = PromptTemplate.from_template(template_str_safe)
        return prompt_template.format(**filtered_vars)

    @staticmethod
    def _escape_braces_in_json_block(text: str) -> str:
        def replacer(match: re.Match) -> str:
            content = match.group(1)
            content_escaped = content.replace("{", "{{").replace("}", "}}")
            return f"```json\n{content_escaped}\n```"

        return re.sub(r"```json\n(.*?)\n```", replacer, text, flags=re.DOTALL)

    @staticmethod
    def _should_escape_variable(var_name: str) -> bool:
        keywords = ["description", "_text", "_list"]
        return any(k in var_name for k in keywords)

    @staticmethod
    def _escape_braces(text: str) -> str:
        return text.replace("{", "{{").replace("}", "}}")


# -----------------------------------------------------------------------------
# YAMLPromptLoader (new)
# -----------------------------------------------------------------------------

@dataclass
class YamlPromptSpec:
    id: str
    template: str
    task_variables: List[Dict[str, Any]]
    static_variables: List[Dict[str, Any]]
    raw: Dict[str, Any]


class YAMLPromptLoader:
    """
    Minimal YAML prompt loader:
    - Prompts are stored as .yaml/.yml files under prompt_dir (supports nested dirs).
    - Loading:
        - load_by_id("knowledge_extraction/system_prompt") works (auto adds .yaml/.yml).
        - load_by_id("system_prompt") works (searches by filename, must be unique).
        - load_by_path("knowledge_extraction/system_prompt.yaml") works (explicit extension).
    - Rendering:
        - Variables split into task_variables (per call) and static_variables (prebuilt / schema-driven).
        - Safe rendering: regex replace {var} only for declared names.
          Does NOT use str.format (safe for JSON dumps containing {}).
    """

    def __init__(self, prompt_dir: str, global_static: Optional[Dict[str, Any]] = None):
        self.prompt_dir = Path(prompt_dir)
        if not self.prompt_dir.exists():
            raise FileNotFoundError(f"Prompt dir not found: {self.prompt_dir}")
        self.global_static = global_static or {}

    # ----------------------------
    # Public: Loading
    # ----------------------------

    def load_by_path(self, rel_path: str) -> YamlPromptSpec:
        """
        Load by explicit relative path under prompt_dir.
        IMPORTANT: This expects the path to include .yaml/.yml.
        Example: "knowledge_extraction/system_prompt.yaml"
        """
        rp = (rel_path or "").strip()
        if not rp:
            raise ValueError("rel_path must be a non-empty string")

        p = self.prompt_dir / rp
        if not (p.exists() and p.is_file()):
            raise FileNotFoundError(f"Prompt yaml not found: {p}")
        return self._load_file(p)

    def load_by_id(self, prompt_id: str) -> YamlPromptSpec:
        """
        Load by id with extension auto-append:
        - If prompt_id includes slashes, treat it as a nested relative id, auto-append suffix.
          e.g., "knowledge_extraction/system_prompt" -> try .yaml then .yml
        - If prompt_id is a bare name, search by filename under prompt_dir, must be unique.
          e.g., "system_prompt" -> rglob system_prompt.yaml/yml
        Also supports absolute path input.
        """
        pid = (prompt_id or "").strip()
        if not pid:
            raise ValueError("prompt_id must be a non-empty string")

        # 1) Absolute path: load directly if it exists.
        p_abs = Path(pid)
        if p_abs.is_absolute() and p_abs.exists() and p_abs.is_file():
            return self._load_file(p_abs)

        # 2) If prompt_id accidentally includes prompt_dir prefix, strip it.
        pid = self._strip_prompt_dir_prefix(pid)

        # 3) If looks like nested relative (has slash/backslash), resolve deterministically with suffix.
        if "/" in pid or "\\" in pid:
            rel = pid.replace("\\", "/")
            rel_candidates = self._with_yaml_suffix(rel)
            for r in rel_candidates:
                p = self.prompt_dir / r
                if p.exists() and p.is_file():
                    return self._load_file(p)
            raise FileNotFoundError(f"Prompt yaml not found: {self.prompt_dir / rel}")

        # 4) Otherwise treat as bare name: search by filename, must be unique.
        name = pid
        candidates = list(self.prompt_dir.rglob(f"{name}.yaml")) + list(self.prompt_dir.rglob(f"{name}.yml"))
        if not candidates:
            raise FileNotFoundError(f"Prompt yaml id not found: {prompt_id} under {self.prompt_dir}")
        if len(candidates) > 1:
            raise RuntimeError(
                f"Ambiguous prompt_id '{prompt_id}'. Multiple matches:\n"
                + "\n".join(str(x) for x in candidates)
            )
        return self._load_file(candidates[0])

    # ----------------------------
    # Public: Rendering
    # ----------------------------
    def render_prompt(
        self,
        spec: Union[YamlPromptSpec, str],
        *,
        task_values: Optional[Dict[str, Any]] = None,
        static_values: Optional[Dict[str, Any]] = None,
        strict: bool = True,
    ) -> str:
        # Alias of render() for compatibility with JSONPromptLoader.render_prompt
        return self.render(
            spec,
            task_values=task_values,
            static_values=static_values,
            strict=strict,
        )
    
    def render(
        self,
        spec: Union[YamlPromptSpec, str],
        *,
        task_values: Optional[Dict[str, Any]] = None,
        static_values: Optional[Dict[str, Any]] = None,
        strict: bool = True,
    ) -> str:
        """
        Render a YAML prompt.
        - spec can be a YamlPromptSpec or a string id/path.
        - IMPORTANT (Option A):
            Only treat it as load_by_path if it explicitly ends with .yaml/.yml.
            Otherwise always load_by_id (even if it contains '/').
        - strict=True: enforce required vars are present.
        """
        spec_obj: YamlPromptSpec
        if isinstance(spec, str):
            s = (spec or "").strip()
            if not s:
                raise ValueError("spec must be a non-empty string")

            if s.endswith(".yaml") or s.endswith(".yml"):
                spec_obj = self.load_by_path(s)
            else:
                # This is the key fix: nested dirs via slash are handled by load_by_id
                spec_obj = self.load_by_id(s)
        else:
            spec_obj = spec

        task_values = task_values or {}
        static_values = static_values or {}

        declared = _declared_names(spec_obj.task_variables) | _declared_names(spec_obj.static_variables)
        required_task = _required_names(spec_obj.task_variables)
        required_static = _required_names(spec_obj.static_variables)

        merged_values: Dict[str, Any] = {}
        merged_values.update(self.global_static)
        merged_values.update(static_values)
        merged_values.update(task_values)

        if strict:
            missing_static = sorted([k for k in required_static if k not in merged_values])
            missing_task = sorted([k for k in required_task if k not in merged_values])
            if missing_static or missing_task:
                raise ValueError(f"Missing vars: static={missing_static} task={missing_task}")

        return _safe_replace(spec_obj.template, declared=declared, values=merged_values)

    # ----------------------------
    # Internals
    # ----------------------------

    def _load_file(self, path: Path) -> YamlPromptSpec:
        with open(path, "r", encoding="utf-8") as f:
            obj = yaml.safe_load(f) or {}

        if not isinstance(obj, dict):
            raise ValueError(f"Invalid yaml (not dict): {path}")

        pid = str(obj.get("id") or "").strip() or path.stem
        template = obj.get("template")

        if not isinstance(template, str) or not template.strip():
            raise ValueError(f"Missing template in {path}")

        task_vars = obj.get("task_variables") or []
        static_vars = obj.get("static_variables") or []

        if not isinstance(task_vars, list):
            raise ValueError(f"task_variables must be list in {path}")
        if not isinstance(static_vars, list):
            raise ValueError(f"static_variables must be list in {path}")

        return YamlPromptSpec(
            id=pid,
            template=template,
            task_variables=task_vars,
            static_variables=static_vars,
            raw=obj,
        )

    def _strip_prompt_dir_prefix(self, pid: str) -> str:
        """
        If pid includes prompt_dir as a prefix (absolute or string prefix), strip it.
        """
        s = pid.strip()
        if not s:
            return s

        # Try resolved path comparison when possible
        try:
            prompt_dir_resolved = self.prompt_dir.resolve()
            pid_path = Path(s)
            # If pid_path is relative, resolve relative to cwd; this can be misleading,
            # so only use resolve-stripping when the resolved pid truly starts with prompt_dir.
            pid_resolved = pid_path.resolve()
            if str(pid_resolved).startswith(str(prompt_dir_resolved) + str(Path.sep)):
                rel = pid_resolved.relative_to(prompt_dir_resolved)
                return rel.as_posix()
        except Exception:
            pass

        # Fallback: pure string prefix strip
        prompt_dir_str = str(self.prompt_dir).rstrip("/\\")
        norm_pid = s.replace("\\", "/")
        norm_dir = prompt_dir_str.replace("\\", "/").rstrip("/")
        if norm_pid.startswith(norm_dir + "/"):
            return norm_pid[len(norm_dir) + 1 :]

        return s

    @staticmethod
    def _with_yaml_suffix(rel: str) -> List[str]:
        r = (rel or "").strip()
        if not r:
            return []
        if r.endswith(".yaml") or r.endswith(".yml"):
            return [r]
        return [r + ".yaml", r + ".yml"]
from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional


logger = logging.getLogger(__name__)


def _normalize_language(language: str) -> str:
    lang = str(language or "").strip().lower()
    return lang if lang in {"zh", "en"} else "zh"


class ToolMetadataProvider:
    """
    Load tool metadata from task_specs/tool_metadata/{language}.

    Each JSON file should be a dict keyed by tool name:
    {
      "tool_name": {
        "name": "tool_name",
        "description": "...",
        "parameters": [...]
      }
    }
    """

    def __init__(
        self,
        metadata_dir: str,
        *,
        language: str = "zh",
        fallback_language: Optional[str] = "zh",
        runtime_metadata_dir: str = "",
    ) -> None:
        self.language = _normalize_language(language)
        self.fallback_language = _normalize_language(fallback_language or "")
        self.metadata_dir = Path(str(metadata_dir or "")).expanduser()
        self.runtime_metadata_dir = Path(str(runtime_metadata_dir or "")).expanduser() if str(runtime_metadata_dir or "").strip() else None
        self._metadata: Dict[str, Dict[str, Any]] = {}
        self._fallback_metadata: Dict[str, Dict[str, Any]] = {}
        self._load()

    @classmethod
    def from_config(cls, config: Any, *, include_runtime_overrides: bool = False) -> "ToolMetadataProvider":
        global_cfg = getattr(config, "global_config", None) or getattr(config, "global_", None)
        metadata_dir = str(getattr(global_cfg, "tool_metadata_dir", "") or "").strip()
        language = str(getattr(global_cfg, "language", "") or getattr(global_cfg, "locale", "") or "zh").strip()
        task_specs_root = Path(str(getattr(global_cfg, "task_specs_root", "./task_specs") or "./task_specs")).expanduser()
        if not metadata_dir:
            metadata_dir = str(task_specs_root / "tool_metadata" / _normalize_language(language))
        fallback_language = "zh" if _normalize_language(language) != "zh" else None
        runtime_metadata_dir = ""
        if include_runtime_overrides:
            sm_cfg = getattr(config, "strategy_memory", None)
            runtime_metadata_dir = str(getattr(sm_cfg, "tool_metadata_runtime_dir", "") or "").strip()
        return cls(
            metadata_dir,
            language=language,
            fallback_language=fallback_language,
            runtime_metadata_dir=runtime_metadata_dir,
        )

    def _load(self) -> None:
        self._metadata = self._load_language_dir(self.metadata_dir)
        if self.runtime_metadata_dir is not None:
            runtime_dir = self.runtime_metadata_dir / self.language
            self._metadata = self._merge_metadata(self._metadata, self._load_language_dir(runtime_dir))
        if self.fallback_language and self.fallback_language != self.language:
            fallback_dir = self.metadata_dir.parent / self.fallback_language
            self._fallback_metadata = self._load_language_dir(fallback_dir)
            if self.runtime_metadata_dir is not None:
                runtime_fallback_dir = self.runtime_metadata_dir / self.fallback_language
                self._fallback_metadata = self._merge_metadata(
                    self._fallback_metadata,
                    self._load_language_dir(runtime_fallback_dir),
                )
        else:
            self._fallback_metadata = {}

    @staticmethod
    def _load_language_dir(dir_path: Path) -> Dict[str, Dict[str, Any]]:
        if not dir_path.exists() or not dir_path.is_dir():
            return {}
        merged: Dict[str, Dict[str, Any]] = {}
        for path in sorted(dir_path.glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning("Failed to load tool metadata file: %s error=%s", path, exc)
                continue
            if not isinstance(payload, dict):
                logger.warning("Skip invalid tool metadata file (not dict): %s", path)
                continue
            for tool_name, meta in payload.items():
                if not isinstance(meta, dict):
                    continue
                name = str(meta.get("name") or tool_name or "").strip()
                if not name:
                    continue
                row: Dict[str, Any] = {"name": name}
                if "description" in meta:
                    row["description"] = str(meta.get("description") or "").strip()
                if isinstance(meta.get("parameters"), list):
                    row["parameters"] = copy.deepcopy(meta.get("parameters") or [])
                merged[name] = row
        return merged

    @staticmethod
    def _merge_metadata(base: Dict[str, Dict[str, Any]], override: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        if not override:
            return dict(base or {})
        merged: Dict[str, Dict[str, Any]] = {k: copy.deepcopy(v) for k, v in (base or {}).items()}
        for tool_name, meta in (override or {}).items():
            current = copy.deepcopy(merged.get(tool_name, {}))
            current.update(copy.deepcopy(meta or {}))
            merged[tool_name] = current
        return merged

    def get_tool_metadata(self, tool_name: str) -> Optional[Dict[str, Any]]:
        name = str(tool_name or "").strip()
        if not name:
            return None
        meta = self._metadata.get(name) or self._fallback_metadata.get(name)
        if not meta:
            return None
        return copy.deepcopy(meta)

    def resolve_tool_metadata(
        self,
        tool_name: str,
        *,
        fallback_description: str = "",
        fallback_parameters: Optional[list[dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        meta = self.get_tool_metadata(tool_name) or {}
        parameters = copy.deepcopy(meta.get("parameters") if isinstance(meta.get("parameters"), list) else None)
        if parameters is None:
            parameters = copy.deepcopy(fallback_parameters or [])
        description = str(meta.get("description") if "description" in meta else fallback_description or "").strip()
        return {
            "name": str(meta.get("name") or tool_name or "").strip(),
            "description": description,
            "parameters": parameters,
        }

    def apply_to_tool(self, tool: Any) -> Any:
        tool_name = str(getattr(tool, "name", "") or "").strip()
        if not tool_name:
            return tool
        meta = self.get_tool_metadata(tool_name)
        if not meta:
            return tool
        if meta.get("name"):
            setattr(tool, "name", str(meta["name"]))
        if "description" in meta:
            setattr(tool, "description", str(meta.get("description") or ""))
        if "parameters" in meta and isinstance(meta.get("parameters"), list):
            setattr(tool, "parameters", copy.deepcopy(meta["parameters"]))
        return tool

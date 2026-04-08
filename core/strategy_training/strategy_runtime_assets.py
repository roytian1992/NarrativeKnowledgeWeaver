from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List

from core.utils.general_utils import json_dump_atomic


class StrategyRuntimeAssetManager:
    def __init__(self, *, library_path: str, tool_metadata_runtime_dir: str) -> None:
        self.library_path = Path(str(library_path or "").strip()).expanduser()
        self.tool_metadata_runtime_dir = Path(str(tool_metadata_runtime_dir or "").strip()).expanduser()

    def clear_library(self, *, aggregation_mode: str = "narrative", dataset_name: str = "") -> Dict[str, Any]:
        payload = {
            "library_version": 3,
            "aggregation_mode": str(aggregation_mode or "narrative").strip() or "narrative",
            "dataset_name": str(dataset_name or "").strip(),
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "pattern_count": 0,
            "patterns": [],
            "template_count": 0,
            "templates": [],
        }
        self.library_path.parent.mkdir(parents=True, exist_ok=True)
        json_dump_atomic(str(self.library_path), payload)
        return payload

    def clear_tool_metadata(self) -> None:
        if self.tool_metadata_runtime_dir.exists():
            shutil.rmtree(self.tool_metadata_runtime_dir)
        for lang in ("zh", "en"):
            lang_dir = self.tool_metadata_runtime_dir / lang
            lang_dir.mkdir(parents=True, exist_ok=True)
            json_dump_atomic(str(lang_dir / "strategy_runtime_overrides.json"), {})

    def clear_all(self, *, aggregation_mode: str = "narrative", dataset_name: str = "") -> Dict[str, Any]:
        payload = self.clear_library(aggregation_mode=aggregation_mode, dataset_name=dataset_name)
        self.clear_tool_metadata()
        source_index_path = self.library_path.parent / "template_source_index.json"
        json_dump_atomic(str(source_index_path), {})
        return payload

    def export_tool_metadata_overrides(self, rows: List[Dict[str, Any]]) -> Dict[str, str]:
        grouped: Dict[str, Dict[str, Dict[str, Any]]] = {"zh": {}, "en": {}}
        for row in rows or []:
            if not isinstance(row, dict):
                continue
            if str(row.get("decision", "") or "").strip().lower() != "revise":
                continue
            tool_name = str(row.get("tool_name", "") or "").strip()
            proposed_description = str(row.get("proposed_description", "") or "").strip()
            language = str(row.get("language", "") or "").strip().lower()
            if language not in grouped or not tool_name or not proposed_description:
                continue
            grouped[language][tool_name] = {
                "name": tool_name,
                "description": proposed_description,
            }

        written: Dict[str, str] = {}
        for language, payload in grouped.items():
            lang_dir = self.tool_metadata_runtime_dir / language
            lang_dir.mkdir(parents=True, exist_ok=True)
            out_path = lang_dir / "strategy_runtime_overrides.json"
            json_dump_atomic(str(out_path), payload)
            written[language] = str(out_path)
        return written

    def export_template_source_index(self, payload: Dict[str, Any]) -> str:
        out_path = self.library_path.parent / "template_source_index.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        json_dump_atomic(str(out_path), payload if isinstance(payload, dict) else {})
        return str(out_path)

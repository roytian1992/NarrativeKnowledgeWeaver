from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List

from core.utils.general_utils import append_jsonl, ensure_dir, load_jsonl


@dataclass
class OnlineBufferPaths:
    root_dir: str
    real_trace_path: str
    strategy_buffer_path: str
    failure_reflection_path: str
    synthetic_qa_path: str


class OnlineStrategyBuffer:
    def __init__(self, paths: OnlineBufferPaths) -> None:
        self.paths = paths
        ensure_dir(self.paths.root_dir)

    @classmethod
    def from_config(cls, config) -> "OnlineStrategyBuffer":
        cfg = getattr(config, "strategy_memory", None)
        root_dir = str(getattr(cfg, "online_buffer_dir", "data/memory/online") or "data/memory/online").strip()
        paths = OnlineBufferPaths(
            root_dir=root_dir,
            real_trace_path=str(getattr(cfg, "online_real_trace_path", os.path.join(root_dir, "real_traces.jsonl")) or os.path.join(root_dir, "real_traces.jsonl")).strip(),
            strategy_buffer_path=str(getattr(cfg, "online_strategy_buffer_path", os.path.join(root_dir, "online_strategy_buffer.jsonl")) or os.path.join(root_dir, "online_strategy_buffer.jsonl")).strip(),
            failure_reflection_path=str(getattr(cfg, "online_failure_reflection_path", os.path.join(root_dir, "failure_reflections.jsonl")) or os.path.join(root_dir, "failure_reflections.jsonl")).strip(),
            synthetic_qa_path=str(getattr(cfg, "online_synthetic_qa_path", os.path.join(root_dir, "synthetic_qa_buffer.jsonl")) or os.path.join(root_dir, "synthetic_qa_buffer.jsonl")).strip(),
        )
        return cls(paths)

    def append_real_trace(self, row: Dict[str, Any]) -> None:
        append_jsonl(self.paths.real_trace_path, row)

    def append_strategy_candidate(self, row: Dict[str, Any]) -> None:
        append_jsonl(self.paths.strategy_buffer_path, row)

    def append_failure_reflection(self, row: Dict[str, Any]) -> None:
        append_jsonl(self.paths.failure_reflection_path, row)

    def append_synthetic_qa(self, row: Dict[str, Any]) -> None:
        append_jsonl(self.paths.synthetic_qa_path, row)

    def load_real_traces(self) -> List[Dict[str, Any]]:
        return load_jsonl(self.paths.real_trace_path)

    def load_strategy_candidates(self) -> List[Dict[str, Any]]:
        return load_jsonl(self.paths.strategy_buffer_path)

    def load_failure_reflections(self) -> List[Dict[str, Any]]:
        return load_jsonl(self.paths.failure_reflection_path)

    def load_synthetic_qas(self) -> List[Dict[str, Any]]:
        return load_jsonl(self.paths.synthetic_qa_path)


from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from core.utils.config import KAGConfig


logger = logging.getLogger(__name__)


def resolve_task_specs_root(cfg: KAGConfig, *, repo_root: Optional[Path] = None) -> Path:
    raw_root = str(getattr(getattr(cfg, "global_config", None), "task_specs_root", "") or "./task_specs").strip()
    root = Path(raw_root)
    if root.is_absolute():
        return root
    base = repo_root or Path.cwd()
    return (base / root).resolve()


def load_task_spec_json(
    cfg: KAGConfig,
    *,
    relative_path: str,
    repo_root: Optional[Path] = None,
    default: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    fallback = dict(default or {})
    task_specs_root = resolve_task_specs_root(cfg, repo_root=repo_root)
    spec_path = task_specs_root / str(relative_path or "").strip()
    if not spec_path.exists():
        logger.warning("Task spec not found: %s", spec_path)
        return fallback
    try:
        payload = json.loads(spec_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Failed to load task spec: %s error=%s", spec_path, exc)
        return fallback
    if not isinstance(payload, dict):
        logger.warning("Task spec must be a JSON object: %s", spec_path)
        return fallback
    return payload

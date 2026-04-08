from __future__ import annotations

from pathlib import Path
import sys


def _find_external_package_init() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    current_init = Path(__file__).resolve()
    for raw_path in list(sys.path):
        if not raw_path:
            continue
        try:
            base = Path(raw_path).resolve()
        except Exception:
            continue
        if base == repo_root:
            continue
        candidate = base / "qwen_agent" / "__init__.py"
        if candidate.exists() and candidate.resolve() != current_init:
            return candidate.resolve()
    raise ImportError(
        "Unable to locate the external 'qwen_agent' package outside the repo root. "
        "Install it into the active environment before using the qwen backend."
    )


_EXTERNAL_INIT = _find_external_package_init()
__file__ = str(_EXTERNAL_INIT)
__path__ = [str(_EXTERNAL_INIT.parent)]

with _EXTERNAL_INIT.open("rb") as f:
    _CODE = compile(f.read(), str(_EXTERNAL_INIT), "exec")

exec(_CODE, globals(), globals())

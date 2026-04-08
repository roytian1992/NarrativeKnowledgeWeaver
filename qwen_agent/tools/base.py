from __future__ import annotations

from typing import Any, Callable, TypeVar


T = TypeVar("T")


class BaseTool:
    name = ""
    description = ""
    parameters = []

    def call(self, params: Any, **kwargs) -> Any:
        raise NotImplementedError


def register_tool(name: str) -> Callable[[T], T]:
    tool_name = str(name or "").strip()

    def decorator(obj: T) -> T:
        if tool_name and not getattr(obj, "name", ""):
            setattr(obj, "name", tool_name)
        return obj

    return decorator


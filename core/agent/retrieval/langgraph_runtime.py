from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Sequence, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, StateGraph


logger = logging.getLogger(__name__)


class _AgentState(TypedDict):
    messages: List[BaseMessage]
    remaining_llm_calls: int
    remaining_tool_calls: int


_JSON_TYPE_MAP: Dict[str, str] = {
    "str": "string",
    "string": "string",
    "int": "integer",
    "integer": "integer",
    "float": "number",
    "number": "number",
    "bool": "boolean",
    "boolean": "boolean",
    "array": "array",
    "list": "array",
    "object": "object",
    "dict": "object",
}


def _message_get(message: Any, key: str, default: Any = None) -> Any:
    if isinstance(message, dict):
        return message.get(key, default)
    getter = getattr(message, "get", None)
    if callable(getter):
        try:
            return getter(key, default)
        except Exception:
            pass
    return getattr(message, key, default)


def _normalize_json_type(value: Any) -> str:
    raw = str(value or "").strip().lower()
    return _JSON_TYPE_MAP.get(raw, "string")


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict):
                text = str(item.get("text") or item.get("content") or "").strip()
                if text:
                    parts.append(text)
                    continue
            parts.append(str(item))
        return "\n".join(part for part in parts if part)
    if content is None:
        return ""
    return str(content)


def _serialize_tool_args(args: Any) -> str:
    if isinstance(args, str):
        return args
    try:
        return json.dumps(args if args is not None else {}, ensure_ascii=False)
    except Exception:
        return json.dumps({"value": str(args)}, ensure_ascii=False)


def _extract_first_json_object(text: Any) -> Optional[Dict[str, Any]]:
    raw = str(text or "").strip()
    if not raw:
        return None
    try:
        payload = json.loads(raw)
        return payload if isinstance(payload, dict) else None
    except Exception:
        pass

    start = raw.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for idx in range(start, len(raw)):
        ch = raw[idx]
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = raw[start : idx + 1]
                try:
                    payload = json.loads(candidate)
                except Exception:
                    return None
                return payload if isinstance(payload, dict) else None
    return None


def _clip_text(value: Any, *, limit: int = 4000) -> str:
    text = _content_to_text(value).strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


class LangGraphAssistantRuntime:
    """LangGraph-backed assistant runtime that keeps the existing tool contract."""

    def __init__(
        self,
        *,
        function_list: Sequence[Any],
        llm: Any,
        system_message: str,
        rag_cfg: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.function_list = list(function_list or [])
        self.llm = llm
        self.system_message = str(system_message or "").strip()
        self.rag_cfg = dict(rag_cfg or {})
        self.max_tool_calls_per_run = max(1, int(self.rag_cfg.get("max_tool_calls_per_run", 3) or 3))
        self.tool_map = {
            str(getattr(tool, "name", "") or "").strip(): tool
            for tool in self.function_list
            if str(getattr(tool, "name", "") or "").strip()
        }
        self._manual_tool_calling_enabled = bool(self.tool_map)
        self._manual_tool_system_message = self._build_manual_tool_system_message()
        self.graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(_AgentState)
        graph.add_node("model", self._model_node)
        graph.add_node("tools", self._tools_node)
        graph.set_entry_point("model")
        graph.add_conditional_edges("model", self._route_after_model, {"tools": "tools", "end": END})
        graph.add_edge("tools", "model")
        return graph.compile()

    def _build_manual_tool_system_message(self) -> str:
        lines: List[str] = []
        if self.system_message:
            lines.append(self.system_message)
        if not self.tool_map:
            return "\n\n".join(part for part in lines if part).strip()

        lines.extend(
            [
                "Use JSON-only tool calling.",
                "If you need a tool, respond with JSON only:",
                '{"tool_name":"<tool name>","tool_arguments":{"arg":"value"}}',
                "If you can answer the user directly, respond with JSON only:",
                '{"final_answer":"<answer>"}',
                "Rules:",
                "- Call at most one tool per turn.",
                f"- Across the whole question, use at most {self.max_tool_calls_per_run} tool calls in total.",
                "- Use only the listed tool names.",
                "- `tool_arguments` must be a JSON object.",
                "- Do not wrap JSON in markdown fences.",
                "- For multiple-choice QA, compare the competing options against retrieved evidence before producing `final_answer`.",
                "- If the answer depends on motive, attitude, warning, implication, or a specific scene, prefer content retrieval tools over entity-profile shortcuts.",
                "- Build a short evidence chain across turns when needed, for example A -> B -> C, instead of calling many tools at once.",
                "",
                "Available tools:",
            ]
        )
        for tool in self.function_list:
            name = str(getattr(tool, "name", "") or "").strip()
            if not name:
                continue
            description = str(getattr(tool, "description", "") or "").strip()
            param_parts: List[str] = []
            for row in list(getattr(tool, "parameters", []) or []):
                if not isinstance(row, dict):
                    continue
                param_name = str(row.get("name") or "").strip()
                if not param_name:
                    continue
                param_type = _normalize_json_type(row.get("type"))
                required = " required" if bool(row.get("required")) else ""
                param_desc = str(row.get("description") or "").strip()
                if param_desc:
                    param_parts.append(f"{param_name}:{param_type}{required} - {param_desc}")
                else:
                    param_parts.append(f"{param_name}:{param_type}{required}")
            tool_line = f"- {name}: {description}" if description else f"- {name}"
            if param_parts:
                tool_line += f" | params: {'; '.join(param_parts)}"
            lines.append(tool_line)
        return "\n".join(part for part in lines if part).strip()

    def _build_manual_tool_messages(
        self,
        messages: Sequence[BaseMessage],
        *,
        remaining_tool_calls: int,
    ) -> List[BaseMessage]:
        transcript_lines: List[str] = []
        for message in list(messages or []):
            if isinstance(message, SystemMessage):
                continue
            if isinstance(message, HumanMessage):
                text = _clip_text(message.content, limit=3000)
                if text:
                    transcript_lines.append(f"User:\n{text}")
                continue
            if isinstance(message, AIMessage):
                tool_calls = list(getattr(message, "tool_calls", []) or [])
                if tool_calls:
                    for tool_call in tool_calls:
                        transcript_lines.append(
                            "Assistant requested tool:\n"
                            + json.dumps(
                                {
                                    "tool_name": str(tool_call.get("name") or "").strip(),
                                    "tool_arguments": tool_call.get("args", {}) if isinstance(tool_call.get("args", {}), dict) else {"value": tool_call.get("args")},
                                },
                                ensure_ascii=False,
                            )
                        )
                    text = _clip_text(message.content, limit=2000)
                    if text:
                        transcript_lines.append(f"Assistant:\n{text}")
                    continue
                text = _clip_text(message.content, limit=3000)
                if text:
                    transcript_lines.append(f"Assistant:\n{text}")
                continue
            if isinstance(message, ToolMessage):
                tool_name = str(getattr(message, "name", "") or "unknown_tool").strip()
                transcript_lines.append(f"Tool result ({tool_name}):\n{_clip_text(message.content, limit=6000)}")

        transcript = "\n\n".join(line for line in transcript_lines if line).strip()
        if not transcript:
            transcript = "No prior conversation."
        human_prompt = "\n\n".join(
            [
                "Conversation so far:",
                transcript,
                f"Remaining tool-call budget for this question: {max(0, int(remaining_tool_calls))}.",
                "If the remaining budget is 0, do not call any tool and answer directly.",
                "Respond with JSON only.",
            ]
        ).strip()
        return [
            SystemMessage(content=self._manual_tool_system_message),
            HumanMessage(content=human_prompt),
        ]

    def _coerce_manual_tool_response(self, response: BaseMessage, *, call_index: int) -> AIMessage:
        text = _content_to_text(getattr(response, "content", ""))
        payload = _extract_first_json_object(text)
        if isinstance(payload, dict):
            tool_name = str(
                payload.get("tool_name")
                or payload.get("name")
                or payload.get("tool")
                or ""
            ).strip()
            if tool_name and tool_name in self.tool_map:
                raw_args = (
                    payload.get("tool_arguments")
                    if "tool_arguments" in payload
                    else payload.get("arguments")
                )
                if raw_args is None:
                    raw_args = payload.get("args", {})
                if not isinstance(raw_args, dict):
                    raw_args = {"value": raw_args}
                return AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": f"manual_tool_call_{call_index}_{tool_name}",
                            "name": tool_name,
                            "args": raw_args,
                            "type": "tool_call",
                        }
                    ],
                )
            final_answer = payload.get("final_answer")
            if final_answer is None and "answer" in payload and not tool_name:
                final_answer = payload.get("answer")
            if final_answer is not None:
                return AIMessage(content=_content_to_text(final_answer))
        return AIMessage(content=text)

    def _model_node(self, state: _AgentState) -> Dict[str, Any]:
        messages = list(state.get("messages") or [])
        remaining = int(state.get("remaining_llm_calls", 0) or 0)
        remaining_tool_calls = int(state.get("remaining_tool_calls", 0) or 0)
        if remaining <= 0:
            return {
                "messages": messages,
                "remaining_llm_calls": 0,
                "remaining_tool_calls": max(0, remaining_tool_calls),
            }
        if self._manual_tool_calling_enabled:
            manual_messages = self._build_manual_tool_messages(
                messages,
                remaining_tool_calls=remaining_tool_calls,
            )
            raw_response = self.llm.invoke(manual_messages)
            response = self._coerce_manual_tool_response(raw_response, call_index=remaining)
            if remaining_tool_calls <= 0 and isinstance(response, AIMessage) and list(getattr(response, "tool_calls", []) or []):
                response = AIMessage(
                    content=json.dumps(
                        {
                            "final_answer": "I have exhausted the tool-call budget for this question and must answer from the evidence already retrieved."
                        },
                        ensure_ascii=False,
                    )
                )
                response = self._coerce_manual_tool_response(response, call_index=remaining)
        else:
            response = self.llm.invoke(messages)
        return {
            "messages": messages + [response],
            "remaining_llm_calls": remaining - 1,
            "remaining_tool_calls": max(0, remaining_tool_calls),
        }

    def _route_after_model(self, state: _AgentState) -> str:
        messages = list(state.get("messages") or [])
        if not messages:
            return "end"
        last = messages[-1]
        remaining_tool_calls = int(state.get("remaining_tool_calls", 0) or 0)
        if isinstance(last, AIMessage) and list(getattr(last, "tool_calls", []) or []) and remaining_tool_calls > 0:
            return "tools"
        return "end"

    def _tools_node(self, state: _AgentState) -> Dict[str, Any]:
        messages = list(state.get("messages") or [])
        if not messages:
            return {
                "messages": messages,
                "remaining_llm_calls": int(state.get("remaining_llm_calls", 0) or 0),
                "remaining_tool_calls": int(state.get("remaining_tool_calls", 0) or 0),
            }
        last = messages[-1]
        if not isinstance(last, AIMessage):
            return {
                "messages": messages,
                "remaining_llm_calls": int(state.get("remaining_llm_calls", 0) or 0),
                "remaining_tool_calls": int(state.get("remaining_tool_calls", 0) or 0),
            }

        tool_messages: List[ToolMessage] = []
        for tool_call in list(getattr(last, "tool_calls", []) or []):
            tool_name = str(tool_call.get("name") or "").strip()
            tool_call_id = str(tool_call.get("id") or tool_name or "tool_call")
            args = tool_call.get("args", {})
            tool = self.tool_map.get(tool_name)

            if tool is None:
                output = json.dumps(
                    {
                        "error": {
                            "code": "tool_not_found",
                            "message": f"Tool not found: {tool_name}",
                        }
                    },
                    ensure_ascii=False,
                )
                status = "error"
            else:
                try:
                    output = tool.call(_serialize_tool_args(args))
                    status = "success"
                except Exception as exc:
                    logger.exception("LangGraph tool execution failed: tool=%s err=%s", tool_name, exc)
                    output = json.dumps(
                        {
                            "error": {
                                "code": "tool_execution_failed",
                                "message": str(exc),
                            }
                        },
                        ensure_ascii=False,
                    )
                    status = "error"

            tool_messages.append(
                ToolMessage(
                    content=_content_to_text(output),
                    name=tool_name or None,
                    tool_call_id=tool_call_id,
                    status=status,
                )
            )

        return {
            "messages": messages + tool_messages,
            "remaining_llm_calls": int(state.get("remaining_llm_calls", 0) or 0),
            "remaining_tool_calls": max(0, int(state.get("remaining_tool_calls", 0) or 0) - len(tool_messages)),
        }

    @staticmethod
    def _coerce_input_message(message: Any) -> Optional[BaseMessage]:
        if isinstance(message, BaseMessage):
            return message
        role = str(_message_get(message, "role") or "").strip().lower()
        content = _content_to_text(_message_get(message, "content", ""))
        if role == "system":
            return SystemMessage(content=content)
        if role == "user":
            return HumanMessage(content=content)
        if role == "assistant":
            function_call = _message_get(message, "function_call")
            if isinstance(function_call, dict):
                raw_args = function_call.get("arguments", {})
                if isinstance(raw_args, str):
                    try:
                        raw_args = json.loads(raw_args)
                    except Exception:
                        raw_args = {"raw": raw_args}
                return AIMessage(
                    content=content,
                    tool_calls=[
                        {
                            "id": str(_message_get(message, "tool_call_id") or function_call.get("name") or "tool_call"),
                            "name": str(function_call.get("name") or "").strip(),
                            "args": raw_args if isinstance(raw_args, dict) else {"value": raw_args},
                            "type": "tool_call",
                        }
                    ],
                )
            return AIMessage(content=content)
        if role == "function":
            return ToolMessage(
                content=content,
                name=str(_message_get(message, "name") or "").strip() or None,
                tool_call_id=str(_message_get(message, "tool_call_id") or _message_get(message, "name") or "tool_call"),
            )
        return None

    def _to_legacy_responses(self, messages: Sequence[BaseMessage]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        pending_calls: Dict[str, Dict[str, Any]] = {}

        for message in messages:
            if isinstance(message, SystemMessage):
                continue
            if isinstance(message, HumanMessage):
                continue
            if isinstance(message, AIMessage):
                tool_calls = list(getattr(message, "tool_calls", []) or [])
                if tool_calls:
                    for tool_call in tool_calls:
                        call_id = str(tool_call.get("id") or tool_call.get("name") or "tool_call")
                        pending_calls[call_id] = dict(tool_call)
                    content = _content_to_text(message.content).strip()
                    if content:
                        out.append({"role": "assistant", "content": content})
                    continue
                out.append({"role": "assistant", "content": _content_to_text(message.content)})
                continue
            if isinstance(message, ToolMessage):
                call_id = str(getattr(message, "tool_call_id", "") or getattr(message, "id", "") or getattr(message, "name", "") or "tool_call")
                tool_call = pending_calls.pop(call_id, None)
                if tool_call is None and pending_calls:
                    first_key = next(iter(pending_calls.keys()))
                    tool_call = pending_calls.pop(first_key)
                    call_id = first_key
                if tool_call is not None:
                    out.append(
                        {
                            "role": "assistant",
                            "content": "",
                            "function_call": {
                                "name": str(tool_call.get("name") or getattr(message, "name", "") or "unknown_tool"),
                                "arguments": _serialize_tool_args(tool_call.get("args", {})),
                            },
                            "tool_call_id": call_id,
                        }
                    )
                out.append(
                    {
                        "role": "function",
                        "name": str(getattr(message, "name", "") or (tool_call.get("name") if tool_call else "") or "unknown_tool"),
                        "content": _content_to_text(message.content),
                        "tool_call_id": call_id,
                    }
                )

        for call_id, tool_call in pending_calls.items():
            out.append(
                {
                    "role": "assistant",
                    "content": "",
                    "function_call": {
                        "name": str(tool_call.get("name") or "unknown_tool"),
                        "arguments": _serialize_tool_args(tool_call.get("args", {})),
                    },
                    "tool_call_id": call_id,
                }
            )
        return out

    def run_nonstream(self, messages: List[Dict[str, Any]], lang: str = "zh", **kwargs) -> List[Dict[str, Any]]:
        del lang
        max_llm_calls_per_run = kwargs.pop("max_llm_calls_per_run", None)
        max_calls = 8
        if isinstance(max_llm_calls_per_run, (int, float)) and int(max_llm_calls_per_run) > 0:
            max_calls = int(max_llm_calls_per_run)

        input_messages: List[BaseMessage] = []
        if self.system_message:
            input_messages.append(SystemMessage(content=self.system_message))
        for message in list(messages or []):
            coerced = self._coerce_input_message(message)
            if coerced is not None:
                input_messages.append(coerced)

        result = self.graph.invoke(
            {
                "messages": input_messages,
                "remaining_llm_calls": max(1, max_calls),
                "remaining_tool_calls": self.max_tool_calls_per_run,
            }
        )
        return self._to_legacy_responses(list(result.get("messages") or []))

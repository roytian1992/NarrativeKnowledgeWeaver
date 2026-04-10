from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from typing import Any, Dict, List, Optional, Sequence, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, StateGraph

from core.agent.retrieval.tool_routing_heuristics import heuristic_tool_boosts, tool_stage


logger = logging.getLogger(__name__)


class _AgentState(TypedDict):
    messages: List[BaseMessage]
    remaining_llm_calls: int
    remaining_tool_rounds: int
    tool_round_index: int


class _S4AgentState(TypedDict):
    messages: List[BaseMessage]
    remaining_llm_calls: int
    remaining_tool_rounds: int
    tool_round_index: int
    evidence_pool: List[Dict[str, Any]]
    entities_found: List[Dict[str, Any]]
    plan_notes: str
    evaluation_notes: str
    next_step: str
    draft_answer: str
    last_round_new_evidence_count: int
    stagnation_count: int


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


def _extract_first_json_payload(text: Any) -> Optional[Any]:
    raw = str(text or "").strip()
    if not raw:
        return None
    try:
        payload = json.loads(raw)
        return payload if isinstance(payload, (dict, list)) else None
    except Exception:
        pass

    start_candidates = [(raw.find("{"), "{", "}"), (raw.find("["), "[", "]")]
    start_candidates = [row for row in start_candidates if row[0] >= 0]
    if not start_candidates:
        return None
    start, opening, closing = min(start_candidates, key=lambda row: row[0])
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
        if ch == opening:
            depth += 1
        elif ch == closing:
            depth -= 1
            if depth == 0:
                candidate = raw[start : idx + 1]
                try:
                    payload = json.loads(candidate)
                except Exception:
                    return None
                return payload if isinstance(payload, (dict, list)) else None
    return None


def _clip_text(value: Any, *, limit: int = 4000) -> str:
    text = _content_to_text(value).strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


_CHOICE_LINE_RE = re.compile(r"(?m)^\s*([A-Z])\.\s+")

_OPTION_STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "of",
    "to",
    "in",
    "on",
    "for",
    "from",
    "with",
    "by",
    "as",
    "at",
    "into",
    "during",
    "about",
    "after",
    "before",
    "between",
    "through",
    "under",
    "over",
    "is",
    "was",
    "were",
    "are",
    "be",
    "been",
    "being",
    "it",
    "its",
    "their",
    "them",
    "his",
    "her",
    "they",
    "he",
    "she",
    "that",
    "this",
    "these",
    "those",
    "what",
    "which",
    "who",
    "whom",
    "whose",
    "when",
    "where",
    "why",
    "how",
    "did",
    "does",
    "do",
    "doing",
    "done",
    "have",
    "has",
    "had",
    "having",
    "more",
    "most",
    "least",
    "likely",
    "option",
    "choice",
    "character",
    "person",
    "people",
}


def _stable_digest(value: Any) -> str:
    text = _content_to_text(value)
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()


def _parse_mcq_question(question_text: str) -> Dict[str, Any]:
    raw = str(question_text or "").strip()
    matches = list(_CHOICE_LINE_RE.finditer(raw))
    if not matches:
        return {"question_stem": raw, "choices": {}, "choice_order": []}
    stem = raw[: matches[0].start()].strip()
    choices: Dict[str, str] = {}
    order: List[str] = []
    for idx, match in enumerate(matches):
        label = str(match.group(1) or "").strip().upper()
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(raw)
        text = raw[start:end].strip()
        if not label or not text:
            continue
        choices[label] = text
        order.append(label)
    return {"question_stem": stem or raw, "choices": choices, "choice_order": order}


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
        self.max_tool_rounds_per_run = max(
            1,
            int(
                self.rag_cfg.get(
                    "max_tool_rounds_per_run",
                    self.rag_cfg.get("max_tool_calls_per_run", 3),
                )
                or 3
            ),
        )
        self.first_round_max_tool_calls = max(
            1,
            int(self.rag_cfg.get("first_round_max_tool_calls", 5) or 5),
        )
        self.followup_round_max_tool_calls = max(
            1,
            int(self.rag_cfg.get("followup_round_max_tool_calls", 1) or 1),
        )
        self.parallel_tool_workers = max(
            1,
            int(
                self.rag_cfg.get(
                    "parallel_tool_workers",
                    min(4, self.first_round_max_tool_calls),
                )
                or min(4, self.first_round_max_tool_calls)
            ),
        )
        self.tool_cache_max_entries = max(
            0,
            int(self.rag_cfg.get("tool_cache_max_entries", 256) or 256),
        )
        self.tool_map = {
            str(getattr(tool, "name", "") or "").strip(): tool
            for tool in self.function_list
            if str(getattr(tool, "name", "") or "").strip()
        }
        self._manual_tool_calling_enabled = bool(self.tool_map)
        self._tool_cache_lock = Lock()
        self._tool_result_cache: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
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
                "If you need one tool, respond with JSON only:",
                '{"tool_name":"<tool name>","tool_arguments":{"arg":"value"}}',
                "If you need multiple tools in the same round, respond with JSON only:",
                '{"tool_calls":[{"tool_name":"<tool name>","tool_arguments":{"arg":"value"}}]}',
                "If you can answer the user directly, respond with JSON only:",
                '{"final_answer":"<answer>"}',
                "Rules:",
                f"- Across the whole question, use at most {self.max_tool_rounds_per_run} tool rounds in total.",
                f"- In the first tool round, you may call up to {self.first_round_max_tool_calls} complementary tools in parallel.",
                f"- After the first tool round, call at most {self.followup_round_max_tool_calls} tool per round.",
                "- Use only the listed tool names.",
                "- `tool_arguments` must be a JSON object.",
                "- Do not wrap JSON in markdown fences.",
                "- For multiple-choice QA, compare the competing options against retrieved evidence before producing `final_answer`.",
                "- If the answer depends on motive, attitude, warning, implication, or a specific scene, prefer content retrieval tools over entity-profile shortcuts.",
                "- The first tool round is for broad evidence gathering; later rounds should only refine or verify the most promising clues.",
                "- Build a short evidence chain across rounds when needed, for example [A+B+C] -> D -> answer.",
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
        remaining_tool_rounds: int,
        tool_round_index: int,
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
        next_round_index = max(0, int(tool_round_index)) + 1
        current_round_tool_limit = self._tool_limit_for_round(next_round_index)
        human_prompt = "\n\n".join(
            [
                "Conversation so far:",
                transcript,
                f"Current tool round: {next_round_index}.",
                f"Remaining tool rounds for this question: {max(0, int(remaining_tool_rounds))}.",
                f"In the current tool round, you may call at most {current_round_tool_limit} tool(s).",
                "If the remaining tool rounds are 0, do not call any tool and answer directly.",
                "Respond with JSON only.",
            ]
        ).strip()
        return [
            SystemMessage(content=self._manual_tool_system_message),
            HumanMessage(content=human_prompt),
        ]

    def _tool_limit_for_round(self, round_index: int) -> int:
        if int(round_index or 0) <= 1:
            return self.first_round_max_tool_calls
        return self.followup_round_max_tool_calls

    def _tool_call_signature(self, *, tool_name: str, args: Any) -> str:
        try:
            args_text = json.dumps(args if args is not None else {}, ensure_ascii=False, sort_keys=True)
        except Exception:
            args_text = _serialize_tool_args(args)
        return _stable_digest(f"{str(tool_name or '').strip()}::{args_text}")

    def _get_cached_tool_result(self, *, signature: str) -> Optional[Dict[str, Any]]:
        if self.tool_cache_max_entries <= 0 or not signature:
            return None
        with self._tool_cache_lock:
            cached = self._tool_result_cache.get(signature)
            if cached is None:
                return None
            self._tool_result_cache.move_to_end(signature)
            return dict(cached)

    def _put_cached_tool_result(self, *, signature: str, value: Dict[str, Any]) -> None:
        if self.tool_cache_max_entries <= 0 or not signature:
            return
        with self._tool_cache_lock:
            self._tool_result_cache[signature] = dict(value)
            self._tool_result_cache.move_to_end(signature)
            while len(self._tool_result_cache) > self.tool_cache_max_entries:
                self._tool_result_cache.popitem(last=False)

    def _execute_single_tool_call(self, *, tool_name: str, args: Any) -> Dict[str, Any]:
        signature = self._tool_call_signature(tool_name=tool_name, args=args)
        cached = self._get_cached_tool_result(signature=signature)
        if cached is not None:
            return {
                "tool_name": tool_name,
                "args": args,
                "output": str(cached.get("output") or ""),
                "status": str(cached.get("status") or "success"),
                "cache_hit": True,
                "signature": signature,
            }

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
            return {
                "tool_name": tool_name,
                "args": args,
                "output": output,
                "status": "error",
                "cache_hit": False,
                "signature": signature,
            }

        try:
            output = tool.call(_serialize_tool_args(args))
            result = {
                "tool_name": tool_name,
                "args": args,
                "output": _content_to_text(output),
                "status": "success",
                "cache_hit": False,
                "signature": signature,
            }
            self._put_cached_tool_result(
                signature=signature,
                value={"output": result["output"], "status": result["status"]},
            )
            return result
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
            return {
                "tool_name": tool_name,
                "args": args,
                "output": output,
                "status": "error",
                "cache_hit": False,
                "signature": signature,
            }

    def _execute_tool_calls_batch(
        self,
        *,
        tool_calls: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        calls = list(tool_calls or [])
        if not calls:
            return []
        if len(calls) == 1 or self.parallel_tool_workers <= 1:
            return [
                self._execute_single_tool_call(
                    tool_name=str(call.get("name") or "").strip(),
                    args=call.get("args", {}),
                )
                for call in calls
            ]

        results: List[Optional[Dict[str, Any]]] = [None] * len(calls)
        max_workers = min(len(calls), self.parallel_tool_workers)
        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="nkw-tools") as executor:
            future_map = {
                executor.submit(
                    self._execute_single_tool_call,
                    tool_name=str(call.get("name") or "").strip(),
                    args=call.get("args", {}),
                ): idx
                for idx, call in enumerate(calls)
            }
            for future in as_completed(future_map):
                idx = future_map[future]
                results[idx] = future.result()
        return [row for row in results if isinstance(row, dict)]

    def _normalize_tool_call_payload(
        self,
        row: Any,
        *,
        call_index: int,
        slot_index: int,
    ) -> Optional[Dict[str, Any]]:
        if not isinstance(row, dict):
            return None
        tool_name = str(
            row.get("tool_name")
            or row.get("name")
            or row.get("tool")
            or ""
        ).strip()
        if not tool_name or tool_name not in self.tool_map:
            return None
        raw_args = row.get("tool_arguments")
        if raw_args is None and "arguments" in row:
            raw_args = row.get("arguments")
        if raw_args is None:
            raw_args = row.get("args", {})
        if not isinstance(raw_args, dict):
            raw_args = {"value": raw_args}
        return {
            "id": f"manual_tool_call_{call_index}_{slot_index}_{tool_name}",
            "name": tool_name,
            "args": raw_args,
            "type": "tool_call",
        }

    def _coerce_manual_tool_response(
        self,
        response: BaseMessage,
        *,
        call_index: int,
        tool_round_index: int,
    ) -> AIMessage:
        text = _content_to_text(getattr(response, "content", ""))
        payload = _extract_first_json_payload(text)
        allowed_tool_calls = self._tool_limit_for_round(max(0, int(tool_round_index or 0)) + 1)
        if isinstance(payload, dict):
            tool_calls_payload = payload.get("tool_calls")
            if isinstance(tool_calls_payload, list):
                tool_calls: List[Dict[str, Any]] = []
                for slot_index, row in enumerate(tool_calls_payload, start=1):
                    tool_call = self._normalize_tool_call_payload(
                        row,
                        call_index=call_index,
                        slot_index=slot_index,
                    )
                    if tool_call is not None:
                        tool_calls.append(tool_call)
                if tool_calls:
                    if len(tool_calls) > allowed_tool_calls:
                        logger.warning(
                            "Manual tool response exceeded current round limit: got=%d allowed=%d",
                            len(tool_calls),
                            allowed_tool_calls,
                        )
                        tool_calls = tool_calls[:allowed_tool_calls]
                    return AIMessage(content="", tool_calls=tool_calls)
            tool_call = self._normalize_tool_call_payload(
                payload,
                call_index=call_index,
                slot_index=1,
            )
            if tool_call is not None:
                return AIMessage(content="", tool_calls=[tool_call])
            final_answer = payload.get("final_answer")
            if final_answer is None and "answer" in payload and not tool_call:
                final_answer = payload.get("answer")
            if final_answer is not None:
                return AIMessage(content=_content_to_text(final_answer))
        elif isinstance(payload, list):
            tool_calls = []
            for slot_index, row in enumerate(payload, start=1):
                tool_call = self._normalize_tool_call_payload(
                    row,
                    call_index=call_index,
                    slot_index=slot_index,
                )
                if tool_call is not None:
                    tool_calls.append(tool_call)
            if tool_calls:
                if len(tool_calls) > allowed_tool_calls:
                    logger.warning(
                        "Manual tool response exceeded current round limit: got=%d allowed=%d",
                        len(tool_calls),
                        allowed_tool_calls,
                    )
                    tool_calls = tool_calls[:allowed_tool_calls]
                return AIMessage(content="", tool_calls=tool_calls)
        return AIMessage(content=text)

    def _model_node(self, state: _AgentState) -> Dict[str, Any]:
        messages = list(state.get("messages") or [])
        remaining = int(state.get("remaining_llm_calls", 0) or 0)
        remaining_tool_rounds = int(state.get("remaining_tool_rounds", 0) or 0)
        tool_round_index = int(state.get("tool_round_index", 0) or 0)
        if remaining <= 0:
            return {
                "messages": messages,
                "remaining_llm_calls": 0,
                "remaining_tool_rounds": max(0, remaining_tool_rounds),
                "tool_round_index": max(0, tool_round_index),
            }
        if self._manual_tool_calling_enabled:
            manual_messages = self._build_manual_tool_messages(
                messages,
                remaining_tool_rounds=remaining_tool_rounds,
                tool_round_index=tool_round_index,
            )
            raw_response = self.llm.invoke(manual_messages)
            response = self._coerce_manual_tool_response(
                raw_response,
                call_index=remaining,
                tool_round_index=tool_round_index,
            )
            if remaining_tool_rounds <= 0 and isinstance(response, AIMessage) and list(getattr(response, "tool_calls", []) or []):
                response = AIMessage(
                    content=json.dumps(
                        {
                            "final_answer": "I have exhausted the tool-round budget for this question and must answer from the evidence already retrieved."
                        },
                        ensure_ascii=False,
                    )
                )
                response = self._coerce_manual_tool_response(
                    response,
                    call_index=remaining,
                    tool_round_index=tool_round_index,
                )
        else:
            response = self.llm.invoke(messages)
        return {
            "messages": messages + [response],
            "remaining_llm_calls": remaining - 1,
            "remaining_tool_rounds": max(0, remaining_tool_rounds),
            "tool_round_index": max(0, tool_round_index),
        }

    def _route_after_model(self, state: _AgentState) -> str:
        messages = list(state.get("messages") or [])
        if not messages:
            return "end"
        last = messages[-1]
        remaining_tool_rounds = int(state.get("remaining_tool_rounds", 0) or 0)
        if isinstance(last, AIMessage) and list(getattr(last, "tool_calls", []) or []) and remaining_tool_rounds > 0:
            return "tools"
        return "end"

    def _tools_node(self, state: _AgentState) -> Dict[str, Any]:
        messages = list(state.get("messages") or [])
        if not messages:
            return {
                "messages": messages,
                "remaining_llm_calls": int(state.get("remaining_llm_calls", 0) or 0),
                "remaining_tool_rounds": int(state.get("remaining_tool_rounds", 0) or 0),
                "tool_round_index": int(state.get("tool_round_index", 0) or 0),
            }
        last = messages[-1]
        if not isinstance(last, AIMessage):
            return {
                "messages": messages,
                "remaining_llm_calls": int(state.get("remaining_llm_calls", 0) or 0),
                "remaining_tool_rounds": int(state.get("remaining_tool_rounds", 0) or 0),
                "tool_round_index": int(state.get("tool_round_index", 0) or 0),
            }

        tool_calls = list(getattr(last, "tool_calls", []) or [])
        tool_results = self._execute_tool_calls_batch(tool_calls=tool_calls)
        tool_messages: List[ToolMessage] = []
        for tool_call, result in zip(tool_calls, tool_results):
            tool_name = str(tool_call.get("name") or "").strip()
            tool_call_id = str(tool_call.get("id") or tool_name or "tool_call")
            tool_messages.append(
                ToolMessage(
                    content=_content_to_text(result.get("output", "")),
                    name=tool_name or None,
                    tool_call_id=tool_call_id,
                    status=str(result.get("status") or "error"),
                )
            )

        return {
            "messages": messages + tool_messages,
            "remaining_llm_calls": int(state.get("remaining_llm_calls", 0) or 0),
            "remaining_tool_rounds": max(0, int(state.get("remaining_tool_rounds", 0) or 0) - (1 if tool_messages else 0)),
            "tool_round_index": int(state.get("tool_round_index", 0) or 0) + (1 if tool_messages else 0),
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
                "remaining_tool_rounds": self.max_tool_rounds_per_run,
                "tool_round_index": 0,
            }
        )
        return self._to_legacy_responses(list(result.get("messages") or []))


class S4LangGraphAssistantRuntime(LangGraphAssistantRuntime):
    """Structured LangGraph runtime with planner/evaluator/finalizer and evidence pooling."""

    def __init__(
        self,
        *,
        function_list: Sequence[Any],
        llm: Any,
        system_message: str,
        rag_cfg: Optional[Dict[str, Any]] = None,
    ) -> None:
        rag_cfg = dict(rag_cfg or {})
        self.max_evidence_items_for_prompt = max(4, int(rag_cfg.get("s4_max_evidence_items_for_prompt", 12) or 12))
        self.max_entities_for_prompt = max(4, int(rag_cfg.get("s4_max_entities_for_prompt", 10) or 10))
        self.max_stagnation_rounds = max(1, int(rag_cfg.get("s4_max_stagnation_rounds", 1) or 1))
        self.evaluator_finalize_confidence = float(rag_cfg.get("s4_evaluator_finalize_confidence", 0.78) or 0.78)
        self.s4_candidate_tool_window = max(4, int(rag_cfg.get("s4_candidate_tool_window", 7) or 7))
        super().__init__(
            function_list=function_list,
            llm=llm,
            system_message=system_message,
            rag_cfg=rag_cfg,
        )

    @staticmethod
    def _question_profile(question_text: str) -> Dict[str, Any]:
        parsed = _parse_mcq_question(question_text)
        stem = str(parsed.get("question_stem", "") or question_text or "").strip()
        lowered = stem.lower()
        is_mcq = bool(parsed.get("choice_order"))
        chronology = any(
            token in lowered
            for token in [
                "before",
                "after",
                "first",
                "last",
                "earlier",
                "later",
                "timeline",
                "chronology",
                "previously",
                "subsequently",
                "when",
            ]
        )
        narrative = any(
            token in lowered
            for token in [
                "why",
                "reason",
                "motive",
                "attitude",
                "feel",
                "emotion",
                "warning",
                "imply",
                "implication",
                "suggest",
                "theme",
                "lesson",
                "relationship",
                "dynamic",
                "conflict",
                "decide",
                "calm",
                "afraid",
            ]
        )
        entity = any(
            token in lowered
            for token in [
                "who",
                "whose",
                "character",
                "person",
                "between",
                "relation",
                "interact",
                "interaction",
            ]
        )
        return {
            "parsed": parsed,
            "stem": stem,
            "is_mcq": is_mcq,
            "needs_chronology": chronology,
            "needs_narrative": narrative,
            "needs_entity_grounding": entity,
        }

    @staticmethod
    def _s4_plannable_tool_names() -> set[str]:
        return {
            "bm25_search_docs",
            "section_evidence_search",
            "vdb_search_sentences",
            "vdb_get_docs_by_document_ids",
            "search_sections",
            "choice_grounded_evidence_search",
            "narrative_hierarchical_search",
            "entity_event_trace_search",
            "fact_timeline_resolution_search",
            "retrieve_entity_by_name",
            "get_entity_sections",
            "search_interactions",
            "search_related_entities",
            "get_relations_between_entities",
            "lookup_titles_by_document_ids",
            "lookup_document_ids_by_title",
            "vdb_search_hierdocs",
        }

    def _tool_usage_stats(self, state: _S4AgentState) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for row in list(state.get("evidence_pool") or []):
            if not isinstance(row, dict):
                continue
            name = str(row.get("source_tool") or "").strip()
            if not name:
                continue
            counts[name] = counts.get(name, 0) + 1
        return counts

    @staticmethod
    def _extract_choice_label_from_text(text: Any, valid_labels: Sequence[str]) -> str:
        raw = str(text or "").strip()
        if not raw:
            return ""
        valid = {str(label or "").strip().upper() for label in valid_labels if str(label or "").strip()}
        payload = _extract_first_json_payload(raw)
        if isinstance(payload, dict):
            for key in ("selected_label", "answer_choice", "choice", "option", "label", "predicted_choice"):
                value = str(payload.get(key) or "").strip().upper()
                if value in valid:
                    return value
        patterns = [
            r"\[Recommended Choice\]\s*([A-Z])\b",
            r"\[Suggested Choice\]\s*([A-Z])\b",
            r"selected[_ ]label\s*[:=]\s*['\"]?([A-Z])['\"]?",
            r"selected\s*[:=]\s*['\"]?([A-Z])['\"]?",
            r"recommended[_ ]choice[^A-Z]*([A-Z])\b",
            r"answer_choice\s*[:=]\s*['\"]?([A-Z])['\"]?",
            r"choice\s*[:=]\s*['\"]?([A-Z])['\"]?",
            r"option\s*[:=]\s*['\"]?([A-Z])['\"]?",
            r"\b(?:best|correct|final)\s+(?:choice|answer)\s*[:=]?\s*([A-Z])\b",
            r"\b(?:choose|pick|select)\s+([A-Z])\b",
            r"^\s*([A-Z])[\.\):\-\s]",
        ]
        for pattern in patterns:
            match = re.search(pattern, raw, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL)
            if not match:
                continue
            value = str(match.group(1) or "").strip().upper()
            if value in valid:
                return value
        if len(raw) == 1 and raw.upper() in valid:
            return raw.upper()
        return ""

    @staticmethod
    def _option_keywords(text: Any) -> List[str]:
        tokens = re.findall(r"[A-Za-z][A-Za-z0-9_-]+", str(text or "").lower())
        out: List[str] = []
        seen: set[str] = set()
        for token in tokens:
            if len(token) < 3 or token in _OPTION_STOPWORDS:
                continue
            if token in seen:
                continue
            seen.add(token)
            out.append(token)
        return out

    def _choice_overlap_score(self, *, choice_text: str, evidence_text: str) -> float:
        keywords = self._option_keywords(choice_text)
        if not keywords:
            return 0.0
        lowered = str(evidence_text or "").lower()
        hits = sum(1 for token in keywords if token in lowered)
        if hits <= 0:
            return 0.0
        return min(1.0, hits / max(1, len(keywords)))

    def _build_option_evidence_board(self, state: _S4AgentState) -> Dict[str, Any]:
        question_text = _content_to_text(self._extract_latest_user_text(state))
        profile = self._question_profile(question_text)
        parsed = profile.get("parsed") if isinstance(profile.get("parsed"), dict) else {}
        choice_order = [str(label).strip().upper() for label in (parsed.get("choice_order") or []) if str(label).strip()]
        choices = parsed.get("choices") if isinstance(parsed.get("choices"), dict) else {}
        if not choice_order or not choices:
            return {
                "is_mcq": False,
                "choice_order": [],
                "top_label": "",
                "second_label": "",
                "margin": 0.0,
                "lines": [],
            }

        board: Dict[str, Dict[str, Any]] = {
            label: {
                "label": label,
                "choice_text": str(choices.get(label, "") or "").strip(),
                "score": 0.0,
                "mention_count": 0,
                "snippets": [],
                "recommended_by": [],
                "tools": [],
            }
            for label in choice_order
        }

        for row in list(state.get("evidence_pool") or []):
            if not isinstance(row, dict):
                continue
            tool_name = str(row.get("source_tool") or "").strip()
            if not tool_name:
                continue
            content = _content_to_text(row.get("content") or "")
            if not content:
                continue
            importance = int(row.get("importance", 0) or 0)
            option_labels = {
                str(label).strip().upper()
                for label in (row.get("option_labels") or [])
                if str(label).strip()
            }
            recommended_label = self._extract_choice_label_from_text(content, choice_order)
            content_snippet = _clip_text(content, limit=220)
            for label in choice_order:
                score_delta = 0.0
                overlap = self._choice_overlap_score(choice_text=str(choices.get(label, "") or ""), evidence_text=content)
                direct_label = label in option_labels
                if recommended_label == label:
                    score_delta += 2.8 + 0.15 * importance
                elif recommended_label and recommended_label != label and tool_name == "choice_grounded_evidence_search":
                    score_delta -= 0.15
                if direct_label:
                    score_delta += 0.9 + 0.1 * importance
                if overlap >= 0.66:
                    score_delta += 1.35
                elif overlap >= 0.40:
                    score_delta += 0.85
                elif overlap >= 0.22:
                    score_delta += 0.35
                if tool_name == "choice_grounded_evidence_search" and (recommended_label == label or direct_label or overlap >= 0.22):
                    score_delta += 0.8
                if tool_name in {"narrative_hierarchical_search", "entity_event_trace_search", "fact_timeline_resolution_search"} and overlap >= 0.22:
                    score_delta += 0.5
                if score_delta <= 0.0:
                    continue
                item = board[label]
                item["score"] = float(item.get("score", 0.0) or 0.0) + score_delta
                item["mention_count"] = int(item.get("mention_count", 0) or 0) + 1
                tools = list(item.get("tools") or [])
                if tool_name not in tools:
                    tools.append(tool_name)
                item["tools"] = tools
                if recommended_label == label:
                    recommended_by = list(item.get("recommended_by") or [])
                    if tool_name not in recommended_by:
                        recommended_by.append(tool_name)
                    item["recommended_by"] = recommended_by
                snippets = list(item.get("snippets") or [])
                if len(snippets) < 2:
                    snippets.append(f"{tool_name}: {content_snippet}")
                item["snippets"] = snippets

        ranked = sorted(
            board.values(),
            key=lambda item: (
                float(item.get("score", 0.0) or 0.0),
                int(item.get("mention_count", 0) or 0),
                str(item.get("label") or ""),
            ),
            reverse=True,
        )
        top_label = str(ranked[0].get("label") or "").strip() if ranked else ""
        second_label = str(ranked[1].get("label") or "").strip() if len(ranked) > 1 else ""
        top_score = float(ranked[0].get("score", 0.0) or 0.0) if ranked else 0.0
        second_score = float(ranked[1].get("score", 0.0) or 0.0) if len(ranked) > 1 else 0.0
        lines: List[str] = []
        for item in ranked:
            snippets = list(item.get("snippets") or [])
            snippet_text = " | ".join(snippets[:2]) if snippets else "no strong evidence yet"
            tools_text = ",".join(str(x).strip() for x in (item.get("tools") or []) if str(x).strip()) or "-"
            lines.append(
                f"{str(item.get('label') or '').strip()}: "
                f"score={float(item.get('score', 0.0) or 0.0):.2f}; "
                f"mentions={int(item.get('mention_count', 0) or 0)}; "
                f"tools={tools_text}; "
                f"snippets={snippet_text}"
            )
        return {
            "is_mcq": True,
            "choice_order": choice_order,
            "choices": choices,
            "board": board,
            "ranked": ranked,
            "top_label": top_label,
            "second_label": second_label,
            "top_score": top_score,
            "second_score": second_score,
            "margin": top_score - second_score,
            "lines": lines,
        }

    def _summarize_option_board(self, state: _S4AgentState) -> str:
        board = self._build_option_evidence_board(state)
        if not bool(board.get("is_mcq")):
            return "(not an MCQ)"
        lines = list(board.get("lines") or [])
        top_label = str(board.get("top_label") or "").strip()
        second_label = str(board.get("second_label") or "").strip()
        margin = float(board.get("margin", 0.0) or 0.0)
        prefix = f"Top gap: {top_label or '?'} vs {second_label or '?'} margin={margin:.2f}"
        return "\n".join([prefix] + lines) if lines else prefix

    @staticmethod
    def _tool_family(tool_name: str) -> str:
        name = str(tool_name or "").strip()
        if name in {"bm25_search_docs", "section_evidence_search", "vdb_search_sentences", "vdb_search_hierdocs"}:
            return "local_evidence"
        if name in {"choice_grounded_evidence_search"}:
            return "choice_evidence"
        if name in {"narrative_hierarchical_search", "entity_event_trace_search"}:
            return "narrative"
        if name in {"fact_timeline_resolution_search"}:
            return "timeline"
        if name in {"retrieve_entity_by_name", "get_entity_sections", "search_interactions", "search_related_entities", "get_relations_between_entities"}:
            return "entity_relation"
        if name in {"vdb_get_docs_by_document_ids", "search_sections", "lookup_titles_by_document_ids", "lookup_document_ids_by_title"}:
            return "followup_lookup"
        return "other"

    def _backbone_tool_calls_for_state(self, state: _S4AgentState) -> List[Dict[str, Any]]:
        if int(state.get("tool_round_index", 0) or 0) != 0:
            return []
        if list(state.get("evidence_pool") or []):
            return []
        if int(state.get("remaining_tool_rounds", 0) or 0) <= 0:
            return []
        question_text = _content_to_text(self._extract_latest_user_text(state))
        profile = self._question_profile(question_text)
        parsed = profile.get("parsed") if isinstance(profile.get("parsed"), dict) else {}
        stem = str(parsed.get("question_stem") or question_text or "").strip()
        calls: List[Dict[str, Any]] = []

        def add(tool_name: str, args: Dict[str, Any]) -> None:
            if tool_name not in self.tool_map:
                return
            call_index = len(calls) + 1
            calls.append(
                {
                    "id": f"s4_backbone_call_{call_index}_{tool_name}",
                    "name": tool_name,
                    "args": args,
                    "type": "tool_call",
                }
            )

        add(
            "section_evidence_search",
            {
                "query": question_text if profile["is_mcq"] else stem,
                "section_top_k": 5,
                "max_length": 240,
            },
        )
        if profile["is_mcq"]:
            add(
                "choice_grounded_evidence_search",
                {
                    "query": question_text,
                    "section_top_k": 3,
                    "document_top_k": 2,
                    "sentence_top_k": 3,
                    "max_length": 180,
                    "use_llm_judge": False,
                },
            )

        if profile["needs_chronology"]:
            add(
                "fact_timeline_resolution_search",
                {
                    "query": question_text if profile["is_mcq"] else stem,
                    "section_top_k": 4,
                    "sentence_top_k": 5,
                    "document_top_k": 4,
                    "event_top_k": 4,
                    "max_length": 180,
                    "use_llm_choice_judge": False,
                },
            )
        elif profile["needs_narrative"]:
            add(
                "narrative_hierarchical_search",
                {
                    "query": stem,
                    "storyline_top_k": 3,
                    "episode_top_k": 4,
                    "event_top_k": 6,
                    "document_top_k": 4,
                    "max_evidence_length": 180,
                },
            )
        elif profile["needs_entity_grounding"]:
            add(
                "retrieve_entity_by_name",
                {
                    "query": stem,
                    "top_k": 5,
                    "resolve_source_documents": True,
                },
            )
        else:
            add(
                "bm25_search_docs",
                {
                    "query": stem,
                    "k": 6,
                },
            )
        return self._prune_tool_calls_for_state(
            calls,
            state=state,
            allowed_tool_names=self._candidate_tool_names_for_state(state),
        )

    def _candidate_tool_names_for_state(self, state: _S4AgentState) -> List[str]:
        question_text = _content_to_text(self._extract_latest_user_text(state))
        profile = self._question_profile(question_text)
        boosts = heuristic_tool_boosts(question_text)
        usage = self._tool_usage_stats(state)
        allowed = self._s4_plannable_tool_names()
        scores: Dict[str, float] = {}
        current_round = int(state.get("tool_round_index", 0) or 0) + 1
        option_board = self._build_option_evidence_board(state)
        unresolved_mcq = bool(
            profile["is_mcq"]
            and option_board.get("top_label")
            and float(option_board.get("margin", 0.0) or 0.0) < 1.35
        )

        for name in self.tool_map:
            if name not in allowed:
                continue
            stage = tool_stage(name)
            score = 0.0
            if stage == "core":
                score += 2.0
            elif stage == "extended":
                score += 1.0
            score += float(boosts.get(name, 0.0) or 0.0)

            if name == "choice_grounded_evidence_search" and profile["is_mcq"]:
                score += 3.6
            if name in {"bm25_search_docs", "section_evidence_search"}:
                score += 1.5
            if name == "vdb_search_sentences":
                score += 1.0
            if name in {"narrative_hierarchical_search", "entity_event_trace_search"} and profile["needs_narrative"]:
                score += 1.8
            if name == "fact_timeline_resolution_search" and profile["needs_chronology"]:
                score += 2.2
            if name in {"retrieve_entity_by_name", "get_entity_sections", "search_interactions"} and profile["needs_entity_grounding"]:
                score += 1.4

            used = int(usage.get(name, 0) or 0)
            if used:
                # Repeated direct-search calls are usually low value in s4.
                score -= 0.9 * used
                if current_round >= 2 and name in {"bm25_search_docs", "section_evidence_search", "choice_grounded_evidence_search"}:
                    score -= 0.6

            if current_round >= 2 and used == 0 and name in {
                "narrative_hierarchical_search",
                "entity_event_trace_search",
                "fact_timeline_resolution_search",
                "vdb_get_docs_by_document_ids",
            }:
                score += 0.8

            if current_round >= 2 and unresolved_mcq:
                if profile["needs_narrative"] and name in {"narrative_hierarchical_search", "entity_event_trace_search"}:
                    score += 1.4
                if profile["needs_chronology"] and name == "fact_timeline_resolution_search":
                    score += 1.6
                if not profile["needs_narrative"] and not profile["needs_chronology"] and name in {"vdb_search_sentences", "bm25_search_docs"}:
                    score += 0.9

            scores[name] = score

        ranked = [name for name, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0])) if scores.get(name, 0.0) > 0.0]
        shortlist = ranked[: self.s4_candidate_tool_window]

        def ensure(name: str) -> None:
            if name in self.tool_map and name in allowed and name not in shortlist:
                shortlist.append(name)

        ensure("bm25_search_docs")
        ensure("section_evidence_search")
        if profile["is_mcq"]:
            ensure("choice_grounded_evidence_search")
        if profile["needs_narrative"]:
            ensure("narrative_hierarchical_search")
        if profile["needs_chronology"]:
            ensure("fact_timeline_resolution_search")

        return shortlist[: self.s4_candidate_tool_window + 2]

    def _tool_line(self, tool_name: str) -> str:
        tool = self.tool_map.get(tool_name)
        if tool is None:
            return ""
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
        line = f"- {tool_name}: {description}" if description else f"- {tool_name}"
        if param_parts:
            line += f" | params: {'; '.join(param_parts)}"
        return line

    def _tool_call_signature(self, *, tool_name: str, args: Any) -> str:
        try:
            args_text = json.dumps(args if args is not None else {}, ensure_ascii=False, sort_keys=True)
        except Exception:
            args_text = _serialize_tool_args(args)
        return _stable_digest(f"{tool_name}::{args_text}")

    def _build_graph(self):
        graph = StateGraph(_S4AgentState)
        graph.add_node("planner", self._planner_node)
        graph.add_node("tools", self._s4_tools_node)
        graph.add_node("evaluator", self._evaluator_node)
        graph.add_node("finalizer", self._finalizer_node)
        graph.set_entry_point("planner")
        graph.add_conditional_edges(
            "planner",
            self._route_after_planner,
            {
                "tools": "tools",
                "finalizer": "finalizer",
            },
        )
        graph.add_edge("tools", "evaluator")
        graph.add_conditional_edges(
            "evaluator",
            self._route_after_evaluator,
            {
                "planner": "planner",
                "finalizer": "finalizer",
            },
        )
        graph.add_edge("finalizer", END)
        return graph.compile()

    def _build_manual_tool_system_message(self) -> str:
        lines: List[str] = []
        if self.system_message:
            lines.append(self.system_message)
        lines.extend(
            [
                "You are a retrieval planner operating inside a multi-stage LangGraph agent.",
                "Your job is to gather evidence in a small number of rounds, not to answer immediately.",
                "You must decide which tools to call first, then let a separate evaluator decide whether more evidence is needed.",
                "Use JSON-only tool calling when selecting tools.",
                "Allowed planner outputs:",
                '{"tool_name":"<tool name>","tool_arguments":{"arg":"value"}}',
                '{"tool_calls":[{"tool_name":"<tool name>","tool_arguments":{"arg":"value"}}]}',
                '{"final_answer":"<answer>"}',
                f"Across the whole question, use at most {self.max_tool_rounds_per_run} tool rounds.",
                f"In the first tool round, you may call up to {self.first_round_max_tool_calls} complementary tools.",
                f"After the first round, call at most {self.followup_round_max_tool_calls} tool per round.",
                "Prefer diverse evidence in the first round and targeted follow-up afterward.",
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

    @staticmethod
    def _tool_importance(tool_name: str) -> int:
        name = str(tool_name or "").strip()
        high = {
            "choice_grounded_evidence_search",
            "section_evidence_search",
            "narrative_hierarchical_search",
            "entity_event_trace_search",
            "fact_timeline_resolution_search",
        }
        medium = {
            "bm25_search_docs",
            "vdb_search_sentences",
            "vdb_get_docs_by_document_ids",
            "retrieve_entity_by_name",
            "search_interactions",
        }
        if name in high:
            return 5
        if name in medium:
            return 4
        return 3

    @staticmethod
    def _parse_json_if_possible(text: str) -> Any:
        raw = str(text or "").strip()
        if not raw:
            return None
        try:
            return json.loads(raw)
        except Exception:
            return None

    def _normalize_evidence_item(self, *, tool_name: str, tool_args: Any, content: Any) -> Dict[str, Any]:
        text = _clip_text(content, limit=2500)
        lowered = text.lower()
        parsed = self._parse_json_if_possible(text)
        option_labels: List[str] = []
        if isinstance(parsed, dict):
            for key in ("answer_choice", "choice", "option", "label", "predicted_choice"):
                value = str(parsed.get(key) or "").strip().upper()
                if re.fullmatch(r"[A-Z]", value) and value not in option_labels:
                    option_labels.append(value)
        for label in re.findall(r"\b([A-D])\b", text):
            if label not in option_labels:
                option_labels.append(label)
        return {
            "source_tool": str(tool_name or "").strip(),
            "content": text,
            "timestamp": time.time(),
            "importance": self._tool_importance(tool_name),
            "call_signature": self._tool_call_signature(tool_name=str(tool_name or "").strip(), args=tool_args),
            "content_signature": _stable_digest(text[:1600]),
            "has_error": lowered.startswith("{\"error\"") or "\"error\"" in lowered[:200],
            "option_labels": option_labels[:4],
        }

    def _extract_entities_from_tool_output(self, *, tool_name: str, content: Any) -> List[Dict[str, Any]]:
        name = str(tool_name or "").strip()
        if name not in {
            "retrieve_entity_by_name",
            "search_related_entities",
            "get_entity_sections",
            "get_relations_between_entities",
        }:
            return []
        parsed = self._parse_json_if_possible(_content_to_text(content))
        if parsed is None:
            return []
        rows = parsed if isinstance(parsed, list) else [parsed]
        out: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for row in rows:
            if not isinstance(row, dict):
                continue
            entity_id = str(
                row.get("entity_id")
                or row.get("id")
                or row.get("node_id")
                or ""
            ).strip()
            entity_name = str(
                row.get("entity_name")
                or row.get("name")
                or row.get("label")
                or ""
            ).strip()
            entity_type = str(
                row.get("entity_type")
                or row.get("type")
                or ""
            ).strip()
            key = f"{entity_id}::{entity_name}".strip(":")
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(
                {
                    "entity_id": entity_id,
                    "entity_name": entity_name,
                    "entity_type": entity_type,
                    "source_tool": name,
                }
            )
        return out

    def _dedup_entities(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for row in items or []:
            if not isinstance(row, dict):
                continue
            key = f"{str(row.get('entity_id') or '').strip()}::{str(row.get('entity_name') or '').strip()}".strip(":")
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(dict(row))
        return out

    def _summarize_evidence_pool(self, evidence_pool: List[Dict[str, Any]]) -> str:
        rows = sorted(
            [row for row in (evidence_pool or []) if isinstance(row, dict)],
            key=lambda row: (
                int(row.get("importance", 0) or 0),
                float(row.get("timestamp", 0.0) or 0.0),
            ),
            reverse=True,
        )
        lines: List[str] = []
        for idx, row in enumerate(rows[: self.max_evidence_items_for_prompt], start=1):
            option_text = ""
            option_labels = [str(x).strip() for x in (row.get("option_labels") or []) if str(x).strip()]
            if option_labels:
                option_text = f" options={','.join(option_labels)}"
            lines.append(
                f"[{idx}] tool={str(row.get('source_tool','') or '').strip()} "
                f"importance={int(row.get('importance', 0) or 0)}{option_text}\n{_clip_text(row.get('content', ''), limit=1400)}"
            )
        return "\n\n".join(lines) if lines else "(no evidence yet)"

    def _prune_tool_calls_for_state(
        self,
        tool_calls: List[Dict[str, Any]],
        *,
        state: _S4AgentState,
        allowed_tool_names: List[str],
    ) -> List[Dict[str, Any]]:
        allowed = set(allowed_tool_names or [])
        prior_signatures = {
            str(row.get("call_signature") or "").strip()
            for row in list(state.get("evidence_pool") or [])
            if isinstance(row, dict) and str(row.get("call_signature") or "").strip()
        }
        prior_counts = self._tool_usage_stats(state)
        prior_family_counts: Dict[str, int] = {}
        for tool_name, count in prior_counts.items():
            family = self._tool_family(tool_name)
            prior_family_counts[family] = prior_family_counts.get(family, 0) + int(count or 0)
        kept: List[Dict[str, Any]] = []
        seen_this_round: set[str] = set()
        seen_tool_names_this_round: set[str] = set()
        seen_family_counts: Dict[str, int] = {}
        current_round = int(state.get("tool_round_index", 0) or 0) + 1
        for row in list(tool_calls or []):
            tool_name = str(row.get("name") or "").strip()
            if not tool_name or (allowed and tool_name not in allowed):
                continue
            family = self._tool_family(tool_name)
            signature = self._tool_call_signature(tool_name=tool_name, args=row.get("args", {}))
            if signature in prior_signatures or signature in seen_this_round:
                continue
            if tool_name in seen_tool_names_this_round and tool_name in {
                "bm25_search_docs",
                "section_evidence_search",
                "choice_grounded_evidence_search",
                "search_sections",
                "narrative_hierarchical_search",
            }:
                continue
            if prior_counts.get(tool_name, 0) >= 1 and tool_name in {"choice_grounded_evidence_search", "fact_timeline_resolution_search"}:
                continue
            if prior_counts.get(tool_name, 0) >= 2 and tool_name in {
                "bm25_search_docs",
                "section_evidence_search",
                "choice_grounded_evidence_search",
                "search_sections",
            }:
                continue
            family_cap = 2 if family == "local_evidence" and current_round <= 1 else 1
            if seen_family_counts.get(family, 0) >= family_cap:
                continue
            if family == "choice_evidence" and prior_family_counts.get(family, 0) >= 1:
                continue
            if family in {"narrative", "timeline"} and prior_family_counts.get(family, 0) >= 1 and current_round >= 2:
                continue
            kept.append(row)
            seen_this_round.add(signature)
            seen_tool_names_this_round.add(tool_name)
            seen_family_counts[family] = seen_family_counts.get(family, 0) + 1
        return kept

    def _summarize_entities(self, entities_found: List[Dict[str, Any]]) -> str:
        rows = [row for row in (entities_found or []) if isinstance(row, dict)]
        if not rows:
            return "(none)"
        lines: List[str] = []
        for row in rows[: self.max_entities_for_prompt]:
            lines.append(
                f"- {str(row.get('entity_name','') or '').strip()} "
                f"(id={str(row.get('entity_id','') or '').strip()}, type={str(row.get('entity_type','') or '').strip()}, source={str(row.get('source_tool','') or '').strip()})"
            )
        return "\n".join(lines)

    def _planner_messages(self, state: _S4AgentState) -> List[BaseMessage]:
        evidence_summary = self._summarize_evidence_pool(list(state.get("evidence_pool") or []))
        entities_summary = self._summarize_entities(list(state.get("entities_found") or []))
        option_board_summary = self._summarize_option_board(state)
        question_text = _content_to_text(self._extract_latest_user_text(state))
        profile = self._question_profile(question_text)
        option_board = self._build_option_evidence_board(state)
        next_round = int(state.get("tool_round_index", 0) or 0) + 1
        tool_limit = self._tool_limit_for_round(next_round)
        candidate_tool_names = self._candidate_tool_names_for_state(state)
        used_tools = sorted(self._tool_usage_stats(state).items(), key=lambda item: (-item[1], item[0]))
        used_tools_text = ", ".join(f"{name} x{count}" for name, count in used_tools[:8]) if used_tools else "(none)"
        planner_lines: List[str] = [
            "You are a retrieval planner inside a planner-tools-evaluator-finalizer loop.",
            "Return JSON only.",
            "Use only the candidate tools listed below.",
            '{"tool_calls":[{"tool_name":"<tool name>","tool_arguments":{"arg":"value"}}]}',
            '{"final_answer":"<short grounded answer intent>"}',
            f"In this round you may call at most {tool_limit} tool(s).",
            "Do not repeat the same tool with near-identical arguments.",
            "Prefer one high-value verification step over another broad search.",
        ]
        if profile["is_mcq"]:
            planner_lines.extend(
                [
                    "This is a multiple-choice question.",
                    "Prioritize tools that distinguish the options.",
                    "If `choice_grounded_evidence_search` is available and unused, strongly consider it early.",
                ]
            )
            if next_round >= 2 and str(option_board.get("top_label") or "").strip():
                planner_lines.append(
                    f"Current unresolved comparison: {str(option_board.get('top_label') or '').strip()} vs {str(option_board.get('second_label') or '').strip() or 'others'} "
                    f"(margin={float(option_board.get('margin', 0.0) or 0.0):.2f})."
                )
                planner_lines.append("In later rounds, choose one tool that best separates the top competing options instead of repeating broad retrieval.")
        if profile["needs_narrative"]:
            planner_lines.append("Because the question asks about motive, implication, attitude, or narrative meaning, keep at least one narrative-capable tool in consideration.")
        planner_lines.extend(["", "Candidate tools:"])
        for name in candidate_tool_names:
            line = self._tool_line(name)
            if line:
                planner_lines.append(line)
        human_prompt = "\n\n".join(
            [
                f"Question:\n{_clip_text(question_text, limit=2500)}",
                f"Current evidence summary:\n{evidence_summary}",
                f"Option evidence board:\n{option_board_summary}",
                f"Resolved entities:\n{entities_summary}",
                f"Tools already used:\n{used_tools_text}",
                f"Planner notes from previous rounds:\n{str(state.get('plan_notes', '') or '').strip() or '(none)'}",
                f"Evaluator notes:\n{str(state.get('evaluation_notes', '') or '').strip() or '(none)'}",
                f"Remaining tool rounds: {int(state.get('remaining_tool_rounds', 0) or 0)}",
                f"Current round index: {next_round}",
                f"Tool limit for this round: {tool_limit}",
                "Decide whether to gather more evidence or stop and defer to the finalizer.",
                "If more evidence is needed, return JSON tool calls only. Use a diverse evidence bundle in round 1 and a single targeted tool later.",
                "For MCQ follow-up rounds, explicitly target the strongest unresolved distinction between the current top options.",
                "If the evidence is already sufficient, return a short JSON object with `final_answer` summarizing the answer intent, not a long explanation.",
            ]
        ).strip()
        return [
            SystemMessage(content="\n".join(planner_lines).strip()),
            HumanMessage(content=human_prompt),
        ]

    @staticmethod
    def _extract_latest_user_text(state: _S4AgentState) -> str:
        for message in reversed(list(state.get("messages") or [])):
            if isinstance(message, HumanMessage):
                return _content_to_text(message.content)
        return ""

    def _planner_node(self, state: _S4AgentState) -> Dict[str, Any]:
        remaining = int(state.get("remaining_llm_calls", 0) or 0)
        if remaining <= 0:
            return {
                "messages": list(state.get("messages") or []),
                "remaining_llm_calls": 0,
                "remaining_tool_rounds": int(state.get("remaining_tool_rounds", 0) or 0),
                "tool_round_index": int(state.get("tool_round_index", 0) or 0),
                "evidence_pool": list(state.get("evidence_pool") or []),
                "entities_found": list(state.get("entities_found") or []),
                "plan_notes": str(state.get("plan_notes", "") or "").strip(),
                "evaluation_notes": str(state.get("evaluation_notes", "") or "").strip(),
                "next_step": "finalizer",
                "draft_answer": str(state.get("draft_answer", "") or "").strip(),
                "last_round_new_evidence_count": int(state.get("last_round_new_evidence_count", 0) or 0),
                "stagnation_count": int(state.get("stagnation_count", 0) or 0),
            }
        backbone_calls = self._backbone_tool_calls_for_state(state)
        if backbone_calls:
            plan_note = (
                f"round 1 backbone selected {len(backbone_calls)} tool(s): "
                + ", ".join(str(row.get("name") or "").strip() for row in backbone_calls)
            )
            response = AIMessage(content="", tool_calls=backbone_calls)
            return {
                "messages": list(state.get("messages") or []) + [response],
                "remaining_llm_calls": remaining,
                "remaining_tool_rounds": int(state.get("remaining_tool_rounds", 0) or 0),
                "tool_round_index": int(state.get("tool_round_index", 0) or 0),
                "evidence_pool": list(state.get("evidence_pool") or []),
                "entities_found": list(state.get("entities_found") or []),
                "plan_notes": plan_note,
                "evaluation_notes": str(state.get("evaluation_notes", "") or "").strip(),
                "next_step": "tools",
                "draft_answer": str(state.get("draft_answer", "") or "").strip(),
                "last_round_new_evidence_count": int(state.get("last_round_new_evidence_count", 0) or 0),
                "stagnation_count": int(state.get("stagnation_count", 0) or 0),
            }
        raw_response = self.llm.invoke(self._planner_messages(state))
        response = self._coerce_manual_tool_response(
            raw_response,
            call_index=remaining,
            tool_round_index=int(state.get("tool_round_index", 0) or 0),
        )
        tool_calls = self._prune_tool_calls_for_state(
            list(getattr(response, "tool_calls", []) or []),
            state=state,
            allowed_tool_names=self._candidate_tool_names_for_state(state),
        )
        draft_answer = str(state.get("draft_answer", "") or "").strip()
        next_step = "tools" if tool_calls and int(state.get("remaining_tool_rounds", 0) or 0) > 0 else "finalizer"
        if not tool_calls:
            draft_answer = _content_to_text(response.content).strip() or draft_answer
        else:
            response = AIMessage(content="", tool_calls=tool_calls)
        plan_note = _content_to_text(getattr(raw_response, "content", "")).strip()
        if tool_calls and not plan_note:
            plan_note = f"planner selected {len(tool_calls)} tool(s) for round {int(state.get('tool_round_index', 0) or 0) + 1}"
        return {
            "messages": list(state.get("messages") or []) + ([response] if tool_calls else []),
            "remaining_llm_calls": remaining - 1,
            "remaining_tool_rounds": int(state.get("remaining_tool_rounds", 0) or 0),
            "tool_round_index": int(state.get("tool_round_index", 0) or 0),
            "evidence_pool": list(state.get("evidence_pool") or []),
            "entities_found": list(state.get("entities_found") or []),
            "plan_notes": plan_note or str(state.get("plan_notes", "") or "").strip(),
            "evaluation_notes": str(state.get("evaluation_notes", "") or "").strip(),
            "next_step": next_step,
            "draft_answer": draft_answer,
            "last_round_new_evidence_count": int(state.get("last_round_new_evidence_count", 0) or 0),
            "stagnation_count": int(state.get("stagnation_count", 0) or 0),
        }

    def _route_after_planner(self, state: _S4AgentState) -> str:
        step = str(state.get("next_step", "") or "").strip().lower()
        return "tools" if step == "tools" else "finalizer"

    def _s4_tools_node(self, state: _S4AgentState) -> Dict[str, Any]:
        messages = list(state.get("messages") or [])
        if not messages or not isinstance(messages[-1], AIMessage):
            return {
                "messages": messages,
                "remaining_llm_calls": int(state.get("remaining_llm_calls", 0) or 0),
                "remaining_tool_rounds": int(state.get("remaining_tool_rounds", 0) or 0),
                "tool_round_index": int(state.get("tool_round_index", 0) or 0),
                "evidence_pool": list(state.get("evidence_pool") or []),
                "entities_found": list(state.get("entities_found") or []),
                "plan_notes": str(state.get("plan_notes", "") or "").strip(),
                "evaluation_notes": str(state.get("evaluation_notes", "") or "").strip(),
                "next_step": "finalizer",
                "draft_answer": str(state.get("draft_answer", "") or "").strip(),
                "last_round_new_evidence_count": 0,
                "stagnation_count": int(state.get("stagnation_count", 0) or 0) + 1,
            }
        last = messages[-1]
        evidence_pool = list(state.get("evidence_pool") or [])
        entities_found = list(state.get("entities_found") or [])
        tool_calls = list(getattr(last, "tool_calls", []) or [])
        tool_results = self._execute_tool_calls_batch(tool_calls=tool_calls)
        tool_messages: List[ToolMessage] = []
        seen_content_signatures = {
            str(row.get("content_signature") or "").strip()
            for row in evidence_pool
            if isinstance(row, dict) and str(row.get("content_signature") or "").strip()
        }
        new_evidence_count = 0
        for tool_call, result in zip(tool_calls, tool_results):
            tool_name = str(tool_call.get("name") or "").strip()
            tool_call_id = str(tool_call.get("id") or tool_name or "tool_call")
            args = tool_call.get("args", {})
            output = result.get("output", "")
            status = str(result.get("status") or "error")
            tool_messages.append(
                ToolMessage(
                    content=_content_to_text(output),
                    name=tool_name or None,
                    tool_call_id=tool_call_id,
                    status=status,
                )
            )
            evidence_item = self._normalize_evidence_item(tool_name=tool_name, tool_args=args, content=output)
            evidence_pool.append(evidence_item)
            entities_found.extend(self._extract_entities_from_tool_output(tool_name=tool_name, content=output))
            content_signature = str(evidence_item.get("content_signature") or "").strip()
            if content_signature and content_signature not in seen_content_signatures and not bool(evidence_item.get("has_error")):
                seen_content_signatures.add(content_signature)
                new_evidence_count += 1
        entities_found = self._dedup_entities(entities_found)
        stagnation_count = int(state.get("stagnation_count", 0) or 0)
        if new_evidence_count <= 0:
            stagnation_count += 1
        else:
            stagnation_count = 0
        return {
            "messages": messages + tool_messages,
            "remaining_llm_calls": int(state.get("remaining_llm_calls", 0) or 0),
            "remaining_tool_rounds": max(0, int(state.get("remaining_tool_rounds", 0) or 0) - (1 if tool_messages else 0)),
            "tool_round_index": int(state.get("tool_round_index", 0) or 0) + (1 if tool_messages else 0),
            "evidence_pool": evidence_pool,
            "entities_found": entities_found,
            "plan_notes": str(state.get("plan_notes", "") or "").strip(),
            "evaluation_notes": str(state.get("evaluation_notes", "") or "").strip(),
            "next_step": "evaluator",
            "draft_answer": str(state.get("draft_answer", "") or "").strip(),
            "last_round_new_evidence_count": new_evidence_count,
            "stagnation_count": stagnation_count,
        }

    def _evaluator_messages(self, state: _S4AgentState) -> List[BaseMessage]:
        evidence_summary = self._summarize_evidence_pool(list(state.get("evidence_pool") or []))
        entities_summary = self._summarize_entities(list(state.get("entities_found") or []))
        option_board_summary = self._summarize_option_board(state)
        question_text = _content_to_text(self._extract_latest_user_text(state))
        profile = self._question_profile(question_text)
        parsed = profile.get("parsed") if isinstance(profile.get("parsed"), dict) else {}
        choice_order = list(parsed.get("choice_order") or [])
        choices = parsed.get("choices") if isinstance(parsed.get("choices"), dict) else {}
        choice_block = "\n".join(
            f"{label}. {str(choices.get(label, '') or '').strip()}"
            for label in choice_order
            if str(choices.get(label, "") or "").strip()
        )
        human_prompt = "\n\n".join(
            [
                f"Question:\n{_clip_text(question_text, limit=2500)}",
                f"Choices:\n{choice_block or '(not an MCQ)'}",
                f"Evidence pool:\n{evidence_summary}",
                f"Option evidence board:\n{option_board_summary}",
                f"Resolved entities:\n{entities_summary}",
                f"Planner notes:\n{str(state.get('plan_notes', '') or '').strip() or '(none)'}",
                f"Tool rounds used so far: {int(state.get('tool_round_index', 0) or 0)}",
                f"Remaining tool rounds: {int(state.get('remaining_tool_rounds', 0) or 0)}",
                f"New evidence items from the last round: {int(state.get('last_round_new_evidence_count', 0) or 0)}",
                "Decide whether the evidence is sufficient to answer the question.",
                "Return JSON only with keys:",
                '{"next_step":"planner|finalizer","confidence":0.0,"missing_information":["..."],"contradictions":["..."],"leading_option":"","runner_up_option":"","option_margin":0.0,"evaluator_notes":"..."}',
                "Use `finalizer` if the evidence is already sufficient or if additional search is unlikely to help.",
            ]
        ).strip()
        return [
            SystemMessage(
                content=(
                    f"{self.system_message}\n\n"
                    "You are the evidence evaluator. Do not call tools. "
                    "Only judge coverage, contradictions, what is still missing, and whether one option is already clearly better supported."
                ).strip()
            ),
            HumanMessage(content=human_prompt),
        ]

    def _evaluator_node(self, state: _S4AgentState) -> Dict[str, Any]:
        remaining = int(state.get("remaining_llm_calls", 0) or 0)
        remaining_rounds = int(state.get("remaining_tool_rounds", 0) or 0)
        last_new = int(state.get("last_round_new_evidence_count", 0) or 0)
        stagnation_count = int(state.get("stagnation_count", 0) or 0)
        default_next = "planner" if remaining_rounds > 0 else "finalizer"
        evaluation_notes = str(state.get("evaluation_notes", "") or "").strip()
        question_text = _content_to_text(self._extract_latest_user_text(state))
        profile = self._question_profile(question_text)
        used_tools = set(self._tool_usage_stats(state).keys())
        option_board = self._build_option_evidence_board(state)
        margin = float(option_board.get("margin", 0.0) or 0.0)
        top_label = str(option_board.get("top_label") or "").strip()
        second_label = str(option_board.get("second_label") or "").strip()

        if last_new <= 0 or stagnation_count >= self.max_stagnation_rounds:
            default_next = "finalizer"
            evaluation_notes = evaluation_notes or "No meaningful new evidence was produced in the last round."

        if profile["is_mcq"] and top_label and margin >= 1.45:
            default_next = "finalizer"
            evaluation_notes = evaluation_notes or f"Option-level evidence already favors {top_label} by a usable margin ({margin:.2f})."

        if int(state.get("tool_round_index", 0) or 0) >= 2 and (
            {"choice_grounded_evidence_search", "narrative_hierarchical_search", "fact_timeline_resolution_search"} & used_tools
        ):
            default_next = "finalizer"
            evaluation_notes = evaluation_notes or "Two rounds have already gathered specialized evidence; further search is unlikely to improve the answer enough."

        if profile["is_mcq"] and int(state.get("tool_round_index", 0) or 0) >= 1 and top_label and second_label and margin < 0.9 and remaining_rounds > 0:
            default_next = "planner"
            evaluation_notes = (
                evaluation_notes
                or f"The main unresolved issue is distinguishing {top_label} from {second_label}; use one targeted follow-up tool instead of another broad search."
            )

        if remaining <= 0:
            return {
                "messages": list(state.get("messages") or []),
                "remaining_llm_calls": 0,
                "remaining_tool_rounds": remaining_rounds,
                "tool_round_index": int(state.get("tool_round_index", 0) or 0),
                "evidence_pool": list(state.get("evidence_pool") or []),
                "entities_found": list(state.get("entities_found") or []),
                "plan_notes": str(state.get("plan_notes", "") or "").strip(),
                "evaluation_notes": evaluation_notes,
                "next_step": "finalizer",
                "draft_answer": str(state.get("draft_answer", "") or "").strip(),
                "last_round_new_evidence_count": last_new,
                "stagnation_count": stagnation_count,
            }

        if default_next == "finalizer":
            return {
                "messages": list(state.get("messages") or []),
                "remaining_llm_calls": remaining,
                "remaining_tool_rounds": remaining_rounds,
                "tool_round_index": int(state.get("tool_round_index", 0) or 0),
                "evidence_pool": list(state.get("evidence_pool") or []),
                "entities_found": list(state.get("entities_found") or []),
                "plan_notes": str(state.get("plan_notes", "") or "").strip(),
                "evaluation_notes": evaluation_notes,
                "next_step": "finalizer",
                "draft_answer": str(state.get("draft_answer", "") or "").strip(),
                "last_round_new_evidence_count": last_new,
                "stagnation_count": stagnation_count,
            }

        raw_response = self.llm.invoke(self._evaluator_messages(state))
        payload = _extract_first_json_payload(_content_to_text(getattr(raw_response, "content", "")))
        next_step = default_next
        if isinstance(payload, dict):
            parsed_step = str(payload.get("next_step", "") or "").strip().lower()
            confidence = float(payload.get("confidence", 0.0) or 0.0)
            if parsed_step in {"planner", "finalizer"}:
                next_step = parsed_step
            if confidence >= self.evaluator_finalize_confidence:
                next_step = "finalizer"
            notes_parts = [
                str(payload.get("evaluator_notes", "") or "").strip(),
                (f"Leading option: {str(payload.get('leading_option') or '').strip()}" if str(payload.get("leading_option") or "").strip() else ""),
                (
                    f"Runner-up option: {str(payload.get('runner_up_option') or '').strip()}"
                    if str(payload.get("runner_up_option") or "").strip()
                    else ""
                ),
                (
                    f"Option margin: {float(payload.get('option_margin', 0.0) or 0.0):.2f}"
                    if payload.get("option_margin") is not None
                    else ""
                ),
                ("Missing: " + "; ".join(str(x).strip() for x in (payload.get("missing_information") or []) if str(x).strip()))
                if isinstance(payload.get("missing_information"), list) and payload.get("missing_information")
                else "",
                ("Contradictions: " + "; ".join(str(x).strip() for x in (payload.get("contradictions") or []) if str(x).strip()))
                if isinstance(payload.get("contradictions"), list) and payload.get("contradictions")
                else "",
            ]
            evaluation_notes = "\n".join(part for part in notes_parts if part).strip() or evaluation_notes
        return {
            "messages": list(state.get("messages") or []),
            "remaining_llm_calls": remaining - 1,
            "remaining_tool_rounds": remaining_rounds,
            "tool_round_index": int(state.get("tool_round_index", 0) or 0),
            "evidence_pool": list(state.get("evidence_pool") or []),
            "entities_found": list(state.get("entities_found") or []),
            "plan_notes": str(state.get("plan_notes", "") or "").strip(),
            "evaluation_notes": evaluation_notes,
            "next_step": next_step,
            "draft_answer": str(state.get("draft_answer", "") or "").strip(),
            "last_round_new_evidence_count": last_new,
            "stagnation_count": stagnation_count,
        }

    def _route_after_evaluator(self, state: _S4AgentState) -> str:
        step = str(state.get("next_step", "") or "").strip().lower()
        return "planner" if step == "planner" else "finalizer"

    def _finalizer_messages(self, state: _S4AgentState) -> List[BaseMessage]:
        evidence_summary = self._summarize_evidence_pool(list(state.get("evidence_pool") or []))
        entities_summary = self._summarize_entities(list(state.get("entities_found") or []))
        option_board_summary = self._summarize_option_board(state)
        draft_answer = str(state.get("draft_answer", "") or "").strip()
        question_text = _content_to_text(self._extract_latest_user_text(state))
        profile = self._question_profile(question_text)
        parsed = profile.get("parsed") if isinstance(profile.get("parsed"), dict) else {}
        choice_order = list(parsed.get("choice_order") or [])
        choices = parsed.get("choices") if isinstance(parsed.get("choices"), dict) else {}
        choice_block = "\n".join(
            f"{label}. {str(choices.get(label, '') or '').strip()}"
            for label in choice_order
            if str(choices.get(label, "") or "").strip()
        )
        if profile["is_mcq"]:
            return [
                SystemMessage(
                    content=(
                        f"{self.system_message}\n\n"
                        "You are the final multiple-choice answer generator. Do not call tools. "
                        "Choose the single best-supported option from the evidence."
                    ).strip()
                ),
                HumanMessage(
                    content="\n\n".join(
                        [
                            f"Question:\n{_clip_text(question_text, limit=2500)}",
                            f"Choices:\n{choice_block}",
                            f"Evidence pool:\n{evidence_summary}",
                            f"Option evidence board:\n{option_board_summary}",
                            f"Resolved entities:\n{entities_summary}",
                            f"Evaluator notes:\n{str(state.get('evaluation_notes', '') or '').strip() or '(none)'}",
                            f"Draft answer hint:\n{draft_answer or '(none)'}",
                            "Return JSON only:",
                            '{"answer_choice":"A","answer_text":"...","evidence":"...","confidence":0.0}',
                            "Rules:",
                            "- Choose exactly one option label.",
                            "- `answer_text` should restate the chosen option briefly.",
                            "- `evidence` should mention the strongest supporting clue.",
                            "- `confidence` must be between 0 and 1.",
                            "- Prefer the option with the strongest direct support and the fewest unsupported assumptions.",
                            "- If the top option and runner-up are close, explain why the chosen option has the stronger grounded clue.",
                        ]
                    ).strip()
                ),
            ]
        human_prompt = "\n\n".join(
            [
                f"Question:\n{_clip_text(question_text, limit=2500)}",
                f"Evidence pool:\n{evidence_summary}",
                f"Resolved entities:\n{entities_summary}",
                f"Evaluator notes:\n{str(state.get('evaluation_notes', '') or '').strip() or '(none)'}",
                f"Draft answer hint:\n{draft_answer or '(none)'}",
                "Produce the final answer to the user based only on the evidence above.",
                "If the evidence remains ambiguous, answer conservatively and acknowledge the ambiguity instead of inventing facts.",
                'Return JSON only: {"final_answer":"<answer>"}',
            ]
        ).strip()
        return [
            SystemMessage(
                content=(
                    f"{self.system_message}\n\n"
                    "You are the final answer generator. Do not call any tools. "
                    "Synthesize the evidence pool into one grounded answer."
                ).strip()
            ),
            HumanMessage(content=human_prompt),
        ]

    def _finalizer_node(self, state: _S4AgentState) -> Dict[str, Any]:
        remaining = int(state.get("remaining_llm_calls", 0) or 0)
        messages = list(state.get("messages") or [])
        if remaining <= 0:
            fallback = str(state.get("draft_answer", "") or "").strip()
            if not fallback:
                fallback = "I do not have enough grounded evidence to provide a confident answer."
            return {
                "messages": messages + [AIMessage(content=fallback)],
                "remaining_llm_calls": 0,
                "remaining_tool_rounds": int(state.get("remaining_tool_rounds", 0) or 0),
                "tool_round_index": int(state.get("tool_round_index", 0) or 0),
                "evidence_pool": list(state.get("evidence_pool") or []),
                "entities_found": list(state.get("entities_found") or []),
                "plan_notes": str(state.get("plan_notes", "") or "").strip(),
                "evaluation_notes": str(state.get("evaluation_notes", "") or "").strip(),
                "next_step": "finalizer",
                "draft_answer": fallback,
                "last_round_new_evidence_count": int(state.get("last_round_new_evidence_count", 0) or 0),
                "stagnation_count": int(state.get("stagnation_count", 0) or 0),
            }
        raw_response = self.llm.invoke(self._finalizer_messages(state))
        payload = _extract_first_json_payload(_content_to_text(getattr(raw_response, "content", "")))
        final_answer = ""
        if isinstance(payload, dict):
            final_answer = _content_to_text(
                payload.get("final_answer")
                or payload.get("answer")
                or payload.get("answer_text")
                or ""
            ).strip()
            answer_choice = str(
                payload.get("answer_choice")
                or payload.get("choice")
                or payload.get("option")
                or ""
            ).strip()
            if answer_choice and final_answer:
                final_answer = json.dumps(
                    {
                        "answer_choice": answer_choice,
                        "answer_text": final_answer,
                        "evidence": _content_to_text(payload.get("evidence") or "").strip(),
                        "confidence": payload.get("confidence", 0.0),
                    },
                    ensure_ascii=False,
                )
        if not final_answer:
            final_answer = _content_to_text(getattr(raw_response, "content", "")).strip()
        if not final_answer:
            final_answer = str(state.get("draft_answer", "") or "").strip()
        return {
            "messages": messages + [AIMessage(content=final_answer)],
            "remaining_llm_calls": remaining - 1,
            "remaining_tool_rounds": int(state.get("remaining_tool_rounds", 0) or 0),
            "tool_round_index": int(state.get("tool_round_index", 0) or 0),
            "evidence_pool": list(state.get("evidence_pool") or []),
            "entities_found": list(state.get("entities_found") or []),
            "plan_notes": str(state.get("plan_notes", "") or "").strip(),
            "evaluation_notes": str(state.get("evaluation_notes", "") or "").strip(),
            "next_step": "finalizer",
            "draft_answer": final_answer,
            "last_round_new_evidence_count": int(state.get("last_round_new_evidence_count", 0) or 0),
            "stagnation_count": int(state.get("stagnation_count", 0) or 0),
        }

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
                "remaining_tool_rounds": self.max_tool_rounds_per_run,
                "tool_round_index": 0,
                "evidence_pool": [],
                "entities_found": [],
                "plan_notes": "",
                "evaluation_notes": "",
                "next_step": "planner",
                "draft_answer": "",
                "last_round_new_evidence_count": 0,
                "stagnation_count": 0,
            }
        )
        return self._to_legacy_responses(list(result.get("messages") or []))


def create_langgraph_assistant_runtime(
    *,
    function_list: Sequence[Any],
    llm: Any,
    system_message: str,
    rag_cfg: Optional[Dict[str, Any]] = None,
) -> LangGraphAssistantRuntime:
    runtime_cfg = dict(rag_cfg or {})
    runtime_variant = str(
        runtime_cfg.get("assistant_runtime_variant")
        or os.environ.get("NKW_LANGGRAPH_RUNTIME_VARIANT", "s3")
        or "s3"
    ).strip().lower()
    if runtime_variant == "s4":
        return S4LangGraphAssistantRuntime(
            function_list=function_list,
            llm=llm,
            system_message=system_message,
            rag_cfg=runtime_cfg,
        )
    return LangGraphAssistantRuntime(
        function_list=function_list,
        llm=llm,
        system_message=system_message,
        rag_cfg=runtime_cfg,
    )

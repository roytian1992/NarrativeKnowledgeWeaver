from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


def clip_text(value: Any, limit: int) -> str:
    text = str(value or "").strip()
    if limit <= 0 or len(text) <= limit:
        return text
    marker = "\n...[truncated]...\n"
    head = max(0, int(limit * 0.7))
    tail = max(0, limit - head - len(marker))
    if tail <= 0:
        return text[:limit]
    return text[:head] + marker + text[-tail:]


def summarize_tool_uses(tool_uses: List[Dict[str, Any]], *, max_items: int = 6) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for item in tool_uses[: max(1, int(max_items or 1))]:
        if not isinstance(item, dict):
            continue
        out.append(
            {
                "tool_name": str(item.get("tool_name", "") or "").strip(),
                "tool_arguments": clip_text(item.get("tool_arguments", ""), 280),
                "tool_output_summary": clip_text(item.get("tool_output", ""), 420),
            }
        )
    return out


@dataclass
class StrategySubagentCandidate:
    candidate_id: str
    template_id: str
    pattern_name: str
    similarity: float
    recommended_chain: List[str]
    query_abstract: str
    routing_hint: str
    answer: str
    tool_uses: List[Dict[str, Any]]
    responses: List[Dict[str, Any]]
    error: str = ""
    selection_judge: Dict[str, Any] = field(default_factory=dict)

    def to_selector_payload(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "template_id": self.template_id,
            "pattern_name": self.pattern_name,
            "similarity": round(float(self.similarity or 0.0), 6),
            "recommended_chain": list(self.recommended_chain or []),
            "query_abstract": self.query_abstract,
            "routing_hint": clip_text(self.routing_hint, 600),
            "answer": clip_text(self.answer, 1800),
            "tool_trace": summarize_tool_uses(self.tool_uses),
            "selection_judge": {
                "score": round(float((self.selection_judge or {}).get("score", 0.0) or 0.0), 6),
                "is_success": bool((self.selection_judge or {}).get("is_success", False)),
                "reason": clip_text((self.selection_judge or {}).get("reason", ""), 240),
            },
            "error": clip_text(self.error, 300),
        }

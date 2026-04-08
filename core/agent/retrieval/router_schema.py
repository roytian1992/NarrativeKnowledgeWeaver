from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional


QUESTION_TYPES = {
    "exact_fact",
    "relationship",
    "attitude_or_state",
    "implication_or_inference",
    "chronology",
    "section_localization",
    "unknown",
}

ANSWER_SHAPES = {
    "mcq_option",
    "short_fact",
    "explanation",
    "list",
}

EVIDENCE_NEEDS = {
    "local",
    "local_plus_narrative",
    "entity_plus_local",
    "chronology",
    "mixed",
}

QUESTION_KEYWORDS: Dict[str, List[str]] = {
    "relationship": ["relationship", "between", "feel about", "opinion about", "interact", "connection"],
    "chronology": ["before", "after", "timeline", "first", "second", "eventually", "later", "earlier", "year", "age", "how long"],
    "section_localization": ["which section", "which scene", "which chapter", "where in the story", "section"],
    "attitude_or_state": ["feel", "attitude", "emotion", "state of mind", "why does", "why is", "calm", "afraid", "angry", "believe"],
    "implication_or_inference": ["imply", "implied", "suggest", "significance", "most likely", "warning", "dilemma", "motive", "motivation", "why"],
    "exact_fact": ["who", "what", "which", "where", "when", "name", "title"],
}

ENTITY_ANCHOR_RE = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}\b")
OPTION_BLOCK_RE = re.compile(r"(?m)^\s*[A-D][\.\):]\s+.+$")
INLINE_OPTION_RE = re.compile(r"\([A-D]\)\s*[^()\n]{2,}")


def unique_names(items: Iterable[str], *, valid: Optional[Iterable[str]] = None, limit: Optional[int] = None) -> List[str]:
    valid_set = {str(x).strip() for x in (valid or []) if str(x).strip()} if valid is not None else None
    out: List[str] = []
    for raw in items:
        name = str(raw or "").strip()
        if not name:
            continue
        if valid_set is not None and name not in valid_set:
            continue
        if name in out:
            continue
        out.append(name)
        if limit is not None and len(out) >= max(1, int(limit)):
            break
    return out


def detect_mcq(query: str) -> bool:
    text = str(query or "")
    lowered = text.lower()
    if "choices:" in lowered or "options:" in lowered:
        return True
    if len(INLINE_OPTION_RE.findall(text)) >= 2:
        return True
    return bool(OPTION_BLOCK_RE.search(text))


def detect_question_type(query: str) -> str:
    lowered = str(query or "").strip().lower()
    for label in ["relationship", "chronology", "section_localization", "attitude_or_state", "implication_or_inference"]:
        if any(needle in lowered for needle in QUESTION_KEYWORDS.get(label, [])):
            return label
    if any(needle in lowered for needle in QUESTION_KEYWORDS.get("exact_fact", [])):
        return "exact_fact"
    return "unknown"


def detect_answer_shape(query: str, *, question_type: Optional[str] = None) -> str:
    if detect_mcq(query):
        return "mcq_option"
    lowered = str(query or "").strip().lower()
    if any(token in lowered for token in ["list", "which of the following", "what are the", "name the"]):
        return "list"
    if (question_type or "") in {"attitude_or_state", "implication_or_inference"}:
        return "explanation"
    return "short_fact"


def detect_evidence_need(query: str, *, question_type: Optional[str] = None) -> str:
    qtype = str(question_type or "").strip() or detect_question_type(query)
    if qtype == "chronology":
        return "chronology"
    if qtype in {"attitude_or_state", "implication_or_inference"}:
        return "local_plus_narrative"
    if qtype == "relationship":
        return "entity_plus_local"
    if qtype in {"exact_fact", "section_localization"}:
        return "local"
    return "mixed"


def has_entity_anchor(query: str) -> bool:
    text = str(query or "")
    matches = ENTITY_ANCHOR_RE.findall(text)
    blocked = {
        "Question",
        "Choices",
        "Options",
        "Mandatory Retrieval Guard",
        "Use",
        "Return",
        "Answer",
        "Why",
        "What",
        "Which",
        "Who",
        "Where",
        "When",
        "How",
    }
    for match in matches:
        normalized = str(match or "").strip()
        if not normalized:
            continue
        if normalized in blocked:
            continue
        return True
    return False


def deterministic_parse(query: str) -> Dict[str, Any]:
    question_type = detect_question_type(query)
    answer_shape = detect_answer_shape(query, question_type=question_type)
    evidence_need = detect_evidence_need(query, question_type=question_type)
    return {
        "question_type": question_type,
        "answer_shape": answer_shape,
        "evidence_need": evidence_need,
        "has_entity_anchor": has_entity_anchor(query),
        "must_compare_options": bool(answer_shape == "mcq_option"),
        "is_mcq": detect_mcq(query),
    }


def normalize_question_type(raw: Any, *, fallback: str = "unknown") -> str:
    value = str(raw or "").strip().lower()
    aliases = {
        "exact_fact": "exact_fact",
        "fact": "exact_fact",
        "relationship": "relationship",
        "relation": "relationship",
        "attitude": "attitude_or_state",
        "state": "attitude_or_state",
        "attitude_or_state": "attitude_or_state",
        "implication": "implication_or_inference",
        "inference": "implication_or_inference",
        "implication_or_inference": "implication_or_inference",
        "chronology": "chronology",
        "timeline": "chronology",
        "section_localization": "section_localization",
        "localization": "section_localization",
        "unknown": "unknown",
    }
    normalized = aliases.get(value, value)
    if normalized in QUESTION_TYPES:
        return normalized
    return fallback if fallback in QUESTION_TYPES else "unknown"


def normalize_answer_shape(raw: Any, *, fallback: str = "short_fact", query: str = "") -> str:
    if detect_mcq(query):
        return "mcq_option"
    value = str(raw or "").strip().lower()
    aliases = {
        "mcq": "mcq_option",
        "multiple_choice": "mcq_option",
        "mcq_option": "mcq_option",
        "option": "mcq_option",
        "short_fact": "short_fact",
        "fact": "short_fact",
        "span": "short_fact",
        "quote": "short_fact",
        "explanation": "explanation",
        "reasoning": "explanation",
        "why": "explanation",
        "list": "list",
        "items": "list",
    }
    normalized = aliases.get(value, value)
    if normalized in ANSWER_SHAPES:
        return normalized
    return fallback if fallback in ANSWER_SHAPES else "short_fact"


def normalize_evidence_need(raw: Any, *, fallback: str = "mixed") -> str:
    value = str(raw or "").strip().lower()
    aliases = {
        "local": "local",
        "local_plus_narrative": "local_plus_narrative",
        "narrative": "local_plus_narrative",
        "entity_plus_local": "entity_plus_local",
        "entity": "entity_plus_local",
        "chronology": "chronology",
        "timeline": "chronology",
        "mixed": "mixed",
    }
    normalized = aliases.get(value, value)
    if normalized in EVIDENCE_NEEDS:
        return normalized
    return fallback if fallback in EVIDENCE_NEEDS else "mixed"

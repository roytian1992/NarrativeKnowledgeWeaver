from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List

from core.memory.query_abstractor import QueryAbstractor
from core.utils.format import correct_json_format
from core.utils.function_manager import process_with_format_guarantee
from core.utils.general_text import general_repair_template
from core.utils.general_utils import parse_json_object_from_text

logger = logging.getLogger(__name__)


ALLOWED_PROBLEM_TYPES = {
    "section_localization",
    "content_span_lookup",
    "entity_attribute_lookup",
    "relation_or_interaction_lookup",
    "causal_or_explanatory_lookup",
    "fact_retrieval",
}

ALLOWED_TARGET_CATEGORIES = {
    "character",
    "object_or_device",
    "section",
    "place",
    "event",
    "narrative_fact",
}

ALLOWED_ANSWER_SHAPES = {
    "mcq_option",
    "short_fact",
    "explanation",
    "list",
}

_RE_MCQ = re.compile(
    r"(choices?\s*:|options?\s*:|\([A-D]\)\s+|\b[A-D][\.\):、]\s+|\b[A-D]\s*[:：]\s+)",
    re.IGNORECASE,
)
_RE_MCQ_MARKER = re.compile(r"(\([A-D]\)\s+|\b[A-D][\.\):、]\s+|\b[A-D]\s*[:：]\s+)", re.IGNORECASE)
_RE_QUESTION_BLOCK = re.compile(
    r"(?:^|\n)\s*question\s*:\s*(.+?)(?:(?:\n\s*(?:choices?|options?)\s*:)|$)",
    re.IGNORECASE | re.DOTALL,
)
_RE_CHOICES_BLOCK = re.compile(
    r"(?:^|\n)\s*(choices?|options?)\s*:\s*(.+)$",
    re.IGNORECASE | re.DOTALL,
)


def _sanitize_text(text: Any) -> str:
    return str(text or "").strip()


def canonicalize_query_text(query: Any) -> str:
    text = str(query or "").strip()
    if not text:
        return ""
    question_match = _RE_QUESTION_BLOCK.search(text)
    choices_match = _RE_CHOICES_BLOCK.search(text)
    if question_match:
        question_text = str(question_match.group(1) or "").strip()
        if choices_match:
            choices_text = str(choices_match.group(0) or "").strip()
            return f"{question_text}\n{choices_text}".strip()
        return question_text
    return text


def _sanitize_list(items: Any, *, limit: int = 8) -> List[str]:
    if not isinstance(items, list):
        return []
    out: List[str] = []
    for item in items:
        value = str(item or "").strip()
        if value and value not in out:
            out.append(value)
        if len(out) >= limit:
            break
    return out


def is_multiple_choice_query(query: Any) -> bool:
    text = str(query or "").strip()
    if not text:
        return False
    if bool(re.search(r"choices?\s*:|options?\s*:", text, re.IGNORECASE)):
        return True
    return len(list(_RE_MCQ_MARKER.finditer(text))) >= 2


def strip_multiple_choice_options(query: Any) -> str:
    text = canonicalize_query_text(query)
    if not text:
        return ""
    markers = list(_RE_MCQ_MARKER.finditer(text))
    if len(markers) < 2:
        return text
    stem = text[: markers[0].start()].strip()
    stem = re.sub(r"(choices?|options?)\s*:?\s*$", "", stem, flags=re.IGNORECASE).strip()
    return stem or text


def normalize_answer_shape(raw: Any, *, original_query: str = "") -> str:
    if is_multiple_choice_query(original_query):
        return "mcq_option"
    text = str(raw or "").strip().lower()
    if not text:
        return "short_fact"
    if text in ALLOWED_ANSWER_SHAPES:
        return text
    if any(key in text for key in ["choice", "option", "single_choice", "multiple_choice", "mcq", "a/b/c/d"]):
        return "mcq_option"
    if any(key in text for key in ["section_names", "sections", "scene_names", "scene_list", "list", "multi", "set", "enumeration"]):
        return "list"
    if any(key in text for key in ["why", "how", "reason", "cause", "explain", "explanation", "analysis"]):
        return "explanation"
    if any(key in text for key in ["json", "quote", "span", "with_evidence", "single_choice_with_text", "section_title"]):
        return "short_fact"
    return "short_fact"


def normalize_problem_type(raw: Any) -> str:
    text = str(raw or "").strip().lower()
    if not text:
        return "fact_retrieval"
    if text in ALLOWED_PROBLEM_TYPES:
        return text
    if any(key in text for key in ["section", "scene", "chapter", "localization", "where in document"]):
        return "section_localization"
    if any(key in text for key in ["content", "quote", "span", "dialogue", "text evidence"]):
        return "content_span_lookup"
    if any(key in text for key in ["attribute", "fullname", "full_name", "name", "value", "property"]):
        return "entity_attribute_lookup"
    if any(key in text for key in ["relation", "interaction", "pair", "between entities", "relationship"]):
        return "relation_or_interaction_lookup"
    if any(key in text for key in ["causal", "explan", "reason", "why", "how"]):
        return "causal_or_explanatory_lookup"
    return "fact_retrieval"


def normalize_target_category(raw: Any) -> str:
    text = str(raw or "").strip().lower()
    if not text:
        return "narrative_fact"
    if text in ALLOWED_TARGET_CATEGORIES:
        return text
    if any(key in text for key in ["character", "person", "people", "speaker", "role"]):
        return "character"
    if any(key in text for key in ["object", "device", "machine", "robot", "artifact", "item"]):
        return "object_or_device"
    if any(key in text for key in ["section", "scene", "chapter", "passage"]):
        return "section"
    if any(key in text for key in ["place", "location", "room", "address"]):
        return "place"
    if any(key in text for key in ["event", "incident", "action", "happening"]):
        return "event"
    return "narrative_fact"


def sanitize_retrieval_goals(items: Any, *, query: str = "", problem_type: str = "", answer_shape: str = "") -> List[str]:
    goals = _sanitize_list(items, limit=6)
    cleaned: List[str] = []
    for item in goals:
        value = str(item or "").strip()
        if not value:
            continue
        lowered = value.lower()
        if any(key in lowered for key in ["json", "output format", "format as", "return in", "answer in letter"]):
            continue
        if value not in cleaned:
            cleaned.append(value)

    if cleaned:
        return cleaned[:4]

    fallback: List[str] = []
    if problem_type == "section_localization":
        fallback.extend(["identify the supporting sections", "verify the answer with direct section evidence"])
    elif problem_type == "content_span_lookup":
        fallback.extend(["find the exact supporting text span", "verify the answer against the retrieved text"])
    elif problem_type == "entity_attribute_lookup":
        fallback.extend(["find the target entity mention", "verify the canonical attribute value from source evidence"])
    elif problem_type == "relation_or_interaction_lookup":
        fallback.extend(["identify the involved entities", "retrieve evidence describing their interaction or relation"])
    elif problem_type == "causal_or_explanatory_lookup":
        fallback.extend(["find the causal event or explanation evidence", "summarize the explanation grounded in retrieved evidence"])
    else:
        fallback.extend(["retrieve direct supporting evidence", "verify the answer before concluding"])
    if answer_shape == "list":
        fallback.append("keep only evidence-backed items")
    if is_multiple_choice_query(query):
        fallback.append("compare the retrieved evidence against the answer options")
    deduped: List[str] = []
    for item in fallback:
        if item not in deduped:
            deduped.append(item)
    return deduped[:4]


def normalize_query_pattern_payload(payload: Dict[str, Any], *, original_query: str = "", fallback: Dict[str, Any] | None = None) -> Dict[str, Any]:
    base = fallback or {}
    query_abstract = _sanitize_text(payload.get("query_abstract")) or _sanitize_text(base.get("query_abstract"))
    problem_type = normalize_problem_type(payload.get("problem_type") or base.get("problem_type"))
    target_category = normalize_target_category(payload.get("target_category") or base.get("target_category"))
    answer_shape = normalize_answer_shape(
        payload.get("answer_shape") or base.get("answer_shape"),
        original_query=original_query,
    )
    retrieval_goals = sanitize_retrieval_goals(
        payload.get("retrieval_goals") or base.get("retrieval_goals"),
        query=original_query,
        problem_type=problem_type,
        answer_shape=answer_shape,
    )
    return {
        "query_abstract": query_abstract,
        "problem_type": problem_type,
        "target_category": target_category,
        "answer_shape": answer_shape,
        "retrieval_goals": retrieval_goals,
        "notes": _sanitize_text(payload.get("notes")) or _sanitize_text(base.get("notes")),
    }


class StrategyQueryPatternExtractor:
    """
    Extract a reusable, non-text-specific query pattern for strategy matching.
    """

    _RE_SECTION = re.compile(r"场次|场景|章节|scene|scenes|section|sections|chapter", re.IGNORECASE)
    _RE_CONTENT = re.compile(r"什么内容|说了什么|写着什么|标语|屏幕|原文|内容|台词|对话|dialogue|quote", re.IGNORECASE)
    _RE_RELATION = re.compile(r"关系|互动|交互|对话|speaks|interaction|relationship", re.IGNORECASE)
    _RE_NAME = re.compile(r"全称|简称|是什么|是谁|何人|叫什么|name|full name", re.IGNORECASE)
    _RE_REASON = re.compile(r"为什么|为何|如何|怎么|原因|because|why|how", re.IGNORECASE)
    _RE_LIST = re.compile(r"哪些|哪几|列出|分别|all|list|which", re.IGNORECASE)
    _RE_CHARACTER = re.compile(r"角色|人物|谁|他|她|character|person", re.IGNORECASE)
    _RE_OBJECT = re.compile(r"物体|设备|装置|object|machine|robot", re.IGNORECASE)
    _RE_AGE = re.compile(r"几岁|多少岁|年龄|\d+\s*岁", re.IGNORECASE)
    _RE_SCENE = re.compile(r"场次|场景|哪场|哪几场|scene|scenes|章节|chapter|section", re.IGNORECASE)
    _RE_TITLE_REF = re.compile(r"(?:^|[^\d])(\d{1,4})\s*场", re.IGNORECASE)
    _RE_FULLNAME = re.compile(r"全称|简称|正式命名|缩写|full name", re.IGNORECASE)
    _RE_QUOTE = re.compile(r"说了什么|说过什么|台词|对白|遗言|临终|quote", re.IGNORECASE)
    _RE_CLOTH = re.compile(r"穿了|穿着|衣服|西装|制服|着装|服装", re.IGNORECASE)
    _RE_LOCATION_RETURN = re.compile(r"回了|回到|去了|返回|到了", re.IGNORECASE)
    _RE_ACTION_FREQ = re.compile(r"经常|总是|常常|习惯|frequent|often", re.IGNORECASE)

    def __init__(
        self,
        prompt_loader,
        llm,
        *,
        prompt_id: str = "memory/extract_strategy_query_pattern",
        abstraction_mode: str = "hybrid",
    ) -> None:
        self.prompt_loader = prompt_loader
        self.llm = llm
        self.prompt_id = str(prompt_id or "memory/extract_strategy_query_pattern").strip()
        self.abstraction_mode = str(abstraction_mode or "hybrid").strip().lower()
        if self.abstraction_mode not in {"rule", "llm", "hybrid"}:
            self.abstraction_mode = "hybrid"
        self.rule_abstractor = QueryAbstractor(
            llm=None,
            prompt_loader=prompt_loader,
            prompt_id="memory/abstract_query_for_routing",
            abstraction_mode="rule",
        )

    def _heuristic_pattern(self, query: str) -> Dict[str, Any]:
        text = _sanitize_text(canonicalize_query_text(query))
        base_text = strip_multiple_choice_options(text)
        abstract_text = self.rule_abstractor.abstract(base_text or text)

        if self._RE_SECTION.search(base_text or text):
            problem_type = "section_localization"
        elif self._RE_RELATION.search(base_text or text):
            problem_type = "relation_or_interaction_lookup"
        elif self._RE_CONTENT.search(base_text or text):
            problem_type = "content_span_lookup"
        elif self._RE_NAME.search(base_text or text):
            problem_type = "entity_attribute_lookup"
        elif self._RE_REASON.search(base_text or text):
            problem_type = "causal_or_explanatory_lookup"
        else:
            problem_type = "fact_retrieval"

        if self._RE_CHARACTER.search(base_text or text):
            target_category = "character"
        elif self._RE_OBJECT.search(base_text or text):
            target_category = "object_or_device"
        elif self._RE_SECTION.search(base_text or text):
            target_category = "section"
        else:
            target_category = "narrative_fact"

        if is_multiple_choice_query(text):
            answer_shape = "mcq_option"
        elif self._RE_LIST.search(base_text or text):
            answer_shape = "list"
        elif self._RE_REASON.search(base_text or text):
            answer_shape = "explanation"
        else:
            answer_shape = "short_fact"

        retrieval_goals: List[str] = []
        if problem_type == "section_localization":
            retrieval_goals.extend(["identify relevant sections", "return precise section titles"])
        if problem_type == "content_span_lookup":
            retrieval_goals.extend(["locate supporting text span", "quote or paraphrase exact content"])
        if problem_type == "relation_or_interaction_lookup":
            retrieval_goals.extend(["identify relevant entities", "retrieve interaction or relation evidence"])
        if problem_type == "entity_attribute_lookup":
            retrieval_goals.append("retrieve canonical attribute value")
        if problem_type == "causal_or_explanatory_lookup":
            retrieval_goals.extend(["find causal evidence", "summarize explanation grounded in retrieval"])
        if not retrieval_goals:
            retrieval_goals.append("retrieve evidence before answering")

        return normalize_query_pattern_payload(
            {
            "query_abstract": abstract_text,
            "problem_type": problem_type,
            "target_category": target_category,
            "answer_shape": answer_shape,
            "retrieval_goals": retrieval_goals,
            "notes": "heuristic_fallback",
            },
            original_query=text,
        )

    def _sanitize_result(self, data: Dict[str, Any], fallback: Dict[str, Any], *, original_query: str = "") -> Dict[str, Any]:
        if not isinstance(data, dict):
            return fallback
        result = normalize_query_pattern_payload(data, original_query=original_query, fallback=fallback)
        abstract_seed = result["query_abstract"] or fallback["query_abstract"]
        if is_multiple_choice_query(original_query):
            abstract_seed = strip_multiple_choice_options(abstract_seed) or strip_multiple_choice_options(original_query)
        result["query_abstract"] = self.rule_abstractor.abstract(abstract_seed)
        if not result["query_abstract"]:
            result["query_abstract"] = fallback["query_abstract"]
        return result

    def _harmonize_result(self, query: str, result: Dict[str, Any], fallback: Dict[str, Any]) -> Dict[str, Any]:
        text = _sanitize_text(canonicalize_query_text(query))
        if not text:
            return result

        harmonized = dict(result or {})
        if is_multiple_choice_query(text):
            harmonized["answer_shape"] = "mcq_option"
        if self._RE_FULLNAME.search(text):
            harmonized["problem_type"] = "entity_attribute_lookup"
            harmonized["answer_shape"] = "short_fact"
            harmonized["retrieval_goals"] = [
                "find the canonical full name or formal naming",
                "verify the exact name from lexical or source evidence",
            ]
        elif self._RE_AGE.search(text) and self._RE_SCENE.search(text):
            harmonized["problem_type"] = "section_localization"
            harmonized["answer_shape"] = "list"
            harmonized["retrieval_goals"] = [
                "identify scenes satisfying the age constraint",
                "verify the age evidence inside each candidate scene",
            ]
        elif self._RE_AGE.search(text) and self._RE_TITLE_REF.search(text):
            harmonized["problem_type"] = "entity_attribute_lookup"
            harmonized["answer_shape"] = "short_fact"
            harmonized["retrieval_goals"] = [
                "locate the referenced scene",
                "verify the character age from explicit scene evidence",
            ]
        elif self._RE_QUOTE.search(text):
            harmonized["problem_type"] = "content_span_lookup"
            harmonized["answer_shape"] = "short_fact"
            harmonized["retrieval_goals"] = [
                "locate the relevant scene or passage",
                "extract the exact spoken line or the closest supported wording",
            ]
        elif self._RE_CLOTH.search(text) and self._RE_SCENE.search(text):
            harmonized["problem_type"] = "section_localization"
            harmonized["answer_shape"] = "list"
            harmonized["retrieval_goals"] = [
                "find scenes where the character appears",
                "verify the appearance or clothing detail in each scene",
            ]
        elif self._RE_LOCATION_RETURN.search(text) and self._RE_SCENE.search(text):
            harmonized["problem_type"] = "section_localization"
            harmonized["answer_shape"] = "list"
            harmonized["retrieval_goals"] = [
                "find scenes involving the return to the target place",
                "verify the return event with direct source evidence",
            ]
        elif self._RE_ACTION_FREQ.search(text):
            harmonized["problem_type"] = "relation_or_interaction_lookup"
            harmonized["answer_shape"] = "short_fact"
            harmonized["retrieval_goals"] = [
                "find evidence of repeated interactions between the entities",
                "summarize the repeated action from direct evidence",
            ]
        result = normalize_query_pattern_payload(harmonized, original_query=text, fallback=fallback)
        abstract_seed = result["query_abstract"] or fallback["query_abstract"] or text
        if is_multiple_choice_query(text):
            abstract_seed = strip_multiple_choice_options(abstract_seed) or strip_multiple_choice_options(text)
        result["query_abstract"] = self.rule_abstractor.abstract(abstract_seed)
        if not result["query_abstract"]:
            result["query_abstract"] = fallback["query_abstract"]
        return result

    def extract(self, query: str) -> Dict[str, Any]:
        canonical_query = canonicalize_query_text(query)
        fallback = self._heuristic_pattern(canonical_query)
        if self.abstraction_mode == "rule" or self.llm is None or self.prompt_loader is None:
            return fallback
        try:
            user_prompt = self.prompt_loader.render(
                self.prompt_id,
                static_values={},
                task_values={
                    "original_query": _sanitize_text(canonical_query),
                    "rule_abstract": fallback["query_abstract"],
                },
                strict=True,
            )
        except Exception as exc:
            logger.warning("strategy query pattern prompt render failed: %s", exc)
            return fallback

        try:
            corrected_json, status = process_with_format_guarantee(
                llm_client=self.llm,
                messages=[{"role": "user", "content": user_prompt}],
                required_fields=["query_abstract", "problem_type", "target_category", "answer_shape", "retrieval_goals"],
                field_validators={
                    "query_abstract": lambda v: isinstance(v, str) and bool(v.strip()),
                    "problem_type": lambda v: isinstance(v, str) and bool(v.strip()),
                    "target_category": lambda v: isinstance(v, str) and bool(v.strip()),
                    "answer_shape": lambda v: isinstance(v, str) and bool(v.strip()),
                    "retrieval_goals": lambda v: isinstance(v, list),
                },
                max_retries=2,
                repair_template=general_repair_template,
            )
        except Exception as exc:
            logger.warning("strategy query pattern extraction failed: %s", exc)
            return {
                **fallback,
                "notes": f"heuristic_fallback_due_to_error: {str(exc).strip()[:300]}",
            }
        if status != "success":
            return fallback

        payload = parse_json_object_from_text(corrected_json)
        sanitized = normalize_query_pattern_payload(payload or {}, original_query=_sanitize_text(canonical_query), fallback=fallback)
        abstract_seed = sanitized["query_abstract"] or fallback["query_abstract"] or canonical_query
        if is_multiple_choice_query(canonical_query):
            abstract_seed = strip_multiple_choice_options(abstract_seed) or strip_multiple_choice_options(query)
        sanitized["query_abstract"] = self.rule_abstractor.abstract(abstract_seed)
        if not sanitized["query_abstract"]:
            sanitized["query_abstract"] = fallback["query_abstract"]
        return self._harmonize_result(canonical_query, sanitized, fallback)

    @staticmethod
    def pattern_to_text(pattern: Dict[str, Any]) -> str:
        pattern = pattern or {}
        goals = ", ".join(_sanitize_list(pattern.get("retrieval_goals")))
        parts = [
            f"retrieval_goals={goals}",
            f"query_abstract={_sanitize_text(pattern.get('query_abstract'))}",
        ]
        return " ; ".join(x for x in parts if x and not x.endswith("="))


def extract_strategy_query_pattern_with_guard(
    *,
    llm,
    prompt_loader,
    original_query: str,
    prompt_id: str = "memory/extract_strategy_query_pattern",
    abstraction_mode: str = "hybrid",
) -> str:
    extractor = StrategyQueryPatternExtractor(
        prompt_loader=prompt_loader,
        llm=llm,
        prompt_id=prompt_id,
        abstraction_mode=abstraction_mode,
    )
    result = extractor.extract(original_query)
    return correct_json_format(json.dumps(result, ensure_ascii=False))


class StrategyQueryPatternTool:
    def __init__(self, prompt_loader, llm, prompt_id: str = "memory/extract_strategy_query_pattern"):
        self.prompt_loader = prompt_loader
        self.llm = llm
        self.prompt_id = prompt_id

    def call(self, params: str, **kwargs) -> str:
        try:
            payload = json.loads(params) if isinstance(params, str) else (params or {})
        except Exception as exc:
            return correct_json_format(json.dumps({"error": f"params parse failed: {exc}"}, ensure_ascii=False))
        return extract_strategy_query_pattern_with_guard(
            llm=self.llm,
            prompt_loader=self.prompt_loader,
            original_query=str(payload.get("original_query", "") or ""),
            prompt_id=self.prompt_id,
            abstraction_mode=str(payload.get("abstraction_mode", "hybrid") or "hybrid"),
        )

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

from core.builder.manager.information_manager import InformationExtractor
from core.utils.general_utils import safe_str
from core.utils.format import DOC_TYPE_META


_WHITESPACE_RE = re.compile(r"\s+")
_INTERACTIVE_OBJECT_KEYWORDS = (
    "机器人", "机械人", "ai", "人工智能", "智能体", "助手", "管家",
    "系统", "程序", "模型", "终端", "bot", "robot", "android",
    "assistant", "agent", "chatbot", "drone",
)


def _norm_text(x: Any) -> str:
    return _WHITESPACE_RE.sub(" ", safe_str(x)).strip()


def _primary_type(x: Any) -> str:
    if isinstance(x, str):
        return x.strip()
    if isinstance(x, list) and x:
        for t in x:
            s = safe_str(t).strip()
            if s:
                return s
    return ""


def _stable_id(key: str, prefix: str = "int_") -> str:
    return prefix + hashlib.md5(key.encode("utf-8")).hexdigest()[:16]


class InteractionExtractionAgent:
    """
    Extract interaction records (dialogue + action interaction) from text chunks.

    Notes:
    - Uses InformationExtractor.extract_interactions (LLM) for generation.
    - Validates output against interaction schema (from/to, allow_null_object, polarity).
    - Produces flat records ready for JSON persistence and SQL import.
    """

    def __init__(
        self,
        config: Any,
        llm: Any,
        interaction_schema: Dict[str, List[Dict[str, Any]]],
    ):
        self.config = config
        self.llm = llm
        self.extractor = InformationExtractor(config, llm)
        self.interaction_schema = interaction_schema or {}
        doc_type = safe_str(getattr(getattr(config, "global_config", None), "doc_type", "")).strip() or "general"
        if doc_type not in DOC_TYPE_META:
            doc_type = "general"
        self.meta = DOC_TYPE_META[doc_type]
        self.title_key = safe_str(self.meta.get("title", "title")).strip() or "title"
        self.subtitle_key = safe_str(self.meta.get("subtitle", "subtitle")).strip() or "subtitle"
        section_label = safe_str(self.meta.get("section_label", "Document")).strip() or "Document"
        section_prefix = re.sub(r"[^A-Za-z0-9_]+", "_", section_label).strip("_").lower() or "document"
        self.section_id_key = f"{section_prefix}_id"

    def _is_interactive_object(self, obj: Dict[str, Any]) -> bool:
        name = _norm_text((obj or {}).get("name")).lower()
        aliases = (obj or {}).get("aliases") or []
        if not isinstance(aliases, list):
            aliases = []
        alias_text = " ".join([_norm_text(a).lower() for a in aliases if _norm_text(a)])
        haystack = f"{name} {alias_text}".strip()
        if not haystack:
            return False
        return any(k in haystack for k in _INTERACTIVE_OBJECT_KEYWORDS)

    def _entities_text_for_extractor(self, entities: List[Dict[str, Any]]) -> str:
        lines: List[str] = []
        for e in entities or []:
            if not isinstance(e, dict):
                continue
            name = _norm_text(e.get("name"))
            etype = _norm_text(e.get("type"))
            if not name or not etype:
                continue
            lines.append(f"- {name} | type: {etype}")
        return "\n".join(lines)

    def _interactions_text_for_extractor(self, interactions: Dict[str, Dict[str, Any]]) -> str:
        lines: List[str] = []
        for x in interactions.values():
            if not isinstance(x, dict):
                continue
            subject = _norm_text(x.get("subject_name"))
            obj = _norm_text(x.get("object_name"))
            itype = _norm_text(x.get("interaction_type"))
            content = _norm_text(x.get("content"))
            if not subject or not itype:
                continue
            if obj:
                lines.append(f"- ({subject})-[{itype}]->({obj}): {content}")
            else:
                lines.append(f"- ({subject})-[{itype}]->(NULL): {content}")
        return "\n".join(lines)

    def _parse_interaction_list(self, raw: Any) -> List[Dict[str, Any]]:
        data: Any = raw
        if isinstance(raw, str):
            s = raw.strip()
            if not s:
                return []
            try:
                data = json.loads(s)
            except Exception:
                return []

        if isinstance(data, list):
            return [x for x in data if isinstance(x, dict)]

        if isinstance(data, dict):
            if isinstance(data.get("interactions"), list):
                return [x for x in data.get("interactions", []) if isinstance(x, dict)]
            if any(k in data for k in ["subject", "interaction_type", "type", "content"]):
                return [data]

        return []

    def _build_candidate_index(
        self,
        entity_candidates: List[Dict[str, Any]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        idx: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for e in entity_candidates or []:
            if not isinstance(e, dict):
                continue
            name = _norm_text(e.get("name"))
            etype = _norm_text(e.get("type"))
            eid = _norm_text(e.get("id"))
            if not name or not etype or not eid:
                continue

            normalized = {
                "id": eid,
                "name": name,
                "type": etype,
                "aliases": [a for a in (e.get("aliases") or []) if _norm_text(a)],
            }
            idx[name.lower()].append(normalized)

            for a in normalized["aliases"]:
                idx[_norm_text(a).lower()].append(normalized)

        return idx

    def _resolve_entity(
        self,
        entity_name: str,
        *,
        candidate_index: Dict[str, List[Dict[str, Any]]],
        allowed_types: List[str],
    ) -> Optional[Dict[str, Any]]:
        key = _norm_text(entity_name).lower()
        if not key:
            return None

        cands = candidate_index.get(key, [])
        if not cands:
            return None

        allow = {safe_str(t).strip() for t in (allowed_types or []) if safe_str(t).strip()}
        if not allow:
            return cands[0]

        typed = [c for c in cands if _norm_text(c.get("type")) in allow]
        if typed:
            typed.sort(key=lambda x: (_norm_text(x.get("name")), _norm_text(x.get("id"))))
            return typed[0]
        return None

    def _validate_and_normalize_interactions(
        self,
        *,
        interactions: List[Dict[str, Any]],
        group_name: str,
        group_rules: List[Dict[str, Any]],
        entity_candidates: List[Dict[str, Any]],
        document_id: str,
        chunk_id: str,
        section_id: str,
        section_title: str,
        subsection_title: str,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        out: List[Dict[str, Any]] = []
        feedbacks: List[Dict[str, Any]] = []

        rule_finder: Dict[str, Dict[str, Any]] = {}
        for r in group_rules or []:
            if not isinstance(r, dict):
                continue
            rt = _norm_text(r.get("type"))
            if not rt:
                continue
            rule_finder[rt] = r

        candidate_index = self._build_candidate_index(entity_candidates)

        for i, item in enumerate(interactions or []):
            subject_name = _norm_text(item.get("subject"))
            object_name = _norm_text(item.get("object"))
            interaction_type = _norm_text(item.get("interaction_type") or item.get("type"))
            polarity = _norm_text(item.get("polarity")).lower()
            content = _norm_text(item.get("content"))

            err_ctx = {
                "group": group_name,
                "document_id": document_id,
                "chunk_id": chunk_id,
                "index": i,
                "subject": subject_name,
                "object": object_name,
                "interaction_type": interaction_type,
            }

            if not subject_name:
                feedbacks.append({**err_ctx, "error_type": "subject_missing", "feedback": "subject is empty"})
                continue
            if not interaction_type:
                feedbacks.append({**err_ctx, "error_type": "interaction_type_missing", "feedback": "interaction_type is empty"})
                continue
            if not content:
                feedbacks.append({**err_ctx, "error_type": "content_missing", "feedback": "content is empty"})
                continue

            rule = rule_finder.get(interaction_type)
            if not rule:
                feedbacks.append({**err_ctx, "error_type": "undefined_interaction_type", "feedback": f"interaction_type [{interaction_type}] is not defined in schema"})
                continue

            from_types = [safe_str(x).strip() for x in (rule.get("from") or []) if safe_str(x).strip()]
            to_types = [safe_str(x).strip() for x in (rule.get("to") or []) if safe_str(x).strip()]
            allow_null_object = bool(rule.get("allow_null_object", False))
            polarity_required = bool(rule.get("polarity_required", False))
            polarity_values = [safe_str(x).strip().lower() for x in (rule.get("polarity_values") or []) if safe_str(x).strip()]

            subj = self._resolve_entity(
                subject_name,
                candidate_index=candidate_index,
                allowed_types=from_types,
            )
            if not subj:
                feedbacks.append({**err_ctx, "error_type": "subject_not_found_or_type_violation", "feedback": f"subject [{subject_name}] not found or type not in {from_types}"})
                continue

            obj: Optional[Dict[str, Any]] = None
            if object_name:
                obj = self._resolve_entity(
                    object_name,
                    candidate_index=candidate_index,
                    allowed_types=to_types,
                )
                if not obj:
                    feedbacks.append({**err_ctx, "error_type": "object_not_found_or_type_violation", "feedback": f"object [{object_name}] not found or type not in {to_types}"})
                    continue
            else:
                if not allow_null_object:
                    feedbacks.append({**err_ctx, "error_type": "object_required", "feedback": f"object is required for interaction_type [{interaction_type}]"})
                    continue

            if polarity_required:
                if polarity not in polarity_values:
                    feedbacks.append({**err_ctx, "error_type": "polarity_missing_or_invalid", "feedback": f"polarity must be one of {polarity_values}"})
                    continue
            else:
                if polarity not in polarity_values:
                    polarity = ""

            object_id = _norm_text((obj or {}).get("id"))
            object_name_final = _norm_text((obj or {}).get("name"))
            object_type = _norm_text((obj or {}).get("type"))

            if obj is not None and object_type == "Object":
                if not self._is_interactive_object(obj):
                    feedbacks.append(
                        {
                            **err_ctx,
                            "error_type": "object_non_interactive",
                            "feedback": (
                                f"object [{object_name_final}] is not an interactive/agentive Object "
                                f"(robot/AI/system-like)."
                            ),
                        }
                    )
                    continue

            rid_key = "||".join([
                document_id,
                chunk_id,
                _norm_text(subj.get("id")),
                object_id,
                interaction_type,
                polarity,
                content,
            ])
            rid = _stable_id(rid_key)

            out.append(
                {
                    "rid": rid,
                    "document_id": document_id,
                    "chunk_id": chunk_id,
                    "section_id": section_id,
                    self.title_key: section_title,
                    self.subtitle_key: subsection_title,
                    "subject_id": _norm_text(subj.get("id")),
                    "subject_name": _norm_text(subj.get("name")),
                    "subject_type": _norm_text(subj.get("type")),
                    "object_id": object_id,
                    "object_name": object_name_final,
                    "object_type": object_type,
                    "interaction_type": interaction_type,
                    "polarity": polarity,
                    "content": content,
                }
            )

        return out, feedbacks

    def _dedupe_interaction_records(self, interactions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        deduped: List[Dict[str, Any]] = []
        seen: set[Tuple[str, str, str, str, str, str]] = set()
        for item in interactions or []:
            if not isinstance(item, dict):
                continue
            key = (
                _norm_text(item.get("document_id")),
                _norm_text(item.get("subject_id")),
                _norm_text(item.get("object_id")),
                _norm_text(item.get("interaction_type")),
                _norm_text(item.get("polarity")),
                _norm_text(item.get("content")),
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        return deduped

    def _extract_interactions_one_chunk(
        self,
        *,
        cleaned_text: str,
        document_id: str,
        chunk_id: str,
        section_id: str,
        section_title: str,
        subsection_title: str,
        entity_candidates: List[Dict[str, Any]],
        prev_interactions: Dict[str, Dict[str, Any]],
        rid_namespace: str,
    ) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
        all_interactions = dict(prev_interactions or {})
        all_feedbacks: Dict[str, List[Dict[str, Any]]] = {}

        entities_text = self._entities_text_for_extractor(entity_candidates)
        extracted_interactions_ctx = self._interactions_text_for_extractor(all_interactions)

        for group_name, group_rules in self.interaction_schema.items():
            raw = self.extractor.extract_interactions(
                text=cleaned_text,
                extracted_entities=entities_text,
                extracted_interactions=extracted_interactions_ctx,
                interaction_group=group_name,
                previous_results=None,
                feedbacks=None,
            )
            parsed = self._parse_interaction_list(raw)

            normalized, feedbacks = self._validate_and_normalize_interactions(
                interactions=parsed,
                group_name=group_name,
                group_rules=group_rules,
                entity_candidates=entity_candidates,
                document_id=document_id,
                chunk_id=chunk_id,
                section_id=section_id,
                section_title=section_title,
                subsection_title=subsection_title,
            )

            for x in normalized:
                rid = _norm_text(x.get("rid"))
                if not rid:
                    fallback_key = f"{rid_namespace}:{group_name}:{len(all_interactions)}"
                    rid = _stable_id(fallback_key)
                    x["rid"] = rid
                all_interactions[rid] = x

            if feedbacks:
                all_feedbacks.setdefault(group_name, [])
                all_feedbacks[group_name].extend(feedbacks)

        return all_interactions, all_feedbacks

    def run_document(
        self,
        *,
        document_id: str,
        payload: Dict[str, Any],
        chunk_map: Dict[str, Dict[str, Any]],
        entity_candidates: List[Dict[str, Any]],
        document_rid_namespace: str,
    ) -> Dict[str, Any]:
        try:
            chunk_ids = payload.get("chunk_ids") or []
            if not isinstance(chunk_ids, list):
                chunk_ids = []

            doc_md = payload.get("document_metadata") or {}
            if not isinstance(doc_md, dict):
                doc_md = {}

            chunk_items: List[Dict[str, Any]] = []
            for i, chunk_id in enumerate(chunk_ids):
                chunk = chunk_map.get(chunk_id) or {}
                if not isinstance(chunk, dict):
                    chunk = {}

                content = safe_str(chunk.get("content")).strip()
                if not content:
                    continue

                chunk_md = chunk.get("metadata") or {}
                if not isinstance(chunk_md, dict):
                    chunk_md = {}

                section_id = _norm_text(
                    chunk_md.get(self.section_id_key)
                    or doc_md.get(self.section_id_key)
                    or chunk_md.get("scene_id")
                    or doc_md.get("scene_id")
                    or doc_md.get("raw_doc_id")
                    or doc_md.get("doc_segment_id")
                    or document_id
                )
                section_title = _norm_text(
                    chunk_md.get(self.title_key)
                    or doc_md.get(self.title_key)
                    or chunk_md.get("title")
                    or doc_md.get("title")
                )
                subsection_title = _norm_text(
                    chunk_md.get(self.subtitle_key)
                    or doc_md.get(self.subtitle_key)
                    or chunk_md.get("subtitle")
                    or doc_md.get("subtitle")
                )

                chunk_items.append(
                    {
                        "index": i,
                        "chunk_id": _norm_text(chunk_id),
                        "content": content,
                        "section_id": section_id,
                        "section_title": section_title,
                        "subsection_title": subsection_title,
                    }
                )

            all_feedbacks: Dict[str, List[Dict[str, Any]]] = {}
            interactions_flat: List[Dict[str, Any]] = []

            if chunk_items:
                raw_workers = getattr(self.config, "interaction_chunk_parallel_workers", 4)
                try:
                    worker_limit = max(1, int(raw_workers or 4))
                except Exception:
                    worker_limit = 4
                worker_count = min(worker_limit, len(chunk_items))

                def _run_chunk(item: Dict[str, Any]) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
                    rid_ns = f"{document_rid_namespace}.chunk{item['index']}"
                    return self._extract_interactions_one_chunk(
                        cleaned_text=item["content"],
                        document_id=document_id,
                        chunk_id=item["chunk_id"],
                        section_id=item["section_id"],
                        section_title=item["section_title"],
                        subsection_title=item["subsection_title"],
                        entity_candidates=entity_candidates,
                        prev_interactions={},
                        rid_namespace=rid_ns,
                    )

                with ThreadPoolExecutor(max_workers=worker_count) as ex:
                    fut2item = {ex.submit(_run_chunk, item): item for item in chunk_items}
                    for fut in as_completed(fut2item):
                        chunk_interactions, feedbacks = fut.result()
                        interactions_flat.extend(list((chunk_interactions or {}).values()))
                        for k, items in (feedbacks or {}).items():
                            all_feedbacks.setdefault(k, [])
                            all_feedbacks[k].extend(items)

            interactions_out = self._dedupe_interaction_records(interactions_flat)
            interactions_out.sort(
                key=lambda x: (
                    _norm_text(x.get("document_id")),
                    _norm_text(x.get("chunk_id")),
                    _norm_text(x.get("subject_name")),
                    _norm_text(x.get("object_name")),
                    _norm_text(x.get("interaction_type")),
                )
            )

            feedback_count = sum(len(v) for v in all_feedbacks.values())

            return {
                "ok": True,
                "document_id": document_id,
                "chunk_ids": chunk_ids,
                "interactions": interactions_out,
                "feedback_count": feedback_count,
                "feedbacks": all_feedbacks,
                "error": "",
            }
        except Exception as e:
            return {
                "ok": False,
                "document_id": document_id,
                "chunk_ids": payload.get("chunk_ids") or [],
                "interactions": [],
                "feedback_count": 0,
                "feedbacks": {},
                "error": f"{type(e).__name__}: {e}",
            }

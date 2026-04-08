from __future__ import annotations

import json
import logging
import os
import time
import re
import math
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from core.utils.prompt_loader import YAMLPromptLoader
from core.utils.general_utils import (
    clamp_float,
    json_dump_atomic,
    load_json_or_default,
    parse_json_object_from_text,
)

logger = logging.getLogger(__name__)


class MemoryDistiller:
    """
    Periodically distill raw extraction memories into reusable guidance memories.
    """

    DERIVED_TYPES = {"guideline", "anti_pattern", "canonical_map"}
    SUMMARY_TYPES = {"guideline", "anti_pattern", "canonical_map"}
    SUMMARY_MEMORY_TYPE_MAP = {
        "guideline": "guideline_summary",
        "anti_pattern": "anti_pattern_summary",
        "canonical_map": "canonical_map_summary",
    }
    _DISTILL_MAX_INPUT_CHARS = 12000
    _DISTILL_BATCH_SIZE = 20
    _DISTILL_MAX_OUTPUT_TOKENS = 1024

    def __init__(
        self,
        *,
        llm: Any,
        memory_store: Any,
        problem_solver: Any = None,
        store_path: str,
        enabled: bool = True,
        cycle_every_n_docs: int = 20,
        min_new_entries: int = 100,
        max_source_entries: int = 200,
        max_new_memories: int = 30,
        prompt_dir: str = "task_specs/prompts_en",
        prompt_id: str = "memory/distill_extraction_memory",
    ) -> None:
        self.llm = llm
        self.memory_store = memory_store
        self.problem_solver = problem_solver
        self.store_path = store_path
        self.enabled = bool(enabled)
        self.cycle_every_n_docs = max(1, int(cycle_every_n_docs))
        self.min_new_entries = max(1, int(min_new_entries))
        self.max_source_entries = max(10, int(max_source_entries))
        self.max_new_memories = max(1, int(max_new_memories))
        self.prompt_id = prompt_id
        self.prompt_loader = YAMLPromptLoader(prompt_dir)

        os.makedirs(self.store_path, exist_ok=True)
        self.state_path = os.path.join(self.store_path, "distill_state.json")
        self.latest_memories_path = os.path.join(self.store_path, "distilled_memories_latest.json")
        self.summary_memories_path = os.path.join(self.store_path, "distilled_memory_summaries.json")

    def _load_state(self) -> Dict[str, Any]:
        state = load_json_or_default(self.state_path, {})
        if not isinstance(state, dict):
            state = {}
        return {
            "docs_since_last": int(state.get("docs_since_last", 0)),
            "cycle_id": int(state.get("cycle_id", 0)),
            "distilled_raw_ids": [
                str(x).strip()
                for x in (state.get("distilled_raw_ids") or [])
                if str(x).strip()
            ],
            "summary_entry_ids": {
                str(k).strip(): str(v).strip()
                for k, v in (state.get("summary_entry_ids") or {}).items()
                if str(k).strip() and str(v).strip()
            },
        }

    def _save_state(self, state: Dict[str, Any]) -> None:
        json_dump_atomic(self.state_path, state)

    def _load_existing_summaries(self) -> List[Dict[str, Any]]:
        obj = load_json_or_default(self.summary_memories_path, [])
        if not isinstance(obj, list):
            return []
        return [x for x in obj if isinstance(x, dict)]

    def _save_summaries(self, rows: List[Dict[str, Any]]) -> None:
        json_dump_atomic(self.summary_memories_path, rows or [])

    @staticmethod
    def _entry_brief(row: Dict[str, Any], max_len: int = 180) -> Dict[str, Any]:
        content = str(row.get("content", "")).strip()
        if len(content) > max_len:
            content = content[: max_len - 3] + "..."
        kws = row.get("keywords") or []
        if not isinstance(kws, list):
            kws = []
        return {
            "id": str(row.get("id", "")),
            "type": str(row.get("type", "")),
            "content": content,
            "keywords": [str(x).strip() for x in kws if str(x).strip()][:10],
            "confidence": clamp_float(row.get("confidence", 0.7), default=0.7),
            "source": str(row.get("source", "")),
            "hit_count": int(row.get("hit_count", 0)),
            "updated_at": float(row.get("updated_at", 0.0)),
        }

    def _run_llm_distill(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        # Keep prompt bounded to avoid model context overflow.
        payload_txt = json.dumps(payload, ensure_ascii=False)
        if len(payload_txt) > self._DISTILL_MAX_INPUT_CHARS:
            compact = dict(payload)
            compact["new_rows"] = list((payload.get("new_rows") or [])[: max(1, self._DISTILL_BATCH_SIZE // 2)])
            compact["source_rows"] = list((payload.get("source_rows") or [])[: max(0, self._DISTILL_BATCH_SIZE // 4)])
            payload = compact
            payload_txt = json.dumps(payload, ensure_ascii=False)

        if self.problem_solver is not None:
            try:
                out_txt = self.problem_solver.extract_distilled_memories(
                    source_payload=payload_txt,
                    requested_max=self.max_new_memories,
                )
                parsed = parse_json_object_from_text(out_txt)
                if isinstance(parsed, dict):
                    return parsed
            except Exception as e:
                logger.warning("MemoryDistiller extractor call failed: %s", e)

        user_prompt = self.prompt_loader.render(
            self.prompt_id,
            task_values={
                "source_payload": payload_txt,
                "requested_max": str(self.max_new_memories),
            },
            static_values={},
            strict=False,
        )
        try:
            out = self.llm.invoke(
                [
                    SystemMessage(content="You are a precise JSON generator."),
                    HumanMessage(content=user_prompt),
                ],
                max_tokens=self._DISTILL_MAX_OUTPUT_TOKENS,
            )
            text = getattr(out, "content", str(out))
        except Exception as e:
            logger.warning("MemoryDistiller LLM call failed: %s", e)
            return {}
        parsed = parse_json_object_from_text(text)
        if not parsed:
            logger.warning("MemoryDistiller got non-JSON output.")
            return {}
        return parsed

    @staticmethod
    def _group_for_summary(memories: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        groups: Dict[str, List[Dict[str, Any]]] = {"guideline": [], "anti_pattern": [], "canonical_map": []}
        for m in memories or []:
            t = str(m.get("type", "")).strip()
            if t in groups:
                groups[t].append(
                    {
                        "id": str(m.get("id", "")),
                        "content": str(m.get("content", "")),
                        "keywords": list(m.get("keywords") or []),
                        "confidence": float(m.get("confidence", 0.7) or 0.7),
                    }
                )
        return groups

    def _summarize_distilled_memories(
        self,
        *,
        state: Dict[str, Any],
        cycle_id: int,
        new_memories: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if self.problem_solver is None:
            return self._load_existing_summaries()
        grouped = self._group_for_summary(new_memories)
        active_types = [t for t, rows in grouped.items() if rows]
        if not active_types:
            return self._load_existing_summaries()

        existing = self._load_existing_summaries()
        existing_map = {
            str(x.get("type", "")).strip(): x for x in existing if str(x.get("type", "")).strip() in self.SUMMARY_TYPES
        }

        out = self.problem_solver.summarize_distilled_memories(
            grouped_memories_json=json.dumps(grouped, ensure_ascii=False),
            existing_summaries_json=json.dumps(existing, ensure_ascii=False),
        )
        parsed = parse_json_object_from_text(out)
        if not isinstance(parsed, dict):
            return existing
        rows = parsed.get("summaries") or []
        if not isinstance(rows, list):
            return existing

        summary_entry_ids = dict(state.get("summary_entry_ids") or {})
        next_map = dict(existing_map)

        for row in rows:
            if not isinstance(row, dict):
                continue
            t = str(row.get("type", "")).strip()
            if t not in self.SUMMARY_TYPES:
                continue
            content = str(row.get("content", "")).strip()
            if not content:
                continue
            keywords = [str(x).strip() for x in (row.get("keywords") or []) if str(x).strip()][:8]
            confidence = clamp_float(row.get("confidence", 0.7), default=0.7)
            summary_id = str(row.get("summary_id", "")).strip() or f"distill_summary_{t}"
            mem_type = self.SUMMARY_MEMORY_TYPE_MAP[t]
            payload = {
                "type": mem_type,
                "content": content,
                "keywords": keywords,
                "confidence": confidence,
                "source": f"memory_distill_summary_cycle_{cycle_id}",
                "memory_scope": "shared",
            }
            eid = summary_entry_ids.get(t, "")
            if eid:
                try:
                    self.memory_store.update(
                        eid,
                        type=mem_type,
                        content=content,
                        keywords=keywords,
                        confidence=confidence,
                        source=f"memory_distill_summary_cycle_{cycle_id}",
                        memory_scope="shared",
                    )
                    payload["id"] = eid
                except Exception:
                    payload["id"] = ""
            if not payload.get("id"):
                new_id = self.memory_store.add(payload)
                if new_id:
                    payload["id"] = new_id
                    summary_entry_ids[t] = new_id

            payload["summary_id"] = summary_id
            payload["type"] = t
            next_map[t] = payload

        state["summary_entry_ids"] = summary_entry_ids
        out_rows = [next_map[k] for k in sorted(next_map.keys()) if k in self.SUMMARY_TYPES]
        self._save_summaries(out_rows)
        return out_rows

    @staticmethod
    def _norm_text(text: str) -> str:
        s = re.sub(r"\s+", " ", str(text or "").strip().lower())
        return s

    @staticmethod
    def _jaccard(a: List[str], b: List[str]) -> float:
        sa = {str(x).strip().lower() for x in (a or []) if str(x).strip()}
        sb = {str(x).strip().lower() for x in (b or []) if str(x).strip()}
        if not sa or not sb:
            return 0.0
        inter = len(sa & sb)
        union = len(sa | sb)
        return (inter / union) if union > 0 else 0.0

    @staticmethod
    def _cosine(a: Optional[List[float]], b: Optional[List[float]]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = 0.0
        na = 0.0
        nb = 0.0
        for x, y in zip(a, b):
            dot += x * y
            na += x * x
            nb += y * y
        if na <= 1e-12 or nb <= 1e-12:
            return 0.0
        return dot / (math.sqrt(na) * math.sqrt(nb))

    def _merge_rows_for_distill(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not rows:
            return []

        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for r in rows:
            key = f"{str(r.get('type','')).strip()}||{str(r.get('memory_scope','shared')).strip().lower()}"
            grouped.setdefault(key, []).append(r)

        out: List[Dict[str, Any]] = []
        for bucket_rows in grouped.values():
            vectors: List[Optional[List[float]]] = [None] * len(bucket_rows)
            try:
                emb_model = getattr(self.memory_store, "embedding_model", None)
                if emb_model is not None and bucket_rows:
                    vectors = emb_model.embed_documents([str(r.get("content", "")) for r in bucket_rows])
            except Exception:
                vectors = [None] * len(bucket_rows)

            clusters: List[Dict[str, Any]] = []
            for i, r in enumerate(bucket_rows):
                rid = str(r.get("id", "")).strip()
                kws = [str(x).strip() for x in (r.get("keywords") or []) if str(x).strip()]
                txt = self._norm_text(str(r.get("content", "")))
                vec = vectors[i] if i < len(vectors) else None
                matched = None
                for c in clusters:
                    if txt and txt == c["txt"]:
                        matched = c
                        break
                    if self._jaccard(kws, c["keywords"]) >= 0.6:
                        matched = c
                        break
                    if self._cosine(vec, c.get("vec")) >= 0.92:
                        matched = c
                        break
                if matched is None:
                    clusters.append(
                        {
                            "row": dict(r),
                            "ids": [rid] if rid else [],
                            "txt": txt,
                            "keywords": kws[:12],
                            "vec": vec,
                        }
                    )
                    continue
                merged_row = matched["row"]
                if len(str(r.get("content", ""))) > len(str(merged_row.get("content", ""))):
                    merged_row["content"] = str(r.get("content", ""))
                mk = list(dict.fromkeys((matched["keywords"] or []) + kws))[:12]
                matched["keywords"] = mk
                merged_row["keywords"] = mk
                if rid:
                    matched["ids"].append(rid)
                merged_row["confidence"] = max(
                    float(merged_row.get("confidence", 0.0) or 0.0),
                    float(r.get("confidence", 0.0) or 0.0),
                )
                matched["row"] = merged_row

            for c in clusters:
                row = dict(c["row"])
                row["distilled_source_ids"] = list(dict.fromkeys([x for x in c["ids"] if x]))
                out.append(row)
        return out

    @staticmethod
    def _normalize_item(item: Any) -> Optional[Dict[str, Any]]:
        if isinstance(item, str):
            content = item.strip()
            if not content:
                return None
            return {
                "content": content,
                "keywords": [],
                "confidence": 0.7,
                "reuse_level": "domain_reusable",
                "memory_scope": "shared",
                "applicable_tasks": [],
                "applicable_doc_types": [],
                "language": "",
                "tags": [],
                "when_to_apply": "",
                "when_not_to_apply": "",
                "evidence_refs": [],
                "ttl_days": 30,
            }

        if not isinstance(item, dict):
            return None

        content = (
            str(item.get("content", "")).strip()
            or str(item.get("rule", "")).strip()
            or str(item.get("description", "")).strip()
        )
        if not content:
            return None

        kws = item.get("keywords") or item.get("entities") or item.get("anchors") or []
        if isinstance(kws, str):
            kws = [kws]
        if not isinstance(kws, list):
            kws = []
        keywords = [str(x).strip() for x in kws if str(x).strip()][:12]
        confidence = clamp_float(item.get("confidence", 0.7), default=0.7)
        reuse_level = str(item.get("reuse_level", "domain_reusable")).strip().lower() or "domain_reusable"
        if reuse_level not in {"task_only", "domain_reusable", "global_reusable"}:
            reuse_level = "domain_reusable"
        memory_scope = str(item.get("memory_scope", "shared")).strip().lower() or "shared"
        if memory_scope not in {"entity_extraction", "relation_extraction", "shared"}:
            memory_scope = "shared"

        def _as_list_str(v: Any, limit: int = 12) -> List[str]:
            if isinstance(v, str):
                v = [v]
            if not isinstance(v, list):
                return []
            out = []
            for x in v:
                s = str(x).strip()
                if s:
                    out.append(s)
                if len(out) >= limit:
                    break
            return out

        return {
            "content": content,
            "keywords": keywords,
            "confidence": confidence,
            "reuse_level": reuse_level,
            "memory_scope": memory_scope,
            "applicable_tasks": _as_list_str(item.get("applicable_tasks"), limit=8),
            "applicable_doc_types": _as_list_str(item.get("applicable_doc_types"), limit=8),
            "language": str(item.get("language", "")).strip().lower(),
            "tags": _as_list_str(item.get("tags"), limit=12),
            "when_to_apply": str(item.get("when_to_apply", "")).strip(),
            "when_not_to_apply": str(item.get("when_not_to_apply", "")).strip(),
            "evidence_refs": _as_list_str(item.get("evidence_refs"), limit=20),
            "ttl_days": max(1, int(float(item.get("ttl_days", 30)))),
        }

    def maybe_run(
        self,
        *,
        docs_processed: int = 0,
        force: bool = False,
        reason: str = "",
        extra_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        report: Dict[str, Any] = {
            "ran": False,
            "reason": reason or "periodic",
            "trigger": "",
            "input_new_entries": 0,
            "added_memories": 0,
        }
        if not self.enabled:
            report["trigger"] = "disabled"
            return report
        if self.memory_store is None:
            report["trigger"] = "no_memory_store"
            return report

        state = self._load_state()
        state["docs_since_last"] = int(state.get("docs_since_last", 0)) + max(0, int(docs_processed))
        all_source_rows = self.memory_store.get_entries(
            exclude_types=self.DERIVED_TYPES,
            limit=None,
            newest_first=True,
        )
        distilled_raw_ids = {str(x).strip() for x in (state.get("distilled_raw_ids") or []) if str(x).strip()}
        undistilled_rows = [
            r
            for r in all_source_rows
            if str(r.get("id", "")).strip() and str(r.get("id", "")).strip() not in distilled_raw_ids
        ]
        undistilled_count = len(undistilled_rows)
        report["input_new_entries"] = int(undistilled_count)

        trigger = ""
        if force:
            trigger = "force"
        elif state["docs_since_last"] >= self.cycle_every_n_docs:
            trigger = "doc_cycle"
        elif undistilled_count >= self.min_new_entries:
            trigger = "new_entries"

        if not trigger:
            self._save_state(state)
            report["trigger"] = "not_reached"
            return report

        if not undistilled_rows:
            state["docs_since_last"] = 0
            self._save_state(state)
            report["trigger"] = "no_undistilled_rows"
            return report

        cycle_id = int(state.get("cycle_id", 0)) + 1
        added = 0
        saved_memories: List[Dict[str, Any]] = []
        merged_rows = self._merge_rows_for_distill(undistilled_rows)
        batch_size = self._DISTILL_BATCH_SIZE
        total_batches = (len(merged_rows) + batch_size - 1) // batch_size
        processed_raw_ids: List[str] = []
        succeeded_batches = 0

        for bi in range(total_batches):
            batch = merged_rows[bi * batch_size : (bi + 1) * batch_size]
            payload = {
                "trigger": trigger,
                "new_rows": [self._entry_brief(x) for x in batch],
                "source_rows": [],
                "extra_context": {
                    **(extra_context or {}),
                    "batch_index": bi + 1,
                    "batch_total": total_batches,
                },
            }
            distilled = self._run_llm_distill(payload)
            if not distilled:
                continue

            source = f"memory_distill_cycle_{cycle_id}_batch_{bi+1}"
            outputs: List[Dict[str, Any]] = []
            mapping = [
                ("guidelines", "guideline"),
                ("anti_patterns", "anti_pattern"),
                ("canonical_maps", "canonical_map"),
            ]
            for key, typ in mapping:
                vals = distilled.get(key) or []
                if not isinstance(vals, list):
                    continue
                for raw in vals:
                    row = self._normalize_item(raw)
                    if row is None:
                        continue
                    row["type"] = typ
                    row["source"] = source
                    outputs.append(row)

            outputs = outputs[: self.max_new_memories]
            for row in outputs:
                eid = self.memory_store.add(row)
                if eid:
                    added += 1
                    saved = dict(row)
                    saved["id"] = eid
                    saved_memories.append(saved)

            for x in batch:
                for sid in (x.get("distilled_source_ids") or [x.get("id")]):
                    sid_txt = str(sid or "").strip()
                    if sid_txt:
                        processed_raw_ids.append(sid_txt)
            succeeded_batches += 1

        try:
            self.memory_store.flush()
        except Exception as e:
            logger.warning("MemoryDistiller flush failed: %s", e)

        now = time.time()
        state["cycle_id"] = cycle_id
        state["docs_since_last"] = 0
        if processed_raw_ids:
            merged_ids = list(distilled_raw_ids | set(processed_raw_ids))
            state["distilled_raw_ids"] = merged_ids
        self._save_state(state)

        report.update(
            {
                "ran": succeeded_batches > 0,
                "trigger": trigger,
                "cycle_id": cycle_id,
                "added_memories": added,
                "timestamp": now,
                "batch_total": total_batches,
                "batch_succeeded": succeeded_batches,
            }
        )
        memories_payload = {
            "cycle_id": cycle_id,
            "trigger": trigger,
            "timestamp": now,
            "source": f"memory_distill_cycle_{cycle_id}",
            "batch_total": total_batches,
            "batch_succeeded": succeeded_batches,
            "memories": saved_memories,
        }
        summary_rows = self._summarize_distilled_memories(
            state=state,
            cycle_id=cycle_id,
            new_memories=saved_memories,
        )
        memories_payload["summary_memories"] = summary_rows
        self._save_state(state)
        json_dump_atomic(self.latest_memories_path, memories_payload)
        logger.info(
            "MemoryDistiller cycle=%d trigger=%s undistilled=%d batches=%d/%d added=%d",
            cycle_id,
            trigger,
            undistilled_count,
            succeeded_batches,
            total_batches,
            added,
        )
        return report

    def clear_artifacts(self) -> None:
        for p in [
            self.state_path,
            self.latest_memories_path,
            self.summary_memories_path,
        ]:
            if os.path.exists(p):
                try:
                    os.remove(p)
                except Exception as e:
                    logger.warning("MemoryDistiller clear_artifacts failed: %s (%s)", p, e)

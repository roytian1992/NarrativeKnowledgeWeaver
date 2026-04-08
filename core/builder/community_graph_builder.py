# -*- coding: utf-8 -*-
from __future__ import annotations

import hashlib
import json
import logging
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from core import KAGConfig
from core.builder.manager.document_manager import DocumentParser
from core.model_providers.openai_llm import OpenAILLM
from core.models.data import Document
from core.storage.graph_store import GraphStore
from core.storage.vector_store import VectorStore
from core.utils.function_manager import run_concurrent_with_retries
from core.utils.general_utils import (
    _to_vec_list,
    dedupe_list,
    ensure_dir,
    json_dump_atomic,
    safe_dict,
    safe_list,
    safe_str,
    stable_relation_id,
)
from core.utils.graph_query_utils import GraphQueryUtils

logger = logging.getLogger(__name__)

P_COMMUNITY_CONTAINS = "COMMUNITY_CONTAINS"
P_COMMUNITY_PARENT_OF = "COMMUNITY_PARENT_OF"


class CommunityGraphBuilder:
    """Build Leiden communities and GraphRAG-style community reports on top of the base KG."""

    def __init__(self, config: KAGConfig, *, doc_type: Optional[str] = None) -> None:
        self.config = config
        self.doc_type = doc_type or config.global_config.doc_type
        self.cfg = config.community_graph_builder
        self.base_dir = safe_str(getattr(self.cfg, "file_path", "") or "data/community_graph")
        self.max_workers = max(1, int(getattr(self.cfg, "max_workers", 16) or 16))

        self.graph_store = GraphStore(config)
        self.graph_query_utils = GraphQueryUtils(self.graph_store, doc_type=self.doc_type)
        self.llm = OpenAILLM(config)
        self.document_parser = DocumentParser(config, self.llm)
        self.community_summary_store = VectorStore(config, "community")

        self.out_global_dir = ensure_dir(os.path.join(self.base_dir, "global"))
        self.assignments_path = os.path.join(self.out_global_dir, "community_assignments.json")
        self.communities_path = os.path.join(self.out_global_dir, "communities.json")
        self.membership_edges_path = os.path.join(self.out_global_dir, "community_membership_edges.json")
        self.hierarchy_edges_path = os.path.join(self.out_global_dir, "community_hierarchy_edges.json")
        self.reports_path = os.path.join(self.out_global_dir, "community_reports.json")

    def close(self) -> None:
        try:
            self.graph_store.close()
        except Exception:
            pass

    def clear_community_aggregation(self) -> Dict[str, int]:
        deleted = self.graph_query_utils.delete_nodes_by_labels(["Community"])
        cleared = self.graph_query_utils.clear_community_assignments(
            write_property=self.cfg.write_property,
            intermediate_property=self.cfg.intermediate_property,
        )
        try:
            self.community_summary_store.delete_collection()
            summary_index_cleared = 1
        except Exception:
            summary_index_cleared = 0
        return {
            "deleted_community_nodes": deleted,
            "cleared_assignments": cleared,
            "cleared_summary_index": summary_index_cleared,
        }

    def clear_narrative_aggregation(self) -> Dict[str, int]:
        deleted = self.graph_query_utils.delete_nodes_by_labels(["Episode", "Storyline"])
        return {"deleted_narrative_nodes": deleted}

    @staticmethod
    def _primary_label(labels: List[str]) -> str:
        order = {
            "Event": 0,
            "Occasion": 1,
            "Character": 2,
            "Location": 3,
            "Object": 4,
            "TimePoint": 5,
            "Organization": 6,
            "Concept": 7,
        }
        cleaned = [safe_str(x) for x in (labels or []) if safe_str(x) and safe_str(x) != "Entity"]
        if not cleaned:
            return "Entity"
        cleaned.sort(key=lambda x: (order.get(x, 999), x))
        return cleaned[0]

    @staticmethod
    def _parse_report(raw: str, fallback_text: str = "") -> Dict[str, Any]:
        try:
            data = json.loads(raw) if isinstance(raw, str) else (raw or {})
        except Exception:
            data = {}
        title = safe_str(data.get("title")).strip()
        summary = safe_str(data.get("summary")).strip()
        try:
            rating = float(data.get("rating", 0.0) or 0.0)
        except Exception:
            rating = 0.0
        rating = min(10.0, max(0.0, rating))
        rating_explanation = safe_str(data.get("rating_explanation")).strip()
        findings: List[Dict[str, str]] = []
        for item in safe_list(data.get("findings")):
            if not isinstance(item, dict):
                continue
            finding_summary = safe_str(item.get("summary")).strip()
            explanation = safe_str(item.get("explanation")).strip()
            if not finding_summary or not explanation:
                continue
            findings.append({"summary": finding_summary, "explanation": explanation})
        if not findings and summary:
            findings = [{"summary": "社区概览", "explanation": summary}]
        if not summary:
            summary = safe_str(fallback_text).strip()
        return {
            "title": title,
            "summary": summary,
            "rating": rating,
            "rating_explanation": rating_explanation,
            "findings": findings,
        }

    @staticmethod
    def _community_key(level: int, raw_id: int) -> str:
        return f"{level}:{raw_id}"

    @staticmethod
    def _stable_community_id(level: int, member_ids: List[str]) -> str:
        payload = f"{level}|{'|'.join(sorted(member_ids))}"
        suffix = hashlib.md5(payload.encode("utf-8")).hexdigest()[:12]
        return f"com_l{level}_{suffix}"

    @staticmethod
    def _build_report_full_content(report: Dict[str, Any]) -> str:
        lines: List[str] = []
        title = safe_str(report.get("title")).strip()
        summary = safe_str(report.get("summary")).strip()
        rating = report.get("rating", 0.0)
        rating_explanation = safe_str(report.get("rating_explanation")).strip()
        findings = safe_list(report.get("findings"))
        if title:
            lines.append(f"# {title}")
        if summary:
            lines.append(summary)
        if rating_explanation:
            lines.append(f"Importance Rating: {rating}")
            lines.append(f"Rating Explanation: {rating_explanation}")
        if findings:
            lines.append("Detailed Findings:")
            for item in findings:
                if not isinstance(item, dict):
                    continue
                finding_summary = safe_str(item.get("summary")).strip()
                explanation = safe_str(item.get("explanation")).strip()
                if finding_summary:
                    lines.append(f"## {finding_summary}")
                if explanation:
                    lines.append(explanation)
        return "\n\n".join([x for x in lines if safe_str(x).strip()]).strip()

    @staticmethod
    def _clip_join_sections(sections: List[str], max_chars: int) -> str:
        out: List[str] = []
        used = 0
        limit = max(500, int(max_chars or 12000))
        for section in sections:
            chunk = safe_str(section).strip()
            if not chunk:
                continue
            extra = len(chunk) + (2 if out else 0)
            if used + extra <= limit:
                out.append(chunk)
                used += extra
                continue
            remaining = limit - used - (2 if out else 0)
            if remaining > 120:
                out.append(chunk[:remaining].rstrip())
            break
        return "\n\n".join(out).strip()

    def _select_member_preview(self, members: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
        rows = []
        for item in members:
            label = self._primary_label(item.get("labels") or [])
            rows.append(
                {
                    "node_id": safe_str(item.get("node_id")),
                    "name": safe_str(item.get("name")),
                    "description": safe_str(item.get("description")),
                    "label": label,
                    "source_documents": safe_list(item.get("source_documents")),
                }
            )
        rows.sort(key=lambda x: (self._primary_label([x["label"]]), x["name"], x["node_id"]))
        return rows[: max(1, int(limit or 1))]

    def _fetch_node_details_by_ids(self, node_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        ids = [safe_str(x) for x in node_ids if safe_str(x)]
        if not ids:
            return {}
        return self.graph_query_utils.get_node_details_by_ids(ids)

    def _build_leaf_context(self, card: Dict[str, Any]) -> str:
        sections: List[str] = []
        lines = [
            "-----CommunityMetadata-----",
            "field,value",
            f"level,{int(card.get('level', 0) or 0)}",
            f"raw_community_id,{int(card.get('raw_community_id', 0) or 0)}",
            f"member_count,{int(card.get('member_count', 0) or 0)}",
        ]
        sections.append("\n".join(lines))

        type_counts = safe_dict(card.get("type_counts"))
        if type_counts:
            type_lines = ["-----MemberTypeCounts-----", "type,count"]
            for label in sorted(type_counts.keys()):
                type_lines.append(f"{label},{int(type_counts[label] or 0)}")
            sections.append("\n".join(type_lines))

        entity_lines = ["-----Entities-----", "id,name,type,description"]
        for member in self._select_member_preview(safe_list(card.get("members")), int(self.cfg.summary_max_members or 40)):
            entity_lines.append(
                ",".join(
                    [
                        member["node_id"].replace(",", " "),
                        (member["name"] or "").replace(",", " "),
                        member["label"].replace(",", " "),
                        (member["description"] or "").replace(",", " "),
                    ]
                )
            )
        sections.append("\n".join(entity_lines))

        internal_relations = safe_list(card.get("internal_relations"))
        if internal_relations:
            rel_lines = ["-----Relationships-----", "id,source,target,predicate,description"]
            for rel in internal_relations[: max(10, int(self.cfg.summary_max_relations or 80))]:
                rel_lines.append(
                    ",".join(
                        [
                            safe_str(rel.get("id")).replace(",", " "),
                            (safe_str(rel.get("subject_name")) or safe_str(rel.get("subject_id"))).replace(",", " "),
                            (safe_str(rel.get("object_name")) or safe_str(rel.get("object_id"))).replace(",", " "),
                            safe_str(rel.get("predicate") or rel.get("relation_type")).replace(",", " "),
                            safe_str(rel.get("description")).replace(",", " "),
                        ]
                    )
                )
            sections.append("\n".join(rel_lines))

        return self._clip_join_sections(sections, int(self.cfg.report_max_input_chars or 12000))

    def _collect_internal_relation_source_documents(
        self,
        node_ids: List[str],
        *,
        exclude_relation_types: Optional[List[str]] = None,
    ) -> List[str]:
        clean_ids = [safe_str(x) for x in (node_ids or []) if safe_str(x)]
        if not clean_ids:
            return []
        return self.graph_query_utils.collect_internal_relation_source_documents(
            clean_ids,
            exclude_relation_types=[safe_str(x) for x in safe_list(exclude_relation_types) if safe_str(x)],
        )

    def _collect_community_source_documents(
        self,
        members: List[Dict[str, Any]],
        *,
        node_ids: List[str],
        exclude_relation_types: Optional[List[str]] = None,
    ) -> List[str]:
        source_documents: List[str] = []
        for member in members:
            source_documents.extend([safe_str(x) for x in safe_list(member.get("source_documents")) if safe_str(x)])
        source_documents.extend(
            self._collect_internal_relation_source_documents(
                node_ids,
                exclude_relation_types=exclude_relation_types,
            )
        )
        return dedupe_list(source_documents)

    def _build_parent_context(self, card: Dict[str, Any], key2card: Dict[str, Dict[str, Any]]) -> str:
        sections: List[str] = []
        lines = [
            "-----CommunityMetadata-----",
            "field,value",
            f"level,{int(card.get('level', 0) or 0)}",
            f"raw_community_id,{int(card.get('raw_community_id', 0) or 0)}",
            f"member_count,{int(card.get('member_count', 0) or 0)}",
        ]
        sections.append("\n".join(lines))

        child_cards: List[Dict[str, Any]] = []
        for child_key in safe_list(card.get("child_keys")):
            child = key2card.get(child_key)
            if child:
                child_cards.append(child)
        child_cards.sort(
            key=lambda x: (
                -float(safe_dict(x.get("report")).get("rating", 0.0) or 0.0),
                -int(x.get("member_count", 0) or 0),
                safe_str(x.get("community_id")),
            )
        )
        if child_cards:
            header = ["-----ChildCommunityReports-----", "community_id,title,rating,summary"]
            for child in child_cards[: max(1, int(self.cfg.report_max_child_reports or 8))]:
                report = safe_dict(child.get("report"))
                header.append(
                    ",".join(
                        [
                            safe_str(child.get("community_id")).replace(",", " "),
                            safe_str(report.get("title")).replace(",", " "),
                            str(report.get("rating", 0.0)).replace(",", " "),
                            safe_str(report.get("summary")).replace(",", " "),
                        ]
                    )
                )
            sections.append("\n".join(header))

            finding_lines = ["-----ChildFindings-----"]
            max_child_findings = max(1, int(self.cfg.report_max_child_findings or 3))
            for child in child_cards[: max(1, int(self.cfg.report_max_child_reports or 8))]:
                report = safe_dict(child.get("report"))
                finding_lines.append(
                    f"[Child {safe_str(child.get('community_id'))}] {safe_str(report.get('title'))}"
                )
                for finding in safe_list(report.get("findings"))[:max_child_findings]:
                    if not isinstance(finding, dict):
                        continue
                    finding_lines.append(f"- {safe_str(finding.get('summary'))}: {safe_str(finding.get('explanation'))}")
            sections.append("\n".join(finding_lines))

        sections.append(self._build_leaf_context(card))
        return self._clip_join_sections(sections, int(self.cfg.report_max_input_chars or 12000))

    def _generate_reports_for_level(self, cards: List[Dict[str, Any]], level: int) -> None:
        if not cards:
            return

        def _task(card: Dict[str, Any]) -> Dict[str, Any]:
            context = safe_str(card.get("report_context")).strip()
            raw = self.document_parser.generate_community_report(
                text=context,
                max_length=int(self.cfg.report_max_length or 1800),
                max_findings=int(self.cfg.report_max_findings or 6),
            )
            report = self._parse_report(raw, fallback_text=context[: int(self.cfg.summary_max_length or 300)])
            report["full_content"] = self._build_report_full_content(report)
            return report

        timeout_default = max(
            120.0,
            float(getattr(self.config.knowledge_graph_builder, "per_task_timeout", 2400) or 2400) / 2.0,
        )
        results_map, _failed_indices = run_concurrent_with_retries(
            items=cards,
            task_fn=_task,
            per_task_timeout=timeout_default,
            max_retry_rounds=2,
            max_in_flight=self.max_workers,
            max_workers=self.max_workers,
            thread_name_prefix=f"community_report_l{level}",
            desc_prefix=f"Generating community reports / level {level}",
            treat_empty_as_failure=True,
            is_empty_fn=lambda x: not isinstance(x, dict) or not safe_str(x.get("summary")).strip(),
        )

        for idx, card in enumerate(cards):
            report = safe_dict(results_map.get(idx))
            if not report:
                fallback_summary = safe_str(card.get("report_context"))[: int(self.cfg.summary_max_length or 300)].strip()
                report = {
                    "title": safe_str(card.get("fallback_title")),
                    "summary": fallback_summary,
                    "rating": 0.0,
                    "rating_explanation": "",
                    "findings": [{"summary": "上下文回退", "explanation": fallback_summary}] if fallback_summary else [],
                    "full_content": fallback_summary,
                }
            card["report"] = report

    def _build_communities_from_assignments(
        self,
        assignments: List[Dict[str, Any]],
        *,
        exclude_relation_types: List[str],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        buckets: Dict[str, Dict[str, Any]] = {}
        membership_pairs: Dict[Tuple[str, str], Dict[str, Any]] = {}
        hierarchy_pairs: Dict[Tuple[str, str], Dict[str, Any]] = {}
        all_node_ids: List[str] = []

        for item in assignments:
            node_id = safe_str(item.get("node_id"))
            path = [int(x) for x in safe_list(item.get("community_path")) if str(x).strip()]
            if not node_id or not path:
                continue
            all_node_ids.append(node_id)
            for idx, raw_cid in enumerate(path):
                level = idx + 1
                key = self._community_key(level, raw_cid)
                bucket = buckets.setdefault(
                    key,
                    {"level": level, "raw_community_id": int(raw_cid), "members": []},
                )
                bucket["members"].append(item)

            leaf_key = self._community_key(1, path[0])
            membership_pairs[(leaf_key, node_id)] = {"leaf_key": leaf_key, "node_id": node_id}
            for idx in range(len(path) - 1):
                child_key = self._community_key(idx + 1, path[idx])
                parent_key = self._community_key(idx + 2, path[idx + 1])
                hierarchy_pairs[(parent_key, child_key)] = {"parent_key": parent_key, "child_key": child_key}

        node_info_map = self._fetch_node_details_by_ids(dedupe_list(all_node_ids))
        community_cards: List[Dict[str, Any]] = []
        key2entity: Dict[str, Dict[str, Any]] = {}
        key2card: Dict[str, Dict[str, Any]] = {}

        keys_sorted = sorted(
            buckets.keys(),
            key=lambda x: (int(x.split(":", 1)[0]), int(x.split(":", 1)[1])),
        )

        def _one_community(key: str) -> Dict[str, Any]:
            bucket = buckets[key]
            members = bucket["members"]
            level = int(bucket["level"])
            raw_community_id = int(bucket["raw_community_id"])
            member_ids = [safe_str(m.get("node_id")) for m in members if safe_str(m.get("node_id"))]
            member_ids = dedupe_list(member_ids)
            community_id = self._stable_community_id(level, member_ids)
            detailed_members = []
            for member_id in member_ids:
                detailed = safe_dict(node_info_map.get(member_id))
                if detailed:
                    detailed_members.append(detailed)
            preview = self._select_member_preview(detailed_members, limit=6)
            preview_names = [safe_str(x.get("name")) for x in preview if safe_str(x.get("name"))]
            type_counts: Dict[str, int] = defaultdict(int)
            for member in detailed_members:
                type_counts[self._primary_label(member.get("labels") or [])] += 1
            internal_relations = self.graph_query_utils.fetch_internal_relations_by_node_ids(
                member_ids,
                exclude_relation_types=exclude_relation_types,
                limit=max(10, int(self.cfg.summary_max_relations or 80)),
            )
            source_documents = self._collect_community_source_documents(
                detailed_members,
                node_ids=member_ids,
                exclude_relation_types=exclude_relation_types,
            )
            report_id = f"{community_id}_report"
            name_hint = " / ".join(preview_names[:3]) or f"Community {raw_community_id}"
            name_hint = name_hint[:60].strip() or f"Community L{level} {raw_community_id}"
            return {
                "key": key,
                "community_id": community_id,
                "report_id": report_id,
                "level": level,
                "raw_community_id": raw_community_id,
                "member_ids": member_ids,
                "member_count": len(member_ids),
                "members": detailed_members,
                "type_counts": dict(type_counts),
                "top_members": preview_names,
                "source_documents": source_documents,
                "internal_relations": internal_relations,
                "fallback_title": f"Community L{level}: {name_hint}",
                "child_keys": [],
            }

        prepare_results_map, _prepare_failed = run_concurrent_with_retries(
            items=keys_sorted,
            task_fn=_one_community,
            per_task_timeout=max(120.0, float(getattr(self.config.knowledge_graph_builder, "per_task_timeout", 2400) or 2400) / 6.0),
            max_retry_rounds=2,
            max_in_flight=self.max_workers,
            max_workers=self.max_workers,
            thread_name_prefix="community_card",
            desc_prefix="Preparing communities",
            treat_empty_as_failure=True,
            is_empty_fn=lambda x: not isinstance(x, dict),
        )
        community_cards = [
            prepare_results_map[idx]
            for idx in sorted(prepare_results_map.keys())
            if isinstance(prepare_results_map.get(idx), dict)
        ]
        for card in community_cards:
            key2card[card["key"]] = card
        for item in hierarchy_pairs.values():
            parent = key2card.get(item["parent_key"])
            if parent is not None:
                parent.setdefault("child_keys", []).append(item["child_key"])

        levels_sorted = sorted({int(card.get("level", 0) or 0) for card in community_cards})
        for level in levels_sorted:
            level_cards = [card for card in community_cards if int(card.get("level", 0) or 0) == level]
            for card in level_cards:
                child_keys = safe_list(card.get("child_keys"))
                if child_keys:
                    card["report_context"] = self._build_parent_context(card, key2card)
                else:
                    card["report_context"] = self._build_leaf_context(card)
            self._generate_reports_for_level(level_cards, level)

        communities: List[Dict[str, Any]] = []
        report_docs: List[Dict[str, Any]] = []
        for built in community_cards:
            report = safe_dict(built.get("report"))
            entity_name = safe_str(report.get("title")).strip() or safe_str(built.get("fallback_title")).strip()
            entity_description = safe_str(report.get("summary")).strip()
            entity = {
                "id": safe_str(built.get("community_id")),
                "name": entity_name,
                "type": ["Community"],
                "aliases": [],
                "description": entity_description,
                "scope": "global",
                "source_documents": safe_list(built.get("source_documents")),
                "properties": {
                    "level": int(built.get("level", 0) or 0),
                    "raw_community_id": int(built.get("raw_community_id", 0) or 0),
                    "member_count": int(built.get("member_count", 0) or 0),
                    "member_type_counts": safe_dict(built.get("type_counts")),
                    "top_members": safe_list(built.get("top_members")),
                    "report_id": safe_str(built.get("report_id")),
                    "rating": float(report.get("rating", 0.0) or 0.0),
                    "rating_explanation": safe_str(report.get("rating_explanation")),
                },
            }
            report_doc = {
                "id": safe_str(built.get("report_id")),
                "content": safe_str(report.get("full_content")).strip(),
                "metadata": {
                    "community_id": safe_str(built.get("community_id")),
                    "community_name": entity_name,
                    "title": entity_name,
                    "summary": entity_description,
                    "level": int(built.get("level", 0) or 0),
                    "member_count": int(built.get("member_count", 0) or 0),
                    "source_documents": safe_list(built.get("source_documents")),
                    "top_members": safe_list(built.get("top_members")),
                    "rating": float(report.get("rating", 0.0) or 0.0),
                    "rating_explanation": safe_str(report.get("rating_explanation")),
                    "findings": safe_list(report.get("findings")),
                },
            }
            communities.append(entity)
            report_docs.append(report_doc)
            key2entity[safe_str(built.get("key"))] = entity

        membership_edges: List[Dict[str, Any]] = []
        for item in membership_pairs.values():
            community = key2entity.get(item["leaf_key"])
            node_id = item["node_id"]
            if not community or not node_id:
                continue
            membership_edges.append(
                {
                    "id": stable_relation_id(community["id"], P_COMMUNITY_CONTAINS, node_id, prefix="rel_com_contains_"),
                    "subject_id": community["id"],
                    "object_id": node_id,
                    "predicate": P_COMMUNITY_CONTAINS,
                    "relation_name": P_COMMUNITY_CONTAINS,
                    "description": "Community contains this member node.",
                    "source_documents": community.get("source_documents", []),
                    "properties": {"community_level": safe_dict(community.get("properties")).get("level", 1)},
                    "confidence": 1.0,
                }
            )

        hierarchy_edges: List[Dict[str, Any]] = []
        for item in hierarchy_pairs.values():
            parent = key2entity.get(item["parent_key"])
            child = key2entity.get(item["child_key"])
            if not parent or not child:
                continue
            hierarchy_edges.append(
                {
                    "id": stable_relation_id(parent["id"], P_COMMUNITY_PARENT_OF, child["id"], prefix="rel_com_parent_"),
                    "subject_id": parent["id"],
                    "object_id": child["id"],
                    "predicate": P_COMMUNITY_PARENT_OF,
                    "relation_name": P_COMMUNITY_PARENT_OF,
                    "description": "Parent community contains this child community at a finer level.",
                    "source_documents": dedupe_list(parent.get("source_documents", []) + child.get("source_documents", [])),
                    "properties": {
                        "parent_level": safe_dict(parent.get("properties")).get("level"),
                        "child_level": safe_dict(child.get("properties")).get("level"),
                    },
                    "confidence": 1.0,
                }
            )

        communities.sort(key=lambda x: (safe_dict(x.get("properties")).get("level", 0), x.get("id", "")))
        report_docs.sort(key=lambda x: (safe_dict(x.get("metadata")).get("level", 0), x.get("id", "")))
        membership_edges.sort(key=lambda x: x.get("id", ""))
        hierarchy_edges.sort(key=lambda x: x.get("id", ""))
        return communities, membership_edges, hierarchy_edges, report_docs

    def _store_report_docs_to_vector_store(
        self,
        report_docs: List[Dict[str, Any]],
        *,
        batch_size: int = 256,
    ) -> int:
        if not report_docs:
            return 0

        documents: List[Document] = []
        for row in report_docs:
            doc_id = safe_str(row.get("id")).strip()
            content = safe_str(row.get("content")).strip()
            metadata = safe_dict(row.get("metadata"))
            if not doc_id or not content:
                continue
            documents.append(
                Document(
                    id=doc_id,
                    content=content,
                    metadata={
                        "community_id": safe_str(metadata.get("community_id")),
                        "community_name": safe_str(metadata.get("community_name")),
                        "title": safe_str(metadata.get("title")),
                        "summary": safe_str(metadata.get("summary")),
                        "level": int(metadata.get("level", 0) or 0),
                        "member_count": int(metadata.get("member_count", 0) or 0),
                        "rating": float(metadata.get("rating", 0.0) or 0.0),
                        "rating_explanation": safe_str(metadata.get("rating_explanation")),
                        "source_documents": json.dumps(safe_list(metadata.get("source_documents")), ensure_ascii=False),
                        "top_members": json.dumps(safe_list(metadata.get("top_members")), ensure_ascii=False),
                        "findings": json.dumps(safe_list(metadata.get("findings")), ensure_ascii=False),
                    },
                )
            )

        if not documents:
            return 0

        self.community_summary_store.store_documents(documents, batch_size=batch_size)
        return len(documents)

    def _ensure_community_embeddings(
        self,
        *,
        communities: List[Dict[str, Any]],
        batch_size: int = 256,
    ) -> List[Dict[str, Any]]:
        if not communities:
            return communities
        self.graph_query_utils.load_embedding_model(self.config.embedding)
        model = getattr(self.graph_query_utils, "model", None)
        if model is None or not hasattr(model, "encode"):
            raise RuntimeError("graph_query_utils.model is not available or lacks encode()")

        need_indices: List[int] = []
        texts: List[str] = []
        for idx, row in enumerate(communities):
            if isinstance(row.get("embedding"), list) and row.get("embedding"):
                continue
            text = "\n".join([safe_str(row.get("name")), safe_str(row.get("description"))]).strip()
            if not text:
                continue
            need_indices.append(idx)
            texts.append(text)

        if not need_indices:
            return communities

        bs = max(1, int(batch_size))
        for start in range(0, len(texts), bs):
            vectors = model.encode(texts[start : start + bs])
            vec_list = _to_vec_list(vectors)
            for offset, vec in enumerate(vec_list):
                idx = need_indices[start + offset]
                communities[idx]["embedding"] = vec
        return communities

    def _persist_entity_embeddings_to_graph(self, communities: List[Dict[str, Any]], batch_size: int = 500) -> int:
        del batch_size
        if not communities:
            return 0
        graph = self.graph_store.get_graph()
        rows: List[Dict[str, Any]] = []
        for community in communities:
            cid = safe_str(community.get("id"))
            emb = community.get("embedding")
            vec = None
            if isinstance(emb, list) and emb and isinstance(emb[0], (int, float)):
                vec = [float(x) for x in emb]
            else:
                vv = _to_vec_list(emb)
                if vv and vv[0]:
                    vec = [float(x) for x in vv[0]]
            if cid and vec:
                rows.append({"id": cid, "embedding": vec})
        if not rows:
            return 0

        updated = 0
        for row in rows:
            cid = row["id"]
            if not graph.has_node(cid):
                continue
            graph.nodes[cid]["embedding"] = row["embedding"]
            updated += 1
        if updated:
            self.graph_store.persist()
        return updated

    def load_json_to_graph_store(
        self,
        *,
        communities: List[Dict[str, Any]],
        membership_edges: List[Dict[str, Any]],
        hierarchy_edges: List[Dict[str, Any]],
    ) -> None:
        self.graph_query_utils.save_to_graph_store(communities, [])
        if membership_edges:
            self.graph_query_utils.save_to_graph_store([], membership_edges)
        if hierarchy_edges:
            self.graph_query_utils.save_to_graph_store([], hierarchy_edges)

    def run(
        self,
        *,
        clear_previous_community: bool = False,
        clear_previous_narrative: bool = False,
    ) -> Dict[str, Any]:
        projection_graph_name = self.cfg.projection_graph_name
        if clear_previous_community:
            logger.info("[CommunityGraph] clearing previous community aggregation")
            self.clear_community_aggregation()
        if clear_previous_narrative:
            logger.info("[CommunityGraph] clearing previous narrative aggregation")
            self.clear_narrative_aggregation()

        projection = None
        try:
            projection = self.graph_query_utils.project_graph_for_community_detection(
                graph_name=projection_graph_name,
                exclude_node_labels=safe_list(self.cfg.exclude_node_labels),
                exclude_relation_types=safe_list(self.cfg.exclude_relation_types),
                weight_property=safe_str(self.cfg.relationship_weight_property),
                use_confidence_as_weight=bool(getattr(self.cfg, "use_confidence_as_weight", False)),
                force_refresh=True,
            )
            if not projection.get("ok"):
                raise RuntimeError(f"Community graph projection failed: {projection}")
            if int(projection.get("node_count", 0) or 0) == 0:
                raise RuntimeError("Community graph projection produced zero nodes.")

            use_weighted_projection = bool(safe_str(self.cfg.relationship_weight_property)) or bool(
                getattr(self.cfg, "use_confidence_as_weight", False)
            )
            assignments = self.graph_query_utils.run_leiden_community_detection(
                graph_name=projection_graph_name,
                write_property=self.cfg.write_property,
                intermediate_property=self.cfg.intermediate_property,
                include_intermediate_communities=bool(self.cfg.include_intermediate_communities),
                relationship_weight_property="weight" if use_weighted_projection else "",
                gamma=float(self.cfg.gamma),
                theta=float(self.cfg.theta),
                tolerance=float(self.cfg.tolerance),
                max_levels=int(self.cfg.max_levels),
                concurrency=int(self.cfg.concurrency),
                random_seed=int(self.cfg.random_seed),
                min_community_size=int(self.cfg.min_community_size),
                persist_assignments=True,
            )
            json_dump_atomic(self.assignments_path, assignments)

            communities, membership_edges, hierarchy_edges, report_docs = self._build_communities_from_assignments(
                assignments,
                exclude_relation_types=safe_list(self.cfg.exclude_relation_types),
            )

            json_dump_atomic(self.communities_path, communities)
            json_dump_atomic(self.membership_edges_path, membership_edges)
            json_dump_atomic(self.hierarchy_edges_path, hierarchy_edges)
            json_dump_atomic(self.reports_path, report_docs)

            self.load_json_to_graph_store(
                communities=communities,
                membership_edges=membership_edges,
                hierarchy_edges=hierarchy_edges,
            )
            stored_report_docs = self._store_report_docs_to_vector_store(report_docs)

            return {
                "projection": projection,
                "assignment_count": len(assignments),
                "community_count": len(communities),
                "membership_edge_count": len(membership_edges),
                "hierarchy_edge_count": len(hierarchy_edges),
                "stored_report_docs": stored_report_docs,
                "communities_path": self.communities_path,
                "reports_path": self.reports_path,
            }
        finally:
            self.graph_query_utils._projected_graphs.pop(projection_graph_name, None)

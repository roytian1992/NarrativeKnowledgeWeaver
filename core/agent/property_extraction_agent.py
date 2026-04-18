# core/agent/property_extraction_agent.py
from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
from langchain.text_splitter import RecursiveCharacterTextSplitter

from core.builder.manager.information_manager import InformationExtractor
from core.utils.general_utils import (
    word_len,
    safe_str,
    is_nonempty_str,
    filter_nodes_by_centrality,
)

logger = logging.getLogger(__name__)


@dataclass
class PropertyExtractionResult:
    ok: bool
    updated_nodes: int
    selected_nodes: int
    flattened_tasks: int
    failed_extract_tasks: int
    failed_merge_nodes: int
    entity_info: Dict[str, Dict[str, Any]]  # updated entities_by_id


class PropertyExtractionAgent:
    """
    Graph-based property extraction agent.

    Responsibilities:
    - Select nodes by centrality (scope/type/threshold/topK)
    - Build per-node context text from node description + incident edge descriptions
    - Chunk context into manageable pieces
    - Extract properties per chunk (concurrent via caller's runner)
    - Aggregate and merge once per node
    - Return updated entity_info dict (caller decides where/how to dump)

    Dependencies:
    - InformationExtractor for (extract_entity_properties, merge_entity_properties)
    - Your existing filter_nodes_by_centrality + word_len utilities
    """

    def __init__(self, config: Any, llm: Any):
        self.config = config
        self.llm = llm
        self.info = InformationExtractor(config, llm)

    # ----------------------------
    # edge payload helpers
    # ----------------------------
    def _edge_relation_type(self, data: Any) -> str:
        if not isinstance(data, dict):
            return ""
        rt = data.get("relation_type")
        if isinstance(rt, str) and rt.strip():
            return rt.strip()
        rel = data.get("relation")
        if isinstance(rel, dict):
            rt2 = rel.get("relation_type")
            if isinstance(rt2, str) and rt2.strip():
                return rt2.strip()
        return ""

    def _edge_description(self, data: Any) -> str:
        if not isinstance(data, dict):
            return ""
        d = data.get("description")
        if isinstance(d, str) and d.strip():
            return d.strip()
        rel = data.get("relation")
        if isinstance(rel, dict):
            d2 = rel.get("description")
            if isinstance(d2, str) and d2.strip():
                return d2.strip()
        return ""

    def _get_node_name_type_scope(
        self,
        G: nx.Graph,
        nid: str,
        *,
        general_semantic_type: str,
    ) -> Tuple[str, str, str]:
        node_data = (G.nodes.get(nid, {}) or {})
        name = safe_str(node_data.get("name"))
        t = node_data.get("type")
        if isinstance(t, list):
            t = t[0] if t else general_semantic_type
        entity_type = safe_str(t) or general_semantic_type
        node_scope = safe_str(node_data.get("scope")) or ""
        return name, entity_type, node_scope

    def _build_full_text(
        self,
        G: nx.Graph,
        nid: str,
        *,
        exclude_relation_types: Optional[Set[str]] = None,
        include_relation_types: Optional[Set[str]] = None,
        max_edge_descriptions_per_node: Optional[int] = None,
        max_context_words_per_node: Optional[int] = None,
        dedupe_edge_descriptions: bool = True,
    ) -> str:
        node_data = (G.nodes.get(nid, {}) or {})
        node_desc = safe_str(node_data.get("description"))

        ctx_items: List[Tuple[str, str]] = []

        def _keep_type(rtype: str) -> bool:
            if exclude_relation_types and rtype in exclude_relation_types:
                return False
            if include_relation_types and rtype not in include_relation_types:
                return False
            return True

        if isinstance(G, (nx.MultiDiGraph, nx.MultiGraph)):
            if G.is_directed():
                for _, _, _, data in G.out_edges(nid, keys=True, data=True):
                    rtype = self._edge_relation_type(data)
                    if not _keep_type(rtype):
                        continue
                    desc = self._edge_description(data)
                    if desc:
                        ctx_items.append((rtype, desc))

                for _, _, _, data in G.in_edges(nid, keys=True, data=True):
                    rtype = self._edge_relation_type(data)
                    if not _keep_type(rtype):
                        continue
                    desc = self._edge_description(data)
                    if desc:
                        ctx_items.append((rtype, desc))
            else:
                for _, _, _, data in G.edges(nid, keys=True, data=True):
                    rtype = self._edge_relation_type(data)
                    if not _keep_type(rtype):
                        continue
                    desc = self._edge_description(data)
                    if desc:
                        ctx_items.append((rtype, desc))
        else:
            for _, _, data in G.edges(nid, data=True):
                rtype = self._edge_relation_type(data)
                if not _keep_type(rtype):
                    continue
                desc = self._edge_description(data)
                if desc:
                    ctx_items.append((rtype, desc))

        if dedupe_edge_descriptions:
            deduped_items: List[Tuple[str, str]] = []
            seen_desc: Set[str] = set()
            for rtype, desc in ctx_items:
                norm = safe_str(desc).strip()
                if not norm or norm in seen_desc:
                    continue
                seen_desc.add(norm)
                deduped_items.append((rtype, norm))
            ctx_items = deduped_items

        if isinstance(max_edge_descriptions_per_node, int) and max_edge_descriptions_per_node > 0:
            ctx_items = sorted(
                ctx_items,
                key=lambda item: (word_len(item[1]), len(item[1])),
                reverse=True,
            )[: int(max_edge_descriptions_per_node)]

        ctx: List[str] = []
        word_budget = max_context_words_per_node if isinstance(max_context_words_per_node, int) else None
        used_words = word_len(node_desc) if node_desc else 0
        for _, desc in ctx_items:
            if word_budget is not None and word_budget > 0:
                next_words = word_len(desc)
                if ctx and used_words + next_words > int(word_budget):
                    continue
                if (not ctx) and node_desc and used_words + next_words > int(word_budget):
                    continue
                used_words += next_words
            ctx.append(desc)

        text = (node_desc + "\n" + "\n".join(ctx)).strip()
        return text

    # ----------------------------
    # main API
    # ----------------------------
    def run(
        self,
        *,
        G: nx.Graph,
        entities_by_id: Dict[str, Dict[str, Any]],
        entity_schema: Optional[List[Dict[str, Any]]] = None,
        general_semantic_type: str = "Concept",
        # selection
        scope: str = "global",
        metric: str = "total_degree",
        threshold: float = 2.0,
        num_top: Optional[int] = None,
        node_type: Optional[str] = None,
        exclude_relation_types: Optional[Set[str]] = None,
        include_relation_types: Optional[Set[str]] = None,
        # chunking
        chunk_size: int = 1200,
        chunk_overlap: int = 200,
        max_edge_descriptions_per_node: Optional[int] = None,
        max_context_words_per_node: Optional[int] = None,
        dedupe_edge_descriptions: bool = True,
        # concurrency control is handled by caller; here only prepare tasks + pure functions
        max_workers: int = 8,
        per_task_timeout: float = 120.0,
        retries: int = 2,
        # merge
        num_properties: int = 10,
        merge_max_workers: Optional[int] = None,
        merge_timeout: float = 120.0,
        merge_retries: int = 2,
        # runner hooks (inject your existing concurrent runners)
        run_concurrent_with_retries_fn=None,
        verbose: bool = True,
    ) -> PropertyExtractionResult:
        """
        NOTE:
        - This agent does NOT import your run_concurrent_with_retries directly,
          because you may want to swap implementation (thread/process/async).
        - Caller must pass run_concurrent_with_retries_fn with the same signature you already use.
        """

        if run_concurrent_with_retries_fn is None:
            raise ValueError("run_concurrent_with_retries_fn must be provided")

        if exclude_relation_types is None:
            exclude_relation_types = set()

        # 1) schema-derived hints (optional, kept for future prompt conditioning)
        type2properties: Dict[str, Any] = {}
        type2description: Dict[str, str] = {}
        if isinstance(entity_schema, list):
            for ent in entity_schema:
                if not isinstance(ent, dict):
                    continue
                t = ent.get("type")
                if not t:
                    continue
                t = str(t)
                type2properties[t] = ent.get("properties", [])
                type2description[t] = str(ent.get("description", "") or "")

        # 2) select nodes by centrality
        selected = filter_nodes_by_centrality(
            G,
            metric=metric,  # type: ignore
            threshold=float(threshold),
            scope=scope,
            node_type=node_type,
            exclude_relation_types=set(exclude_relation_types) if exclude_relation_types else None,
            include_relation_types=set(include_relation_types) if include_relation_types else None,
            num_top=num_top,
        )
        node_ids = [safe_str(n) for (n, _) in selected if safe_str(n)]

        if verbose:
            logger.info(
                f"[PropertyAgent] selected nodes={len(node_ids)} "
                f"(scope={scope} metric={metric} threshold={threshold} num_top={num_top} node_type={node_type})"
            )

        if not node_ids:
            return PropertyExtractionResult(
                ok=True,
                updated_nodes=0,
                selected_nodes=0,
                flattened_tasks=0,
                failed_extract_tasks=0,
                failed_merge_nodes=0,
                entity_info=entities_by_id,
            )

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=int(chunk_size),
            chunk_overlap=int(chunk_overlap),
            length_function=word_len,
        )
        single_shot_threshold = max(1, int(chunk_size))

        # 3) flatten node->chunks into tasks
        tasks: List[Dict[str, Any]] = []
        single_shot_items: List[Dict[str, Any]] = []
        node_meta: Dict[str, Dict[str, Any]] = {}

        for nid in node_ids:
            name, entity_type, node_scope = self._get_node_name_type_scope(
                G, nid, general_semantic_type=general_semantic_type
            )
            if not name:
                continue
            if scope and node_scope and node_scope != scope:
                continue

            full_text = self._build_full_text(
                G,
                nid,
                exclude_relation_types=exclude_relation_types,
                include_relation_types=include_relation_types,
                max_edge_descriptions_per_node=max_edge_descriptions_per_node,
                max_context_words_per_node=max_context_words_per_node,
                dedupe_edge_descriptions=dedupe_edge_descriptions,
            )
            if not full_text:
                continue

            node_meta[nid] = {"name": name, "type": entity_type, "scope": (node_scope or scope)}

            if word_len(full_text) <= single_shot_threshold:
                single_shot_items.append(
                    {
                        "node_id": nid,
                        "name": name,
                        "entity_type": entity_type,
                        "text": full_text,
                    }
                )
                continue

            chunks = splitter.split_text(full_text)
            for cidx, chunk in enumerate(chunks):
                chunk = (chunk or "").strip()
                if not chunk:
                    continue
                tasks.append(
                    {
                        "node_id": nid,
                        "chunk_index": cidx,
                        "name": name,
                        "entity_type": entity_type,
                        "text": chunk,
                    }
                )

        total_extract_tasks = len(tasks) + len(single_shot_items)
        if verbose:
            logger.info(
                f"[PropertyAgent] flattened tasks={total_extract_tasks} "
                f"(chunk_tasks={len(tasks)} single_shot_nodes={len(single_shot_items)}) from nodes={len(node_meta)}"
            )

        if not tasks and not single_shot_items:
            return PropertyExtractionResult(
                ok=True,
                updated_nodes=0,
                selected_nodes=len(node_ids),
                flattened_tasks=0,
                failed_extract_tasks=0,
                failed_merge_nodes=0,
                entity_info=entities_by_id,
            )

        def _extract_task_fn(t: Dict[str, Any]) -> Dict[str, Any]:
            nid = t["node_id"]
            raw = self.info.extract_entity_properties(
                text=t["text"],
                entity_name=t["name"],
                entity_type=t["entity_type"],
            )

            obj = json.loads(raw) if isinstance(raw, str) else (raw or {})
            if not isinstance(obj, dict):
                obj = {}

            props = obj.get("properties", {}) or {}
            if not isinstance(props, dict):
                props = {}

            new_desc = obj.get("new_description", "") or ""
            if not isinstance(new_desc, str):
                new_desc = str(new_desc)

            return {
                "ok": True,
                "node_id": nid,
                "chunk_index": int(t.get("chunk_index", 0)),
                "properties": props,
                "new_description": new_desc,
            }

        def _single_shot_task_fn(t: Dict[str, Any]) -> Dict[str, Any]:
            raw = self.info.extract_entity_properties(
                text=t["text"],
                entity_name=t["name"],
                entity_type=t["entity_type"],
            )

            obj = json.loads(raw) if isinstance(raw, str) else (raw or {})
            if not isinstance(obj, dict):
                obj = {}

            props = obj.get("properties", {}) or {}
            if not isinstance(props, dict):
                props = {}

            new_desc = obj.get("new_description", "") or ""
            if not isinstance(new_desc, str):
                new_desc = str(new_desc)

            return {
                "ok": True,
                "node_id": t["node_id"],
                "properties": props,
                "new_description": new_desc,
            }

        if tasks:
            results_map, failed_indices = run_concurrent_with_retries_fn(
                items=tasks,
                task_fn=_extract_task_fn,
                per_task_timeout=float(per_task_timeout),
                max_retry_rounds=max(1, int(retries)),
                max_in_flight=int(max_workers),
                max_workers=int(max_workers),
                thread_name_prefix="prop",
                desc_prefix="Extracting properties from node chunks",
                treat_empty_as_failure=True,
                is_empty_fn=lambda r: (not r) or (not isinstance(r, dict)) or (not r.get("ok")),
            )
        else:
            results_map, failed_indices = {}, []

        if single_shot_items:
            single_shot_map, single_shot_failed = run_concurrent_with_retries_fn(
                items=single_shot_items,
                task_fn=_single_shot_task_fn,
                per_task_timeout=float(per_task_timeout),
                max_retry_rounds=max(1, int(retries)),
                max_in_flight=int(max_workers),
                max_workers=int(max_workers),
                thread_name_prefix="prop_single",
                desc_prefix="Extracting properties from short nodes",
                treat_empty_as_failure=True,
                is_empty_fn=lambda r: (not r) or (not isinstance(r, dict)) or (not r.get("ok")),
            )
        else:
            single_shot_map, single_shot_failed = {}, []

        failed_extract_tasks = len(failed_indices or []) + len(single_shot_failed or [])
        if verbose and failed_extract_tasks:
            logger.warning(f"[PropertyAgent] failed extract tasks={failed_extract_tasks} / {total_extract_tasks}")

        # 5) aggregate per node for long-text chunk path
        per_node_desc: Dict[str, List[str]] = defaultdict(list)
        per_node_props: Dict[str, Dict[str, Any]] = defaultdict(dict)

        for i in range(len(tasks)):
            r = results_map.get(i)
            if not isinstance(r, dict) or not r.get("ok"):
                continue
            nid = safe_str(r.get("node_id"))
            if not nid:
                continue

            d = safe_str(r.get("new_description"))
            if d:
                per_node_desc[nid].append(d)

            props = r.get("properties") or {}
            if isinstance(props, dict):
                per_node_props[nid].update(props)

        def _apply_final_result(nid: str, final_properties: Dict[str, Any], new_description: str) -> bool:
            meta = node_meta.get(nid, {}) or {}
            name = safe_str(meta.get("name")) or safe_str((G.nodes.get(nid, {}) or {}).get("name"))
            entity_type = safe_str(meta.get("type")) or safe_str((G.nodes.get(nid, {}) or {}).get("type")) or general_semantic_type
            node_scope = safe_str(meta.get("scope")) or safe_str((G.nodes.get(nid, {}) or {}).get("scope")) or scope

            if nid in entities_by_id and isinstance(entities_by_id[nid], dict):
                entities_by_id[nid]["properties"] = final_properties
                if new_description:
                    entities_by_id[nid]["description"] = new_description
            else:
                entities_by_id[nid] = {
                    "id": nid,
                    "name": name,
                    "type": entity_type,
                    "scope": node_scope,
                    "description": new_description,
                    "properties": final_properties,
                }
            return True

        updated_nodes = 0
        for i in range(len(single_shot_items)):
            res = single_shot_map.get(i)
            if not isinstance(res, dict) or not res.get("ok"):
                continue
            nid = safe_str(res.get("node_id"))
            if not nid:
                continue

            final_properties = res.get("properties") or {}
            if not isinstance(final_properties, dict):
                final_properties = {}
            new_description = safe_str(res.get("new_description"))

            if _apply_final_result(nid, final_properties, new_description):
                updated_nodes += 1

        # 6) merge once per node for long-text path only
        merge_items: List[Dict[str, Any]] = []
        for nid, meta in node_meta.items():
            name = safe_str(meta.get("name"))
            if not name:
                continue
            full_description = "\n".join(per_node_desc.get(nid, [])).strip()
            props_dict = per_node_props.get(nid, {}) or {}
            if (not full_description) and (not props_dict):
                continue
            merge_items.append(
                {
                    "node_id": nid,
                    "name": name,
                    "type": safe_str(meta.get("type")) or general_semantic_type,
                    "scope": safe_str(meta.get("scope")) or scope,
                    "full_description": full_description,
                    "properties": props_dict,
                }
            )

        if verbose:
            logger.info(f"[PropertyAgent] merge nodes={len(merge_items)} (concurrent)")

        if not merge_items:
            return PropertyExtractionResult(
                ok=True,
                updated_nodes=updated_nodes,
                selected_nodes=len(node_ids),
                flattened_tasks=total_extract_tasks,
                failed_extract_tasks=failed_extract_tasks,
                failed_merge_nodes=0,
                entity_info=entities_by_id,
            )

        def _merge_task_fn(t: Dict[str, Any]) -> Dict[str, Any]:
            nid = t["node_id"]
            name = t["name"]
            full_description = t["full_description"]
            props_dict = t["properties"]

            merged_raw = self.info.merge_entity_properties(
                entity_name=name,
                full_description=full_description,
                properties=json.dumps(props_dict, ensure_ascii=False, indent=2),
                num_properties=int(num_properties),
            )

            merged_obj = json.loads(merged_raw) if isinstance(merged_raw, str) else (merged_raw or {})
            if not isinstance(merged_obj, dict):
                merged_obj = {}

            final_properties = merged_obj.get("properties", {}) or {}
            if not isinstance(final_properties, dict):
                final_properties = {}

            new_description = merged_obj.get("new_description", "") or ""
            if not isinstance(new_description, str):
                new_description = str(new_description)

            return {
                "ok": True,
                "node_id": nid,
                "properties": final_properties,
                "new_description": new_description,
            }

        if merge_max_workers is None:
            merge_max_workers = int(max_workers)

        merge_map, merge_failed = run_concurrent_with_retries_fn(
            items=merge_items,
            task_fn=_merge_task_fn,
            per_task_timeout=float(merge_timeout),
            max_retry_rounds=max(1, int(merge_retries)),
            max_in_flight=int(merge_max_workers),
            max_workers=int(merge_max_workers),
            thread_name_prefix="prop_merge",
            desc_prefix="Merging entity properties",
            treat_empty_as_failure=True,
            is_empty_fn=lambda r: (not r) or (not isinstance(r, dict)) or (not r.get("ok")),
        )

        failed_merge_nodes = len(merge_failed or [])
        if verbose and failed_merge_nodes:
            logger.warning(f"[PropertyAgent] failed merge nodes={failed_merge_nodes} / {len(merge_items)}")

        for i in range(len(merge_items)):
            res = merge_map.get(i)
            if not isinstance(res, dict) or not res.get("ok"):
                continue
            nid = safe_str(res.get("node_id"))
            if not nid:
                continue

            final_properties = res.get("properties") or {}
            if not isinstance(final_properties, dict):
                final_properties = {}
            new_description = safe_str(res.get("new_description"))

            if _apply_final_result(nid, final_properties, new_description):
                updated_nodes += 1

        return PropertyExtractionResult(
            ok=True,
            updated_nodes=updated_nodes,
            selected_nodes=len(node_ids),
            flattened_tasks=len(tasks),
            failed_extract_tasks=failed_extract_tasks,
            failed_merge_nodes=failed_merge_nodes,
            entity_info=entities_by_id,
        )

"""Graph query utility layer backed by a local NetworkX runtime graph."""

from __future__ import annotations

from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import json
import logging
import math
import os
import random
import re

import networkx as nx
import numpy as np
from networkx.algorithms.community import louvain_communities
from tqdm import tqdm

from core.model_providers.openai_embedding import OpenAIEmbeddingModel
from core.models.data import Entity, Relation
from core.storage.vector_store import VectorStore
from core.utils.config import EmbeddingConfig
from core.utils.format import DOC_TYPE_META
from core.utils.general_utils import (
    _is_none_relation,
    _to_vec_list,
    build_search_keywords,
    dedupe_list,
    get_doc_text,
    load_json,
    safe_dict,
    safe_list,
    safe_str,
    stable_relation_id,
)


logger = logging.getLogger(__name__)
_VALID_PROP_TOKEN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _flatten_vector(vec: Any) -> List[float]:
    out = _to_vec_list(vec)
    if not out:
        return []
    first = out[0]
    if not isinstance(first, list):
        return []
    return [float(x) for x in first]


def _as_1d_embedding(vec: Any) -> Optional[List[float]]:
    out = _to_vec_list(vec)
    if not out:
        return None
    first = out[0]
    if not isinstance(first, list) or not first:
        return None
    return [float(x) for x in first]


def _quote_property_token(name: str) -> str:
    token = str(name or "").strip()
    if not _VALID_PROP_TOKEN.match(token):
        raise ValueError(f"Unsafe property token: {name!r}")
    return f"`{token}`"


def _as_list_str(value: Any) -> List[str]:
    if isinstance(value, list):
        out: List[str] = []
        seen = set()
        for item in value:
            text = str(item or "").strip()
            if not text or text in seen:
                continue
            seen.add(text)
            out.append(text)
        return out
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    return []


def _normalize_entity_surface(text: Any) -> str:
    raw = safe_str(text).strip().lower()
    if not raw:
        return ""
    return re.sub(r"[\s\-_:/,.;，。！？!?'\"“”‘’()\[\]{}<>《》【】]+", "", raw)


def _looks_like_base_entity_name(text: Any) -> bool:
    raw = safe_str(text).strip()
    if not raw:
        return False
    bad_markers = ("的", "了", "着", "在", "将", "把", "并", "后", "前", "时", "与", "跟")
    return not any(marker in raw for marker in bad_markers)


def _is_base_entity_label(labels: Any) -> bool:
    clean = {safe_str(x).strip() for x in _as_list_str(labels) if safe_str(x).strip()}
    blocked = {"Event", "Episode", "Storyline", "Community", "Document", "Scene", "Chapter"}
    return bool(clean) and not (clean & blocked)


def _safe_properties(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        raw = value.strip()
        if raw.startswith("{") and raw.endswith("}"):
            try:
                obj = json.loads(raw)
                return obj if isinstance(obj, dict) else {}
            except Exception:
                return {}
    return {}


class GraphQueryUtils:
    """Query and graph-analytics helpers for the local runtime graph."""

    def __init__(self, graph_store, doc_type: str = "screenplay"):
        if doc_type not in DOC_TYPE_META:
            raise ValueError(f"Unsupported doc_type: {doc_type}")
        self.doc_type = doc_type
        self.meta = DOC_TYPE_META[doc_type]
        self.graph_store = graph_store
        self.model = None
        self.config = None
        self.dim = None
        self.embedding_field = "embedding"
        self.entity_vector_category = "graph_entity"
        self.relation_vector_category = "graph_relation"
        self._entity_vector_store_handle: Optional[VectorStore] = None
        self._relation_vector_store_handle: Optional[VectorStore] = None
        self._document_title_index_loaded = False
        self._document_id2title: Dict[str, str] = {}
        self._document_id2subtitle: Dict[str, str] = {}
        self._projected_graphs: Dict[str, nx.Graph] = {}

    def _graph(self) -> nx.MultiDiGraph:
        return self.graph_store.get_graph()

    def _persist(self) -> None:
        self.graph_store.persist()

    def _get_entity_vector_store(self) -> Optional[VectorStore]:
        if self._entity_vector_store_handle is None:
            try:
                self._entity_vector_store_handle = VectorStore(
                    self.graph_store.config,
                    category=self.entity_vector_category,
                    load_embedding_model=False,
                )
            except Exception:
                logger.exception("Failed to initialize entity vector store.")
                self._entity_vector_store_handle = None
        return self._entity_vector_store_handle

    def _get_relation_vector_store(self) -> Optional[VectorStore]:
        if self._relation_vector_store_handle is None:
            try:
                self._relation_vector_store_handle = VectorStore(
                    self.graph_store.config,
                    category=self.relation_vector_category,
                    load_embedding_model=False,
                )
            except Exception:
                logger.exception("Failed to initialize relation vector store.")
                self._relation_vector_store_handle = None
        return self._relation_vector_store_handle

    def _node_embedding_text(self, node: Dict[str, Any]) -> str:
        name = safe_str(node.get("name")).strip()
        desc = safe_str(node.get("description") or node.get("summary")).strip()
        props_dict = _safe_properties(node.get("properties", {}))
        text = f"{name}.{desc}".strip(".")
        if props_dict:
            prop_text = "；".join([f"{k}：{v}" for k, v in props_dict.items()])
            text = f"{text}.{prop_text}".strip(".")
        return text or name or safe_str(node.get("id"))

    def _relation_embedding_text(
        self,
        rel: Dict[str, Any],
        *,
        src_name: str = "",
        dst_name: str = "",
    ) -> str:
        desc = safe_str(rel.get("description")).strip()
        if desc:
            return desc
        pred = safe_str(rel.get("predicate") or rel.get("relation_type")).strip()
        rel_name = safe_str(rel.get("relation_name")).strip()
        text = " ".join(x for x in [src_name, pred or rel_name, dst_name] if x)
        return text or pred or rel_name or safe_str(rel.get("id"))

    def _build_entity_vector_record(
        self,
        node_id: str,
        data: Dict[str, Any],
        embedding: List[float],
    ) -> Dict[str, Any]:
        node_labels = [x for x in self._node_labels(data) if x]
        return {
            "id": node_id,
            "content": self._node_embedding_text(data) or node_id,
            "metadata": {
                "kind": "graph_entity",
                "name": safe_str(data.get("name")),
                "primary_label": self._primary_label(data),
                "labels": "|".join(node_labels),
            },
            "embedding": [float(x) for x in embedding],
        }

    def _build_relation_vector_record(
        self,
        rel_id: str,
        src_id: str,
        dst_id: str,
        data: Dict[str, Any],
        embedding: List[float],
    ) -> Dict[str, Any]:
        graph = self._graph()
        src_name = safe_str(graph.nodes[src_id].get("name")) if graph.has_node(src_id) else ""
        dst_name = safe_str(graph.nodes[dst_id].get("name")) if graph.has_node(dst_id) else ""
        pred = safe_str(data.get("predicate") or data.get("relation_type")).strip()
        return {
            "id": rel_id,
            "content": self._relation_embedding_text(data, src_name=src_name, dst_name=dst_name) or rel_id,
            "metadata": {
                "kind": "graph_relation",
                "predicate": pred,
                "subject_id": src_id,
                "object_id": dst_id,
                "subject_name": src_name,
                "object_name": dst_name,
            },
            "embedding": [float(x) for x in embedding],
        }

    def _find_relation_edge(self, rel_id: str) -> Optional[Tuple[str, str, str, Dict[str, Any]]]:
        target = safe_str(rel_id).strip()
        if not target:
            return None
        for src, dst, key, data in self._iter_edges():
            if safe_str(data.get("id")) == target or safe_str(key) == target:
                return src, dst, key, data
        return None

    def get_embedding_cache_status(self) -> Dict[str, int]:
        graph_node_embedding_count = sum(
            1 for _node_id, data in self._iter_node_items() if _as_1d_embedding(data.get(self.embedding_field)) is not None
        )
        graph_relation_embedding_count = sum(
            1 for _src, _dst, _key, data in self._iter_edges() if _as_1d_embedding(data.get(self.embedding_field)) is not None
        )
        entity_store = self._get_entity_vector_store()
        relation_store = self._get_relation_vector_store()
        entity_store_count = 0
        relation_store_count = 0
        if entity_store is not None:
            try:
                entity_store_count = int((entity_store.get_stats() or {}).get("document_count", 0) or 0)
            except Exception:
                entity_store_count = 0
        if relation_store is not None:
            try:
                relation_store_count = int((relation_store.get_stats() or {}).get("document_count", 0) or 0)
            except Exception:
                relation_store_count = 0
        return {
            "graph_node_embedding_count": graph_node_embedding_count,
            "graph_relation_embedding_count": graph_relation_embedding_count,
            "entity_vector_count": entity_store_count,
            "relation_vector_count": relation_store_count,
        }

    def _node_labels(self, data: Dict[str, Any]) -> List[str]:
        labels = data.get("type", data.get("labels", []))
        return [x for x in _as_list_str(labels) if x]

    def _node_has_label(self, data: Dict[str, Any], label: str) -> bool:
        target = str(label or "").strip()
        if not target:
            return False
        return target in self._node_labels(data)

    def _primary_label(self, data: Dict[str, Any]) -> str:
        labels = [x for x in self._node_labels(data) if x != "Entity"]
        return labels[0] if labels else "Entity"

    def _build_entity_from_data(self, data: Dict[str, Any]) -> Entity:
        props = _safe_properties(data.get("properties", {}))
        labels = [lbl for lbl in self._node_labels(data) if lbl != "Entity"]
        return Entity(
            id=safe_str(data.get("id")),
            name=safe_str(data.get("name")),
            scope=safe_str(data.get("scope")) or "Unknown",
            type=labels if labels else "Unknown",
            aliases=_as_list_str(data.get("aliases", [])),
            description=safe_str(data.get("description") or data.get("summary")),
            properties=props,
            source_documents=_as_list_str(data.get("source_documents", [])),
        )

    def _edge_to_relation(self, src_id: str, dst_id: str, data: Dict[str, Any]) -> Relation:
        predicate = safe_str(data.get("predicate") or data.get("relation_type")).strip() or "RELATED_TO"
        return Relation(
            id=safe_str(data.get("id")) or stable_relation_id(src_id, predicate, dst_id),
            subject_id=safe_str(data.get("subject_id")) or src_id,
            object_id=safe_str(data.get("object_id")) or dst_id,
            predicate=predicate,
            relation_name=data.get("relation_name"),
            persistence=data.get("persistence"),
            description=data.get("description"),
            source_documents=_as_list_str(data.get("source_documents", [])),
            confidence=float(data.get("confidence", 1.0) or 1.0),
            properties=_safe_properties(data.get("properties", {})),
        )

    def _strip_embedding_fields(self, d: Dict[str, Any]) -> Dict[str, Any]:
        out = {}
        for k, v in (d or {}).items():
            if str(k).lower() in {"embedding", "relation_embedding", "node_embedding"}:
                continue
            out[k] = v
        return out

    def _node_to_minimal_dict(self, node_obj: Dict[str, Any]) -> Dict[str, Any]:
        raw = self._strip_embedding_fields(dict(node_obj or {}))
        raw["labels"] = [x for x in self._node_labels(raw) if x != "Entity"]
        return raw

    def _rel_to_minimal_dict(self, rel_obj: Dict[str, Any]) -> Dict[str, Any]:
        props = self._strip_embedding_fields(dict(rel_obj or {}))
        rel_type = props.get("predicate") or props.get("type") or "RELATED"
        return {
            "id": props.get("id"),
            "type": rel_type,
            "predicate": rel_type,
            "relation_name": props.get("relation_name"),
            "description": props.get("description") or "",
            "properties": props,
        }

    def _iter_edges(self):
        yield from self._graph().edges(keys=True, data=True)

    def _iter_node_items(self):
        yield from self._graph().nodes(data=True)

    def _edge_matches_relation_type(self, data: Dict[str, Any], relation_types: Optional[List[str]]) -> bool:
        if not relation_types:
            return True
        allowed = {str(x).strip() for x in relation_types if str(x).strip()}
        pred = safe_str(data.get("predicate") or data.get("relation_type")).strip()
        return pred in allowed

    def _edge_weight(self, data: Dict[str, Any], *, weight_property: str = "", use_confidence_as_weight: bool = False) -> float:
        if weight_property:
            props = _safe_properties(data.get("properties", {}))
            value = props.get(weight_property, data.get(weight_property))
            try:
                return float(value)
            except Exception:
                return 1.0
        if use_confidence_as_weight:
            try:
                return float(data.get("confidence", 1.0) or 1.0)
            except Exception:
                return 1.0
        return 1.0

    def load_embedding_model(self, config: EmbeddingConfig):
        self.config = config
        self.model = OpenAIEmbeddingModel(config)
        self.dim = getattr(config, "dimensions", None)

    def encode_node_embedding(self, node: Dict[str, Any]) -> List[float]:
        text = self._node_embedding_text(node)
        if self.config is not None and len(text) >= int(getattr(self.config, "max_tokens", 8192) or 8192):
            text = text[: int(getattr(self.config, "max_tokens", 8192) or 8192)]
        embed = self.model.encode(text)
        vec = _as_1d_embedding(embed)
        return vec or []

    def encode_relation_embedding(self, rel: Dict[str, Any]) -> Optional[List[float]]:
        desc = safe_str(rel.get("description")).strip()
        if not desc:
            return None
        embed = self.model.encode(desc)
        return _as_1d_embedding(embed)

    def update_node_embedding(self, node_id: str, embedding: List[float]) -> None:
        graph = self._graph()
        vector = _as_1d_embedding(embedding)
        if not vector or not graph.has_node(node_id):
            return
        graph.nodes[node_id][self.embedding_field] = vector
        entity_store = self._get_entity_vector_store()
        if entity_store is not None:
            entity_store.upsert_records([self._build_entity_vector_record(node_id, graph.nodes[node_id], vector)])
        self._persist()

    def update_relation_embedding(self, rel_id: str, embedding: List[float]) -> None:
        vector = _as_1d_embedding(embedding)
        if not vector:
            return
        found = self._find_relation_edge(rel_id)
        if found is None:
            return
        src, dst, key, data = found
        graph = self._graph()
        graph[src][dst][key][self.embedding_field] = vector
        relation_store = self._get_relation_vector_store()
        if relation_store is not None:
            record_id = safe_str(data.get("id")) or safe_str(key)
            relation_store.upsert_records([self._build_relation_vector_record(record_id, src, dst, graph[src][dst][key], vector)])
        self._persist()

    def count_nodes_with_label(self, label: str) -> int:
        target = safe_str(label).strip()
        if not target:
            return 0
        return sum(1 for _, data in self._iter_node_items() if self._node_has_label(data, target))

    def list_entity_types(self) -> List[str]:
        labels: Set[str] = set()
        for _, data in self._iter_node_items():
            labels.update(self._node_labels(data))
        labels.discard("*")
        return sorted(labels)

    def compute_centrality(
        self,
        *,
        graph_name: str = "centrality_graph",
        force_refresh: bool = True,
        include_betweenness: bool = True,
    ) -> Dict[str, Any]:
        graph = self._graph()
        section_label = str(self.meta.get("section_label", "Document")).strip() or "Document"
        contains_pred = str(self.meta.get("contains_pred", "CONTAINS")).strip() or "CONTAINS"
        sg = nx.Graph()
        for node_id, data in self._iter_node_items():
            if self._node_has_label(data, section_label):
                continue
            sg.add_node(node_id)
        for src, dst, _key, data in self._iter_edges():
            if src not in sg or dst not in sg:
                continue
            pred = safe_str(data.get("predicate") or data.get("relation_type")).strip()
            if pred == contains_pred:
                continue
            weight = self._edge_weight(data, use_confidence_as_weight=True)
            if sg.has_edge(src, dst):
                sg[src][dst]["weight"] = max(float(sg[src][dst].get("weight", 1.0)), weight)
            else:
                sg.add_edge(src, dst, weight=weight)
        if sg.number_of_nodes() == 0:
            return {"ok": True, "node_count": 0, "relationship_count": 0, "written": 0}

        pagerank = nx.pagerank(sg, weight="weight")
        degree = dict(sg.degree())
        betweenness: Dict[str, float] = {}
        if include_betweenness:
            betweenness = nx.betweenness_centrality(sg, weight="weight", normalized=True)

        for node_id in sg.nodes:
            if graph.has_node(node_id):
                graph.nodes[node_id]["pagerank"] = float(pagerank.get(node_id, 0.0))
                graph.nodes[node_id]["degree"] = float(degree.get(node_id, 0.0))
                graph.nodes[node_id]["betweenness"] = float(betweenness.get(node_id, 0.0))
        self._persist()
        return {
            "ok": True,
            "graph_name": graph_name,
            "node_count": sg.number_of_nodes(),
            "relationship_count": sg.number_of_edges(),
            "written": sg.number_of_nodes(),
        }

    def delete_nodes_by_labels(self, labels: List[str]) -> int:
        clean = {str(x).strip() for x in (labels or []) if str(x).strip()}
        if not clean:
            return 0
        graph = self._graph()
        to_delete = [node_id for node_id, data in self._iter_node_items() if clean & set(self._node_labels(data))]
        for node_id in to_delete:
            graph.remove_node(node_id)
        if to_delete:
            self._persist()
        return len(to_delete)

    def clear_community_assignments(
        self,
        *,
        write_property: str = "community_id",
        intermediate_property: str = "community_path",
    ) -> int:
        _quote_property_token(write_property)
        _quote_property_token(intermediate_property)
        changed = 0
        graph = self._graph()
        for _, data in self._iter_node_items():
            if write_property in data or intermediate_property in data:
                data.pop(write_property, None)
                data.pop(intermediate_property, None)
                changed += 1
        if changed:
            self._persist()
        return changed

    def project_graph_for_community_detection(
        self,
        *,
        graph_name: str = "community_projection",
        exclude_node_labels: Optional[List[str]] = None,
        exclude_relation_types: Optional[List[str]] = None,
        weight_property: str = "",
        use_confidence_as_weight: bool = False,
        force_refresh: bool = True,
    ) -> Dict[str, Any]:
        if force_refresh:
            self._projected_graphs.pop(graph_name, None)
        exclude_node_labels = {str(x).strip() for x in (exclude_node_labels or []) if str(x).strip()}
        exclude_relation_types = {str(x).strip() for x in (exclude_relation_types or []) if str(x).strip()}

        proj = nx.Graph()
        for node_id, data in self._iter_node_items():
            labels = set(self._node_labels(data))
            if exclude_node_labels & labels:
                continue
            proj.add_node(node_id)
        for src, dst, _key, data in self._iter_edges():
            if src not in proj or dst not in proj:
                continue
            pred = safe_str(data.get("predicate") or data.get("relation_type")).strip()
            if pred in exclude_relation_types:
                continue
            weight = self._edge_weight(
                data,
                weight_property=str(weight_property or "").strip(),
                use_confidence_as_weight=use_confidence_as_weight,
            )
            if proj.has_edge(src, dst):
                proj[src][dst]["weight"] = max(float(proj[src][dst].get("weight", 1.0)), weight)
            else:
                proj.add_edge(src, dst, weight=weight)

        self._projected_graphs[graph_name] = proj
        return {
            "ok": True,
            "graph_name": graph_name,
            "node_count": proj.number_of_nodes(),
            "relationship_count": proj.number_of_edges(),
        }

    def run_leiden_community_detection(
        self,
        *,
        graph_name: str = "community_projection",
        write_property: str = "community_id",
        intermediate_property: str = "community_path",
        include_intermediate_communities: bool = True,
        relationship_weight_property: str = "",
        gamma: float = 1.0,
        theta: float = 0.01,
        tolerance: float = 0.0001,
        max_levels: int = 10,
        concurrency: int = 4,
        random_seed: int = 42,
        min_community_size: int = 3,
        persist_assignments: bool = True,
    ) -> List[Dict[str, Any]]:
        del theta, tolerance, concurrency
        proj = self._projected_graphs.get(graph_name)
        if proj is None:
            raise RuntimeError(f"Projected graph not found: {graph_name}")
        if proj.number_of_nodes() == 0:
            return []

        weight_key = str(relationship_weight_property or "").strip() or "weight"
        community_id_seed = 1
        topdown_paths: Dict[str, List[int]] = {}

        def next_community_id() -> int:
            nonlocal community_id_seed
            cid = community_id_seed
            community_id_seed += 1
            return cid

        def assign_terminal(nodes: Set[str], prefix: List[int]) -> None:
            if prefix:
                for node_id in nodes:
                    topdown_paths[node_id] = list(prefix)
                return
            cid = next_community_id()
            for node_id in nodes:
                topdown_paths[node_id] = [cid]

        def recurse(nodes: Set[str], prefix: List[int], depth: int) -> None:
            if not nodes:
                return
            if depth >= max(1, int(max_levels or 10)):
                assign_terminal(nodes, prefix)
                return
            sg = proj.subgraph(nodes).copy()
            if sg.number_of_nodes() < max(2, int(min_community_size or 1) * 2) or sg.number_of_edges() == 0:
                assign_terminal(nodes, prefix)
                return
            try:
                parts = louvain_communities(
                    sg,
                    weight=weight_key if any(weight_key in attrs for *_e, attrs in sg.edges(data=True)) else None,
                    resolution=float(gamma),
                    seed=random_seed + depth,
                )
            except Exception as e:
                logger.warning("Community detection fallback to terminal split failed at depth=%d: %s", depth, e)
                assign_terminal(nodes, prefix)
                return
            parts = [set(part) for part in parts if part]
            if len(parts) <= 1:
                assign_terminal(nodes, prefix)
                return
            for part in parts:
                cid = next_community_id()
                child_prefix = prefix + [cid]
                should_recurse = include_intermediate_communities and len(part) >= max(4, int(min_community_size or 1) * 2)
                if should_recurse:
                    recurse(part, child_prefix, depth + 1)
                else:
                    assign_terminal(part, child_prefix)

        recurse(set(proj.nodes()), [], 0)

        assignments: List[Dict[str, Any]] = []
        graph = self._graph()
        for node_id in sorted(proj.nodes()):
            topdown = topdown_paths.get(node_id, [])
            if not topdown:
                topdown = [next_community_id()]
            leaf_first = list(reversed(topdown))
            assignments.append(
                {
                    "node_id": node_id,
                    "labels": [x for x in self._node_labels(graph.nodes[node_id]) if x != "Entity"],
                    "community_id": int(leaf_first[0]),
                    "community_path": leaf_first,
                }
            )
            if persist_assignments:
                graph.nodes[node_id][write_property] = int(leaf_first[0])
                graph.nodes[node_id][intermediate_property] = leaf_first
        if persist_assignments:
            self._persist()
        return assignments

    def fetch_community_assignments(
        self,
        *,
        write_property: str = "community_id",
        intermediate_property: str = "community_path",
        exclude_node_labels: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        excluded = {str(x).strip() for x in (exclude_node_labels or []) if str(x).strip()}
        out: List[Dict[str, Any]] = []
        for node_id, data in self._iter_node_items():
            if write_property not in data:
                continue
            labels = [x for x in self._node_labels(data) if x != "Entity"]
            if excluded & set(labels):
                continue
            out.append(
                {
                    "node_id": node_id,
                    "labels": labels,
                    "community_id": int(data.get(write_property, 0) or 0),
                    "community_path": [int(x) for x in safe_list(data.get(intermediate_property, [])) if str(x).strip()],
                    "name": safe_str(data.get("name")),
                    "description": safe_str(data.get("description")),
                    "source_documents": _as_list_str(data.get("source_documents", [])),
                    "properties": _safe_properties(data.get("properties", {})),
                }
            )
        return out

    def fetch_internal_relations_by_node_ids(
        self,
        node_ids: List[str],
        *,
        exclude_relation_types: Optional[List[str]] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        clean_ids = {str(x).strip() for x in (node_ids or []) if str(x).strip()}
        excluded = {str(x).strip() for x in (exclude_relation_types or []) if str(x).strip()}
        if not clean_ids:
            return []
        out: List[Dict[str, Any]] = []
        for src, dst, _key, data in self._iter_edges():
            if src not in clean_ids or dst not in clean_ids:
                continue
            pred = safe_str(data.get("predicate") or data.get("relation_type")).strip()
            if pred in excluded:
                continue
            out.append(
                {
                    "id": safe_str(data.get("id")) or stable_relation_id(src, pred or "RELATED_TO", dst),
                    "subject_id": src,
                    "subject_name": safe_str(self._graph().nodes[src].get("name")) or src,
                    "object_id": dst,
                    "object_name": safe_str(self._graph().nodes[dst].get("name")) or dst,
                    "relation_type": pred,
                    "predicate": pred,
                    "description": safe_str(data.get("description")),
                    "confidence": float(data.get("confidence", 1.0) or 1.0),
                    "source_documents": _as_list_str(data.get("source_documents", [])),
                }
            )
            if len(out) >= max(1, int(limit or 100)):
                break
        return out

    def collect_internal_relation_source_documents(
        self,
        node_ids: List[str],
        *,
        exclude_relation_types: Optional[List[str]] = None,
    ) -> List[str]:
        docs: List[str] = []
        for row in self.fetch_internal_relations_by_node_ids(
            node_ids,
            exclude_relation_types=exclude_relation_types,
            limit=max(100, len(node_ids) * 20),
        ):
            docs.extend(_as_list_str(row.get("source_documents", [])))
        return dedupe_list(docs)

    def get_child_communities(self, community_id: str, limit: Optional[int] = None) -> List[Entity]:
        out: List[Entity] = []
        graph = self._graph()
        if not graph.has_node(community_id):
            return []
        for _src, dst, _key, data in graph.out_edges(community_id, keys=True, data=True):
            pred = safe_str(data.get("predicate") or data.get("relation_type")).strip()
            if pred != "COMMUNITY_PARENT_OF" or not graph.has_node(dst):
                continue
            out.append(self._build_entity_from_data(graph.nodes[dst]))
            if limit and len(out) >= int(limit):
                break
        return out

    def get_community_member_entities(
        self,
        community_id: str,
        *,
        entity_types: Optional[List[str]] = None,
        limit: Optional[int] = None,
    ) -> List[Entity]:
        allowed = {str(x).strip() for x in (entity_types or []) if str(x).strip()}
        out: List[Entity] = []
        graph = self._graph()
        if not graph.has_node(community_id):
            return []
        for _src, dst, _key, data in graph.out_edges(community_id, keys=True, data=True):
            pred = safe_str(data.get("predicate") or data.get("relation_type")).strip()
            if pred != "COMMUNITY_CONTAINS" or not graph.has_node(dst):
                continue
            labels = set(self._node_labels(graph.nodes[dst]))
            if allowed and not (allowed & labels):
                continue
            out.append(self._build_entity_from_data(graph.nodes[dst]))
            if limit and len(out) >= int(limit):
                break
        return out

    def list_community_levels(self) -> List[int]:
        levels: Set[int] = set()
        for _node_id, data in self._iter_node_items():
            if "Community" not in self._node_labels(data):
                continue
            props = _safe_properties(data.get("properties", {}))
            level = props.get("level")
            try:
                if level is not None:
                    levels.add(int(level))
            except Exception:
                continue
        return sorted(levels)

    def fetch_all_relations(self, relation_types: Optional[List[str]] = None) -> List[Dict]:
        allowed = {str(x).strip() for x in (relation_types or []) if str(x).strip()}
        rows: List[Dict[str, Any]] = []
        for _src, _dst, key, data in self._iter_edges():
            pred = safe_str(data.get("predicate") or data.get("relation_type")).strip()
            if allowed and pred not in allowed:
                continue
            row = dict(data)
            row.setdefault("id", safe_str(key))
            row["predicate"] = pred
            rows.append(row)
        return rows

    def search_related_entities(
        self,
        source_id: str,
        predicate: Optional[str] = None,
        relation_types: Optional[List[str]] = None,
        entity_types: Optional[List[str]] = None,
        limit: Optional[int] = None,
        return_relations: bool = False,
    ) -> Union[List[Entity], List[Tuple[Entity, Relation]]]:
        graph = self._graph()
        if not graph.has_node(source_id):
            return []
        wanted_pred = safe_str(predicate).strip()
        allowed_entity_types = {str(x).strip() for x in (entity_types or []) if str(x).strip()}
        out: List[Any] = []

        def maybe_append(target_id: str, edge_data: Dict[str, Any], forward: bool) -> None:
            if not graph.has_node(target_id):
                return
            pred = safe_str(edge_data.get("predicate") or edge_data.get("relation_type")).strip()
            if wanted_pred and pred != wanted_pred:
                return
            if relation_types and not self._edge_matches_relation_type(edge_data, relation_types):
                return
            target_data = graph.nodes[target_id]
            if allowed_entity_types and not (allowed_entity_types & set(self._node_labels(target_data))):
                return
            entity = self._build_entity_from_data(target_data)
            relation = self._edge_to_relation(
                source_id if forward else target_id,
                target_id if forward else source_id,
                edge_data,
            )
            out.append((entity, relation) if return_relations else entity)

        for _src, dst, _key, data in graph.out_edges(source_id, keys=True, data=True):
            maybe_append(dst, data, True)
            if limit and len(out) >= int(limit):
                return out[: int(limit)]
        for src, _dst, _key, data in graph.in_edges(source_id, keys=True, data=True):
            maybe_append(src, data, False)
            if limit and len(out) >= int(limit):
                return out[: int(limit)]
        return out[: int(limit)] if limit else out

    def fetch_all_nodes(self, node_types: List[str]) -> List[Dict]:
        wanted = {str(x).strip() for x in (node_types or []) if str(x).strip()}
        rows: List[Dict[str, Any]] = []
        for node_id, data in self._iter_node_items():
            labels = self._node_labels(data)
            if wanted and not (wanted & set(labels)):
                continue
            rows.append(
                {
                    "labels": labels,
                    "id": node_id,
                    "name": safe_str(data.get("name")),
                    "description": safe_str(data.get("description")),
                    "properties": _safe_properties(data.get("properties", {})),
                    "source_documents": _as_list_str(data.get("source_documents", [])),
                }
            )
        return rows

    def get_entity_by_id(self, entity_id: str) -> Optional[Entity]:
        graph = self._graph()
        if not graph.has_node(entity_id):
            return None
        return self._build_entity_from_data(graph.nodes[entity_id])

    def get_entities_by_ids(self, entity_ids: List[str]) -> Dict[str, Entity]:
        out: Dict[str, Entity] = {}
        for entity_id in entity_ids or []:
            ent = self.get_entity_by_id(entity_id)
            if ent is not None:
                out[ent.id] = ent
        return out

    def get_node_details_by_ids(self, node_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        graph = self._graph()
        for node_id in node_ids or []:
            sid = safe_str(node_id).strip()
            if not sid or not graph.has_node(sid):
                continue
            data = graph.nodes[sid]
            out[sid] = {
                "node_id": sid,
                "name": safe_str(data.get("name")),
                "labels": [x for x in self._node_labels(data) if x != "Entity"],
                "description": safe_str(data.get("description") or data.get("summary")),
                "source_documents": _as_list_str(data.get("source_documents", [])),
            }
        return out

    def search_entities_by_type_ranked(
        self,
        entity_type: str,
        keyword: str = "",
        limit: int = 200,
    ) -> List[Dict[str, Any]]:
        et = str(entity_type or "").strip() or "Entity"
        raw_keyword = str(keyword or "").strip()
        keywords = [x.lower() for x in build_search_keywords(raw_keyword, top_k=10) if str(x).strip()]
        if raw_keyword and not keywords:
            keywords = [raw_keyword.lower()]

        ranked_rows: List[Dict[str, Any]] = []
        for _node_id, data in self._iter_node_items():
            labels = set(self._node_labels(data))
            if et != "Entity" and et not in labels:
                continue
            entity = self._build_entity_from_data(data)
            if not keywords:
                ranked_rows.append({"entity": entity, "keyword_score_raw": 0.0, "matched_keyword_count": 0})
                continue

            name_lc = safe_str(data.get("name")).lower()
            aliases_lc = [safe_str(x).lower() for x in _as_list_str(data.get("aliases", []))]
            desc_lc = safe_str(data.get("description")).lower()
            summary_lc = safe_str(data.get("summary")).lower()
            keyword_score_raw = 0.0
            matched_keyword_count = 0
            for kw in keywords:
                token_score = 0
                token_score += 10 if name_lc == kw else 0
                token_score += 9 if any(alias == kw for alias in aliases_lc) else 0
                token_score += 6 if kw and kw in name_lc else 0
                token_score += 5 if any(kw and kw in alias for alias in aliases_lc) else 0
                token_score += 3 if len(kw) >= 2 and name_lc and kw in name_lc else 0
                token_score += 2 if kw and kw in desc_lc else 0
                token_score += 2 if kw and kw in summary_lc else 0
                keyword_score_raw += token_score
                if token_score > 0:
                    matched_keyword_count += 1
            if keyword_score_raw <= 0:
                continue
            ranked_rows.append(
                {
                    "entity": entity,
                    "keyword_score_raw": float(keyword_score_raw),
                    "matched_keyword_count": int(matched_keyword_count),
                }
            )
        ranked_rows.sort(
            key=lambda row: (
                int(row.get("matched_keyword_count", 0) or 0),
                float(row.get("keyword_score_raw", 0.0) or 0.0),
                -len(safe_str(getattr(row.get("entity"), "name", ""))),
            ),
            reverse=True,
        )
        return ranked_rows[: max(1, int(limit or 200))]

    def search_entities_by_type(self, entity_type: str, keyword: str = "", limit: int = 200) -> List[Entity]:
        return [row["entity"] for row in self.search_entities_by_type_ranked(entity_type, keyword=keyword, limit=limit)]

    def get_entity_info(
        self,
        entity_id: str,
        contain_properties: bool = False,
        contain_relations: bool = False,
        resolve_source_documents: bool = False,
    ) -> str:
        ent = self.get_entity_by_id(entity_id)
        if not ent:
            return f"Entity not found: {entity_id}"
        lines = [f"name: {ent.name}", f"id: {ent.id}", f"type: {ent.type}"]
        if ent.description:
            lines.append(f"description: {ent.description}")
        if ent.aliases:
            lines.append(f"aliases: {', '.join([str(x) for x in ent.aliases])}")
        if ent.source_documents:
            lines.append(f"source_documents: {', '.join([str(x) for x in ent.source_documents])}")
            if resolve_source_documents:
                section_lines = self.format_source_document_sections(ent.source_documents)
                if section_lines:
                    lines.append(f"{self.meta.get('section_label', 'Document')} details:")
                    lines.extend(section_lines)
        if contain_properties and ent.properties:
            lines.append("属性:")
            for k, v in self._strip_embedding_fields(ent.properties).items():
                if v not in (None, "", [], {}, ()):
                    lines.append(f"- {k}: {v}")
        if contain_relations:
            rel_pairs = self.search_related_entities(source_id=entity_id, limit=50, return_relations=True)
            if rel_pairs:
                lines.append("关系:")
                for _, rel in rel_pairs:
                    lines.append(f"- ({rel.subject_id})-[{rel.predicate}]->({rel.object_id})")
        return "\n".join(lines)

    @staticmethod
    def _compose_title_key(title: str, subtitle: str) -> str:
        title = safe_str(title).strip()
        subtitle = safe_str(subtitle).strip()
        if title and subtitle:
            return f"{title} / {subtitle}"
        return title or subtitle

    def _load_document_title_index(self) -> None:
        if self._document_title_index_loaded:
            return
        self._document_title_index_loaded = True
        self._document_id2title = {}
        self._document_id2subtitle = {}

        kg_cfg = getattr(getattr(self.graph_store, "config", None), "knowledge_graph_builder", None)
        kg_dir = safe_str(getattr(kg_cfg, "file_path", "")).strip() or "data/knowledge_graph"
        doc2chunks_path = os.path.join(kg_dir, "doc2chunks.json")
        index_path = os.path.join(kg_dir, "doc2chunks_index.json")

        title_field = safe_str(self.meta.get("title", "title")).strip() or "title"
        subtitle_field = safe_str(self.meta.get("subtitle", "subtitle")).strip() or "subtitle"

        try:
            doc2chunks = load_json(doc2chunks_path) if os.path.exists(doc2chunks_path) else {}
        except Exception:
            logger.exception("document title index load failed: %s", doc2chunks_path)
            doc2chunks = {}

        if isinstance(doc2chunks, dict) and doc2chunks:
            for document_id, pack in doc2chunks.items():
                sid = safe_str(document_id).strip()
                if not sid:
                    continue
                md = safe_dict((pack or {}).get("document_metadata"))
                title = safe_str(md.get(title_field) or md.get("title") or md.get("doc_title") or md.get("source_title") or "").strip()
                subtitle = safe_str(md.get(subtitle_field) or md.get("subtitle") or md.get("sub_title") or md.get("sub_scene_name") or "").strip()
                self._document_id2title[sid] = title
                self._document_id2subtitle[sid] = subtitle
            return

        try:
            fallback_index = load_json(index_path) if os.path.exists(index_path) else {}
        except Exception:
            logger.exception("document title fallback index load failed: %s", index_path)
            fallback_index = {}

        if not isinstance(fallback_index, dict):
            return
        for sid, title_key in safe_dict(fallback_index.get("key2title")).items():
            clean_sid = safe_str(sid).strip()
            clean_title = safe_str(title_key).strip()
            if not clean_sid or not clean_title:
                continue
            self._document_id2title[clean_sid] = clean_title
            self._document_id2subtitle[clean_sid] = ""

    def resolve_document_ids_to_section_records(
        self,
        document_ids: List[str],
        *,
        deduplicate: bool = True,
    ) -> List[Dict[str, Any]]:
        self._load_document_title_index()
        clean_ids: List[str] = []
        seen_ids: Set[str] = set()
        for item in document_ids or []:
            sid = safe_str(item).strip()
            if not sid or sid in seen_ids:
                continue
            seen_ids.add(sid)
            clean_ids.append(sid)
        if not clean_ids:
            return []
        if not deduplicate:
            rows: List[Dict[str, Any]] = []
            for sid in clean_ids:
                title = safe_str(self._document_id2title.get(sid, "")).strip()
                subtitle = safe_str(self._document_id2subtitle.get(sid, "")).strip()
                rows.append(
                    {
                        "document_ids": [sid],
                        "title": title,
                        "subtitle": subtitle,
                        "title_key": self._compose_title_key(title, subtitle) or sid,
                    }
                )
            return rows
        grouped: Dict[str, Dict[str, Any]] = {}
        order: List[str] = []
        for sid in clean_ids:
            title = safe_str(self._document_id2title.get(sid, "")).strip()
            subtitle = safe_str(self._document_id2subtitle.get(sid, "")).strip()
            title_key = self._compose_title_key(title, subtitle) or sid
            if title_key not in grouped:
                grouped[title_key] = {
                    "document_ids": [],
                    "title": title,
                    "subtitle": subtitle,
                    "title_key": title_key,
                }
                order.append(title_key)
            grouped[title_key]["document_ids"].append(sid)
        return [grouped[key] for key in order]

    def format_source_document_sections(
        self,
        document_ids: List[str],
        *,
        include_document_ids: bool = False,
    ) -> List[str]:
        rows = self.resolve_document_ids_to_section_records(document_ids, deduplicate=True)
        lines: List[str] = []
        for row in rows:
            title = safe_str(row.get("title")).strip()
            subtitle = safe_str(row.get("subtitle")).strip()
            if title and subtitle:
                line = f"- {title} / {subtitle}"
            elif title:
                line = f"- {title}"
            elif subtitle:
                line = f"- {subtitle}"
            else:
                line = f"- {row.get('title_key', '')}"
            if include_document_ids:
                ids = [safe_str(x).strip() for x in safe_list(row.get("document_ids")) if safe_str(x).strip()]
                if ids:
                    line += f" [document_ids: {', '.join(ids)}]"
            lines.append(line)
        return lines

    def _entity_surface_texts(self, ent: Entity) -> List[str]:
        texts: List[str] = []
        for value in [getattr(ent, "name", "")] + list(getattr(ent, "aliases", []) or []):
            raw = safe_str(value).strip()
            if raw and raw not in texts:
                texts.append(raw)
        return texts

    def _variant_surface_score(self, seed: str, candidate: str) -> float:
        seed_norm = _normalize_entity_surface(seed)
        cand_norm = _normalize_entity_surface(candidate)
        if not seed_norm or not cand_norm:
            return 0.0
        if seed_norm == cand_norm:
            return 1.0
        short, long_ = (seed_norm, cand_norm) if len(seed_norm) <= len(cand_norm) else (cand_norm, seed_norm)
        if len(short) < 2 or short not in long_:
            return 0.0
        extra = max(0, len(long_) - len(short))
        if extra <= 4:
            return 0.85
        if extra <= 8:
            return 0.65
        return 0.0

    def find_name_variant_entities(self, entity_id: str, *, limit: int = 10) -> List[Entity]:
        ent = self.get_entity_by_id(entity_id)
        if not ent:
            return []
        seed_texts = self._entity_surface_texts(ent)
        if not seed_texts:
            return []
        candidates: Dict[str, Tuple[float, Entity]] = {}
        query_limit = max(12, int(limit or 10) * 5)
        for seed in seed_texts:
            ranked_rows = self.search_entities_by_type_ranked("Entity", keyword=seed, limit=query_limit) or []
            for row in ranked_rows:
                other = row.get("entity")
                if other is None or getattr(other, "id", None) == ent.id:
                    continue
                if not _is_base_entity_label(getattr(other, "type", [])):
                    continue
                other_name = safe_str(getattr(other, "name", "")).strip()
                if not _looks_like_base_entity_name(other_name):
                    continue
                other_texts = self._entity_surface_texts(other)
                best_score = 0.0
                for seed_text in seed_texts:
                    for other_text in other_texts:
                        best_score = max(best_score, self._variant_surface_score(seed_text, other_text))
                if best_score < 0.8:
                    continue
                prev = candidates.get(other.id)
                if prev is None or best_score > prev[0]:
                    candidates[other.id] = (best_score, other)
        rows = sorted(
            candidates.values(),
            key=lambda item: (item[0], len(safe_str(getattr(item[1], "name", "")).strip()), safe_str(getattr(item[1], "name", "")).strip()),
            reverse=True,
        )
        return [item[1] for item in rows[: max(0, int(limit or 10))]]

    def get_entity_section_info(self, entity_id: str, *, include_document_ids: bool = False) -> str:
        ent = self.get_entity_by_id(entity_id)
        if not ent:
            return f"Entity not found: {entity_id}"
        variant_entities = self.find_name_variant_entities(entity_id, limit=10)
        merged_document_ids = dedupe_list(
            list(ent.source_documents or [])
            + [doc_id for other in variant_entities for doc_id in safe_list(getattr(other, "source_documents", []) or [])]
        )
        lines = [f"name: {ent.name}", f"id: {ent.id}"]
        if ent.source_documents:
            if include_document_ids:
                lines.append(f"direct_source_documents: {', '.join([str(x) for x in ent.source_documents])}")
            else:
                direct_section_count = len(self.resolve_document_ids_to_section_records(ent.source_documents, deduplicate=True))
                lines.append(f"direct_{self.meta.get('section_label', 'Document').lower()}_count: {direct_section_count}")
        if variant_entities:
            lines.append("possible coreferent variants:")
            for other in variant_entities:
                other_types = getattr(other, "type", [])
                if isinstance(other_types, list):
                    type_text = ", ".join([safe_str(x).strip() for x in other_types if safe_str(x).strip()])
                else:
                    type_text = safe_str(other_types).strip()
                line = f"- {other.name} (id: {other.id})"
                if type_text:
                    line += f" [type: {type_text}]"
                lines.append(line)
        if merged_document_ids:
            if include_document_ids:
                lines.append(f"合并后的document_id为：{', '.join([str(x) for x in merged_document_ids])}")
            else:
                merged_section_count = len(self.resolve_document_ids_to_section_records(merged_document_ids, deduplicate=True))
                lines.append(f"合并后的{self.meta.get('section_label', 'Document')}数：{merged_section_count}")
        section_lines = self.format_source_document_sections(merged_document_ids, include_document_ids=include_document_ids)
        if section_lines:
            lines.append(f"相关{self.meta.get('section_label', 'Document')}信息：")
            lines.extend(section_lines)
        else:
            lines.append(f"未找到可解析的{self.meta.get('section_label', 'Document')}标题。")
        return "\n".join(lines)

    def get_relation_summary(self, src_id: str, tgt_id: str, relation_type: str) -> str:
        graph = self._graph()
        for _src, _dst, _key, data in graph.edges(src_id, keys=True, data=True):
            if _dst != tgt_id:
                continue
            pred = safe_str(data.get("predicate") or data.get("relation_type")).strip()
            if pred != relation_type:
                continue
            s_name = safe_str(graph.nodes[src_id].get("name")) or src_id
            t_name = safe_str(graph.nodes[tgt_id].get("name")) or tgt_id
            r_name = safe_str(data.get("relation_name")).strip()
            desc = safe_str(data.get("description")).strip()
            mid = f"{pred}({r_name})" if r_name else pred
            return f"{s_name} -[{mid}]-> {t_name}：{desc}" if desc else f"{s_name} -[{mid}]-> {t_name}"
        return ""

    def get_common_neighbors(
        self,
        id1: str,
        id2: str,
        rel_types: Optional[List[str]] = None,
        direction: str = "any",
        limit: Optional[int] = None,
    ) -> List[Entity]:
        rows = self.get_common_neighbors_with_rels(id1, id2, rel_types=rel_types, direction=direction, limit=limit)
        return [row["entity"] for row in rows]

    def get_common_neighbors_with_rels(
        self,
        id1: str,
        id2: str,
        rel_types: Optional[List[str]] = None,
        direction: str = "any",
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        graph = self._graph()
        if not graph.has_node(id1) or not graph.has_node(id2):
            return []
        direction = str(direction or "any").lower()
        allowed = {str(x).strip() for x in (rel_types or []) if str(x).strip()}

        def side_neighbors(node_id: str, side: str) -> Dict[str, List[str]]:
            out: Dict[str, List[str]] = defaultdict(list)
            if side in {"out", "any"}:
                for _src, dst, _key, data in graph.out_edges(node_id, keys=True, data=True):
                    pred = safe_str(data.get("predicate") or data.get("relation_type")).strip()
                    if allowed and pred not in allowed:
                        continue
                    out[dst].append(pred)
            if side in {"in", "any"}:
                for src, _dst, _key, data in graph.in_edges(node_id, keys=True, data=True):
                    pred = safe_str(data.get("predicate") or data.get("relation_type")).strip()
                    if allowed and pred not in allowed:
                        continue
                    out[src].append(pred)
            return out

        if direction == "out":
            a_map = side_neighbors(id1, "out")
            b_map = side_neighbors(id2, "out")
        elif direction == "in":
            a_map = side_neighbors(id1, "in")
            b_map = side_neighbors(id2, "in")
        else:
            a_map = side_neighbors(id1, "any")
            b_map = side_neighbors(id2, "any")

        common_ids = sorted(set(a_map) & set(b_map))
        out: List[Dict[str, Any]] = []
        for node_id in common_ids:
            out.append(
                {
                    "entity": self._build_entity_from_data(graph.nodes[node_id]),
                    "rels_from_a": dedupe_list(a_map.get(node_id, [])),
                    "rels_from_b": dedupe_list(b_map.get(node_id, [])),
                }
            )
            if limit and len(out) >= int(limit):
                break
        return out

    def query_similar_entities(
        self,
        text: str,
        top_k: int = 5,
        normalize: bool = False,
        label_filter: Optional[List[str] | str] = None,
    ) -> List[Dict[str, Any]]:
        if self.model is None:
            raise RuntimeError("Embedding model is not loaded. Call load_embedding_model first.")
        qv = _flatten_vector(self.model.encode(text or ""))
        if not qv:
            return []
        if normalize:
            arr = np.asarray(qv, dtype=float)
            norm = float(np.linalg.norm(arr))
            if norm > 1e-12:
                qv = (arr / norm).tolist()
        labels: List[str] = []
        if isinstance(label_filter, str):
            s = label_filter.strip()
            if s:
                labels = [s]
        elif isinstance(label_filter, list):
            labels = [str(x).strip() for x in label_filter if str(x).strip()]
        scored: List[Dict[str, Any]] = []
        q = np.asarray(qv, dtype=float)
        qn = float(np.linalg.norm(q))
        if qn <= 1e-12:
            return []
        for node_id, data in self._iter_node_items():
            node_labels = [x for x in self._node_labels(data) if x != "Entity"]
            if labels and not (set(labels) & set(node_labels)):
                continue
            ev = _flatten_vector(data.get(self.embedding_field))
            if not ev:
                continue
            e = np.asarray(ev, dtype=float)
            if e.shape != q.shape:
                continue
            en = float(np.linalg.norm(e))
            if en <= 1e-12:
                continue
            sim = float(np.dot(q, e) / (qn * en))
            scored.append(
                {
                    "id": node_id,
                    "name": safe_str(data.get("name")),
                    "labels": node_labels,
                    "description": safe_str(data.get("description") or data.get("summary")),
                    "score": sim,
                }
            )
        scored.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        return scored[: max(1, int(top_k or 5))]

    def query_similar_relations(
        self,
        text: str,
        top_k: int = 5,
        min_score: float = -1.0,
        normalize: bool = False,
        relation_type: str = "",
        predicate: str = "",
    ) -> List[Dict[str, Any]]:
        qtxt = safe_str(text).strip()
        if qtxt and self.model is None:
            raise RuntimeError("Embedding model is not loaded. Call load_embedding_model first.")
        qv: List[float] = []
        if qtxt:
            qv = _flatten_vector(self.model.encode(qtxt))
            if not qv:
                return []
            if normalize:
                arr = np.asarray(qv, dtype=float)
                norm = float(np.linalg.norm(arr))
                if norm > 1e-12:
                    qv = (arr / norm).tolist()
        rel_type = safe_str(relation_type).strip() or safe_str(predicate).strip()
        scored: List[Dict[str, Any]] = []
        q = np.asarray(qv, dtype=float) if qv else None
        qn = float(np.linalg.norm(q)) if q is not None else 0.0
        if q is not None and qn <= 1e-12:
            return []
        for src, dst, key, data in self._iter_edges():
            pred = safe_str(data.get("predicate") or data.get("relation_type")).strip()
            if rel_type and pred != rel_type:
                continue
            desc = safe_str(data.get("description")).strip()
            if not desc:
                continue
            sim = 0.0
            if q is not None:
                ev = _flatten_vector(data.get(self.embedding_field))
                if not ev:
                    continue
                e = np.asarray(ev, dtype=float)
                if e.shape != q.shape:
                    continue
                en = float(np.linalg.norm(e))
                if en <= 1e-12:
                    continue
                sim = float(np.dot(q, e) / (qn * en))
                if sim < float(min_score):
                    continue
            scored.append(
                {
                    "relation_id": safe_str(data.get("id")) or safe_str(key),
                    "subject_id": src,
                    "subject_name": safe_str(self._graph().nodes[src].get("name")),
                    "object_id": dst,
                    "object_name": safe_str(self._graph().nodes[dst].get("name")),
                    "predicate": pred,
                    "relation_name": safe_str(data.get("relation_name")),
                    "description": desc,
                    "source_documents": _as_list_str(data.get("source_documents", [])),
                    "score": sim,
                }
            )
        scored.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        return scored[: max(1, int(top_k or 5))]

    def find_paths_between_nodes(
        self,
        src_id: str,
        dst_id: str,
        max_depth: int = 4,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        graph = self._graph()
        if not graph.has_node(src_id) or not graph.has_node(dst_id):
            return []
        max_depth = int(max(1, max_depth))
        ug = nx.Graph()
        rep_edges: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for src, dst, key, data in self._iter_edges():
            ug.add_edge(src, dst)
            pair = tuple(sorted((src, dst)))
            rep_edges.setdefault(pair, {"key": key, "data": data, "start": src, "end": dst})
        out: List[Dict[str, Any]] = []
        try:
            for path in nx.all_simple_paths(ug, source=src_id, target=dst_id, cutoff=max_depth):
                nodes = [self._node_to_minimal_dict(graph.nodes[node_id]) for node_id in path]
                rels: List[Dict[str, Any]] = []
                for a, b in zip(path[:-1], path[1:]):
                    rep = rep_edges.get(tuple(sorted((a, b))))
                    if not rep:
                        continue
                    rr = self._rel_to_minimal_dict(rep["data"])
                    rr["start"] = rep["start"]
                    rr["end"] = rep["end"]
                    rels.append(rr)
                out.append({"length": len(rels), "nodes": nodes, "relationships": rels})
                if len(out) >= max(1, int(limit or 5)):
                    break
        except nx.NetworkXNoPath:
            return []
        return out

    def get_k_hop_subgraph(self, center_ids: List[str], k: int = 2, limit_nodes: int = 200) -> Dict[str, Any]:
        graph = self._graph()
        clean_centers = [str(x).strip() for x in (center_ids or []) if str(x).strip() and graph.has_node(str(x).strip())]
        if not clean_centers:
            return {"nodes": [], "relationships": []}
        k = int(max(1, k))
        limit_nodes = int(max(1, limit_nodes))
        ug = graph.to_undirected()
        visited: List[str] = []
        seen: Set[str] = set()
        queue = deque((center, 0) for center in clean_centers)
        while queue and len(visited) < limit_nodes:
            node_id, depth = queue.popleft()
            if node_id in seen:
                continue
            seen.add(node_id)
            visited.append(node_id)
            if depth >= k:
                continue
            for nbr in ug.neighbors(node_id):
                if nbr not in seen:
                    queue.append((nbr, depth + 1))
        node_dicts = [self._node_to_minimal_dict(graph.nodes[node_id]) for node_id in visited]
        rel_dicts: List[Dict[str, Any]] = []
        allowed = set(visited)
        for src, dst, key, data in self._iter_edges():
            if src not in allowed or dst not in allowed:
                continue
            rr = self._rel_to_minimal_dict(data)
            rr["id"] = rr.get("id") or safe_str(key)
            rr["start"] = src
            rr["end"] = dst
            rel_dicts.append(rr)
        return {"nodes": node_dicts, "relationships": rel_dicts}

    def top_k_by_centrality(
        self,
        metric: str,
        top_k: int = 50,
        node_labels: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        metric_map = {"pagerank": "pagerank", "degree": "degree", "betweenness": "betweenness"}
        m = metric_map.get(str(metric or "").lower())
        if not m:
            return []
        labels = {str(x).strip() for x in (node_labels or []) if str(x).strip()}
        rows: List[Dict[str, Any]] = []
        for node_id, data in self._iter_node_items():
            if m not in data:
                continue
            node_types = [x for x in self._node_labels(data) if x != "Entity"]
            if labels and not (labels & set(node_types)):
                continue
            rows.append(
                {
                    "id": node_id,
                    "name": safe_str(data.get("name")),
                    "labels": node_types,
                    "score": data.get(m),
                }
            )
        rows.sort(key=lambda x: float(x.get("score", 0.0) or 0.0), reverse=True)
        return rows[: max(1, int(top_k or 50))]

    def find_co_section_entities(
        self,
        entity_id: str,
        include_types: Optional[List[str]] = None,
    ) -> List[Entity]:
        graph = self._graph()
        if not graph.has_node(entity_id):
            return []
        docs = set(_as_list_str(graph.nodes[entity_id].get("source_documents", [])))
        allowed = {str(x).strip() for x in (include_types or []) if str(x).strip()}
        out: List[Entity] = []
        for other_id, data in self._iter_node_items():
            if other_id == entity_id:
                continue
            if not docs.intersection(_as_list_str(data.get("source_documents", []))):
                continue
            labels = set(self._node_labels(data))
            if allowed and not (allowed & labels):
                continue
            out.append(self._build_entity_from_data(data))
        return out

    def find_co_section_entities_grouped(
        self,
        entity_id: str,
        *,
        include_types: Optional[List[str]] = None,
        document_ids: Optional[List[str]] = None,
        max_sections: Optional[int] = None,
        max_entities_per_section: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        graph = self._graph()
        if not graph.has_node(entity_id):
            return []
        focus_docs = set(_as_list_str(graph.nodes[entity_id].get("source_documents", [])))
        if document_ids:
            focus_docs &= {str(x).strip() for x in document_ids if str(x).strip()}
        allowed = {str(x).strip() for x in (include_types or []) if str(x).strip()}
        section_label = str(self.meta.get("section_label", "Document")).strip() or "Document"
        by_doc: Dict[str, Dict[str, Any]] = {}
        for doc_id in sorted(focus_docs):
            by_doc[doc_id] = {
                "group_key": doc_id,
                "section_id": "",
                "section_name": "",
                "section_description": "",
                "document_ids": [doc_id],
                "entities": [],
            }
        for node_id, data in self._iter_node_items():
            if section_label in self._node_labels(data):
                node_docs = set(_as_list_str(data.get("source_documents", [])))
                for doc_id in focus_docs & node_docs:
                    by_doc[doc_id]["section_id"] = node_id
                    by_doc[doc_id]["section_name"] = safe_str(data.get("name"))
                    by_doc[doc_id]["section_description"] = safe_str(data.get("description"))
        for other_id, data in self._iter_node_items():
            if other_id == entity_id:
                continue
            labels = set(self._node_labels(data))
            if allowed and not (allowed & labels):
                continue
            other_docs = set(_as_list_str(data.get("source_documents", [])))
            for doc_id in focus_docs & other_docs:
                by_doc[doc_id]["entities"].append(self._build_entity_from_data(data))
        rows = list(by_doc.values())
        if max_entities_per_section is not None:
            for row in rows:
                row["entities"] = row["entities"][: max(1, int(max_entities_per_section))]
        if max_sections is not None:
            rows = rows[: max(1, int(max_sections))]
        return rows

    def process_all_embeddings(
        self,
        entity_types: List[str] = None,
        exclude_entity_types: List[str] = [],
        relation_types: Optional[List[str]] = None,
        exclude_relation_types: Optional[List[str]] = None,
        max_workers: int = None,
    ):
        entity_types = entity_types or []
        exclude_entity_types = exclude_entity_types or []
        exclude_relation_types = exclude_relation_types or []

        if not entity_types:
            entity_types = self.list_entity_types()
        entity_types = [t for t in entity_types if t not in set(exclude_entity_types)]
        if max_workers is None:
            cpu = os.cpu_count() or 4
            max_workers = min(32, cpu * 4)

        graph = self._graph()
        entity_store = self._get_entity_vector_store()
        relation_store = self._get_relation_vector_store()

        node_ids: List[str] = []
        node_inputs: List[Tuple[str, Dict[str, Any]]] = []
        for node_id, data in self._iter_node_items():
            labels = set(self._node_labels(data))
            if entity_types and not (set(entity_types) & labels):
                continue
            node_ids.append(node_id)
            node_inputs.append((node_id, dict(data)))

        node_store_rows = entity_store.get_records_by_ids(node_ids, include_embeddings=True) if entity_store is not None else []
        node_store_map = {safe_str(row.get("id")): row for row in node_store_rows if safe_str(row.get("id"))}

        node_candidates: List[Tuple[str, Dict[str, Any]]] = []
        node_vector_upserts: Dict[str, Dict[str, Any]] = {}
        synced_nodes_from_store = 0
        backfilled_nodes_to_store = 0
        graph_dirty = False
        for node_id, data in node_inputs:
            graph_embedding = _as_1d_embedding(data.get(self.embedding_field))
            store_embedding = _as_1d_embedding((node_store_map.get(node_id) or {}).get("embedding"))
            if graph_embedding is not None:
                if store_embedding is None and entity_store is not None:
                    node_vector_upserts[node_id] = self._build_entity_vector_record(node_id, data, graph_embedding)
                    backfilled_nodes_to_store += 1
                continue
            if store_embedding is not None and graph.has_node(node_id):
                graph.nodes[node_id][self.embedding_field] = store_embedding
                synced_nodes_from_store += 1
                graph_dirty = True
                continue
            node_candidates.append((node_id, data))

        failed_nodes = 0
        if node_candidates:
            if self.model is None:
                self.load_embedding_model(self.graph_store.config.embedding)
            print("🚀 开始处理节点嵌入...")
            print(f"📌 实体类型标签: {entity_types}")
            updates: List[Tuple[str, List[float]]] = []
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futures = {ex.submit(self.encode_node_embedding, node): node_id for node_id, node in node_candidates}
                for fut in tqdm(as_completed(futures), total=len(futures), desc="Encoding Nodes", ncols=80):
                    node_id = futures[fut]
                    try:
                        updates.append((node_id, fut.result()))
                    except Exception as e:
                        failed_nodes += 1
                        print(f"⚠️ Node {node_id} embedding failed: {e}")
            for node_id, embedding in updates:
                vector = _as_1d_embedding(embedding)
                if vector is None or not graph.has_node(node_id):
                    continue
                graph.nodes[node_id][self.embedding_field] = vector
                node_vector_upserts[node_id] = self._build_entity_vector_record(node_id, graph.nodes[node_id], vector)
                graph_dirty = True
        if node_vector_upserts and entity_store is not None:
            entity_store.upsert_records(list(node_vector_upserts.values()))
        logger.info(
            "Entity embedding cache sync done: encoded=%d synced_from_store=%d backfilled_to_store=%d failed=%d",
            len(node_candidates),
            synced_nodes_from_store,
            backfilled_nodes_to_store,
            failed_nodes,
        )

        allowed_relation_types = {str(x).strip() for x in (relation_types or []) if str(x).strip()}
        excluded_relation_types = {str(x).strip() for x in (exclude_relation_types or []) if str(x).strip()}
        edge_ids: List[str] = []
        edge_inputs: List[Tuple[str, str, str, str, Dict[str, Any]]] = []
        edge_candidates: List[Tuple[str, str, str, Dict[str, Any]]] = []
        for src, dst, key, data in self._iter_edges():
            pred = safe_str(data.get("predicate") or data.get("relation_type")).strip()
            if allowed_relation_types and pred not in allowed_relation_types:
                continue
            if pred in excluded_relation_types:
                continue
            rel_id = safe_str(data.get("id")) or safe_str(key)
            if not rel_id:
                continue
            edge_ids.append(rel_id)
            edge_inputs.append((src, dst, key, rel_id, dict(data)))

        edge_store_rows = relation_store.get_records_by_ids(edge_ids, include_embeddings=True) if relation_store is not None else []
        edge_store_map = {safe_str(row.get("id")): row for row in edge_store_rows if safe_str(row.get("id"))}

        failed_rels = 0
        skipped_rels = 0
        synced_rels_from_store = 0
        backfilled_rels_to_store = 0
        relation_vector_upserts: Dict[str, Dict[str, Any]] = {}
        for src, dst, key, rel_id, data in edge_inputs:
            graph_embedding = _as_1d_embedding(data.get(self.embedding_field))
            store_embedding = _as_1d_embedding((edge_store_map.get(rel_id) or {}).get("embedding"))
            if graph_embedding is not None:
                if store_embedding is None and relation_store is not None:
                    relation_vector_upserts[rel_id] = self._build_relation_vector_record(rel_id, src, dst, data, graph_embedding)
                    backfilled_rels_to_store += 1
                continue
            if store_embedding is not None and self._graph().has_edge(src, dst, key):
                self._graph()[src][dst][key][self.embedding_field] = store_embedding
                synced_rels_from_store += 1
                graph_dirty = True
                continue
            edge_candidates.append((src, dst, key, data))

        if edge_candidates:
            if self.model is None:
                self.load_embedding_model(self.graph_store.config.embedding)
            print("🚀 开始处理关系嵌入...")
            updates: List[Tuple[str, str, str, List[float]]] = []
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futures = {ex.submit(self.encode_relation_embedding, data): (src, dst, key) for src, dst, key, data in edge_candidates}
                for fut in tqdm(as_completed(futures), total=len(futures), desc="Encoding Relations", ncols=80):
                    src, dst, key = futures[fut]
                    try:
                        embedding = fut.result()
                        if embedding is None:
                            skipped_rels += 1
                            continue
                        updates.append((src, dst, key, embedding))
                    except Exception as e:
                        failed_rels += 1
                        print(f"⚠️ Relation {key} embedding failed: {e}")
            for src, dst, key, embedding in updates:
                vector = _as_1d_embedding(embedding)
                if vector is None or not self._graph().has_edge(src, dst, key):
                    continue
                rel_id = safe_str(self._graph()[src][dst][key].get("id")) or safe_str(key)
                self._graph()[src][dst][key][self.embedding_field] = vector
                relation_vector_upserts[rel_id] = self._build_relation_vector_record(
                    rel_id,
                    src,
                    dst,
                    self._graph()[src][dst][key],
                    vector,
                )
                graph_dirty = True
        if relation_vector_upserts and relation_store is not None:
            relation_store.upsert_records(list(relation_vector_upserts.values()))
        if graph_dirty:
            self._persist()
        logger.info(
            "Relation embedding cache sync done: encoded=%d synced_from_store=%d backfilled_to_store=%d skipped=%d failed=%d",
            len(edge_candidates),
            synced_rels_from_store,
            backfilled_rels_to_store,
            skipped_rels,
            failed_rels,
        )

    def ensure_entity_superlabel(self):
        graph = self._graph()
        changed = 0
        for _node_id, data in self._iter_node_items():
            if data.get(self.embedding_field) is None:
                continue
            types = self._node_labels(data)
            if "Entity" not in types:
                data["type"] = dedupe_list(types + ["Entity"])
                changed += 1
        if changed:
            self._persist()
        print("[✓] 已为所有含 embedding 的节点添加超标签 :Entity")

    def create_vector_index(self, index_name="entityEmbeddingIndex", similarity="cosine"):
        logger.info(
            "Local vector index is implicit in NetworkX backend. name=%s similarity=%s dim=%s",
            index_name,
            similarity,
            self.dim,
        )

    def save_to_graph_store(self, entities: List[Dict[str, Any]], relations: List[Dict[str, Any]]) -> None:
        upsert_entities: List[Entity] = []
        upsert_relations: List[Relation] = []
        for e in entities:
            if not isinstance(e, dict):
                continue
            try:
                upsert_entities.append(Entity(**e))
            except Exception:
                continue
        for r in relations:
            if not isinstance(r, dict):
                continue
            pred = safe_str(r.get("predicate") or r.get("relation_type")).strip()
            if not pred or _is_none_relation(pred):
                continue
            row = dict(r)
            row["predicate"] = pred
            row.setdefault("confidence", 1.0)
            if not isinstance(row.get("properties"), dict):
                row["properties"] = {}
            try:
                upsert_relations.append(Relation(**row))
            except Exception:
                continue
        if upsert_entities:
            self.graph_store.upsert_entities(upsert_entities)
        if upsert_relations:
            self.graph_store.upsert_relations(upsert_relations)

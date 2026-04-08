"""
Local graph storage module backed by NetworkX.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
import json
import logging
import os
import pickle
import tempfile

import networkx as nx

from ..models.data import Entity, KnowledgeGraph, Relation
from ..utils.config import KAGConfig

logger = logging.getLogger(__name__)


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


def _merge_unique_list(left: Any, right: Any) -> List[str]:
    out: List[str] = []
    seen = set()
    for seq in (_as_list_str(left), _as_list_str(right)):
        for item in seq:
            if item in seen:
                continue
            seen.add(item)
            out.append(item)
    return out


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

class GraphStore:
    """Local runtime graph storage backed by a pickled NetworkX MultiDiGraph."""

    def __init__(self, config: KAGConfig) -> None:
        self.config = config
        self.path = self._resolve_store_path()
        self.graph: nx.MultiDiGraph = nx.MultiDiGraph()
        self._load()

    def _resolve_store_path(self) -> str:
        raw = str(getattr(getattr(self.config, "storage", None), "graph_store_path", "") or "").strip()
        if raw:
            return raw
        kg_dir = str(getattr(getattr(self.config, "knowledge_graph_builder", None), "file_path", "") or "").strip()
        if not kg_dir:
            kg_dir = "data/knowledge_graph"
        return os.path.join(kg_dir, "graph_runtime.pkl")

    def _load(self) -> None:
        path = Path(self.path)
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            self.graph = nx.MultiDiGraph()
            logger.info("Initialized empty local graph store at %s", self.path)
            return
        try:
            with path.open("rb") as f:
                obj = pickle.load(f)
            if isinstance(obj, nx.MultiDiGraph):
                self.graph = obj
            elif isinstance(obj, nx.DiGraph):
                self.graph = nx.MultiDiGraph(obj)
            else:
                raise TypeError(f"Unsupported graph object type: {type(obj)!r}")
            logger.info(
                "Loaded local graph store from %s (nodes=%d edges=%d)",
                self.path,
                self.graph.number_of_nodes(),
                self.graph.number_of_edges(),
            )
        except Exception as e:
            logger.exception("Failed to load local graph store %s: %s", self.path, e)
            self.graph = nx.MultiDiGraph()

    def persist(self) -> None:
        path = Path(self.path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("wb", delete=False, dir=str(path.parent), prefix=path.name + ".", suffix=".tmp") as f:
            pickle.dump(self.graph, f)
            tmp_path = f.name
        os.replace(tmp_path, self.path)

    def close(self) -> None:
        self.persist()
        logger.info("Local graph store persisted to %s", self.path)

    def get_graph(self) -> nx.MultiDiGraph:
        return self.graph

    def reset_knowledge_graph(self) -> None:
        self.graph = nx.MultiDiGraph()
        self.persist()
        logger.info("Local graph store reset.")

    def _entity_to_attrs(self, entity: Entity, default_type: str = "Concept") -> Dict[str, Any]:
        raw_types = entity.type if getattr(entity, "type", None) is not None else default_type
        types = _as_list_str(raw_types)
        if not types:
            types = [default_type]
        attrs = {
            "id": entity.id,
            "name": str(getattr(entity, "name", "") or ""),
            "type": types,
            "aliases": _as_list_str(getattr(entity, "aliases", []) or []),
            "properties": _safe_properties(getattr(entity, "properties", {}) or {}),
            "description": str(getattr(entity, "description", "") or ""),
            "scope": str(getattr(entity, "scope", "") or ""),
            "source_documents": _as_list_str(getattr(entity, "source_documents", []) or []),
        }
        version = getattr(entity, "version", None)
        if version is not None:
            attrs["version"] = version
        return attrs

    def _relation_to_attrs(self, relation: Relation) -> Dict[str, Any]:
        attrs = {
            "id": relation.id,
            "subject_id": relation.subject_id,
            "object_id": relation.object_id,
            "predicate": str(getattr(relation, "predicate", "") or ""),
            "relation_name": getattr(relation, "relation_name", None),
            "persistence": getattr(relation, "persistence", None),
            "description": getattr(relation, "description", None),
            "confidence": float(getattr(relation, "confidence", 1.0) or 1.0),
            "properties": _safe_properties(getattr(relation, "properties", {}) or {}),
            "source_documents": _as_list_str(getattr(relation, "source_documents", []) or []),
        }
        return attrs

    def _merge_node_attrs(self, existing: Dict[str, Any], incoming: Dict[str, Any]) -> Dict[str, Any]:
        merged = dict(existing)
        for key, value in incoming.items():
            if key in {"aliases", "source_documents", "type"}:
                merged[key] = _merge_unique_list(existing.get(key), value)
                continue
            if key == "properties":
                props = _safe_properties(existing.get("properties", {}))
                props.update(_safe_properties(value))
                merged["properties"] = props
                continue
            if value in (None, "", [], {}, ()):
                if key not in merged:
                    merged[key] = value
                continue
            merged[key] = value
        return merged

    def _merge_edge_attrs(self, existing: Dict[str, Any], incoming: Dict[str, Any]) -> Dict[str, Any]:
        merged = dict(existing)
        for key, value in incoming.items():
            if key == "source_documents":
                merged[key] = _merge_unique_list(existing.get(key), value)
                continue
            if key == "properties":
                props = _safe_properties(existing.get("properties", {}))
                props.update(_safe_properties(value))
                merged["properties"] = props
                continue
            if value in (None, "", [], {}, ()):
                if key not in merged:
                    merged[key] = value
                continue
            merged[key] = value
        return merged

    def _store_entity(self, _session: Any, entity: Entity, default_type: str = "Concept") -> None:
        attrs = self._entity_to_attrs(entity, default_type=default_type)
        node_id = attrs["id"]
        existing = dict(self.graph.nodes[node_id]) if self.graph.has_node(node_id) else {}
        self.graph.add_node(node_id, **self._merge_node_attrs(existing, attrs))

    def _store_relation(self, _session: Any, relation: Relation) -> None:
        attrs = self._relation_to_attrs(relation)
        subject_id = attrs["subject_id"]
        object_id = attrs["object_id"]
        if not self.graph.has_node(subject_id):
            self.graph.add_node(subject_id, id=subject_id, name=subject_id, type=["Entity"], aliases=[], properties={}, description="", scope="", source_documents=[])
        if not self.graph.has_node(object_id):
            self.graph.add_node(object_id, id=object_id, name=object_id, type=["Entity"], aliases=[], properties={}, description="", scope="", source_documents=[])
        edge_key = attrs["id"]
        existing = dict(self.graph.get_edge_data(subject_id, object_id, edge_key, default={}) or {})
        self.graph.add_edge(subject_id, object_id, key=edge_key, **self._merge_edge_attrs(existing, attrs))

    def store_knowledge_graph(self, kg: KnowledgeGraph) -> None:
        for entity in kg.entities.values():
            self._store_entity(None, entity)
        for relation in kg.relations.values():
            self._store_relation(None, relation)
        self.persist()

    def upsert_entities(
        self,
        entities: Iterable[Entity],
        *,
        default_type: str = "Concept",
        verbose: bool = False,
    ) -> int:
        count = 0
        for entity in entities or []:
            if not isinstance(getattr(entity, "id", None), str) or not entity.id.strip():
                continue
            self._store_entity(None, entity, default_type=default_type)
            count += 1
        if count:
            self.persist()
        if verbose:
            logger.info("[LocalGraph] upsert_entities=%d", count)
        return count

    def upsert_relations(
        self,
        relations: Iterable[Relation],
        *,
        default_confidence: float = 1.0,
        verbose: bool = False,
    ) -> int:
        count = 0
        for relation in relations or []:
            if not isinstance(getattr(relation, "id", None), str) or not relation.id.strip():
                continue
            if getattr(relation, "confidence", None) is None:
                relation.confidence = float(default_confidence)
            self._store_relation(None, relation)
            count += 1
        if count:
            self.persist()
        if verbose:
            logger.info("[LocalGraph] upsert_relations=%d", count)
        return count

    def search_entities(self, query: str, limit: int = 10) -> List[Entity]:
        raw = str(query or "").strip().lower()
        out: List[Entity] = []
        for _, data in self.graph.nodes(data=True):
            name = str(data.get("name", "") or "").lower()
            aliases = [str(x or "").lower() for x in _as_list_str(data.get("aliases", []))]
            if raw and raw not in name and not any(raw in alias for alias in aliases):
                continue
            out.append(
                Entity(
                    id=str(data.get("id") or ""),
                    name=str(data.get("name") or ""),
                    type=data.get("type") or ["Entity"],
                    aliases=_as_list_str(data.get("aliases", [])),
                    properties=_safe_properties(data.get("properties", {})),
                    description=str(data.get("description", "") or ""),
                    scope=str(data.get("scope", "") or ""),
                    source_documents=_as_list_str(data.get("source_documents", [])),
                )
            )
            if len(out) >= max(1, int(limit)):
                break
        return out

    def search_relations(self, entity_name: str, limit: int = 10) -> List[Relation]:
        target = str(entity_name or "").strip()
        out: List[Relation] = []
        for src, dst, key, data in self.graph.edges(keys=True, data=True):
            s_name = str(self.graph.nodes[src].get("name", "") or "")
            o_name = str(self.graph.nodes[dst].get("name", "") or "")
            if target not in {s_name, o_name}:
                continue
            out.append(
                Relation(
                    id=str(data.get("id") or key),
                    subject_id=str(data.get("subject_id") or src),
                    object_id=str(data.get("object_id") or dst),
                    predicate=str(data.get("predicate") or ""),
                    relation_name=data.get("relation_name"),
                    persistence=data.get("persistence"),
                    description=data.get("description"),
                    source_documents=_as_list_str(data.get("source_documents", [])),
                    confidence=float(data.get("confidence", 1.0) or 1.0),
                    properties=_safe_properties(data.get("properties", {})),
                )
            )
            if len(out) >= max(1, int(limit)):
                break
        return out

    def get_stats(self) -> Dict[str, Any]:
        return {
            "status": "connected",
            "entities": self.graph.number_of_nodes(),
            "relations": self.graph.number_of_edges(),
            "path": self.path,
        }

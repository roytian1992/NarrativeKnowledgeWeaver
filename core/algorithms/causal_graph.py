# -*- coding: utf-8 -*-
# algorithms/episode_graph.py
from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union

import networkx as nx


def _norm_type(x: Any) -> str:
    return str(x or "").strip().lower()


def _safe_float(x: Any, default: float = 1.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _as_set(xs: Any) -> Set[str]:
    if xs is None:
        return set()
    if isinstance(xs, set):
        return set(_norm_type(x) for x in xs if _norm_type(x))
    if isinstance(xs, (list, tuple)):
        return set(_norm_type(x) for x in xs if _norm_type(x))
    if isinstance(xs, str):
        t = _norm_type(xs)
        return {t} if t else set()
    return set()


def _parse_type_weight(obj: Any) -> Dict[str, float]:
    """
    Accept:
      - dict like {"causes": 1.0, ...}
      - JSON string (possibly multi-line) representing a dict
    Return:
      - dict[str, float] with normalized keys and float values
    """
    if obj is None:
        return {}

    raw: Any = obj

    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return {}
        try:
            raw = json.loads(s)
        except Exception as e:
            raise ValueError(f"type_weight is a string but not valid JSON: {e}") from e

    if not isinstance(raw, dict):
        raise ValueError(f"type_weight must be dict or JSON-string of dict, got: {type(raw)}")

    out: Dict[str, float] = {}
    for k, v in raw.items():
        kk = _norm_type(k)
        if not kk:
            continue
        out[kk] = _safe_float(v, 0.0)
    return out


def load_causal_graph_cfg(config: Any) -> Dict[str, Any]:
    """
    Read causal_graph config from:
      config.narrative_graph_builder.causal_graph

    This keeps episode_graph.py independent of config.py hard-coding.
    It only assumes KAGConfig exposes attributes similarly to your other builders.

    Expected YAML:
      narrative_graph_builder:
        causal_graph:
          tau_conf: 0.0
          tau_eff: 0.0
          unified_pred: "EPISODE_CAUSAL_LINK"
          skip_types: [...]
          flipped_types: [...]
          type_weight: |  {...JSON...}
    """
    # defensive access
    ngb = getattr(config, "narrative_graph_builder", None)
    cg = getattr(ngb, "causal_graph", None) if ngb is not None else None

    # allow dict style too (in case you sometimes load config as dict)
    if cg is None and isinstance(config, dict):
        cg = (config.get("narrative_graph_builder") or {}).get("causal_graph")
    if cg is None:
        cg = {}

    def _get(key: str, default: Any) -> Any:
        if isinstance(cg, dict):
            return cg.get(key, default)
        return getattr(cg, key, default)

    tau_conf = _safe_float(_get("tau_conf", 0.0), 0.0)
    tau_eff = _safe_float(_get("tau_eff", 0.0), 0.0)
    unified_pred = str(_get("unified_pred", "EPISODE_CAUSAL_LINK") or "EPISODE_CAUSAL_LINK").strip()

    skip_types = _as_set(_get("skip_types", []))
    flipped_types = _as_set(_get("flipped_types", []))

    type_weight = _parse_type_weight(_get("type_weight", {}))

    return {
        "tau_conf": tau_conf,
        "tau_eff": tau_eff,
        "unified_pred": unified_pred,
        "skip_types": skip_types,
        "flipped_types": flipped_types,
        "type_weight": type_weight,
    }


def build_episode_graph(
    episodes: List[Dict[str, Any]],
    relations: List[Dict[str, Any]],
    *,
    cfg: Optional[Dict[str, Any]] = None,
    require_nodes_exist: bool = True,
) -> nx.DiGraph:
    """
    Build a DiGraph from episodes + relations.

    Two regimes (same as你原来逻辑，但 cfg 来自 YAML):
      - Regime B (post-cycle-break): r.predicate == unified_pred
          keep direction (subject_id -> object_id), use r.effective_weight if exists
      - Regime A (raw): use r.relation_type (or r.predicate) weighted by type_weight
          effective_weight = confidence * type_weight
          flip direction if relation_type in flipped_types (e.g., elaborates)

    Multiple edges collapsing:
      - keep the best (u,v) by effective_weight
      - push others into evidence_pool (best edge holds pool)
    """
    cfg = cfg or {}
    tau_conf = _safe_float(cfg.get("tau_conf", 0.0), 0.0)
    tau_eff = _safe_float(cfg.get("tau_eff", 0.0), 0.0)
    unified_pred = str(cfg.get("unified_pred", "EPISODE_CAUSAL_LINK") or "EPISODE_CAUSAL_LINK").strip()

    skip_types: Set[str] = set(cfg.get("skip_types") or set())
    flipped_types: Set[str] = set(cfg.get("flipped_types") or set())
    type_weight: Dict[str, float] = dict(cfg.get("type_weight") or {})

    G = nx.DiGraph()

    # nodes
    for e in episodes or []:
        if not isinstance(e, dict):
            continue
        eid = e.get("id")
        if not eid:
            continue
        G.add_node(
            eid,
            name=e.get("name"),
            description=e.get("description"),
            properties=e.get("properties", {}) if isinstance(e.get("properties"), dict) else {},
            source_documents=e.get("source_documents", []),
        )

    best: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for r in relations or []:
        if not isinstance(r, dict):
            continue

        s = r.get("subject_id") or r.get("subject")
        o = r.get("object_id") or r.get("object")
        if not s or not o or s == o:
            continue

        if require_nodes_exist and (s not in G.nodes or o not in G.nodes):
            continue

        pred = str(r.get("predicate", "") or "").strip()

        # ------------------------------------------------------------
        # Regime B: unified DAG edge
        # ------------------------------------------------------------
        if pred == unified_pred:
            u, v = str(s), str(o)

            conf = _safe_float(r.get("confidence", 1.0), 1.0)
            if conf < tau_conf:
                continue

            ew = _safe_float(r.get("effective_weight", 0.0), 0.0)
            if ew < tau_eff:
                continue

            attr = {
                "predicate": unified_pred,
                "relation_type": _norm_type(r.get("relation_type") or unified_pred),
                "confidence": conf,
                "type_weight": _safe_float(r.get("type_weight", 1.0), 1.0),
                "effective_weight": ew,
                "description": r.get("description", ""),
                "properties": r.get("properties", {}) if isinstance(r.get("properties"), dict) else {},
                "evidence_pool": r.get("evidence_pool", []) if isinstance(r.get("evidence_pool"), list) else [],
                "source_relation_ids": r.get("source_relation_ids", [])
                if isinstance(r.get("source_relation_ids"), list)
                else ([r.get("id")] if r.get("id") else []),
            }

            key = (u, v)
            if key not in best:
                best[key] = attr
            else:
                if float(attr["effective_weight"]) > float(best[key]["effective_weight"]):
                    best[key].setdefault("evidence_pool", []).append(dict(best[key]))
                    best[key] = attr
                else:
                    best[key].setdefault("evidence_pool", []).append(dict(attr))
            continue

        # ------------------------------------------------------------
        # Regime A: raw relation
        # ------------------------------------------------------------
        rel_type = _norm_type(r.get("relation_type") or r.get("predicate"))
        if not rel_type:
            continue
        if skip_types and rel_type in skip_types:
            continue
        if type_weight and rel_type not in type_weight:
            # IMPORTANT: no hard-code here. If YAML doesn't include the type, we skip.
            continue

        conf = _safe_float(r.get("confidence", 1.0), 1.0)
        if conf < tau_conf:
            continue

        u, v = (str(o), str(s)) if (flipped_types and rel_type in flipped_types) else (str(s), str(o))

        tw = float(type_weight.get(rel_type, 1.0))
        eff = conf * tw
        if eff < tau_eff:
            continue

        attr = {
            "predicate": unified_pred,
            "relation_type": rel_type,
            "confidence": conf,
            "type_weight": tw,
            "effective_weight": eff,
            "description": r.get("description", ""),
            "properties": r.get("properties", {}) if isinstance(r.get("properties"), dict) else {},
            "evidence_pool": [],
            "source_relation_ids": [r.get("id")] if r.get("id") else [],
        }

        key = (u, v)
        if key not in best:
            best[key] = attr
        else:
            if float(attr["effective_weight"]) > float(best[key]["effective_weight"]):
                best[key].setdefault("evidence_pool", []).append(dict(best[key]))
                best[key] = attr
            else:
                best[key].setdefault("evidence_pool", []).append(dict(attr))

    for (u, v), attr in best.items():
        if u in G.nodes and v in G.nodes and u != v:
            G.add_edge(u, v, **attr)

    return G


def export_relations_from_graph(
    G: nx.DiGraph,
    *,
    predicate: str = "EPISODE_CAUSAL_LINK",
    keep_fields: Optional[Iterable[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Export graph edges into JSON-ready relations list.
    """
    if keep_fields is None:
        keep_fields = [
            "relation_type",
            "confidence",
            "type_weight",
            "effective_weight",
            "description",
            "properties",
            "evidence_pool",
            "source_relation_ids",
        ]

    out: List[Dict[str, Any]] = []
    for u, v, d in G.edges(data=True):
        item = {
            "subject_id": u,
            "object_id": v,
            "predicate": predicate,
            "relation_name": predicate,
            "confidence": _safe_float(d.get("confidence", 1.0), 1.0),
            "description": d.get("description", ""),
            "properties": d.get("properties", {}) if isinstance(d.get("properties"), dict) else {},
        }
        for k in keep_fields:
            if k in ("confidence", "description", "properties"):
                continue
            if k in d:
                item[k] = d.get(k)
        out.append(item)
    return out

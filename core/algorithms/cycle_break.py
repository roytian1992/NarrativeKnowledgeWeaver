# -*- coding: utf-8 -*-
# algorithms/cycle_break.py
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
from tqdm import tqdm

# single source of truth
from .causal_graph import (
    build_episode_graph as build_causal_graph,
    export_relations_from_graph,
    load_causal_graph_cfg,
)


# =============================================================================
# Minimal helpers (kept minimal)
# =============================================================================
def _norm_type(x: Any) -> str:
    return str(x or "").strip().lower()


def _safe_float(x: Any, default: float = 1.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


# =============================================================================
# Config loader (from KAGConfig)
# =============================================================================
def load_cycle_break_cfg(config: Any) -> Dict[str, Any]:
    """
    Read cycle_break config from:
      config.narrative_graph_builder.cycle_break

    Expected YAML:
      narrative_graph_builder:
        cycle_break:
          delta_tie: 0.02
          max_iters: 10
    """
    ngb = getattr(config, "narrative_graph_builder", None)
    cb = getattr(ngb, "cycle_break", None) if ngb is not None else None

    if cb is None and isinstance(config, dict):
        cb = (config.get("narrative_graph_builder") or {}).get("cycle_break")
    if cb is None:
        cb = {}

    def _get(key: str, default: Any) -> Any:
        if isinstance(cb, dict):
            return cb.get(key, default)
        return getattr(cb, key, default)

    def _safe_float_local(x: Any, default: float) -> float:
        try:
            return float(x)
        except Exception:
            return float(default)

    def _safe_int_local(x: Any, default: int) -> int:
        try:
            return int(x)
        except Exception:
            return int(default)

    delta_tie = _safe_float_local(_get("delta_tie", 0.02), 0.02)
    max_iters = _safe_int_local(_get("max_iters", _get("max_iter", 10)), 10)  # tolerate legacy key

    return {
        "delta_tie": float(delta_tie),
        "max_iters": int(max_iters),
    }


def load_full_cycle_break_cfg(config: Any) -> Dict[str, Any]:
    """
    Convenience: merge causal_graph cfg + cycle_break cfg.
    """
    out = load_cycle_break_cfg(config)
    out["graph_cfg"] = load_causal_graph_cfg(config)
    return out


# =============================================================================
# Stage I: SCC cycle breaking (effective-weight based)
# =============================================================================
def _edge_cost_for_cycle_break(edge_attr: Dict[str, Any]) -> float:
    # smaller cost => remove earlier
    ew = _safe_float(edge_attr.get("effective_weight", 0.0), 0.0)
    return 1.0 - ew


def stage1_break_scc_cycles(
    G: nx.DiGraph,
    *,
    delta_tie: float,
    enable_llm: bool = False,
    llm_choose_edge_fn=None,
    iter_id: int = 1,
    log: Optional[List[Dict[str, Any]]] = None,
    show_progress: bool = False,
) -> Tuple[bool, List[Dict[str, Any]]]:
    """
    Break cycles inside SCCs by removing the lowest-cost edges, cost = 1 - effective_weight.

    Optional LLM tie-breaking:
      - only when enable_llm=True and llm_choose_edge_fn is provided
      - only when multiple best edges tie within delta_tie
    """
    if log is None:
        log = []

    removed_any = False

    def _scc_list() -> List[Set[str]]:
        return [set(c) for c in nx.strongly_connected_components(G) if len(c) > 1]

    outer = range(10**9)
    if show_progress:
        outer = tqdm(outer, desc=f"Stage1 SCC break iter={iter_id}", unit="round")

    for _ in outer:
        sccs = _scc_list()
        if not sccs:
            break

        for comp in sccs:
            while True:
                S = G.subgraph(comp).copy()
                if nx.is_directed_acyclic_graph(S):
                    break

                edges: List[Tuple[str, str, Dict[str, Any], float]] = []
                for u, v, d in S.edges(data=True):
                    cost = float(_edge_cost_for_cycle_break(d))
                    edges.append((u, v, dict(d), cost))
                if not edges:
                    break

                edges.sort(key=lambda x: x[3])
                best_cost = edges[0][3]
                tie = [e for e in edges if abs(e[3] - best_cost) < float(delta_tie)]

                if enable_llm and llm_choose_edge_fn is not None and len(tie) > 1:
                    pick = llm_choose_edge_fn(G, list(comp), tie)
                    if pick is None:
                        u, v, d, cost = edges[0]
                    else:
                        u, v = pick
                        d = dict(G[u][v]) if G.has_edge(u, v) else {}
                        cost = float(_edge_cost_for_cycle_break(d))
                else:
                    u, v, d, cost = edges[0]

                if G.has_edge(u, v):
                    G.remove_edge(u, v)
                    removed_any = True
                    log.append(
                        {
                            "iter": iter_id,
                            "stage": "SCC_BREAK",
                            "action": "remove_edge",
                            "edge": {"from": u, "to": v},
                            "relation_type": d.get("relation_type"),
                            "effective_weight": d.get("effective_weight"),
                            "cost": cost,
                            "reason": "break_cycle_in_scc",
                        }
                    )

    if show_progress and hasattr(outer, "close"):
        outer.close()

    return removed_any, log


# =============================================================================
# Stage II: Triangle pruning (LLM-first)
# =============================================================================
def stage2_triangle_prune_llm_first(
    G: nx.DiGraph,
    *,
    narrative_manager,
    enable_llm: bool = True,
    max_llm_calls: Optional[int] = None,
    iter_id: int = 1,
    log: Optional[List[Dict[str, Any]]] = None,
    show_progress: bool = True,
    max_workers: int = 64,
    llm_decided_keep: Optional[Set[Tuple[str, str]]] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Set[Tuple[str, str]]]:
    """
    For each triangle A->B, B->C, A->C:
      - If all three edges are causes: ALWAYS call LLM to decide remove A->C or keep.
      - Else strong-cause protection:
          If A->C is causes and at least one of A->B or B->C is NOT causes:
              auto keep A->C (no LLM).
      - Otherwise:
          call LLM to decide remove A->C or keep.

    llm_decided_keep: cross-iteration memo set of (a, c) pairs already decided "keep" by LLM.
                      Triangles whose AC edge is already in this set are skipped.
    Returns (stats, log, llm_decided_keep).
    """
    if log is None:
        log = []
    if llm_decided_keep is None:
        llm_decided_keep = set()
    if not enable_llm or narrative_manager is None:
        raise ValueError("Stage2 requires narrative_manager and enable_llm=True")

    stats = {
        "triangles_total": 0,
        "llm_candidates": 0,
        "llm_skipped_memo": 0,
        "llm_called": 0,
        "llm_remove": 0,
        "llm_keep": 0,
        "auto_keep_strong_cause": 0,
        "marked_remove": 0,
        "removed": 0,
    }

    def _node_info(nid: str) -> Dict[str, Any]:
        d = dict(G.nodes[nid]) if nid in G.nodes else {}
        return {
            "id": nid,
            "name": d.get("name"),
            "description": d.get("description"),
            "properties": d.get("properties", {}),
        }

    def _edge_info(u: str, v: str) -> Dict[str, Any]:
        d = dict(G[u][v]) if G.has_edge(u, v) else {}
        return {
            "from": u,
            "to": v,
            "predicate": d.get("predicate"),
            "relation_type": d.get("relation_type"),
            "confidence": d.get("confidence"),
            "type_weight": d.get("type_weight"),
            "effective_weight": d.get("effective_weight"),
            "description": d.get("description"),
            "properties": d.get("properties", {}),
            "evidence_pool": d.get("evidence_pool", []),
        }

    def _rtype(u: str, v: str) -> str:
        if not G.has_edge(u, v):
            return ""
        return _norm_type(G[u][v].get("relation_type"))

    llm_jobs: List[Tuple[str, str, str, str, str, str]] = []
    nodes = list(G.nodes)

    it = nodes
    if show_progress:
        it = tqdm(nodes, desc=f"Stage2 collect triangles iter={iter_id}", unit="node")

    for a in it:
        for b in list(G.successors(a)):
            for c in list(G.successors(b)):
                if c == a or c == b:
                    continue
                if not G.has_edge(a, c):
                    continue

                stats["triangles_total"] += 1

                t_ab = _rtype(a, b)
                t_bc = _rtype(b, c)
                t_ac = _rtype(a, c)
                all_causes = (t_ab == "causes" and t_bc == "causes" and t_ac == "causes")

                if (t_ac == "causes") and (not all_causes) and (t_ab != "causes" or t_bc != "causes"):
                    stats["auto_keep_strong_cause"] += 1
                    log.append(
                        {
                            "iter": iter_id,
                            "stage": "TRIANGLE",
                            "action": "auto_keep",
                            "rule": "STRONG_CAUSE_PROTECT",
                            "edge": {"from": a, "to": c},
                            "types": {"ab": t_ab, "bc": t_bc, "ac": t_ac},
                            "reason": "A->C is causes but chain has non-causes edge(s)",
                        }
                    )
                    continue

                # skip if this AC edge was already decided "keep" in a prior iteration
                if (a, c) in llm_decided_keep:
                    stats["llm_skipped_memo"] += 1
                    continue

                llm_jobs.append((a, b, c, t_ab, t_bc, t_ac))

        if show_progress and hasattr(it, "set_postfix"):
            it.set_postfix(
                tri=stats["triangles_total"],
                llm_cand=len(llm_jobs),
                auto_keep=stats["auto_keep_strong_cause"],
                memo_skip=stats["llm_skipped_memo"],
            )

    if show_progress and hasattr(it, "close"):
        it.close()

    stats["llm_candidates"] = len(llm_jobs)
    if max_llm_calls is not None:
        llm_jobs = llm_jobs[: int(max_llm_calls)]

    if not llm_jobs:
        return stats, log, llm_decided_keep

    to_remove: Set[Tuple[str, str]] = set()

    def _call_one(job):
        a, b, c, t_ab, t_bc, t_ac = job
        out = narrative_manager.prune_causal_edge(
            entity_a=_node_info(a),
            entity_b=_node_info(b),
            entity_c=_node_info(c),
            relation_ab=_edge_info(a, b),
            relation_bc=_edge_info(b, c),
            relation_ac=_edge_info(a, c),
        )
        parsed = json.loads(out) if isinstance(out, str) else (out or {})
        remove_edge = bool(parsed.get("remove_edge", False))
        reason = parsed.get("reason", "")
        return (a, b, c, t_ab, t_bc, t_ac, remove_edge, reason)

    futures = []
    with ThreadPoolExecutor(max_workers=int(max_workers)) as ex:
        for job in llm_jobs:
            futures.append(ex.submit(_call_one, job))

        it2 = as_completed(futures)
        if show_progress:
            it2 = tqdm(it2, total=len(futures), desc=f"Stage2 LLM prune iter={iter_id}", unit="tri")

        for fut in it2:
            try:
                a, b, c, t_ab, t_bc, t_ac, remove_edge, reason = fut.result()
            except Exception as e:
                log.append({"iter": iter_id, "stage": "TRIANGLE", "action": "llm_error", "error": str(e)})
                continue

            stats["llm_called"] += 1

            if remove_edge:
                stats["llm_remove"] += 1
                stats["marked_remove"] += 1
                to_remove.add((a, c))
                log.append(
                    {
                        "iter": iter_id,
                        "stage": "TRIANGLE",
                        "action": "llm_mark_remove",
                        "edge": {"from": a, "to": c},
                        "types": {"ab": t_ab, "bc": t_bc, "ac": t_ac},
                        "reason": reason,
                    }
                )
            else:
                stats["llm_keep"] += 1
                llm_decided_keep.add((a, c))
                log.append(
                    {
                        "iter": iter_id,
                        "stage": "TRIANGLE",
                        "action": "llm_keep",
                        "edge": {"from": a, "to": c},
                        "types": {"ab": t_ab, "bc": t_bc, "ac": t_ac},
                        "reason": reason,
                    }
                )

        if show_progress and hasattr(it2, "close"):
            it2.close()

    for u, v in to_remove:
        if G.has_edge(u, v):
            G.remove_edge(u, v)
            stats["removed"] += 1
            log.append(
                {
                    "iter": iter_id,
                    "stage": "TRIANGLE",
                    "action": "remove_edge",
                    "edge": {"from": u, "to": v},
                    "reason": "triangle_prune_llm_first",
                }
            )

    return stats, log, llm_decided_keep


# =============================================================================
# Runners (public API)
# =============================================================================
def run_heuristic(
    episodes: List[Dict[str, Any]],
    relations: List[Dict[str, Any]],
    *,
    config: Any,
    show_progress: bool = False,
    graph_cfg: Optional[Dict[str, Any]] = None,
    cycle_cfg: Optional[Dict[str, Any]] = None,
) -> Tuple[nx.DiGraph, List[Dict[str, Any]]]:
    """
    Pure effective-weight based:
      - build graph (shared loader)
      - repeatedly break SCC cycles until convergence or max_iter
    """
    if graph_cfg is None:
        graph_cfg = load_causal_graph_cfg(config)
    if cycle_cfg is None:
        cycle_cfg = load_cycle_break_cfg(config)

    G = build_causal_graph(episodes, relations, cfg=graph_cfg)
    log: List[Dict[str, Any]] = []

    max_iters = int(cycle_cfg["max_iters"])
    delta_tie = float(cycle_cfg["delta_tie"])

    for it in range(1, max_iters + 1):
        removed, log = stage1_break_scc_cycles(
            G,
            delta_tie=delta_tie,
            enable_llm=False,
            llm_choose_edge_fn=None,
            iter_id=it,
            log=log,
            show_progress=show_progress,
        )
        has_scc = any(len(c) > 1 for c in nx.strongly_connected_components(G))
        if (not has_scc) or (not removed):
            log.append({"iter": it, "stage": "CONVERGE", "action": "break", "reason": "no_scc_or_no_removal"})
            break

    return G, log


def run_saber(
    episodes: List[Dict[str, Any]],
    relations: List[Dict[str, Any]],
    *,
    config: Any,
    narrative_manager,
    max_llm_calls_per_iter: int = 200,
    show_progress: bool = True,
    triangle_workers: int = 64,
    graph_cfg: Optional[Dict[str, Any]] = None,
    cycle_cfg: Optional[Dict[str, Any]] = None,
) -> Tuple[nx.DiGraph, List[Dict[str, Any]]]:
    """
    SABER:
      For each iter:
        Stage II: triangle prune (LLM-first)
        Stage I: SCC break cycles (effective-weight)
      Converge when:
        - no SCC > 1
        - and Stage II removed 0
        - and Stage I removed 0
    """
    if graph_cfg is None:
        graph_cfg = load_causal_graph_cfg(config)
    if cycle_cfg is None:
        cycle_cfg = load_cycle_break_cfg(config)

    G = build_causal_graph(episodes, relations, cfg=graph_cfg)
    log: List[Dict[str, Any]] = []

    max_iters = int(cycle_cfg["max_iters"])
    delta_tie = float(cycle_cfg["delta_tie"])

    llm_decided_keep: Set[Tuple[str, str]] = set()

    for it in range(1, max_iters + 1):
        stats_tri, log, llm_decided_keep = stage2_triangle_prune_llm_first(
            G,
            narrative_manager=narrative_manager,
            enable_llm=True,
            max_llm_calls=int(max_llm_calls_per_iter),
            iter_id=it,
            log=log,
            show_progress=show_progress,
            max_workers=int(triangle_workers),
            llm_decided_keep=llm_decided_keep,
        )

        removed_scc, log = stage1_break_scc_cycles(
            G,
            delta_tie=delta_tie,
            enable_llm=False,
            llm_choose_edge_fn=None,
            iter_id=it,
            log=log,
            show_progress=False,
        )

        has_scc = any(len(c) > 1 for c in nx.strongly_connected_components(G))
        removed_tri = bool(stats_tri.get("removed", 0) > 0)

        if (not has_scc) and (not removed_scc) and (not removed_tri):
            log.append({"iter": it, "stage": "CONVERGE", "action": "break", "reason": "no_scc_and_no_removals"})
            break

    return G, log

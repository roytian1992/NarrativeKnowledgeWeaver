# -*- coding: utf-8 -*-
# algorithms/chain_extraction.py
from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple

import networkx as nx

from .causal_graph import build_episode_graph, load_causal_graph_cfg


# =============================================================================
# Config loaders (from KAGConfig)
# =============================================================================
def load_chain_extraction_cfg(config: Any) -> Dict[str, Any]:
    """
    Read chain_extraction config from:
      config.narrative_graph_builder.chain_extraction

    Expected YAML:
      narrative_graph_builder:
        chain_extraction:
          min_effective_weight: 0.0
          min_similarity: 0.0
          min_common_neighbors: 0.0
          max_paths_per_source: 200
          max_total_paths: 5000
          max_depth: null
          trie_min_len: 3
          trie_include_cutpoint: true
          trie_drop_contained: true
          trie_keep_terminal_pairs: false
    """
    ngb = getattr(config, "narrative_graph_builder", None)
    ce = getattr(ngb, "chain_extraction", None) if ngb is not None else None

    if ce is None and isinstance(config, dict):
        ce = (config.get("narrative_graph_builder") or {}).get("chain_extraction")
    if ce is None:
        ce = {}

    def _get(key: str, default: Any) -> Any:
        if isinstance(ce, dict):
            return ce.get(key, default)
        return getattr(ce, key, default)

    def _safe_float(x: Any, default: float = 0.0) -> float:
        try:
            return float(x)
        except Exception:
            return float(default)

    def _safe_int(x: Any, default: int) -> int:
        try:
            return int(x)
        except Exception:
            return int(default)

    def _safe_bool(x: Any, default: bool) -> bool:
        if isinstance(x, bool):
            return x
        if isinstance(x, (int, float)):
            return bool(x)
        if isinstance(x, str):
            s = x.strip().lower()
            if s in ("true", "1", "yes", "y", "t"):
                return True
            if s in ("false", "0", "no", "n", "f"):
                return False
        return bool(default)

    # thresholds
    min_effective_weight = _safe_float(_get("min_effective_weight", 0.0), 0.0)

    ms = _get("min_similarity", None)
    min_similarity = None if ms is None else _safe_float(ms, 0.0)

    mcn = _get("min_common_neighbors", None)
    min_common_neighbors = None if mcn is None else _safe_int(mcn, 0)

    # bounded path extraction
    max_paths_per_source = _safe_int(_get("max_paths_per_source", 200), 200)
    max_total_paths = _safe_int(_get("max_total_paths", 5000), 5000)

    md = _get("max_depth", None)
    max_depth = None if md in (None, "", "null") else _safe_int(md, 0)

    # trie trunk extraction
    trie_min_len = _safe_int(_get("trie_min_len", 3), 3)
    trie_include_cutpoint = _safe_bool(_get("trie_include_cutpoint", True), True)
    trie_drop_contained = _safe_bool(_get("trie_drop_contained", True), True)
    trie_keep_terminal_pairs = _safe_bool(_get("trie_keep_terminal_pairs", False), False)

    return {
        "min_effective_weight": float(min_effective_weight),
        "min_similarity": min_similarity,
        "min_common_neighbors": min_common_neighbors,
        "max_paths_per_source": int(max_paths_per_source),
        "max_total_paths": int(max_total_paths),
        "max_depth": max_depth,
        "trie_min_len": int(trie_min_len),
        "trie_include_cutpoint": bool(trie_include_cutpoint),
        "trie_drop_contained": bool(trie_drop_contained),
        "trie_keep_terminal_pairs": bool(trie_keep_terminal_pairs),
    }


def load_full_chain_cfg(config: Any) -> Dict[str, Any]:
    """
    Convenience: merge causal_graph cfg + chain_extraction cfg.
    """
    out = load_chain_extraction_cfg(config)
    out["graph_cfg"] = load_causal_graph_cfg(config)
    return out


# =============================================================================
# Helpers
# =============================================================================
def build_episode_index(episodes: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    idx: Dict[str, Dict[str, Any]] = {}
    for ep in episodes or []:
        if not isinstance(ep, dict):
            continue
        eid = str(ep.get("id", "")).strip()
        if eid:
            idx[eid] = ep
    return idx


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def get_sources(G: nx.DiGraph) -> List[str]:
    return [n for n in G.nodes if G.in_degree(n) == 0]


def get_sinks(G: nx.DiGraph) -> List[str]:
    return [n for n in G.nodes if G.out_degree(n) == 0]


# =============================================================================
# A) Bounded path extraction in DAG
# =============================================================================
def make_edge_keep_fn_from_graph(
    G: nx.DiGraph,
    *,
    min_effective_weight: float = 0.0,
    min_similarity: Optional[float] = None,
    min_common_neighbors: Optional[int] = None,
) -> Callable[[str, str], bool]:
    """
    Edge filter using:
      - edge.effective_weight
      - edge.properties.similarity
      - edge.properties.common_neighbors
    """
    def _f(u: str, v: str) -> bool:
        if not G.has_edge(u, v):
            return False

        # DiGraph in episode_graph stores single edge attrs (not MultiDiGraph)
        d = dict(G[u][v])

        ew = _safe_float(d.get("effective_weight", 0.0), 0.0)
        if ew < float(min_effective_weight):
            return False

        props = d.get("properties", {}) if isinstance(d.get("properties"), dict) else {}

        if min_similarity is not None:
            sim = _safe_float(props.get("similarity", 0.0), 0.0)
            if sim < float(min_similarity):
                return False

        if min_common_neighbors is not None:
            try:
                cn = int(props.get("common_neighbors", 0) or 0)
            except Exception:
                cn = 0
            if cn < int(min_common_neighbors):
                return False

        return True

    return _f


def bounded_dfs_paths(
    G: nx.DiGraph,
    start: str,
    *,
    sinks: Optional[Set[str]] = None,
    max_depth: Optional[int] = None,
    max_paths: int = 200,
    edge_keep_fn: Optional[Callable[[str, str], bool]] = None,
) -> List[List[str]]:
    """
    Enumerate paths from start to sinks with bounds to avoid blow-up.
    """
    if start not in G.nodes:
        return []

    if sinks is None:
        sinks = set(get_sinks(G))

    out: List[List[str]] = []
    stack: List[Tuple[str, List[str]]] = [(start, [start])]

    while stack and len(out) < int(max_paths):
        cur, path = stack.pop()

        if cur in sinks:
            out.append(path)
            continue

        if max_depth is not None and len(path) >= int(max_depth):
            continue

        succs = list(G.successors(cur))
        succs.sort(
            key=lambda v: float(G[cur][v].get("effective_weight", 0.0)) if G.has_edge(cur, v) else 0.0,
            reverse=True,
        )

        for nxt in succs:
            if nxt in path:
                continue
            if edge_keep_fn is not None and not bool(edge_keep_fn(cur, nxt)):
                continue
            stack.append((nxt, path + [nxt]))

    return out


def extract_paths_from_sources_bounded(
    G: nx.DiGraph,
    *,
    max_paths_per_source: int = 200,
    max_total_paths: int = 5000,
    max_depth: Optional[int] = None,
    edge_keep_fn: Optional[Callable[[str, str], bool]] = None,
) -> List[List[str]]:
    sources = get_sources(G)
    sinks = set(get_sinks(G))

    all_paths: List[List[str]] = []
    for s in sources:
        if len(all_paths) >= int(max_total_paths):
            break
        paths = bounded_dfs_paths(
            G,
            s,
            sinks=sinks,
            max_depth=max_depth,
            max_paths=max_paths_per_source,
            edge_keep_fn=edge_keep_fn,
        )
        all_paths.extend(paths)
        if len(all_paths) >= int(max_total_paths):
            all_paths = all_paths[: int(max_total_paths)]
            break

    return all_paths


# =============================================================================
# B) Trie segmentation
# =============================================================================
class _TrieNode:
    __slots__ = ("children", "count", "end_count", "token")

    def __init__(self, token: Optional[str] = None):
        self.children: Dict[str, _TrieNode] = {}
        self.count: int = 0
        self.end_count: int = 0
        self.token: Optional[str] = token


def _trie_insert(root: _TrieNode, chain: List[str]) -> None:
    node = root
    node.count += 1
    for t in chain:
        if t not in node.children:
            node.children[t] = _TrieNode(t)
        node = node.children[t]
        node.count += 1
    node.end_count += 1


def _is_substring(sub: List[str], full: List[str]) -> bool:
    n, m = len(sub), len(full)
    if n == 0 or n > m:
        return False
    for i in range(m - n + 1):
        if full[i : i + n] == sub:
            return True
    return False


def _collect_segments_from_trie(
    root: _TrieNode,
    *,
    min_len: int = 2,
    include_cutpoint: bool = True,
    keep_terminal_pairs: bool = False,
) -> List[List[str]]:
    segments: List[List[str]] = []
    stack: List[Tuple[_TrieNode, List[str], int]] = [(root, [], 0)]

    while stack:
        node, path, last_cut = stack.pop()

        if node.token is not None:
            path = path + [node.token]

        is_branch = len(node.children) >= 2
        is_chain_end = node.end_count > 0

        if is_branch or is_chain_end:
            seg_len = len(path) - last_cut

            if seg_len >= int(min_len):
                segments.append(path[last_cut:])
            elif keep_terminal_pairs and is_chain_end and len(path) >= 2:
                segments.append(path[-2:])

            for child in node.children.values():
                new_cut = len(path) - (1 if include_cutpoint else 0)
                if new_cut < 0:
                    new_cut = 0
                stack.append((child, path, new_cut))
        else:
            for child in node.children.values():
                stack.append((child, path, last_cut))

    seen: Set[Tuple[str, ...]] = set()
    uniq: List[List[str]] = []
    for seg in segments:
        k = tuple(seg)
        if k not in seen:
            seen.add(k)
            uniq.append(seg)
    return uniq


def extract_trunks_by_path_trie(
    paths: List[List[str]],
    *,
    min_len: int = 3,
    include_cutpoint: bool = True,
    drop_contained: bool = True,
    keep_terminal_pairs: bool = False,
) -> List[List[str]]:
    root = _TrieNode()
    for ch in paths:
        if ch:
            _trie_insert(root, ch)

    segs = _collect_segments_from_trie(
        root,
        min_len=min_len,
        include_cutpoint=include_cutpoint,
        keep_terminal_pairs=keep_terminal_pairs,
    )

    if not drop_contained:
        return segs

    segs_sorted = sorted(segs, key=lambda s: (-len(s), s))
    kept: List[List[str]] = []
    for s in segs_sorted:
        if not any(_is_substring(s, t) for t in kept):
            kept.append(s)
    return kept


# =============================================================================
# C) Minimum path cover trunks
# =============================================================================
def dag_minimum_path_cover(
    nodes: Iterable[str],
    edges: Iterable[Tuple[str, str]],
    *,
    edge_filter_fn: Optional[Callable[[str, str], bool]] = None,
) -> List[List[str]]:
    V: List[str] = list(nodes)
    V_set: Set[str] = set(V)

    used_edges: List[Tuple[str, str]] = []
    for u, v in edges:
        if u in V_set and v in V_set:
            if edge_filter_fn is None or bool(edge_filter_fn(u, v)):
                used_edges.append((u, v))

    B = nx.Graph()
    left_nodes = [("L", u) for u in V]
    right_nodes = [("R", u) for u in V]
    B.add_nodes_from(left_nodes, bipartite=0)
    B.add_nodes_from(right_nodes, bipartite=1)

    for u, v in used_edges:
        B.add_edge(("L", u), ("R", v))

    matching = nx.algorithms.bipartite.matching.hopcroft_karp_matching(
        B, top_nodes=set(left_nodes)
    )

    successor: Dict[str, str] = {}
    predecessor: Dict[str, str] = {}

    for lu in left_nodes:
        if lu in matching:
            rv = matching[lu]
            if isinstance(rv, tuple) and len(rv) == 2 and rv[0] == "R":
                u = lu[1]
                v = rv[1]
                successor[u] = v
                predecessor[v] = u

    starts = [u for u in V if u not in predecessor]

    paths: List[List[str]] = []
    visited: Set[str] = set()

    for s in starts:
        cur = s
        path = [cur]
        visited.add(cur)
        while cur in successor:
            nxt = successor[cur]
            if nxt in visited:
                break
            path.append(nxt)
            visited.add(nxt)
            cur = nxt
        paths.append(path)

    for u in V:
        if u not in visited:
            paths.append([u])
            visited.add(u)

    return paths


def extract_trunks_by_min_path_cover(
    G: nx.DiGraph,
    *,
    edge_filter_fn: Optional[Callable[[str, str], bool]] = None,
) -> List[List[str]]:
    nodes = list(G.nodes)
    edges = list(G.edges)
    trunks = dag_minimum_path_cover(nodes, edges, edge_filter_fn=edge_filter_fn)
    return sorted(trunks, key=lambda p: (-len(p), p))


# =============================================================================
# D) Output alignment
# =============================================================================
def align_trunks_output(
    trunks: List[List[str]],
    ep_idx: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for p in trunks:
        item: Dict[str, Any] = {"trunk": p}
        item["trunk_data"] = [
            {
                "id": eid,
                "name": ep_idx.get(eid, {}).get("name", ""),
                "description": ep_idx.get(eid, {}).get("description", ""),
                "source_documents": ep_idx.get(eid, {}).get("source_documents", []),
                "properties": ep_idx.get(eid, {}).get("properties", {}),
            }
            for eid in p
        ]
        out.append(item)
    return out


# =============================================================================
# Build graph + index
# =============================================================================
def build_graph_and_index(
    episodes: List[Dict[str, Any]],
    relations: List[Dict[str, Any]],
    *,
    graph_cfg: Optional[Dict[str, Any]] = None,
) -> Tuple[nx.DiGraph, Dict[str, Dict[str, Any]]]:
    ep_idx = build_episode_index(episodes)
    G = build_episode_graph(episodes, relations, cfg=graph_cfg)
    return G, ep_idx


# =============================================================================
# High-level APIs (what you import and use)
# =============================================================================
def run_trunk_extraction_trie(
    episodes: List[Dict[str, Any]],
    relations: List[Dict[str, Any]],
    *,
    config: Any,
    graph_cfg: Optional[Dict[str, Any]] = None,
    chain_cfg: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    1) build graph (shared loader)
    2) bounded path extraction
    3) trie-based trunk extraction
    4) align output
    """
    if graph_cfg is None:
        graph_cfg = load_causal_graph_cfg(config)
    if chain_cfg is None:
        chain_cfg = load_chain_extraction_cfg(config)

    G, ep_idx = build_graph_and_index(episodes, relations, graph_cfg=graph_cfg)

    edge_keep_fn = make_edge_keep_fn_from_graph(
        G,
        min_effective_weight=chain_cfg["min_effective_weight"],
        min_similarity=chain_cfg["min_similarity"],
        min_common_neighbors=chain_cfg["min_common_neighbors"],
    )

    paths = extract_paths_from_sources_bounded(
        G,
        max_paths_per_source=chain_cfg["max_paths_per_source"],
        max_total_paths=chain_cfg["max_total_paths"],
        max_depth=chain_cfg["max_depth"],
        edge_keep_fn=edge_keep_fn,
    )

    trunks = extract_trunks_by_path_trie(
        paths,
        min_len=chain_cfg["trie_min_len"],
        include_cutpoint=chain_cfg["trie_include_cutpoint"],
        drop_contained=chain_cfg["trie_drop_contained"],
        keep_terminal_pairs=chain_cfg["trie_keep_terminal_pairs"],
    )

    return align_trunks_output(trunks, ep_idx)


def run_trunk_extraction_mpc(
    episodes: List[Dict[str, Any]],
    relations: List[Dict[str, Any]],
    *,
    config: Any,
    graph_cfg: Optional[Dict[str, Any]] = None,
    chain_cfg: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    1) build graph (shared loader)
    2) minimum path cover trunks
    3) align output
    """
    if graph_cfg is None:
        graph_cfg = load_causal_graph_cfg(config)
    if chain_cfg is None:
        chain_cfg = load_chain_extraction_cfg(config)

    G, ep_idx = build_graph_and_index(episodes, relations, graph_cfg=graph_cfg)

    edge_keep_fn = make_edge_keep_fn_from_graph(
        G,
        min_effective_weight=chain_cfg["min_effective_weight"],
        min_similarity=chain_cfg["min_similarity"],
        min_common_neighbors=chain_cfg["min_common_neighbors"],
    )

    trunks = extract_trunks_by_min_path_cover(G, edge_filter_fn=edge_keep_fn)
    return align_trunks_output(trunks, ep_idx)


def extract_storyline_candidates(
    episodes: List[Dict[str, Any]],
    relations: List[Dict[str, Any]],
    *,
    config: Any,
    method: str = "trie",
    graph_cfg: Optional[Dict[str, Any]] = None,
    chain_cfg: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Unified entrypoint for candidate storyline trunk extraction.

    method:
      - "trie" (also accepts "tri")
      - "mpc"
    """
    m = (method or "").strip().lower()
    if m == "tri":
        m = "trie"

    if m == "trie":
        out = run_trunk_extraction_trie(
            episodes=episodes,
            relations=relations,
            config=config,
            graph_cfg=graph_cfg,
            chain_cfg=chain_cfg,
        )
    elif m == "mpc":
        out = run_trunk_extraction_mpc(
            episodes=episodes,
            relations=relations,
            config=config,
            graph_cfg=graph_cfg,
            chain_cfg=chain_cfg,
        )
    else:
        raise ValueError(f"Unknown storyline extraction method: {method}")

    for item in out:
        if isinstance(item, dict):
            item.setdefault("method", m)
    return out

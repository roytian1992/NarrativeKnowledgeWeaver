# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import List, Dict, Tuple, Optional, Any, Set, Callable
from collections import defaultdict, Counter

__all__ = [
    # Existing utilities (kept compatible)
    "remove_subset_paths",
    "overlapping_similarity",
    "remove_similar_paths",
    "_maximal_chain_indices_by_set",
    "_all_covering_windows",
    "get_frequent_subchains_with_subset_removal",
    "get_frequent_subchains",
    # New: confidence-based splitting utilities
    "split_chains_by_confidence",
    "preprocess_chains_by_confidence",
    # New: trunk–branch segmentation
    "segment_trunks_and_branches",
    # Optional: sequence similarity + diversification
    "_lcs_ratio",
    "mmr_select_sequences",
]

# ----------------------------------------------------------------------
# ========================== Core helper functions =====================
# ----------------------------------------------------------------------

def remove_subset_paths(chains: List[List[str]]) -> List[List[str]]:
    """
    Remove chains that are strict subsets (as sets) of any other chain.

    Args:
        chains: List of chains (each chain is a list of node IDs).

    Returns:
        Filtered list of chains with strict set-subsets removed.
    """
    filtered = []
    for i, chain in enumerate(chains):
        set_chain = set(chain)
        remove = False
        for j, other in enumerate(chains):
            if i == j:
                continue
            if set_chain.issubset(set(other)) and len(set_chain) < len(set(other)):
                remove = True
                break
        if not remove:
            filtered.append(chain)
    return filtered


def overlapping_similarity(set1: Set[str], set2: Set[str]) -> float:
    """
    Compute overlap similarity between two sets:
        |set1 ∩ set2| / min(|set1|, |set2|)

    Args:
        set1: First set.
        set2: Second set.

    Returns:
        Similarity score in [0, 1].
    """
    if not set1 and not set2:
        return 1.0
    return len(set1 & set2) / min([len(set1), len(set2)])


def remove_similar_paths(chains: List[List[str]], threshold: float = 0.8) -> List[List[str]]:
    """
    Greedy deduplication by set-overlap similarity against already-kept chains.

    Args:
        chains: List of chains.
        threshold: Overlap similarity threshold for pruning.

    Returns:
        Filtered list of chains.
    """
    filtered: List[List[str]] = []
    for chain in chains:
        set_chain = set(chain)
        keep = True
        for kept in filtered:
            sim = overlapping_similarity(set_chain, set(kept))
            if sim >= threshold:
                keep = False
                break
        if keep:
            filtered.append(chain)
    return filtered


def _maximal_chain_indices_by_set(chains: List[List[str]], min_length: int = 2) -> List[int]:
    """
    Return indices of chains that are maximal by set inclusion among chains
    with length >= min_length.

    Args:
        chains: List of chains.
        min_length: Minimum chain length to consider.

    Returns:
        Indices into `chains` that correspond to maximal sets.
    """
    idxs = [i for i, ch in enumerate(chains) if len(ch) >= min_length]
    sets = [set(chains[i]) for i in idxs]
    inv = defaultdict(set)
    order = sorted(range(len(idxs)), key=lambda k: -len(sets[k]))
    kept_local = []
    kept_sets = []
    for k in order:
        s = sets[k]
        if not s:
            if any(len(ts) > 0 for ts in kept_sets):
                continue
            kept_idx = len(kept_sets)
            kept_local.append(k)
            kept_sets.append(s)
            continue
        sig = sorted(s, key=lambda e: len(inv[e]))
        cand = inv[sig[0]].copy() if sig else set()
        for e in sig[1:]:
            cand &= inv[e]
            if not cand:
                break
        has_strict_superset = any(len(kept_sets[j]) > len(s) for j in cand)
        if has_strict_superset:
            continue
        kept_idx = len(kept_sets)
        kept_local.append(k)
        kept_sets.append(s)
        for e in s:
            inv[e].add(kept_idx)
    return [idxs[k] for k in kept_local]


def _all_covering_windows(chain: List[str], universe: Set[str], min_length: int) -> List[List[str]]:
    """
    Enumerate all contiguous sub-sequences (windows) within `chain` that cover
    every element in `universe` at least once, with window length >= min_length.

    Args:
        chain: Original sequence.
        universe: Set of required elements to be covered.
        min_length: Minimum length of windows.

    Returns:
        List of windows (each a list of node IDs).
    """
    need = {x: 1 for x in universe}
    have = Counter()
    covered = 0
    need_total = len(need)
    res = []
    n = len(chain)
    r = 0
    for l in range(n):
        while r < n and covered < need_total:
            x = chain[r]
            if x in need:
                have[x] += 1
                if have[x] == 1:
                    covered += 1
            r += 1
        if covered == need_total:
            min_r = r - 1
            start_r = max(min_r, l + min_length - 1)
            if start_r < n:
                for rr in range(start_r, n):
                    res.append(chain[l:rr + 1])
        x = chain[l]
        if x in need:
            have[x] -= 1
            if have[x] == 0:
                covered -= 1
    return res


def get_frequent_subchains_with_subset_removal(
    chains: List[List[str]],
    min_length: int = 2,
) -> List[List[str]]:
    """
    Get all covering windows for chains that are maximal by set inclusion.

    Args:
        chains: Input chains.
        min_length: Minimum subchain length.

    Returns:
        List of subchains (sorted by descending length, then lexicographically).
    """
    maximal_idxs = _maximal_chain_indices_by_set(chains, min_length=min_length)
    out: List[List[str]] = []
    for i in maximal_idxs:
        ch = chains[i]
        uni = set(ch)
        out.extend(_all_covering_windows(ch, uni, min_length))
    out.sort(key=lambda seq: (-len(seq), tuple(seq)))
    return out


def get_frequent_subchains(chains: List[List[str]], min_length: int = 2, min_count: int = 2):
    """
    Count all contiguous subchains and return those with frequency >= min_count.

    Args:
        chains: Input chains.
        min_length: Minimum subchain length.
        min_count: Frequency threshold.

    Returns:
        List of frequent subchains (each a list of node IDs).
    """
    counter = Counter()
    for chain in chains:
        n = len(chain)
        for i in range(n):
            for j in range(i + min_length, n + 1):
                sub = tuple(chain[i:j])
                counter[sub] += 1
    results = [(list(sub), cnt) for sub, cnt in counter.items() if cnt >= min_count]
    results.sort(key=lambda x: (-x[1], -len(x[0]), x[0]))
    return [pair[0] for pair in results]


# ----------------------------------------------------------------------
# ==================== Confidence-based splitting & prep ===============
# ----------------------------------------------------------------------

def split_chains_by_confidence(
    chains: List[List[str]],
    edge_info_fn: Callable[[str, str], Optional[Dict[str, Any]]],
    *,
    min_confidence: float = 0.5,
    type_weights: Optional[Dict[str, float]] = None,
    allow_indirect: bool = True,
) -> List[List[str]]:
    """
    Split chains at low-confidence edges and return the resulting subchains.

    Rules:
        - `min_confidence` sets the threshold for effective confidence.
        - `type_weights` reweights confidence per relation type (confidence * weight).
          Default weights:
              {'EVENT_CAUSES': 1.0, 'EVENT_INDIRECT_CAUSES': 0.6, 'EVENT_PART_OF': 0.0}
        - If `allow_indirect` is False, split at 'EVENT_INDIRECT_CAUSES' edges.
        - 'EVENT_PART_OF' is always treated as a split (not a path edge).

    Args:
        chains: Input chains.
        edge_info_fn: Callable (u, v) -> { "type": str, "confidence": float } or None.
        min_confidence: Threshold for effective confidence after weighting.
        type_weights: Optional mapping from relation type to weight.
        allow_indirect: Whether to allow indirect-causation edges.

    Returns:
        Deduplicated list of resulting subchains (length >= 2).
    """
    if type_weights is None:
        type_weights = {
            "EVENT_CAUSES": 1.0,
            "EVENT_INDIRECT_CAUSES": 0.6,
            "EVENT_PART_OF": 0.0,
        }

    out: List[List[str]] = []

    for ch in chains:
        if not ch or len(ch) == 1:
            continue

        cur: List[str] = [ch[0]]

        for u, v in zip(ch, ch[1:]):
            info = edge_info_fn(u, v) if edge_info_fn else None
            etype = (info or {}).get("type", None)
            conf = float((info or {}).get("confidence", 0.0))

            # Always split on PART_OF (not treated as a path edge)
            if etype == "EVENT_PART_OF":
                if len(cur) >= 2:
                    out.append(cur)
                cur = [v]
                continue

            # Optionally split on indirect causation
            if (etype == "EVENT_INDIRECT_CAUSES") and (not allow_indirect):
                if len(cur) >= 2:
                    out.append(cur)
                cur = [v]
                continue

            # Compare weighted confidence to threshold
            w = type_weights.get(etype, type_weights.get("EVENT_INDIRECT_CAUSES", 0.6))
            eff_conf = conf * w

            if eff_conf >= min_confidence:
                cur.append(v)
            else:
                if len(cur) >= 2:
                    out.append(cur)
                cur = [v]

        if len(cur) >= 2:
            out.append(cur)

    # Deduplicate while preserving order
    seen = set()
    uniq: List[List[str]] = []
    for seg in out:
        t = tuple(seg)
        if t not in seen:
            seen.add(t)
            uniq.append(seg)
    return uniq


def preprocess_chains_by_confidence(
    chains: List[List[str]],
    edge_info_fn: Callable[[str, str], Optional[Dict[str, Any]]],
    *,
    min_confidence: float = 0.5,
    type_weights: Optional[Dict[str, float]] = None,
    allow_indirect: bool = True,
    apply_split: bool = True,
) -> List[List[str]]:
    """
    Preprocess chains with optional confidence-based splitting.

    Behavior:
        - If `apply_split` is True, split chains via `split_chains_by_confidence`.
        - Otherwise, only filter out chains of length < 2.

    Args:
        chains: Input chains.
        edge_info_fn: Edge info provider (u, v) -> dict or None.
        min_confidence: Threshold for effective confidence.
        type_weights: Optional per-type weights for confidence.
        allow_indirect: Whether to allow indirect edges.
        apply_split: Whether to perform splitting.

    Returns:
        Preprocessed chains.
    """
    if apply_split:
        return split_chains_by_confidence(
            chains,
            edge_info_fn,
            min_confidence=min_confidence,
            type_weights=type_weights,
            allow_indirect=allow_indirect,
        )
    # No splitting; basic filtering only
    filtered = [ch for ch in chains if len(ch) >= 2]
    return filtered


# ----------------------------------------------------------------------
# =================== Trunk–branch segmentation (path trie) ============
# ----------------------------------------------------------------------
# ----------------------------- #
#   Trunk–Branch segmentation   #
# ----------------------------- #

class _TrieNode:
    """
    Path-trie node used to merge multiple chains into a single tree and to
    segment at branch points or chain endpoints.
    """
    __slots__ = ("children", "count", "end_count", "token")
    def __init__(self, token: Optional[str] = None):
        self.children: Dict[str, _TrieNode] = {}
        self.count: int = 0         # Number of chains passing through this node
        self.end_count: int = 0     # Number of chains ending at this node
        self.token: Optional[str] = token

def _trie_insert(root: _TrieNode, chain: List[str]) -> None:
    """
    Insert a chain into the trie.

    Args:
        root: Trie root node.
        chain: Chain to be inserted.
    """
    node = root
    node.count += 1
    for t in chain:
        if t not in node.children:
            node.children[t] = _TrieNode(t)
        node = node.children[t]
        node.count += 1
    node.end_count += 1


def _collect_segments_from_trie(
    root: _TrieNode,
    *,
    min_len: int = 2,
    include_cutpoint: bool = False,
    keep_terminal_pairs: bool = False,
) -> List[List[str]]:
    """
    Collect segments from a trie according to trunk–branch rules.

    Rules:
      - Segment at branch points (children >= 2) or where any chain ends (end_count > 0):
          * Emit the path segment since the last cut (inclusive) through the current node.
          * For each child, start the next segment either:
                - from the node after the cut (include_cutpoint=False), or
                - from the cut node itself (include_cutpoint=True).
      - If a segment is shorter than `min_len` and `keep_terminal_pairs=True` and
        the current node is a chain end, backfill with the last edge to ensure
        leaf nodes appear at least once.

    Args:
        root: Trie root.
        min_len: Minimum segment length.
        include_cutpoint: Whether to include the cut node at the start of the next segment.
        keep_terminal_pairs: Whether to ensure leaf nodes appear as at least a length-2 segment.

    Returns:
        Unique list of segments with stable generation order.
    """
    segments: List[List[str]] = []
    # Stack items: (node, path_so_far, last_cut_index)
    stack: List[Tuple[_TrieNode, List[str], int]] = [(root, [], 0)]

    while stack:
        node, path, last_cut = stack.pop()

        # Append current token to the path (root token is None and skipped)
        if node.token is not None:
            path = path + [node.token]

        is_branch = len(node.children) >= 2
        is_chain_end = node.end_count > 0

        if is_branch or is_chain_end:
            seg_len = len(path) - last_cut

            if seg_len >= min_len:
                segments.append(path[last_cut:])
            elif keep_terminal_pairs and is_chain_end and len(path) >= 2:
                # Terminal backfill with the last edge (e.g., [B,G] or [H,G])
                segments.append(path[-2:])

            # Continue from branch: decide next segment start index
            for child in node.children.values():
                # If include_cutpoint=True, start next segment from the cut node (include current)
                new_cut = len(path) - (1 if include_cutpoint else 0)
                if new_cut < 0:
                    new_cut = 0
                stack.append((child, path, new_cut))
        else:
            # Continue down without cutting
            for child in node.children.values():
                stack.append((child, path, last_cut))

    # Deduplicate with stable order
    seen: Set[Tuple[str, ...]] = set()
    uniq: List[List[str]] = []
    for seg in segments:
        key = tuple(seg)
        if key not in seen:
            seen.add(key)
            uniq.append(seg)
    return uniq

def segment_trunks_and_branches(
    chains: List[List[str]],
    min_len: int = 2,
    drop_contained: bool = True,
    include_cutpoint: bool = False,
    keep_terminal_pairs: bool = False,
) -> List[List[str]]:
    """
    Segment multiple chains into trunk–branch segments using a path trie.

    Args:
        chains: Input chains (each a sequence of event IDs).
        min_len: Minimum segment length.
        drop_contained: If True, remove segments that are contiguous substrings of longer ones.
        include_cutpoint: If True, start new segments from the cut node (include it).
        keep_terminal_pairs: If True, ensure leaves appear at least once as length-2 segments.

    Returns:
        List of segments after trunk–branch segmentation (deduplicated; and if
        `drop_contained` is True, de-contained by contiguous substring check).
    """
    # Build trie
    root = _TrieNode()
    for ch in chains:
        if ch:
            _trie_insert(root, ch)

    # Collect segments
    segs = _collect_segments_from_trie(
        root,
        min_len=min_len,
        include_cutpoint=include_cutpoint,
        keep_terminal_pairs=keep_terminal_pairs,
    )

    if not drop_contained:
        return segs

    # Remove segments that are contiguous substrings of longer segments
    segs_sorted = sorted(segs, key=lambda s: (-len(s), s))
    kept: List[List[str]] = []
    for s in segs_sorted:
        if not any(_is_substring(s, t) for t in kept):
            kept.append(s)
    return kept


def _is_substring(sub: List[str], full: List[str]) -> bool:
    """
    Check whether `sub` is a contiguous substring of `full`.

    Args:
        sub: Candidate subsequence.
        full: Full sequence.

    Returns:
        True if `sub` is a contiguous substring of `full`, else False.
    """
    n, m = len(sub), len(full)
    if n == 0 or n > m:
        return False
    for i in range(m - n + 1):
        if full[i:i+n] == sub:
            return True
    return False

# ----------------------------------------------------------------------
# ===================== Sequence similarity (LCS) & MMR ================
# ----------------------------------------------------------------------

def _lcs_ratio(a: List[str], b: List[str]) -> float:
    """
    Order-aware sequence similarity via LCS:
        LCS(a, b) / min(len(a), len(b)) in [0, 1].

    Args:
        a: First sequence.
        b: Second sequence.

    Returns:
        LCS ratio in [0, 1].
    """
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return 0.0
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        ai = a[i - 1]
        for j in range(1, m + 1):
            if ai == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = dp[i - 1][j] if dp[i - 1][j] >= dp[i][j - 1] else dp[i][j - 1]
    lcs = dp[n][m]
    return lcs / float(min(n, m))

def mmr_select_sequences(
    sequences: List[List[str]],
    scores: List[float],
    top_k: int = 200,
    lambda_: float = 0.75,
) -> List[List[str]]:
    """
    Diversify selection via Maximal Marginal Relevance (MMR) using LCS-based similarity.

    Objective:
        maximize λ * score(seq) − (1 − λ) * max_sim_with_selected(seq)

    Args:
        sequences: Candidate sequences.
        scores: Relevance scores aligned with `sequences`.
        top_k: Maximum number of sequences to select.
        lambda_: Trade-off between relevance and diversity (higher = more relevance).

    Returns:
        Selected sequences in chosen order.
    """
    assert len(sequences) == len(scores)
    order = sorted(range(len(sequences)), key=lambda i: scores[i], reverse=True)
    selected: List[int] = []
    used = set()

    while order and len(selected) < top_k:
        best_i, best_val = None, float("-inf")
        for i in order:
            if i in used:
                continue
            sim_penalty = 0.0 if not selected else max(_lcs_ratio(sequences[i], sequences[j]) for j in selected)
            val = lambda_ * scores[i] - (1.0 - lambda_) * sim_penalty
            if val > best_val:
                best_val, best_i = val, i
        if best_i is None:
            break
        selected.append(best_i)
        used.add(best_i)
        order.remove(best_i)

    return [sequences[i] for i in selected]

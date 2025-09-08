# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import List, Dict, Tuple, Optional, Any, Set, Callable
from collections import defaultdict, Counter

__all__ = [
    # ===== 你已有的函数：保持兼容 =====
    "remove_subset_paths",
    "overlapping_similarity",
    "remove_similar_paths",
    "_maximal_chain_indices_by_set",
    "_all_covering_windows",
    "get_frequent_subchains_with_subset_removal",
    "get_frequent_subchains",
    # ===== 新增：confidence阈值切断 =====
    "split_chains_by_confidence",
    "preprocess_chains_by_confidence",
    # ===== 新增：主干—分支切分 =====
    "segment_trunks_and_branches",
    # ===== 可选：序列相似度 + 多样化 =====
    "_lcs_ratio",
    "mmr_select_sequences",
]

# ----------------------------------------------------------------------
# =============== 你原有的工具函数：原样保留（兼容） ===============
# ----------------------------------------------------------------------

def remove_subset_paths(chains: List[List[str]]) -> List[List[str]]:
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
    if not set1 and not set2:
        return 1.0
    return len(set1 & set2) / min([len(set1), len(set2)])


def remove_similar_paths(chains: List[List[str]], threshold: float = 0.8) -> List[List[str]]:
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
    maximal_idxs = _maximal_chain_indices_by_set(chains, min_length=min_length)
    out: List[List[str]] = []
    for i in maximal_idxs:
        ch = chains[i]
        uni = set(ch)
        out.extend(_all_covering_windows(ch, uni, min_length))
    out.sort(key=lambda seq: (-len(seq), tuple(seq)))
    return out


def get_frequent_subchains(chains: List[List[str]], min_length: int = 2, min_count: int = 2):
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
# =============== 新增：基于 confidence 的链切断/预处理 ===============
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
    在“低置信边”处切断事件链，返回切分后的子链列表。
    - min_confidence: 置信度阈值（默认0.5，可控）
    - type_weights:   不同边类型的权重（乘在confidence上再与阈值比较）
        默认: {'EVENT_CAUSES':1.0, 'EVENT_INDIRECT_CAUSES':0.6, 'EVENT_PART_OF':0.0}
    - allow_indirect: 若为False，则一律在间接因果边处切断

    注意：该函数不触碰你的判定逻辑，仅做“候选链清洗/切分”。
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

            # 处理 PART_OF：不作为路径边使用，一律切断
            if etype == "EVENT_PART_OF":
                if len(cur) >= 2:
                    out.append(cur)
                cur = [v]
                continue

            # 不允许间接因果：切断
            if (etype == "EVENT_INDIRECT_CAUSES") and (not allow_indirect):
                if len(cur) >= 2:
                    out.append(cur)
                cur = [v]
                continue

            # 依据类型加权后的置信度比较阈值
            w = type_weights.get(etype, type_weights.get("EVENT_INDIRECT_CAUSES", 0.6))
            eff_conf = conf * w

            if eff_conf >= min_confidence:
                # 边通过阈值，继续延长当前子链
                cur.append(v)
            else:
                # 低置信边：切断
                if len(cur) >= 2:
                    out.append(cur)
                cur = [v]

        if len(cur) >= 2:
            out.append(cur)

    # 去重并保持稳定性
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
    预处理入口：
    - 如果 apply_split=True：调用 split_chains_by_confidence 切断低置信边
    - 否则：仅过滤掉“含有任何有效边的长度<2”的链
    """
    if apply_split:
        return split_chains_by_confidence(
            chains,
            edge_info_fn,
            min_confidence=min_confidence,
            type_weights=type_weights,
            allow_indirect=allow_indirect,
        )
    # 不切断，仅做基本过滤
    filtered = [ch for ch in chains if len(ch) >= 2]
    return filtered


# ----------------------------------------------------------------------
# =============== 新增：主干—分支 切分（路径前缀树） ===============
# ----------------------------------------------------------------------
# ----------------------------- #
#   Trunk–Branch segmentation   #
# ----------------------------- #

class _TrieNode:
    """路径前缀树节点：用于把多条事件链合并成一棵树；在分叉或链末尾处分段。"""
    __slots__ = ("children", "count", "end_count", "token")
    def __init__(self, token: Optional[str] = None):
        self.children: Dict[str, _TrieNode] = {}
        self.count: int = 0         # 经过该节点的链条数量
        self.end_count: int = 0     # 在该节点结束的链条数量
        self.token: Optional[str] = token

def _trie_insert(root: _TrieNode, chain: List[str]) -> None:
    """把一条链插入到 trie 中。"""
    node = root
    node.count += 1
    for t in chain:
        if t not in node.children:
            node.children[t] = _TrieNode(t)
        node = node.children[t]
        node.count += 1
    node.end_count += 1

def _is_substring(sub: List[str], full: List[str]) -> bool:
    """判断 sub 是否为 full 的连续子串。"""
    n, m = len(sub), len(full)
    if n == 0 or n > m:
        return False
    for i in range(m - n + 1):
        if full[i:i+n] == sub:
            return True
    return False

def _collect_segments_from_trie(
    root: _TrieNode,
    *,
    min_len: int = 2,
    include_cutpoint: bool = False,
    keep_terminal_pairs: bool = False,
) -> List[List[str]]:
    """
    从 trie 中收集分段。
    规则：
      - 遇到“分叉点”（children>=2）或“有链在此结束”（end_count>0）即切段：
          * 输出从上一次切点后的路径到当前节点作为一个段；
          * 对每个子节点，把“下一段”的起点设为：
                include_cutpoint=False -> 从切点**之后**开始
                include_cutpoint=True  -> **包含**切点开始
      - 若段长 < min_len 且 keep_terminal_pairs=True 且当前位置为链终点，
        则回填最后一条边形成二元段（保证叶子至少出现一次）。

    返回：
      唯一（去重）的段列表，保持生成顺序的稳定性。
    """
    segments: List[List[str]] = []
    # 栈元素: (node, path_so_far, last_cut_index)
    stack: List[Tuple[_TrieNode, List[str], int]] = [(root, [], 0)]

    while stack:
        node, path, last_cut = stack.pop()

        # 把当前 token 追加到路径上（根节点 token=None 不追加）
        if node.token is not None:
            path = path + [node.token]

        is_branch = len(node.children) >= 2
        is_chain_end = node.end_count > 0

        if is_branch or is_chain_end:
            seg_len = len(path) - last_cut

            if seg_len >= min_len:
                segments.append(path[last_cut:])
            elif keep_terminal_pairs and is_chain_end and len(path) >= 2:
                # 终端回填二元段（例如 [B,G] 或 [H,G]）
                segments.append(path[-2:])

            # 分叉续跑：下一段的“切点”位置
            for child in node.children.values():
                # include_cutpoint=True 时，下一段从切点开始（包含当前节点）
                new_cut = len(path) - (1 if include_cutpoint else 0)
                if new_cut < 0:
                    new_cut = 0
                stack.append((child, path, new_cut))
        else:
            # 继续向下，无需切段
            for child in node.children.values():
                stack.append((child, path, last_cut))

    # 去重并保持稳定性
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
    *,
    include_cutpoint: bool = False,
    keep_terminal_pairs: bool = False,
) -> List[List[str]]:
    """
    把多条链按“主干—分支”结构切段。

    参数：
      - chains: 原始候选链（每条为事件 ID 序列）
      - min_len: 最短段长（默认 2）
      - drop_contained: 是否去掉被更长段完全包含的短段（基于有序子串判断）
      - include_cutpoint: 分叉后是否**包含切点**作为新段起点（默认 False）
      - keep_terminal_pairs: 对叶子回填最后一条边形成二元段（默认 False）

    返回：
      按规则分段后的序列列表（已去重；如 drop_contained=True，则做“包含去重”）。
    """
    # 构建 trie
    root = _TrieNode()
    for ch in chains:
        if ch:
            _trie_insert(root, ch)

    # 收集分段
    segs = _collect_segments_from_trie(
        root,
        min_len=min_len,
        include_cutpoint=include_cutpoint,
        keep_terminal_pairs=keep_terminal_pairs,
    )

    if not drop_contained:
        return segs

    # 去掉被更长段完全包含的短段（有序子串）
    segs_sorted = sorted(segs, key=lambda s: (-len(s), s))
    kept: List[List[str]] = []
    for s in segs_sorted:
        if not any(_is_substring(s, t) for t in kept):
            kept.append(s)
    return kept



def _trie_insert(root: _TrieNode, chain: List[str]) -> None:
    node = root
    node.count += 1
    for t in chain:
        if t not in node.children:
            node.children[t] = _TrieNode(t)
        node = node.children[t]
        node.count += 1
    node.end_count += 1

def _collect_segments_from_trie(root: _TrieNode, min_len: int = 2) -> List[List[str]]:
    """
    规则：
    - 从根向下累计 path。
    - 遇到“分叉点”（children>=2）或“有链在此结束”（end_count>0）即切段：
        * 输出从上一次切点后的路径到当前节点作为一个段；
        * 对每个子节点，从下一位置另起一段（不重复包含切点）。
    """
    segments: List[List[str]] = []
    stack: List[Tuple[_TrieNode, List[str], int]] = [(root, [], 0)]  # (node, path, last_cut_index)

    while stack:
        node, path, last_cut = stack.pop()

        if node.token is not None:
            path = path + [node.token]

        is_branch = len(node.children) >= 2
        is_chain_end_here = node.end_count > 0

        if is_branch or is_chain_end_here:
            if len(path) - last_cut >= min_len:
                segments.append(path[last_cut:])

            for child in node.children.values():
                stack.append((child, path, len(path)))  # 子段从下一个位置开始
        else:
            for child in node.children.values():
                stack.append((child, path, last_cut))

    # 去重（保持插入顺序的基础上去重）
    seen = set()
    uniq_segments: List[List[str]] = []
    for seg in segments:
        tup = tuple(seg)
        if tup not in seen:
            seen.add(tup)
            uniq_segments.append(seg)
    return uniq_segments

def _is_substring(sub: List[str], full: List[str]) -> bool:
    """判断 sub 是否为 full 的有序子串（连续）。"""
    n, m = len(sub), len(full)
    if n == 0 or n > m:
        return False
    for i in range(m - n + 1):
        if full[i:i+n] == sub:
            return True
    return False

def segment_trunks_and_branches(
    chains: List[List[str]],
    min_len: int = 2,
    drop_contained: bool = True
) -> List[List[str]]:
    """
    把多条链按“主干—分支”结构切段：
        例： [A,B,C,D,E,F,G], [A,B,C,D,H,I,K]
          -> [A,B,C,D], [E,F,G], [H,I,K]
    参数：
        - min_len:   最短段长
        - drop_contained: 是否去掉被更长段完全包含的短段（基于有序子串判断）
    """
    root = _TrieNode()
    for ch in chains:
        if ch:
            _trie_insert(root, ch)

    segs = _collect_segments_from_trie(root, min_len=min_len)

    if not drop_contained:
        return segs

    # 去掉被更长段完全包含的短段
    segs_sorted = sorted(segs, key=lambda s: (-len(s), s))
    kept: List[List[str]] = []
    for s in segs_sorted:
        if not any(_is_substring(s, t) for t in kept):
            kept.append(s)
    return kept


# ----------------------------------------------------------------------
# =============== 可选：序列相似度（LCS）& MMR 多样化 ===============
# ----------------------------------------------------------------------

def _lcs_ratio(a: List[str], b: List[str]) -> float:
    """
    有序链相似度：LCS(a,b) / min(len(a), len(b))，范围 [0,1]。
    反映“顺序一致”的重合程度。
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
    基于 MMR 的多样化选择（序列相似度用 LCS 比例）：
        maximize λ*score(seq) − (1−λ)*max_sim_with_selected(seq)
    只依赖标准库；如果不需要，可以不调用。
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

import json
import pickle
import glob
import networkx as nx
import hashlib
from typing import List, Dict, Tuple, Optional, Any, Set
from tqdm import tqdm
import time
from collections import Counter
from pathlib import Path
from core.model_providers.openai_llm import OpenAILLM
from core.utils.neo4j_utils import Neo4jUtils
from core.models.data import Entity
from core.builder.manager.graph_manager import GraphManager
from core.storage.graph_store import GraphStore
from core.storage.vector_store import VectorStore
from core.utils.prompt_loader import PromptLoader
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError, wait, FIRST_COMPLETED
from core.utils.format import correct_json_format, format_event_card
import logging
from collections import defaultdict
import os
from core.builder.graph_builder import DOC_TYPE_META


def _run_with_soft_timeout_and_retries(
    self,
    items: List[Any],
    *,
    work_fn,                      # (item) -> result
    key_fn,                       # (item) -> hashable key
    desc_label: str,
    per_task_timeout: float = 600.0,
    retries: int = 2,
    retry_backoff: float = 1.0,
    allow_placeholder_first_round: bool = False,
    placeholder_fn=None,          # (item, exc=None) -> placeholder_result
    should_retry=None             # (result) -> bool  True表示需要进入下一轮
) -> Tuple[Dict[Any, Any], Set[Any]]:
    """
    通用并发执行器：逐轮并发 + 软超时 + 失败/超时仅重试未成功项。
    返回：(results_map, still_failed_keys)
    """
    from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED, TimeoutError as FuturesTimeoutError
    import time
    from tqdm import tqdm

    total = len(items)
    if total == 0:
        return {}, set()

    results: Dict[Any, Any] = {}
    remaining_items = items[:]  # 每轮仅提交未成功项
    still_failed_keys: Set[Any] = set()

    max_rounds = max(1, retries)
    for round_id in range(1, max_rounds + 1):
        if not remaining_items:
            break

        round_failures: Set[Any] = set()
        round_timeouts: Set[Any] = set()

        with ThreadPoolExecutor(max_workers=self.max_workers, thread_name_prefix=f"concur-r{round_id}") as executor:
            fut_info: Dict[Any, Dict[str, Any]] = {}
            for it in remaining_items:
                f = executor.submit(work_fn, it)
                fut_info[f] = {"start": time.monotonic(), "item": it, "key": key_fn(it)}

            pbar = tqdm(total=len(fut_info), desc=f"{desc_label}（第{round_id}/{max_rounds}轮）", ncols=100)
            pending = set(fut_info.keys())

            while pending:
                done, pending = wait(pending, timeout=0.25, return_when=FIRST_COMPLETED)

                # 收集已完成
                for f in done:
                    meta = fut_info.pop(f, None)
                    key = meta["key"]
                    item = meta["item"]
                    try:
                        res = f.result()
                        results[key] = res
                        # 若定义了 should_retry，并判断需要重试，则加入失败集合
                        if callable(should_retry) and should_retry(res):
                            round_failures.add(key)
                    except Exception as e:
                        # 异常：首轮可写占位
                        if allow_placeholder_first_round and round_id == 1 and callable(placeholder_fn):
                            try:
                                results[key] = placeholder_fn(item, exc=e)
                            except Exception:
                                pass
                        round_failures.add(key)
                    pbar.update(1)

                # 处理软超时：取消、可占位、记失败
                now = time.monotonic()
                to_forget = []
                for f in list(pending):
                    meta = fut_info[f]
                    if now - meta["start"] >= per_task_timeout:
                        key = meta["key"]
                        item = meta["item"]
                        try:
                            f.cancel()
                        except Exception:
                            pass
                        if allow_placeholder_first_round and round_id == 1 and callable(placeholder_fn):
                            try:
                                results[key] = placeholder_fn(item, exc=FuturesTimeoutError(f"soft-timeout {per_task_timeout}s"))
                            except Exception:
                                pass
                        round_timeouts.add(key)
                        pbar.update(1)
                        to_forget.append(f)

                for f in to_forget:
                    pending.remove(f)
                    fut_info.pop(f, None)

            pbar.close()

        # 下一轮仅重试当轮失败/超时项
        keys_to_retry = round_failures | round_timeouts
        still_failed_keys |= keys_to_retry
        # 将 keys 映射回 item
        key_set = set(keys_to_retry)
        remaining_items = [it for it in items if key_fn(it) in key_set]

        # 轮间退避
        if remaining_items and round_id < max_rounds and retry_backoff > 0:
            try:
                time.sleep(retry_backoff)
            except Exception:
                pass

    return results, still_failed_keys


# -----------------------------
# 工具函数：链去重/相似度/高频子链
# -----------------------------
def remove_subset_paths(chains: List[List[str]]) -> List[List[str]]:
    """
    删除所有事件集合是其他链事件集合子集的链（忽略顺序、连续性）
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
    """计算两个集合的相似度（|A∩B| / min(|A|, |B|)）"""
    if not set1 and not set2:
        return 1.0
    return len(set1 & set2) / min([len(set1), len(set2)])


def remove_similar_paths(chains: List[List[str]], threshold: float = 0.8) -> List[List[str]]:
    """
    删除与已保留链相似度 >= threshold 的链
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


from typing import List, Set, Tuple, Dict
from collections import defaultdict, Counter

def _maximal_chain_indices_by_set(chains: List[List[str]], min_length: int = 2) -> List[int]:
    """
    返回“事件集合严格极大”的原始链下标集合（仅考虑长度 >= min_length 的链），
    即：set(chain_i) 不是任何 set(chain_j) 的严格子集。
    使用倒排索引加交集做候选剪枝，避免 O(N^2) 全比较。
    """
    # 只保留能产生子链的原始链（长度 >= min_length）
    idxs = [i for i, ch in enumerate(chains) if len(ch) >= min_length]
    sets = [set(chains[i]) for i in idxs]

    # 倒排索引：元素 -> 含有该元素的“已保留极大集合”的下标（在 kept_idxs 序）
    inv = defaultdict(set)

    # 先按“集合大小”降序，这样更大的集合先进入“已保留集合池”，
    # 判断某集合是否有“严格超集”时，只需在池里查。
    order = sorted(range(len(idxs)), key=lambda k: -len(sets[k]))

    kept_local = []         # 在本函数内部的局部下标（指向 idxs/sets 的下标）
    kept_sets = []          # 已保留的集合本体（与 kept_local 对齐）

    for k in order:
        s = sets[k]
        if not s:
            # 空集：若池中已有任一非空集合，则空集是严格子集，丢弃；否则保留第一个空集（与原逻辑等价）
            if any(len(ts) > 0 for ts in kept_sets):
                continue
            kept_idx = len(kept_sets)
            kept_local.append(k)
            kept_sets.append(s)
            # 空集不进倒排
            continue

        # 用倒排索引做候选“严格超集”召回：交所有元素的桶
        sig = sorted(s, key=lambda e: len(inv[e]))  # 稀有元素优先，交集更小
        cand = inv[sig[0]].copy() if sig else set()
        for e in sig[1:]:
            cand &= inv[e]
            if not cand:
                break

        # 候选里只要存在 size 更大的已保留集合，即为严格超集
        has_strict_superset = any(len(kept_sets[j]) > len(s) for j in cand)
        if has_strict_superset:
            continue

        # 保留，并更新倒排
        kept_idx = len(kept_sets)
        kept_local.append(k)
        kept_sets.append(s)
        for e in s:
            inv[e].add(kept_idx)

    # 把局部下标转换为原始链下标
    return [idxs[k] for k in kept_local]


def _all_covering_windows(chain: List[str], universe: Set[str], min_length: int) -> List[List[str]]:
    """
    返回 chain 中所有“覆盖了 universe 全部元素”的连续子链（长度>=min_length）。
    这些窗口的元素集合恰好等于 universe。
    用滑动窗口为每个左端点找到最短右端点，然后右端点向右扩展都合法。
    """
    need = {x: 1 for x in universe}
    have = Counter()
    covered = 0
    need_total = len(need)

    res = []
    n = len(chain)
    r = 0

    for l in range(n):
        # 扩右直到覆盖全部元素或到尾
        while r < n and covered < need_total:
            x = chain[r]
            if x in need:
                have[x] += 1
                if have[x] == 1:  # 第一次覆盖该元素
                    covered += 1
            r += 1

        if covered == need_total:
            # 找到最短覆盖窗口 [l, r-1]，任何 r' >= r-1 也覆盖
            min_r = r - 1
            start_r = max(min_r, l + min_length - 1)
            if start_r < n:
                for rr in range(start_r, n):
                    # 扩展不会引入 universe 之外的新元素（因为 universe=该链的全集）
                    res.append(chain[l:rr+1])

        # 左端点右移前，移除它的贡献
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
    组合版本（仅针对 min_count=1 的场景）：
    等价于：
      A = get_frequent_subchains(chains, min_length=min_length, min_count=1)
      B = remove_subset_paths(A)
    但无需枚举所有子链，只生成最终会存活的那些子链。

    说明：
    - 先找出“事件集合严格极大”的原始链（长度>=min_length才可能产生子链）；
    - 对这些链，枚举所有“覆盖其全集的子链窗口”（其元素集合 == 该链去重后的全集）。
    - 输出即为两步管线的最终存活集合（顺序可能不同，但内容一致）。
    """
    # 1) 取严格极大集合对应的原始链
    maximal_idxs = _maximal_chain_indices_by_set(chains, min_length=min_length)

    # 2) 对每条极大链，枚举所有覆盖全集的窗口（长度>=min_length）
    out: List[List[str]] = []
    for i in maximal_idxs:
        ch = chains[i]
        uni = set(ch)
        out.extend(_all_covering_windows(ch, uni, min_length))

    # 可选：做一个稳定排序（纯为可读性；与原两步的“最终保留集合”内容一致）
    # 排序规则与原 get_frequent_subchains 的次序接近：长度降序，词典序
    out.sort(key=lambda seq: (-len(seq), tuple(seq)))
    return out



def get_frequent_subchains(chains: List[List[str]], min_length: int = 2, min_count: int = 2):
    """
    统计事件链中出现频率较高的连续子链
    Args:
        chains: 事件链列表
        min_length: 最短子链长度
        min_count: 最少出现次数（频率阈值）
    Returns:
        List[List[str]]  子链列表（按频率与长度降序）
    """
    counter = Counter()
    for chain in chains:
        n = len(chain)
        # 枚举所有连续子链
        for i in range(n):
            for j in range(i + min_length, n + 1):
                sub = tuple(chain[i:j])
                counter[sub] += 1
    # 过滤低频
    results = [(list(sub), cnt) for sub, cnt in counter.items() if cnt >= min_count]
    # 按频率、长度排序
    results.sort(key=lambda x: (-x[1], -len(x[0]), x[0]))
    return [pair[0] for pair in results]


# -----------------------------
# 主类：EventCausalityBuilder
# -----------------------------
class EventCausalityBuilder:
    """
    事件因果图构建器

    主要功能：
    1. 从Neo4j加载和排序事件
    2. 通过连通体和社区过滤事件对
    3. 使用extractor检查因果关系
    4. 构建有向带权NetworkX图
    5. 保存和加载图数据（全部写入 EPG 路径）
    6. 构建Plot情节单元图谱
    """

    def __init__(self, config):
        """
        初始化事件因果图构建器

        Args:
            config: KAG配置对象
        """
        self.config = config
        self.llm = OpenAILLM(config)
        self.graph_store = GraphStore(config)
        self.vector_store = VectorStore(config, "documents")
        self.event_fallback = []  # 可以加入Goal和Action

        self.doc_type = config.knowledge_graph_builder.doc_type
        if self.doc_type not in DOC_TYPE_META:
            raise ValueError(f"Unsupported doc_type: {self.doc_type}")
        self.meta = DOC_TYPE_META[self.doc_type]

        self.neo4j_utils = Neo4jUtils(self.graph_store.driver, self.doc_type)
        self.neo4j_utils.load_embedding_model(config.graph_embedding)

        # 初始化Plot相关组件
        prompt_dir = config.knowledge_graph_builder.prompt_dir
        self.prompt_loader = PromptLoader(prompt_dir)
        settings_path = os.path.join(self.config.storage.graph_schema_path, "settings.json")
        if not os.path.exists(settings_path):
            settings_path = self.config.probing.default_background_path

        settings = json.load(open(settings_path, "r", encoding="utf-8"))

        self.system_prompt_text = self.construct_system_prompt(
            background=settings.get("background"),
            abbreviations=settings.get("abbreviations", [])
        )

        self.graph_analyzer = GraphManager(config, self.llm)

        # Plot构建配置参数（默认值）
        self.causality_threshold = "Medium"
        self.logger = logging.getLogger(__name__)
        self.sorted_scenes = []
        self.event_list = []
        self.event2section_map = {}
        self.max_depth = config.event_plot_graph_builder.max_depth
        self.check_weakly_connected_components = True
        self.min_component_size = config.event_plot_graph_builder.min_connected_component_size
        self.max_workers = config.event_plot_graph_builder.max_workers
        self.max_iteration = config.event_plot_graph_builder.max_iterations
        self.check_weakly_connected_components = config.event_plot_graph_builder.check_weakly_connected_components
        self.max_num_triangles = config.event_plot_graph_builder.max_num_triangles

        # 因果关系强度到权重的映射
        self.event_cards: Dict[str, Dict[str, Any]] = {}

        self.logger.info("EventCausalityBuilder初始化完成")

    # -----------------------------
    # Prompt 构建
    # -----------------------------
    def construct_system_prompt(self, background, abbreviations):
        background_info = self.get_background_info(background, abbreviations)
        if self.doc_type == "screenplay":
            system_prompt_id = "agent_prompt_screenplay"
        else:
            system_prompt_id = "agent_prompt_novel"
        system_prompt_text = self.prompt_loader.render_prompt(system_prompt_id, {"background_info": background_info})
        return system_prompt_text

    def get_background_info(self, background, abbreviations):
        bg_block = f"**背景设定**：{background}\n" if background else ""

        def fmt(item: dict) -> str:
            if not isinstance(item, dict):
                return ""
            abbr = (
                item.get("abbr")
                or item.get("full")
                or next((v for k, v in item.items() if isinstance(v, str) and v.strip()), "N/A")
            )
            parts = []
            for k, v in item.items():
                if k in ("abbr", "full"):
                    continue
                if isinstance(v, str) and v.strip():
                    parts.append(v.strip())
            return f"- **{abbr}**: " + " - ".join(parts) if parts else f"- **{abbr}**"

        abbr_block = "\n".join(fmt(item) for item in abbreviations if isinstance(item, dict))
        if background and abbr_block:
            background_info = f"{bg_block}\n{abbr_block}"
        else:
            background_info = bg_block or abbr_block
        return background_info

    # -----------------------------
    # 事件列表构建
    # -----------------------------
    def build_event_list(self) -> List[Entity]:
        """
        构建排序后的事件列表
        Returns:
            排序后的事件列表
        """
        print("🔍 开始构建事件列表...")

        # 1. 获取所有 section 并排序
        section_entities = self.neo4j_utils.search_entities_by_type(
            entity_type=self.meta["section_label"]
        )
        self.sorted_sections = sorted(
            section_entities,
            key=lambda e: int(e.properties.get("order", 99999))
        )
        print(f"✅ 找到 {len(self.sorted_sections)} 个section")

        # 2. 从场景中提取事件
        event_list = []
        event2section_map = {}
        for section in tqdm(self.sorted_sections, desc="提取场景中的事件"):
            results = self.neo4j_utils.search_related_entities(
                source_id=section.id,
                predicate=self.meta["contains_pred"],
                entity_types=["Event"],
                return_relations=False
            )
            # fallback（可选）
            if not results and self.event_fallback:
                results = self.neo4j_utils.search_related_entities(
                    source_id=section.id,
                    relation_type=self.meta["contains_pred"],
                    entity_types=self.event_fallback,
                    return_relations=False
                )
            for result in results:
                if result.id not in event2section_map:
                    event2section_map[result.id] = section.id
                    event_list.append(result)

        self.event_list = event_list
        self.event2section_map = event2section_map

        print(f"✅ 构建完成，共找到 {len(event_list)} 个事件")
        return event_list

    # -----------------------------
    # 事件卡片并发预生成（支持EPG路径缓存）
    # -----------------------------
    def precompute_event_cards(
        self,
        events: List[Entity],
        per_task_timeout: float = 300,
        max_retries: int = 3,
        retry_timeout: float = 60.0,   # 兼容原参；统一策略里用不到这个单独超时
    ) -> Dict[str, Dict[str, Any]]:
        """
        并发为所有事件生成 event_card：
        - 读/写 EPG/event_cards.json
        - 首轮失败/超时写占位；后续轮仅重试失败项
        """
        base = self.config.storage.event_plot_graph_path
        os.makedirs(base, exist_ok=True)
        cache_path = os.path.join(base, "event_cards.json")

        # 读取缓存
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    self.event_cards = json.load(f)
                    if not isinstance(self.event_cards, dict):
                        self.event_cards = {}
            except Exception:
                self.event_cards = {}
        else:
            self.event_cards = {}

        existing = set(self.event_cards.keys())
        pending_events = [e for e in events if e.id not in existing]
        if not pending_events:
            print(f"🗂️ 事件卡片已存在：{len(self.event_cards)} 个，跳过生成。")
            return self.event_cards

        def _collect_related_context_by_section(ev: Entity) -> str:
            ctx_set = set()
            sec_id = self.event2section_map.get(ev.id)
            if sec_id:
                sec = self.neo4j_utils.get_entity_by_id(sec_id)
                titles = sec.properties.get(self.meta['title'], [])
                if isinstance(titles, str):
                    titles = [titles]
                for t in titles or []:
                    try:
                        docs = self.vector_store.search_by_metadata({"title": t})
                        for d in docs:
                            if getattr(d, "content", None):
                                ctx_set.add(d.content)
                    except Exception:
                        pass
            if not ctx_set:
                try:
                    node = self.neo4j_utils.get_entity_by_id(ev.id)
                    chunk_ids = set((node.source_chunks or [])[:50])
                    if chunk_ids:
                        docs = self.vector_store.search_by_ids(list(chunk_ids))
                        for d in docs:
                            if getattr(d, "content", None):
                                ctx_set.add(d.content)
                except Exception:
                    pass
            return "\n".join(ctx_set)

        def _build_one(ev: Entity) -> str:
            info = self.neo4j_utils.get_entity_info(ev.id, "事件", True, True)
            related_ctx = _collect_related_context_by_section(ev)
            out = self.graph_analyzer.generate_event_context(info, related_ctx)
            card = json.loads(correct_json_format(out))["event_card"]
            card = format_event_card(card)  # 大多数实现会返回 str；如返回 dict 可 json.dumps
            return card

        def _placeholder(ev: Entity, exc=None) -> str:
            skeleton = {
                "name": ev.properties.get("name") or ev.name or f"event_{ev.id}",
                "summary": "",
                "time_hint": "unknown",
                "locations": [],
                "participants": [],
                "action": "",
                "outcomes": [],
                "evidence": ""
            }
            return json.dumps(skeleton, ensure_ascii=False)

        # 统一并发执行
        res_map, still_failed = self._run_with_soft_timeout_and_retries(
            pending_events,
            work_fn=_build_one,
            key_fn=lambda e: e.id,
            desc_label="预生成事件卡片",
            per_task_timeout=per_task_timeout,
            retries=max_retries,
            retry_backoff=1.0,
            allow_placeholder_first_round=True,
            placeholder_fn=_placeholder,
            should_retry=None  # 对于卡片，只在异常/超时重试；正常结果不重试
        )

        # 写回
        for k, v in res_map.items():
            self.event_cards[k] = v
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(self.event_cards, f, ensure_ascii=False, indent=2)

        print(f"🗂️ 事件卡片生成完成：总计 {len(self.event_cards)}，本次缺口余 {len(still_failed)}")
        return self.event_cards

    # -----------------------------
    # 候选事件对过滤
    # -----------------------------
    def filter_event_pairs_by_community(
        self,
        events: List[Entity],
        max_depth: int = 3
    ) -> List[Tuple[Entity, Entity]]:
        """
        利用 Neo4j 中 Louvain 结果直接筛选同社区且 max_depth 内可达的事件对
        """
        id2entity = {e.id: e for e in events}
        pairs = self.neo4j_utils.fetch_event_pairs_same_community()
        filtered_pairs = []
        for row in pairs:
            src_id, dst_id = row["srcId"], row["dstId"]
            if src_id in id2entity and dst_id in id2entity:
                filtered_pairs.append((id2entity[src_id], id2entity[dst_id]))
        print(f"[✓] 同社区事件对: {len(filtered_pairs)}")
        return filtered_pairs

    def sort_event_pairs_by_section_order(
        self, pairs: List[Tuple[Entity, Entity]]
    ) -> List[Tuple[Entity, Entity]]:
        def get_order(evt: Entity) -> int:
            sec_id = self.event2section_map.get(evt.id)
            if not sec_id:
                return 99999
            sec = self.neo4j_utils.get_entity_by_id(sec_id)
            return int(sec.properties.get("order", 99999))
        ordered = []
        for e1, e2 in pairs:
            ordered.append((e1, e2) if get_order(e1) <= get_order(e2) else (e2, e1))
        return ordered

    def filter_pair_by_distance_and_similarity(self, pairs):
        filtered_pairs = []
        for pair in tqdm(pairs, desc="筛选节点对"):
            src_id, tgt_id = pair[0].id, pair[1].id
            reachable = self.neo4j_utils.check_nodes_reachable(
                src_id, tgt_id,
                excluded_rels=[self.meta["contains_pred"], "EVENT_CAUSES", "EVENT_INDIRECT_CAUSES", "EVENT_PART_OF"],
                max_depth=self.max_depth
            )
            if reachable:
                filtered_pairs.append(pair)
            else:
                score = self.neo4j_utils.compute_semantic_similarity(src_id, tgt_id)
                if score >= 0.5:
                    filtered_pairs.append(pair)
        return filtered_pairs

    # -----------------------------
    # 因果性判定（并发 + 断点续跑）
    # -----------------------------
    def check_causality_batch(
        self,
        pairs: List[Tuple[Entity, Entity]]
    ) -> Dict[Tuple[str, str], Dict[str, Any]]:
        """
        并发检查事件对的因果关系（统一并发/软超时/重试）
        - 首轮：失败/超时写占位
        - 仅对失败项重试；成功覆盖旧结果
        """
        PER_TASK_TIMEOUT = 1800
        MAX_RETRIES = 2
        RETRY_BACKOFF = 2.0

        def _make_result(src_event, tgt_event,
                        relation="NONE",
                        reason="",
                        temporal_order="Unknown",
                        confidence=0.0,
                        raw_result="",
                        timeout=False) -> Dict[str, Any]:
            res = {
                "src_event": src_event,
                "tgt_event": tgt_event,
                "relation": relation,
                "reason": reason,
                "temporal_order": temporal_order,
                "confidence": float(confidence) if confidence is not None else 0.0,
                "raw_result": raw_result
            }
            if timeout:
                res["causality_timeout"] = True
            return res

        def _get_common_neighbor_info(src_id, tgt_id):
            commons = self.neo4j_utils.get_common_neighbors(src_id, tgt_id, limit=50)
            info = "两个事件具有的共同邻居的信息为：\n"
            if not commons:
                return info + "无"
            for ent_ in commons:
                try:
                    ent_type = "/".join(ent_.type) if isinstance(ent_.type, (list, set, tuple)) else str(ent_.type)
                except Exception:
                    ent_type = "Unknown"
                info += f"- 实体名称：{ent_.name}，实体类型：{ent_type}，相关描述为：{ent_.description}\n"
            return info

        def _ensure_card(e: Entity, info_text: str):
            if e.id in self.event_cards:
                return self.event_cards[e.id]
            out = self.graph_analyzer.generate_event_context(info_text, "")
            card = json.loads(correct_json_format(out))["event_card"]
            card = format_event_card(card)
            self.event_cards[e.id] = card
            return card

        def _work(pair: Tuple[Entity, Entity]) -> Dict[str, Any]:
            src_event, tgt_event = pair
            try:
                info_1 = self.neo4j_utils.get_entity_info(src_event.id, "事件", True, True)
                info_2 = self.neo4j_utils.get_entity_info(tgt_event.id, "事件", True, True)
                related_context = info_1 + "\n" + info_2 + "\n" + _get_common_neighbor_info(src_event.id, tgt_event.id)
                src_card = _ensure_card(src_event, info_1)
                tgt_card = _ensure_card(tgt_event, info_2)

                result_json = self.graph_analyzer.check_event_causality(
                    src_card, tgt_card,
                    system_prompt=self.system_prompt_text,
                    related_context=related_context
                )
                if isinstance(result_json, dict):
                    result_dict = result_json
                    raw_str = json.dumps(result_json, ensure_ascii=False)
                else:
                    result_dict = json.loads(correct_json_format(result_json))
                    raw_str = result_json

                relation = result_dict.get("relation", "NONE")
                reason = result_dict.get("reason", "")
                temporal_order = result_dict.get("temporal_order", "Unknown")
                confidence = result_dict.get("confidence", 0.3)

                return _make_result(
                    src_event, tgt_event,
                    relation=relation,
                    reason=reason,
                    temporal_order=temporal_order,
                    confidence=confidence,
                    raw_result=raw_str,
                    timeout=False
                )
            except Exception as e:
                return _make_result(
                    src_event, tgt_event,
                    relation="NONE",
                    reason=f"检查过程出错: {e}",
                    temporal_order="Unknown",
                    confidence=0.0,
                    raw_result="",
                    timeout=True
                    )

        def _placeholder(pair: Tuple[Entity, Entity], exc=None) -> Dict[str, Any]:
            src_event, tgt_event = pair
            return _make_result(
                src_event, tgt_event,
                relation="NONE",
                reason="软超时/异常，占位返回",
                temporal_order="Unknown",
                confidence=0.0,
                raw_result="",
                timeout=True
            )

        def _should_retry(res: Dict[str, Any]) -> bool:
            if res.get("causality_timeout"):
                return True
            reason = (res.get("reason") or "").strip()
            return ("出错" in reason)

        print(f"🔍 开始并发检查 {len(pairs)} 对事件的因果关系...")

        res_map, still_failed = self._run_with_soft_timeout_and_retries(
            pairs,
            work_fn=_work,
            key_fn=lambda p: (p[0].id, p[1].id),
            desc_label="并发检查因果关系",
            per_task_timeout=PER_TASK_TIMEOUT,
            retries=MAX_RETRIES,
            retry_backoff=RETRY_BACKOFF,
            allow_placeholder_first_round=True,
            placeholder_fn=_placeholder,
            should_retry=_should_retry
        )

        # 标注最终仍失败的键
        for k in still_failed:
            if k in res_map:
                res_map[k]["final_fallback"] = True
                res_map[k]["retries"] = MAX_RETRIES

        print(f"✅ 因果关系并发检查完成（成功 {len(pairs) - len(still_failed)} / {len(pairs)}，仍失败 {len(still_failed)}）")
        return res_map

    # -----------------------------
    # 初始化：子图 + Louvain
    # -----------------------------
    def initialize(self):
        for relation_type in ["EVENT_CAUSES", "EVENT_INDIRECT_CAUSES", "EVENT_PART_OF"]:
            self.neo4j_utils.delete_relation_type(relation_type)

        self.neo4j_utils.create_subgraph(
            graph_name="knowledge_graph",
            exclude_entity_types=[self.meta["section_label"]],
            exclude_relation_types=[self.meta["contains_pred"]],
            force_refresh=True
        )
        self.neo4j_utils.run_louvain(
            graph_name="knowledge_graph",
            write_property="community",
            force_run=True
        )

        self.clear_directory(self.config.event_plot_graph_path)

    def clear_directory(self, path):
        for file in glob.glob(os.path.join(path, "*.json")):
            try:
                os.remove(file)
            except Exception as e:
                print(f"删除失败: {file} -> {e}")

    # -----------------------------
    # 主流程1：构建事件因果图（持久化）
    # -----------------------------
    def build_event_causality_graph(
        self,
        limit_events: Optional[int] = None
    ) -> None:
        """
        完整的事件因果图构建流程（每步结束都把产物写入 EPG 路径）
        产物：
          - event_cards.json
          - event_causality_results.pkl
        """
        print("🚀 开始完整的事件因果图构建流程...")

        base = self.config.storage.event_plot_graph_path
        os.makedirs(base, exist_ok=True)

        # 2. 构建事件列表
        print("\n🔍 构建事件列表...")
        event_list = self.build_event_list()

        # 3. 限制事件数量（用于测试）
        if limit_events and limit_events < len(event_list):
            event_list = event_list[:limit_events]
            print(f"⚠️ 限制处理事件数量为: {limit_events}")

        # 读取已有 event_cards（如果存在），否则生成
        event_cards_path = os.path.join(base, "event_cards.json")
        if os.path.exists(event_cards_path):
            try:
                with open(event_cards_path, "r", encoding="utf-8") as f:
                    self.event_cards = json.load(f)
                    if not isinstance(self.event_cards, dict):
                        self.event_cards = {}
                print(f"🗂️ 复用已有事件卡片：{len(self.event_cards)}")
            except Exception:
                self.event_cards = {}
        else:
            print("\n🧩 并发预生成事件卡片...")
            self.precompute_event_cards(event_list)

        # 4. 过滤事件对
        print("\n🔍 过滤事件对...")
        filtered_pairs = self.filter_event_pairs_by_community(event_list, max_depth=self.max_depth)
        filtered_pairs = self.filter_pair_by_distance_and_similarity(filtered_pairs)
        filtered_pairs = self.sort_event_pairs_by_section_order(filtered_pairs)
        print("     最终候选事件对数量： ", len(filtered_pairs))

        # 5. 检查因果关系（读取 self.event_cards）
        print("\n🔍 检查因果关系...")
        causality_results = self.check_causality_batch(filtered_pairs)

        # —— 保存产物到 EPG 路径
        with open(os.path.join(base, "event_causality_results.pkl"), "wb") as f:
            pickle.dump(causality_results, f)
        with open(event_cards_path, "w", encoding="utf-8") as f:
            json.dump(self.event_cards, f, ensure_ascii=False, indent=2)

        # 6. 写回 EVENT 关系
        print("\n🔗 写回Event间关系...")
        self.write_event_cause_edges(causality_results)
        self.neo4j_utils.create_event_causality_graph("event_causality_graph", force_refresh=True)

    # -----------------------------
    # 写回事件因果边
    # -----------------------------
    def write_event_cause_edges(self, causality_results):
        rows = []
        for (src_id, dst_id), res in causality_results.items():
            rel = (res.get("relation") or "").upper()
            if rel != "NONE":
                confidence = float(res.get("confidence", 0.3) or 0.0)
                rows.append({
                    "srcId": src_id,
                    "dstId": dst_id,
                    "confidence": confidence,
                    "reason": res.get("reason", ""),
                    "predicate": res.get("relation", "NONE")
                })
        self.neo4j_utils.write_event_causes(rows)

    # -----------------------------
    # SABER：结构约束的边精简（持久化每轮删除）
    # -----------------------------
    def detect_flattened_causal_patterns(self, edges: List[Dict]) -> List[Dict]:
        forward_graph = defaultdict(set)
        edge_set = set()
        for edge in edges:
            sid = edge["sid"]
            tid = edge["tid"]
            forward_graph[sid].add(tid)
            edge_set.add((sid, tid))

        patterns = []
        for a, a_children in forward_graph.items():
            a_children = list(a_children)
            if len(a_children) < 2:
                continue
            internal_links = []
            for i in range(len(a_children)):
                for j in range(len(a_children)):
                    if i == j:
                        continue
                    u, v = a_children[i], a_children[j]
                    if (u, v) in edge_set:
                        internal_links.append((u, v))
            if internal_links:
                patterns.append({
                    "source": a,
                    "targets": a_children,
                    "internal_links": internal_links
                })
        return patterns

    def filter_weak_edges_in_patterns(
        self,
        patterns: List[Dict],
        edge_map: Dict[Tuple[str, str], Dict],
        conf_threshold: float = 0.5
    ) -> List[Dict]:
        cleaned_patterns = []
        for pat in patterns:
            src = pat["source"]
            targets = pat["targets"]
            internals = pat["internal_links"]

            new_targets = []
            for t in targets:
                info = edge_map.get((src, t))
                confidence = info.get("confidence", 0) if info else 0
                if not info:
                    continue
                if confidence < conf_threshold:
                    new_targets.append(t)

            new_internals = []
            for u, v in internals:
                info = edge_map.get((u, v))
                confidence = info.get("confidence", 0) if info else 0
                if not info:
                    continue
                if not confidence < conf_threshold:
                    new_internals.append((u, v))

            if len(new_targets) >= 2 and new_internals:
                cleaned_patterns.append({
                    "source": src,
                    "targets": new_targets,
                    "internal_links": new_internals
                })
        return cleaned_patterns

    def collect_removed_edges(self,
                              original_patterns: List[Dict],
                              filtered_patterns: List[Dict]
                              ) -> Set[Tuple[str, str]]:
        def extract_edges(patterns: List[Dict]) -> Set[Tuple[str, str]]:
            edge_set = set()
            for pat in patterns:
                src = pat["source"]
                for tgt in pat["targets"]:
                    edge_set.add((src, tgt))
                edge_set.update(pat["internal_links"])
            return edge_set

        origin_edges = extract_edges(original_patterns)
        filtered_edges = extract_edges(filtered_patterns)
        removed_edges = origin_edges - filtered_edges
        print(f"[+] Found {len(removed_edges)} candidate edges removed due to pattern collapse")
        return list(removed_edges)

    def filter_pattern(self, pattern, edge_map):
        source = pattern["source"]
        targets = pattern["targets"]
        internal_links = pattern["internal_links"]
        context_to_check = []
        for link in internal_links:
            mid_tgt_sim = self.neo4j_utils.compute_semantic_similarity(link[0], link[1])
            src_mid_sim = self.neo4j_utils.compute_semantic_similarity(source, link[0])
            src_tgt_sim = self.neo4j_utils.compute_semantic_similarity(source, link[1])

            mid_tgt_conf = edge_map.get((link[0], link[1]))["confidence"]
            src_mid_conf = edge_map.get((source, link[0]))["confidence"]
            src_tgt_conf = edge_map.get((source, link[1]))["confidence"]

            # if (src_mid_sim > src_tgt_sim and mid_tgt_sim > src_tgt_sim) or (src_mid_conf > src_tgt_conf and mid_tgt_conf > src_tgt_conf):
            context_to_check.append({
                "entities": [source, link[0], link[1]],
                "details": [
                    {"edge": [source, link[0]], "similarity": src_mid_sim, "confidence": src_mid_conf},
                    {"edge": [source, link[1]], "similarity": src_tgt_sim, "confidence": src_tgt_conf},
                    {"edge": [link[0], link[1]], "similarity": mid_tgt_sim, "confidence": mid_tgt_conf},
                ]
            })
        return context_to_check

    def prepare_context(self, pattern_detail):
        def _safe_str(x: Any) -> str:
            return x if isinstance(x, str) else ("" if x is None else str(x))

        event_details = self.neo4j_utils.get_event_details(pattern_detail["entities"])
        full_event_details = "三个事件实体的描述如下：\n"

        for i, event_info in enumerate(event_details):
            event_id = event_info["event_id"]
            full_event_details += f"**事件{i+1}的相关描述如下：**\n事件id：{event_id}\n"

            background = self.neo4j_utils.get_entity_info(event_id, "事件", True, True)
            background = _safe_str(background)

            props_raw = event_info.get("event_properties")
            if isinstance(props_raw, dict):
                event_props = props_raw
            elif isinstance(props_raw, str) and props_raw.strip():
                try:
                    event_props = json.loads(props_raw)
                    if not isinstance(event_props, dict):
                        event_props = {}
                except Exception:
                    event_props = {}
            else:
                event_props = {}

            non_empty_props = {k: v for k, v in event_props.items() if isinstance(v, str) and v.strip()}

            if non_empty_props:
                background += "\n事件的属性如下：\n"
                for k, v in non_empty_props.items():
                    background += f"- {k}：{v}\n"

            if i + 1 != len(event_details):
                background += "\n"

            full_event_details += background

        full_relation_details = "它们之间已经存在的因果关系有：\n"
        relation_details = pattern_detail["details"]
        for i, relation_info in enumerate(relation_details):
            src, tgt = relation_info["edge"]
            rel_summary = self.neo4j_utils.get_relation_summary(src, tgt, "EVENT_CAUSES")
            rel_summary = _safe_str(rel_summary)
            background = f"{i+1}. " + rel_summary
            background += f"\n关系的语义相似度为：{round(relation_info['similarity'], 4)}，置信度为：{relation_info['confidence']}。"
            if i + 1 != len(relation_details):
                background += "\n\n"
            full_relation_details += background

        return full_event_details, full_relation_details

    def run_SABER(self):
        """
        执行基于结构+LLM的因果边精简优化过程
        产物：
          - saber_removed_edges_round_{i}.json（每轮一份，写入 EPG）
        """
        loop_count = 0
        while True:
            print(f"\n===== [第 {loop_count + 1} 轮优化] =====")

            scc_components = self.neo4j_utils.fetch_scc_components("event_causality_graph", 2)
            wcc_components = []
            if self.check_weakly_connected_components:
                wcc_components = self.neo4j_utils.fetch_wcc_components("event_causality_graph", self.min_component_size)

            connected_components = scc_components + wcc_components
            print(f"📌 当前连通体数量：SCC={len(scc_components)}，WCC={len(wcc_components)}")

            all_triangles = []
            edge_map_global = {}

            for cc in connected_components:
                node_map, edges = self.neo4j_utils.load_connected_components_subgraph(cc)
                edge_map = {
                    (e["sid"], e["tid"]): {"confidence": e.get("confidence", 0.0)}
                    for e in edges
                }
                edge_map_global.update(edge_map)

                old_patterns = self.detect_flattened_causal_patterns(edges)
                new_patterns = self.filter_weak_edges_in_patterns(old_patterns, edge_map, conf_threshold=0.5)
                for pattern in new_patterns:
                    all_triangles += self.filter_pattern(pattern, edge_map)

            print(f"🔺 本轮需判断的三元因果结构数量：{len(all_triangles)}")
            if len(all_triangles) >= self.max_num_triangles:
                print(f"⚠️ 检测到三元结构数量过多，只选择前{self.max_num_triangles}个进行处理。")
                all_triangles = all_triangles[:self.max_num_triangles]
                return

            if loop_count >= 1:
                if len(scc_components) == 0 and len(all_triangles) == 0:
                    print("✅ 图结构已无强连通体，且无待判定三元结构，任务终止。")
                    break
            elif loop_count >= self.max_iteration:
                break

            removed_edges = []

            def process_triangle(triangle_):
                try:
                    event_details, relation_details = self.prepare_context(triangle_)
                    chunks = [self.neo4j_utils.get_entity_by_id(ent_id).source_chunks[0] for ent_id in triangle_["entities"]]
                    chunks = list(set(chunks))
                    documents = self.vector_store.search_by_ids(chunks)
                    results = {doc.content for doc in documents}
                    related_context = "\n".join(list(results))

                    output = self.graph_analyzer.evaluate_event_redundancy(
                        event_details, relation_details, self.system_prompt_text, related_context
                    )
                    output = json.loads(correct_json_format(output))
                    if output.get("remove_edge", False):
                        return (triangle_["entities"][0], triangle_["entities"][2])
                except Exception as e:
                    print(f"[⚠️ 错误] Triangle 判断失败: {triangle_['entities']}, 错误信息: {str(e)}")
                return None

            print(f"🧠 正在并发判断三元结构...")
            tri_map, tri_failed = self._run_with_soft_timeout_and_retries(
                all_triangles,
                work_fn=process_triangle,                 # (tri) -> Optional[Tuple[src, tgt]]
                key_fn=lambda tri: tuple(tri["entities"]),
                desc_label="LLM判断三元结构",
                per_task_timeout=600.0,
                retries=1,
                retry_backoff=1.0,
                allow_placeholder_first_round=False,
                placeholder_fn=None,
                should_retry=lambda r: r is None          # None 代表未能给出裁决
            )

            removed_edges = [edge for edge in tri_map.values() if edge]
            print(f"❌ 本轮待定移除边数量：{len(set(removed_edges))}")

            # —— 删除边
            for edge in removed_edges:
                self.neo4j_utils.delete_relation_by_ids(edge[0], edge[1], ["EVENT_CAUSES", "EVENT_INDIRECT_CAUSES", "EVENT_PART_OF"])

            # —— 持久化本轮删除日志（EPG）
            base = self.config.storage.event_plot_graph_path
            os.makedirs(base, exist_ok=True)
            saber_log_path = os.path.join(base, f"saber_removed_edges_round_{loop_count+1}.json")
            try:
                with open(saber_log_path, "w", encoding="utf-8") as f:
                    json.dump([list(e) for e in set(removed_edges)], f, ensure_ascii=False, indent=2)
            except Exception:
                pass

            # —— 刷新 GDS 图
            self.neo4j_utils.create_event_causality_graph("event_causality_graph", min_confidence=0, force_refresh=True)
            loop_count += 1

    # -----------------------------
    # 工具：提取所有事件链 & 文本上下文
    # -----------------------------
    def get_all_event_chains(self, min_confidence: float = 0.0):
        """
        获取所有可能的事件链（从起点到没有出边的终点）
        """
        starting_events = self.neo4j_utils.get_starting_events()
        chains = []
        for event in starting_events:
            all_chains = self.neo4j_utils.find_event_chain(event, min_confidence)
            chains.extend([chain for chain in all_chains if len(chain) >= 2])
        return chains

    def prepare_chain_context(self, chain):
        if len(chain) > 1:
            context = "事件链：" + "->".join(chain) + "\n\n事件具体信息如下：\n"
        else:
            context = f"事件：{chain[0]}" + "\n\n事件具体信息如下：\n"
        for i, event in enumerate(chain):
            # 这里直接使用 event_cards，避免再次拼装长文本
            context += f"事件{i+1}：{event}\n" + self.event_cards[event] + "\n"
        return context

    def prepare_graph_embeddings(self):
        self.neo4j_utils.load_embedding_model(self.config.graph_embedding)
        self.neo4j_utils.create_vector_index()
        self.neo4j_utils.process_all_embeddings(
            entity_types=["Event", "Plot"]
        )
        self.neo4j_utils.ensure_entity_superlabel()
        print("✅ 事件情节图向量构建完成")

    # -----------------------------
    # 主流程2：构建情节-事件图（读写EPG）
    # -----------------------------
    def build_event_plot_graph(self):
        """
        构建情节-事件图
        读：EPG/event_cards.json
        写：EPG/filtered_event_chains.json
        """
        # 清空旧的 Plot 图与关系（已适配新六种 Plot 关系 + HAS_EVENT）
        self.neo4j_utils.reset_event_plot_graph()

        base = self.config.storage.event_plot_graph_path
        os.makedirs(base, exist_ok=True)

        # 读取事件卡片（EPG）
        cards_path = os.path.join(base, "event_cards.json")
        with open(cards_path, "r", encoding="utf-8") as f:
            self.event_cards = json.load(f)

        all_chains = self.get_all_event_chains(min_confidence=0.0)
        print("[✓] 当前事件链总数：", len(all_chains))

        # filtered_chains = get_frequent_subchains(all_chains, 2, 1)
        # filtered_chains = remove_subset_paths(filtered_chains)
        filtered_chains = get_frequent_subchains_with_subset_removal(all_chains, 2)
        filtered_chains = remove_similar_paths(filtered_chains, 0.75)
        print("[✓] 过滤后事件链总数：", len(filtered_chains))

        # —— 保存筛后链条到 EPG
        with open(os.path.join(base, "filtered_event_chains.json"), "w", encoding="utf-8") as f:
            json.dump(filtered_chains, f, ensure_ascii=False, indent=2)

        def _stable_plot_id(title: str, chain: list[str]) -> str:
            key = f"{title}||{'->'.join(chain)}"
            return "plot_" + hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]

        def _to_bool(v) -> bool:
            if isinstance(v, bool):
                return v
            if v is None:
                return False
            return str(v).strip().lower() in ("true", "yes", "1")

        def process_chain(chain):
            try:
                event_chain_info = self.prepare_chain_context(chain)

                chunk_ids = []
                for ent_id in chain:
                    ent = self.neo4j_utils.get_entity_by_id(ent_id)
                    if not ent:
                        continue
                    sc = ent.source_chunks or []
                    if sc:
                        chunk_ids.append(sc[0])
                chunk_ids = list(set(chunk_ids))

                related_context = ""
                if chunk_ids:
                    documents = self.vector_store.search_by_ids(chunk_ids)
                    contents = {getattr(doc, "content", "") for doc in documents if getattr(doc, "content", "")}
                    related_context = "\n".join(list(contents))

                raw = self.graph_analyzer.generate_event_plot(
                    event_chain_info=event_chain_info,
                    system_prompt=self.system_prompt_text,
                    related_context=related_context
                )
                result = json.loads(correct_json_format(raw))

                if not _to_bool(result.get("is_plot")):
                    return False

                plot_info = result.get("plot_info") or {}
                title = (plot_info.get("title") or "").strip()
                if not title:
                    title = f"情节链：{chain[0]}→{chain[-1]}"

                plot_info["id"] = _stable_plot_id(title, chain)
                plot_info["event_ids"] = chain
                plot_info["reason"] = result.get("reason", "")

                self.neo4j_utils.write_plot_to_neo4j(plot_data=plot_info)
                return True

            except Exception as e:
                print(f"[!] 处理事件链 {chain} 时出错: {e}")
                return False

        chain_map, chain_failed = self._run_with_soft_timeout_and_retries(
            filtered_chains,
            work_fn=process_chain,                 # (chain) -> bool
            key_fn=lambda ch: tuple(ch),
            desc_label="并发生成情节图谱",
            per_task_timeout=900.0,
            retries=2,
            retry_backoff=1.0,
            allow_placeholder_first_round=False,
            placeholder_fn=None,
            should_retry=lambda ok: not bool(ok)   # False/异常/超时 -> 重试
        )

        success_count = sum(1 for v in chain_map.values() if v)
        print(f"[✓] 成功生成情节数量：{success_count}/{len(filtered_chains)} ；仍失败 {len(chain_failed)}")

        return

    # -----------------------------
    # 主流程3：抽取情节间关系（写EPG）
    # -----------------------------
    def generate_plot_relations(self):
        """
        基于候选情节对，判定并写入情节间关系。
        关系集：
        - 有向：PLOT_PREREQUISITE_FOR, PLOT_ADVANCES, PLOT_BLOCKS, PLOT_RESOLVES
        - 无向：PLOT_CONFLICTS_WITH, PLOT_PARALLELS
        产物：EPG/plot_relations_created.json
        """
        # 预处理：向量、GDS 图与嵌入
        self.neo4j_utils.process_all_embeddings(entity_types=[self.meta["section_label"]])
        self.neo4j_utils.create_event_plot_graph()
        self.neo4j_utils.run_node2vec()

        # 召回候选情节对
        all_plot_pairs = self.neo4j_utils.get_plot_pairs(threshold=0)
        print("[✓] 待判定情节关系数量：", len(all_plot_pairs))

        DIRECTED = {
            "PLOT_PREREQUISITE_FOR",
            "PLOT_ADVANCES",
            "PLOT_BLOCKS",
            "PLOT_RESOLVES",
        }
        UNDIRECTED = {
            "PLOT_CONFLICTS_WITH",
            "PLOT_PARALLELS",
        }
        VALID_TYPES = DIRECTED | UNDIRECTED | {"None", None}

        # 并发参数
        PER_TASK_TIMEOUT = 900.0
        MAX_RETRIES = 2
        RETRY_BACKOFF = 1.0

        def _make_edge(src_id, tgt_id, rtype, confidence, reason):
            return {
                "src": src_id,
                "tgt": tgt_id,
                "relation_type": rtype,
                "confidence": float(confidence) if confidence is not None else 0.0,
                "reason": reason or ""
            }

        # 工作函数：返回 {"status": "ok"|"none"|"error", "edges": List[edge], "reason": str}
        def _work(pair: dict) -> dict:
            try:
                plot_A_info = self.neo4j_utils.get_entity_info(
                    pair["src"], "情节", contain_properties=True, contain_relations=True
                )
                plot_B_info = self.neo4j_utils.get_entity_info(
                    pair["tgt"], "情节", contain_properties=True, contain_relations=True
                )

                # LLM/规则判定
                result = self.graph_analyzer.extract_plot_relation(
                    plot_A_info, plot_B_info, self.system_prompt_text
                )

                # 尝试修正/解析 JSON
                try:
                    result = json.loads(correct_json_format(result))
                except Exception:
                    if not isinstance(result, dict):
                        raise

                rtype = result.get("relation_type")
                direction = result.get("direction", None)  # "A->B"/"B->A"（有向）
                confidence = result.get("confidence", 0.0)
                reason = result.get("reason", "")

                # 过滤非法类型
                if rtype not in VALID_TYPES:
                    return {"status": "error", "edges": [], "reason": f"unknown relation_type: {rtype}"}

                # 无关系
                if rtype in {"None", None}:
                    return {"status": "none", "edges": [], "reason": "no-relation"}

                # 生成边
                if rtype in DIRECTED:
                    if direction == "A->B":
                        edges = [_make_edge(pair["src"], pair["tgt"], rtype, confidence, reason)]
                    elif direction == "B->A":
                        edges = [_make_edge(pair["tgt"], pair["src"], rtype, confidence, reason)]
                    else:
                        # 缺少或非法方向：作为错误以便重试
                        return {"status": "error", "edges": [], "reason": f"missing/invalid direction: {direction}"}
                else:  # 无向：写双向边
                    edges = [
                        _make_edge(pair["src"], pair["tgt"], rtype, confidence, reason),
                        _make_edge(pair["tgt"], pair["src"], rtype, confidence, reason),
                    ]

                return {"status": "ok", "edges": edges, "reason": ""}

            except Exception as e:
                return {"status": "error", "edges": [], "reason": str(e)}

        # 仅对 "error" 重试；"none" 不重试；"ok" 不重试
        def _should_retry(res: dict) -> bool:
            return isinstance(res, dict) and res.get("status") == "error"

        # 统一并发执行
        res_map, still_failed = self._run_with_soft_timeout_and_retries(
            all_plot_pairs,
            work_fn=_work,
            key_fn=lambda p: (p["src"], p["tgt"]),
            desc_label="抽取情节关系",
            per_task_timeout=PER_TASK_TIMEOUT,
            retries=MAX_RETRIES,
            retry_backoff=RETRY_BACKOFF,
            allow_placeholder_first_round=False,
            placeholder_fn=None,
            should_retry=_should_retry
        )

        # 汇总边
        edges_to_add = []
        for out in res_map.values():
            if isinstance(out, dict) and out.get("status") == "ok" and out.get("edges"):
                edges_to_add.extend(out["edges"])

        # 批量写入 Neo4j + EPG
        if edges_to_add:
            self.neo4j_utils.create_plot_relations(edges_to_add)
            print(f"[✓] 已创建情节关系 {len(edges_to_add)} 条")

            base = self.config.storage.event_plot_graph_path
            os.makedirs(base, exist_ok=True)
            with open(os.path.join(base, "plot_relations_created.json"), "w", encoding="utf-8") as f:
                json.dump(edges_to_add, f, ensure_ascii=False, indent=2)
        else:
            print("[!] 没有生成任何情节关系")

        if still_failed:
            print(f"[!] 仍失败的情节对数量：{len(still_failed)}")

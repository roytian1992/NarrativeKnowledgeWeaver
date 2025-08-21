"""
事件因果图构建器
负责构建事件因果关系的有向带权图和情节单元图谱
"""

import json
import pickle
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


def jaccard_similarity(set1: Set[str], set2: Set[str]) -> float:
    """计算两个集合的 Jaccard 相似度（用户自定义：|A∩B| / max(|A|, |B|)）"""
    if not set1 and not set2:
        return 1.0
    return len(set1 & set2) / max([len(set1), len(set2)])


def remove_similar_paths(chains: List[List[str]], threshold: float = 0.8) -> List[List[str]]:
    """
    删除与已保留链 Jaccard 相似度 >= threshold 的链
    """
    filtered: List[List[str]] = []
    for chain in chains:
        set_chain = set(chain)
        keep = True
        for kept in filtered:
            sim = jaccard_similarity(set_chain, set(kept))
            if sim >= threshold:
                keep = False
                break
        if keep:
            filtered.append(chain)
    return filtered


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
    # 按频率排序
    results.sort(key=lambda x: (-x[1], -len(x[0]), x[0]))

    return [pair[0] for pair in results]


class EventCausalityBuilder:
    """
    事件因果图构建器

    主要功能：
    1. 从Neo4j加载和排序事件
    2. 通过连通体和社区过滤事件对
    3. 使用extractor检查因果关系
    4. 构建有向带权NetworkX图
    5. 保存和加载图数据
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
            background=settings["background"],
            abbreviations=settings["abbreviations"]
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

        # ---------- 2) 缩写表（键名宽容） ----------
        def fmt(item: dict) -> str:
            """
            将一个缩写项转为 Markdown 列表条目。任何字段都可选，标题字段优先级为：
            abbr > full > 其他字段 > N/A
            """
            if not isinstance(item, dict):
                return ""

            # 标题字段优先级
            abbr = (
                item.get("abbr")
                or item.get("full")
                or next((v for k, v in item.items() if isinstance(v, str) and v.strip()), "N/A")
            )

            # 剩下字段去除标题字段
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

    def build_event_list(self) -> List[Entity]:
        """
        构建排序后的事件列表

        Returns:
            排序后的事件列表
        """
        print("🔍 开始构建事件列表...")

        # 1. 获取所有场景并排序
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
            # 优先查找事件
            results = self.neo4j_utils.search_related_entities(
                source_id=section.id,
                predicate=self.meta["contains_pred"],
                entity_types=["Event"],
                return_relations=False
            )

            # 如果场景中没有事件，则用动作或者目标来填充（如果你启用了 fallback）
            if not results and self.event_fallback:
                results = self.neo4j_utils.search_related_entities(
                    source_id=section.id,
                    relation_type=self.meta["contains_pred"],  # 若方法兼容 alias，这里可保留
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

    # =========================
    # 事件卡片并发预生成（新增）
    # =========================
    def precompute_event_cards(
        self,
        events: List[Entity],
        per_task_timeout: float = 180,
        max_retries: int = 3,
        retry_timeout: float = 60.0,
    ) -> Dict[str, Dict[str, Any]]:
        """
        并发为所有事件生成 event_card：
        - 软超时 + 多轮重试（仅对失败项）
        - 生成结果写入 self.event_cards 并落盘
        返回：{event_id: event_card}
        """
        from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
        import time

        def _collect_related_context_by_section(ev: Entity) -> str:
            """
            优先通过所属 section 的 title 召回文档；若失败，回退到事件的 source_chunks。
            """
            ctx_set = set()

            # 1) 通过 section 标题检索
            sec_id = self.event2section_map.get(ev.id)
            if sec_id:
                sec = self.neo4j_utils.get_entity_by_id(sec_id)
                titles = sec.properties.get(self.meta["title"], [])
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

            # 2) 回退：事件自己的 source_chunks
            if not ctx_set:
                try:
                    node = self.neo4j_utils.get_entity_by_id(ev.id)
                    chunk_ids = set((node.source_chunks or [])[:50])  # 安全上限
                    if chunk_ids:
                        docs = self.vector_store.search_by_ids(list(chunk_ids))
                        for d in docs:
                            if getattr(d, "content", None):
                                ctx_set.add(d.content)
                except Exception:
                    pass

            return "\n".join(ctx_set)

        def _build_one(ev: Entity) -> Tuple[str, Dict[str, Any]]:
            # 事件自身结构化信息（含属性/关系）
            info = self.neo4j_utils.get_entity_info(
                ev.id, entity_type="事件",
                contain_properties=True, contain_relations=True
            )
            related_ctx = _collect_related_context_by_section(ev)

            # 生成卡片
            out = self.graph_analyzer.generate_event_context(info, related_ctx)
            card = json.loads(correct_json_format(out))["event_card"]
            card = format_event_card(card)
            return ev.id, card

        def _run_batch(evts: List[Entity], timeout: float, allow_placeholder: bool, desc: str):
            """通用并发跑一批；返回 (batch_map, failed_ids)。"""
            results: Dict[str, Any] = {}
            failed: Set[str] = set()

            executor = ThreadPoolExecutor(max_workers=self.max_workers, thread_name_prefix="card")
            try:
                fut_info = {}
                for ev in evts:
                    f = executor.submit(_build_one, ev)
                    fut_info[f] = {"start": time.monotonic(), "event": ev}

                pbar = tqdm(total=len(fut_info), desc=desc, ncols=100)
                pending = set(fut_info.keys())

                while pending:
                    done, pending = wait(pending, timeout=0.25, return_when=FIRST_COMPLETED)

                    # 收集已完成
                    for f in done:
                        ev = fut_info[f]["event"]
                        try:
                            eid, card = f.result()
                            results[eid] = card
                        except Exception:
                            failed.add(ev.id)
                            if allow_placeholder:
                                # 占位 skeleton，保证有值；存为 str 避免拼接报错
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
                                results[ev.id] = json.dumps(skeleton, ensure_ascii=False)
                        pbar.update(1)
                        fut_info.pop(f, None)

                    # 软超时
                    now = time.monotonic()
                    to_forget = []
                    for f in list(pending):
                        start = fut_info[f]["start"]
                        if now - start >= timeout:
                            ev = fut_info[f]["event"]
                            f.cancel()
                            failed.add(ev.id)
                            if allow_placeholder:
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
                                results[ev.id] = json.dumps(skeleton, ensure_ascii=False)
                            pbar.update(1)
                            to_forget.append(f)

                    for f in to_forget:
                        pending.remove(f)
                        fut_info.pop(f, None)

                pbar.close()
            finally:
                executor.shutdown(wait=False, cancel_futures=True)

            return results, failed

        # === 首轮：允许占位 ===
        head_map, failed_ids = _run_batch(
            events, timeout=per_task_timeout, allow_placeholder=True, desc="预生成事件卡片（首轮）"
        )
        card_map = dict(head_map)

        # === 重试：仅对失败项，成功则覆盖占位 ===
        need_ids = list(failed_ids)
        for attempt in range(1, max_retries + 1):
            if not need_ids:
                break
            # 轻微退避
            try:
                time.sleep(min(2 ** (attempt - 1), 5.0))
            except Exception:
                pass

            id2evt = {e.id: e for e in events}
            retry_evts = [id2evt[i] for i in need_ids if i in id2evt]
            retry_map, retry_failed = _run_batch(
                retry_evts, timeout=retry_timeout, allow_placeholder=False,
                desc=f"预生成事件卡片（重试 {attempt}/{max_retries}）"
            )

            # 覆盖成功项（替换掉首轮占位）
            for eid, card in retry_map.items():
                card_map[eid] = card

            need_ids = list(retry_failed)

        # 写入内存 + 落盘
        self.event_cards = card_map
        base = self.config.storage.knowledge_graph_path
        os.makedirs(base, exist_ok=True)
        with open(os.path.join(base, "event_cards.json"), "w", encoding="utf-8") as f:
            json.dump(self.event_cards, f, ensure_ascii=False, indent=2)

        print(f"🗂️ 事件卡片生成完成：成功 {len(card_map)} / 总计 {len(events)}；仍缺失 {len(need_ids)}")
        return card_map


    def filter_event_pairs_by_community(
        self,
        events: List[Entity],
        max_depth: int = 3
    ) -> List[Tuple[Entity, Entity]]:
        """
        利用 Neo4j 中 Louvain 结果直接筛选同社区且 max_depth 内可达的事件对
        """
        # 把事件 ID 做成集合，便于后面实体映射
        id2entity = {e.id: e for e in events}

        pairs = self.neo4j_utils.fetch_event_pairs_same_community()
        filtered_pairs = []
        for row in pairs:
            src_id, dst_id = row["srcId"], row["dstId"]
            if src_id in id2entity and dst_id in id2entity:
                filtered_pairs.append((id2entity[src_id], id2entity[dst_id]))

        print(f"[✓] 同社区事件对: {len(filtered_pairs)}")
        return filtered_pairs

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

    def check_causality_batch(
        self,
        pairs: List[Tuple[Entity, Entity]]
    ) -> Dict[Tuple[str, str], Dict[str, Any]]:
        """
        并发检查事件对的因果关系（软超时 + 失败收集 + 末尾多轮重试）
        - 依赖 self.event_cards（在主流程预生成），缺失项会兜底即时补建一次
        - 首轮：完成即收集；超时/异常 -> 先占位 + 记入重试队列
        - 末尾：仅对失败项做 N 轮重试；成功即覆盖旧结果
        - 返回：{(src_id, tgt_id): result_dict}
        """

        PER_TASK_TIMEOUT = 1800  # 秒；首轮软超时
        MAX_RETRIES = 2          # 重试轮数，可调
        RETRY_BACKOFF = 2.0      # 指数回退底数：1, 2, 4...秒
        RETRY_TIMEOUT = 600      # 重试轮的单任务软超时（可比首轮更短/更长）

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
                "relation": relation,               # CAUSES | INDIRECT_CAUSES | PART_OF | NONE
                "reason": reason,
                "temporal_order": temporal_order,   # E1_before_E2 | E2_before_E1 | Overlap_or_Simultaneous | Unknown
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

        def _ensure_card(e: Entity, info_text: str) -> Dict[str, Any]:
            """优先读缓存；缺失时兜底构建一次并写回缓存（极少触发）"""
            if e.id in self.event_cards:
                return self.event_cards[e.id]
            out = self.graph_analyzer.generate_event_context(info_text, "")
            card = json.loads(correct_json_format(out))["event_card"]
            card = format_event_card(card)
            self.event_cards[e.id] = card
            return card

        def _process_pair(pair: Tuple[Entity, Entity]):
            src_event, tgt_event = pair
            pair_key = (src_event.id, tgt_event.id)
            try:
                # 1) 获取事件信息文本（含属性与关系）
                info_1 = self.neo4j_utils.get_entity_info(
                    src_event.id, entity_type="事件",
                    contain_properties=True, contain_relations=True
                )
                info_2 = self.neo4j_utils.get_entity_info(
                    tgt_event.id, entity_type="事件",
                    contain_properties=True, contain_relations=True
                )

                # 2) 共同邻居与上下文（此处主要拼接结构化文本 + 共同邻居摘要）
                related_context = info_1 + "\n" + info_2 + "\n" + _get_common_neighbor_info(src_event.id, tgt_event.id)

                # 3) 读取预生成的 event_card（缺失则兜底构建一次）
                src_event_card = _ensure_card(src_event, info_1)
                tgt_event_card = _ensure_card(tgt_event, info_2)

                # 4) LLM 判定
                result_json = self.graph_analyzer.check_event_causality(
                    src_event_card, tgt_event_card,
                    system_prompt=self.system_prompt_text,
                    related_context=related_context
                )

                # 5) 解析 JSON
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

                return pair_key, _make_result(
                    src_event, tgt_event,
                    relation=relation,
                    reason=reason,
                    temporal_order=temporal_order,
                    confidence=confidence,
                    raw_result=raw_str,
                    timeout=False
                )

            except Exception as e:
                # 失败降级：NONE + 错误信息（首轮会进入重试队列）
                return pair_key, _make_result(
                    src_event, tgt_event,
                    relation="NONE",
                    reason=f"检查过程出错: {e}",
                    temporal_order="Unknown",
                    confidence=0.0,
                    raw_result="",
                    timeout=True  # 统一当作需重试
                )

        def _run_batch(pairs_to_run: List[Tuple[Entity, Entity]], per_task_timeout: float,
                       allow_placeholders: bool, desc: str):
            """通用并发跑一批；返回 (batch_results, failed_keys)。"""
            results_batch: Dict[Tuple[str, str], Dict[str, Any]] = {}
            failed_keys: set = set()

            executor = ThreadPoolExecutor(max_workers=self.max_workers, thread_name_prefix="causal")
            try:
                fut_info: Dict[Any, Dict[str, Any]] = {}
                for pair in pairs_to_run:
                    f = executor.submit(_process_pair, pair)
                    fut_info[f] = {"start": time.monotonic(), "pair": pair}

                pbar = tqdm(total=len(fut_info), desc=desc, ncols=100)
                pending = set(fut_info.keys())

                while pending:
                    done, pending = wait(pending, timeout=0.25, return_when=FIRST_COMPLETED)

                    # 收集已完成
                    for f in done:
                        pair = fut_info[f]["pair"]
                        src_event, tgt_event = pair
                        key = (src_event.id, tgt_event.id)
                        try:
                            k2, res = f.result()  # 已完成，不阻塞
                            results_batch[k2] = res
                            # 标记失败键（如果是超时标记 true 也进入失败集）
                            if res.get("causality_timeout"):
                                failed_keys.add(k2)
                        except Exception as e:
                            # 收集阶段异常
                            res = _make_result(
                                src_event, tgt_event,
                                relation="NONE",
                                reason=f"结果收集出错: {e}",
                                temporal_order="Unknown",
                                confidence=0.0,
                                raw_result="",
                                timeout=True
                            )
                            results_batch[key] = res
                            failed_keys.add(key)
                        pbar.update(1)
                        fut_info.pop(f, None)

                    # 检查软超时
                    now = time.monotonic()
                    to_forget = []
                    for f in list(pending):
                        start = fut_info[f]["start"]
                        if now - start >= per_task_timeout:
                            pair = fut_info[f]["pair"]
                            src_event, tgt_event = pair
                            key = (src_event.id, tgt_event.id)
                            # 试图取消：未启动的会成功；运行中返回 False，但我们不再等待
                            f.cancel()
                            if allow_placeholders:
                                res = _make_result(
                                    src_event, tgt_event,
                                    relation="NONE",
                                    reason="软超时，占位返回",
                                    temporal_order="Unknown",
                                    confidence=0.0,
                                    raw_result="",
                                    timeout=True
                                )
                                results_batch[key] = res
                            failed_keys.add(key)
                            pbar.update(1)
                            to_forget.append(f)

                    for f in to_forget:
                        pending.remove(f)
                        fut_info.pop(f, None)

                pbar.close()
            finally:
                executor.shutdown(wait=False, cancel_futures=True)

            return results_batch, failed_keys

        # ------- 主流程 -------
        print(f"🔍 开始并发检查 {len(pairs)} 对事件的因果关系...")

        # 建立 key->pair 映射，便于重试
        key2pair: Dict[Tuple[str, str], Tuple[Entity, Entity]] = {
            (src.id, tgt.id): (src, tgt) for (src, tgt) in pairs
        }

        # 首轮：允许占位（保证“完成即收集”体验）
        head_results, failed_keys = _run_batch(
            pairs_to_run=pairs,
            per_task_timeout=PER_TASK_TIMEOUT,
            allow_placeholders=True,
            desc="并发检查因果关系（首轮）"
        )

        results: Dict[Tuple[str, str], Dict[str, Any]] = dict(head_results)

        # 需要重试的 key：timeout 或异常
        def _needs_retry(key: Tuple[str, str], res: Dict[str, Any]) -> bool:
            # 规则：超时 或 reason 含“出错”
            if res.get("causality_timeout"):
                return True
            reason = (res.get("reason") or "").strip()
            return ("出错" in reason)

        needs_retry = [k for k in failed_keys if k in results and _needs_retry(k, results[k])]
        print(f"⏩ 首轮后准备重试：{len(needs_retry)} / {len(pairs)}")

        # 多轮重试：不再写占位，成功才覆盖旧值；仍然使用软超时，但通常更短
        for attempt in range(1, MAX_RETRIES + 1):
            if not needs_retry:
                break
            # 指数回退（轻微休眠，避免瞬时抖动）
            backoff = (RETRY_BACKOFF ** (attempt - 1))
            try:
                time.sleep(min(backoff, 5.0))
            except Exception:
                pass

            pairs_for_retry = [key2pair[k] for k in needs_retry if k in key2pair]
            batch_desc = f"并发检查因果关系（重试第 {attempt}/{MAX_RETRIES} 轮）"
            retry_results, retry_failed = _run_batch(
                pairs_to_run=pairs_for_retry,
                per_task_timeout=RETRY_TIMEOUT,
                allow_placeholders=False,  # 重试不写占位，只收成功与失败集合
                desc=batch_desc
            )

            # 覆盖成功项（那些没有 timeout 标记、且无“出错”字样的）
            improved = 0
            for k, v in retry_results.items():
                if not _needs_retry(k, v):
                    results[k] = v
                    improved += 1
                else:
                    # 仍失败的保持旧值（旧值可能是占位）
                    pass

            print(f"🔁 重试第 {attempt} 轮：成功覆盖 {improved} 项，仍需重试 {len(retry_failed)} 项")

            # 下一轮只对仍失败的进行
            needs_retry = [k for k in retry_failed]

        print(f"✅ 因果关系并发检查完成（成功 {len(results) - len(needs_retry)} / {len(pairs)}，仍失败 {len(needs_retry)}）")

        # 最终若仍有失败项，可在这里统一打上“final_fallback”标记（可选）
        for k in needs_retry:
            if k in results:
                r = results[k]
                r["final_fallback"] = True
                r["retries"] = MAX_RETRIES

        return results

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

    def initialize(self):
        # 1. 创建子图和计算社区划分
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

    def filter_pair_by_distance_and_similarity(self, pairs):
        filtered_pairs = []
        for pair in tqdm(pairs, desc="筛选节点对"):
            src_id, tgt_id = pair[0].id, pair[1].id
            reachable = self.neo4j_utils.check_nodes_reachable(
                src_id, tgt_id,
                excluded_rels=[self.meta["contains_pred"], "EVENT_CAUSES", "EVENT_INDIRECT_CAUSES", "EVENT_PART_OF"],
                max_depth=self.max_depth
            )
            if reachable:  # 如果节点间距离 <= max_depth，保留。
                filtered_pairs.append(pair)
            else:
                score = self.neo4j_utils.compute_semantic_similarity(src_id, tgt_id)
                if score >= 0.5:  # 语义相似度阈值与注释一致
                    filtered_pairs.append(pair)
        return filtered_pairs

    def build_event_causality_graph(
        self,
        limit_events: Optional[int] = None
    ) -> None:
        """
        完整的事件因果图构建流程

        Args:
            limit_events: 限制处理的事件数量（用于测试）

        Returns:
            构建完成的Neo4j有向图
        """
        print("🚀 开始完整的事件因果图构建流程...")

        # 2. 构建事件列表
        print("\n🔍 构建事件列表...")
        event_list = self.build_event_list()

        # 3. 限制事件数量（用于测试）
        if limit_events and limit_events < len(event_list):
            event_list = event_list[:limit_events]
            print(f"⚠️ 限制处理事件数量为: {limit_events}")

        # === ✅ 新增：并发预生成所有事件卡片（缓存+落盘） ===
        print("\n🧩 并发预生成事件卡片...")
        self.precompute_event_cards(event_list)

        # 4. 过滤事件对
        print("\n🔍 过滤事件对...")
        filtered_pairs = self.filter_event_pairs_by_community(event_list, max_depth=self.max_depth)
        filtered_pairs = self.filter_pair_by_distance_and_similarity(filtered_pairs)
        filtered_pairs = self.sort_event_pairs_by_section_order(filtered_pairs)
        print("     最终候选事件对数量： ", len(filtered_pairs))

        # 5. 检查因果关系（将直接读取 self.event_cards）
        print("\n🔍 检查因果关系...")
        causality_results = self.check_causality_batch(filtered_pairs)

        base = self.config.storage.knowledge_graph_path
        with open(os.path.join(base, "event_casality_results.pkl"), "wb") as f:
            pickle.dump(causality_results, f)

        with open(os.path.join(base, "event_cards.json"), "w", encoding="utf-8") as f:
            json.dump(self.event_cards, f, ensure_ascii=False, indent=2)

        # 6. 写回 EVENT 关系
        print("\n🔗 写回Event间关系...")
        self.write_event_cause_edges(causality_results)
        self.neo4j_utils.create_event_causality_graph("event_causality_graph", force_refresh=True)

    def detect_flattened_causal_patterns(self, edges: List[Dict]) -> List[Dict]:
        """
        从边集中发现类似 A→B, A→C, A→D 且存在 B→D 的冗余结构，用于后续因果链精炼

        Returns:
            List of {
                "source": A,
                "targets": [B, C, D],
                "internal_links": [(B, D), (C, D)]
            }
        """
        # 构建邻接表和反向边集合
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
                continue  # 至少两个指向才可能构成该模式

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
        """
        从 flattened patterns 中剔除 weight 和 confidence 都偏低的边
        """
        cleaned_patterns = []
        for pat in patterns:
            src = pat["source"]
            targets = pat["targets"]
            internals = pat["internal_links"]

            # 过滤 source → target 边
            new_targets = []
            for t in targets:
                info = edge_map.get((src, t))
                confidence = info.get("confidence", 0) if info else 0
                if not info:
                    continue
                if confidence < conf_threshold:
                    new_targets.append(t)

            # 过滤 internal 边
            new_internals = []
            for u, v in internals:
                info = edge_map.get((u, v))
                confidence = info.get("confidence", 0) if info else 0
                if not info:
                    continue
                if not confidence < conf_threshold:
                    new_internals.append((u, v))

            # 保留结构
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
        """
        比较两组 pattern 结构，收集被删除导致结构变化的边

        Returns:
            被标记为删除候选的边集合（sid, tid）
        """
        # 抽取原始结构中的全部边
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
        return list(removed_edges)  # 原逻辑返回 list；保持一致

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

            if (src_mid_sim > src_tgt_sim and mid_tgt_sim > src_tgt_sim) or (src_mid_conf > src_tgt_conf and mid_tgt_conf > src_tgt_conf):
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

            # 1) 可能返回 None：统一转成字符串
            background = self.neo4j_utils.get_entity_info(event_id, "事件", True, True)
            background = _safe_str(background)

            # 2) event_properties 可能为 None / str(JSON) / dict
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

            # 仅保留非空字符串属性
            non_empty_props = {k: v for k, v in event_props.items() if isinstance(v, str) and v.strip()}

            if non_empty_props:
                background += "\n事件的属性如下：\n"
                for k, v in non_empty_props.items():
                    background += f"- {k}：{v}\n"

            if i + 1 != len(event_details):
                background += "\n"

            full_event_details += background

        # 关系细节字符串
        full_relation_details = "它们之间已经存在的因果关系有：\n"
        relation_details = pattern_detail["details"]
        for i, relation_info in enumerate(relation_details):
            src, tgt = relation_info["edge"]
            # 3) get_relation_summary 也做一次兜底转字符串
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
        """
        loop_count = 0
        removed_edges: List[Tuple[str, str]] = []  # 防止未定义引用
        while True:
            print(f"\n===== [第 {loop_count + 1} 轮优化] =====")

            # === 获取连通体（优先 SCC，再选 WCC） ===
            scc_components = self.neo4j_utils.fetch_scc_components("event_causality_graph", 2)
            wcc_components = []
            if self.check_weakly_connected_components:
                wcc_components = self.neo4j_utils.fetch_wcc_components("event_causality_graph", self.min_component_size)

            connected_components = scc_components + wcc_components
            print(f"📌 当前连通体数量：SCC={len(scc_components)}，WCC={len(wcc_components)}")

            # === 构造所有 triangle 和边信息 ===
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

            # === ✅ 提前退出条件 ===
            if loop_count >= 1:
                if len(scc_components) == 0 and len(set(removed_edges)) == 0:
                    print("✅ 图结构已无强连通体，且无待判定三元结构，任务终止。")
                    break
            elif loop_count >= self.max_iteration:
                break

            # === 并发处理三元结构 ===
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
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = [executor.submit(process_triangle, tri) for tri in all_triangles]
                for f in tqdm(as_completed(futures), total=len(futures), desc="LLM判断"):
                    res = f.result()
                    if res:
                        removed_edges.append(res)

            print(f"❌ 本轮待定移除边数量：{len(set(removed_edges))}")

            # === 删除边 ===
            for edge in removed_edges:
                self.neo4j_utils.delete_relation_by_ids(edge[0], edge[1], "EVENT_CAUSES")

            # === 刷新 GDS 图 ===
            self.neo4j_utils.create_event_causality_graph("event_causality_graph", min_confidence=0, force_refresh=True)
            loop_count += 1

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
            # context += f"事件{i+1}：{event}\n" + self.neo4j_utils.get_entity_info(event, "事件", False, True) + "\n"
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

    def generate_plot_relations(self):
        """
        基于候选情节对，判定并写入情节间关系。
        关系集（最终版）：
        - 有向：PLOT_PREREQUISITE_FOR, PLOT_ADVANCES, PLOT_BLOCKS, PLOT_RESOLVES
        - 无向：PLOT_CONFLICTS_WITH, PLOT_PARALLELS
        兼容旧类型：PLOT_CONTRIBUTES_TO / PLOT_SETS_UP → 统一映射为 PLOT_ADVANCES
        """
        # 预处理：向量、GDS 图与嵌入
        self.neo4j_utils.process_all_embeddings(entity_types=["Plot", self.meta["section_label"]])
        self.neo4j_utils.create_event_plot_graph()
        self.neo4j_utils.run_node2vec()

        # 召回候选情节对
        all_plot_pairs = self.neo4j_utils.get_plot_pairs(threshold=0)
        print("[✓] 待判定情节关系数量：", len(all_plot_pairs))

        # 关系类型与向后兼容映射
        LEGACY_MAP = {
            "PLOT_CONTRIBUTES_TO": "PLOT_ADVANCES",
            "PLOT_SETS_UP": "PLOT_ADVANCES"
        }
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

        edges_to_add = []

        def _make_edge(src_id, tgt_id, rtype, confidence, reason):
            return {
                "src": src_id,
                "tgt": tgt_id,
                "relation_type": rtype,
                "confidence": float(confidence) if confidence is not None else 0.0,
                "reason": reason or ""
            }

        def _parse_direction_to_edge(pair, direction_str, rtype, confidence, reason):
            """
            将 A/B 方向映射为真实 src/tgt 边；返回 [edge] 或 []。
            direction_str: "A->B" / "B->A"
            """
            if direction_str == "A->B":
                return [_make_edge(pair["src"], pair["tgt"], rtype, confidence, reason)]
            elif direction_str == "B->A":
                return [_make_edge(pair["tgt"], pair["src"], rtype, confidence, reason)]
            else:
                # 无效或缺失方向，跳过该对（仅对有向关系生效）
                print(f"[!] 跳过：有向关系缺少有效方向 direction={direction_str} pair={pair}")
                return []

        def process_pair(pair):
            try:
                plot_A_info = self.neo4j_utils.get_entity_info(pair["src"], "情节", contain_properties=True, contain_relations=True)
                plot_B_info = self.neo4j_utils.get_entity_info(pair["tgt"], "情节", contain_properties=True, contain_relations=True)

                # 调用关系判定（LLM/规则）
                result = self.graph_analyzer.extract_plot_relation(plot_A_info, plot_B_info, self.system_prompt_text)

                # 尝试修正/解析 JSON
                try:
                    result = json.loads(correct_json_format(result))
                except Exception:
                    # 若直接是 dict 则保留，否则抛出
                    if isinstance(result, dict):
                        pass
                    else:
                        raise

                # 读取字段
                rtype = result.get("relation_type")
                direction = result.get("direction", None)  # 有向时应为 "A->B" / "B->A"，无向或 None 用 null
                confidence = result.get("confidence", 0.0)
                reason = result.get("reason", "")

                # 兼容旧枚举
                if rtype in LEGACY_MAP:
                    rtype = LEGACY_MAP[rtype]

                # 过滤无效类型
                if rtype not in VALID_TYPES:
                    print(f"[!] 未知 relation_type={rtype}，跳过 pair={pair}")
                    return []

                # None 或无关系
                if rtype in {"None", None}:
                    return []

                pair_edges = []

                # 有向关系
                if rtype in DIRECTED:
                    pair_edges.extend(_parse_direction_to_edge(pair, direction, rtype, confidence, reason))

                # 无向关系：写双向边
                elif rtype in UNDIRECTED:
                    pair_edges.append(_make_edge(pair["src"], pair["tgt"], rtype, confidence, reason))
                    pair_edges.append(_make_edge(pair["tgt"], pair["src"], rtype, confidence, reason))

                return pair_edges

            except Exception as e:
                print(f"[⚠] 处理情节对 {pair} 出错: {e}")
                return []

        # 并发处理
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from tqdm import tqdm

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(process_pair, pair) for pair in all_plot_pairs]
            for future in tqdm(as_completed(futures), total=len(futures), desc="抽取情节关系"):
                try:
                    res = future.result()
                    if res:
                        edges_to_add.extend(res)
                except Exception as e:
                    print(f"[⚠] future 结果处理出错: {e}")

        # 批量写入 Neo4j
        if edges_to_add:
            self.neo4j_utils.create_plot_relations(edges_to_add)
            print(f"[✓] 已创建情节关系 {len(edges_to_add)} 条")
        else:
            print("[!] 没有生成任何情节关系")


    def build_event_plot_graph(self):
        # 清空旧的 Plot 图与关系（已适配新六种 Plot 关系 + HAS_EVENT）
        self.neo4j_utils.reset_event_plot_graph()

        base = self.config.storage.knowledge_graph_path
        print("事件卡片信息...")
        with open(os.path.join(base, "event_cards.json"), "r", encoding="utf-8") as f:
            self.event_cards = json.load(f)

        # 建议：这里的 get_all_event_chains 内部请确认已使用三类事件关系 + confidence 过滤
        all_chains = self.get_all_event_chains(min_confidence=0.0)

        print("[✓] 当前事件链总数：", len(all_chains))
        filtered_chains = get_frequent_subchains(all_chains, 2, 1)
        filtered_chains = remove_subset_paths(filtered_chains)
        filtered_chains = remove_similar_paths(filtered_chains, 0.7)
        print("[✓] 过滤后事件链总数：", len(filtered_chains))

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
                # 组织上下文
                event_chain_info = self.prepare_chain_context(chain)

                # 取相关文档片段（容错空值）
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

                # 生成情节判定
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
                    # 兜底标题：首尾事件名或ID
                    title = f"情节链：{chain[0]}→{chain[-1]}"

                plot_info["id"] = _stable_plot_id(title, chain)
                plot_info["event_ids"] = chain
                plot_info["reason"] = result.get("reason", "")

                # 写入 Neo4j（已在 Neo4jUtils 内适配新关系/字段）
                self.neo4j_utils.write_plot_to_neo4j(plot_data=plot_info)
                return True

            except Exception as e:
                print(f"[!] 处理事件链 {chain} 时出错: {e}")
                return False

        success_count = 0
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(process_chain, chain) for chain in filtered_chains]
            for future in tqdm(as_completed(futures), total=len(futures), desc="并发生成情节图谱"):
                try:
                    if future.result():
                        success_count += 1
                except Exception as e:
                    print(f"[!] 子任务异常：{e}")

        print(f"[✓] 成功生成情节数量：{success_count}/{len(filtered_chains)}")
        return


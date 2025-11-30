# -*- coding: utf-8 -*-
import os
import json
import re
import sqlite3
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

import pandas as pd
from tqdm import tqdm
from core.utils.prompt_loader import PromptLoader
from core.agent.character_status_extraction_agent import CharacterStatusExtractionAgent

from core import KAGConfig
from core.builder.graph_builder import DOC_TYPE_META
from core.builder.manager.supplementary_manager import SupplementaryExtractor
from core.model_providers.openai_llm import OpenAILLM
from core.storage.graph_store import GraphStore
from core.storage.vector_store import VectorStore
from core.utils.format import correct_json_format
from core.utils.function_manager import run_with_soft_timeout_and_retries
from core.utils.neo4j_utils import Neo4jUtils
from core.utils.render import generate_html
from core.functions.tool_calls.sqldb_tools import Search_By_Scene

DECISION_CONF_MAP = {
    "low": 0,
    "medium": 1,
    "high": 2,
}


# ==============================================================================
# 通用工具函数
# ==============================================================================

def normalize_bool(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    if isinstance(x, str):
        return x.strip().lower() in {"true", "yes", "1", "是", "连续", "接得上"}
    return False


def clean_cmp_info(text: str) -> str:
    """
    清理 CMP 文本：
    1. 去掉 '场次「...」 的相关记录：' 这一整段前缀
    2. 删除每条记录里的：
       - '补充信息：...'
       - 'chunk_id：...'
       - '场次：...'
       - '场次名：...'
    """
    if not isinstance(text, str):
        return ""

    s = text

    # 1) 去掉场次前缀
    s = re.sub(r'^场次「.*?」 的相关记录：\n?', "", s, flags=re.S)

    # 2) 删掉补充信息 / chunk_id / 场次 / 场次名 这些尾部字段
    patterns = [
        r'；?补充信息：[^；\n]*',
        r'；?chunk_id：[^\n；]*',
        r'；?场次名：[^\n；]*',
        r'；?场次：[^\n；]*',
    ]
    for p in patterns:
        s = re.sub(p, "", s)

    # 3) 收拾一下多余分号和空白
    s = re.sub(r'；{2,}', "；", s)
    s = re.sub(r'；\s*\n', "\n", s)
    s = re.sub(r'；\s*$', "", s)

    return s.strip()


def extract_minimal_cmp_list(cmp_text: str) -> List[str]:
    """
    输入: 原始 cmp_info 文本（包含多条道具记录）
    输出: 去重后的精简 CMP 列表（只保留 名称 / 类别）
    """
    if not isinstance(cmp_text, str):
        return []

    s = clean_cmp_info(cmp_text)

    # 形如：- 名称：抗荷服；类别：wardrobe；...
    pattern = r'-\s*名称：([^；\n]+).*?类别：([^；\n]+)'
    matches = re.findall(pattern, s, flags=re.S)

    results = set()
    clean_list = []

    for name, cls in matches:
        name = name.strip()
        cls = cls.strip()
        key = (name, cls)
        if key in results:
            continue
        results.add(key)
        clean_list.append(f"- 名称：{name}；类别：{cls}")

    return clean_list


def build_scenes_dict(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    将包含 scene_id, title, summary, cmp_info（以及可选 version）的 DataFrame
    转换为 scenes 字典。
    """
    scenes: Dict[str, Dict[str, Any]] = {}

    for _, row in df.iterrows():
        sid = row["scene_id"]
        scenes[sid] = {
            "scene_id": sid,
            "scene_title": row.get("title", "") or "",
            "summary": row.get("summary", "") or "",
            "cmp_info": row.get("cmp_info", "") or "",
            "version": row.get("version", "") or "",   # ⭐ 新增，可为空
        }

    return scenes


def build_scene_metadata_blocks(chain: List[str], chains_df: pd.DataFrame) -> str:
    """
    根据 chain 中的 scene_id，在 chains_df 中取出元数据并整理成给 LLM 的文本块。
    """
    blocks = []

    for scene_id in chain:
        row = chains_df.loc[chains_df["scene_id"] == scene_id].head(1)
        if row.empty:
            blocks.append(f"- Scene ID: {scene_id}\n  [WARNING] Not found.\n")
            continue

        r = row.iloc[0]
        title = r.get("title", "")
        summary = r.get("summary", "")
        cmp_raw = r.get("cmp_info", "")
        cmp_clean = extract_minimal_cmp_list(cmp_raw)

        block = (
            f"- Scene ID: {scene_id}\n"
            f"  Title: {title}\n"
            f"  Summary: {summary}\n"
            f"  CMP:\n    " + "\n    ".join(cmp_clean)
        )
        blocks.append(block)

    return "\n\n".join(blocks)


def find_maximal_cliques(graph: Dict[str, set]) -> List[List[str]]:
    """
    Bron–Kerbosch 算法：找出所有最大团（maximal cliques）。
    graph: dict[str, set[str]]
    返回: list[list[scene_id]]
    """
    cliques: List[List[str]] = []

    def bronk(R: set, P: set, X: set):
        if not P and not X:
            if len(R) >= 2:  # 至少 2 场才算有意义
                cliques.append(sorted(R))
            return

        # pivot 选择：稍微剪枝
        pivot = None
        if P or X:
            pivot = max(P.union(X), key=lambda v: len(P & graph[v]))

        for v in list(P - (graph[pivot] if pivot else set())):
            bronk(
                R | {v},
                P & graph[v],
                X & graph[v],
            )
            P.remove(v)
            X.add(v)

    all_nodes = set(graph.keys())
    bronk(set(), all_nodes, set())
    return cliques


# ==============================================================================
# 主类：SupplementaryBuilder
# ==============================================================================

class SupplementaryBuilder:
    """
    批量抽取全图谱 Scene 的角色状态 & 接戏判定。
    """

    EXCLUDE_PROP_KEYS = {"order", "timelines", "partition"}

    def __init__(self, config: KAGConfig):
        self.config = config

        # 依赖初始化
        self.llm = OpenAILLM(config)
        self.graph_store = GraphStore(config)
        self.vector_store = VectorStore(config, "documents")

        doc_type = config.knowledge_graph_builder.doc_type
        self.meta = DOC_TYPE_META[doc_type]
        self.neo4j_utils = Neo4jUtils(self.graph_store.driver, doc_type)
        self.neo4j_utils.load_embedding_model(config.graph_embedding)

        self.information_extractor = SupplementaryExtractor(config, self.llm)

        # 接戏判定：基于场次的结构化信息
        db_path = os.path.join(self.config.storage.sql_database_path, "CMP.db")
        if not os.path.exists(db_path):
            print(f"⚠️ CMP 数据库不存在：{db_path}，接戏判定中的 cmp_info 将为空字符串。")
            self.cmp_tool: Optional[Search_By_Scene] = None
        else:
            self.cmp_tool = Search_By_Scene(db_path)
            
        prompt_dir = self.config.knowledge_graph_builder.prompt_dir
        prompt_loader = PromptLoader(prompt_dir)
        self.character_status_agent = CharacterStatusExtractionAgent(
            config=config,
            llm=self.llm,
            prompt_loader=prompt_loader,
        )

    def _use_version_partition(self) -> bool:
        """
        是否启用按 version 分区的逻辑：
        - 如果 config.knowledge_graph.versions 为空 或 只有 ["default"]，
          认为是旧行为：不按 version 过滤，所有 Scene 都可互相接戏判断。
        - 否则（例如 ["Part_1", "Part_2"]），只在 version 一致的场次间做接戏判定。
        """
        kg_cfg = getattr(self.config, "knowledge_graph", None)
        versions = getattr(kg_cfg, "versions", None) if kg_cfg is not None else None

        if not versions:
            # 未配置版本，保持旧逻辑
            return False

        if isinstance(versions, (list, tuple)) and len(versions) == 1 and versions[0] == "default":
            # 只有一个 default，认为是单版本旧逻辑
            return False

        return True

    # ----------------------------------------------------------------------
    # 一、角色状态抽取
    # ----------------------------------------------------------------------
    def extract_character_status_for_all_scenes(
        self,
        *,
        per_task_timeout: float = 600.0,
        retries: int = 3,
        retry_backoff: float = 2.0,
        max_workers: int = 16,
        save: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        跑全量场次抽取并返回 list[dict]。默认保存到 character_status.json。
        """
        scenes = self.neo4j_utils.fetch_all_nodes(node_types=["Scene"]) or []
        if not scenes:
            print("未在图谱中发现任何 Scene 节点。")
            if save:
                self._save_results([], warn="无 Scene")
            return []

        results_map, still_failed = run_with_soft_timeout_and_retries(
            items=scenes,
            work_fn=self._process_one_scene_status,
            key_fn=lambda s: self._get_scene_id_from_node(s),
            desc_label="抽取角色状态",
            per_task_timeout=per_task_timeout,
            retries=retries,
            retry_backoff=retry_backoff,
            max_workers=max_workers,
        )

        out_list: List[Dict[str, Any]] = []
        for _, val in results_map.items():
            if isinstance(val, dict):
                out_list.append(val)

        if save:
            self._save_results(out_list)

        if still_failed:
            print(
                f"⚠️ 仍有 {len(still_failed)} 个场次在重试后失败："
                f"{sorted(list(still_failed))[:10]}{' ...' if len(still_failed) > 10 else ''}"
            )

        return out_list

    def _process_one_scene_status(self, scene_node: Dict[str, Any]) -> Dict[str, Any]:
        """
        返回当前场次的抽取结果 dict：
        - 若无角色列表 => 返回 {'scene_id': ..., 'characters': [], ...(合并属性)}
        - 否则调用带反思机制的 Agent 抽取，并合并属性
        """
        props = self._parse_scene_properties(scene_node)
        scene_id = self._get_scene_id(scene_node, props)
        if not scene_id:
            raise RuntimeError("无法确定 scene_id（缺少节点 id 与 props.scene_id）")

        # 角色列表
        character_list = self._get_character_list(scene_id)
        if not character_list:
            # 无角色：仅保留场次元信息 + 空角色列表
            result: Dict[str, Any] = {
                "scene_id": scene_id,
                "characters": [],
            }
            self._merge_scene_properties(result, props)
            return result

        # 场景文本（即使无角色也提前拉出来，方便排错）
        scene_contents = self._get_scene_contents(scene_node, scene_id)

        # ---- 使用 CharacterStatusExtractionAgent，内部会自动做反思循环 ----
        try:
            agent_result = self.character_status_agent.run(
                scene_contents=scene_contents,
                character_list=character_list,
            )
            
        except Exception as e:
            # 兜底：Agent 出错时，不让整个 pipeline 崩掉
            print(f"⚠️ 场景 {scene_id} 角色状态抽取 Agent 失败：{e}")
            agent_result = {}

        char_list = agent_result.get("results", []) if isinstance(agent_result, dict) else []
        if not isinstance(char_list, list):
            char_list = []

        # 可以顺带把 score / attempts / feedbacks 存进去，方便后续分析
        result: Dict[str, Any] = {
            "scene_id": scene_id,
            "characters": char_list,
        }

        score = agent_result.get("score") if isinstance(agent_result, dict) else None
        attempts = agent_result.get("attempts") if isinstance(agent_result, dict) else None
        feedbacks = agent_result.get("feedbacks") if isinstance(agent_result, dict) else None

        if score is not None:
            result["cs_score"] = float(score)
        if attempts is not None:
            result["cs_attempts"] = int(attempts)
        if feedbacks:
            result["cs_feedbacks"] = feedbacks

        # 合并场景属性（不覆盖已有 scene_id / characters 等）
        self._merge_scene_properties(result, props)
        return result

    # ----------------------------------------------------------------------
    # 二、接戏判定（Scene 对）
    # ----------------------------------------------------------------------
    def check_scene_continuity(
        self,
        *,
        per_task_timeout: float = 1800.0,
        retries: int = 3,
        retry_backoff: float = 2.0,
        max_workers: int = 16,
        order_threshold: int = 30,
        save: bool = True,
        only_true: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        批量对所有存在共同角色的 Scene 对进行接戏判定。

        输出字段示意（单条）：
        - scene_id_1, scene_name_1, summary_1, cmp_info_1
        - scene_id_2, scene_name_2, summary_2, cmp_info_2
        - common_neighbor_info   # 共同角色的摘要信息（多行字符串）
        - neighbor_ids           # [id1, id2, ...]
        - is_continuity          # LLM 判定结果（bool 或字符串）
        - reason                 # 判定理由（来自 LLM）
        """
        pairs = self._get_scene_pairs_with_common_neighbors()
        if not pairs:
            print("未找到任何存在共同角色的 Scene 对。")
            if save:
                self._save_scene_continuity_results([], warn="无 Scene Pair")
            return []

        # 先按场次顺序距离过滤一遍
        filtered_pairs: List[Dict[str, Any]] = []
        for p in pairs:
            try:
                if self._filter_pair_by_order(p, threshold=order_threshold):
                    filtered_pairs.append(p)
            except Exception as e:
                print(f"⚠️ 过滤场次对时出错，跳过该 pair: {p}，错误：{e}")

        if not filtered_pairs:
            print("存在共同角色的场次对中，没有满足 order_threshold 限制的 pair。")
            if save:
                self._save_scene_continuity_results([], warn="无符合条件的 Scene Pair")
            return []

        results_map, still_failed = run_with_soft_timeout_and_retries(
            items=filtered_pairs,
            work_fn=self._process_one_scene_pair,
            key_fn=lambda row: f"{row['scene1']}__{row['scene2']}",
            desc_label="接戏判定",
            per_task_timeout=per_task_timeout,
            retries=retries,
            retry_backoff=retry_backoff,
            max_workers=max_workers,
        )

        out_list: List[Dict[str, Any]] = []
        for _, val in results_map.items():
            if isinstance(val, dict):
                out_list.append(val)

        # 只保留判定为 True 的结果
        if only_true:
            kept: List[Dict[str, Any]] = []
            for item in out_list:
                v = item.get("is_continuity")
                keep = normalize_bool(v)
                if keep:
                    kept.append(item)

            out_list = kept
            print(f"仅保留接戏为 True 的结果，共 {len(out_list)} 条。")

        if save:
            out_path = os.path.join(
                self.config.storage.knowledge_graph_path, "scene_continuity.json"
            )
            with open(out_path, "w", encoding="utf-8") as fw:
                json.dump(out_list, fw, ensure_ascii=False, indent=2)
            print(f"✅ 已保存 {len(out_list)} 条接戏判定结果 -> {out_path}")

        if still_failed:
            print(
                f"⚠️ 仍有 {len(still_failed)} 个场次对在重试后失败："
                f"{sorted(list(still_failed))[:10]}{' ...' if len(still_failed) > 10 else ''}"
            )

        return out_list

    def generate_continuity_chains(self, threshold: int = 3) -> None:
        """
        基于 scene_continuity.json 中接戏为 True 的边，使用最大团算法生成接戏链：
        - 长度 >= threshold 的大链：走 LLM 多轮评估 + split/drop/keep
        - 长度在 [2, threshold) 的小链：原样保留（不经 LLM）
        - 所有长度为 1 的链在最终输出前剔除
        """
        continuity_path = os.path.join(
            self.config.storage.knowledge_graph_path, "scene_continuity.json"
        )
        if not os.path.exists(continuity_path):
            raise FileNotFoundError(f"未找到接戏判定结果文件：{continuity_path}")

        with open(continuity_path, "r", encoding="utf-8") as f:
            continuity_results = json.load(f) or []

        if not continuity_results:
            print("scene_continuity.json 为空，无法生成接戏链。")
            return

        df = pd.DataFrame(continuity_results)
        df["is_continuity"] = df["is_continuity"].apply(normalize_bool)
        df_true = df[df["is_continuity"]].copy()

        # ---------- 构建接戏边 & 场次元数据 ----------
        # ---------- 构建接戏边 & 场次元数据 ----------
        edges = set()
        scene_meta: Dict[str, Dict[str, Any]] = {}

        def _update_meta(
            sid: str,
            title: str,
            summary: str,
            cmp_info: str,
            version: str,
        ) -> None:
            if sid not in scene_meta:
                scene_meta[sid] = {
                    "scene_id": sid,
                    "title": title,
                    "summary": summary,
                    "cmp_info": cmp_info,
                    "version": version or "",
                }


        for _, r in df_true.iterrows():
            s1, s2 = r["scene_id_1"], r["scene_id_2"]
            if s1 == s2:
                continue
            edges.add(tuple(sorted((s1, s2))))

            v = r.get("version", "")  # ⭐ 这条判定记录的 version

            _update_meta(
                s1,
                r.get("scene_title_1", ""),
                r.get("summary_1", ""),
                r.get("cmp_info_1", ""),
                v,
            )
            _update_meta(
                s2,
                r.get("scene_title_2", ""),
                r.get("summary_2", ""),
                r.get("cmp_info_2", ""),
                v,
            )


        print(f"接戏边数: {len(edges)}")
        print(f"涉及场次数: {len(scene_meta)}")

        # 邻接表：scene_id -> set(neighbor_scene_id)
        graph: Dict[str, set] = defaultdict(set)
        for a, b in edges:
            graph[a].add(b)
            graph[b].add(a)

        if not graph:
            print("接戏图为空，无法生成链。")
            return

        # 度分布
        degrees = {sid: len(neis) for sid, neis in graph.items()}
        print("最大度:", max(degrees.values()))
        print("平均度:", sum(degrees.values()) / len(degrees))

        # 最大团 -> 原始接戏链（含大链 + 小链）
        cliques = find_maximal_cliques(graph)
        print(f"找到最大团数量: {len(cliques)}")

        # 丢掉长度 < 2 的团（单点没意义）
        cliques = [c for c in cliques if len(c) >= 2]

        # 构建一个 DataFrame，方便给 LLM 拼 metadata（只需要场次级别信息即可）
        chains_records: List[Dict[str, Any]] = []
        for chain_idx, chain in enumerate(cliques, start=1):
            for order_in_chain, scene_id in enumerate(chain, start=1):
                meta = scene_meta.get(scene_id, {})
                chains_records.append(
                    {
                        "chain_id": chain_idx,
                        "order_in_chain": order_in_chain,
                        "scene_id": scene_id,
                        "title": meta.get("title", ""),
                        "summary": meta.get("summary", ""),
                        "cmp_info": meta.get("cmp_info", ""),
                        "version": meta.get("version", ""),  # ⭐ 新增
                    }
                )


        chains_df = pd.DataFrame(chains_records)

        # 根据长度区分大链 / 小链
        chain_sizes = chains_df.groupby("chain_id")["scene_id"].count()

        # 小链：长度在 [2, threshold) 的链，直接保留，不走 LLM
        small_chain_ids = chain_sizes[
            (chain_sizes >= 2) & (chain_sizes < threshold)
        ].index.tolist()
        small_chains_df = chains_df[chains_df["chain_id"].isin(small_chain_ids)]
        small_chains: List[List[str]] = (
            small_chains_df.groupby("chain_id")["scene_id"].apply(list).tolist()
        )

        # 大链：长度 >= threshold 的链，交给 LLM 进一步判断
        large_chain_ids = chain_sizes[chain_sizes >= threshold].index.tolist()
        large_chains_df = chains_df[chains_df["chain_id"].isin(large_chain_ids)]
        large_chains: List[List[str]] = (
            large_chains_df.groupby("chain_id")["scene_id"].apply(list).tolist()
        )

        print(f"大链数量(>= {threshold}): {len(large_chains)}")
        print(f"小链数量([2, {threshold})): {len(small_chains)}")

        # ====== 对大链做 LLM 多数投票评估 ======
        refined_large_chains: List[List[str]] = []
        for chain in tqdm(large_chains, desc="LLM 评估接戏大链"):
            scene_metadata_blocks = build_scene_metadata_blocks(chain, chains_df)
            final_eval = self.evaluate_chain_with_majority_vote(
                chain,
                scene_metadata_blocks,
            )
            decision = final_eval["keep_decision"]
            if decision == "split":
                suggested = final_eval.get("suggested_splits", [])
                # 兜底：确保 suggested_splits 全是 list
                for seg in suggested:
                    if isinstance(seg, list) and len(seg) > 0:
                        refined_large_chains.append(seg)
            elif decision == "keep":
                refined_large_chains.append(chain)
            # 若为 drop，则不加入

        # ====== 合并：大链结果 + 小链原样 ======
                # ====== 合并：大链结果 + 小链原样 ======
        all_chains = refined_large_chains + small_chains

        # 剔除长度为 1 的链（有些 split 可能给出单点）
        all_chains = [chain for chain in all_chains if len(chain) >= 2]

        # 先按集合去重一次，避免明显重复
        unique_chains: List[List[str]] = []
        seen = set()
        for chain in all_chains:
            key = tuple(sorted(chain))
            if key in seen:
                continue
            seen.add(key)
            unique_chains.append(chain)

        print(f"初步去重后接戏链数量: {len(unique_chains)}")

        # === 新增：合并有较大重叠的链（解决 A B C D E / A C D E F 分裂问题） ===
        merged_chains = self._merge_overlapping_chains(unique_chains, min_overlap=2)
        print(f"合并重叠后接戏链数量: {len(merged_chains)}")

        # 保存 continuity_chains.json
        kg_base = self.config.storage.knowledge_graph_path
        chains_out_path = os.path.join(kg_base, "continuity_chains.json")
        with open(chains_out_path, "w", encoding="utf-8") as fw:
            json.dump(merged_chains, fw, ensure_ascii=False, indent=2)
        print(f"✅ 已保存接戏链结果，共 {len(merged_chains)} 条 -> {chains_out_path}")


        # scene 信息字典（给可视化用，这里可以直接用 chains_df 的场景元数据）
        scenes = build_scenes_dict(chains_df)
        scenes_out_path = os.path.join(kg_base, "scene_information.json")
        with open(scenes_out_path, "w", encoding="utf-8") as fw:
            json.dump(scenes, fw, ensure_ascii=False, indent=2)

        # 生成 HTML 可视化（使用最终 unique_chains）
        html_path = os.path.join(kg_base, "接戏结果展示.html")
        generate_html(scenes, unique_chains, html_path)
        print(f"✅ 已生成接戏结果展示 HTML -> {html_path}")

    def _merge_overlapping_chains(
        self,
        chains: List[List[str]],
        min_overlap: int = 2,
    ) -> List[List[str]]:
        """
        将有较大重叠的接戏链做集合级合并：
        - 如果两条链的交集场次数 >= min_overlap，则合并为并集
        - 反复迭代直到无法继续合并
        - 返回合并后的链列表（每条链去重、长度>=2、集合意义上去重）

        注意：
        - 这里采用 scene_id 的排序作为输出顺序，如果你更在意“剧本顺序”，
          可以改成按 Scene 的 order 属性排序。
        """
        # 先去掉空链，并对单条链内部做去重
        chains = [list(dict.fromkeys(chain)) for chain in chains if chain]

        if not chains:
            return []

        changed = True
        while changed:
            changed = False
            new_chains: List[List[str]] = []
            used = [False] * len(chains)

            for i in range(len(chains)):
                if used[i]:
                    continue

                base_set = set(chains[i])
                merged = False

                for j in range(i + 1, len(chains)):
                    if used[j]:
                        continue

                    other_set = set(chains[j])
                    inter = base_set & other_set

                    # 重叠达到阈值，则合并
                    if len(inter) >= min_overlap:
                        base_set |= other_set
                        used[j] = True
                        changed = True
                        merged = True

                # i 自己也标记已处理
                used[i] = True

                # 把合并后的 base_set 放进新链表
                if merged:
                    # 这里简单按 scene_id 排序；如果想按剧本顺序，
                    # 可以改成用一个 scene_order_map 排序
                    merged_chain = sorted(base_set)
                    new_chains.append(merged_chain)
                else:
                    # 没合并到别人，就原样保留
                    new_chains.append(chains[i])

            chains = new_chains

        # 最后一轮：丢掉长度 < 2 的，并按集合去重
        final_chains: List[List[str]] = []
        seen_sets = set()
        for chain in chains:
            if len(chain) < 2:
                continue
            key = tuple(sorted(chain))
            if key in seen_sets:
                continue
            seen_sets.add(key)
            final_chains.append(chain)

        return final_chains

    # ----------------------------------------------------------------------
    # 三、接戏链的 LLM 多数投票评估
    # ----------------------------------------------------------------------
    def _single_chain_eval(self, chain: List[str], metadata_block: str) -> Dict[str, Any]:
        """单次 LLM 调用 + JSON 解析，方便在线程池里复用。"""
        raw_output = self.information_extractor.check_continuity_chain(
            scene_id_list=chain,
            scene_metadata_blocks=metadata_block,
        )

        parsed = json.loads(correct_json_format(raw_output))

        # 必要字段兜底
        parsed.setdefault("coherence_score", 0.0)
        parsed.setdefault("keep_decision", "keep")
        parsed.setdefault("suggested_splits", [])
        parsed.setdefault("rationale", "")
        parsed.setdefault("decision_confidence", "medium")

        # 再兜底一次，防止奇怪类型
        if not isinstance(parsed["suggested_splits"], list):
            parsed["suggested_splits"] = []

        if parsed["keep_decision"] not in ["keep", "split", "drop"]:
            parsed["keep_decision"] = "keep"

        if parsed["decision_confidence"] not in DECISION_CONF_MAP:
            parsed["decision_confidence"] = "medium"

        return parsed

    def evaluate_chain_with_majority_vote(
        self,
        chain: List[str],
        metadata_blocks: str,
        n_runs: int = 5,
        max_workers: int = 5,
    ) -> Dict[str, Any]:
        """
        对同一条接戏链进行 n_runs 次 LLM 评估，
        使用多数投票 + 置信度/分数选择最终决策。
        """
        results: List[Dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(self._single_chain_eval, chain, metadata_blocks)
                for _ in range(n_runs)
            ]
            for fut in as_completed(futures):
                try:
                    results.append(fut.result())
                except Exception as e:
                    print(f"[WARN] single eval failed: {e}")

        if not results:
            # 全挂了就给一个保守兜底
            return {
                "coherence_score": 0.0,
                "keep_decision": "keep",
                "suggested_splits": [],
                "rationale": "LLM 调用全部失败，采用保守 keep 兜底。",
                "votes": Counter(),
                "all_results": [],
            }

        # 多数投票选 keep_decision
        keep_votes = [r["keep_decision"] for r in results]
        vote_counter = Counter(keep_votes)
        final_decision = vote_counter.most_common(1)[0][0]  # "keep" | "split" | "drop"

        # 工具函数：把 decision_confidence 变成数值方便排序
        def _conf_value(r: Dict[str, Any]) -> int:
            return DECISION_CONF_MAP.get(r.get("decision_confidence", "medium"), 1)

        # 分支 1：多数认为应该 split
        if final_decision == "split":
            split_candidates = [r for r in results if r["keep_decision"] == "split"]
            if not split_candidates:
                # 理论上不应该发生，但保险起见
                best = max(results, key=lambda r: r.get("coherence_score", 0.0))
                return {
                    "coherence_score": best["coherence_score"],
                    "keep_decision": "keep",
                    "suggested_splits": [],
                    "rationale": best["rationale"],
                    "votes": vote_counter,
                    "all_results": results,
                }

            # 这里一般希望选「信心高 + coherence_score 低」的那个
            best = sorted(
                split_candidates,
                key=lambda r: (_conf_value(r), r.get("coherence_score", 0.0)),
            )[0]

            return {
                "coherence_score": best["coherence_score"],
                "keep_decision": "split",
                "suggested_splits": best["suggested_splits"],
                "rationale": best["rationale"],
                "votes": vote_counter,
                "all_results": results,
            }

        # 分支 2：多数认为 keep 或 drop
        decision_candidates = [
            r for r in results if r["keep_decision"] == final_decision
        ]
        if not decision_candidates:
            decision_candidates = results  # 兜底

        if final_decision == "keep":
            # keep → 选 coherence_score 最高的
            best = max(decision_candidates, key=lambda r: r.get("coherence_score", 0.0))
        else:
            # drop → 选 coherence_score 最低的
            best = min(decision_candidates, key=lambda r: r.get("coherence_score", 0.0))

        return {
            "coherence_score": best["coherence_score"],
            "keep_decision": final_decision,
            "suggested_splits": best["suggested_splits"]
            if final_decision == "split"
            else [],
            "rationale": best["rationale"],
            "votes": vote_counter,
            "all_results": results,
        }

    # ----------------------------------------------------------------------
    # 四、Scene pair 构造 & LLM 调用
    # ----------------------------------------------------------------------
    def _get_scene_pairs_with_common_neighbors(self) -> List[Dict[str, Any]]:
        """
        计算所有存在共同邻居(角色 Character)的 Scene 对。

        - 如果 config.knowledge_graph.versions 未配置，或者为 ["default"]，
          则保持旧行为：不过滤 version，所有 Scene 都可以互相组成 pair，
          但返回结果中会统一带上 "version": "default"（或配置中的那一项）。

        - 如果 versions 里有多个值（例如 ["Part_1", "Part_2"]），
          则对每个 version 分别查询，只返回同一 version 下的场次对，
          并在结果里带上该 version。

        返回结构：
        [
          {
            "scene1": <scene_id_1>,
            "scene2": <scene_id_2>,
            "neighbor_ids": [<char_id_1>, <char_id_2>, ...],
            "version": <version for both scene1 and scene2>,
          },
          ...
        ]
        """
        kg_cfg = getattr(self.config, "knowledge_graph", None)
        versions = getattr(kg_cfg, "versions", None) if kg_cfg is not None else None

        pairs: List[Dict[str, Any]] = []

        # ---------- 情况一：未配置 versions 或仅有 ["default"]，保持旧逻辑 ----------
        if (
            not versions
            or (
                isinstance(versions, (list, tuple))
                and len(versions) == 1
                and versions[0] == "default"
            )
        ):
            cypher = """
            MATCH (s1:Scene)-[]-(e:Character)-[]-(s2:Scene)
            WHERE id(s1) < id(s2)
            WITH s1, s2, collect(DISTINCT e.id) AS common
            WHERE size(common) > 0
            RETURN s1.id AS scene1,
                   s2.id AS scene2,
                   common AS neighbor_ids
            ORDER BY scene1, scene2
            """
            rows = self.neo4j_utils.execute_query(cypher) or []
            default_version = (
                versions[0] if (isinstance(versions, (list, tuple)) and versions) else "default"
            )
            for row in rows:
                row["version"] = default_version
                pairs.append(row)
            return pairs

        # ---------- 情况二：有多个 version（例如 ["Part_1", "Part_2"]） ----------
        for v in versions:
            if not v:
                continue
            # 简单做一下引号转义，防止 Cypher 拼接炸掉
            v_str = str(v).replace("'", "\\'")
            cypher = f"""
            MATCH (s1:Scene)-[]-(e:Character)-[]-(s2:Scene)
            WHERE id(s1) < id(s2)
              AND s1.version = '{v_str}'
              AND s2.version = '{v_str}'
            WITH s1, s2, collect(DISTINCT e.id) AS common
            WHERE size(common) > 0
            RETURN s1.id AS scene1,
                   s2.id AS scene2,
                   common AS neighbor_ids
            ORDER BY scene1, scene2
            """
            rows = self.neo4j_utils.execute_query(cypher) or []
            for row in rows:
                row["version"] = v
                pairs.append(row)

        return pairs

    def _filter_pair_by_order(
        self, row: Dict[str, Any], threshold: int = 50
    ) -> bool:
        """
        使用 Scene 的 order 属性限制两场之间的距离。
        """
        scene1 = row["scene1"]
        scene2 = row["scene2"]

        ent1 = self.neo4j_utils.get_entity_by_id(scene1)
        ent2 = self.neo4j_utils.get_entity_by_id(scene2)

        order1 = ent1.properties.get("order")
        order2 = ent2.properties.get("order")
        if order1 is None or order2 is None:
            # 若缺失 order，就保守不过滤
            return True

        try:
            return abs(int(order1) - int(order2)) <= threshold
        except Exception:
            return True

    def _parse_pair_info(self, pair: Dict[str, Any]) -> Dict[str, Any]:
        """
        将 pair 信息整理为 check_screenplay_continuity 所需要的参数。
        """
        scene1_id = pair["scene1"]
        scene2_id = pair["scene2"]

        scene1_ent = self.neo4j_utils.get_entity_by_id(scene1_id)
        scene2_ent = self.neo4j_utils.get_entity_by_id(scene2_id)

        scene_title1 = scene1_ent.name
        scene_title2 = scene2_ent.name

        scene_name1 = scene1_ent.properties["scene_name"]
        scene_subname1 = scene1_ent.properties["sub_scene_name"]
        scene_name2 = scene2_ent.properties["scene_name"]
        scene_subname2 = scene2_ent.properties["sub_scene_name"]

        summary1 = scene1_ent.description
        summary2 = scene2_ent.description

        if self.cmp_tool is not None:
            # 场次 1
            try:
                input_dict1 = {"scene_name": scene_name1}
                if scene_subname1:
                    input_dict1["subscene_name"] = scene_subname1
                cmp_info1 = self.cmp_tool.call(
                    json.dumps(input_dict1, ensure_ascii=False)
                )
            except Exception:
                cmp_info1 = ""

            # 场次 2（修复：之前误用 input_dict1）
            try:
                input_dict2 = {"scene_name": scene_name2}
                if scene_subname2:
                    input_dict2["subscene_name"] = scene_subname2
                cmp_info2 = self.cmp_tool.call(
                    json.dumps(input_dict2, ensure_ascii=False)
                )
            except Exception:
                cmp_info2 = ""
        else:
            cmp_info1 = ""
            cmp_info2 = ""

        # 共同角色信息
        common_neighbors = pair.get("neighbor_ids") or []
        common_neighbor_info_lines: List[str] = []
        for neighbor_id in common_neighbors:
            try:
                neighbor = self.neo4j_utils.get_entity_by_id(neighbor_id)
                common_neighbor_info_lines.append(
                    f"- {neighbor.name}: {neighbor.description}"
                )
            except Exception:
                continue
        common_neighbor_info = "\n".join(common_neighbor_info_lines)

        return {
            "scene_name1": scene_name1,
            "scene_title1": scene_title1,
            "summary1": summary1,
            "cmp_info1": cmp_info1,
            "scene_name2": scene_name2,
            "scene_title2": scene_title2,
            "summary2": summary2,
            "cmp_info2": cmp_info2,
            "common_neighbor_info": common_neighbor_info,
        }

    def _process_one_scene_pair(self, pair: Dict[str, Any]) -> Dict[str, Any]:
        """
        单个场次对的接戏判定逻辑（供并发调度调用）。
        """
        pair_info = self._parse_pair_info(pair)

        # LLM 调用
        scene_title1 = pair_info.pop("scene_title1")
        scene_title2 = pair_info.pop("scene_title2")
        output_str = self.information_extractor.check_screenplay_continuity(
            **pair_info
        )

        try:
            result_dict = json.loads(output_str)
        except Exception:
            result_dict = json.loads(correct_json_format(output_str))

        scene1_id = pair["scene1"]
        scene2_id = pair["scene2"]
        neighbor_ids = pair.get("neighbor_ids") or []

        out: Dict[str, Any] = {
            "scene_id_1": scene1_id,
            "scene_title_1": scene_title1,
            "scene_name_1": pair_info["scene_name1"],
            "summary_1": pair_info["summary1"],
            "cmp_info_1": pair_info["cmp_info1"],
            "scene_id_2": scene2_id,
            "scene_title_2": scene_title2,
            "scene_name_2": pair_info["scene_name2"],
            "summary_2": pair_info["summary2"],
            "cmp_info_2": pair_info["cmp_info2"],
            "common_neighbor_info": pair_info["common_neighbor_info"],
            "neighbor_ids": neighbor_ids,
            "version": pair.get("version", ""),  # ⭐ 新增：这条 pair 所属的 Part
        }

        # 把 LLM 的字段扁平合入（is_continuity, reason 等）
        if isinstance(result_dict, dict):
            out.update(result_dict)
        else:
            out["raw_output"] = output_str

        return out

    # ----------------------------------------------------------------------
    # 五、通用工具方法
    # ----------------------------------------------------------------------
    def _save_results(
        self, out_list: List[Dict[str, Any]], warn: Optional[str] = None
    ) -> str:
        base = self.config.storage.knowledge_graph_path
        os.makedirs(base, exist_ok=True)
        out_path = os.path.join(base, "character_status.json")
        with open(out_path, "w", encoding="utf-8") as fw:
            json.dump(out_list, fw, ensure_ascii=False, indent=2)
        msg = f"✅ 已保存 {len(out_list)} 条结果 -> {out_path}"
        if warn:
            msg += f"（{warn}）"
        print(msg)
        return out_path

    def _save_scene_continuity_results(
        self, out_list: List[Dict[str, Any]], warn: Optional[str] = None
    ) -> str:
        base = self.config.storage.knowledge_graph_path
        os.makedirs(base, exist_ok=True)
        out_path = os.path.join(base, "scene_continuity.json")
        with open(out_path, "w", encoding="utf-8") as fw:
            json.dump(out_list, fw, ensure_ascii=False, indent=2)
        msg = f"✅ 已保存 {len(out_list)} 条接戏结果 -> {out_path}"
        if warn:
            msg += f"（{warn}）"
        print(msg)
        return out_path

    def _parse_scene_properties(self, scene_node: Dict[str, Any]) -> Dict[str, Any]:
        raw_props = scene_node.get("properties", "")
        if isinstance(raw_props, dict):
            return raw_props
        if isinstance(raw_props, str):
            try:
                return json.loads(raw_props)
            except Exception:
                fixed = correct_json_format(raw_props)
                return json.loads(fixed)
        return {}

    def _get_scene_id_from_node(self, scene_node: Dict[str, Any]) -> str:
        props = self._parse_scene_properties(scene_node)
        return self._get_scene_id(scene_node, props)

    def _get_scene_id(
        self, scene_node: Dict[str, Any], props: Dict[str, Any]
    ) -> str:
        if "id" in scene_node and scene_node["id"]:
            return scene_node["id"]
        return props.get("scene_id", "")

    def _default_timelines(self, props: Dict[str, Any]) -> List[Dict[str, Any]]:
        ts = props.get("timelines")
        if ts:
            return ts
        return [{"time": "scene"}]

    def _get_character_list(self, scene_id: str) -> List[str]:
        try:
            rel_ents = self.neo4j_utils.search_related_entities(
                source_id=scene_id,
                entity_types=["Character"],
            )
            return [ent.name for ent in rel_ents if ent.scope=="global" and ent.properties.get("name")] if rel_ents else []
        except Exception:
            return []

    def _get_scene_contents(self, scene_node: Dict[str, Any], scene_id: str) -> str:
        chunk_ids = scene_node.get("source_chunks") or []
        if not chunk_ids:
            raise RuntimeError(f"{scene_id} 未找到 source_chunks")
        docs = self.vector_store.search_by_ids(doc_ids=chunk_ids)
        if not docs:
            raise RuntimeError(f"{scene_id} 通过 chunk_ids 未检索到文档内容")
        text = "\n".join([getattr(doc, "content", "") or "" for doc in docs]).strip()
        if not text:
            raise RuntimeError(f"{scene_id} 场景文本为空")
        return text

    def _merge_scene_properties(
        self, base_result: Dict[str, Any], scene_props: Dict[str, Any]
    ) -> None:
        for k, v in scene_props.items():
            if k in self.EXCLUDE_PROP_KEYS:
                continue
            if isinstance(v, str):
                if v.strip():
                    base_result[k] = v
            elif v is not None:
                base_result[k] = v

    # ----------------------------------------------------------------------
    # 六、角色状态 SQLite 数据库构建
    # ----------------------------------------------------------------------
    def build_character_status_database(self) -> str:
        """
        读取 extract_character_status_for_all_scenes() 生成的 character_status.json，
        将每个场次下的 characters 扁平展开为逐行记录，并保存到
        self.config.storage.sql_database_path 下的 SQLite 与 CSV。

        预期每个 item 的结构示例：
        {
            "scene_id": "...",
            "scene_name": "...",
            "sub_scene_name": "...",
            "version": "...",
            "scene_category": "...",
            "lighting": "...",
            "space": "...",
            "region": "...",
            "main_location": "...",
            "sub_location": "...",
            "summary": "...",
            "characters": [
                {"name": "角色A", "status": "该角色在本场景中的客观状态"},
                {"name": "角色B", "status": "..."}
            ]
        }
        """
        kg_base = self.config.storage.knowledge_graph_path
        in_path = os.path.join(kg_base, "character_status.json")
        if not os.path.exists(in_path):
            raise FileNotFoundError(
                f"未找到文件：{in_path}，请先运行 extract_character_status_for_all_scenes()"
            )

        with open(in_path, "r", encoding="utf-8") as fr:
            items = json.load(fr) or []

        rows: List[Dict[str, Any]] = []
        for it in items:
            # 场次级别元信息（和之前保持一致，便于下游复用）
            base = {
                "title": it.get("scene_name", ""),
                "subtitle": it.get("sub_scene_name", ""),
                "version": it.get("version", ""),
                "scene_category": it.get("scene_category", ""),
                "lighting": it.get("lighting", ""),
                "space": it.get("space", ""),
                "region": it.get("region", ""),
                "main_location": it.get("main_location", ""),
                "sub_location": it.get("sub_location", ""),
                "summary": it.get("summary", ""),
            }

            characters = it.get("characters") or []
            if not isinstance(characters, list):
                # 兜底：异常结构时当作无角色
                characters = []

            if not characters:
                # 无角色信息时依然保留一行占位记录，方便后续统计
                rows.append(
                    {
                        "character": "",
                        "status": "",
                        **base,
                    }
                )
                continue

            # 正常展开每个角色
            for ch in characters:
                if not isinstance(ch, dict):
                    continue
                rows.append(
                    {
                        "character": ch.get("name", ""),
                        "status": ch.get("status", ""),
                        **base,
                    }
                )

        # 构建 DataFrame（注意：已不再包含 timepoint 列）
        df = pd.DataFrame(
            rows,
            columns=[
                "character",
                "status",
                "title",
                "subtitle",
                "version",
                "scene_category",
                "lighting",
                "space",
                "region",
                "main_location",
                "sub_location",
                "summary",
            ],
        )

        # 统一去空格
        for c in df.columns:
            if df[c].dtype == "object":
                df[c] = df[c].astype("string").str.strip()

        # 去重（完全相同行去重）
        if not df.empty:
            df = df.drop_duplicates().reset_index(drop=True)

        # ---------- 落库 ----------
        os.makedirs(self.config.storage.sql_database_path, exist_ok=True)
        db_path = os.path.join(
            self.config.storage.sql_database_path, "CharacterStatus.db"
        )
        if os.path.exists(db_path):
            os.remove(db_path)
        conn = sqlite3.connect(db_path)

        df.to_sql("CharacterStatus", conn, if_exists="replace", index=False)
        conn.commit()

        # 同步导出 CSV
        csv_path = os.path.join(
            self.config.storage.sql_database_path, "CharacterStatus.csv"
        )
        df.to_csv(csv_path, index=False, encoding="utf-8")

        conn.close()
        print(
            f"✅ 角色状态数据库已生成：{db_path}（表：CharacterStatus），CSV：{csv_path}"
        )
        return db_path

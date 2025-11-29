import os
import time
import json
from typing import Optional, Dict, Any, List, Tuple
from copy import deepcopy
from collections import defaultdict

import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

from ..utils.config import KAGConfig
from core.builder.manager.document_manager import DocumentParser
from core.utils.format import correct_json_format


# =========================
# Similarity & clustering utilities
# =========================
def compute_weighted_similarity_and_laplacian(entity_dict, alpha=0.8, knn_k=40):
    """
    Robust: handle tiny n (n<3), cap knn_k, and return a safe estimated_k.
    """
    names = list(entity_dict.keys())
    n = len(names)
    if n == 0:
        return 0, np.zeros((0, 0))
    if n == 1:
        # 单样本，无需聚类
        sim = np.array([[1.0]])
        return 1, sim

    name_embs = np.vstack([entity_dict[nm]['name_embedding'] for nm in names])
    desc_embs = np.vstack([entity_dict[nm]['description_embedding'] for nm in names])

    sim_name = cosine_similarity(name_embs)
    sim_desc = cosine_similarity(desc_embs)
    sim = alpha * sim_name + (1 - alpha) * sim_desc

    # kNN 构图：最多取 n-1 个邻居
    k = max(1, min(knn_k, n - 1))
    adj = np.zeros((n, n), dtype=float)
    for i in range(n):
        # 排除自己后取 top-k
        idx = np.argsort(sim[i])[-(k + 1):-1]  # 已排除 self
        if idx.size:
            adj[i, idx] = sim[i, idx]
    adj = np.maximum(adj, adj.T)  # 对称化

    deg = np.diag(adj.sum(axis=1))
    lap = deg - adj

    eigvals = np.linalg.eigvalsh(lap)
    # n<3 时 gaps[1:] 会为空，直接回退
    if n < 3:
        estimated_k = 1
        return estimated_k, sim

    gaps = np.diff(eigvals)          # 长度 n-1
    inner = gaps[1:]                 # 跳过第一处间隙
    if inner.size == 0 or not np.isfinite(inner).any():
        # 回退：保守取 1
        estimated_k = 1
    else:
        estimated_k = int(np.argmax(inner) + 1)

    # 上下限保护
    estimated_k = max(1, min(estimated_k, n))
    return estimated_k, sim


def run_kmeans_clustering(entity_dict, n_clusters, alpha=0.8):
    """
    Robust KMeans: ensure 2 <= n_clusters <= n, safe fallback for tiny n.
    """
    names = list(entity_dict.keys())
    n = len(names)
    if n < 2:
        return []

    name_embs = np.vstack([entity_dict[nm]['name_embedding'] for nm in names])
    desc_embs = np.vstack([entity_dict[nm]['description_embedding'] for nm in names])

    combined_embs = np.hstack([name_embs * alpha, desc_embs * (1 - alpha)])

    # 约束簇数
    if n_clusters is None or n_clusters < 2:
        # 常用回退：sqrt(n) 或 2 中较大者
        n_clusters = max(2, int(np.sqrt(n)))
    n_clusters = min(max(2, n_clusters), n)

    # 极端情况下 KMeans 可能告警/报错，做一次兜底
    try:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        labels = kmeans.fit_predict(combined_embs)
    except Exception:
        # 回退到两簇或单簇（如果 n==2，必为两簇）
        if n >= 2:
            kmeans = KMeans(n_clusters=min(2, n), random_state=42, n_init='auto')
            labels = kmeans.fit_predict(combined_embs)
        else:
            return []

    cluster_result = defaultdict(list)
    for name, label in zip(names, labels):
        cluster_result[label].append(name)

    # 仅保留 size>=2 的候选
    collected_clusters = [group for group in cluster_result.values() if len(group) >= 2]
    return collected_clusters


# =========================
# Main class
# =========================
class GraphPreprocessor:
    """Knowledge-graph preprocessing (concurrent soft timeout, multi-type retention, configurable cleanup/priority, and scope convergence)."""

    # Default type priority (high -> low)
    TYPE_PRIORITY: Tuple[str, ...] = (
        "Event", "Action", "Object", "Location", "Character", "Organization", "Concept"
    )

    # Default strategies for multi-typed entities
    ALLOW_MULTI_TYPE: bool = True                   # Allow multiple types per entity name
    MAX_TYPES_PER_NODE: int = 3                     # Keep at most N types (by priority); None = no cap
    STRIP_CONCEPT_WHEN_OTHERS: bool = True          # If Concept co-exists with others, drop Concept
    UNIFY_ACTION_TO_EVENT: bool = True              # If Action and Event co-exist, keep Event only

    def __init__(self, config: KAGConfig, llm, system_prompt):
        """
        Args:
            config (KAGConfig): Configuration object.
            llm: Backing LLM instance used by the document parser.
            system_prompt (str): System prompt passed into merge/summarize validators.
        """
        self.config = config
        self.system_prompt_text = system_prompt
        self.document_parser = DocumentParser(config, llm)
        self.model = self.load_embedding_model()
        # Load concurrency level from config if available; else fallback to 4
        self.max_worker = getattr(self.config.document_processing, "max_workers", 4)

    # ---------- Generic concurrency: soft timeout ----------
    def _max_workers(self) -> int:
        return int(getattr(self, "max_workers", getattr(self, "max_worker", 4)))

    def _soft_timeout_pool(self, work_items, submit_fn, *,
                           per_task_timeout: float = 180.0,
                           desc: str = "Concurrent tasks",
                           thread_prefix: str = "pool"):
        """
        Generic concurrent executor with soft timeouts:
        - Non-blocking timeouts,
        - Gather-on-completion,
        - Graceful degradation on exceptions/timeouts.

        Returns:
            List[Tuple[bool, Any, Any]]:
                (timeout_or_error, item, result_or_exception)
        """
        max_workers = self._max_workers()
        executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix=thread_prefix)
        pbar = tqdm(total=len(work_items), desc=desc, ncols=100)

        results = []
        try:
            fut_info = {}
            now_clock = time.monotonic
            for item in work_items:
                f = executor.submit(submit_fn, item)
                fut_info[f] = {"start": now_clock(), "item": item}

            pending = set(fut_info.keys())
            while pending:
                done, pending = wait(pending, timeout=0.25, return_when=FIRST_COMPLETED)
                # Collect finished tasks
                for f in done:
                    item = fut_info[f]["item"]
                    try:
                        res = f.result()
                        results.append((False, item, res))
                    except Exception as e:
                        results.append((True, item, e))
                    pbar.update(1)
                    fut_info.pop(f, None)

                # Soft timeouts
                now = now_clock()
                to_forget = []
                for f in pending:
                    if now - fut_info[f]["start"] >= per_task_timeout:
                        item = fut_info[f]["item"]
                        f.cancel()
                        results.append((True, item, None))
                        pbar.update(1)
                        to_forget.append(f)
                for f in to_forget:
                    pending.remove(f)
                    fut_info.pop(f, None)
        finally:
            pbar.close()
            executor.shutdown(wait=False, cancel_futures=True)
        return results

    # ---------- Model & data prep ----------
    def load_embedding_model(self):
        """
        Load the embedding model based on configuration.
        """
        if self.config.graph_embedding.provider == "openai":
            from core.model_providers.openai_embedding import OpenAIEmbeddingModel
            model = OpenAIEmbeddingModel(self.config.graph_embedding)
        else:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(self.config.graph_embedding.model_name)
        return model

    def collect_global_entities(self, extraction_results):
        """
        Collect and merge entities marked as 'global' across extraction results,
        grouped by allowed types. Concatenate descriptions for duplicates.

        Args:
            extraction_results (List[Dict[str, Any]]): Extraction outputs with 'entities'.

        Returns:
            Dict[str, Dict[str, Dict[str, Any]]]: type -> name -> entity dict
        """
        global_entities: Dict[str, List[Dict[str, Any]]] = dict()
        for result in extraction_results:
            for entity in result.get("entities", []):
                if entity.get("scope") == "global" and entity.get("type") in [
                    "Character", "Object", "Concept", "Event", "Location", "Organization"
                ]:
                    t = entity["type"]
                    global_entities.setdefault(t, []).append(entity)

        merged_global_entities: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for t, ents in global_entities.items():
            store = {}
            for ent in ents:
                nm = ent["name"]
                if nm in store:
                    store[nm]["description"] = store[nm].get("description", "") + ent.get("description", "")
                else:
                    store[nm] = ent
            merged_global_entities[t] = store
        return merged_global_entities

    def compute_embeddings(self, filtered_entities: List[Dict[str, Any]]):
        """
        Compute and attach embeddings for entity name and description/summary.

        Args:
            filtered_entities: List of entity dicts.

        Returns:
            The same list with 'name_embedding' and 'description_embedding' populated.
        """
        for entity in tqdm(filtered_entities, desc="Compute entity embeddings"):
            name_embedding = self.model.encode(entity["name"])
            desc_text = entity.get("summary", entity.get("description", ""))  # prefer summary
            description_embedding = self.model.encode(desc_text.strip())
            entity["name_embedding"] = name_embedding
            entity["description_embedding"] = description_embedding
        return filtered_entities

    def add_entity_summary(self, merged_global_entities: Dict[str, Dict[str, Dict[str, Any]]],
                           per_task_timeout: float = 180.0):
        """
        Summarize entity descriptions in parallel (soft timeout). If a description
        is already short, use it as-is.

        Args:
            merged_global_entities: type -> name -> entity dict
            per_task_timeout: Per-task timeout in seconds.

        Returns:
            Dict[str, List[Dict[str, Any]]]: type -> list of entities (with 'summary')
        """
        entity_list: List[Dict[str, Any]] = []
        for t in merged_global_entities:
            entity_list.extend(list(merged_global_entities[t].values()))

        def summarize_entity(entity: Dict[str, Any]) -> Dict[str, Any]:
            desc = entity.get("description", "")
            if len(desc) >= 300:
                raw = self.document_parser.summarize_paragraph(text=desc, max_length=250)
                data = json.loads(correct_json_format(raw))
                entity["summary"] = data.get("summary", desc)
            else:
                entity["summary"] = desc
            return entity

        results = self._soft_timeout_pool(
            entity_list, summarize_entity, per_task_timeout=per_task_timeout,
            desc="Summarize entities (concurrent)", thread_prefix="summ"
        )

        updated: List[Dict[str, Any]] = []
        for timeout_or_error, item, res in results:
            if timeout_or_error or res is None:
                item["summary"] = item.get("description", "")
                updated.append(item)
            else:
                updated.append(res)

        merged_new: Dict[str, List[Dict[str, Any]]] = {}
        for e in updated:
            merged_new.setdefault(e["type"], []).append(e)
        return merged_new

    def detect_candidates(self, merged_global_entities: Dict[str, List[Dict[str, Any]]]):
        """
        Robust candidate detection: skip tiny sets; cap knn; safe n_clusters.
        """
        candidates = []
        for t, ent_list in merged_global_entities.items():
            # 聚合同名（同名合并描述，避免重复干扰）
            by_name = {}
            for e in ent_list:
                nm = e["name"]
                if nm in by_name:
                    by_name[nm]["description"] = by_name[nm].get("description", "") + e.get("description", "")
                    # 保留已有 embedding/summary
                else:
                    by_name[nm] = e

            n = len(by_name)
            if n < 3:
                # 样本太少，不做谱估计与聚类，直接跳过
                continue

            # knn_k：随 n 调整并上限保护
            knn_k = max(5, min(int(n / 4), 25, n - 1))

            # 谱估计 + 聚类
            estimated_k, _ = compute_weighted_similarity_and_laplacian(by_name, alpha=0.8, knn_k=knn_k)

            # 给一个稳健的 n_clusters 回退：介于 2..n 之间
            # 原式：max(2, int((estimated_k + n/2) / 2))
            rough = max(2, int((estimated_k + n / 2) / 2))
            n_clusters = min(max(2, rough), n)

            clusters = run_kmeans_clustering(by_name, n_clusters=n_clusters, alpha=0.5)
            candidates.extend(clusters)
        return candidates


    def merge_entities(self, all_candidates_with_info, per_task_timeout: float = 120.0):
        """
        For each candidate group, ask the LLM to propose merges (canonical name + aliases).

        Args:
            all_candidates_with_info: List of groups, each a list of entity dicts with name/summary.

        Returns:
            Dict[str, str]: alias -> canonical_name mapping.
        """
        def _empty():
            return {"merges": [], "unmerged": []}

        def _run_group(group):
            try:
                parts = []
                for i, e in enumerate(group):
                    name = e.get("name")
                    summary = e.get("summary") or e.get("description") or ""
                    parts.append(
                        f"Entity {i + 1} name: {name}\n - Description: {summary}\n"
                    )
                entity_descriptions = "\n".join(parts)
                raw = self.document_parser.merge_entities(
                    entity_descriptions=entity_descriptions,
                    system_prompt=self.system_prompt_text
                )
                data = raw if isinstance(raw, dict) else json.loads(correct_json_format(raw))
                merges = data.get("merges", []) if isinstance(data, dict) else []
                unmerged = data.get("unmerged", []) if isinstance(data, dict) else []
                if not isinstance(merges, list):
                    merges = []
                if not isinstance(unmerged, list):
                    unmerged = []
                return {"merges": merges, "unmerged": unmerged}
            except Exception as e:
                names = [str(ent.get("name", "")) for ent in group]
                print(f"[!] Entity merge failed: {names} -> {e}")
                return _empty()

        results = self._soft_timeout_pool(
            all_candidates_with_info, _run_group, per_task_timeout=per_task_timeout,
            desc="Entity merge decision (concurrent)", thread_prefix="merge"
        )

        rename_map: Dict[str, str] = {}
        for timeout_or_error, _item, res in results:
            data = res if (res and isinstance(res, dict)) else {"merges": [], "unmerged": []}
            for m in data.get("merges", []) or []:
                canonical = m.get("canonical_name")
                if not canonical:
                    continue
                for alias in (m.get("aliases") or []):
                    if isinstance(alias, str) and alias:
                        rename_map[alias] = canonical
        return rename_map

    # ---------- Apply rules on a deep copy and return ----------
    def _apply_entity_rules(
        self,
        extraction_results: List[Dict[str, Any]],
        *,
        type_rules: Optional[Dict[str, Dict[str, str]]] = None,   # {entity_name: {old_type: new_type, ...}}
        scope_rules: Optional[Dict[str, str]] = None,              # {entity_name: "global"/"local"}
        normalize_types: bool = True                                # 仅用于整理 types 列表；不应强行改动主 type
    ) -> Tuple[List[Dict[str, Any]], int, int]:
        """
        在 extraction_results 的深拷贝上应用类型/作用域规则。

        关键语义调整：
        - 仅当命中“该实体名对应的 type_rules”时，才修改该实体的 types/主 type；
        - 主 type（ent["type"]）只在其自身被映射（old_type 命中）时才会改变；
        - normalize_types=True 只用于整理 ent["types"] 列表（去重/清理/限长），不再自动把主 type 设为列表第一个；
        若归一化后主 type 不在列表中，会把主 type 注入回列表以保持一致性。
        """

        new_results = deepcopy(extraction_results)
        type_changed = 0
        scope_changed = 0

        type_rules = type_rules or {}
        scope_rules = scope_rules or {}

        for doc in new_results:
            ents = doc.get("entities", [])
            for ent in ents:
                nm = ent.get("name")
                if not nm:
                    continue

                # ---- 收集原始主类型与列表 ----
                original_primary = ent.get("type") if isinstance(ent.get("type"), str) else None
                original_list = []
                if isinstance(ent.get("types"), list):
                    original_list = [t for t in ent["types"] if isinstance(t, str) and t.strip()]
                # 确保主类型也在集合中（保持一致）
                if isinstance(original_primary, str) and original_primary.strip():
                    if original_primary not in original_list:
                        original_list = [original_primary] + original_list

                # 去重但保留相对次序（主类型优先）
                seen = set()
                merged_types = []
                for t in original_list:
                    if t not in seen:
                        seen.add(t)
                        merged_types.append(t)

                before_primary = original_primary
                before_types_list = list(merged_types)

                # ---- 作用域规则（独立于类型规则；立即写回）----
                if nm in scope_rules and isinstance(scope_rules[nm], str):
                    tgt_norm = scope_rules[nm].strip().lower()
                    if tgt_norm in ("global", "local"):
                        tgt = "global" if tgt_norm == "global" else "local"
                        if ent.get("scope") != tgt:
                            ent["scope"] = tgt
                            scope_changed += 1

                # ---- 类型规则：仅当该 name 存在映射时才处理 ----
                mapping = type_rules.get(nm, None)
                if mapping and isinstance(mapping, dict):
                    # 1) 列表映射
                    mapped_list = []
                    any_list_mapped = False
                    for t in merged_types:
                        new_t = mapping.get(t, t)
                        mapped_list.append(new_t)
                        if new_t != t:
                            any_list_mapped = True

                    # 2) 主类型映射（仅当主类型命中时才改变主类型）
                    if isinstance(before_primary, str) and before_primary.strip():
                        mapped_primary = mapping.get(before_primary, before_primary)
                    else:
                        mapped_primary = before_primary
                    primary_changed = (mapped_primary != before_primary)

                    merged_types = mapped_list

                    # 如果需要整理 types 列表
                    if normalize_types:
                        cleaned_list = self._sanitize_type_set(set(merged_types))
                        # 确保主类型在列表中（主类型可能因为清理不在集合里）
                        if isinstance(mapped_primary, str) and mapped_primary not in cleaned_list:
                            # 主类型优先插入到列表最前（不改变主类型本身）
                            cleaned_list = [mapped_primary] + [t for t in cleaned_list if t != mapped_primary]
                        final_types_list = cleaned_list if cleaned_list else ( [mapped_primary] if mapped_primary else ["Concept"] )
                    else:
                        # 仅去重保序
                        tmp_seen = set()
                        final_types_list = []
                        for t in merged_types:
                            if t not in tmp_seen:
                                tmp_seen.add(t)
                                final_types_list.append(t)
                        if not final_types_list:
                            final_types_list = [mapped_primary] if mapped_primary else ["Concept"]
                        # 确保主类型在列表中
                        if isinstance(mapped_primary, str) and mapped_primary not in final_types_list:
                            final_types_list = [mapped_primary] + [t for t in final_types_list if t != mapped_primary]

                    # ---- 写回（仅当有变化）----
                    wrote = False

                    # types 列表变化？
                    if final_types_list != before_types_list:
                        ent["types"] = final_types_list if self.ALLOW_MULTI_TYPE else [final_types_list[0]]
                        wrote = True
                    else:
                        # 未变化也要确保字段存在（与原始保持一致）
                        if self.ALLOW_MULTI_TYPE:
                            ent["types"] = before_types_list
                        else:
                            ent["types"] = [before_types_list[0]] if before_types_list else []

                    # 主类型是否需要改变？（仅当主类型被映射命中时）
                    if primary_changed:
                        ent["type"] = mapped_primary
                        wrote = True
                    else:
                        # 不改变主类型；但如果为空，回退一个
                        if not isinstance(ent.get("type"), str) or not ent["type"].strip():
                            ent["type"] = mapped_primary if isinstance(mapped_primary, str) and mapped_primary.strip() else \
                                        (final_types_list[0] if final_types_list else "Concept")

                    if wrote:
                        type_changed += 1

                else:
                    # 没有命中任何类型规则：完全不动 types 与主 type（除非字段缺失则稳健补齐）
                    if "types" not in ent or not isinstance(ent["types"], list):
                        ent["types"] = before_types_list
                    if not isinstance(ent.get("type"), str) or not ent["type"].strip():
                        ent["type"] = before_primary or (before_types_list[0] if before_types_list else "Concept")
                    # 不计入 type_changed

        return new_results, type_changed, scope_changed


    # ---------- Multi-type: set cleanup & primary selection ----------
    def _sanitize_type_set(self, tset: set) -> List[str]:
        """
        Clean and sort a set of types, returning an ordered list by priority.

        Rules:
        - If Concept co-exists with other types, remove Concept (configurable).
        - If Action and Event co-exist, keep Event only (configurable).
        - Sort by TYPE_PRIORITY; unknown types are placed after known types but above Concept.
        - If MAX_TYPES_PER_NODE is set, truncate to that many types.
        """
        pr_index = {t: i for i, t in enumerate(self.TYPE_PRIORITY)}
        s = set(tset)

        if self.STRIP_CONCEPT_WHEN_OTHERS and "Concept" in s and len(s - {"Concept"}) >= 1:
            s.discard("Concept")

        if self.UNIFY_ACTION_TO_EVENT and "Action" in s and "Event" in s:
            s.discard("Action")  # keep Event

        def score(t: str) -> int:
            if t == "Concept":
                return 10_000_000  # lowest priority
            return pr_index.get(t, 10_000)  # unknown types after known, before Concept

        ordered = sorted(s, key=score)
        if self.MAX_TYPES_PER_NODE is not None and self.MAX_TYPES_PER_NODE > 0:
            ordered = ordered[: self.MAX_TYPES_PER_NODE]
        return ordered

    @staticmethod
    def _collect_type_sets(results: List[Dict[str, Any]]) -> Dict[str, set]:
        """
        Collect the union of types per entity name from both `type` and `types`.
        """
        m = defaultdict(set)
        for doc in results:
            for ent in doc.get("entities", []):
                nm = ent.get("name")
                tp = ent.get("type")
                if nm and isinstance(tp, str):
                    m[nm].add(tp)
                if nm and isinstance(ent.get("types"), list):
                    for t in ent["types"]:
                        if isinstance(t, str):
                            m[nm].add(t)
        return m

    def _attach_multilabel(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Attach `types` (multi-label) and set primary `type` to the highest-priority label.
        但对已消歧名的实体（名字形如 name(Type)）启用“固定类型”保护：
        - 直接把 Type 作为唯一类型与主 type 写回
        - 不参与基于名字的全局聚合
        """
        import re
        new_results = deepcopy(results)

        # 收集全局（按名字）的类型集合
        type_sets = self._collect_type_sets(new_results)

        # 识别已消歧名的名字：末尾括号中的内容作为固定类型
        disambig_pattern = re.compile(r"^(.*)\(([^)]+)\)$")
        pinned_types: Dict[str, str] = {}  # name -> pinned_type
        for name in list(type_sets.keys()):
            m = disambig_pattern.match(name)
            if m:
                pinned_types[name] = m.group(2)

        # 对已消歧名的 name，直接覆盖其聚合类型为 pinned_type
        for name, ptype in pinned_types.items():
            type_sets[name] = {ptype}

        # 计算每个名字应写回的类型列表
        per_name_types: Dict[str, List[str]] = {}
        for name, s in type_sets.items():
            if name in pinned_types:
                cleaned = [pinned_types[name]]
            else:
                cleaned = self._sanitize_type_set(set(s)) or ["Concept"]
            per_name_types[name] = cleaned

        # 写回
        for doc in new_results:
            for ent in doc.get("entities", []):
                nm = ent.get("name")
                if not nm:
                    continue
                if nm in per_name_types:
                    types_list = per_name_types[nm]
                    # 已消歧名的，严格固定主 type 与 types
                    if nm in pinned_types:
                        ent["types"] = [types_list[0]]
                        ent["type"] = types_list[0]
                    else:
                        ent["types"] = types_list if self.ALLOW_MULTI_TYPE else [types_list[0]]
                        ent["type"] = types_list[0]
        return new_results


    # ---------- Type refinement (concurrent + rules + multi-label writeback) ----------
    def refine_entity_types(self, extraction_results, per_task_timeout: float = 120.0):
        """
        Two-stage refinement (base rules -> LLM). Base rules are applied once and then discarded.
        After applying LLM rules, any names that STILL have multiple types *without* LLM guidance
        will be disambiguated by *renaming* each occurrence to "name(type)".
        This renaming is applied to both entities and relations.
        """

        # ---- helpers ----
        def _collect_multi(results):
            m = {}
            for doc in results:
                for ent in doc.get("entities", []):
                    nm = ent.get("name")
                    if not nm:
                        continue
                    m.setdefault(nm, set())
                    tp = ent.get("type")
                    if isinstance(tp, str) and tp.strip():
                        m[nm].add(tp.strip())
                    tps = ent.get("types")
                    if isinstance(tps, list):
                        for t in tps:
                            if isinstance(t, str) and t.strip():
                                m[nm].add(t.strip())
            return {k: v for k, v in m.items() if len(v) > 1}

        def _best_non_concept(ts: set) -> str:
            pr_index = {t: i for i, t in enumerate(self.TYPE_PRIORITY)}
            cands = [t for t in ts if t != "Concept"]
            if not cands:
                return "Concept"
            return min(cands, key=lambda t: pr_index.get(t, 10_000))

        def _type_priority_key(t: str) -> int:
            pr_index = {tp: i for i, tp in enumerate(self.TYPE_PRIORITY)}
            if t == "Concept":
                return 10_000_000
            return pr_index.get(t, 10_000)

        def _rename_ambiguous_names(results, unresolved_names: set):
            """
            对仍多类型且 LLM 无规则的基础名按实例消歧，并对齐 relations。
            - 实体：name -> "name(type)"，并将该实例的 types 固定为 [type]
            - 关系：若 subject/object 仍是裸的基础名，用本 doc 的首选类型改写为 "name(type)"
            * 首选类型 = 文档内该基础名出现频次最高的类型；频次并列按 TYPE_PRIORITY 排序最靠前者
            """
            import re
            from collections import defaultdict, Counter
            new_results = deepcopy(results)

            # 用于 relations 端点选择“首选类型”的优先级
            pr_index = {t: i for i, t in enumerate(self.TYPE_PRIORITY)}
            def _prio(t: str) -> int:
                # Concept 最低优先级
                return 10_000_000 if t == "Concept" else pr_index.get(t, 10_000)

            bracket_pat = re.compile(r"^(.*)\(([^)]+)\)$")  # 已消歧名检测

            for doc in new_results:
                ents = doc.get("entities", []) or []
                rels = doc.get("relations", []) or []

                # 1) 统计：该文档里，目标基础名 -> 出现的类型列表（按实体实例）
                base_types = defaultdict(list)
                for e in ents:
                    base = e.get("name", "")
                    # 注意：此时可能已有部分实体被消歧名过，先提取基础名
                    m = bracket_pat.match(base)
                    if m:
                        base, typ = m.group(1), m.group(2)
                        if base in unresolved_names:
                            base_types[base].append(typ)
                    else:
                        if base in unresolved_names:
                            base_types[base].append(e.get("type"))

                # 2) 选 relations 端点的首选类型（频次优先，频次并列用优先级靠前者）
                preferred_rel_type = {}
                for base, tlist in base_types.items():
                    cnt = Counter(tlist)
                    # 候选按：频次降序，优先级升序
                    preferred_rel_type[base] = sorted(cnt.keys(), key=lambda t: (-cnt[t], _prio(t)))[0]

                # 3) 逐个实体重命名：name -> "name(type)"，并固定 types 为 [type]
                for e in ents:
                    nm = e.get("name", "")
                    m = bracket_pat.match(nm)
                    if m:
                        # 已经是 name(type) 的，跳过（但仍确保 types 与主 type 对齐）
                        base, typ = m.group(1), m.group(2)
                        if base in unresolved_names:
                            e["types"] = [typ]
                            e["type"] = typ
                        continue

                    if nm in unresolved_names:
                        cur_t = e.get("type")
                        # 如果当前类型缺失，回退到 relations 的首选类型或 Concept
                        if not isinstance(cur_t, str) or not cur_t:
                            cur_t = preferred_rel_type.get(nm, "Concept")
                        e["name"] = f"{nm}({cur_t})"
                        e["types"] = [cur_t]
                        e["type"] = cur_t

                # 4) relations：把裸的基础名端点替换为 "name(preferred_type)"
                def _rewrite_endpoint(val):
                    if not isinstance(val, str) or not val:
                        return val
                    # 已带括号的视为已消歧，不改
                    if bracket_pat.match(val):
                        return val
                    if val in preferred_rel_type:
                        return f"{val}({preferred_rel_type[val]})"
                    return val

                for r in rels:
                    r["subject"] = _rewrite_endpoint(r.get("subject"))
                    r["object"]  = _rewrite_endpoint(r.get("object"))

                    # 可选：relation_name 中若出现裸的基础名，且未带括号，则同步替换
                    rn = r.get("relation_name")
                    if isinstance(rn, str) and rn:
                        updated = rn
                        for base, t in preferred_rel_type.items():
                            # 只在未出现括号版本时替换，避免重复
                            if base in updated and f"{base}(" not in updated:
                                updated = updated.replace(base, f"{base}({t})")
                        r["relation_name"] = updated

            return new_results

        # ---- PASS 0: inspect ----
        entity_type_checker = _collect_multi(extraction_results)
        print("Refining entity types (pass0)", entity_type_checker)

        # ---- PASS 1: build & APPLY BASE RULES, then discard them ----
        base_rules: Dict[str, Dict[str, str]] = {}
        for name, types in entity_type_checker.items():
            tset = set(types)
            if "Concept" in tset and len(tset) >= 2 and self.STRIP_CONCEPT_WHEN_OTHERS:
                base_rules.setdefault(name, {})
                base_rules[name]["Concept"] = _best_non_concept(tset)
            if self.UNIFY_ACTION_TO_EVENT and "Action" in tset and "Event" in tset:
                base_rules.setdefault(name, {})
                base_rules[name]["Action"] = "Event"

        if base_rules:
            print("Base type rules (pass1):", base_rules)

        # Apply base rules ONCE
        mid_results, _, _ = self._apply_entity_rules(
            extraction_results=extraction_results,
            type_rules=base_rules,
            scope_rules=None,
            normalize_types=True
        )
        # mid_results = self._attach_multilabel(mid_results)

        # ---- PASS 2: re-check; only unresolved conflicts go to LLM ----
        entity_type_checker_2 = _collect_multi(mid_results)
        print("Refining entity types (pass1 -> pass2 input)", entity_type_checker_2)

        to_check = [(n, tset) for n, tset in entity_type_checker_2.items() if len(tset) > 1]
        print("*****to_check (pass2)", to_check)

        llm_rules: Dict[str, Dict[str, str]] = {}

        def _check_entity(item):
            entity, types = item
            ctx = self.prepare_context_by_type(entity_name=entity, extraction_results=mid_results, types=list(types))
            raw = self.document_parser.validate_entity_type(ctx)
            data = json.loads(correct_json_format(raw))
            rules = {}
            if isinstance(data, dict) and data.get("filtering_rules"):
                for d in data["filtering_rules"]:
                    if isinstance(d, dict):
                        rules.update(d)  # {old_type: new_type}
            print("LLM rules for", entity, "=>", rules)
            return entity, rules

        if to_check:
            results = self._soft_timeout_pool(
                to_check, _check_entity, per_task_timeout=per_task_timeout,
                desc="Validate entity types (pass2, concurrent)", thread_prefix="etype2"
            )
            for timeout_or_error, _item, res in results:
                if timeout_or_error or res is None:
                    continue
                entity, rules = res
                # 记录即使为空 dict，也要知道该实体在pass2没有给出规则
                llm_rules[entity] = rules or {}

        # IMPORTANT: Base rules are discarded now. Only apply LLM rules in the final pass.
        if any(v for v in llm_rules.values()):
            # 有非空规则才应用到 mid_results
            filtered_llm_rules = {k: v for k, v in llm_rules.items() if v}
            print("Final LLM-only rules:", filtered_llm_rules)
            new_results, _, _ = self._apply_entity_rules(
                extraction_results=mid_results,
                type_rules=filtered_llm_rules,
                scope_rules=None,
                normalize_types=True
            )
        else:
            new_results = mid_results

        # ---- FINAL: 对仍然多类型且 LLM 无规则的名字执行重命名 name(type) ----
        # 重新检查（在应用了非空 LLM 规则之后）
        checker_after_llm = _collect_multi(new_results)
        # 未给出规则（或规则为空 dict）的名字，且依然多类型 → 需要重命名
        unresolved_names = {
            n for n, tset in checker_after_llm.items()
            if len(tset) > 1 and (n not in llm_rules or not llm_rules.get(n))
        }
        if unresolved_names:
            print("Renaming unresolved multi-typed names:", unresolved_names)
            new_results = _rename_ambiguous_names(new_results, unresolved_names)

        # Final multi-label write-back
        new_results = self._attach_multilabel(new_results)
        return new_results

    # ---------- Scope normalization & convergence (independent of multi-label) ----------
    @staticmethod
    def _norm_scope(val: Optional[str]) -> Optional[str]:
        """
        Normalize scope to {'global','local'}; return None for other values.

        Note: This mapping intentionally preserves multilingual keys.
        """
        if not isinstance(val, str):
            return None
        v = val.strip().lower()
        mapping = {
            "global": "global", "全局": "global", "整体": "global", "总体": "global",
            "local": "local", "局部": "local", "片段": "local", "场景": "local"
        }
        return mapping.get(v, "global" if v == "global" else ("local" if v == "local" else None))

    @staticmethod
    def _collect_scope_counts(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
        """
        Count occurrences of normalized scope per entity name.
        """
        counts = defaultdict(lambda: {"global": 0, "local": 0})
        for doc in results:
            for ent in doc.get("entities", []):
                nm = ent.get("name")
                sc = GraphPreprocessor._norm_scope(ent.get("scope"))
                if nm and sc in ("global", "local"):
                    counts[nm][sc] += 1
        return counts

    def refine_entity_scope(self, extraction_results, per_task_timeout: float = 180.0,
                            tie_breaker: str = "global"):
        """
        Scope refinement procedure:
        1) Query LLM for suggestions (concurrent + soft timeout);
        2) Normalize all scopes;
        3) Majority vote fallback; ties resolved via `tie_breaker`;
        4) Return a deep-copied result where the same name does not carry multiple scopes.

        Args:
            extraction_results: Original extraction results.
            per_task_timeout: Timeout for concurrent LLM checks.
            tie_breaker: 'global' or 'local' for perfect ties.

        Returns:
            New results with unified scopes per entity name.
        """
        # Identify names with multiple scopes
        entity_scope_checker: Dict[str, set] = {}
        for doc in extraction_results:
            for ent in doc.get("entities", []):
                nm = ent.get("name")
                if nm:
                    entity_scope_checker.setdefault(nm, set()).add(ent.get("scope"))

        to_check_names = [k for k, s in entity_scope_checker.items() if len(s) > 1]

        def _check_scope(entity_name: str):
            ctx = self.prepare_context_by_scope(entity_name=entity_name, extraction_results=extraction_results)
            raw = self.document_parser.validate_entity_scope(ctx)
            data = json.loads(correct_json_format(raw))
            scope_val = data.get("scope") if isinstance(data, dict) else None
            scope_std = self._norm_scope(scope_val)
            return entity_name, scope_std

        scope_rules: Dict[str, str] = {}
        if to_check_names:
            results = self._soft_timeout_pool(
                to_check_names, _check_scope, per_task_timeout=per_task_timeout,
                desc="Validate entity scope (concurrent)", thread_prefix="escope"
            )
            for timeout_or_error, _item, res in results:
                if timeout_or_error or res is None:
                    continue
                name, scope_std = res
                if scope_std in ("global", "local"):
                    scope_rules[name] = scope_std

        # Apply LLM-derived scope rules first
        new_results, _, _ = self._apply_entity_rules(
            extraction_results=extraction_results, type_rules=None, scope_rules=scope_rules
        )

        # Normalize all scopes
        for doc in new_results:
            for ent in doc.get("entities", []):
                sc = self._norm_scope(ent.get("scope"))
                if sc in ("global", "local"):
                    ent["scope"] = sc

        # Majority-vote fallback
        counts = self._collect_scope_counts(new_results)
        fallback_rules: Dict[str, str] = {}
        for name, c in counts.items():
            g, l = c["global"], c["local"]
            if g > 0 and l > 0:
                if g > l:
                    target = "global"
                elif l > g:
                    target = "local"
                else:
                    target = "global" if tie_breaker == "global" else "local"
                fallback_rules[name] = target

        if fallback_rules:
            new_results, _, _ = self._apply_entity_rules(
                extraction_results=new_results, type_rules=None, scope_rules=fallback_rules
            )
        return new_results

    # ---------- Misc helpers ----------
    def get_entity_info(self, entity_name, extraction_results, scope=None, entity_type=None):
        """
        Retrieve entities matching the provided filters.

        Args:
            entity_name (str): Name to match.
            extraction_results: Source results.
            scope (Optional[str]): 'global' or 'local' to filter by scope.
            entity_type (Optional[str]): Primary type to filter by.

        Returns:
            List[Dict[str, Any]]: Matching entities.
        """
        results = []
        for doc in extraction_results:
            for ent in doc.get("entities", []):
                if ent.get("name") != entity_name:
                    continue
                if scope is not None and ent.get("scope") != scope:
                    continue
                if entity_type is not None and ent.get("type") != entity_type:
                    continue
                results.append(ent)
        return results

    def prepare_context_by_type(self, entity_name, extraction_results, types):
        """
        Build a type-focused context string for LLM validation.

        Args:
            entity_name (str): Entity name.
            extraction_results: Source results.
            types (List[str]): Types to include.

        Returns:
            str: Formatted context.
        """
        context = ""
        for type_ in types:
            context += f"Entity type: {type_}\n"
            results = self.get_entity_info(entity_name, extraction_results, entity_type=type_)
            for result in results:
                context += f"- Name: {result['name']}, Description: {result.get('description','')}\n"
            context += "\n"
        return context

    def prepare_context_by_scope(self, entity_name, extraction_results):
        """
        Build a scope-focused context string for LLM validation.

        Args:
            entity_name (str): Entity name.
            extraction_results: Source results.

        Returns:
            str: Formatted context.
        """
        context = ""
        for scope_ in ["global", "local"]:
            context += f"Entity scope: {scope_}\n"
            results = self.get_entity_info(entity_name, extraction_results, scope=scope_)
            for result in results:
                context += f"- Name: {result['name']}, Description: {result.get('description','')}\n"
            context += "\n"
        return context

    # ---------- End-to-end entity disambiguation ----------
    def run_entity_disambiguation(self, extraction_results):
        """
        End-to-end pipeline:
        - Collect and merge global entities (summarize -> embed -> cluster -> LLM merge),
        - Build alias->canonical rename map,
        - Apply renames to entities and relations,
        - Re-attach multi-label types to avoid Concept bleed and preserve labels.
        """
        merged_global_entities = self.collect_global_entities(extraction_results)
        merged_global_entities = self.add_entity_summary(merged_global_entities)

        for t in list(merged_global_entities.keys()):
            merged_global_entities[t] = self.compute_embeddings(merged_global_entities[t])

        all_candidates = self.detect_candidates(merged_global_entities)

        entity_info_map = {}
        for t, entities in merged_global_entities.items():
            for entity in entities:
                entity_info_map[entity["name"]] = entity

        all_candidates_with_info = []
        for cand_group in all_candidates:
            group = [entity_info_map[name] for name in cand_group if name in entity_info_map]
            if len(group) >= 2:
                all_candidates_with_info.append(group)

        rename_map = self.merge_entities(all_candidates_with_info)

        base = self.config.storage.knowledge_graph_path
        os.makedirs(base, exist_ok=True)
        json.dump(rename_map,
                  open(os.path.join(base, "rename_map.json"), "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)

        # Apply renames in-place for entities and relations
        for result in extraction_results:
            for ent in result.get("entities", []):
                ent["name"] = rename_map.get(ent["name"], ent["name"])
                if "name_embedding" in ent:
                    del ent["name_embedding"]
                if "description_embedding" in ent:
                    del ent["description_embedding"]
            for rel in result.get("relations", []):
                try:
                    rel["subject"] = rename_map.get(rel["subject"], rel["subject"])
                    rel["object"] = rename_map.get(rel["object"], rel["object"])
                except Exception:
                    print(rel)
                    continue

        # Re-attach multi-label types post-rename
        extraction_results = self._attach_multilabel(extraction_results)
        return extraction_results

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
# 相似度与聚类工具
# =========================
def compute_weighted_similarity_and_laplacian(entity_dict, alpha=0.8, knn_k=40):
    names = list(entity_dict.keys())
    name_embs = np.vstack([entity_dict[n]['name_embedding'] for n in names])
    desc_embs = np.vstack([entity_dict[n]['description_embedding'] for n in names])

    sim_name = cosine_similarity(name_embs)
    sim_desc = cosine_similarity(desc_embs)
    sim = alpha * sim_name + (1 - alpha) * sim_desc

    n = sim.shape[0]
    adj = np.zeros((n, n))
    for i in range(n):
        idx = np.argsort(sim[i])[-(knn_k + 1):-1]  # 排除自己
        adj[i, idx] = sim[i, idx]
    adj = np.maximum(adj, adj.T)  # 对称化

    deg = np.diag(adj.sum(axis=1))
    lap = deg - adj

    eigvals = np.linalg.eigvalsh(lap)
    gaps = np.diff(eigvals)
    estimated_k = int(np.argmax(gaps[1:]) + 1)  # 跳过第一个gap
    return estimated_k, sim


def run_kmeans_clustering(entity_dict, n_clusters, alpha=0.8):
    names = list(entity_dict.keys())
    name_embs = np.vstack([entity_dict[n]['name_embedding'] for n in names])
    desc_embs = np.vstack([entity_dict[n]['description_embedding'] for n in names])

    combined_embs = np.hstack([
        name_embs * alpha,
        desc_embs * (1 - alpha)
    ])

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(combined_embs)

    cluster_result = defaultdict(list)
    for name, label in zip(names, labels):
        cluster_result[label].append(name)

    collected_clusters = []
    for _, group in cluster_result.items():
        if len(group) >= 2:
            collected_clusters.append(group)
    return collected_clusters


# =========================
# 主类
# =========================
class GraphPreprocessor:
    """知识图谱预处理（并发软超时 + 多类型保留 + 可配置清洗/优先级 + scope 收敛）"""

    # ---- 你可以按需调整默认类型优先级（从高到低）----
    TYPE_PRIORITY: Tuple[str, ...] = (
        "Event", "Action", "Object", "Location", "Character", "Organization", "Concept"
    )

    # ---- 多类型行为的默认策略 ----
    ALLOW_MULTI_TYPE: bool = True                   # ★ 允许同名实体保留多类型
    MAX_TYPES_PER_NODE: int = 3                     # ★ 最多保留几个类型（优先级截断）；None 表示不截断
    STRIP_CONCEPT_WHEN_OTHERS: bool = True          # ★ 若包含 Concept 且有其他类型，移除 Concept
    UNIFY_ACTION_TO_EVENT: bool = True              # ★ 若 Action 和 Event 同时存在，就把 Action → Event（只保留 Event）

    def __init__(self, config: KAGConfig, llm, system_prompt):
        self.config = config
        self.system_prompt_text = system_prompt
        self.document_parser = DocumentParser(config, llm)
        self.model = self.load_embedding_model()
        # 从配置读取并发数；若无则退回默认 4
        self.max_worker = getattr(self.config.document_processing, "max_workers", 4)

    # ---------- 通用并发：软超时 ----------
    def _max_workers(self) -> int:
        return int(getattr(self, "max_workers", getattr(self, "max_worker", 4)))

    def _soft_timeout_pool(self, work_items, submit_fn, *,
                           per_task_timeout: float = 120.0,
                           desc: str = "并发任务",
                           thread_prefix: str = "pool"):
        """
        通用并发执行器：软超时（不阻塞收尾）、完成即收集、异常/超时降级。
        返回：List[Tuple[bool timeout_or_error, Any item, Any result_or_exc]]
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
                # 收集已完成
                for f in done:
                    item = fut_info[f]["item"]
                    try:
                        res = f.result()
                        results.append((False, item, res))
                    except Exception as e:
                        results.append((True, item, e))
                    pbar.update(1)
                    fut_info.pop(f, None)

                # 软超时
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

    # ---------- 模型与数据准备 ----------
    def load_embedding_model(self):
        if self.config.graph_embedding.provider == "openai":
            from core.model_providers.openai_embedding import OpenAIEmbeddingModel
            model = OpenAIEmbeddingModel(self.config.graph_embedding)
        else:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(self.config.graph_embedding.model_name)
        return model

    def collect_global_entities(self, extraction_results):
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
        for entity in tqdm(filtered_entities, desc="计算实体向量"):
            name_embedding = self.model.encode(entity["name"])
            desc_text = entity.get("summary", entity.get("description", ""))  # 优先 summary
            description_embedding = self.model.encode(desc_text)
            entity["name_embedding"] = name_embedding
            entity["description_embedding"] = description_embedding
        return filtered_entities

    def add_entity_summary(self, merged_global_entities: Dict[str, Dict[str, Dict[str, Any]]],
                           per_task_timeout: float = 120.0):
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
            desc="生成摘要（并发）", thread_prefix="summ"
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
        candidates = []
        for t, ent_list in merged_global_entities.items():
            by_name = {}
            for e in ent_list:
                nm = e["name"]
                if nm in by_name:
                    by_name[nm]["description"] = by_name[nm].get("description", "") + e.get("description", "")
                else:
                    by_name[nm] = e

            knn_k = max(5, min(int(len(by_name) / 4), 25))
            estimated_k, _ = compute_weighted_similarity_and_laplacian(by_name, alpha=0.8, knn_k=knn_k)
            n_clusters = max(2, int((estimated_k + len(by_name) / 2) / 2))

            clusters = run_kmeans_clustering(by_name, n_clusters=n_clusters, alpha=0.5)
            candidates.extend(clusters)
        return candidates

    def merge_entities(self, all_candidates_with_info, per_task_timeout: float = 120.0):
        def _empty():
            return {"merges": [], "unmerged": []}

        def _run_group(group):
            try:
                parts = []
                for i, e in enumerate(group):
                    name = e.get("name")
                    summary = e.get("summary") or e.get("description") or ""
                    parts.append(f"实体{i + 1}的名称：{name}\n实体的相关描述为：\n{summary}\n")
                entity_descriptions = "".join(parts)
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
                print(f"❗实体合并失败: {names} -> {e}")
                return _empty()

        results = self._soft_timeout_pool(
            all_candidates_with_info, _run_group, per_task_timeout=per_task_timeout,
            desc="实体合并判断（并发）", thread_prefix="merge"
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

    # ---------- 统一应用规则，深拷贝并返回 ----------
    @staticmethod
    def _apply_entity_rules(
        extraction_results: List[Dict[str, Any]],
        *,
        type_rules: Optional[Dict[str, Dict[str, str]]] = None,   # {entity_name: {old_type: new_type, ...}}
        scope_rules: Optional[Dict[str, str]] = None              # {entity_name: "global"/"local"}
    ) -> Tuple[List[Dict[str, Any]], int, int]:
        """深拷贝 extraction_results，在副本上应用规则并返回。"""
        new_results = deepcopy(extraction_results)
        type_changed = 0
        scope_changed = 0

        for doc in new_results:
            ents = doc.get("entities", [])
            for ent in ents:
                nm = ent.get("name")

                # 类型规则
                if type_rules and nm in type_rules:
                    mapping = type_rules[nm]
                    old_t = ent.get("type")
                    new_t = mapping.get(old_t)
                    if new_t and new_t != old_t:
                        ent["type"] = new_t
                        type_changed += 1

                # scope 规则（小写规范）
                if scope_rules and nm in scope_rules:
                    target = scope_rules[nm]
                    if isinstance(target, str):
                        tgt_norm = target.lower()
                        if tgt_norm in ("global", "local"):
                            tgt = "global" if tgt_norm == "global" else "local"
                            if ent.get("scope") != tgt:
                                ent["scope"] = tgt
                                scope_changed += 1
        return new_results, type_changed, scope_changed

    # ---------- 多类型：集合清洗 & 主类型选择 & 限幅 ----------
    def _sanitize_type_set(self, tset: set) -> List[str]:
        """
        对一个同名实体的类型集合做清洗与裁剪，返回排序后的类型列表（高优先在前）。
        规则：
        - 若包含 Concept 且存在其他类型：删除 Concept（可配置）
        - 若 Action 和 Event 同时存在：把 Action → Event（可配置）
        - 结果按 TYPE_PRIORITY 排序；若有未知类型，排在已知类型之后、Concept 之前
        - 若设置了 MAX_TYPES_PER_NODE，则按顺序截断
        """
        pr_index = {t: i for i, t in enumerate(self.TYPE_PRIORITY)}
        s = set(tset)

        if self.STRIP_CONCEPT_WHEN_OTHERS and "Concept" in s and len(s - {"Concept"}) >= 1:
            s.discard("Concept")

        if self.UNIFY_ACTION_TO_EVENT and "Action" in s and "Event" in s:
            s.discard("Action")  # 只保留 Event

        def score(t: str) -> int:
            if t == "Concept":
                return 10_000_000  # 最低
            return pr_index.get(t, 10_000)  # 未知类型排在已知之后，但高于 Concept

        ordered = sorted(s, key=score)
        if self.MAX_TYPES_PER_NODE is not None and self.MAX_TYPES_PER_NODE > 0:
            ordered = ordered[: self.MAX_TYPES_PER_NODE]
        return ordered

    @staticmethod
    def _collect_type_sets(results: List[Dict[str, Any]]) -> Dict[str, set]:
        m = defaultdict(set)
        for doc in results:
            for ent in doc.get("entities", []):
                nm = ent.get("name")
                tp = ent.get("type")
                if nm and isinstance(tp, str):
                    m[nm].add(tp)
                # 同时兼容已有的多类型字段，合并起来（如果之前就存过）
                if nm and isinstance(ent.get("types"), list):
                    for t in ent["types"]:
                        if isinstance(t, str):
                            m[nm].add(t)
        return m

    def _attach_multilabel(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        ★ 核心：为每个实体写入 `types`（多标签）并设置主类型 `type` 为其中优先级最高的一个。
        - 若 ALLOW_MULTI_TYPE=False，则退化为只保留主类型（与之前单类型一致）。
        """
        new_results = deepcopy(results)
        type_sets = self._collect_type_sets(new_results)

        # 逐实体计算清洗后的类型列表
        per_name_types: Dict[str, List[str]] = {}
        for name, s in type_sets.items():
            cleaned = self._sanitize_type_set(s)
            # 如果清洗后为空（极端情况），至少保留 Concept 兜底
            if not cleaned:
                cleaned = ["Concept"]
            per_name_types[name] = cleaned

        # 写回：types & type（主类型）
        for doc in new_results:
            for ent in doc.get("entities", []):
                nm = ent.get("name")
                if nm in per_name_types:
                    types_list = per_name_types[nm]
                    ent["types"] = types_list if self.ALLOW_MULTI_TYPE else [types_list[0]]
                    ent["type"] = types_list[0]  # 主类型用于兼容下游
        return new_results

    # ---------- 类型细化（并发 + 规则 + 多类型写回） ----------
    def refine_entity_types(self, extraction_results, per_task_timeout: float = 120.0):
        # 统计：实体 -> 类型集合
        entity_type_checker: Dict[str, set] = {}
        for doc in extraction_results:
            for ent in doc.get("entities", []):
                entity_type_checker.setdefault(ent["name"], set()).add(ent["type"])

        # 仅保留多类型
        entity_type_checker = {k: v for k, v in entity_type_checker.items() if len(v) > 1}

        # 规则预过滤：把 Concept → 非 Concept（若存在），Action → Event（可配置）
        type_rules: Dict[str, Dict[str, str]] = {}

        def best_non_concept(ts: set) -> Optional[str]:
            pr_index = {t: i for i, t in enumerate(self.TYPE_PRIORITY)}
            candidates = [t for t in ts if t != "Concept"]
            if not candidates:
                return None
            return min(candidates, key=lambda t: pr_index.get(t, 10_000))

        for name, types in entity_type_checker.items():
            tset = types if isinstance(types, set) else set(types)

            if "Concept" in tset and len(tset) >= 2 and self.STRIP_CONCEPT_WHEN_OTHERS:
                target = best_non_concept(tset) or "Concept"
                type_rules.setdefault(name, {})
                type_rules[name]["Concept"] = target

            if self.UNIFY_ACTION_TO_EVENT and "Action" in tset and "Event" in tset:
                type_rules.setdefault(name, {})
                type_rules[name]["Action"] = "Event"

        # 待检查名单（LLM 验证）
        to_check = [(n, t) for n, t in entity_type_checker.items() if n not in type_rules]

        def _check_entity(item):
            entity, types = item
            ctx = self.prepare_context_by_type(entity_name=entity, extraction_results=extraction_results, types=list(types))
            raw = self.document_parser.validate_entity_type(ctx)
            data = json.loads(correct_json_format(raw))
            rules = {}
            if isinstance(data, dict) and data.get("filtering_rules"):
                for d in data["filtering_rules"]:
                    if isinstance(d, dict):
                        rules.update(d)  # {old_type: new_type}
            return entity, rules

        if to_check:
            results = self._soft_timeout_pool(
                to_check, _check_entity, per_task_timeout=per_task_timeout,
                desc="检查实体类型（并发）", thread_prefix="etype"
            )
            for timeout_or_error, _item, res in results:
                if timeout_or_error or res is None:
                    continue
                entity, rules = res
                if rules:
                    type_rules[entity] = {**type_rules.get(entity, {}), **rules}

        # 先应用规则（在副本上）
        new_results, _type_changed, _ = self._apply_entity_rules(
            extraction_results, type_rules=type_rules, scope_rules=None
        )
        # ★ 写回多类型（不会收敛成单一类型）
        new_results = self._attach_multilabel(new_results)
        return new_results

    # ---------- scope 标准化与收敛（不影响类型多标签） ----------
    @staticmethod
    def _norm_scope(val: Optional[str]) -> Optional[str]:
        """统一 scope 取值到 {'global','local'}，其余返回 None。"""
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
        counts = defaultdict(lambda: {"global": 0, "local": 0})
        for doc in results:
            for ent in doc.get("entities", []):
                nm = ent.get("name")
                sc = GraphPreprocessor._norm_scope(ent.get("scope"))
                if nm and sc in ("global", "local"):
                    counts[nm][sc] += 1
        return counts

    def refine_entity_scope(self, extraction_results, per_task_timeout: float = 120.0,
                            tie_breaker: str = "global"):
        """
        1) 先按 LLM 判定（并发+软超时）；
        2) 全量标准化；
        3) 多数票兜底；平局按 tie_breaker；
        4) 返回深拷贝的新对象（同名不再出现多 scope）。
        """
        # 找出多 scope 的实体
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
                desc="检查实体scope（并发）", thread_prefix="escope"
            )
            for timeout_or_error, _item, res in results:
                if timeout_or_error or res is None:
                    continue
                name, scope_std = res
                if scope_std in ("global", "local"):
                    scope_rules[name] = scope_std

        # 先应用已判定规则
        new_results, _, _ = self._apply_entity_rules(
            extraction_results, type_rules=None, scope_rules=scope_rules
        )

        # 全量标准化
        for doc in new_results:
            for ent in doc.get("entities", []):
                sc = self._norm_scope(ent.get("scope"))
                if sc in ("global", "local"):
                    ent["scope"] = sc

        # 多数票兜底
        counts = self._collect_scope_counts(new_results)
        fallback_rules: Dict[str, str] = {}
        for name, c in counts.items():
            g, l = c["global"], c["local"]
            if g > 0 and l > 0:  # 冲突
                if g > l:
                    target = "global"
                elif l > g:
                    target = "local"
                else:
                    target = "global" if tie_breaker == "global" else "local"
                fallback_rules[name] = target

        if fallback_rules:
            new_results, _, _ = self._apply_entity_rules(
                new_results, type_rules=None, scope_rules=fallback_rules
            )
        return new_results

    # ---------- 其他辅助 ----------
    def get_entity_info(self, entity_name, extraction_results, scope=None, entity_type=None):
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
        context = ""
        for type_ in types:
            context += f"实体类型：{type_}\n"
            results = self.get_entity_info(entity_name, extraction_results, entity_type=type_)
            for result in results:
                context += f"- 实体名称: {result['name']}，相关描述为：{result.get('description','')}\n"
            context += "\n"
        return context

    def prepare_context_by_scope(self, entity_name, extraction_results):
        context = ""
        for scope_ in ["global", "local"]:
            context += f"实体scope：{scope_}\n"
            results = self.get_entity_info(entity_name, extraction_results, scope=scope_)
            for result in results:
                context += f"- 实体名称: {result['name']}，相关描述为：{result.get('description','')}\n"
            context += "\n"
        return context

    # ---------- 实体消歧主流程 ----------
    def run_entity_disambiguation(self, extraction_results):
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

        # 对实体/关系名应用重命名（原地改写即可）
        for result in extraction_results:
            for ent in result.get("entities", []):
                ent["name"] = rename_map.get(ent["name"], ent["name"])
            for rel in result.get("relations", []):
                rel["subject"] = rename_map.get(rel["subject"], rel["subject"])
                rel["object"] = rename_map.get(rel["object"], rel["object"])

        # ★ 重命名后再写回多类型，避免 Concept 污染且保留多标签
        extraction_results = self._attach_multilabel(extraction_results)
        return extraction_results

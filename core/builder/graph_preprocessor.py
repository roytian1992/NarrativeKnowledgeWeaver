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
    Compute a weighted similarity matrix (name vs. description), build a symmetric
    k-NN graph, derive its Laplacian, and estimate cluster count from spectral gaps.

    Args:
        entity_dict (Dict[str, Dict[str, Any]]): name -> { 'name_embedding', 'description_embedding', ... }
        alpha (float): Weight for name vs. description similarities (0..1).
        knn_k (int): Number of neighbors for the k-NN graph.

    Returns:
        Tuple[int, np.ndarray]:
            - estimated_k: Estimated number of clusters from Laplacian eigen-gaps.
            - sim: Weighted similarity matrix.
    """
    names = list(entity_dict.keys())
    name_embs = np.vstack([entity_dict[n]['name_embedding'] for n in names])
    desc_embs = np.vstack([entity_dict[n]['description_embedding'] for n in names])

    sim_name = cosine_similarity(name_embs)
    sim_desc = cosine_similarity(desc_embs)
    sim = alpha * sim_name + (1 - alpha) * sim_desc

    n = sim.shape[0]
    adj = np.zeros((n, n))
    for i in range(n):
        idx = np.argsort(sim[i])[-(knn_k + 1):-1]  # exclude self
        adj[i, idx] = sim[i, idx]
    adj = np.maximum(adj, adj.T)  # symmetrize

    deg = np.diag(adj.sum(axis=1))
    lap = deg - adj

    eigvals = np.linalg.eigvalsh(lap)
    gaps = np.diff(eigvals)
    estimated_k = int(np.argmax(gaps[1:]) + 1)  # skip the first gap
    return estimated_k, sim


def run_kmeans_clustering(entity_dict, n_clusters, alpha=0.8):
    """
    Run KMeans on concatenated (weighted) embeddings to produce clusters.

    Args:
        entity_dict (Dict[str, Dict[str, Any]]): name -> { 'name_embedding', 'description_embedding', ... }
        n_clusters (int): Number of clusters.
        alpha (float): Weight for name vs. description embeddings in concatenation.

    Returns:
        List[List[str]]: Clusters as lists of names; only clusters with size >= 2 are returned.
    """
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
        Detect candidate groups of same-named entities to be merged.
        Uses spectral estimation for cluster count and KMeans for grouping.

        Args:
            merged_global_entities: type -> list of entities (with embeddings)

        Returns:
            List[List[str]]: Candidate groups (lists of names) to consider for merging.
        """
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
        normalize_types: bool = True                                # Apply mapping then normalize type/types
    ) -> Tuple[List[Dict[str, Any]], int, int]:
        """
        Apply type/scope rules on a deep copy of `extraction_results`.

        Enhancements:
        - Merge `type` and `types` for each entity, apply mappings and cleanup once;
        - Cleanup uses the class strategies (drop Concept, Action→Event, sort by priority, cap count);
        - Write back both `types` and primary `type`.

        Returns:
            (new_results, type_changed_count, scope_changed_count)
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

                # 1) Aggregate current types (type ∪ types)
                current_types: List[str] = []
                if isinstance(ent.get("type"), str) and ent["type"].strip():
                    current_types.append(ent["type"].strip())
                if isinstance(ent.get("types"), list):
                    for t in ent["types"]:
                        if isinstance(t, str) and t.strip():
                            current_types.append(t.strip())

                # Deduplicate while preserving order
                seen = set()
                merged_types = []
                for t in current_types:
                    if t not in seen:
                        seen.add(t)
                        merged_types.append(t)

                before_types = list(merged_types)

                # 2) Apply per-entity type mapping (old_type -> new_type)
                if nm in type_rules and isinstance(type_rules[nm], dict):
                    mapping = type_rules[nm]
                    mapped = []
                    for t in merged_types:
                        new_t = mapping.get(t, t)
                        mapped.append(new_t)
                    merged_types = mapped

                # 3) Normalize scope via rules
                if nm in scope_rules and isinstance(scope_rules[nm], str):
                    tgt_norm = scope_rules[nm].strip().lower()
                    if tgt_norm in ("global", "local"):
                        tgt = "global" if tgt_norm == "global" else "local"
                        if ent.get("scope") != tgt:
                            ent["scope"] = tgt
                            scope_changed += 1

                # 4) Normalize type set (drop Concept, unify Action->Event, sort, cap)
                if normalize_types:
                    cleaned_list = self._sanitize_type_set(set(merged_types))
                else:
                    cleaned_list = list(dict.fromkeys(merged_types))

                # Ensure at least one type remains
                if not cleaned_list:
                    cleaned_list = ["Concept"]

                # 5) Write back `types` and primary `type`
                if self.ALLOW_MULTI_TYPE:
                    ent["types"] = cleaned_list
                else:
                    ent["types"] = [cleaned_list[0]]

                old_type = ent.get("type")
                ent["type"] = cleaned_list[0]

                if cleaned_list != before_types or ent["type"] != old_type:
                    type_changed += 1

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
        If ALLOW_MULTI_TYPE is False, only keep the primary type.
        """
        new_results = deepcopy(results)
        type_sets = self._collect_type_sets(new_results)

        per_name_types: Dict[str, List[str]] = {}
        for name, s in type_sets.items():
            cleaned = self._sanitize_type_set(s)
            if not cleaned:
                cleaned = ["Concept"]
            per_name_types[name] = cleaned

        for doc in new_results:
            for ent in doc.get("entities", []):
                nm = ent.get("name")
                if nm in per_name_types:
                    types_list = per_name_types[nm]
                    ent["types"] = types_list if self.ALLOW_MULTI_TYPE else [types_list[0]]
                    ent["type"] = types_list[0]
        return new_results

    # ---------- Type refinement (concurrent + rules + multi-label writeback) ----------
    def refine_entity_types(self, extraction_results, per_task_timeout: float = 120.0):
        """
        Refine entity types by:
        - Collecting multi-typed entities,
        - Applying deterministic rules (Concept removal, Action→Event),
        - Asking the LLM for additional filtering when necessary,
        - Applying rules and writing back multi-label `types` and primary `type`.
        """
        entity_type_checker: Dict[str, set] = {}
        for doc in extraction_results:
            for ent in doc.get("entities", []):
                entity_type_checker.setdefault(ent["name"], set()).add(ent["type"])

        entity_type_checker = {k: v for k, v in entity_type_checker.items() if len(v) > 1}

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
                desc="Validate entity types (concurrent)", thread_prefix="etype"
            )
            for timeout_or_error, _item, res in results:
                if timeout_or_error or res is None:
                    continue
                entity, rules = res
                if rules:
                    type_rules[entity] = {**type_rules.get(entity, {}), **rules}

        new_results, _type_changed, _ = self._apply_entity_rules(
            extraction_results=extraction_results, type_rules=type_rules, scope_rules=None
        )
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
                rel["subject"] = rename_map.get(rel["subject"], rel["subject"])
                rel["object"] = rename_map.get(rel["object"], rel["object"])

        # Re-attach multi-label types post-rename
        extraction_results = self._attach_multilabel(extraction_results)
        return extraction_results

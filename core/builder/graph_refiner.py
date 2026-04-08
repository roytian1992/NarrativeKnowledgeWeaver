import os
import json
import time
import unicodedata
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Literal, Set
from collections import defaultdict, Counter
from itertools import combinations
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED, as_completed
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from core.utils.config import KAGConfig
from core.builder.manager.document_manager import DocumentParser
from core.builder.manager.error_manager import ProblemSolver
from core.utils.format import correct_json_format
from core.utils.general_utils import safe_str, load_json, word_len, truncate_by_word_len
import re

# =========================
# Basic Utils
# =========================

def extract_sentences_with_keyword(
    text: str,
    keyword: str,
    *,
    span: int = 0,
    dedup: bool = True,
) -> List[str]:
    """
    Find ALL sentences containing keyword (case-insensitive), and return a list
    of concatenated sentence windows [idx-span, idx+span] for each match.

    Args:
        text: input text
        keyword: keyword to search for (case-insensitive)
        span: number of neighboring sentences before/after to include
        dedup: whether to deduplicate identical returned windows (keeps order)

    Returns:
        List of concatenated windows (each as a single string)
    """
    if not text or not keyword:
        return []

    # 1) normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # 2) sentence split
    sentences = re.split(r"(?<=[.!?])\s+", text)

    # 3) keyword match
    pattern = re.compile(re.escape(keyword), re.IGNORECASE)

    windows: List[str] = []
    for idx, sent in enumerate(sentences):
        if pattern.search(sent):
            start = max(0, idx - span)
            end = min(len(sentences), idx + span + 1)
            windows.append(" ".join(sentences[start:end]))

    if not dedup:
        return windows

    # stable dedup (preserve order)
    seen = set()
    out: List[str] = []
    for w in windows:
        if w not in seen:
            seen.add(w)
            out.append(w)
    return out

def build_name_type_dict(
    document_results: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, List[str]]]:
    """
    Build:
      {
        entity_name: {
          type1: [document_id1, document_id2, ...],
          type2: [...],
        },
        ...
      }
    Only keep entity_name that appears with >=2 distinct types.
    document keys are deduped (stable order).
    """
    # name -> type -> ordered unique document_ids
    name2type2documents: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
    name2type2seen = defaultdict(lambda: defaultdict(set))

    for document_id, payload in (document_results or {}).items():
        if not isinstance(payload, dict):
            continue
        for ent in (payload.get("entities") or []):
            if not isinstance(ent, dict):
                continue
            name = ent.get("name")
            typ = ent.get("type")
            if not isinstance(name, str) or not name.strip():
                continue
            if not isinstance(typ, str) or not typ.strip():
                continue

            # stable dedup for documents per (name,type)
            if document_id not in name2type2seen[name][typ]:
                name2type2seen[name][typ].add(document_id)
                name2type2documents[name][typ].append(document_id)

    # filter to multi-type names only
    out: Dict[str, Dict[str, List[str]]] = {}
    for name, type2chs in name2type2documents.items():
        if len(type2chs) >= 2:
            out[name] = dict(type2chs)

    return out



def build_typepair_relation_index(relation_type_info: Dict[str, Dict[str, Any]]) -> Dict[Tuple[str, str], Set[str]]:
    """
    Unordered (type_a, type_b) -> {relation_type}
    """
    idx: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
    for rtype, info in relation_type_info.items():
        from_types = info.get("from", []) or []
        to_types = info.get("to", []) or []
        for a in from_types:
            for b in to_types:
                idx[tuple(sorted((a, b)))].add(rtype)
    return dict(idx)


def get_allowed_relations_between_types(
    typepair_index: Dict[Tuple[str, str], Set[str]],
    t1: str,
    t2: str,
) -> List[str]:
    return sorted(list(typepair_index.get(tuple(sorted((t1, t2))), set())))

def _safe_json_loads(x: Any) -> Dict[str, Any]:
    """
    Robustly parse LLM output.
    - If x is dict, return it
    - If x is str, try json.loads, then correct_json_format fallback
    """
    if isinstance(x, dict):
        return x
    if not isinstance(x, str):
        return {}
    try:
        return json.loads(x)
    except Exception:
        try:
            return json.loads(correct_json_format(x))
        except Exception:
            return {}

def _normalize_decision(s: Any) -> str:
    if not isinstance(s, str):
        return ""
    return s.strip().lower()

# =========================
# Similarity & clustering utilities
# =========================
def compute_weighted_similarity_and_laplacian(entity_dict, alpha=0.8, knn_k=40):
    """
    Robust: handle tiny n, cap knn_k, return (estimated_k, similarity_matrix).
    entity_dict: name -> {"name_embedding": ..., "description_embedding": ...}
    """
    names = list(entity_dict.keys())
    n = len(names)
    if n == 0:
        return 0, np.zeros((0, 0))
    if n == 1:
        sim = np.array([[1.0]])
        return 1, sim

    name_embs = np.vstack([entity_dict[nm]["name_embedding"] for nm in names])
    desc_embs = np.vstack([entity_dict[nm]["description_embedding"] for nm in names])

    sim_name = cosine_similarity(name_embs)
    sim_desc = cosine_similarity(desc_embs)
    sim = alpha * sim_name + (1 - alpha) * sim_desc

    k = max(1, min(knn_k, n - 1))
    adj = np.zeros((n, n), dtype=float)
    for i in range(n):
        idx = np.argsort(sim[i])[-(k + 1) : -1]
        if idx.size:
            adj[i, idx] = sim[i, idx]
    adj = np.maximum(adj, adj.T)

    deg = np.diag(adj.sum(axis=1))
    lap = deg - adj

    eigvals = np.linalg.eigvalsh(lap)
    if n < 3:
        return 1, sim

    gaps = np.diff(eigvals)
    inner = gaps[1:]
    if inner.size == 0 or not np.isfinite(inner).any():
        estimated_k = 1
    else:
        estimated_k = int(np.argmax(inner) + 1)

    estimated_k = max(1, min(estimated_k, n))
    return estimated_k, sim


def run_kmeans_clustering(entity_dict, n_clusters, alpha=0.8):
    """
    Robust KMeans: returns list of clusters (each cluster is list[str] names), only keep size>=2 clusters.
    """
    names = list(entity_dict.keys())
    n = len(names)
    if n < 2:
        return []

    name_embs = np.vstack([entity_dict[nm]["name_embedding"] for nm in names])
    desc_embs = np.vstack([entity_dict[nm]["description_embedding"] for nm in names])
    combined = np.hstack([name_embs * alpha, desc_embs * (1 - alpha)])

    if n_clusters is None or n_clusters < 2:
        n_clusters = max(2, int(np.sqrt(n)))
    n_clusters = min(max(2, n_clusters), n)

    try:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
        labels = kmeans.fit_predict(combined)
    except Exception:
        kmeans = KMeans(n_clusters=min(2, n), random_state=42, n_init="auto")
        labels = kmeans.fit_predict(combined)

    groups = defaultdict(list)
    for name, lab in zip(names, labels):
        groups[lab].append(name)

    return [g for g in groups.values() if len(g) >= 2]


# =========================
# Main class
# =========================
class GraphRefiner:
    """
    Refiner for new extraction format:
      document_results: {document_id -> payload}
      payload:
        {
          "ok": bool,
          "document_id": str,
          "document_metadata": dict,
          "chunk_ids": [...],
          "entities": [ {name,type,description?,scope?, ...}, ... ],
          "relations": [ {subject,object,relation_type,...}, ... ],
          "error": str
        }
    """

    def __init__(self, config: KAGConfig, llm, system_prompt: str = "", memory_store=None):
        self.config = config
        self.system_prompt_text = system_prompt or ""
        self.document_parser = DocumentParser(config, llm)
        self.problem_solver = ProblemSolver(config, llm)
        self.model = self._load_embedding_model()
        self.max_worker = int(getattr(self.config.document_processing, "max_workers", 4))
        self.memory_store = memory_store  # Optional[ExtractionMemoryStore]
        self.load_documents_and_schema()
        print("default_scope =", self.type2default_scope)


    def load_documents_and_schema(self) -> None:
        """
        Load:
        - entity schema
        - relation schema
        - entity extraction task config
        - doc2chunks (documents)

        All paths come from KAGConfig:
        global.schema_dir
        global.task_dir
        knowledge_graph_builder.file_path
        """
        import os

        # ---------- dirs from YAML ----------
        schema_dir = getattr(self.config, "global_", None)
        schema_dir = getattr(schema_dir, "schema_dir", "") if schema_dir is not None else ""
        schema_dir = str(schema_dir or "").strip()

        task_dir = getattr(self.config, "global_", None)
        task_dir = getattr(task_dir, "task_dir", "") if task_dir is not None else ""
        task_dir = str(task_dir or "").strip()

        kg_dir = getattr(self.config, "knowledge_graph_builder", None)
        kg_dir = getattr(kg_dir, "file_path", "") if kg_dir is not None else ""
        kg_dir = str(kg_dir or "").strip()

        if not schema_dir:
            raise ValueError("config.global.schema_dir is empty")
        if not task_dir:
            raise ValueError("config.global.task_dir is empty")
        if not kg_dir:
            raise ValueError("config.knowledge_graph_builder.file_path is empty")

        # ---------- file paths ----------
        entity_schema_path = os.path.join(schema_dir, "default_entity_schema.json")
        relation_schema_path = os.path.join(schema_dir, "default_relation_schema.json")

        # 你的任务文件放在 task_dir 下面（不再 hard-code core/task_settings）
        entity_task_path = os.path.join(task_dir, "entity_extraction_task.json")

        doc2chunks_path = os.path.join(kg_dir, "doc2chunks.json")

        # ---------- load ----------
        self.entity_schema = load_json(entity_schema_path)
        self.relation_schema = load_json(relation_schema_path)
        self.entity_extraction_task = load_json(entity_task_path)
        self.documents = load_json(doc2chunks_path)

        # -------------------------
        # Build relation type registry
        # -------------------------
        self.relation_type_info: Dict[str, Dict[str, Any]] = {}
        self.opposite_pairs: Set[frozenset] = set()
        self.opposite_of: Dict[str, str] = {}

        if isinstance(self.relation_schema, dict):
            for group in self.relation_schema.values():
                if not isinstance(group, list):
                    continue
                for r in group:
                    if not isinstance(r, dict):
                        continue
                    rtype = r.get("type")
                    if not isinstance(rtype, str) or not rtype.strip():
                        continue
                    rtype = rtype.strip()

                    if rtype in self.relation_type_info:
                        raise ValueError(f"Duplicate relation_type detected across groups: {rtype}")

                    info = {
                        "from": r.get("from", []) or [],
                        "to": r.get("to", []) or [],
                        "direction": r.get("direction", "directed"),
                        "description": r.get("description", "") or "",
                    }

                    # NEW: opposite (nullable / missing allowed)
                    opp = r.get("opposite", None)
                    if isinstance(opp, str):
                        opp = opp.strip()
                        if not opp:
                            opp = None
                    else:
                        opp = None
                    if opp is not None:
                        info["opposite"] = opp
                        self.opposite_of[rtype] = opp

                    self.relation_type_info[rtype] = info

        for a, b in list(self.opposite_of.items()):
            if b not in self.relation_type_info:
                raise ValueError(f"Relation type '{a}' has opposite='{b}', but '{b}' not found in schema.")
            if a == b:
                raise ValueError(f"Relation type '{a}' has opposite to itself, invalid.")
            self.opposite_pairs.add(frozenset([a, b]))


        self.typepair_index = build_typepair_relation_index(self.relation_type_info)

        # -------------------------
        # Build scope default rules from entity_extraction_task.json
        # Rule:
        # - If type has "default_scope": use it.
        # - Else if scope_rules has ONLY ONE key: default_scope = that key.
        # - Else: no default (leave as-is / let vote or LLM handle).
        # -------------------------
        self.type2default_scope: Dict[str, Optional[str]] = {}

        if isinstance(self.entity_extraction_task, list):
            for task in self.entity_extraction_task:
                if not isinstance(task, dict):
                    continue
                for t in (task.get("types") or []):
                    if not isinstance(t, dict):
                        continue
                    tp = t.get("type")
                    if not isinstance(tp, str) or not tp.strip():
                        continue
                    tp = tp.strip()

                    # 1) explicit default_scope
                    ds = t.get("default_scope", None)
                    ds_norm = self._norm_scope(ds)
                    if ds_norm in ("global", "local"):
                        self.type2default_scope[tp] = ds_norm
                        continue

                    # 2) scope_rules single-key implies default
                    scope_rules = t.get("scope_rules", None)
                    if isinstance(scope_rules, dict):
                        keys = [k for k in scope_rules.keys() if isinstance(k, str)]
                        keys_norm = [self._norm_scope(k) for k in keys]
                        keys_norm = [k for k in keys_norm if k in ("global", "local")]
                        keys_norm = list(dict.fromkeys(keys_norm))  # stable dedup
                        if len(keys_norm) == 1:
                            self.type2default_scope[tp] = keys_norm[0]
                            continue

                    self.type2default_scope[tp] = None

        # -------------------------
        # Build dynamic entity type rules from default_entity_schema.json
        # -------------------------
        if not isinstance(self.entity_schema, list):
            raise ValueError("default_entity_schema.json must be a JSON list of entity type specs.")

        # type -> category
        self.type2category: Dict[str, str] = {}
        # type -> schema order index (stable tie-breaker)
        self.type2order: Dict[str, int] = {}

        for i, spec in enumerate(self.entity_schema):
            if not isinstance(spec, dict):
                continue

            t = spec.get("type")
            if not isinstance(t, str) or not t.strip():
                continue
            t = t.strip()

            cat = spec.get("category")
            if not isinstance(cat, str) or not cat.strip():
                cat = "general_semantic"
            cat = cat.strip()

            self.type2category[t] = cat
            self.type2order[t] = i

            # Ensure we have an entry in type2default_scope even if task file omitted it
            if t not in self.type2default_scope:
                self.type2default_scope[t] = None

        # Allowed types are exactly those defined in schema
        self.ALLOWED_TYPES: Tuple[str, ...] = tuple(self.type2category.keys())
        if not self.ALLOWED_TYPES:
            raise ValueError("No entity types found in default_entity_schema.json")

        # category priority: induced > referential > time > general_semantic
        self.CATEGORY_PRIORITY: Dict[str, int] = {
            "induced": 0,
            "referential": 1,
            "time": 2,
            "general_semantic": 3,
        }

        # general_semantic fallback type: schema must contain exactly one
        semantic_types = [t for t, c in self.type2category.items() if c == "general_semantic"]
        if len(semantic_types) != 1:
            raise ValueError(
                f"Schema must contain exactly 1 general_semantic type, got {len(semantic_types)}: {semantic_types}"
            )
        self.GENERAL_SEMANTIC_TYPE: str = semantic_types[0]

        # Define full priority list of types (stable)
        def _type_rank(t: str) -> Tuple[int, int]:
            cat_ = self.type2category.get(t, "general_semantic")
            return (self.CATEGORY_PRIORITY.get(cat_, 99), self.type2order.get(t, 10_000_000))

        self.TYPE_PRIORITY: Tuple[str, ...] = tuple(sorted(self.ALLOWED_TYPES, key=_type_rank))

        # Convenience: induced types set
        self.INDUCED_TYPES: Set[str] = {t for t, c in self.type2category.items() if c == "induced"}

    # ------------------------------------------------------------------
    # Memory writing helpers
    # ------------------------------------------------------------------

    def _write_type_memories(self, audits: List[Dict[str, Any]]) -> None:
        """
        Generate ExtractionMemoryStore entries from refine_entity_types audit results.

        - decision="drop"  → type_rule memory
        - decision="keep"  → alias memory (old_name ↔ new_name)
        """
        if self.memory_store is None:
            return

        for audit in (audits or []):
            if not isinstance(audit, dict):
                continue

            entity_name = str(audit.get("entity_name", "")).strip()
            major_type = str(audit.get("major_type", "")).strip()
            minor_type = str(audit.get("minor_type", "")).strip()
            decision = str(audit.get("decision", "")).strip()
            reason = str(audit.get("reason", "")).strip()

            if not entity_name:
                continue

            if decision == "drop":
                # type_rule: entity should be typed as major_type, not minor_type
                content = (
                    f"'{entity_name}' should be typed as {major_type}, not {minor_type}"
                    + (f": {reason}" if reason else "")
                )
                self.memory_store.add({
                    "type": "type_rule",
                    "content": content,
                    "keywords": [entity_name],
                    "confidence": 0.8,
                    "source": "refine_entity_types",
                    "memory_scope": "entity_extraction",
                })

            elif decision == "keep":
                # alias: the old name and new name are distinct entities
                for action in (audit.get("actions") or []):
                    if not isinstance(action, dict):
                        continue
                    new_name = str(action.get("new_name", "")).strip()
                    if new_name and new_name != entity_name:
                        content = (
                            f"'{entity_name}' ({major_type}) and '{new_name}' ({minor_type}) "
                            f"are distinct entities; use '{new_name}' when referring to the {minor_type}"
                        )
                        self.memory_store.add({
                            "type": "alias",
                            "content": content,
                            "keywords": [entity_name, new_name],
                            "confidence": 0.85,
                            "source": "refine_entity_types",
                            "memory_scope": "entity_extraction",
                        })

    def _format_relation_list_desc(self, relation_types):
        type_descs = []
        for rtype in relation_types:
            desc = self.relation_type_info[rtype]["description"]
            # direction = self.relation_type_info[rtype].get("direction", "directed")
            # symbol = "-->" if direction == "directed" else "<-->"
            # rule = "/".join(self.relation_type_info[rtype].get("from", [])) + symbol + "/".join(self.relation_type_info[rtype].get("to", []))
            type_descs.append(f"{rtype}: {desc}")
        return "\n".join(type_descs)

    # ---------- concurrency helper (soft timeout) ----------
    def _soft_timeout_pool(
        self,
        work_items,
        submit_fn,
        *,
        per_task_timeout: float = 180.0,
        desc: str = "Concurrent tasks",
        thread_prefix: str = "pool",
    ):
        executor = ThreadPoolExecutor(max_workers=self.max_worker, thread_name_prefix=thread_prefix)
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

                for f in done:
                    item = fut_info[f]["item"]
                    try:
                        res = f.result()
                        results.append((False, item, res))
                    except Exception as e:
                        results.append((True, item, e))
                    pbar.update(1)
                    fut_info.pop(f, None)

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

    # ---------- embedding model ----------
    def _load_embedding_model(self):
        from core.model_providers.openai_embedding import OpenAIEmbeddingModel
        return OpenAIEmbeddingModel(self.config.embedding)

    # ---------- small utils ----------
    def _type_prio(self, t: str) -> int:
        # lower is higher priority
        cat = self.type2category.get(t, "general_semantic")
        return self.CATEGORY_PRIORITY.get(cat, 10_000_000) * 1_000_000 + self.type2order.get(t, 10_000_000)

    def _sanitize_type(self, t: Any) -> Optional[str]:
        if not isinstance(t, str):
            return None
        tt = t.strip()
        if not tt:
            return None
        if tt not in self.ALLOWED_TYPES:
            return None
        return tt

    def _best_type(self, types: List[str]) -> str:
        cands = [t for t in types if isinstance(t, str) and t.strip() in self.ALLOWED_TYPES]
        if not cands:
            return self.GENERAL_SEMANTIC_TYPE
        return sorted(cands, key=self._type_prio)[0]


    @staticmethod
    def _norm_scope(val: Any) -> Optional[str]:
        if not isinstance(val, str):
            return None
        v = val.strip().lower()
        mapping = {
            "global": "global",
            "全局": "global",
            "整体": "global",
            "总体": "global",
            "local": "local",
            "局部": "local",
            "片段": "local",
            "场景": "local",
        }
        if v in mapping:
            return mapping[v]
        if v == "global":
            return "global"
        if v == "local":
            return "local"
        return None

    def _lexical_postpass_enabled_types(self) -> Set[str]:
        # Keep the lexical post-pass conservative in the first version.
        return {"Object", "Concept"}

    def _current_language(self) -> str:
        for attr in ("global_config", "global_"):
            cfg = getattr(self.config, attr, None)
            lang = safe_str(getattr(cfg, "language", "")).strip().lower() if cfg is not None else ""
            if lang in {"zh", "en"}:
                return lang
        return "en"

    def _lexical_generic_suffixes(self) -> List[str]:
        zh_suffixes = [
            "设备",
            "主机",
            "装置",
            "模块",
            "终端",
            "单元",
            "机体",
            "机器",
            "平台",
        ]
        en_suffixes = [
            "DEVICE",
            "HOST",
            "MODULE",
            "TERMINAL",
            "UNIT",
            "MACHINE",
            "PLATFORM",
        ]
        ordered = zh_suffixes + en_suffixes if self._current_language() == "zh" else en_suffixes + zh_suffixes
        seen = set()
        out: List[str] = []
        for suffix in sorted(ordered, key=len, reverse=True):
            if suffix not in seen:
                seen.add(suffix)
                out.append(suffix)
        return out

    @staticmethod
    def _normalize_name_for_lexical(name: Any) -> str:
        text = unicodedata.normalize("NFKC", safe_str(name))
        text = text.strip().upper()
        text = re.sub(r"[\s·•\-_]+", "", text)
        text = re.sub(r"[()（）\\[\\]{}<>《》“”\"'`]", "", text)
        text = re.sub(r"[，,。！？!?：:；;/\\\\]", "", text)
        return text

    def _strip_lexical_suffixes(self, normalized_name: str) -> str:
        core = safe_str(normalized_name).strip()
        if not core:
            return core
        changed = True
        suffixes = self._lexical_generic_suffixes()
        while changed and core:
            changed = False
            for suffix in suffixes:
                if core.endswith(suffix) and len(core) > len(suffix):
                    core = core[: -len(suffix)]
                    changed = True
                    break
        return core

    @staticmethod
    def _extract_lexical_anchors(normalized_name: str) -> List[str]:
        anchors: List[str] = []
        for token in re.findall(r"[A-Z0-9]+", safe_str(normalized_name)):
            token = token.strip().upper()
            if not token:
                continue
            has_alpha = bool(re.search(r"[A-Z]", token))
            has_digit = bool(re.search(r"\d", token))
            if has_alpha and (has_digit or len(token) >= 4):
                anchors.append(token)
        # stable dedup
        seen = set()
        out: List[str] = []
        for token in anchors:
            if token not in seen:
                seen.add(token)
                out.append(token)
        return out

    def _build_lexical_signature(self, name: Any) -> Dict[str, Any]:
        normalized_name = self._normalize_name_for_lexical(name)
        core_name = self._strip_lexical_suffixes(normalized_name)
        return {
            "raw_name": safe_str(name).strip(),
            "normalized_name": normalized_name,
            "core_name": core_name,
            "anchors": self._extract_lexical_anchors(normalized_name),
        }

    @staticmethod
    def _extract_model_identifiers(name: Any) -> Set[str]:
        normalized = GraphRefiner._normalize_name_for_lexical(name)
        if not normalized:
            return set()
        identifiers: Set[str] = set()
        for token in re.findall(r"[A-Z0-9]+", normalized):
            token = token.strip().upper()
            if not token:
                continue
            has_alpha = bool(re.search(r"[A-Z]", token))
            has_digit = bool(re.search(r"\d", token))
            if has_alpha and has_digit and len(token) >= 3:
                identifiers.add(token)
        return identifiers

    @staticmethod
    def _is_model_family_name(name: Any) -> bool:
        normalized = GraphRefiner._normalize_name_for_lexical(name)
        if not normalized:
            return False
        family_markers = ("系列", "SERIES", "FAMILY", "LINE")
        return any(marker in normalized for marker in family_markers)

    def _should_block_model_merge(self, alias: str, canonical: str) -> Tuple[bool, str]:
        alias_ids = self._extract_model_identifiers(alias)
        canonical_ids = self._extract_model_identifiers(canonical)

        if alias_ids and canonical_ids and alias_ids != canonical_ids:
            return True, (
                "model_identifier_conflict:"
                f" alias_ids={sorted(alias_ids)}, canonical_ids={sorted(canonical_ids)}"
            )

        alias_is_family = self._is_model_family_name(alias)
        canonical_is_family = self._is_model_family_name(canonical)

        if alias_ids and canonical_is_family:
            return True, (
                "specific_model_to_family_blocked:"
                f" alias_ids={sorted(alias_ids)}, canonical={canonical}"
            )
        if canonical_ids and alias_is_family:
            return True, (
                "family_to_specific_blocked:"
                f" canonical_ids={sorted(canonical_ids)}, alias={alias}"
            )

        return False, ""

    @staticmethod
    def _char_ngrams(text: str, n: int) -> Set[str]:
        if not text:
            return set()
        if len(text) <= n:
            return {text}
        return {text[i : i + n] for i in range(len(text) - n + 1)}

    @staticmethod
    def _lcs_length(a: str, b: str) -> int:
        if not a or not b:
            return 0
        if len(a) < len(b):
            a, b = b, a
        prev = [0] * (len(b) + 1)
        for ca in a:
            cur = [0]
            for j, cb in enumerate(b, 1):
                if ca == cb:
                    cur.append(prev[j - 1] + 1)
                else:
                    cur.append(max(prev[j], cur[-1]))
            prev = cur
        return prev[-1]

    def _compute_lexical_pair_features(self, sig_a: Dict[str, Any], sig_b: Dict[str, Any]) -> Dict[str, Any]:
        norm_a = safe_str(sig_a.get("normalized_name"))
        norm_b = safe_str(sig_b.get("normalized_name"))
        core_a = safe_str(sig_a.get("core_name"))
        core_b = safe_str(sig_b.get("core_name"))
        anchors_a = list(sig_a.get("anchors") or [])
        anchors_b = list(sig_b.get("anchors") or [])
        anchor_overlap = sorted(set(anchors_a) & set(anchors_b))
        shorter, longer = sorted([norm_a, norm_b], key=len)
        shorter_contained = bool(shorter) and shorter in longer
        lcs_len = self._lcs_length(norm_a, norm_b)
        shorter_len = min(len(norm_a), len(norm_b)) or 1
        lcs_ratio = float(lcs_len) / float(shorter_len)
        grams_a = self._char_ngrams(norm_a, 2) | self._char_ngrams(norm_a, 3)
        grams_b = self._char_ngrams(norm_b, 2) | self._char_ngrams(norm_b, 3)
        denom = len(grams_a | grams_b) or 1
        char_jaccard = float(len(grams_a & grams_b)) / float(denom)
        extra_length = max(len(longer) - len(shorter), 0)
        return {
            "exact_match": bool(norm_a and norm_a == norm_b),
            "core_match": bool(core_a and core_a == core_b),
            "anchor_match": bool(anchor_overlap),
            "anchor_conflict": bool(anchors_a and anchors_b and not anchor_overlap),
            "anchor_overlap": anchor_overlap,
            "shorter_contained": shorter_contained,
            "containment_ratio": 1.0 if shorter_contained else lcs_ratio,
            "lcs_ratio": lcs_ratio,
            "char_jaccard": char_jaccard,
            "extra_length": extra_length,
            "normalized_name_a": norm_a,
            "normalized_name_b": norm_b,
            "core_name_a": core_a,
            "core_name_b": core_b,
            "anchors_a": anchors_a,
            "anchors_b": anchors_b,
        }

    @staticmethod
    def _lexical_pair_is_candidate(features: Dict[str, Any]) -> Tuple[bool, str]:
        if not isinstance(features, dict):
            return False, "invalid_features"
        if features.get("anchor_conflict"):
            return False, "anchor_conflict"
        if features.get("exact_match"):
            return True, "exact_match"
        if features.get("core_match"):
            return True, "core_match"
        if features.get("anchor_match") and features.get("shorter_contained") and int(features.get("extra_length", 0) or 0) <= 2:
            return True, "anchor_contained_short_extra"
        return False, "no_rule_matched"

    @staticmethod
    def _render_lexical_feature_line(features: Dict[str, Any]) -> str:
        return (
            f"exact_match={bool(features.get('exact_match'))}, "
            f"core_match={bool(features.get('core_match'))}, "
            f"anchor_match={bool(features.get('anchor_match'))}, "
            f"anchor_conflict={bool(features.get('anchor_conflict'))}, "
            f"containment_ratio={float(features.get('containment_ratio', 0.0)):.3f}, "
            f"lcs_ratio={float(features.get('lcs_ratio', 0.0)):.3f}, "
            f"char_jaccard={float(features.get('char_jaccard', 0.0)):.3f}, "
            f"extra_length={int(features.get('extra_length', 0) or 0)}"
        )

    def _compress_rename_map(self, rename_map: Dict[str, str]) -> Dict[str, str]:
        if not isinstance(rename_map, dict) or not rename_map:
            return {}

        def _resolve(name: str) -> str:
            cur = safe_str(name).strip()
            seen: Set[str] = set()
            while cur in rename_map and cur not in seen:
                seen.add(cur)
                nxt = safe_str(rename_map.get(cur)).strip()
                if not nxt or nxt == cur:
                    break
                cur = nxt
            return cur

        compressed: Dict[str, str] = {}
        for alias, canonical in rename_map.items():
            alias_s = safe_str(alias).strip()
            if not alias_s:
                continue
            resolved = _resolve(alias_s)
            if resolved and resolved != alias_s:
                compressed[alias_s] = resolved
        return compressed

    def _build_lexical_candidate_groups(
        self,
        type2items: Dict[str, Dict[str, Dict[str, Any]]],
        rename_map: Dict[str, str],
    ) -> Tuple[List[List[Dict[str, Any]]], List[Dict[str, Any]]]:
        candidate_groups: List[List[Dict[str, Any]]] = []
        audits: List[Dict[str, Any]] = []
        eligible_types = self._lexical_postpass_enabled_types()
        renamed_aliases = set(rename_map.keys())

        for tp, by_name in (type2items or {}).items():
            if tp not in eligible_types:
                continue
            active_names = [nm for nm in by_name.keys() if nm not in renamed_aliases]
            if len(active_names) < 2:
                continue

            info_by_name: Dict[str, Dict[str, Any]] = {}
            for nm in active_names:
                info = dict(by_name.get(nm) or {})
                info["lexical_signature"] = self._build_lexical_signature(nm)
                info_by_name[nm] = info

            buckets: Dict[Tuple[str, str], List[str]] = defaultdict(list)
            for nm, info in info_by_name.items():
                sig = info.get("lexical_signature") or {}
                normalized = safe_str(sig.get("normalized_name")).strip()
                core = safe_str(sig.get("core_name")).strip()
                if normalized:
                    buckets[("normalized", normalized)].append(nm)
                if core:
                    buckets[("core", core)].append(nm)
                for anchor in sig.get("anchors") or []:
                    if anchor:
                        buckets[("anchor", anchor)].append(nm)

            pair_map: Dict[Tuple[str, str], Dict[str, Any]] = {}
            for (bucket_kind, bucket_key), names in buckets.items():
                uniq_names: List[str] = []
                seen_names: Set[str] = set()
                for nm in names:
                    if nm not in seen_names:
                        seen_names.add(nm)
                        uniq_names.append(nm)

                if len(uniq_names) < 2:
                    continue

                if len(uniq_names) > 32:
                    audits.append(
                        {
                            "kind": "lexical_bucket_skip",
                            "timestamp": time.time(),
                            "type": tp,
                            "bucket_kind": bucket_kind,
                            "bucket_key": bucket_key,
                            "bucket_size": len(uniq_names),
                            "reason": "bucket_too_large",
                        }
                    )
                    continue

                for name_a, name_b in combinations(sorted(uniq_names), 2):
                    pair_key = tuple(sorted((name_a, name_b)))
                    features = self._compute_lexical_pair_features(
                        info_by_name[name_a]["lexical_signature"],
                        info_by_name[name_b]["lexical_signature"],
                    )
                    is_candidate, trigger = self._lexical_pair_is_candidate(features)
                    if not is_candidate:
                        continue

                    if pair_key not in pair_map:
                        pair_map[pair_key] = {
                            "kind": "lexical_candidate_pair",
                            "timestamp": time.time(),
                            "type": tp,
                            "name_a": pair_key[0],
                            "name_b": pair_key[1],
                            "trigger": trigger,
                            "bucket_hits": [f"{bucket_kind}:{bucket_key}"],
                            "features": features,
                        }
                    else:
                        pair_map[pair_key]["bucket_hits"].append(f"{bucket_kind}:{bucket_key}")

            audits.extend(pair_map.values())
            if not pair_map:
                continue

            adjacency: Dict[str, Set[str]] = defaultdict(set)
            for name_a, name_b in pair_map.keys():
                adjacency[name_a].add(name_b)
                adjacency[name_b].add(name_a)

            visited: Set[str] = set()
            for start in sorted(adjacency.keys()):
                if start in visited:
                    continue
                stack = [start]
                component: List[str] = []
                while stack:
                    cur = stack.pop()
                    if cur in visited:
                        continue
                    visited.add(cur)
                    component.append(cur)
                    stack.extend(sorted(adjacency.get(cur, set()) - visited))

                component = sorted(component)
                if len(component) < 2:
                    continue
                if len(component) > 8:
                    audits.append(
                        {
                            "kind": "lexical_group_skip",
                            "timestamp": time.time(),
                            "type": tp,
                            "group": component,
                            "reason": "group_too_large",
                        }
                    )
                    continue

                candidate_groups.append([info_by_name[nm] for nm in component if nm in info_by_name])
                audits.append(
                    {
                        "kind": "lexical_candidate_group",
                        "timestamp": time.time(),
                        "type": tp,
                        "group": component,
                    }
                )

        return candidate_groups, audits

    # =========================
    # 1) refine entity types (with LLM)
    # =========================
    def refine_entity_types(
        self,
        document_results: Dict[str, Dict[str, Any]],
        *,
        per_task_timeout: float = 120.0,  # 当前实现未做线程级 hard timeout
        max_workers: Optional[int] = None,
        context_span: int = 1,
        max_context_words: int = 2000,
        blank_description_on_drop: bool = True,
        dump_audit: bool = True,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Corrected logic aligned with your prompt:

        For each entity_name with multiple types:
        - major_type = most frequent type across documents
        - for each minor_type:
            - ask LLM override check
                - decision="keep": keep minor_type (no type change), but MUST create new_name and rename entity in minor documents
                - decision="drop": override type in minor documents to major_type (no rename). Then fix impacted relations via schema fixer.

        NEW RULE (your request):
        - If one of the conflicting types is Event, DO NOT run check_entity_type for it.
        - Instead, directly rename the Event entity instance(s) via self.problem_solver.rename_event.
        - rename_event input text = ent["description"] + "\\n\\n" + gathered_text
        - After renaming Event, remove "Event" from the conflict set and continue resolving remaining type conflicts.

        Writes audit JSON (optional) to:
        {config.knowledge_graph_builder.filepath}/refine_entity_types_audit_YYYYmmdd_HHMMSS.json
        """
        results = deepcopy(document_results)
        name_type_dict = build_name_type_dict(results)  # {name: {type: [document_ids...]}}
        if not name_type_dict:
            return results

        workers = max_workers or self.max_worker or 4

        # ------------------------------------------------------------
        # Helpers
        # ------------------------------------------------------------
        def _gather_related_text(entity_name: str, documents: List[str]) -> str:
            ctx: List[str] = []
            for ck in documents:
                doc = (self.documents or {}).get(ck, {})
                chunks = doc.get("chunks", []) or []
                for ch in chunks:
                    content = ch.get("content", "")
                    if not isinstance(content, str) or not content.strip():
                        continue
                    ctx.extend(
                        extract_sentences_with_keyword(
                            text=content,
                            keyword=entity_name,
                            span=context_span,
                            dedup=True,
                        )
                    )
            text = "\n".join([s for s in ctx if isinstance(s, str) and s.strip()]).strip()
            if not text:
                return ""
            if word_len(text, lang="auto") > max_context_words:
                text = truncate_by_word_len(text, max_context_words, lang="auto")
            return text

        def _relation_key(r: Dict[str, Any], idx: int) -> str:
            rid = r.get("rid")
            if isinstance(rid, str) and rid.strip():
                return rid.strip()
            return f"__idx__:{idx}"

        def _apply_relation_rewrite_in_place(rel: Dict[str, Any], out: Dict[str, Any]) -> None:
            # Only overwrite safe fields; keep rid/conf/provenance intact.
            for k in ["subject", "object", "relation_type", "relation_name", "description", "persistence"]:
                if k in out:
                    rel[k] = out[k]

        def _truncate_raw(x: Any, limit: int = 4000) -> str:
            if x is None:
                return ""
            if isinstance(x, (dict, list)):
                try:
                    x = json.dumps(x, ensure_ascii=False)
                except Exception:
                    x = str(x)
            if not isinstance(x, str):
                x = str(x)
            s = x.strip()
            if len(s) <= limit:
                return s
            return s[:limit] + "..."

        def _dump_audit_json(audits: List[Dict[str, Any]]) -> str:
            base = self.config.knowledge_graph_builder.file_path
            save_dir = os.path.join(base, "audit_logs")
            os.makedirs(save_dir, exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
            path = os.path.join(save_dir, f"refine_entity_types_audit.json")
            payload = {
                "step": "refine_entity_types",
                "created_at": ts,
                "num_entities_checked": len(name_type_dict),
                "num_audit_items": len(audits),
                "audits": audits,
            }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            return path

        # ------------------------------------------------------------
        # Worker: compute patches + audits (no mutation outside local reads)
        # ------------------------------------------------------------
        def _process_one_entity(entity_name: str, type2documents_in: Dict[str, List[str]]) -> Dict[str, Any]:
            patches = {
                "entity_type_updates": [],
                "entity_desc_blank": [],
                "entity_renames": [],
                "relation_endpoint_renames": [],
                "relation_deletes": [],
                "relation_rewrites": [],
            }
            audits: List[Dict[str, Any]] = []

            type2documents = dict(type2documents_in)
            type_counts = {tp: len(chs) for tp, chs in type2documents.items()}

            # If >2 types and Concept exists: absorb Concept into smallest non-Concept bucket
            if len(type2documents) > 2 and "Concept" in type2documents:
                non_concept = [t for t in type2documents.keys() if t != "Concept"]
                if non_concept:
                    absorb_to = min(non_concept, key=lambda t: type_counts.get(t, 0))
                    type2documents[absorb_to] = list(type2documents.get(absorb_to, [])) + list(type2documents["Concept"])
                    type2documents.pop("Concept", None)
                    type_counts = {tp: len(chs) for tp, chs in type2documents.items()}

            if len(type2documents) < 2:
                return {"patches": patches, "audits": audits}

            # ============================================================
            # NEW: If this name has any induced types in conflicts, rename those induced instances directly
            # ============================================================
            induced_types_here = [t for t in list(type2documents.keys()) if t in self.INDUCED_TYPES]
            if induced_types_here:
                for induced_type in induced_types_here:
                    induced_documents = list(type2documents.get(induced_type, []) or [])
                    if not induced_documents:
                        continue

                    gathered = _gather_related_text(entity_name, induced_documents)

                    for ck in induced_documents:
                        payload = results.get(ck, {})
                        if not isinstance(payload, dict):
                            continue

                        # locate matching induced entity instance (best effort)
                        target_ent = None
                        for e in (payload.get("entities") or []):
                            if not isinstance(e, dict):
                                continue
                            if e.get("name") == entity_name and e.get("type") == induced_type:
                                target_ent = e
                                break
                        if target_ent is None:
                            for e in (payload.get("entities") or []):
                                if isinstance(e, dict) and e.get("name") == entity_name:
                                    target_ent = e
                                    break

                        ent_desc = safe_str(target_ent.get("description", "") if isinstance(target_ent, dict) else "").strip()
                        rename_text = (ent_desc + "\n\n" + gathered).strip() if gathered else ent_desc

                        raw_rename = None
                        new_name = None
                        rename_reason = ""
                        try:
                            raw_rename = self.problem_solver.rename_entity(
                                text=rename_text,
                                entity_name=entity_name,
                                entity_type=induced_type,
                            )
                            data = _safe_json_loads(raw_rename)
                            new_name = data.get("new_name")
                            rename_reason = data.get("reason") or data.get("rationale") or ""
                        except Exception as e:
                            raw_rename = {"error": str(e)}
                            new_name = None

                        audit_item = {
                            "entity_name": entity_name,
                            "major_type": None,
                            "minor_type": induced_type,
                            "major_percentage": None,
                            "decision": "rename_induced",
                            "reason": rename_reason if isinstance(rename_reason, str) else "",
                            "llm_raw": _truncate_raw(raw_rename),
                            "minor_documents": [ck],
                            "timestamp": time.time(),
                            "actions": [],
                            "relation_fixes": [],
                        }

                        if isinstance(new_name, str):
                            new_name = new_name.strip()
                        else:
                            new_name = ""

                        if new_name and new_name != entity_name:
                            patches["entity_renames"].append((ck, entity_name, new_name))
                            patches["relation_endpoint_renames"].append((ck, entity_name, new_name))
                            audit_item["actions"].append(
                                {
                                    "document_id": ck,
                                    "action": "rename_induced",
                                    "old_name": entity_name,
                                    "new_name": new_name,
                                    "type_kept": induced_type,
                                }
                            )
                        else:
                            audit_item["actions"].append(
                                {
                                    "document_id": ck,
                                    "action": "rename_induced_noop",
                                    "old_name": entity_name,
                                    "new_name": new_name or entity_name,
                                    "type_kept": induced_type,
                                }
                            )

                        audits.append(audit_item)

                    # remove this induced type from conflicts and continue
                    type2documents.pop(induced_type, None)
                    type_counts = {tp: len(chs) for tp, chs in type2documents.items()}

                if len(type2documents) < 2:
                    return {"patches": patches, "audits": audits}

            # ------------------------------------------------------------
            # Original logic for non-Event type conflicts
            # ------------------------------------------------------------
            major_type = max(type_counts.items(), key=lambda kv: kv[1])[0]
            minor_types = [t for t in type2documents.keys() if t != major_type]

            for minor_type in minor_types:
                minor_documents = type2documents.get(minor_type, []) or []
                if not minor_documents:
                    continue

                minor_count = len(minor_documents)
                major_count = type_counts.get(major_type, 0)
                denom = max(1, minor_count + major_count)
                major_percentage = round(major_count / denom * 100, 1)

                related_text = _gather_related_text(entity_name, minor_documents)

                raw = self.problem_solver.check_entity_type(
                    text=related_text,
                    entity_name=entity_name,
                    minority_type=minor_type,
                    majority_type=major_type,
                    majority_percentage=major_percentage,
                )
                data = _safe_json_loads(raw)
                decision = _normalize_decision(data.get("decision"))
                reason = data.get("reason") or data.get("rationale") or ""

                audit_item: Dict[str, Any] = {
                    "entity_name": entity_name,
                    "major_type": major_type,
                    "minor_type": minor_type,
                    "major_percentage": major_percentage,
                    "decision": decision,
                    "reason": reason if isinstance(reason, str) else "",
                    "llm_raw": _truncate_raw(raw),
                    "minor_documents": list(minor_documents),
                    "timestamp": time.time(),
                    "actions": [],
                    "relation_fixes": [],
                }

                # -----------------------------
                # KEEP: keep minority type, but MUST rename minority usage to new_name
                # -----------------------------
                if decision == "keep":
                    new_name = data.get("new_name")
                    if not isinstance(new_name, str) or not new_name.strip():
                        # keep without new_name is unsafe: skip
                        audit_item["decision"] = "keep_missing_new_name"
                        audits.append(audit_item)
                        continue
                    new_name = new_name.strip()

                    for ck in minor_documents:
                        patches["entity_renames"].append((ck, entity_name, new_name))
                        patches["relation_endpoint_renames"].append((ck, entity_name, new_name))
                        audit_item["actions"].append(
                            {
                                "document_id": ck,
                                "action": "rename_keep_minority",
                                "old_name": entity_name,
                                "new_name": new_name,
                                "type_kept": minor_type,
                            }
                        )

                    audits.append(audit_item)
                    continue

                # -----------------------------
                # DROP: override minority type to majority type (no rename)
                # then fix impacted relations under new endpoint types
                # -----------------------------
                if decision == "drop":
                    for ck in minor_documents:
                        patches["entity_type_updates"].append((ck, entity_name, major_type))
                        if blank_description_on_drop:
                            patches["entity_desc_blank"].append((ck, entity_name))

                        audit_item["actions"].append(
                            {
                                "document_id": ck,
                                "action": "override_type_drop",
                                "name": entity_name,
                                "old_type": minor_type,
                                "new_type": major_type,
                                "blank_description": bool(blank_description_on_drop),
                            }
                        )

                        payload = results.get(ck, {})
                        if not isinstance(payload, dict):
                            continue
                        ents = payload.get("entities", []) or []
                        rels = payload.get("relations", []) or []

                        # Build name->type map, but override entity_name to major_type for schema checking
                        name2type: Dict[str, str] = {}
                        for e in ents:
                            if not isinstance(e, dict):
                                continue
                            nm = e.get("name")
                            tp = e.get("type")
                            if isinstance(nm, str) and nm.strip() and isinstance(tp, str) and tp.strip():
                                name2type[nm] = tp
                        name2type[entity_name] = major_type  # override for checking

                        # impacted relations
                        impacted_rel_indices = []
                        for i, r in enumerate(rels):
                            if not isinstance(r, dict):
                                continue
                            if r.get("subject") == entity_name or r.get("object") == entity_name:
                                impacted_rel_indices.append(i)

                        if not impacted_rel_indices:
                            continue

                        feedback = (
                            f'The entity "{entity_name}" type in this document is overridden to "{major_type}" '
                            f"based on global usage. Please fix each impacted relation if it becomes invalid under the schema."
                        )

                        for idx in impacted_rel_indices:
                            r = rels[idx]
                            rel_key = _relation_key(r, idx)

                            subj = r.get("subject")
                            obj = r.get("object")
                            if not isinstance(subj, str) or not isinstance(obj, str):
                                patches["relation_deletes"].append((ck, rel_key))
                                audit_item["relation_fixes"].append(
                                    {
                                        "document_id": ck,
                                        "rel_key": rel_key,
                                        "decision": "drop_malformed",
                                        "before": None,
                                        "after": None,
                                        "reason": "subject/object not string",
                                        "llm_raw": "",
                                    }
                                )
                                continue

                            subj_type = name2type.get(subj)
                            obj_type = name2type.get(obj)
                            before = {
                                "subject": subj,
                                "object": obj,
                                "relation_type": r.get("relation_type"),
                                "relation_name": r.get("relation_name"),
                                "description": r.get("description"),
                                "persistence": r.get("persistence"),
                            }

                            if not isinstance(subj_type, str) or not isinstance(obj_type, str):
                                patches["relation_deletes"].append((ck, rel_key))
                                audit_item["relation_fixes"].append(
                                    {
                                        "document_id": ck,
                                        "rel_key": rel_key,
                                        "decision": "drop_missing_type",
                                        "before": before,
                                        "after": None,
                                        "reason": "endpoint type missing in document entities",
                                        "llm_raw": "",
                                    }
                                )
                                continue

                            allowed = get_allowed_relations_between_types(self.typepair_index, subj_type, obj_type)
                            allowed += get_allowed_relations_between_types(self.typepair_index, obj_type, subj_type)
                            allowed = sorted(set([a for a in allowed if isinstance(a, str) and a.strip()]))

                            if not allowed:
                                patches["relation_deletes"].append((ck, rel_key))
                                audit_item["relation_fixes"].append(
                                    {
                                        "document_id": ck,
                                        "rel_key": rel_key,
                                        "decision": "drop_no_allowed_types",
                                        "before": before,
                                        "after": None,
                                        "reason": "no allowed relation types for endpoint types",
                                        "llm_raw": "",
                                    }
                                )
                                continue

                            allowed_text = self._format_relation_list_desc(allowed)

                            r_min = {
                                "subject": subj,
                                "object": obj,
                                "relation_type": r.get("relation_type"),
                                "relation_name": r.get("relation_name"),
                                "description": r.get("description"),
                                "persistence": r.get("persistence"),
                            }

                            raw_fix = self.problem_solver.fix_relation_error(
                                text=related_text,
                                extracted_relation=json.dumps(r_min, ensure_ascii=False, indent=2),
                                feedback=feedback,
                                allowed_relation_types=allowed_text,
                            )
                            fix_data = _safe_json_loads(raw_fix)
                            r_decision = _normalize_decision(fix_data.get("decision"))
                            r_reason = fix_data.get("reason") or fix_data.get("rationale") or ""

                            r_output = fix_data.get("output")
                            if r_output is None:
                                r_output = fix_data.get("ouput")  # backward compat

                            if r_decision == "drop":
                                patches["relation_deletes"].append((ck, rel_key))
                                audit_item["relation_fixes"].append(
                                    {
                                        "document_id": ck,
                                        "rel_key": rel_key,
                                        "decision": "drop",
                                        "allowed_relation_types": allowed,
                                        "before": before,
                                        "after": None,
                                        "reason": r_reason if isinstance(r_reason, str) else "",
                                        "llm_raw": _truncate_raw(raw_fix),
                                    }
                                )
                                continue

                            if isinstance(r_output, str):
                                r_output = _safe_json_loads(r_output)
                            if not isinstance(r_output, dict):
                                patches["relation_deletes"].append((ck, rel_key))
                                audit_item["relation_fixes"].append(
                                    {
                                        "document_id": ck,
                                        "rel_key": rel_key,
                                        "decision": "drop_bad_output",
                                        "allowed_relation_types": allowed,
                                        "before": before,
                                        "after": None,
                                        "reason": "LLM output not a dict",
                                        "llm_raw": _truncate_raw(raw_fix),
                                    }
                                )
                                continue

                            patches["relation_rewrites"].append((ck, rel_key, r_output))

                            after = {
                                "subject": r_output.get("subject", subj),
                                "object": r_output.get("object", obj),
                                "relation_type": r_output.get("relation_type"),
                                "relation_name": r_output.get("relation_name"),
                                "description": r_output.get("description"),
                                "persistence": r_output.get("persistence"),
                            }
                            audit_item["relation_fixes"].append(
                                {
                                    "document_id": ck,
                                    "rel_key": rel_key,
                                    "decision": "rewrite",
                                    "allowed_relation_types": allowed,
                                    "before": before,
                                    "after": after,
                                    "reason": r_reason if isinstance(r_reason, str) else "",
                                    "llm_raw": _truncate_raw(raw_fix),
                                }
                            )

                    audits.append(audit_item)
                    continue

                # other / unknown
                audit_item["decision"] = f"unknown:{decision}"
                audits.append(audit_item)

            return {"patches": patches, "audits": audits}

        # ------------------------------------------------------------
        # Run concurrently over entity_names with progress bar
        # ------------------------------------------------------------
        all_patches: List[Dict[str, Any]] = []
        all_audits: List[Dict[str, Any]] = []

        with ThreadPoolExecutor(max_workers=workers) as ex:
            fut2name = {
                ex.submit(_process_one_entity, entity_name, dict(type2documents)): entity_name
                for entity_name, type2documents in name_type_dict.items()
            }

            pbar = tqdm(total=len(fut2name), desc="Refine entity types", ncols=100)
            try:
                for fut in as_completed(fut2name):
                    name = fut2name[fut]
                    out = fut.result()
                    if isinstance(out, dict):
                        p = out.get("patches")
                        a = out.get("audits")
                        if isinstance(p, dict):
                            all_patches.append(p)
                        if isinstance(a, list):
                            all_audits.extend([x for x in a if isinstance(x, dict)])
                    pbar.update(1)
                    pbar.set_postfix_str(str(name)[:40])
            finally:
                pbar.close()

        # Memory writing is handled inside knowledge_extraction_agent refine flow.
        # Keep audit generation here, but do not write extraction memories in graph_refiner.

        # ------------------------------------------------------------
        # Apply patches sequentially
        # ------------------------------------------------------------

        # 1) type overrides (drop-case)
        for p in all_patches:
            for ck, name, new_type in p.get("entity_type_updates", []) or []:
                payload = results.get(ck)
                if not isinstance(payload, dict):
                    continue
                for e in payload.get("entities", []) or []:
                    if isinstance(e, dict) and e.get("name") == name:
                        e["type"] = new_type

        # 2) blank description (drop-case)
        if blank_description_on_drop:
            for p in all_patches:
                for ck, name in p.get("entity_desc_blank", []) or []:
                    payload = results.get(ck)
                    if not isinstance(payload, dict):
                        continue
                    for e in payload.get("entities", []) or []:
                        if isinstance(e, dict) and e.get("name") == name:
                            e["description"] = ""

        # 3) entity rename (keep-case + event-rename)
        for p in all_patches:
            for ck, old_name, new_name in p.get("entity_renames", []) or []:
                payload = results.get(ck)
                if not isinstance(payload, dict):
                    continue
                for e in payload.get("entities", []) or []:
                    if isinstance(e, dict) and e.get("name") == old_name:
                        e["name"] = new_name

        # 4) relation endpoint rename (keep-case + event-rename)
        for p in all_patches:
            for ck, old_name, new_name in p.get("relation_endpoint_renames", []) or []:
                payload = results.get(ck)
                if not isinstance(payload, dict):
                    continue
                rels = payload.get("relations", []) or []
                for r in rels:
                    if not isinstance(r, dict):
                        continue
                    if r.get("subject") == old_name:
                        r["subject"] = new_name
                    if r.get("object") == old_name:
                        r["object"] = new_name

        # 5) relation deletes (drop-case)
        deletes_by_ck: Dict[str, set] = defaultdict(set)
        for p in all_patches:
            for ck, rel_key in p.get("relation_deletes", []) or []:
                deletes_by_ck[ck].add(rel_key)

        for ck, keys in deletes_by_ck.items():
            payload = results.get(ck)
            if not isinstance(payload, dict):
                continue
            rels = payload.get("relations", []) or []
            if not isinstance(rels, list) or not rels:
                continue

            new_rels = []
            for idx, r in enumerate(rels):
                if not isinstance(r, dict):
                    continue
                rk = _relation_key(r, idx)
                if rk in keys:
                    continue
                new_rels.append(r)
            payload["relations"] = new_rels

        # 6) relation rewrites (drop-case)
        rewrites_by_ck: Dict[str, List[Tuple[str, Dict[str, Any]]]] = defaultdict(list)
        for p in all_patches:
            for ck, rel_key, out in p.get("relation_rewrites", []) or []:
                if isinstance(out, dict):
                    rewrites_by_ck[ck].append((rel_key, out))

        for ck, items in rewrites_by_ck.items():
            payload = results.get(ck)
            if not isinstance(payload, dict):
                continue
            rels = payload.get("relations", []) or []
            if not isinstance(rels, list) or not rels:
                continue

            key2idx: Dict[str, int] = {}
            for idx, r in enumerate(rels):
                if isinstance(r, dict):
                    key2idx[_relation_key(r, idx)] = idx

            for rel_key, out in items:
                idx = key2idx.get(rel_key)
                if idx is None:
                    continue
                r = rels[idx]
                if isinstance(r, dict):
                    _apply_relation_rewrite_in_place(r, out)

            payload["relations"] = rels

        # ------------------------------------------------------------
        # Dump audit
        # ------------------------------------------------------------
        if dump_audit:
            _dump_audit_json(all_audits)

        return results

    # =========================
    # 2) refine entity scope
    # =========================
    def refine_entity_scope(
        self,
        document_results: Dict[str, Dict[str, Any]],
        *,
        enable_llm: bool = False,
        per_task_timeout: float = 180.0,
        tie_breaker: str = "global",
    ) -> Dict[str, Dict[str, Any]]:
        """
        Scope refinement (event-free / induced-aware):

        Pipeline:
        1) Normalize existing scope if present.
        2) If missing/invalid scope:
            - apply default scope from self.type2default_scope if available.
            - otherwise leave unset (no hard-coded special types).
        3) For same entity name with multiple scopes across docs:
            - optional LLM validation can override for ambiguous names
            - otherwise majority vote, tie -> tie_breaker
        4) Optional: for induced types, always enforce local scope (recommended by your rule intent).

        Notes:
        - No special casing for any literal type name (no "Event"/"Occasion").
        - "induced" handling is driven by self.INDUCED_TYPES (built from schema category).
        - If your extractor already sets scope reliably, this becomes mostly normalization + voting.
        """
        assert tie_breaker in ("global", "local")
        results = deepcopy(document_results)

        # -------------------------
        # helpers
        # -------------------------
        def _is_induced_type(tp: Any) -> bool:
            return isinstance(tp, str) and tp in getattr(self, "INDUCED_TYPES", set())

        def _set_scope(ent: Dict[str, Any], scope_val: str) -> None:
            ent["scope"] = scope_val

        # -------------------------
        # 1) normalize / fill defaults
        # -------------------------
        for _ck, payload in (results or {}).items():
            if not isinstance(payload, dict):
                continue
            ents = payload.get("entities") or []
            if not isinstance(ents, list):
                continue

            for ent in ents:
                if not isinstance(ent, dict):
                    continue

                tp = ent.get("type")

                # normalize existing
                sc = self._norm_scope(ent.get("scope"))
                if sc in ("global", "local"):
                    _set_scope(ent, sc)
                    continue

                # apply default scope if available (from tasks json rules you built)
                ds = self.type2default_scope.get(tp)
                ds = self._norm_scope(ds)
                if ds in ("global", "local"):
                    _set_scope(ent, ds)
                # else: keep as-is (unset/invalid)

        # -------------------------
        # 2) collect multi-scope names (only for non-induced)
        # -------------------------
        name_scopes: Dict[str, set] = defaultdict(set)
        for _ck, payload in (results or {}).items():
            if not isinstance(payload, dict):
                continue
            for ent in (payload.get("entities") or []):
                if not isinstance(ent, dict):
                    continue
                tp = ent.get("type")
                if _is_induced_type(tp):
                    continue

                nm = ent.get("name")
                if not isinstance(nm, str) or not nm.strip():
                    continue

                scn = self._norm_scope(ent.get("scope"))
                if scn in ("global", "local"):
                    name_scopes[nm.strip()].add(scn)

        to_check = [n for n, s in name_scopes.items() if len(s) > 1]

        # -------------------------
        # 3) optional LLM scope check (only for ambiguous names, non-induced)
        # -------------------------
        llm_scope_rules: Dict[str, str] = {}

        def prepare_context_by_scope(entity_name: str) -> str:
            lines: List[str] = []
            for scope_ in ("global", "local"):
                lines.append(f"Entity scope: {scope_}")
                for ck, payload in (results or {}).items():
                    if not isinstance(payload, dict):
                        continue
                    for ent in (payload.get("entities") or []):
                        if not isinstance(ent, dict):
                            continue
                        if ent.get("name") != entity_name:
                            continue
                        if _is_induced_type(ent.get("type")):
                            continue

                        sc = self._norm_scope(ent.get("scope"))
                        if sc != scope_:
                            continue

                        desc = ent.get("description", "") or ""
                        if isinstance(desc, str) and word_len(desc, lang="auto") > 300:
                            desc = truncate_by_word_len(desc, 300, lang="auto")

                        lines.append(f"- document: {ck}, Type: {ent.get('type')}, Description: {desc}")
                lines.append("")
            return "\n".join(lines).strip()

        if enable_llm and to_check:
            def _check_scope(name: str):
                ctx = prepare_context_by_scope(name)
                raw = self.document_parser.validate_entity_scope(ctx)
                data = raw if isinstance(raw, dict) else json.loads(correct_json_format(raw))
                sc = self._norm_scope(data.get("scope") if isinstance(data, dict) else None)
                return name, sc

            pool_out = self._soft_timeout_pool(
                to_check,
                _check_scope,
                per_task_timeout=per_task_timeout,
                desc="Validate entity scope (LLM)",
                thread_prefix="escope",
            )
            for timeout_or_error, _item, res in pool_out:
                if timeout_or_error or res is None:
                    continue
                nm, sc = res
                if sc in ("global", "local"):
                    llm_scope_rules[nm] = sc

        # -------------------------
        # 4) majority vote fallback (after llm suggestions)
        # -------------------------
        counts: Dict[str, Dict[str, int]] = defaultdict(lambda: {"global": 0, "local": 0})

        for _ck, payload in (results or {}).items():
            if not isinstance(payload, dict):
                continue
            for ent in (payload.get("entities") or []):
                if not isinstance(ent, dict):
                    continue

                tp = ent.get("type")

                nm = ent.get("name")
                if not isinstance(nm, str) or not nm.strip():
                    continue
                nm = nm.strip()

                # llm rule if present
                if nm in llm_scope_rules:
                    _set_scope(ent, llm_scope_rules[nm])
                    continue

                sc = self._norm_scope(ent.get("scope"))
                if sc in ("global", "local"):
                    _set_scope(ent, sc)
                    counts[nm][sc] += 1

        # apply majority vote to ambiguous names
        for nm, c in counts.items():
            g, l = int(c.get("global", 0)), int(c.get("local", 0))
            if g > 0 and l > 0:
                if g > l:
                    target = "global"
                elif l > g:
                    target = "local"
                else:
                    target = tie_breaker

                for _ck, payload in (results or {}).items():
                    if not isinstance(payload, dict):
                        continue
                    for ent in (payload.get("entities") or []):
                        if not isinstance(ent, dict):
                            continue
                        if ent.get("name") != nm:
                            continue

                        # induced still wins if somehow present under same name
                        if _is_induced_type(ent.get("type")):
                            _set_scope(ent, "local")
                        else:
                            _set_scope(ent, target)

        return results

    # =========================
    # 3) entity disambiguation (summary + embed + cluster + LLM merge)
    # =========================
    def run_entity_disambiguation(
        self,
        document_results: Dict[str, Dict[str, Any]],
        *,
        require_scope_global: bool = True,
        exclude_induced: bool = True,
        per_task_timeout_summary: float = 180.0,
        per_task_timeout_merge: float = 120.0,
        dump_maps: bool = True,
        dump_audit: bool = True,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Disambiguation pipeline (with audit logging):
        - Summarize entities and write back `summary`
        - Embed (name + summary)
        - Cluster candidates (per type)
        - LLM merge decision -> rename_map (alias -> canonical)
        - Apply rename_map to entities and relations (and write aliases onto canonical entity)

        Hard constraints:
        - exclude_induced: induced-category entity types do NOT participate
        - require_scope_global: only global-scoped entities participate
        - rename_map application respects the same constraints

        Implementation notes:
        - No hard-coded type strings.
        - "missing type" is normalized to GENERAL_SEMANTIC_TYPE (schema-defined single fallback).
        - Relation endpoint renaming uses per-document allow_names that includes BOTH old and new names.
        """

        results = deepcopy(document_results)

        # -------------------------
        # helpers
        # -------------------------
        def _truncate_raw(x: Any, limit: int = 4000) -> str:
            if x is None:
                return ""
            if isinstance(x, (dict, list)):
                try:
                    x = json.dumps(x, ensure_ascii=False)
                except Exception:
                    x = str(x)
            if not isinstance(x, str):
                x = str(x)
            s = x.strip()
            return s if len(s) <= limit else (s[:limit] + "...")

        def _dump_audit_json(audits: List[Dict[str, Any]]) -> str:
            base = self.config.knowledge_graph_builder.file_path
            save_dir = os.path.join(base, "audit_logs")
            os.makedirs(save_dir, exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
            path = os.path.join(save_dir, "run_entity_disambiguation_audit.json")
            payload = {
                "step": "run_entity_disambiguation",
                "created_at": ts,
                "config": {
                    "require_scope_global": bool(require_scope_global),
                    "exclude_induced": bool(exclude_induced),
                    "per_task_timeout_summary": float(per_task_timeout_summary),
                    "per_task_timeout_merge": float(per_task_timeout_merge),
                },
                "audits": audits,
            }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            return path

        def _is_induced_type(tp: Any) -> bool:
            return isinstance(tp, str) and tp in getattr(self, "INDUCED_TYPES", set())

        def _norm_type(tp: Any) -> str:
            if isinstance(tp, str) and tp.strip():
                t = tp.strip()
                if t in getattr(self, "ALLOWED_TYPES", ()):
                    return t
                # if a non-empty unknown slips in, still fall back
            return getattr(self, "GENERAL_SEMANTIC_TYPE")

        def _is_eligible_entity(ent: Any) -> bool:
            if not isinstance(ent, dict):
                return False

            tp = _norm_type(ent.get("type"))
            if exclude_induced and _is_induced_type(tp):
                return False

            if require_scope_global:
                sc = self._norm_scope(ent.get("scope"))
                if sc != "global":
                    return False

            nm = ent.get("name")
            return isinstance(nm, str) and bool(nm.strip())

        # -------------------------
        # audits meta
        # -------------------------
        audits: List[Dict[str, Any]] = []
        audit_meta: Dict[str, Any] = {
            "kind": "meta",
            "timestamp": time.time(),
            "stats": {
                "documents": len(results or {}),
                "entities_total": 0,
                "relations_total": 0,
                "entities_considered": 0,
                "entities_skipped_induced": 0,
                "entities_skipped_scope": 0,
            },
        }

        for _ck, payload in (results or {}).items():
            if not isinstance(payload, dict):
                continue
            ents = payload.get("entities") or []
            rels = payload.get("relations") or []
            if isinstance(ents, list):
                audit_meta["stats"]["entities_total"] += len(ents)
            if isinstance(rels, list):
                audit_meta["stats"]["relations_total"] += len(rels)

        audits.append(audit_meta)

        # -------------------------
        # 1) collect entities for summarization
        # -------------------------
        key2ents = defaultdict(list)  # (name,type) -> list[ent_ref]
        key2doc_ids: Dict[Tuple[str, str], List[str]] = defaultdict(list)
        key2doc_seen: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
        for ck, payload in (results or {}).items():
            if not isinstance(payload, dict):
                continue

            ents = payload.get("entities") or []
            if not isinstance(ents, list):
                continue

            for ent in ents:
                if not isinstance(ent, dict):
                    continue

                nm = ent.get("name")
                if not isinstance(nm, str) or not nm.strip():
                    continue
                nm = nm.strip()

                tp = _norm_type(ent.get("type"))
                ent["type"] = tp  # normalize in-place

                if exclude_induced and _is_induced_type(tp):
                    audit_meta["stats"]["entities_skipped_induced"] += 1
                    continue

                if require_scope_global:
                    sc = self._norm_scope(ent.get("scope"))
                    if sc != "global":
                        audit_meta["stats"]["entities_skipped_scope"] += 1
                        continue

                audit_meta["stats"]["entities_considered"] += 1
                key2ents[(nm, tp)].append(ent)
                if ck not in key2doc_seen[(nm, tp)]:
                    key2doc_seen[(nm, tp)].add(ck)
                    key2doc_ids[(nm, tp)].append(ck)

        # -------------------------
        # 2) summarization tasks
        # -------------------------
        items = list(key2ents.keys())

        def _summarize_one(key: Tuple[str, str]):
            name, tp = key
            ents = key2ents[key]

            texts: List[str] = []
            for e in ents[:6]:
                d = e.get("description", "") or ""
                if isinstance(d, str) and d.strip():
                    texts.append(d.strip())

            raw_text = "\n".join(texts).strip()
            if not raw_text:
                return key, "", {"decision": "no_text", "llm_raw": ""}

            # short enough: no need to LLM
            if len(raw_text) < 300:
                return key, raw_text, {"decision": "pass_through", "llm_raw": ""}

            raw = self.document_parser.summarize_paragraph(text=raw_text, max_length=250)
            data = raw if isinstance(raw, dict) else json.loads(correct_json_format(raw))
            summ = data.get("summary") if isinstance(data, dict) else ""
            if not isinstance(summ, str) or not summ.strip():
                summ = raw_text[:250]
                return key, summ.strip(), {"decision": "fallback", "llm_raw": _truncate_raw(raw)}
            return key, summ.strip(), {"decision": "llm", "llm_raw": _truncate_raw(raw)}

        summary_map: Dict[Tuple[str, str], str] = {}
        if items:
            pool_out = self._soft_timeout_pool(
                items,
                _summarize_one,
                per_task_timeout=per_task_timeout_summary,
                desc="Summarize entities",
                thread_prefix="summ",
            )
            for timeout_or_error, item, res in pool_out:
                audit_item = {
                    "kind": "summary",
                    "timestamp": time.time(),
                    "name": item[0],
                    "type": item[1],
                    "timeout_or_error": bool(timeout_or_error),
                    "decision": "",
                    "llm_raw": "",
                }

                if timeout_or_error or res is None:
                    ents = key2ents[item]
                    d0 = ents[0].get("description", "") if ents else ""
                    summary_map[item] = d0 if isinstance(d0, str) else ""
                    audit_item["decision"] = "timeout_or_error"
                    audits.append(audit_item)
                    continue

                key, summ, meta = res
                summary_map[key] = summ
                audit_item["decision"] = meta.get("decision", "")
                audit_item["llm_raw"] = meta.get("llm_raw", "")
                if audit_item["decision"] not in ("pass_through", "no_text"):
                    audits.append(audit_item)

        # -------------------------
        # 3) write back summary
        # -------------------------
        for (nm, tp), ents in key2ents.items():
            summ = summary_map.get((nm, tp), "") or ""
            for e in ents:
                e["summary"] = summ

        # -------------------------
        # 4) build per-type pool for clustering
        # -------------------------
        type2items = defaultdict(dict)
        for (nm, tp), _ents in key2ents.items():
            summ = summary_map.get((nm, tp), "") or ""
            type2items[tp][nm] = {
                "name": nm,
                "type": tp,
                "summary": summ,
                "document_ids": list(key2doc_ids.get((nm, tp), []) or []),
            }

        # -------------------------
        # 5) compute embeddings
        # -------------------------
        embed_stats = {"by_type": {}}
        for tp, by_name in type2items.items():
            if not by_name:
                continue
            embed_stats["by_type"][tp] = {"n": len(by_name)}
            for nm, info in tqdm(by_name.items(), desc=f"Compute embeddings [{tp}]"):
                info["name_embedding"] = self.model.encode(info["name"])
                info["description_embedding"] = self.model.encode((info.get("summary") or "").strip())
        audits.append({"kind": "embedding", "timestamp": time.time(), "stats": embed_stats})

        # -------------------------
        # 6) detect clusters
        # -------------------------
        all_candidates: List[List[Dict[str, Any]]] = []
        cluster_audits: List[Dict[str, Any]] = []

        for tp, by_name in type2items.items():
            if len(by_name) < 3:
                cluster_audits.append(
                    {"kind": "cluster", "timestamp": time.time(), "type": tp, "decision": "skip_small", "n": len(by_name)}
                )
                continue

            n = len(by_name)
            knn_k = max(5, min(int(n / 4), 25, n - 1))
            estimated_k, _sim = compute_weighted_similarity_and_laplacian(by_name, alpha=0.8, knn_k=knn_k)
            rough = max(2, int((estimated_k + n / 2) / 2))
            n_clusters = min(max(2, rough), n)

            clusters = run_kmeans_clustering(by_name, n_clusters=n_clusters, alpha=0.5)
            cluster_audits.append(
                {
                    "kind": "cluster",
                    "timestamp": time.time(),
                    "type": tp,
                    "decision": "ok",
                    "n": n,
                    "knn_k": knn_k,
                    "estimated_k": int(estimated_k),
                    "n_clusters": int(n_clusters),
                    "num_clusters_size_ge_2": len(clusters),
                    "clusters": [list(g) for g in clusters][:50],
                }
            )

            for group_names in clusters:
                group = [by_name[nm] for nm in group_names if nm in by_name]
                if len(group) >= 2:
                    all_candidates.append(group)

        audits.extend(cluster_audits)

        # -------------------------
        # 7) LLM merge -> rename_map
        # -------------------------
        def _render_merge_group_payload(group: List[Dict[str, Any]], *, include_lexical: bool = False) -> str:
            parts: List[str] = []
            for i, e in enumerate(group):
                parts.append(f"Entity {i + 1} name: {e.get('name')}")
                parts.append(f" - Description: {e.get('summary','')}")
                if include_lexical:
                    sig = e.get("lexical_signature") or {}
                    doc_ids = list(e.get("document_ids") or [])
                    parts.append(
                        " - Lexical signature: "
                        f"normalized={safe_str(sig.get('normalized_name'))}, "
                        f"core={safe_str(sig.get('core_name'))}, "
                        f"anchors={', '.join(sig.get('anchors') or []) or '(none)'}"
                    )
                    if doc_ids:
                        parts.append(f" - Source documents: {', '.join(doc_ids[:12])}")
            if include_lexical and len(group) >= 2:
                parts.append("")
                parts.append("[Lexical pair evidence]")
                for left, right in combinations(group, 2):
                    features = self._compute_lexical_pair_features(
                        left.get("lexical_signature") or self._build_lexical_signature(left.get("name")),
                        right.get("lexical_signature") or self._build_lexical_signature(right.get("name")),
                    )
                    parts.append(
                        f"- {left.get('name')} <-> {right.get('name')}: "
                        + self._render_lexical_feature_line(features)
                    )
            return "\n".join(parts)

        def _run_merge_group(group: List[Dict[str, Any]]) -> Dict[str, Any]:
            try:
                entity_descriptions = _render_merge_group_payload(group, include_lexical=False)
                raw = self.problem_solver.merge_entities(entity_descriptions=entity_descriptions)
                data = raw if isinstance(raw, dict) else json.loads(correct_json_format(raw))

                merges = data.get("merges", []) if isinstance(data, dict) else []
                unmerged = data.get("unmerged", []) if isinstance(data, dict) else []
                if not isinstance(merges, list):
                    merges = []
                if not isinstance(unmerged, list):
                    unmerged = []

                return {"ok": True, "merges": merges, "unmerged": unmerged, "llm_raw": _truncate_raw(raw)}
            except Exception as e:
                return {"ok": False, "merges": [], "unmerged": [], "llm_raw": _truncate_raw({"error": str(e)})}

        def _run_lexical_merge_group(group: List[Dict[str, Any]]) -> Dict[str, Any]:
            try:
                entity_descriptions = _render_merge_group_payload(group, include_lexical=True)
                raw = self.problem_solver.merge_entities(entity_descriptions=entity_descriptions)
                data = raw if isinstance(raw, dict) else json.loads(correct_json_format(raw))

                merges = data.get("merges", []) if isinstance(data, dict) else []
                unmerged = data.get("unmerged", []) if isinstance(data, dict) else []
                if not isinstance(merges, list):
                    merges = []
                if not isinstance(unmerged, list):
                    unmerged = []

                return {"ok": True, "merges": merges, "unmerged": unmerged, "llm_raw": _truncate_raw(raw)}
            except Exception as e:
                return {"ok": False, "merges": [], "unmerged": [], "llm_raw": _truncate_raw({"error": str(e)})}

        def _apply_merges_to_rename_map(
            merges: List[Dict[str, Any]],
            *,
            audit_item: Dict[str, Any],
            stage_label: str,
        ) -> None:
            applied = []
            conflicts = []
            for m in merges:
                if not isinstance(m, dict):
                    continue
                canonical = safe_str(m.get("canonical_name")).strip()
                if not canonical:
                    continue

                for alias in (m.get("aliases") or []):
                    alias = safe_str(alias).strip()
                    if not alias or alias == canonical:
                        continue

                    blocked, block_reason = self._should_block_model_merge(alias, canonical)
                    if blocked:
                        conflicts.append((alias, canonical, block_reason))
                        continue

                    existing = safe_str(rename_map.get(alias, "")).strip()
                    if existing and existing != canonical:
                        existing_resolved = self._compress_rename_map({**rename_map, alias: existing}).get(alias, existing)
                        proposed_resolved = self._compress_rename_map({**rename_map, alias: canonical}).get(alias, canonical)
                        if existing_resolved != proposed_resolved:
                            conflicts.append((alias, existing, canonical))
                            continue

                    rename_map[alias] = canonical
                    applied.append((alias, canonical))

            audit_item["applied_pairs"] = applied[:200]
            if conflicts:
                audit_item["conflicts"] = conflicts[:200]
                audits.append(
                    {
                        "kind": "rename_map_conflict",
                        "timestamp": time.time(),
                        "stage": stage_label,
                        "conflicts": conflicts[:200],
                    }
                )

        rename_map: Dict[str, str] = {}
        merge_group_audits: List[Dict[str, Any]] = []

        if all_candidates:
            pool_out = self._soft_timeout_pool(
                all_candidates,
                _run_merge_group,
                per_task_timeout=per_task_timeout_merge,
                desc="Entity merge decision (LLM)",
                thread_prefix="merge",
            )

            for timeout_or_error, group, res in pool_out:
                group_names: List[str] = []
                group_type = None
                for e in (group or []):
                    if isinstance(e, dict):
                        if group_type is None:
                            group_type = e.get("type")
                        nm = e.get("name")
                        if isinstance(nm, str):
                            group_names.append(nm)

                audit_item = {
                    "kind": "merge_group",
                    "timestamp": time.time(),
                    "type": group_type,
                    "group": group_names,
                    "timeout_or_error": bool(timeout_or_error),
                    "ok": False,
                    "merges": [],
                    "unmerged": [],
                    "llm_raw": "",
                    "applied_pairs": [],
                }

                if timeout_or_error or res is None:
                    merge_group_audits.append(audit_item)
                    continue

                data = res if isinstance(res, dict) else {"ok": False, "merges": [], "unmerged": [], "llm_raw": ""}
                audit_item["ok"] = bool(data.get("ok"))
                audit_item["llm_raw"] = data.get("llm_raw", "")

                merges = data.get("merges", []) or []
                unmerged = data.get("unmerged", []) or []
                audit_item["merges"] = merges
                audit_item["unmerged"] = unmerged

                _apply_merges_to_rename_map(merges, audit_item=audit_item, stage_label="cluster_llm")
                merge_group_audits.append(audit_item)

        audits.extend(merge_group_audits)

        # -------------------------
        # 7.5) lexical post-pass on unresolved names
        # -------------------------
        lexical_candidate_groups, lexical_candidate_audits = self._build_lexical_candidate_groups(type2items, rename_map)
        audits.extend(lexical_candidate_audits)

        lexical_merge_audits: List[Dict[str, Any]] = []
        if lexical_candidate_groups:
            pool_out = self._soft_timeout_pool(
                lexical_candidate_groups,
                _run_lexical_merge_group,
                per_task_timeout=per_task_timeout_merge,
                desc="Entity lexical post-pass merge (LLM)",
                thread_prefix="lexmerge",
            )

            for timeout_or_error, group, res in pool_out:
                group_names: List[str] = []
                group_type = None
                for e in (group or []):
                    if isinstance(e, dict):
                        if group_type is None:
                            group_type = e.get("type")
                        nm = e.get("name")
                        if isinstance(nm, str):
                            group_names.append(nm)

                audit_item = {
                    "kind": "lexical_merge_group",
                    "timestamp": time.time(),
                    "type": group_type,
                    "group": group_names,
                    "timeout_or_error": bool(timeout_or_error),
                    "ok": False,
                    "merges": [],
                    "unmerged": [],
                    "llm_raw": "",
                    "applied_pairs": [],
                }

                if timeout_or_error or res is None:
                    lexical_merge_audits.append(audit_item)
                    continue

                data = res if isinstance(res, dict) else {"ok": False, "merges": [], "unmerged": [], "llm_raw": ""}
                audit_item["ok"] = bool(data.get("ok"))
                audit_item["llm_raw"] = data.get("llm_raw", "")
                merges = data.get("merges", []) or []
                unmerged = data.get("unmerged", []) or []
                audit_item["merges"] = merges
                audit_item["unmerged"] = unmerged

                _apply_merges_to_rename_map(merges, audit_item=audit_item, stage_label="lexical_postpass_llm")
                lexical_merge_audits.append(audit_item)

        audits.extend(lexical_merge_audits)
        rename_map = self._compress_rename_map(rename_map)

        # -------------------------
        # 8) apply rename_map (with aliases write-back)
        # -------------------------
        apply_audit = {
            "kind": "apply_rename_map",
            "timestamp": time.time(),
            "stats": {
                "rename_map_size": len(rename_map),
                "entity_renames": 0,
                "relation_endpoint_renames": 0,
                "documents_touched": 0,
                "remaining_alias_endpoints_after_apply": 0,
                "aliases_written": 0,
            },
            "examples": {"entity_renames": [], "relation_renames": [], "aliases": []},
        }

        if rename_map:
            for ck, payload in (results or {}).items():
                if not isinstance(payload, dict):
                    continue

                document_touched = False
                ents = payload.get("entities") or []
                rels = payload.get("relations") or []
                if not isinstance(ents, list):
                    ents = []
                if not isinstance(rels, list):
                    rels = []

                allow_names: set = set()

                # index: name -> canonical entity dict (eligible only)
                name2ent: Dict[str, Dict[str, Any]] = {}
                for ent in ents:
                    if not _is_eligible_entity(ent):
                        continue
                    nm0 = ent.get("name")
                    if isinstance(nm0, str) and nm0.strip():
                        name2ent[nm0.strip()] = ent
                    if "aliases" not in ent or not isinstance(ent.get("aliases"), list):
                        ent["aliases"] = []

                # 8.1 rename entities + write aliases to canonical entity
                for ent in ents:
                    if not _is_eligible_entity(ent):
                        continue

                    old_nm = ent.get("name")
                    if not isinstance(old_nm, str) or not old_nm.strip():
                        continue
                    old_nm = old_nm.strip()

                    allow_names.add(old_nm)

                    if old_nm in rename_map:
                        new_nm = rename_map[old_nm]
                        if isinstance(new_nm, str) and new_nm.strip() and new_nm.strip() != old_nm:
                            new_nm = new_nm.strip()

                            ent["name"] = new_nm
                            allow_names.add(new_nm)

                            # ensure canonical entry exists
                            if new_nm not in name2ent:
                                name2ent[new_nm] = ent

                            canonical_ent = name2ent.get(new_nm, ent)
                            if "aliases" not in canonical_ent or not isinstance(canonical_ent.get("aliases"), list):
                                canonical_ent["aliases"] = []

                            if old_nm != new_nm and old_nm not in canonical_ent["aliases"]:
                                canonical_ent["aliases"].append(old_nm)
                                apply_audit["stats"]["aliases_written"] += 1
                                if len(apply_audit["examples"]["aliases"]) < 50:
                                    apply_audit["examples"]["aliases"].append(
                                        {"document_id": ck, "canonical": new_nm, "alias": old_nm}
                                    )

                            apply_audit["stats"]["entity_renames"] += 1
                            document_touched = True
                            if len(apply_audit["examples"]["entity_renames"]) < 50:
                                apply_audit["examples"]["entity_renames"].append(
                                    {"document_id": ck, "old": old_nm, "new": new_nm, "type": ent.get("type")}
                                )

                # 8.2 rename relation endpoints (only if endpoint is in allow_names)
                for rel in rels:
                    if not isinstance(rel, dict):
                        continue

                    sub = rel.get("subject")
                    obj = rel.get("object")
                    changed = False

                    if isinstance(sub, str):
                        sub_s = sub.strip()
                        if sub_s in rename_map and sub_s in allow_names:
                            new_sub = rename_map[sub_s]
                            if isinstance(new_sub, str) and new_sub.strip() and new_sub.strip() != sub_s:
                                rel["subject"] = new_sub.strip()
                                changed = True
                                apply_audit["stats"]["relation_endpoint_renames"] += 1
                                if len(apply_audit["examples"]["relation_renames"]) < 50:
                                    apply_audit["examples"]["relation_renames"].append(
                                        {"document_id": ck, "side": "subject", "old": sub_s, "new": new_sub.strip()}
                                    )

                    if isinstance(obj, str):
                        obj_s = obj.strip()
                        if obj_s in rename_map and obj_s in allow_names:
                            new_obj = rename_map[obj_s]
                            if isinstance(new_obj, str) and new_obj.strip() and new_obj.strip() != obj_s:
                                rel["object"] = new_obj.strip()
                                changed = True
                                apply_audit["stats"]["relation_endpoint_renames"] += 1
                                if len(apply_audit["examples"]["relation_renames"]) < 50:
                                    apply_audit["examples"]["relation_renames"].append(
                                        {"document_id": ck, "side": "object", "old": obj_s, "new": new_obj.strip()}
                                    )

                    if changed:
                        document_touched = True

                if document_touched:
                    apply_audit["stats"]["documents_touched"] += 1

            # diagnostics: remaining endpoints that are still aliases
            remaining = 0
            for _ck, payload in (results or {}).items():
                if not isinstance(payload, dict):
                    continue
                for rel in (payload.get("relations") or []):
                    if not isinstance(rel, dict):
                        continue
                    s = rel.get("subject")
                    o = rel.get("object")
                    if isinstance(s, str) and s.strip() in rename_map:
                        remaining += 1
                    if isinstance(o, str) and o.strip() in rename_map:
                        remaining += 1
            apply_audit["stats"]["remaining_alias_endpoints_after_apply"] = int(remaining)

        audits.append(apply_audit)

        # -------------------------
        # 9) dump rename_map and audit
        # -------------------------
        if dump_maps:
            base = self.config.knowledge_graph_builder.file_path
            os.makedirs(base, exist_ok=True)
            with open(os.path.join(base, "rename_map.json"), "w", encoding="utf-8") as f:
                json.dump(rename_map, f, ensure_ascii=False, indent=2)

        if dump_audit:
            _dump_audit_json(audits)

        return results

    # =========================
    # 4) end-to-end
    # =========================
    def run_all(
        self,
        document_results: Dict[str, Dict[str, Any]],
        *,
        verbose: bool = True,
        type_timeout: float = 120.0,
        scope_timeout: float = 180.0,
        disambig_timeout_summary: float = 180.0,
        disambig_timeout_merge: float = 120.0,
        enable_scope_refine_llm: bool = False,
    ) -> Dict[str, Dict[str, Any]]:
        if verbose:
            print("Refining entity types (LLM)...")
        out = self.refine_entity_types(
            document_results,
            per_task_timeout=type_timeout,
        )

        if verbose:
            print("Refining entity scope...")
        out = self.refine_entity_scope(
            out,
            enable_llm=enable_scope_refine_llm,
            per_task_timeout=scope_timeout,
            tie_breaker="global",
        )

        if verbose:
            print("Running entity disambiguation (summary will be written into entities)...")

        out = self.run_entity_disambiguation(
            out,
            require_scope_global=True,
            exclude_induced=False,
            per_task_timeout_summary=disambig_timeout_summary,
            per_task_timeout_merge=disambig_timeout_merge,
            dump_maps=True,
        )


        return out

    def merge_relations(
        self,
        *,
        input_path: Optional[str] = None,
        output_path: Optional[str] = None,
        enable_llm: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Read relation_basic_info.json and merge relations with optional conflict checking.

        Conflict check (dynamic, schema-driven):
        - Only for relations with persistence in {"stable","phase"} (non-momentary).
        - If two opposite relation types (via schema field "opposite") appear between the same unordered entity pair,
        run LLM to decide which relation types to KEEP.
        - All not-kept opposite types are DELETED (no downgrade).

        Merge:
        - stable dedup by id (optional)
        - relations with persistence="momentary" are NOT merged (kept one-by-one)
        - group by (relation_type, subject_id, object_id) if ids exist, else (relation_type, subject, object)
        - aggregate source_documents from document_id (always list)
        - relation_name / description pick longest
        - keep base id (first item)
        - drop chunk_ids
        """
        base = self.config.knowledge_graph_builder.file_path
        if input_path is None:
            input_path = os.path.join(base, "relation_basic_info.json")

        rels = load_json(input_path)
        if not isinstance(rels, list):
            return []

        def _s(x: Any) -> str:
            return (safe_str(x) or "").strip()

        def _safe_json_loads(raw: Any) -> Any:
            if raw is None:
                return None
            if isinstance(raw, (dict, list)):
                return raw
            if not isinstance(raw, str):
                try:
                    return json.loads(str(raw))
                except Exception:
                    return None
            s = raw.strip()
            if not s:
                return None
            try:
                return json.loads(s)
            except Exception:
                try:
                    return json.loads(correct_json_format(s))
                except Exception:
                    return None

        def _pick_longest(a: str, b: str) -> str:
            a = (a or "").strip()
            b = (b or "").strip()
            return b if len(b) > len(a) else a

        def _rtype_desc(rtype: str) -> str:
            info = (self.relation_type_info or {}).get(rtype, {}) or {}
            d = info.get("description", "")
            return d.strip() if isinstance(d, str) else ""

        # Only non-momentary participates in conflict check
        NON_MOMENTARY = {"stable", "phase"}

        # -------------------------
        # 0) stable dedup by id
        # -------------------------
        seen_ids: Set[str] = set()
        cleaned: List[Dict[str, Any]] = []
        for r in rels:
            if not isinstance(r, dict):
                continue
            rid = _s(r.get("id"))
            if rid:
                if rid in seen_ids:
                    continue
                seen_ids.add(rid)
            cleaned.append(r)

        # -------------------------
        # 1) build pair2edges only for non-momentary and only for types that have an opposite pair
        # -------------------------
        opposite_pairs: Set[frozenset] = getattr(self, "opposite_pairs", set()) or set()
        opposite_of: Dict[str, str] = getattr(self, "opposite_of", {}) or {}

        # For fast membership: type -> frozenset({type, opposite})
        type2pairkey: Dict[str, frozenset] = {}
        for a, b in opposite_of.items():
            pairkey = frozenset([a, b])
            if pairkey in opposite_pairs:
                type2pairkey[a] = pairkey
                type2pairkey[b] = pairkey

        pair2edges: Dict[Tuple[str, str, frozenset], List[Dict[str, Any]]] = defaultdict(list)
        for r in cleaned:
            sid = _s(r.get("subject_id"))
            oid = _s(r.get("object_id"))
            if not sid or not oid:
                continue

            persistence = _s(r.get("persistence") or "phase")
            if persistence not in NON_MOMENTARY:
                continue

            rt = _s(r.get("relation_type") or r.get("predicate"))
            if not rt:
                continue

            pk = type2pairkey.get(rt)
            if pk is None:
                continue

            a, b = (sid, oid) if sid <= oid else (oid, sid)
            pair2edges[(a, b, pk)].append(r)

        def _pick_pair_display_names(rs: List[Dict[str, Any]], sid_a: str, sid_b: str) -> Tuple[str, str]:
            name_a = ""
            name_b = ""
            for x in rs:
                s_id = _s(x.get("subject_id"))
                o_id = _s(x.get("object_id"))
                s_nm = _s(x.get("subject"))
                o_nm = _s(x.get("object"))
                if s_id == sid_a and not name_a:
                    name_a = s_nm
                if s_id == sid_b and not name_b:
                    name_b = s_nm
                if o_id == sid_a and not name_a:
                    name_a = o_nm
                if o_id == sid_b and not name_b:
                    name_b = o_nm
                if name_a and name_b:
                    break
            return name_a or sid_a, name_b or sid_b

        def _build_multiedge_information(
            *,
            name_a: str,
            name_b: str,
            edges: List[Dict[str, Any]],
            type_a: str,
            type_b: str,
        ) -> str:
            t1_desc = _rtype_desc(type_a)
            t2_desc = _rtype_desc(type_b)

            cand = [e for e in edges if _s(e.get("relation_type") or e.get("predicate")) in (type_a, type_b)]
            cand = sorted(cand, key=lambda e: (_s(e.get("document_id")), _s(e.get("id"))))

            lines: List[str] = []
            lines.append(f'Relations between "{name_a}" and "{name_b}":')
            lines.append(f"{type_a}: {t1_desc}")
            lines.append(f"{type_b}: {t2_desc}")
            lines.append("")
            lines.append("Mentions:")
            for e in cand:
                rt = _s(e.get("relation_type") or e.get("predicate"))
                ck = _s(e.get("document_id"))
                rn = _s(e.get("relation_name"))
                ds = _s(e.get("description"))
                lines.append(f"- {ck} | {rt} | {rn} | {ds}")
            return "\n".join(lines).strip()

        def _extract_kept_types(raw: Any, allowed: Set[str]) -> Set[str]:
            data = _safe_json_loads(raw)
            kept = None
            if isinstance(data, dict):
                kept = data.get("keep_relation_type")
                if kept is None:
                    kept = data.get("kept_relation_types")
                if kept is None:
                    kept = data.get("keep_types")

            out: Set[str] = set()
            if isinstance(kept, str) and kept.strip():
                out.add(kept.strip())
            elif isinstance(kept, list):
                for x in kept:
                    if isinstance(x, str) and x.strip():
                        out.add(x.strip())

            # conservative fallback: keep all if model gave nothing
            if not out:
                return set(allowed)

            out = out.intersection(allowed)
            return out if out else set(allowed)

        # -------------------------
        # 2) LLM decide kept types per (entity_pair, opposite_pair) ONLY if both types exist
        # -------------------------
        pair2kept: Dict[Tuple[str, str, frozenset], Set[str]] = {}
        for (sid_a, sid_b, pk), rs in pair2edges.items():
            types_here = set(_s(x.get("relation_type") or x.get("predicate")) for x in rs)
            if len(types_here.intersection(set(pk))) < 2:
                pair2kept[(sid_a, sid_b, pk)] = set(pk)
                continue

            if enable_llm:
                t_list = sorted(list(pk))
                t1, t2 = t_list[0], t_list[1]

                name_a, name_b = _pick_pair_display_names(rs, sid_a, sid_b)
                multiedge_information = _build_multiedge_information(
                    name_a=name_a,
                    name_b=name_b,
                    edges=rs,
                    type_a=t1,
                    type_b=t2,
                )

                raw = self.problem_solver.resolve_relation_conflict(
                    subject_entity=name_a,
                    object_entity=name_b,
                    text=multiedge_information,
                )
                pair2kept[(sid_a, sid_b, pk)] = _extract_kept_types(raw, allowed=set(pk))
            else:
                pair2kept[(sid_a, sid_b, pk)] = set(pk)

        # -------------------------
        # 3) Apply deletion for not-kept opposite types (non-momentary only)
        # -------------------------
        filtered: List[Dict[str, Any]] = []
        for r in cleaned:
            if not isinstance(r, dict):
                continue

            sid = _s(r.get("subject_id"))
            oid = _s(r.get("object_id"))
            rt = _s(r.get("relation_type") or r.get("predicate"))
            persistence = _s(r.get("persistence") or "phase")

            if sid and oid and rt and persistence in NON_MOMENTARY:
                pk = type2pairkey.get(rt)
                if pk is not None:
                    a, b = (sid, oid) if sid <= oid else (oid, sid)
                    kept = pair2kept.get((a, b, pk), set(pk))
                    if rt not in kept:
                        # DELETE this edge
                        continue

            filtered.append(r)

        # -------------------------
        # 4) Merge buckets
        # -------------------------
        def _merge_key(r: Dict[str, Any]) -> Tuple[str, str, str]:
            rt = _s(r.get("relation_type") or r.get("predicate"))
            sid = _s(r.get("subject_id"))
            oid = _s(r.get("object_id"))
            if rt and sid and oid:
                return (rt, sid, oid)
            return (rt, _s(r.get("subject")), _s(r.get("object")))

        buckets: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
        for idx, r in enumerate(filtered):
            if not isinstance(r, dict):
                continue
            persistence = _s(r.get("persistence") or "phase").lower()
            # Keep momentary edges as independent facts.
            if persistence == "momentary":
                rid = _s(r.get("id"))
                doc_id = _s(r.get("document_id"))
                buckets[("__NO_MERGE__", rid or f"{doc_id}#{idx}")].append(r)
                continue
            buckets[_merge_key(r)].append(r)

        merged: List[Dict[str, Any]] = []
        for _k, group in tqdm(buckets.items(), "Merging Relations", total=len(buckets)):
            base = dict(group[0])

            # source_documents aggregate from document_id
            src_docs: List[str] = []
            seen: Set[str] = set()
            for r in group:
                ck = _s(r.get("document_id"))
                if ck and ck not in seen:
                    seen.add(ck)
                    src_docs.append(ck)
            base["source_documents"] = src_docs

            # longest relation_name / description
            best_rn = ""
            best_desc = ""
            for r in group:
                best_rn = _pick_longest(best_rn, _s(r.get("relation_name")))
                best_desc = _pick_longest(best_desc, _s(r.get("description")))
            base["relation_name"] = best_rn
            base["description"] = best_desc

            # drop chunk_ids
            base.pop("chunk_ids", None)

            merged.append(base)

        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(merged, f, ensure_ascii=False, indent=2)

        return merged

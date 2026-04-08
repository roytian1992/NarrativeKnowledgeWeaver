# core/utils/general_utils.py
from __future__ import annotations

import json
import os
import pickle
import hashlib
import re
import threading
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Literal, Iterator, TypeVar
from collections import defaultdict

import math
import statistics
import time
import networkx as nx


_ZH_RE = re.compile(r"[\u4e00-\u9fff]")
_EN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]*")
_TOKEN_SPLIT_RE = re.compile(r"[\s,.;:!?，。；：！？/|()（）【】\\[\\]<>《》\"'“”‘’]+")


def detect_lang(text: str) -> str:
    zh = len(_ZH_RE.findall(text or ""))
    en = len(_EN_RE.findall(text or ""))
    return "zh" if zh > en else "en"


def _extract_keywords_en(text: str, top_k: int = 10) -> List[str]:
    try:
        import yake

        extractor = yake.KeywordExtractor(lan="en", n=3, dedupLim=0.9, top=top_k)
        kws = extractor.extract_keywords(text)
        return [str(kw).strip() for kw, _ in kws if str(kw).strip()]
    except Exception:
        words = _EN_RE.findall(text or "")
        seen: Set[str] = set()
        out: List[str] = []
        for word in words:
            key = word.lower()
            if len(key) < 3 or key in seen:
                continue
            seen.add(key)
            out.append(word)
            if len(out) >= top_k:
                break
        return out


def _extract_keywords_zh(text: str, top_k: int = 10) -> List[str]:
    try:
        import jieba.analyse

        pairs = jieba.analyse.extract_tags(text, topK=top_k, withWeight=True)
        return [str(kw).strip() for kw, _ in pairs if str(kw).strip()]
    except Exception:
        chars = _ZH_RE.findall(text or "")
        out: List[str] = []
        seen: Set[str] = set()
        for n in (2, 3, 4):
            for i in range(len(chars) - n + 1):
                gram = "".join(chars[i : i + n])
                if gram in seen:
                    continue
                seen.add(gram)
                out.append(gram)
                if len(out) >= top_k:
                    return out
        return out[:top_k]


def extract_keywords(text: str, top_k: int = 10) -> List[str]:
    raw = str(text or "").strip()
    if not raw:
        return []
    if detect_lang(raw) == "zh":
        return _extract_keywords_zh(raw, top_k=top_k)
    return _extract_keywords_en(raw, top_k=top_k)


def build_search_keywords(text: str, top_k: int = 10) -> List[str]:
    raw = str(text or "").strip()
    if not raw:
        return []

    candidates: List[str] = [raw]
    candidates.extend(extract_keywords(raw, top_k=top_k))
    candidates.extend(part.strip() for part in _TOKEN_SPLIT_RE.split(raw) if part.strip())
    candidates.extend(_EN_RE.findall(raw))

    seen: Set[str] = set()
    out: List[str] = []
    for item in candidates:
        token = str(item or "").strip()
        if not token:
            continue
        lowered = token.lower()
        if lowered in seen:
            continue
        if len(lowered) < 2 and not _ZH_RE.search(token):
            continue
        seen.add(lowered)
        out.append(token)
        if len(out) >= max(1, top_k):
            break
    return out


def compress_query_for_vector_search(text: str, top_k: int = 8) -> str:
    keywords = build_search_keywords(text, top_k=top_k)
    if not keywords:
        return str(text or "").strip()
    if len(keywords) > 1:
        return " ".join(keywords[1:])
    return keywords[0]


def tokenize_mixed_text(text: Any) -> List[str]:
    raw = str(text or "").strip().lower()
    if not raw:
        return []
    out: List[str] = []
    buf: List[str] = []
    for ch in raw:
        if ch.isascii() and (ch.isalnum() or ch in {"_", "-"}):
            buf.append(ch)
            continue
        if buf:
            out.append("".join(buf))
            buf = []
        if "\u4e00" <= ch <= "\u9fff":
            out.append(ch)
    if buf:
        out.append("".join(buf))
    return [item for item in out if item]


def token_jaccard_overlap(left: Any, right: Any) -> float:
    lt = set(tokenize_mixed_text(left))
    rt = set(tokenize_mixed_text(right))
    if not lt or not rt:
        return 0.0
    return float(len(lt & rt)) / float(len(lt | rt))



def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p


def now_ms() -> int:
    return int(time.time() * 1000)


def json_dump_atomic(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp.{now_ms()}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


_JSONL_LOCKS: Dict[str, threading.Lock] = {}


def _jsonl_lock(path: str) -> threading.Lock:
    key = os.path.abspath(str(path or ""))
    lock = _JSONL_LOCKS.get(key)
    if lock is None:
        lock = threading.Lock()
        _JSONL_LOCKS[key] = lock
    return lock


def append_jsonl(path: str, row: Dict[str, Any]) -> None:
    if not path:
        raise ValueError("path must not be empty")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = dict(row or {})
    line = json.dumps(payload, ensure_ascii=False)
    with _jsonl_lock(path):
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    if not path or not os.path.exists(path):
        return []
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            text = str(raw or "").strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except Exception:
                continue
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def safe_json_loads(s: Any) -> Any:
    if isinstance(s, (dict, list)):
        return s
    if not isinstance(s, str):
        return None
    ss = s.strip()
    if not ss:
        return None
    return json.loads(ss)


def load_json_or_default(path: str, default: Any) -> Any:
    """
    Load JSON from path. Return default if file does not exist or JSON is invalid.
    """
    try:
        return load_json(path)
    except Exception:
        return default


def parse_json_object_from_text(text: Any) -> Optional[Dict[str, Any]]:
    """
    Parse first valid JSON object from raw text.

    Supports:
    - pure JSON object string
    - markdown ```json blocks
    - first {...} span fallback
    """
    if not isinstance(text, str):
        return None
    raw = text.strip()
    if not raw:
        return None
    try:
        obj = safe_json_loads(raw)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    marker = "```json"
    if marker in raw and "```" in raw:
        try:
            block = raw.split(marker, 1)[1].split("```", 1)[0].strip()
            obj = safe_json_loads(block)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    l = raw.find("{")
    r = raw.rfind("}")
    if l >= 0 and r > l:
        try:
            obj = safe_json_loads(raw[l : r + 1])
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    return None


def clamp_float(x: Any, *, low: float = 0.0, high: float = 1.0, default: float = 0.0) -> float:
    try:
        v = float(x)
    except Exception:
        v = default
    if low > high:
        low, high = high, low
    return max(low, min(high, v))


def cosine_sim(a: List[float], b: List[float]) -> Optional[float]:
    if not a or not b:
        return None
    if len(a) != len(b):
        return None
    dot = 0.0
    na = 0.0
    nb = 0.0
    for i in range(len(a)):
        x = float(a[i])
        y = float(b[i])
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0.0 or nb <= 0.0:
        return None
    return dot / (math.sqrt(na) * math.sqrt(nb))



# =============================================================================
# JSON / Pickle IO
# =============================================================================

def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def dump_pickle(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path: str) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


# =============================================================================
# String / id helpers
# =============================================================================

def join_bullets(lines: Any) -> str:
    if not isinstance(lines, list):
        return ""
    cleaned = [x.strip() for x in lines if isinstance(x, str) and x.strip()]
    if not cleaned:
        return ""
    return "\n".join([f"- {x}" for x in cleaned])

def _to_vec_list(vecs) -> List[List[float]]:
    """
    Convert model.encode output to List[List[float]].
    Handles: list, numpy.ndarray, torch.Tensor, single-vector outputs.
    """
    if vecs is None:
        return []

    # torch.Tensor
    if hasattr(vecs, "detach") and hasattr(vecs, "cpu"):
        vecs = vecs.detach().cpu()

    # numpy / torch -> list
    if hasattr(vecs, "tolist"):
        vecs = vecs.tolist()

    # now vecs should be list
    if not isinstance(vecs, list):
        return []

    # single vector: [d]
    if vecs and isinstance(vecs[0], (int, float)):
        return [ [float(x) for x in vecs] ]

    # batch: [[d], [d], ...]
    out: List[List[float]] = []
    for v in vecs:
        if isinstance(v, list) and v:
            out.append([float(x) for x in v])
    return out

def safe_title(title: str) -> str:
    t = (title or "").strip()
    return t if t else "<UNTITLED>"


def safe_str(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x.strip()
    return str(x).strip()

def pretty_json(x: Any) -> str:
    return json.dumps(x, ensure_ascii=False, indent=2)

def is_nonempty_str(x: Any) -> bool:
    return isinstance(x, str) and bool(x.strip())


def stable_id(key: str, prefix: str) -> str:
    return prefix + hashlib.md5(key.encode("utf-8")).hexdigest()[:12]


def safe_list(v: Any) -> List[Any]:
    return v if isinstance(v, list) else []


def safe_dict(v: Any) -> Dict[str, Any]:
    return v if isinstance(v, dict) else {}

def truncate_by_word_len(s: str, max_len: int, *, lang: Lang = "auto", suffix: str = "...") -> str:
    """
    Truncate string by word_len (en: words, zh: hanzi, auto: words+hanzi).
    Keeps original punctuation/spacing up to the cut position.
    """
    _EN_WORD_RE = re.compile(r"\b[A-Za-z0-9]+\b")
    # 中文汉字
    _ZH_CHAR_RE = re.compile(r"[\u4e00-\u9fff]")
    # auto：把“英文词”或“中文汉字”都当作 1 个单位
    _AUTO_UNIT_RE = re.compile(r"\b[A-Za-z0-9]+\b|[\u4e00-\u9fff]")

    s = (s or "").strip()
    if not s or max_len <= 0:
        return ""

    if word_len(s, lang=lang) <= max_len:
        return s

    if lang == "en":
        it = _EN_WORD_RE.finditer(s)
    elif lang == "zh":
        it = _ZH_CHAR_RE.finditer(s)
    else:
        it = _AUTO_UNIT_RE.finditer(s)

    cut = 0
    count = 0
    for m in it:
        count += 1
        cut = m.end()
        if count >= max_len:
            break

    # 保险：如果没找到 match（极少数情况），退化为字符截断
    if cut <= 0:
        cut = min(len(s), max_len)

    return s[:cut].rstrip() + suffix


T = TypeVar("T")
K = TypeVar("K")


def stable_relation_id(subject_id: str, predicate: str, object_id: str, *, prefix: str = "rel_") -> str:
    """
    Stable relation id for any edge type.
    Use a delimiter that is unlikely to appear in ids.
    """
    s = safe_str(subject_id)
    p = safe_str(predicate)
    o = safe_str(object_id)
    return stable_id(f"{s}||{p}||{o}", prefix=prefix)


_WS_RE = re.compile(r"\s+")


def normalize_text_for_id(s: str) -> str:
    """
    Normalize text for stable hashing: trim + collapse whitespace.
    Keep case to avoid over-merging by default.
    """
    s = safe_str(s)
    if not s:
        return ""
    return _WS_RE.sub(" ", s).strip()


def stable_episode_id(name: str, description: str, *, prefix: str = "ep_") -> str:
    """
    Stable episode id derived from (name, description).
    """
    n = normalize_text_for_id(name)
    d = normalize_text_for_id(description)
    return stable_id(f"{n}||{d}", prefix=prefix)


def dedupe_list(xs: Any) -> List[Any]:
    """
    Dedupe while preserving order.
    Works for hashable items; non-hashables are kept (best-effort).
    """
    if not isinstance(xs, list):
        return []
    out: List[Any] = []
    seen: Set[Any] = set()
    for x in xs:
        try:
            if x in seen:
                continue
            seen.add(x)
            out.append(x)
        except TypeError:
            # unhashable, keep it (best-effort)
            out.append(x)
    return out


def dedupe_by_key(
    items: Any,
    key_fn: Callable[[Any], Any],
    *,
    prefer_fn: Optional[Callable[[Any, Any], Any]] = None,
) -> List[Any]:
    """
    Deduplicate items by key while preserving first-seen order.
    If prefer_fn is provided, it decides which item to keep for a key:
        prefer_fn(old_item, new_item) -> chosen_item
    """
    if not isinstance(items, list):
        return []
    out: List[Any] = []
    key2idx: Dict[Any, int] = {}
    for it in items:
        try:
            k = key_fn(it)
        except Exception:
            # if key fails, treat as unique
            out.append(it)
            continue

        if k not in key2idx:
            key2idx[k] = len(out)
            out.append(it)
        else:
            if prefer_fn is not None:
                idx = key2idx[k]
                try:
                    out[idx] = prefer_fn(out[idx], it)
                except Exception:
                    # if prefer fails, keep existing
                    pass
    return out


def get_doc_text(doc2chunks: Any, part_id: str, *, joiner: str = "\n") -> str:
    """
    Extract joined text for a document part from doc2chunks.

    Expected doc2chunks format:
      doc2chunks[part_id]["chunks"] = [{"content": "...", "metadata": {...}}, ...]
    """
    if not isinstance(doc2chunks, dict):
        return ""
    pid = safe_str(part_id)
    if not pid:
        return ""
    pack = doc2chunks.get(pid, {}) or {}
    chunks = pack.get("chunks") or []
    if not isinstance(chunks, list) or not chunks:
        return ""

    # keep deterministic order
    try:
        chunks_sorted = sorted(chunks, key=order_key)
    except Exception:
        chunks_sorted = chunks

    texts: List[str] = []
    for ch in chunks_sorted:
        if not isinstance(ch, dict):
            continue
        c = ch.get("content")
        if isinstance(c, str) and c.strip():
            texts.append(c.strip())

    return joiner.join(texts)

def _is_none_relation(pred: str) -> bool:
    t = (pred or "").strip().upper()
    return t in {"NONE", "NO_RELATION", "NULL", "N/A", "NA"}


def entity_belongs_to_part(ent: Any, part_id: str) -> bool:
    """
    Check whether an Entity (or dict-like) belongs to a given document part_id via source_documents.

    Compatible with:
      - Entity.source_documents (list[str])
      - dict["source_documents"]
    """
    pid = safe_str(part_id)
    if not pid:
        return False

    src_docs = None
    if hasattr(ent, "source_documents"):
        src_docs = getattr(ent, "source_documents")
    elif isinstance(ent, dict):
        src_docs = ent.get("source_documents")

    if not isinstance(src_docs, list):
        return False

    for x in src_docs:
        if isinstance(x, str) and x.strip() == pid:
            return True
    return False


def filter_entities_by_part(entities: Any, part_id: str) -> List[Any]:
    """
    Filter entities that belong to the given part_id (by source_documents).
    """
    if not isinstance(entities, list):
        return []
    pid = safe_str(part_id)
    if not pid:
        return []
    out: List[Any] = []
    for e in entities:
        try:
            if entity_belongs_to_part(e, pid):
                out.append(e)
        except Exception:
            continue
    return out


def format_entities_brief(
    entities: Any,
    *,
    bullet: str = "- ",
    max_items: Optional[int] = None,
) -> str:
    """
    Format entities into a brief bullet list used by prompts.

    Output fields are conservative:
      id, name, type, description
    """
    if not isinstance(entities, list) or not entities:
        return ""

    lines: List[str] = []
    n = 0
    for ent in entities:
        if max_items is not None and n >= int(max_items):
            break

        if hasattr(ent, "id"):
            eid = safe_str(getattr(ent, "id", ""))
            name = safe_str(getattr(ent, "name", ""))
            etype = getattr(ent, "type", "")
            desc = safe_str(getattr(ent, "description", ""))
        elif isinstance(ent, dict):
            eid = safe_str(ent.get("id", ""))
            name = safe_str(ent.get("name", ""))
            etype = ent.get("type", "")
            desc = safe_str(ent.get("description", ""))
        else:
            continue

        if isinstance(etype, list):
            etype_str = ", ".join([safe_str(x) for x in etype if safe_str(x)])
        else:
            etype_str = safe_str(etype)

        if not eid and not name:
            continue

        lines.append(f"{bullet}id: {eid}, name: {name}, type: {etype_str}, description: {desc}")
        n += 1

    return "\n".join(lines)


def batch_iter(xs: Any, batch_size: int) -> Iterator[List[Any]]:
    """
    Yield list batches from a list-like input.
    """
    if batch_size <= 0:
        batch_size = 1
    if not isinstance(xs, list):
        xs = list(xs) if xs is not None else []
    for i in range(0, len(xs), batch_size):
        yield xs[i : i + batch_size]


def canonical_pair(a: str, b: str) -> Tuple[str, str]:
    """
    Canonicalize a pair of ids (order-insensitive).
    """
    aa = safe_str(a)
    bb = safe_str(b)
    return (aa, bb) if aa <= bb else (bb, aa)


def dedupe_pairs(pairs: Any) -> List[Tuple[str, str]]:
    """
    Deduplicate pairs (order-insensitive) while preserving first-seen order.
    """
    if pairs is None:
        return []
    out: List[Tuple[str, str]] = []
    seen: Set[Tuple[str, str]] = set()
    for p in pairs:
        if not isinstance(p, (list, tuple)) or len(p) != 2:
            continue
        a, b = p[0], p[1]
        if not (isinstance(a, str) and isinstance(b, str)):
            a = safe_str(a)
            b = safe_str(b)
        if not a or not b or a == b:
            continue
        cp = canonical_pair(a, b)
        if cp in seen:
            continue
        seen.add(cp)
        out.append(cp)
    return out


def iter_pairs_from_buckets(buckets: Any) -> Iterator[Tuple[str, str]]:
    """
    Generate candidate pairs from buckets:
      buckets[key] = [episode_id1, episode_id2, ...]

    This is the recommended way to avoid O(N^2) all-pairs.
    """
    if not isinstance(buckets, dict):
        return
    emitted: Set[Tuple[str, str]] = set()
    for _, ids in buckets.items():
        if not isinstance(ids, list) or len(ids) < 2:
            continue
        ids_clean = [safe_str(x) for x in ids if safe_str(x)]
        # local dedupe
        ids_clean = dedupe_list(ids_clean)
        m = len(ids_clean)
        for i in range(m):
            for j in range(i + 1, m):
                a, b = ids_clean[i], ids_clean[j]
                if not a or not b or a == b:
                    continue
                cp = canonical_pair(a, b)
                if cp in emitted:
                    continue
                emitted.add(cp)
                yield cp

# =============================================================================
# Chunk metadata helpers (from graph_builder.py)
# =============================================================================

def order_key(chunk: Dict[str, Any]) -> Tuple[int, int]:
    md = chunk.get("metadata", {}) or {}
    order = md.get("order", None)
    if order is None:
        order = md.get("chunk_index", 0)
    chunk_index = md.get("chunk_index", 0)

    try:
        return (int(order), int(chunk_index))
    except Exception:
        return (0, 0)


def document_metadata_from_first_chunk(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Assume metadata is document-level and copied to each chunk.
    Take metadata from first chunk and drop obvious chunk-level fields.
    """
    if not chunks:
        return {}

    md = (chunks[0].get("metadata") or {}).copy()

    drop = {
        "chunk_index",
        "chunk_type",
        "partition",
        "total_doc_chunks",
        "total_description_chunks",
    }
    for k in list(md.keys()):
        if k in drop:
            md.pop(k, None)

    return md


def strip_part_suffix(document_id: str) -> str:
    """
    Remove trailing "_part_<n>" if present.
    Examples:
      "scene_11_part_1" -> "scene_11"
      "scene_10" -> unchanged
    """
    ck = (document_id or "").strip()
    ck = re.sub(r"_part_\d+\s*$", "", ck)
    return ck.strip()


# =============================================================================
# Text length utility (for chunking)
# =============================================================================

Lang = Literal["en", "zh", "auto"]


def word_len(s: str, lang: Lang = "auto") -> int:
    s = (s or "").strip()
    if not s:
        return 0

    if lang == "en":
        return len(re.findall(r"\b[A-Za-z0-9]+\b", s))

    if lang == "zh":
        return len(re.findall(r"[\u4e00-\u9fff]", s))

    zh_count = len(re.findall(r"[\u4e00-\u9fff]", s))
    en_count = len(re.findall(r"\b[A-Za-z0-9]+\b", s))
    if zh_count > en_count:
        return zh_count
    return en_count


# =============================================================================
# Centrality utilities
# =============================================================================

CentralityMetric = Literal["pagerank", "total_degree", "closeness", "betweenness"]


def node_attr(G: nx.Graph, n: Any, key: str, default: Any = None) -> Any:
    return (G.nodes.get(n, {}) or {}).get(key, default)


def get_nodes_by_type_and_scope(
    G: nx.Graph,
    *,
    node_type: Optional[str] = None,
    scope: Optional[str] = "global",
    type_key: str = "type",
    scope_key: str = "scope",
) -> List[Any]:
    out: List[Any] = []
    for n in G.nodes:
        if node_type is not None and node_attr(G, n, type_key) != node_type:
            continue
        if scope is not None and node_attr(G, n, scope_key) != scope:
            continue
        out.append(n)
    return out


def _extract_relation_type_from_edge_data(
    data: Any,
    *,
    relation_field: str = "relation",
    relation_type_field: str = "relation_type",
) -> Optional[str]:
    """
    Supports two edge data layouts:
      A) nested: data["relation"]["relation_type"]
      B) flat:   data["relation_type"]
    """
    if not isinstance(data, dict):
        return None

    rel = data.get(relation_field, None)
    if isinstance(rel, dict):
        rt = rel.get(relation_type_field, None)
        if isinstance(rt, str) and rt.strip():
            return rt.strip()

    rt2 = data.get(relation_type_field, None)
    if isinstance(rt2, str) and rt2.strip():
        return rt2.strip()

    return None


def filter_graph_by_relation_type(
    G: nx.Graph,
    *,
    exclude_relation_types: Optional[Set[str]] = None,
    include_relation_types: Optional[Set[str]] = None,
    relation_field: str = "relation",
    relation_type_field: str = "relation_type",
    keep_isolated_nodes: bool = True,
) -> nx.Graph:
    if include_relation_types and exclude_relation_types:
        raise ValueError("Provide only one of include_relation_types or exclude_relation_types.")

    H = G.__class__()
    if keep_isolated_nodes:
        H.add_nodes_from(G.nodes(data=True))

    def keep_edge(rtype: Optional[str]) -> bool:
        if include_relation_types is not None:
            return rtype in include_relation_types
        if exclude_relation_types is not None:
            return rtype not in exclude_relation_types
        return True

    if isinstance(G, (nx.MultiGraph, nx.MultiDiGraph)):
        for u, v, k, data in G.edges(keys=True, data=True):
            data = data or {}
            rtype = _extract_relation_type_from_edge_data(
                data,
                relation_field=relation_field,
                relation_type_field=relation_type_field,
            )

            if include_relation_types is not None and rtype is None:
                continue
            if not keep_edge(rtype):
                continue

            H.add_edge(u, v, key=k, **(data if isinstance(data, dict) else {}))
    else:
        for u, v, data in G.edges(data=True):
            data = data or {}
            rtype = _extract_relation_type_from_edge_data(
                data,
                relation_field=relation_field,
                relation_type_field=relation_type_field,
            )

            if include_relation_types is not None and rtype is None:
                continue
            if not keep_edge(rtype):
                continue

            H.add_edge(u, v, **(data if isinstance(data, dict) else {}))

    return H


def _to_simple_graph(
    G: nx.Graph,
    *,
    directed: Optional[bool] = None,
    edge_weight_attr: Optional[str] = None,
    multiedge_weight_mode: Literal["count", "sum"] = "count",
) -> nx.Graph:
    if directed is None:
        directed = G.is_directed()

    SG = nx.DiGraph() if directed else nx.Graph()
    SG.add_nodes_from(G.nodes(data=True))

    if isinstance(G, (nx.MultiGraph, nx.MultiDiGraph)):
        for u, v, k, data in G.edges(keys=True, data=True):
            data = data or {}
            w = 1.0

            if multiedge_weight_mode == "sum" and edge_weight_attr is not None:
                if isinstance(data, dict) and edge_weight_attr in data and data[edge_weight_attr] is not None:
                    try:
                        w = float(data[edge_weight_attr])
                    except Exception:
                        w = 1.0

            if SG.has_edge(u, v):
                if multiedge_weight_mode == "count":
                    SG[u][v]["weight"] = float(SG[u][v].get("weight", 1.0)) + 1.0
                else:
                    SG[u][v]["weight"] = float(SG[u][v].get("weight", 0.0)) + w
            else:
                if multiedge_weight_mode == "count":
                    SG.add_edge(u, v, weight=1.0)
                else:
                    SG.add_edge(u, v, weight=w)
    else:
        SG = G.copy()
        for u, v, data in SG.edges(data=True):
            if isinstance(data, dict) and "weight" not in data:
                if edge_weight_attr is not None and edge_weight_attr in data and data[edge_weight_attr] is not None:
                    try:
                        data["weight"] = float(data[edge_weight_attr])
                    except Exception:
                        data["weight"] = 1.0
                else:
                    data["weight"] = 1.0

    return SG


def compute_centrality(
    G: nx.Graph,
    metric: CentralityMetric,
    *,
    exclude_relation_types: Optional[Set[str]] = None,
    include_relation_types: Optional[Set[str]] = None,
    pagerank_alpha: float = 0.85,
    distance_weight: Optional[str] = None,
    multiedge_weight_mode: Literal["count", "sum"] = "count",
    edge_weight_attr: Optional[str] = None,
    relation_field: str = "relation",
    relation_type_field: str = "relation_type",
) -> Dict[Any, float]:
    metric = (metric or "").lower()  # type: ignore

    if include_relation_types is not None or exclude_relation_types is not None:
        G = filter_graph_by_relation_type(
            G,
            include_relation_types=include_relation_types,
            exclude_relation_types=exclude_relation_types,
            relation_field=relation_field,
            relation_type_field=relation_type_field,
            keep_isolated_nodes=True,
        )

    if metric == "total_degree":
        if G.is_directed():
            return {n: float(G.in_degree(n) + G.out_degree(n)) for n in G.nodes}
        return {n: float(G.degree(n)) for n in G.nodes}

    if metric == "pagerank":
        SG = _to_simple_graph(
            G,
            directed=G.is_directed(),
            edge_weight_attr=edge_weight_attr,
            multiedge_weight_mode=multiedge_weight_mode,
        )
        return nx.pagerank(SG, alpha=pagerank_alpha, weight="weight")

    if metric in ("closeness", "betweenness"):
        SG = _to_simple_graph(
            G,
            directed=G.is_directed(),
            edge_weight_attr=edge_weight_attr,
            multiedge_weight_mode=multiedge_weight_mode,
        )
        if metric == "closeness":
            return nx.closeness_centrality(SG, distance=distance_weight)
        return nx.betweenness_centrality(SG, weight=distance_weight, normalized=True)

    raise ValueError(f"Unknown metric: {metric}")


def filter_nodes_by_centrality(
    G: nx.Graph,
    *,
    metric: CentralityMetric,
    threshold: float,
    node_type: Optional[str] = None,
    scope: Optional[str] = "global",
    num_top: Optional[int] = None,
    exclude_relation_types: Optional[Set[str]] = None,
    include_relation_types: Optional[Set[str]] = None,
    pagerank_alpha: float = 0.85,
    distance_weight: Optional[str] = None,
    multiedge_weight_mode: Literal["count", "sum"] = "count",
    edge_weight_attr: Optional[str] = None,
    relation_field: str = "relation",
    relation_type_field: str = "relation_type",
) -> List[Tuple[Any, float]]:
    candidates = set(get_nodes_by_type_and_scope(G, node_type=node_type, scope=scope))

    scores = compute_centrality(
        G,
        metric,
        exclude_relation_types=exclude_relation_types,
        include_relation_types=include_relation_types,
        pagerank_alpha=pagerank_alpha,
        distance_weight=distance_weight,
        multiedge_weight_mode=multiedge_weight_mode,
        edge_weight_attr=edge_weight_attr,
        relation_field=relation_field,
        relation_type_field=relation_type_field,
    )

    selected = [(n, float(scores.get(n, 0.0))) for n in candidates if float(scores.get(n, 0.0)) > float(threshold)]
    selected.sort(key=lambda x: x[1], reverse=True)

    if num_top is not None:
        selected = selected[: int(num_top)]
    return selected


# =============================================================================
# Optional: score stats/report helpers (no plotting)
# =============================================================================

@dataclass
class CentralityStats:
    count: int
    min: float
    max: float
    mean: float
    median: float
    stdev: float
    p5: float
    p25: float
    p75: float
    p95: float


def _percentile(vals: List[float], p: float) -> float:
    if not vals:
        return float("nan")
    s = sorted(vals)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return s[int(k)]
    return s[f] * (c - k) + s[c] * (k - f)


def summarize_scores(scores: Dict[Any, float], nodes: Iterable[Any]) -> CentralityStats:
    vals = [float(scores.get(n, 0.0)) for n in nodes]
    vals = [v for v in vals if not math.isnan(v)]
    if not vals:
        return CentralityStats(
            0, float("nan"), float("nan"), float("nan"), float("nan"), float("nan"),
            float("nan"), float("nan"), float("nan"), float("nan")
        )

    stdev = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    return CentralityStats(
        count=len(vals),
        min=min(vals),
        max=max(vals),
        mean=sum(vals) / len(vals),
        median=statistics.median(vals),
        stdev=stdev,
        p5=_percentile(vals, 0.05),
        p25=_percentile(vals, 0.25),
        p75=_percentile(vals, 0.75),
        p95=_percentile(vals, 0.95),
    )


def export_centrality_report(
    G: nx.Graph,
    *,
    metric: CentralityMetric,
    node_type: Optional[str] = None,
    scope: Optional[str] = "global",
    top_k: int = 30,
    exclude_relation_types: Optional[Set[str]] = None,
    include_relation_types: Optional[Set[str]] = None,
    pagerank_alpha: float = 0.85,
    distance_weight: Optional[str] = None,
    multiedge_weight_mode: Literal["count", "sum"] = "count",
    edge_weight_attr: Optional[str] = None,
    relation_field: str = "relation",
    relation_type_field: str = "relation_type",
) -> Dict[str, Any]:
    nodes = get_nodes_by_type_and_scope(G, node_type=node_type, scope=scope)

    scores = compute_centrality(
        G,
        metric,
        exclude_relation_types=exclude_relation_types,
        include_relation_types=include_relation_types,
        pagerank_alpha=pagerank_alpha,
        distance_weight=distance_weight,
        multiedge_weight_mode=multiedge_weight_mode,
        edge_weight_attr=edge_weight_attr,
        relation_field=relation_field,
        relation_type_field=relation_type_field,
    )

    stats = summarize_scores(scores, nodes)

    rows: List[Dict[str, Any]] = []
    for n in nodes:
        rows.append(
            {
                "id": n,
                "name": node_attr(G, n, "name"),
                "type": node_attr(G, n, "type"),
                "scope": node_attr(G, n, "scope"),
                "score": float(scores.get(n, 0.0)),
                "total_degree": float(G.degree(n)) if not G.is_directed() else float(G.in_degree(n) + G.out_degree(n)),
            }
        )

    rows.sort(key=lambda r: r["score"], reverse=True)
    top_rows = rows[: int(top_k)]

    return {
        "metric": metric,
        "node_type": node_type,
        "scope": scope,
        "excluded_relation_types": sorted(list(exclude_relation_types)) if exclude_relation_types else [],
        "included_relation_types": sorted(list(include_relation_types)) if include_relation_types else [],
        "stats": stats.__dict__,
        "top_rows": top_rows,
        "all_rows": rows,
    }

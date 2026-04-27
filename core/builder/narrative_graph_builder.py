# -*- coding: utf-8 -*-
# core/builder/narrative_graph_builder.py
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Set, Tuple

from tqdm import tqdm

from core import KAGConfig
from core.builder.manager.narrative_analysis_manager import NarrativeManager
from core.model_providers.openai_llm import OpenAILLM
from core.storage.graph_store import GraphStore
from core.utils.graph_query_utils import GraphQueryUtils
from core.utils.format import DOC_TYPE_META
from core.utils.general_utils import (
    _is_none_relation,
    _to_vec_list,
    cosine_sim,
    dedupe_list,
    ensure_dir,
    filter_entities_by_part,
    format_entities_brief,
    get_doc_text,
    json_dump_atomic,
    load_json,
    safe_dict,
    safe_list,
    safe_str,
    stable_relation_id,
    word_len,
)

from core.algorithms.cycle_break import (
    export_relations_from_graph,
    load_causal_graph_cfg,
    load_cycle_break_cfg,
    run_heuristic,
    run_saber,
)
from core.algorithms.chain_extraction import extract_storyline_candidates

logger = logging.getLogger(__name__)

# Episode / Storyline support predicates
P_EP_CONTAINS = "EPISODE_CONTAINS"
P_STORYLINE_CONTAINS = "STORYLINE_CONTAINS"


class NarrativeGraphBuilder:
    """
    JSON-first narrative graph builder.

    Phase 1 outputs:
      - episodes/episodes_by_document/{doc_id}.json
      - global/episodes.json
      - global/episode_support_edges.json
      - global/episode_packs.json

    Phase 2 outputs:
      - episodes/candidate_pairs.json
      - episodes/episode_pairs/pair_*.json (optional)
      - global/episode_relations.json

    Cycle break output:
      - global/episode_relations_dag.json
    """

    def __init__(self, config: KAGConfig, *, doc_type: Optional[str] = None) -> None:
        self.config = config

        self.doc_type = (
            doc_type
            or safe_str(getattr(getattr(config, "global_config", None), "doc_type", ""))
            or "screenplay"
        )

        self.kg_dir = (
            safe_str(getattr(getattr(config, "knowledge_graph_builder", None), "file_path", ""))
            or "data/knowledge_graph"
        )
        self.base_dir = (
            safe_str(getattr(getattr(config, "narrative_graph_builder", None), "file_path", ""))
            or "data/narrative_graph"
        )

        self.max_workers = max(
            1,
            int(safe_str(getattr(getattr(config, "narrative_graph_builder", None), "max_workers", "32")) or "32"),
        )
        chain_cfg = getattr(getattr(config, "narrative_graph_builder", None), "chain_extraction", None)
        default_method = safe_str(getattr(chain_cfg, "method", "") or "trie").strip().lower()
        if default_method == "tri":
            default_method = "trie"
        if default_method not in {"trie", "mpc"}:
            default_method = "trie"
        self.storyline_method_default = default_method

        raw_enable_storyline_rel = getattr(getattr(config, "narrative_graph_builder", None), "enable_storyline_relations", True)
        if isinstance(raw_enable_storyline_rel, bool):
            self.enable_storyline_relations_default = raw_enable_storyline_rel
        else:
            self.enable_storyline_relations_default = safe_str(raw_enable_storyline_rel).strip().lower() not in {
                "",
                "0",
                "false",
                "no",
                "off",
            }

        self.llm = OpenAILLM(config)
        self.graph_store = GraphStore(config)
        self.graph_query_utils = GraphQueryUtils(self.graph_store, doc_type=self.doc_type)
        self.narrative_manager = NarrativeManager(config, self.llm)

        self.doc2chunks_path = os.path.join(self.kg_dir, "doc2chunks.json")

        self.out_episodes_dir = ensure_dir(os.path.join(self.base_dir, "episodes"))
        self.out_episodes_by_doc_dir = ensure_dir(os.path.join(self.out_episodes_dir, "episodes_by_document"))
        self.out_rel_pairs_dir = ensure_dir(os.path.join(self.out_episodes_dir, "episode_pairs"))

        self.out_global_dir = ensure_dir(os.path.join(self.base_dir, "global"))
        self.global_episodes_path = os.path.join(self.out_global_dir, "episodes.json")
        self.global_ep_support_edges_path = os.path.join(self.out_global_dir, "episode_support_edges.json")
        self.global_ep_packs_path = os.path.join(self.out_global_dir, "episode_packs.json")
        self.global_ep_ep_edges_path = os.path.join(self.out_global_dir, "episode_relations.json")
        self.dag_path = os.path.join(self.out_global_dir, "episode_relations_dag.json")
        self.global_storyline_candidates_path = os.path.join(self.out_global_dir, "storyline_candidates.json")
        self.global_storyline_chain_communities_path = os.path.join(self.out_global_dir, "storyline_chain_communities.json")
        self.global_storylines_path = os.path.join(self.out_global_dir, "storylines.json")
        self.global_storyline_support_edges_path = os.path.join(self.out_global_dir, "storyline_support_edges.json")
        self.global_storyline_relations_path = os.path.join(self.out_global_dir, "storyline_relations.json")

    # ================================================================
    # Phase 1
    # ================================================================
    def extract_episodes(
        self,
        *,
        document_node_types: Optional[List[str]] = None,
        limit_documents: Optional[int] = None,
        document_concurrency: Optional[int] = None,
        store_episode_support_edges: bool = True,
        ensure_episode_embeddings: bool = True,
        per_document_retries: int = 2,
        per_part_retries: int = 2,
        backoff_seconds: float = 0.8,
        jitter_seconds: float = 0.2,
        per_document_timeout: Optional[float] = None,  # kept for API compatibility, no hard-kill
        save_episodes_by_document: bool = True,
        merge_global_episodes: bool = True,
        save_global_support_edges: bool = True,
        save_global_packs: bool = True,
        embedding_text_field: str = "name_desc",
        embedding_batch_size: int = 256,
    ) -> List[Dict[str, Any]]:
        user_specified_doc_types = document_node_types is not None
        if document_node_types is None:
            meta = DOC_TYPE_META.get(self.doc_type, DOC_TYPE_META.get("general", {}))
            section_label = safe_str(meta.get("section_label", "Document")).strip() or "Document"
            document_node_types = [section_label]

        try:
            obj = load_json(self.doc2chunks_path)
            doc2chunks: Dict[str, Any] = obj if isinstance(obj, dict) else {}
        except Exception:
            logger.exception("[NarrativeGraph][P1] failed to load doc2chunks: %s", self.doc2chunks_path)
            doc2chunks = {}

        documents = self.graph_query_utils.fetch_all_nodes(node_types=document_node_types) or []
        if not documents:
            fallback_types: List[str] = []
            for v in (DOC_TYPE_META or {}).values():
                lb = safe_str((v or {}).get("section_label"))
                if lb and lb not in fallback_types:
                    fallback_types.append(lb)
            for lb in ["Document", "Scene", "Chapter"]:
                if lb not in fallback_types:
                    fallback_types.append(lb)

            if fallback_types:
                logger.warning(
                    "[NarrativeGraph][P1] no document nodes for node_types=%s (doc_type=%s). fallback labels=%s",
                    document_node_types,
                    self.doc_type,
                    fallback_types,
                )
                documents = self.graph_query_utils.fetch_all_nodes(node_types=fallback_types) or []
                if documents:
                    document_node_types = fallback_types
                    if user_specified_doc_types:
                        logger.warning(
                            "[NarrativeGraph][P1] user-specified document_node_types produced 0 docs; using fallback labels."
                        )
                else:
                    logger.warning(
                        "[NarrativeGraph][P1] fallback labels also returned 0 docs. check doc_type/node labels in the local graph."
                    )

        documents = documents[: int(limit_documents)] if limit_documents is not None else documents

        logger.info(
            "[NarrativeGraph][P1] documents=%d node_types=%s store_support=%s ensure_emb=%s base_dir=%s kg_dir=%s",
            len(documents),
            document_node_types,
            "yes" if store_episode_support_edges else "no",
            "yes" if ensure_episode_embeddings else "no",
            self.base_dir,
            self.kg_dir,
        )

        def _retry(run_fn, desc: str, n: int) -> Any:
            last = None
            for i in range(max(1, int(n))):
                try:
                    return run_fn()
                except Exception as e:
                    last = e
                    sleep = float(backoff_seconds) * (2**i)
                    if jitter_seconds > 0:
                        sleep += (hash(f"{desc}|{i}") % 1000) / 1000.0 * float(jitter_seconds)
                    logger.warning("[NarrativeGraph][retry] %s round=%d/%d err=%s", desc, i + 1, n, e)
                    if sleep > 0:
                        import time as _t

                        _t.sleep(sleep)
            raise last or RuntimeError(desc)

        merged_eps: List[Dict[str, Any]] = []
        merged_edges: List[Dict[str, Any]] = []
        merged_packs: List[Dict[str, Any]] = []

        workers = max(1, int(document_concurrency or self.max_workers))
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [
                ex.submit(
                    self._p1_one_doc,
                    doc,
                    doc2chunks,
                    per_document_retries,
                    per_part_retries,
                    _retry,
                    store_episode_support_edges,
                    save_episodes_by_document,
                )
                for doc in documents
            ]

            for fut in tqdm(as_completed(futs), total=len(futs), desc="Phase1 docs", unit="doc"):
                try:
                    eps, edges, packs = fut.result()
                    if eps:
                        merged_eps.extend(eps)
                    if store_episode_support_edges and edges:
                        merged_edges.extend(edges)
                    if packs:
                        merged_packs.extend(packs)
                except Exception as e:
                    logger.exception("[NarrativeGraph][P1] doc failed: %s", e)

        if merge_global_episodes:
            merged_eps = self._dedupe_entities_by_id(merged_eps)
            json_dump_atomic(self.global_episodes_path, merged_eps)

        if store_episode_support_edges and save_global_support_edges:
            merged_edges = self._dedupe_relations(merged_edges)
            json_dump_atomic(self.global_ep_support_edges_path, merged_edges)

        if save_global_packs:
            merged_packs = self._dedupe_packs_by_episode_id(merged_packs)
            json_dump_atomic(self.global_ep_packs_path, merged_packs)

        if ensure_episode_embeddings and merged_eps:
            try:
                merged_eps = self._ensure_episode_embeddings(
                    episodes=merged_eps,
                    embedding_text_field=embedding_text_field,
                    batch_size=embedding_batch_size,
                    save_path=self.global_episodes_path,
                )
            except Exception as e:
                logger.exception("[NarrativeGraph][P1] ensure embeddings failed: %s", e)

        logger.info(
            "[NarrativeGraph][P1] done: episodes=%d support_edges=%d packs=%d",
            len(merged_eps),
            len(merged_edges),
            len(merged_packs),
        )
        return merged_eps

    def _p1_one_doc(
        self,
        doc_node: Dict[str, Any],
        doc2chunks: Dict[str, Any],
        per_document_retries: int,
        per_part_retries: int,
        retry_fn,
        store_support: bool,
        save_by_doc: bool,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        doc_id = safe_str(doc_node.get("id"))
        if not doc_id:
            return [], [], []

        def _run_once():
            return self._p1_one_doc_once(doc_node, doc2chunks, per_part_retries, retry_fn, store_support)

        try:
            eps, edges, packs = retry_fn(_run_once, f"phase1 doc={doc_id}", per_document_retries)
        except Exception as e:
            logger.exception("[NarrativeGraph][P1] doc failed after retries: %s err=%s", doc_id, e)
            return [], [], []

        if save_by_doc and eps:
            try:
                json_dump_atomic(os.path.join(self.out_episodes_by_doc_dir, f"{doc_id}.json"), eps)
            except Exception as e:
                logger.warning("[NarrativeGraph][P1] save episodes_by_doc failed: %s err=%s", doc_id, e)

        return eps, edges, packs

    def _p1_one_doc_once(
        self,
        doc_node: Dict[str, Any],
        doc2chunks: Dict[str, Any],
        per_part_retries: int,
        retry_fn,
        store_support: bool,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        doc_id = safe_str(doc_node.get("id"))
        props = safe_dict(doc_node.get("properties"))
        src_parts = [safe_str(x) for x in safe_list(doc_node.get("source_documents")) if safe_str(x)]
        if not src_parts:
            src_parts = [safe_str(x) for x in safe_list(props.get("source_documents")) if safe_str(x)]
        if not src_parts:
            src_parts = [safe_str(x) for x in safe_list(doc_node.get("parts")) if safe_str(x)]
        if not src_parts:
            src_parts = [safe_str(x) for x in safe_list(props.get("parts")) if safe_str(x)]
        if not doc_id or not src_parts:
            return [], [], []

        all_eo = self.graph_query_utils.search_related_entities(source_id=doc_id, entity_types=["Event", "Occasion"]) or []
        if not all_eo:
            return [], [], []

        reached: List[Any] = []
        existing: List[Dict[str, Any]] = []
        packed_parts = self._pack_source_parts_for_episode_extraction(src_parts, doc2chunks)

        for i, part_group in enumerate(packed_parts):
            texts: List[str] = []
            for part_id in part_group:
                reached.extend(filter_entities_by_part(all_eo, part_id))
                part_text = safe_str(get_doc_text(doc2chunks, part_id)).strip()
                if part_text:
                    texts.append(part_text)
            text = "\n\n".join(texts).strip()
            if not safe_str(text):
                continue

            def _one_call() -> List[Dict[str, Any]]:
                entity_info = format_entities_brief(reached)
                if i == 0 or not existing:
                    out = self.narrative_manager.extract_episodes(text=text, entities=entity_info, goal="extract")
                else:
                    out = self.narrative_manager.extract_episodes(
                        text=text,
                        entities=entity_info,
                        goal="update",
                        existing_episodes=json.dumps(existing, ensure_ascii=False, indent=2),
                    )
                try:
                    parsed = json.loads(out.strip()) if isinstance(out, str) else out
                except Exception:
                    parsed = None
                if isinstance(parsed, dict) and isinstance(parsed.get("episodes"), list):
                    return [x for x in parsed["episodes"] if isinstance(x, dict)]
                return existing

            desc = f"extract_episodes doc={doc_id} parts={','.join(part_group)}"
            try:
                existing = retry_fn(_one_call, desc, per_part_retries)
            except Exception as e:
                logger.exception("[NarrativeGraph][P1] %s failed: %s", desc, e)

        packs, eps, edges = self._build_episode_entities_and_edges(doc_id, src_parts, existing)
        return eps, (edges if store_support else []), packs

    def _pack_source_parts_for_episode_extraction(
        self,
        source_parts: List[str],
        doc2chunks: Dict[str, Any],
        *,
        short_part_word_threshold: int = 420,
        max_pack_words: int = 1400,
    ) -> List[List[str]]:
        packed: List[List[str]] = []
        current_group: List[str] = []
        current_words = 0

        for part_id in source_parts or []:
            text = safe_str(get_doc_text(doc2chunks, part_id))
            words = word_len(text)
            if words <= 0:
                continue

            # Keep longer parts isolated; only pack short neighboring parts to reduce update rounds.
            if words > int(short_part_word_threshold):
                if current_group:
                    packed.append(current_group)
                    current_group = []
                    current_words = 0
                packed.append([part_id])
                continue

            if current_group and current_words + words > int(max_pack_words):
                packed.append(current_group)
                current_group = []
                current_words = 0

            current_group.append(part_id)
            current_words += words

        if current_group:
            packed.append(current_group)

        if not packed:
            return [[p] for p in source_parts if safe_str(p)]
        return packed

    def _build_episode_entities_and_edges(
        self,
        doc_id: str,
        source_documents: List[str],
        episodes: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        eps_out: List[Dict[str, Any]] = []
        edges: List[Dict[str, Any]] = []
        packs: List[Dict[str, Any]] = []

        def _ep_id(name: str, desc: str) -> str:
            key = f"{safe_str(doc_id)}||{safe_str(name)}||{safe_str(desc)}"
            return "ep_" + hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]

        def _ctx_ids(ev_id: str) -> Tuple[List[str], List[str], List[str]]:
            def _ids(xs: List[Any]) -> List[str]:
                out: List[str] = []
                for x in xs or []:
                    if hasattr(x, "id"):
                        out.append(safe_str(getattr(x, "id", "")))
                    elif isinstance(x, dict):
                        out.append(safe_str(x.get("id")))
                return [y for y in out if y]

            chars = self.graph_query_utils.search_related_entities(ev_id, entity_types=["Character"]) or []
            locs = self.graph_query_utils.search_related_entities(ev_id, entity_types=["Location"]) or []
            times = self.graph_query_utils.search_related_entities(ev_id, entity_types=["TimePoint"]) or []
            return dedupe_list(_ids(chars)), dedupe_list(_ids(locs)), dedupe_list(_ids(times))

        temp: List[Dict[str, Any]] = []
        all_ref_ids: Set[str] = set()

        for ep in episodes or []:
            if not isinstance(ep, dict):
                continue
            name, desc = safe_str(ep.get("name")), safe_str(ep.get("description"))
            if not name and not desc:
                continue

            eid = _ep_id(name, desc)
            ev_ids = dedupe_list([safe_str(x) for x in safe_list(ep.get("related_events")) if safe_str(x)])
            oc_ids = dedupe_list([safe_str(x) for x in safe_list(ep.get("related_occasions")) if safe_str(x)])

            ch_ids: List[str] = []
            loc_ids: List[str] = []
            tm_ids: List[str] = []
            for ev_id in ev_ids:
                c, l, t = _ctx_ids(ev_id)
                ch_ids.extend(c)
                loc_ids.extend(l)
                tm_ids.extend(t)

            ch_ids = dedupe_list([x for x in ch_ids if x])
            loc_ids = dedupe_list([x for x in loc_ids if x])
            tm_ids = dedupe_list([x for x in tm_ids if x])

            all_ref_ids.update(ev_ids)
            all_ref_ids.update(oc_ids)
            all_ref_ids.update(ch_ids)
            all_ref_ids.update(loc_ids)
            all_ref_ids.update(tm_ids)

            temp.append(
                {
                    "eid": eid,
                    "name": name,
                    "desc": desc,
                    "ev_ids": ev_ids,
                    "oc_ids": oc_ids,
                    "ch_ids": ch_ids,
                    "loc_ids": loc_ids,
                    "tm_ids": tm_ids,
                }
            )

        ref_map = self.graph_query_utils.get_entities_by_ids(list(all_ref_ids)) or {}

        def _name(_id: str) -> str:
            v = ref_map.get(_id)
            if v is None:
                return ""
            if hasattr(v, "name"):
                return safe_str(getattr(v, "name", ""))
            if isinstance(v, dict):
                return safe_str(v.get("name"))
            return ""

        def _edge(pred: str, subj_id: str, obj_id: str, rel_name: str, desc_txt: str, prefix: str) -> Dict[str, Any]:
            return {
                "id": stable_relation_id(subj_id, pred, obj_id, prefix=prefix),
                "subject_id": subj_id,
                "object_id": obj_id,
                "predicate": pred,
                "relation_name": rel_name,
                "confidence": 1.0,
                "description": desc_txt,
                "source_documents": list(source_documents),
                "properties": {},
            }

        for item in temp:
            eid = item["eid"]
            name = item["name"]
            desc = item["desc"]
            ev_ids = item["ev_ids"]
            oc_ids = item["oc_ids"]
            ch_ids = item["ch_ids"]
            loc_ids = item["loc_ids"]
            tm_ids = item["tm_ids"]

            props: Dict[str, Any] = {}
            if ev_ids:
                props["related_events"] = [n for n in (_name(x) for x in ev_ids) if n]
            if oc_ids:
                props["related_occasions"] = [n for n in (_name(x) for x in oc_ids) if n]
            if ch_ids:
                props["related_characters"] = [n for n in (_name(x) for x in ch_ids) if n]
            if loc_ids:
                props["related_locations"] = [n for n in (_name(x) for x in loc_ids) if n]
            if tm_ids:
                props["related_timepoints"] = [n for n in (_name(x) for x in tm_ids) if n]

            ep_ent = {
                "id": eid,
                "name": name or eid,
                "type": ["Episode"],
                "aliases": [],
                "description": desc,
                "scope": "global",
                "source_documents": list(source_documents),
                "properties": props,
                "version": "default",
            }
            eps_out.append(ep_ent)

            edges.extend(
                _edge(P_EP_CONTAINS, eid, x, "contains", "Episode contains the related Event.", "rel_ep_ev_")
                for x in ev_ids
            )
            edges.extend(
                _edge(
                    P_EP_CONTAINS,
                    eid,
                    x,
                    "contains",
                    "Episode contains the related Occasion.",
                    "rel_ep_oc_",
                )
                for x in oc_ids
            )
            edges.extend(
                _edge(
                    P_EP_CONTAINS,
                    eid,
                    x,
                    "contains",
                    "Episode contains the related Character context.",
                    "rel_ep_ch_",
                )
                for x in ch_ids
            )
            edges.extend(
                _edge(
                    P_EP_CONTAINS,
                    eid,
                    x,
                    "contains",
                    "Episode contains the related Location context.",
                    "rel_ep_loc_",
                )
                for x in loc_ids
            )
            edges.extend(
                _edge(
                    P_EP_CONTAINS,
                    eid,
                    x,
                    "contains",
                    "Episode contains the related TimePoint context.",
                    "rel_ep_time_",
                )
                for x in tm_ids
            )

            packs.append(
                {
                    "episode_entity": ep_ent,
                    "related_event_ids": ev_ids,
                    "related_occasion_ids": oc_ids,
                    "related_character_ids": ch_ids,
                    "related_location_ids": loc_ids,
                    "related_time_ids": tm_ids,
                    "doc_id": doc_id,
                    "source_documents": list(source_documents),
                }
            )

        return packs, eps_out, edges

    @staticmethod
    def _source_order_key(source_id: str) -> Tuple[Any, ...]:
        s = safe_str(source_id).strip()
        nums = [int(x) for x in re.findall(r"\d+", s)]
        if nums:
            padded = (nums + [0, 0, 0, 0])[:4]
            return tuple(padded + [s])
        return (10**9, 0, 0, 0, s)

    def _load_episode_local_order_index(self) -> Dict[str, int]:
        order: Dict[str, int] = {}
        try:
            filenames = sorted(os.listdir(self.out_episodes_by_doc_dir))
        except Exception:
            return order

        for filename in filenames:
            if not filename.endswith(".json"):
                continue
            path = os.path.join(self.out_episodes_by_doc_dir, filename)
            try:
                obj = load_json(path)
            except Exception:
                continue
            items = [x for x in obj if isinstance(x, dict)] if isinstance(obj, list) else []
            for idx, ep in enumerate(items):
                eid = safe_str(ep.get("id"))
                if eid and eid not in order:
                    order[eid] = int(idx)
        return order

    def _episode_order_key(
        self,
        episode: Dict[str, Any],
        *,
        local_order_index: Optional[Dict[str, int]] = None,
    ) -> Optional[Tuple[Any, ...]]:
        if not isinstance(episode, dict):
            return None
        docs = [safe_str(x) for x in safe_list(episode.get("source_documents")) if safe_str(x)]
        props = safe_dict(episode.get("properties"))
        doc_id = safe_str(props.get("doc_id"))
        if not docs and doc_id:
            docs = [doc_id]
        if not docs:
            return None

        base = min(self._source_order_key(x) for x in docs)
        eid = safe_str(episode.get("id"))
        local_idx = (local_order_index or {}).get(eid, 0)
        return tuple(list(base) + [int(local_idx), eid])

    @staticmethod
    def _description_implies_reverse_cause(description: str, subject_id: str, object_id: str) -> bool:
        desc = safe_str(description).lower()
        s = safe_str(subject_id).lower()
        o = safe_str(object_id).lower()
        if not desc or not s or not o or s not in desc or o not in desc:
            return False
        s_pos = desc.find(s)
        o_pos = desc.find(o)
        if o_pos < 0 or s_pos < 0 or o_pos >= s_pos:
            return False
        between = desc[o_pos:s_pos]
        causal_cues = (
            "cause",
            "trigger",
            "lead to",
            "leads to",
            "led to",
            "result in",
            "results in",
            "resulting in",
            "prompt",
            "force",
            "enable",
            "produce",
            "bring about",
            "gives rise",
        )
        return any(cue in between for cue in causal_cues)

    def _normalize_episode_relation_direction(
        self,
        relation: Dict[str, Any],
        *,
        ep_by_id: Dict[str, Dict[str, Any]],
        local_order_index: Optional[Dict[str, int]] = None,
        enabled: bool = True,
        check_types: Optional[Set[str]] = None,
    ) -> Dict[str, Any]:
        if not enabled or not isinstance(relation, dict):
            return relation

        rel_type = safe_str(relation.get("predicate") or relation.get("relation_type")).strip().lower()
        if check_types is not None and rel_type not in check_types:
            return relation

        s_id = safe_str(relation.get("subject_id"))
        o_id = safe_str(relation.get("object_id"))
        if not s_id or not o_id or s_id == o_id:
            return relation

        s_ep = ep_by_id.get(s_id)
        o_ep = ep_by_id.get(o_id)
        s_key = self._episode_order_key(s_ep or {}, local_order_index=local_order_index)
        o_key = self._episode_order_key(o_ep or {}, local_order_index=local_order_index)
        if s_key is None or o_key is None or s_key <= o_key:
            return relation

        props = safe_dict(relation.get("properties"))
        props["direction_sanity_order_conflict"] = True
        props["original_subject_order_key"] = list(s_key)
        props["original_object_order_key"] = list(o_key)

        if rel_type == "causes" and not self._description_implies_reverse_cause(
            safe_str(relation.get("description")),
            s_id,
            o_id,
        ):
            flagged = dict(relation)
            flagged["properties"] = props
            return flagged

        fixed = dict(relation)
        fixed["subject_id"], fixed["object_id"] = o_id, s_id
        fixed["id"] = stable_relation_id(o_id, safe_str(fixed.get("predicate")), s_id, prefix="rel_ep_ep_")
        props["direction_sanity_checked"] = True
        props["direction_sanity_action"] = "flipped_by_episode_order"
        props["original_subject_id"] = s_id
        props["original_object_id"] = o_id
        fixed["properties"] = props
        return fixed

    # ================================================================
    # Phase 2
    # ================================================================
    def extract_episode_relations(
        self,
        *,
        episodes_path: Optional[str] = None,
        packs_path: Optional[str] = None,
        episode_pair_concurrency: Optional[int] = None,
        max_episode_pairs_global: int = 200000,
        cross_document_only: bool = False,
        similarity_threshold: float = 0.5,
        ensure_episode_embeddings: bool = True,
        per_pair_retries: int = 2,
        backoff_seconds: float = 0.6,
        jitter_seconds: float = 0.2,
        save_pair_json: bool = True,
        show_pair_progress: bool = False,
        save_candidate_pairs: bool = True,
        candidate_pairs_path: Optional[str] = None,
        include_shared_neighbor_ids: bool = True,
        embedding_text_field: str = "name_desc",
        embedding_batch_size: int = 256,
        dynamic_similarity_threshold: Optional[bool] = None,
        min_candidate_pairs: Optional[int] = None,
        threshold_floor: Optional[float] = None,
        threshold_step: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        ep_path = episodes_path or self.global_episodes_path
        pk_path = packs_path or self.global_ep_packs_path
        cand_path = candidate_pairs_path or os.path.join(self.out_episodes_dir, "candidate_pairs.json")

        try:
            obj = load_json(ep_path)
            episodes = [x for x in obj if isinstance(x, dict)] if isinstance(obj, list) else []
        except Exception:
            logger.exception("[NarrativeGraph][P2] failed to load episodes: %s", ep_path)
            episodes = []

        try:
            obj = load_json(pk_path)
            packs = [x for x in obj if isinstance(x, dict)] if isinstance(obj, list) else []
        except Exception:
            logger.exception("[NarrativeGraph][P2] failed to load packs: %s", pk_path)
            packs = []

        if len(episodes) < 2 or len(packs) < 2:
            logger.warning(
                "[NarrativeGraph][P2] insufficient episodes/packs: episodes=%d packs=%d",
                len(episodes),
                len(packs),
            )
            if save_candidate_pairs:
                json_dump_atomic(cand_path, [])
            json_dump_atomic(self.global_ep_ep_edges_path, [])
            return []

        if ensure_episode_embeddings:
            try:
                episodes = self._ensure_episode_embeddings(
                    episodes=episodes,
                    embedding_text_field=embedding_text_field,
                    batch_size=embedding_batch_size,
                    save_path=ep_path,
                )
            except Exception as e:
                logger.exception("[NarrativeGraph][P2] ensure embeddings failed: %s", e)

        ep_by_id = {safe_str(e.get("id")): e for e in episodes if safe_str(e.get("id"))}
        pack_by_ep = {
            safe_str(p.get("episode_entity", {}).get("id")): p
            for p in packs
            if safe_str(p.get("episode_entity", {}).get("id"))
        }

        cfg_ngb = getattr(self.config, "narrative_graph_builder", None)
        valid_anchor_fields = [
            "related_event_ids",
            "related_occasion_ids",
            "related_character_ids",
            "related_location_ids",
            "related_time_ids",
        ]

        def _clean_anchor_fields(raw: Any, fallback: List[str]) -> List[str]:
            vals = raw if isinstance(raw, list) else fallback
            out: List[str] = []
            for item in vals or []:
                field = safe_str(item).strip()
                if field in valid_anchor_fields and field not in out:
                    out.append(field)
            return out or list(fallback)

        primary_anchor_fields = _clean_anchor_fields(
            getattr(cfg_ngb, "episode_relation_primary_anchor_fields", None),
            ["related_event_ids", "related_occasion_ids"],
        )
        context_anchor_fields = _clean_anchor_fields(
            getattr(cfg_ngb, "episode_relation_context_anchor_fields", None),
            ["related_character_ids", "related_location_ids", "related_time_ids"],
        )
        context_anchor_fields = [f for f in context_anchor_fields if f not in primary_anchor_fields]

        default_anchor_weights: Dict[str, float] = {
            "related_event_ids": 1.0,
            "related_occasion_ids": 2.5,
            "related_character_ids": 1.5,
            "related_location_ids": 0.6,
            "related_time_ids": 0.25,
        }
        anchor_weights = dict(default_anchor_weights)
        raw_anchor_weights = getattr(cfg_ngb, "episode_relation_anchor_weights", None)
        if isinstance(raw_anchor_weights, dict):
            for k, v in raw_anchor_weights.items():
                kk = safe_str(k).strip()
                if kk not in valid_anchor_fields:
                    continue
                try:
                    anchor_weights[kk] = float(v)
                except Exception:
                    continue

        max_primary_bucket_size = max(0, int(getattr(cfg_ngb, "episode_relation_max_primary_bucket_size", 24) or 24))
        max_context_bucket_size = max(0, int(getattr(cfg_ngb, "episode_relation_max_context_bucket_size", 12) or 12))
        topk_per_episode = max(0, int(getattr(cfg_ngb, "episode_relation_topk_per_episode", 24) or 24))
        min_weighted_score = float(getattr(cfg_ngb, "episode_relation_min_weighted_score", 1.0) or 0.0)

        buckets_by_field: Dict[str, Dict[str, List[str]]] = {
            field: defaultdict(list) for field in (primary_anchor_fields + context_anchor_fields)
        }
        episode_seed_rank: Dict[str, float] = {}

        for ep_id, p in pack_by_ep.items():
            seed_score = 0.0
            for field in (primary_anchor_fields + context_anchor_fields):
                ids = dedupe_list([safe_str(x) for x in safe_list(p.get(field)) if safe_str(x)])
                for nei in ids:
                    buckets_by_field[field][nei].append(ep_id)
                seed_score += float(anchor_weights.get(field, 1.0)) * len(ids)
            episode_seed_rank[ep_id] = seed_score

        pair_common: Dict[Tuple[str, str], int] = defaultdict(int)
        pair_shared: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
        pair_weighted: Dict[Tuple[str, str], float] = defaultdict(float)
        pair_primary: Dict[Tuple[str, str], int] = defaultdict(int)
        pair_context: Dict[Tuple[str, str], int] = defaultdict(int)
        pair_field_counts: Dict[Tuple[str, str], Dict[str, int]] = defaultdict(dict)
        candidate_stats: Dict[str, int] = {
            "primary_anchor_buckets": 0,
            "context_anchor_buckets": 0,
            "trimmed_primary_anchor_buckets": 0,
            "trimmed_context_anchor_buckets": 0,
            "trimmed_primary_anchor_episodes": 0,
            "trimmed_context_anchor_episodes": 0,
            "local_window_pairs_added": 0,
        }

        def _sorted_bucket_episode_ids(ep_ids: List[str], bucket_limit: int) -> List[str]:
            uniq = dedupe_list([safe_str(x) for x in (ep_ids or []) if safe_str(x)])
            uniq = [eid for eid in uniq if eid in pack_by_ep]
            if len(uniq) < 2:
                return uniq
            uniq.sort(key=lambda eid: (episode_seed_rank.get(eid, 0.0), eid), reverse=True)
            if bucket_limit > 0 and len(uniq) > bucket_limit:
                return uniq[:bucket_limit]
            return uniq

        def _accumulate_pairs_for_field(
            field: str,
            *,
            bucket_limit: int,
            is_primary: bool,
        ) -> None:
            bucket_map = buckets_by_field.get(field) or {}
            field_weight = float(anchor_weights.get(field, 1.0))
            for anchor_id, ep_ids in bucket_map.items():
                raw_ids = dedupe_list([safe_str(x) for x in (ep_ids or []) if safe_str(x)])
                if len(raw_ids) < 2:
                    continue
                if is_primary:
                    candidate_stats["primary_anchor_buckets"] += 1
                else:
                    candidate_stats["context_anchor_buckets"] += 1

                trimmed_ids = _sorted_bucket_episode_ids(raw_ids, bucket_limit)
                if bucket_limit > 0 and len(trimmed_ids) < len(raw_ids):
                    removed = len(raw_ids) - len(trimmed_ids)
                    if is_primary:
                        candidate_stats["trimmed_primary_anchor_buckets"] += 1
                        candidate_stats["trimmed_primary_anchor_episodes"] += removed
                    else:
                        candidate_stats["trimmed_context_anchor_buckets"] += 1
                        candidate_stats["trimmed_context_anchor_episodes"] += removed

                if len(trimmed_ids) < 2:
                    continue

                for i in range(len(trimmed_ids)):
                    for j in range(i + 1, len(trimmed_ids)):
                        a, b = trimmed_ids[i], trimmed_ids[j]
                        if not a or not b or a == b:
                            continue
                        key = (a, b) if a < b else (b, a)
                        pa, pb = pack_by_ep.get(key[0]), pack_by_ep.get(key[1])
                        if not pa or not pb:
                            continue
                        if cross_document_only and safe_str(pa.get("doc_id")) == safe_str(pb.get("doc_id")):
                            continue

                        pair_common[key] += 1
                        pair_weighted[key] += field_weight
                        if include_shared_neighbor_ids:
                            pair_shared[key].add(anchor_id)
                        if is_primary:
                            pair_primary[key] += 1
                        else:
                            pair_context[key] += 1

                        field_counts = pair_field_counts[key]
                        field_counts[field] = int(field_counts.get(field, 0)) + 1

        for field in primary_anchor_fields:
            _accumulate_pairs_for_field(
                field,
                bucket_limit=max_primary_bucket_size,
                is_primary=True,
            )

        if context_anchor_fields:
            for field in context_anchor_fields:
                _accumulate_pairs_for_field(
                    field,
                    bucket_limit=max_context_bucket_size,
                    is_primary=False,
                )

        include_local_window = bool(
            getattr(cfg_ngb, "episode_relation_include_local_window_candidates", False)
        )
        local_window_size = max(
            0, int(getattr(cfg_ngb, "episode_relation_local_window_size", 2) or 0)
        )
        local_window_weight = float(
            getattr(cfg_ngb, "episode_relation_local_window_weight", 0.75) or 0.0
        )
        if include_local_window and local_window_size > 0:
            local_order_for_candidates = self._load_episode_local_order_index()
            grouped_episode_ids: Dict[str, List[str]] = defaultdict(list)
            for ep_id, p in pack_by_ep.items():
                if cross_document_only:
                    continue
                docs = [safe_str(x) for x in safe_list(p.get("source_documents")) if safe_str(x)]
                group_key = safe_str(p.get("doc_id"))
                if not group_key and docs:
                    group_key = docs[0]
                if not group_key:
                    group_key = "__global__"
                grouped_episode_ids[group_key].append(ep_id)

            for ep_ids in grouped_episode_ids.values():
                ordered = [
                    eid
                    for eid in ep_ids
                    if eid in ep_by_id and self._episode_order_key(
                        ep_by_id.get(eid) or {},
                        local_order_index=local_order_for_candidates,
                    )
                    is not None
                ]
                ordered.sort(
                    key=lambda eid: self._episode_order_key(
                        ep_by_id.get(eid) or {},
                        local_order_index=local_order_for_candidates,
                    )
                )
                for i, a in enumerate(ordered):
                    for b in ordered[i + 1 : i + 1 + local_window_size]:
                        if not a or not b or a == b:
                            continue
                        key = (a, b) if a < b else (b, a)
                        if key not in pair_common:
                            candidate_stats["local_window_pairs_added"] += 1
                        pair_common[key] += 1
                        pair_weighted[key] += local_window_weight
                        pair_context[key] += 1
                        field_counts = pair_field_counts[key]
                        field_counts["local_order_window"] = int(field_counts.get("local_order_window", 0)) + 1

        if not pair_common:
            logger.info(
                "[NarrativeGraph][P2] candidate_pairs_by_neighbors=0 primary=%s context=%s",
                primary_anchor_fields,
                context_anchor_fields,
            )
            if save_candidate_pairs:
                json_dump_atomic(cand_path, [])
            json_dump_atomic(self.global_ep_ep_edges_path, [])
            return []

        ranked_pairs = list(pair_common.keys())
        if min_weighted_score > 0:
            ranked_pairs = [key for key in ranked_pairs if float(pair_weighted.get(key, 0.0)) >= min_weighted_score]

        ranked_pairs.sort(
            key=lambda key: (
                int(pair_common.get(key, 0)),
                float(pair_weighted.get(key, 0.0)),
                int(pair_primary.get(key, 0)),
                int(pair_context.get(key, 0)),
                key[0],
                key[1],
            ),
            reverse=True,
        )

        if topk_per_episode > 0:
            kept_pairs: List[Tuple[str, str]] = []
            episode_degree: Dict[str, int] = defaultdict(int)
            for key in ranked_pairs:
                a, b = key
                if episode_degree[a] >= topk_per_episode or episode_degree[b] >= topk_per_episode:
                    continue
                kept_pairs.append(key)
                episode_degree[a] += 1
                episode_degree[b] += 1
            cand_pairs = kept_pairs[: int(max_episode_pairs_global)]
        else:
            cand_pairs = ranked_pairs[: int(max_episode_pairs_global)]

        logger.info(
            "[NarrativeGraph][P2] candidate generation: primary=%s context=%s cand_raw=%d cand_kept=%d "
            "primary_buckets=%d context_buckets=%d trim_primary=%d/%d trim_context=%d/%d "
            "local_window_added=%d topk_per_ep=%d min_weight=%.2f strategy=shared_anchor_union",
            primary_anchor_fields,
            context_anchor_fields,
            len(ranked_pairs),
            len(cand_pairs),
            candidate_stats["primary_anchor_buckets"],
            candidate_stats["context_anchor_buckets"],
            candidate_stats["trimmed_primary_anchor_buckets"],
            candidate_stats["trimmed_primary_anchor_episodes"],
            candidate_stats["trimmed_context_anchor_buckets"],
            candidate_stats["trimmed_context_anchor_episodes"],
            candidate_stats["local_window_pairs_added"],
            topk_per_episode,
            min_weighted_score,
        )

        thr = (
            float(similarity_threshold)
            if similarity_threshold is not None
            else float(getattr(cfg_ngb, "episode_relation_similarity_threshold", 0.55) or 0.55)
        )
        dynamic_enabled = (
            bool(dynamic_similarity_threshold)
            if dynamic_similarity_threshold is not None
            else bool(getattr(cfg_ngb, "episode_relation_dynamic_threshold", True))
        )
        min_pairs_target = (
            max(0, int(min_candidate_pairs or 0))
            if min_candidate_pairs is not None
            else max(0, int(getattr(cfg_ngb, "episode_relation_min_candidate_pairs", 10) or 0))
        )
        thr_floor = (
            float(threshold_floor)
            if threshold_floor is not None
            else float(getattr(cfg_ngb, "episode_relation_threshold_floor", 0.20) or 0.20)
        )
        thr_step = (
            float(threshold_step)
            if threshold_step is not None
            else float(getattr(cfg_ngb, "episode_relation_threshold_step", 0.05) or 0.05)
        )
        backfill_by_similarity = bool(getattr(cfg_ngb, "episode_relation_backfill_by_similarity", True))
        backfill_max_episodes = max(2, int(getattr(cfg_ngb, "episode_relation_backfill_max_episodes", 500) or 500))
        thr_step = 0.05 if thr_step <= 0 else abs(thr_step)
        thr_floor = min(thr_floor, thr)
        context_requires_primary_pair = bool(getattr(cfg_ngb, "episode_relation_context_requires_primary_pair", True))
        context_only_similarity_threshold = max(
            thr,
            float(getattr(cfg_ngb, "episode_relation_context_only_similarity_threshold", 0.65) or 0.65),
        )
        context_only_min_weighted_score = float(
            getattr(cfg_ngb, "episode_relation_context_only_min_weighted_score", 1.5) or 0.0
        )
        context_only_require_character = bool(
            getattr(cfg_ngb, "episode_relation_context_only_require_character", True)
        )
        local_window_similarity_threshold = float(
            getattr(cfg_ngb, "episode_relation_local_window_similarity_threshold", 0.30) or 0.0
        )

        def _pair_anchor_policy(key: Tuple[str, str], sim: Optional[float]) -> Tuple[bool, str]:
            if int(pair_primary.get(key, 0)) > 0:
                return True, "primary_anchor"
            fields = pair_field_counts.get(key, {})
            if int(fields.get("local_order_window", 0)) > 0:
                if sim is None or float(sim) < local_window_similarity_threshold:
                    return False, "weak_local_order_similarity"
                return True, "local_order_window"
            if not context_requires_primary_pair:
                return True, "context_anchor"
            if context_only_require_character and int(fields.get("related_character_ids", 0)) <= 0:
                return False, "context_without_character"
            if float(pair_weighted.get(key, 0.0)) < context_only_min_weighted_score:
                return False, "weak_context_weight"
            if sim is None or float(sim) < context_only_similarity_threshold:
                return False, "weak_context_similarity"
            return True, "strong_context_anchor"

        def _filter_pairs_by_threshold(threshold: float) -> List[Tuple[str, str]]:
            out: List[Tuple[str, str]] = []
            for a, b in cand_pairs:
                key = (a, b) if a < b else (b, a)
                sim = pair_sim.get(key)
                if sim is None:
                    continue
                if threshold <= -1.0 or sim >= threshold:
                    keep, _reason = _pair_anchor_policy(key, sim)
                    if keep:
                        out.append(key)
            seen_local: Set[Tuple[str, str]] = set()
            return [p for p in out if not (p in seen_local or seen_local.add(p))]

        precomputed_pair_sim: Dict[Tuple[str, str], float] = {}

        if dynamic_enabled and min_pairs_target > 0 and len(cand_pairs) < min_pairs_target and backfill_by_similarity:
            episode_ids_for_backfill = [eid for eid in ep_by_id.keys() if eid in pack_by_ep]
            if len(episode_ids_for_backfill) <= backfill_max_episodes:
                semantic_candidates: List[Tuple[Tuple[str, str], float]] = []
                seen_cand_pairs = set(cand_pairs)
                for i in range(len(episode_ids_for_backfill)):
                    a = episode_ids_for_backfill[i]
                    ea = ep_by_id.get(a) or {}
                    va = ea.get("embedding")
                    if not isinstance(va, list):
                        continue
                    pa = pack_by_ep.get(a) or {}
                    for j in range(i + 1, len(episode_ids_for_backfill)):
                        b = episode_ids_for_backfill[j]
                        key = (a, b) if a < b else (b, a)
                        if key in seen_cand_pairs:
                            continue
                        pb = pack_by_ep.get(b) or {}
                        if cross_document_only and safe_str(pa.get("doc_id")) == safe_str(pb.get("doc_id")):
                            continue
                        eb = ep_by_id.get(b) or {}
                        vb = eb.get("embedding")
                        if not isinstance(vb, list):
                            continue
                        sim = cosine_sim(va, vb)
                        if sim is None:
                            continue
                        semantic_candidates.append((key, float(sim)))
                semantic_candidates.sort(key=lambda item: item[1], reverse=True)
                need = max(0, min_pairs_target - len(cand_pairs))
                added = 0
                for key, sim in semantic_candidates:
                    cand_pairs.append(key)
                    pair_common.setdefault(key, 0)
                    if include_shared_neighbor_ids:
                        pair_shared.setdefault(key, set())
                    precomputed_pair_sim[key] = sim
                    added += 1
                    if added >= need or len(cand_pairs) >= int(max_episode_pairs_global):
                        break
                if added > 0:
                    logger.info(
                        "[NarrativeGraph][P2] semantic backfill added=%d target=%d cand_total=%d",
                        added,
                        min_pairs_target,
                        len(cand_pairs),
                    )

        pair_sim: Dict[Tuple[str, str], float] = {}

        for a, b in cand_pairs:
            ea, eb = ep_by_id.get(a), ep_by_id.get(b)
            if not ea or not eb:
                continue
            key = (a, b) if a < b else (b, a)
            if key in precomputed_pair_sim:
                pair_sim[key] = float(precomputed_pair_sim[key])
                continue
            va, vb = ea.get("embedding"), eb.get("embedding")
            if not isinstance(va, list) or not isinstance(vb, list):
                continue
            sim = cosine_sim(va, vb)
            if sim is None:
                continue
            pair_sim[key] = float(sim)

        selected_pair_policy: Dict[Tuple[str, str], str] = {}

        def _scene_key(p: Dict[str, Any]) -> str:
            docs = safe_list(p.get("source_documents"))
            if docs:
                return safe_str(docs[0])
            return safe_str(p.get("doc_id"))

        def _causal_candidate_score(key: Tuple[str, str]) -> float:
            a, b = key
            pa, pb = pack_by_ep.get(a) or {}, pack_by_ep.get(b) or {}
            fields = pair_field_counts.get(key, {})
            same_doc = 1.0 if safe_str(pa.get("doc_id")) and safe_str(pa.get("doc_id")) == safe_str(pb.get("doc_id")) else 0.0
            same_scene = 1.0 if _scene_key(pa) and _scene_key(pa) == _scene_key(pb) else 0.0
            sim = float(pair_sim.get(key, 0.0) or 0.0)
            weight = float(pair_weighted.get(key, 0.0) or 0.0)
            return (
                2.0 * same_doc
                + 1.0 * same_scene
                + 0.8 * sim
                + 0.25 * int(fields.get("related_character_ids", 0))
                + 0.15 * int(fields.get("related_location_ids", 0))
                + 0.25 * int(fields.get("related_occasion_ids", 0))
                + 0.10 * int(fields.get("related_event_ids", 0))
                - 0.03 * max(0.0, weight - 1.5)
            )

        selection_mode = safe_str(
            getattr(cfg_ngb, "episode_relation_candidate_selection_mode", "threshold")
        ).strip().lower() or "threshold"
        candidate_budget_total = max(
            0, int(getattr(cfg_ngb, "episode_relation_candidate_budget_total", 0) or 0)
        )
        candidate_primary_budget = max(
            0, int(getattr(cfg_ngb, "episode_relation_candidate_primary_budget", 180) or 180)
        )
        direction_sanity_enabled = bool(
            getattr(cfg_ngb, "episode_relation_enable_direction_sanity_check", True)
        )
        raw_direction_check_types = getattr(cfg_ngb, "episode_relation_direction_sanity_check_types", ["causes"])
        if isinstance(raw_direction_check_types, str):
            direction_check_types = {safe_str(raw_direction_check_types).strip().lower()}
        else:
            direction_check_types = {
                safe_str(x).strip().lower()
                for x in safe_list(raw_direction_check_types)
                if safe_str(x).strip()
            }
        if not direction_check_types:
            direction_check_types = {"causes"}
        episode_local_order_index = self._load_episode_local_order_index() if direction_sanity_enabled else {}

        def _filter_pairs_causal_balanced(threshold: float) -> List[Tuple[str, str]]:
            eligible: List[Tuple[str, str]] = []
            seen_eligible: Set[Tuple[str, str]] = set()
            for a, b in cand_pairs:
                key = (a, b) if a < b else (b, a)
                if key in seen_eligible:
                    continue
                sim = pair_sim.get(key)
                if sim is None or float(sim) < threshold:
                    continue
                seen_eligible.add(key)
                eligible.append(key)

            if not candidate_budget_total or len(eligible) <= candidate_budget_total:
                out = eligible
            else:
                rank_index = {key: i for i, key in enumerate(cand_pairs)}
                primary = [key for key in eligible if int(pair_primary.get(key, 0)) > 0]
                context = [key for key in eligible if int(pair_primary.get(key, 0)) <= 0]
                primary.sort(key=lambda key: rank_index.get(key, 10**9))
                context.sort(
                    key=lambda key: (
                        _causal_candidate_score(key),
                        float(pair_sim.get(key, 0.0) or 0.0),
                        int(pair_common.get(key, 0)),
                    ),
                    reverse=True,
                )
                out = primary[:candidate_primary_budget]
                remaining = max(0, candidate_budget_total - len(out))
                out.extend(context[:remaining])

            selected_pair_policy.clear()
            for key in out:
                if int(pair_primary.get(key, 0)) > 0:
                    selected_pair_policy[key] = "primary_anchor"
                else:
                    selected_pair_policy[key] = "causal_balanced_context"
            out.sort(key=lambda key: (cand_pairs.index(key) if key in cand_pairs else 10**9))
            return out

        if selection_mode == "causal_balanced":
            filtered = _filter_pairs_causal_balanced(thr)
        else:
            filtered = _filter_pairs_by_threshold(thr)
            selected_pair_policy = {
                key: _pair_anchor_policy(key, pair_sim.get(key))[1] for key in filtered
            }
        selected_thr = thr

        if selection_mode != "causal_balanced" and dynamic_enabled and min_pairs_target > 0 and len(filtered) < min_pairs_target and pair_sim:
            trial_thr = thr
            while trial_thr - thr_step >= thr_floor - 1e-9:
                trial_thr = max(thr_floor, trial_thr - thr_step)
                trial_filtered = _filter_pairs_by_threshold(trial_thr)
                logger.info(
                    "[NarrativeGraph][P2] dynamic threshold fallback: base=%.3f try=%.3f filtered=%d target=%d",
                    thr,
                    trial_thr,
                    len(trial_filtered),
                    min_pairs_target,
                )
                filtered = trial_filtered
                selected_thr = trial_thr
                if len(filtered) >= min_pairs_target or trial_thr <= thr_floor + 1e-9:
                    break
            selected_pair_policy = {
                key: _pair_anchor_policy(key, pair_sim.get(key))[1] for key in filtered
            }

        logger.info(
            "[NarrativeGraph][P2] pairs: cand=%d filtered=%d thr=%.3f base_thr=%.3f "
            "context_only_thr=%.3f context_only_min_weight=%.2f cross_doc_only=%s dynamic=%s target=%d "
            "selection=%s budget=%d primary_budget=%d",
            len(cand_pairs),
            len(filtered),
            selected_thr,
            thr,
            context_only_similarity_threshold,
            context_only_min_weighted_score,
            "yes" if cross_document_only else "no",
            "yes" if dynamic_enabled else "no",
            min_pairs_target,
            selection_mode,
            candidate_budget_total,
            candidate_primary_budget,
        )

        if not filtered:
            if save_candidate_pairs:
                json_dump_atomic(cand_path, [])
            json_dump_atomic(self.global_ep_ep_edges_path, [])
            return []

        if save_candidate_pairs:
            try:
                to_save: List[Dict[str, Any]] = []
                for a, b in filtered:
                    item: Dict[str, Any] = {
                        "a_id": a,
                        "b_id": b,
                        "common_neighbors": int(pair_common.get((a, b), 0)),
                        "weighted_score": float(pair_weighted.get((a, b), 0.0)),
                        "primary_shared_anchors": int(pair_primary.get((a, b), 0)),
                        "context_shared_anchors": int(pair_context.get((a, b), 0)),
                        "similarity": pair_sim.get((a, b)),
                        "threshold_used": selected_thr,
                        "anchor_policy": selected_pair_policy.get(
                            (a, b), _pair_anchor_policy((a, b), pair_sim.get((a, b)))[1]
                        ),
                        "shared_anchor_breakdown": dict(pair_field_counts.get((a, b), {})),
                    }
                    if include_shared_neighbor_ids:
                        item["shared_neighbor_ids"] = sorted(list(pair_shared.get((a, b), set())))
                    to_save.append(item)
                json_dump_atomic(cand_path, to_save)
            except Exception as e:
                logger.exception("[NarrativeGraph][P2] save candidate_pairs failed: %s", e)

        def _ep_info(e: Dict[str, Any]) -> Dict[str, Any]:
            props = safe_dict(e.get("properties"))
            compact_props = {
                "doc_id": safe_str(props.get("doc_id")),
                "related_events": safe_list(props.get("related_events")),
                "related_occasions": safe_list(props.get("related_occasions")),
                "related_characters": safe_list(props.get("related_characters")),
                "related_locations": safe_list(props.get("related_locations")),
                "related_timepoints": safe_list(props.get("related_timepoints")),
            }
            compact_props = {k: v for k, v in compact_props.items() if v or (isinstance(v, str) and v)}
            return {
                "id": safe_str(e.get("id")),
                "name": safe_str(e.get("name")),
                "description": safe_str(e.get("description")),
                "properties": compact_props,
            }

        def _pair_file(a_id: str, b_id: str) -> str:
            return "pair_" + hashlib.md5(f"{a_id}||{b_id}".encode("utf-8")).hexdigest()[:16] + ".json"

        def _run_pair_once(a_id: str, b_id: str) -> Optional[Dict[str, Any]]:
            ea, eb = ep_by_id.get(a_id), ep_by_id.get(b_id)
            if not ea or not eb:
                return None

            out = self.narrative_manager.extract_narrative_relation(
                subject_entity_info=_ep_info(ea),
                object_entity_info=_ep_info(eb),
                entity_type="Episode",
            )

            try:
                parsed = json.loads(out.strip()) if isinstance(out, str) else out
            except Exception:
                return None
            if not isinstance(parsed, dict):
                return None

            s_id = safe_str(parsed.get("subject_id") or a_id)
            o_id = safe_str(parsed.get("object_id") or b_id)
            rel_type = safe_str(parsed.get("relation_type") or parsed.get("predicate")).strip()
            if not s_id or not o_id or not rel_type or _is_none_relation(rel_type):
                return None

            try:
                conf = float(parsed.get("confidence", 1.0))
            except Exception:
                conf = 1.0

            pa, pb = pack_by_ep.get(a_id, {}), pack_by_ep.get(b_id, {})
            src_docs = dedupe_list(
                [
                    safe_str(x)
                    for x in (safe_list(pa.get("source_documents")) + safe_list(pb.get("source_documents")))
                    if safe_str(x)
                ]
            )

            rid = stable_relation_id(s_id, rel_type, o_id, prefix="rel_ep_ep_")
            key = (a_id, b_id) if a_id < b_id else (b_id, a_id)

            props = safe_dict(parsed.get("properties"))
            props.setdefault("similarity", pair_sim.get(key))
            props.setdefault("common_neighbors", int(pair_common.get(key, 0)))
            props.setdefault("weighted_score", float(pair_weighted.get(key, 0.0)))
            props.setdefault("primary_shared_anchors", int(pair_primary.get(key, 0)))
            props.setdefault("context_shared_anchors", int(pair_context.get(key, 0)))
            props.setdefault("shared_anchor_breakdown", dict(pair_field_counts.get(key, {})))
            if include_shared_neighbor_ids:
                props.setdefault("shared_neighbor_ids", sorted(list(pair_shared.get(key, set()))))

            rel = {
                "id": rid,
                "subject_id": s_id,
                "object_id": o_id,
                "predicate": rel_type,
                "relation_name": rel_type,
                "confidence": conf,
                "description": safe_str(parsed.get("description")),
                "source_documents": src_docs,
                "properties": props,
            }
            return self._normalize_episode_relation_direction(
                rel,
                ep_by_id=ep_by_id,
                local_order_index=episode_local_order_index,
                enabled=direction_sanity_enabled,
                check_types=direction_check_types,
            )

        def _run_pair_with_retries(a_id: str, b_id: str) -> Optional[Dict[str, Any]]:
            desc = f"ep_pair {a_id}->{b_id}"
            last = None
            for i in range(max(1, int(per_pair_retries))):
                try:
                    return _run_pair_once(a_id, b_id)
                except Exception as e:
                    last = e
                    sleep = float(backoff_seconds) * (2**i)
                    if jitter_seconds > 0:
                        sleep += (hash(f"{desc}|{i}") % 1000) / 1000.0 * float(jitter_seconds)
                    logger.warning(
                        "[NarrativeGraph][retry] %s round=%d/%d err=%s",
                        desc,
                        i + 1,
                        per_pair_retries,
                        e,
                    )
                    if sleep > 0:
                        import time as _t

                        _t.sleep(sleep)
            if last:
                logger.warning("[NarrativeGraph][P2] pair failed after retries: %s err=%s", desc, last)
            return None

        workers = max(1, int(episode_pair_concurrency or self.max_workers))
        out_rels: List[Dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=workers) as ex:
            fut2pair = {ex.submit(_run_pair_with_retries, a, b): (a, b) for a, b in filtered}
            it = as_completed(fut2pair)
            if show_pair_progress:
                it = tqdm(it, total=len(fut2pair), desc="Phase2 pairs", unit="pair", leave=False)

            for fut in it:
                a_id, b_id = fut2pair[fut]
                try:
                    r = fut.result()
                    if not r:
                        continue
                    out_rels.append(r)
                    if save_pair_json:
                        json_dump_atomic(os.path.join(self.out_rel_pairs_dir, _pair_file(a_id, b_id)), r)
                except Exception as e:
                    logger.exception("[NarrativeGraph][P2] pair failed: %s", e)

        deduped = self._dedupe_relations(out_rels)
        json_dump_atomic(self.global_ep_ep_edges_path, deduped)

        logger.info(
            "[NarrativeGraph][P2] done: cand=%d filtered=%d ep2ep_edges=%d merged_out=%s cand_out=%s",
            len(cand_pairs),
            len(filtered),
            len(deduped),
            self.global_ep_ep_edges_path,
            cand_path,
        )
        return deduped

    # ================================================================
    # Cycle break
    # ================================================================
    def break_episode_cycles(
        self,
        *,
        method: str = "heuristic",  # heuristic | saber
        episodes_path: Optional[str] = None,
        episode_relations_path: Optional[str] = None,
        out_relations_path: Optional[str] = None,
        # overrides (override YAML if provided)
        tau_conf: Optional[float] = None,
        tau_eff: Optional[float] = None,
        delta_tie: Optional[float] = None,
        max_iter: Optional[int] = None,
        type_weight: Optional[Dict[str, float]] = None,
        skip_types: Optional[Set[str]] = None,
        flipped_types: Optional[Set[str]] = None,
        unified_pred: Optional[str] = None,
        max_llm_calls_per_iter: int = 200,
        triangle_workers: int = 64,
        show_progress: bool = True,
        save_log: bool = True,
        log_path: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        m = (method or "").strip().lower()
        if m not in {"heuristic", "saber"}:
            raise ValueError(f"Unknown cycle break method: {method}")

        ep_path = episodes_path or self.global_episodes_path
        rel_path = episode_relations_path or self.global_ep_ep_edges_path
        out_path = out_relations_path or self.dag_path

        episodes: List[Dict[str, Any]] = []
        relations: List[Dict[str, Any]] = []

        try:
            obj = load_json(ep_path)
            episodes = [x for x in obj if isinstance(x, dict)] if isinstance(obj, list) else []
        except Exception:
            logger.exception("[NarrativeGraph][CycleBreak] failed to load episodes: %s", ep_path)

        try:
            obj = load_json(rel_path)
            relations = [x for x in obj if isinstance(x, dict)] if isinstance(obj, list) else []
        except Exception:
            logger.exception("[NarrativeGraph][CycleBreak] failed to load relations: %s", rel_path)

        graph_cfg = load_causal_graph_cfg(self.config)
        cycle_cfg = load_cycle_break_cfg(self.config)

        # apply overrides
        if tau_conf is not None:
            graph_cfg["tau_conf"] = float(tau_conf)
        if tau_eff is not None:
            graph_cfg["tau_eff"] = float(tau_eff)
        if unified_pred is not None:
            graph_cfg["unified_pred"] = safe_str(unified_pred) or graph_cfg.get("unified_pred")
        if skip_types is not None:
            graph_cfg["skip_types"] = set([safe_str(x) for x in (skip_types or set()) if safe_str(x)])
        if flipped_types is not None:
            graph_cfg["flipped_types"] = set([safe_str(x) for x in (flipped_types or set()) if safe_str(x)])
        if type_weight is not None:
            graph_cfg["type_weight"] = {safe_str(k): float(v) for k, v in (type_weight or {}).items() if safe_str(k)}

        if delta_tie is not None:
            cycle_cfg["delta_tie"] = float(delta_tie)
        if max_iter is not None:
            cycle_cfg["max_iter"] = int(max_iter)

        if m == "heuristic":
            G, cb_log = run_heuristic(
                episodes,
                relations,
                config=self.config,
                graph_cfg=graph_cfg,
                cycle_cfg=cycle_cfg,
                show_progress=False,
            )
        else:
            G, cb_log = run_saber(
                episodes,
                relations,
                narrative_manager=self.narrative_manager,
                config=self.config,
                graph_cfg=graph_cfg,
                cycle_cfg=cycle_cfg,
                max_llm_calls_per_iter=int(max_llm_calls_per_iter),
                show_progress=bool(show_progress),
                triangle_workers=int(triangle_workers),
            )

        pred = safe_str(graph_cfg.get("unified_pred")) or "EPISODE_CAUSAL_LINK"
        new_rels = export_relations_from_graph(G, predicate=pred)
        json_dump_atomic(out_path, new_rels)

        if save_log:
            lp = log_path or os.path.join(self.out_global_dir, f"episode_cycle_break_{m}_log.json")
            json_dump_atomic(lp, cb_log)

        logger.info(
            "[NarrativeGraph][CycleBreak] method=%s in_edges=%d out_edges=%d out=%s",
            m,
            len(relations),
            len(new_rels),
            out_path,
        )
        return new_rels

    # ================================================================
    # Storyline
    # ================================================================
    @staticmethod
    def _is_contiguous_subchain(sub: List[str], full: List[str]) -> bool:
        n, m = len(sub), len(full)
        if n == 0 or n > m:
            return False
        for i in range(m - n + 1):
            if full[i : i + n] == sub:
                return True
        return False

    def _storyline_candidate_anchor_set(self, candidate: Dict[str, Any]) -> Set[str]:
        anchors: Set[str] = set()
        for item in safe_list(candidate.get("trunk_data")):
            if not isinstance(item, dict):
                continue
            props = safe_dict(item.get("properties"))
            for field in ("related_characters", "related_locations", "related_occasions", "related_timepoints"):
                for value in safe_list(props.get(field)):
                    s = safe_str(value).strip().lower()
                    if s:
                        anchors.add(f"{field}:{s}")
            for src in safe_list(item.get("source_documents")):
                s = safe_str(src).strip().lower()
                if s:
                    anchors.add(f"doc:{s}")
        return anchors

    def _storyline_candidate_score(self, candidate: Dict[str, Any]) -> Tuple[int, int, int]:
        trunk = [safe_str(x) for x in safe_list(candidate.get("trunk")) if safe_str(x)]
        src_docs: Set[str] = set()
        for item in safe_list(candidate.get("trunk_data")):
            if not isinstance(item, dict):
                continue
            for src in safe_list(item.get("source_documents")):
                s = safe_str(src).strip()
                if s:
                    src_docs.add(s)
        anchors = self._storyline_candidate_anchor_set(candidate)
        return (len(trunk), len(src_docs), len(anchors))

    @staticmethod
    def _storyline_candidate_episode_ids(candidate: Dict[str, Any]) -> Set[str]:
        return {safe_str(x) for x in safe_list(candidate.get("trunk")) if safe_str(x)}

    @staticmethod
    def _episode_primary_anchor_set(
        episode: Dict[str, Any],
        *,
        fields: Set[str],
    ) -> Set[str]:
        props = safe_dict(episode.get("properties"))
        anchors: Set[str] = set()
        for field in fields:
            for value in safe_list(props.get(field)):
                s = safe_str(value).strip().lower()
                if s:
                    anchors.add(f"{field}:{s}")
        return anchors

    def _prune_storyline_candidates(
        self,
        candidates: List[Dict[str, Any]],
        *,
        jaccard_threshold: float = 0.6,
        min_shared_anchor_count: int = 1,
    ) -> List[Dict[str, Any]]:
        ranked = sorted(
            [c for c in candidates if isinstance(c, dict)],
            key=lambda c: self._storyline_candidate_score(c),
            reverse=True,
        )

        kept: List[Dict[str, Any]] = []
        kept_trunks: List[List[str]] = []
        kept_anchors: List[Set[str]] = []

        for cand in ranked:
            trunk = [safe_str(x) for x in safe_list(cand.get("trunk")) if safe_str(x)]
            if not trunk:
                continue
            trunk_set = set(trunk)
            cand_anchors = self._storyline_candidate_anchor_set(cand)

            redundant = False
            for kept_trunk, kept_anchor in zip(kept_trunks, kept_anchors):
                kept_set = set(kept_trunk)
                shared_nodes = len(trunk_set.intersection(kept_set))
                union_nodes = len(trunk_set.union(kept_set))
                jacc = (shared_nodes / union_nodes) if union_nodes > 0 else 0.0
                shared_anchor_count = len(cand_anchors.intersection(kept_anchor))

                if self._is_contiguous_subchain(trunk, kept_trunk):
                    redundant = True
                    break

                if (
                    jacc >= float(jaccard_threshold)
                    and shared_anchor_count >= int(min_shared_anchor_count)
                ):
                    redundant = True
                    break

            if redundant:
                continue

            kept.append(cand)
            kept_trunks.append(trunk)
            kept_anchors.append(cand_anchors)

        return kept

    def _cluster_storyline_candidates(
        self,
        candidates: List[Dict[str, Any]],
        *,
        containment_threshold: float = 0.75,
        jaccard_threshold: float = 0.5,
        anchor_jaccard_threshold: float = 0.6,
        same_tail_min_shared_episodes: int = 2,
    ) -> List[Dict[str, Any]]:
        valid_candidates = [c for c in candidates if isinstance(c, dict)]
        n = len(valid_candidates)
        if n == 0:
            return []

        parent = list(range(n))

        def _find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def _union(a: int, b: int) -> None:
            ra, rb = _find(a), _find(b)
            if ra != rb:
                parent[rb] = ra

        episode_sets = [self._storyline_candidate_episode_ids(c) for c in valid_candidates]
        anchor_sets = [self._storyline_candidate_anchor_set(c) for c in valid_candidates]
        tails = [
            safe_str(safe_list(c.get("trunk"))[-1]) if safe_list(c.get("trunk")) else ""
            for c in valid_candidates
        ]

        for i in range(n):
            for j in range(i + 1, n):
                a_eps, b_eps = episode_sets[i], episode_sets[j]
                if not a_eps or not b_eps:
                    continue
                shared_eps = len(a_eps.intersection(b_eps))
                union_eps = len(a_eps.union(b_eps))
                containment = shared_eps / max(1, min(len(a_eps), len(b_eps)))
                jaccard = shared_eps / max(1, union_eps)

                a_anchor, b_anchor = anchor_sets[i], anchor_sets[j]
                anchor_union = len(a_anchor.union(b_anchor))
                anchor_jaccard = len(a_anchor.intersection(b_anchor)) / anchor_union if anchor_union else 0.0
                same_tail = tails[i] and tails[i] == tails[j] and shared_eps >= int(same_tail_min_shared_episodes)

                if (
                    containment >= float(containment_threshold)
                    and anchor_jaccard >= float(anchor_jaccard_threshold)
                ) or (
                    jaccard >= float(jaccard_threshold)
                    and anchor_jaccard >= float(anchor_jaccard_threshold)
                ) or same_tail:
                    _union(i, j)

        groups: Dict[int, List[int]] = defaultdict(list)
        for idx in range(n):
            groups[_find(idx)].append(idx)

        communities: List[Dict[str, Any]] = []
        for member_indices in groups.values():
            members = [valid_candidates[i] for i in member_indices]
            members = sorted(members, key=lambda c: self._storyline_candidate_score(c), reverse=True)
            support_episode_ids: List[str] = []
            seen_eps: Set[str] = set()
            for cand in members:
                for episode_id in safe_list(cand.get("trunk")):
                    eid = safe_str(episode_id)
                    if eid and eid not in seen_eps:
                        seen_eps.add(eid)
                        support_episode_ids.append(eid)

            key = "||".join("|".join([safe_str(x) for x in safe_list(c.get("trunk")) if safe_str(x)]) for c in members)
            community_id = "slc_" + hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
            representative = members[0] if members else {}
            communities.append(
                {
                    "id": community_id,
                    "representative_trunk": [safe_str(x) for x in safe_list(representative.get("trunk")) if safe_str(x)],
                    "support_episode_ids": support_episode_ids,
                    "community_size": len(members),
                    "chains": members,
                    "properties": {
                        "episode_count": len(support_episode_ids),
                        "anchor_count": len(self._storyline_candidate_anchor_set(representative)) if representative else 0,
                    },
                }
            )

        communities = sorted(
            communities,
            key=lambda c: (int(c.get("community_size", 0)), len(safe_list(c.get("support_episode_ids")))),
            reverse=True,
        )
        return communities

    @staticmethod
    def _filter_storyline_seed_relations(
        relations: List[Dict[str, Any]],
        *,
        allowed_types: Set[str],
    ) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        allowed = {safe_str(x).strip().lower() for x in allowed_types if safe_str(x)}
        for rel in relations or []:
            if not isinstance(rel, dict):
                continue
            rel_type = safe_str(rel.get("relation_type") or rel.get("predicate")).strip().lower()
            if not rel_type:
                continue
            if rel_type in allowed:
                out.append(rel)
        return out

    def _select_storyline_bridge_relations(
        self,
        *,
        episodes: List[Dict[str, Any]],
        relations: List[Dict[str, Any]],
        weak_types: Set[str],
        primary_anchor_fields: Set[str],
        min_shared_primary_anchor_count: int,
    ) -> List[Dict[str, Any]]:
        ep_by_id = {
            safe_str(ep.get("id")): ep
            for ep in episodes or []
            if isinstance(ep, dict) and safe_str(ep.get("id"))
        }
        anchor_map = {
            ep_id: self._episode_primary_anchor_set(ep, fields=primary_anchor_fields)
            for ep_id, ep in ep_by_id.items()
        }

        out: List[Dict[str, Any]] = []
        allowed = {safe_str(x).strip().lower() for x in weak_types if safe_str(x)}
        for rel in relations or []:
            if not isinstance(rel, dict):
                continue
            rel_type = safe_str(rel.get("relation_type") or rel.get("predicate")).strip().lower()
            if rel_type not in allowed:
                continue
            s_id = safe_str(rel.get("subject_id"))
            o_id = safe_str(rel.get("object_id"))
            if not s_id or not o_id:
                continue
            shared = len(anchor_map.get(s_id, set()).intersection(anchor_map.get(o_id, set())))
            if shared >= int(min_shared_primary_anchor_count):
                out.append(rel)
        return out

    def build_storyline_candidates(
        self,
        *,
        episodes_path: Optional[str] = None,
        episode_relations_dag_path: Optional[str] = None,
        method: Optional[str] = None,  # trie | mpc
        min_trunk_len: int = 2,
        out_candidates_path: Optional[str] = None,
        strong_relation_types: Optional[Set[str]] = None,
        weak_relation_types: Optional[Set[str]] = None,
        allow_weak_bridges: bool = True,
        weak_bridge_primary_anchor_fields: Optional[Set[str]] = None,
        weak_bridge_min_shared_primary_anchor_count: int = 1,
        allow_weak_fallback: bool = True,
        candidate_jaccard_prune_threshold: float = 0.6,
        candidate_min_shared_anchor_count: int = 1,
    ) -> List[Dict[str, Any]]:
        ep_path = episodes_path or self.global_episodes_path
        dag_path = episode_relations_dag_path or self.dag_path
        out_path = out_candidates_path or self.global_storyline_candidates_path

        try:
            obj = load_json(ep_path)
            episodes = [x for x in obj if isinstance(x, dict)] if isinstance(obj, list) else []
        except Exception:
            logger.exception("[NarrativeGraph][Storyline] failed to load episodes: %s", ep_path)
            episodes = []

        try:
            obj = load_json(dag_path)
            dag_rels = [x for x in obj if isinstance(x, dict)] if isinstance(obj, list) else []
        except Exception:
            logger.exception("[NarrativeGraph][Storyline] failed to load dag relations: %s", dag_path)
            dag_rels = []

        if not episodes or not dag_rels:
            json_dump_atomic(out_path, [])
            logger.warning(
                "[NarrativeGraph][Storyline] empty candidates due to missing episodes/dag: episodes=%d dag=%d",
                len(episodes),
                len(dag_rels),
            )
            return []

        m = safe_str(method or self.storyline_method_default or "trie").lower()
        if m == "tri":
            m = "trie"

        strong_types = {safe_str(x).strip().lower() for x in (strong_relation_types or {"causes", "elaborates"}) if safe_str(x)}
        weak_types = {safe_str(x).strip().lower() for x in (weak_relation_types or {"precedes"}) if safe_str(x)}
        primary_anchor_fields = {
            safe_str(x).strip()
            for x in (weak_bridge_primary_anchor_fields or {"related_characters", "related_occasions"})
            if safe_str(x)
        }

        selected_rels = self._filter_storyline_seed_relations(dag_rels, allowed_types=strong_types)
        seed_mode = "strong_only"
        if selected_rels and allow_weak_bridges:
            bridge_rels = self._select_storyline_bridge_relations(
                episodes=episodes,
                relations=dag_rels,
                weak_types=weak_types,
                primary_anchor_fields=primary_anchor_fields,
                min_shared_primary_anchor_count=int(weak_bridge_min_shared_primary_anchor_count),
            )
            if bridge_rels:
                merged: List[Dict[str, Any]] = []
                seen_rel_keys: Set[Tuple[str, str, str]] = set()
                for rel in selected_rels + bridge_rels:
                    key = (
                        safe_str(rel.get("subject_id")),
                        safe_str(rel.get("relation_type") or rel.get("predicate")).strip().lower(),
                        safe_str(rel.get("object_id")),
                    )
                    if key in seen_rel_keys:
                        continue
                    seen_rel_keys.add(key)
                    merged.append(rel)
                selected_rels = merged
                seed_mode = "strong_plus_bridge"
        if not selected_rels:
            selected_rels = dag_rels
            seed_mode = "all_relations_no_type_match"

        candidates = extract_storyline_candidates(
            episodes=episodes,
            relations=selected_rels,
            config=self.config,
            method=m,
        )

        if not candidates and allow_weak_fallback:
            relaxed_types = strong_types.union(weak_types)
            fallback_rels = self._filter_storyline_seed_relations(dag_rels, allowed_types=relaxed_types)
            if fallback_rels:
                candidates = extract_storyline_candidates(
                    episodes=episodes,
                    relations=fallback_rels,
                    config=self.config,
                    method=m,
                )
                selected_rels = fallback_rels
                seed_mode = "strong_plus_weak_fallback"

        out: List[Dict[str, Any]] = []
        seen: Set[Tuple[str, ...]] = set()
        for c in candidates:
            if not isinstance(c, dict):
                continue
            trunk = [safe_str(x) for x in safe_list(c.get("trunk")) if safe_str(x)]
            if len(trunk) < max(1, int(min_trunk_len)):
                continue
            key = tuple(trunk)
            if key in seen:
                continue
            seen.add(key)
            cc = dict(c)
            cc["trunk"] = trunk
            cc["method"] = m
            out.append(cc)

        pruned = self._prune_storyline_candidates(
            out,
            jaccard_threshold=float(candidate_jaccard_prune_threshold),
            min_shared_anchor_count=int(candidate_min_shared_anchor_count),
        )

        json_dump_atomic(out_path, pruned)
        logger.info(
            "[NarrativeGraph][Storyline] candidates built: method=%s seed_mode=%s rels_in=%d raw=%d dedup=%d pruned=%d saved=%s",
            m,
            seed_mode,
            len(selected_rels),
            len(candidates),
            len(out),
            len(pruned),
            out_path,
        )
        return pruned

    def extract_storylines_from_candidates(
        self,
        *,
        candidates_path: Optional[str] = None,
        out_storylines_path: Optional[str] = None,
        out_support_edges_path: Optional[str] = None,
        out_chain_communities_path: Optional[str] = None,
        community_mode: bool = False,
        community_containment_threshold: float = 0.75,
        community_jaccard_threshold: float = 0.5,
        community_anchor_jaccard_threshold: float = 0.6,
        community_same_tail_min_shared_episodes: int = 2,
        max_community_variant_chains: int = 6,
        ensure_storyline_embeddings: bool = True,
        embedding_text_field: str = "name_desc",
        embedding_batch_size: int = 256,
        storyline_extraction_concurrency: Optional[int] = None,
        show_storyline_progress: bool = True,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        cand_path = candidates_path or self.global_storyline_candidates_path
        sl_path = out_storylines_path or self.global_storylines_path
        se_path = out_support_edges_path or self.global_storyline_support_edges_path
        comm_path = out_chain_communities_path or self.global_storyline_chain_communities_path

        try:
            obj = load_json(cand_path)
            candidates = [x for x in obj if isinstance(x, dict)] if isinstance(obj, list) else []
        except Exception:
            logger.exception("[NarrativeGraph][Storyline] failed to load candidates: %s", cand_path)
            candidates = []

        if not candidates:
            json_dump_atomic(sl_path, [])
            json_dump_atomic(se_path, [])
            if community_mode:
                json_dump_atomic(comm_path, [])
            logger.warning("[NarrativeGraph][Storyline] no candidates found: %s", cand_path)
            return [], []

        def _chain_information(c: Dict[str, Any]) -> str:
            chunks: List[str] = []
            for x in safe_list(c.get("trunk_data")):
                if not isinstance(x, dict):
                    continue
                name = safe_str(x.get("name")).strip()
                desc = safe_str(x.get("description")).strip()
                if name and desc:
                    chunks.append(f"{name}: {desc}")
                elif name or desc:
                    chunks.append(name or desc)
            if chunks:
                return "\n".join(chunks)
            trunk_ids = [safe_str(x) for x in safe_list(c.get("trunk")) if safe_str(x)]
            return "\n".join(trunk_ids)

        def _chain_title(c: Dict[str, Any]) -> str:
            names: List[str] = []
            for x in safe_list(c.get("trunk_data")):
                if isinstance(x, dict):
                    name = safe_str(x.get("name")).strip()
                    if name:
                        names.append(name)
            return " -> ".join(names) if names else " -> ".join([safe_str(x) for x in safe_list(c.get("trunk")) if safe_str(x)])

        def _community_chain_information(comm: Dict[str, Any]) -> str:
            chains = [x for x in safe_list(comm.get("chains")) if isinstance(x, dict)]
            representative = chains[0] if chains else {}

            episode_by_id: Dict[str, Dict[str, Any]] = {}
            for chain in chains:
                for item in safe_list(chain.get("trunk_data")):
                    if not isinstance(item, dict):
                        continue
                    eid = safe_str(item.get("id"))
                    if eid and eid not in episode_by_id:
                        episode_by_id[eid] = item

            support_lines: List[str] = []
            for eid in safe_list(comm.get("support_episode_ids")):
                item = episode_by_id.get(safe_str(eid), {})
                name = safe_str(item.get("name")).strip()
                desc = safe_str(item.get("description")).strip()
                if name and desc:
                    support_lines.append(f"- {name}: {desc}")
                elif name or desc:
                    support_lines.append(f"- {name or desc}")

            variant_lines: List[str] = []
            for chain in chains[: max(1, int(max_community_variant_chains))]:
                title = _chain_title(chain)
                if title:
                    variant_lines.append(f"- {title}")

            rep_info = _chain_information(representative) if representative else ""
            return "\n".join(
                [
                    "CHAIN COMMUNITY MODE:",
                    "The following episode chains are highly overlapping variants of the same candidate narrative thread.",
                    "Synthesize exactly ONE broader, distinctive Storyline that captures the shared central progression.",
                    "Do not produce a thin summary of only one variant, and do not repeat generic impacts such as vague tension unless a concrete downstream condition is stated.",
                    "",
                    "Representative chain:",
                    rep_info,
                    "",
                    "Merged support episodes:",
                    "\n".join(support_lines),
                    "",
                    "Variant chain titles:",
                    "\n".join(variant_lines),
                ]
            ).strip()

        def _unique_str_list(xs: List[Any]) -> List[str]:
            out: List[str] = []
            seen: Set[str] = set()
            for x in xs:
                s = safe_str(x).strip()
                if not s or s in seen:
                    continue
                seen.add(s)
                out.append(s)
            return out

        storylines: List[Dict[str, Any]] = []
        support_edges: List[Dict[str, Any]] = []

        storyline_inputs: List[Dict[str, Any]]
        if community_mode:
            communities = self._cluster_storyline_candidates(
                candidates,
                containment_threshold=float(community_containment_threshold),
                jaccard_threshold=float(community_jaccard_threshold),
                anchor_jaccard_threshold=float(community_anchor_jaccard_threshold),
                same_tail_min_shared_episodes=int(community_same_tail_min_shared_episodes),
            )
            json_dump_atomic(comm_path, communities)
            storyline_inputs = communities
            logger.info(
                "[NarrativeGraph][Storyline] chain communities: candidates=%d communities=%d saved=%s",
                len(candidates),
                len(communities),
                comm_path,
            )
        else:
            storyline_inputs = candidates

        def _extract_one_storyline(c: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
            if community_mode:
                trunk = [safe_str(x) for x in safe_list(c.get("support_episode_ids")) if safe_str(x)]
                info = _community_chain_information(c)
                source_chains = [x for x in safe_list(c.get("chains")) if isinstance(x, dict)]
            else:
                trunk = [safe_str(x) for x in safe_list(c.get("trunk")) if safe_str(x)]
                info = _chain_information(c)
                source_chains = [c]
            if not trunk:
                return [], []

            if not safe_str(info):
                return [], []

            try:
                raw = self.narrative_manager.extract_storyline(chain_information=info)
                parsed = json.loads(raw.strip()) if isinstance(raw, str) else raw
            except Exception:
                logger.exception("[NarrativeGraph][Storyline] storyline LLM/parse failed")
                parsed = None
            if not isinstance(parsed, dict):
                return [], []
            if safe_str(parsed.get("error")).strip():
                return [], []

            name = safe_str(parsed.get("name")).strip()
            desc = safe_str(parsed.get("description")).strip()
            if not name or not desc:
                return [], []

            method = safe_str(c.get("method")).strip().lower() or "trie"
            if community_mode:
                sid_key = f"community||{safe_str(c.get('id'))}||{'|'.join(trunk)}"
            else:
                sid_key = f"{method}||{'|'.join(trunk)}"
            sid = "sl_" + hashlib.sha1(sid_key.encode("utf-8")).hexdigest()[:16]

            all_source_docs: List[str] = []
            for chain in source_chains:
                for x in safe_list(chain.get("trunk_data")):
                    if isinstance(x, dict):
                        all_source_docs.extend(safe_list(x.get("source_documents")))
            source_docs = _unique_str_list(all_source_docs)

            props = {
                "impact": safe_str(parsed.get("impact")).strip(),
                "key_characters": _unique_str_list(safe_list(parsed.get("key_characters"))),
                "key_locations": _unique_str_list(safe_list(parsed.get("key_locations"))),
                "method": method,
                "chain_information": info,
                "trunk_size": len(trunk),
            }
            if community_mode:
                props["community_id"] = safe_str(c.get("id"))
                props["community_size"] = int(c.get("community_size", 0) or 0)
                props["support_episode_count"] = len(trunk)

            storyline_ent = {
                "id": sid,
                "name": name,
                "type": ["Storyline"],
                "aliases": [],
                "description": desc,
                "scope": "global",
                "source_documents": source_docs,
                "properties": props,
            }

            local_support_edges: List[Dict[str, Any]] = []
            for eid in trunk:
                rid = stable_relation_id(sid, P_STORYLINE_CONTAINS, eid, prefix="rel_sl_ep_")
                local_support_edges.append(
                    {
                        "id": rid,
                        "subject_id": sid,
                        "object_id": eid,
                        "predicate": P_STORYLINE_CONTAINS,
                        "relation_name": "contains",
                        "confidence": 1.0,
                        "description": "Storyline contains Episode chain member.",
                        "source_documents": source_docs,
                        "properties": {"method": method},
                    }
                )
            return [storyline_ent], local_support_edges

        workers = int(storyline_extraction_concurrency or self.max_workers or 1)
        workers = max(1, min(workers, len(storyline_inputs)))
        if workers <= 1 or len(storyline_inputs) <= 1:
            iterator = storyline_inputs
            if show_storyline_progress:
                iterator = tqdm(iterator, desc="Extract storylines", ncols=100)
            for c in iterator:
                local_storylines, local_support_edges = _extract_one_storyline(c)
                storylines.extend(local_storylines)
                support_edges.extend(local_support_edges)
        else:
            logger.info(
                "[NarrativeGraph][Storyline] extracting with concurrency=%d inputs=%d",
                workers,
                len(storyline_inputs),
            )
            results_by_idx: Dict[int, Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]] = {}
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="storyline_extract") as executor:
                future_to_idx = {
                    executor.submit(_extract_one_storyline, c): idx
                    for idx, c in enumerate(storyline_inputs)
                }
                iterator = as_completed(future_to_idx)
                if show_storyline_progress:
                    iterator = tqdm(iterator, total=len(future_to_idx), desc="Extract storylines", ncols=100)
                for future in iterator:
                    idx = future_to_idx[future]
                    try:
                        results_by_idx[idx] = future.result()
                    except Exception:
                        logger.exception("[NarrativeGraph][Storyline] worker failed")
                        continue
            for idx in range(len(storyline_inputs)):
                local_storylines, local_support_edges = results_by_idx.get(idx, ([], []))
                storylines.extend(local_storylines or [])
                support_edges.extend(local_support_edges or [])

        storylines = self._dedupe_entities_by_id(storylines)
        support_edges = self._dedupe_relations(support_edges)

        if ensure_storyline_embeddings and storylines:
            try:
                storylines = self._ensure_storyline_embeddings(
                    storylines=storylines,
                    embedding_text_field=embedding_text_field,
                    batch_size=embedding_batch_size,
                    save_path=sl_path,
                )
            except Exception as e:
                logger.exception("[NarrativeGraph][Storyline] ensure embeddings failed: %s", e)

        json_dump_atomic(sl_path, storylines)
        json_dump_atomic(se_path, support_edges)
        logger.info(
            "[NarrativeGraph][Storyline] extracted: storylines=%d support_edges=%d saved=%s,%s",
            len(storylines),
            len(support_edges),
            sl_path,
            se_path,
        )
        return storylines, support_edges

    def extract_storyline_relations(
        self,
        *,
        storylines_path: Optional[str] = None,
        out_relations_path: Optional[str] = None,
        max_storyline_pairs_global: int = 50000,
        similarity_threshold: float = 0.5,
        overlap_pair_only: bool = True,
        min_shared_anchor_count: int = 1,
        show_pair_progress: bool = False,
        storyline_pair_concurrency: Optional[int] = None,
        topk_neighbors_per_storyline: Optional[int] = None,
        per_pair_retries: int = 3,
        backoff_seconds: float = 0.8,
        jitter_seconds: float = 0.2,
        progress_log_every: int = 250,
    ) -> List[Dict[str, Any]]:
        sl_path = storylines_path or self.global_storylines_path
        out_path = out_relations_path or self.global_storyline_relations_path

        try:
            obj = load_json(sl_path)
            storylines = [x for x in obj if isinstance(x, dict)] if isinstance(obj, list) else []
        except Exception:
            logger.exception("[NarrativeGraph][Storyline] failed to load storylines: %s", sl_path)
            storylines = []

        if len(storylines) < 2:
            json_dump_atomic(out_path, [])
            logger.warning("[NarrativeGraph][Storyline] insufficient storylines: %d", len(storylines))
            return []

        try:
            storylines = self._ensure_storyline_embeddings(
                storylines=storylines,
                embedding_text_field="name_desc",
                batch_size=256,
                save_path=sl_path,
            )
        except Exception as e:
            logger.warning("[NarrativeGraph][Storyline] embedding prep failed: %s", e)

        def _anchors(s: Dict[str, Any]) -> Set[str]:
            props = safe_dict(s.get("properties"))
            chars = [safe_str(x).lower() for x in safe_list(props.get("key_characters")) if safe_str(x)]
            locs = [safe_str(x).lower() for x in safe_list(props.get("key_locations")) if safe_str(x)]
            return set(chars + locs)

        min_shared_anchor_count = max(0, int(min_shared_anchor_count or 0))

        def _collect_pair_candidates(
            *,
            current_min_shared_anchor_count: int,
            current_similarity_threshold: float,
        ) -> Tuple[List[Tuple[str, str, float, int]], Dict[str, int]]:
            total_pairs_considered = 0
            reject_missing_id = 0
            reject_low_overlap = 0
            reject_missing_embedding = 0
            reject_low_similarity = 0
            pair_cands: List[Tuple[str, str, float, int]] = []
            per_storyline_cands: Dict[str, List[Tuple[str, str, float, int]]] = defaultdict(list)
            for i in range(len(storylines)):
                a = storylines[i]
                a_id = safe_str(a.get("id"))
                a_emb = a.get("embedding")
                a_anchor = _anchors(a)
                if not a_id:
                    continue
                for j in range(i + 1, len(storylines)):
                    total_pairs_considered += 1
                    b = storylines[j]
                    b_id = safe_str(b.get("id"))
                    b_emb = b.get("embedding")
                    b_anchor = _anchors(b)
                    if not b_id:
                        reject_missing_id += 1
                        continue
                    shared = len(a_anchor.intersection(b_anchor))
                    if overlap_pair_only and shared < current_min_shared_anchor_count:
                        reject_low_overlap += 1
                        continue
                    sim = cosine_sim(a_emb, b_emb) if isinstance(a_emb, list) and isinstance(b_emb, list) else None
                    if sim is None:
                        reject_missing_embedding += 1
                        continue
                    if sim < float(current_similarity_threshold):
                        reject_low_similarity += 1
                        continue
                    item = (a_id, b_id, float(sim), int(shared))
                    pair_cands.append(item)
                    per_storyline_cands[a_id].append(item)
                    per_storyline_cands[b_id].append(item)
            if topk_neighbors_per_storyline is not None and int(topk_neighbors_per_storyline) > 0:
                selected: Dict[Tuple[str, str], Tuple[str, str, float, int]] = {}
                topk = int(topk_neighbors_per_storyline)
                for items in per_storyline_cands.values():
                    ranked_items = sorted(items, key=lambda x: (x[3], x[2]), reverse=True)[:topk]
                    for item in ranked_items:
                        key = tuple(sorted((item[0], item[1])))
                        selected[key] = item
                pair_cands = list(selected.values())
            pair_cands = sorted(pair_cands, key=lambda x: (x[3], x[2]), reverse=True)[: int(max_storyline_pairs_global)]
            stats = {
                "total_pairs_considered": total_pairs_considered,
                "reject_missing_id": reject_missing_id,
                "reject_low_overlap": reject_low_overlap,
                "reject_missing_embedding": reject_missing_embedding,
                "reject_low_similarity": reject_low_similarity,
                "topk_neighbors_per_storyline": int(topk_neighbors_per_storyline or 0),
            }
            return pair_cands, stats

        active_min_shared_anchor_count = int(min_shared_anchor_count)
        active_similarity_threshold = float(similarity_threshold)
        pair_cands, pair_stats = _collect_pair_candidates(
            current_min_shared_anchor_count=active_min_shared_anchor_count,
            current_similarity_threshold=active_similarity_threshold,
        )
        if not pair_cands and overlap_pair_only and len(storylines) > 2:
            while not pair_cands and active_min_shared_anchor_count > 0:
                active_min_shared_anchor_count -= 1
                active_similarity_threshold = max(0.0, active_similarity_threshold - 0.05)
                pair_cands, pair_stats = _collect_pair_candidates(
                    current_min_shared_anchor_count=active_min_shared_anchor_count,
                    current_similarity_threshold=active_similarity_threshold,
                )
                logger.info(
                    "[NarrativeGraph][Storyline] fallback relax: min_shared=%d thr=%.3f pairs=%d",
                    active_min_shared_anchor_count,
                    active_similarity_threshold,
                    len(pair_cands),
                )

        sl_by_id = {safe_str(s.get("id")): s for s in storylines if safe_str(s.get("id"))}

        logger.info(
            "[NarrativeGraph][Storyline] relation candidates: pairs=%d overlap_only=%s min_shared=%d thr=%.3f topk=%d "
            "(total_pairs=%d, missing_id=%d, low_overlap=%d, missing_emb=%d, low_sim=%d)",
            len(pair_cands),
            "yes" if overlap_pair_only else "no",
            active_min_shared_anchor_count,
            active_similarity_threshold,
            pair_stats.get("topk_neighbors_per_storyline", 0),
            pair_stats["total_pairs_considered"],
            pair_stats["reject_missing_id"],
            pair_stats["reject_low_overlap"],
            pair_stats["reject_missing_embedding"],
            pair_stats["reject_low_similarity"],
        )

        llm_parse_fail = 0
        llm_none = 0
        llm_self_loop = 0
        llm_kept = 0

        def _sl_info(s: Dict[str, Any]) -> Dict[str, Any]:
            props = safe_dict(s.get("properties"))
            compact_props = {
                "scope": safe_str(s.get("scope")),
                "key_characters": safe_list(props.get("key_characters")),
                "key_locations": safe_list(props.get("key_locations")),
                "key_events": safe_list(props.get("key_events")),
                "impact": safe_str(props.get("impact")),
            }
            compact_props = {k: v for k, v in compact_props.items() if v or (isinstance(v, str) and v)}
            return {
                "id": safe_str(s.get("id")),
                "name": safe_str(s.get("name")),
                "description": safe_str(s.get("description")),
                "properties": compact_props,
            }

        def _run_pair_once(a_id: str, b_id: str, sim: float, shared: int) -> Tuple[str, Optional[Dict[str, Any]]]:
            sa = sl_by_id.get(a_id)
            sb = sl_by_id.get(b_id)
            if not sa or not sb:
                return "missing_entity", None

            out = self.narrative_manager.extract_narrative_relation(
                subject_entity_info=_sl_info(sa),
                object_entity_info=_sl_info(sb),
                entity_type="Storyline",
            )
            try:
                parsed = json.loads(out.strip()) if isinstance(out, str) else out
            except Exception:
                parsed = None
            if not isinstance(parsed, dict):
                return "parse_fail", None

            rel_type = safe_str(parsed.get("relation_type") or parsed.get("predicate")).strip()
            s_id = safe_str(parsed.get("subject_id"))
            o_id = safe_str(parsed.get("object_id"))
            if not rel_type or _is_none_relation(rel_type) or not s_id or not o_id:
                return "none", None
            if s_id == o_id:
                return "self_loop", None

            try:
                conf = float(parsed.get("confidence", 1.0))
            except Exception:
                conf = 1.0

            source_docs = dedupe_list(
                [safe_str(x) for x in (safe_list(sa.get("source_documents")) + safe_list(sb.get("source_documents"))) if safe_str(x)]
            )
            rid = stable_relation_id(s_id, rel_type, o_id, prefix="rel_sl_sl_")
            props = {
                "similarity": float(sim),
                "shared_anchor_count": int(shared),
            }
            return (
                "kept",
                {
                    "id": rid,
                    "subject_id": s_id,
                    "object_id": o_id,
                    "predicate": rel_type,
                    "relation_name": rel_type,
                    "confidence": conf,
                    "description": safe_str(parsed.get("description")),
                    "source_documents": source_docs,
                    "properties": props,
                },
            )

        def _run_pair_with_retries(a_id: str, b_id: str, sim: float, shared: int) -> Tuple[str, Optional[Dict[str, Any]]]:
            desc = f"storyline_pair {a_id}->{b_id}"
            last = None
            for i in range(max(1, int(per_pair_retries))):
                try:
                    return _run_pair_once(a_id, b_id, sim, shared)
                except Exception as e:
                    last = e
                    sleep = float(backoff_seconds) * (2**i)
                    if jitter_seconds > 0:
                        sleep += (hash(f"{desc}|{i}") % 1000) / 1000.0 * float(jitter_seconds)
                    logger.warning(
                        "[NarrativeGraph][retry] %s round=%d/%d err=%s",
                        desc,
                        i + 1,
                        per_pair_retries,
                        e,
                    )
                    if sleep > 0:
                        import time as _t

                        _t.sleep(sleep)
            if last:
                logger.warning("[NarrativeGraph][Storyline] pair failed after retries: %s err=%s", desc, last)
            return "exception", None

        out_rels: List[Dict[str, Any]] = []
        workers = max(1, int(storyline_pair_concurrency or self.max_workers))
        workers = min(workers, len(pair_cands)) if pair_cands else 1
        with ThreadPoolExecutor(max_workers=workers) as ex:
            fut2pair = {
                ex.submit(_run_pair_with_retries, a_id, b_id, sim, shared): (a_id, b_id)
                for a_id, b_id, sim, shared in pair_cands
            }
            iterable = as_completed(fut2pair)
            if show_pair_progress:
                iterable = tqdm(iterable, total=len(fut2pair), desc="Storyline pairs", unit="pair", leave=False)

            for idx, fut in enumerate(iterable, start=1):
                try:
                    status, rel_payload = fut.result()
                except Exception as e:
                    logger.exception("[NarrativeGraph][Storyline] pair failed: %s", e)
                    status, rel_payload = "exception", None

                if status == "parse_fail":
                    llm_parse_fail += 1
                elif status == "none":
                    llm_none += 1
                elif status == "self_loop":
                    llm_self_loop += 1
                elif status == "kept" and rel_payload:
                    out_rels.append(rel_payload)
                    llm_kept += 1

                if progress_log_every and idx % int(progress_log_every) == 0:
                    logger.info(
                        "[NarrativeGraph][Storyline] progress: done=%d/%d kept=%d none=%d parse_fail=%d self_loop=%d",
                        idx,
                        len(pair_cands),
                        llm_kept,
                        llm_none,
                        llm_parse_fail,
                        llm_self_loop,
                    )

        out_rels = self._dedupe_relations(out_rels)
        json_dump_atomic(out_path, out_rels)

        logger.info(
            "[NarrativeGraph][Storyline] relations extracted: pairs=%d rels=%d saved=%s "
            "(total_pairs=%d, missing_id=%d, low_overlap=%d, missing_emb=%d, low_sim=%d, llm_kept=%d, llm_none=%d, llm_parse_fail=%d, llm_self_loop=%d)",
            len(pair_cands),
            len(out_rels),
            out_path,
            pair_stats["total_pairs_considered"],
            pair_stats["reject_missing_id"],
            pair_stats["reject_low_overlap"],
            pair_stats["reject_missing_embedding"],
            pair_stats["reject_low_similarity"],
            llm_kept,
            llm_none,
            llm_parse_fail,
            llm_self_loop,
        )
        return out_rels

    # ================================================================
    # JSON -> runtime graph
    # ================================================================
    def load_json_to_graph_store(
        self,
        *,
        episodes_path: Optional[str] = None,
        support_edges_path: Optional[str] = None,
        episode_relations_path: Optional[str] = None,
        episode_relations_dag_path: Optional[str] = None,
        storylines_path: Optional[str] = None,
        storyline_support_edges_path: Optional[str] = None,
        storyline_relations_path: Optional[str] = None,
        store_support_edges: bool = True,
        store_episode_relations: bool = True,
        store_episode_dag_relations: bool = True,
        store_episodes: bool = True,
        store_storylines: bool = True,
        store_storyline_support_edges: bool = True,
        store_storyline_relations: bool = True,
    ) -> None:
        ep_path = episodes_path or self.global_episodes_path
        se_path = support_edges_path or self.global_ep_support_edges_path
        fine_path = episode_relations_path or self.global_ep_ep_edges_path
        dag_path = episode_relations_dag_path or self.dag_path
        sl_path = storylines_path or self.global_storylines_path
        sl_se_path = storyline_support_edges_path or self.global_storyline_support_edges_path
        sl_rel_path = storyline_relations_path or self.global_storyline_relations_path

        episodes: List[Dict[str, Any]] = []
        support_edges: List[Dict[str, Any]] = []
        fine_all: List[Dict[str, Any]] = []
        dag_raw: List[Dict[str, Any]] = []
        storylines: List[Dict[str, Any]] = []
        storyline_support_edges: List[Dict[str, Any]] = []
        storyline_relations: List[Dict[str, Any]] = []

        if store_episodes:
            try:
                obj = load_json(ep_path)
                episodes = [x for x in obj if isinstance(x, dict)] if isinstance(obj, list) else []
            except Exception:
                logger.exception("[NarrativeGraph][JSON->Graph] failed to load episodes: %s", ep_path)

        if store_support_edges:
            try:
                obj = load_json(se_path)
                support_edges = [x for x in obj if isinstance(x, dict)] if isinstance(obj, list) else []
            except Exception:
                logger.exception("[NarrativeGraph][JSON->Graph] failed to load support_edges: %s", se_path)

        if store_episode_relations:
            try:
                obj = load_json(fine_path)
                fine_all = [x for x in obj if isinstance(x, dict)] if isinstance(obj, list) else []
            except Exception:
                logger.exception("[NarrativeGraph][JSON->Graph] failed to load fine relations: %s", fine_path)

        if store_episode_dag_relations:
            if not os.path.exists(dag_path):
                raise FileNotFoundError(f"DAG relations not found: {dag_path}. Runtime graph writing is based on _dag.json.")
            try:
                obj = load_json(dag_path)
                dag_raw = [x for x in obj if isinstance(x, dict)] if isinstance(obj, list) else []
            except Exception:
                logger.exception("[NarrativeGraph][JSON->Graph] failed to load dag relations: %s", dag_path)
                dag_raw = []

        if store_storylines:
            try:
                obj = load_json(sl_path)
                storylines = [x for x in obj if isinstance(x, dict)] if isinstance(obj, list) else []
            except Exception:
                logger.exception("[NarrativeGraph][JSON->Graph] failed to load storylines: %s", sl_path)

        if store_storyline_support_edges:
            try:
                obj = load_json(sl_se_path)
                storyline_support_edges = [x for x in obj if isinstance(x, dict)] if isinstance(obj, list) else []
            except Exception:
                logger.exception("[NarrativeGraph][JSON->Graph] failed to load storyline support edges: %s", sl_se_path)

        if store_storyline_relations:
            try:
                obj = load_json(sl_rel_path)
                storyline_relations = [x for x in obj if isinstance(x, dict)] if isinstance(obj, list) else []
            except Exception:
                logger.exception("[NarrativeGraph][JSON->Graph] failed to load storyline relations: %s", sl_rel_path)

        allowed_fine: Set[str] = set(
            safe_str(x)
            for r in dag_raw
            for x in safe_list(r.get("source_relation_ids"))
            if safe_str(x)
        )

        fine_rels: List[Dict[str, Any]] = []
        if store_episode_relations:
            fine_rels = [r for r in fine_all if safe_str(r.get("id")) in allowed_fine] if allowed_fine else []
            fine_rels = self._dedupe_relations(fine_rels)

        dag_norm: List[Dict[str, Any]] = []
        if store_episode_dag_relations:
            dag_norm = self._normalize_episode_dag_relations_for_graph_store(dag_raw)

        logger.info(
            "[NarrativeGraph][JSON->Graph] store_episodes=%s episodes=%d store_support=%s support=%d "
            "store_fine=%s fine=%d (allowed=%d) store_dag=%s dag=%d "
            "store_storylines=%s storylines=%d store_sl_support=%s sl_support=%d store_sl_rel=%s sl_rel=%d",
            "yes" if store_episodes else "no",
            len(episodes),
            "yes" if store_support_edges else "no",
            len(support_edges),
            "yes" if store_episode_relations else "no",
            len(fine_rels),
            len(allowed_fine),
            "yes" if store_episode_dag_relations else "no",
            len(dag_norm),
            "yes" if store_storylines else "no",
            len(storylines),
            "yes" if store_storyline_support_edges else "no",
            len(storyline_support_edges),
            "yes" if store_storyline_relations else "no",
            len(storyline_relations),
        )

        if store_episodes and episodes:
            self.graph_query_utils.save_to_graph_store(episodes, [])
        if store_support_edges and support_edges:
            self.graph_query_utils.save_to_graph_store([], support_edges)
        if store_episode_relations and fine_rels:
            self.graph_query_utils.save_to_graph_store([], fine_rels)
        if store_episode_dag_relations and dag_norm:
            self.graph_query_utils.save_to_graph_store([], dag_norm)
        if store_storylines and storylines:
            self.graph_query_utils.save_to_graph_store(storylines, [])
        if store_storyline_support_edges and storyline_support_edges:
            self.graph_query_utils.save_to_graph_store([], storyline_support_edges)
        if store_storyline_relations and storyline_relations:
            self.graph_query_utils.save_to_graph_store([], storyline_relations)

        if store_episodes and episodes:
            self._persist_entity_embeddings_to_graph(
                entities=episodes,
                allowed_labels=["Episode"],
            )
        if store_storylines and storylines:
            self._persist_entity_embeddings_to_graph(
                entities=storylines,
                allowed_labels=["Storyline"],
            )

    # ================================================================
    # Embeddings
    # ================================================================
    def _persist_entity_embeddings_to_graph(
        self,
        *,
        entities: List[Dict[str, Any]],
        allowed_labels: List[str],
    ) -> int:
        if not entities:
            return 0

        def _as_1d_vec(v: Any) -> Optional[List[float]]:
            if v is None:
                return None
            # Preferred: already a single vector
            if isinstance(v, list) and v and isinstance(v[0], (int, float)):
                return [float(x) for x in v]
            # Fallback: accidentally wrapped batch form [[...]]
            vv = _to_vec_list(v)
            if vv and isinstance(vv[0], list) and vv[0]:
                return [float(x) for x in vv[0]]
            return None

        labels = [safe_str(x) for x in (allowed_labels or []) if safe_str(x)]
        if not labels:
            return 0

        graph = self.graph_store.get_graph()
        updated = 0
        for e in entities:
            if not isinstance(e, dict):
                continue
            eid = safe_str(e.get("id"))
            if not eid or not graph.has_node(eid):
                continue
            node_labels = [safe_str(x) for x in safe_list(graph.nodes[eid].get("type")) if safe_str(x)]
            if not (set(labels) & set(node_labels)):
                continue
            emb = e.get("embedding")
            vec = _as_1d_vec(emb)
            if not vec:
                continue
            graph.nodes[eid]["embedding"] = vec
            updated += 1

        if updated:
            self.graph_store.persist()

        logger.info(
            "[NarrativeGraph][JSON->Graph] embeddings persisted: labels=%s candidates=%d updated=%d",
            labels,
            len(entities),
            updated,
        )
        return updated

    def _episode_embedding_text(self, e: Dict[str, Any], mode: str) -> str:
        name, desc = safe_str(e.get("name")), safe_str(e.get("description"))
        if mode == "name_only":
            return name
        if mode == "desc_only":
            return desc
        return f"{name}\n{desc}".strip() if name and desc else (name or desc)

    def _ensure_episode_embeddings(
        self,
        *,
        episodes: List[Dict[str, Any]],
        embedding_text_field: str,
        batch_size: int,
        save_path: Optional[str],
    ) -> List[Dict[str, Any]]:
        if not episodes:
            return episodes

        self.graph_query_utils.load_embedding_model(self.config.embedding)
        model = getattr(self.graph_query_utils, "model", None)
        if model is None or not hasattr(model, "encode"):
            raise RuntimeError("graph_query_utils.model is not available or lacks encode()")

        need: List[int] = []
        texts: List[str] = []
        for i, e in enumerate(episodes):
            if not isinstance(e, dict):
                continue
            emb = e.get("embedding")
            if isinstance(emb, list) and emb:
                continue
            t = self._episode_embedding_text(e, embedding_text_field)
            if safe_str(t):
                need.append(i)
                texts.append(t)

        if not need:
            return episodes

        bs = max(1, int(batch_size))
        for s in range(0, len(texts), bs):
            vecs = model.encode(texts[s : s + bs])
            vec_list = _to_vec_list(vecs)
            for j, v in enumerate(vec_list):
                idx = need[s + j]
                if 0 <= idx < len(episodes):
                    episodes[idx]["embedding"] = v

        if save_path:
            try:
                json_dump_atomic(save_path, episodes)
            except Exception as e:
                logger.warning("[NarrativeGraph] persist embeddings failed: %s err=%s", save_path, e)
        return episodes

    def _storyline_embedding_text(self, s: Dict[str, Any], mode: str) -> str:
        name = safe_str(s.get("name"))
        desc = safe_str(s.get("description"))
        props = safe_dict(s.get("properties"))
        impact = safe_str(props.get("impact"))
        if mode == "name_only":
            return name
        if mode == "desc_only":
            return desc
        parts = [x for x in [name, desc, impact] if safe_str(x)]
        return "\n".join(parts).strip()

    def _ensure_storyline_embeddings(
        self,
        *,
        storylines: List[Dict[str, Any]],
        embedding_text_field: str,
        batch_size: int,
        save_path: Optional[str],
    ) -> List[Dict[str, Any]]:
        if not storylines:
            return storylines

        self.graph_query_utils.load_embedding_model(self.config.embedding)
        model = getattr(self.graph_query_utils, "model", None)
        if model is None or not hasattr(model, "encode"):
            raise RuntimeError("graph_query_utils.model is not available or lacks encode()")

        need: List[int] = []
        texts: List[str] = []
        for i, s in enumerate(storylines):
            if not isinstance(s, dict):
                continue
            emb = s.get("embedding")
            if isinstance(emb, list) and emb:
                continue
            t = self._storyline_embedding_text(s, embedding_text_field)
            if safe_str(t):
                need.append(i)
                texts.append(t)

        if not need:
            return storylines

        bs = max(1, int(batch_size))
        for st in range(0, len(texts), bs):
            vecs = model.encode(texts[st : st + bs])
            vec_list = _to_vec_list(vecs)
            for j, v in enumerate(vec_list):
                idx = need[st + j]
                if 0 <= idx < len(storylines):
                    storylines[idx]["embedding"] = v

        if save_path:
            try:
                json_dump_atomic(save_path, storylines)
            except Exception as e:
                logger.warning("[NarrativeGraph] persist storyline embeddings failed: %s err=%s", save_path, e)
        return storylines

    # ================================================================
    # Dedupe helpers
    # ================================================================
    def _dedupe_entities_by_id(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        seen: Set[str] = set()
        for e in entities or []:
            if not isinstance(e, dict):
                continue
            eid = safe_str(e.get("id"))
            if eid and eid not in seen:
                seen.add(eid)
                out.append(e)
        return out

    def _dedupe_packs_by_episode_id(self, packs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        seen: Set[str] = set()
        for p in packs or []:
            if not isinstance(p, dict):
                continue
            ep = p.get("episode_entity") or {}
            eid = safe_str(ep.get("id")) if isinstance(ep, dict) else ""
            if eid and eid not in seen:
                seen.add(eid)
                out.append(p)
        return out

    def _dedupe_relations(self, rels: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen: Set[Tuple[str, str, str]] = set()
        out: List[Dict[str, Any]] = []
        for r in rels or []:
            if not isinstance(r, dict):
                continue
            pred = safe_str(r.get("predicate") or r.get("relation_type")).strip()
            s, o = safe_str(r.get("subject_id")), safe_str(r.get("object_id"))
            if not pred or _is_none_relation(pred) or not s or not o:
                continue
            k = (pred, s, o)
            if k in seen:
                continue
            seen.add(k)
            rr = dict(r)
            rr["predicate"] = pred
            rr.setdefault("confidence", 1.0)
            if not isinstance(rr.get("properties"), dict):
                rr["properties"] = {}
            out.append(rr)
        return out

    def _normalize_episode_dag_relations_for_graph_store(self, rels: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for r in rels or []:
            if not isinstance(r, dict):
                continue
            s_id, o_id = safe_str(r.get("subject_id")), safe_str(r.get("object_id"))
            pred = safe_str(r.get("predicate")).strip()
            if not s_id or not o_id or not pred or _is_none_relation(pred):
                continue

            props = safe_dict(r.get("properties"))
            rel_type = safe_str(r.get("relation_type")).strip()
            if rel_type:
                props["relation_type"] = rel_type
            for k in ("type_weight", "effective_weight", "source_relation_ids", "evidence_pool"):
                if k in r:
                    props[k] = safe_list(r.get(k)) if k in {"source_relation_ids", "evidence_pool"} else r.get(k)

            rid = safe_str(r.get("id")) or stable_relation_id(s_id, pred, o_id, prefix="rel_ep_dag_")
            rr = dict(r)
            rr["id"] = rid
            rr["predicate"] = pred
            rr["relation_name"] = safe_str(r.get("relation_name")) or pred
            rr["properties"] = props
            rr.setdefault("confidence", 1.0)
            out.append(rr)

        return self._dedupe_relations(out)

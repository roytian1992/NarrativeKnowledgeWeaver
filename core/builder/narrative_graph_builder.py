# narrative_graph_builder.py

import json
import pickle
import glob
import hashlib
import logging
import os
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED, TimeoutError as FuturesTimeoutError
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from tqdm import tqdm

from core.model_providers.openai_llm import OpenAILLM
from core.utils.neo4j_utils import Neo4jUtils
from core.models.data import Entity
from core.builder.manager.graph_manager import GraphManager
from core.storage.graph_store import GraphStore
from core.storage.vector_store import VectorStore
from core.utils.prompt_loader import PromptLoader
from core.utils.format import correct_json_format, format_event_card
from core.builder.graph_builder import DOC_TYPE_META
from core.utils.function_manager import run_with_soft_timeout_and_retries

# Event chain preprocessing and deduplication utilities (from event_processor.py)
from core.builder.event_processor import (
    segment_trunks_and_branches,
    remove_subset_paths,
    remove_similar_paths,
)

PER_TASK_TIMEOUT = 900.0
MAX_RETRIES = 3
RETRY_BACKOFF = 30


class EventCausalityBuilder:
    """
    A builder for constructing event causality and plot graphs.
    Provides utilities for:
    - Event list extraction
    - Event card pre-generation
    - Event pair filtering
    - Causality checking (parallel, with retries)
    - Plot chain construction and relation inference
    """

    def __init__(self, config):
        """
        Initialize EventCausalityBuilder.

        Args:
            config: Configuration object containing builder, storage, and model settings.
        """
        self.config = config
        self.llm = OpenAILLM(config)
        self.graph_store = GraphStore(config)
        self.vector_store = VectorStore(config, "documents")
        self.event_fallback = []
        self.doc_type = config.knowledge_graph_builder.doc_type
        if self.doc_type not in DOC_TYPE_META:
            raise ValueError(f"Unsupported doc_type: {self.doc_type}")
        self.meta = DOC_TYPE_META[self.doc_type]
        self.neo4j_utils = Neo4jUtils(self.graph_store.driver, self.doc_type)
        self.neo4j_utils.load_embedding_model(config.graph_embedding)
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

        # Parameters
        self.causality_threshold = "Medium"
        self.logger = logging.getLogger(__name__)
        self.sorted_scenes = []
        self.event_list = []
        self.event2section_map: Dict[str, str] = {}
        self.max_depth = config.event_plot_graph_builder.max_depth
        self.check_weakly_connected_components = config.event_plot_graph_builder.check_weakly_connected_components
        self.min_component_size = config.event_plot_graph_builder.min_connected_component_size
        self.max_workers = config.event_plot_graph_builder.max_workers
        self.max_iteration = config.event_plot_graph_builder.max_iterations
        self.max_num_triangles = config.event_plot_graph_builder.max_num_triangles
        self.event_cards: Dict[str, Dict[str, Any]] = {}

        # Edge confidence threshold for plot extraction (default 0.5)
        self.min_edge_confidence = config.event_plot_graph_builder.min_confidence

        # Similarity threshold for chain deduplication (default 0.75)
        self.chain_similarity_threshold = 0.75

        self.logger.info("EventCausalityBuilder initialized")

    # ----------------------------- Prompt Construction -----------------------------
    def construct_system_prompt(self, background, abbreviations):
        """
        Construct the system prompt with background and abbreviation information.
        """
        background_info = self.get_background_info(background, abbreviations)
        system_prompt_id = "agent_prompt_screenplay" if self.doc_type == "screenplay" else "agent_prompt_novel"
        system_prompt_text = self.prompt_loader.render_prompt(
            system_prompt_id, {"background_info": background_info}
        )
        return system_prompt_text

    def get_background_info(self, background, abbreviations):
        """
        Format background and abbreviation blocks for prompt injection.
        """
        bg_block = f"**Background**: {background}\n" if background else ""

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

    # ----------------------------- Event List Construction -----------------------------
    def build_event_list(self) -> List[Entity]:
        """
        Build an ordered list of events from sections, based on the 'contains' relation.
        """
        print("🔍 Building event list...")
        section_entities = self.neo4j_utils.search_entities_by_type(
            entity_type=self.meta["section_label"]
        )
        self.sorted_sections = sorted(
            section_entities,
            key=lambda e: int(e.properties.get("order", 99999))
        )
        print(f"✅ Found {len(self.sorted_sections)} sections")

        event_list: List[Entity] = []
        event2section_map: Dict[str, str] = {}
        for section in tqdm(self.sorted_sections, desc="Extracting events from sections"):
            results = self.neo4j_utils.search_related_entities(
                source_id=section.id,
                predicate=self.meta["contains_pred"],
                entity_types=["Event"],
                return_relations=False
            )
            if not results and self.event_fallback:
                results = self.neo4j_utils.search_related_entities(
                    source_id=section.id,
                    relation_types=[self.meta["contains_pred"]],
                    entity_types=self.event_fallback,
                    return_relations=False
                )
            for result in results:
                if result.id not in event2section_map:
                    event2section_map[result.id] = section.id
                    event_list.append(result)

        self.event_list = event_list
        self.event2section_map = event2section_map

        print(f"✅ Event list built, total {len(event_list)} events")
        return event_list

    # ----------------------------- Event Card Precomputation -----------------------------
        # ----------------------------- Event Card Precomputation -----------------------------
    def precompute_event_cards(
        self,
        events: List[Entity],
        per_task_timeout: float = 300,
        max_retries: int = 3,
        retry_timeout: float = 60.0,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Precompute event cards for a list of events using concurrent execution.
        Caches results to disk and skips already-computed events.

        Args:
            events (List[Entity]): List of event entities to process.
            per_task_timeout (float): Per-task soft timeout (seconds) for LLM calls.
            max_retries (int): Number of retry rounds for failed tasks.
            retry_timeout (float): Backoff for retries (seconds). (Kept for interface compatibility)

        Returns:
            Dict[str, Dict[str, Any]]: Mapping from event_id to serialized event card (JSON string).
        """
        base = self.config.storage.event_plot_graph_path
        os.makedirs(base, exist_ok=True)
        cache_path = os.path.join(base, "event_cards.json")

        # Load cache if it exists
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
            print(f"🗂️ Event cards already exist: {len(self.event_cards)} cached, skipping generation.")
            return self.event_cards

        def _gen_event_ctx_with_timeout(info: str, related_ctx: str, timeout: float):
            """
            Wrapper for generate_event_context that tolerates an optional timeout parameter.
            """
            try:
                return self.graph_analyzer.generate_event_context(info, related_ctx, timeout=timeout)
            except TypeError:
                # Backward compatibility when underlying function has no 'timeout'
                return self.graph_analyzer.generate_event_context(info, related_ctx)

        def _collect_related_context_by_section(ev: Entity) -> str:
            """
            Collect textual context relevant to an event by its section (or fallback by chunks).
            Returns a concatenated string of distinct contexts.
            """
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
            """
            Build a single event card JSON string for the given event.
            """
            info = self.neo4j_utils.get_entity_info(ev.id, "Event", True, True)
            related_ctx = _collect_related_context_by_section(ev)
            llm_timeout = max(5.0, min(per_task_timeout - 5.0, 1200.0))
            out = _gen_event_ctx_with_timeout(info, related_ctx, timeout=llm_timeout)
            card = json.loads(correct_json_format(out))["event_card"]
            card = format_event_card(card)
            return card

        def _placeholder(ev: Entity, exc=None) -> str:
            """
            Placeholder event card JSON string used when a task times out or fails.
            """
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

        res_map, still_failed = run_with_soft_timeout_and_retries(
            pending_events,
            work_fn=_build_one,
            key_fn=lambda e: e.id,
            desc_label="Precompute event cards",
            per_task_timeout=per_task_timeout,
            retries=max_retries,
            retry_backoff=30,
            allow_placeholder_first_round=True,
            placeholder_fn=_placeholder,
            should_retry=None,
            max_workers=self.max_workers
        )

        for k, v in res_map.items():
            self.event_cards[k] = v
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(self.event_cards, f, ensure_ascii=False, indent=2)

        print(f"🗂️ Event card generation finished: total {len(self.event_cards)}, remaining gaps {len(still_failed)}")
        return self.event_cards

    # ----------------------------- Candidate Event-Pair Filtering -----------------------------
    def filter_event_pairs_by_community(
        self,
        events: List[Entity],
        max_depth: int = 3
    ) -> List[Tuple[Entity, Entity]]:
        """
        Filter candidate event pairs to those within the same community.

        Args:
            events (List[Entity]): List of candidate events.
            max_depth (int): Unused here, kept for interface compatibility.

        Returns:
            List[Tuple[Entity, Entity]]: Filtered event pairs (source, target).
        """
        id2entity = {e.id: e for e in events}
        pairs = self.neo4j_utils.fetch_event_pairs_same_community()
        filtered_pairs = []
        for row in pairs:
            src_id, dst_id = row["srcId"], row["dstId"]
            if src_id in id2entity and dst_id in id2entity:
                filtered_pairs.append((id2entity[src_id], id2entity[dst_id]))
        print(f"[✓] Event pairs in the same community: {len(filtered_pairs)}")
        return filtered_pairs

    def sort_event_pairs_by_section_order(
        self, pairs: List[Tuple[Entity, Entity]]
    ) -> List[Tuple[Entity, Entity]]:
        """
        Sort event pairs based on the order of their sections (ascending).
        Ensures pair ordering matches chronological/structural section order.

        Args:
            pairs (List[Tuple[Entity, Entity]]): List of event pairs.

        Returns:
            List[Tuple[Entity, Entity]]: Order-normalized pairs.
        """
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
        """
        Keep pairs that are reachable within max depth or are semantically similar.

        Args:
            pairs (List[Tuple[Entity, Entity]]): Input event pairs.

        Returns:
            List[Tuple[Entity, Entity]]: Filtered event pairs.
        """
        filtered_pairs = []
        for pair in tqdm(pairs, desc="Filter node pairs"):
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
                if score is not None and score >= 0.5:
                    filtered_pairs.append(pair)
        return filtered_pairs

    # ----------------------------- Causality Checking (Parallel + Checkpointing) -----------------------------
    def check_causality_batch(
        self,
        pairs: List[Tuple[Entity, Entity]]
    ) -> Dict[Tuple[str, str], Dict[str, Any]]:
        """
        Check causality for event pairs concurrently, with soft timeouts and retries.

        Args:
            pairs (List[Tuple[Entity, Entity]]): List of event pairs to evaluate.

        Returns:
            Dict[Tuple[str, str], Dict[str, Any]]: Mapping from (src_id, tgt_id) to causality result dict.
        """
        PER_TASK_TIMEOUT = 1800
        MAX_RETRIES = 3
        RETRY_BACKOFF = 2.0

        def _make_result(src_event, tgt_event,
                        relation="NONE",
                        reason="",
                        temporal_order="Unknown",
                        confidence=0.0,
                        raw_result="",
                        timeout=False) -> Dict[str, Any]:
            """
            Uniform result structure for a causality check.
            """
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
            """
            Retrieve shared neighbor information for two events and format it as text.
            """
            commons = self.neo4j_utils.get_common_neighbors(src_id, tgt_id, limit=50)
            info = "Common neighbors for the two events are as follows:\n"
            if not commons:
                return info + "None"
            for ent_ in commons:
                try:
                    ent_type = "/".join(ent_.type) if isinstance(ent_.type, (list, set, tuple)) else str(ent_.type)
                except Exception:
                    ent_type = "Unknown"
                info += f"- Name: {ent_.name}, Type: {ent_type}, Description: {ent_.description}\n"
            return info

        def _ensure_card(e: Entity, info_text: str):
            """
            Ensure an event card exists for the given event, generating if absent.
            """
            if e.id in self.event_cards:
                return self.event_cards[e.id]
            try:
                out = self.graph_analyzer.generate_event_context(info_text, "", timeout=PER_TASK_TIMEOUT - 60)
            except TypeError:
                out = self.graph_analyzer.generate_event_context(info_text, "")
            card = json.loads(correct_json_format(out))["event_card"]
            card = format_event_card(card)
            self.event_cards[e.id] = card
            return card

        def _work(pair: Tuple[Entity, Entity]) -> Dict[str, Any]:
            """
            Worker function to evaluate causality for a single event pair.
            """
            src_event, tgt_event = pair
            try:
                info_1 = self.neo4j_utils.get_entity_info(src_event.id, "Event", True, True)
                info_2 = self.neo4j_utils.get_entity_info(tgt_event.id, "Event", True, True)
                related_context = info_1 + "\n" + info_2 + "\n" + _get_common_neighbor_info(src_event.id, tgt_event.id)
                src_card = _ensure_card(src_event, info_1)
                tgt_card = _ensure_card(tgt_event, info_2)

                try:
                    result_json = self.graph_analyzer.check_event_causality(
                        src_card, tgt_card,
                        system_prompt=self.system_prompt_text,
                        related_context=related_context,
                        timeout=max(5.0, PER_TASK_TIMEOUT - 60.0)
                    )
                except TypeError:
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
                    reason=f"Error during check: {e}",
                    temporal_order="Unknown",
                    confidence=0.0,
                    raw_result="",
                    timeout=True
                )

        def _placeholder(pair: Tuple[Entity, Entity], exc=None) -> Dict[str, Any]:
            """
            Placeholder result used when a task times out or encounters an exception.
            """
            src_event, tgt_event = pair
            return _make_result(
                src_event, tgt_event,
                relation="NONE",
                reason="Soft timeout/exception; placeholder result",
                temporal_order="Unknown",
                confidence=0.0,
                raw_result="",
                timeout=True
            )

        def _should_retry(res: Dict[str, Any]) -> bool:
            """
            Decide whether to retry based on the result.
            """
            if res.get("causality_timeout"):
                return True
            reason = (res.get("reason") or "").strip()
            return ("Error" in reason)

        print(f"🔍 Parallel causality check for {len(pairs)} event pairs starting...")

        res_map, still_failed = run_with_soft_timeout_and_retries(
            pairs,
            work_fn=_work,
            key_fn=lambda p: (p[0].id, p[1].id),
            desc_label="Parallel causality check",
            per_task_timeout=PER_TASK_TIMEOUT,
            retries=MAX_RETRIES,
            retry_backoff=RETRY_BACKOFF,
            allow_placeholder_first_round=True,
            placeholder_fn=_placeholder,
            should_retry=_should_retry,
            max_workers=self.max_workers
        )

        for k in still_failed:
            if k in res_map:
                res_map[k]["final_fallback"] = True
                res_map[k]["retries"] = MAX_RETRIES

        print(f"✅ Causality check completed (success {len(pairs) - len(still_failed)} / {len(pairs)}; still failed {len(still_failed)})")
        return res_map

    # ----------------------------- Initialization: Subgraph + Louvain -----------------------------
    def initialize(self, keep_event_cards=False):
        """
        Initialize the working graph:
        - Remove previous causality relations
        - Create a working subgraph (excluding section nodes and 'contains' relations)
        - Run Louvain community detection
        - Clear output directory (optionally keeping precomputed event cards)
        """
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

        self.clear_directory(self.config.storage.event_plot_graph_path, keep_event_cards)

    def clear_directory(self, path, keep_event_cards=False):
        """
        Clear JSON artifacts in the target directory, optionally keeping event cards.

        Args:
            path (str): Directory to clean.
            keep_event_cards (bool): If True, skip files that contain 'event_card' in the name.
        """
        for file in glob.glob(os.path.join(path, "*.json")):
            try:
                if keep_event_cards and "event_card" in file:
                    continue
                os.remove(file)
            except Exception as e:
                print(f"Failed to delete: {file} -> {e}")

    # ----------------------------- Main Pipeline 1: Build Event Causality Graph -----------------------------
    def build_event_causality_graph(
        self,
        limit_events: Optional[int] = None
    ) -> None:
        """
        Build the full event causality graph:
        1) Build event list
        2) Precompute/Load event cards
        3) Filter candidate event pairs
        4) Check causality in parallel
        5) Persist results and write edges back to Neo4j
        6) Create an event-causality projection graph
        """
        print("🚀 Starting the full event causality graph construction pipeline...")

        base = self.config.storage.event_plot_graph_path
        os.makedirs(base, exist_ok=True)

        print("\n🔍 Building event list...")
        event_list = self.build_event_list()

        if limit_events and limit_events < len(event_list):
            event_list = event_list[:limit_events]
            print(f"⚠️ Limit number of events to process: {limit_events}")

        event_cards_path = os.path.join(base, "event_cards.json")
        if os.path.exists(event_cards_path):
            try:
                with open(event_cards_path, "r", encoding="utf-8") as f:
                    self.event_cards = json.load(f)
                    if not isinstance(self.event_cards, dict):
                        self.event_cards = {}
                print(f"🗂️ Reusing existing event cards: {len(self.event_cards)}")
            except Exception:
                self.event_cards = {}
        else:
            print("\n🧩 Precomputing event cards in parallel...")
            self.precompute_event_cards(event_list)

        print("\n🔍 Filtering event pairs...")
        filtered_pairs = self.filter_event_pairs_by_community(event_list, max_depth=self.max_depth)
        filtered_pairs = self.filter_pair_by_distance_and_similarity(filtered_pairs)
        filtered_pairs = self.sort_event_pairs_by_section_order(filtered_pairs)
        print("     Final number of candidate event pairs: ", len(filtered_pairs))

        print("\n🔍 Checking causality...")
        causality_results = self.check_causality_batch(filtered_pairs)

        results = []
        for (src_id, tgt_id), info in causality_results.items():
            results.append({
                "src_id": src_id,
                "tgt_id": tgt_id,
                "relation": info.get("relation")
            })    

        with open(os.path.join(base, "event_causality_results.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        with open(event_cards_path, "w", encoding="utf-8") as f:
            json.dump(self.event_cards, f, ensure_ascii=False, indent=2)

        print("\n🔗 Writing event-to-event relations back to Neo4j...")
        self.write_event_cause_edges(causality_results)
        self.neo4j_utils.create_event_causality_graph("event_causality_graph", force_refresh=True)

    def write_event_cause_edges(self, causality_results):
        """
        Write event causality edges to Neo4j.

        Args:
            causality_results (Dict[Tuple[str, str], Dict[str, Any]]): Causality result map.
        """
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

    # ----------------------------- SABER: Structure-Constrained Edge Pruning -----------------------------
    def detect_flattened_causal_patterns(self, edges: List[Dict]) -> List[Dict]:
        """
        Detect patterns where a single source has multiple targets with internal links among targets.

        Args:
            edges (List[Dict]): List of edges with 'sid' and 'tid'.

        Returns:
            List[Dict]: List of patterns with structure:
                {
                  "source": <event_id>,
                  "targets": [<event_id>, ...],
                  "internal_links": [(u, v), ...]
                }
        """
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
        """
        Filter out targets with strong source->target confidence and keep internal links with confidence >= threshold.
        The remaining pattern highlights suspect branches to re-check.

        Args:
            patterns (List[Dict]): Patterns detected by detect_flattened_causal_patterns.
            edge_map (Dict[Tuple[str, str], Dict]): Map from (src, tgt) to edge info with 'confidence'.
            conf_threshold (float): Confidence threshold.

        Returns:
            List[Dict]: Cleaned patterns focusing on weak source->target edges with some internal links preserved.
        """
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
        """
        Compute the set difference of edges between original and filtered patterns.

        Args:
            original_patterns (List[Dict]): Original pattern list.
            filtered_patterns (List[Dict]): Pattern list after filtering.

        Returns:
            Set[Tuple[str, str]]: Edges that were removed by filtering.
        """
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
        """
        Prepare context details (similarities and confidences) for a triangle pattern.

        Args:
            pattern (Dict): Pattern with 'source', 'targets', and 'internal_links'.
            edge_map (Dict): Edge info map with confidences.

        Returns:
            List[Dict]: Context records for LLM judgment, one per internal link.
        """
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

            context_to_check.append({
                "entities": [source, link[0], link[1]],
                "details": [
                    {"edge": [source, link[0]], "similarity": mid_tgt_sim, "confidence": src_mid_conf},
                    {"edge": [source, link[1]], "similarity": src_tgt_sim, "confidence": src_tgt_conf},
                    {"edge": [link[0], link[1]], "similarity": src_mid_sim, "confidence": mid_tgt_conf},
                ]
            })
        return context_to_check

    def prepare_context(self, pattern_detail):
        """
        Prepare textual context for LLM-based redundancy evaluation on a triangle pattern.

        Args:
            pattern_detail (Dict): Record from filter_pattern with 'entities' and 'details'.

        Returns:
            Tuple[str, str]:
                - full_event_details: Formatted descriptions of the three events.
                - full_relation_details: Existing causality relations among them with similarities/confidences.
        """
        def _safe_str(x: Any) -> str:
            return x if isinstance(x, str) else ("" if x is None else str(x))

        event_details = self.neo4j_utils.get_event_details(pattern_detail["entities"])
        full_event_details = "Descriptions of the three event entities are as follows:\n"

        for i, event_info in enumerate(event_details):
            event_id = event_info["event_id"]
            full_event_details += f"**Descriptions for Event {i+1}:**\nEvent id: {event_id}\n"

            background = self.neo4j_utils.get_entity_info(event_id, "Event", True, True)
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
                background += "\nEvent properties are as follows:\n"
                for k, v in non_empty_props.items():
                    background += f"- {k}: {v}\n"

            if i + 1 != len(event_details):
                background += "\n"

            full_event_details += background

        full_relation_details = "Existing causality relations among them are as follows:\n"
        relation_details = pattern_detail["details"]
        for i, relation_info in enumerate(relation_details):
            src, tgt = relation_info["edge"]
            rel_summary = self.neo4j_utils.get_relation_summary(src, tgt, "EVENT_CAUSES")
            rel_summary = _safe_str(rel_summary)
            background = f"{i+1}. " + rel_summary
            background += f"\nRelation semantic similarity: {round(relation_info['similarity'], 4)}, confidence: {relation_info['confidence']}."
            if i + 1 != len(relation_details):
                background += "\n\n"
            full_relation_details += background

        return full_event_details, full_relation_details

    def run_SABER(self):
        """
        Run the SABER pruning loop:
        - Detect problematic triangle patterns within SCC/WCC components
        - Ask an LLM to judge redundancy
        - Remove edges accordingly
        - Iterate until convergence or iteration limit
        """
        loop_count = 0
        while True:
            print(f"\n===== [Optimization round {loop_count + 1}] =====")

            scc_components = self.neo4j_utils.fetch_scc_components("event_causality_graph", 2)
            wcc_components = []
            if self.check_weakly_connected_components:
                wcc_components = self.neo4j_utils.fetch_wcc_components("event_causality_graph", self.min_component_size)

            connected_components = scc_components + wcc_components
            print(f"📌 Current number of connected components: SCC={len(scc_components)}, WCC={len(wcc_components)}")

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

            print(f"🔺 Number of triangle structures to judge this round: {len(all_triangles)}")
            if len(all_triangles) >= self.max_num_triangles:
                print(f"⚠️ Too many triangle structures detected; only the first {self.max_num_triangles} will be processed.")
                all_triangles = all_triangles[:self.max_num_triangles]
                return

            if loop_count >= 1:
                if len(scc_components) == 0 and len(all_triangles) == 0:
                    print("✅ No SCCs and no pending triangles; stopping.")
                    break
            elif loop_count >= self.max_iteration:
                break

            def process_triangle(triangle_):
                """
                Determine whether to remove an edge based on LLM judgment for a single triangle.
                """
                try:
                    event_details, relation_details = self.prepare_context(triangle_)
                    chunks = [self.neo4j_utils.get_entity_by_id(ent_id).source_chunks[0] for ent_id in triangle_["entities"]]
                    chunks = list(set(chunks))
                    documents = self.vector_store.search_by_ids(chunks)
                    results = {doc.content for doc in documents}
                    related_context = "\n".join(list(results))
                    try:
                        output = self.graph_analyzer.evaluate_event_redundancy(
                            event_details, relation_details, self.system_prompt_text, related_context,
                            timeout=600.0 - 5.0
                        )
                    except TypeError:
                        output = self.graph_analyzer.evaluate_event_redundancy(
                            event_details, relation_details, self.system_prompt_text, related_context
                        )
                    output = json.loads(correct_json_format(output))
                    if output.get("remove_edge", False):
                        return (triangle_["entities"][0], triangle_["entities"][2])
                except Exception as e:
                    print(f"[⚠️ Error] Triangle judgment failed: {triangle_['entities']}, error: {str(e)}")
                return None

            print(f"🧠 Judging triangle structures in parallel...")
            tri_map, tri_failed = run_with_soft_timeout_and_retries(
                all_triangles,
                work_fn=process_triangle,
                key_fn=lambda tri: tuple(tri["entities"]),
                desc_label="LLM triangle judgment",
                per_task_timeout=600.0,
                retries=1,
                retry_backoff=30,
                allow_placeholder_first_round=False,
                placeholder_fn=None,
                should_retry=lambda r: r is None,
                max_workers=self.max_workers
            )

            removed_edges = [edge for edge in tri_map.values() if edge]
            loop_count += 1
            print(f"❌ Number of edges to remove this round: {len(set(removed_edges))}")

            for edge in removed_edges:
                self.neo4j_utils.delete_relation_by_ids(edge[0], edge[1], ["EVENT_CAUSES", "EVENT_INDIRECT_CAUSES", "EVENT_PART_OF"])

            base = self.config.storage.event_plot_graph_path
            os.makedirs(base, exist_ok=True)
            saber_log_path = os.path.join(base, f"saber_removed_edges_round_{loop_count+1}.json")
            try:
                with open(saber_log_path, "w", encoding="utf-8") as f:
                    json.dump([list(e) for e in set(removed_edges)], f, ensure_ascii=False, indent=2)
            except Exception:
                pass

            self.neo4j_utils.create_event_causality_graph("event_causality_graph", min_confidence=0, force_refresh=True)

    # ----------------------------- Utilities: Extract All Event Chains & Text Context -----------------------------
    def get_all_event_chains(self, min_confidence: float = 0.0):
        """
        Retrieve all event chains starting from 'starting events' with a given confidence threshold.

        Args:
            min_confidence (float): Minimum edge confidence to include in chains.

        Returns:
            List[List[str]]: List of event chains (each a list of event IDs).
        """
        starting_events = self.neo4j_utils.get_starting_events()
        chains = []
        for event in starting_events:
            all_chains = self.neo4j_utils.find_event_chain(event, min_confidence)
            chains.extend([chain for chain in all_chains if len(chain) >= 2])
        return chains

    def prepare_chain_context(self, chain: List[str]) -> str:
        """
        Prepare a textual summary for a given chain using precomputed event cards.

        Args:
            chain (List[str]): Sequence of event IDs.

        Returns:
            str: Formatted chain description including event card content.
        """
        if len(chain) > 1:
            context = "Event chain: " + "->".join(chain) + "\n\nDetailed event information:\n"
        else:
            context = f"Event: {chain[0]}" + "\n\nDetailed event information:\n"
        for i, event in enumerate(chain):
            context += f"Event {i+1}: {event}\n" + self.event_cards[event] + "\n"
        return context

    def prepare_graph_embeddings(self):
        """
        Build/update graph embeddings (entities and vector index) used for querying and similarity.
        """
        self.neo4j_utils.load_embedding_model(self.config.graph_embedding)
        self.neo4j_utils.create_vector_index()
        self.neo4j_utils.process_all_embeddings(
            entity_types=["Event", "Plot"]
        )
        self.neo4j_utils.ensure_entity_superlabel()
        print("✅ Vector construction for event-plot graph completed")

    # ----------------------------- Main Pipeline 2: Build Plot-Event Graph -----------------------------
    def build_event_plot_graph(self):
        """
        Build plot nodes from event chains, deduplicate chains, and write plots into Neo4j.
        """
        # Reset the old Plot graph
        self.neo4j_utils.reset_event_plot_graph()

        base = self.config.storage.event_plot_graph_path
        os.makedirs(base, exist_ok=True)

        cards_path = os.path.join(base, "event_cards.json")
        with open(cards_path, "r", encoding="utf-8") as f:
            self.event_cards = json.load(f)

        # 1) Use an edge-confidence threshold (default 0.5) to retrieve event chains
        all_chains = self.get_all_event_chains(min_confidence=self.min_edge_confidence)
        print("[✓] Total number of event chains (after thresholding):", len(all_chains))

        # 2) Structural segmentation (trunks/branches) -> "plot candidate segments"
        filtered_chains = segment_trunks_and_branches(
            all_chains,
            min_len=2,
            include_cutpoint=True,      # Recommended: include cutpoints
            keep_terminal_pairs=True    # Recommended: ensure leaves appear at least once
        )
        print("[✓] Number of candidate segments after structural segmentation:", len(filtered_chains))
    
        before = len(filtered_chains)
        filtered_chains = remove_subset_paths(filtered_chains)
        filtered_chains = remove_similar_paths(filtered_chains, threshold=self.chain_similarity_threshold)
        print(f"[✓] Number of candidate segments after deduplication: {len(filtered_chains)} (original {before})")

        # Persist candidate chains for debugging
        with open(os.path.join(base, "filtered_event_chains.json"), "w", encoding="utf-8") as f:
            json.dump(filtered_chains, f, ensure_ascii=False, indent=2)

        def _stable_plot_id(title: str, chain: List[str]) -> str:
            key = f"{title}||{'->'.join(chain)}"
            return "plot_" + hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]

        def _to_bool(v) -> bool:
            if isinstance(v, bool):
                return v
            if v is None:
                return False
            return str(v).strip().lower() in ("true", "yes", "1")

        def process_chain(chain: List[str]):
            """
            Produce a plot node from a single event chain via LLM; write to Neo4j if a plot is confirmed.

            Returns:
                True if a plot node is created; False if rejected by the LLM; None on error (for retry).
            """
            try:
                event_chain_info = self.prepare_chain_context(chain)
                # Aggregate evidence context (by chunk)
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

                try:
                    raw = self.graph_analyzer.generate_event_plot(
                        event_chain_info=event_chain_info,
                        system_prompt=self.system_prompt_text,
                        related_context=related_context,
                        timeout=900.0 - 30.0
                    )
                except TypeError:
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
                    title = f"Plot chain: {chain[0]}→{chain[-1]}"

                plot_info["id"] = _stable_plot_id(title, chain)
                plot_info["event_ids"] = chain
                plot_info["reason"] = result.get("reason", "")

                # Write Plot node and HAS_EVENT edges
                self.neo4j_utils.write_plot_to_neo4j(plot_data=plot_info)
                return True

            except Exception as e:
                print(f"[!] Error processing event chain {chain}: {e}")
                return None

        chain_map, chain_failed = run_with_soft_timeout_and_retries(
            filtered_chains,
            work_fn=process_chain,
            key_fn=lambda ch: tuple(ch),
            desc_label="Generate plot graph in parallel",
            per_task_timeout=600.0,
            retries=3,
            retry_backoff=60,
            allow_placeholder_first_round=False,
            placeholder_fn=None,
            should_retry=lambda r: r is None,
            max_workers=self.max_workers
        )

        success_count = sum(1 for v in chain_map.values() if v)
        print(f"[✓] Successfully generated plots: {success_count}/{len(filtered_chains)}; still failed {len(chain_failed)}")

        return

    # ----------------------------- Main Pipeline 3: Extract Plot-Plot Relations -----------------------------
    def generate_plot_relations(self):
        """
        Generate relations between Plot nodes.

        Directed relations:
            - PLOT_PREREQUISITE_FOR
            - PLOT_ADVANCES
            - PLOT_BLOCKS
            - PLOT_RESOLVES

        Undirected relations:
            - PLOT_CONFLICTS_WITH
            - PLOT_PARALLELS

        Output artifact:
            EPG/plot_relations_created.json
        """
        self.neo4j_utils.process_all_embeddings(entity_types=[self.meta["section_label"]])
        self.neo4j_utils.create_event_plot_graph()
        self.neo4j_utils.run_node2vec()

        all_plot_pairs = self.neo4j_utils.get_plot_pairs(threshold=0)
        print("[✓] Number of plot pairs to evaluate:", len(all_plot_pairs))

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

        def _make_edge(src_id, tgt_id, rtype, confidence, reason):
            return {
                "src": src_id,
                "tgt": tgt_id,
                "relation_type": rtype,
                "confidence": float(confidence) if confidence is not None else 0.0,
                "reason": reason or ""
            }

        def _work(pair: dict) -> dict:
            """
            Worker function to classify the relation between a pair of plots and produce edges.
            """
            try:
                plot_A_info = self.neo4j_utils.get_entity_info(
                    pair["src"], "Plot", contain_properties=True, contain_relations=True
                )
                plot_B_info = self.neo4j_utils.get_entity_info(
                    pair["tgt"], "Plot", contain_properties=True, contain_relations=True
                )
                try:
                    result = self.graph_analyzer.extract_plot_relation(
                        plot_A_info, plot_B_info, self.system_prompt_text, timeout=PER_TASK_TIMEOUT - 30.0
                    )
                except TypeError:
                    result = self.graph_analyzer.extract_plot_relation(
                        plot_A_info, plot_B_info, self.system_prompt_text
                    )

                try:
                    result = json.loads(correct_json_format(result))
                except Exception:
                    if not isinstance(result, dict):
                        raise

                rtype = result.get("relation_type")
                direction = result.get("direction", None)
                confidence = result.get("confidence", 0.0)
                reason = result.get("reason", "")

                if rtype not in VALID_TYPES:
                    return {"status": "error", "edges": [], "reason": f"unknown relation_type: {rtype}"}

                if rtype in {"None", None}:
                    return {"status": "none", "edges": [], "reason": "no-relation"}

                if rtype in DIRECTED:
                    if direction == "A->B":
                        edges = [_make_edge(pair["src"], pair["tgt"], rtype, confidence, reason)]
                    elif direction == "B->A":
                        edges = [_make_edge(pair["tgt"], pair["src"], rtype, confidence, reason)]
                    else:
                        return {"status": "error", "edges": [], "reason": f"missing/invalid direction: {direction}"}
                else:
                    edges = [
                        _make_edge(pair["src"], pair["tgt"], rtype, confidence, reason),
                        _make_edge(pair["tgt"], pair["src"], rtype, confidence, reason),
                    ]

                return {"status": "ok", "edges": edges, "reason": ""}

            except Exception as e:
                return {"status": "error", "edges": [], "reason": str(e)}

        def _should_retry(res: dict) -> bool:
            """
            Retry only when status is 'error'.
            """
            return isinstance(res, dict) and res.get("status") == "error"

        res_map, still_failed = run_with_soft_timeout_and_retries(
            all_plot_pairs,
            work_fn=_work,
            key_fn=lambda p: (p["src"], p["tgt"]),
            desc_label="Extract plot relations",
            per_task_timeout=PER_TASK_TIMEOUT,
            retries=MAX_RETRIES,
            retry_backoff=RETRY_BACKOFF,
            allow_placeholder_first_round=False,
            placeholder_fn=None,
            should_retry=_should_retry,
            max_workers=self.max_workers
        )

        edges_to_add = []
        for out in res_map.values():
            if isinstance(out, dict) and out.get("status") == "ok" and out.get("edges"):
                edges_to_add.extend(out["edges"])

        if edges_to_add:
            self.neo4j_utils.create_plot_relations(edges_to_add)
            print(f"[✓] Created {len(edges_to_add)} plot relations")

            base = self.config.storage.event_plot_graph_path
            os.makedirs(base, exist_ok=True)
            with open(os.path.join(base, "plot_relations_created.json"), "w", encoding="utf-8") as f:
                json.dump(edges_to_add, f, ensure_ascii=False, indent=2)
        else:
            print("[!] No plot relations were generated")

        if still_failed:
            print(f"[!] Number of plot pairs still failing: {len(still_failed)}")

# narrative_graph_builder.py
# -*- coding: utf-8 -*-

import json
import pickle
import glob
import hashlib
import logging
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

from tqdm import tqdm
from core.builder.manager.document_manager import DocumentParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
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

# Event chain preprocessing and deduplication
from core.builder.event_processor import (
    segment_trunks_and_branches,
    remove_subset_paths,
    remove_similar_paths,
)

PER_TASK_TIMEOUT = 900.0
MAX_RETRIES = 3
RETRY_BACKOFF = 30


# ============ 小工具：JSONL ============
def write_jsonl(path: str, rows: List[Dict[str, Any]]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


class EventCausalityBuilder:
    """
    两阶段（产物/落库）合并为 6 个高层方法：
      1) initialize(keep_event_cards=True)
      2) produce_causality_artifacts(limit_events=None)
      3) materialize_causality_graph()
      4) produce_plot_artifacts()
      5) materialize_plot_graph()
      6) prepare_graph_embeddings()
    """

    def __init__(self, config):
        self.config = config
        self.llm = OpenAILLM(config)
        self.graph_store = GraphStore(config)
        self.vector_store = VectorStore(config, "documents")
        self.doc_type = config.knowledge_graph_builder.doc_type
        if self.doc_type not in DOC_TYPE_META:
            raise ValueError(f"Unsupported doc_type: {self.doc_type}")
        self.meta = DOC_TYPE_META[self.doc_type]
        self.neo4j_utils = Neo4jUtils(self.graph_store.driver, self.doc_type)
        # print("****", self.neo4j_utils)
        self.neo4j_utils.load_embedding_model(config.graph_embedding)
        prompt_dir = config.knowledge_graph_builder.prompt_dir
        self.prompt_loader = PromptLoader(prompt_dir)

        # settings for prompt
        settings_path = os.path.join(self.config.storage.graph_schema_path, "settings.json")
        if not os.path.exists(settings_path):
            settings_path = self.config.probing.default_background_path
        settings = json.load(open(settings_path, "r", encoding="utf-8"))
        self.system_prompt_text = self.construct_system_prompt(
            background=settings.get("background"),
            abbreviations=settings.get("abbreviations", []),
        )
        self.graph_analyzer = GraphManager(config, self.llm)

        # params
        self.causality_threshold = "Medium"
        self.logger = logging.getLogger(__name__)
        self.event_fallback = []
        self.sorted_sections = []
        self.event_list: List[Entity] = []
        self.event2section_map: Dict[str, str] = {}
        self.max_depth = config.event_plot_graph_builder.max_depth
        self.check_weakly_connected_components = config.event_plot_graph_builder.check_weakly_connected_components
        self.min_component_size = config.event_plot_graph_builder.min_connected_component_size
        self.max_workers = config.event_plot_graph_builder.max_workers
        self.max_iteration = config.event_plot_graph_builder.max_iterations
        self.max_num_triangles = config.event_plot_graph_builder.max_num_triangles
        self.chain_similarity_threshold = 0.75
        self.min_edge_confidence = config.event_plot_graph_builder.min_confidence

        self.event_cards: Dict[str, Dict[str, Any]] = {}
        self.base_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.document_processing.chunk_size,
            chunk_overlap=config.document_processing.chunk_overlap,
        )
        self.document_parser = DocumentParser(config, self.llm)

        self.logger.info("EventCausalityBuilder initialized")

    # ---------------- Prompt ----------------
    def construct_system_prompt(self, background, abbreviations):
        background_info = self.get_background_info(background, abbreviations)
        system_prompt_id = "agent_prompt_screenplay" if self.doc_type == "screenplay" else "agent_prompt_novel"
        return self.prompt_loader.render_prompt(system_prompt_id, {"background_info": background_info})

    def get_background_info(self, background, abbreviations):
        bg_block = f"**Background**: {background}\n" if background else ""

        def fmt(item: dict) -> str:
            if not isinstance(item, dict):
                return ""
            abbr = item.get("abbr") or item.get("full") or next(
                (v for k, v in item.items() if isinstance(v, str) and v.strip()), "N/A"
            )
            parts = []
            for k, v in item.items():
                if k in ("abbr", "full"):
                    continue
                if isinstance(v, str) and v.strip():
                    parts.append(v.strip())
            return f"- **{abbr}**: " + " - ".join(parts) if parts else f"- **{abbr}**"

        abbr_block = "\n".join(fmt(item) for item in abbreviations if isinstance(item, dict))
        return f"{bg_block}\n{abbr_block}" if (background and abbr_block) else (bg_block or abbr_block)

    # ---------------- High-level API (6 methods) ----------------

    def initialize(self, keep_event_cards: bool = True):
        """
        清理旧因果边/投影，创建工作子图 & Louvain；清理输出目录（可保留卡片）。
        """
        for relation_type in ["EVENT_CAUSES", "EVENT_INDIRECT_CAUSES", "EVENT_PART_OF"]:
            self.neo4j_utils.delete_relation_type(relation_type)

        self.neo4j_utils.create_subgraph(
            graph_name="knowledge_graph",
            exclude_entity_types=[self.meta["section_label"]],
            exclude_relation_types=[self.meta["contains_pred"]],
            force_refresh=True,
        )
        self.neo4j_utils.run_louvain(
            graph_name="knowledge_graph", write_property="community", force_run=True
        )

        self._clear_directory(self.config.storage.event_plot_graph_path, keep_event_cards)

    def produce_causality_artifacts(self, limit_events: Optional[int] = None):
        """
        计算 & 落 JSON（事件清单/候选对/因果检测）：
          - sections_sorted.json
          - event2section.json
          - events.jsonl
          - event_cards.json
          - candidate_pairs.jsonl
          - causality_results.jsonl
        """
        base = self.config.storage.event_plot_graph_path
        os.makedirs(base, exist_ok=True)

        # 1) 事件清单
        event_list = self._build_event_list()
        if limit_events and limit_events < len(event_list):
            event_list = event_list[:limit_events]

        sections_meta = [
            {"id": s.id, "name": s.name, "order": int(s.properties.get("order", 99999))}
            for s in self.sorted_sections
        ]
        with open(os.path.join(base, "sections_sorted.json"), "w", encoding="utf-8") as f:
            json.dump(sections_meta, f, ensure_ascii=False, indent=2)
        with open(os.path.join(base, "event2section.json"), "w", encoding="utf-8") as f:
            json.dump(self.event2section_map, f, ensure_ascii=False, indent=2)
        write_jsonl(
            os.path.join(base, "events.jsonl"),
            [{"id": e.id, "name": e.name, "type": e.type, "source_chunks": e.source_chunks} for e in event_list],
        )

        # 2) 事件卡片（缓存）
        cards_path = os.path.join(base, "event_cards.json")
        if os.path.exists(cards_path):
            try:
                self.event_cards = json.load(open(cards_path, "r", encoding="utf-8"))
                if not isinstance(self.event_cards, dict):
                    self.event_cards = {}
            except Exception:
                self.event_cards = {}
        if not self.event_cards:
            self._precompute_event_cards(event_list)
        with open(cards_path, "w", encoding="utf-8") as f:
            json.dump(self.event_cards, f, ensure_ascii=False, indent=2)

        # 3) 候选对（社区/距离/相似度）
        pairs = self._filter_event_pairs_by_community(event_list)
        pairs = self._filter_pair_by_distance_and_similarity(pairs)
        pairs = self._sort_event_pairs_by_section_order(pairs)
        write_jsonl(
            os.path.join(base, "candidate_pairs.jsonl"),
            [{"src": p[0].id, "tgt": p[1].id} for p in pairs],
        )

        # 4) 并行因果检测（结果 JSONL）
        results_map = self._check_causality_batch(pairs)
        write_jsonl(
            os.path.join(base, "causality_results.jsonl"),
            [
                {
                    "src": sid,
                    "tgt": tid,
                    "relation": info.get("relation", "NONE"),
                    "temporal_order": info.get("temporal_order", "Unknown"),
                    "confidence": float(info.get("confidence", 0.0) or 0.0),
                    "reason": info.get("reason", ""),
                    "timeout": bool(info.get("causality_timeout", False)),
                }
                for (sid, tid), info in results_map.items()
            ],
        )

    def materialize_causality_graph(self):
        """
        读取 JSON/JSONL 并将事件-事件因果边写入 Neo4j：
          - 读取 causality_results.jsonl
          - 读取（若存在）saber_removed_edges_round_*.json，产出最终边集
          - 写入 EVENT_* 边
          - 创建 event_causality_graph 投影
        """
        base = self.config.storage.event_plot_graph_path
        results_rows = read_jsonl(os.path.join(base, "causality_results.jsonl"))

        removed: Set[Tuple[str, str]] = set()
        for p in glob.glob(os.path.join(base, "saber_removed_edges_round_*.json")):
            try:
                removed_list = json.load(open(p, "r", encoding="utf-8"))
                for u, v in removed_list:
                    removed.add((u, v))
            except Exception:
                pass

        rows_to_write = []
        for r in results_rows:
            key = (r["src"], r["tgt"])
            if key in removed:
                continue
            rel = (r.get("relation") or "NONE").upper()
            if rel == "NONE":
                continue
            rows_to_write.append(
                {
                    "srcId": r["src"],
                    "dstId": r["tgt"],
                    "confidence": float(r.get("confidence", 0.0) or 0.0),
                    "reason": r.get("reason", ""),
                    "predicate": r.get("relation", "NONE"),
                }
            )

        if rows_to_write:
            self.neo4j_utils.write_event_causes(rows_to_write)
        self.neo4j_utils.create_event_causality_graph("event_causality_graph", force_refresh=True, min_confidence=0.5)




    # def produce_plot_artifacts(self):
    #     """
    #     计算 & 落 JSON（Plot 节点/与事件的边/Plot-Plot关系）：
    #       - filtered_event_chains.json
    #       - plots.jsonl
    #       - plot_has_event_edges.jsonl
    #       - plot_relations.jsonl
    #     仅落盘，不写库。
    #     """
    #     base = self.config.storage.event_plot_graph_path
    #     os.makedirs(base, exist_ok=True)

    #     # 读取/确保 event_cards
    #     cards_path = os.path.join(base, "event_cards.json")
    #     with open(cards_path, "r", encoding="utf-8") as f:
    #         self.event_cards = json.load(f)

    #     # 1) 根据阈值获取事件链
    #     all_chains = self._get_all_event_chains(min_confidence=self.min_edge_confidence)
    #     filtered_chains = segment_trunks_and_branches(
    #         all_chains, min_len=2, include_cutpoint=True, keep_terminal_pairs=True
    #     )
    #     before = len(filtered_chains)
    #     filtered_chains = remove_subset_paths(filtered_chains)
    #     filtered_chains = remove_similar_paths(filtered_chains, threshold=self.chain_similarity_threshold)

    #     with open(os.path.join(base, "filtered_event_chains.json"), "w", encoding="utf-8") as f:
    #         json.dump(filtered_chains, f, ensure_ascii=False, indent=2)

    #     # 2) 生成 Plot 节点（JSONL）与 HAS_EVENT 边（JSONL）
    #     plot_rows: List[Dict[str, Any]] = []
    #     has_edges: List[Dict[str, Any]] = []

    #     def _stable_plot_id(title: str, chain: List[str]) -> str:
    #         key = f"{title}||{'->'.join(chain)}"
    #         return "plot_" + hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]

    #     def _to_bool(v) -> bool:
    #         if isinstance(v, bool):
    #             return v
    #         if v is None:
    #             return False
    #         return str(v).strip().lower() in ("true", "yes", "1")

    #     def process_chain(chain: List[str]):
    #         try:
    #             context = self._prepare_chain_context(chain)
    #             # 汇聚证据上下文
    #             chunk_ids = []
    #             for ent_id in chain:
    #                 ent = self.neo4j_utils.get_entity_by_id(ent_id)
    #                 if not ent:
    #                     continue
    #                 sc = ent.source_chunks or []
    #                 if sc:
    #                     chunk_ids.append(sc[0])
    #             chunk_ids = list(set(chunk_ids))

    #             related_context = ""
    #             if chunk_ids:
    #                 documents = self.vector_store.search_by_ids(chunk_ids)
    #                 contents = {getattr(doc, "content", "") for doc in documents if getattr(doc, "content", "")}
    #                 related_context = "\n".join(list(contents))

    #             try:
    #                 raw = self.graph_analyzer.generate_event_plot(
    #                     event_chain_info=context,
    #                     system_prompt=self.system_prompt_text,
    #                     related_context=related_context,
    #                     timeout=900.0 - 30.0,
    #                 )
    #             except TypeError:
    #                 raw = self.graph_analyzer.generate_event_plot(
    #                     event_chain_info=context,
    #                     system_prompt=self.system_prompt_text,
    #                     related_context=related_context,
    #                 )
    #             result = json.loads(correct_json_format(raw))
    #             if not _to_bool(result.get("is_plot")):
    #                 return False

    #             plot_info = result.get("plot_info") or {}
    #             title = (plot_info.get("title") or "").strip()
    #             if not title:
    #                 title = f"Plot chain: {chain[0]}→{chain[-1]}"

    #             plot_id = _stable_plot_id(title, chain)
    #             plot_rows.append(
    #                 {
    #                     "id": plot_id,
    #                     "title": title,
    #                     "desc": plot_info.get("description", ""),
    #                     "event_ids": chain,
    #                     "reason": result.get("reason", ""),
    #                     "properties": {k: v for k, v in (plot_info.items()) if k not in {"title", "description"}},
    #                 }
    #             )
    #             for eid in chain:
    #                 has_edges.append({"plot_id": plot_id, "event_id": eid})
    #             return True
    #         except Exception:
    #             return None

    #     chain_map, chain_failed = run_with_soft_timeout_and_retries(
    #         filtered_chains,
    #         work_fn=process_chain,
    #         key_fn=lambda ch: tuple(ch),
    #         desc_label="Generate plot artifacts in parallel",
    #         per_task_timeout=600.0,
    #         retries=3,
    #         retry_backoff=60,
    #         allow_placeholder_first_round=False,
    #         placeholder_fn=None,
    #         should_retry=lambda r: r is None,
    #         max_workers=self.max_workers,
    #     )

    #     write_jsonl(os.path.join(base, "plots.jsonl"), plot_rows)
    #     write_jsonl(os.path.join(base, "plot_has_event_edges.jsonl"), has_edges)

    #     # 3) Plot-Plot 关系（只落盘）
    #     plot_pairs = self.neo4j_utils.get_plot_pairs(threshold=0)  # [{"src":..,"tgt":..},...]
    #     DIRECTED = {"PLOT_PREREQUISITE_FOR", "PLOT_ADVANCES", "PLOT_BLOCKS", "PLOT_RESOLVES"}
    #     UNDIRECTED = {"PLOT_CONFLICTS_WITH", "PLOT_PARALLELS"}
    #     VALID = DIRECTED | UNDIRECTED | {"None", None}

    #     def _work(pair: dict) -> dict:
    #         try:
    #             plot_A_info = self.neo4j_utils.get_entity_info(
    #                 pair["src"], "Plot", contain_properties=True, contain_relations=True
    #             )
    #             plot_B_info = self.neo4j_utils.get_entity_info(
    #                 pair["tgt"], "Plot", contain_properties=True, contain_relations=True
    #             )
    #             try:
    #                 result = self.graph_analyzer.extract_plot_relation(
    #                     plot_A_info, plot_B_info, self.system_prompt_text, timeout=PER_TASK_TIMEOUT - 30.0
    #                 )
    #             except TypeError:
    #                 result = self.graph_analyzer.extract_plot_relation(
    #                     plot_A_info, plot_B_info, self.system_prompt_text
    #                 )
    #             try:
    #                 result = json.loads(correct_json_format(result))
    #             except Exception:
    #                 if not isinstance(result, dict):
    #                     raise
    #             rtype = result.get("relation_type")
    #             direction = result.get("direction", None)
    #             confidence = result.get("confidence", 0.0)
    #             reason = result.get("reason", "")
    #             if rtype not in VALID:
    #                 return {"status": "error", "edges": []}
    #             if rtype in {"None", None}:
    #                 return {"status": "none", "edges": []}
    #             if rtype in DIRECTED:
    #                 if direction == "A->B":
    #                     edges = [{"src": pair["src"], "tgt": pair["tgt"], "relation_type": rtype, "confidence": confidence, "reason": reason}]
    #                 elif direction == "B->A":
    #                     edges = [{"src": pair["tgt"], "tgt": pair["src"], "relation_type": rtype, "confidence": confidence, "reason": reason}]
    #                 else:
    #                     return {"status": "error", "edges": []}
    #             else:
    #                 edges = [
    #                     {"src": pair["src"], "tgt": pair["tgt"], "relation_type": rtype, "confidence": confidence, "reason": reason},
    #                     {"src": pair["tgt"], "tgt": pair["src"], "relation_type": rtype, "confidence": confidence, "reason": reason},
    #                 ]
    #             return {"status": "ok", "edges": edges}
    #         except Exception:
    #             return {"status": "error", "edges": []}

    #     res_map, still_failed = run_with_soft_timeout_and_retries(
    #         plot_pairs,
    #         work_fn=_work,
    #         key_fn=lambda p: (p["src"], p["tgt"]),
    #         desc_label="Extract plot relations (artifact phase)",
    #         per_task_timeout=PER_TASK_TIMEOUT,
    #         retries=MAX_RETRIES,
    #         retry_backoff=RETRY_BACKOFF,
    #         allow_placeholder_first_round=False,
    #         placeholder_fn=None,
    #         should_retry=lambda r: (isinstance(r, dict) and r.get("status") == "error"),
    #         max_workers=self.max_workers,
    #     )

    #     rel_edges = []
    #     for out in res_map.values():
    #         if isinstance(out, dict) and out.get("status") == "ok" and out.get("edges"):
    #             rel_edges.extend(out["edges"])
    #     write_jsonl(os.path.join(base, "plot_relations.jsonl"), rel_edges)

    def extract_event_chains(self, min_confidence: Optional[float] = None) -> List[List[str]]:
        """
        读取 event 因果图，筛选/分段/去冗余，保存至 filtered_event_chains.json，并返回链表。
        """
        base = self.config.storage.event_plot_graph_path
        os.makedirs(base, exist_ok=True)

        # 确保 cards 在后续链上下文生成里可用
        cards_path = os.path.join(base, "event_cards.json")
        with open(cards_path, "r", encoding="utf-8") as f:
            self.event_cards = json.load(f)

        min_conf = self.min_edge_confidence if min_confidence is None else float(min_confidence)
        all_chains = self._get_all_event_chains(min_confidence=min_conf)

        filtered_chains = segment_trunks_and_branches(
            all_chains, min_len=2, include_cutpoint=True, keep_terminal_pairs=True
        )
        filtered_chains = remove_subset_paths(filtered_chains)
        filtered_chains = remove_similar_paths(filtered_chains, threshold=self.chain_similarity_threshold)

        with open(os.path.join(base, "filtered_event_chains.json"), "w", encoding="utf-8") as f:
            json.dump(filtered_chains, f, ensure_ascii=False, indent=2)

        return filtered_chains


    def build_and_save_plot_nodes(self, chains: Optional[List[List[str]]] = None) -> Tuple[List[Dict[str,Any]], List[Dict[str,Any]]]:
        """
        给定事件链，调用 LLM 产出 plot_info，保存 Plot 节点与 HAS_EVENT 边到 jsonl 文件。
        返回 (plot_rows, has_edges) 便于上层测试或后续处理。
        """
        base = self.config.storage.event_plot_graph_path
        os.makedirs(base, exist_ok=True)

        # 如未传入，从磁盘读取（便于分步运行）
        if chains is None:
            with open(os.path.join(base, "filtered_event_chains.json"), "r", encoding="utf-8") as f:
                chains = json.load(f)

        plot_rows: List[Dict[str, Any]] = []
        has_edges: List[Dict[str, Any]] = []

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
            try:
                context = self._prepare_chain_context(chain)
                # 汇聚证据上下文（与原版一致）
                chunk_ids = []
                for ent_id in chain:
                    ent = self.neo4j_utils.get_entity_by_id(ent_id)
                    if ent and (ent.source_chunks or []):
                        chunk_ids.append(ent.source_chunks[0])
                chunk_ids = list(set(chunk_ids))
                related_context = ""
                if chunk_ids:
                    documents = self.vector_store.search_by_ids(chunk_ids)
                    contents = {getattr(doc, "content", "") for doc in documents if getattr(doc, "content", "")}
                    related_context = "\n".join(list(contents))

                try:
                    raw = self.graph_analyzer.generate_event_plot(
                        event_chain_info=context,
                        system_prompt=self.system_prompt_text,
                        related_context=related_context,
                        timeout=900.0 - 30.0,
                    )
                except TypeError:
                    raw = self.graph_analyzer.generate_event_plot(
                        event_chain_info=context,
                        system_prompt=self.system_prompt_text,
                        related_context=related_context,
                    )
                result = json.loads(correct_json_format(raw))
                if not _to_bool(result.get("is_plot")):
                    return False

                plot_info = result.get("plot_info") or {}
                title = (plot_info.get("title") or "").strip() or f"Plot chain: {chain[0]}→{chain[-1]}"
                plot_id = _stable_plot_id(title, chain)

                plot_rows.append(
                    {
                        "id": plot_id,
                        "title": title,
                        "desc": plot_info.get("description", ""),
                        "event_ids": chain,
                        "reason": result.get("reason", ""),
                        "properties": {k: v for k, v in (plot_info.items()) if k not in {"title", "description"}},
                    }
                )
                for eid in chain:
                    has_edges.append({"plot_id": plot_id, "event_id": eid})
                return True
            except Exception:
                return None

        # 并行执行，与原实现一致
        chain_map, chain_failed = run_with_soft_timeout_and_retries(
            chains,
            work_fn=process_chain,
            key_fn=lambda ch: tuple(ch),
            desc_label="Generate plot artifacts in parallel",
            per_task_timeout=600.0,
            retries=3,
            retry_backoff=60,
            allow_placeholder_first_round=False,
            placeholder_fn=None,
            should_retry=lambda r: r is None,
            max_workers=self.max_workers,
        )

        write_jsonl(os.path.join(base, "plots.jsonl"), plot_rows)
        write_jsonl(os.path.join(base, "plot_has_event_edges.jsonl"), has_edges)
        return plot_rows, has_edges


    def build_and_save_plot_relations(self) -> List[Dict[str,Any]]:
        """
        基于现有 Plot 节点对，抽取 Plot-Plot 关系，保存至 plot_relations.jsonl。
        返回 edges 列表便于测试或复用。
        """
        base = self.config.storage.event_plot_graph_path
        os.makedirs(base, exist_ok=True)

        plot_pairs = self.neo4j_utils.get_plot_pairs(threshold=0)  # [{"src":..,"tgt":..},...]
        print("[CHECK] Plot pairs to process:", len(plot_pairs))

        DIRECTED = {"PLOT_PREREQUISITE_FOR", "PLOT_ADVANCES", "PLOT_BLOCKS", "PLOT_RESOLVES"}
        UNDIRECTED = {"PLOT_CONFLICTS_WITH", "PLOT_PARALLELS"}
        VALID = DIRECTED | UNDIRECTED | {"None", None}

        def _work(pair: dict) -> dict:
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
                if rtype not in VALID:
                    return {"status": "error", "edges": []}
                if rtype in {"None", None}:
                    return {"status": "none", "edges": []}
                if rtype in DIRECTED:
                    if direction == "A->B":
                        edges = [{"src": pair["src"], "tgt": pair["tgt"], "relation_type": rtype, "confidence": confidence, "reason": reason}]
                    elif direction == "B->A":
                        edges = [{"src": pair["tgt"], "tgt": pair["src"], "relation_type": rtype, "confidence": confidence, "reason": reason}]
                    else:
                        return {"status": "error", "edges": []}
                else:
                    edges = [
                        {"src": pair["src"], "tgt": pair["tgt"], "relation_type": rtype, "confidence": confidence, "reason": reason},
                        {"src": pair["tgt"], "tgt": pair["src"], "relation_type": rtype, "confidence": confidence, "reason": reason},
                    ]
                return {"status": "ok", "edges": edges}
            except Exception:
                return {"status": "error", "edges": []}

        res_map, still_failed = run_with_soft_timeout_and_retries(
            plot_pairs,
            work_fn=_work,
            key_fn=lambda p: (p["src"], p["tgt"]),
            desc_label="Extract plot relations (artifact phase)",
            per_task_timeout=PER_TASK_TIMEOUT,
            retries=MAX_RETRIES,
            retry_backoff=RETRY_BACKOFF,
            allow_placeholder_first_round=False,
            placeholder_fn=None,
            should_retry=lambda r: (isinstance(r, dict) and r.get("status") == "error"),
            max_workers=self.max_workers,
        )

        rel_edges = []
        for out in res_map.values():
            if isinstance(out, dict) and out.get("status") == "ok" and out.get("edges"):
                rel_edges.extend(out["edges"])

        write_jsonl(os.path.join(base, "plot_relations.jsonl"), rel_edges)
        return rel_edges


    
    def materialize_plot_graph(self):
        """
        读取 JSON/JSONL 将 Plot 节点、HAS_EVENT 边、Plot-Plot 关系写入 Neo4j：
        - plots.jsonl
        - plot_has_event_edges.jsonl
        - plot_relations.jsonl
        这里会将 JSON 规范化为 create_plot_node 期望的结构：
        id, name(=title), description(=summary)，以及常见字段提升为顶层；
        properties 作为 map 传入（不转字符串）。
        """
        base = self.config.storage.event_plot_graph_path
        self.neo4j_utils.reset_event_plot_graph()

        # 读取产物
        plot_nodes = read_jsonl(os.path.join(base, "plots.jsonl"))
        has_edges  = read_jsonl(os.path.join(base, "plot_has_event_edges.jsonl"))
        plot_rel_edges = read_jsonl(os.path.join(base, "plot_relations.jsonl"))

        # ---------- 写 Plot 节点 ----------
        for pn in plot_nodes:
            props = pn.get("properties", {}) or {}

            # 统一 name / description（summary）映射与兜底
            name = pn.get("name") or pn.get("title", "")
            description = (
                pn.get("summary")
                or pn.get("description")
                or pn.get("desc")
                or props.get("summary")
                or ""
            )

            # 常见语义字段：优先 properties，其次顶层
            def pick(key):
                return props.get(key, pn.get(key))

            plot_data = {
                "id": pn["id"],
                "name": name,
                "title": name,                       # create_plot_node 期望字段    
                "description": description,               # create_plot_node 期望字段
                "reason": pn.get("reason", ""),
                "event_ids": list(dict.fromkeys(pn.get("event_ids", []) or [])),
                "main_characters": pick("main_characters"),
                "locations": pick("locations"),
                "time": pick("time"),
                "theme": pick("theme"),
                "goal": pick("goal"),
                "conflict": pick("conflict"),
                "resolution": pick("resolution"),
                "properties": props,                      # 作为 map 传入，便于 Cypher 展开
            }

            # 首选你的专用接口（更贴合你的数据模型）
            created = False
            if hasattr(self.neo4j_utils, "create_plot_node"):
                try:
                    created = bool(self.neo4j_utils.create_plot_node(plot_data))
                except Exception as e:
                    print(f"创建 Plot 节点失败(create_plot_node): {e}")

            # 退化方案：用通用 write_plot_to_neo4j（如果存在）
            if not created and hasattr(self.neo4j_utils, "write_plot_to_neo4j"):
                try:
                    # 该接口可能期待 title/summary/description；准备一个兼容 payload
                    fallback_payload = {
                        "id": plot_data["id"],
                        "title": plot_data["name"],
                        "summary": plot_data["description"],
                        "description": plot_data["description"],
                        "reason": plot_data["reason"],
                        "event_ids": plot_data["event_ids"],
                        "properties": plot_data["properties"],
                    }
                    self.neo4j_utils.write_plot_to_neo4j(fallback_payload)
                except Exception as e:
                    print(f"创建 Plot 节点失败(write_plot_to_neo4j): {e}")

        # ---------- 写 HAS_EVENT ----------
        if has_edges:
            # 优先批量接口（如果你按我前面建议添加了）
            if hasattr(self.neo4j_utils, "create_plot_has_event_edges"):
                try:
                    self.neo4j_utils.create_plot_has_event_edges(has_edges)
                except Exception as e:
                    print(f"批量 HAS_EVENT 写入失败: {e}")
            else:
                # 回退：按 plot_id 聚合，再用 create_plot_event_relationships
                pid2eids = {}
                for e in has_edges:
                    pid = e.get("plot_id") or e.get("src") or e.get("from") or e.get("source")
                    eid = e.get("event_id") or e.get("tgt") or e.get("to") or e.get("target")
                    if not pid or not eid:
                        continue
                    pid2eids.setdefault(pid, set()).add(eid)
                if hasattr(self.neo4j_utils, "create_plot_event_relationships"):
                    for pid, eids in pid2eids.items():
                        try:
                            self.neo4j_utils.create_plot_event_relationships(pid, list(eids))
                        except Exception as e:
                            print(f"写入 HAS_EVENT (plot={pid}) 失败: {e}")
                else:
                    # 最保守：逐条 merge（如你没有上述接口）
                    for e in has_edges:
                        pid = e.get("plot_id") or e.get("src") or e.get("from") or e.get("source")
                        eid = e.get("event_id") or e.get("tgt") or e.get("to") or e.get("target")
                        if not pid or not eid:
                            continue
                        try:
                            cypher = """
                            MATCH (p:Plot {id: $pid}), (ev:Event {id: $eid})
                            MERGE (p)-[:HAS_EVENT]->(ev)
                            """
                            self.neo4j_utils.execute_query(cypher, {"pid": pid, "eid": eid})
                        except Exception as ex:
                            print(f"逐条写入 HAS_EVENT 失败: plot={pid}, event={eid}, err={ex}")

        # ---------- 写 Plot-Plot 关系 ----------
        if plot_rel_edges:
            if hasattr(self.neo4j_utils, "create_plot_relations"):
                try:
                    self.neo4j_utils.create_plot_relations(plot_rel_edges)
                except Exception as e:
                    print(f"写入 Plot-Plot 关系失败: {e}")
            else:
                # 兜底：选择一套最常见的关系类型写入
                for r in plot_rel_edges:
                    try:
                        src = r.get("src"); tgt = r.get("tgt")
                        rtype = r.get("relation_type") or "PLOT_RELATES"
                        conf = float(r.get("confidence", 0.0) or 0.0)
                        reason = r.get("reason", "")
                        cypher = f"""
                        MATCH (a:Plot {{id: $src}}), (b:Plot {{id: $tgt}})
                        MERGE (a)-[rel:{rtype}]->(b)
                        SET rel.confidence = $conf, rel.reason = $reason
                        """
                        self.neo4j_utils.execute_query(cypher, {"src": src, "tgt": tgt, "conf": conf, "reason": reason})
                    except Exception as ex:
                        print(f"逐条写入 Plot-Plot 关系失败: {ex}")

        # ---------- 生成 Plot 图投影/embedding ----------
        self.neo4j_utils.create_event_plot_graph()
        self.neo4j_utils.run_node2vec()


    def prepare_graph_embeddings(self):
        """
        构建/更新图嵌入与向量索引（事件/Plot）。
        """
        self.neo4j_utils.load_embedding_model(self.config.graph_embedding)
        self.neo4j_utils.create_vector_index()
        self.neo4j_utils.process_all_embeddings(entity_types=["Event", "Plot"])
        self.neo4j_utils.ensure_entity_superlabel()
        print("✅ Vector construction for event/plot graph completed")

    # ---------------- Internals (原子能力，供上面 6 方法调用) ----------------

    def _clear_directory(self, path, keep_event_cards=False):
        for file in glob.glob(os.path.join(path, "*.json")):
            try:
                if keep_event_cards and os.path.basename(file) == "event_cards.json":
                    continue
                os.remove(file)
            except Exception as e:
                print(f"Failed to delete: {file} -> {e}")

    def _build_event_list(self) -> List[Entity]:
        """
        从章节的 contains 关系里抽 Event 列表。
        """
        section_entities = self.neo4j_utils.search_entities_by_type(entity_type=self.meta["section_label"])
        self.sorted_sections = sorted(section_entities, key=lambda e: int(e.properties.get("order", 99999)))

        event_list: List[Entity] = []
        event2section_map: Dict[str, str] = {}
        for section in tqdm(self.sorted_sections, desc="Extracting events from sections"):
            results = self.neo4j_utils.search_related_entities(
                source_id=section.id,
                predicate=self.meta["contains_pred"],
                entity_types=["Event"],
                return_relations=False,
            )
            if not results and self.event_fallback:
                results = self.neo4j_utils.search_related_entities(
                    source_id=section.id,
                    relation_types=[self.meta["contains_pred"]],
                    entity_types=self.event_fallback,
                    return_relations=False,
                )
            for result in results:
                if result.id not in event2section_map:
                    event2section_map[result.id] = section.id
                    event_list.append(result)

        self.event_list = event_list
        self.event2section_map = event2section_map
        return event_list

    def _precompute_event_cards(self, events: List[Entity]) -> Dict[str, Dict[str, Any]]:
        """
        并发生成/缓存事件卡片，落到 event_cards.json。
        """
        base = self.config.storage.event_plot_graph_path
        os.makedirs(base, exist_ok=True)
        cache_path = os.path.join(base, "event_cards.json")

        # 读缓存
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

        def _collect_related_context_by_section(ev: Entity) -> str:
            ctx_set = set()
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

            full_ctx = "\n".join(ctx_set)
            goal = f"Please focus on information relevant to '{ev.name}'."
            if len(full_ctx) >= 2000:
                full_ctx_splitted = self.base_splitter.split_text(full_ctx)
                summaries = []
                for chunk in full_ctx_splitted:
                    chunk_result = self.document_parser.summarize_paragraph(chunk, 100, "", goal)
                    parsed = json.loads(correct_json_format(chunk_result)).get("summary", [])
                    if isinstance(parsed, list):
                        summaries.extend(parsed)
                full_ctx = "\n".join(summaries) if summaries else full_ctx
            return full_ctx

        def _build_one(ev: Entity) -> str:
            info = self.neo4j_utils.get_entity_info(ev.id, "Event", True, True)
            related_ctx = _collect_related_context_by_section(ev)
            try:
                out = self.graph_analyzer.generate_event_context(info, related_ctx, timeout=min(PER_TASK_TIMEOUT - 5.0, 1200.0))
            except TypeError:
                out = self.graph_analyzer.generate_event_context(info, related_ctx)
            card = json.loads(correct_json_format(out))["event_card"]
            card = format_event_card(card)
            return card

        def _placeholder(ev: Entity, exc=None) -> str:
            return _collect_related_context_by_section(ev)

        res_map, still_failed = run_with_soft_timeout_and_retries(
            pending_events,
            work_fn=_build_one,
            key_fn=lambda e: e.id,
            desc_label="Precompute event cards",
            per_task_timeout=600.0,
            retries=5,
            retry_backoff=30,
            allow_placeholder_first_round=True,
            placeholder_fn=_placeholder,
            should_retry=None,
            max_workers=self.max_workers,
        )
        for k, v in res_map.items():
            self.event_cards[k] = v
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(self.event_cards, f, ensure_ascii=False, indent=2)
        return self.event_cards

    def _filter_event_pairs_by_community(self, events: List[Entity]) -> List[Tuple[Entity, Entity]]:
        id2entity = {e.id: e for e in events}
        pairs = self.neo4j_utils.fetch_event_pairs_same_community()
        filtered_pairs = []
        for row in pairs:
            src_id, dst_id = row["srcId"], row["dstId"]
            if src_id in id2entity and dst_id in id2entity:
                filtered_pairs.append((id2entity[src_id], id2entity[dst_id]))
        return filtered_pairs

    def _sort_event_pairs_by_section_order(self, pairs: List[Tuple[Entity, Entity]]) -> List[Tuple[Entity, Entity]]:
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

    def _filter_pair_by_distance_and_similarity(self, pairs):
        filtered_pairs = []
        for pair in tqdm(pairs, desc="Filter node pairs"):
            src_id, tgt_id = pair[0].id, pair[1].id
            reachable = self.neo4j_utils.check_nodes_reachable(
                src_id,
                tgt_id,
                excluded_rels=[self.meta["contains_pred"], "EVENT_CAUSES", "EVENT_INDIRECT_CAUSES", "EVENT_PART_OF"],
                max_depth=self.max_depth,
            )
            if reachable:
                filtered_pairs.append(pair)
            else:
                score = self.neo4j_utils.compute_semantic_similarity(src_id, tgt_id)
                if score is not None and score >= 0.7:
                    filtered_pairs.append(pair)
        return filtered_pairs

    def _check_causality_batch(self, pairs: List[Tuple[Entity, Entity]]) -> Dict[Tuple[str, str], Dict[str, Any]]:
        PT = 1800
        def _make_result(src_event, tgt_event, relation="NONE", reason="", temporal_order="Unknown", confidence=0.0, raw_result="", timeout=False):
            res = {
                "src_event": src_event,
                "tgt_event": tgt_event,
                "relation": relation,
                "reason": reason,
                "temporal_order": temporal_order,
                "confidence": float(confidence) if confidence is not None else 0.0,
                "raw_result": raw_result,
            }
            if timeout:
                res["causality_timeout"] = True
            return res

        def _get_common_neighbor_info(src_id, tgt_id):
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
            if e.id in self.event_cards:
                return self.event_cards[e.id]
            try:
                out = self.graph_analyzer.generate_event_context(info_text, "", timeout=PT - 60)
            except TypeError:
                out = self.graph_analyzer.generate_event_context(info_text, "")
            card = json.loads(correct_json_format(out))["event_card"]
            card = format_event_card(card)
            self.event_cards[e.id] = card
            return card

        def _work(pair: Tuple[Entity, Entity]) -> Dict[str, Any]:
            src_event, tgt_event = pair
            try:
                info_1 = self.neo4j_utils.get_entity_info(src_event.id, "Event", True, True)
                info_2 = self.neo4j_utils.get_entity_info(tgt_event.id, "Event", True, True)
                related_context = info_1 + "\n" + info_2 + "\n" + _get_common_neighbor_info(src_event.id, tgt_event.id)
                src_card = _ensure_card(src_event, info_1)
                tgt_card = _ensure_card(tgt_event, info_2)
                try:
                    result_json = self.graph_analyzer.check_event_causality(
                        src_card, tgt_card, system_prompt=self.system_prompt_text, related_context=related_context, timeout=max(5.0, PT - 60.0)
                    )
                except TypeError:
                    result_json = self.graph_analyzer.check_event_causality(
                        src_card, tgt_card, system_prompt=self.system_prompt_text, related_context=related_context
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
                return _make_result(src_event, tgt_event, relation, reason, temporal_order, confidence, raw_str, timeout=False)
            except Exception as e:
                return _make_result(src_event, tgt_event, relation="NONE", reason=f"Error during check: {e}", timeout=True)

        def _placeholder(pair: Tuple[Entity, Entity], exc=None) -> Dict[str, Any]:
            src_event, tgt_event = pair
            return _make_result(src_event, tgt_event, relation="NONE", reason="Soft timeout/exception; placeholder", timeout=True)

        def _should_retry(res: Dict[str, Any]) -> bool:
            if res.get("causality_timeout"):
                return True
            reason = (res.get("reason") or "").strip()
            return ("Error" in reason)

        res_map, still_failed = run_with_soft_timeout_and_retries(
            pairs,
            work_fn=_work,
            key_fn=lambda p: (p[0].id, p[1].id),
            desc_label="Parallel causality check",
            per_task_timeout=PT,
            retries=MAX_RETRIES,
            retry_backoff=RETRY_BACKOFF,
            allow_placeholder_first_round=True,
            placeholder_fn=_placeholder,
            should_retry=_should_retry,
            max_workers=self.max_workers,
        )
        for k in still_failed:
            if k in res_map:
                res_map[k]["final_fallback"] = True
                res_map[k]["retries"] = MAX_RETRIES
        return res_map

    def _get_all_event_chains(self, min_confidence: float = 0.0):
        starting_events = self.neo4j_utils.get_starting_events()
        chains = []
        for event in starting_events:
            all_chains = self.neo4j_utils.find_event_chain(event, min_confidence)
            chains.extend([chain for chain in all_chains if len(chain) >= 2])
        return chains

    def _prepare_chain_context(self, chain: List[str]) -> str:
        if len(chain) > 1:
            context = "Event chain: " + "->".join(chain) + "\n\nDetailed event information:\n"
        else:
            context = f"Event: {chain[0]}" + "\n\nDetailed event information:\n"
        for i, event in enumerate(chain):
            context += f"Event {i+1}: {event}\n" + self.event_cards.get(event, "") + "\n"
        return context

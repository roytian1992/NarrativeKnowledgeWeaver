# kag/builder/kg_builder_2.py
from __future__ import annotations

import json
import os
import sqlite3
import pickle
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed, TimeoutError, wait
from collections import defaultdict
from copy import deepcopy
from typing import Any, Dict, List, Optional
import asyncio
import pandas as pd
from tqdm import tqdm
from kag.utils.prompt_loader import PromptLoader
from kag.models.data import Entity, KnowledgeGraph, Relation, TextChunk, Document
from ..storage.document_store import DocumentStore
from ..storage.graph_store import GraphStore
from ..storage.vector_store import VectorStore
from ..utils.config import KAGConfig
from ..utils.neo4j_utils import Neo4jUtils
from kag.llm.llm_manager import LLMManager
from kag.agent.kg_extraction_agent import InformationExtractionAgent
from kag.agent.attribute_extraction_agent import AttributeExtractionAgent
from .document_processor import DocumentProcessor
from kag.builder.graph_preprocessor import GraphPreprocessor

# ──────────────────────────────────────────────────────────────────────────────
# doc-type ↔︎ 元字段 / 标签 / 谓词映射
# ──────────────────────────────────────────────────────────────────────────────
DOC_TYPE_META: Dict[str, Dict[str, str]] = {
    "screenplay": {
        "section_label": "Scene",
        "title": "scene_name",
        "subtitle": "sub_scene_name",
        "contains_pred": "SCENE_CONTAINS",
    },
    "novel": {
        "section_label": "Chapter",
        "title": "chapter_name",
        "subtitle": "sub_chapter_name",
        "contains_pred": "CHAPTER_CONTAINS",
    },
}

# ═════════════════════════════════════════════════════════════════════════════
#                               Builder
# ═════════════════════════════════════════════════════════════════════════════
class KnowledgeGraphBuilder:
    """知识图谱构建器（支持多文档格式）"""
    def __init__(self, config: KAGConfig, doc_type: str = "screenplay", background_path: str = ""):
        if doc_type not in DOC_TYPE_META:
            raise ValueError(f"Unsupported doc_type: {doc_type}")
        self.max_workers = 32
        self.multi_mode = "async"
        
        self.config = config
        self.meta = DOC_TYPE_META[doc_type]
        prompt_dir = (
            config.prompt_dir
            if hasattr(config, "prompt_dir")
            else os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "kag/prompts"
            )
        )
        self.prompt_loader = PromptLoader(prompt_dir)
            
        # LLM & Processor
        self.llm_manager = LLMManager(config)
        self.llm = self.llm_manager.get_llm()
        self.processor = DocumentProcessor(config, self.llm, doc_type, max_worker=self.max_workers)

        # 存储 / 数据库
        self.graph_store = GraphStore(config)
        self.neo4j_utils = Neo4jUtils(self.graph_store.driver, doc_type=doc_type)
        self.vector_store = VectorStore(config)
        self.document_store = DocumentStore(config)

        # 运行数据
        self.kg = KnowledgeGraph()

        # 可选 schema / 缩写
        self._load_schema("kag/schema/graph_schema.json")
        self.background_info = ""
        if background_path:
            print("📖加载背景信息")
            # glossary_path = os.path.join("kag/schema", glossary, "settings_schema.json")
            self._load_settings(background_path)
        
        if doc_type == "screenplay":
            system_prompt_id = "agent_prompt_screenplay"
        else:
            system_prompt_id = "agent_prompt_novel"
            
        self.system_prompt_text = self.prompt_loader.render_prompt(system_prompt_id, {"background_info": self.background_info})
        
        # 抽取 agent
        self.information_extraction_agent = InformationExtractionAgent(config, self.llm, self.system_prompt_text)
        self.attribute_extraction_agent = AttributeExtractionAgent(config, self.llm, self.system_prompt_text)
        self.graph_preprocessor = GraphPreprocessor(config, self.llm, system_prompt=self.system_prompt_text)

    def _load_settings(self, path: str):
        """
        读取 background + abbreviations，并将其合并到 self.abbreviation_info（一段 Markdown 文本）。

        JSON 结构示例（字段均可选）：
        {
            "background": "……",
            "abbreviations": [
                { "abbr": "UEG", "full": "United Earth Government", "zh": "联合政府", "description": "全球统一政府。" },
                { "symbol": "AI", "meaning": "人工智能", "comment": "广泛应用于…" }
            ]
        }
        """
        self.background_info = ""

        if not os.path.exists(path):
            return

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # ---------- 1) 背景段落（可选） ----------
        background = data.get("background", "").strip()
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

        abbr_list = data.get("abbreviations", [])
        abbr_block = "\n".join(fmt(item) for item in abbr_list if isinstance(item, dict))

        if background and abbr_block:
            self.background_info = f"{bg_block}\n{abbr_block}"
        else:
            self.background_info = bg_block or abbr_block
        print(f"✅ 成功从{path}加载背景信息")

    def _load_schema(self, path: str):
        if not os.path.exists(path):
            self.entity_types, self.relation_type_groups = [], {}
            return
        sch = json.load(open(path, "r", encoding="utf-8"))
        self.entity_types = sch.get("entities", [])
        self.relation_type_groups = sch.get("relations", {})


    def prepare_chunks(self, json_file_path: str, verbose: bool = True):
        if verbose:
            print(f"🚀 开始构建知识图谱: {json_file_path}")

        if verbose:
            print("📖 加载文档...")
        
        documents = self.processor.load_from_json(json_file_path, extract_metadata=True)
        
        if verbose:
            print(f"✅ 成功加载 {len(documents)} 个文档")

        # 并发切块
        all_docs = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as exe:
            futs = [exe.submit(self.processor.prepare_chunk, d) for d in documents]
            for fut in tqdm(as_completed(futs), total=len(futs), desc="并发拆分中"):
                grp = fut.result()
                all_docs.extend(grp["document_chunks"])

        # 落盘
        base = self.config.storage.knowledge_graph_path
        os.makedirs(base, exist_ok=True)
        json.dump([c.dict() for c in all_docs],
                  open(os.path.join(base, "all_document_chunks.json"), "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)

        if verbose:
            print(f"✅ 生成 {len(all_docs)} 个文本块")
            
    # def prepare_chunks(
    #     self,
    #     json_file_path: str,
    #     verbose: bool = True,
    #     per_task_timeout: int = 120,
    #     max_workers: int = None
    # ):
    #     max_workers = max_workers or self.max_workers
    #     if verbose:
    #         print(f"🚀 开始构建知识图谱: {json_file_path}")
    #         print("📖 加载文档...")
    #     documents = self.processor.load_from_json(json_file_path, extract_metadata=True)
    #     if verbose:
    #         print(f"✅ 成功加载 {len(documents)} 个文档")

    #     # 1) 并发尝试
    #     all_docs = []
    #     timed_out_docs = []
    #     failed_docs = []

    #     with ThreadPoolExecutor(max_workers=max_workers) as exe:
    #         futures = {exe.submit(self.processor.prepare_chunk, doc): doc for doc in documents}

    #         # 等待全部完成或单个超时，但不阻塞到永久
    #         done, not_done = wait(futures, timeout=None)  # 不设置 overall timeout

    #         # 收集已完成
    #         for fut in tqdm(done, total=len(done), desc="并发拆分处理中"):
    #             doc = futures[fut]
    #             try:
    #                 grp = fut.result(timeout=per_task_timeout)
    #                 all_docs.extend(grp["document_chunks"])
    #             except TimeoutError:
    #                 if verbose:
    #                     print(f"⚠️ 文档 {getattr(doc,'id',None)} 并发超时，稍后回退同步切分")
    #                 timed_out_docs.append(doc)
    #             except Exception as e:
    #                 if verbose:
    #                     print(f"❌ 文档 {getattr(doc,'id',None)} 并发失败：{e}，稍后回退同步切分")
    #                 failed_docs.append(doc)

    #         # 剩下没 done 的，也当作超时
    #         for fut in not_done:
    #             doc = futures[fut]
    #             if verbose:
    #                 print(f"⚠️ 文档 {getattr(doc,'id',None)} 未完成，稍后回退同步切分")
    #             timed_out_docs.append(doc)

    #     # 2) 同步保底：对所有超时／失败文档逐个切分（无超时限制）
    #     fallback = timed_out_docs + failed_docs
    #     if fallback and verbose:
    #         print(f"🔄 开始同步保底切分 {len(fallback)} 个文档（无超时限制）")
    #     for doc in tqdm(fallback, desc="同步保底切分中"):
    #         try:
    #             grp = self.processor.prepare_chunk(doc)
    #             all_docs.extend(grp["document_chunks"])
    #         except Exception as e:
    #             # 真正卡住或其他异常，这里抛出让你看到具体是哪个文档
    #             raise RuntimeError(f"文档 {getattr(doc,'id',None)} 同步保底切分失败: {e}")

    #     # 3) 落盘
    #     base = self.config.storage.knowledge_graph_path
    #     os.makedirs(base, exist_ok=True)
    #     out_file = os.path.join(base, "all_document_chunks.json")
    #     with open(out_file, "w", encoding="utf-8") as f:
    #         json.dump([c.dict() for c in all_docs], f, ensure_ascii=False, indent=2)

    #     if verbose:
    #         print(f"✅ 共生成 {len(all_docs)} 个文本块，保存在 {out_file}")
        
    # ═════════════════════════════════════════════════════════════════════
    #  2) 存储 Chunk（RDB + VDB）
    # ═════════════════════════════════════════════════════════════════════
    def store_chunks(self, verbose: bool = True):
        base = self.config.storage.knowledge_graph_path

        # 描述块
        doc_chunks = [TextChunk(**o) for o in
                       json.load(open(os.path.join(base, "all_document_chunks.json"), "r", encoding="utf-8"))]

        # 写入 KG（Document + Chunk）
        for ch in doc_chunks:
            self.kg.add_document(self.processor.prepare_document(ch))
            self.kg.add_chunk(ch)

        # 写入向量数据库
        if verbose:
            print("💾 存储到向量数据库...")
        self._store_vectordb(verbose)

    # ═════════════════════════════════════════════════════════════════════
    #  3) 实体 / 关系 抽取
    # ═════════════════════════════════════════════════════════════════════
    def extract_entity_and_relation(self, verbose: bool = True):
        if self.multi_mode == "async":
            asyncio.run(self.extract_entity_and_relation_async(verbose=verbose))
        else:
            self.extract_entity_and_relation_threaded(verbose)
            
                
    def extract_entity_and_relation_threaded(self, verbose: bool = True):
   
        base = self.config.storage.knowledge_graph_path
        desc_chunks = [TextChunk(**o) for o in
                    json.load(open(os.path.join(base, "all_document_chunks.json"), "r", encoding="utf-8"))]

        if verbose:
            print("🧠 实体与关系信息抽取中...")

        def _run(ch: TextChunk):
            try:
                if not ch.content.strip():
                    result = {"entities": [], "relations": []}
                else:
                    result = self.information_extraction_agent.run(ch.content)
                result.update(chunk_id=ch.id, chunk_metadata=ch.metadata)
                return result
            except Exception as e:
                return {
                    "chunk_id": ch.id,
                    "chunk_metadata": ch.metadata,
                    "entities": [],
                    "relations": [],
                    "error": str(e)
                }

        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(_run, ch) for ch in desc_chunks]
            for fut in tqdm(as_completed(futures), total=len(futures), desc="并发抽取中"):
                results.append(fut.result())  # 谁先完成谁就加入列表

        # ✅ 最后统一写入
        output_path = os.path.join(base, "extraction_results.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        if verbose:
            print(f"✅ 实体与关系信息抽取完成，共处理 {len(results)} 个文本块")
            print(f"💾 结果已保存至 {output_path}")
            
    
    async def extract_entity_and_relation_async(self, verbose: bool = True):
        """
        使用 asyncio 并发执行 .arun()，并统一写入结果到 extraction_results.json
        """
        base = self.config.storage.knowledge_graph_path
        desc_chunks = [TextChunk(**o) for o in
                    json.load(open(os.path.join(base, "all_document_chunks.json"), "r", encoding="utf-8"))]

        if verbose:
            print("🧠 实体与关系信息异步抽取中...")

        sem = asyncio.Semaphore(self.max_workers)

        async def _arun(ch: TextChunk):
            async with sem:
                try:
                    if not ch.content.strip():
                        result = {"entities": [], "relations": []}
                    else:
                        result = await self.information_extraction_agent.arun(ch.content)
                    result.update(chunk_id=ch.id, chunk_metadata=ch.metadata)
                    return result
                except Exception as e:
                    return {
                        "chunk_id": ch.id,
                        "chunk_metadata": ch.metadata,
                        "entities": [],
                        "relations": [],
                        "error": str(e)
                    }

        tasks = [_arun(ch) for ch in desc_chunks]
        results = []
        for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="异步抽取中"):
            res = await coro
            results.append(res)

        output_path = os.path.join(base, "extraction_results.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        if verbose:
            print(f"✅ 实体与关系信息抽取完成，共处理 {len(results)} 个文本块")
            print(f"💾 已保存至：{output_path}")

    # ═════════════════════════════════════════════════════════════════════
    #  4) 属性抽取
    # ═════════════════════════════════════════════════════════════════════
    
    def extract_entity_attributes(self, verbose: bool = True) -> Dict[str, Entity]:
        if self.multi_mode == "async":
            return asyncio.run(self.extract_entity_attributes_async(verbose=verbose))
        else:
            return self._extract_entity_attributes_threaded(verbose=verbose)

    
    async def extract_entity_attributes_async(self, verbose: bool = True) -> Dict[str, Entity]:
        """
        ⚡ 异步批量属性抽取  
        · 按 extract_entity_and_relation_async 生成的 entity_map 去并发  
        · 每个实体调用 attribute_extraction_agent.arun()  
        · 内部 arun 已带超时＋重试保护，不会卡死
        """
        base = self.config.storage.knowledge_graph_path
        results = json.load(open(os.path.join(base, "extraction_results.json"), "r", encoding="utf-8"))
        
        #print(results[0])
        # 将实体合并 / 去重
        entity_map = self.merge_entities_info(results)            # {name: Entity}

        if verbose:
            print(f"🔎 开始属性抽取（异步），实体数：{len(entity_map)}")

        sem = asyncio.Semaphore(self.max_workers)
        updated_entities: Dict[str, Entity] = {}

        async def _arun_attr(name: str, ent: Entity):
            async with sem:
                try:
                    txt = ent.description or ""
                    if not txt.strip():
                        return name, None

                    # AttributeExtractionAgent.arun 已自带 timeout+重试
                    res = await self.attribute_extraction_agent.arun(
                        text=txt,
                        entity_name=name,
                        entity_type=ent.type,
                        source_chunks=ent.source_chunks,
                        original_text=""
                    )

                    if res.get("error"):          # 超时或异常
                        return name, None

                    attrs = res.get("attributes", {}) or {}
                    if isinstance(attrs, str):
                        try:
                            attrs = json.loads(attrs)
                        except json.JSONDecodeError:
                            attrs = {}

                    new_ent = deepcopy(ent)
                    new_ent.properties = attrs

                    nd = res.get("new_description", "")
                    if nd:
                        new_ent.description = nd

                    return name, new_ent
                except Exception as e:
                    if verbose:
                        print(f"[ERROR] 属性抽取失败（异步）：{name}: {e}")
                    return name, None

        # 并发执行
        tasks = [_arun_attr(n, e) for n, e in entity_map.items()]
        for coro in tqdm(asyncio.as_completed(tasks),
                               total=len(tasks),
                               desc="属性抽取中（async）"):
            n, e2 = await coro
            if e2:
                updated_entities[n] = e2

        # 写文件
        output_path = os.path.join(base, "entity_info.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({k: v.dict() for k, v in updated_entities.items()},
                      f, ensure_ascii=False, indent=2)

        if verbose:
            print(f"✅ 属性抽取完成，共处理实体 {len(updated_entities)} 个")
            print(f"💾 已保存至：{output_path}")

        return updated_entities
    
    
    def _extract_entity_attributes_threaded(self, verbose: bool = True) -> Dict[str, Entity]:
        base = self.config.storage.knowledge_graph_path
        results = json.load(open(os.path.join(base, "extraction_results.json"), "r", encoding="utf-8"))
        
        entity_map = self.merge_entities_info(results)

        if verbose:
            print("🔎 属性抽取中（线程）...")

        def _run_attr(name: str, ent: Entity):
            txt = ent.description or ""
            if not txt.strip():
                return name, None
            try:
                res = self.attribute_extraction_agent.run(
                    text=txt,
                    entity_name=name,
                    entity_type=ent.type,
                    source_chunks=ent.source_chunks,
                    original_text=""
                )
                return self._postprocess_attribute(name, ent, res)
            except Exception as e:
                if verbose:
                    print(f"[ERROR] 属性抽取失败（同步）：{name}: {e}")
                return name, None

        updated: Dict[str, Entity] = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as exe:
            futs = [exe.submit(_run_attr, n, e) for n, e in entity_map.items()]
            for fut in tqdm(as_completed(futs), total=len(futs), desc="属性抽取中（线程）"):
                name, ent2 = fut.result()
                if ent2:
                    updated[name] = ent2

        output_path = os.path.join(base, "entity_info.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({k: v.dict() for k, v in updated.items()}, f, ensure_ascii=False, indent=2)

        if verbose:
            print(f"✅ 属性抽取完成，共处理实体 {len(updated)} 个")
            print(f"💾 已保存至：{output_path}")

        return updated

    
    def _extract_entity_attributes_threaded(self, verbose: bool = True) -> Dict[str, Entity]:
        base = self.config.storage.knowledge_graph_path
        results = json.load(open(os.path.join(base, "extraction_results.json"), "r", encoding="utf-8"))

        entity_map = self._merge_entities(results)

        if verbose:
            print("🔎 属性抽取中（线程）...")

        def _run_attr(name: str, ent: Entity):
            txt = ent.description or ""
            if not txt.strip():
                return name, None
            try:
                res = self.attribute_extraction_agent.run(
                    text=txt,
                    entity_name=name,
                    entity_type=ent.type,
                    source_chunks=ent.source_chunks,
                    original_text=""
                )
                return self._postprocess_attribute(name, ent, res)
            except Exception as e:
                if verbose:
                    print(f"[ERROR] 属性抽取失败（同步）：{name}: {e}")
                return name, None

        updated: Dict[str, Entity] = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as exe:
            futs = [exe.submit(_run_attr, n, e) for n, e in entity_map.items()]
            for fut in tqdm(as_completed(futs), total=len(futs), desc="属性抽取中（线程）"):
                name, ent2 = fut.result()
                if ent2:
                    updated[name] = ent2

        output_path = os.path.join(base, "entity_info.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({k: v.dict() for k, v in updated.items()}, f, ensure_ascii=False, indent=2)

        if verbose:
            print(f"✅ 属性抽取完成，共处理实体 {len(updated)} 个")
            print(f"💾 已保存至：{output_path}")

        return updated

    # ═════════════════════════════════════════════════════════════════════
    #  5) 构建并存储图谱
    # ═════════════════════════════════════════════════════════════════════
    def build_graph_from_results(self, verbose: bool = True) -> KnowledgeGraph:
        if verbose:
            print("📂 加载已有抽取结果和实体信息...")

        base = self.config.storage.knowledge_graph_path
        results = json.load(open(os.path.join(base, "extraction_results.json"), "r", encoding="utf-8"))
        ent_raw = json.load(open(os.path.join(base, "entity_info.json"), "r", encoding="utf-8"))
        
        with open(os.path.join(base, "section_entities_collection.pkl"), "rb") as f:
            self.section_entities_collection = pickle.load(f)
            
       #  self.section_entities_collection = json.load(open(os.path.join(base, "section_entities_collection.json"), "r", encoding="utf-8"))
        
        # id → Entity
        entity_map = {d["id"]: Entity(**d) for d in ent_raw.values()}
        name2id: Dict[str, str] = {e.name: e.id for e in entity_map.values()}
        
        for e in entity_map.values():
            for al in e.aliases:
                name2id.setdefault(al, e.id)
            self.kg.add_entity(e)

        if verbose:
            print("🔗 构建知识图谱...")

        self.section_names = []
        for res in results:
            md = res.get("chunk_metadata", {})
            # Section 实体
            secs = self._create_section_entities(md, res["chunk_id"])
            for se in secs:
                # if se.id not in self.kg.entities:
                if se.name not in self.section_names and se.id not in self.kg.entities:
                    self.kg.add_entity(se)
                    self.section_names.append(se.name)

            # Section contains inner entities
            # inner = [entity_map[name2id[e["name"]]]
            #          for e in res.get("entities", []) if e["name"] in name2id]
            inner = self.section_entities_collection[se.name]
            for se in secs:
                self._link_section_to_entities(se, inner, res["chunk_id"])

            # 普通关系
            for rdata in res.get("relations", []):
                rel = self._create_relation_from_data(rdata, res["chunk_id"], entity_map, name2id)
                if rel:
                    self.kg.add_relation(rel)

        # 写入数据库
        if verbose:
            print("💾 存储到数据库...")
        self._store_knowledge_graph(verbose)
        self.neo4j_utils.enrich_event_nodes_with_context()

        if verbose:
            st = self.kg.stats()
            print(f"🎉 知识图谱构建完成!")
            # print(f"   - 实体数量: {st['entities']}")
            # print(f"   - 关系数量: {st['relations']}")
            graph_stats = self.graph_store.get_stats()
            print(f"   - 实体数量: {graph_stats['entities']}")
            print(f"   - 关系数量: {graph_stats['relations']}")
            print(f"   - 文档数量: {st['documents']}")
            print(f"   - 文本块数量: {st['chunks']}")

        return self.kg

    # ═════════════════════════════════════════════════════════════════════
    #  内部工具
    # ═════════════════════════════════════════════════════════════════════
    # -------- 合并实体（根据 doc_type 适配） --------
    def merge_entities_info(self, extraction_results):
        """
        遍历信息抽取结果，合并 / 去重实体。
        - “局部作用域”实体（scope == local）若命名冲突，会在前面加上
          “场景N …” 或 “章节N …” 作为前缀，避免重名。
        - section 的编号优先使用 chunk_metadata.order；若无，则退化为 title。
        """
        extraction_results = self.graph_preprocessor.run_entity_disambiguation(extraction_results)
        
        entity_map: Dict[str, Entity] = {}
        self.chunk2section_map = {result["chunk_id"]: result["chunk_metadata"]["doc_title"] for result in extraction_results}
        self.section_entities_collection = dict()
        
        base = self.config.storage.knowledge_graph_path
        output_path = os.path.join(base, "chunk2section.json")
        # with open(output_path, "w") as f:
        #     json.dump(self.chunk2section_map, f)
    
        # 中文前缀词：Scene → 场景；Chapter → 章节
        for result in extraction_results:
            md = result.get("chunk_metadata", {}) or {}
            label = md.get('doc_title', md.get('subtitle', md.get('title', "")))
            
            if label not in self.section_entities_collection:
                self.section_entities_collection[label] = []
                
            # —— 处理当前 chunk 抽取出的实体 ——
            for ent_data in result.get("entities", []):
                # 冲突处理：局部实体重名前加前缀
                if (ent_data.get("scope", "").lower() == "local" or ent_data.get("type", "") in ["Action", "Emotion", "Goal"])and ent_data["name"] in entity_map:
                    existing_entity = entity_map[ent_data["name"]]
                    existing_chunk_id = existing_entity.source_chunks[0]
                    existing_section_name = self.chunk2section_map[existing_chunk_id]
                    current_section_name = md["doc_title"]
                    if current_section_name != existing_section_name: # 如果不属于同章节的local，需要重命名。
                        new_name = f"{ent_data['name']}_in_{label}"
                        suffix = 1
                        while new_name in entity_map:        # 仍冲突则追加 _n
                            suffix += 1
                            new_name = f"{ent_data['name']}_in_{label}_{suffix}"
                        ent_data["name"] = new_name

                # 创建 / 合并
                ent_obj = self._create_entity_from_data(ent_data, result["chunk_id"])
                existing = self._find_existing_entity(ent_obj, entity_map)
                if existing:
                    self._merge_entities(existing, ent_obj)
                else:
                    entity_map[ent_obj.name] = ent_obj
                self.section_entities_collection[label].append(ent_obj)

        
        output_path = os.path.join(base, "section_entities_collection.pkl")
        # print("[CHECK] self.section_entities_collection: ", self.section_entities_collection)
        with open(output_path, "wb") as f:
            pickle.dump(self.section_entities_collection, f)
        
        return entity_map

    
    def _find_existing_entity(self, entity: Entity, entity_map: Dict[str, Entity]) -> Optional[Entity]:
        """查找已存在的实体"""
        if entity.type == "Event":
            return None
        if entity.name in entity_map:
            return entity_map[entity.name]
        for existing_entity in entity_map.values():
            if entity.name in existing_entity.aliases:
                return existing_entity
            if any(alias in existing_entity.aliases for alias in entity.aliases):
                return existing_entity
        return None
    
    
    def _merge_entities(self, existing: Entity, new: Entity) -> None:
        """合并实体信息"""
        for alias in new.aliases:
            if alias not in existing.aliases:
                existing.aliases.append(alias)
        existing.properties.update(new.properties)
        for chunk_id in new.source_chunks:
            if chunk_id not in existing.source_chunks:
                existing.source_chunks.append(chunk_id)
        if new.description:
            if not existing.description:
                existing.description = new.description
            elif new.description not in existing.description:
                existing.description = existing.description + "\n" + new.description
    
    def _ensure_entity_exists(self, entity_id: str, entity_map: Dict[str, Entity]) -> Optional[Entity]:
        return entity_map.get(entity_id, None)

    # -------- Section / Contains --------
    def _create_section_entities(self, md: Dict[str, Any], chunk_id: str) -> List[Entity]:
        """
        创建章节/场景实体。
        - title/subtitle 总是从 "title"/"subtitle" 字段读取
        - Entity.properties 写入映射字段（如 scene_name） + 其他有用 metadata 字段
        """
        raw_title = md.get("title", "").strip()
        raw_subtitle = md.get("subtitle", "").strip()
        order = md.get("order", None)

        if not raw_title:
            return []

        label = self.meta["section_label"]
        full_name = md.get("doc_title", f"{label}{raw_title}-{raw_subtitle}" if raw_subtitle else f"{label}{raw_title}")
        eid = f"{label.lower()}_{order}" if order is not None else f"{label.lower()}_{hash(full_name) % 1_000_000}"

        title_field = self.meta["title"]
        subtitle_field = self.meta["subtitle"]

        # 构建 properties：写入 title/subtitle 映射字段 + 其他有效字段
        excluded = {"chunk_index", "chunk_type", "doc_title", "title", "subtitle", "total_description_chunks", "total_doc_chunks"}
        properties = {
            title_field: raw_title,
            subtitle_field: raw_subtitle,
        }
        
        if order is not None:
            properties["order"] = order

        for k, v in md.items():
            if k not in excluded:
                properties[k] = v

        return [
            Entity(
                id=eid,
                name=full_name,
                type=label,
                description=md.get("summary", ""),  # 可选：用 summary 作为简要描述
                properties=properties,
                source_chunks=[] # 超节点不需要chunk_ids
            )
        ]


    def _link_section_to_entities(self, section: Entity, inners: List[Entity], chunk_id: str):
        pred = self.meta["contains_pred"]
        for tgt in inners:
            rid = f"rel_{hash(f'{section.id}_{pred}_{tgt.id}') % 1_000_000}"
            self.kg.add_relation(
                Relation(id=rid, subject_id=section.id, predicate=pred,
                         object_id=tgt.id, properties={}, source_chunks=[chunk_id])
            )

    # -------- Entity / Relation creation --------
    @staticmethod
    def _create_entity_from_data(data: Dict, chunk_id: str) -> Entity:
        return Entity(id=f"ent_{hash(data['name']) % 1_000_000}",
                      name=data["name"],
                      type=data.get("type", "Concept"),
                      scope=data.get("scope", "local"),
                      description=data.get("description", ""),
                      aliases=data.get("aliases", []),
                      source_chunks=[chunk_id])

    @staticmethod
    def _create_relation_from_data(d: Dict, chunk_id: str,
                                   entity_map: Dict[str, Entity],
                                   name2id: Dict[str, str]) -> Optional[Relation]:
        subj = d.get("subject") or d.get("source") or d.get("head") or d.get("head_entity")
        obj = d.get("object") or d.get("target") or d.get("tail") or d.get("tail_entity")
        pred = d.get("predicate") or d.get("relation") or d.get("relation_type")
        if not subj or not obj or not pred:
            return None
        sid, oid = name2id.get(subj), name2id.get(obj)
        if not sid or not oid:
            return None
        rid = f"rel_{hash(f'{sid}_{pred}_{oid}') % 1_000_000}"
        
        return Relation(
            id=rid,
            subject_id=sid,
            predicate=pred,
            object_id=oid,
            properties={
                "description": d.get("description", ""),
                "relation_name": d.get("relation_name", "")
            },
            source_chunks=[chunk_id]
        )
        # return Relation(id=rid, subject_id=sid, predicate=pred,
        #                 object_id=oid, properties={}, source_chunks=[chunk_id])

    # -------- 存储 --------
    # def _build_relational_database(self, dialog_chunks: List[TextChunk]):
    #     rows = [{
    #         "id": c.id,
    #         "content": c.content.split("：")[-1].strip(),
    #         "character": c.metadata.get("character", ""),
    #         "type": c.metadata.get("type") or "regular",
    #         "remark": "，".join(c.metadata.get("remark", [])),
    #         "title": c.metadata.get("title", ""),
    #         "subtitle": c.metadata.get("subtitle", ""),
    #     } for c in dialog_chunks]

    #     db_dir = self.config.storage.sql_database_path
    #     os.makedirs(db_dir, exist_ok=True)
    #     db_path = os.path.join(db_dir, "conversations.db")
    #     if os.path.exists(db_path):
    #         os.remove(db_path)
    #     df = pd.DataFrame(rows)
    #     df.to_sql("dialogues", sqlite3.connect(db_path), if_exists="replace", index=False)

    def _store_vectordb(self, verbose: bool):
        try:
            self.vector_store.delete_collection()
            self.vector_store._initialize()
            self.vector_store.store_documents(list(self.kg.documents.values()))
        except Exception as e:
            if verbose:
                print(f"⚠️ 存储失败: {e}")

    def _store_knowledge_graph(self, verbose: bool):
        try:
            self.graph_store.store_knowledge_graph(self.kg)
        except Exception as e:
            if verbose:
                print(f"⚠️ 存储失败: {e}")

    # ═════════════════════════════════════════════════════════════════════
    #  Embedding & Stats
    # ═════════════════════════════════════════════════════════════════════
    def prepare_graph_embeddings(self):
        self.neo4j_utils.load_emebdding_model(self.config.memory.embedding_model_name)
        self.neo4j_utils.create_vector_index()
        self.neo4j_utils.process_all_embeddings(
            exclude_entity_types=[self.meta["section_label"]]
            # exclude_relation_types=[self.meta["contains_pred"]],
        )
        self.neo4j_utils.ensure_entity_superlabel()
        print("✅ 图向量构建完成")

    #
    def get_stats(self) -> Dict[str, Any]:
        return {
            "knowledge_graph": self.kg.stats(),
            "graph_store": self.graph_store.get_stats(),
            "vector_store": self.vector_store.get_stats(),
        }

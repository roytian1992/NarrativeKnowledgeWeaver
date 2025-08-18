# core/builder/graph_builder.py
from __future__ import annotations

import json
import os
import sqlite3
import pickle
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError, wait, FIRST_COMPLETED
import time
from collections import defaultdict
from copy import deepcopy
from typing import Any, Dict, List, Optional
import asyncio
import pandas as pd
import random
import re
import glob
from tqdm import tqdm
from core.utils.prompt_loader import PromptLoader
from core.utils.format import correct_json_format
from core.models.data import Entity, KnowledgeGraph, Relation, TextChunk, Document
from ..storage.document_store import DocumentStore
from ..storage.graph_store import GraphStore
from ..storage.vector_store import VectorStore
from core.memory.vector_memory import VectorMemory
from ..utils.config import KAGConfig
from ..utils.neo4j_utils import Neo4jUtils
from core.model_providers.openai_llm import OpenAILLM
from core.agent.knowledge_extraction_agent import InformationExtractionAgent
from core.agent.attribute_extraction_agent import AttributeExtractionAgent
from core.agent.graph_probing_agent import GraphProbingAgent
from .document_processor import DocumentProcessor
from core.builder.graph_preprocessor import GraphPreprocessor
from core.utils.format import DOC_TYPE_META
from core.builder.reflection import DynamicReflector
from collections import defaultdict

def _normalize_type(t):
    """
    - 输入可为 str 或 list
    - 若包含 'Event' → 把 'Event' 放到首位
    - 去重但保持原相对顺序
    - 若为空 → 返回 'Concept'
    """
    if not t:
        return "Concept"
    if isinstance(t, str):
        return t

    # list 情况
    seen = set()
    ordered = []
    for x in t:
        if x and x not in seen:
            seen.add(x)
            ordered.append(x)

    if "Event" in seen:
        ordered = ["Event"] + [x for x in ordered if x != "Event"]

    return ordered if len(ordered) > 1 else (ordered[0] if ordered else "Concept")
    
# ═════════════════════════════════════════════════════════════════════════════
#                               Builder
# ═════════════════════════════════════════════════════════════════════════════
class KnowledgeGraphBuilder:
    """知识图谱构建器（支持多文档格式）"""
    def __init__(self, config: KAGConfig):
        self.doc_type = config.knowledge_graph_builder.doc_type
        if self.doc_type not in DOC_TYPE_META:
            raise ValueError(f"Unsupported doc_type: {self.doc_type}")
        
        self.config = config
        self.max_workers = config.knowledge_graph_builder.max_workers
        self.meta = DOC_TYPE_META[self.doc_type]
        self.section_chunk_ids = defaultdict(set)

        prompt_dir = config.knowledge_graph_builder.prompt_dir
        self.prompt_loader = PromptLoader(prompt_dir)
        self.llm = OpenAILLM(config)
        self.processor = DocumentProcessor(config, self.llm, self.doc_type)

        # 存储 / 数据库
        self.graph_store = GraphStore(config)
        self.neo4j_utils = Neo4jUtils(self.graph_store.driver, doc_type=self.doc_type)
        self.document_vector_store = VectorStore(config, "documents")
        self.sentence_vector_store = VectorStore(config, "sentences")
        # self.document_store = DocumentStore(config)
        # 初始化记忆模块
        self.reflector = DynamicReflector(config)

        # 运行数据
        self.kg = KnowledgeGraph()
        self.probing_mode = self.config.probing.probing_mode
        self.graph_probing_agent = GraphProbingAgent(self.config, self.llm, self.reflector)
        
    def clear_directory(self, path):
        for file in glob.glob(os.path.join(path, "*.json")):
            try:
                os.remove(file)
            except Exception as e:
                print(f"删除失败: {file} -> {e}")

    def construct_system_prompt(self, background, abbreviations):
        
        background_info = self.get_background_info(background, abbreviations)
        
        if self.doc_type == "screenplay":
            system_prompt_id = "agent_prompt_screenplay"
        else:
            system_prompt_id = "agent_prompt_novel"
            
        system_prompt_text = self.prompt_loader.render_prompt(system_prompt_id, {"background_info": background_info})
        return system_prompt_text
    
    def get_background_info(self, background, abbreviations):
        bg_block = f"**背景设定**：{background}\n" 

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

        abbr_block = "\n".join(fmt(item) for item in abbreviations if isinstance(item, dict))

        if background and abbr_block:
            background_info = f"{bg_block}\n{abbr_block}"
        else:
            background_info = bg_block or abbr_block
            
        return background_info
        
    def prepare_chunks(self, json_file_path: str, verbose: bool = True):
        """
        并发拆分文档为 TextChunk，按【完成顺序】收集；
        单文档软超时=120s（可调），超时则跳过其块并继续处理其它文档。
        """
        from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
        import time
        from tqdm import tqdm
        import os, json

        PER_TASK_TIMEOUT = 120.0  # 可调；比如 180

        # —— 初始化/清理 —— #
        self.reflector.clear()
        base = self.config.storage.knowledge_graph_path
        self.clear_directory(base)
        if verbose:
            print(f"🚀 开始构建知识图谱: {json_file_path}")
            print("📖 加载文档...")

        documents = self.processor.load_from_json(json_file_path, extract_metadata=True)
        if verbose:
            print(f"✅ 成功加载 {len(documents)} 个文档")

        # —— 单文档拆分任务 —— #
        def _run(doc):
            # 期望返回 {"document_chunks": List[TextChunk]}
            return self.processor.prepare_chunk(doc)

        # —— 并发执行：完成即收集 + 软超时 —— #
        all_chunks = []
        timeouts = []
        failures = []

        max_workers = getattr(self, "max_workers", getattr(self, "max_worker", 4))
        executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="chunk")

        try:
            fut_info = {}  # future -> {"start": float, "doc": Any, "idx": int}
            for idx, d in enumerate(documents):
                f = executor.submit(_run, d)
                fut_info[f] = {"start": time.time(), "doc": d, "idx": idx}

            pbar = tqdm(total=len(fut_info), desc="并发拆分中", ncols=100)
            pending = set(fut_info.keys())

            while pending:
                # 1) 先收集已完成的
                done, pending = wait(pending, timeout=0.25, return_when=FIRST_COMPLETED)
                for f in done:
                    info = fut_info.pop(f, None)
                    try:
                        grp = f.result()  # 已完成，不阻塞
                        chunks = (grp or {}).get("document_chunks", [])
                        if isinstance(chunks, list):
                            all_chunks.extend(chunks)
                        else:
                            failures.append(info["idx"])
                    except Exception:
                        failures.append(info["idx"])
                    pbar.update(1)

                # 2) 检查未完成是否超过软超时；超时则不再等待
                now = time.time()
                to_forget = []
                for f in pending:
                    start = fut_info[f]["start"]
                    if now - start >= PER_TASK_TIMEOUT:
                        info = fut_info[f]
                        f.cancel()  # 若已在运行则返回 False；无论如何我们不再等待
                        timeouts.append(info["idx"])
                        pbar.update(1)
                        to_forget.append(f)
                if to_forget:
                    for f in to_forget:
                        pending.remove(f)
                        fut_info.pop(f, None)

            pbar.close()
        finally:
            # 不等待未完成的线程；取消队列里尚未开始的任务，避免退出卡住
            executor.shutdown(wait=False, cancel_futures=True)

        # —— 落盘 —— #
        os.makedirs(base, exist_ok=True)
        out_path = os.path.join(base, "all_document_chunks.json")
        with open(out_path, "w", encoding="utf-8") as fw:
            json.dump([c.dict() for c in all_chunks], fw, ensure_ascii=False, indent=2)

        if verbose:
            print(f"✅ 生成 {len(all_chunks)} 个文本块")
            if timeouts:
                print(f"⏱️ 有 {len(timeouts)} 个文档在 {int(PER_TASK_TIMEOUT)}s 内未完成拆分（已跳过）：{timeouts}")
            if failures:
                print(f"❗有 {len(failures)} 个文档拆分失败（已跳过）：{failures}")

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
        
        
    def run_graph_probing(self, verbose: bool = True, sample_ratio: float = None):
        self.reflector.clear()
        
        base = self.config.storage.graph_schema_path
        os.makedirs(base, exist_ok=True)
        self.clear_directory(base)
        
        if self.probing_mode == "fixed" or self.probing_mode == "adjust" :
            schema = json.load(open(self.config.probing.default_graph_schema_path, "r", encoding="utf-8"))
            if os.path.exists(self.config.probing.default_background_path):
                settings = json.load(open(os.path.join(self.config.probing.default_background_path), "r", encoding="utf-8"))        
            else:
                settings = {"background": "", "abbreviations": []}
        else:
            schema = {}
            settings = {"background": "", "abbreviations": []}
        
        if self.probing_mode != "fixed":
            schema , settings = self.update_schema(schema, background=settings["background"], abbreviations=settings["abbreviations"],
                                                   verbose=verbose, sample_ratio=sample_ratio)
        else:
            if verbose:
                print("📌 跳过probing步骤")
            
        with open(os.path.join(base, "graph_schema.json"), "w", encoding="utf-8") as f:
            json.dump(schema, f, ensure_ascii=False, indent=2)
            
        with open(os.path.join(base, "settings.json"), "w", encoding="utf-8") as f:
            json.dump(settings, f, ensure_ascii=False, indent=2)
            
            
    def initialize_agents(self):
        
        base = self.config.storage.graph_schema_path
        schema_path = os.path.join(base, "graph_schema.json")
        settings_path = os.path.join(base, "settings.json")
        if os.path.exists(schema_path):
            schema = json.load(open(schema_path, "r", encoding="utf-8"))
        elif os.path.exists(self.config.probing.default_graph_schema_path):
            schema = json.load(open(self.config.probing.default_graph_schema_path, "r", encoding="utf-8"))
        else:
            raise FileNotFoundError("没有graph schema，请提供正确路径或者先运行probing")
        
        if os.path.exists(settings_path):
            settings = json.load(open(settings_path, "r", encoding="utf-8"))
        elif os.path.exists(self.config.probing.default_background_path):
            settings = json.load(open(self.config.probing.default_background_path, "r", encoding="utf-8"))
        else:
            settings = {"background": "", "abbreviations": []}
 
        self.system_prompt_text = self.construct_system_prompt(
            background=settings["background"],
            abbreviations=settings["abbreviations"]
        )

        # 抽取 agent
        self.information_extraction_agent = InformationExtractionAgent(self.config, self.llm, self.system_prompt_text, schema, self.reflector)
        self.attribute_extraction_agent = AttributeExtractionAgent(self.config, self.llm, self.system_prompt_text, schema)
        self.graph_preprocessor = GraphPreprocessor(self.config, self.llm, system_prompt=self.system_prompt_text)
        
    def update_schema(self, schema: Dict = {}, background: str = "", abbreviations: List = [], verbose: bool = True, sample_ratio: float = None):
        
        base = self.config.storage.knowledge_graph_path
        doc_chunks = [TextChunk(**o) for o in
                       json.load(open(os.path.join(base, "all_document_chunks.json"), "r", encoding="utf-8"))]
        if not sample_ratio:
            sample_ratio = 0.35
        k = int(len(doc_chunks) * sample_ratio)
        sampled_chunks = random.sample(doc_chunks, k=k)
        # 保存一些洞察，用于之后的探索。
        sampled_chunks = self.processor.extract_insights(sampled_chunks)
        
        for chunk in tqdm(sampled_chunks, desc="保存洞见中", total=len(sampled_chunks)):
            insights = chunk.metadata.get("insights", []) 
            for item in insights:
                # print("[CHECK] insight", item)
                self.reflector.insight_memory.add(text=item, metadata={})
        if verbose:
            print("💾 保存洞见完成")
            
        params = dict()
        params["documents"] = sampled_chunks   
        params["schema"] = schema  
        params["background"] = background
        params["abbreviations"] = abbreviations
        
        result = self.graph_probing_agent.run(params)
        # result = json.loads(correct_json_format(result))
        
        return result["schema"], result["settings"]
        
    # ═════════════════════════════════════════════════════════════════════
    #  3) 实体 / 关系 抽取
    # ═════════════════════════════════════════════════════════════════════
    def extract_entity_and_relation(self, verbose: bool = True):
        asyncio.run(self.extract_entity_and_relation_async(verbose=verbose))
            
    async def extract_entity_and_relation_async(self, verbose: bool = True):
        """
        并发抽取 → 记录失败的 chunk → 统一重试一轮 → 写盘
        """
        base = self.config.storage.knowledge_graph_path
        desc_chunks = [TextChunk(**o) for o in
                    json.load(open(os.path.join(base, "all_document_chunks.json"), "r", encoding="utf-8"))]

        if verbose:
            print("🧠 实体与关系信息异步抽取中...")

        sem = asyncio.Semaphore(self.max_workers)

        async def _arun_once(ch: TextChunk):
            async with sem:
                try:
                    if not ch.content.strip():
                        result = {"entities": [], "relations": []}
                    else:
                        result = await self.information_extraction_agent.arun(
                            ch.content,
                            timeout=self.config.agent.async_timeout,
                            max_attempts=self.config.agent.async_max_attempts,
                            backoff_seconds=self.config.agent.async_backoff_seconds
                        )
                    result.update(chunk_id=ch.id, chunk_metadata=ch.metadata)
                    return result
                except Exception as e:
                    if verbose:
                        print(f"[ERROR] 抽取失败 chunk_id={ch.id} | {e.__class__.__name__}: {e}")
                    return {
                        "chunk_id": ch.id,
                        "chunk_metadata": ch.metadata,
                        "entities": [],
                        "relations": [],
                        "error": f"{e.__class__.__name__}: {e}"
                    }

        async def _arun_with_ch(ch: TextChunk):
            """返回 (chunk, result) 方便直接记录失败的 chunk。"""
            res = await _arun_once(ch)
            return ch, res

        # ====== 首轮并发 ======
        tasks = [_arun_with_ch(ch) for ch in desc_chunks]
        first_round_pairs = []   # [(ch, res), ...]
        failed_chs = []          # [ch, ...]

        for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="异步抽取中"):
            ch, res = await coro
            first_round_pairs.append((ch, res))
            if res.get("error"):
                failed_chs.append(ch)

        # ====== 统一重试（只一轮）======
        retry_pairs = []
        if failed_chs:
            if verbose:
                print(f"🔄 开始重试失败的 {len(failed_chs)} 个文本块...")
            retry_tasks = [_arun_with_ch(ch) for ch in failed_chs]
            for coro in tqdm(asyncio.as_completed(retry_tasks), total=len(retry_tasks), desc="重试抽取中"):
                ch, res = await coro
                retry_pairs.append((ch, res))

        # ====== 合并结果（用失败的 ch 过滤首轮对应结果，再追加重试结果）======
        failed_ids = {ch.id for ch in failed_chs}
        final_results = [res for ch, res in first_round_pairs if ch.id not in failed_ids]
        final_results += [res for _, res in retry_pairs]

        # ====== 落盘 ======
        output_path = os.path.join(base, "extraction_results.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(final_results, f, ensure_ascii=False, indent=2)

        if verbose:
            still_failed = sum(1 for r in final_results if r.get("error"))
            print(f"✅ 实体与关系信息抽取完成，共处理 {len(final_results)} 个文本块")
            if still_failed:
                print(f"⚠️ 仍有 {still_failed} 个文本块在重试后失败（保留 error 字段以便排查）")
            print(f"💾 已保存至：{output_path}")


    # ═════════════════════════════════════════════════════════════════════
    #  4) 属性抽取
    # ═════════════════════════════════════════════════════════════════════
    def run_extraction_refinement(self, verbose=False):
        """
        实体消歧
        """
        KW = ("闪回", "一组蒙太奇")
        base = self.config.storage.knowledge_graph_path
        extraction_results = json.load(open(os.path.join(base, "extraction_results.json"), "r", encoding="utf-8"))
        for doc in extraction_results:
            # 1) 先删实体：名字含关键词的一律删除
            ents = doc.get("entities", [])
            removed_names = {e.get("name", "") for e in ents if any(k in e.get("name", "") for k in KW)}
            doc["entities"] = [e for e in ents if e.get("name", "") not in removed_names]

            # 2) 再删关系：subject/object 含关键词，或指向已删除实体名
            rels = doc.get("relations", [])
            doc["relations"] = [
                r for r in rels
                if not any(k in r.get("subject", "") or k in r.get("object", "") for k in KW)
                and r.get("subject", "") not in removed_names
                and r.get("object", "") not in removed_names
            ]
    
        if verbose:
            print("📌 优化实体类型")
        extraction_results =self.graph_preprocessor.refine_entity_types(extraction_results)
        if verbose:
            print("📌 优化实体scope")
        extraction_results = self.graph_preprocessor.refine_entity_scope(extraction_results)
        if verbose:
            print("📌 实体消歧")
        extraction_results = self.graph_preprocessor.run_entity_disambiguation(extraction_results)
        
        base = self.config.storage.knowledge_graph_path
        with open(os.path.join(base, "extraction_results_refined.json"), "w") as f:
            json.dump(extraction_results, f, ensure_ascii=False, indent=2)
        

    def extract_entity_attributes(self, verbose: bool = True) -> Dict[str, Entity]:
        asyncio.run(self.extract_entity_attributes_async(verbose=verbose))
    
    async def extract_entity_attributes_async(self, verbose: bool = True) -> Dict[str, Entity]:
        """
        ⚡ 异步批量属性抽取  
        · 按 extract_entity_and_relation_async 生成的 entity_map 去并发  
        · 每个实体调用 attribute_extraction_agent.arun()  
        · 内部 arun 已带超时＋重试保护，不会卡死
        """
        base = self.config.storage.knowledge_graph_path
        results = json.load(open(os.path.join(base, "extraction_results_refined.json"), "r", encoding="utf-8"))
        
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
                        original_text="",
                        timeout=self.config.agent.async_timeout,
                        max_attempts=self.config.agent.async_max_attempts,
                        backoff_seconds=self.config.agent.async_backoff_seconds
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

    # ═════════════════════════════════════════════════════════════════════
    #  5) 构建并存储图谱
    # ═════════════════════════════════════════════════════════════════════
    def build_graph_from_results(self, verbose: bool = True) -> KnowledgeGraph:
        if verbose:
            print("📂 加载已有抽取结果和实体信息...")

        base = self.config.storage.knowledge_graph_path
        results = json.load(open(os.path.join(base, "extraction_results_refined.json"), "r", encoding="utf-8"))
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
                else:
                    exist = self.kg.entities.get(se.id)
                    if exist:
                        # 去重并保持稳定顺序
                        merged = list(dict.fromkeys(list(exist.source_chunks) + list(se.source_chunks)))
                        exist.source_chunks = merged
                

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
                t = ent_data.get("type", "")
                is_event = (t == "Event") or (isinstance(t, list) and "Event" in t)
                is_action_like = (
                    (isinstance(t, str) and t in ["Action", "Emotion", "Goal"]) or
                    (isinstance(t, list) and any(x in ["Action", "Emotion", "Goal"] for x in t))
                )
                if is_event:
                    is_action_like = False
                            
                # 冲突处理：局部实体重名前加前缀
                if (ent_data.get("scope", "").lower() == "local" or is_action_like) and ent_data["name"] in entity_map:
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
        if (entity.type == "Event") or (isinstance(entity.type, list) and "Event" in entity.type):
            return None
        if entity.name in entity_map:
            return entity_map[entity.name]
        for existing_entity in entity_map.values():
            if entity.name in existing_entity.aliases:
                return existing_entity
            if any(alias in existing_entity.aliases for alias in entity.aliases):
                return existing_entity
        return None
    
    def _merge_types(self, a, b):
        """
        把 a 和 b 的类型并集后做规范化（Event 优先 + 去重保序）
        """
        a_list = a if isinstance(a, list) else ([a] if a else [])
        b_list = b if isinstance(b, list) else ([b] if b else [])
        merged = []
        seen = set()
        for x in a_list + b_list:
            if x and x not in seen:
                seen.add(x)
                merged.append(x)
        if "Event" in seen:
            merged = ["Event"] + [x for x in merged if x != "Event"]
        return merged if len(merged) > 1 else (merged[0] if merged else "Concept")


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
                
        existing.type = self._merge_types(existing.type, new.type)
        
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

        self.section_chunk_ids[eid].add(chunk_id)
        agg_chunks = sorted(self.section_chunk_ids[eid])

        return [
            Entity(
                id=eid,
                name=full_name,
                type=label,
                description=md.get("summary", ""),  # 可选：用 summary 作为简要描述
                properties=properties,
                source_chunks=agg_chunks 
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
                      type=_normalize_type(data.get("type", "Concept")),
                      scope=data.get("scope", "local"),
                      description=data.get("description", ""),
                      aliases=data.get("aliases", []),
                      source_chunks=[chunk_id])

    @staticmethod
    def _create_relation_from_data(d: Dict, chunk_id: str,
                                   entity_map: Dict[str, Entity],
                                   name2id: Dict[str, str]) -> Optional[Relation]:
        subj = d.get("subject") or d.get("source") or d.get("head") or d.get("relation_subject")
        obj = d.get("object") or d.get("target") or d.get("tail") or d.get("relation_object")
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
            all_documents = list(self.kg.documents.values())
            self.document_vector_store.delete_collection()
            self.document_vector_store._initialize()
            self.document_vector_store.store_documents(all_documents)
            
            all_sentences = []
            for doc in all_documents:
                content = doc.content
                sentences = re.split(r'(?<=[。！？])', content)
                for i, sentence in enumerate(sentences):
                    all_sentences.append(Document(id=f"{doc.id}-{i+1}", content=sentence, metadata=doc.metadata))
            
            self.sentence_vector_store.delete_collection()
            self.sentence_vector_store._initialize()
            self.sentence_vector_store.store_documents(all_sentences)
            
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
        self.neo4j_utils.load_embedding_model(self.config.graph_embedding)
        self.neo4j_utils.create_vector_index()
        self.neo4j_utils.process_all_embeddings(
            # exclude_entity_types=[self.meta["section_label"]]
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

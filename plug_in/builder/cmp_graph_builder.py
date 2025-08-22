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
import random
import re, hashlib
import glob
from tqdm import tqdm
from core.utils.prompt_loader import PromptLoader
from core.utils.format import correct_json_format
from core.models.data import Entity, KnowledgeGraph, Relation, TextChunk, Document
from core.storage.graph_store import GraphStore
from core.utils.config import KAGConfig
from core.utils.neo4j_utils import Neo4jUtils
from core.model_providers.openai_llm import OpenAILLM
from core.agent.knowledge_extraction_agent import InformationExtractionAgent
from core.agent.attribute_extraction_agent import AttributeExtractionAgent
from core.utils.format import DOC_TYPE_META
from plug_in.builder.reflection import DynamicReflector
from collections import defaultdict

def _norm_name(c) -> str:
    if c.scope == "local" and "_" in c.name:
        return c.name.split("_")[0]
    else:
        return c.name
   
# ═════════════════════════════════════════════════════════════════════════════
#                               Builder
# ═════════════════════════════════════════════════════════════════════════════
class CMPKnowledgeGraphBuilder:
    """知识图谱构建器（支持多文档格式）"""
    def __init__(self, config: KAGConfig):
        self.doc_type = config.knowledge_graph_builder.doc_type
        if self.doc_type not in DOC_TYPE_META:
            raise ValueError(f"Unsupported doc_type: {self.doc_type}")
        
        self.config = config
        self.max_workers = config.knowledge_graph_builder.max_workers
        self.meta = DOC_TYPE_META[self.doc_type]
        self.section_chunk_ids = defaultdict(set)
        self.plug_in_path = "./plug_in"
        prompt_dir = os.path.join(self.plug_in_path, "prompts")
        self.prompt_loader = PromptLoader(prompt_dir)
        self.llm = OpenAILLM(config)

        # 存储 / 数据库
        self.graph_store = GraphStore(config)
        self.neo4j_utils = Neo4jUtils(self.graph_store.driver, doc_type=self.doc_type)
        # 初始化记忆模块
        self.reflector = DynamicReflector(config)
        self.reflector.clear()
        # 运行数据
        self.kg = KnowledgeGraph()
        self.item_name2object_id: Dict[str, str] = {}
        self.preload_characters_and_objects()


    def preload_characters_and_objects(self):
        characters = self.neo4j_utils.search_entities_by_type("Character", "")
        objects = self.neo4j_utils.search_entities_by_type("Object", "")
        self.character_name2id: Dict[str, str] = {}   # 角色名/别名 -> 现有 Character.id
        self.object_name2id: Dict[str, str] = {}      # 物品名/别名 -> 现有 Object.id
        for chracter in characters:
            if chracter.scope == "local" and "_" in chracter.name:
                character_name = chracter.name.split("_")[0]
                self.character_name2id[chracter.name] = chracter.id
            else:
                character_name = chracter.name
            self.character_name2id[character_name] = chracter.id
        for obj in objects:
            if obj.scope == "local" and "_" in obj.name:
                object_name = obj.name.split("_")[0]
                self.object_name2id[object_name] = obj.id 
            else:
                object_name = obj.name
            self.object_name2id[object_name] = obj.id  

    def clear_directory(self, path):
        for file in glob.glob(os.path.join(path, "*.json")):
            try:
                os.remove(file)
            except Exception as e:
                print(f"删除失败: {file} -> {e}")

    def construct_system_prompt(self, background, abbreviations):
        background_info = self.get_background_info(background, abbreviations)
        system_prompt_id = "agent_prompt"
        system_prompt_text = self.prompt_loader.render_prompt(system_prompt_id, {"background_info": background_info})
        return system_prompt_text
    

    def get_related_content(self, section_name: str) -> str:
        scenes = self.neo4j_utils.search_entities_by_type(self.meta["section_label"], section_name) or []
        if not scenes:
            return ""  # 找不到场景就不加提示，避免异常

        scene_id = scenes[0].id
        characters = self.neo4j_utils.search_related_entities(
            scene_id, predicate=self.meta["contains_pred"], entity_types=["Character"]
        )
        objects = self.neo4j_utils.search_related_entities(
            scene_id, predicate=self.meta["contains_pred"], entity_types=["Object"]
        )
        
        character_info = "、".join([_norm_name(c) for c in characters])
        object_info = "、".join([_norm_name(o) for o in objects])

        parts = []
        if character_info:
            parts.append(f"当前场景包含角色有：{character_info}。")
        if object_info:
            parts.append(
                "当前场景包含物品有：" + object_info +
                "。在抽取时优先对照已有物品清单，若其中包含服化道相关项则直接抽取；若清单中没有，但上下文另有服化道元素，则补充抽取。"
            )
        return "\n".join(parts)

    
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
        

            
    def initialize_agents(self):
        
        schema_path = os.path.join(self.plug_in_path, "schema", "graph_schema.json")
        settings_path = os.path.join(self.config.storage.graph_schema_path, "settings.json")
        if os.path.exists(schema_path):
            schema = json.load(open(schema_path, "r", encoding="utf-8"))
        else:
            raise FileNotFoundError("没有plug_in的graph schema")
        
        self.entity_white_list = set([item["type"] for item in schema.get("entities", [])])
        relations = []
        for k, v in schema.get("relations", {}).items():
            relations.extend(v)
        self.relation_white_list = set([item["type"] for item in relations])
        
        if os.path.exists(settings_path):
            settings = json.load(open(settings_path, "r", encoding="utf-8"))
        else:
            settings = {"background": "", "abbreviations": []}
 
        self.system_prompt_text = self.construct_system_prompt(
            background=settings["background"],
            abbreviations=settings["abbreviations"]
        )
        for entity_type in self.entity_white_list:
            if entity_type != "Character":
                self.neo4j_utils.delete_entity_type(entity_type, exclude_labels=["Object"])
                
        # 抽取 agent
        self.information_extraction_agent = InformationExtractionAgent(self.config, self.llm, self.system_prompt_text, schema, self.reflector, prompt_loader=self.prompt_loader)
        self.attribute_extraction_agent = AttributeExtractionAgent(self.config, self.llm, self.system_prompt_text, schema, prompt_loader=self.prompt_loader)
        
    # ═════════════════════════════════════════════════════════════════════
    #  3) 实体 / 关系 抽取
    # ═════════════════════════════════════════════════════════════════════
    def extract_entity_and_relation(self, verbose: bool = True):
        asyncio.run(self.extract_entity_and_relation_async(verbose=verbose))
            
    async def extract_entity_and_relation_async(self, verbose: bool = True):
        """
        并发抽取 → 记录失败的 chunk → 统一重试一轮 → 写盘
        """

        desc_chunks = [TextChunk(**o) for o in
                    json.load(open(os.path.join(self.config.storage.knowledge_graph_path, "all_document_chunks.json"), "r", encoding="utf-8"))]

        if verbose:
            print("🧠 实体与关系信息异步抽取中...")

        sem = asyncio.Semaphore(self.max_workers)

        async def _arun_once(ch: TextChunk):
            async with sem:
                try:
                    if not ch.content.strip():
                        result = {"entities": [], "relations": []}
                    else:
                        raw_content = ch.content.strip()
                        section_name = ch.metadata.get("doc_title", "")
                        related_content = self.get_related_content(section_name) 
                        content = raw_content + "\n" + related_content
                        result = await self.information_extraction_agent.arun(
                            content,
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
        base = os.path.join(self.config.storage.knowledge_graph_path, "plug_in")
        os.makedirs(base, exist_ok=True)
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

    def extract_entity_attributes(self, verbose: bool = True) -> Dict[str, Entity]:
        asyncio.run(self.extract_entity_attributes_async(verbose=verbose))
    
    async def extract_entity_attributes_async(self, verbose: bool = True) -> Dict[str, Entity]:
        """
        ⚡ 异步批量属性抽取  
        · 按 extract_entity_and_relation_async 生成的 entity_map 去并发  
        · 每个实体调用 attribute_extraction_agent.arun()  
        · 内部 arun 已带超时＋重试保护，不会卡死
        """
        base = os.path.join(self.config.storage.knowledge_graph_path, "plug_in")
        os.makedirs(base, exist_ok=True)
        results = json.load(open(os.path.join(base, "extraction_results.json"), "r", encoding="utf-8"))
        
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

                    if res.get("error"):  # 超时或异常
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
        tasks = []
        for n, e in entity_map.items():
            # 只处理 CMP 白名单里的类型，且跳过 Character
            if e.type[0] not in self.entity_white_list or e.type[0] == "Character":
                continue
            tasks.append(_arun_attr(n, e))

        for coro in tqdm(asyncio.as_completed(tasks),
                            total=len(tasks),
                            desc="属性抽取中（async）"):
            n, e2 = await coro
            if e2:
                updated_entities[n] = e2
                target_id = None
                if n in self.item_name2object_id:
                    # —— 已存在的 Object —— 增加 CMP 标签
                    target_id = self.item_name2object_id[n]
                    try:
                        self.neo4j_utils.add_labels(target_id, self._to_type_list(e2.type))
                    except Exception as ex:
                        if verbose:
                            print(f"[WARN] Neo4j 添加标签失败: {n} ({target_id}): {ex}")
                else:
                    # —— 纯 CMP 节点 —— 直接用自身 ID
                    target_id = e2.id

                if target_id and e2.properties:
                    try:
                        self.neo4j_utils.update_entity_properties(target_id, e2.properties)
                    except Exception as ex:
                        if verbose:
                            print(f"[WARN] Neo4j 属性写回失败: {n} ({target_id}): {ex}")


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
    @staticmethod
    def _to_type_list(t) -> List[str]:
        """
        将传入的 t 统一规范为 List[str]：
        - None         -> []
        - 'Styling'    -> ['Styling']
        - ['A','B',''] -> ['A','B']
        - 其它类型     -> []
        """
        # print("[CHECK] t", t)
        if t is None:
            return []
        if isinstance(t, list):
            return [x for x in t if isinstance(x, str) and x.strip()]
        if isinstance(t, str):
            s = t.strip()
            return [s] if s else []
        return []

    
    def _ensure_entity_exists(
            self,
            entity_payload: Any,
            ent_by_name: Dict[str, "Entity"],
            ent_by_id: Dict[str, "Entity"],
        ) -> Optional[str]:
            """
            确保实体在 self.kg.entities 里存在：
            - 命中 KG/缓存：合并 properties
            - 命中 ent_by_id / ent_by_name：加入 KG
            - 否则创建最小实体（稳定 ent_cmp_*），保留多类型（List[str]）
            返回实体 id 或 None
            """
            ep = self._normalize_entity_payload(entity_payload, ent_by_name, ent_by_id)
            if not ep:
                return None

            props_in = self._to_props_dict(ep.get("properties", {}))
            e_name = ep.get("name")
            e_id   = ep.get("id")
            e_type = self._to_type_list(ep.get("type"))

            # 1) KG 内已有
            if e_id and e_id in self.kg.entities:
                old = self.kg.entities[e_id]
                old.properties = {**self._to_props_dict(getattr(old, "properties", {})), **props_in}
                return e_id

            # 2) ent_by_id 中已有
            if e_id and e_id in ent_by_id:
                e = ent_by_id[e_id]
                e.properties = {**self._to_props_dict(getattr(e, "properties", {})), **props_in}
                if e.id not in self.kg.entities:
                    self.kg.add_entity(e)
                return e.id

            # 3) 通过 name 命中
            if not e_id and e_name and e_name in ent_by_name:
                e = ent_by_name[e_name]
                e.properties = {**self._to_props_dict(getattr(e, "properties", {})), **props_in}
                if e.id not in self.kg.entities:
                    self.kg.add_entity(e)
                return e.id

            # 4) 全新最小实体（稳定 CMP id）
            if not e_id:
                base = f"{e_name or ''}|{','.join(e_type) or ''}"
                e_id = "ent_cmp_" + hashlib.md5(base.encode("utf-8")).hexdigest()[:12]

            payload = {
                "id": e_id,
                "name": e_name or e_id,
                "type": e_type or ["Concept"],
                "description": "",
                "aliases": ep.get("aliases", []),
                "properties": props_in,
                "source_chunks": [],
            }
            try:
                new_e = Entity(**payload)
            except TypeError:
                minimal = {k: payload[k] for k in ("id", "name", "type", "properties") if k in payload}
                new_e = Entity(**minimal)

            if new_e.id not in self.kg.entities:
                self.kg.add_entity(new_e)
            return new_e.id
    

    def _normalize_entity_payload(
            self,
            raw: Any,
            ent_by_name: Dict[str, "Entity"],
            ent_by_id: Dict[str, "Entity"],
        ) -> Optional[Dict[str, Any]]:
            """
            把 subject/object 的任意形态（dict / id 字符串 / name 字符串）统一成规范 payload：
            返回子集：{"id": str|None, "name": str|None, "type": List[str], "properties": dict, "aliases": list}
            不修改章节逻辑。
            """
            if not raw:
                return None

            def _from_entity(e: "Entity") -> Dict[str, Any]:
                return {
                    "id": e.id,
                    "name": e.name,
                    "type": self._to_type_list(getattr(e, "type", [])),
                    "properties": self._to_props_dict(getattr(e, "properties", {})),
                    "aliases": list(getattr(e, "aliases", []) or []),
                }

            # dict：优先 id，其次 name；否则按字段原样规整
            if isinstance(raw, dict):
                rid = (raw.get("id") or "").strip()
                rname = (raw.get("name") or "").strip()
                if rid and rid in ent_by_id:
                    return _from_entity(ent_by_id[rid])
                if rname:
                    if rname in ent_by_name:
                        return _from_entity(ent_by_name[rname])
                    if rname in self.character_name2id:
                        return {"id": self.character_name2id[rname], "name": rname, "type": ["Character"], "properties": {}, "aliases": []}
                    if rname in self.object_name2id:
                        return {"id": self.object_name2id[rname], "name": rname, "type": ["Object"], "properties": {}, "aliases": []}
                return {
                    "id": rid or None,
                    "name": rname or None,
                    "type": self._to_type_list(raw.get("type")),
                    "properties": self._to_props_dict(raw.get("properties", {})),
                    "aliases": list(raw.get("aliases", []) or []),
                }

            # 字符串：可能是 id 或 name
            if isinstance(raw, str):
                s = raw.strip()
                if not s:
                    return None
                # 先按 id 命中
                if s in ent_by_id:
                    return _from_entity(ent_by_id[s])
                # 再按 name 命中
                if s in ent_by_name:
                    return _from_entity(ent_by_name[s])
                # 兜底：已有角色/物品映射
                if s in self.character_name2id:
                    return {"id": self.character_name2id[s], "name": s, "type": ["Character"], "properties": {}, "aliases": []}
                if s in self.object_name2id:
                    return {"id": self.object_name2id[s], "name": s, "type": ["Object"], "properties": {}, "aliases": []}
                # 像 id 的模式
                if s.startswith("ent_") or re.match(r"^[A-Za-z]{2,5}_[A-Za-z0-9\-]+$", s):
                    return {"id": s, "name": None, "type": [], "properties": {}, "aliases": []}
                # 作为名称返回
                return {"id": None, "name": s, "type": [], "properties": {}, "aliases": []}

            return None
    
    def build_graph_from_results(self, verbose: bool = True) -> KnowledgeGraph:
        """
        从 plug_in/extraction_results.json 与 entity_info.json 构建 KG，并写入 Neo4j：
        1) 规范化/加载实体（先载入 Neo4j 既有 Character/Object 防止关系触发空节点）
        2) 扫描抽取结果补实体、创建 Section 节点并建立包含关系
        3) 生成并写入关系
        4) 最后统一写库与后处理（上下文富集与中心性计算）
        """
        # 为本函数取别名，便于局部调用
        _to_props_dict = self._to_props_dict

        if verbose:
            print("📂 加载已有抽取结果和实体信息...")

        base = os.path.join(self.config.storage.knowledge_graph_path, "plug_in")
        os.makedirs(base, exist_ok=True)

        # ---- 读取抽取/属性/分组文件 ----
        results_path = os.path.join(base, "extraction_results.json")
        entinfo_path = os.path.join(base, "entity_info.json")
        secs_pkl_path = os.path.join(base, "section_entities_collection.pkl")

        # 必要文件检查
        if not os.path.exists(results_path):
            raise FileNotFoundError(f"缺少抽取结果文件：{results_path}")
        if not os.path.exists(entinfo_path):
            # 没做属性抽取也允许继续（给个空）
            if verbose:
                print(f"⚠️ 未发现属性信息文件：{entinfo_path}，将仅依据抽取结果构建图。")
            ent_raw = {}
        else:
            ent_raw = json.load(open(entinfo_path, "r", encoding="utf-8"))

        results = json.load(open(results_path, "r", encoding="utf-8"))

        # section_entities_collection 如存在可复用（不强依赖）
        if os.path.exists(secs_pkl_path):
            with open(secs_pkl_path, "rb") as f:
                self.section_entities_collection = pickle.load(f)
        else:
            self.section_entities_collection = {}

        # ---- 1) 规范化 entity_info.json → Entity 映射 ----
        ent_by_name: Dict[str, Entity] = {}
        for name, payload in ent_raw.items():
            payload = dict(payload)
            payload["properties"] = _to_props_dict(payload.get("properties", {}))
            # print("[CHECK] payload: ", payload)
            try:
                ent_by_name[name] = Entity(**payload)
            except TypeError:
                # 最小化回退（防御）
                minimal = {
                    "id": payload.get("id") or f"ent_cmp_{hashlib.md5(name.encode('utf-8')).hexdigest()[:12]}",
                    "name": name,
                    "type": payload.get("type") or "Concept",
                    "properties": payload.get("properties", {}),
                }
                ent_by_name[name] = Entity(**minimal)

        ent_by_id: Dict[str, Entity] = {e.id: e for e in ent_by_name.values()}

        # ---- 2) 预加载 Neo4j 中已有角色/物品，先入 KG，避免后续关系 MERGE 出“空节点” ----
        def _safe_add(e: Entity):
            if not e:
                return
            e.properties = _to_props_dict(getattr(e, "properties", {}))
            # print("[CHECK] 预加载实体: ", e.properties)
            if e.id not in self.kg.entities:
                self.kg.add_entity(e)

        pre_chars = self.neo4j_utils.search_entities_by_type("Character", "")
        pre_objs = self.neo4j_utils.search_entities_by_type("Object", "")

        for e in pre_chars or []:
            _safe_add(e)
        for e in pre_objs or []:
            _safe_add(e)

        # ---- 3) 把 ent_raw 的实体也写入 KG（以 id 去重）----
        # ---- 3) 把 ent_raw 的实体也写入 KG（以 id 去重），并合并属性 ----
        for e in ent_by_name.values():
            if e.id in self.kg.entities:
                exist = self.kg.entities[e.id]
                # 合并属性（entity_info 优先）
                exist.properties = {**(getattr(exist, "properties", {}) or {}), **(getattr(e, "properties", {}) or {})}

                print("[CHECK] 合并实体: ", exist.properties)

                # 合并别名/描述/来源（按需）
                for a in getattr(e, "aliases", []) or []:
                    if a and a not in getattr(exist, "aliases", []):
                        exist.aliases.append(a)
                if getattr(e, "description", "") and getattr(e, "description", "") not in (getattr(exist, "description", "") or ""):
                    exist.description = (exist.description + "\n" if exist.description else "") + e.description
                # 不重新 add_entity
            else:
                self.kg.add_entity(e)


        if verbose:
            print("🔗 构建知识图谱...")

        self.section_names = []


        # ==== 先扫一遍，补“只在关系里出现”的实体 ====
        for res in results:
            for ed in res.get("entities", []) or []:
                self._ensure_entity_exists(ed, ent_by_name, ent_by_id)

        # ==== Section 节点与包含关系 + 普通关系 ====
        for res in results:
            md = res.get("chunk_metadata", {}) or {}

            # 先创建/合并 Section 节点
            secs = self._create_section_entities(md, res["chunk_id"])
            for se in secs:
                if se.name not in self.section_names and se.id not in self.kg.entities:
                    self.kg.add_entity(se)
                    self.section_names.append(se.name)
                else:
                    # 合并 source_chunks
                    exist = self.kg.entities.get(se.id)
                    if exist:
                        merged = list(dict.fromkeys(list(getattr(exist, "source_chunks", [])) + list(getattr(se, "source_chunks", []))))
                        exist.source_chunks = merged

            # 把当前 section 与该 section 已收集的实体链接
            for se in secs:
                inner = self.section_entities_collection.get(se.name, [])
                self._link_section_to_entities(se, inner, res["chunk_id"])

            # 再处理普通关系
            for rdata in res.get("relations", []) or []:
                # 兜底保证端点存在
                self._ensure_entity_exists(rdata.get("subject"), ent_by_name, ent_by_id)
                self._ensure_entity_exists(rdata.get("object"), ent_by_name, ent_by_id)

                rel = self._create_relation_from_data(rdata, res["chunk_id"], ent_by_name)
                if rel:
                    self.kg.add_relation(rel)

        # ---- 5) 写库（先实体后关系）并后处理 ----
        if verbose:
            print("💾 存储到数据库...")

        # 5.1 实体 MERGE（含属性）
        for e in self.kg.entities.values():
            props = _to_props_dict(getattr(e, "properties", {}))
            try:
                self.neo4j_utils.merge_entity_with_properties(
                    node_id=e.id,
                    name=e.name,
                    etypes=e.type,
                    aliases=getattr(e, "aliases", []),
                    props=props,                 # 保持原样传入（可能为空）
                    store_mode="both"
                )
            except Exception as ex:
                if verbose:
                    print(f"[WARN] 写入实体失败：{e.id} / {e.name} -> {ex}")

        # 5.2 关系写入
        self._store_knowledge_graph(verbose)

        # 5.3 其它后处理
        try:
            self.neo4j_utils.enrich_event_nodes_with_context()
        except Exception as ex:
            if verbose:
                print(f"[WARN] enrich_event_nodes_with_context 失败：{ex}")

        try:
            self.neo4j_utils.compute_centrality(exclude_rel_types=[self.meta['contains_pred']])
        except Exception as ex:
            if verbose:
                print(f"[WARN] compute_centrality 失败：{ex}")

        if verbose:
            st = self.kg.stats()
            print("🎉 道具图谱构建完成!")
            try:
                graph_stats = self.graph_store.get_stats()
                print(f" - 实体数量: {graph_stats.get('entities')}")
                print(f" - 关系数量: {graph_stats.get('relations')}")
            except Exception:
                pass
            print(f" - 文档数量: {st['documents']}")
            print(f" - 文本块数量: {st['chunks']}")

        return self.kg

    @staticmethod
    def _to_props_dict(props):
        if props is None:
            return {}
        if isinstance(props, dict):
            return props
        if isinstance(props, str):
            s = props.strip()
            if not s:
                return {}
            try:
                return json.loads(s)
            except Exception:
                return {}
        return {}

    # ═════════════════════════════════════════════════════════════════════
    #  内部工具
    # ═════════════════════════════════════════════════════════════════════
    # -------- 合并实体（根据 doc_type 适配） --------
    def merge_entities_info(self, extraction_results):
        """
        遍历信息抽取结果，合并 / 去重实体。
        - 仅保留白名单内的实体类型。
        - Character 不新建节点，只尝试映射到已有 ID。
        - Object / WardrobeItem / PropItem 优先映射到已有 Object，否则新建。
        - 其它白名单类型正常新建/合并。
        """
        entity_map: Dict[str, Entity] = {}
        self.chunk2section_map = {r["chunk_id"]: r["chunk_metadata"]["doc_title"] for r in extraction_results}
        self.section_entities_collection = {}

        base = os.path.join(self.config.storage.knowledge_graph_path, "plug_in")
        os.makedirs(base, exist_ok=True)

        for result in extraction_results:
            md = result.get("chunk_metadata", {}) or {}
            label = md.get("doc_title", md.get("subtitle", md.get("title", "")))
            if label not in self.section_entities_collection:
                self.section_entities_collection[label] = []

            # 遍历当前 chunk 的实体
            for ent_data in result.get("entities", []):
                t = ent_data.get("type", "")
                name = ent_data.get("name")
                if not name or t not in self.entity_white_list:
                    continue

                # ---- 角色处理：不建节点，只映射 ----
                if t == "Character":
                    char_id = self.character_name2id.get(name)
                    if not char_id:
                        # 如果需要可以打印 warn
                        # print(f"[WARN] Character {name} 未在Neo4j找到，跳过创建")
                        pass
                    continue

                # ---- 冲突处理：local实体或 action-like 实体重命名 ----
                is_action_like = t in {"Action", "Emotion", "Goal"}
                if (ent_data.get("scope", "").lower() == "local" or is_action_like) and name in entity_map:
                    existing_entity = entity_map[name]
                    existing_chunk_id = existing_entity.source_chunks[0]
                    existing_section_name = self.chunk2section_map[existing_chunk_id]
                    current_section_name = md.get("doc_title", "")
                    if current_section_name != existing_section_name:
                        new_name = f"{name}_in_{label}"
                        suffix = 1
                        while new_name in entity_map:
                            suffix += 1
                            new_name = f"{name}_in_{label}_{suffix}"
                        ent_data["name"] = new_name
                        name = new_name

                # ---- 物品处理：尝试对齐已有 Object ----
                if t in self.entity_white_list and t != "Character":
                    obj_id = self.object_name2id.get(name)
                    if obj_id:
                        self.item_name2object_id[name] = obj_id
                        continue  # 已对齐，不需要新建节点

                # ---- 创建 / 合并 ----
                ent_obj = self._create_entity_from_data(ent_data, result["chunk_id"])
                # print("[CHECK] 创建实体: ", ent_obj)
                if not ent_obj:
                    continue
                existing = self._find_existing_entity(ent_obj, entity_map)
                if existing:
                    self._merge_entities(existing, ent_obj)
                else:
                    entity_map[ent_obj.name] = ent_obj

                self.section_entities_collection[label].append(ent_obj)

        # 存一份 section_entities_collection
        output_path = os.path.join(base, "section_entities_collection.pkl")
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
    
    def _merge_types(self, a, b) -> List[str]:
        """
        把 a 和 b 的类型并集后做规范化（Event 优先 + 去重保序）
        始终返回 List[str]；若为空则返回 ['Concept']。
        """
        a_list = self._to_type_list(a)
        b_list = self._to_type_list(b)

        merged, seen = [], set()
        for x in a_list + b_list:
            if x and x not in seen:
                seen.add(x)
                merged.append(x)

        if "Event" in seen:
            merged = ["Event"] + [x for x in merged if x != "Event"]

        return merged or ["Concept"]



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
                scope="local",  
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
    
    def _create_entity_from_data(self, data: Dict, chunk_id: str) -> Optional[Entity]:
        """
        - 允许的类型：self.entity_white_list（已从 schema 注入）
        - 若所有类型都不在白名单：返回 None
        - **保留多类型（List[str]）**
        """
        name = (data.get("name") or "").strip()
        if not name:
            return None

        types_raw = data.get("type")
        types_all = self._to_type_list(types_raw)
        types_in  = [x for x in types_all if x in self.entity_white_list]
        if not types_in:
            return None

        return Entity(
            id=f"ent_{hash(name) % 1_000_000}",   # 你原来的稳定 id 方案，保留
            name=name,
            type=types_in,                         # ← 保留多类型
            scope=data.get("scope", "local"),
            description=data.get("description", ""),
            aliases=data.get("aliases", []),
            source_chunks=[chunk_id],
            properties=self._to_props_dict(data.get("properties", {})),  # 若抽取里已带属性也不丢
        )




    def _create_relation_from_data(
        self,
        d: Dict,
        chunk_id: str,
        ent_by_name: Dict[str, Entity]
    ) -> Optional[Relation]:
        # 兼容多种键名
        subj = d.get("subject") or d.get("source") or d.get("head") or d.get("from_entity")
        obj  = d.get("object")  or d.get("target") or d.get("tail") or d.get("to_entity")
        pred = d.get("predicate") or d.get("relation") or d.get("relation_type") or d.get("type")

        if not subj or not obj or not pred:
            return None
        if pred not in self.relation_white_list:
            return None

        # 源/目标ID解析顺序：抽取批内新实体 > 既有角色 > 已对齐的Object > 既有Object
        def _resolve_id(name: str) -> Optional[str]:
            if name in ent_by_name:
                return ent_by_name[name].id
            if name in self.character_name2id:
                return self.character_name2id[name]
            if name in self.item_name2object_id:
                return self.item_name2object_id[name]
            if name in self.object_name2id:
                return self.object_name2id[name]
            return None

        sid = _resolve_id(subj)
        oid = _resolve_id(obj)
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
            source_chunks=[chunk_id],     # ← 固定用当前 chunk_id
        )

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

    def _store_knowledge_graph(self, verbose: bool):
        try:
            # self.graph_store.reset_knowledge_graph()
            self.graph_store.store_knowledge_graph(self.kg)
        except Exception as e:
            if verbose:
                print(f"⚠️ 存储失败: {e}")


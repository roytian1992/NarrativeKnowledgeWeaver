# kag/builder/graph_builder.py

"""
知识图谱构建器主模块 - 集成新增功能

整合信息抽取、数据处理和存储功能，支持优化的剧本处理策略
"""
from copy import deepcopy
import time
from typing import List, Dict, Any, Optional
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import json
from pathlib import Path
from ..models.entities import KnowledgeGraph, Entity, Relation, Document, TextChunk

from ..utils.config import KAGConfig
from ..utils.format import correct_json_format
from .processor import DocumentProcessor
# from .extractor import InformationExtractor
from ..storage.graph_store import GraphStore
from ..storage.document_store import DocumentStore
from ..storage.vector_store import VectorStore
import pandas as pd
import sqlite3
import pickle
from kag.llm.llm_manager import LLMManager
# from kag.builder.reflection import DynamicReflector
from kag.agent.kg_extraction_agent import InformationExtractionAgent
from kag.agent.attribute_extraction_agent import AttributeExtractionAgent
from dataclasses import asdict 
from kag.utils.neo4j_utils import Neo4jUtils
# from ..schema.kg_schema import ENTITY_TYPES, RELATION_TYPE_GROUPS
import os


class KnowledgeGraphBuilder:
    """知识图谱构建器 - 集成新增功能"""
    
    def __init__(self, config: KAGConfig):
        self.config = config
        # self.reset()
        self.llm_manager = LLMManager(config)
        self.llm = self.llm_manager.get_llm()
        self.graph_store = GraphStore(config)
        self.neo4j_utils = Neo4jUtils(self.graph_store.driver)
        self.vector_store = VectorStore(config)
        self.document_store = DocumentStore(config)
        self.kg = KnowledgeGraph()
        self.max_workers = 32
        self.load_schema("kag/schema/graph_schema.json")
        self.load_abbreviations("kag/schema/settings_schema.json")
        self.processor = DocumentProcessor(config, self.llm)
        self.information_extraction_agent = InformationExtractionAgent(config, self.llm)
        self.attribute_extraction_agent = AttributeExtractionAgent(config, self.llm)
        
    def load_abbreviations(self, path):
        """从JSON文件加载缩写列表，返回格式化后的文本（适合插入提示词）"""
        with open(path, 'r', encoding='utf-8') as f:
            abbr = json.load(f)
        abbr_list = abbr.get("abbreviations", [])

        formatted = []
        for item in abbr_list:
            line = f"- **{item['abbr']}**: {item['full']}（{item['zh']}） - {item['description']}"
            formatted.append(line)
        self.abbreviation_info = "\n".join(formatted)


    def load_schema(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            schema = json.load(f)
        self.entity_types = schema.get("entities")
        self.relation_type_groups = schema.get("relations")

        self.entity_type_description_text = "\n".join(
            f"- {item['type']}: {item['description']}" for item in self.entity_types
        )

        self.relation_type_description_text = "\n".join(
            f"- {item['type']}: {item['description']}"
            for group in self.relation_type_groups.values()
            for item in group
        )

        RELATION_TYPES = []
        for group in self.relation_type_groups.values():
            RELATION_TYPES.extend(group)

        print("✅ 成功加载知识图谱模式")


    def reset(self):
        path = Path(self.config.storage.knowledge_graph_path)
        for json_file in path.glob("*.json"):
            json_file.unlink()  # 删除文件


    def prepare_chunks(self, json_file_path: str, verbose: bool = True) -> Dict[str, Any]:
        """从JSON文件构建知识图谱前的处理和信息抽取，拆分构图逻辑为独立步骤"""

        if verbose:
            print(f"🚀 开始构建知识图谱: {json_file_path}")

        # 1. 加载文档
        if verbose:
            print("📖 加载文档...")
        documents = self.processor.load_from_json(json_file_path)
        if verbose:
            print(f"✅ 成功加载 {len(documents)} 个文档")

        # 2. 拆分文档
        all_description_chunks = []
        all_conversation_chunks = []
        
        for doc in tqdm(documents, total=len(documents), desc="文本拆分中"):
            chunk_groups = self.processor.prepare_chunk(doc)
            all_description_chunks.extend(chunk_groups["description_chunks"])
            all_conversation_chunks.extend(chunk_groups["conversation_chunks"])
            
        # 3. 存储文本块
        base_path = self.config.storage.knowledge_graph_path
        with open(os.path.join(base_path, "all_description_chunks.json"), "w", encoding="utf-8") as f:
            json.dump([chunk.dict() for chunk in all_description_chunks], f, ensure_ascii=False, indent=2)
        with open(os.path.join(base_path, "all_conversation_chunks.json"), "w", encoding="utf-8") as f:
            json.dump([chunk.dict() for chunk in all_conversation_chunks], f, ensure_ascii=False, indent=2)

        if verbose:
            print(f"✅ 生成 {len(all_description_chunks)} 个剧本文本块")

    def store_chunks(self, verbose: bool = True) -> None:
        # 4. 存储对话信息到关系数据库
        self.vector_store.delete_collection()
        self.vector_store._initialize()
        base_path = self.config.storage.knowledge_graph_path
        with open(os.path.join(base_path, "all_description_chunks.json"), "r", encoding="utf-8") as f:
            description_data = json.load(f)
        with open(os.path.join(base_path, "all_conversation_chunks.json"), "r", encoding="utf-8") as f:
            conversation_data = json.load(f)
            
        all_description_chunks = [TextChunk(**chunk) for chunk in description_data]
        all_conversation_chunks = [TextChunk(**chunk) for chunk in conversation_data]

        if verbose:
            print("💾 存储到关系数据库...")
        for chunk in all_description_chunks:
            self.kg.add_document(self.processor.prepare_document(chunk))
            self.kg.add_chunk(chunk)
            
        self._build_relational_database(all_conversation_chunks)

        # 5. 存储文档信息到向量数据库
        if verbose:
            print("💾 存储到向量数据库...")
        self._store_vectordb(verbose)
    
    def extract_entity_and_relation(self, verbose: bool = True) -> List[Dict[str, Any]]:
        """抽取实体与关系信息，保存为 extraction_results.json"""

        base_path = self.config.storage.knowledge_graph_path
        with open(os.path.join(base_path, "all_description_chunks.json"), "r", encoding="utf-8") as f:
            all_description_chunks = json.load(f)

        all_description_chunks = [TextChunk(**chunk) for chunk in all_description_chunks]

        if verbose:
            print("🧠 实体与关系信息抽取中...")

        extraction_results = self._kg_extraction_multithread(all_description_chunks, self.max_workers)
        extraction_results = [r for r in extraction_results if r is not None]

        with open(os.path.join(base_path, "extraction_results.json"), "w", encoding="utf-8") as f:
            json.dump(extraction_results, f, ensure_ascii=False, indent=2)

        if verbose:
            print(f"✅ 实体与关系信息抽取完成，共处理 {len(extraction_results)} 个文本块")

        return extraction_results
    
    def extract_entity_attributes(self, verbose: bool = True) -> Dict[str, Entity]:
        """基于已有实体抽取结果，抽取属性并保存为 entity_info.json"""

        base_path = self.config.storage.knowledge_graph_path
        with open(os.path.join(base_path, "extraction_results.json"), "r", encoding="utf-8") as f:
            extraction_results = json.load(f)

        # 合并并去重实体
        entity_map = self.merge_entities_info(extraction_results)

        if verbose:
            print("🔎 属性抽取中...")

        entity_map = self._attribute_extraction_multithread(entity_map, self.max_workers)

        # 保存
        entity_map_ = {k: v.dict() for k, v in entity_map.items()}
        with open(os.path.join(base_path, "entity_info.json"), "w", encoding="utf-8") as f:
            json.dump(entity_map_, f, ensure_ascii=False, indent=2)

        if verbose:
            print(f"✅ 属性抽取完成，共处理实体 {len(entity_map)} 个")

        return entity_map


    def build_graph_from_results(self, verbose: bool = True) -> KnowledgeGraph:
        """从抽取结果文件构建知识图谱并写入图数据库"""

        if verbose:
            print("📂 加载已有抽取结果和实体信息...")

        base_path = self.config.storage.knowledge_graph_path

        # 加载抽取结果
        extraction_file = os.path.join(base_path, "extraction_results.json")
        with open(extraction_file, "r", encoding="utf-8") as f:
            extraction_results = json.load(f)

        # 加载实体信息
        entity_file = os.path.join(base_path, "entity_info.json")
        with open(entity_file, "r", encoding="utf-8") as f:
            entity_info_raw = json.load(f)

        # 重构实体对象并创建 id->Entity 映射
        entity_map = {
            data["id"]: Entity(**data)
            for data in entity_info_raw.values()
        }
        
        # print("***: ", entity_info_raw)

        name_to_id = {}
        for entity in entity_map.values():
            name_to_id[entity.name] = entity.id
            for alias in entity.aliases:
                if alias not in name_to_id:
                    name_to_id[alias] = entity.id


        # 构建图谱
        if verbose:
            print("🔗 构建知识图谱...")
        self._build_knowledge_graph(entity_map, extraction_results, name_to_id, verbose)

        # 存储图谱
        if verbose:
            print("💾 存储到数据库...")
        self._store_knowledge_graph(verbose)

        # 打印统计信息
        if verbose:
            stats = self.kg.stats()
            print(f"🎉 知识图谱构建完成!")
            print(f"   - 实体数量: {stats['entities']}")
            print(f"   - 关系数量: {stats['relations']}")
            print(f"   - 文档数量: {stats['documents']}")
            print(f"   - 文本块数量: {stats['chunks']}")

            num_scene = sum(1 for r in self.kg.relations.values() if r.predicate == "SCENE_CONTAINS")
            num_other = sum(1 for r in self.kg.relations.values() if r.predicate != "SCENE_CONTAINS")
            print(f"   - SCENE_CONTAINS 关系数: {num_scene}")
            print(f"   - 其他实体关系数: {num_other}")

        return self.kg
        

    
    def _kg_extraction(self, chunks: List, verbose: bool) -> List[Dict]:
        """并行信息抽取（增强版）：支持反思与低分重抽 + 得分最优保存"""
        extraction_results = []
    
        for chunk in tqdm(chunks):
            content =   chunk.content
            result = self.information_extraction_agent.run(content)
            result["chunk_id"] = chunk.id
            result["scene_metadata"] = chunk.metadata
            extraction_results.append(result)
            
        return extraction_results

    def _kg_extraction_multithread(self, chunks: List, max_workers: int = 8) -> List[Dict]:
        """并行信息抽取（增强版）：支持反思与低分重抽 + 得分最优保存 + 并发加速"""
        extraction_results = []

        def process_chunk(chunk):
            if len(chunk.content.strip()) > 0:
                result = self.information_extraction_agent.run(chunk.content)
            else:
                result = {
                    "entities": [],
                    "relations": [],
                    "suggestions": [],
                    "issues": [],
                    "score": 0,
                }
            result["chunk_id"] = chunk.id
            result["scene_metadata"] = chunk.metadata
            return result

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_chunk, chunk) for chunk in chunks]
            for future in tqdm(as_completed(futures), total=len(futures), desc="并发抽取中"):
                extraction_results.append(future.result())

        return extraction_results


    def _attribute_extraction(self, entity_map: Dict[str, Entity]) -> Dict[str, Entity]:
        new_entity_map = {}

        for entity_name, entity in tqdm(entity_map.items(), desc="属性抽取中（串行）"):
            entity_type = entity.type.name
            text = entity.description or ""

            if not text.strip():
                continue

            try:
                result = self.attribute_extraction_agent.run(
                    text=text,
                    entity_name=entity_name,
                    entity_type=entity_type,
                    original_text=""
                )
                attributes = result.get("attributes", {})
                if isinstance(attributes, str):
                    try:
                        attributes = json.loads(attributes)
                    except json.JSONDecodeError:
                        print(f"[ERROR] 无法解析 JSON: {attributes}")
                        attributes = {}

                # 如果是 list，则取第一个（保守处理）
                if isinstance(attributes, list):
                    if attributes:
                        attributes = attributes[0]
                    else:
                        attributes = {}
                    
                new_entity = deepcopy(entity)
                new_entity.properties = attributes
                new_entity.description = ""
                new_entity_map[entity_name] = new_entity

            except Exception as e:
                print(f"[ERROR] 抽取失败：{entity_name} - {e}")

        return new_entity_map


    def _attribute_extraction_multithread(self, entity_map: Dict[str, Entity], max_workers: int = 8) -> Dict[str, Entity]:
        new_entity_map = {}

        def process(entity_name, entity):
            entity_type = entity.type
            # print("[CHECK] entity_type：", entity_type)
            source_chunks = entity.source_chunks
            text = entity.description or ""
            if not text.strip():
                return entity_name, None  # 空内容跳过

            try:
                result = self.attribute_extraction_agent.run(
                    text=text,
                    entity_name=entity_name,
                    entity_type=entity_type,
                    source_chunks=source_chunks,
                    original_text=""
                )
                attributes = result.get("attributes", {})
                # description = result.get("description", "")
                # print("[CHECK] description: ", result)
                if "new_description" not in result:
                    print("[CHECK] result: ", result)
                description = result.get("new_description", "")
                
                # print("[CHECK] result: ", result)
                # print("[CHECK] 新的描述: ", description)
                if isinstance(attributes, str):
                    attributes = json.loads(attributes)

                new_entity = deepcopy(entity)
                new_entity.properties = attributes
                # if new_entity.type == "Event":
                #     print("[CHECK] attributes: ", attributes)
                if description:
                    new_entity.description = description
                return entity_name, new_entity

            except Exception as e:
                print(f"[ERROR] 抽取失败：{entity_name} - {e}")
                return entity_name, None

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(process, name, entity)
                for name, entity in entity_map.items()
            ]
            for future in tqdm(as_completed(futures), total=len(futures), desc="属性抽取中（并发）"):
                entity_name, updated_entity = future.result()
                if updated_entity:
                    new_entity_map[entity_name] = updated_entity

        return new_entity_map

    
    def _build_relational_database(self, conversation_chunks: List):
        conversation_data = []
        for item in conversation_chunks:
            conversation_data.append({
                "id": item.id,
                "content": item.content.split("：")[-1].strip(),
                "character": item.metadata["character"].strip(),
                "type": item.metadata.get("type") or "regular",
                "remark": "，".join(item.metadata.get("remark", [])),
                "scene_number": item.metadata.get("scene_number"),
                "sub_scene_number": item.metadata.get("sub_scene_number"),
            })

        df = pd.DataFrame(conversation_data)
        db_path = os.path.join(self.config.storage.sql_database_path, "conversations.db")
        if os.path.exists(db_path):
            os.remove(db_path)

        connection = sqlite3.connect(db_path)
        df.to_sql('人物对话', connection, if_exists='replace', index=False)

    
    def _build_knowledge_graph(
        self,
        entity_map: Dict[str, Entity],
        extraction_results: List[Dict],
        name_to_id: Dict[str, str],
        verbose: bool = True
    ):
        if verbose:
            print("🔗 正在构建知识图谱...")

        # 1. 添加实体
        for ent in entity_map.values():
            self.kg.add_entity(ent)

        # 2. 遍历每个 chunk，构建场景与普通关系
        for result in extraction_results:
            chunk_id = result["chunk_id"]

            # 场景实体
            scene_entities = self._create_scene_entities(result.get("scene_metadata", {}), chunk_id)
            for se in scene_entities:
                if se.name not in name_to_id:
                    name_to_id[se.name] = se.id
                    entity_map[se.id] = se
                self.kg.add_entity(se)

            # Scene → contains → inner entities
            inner_entity_objs = [
                entity_map[name_to_id[e_data["name"]]]
                for e_data in result.get("entities", [])
                if e_data["name"] in name_to_id
            ]
            for scene_ent in scene_entities:
                self._link_scene_to_entities(scene_ent, inner_entity_objs, chunk_id)

            # 普通实体关系
            for r_data in result.get("relations", []):
                rel = self._create_relation_from_data(r_data, chunk_id, entity_map, name_to_id)
                # if not rel:
                #     print("[CHECK] r_data: ", r_data)
                if rel:
                    self.kg.add_relation(rel)


    def merge_entities_info(self, extraction_results):
        entity_map = {}  # 用于实体去重和合并
        for result in extraction_results:
            scene_md = result.get("scene_metadata", {})
            if scene_md.get("sub_scene_number"):
                play_name = f"场景{scene_md.get('scene_number')}-{scene_md.get('sub_scene_number')}"
            else:
                play_name = f"场景{scene_md.get('scene_number')}"
            
            # 处理基础实体
            for entity_data in result.get("entities", []):
                if entity_data.get("scope").lower()=="local" and entity_data["name"] in entity_map:
                # 在已有名字前加场景前缀；如前缀已存在则再追加计数
                    new_name = f"{play_name}中的{entity_data['name']}"
                    suffix = 1
                    while new_name in entity_map:        # 仍冲突就加 _n
                        suffix += 1
                        new_name = f"{play_name}中的{entity_data['name']}_{suffix}"
                    entity_data["name"] = new_name
                
                entity = self._create_entity_from_data(entity_data, result["chunk_id"])
                existing_entity = self._find_existing_entity(entity, entity_map)
                if existing_entity:
                    self._merge_entities(existing_entity, entity)
                else:
                    entity_map[entity.name] = entity
        return entity_map
    
    def _create_scene_entities(
            self,
            scene_metadata: Dict[str, Any],
            chunk_id: str
    ) -> List[Entity]:
        """仅创建场景实体（不再生成地点实体）"""
        entities = []
        if scene_metadata.get("sub_scene_number", ""):
            play_name = f"场景{scene_metadata.get("scene_number")}-{scene_metadata.get("sub_scene_number")}"
        else:
            play_name = f"场景{scene_metadata.get("scene_number")}"
        
        if play_name:
            scene_entity = Entity(
                id=f"scene_{hash(play_name) % 1_000_000}",
                name=play_name,
                type="Scene",                           # 直接字符串
                description=f"属于场景: {scene_metadata.get("scene_name", "")}",
                properties=scene_metadata,             # 挂全部元数据
                source_chunks=[chunk_id],
            )
            entities.append(scene_entity)

        return entities

    
    def _create_entity_from_data(self, entity_data: Dict, chunk_id: str) -> Entity:
        """从数据创建实体"""
        entity_type = entity_data.get("type", "Concept")

        return Entity(
            id=f"entity_{hash(entity_data['name']) % 1000000}",
            name=entity_data["name"],
            type=entity_type,
            description=entity_data.get("description", ""),
            aliases=entity_data.get("aliases", []),
            source_chunks=[chunk_id]
        )
    
    def _create_relation_from_data(
        self,
        relation_data: Dict,
        chunk_id: str,
        entity_map: Dict[str, Entity],
        name_to_id: Dict[str, str]
    ) -> Optional[Relation]:
        """从数据创建关系"""
        subject_name = (
            relation_data.get("subject")
            or relation_data.get("source")
            or relation_data.get("head")
            or relation_data.get("head_entity")
        )
        object_name = (
            relation_data.get("object")
            or relation_data.get("target")
            or relation_data.get("tail")
            or relation_data.get("tail_entity")
        )
        predicate = (
            relation_data.get("predicate")
            or relation_data.get("relation")
            or relation_data.get("relation_type")
        )
        
        if not subject_name or not object_name or not predicate:
            return None


        subject_id = name_to_id.get(subject_name)
        object_id = name_to_id.get(object_name)

        if not subject_id:
            print("[CHECK] subject: ", subject_name, predicate, object_name)
            
        if not object_id:
            print("[CHECK] object: ", subject_name, predicate, object_name)
            
        if not subject_id or not object_id:
             return None

        relation_id_str = f"{subject_id}_{predicate}_{object_id}"
        return Relation(
            id=f"rel_{hash(relation_id_str) % 1000000}",
            subject_id=subject_id,
            predicate=predicate,
            object_id=object_id,
            properties={
                "description": relation_data.get("description", ""),
                "relation_name": relation_data.get("relation_name", "")
            },
            source_chunks=[chunk_id]
        )

    
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

    
    def _link_scene_to_entities(
            self,
            scene_entity: Entity,
            inner_entities: List[Entity],
            chunk_id: str
    ) -> None:
        """
        为当前场景实体 scene_entity 与其内部实体 inner_entities
        创建 "SCENE_CONTAINS" 关系并写入 self.kg
        """
        for target in inner_entities:
            rel_id = f"{scene_entity.id}_scene_contains_{target.id}"
            relation = Relation(
                id=f"rel_{hash(rel_id) % 1_000_000}",
                subject_id=scene_entity.id,
                object_id=target.id,
                predicate="SCENE_CONTAINS",
                properties={},
                source_chunks=[chunk_id],
                # confidence=1.0,
            )
            self.kg.add_relation(relation)


    def _store_vectordb(self, verbose: bool):
        try:
            if verbose:
                print("   - 存储到向量数据库...")
            self.vector_store.delete_collection()
            self.vector_store._initialize()
            self.vector_store.store_documents(list(self.kg.documents.values()))
        except Exception as e:
            if verbose:
                print(f"⚠️ 存储失败: {str(e)}")



    def _store_knowledge_graph(self, verbose: bool) -> None:
        """存储知识图谱到数据库"""
        try:
            if verbose:
                print("   - 存储到Neo4j...")
            self.graph_store.store_knowledge_graph(self.kg)
        except Exception as e:
            if verbose:
                print(f"⚠️ 存储失败: {str(e)}")
                
    def prepare_graph_embeddings(self):
        self.neo4j_utils.load_emebdding_model(self.config.memory.embedding_model_name)
        self.neo4j_utils.create_vector_index()
        self.neo4j_utils.process_all_embeddings(exclude_node_types=["Scene"], exclude_rel_types=["SCENE_CONTAINS"])
        self.neo4j_utils.ensure_entity_superlabel()
        print("✅ 图向量构建完成")

    def search_entities(self, query: str, limit: int = 10) -> List[Entity]:
        return self.graph_store.search_entities(query, limit)
    
    def search_relations(self, entity_name: str, limit: int = 10) -> List[Relation]:
        return self.graph_store.search_relations(entity_name, limit)
    
    def semantic_search(self, query: str, limit: int = 5) -> List[Document]:
        return self.vector_store.search(query, limit)
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            "knowledge_graph": self.kg.stats(),
            "graph_store": self.graph_store.get_stats(),
            "vector_store": self.vector_store.get_stats()
        }


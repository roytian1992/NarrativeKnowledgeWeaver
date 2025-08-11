import json
import os
from typing import List, Dict, Any, Optional

from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm
from itertools import chain

from ..utils.config import KAGConfig
# from kag.functions.regular_functions import MetadataParser, SemanticSplitter
from kag.builder.document_parser import DocumentParser
from kag.utils.format import correct_json_format, safe_text_for_json
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import numpy as np
from collections import defaultdict
from kag.utils.format import correct_json_format


def compute_weighted_similarity_and_laplacian(entity_dict, alpha=0.8, knn_k=40, top_k=10):
    names = list(entity_dict.keys())
    name_embs = np.vstack([entity_dict[n]['name_embedding'] for n in names])
    desc_embs = np.vstack([entity_dict[n]['description_embedding'] for n in names])

    # 分别计算 name 和 description 的相似度
    sim_name = cosine_similarity(name_embs)
    sim_desc = cosine_similarity(desc_embs)
    
    # 加权融合
    sim = alpha * sim_name + (1 - alpha) * sim_desc
    
    # 构建邻接矩阵（KNN图）
    n = sim.shape[0]
    adj = np.zeros((n, n))
    for i in range(n):
        idx = np.argsort(sim[i])[-(knn_k+1):-1]  # 排除自己
        adj[i, idx] = sim[i, idx]
    adj = np.maximum(adj, adj.T)  # 对称化

    # 构建图拉普拉斯矩阵
    deg = np.diag(adj.sum(axis=1))
    lap = deg - adj

    # 特征值计算
    eigvals = np.linalg.eigvalsh(lap)
    gaps = np.diff(eigvals)
    # print("gaps: ", gaps[0], gaps[1])
    estimated_k = int(np.argmax(gaps[1:]) + 1)  # 跳过第一个gap
    
    return estimated_k, sim

    
def run_kmeans_clustering(entity_dict, n_clusters, alpha=0.8):
    """
    使用 KMeans 对实体聚类（支持 name/desc embedding 加权拼接）
    """
    names = list(entity_dict.keys())
    name_embs = np.vstack([entity_dict[n]['name_embedding'] for n in names])
    desc_embs = np.vstack([entity_dict[n]['description_embedding'] for n in names])

    # 加权拼接
    combined_embs = np.hstack([
        name_embs * alpha,
        desc_embs * (1 - alpha)
    ])

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(combined_embs)

    cluster_result = defaultdict(list)
    for name, label in zip(names, labels):
        cluster_result[label].append(name)

    clusters = dict(cluster_result)
    collected_clusters = []
    for label, group in clusters.items():
        if len(group) >= 2:
            # print(f"\n📦 Cluster {label}:")
            collected_clusters.append(group)
    return collected_clusters


class GraphPreprocessor:
    """通用文档处理器"""

    # ------------------------------------------------------------------ #
    # 初始化
    # ------------------------------------------------------------------ #
    def __init__(self, config: KAGConfig, llm, system_prompt):
        self.config = config        
        self.system_prompt_text = system_prompt
        
        self.document_parser = DocumentParser(config, llm)
        self.model = self.load_embedding_model(config.memory.embedding_model_name)
        self.max_worker = 16
        # self.rename = dict()
            
    def load_embedding_model(self, model_name):
        if self.config.embedding.provider == "openai":
            from kag.model_providers.openai_embedding import OpenAIEmbeddingModel
            model = OpenAIEmbeddingModel(self.config)
        else:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(model_name)
        return model
        
    def collect_global_entities(self, extraction_results):                
        global_entities = dict() # collect global entities by type
        for result in extraction_results:
            entities = result["entities"]
            for entity in entities:
                if entity["scope"] == "global" and entity["type"] in ["Character", "Object", "Concept", "Event"]:
                    if entity["type"] in global_entities:
                        global_entities[entity["type"]].append(entity)
                    else:
                        global_entities[entity["type"]] =[entity]
         
        merged_global_entities = dict()
        for type in global_entities:
            filtered_entities = dict()
            for entity in global_entities[type]:
                if entity["name"] in filtered_entities:
                    filtered_entities[entity["name"]]["description"] += entity["description"]
                else:
                    filtered_entities[entity["name"]] = entity
                merged_global_entities[type] = filtered_entities
    
        return merged_global_entities
    
    def compute_embeddings(self, filtered_entities):
        # print("[CHECK] filtered_entities: ", type(filtered_entities))
        for entity in tqdm(filtered_entities):
            # entity = filtered_entities[entity_name]
            name_embedding = self.model.encode(entity["name"])
            description_embedding = self.model.encode(entity.get("summary", entity["description"])) # 优先使用summary
            entity["name_embedding"] = name_embedding
            entity["description_embedding"] = description_embedding
        return filtered_entities        

    def add_entity_summary(self, merged_global_entities):
        """
        使用多线程并发生成实体摘要（若 description 足够长），更新 merged_global_entities。
        """
        # 展开 entity 列表（方便并发处理）
        entity_list = []
        for type in merged_global_entities:
            for entity in merged_global_entities[type].values():
                entity_list.append(entity)

        # 定义处理函数
        def summarize_entity(entity):
            try:
                if len(entity["description"]) >= 300:
                    result = self.document_parser.summarize_paragraph(
                        text=entity["description"], max_length=250
                    )
                    result = json.loads(correct_json_format(result))
                    summary = result["summary"]
                else:
                    summary = entity["description"]
                entity["summary"] = summary
            except Exception as e:
                entity["summary"] = entity["description"]  # fallback
                print(f"❗摘要失败: {entity['name']} -> {str(e)}")
            return entity

        # 多线程并发处理
        with ThreadPoolExecutor(max_workers=self.max_worker) as executor:
            futures = [executor.submit(summarize_entity, entity) for entity in entity_list]
            entity_list_updated = [future.result() for future in tqdm(as_completed(futures), total=len(futures), desc="生成摘要")]

        # 重新按类型聚合（返回结构一致）
        merged_global_entities_new = dict()
        for entity in entity_list_updated:
            if entity["type"] in merged_global_entities_new:
                merged_global_entities_new[entity["type"]].append(entity)
            else:
                merged_global_entities_new[entity["type"]] = [entity]

        return merged_global_entities_new

    # def add_entity_summary(self, merged_global_entities):
        
    #     # 展开，方便多线程并发
    #     entity_list = []
    #     for type in merged_global_entities:
    #         for entities in merged_global_entities[type]:
    #             entity_list.extend(entities)
                
    #     for entity in entity_list: # 转成多线程并发：
    #         if len(entity["description"]) >= 300:
    #             result = self.document_parser.paragraph_summarizer(text=entity["description"], max_length=250)
    #             result = json.loads(correct_json_format(result))
    #             summary = result["summary"]
    #         else:
    #             summary = entity["description"]
    #         entity["summary"] = summary
        
    #     merged_global_entities_new = dict() # 重新合并
    #     for entity in entity_list:
    #         if entity["type"] in merged_global_entities_new:
    #             merged_global_entities_new[entity["type"]].append(entity)
    #         else:
    #             merged_global_entities_new[entity["type"]] = [entity]
                
    #     return merged_global_entities_new
           
    def detect_candidates(self, merged_global_entities):
        candidates = []
        for type in merged_global_entities:
            filtered_entities = dict()
            for entity in merged_global_entities[type].copy():
                if entity["name"] in filtered_entities:
                    filtered_entities[entity["name"]]["description"] += entity["description"]
                else:
                    filtered_entities[entity["name"]] = entity
                    
            knn_k = min(int(len(filtered_entities)/4) ,25) 
            estimated_k, sim_matrix = compute_weighted_similarity_and_laplacian(filtered_entities, alpha=0.8, knn_k=25)
            n_clusters=int((estimated_k+len(filtered_entities)/2)/2)
            collected_clusters = run_kmeans_clustering(
                filtered_entities,
                n_clusters=n_clusters,
                alpha=0.5
            )
            candidates.extend(collected_clusters)
            
        return candidates
    
    # def merge_entities(self, all_candidates_with_info):
    #     rename_map = dict()
    #     for candidate in all_candidates_with_info:
    #         entity_descriptions = ""
    #         for i, entity in enumerate(candidate):
    #             entity_name = entity["name"]
    #             entity_summary = entity.get("summary", entity["description"])
    #             entity_descriptions += f"实体{i+1}的名称：{entity_name}\n{entity_summary}\n"
            
    #         result = self.document_parser.merge_entities(entity_descriptions=entity_descriptions, system_prompt=self.system_prompt_text)
    #         result = json.loads(correct_json_format(result))
    #         merges = result["merges"]
    #         unmerged = result["unmerged"]
    #         for merge in merges:
    #             for alias in merge["aliases"]:
    #                 rename_map[alias] = merge["canonical_name"]
        
    #     return rename_map
    
    def merge_entities(self, all_candidates_with_info):
        """
        并发调用 LLM 合并判断，返回 alias → canonical_name 的重命名映射表
        """
        rename_map = dict()

        # 单个候选组处理逻辑
        def process_group(candidate_group):
            try:
                entity_descriptions = ""
                for i, entity in enumerate(candidate_group):
                    entity_name = entity["name"]
                    entity_summary = entity.get("summary", entity["description"])
                    entity_descriptions += f"实体{i+1}的名称：{entity_name}\n{entity_summary}\n"

                result = self.document_parser.merge_entities(
                    entity_descriptions=entity_descriptions,
                    system_prompt=self.system_prompt_text
                )
                result = json.loads(correct_json_format(result))
                return result  # 返回完整结果结构
            except Exception as e:
                print(f"❗实体合并失败: {[e['name'] for e in candidate_group]} -> {e}")
                return {"merges": [], "unmerged": []}

        # 并发执行
        with ThreadPoolExecutor(max_workers=self.max_worker) as executor:
            futures = [executor.submit(process_group, group) for group in all_candidates_with_info]
            results = [future.result() for future in tqdm(as_completed(futures), total=len(futures), desc="实体合并判断")]

        # 聚合重命名映射
        for result in results:
            for merge in result.get("merges", []):
                canonical = merge["canonical_name"]
                for alias in merge.get("aliases", []):
                    rename_map[alias] = canonical

        return rename_map
    
    def run_entity_disambiguation(self, extraction_results):
        merged_global_entities = self.collect_global_entities(extraction_results)
        merged_global_entities = self.add_entity_summary(merged_global_entities)
        
        for type in merged_global_entities:
            merged_global_entities[type] = self.compute_embeddings(merged_global_entities[type])
        
        all_candidates = self.detect_candidates(merged_global_entities)
        
        entity_info_map = dict()
        for type in merged_global_entities:
            entities = merged_global_entities[type]
            for entity in entities:
                #print("[CHECK] entity: ", entity)
                entity_info_map[entity["name"]] = entity
        
        all_candidates_with_info = []
        for candidates in all_candidates:
            group = []
            for entity in candidates:
                group.append(entity_info_map[entity])
            all_candidates_with_info.append(group)
        rename_map = self.merge_entities(all_candidates_with_info)
        
        base = self.config.storage.knowledge_graph_path
        os.makedirs(base, exist_ok=True)
        json.dump(rename_map,
                  open(os.path.join(base, "rename_map.json"), "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
        
        for result in extraction_results:
            for entity in result["entities"]:
                entity["name"] = rename_map.get(entity["name"], entity["name"])
                
            for relation in result["relations"]:
                relation["subject"] = rename_map.get(relation["subject"], relation["subject"])
                relation["object"] = rename_map.get(relation["object"], relation["object"])
        
        return extraction_results
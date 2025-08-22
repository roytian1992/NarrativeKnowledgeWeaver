from typing import Optional, List
from core.utils.config import KAGConfig
from core.memory.vector_memory import VectorMemory
from ..utils.format import correct_json_format
from core.model_providers.openai_rerank import OpenAIRerankModel
import json
import re

class DynamicReflector:
    def __init__(self, config: KAGConfig):
        self.config = config
        self.history_memory = VectorMemory(self.config, "history_memory")
        self.insight_memory = VectorMemory(self.config, "insight_memory")
        self.reranker = OpenAIRerankModel(self.config)
        self.entity_extraction_memory = dict()
        self.history_memory_size = self.config.memory.history_memory_size
        self.insight_memory_size = self.config.memory.insight_memory_size
    
    def clear(self):
        self.history_memory.clear()
        self.insight_memory.clear()
        
    def generate_logs(self, extraction_result):
        """
        根据抽取结果生成日志
        """
        logs = []

        # 1) 处理实体
        entities = extraction_result.get("entities", []) # 
        if not entities:
            logs.append("无可识别实体或者实体抽取失败，无相关日志。")
        else:
            for ent in entities:
                name = ent.get("name", "UNKNOWN")
                ent_type = ent.get("type", "UNKNOWN")
                desc = ent.get("description", "") or "空白"
                scope = ent.get("scope", "")
                logs.append(
                    f"实体『{name}』(类型: {ent_type})被抽取出来，相关描述为：{desc}。当前实体scope为：{scope}"
                )
                if name in self.entity_extraction_memory:
                    related_history = list(self.entity_extraction_memory[name])
                    for history in related_history:
                        logs.append(f"- 在之前的实体抽取结果检测到：{history}")

        # 2) 处理关系
        relations = extraction_result.get("relations", [])
        if not relations:
            logs.append("无可识别关系或关系抽取失败，无相关日志。")
        else:
            for rel in relations:
                subj = rel.get("subject", "UNKNOWN")
                obj = rel.get("object", "UNKNOWN")
                relation_type = rel.get("relation_type", "UNKNOWN")
                relation_name = rel.get('relation_name', "UNKNOWN")
                desc = rel.get("description", "") or "空白"
                logs.append(
                    f"实体『{subj}』与实体『{obj}』存在关系「{relation_name}」（类型为{relation_type}），相关描述为：{desc}"
                )

        return logs
      
    def _store_memory(self, content, reflections):
        insights = reflections.get("insights", []) 
        
        for item in insights:
            self.insight_memory.add(text=item, metadata={})
        
        sentences = re.split(r'(?<=[。！？])', content)
        
        entities = reflections.get("entities", [])
        relations = reflections.get("relations", [])
        score = reflections.get("score", 0)
        
        documents = dict()
        
        for entity in entities:
            entity_name = entity.get("name", "")
            entity_type = entity.get("type", "")
            entity_scope = entity.get("scope", "")
            
            matches = [s.strip() for s in sentences if entity_name in s]
            if entity_name and entity_type:
                if entity_name in self.entity_extraction_memory:
                    self.entity_extraction_memory[entity_name].add(f"抽取了实体: {entity_name} (实体类型: {entity_type}, scope: {entity_scope})\n")
                else:
                    self.entity_extraction_memory[entity_name] = set([f"抽取了实体: {entity_name} (实体类型: {entity_type}, scope: {entity_scope})\n"])
                    
                for match in matches:
                    if match in documents:
                        documents[match] += f"- 抽取了实体: {entity_name} (实体类型: {entity_type}, scope: {entity_scope})\n"
                    else:
                        documents[match] = f"- 抽取了实体: {entity_name} (实体类型: {entity_type}, scope: {entity_scope})\n"

        for relation in relations:
            subject = relation.get("subject", "")
            object_ = relation.get("object", "")
            relation_name = relation.get("relation_name", "")
            relation_type = relation.get("relation_type", "")
            matches = [s.strip() for s in sentences if subject in s and object_ in s]
            if subject and object_ and relation_name:
                for match in matches:
                    if match in documents:
                        documents[match] += f"- 抽取了关系: {subject}-{relation_name}->{object_} (关系类型: {relation_type})\n"
                    else:
                        documents[match] = f"- 抽取了关系: {subject}-{relation_name}->{object_} (关系类型: {relation_type})\n"

        for match in documents:
            documents[match] += f"当前抽取得分为{score}"
            self.history_memory.add(text=match, metadata={"history": documents[match]})
            

    def _search_relevant_reflections(self, context):
        def create_record(content, history):
            return f"原文如下：\n{content}\n{history}"
        
        sentences = re.split(r'(?<=[。！？])', context)
        related_history = []
        for sentence in sentences:
            if len(sentence) > 500:
                sentence = sentence[:500]
            results = self.history_memory.get(sentence, self.history_memory_size)
            retrieved_docs = [create_record(doc.page_content, doc.metadata.get("history")) for doc in results]
            query = f"从下面的文中抽取实体和关系：\n{sentence}"
            retrieved_docs = self.reranker.rerank(query=query, documents=retrieved_docs)
            retrieved_docs = [doc["document"]["text"] for doc in retrieved_docs if doc["relevance_score"] >= 0.5]
            related_history.extend(retrieved_docs)
        
        
        related_insights = []
        for sentence in sentences:
            documents = self.insight_memory.get(sentence, self.insight_memory_size)
            related_insights.extend([doc.page_content for doc in documents])
            
        related_insights = self.reranker.rerank(query="与文章相关的洞见", documents=related_insights, top_n=10)
        related_insights = [insight["document"]["text"] for insight in related_insights]
        
        return related_history, related_insights
            


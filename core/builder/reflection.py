from typing import Optional, List
from core.utils.config import KAGConfig
from core.memory.vector_memory import VectorMemory
from ..utils.format import correct_json_format
from core.model_providers.openai_rerank import OpenAIRerankModel
import json
import re

class DynamicReflector:
    def __init__(self, config: KAGConfig):
        """
        Initialize the DynamicReflector.

        Args:
            config (KAGConfig): Global configuration object.
        """
        self.config = config
        self.history_memory = VectorMemory(self.config, "history_memory")
        self.insight_memory = VectorMemory(self.config, "insight_memory")
        self.reranker = OpenAIRerankModel(self.config)
        self.entity_extraction_memory = dict()
        self.history_memory_size = self.config.memory.history_memory_size
        self.insight_memory_size = self.config.memory.insight_memory_size
    
    def clear(self):
        """
        Clear all stored history and insight memories.
        """
        self.history_memory.clear()
        self.insight_memory.clear()
        
    def generate_logs(self, extraction_result):
        """
        Generate extraction logs from entities and relations.

        Args:
            extraction_result (dict): The current extraction result containing 
                "entities" and "relations" fields.

        Returns:
            List[str]: A list of formatted log strings summarizing the extraction.
        """
        logs = []

        # 1) Handle entities
        entities = extraction_result.get("entities", [])
        if not entities:
            logs.append("No recognizable entities or entity extraction failed; no logs available.")
        else:
            for ent in entities:
                name = ent.get("name", "UNKNOWN")
                ent_type = ent.get("type", "UNKNOWN")
                desc = ent.get("description", "") or "EMPTY"
                scope = ent.get("scope", "")
                logs.append(
                    f"Entity '{name}' (Type: {ent_type}) was extracted. Description: {desc}. Current scope: {scope}"
                )
                if name in self.entity_extraction_memory:
                    related_history = list(self.entity_extraction_memory[name])
                    for history in related_history:
                        logs.append(f"- Previously detected in entity extraction: {history}")

        # 2) Handle relations
        relations = extraction_result.get("relations", [])
        if not relations:
            logs.append("No recognizable relations or relation extraction failed; no logs available.")
        else:
            for rel in relations:
                subj = rel.get("subject", "UNKNOWN")
                obj = rel.get("object", "UNKNOWN")
                relation_type = rel.get("relation_type", "UNKNOWN")
                relation_name = rel.get("relation_name", "UNKNOWN")
                desc = rel.get("description", "") or "EMPTY"
                logs.append(
                    f"Entity '{subj}' and entity '{obj}' have relation '{relation_name}' "
                    f"(Type: {relation_type}). Description: {desc}"
                )

        return logs
      
    def _store_memory(self, content, reflections):
        """
        Store reflections into memory (insight and history).

        Args:
            content (str): Original text content.
            reflections (dict): Reflection results including entities, relations, 
                insights, and score.
        """
        insights = reflections.get("insights", []) 
        for item in insights:
            self.insight_memory.add(text=item, metadata={})
        
        sentences = re.split(r'(?<=[。！？])', content)
        
        entities = reflections.get("entities", [])
        relations = reflections.get("relations", [])
        score = reflections.get("score", 0)
        
        documents = dict()
        
        # Store entity extractions
        for entity in entities:
            entity_name = entity.get("name", "")
            entity_type = entity.get("type", "")
            entity_scope = entity.get("scope", "")
            
            matches = [s.strip() for s in sentences if entity_name in s]
            if entity_name and entity_type:
                if entity_name in self.entity_extraction_memory:
                    self.entity_extraction_memory[entity_name].add(
                        f"Extracted entity: {entity_name} (Type: {entity_type}\n"
                    )
                else:
                    self.entity_extraction_memory[entity_name] = set([
                        f"Extracted entity: {entity_name} (Type: {entity_type})\n"
                    ])
                    
                for match in matches:
                    if match in documents:
                        documents[match] += f"- Extracted entity: {entity_name} (Type: {entity_type})\n"
                    else:
                        documents[match] = f"- Extracted entity: {entity_name} (Type: {entity_type})\n"

        # Store relation extractions
        for relation in relations:
            subject = relation.get("subject", "")
            object_ = relation.get("object", "")
            relation_name = relation.get("relation_name", "")
            relation_type = relation.get("relation_type", "")
            matches = [s.strip() for s in sentences if subject in s and object_ in s]
            if subject and object_ and relation_name:
                for match in matches:
                    if match in documents:
                        documents[match] += f"- Extracted relation: {subject}-{relation_name}->{object_} (Type: {relation_type})\n"
                    else:
                        documents[match] = f"- Extracted relation: {subject}-{relation_name}->{object_} (Type: {relation_type})\n"

        # Add score to all matched documents and store them in history memory
        for match in documents:
            documents[match] += f"Current extraction score: {score}"
            self.history_memory.add(text=match, metadata={"history": documents[match]})
            
    def _search_relevant_reflections(self, context):
        """
        Search for reflections (history and insights) relevant to the given context.

        Args:
            context (str): Input text for which related reflections are searched.

        Returns:
            Tuple[List[str], List[str]]:
                - related_history: List of relevant historical extractions.
                - related_insights: List of relevant insights.
        """
        def create_record(content, history):
            return f"Original text:\n{content}\n{history}"
        
        sentences = re.split(r'(?<=[。！？])', context)
        related_history = []
        
        # Search history memory
        for sentence in sentences:
            if len(sentence) > 500:
                sentence = sentence[:500]
            results = self.history_memory.get(sentence, self.history_memory_size)
            retrieved_docs = [
                create_record(doc.page_content, doc.metadata.get("history")) 
                for doc in results
            ]
            query = f"Extract entities and relations from the following sentence:\n{sentence}"
            retrieved_docs = self.reranker.rerank(query=query, documents=retrieved_docs)
            retrieved_docs = [doc["document"]["text"] for doc in retrieved_docs if doc["relevance_score"] >= 0.5]
            related_history.extend(retrieved_docs)
        
        # Search insight memory
        related_insights = []
        for sentence in sentences:
            documents = self.insight_memory.get(sentence, self.insight_memory_size)
            related_insights.extend([doc.page_content for doc in documents])
            
        related_insights = self.reranker.rerank(query="Insights related to the text", documents=related_insights, top_n=10)
        related_insights = [insight["document"]["text"] for insight in related_insights]
        
        return related_history, related_insights

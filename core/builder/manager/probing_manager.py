# kag/builder/manager/probing_manager.py

"""
图探测器模块
"""
import json
from typing import Dict, Any
from core.utils.config import KAGConfig
from core.functions.regular_functions import SchemaPruner, SchemaReflector, BackgroundParser, RelationSchemaParser, EntitySchemaParser, AbbreviationParser, FeedbackSummarizer

from core.utils.prompt_loader import PromptLoader
import os

class GraphProber:
    """图探测器"""

    def __init__(self, config: KAGConfig, llm):
        self.config = config
        self.llm = llm
        prompt_dir = self.config.knowledge_graph_builder.prompt_dir
        self.prompt_loader = PromptLoader(prompt_dir)
        self.schema_pruner = SchemaPruner(self.prompt_loader, self.llm)
        self.schema_reflector = SchemaReflector(self.prompt_loader, self.llm)
        self.background_parser = BackgroundParser(self.prompt_loader, self.llm)
        self.abbreviation_parser = AbbreviationParser(self.prompt_loader, self.llm)
        self.relation_schema_parser = RelationSchemaParser(self.prompt_loader, self.llm)
        self.entity_schema_parser = EntitySchemaParser(self.prompt_loader, self.llm)
        self.feedback_summarizer = FeedbackSummarizer(self.prompt_loader, self.llm)

    def update_background(
        self,
        text: str,
        current_background: str = None,
    ) -> str:
        params = {
                "text": text,
                "current_background": current_background
            }
        result = self.background_parser.call(
            params=json.dumps(params)
        )
        print("[CHECK] background: ", result)
        return result
    
    def update_abbreviations(
        self,
        text: str,
        current_background: str = None,
    ) -> str:
        params = {
                "text": text,
                "current_background": current_background
            }
        result = self.abbreviation_parser.call(
            params=json.dumps(params)
        )
        print("[CHECK] abbreviations: ", result)
        return result


    def update_entity_schema(
        self,
        text: str,
        current_schema: str = None,
        feedbacks: str = None,
        task_goals: str = None,
    ) -> str:
        result = self.entity_schema_parser.call(
            params=json.dumps({
                "text": text,
                "current_schema": current_schema,
                "feedbacks": feedbacks,
                "task_goals": task_goals,
            })
        )
        # print("[CHECK] entity_schema: ", result)
        return result

    def update_relation_schema(
        self,
        text: str,
        entity_schema: str = None,
        current_schema: str = None, 
        feedbacks: str = None,
        task_goals: str = None,
    ) -> str:
        params = {
            "text": text,
            "entity_schema": entity_schema,
            "current_schema": current_schema,
            "feedbacks": feedbacks,
            "task_goals": task_goals,
        }
        result = self.relation_schema_parser.call(
            params=json.dumps(params)
        )
        # print("[CHECK] relation_schema: ", result)
        return result

    def prune_schema(
        self,
        entity_type_distribution: str,
        relation_type_distribution: str,
        entity_type_description_text: str,
        relation_type_description_text: str
    ) -> str:
        params = {
            "entity_type_distribution": entity_type_distribution,
            "relation_type_distribution": relation_type_distribution,
            "entity_type_description_text": entity_type_description_text,
            "relation_type_description_text": relation_type_description_text
        }
        result = self.schema_pruner.call(params=json.dumps(params))
        # print("[CHECK] pruning output: ", result)
        return result

    def reflect_schema(
        self,
        schema: str,
        feedbacks: str,
    ) -> str:
   
        params = {
            "schema": schema,
            "feedbacks": feedbacks
        }
        result = self.schema_reflector.call(params=json.dumps(params))
        return result
    
    def summarize_feedbacks(
        self,
        context: str,
        max_items: int = 8,
    ) -> str:
   
        params = {
            "context": context,
            "max_items": max_items
        }
        result = self.feedback_summarizer.call(params=json.dumps(params))
        return result

# kag/builder/extractor.py

"""
信息抽取器模块
对接 Agent，提供 Extractor 对外接口
"""
import json
from typing import Dict, Any
from core.utils.config import KAGConfig
from core.functions.regular_functions import (EntityExtractor, RelationExtractor, 
    ExtractionReflector, AttributeExtractor, AttributeReflector)
# from kag.schema.kg_schema import ENTITY_TYPES, RELATION_TYPE_GROUPS
from core.utils.prompt_loader import PromptLoader
import os

class InformationExtractor:
    """信息抽取器"""

    def __init__(self, config: KAGConfig, llm):
        self.config = config
        self.llm = llm
        prompt_dir = self.config.knowledge_graph_builder.prompt_dir
        self.prompt_loader = PromptLoader(prompt_dir)
        self.entity_extraction = EntityExtractor(self.prompt_loader, self.llm)
        self.relation_extraction = RelationExtractor(self.prompt_loader, self.llm)
        self.extraction_reflection = ExtractionReflector(self.prompt_loader, self.llm)
        self.attribute_extraction = AttributeExtractor(self.prompt_loader, self.llm)
        self.attribute_reflection = AttributeReflector(self.prompt_loader, self.llm)

    def extract_entities(
        self,
        text: str,
        entity_type_description_text: str,
        system_prompt: str,
        reflection_results: dict,
        enable_thinking: bool = True
    ) -> str:
        """从文本中抽取实体"""
        params = {
                "text": text,
                "entity_type_description_text": entity_type_description_text,
                "system_prompt": system_prompt,
                "reflection_results": reflection_results
            }
        result = self.entity_extraction.call(
            params=json.dumps(params),
            enable_thinking=enable_thinking
        )
        # print("[CHECK] entity extraction result: ", result)
        return result


    def extract_relations(
        self,
        text: str,
        entity_list: str,
        relation_type_description_text: str,
        system_prompt: str,
        reflection_results: dict|str,
        enable_thinking: bool = True,
       # entity_extraction_results: dict|str,
    ) -> str:
        """从文本中抽取关系"""
        params = {
                "text": text,
                "entity_list": entity_list,
                "relation_type_description_text": relation_type_description_text,
                "reflection_results": reflection_results,
                "system_prompt": system_prompt,
                # "entity_extraction_results": entity_extraction_results
            }
        result = self.relation_extraction.call(
            params=json.dumps(params), enable_thinking=enable_thinking
        )
        # print("[CHECK] relation extraction result: ", result)
        return result

    def reflect_extractions(
        self,
        logs: str,
        entity_type_description_text: str,
        relation_type_description_text: str,
        system_prompt: str,
        original_text: str = None,
        previous_reflection: dict|str = None,
        version: str = "default", 
    ) -> str:
        """反思抽取结果的质量"""
        params = {
                "logs": logs,
                "entity_type_description_text": entity_type_description_text,
                "relation_type_description_text": relation_type_description_text,
                "original_text": original_text,
                "system_prompt": system_prompt,
                "previous_reflection": previous_reflection,
                "version": version,
            }
        result = self.extraction_reflection.call(
            params=json.dumps(params)
        )
        # print("[CHECK] 传入日志: ", logs)
        # print("[CHECK] reflection result: ", result)
        return result

    def extract_entity_attributes(
        self,
        text: str,
        entity_name: str, 
        description: str,
        entity_type: str,
        attribute_definitions: str,
        system_prompt: str = "",
        previous_results: str = None,
        feedbacks: str = None,
        original_text: str = None,
    ) -> str:
        """从文本和实体描述中抽取结构化属性"""

        params = {
            "text": text,
            "description": description,
            "entity_name": entity_name,
            "entity_type": entity_type,
            "attribute_definitions": attribute_definitions,
            "system_prompt": system_prompt,
            "previous_results": previous_results,
            "feedbacks": feedbacks,
            "original_text": original_text
        }

        result = self.attribute_extraction.call(params=json.dumps(params))
        # print("[CHECK] entity name: ", entity_name)
        # print("[CHECK] input text: ", text)
        #print("[CHECK] entity attribute extraction result: ", result)
        return result

    def reflect_entity_attributes(
        self,
        entity_name: str,
        entity_type: str,
        description: str,
        attribute_definitions: str,
        attributes: str,
        system_prompt: str = ""
    ) -> str:
        """
        对属性抽取结果进行反思评估，判断是否完整、是否需要补充上下文等
        """
        params = {
            "entity_name": entity_name,
            "entity_type": entity_type,
            "description": description,
            "attribute_definitions": attribute_definitions,
            "attributes": attributes,
            "system_prompt": system_prompt
        }

        result = self.attribute_reflection.call(params=json.dumps(params))
        # print("[CHECK] attribute reflection result:", result)
        return result

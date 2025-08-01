# kag/builder/extractor.py

"""
信息抽取器模块
对接 Agent，提供 Extractor 对外接口
"""
import json
from typing import Dict, Any
from kag.utils.config import KAGConfig
from kag.functions.regular_functions import (EntityExtractor, RelationExtractor, 
    ExtractionReflector, AttributeExtractor, AttributeReflector, EventCausalityChecker)
# from kag.schema.kg_schema import ENTITY_TYPES, RELATION_TYPE_GROUPS
from kag.utils.prompt_loader import PromptLoader
import os

class InformationExtractor:
    """信息抽取器"""

    def __init__(self, config: KAGConfig, llm):
        self.config = config
        self.llm = llm
        
        prompt_dir = config.prompt_dir if hasattr(config, 'prompt_dir') else os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "kag/prompts")
        self.prompt_loader = PromptLoader(prompt_dir)

        self.entity_extraction = EntityExtractor(self.prompt_loader, self.llm)
        self.relation_extraction = RelationExtractor(self.prompt_loader, self.llm)
        self.extraction_reflection = ExtractionReflector(self.prompt_loader, self.llm)
        self.attribute_extraction = AttributeExtractor(self.prompt_loader, self.llm)
        self.attribute_reflection = AttributeReflector(self.prompt_loader, self.llm)
        self.event_causality_checker = EventCausalityChecker(self.prompt_loader, self.llm)  

    def extract_entities(
        self,
        text: str,
        entity_type_description_text: str,
        abbreviations: str,
        reflection_results: dict
    ) -> str:
        """从文本中抽取实体"""
        params = {
                "text": text,
                "entity_type_description_text": entity_type_description_text,
                "abbreviations": abbreviations,
                "reflection_results": reflection_results
            }
        result = self.entity_extraction.call(
            params=json.dumps(params)
        )
        # print("[CHECK] entity extraction result: ", result)
        return result


    def extract_relations(
        self,
        text: str,
        entity_list: str,
        relation_type_description_text: str,
        abbreviations: str,
        reflection_results: dict|str,
       # entity_extraction_results: dict|str,
    ) -> str:
        """从文本中抽取关系"""
        result = self.relation_extraction.call(
            params=json.dumps({
                "text": text,
                "entity_list": entity_list,
                "relation_type_description_text": relation_type_description_text,
                "reflection_results": reflection_results,
                "abbreviations": abbreviations,
                # "entity_extraction_results": entity_extraction_results
            })
        )
        # print("[CHECK] relation extraction result: ", result)
        return result

    def reflect_extractions(
        self,
        logs: str,
        entity_type_description_text: str,
        relation_type_description_text: str,
        abbreviations: str,
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
                "abbreviations": abbreviations,
                "previous_reflection": previous_reflection,
                "version": version,
            }
        result = self.extraction_reflection.call(
            params=json.dumps(params)
        )
        # print("[CHECK] 传入日志: ", logs)
        # print("[CHECK] reflection extraction result: ", result)
        return result

    def extract_entity_attributes(
        self,
        text: str,
        entity_name: str, 
        description: str,
        entity_type: str,
        attribute_definitions: str,
        abbreviations: str = "",
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
            "abbreviations": abbreviations,
            "previous_results": previous_results,
            "feedbacks": feedbacks,
            "original_text": original_text
        }

        result = self.attribute_extraction.call(params=json.dumps(params))
        # print("[CHECK] entity name: ", entity_name)
        # print("[CHECK] input text: ", text)
        print("[CHECK] entity attribute extraction result: ", result)
        return result

    def reflect_entity_attributes(
        self,
        entity_name: str,
        entity_type: str,
        description: str,
        attribute_definitions: str,
        attributes: str,
        abbreviations: str = ""
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
            "abbreviations": abbreviations
        }

        result = self.attribute_reflection.call(params=json.dumps(params))
        print("[CHECK] attribute reflection result:", result)
        return result

    def check_event_causality(
        self,
        event_1_info: str,
        event_2_info: str,
        abbreviations: str = ""
    ) -> str:
        """判断两个事件是否存在因果关系"""
        params = {
            "event_1_info": event_1_info,
            "event_2_info": event_2_info,
            "abbreviations": abbreviations
        }
        result = self.event_causality_checker.call(params=json.dumps(params))
        # print("[CHECK] check event causality result: ", result)
        return result

# kag/builder/extractor.py

"""
信息抽取器模块
对接 Agent，提供 Extractor 对外接口
"""
import json
from typing import Dict, Any
from core.utils.config import KAGConfig
from core.functions.regular_functions import (EventCausalityChecker, RedundancyEvaluator, PlotUnitExtractor, PlotRelationExtractor, EventContextGenerator)
# from kag.schema.kg_schema import ENTITY_TYPES, RELATION_TYPE_GROUPS
from core.utils.prompt_loader import PromptLoader
import os

class GraphManager:
    """信息抽取器"""

    def __init__(self, config: KAGConfig, llm):
        self.config = config
        self.llm = llm
        
        prompt_dir = self.config.knowledge_graph_builder.prompt_dir
        self.prompt_loader = PromptLoader(prompt_dir)

        self.event_causality_checker = EventCausalityChecker(self.prompt_loader, self.llm)  
        self.redundancy_evaluator = RedundancyEvaluator(self.prompt_loader, self.llm)
        self.plot_generator = PlotUnitExtractor(self.prompt_loader, self.llm)
        self.plot_relation_extractor = PlotRelationExtractor(self.prompt_loader, self.llm)
        self.event_context_generator = EventContextGenerator(self.prompt_loader, self.llm)

    def check_event_causality(
        self,
        event_1_info: str,
        event_2_info: str,
        system_prompt: str = "",
        related_context: str = ""
    ) -> str:
        """判断两个事件是否存在因果关系"""
        params = {
            "event_1_info": event_1_info,
            "event_2_info": event_2_info,
            "system_prompt": system_prompt,
            "related_context": related_context
        }
        result = self.event_causality_checker.call(params=json.dumps(params))
        # print("[CHECK] check event causality result: ", result)
        return result
    
    def evaluate_event_redundancy(
        self,
        event_details: str,
        relation_details: str,
        system_prompt: str = "",
        related_context: str = ""
    ):
        """判断两个事件是否存在因果关系"""
        params = {
            "event_details": event_details,
            "relation_details": relation_details,
            "system_prompt": system_prompt,
            "related_context": related_context
        }
        result = self.redundancy_evaluator.call(params=json.dumps(params))
        # print("[CHECK] check event causality result: ", result)
        return result
    
    def generate_event_plot(
        self,
        event_chain_info: str,
        system_prompt: str = "",
        related_context: str = ""
    ):
        """判断两个事件是否存在因果关系"""
        params = {
            "event_chain_info": event_chain_info,
            "system_prompt": system_prompt,
            "related_context": related_context
        }
        result = self.plot_generator.call(params=json.dumps(params))
        # print("[CHECK] check event causality result: ", result)
        return result
    
    def extract_plot_relation(
        self,
        plot_A_info: str,
        plot_B_info: str,
        system_prompt: str = ""
    ):
        """判断两个事件是否存在情节关系"""
        params = {
            "plot_A_info": plot_A_info,
            "plot_B_info": plot_B_info,
            "system_prompt": system_prompt,
        }
        result = self.plot_relation_extractor.call(params=json.dumps(params))
        # print("[CHECK] check event causality result: ", result)
        return result
    
    def generate_event_context(
        self,
        event_info: str,
        related_context: str,
        system_prompt: str = ""
    ):
        """判断两个事件是否存在情节关系"""
        params = {
            "event_info": event_info,
            "related_context": related_context,
            "system_prompt": system_prompt,
        }
        result = self.event_context_generator.call(params=json.dumps(params))
        # print("[CHECK] check event causality result: ", result)
        return result
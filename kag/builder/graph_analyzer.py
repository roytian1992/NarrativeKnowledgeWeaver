# kag/builder/extractor.py

"""
信息抽取器模块
对接 Agent，提供 Extractor 对外接口
"""
import json
from typing import Dict, Any
from kag.utils.config import KAGConfig
from kag.functions.regular_functions import (EventCausalityChecker, RedundancyEvaluator, PlotGenerator)
# from kag.schema.kg_schema import ENTITY_TYPES, RELATION_TYPE_GROUPS
from kag.utils.prompt_loader import PromptLoader
import os

class GraphAnalyzer:
    """信息抽取器"""

    def __init__(self, config: KAGConfig, llm):
        self.config = config
        self.llm = llm
        
        prompt_dir = config.prompt_dir if hasattr(config, 'prompt_dir') else os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "kag/prompts")
        self.prompt_loader = PromptLoader(prompt_dir)

        self.event_causality_checker = EventCausalityChecker(self.prompt_loader, self.llm)  
        self.redundancy_evaluator = RedundancyEvaluator(self.prompt_loader, self.llm)
        self.plot_generator = PlotGenerator(self.prompt_loader, self.llm)

    def check_event_causality(
        self,
        event_1_info: str,
        event_2_info: str,
        system_prompt: str = ""
    ) -> str:
        """判断两个事件是否存在因果关系"""
        params = {
            "event_1_info": event_1_info,
            "event_2_info": event_2_info,
            "system_prompt": system_prompt
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
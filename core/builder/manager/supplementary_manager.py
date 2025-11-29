# -*- coding: utf-8 -*-

import json
from typing import Dict, Any
from core.utils.config import KAGConfig
from core.functions.regular_functions import (
    CharacterStatusExtractor,
    ContinuityChecker,
    ContinuityChainChecker,
    CharacterStatusReflector
)
from core.utils.prompt_loader import PromptLoader
import os


class SupplementaryExtractor:
    """
    Supplementary information extractor facade.
    """

    def __init__(self, config: KAGConfig, llm, prompt_loader: PromptLoader = None):
        self.config = config
        self.llm = llm
        if not prompt_loader:
            prompt_dir = self.config.knowledge_graph_builder.prompt_dir
            self.prompt_loader = PromptLoader(prompt_dir)
        else:
            self.prompt_loader = prompt_loader

        self.character_status_extraction = CharacterStatusExtractor(self.prompt_loader, self.llm)
        self.character_status_reflection = CharacterStatusReflector(self.prompt_loader, self.llm)
        self.continuity_check = ContinuityChecker(self.prompt_loader, self.llm)
        self.continuity_chain_checker = ContinuityChainChecker(self.prompt_loader, self.llm)

    def extract_character_status(
        self,
        scene_contents: str,
        character_list: str,
        feedbacks: str = "",
        enable_thinking: bool = True,
    ) -> str:
        """Extract character status over time from scene contents."""
        params = {
            "scene_contents": scene_contents,
            "character_list": character_list,
            "feedbacks": feedbacks,
        }
        result = self.character_status_extraction.call(params=json.dumps(params), enable_thinking=enable_thinking)
        return result 
    
    def reflect_character_status(
        self,
        extracted_results: str,
        character_list: str,
        scene_contents: str,
        enable_thinking: bool = True,
    ) -> str:
        """Reflect on the quality of extracted character status results."""
        params = {
            "scene_contents": scene_contents,
            "extracted_results": extracted_results,
            "character_list": character_list,
        }
        result = self.character_status_reflection.call(params=json.dumps(params), enable_thinking=enable_thinking)
        return result
    
    def check_screenplay_continuity(
        self,
        scene_name1: str,
        summary1: str,
        cmp_info1: str,
        scene_name2: str,
        summary2: str,
        cmp_info2: str,
        common_neighbor_info: str = "",
        enable_thinking: bool = True,
    ) -> str:
        """
        Check continuity (接戏/连戏关系) between two scenes based on scene info and CMP data.

        Args:
            scene_name1: 场景一名称（可能包含地点信息）
            summary1: 场景一简介（剧情概述）
            cmp_info1: 场景一的服化道信息（Costume/Makeup/Props）
            scene_name2: 场景二名称（可能包含地点信息）
            summary2: 场景二简介（剧情概述）
            cmp_info2: 场景二的服化道信息（Costume/Makeup/Props）
            common_neighbor_info: 两个场景在知识图谱中的共同邻居信息（可选）
            enable_thinking: 是否启用思维链（传给 LLM 侧的控制参数）

        Returns:
            str: JSON 字符串，形如：
                {
                  "is_continuity": true/false,
                  "reason": "解释理由"
                }
        """
        params = {
            "scene_name1": scene_name1,
            "summary1": summary1,
            "cmp_info1": cmp_info1,
            "scene_name2": scene_name2,
            "summary2": summary2,
            "cmp_info2": cmp_info2,
            "common_neighbor_info": common_neighbor_info,
        }

        result = self.continuity_check.call(
            params=json.dumps(params, ensure_ascii=False),
            enable_thinking=enable_thinking
        )
        return result
    
    def check_continuity_chain(
            self,
            scene_id_list,
            scene_metadata_blocks: str,
            enable_thinking: bool = True,
        ) -> str:
        """
        Evaluate production continuity of a multi-scene continuity chain.

        Args:
            scene_id_list: 场景 ID 列表，可以是 list 或 字符串。
                        若是 list，会自动转为逗号分隔字符串传给 Prompt。
            scene_metadata_blocks: 多场景 metadata 汇总文本（通常来自 summarizer）
            enable_thinking: 是否启用思维链推理（传给 LLM 底层）

        Returns:
            str: JSON 字符串，类似：
                {
                "coherence_score": 0.87,
                "keep_decision": "split",
                "suggested_splits": [
                    ["scene_5", "scene_6"],
                    ["scene_10"]
                ],
                "rationale": "说明理由",
                "decision_confidence": "medium"
                }
        """
        # Ensure list → str
        if isinstance(scene_id_list, list):
            scene_id_list = ", ".join(str(s) for s in scene_id_list)

        params = {
            "scene_id_list": scene_id_list,
            "scene_metadata_blocks": scene_metadata_blocks,
        }

        result = self.continuity_chain_checker.call(
            params=json.dumps(params, ensure_ascii=False),
            enable_thinking=enable_thinking
        )

        return result


# kag/builder/extractor.py

"""
信息抽取器模块
对接 Agent，提供 Extractor 对外接口
"""
import json
from typing import Any, Dict, List, Optional
from core.utils.config import KAGConfig
from core.functions.regular_functions import MetadataParser, SemanticSplitter, ParagraphSummarizer, EntityMerger, InsightExtractor, EntityTypeValidator, EntityScopeValidator
from core.utils.prompt_loader import PromptLoader
import os

class DocumentParser:
    """文本处理器"""

    def __init__(self, config: KAGConfig, llm):
        self.config = config
        self.llm = llm
        prompt_dir = self.config.knowledge_graph_builder.prompt_dir
        self.prompt_loader = PromptLoader(prompt_dir)

        self.metadata_parser = MetadataParser(self.prompt_loader, self.llm)
        self.semantic_splitter = SemanticSplitter(self.prompt_loader, self.llm)
        self.paragraph_summarizer = ParagraphSummarizer(self.prompt_loader, self.llm)
        self.entity_merger = EntityMerger(self.prompt_loader, self.llm)
        self.insight_extractor = InsightExtractor(self.prompt_loader, self.llm)
        self.entity_type_validator = EntityTypeValidator(self.prompt_loader, self.llm)
        self.entity_scope_validator = EntityScopeValidator(self.prompt_loader, self.llm)

    def parse_metadata(
        self,
        text: str,
        title: str, 
        subtitle: str,
        doc_type: str = "screenplay"
    ) -> str:
        """从文本中抽取实体"""
        params = {
                "text": text,
                "title": title,
                "subtitle": subtitle,
                "doc_type": doc_type
            }
        result = self.metadata_parser.call(
            params=json.dumps(params, ensure_ascii=False)
        )
        # print("[CHECK] entity extraction result: ", result)
        return result
    
    def extract_insights(
        self,
        text: str
    ) -> str:
        """从文本中抽取实体"""
        params = {
                "text": text
            }
        result = self.insight_extractor.call(
            params=json.dumps(params, ensure_ascii=False)
        )
        # print("[CHECK] insight extraction result: ", result)
        return result
            
    def split_text(
        self,
        text: str,
        max_segments: int=3,
    ) -> str:
        """从文本中抽取实体"""
        params = {
                "text": text,
                "max_segments": max_segments
            }
        
        result = self.semantic_splitter.call(
            params=json.dumps(params, ensure_ascii=False)
        )
        # print("[CHECK] entity extraction result: ", result)
        return result

    def summarize_paragraph(
        self,
        text: str,
        max_length: int = 200, 
        previous_summary: str = ""
    ) -> str:
        """从文本中抽取实体"""
        params = {
                "text": text.strip(),
                "max_length": max_length,
                "previous_summary": previous_summary
            }
        result = self.paragraph_summarizer.call(
            params=json.dumps(params, ensure_ascii=False)
        )
        
        # print("[CHECK] params: ", params)
        return result
    
    def merge_entities(
        self,
        entity_descriptions: str,
        system_prompt: str = "",
        related_context: str = ""
    ):
        """判断两个事件是否存在因果关系"""
        params = {
            "entity_descriptions": entity_descriptions,
            "system_prompt": system_prompt,
            "related_context": related_context
        }
        result = self.entity_merger.call(params=json.dumps(params))
        # print("[CHECK] check event causality result: ", result)
        return result
    
    def validate_entity_type(
        self,
        context: str
    ):
        """判断实体类型是否合法"""
        params = {
            "context": context
        }
        result = self.entity_type_validator.call(
            params=json.dumps(params, ensure_ascii=False)
        )
        return result
        
    def validate_entity_scope(
        self,
        context: str
    ):
        """判断实体范围是否合法"""
        params = {
            "context": context
        }
        result = self.entity_scope_validator.call(
            params=json.dumps(params, ensure_ascii=False)
        )
        return result
    
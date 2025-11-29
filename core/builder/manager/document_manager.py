# -*- coding: utf-8 -*-
"""
Document Manager Module
=======================

This module provides **document-level preprocessing utilities** for narrative
analysis. It wraps LLM-based functions such as metadata parsing, semantic
splitting, summarization, and insight extraction, serving as the first stage
before graph construction.

Key functionalities:
- Parse document metadata (title, subtitle, type).
- Perform semantic-based text segmentation.
- Summarize text at the paragraph level (rolling summarization supported).
- Extract high-level insights from narrative text.
- Merge duplicate or similar entity mentions.
- Validate entity types and scope (global/local consistency).

Class:
    DocumentParser
        Provides interfaces for all document-level operations.

Usage:
    - parser = DocumentParser(config, llm)
    - meta = parser.parse_metadata(text, title="Episode 1", subtitle="Pilot")
    - summary = parser.summarize_paragraph(paragraph_text)
    - segments = parser.split_text(long_text)
"""

import json
from typing import Any, Dict
from core.utils.config import KAGConfig
from core.functions.regular_functions import (
    MetadataParser, 
    SemanticSplitter, 
    ParagraphSummarizer, 
    EntityMerger, 
    InsightExtractor, 
    EntityTypeValidator, 
    EntityScopeValidator,
    TimelineParser,
    AgenticSearch
)
from core.utils.prompt_loader import PromptLoader
import os


class DocumentParser:
    """
    Document-level processing utility.

    This class manages narrative preprocessing operations:
    metadata parsing, semantic text splitting, summarization,
    insight extraction, entity merging, and entity validation.
    """

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
        self.timeline_parser = TimelineParser(self.prompt_loader, self.llm)
        self.agentic_search = AgenticSearch(self.prompt_loader, self.llm)

    def parse_metadata(
        self,
        text: str,
        title: str, 
        subtitle: str,
        doc_type: str = "screenplay"
    ) -> str:
        """Extract document-level metadata such as title, subtitle, and type."""
        params = {
            "text": text,
            "title": title,
            "subtitle": subtitle,
            "doc_type": doc_type
        }
        result = self.metadata_parser.call(
            params=json.dumps(params, ensure_ascii=False)
        )
        return result
    
    def parse_time_elements(
        self,
        text: str,
        existing: str = None,
    ):
        params = {
            "text": text,
            "existing": existing or "",
        }
        result = self.timeline_parser.call(
            params=json.dumps(params, ensure_ascii=False)
        )
        print(result)
        return result
    
    def extract_insights(
        self,
        text: str
    ) -> str:
        """Extract insights (important information, themes, or narrative clues) from the text."""
        params = {"text": text}
        result = self.insight_extractor.call(
            params=json.dumps(params, ensure_ascii=False)
        )
        return result
            
    def split_text(
        self,
        text: str,
        max_segments: int = 3,
    ) -> str:
        """Split text into semantically coherent segments."""
        params = {
            "text": text,
            "max_segments": max_segments
        }
        result = self.semantic_splitter.call(
            params=json.dumps(params, ensure_ascii=False)
        )
        return result

    def summarize_paragraph(
        self,
        text: str,
        max_length: int = 200, 
        previous_summary: str = "",
        goal: str = ""
    ) -> str:
        """Summarize a paragraph, optionally using a previous rolling summary."""
        params = {
            "text": text.strip(),
            "max_length": max_length,
            "previous_summary": previous_summary,
            "goal": goal
        }
        result = self.paragraph_summarizer.call(
            params=json.dumps(params, ensure_ascii=False)
        )
        return result
    
    def search_content(
        self,
        text: str,
        max_length: int = 200, 
        goal: str = ""
    ) -> str:
        """Summarize a paragraph, optionally using a previous rolling summary."""
        params = {
            "text": text.strip(),
            "max_length": max_length,
            "goal": goal
        }
        result = self.agentic_search.call(
            params=json.dumps(params, ensure_ascii=False)
        )
        return result
    
    def merge_entities(
        self,
        entity_descriptions: str,
        system_prompt: str = "",
        related_context: str = ""
    ):
        """Merge similar or duplicate entities based on descriptions and context."""
        params = {
            "entity_descriptions": entity_descriptions,
            "system_prompt": system_prompt,
            "related_context": related_context
        }
        result = self.entity_merger.call(params=json.dumps(params))
        return result
    
    def validate_entity_type(
        self,
        context: str
    ):
        """Validate whether an entity type is legal/consistent with the schema."""
        params = {"context": context}
        result = self.entity_type_validator.call(
            params=json.dumps(params, ensure_ascii=False)
        )
        return result
        
    def validate_entity_scope(
        self,
        context: str
    ):
        """Validate whether the entity scope (e.g., local/global) is correct."""
        params = {"context": context}
        result = self.entity_scope_validator.call(
            params=json.dumps(params, ensure_ascii=False)
        )
        return result

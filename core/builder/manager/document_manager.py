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
    InsightExtractor, 
    RelatedContentExtractor,
    CandidateRelevanceScorer,
)
from core.functions.aggregation_functions import CommunityReportGenerator
from core.utils.prompt_loader import YAMLPromptLoader
import os


class DocumentParser:
    """
    Document-level processing utility.

    This class manages narrative preprocessing operations:
    metadata parsing, semantic text splitting, summarization,
    insight extraction, entity merging, and entity validation.
    """

    def __init__(self, config: KAGConfig, llm, ):
        self.config = config
        self.llm = llm
        prompt_dir = self.config.global_config.prompt_dir
        task_dir = self.config.global_config.task_dir
        metadata_parsing_task = os.path.join(task_dir, "metadata_parsing_task.json")
        insight_extraction_task = os.path.join(task_dir, "insight_extraction_task.json")
        self.prompt_loader = YAMLPromptLoader(prompt_dir)
        self.metadata_parser = MetadataParser(prompt_loader=self.prompt_loader, llm=self.llm, task_schema_path=metadata_parsing_task)
        self.semantic_splitter = SemanticSplitter(prompt_loader=self.prompt_loader, llm=self.llm)
        self.paragraph_summarizer = ParagraphSummarizer(prompt_loader=self.prompt_loader, llm=self.llm)
        self.insight_extractor = InsightExtractor(prompt_loader=self.prompt_loader, llm=self.llm, task_schema_path=insight_extraction_task)
        self.related_content_extractor = RelatedContentExtractor(prompt_loader=self.prompt_loader, llm=self.llm)
        self.candidate_relevance_scorer = CandidateRelevanceScorer(prompt_loader=self.prompt_loader, llm=self.llm)
        self.community_report_generator = CommunityReportGenerator(prompt_loader=self.prompt_loader, llm=self.llm)

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

    def generate_title_and_metadata(
        self,
        text: str,
        title: str = "",
        subtitle: str = "",
        doc_type: str = "general",
    ) -> str:
        """Generate a segment-specific title from content and extract metadata."""
        params = {
            "text": text,
            "title": title,
            "subtitle": subtitle,
            "doc_type": doc_type,
        }
        result = self.metadata_parser.generate_title_and_metadata(
            params=json.dumps(params, ensure_ascii=False)
        )
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

    def search_related_content(
        self,
        text: str,
        goal: str,
        max_length: int = 300,
    ) -> str:
        """Extract goal-related content from one document text."""
        params = {
            "text": (text or "").strip(),
            "goal": (goal or "").strip(),
            "max_length": max_length,
        }
        result = self.related_content_extractor.call(
            params=json.dumps(params, ensure_ascii=False)
        )
        return result

    def score_candidate_relevance(
        self,
        text: str,
        goal: str,
    ) -> str:
        """Estimate whether one candidate text is likely to help answer the goal."""
        params = {
            "text": (text or "").strip(),
            "goal": (goal or "").strip(),
        }
        result = self.candidate_relevance_scorer.call(
            params=json.dumps(params, ensure_ascii=False)
        )
        return result

    def generate_community_report(
        self,
        text: str,
        max_length: int = 1800,
        max_findings: int = 6,
        goal: str = "",
    ) -> str:
        """Generate a GraphRAG-style community report from structured community context."""
        params = {
            "text": (text or "").strip(),
            "max_length": max_length,
            "max_findings": max_findings,
            "goal": (goal or "").strip(),
        }
        result = self.community_report_generator.call(
            params=json.dumps(params, ensure_ascii=False)
        )
        return result
    

# kag/builder/manager/probing_manager.py

# -*- coding: utf-8 -*-
"""
Graph Probing Module
====================

This module provides utilities for **schema probing and refinement** during
knowledge graph construction. It wraps several LLM-backed function tools
that can incrementally update background information, abbreviations,
entity/relation schemas, prune unused types, and reflect on schema quality.

Supported tasks:
----------------
- Update background context and narrative setting.
- Update and expand abbreviation lists.
- Update entity schema with feedback and task goals.
- Update relation schema with feedback and task goals.
- Prune schema types based on distribution statistics.
- Reflect on schema quality and produce reasoning/score.
- Summarize feedback into concise actionable items.

Class:
------
    GraphProber
        Provides high-level interfaces for schema probing and refinement.

Usage example:
--------------
    - config = KAGConfig.load("config.yaml")
    - prober = GraphProber(config, llm)
    - background = prober.update_background("Some text", current_background="")
    - schema = prober.update_entity_schema("Entity insights", current_schema="{}")
"""

import json
from typing import Dict, Any
from core.utils.config import KAGConfig
from core.functions.regular_functions import (
    SchemaPruner,
    SchemaReflector,
    BackgroundParser,
    RelationSchemaParser,
    EntitySchemaParser,
    AbbreviationParser,
    FeedbackSummarizer,
)
from core.utils.prompt_loader import PromptLoader
import os


class GraphProber:
    """
    Graph schema probing and refinement utility.

    Encapsulates LLM-backed tools to maintain schema consistency:
    background parsing, abbreviation updates, entity/relation schema
    updates, pruning, reflection, and feedback summarization.
    """

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

    # ---------------- Background & Abbreviations ----------------

    def update_background(self, text: str, current_background: str = None) -> str:
        """Update narrative background information."""
        params = {"text": text, "current_background": current_background}
        result = self.background_parser.call(params=json.dumps(params))
        print("[CHECK] background: ", result)
        return result

    def update_abbreviations(self, text: str, current_background: str = None) -> str:
        """Update abbreviation/terminology list."""
        params = {"text": text, "current_background": current_background}
        result = self.abbreviation_parser.call(params=json.dumps(params))
        print("[CHECK] abbreviations: ", result)
        return result

    # ---------------- Entity & Relation Schema ----------------

    def update_entity_schema(
        self,
        text: str,
        current_schema: str = None,
        feedbacks: str = None,
        task_goals: str = None,
    ) -> str:
        """Update entity schema given new insights, feedback, and task goals."""
        result = self.entity_schema_parser.call(
            params=json.dumps(
                {
                    "text": text,
                    "current_schema": current_schema,
                    "feedbacks": feedbacks,
                    "task_goals": task_goals,
                }
            )
        )
        print("[CHECK] entity_schema: ", result)
        return result

    def update_relation_schema(
        self,
        text: str,
        entity_schema: str = None,
        current_schema: str = None,
        feedbacks: str = None,
        task_goals: str = None,
    ) -> str:
        """Update relation schema given new insights, feedback, and task goals."""
        params = {
            "text": text,
            "entity_schema": entity_schema,
            "current_schema": current_schema,
            "feedbacks": feedbacks,
            "task_goals": task_goals,
        }
        result = self.relation_schema_parser.call(params=json.dumps(params))
        print("[CHECK] relation_schema: ", result)
        return result

    # ---------------- Pruning & Reflection ----------------

    def prune_schema(
        self,
        entity_type_distribution: str,
        relation_type_distribution: str,
        entity_type_description_text: str,
        relation_type_description_text: str,
    ) -> str:
        """Prune schema types based on observed distribution statistics."""
        params = {
            "entity_type_distribution": entity_type_distribution,
            "relation_type_distribution": relation_type_distribution,
            "entity_type_description_text": entity_type_description_text,
            "relation_type_description_text": relation_type_description_text,
        }
        result = self.schema_pruner.call(params=json.dumps(params))
        # print("[CHECK] pruning output: ", result)
        return result

    def reflect_schema(self, schema: str, feedbacks: str) -> str:
        """Reflect on schema quality and incorporate feedbacks."""
        params = {"schema": schema, "feedbacks": feedbacks}
        result = self.schema_reflector.call(params=json.dumps(params))
        return result

    # ---------------- Feedback Summarization ----------------

    def summarize_feedbacks(self, context: str, max_items: int = 8) -> str:
        """Summarize verbose feedback text into concise actionable points."""
        params = {"context": context, "max_items": max_items}
        result = self.feedback_summarizer.call(params=json.dumps(params))
        return result

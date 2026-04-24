# -*- coding: utf-8 -*-
"""
Extractor Module
================

This module provides a unified **information extraction interface** that connects
to multiple LLM-backed function tools. It serves as the central entry point for
entity, relation, attribute extraction tasks.

Supported tasks:
----------------
- Entity extraction
- Relation extraction
- Property Extraction
- Property Merging


Class:
------
    InformationExtractor
        High-level facade for invoking extraction tasks.

Usage example:
--------------
    >>> from core.utils.config import KAGConfig
    >>> from core.utils.prompt_loader import PromptLoader
    >>> config = KAGConfig.load("config.yaml")
    >>> extractor = InformationExtractor(config, llm)
    >>> entities = extractor.extract_entities(text, entity_type="Character")
"""

import json
from typing import Dict, Any
from core.utils.config import KAGConfig
from core.functions.regular_functions import (
    EntityExtractor,
    OpenRelationExtractor,
    SchemaRelationGrounder,
    RelationExtractor,
    PropertyExtractor,
    PropertyFinalizer,
    InteractionExtractor,
)

from core.utils.prompt_loader import JSONPromptLoader, YAMLPromptLoader
import os


class InformationExtractor:
    """
    High-level information extractor facade.

    Wraps multiple tool interfaces (entity, relation, property) and
    exposes them as easy-to-use methods.
    """

    def __init__(self, config: KAGConfig, llm):
        self.config = config
        self.llm = llm
        prompt_dir = self.config.global_config.prompt_dir
        self.prompt_loader = YAMLPromptLoader(prompt_dir)
        schema_dir = self.config.global_config.schema_dir
        task_dir = self.config.global_config.task_dir

        entity_schema_path = os.path.join(schema_dir, "default_entity_schema.json")
        relation_schema_path = os.path.join(schema_dir, "default_relation_schema.json")
        interaction_schema_path = os.path.join(schema_dir, "default_interaction_schema.json")
        entity_task_path = os.path.join(task_dir, "entity_extraction_task.json")
        open_relation_task_path = os.path.join(task_dir, "open_relation_extraction_task.json")
        schema_relation_grounding_task_path = os.path.join(task_dir, "schema_relation_grounding_task.json")
        relation_task_path = os.path.join(task_dir, "relation_extraction_task.json")
        interaction_task_path = os.path.join(task_dir, "interaction_extraction_task.json")

        self.entity_extraction = EntityExtractor(llm=self.llm, prompt_loader=self.prompt_loader, task_schema_path=entity_task_path, entity_schema_path=entity_schema_path)
        self.open_relation_extraction = OpenRelationExtractor(
            llm=self.llm,
            prompt_loader=self.prompt_loader,
            task_schema_path=open_relation_task_path,
            relation_schema_path=relation_schema_path,
        )
        self.schema_relation_grounding = SchemaRelationGrounder(
            llm=self.llm,
            prompt_loader=self.prompt_loader,
            task_schema_path=schema_relation_grounding_task_path,
            relation_schema_path=relation_schema_path,
        )
        self.relation_extraction = RelationExtractor(llm=self.llm, prompt_loader=self.prompt_loader, task_schema_path=relation_task_path, relation_schema_path=relation_schema_path)
        self.interaction_extraction = InteractionExtractor(
            llm=self.llm,
            prompt_loader=self.prompt_loader,
            task_schema_path=interaction_task_path,
            interaction_schema_path=interaction_schema_path,
        )

        self.property_extraction = PropertyExtractor(llm=self.llm, prompt_loader=self.prompt_loader, schema_path=entity_schema_path)
        self.property_merger = PropertyFinalizer(llm=self.llm, prompt_loader=self.prompt_loader)
    # ---------------- Entity & Relation ----------------

    def extract_entities(
        self,
        text: str,
        entity_group: str,
        previous_results: str = None,
        extracted_entities: str = None,
        feedbacks: str = None,
        memory_context: str = "",
    ) -> str:
        """Extract entities from text."""
        params = {
            "text": text,
            "entity_group": entity_group,
            "previous_results": previous_results,
            "extracted_entities": extracted_entities,
            "feedbacks": feedbacks,
            "memory_context": memory_context,
        }
        result = self.entity_extraction.call(params=json.dumps(params))
        return result
    
    def extract_relations(
        self,
        text: str,
        previous_results: str = None,
        extracted_entities: str = None,
        extracted_relations: str = None,
        relation_group: str = None,
        feedbacks: str = None,
        memory_context: str = "",
    ) -> str:
        """Extract relations from text."""
        params = {
            "text": text,
            "previous_results": previous_results,
            "extracted_entities": extracted_entities,
            "extracted_relations": extracted_relations,
            "relation_group": relation_group,
            "feedbacks": feedbacks,
            "memory_context": memory_context,
        }
        result = self.relation_extraction.call(params=json.dumps(params))
        return result

    def extract_open_relations(
        self,
        text: str,
        extracted_entities: str = None,
        previous_results: str = None,
        feedbacks: str = None,
        memory_context: str = "",
        relation_hints: str = "",
        focus_entities: str = "",
    ) -> str:
        """Extract open-form relations from text without schema typing."""
        params = {
            "text": text,
            "extracted_entities": extracted_entities,
            "previous_results": previous_results,
            "feedbacks": feedbacks,
            "memory_context": memory_context,
            "relation_hints": relation_hints,
            "focus_entities": focus_entities,
        }
        result = self.open_relation_extraction.call(params=json.dumps(params))
        return result

    def ground_open_relations(
        self,
        text: str,
        proposals: Any,
        memory_context: str = "",
    ) -> str:
        """Ground open-form relation proposals to schema relation types."""
        params = {
            "text": text,
            "proposals": proposals,
            "memory_context": memory_context,
        }
        result = self.schema_relation_grounding.call(params=json.dumps(params))
        return result

    def extract_interactions(
        self,
        text: str,
        extracted_entities: str = None,
        extracted_interactions: str = None,
        interaction_group: str = None,
        previous_results: str = None,
        feedbacks: str = None,
    ) -> str:
        """Extract interaction records from text."""
        params = {
            "text": text,
            "extracted_entities": extracted_entities,
            "extracted_interactions": extracted_interactions,
            "interaction_group": interaction_group,
            "previous_results": previous_results,
            "feedbacks": feedbacks,
        }
        result = self.interaction_extraction.call(params=json.dumps(params))
        return result

    # ---------------- Attributes ----------------
    def extract_entity_properties(
        self,
        text: str,
        entity_name: str,
        entity_type: str
    ) -> str:
        params = {
            "text": text,
            "entity_name": entity_name,
            "entity_type": entity_type
        }
        result = self.property_extraction.call(params=json.dumps(params))
        return result
    
    def merge_entity_properties(
        self,
        entity_name: str,
        properties: str,
        full_description: str,
        num_properties: int = 5
    ) -> str:
        params = {
            "entity_name": entity_name,
            "properties": properties,
            "num_properties": num_properties,
            "full_description": full_description
        }
        result = self.property_merger.call(params=json.dumps(params))
        return result

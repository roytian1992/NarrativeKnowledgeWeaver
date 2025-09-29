# -*- coding: utf-8 -*-
"""
Extractor Module
================

This module provides a unified **information extraction interface** that connects
to multiple LLM-backed function tools. It serves as the central entry point for
entity, relation, attribute, and CMP (Costume/Makeup/Props) extraction tasks,
with optional reflection for iterative improvement.

Supported tasks:
----------------
- Entity extraction
- Relation extraction
- Extraction reflection (quality audit)
- Attribute extraction & reflection
- Prop item extraction
- Wardrobe (costume) extraction
- Styling (makeup/hair) extraction
- CMP reflection

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
    >>> entities = extractor.extract_entities(
    ...     text="John picked up the sword and left the castle.",
    ...     entity_type_description_text="Character, Object, Location",
    ...     system_prompt="Extract entities from text",
    ...     reflection_results={}
    ... )
"""

import json
from typing import Dict, Any
from core.utils.config import KAGConfig
from core.functions.regular_functions import (
    EntityExtractor,
    RelationExtractor,
    ExtractionReflector,
    AttributeExtractor,
    AttributeReflector,
    PropItemExtractor,
    WardrobeExtractor,
    StylingExtractor,
    CMPReflector,
)
from core.utils.prompt_loader import PromptLoader
import os


class InformationExtractor:
    """
    High-level information extractor facade.

    Wraps multiple tool interfaces (entity, relation, attribute, CMP) and
    exposes them as easy-to-use methods. Supports optional reflection
    for iterative quality improvements.
    """

    def __init__(self, config: KAGConfig, llm, prompt_loader: PromptLoader = None):
        self.config = config
        self.llm = llm
        if not prompt_loader:
            prompt_dir = self.config.knowledge_graph_builder.prompt_dir
            self.prompt_loader = PromptLoader(prompt_dir)
        else:
            self.prompt_loader = prompt_loader

        # Wrapped function tools
        self.entity_extraction = EntityExtractor(self.prompt_loader, self.llm)
        self.relation_extraction = RelationExtractor(self.prompt_loader, self.llm)
        self.extraction_reflection = ExtractionReflector(self.prompt_loader, self.llm)
        self.attribute_extraction = AttributeExtractor(self.prompt_loader, self.llm)
        self.attribute_reflection = AttributeReflector(self.prompt_loader, self.llm)
        self.propitem_extraction = PropItemExtractor(self.prompt_loader, self.llm)
        self.styling_extraction = StylingExtractor(self.prompt_loader, self.llm)
        self.wardrobe_extraction = WardrobeExtractor(self.prompt_loader, self.llm)
        self.custume_reflection = CMPReflector(self.prompt_loader, self.llm)

    # ---------------- Entity & Relation ----------------

    def extract_entities(
        self,
        text: str,
        entity_type_description_text: str,
        system_prompt: str,
        reflection_results: dict,
        enable_thinking: bool = True,
    ) -> str:
        """Extract entities from text."""
        params = {
            "text": text,
            "entity_type_description_text": entity_type_description_text,
            "system_prompt": system_prompt,
            "reflection_results": reflection_results,
        }
        result = self.entity_extraction.call(params=json.dumps(params), enable_thinking=enable_thinking)
        return result

    def extract_relations(
        self,
        text: str,
        entity_list: str,
        relation_type_description_text: str,
        system_prompt: str,
        reflection_results: dict | str,
        enable_thinking: bool = True,
    ) -> str:
        """Extract relations from text."""
        params = {
            "text": text,
            "entity_list": entity_list,
            "relation_type_description_text": relation_type_description_text,
            "reflection_results": reflection_results,
            "system_prompt": system_prompt,
        }
        result = self.relation_extraction.call(params=json.dumps(params), enable_thinking=enable_thinking)
        return result

    def reflect_extractions(
        self,
        logs: str,
        entity_type_description_text: str,
        relation_type_description_text: str,
        system_prompt: str,
        original_text: str = None,
        previous_reflection: dict | str = None,
        enable_thinking: bool = True,
        version: str = "default",
    ) -> str:
        """Reflect on extraction quality and produce feedback."""
        params = {
            "logs": logs,
            "entity_type_description_text": entity_type_description_text,
            "relation_type_description_text": relation_type_description_text,
            "original_text": original_text,
            "system_prompt": system_prompt,
            "previous_reflection": previous_reflection,
            "version": version,
        }
        result = self.extraction_reflection.call(params=json.dumps(params), enable_thinking=enable_thinking)
        return result

    # ---------------- Attributes ----------------

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
        enable_thinking: bool = True,
    ) -> str:
        """Extract structured attributes for a given entity."""
        params = {
            "text": text,
            "description": description,
            "entity_name": entity_name,
            "entity_type": entity_type,
            "attribute_definitions": attribute_definitions,
            "system_prompt": system_prompt,
            "previous_results": previous_results,
            "feedbacks": feedbacks,
            "original_text": original_text,
            "enable_thinking": enable_thinking,
        }
        result = self.attribute_extraction.call(params=json.dumps(params))
        return result

    def reflect_entity_attributes(
        self,
        entity_type: str,
        description: str,
        attribute_definitions: str,
        attributes: str,
        original_text: str = "",
        system_prompt: str = "",
        enable_thinking: bool = True,
    ) -> str:
        """Reflect on attribute extraction quality (completeness, correctness, retry needs)."""
        params = {
            "entity_type": entity_type,
            "description": description,
            "attribute_definitions": attribute_definitions,
            "attributes": attributes,
            "original_text": original_text,
            "system_prompt": system_prompt,
            "enable_thinking": enable_thinking,
        }
        result = self.attribute_reflection.call(params=json.dumps(params))
        return result

    # ---------------- CMP (Costume / Makeup / Props) ----------------

    def extract_propitem(self, content: str, system_prompt: str, reflection_results: dict) -> str:
        """Extract prop items from text."""
        params = {"content": content, "system_prompt": system_prompt, "reflection_results": reflection_results}
        result = self.propitem_extraction.call(params=json.dumps(params))
        return result

    def extract_wardrobe(self, content: str, system_prompt: str, reflection_results: dict) -> str:
        """Extract wardrobe (costume) items from text."""
        params = {"content": content, "system_prompt": system_prompt, "reflection_results": reflection_results}
        result = self.wardrobe_extraction.call(params=json.dumps(params))
        return result

    def extract_styling(self, content: str, system_prompt: str, reflection_results: dict) -> str:
        """Extract styling (makeup/hair) details from text."""
        params = {"content": content, "system_prompt": system_prompt, "reflection_results": reflection_results}
        result = self.styling_extraction.call(params=json.dumps(params))
        return result

    def reflect_cmp_extractions(
        self,
        logs: str,
        content: str,
        system_prompt: str,
        previous_reflection: dict | str = None,
    ) -> str:
        """Reflect on the overall quality of CMP (Costume/Makeup/Props) extractions."""
        params = {
            "logs": logs,
            "content": content,
            "system_prompt": system_prompt,
            "previous_reflection": previous_reflection,
        }
        result = self.custume_reflection.call(params=json.dumps(params))
        return result

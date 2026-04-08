# -*- coding: utf-8 -*-
"""
Graph Manager Module
====================

This module provides narrative-analysis utilities for knowledge graph
construction. It focuses on event-level and higher-level narrative unit
analysis, operating on extracted entities/relations and refining them into
structured narrative layers.

Key functionalities:
- Check causality between events.
- Detect redundancy among events and relations.
- Generate higher-level narrative units from event chains.
- Extract and classify relations between narrative units.
- Generate enriched narrative context for events.

Class:
    GraphManager
        Provides interfaces for event-level and narrative-level reasoning tasks.

Usage:
    - manager = GraphManager(config, llm)
    - causal = manager.check_event_causality(event1, event2)
    - redundant = manager.evaluate_event_redundancy(events, relations)
    - plot = manager.generate_event_plot(event_chain)
    - rel = manager.extract_plot_relation(plotA, plotB)
    - ctx = manager.generate_event_context(event, background)
"""

import json
from typing import Dict, Any
from core.utils.config import KAGConfig
from core.functions.aggregation_functions import (
    EpisodeExtractor,
    NarrativeRelationExtractor,
    CausalLinkPruner,
    StorylineExtractor
)
from core.utils.prompt_loader import YAMLPromptLoader
import os


class NarrativeManager:
    """
    Event-level reasoning utility.

    This class manages event causality checking, redundancy evaluation,
    narrative unit generation, narrative relation extraction, and event context
    generation for narrative knowledge graphs.
    """

    def __init__(self, config: KAGConfig, llm):
        self.config = config
        self.llm = llm

        prompt_dir = self.config.global_config.prompt_dir
        self.prompt_loader = YAMLPromptLoader(prompt_dir)
        schema_dir = self.config.global_config.schema_dir
        entity_schema_path = os.path.join(schema_dir, "default_entity_schema.json")
        narrative_entity_schema_path = os.path.join(schema_dir, "default_narrative_entity_schema.json")
        narrative_relation_schema_path = os.path.join(schema_dir, "default_narrative_relation_schema.json")

        self.episode_extractor = EpisodeExtractor(
            llm=self.llm,
            prompt_loader=self.prompt_loader,
            entity_schema_path=entity_schema_path,
            narrative_entity_schema_path=narrative_entity_schema_path,
        )
        self.narrative_relation_extractor = NarrativeRelationExtractor(
            llm=self.llm,
            prompt_loader=self.prompt_loader,
            schema_path=narrative_relation_schema_path,
            narrative_entity_schema_path=narrative_entity_schema_path,
        )
        self.causal_link_pruner = CausalLinkPruner(llm=self.llm, prompt_loader=self.prompt_loader)
        self.storyline_extractor = StorylineExtractor(
            llm=self.llm,
            prompt_loader=self.prompt_loader,
            narrative_entity_schema_path=narrative_entity_schema_path,
        )

    def extract_episodes(
        self,
        text: str,
        entities: str,
        existing_episodes: str = "",
        goal: str = "extract"
    ) -> str:
        """Extract episodes from text and entities."""
        params = {
            "text": text,
            "entities": entities,
            "existing_episodes": existing_episodes,
            "goal": goal,
        }
        result = self.episode_extractor.call(params=json.dumps(params))
        return result
    
    def extract_storyline(
        self,
        chain_information: str
    ) -> str:
        """Extract storylines from episode chains"""
        params = {
            "chain_information": chain_information
        }
        result = self.storyline_extractor.call(params=json.dumps(params))
        return result
    
    def extract_narrative_relation(
        self,
        subject_entity_info: Dict[str, Any],
        object_entity_info: Dict[str, Any],
        entity_type: str
    ) -> str:
        """
        Classify relation between two narrative units (e.g., Episode / Storyline).

        Inputs:
          - subject_entity_info: dict with id/name/description/properties/etc.
          - object_entity_info: dict with id/name/description/properties/etc.
          - entity_type: "Episode" or "Storyline" (or your allowed enum)
        
        """
        params = {
            "subject_entity_info": subject_entity_info,
            "object_entity_info": object_entity_info,
            "entity_type": entity_type,
        }
        result = self.narrative_relation_extractor.call(params=json.dumps(params))
        return result

    def prune_causal_edge(
        self,
        *,
        entity_a: Any,
        entity_b: Any,
        entity_c: Any,
        relation_ab: Any,
        relation_bc: Any,
        relation_ac: Any,
    ) -> str:
        """
        Evaluate whether the direct edge A -> C should be removed given A -> B -> C.

        All inputs are variables:
          - entity_a / entity_b / entity_c: any JSON-serializable object or string
          - relation_ab / relation_bc / relation_ac: any JSON-serializable object or string

        Returns JSON-only string:
          {"remove_edge": bool, "reason": str}
        """
        params = {
            "entity_a": entity_a,
            "entity_b": entity_b,
            "entity_c": entity_c,
            "relation_ab": relation_ab,
            "relation_bc": relation_bc,
            "relation_ac": relation_ac,
        }
        result = self.causal_link_pruner.call(
            params=json.dumps(params, ensure_ascii=False),
        )
        return result
    

# -*- coding: utf-8 -*-
"""
Graph Manager Module
====================

This module provides **graph-level reasoning utilities** for knowledge graph
construction. It focuses on event-level and plot-level analysis, operating on
extracted entities/relations and refining them into structured narratives.

Key functionalities:
- Check causality between events.
- Detect redundancy among events and relations.
- Generate higher-level plot units from event chains.
- Extract and classify relations between plot units.
- Generate enriched narrative context for events.

Class:
    GraphManager
        Provides interfaces for all event/plot-level graph reasoning tasks.

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
from core.functions.regular_functions import (
    EventCausalityChecker,
    RedundancyEvaluator,
    PlotUnitExtractor,
    PlotRelationExtractor,
    EventContextGenerator,
)
from core.utils.prompt_loader import PromptLoader
import os


class GraphManager:
    """
    Graph-level reasoning utility.

    This class manages event causality checking, redundancy evaluation,
    plot unit generation, plot relation extraction, and event context
    generation for narrative knowledge graphs.
    """

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
        related_context: str = "",
    ) -> str:
        """Check whether two events are causally related."""
        params = {
            "event_1_info": event_1_info,
            "event_2_info": event_2_info,
            "system_prompt": system_prompt,
            "related_context": related_context,
        }
        result = self.event_causality_checker.call(params=json.dumps(params))
        return result

    def evaluate_event_redundancy(
        self,
        event_details: str,
        relation_details: str,
        system_prompt: str = "",
        related_context: str = "",
    ):
        """Evaluate whether events or relations are redundant/overlapping."""
        params = {
            "event_details": event_details,
            "relation_details": relation_details,
            "system_prompt": system_prompt,
            "related_context": related_context,
        }
        result = self.redundancy_evaluator.call(params=json.dumps(params))
        return result

    def generate_event_plot(
        self,
        event_chain_info: str,
        system_prompt: str = "",
        related_context: str = "",
    ):
        """Generate plot units (higher-level narrative segments) from an event chain."""
        params = {
            "event_chain_info": event_chain_info,
            "system_prompt": system_prompt,
            "related_context": related_context,
        }
        result = self.plot_generator.call(params=json.dumps(params))
        return result

    def extract_plot_relation(
        self,
        plot_A_info: str,
        plot_B_info: str,
        system_prompt: str = "",
    ):
        """Extract and classify the narrative relation between two plot units."""
        params = {
            "plot_A_info": plot_A_info,
            "plot_B_info": plot_B_info,
            "system_prompt": system_prompt,
        }
        result = self.plot_relation_extractor.call(params=json.dumps(params))
        return result

    def generate_event_context(
        self,
        event_info: str,
        related_context: str,
        system_prompt: str = "",
    ):
        """Generate extended narrative context for a given event."""
        params = {
            "event_info": event_info,
            "related_context": related_context,
            "system_prompt": system_prompt,
        }
        result = self.event_context_generator.call(params=json.dumps(params))
        return result

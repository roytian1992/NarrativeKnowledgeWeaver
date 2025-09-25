import json
import logging
from typing import Dict, List, Any
from pathlib import Path
import os
import re
import asyncio
from collections import Counter

from tqdm import tqdm
from langgraph.graph import StateGraph, END

from core.utils.format import correct_json_format
from core.builder.manager.information_manager import InformationExtractor
from core.builder.manager.probing_manager import GraphProber
from core.agent.knowledge_extraction_agent import InformationExtractionAgent
from core.model_providers.openai_rerank import OpenAIRerankModel
from core.utils.prompt_loader import PromptLoader
from ..utils.config import KAGConfig

logger = logging.getLogger(__name__)


class GraphProbingAgent:
    """
    Probing agent that iteratively refines a graph schema (entities & relations)
    using: (i) retrieved 'experience' (insights), (ii) probing prompts, and
    (iii) test extractions + reflection/feedback loops.

    Inputs/Outputs and external behaviors are kept identical to the original code.
    All user-facing messages are English; prints replaced with logging.
    """

    def __init__(self, config: KAGConfig, llm, reflector):
        self.config = config
        self.llm = llm
        self.extractor = InformationExtractor(config, llm)
        prompt_dir = config.knowledge_graph_builder.prompt_dir
        self.prompt_loader = PromptLoader(prompt_dir)
        self.prober = GraphProber(config, llm)
        self.reranker = OpenAIRerankModel(config)
        self.doc_type = config.knowledge_graph_builder.doc_type
        self.reflector = reflector
        self.graph = self._build_graph()
        self.feedbacks: List[str] = []
        self.score_threshold = self.config.agent.score_threshold
        self.relation_prune_threshold = self.config.probing.relation_prune_threshold
        self.entity_prune_threshold = self.config.probing.entity_prune_threshold
        self.max_workers = self.config.probing.max_workers
        self.max_retries = self.config.probing.max_retries
        self.experience_limit = self.config.probing.experience_limit
        self.probing_mode = self.config.probing.probing_mode
        self.refine_background = self.config.probing.refine_background

        if self.config.probing.task_goal:
            self.task_goals = "\n".join(self.load_goals(self.config.probing.task_goal))
        else:
            self.task_goals = ""

    # --------------------------------------------------------------------- #
    # Prompt construction helpers
    # --------------------------------------------------------------------- #
    def construct_system_prompt(self, background, abbreviations):
        """
        Render the system prompt by combining background and abbreviations.
        """
        background_info = self.get_background_info(background, abbreviations)

        if self.doc_type == "screenplay":
            system_prompt_id = "agent_prompt_screenplay"
        else:
            system_prompt_id = "agent_prompt_novel"

        system_prompt_text = self.prompt_loader.render_prompt(
            system_prompt_id, {"background_info": background_info}
        )
        return system_prompt_text

    def load_goals(self, path: str):
        """
        Load task goals from a text file.
        - Skips empty lines or comments starting with '#'
        - Strips leading indices/symbols like '1.', '2)', '3、', '4:' or '-'
        - Deduplicates while preserving order
        """
        text = Path(path).read_text(encoding="utf-8-sig")
        goals = []
        for line in text.splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            s = re.sub(r'^\s*(?:[-•]+|\d+[.)、:：]?)\s*', '', s)
            goals.append(s)
        return list(dict.fromkeys(goals))

    def get_background_info(self, background, abbreviations):
        """
        Render background and a glossary list into Markdown.

        Expected abbreviation fields (all optional except description):
          - name, description (required), abbr, full, zh

        If both 'name' and 'abbr' are missing, or description is missing,
        the item is skipped. Title preference: name (with abbr) > name > abbr > 'N/A'.
        """
        def _s(val):
            return val.strip() if isinstance(val, str) and val.strip() else None

        bg_block = f"**Background**: {background}\n" if _s(background) else ""

        def fmt(item: dict) -> str:
            if not isinstance(item, dict):
                return ""

            name = _s(item.get("name"))
            desc = _s(item.get("description"))
            abbr = _s(item.get("abbr"))
            full = _s(item.get("full"))
            zh   = _s(item.get("zh"))

            if not desc or (not name and not abbr):
                return ""

            # Title: prefer name; if abbr exists and differs, show "name (abbr)"
            if name and abbr and name.lower() != abbr.lower():
                title = f"{name} ({abbr})"
            else:
                title = name or abbr or "N/A"

            parts = [desc]
            if full:
                parts.append(full)
            if zh:
                parts.append(zh)

            return f"- **{title}**: " + " - ".join(parts) if parts else f"- **{title}**"

        abbr_lines: List[str] = []
        if isinstance(abbreviations, list):
            for it in abbreviations:
                line = fmt(it)
                if line:
                    abbr_lines.append(line)

        abbr_block = "\n".join(abbr_lines)

        if bg_block and abbr_block:
            return f"{bg_block}\nHere are some domain terms and abbreviations:\n{abbr_block}"
        else:
            return bg_block or abbr_block

    def load_schema(self, schema):
        """
        Flatten schema into readable text blocks for entities and relations.
        """
        entity_types = schema.get("entities")
        relation_type_groups = schema.get("relations")

        entity_type_description_text = "\n".join(
            f"- {e['type']}: {e['description']}" for e in entity_types
        )
        relation_type_description_text = "\n".join(
            f"- {r['type']}: {r['description']}"
            for group in relation_type_groups.values()
            for r in group
        )
        return entity_type_description_text, relation_type_description_text

    # --------------------------------------------------------------------- #
    # Experience search (insight memory + reranking)
    # --------------------------------------------------------------------- #
    def search_related_experience(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Retrieve previously stored insights to guide background, abbreviations,
        and schema refinement. Queries use bilingual keywords (English/Chinese)
        to maximize recall for existing Chinese insights.
        """
        k = 2 * self.experience_limit

        # Background / plot / events
        docs_bg = self.reflector.insight_memory.get(
            query="background, story, plot, events, characters; 背景信息、故事、情节、事件、人物",
            k=k
        )
        documents_bg = [doc.page_content for doc in docs_bg]
        reranked_bg = self.reranker.rerank(
            query="background, story, plot, events; 背景信息、故事、情节、事件",
            documents=documents_bg,
            top_n=min(self.experience_limit, len(documents_bg))
        )
        related_insights_for_background = [
            item["document"]["text"]
            for item in reranked_bg
            if item.get("relevance_score", 0) >= 0.3
        ]
        # Keep the original functional note about Plot to preserve behavior.
        related_insights_for_background += [
            "Entity type 'Plot' does not need to be extracted; it will be constructed later based on extracted 'Event' entities. （情节/Plot这种实体类型不需要抽取，会在后续的任务中基于抽取的事件/Event进行构建。）"
        ]

        # Abbreviations / glossary
        docs_abbr = self.reflector.insight_memory.get(
            query="terms, glossary, abbreviations; 术语、缩写",
            k=k
        )
        documents_abbr = [doc.page_content for doc in docs_abbr]
        reranked_abbr = self.reranker.rerank(
            query="terms, abbreviations; 术语、缩写",
            documents=documents_abbr,
            top_n=min(self.experience_limit, len(documents_abbr))
        )
        related_insights_for_abbreviations = [
            item["document"]["text"]
            for item in reranked_abbr
            if item.get("relevance_score", 0) >= 0.5 and ("术语" in item["document"]["text"] or "term" in item["document"]["text"].lower())
        ]

        # Entity schema
        docs_ent = self.reflector.insight_memory.get(
            query="entities, characters, objects, emotions, actions, events, concepts; 人物、物品、情感、动作、事件、概念、实体",
            k=k
        )
        documents_ent = [doc.page_content for doc in docs_ent]
        reranked_ent = self.reranker.rerank(
            query="entity schema; 实体 schema",
            documents=documents_ent,
            top_n=min(self.experience_limit, len(documents_ent))
        )
        related_insights_for_entity_schema = [
            item["document"]["text"]
            for item in reranked_ent
            if item.get("relevance_score", 0) >= 0.5
        ]

        # Relation schema
        docs_rel = self.reflector.insight_memory.get(
            query="events, actions, relations; 事件、动作、关系",
            k=k
        )
        documents_rel = [doc.page_content for doc in docs_rel]
        reranked_rel = self.reranker.rerank(
            query="relation schema; 关系 schema",
            documents=documents_rel,
            top_n=min(self.experience_limit, len(documents_rel))
        )
        related_insights_for_relation_schema = [
            item["document"]["text"]
            for item in reranked_rel
            if item.get("relevance_score", 0) >= 0.5
        ]

        return {
            **state,
            "background_insights": "\n".join(related_insights_for_background),
            "abbreviation_insights": "\n".join(related_insights_for_abbreviations),
            "entity_schema_insights": "\n".join(related_insights_for_entity_schema),
            "relation_schema_insights": "\n".join(related_insights_for_relation_schema),
        }

    # --------------------------------------------------------------------- #
    # Schema generation / updates
    # --------------------------------------------------------------------- #
    def generate_schema(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update background + abbreviations (optional), then update entity/relation schemas
        using insights and accumulated feedbacks.
        """
        # 1) Background & abbreviations
        if self.refine_background:
            current_background_info = state.get("background", "")
            # Prepend summaries to background to keep behavior consistent
            current_background_info = "\n".join(state.get("summaries", [])) + "\n" + current_background_info

            background_insights = state.get("background_insights", "")
            result = self.prober.update_background(
                text=background_insights,
                current_background=current_background_info
            )
            result = json.loads(correct_json_format(result))
            new_background = result.get("background", "") or current_background_info

            current_abbreviations = state.get("abbreviations", [])
            abbreviation_insights = state.get("abbreviation_insights", "")
            background_info = self.get_background_info(new_background, current_abbreviations)

            result = self.prober.update_abbreviations(
                text=abbreviation_insights,
                current_background=background_info
            )
            result = json.loads(correct_json_format(result))
            abbreviations_ = result.get("abbreviations", [])

            new_abbreviations = current_abbreviations.copy()
            current_abbr_list = [item.get("name") for item in current_abbreviations]
            for item in abbreviations_:
                if item.get("name") not in current_abbr_list:
                    new_abbreviations.append(item)

        else:
            new_background = state.get("background", "")
            new_abbreviations = state.get("abbreviations", [])

        # 2) Entity schema
        current_schema = state.get("schema", {})
        entity_schema_insights = state.get("entity_schema_insights", "")
        current_entity_schema = current_schema.get("entities", {})
        entity_schema_feedbacks = "\n".join(self.feedbacks)

        result = self.prober.update_entity_schema(
            text=entity_schema_insights,
            feedbacks=entity_schema_feedbacks,
            current_schema=json.dumps(current_entity_schema, indent=2, ensure_ascii=False),
            task_goals=self.task_goals
        )
        result = json.loads(correct_json_format(result))
        new_entity_schema = result.get("entities", [])

        # 3) Relation schema
        relation_schema_insights = state.get("relation_schema_insights", "")
        current_relation_schema = current_schema.get("relations", {})
        relation_schema_feedbacks = "\n".join(self.feedbacks)

        result = self.prober.update_relation_schema(
            text=relation_schema_insights,
            feedbacks=relation_schema_feedbacks,
            current_schema=json.dumps(current_relation_schema, indent=2, ensure_ascii=False),
            task_goals=self.task_goals
        )
        result = json.loads(correct_json_format(result))
        new_relation_schema = result.get("relations", [])

        # Clear temporary feedback memory
        self.feedbacks = []

        return {
            **state,
            "schema": {"entities": new_entity_schema, "relations": new_relation_schema},
            "background": new_background,
            "abbreviations": new_abbreviations,
        }

    # --------------------------------------------------------------------- #
    # Test extractions to measure schema quality
    # --------------------------------------------------------------------- #
    def test_extractions(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run test extractions over sampled documents using the current schema,
        gather type distributions, and compute drop lists based on thresholds.
        """
        all_documents = state["documents"]
        system_prompt = self.construct_system_prompt(state["background"], state["abbreviations"])
        schema = state["schema"]

        extraction_results = asyncio.run(
            self.test_extractions_(all_documents, system_prompt, schema)
        )

        entity_types: List[str] = []
        relation_types: List[str] = []

        for result in extraction_results:
            try:
                # Accumulate issues (feedback memory)
                self.feedbacks.extend(result.get("issues", []))
                entities = result.get("entities", [])
                for entity in entities:
                    entity_types.append(entity.get("type"))
                relations = result.get("relations", [])
                for relation in relations:
                    # Note: keep original key 'relation_type' to match data shape
                    relation_types.append(relation.get("relation_type"))
            except Exception as _:
                logger.warning("Unexpected extraction result format; skipping one result.")

        entity_type_counter = Counter(entity_types)
        relation_type_counter = Counter(relation_types)

        entity_total = len(entity_types)
        relation_total = len(relation_types)

        # Types to drop (by proportion)
        if entity_total > 0:
            entity_types_to_drop = [
                key for key, cnt in entity_type_counter.items()
                if (cnt / entity_total) < self.entity_prune_threshold
            ]
        else:
            entity_types_to_drop = []

        if relation_total > 0:
            relation_types_to_drop = [
                key for key, cnt in relation_type_counter.items()
                if (cnt / relation_total) < self.relation_prune_threshold
            ]
        else:
            relation_types_to_drop = []

        # Distributions (English)
        entity_type_distribution = ""
        if entity_total > 0:
            for _type, cnt in entity_type_counter.items():
                entity_type_distribution += (
                    f"Entity type {_type}: count = {cnt}, ratio = {round(cnt / entity_total, 3)}\n"
                )

        relation_type_distribution = ""
        if relation_total > 0:
            for _type, cnt in relation_type_counter.items():
                relation_type_distribution += (
                    f"Relation type {_type}: count = {cnt}, ratio = {round(cnt / relation_total, 3)}\n"
                )

        logger.info("Entity types suggested to drop: %s", entity_types_to_drop)
        logger.info("Relation types suggested to drop: %s", relation_types_to_drop)

        return {
            **state,
            "entity_type_distribution": entity_type_distribution,
            "relation_type_distribution": relation_type_distribution,
            "entity_types_to_drop": entity_types_to_drop,
            "relation_types_to_drop": relation_types_to_drop,
        }

    async def test_extractions_(self, all_documents, system_prompt, schema):
        """
        Async worker that runs extraction with 'probing' mode and returns raw results.
        """
        information_extraction_agent = InformationExtractionAgent(
            self.config, self.llm, system_prompt,
            schema=schema, reflector=self.reflector, mode="probing"
        )
        sem = asyncio.Semaphore(self.max_workers)

        async def _arun(ch):
            async with sem:
                try:
                    if not ch.content.strip():
                        result = {"entities": [], "relations": [], "score": 0, "issues": [], "insights": []}
                    else:
                        result = await information_extraction_agent.arun(
                            ch.content, timeout=300, max_attempts=3, backoff_seconds=60
                        )
                    return result
                except Exception as e:
                    return {"error": str(e), "entities": [], "relations": [], "score": 0, "issues": [], "insights": []}

        tasks = [_arun(ch) for ch in all_documents]
        results = []
        for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Async extraction (probing)"):
            res = await coro
            results.append(res)
        return results

    # --------------------------------------------------------------------- #
    # Pruning and reflection
    # --------------------------------------------------------------------- #
    def prune_graph_schema(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ask the prober to prune low-signal types based on observed distributions.
        Also accumulate feedbacks returned by the prober.
        """
        entity_type_distribution = state["entity_type_distribution"]
        relation_type_distribution = state["relation_type_distribution"]
        entity_type_description_text, relation_type_description_text = self.load_schema(state["schema"])

        result = self.prober.prune_schema(
            entity_type_distribution=entity_type_distribution,
            relation_type_distribution=relation_type_distribution,
            entity_type_description_text=entity_type_description_text,
            relation_type_description_text=relation_type_description_text
        )
        result = json.loads(correct_json_format(result))
        self.feedbacks.extend(result.get("feedbacks", []))  # accumulate memory
        return state

    def reflect_graph_schema(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Summarize feedbacks in batches and reflect to produce a score & reason.
        Apply pruning by removing types previously marked as drop.
        Keep the best-so-far schema by score.
        """
        current_schema = state["schema"]
        BATCH_SIZE = 300

        # Batch summaries of feedbacks
        summaries: List[str] = []
        for i in range(0, len(self.feedbacks), BATCH_SIZE):
            batch = self.feedbacks[i:i + BATCH_SIZE]
            all_feedbacks = "\n".join(batch)
            result = self.prober.summarize_feedbacks(context=all_feedbacks, max_items=20)
            parsed = json.loads(correct_json_format(result)).get("feedbacks", [])
            summaries.extend(parsed)

        self.feedbacks = summaries

        result = self.prober.reflect_schema(
            schema=json.dumps(current_schema, indent=2, ensure_ascii=False),
            feedbacks=self.feedbacks
        )
        result = json.loads(correct_json_format(result))
        score = float(result["score"])
        reason = result["reason"]

        # Remove low-frequency types (< thresholds) as computed in test_extractions
        entities = current_schema.get("entities", [])
        new_entities = [ent for ent in entities if ent["type"] not in state.get("entity_types_to_drop", [])]

        relations = current_schema.get("relations", {})
        new_relations = {}
        for rel_group, rels in relations.items():
            new_rels = [rel for rel in rels if rel["type"] not in state.get("relation_types_to_drop", [])]
            new_relations[rel_group] = new_rels

        new_schema = {"entities": new_entities, "relations": new_relations}
        best_score = state.get("best_score", 0)
        current_output = {
            "schema": new_schema,
            "settings": {"background": state["background"], "abbreviations": state["abbreviations"]}
        }

        if score > best_score:
            best_output = current_output
        else:
            best_output = state.get("best_output", {})

        # Keep overall distributions as feedback for next round
        self.feedbacks.append(state.get("entity_type_distribution", ""))
        self.feedbacks.append(state.get("relation_type_distribution", ""))

        return {
            **state,
            "score": score,
            "reason": reason,
            "retry_count": state["retry_count"] + 1,
            "best_score": max(score, best_score),
            "best_output": best_output
        }

    # --------------------------------------------------------------------- #
    # Control flow graph (LangGraph)
    # --------------------------------------------------------------------- #
    def _score_check(self, state: Dict[str, Any]) -> str:
        """
        Decide whether the current schema is good, should retry, or give up.
        """
        if state["score"] >= self.score_threshold:
            result = "good"
        elif state["retry_count"] >= self.max_retries:
            result = "giveup"
        else:
            result = "retry"
        return result

    def _build_graph(self):
        """
        Build the LangGraph state machine for probing.
        """
        builder = StateGraph(dict)
        builder.add_node("search_related_experience", self.search_related_experience)
        builder.add_node("generate_schema", self.generate_schema)
        builder.add_node("test_extractions", self.test_extractions)
        builder.add_node("prune_graph_schema", self.prune_graph_schema)
        builder.add_node("reflect_graph_schema", self.reflect_graph_schema)

        builder.set_entry_point("search_related_experience")
        builder.add_edge("search_related_experience", "generate_schema")
        builder.add_edge("generate_schema", "test_extractions")
        builder.add_edge("test_extractions", "prune_graph_schema")
        builder.add_edge("prune_graph_schema", "reflect_graph_schema")

        builder.add_conditional_edges(
            "reflect_graph_schema", self._score_check, {
                "good": END,
                "retry": "search_related_experience",
                "giveup": END
            }
        )

        return builder.compile()

    # --------------------------------------------------------------------- #
    # Public entry
    # --------------------------------------------------------------------- #
    def run(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the probing loop with the given params. Returns the best output
        (schema + settings) if available; otherwise, returns the final state.
        """
        result = self.graph.invoke({
            "documents": params["documents"],
            "schema": params.get("schema", {}),
            "background": params.get("background", ""),
            "abbreviations": params.get("abbreviations", []),
            "summaries": params.get("summaries", []),
            "retry_count": 0,
            "best_score": 0,
            "best_output": {}
        })
        return result.get("best_output", result)

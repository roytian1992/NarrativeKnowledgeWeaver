import json
import re
import asyncio
import logging
from typing import Dict, Any, List
from langgraph.graph import StateGraph, END
from core.utils.format import correct_json_format
from ..storage.vector_store import VectorStore
from core.memory.vector_memory import VectorMemory
from core.builder.manager.information_manager import InformationExtractor
from core.model_providers.openai_rerank import OpenAIRerankModel
from retriever.vectordb_retriever import ParentChildRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from core.builder.manager.document_manager import DocumentParser

logger = logging.getLogger(__name__)


def format_property_definitions(properties: Dict[str, str]) -> str:
    """
    Render a property definition block in bullet form for prompts.

    Args:
        properties: Mapping from property key to its textual description.

    Returns:
        A bullet-list string, one property per line, in the form:
        - **key**: description
    """
    return "\n".join([f"- **{key}**: {desc}" for key, desc in properties.items()])


class AttributeExtractionAgent:
    """
    Attribute Extraction with reflection loop (LangGraph-based).

    Flow:
      1) get_related_context (runs once): Build a consolidated context by
         (a) fetching source chunks and (b) optionally retrieving parent/child
         augmentations for global entity types.
      2) extract: Run attribute extraction with the consolidated context.
      3) reflect: Run reflection to score/fix attributes; if score >= threshold
         or max retries reached -> stop; else -> go back to extract.

    Scoring rule:
      - Prefer 'score' returned by the reflector (0-10).
      - If absent, estimate score from attribute completeness (0-10).
    """

    def __init__(
        self,
        config,
        llm,
        system_prompt,
        schema,
        enable_thinking: bool = True,
        prompt_loader=None,
        global_entity_types=None,
    ):
        """
        Args:
            config: System configuration object.
            llm: LLM provider/adapter used by internal extract/reflect routines.
            system_prompt: System prompt text for extraction & reflection.
            schema: Graph schema containing entity type definitions and properties.
            enable_thinking: Reserved flag for LLM thinking modes (kept for compatibility).
            prompt_loader: Optional prompt loader (injected if needed).
            global_entity_types: Types that should trigger extra retrieval context.
                                 Defaults to ["Character", "Concept", "Object", "Location"].
        """
        self.config = config
        self.extractor = InformationExtractor(config, llm, prompt_loader=prompt_loader)
        self.history_memory = VectorMemory(config, "history_memory")
        self.load_schema(schema)
        self.system_prompt = system_prompt

        # Retrieval resources
        self.reranker = OpenAIRerankModel(config)
        self.document_vector_store = VectorStore(config, "documents")
        self.sentence_vector_store = VectorStore(config, "sentences")
        self.retriever = ParentChildRetriever(
            doc_vs=self.document_vector_store,
            sent_vs=self.sentence_vector_store,
            reranker=self.reranker,
        )
        self.enable_thinking = enable_thinking

        # Retry/score controls (readable defaults if not in config)
        self.max_retries = getattr(config, "attribute_max_retries", 3)
        self.min_score = getattr(config, "attribute_min_score", 7)

        # Build the LangGraph
        self.graph = self._build_graph()

        # Base recursive splitter
        self.base_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.document_processing.chunk_size, chunk_overlap=config.document_processing.chunk_overlap
        )
        self.document_parser = DocumentParser(config, llm)

    # ---------------- Schema parsing ----------------
    def load_schema(self, schema: Dict[str, Any]):
        """
        Parse schema fields required by the agent.

        Expected schema format:
            {
              "entities": [
                {"type": "...", "description": "...", "properties": {...}},
                ...
              ],
              ...
            }
        """
        self.entity_types = schema.get("entities")
        self.schema_type_order = [e["type"] for e in self.entity_types]
        self.type2description = {e["type"]: e["description"] for e in self.entity_types}
        self.type2property = {e["type"]: e["properties"] for e in self.entity_types}

    def _resolve_type(self, entity_type):
        """
        Resolve a possibly-list type into a single canonical type following
        schema order priority; 'Event' dominates if present.
        """
        if isinstance(entity_type, list):
            if "Event" in entity_type:
                return "Event"
            for t in self.schema_type_order:
                if t in entity_type:
                    return t
            return entity_type[0] if entity_type else "Concept"
        return entity_type or "Concept"

    def _resolve_properties(self, entity_type):
        """
        Merge properties if a list of types is provided; 'Event' dominates.
        """
        if isinstance(entity_type, list):
            if "Event" in entity_type:
                return self.type2property.get("Event", {})
            merged = {}
            wanted = set(entity_type)
            for t in self.schema_type_order:
                if t in wanted:
                    for k, v in (self.type2property.get(t, {}) or {}).items():
                        if k not in merged:
                            merged[k] = v
            return merged
        if entity_type == "Event":
            return self.type2property.get("Event", {})
        return self.type2property.get(entity_type, {})

    def _resolve_description(self, entity_type):
        """
        Build a composite description if multiple types appear; 'Event' dominates.
        """
        if isinstance(entity_type, list):
            if "Event" in entity_type:
                return self.type2description.get("Event", "")
            descs = []
            wanted = set(entity_type)
            for t in self.schema_type_order:
                if t in wanted:
                    d = self.type2description.get(t, "")
                    if d:
                        descs.append(f"[{t}] {d}")
            return " ; ".join(descs)
        if entity_type == "Event":
            return self.type2description.get("Event", "")
        return self.type2description.get(entity_type, "")

    # ---------------- Utilities ----------------
    @staticmethod
    def _parse_attribute_keys(attribute_definitions: str) -> List[str]:
        """
        Parse property keys from bullet lines of the form '- **key**: desc'.
        """
        keys = re.findall(r"-\s*\*\*(.+?)\*\*", attribute_definitions or "")
        return [k.strip() for k in keys if k.strip()]

    @staticmethod
    def _ensure_dict(maybe_json_or_dict):
        """
        Ensure the input is a dict.
        - If string, try to parse JSON; if not JSON, return empty dict.
        """
        if isinstance(maybe_json_or_dict, dict):
            return maybe_json_or_dict
        if isinstance(maybe_json_or_dict, str) and maybe_json_or_dict.strip():
            try:
                return json.loads(maybe_json_or_dict)
            except Exception:
                return {}
        return {}

    @staticmethod
    def _json_dumps(obj) -> str:
        """
        Safe JSON dumps with non-ASCII preserved.
        """
        try:
            return json.dumps(obj, ensure_ascii=False)
        except Exception:
            return "{}"

    def _estimate_score_from_completeness(self, attrs: Dict[str, Any], expected_keys: List[str]) -> float:
        """
        Estimate a 0-10 score from attribute completeness when reflector score is missing.
        """
        if not expected_keys:
            # No schema: fall back to non-empty ratio over current keys
            all_keys = list(attrs.keys())
            expected_keys = all_keys
        if not expected_keys:
            return 0.0

        non_empty = 0
        for k in expected_keys:
            v = attrs.get(k, "")
            if isinstance(v, str):
                non_empty += 1 if v.strip() else 0
            else:
                non_empty += 1 if v is not None else 0
        ratio = non_empty / max(1, len(expected_keys))
        return round(ratio * 10.0, 2)

    def get_summary_per_chunk(self, text, goal):
        new_text_splitted = self.base_splitter.split_text(text)
        # prev = ""
        summaries = []
        for chunk in new_text_splitted:
            chunk_result = self.document_parser.search_content(text=chunk, goal=goal, max_length=50)
            parsed = json.loads(correct_json_format(chunk_result)).get("related_content", [])
            if parsed:
                summaries.extend(parsed)
        # new_text = "\n".join(summaries) 
        return summaries

    def get_rolling_summary(self, text, goal, max_size=1000):
        # from tqdm import tqdm
        new_text_splitted = self.base_splitter.split_text(text)
        prev = ""
        for chunk in new_text_splitted:
            chunk_result = self.document_parser.summarize_paragraph(chunk, max_size, prev, goal)
            parsed = json.loads(correct_json_format(chunk_result)).get("summary", [])
            prev = parsed
        final_text = parsed
        return final_text
    
    
    def summarize_text(self, text, goal, max_size=1000):
        summaries = self.get_summary_per_chunk(text, goal)
        new_text = "\n".join(summaries)
        if len(new_text) >= max_size:
            final_text = self.get_rolling_summary(text, goal, max_size)
        else:
            final_text = new_text
        return final_text, summaries
    
    # ---------------- Node: build consolidated context (once) ----------------
    def get_related_context(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build a consolidated context exactly once:

        - Load source chunks by IDs from the document vector store.
        - If entity type is "global" (e.g., Character/Object/Location/Concept),
          augment with ParentChildRetriever results.
        - If the merged text is long, summarize it with rolling summaries.
        """

        # print("****", state)
        entity_name = state["entity_name"]
        entity_type = state["entity_type"]
        version = state["version"]
        source_chunks = state.get("source_chunks", []) or []
        additional_chunks = state.get("additional_chunks", []) or []
        goal = f"Please focus on information relevant to '{entity_name}'." 
        feedbacks = state.get("feedbacks", [])
        if feedbacks:
            goal += " Consider the following feedbacks:\n" + "\n".join(f"- {f}" for f in feedbacks if f)
        
        content = state.get("content", "")
        # Fetch original chunks
        doc_objs = self.document_vector_store.search_by_ids(source_chunks) or []
        doc_texts = []
        for d in doc_objs:
            if hasattr(d, "content"):
                doc_texts.append(d.content)
            elif isinstance(d, dict) and "content" in d:
                doc_texts.append(d["content"])

        related_text = "\n".join(doc_texts)
        per_chunk_summaries = state.get("per_chunk_summaries", "")
        if len(related_text) >= 2000 and per_chunk_summaries:
            final_text = self.get_rolling_summary(per_chunk_summaries, goal, 800)
            return {**state, "content": final_text}
        
        if version == "Part_1":
            related_text, per_chunk_summaries = self.summarize_text(related_text, goal, 800)  
            final_text = "当前版本整理出来的相关信息：\n" + related_text    

        else:
            related_text, per_chunk_summaries = self.summarize_text(related_text, goal, 500) 
            doc_objs_prev = self.document_vector_store.search_by_ids(additional_chunks) or []
            doc_texts_prev = []
            for d in doc_objs_prev:
                if hasattr(d, "content"):
                    doc_texts_prev.append(d.content)
                elif isinstance(d, dict) and "content" in d:
                    doc_texts_prev.append(d["content"])
            additional_text = "\n".join(doc_texts_prev)
            additional_text, per_chunk_summaries_ = self.summarize_text(additional_text, goal, 300)
            per_chunk_summaries = ["上一版本（Part_1）的相关信息: \n" + "\n".join(per_chunk_summaries_) + "\n注意：若是存在冲突，以当前版本（Part_2）的信息为准。"] + per_chunk_summaries
            # print("*** related_text: ", type(related_text))
            # print("*** additional_text: ", type(additional_text))
            final_text = "当前版本（Part_2）整理出来的相关信息：\n" + related_text + "\n" + "上一版本（Part_1）的相关信息: \n" + additional_text + "\n注意：若是存在冲突，以当前版本（Part_2）的信息为准。"

        
        return {**state, "content": final_text, "per_chunk_summaries": per_chunk_summaries}

    # ---------------- Node: extract ----------------
    def extract_native(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run attribute extraction on the consolidated context and store the raw result.
        """
        entity_name = state["entity_name"]
        entity_type_raw = state["entity_type"]
        entity_type = self._resolve_type(entity_type_raw)
        type_description = self._resolve_description(entity_type_raw)
        properties = self._resolve_properties(entity_type_raw)

        feedbacks_list = state.get("feedbacks", []) or []
        feedbacks = "\n".join(feedbacks_list)
        attribute_definitions = format_property_definitions(properties)

        result = self.extractor.extract_entity_attributes(
            text=state["content"],                # consolidated context
            entity_name=entity_name,
            description=type_description,
            entity_type=entity_type,
            attribute_definitions=attribute_definitions,
            system_prompt=self.system_prompt,
            previous_results=state.get("previous_result", ""),
            feedbacks=feedbacks,
            enable_thinking=False
        )
        # Normalize and parse JSON output from extractor
        result = json.loads(correct_json_format(result))

        attrs = self._ensure_dict(result.get("attributes", {}))
        new_desc = result.get("new_description", "")

        return {
            **state,
            "attributes": attrs,
            "new_description": new_desc,
            "previous_result": self._json_dumps(result),
        }

    def extract_incremental(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run attribute extraction on the consolidated context and store the raw result.
        """
        entity_name = state["entity_name"]
        entity_type_raw = state["entity_type"]
        entity_type = self._resolve_type(entity_type_raw)
        type_description = self._resolve_description(entity_type_raw)
        properties = self._resolve_properties(entity_type_raw)

        feedbacks_list = state.get("feedbacks", []) or []
        feedbacks = "\n".join(feedbacks_list)

        attribute_definitions = format_property_definitions(properties)
        previous_results = json.loads(state.get("previous_result", ""))
        prev_attributes = previous_results.get("attributes", {})
        prev_description = previous_results.get("new_description", "")


        per_chunk_summaries = state.get("per_chunk_summaries", [])
        full_summary = "\n".join(per_chunk_summaries)
        chunks = self.base_splitter.split_text(full_summary)

        for chunk in chunks:
            previous_results = json.loads(state.get("previous_result", ""))
            result = self.extractor.update_entity_attributes(
                text=chunk,                # consolidated context
                entity_name=entity_name,
                description=type_description,
                entity_type=entity_type,
                attribute_definitions=attribute_definitions,
                system_prompt=self.system_prompt,
                prev_attributes=prev_attributes,
                prev_description=prev_description,
                feedbacks=feedbacks,
                enable_thinking=False
            )
            result = json.loads(correct_json_format(result))
            attrs = self._ensure_dict(result.get("attributes", {}))
            new_desc = result.get("new_description", "")
            attribute_definitions = format_property_definitions(result.get("attribute_definitions", {}))
            prev_attributes = attrs
            prev_description = new_desc

        return {
            **state,
            "attributes": attrs,
            "new_description": new_desc,
            "attribute_definitions": attribute_definitions,
            "previous_result": self._json_dumps(result),
        }
    
    # ---------------- Node: reflect/score/fix ----------------
    def reflect(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Reflect and optionally fix attributes.

        The reflector may return either:
          A) Quality review:
             { "feedbacks": [...], "score": float, "attributes_to_retry": [...] }
          B) Patched attributes:
             { "attributes": {...}, "new_description": "..." } (no score)

        If 'score' is absent, estimate from completeness.
        """
        entity_type_raw = state["entity_type"]
        entity_type = self._resolve_type(entity_type_raw)
        description = self._resolve_description(entity_type_raw)
        properties = self._resolve_properties(entity_type_raw)
        attribute_definitions = state.get("attribute_definitions") or format_property_definitions(properties)
        expected_keys = self._parse_attribute_keys(attribute_definitions)

        # Prepare attributes string for prompt
        attrs_for_prompt = state.get("attributes", {})
        attrs_json_for_prompt = self._json_dumps(self._ensure_dict(attrs_for_prompt))

        result = self.extractor.reflect_entity_attributes(
            entity_type=entity_type,
            description=description,
            attribute_definitions=attribute_definitions,
            attributes=attrs_json_for_prompt,
            system_prompt=self.system_prompt,
            enable_thinking=False,
        )

        result = json.loads(correct_json_format(result))

        # Attempts
        attempt = int(state.get("attempt", 0)) + 1

        # Merge feedbacks
        feedbacks_old = state.get("feedbacks", []) or []
        feedbacks_new = result.get("feedbacks", []) or []
        merged_feedbacks = feedbacks_old + feedbacks_new

        # If reflector returned patched attributes/description, apply them
        attrs_fixed = self._ensure_dict(result.get("attributes", {}))
        new_desc_fixed = result.get("new_description", None)

        current_attrs = attrs_fixed if attrs_fixed else self._ensure_dict(state.get("attributes", {}))
        current_desc = new_desc_fixed if isinstance(new_desc_fixed, str) else state.get("new_description", "")

        # Score: prefer reflector score; otherwise estimate from completeness
        score = result.get("score", None)
        if score is None:
            score = self._estimate_score_from_completeness(current_attrs, expected_keys)

        # Determine which fields to retry
        retry_fields = result.get("attributes_to_retry", None)
        if retry_fields is None:
            retry_fields = []
            for k in expected_keys or current_attrs.keys():
                v = current_attrs.get(k, "")
                if (isinstance(v, str) and not v.strip()) or (v is None):
                    retry_fields.append(k)

        return {
            **state,
            "attempt": attempt,
            "score": float(score),
            "feedbacks": merged_feedbacks,
            "attributes_to_retry": retry_fields,
            # Updated view
            "attributes": current_attrs,
            "new_description": current_desc,
            # Also store the current view as 'previous_result' for the next round
            "previous_result": self._json_dumps({
                "new_description": current_desc,
                "attributes": current_attrs
            })
        }

    # ---------------- Branching logic ----------------
    def _check_reflection(self, state: Dict[str, Any]) -> str:
        """
        Decide whether to stop or retry based on score and attempts.
        """
        attempt = int(state.get("attempt", 0))
        score = float(state.get("score", 0.0))
        if score >= self.min_score:
            return "complete"
        if attempt >= self.max_retries:
            return "complete"
        return "retry"

    # ---------------- Graph construction ----------------
    def _build_graph(self):
        """
        Build and compile the LangGraph state machine.
        """
        builder = StateGraph(dict)
        builder.add_node("get_related_context", self.get_related_context)
        builder.add_node("extract_native", self.extract_native)
        builder.add_node("extract_incremental", self.extract_incremental)
        builder.add_node("reflect", self.reflect)

        builder.set_entry_point("get_related_context")
        builder.add_edge("get_related_context", "extract_native")
        builder.add_edge("extract_native", "reflect")
        builder.add_edge("extract_incremental", "reflect")
        builder.add_conditional_edges("reflect", self._check_reflection, {
            "complete": END,
            "retry": "extract_incremental"
        })
        return builder.compile()


    # # ---------------- Public sync interface (unchanged) ----------------
    def run(
        self,
        text: str,
        entity_name: str,
        entity_type: str,
        version: str,
        source_chunks: list = [],
        additional_chunks: list = [],
    ):
        """
        Synchronous entry. The internal flow may still run multiple reflection rounds.
        """
        return self.graph.invoke({
            "content": text,                      # Consolidated context (may be replaced in get_related_context)
            "entity_name": entity_name,
            "entity_type": entity_type,
            "versib": version,
            "source_chunks": source_chunks or [],
            "additional_chunks": additional_chunks or [],
            "previous_result": "",
            "feedbacks": [],
            "attempt": 0,
            "score": 0.0
        })

    # ---------------- Public async interface (unchanged) ----------------
    async def arun(
        self,
       text: str,
        entity_name: str,
        entity_type: str,
        version: str,
        source_chunks: list = [],
        additional_chunks: list = [],
        timeout: float = 600.0,
        max_attempts: int = 3,
        backoff_seconds: float = 120.0,
    ):
        """
        Async entry with overall timeout & light retry.
        The extract/reflect retries are governed by the graph flow and score.
        """
        payload = {
            "content": text,                      # Consolidated context (may be replaced in get_related_context)
            "entity_name": entity_name,
            "entity_type": entity_type,
            "version": version,
            "source_chunks": source_chunks or [],
            "additional_chunks": additional_chunks or [],
            "previous_result": "",
            "feedbacks": [],
            "attempt": 0,
            "score": 0.0
        }

        try:
            coro = self.graph.ainvoke(payload)
            result = await asyncio.wait_for(coro, timeout=timeout)
            return result.get("best_result", result)
        except asyncio.TimeoutError:
            # Only retry the overall call a few times. The internal loop handles its own retries.
            for i in range(1, max_attempts):
                try:
                    await asyncio.sleep(backoff_seconds * i)
                    result = await asyncio.wait_for(self.graph.ainvoke(payload), timeout=timeout)
                    return result.get("best_result", result)
                except asyncio.TimeoutError:
                    continue
            logger.error("AttributeExtractionAgent.arun timed out after %d attempts", max_attempts)
            return {"attributes": {}, "error": f"timeout after {max_attempts} attempts"}
        except Exception as e:
            logger.exception("AttributeExtractionAgent.arun failed: %s", e)
            return {"attributes": {}, "error": str(e)}

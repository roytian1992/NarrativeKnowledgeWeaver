import json
from enum import Enum
from typing import Dict
from langgraph.graph import StateGraph, END
from kag.utils.format import correct_json_format
from ..storage.vector_store import VectorStore
from kag.builder.extractor import InformationExtractor


def format_property_definitions(properties: Dict[str, str]) -> str:
    return "\n".join([f"- **{key}**：{desc}" for key, desc in properties.items()])


class AttributeExtractionAgent:
    def __init__(self, config, llm):
        self.config = config
        self.extractor = InformationExtractor(config, llm)
        self.vector_store = VectorStore(config)
        self.load_schema("kag/schema/graph_schema.json")
        self.load_abbreviations("kag/schema/settings_schema.json")
        self.graph = self._build_graph()

    def load_abbreviations(self, path):
        with open(path, "r", encoding="utf-8") as f:
            abbr = json.load(f)
        self.abbreviation_info = "\n".join(
            f"- **{a['abbr']}**: {a['full']}（{a['zh']}） - {a['description']}"
            for a in abbr.get("abbreviations", [])
        )

    def load_schema(self, path):
        with open(path, "r", encoding="utf-8") as f:
            schema = json.load(f)
        self.entity_types = schema.get("entities")
        self.type2description = {e["type"]: e["description"] for e in self.entity_types}
        self.type2property = {e["type"]: e["properties"] for e in self.entity_types}
        
    def extract(self, state):
        entity_name = state["entity_name"]
        entity_type = state["entity_type"]
        feedbacks = state.get("feedbacks", [])
        feedbacks = "\n".join(feedbacks)
        type_description = self.type2description.get(entity_type, "")
        properties = self.type2property.get(entity_type, {})
        attribute_definitions = format_property_definitions(properties)

        result = self.extractor.extract_entity_attributes(
            text=state["content"],
            entity_name=entity_name,
            description=type_description,
            entity_type=entity_type,
            attribute_definitions=attribute_definitions,
            abbreviations=self.abbreviation_info,
            previous_results=state.get("previous_result", ""),
            feedbacks=feedbacks,
            original_text=state.get("original_text", "")
        )

        result = json.loads(correct_json_format(result))
        # print("[CHECK] result: ", result)
        # print("[attributes ]", result["attributes"])
        return {
            **state,
            "attributes": json.dumps(result.get("attributes", {}), ensure_ascii=False),
            "new_description": result.get("new_description", ""),
            "previous_result": json.dumps(result, ensure_ascii=False),
        }

    def reflect(self, state):
        entity_name = state["entity_name"]
        entity_type = state["entity_type"]
        description = self.type2description.get(entity_type, "")
        attribute_definitions = format_property_definitions(
            self.type2property.get(entity_type, {})
        )
        attributes = state.get("attributes", {})
        # print("[CHECK]", attributes, type(attributes))

        result = self.extractor.reflect_entity_attributes(
            entity_name=entity_name,
            entity_type=entity_type,
            description=description,
            attribute_definitions=attribute_definitions,
            attributes=attributes
        )
        result = json.loads(correct_json_format(result))
        return {
            **state,
            "feedbacks": result.get("feedbacks", []),
            "need_additional_context": result.get("need_additional_context", False),
            "attributes_to_retry": result.get("attributes_to_retry", [])
        }

    def get_additional_context(self, state):
        entity_name = state["entity_name"]
        attributes_to_retry = state["attributes_to_retry"]
        entity_type = state["entity_type"]
        source_chunks = state["source_chunks"]
        if entity_type in ["Event", "Action", "Emotion", "Goal"]:
            documents = self.vector_store.search_by_ids(source_chunks)
            # if not documents:
            #     documents = self.vector_store.search(query=entity_name, limit=1)
            # print("[CHECK] entity_name: ", entity_name)
            # print("[CHECK] source_chunks: ", source_chunks)
            # print("[CHECK] source chunk result: ", documents)
        else:
            documents_by_id = self.vector_store.search_by_ids(source_chunks)
            documents_by_vec = self.vector_store.search(query=entity_name, limit=5)
            documents = documents_by_id + documents_by_vec
        results = {doc.content for doc in documents}
        new_text = "\n".join(list(results))

        return {
            **state,
            "content": new_text,
            "has_added_context": True
        }

    def _check_reflection(self, state):
        if state.get("need_additional_context", False) and not state.get("has_added_context", False):
            return "need_more"
        else:
            return "complete"

    def _build_graph(self):
        builder = StateGraph(dict)
        builder.add_node("extract", self.extract)
        builder.add_node("reflect", self.reflect)
        builder.add_node("get_additional_context", self.get_additional_context)

        builder.set_entry_point("extract")
        builder.add_edge("extract", "reflect")
        builder.add_conditional_edges("reflect", self._check_reflection, {
            "complete": END,
            "need_more": "get_additional_context"
        })
        builder.add_edge("get_additional_context", "extract")

        return builder.compile()

    def run(self, text: str, entity_name: str, entity_type: str, source_chunks: list = [], original_text: str = None):
        return self.graph.invoke({
            "content": text,
            "entity_name": entity_name,
            "entity_type": entity_type,
            "source_chunks": source_chunks,
            "original_text": original_text or text,
            "previous_result": "",
            "feedbacks": [],
            "has_added_context": False
        })

    async def arun(self, text: str, entity_name: str, entity_type: str, source_chunks: list = [], original_text: str = None):
        return await self.graph.ainvoke({
            "content": text,
            "entity_name": entity_name,
            "entity_type": entity_type,
            "source_chunks": source_chunks,
            "original_text": original_text or text,
            "previous_result": "",
            "feedbacks": [],
            "has_added_context": False
        })

import json
from enum import Enum
from langgraph.graph import StateGraph, END
from kag.utils.format import correct_json_format
from kag.builder.reflection import DynamicReflector
from kag.builder.extractor import InformationExtractor

class InformationExtractionAgent:
    def __init__(self, config, llm):
        self.config = config
        self.extractor = InformationExtractor(config, llm)
        self.load_schema("kag/schema/graph_schema.json")
        self.load_abbreviations("kag/schema/settings_schema.json")
        self.graph = self._build_graph()
        self.reflector = DynamicReflector(config)
        self.score_threshold = self.config.extraction.score_threshold
        self.max_retries = self.config.extraction.max_retries
        
        # print("[CHECK] score threshold: ", self.score_threshold)
        # print("[CHECK] max retries: ", self.max_retries)

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
        self.relation_type_groups = schema.get("relations")

        self.entity_type_description_text = "\n".join(
            f"- {e['type']}: {e['description']}" for e in self.entity_types
        )
        self.relation_type_description_text = "\n".join(
            f"- {r['type']}: {r['description']}"
            for group in self.relation_type_groups.values()
            for r in group
        )

        self.EntityType = Enum("EntityType", {e["type"]: e["type"] for e in self.entity_types})
        flat_relations = [r for g in self.relation_type_groups.values() for r in g]
        self.RelationType = Enum("RelationType", {r["type"]: r["type"] for r in flat_relations})


    def search_relevant_experience(self, state):
        content = state["content"].strip()

        issues, suggestions = self.reflector._search_relevant_reflections(content, k=5)
        reflection_results = {"issues": issues, "suggestions": suggestions}

        return {
            "reflection_results": reflection_results,
            "content": content,
            "retry_count": state.get("retry_count", 0),
            "best_score": state.get("best_score", 0),
            "best_result": state.get("best_result", {})
        }

    def extract_entities(self, state):
        content = state["content"].strip()
        retry_count = state.get("retry_count", 0)
        reflection_results = state.get("reflection_results", {})
        
        result = self.extractor.extract_entities(
            text=state["content"],
            entity_type_description_text=self.entity_type_description_text,
            abbreviations=self.abbreviation_info,
            reflection_results=reflection_results 
        )
        reflection_results["previous_entities"] = result
        
        result = json.loads(correct_json_format(result))
        entities = result.get("entities", [])
        entity_list_str = "、".join([f"{e.get('name', '?')}({e.get('type', '?')})" for e in entities]) if entities else "无"
        
        return {
            "entities": result.get("entities", []),
            "reflection_results": reflection_results,
            "entity_list": entity_list_str,
            "content": content,
            "retry_count": state.get("retry_count", 0),
            "best_score": state.get("best_score", 0),
            "best_result": state.get("best_result", {})
        }

    def extract_relations(self, state):
        reflection_results = state.get("reflection_results", {})

        result = self.extractor.extract_relations(
            text=state["content"],
            entity_list=state["entity_list"],
            relation_type_description_text=self.relation_type_description_text,
            abbreviations=self.abbreviation_info,
            reflection_results=reflection_results,
        )
        reflection_results["previous_relations"] = result
        
        result = json.loads(correct_json_format(result))
        
        return {
            "entities": state["entities"],
            "relations": result.get("relations", []),
            "reflection_results": reflection_results,
            "entity_list": state["entity_list"],
            "content": state["content"],
            "retry_count": state["retry_count"],
            "best_score": state["best_score"],
            "best_result": state["best_result"]
        }

    def reflect(self, state):
        reflection_results = state.get("reflection_results", {})
        logs = "\n".join(self.reflector.generate_logs(state))
        # print("抽取日志: ", logs)
        if len(state["content"]) <= 100:
            version = "short"
        else:
            version = "default"
        result = self.extractor.reflect_extractions(
            logs=logs,
            entity_type_description_text=self.entity_type_description_text,
            relation_type_description_text=self.relation_type_description_text,
            original_text=state["content"],
            previous_reflection=reflection_results,
            abbreviations=self.abbreviation_info,
            version=version
        )
        result = json.loads(correct_json_format(result))
        score = result["score"]
        best_score = state.get("best_score", 0)
        
        reflection_results["score"] = score
        reflection_results["suggestions"] = result.get("suggestions", [])
        reflection_results["issues"] = result.get("current_issues", [])
        
        # 如果本轮更优则更新 best_result
        if score > best_score:
            best_result = {
                "entities": state.get("entities", []),
                "relations": state.get("relations", []),
                "score": score,
                "suggestions": result.get("suggestions", []),
                "issues": result.get("current_issues", [])
            }
        else:
            best_result = state.get("best_result", {})

        self.reflector._store_memory(state["content"], result)
        
        return {
            "score": int(score),
            "content": state["content"],
            "reflection_results": reflection_results,
            "retry_count": state["retry_count"] + 1,
            "best_score": max(score, best_score),
            "best_result": best_result
        }

    def _score_check(self, state):
        if state["score"] >= self.score_threshold:
            result = "good"
        elif state["retry_count"] >= self.max_retries:
            result = "giveup"
        else:
            result = "retry"
        # print("[CHECK] decision: ", result)
        return result

    def _build_graph(self):
        builder = StateGraph(dict)
        builder.add_node("search_relevant_experience", self.search_relevant_experience)
        builder.add_node("extract_entities", self.extract_entities)
        builder.add_node("extract_relations", self.extract_relations)
        builder.add_node("reflect", self.reflect)

        builder.set_entry_point("search_relevant_experience")
        builder.add_edge("search_relevant_experience", "extract_entities")
        builder.add_edge("extract_entities", "extract_relations")
        builder.add_edge("extract_relations", "reflect")
        builder.add_conditional_edges("reflect", self._score_check, {
            "good": END,
            "retry": "extract_entities",
            "giveup": END
        })

        return builder.compile()

    def run(self, text: str):
        result = self.graph.invoke({
            "content": text,
            "retry_count": 0,
            "best_score": 0,
            "best_result": {}
        })
        return result.get("best_result", result)

    async def arun(self, text: str):
        result = await self.graph.ainvoke({
            "content": text,
            "retry_count": 0,
            "best_score": 0,
            "best_result": {}
        })
        return result.get("best_result", result)

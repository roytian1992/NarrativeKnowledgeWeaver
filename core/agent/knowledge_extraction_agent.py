import json
from enum import Enum
from langgraph.graph import StateGraph, END
from core.utils.format import correct_json_format
from core.builder.manager.information_manager import InformationExtractor
import asyncio  
import os

def generate_suggestions(related_insights, related_history):
    suggestions = ""
    if related_insights:
        suggestions += "之前的阅读的过程中有以下这些发现：\n" + "\n".join(related_insights) + "\n"
        
    if related_history:
        suggestions += "之前的一些抽取示例：\n" + "\n".join(related_history)
        suggestions += "\n其中7分以上为优秀示例，5分以下可以视作反面示例。"
    return suggestions

class InformationExtractionAgent:
    def __init__(self, config, llm, system_prompt, schema, reflector, mode="regular"):
        self.config = config
        self.extractor = InformationExtractor(config, llm)
        self.load_schema(schema)
        self.graph = self._build_graph()
        self.reflector = reflector
        self.score_threshold = self.config.agent.score_threshold
        self.max_retries = self.config.agent.max_retries
        self.system_prompt = system_prompt
        self.mode = mode
        if  self.mode== "probing":
            self.max_retries = 0
        # print("[CHECK] score threshold: ", self.score_threshold)
        # print("[CHECK] max retries: ", self.max_retries)
        
    def load_schema(self, schema):
        entity_types = schema.get("entities")
        relation_type_groups = schema.get("relations")

        self.entity_type_description_text = "\n".join(
            f"- {e['type']}: {e['description']}" for e in entity_types
        )
        self.relation_type_description_text = "\n".join(
            f"- {r['type']}: {r['description']}"
            for group in relation_type_groups.values()
            for r in group
        )
        
    def search_relevant_experience(self, state):
        content = state["content"].strip()

        related_history, related_insights = self.reflector._search_relevant_reflections(content)
        
        suggestions = generate_suggestions(related_insights, related_history)

        reflection_results = {"suggestions": suggestions, 
                              "related_insights": related_insights,
                              "related_history": related_history,
                              "issues": []
                              }

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
            system_prompt=self.system_prompt,
            reflection_results=reflection_results,
            enable_thinking=False if self.mode == "probing" else True
        )
        reflection_results["previous_entities"] = result
        
        result = json.loads(correct_json_format(result))
        entities = result.get("entities", [])
        entity_list_str = "、".join([f"{e.get('name', '?')}({e.get('type', '?')})" for e in entities]) if entities else ""
        
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
            system_prompt=self.system_prompt,
            reflection_results=reflection_results,
            enable_thinking=False if self.mode == "probing" else True
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
        if self.mode == "probing":
            version = "probing"
        result = self.extractor.reflect_extractions(
            logs=logs,
            entity_type_description_text=self.entity_type_description_text,
            relation_type_description_text=self.relation_type_description_text,
            original_text=state["content"],
            previous_reflection=reflection_results,
            system_prompt=self.system_prompt,
            version=version
        )
        result = json.loads(correct_json_format(result))
        score = result["score"]
        best_score = state.get("best_score", 0)
        
        reflection_results["score"] = score
        reflection_results.setdefault("related_insights", []).extend(result.get("insights", []))

        reflection_results["suggestions"] = generate_suggestions(reflection_results.get("related_insights", []), reflection_results.get("related_history", []))
        reflection_results["issues"] = result.get("current_issues", [])
        
        current_result = {
            "entities": state.get("entities", []),
            "relations": state.get("relations", []),
            "score": score,
            "insights": result.get("insights", []),
            "issues": result.get("current_issues", [])
        }
        
        # 如果本轮更优则更新 best_result
        if score > best_score:
            best_result = current_result
        else:
            best_result = state.get("best_result", {})

        self.reflector._store_memory(state["content"], current_result)
        
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

    # async def arun(self, text: str):
    #     result = await self.graph.ainvoke({
    #         "content": text,
    #         "retry_count": 0,
    #         "best_score": 0,
    #         "best_result": {}
    #     })
    #     return result.get("best_result", result)
    
    async def arun(self, text: str,
                   timeout: int = 600,
                   max_attempts: int = 5,
                   backoff_seconds: int = 30):
        """
        异步抽取（带超时 + 重试）
        ・timeout        每次调用最多等待秒数
        ・max_attempts   总尝试次数 = 1 次正常 + (max_attempts-1) 次重试
        ・backoff_seconds 简单线性退避（30,60,…）
        """
        attempt = 0
        while attempt < max_attempts:
            try:
                coro = self.graph.ainvoke({
                    "content": text,
                    "retry_count": 0,
                    "best_score": 0,
                    "best_result": {}
                })
                result = await asyncio.wait_for(coro, timeout=timeout)
                return result.get("best_result", result)     # <<< 正常返回
            except asyncio.TimeoutError:
                attempt += 1
                if attempt >= max_attempts:
                    # 超时仍失败 —— 返回空抽取，供上层继续流程
                    print("[CHECK] text: ", text)
                    print({"entities": [], "relations": [], "score": 0, "insights": [], "issues": [],
                            "error": f"timeout after {max_attempts} attempts"})
                    return {"entities": [], "relations": [], "score": 0, "insights": [], "issues": [],
                            "error": f"timeout after {max_attempts} attempts"}
                # 退避后重试
                await asyncio.sleep(backoff_seconds * attempt)
            except Exception as e:
                # 其它异常不重试（你可按需改成也重试）
                print("[CHECK] text: ", text)
                print({"entities": [], "relations": [], "error": str(e), "score": 0, "insights": [], "issues": [],})
                return {"entities": [], "relations": [], "error": str(e), "score": 0, "insights": [], "issues": [],}

import json
from enum import Enum
from langgraph.graph import StateGraph, END
from core.utils.format import correct_json_format
from typing import List, Dict, Any, Tuple, Set
from core.builder.manager.information_manager import InformationExtractor
import asyncio  
import os
import logging

# 全局日志器
logger = logging.getLogger(__name__)



def generate_suggestions(related_insights, related_history):
    suggestions = ""
    if related_insights:
        suggestions += "During previous reading, the following insights were found:\n" + "\n".join(related_insights) + "\n"

    if related_history:
        suggestions += "Some previous extraction examples:\n" + "\n".join(related_history) + "\n"
        suggestions += "\nExamples with a score above 7 are considered good, while those below 5 can be treated as negative examples."
    return suggestions

class InformationExtractionAgent:
    def __init__(self, config, llm, system_prompt, schema, reflector, mode="regular", prompt_loader=None):
        self.config = config
        self.extractor = InformationExtractor(config, llm, prompt_loader=prompt_loader)
        self.load_schema(schema)
        self.graph = self._build_graph()
        self.reflector = reflector
        self.score_threshold = self.config.agent.score_threshold
        self.max_retries = self.config.agent.max_retries
        self.system_prompt = system_prompt
        self.mode = mode
        if self.mode == "probing":
            self.max_retries = 0
        # print("[CHECK] score threshold: ", self.score_threshold)
        # print("[CHECK] max retries: ", self.max_retries)
    
    def set_mode(self, mode):
        self.mode = mode
        if self.mode == "probing":
            self.max_retries = 0
        else:
            self.max_retries = self.config.agent.max_retries

    def load_schema(self, schema):
        entity_types = schema.get("entities")
        relation_type_groups = schema.get("relations")

        self.entity_type_description_text = "\n".join(
            f"- {e['type']}: {e['description']}" for e in entity_types
        )
        self.relation_type_description_text = {
            group_name: "\n".join(
                f"- {r['type']}: {r['description']}"
                for r in group_relations
            )
            for group_name, group_relations in relation_type_groups.items()
        }

        
    def search_relevant_experience(self, state):
        content = state["content"].strip()

        related_history, related_insights = self.reflector._search_relevant_reflections(content)
        # print("唐宏伟是个傻逼！")
        suggestions = generate_suggestions(related_insights, related_history)
        # print("[DEBUG] suggestions: ", suggestions)

        reflection_results = {
            "suggestions": suggestions, 
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
        previous_entity_results = state.get("previous_entity_results", {})
        
        result = self.extractor.extract_entities(
            text=state["content"],
            entity_type_description_text=self.entity_type_description_text,
            system_prompt=self.system_prompt,
            reflection_results=reflection_results,
            previous_results=previous_entity_results,
            enable_thinking=False if self.mode == "probing" or retry_count==0 else True
        )
        # reflection_results["previous_entities"] = result
        previous_entity_results = result
        result = json.loads(correct_json_format(result))
        entities = result.get("entities", [])
        entity_list_str = "、".join([f"{e.get('name', '?')}({e.get('type', '?')})" for e in entities]) if entities else ""
        
        return {
            "entities": result.get("entities", []),
            "reflection_results": reflection_results,
            "previous_entity_results": previous_entity_results,
            "entity_list": entity_list_str,
            "content": content,
            "retry_count": state.get("retry_count", 0),
            "best_score": state.get("best_score", 0),
            "best_result": state.get("best_result", {})
        }

    def extract_relations(self, state):
        reflection_results = state.get("reflection_results", {})
        previous_relation_results = state.get("previous_relation_results", {})
        retry_count = state.get("retry_count", 0)
        for group in self.relation_type_description_text:
            result = self.extractor.extract_relations(
                text=state["content"],
                entity_list=state["entity_list"],
                previous_results=previous_relation_results.get("group", {}),
                relation_type_description_text=self.relation_type_description_text[group],
                system_prompt=self.system_prompt,
                reflection_results=reflection_results,
                enable_thinking=False if self.mode == "probing" or retry_count==0 else True
            )
            result = json.loads(correct_json_format(result))
            previous_relation_results[group] = result.get("relations", [])

        
        return {
            "entities": state["entities"],
            "relations": result.get("relations", []),
            "previous_relation_results": previous_relation_results,
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
        # print("Extraction logs: ", logs)
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
            enable_thinking=False if self.mode == "probing" else True,
            version=version
        )
        result = json.loads(correct_json_format(result))
        score = result["score"]
        best_score = state.get("best_score", 0)
        
        reflection_results["score"] = score
        reflection_results.setdefault("related_insights", []).extend(result.get("insights", []))

        reflection_results["suggestions"] = generate_suggestions(
            reflection_results.get("related_insights", []), 
            reflection_results.get("related_history", [])
        )
        reflection_results["issues"] = result.get("current_issues", [])
        
        current_result = {
            "entities": state.get("entities", []),
            "relations": state.get("relations", []),
            "score": score,
            "insights": result.get("insights", []),
            "issues": result.get("current_issues", [])
        }
        
        # Update best_result if this round achieves a higher score
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
    
    # async def arun(self, text: str,
    #                timeout: int = 600,
    #                max_attempts: int = 5,
    #                backoff_seconds: int = 30):
    #     """
    #     Asynchronous extraction with timeout and retry
    #     ・timeout        Maximum wait time per attempt (in seconds)
    #     ・max_attempts   Total attempts = 1 normal + (max_attempts-1) retries
    #     ・backoff_seconds Linear backoff (30, 60, …)
    #     """
    #     attempt = 0
    #     while attempt < max_attempts:
    #         try:
    #             coro = self.graph.ainvoke({
    #                 "content": text,
    #                 "retry_count": 0,
    #                 "best_score": 0,
    #                 "best_result": {}
    #             })
    #             result = await asyncio.wait_for(coro, timeout=timeout)
    #             return result.get("best_result", result)     # <<< Normal return
    #         except asyncio.TimeoutError:
    #             attempt += 1
    #             if attempt >= max_attempts:
    #                 # Timeout failure — return empty extraction for upper-level handling
    #                 print("[CHECK] text: ", text)
    #                 print({"entities": [], "relations": [], "score": 0, "insights": [], "issues": [],
    #                        "error": f"timeout after {max_attempts} attempts"})
    #                 return {"entities": [], "relations": [], "score": 0, "insights": [], "issues": [],
    #                         "error": f"timeout after {max_attempts} attempts"}
    #             # Backoff before retry
    #             await asyncio.sleep(backoff_seconds * attempt)
    #         except Exception as e:
    #             # Other exceptions are not retried (you can change this to retry if needed)
    #             print("[CHECK] text: ", text)
    #             print({"entities": [], "relations": [], "error": str(e), "score": 0, "insights": [], "issues": [],})
    #             return {"entities": [], "relations": [], "error": str(e), "score": 0, "insights": [], "issues": [],}


    async def arun(self, text: str,
                   timeout: int = 600,
                   max_attempts: int = 5,
                   backoff_seconds: int = 30):
        """
        异步抽取入口（带多次重试 + 全局清理）
        - timeout: 每次执行最大等待时间
        - max_attempts: 最大尝试次数
        - backoff_seconds: 线性退避间隔
        - 自动 cancel LangGraph 内部挂起任务
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

                # 等待结果，超时则重试
                result = await asyncio.wait_for(coro, timeout=timeout)
                await asyncio.sleep(0)  # small yield for event loop flush
                return result.get("best_result", result)

            except asyncio.TimeoutError:
                attempt += 1
                logger.warning(f"[TIMEOUT] arun() timeout ({timeout}s) attempt {attempt}/{max_attempts}")

                # 1️⃣ cancel 所有未完成任务（包括 LangGraph 内部节点）
                for task in list(asyncio.all_tasks()):
                    if task is not asyncio.current_task() and not task.done():
                        task.cancel()

                # 2️⃣ 等待清理
                try:
                    await asyncio.sleep(0)
                    await asyncio.wait_for(asyncio.gather(*asyncio.all_tasks(), return_exceptions=True), timeout=5)
                except Exception:
                    pass

                if attempt >= max_attempts:
                    logger.warning(f"[TIMEOUT-FAIL] Giving up after {max_attempts} attempts")
                    return {
                        "entities": [],
                        "relations": [],
                        "score": 0,
                        "insights": [],
                        "issues": [],
                        "error": f"timeout after {max_attempts} attempts"
                    }

                # 3️⃣ 线性退避
                await asyncio.sleep(backoff_seconds * attempt)

            except Exception as e:
                # 其他错误：立即返回空结果
                logger.error(f"[ERROR] arun() failed: {repr(e)}")
                try:
                    for task in list(asyncio.all_tasks()):
                        if task is not asyncio.current_task() and not task.done():
                            task.cancel()
                except Exception:
                    pass
                await asyncio.sleep(0)
                return {
                    "entities": [],
                    "relations": [],
                    "score": 0,
                    "issues": [],
                    "insights": [],
                    "error": str(e)
                }

        # fallback（理论不会到）
        return {"entities": [], "relations": [], "score": 0, "issues": [], "insights": [], "error": "unexpected exit"}

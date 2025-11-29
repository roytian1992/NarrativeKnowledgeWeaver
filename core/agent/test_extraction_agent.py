import json
from langgraph.graph import StateGraph, END
from core.utils.format import correct_json_format
from core.builder.manager.information_manager import InformationExtractor
import asyncio

def _empty_reflection():
    # 你只要空结构，不要 related_history/related_insights/suggestions
    return {"issues": [], "score": 0}

class InformationExtractionAgent:
    def __init__(self,
                 config,
                 llm,
                 system_prompt,
                 reflector,
                 mode: str = "regular",
                 prompt_loader=None,
                 entity_type_description_text: str = "",
                 relation_type_description_text: str = ""):
        # 先初始化全部成员
        self.config = config
        self.extractor = InformationExtractor(config, llm, prompt_loader=prompt_loader)
        self.reflector = reflector
        self.system_prompt = system_prompt
        self.mode = mode
        self.entity_type_description_text = entity_type_description_text or ""
        self.relation_type_description_text = relation_type_description_text or ""
        self.score_threshold = self.config.agent.score_threshold
        self.max_retries = self.config.agent.max_retries
        if self.mode == "probing":
            self.max_retries = 0

        # 最后再构图（非常关键）
        self.graph = self._build_graph()

    def set_mode(self, mode: str):
        self.mode = mode
        self.max_retries = 0 if mode == "probing" else self.config.agent.max_retries

    # ---------------------------
    # Step 1: 实体抽取
    # ---------------------------
    def extract_entities(self, state: dict):
        # print("[NODE->extract_entities] entering...")
        content = (state.get("content") or "").strip()
        reflection_results = state.get("reflection_results") or _empty_reflection()

        # 允许每次调用覆盖类型描述（若未传则用实例默认值）
        entity_type_desc = state.get("entity_type_description_text", self.entity_type_description_text)

        raw = self.extractor.extract_entities(
            text=content,
            entity_type_description_text=entity_type_desc,
            system_prompt=self.system_prompt,
            reflection_results=reflection_results,
            enable_thinking=(False if self.mode == "probing" else True)
        )
        reflection_results["previous_entities"] = raw
        # print("[DEBUG] entity result before correct: ", raw)

        parsed = json.loads(correct_json_format(raw))
        # print("[DEBUG] entity result after correct: ", parsed)
        entities = parsed.get("entities", []) or []

        entity_list_str = "、".join([f"{e.get('name', '?')}({e.get('type', '?')})" for e in entities]) if entities else ""
        # print("[DEBUG] entity_list_str: ", entity_list_str)

        out = {
            "entities": entities,
            "entity_list": entity_list_str,
            "content": content,
            "reflection_results": reflection_results,
            "retry_count": state.get("retry_count", 0),
            "best_score": state.get("best_score", 0),
            "best_result": state.get("best_result", {}),
            # 继续传递覆盖文本
            "entity_type_description_text": entity_type_desc,
            "relation_type_description_text": state.get("relation_type_description_text", self.relation_type_description_text),
        }
        # print("[DEBUG] out: ", out)
        # print("[NODE->extract_entities] exit.")
        return out

    # ---------------------------
    # Step 2: 关系抽取
    # ---------------------------
    def extract_relations(self, state: dict):
        # print("[NODE->extract_relations] entering...")
        reflection_results = state.get("reflection_results") or {}
        relation_type_desc = state.get("relation_type_description_text", self.relation_type_description_text)

        raw = self.extractor.extract_relations(
            text=state.get("content", ""),
            entity_list=state.get("entity_list", ""),
            relation_type_description_text=relation_type_desc,
            system_prompt=self.system_prompt,
            reflection_results=reflection_results,
            enable_thinking=(False if self.mode == "probing" else True)
        )
        # print("[DEBUG] relation result: ", raw)
        reflection_results["previous_relations"] = raw

        parsed = json.loads(correct_json_format(raw))
        relations = parsed.get("relations", []) or []

        out = {
            "entities": state.get("entities", []),
            "relations": relations,
            "entity_list": state.get("entity_list", ""),
            "content": state.get("content", ""),
            "reflection_results": reflection_results or {},
            "retry_count": state.get("retry_count", 0),
            "best_score": state.get("best_score", 0),
            "best_result": state.get("best_result", {}),
            # 继续传递覆盖文本
            "entity_type_description_text": state.get("entity_type_description_text", self.entity_type_description_text),
            "relation_type_description_text": relation_type_desc,
        }
        # print("[NODE->extract_relations] exit.")

        return out

    # ---------------------------
    # Step 3: 反思评分
    # ---------------------------
    def reflect(self, state: dict):
        # print("[NODE->reflect] entering...")
        reflection_results = state.get("reflection_results") or {}
        logs = "\n".join(self.reflector.generate_logs(state))

        if len(state.get("content", "")) <= 100:
            version = "short"
        else:
            version = "default"
        if self.mode == "probing":
            version = "probing"

        entity_type_desc = state.get("entity_type_description_text", self.entity_type_description_text)
        relation_type_desc = state.get("relation_type_description_text", self.relation_type_description_text)

        raw = self.extractor.reflect_extractions(
            logs=logs,
            entity_type_description_text=entity_type_desc,
            relation_type_description_text=relation_type_desc,
            original_text=state.get("content", ""),
            previous_reflection=reflection_results,
            system_prompt=self.system_prompt,
            enable_thinking=(False if self.mode == "probing" else True),
            version=version
        )
        parsed = json.loads(correct_json_format(raw))
        score = int(parsed.get("score", 0))

        best_score = int(state.get("best_score", 0))
        reflection_results["score"] = score
        # 只保留 issues；不保留 history/insights/suggestions
        reflection_results["issues"] = parsed.get("current_issues", []) or []

        current_result = {
            "entities": state.get("entities", []),
            "relations": state.get("relations", []),
            "score": score,
            "insights": [],  # 你不需要
            "issues": reflection_results["issues"],
        }
        best_result = current_result if score > best_score else state.get("best_result", {})

        # # 记忆可保留；如不需要可去掉
        # try:
        #     self.reflector._store_memory(state.get("content", ""), current_result)
        # except Exception as e:
        #     reflection_results.setdefault("issues", []).append(f"[MEMORY-WARN] {e}")

        out = {
            "score": score,
            "content": state.get("content", ""),
            "reflection_results": reflection_results,
            "retry_count": int(state.get("retry_count", 0)) + 1,
            "best_score": max(score, best_score),
            "best_result": best_result,
            "entity_type_description_text": entity_type_desc,
            "relation_type_description_text": relation_type_desc,
        }
        # print("[NODE->reflect] exit.")
        return out

    # ---------------------------
    # Routing
    # ---------------------------
    def _score_check(self, state: dict):
        if state["score"] >= self.score_threshold:
            return "good"
        elif state["retry_count"] >= self.max_retries:
            return "giveup"
        else:
            return "retry"

    def _build_graph(self):
        builder = StateGraph(dict)
        builder.add_node("extract_entities", self.extract_entities)
        builder.add_node("extract_relations", self.extract_relations)
        builder.add_node("reflect", self.reflect)

        builder.set_entry_point("extract_entities")
        builder.add_edge("extract_entities", "extract_relations")
        builder.add_edge("extract_relations", "reflect")
        builder.add_conditional_edges("reflect", self._score_check, {
            "good": END,
            "retry": "extract_entities",
            "giveup": END
        })
        return builder.compile()

    # ---------------------------
    # Public APIs
    # ---------------------------
    def run(self, text: str,
            *,
            relation_type_description_text: str | None = None,
            entity_type_description_text: str | None = None,
            reflection_results: dict | None = None):
        """同步调用，允许覆盖描述"""
        result = self.graph.invoke({
            "content": text,
            "retry_count": 0,
            "best_score": 0,
            "best_result": {},
            "relation_type_description_text": relation_type_description_text or self.relation_type_description_text,
            "entity_type_description_text": entity_type_description_text or self.entity_type_description_text,
            "reflection_results": reflection_results or _empty_reflection(),
        })
        return result.get("best_result", result)

    async def arun(self, text: str,
                   *,
                   relation_type_description_text: str | None = None,
                   entity_type_description_text: str | None = None,
                   reflection_results: dict | None = None,
                   timeout: int = 600,
                   max_attempts: int = 5,
                   backoff_seconds: int = 30):
        """异步调用，允许覆盖描述"""
        attempt = 0
        while attempt < max_attempts:
            try:
                coro = self.graph.ainvoke({
                    "content": text,
                    "retry_count": 0,
                    "best_score": 0,
                    "best_result": {},
                    "relation_type_description_text": relation_type_description_text or self.relation_type_description_text,
                    "entity_type_description_text": entity_type_description_text or self.entity_type_description_text,
                    "reflection_results": reflection_results or _empty_reflection(),
                })
                result = await asyncio.wait_for(coro, timeout=timeout)
                return result.get("best_result", result)
            except asyncio.TimeoutError:
                attempt += 1
                if attempt >= max_attempts:
                    return {"entities": [], "relations": [], "score": 0,
                            "issues": [], "error": f"timeout after {max_attempts} attempts"}
                await asyncio.sleep(backoff_seconds * attempt)
            except Exception as e:
                return {"entities": [], "relations": [], "score": 0,
                        "issues": [], "error": str(e)}
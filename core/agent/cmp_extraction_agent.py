# core/builder/cmp_extraction_agent.py

import json
import asyncio
from langgraph.graph import StateGraph, END
from core.utils.format import correct_json_format
from core.builder.manager.information_manager import InformationExtractor

class CMPExtractionAgent:
    """
    服化道抽取代理（Wardrobe / Styling / PropItem）
    - 抽取节点内完成三路抽取与合并，每条记录添加 type: wardrobe|styling|propitem
    - 反思阶段的 logs 为“列表字符串”而非 JSON
    - 最终 best_result 仅返回合并后的 results、feedbacks、score
    """

    def __init__(self, config, llm, system_prompt, prompt_loader=None):
        self.config = config
        self.extractor = InformationExtractor(config, llm, prompt_loader=prompt_loader)
        self.system_prompt = system_prompt
        self.score_threshold = self.config.agent.score_threshold
        self.max_retries = self.config.agent.max_retries
        self.graph = self._build_graph()

    # ---------------- LangGraph 节点 ----------------

    def extract_cmp(self, state: dict):
        content = state["content"].strip()
        feedbacks = state.get("feedbacks", [])

        # 三路抽取（可把各自历史喂回各自 extractor，如需的话）
        w_raw = self.extractor.extract_wardrobe(
            content=content,
            system_prompt=self.system_prompt,
            reflection_results={"previous_results": state.get("wardrobe_results", []), "feedbacks":  feedbacks}
        )
        s_raw = self.extractor.extract_styling(
            content=content,
            system_prompt=self.system_prompt,
            reflection_results={"previous_results": state.get("styling_results", []), "feedbacks": feedbacks}
        )
        p_raw = self.extractor.extract_propitem(
            content=content,
            system_prompt=self.system_prompt,
            reflection_results={"previous_results": state.get("propitem_results", []), "feedbacks": feedbacks}
        )

        # 解析 + 取 results 数组
        try:
            w_obj = json.loads(correct_json_format(w_raw))
            wardrobe_results = w_obj["results"] if isinstance(w_obj, dict) and "results" in w_obj else []
        except Exception:
            wardrobe_results = []
        try:
            s_obj = json.loads(correct_json_format(s_raw))
            styling_results = s_obj["results"] if isinstance(s_obj, dict) and "results" in s_obj else []
        except Exception:
            styling_results = []
        try:
            p_obj = json.loads(correct_json_format(p_raw))
            propitem_results = p_obj["results"] if isinstance(p_obj, dict) and "results" in p_obj else []
        except Exception:
            propitem_results = []

        # 在这里直接合并，并给每条打上 type 字段（不单独建节点）
        merged = []
        for item in wardrobe_results:
            rec = dict(item)
            # rec["type"] = "wardrobe"
            merged.append(rec)
        for item in styling_results:
            rec = dict(item)
            # rec["type"] = "styling"
            merged.append(rec)
        for item in propitem_results:
            rec = dict(item)
            # rec["type"] = "propitem"
            merged.append(rec)

        # 分类型结果仍保留在 state 里，便于下轮同类型参考；最终返回只走 best_result
        return {
            "content": content,
            "results": merged,
            "wardrobe_results": wardrobe_results,
            "styling_results": styling_results,
            "propitem_results": propitem_results,
            "feedbacks": state.get("feedbacks", []),
            "score": state.get("score", 0),
            "retry_count": state.get("retry_count", 0),
            "best_score": state.get("best_score", 0),
            "best_result": state.get("best_result", {"results": [], "feedbacks": [], "score": 0})
        }

    def reflect_cmp(self, state: dict):
        content = state["content"]
        results = state.get("results", [])
        reflection_results = state.get("reflection_results", {})

        # === 这里：logs 用“列表字符串”，非 JSON ===
        # 形如：
        # - propitem | name=窗帘 | subcategory=curtain | appearance= | status=拉开 | character=刘培强 | evidence="拉开窗帘"
        lines = []
        for r in results:
            line = (
                f"- 名称={r.get('name','')} | "
                f"类别={r.get('category','')} | "
                f"子类={r.get('subcategory','')} | "
                f"外观={r.get('appearance','')} | "
                f"状态={r.get('status','')} | "
                f"角色={r.get('character','')} | "
                f"补充信息={r.get('notes','')} | "
                f"证据={json.dumps(r.get('evidence',''), ensure_ascii=False)}"
            )
            lines.append(line)
        logs = "\n".join(lines)


        raw = self.extractor.reflect_cmp_extractions(
            logs=logs,
            content=content,
            system_prompt=self.system_prompt,
            previous_reflection=reflection_results
        )

        # 解析反思
        try:
            ref = json.loads(correct_json_format(raw))
        except Exception:
            ref = {"feedbacks": [], "score": 0}

        try:
            score = int(float(ref.get("score", 0)))
        except Exception:
            score = 0

        feedbacks = ref.get("feedbacks", [])

        # 记录最佳（只存合并结果）
        best_score = state.get("best_score", 0)
        if score > best_score:
            best_result = {
                "results": results,      # 合并后的最终结果（每条含 type）
                "feedbacks": feedbacks,
                "score": score
            }
            best_score = score
        else:
            best_result = state.get("best_result", {"results": [], "feedbacks": [], "score": 0})

        return {
            "content": content,
            "results": results,
            "wardrobe_results": state.get("wardrobe_results", []),
            "styling_results": state.get("styling_results", []),
            "propitem_results": state.get("propitem_results", []),
            "feedbacks": feedbacks,
            "score": score,
            "retry_count": state.get("retry_count", 0) + 1,
            "best_score": best_score,
            "best_result": best_result,
            "reflection_results": {"feedbacks": feedbacks, "score": score}
        }

    def _score_check(self, state: dict):
        if state["score"] >= self.score_threshold:
            return "good"
        elif state["retry_count"] >= self.max_retries:
            return "giveup"
        else:
            return "retry"

    def _build_graph(self):
        builder = StateGraph(dict)
        builder.add_node("extract_cmp", self.extract_cmp)
        builder.add_node("reflect_cmp", self.reflect_cmp)

        builder.set_entry_point("extract_cmp")
        builder.add_edge("extract_cmp", "reflect_cmp")
        builder.add_conditional_edges("reflect_cmp", self._score_check, {
            "good": END,
            "retry": "extract_cmp",
            "giveup": END
        })
        return builder.compile()

    # ---------------- 对外 API ----------------

    def run(self, text: str):
        result = self.graph.invoke({
            "content": text,
            "retry_count": 0,
            "best_score": 0,
            "best_result": {"results": [], "feedbacks": [], "score": 0},
            "reflection_results": {}
        })
        # 最终只返回 best_result（已是合并后的 results）
        return result.get("best_result", {"results": [], "feedbacks": [], "score": 0})


    async def arun(
        self,
        text: str,
        timeout: int = 120,
        max_attempts: int = 3,
        backoff_seconds: int = 30,
    ):
        """
        异步接口：保留超时与退避；抽取-反思的重试由图与 score 控制
        """
        payload = {
            "content": text,
            "retry_count": 0,
            "best_score": 0,
            "best_result": {"results": [], "feedbacks": [], "score": 0},
            "reflection_results": {}
        }

        try:
            coro = self.graph.ainvoke(payload)
            result = await asyncio.wait_for(coro, timeout=timeout)
            return result.get("best_result", result)
        except asyncio.TimeoutError:
            # 仅针对“整体调用卡顿”做少量重试；流程内部重试不受影响
            for i in range(1, max_attempts):
                try:
                    await asyncio.sleep(backoff_seconds * i)
                    result = await asyncio.wait_for(self.graph.ainvoke(payload), timeout=timeout)
                    return result.get("best_result", result)
                except asyncio.TimeoutError:
                    continue
            return {"results": [], "feedbacks": [], "score": 0, "error": "timeout"}
        except Exception as e:
            return {"results": [], "feedbacks": [], "score": 0, "error": "timeout"}



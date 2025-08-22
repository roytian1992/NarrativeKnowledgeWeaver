import json
import re
import asyncio
from typing import Dict, Any, List
from langgraph.graph import StateGraph, END
from core.utils.format import correct_json_format
from ..storage.vector_store import VectorStore
from core.memory.vector_memory import VectorMemory
from core.builder.manager.information_manager import InformationExtractor
from core.model_providers.openai_rerank import OpenAIRerankModel
from retriever.vectordb_retriever import ParentChildRetriever


def format_property_definitions(properties: Dict[str, str]) -> str:
    return "\n".join([f"- **{key}**：{desc}" for key, desc in properties.items()])


class AttributeExtractionAgent:
    """
    - 首次节点：get_related_context（只运行一次，合并初始上下文）
    - 主循环：extract -> reflect -> (score 达标/达到最大重试即结束；否则回到 extract)
    - 评分策略：优先用反思器返回的 score；若无，则按“非空比例”估算到 0-10
    """

    def __init__(self, config, llm, system_prompt, schema, enable_thinking=True, prompt_loader=None, global_entity_types=None):
        self.config = config
        self.extractor = InformationExtractor(config, llm, prompt_loader=prompt_loader)
        self.history_memory = VectorMemory(config, "history_memory")
        self.load_schema(schema)
        self.system_prompt = system_prompt
        if global_entity_types is None:
            self.global_entity_types = ["Character", "Concept", "Object", "Location"]
        else:
            self.global_entity_types = global_entity_types

        # 检索资源
        self.reranker = OpenAIRerankModel(config)
        self.document_vector_store = VectorStore(config, "documents")
        self.sentence_vector_store = VectorStore(config, "sentences")
        self.retriever = ParentChildRetriever(
            doc_vs=self.document_vector_store,
            sent_vs=self.sentence_vector_store,
            reranker=self.reranker
        )
        self.enable_thinking = enable_thinking

        # 可从 config 注入，否则取默认
        self.max_retries = getattr(config, "attribute_max_retries", 3)
        self.min_score = getattr(config, "attribute_min_score", 7)

        self.graph = self._build_graph()

    # ---------------- schema 解析 ----------------
    def load_schema(self, schema):
        self.entity_types = schema.get("entities")
        self.schema_type_order = [e["type"] for e in self.entity_types]
        self.type2description = {e["type"]: e["description"] for e in self.entity_types}
        self.type2property = {e["type"]: e["properties"] for e in self.entity_types}

    def _resolve_type(self, entity_type):
        if isinstance(entity_type, list):
            if "Event" in entity_type:
                return "Event"
            for t in self.schema_type_order:
                if t in entity_type:
                    return t
            return entity_type[0] if entity_type else "Concept"
        return entity_type or "Concept"

    def _resolve_properties(self, entity_type):
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
            return "；".join(descs)
        if entity_type == "Event":
            return self.type2description.get("Event", "")
        return self.type2description.get(entity_type, "")

    # ---------------- 工具函数 ----------------
    @staticmethod
    def _parse_attribute_keys(attribute_definitions: str) -> List[str]:
        # 解析 "- **key**：" 形式的属性名
        keys = re.findall(r"-\s*\*\*(.+?)\*\*", attribute_definitions or "")
        return [k.strip() for k in keys if k.strip()]

    @staticmethod
    def _ensure_dict(maybe_json_or_dict):
        if isinstance(maybe_json_or_dict, dict):
            return maybe_json_or_dict
        if isinstance(maybe_json_or_dict, str) and maybe_json_or_dict.strip():
            try:
                return json.loads(maybe_json_or_dict)
            except Exception:
                # 若是纯文本，返回空 dict
                return {}
        return {}

    @staticmethod
    def _json_dumps(obj) -> str:
        try:
            return json.dumps(obj, ensure_ascii=False)
        except Exception:
            return "{}"

    def _estimate_score_from_completeness(self, attrs: Dict[str, Any], expected_keys: List[str]) -> float:
        """基于非空比例估分到 0-10（无 score 时兜底）。"""
        if not expected_keys:
            # 无 schema 时，用已有非空比例
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
                # 非字符串也视为“有值”
                non_empty += 1 if v is not None else 0
        ratio = non_empty / max(1, len(expected_keys))
        return round(ratio * 10.0, 2)

    # ---------------- 节点：上下文（仅一次） ----------------
    def get_related_context(self, state):
        """
        一次性构建完整上下文：
        documents = search_by_ids(source_chunks)
        +（若非事件/动作/情感/目标）ParentChildRetriever 检索的补充
        """
        entity_name = state["entity_name"]
        entity_type = state["entity_type"]
        source_chunks = state.get("source_chunks", []) or []

        doc_objs = self.document_vector_store.search_by_ids(source_chunks) or []
        doc_texts = []
        for d in doc_objs:
            if hasattr(d, "content"):
                doc_texts.append(d.content)
            elif isinstance(d, dict) and "content" in d:
                doc_texts.append(d["content"])

        extra_texts = []
        if entity_type in self.global_entity_types:
            try:
                extra = self.retriever.retrieve(entity_name, ks=20, kp=5, window=1, topn=5) or []
                for item in extra:
                    if hasattr(item, "content"):
                        extra_texts.append(item.content)
                    elif isinstance(item, dict) and "content" in item:
                        extra_texts.append(item["content"])
            except Exception:
                pass

        merged_texts = doc_texts + extra_texts
        new_text = "\n".join(t for t in merged_texts if t) or state.get("content", "")

        return {**state, "content": new_text}

    # ---------------- 节点：抽取 ----------------
    def extract(self, state):
        entity_name = state["entity_name"]
        entity_type_raw = state["entity_type"]
        entity_type = self._resolve_type(entity_type_raw)
        type_description = self._resolve_description(entity_type_raw)
        properties = self._resolve_properties(entity_type_raw)

        feedbacks_list = state.get("feedbacks", []) or []
        feedbacks = "\n".join(feedbacks_list)
        attribute_definitions = format_property_definitions(properties)

        result = self.extractor.extract_entity_attributes(
            text=state["content"],                       # 完整上下文
            entity_name=entity_name,
            description=type_description,
            entity_type=entity_type,
            attribute_definitions=attribute_definitions,
            system_prompt=self.system_prompt,
            previous_results=state.get("previous_result", ""),
            feedbacks=feedbacks,
            original_text=state.get("original_text", "")
        )
        # extractor 返回 JSON 字符串；校正并解析
        result = json.loads(correct_json_format(result))
        # print("[CHECK] attribute extraction result:", result)

        # 以 dict 保存 attributes，便于后续反思与评分
        attrs = self._ensure_dict(result.get("attributes", {}))
        new_desc = result.get("new_description", "")

        return {
            **state,
            "attributes": attrs,
            "new_description": new_desc,
            "previous_result": self._json_dumps(result),
        }

    # ---------------- 节点：反思/评分/修复 ----------------
    def reflect(self, state):
        """
        与 AttributeReflector 对接：
        - 兼容两种返回：
          A) 质量审查：{feedbacks, score, attributes_to_retry}
          B) 修复输出：{attributes, new_description}（无 score）
        - 若无 score：按字段“非空比例”估分
        - 若返回了 attributes/new_description：回填到 state 作为下一轮起点
        """
        entity_type_raw = state["entity_type"]
        entity_type = self._resolve_type(entity_type_raw)
        description = self._resolve_description(entity_type_raw)
        properties = self._resolve_properties(entity_type_raw)

        attribute_definitions = format_property_definitions(properties)
        expected_keys = self._parse_attribute_keys(attribute_definitions)

        # 传给反思器的 attributes 以字符串形式喂入提示
        attrs_for_prompt = state.get("attributes", {})
        attrs_json_for_prompt = self._json_dumps(self._ensure_dict(attrs_for_prompt))

        # 允许在反思阶段提供原文，便于“正确性”核对
        
        result = self.extractor.reflect_entity_attributes(
            entity_type=entity_type,
            description=description,
            attribute_definitions=attribute_definitions,
            attributes=attrs_json_for_prompt,
            system_prompt=self.system_prompt,
            original_text=state.get("original_text", "")
        )
        # print("[CHECK] attribute reflection result:", result)
        result = json.loads(correct_json_format(result))

        # 累计轮次
        attempt = int(state.get("attempt", 0)) + 1

        # 读取可用的反馈/列表
        feedbacks_old = state.get("feedbacks", []) or []
        feedbacks_new = result.get("feedbacks", []) or []
        merged_feedbacks = feedbacks_old + feedbacks_new

        # 若反思器返回了直接修复后的 attributes/new_description，则回填
        attrs_fixed = self._ensure_dict(result.get("attributes", {}))
        new_desc_fixed = result.get("new_description", None)

        current_attrs = attrs_fixed if attrs_fixed else self._ensure_dict(state.get("attributes", {}))
        current_desc = new_desc_fixed if isinstance(new_desc_fixed, str) else state.get("new_description", "")

        # 评分：优先用反思器的 score；否则按完整度估分
        score = result.get("score", None)
        if score is None:
            score = self._estimate_score_from_completeness(current_attrs, expected_keys)

        # 需要重试的字段：反思器给了就用；否则找出空/缺失字段
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
            # 回填后的最新视图
            "attributes": current_attrs,
            "new_description": current_desc,
            # 也把“当前视图”写入 previous_result，便于下一轮作为“已有结果”
            "previous_result": self._json_dumps({
                "new_description": current_desc,
                "attributes": current_attrs
            })
        }

    # ---------------- 分支逻辑 ----------------
    def _check_reflection(self, state):
        attempt = int(state.get("attempt", 0))
        score = float(state.get("score", 0.0))
        if score >= self.min_score:
            return "complete"
        if attempt >= self.max_retries:
            return "complete"
        return "retry"

    # ---------------- 图结构 ----------------
    def _build_graph(self):
        builder = StateGraph(dict)
        builder.add_node("get_related_context", self.get_related_context)
        builder.add_node("extract", self.extract)
        builder.add_node("reflect", self.reflect)

        builder.set_entry_point("get_related_context")
        builder.add_edge("get_related_context", "extract")
        builder.add_edge("extract", "reflect")
        builder.add_conditional_edges("reflect", self._check_reflection, {
            "complete": END,
            "retry": "extract"
        })
        return builder.compile()

    # ---------------- 外部接口（保持不变） ----------------
    def run(self, text: str, entity_name: str, entity_type: str, source_chunks: list = [], original_text: str = None):
        return self.graph.invoke({
            "content": text,                      # 完整上下文（首次会被 get_related_context 覆盖/合并）
            "entity_name": entity_name,
            "entity_type": entity_type,
            "source_chunks": source_chunks or [],
            "original_text": original_text or text,
            "previous_result": "",
            "feedbacks": [],
            "attempt": 0,
            "score": 0.0
        })

    async def arun(
        self,
        text: str,
        entity_name: str,
        entity_type: str,
        source_chunks: list = None,
        original_text: str | None = None,
        timeout: int = 120,
        max_attempts: int = 3,
        backoff_seconds: int = 30,
    ):
        """
        异步接口：保留超时与退避；抽取-反思的重试由图与 score 控制
        """
        payload = {
            "content": text,
            "entity_name": entity_name,
            "entity_type": entity_type,
            "source_chunks": source_chunks or [],
            "original_text": original_text or text,
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
            # 仅针对“整体调用卡顿”做少量重试；流程内部重试不受影响
            for i in range(1, max_attempts):
                try:
                    await asyncio.sleep(backoff_seconds * i)
                    result = await asyncio.wait_for(self.graph.ainvoke(payload), timeout=timeout)
                    return result.get("best_result", result)
                except asyncio.TimeoutError:
                    continue
            return {"attributes": {}, "error": f"timeout after {max_attempts} attempts"}
        except Exception as e:
            return {"attributes": {}, "error": str(e)}

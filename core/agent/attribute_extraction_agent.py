import json
from enum import Enum
from typing import Dict
from langgraph.graph import StateGraph, END
from core.utils.format import correct_json_format
from ..storage.vector_store import VectorStore
from core.memory.vector_memory import VectorMemory
from core.builder.manager.information_manager import InformationExtractor
import asyncio 
from core.model_providers.openai_rerank import OpenAIRerankModel
from retriever.vectordb_retriever import ParentChildRetriever

def format_property_definitions(properties: Dict[str, str]) -> str:
    return "\n".join([f"- **{key}**：{desc}" for key, desc in properties.items()])


class AttributeExtractionAgent:
    def __init__(self, config, llm, system_prompt, schema, enable_thinking=True):
        self.config = config
        self.extractor = InformationExtractor(config, llm)
        # self.vector_store = VectorStore(config)
        self.history_memory = VectorMemory(config, "history_memory")
        self.load_schema(schema)
        self.graph = self._build_graph()
        self.system_prompt = system_prompt
        self.reranker = OpenAIRerankModel(config)
        self.document_vector_store = VectorStore(config, "documents")
        self.sentence_vector_store = VectorStore(config, "sentences")
        self.enable_thinking = enable_thinking
        self.retriever = ParentChildRetriever(doc_vs=self.document_vector_store, sent_vs=self.sentence_vector_store, reranker=self.reranker)
        

    def load_schema(self, schema):
        self.entity_types = schema.get("entities")
        self.schema_type_order = [e["type"] for e in self.entity_types]  # 记录类型优先顺序
        self.type2description = {e["type"]: e["description"] for e in self.entity_types}
        self.type2property = {e["type"]: e["properties"] for e in self.entity_types}
        
    def _resolve_type(self, entity_type):
        """返回主类型：含 Event → 'Event'；否则按 schema 顺序取第一个；兜底 'Concept'。"""
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
        合并 properties：
        - 若含 Event：直接返回 Event 的 properties
        - 否则按 schema_type_order 遍历，做并集；遇到同 key 不覆盖（保留先出现者）
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
        # 单一类型
        if entity_type == "Event":
            return self.type2property.get("Event", {})
        return self.type2property.get(entity_type, {})
    
    def _resolve_description(self, entity_type):
        """
        合并描述（用于提示）：
        - 若含 Event：只返回 Event 描述
        - 否则把每个类型的描述按顺序拼接（可选）
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
            return "；".join(descs)
        if entity_type == "Event":
            return self.type2description.get("Event", "")
        return self.type2description.get(entity_type, "")
        
    def extract(self, state):
        entity_name = state["entity_name"]
        # entity_type = state["entity_type"]
        # type_description = self.type2description.get(entity_type, "")
        # properties = self.type2property.get(entity_type, {})
        entity_type_raw = state["entity_type"]
        entity_type = self._resolve_type(entity_type_raw)
        type_description = self._resolve_description(entity_type_raw)   # 合并后的描述（或 Event）
        properties = self._resolve_properties(entity_type_raw)          # 合并后的属性（或 Event）
        
        feedbacks = state.get("feedbacks", [])
        feedbacks = "\n".join(feedbacks)
        
        attribute_definitions = format_property_definitions(properties)

        result = self.extractor.extract_entity_attributes(
            text=state["content"],
            entity_name=entity_name,
            description=type_description,
            entity_type=entity_type,
            attribute_definitions=attribute_definitions,
            system_prompt=self.system_prompt,
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
        # entity_type = state["entity_type"]
        # description = self.type2description.get(entity_type, "")
        # attribute_definitions = format_property_definitions(
        #     self.type2property.get(entity_type, {})
        # )
        entity_type_raw = state["entity_type"]
        entity_type = self._resolve_type(entity_type_raw)
        description = self._resolve_description(entity_type_raw)
        attribute_definitions = format_property_definitions(
            self._resolve_properties(entity_type_raw)
        )
        attributes = state.get("attributes", {})
        # print("[CHECK]", attributes, type(attributes))

        result = self.extractor.reflect_entity_attributes(
            entity_name=entity_name,
            entity_type=entity_type,
            description=description,
            attribute_definitions=attribute_definitions,
            attributes=attributes,
            system_prompt=self.system_prompt,
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
            documents = self.document_vector_store.search_by_ids(source_chunks)
            if not documents:
                results = [doc.content for doc in self.retriever.retrieve(entity_name, ks=10, kp=5, window=1, topn=5)]
        else:
            results = [doc.content for doc in self.retriever.retrieve(entity_name, ks=20, kp=5, window=1, topn=5)]

        query = f"以下哪些内容与实体{entity_name}的属性：" + "、".join(attributes_to_retry) + "相关？"
        retrieved_docs = self.reranker.rerank(query=query, documents=results)
        retrieved_docs = [doc["document"]["text"] for doc in retrieved_docs if doc["relevance_score"] >= 0.5]
        
        new_text = "\n".join(retrieved_docs)

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
        异步属性抽取（带超时 + 重试）

        Parameters
        ----------
        text : str
            待抽取文本（通常是实体描述）
        entity_name : str
            实体名称
        entity_type : str
            实体类型
        source_chunks : list
            来源 chunk ID 列表
        original_text : str | None
            原始全文（可选）
        timeout : int
            单次调用最长等待秒数
        max_attempts : int
            最大尝试次数（1 次正常 + max_attempts-1 次重试）
        backoff_seconds : int
            超时后退避秒数的基准（线性：n×backoff_seconds）
        """
        source_chunks = source_chunks or []
        attempt = 0

        while attempt < max_attempts:
            try:
                coro = self.graph.ainvoke({
                    "content": text,
                    "entity_name": entity_name,
                    "entity_type": entity_type,
                    "source_chunks": source_chunks,
                    "original_text": original_text or text,
                    "previous_result": "",
                    "feedbacks": [],
                    "has_added_context": False
                })
                result = await asyncio.wait_for(coro, timeout=timeout)
                return result.get("best_result", result)   # 正常成功
            except asyncio.TimeoutError:
                attempt += 1
                if attempt >= max_attempts:
                    return {"attributes": {}, "error": f"timeout after {max_attempts} attempts"}
                await asyncio.sleep(backoff_seconds * attempt)  # 简单退避
            except Exception as e:
                # 其它异常不做重试；若想也重试，可改为同超时逻辑
                return {"attributes": {}, "error": str(e)}

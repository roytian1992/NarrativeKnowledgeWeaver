# qa_agent.py （检索器版）
from __future__ import annotations
from typing import List, Dict, Any, Optional, Literal
import os
import json

from core.utils.config import KAGConfig
from core.storage.graph_store import GraphStore
from core.storage.vector_store import VectorStore
from core.utils.neo4j_utils import Neo4jUtils
from core.model_providers.openai_rerank import OpenAIRerankModel
from qwen_agent.agents import Assistant
from core.model_providers.openai_llm import OpenAILLM
from langchain_core.documents import Document


# 注意：你原先写的是 DOCT_TYPE_DESCRIPTION，这里修正为 DOC_TYPE_DESCRIPTION
from core.utils.format import DOC_TYPE_META, DOC_TYPE_DESCRIPTION

from core.functions.tool_calls import (
    # 图数据库工具
    EntityRetrieverName,
    EntityRetrieverID,
    SearchRelatedEntities,
    GetRelationSummary,
    GetCommonNeighbors,
    QuerySimilarEntities,
    FindPathsBetweenNodes,
    TopKByCentrality,
    FindRelatedEventsAndPlots,
    GetKHopSubgraph,
    # 向量数据库工具
    VDBHierdocsSearchTool,
    VDBDocsSearchTool,
    VDBGetDocsByChunkIDsTool,
    VDBSentencesSearchTool,
    GetCoSectionEntities,
    Search_By_Character,
    Search_By_Scene,
    Chunk_To_Scene,
    Scene_To_Chunks,
    NLP2SQL_Query,
    BM25SearchDocsTool
)

# —— 检索器版系统提示词（仅检索，不脑补）——
DEFAULT_SYSTEM_MESSAGE = (
    "你是一名知识图谱 + 向量数据库驱动的问答智能体。\n"
    "当问题包含明确的实体时，优先调用图谱检索工具；当问题更偏段落/背景/长文内容时，优先调用向量检索工具（层级检索优先）。\n"
    "最终的答案不要省略必要的、和问题相关的内容。你的作用更像是一个信息检索器，不要给用户提出建议。\n"
)

def prepare_knowledge(schema: Dict[str, Any], doc_type: str) -> str:
    """
    读取图谱 schema，返回实体类型和关系类型的描述文本（用于注入到 Assistant 的 knowledge）。
    """
    entity_types = schema.get("entities", [])
    relation_type_groups = schema.get("relations", {})

    # ===== 实体类型描述 =====
    entity_type_description_text = "\n".join(
        f"- {e['type']}: {e.get('description', '')}" for e in entity_types
    )
    entity_type_description_text += DOC_TYPE_DESCRIPTION.get(doc_type, {}).get("entity", "")
    entity_type_description_text += "\n- Plot: 故事情节，表示剧本/小说中的主要情节线。"

    # ===== 关系类型描述 =====
    relation_type_description_text = "\n".join(
        f"- {r['type']}: {r.get('description', '')}"
        for group in relation_type_groups.values()
        for r in group
    )
    relation_type_description_text += DOC_TYPE_DESCRIPTION.get(doc_type, {}).get("relation", "")

    relation_type_description_text += """\n\n【情节、事件相关的关系类型】：
- HAS_EVENT: 情节包含关系（Plot → Event）
- EVENT_CAUSES: 事件因果关系（Event A 导致 Event B）
- EVENT_INDIRECT_CAUSES: 事件间的间接因果关系（Event A 间接触发 Event B）
- EVENT_PART_OF: 事件组成关系（Event A 属于更大 Event B 的一部分）
- PLOT_PREREQUISITE_FOR: 情节先决关系（Plot A 是 Plot B 的前提条件）
- PLOT_ADVANCES: 情节推进关系（Plot A 推动了 Plot B 的发展）
- PLOT_BLOCKS: 情节阻碍关系（Plot A 阻碍或延缓了 Plot B）
- PLOT_RESOLVES: 情节解决关系（Plot A 解决了 Plot B 的冲突或悬念）
- PLOT_CONFLICTS_WITH: 情节冲突关系（Plot A 与 Plot B 相互对立或冲突）
- PLOT_PARALLELS: 情节并行关系（Plot A 与 Plot B 平行展开，存在呼应或对照）
"""
    full_knowledge = (
        "当前 Neo4j 知识图谱包含以下内容：\n"
        f"【实体类型】\n{entity_type_description_text}\n\n"
        f"【基础关系类型】\n{relation_type_description_text}\n"
        f"实体属性 source_chunks 表示实体来源的文档片段的chunk_id列表，可以通过向量数据库工具 vdb_get_docs_by_chunk_ids 定位到文档片段的内容。\n"
        f"需要基于关键词进行检索时可以调用 bm25_search_docs。"
    )
    sql_cols = ["名称", "类别", "子类别", "外观", "状态", "相关角色", "文中线索", "补充信息", "chunk_id", "场次", "场次名", "子场次名"]
    col_txt = ", ".join(sql_cols)
    if doc_type == "screenplay":
        full_knowledge += "\n另外，服饰、化妆、道具的信息保存在 SQL 数据库中，表名为 CMP_info，有以下一些column name：\n{col_txt}\n"
    return full_knowledge

Mode = Literal["hybrid", "graph_only", "vector_only"]

class QuestionAnsweringAgent:
    """
    纯检索器版本：
      - 保留 Mode 控制（graph_only / vector_only / hybrid）
      - 保留系统提示词与 knowledge 注入
      - 删除多轮/历史逻辑；每次调用 ask() 仅以单轮消息驱动工具检索
    """

    def __init__(
        self,
        config: KAGConfig,
        *,
        doc_type: Optional[str] = None,
        system_message: Optional[str] = None,
        rag_cfg: Optional[Dict[str, Any]] = None,
        reranker: Optional[Any] = None,
        extra_tools: Optional[List[Any]] = None,
        mode: Mode = "hybrid",
        doc_path: str = None,
    ):
        self.config = config
        self.db_path = os.path.join(self.config.storage.sql_database_path, "CMP.db")
        self.doc_type = doc_type or config.knowledge_graph_builder.doc_type
        self._base_system_message = system_message or DEFAULT_SYSTEM_MESSAGE
        self.rag_cfg = rag_cfg or {}

        # —— Mode 策略 ——
        self.mode: Mode = mode

        # ---- Graph / Neo4j ----
        self.graph_store = GraphStore(config)
        self.neo4j_utils = Neo4jUtils(self.graph_store.driver, doc_type=self.doc_type)
        self.neo4j_utils.load_embedding_model(config.graph_embedding)
        self.llm = OpenAILLM(config)
        # ---- Vector Stores ----
        self.document_vector_store = VectorStore(config, "documents")
        self.sentence_vector_store = VectorStore(config, "sentences")
        self.reranker = reranker or OpenAIRerankModel(config)

        # ---- LLM ----
        self.llm_cfg = {
            "model": config.llm.model_name,
            "model_server": config.llm.base_url,
            "api_key": config.llm.api_key,
        }

        # 工具池（分组）
        self._graph_tools = self._build_graph_tools()
        self._vdb_tools = self._build_vdb_tools(reranker=self.reranker)
        self._native_tools = self._build_native_tools(doc_path=doc_path)
        self._extra_tools = extra_tools or []

        # mode -> knowledge / system_message
        self._current_knowledge = self._build_knowledge(self.mode)
        self._current_system_message = self._build_system_message(self.mode)

        # Assistant（由 mode 决定工具 + 系统提示词）
        self._rebuild_assistant()

    # ---------- Knowledge / System Message ----------
    def _build_system_message(self, mode: Mode) -> str:
        """不同 Mode 对应不同系统提示词收尾说明（不改变核心“只检索”原则）"""
        if mode == "graph_only":
            suffix = "（当前模式：仅图数据库工具；请只使用图谱工具、关键词检索、SQL数据库进行检索与返回结果。）"
        elif mode == "vector_only":
            suffix = "（当前模式：仅向量数据库工具；请只依据向量检索、关键词检索、SQL数据库与重排结果返回片段。）"
        elif mode == "native":
            suffix = "（当前模式：仅原始工具；请只依据SQL数据库和关键词检索器。）"
        else:
            suffix = "（当前模式：图谱+向量混合；请根据查询在图/向量工具间自适应选择，但仍只返回检索结果。）"

        if self._current_knowledge:
            suffix += f"\n\n知识背景：\n{self._current_knowledge}"
        return f"{self._base_system_message}\n\n{suffix}"

    def _load_graph_schema(self) -> Optional[Dict[str, Any]]:
        schema_path = os.path.join(self.config.storage.graph_schema_path, "graph_schema.json")
        if os.path.exists(schema_path):
            with open(schema_path, "r", encoding="utf-8") as f:
                graph_schema = json.load(f)
                return graph_schema

        default_path = getattr(self.config.probing, "default_graph_schema_path", None)
        if default_path and os.path.exists(default_path):
            with open(default_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    def _build_knowledge(self, mode: Mode) -> str:
        """
        - graph_only：注入图谱 schema 说明文本；
        - vector_only：留空（如需，可自行扩展为向量侧知识）；
        - hybrid：先注入图谱知识（向量侧先留空）。
        """
        if mode == "graph_only":
            schema = self._load_graph_schema()
            return prepare_knowledge(schema, self.doc_type) if schema else ""
        elif mode == "vector_only":
            return ""
        else:  # hybrid
            parts = []
            schema = self._load_graph_schema()
            if schema:
                parts.append(prepare_knowledge(schema, self.doc_type))
            return "\n\n".join([p for p in parts if p])

    # ---------- Public: Mode ----------
    def set_mode(self, mode: Mode) -> None:
        """切换 mode：重建 system_message / knowledge / Assistant（工具集合）。"""
        if mode not in ("hybrid", "graph_only", "vector_only"):
            raise ValueError(f"Unsupported mode: {mode}")
        if mode != self.mode:
            self.mode = mode
            self._current_knowledge = self._build_knowledge(mode)
            self._current_system_message = self._build_system_message(mode)
            self._rebuild_assistant()

    # ---------- Public: 检索 ----------
    def ask(
        self,
        user_text: str,
        *,
        lang: Literal["zh", "en"] = "zh",
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        单轮检索：
        - 不维护/回写任何历史；
        - 仅用本轮 user_text 触发工具检索并返回 Assistant 的完整消息序列（含 function 调用结果）。
        """
        system_prompt = """你在选择回答问题的工具时，可以参考以下经验规则（不是硬性约束，而是倾向性建议）：

1. 缩略语、全称与术语解释类问题  
    - 使用关键词检索工具（如 BM25）定位定义或者retrieve_entity_by_name，如果不确定实体类型就设为Entity。  
    - 必要时交叉验证多个片段，确保答案完整。

2. 角色出场、行为或服装类问题  
    - 使用角色检索工具或实体检索工具获取相关片段。  
    - 如果需要场次信息，结合场次映射工具（chunk_to_scene）。

3. 事件时间线与结果类问题  
    - 使用向量数据库检索（vdb 系列工具）获取全局性或时间相关的段落。  
    - 需要映射到场次时，再使用 chunk_to_scene。

4. 物品出现与状态变化类问题  
    - 先用BM25关键词检索找到出现位置。  
    - 再用向量检索补充状态、变化等上下文信息。

5. 地点或场景变化类问题  
    - 使用实体检索工具retrieve_entity_by_name获取地点或场景的定义。  
    - 结合向量数据库检索找到变化后的描述。  
    - 若要求场次，使用 chunk_to_scene 映射。

6. 装备定义与属性类问题  
    - 使用结构化查询工具（如 nlp2sql_query）从属性表中获取定义或属性。  
    - 如果涉及使用者或场次，再结合关键词检索或角色检索。

7. 其它容易出错的问题
    - 高风险类型：外貌细节、型号枚举、虚构时间线、物品来源、经济/生活细节。
    - 建议：优先调用 bm25_search_docs 检索原始文档再作答。
    - 补充：如果其他检索方式没有结果，也建议再试一次 bm25_search_docs，而不是直接空答。
"""
        system_prompt = "你是一个可以基于问题选择工具回答的智能助手，希望你能灵活的使用工具，失败后用别的尝试。"
        messages: List[Dict[str, Any]] = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_text }]
        resp = self.assistant.run_nonstream(
            messages=messages,
            lang=lang,
            **kwargs,
        )
        return resp

    def extract_final_text(self, responses: List[Dict[str, Any]]) -> str:
        """
        若需要拿到最终自然语言文本，可用此函数：
        - 优先取最后一个 assistant 文本；
        - 退化为拼接 function 输出（如果没有 assistant 文本）。
        """
        final_text = ""
        fallback_chunks: List[str] = []
        for msg in responses:
            role = msg.get("role")
            content = msg.get("content") or ""
            if role == "assistant" and content.strip():
                final_text = content
            elif role == "function" and content.strip():
                fallback_chunks.append(content.strip())
        if final_text:
            return final_text.strip()
        if fallback_chunks:
            return "\n\n".join(fallback_chunks).strip()
        return ""
    
    def extract_tool_uses(self, responses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        提取中间使用过的工具调用过程（调用名、参数、输出）。
        返回格式：
        [
          {
            "tool_name": ...,
            "tool_arguments": ...,
            "tool_output": ...
          },
          ...
        ]
        """
        tool_uses: List[Dict[str, Any]] = []
        pending: Dict[str, Any] = None  # 临时保存 assistant 的 function_call

        for msg in responses:
            role = msg.get("role")
            # assistant 发起的工具调用
            if role == "assistant" and "function_call" in msg:
                fc = msg["function_call"] or {}
                pending = {
                    "tool_name": fc.get("name") or "unknown_tool",
                    "tool_arguments": fc.get("arguments") or "",
                    "tool_output": None,
                }

            # function 返回的结果
            elif role == "function":
                # 优先用 function 消息里的 name 作为 tool_name（有时 assistant 没写）
                tool_name = msg.get("name") or (pending.get("tool_name") if pending else "unknown_tool")
                tool_output = msg.get("content") or ""

                if pending and pending["tool_output"] is None:
                    pending["tool_output"] = tool_output
                    # 如果 function 消息带有 name，覆盖掉
                    if msg.get("name"):
                        pending["tool_name"] = msg["name"]
                    tool_uses.append(pending)
                    pending = None
                else:
                    # 孤立的 function 消息（没有对应的 function_call）
                    tool_uses.append({
                        "tool_name": tool_name,
                        "tool_arguments": None,
                        "tool_output": tool_output,
                    })

        return tool_uses


    # ---------- Internal ----------
    def _build_graph_tools(self) -> List[Any]:
        emb_cfg = self.config.graph_embedding
        return [
            EntityRetrieverName(self.neo4j_utils, emb_cfg),
            EntityRetrieverID(self.neo4j_utils, emb_cfg),
            SearchRelatedEntities(self.neo4j_utils, emb_cfg),
            GetRelationSummary(self.neo4j_utils),
            GetCommonNeighbors(self.neo4j_utils),
            QuerySimilarEntities(self.neo4j_utils, emb_cfg),
            FindPathsBetweenNodes(self.neo4j_utils),
            TopKByCentrality(self.neo4j_utils),
            GetCoSectionEntities(self.neo4j_utils),
            FindRelatedEventsAndPlots(self.neo4j_utils),
            GetKHopSubgraph(self.neo4j_utils),
        ]

    def _build_vdb_tools(self, *, reranker: Any) -> List[Any]:
        return [
            VDBDocsSearchTool(self.document_vector_store),
            VDBGetDocsByChunkIDsTool(self.document_vector_store),
            VDBSentencesSearchTool(self.sentence_vector_store),
            VDBHierdocsSearchTool(
                document_vector_store=self.document_vector_store,
                sentence_vector_store=self.sentence_vector_store,
                reranker=reranker,
            ),
        ]
    
    def _build_native_tools(self, *, doc_path: str = None) -> List[Any]:
        if not doc_path:
            base = os.path.join(self.config.storage.knowledge_graph_path, "all_document_chunks.json")
        else:
            base = doc_path
        with open(base, "r") as f:
            data = json.load(f)
        
        keys_to_drop = {"chunk_index", "chunk_type", "doc_title", "order", "total_doc_chunks"}
        documents: List[Document] = []

        for item in data:
            chunk_id = item.get("id")
            content = (item.get("content") or "").strip()
            if not chunk_id or not content:
                continue

            meta: Dict[str, Any] = dict(item.get("metadata") or {})
            meta["chunk_id"] = chunk_id
            for key in list(keys_to_drop):
                if key in meta:
                    del meta[key]

            # ✅ 统一用 LangChain 的 Document
            documents.append(Document(page_content=content, metadata=meta))

        return [
            Search_By_Character(self.db_path),
            Search_By_Scene(self.db_path),
            Chunk_To_Scene(self.db_path),
            Scene_To_Chunks(self.db_path),
            NLP2SQL_Query(self.db_path, self.llm),
            BM25SearchDocsTool(documents)
        ]

    def _select_tools(self, mode: Mode) -> List[Any]:
        if mode == "graph_only":
            return [*self._graph_tools, *self._native_tools, *self._extra_tools]
        elif mode == "vector_only":
            return [*self._vdb_tools, *self._native_tools, *self._extra_tools]
        elif mode == "native":
            return [*self._native_tools, *self._extra_tools]
        else:
            return [*self._graph_tools, *self._vdb_tools, *self._native_tools, *self._extra_tools]

    def _rebuild_assistant(self) -> None:
        tools = self._select_tools(self.mode)
        self.assistant = Assistant(
            function_list=tools,
            llm=self.llm_cfg,
            system_message=self._current_system_message,
            rag_cfg=self.rag_cfg,
        )

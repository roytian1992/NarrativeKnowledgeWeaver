import json
from typing import Dict
from langgraph.graph import StateGraph, END
from core.utils.format import correct_json_format
from core.builder.manager.information_manager import InformationExtractor
from core.builder.manager.probing_manager import GraphProber
# from core.builder.reflection import DynamicReflector
from core.agent.knowledge_extraction_agent import InformationExtractionAgent
import asyncio 
from core.model_providers.openai_rerank import OpenAIRerankModel
from collections import Counter
from core.utils.prompt_loader import PromptLoader
import os
from pathlib import Path
from tqdm import tqdm
from ..utils.config import KAGConfig
import re

class GraphProbingAgent:
    def __init__(self, config: KAGConfig, llm, reflector):
        self.config = config
        self.llm = llm
        self.extractor = InformationExtractor(config, llm)
        prompt_dir = config.knowledge_graph_builder.prompt_dir
        self.prompt_loader = PromptLoader(prompt_dir)
        self.prober = GraphProber(config, llm)
        self.reranker = OpenAIRerankModel(config)
        self.doc_type = config.knowledge_graph_builder.doc_type
        self.reflector = reflector
        self.graph = self._build_graph()
        self.feedbacks = []
        self.score_threshold = self.config.agent.score_threshold
        self.relation_prune_threshold = self.config.probing.relation_prune_threshold
        self.entity_prune_threshold = self.config.probing.entity_prune_threshold
        self.max_workers = self.config.probing.max_workers
        self.max_retries = self.config.probing.max_retries
        self.experience_limit = self.config.probing.experience_limit
        self.probing_mode = self.config.probing.probing_mode
        self.refine_background = self.config.probing.refine_background
        if self.config.probing.task_goal:
            self.task_goals = "\n".join(self.load_goals(self.config.probing.task_goal))
        else:
            self.task_goals = ""
       
    def construct_system_prompt(self, background, abbreviations):
        
        background_info = self.get_background_info(background, abbreviations)
        
        if self.doc_type == "screenplay":
            system_prompt_id = "agent_prompt_screenplay"
        else:
            system_prompt_id = "agent_prompt_novel"
            
        system_prompt_text = self.prompt_loader.render_prompt(system_prompt_id, {"background_info": background_info})
        return system_prompt_text
    
    def load_goals(self, path: str):
        # 读文本（utf-8-sig 兼容带 BOM 的文件）
        text = Path(path).read_text(encoding="utf-8-sig")
        goals = []
        for line in text.splitlines():
            s = line.strip()
            if not s or s.startswith("#"):  # 跳过空行/注释
                continue
            # 去掉行首序号/符号（如：1. / 2) / 3、/ 4: / - ）
            s = re.sub(r'^\s*(?:[-•]+|\d+[.)、:：]?)\s*', '', s)
            goals.append(s)
        # 去重且保序
        return list(dict.fromkeys(goals))
    
    def get_background_info(self, background, abbreviations):
        """将背景与术语列表渲染为 Markdown。
        - 必填：name、description
        - 可选：abbr、full、zh
        - 当 name 与 abbr 相同（不区分大小写）时，只显示一次，不重复
        """
        def _s(val):
            return val.strip() if isinstance(val, str) and val.strip() else None

        bg_block = f"**背景设定**：{background}\n" if _s(background) else ""

        def fmt(item: dict) -> str:
            if not isinstance(item, dict):
                return ""

            name = _s(item.get("name"))
            desc = _s(item.get("description"))
            abbr = _s(item.get("abbr"))
            full = _s(item.get("full"))
            zh   = _s(item.get("zh"))

            # 若既无 name 又无 abbr，或无 description，则跳过（不生成空行）
            if not desc or (not name and not abbr):
                return ""

            # 标题：优先 name；若有 abbr 且与 name 不同，则显示 name (abbr)
            title = None
            if name and abbr and name.lower() != abbr.lower():
                title = f"{name} ({abbr})"
            else:
                # name 优先；否则退回 abbr；再退回 'N/A'
                title = name or abbr or "N/A"

            parts = [desc]
            if full:
                parts.append(full)
            if zh:
                parts.append(zh)

            return f"- **{title}**: " + " - ".join(parts) if parts else f"- **{title}**"

        abbr_lines = []
        if isinstance(abbreviations, list):
            for it in abbreviations:
                line = fmt(it)
                if line:
                    abbr_lines.append(line)

        abbr_block = "\n".join(abbr_lines)

        if bg_block and abbr_block:
            return f"{bg_block}\n以下是一些专业术语和缩写：\n{abbr_block}"
        else:
            return bg_block or abbr_block


    def load_schema(self, schema):
        entity_types = schema.get("entities")
        relation_type_groups = schema.get("relations")

        entity_type_description_text = "\n".join(
            f"- {e['type']}: {e['description']}" for e in entity_types
        )
        relation_type_description_text = "\n".join(
            f"- {r['type']}: {r['description']}"
            for group in relation_type_groups.values()
            for r in group
        )
        return entity_type_description_text, relation_type_description_text 
        
    def search_related_experience(self, state):
        
        documents = self.reflector.insight_memory.get(query="人物、关系、背景信息、故事、情节、事件", k=2*self.experience_limit)
        
        documents = [doc.page_content for doc in documents]
        
        related_insights_for_background = self.reranker.rerank(query="背景信息、故事、情节、事件", 
                                                               documents=documents, top_n=min(self.experience_limit, len(documents)))
        
        related_insights_for_background = [insight["document"]["text"] for insight in related_insights_for_background \
                                                if insight["relevance_score"] >= 0.3]
        related_insights_for_background += ["情节/Plot这种实体类型不需要抽取，会在后续的任务中基于抽取的事件/Event进行构建。"]
        
        documents = self.reflector.insight_memory.get(query="术语、缩写", k=2*self.experience_limit)
        documents = [doc.page_content for doc in documents]
        related_insights_for_abbreviations = self.reranker.rerank(query="术语、缩写", 
                                                               documents=documents, top_n=min(self.experience_limit, len(documents)))
        related_insights_for_abbreviations = [insight["document"]["text"] for insight in related_insights_for_abbreviations \
                                                if insight["relevance_score"] >= 0.5 and "术语" in insight["document"]["text"]]
        
        
        documents = self.reflector.insight_memory.get(query="人物、物品、情感、动作、事件、概念、实体", k=2*self.experience_limit)
        documents = [doc.page_content for doc in documents]
        related_insights_for_entity_schema = self.reranker.rerank(query="实体 schema", 
                                                                  documents=documents, top_n=min(self.experience_limit, len(documents)))
        related_insights_for_entity_schema = [insight["document"]["text"] for insight in related_insights_for_entity_schema \
                                                if insight["relevance_score"] >= 0.5]
        
        documents = self.reflector.insight_memory.get(query="事件、情节、动作、关系", k=2*self.experience_limit)
        documents = [doc.page_content for doc in documents]
        related_insights_for_relation_schema = self.reranker.rerank(query="关系 schema", 
                                                                    documents=documents, top_n=min(self.experience_limit, len(documents)))
        related_insights_for_relation_schema = [insight["document"]["text"] for insight in related_insights_for_relation_schema \
                                                if insight["relevance_score"] >= 0.5]
        
        # print({"background_insights": len(related_insights_for_background),
        #     "abbreviation_insights":  len(related_insights_for_abbreviations),
        #     "entity_schema_insights": len(related_insights_for_entity_schema),
        #     "relation_schema_insights": len(related_insights_for_relation_schema)})
        
        return {
            **state,
            "background_insights": "\n".join(related_insights_for_background),
            "abbreviation_insights": "\n".join(related_insights_for_abbreviations),
            "entity_schema_insights": "\n".join(related_insights_for_entity_schema),
            "relation_schema_insights": "\n".join(related_insights_for_relation_schema)
        }

    def generate_schema(self, state):
        # 更新背景信息
        if self.refine_background:
            current_background_info = state.get("background", "")
            background_insights = state.get("background_insights", "")
            result = self.prober.update_background(text=background_insights, 
                                                   current_background=current_background_info)
            result = json.loads(correct_json_format(result))
            new_background = result.get("background", "")
            if not new_background:
                new_background = current_background_info 
            
            current_abbreviations = state.get("abbreviations", [])                
            abbreviation_insights = state.get("abbreviation_insights", "")
            background_info = self.get_background_info(new_background, current_abbreviations)
            # print("[CHECK] background_info: ", background_info)
            result = self.prober.update_abbreviations(text=abbreviation_insights, 
                                                      current_background=background_info)
            result = json.loads(correct_json_format(result))
            abbreviations_ = result.get("abbreviations", [])
            new_abbreviations = current_abbreviations.copy()
            current_abbr_list = [item["name"] for item in current_abbreviations]
            for item in abbreviations_:
                if item["name"] not in current_abbr_list:
                    new_abbreviations.append(item)
                    
            # print("[CHECK] new_abbreviations: ", new_abbreviations)
                    
        else:
            new_background = state.get("background", "")
            new_abbreviations = state.get("abbreviations", [])
            
        current_schema = state.get("schema", {})
        # 更新实体schema
        entity_schema_insights = state.get("entity_schema_insights", "")
        current_entity_schema = current_schema.get("entities", {})
        # if self.feedbacks: 
        #     entity_schema_feedbacks = self.reranker.rerank(query="寻找与entity schema相关的建议", documents=self.feedbacks, top_n=100)
        #     entity_schema_feedbacks = [state.get("reason", "")] + [doc["document"]["text"] for doc in entity_schema_feedbacks if doc["relevance_score"] >= 0.5]
        #     entity_schema_feedbacks = "\n".join(entity_schema_feedbacks)
        # else: 
        #     entity_schema_feedbacks = ""
        entity_schema_feedbacks = "\n".join(self.feedbacks)
            
        result = self.prober.update_entity_schema(text=entity_schema_insights, 
                                                  feedbacks=entity_schema_feedbacks, 
                                                  current_schema=json.dumps(current_entity_schema, indent=2, ensure_ascii=False),
                                                  task_goals=self.task_goals)
        result = json.loads(correct_json_format(result))
        new_entity_schema = result.get("entities", [])
              
        # 更新关系schema
        relation_schema_insights = state.get("relation_schema_insights", "")
        current_relation_schema = current_schema.get("relations", {})
        # if self.feedbacks: 
        #     relation_schema_feedbacks = self.reranker.rerank(query="寻找与relation schema相关的建议", documents=self.feedbacks, top_n=100)
        #     relation_schema_feedbacks = [state.get("reason", "")] + [doc["document"]["text"] for doc in relation_schema_feedbacks if doc["relevance_score"] >= 0.5]
        #     relation_schema_feedbacks = "\n".join(relation_schema_feedbacks)
        # else: 
        #     relation_schema_feedbacks = ""
        relation_schema_feedbacks = "\n".join(self.feedbacks)
        result = self.prober.update_relation_schema(text=relation_schema_insights, 
                                                    feedbacks=relation_schema_feedbacks, 
                                                    current_schema=json.dumps(current_relation_schema, indent=2, ensure_ascii=False),
                                                    task_goals=self.task_goals)
        result = json.loads(correct_json_format(result))
        new_relation_schema = result.get("relations", [])
        
        self.feedbacks = [] # 清空临时记忆
        
        return {
            **state,
            "schema": {"entities": new_entity_schema, "relations": new_relation_schema},
            "background": new_background ,
            "abbreviations": new_abbreviations,
        }

    def test_extractions(self, state):
        all_documents = state["documents"]
        system_prompt = self.construct_system_prompt(state["background"], state["abbreviations"])
        schema = state["schema"]
        extraction_results = asyncio.run(self.test_extractions_(all_documents, system_prompt, schema))
        # print("[CHECK] extraction_results: ", extraction_results)
        entity_types = []
        relation_types = []
        
        for result in extraction_results:
            # print("[CHECK] result: ", result.get("issues", []))
            try:
                self.feedbacks.extend(result.get("issues", [])) # 添加记忆
                entities = result["entities"]
                for entity in entities:
                    entity_types.append(entity["type"])
                relations = result["relations"]
                for relation in relations:
                    relation_types.append(relation["relation_type"])
            except:
                print("result")

                
        entity_type_counter = Counter(entity_types)
        relation_type_counter = Counter(relation_types)

        entity_total = len(entity_types)
        relation_total = len(relation_types)

        # THRESHOLD = self.prune_threshold  # 少于 5% 的类型将被丢弃

        # ------- 计算需丢弃的类型（按占比而非次数） -------
        if entity_total > 0:
            entity_types_to_drop = [
                key for key, cnt in entity_type_counter.items()
                if (cnt / entity_total) < self.entity_prune_threshold
            ]
        else:
            entity_types_to_drop = []

        if relation_total > 0:
            relation_types_to_drop = [
                key for key, cnt in relation_type_counter.items()
                if (cnt / relation_total) < self.relation_prune_threshold
            ]
        else:
            relation_types_to_drop = []

        # ------- 分布文本（维持你原有的比例=小数形式）-------
        entity_type_distribution = ""
        if entity_total > 0:
            for _type, cnt in entity_type_counter.items():
                entity_type_distribution += (
                    f"实体类型 {_type} 数量为：{cnt}  比例为：{round(cnt / entity_total, 3)}\n"
                )

        relation_type_distribution = ""
        if relation_total > 0:
            for _type, cnt in relation_type_counter.items():
                relation_type_distribution += (
                    f"关系类型 {_type} 数量为：{cnt}  比例为：{round(cnt / relation_total, 3)}\n"
                )
        
        print("entity_types_to_drop：")
        print(entity_types_to_drop)
        print("relation_types_to_drop")
        print(relation_types_to_drop)

        
        return {
            **state,
            "entity_type_distribution": entity_type_distribution,
            "relation_type_distribution": relation_type_distribution,
            "entity_types_to_drop": entity_types_to_drop,
            "relation_types_to_drop": relation_types_to_drop,
        }

    async def test_extractions_(self, all_documents, system_prompt, schema):
        information_extraction_agent = InformationExtractionAgent(self.config, self.llm, system_prompt,
                                                                  schema=schema, reflector=self.reflector, mode="probing")
        sem = asyncio.Semaphore(self.max_workers)
        async def _arun(ch):
            async with sem:
                try:
                    if not ch.content.strip():
                        result = {"entities": [], "relations": [], "score": 0, "issues": [], "insights": []}
                    else:
                        result = await information_extraction_agent.arun(ch.content, timeout=300, max_attempts=3, backoff_seconds=60)
                        
                    return result
                except Exception as e:
                    return {"error": str(e), "entities": [], "relations": [], "score": 0, "issues": [], "insights": []}

        tasks = [_arun(ch) for ch in all_documents]
        results = []
        for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="异步抽取中"):
            res = await coro
            results.append(res)
        return results 

    def prune_graph_schema(self, state):
        entity_type_distribution = state["entity_type_distribution"]
        relation_type_distribution = state["relation_type_distribution"]
        entity_type_description_text, relation_type_description_text = self.load_schema(state["schema"])
        result = self.prober.prune_schema(entity_type_distribution=entity_type_distribution, 
                                          relation_type_distribution=relation_type_distribution,
                                          entity_type_description_text=entity_type_description_text,
                                          relation_type_description_text=relation_type_description_text)
        result = json.loads(correct_json_format(result))
        self.feedbacks.extend(result.get("feedbacks", [])) # 添加记忆
        
        return state
    
    def reflect_graph_schema(self, state):
        current_schema = state["schema"]
        BATCH_SIZE = 300

        # 按批次处理 feedbacks
        summaries = []
        for i in range(0, len(self.feedbacks), BATCH_SIZE):
            batch = self.feedbacks[i:i + BATCH_SIZE]
            all_feedbacks = "\n".join(batch)
            result = self.prober.summarize_feedbacks(context=all_feedbacks, max_items=20)
            parsed = json.loads(correct_json_format(result)).get("feedbacks", [])
            summaries.extend(parsed)

        self.feedbacks = summaries

        result = self.prober.reflect_schema(schema=json.dumps(current_schema, indent=2, ensure_ascii=False), feedbacks=self.feedbacks)
        result = json.loads(correct_json_format(result))
        score = float(result["score"])
        reason = result["reason"]
        
        # 删除部分数量小于5%的节点类型和关系类型
        
        entities = current_schema.get("entities", [])
        new_entities = []
        for ent in entities:
            if ent["type"] not in state.get("entity_types_to_drop", []):
                new_entities.append(ent)
        
        relations = current_schema.get("relations", {})
        new_relations = dict()
        for rel_type in relations:
            rels = relations[rel_type]
            new_rels = []
            for rel in rels:
                if rel["type"] not in state.get("relation_types_to_drop", []):
                    new_rels.append(rel)
            new_relations[rel_type] = new_rels
        
        new_schema = {"entities": new_entities, "relations": new_relations}
        best_score = state.get("best_score", 0)
        current_output = {
            "schema": new_schema,
            "settings": {"background": state["background"], "abbreviations": state["abbreviations"]}
        }
        
        if score > best_score:
            best_output = current_output
        else:
            best_output = state.get("best_output", {})
        
 
        # print("[CHECK] 最终建议的数量： ", len(self.feedbacks))
        self.feedbacks.append(state.get("entity_type_distribution",""))
        self.feedbacks.append(state.get("relation_type_distribution",""))
        
        return {
            **state,
            "score": score,
            "reason": reason,
            "retry_count": state["retry_count"] + 1,
            "best_score": max(score, best_score),
            "best_output": best_output
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
        builder.add_node("search_related_experience", self.search_related_experience)
        builder.add_node("generate_schema", self.generate_schema)
        builder.add_node("test_extractions", self.test_extractions)
        builder.add_node("prune_graph_schema", self.prune_graph_schema)
        builder.add_node("reflect_graph_schema", self.reflect_graph_schema)

        builder.set_entry_point("search_related_experience")
        builder.add_edge("search_related_experience", "generate_schema")
        builder.add_edge("generate_schema", "test_extractions")
        builder.add_edge("test_extractions", "prune_graph_schema")
        builder.add_edge("prune_graph_schema", "reflect_graph_schema")
        
        builder.add_conditional_edges("reflect_graph_schema", self._score_check, {
            "good": END,
            "retry": "search_related_experience",
            "giveup": END
        })

        return builder.compile()

    def run(self, params):
        result = self.graph.invoke({
            "documents": params["documents"],
            "schema": params.get("schema", {}),
            "background": params.get("background", ""),
            "abbreviations": params.get("abbreviations", []),
            "retry_count": 0,
            "best_score": 0,
            "best_output": {}
        })
        return result.get("best_output", result)
    

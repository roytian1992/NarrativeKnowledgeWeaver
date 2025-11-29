import json
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
import os
import re
import asyncio
import threading
import traceback
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

from tqdm import tqdm
from langgraph.graph import StateGraph, END

from core.utils.format import correct_json_format
from core.builder.manager.information_manager import InformationExtractor
from core.builder.manager.probing_manager import GraphProber
from core.agent.knowledge_extraction_agent import InformationExtractionAgent
from core.model_providers.openai_rerank import OpenAIRerankModel
from core.utils.prompt_loader import PromptLoader
from ..utils.config import KAGConfig

logger = logging.getLogger(__name__)


class GraphProbingAgent:
    """
    Probing agent that iteratively refines a graph schema (entities & relations)
    using: (i) retrieved 'experience' (insights), (ii) probing prompts, and
    (iii) test extractions + reflection/feedback loops.

    Inputs/Outputs kept identical. Logging in English.
    """

    # ------------------------------ Init ---------------------------------- #
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
        self.feedbacks: List[str] = []
        self.score_threshold = self.config.agent.score_threshold
        self.relation_prune_threshold = self.config.probing.relation_prune_threshold
        self.entity_prune_threshold = self.config.probing.entity_prune_threshold
        self.max_workers = max(int(self.config.probing.max_workers or 1), 1)
        self.max_retries = int(self.config.probing.max_retries or 2)
        self.experience_limit = int(self.config.probing.experience_limit or 6)
        self.probing_mode = self.config.probing.probing_mode
        self.refine_background = bool(self.config.probing.refine_background)

        # 外层超时&退避（用于交给 InformationExtractionAgent）
        self.item_timeout: float = float(getattr(self.config.probing, "item_timeout", 300.0))
        self.backoff_seconds: float = float(getattr(self.config.probing, "backoff_seconds", 30.0))
        self.max_attempts: int = int(getattr(self.config.probing, "max_attempts", 3))

        # 仅用于诊断（默认关闭）
        self.post_diag_join_seconds: float = float(getattr(self.config.probing, "post_diag_join_seconds", 0.5))
        self.enable_thread_diag: bool = bool(getattr(self.config.probing, "enable_thread_diag", False))

        if getattr(self.config.probing, "task_goal", None):
            self.task_goals = "\n".join(self.load_goals(self.config.probing.task_goal))
        else:
            self.task_goals = ""

        # ===== 本次补丁新增：固定常量（不新增配置项）=====
        self.NODE_TIMEOUT_SECONDS = 300.0   # 单节点（如 generate_schema / search_related_experience）总时限
        self.CALL_TIMEOUT_SECONDS = 300.0   # 单次阻塞调用（如 prober.update_*、rerank）时限
        self.CLOSE_TIMEOUT_SECONDS = 5.0    # 清理信息抽取组件的时限

    # ----------------------- Prompt construction -------------------------- #
    def construct_system_prompt(self, background, abbreviations):
        background_info = self.get_background_info(background, abbreviations)
        system_prompt_id = "agent_prompt_screenplay" if self.doc_type == "screenplay" else "agent_prompt_novel"
        system_prompt_text = self.prompt_loader.render_prompt(
            system_prompt_id, {"background_info": background_info}
        )
        return system_prompt_text

    def load_goals(self, path: str):
        text = Path(path).read_text(encoding="utf-8-sig")
        goals = []
        for line in text.splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            s = re.sub(r'^\s*(?:[-•]+|\d+[.)、:：]?)\s*', '', s)
            goals.append(s)
        return list(dict.fromkeys(goals))

    def get_background_info(self, background, abbreviations):
        def _s(val):
            return val.strip() if isinstance(val, str) and val.strip() else None

        bg_block = f"**Background**: {background}\n" if _s(background) else ""

        def fmt(item: dict) -> str:
            if not isinstance(item, dict):
                return ""
            name = _s(item.get("name"))
            desc = _s(item.get("description"))
            abbr = _s(item.get("abbr"))
            full = _s(item.get("full"))
            zh   = _s(item.get("zh"))
            if not desc or (not name and not abbr):
                return ""
            if name and abbr and name.lower() != abbr.lower():
                title = f"{name} ({abbr})"
            else:
                title = name or abbr or "N/A"
            parts = [desc]
            if full:
                parts.append(full)
            if zh:
                parts.append(zh)
            return f"- **{title}**: " + " - ".join(parts) if parts else f"- **{title}**"

        abbr_lines: List[str] = []
        if isinstance(abbreviations, list):
            for it in abbreviations:
                line = fmt(it)
                if line:
                    abbr_lines.append(line)
        abbr_block = "\n".join(abbr_lines)

        if bg_block and abbr_block:
            return f"{bg_block}\nHere are some domain terms and abbreviations:\n{abbr_block}"
        else:
            return bg_block or abbr_block

    def load_schema(self, schema):
        entity_types = schema.get("entities", []) or []
        relation_type_groups = schema.get("relations", {}) or {}
        entity_type_description_text = "\n".join(
            f"- {e.get('type','')}: {e.get('description','')}" for e in entity_types
        )
        relation_type_description_text = "\n".join(
            f"- {r.get('type','')}: {r.get('description','')}"
            for group in relation_type_groups.values()
            for r in (group or [])
        )
        return entity_type_description_text, relation_type_description_text

    # ------------------- Experience search (rerank) ----------------------- #
    def search_related_experience(self, state: Dict[str, Any]) -> Dict[str, Any]:
        logger.info("[search_related_experience] start")
        try:
            k = 2 * self.experience_limit

            # Background / plot / events
            docs_bg = self.reflector.insight_memory.get(
                query="background, story, plot, events, characters; 背景信息、故事、情节、事件、人物",
                k=k
            ) or []
            documents_bg = [doc.page_content for doc in docs_bg]
            reranked_bg = self._run_blocking_with_timeout(
                self.reranker.rerank,
                query="background, story, plot, events; 背景信息、故事、情节、事件",
                documents=documents_bg,
                top_n=min(self.experience_limit, len(documents_bg)),
                timeout=self.CALL_TIMEOUT_SECONDS,
            ) or []
            related_insights_for_background = [
                item["document"]["text"]
                for item in reranked_bg
                if item.get("relevance_score", 0) >= 0.3 and "document" in item and "text" in item["document"]
            ]
            related_insights_for_background += [
                "Entity type 'Plot' does not need to be extracted; it will be constructed later based on extracted 'Event' entities. （情节/Plot这种实体类型不需要抽取，会在后续的任务中基于抽取的事件/Event进行构建。）"
            ]

            # Abbreviations / glossary
            docs_abbr = self.reflector.insight_memory.get(
                query="terms, glossary, abbreviations; 术语、缩写",
                k=k
            ) or []
            documents_abbr = [doc.page_content for doc in docs_abbr]
            reranked_abbr = self._run_blocking_with_timeout(
                self.reranker.rerank,
                query="terms, abbreviations; 术语、缩写",
                documents=documents_abbr,
                top_n=min(self.experience_limit, len(documents_abbr)),
                timeout=self.CALL_TIMEOUT_SECONDS,
            ) or []
            related_insights_for_abbreviations = [
                item["document"]["text"]
                for item in reranked_abbr
                if item.get("relevance_score", 0) >= 0.5
                and "document" in item and "text" in item["document"]
                and ("术语" in item["document"]["text"] or "term" in item["document"]["text"].lower())
            ]

            # Entity schema
            docs_ent = self.reflector.insight_memory.get(
                query="entities, characters, objects, emotions, actions, events, concepts; 人物、物品、情感、动作、事件、概念、实体",
                k=k
            ) or []
            documents_ent = [doc.page_content for doc in docs_ent]
            reranked_ent = self._run_blocking_with_timeout(
                self.reranker.rerank,
                query="entity schema; 实体 schema",
                documents=documents_ent,
                top_n=min(self.experience_limit, len(documents_ent)),
                timeout=self.CALL_TIMEOUT_SECONDS,
            ) or []
            related_insights_for_entity_schema = [
                item["document"]["text"]
                for item in reranked_ent
                if item.get("relevance_score", 0) >= 0.5 and "document" in item and "text" in item["document"]
            ]

            # Relation schema
            docs_rel = self.reflector.insight_memory.get(
                query="events, actions, relations; 事件、动作、关系",
                k=k
            ) or []
            documents_rel = [doc.page_content for doc in docs_rel]
            reranked_rel = self._run_blocking_with_timeout(
                self.reranker.rerank,
                query="relation schema; 关系 schema",
                documents=documents_rel,
                top_n=min(self.experience_limit, len(documents_rel)),
                timeout=self.CALL_TIMEOUT_SECONDS,
            ) or []
            related_insights_for_relation_schema = [
                item["document"]["text"]
                for item in reranked_rel
                if item.get("relevance_score", 0) >= 0.5 and "document" in item and "text" in item["document"]
            ]

            return {
                **state,
                "background_insights": "\n".join(related_insights_for_background),
                "abbreviation_insights": "\n".join(related_insights_for_abbreviations),
                "entity_schema_insights": "\n".join(related_insights_for_entity_schema),
                "relation_schema_insights": "\n".join(related_insights_for_relation_schema),
            }
        except Exception as e:
            logger.exception("search_related_experience failed: %s", e)
            # 不让异常向外冒泡，保持状态机可继续
            return {
                **state,
                "background_insights": state.get("background_insights", ""),
                "abbreviation_insights": state.get("abbreviation_insights", ""),
                "entity_schema_insights": state.get("entity_schema_insights", ""),
                "relation_schema_insights": state.get("relation_schema_insights", ""),
            }

    # ---------------- Schema generation / updates ------------------------- #
    def generate_schema(self, state: Dict[str, Any]) -> Dict[str, Any]:
        logger.info("[generate_schema] start (refine_background=%s)", self.refine_background)
        t0 = time.monotonic()
        try:
            # 0) 当前状态
            current_schema = state.get("schema", {}) or {}
            new_background = state.get("background", "") or ""
            new_abbreviations = list(state.get("abbreviations", []) or [])
            entity_schema_insights = state.get("entity_schema_insights", "") or ""
            relation_schema_insights = state.get("relation_schema_insights", "") or ""
            entity_schema_feedbacks = "\n".join(self.feedbacks)
            relation_schema_feedbacks = entity_schema_feedbacks

            # 1) 背景/缩写（可选，逐步有超时）
            if self.refine_background:
                logger.info("[generate_schema] refine background/abbr ...")

                def _update_bg():
                    txt = state.get("background_insights", "") or ""
                    cur_bg = (state.get("background") or "")
                    cur_bg = "\n".join(state.get("summaries", []) or []) + (("\n" + cur_bg) if cur_bg else "")
                    return self.prober.update_background(text=txt, current_background=cur_bg)

                raw_bg = self._run_blocking_with_timeout(_update_bg, timeout=self.CALL_TIMEOUT_SECONDS)
                if raw_bg:
                    try:
                        result = json.loads(correct_json_format(raw_bg) or "{}")
                        new_background = result.get("background", "") or new_background
                    except Exception:
                        logger.warning("[generate_schema] parse background failed, keep previous")

                def _update_abbr():
                    abbr_txt = state.get("abbreviation_insights", "") or ""
                    bg_info = self.get_background_info(new_background, new_abbreviations)
                    return self.prober.update_abbreviations(text=abbr_txt, current_background=bg_info)

                raw_abbr = self._run_blocking_with_timeout(_update_abbr, timeout=self.CALL_TIMEOUT_SECONDS)
                if raw_abbr:
                    try:
                        result = json.loads(correct_json_format(raw_abbr) or "{}")
                        cand = result.get("abbreviations", []) or []
                        existed = {(it.get("name") or "").strip() for it in new_abbreviations if isinstance(it, dict)}
                        for it in cand:
                            if isinstance(it, dict):
                                name = (it.get("name") or "").strip()
                                if name and name not in existed:
                                    new_abbreviations.append(it)
                                    existed.add(name)
                    except Exception:
                        logger.warning("[generate_schema] parse abbreviations failed, keep previous")
            else:
                logger.info("[generate_schema] skip background/abbr refinement")

            # 2) 实体 schema
            logger.info("[generate_schema] update entity schema ...")

            def _update_entity():
                return self.prober.update_entity_schema(
                    text=entity_schema_insights,
                    feedbacks=entity_schema_feedbacks,
                    current_schema=json.dumps(current_schema.get("entities", {}) or {}, indent=2, ensure_ascii=False),
                    task_goals=self.task_goals
                )

            raw_ent = self._run_blocking_with_timeout(_update_entity, timeout=self.NODE_TIMEOUT_SECONDS)
            if raw_ent:
                try:
                    result = json.loads(correct_json_format(raw_ent) or "{}")
                    new_entity_schema = result.get("entities", []) or current_schema.get("entities", [])
                except Exception:
                    logger.warning("[generate_schema] parse entity schema failed, keep previous")
                    new_entity_schema = current_schema.get("entities", [])
            else:
                logger.warning("[generate_schema] entity schema timeout/error, keep previous")
                new_entity_schema = current_schema.get("entities", [])

            # 3) 关系 schema
            logger.info("[generate_schema] update relation schema ...")

            def _update_relation():
                return self.prober.update_relation_schema(
                    text=relation_schema_insights,
                    feedbacks=relation_schema_feedbacks,
                    current_schema=json.dumps(current_schema.get("relations", {}) or {}, indent=2, ensure_ascii=False),
                    task_goals=self.task_goals
                )

            raw_rel = self._run_blocking_with_timeout(_update_relation, timeout=self.NODE_TIMEOUT_SECONDS)
            if raw_rel:
                try:
                    result = json.loads(correct_json_format(raw_rel) or "{}")
                    new_relation_schema = result.get("relations", []) or current_schema.get("relations", {})
                except Exception:
                    logger.warning("[generate_schema] parse relation schema failed, keep previous")
                    new_relation_schema = current_schema.get("relations", {})
            else:
                logger.warning("[generate_schema] relation schema timeout/error, keep previous")
                new_relation_schema = current_schema.get("relations", {})

            # 4) 清空本轮反馈 & 返回
            self.feedbacks = []
            logger.info("[generate_schema] done in %.2fs", time.monotonic() - t0)
            return {
                **state,
                "schema": {"entities": new_entity_schema, "relations": new_relation_schema},
                "background": new_background,
                "abbreviations": new_abbreviations,
            }
        except Exception as e:
            logger.exception("generate_schema failed: %s", e)
            return state

    # ---------------- Test extractions (with cleanup) --------------------- #
    def test_extractions(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run test extractions using the current schema; compute distributions and drop lists.
        Internally runs async workload with safe loop handling.
        """
        all_documents = state.get("documents") or []
        system_prompt = self.construct_system_prompt(state.get("background", ""), state.get("abbreviations", []))
        schema = state.get("schema") or {}

        try:
            extraction_results = self._run_coro_sync(
                self._test_extractions_and_cleanup(all_documents, system_prompt, schema)
            )
        except Exception as e:
            logger.exception("test_extractions async runner failed: %s", e)
            extraction_results = []

        entity_types: List[str] = []
        relation_types: List[str] = []

        for result in extraction_results:
            try:
                self.feedbacks.extend(result.get("issues", []) or [])
                for entity in result.get("entities", []) or []:
                    entity_types.append(entity.get("type"))
                for relation in result.get("relations", []) or []:
                    relation_types.append(relation.get("relation_type"))
            except Exception:
                logger.warning("Unexpected extraction result format; skipping one result.")

        entity_type_counter = Counter(entity_types)
        relation_type_counter = Counter(relation_types)

        entity_total = len(entity_types)
        relation_total = len(relation_types)

        entity_types_to_drop = [
            key for key, cnt in entity_type_counter.items()
            if entity_total > 0 and (cnt / entity_total) < float(self.entity_prune_threshold)
        ]
        relation_types_to_drop = [
            key for key, cnt in relation_type_counter.items()
            if relation_total > 0 and (cnt / relation_total) < float(self.relation_prune_threshold)
        ]

        entity_type_distribution = ""
        if entity_total > 0:
            for _type, cnt in entity_type_counter.items():
                entity_type_distribution += (
                    f"Entity type {_type}: count = {cnt}, ratio = {round(cnt / entity_total, 3)}\n"
                )

        relation_type_distribution = ""
        if relation_total > 0:
            for _type, cnt in relation_type_counter.items():
                relation_type_distribution += (
                    f"Relation type {_type}: count = {cnt}, ratio = {round(cnt / relation_total, 3)}\n"
                )

        logger.info("Entity types suggested to drop: %s", entity_types_to_drop)
        logger.info("Relation types suggested to drop: %s", relation_types_to_drop)

        if self.enable_thread_diag:
            self._short_join_non_daemon_threads(self.post_diag_join_seconds)

        return {
            **state,
            "entity_type_distribution": entity_type_distribution,
            "relation_type_distribution": relation_type_distribution,
            "entity_types_to_drop": entity_types_to_drop,
            "relation_types_to_drop": relation_types_to_drop,
        }

    # 放在 GraphProbingAgent 类内部，替换原有的 async def _test_extractions_and_cleanup(...)
    async def _test_extractions_and_cleanup(self, all_documents, system_prompt, schema):
        """
        Async part (rewritten with robust, time-boxed cleanup):
        - bounded concurrency with per-item hard timeout
        - realtime tqdm: update(1) on every finished task
        - cancel only tasks we create
        - TIME-BOXED cleanup: component close + pending task teardown
        """

        information_extraction_agent = InformationExtractionAgent(
            self.config, self.llm, system_prompt,
            schema=schema, reflector=self.reflector, mode="probing"
        )

        sem = asyncio.Semaphore(max(int(self.max_workers or 1), 1))
        created_tasks: List[asyncio.Task] = []
        results: List[dict] = []

        async def _arun_once(ch):
            try:
                if not getattr(ch, "content", None) or not (ch.content or "").strip():
                    return {"entities": [], "relations": [], "score": 0, "issues": [], "insights": []}
                return await information_extraction_agent.arun(
                    ch.content,
                    timeout=float(self.item_timeout),
                    max_attempts=int(self.max_attempts),
                    backoff_seconds=float(self.backoff_seconds),
                )
            except Exception as e:
                return {
                    "error": str(e),
                    "entities": [],
                    "relations": [],
                    "score": 0,
                    "issues": [],
                    "insights": [],
                }

        async def _arun_with_timeout(ch):
            """
            执行单个 chunk 抽取任务（带强制超时 + 清理机制）
            """
            async with sem:
                agent_task = asyncio.create_task(_arun_once(ch))
                try:
                    return await asyncio.wait_for(agent_task, timeout=float(self.item_timeout))
                except asyncio.TimeoutError:
                    if not agent_task.done():
                        agent_task.cancel()
                        try:
                            await asyncio.wait_for(agent_task, timeout=5)
                        except Exception:
                            pass
                    logger.warning(f"[TIMEOUT] Item timeout ({self.item_timeout}s): {getattr(ch, 'id', None)}")
                    return {
                        "error": "hard-timeout",
                        "entities": [],
                        "relations": [],
                        "score": 0,
                        "issues": [],
                        "insights": [],
                    }
                except Exception as e:
                    logger.warning(f"[ERROR] Exception in extraction: {repr(e)}")
                    if not agent_task.done():
                        agent_task.cancel()
                        try:
                            await asyncio.wait_for(agent_task, timeout=3)
                        except Exception:
                            pass
                    return {
                        "error": str(e),
                        "entities": [],
                        "relations": [],
                        "score": 0,
                        "issues": [],
                        "insights": [],
                    }

        total_n = len(all_documents)
        pbar = tqdm(total=total_n, desc="Async extraction (probing)")

        async def _run_and_count(ch):
            try:
                return await _arun_with_timeout(ch)
            finally:
                try:
                    pbar.update(1)
                except Exception:
                    pass

        try:
            for ch in all_documents:
                t = asyncio.create_task(_run_and_count(ch))
                created_tasks.append(t)

            for fut in asyncio.as_completed(created_tasks):
                try:
                    r = await fut
                except Exception as e:
                    r = {"error": repr(e), "entities": [], "relations": [], "score": 0, "issues": [], "insights": []}
                results.append(r)

            await asyncio.sleep(0)
            return results

        finally:
            try:
                pbar.close()
            except Exception:
                pass

            try:
                await asyncio.wait_for(self._a_close_component(information_extraction_agent),
                                       timeout=self.CLOSE_TIMEOUT_SECONDS)
            except Exception:
                pass

            try:
                pending = [t for t in created_tasks if not t.done()]
                if pending:
                    for t in pending:
                        t.cancel()
                    try:
                        await asyncio.wait_for(asyncio.gather(*pending, return_exceptions=True),
                                               timeout=self.CLOSE_TIMEOUT_SECONDS)
                    except Exception:
                        pass
            except Exception:
                pass

            try:
                await asyncio.sleep(0)
            except Exception:
                pass

            if self.enable_thread_diag:
                try:
                    self._short_join_non_daemon_threads(timeout=self.post_diag_join_seconds)
                except Exception:
                    pass

    # 新增一个“尽力关闭组件”的异步助手
    @staticmethod
    async def _a_close_component(obj: Optional[object]):
        """
        Close the component itself and its common resource attributes (best-effort, no-throw).
        """
        if obj is None:
            return

        try:
            await GraphProbingAgent._a_best_effort_close(obj)
        except Exception:
            pass

        common_attrs = (
            "neo4j_utils", "driver", "session", "engine",
            "executor", "thread_pool", "pool",
            "vector_store", "graph_store", "client", "db", "conn",
            "http", "transport", "session_http", "aiohttp_session",
        )
        for attr in common_attrs:
            try:
                sub = getattr(obj, attr, None)
                if sub is not None:
                    await GraphProbingAgent._a_best_effort_close(sub)
            except Exception:
                pass

    # ---------------- Pruning and reflection ----------------------------- #
    def prune_graph_schema(self, state: Dict[str, Any]) -> Dict[str, Any]:
        try:
            BATCH_SIZE = 200
            summaries: List[str] = []
            total = (len(self.feedbacks) + BATCH_SIZE - 1) // BATCH_SIZE
            for i in tqdm(range(0, len(self.feedbacks), BATCH_SIZE), desc="Summarizing feedbacks for reflection", total=total):
                batch = self.feedbacks[i:i + BATCH_SIZE]
                all_feedbacks = "\n".join(batch)
                result = self.prober.summarize_feedbacks(context=all_feedbacks, max_items=10)
                parsed = json.loads(correct_json_format(result) or "{}").get("feedbacks", []) or []
                summaries.extend(parsed)

            self.feedbacks = summaries
            logger.info("Number of feedbacks for reflection: %s", len(self.feedbacks))
            entity_type_distribution = state.get("entity_type_distribution", "") or ""
            relation_type_distribution = state.get("relation_type_distribution", "") or ""
            entity_type_description_text, relation_type_description_text = self.load_schema(state.get("schema") or {})

            result = self.prober.prune_schema(
                entity_type_distribution=entity_type_distribution,
                relation_type_distribution=relation_type_distribution,
                entity_type_description_text=entity_type_description_text,
                relation_type_description_text=relation_type_description_text
            )
            result = json.loads(correct_json_format(result) or "{}")
            logger.info("Pruning suggestions: %s", result.get("feedbacks", "N/A"))
            self.feedbacks.extend(result.get("feedbacks", []) or [])
            return state
        except Exception as e:
            logger.exception("prune_graph_schema failed: %s", e)
            return state

    def reflect_graph_schema(self, state: Dict[str, Any]) -> Dict[str, Any]:
        try:
            current_schema = state.get("schema") or {}
            result = self.prober.reflect_schema(
                schema=json.dumps(current_schema, indent=2, ensure_ascii=False),
                feedbacks=self.feedbacks
            )
            result = json.loads(correct_json_format(result) or "{}")
            score = float(result.get("score", 0))
            reason = result.get("reason", "")

            # Remove low-frequency types
            entities = current_schema.get("entities", []) or []
            new_entities = [ent for ent in entities if ent.get("type") not in (state.get("entity_types_to_drop") or [])]

            relations = current_schema.get("relations", {}) or {}
            new_relations = {}
            to_drop = set(state.get("relation_types_to_drop") or [])
            for rel_group, rels in relations.items():
                rels = rels or []
                new_rels = [rel for rel in rels if rel.get("type") not in to_drop]
                new_relations[rel_group] = new_rels

            new_schema = {"entities": new_entities, "relations": new_relations}
            best_score = float(state.get("best_score", 0))
            current_output = {
                "schema": new_schema,
                "settings": {"background": state.get("background", ""), "abbreviations": state.get("abbreviations", [])}
            }

            best_output = current_output if score > best_score else state.get("best_output", {})

            # Keep distributions as feedback
            if state.get("entity_type_distribution", ""):
                self.feedbacks.append(
                    "Entity type distribution in previous round:\n"
                    + state["entity_type_distribution"]
                    + "\nPlease consider those key factors: relevance to the story, frequency of occurrence, diversity of types, and balance between entities."
                )
            if state.get("relation_type_distribution", ""):
                self.feedbacks.append(
                    "Relation type distribution in previous round:\n"
                    + state["relation_type_distribution"]
                    + "\nPlease consider those key factors: relevance to the story, frequency of occurrence, diversity of types, and balance between relations."
                )

            return {
                **state,
                "score": score,
                "reason": reason,
                "retry_count": int(state.get("retry_count", 0)) + 1,
                "best_score": max(score, best_score),
                "best_output": best_output
            }
        except Exception as e:
            logger.exception("reflect_graph_schema failed: %s", e)
            # 失败时也推进 retry_count，避免无限循环
            return {
                **state,
                "score": float(state.get("score", 0)),
                "reason": f"exception: {e}",
                "retry_count": int(state.get("retry_count", 0)) + 1,
                "best_score": float(state.get("best_score", 0)),
                "best_output": state.get("best_output", {})
            }

    # ---------------- Control flow graph (LangGraph) ---------------------- #
    def _score_check(self, state: Dict[str, Any]) -> str:
        try:
            if float(state.get("score", 0)) >= float(self.score_threshold):
                return "good"
            elif int(state.get("retry_count", 0)) >= int(self.max_retries):
                return "giveup"
            else:
                return "retry"
        except Exception:
            if int(state.get("retry_count", 0)) >= int(self.max_retries):
                return "giveup"
            return "retry"

    def _with_node_timeout(self, fn):
        """
        对整个节点再加一层总时限保险丝；超时则透传 state 继续后续节点。
        不新增配置：直接使用 self.NODE_TIMEOUT_SECONDS。
        """
        def wrapped(state: Dict[str, Any]) -> Dict[str, Any]:
            res = self._run_blocking_with_timeout(fn, state, timeout=self.NODE_TIMEOUT_SECONDS)
            if res is None:
                logger.warning("[node-guard] %s timed out, bypassing node", getattr(fn, "__name__", "node"))
                return state
            return res
        return wrapped

    def _build_graph(self):
        builder = StateGraph(dict)
        builder.add_node("search_related_experience", self._with_node_timeout(self.search_related_experience))
        builder.add_node("generate_schema",          self._with_node_timeout(self.generate_schema))
        builder.add_node("test_extractions",         self.test_extractions)
        builder.add_node("prune_graph_schema",       self.prune_graph_schema)
        builder.add_node("reflect_graph_schema",     self.reflect_graph_schema)

        builder.set_entry_point("search_related_experience")
        builder.add_edge("search_related_experience", "generate_schema")
        builder.add_edge("generate_schema", "test_extractions")
        builder.add_edge("test_extractions", "prune_graph_schema")
        builder.add_edge("prune_graph_schema", "reflect_graph_schema")

        builder.add_conditional_edges(
            "reflect_graph_schema", self._score_check, {
                "good": END,
                "retry": "search_related_experience",
                "giveup": END
            }
        )
        return builder.compile()

    # ---------------- Public entry ------------------------------ #
    def run(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the probing loop with the given params. Returns the best output
        (schema + settings) if available; otherwise, returns the final state.
        """
        result = self.graph.invoke({
            "documents": params["documents"],
            "schema": params.get("schema", {}),
            "background": params.get("background", ""),
            "abbreviations": params.get("abbreviations", []),
            "summaries": params.get("summaries", []),
            "retry_count": 0,
            "best_score": 0,
            "best_output": {}
        })
        return result.get("best_output", result)

    # ===================== Cleanup helpers ================================ #
    @staticmethod
    async def _a_best_effort_close(obj: Optional[object]):
        if obj is None:
            return
        # Async closers
        for name in ("aclose", "async_close", "close_async", "shutdown"):
            m = getattr(obj, name, None)
            if callable(m):
                try:
                    res = m()
                    if asyncio.iscoroutine(res):
                        await res
                    return
                except Exception:
                    pass
        # Sync fallbacks
        for name in ("close", "dispose", "terminate", "stop", "shutdown"):
            m = getattr(obj, name, None)
            if callable(m):
                try:
                    m()
                    return
                except Exception:
                    pass
        # nested attributes
        for attr in ("client", "http", "transport", "session", "pool", "executor", "vector_store"):
            try:
                sub = getattr(obj, attr, None)
                if sub is not None:
                    await GraphProbingAgent._a_best_effort_close(sub)
            except Exception:
                pass

    @staticmethod
    def _short_join_non_daemon_threads(timeout: float = 0.5):
        try:
            import sys
            frames = sys._current_frames()
        except Exception:
            frames = {}
        for t in threading.enumerate():
            if t is threading.main_thread() or t.daemon:
                continue
            try:
                if frames:
                    frame = frames.get(t.ident)
                    if frame:
                        stack_str = "".join(traceback.format_stack(frame))
                        logger.warning("[diag] Non-daemon thread alive: name=%s, ident=%s\nStack:\n%s",
                                       t.name, t.ident, stack_str)
                t.join(timeout=timeout)
            except Exception:
                pass

    # ---------------- Safe sync runner for async coroutines --------------- #
    @staticmethod
    def _run_coro_sync(coro):
        """
        Run a coroutine safely from sync context.
        - If no running loop: use asyncio.run
        - If already in a loop (e.g., Jupyter/LangGraph): start a new loop in a new thread
          and run the coro to completion, returning its result.
        """
        try:
            loop = asyncio.get_running_loop()
            result_holder = {"result": None, "exc": None}
            def _runner():
                try:
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    result_holder["result"] = new_loop.run_until_complete(coro)
                except Exception as e:
                    result_holder["exc"] = e
                finally:
                    try:
                        new_loop.stop()
                    except Exception:
                        pass
                    try:
                        new_loop.close()
                    except Exception:
                        pass
            th = threading.Thread(target=_runner, daemon=True)
            th.start()
            th.join()
            if result_holder["exc"] is not None:
                raise result_holder["exc"]
            return result_holder["result"]
        except RuntimeError:
            return asyncio.run(coro)

    # ---------------- Blocking-call timeout wrapper ----------------------- #
    def _run_blocking_with_timeout(self, func, /, *args, timeout: Optional[float] = None, **kwargs):
        """
        在独立线程里执行阻塞函数；到达 timeout 立即放弃并继续主流程。
        关键：不要用 `with ThreadPoolExecutor(...)`，否则会在退出时 wait=True 卡死。
        """
        to = float(timeout or self.CALL_TIMEOUT_SECONDS)
        start = time.monotonic()
        func_name = getattr(func, "__name__", str(func))
        logger.debug("[timeout-wrapper] begin %s (timeout=%.1fs)", func_name, to)

        ex: Optional[ThreadPoolExecutor] = None
        fut = None
        try:
            ex = ThreadPoolExecutor(max_workers=1, thread_name_prefix="node-call")
            fut = ex.submit(func, *args, **kwargs)
            res = fut.result(timeout=to)
            logger.debug("[timeout-wrapper] done %s (%.2fs)", func_name, time.monotonic() - start)
            return res

        except FuturesTimeout:
            logger.warning("[timeout-wrapper] TIMEOUT in %s after %.1fs", func_name, time.monotonic() - start)
            # 尝试取消 future（若尚未开始）
            if fut is not None:
                try:
                    fut.cancel()
                except Exception:
                    pass
            return None

        except Exception as e:
            logger.warning("[timeout-wrapper] ERROR in %s: %r", func_name, e)
            return None

        finally:
            if ex is not None:
                # 关键：不等待后台线程结束，直接让其作为后台线程自行退出
                try:
                    # Python 3.9+ 支持 cancel_futures；低版本忽略该参数
                    ex.shutdown(wait=False, cancel_futures=True)  # type: ignore[arg-type]
                except TypeError:
                    ex.shutdown(wait=False)

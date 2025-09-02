# kag/builder/database_builder.py

import os
import json
import sqlite3
import asyncio
import time
from typing import List, Dict, Any, Tuple
import pandas as pd
from tqdm import tqdm

from core.utils.config import KAGConfig
from core.model_providers.openai_llm import OpenAILLM
from core.utils.prompt_loader import PromptLoader
from core.agent.cmp_extraction_agent import CMPExtractionAgent


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 统一去空格（只处理会用到的列）
    norm_cols = ['名称','类别','相关角色','chunk_id','场次','子类别','外观','状态','文中线索','补充信息','子场次']
    for c in norm_cols:
        if c in df.columns:
            df[c] = df[c].astype('string').str.strip()

    # 1) 删除名称缺失/空白的行
    df = df[ df['名称'].notna() & (df['名称'] != '') ]

    # 2) 类别=propitem -> prop（不区分大小写）
    if '类别' in df.columns:
        df.loc[df['类别'].str.lower() == 'propitem', '类别'] = 'prop'

    # 3) 按关键列去重（保留第一条）
    keys = [c for c in ['名称','类别','相关角色','chunk_id','场次'] if c in df.columns]
    df = df.drop_duplicates(subset=keys, keep='first').reset_index(drop=True)

    return df



class RelationalDatabaseBuilder:
    """
    读取 all_document_chunks.json -> 服化道抽取 -> 写入 SQLite
    支持失败管理 + 多轮重试（默认2轮：首轮+1次重试）。
    """

    def __init__(self, config: KAGConfig, max_retries: int = 2):
        self.config = config
        self.llm = OpenAILLM(config)
        self.max_retries = max_retries  # 可配置最大尝试次数，包含首轮
        self.prompt_loader = PromptLoader(self.config.knowledge_graph_builder.prompt_dir)
        self.system_prompt = self._init_system_prompt()
        self.agent = CMPExtractionAgent(config, self.llm, self.system_prompt)

    # ---------------- system prompt ----------------
    def _init_system_prompt(self) -> str:
        base = self.config.storage.graph_schema_path
        settings_path = os.path.join(base, "settings.json")
        if os.path.exists(settings_path):
            settings = json.load(open(settings_path, "r", encoding="utf-8"))
        elif os.path.exists(self.config.probing.default_background_path):
            settings = json.load(open(self.config.probing.default_background_path, "r", encoding="utf-8"))
        else:
            settings = {"background": "", "abbreviations": []}

        background_info = self.get_background_info(
            background=settings.get("background", ""),
            abbreviations=settings.get("abbreviations", []),
        )
        # doc_type = self.config.knowledge_graph_builder.doc_type
        system_prompt_id = "agent_prompt_cmp" 
        return self.prompt_loader.render_prompt(system_prompt_id, {"background_info": background_info})

    def get_background_info(self, background: str, abbreviations: List[dict]) -> str:
        bg_block = f"**背景设定**：{background}\n" if background else ""

        def fmt(item: dict) -> str:
            if not isinstance(item, dict):
                return ""
            abbr = (
                item.get("abbr")
                or item.get("full")
                or next((v for k, v in item.items() if isinstance(v, str) and v.strip()), "N/A")
            )
            parts = [v.strip() for k, v in item.items() if k not in ("abbr", "full") and isinstance(v, str) and v.strip()]
            return f"- **{abbr}**: " + " - ".join(parts) if parts else f"- **{abbr}**"

        abbr_block = "\n".join(fmt(x) for x in abbreviations if isinstance(x, dict))
        return f"{bg_block}\n{abbr_block}" if (background and abbr_block) else (bg_block or abbr_block)

    # ---------------- 服化道抽取 ----------------
    def _rows_from_result(self, chunk: Dict[str, Any], merged_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        md = chunk.get("metadata", {}) or {}
        rows = []
        for r in merged_results:
            rows.append({
                "name": r.get("name", ""),
                "category": r.get("category", ""),
                "subcategory": r.get("subcategory", ""),
                "appearance": r.get("appearance", ""),
                "status": r.get("status", ""),
                "character": r.get("character", ""),
                "evidence": r.get("evidence", ""),
                "notes": r.get("notes", ""),
                "chunk_id": chunk.get("id", ""),
                "title": md.get("title", ""),
                "subtitle": md.get("subtitle", ""),
                "scene_id": md.get("scene_id", "")
            })
        return rows

    async def _extract_chunk(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        content = (chunk.get("content") or "").strip()
        if not content:
            return {"chunk": chunk, "rows": [], "error": None}
        try:
            result = await self.agent.arun(content, timeout=self.config.agent.async_timeout)
            merged = result.get("results", []) if isinstance(result, dict) else []
            return {"chunk": chunk, "rows": self._rows_from_result(chunk, merged), "error": None}
        except Exception as e:
            return {"chunk": chunk, "rows": [], "error": f"{e.__class__.__name__}: {e}"}

    async def _gather_once(self, chunks: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        sem = asyncio.Semaphore(self.config.knowledge_graph_builder.max_workers)

        async def _guarded(ch):
            async with sem:
                return await self._extract_chunk(ch)

        rows, failures = [], []
        for coro in tqdm(asyncio.as_completed([_guarded(ch) for ch in chunks]), total=len(chunks), desc="服化道抽取中"):
            res = await coro
            if res["error"]:
                failures.append({"chunk": res["chunk"], "error": res["error"]})
            else:
                rows.extend(res["rows"])
        return rows, failures

    async def _gather_with_retries(self, chunks: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        all_rows, failures = [], []
        current_chunks = chunks

        for attempt in range(1, self.max_retries + 1):
            rows, failures = await self._gather_once(current_chunks)
            all_rows.extend(rows)
            if not failures:
                break
            if attempt < self.max_retries:
                backoff = getattr(self.config.agent, "async_backoff_seconds", 2)
                print(f"🔄 第 {attempt} 轮后仍有 {len(failures)} 个失败，等待 {backoff}s 后重试...")
                await asyncio.sleep(backoff)
                current_chunks = [f["chunk"] for f in failures]

        return all_rows, failures

    # ---------------- 主流程 ----------------
    def extract_cmp_information(self):
        base = self.config.storage.knowledge_graph_path
        input_json_path = os.path.join(base, "all_document_chunks.json")

        with open(input_json_path, "r", encoding="utf-8") as fr:
            chunks = json.load(fr)

        all_rows, still_failed = asyncio.run(self._gather_with_retries(chunks))
        os.makedirs(self.config.storage.sql_database_path, exist_ok=True)
        with open(os.path.join(self.config.storage.sql_database_path, "extraction_results.json"), "w") as f:
            json.dump(all_rows, f, ensure_ascii=False, indent=2)

        print(f"✅ 服化道抽取完成: 成功 {len(all_rows)} 行, 失败 {len(still_failed)} 行 (最大尝试 {self.max_retries} 轮)")

    def build_relational_database(self):
        with open(os.path.join(self.config.storage.sql_database_path, "extraction_results.json"), "r") as f:
            all_rows = json.load(f)
        df_cmp = pd.DataFrame(all_rows)

        # scene_id_list = []
        # for i in range(df_cmp.shape[0]):
        #     row = df_cmp.iloc[0]
        #     if row["subtitle"]:
        #         scene_id = row["subtitle"].split("、")[0]
        #     else:
        #         scene_id = row["title"].split("、")[0]
        #     scene_id_list.append(scene_id)
        # df_cmp["scene_id"] = scene_id_list
        
        df_cmp = df_cmp.rename(columns={
            "name": "名称",
            "category": "类别",
            "subcategory": "子类别",
            "appearance": "外观",
            "status": "状态",
            "character": "相关角色",
            "evidence": "文中线索",
            "chunk_id": "chunk_id",
            "scene_id": "场次",
            "title": "场次名",
            "subtitle": "子场次名",
            "notes": "补充信息"
        })

        db_path = os.path.join(self.config.storage.sql_database_path, "CMP.db")
        os.makedirs(self.config.storage.sql_database_path, exist_ok=True)
        if os.path.exists(db_path):
            os.remove(db_path)
        conn = sqlite3.connect(db_path)
        df_cmp = clean_df(df_cmp)
        df_cmp.to_sql("CMP_info", conn, if_exists="replace", index=False)
        conn.commit()
        print(f"✅ 构建SQL数据库成功: {db_path}")
        df_cmp.to_csv(os.path.join(self.config.storage.sql_database_path, "CMP_info.csv"), index=False)
        conn.close()

    def build_scene_info(self):
        db_path = os.path.join(self.config.storage.sql_database_path, "CMP.db")
        os.makedirs(self.config.storage.sql_database_path, exist_ok=True)
        if os.path.exists(db_path):
            os.remove(db_path)
        conn = sqlite3.connect(db_path)

        rows = []
        base = self.config.storage.knowledge_graph_path
        input_json_path = os.path.join(base, "all_document_chunks.json")
        with open(input_json_path, "r", encoding="utf-8") as fr:
            chunks = json.load(fr)

        for chunk in chunks:
            metadata = chunk.get("metadata", {})
            rows.append({
                "chunk_id": chunk.get("id"),
                "scene_id": metadata.get("scene_id", metadata.get("title").split("、")[0]),
                "title": metadata.get("title"),
                "subtitle": metadata.get("subtitle")
            })

        df_scene = pd.DataFrame(rows)
        df_scene = df_scene.rename(columns={
            "chunk_id": "chunk_id",
            "scene_id": "场次",
            "title": "场次名",
            "subtitle": "子场次名",
        }).drop_duplicates()
        df_scene.to_sql("Scene_info", conn, if_exists="replace", index=False)
        conn.commit()
        df_scene.to_csv(os.path.join(self.config.storage.sql_database_path, "Scene_info.csv"), index=False)
        conn.close()

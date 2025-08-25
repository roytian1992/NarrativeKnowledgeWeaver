# -*- coding: utf-8 -*-
from __future__ import annotations

"""
sqldb_tools.py
- 提供基于 SQLite 的 CMP_info 表查询工具，并注册为 Qwen Agent 可调用工具。
- 工具：
  1) search_by_character  按“相关角色”模糊检索完整记录
  2) search_by_scene      按“场次名/子场次名”检索完整记录
  3) chunk_to_scene       由 chunk_id 映射到 场次名/子场次名
  4) scene_to_chunks      由场次（可选子场次）列出所有 chunk_id
"""

from typing import List, Dict, Any, Optional, Tuple, Union
import json
import sqlite3

from qwen_agent.tools.base import BaseTool, register_tool
from qwen_agent.utils.utils import logger
from langchain_community.utilities import SQLDatabase
from langchain_experimental.sql.base import SQLDatabaseSequentialChain

# 如果你已有 KAGConfig / OpenAILLM，可在 __init__ 里按需切换
from core.utils.config import KAGConfig
from core.model_providers.openai_llm import OpenAILLM

# —— 列清单（与表结构一致） —— #
COLUMNS = [
    "名称", "类别", "子类别", "外观", "状态", "相关角色",
    "文中线索", "补充信息", "chunk_id", "场次", "场次名", "子场次名"
]


def build_cols_sql(conn, table: str, candidate_cols: list[str]) -> str:
    cur = conn.cursor()
    cur.execute(f"PRAGMA table_info('{table}')")
    cols_in_db = {row[1] for row in cur.fetchall()}
    valid_cols = [c for c in candidate_cols if c in cols_in_db]
    return ", ".join([f'"{c}"' for c in valid_cols])


def _parse_sql_from_steps(steps):
    if not steps:
        return None
    for st in steps:
        if isinstance(st, dict) and "sql_cmd" in st:
            return st["sql_cmd"].strip()
        if isinstance(st, str) and st.strip().lower().startswith("select"):
            return st.strip()
    return None


def _format_sql_rows_text(raw):
    if raw is None:
        return "未查询到结果。"
    if isinstance(raw, str):
        try:
            data = ast.literal_eval(raw)
            if isinstance(data, (list, tuple)) and data and isinstance(data[0], (list, tuple)):
                lines = []
                for row in data:
                    items = [str(x) for x in row]
                    lines.append("- " + "；".join(items))
                return "查询结果如下：\n" + "\n".join(lines)
            return raw
        except Exception:
            return raw
    if isinstance(raw, (list, tuple)):
        if not raw:
            return "未查询到结果。"
        lines = []
        for row in raw:
            if isinstance(row, (list, tuple)):
                items = [str(x) for x in row]
                lines.append("- " + "；".join(items))
            else:
                lines.append("- " + str(row))
        return "查询结果如下：\n" + "\n".join(lines)
    return str(raw)


# —— 通用格式化：把查询结果转为自然语言 —— #
def _fmt_kv_line(row: Dict[str, Any], columns: List[str]) -> str:
    items = []
    for c in columns:
        if c in row:
            v = row[c]
            v = "" if v is None else str(v).strip()
            if v != "":
                items.append(f"{c}：{v}")
    return "；".join(items)

def format_rows_dicts_to_nl(rows: List[Dict[str, Any]],
                            columns: Optional[List[str]] = None,
                            dedup: bool = True,
                            header: Optional[str] = "查询结果如下：") -> str:
    if not rows:
        return "未查询到结果。"
    if columns is None:
        # 以 COLUMNS 为主序，再补其它键
        appeared = set()
        ordered = []
        for c in COLUMNS:
            if any(c in r for r in rows):
                ordered.append(c); appeared.add(c)
        for k in rows[0].keys():
            if k not in appeared:
                ordered.append(k); appeared.add(k)
        columns = ordered

    seen: set[Tuple] = set()
    lines: List[str] = []
    for r in rows:
        line = _fmt_kv_line(r, columns)
        if not line:
            continue
        key = tuple((k, r.get(k, None)) for k in columns)
        if dedup and key in seen:
            continue
        seen.add(key)
        lines.append(f"- {line}")

    if not lines:
        return "未查询到结果。"
    if header:
        return header + "\n" + "\n".join(lines)
    return "\n".join(lines)

def format_query_result(
    data: Union[List[Dict[str, Any]], None],
    columns: Optional[List[str]] = None,
    header: Optional[str] = "查询结果如下：",
) -> str:
    if data is None:
        return "未查询到结果。"
    if isinstance(data, list) and (not data or isinstance(data[0], dict)):
        return format_rows_dicts_to_nl(data, columns=columns, header=header)
    return "（无法识别的结果类型，未能格式化。）"

def format_mapping_chunk_to_scene(mapping: Optional[Dict[str, Any]]) -> str:
    """
    专用于 chunk_to_scene：期望字段 chunk_id / 场次名 / 子场次名
    """
    if not mapping:
        return "未找到该 chunk_id 对应的场次信息。"
    chunk = mapping.get("chunk_id", "")
    scene = mapping.get("场次名", "")
    sub = mapping.get("子场次名", "")
    tail = f"；子场次名：{sub}" if sub else ""
    return f"chunk_id：{chunk} 对应 场次名：{scene}{tail}"

def format_scene_to_chunks(scene_name: str,
                           chunks: List[str],
                           subscene_name: Optional[str] = None) -> str:
    """
    专用于 scene_to_chunks 的输出
    """
    if not chunks:
        if subscene_name:
            return f"未在场次「{scene_name}」-「{subscene_name}」下找到任何 chunk_id。"
        return f"未在场次「{scene_name}」下找到任何 chunk_id。"
    head = f"场次「{scene_name}」" + (f" - 「{subscene_name}」" if subscene_name else "")
    lines = "\n".join(f"- {cid}" for cid in chunks)
    return f"{head} 共 {len(chunks)} 个 chunk_id：\n{lines}"


# —— SQLite 基础 —— #
def get_conn(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

def rows_to_dicts(rows: List[sqlite3.Row]) -> List[Dict[str, Any]]:
    return [dict(row) for row in rows]

def ensure_indices(conn: sqlite3.Connection, table: str) -> None:
    cur = conn.cursor()
    cur.execute(f'CREATE INDEX IF NOT EXISTS idx_{table}_related_role ON "{table}"("相关角色");')
    cur.execute(f'CREATE INDEX IF NOT EXISTS idx_{table}_scene        ON "{table}"("场次名");')
    cur.execute(f'CREATE INDEX IF NOT EXISTS idx_{table}_subscene     ON "{table}"("子场次名");')
    cur.execute(f'CREATE INDEX IF NOT EXISTS idx_{table}_chunk        ON "{table}"("chunk_id");')
    conn.commit()


# —— 1) 按角色名模糊检索 —— #
@register_tool("search_by_character")
class Search_By_Character(BaseTool):
    """
    在 SQLite 表的“相关角色”列进行不区分大小写的模糊匹配，返回整行信息。
    """
    name = "search_by_character"
    description = "在 CMP_info 表中按“相关角色”进行模糊检索，返回匹配到的完整记录。"
    parameters = [
        {
            "name": "query",
            "type": "string",
            "description": "角色名关键词（模糊匹配，大小写不敏感）",
            "required": True
        },
        {
            "name": "limit",
            "type": "integer",
            "description": "最多返回的记录条数（可选）",
            "required": False
        }
    ]

    def __init__(self, db_path: str,
                 default_table: str = "CMP_info",
                 build_indices: bool = False):
        self.db_path = db_path
        self.default_table = default_table
        if build_indices:
            conn = get_conn(self.db_path)
            try:
                ensure_indices(conn, self.default_table)
            finally:
                conn.close()
        self.COLS_SQL = build_cols_sql(get_conn(db_path), "CMP_info", COLUMNS)


    def _search_by_character(self, character_keyword: str,
                             limit: Optional[int] = None) -> List[Dict[str, Any]]:
        
        sql = f'''
            SELECT {self.COLS_SQL}
            FROM "{self.default_table}"
            WHERE "相关角色" LIKE ? COLLATE NOCASE
            ORDER BY "场次名","子场次名","chunk_id"
        '''
        params: List[Any] = [f"%{character_keyword}%"]
        if limit is not None:
            sql += " LIMIT ?"
            params.append(int(limit))

        conn = get_conn(self.db_path)
        try:
            cur = conn.cursor()
            cur.execute(sql, tuple(params))
            return rows_to_dicts(cur.fetchall())
        finally:
            conn.close()

    def call(self, params: str, **kwargs) -> str:
        logger.info("🔎 search_by_character: 开始执行 SQL 模糊检索（相关角色）")
        p: Dict[str, Any] = json.loads(params)
        query = str(p.get("query", "")).strip()
        limit = p.get("limit", None)
        if isinstance(limit, str) and limit.isdigit():
            limit = int(limit)
        elif not isinstance(limit, (int, type(None))):
            limit = None

        rows = self._search_by_character(query, limit=limit)
        return format_query_result(rows, header=f"与「{query}」相关的记录：")


# —— 2) 按场次/子场次检索 —— #
@register_tool("search_by_scene")
class Search_By_Scene(BaseTool):
    """
    根据场次（可选子场次）返回所有相关信息；支持模糊/精确匹配。
    """
    name = "search_by_scene"
    description = "在指定表中按“场次名”（可选“子场次名”）检索并返回完整记录，支持模糊匹配。"
    parameters = [
        {
            "name": "scene_name",
            "type": "string",
            "description": "场次名关键词（模糊或精确匹配的关键字段）",
            "required": True
        },
        {
            "name": "subscene_name",
            "type": "string",
            "description": "子场次名关键词（可选；与场次名共同过滤）",
            "required": False
        },
        {
            "name": "fuzzy",
            "type": "boolean",
            "description": "是否使用模糊匹配（LIKE）。默认 true。",
            "required": False
        },
        {
            "name": "limit",
            "type": "integer",
            "description": "最多返回的记录条数（可选）",
            "required": False
        }
    ]

    def __init__(self, db_path: str,
                 default_table: str = "CMP_info",
                 build_indices: bool = False):
        self.db_path = db_path
        self.default_table = default_table
        if build_indices:
            conn = get_conn(self.db_path)
            try:
                ensure_indices(conn, self.default_table)
            finally:
                conn.close()
        self.COLS_SQL = build_cols_sql(get_conn(db_path), "CMP_info", COLUMNS)

    @staticmethod
    def _build_where(scene_name: str,
                     subscene_name: Optional[str],
                     fuzzy: bool) -> tuple[str, list[Any]]:
        where = []
        params: list[Any] = []
        if fuzzy:
            where.append('"场次名" LIKE ? COLLATE NOCASE')
            params.append(f"%{scene_name}%")
        else:
            where.append('"场次名" = ?')
            params.append(scene_name)

        if subscene_name:
            if fuzzy:
                where.append('"子场次名" LIKE ? COLLATE NOCASE')
                params.append(f"%{subscene_name}%")
            else:
                where.append('"子场次名" = ?')
                params.append(subscene_name)

        return " AND ".join(where) if where else "1=1", params

    def _search_by_scene(self, scene_name: str,
                         subscene_name: Optional[str] = None,
                         fuzzy: bool = True,
                         limit: Optional[int] = None) -> List[Dict[str, Any]]:
        where_sql, params = self._build_where(scene_name, subscene_name, fuzzy)
        sql = f'''
            SELECT {self.COLS_SQL}
            FROM "{self.default_table}"
            WHERE {where_sql}
            ORDER BY "场次名","子场次名","chunk_id"
        '''
        if limit is not None:
            sql += " LIMIT ?"
            params.append(int(limit))

        conn = get_conn(self.db_path)
        try:
            cur = conn.cursor()
            cur.execute(sql, tuple(params))
            return rows_to_dicts(cur.fetchall())
        finally:
            conn.close()

    def call(self, params: str, **kwargs) -> str:
        logger.info("🎬 search_by_scene: 按场次检索完整记录")
        p: Dict[str, Any] = json.loads(params)
        scene_name = str(p.get("scene_name", "")).strip()
        subscene_name = (str(p["subscene_name"]).strip()
                         if p.get("subscene_name") not in (None, "") else None)
        fuzzy = True if p.get("fuzzy", True) else False
        limit = p.get("limit", None)
        if isinstance(limit, str) and limit.isdigit():
            limit = int(limit)
        elif not isinstance(limit, (int, type(None))):
            limit = None

        rows = self._search_by_scene(scene_name, subscene_name=subscene_name,
                                     fuzzy=fuzzy, limit=limit)
        head = f"场次「{scene_name}」" + (f" - 「{subscene_name}」" if subscene_name else "")
        return format_query_result(rows, header=f"{head} 的相关记录：")


# —— 3) chunk_id → 场次/子场次 —— #
@register_tool("chunk_to_scene")
class Chunk_To_Scene(BaseTool):
    """
    从 chunk_id 映射到场次：返回 chunk_id、场次名、子场次名。
    """
    name = "chunk_to_scene"
    description = "给定 chunk_id，查询其对应的 场次名 / 子场次名 映射。"
    parameters = [
        {
            "name": "chunk_id",
            "type": "string",
            "description": "要查询的 chunk_id（精确匹配）",
            "required": True
        }
    ]

    def __init__(self, db_path: str,
                 default_table: str = "CMP_info",
                 build_indices: bool = False):
        self.db_path = db_path
        self.default_table = default_table
        if build_indices:
            conn = get_conn(self.db_path)
            try:
                ensure_indices(conn, self.default_table)
            finally:
                conn.close()

    def _chunk_to_scene(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        sql = f'''
            SELECT "chunk_id","场次名","子场次名"
            FROM "{self.default_table}"
            WHERE "chunk_id" = ?
            LIMIT 1
        '''
        conn = get_conn(self.db_path)
        try:
            cur = conn.cursor()
            cur.execute(sql, (chunk_id,))
            row = cur.fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def call(self, params: str, **kwargs) -> str:
        logger.info("🧭 chunk_to_scene: 由 chunk_id 映射到场次")
        p: Dict[str, Any] = json.loads(params)
        chunk_id = str(p.get("chunk_id", "")).strip()
        mapping = self._chunk_to_scene(chunk_id)
        return format_mapping_chunk_to_scene(mapping)


# —— 4) 场次/子场次 → chunk_id 列表 —— #
@register_tool("scene_to_chunks")
class Scene_To_Chunks(BaseTool):
    """
    从场次（可选子场次）映射到所有 chunk_id（去重、排序）。
    """
    name = "scene_to_chunks"
    description = "给定场次名（可选子场次名），列出该范围内所有 chunk_id。支持模糊匹配。"
    parameters = [
        {
            "name": "scene_name",
            "type": "string",
            "description": "场次名关键词（模糊或精确匹配的关键字段）",
            "required": True
        },
        {
            "name": "subscene_name",
            "type": "string",
            "description": "子场次名关键词（可选）",
            "required": False
        },
        {
            "name": "fuzzy",
            "type": "boolean",
            "description": "是否使用模糊匹配（LIKE）。默认 true。",
            "required": False
        }
    ]

    def __init__(self, db_path: str,
                 default_table: str = "CMP_info",
                 build_indices: bool = False):
        self.db_path = db_path
        self.default_table = default_table
        if build_indices:
            conn = get_conn(self.db_path)
            try:
                ensure_indices(conn, self.default_table)
            finally:
                conn.close()

    @staticmethod
    def _build_where(scene_name: str,
                     subscene_name: Optional[str],
                     fuzzy: bool) -> tuple[str, list[Any]]:
        where = []
        params: list[Any] = []
        if fuzzy:
            where.append('"场次名" LIKE ? COLLATE NOCASE')
            params.append(f"%{scene_name}%")
        else:
            where.append('"场次名" = ?')
            params.append(scene_name)

        if subscene_name:
            if fuzzy:
                where.append('"子场次名" LIKE ? COLLATE NOCASE')
                params.append(f"%{subscene_name}%")
            else:
                where.append('"子场次名" = ?')
                params.append(subscene_name)

        return " AND ".join(where) if where else "1=1", params

    def _scene_to_chunks(self, scene_name: str,
                         subscene_name: Optional[str] = None,
                         fuzzy: bool = True) -> List[str]:
        where_sql, params = self._build_where(scene_name, subscene_name, fuzzy)
        sql = f'''
            SELECT DISTINCT "chunk_id"
            FROM "{self.default_table}"
            WHERE {where_sql}
            ORDER BY "chunk_id"
        '''
        conn = get_conn(self.db_path)
        try:
            cur = conn.cursor()
            cur.execute(sql, tuple(params))
            return [r["chunk_id"] for r in cur.fetchall()]
        finally:
            conn.close()

    def call(self, params: str, **kwargs) -> str:
        logger.info("🧩 scene_to_chunks: 由场次映射到 chunk 列表")
        p: Dict[str, Any] = json.loads(params)
        scene_name = str(p.get("scene_name", "")).strip()
        subscene_name = (str(p["subscene_name"]).strip()
                         if p.get("subscene_name") not in (None, "") else None)
        fuzzy = True if p.get("fuzzy", True) else False

        chunks = self._scene_to_chunks(scene_name, subscene_name=subscene_name, fuzzy=fuzzy)
        return format_scene_to_chunks(scene_name, chunks, subscene_name=subscene_name)


@register_tool("nlp2sql_query")
class NLP2SQL_Query(BaseTool):
    """
    用自然语言查询 SQLite 数据库（CMP.db），自动生成 SQL 并执行。
    """

    name = "nlp2sql_query"
    description = f"用自然语言提问服饰、化妆、道具数据库，自动生成并执行 SQL，返回自然语言结果（可选附带 SQL）。该数据库的列名column names有：{COLUMNS}"
    parameters = [
        {
            "name": "query",
            "type": "string",
            "description": "自然语言查询文本",
            "required": True
        },
        {
            "name": "return_sql",
            "type": "boolean",
            "description": "是否在结果中附带生成的 SQL 语句，默认 false",
            "required": False
        }
    ]

    def __init__(self,
                 db_path, llm, instruction="在处理查询时，不要只做精确匹配，应同时考虑字符串包含、同义词扩展或语义相似度等方式来覆盖相关结果。"):
        self.db_path = db_path
        self.db = SQLDatabase.from_uri(f"sqlite:///{self.db_path}")
        self.llm = llm
        self.instruction = instruction

        # 初始化 SQL chain（固定参数）
        self.chain = SQLDatabaseSequentialChain.from_llm(
            llm=self.llm,
            db=self.db,
            verbose=False,
            return_direct=True,
            return_intermediate_steps=True,
            top_k=10000,
            use_query_checker=True
        )

    def call(self, params: str, **kwargs) -> str:
        logger.info("🧠 nl2psql_query: 自然语言转 SQL + 执行")
        p = json.loads(params)
        query = str(p.get("query", "")).strip()

        if not query:
            return "请提供 query。"
        if self.instruction:
            query += "\n" + self.instruction

        return_sql = bool(p.get("return_sql", False))

        try:
            out = self.chain.invoke(query)
            result = out.get("result", "")
            steps = out.get("intermediate_steps", None)
            sql_cmd = _parse_sql_from_steps(steps)

            nl = _format_sql_rows_text(result)
            parts = [f"◼︎ 自然语言结果：\n{nl}"]
            if return_sql and sql_cmd:
                parts.append(f"◼︎ 生成的 SQL：\n```sql\n{sql_cmd}\n```")
            return "\n\n".join(parts)
        except Exception as e:
            logger.exception("nl2sql_query 执行失败")
            return f"查询执行失败：{type(e).__name__}: {e}"
        


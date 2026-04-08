# -*- coding: utf-8 -*-
from __future__ import annotations

"""
sqldb_tools.py

Interaction-oriented SQLite tools.
Target table: Interaction_info
"""

from typing import List, Dict, Any, Tuple
import json
import sqlite3

from qwen_agent.tools.base import BaseTool, register_tool
from qwen_agent.utils.utils import logger
from core.utils.format import DOC_TYPE_META


def _qident(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def _like(v: str) -> str:
    return f"%{v}%"


def _fmt_row(row: Dict[str, Any], cols: List[str]) -> str:
    parts: List[str] = []
    for c in cols:
        if c not in row:
            continue
        v = row.get(c)
        if v is None:
            continue
        s = str(v).strip()
        if not s:
            continue
        parts.append(f"{c}: {s}")
    return " | ".join(parts)


def _fmt_rows(rows: List[Dict[str, Any]], cols: List[str], *, header: str) -> str:
    if not rows:
        return "未查询到结果。"
    lines = []
    for r in rows:
        line = _fmt_row(r, cols)
        if line:
            lines.append(f"- {line}")
    if not lines:
        return "未查询到结果。"
    return header + "\n" + "\n".join(lines)


class _BaseInteractionSQLTool(BaseTool):
    DEFAULT_TABLE = "Interaction_info"

    def __init__(self, db_path: str, table_name: str = DEFAULT_TABLE, doc_type: str = "screenplay"):
        self.db_path = db_path
        self.table_name = table_name or self.DEFAULT_TABLE
        self._cols = self._get_columns()
        self.doc_type = str(doc_type or "screenplay").strip().lower()

        meta = DOC_TYPE_META.get(self.doc_type, DOC_TYPE_META["general"])
        self.section_label = str(meta.get("section_label", "Section")).strip() or "Section"
        self.section_title_field = str(meta.get("title", "title")).strip() or "title"
        self.section_subtitle_field = str(meta.get("subtitle", "subtitle")).strip() or "subtitle"

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _get_columns(self) -> set[str]:
        conn = self._connect()
        try:
            cur = conn.cursor()
            cur.execute(f"PRAGMA table_info({_qident(self.table_name)})")
            return {str(r[1]) for r in cur.fetchall()}
        finally:
            conn.close()

    def _available(self, cols: List[str]) -> List[str]:
        return [c for c in cols if c in self._cols]

    def _resolve_column(self, preferred: str, fallbacks: List[str]) -> str:
        candidates = [preferred] + [c for c in fallbacks if c != preferred]
        for col in candidates:
            if col in self._cols:
                return col
        return ""

    def _query_rows(
        self,
        *,
        select_cols: List[str],
        where_sql: str,
        params: Tuple[Any, ...],
        order_by: str,
        limit: int,
    ) -> List[Dict[str, Any]]:
        cols = self._available(select_cols)
        if not cols:
            return []
        sql = (
            f"SELECT {', '.join(_qident(c) for c in cols)} "
            f"FROM {_qident(self.table_name)} "
            f"WHERE {where_sql} "
            f"ORDER BY {order_by} "
            f"LIMIT ?"
        )
        conn = self._connect()
        try:
            cur = conn.cursor()
            cur.execute(sql, tuple(params) + (int(limit),))
            return [dict(r) for r in cur.fetchall()]
        finally:
            conn.close()


@register_tool("search_dialogues")
class SQLSearchDialogues(_BaseInteractionSQLTool):
    name = "search_dialogues"
    description = "检索对白交互记录，支持按说话方、对象、章节/场景、文档ID和内容过滤。"
    parameters = [
        {"name": "subject", "type": "string", "required": False, "description": "按 subject_name 模糊匹配"},
        {"name": "object", "type": "string", "required": False, "description": "按 object_name 模糊匹配"},
        {"name": "section", "type": "string", "required": False, "description": "按章节/场景标题模糊匹配"},
        {"name": "subsection", "type": "string", "required": False, "description": "按子章节/子场景标题模糊匹配"},
        {"name": "content", "type": "string", "required": False, "description": "按 content 模糊匹配"},
        {"name": "document_id", "type": "string", "required": False, "description": "按 document_id 精确匹配"},
        {"name": "limit", "type": "integer", "required": False, "description": "返回条数，默认 20"},
    ]

    def call(self, params: str, **kwargs) -> str:
        p = json.loads(params or "{}")
        subject = str(p.get("subject", "")).strip()
        obj = str(p.get("object", "")).strip()
        section = str(p.get("section", p.get("scene", ""))).strip()
        subsection = str(p.get("subsection", p.get("sub_scene", ""))).strip()
        content = str(p.get("content", "")).strip()
        document_id = str(p.get("document_id", "")).strip()
        limit = int(p.get("limit", 20) or 20)
        limit = max(1, min(limit, 200))

        section_col = self._resolve_column(self.section_title_field, ["scene_name", "chapter_name", "title"])
        subsection_col = self._resolve_column(
            self.section_subtitle_field,
            ["sub_scene_name", "sub_chapter_name", "subtitle"],
        )

        where = ["interaction_type = ?"]
        vals: List[Any] = ["dialogue"]
        if subject:
            where.append("subject_name LIKE ? COLLATE NOCASE")
            vals.append(_like(subject))
        if obj:
            where.append("object_name LIKE ? COLLATE NOCASE")
            vals.append(_like(obj))
        if section and section_col:
            where.append(f"{_qident(section_col)} LIKE ? COLLATE NOCASE")
            vals.append(_like(section))
        if subsection and subsection_col:
            where.append(f"{_qident(subsection_col)} LIKE ? COLLATE NOCASE")
            vals.append(_like(subsection))
        if content:
            where.append("content LIKE ? COLLATE NOCASE")
            vals.append(_like(content))
        if document_id:
            where.append("document_id = ?")
            vals.append(document_id)

        select_cols = [
            "id",
            "document_id",
            "subject_name",
            "subject_type",
            "object_name",
            "object_type",
            "content",
        ]
        if section_col:
            select_cols.insert(2, section_col)
        if subsection_col:
            select_cols.insert(3 if section_col else 2, subsection_col)
        rows = self._query_rows(
            select_cols=select_cols,
            where_sql=" AND ".join(where),
            params=tuple(vals),
            order_by="id",
            limit=limit,
        )
        logger.info("search_dialogues rows=%s", len(rows))
        return _fmt_rows(rows, self._available(select_cols), header=f"Dialogue 查询结果（{self.section_label}维度）：")


@register_tool("search_interactions")
class SQLSearchInteractions(_BaseInteractionSQLTool):
    name = "search_interactions"
    description = "检索非对白交互记录，支持按主客体、章节/场景、情感极性、文档ID和内容过滤。"
    parameters = [
        {"name": "subject", "type": "string", "required": False, "description": "按 subject_name 模糊匹配"},
        {"name": "object", "type": "string", "required": False, "description": "按 object_name 模糊匹配"},
        {"name": "section", "type": "string", "required": False, "description": "按章节/场景标题模糊匹配"},
        {"name": "subsection", "type": "string", "required": False, "description": "按子章节/子场景标题模糊匹配"},
        {"name": "content", "type": "string", "required": False, "description": "按 content 模糊匹配"},
        {"name": "polarity", "type": "string", "required": False, "description": "positive/negative/neutral"},
        {"name": "document_id", "type": "string", "required": False, "description": "按 document_id 精确匹配"},
        {"name": "limit", "type": "integer", "required": False, "description": "返回条数，默认 20"},
    ]

    def call(self, params: str, **kwargs) -> str:
        p = json.loads(params or "{}")
        subject = str(p.get("subject", "")).strip()
        obj = str(p.get("object", "")).strip()
        section = str(p.get("section", p.get("scene", ""))).strip()
        subsection = str(p.get("subsection", p.get("sub_scene", ""))).strip()
        content = str(p.get("content", "")).strip()
        polarity = str(p.get("polarity", "")).strip().lower()
        document_id = str(p.get("document_id", "")).strip()
        limit = int(p.get("limit", 20) or 20)
        limit = max(1, min(limit, 200))

        section_col = self._resolve_column(self.section_title_field, ["scene_name", "chapter_name", "title"])
        subsection_col = self._resolve_column(
            self.section_subtitle_field,
            ["sub_scene_name", "sub_chapter_name", "subtitle"],
        )

        where = ["interaction_type <> ?"]
        vals: List[Any] = ["dialogue"]
        if subject:
            where.append("subject_name LIKE ? COLLATE NOCASE")
            vals.append(_like(subject))
        if obj:
            where.append("object_name LIKE ? COLLATE NOCASE")
            vals.append(_like(obj))
        if section and section_col:
            where.append(f"{_qident(section_col)} LIKE ? COLLATE NOCASE")
            vals.append(_like(section))
        if subsection and subsection_col:
            where.append(f"{_qident(subsection_col)} LIKE ? COLLATE NOCASE")
            vals.append(_like(subsection))
        if content:
            where.append("content LIKE ? COLLATE NOCASE")
            vals.append(_like(content))
        if polarity:
            where.append("LOWER(polarity) = ?")
            vals.append(polarity)
        if document_id:
            where.append("document_id = ?")
            vals.append(document_id)

        select_cols = [
            "id",
            "document_id",
            "subject_name",
            "subject_type",
            "object_name",
            "object_type",
            "interaction_type",
            "polarity",
            "content",
        ]
        if section_col:
            select_cols.insert(2, section_col)
        if subsection_col:
            select_cols.insert(3 if section_col else 2, subsection_col)
        rows = self._query_rows(
            select_cols=select_cols,
            where_sql=" AND ".join(where),
            params=tuple(vals),
            order_by="id",
            limit=limit,
        )
        logger.info("search_interactions rows=%s", len(rows))
        return _fmt_rows(
            rows,
            self._available(select_cols),
            header=f"Interaction 查询结果（{self.section_label}维度）：",
        )


@register_tool("get_interactions_by_document_ids")
class SQLGetInteractionsByDocumentIDs(_BaseInteractionSQLTool):
    name = "get_interactions_by_document_ids"
    description = "按 document_id 列表批量获取交互记录，可按交互类型过滤。"
    parameters = [
        {"name": "document_ids", "type": "array", "required": True, "description": "document_id 列表"},
        {
            "name": "interaction_type",
            "type": "string",
            "required": False,
            "description": "可选: dialogue 或 interaction；不填表示返回全部",
        },
        {"name": "limit", "type": "integer", "required": False, "description": "返回条数，默认 100"},
    ]

    @staticmethod
    def _dedup_ids(xs: Any) -> List[str]:
        if not isinstance(xs, list):
            return []
        out: List[str] = []
        seen = set()
        for x in xs:
            s = str(x).strip()
            if not s or s in seen:
                continue
            seen.add(s)
            out.append(s)
        return out

    def call(self, params: str, **kwargs) -> str:
        p = json.loads(params or "{}")
        document_ids = self._dedup_ids(p.get("document_ids"))
        interaction_type_filter = str(p.get("interaction_type", "")).strip().lower()
        limit = int(p.get("limit", 100) or 100)
        limit = max(1, min(limit, 500))
        section_col = self._resolve_column(self.section_title_field, ["scene_name", "chapter_name", "title"])
        subsection_col = self._resolve_column(
            self.section_subtitle_field,
            ["sub_scene_name", "sub_chapter_name", "subtitle"],
        )

        if not document_ids:
            return "请提供非空 document_ids。"
        if interaction_type_filter not in ("", "all", "dialogue", "interaction"):
            return "interaction_type 仅支持 dialogue 或 interaction（留空表示全部）。"

        placeholders = ",".join(["?"] * len(document_ids))
        where = [f"document_id IN ({placeholders})"]
        vals: List[Any] = list(document_ids)
        if interaction_type_filter == "dialogue":
            where.append("interaction_type = ?")
            vals.append("dialogue")
        elif interaction_type_filter == "interaction":
            where.append("interaction_type <> ?")
            vals.append("dialogue")

        select_cols = [
            "id",
            "document_id",
            "subject_name",
            "object_name",
            "interaction_type",
            "content",
        ]
        if section_col:
            select_cols.insert(2, section_col)
        if subsection_col:
            select_cols.insert(3 if section_col else 2, subsection_col)
        if interaction_type_filter != "dialogue":
            select_cols.append("polarity")
        rows = self._query_rows(
            select_cols=select_cols,
            where_sql=" AND ".join(where),
            params=tuple(vals),
            order_by="id",
            limit=limit,
        )
        logger.info("get_interactions_by_document_ids rows=%s", len(rows))
        return _fmt_rows(rows, self._available(select_cols), header="按 document_id 查询结果：")

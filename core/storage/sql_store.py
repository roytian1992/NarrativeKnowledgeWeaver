"""
SQL storage module

SQLite-backed storage manager for generic JSON-list persistence.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from typing import Any, Dict, List, Optional

from core.utils.config import KAGConfig
from core.utils.general_utils import safe_str

logger = logging.getLogger(__name__)


class SQLStore:
    """SQLite storage manager."""

    _VALID_SQL_TYPES = {"TEXT", "INTEGER", "REAL", "BLOB"}

    def __init__(
        self,
        config: KAGConfig,
        *,
        db_name: str = "Interaction.db",
        db_path: Optional[str] = None,
    ) -> None:
        self.config = config
        if db_path:
            self.db_path = db_path
        else:
            self.db_path = os.path.join(self.config.storage.sql_database_path, db_name)

        db_dir = os.path.dirname(self.db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)

    @staticmethod
    def _qident(name: str) -> str:
        return '"' + str(name).replace('"', '""') + '"'

    @staticmethod
    def _sanitize_sql_type(t: str) -> str:
        typ = safe_str(t).strip().upper()
        return typ if typ in SQLStore._VALID_SQL_TYPES else "TEXT"

    @staticmethod
    def _infer_sql_type(values: List[Any]) -> str:
        has_text = False
        has_real = False
        has_int = False
        has_blob = False

        for v in values:
            if v is None:
                continue
            if isinstance(v, (dict, list, str)):
                has_text = True
            elif isinstance(v, bool):
                has_int = True
            elif isinstance(v, int):
                has_int = True
            elif isinstance(v, float):
                has_real = True
            elif isinstance(v, (bytes, bytearray)):
                has_blob = True
            else:
                has_text = True

        if has_text:
            return "TEXT"
        if has_blob:
            return "BLOB"
        if has_real:
            return "REAL"
        if has_int:
            return "INTEGER"
        return "TEXT"

    @staticmethod
    def _to_db_value(v: Any) -> Any:
        if v is None:
            return None
        if isinstance(v, bool):
            return int(v)
        if isinstance(v, (dict, list)):
            return json.dumps(v, ensure_ascii=False)
        if isinstance(v, (str, int, float, bytes, bytearray)):
            return v
        return safe_str(v)

    def reset_database(self) -> None:
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def load_json_list(self, *, json_path: str) -> List[Dict[str, Any]]:
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"JSON file not found: {json_path}")

        with open(json_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if not isinstance(raw, list):
            raise ValueError(f"JSON root must be a list of objects: {json_path}")

        normalized: List[Dict[str, Any]] = []
        for x in raw:
            if isinstance(x, dict):
                normalized.append(x)
        return normalized

    def insert_records(
        self,
        *,
        table_name: str,
        records: List[Dict[str, Any]],
        columns: Optional[List[str]] = None,
        column_types: Optional[Dict[str, str]] = None,
        include_auto_id: bool = True,
        reset_database: bool = False,
        reset_table: bool = True,
    ) -> int:
        if reset_database:
            self.reset_database()

        clean_records: List[Dict[str, Any]] = [x for x in (records or []) if isinstance(x, dict)]

        if columns is None:
            columns = []
            seen = set()
            for row in clean_records:
                for k in row.keys():
                    col = safe_str(k).strip()
                    if not col:
                        continue
                    if include_auto_id and col == "id":
                        continue
                    if col in seen:
                        continue
                    seen.add(col)
                    columns.append(col)
        else:
            columns = [safe_str(c).strip() for c in columns if safe_str(c).strip()]
            if include_auto_id:
                columns = [c for c in columns if c != "id"]

        if not include_auto_id and not columns:
            raise ValueError("At least one column is required when include_auto_id=False")

        inferred_types: Dict[str, str] = {}
        for col in columns:
            if isinstance(column_types, dict) and col in column_types:
                inferred_types[col] = self._sanitize_sql_type(column_types[col])
                continue
            inferred_types[col] = self._infer_sql_type([row.get(col) for row in clean_records])

        q_table = self._qident(table_name)
        q_columns = [self._qident(c) for c in columns]

        conn = self._connect()
        try:
            cur = conn.cursor()
            if reset_table:
                cur.execute(f"DROP TABLE IF EXISTS {q_table}")

            create_defs: List[str] = []
            if include_auto_id:
                create_defs.append("id INTEGER PRIMARY KEY AUTOINCREMENT")
            for col in columns:
                create_defs.append(f'{self._qident(col)} {inferred_types[col]}')
            cur.execute(f"CREATE TABLE IF NOT EXISTS {q_table} ({', '.join(create_defs)})")

            inserted = 0
            if columns and clean_records:
                placeholders = ", ".join(["?"] * len(columns))
                cur.executemany(
                    f"INSERT INTO {q_table} ({', '.join(q_columns)}) VALUES ({placeholders})",
                    [
                        tuple(self._to_db_value(row.get(col)) for col in columns)
                        for row in clean_records
                    ],
                )
                inserted = len(clean_records)

            conn.commit()
            logger.info("SQLStore: stored %s rows into %s:%s", inserted, self.db_path, table_name)
            return inserted
        finally:
            conn.close()

    def store_json_list(
        self,
        *,
        json_path: str,
        table_name: str,
        columns: Optional[List[str]] = None,
        column_types: Optional[Dict[str, str]] = None,
        include_auto_id: bool = True,
        reset_database: bool = False,
        reset_table: bool = True,
    ) -> int:
        records = self.load_json_list(json_path=json_path)
        return self.insert_records(
            table_name=table_name,
            records=records,
            columns=columns,
            column_types=column_types,
            include_auto_id=include_auto_id,
            reset_database=reset_database,
            reset_table=reset_table,
        )

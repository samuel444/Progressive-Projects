
"""Lightweight SQLite persistence for reproducible project snapshots."""

from __future__ import annotations

import logging
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

import pandas as pd

from options_risk_engine.utils import flatten_dataframe_columns

logger = logging.getLogger(__name__)


class SQLiteStore:
    def __init__(self, database_path: Path, schema_path: Optional[Path] = None):
        self.database_path = Path(database_path)
        self.schema_path = Path(schema_path) if schema_path is not None else None
        self.database_path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database_path)
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialise(self) -> None:
        if self.schema_path is None:
            raise ValueError("schema_path is required to initialise the database")
        schema = self.schema_path.read_text(encoding="utf-8")
        with self.connect() as connection:
            connection.executescript(schema)
        logger.info("SQLite database initialised at %s", self.database_path)

    def start_run(self, valuation_date: str, notes: str = "") -> str:
        run_id = uuid.uuid4().hex
        with self.connect() as connection:
            connection.execute(
                "INSERT INTO project_runs(run_id, created_at, valuation_date, notes) VALUES (?, ?, ?, ?)",
                (run_id, datetime.now(timezone.utc).isoformat(), valuation_date, notes),
            )
        return run_id

    def write_frame(
        self,
        frame: pd.DataFrame,
        table_name: str,
        if_exists: str = "append",
        run_id: Optional[str] = None,
        index: bool = False,
    ) -> None:
        data = flatten_dataframe_columns(frame)
        if run_id is not None and "run_id" not in data.columns:
            data.insert(0, "run_id", run_id)
        data["stored_at"] = datetime.now(timezone.utc).isoformat()
        with self.connect() as connection:
            data.to_sql(table_name, connection, if_exists=if_exists, index=index)
        logger.info("Stored %d rows in %s", len(data), table_name)

    def read_query(self, query: str, params: tuple = ()) -> pd.DataFrame:
        with self.connect() as connection:
            return pd.read_sql_query(query, connection, params=params)

    def list_tables(self) -> pd.DataFrame:
        return self.read_query(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )

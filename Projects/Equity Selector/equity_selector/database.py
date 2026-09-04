"""Quoted SQLite identifiers and transactional DataFrame writes.

Result tables are derived artifacts with inferred types. Replacing a table with
user-defined indexes, triggers, keys or constraints is refused rather than
silently deleting its schema. Caller owns commit/rollback via `with connection`.
"""

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd


def quote_identifier(value):
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError("SQL identifier must be nonempty text without NUL")
    return '"' + value.replace('"', '""') + '"'


def read_table(path, table):
    path = Path(path).resolve(strict=True)
    with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as connection:
        return pd.read_sql_query(f"SELECT * FROM {quote_identifier(table)}", connection)


def _value(value):
    if value is None or pd.api.types.is_scalar(value) and pd.isna(value):
        return None
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return str(pd.Timestamp(value))
    if isinstance(value, np.generic):
        value = value.item()
    if not isinstance(value, (int, float, str, bytes)):
        raise ValueError("Serialize structured values before writing SQLite")
    return value


def write_frame(frame, name, connection, if_exists="fail", index=False):
    if index:
        raise ValueError("Explicitly reset the index before database writes")
    if if_exists not in {"fail", "append", "replace"}:
        raise ValueError("Unknown if_exists mode")
    if len(frame.columns) == 0 or not frame.columns.is_unique:
        raise ValueError("Database frame needs unique, nonempty columns")
    table = quote_identifier(name)
    columns = [str(column) for column in frame.columns]
    quoted = [quote_identifier(column) for column in columns]
    if len(set(column.casefold() for column in columns)) != len(columns):
        raise ValueError("SQLite column names must be case-insensitively unique")
    rows = [
        tuple(_value(value) for value in row) for row in frame.itertuples(index=False, name=None)
    ]
    exists = connection.execute(
        "SELECT sql FROM sqlite_master WHERE type=? AND name=?", ("table", name)
    ).fetchone()
    if exists and if_exists == "fail":
        raise ValueError(f"Table already exists: {name}")
    if exists and if_exists == "append":
        actual = [row[1] for row in connection.execute(f"PRAGMA table_info({table})")]
        if set(actual) != set(columns):
            raise ValueError(f"Append schema mismatch for {name}")
    if exists and if_exists == "replace":
        schema = connection.execute(f"PRAGMA table_info({table})").fetchall()
        objects = connection.execute(
            "SELECT name FROM sqlite_master WHERE tbl_name=? AND type IN ('index', 'trigger')",
            (name,),
        ).fetchall()
        foreign_keys = connection.execute(f"PRAGMA foreign_key_list({table})").fetchall()
        if (
            objects
            or foreign_keys
            or any(row[3] or row[4] is not None or row[5] for row in schema)
            or "CHECK" in exists[0].upper()
        ):
            raise ValueError(f"Refusing to replace custom schema for {name}")
    if not connection.in_transaction:
        connection.execute("BEGIN")
    connection.execute("SAVEPOINT equity_selector_write")
    try:
        if exists and if_exists == "replace":
            connection.execute(f"DROP TABLE {table}")
        if not exists or if_exists == "replace":
            types = [
                (
                    "INTEGER"
                    if pd.api.types.is_integer_dtype(dtype) or pd.api.types.is_bool_dtype(dtype)
                    else "REAL"
                    if pd.api.types.is_numeric_dtype(dtype)
                    else "TEXT"
                )
                for dtype in frame.dtypes
            ]
            definitions = ", ".join(f"{column} {kind}" for column, kind in zip(quoted, types))
            connection.execute(f"CREATE TABLE {table} ({definitions})")
        placeholders = ", ".join("?" for _ in columns)
        connection.executemany(
            f"INSERT INTO {table} ({', '.join(quoted)}) VALUES ({placeholders})", rows
        )
        connection.execute("RELEASE SAVEPOINT equity_selector_write")
    except Exception:
        connection.execute("ROLLBACK TO SAVEPOINT equity_selector_write")
        connection.execute("RELEASE SAVEPOINT equity_selector_write")
        raise
    return len(frame)

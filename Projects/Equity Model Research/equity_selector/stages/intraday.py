from equity_selector.config import data_root
from equity_selector.files import commit_with_text
from uuid import uuid4
import logging
import re
import sqlite3
from collections import defaultdict
from pathlib import Path
import ast

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)
DATA_DIR = Path(str(data_root()) + "/")
DATABASE = DATA_DIR / "Features_Targets_Data.db"
STOCK_TYPE = "Intraday Medium Liquidity 30"
MAX_PERIODS = 60
SELECTED_FEATURES_FILE = DATA_DIR / "Selected_Features.txt"
STOCK_TYPE_INDICES = {
    "High Liquidity 30": 0,
    "Medium Liquidity 30": 1,
    "Lower Liquidity 30": 2,
    "Sector Spread 30": 3,
    "Intraday High Liquidity 30": 4,
    "Intraday Medium Liquidity 30": 5,
    "Liquidity Barbell 30": 6,
    "Institutional Liquidity 60": 7,
    "Medium Small Liquidity 60": 8,
    "Medium Large Liquidity 60": 9,
    "All Liquidity 90": 10,
}
FEATURE_LOOKBACK_OVERRIDES = {}
TARGET_HORIZON_OVERRIDES = {}
TARGET_LOOKBACK_OVERRIDES = {}


def quote_identifier(value):
    return '"' + str(value).replace('"', '""') + '"'


def table_columns(connection, table):
    rows = connection.execute(
        f"\n        PRAGMA table_info(\n            {quote_identifier(table)}\n        )\n        "
    ).fetchall()
    return [row[1] for row in rows]


BASE_COLUMNS = {"Date", "Ticker", "Open", "High", "Low", "Close", "Volume", "Return"}
TARGET_PREFIXES = (
    "forward ",
    "future ",
    "barrier ",
    "volatility barrier ",
    "maximum favourable excursion ",
    "maximum favorable excursion ",
    "maximum adverse excursion ",
    "time to maximum favourable excursion ",
    "time to maximum favorable excursion ",
    "time to maximum adverse excursion ",
    "top ",
    "bottom ",
)


def is_target_column(column):
    return str(column).strip().lower().startswith(TARGET_PREFIXES)


def explicit_minute_values(name):
    matches = re.findall(
        "(?<![a-z0-9])(\\d+(?:\\.\\d+)?)\\s*(?:m|min|mins|minute|minutes)(?![a-z])",
        str(name).lower(),
    )
    return [float(value) for value in matches]


def standalone_numbers(name):
    matches = re.findall("(?<![a-z0-9.])(\\d+(?:\\.\\d+)?)(?![a-z0-9.%])", str(name).lower())
    return [float(value) for value in matches]


def infer_target_horizon(column):
    if column in TARGET_HORIZON_OVERRIDES:
        return TARGET_HORIZON_OVERRIDES[column]
    name = str(column).strip().lower()
    explicit = explicit_minute_values(name)
    if explicit:
        return int(round(explicit[-1]))
    numbers = standalone_numbers(name)
    if not numbers:
        return None
    if name.startswith("volatility barrier ") and len(numbers) >= 2:
        return int(round(numbers[1]))
    if name.startswith("future return minus risk "):
        return int(round(numbers[0]))
    return int(round(numbers[-1]))


def infer_target_lookback(column):
    if column in TARGET_LOOKBACK_OVERRIDES:
        return TARGET_LOOKBACK_OVERRIDES[column]
    name = str(column).strip().lower()
    if name.startswith("volatility barrier "):
        numbers = standalone_numbers(name)
        if numbers:
            return int(round(numbers[0]))
    return 0


def infer_feature_lookback(column):
    if column in FEATURE_LOOKBACK_OVERRIDES:
        return FEATURE_LOOKBACK_OVERRIDES[column]
    name = str(column).strip().lower()
    explicit = explicit_minute_values(name)
    if explicit:
        return int(round(max(explicit)))
    numbers = standalone_numbers(name)
    if not numbers:
        return 0
    return int(round(max(numbers)))


def analyse_columns(columns):
    kept_columns = []
    null_rules = {}
    for column in columns:
        if column in BASE_COLUMNS:
            kept_columns.append(column)
            if column == "Return":
                null_rules[column] = {"first_rows": 1, "last_rows": 0}
            continue
        if is_target_column(column):
            horizon = infer_target_horizon(column)
            lookback = infer_target_lookback(column)
            if horizon is None:
                logger.warning("Could not infer target horizon; keeping unchanged: %s", column)
                kept_columns.append(column)
                continue
            if horizon > MAX_PERIODS or lookback > MAX_PERIODS:
                logger.info("Removing target: %s", column)
                continue
            kept_columns.append(column)
            null_rules[column] = {"first_rows": max(lookback - 1, 0), "last_rows": horizon}
            continue
        lookback = infer_feature_lookback(column)
        if lookback > MAX_PERIODS:
            logger.info("Removing feature: %s", column)
            continue
        kept_columns.append(column)
        if lookback > 0:
            null_rules[column] = {"first_rows": max(lookback - 1, 0), "last_rows": 0}
    return (kept_columns, null_rules)


def clean_intraday_table():
    if not DATABASE.exists():
        raise FileNotFoundError(DATABASE)
    logger.info("Cleaning %s | table=%s", DATABASE, STOCK_TYPE)
    with sqlite3.connect(DATABASE) as connection:
        connection.execute("BEGIN IMMEDIATE")
        table = quote_identifier(STOCK_TYPE)
        schema = connection.execute(f"PRAGMA table_info({table})").fetchall()
        objects = connection.execute(
            "SELECT name FROM sqlite_master WHERE tbl_name=? AND type IN ('index', 'trigger')",
            (STOCK_TYPE,),
        ).fetchall()
        sql = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (STOCK_TYPE,)
        ).fetchone()
        if (
            objects
            or any((row[3] or row[4] is not None or row[5] for row in schema))
            or (sql and "CHECK" in sql[0].upper())
            or connection.execute(f"PRAGMA foreign_key_list({table})").fetchall()
        ):
            raise ValueError("Intraday conversion cannot replace custom schema constraints/indexes")
        columns = table_columns(connection, STOCK_TYPE)
        if not columns:
            raise ValueError(f"Table does not exist or has no columns: {STOCK_TYPE}")
        required = {"Date", "Ticker"}
        missing = required - set(columns)
        if missing:
            raise ValueError("Missing required columns: " + ", ".join(sorted(missing)))
        kept_columns, null_rules = analyse_columns(columns)
        if not set(kept_columns).issubset(columns) or not set(null_rules).issubset(kept_columns):
            raise ValueError("Intraday rules refer to missing source columns")
        logger.info("Columns: %d -> %d", len(columns), len(kept_columns))
        from equity_selector.feature_mapping import (
            load_feature_mapping,
            updated_feature_mapping_text,
        )

        selected_features = load_feature_mapping(
            SELECTED_FEATURES_FILE, STOCK_TYPE, STOCK_TYPE_INDICES
        )
        kept_column_set = set(kept_columns)
        cleaned_selected_features = {}
        for target, features in selected_features.items():
            if target not in kept_column_set:
                logger.info("Removing target from Selected_Features.txt: %s", target)
                continue
            cleaned_features = [feature for feature in features if feature in kept_column_set]
            removed_features = [feature for feature in features if feature not in kept_column_set]
            for feature in removed_features:
                logger.debug(
                    "Removing feature from Selected_Features.txt | %s | %s", target, feature
                )
            cleaned_selected_features[target] = cleaned_features
        updated_mapping = updated_feature_mapping_text(
            SELECTED_FEATURES_FILE,
            STOCK_TYPE,
            cleaned_selected_features,
            STOCK_TYPE_INDICES,
        )
        original_table = quote_identifier(STOCK_TYPE)
        temporary_name = "__intraday_clean_" + uuid4().hex
        temporary_table = quote_identifier(temporary_name)
        connection.execute(
            f"\n            DROP TABLE IF EXISTS\n                {temporary_table}\n            "
        )
        kept_sql = ", ".join((quote_identifier(column) for column in kept_columns))
        connection.execute(
            f"\n            CREATE TABLE\n                {temporary_table}\n            AS\n            SELECT\n                {kept_sql}\n            FROM\n                {original_table}\n            "
        )
        connection.execute(
            "\n            DROP TABLE IF EXISTS\n                temp.__session_rows\n            "
        )
        connection.execute(
            f'\n            CREATE TEMP TABLE\n                __session_rows\n            AS\n            SELECT\n                rowid\n                    AS source_rowid,\n\n                ROW_NUMBER() OVER (\n                    PARTITION BY\n                        "Ticker",\n                        substr(\n                            CAST(\n                                "Date"\n                                AS TEXT\n                            ),\n                            1,\n                            10\n                        )\n                    ORDER BY\n                        "Date",\n                        rowid\n                )\n                    AS session_row,\n\n                COUNT(*) OVER (\n                    PARTITION BY\n                        "Ticker",\n                        substr(\n                            CAST(\n                                "Date"\n                                AS TEXT\n                            ),\n                            1,\n                            10\n                        )\n                )\n                    AS session_count\n\n            FROM\n                {temporary_table}\n            '
        )
        connection.execute(
            "\n            CREATE INDEX\n                __session_rows_rowid_idx\n            ON\n                __session_rows(\n                    source_rowid\n                )\n            "
        )
        grouped_rules = defaultdict(list)
        for column, rule in null_rules.items():
            first_rows = int(rule["first_rows"])
            last_rows = int(rule["last_rows"])
            if first_rows == 0 and last_rows == 0:
                continue
            grouped_rules[first_rows, last_rows].append(column)
        for (first_rows, last_rows), rule_columns in grouped_rules.items():
            assignments = ", ".join(
                (f"{quote_identifier(column)} = NULL" for column in rule_columns)
            )
            conditions = []
            parameters = []
            if first_rows > 0:
                conditions.append("session_row <= ?")
                parameters.append(first_rows)
            if last_rows > 0:
                conditions.append("session_row > session_count - ?")
                parameters.append(last_rows)
            where_condition = " OR ".join(conditions)
            connection.execute(
                f"\n                UPDATE\n                    {temporary_table}\n\n                SET\n                    {assignments}\n\n                WHERE\n                    rowid IN (\n\n                        SELECT\n                            source_rowid\n\n                        FROM\n                            __session_rows\n\n                        WHERE\n                            {where_condition}\n                    )\n                ",
                tuple(parameters),
            )
        original_rows = connection.execute(
            f"\n                SELECT COUNT(*)\n                FROM {original_table}\n                "
        ).fetchone()[0]
        cleaned_rows = connection.execute(
            f"\n                SELECT COUNT(*)\n                FROM {temporary_table}\n                "
        ).fetchone()[0]
        if original_rows != cleaned_rows:
            raise RuntimeError("Row count changed during intraday cleaning.")
        connection.execute(
            f"\n            DROP TABLE\n                {original_table}\n            "
        )
        connection.execute(
            f"\n            ALTER TABLE\n                {temporary_table}\n\n            RENAME TO\n                {quote_identifier(STOCK_TYPE)}\n            "
        )
        commit_with_text(connection, SELECTED_FEATURES_FILE, updated_mapping)
        logger.info("Intraday mapping updated | targets=%d", len(cleaned_selected_features))
    logger.info("Complete | table replaced: %s", STOCK_TYPE)


if __name__ == "__main__":
    clean_intraday_table()


def run():
    return clean_intraday_table()

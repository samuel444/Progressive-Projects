import logging
import re
import sqlite3
from collections import defaultdict
from pathlib import Path
import ast


########################################
# Logging
########################################

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger(__name__)


########################################
# Settings
########################################

DATA_DIR = Path(
    "/Users/sam/Progressive-Projects/Projects/Equity Selector/data/"
)

DATABASE = (
    DATA_DIR
    / "Features_Targets_Data.db"
)

STOCK_TYPE = (
    #"Intraday High Liquidity 30"
    "Intraday Medium Liquidity 30"
)

MAX_PERIODS = 60

SELECTED_FEATURES_FILE = (
    DATA_DIR
    / "Selected_Features.txt"
)


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


########################################
# Optional Manual Corrections
########################################

FEATURE_LOOKBACK_OVERRIDES = {}

TARGET_HORIZON_OVERRIDES = {}

TARGET_LOOKBACK_OVERRIDES = {}


########################################
# SQL Helpers
########################################

def quote_identifier(value):

    return (
        '"'
        + str(value).replace('"', '""')
        + '"'
    )


def table_columns(
    connection,
    table,
):

    rows = connection.execute(
        f"""
        PRAGMA table_info(
            {quote_identifier(table)}
        )
        """
    ).fetchall()

    return [
        row[1]
        for row in rows
    ]


########################################
# Column Types
########################################

BASE_COLUMNS = {
    "Date",
    "Ticker",
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "Return",
}


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


def is_target_column(
    column,
):

    return (
        str(column)
        .strip()
        .lower()
        .startswith(
            TARGET_PREFIXES
        )
    )


########################################
# Number Extraction
########################################

def explicit_minute_values(
    name,
):

    matches = re.findall(
        r"(?<![a-z0-9])"
        r"(\d+(?:\.\d+)?)\s*"
        r"(?:m|min|mins|minute|minutes)"
        r"(?![a-z])",
        str(name).lower(),
    )

    return [
        float(value)
        for value in matches
    ]


def standalone_numbers(
    name,
):

    matches = re.findall(
        r"(?<![a-z0-9.])"
        r"(\d+(?:\.\d+)?)"
        r"(?![a-z0-9.%])",
        str(name).lower(),
    )

    return [
        float(value)
        for value in matches
    ]


########################################
# Target Horizon
########################################

def infer_target_horizon(
    column,
):

    if (
        column
        in TARGET_HORIZON_OVERRIDES
    ):

        return (
            TARGET_HORIZON_OVERRIDES[
                column
            ]
        )


    name = (
        str(column)
        .strip()
        .lower()
    )


    explicit = (
        explicit_minute_values(
            name
        )
    )


    if explicit:

        return int(
            round(
                explicit[-1]
            )
        )


    numbers = (
        standalone_numbers(
            name
        )
    )


    if not numbers:

        return None


    ####################################
    # Volatility Barrier
    ####################################

    # Volatility Barrier
    # {volatility_window}
    # {horizon}
    # {upper_multiple}
    # {lower_multiple}

    if (
        name.startswith(
            "volatility barrier "
        )
        and len(numbers) >= 2
    ):

        return int(
            round(
                numbers[1]
            )
        )


    ####################################
    # Future Return Minus Risk
    ####################################

    # Future Return Minus Risk
    # {horizon}
    # {risk_weight}

    if name.startswith(
        "future return minus risk "
    ):

        return int(
            round(
                numbers[0]
            )
        )


    ####################################
    # General Target Convention
    ####################################

    return int(
        round(
            numbers[-1]
        )
    )


########################################
# Target Lookback
########################################

def infer_target_lookback(
    column,
):

    if (
        column
        in TARGET_LOOKBACK_OVERRIDES
    ):

        return (
            TARGET_LOOKBACK_OVERRIDES[
                column
            ]
        )


    name = (
        str(column)
        .strip()
        .lower()
    )


    ####################################
    # Volatility Barrier
    ####################################

    if name.startswith(
        "volatility barrier "
    ):

        numbers = (
            standalone_numbers(
                name
            )
        )


        if numbers:

            return int(
                round(
                    numbers[0]
                )
            )


    return 0


########################################
# Feature Lookback
########################################

def infer_feature_lookback(
    column,
):

    if (
        column
        in FEATURE_LOOKBACK_OVERRIDES
    ):

        return (
            FEATURE_LOOKBACK_OVERRIDES[
                column
            ]
        )


    name = (
        str(column)
        .strip()
        .lower()
    )


    explicit = (
        explicit_minute_values(
            name
        )
    )


    if explicit:

        return int(
            round(
                max(
                    explicit
                )
            )
        )


    numbers = (
        standalone_numbers(
            name
        )
    )


    if not numbers:

        return 0


    return int(
        round(
            max(
                numbers
            )
        )
    )


########################################
# Analyse Columns
########################################

def analyse_columns(
    columns,
):

    kept_columns = []

    null_rules = {}


    for column in columns:


        ####################################
        # Base Columns
        ####################################

        if column in BASE_COLUMNS:

            kept_columns.append(
                column
            )


            if column == "Return":

                null_rules[
                    column
                ] = {
                    "first_rows":
                        1,

                    "last_rows":
                        0,
                }


            continue


        ####################################
        # Targets
        ####################################

        if is_target_column(
            column
        ):

            horizon = (
                infer_target_horizon(
                    column
                )
            )

            lookback = (
                infer_target_lookback(
                    column
                )
            )


            if horizon is None:

                logger.warning(
                    "Could not infer target horizon; "
                    "keeping unchanged: %s",
                    column,
                )

                kept_columns.append(
                    column
                )

                continue


            ################################
            # Remove > 60
            ################################

            if (
                horizon > MAX_PERIODS
                or lookback > MAX_PERIODS
            ):

                logger.info(
                    "Removing target: %s",
                    column,
                )

                continue


            kept_columns.append(
                column
            )


            null_rules[
                column
            ] = {

                # rolling(window) first becomes valid
                # on the `window`th row.
                "first_rows":
                    max(
                        lookback - 1,
                        0,
                    ),

                # horizon future observations
                # must stay inside the same day.
                "last_rows":
                    horizon,
            }


            continue


        ####################################
        # Features
        ####################################

        lookback = (
            infer_feature_lookback(
                column
            )
        )


        ################################
        # Remove > 60
        ################################

        if lookback > MAX_PERIODS:

            logger.info(
                "Removing feature: %s",
                column,
            )

            continue


        kept_columns.append(
            column
        )


        if lookback > 0:

            null_rules[
                column
            ] = {
                "first_rows":
                    max(
                        lookback - 1,
                        0,
                    ),

                "last_rows":
                    0,
            }


    return (
        kept_columns,
        null_rules,
    )


########################################
# Clean Table
########################################

def clean_intraday_table():

    if not DATABASE.exists():

        raise FileNotFoundError(
            DATABASE
        )


    logger.info(
        "Cleaning %s | table=%s",
        DATABASE,
        STOCK_TYPE,
    )


    with sqlite3.connect(
        DATABASE
    ) as connection:


        ####################################
        # Read Schema
        ####################################

        columns = table_columns(
            connection,
            STOCK_TYPE,
        )


        if not columns:

            raise ValueError(
                f"Table does not exist "
                f"or has no columns: "
                f"{STOCK_TYPE}"
            )


        required = {
            "Date",
            "Ticker",
        }


        missing = (
            required
            - set(
                columns
            )
        )


        if missing:

            raise ValueError(
                "Missing required columns: "
                + ", ".join(
                    sorted(
                        missing
                    )
                )
            )


        (
            kept_columns,
            null_rules,
        ) = analyse_columns(
            columns
        )


        logger.info(
            "Columns: %d -> %d",
            len(columns),
            len(kept_columns),
        )

        ########################################
        # Update Selected Features
        ########################################

        if STOCK_TYPE not in STOCK_TYPE_INDICES:

            raise ValueError(
                f"No Selected_Features.txt index "
                f"defined for {STOCK_TYPE}"
            )


        stock_type_index = (
            STOCK_TYPE_INDICES[
                STOCK_TYPE
            ]
        )


        with open(
            SELECTED_FEATURES_FILE,
            "r",
        ) as file:

            selected_feature_lines = (
                file.read().splitlines()
            )


        if stock_type_index >= len(
            selected_feature_lines
        ):

            raise ValueError(
                f"Selected_Features.txt has no "
                f"line for {STOCK_TYPE}"
            )


        selected_features = ast.literal_eval(
            selected_feature_lines[
                stock_type_index
            ]
        )


        kept_column_set = set(
            kept_columns
        )


        ####################################
        # Remove Deleted Targets / Features
        ####################################

        cleaned_selected_features = {}


        for (
            target,
            features
        ) in selected_features.items():


            ################################
            # Target Was Removed
            ################################

            if target not in kept_column_set:

                logger.info(
                    "Removing target from "
                    "Selected_Features.txt: %s",
                    target,
                )

                continue


            ################################
            # Remove Deleted Features
            ################################

            cleaned_features = [
                feature
                for feature in features
                if feature in kept_column_set
            ]


            removed_features = [
                feature
                for feature in features
                if feature not in kept_column_set
            ]


            for feature in removed_features:

                logger.info(
                    "Removing feature from "
                    "Selected_Features.txt | "
                    "%s | %s",
                    target,
                    feature,
                )


            cleaned_selected_features[
                target
            ] = cleaned_features


        ####################################
        # Replace Only Corresponding Line
        ####################################

        selected_feature_lines[
            stock_type_index
        ] = str(
            cleaned_selected_features
        )


        with open(
            SELECTED_FEATURES_FILE,
            "w",
        ) as file:

            file.write(
                "\n".join(
                    selected_feature_lines
                )
            )


        logger.info(
            "Updated Selected_Features.txt | "
            "%s | targets=%d",
            STOCK_TYPE,
            len(
                cleaned_selected_features
            ),
        )


        ####################################
        # Temporary Replacement Table
        ####################################

        original_table = (
            quote_identifier(
                STOCK_TYPE
            )
        )

        temporary_name = (
            "__intraday_clean_tmp"
        )

        temporary_table = (
            quote_identifier(
                temporary_name
            )
        )


        connection.execute(
            f"""
            DROP TABLE IF EXISTS
                {temporary_table}
            """
        )


        kept_sql = ", ".join(
            quote_identifier(
                column
            )
            for column
            in kept_columns
        )


        ####################################
        # Copy Only Surviving Columns
        ####################################

        connection.execute(
            f"""
            CREATE TABLE
                {temporary_table}
            AS
            SELECT
                {kept_sql}
            FROM
                {original_table}
            """
        )


        ####################################
        # Session Row Numbers
        ####################################

        connection.execute(
            """
            DROP TABLE IF EXISTS
                __session_rows
            """
        )


        connection.execute(
            f"""
            CREATE TEMP TABLE
                __session_rows
            AS
            SELECT
                rowid
                    AS source_rowid,

                ROW_NUMBER() OVER (
                    PARTITION BY
                        "Ticker",
                        substr(
                            CAST(
                                "Date"
                                AS TEXT
                            ),
                            1,
                            10
                        )
                    ORDER BY
                        "Date",
                        rowid
                )
                    AS session_row,

                COUNT(*) OVER (
                    PARTITION BY
                        "Ticker",
                        substr(
                            CAST(
                                "Date"
                                AS TEXT
                            ),
                            1,
                            10
                        )
                )
                    AS session_count

            FROM
                {temporary_table}
            """
        )


        connection.execute(
            """
            CREATE INDEX
                __session_rows_rowid_idx
            ON
                __session_rows(
                    source_rowid
                )
            """
        )


        ####################################
        # Group Identical Null Rules
        ####################################

        grouped_rules = defaultdict(
            list
        )


        for (
            column,
            rule
        ) in null_rules.items():

            first_rows = int(
                rule[
                    "first_rows"
                ]
            )

            last_rows = int(
                rule[
                    "last_rows"
                ]
            )


            if (
                first_rows == 0
                and last_rows == 0
            ):

                continue


            grouped_rules[
                (
                    first_rows,
                    last_rows,
                )
            ].append(
                column
            )


        ####################################
        # Set Session Boundaries NULL
        ####################################

        for (
            first_rows,
            last_rows
        ), rule_columns in (
            grouped_rules.items()
        ):


            assignments = ", ".join(
                f"{quote_identifier(column)} = NULL"
                for column
                in rule_columns
            )


            conditions = []

            parameters = []


            if first_rows > 0:

                conditions.append(
                    "session_row <= ?"
                )

                parameters.append(
                    first_rows
                )


            if last_rows > 0:

                conditions.append(
                    "session_row > "
                    "session_count - ?"
                )

                parameters.append(
                    last_rows
                )


            where_condition = (
                " OR ".join(
                    conditions
                )
            )


            connection.execute(
                f"""
                UPDATE
                    {temporary_table}

                SET
                    {assignments}

                WHERE
                    rowid IN (

                        SELECT
                            source_rowid

                        FROM
                            __session_rows

                        WHERE
                            {where_condition}
                    )
                """,
                tuple(
                    parameters
                ),
            )


        ####################################
        # Replace Original Table
        ####################################

        original_rows = (
            connection.execute(
                f"""
                SELECT COUNT(*)
                FROM {original_table}
                """
            )
            .fetchone()[0]
        )


        cleaned_rows = (
            connection.execute(
                f"""
                SELECT COUNT(*)
                FROM {temporary_table}
                """
            )
            .fetchone()[0]
        )


        if (
            original_rows
            != cleaned_rows
        ):

            raise RuntimeError(
                "Row count changed during "
                "intraday cleaning."
            )


        connection.execute(
            f"""
            DROP TABLE
                {original_table}
            """
        )


        connection.execute(
            f"""
            ALTER TABLE
                {temporary_table}

            RENAME TO
                {quote_identifier(STOCK_TYPE)}
            """
        )


        connection.commit()


    logger.info(
        "Complete | table replaced: %s",
        STOCK_TYPE,
    )


########################################
# Run
########################################

if __name__ == "__main__":

    clean_intraday_table()
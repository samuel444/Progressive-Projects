import ast
import gc
import itertools
import logging
import math
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from main_package import (
    benchmark_metrics,
    create_models_and_predictions,
    run_portfolio_backtest_from_predictions,
)


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
# Paths
########################################

DATA_DIR = Path(
    "/Users/sam/Progressive-Projects/Projects/Equity Selector/data/"
)

FINAL_RESULTS_DB = (
    DATA_DIR
    / "Final_Test_Results.db"
)

SELECTED_FEATURES_FILE = (
    DATA_DIR
    / "Selected_Features.txt"
)

# This database already contains the completed feature/target panel.
# The table name inside it is exactly STOCK_TYPE.
FEATURE_DATABASE = (
    DATA_DIR
    / "Features_Targets_Data.db"
)


########################################
# Research Settings
########################################

# Nothing after this date is used to determine Horizon Scores.
RESEARCH_END = pd.Timestamp(
    "2023-09-30"
)

# Final fraction of available research dates used to judge Horizon Scores.
# Earlier dates fit the selected predictive models.
HORIZON_VALIDATION_FRACTION = 0.25


########################################
# Stock Type
########################################

STOCK_TYPE = (
    "High Liquidity 30"
    # "Medium Liquidity 30"
    # "Lower Liquidity 30"
    # "Intraday Higher Liquidity 30"
    # "Intraday Medium Liquidity 30"
    # "Sector Spread 30"
    # "Liquidity Barbell 30"
    # "Institutional Liquidity 60"
    # "Medium Small Liquidity 60"
    # "Medium Large Liquidity 60"
    # "All Liquidity 90"
)

STOCK_TYPE_INDICES = {
    "High Liquidity 30": 0,
    "Medium Liquidity 30": 1,
    "Lower Liquidity 30": 2,
    "Sector Spread 30": 3,
    "Intraday Higher Liquidity 30": 4,
    "Intraday Medium Liquidity 30": 5,
    "Liquidity Barbell 30": 6,
    "Institutional Liquidity 60": 7,
    "Medium Small Liquidity 60": 8,
    "Medium Large Liquidity 60": 9,
    "All Liquidity 90": 10,
}

if STOCK_TYPE not in STOCK_TYPE_INDICES:
    raise ValueError(
        f"Unknown STOCK_TYPE: {STOCK_TYPE}"
    )

stock_type_index = (
    STOCK_TYPE_INDICES[
        STOCK_TYPE
    ]
)


########################################
# REWRITTEN: Analysis Mode
########################################

def choose_analysis_mode():
    choices = {
        "1": "DAILY",
        "2": "INTRADAY",
        "3": "COMBINED",
        "daily": "DAILY",
        "intraday": "INTRADAY",
        "combined": "COMBINED",
    }

    while True:
        print("\nSelect analysis mode:")
        print("  1. Daily")
        print("  2. Intraday")
        print("  3. Combined")
        answer = input("Analysis mode [1/2/3]: ").strip().lower()

        if answer in choices:
            return choices[answer]

        print("Invalid selection. Enter 1, 2, or 3.")


ANALYSIS_MODE = choose_analysis_mode()

RESULT_TABLES = {
    "DAILY": f"{STOCK_TYPE} Passed Test Results",
    "INTRADAY": f"Intraday {STOCK_TYPE} Passed Test Results",
}

ACTIVE_ANALYSIS_TYPES = (
    ["DAILY", "INTRADAY"]
    if ANALYSIS_MODE == "COMBINED"
    else [ANALYSIS_MODE]
)


########################################
# SQL Helpers
########################################

def quote_sql_identifier(identifier):
    return (
        '"'
        + str(identifier).replace('"', '""')
        + '"'
    )


def dataframe_memory_mb(dataframe):
    return (
        dataframe
        .memory_usage(
            index=True,
            deep=True,
        )
        .sum()
        / (1024 ** 2)
    )


########################################
# Load Most Predictable Results
########################################

logger.info(
    "Loading passed results | mode=%s | stock type=%s",
    ANALYSIS_MODE,
    STOCK_TYPE,
)

result_parts = []

with sqlite3.connect(
    FINAL_RESULTS_DB
) as connection:

    available_tables = {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    }

    for analysis_type in ACTIVE_ANALYSIS_TYPES:

        result_table = RESULT_TABLES[
            analysis_type
        ]

        if result_table not in available_tables:
            raise ValueError(
                f"Missing {analysis_type} results table: "
                f"{result_table}"
            )

        result_part = pd.read_sql_query(
            f"""
            SELECT *
            FROM {quote_sql_identifier(result_table)}
            """,
            connection,
        )

        if result_part.empty:
            raise ValueError(
                f"Results table is empty: {result_table}"
            )

        result_part[
            "Analysis Type"
        ] = analysis_type

        result_parts.append(
            result_part
        )


test_results = pd.concat(
    result_parts,
    ignore_index=True,
    sort=False,
)

del result_parts


required_result_columns = {
    "Target",
    "Model",
    "Parameters",
    "Target Type",
    "Portfolio Target Type",
    "Horizon",
    "Quality Score",
    "Analysis Type",
}

missing_result_columns = (
    required_result_columns
    .difference(
        test_results.columns
    )
)

if missing_result_columns:
    raise ValueError(
        "Most Predictable Results table is missing: "
        + ", ".join(
            sorted(
                missing_result_columns
            )
        )
    )


########################################
# Load Selected Features
########################################

logger.info(
    "Loading selected features from %s",
    SELECTED_FEATURES_FILE,
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
        f"Selected_Features.txt has no line "
        f"for {STOCK_TYPE} at index "
        f"{stock_type_index}."
    )


selected_feature_line = (
    selected_feature_lines[
        stock_type_index
    ].strip()
)


if selected_feature_line == "":
    raise ValueError(
        f"Selected_Features.txt line "
        f"{stock_type_index} is empty "
        f"for {STOCK_TYPE}."
    )


selected_features = ast.literal_eval(
    selected_feature_line
)


if not isinstance(
    selected_features,
    dict,
):
    raise ValueError(
        "Selected_Features.txt line must "
        "contain a Target -> Features dictionary."
    )


########################################
# Build Selected Model Metadata
#
# Most Predictable Results is the source
# of truth for:
#
# Target
# Model
# Parameters
# Target Type
# Horizon
# Quality Score
########################################

selected_models_df = (
    test_results[
        test_results[
            "Target"
        ]
        .astype(str)
        .isin(
            selected_features.keys()
        )
    ]
    .copy()
)


# Baselines cannot be production models.
selected_models_df = (
    selected_models_df[
        ~selected_models_df[
            "Model"
        ]
        .astype(str)
        .str.contains(
            "Baseline",
            case=False,
            na=False,
        )
    ]
    .copy()
)


selected_models_df[
    "Target Type"
] = (
    selected_models_df[
        "Target Type"
    ]
    .astype(str)
    .str.upper()
    .str.strip()
)


selected_models_df[
    "Horizon"
] = pd.to_numeric(
    selected_models_df[
        "Horizon"
    ],
    errors="coerce",
)


selected_models_df[
    "Quality Score"
] = (
    pd.to_numeric(
        selected_models_df[
            "Quality Score"
        ],
        errors="coerce",
    )
    .clip(
        lower=0.0,
        upper=1.0,
    )
)


selected_models_df[
    "Parameters"
] = (
    selected_models_df[
        "Parameters"
    ]
    .where(
        selected_models_df[
            "Parameters"
        ].notna(),
        "{}",
    )
    .astype(str)
)


selected_models_df = (
    selected_models_df
    .dropna(
        subset=[
            "Target",
            "Model",
            "Horizon",
            "Quality Score",
        ]
    )
    .copy()
)


########################################
# REWRITTEN: Daily / Intraday Horizons
########################################

DAILY_HORIZONS = {
    1,
    5,
    20,
    60,
    120,
    252,
}

INTRADAY_HORIZONS = {
    1,
    5,
    15,
    60,
}

valid_horizon = (
    (
        selected_models_df[
            "Analysis Type"
        ].eq(
            "DAILY"
        )
        & selected_models_df[
            "Horizon"
        ].isin(
            DAILY_HORIZONS
        )
    )
    |
    (
        selected_models_df[
            "Analysis Type"
        ].eq(
            "INTRADAY"
        )
        & selected_models_df[
            "Horizon"
        ].isin(
            INTRADAY_HORIZONS
        )
    )
)

selected_models_df = (
    selected_models_df[
        valid_horizon
    ]
    .copy()
)


if selected_models_df.empty:
    raise ValueError(
        f"No {ANALYSIS_MODE.lower()} selected models remain."
    )


########################################
# One Production Model Per Target
########################################

sort_columns = [
    "Quality Score",
]

ascending = [
    False,
]

if (
    "Predictability Score"
    in selected_models_df.columns
):

    selected_models_df[
        "Predictability Score"
    ] = pd.to_numeric(
        selected_models_df[
            "Predictability Score"
        ],
        errors="coerce",
    )

    sort_columns.append(
        "Predictability Score"
    )

    ascending.append(
        False
    )


selected_models_df = (
    selected_models_df
    .sort_values(
        sort_columns,
        ascending=ascending,
    )
    .drop_duplicates(
        subset=[
            "Analysis Type",
            "Target",
        ],
        keep="first",
    )
    .reset_index(
        drop=True
    )
)


selected_models_df = (
    selected_models_df[
        [
            "Analysis Type",
            "Target",
            "Model",
            "Parameters",
            "Target Type",
            "Portfolio Target Type",
            "Horizon",
            "Quality Score",
        ]
    ]
    .copy()
)


########################################
# Target-Specific Feature Map
########################################

selected_targets = (
    selected_models_df[
        "Target"
    ]
    .astype(str)
    .tolist()
)


missing_feature_targets = [
    target
    for target in selected_targets
    if target not in selected_features
]


if missing_feature_targets:
    raise ValueError(
        "Selected_Features.txt does not contain "
        "feature definitions for:\n"
        + "\n".join(
            missing_feature_targets
        )
    )


target_features = {
    target: list(
        selected_features[
            target
        ]
    )
    for target in selected_targets
}


# Union only the features actually needed by the selected production models.
# This is the key memory-saving step: the full source table is never loaded.
required_features = list(
    dict.fromkeys(
        feature
        for target in selected_targets
        for feature in target_features[
            target
        ]
    )
)


logger.info(
    "Selected models ready | targets=%d | target types=%d | required features=%d",
    len(
        selected_models_df
    ),
    selected_models_df[
        "Portfolio Target Type"
    ].nunique(),
    len(
        required_features
    ),
)


########################################
# Inspect Features/Targets Source Table
########################################

logger.info(
    "Inspecting feature database | %s | table=%s",
    FEATURE_DATABASE,
    STOCK_TYPE,
)

with sqlite3.connect(
    FEATURE_DATABASE
) as metadata_connection:

    table_info = metadata_connection.execute(
        f"PRAGMA table_info({quote_sql_identifier(STOCK_TYPE)})"
    ).fetchall()


SOURCE_TABLE_COLUMNS = [
    row[1]
    for row in table_info
]

SOURCE_TABLE_COLUMN_SET = set(
    SOURCE_TABLE_COLUMNS
)


if not SOURCE_TABLE_COLUMNS:
    raise ValueError(
        f"Source table does not exist or has no columns: {STOCK_TYPE}"
    )


########################################
# Validate Base Source Columns
########################################

required_base_columns = {
    "Date",
    "Ticker",
    "Close",
}

missing_base_columns = (
    required_base_columns
    .difference(
        SOURCE_TABLE_COLUMN_SET
    )
)

if missing_base_columns:
    raise KeyError(
        f"Base columns missing from {STOCK_TYPE}: "
        + ", ".join(
            sorted(
                missing_base_columns
            )
        )
    )


logger.info(
    "Memory-safe SQL mode | full %s table will never be loaded",
    STOCK_TYPE,
)


########################################
# Load Research Dates ONLY
#
# First query only the Date column so the
# internal split can be decided without
# loading any feature matrix into pandas.
########################################

research_end_sql = RESEARCH_END.strftime(
    "%Y-%m-%d"
)


with sqlite3.connect(
    FEATURE_DATABASE
) as connection:

    research_dates_df = pd.read_sql_query(
        f"""
        SELECT DISTINCT {quote_sql_identifier('Date')} AS {quote_sql_identifier('Date')}
        FROM {quote_sql_identifier(STOCK_TYPE)}
        WHERE date({quote_sql_identifier('Date')}) <= date(?)
        ORDER BY {quote_sql_identifier('Date')}
        """,
        connection,
        params=[
            research_end_sql
        ],
    )


research_dates = (
    pd.to_datetime(
        research_dates_df[
            "Date"
        ]
    )
    .dropna()
    .drop_duplicates()
    .sort_values()
    .reset_index(
        drop=True
    )
)


del research_dates_df


if research_dates.empty:
    raise ValueError(
        "No research dates are available "
        "before RESEARCH_END."
    )


########################################
# Internal Horizon Validation Split
#
# Earlier data:
#     model fitting
#
# Purge gap:
#     longest selected target horizon
#
# Final data up to 2023-09-30:
#     Horizon Score optimisation
########################################

validation_start_index = int(
    len(
        research_dates
    )
    * (
        1.0
        - HORIZON_VALIDATION_FRACTION
    )
)


if (
    validation_start_index <= 0
    or validation_start_index >= len(
        research_dates
    )
):
    raise ValueError(
        "Invalid Horizon validation split."
    )


max_selected_horizon = int(
    selected_models_df[
        "Horizon"
    ].max()
)


fit_end_index = (
    validation_start_index
    - max_selected_horizon
)


if fit_end_index <= 0:
    raise ValueError(
        "Not enough research dates for "
        f"a {HORIZON_VALIDATION_FRACTION:.0%} "
        "Horizon validation period plus a "
        f"{max_selected_horizon}-trading-day purge."
    )


fit_start = (
    research_dates.iloc[0]
)

fit_end = (
    research_dates.iloc[
        fit_end_index - 1
    ]
)

validation_start = (
    research_dates.iloc[
        validation_start_index
    ]
)

validation_end = (
    research_dates.iloc[-1]
)


logger.info(
    "Research fit period | %s to %s",
    fit_start.date(),
    fit_end.date(),
)

logger.info(
    "Purged %d trading dates before Horizon validation",
    max_selected_horizon,
)

logger.info(
    "Horizon validation period | %s to %s",
    validation_start.date(),
    validation_end.date(),
)


########################################
# REWRITTEN: Per-Frequency Purge Windows
########################################

RESEARCH_SPLITS = {}

for analysis_type in ACTIVE_ANALYSIS_TYPES:

    analysis_models = (
        selected_models_df[
            selected_models_df[
                "Analysis Type"
            ].eq(
                analysis_type
            )
        ]
    )

    analysis_max_horizon = int(
        analysis_models[
            "Horizon"
        ].max()
    )

    analysis_fit_end_index = (
        validation_start_index
        - analysis_max_horizon
    )

    if analysis_fit_end_index <= 0:
        raise ValueError(
            f"Not enough observations for the "
            f"{analysis_type} validation period "
            f"and a {analysis_max_horizon}-period purge."
        )

    RESEARCH_SPLITS[
        analysis_type
    ] = {
        "fit_start": research_dates.iloc[0],
        "fit_end": research_dates.iloc[
            analysis_fit_end_index - 1
        ],
        "validation_start": research_dates.iloc[
            validation_start_index
        ],
        "validation_end": research_dates.iloc[-1],
        "max_horizon": analysis_max_horizon,
    }

    logger.info(
        "%s split | fit=%s to %s | purge=%d | validation=%s to %s",
        analysis_type,
        RESEARCH_SPLITS[
            analysis_type
        ][
            "fit_start"
        ],
        RESEARCH_SPLITS[
            analysis_type
        ][
            "fit_end"
        ],
        analysis_max_horizon,
        RESEARCH_SPLITS[
            analysis_type
        ][
            "validation_start"
        ],
        RESEARCH_SPLITS[
            analysis_type
        ][
            "validation_end"
        ],
    )


########################################
# Target-By-Target Memory-Safe Loader
#
# This mirrors the earlier model-fitting
# approach: each selected target loads only
# its own selected features, then the large
# feature dataframe is discarded after the
# model has generated predictions.
########################################

def target_sql_columns(
    target,
    features,
):

    columns = [
        "Date",
        "Ticker",
        "Close",
    ]

    if "Return" in SOURCE_TABLE_COLUMN_SET:
        columns.append(
            "Return"
        )

    columns.append(
        target
    )

    columns.extend(
        features
    )

    columns = list(
        dict.fromkeys(
            columns
        )
    )

    missing_columns = [
        column
        for column in columns
        if column not in SOURCE_TABLE_COLUMN_SET
    ]

    if missing_columns:
        raise KeyError(
            f"{target} | Columns missing from {STOCK_TYPE}: "
            f"{missing_columns}"
        )

    return columns


def load_target_period(
    target,
    features,
    start_date,
    end_date,
    split,
):

    columns = target_sql_columns(
        target,
        features,
    )

    sql_columns = ", ".join(
        quote_sql_identifier(
            column
        )
        for column in columns
    )

    query = (
        f"SELECT {sql_columns} "
        f"FROM {quote_sql_identifier(STOCK_TYPE)} "
        f"WHERE date({quote_sql_identifier('Date')}) >= date(?) "
        f"AND date({quote_sql_identifier('Date')}) <= date(?)"
    )

    start_value = pd.Timestamp(
        start_date
    ).strftime(
        "%Y-%m-%d"
    )

    end_value = pd.Timestamp(
        end_date
    ).strftime(
        "%Y-%m-%d"
    )

    logger.info(
        "%s | %s | loading %d/%d source columns",
        target,
        split,
        len(
            columns
        ),
        len(
            SOURCE_TABLE_COLUMNS
        ),
    )

    with sqlite3.connect(
        FEATURE_DATABASE
    ) as connection:

        dataframe = pd.read_sql_query(
            query,
            connection,
            params=[
                start_value,
                end_value,
            ],
        )

    if dataframe.empty:
        raise ValueError(
            f"{target} | No rows loaded for {split}."
        )

    dataframe[
        "Date"
    ] = pd.to_datetime(
        dataframe[
            "Date"
        ]
    )

    dataframe = (
        dataframe
        .sort_values(
            [
                "Date",
                "Ticker",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    if "Return" not in dataframe.columns:
        dataframe[
            "Return"
        ] = (
            dataframe
            .groupby(
                "Ticker",
                sort=False,
            )[
                "Close"
            ]
            .pct_change()
        )

    dataframe[
        "Split"
    ] = split

    logger.info(
        "%s | %s | loaded %d rows x %d columns | %.1f MB",
        target,
        split,
        len(
            dataframe
        ),
        len(
            dataframe.columns
        ),
        dataframe_memory_mb(
            dataframe
        ),
    )

    return dataframe


########################################
# Horizon Validation Benchmark
#
# Features/targets come entirely from the
# pre-built SQLite table. Only ^GSPC is
# downloaded here because it is benchmark
# data, not a model feature panel.
########################################

logger.info(
    "Downloading S&P 500 benchmark only | %s to %s",
    validation_start.date(),
    validation_end.date(),
)

benchmark_download = yf.download(
    "^GSPC",
    start=validation_start.strftime(
        "%Y-%m-%d"
    ),
    end=(
        validation_end
        + pd.Timedelta(
            days=1
        )
    ).strftime(
        "%Y-%m-%d"
    ),
    auto_adjust=True,
    progress=False,
    multi_level_index=False,
)


if benchmark_download.empty:
    raise ValueError(
        "No S&P 500 benchmark data returned."
    )


benchmark_close = benchmark_download[
    "Close"
]

if isinstance(
    benchmark_close,
    pd.DataFrame,
):
    benchmark_close = (
        benchmark_close.iloc[
            :,
            0,
        ]
    )


market_df = pd.DataFrame(
    {
        "Date": pd.to_datetime(
            benchmark_close.index
        ),
        "Close": pd.to_numeric(
            benchmark_close.to_numpy(),
            errors="coerce",
        ),
    }
)

market_df[
    "Return"
] = market_df[
    "Close"
].pct_change()

market_df = (
    market_df
    .dropna(
        subset=[
            "Close"
        ]
    )
    .sort_values(
        "Date"
    )
    .reset_index(
        drop=True
    )
)


market_results = benchmark_metrics(
    market_df
)


logger.info(
    "Benchmark ready | return=%.4f | sharpe=%.4f | max_dd=%.4f | avg_dd=%.4f",
    market_results[
        "Return"
    ],
    market_results[
        "Sharpe Ratio"
    ],
    market_results[
        "Max Drawdown"
    ],
    market_results[
        "Average Drawdown"
    ],
)


HORIZON_SCORE_RANGES = {

    ########################################
    # Signed Returns / Alpha
    ########################################

    "ALPHA": {
        "1m": (0.00, 0.20),
        "5m": (0.00, 0.25),
        "15m": (0.10, 0.30),
        "60m": (0.15, 0.40),

        "1d": (0.20, 0.55),
        "5d": (0.45, 0.75),
        "20d": (0.70, 0.95),
        "60d": (0.85, 1.00),
        "120d": (0.70, 1.00),
        "252d": (0.50, 0.80),
    },

    "RELATIVE_ALPHA": {
        "1m": (0.00, 0.15),
        "5m": (0.00, 0.20),
        "15m": (0.00, 0.25),
        "60m": (0.10, 0.35),

        "1d": (0.20, 0.50),
        "5d": (0.45, 0.75),
        "20d": (0.75, 1.00),
        "60d": (0.85, 1.00),
        "120d": (0.70, 1.00),
        "252d": (0.55, 0.85),
    },

    "RISK_ADJUSTED_ALPHA": {
        "1m": (0.00, 0.20),
        "5m": (0.00, 0.25),
        "15m": (0.05, 0.30),
        "60m": (0.15, 0.45),

        "1d": (0.30, 0.60),
        "5d": (0.55, 0.85),
        "20d": (0.80, 1.00),
        "60d": (0.85, 1.00),
        "120d": (0.65, 0.95),
        "252d": (0.45, 0.70),
    },

    "CROSS_SECTION_ALPHA": {
        "1m": (0.00, 0.15),
        "5m": (0.00, 0.20),
        "15m": (0.00, 0.25),
        "60m": (0.10, 0.35),

        "1d": (0.25, 0.55),
        "5d": (0.50, 0.80),
        "20d": (0.85, 1.00),
        "60d": (0.80, 1.00),
        "120d": (0.55, 0.90),
        "252d": (0.4, 0.65),
    },

    "CROSS_SECTION_DOWNSIDE": {
        "1m": (0.00, 0.20),
        "5m": (0.00, 0.25),
        "15m": (0.10, 0.35),
        "60m": (0.20, 0.50),

        "1d": (0.40, 0.70),
        "5d": (0.65, 0.95),
        "20d": (0.85, 1.00),
        "60d": (0.80, 1.00),
        "120d": (0.55, 0.85),
        "252d": (0.35, 0.60),
    },


    ########################################
    # Direction / Threshold Classification
    ########################################

    "DIRECTION": {
        "1m": (0.00, 0.40),
        "5m": (0.10, 0.45),
        "15m": (0.20, 0.55),
        "60m": (0.40, 0.70),

        "1d": (0.60, 0.90),
        "5d": (0.85, 1.00),
        "20d": (0.75, 1.00),
        "60d": (0.45, 0.80),
        "120d": (0.20, 0.55),
        "252d": (0.00, 0.35),
    },

    "DIRECTION_MULTICLASS": {
        "1m": (0.00, 0.35),
        "5m": (0.10, 0.45),
        "15m": (0.20, 0.55),
        "60m": (0.35, 0.70),

        "1d": (0.55, 0.90),
        "5d": (0.80, 1.00),
        "20d": (0.85, 1.00),
        "60d": (0.65, 0.95),
        "120d": (0.25, 0.60),
        "252d": (0.00, 0.35),
    },

    "ALPHA_BINARY": {
        "1m": (0.00, 0.50),
        "5m": (0.20, 0.60),
        "15m": (0.35, 0.70),
        "60m": (0.50, 0.85),

        "1d": (0.65, 1.00),
        "5d": (0.85, 1.00),
        "20d": (0.70, 1.00),
        "60d": (0.40, 0.75),
        "120d": (0.15, 0.50),
        "252d": (0.00, 0.35),
    },

    "BARRIER_ALPHA": {
        "1m": (0.00, 0.55),
        "5m": (0.25, 0.65),
        "15m": (0.40, 0.75),
        "60m": (0.50, 0.85),

        "1d": (0.65, 1.00),
        "5d": (0.85, 1.00),
        "20d": (0.75, 1.00),
        "60d": (0.45, 0.80),
        "120d": (0.20, 0.55),
        "252d": (0.00, 0.35),
    },


    ########################################
    # Volatility / Absolute Movement
    ########################################

    "VOLATILITY": {
        "1m": (0.00, 0.45),
        "5m": (0.20, 0.55),
        "15m": (0.30, 0.65),
        "60m": (0.45, 0.80),

        "1d": (0.60, 0.95),
        "5d": (0.85, 1.00),
        "20d": (0.80, 1.00),
        "60d": (0.60, 0.95),
        "120d": (0.40, 0.75),
        "252d": (0.25, 0.55),
    },

    "ABSOLUTE_MOVE": {
        "1m": (0.20, 0.55),
        "5m": (0.30, 0.65),
        "15m": (0.40, 0.75),
        "60m": (0.55, 0.85),

        "1d": (0.65, 0.95),
        "5d": (0.85, 1.00),
        "20d": (0.85, 1.00),
        "60d": (0.65, 0.95),
        "120d": (0.35, 0.70),
        "252d": (0.2, 0.45),
    },

    "UPSIDE_VOLATILITY": {
        "1m": (0.10, 0.45),
        "5m": (0.20, 0.55),
        "15m": (0.30, 0.65),
        "60m": (0.45, 0.80),

        "1d": (0.60, 0.95),
        "5d": (0.80, 1.00),
        "20d": (0.85, 1.00),
        "60d": (0.65, 0.95),
        "120d": (0.35, 0.70),
        "252d": (0.25, 0.50),
    },

    "DOWNSIDE_VOLATILITY": {
        "1m": (0.15, 0.50),
        "5m": (0.25, 0.60),
        "15m": (0.35, 0.70),
        "60m": (0.50, 0.85),

        "1d": (0.65, 1.00),
        "5d": (0.85, 1.00),
        "20d": (0.85, 1.00),
        "60d": (0.70, 1.00),
        "120d": (0.40, 0.75),
        "252d": (0.2, 0.55),
    },

    "VOLATILITY_ASYMMETRY": {
        "1m": (0.00, 0.35),
        "5m": (0.10, 0.45),
        "15m": (0.20, 0.55),
        "60m": (0.35, 0.70),

        "1d": (0.50, 0.85),
        "5d": (0.70, 1.00),
        "20d": (0.85, 1.00),
        "60d": (0.75, 1.00),
        "120d": (0.40, 0.75),
        "252d": (0.25, 0.50),
    },

    "VOLATILITY_EVENT": {
        "1m": (0.45, 0.80),
        "5m": (0.55, 0.90),
        "15m": (0.70, 1.00),
        "60m": (0.80, 1.00),

        "1d": (0.80, 1.00),
        "5d": (0.60, 0.95),
        "20d": (0.35, 0.70),
        "60d": (0.10, 0.50),
        "120d": (0.00, 0.35),
        "252d": (0.00, 0.20),
    },


    ########################################
    # Downside / Tail Risk
    ########################################

    "DOWNSIDE": {
        "1m": (0.20, 0.55),
        "5m": (0.30, 0.65),
        "15m": (0.40, 0.75),
        "60m": (0.55, 0.90),

        "1d": (0.70, 1.00),
        "5d": (0.85, 1.00),
        "20d": (0.75, 1.00),
        "60d": (0.50, 0.85),
        "120d": (0.30, 0.65),
        "252d": (0.15, 0.45),
    },

    "TAIL_RISK": {
        "1m": (0.35, 0.70),
        "5m": (0.45, 0.80),
        "15m": (0.55, 0.90),
        "60m": (0.65, 1.00),

        "1d": (0.85, 1.00),
        "5d": (0.80, 1.00),
        "20d": (0.60, 0.90),
        "60d": (0.30, 0.65),
        "120d": (0.10, 0.45),
        "252d": (0.00, 0.30),
    },

    "TAIL_EVENT": {
        "1m": (0.80, 1.00),
        "5m": (0.85, 1.00),
        "15m": (0.80, 1.00),
        "60m": (0.65, 1.00),

        "1d": (0.55, 0.90),
        "5d": (0.30, 0.65),
        "20d": (0.00, 0.40),
        "60d": (0.00, 0.25),
        "120d": (0.00, 0.15),
        "252d": (0.00, 0.10),
    },

    "UPSIDE_RISK": {
        "1m": (0.15, 0.50),
        "5m": (0.25, 0.60),
        "15m": (0.35, 0.70),
        "60m": (0.50, 0.85),

        "1d": (0.65, 1.00),
        "5d": (0.85, 1.00),
        "20d": (0.70, 1.00),
        "60d": (0.40, 0.75),
        "120d": (0.20, 0.55),
        "252d": (0.00, 0.35),
    },

    "UPSIDE_EVENT": {
        "1m": (0.80, 1.00),
        "5m": (0.85, 1.00),
        "15m": (0.80, 1.00),
        "60m": (0.65, 1.00),

        "1d": (0.50, 0.85),
        "5d": (0.25, 0.60),
        "20d": (0.00, 0.35),
        "60d": (0.00, 0.20),
        "120d": (0.00, 0.15),
        "252d": (0.00, 0.10),
    },


    ########################################
    # Excursions
    ########################################

    "UPSIDE_EXCURSION": {
        "1m": (0.50, 0.85),
        "5m": (0.60, 0.95),
        "15m": (0.70, 1.00),
        "60m": (0.75, 1.00),

        "1d": (0.75, 1.00),
        "5d": (0.85, 1.00),
        "20d": (0.70, 1.00),
        "60d": (0.45, 0.80),
        "120d": (0.15, 0.50),
        "252d": (0.00, 0.30),
    },

    "DOWNSIDE_EXCURSION": {
        "1m": (0.55, 0.90),
        "5m": (0.65, 1.00),
        "15m": (0.75, 1.00),
        "60m": (0.80, 1.00),

        "1d": (0.80, 1.00),
        "5d": (0.85, 1.00),
        "20d": (0.75, 1.00),
        "60d": (0.50, 0.85),
        "120d": (0.20, 0.55),
        "252d": (0.00, 0.35),
    },

    "TIME_TO_UPSIDE_EXCURSION": {
        "1m": (0.00, 0.40),
        "5m": (0.15, 0.50),
        "15m": (0.30, 0.65),
        "60m": (0.50, 0.80),

        "1d": (0.50, 0.80),
        "5d": (0.65, 0.95),
        "20d": (0.80, 1.00),
        "60d": (0.70, 1.00),
        "120d": (0.30, 0.65),
        "252d": (0.00, 0.35),
    },

    "TIME_TO_DOWNSIDE_EXCURSION": {
        "1m": (0.00, 0.45),
        "5m": (0.20, 0.55),
        "15m": (0.35, 0.70),
        "60m": (0.55, 0.85),

        "1d": (0.55, 0.85),
        "5d": (0.70, 1.00),
        "20d": (0.85, 1.00),
        "60d": (0.75, 1.00),
        "120d": (0.35, 0.70),
        "252d": (0.10, 0.40),
    },


    ########################################
    # Recovery / Reversal
    ########################################

    "RECOVERY": {
        "1m": (0.80, 1.00),
        "5m": (0.85, 1.00),
        "15m": (0.80, 1.00),
        "60m": (0.65, 1.00),

        "1d": (0.45, 0.80),
        "5d": (0.15, 0.50),
        "20d": (0.00, 0.30),
        "60d": (0.00, 0.15),
        "120d": (0.00, 0.10),
        "252d": (0.00, 0.05),
    },

    "REVERSAL": {
        "1m": (0.80, 1.00),
        "5m": (0.85, 1.00),
        "15m": (0.80, 1.00),
        "60m": (0.65, 1.00),

        "1d": (0.45, 0.80),
        "5d": (0.15, 0.50),
        "20d": (0.00, 0.30),
        "60d": (0.00, 0.15),
        "120d": (0.00, 0.10),
        "252d": (0.00, 0.05),
    },


    ########################################
    # State / Dependence
    ########################################

    "REGIME": {
        "1m": (0.00, 0.25),
        "5m": (0.00, 0.30),
        "15m": (0.00, 0.35),
        "60m": (0.10, 0.45),

        "1d": (0.30, 0.65),
        "5d": (0.55, 0.90),
        "20d": (0.85, 1.00),
        "60d": (0.80, 1.00),
        "120d": (0.60, 0.95),
        "252d": (0.50, 0.75),
    },

    "CORRELATION": {
        "1m": (0.00, 0.25),
        "5m": (0.00, 0.30),
        "15m": (0.05, 0.40),
        "60m": (0.15, 0.50),

        "1d": (0.35, 0.70),
        "5d": (0.60, 0.95),
        "20d": (0.85, 1.00),
        "60d": (0.80, 1.00),
        "120d": (0.60, 0.95),
        "252d": (0.50, 0.75),
    },

    "COVARIANCE": {
        "1m": (0.00, 0.25),
        "5m": (0.00, 0.30),
        "15m": (0.05, 0.40),
        "60m": (0.15, 0.50),

        "1d": (0.35, 0.70),
        "5d": (0.60, 0.95),
        "20d": (0.85, 1.00),
        "60d": (0.80, 1.00),
        "120d": (0.60, 0.95),
        "252d": (0.50, 0.75),
    },


    ########################################
    # Market Structure / Execution
    ########################################

    "LIQUIDITY": {
        "1m": (0.60, 0.95),
        "5m": (0.75, 1.00),
        "15m": (0.85, 1.00),
        "60m": (0.80, 1.00),

        "1d": (0.50, 0.85),
        "5d": (0.20, 0.55),
        "20d": (0.00, 0.35),
        "60d": (0.00, 0.20),
        "120d": (0.00, 0.10),
        "252d": (0.00, 0.05),
    },

    "EXECUTION": {
        "1m": (0.85, 1.00),
        "5m": (0.85, 1.00),
        "15m": (0.80, 1.00),
        "60m": (0.60, 0.95),

        "1d": (0.20, 0.55),
        "5d": (0.00, 0.30),
        "20d": (0.00, 0.15),
        "60d": (0.00, 0.10),
        "120d": (0.00, 0.05),
        "252d": (0.00, 0.00),
    },

    "MARKET_IMPACT": {
        "1m": (0.85, 1.00),
        "5m": (0.85, 1.00),
        "15m": (0.80, 1.00),
        "60m": (0.60, 0.95),

        "1d": (0.20, 0.55),
        "5d": (0.00, 0.30),
        "20d": (0.00, 0.15),
        "60d": (0.00, 0.10),
        "120d": (0.00, 0.05),
        "252d": (0.00, 0.00),
    },
}

########################################
# REWRITTEN: Daily / Intraday Horizon Helpers
########################################

HORIZON_STEP = 0.05
HORIZON_INDEX = 2


def horizon_key(row):

    suffix = (
        "m"
        if str(
            row.get(
                "Analysis Type",
                "DAILY",
            )
        ).upper()
        == "INTRADAY"
        else "d"
    )

    return (
        f"{int(row['Horizon'])}{suffix}"
    )


selected_models_df[
    "Horizon Key"
] = selected_models_df.apply(
    horizon_key,
    axis=1,
)


########################################
# Validate Stored Target Type / Horizon
# Against Horizon Score Ranges
########################################

unsupported_parameters = []

for _, row in selected_models_df.iterrows():

    target_type = (
        str(
            row[
                "Portfolio Target Type"
            ]
        )
        .upper()
        .strip()
    )

    horizon = row[
        "Horizon Key"
    ]

    if (
        target_type
        not in HORIZON_SCORE_RANGES
        or horizon
        not in HORIZON_SCORE_RANGES[
            target_type
        ]
    ):

        unsupported_parameters.append(
            (
                str(
                    row[
                        "Target"
                    ]
                ),
                target_type,
                horizon,
            )
        )


if unsupported_parameters:

    raise ValueError(
        "Selected models contain unsupported "
        "Target Type / Horizon pairs:\n"
        + "\n".join(
            f"{target}: {target_type} {horizon}"
            for (
                target,
                target_type,
                horizon,
            )
            in unsupported_parameters
        )
    )


########################################
# Keep Only Horizon Parameters Actually
# Used By Selected Models
########################################

active_parameters = (
    selected_models_df[
        [
            "Portfolio Target Type",
            "Horizon Key",
        ]
    ]
    .drop_duplicates()
    .sort_values(
        [
            "Portfolio Target Type",
            "Horizon Key",
        ]
    )
    .reset_index(
        drop=True
    )
)


ACTIVE_HORIZON_SCORE_RANGES = {}

for _, row in active_parameters.iterrows():

    target_type = row[
        "Portfolio Target Type"
    ]

    horizon = row[
        "Horizon Key"
    ]

    ACTIVE_HORIZON_SCORE_RANGES.setdefault(
        target_type,
        {},
    )[
        horizon
    ] = (
        HORIZON_SCORE_RANGES[
            target_type
        ][
            horizon
        ]
    )


logger.info(
    "Active Horizon Score parameters=%d",
    len(
        active_parameters
    ),
)


########################################
# Initial Horizon Score
########################################

def horizon_values(
    minimum,
    maximum,
):

    if minimum == maximum:
        return [
            round(
                minimum,
                2,
            )
        ]

    return list(
        np.round(
            np.arange(
                minimum
                + HORIZON_STEP,
                maximum
                + HORIZON_STEP / 2,
                HORIZON_STEP,
            ),
            2,
        )
    )


def get_horizon_score(row):

    target_type = (
        str(
            row[
                "Portfolio Target Type"
            ]
        )
        .upper()
        .strip()
    )

    if "Horizon Key" in row.index:
        horizon = row["Horizon Key"]
    else:
        horizon = horizon_key(row)

    minimum, maximum = (
        ACTIVE_HORIZON_SCORE_RANGES[
            target_type
        ][
            horizon
        ]
    )

    values = horizon_values(
        minimum,
        maximum,
    )

    index = min(
        HORIZON_INDEX,
        len(
            values
        )
        - 1,
    )

    return float(
        values[
            index
        ]
    )


########################################
# Fit Models + Generate Horizon
# Validation Predictions ONCE
########################################

# Required by the model-preparation interface.
# The value is replaced immediately after
# predictions are generated.
selected_models_df[
    "Horizon Score"
] = 1.0


logger.info(
    "Fitting selected models on internal TRAIN "
    "and generating Horizon-validation predictions"
)


prediction_parts = []


for model_number, (_, model_row) in enumerate(
    selected_models_df.iterrows(),
    start=1,
):

    target = str(
        model_row[
            "Target"
        ]
    )

    analysis_type = str(
        model_row[
            "Analysis Type"
        ]
    )

    research_split = RESEARCH_SPLITS[
        analysis_type
    ]

    features = target_features[
        target
    ]

    logger.info(
        "[%d/%d] %s | fitting with %d selected features",
        model_number,
        len(
            selected_models_df
        ),
        target,
        len(
            features
        ),
    )

    target_train_df = load_target_period(
        target=target,
        features=features,
        start_date=research_split[
            "fit_start"
        ],
        end_date=research_split[
            "fit_end"
        ],
        split="TRAIN",
    )

    target_validation_df = load_target_period(
        target=target,
        features=features,
        start_date=research_split[
            "validation_start"
        ],
        end_date=research_split[
            "validation_end"
        ],
        split="BACKTEST",
    )

    target_model_data = pd.concat(
        [
            target_train_df,
            target_validation_df,
        ],
        ignore_index=True,
    )

    one_model_df = pd.DataFrame(
        [
            model_row.to_dict()
        ]
    )

    prepared = create_models_and_predictions(
        dataframe=target_model_data,
        selected_models_df=one_model_df,
        model_features={
            target: features,
        },
    )

    target_predictions = (
        prepared[
            "predictions"
        ]
        .copy()
    )

    target_predictions[
        "Analysis Type"
    ] = model_row[
        "Analysis Type"
    ]

    target_predictions["Portfolio Target Type"] = model_row["Portfolio Target Type"]

    prediction_parts.append(
        target_predictions
    )

    logger.info(
        "[%d/%d] %s | predictions complete | rows=%d",
        model_number,
        len(
            selected_models_df
        ),
        target,
        len(
            target_predictions
        ),
    )

    # Keep only the compact prediction result. The feature matrix for this
    # target is no longer needed and is explicitly released before the next
    # SQL query.
    del target_train_df
    del target_validation_df
    del target_model_data
    del one_model_df
    del prepared
    del target_predictions

    gc.collect()


if not prediction_parts:
    raise ValueError(
        "No selected target predictions were generated."
    )


predictions_df = pd.concat(
    prediction_parts,
    ignore_index=True,
)


del prediction_parts


logger.info(
    "Prediction generation complete | rows=%d | targets=%d | %.1f MB",
    len(
        predictions_df
    ),
    predictions_df[
        "Target"
    ].nunique(),
    dataframe_memory_mb(
        predictions_df
    ),
)

predictions_df[
    "Horizon Key"
] = predictions_df.apply(
    horizon_key,
    axis=1,
)


predictions_df[
    "Signal"
] = (
    pd.to_numeric(
        predictions_df[
            "Signal"
        ],
        errors="coerce",
    )
    .replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    )
    .fillna(
        0.0
    )
)

predictions_df = predictions_df.dropna()


predictions_df[
    "Adjusted Signal"
] = (
    predictions_df[
        "Signal"
    ]
    * predictions_df[
        "Quality Score"
    ]
)

predictions_df = predictions_df[['Date', 'Ticker', 'Return', 'Portfolio Target Type', 'Horizon Key', 'Adjusted Signal']]

predictions_df = (
    predictions_df
    .groupby(
        [
            "Date",
            "Ticker",
            "Portfolio Target Type",
            "Horizon Key"
        ],
        as_index=False,
    )
    .agg(
        Return=(
            "Return",
            "first",
        ),
        Signal=(
            "Adjusted Signal",
            "mean",
        ),
    )
)

pass

predictions_df[
    "Horizon Score"
] = predictions_df.apply(
    get_horizon_score,
    axis=1,
)

initial_predictions_df = predictions_df


logger.info(
    "Initial Horizon Scores assigned | missing=%d",
    predictions_df[
        "Horizon Score"
    ].isna().sum(),
)


########################################
# Backtest Quality
########################################

def backtest_quality(df):

    results = run_portfolio_backtest_from_predictions(
        predictions_df=df,
        rebalance_every=1,
        max_weight=0.30,
        concentration_penalty=0.10,
        trading_fee=0.00,
    )

    delta_return = (
        results[
            "Strategy Return"
        ]
        - market_results[
            "Return"
        ]
    )

    delta_sharpe = (
        results[
            "Sharpe Ratio"
        ]
        - market_results[
            "Sharpe Ratio"
        ]
    )

    delta_max_drawdown = (
        abs(
            market_results[
                "Max Drawdown"
            ]
        )
        - abs(
            results[
                "Max Drawdown"
            ]
        )
    )

    delta_average_drawdown = (
        abs(
            market_results[
                "Average Drawdown"
            ]
        )
        - abs(
            results[
                "Average Drawdown"
            ]
        )
    )

    return (
        0.40
        * np.tanh(
            delta_sharpe
            / 0.50
        )
        + 0.30
        * np.tanh(
            delta_return
            / 0.10
        )
        + 0.20
        * np.tanh(
            delta_max_drawdown
            / 0.10
        )
        + 0.10
        * np.tanh(
            delta_average_drawdown
            / 0.05
        )
    )


########################################
# Build Candidate Horizon Score Grid
########################################

STEP = HORIZON_STEP
FROZEN_INDEX = HORIZON_INDEX


HORIZON_SCORES = {
    target_type: {
        horizon: horizon_values(
            minimum,
            maximum,
        )
        for (
            horizon,
            (
                minimum,
                maximum,
            )
        )
        in horizons.items()
    }
    for (
        target_type,
        horizons,
    )
    in ACTIVE_HORIZON_SCORE_RANGES.items()
}


logger.info(
    "Candidate Horizon Score grid built | parameters=%d | configurations=%d",
    sum(
        len(
            horizons
        )
        for horizons
        in HORIZON_SCORES.values()
    ),
    math.prod(
        len(
            values
        )
        for horizons
        in HORIZON_SCORES.values()
        for values
        in horizons.values()
    ),
)


########################################
# Set Every Score To Frozen Index
########################################

frozen_df = (
    predictions_df.copy()
)


for (
    target_type,
    horizons,
) in HORIZON_SCORES.items():

    for (
        horizon,
        values,
    ) in horizons.items():

        index = min(
            FROZEN_INDEX,
            len(
                values
            )
            - 1,
        )

        frozen_df.loc[
            (
                frozen_df[
                    "Portfolio Target Type"
                ]
                == target_type
            )
            &
            (
                frozen_df[
                    "Horizon Key"
                ]
                == horizon
            ),
            "Horizon Score",
        ] = values[
            index
        ]


########################################
# First Sensitivity Screen
########################################

sensitivity_results = []


for (
    target_type,
    horizons,
) in list(
    HORIZON_SCORES.items()
):

    for (
        horizon,
        values,
    ) in list(
        horizons.items()
    ):

        qualities = []

        logger.info(
            "Sensitivity screen | %s %s | candidates=%d",
            target_type,
            horizon,
            len(
                values
            ),
        )


        for score in values:

            test_df = (
                frozen_df.copy()
            )

            test_df.loc[
                (
                    test_df[
                        "Portfolio Target Type"
                    ]
                    == target_type
                )
                &
                (
                    test_df[
                        "Horizon Key"
                    ]
                    == horizon
                ),
                "Horizon Score",
            ] = score


            BQ = backtest_quality(
                test_df
            )

            qualities.append(
                BQ
            )


        mean_quality = float(
            np.mean(
                qualities
            )
        )

        range_quality = float(np.max(qualities)) - float(np.min(qualities))

        sensitivity_results.append(
            {
                "Portfolio Target Type":
                    target_type,

                "Horizon":
                    horizon,

                "Mean BQ":
                    mean_quality,

                "Range BQ":
                    range_quality,

                "Frozen":
                    range_quality < 0.003,
            }
        )


        logger.info(
            "Sensitivity result | %s %s | Mean BQ=%.6f | Range BQ=%.6f | Frozen=%s",
            target_type,
            horizon,
            mean_quality,
            range_quality,
            range_quality < 0.003,
        )


        ####################################
        # Freeze Insensitive Parameters
        ####################################

        if range_quality < 0.003:

            index = min(
                FROZEN_INDEX,
                len(
                    values
                )
                - 1,
            )

            HORIZON_SCORES[
                target_type
            ][
                horizon
            ] = [
                values[
                    index
                ]
            ]

            logger.info(
                "Frozen | %s %s -> %.2f",
                target_type,
                horizon,
                values[
                    index
                ],
            )


sensitivity_results = pd.DataFrame(
    sensitivity_results
)


logger.info(
    "First sensitivity screen complete | frozen=%d | remaining variable=%d",
    sum(
        len(
            values
        )
        == 1
        for horizons
        in HORIZON_SCORES.values()
        for values
        in horizons.values()
    ),
    sum(
        len(
            values
        )
        > 1
        for horizons
        in HORIZON_SCORES.values()
        for values
        in horizons.values()
    ),
)


########################################
# Random Context Screen
########################################

def random_screen(
    iterations,
    threshold,
):

    global HORIZON_SCORES

    results = []

    logger.info(
        "Random screen started | iterations=%d | threshold=%.3f",
        iterations,
        threshold,
    )


    parameters = [
        (
            target_type,
            horizon,
        )
        for (
            target_type,
            horizons,
        )
        in HORIZON_SCORES.items()
        for horizon
        in horizons
    ]


    for (
        target_type,
        horizon,
    ) in parameters:

        values = (
            HORIZON_SCORES[
                target_type
            ][
                horizon
            ]
            .copy()
        )


        ####################################
        # Already Frozen
        ####################################

        if len(
            values
        ) == 1:

            logger.debug(
                "Skipping frozen parameter | %s %s -> %s",
                target_type,
                horizon,
                values,
            )

            continue


        logger.info(
            "Random screen parameter | %s %s | candidates=%d | backgrounds=%d",
            target_type,
            horizon,
            len(
                values
            ),
            iterations,
        )


        ####################################
        # Same Random Backgrounds For All
        # Candidates Of This Parameter
        ####################################

        random_iterations = []


        for _ in range(
            iterations
        ):

            configuration = {}

            for (
                other_type,
                other_horizons,
            ) in HORIZON_SCORES.items():

                configuration[
                    other_type
                ] = {}

                for (
                    other_horizon,
                    other_values,
                ) in other_horizons.items():

                    configuration[
                        other_type
                    ][
                        other_horizon
                    ] = float(
                        np.random.choice(
                            other_values
                        )
                    )


            random_iterations.append(
                configuration
            )


        quality_by_index = {
            index: []
            for index
            in range(
                len(
                    values
                )
            )
        }


        for configuration in random_iterations:

            for (
                index,
                score,
            ) in enumerate(
                values
            ):

                test_df = (
                    predictions_df.copy()
                )


                ################################
                # Apply Random Background
                ################################

                for (
                    config_type,
                    config_horizons,
                ) in configuration.items():

                    for (
                        config_horizon,
                        config_score,
                    ) in config_horizons.items():

                        test_df.loc[
                            (
                                test_df[
                                    "Portfolio Target Type"
                                ]
                                == config_type
                            )
                            &
                            (
                                test_df[
                                    "Horizon Key"
                                ]
                                == config_horizon
                            ),
                            "Horizon Score",
                        ] = config_score


                ################################
                # Override Focal Candidate
                ################################

                test_df.loc[
                    (
                        test_df[
                            "Portfolio Target Type"
                        ]
                        == target_type
                    )
                    &
                    (
                        test_df[
                            "Horizon Key"
                        ]
                        == horizon
                    ),
                    "Horizon Score",
                ] = score


                BQ = backtest_quality(
                    test_df
                )

                quality_by_index[
                    index
                ].append(
                    BQ
                )


        ####################################
        # Candidate Win Rates
        ####################################

        win_counts = {
            index: 0
            for index
            in quality_by_index
        }


        for iteration in range(
            iterations
        ):

            iteration_scores = {
                index:
                    quality_by_index[
                        index
                    ][
                        iteration
                    ]

                for index
                in quality_by_index
            }


            winning_index = max(
                iteration_scores,
                key=iteration_scores.get,
            )


            win_counts[
                winning_index
            ] += 1


        win_rates = {
            index:
                win_counts[
                    index
                ]
                / iterations

            for index
            in win_counts
        }


        best_index = max(
            win_rates,
            key=win_rates.get,
        )


        remaining_values = [
            value

            for (
                index,
                value,
            )
            in enumerate(
                values
            )

            if (
                win_rates[
                    index
                ]
                > threshold
                or index
                == best_index
            )
        ]


        HORIZON_SCORES[
            target_type
        ][
            horizon
        ] = remaining_values


        logger.info(
            "Random screen result | %s %s | wins=%s | best_index=%d | kept=%s",
            target_type,
            horizon,
            {
                index:
                    round(
                        rate,
                        3,
                    )
                for (
                    index,
                    rate,
                )
                in win_rates.items()
            },
            best_index,
            remaining_values,
        )


        results.append(
            {
                "Portfolio Target Type":
                    target_type,

                "Horizon":
                    horizon,

                "Original Values":
                    values,

                "Mean BQ":
                    {
                        index:
                            np.mean(
                                quality_by_index[
                                    index
                                ]
                            )

                        for index
                        in quality_by_index
                    },

                "Win Rates":
                    win_rates,

                "Best Index":
                    best_index,

                "Best Value":
                    values[
                        best_index
                    ],

                "Remaining Values":
                    remaining_values,

                "Random Iterations":
                    random_iterations,
            }
        )


    logger.info(
        "Random screen complete | iterations=%d | threshold=%.3f | remaining configurations=%d",
        iterations,
        threshold,
        math.prod(
            len(
                values
            )
            for horizons
            in HORIZON_SCORES.values()
            for values
            in horizons.values()
        ),
    )


    return results


########################################
# Scheduled Random Screens
########################################

results = random_screen(
    iterations=20,
    threshold=0.15,
)

results = random_screen(
    iterations=50,
    threshold=0.30,
)

results = random_screen(
    iterations=100,
    threshold=0.35,
)


total_configurations = math.prod(
    len(
        values
    )
    for horizons
    in HORIZON_SCORES.values()
    for values
    in horizons.values()
)


logger.info(
    "Configuration count after scheduled random screens: %d",
    total_configurations,
)


########################################
# Optional Further Pruning
########################################

iters = 100
thres = 0.35


while total_configurations > 1000:

    new_iters = input(
        f"Iterations must be greater than "
        f"{iters} [{iters}]: "
    )

    new_thres = input(
        f"Threshold must be greater than "
        f"{thres} [{thres}]: "
    )


    if (
        new_iters
        and int(
            new_iters
        )
        > iters
    ):

        iters = int(
            new_iters
        )


    if (
        new_thres
        and float(
            new_thres
        )
        > thres
    ):

        thres = float(
            new_thres
        )


    results = random_screen(
        iterations=iters,
        threshold=thres,
    )


    total_configurations = math.prod(
        len(
            values
        )
        for horizons
        in HORIZON_SCORES.values()
        for values
        in horizons.values()
    )


    logger.info(
        "Remaining configurations: %d | iterations=%d | threshold=%.3f",
        total_configurations,
        iters,
        thres,
    )


########################################
# Exhaustive Search
########################################

parameters = [
    (
        target_type,
        horizon,
    )
    for (
        target_type,
        horizons,
    )
    in HORIZON_SCORES.items()
    for horizon
    in horizons
]


possible_values = [
    HORIZON_SCORES[
        target_type
    ][
        horizon
    ]
    for (
        target_type,
        horizon,
    )
    in parameters
]


exhaustive_results = []


exhaustive_total = math.prod(
    len(
        values
    )
    for values
    in possible_values
)


logger.info(
    "Starting exhaustive search | configurations=%d",
    exhaustive_total,
)


log_every = max(
    1,
    exhaustive_total
    // 20,
)


for (
    combination_number,
    combination,
) in enumerate(
    itertools.product(
        *possible_values
    ),
    start=1,
):

    if (
        combination_number == 1
        or combination_number
        % log_every
        == 0
        or combination_number
        == exhaustive_total
    ):

        logger.info(
            "Exhaustive search progress | %d/%d (%.1f%%)",
            combination_number,
            exhaustive_total,
            100
            * combination_number
            / exhaustive_total,
        )


    test_df = (
        predictions_df.copy()
    )

    configuration = {}


    for (
        (
            target_type,
            horizon,
        ),
        score,
    ) in zip(
        parameters,
        combination,
    ):

        test_df.loc[
            (
                test_df[
                    "Portfolio Target Type"
                ]
                == target_type
            )
            &
            (
                test_df[
                    "Horizon Key"
                ]
                == horizon
            ),
            "Horizon Score",
        ] = score


        configuration.setdefault(
            target_type,
            {},
        )[
            horizon
        ] = float(
            score
        )


    BQ = backtest_quality(
        test_df
    )


    if BQ < 0:
        continue


    exhaustive_results.append(
        {
            "BQ":
                float(
                    BQ
                ),

            "Horizon Scores":
                configuration,
        }
    )


########################################
# Sort Highest BQ First
########################################

exhaustive_results.sort(
    key=lambda result:
        result[
            "BQ"
        ],
    reverse=True,
)


logger.info(
    "Exhaustive search complete | non-negative configurations=%d/%d",
    len(
        exhaustive_results
    ),
    exhaustive_total,
)


########################################
# Keep Top 5%
########################################

if exhaustive_results:

    top_n = max(
        1,
        math.ceil(
            len(
                exhaustive_results
            )
            * 0.05
        ),
    )


    top_configurations = (
        exhaustive_results[
            :top_n
        ]
    )

else:

    top_configurations = []

    logger.warning(
        "No non-negative BQ configurations survived."
    )


logger.info(
    "Top configurations retained=%d",
    len(
        top_configurations
    ),
)

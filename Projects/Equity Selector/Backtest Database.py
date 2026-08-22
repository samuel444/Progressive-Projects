import ast
import json
import logging
import re
import sqlite3
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from features import *
from targets import *


warnings.filterwarnings("ignore")

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

# This is the cached database produced by this file and consumed by
# 02_train_and_backtest.py.
OUTPUT_DB = (
    DATA_DIR
    / "Backtest_Features_Targets.db"
)


########################################
# Build Settings
########################################

# Extra raw history used only to warm up rolling features.
FEATURE_WARMUP_YEARS = 3

DEFAULT_REBALANCE_EVERY = 60


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

stock_type_index = STOCK_TYPE_INDICES[STOCK_TYPE]


########################################
# Load Previously Selected Research Results
########################################

logger.info(
    "Loading previously selected model specifications"
)

with sqlite3.connect(
    FINAL_RESULTS_DB
) as connection:

    test_results = pd.read_sql_query(
        f"SELECT * FROM 'Most Predictable Results {STOCK_TYPE}'",
        connection,
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


logger.info(
    "Stock type: %s | selected-feature line index: %d | "
    "predictable-results table: Most Predictable Results %s",
    STOCK_TYPE,
    stock_type_index,
    STOCK_TYPE,
)


########################################
# Select All Investigation-Worthy Models
########################################

# These remain the same investigation thresholds used by the
# existing research pipeline. They are applied to the original
# Predictability Score before the new cross-target Quality Score
# is calculated.
INVESTIGATION_SCORE_THRESHOLDS = {
    "continuous": 0.12,
    "binary": 0.20,
    "multiclass": 0.35,
}


# The main output table requested for the portfolio-weighting
# pipeline. The STOCK_TYPE is included so multiple universes can
# coexist safely in the same SQLite database.
SELECTED_MODELS_TABLE = (
    f"Selected Models {STOCK_TYPE}"
)

# Keep target-specific feature lists in a separate helper table so
# the requested Selected Models table stays clean while downstream
# training code can still recover the correct features for a target.
SELECTED_MODEL_FEATURES_TABLE = (
    f"Selected Model Features {STOCK_TYPE}"
)


########################################
# Portfolio Target Type Classification
########################################

def portfolio_target_type(
    target,
    prediction_type=None,
):

    name = str(target).strip().lower()
    prediction_type = str(
        prediction_type or ""
    ).strip().lower()

    # Execution / state targets first because their names may also
    # contain words such as volatility, return, or risk.
    if (
        "market impact" in name
        or "price impact" in name
    ):
        return "MARKET_IMPACT"

    if (
        "execution" in name
        or "fill probability" in name
        or "fill rate" in name
        or "slippage" in name
    ):
        return "EXECUTION"

    if (
        "liquidity" in name
        or "bid ask spread" in name
        or "bid-ask spread" in name
        or "order book depth" in name
        or "order-book depth" in name
    ):
        return "LIQUIDITY"

    if "covariance" in name:
        return "COVARIANCE"

    if "correlation" in name:
        return "CORRELATION"

    if "regime" in name:
        return "REGIME"

    # Intraday / event behaviour.
    if (
        "recovery" in name
        or "recover" in name
        or "bounce back" in name
    ):
        return "RECOVERY"

    if (
        "reversal" in name
        or "reverse" in name
        or "mean reversion" in name
        or "mean-reversion" in name
    ):
        return "REVERSAL"

    if (
        "sudden drawdown" in name
        or "crash" in name
        or "tail event" in name
        or "extreme downside" in name
        or "downside event" in name
        or "negative spike" in name
    ):
        return "TAIL_EVENT"

    if (
        "upside spike" in name
        or "positive spike" in name
        or "upside event" in name
        or "positive event" in name
    ):
        return "UPSIDE_EVENT"

    # Volatility event must be checked before generic volatility.
    if (
        "volatility barrier" in name
        or "volatility event" in name
        or "volatility spike" in name
        or "volatility breakout" in name
    ):
        return "VOLATILITY_EVENT"

    if (
        "upside volatility" in name
        or "positive volatility" in name
    ):
        return "UPSIDE_RISK"

    # Tail-distribution targets.
    if (
        "maximum adverse excursion" in name
        or "max adverse excursion" in name
        or "expected shortfall" in name
        or "conditional value at risk" in name
        or "conditional var" in name
        or "cvar" in name
        or "value at risk" in name
        or re.search(r"\bvar\b", name)
        or "maximum drawdown" in name
        or "max drawdown" in name
        or "tail risk" in name
    ):
        return "TAIL_RISK"

    if (
        "minimum return" in name
        or "min return" in name
        or "downside deviation" in name
        or "downside volatility" in name
        or "drawdown" in name
        or "downside" in name
    ):
        return "DOWNSIDE"

    if "volatility" in name:
        return "VOLATILITY"

    # Cross-sectional / relative alpha before generic return rules.
    if (
        "top 20" in name
        or "top 25" in name
        or "top 10" in name
        or "top quintile" in name
        or "top quartile" in name
        or "cross sectional" in name
        or "cross-sectional" in name
        or "return rank" in name
        or "return percentile" in name
        or "return quantile" in name
    ):
        return "CROSS_SECTION_ALPHA"

    if (
        "excess return" in name
        or "relative return" in name
        or "abnormal return" in name
        or "residual return" in name
        or "benchmark return" in name
    ):
        return "RELATIVE_ALPHA"

    if "direction" in name:
        return "DIRECTION"

    # A volatility/downside barrier has already been caught above.
    if "barrier" in name:
        return "BARRIER_ALPHA"

    if (
        "return above" in name
        or "return below" in name
        or "positive return" in name
        or "negative return" in name
        or "return event" in name
    ):
        return "ALPHA_BINARY"

    # Risk-adjusted return targets are still reward/alpha targets in
    # the portfolio layer: larger predicted values are desirable.
    if (
        "sharpe" in name
        or "sortino" in name
        or "calmar" in name
        or "risk adjusted" in name
        or "risk-adjusted" in name
    ):
        return "ALPHA"

    if (
        "return" in name
        or "alpha" in name
        or "momentum" in name
    ):
        return "ALPHA"

    # Final fallback uses the research-level prediction family.
    if prediction_type == "volatility":
        return "VOLATILITY"

    if prediction_type == "downside":
        return "DOWNSIDE"

    return "ALPHA"


########################################
# Horizon Extraction
########################################

def target_horizon(
    target,
):

    name = str(target).strip().lower()

    # Prefer values with explicit units. This allows names such as
    # "Sudden Drawdown 15m" or "Direction 1h" to be interpreted
    # correctly even if another number appears earlier in the name.
    explicit = re.findall(
        r"(?<![a-z0-9])"
        r"(\d+(?:\.\d+)?)\s*"
        r"(m|min|mins|minute|minutes|"
        r"h|hr|hrs|hour|hours|"
        r"d|day|days|"
        r"w|week|weeks)"
        r"(?![a-z])",
        name,
    )

    if explicit:
        value = float(
            explicit[-1][0]
        )

        return (
            int(value)
            if value.is_integer()
            else value
        )

    # Daily targets in the existing targets.py convention generally
    # finish with the horizon, e.g. "Forward Return 20". Taking the
    # last standalone number also avoids using an earlier threshold
    # such as the 2 in "Return Above 2% 20".
    numbers = re.findall(
        r"(?<![a-z0-9.])"
        r"(\d+(?:\.\d+)?)"
        r"(?![a-z0-9.%])",
        name,
    )

    if not numbers:
        # Do not fall back to arbitrary numbers because many target
        # names contain thresholds such as 2% or 5%. If no horizon
        # can be distinguished safely, store NULL and log it later.
        return np.nan

    value = float(
        numbers[-1]
    )

    return (
        int(value)
        if value.is_integer()
        else value
    )


########################################
# Cross-Target Model Quality
########################################

def _normalise_column_name(
    value,
):

    return re.sub(
        r"[^a-z0-9]+",
        "",
        str(value).lower(),
    )


def _row_metric(
    row,
    aliases,
):

    column_lookup = {
        _normalise_column_name(column): column
        for column in row.index
    }

    for alias in aliases:

        key = _normalise_column_name(
            alias
        )

        if key not in column_lookup:
            continue

        value = pd.to_numeric(
            pd.Series(
                [
                    row[
                        column_lookup[key]
                    ]
                ]
            ),
            errors="coerce",
        ).iloc[0]

        if pd.notna(value):
            return float(value)

    return None


def _clip_quality(
    value,
):

    return float(
        np.clip(
            value,
            0.0,
            1.0,
        )
    )


def _weighted_available_mean(
    values,
):

    # values = [(weight, value), ...]
    available = [
        (weight, value)
        for weight, value in values
        if value is not None
        and np.isfinite(value)
    ]

    if not available:
        return None

    total_weight = sum(
        weight
        for weight, _ in available
    )

    return sum(
        weight * value
        for weight, value in available
    ) / total_weight


def calculate_quality_score(
    row,
    portfolio_type,
):

    statistical_type = str(
        row.get(
            "Target Type",
            "",
        )
    ).strip().lower()

    existing_predictability = _row_metric(
        row,
        [
            "Predictability Score",
        ],
    )

    ####################################
    # Continuous Targets
    ####################################

    if statistical_type == "continuous":

        rank_ic = _row_metric(
            row,
            [
                "Rank IC",
                "Mean Rank IC",
                "Rank IC Mean",
                "Spearman IC",
                "Spearman Correlation",
                "Spearman",
            ],
        )

        r2 = _row_metric(
            row,
            [
                "R2",
                "R^2",
                "R Squared",
                "R2 Score",
                "Test R2",
                "Test R^2",
            ],
        )

        q_ic = (
            _clip_quality(
                abs(rank_ic) / 0.30
            )
            if rank_ic is not None
            else None
        )

        q_r2 = (
            _clip_quality(
                max(r2, 0.0) / 0.20
            )
            if r2 is not None
            else None
        )

        quality = _weighted_available_mean(
            [
                (0.70, q_ic),
                (0.30, q_r2),
            ]
        )

        if quality is not None:
            return _clip_quality(
                quality
            )

    ####################################
    # Binary Targets
    ####################################

    if statistical_type == "binary":

        roc_auc = _row_metric(
            row,
            [
                "ROC AUC",
                "ROC-AUC",
                "AUC ROC",
                "AUC",
                "Test ROC AUC",
            ],
        )

        pr_auc = _row_metric(
            row,
            [
                "PR AUC",
                "PR-AUC",
                "Average Precision",
                "Average Precision Score",
                "Test PR AUC",
            ],
        )

        positive_rate = _row_metric(
            row,
            [
                "Positive Rate",
                "Positive Class Rate",
                "Positive Fraction",
                "Prevalence",
                "Base Rate",
                "Event Rate",
            ],
        )

        q_roc = (
            _clip_quality(
                (roc_auc - 0.50) / 0.25
            )
            if roc_auc is not None
            else None
        )

        q_pr = None

        if (
            pr_auc is not None
            and positive_rate is not None
            and 0.0 <= positive_rate < 1.0
        ):

            # A +0.30 PR-AUC improvement over the random/base-rate
            # classifier is treated as an excellent result and capped
            # at 1. This is much fairer for rare-event targets than raw
            # accuracy or raw PR-AUC.
            excellent_pr = min(
                1.0,
                positive_rate + 0.30,
            )

            denominator = max(
                excellent_pr - positive_rate,
                1e-12,
            )

            q_pr = _clip_quality(
                (
                    pr_auc - positive_rate
                )
                / denominator
            )

        event_types = {
            "TAIL_EVENT",
            "VOLATILITY_EVENT",
            "UPSIDE_EVENT",
        }

        if portfolio_type in event_types:
            weights = [
                (0.40, q_roc),
                (0.60, q_pr),
            ]
        else:
            weights = [
                (0.60, q_roc),
                (0.40, q_pr),
            ]

        quality = _weighted_available_mean(
            weights
        )

        if quality is not None:
            return _clip_quality(
                quality
            )

    ####################################
    # Multiclass Targets
    ####################################

    if statistical_type == "multiclass":

        macro_f1 = _row_metric(
            row,
            [
                "Macro F1",
                "Macro-F1",
                "F1 Macro",
                "Macro F1 Score",
                "Test Macro F1",
            ],
        )

        macro_auc = _row_metric(
            row,
            [
                "Macro ROC AUC",
                "Macro AUC",
                "OVR Macro AUC",
                "One Vs Rest Macro AUC",
                "Multiclass ROC AUC",
            ],
        )

        number_classes = _row_metric(
            row,
            [
                "Number Classes",
                "Number of Classes",
                "N Classes",
                "Num Classes",
            ],
        )

        if (
            number_classes is None
            or number_classes < 2
        ):
            number_classes = 3.0

        chance_f1 = 1.0 / number_classes
        excellent_f1 = 0.70

        q_f1 = None

        if macro_f1 is not None:

            denominator = max(
                excellent_f1 - chance_f1,
                1e-12,
            )

            q_f1 = _clip_quality(
                (
                    macro_f1 - chance_f1
                )
                / denominator
            )

        q_auc = (
            _clip_quality(
                (macro_auc - 0.50) / 0.25
            )
            if macro_auc is not None
            else None
        )

        quality = _weighted_available_mean(
            [
                (0.70, q_f1),
                (0.30, q_auc),
            ]
        )

        if quality is not None:
            return _clip_quality(
                quality
            )

    ####################################
    # Fallback
    ####################################

    # The existing Predictability Score is already the research
    # pipeline's best summary when the underlying component metrics
    # are not stored in the SQL table. Keep the pipeline operational
    # rather than discarding an otherwise valid model.
    if existing_predictability is not None:
        return _clip_quality(
            existing_predictability
        )

    return 0.0


########################################
# Keep Investigation-Worthy Results
########################################

def investigation_worthy_results():

    available = test_results.copy()

    # Remove baseline models.
    available = available[
        ~available[
            "Model"
        ]
        .astype(str)
        .str.contains(
            "Baseline",
            case=False,
            na=False,
        )
    ].copy()

    statistical_types = (
        available[
            "Target Type"
        ]
        .astype(str)
        .str.lower()
    )

    thresholds = statistical_types.map(
        INVESTIGATION_SCORE_THRESHOLDS
    )

    predictability = pd.to_numeric(
        available[
            "Predictability Score"
        ],
        errors="coerce",
    )

    available = available[
        thresholds.notna()
        & predictability.notna()
        & (
            predictability >= thresholds
        )
    ].copy()

    if available.empty:
        raise ValueError(
            "No investigation-worthy model results "
            f"are available for {STOCK_TYPE}."
        )

    # One production model specification per target. If several model
    # families passed the investigation threshold for the same target,
    # use the best tested specification according to the existing
    # Predictability Score.
    available = (
        available
        .sort_values(
            "Predictability Score",
            ascending=False,
        )
        .drop_duplicates(
            subset=[
                "Target",
            ],
            keep="first",
        )
        .reset_index(
            drop=True
        )
    )

    available[
        "Portfolio Target Type"
    ] = available.apply(
        lambda row: portfolio_target_type(
            row[
                "Target"
            ],
            row.get(
                "Prediction Type",
                "",
            ),
        ),
        axis=1,
    )

    available[
        "Horizon"
    ] = available[
        "Target"
    ].map(
        target_horizon
    )

    available[
        "Quality Score"
    ] = available.apply(
        lambda row: calculate_quality_score(
            row,
            row[
                "Portfolio Target Type"
            ],
        ),
        axis=1,
    )

    missing_horizons = available[
        available[
            "Horizon"
        ].isna()
    ]

    if not missing_horizons.empty:
        logger.warning(
            "%d selected targets do not contain a parseable "
            "numeric horizon. Their Horizon value will be NULL: %s",
            len(
                missing_horizons
            ),
            ", ".join(
                missing_horizons[
                    "Target"
                ].astype(str)
            ),
        )

    return available


selected_model_results = (
    investigation_worthy_results()
)


########################################
# Selected Target / Feature Union
########################################

selected_targets = list(
    dict.fromkeys(
        selected_model_results[
            "Target"
        ].astype(str).tolist()
    )
)


missing_feature_definitions = [
    target
    for target in selected_targets
    if target not in selected_features
]


if missing_feature_definitions:
    raise ValueError(
        "Selected_Features.txt does not contain "
        "feature definitions for:\n"
        + "\n".join(
            missing_feature_definitions
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
    "Selected %d model targets across %d portfolio target types",
    len(
        selected_model_results
    ),
    selected_model_results[
        "Portfolio Target Type"
    ].nunique(),
)


logger.info(
    "Selected-feature union contains %d unique features",
    len(
        required_features
    ),
)


print(
    f"\n{'=' * 100}"
)

print(
    f"SELECTED MODELS | {STOCK_TYPE}"
)

print(
    "=" * 100
)

print(
    selected_model_results[
        [
            "Target",
            "Model",
            "Portfolio Target Type",
            "Horizon",
            "Quality Score",
        ]
    ].to_string(
        index=False
    )
)


########################################
# Ask For Training / Backtest Data
########################################

DEFAULT_TRAIN_TOKENS = [
    "AAPL",
    "MSFT",
]


DEFAULT_BACKTEST_TOKENS = [
    "NVDA",
    "GOOG",
    "META",
    "AMZN",
    "TSLA",
    "NFLX",
    "AVGO",
    "AMD",
    "JPM",
    "BAC",
    "GS",
    "V",
    "MA",
    "WMT",
    "COST",
    "KO",
    "PEP",
    "MCD",
    "XOM",
    "CVX",
    "CAT",
    "GE",
    "BA",
    "JNJ",
    "LLY",
    "UNH",
    "HD",
    "DIS",
]


def get_tokens(
    prompt,
    default,
):

    print(
        "\nDefault:"
    )

    print(
        ", ".join(
            default
        )
    )


    value = input(
        f"\n{prompt} "
        "(comma separated, "
        "Enter for default): "
    ).strip()


    if not value:

        tokens = (
            default.copy()
        )

    else:

        tokens = list(
            dict.fromkeys(
                token
                .strip()
                .upper()

                for token
                in value.split(",")

                if token.strip()
            )
        )


    ####################################
    # ^GSPC Is Benchmark Only
    ####################################

    if "^GSPC" in tokens:

        logger.warning(
            "^GSPC removed from stock universe. "
            "It is downloaded separately as "
            "the market benchmark."
        )

        tokens = [
            token
            for token in tokens
            if token != "^GSPC"
        ]


    if not tokens:

        raise ValueError(
            "At least one stock ticker "
            "must be supplied."
        )


    return tokens


def get_date(
    prompt,
):

    while True:

        value = input(
            prompt
        ).strip()

        try:

            return pd.Timestamp(
                value
            )

        except Exception:

            print(
                "Please enter the date "
                "as YYYY-MM-DD."
            )


########################################
# Training Questions
########################################

print(
    "\n"
    + "=" * 70
)

print(
    "TRAINING DATA"
)

print(
    "=" * 70
)


train_tokens = get_tokens(
    "Training tickers:",
    DEFAULT_TRAIN_TOKENS,
)


train_start = get_date(
    "Training start date "
    "(YYYY-MM-DD): "
)


train_end = get_date(
    "Training end date   "
    "(YYYY-MM-DD): "
)


if train_end <= train_start:

    raise ValueError(
        "Training end date must be "
        "after training start date."
    )


########################################
# Backtest Questions
########################################

print(
    "\n"
    + "=" * 70
)

print(
    "BACKTEST DATA"
)

print(
    "=" * 70
)


backtest_tokens = get_tokens(
    "Backtest tickers:",
    DEFAULT_BACKTEST_TOKENS,
)


backtest_start = get_date(
    "Backtest start date "
    "(YYYY-MM-DD): "
)


backtest_end = get_date(
    "Backtest end date   "
    "(YYYY-MM-DD): "
)


if backtest_end <= backtest_start:

    raise ValueError(
        "Backtest end date must be "
        "after backtest start date."
    )


if backtest_start < train_end:

    raise ValueError(
        "\nBacktest period overlaps "
        "the training period.\n"
        f"Training ends: "
        f"{train_end.date()}\n"
        f"Backtest starts: "
        f"{backtest_start.date()}"
    )


########################################
# Rebalance Question
########################################

rebalance_input = input(
    f"Rebalance every N trading days "
    f"[{DEFAULT_REBALANCE_EVERY}]: "
).strip()


DEFAULT_REBALANCE = (
    int(
        rebalance_input
    )
    if rebalance_input
    else DEFAULT_REBALANCE_EVERY
)


if DEFAULT_REBALANCE <= 0:

    raise ValueError(
        "Rebalance frequency must be "
        "greater than zero."
    )


########################################
# ONE Yfinance Download
########################################

all_tokens = list(
    dict.fromkeys(
        train_tokens
        + backtest_tokens
    )
)


download_start = (
    min(
        train_start,
        backtest_start,
    )
    - pd.DateOffset(
        years=FEATURE_WARMUP_YEARS
    )
)


download_end = (
    max(
        train_end,
        backtest_end,
    )
    + pd.Timedelta(
        days=1
    )
)


symbols = list(
    dict.fromkeys(
        all_tokens
        + [
            "^GSPC"
        ]
    )
)


logger.info(
    "Downloading %d stocks + S&P 500 "
    "in one yfinance call",
    len(
        all_tokens
    ),
)


raw_download = yf.download(
    symbols,
    start=download_start.strftime(
        "%Y-%m-%d"
    ),
    end=download_end.strftime(
        "%Y-%m-%d"
    ),
    auto_adjust=True,
    progress=False,
    group_by="ticker",
    multi_level_index=True,
)


if raw_download.empty:

    raise ValueError(
        "yfinance returned no data."
    )


downloaded_symbols = set(
    raw_download
    .columns
    .get_level_values(0)
)


if "^GSPC" not in downloaded_symbols:

    raise ValueError(
        "S&P 500 (^GSPC) was not "
        "returned by yfinance."
    )


market_df = (
    raw_download[
        "^GSPC"
    ]
    .copy()
    .dropna(
        how="all"
    )
)


market_df.index = pd.to_datetime(
    market_df.index
)


########################################
# Efficient Feature Generation
########################################

def selected_features_complete(
    dataframe,
):

    return set(
        required_features
    ).issubset(
        dataframe.columns
    )


def build_individual_features(
    stock_df,
    benchmark_df,
):

    stock_df = (
        stock_df.copy()
    )


    stock_df[
        "Return"
    ] = (
        stock_df[
            "Close"
        ]
        .pct_change()
    )


    ####################################
    # Same Dependency Order As Research
    ####################################

    feature_steps = [
        (
            "return",
            lambda x:
                all_return_features(x),
        ),
        (
            "momentum",
            lambda x:
                all_momentum_features(x),
        ),
        (
            "volatility",
            lambda x:
                all_volatility_features(x),
        ),
        (
            "range volatility",
            lambda x:
                all_range_volatility_features(x),
        ),
        (
            "trend",
            lambda x:
                all_trend_features(x),
        ),
        (
            "moving average",
            lambda x:
                all_moving_average_features(x),
        ),
        (
            "drawdown",
            lambda x:
                all_drawdown_features(x),
        ),
        (
            "distribution",
            lambda x:
                all_distribution_features(x),
        ),
        (
            "tail risk",
            lambda x:
                all_tail_risk_features(x),
        ),
        (
            "volume",
            lambda x:
                all_volume_features(x),
        ),
        (
            "liquidity",
            lambda x:
                all_liquidity_features(x),
        ),
        (
            "ohlc",
            lambda x:
                all_ohlc_features(x),
        ),
        (
            "market relative",
            lambda x:
                all_market_relative_features(
                    x,
                    market_df=benchmark_df,
                ),
        ),
        (
            "beta",
            lambda x:
                all_beta_features(
                    x,
                    market_df=benchmark_df,
                ),
        ),
        (
            "correlation",
            lambda x:
                all_correlation_features(
                    x,
                    market_df=benchmark_df,
                ),
        ),
        (
            "residual",
            lambda x:
                all_residual_features(
                    x,
                    market_df=benchmark_df,
                ),
        ),
        (
            "technical",
            lambda x:
                all_technical_features(x),
        ),
        (
            "regime",
            lambda x:
                all_regime_features(x),
        ),
        (
            "interaction",
            lambda x:
                all_interaction_features(x),
        ),
        (
            "composite",
            lambda x:
                all_composite_features(x),
        ),
        (
            "experimental",
            lambda x:
                all_experimental_features(x),
        ),
    ]


    for (
        step_name,
        feature_function,
    ) in feature_steps:

        if selected_features_complete(
            stock_df
        ):

            logger.debug(
                "Feature union complete; "
                "stopping before %s",
                step_name,
            )

            break


        stock_df = (
            feature_function(
                stock_df
            )
        )


    return stock_df


def add_cross_stock_features_if_needed(
    panel,
):

    if selected_features_complete(
        panel
    ):

        return panel


    base_columns = {
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "Return",
        "Ticker",
        "Date",
    }


    ####################################
    # Cross-Sectional Features
    ####################################

    if (
        panel[
            "Ticker"
        ]
        .nunique()
        > 1
    ):

        logger.info(
            "Selected union requires "
            "cross-sectional features"
        )


        individual_feature_columns = [
            column
            for column
            in panel.columns
            if column
            not in base_columns
        ]


        panel = (
            all_cross_sectional_features(
                panel,
                columns=(
                    individual_feature_columns
                ),
                date_col="Date",
            )
        )


    if selected_features_complete(
        panel
    ):

        return panel


    ####################################
    # Breadth / Dispersion Features
    ####################################

    if (
        panel[
            "Ticker"
        ]
        .nunique()
        <= 1
    ):

        return panel


    logger.info(
        "Selected union requires "
        "breadth / dispersion features"
    )


    wide_df = panel.pivot(
        index="Date",
        columns="Ticker",
        values=[
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "Return",
        ],
    )


    original_num_columns = len(
        wide_df.columns
    )


    market_features = (
        all_breadth_features(
            wide_df.copy()
        )
    )


    market_features = (
        all_dispersion_features(
            market_features
        )
    )


    new_market_features = (
        market_features
        .iloc[
            :,
            original_num_columns:
        ]
        .copy()
    )


    new_market_features.columns = [
        column[0]
        if isinstance(
            column,
            tuple,
        )
        else column

        for column
        in new_market_features.columns
    ]


    new_market_features = (
        new_market_features.loc[
            :,
            ~new_market_features
            .columns
            .duplicated()
        ]
    )


    new_market_features = (
        new_market_features
        .reset_index()
    )


    panel = panel.merge(
        new_market_features,
        on="Date",
        how="left",
    )


    return panel


def build_feature_panel(
    universe,
    start_date,
    end_date,
    panel_name,
):

    logger.info(
        "Building %s feature panel "
        "for %d requested stocks",
        panel_name,
        len(
            universe
        ),
    )


    warmup_start = (
        start_date
        - pd.DateOffset(
            years=FEATURE_WARMUP_YEARS
        )
    )


    benchmark_slice = market_df[
        (
            market_df.index
            >= warmup_start
        )
        &
        (
            market_df.index
            <= end_date
        )
    ].copy()


    stock_dfs = {}


    for token in universe:

        if token not in downloaded_symbols:

            logger.warning(
                "%s skipped: not returned "
                "by yfinance",
                token,
            )

            continue


        stock_df = (
            raw_download[
                token
            ]
            .copy()
            .dropna(
                how="all"
            )
        )


        stock_df.index = (
            pd.to_datetime(
                stock_df.index
            )
        )


        stock_df = stock_df[
            (
                stock_df.index
                >= warmup_start
            )
            &
            (
                stock_df.index
                <= end_date
            )
        ].copy()


        if stock_df.empty:

            logger.warning(
                "%s skipped: no observations "
                "inside required range",
                token,
            )

            continue


        logger.info(
            "%s | %s feature generation",
            panel_name,
            token,
        )


        stock_df = (
            build_individual_features(
                stock_df,
                benchmark_slice,
            )
        )


        stock_dfs[
            token
        ] = stock_df


    usable_tokens = [
        token
        for token
        in universe
        if token
        in stock_dfs
    ]


    if not usable_tokens:

        raise ValueError(
            f"No stocks remain in "
            f"{panel_name} universe."
        )


    panel_parts = []


    for token in usable_tokens:

        stock_df = (
            stock_dfs[
                token
            ]
            .copy()
        )


        stock_df[
            "Ticker"
        ] = token


        stock_df[
            "Date"
        ] = (
            stock_df.index
        )


        panel_parts.append(
            stock_df
            .reset_index(
                drop=True
            )
        )


    panel = pd.concat(
        panel_parts,
        ignore_index=True,
    )


    panel = (
        add_cross_stock_features_if_needed(
            panel
        )
    )


    missing_features = [
        feature
        for feature
        in required_features
        if feature
        not in panel.columns
    ]


    if missing_features:

        raise ValueError(
            f"\n{panel_name} feature pipeline "
            "could not create:\n"
            + "\n".join(
                missing_features
            )
        )


    ####################################
    # Keep Only Base + Required Union
    ####################################

    columns_to_keep = list(
        dict.fromkeys(
            [
                "Date",
                "Ticker",
                "Open",
                "High",
                "Low",
                "Close",
                "Volume",
                "Return",
            ]
            + required_features
        )
    )


    panel = (
        panel[
            columns_to_keep
        ]
        .copy()
    )


    panel[
        "Date"
    ] = pd.to_datetime(
        panel[
            "Date"
        ]
    )


    panel[
        required_features
    ] = (
        panel[
            required_features
        ]
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
    )


    ####################################
    # Remove Warm-Up Rows
    ####################################

    panel = panel[
        (
            panel[
                "Date"
            ]
            >= start_date
        )
        &
        (
            panel[
                "Date"
            ]
            <= end_date
        )
    ].copy()


    return (
        panel,
        usable_tokens,
    )


########################################
# Build Training Features
########################################

training_features_df, train_tokens = (
    build_feature_panel(
        universe=train_tokens,
        start_date=train_start,
        end_date=train_end,
        panel_name="TRAINING",
    )
)


########################################
# Build Backtest Features
########################################

backtest_features_df, backtest_tokens = (
    build_feature_panel(
        universe=backtest_tokens,
        start_date=backtest_start,
        end_date=backtest_end,
        panel_name="BACKTEST",
    )
)


########################################
# Selected Target Generation
########################################

def add_selected_targets(
    feature_df,
    universe,
    split_end,
    panel_name,
):

    logger.info(
        "Generating selected %s targets",
        panel_name,
    )


    target_parts = []


    benchmark_slice = (
        market_df[
            market_df.index
            <= split_end
        ]
        .copy()
    )


    for token in universe:

        token_df = (
            feature_df[
                feature_df[
                    "Ticker"
                ]
                == token
            ]
            .sort_values(
                "Date"
            )
            .copy()
        )


        if token_df.empty:
            continue


        token_df = (
            token_df
            .set_index(
                "Date"
            )
        )


        if (
            "Ticker"
            in token_df.columns
        ):

            token_df = (
                token_df.drop(
                    columns=[
                        "Ticker"
                    ]
                )
            )


        ####################################
        # Same Target Order As Research
        ####################################

        target_steps = [
            (
                "return",
                lambda x:
                    all_return_targets(
                        x,
                        benchmark_df=(
                            benchmark_slice
                        ),
                    ),
            ),
            (
                "volatility",
                lambda x:
                    all_volatility_targets(x),
            ),
            (
                "direction",
                lambda x:
                    all_direction_targets(x),
            ),
            (
                "barrier",
                lambda x:
                    all_barrier_targets(x),
            ),
            (
                "excursion",
                lambda x:
                    all_excursion_targets(x),
            ),
            (
                "drawdown",
                lambda x:
                    all_drawdown_targets(x),
            ),
            (
                "risk adjusted",
                lambda x:
                    all_risk_adjusted_targets(x),
            ),
        ]


        for (
            step_name,
            target_function,
        ) in target_steps:

            present = set(
                selected_targets
            ).intersection(
                token_df.columns
            )


            if (
                len(
                    present
                )
                == len(
                    selected_targets
                )
            ):

                break


            logger.debug(
                "%s | %s targets",
                token,
                step_name,
            )


            token_df = (
                target_function(
                    token_df
                )
            )


        token_df[
            "Ticker"
        ] = token


        token_df[
            "Date"
        ] = (
            token_df.index
        )


        available_selected = [
            target
            for target
            in selected_targets
            if target
            in token_df.columns
        ]


        target_parts.append(
            token_df[
                [
                    "Date",
                    "Ticker",
                    "Close",
                ]
                + available_selected
            ]
            .reset_index(
                drop=True
            )
        )


    if not target_parts:

        raise ValueError(
            f"No {panel_name} target "
            "data could be generated."
        )


    target_panel = pd.concat(
        target_parts,
        ignore_index=True,
    )


    missing_targets = [
        target
        for target
        in selected_targets
        if target
        not in target_panel.columns
    ]


    ####################################
    # Cross-Sectional Ranking Targets
    ####################################

    if missing_targets:

        logger.info(
            "%s has selected targets not "
            "created individually; trying "
            "ranking targets",
            panel_name,
        )


        ranking_input = (
            feature_df[
                [
                    "Date",
                    "Ticker",
                    "Close",
                ]
            ]
            .copy()
        )


        ranking_input = (
            all_ranking_targets(
                ranking_input,
                ticker_col="Ticker",
                date_col="Date",
                price_col="Close",
            )
        )


        ranking_available = [
            target
            for target
            in missing_targets
            if target
            in ranking_input.columns
        ]


        if ranking_available:

            ranking_targets = (
                ranking_input[
                    [
                        "Date",
                        "Ticker",
                    ]
                    + ranking_available
                ]
                .copy()
            )


            target_panel = (
                target_panel.merge(
                    ranking_targets,
                    on=[
                        "Date",
                        "Ticker",
                    ],
                    how="left",
                )
            )


    missing_targets = [
        target
        for target
        in selected_targets
        if target
        not in target_panel.columns
    ]


    if missing_targets:

        raise ValueError(
            f"\nCould not recreate selected "
            f"{panel_name} targets:\n"
            + "\n".join(
                missing_targets
            )
        )


    target_panel = (
        target_panel[
            [
                "Date",
                "Ticker",
            ]
            + selected_targets
        ]
        .copy()
    )


    result = (
        feature_df.merge(
            target_panel,
            on=[
                "Date",
                "Ticker",
            ],
            how="left",
        )
    )


    return result


########################################
# Build Targets For BOTH Splits
########################################

training_df = (
    add_selected_targets(
        feature_df=(
            training_features_df
        ),
        universe=train_tokens,
        split_end=train_end,
        panel_name="TRAINING",
    )
)


backtest_df = (
    add_selected_targets(
        feature_df=(
            backtest_features_df
        ),
        universe=backtest_tokens,
        split_end=backtest_end,
        panel_name="BACKTEST",
    )
)


training_df[
    "Split"
] = "TRAIN"


backtest_df[
    "Split"
] = "BACKTEST"


model_data = pd.concat(
    [
        training_df,
        backtest_df,
    ],
    ignore_index=True,
)


########################################
# Build Selected Model Metadata
########################################

def _clean_parameters(
    parameters,
):

    if (
        parameters is None
        or (
            isinstance(
                parameters,
                float,
            )
            and pd.isna(
                parameters
            )
        )
    ):
        return "{}"

    return str(
        parameters
    )


selected_models_df = (
    selected_model_results[
        [
            "Target",
            "Model",
            "Parameters",
            "Portfolio Target Type",
            "Horizon",
            "Quality Score",
        ]
    ]
    .copy()
    .rename(
        columns={
            "Portfolio Target Type":
                "Target Type",
        }
    )
)


selected_models_df[
    "Parameters"
] = selected_models_df[
    "Parameters"
].map(
    _clean_parameters
)


selected_models_df[
    "Quality Score"
] = pd.to_numeric(
    selected_models_df[
        "Quality Score"
    ],
    errors="coerce",
).clip(
    lower=0.0,
    upper=1.0,
)


# Keep exactly the requested public metadata format.
selected_models_df = selected_models_df[
    [
        "Target",
        "Model",
        "Parameters",
        "Target Type",
        "Horizon",
        "Quality Score",
    ]
].copy()


# Separate helper table preserves the target-specific feature list
# without adding extra columns to Selected Models {STOCK_TYPE}.
selected_model_features_df = pd.DataFrame(
    [
        {
            "Target": target,
            "Features": json.dumps(
                target_features[
                    target
                ]
            ),
        }
        for target in selected_targets
    ]
)


required_features_df = pd.DataFrame(
    {
        "Feature Order":
            range(
                len(
                    required_features
                )
            ),

        "Feature":
            required_features,
    }
)


########################################
# Configuration Metadata
########################################

config = {
    "stock_type":
        STOCK_TYPE,

    "stock_type_index":
        str(
            stock_type_index
        ),

    "train_tokens":
        json.dumps(
            train_tokens
        ),

    "backtest_tokens":
        json.dumps(
            backtest_tokens
        ),

    "train_start":
        str(
            train_start.date()
        ),

    "train_end":
        str(
            train_end.date()
        ),

    "backtest_start":
        str(
            backtest_start.date()
        ),

    "backtest_end":
        str(
            backtest_end.date()
        ),

    "default_rebalance_every":
        str(
            DEFAULT_REBALANCE
        ),

    "feature_warmup_years":
        str(
            FEATURE_WARMUP_YEARS
        ),

    "selected_feature_count":
        str(
            len(
                required_features
            )
        ),

    "selected_model_count":
        str(
            len(
                selected_models_df
            )
        ),

    "selected_models_table":
        SELECTED_MODELS_TABLE,

    "selected_model_features_table":
        SELECTED_MODEL_FEATURES_TABLE,

    "database_created_at":
        str(
            pd.Timestamp.now()
        ),
}


config_df = pd.DataFrame(
    {
        "Key":
            list(
                config.keys()
            ),

        "Value":
            list(
                config.values()
            ),
    }
)


########################################
# Benchmark Data
########################################

benchmark_df = (
    market_df[
        [
            "Close"
        ]
    ]
    .copy()
)


benchmark_df[
    "Return"
] = (
    benchmark_df[
        "Close"
    ]
    .pct_change()
)


benchmark_df[
    "Date"
] = (
    benchmark_df.index
)


benchmark_df = (
    benchmark_df[
        [
            "Date",
            "Close",
            "Return",
        ]
    ]
    .reset_index(
        drop=True
    )
)


########################################
# Save Database
########################################

DATA_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


logger.info(
    "Writing cached data to %s",
    OUTPUT_DB,
)


with sqlite3.connect(
    OUTPUT_DB
) as connection:

    model_data.to_sql(
        "Model_Data",
        connection,
        if_exists="replace",
        index=False,
    )


    benchmark_df.to_sql(
        "Benchmark",
        connection,
        if_exists="replace",
        index=False,
    )


    selected_models_df.to_sql(
        SELECTED_MODELS_TABLE,
        connection,
        if_exists="replace",
        index=False,
    )


    selected_model_features_df.to_sql(
        SELECTED_MODEL_FEATURES_TABLE,
        connection,
        if_exists="replace",
        index=False,
    )


    required_features_df.to_sql(
        "Required_Features",
        connection,
        if_exists="replace",
        index=False,
    )


    config_df.to_sql(
        "Config",
        connection,
        if_exists="replace",
        index=False,
    )


    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS
        idx_model_data_split_date_ticker
        ON Model_Data (Split, Date, Ticker)
        """
    )


    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS
        idx_benchmark_date
        ON Benchmark (Date)
        """
    )


    connection.commit()
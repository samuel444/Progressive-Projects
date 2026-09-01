import logging
import warnings

import numpy as np
import pandas as pd
import yfinance as yf

from features import *
from targets import *


import ast
import gc
import itertools
import logging
import math
import sqlite3
from pathlib import Path

from main_package import (
    benchmark_metrics,
    create_models_and_predictions,
    run_portfolio_backtest_from_predictions,
)



warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger(__name__)


########################################
# Settings
########################################

MARKET_TICKER = "^GSPC"

# History before the requested start date is used only to warm up
# rolling features. It is removed from the final dataframes.
FEATURE_WARMUP_YEARS = 3

# Extra calendar history after the requested end date allows forward
# targets to be calculated near the end of a historical request.
# If the end date is close to today, unavailable future targets will
# correctly remain NaN.
TARGET_LOOKAHEAD_DAYS = 400


DEFAULT_TOKENS = [
    "AAPL", "MSFT", "AMZN", "NVDA", "META", "TSLA", "AMD", "INTC", "QCOM", "MU",
    "CSCO", "ORCL", "JPM", "BAC", "WFC", "C", "XOM", "CVX", "F", "GM",
    "T", "VZ", "PFE", "JNJ", "WMT", "DIS", "GE", "HD", "NFLX", "GOOG"
]


########################################
# Questions
########################################

def get_tokens(
    prompt,
    default,
):

    print("\nDefault:")
    print(", ".join(default))

    value = input(
        f"\n{prompt} "
        "(comma separated, Enter for default): "
    ).strip()

    if not value:
        tokens = default.copy()

    else:
        tokens = list(
            dict.fromkeys(
                token.strip().upper()
                for token in value.split(",")
                if token.strip()
            )
        )

    if MARKET_TICKER in tokens:
        logger.warning(
            "%s removed from the stock universe because it is "
            "downloaded separately as the market benchmark.",
            MARKET_TICKER,
        )

        tokens = [
            token
            for token in tokens
            if token != MARKET_TICKER
        ]

    if not tokens:
        raise ValueError(
            "At least one stock ticker must be supplied."
        )

    return tokens


def get_date(prompt):

    while True:

        value = input(prompt).strip()

        try:
            return pd.Timestamp(value)

        except Exception:
            print(
                "Please enter the date as YYYY-MM-DD."
            )


print("\n" + "=" * 70)
print("MARKET AND STOCK DATA")
print("=" * 70)


tokens = get_tokens(
    "Stock tickers:",
    DEFAULT_TOKENS,
)


start_date = get_date(
    "Start date (YYYY-MM-DD): "
)


end_date = get_date(
    "End date   (YYYY-MM-DD): "
)


TRAIN_END = get_date(
    "Training data end date (YYYY-MM-DD): "
)


BACKTEST_START = get_date(
    "Backtest start date (YYYY-MM-DD): "
)


BACKTEST_END = get_date(
    "Backtest end date (YYYY-MM-DD): "
)

if end_date <= start_date:
    raise ValueError(
        "End date must be after start date."
    )


########################################
# Download Market And Stocks Once
########################################

download_start = (
    start_date
    )



# yfinance accepts a future end date, but naturally only returns data
# that currently exists. Adding one day keeps its exclusive end date
# behaviour from dropping the latest available trading date.
download_end = end_date 


symbols = list(
    dict.fromkeys(
        tokens
        + [MARKET_TICKER]
    )
)


logger.info(
    "Downloading %d stocks and %s | %s -> %s",
    len(tokens),
    MARKET_TICKER,
    download_start.date(),
    download_end.date(),
)


raw_download = yf.download(
    symbols,
    start=download_start.strftime("%Y-%m-%d"),
    end=download_end.strftime("%Y-%m-%d"),
    auto_adjust=True,
    progress=False,
    group_by="ticker",
    multi_level_index=True,
)


if raw_download.empty:
    raise ValueError(
        "yfinance returned no data."
    )


logger.info(
    "Download complete | rows=%d | symbol groups=%d",
    len(raw_download),
    raw_download.columns.get_level_values(0).nunique(),
)


downloaded_symbols = set(
    raw_download
    .columns
    .get_level_values(0)
)


if MARKET_TICKER not in downloaded_symbols:
    raise ValueError(
        f"Market benchmark {MARKET_TICKER} was not returned by yfinance."
    )


raw_market = (
    raw_download[MARKET_TICKER]
    .copy()
    .dropna(how="all")
)


raw_market.index = pd.to_datetime(
    raw_market.index
)


raw_market.index.name = "Date"


########################################
# Feature Helpers
########################################

def run_step(
    dataframe,
    step_name,
    function,
    panel_name,
):

    columns_before = len(dataframe.columns)

    dataframe = function(dataframe)

    logger.debug(
        "%s | %s complete | columns added=%d | total columns=%d",
        panel_name,
        step_name,
        len(dataframe.columns) - columns_before,
        len(dataframe.columns),
    )

    return dataframe


def add_all_individual_features(
    dataframe,
    benchmark_df,
    panel_name,
    include_market_comparison=True,
):

    dataframe = dataframe.copy()

    dataframe["Return"] = (
        dataframe["Close"].pct_change()
    )

    feature_steps = [
        (
            "return features",
            lambda x: all_return_features(x),
        ),
        (
            "momentum features",
            lambda x: all_momentum_features(x),
        ),
        (
            "volatility features",
            lambda x: all_volatility_features(x),
        ),
        (
            "range-volatility features",
            lambda x: all_range_volatility_features(x),
        ),
        (
            "trend features",
            lambda x: all_trend_features(x),
        ),
        (
            "moving-average features",
            lambda x: all_moving_average_features(x),
        ),
        (
            "drawdown features",
            lambda x: all_drawdown_features(x),
        ),
        (
            "distribution features",
            lambda x: all_distribution_features(x),
        ),
        (
            "tail-risk features",
            lambda x: all_tail_risk_features(x),
        ),
        (
            "volume features",
            lambda x: all_volume_features(x),
        ),
        (
            "liquidity features",
            lambda x: all_liquidity_features(x),
        ),
        (
            "OHLC features",
            lambda x: all_ohlc_features(x),
        ),
    ]

    if include_market_comparison:
        feature_steps.extend(
            [
                (
                    "market-relative features",
                    lambda x: all_market_relative_features(
                        x,
                        market_df=benchmark_df,
                    ),
                ),
                (
                    "beta features",
                    lambda x: all_beta_features(
                        x,
                        market_df=benchmark_df,
                    ),
                ),
                (
                    "correlation features",
                    lambda x: all_correlation_features(
                        x,
                        market_df=benchmark_df,
                    ),
                ),
                (
                    "residual features",
                    lambda x: all_residual_features(
                        x,
                        market_df=benchmark_df,
                    ),
                ),
            ]
        )

    feature_steps.extend(
        [
            (
                "technical features",
                lambda x: all_technical_features(x),
            ),
            (
                "regime features",
                lambda x: all_regime_features(x),
            ),
            (
                "interaction features",
                lambda x: all_interaction_features(x),
            ),
            (
                "composite features",
                lambda x: all_composite_features(x),
            ),
            (
                "experimental features",
                lambda x: all_experimental_features(x),
            ),
        ]
    )

    for step_name, feature_function in feature_steps:
        dataframe = run_step(
            dataframe,
            step_name,
            feature_function,
            panel_name,
        )

    return dataframe


def create_market_breadth_and_dispersion(
    raw_stock_frames,
):

    if len(raw_stock_frames) <= 1:
        logger.warning(
            "Breadth and dispersion features require more than one usable stock."
        )

        return pd.DataFrame(
            index=raw_market.index
        )

    wide_parts = []

    for ticker, stock_df in raw_stock_frames.items():
        part = stock_df[
            [
                "Open",
                "High",
                "Low",
                "Close",
                "Volume",
            ]
        ].copy()

        part["Return"] = part["Close"].pct_change()

        part.columns = pd.MultiIndex.from_product(
            [
                part.columns,
                [ticker],
            ]
        )

        wide_parts.append(part)

    wide = pd.concat(
        wide_parts,
        axis=1,
    ).sort_index()

    original_columns = set(wide.columns)

    wide = all_breadth_features(wide)

    wide = all_dispersion_features(wide)

    market_feature_columns = [
        column
        for column in wide.columns
        if column not in original_columns
    ]

    market_features = wide[
        market_feature_columns
    ].copy()

    market_features.columns = [
        column[0]
        if isinstance(column, tuple)
        else column
        for column in market_features.columns
    ]

    market_features = market_features.loc[
        :,
        ~market_features.columns.duplicated(),
    ]

    return market_features


########################################
# Target Helpers
########################################

def add_all_time_series_targets(
    dataframe,
    benchmark_df,
    panel_name,
):

    target_steps = [
        (
            "return targets",
            lambda x: all_return_targets(
                x,
                benchmark_df=benchmark_df,
            ),
        ),
        (
            "volatility targets",
            lambda x: all_volatility_targets(x),
        ),
        (
            "direction targets",
            lambda x: all_direction_targets(x),
        ),
        (
            "barrier targets",
            lambda x: all_barrier_targets(x),
        ),
        (
            "excursion targets",
            lambda x: all_excursion_targets(x),
        ),
        (
            "drawdown targets",
            lambda x: all_drawdown_targets(x),
        ),
        (
            "risk-adjusted targets",
            lambda x: all_risk_adjusted_targets(x),
        ),
    ]

    for step_name, target_function in target_steps:
        dataframe = run_step(
            dataframe,
            step_name,
            target_function,
            panel_name,
        )

    return dataframe


########################################
# Prepare Raw Stock Frames
########################################

raw_stock_frames = {}


for ticker in tokens:

    if ticker not in downloaded_symbols:
        logger.warning(
            "%s skipped because it was not returned by yfinance.",
            ticker,
        )
        continue

    ticker_df = (
        raw_download[ticker]
        .copy()
        .dropna(how="all")
    )

    ticker_df.index = pd.to_datetime(
        ticker_df.index
    )

    ticker_df.index.name = "Date"

    if ticker_df.empty:
        logger.warning(
            "%s skipped because it has no observations.",
            ticker,
        )
        continue

    raw_stock_frames[ticker] = ticker_df


if not raw_stock_frames:
    raise ValueError(
        "No usable stock data was returned."
    )


logger.info(
    "Raw stock preparation complete | requested=%d | usable=%d",
    len(tokens),
    len(raw_stock_frames),
)


########################################
# Build Market Dataframe
########################################

# Market-relative, beta, correlation and residual functions are not
# applied to the market against itself because they would only create
# constant/degenerate features. Every applicable standalone feature
# family is still applied.

market = raw_market.copy()

market["Return"] = (
    market["Close"]
    .pct_change()
)

market_level_features = (
    create_market_breadth_and_dispersion(
        raw_stock_frames
    )
)

logger.info(
    "S&P500 | beginning all targets"
)


market = add_all_time_series_targets(
    dataframe=market,
    benchmark_df=raw_market,
    panel_name="MARKET",
)


logger.info(
    "MARKET | targets complete | rows=%d | columns=%d",
    len(market),
    len(market.columns),
)


market["Ticker"] = MARKET_TICKER
market["Date"] = market.index

market = market.reset_index(
    drop=True
)



########################################
# Build Individual Stock Dataframes
########################################

stock_parts = []
individual_stock_feature_columns = None


for ticker, raw_stock in raw_stock_frames.items():

    logger.info(
        "STOCKS | %s | beginning all features",
        ticker,
    )

    stock = add_all_individual_features(
        dataframe=raw_stock,
        benchmark_df=raw_market,
        panel_name=f"STOCKS | {ticker}",
        include_market_comparison=True,
    )

    if individual_stock_feature_columns is None:
        individual_stock_feature_columns = set(
            stock.columns
        )

    logger.info(
        "STOCKS | %s | beginning all targets",
        ticker,
    )
    

    stock = add_all_time_series_targets(
        dataframe=stock,
        benchmark_df=raw_market,
        panel_name=f"STOCKS | {ticker}",
    )

    stock["Ticker"] = ticker
    stock["Date"] = stock.index

    stock_parts.append(
        stock.reset_index(drop=True)
    )


stocks = pd.concat(
    stock_parts,
    ignore_index=True,
    sort=False,
)


logger.info(
    "STOCKS | individual panels combined | rows=%d | columns=%d | tickers=%d",
    len(stocks),
    len(stocks.columns),
    stocks["Ticker"].nunique(),
)


########################################
# Add Every Cross-Sectional Stock Feature
########################################

base_columns = {
    "Date",
    "Ticker",
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "Return",
}


cross_sectional_columns = [
    column
    for column in sorted(individual_stock_feature_columns)
    if column not in base_columns
    and column in stocks.columns
    and pd.api.types.is_numeric_dtype(
        stocks[column]
    )
]


if len(raw_stock_frames) > 1:
    logger.info(
        "STOCKS | generating cross-sectional ranks and z-scores for %d features",
        len(cross_sectional_columns),
    )

    stocks = all_cross_sectional_features(
        stocks,
        columns=cross_sectional_columns,
        date_col="Date",
    )


########################################
# Add Every Cross-Sectional Ranking Target
########################################

if len(raw_stock_frames) > 1:
    logger.info(
        "STOCKS | generating all ranking targets"
    )

    stocks = all_ranking_targets(
        stocks,
        ticker_col="Ticker",
        date_col="Date",
        price_col="Close",
    )


########################################
# Add Market Breadth/Dispersion To Stocks
########################################

market_feature_frame = (
    market_level_features
    .reset_index()
)


market_feature_frame = market_feature_frame.rename(
    columns={
        market_feature_frame.columns[0]: "Date"
    }
)


stocks = stocks.merge(
    market_feature_frame,
    on="Date",
    how="left",
    suffixes=("", " Market"),
    validate="many_to_one",
)


logger.info(
    "STOCKS | market breadth/dispersion merged | market columns=%d | rows=%d",
    len(market_feature_frame.columns) - 1,
    len(stocks),
)


########################################
# Clean And Restrict To Requested Dates
########################################

def finish_dataframe(
    dataframe,
    dataframe_name,
):

    dataframe = dataframe.copy()

    dataframe["Date"] = pd.to_datetime(
        dataframe["Date"]
    )

    dataframe = dataframe[
        dataframe["Date"].between(
            start_date,
            end_date,
            inclusive="both",
        )
    ].copy()

    numeric_columns = dataframe.select_dtypes(
        include="number"
    ).columns

    dataframe[numeric_columns] = (
        dataframe[numeric_columns]
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
    )

    dataframe = (
        dataframe
        .sort_values(
            [
                "Date",
                "Ticker",
            ]
        )
        .reset_index(drop=True)
    )

    logger.info(
        "%s complete | rows: %d | columns: %d | tickers: %d | dates: %d",
        dataframe_name,
        len(dataframe),
        len(dataframe.columns),
        dataframe["Ticker"].nunique(),
        dataframe["Date"].nunique(),
    )

    return dataframe


market = finish_dataframe(
    market,
    "MARKET",
)


stocks = finish_dataframe(
    stocks,
    "STOCKS",
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
# Required Existing Dataframes
########################################

if "stocks" not in globals():
    raise NameError(
        "The stocks dataframe must already exist before this code runs."
    )

if "market" not in globals():
    raise NameError(
        "The market dataframe must already exist before this code runs."
    )


stocks = stocks.copy()
market = market.copy()


logger.info(
    "Input dataframes received | stocks=%d rows x %d columns | market=%d rows x %d columns",
    len(stocks),
    len(stocks.columns),
    len(market),
    len(market.columns),
)


required_stock_columns = {
    "Date",
    "Ticker",
    "Close",
    "Return",
}

missing_stock_columns = (
    required_stock_columns
    - set(stocks.columns)
)

if missing_stock_columns:
    raise ValueError(
        "stocks is missing required columns: "
        + ", ".join(sorted(missing_stock_columns))
    )


required_market_columns = {
    "Date",
    "Close",
    "Return",
}

missing_market_columns = (
    required_market_columns
    - set(market.columns)
)

if missing_market_columns:
    raise ValueError(
        "market is missing required columns: "
        + ", ".join(sorted(missing_market_columns))
    )


stocks["Date"] = pd.to_datetime(
    stocks["Date"],
    errors="coerce",
)

market["Date"] = pd.to_datetime(
    market["Date"],
    errors="coerce",
)


stocks = (
    stocks
    .dropna(
        subset=[
            "Date",
            "Ticker",
            "Close",
        ]
    )
    .sort_values(
        [
            "Date",
            "Ticker",
        ]
    )
    .reset_index(drop=True)
)


market = (
    market
    .dropna(
        subset=[
            "Date",
            "Close",
        ]
    )
    .sort_values("Date")
    .reset_index(drop=True)
)


########################################
# Paths And Stock Type
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


STOCK_TYPE = (
    "High Liquidity 30"
    # "Medium Liquidity 30"
    # "Lower Liquidity 30"
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


########################################
# Date Questions
########################################

def get_date(
    prompt,
    default=None,
):

    while True:

        default_text = (
            f" [{pd.Timestamp(default).date()}]"
            if default is not None
            else ""
        )

        value = input(
            f"{prompt}{default_text}: "
        ).strip()

        if not value and default is not None:
            return pd.Timestamp(default)

        try:
            return pd.Timestamp(value)

        except Exception:
            print(
                "Please enter the date as YYYY-MM-DD."
            )


available_start = stocks["Date"].min()
available_end = stocks["Date"].max()



if not (
    available_start
    <= TRAIN_END
    < BACKTEST_START
    <= BACKTEST_END
    <= available_end
):
    raise ValueError(
        "Dates must satisfy:\n"
        "available start <= training end < backtest start "
        "<= backtest end <= available end"
    )


logger.info(
    "Requested training end: %s",
    TRAIN_END.date(),
)

logger.info(
    "Backtest period: %s to %s",
    BACKTEST_START.date(),
    BACKTEST_END.date(),
)


########################################
# Load Selected Feature Lists
########################################

stock_type_index = (
    STOCK_TYPE_INDICES[STOCK_TYPE]
)


with open(
    SELECTED_FEATURES_FILE,
    "r",
) as file:

    selected_feature_lines = (
        file.read().splitlines()
    )


if stock_type_index >= len(selected_feature_lines):
    raise ValueError(
        f"Selected_Features.txt has no line for {STOCK_TYPE}."
    )


selected_feature_line = (
    selected_feature_lines[
        stock_type_index
    ].strip()
)


if not selected_feature_line:
    raise ValueError(
        f"Selected_Features.txt is empty for {STOCK_TYPE}."
    )


selected_features = ast.literal_eval(
    selected_feature_line
)


if not isinstance(selected_features, dict):
    raise ValueError(
        "Selected_Features.txt must contain a Target -> Features dictionary."
    )


logger.info(
    "Selected feature map loaded | targets=%d",
    len(selected_features),
)


########################################
# Load Best Passed Model Per Target
########################################

RESULTS_TABLE = (
    f"{STOCK_TYPE} Passed Test Results"
)


logger.info(
    "Loading passed model results | %s",
    RESULTS_TABLE,
)


with sqlite3.connect(
    FINAL_RESULTS_DB
) as connection:

    test_results = pd.read_sql_query(
        f'SELECT * FROM "{RESULTS_TABLE}"',
        connection,
    )


logger.info(
    "Passed model results loaded | rows=%d | columns=%d",
    len(test_results),
    len(test_results.columns),
)


required_result_columns = {
    "Target",
    "Model",
    "Parameters",
    "Target Type",
    "Portfolio Target Type",
    "Horizon",
    "Quality Score",
}


missing_result_columns = (
    required_result_columns
    - set(test_results.columns)
)


if missing_result_columns:
    raise ValueError(
        "Passed-results table is missing columns: "
        + ", ".join(sorted(missing_result_columns))
    )


selected_models_df = test_results.copy()


selected_models_df = selected_models_df[
    selected_models_df["Target"]
    .astype(str)
    .isin(selected_features)
].copy()


selected_models_df = selected_models_df[
    ~selected_models_df["Model"]
    .astype(str)
    .str.contains(
        "Baseline",
        case=False,
        na=False,
    )
].copy()


selected_models_df["Quality Score"] = (
    pd.to_numeric(
        selected_models_df["Quality Score"],
        errors="coerce",
    )
    .clip(0.0, 1.0)
)


selected_models_df["Horizon"] = pd.to_numeric(
    selected_models_df["Horizon"],
    errors="coerce",
)


selected_models_df["Parameters"] = (
    selected_models_df["Parameters"]
    .where(
        selected_models_df["Parameters"].notna(),
        "{}",
    )
    .astype(str)
)


selected_models_df = selected_models_df.dropna(
    subset=[
        "Target",
        "Model",
        "Quality Score",
        "Horizon",
    ]
).copy()


sort_columns = [
    "Quality Score",
]

sort_ascending = [
    False,
]


if "Predictability Score" in selected_models_df.columns:

    selected_models_df["Predictability Score"] = pd.to_numeric(
        selected_models_df["Predictability Score"],
        errors="coerce",
    )

    sort_columns.append(
        "Predictability Score"
    )

    sort_ascending.append(
        False
    )


selected_models_df = (
    selected_models_df
    .sort_values(
        sort_columns,
        ascending=sort_ascending,
    )
    .drop_duplicates(
        subset=[
            "Target",
        ],
        keep="first",
    )
    .reset_index(drop=True)
)


# main_package expects Target Type to contain the portfolio role and
# Statistical Type to contain continuous/binary/multiclass.
selected_models_df["Statistical Type"] = (
    selected_models_df["Target Type"]
    .astype(str)
    .str.lower()
    .str.strip()
)


selected_models_df["Target Type"] = (
    selected_models_df["Portfolio Target Type"]
    .astype(str)
    .str.upper()
    .str.strip()
)


# Horizon scores are not needed to create Adjusted Signal. The value is
# required by main_package and can be replaced later by optimized scores.
selected_models_df["Horizon Score"] = 1.0


selected_models_df = selected_models_df[
    [
        "Target",
        "Model",
        "Parameters",
        "Target Type",
        "Statistical Type",
        "Portfolio Target Type",
        "Horizon",
        "Horizon Score",
        "Quality Score",
    ]
].copy()


if selected_models_df.empty:
    raise ValueError(
        "No selected production models remain."
    )


########################################
# Validate Targets And Selected Features
########################################

target_features = {}
invalid_targets = []


for _, model_row in selected_models_df.iterrows():

    target = str(model_row["Target"])

    if target not in stocks.columns:
        invalid_targets.append(
            f"{target}: target column missing"
        )
        continue

    features = list(
        selected_features.get(target, [])
    )

    missing_features = [
        feature
        for feature in features
        if feature not in stocks.columns
    ]

    if not features:
        invalid_targets.append(
            f"{target}: no selected features"
        )
        continue

    if missing_features:
        invalid_targets.append(
            f"{target}: missing features {missing_features}"
        )
        continue

    target_features[target] = features


if invalid_targets:
    raise ValueError(
        "stocks cannot support the selected models:\n"
        + "\n".join(invalid_targets)
    )


logger.info(
    "Selected models ready | targets=%d | portfolio types=%d",
    len(selected_models_df),
    selected_models_df["Target Type"].nunique(),
)


########################################
# Helpers
########################################

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


def horizon_key(row):

    return f"{int(float(row['Horizon']))}d"


def purged_training_end(
    requested_train_end,
    backtest_start,
    horizon,
    available_dates,
):

    available_dates = pd.Series(
        pd.to_datetime(available_dates)
    ).drop_duplicates().sort_values().reset_index(drop=True)

    backtest_positions = available_dates.index[
        available_dates >= pd.Timestamp(backtest_start)
    ]

    if len(backtest_positions) == 0:
        raise ValueError(
            "Backtest start is outside the available stock dates."
        )

    backtest_position = int(
        backtest_positions[0]
    )

    cutoff_position = (
        backtest_position
        - int(horizon)
        - 1
    )

    if cutoff_position < 0:
        raise ValueError(
            f"Not enough data to purge a {int(horizon)}-day target."
        )

    leakage_safe_cutoff = available_dates.iloc[
        cutoff_position
    ]

    return min(
        pd.Timestamp(requested_train_end),
        pd.Timestamp(leakage_safe_cutoff),
    )


def apply_horizon_signal_refresh(
    predictions_df,
    rebalance_multiplier,
):

    if not (
        0 < rebalance_multiplier <= 1
    ):
        raise ValueError(
            "rebalance_multiplier must be greater than 0 and no greater than 1."
        )

    required_columns = {
        "Date",
        "Ticker",
        "Portfolio Target Type",
        "Horizon Key",
        "Signal",
    }

    missing_columns = (
        required_columns
        - set(predictions_df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            f"{sorted(missing_columns)}"
        )

    refreshed = (
        predictions_df
        .copy()
        .sort_values(
            [
                "Ticker",
                "Portfolio Target Type",
                "Horizon Key",
                "Date",
            ]
        )
        .reset_index(drop=True)
    )

    group_columns = [
        "Ticker",
        "Portfolio Target Type",
        "Horizon Key",
    ]

    for group_values, group_indexes in refreshed.groupby(
        group_columns,
        sort=False,
    ).groups.items():

        ticker, portfolio_type, horizon = group_values
        horizon = str(horizon).strip().lower()

        if not horizon.endswith("d"):
            raise ValueError(
                f"Daily Horizon Key must end in 'd'. Received {horizon!r} "
                f"for {ticker!r} / {portfolio_type!r}."
            )

        horizon_days = int(
            horizon[:-1]
        )

        refresh_rows = max(
            1,
            int(
                np.ceil(
                    rebalance_multiplier
                    * horizon_days
                )
            ),
        )

        group_indexes = np.asarray(
            list(group_indexes)
        )

        original_signals = refreshed.loc[
            group_indexes,
            "Signal",
        ].to_numpy()

        row_positions = np.arange(
            len(original_signals)
        )

        refresh_start_positions = (
            row_positions
            // refresh_rows
        ) * refresh_rows

        refreshed.loc[
            group_indexes,
            "Signal",
        ] = original_signals[
            refresh_start_positions
        ]

    return (
        refreshed
        .sort_values(
            [
                "Date",
                "Ticker",
                "Portfolio Target Type",
                "Horizon Key",
            ]
        )
        .reset_index(drop=True)
    )


########################################
# Fit Each Selected Model On All Stocks
# And Predict The Requested Backtest
########################################

all_stock_dates = (
    stocks["Date"]
    .drop_duplicates()
    .sort_values()
    .reset_index(drop=True)
)


prediction_parts = []
model_summaries = []
skipped_model_parts = []


for model_number, (_, model_row) in enumerate(
    selected_models_df.iterrows(),
    start=1,
):

    target = str(
        model_row["Target"]
    )

    features = target_features[target]
    horizon = int(
        float(model_row["Horizon"])
    )

    actual_train_end = purged_training_end(
        requested_train_end=TRAIN_END,
        backtest_start=BACKTEST_START,
        horizon=horizon,
        available_dates=all_stock_dates,
    )


    logger.info(
        "[%d/%d] %s | features=%d | requested train end=%s | "
        "purged train end=%s | backtest=%s to %s",
        model_number,
        len(selected_models_df),
        target,
        len(features),
        TRAIN_END.date(),
        actual_train_end.date(),
        BACKTEST_START.date(),
        BACKTEST_END.date(),
    )

    columns = list(
        dict.fromkeys(
            [
                "Date",
                "Ticker",
                "Close",
                "Return",
                target,
            ]
            + features
        )
    )

    target_data = stocks[
        columns
    ].copy()

    train_mask = (
        target_data["Date"]
        <= actual_train_end
    )

    backtest_mask = target_data["Date"].between(
        BACKTEST_START,
        BACKTEST_END,
        inclusive="both",
    )

    target_data = target_data[
        train_mask
        | backtest_mask
    ].copy()

    target_data["Split"] = np.where(
        target_data["Date"] <= actual_train_end,
        "TRAIN",
        "BACKTEST",
    )

    one_model_df = pd.DataFrame(
        [
            model_row.to_dict()
        ]
    )

    prepared = create_models_and_predictions(
        dataframe=target_data,
        selected_models_df=one_model_df,
        model_features={
            target: features,
        },
        strict=True,
    )

    target_predictions = prepared[
        "predictions"
    ].copy()

    target_predictions[
        "Portfolio Target Type"
    ] = model_row[
        "Portfolio Target Type"
    ]

    prediction_parts.append(
        target_predictions
    )

    model_summaries.append(
        prepared["model_summary"]
    )

    if not prepared["skipped_models"].empty:
        skipped_model_parts.append(
            prepared["skipped_models"]
        )

    logger.info(
        "[%d/%d] %s | predictions complete | rows=%d",
        model_number,
        len(selected_models_df),
        target,
        len(target_predictions),
    )

    del target_data
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
    sort=False,
)


model_summary_df = pd.concat(
    model_summaries,
    ignore_index=True,
    sort=False,
)


skipped_models_df = (
    pd.concat(
        skipped_model_parts,
        ignore_index=True,
        sort=False,
    )
    if skipped_model_parts
    else pd.DataFrame()
)


del prediction_parts
del model_summaries
del skipped_model_parts


logger.info(
    "Prediction generation complete | rows=%d | targets=%d | %.1f MB",
    len(predictions_df),
    predictions_df["Target"].nunique(),
    dataframe_memory_mb(predictions_df),
)

selected_model_targets = (
    selected_models_df["Target"]
    .astype(str)
    .drop_duplicates()
    .tolist()
)

########################################
# Training Data
########################################

train_df = stocks[
    stocks["Date"]
    <= actual_train_end
].copy()


logger.info(
    "Training target data prepared | rows=%d | targets=%d | end=%s",
    len(train_df),
    len(selected_model_targets),
    pd.Timestamp(actual_train_end).date(),
)


train_df[
    selected_model_targets
] = (
    train_df[
        selected_model_targets
    ]
    .apply(
        pd.to_numeric,
        errors="coerce",
    )
)


########################################
# Target Mean And Standard Deviation
########################################

target_metrics = (
    train_df[
        selected_model_targets
    ]
    .agg(
        [
            "mean",
            "std",
        ]
    )
    .T
    .rename(
        columns={
            "mean": "Mean",
            "std": "Std",
        }
    )
)


target_metrics.index.name = "Target"


logger.info(
    "Target statistics calculated | targets=%d | missing means=%d | missing stds=%d",
    len(target_metrics),
    int(target_metrics["Mean"].isna().sum()),
    int(target_metrics["Std"].isna().sum()),
)


########################################
# Target Orientation
########################################

TARGET_ORIENTATION = {
    "Forward Return": 1,
    "Forward Log Return": 1,
    "Forward Excess Return": 1,

    "Future Volatility": -1,
    "Future Variance": -1,
    "Future Upside Volatility": 1,
    "Future Downside Volatility": -1,
    "Future Downside Upside Volatility Ratio": -1,

    "Future Mean Absolute Return": 1,
    "Future Maximum Absolute Return": 1,

    "Future Direction": 1,

    "Future Return Above 1 Percent": 1,
    "Future Return Above 2 Percent": 1,
    "Future Return Above 5 Percent": 1,
    "Future Return Above 10 Percent": 1,

    "Three Class Direction 2 Percent": 1,
    "Three Class Direction 5 Percent": 1,

    "Barrier 2.0 -2.0": 1,
    "Barrier 2.0 -5.0": 1,
    "Barrier 5.0 -2.0": 1,
    "Barrier 5.0 -5.0": 1,

    "Volatility Barrier 20 1 1": -1,
    "Volatility Barrier 20 1 2": -1,
    "Volatility Barrier 20 2 1": -1,
    "Volatility Barrier 20 2 2": -1,
    "Volatility Barrier 60 1 1": -1,
    "Volatility Barrier 60 1 2": -1,
    "Volatility Barrier 60 2 1": -1,
    "Volatility Barrier 60 2 2": -1,

    "Maximum Favourable Excursion": 1,
    "Maximum Adverse Excursion": 1,

    "Time To Maximum Favourable Excursion": -1,
    "Time To Maximum Adverse Excursion": 1,

    "Future Maximum Drawdown": 1,
    "Future Minimum Return": 1,

    "Future Return Volatility Ratio": 1,
    "Future Sortino Ratio": 1,
    "Future Return Minus Risk 0.5": 1,
    "Future Return Minus Risk 1": 1,
    "Future Return Minus Risk 2": 1,
    "Future Return Drawdown Ratio": 1,

    "Future Return Rank": 1,

    "Top 20 Percent Future Return": 1,
    "Top 25 Percent Future Return": 1,
    "Bottom 20 Percent Future Return": -1,
    "Bottom 25 Percent Future Return": -1,
}


########################################
# Match Target With Orientation
########################################

def get_target_orientation(
    target,
):
    target_tokens = str(
        target
    ).lower().split()

    matches = []


    for base_target, orientation in (
        TARGET_ORIENTATION.items()
    ):

        base_tokens = (
            base_target
            .lower()
            .split()
        )

        target_iterator = iter(
            target_tokens
        )

        is_match = all(
            any(
                target_token
                == base_token
                for target_token
                in target_iterator
            )
            for base_token
            in base_tokens
        )

        if is_match:

            matches.append(
                (
                    len(base_tokens),
                    orientation,
                    base_target,
                )
            )


    if not matches:

        raise KeyError(
            "No TARGET_ORIENTATION entry "
            f"matched target: {target}"
        )


    # Use the longest and therefore most specific match.
    _, orientation, _ = max(
        matches,
        key=lambda value: value[0],
    )

    return orientation


########################################
# Final Target Values Dictionary
########################################

target_values = {}


for target, metrics in (
    target_metrics.iterrows()
):

    target_values[
        target
    ] = (
        float(
            metrics["Mean"]
        ),
        float(
            metrics["Std"]
        ),
        get_target_orientation(
            target
        ),
    )


########################################
# Create Adjusted Signal
########################################

predictions_df["Date"] = pd.to_datetime(
    predictions_df["Date"]
)


predictions_df["Horizon Key"] = (
    predictions_df.apply(
        horizon_key,
        axis=1,
    )
)


predictions_df["Quality Score"] = (
    pd.to_numeric(
        predictions_df["Quality Score"],
        errors="coerce",
    )
    .clip(0.0, 1.0)
    .fillna(0.0)
)


########################################
# Market Targets In Prediction Format
#
# IMPORTANT:
# These are realised target values, not
# model predictions. They are suitable for
# benchmark comparison and scoring only.
# They must not be used to choose portfolio
# weights because that would leak future
# target information into the backtest.
########################################

selected_model_targets = (
    selected_models_df["Target"]
    .astype(str)
    .drop_duplicates()
    .tolist()
)


available_market_targets = [
    target
    for target in selected_model_targets
    if target in market.columns
]


missing_market_targets = [
    target
    for target in selected_model_targets
    if target not in market.columns
]


if missing_market_targets:
    logger.warning(
        "Market dataframe does not contain %d selected model targets; "
        "they cannot be placed into market_predictions_df: %s",
        len(missing_market_targets),
        missing_market_targets,
    )


if not available_market_targets:
    raise ValueError(
        "The market dataframe does not contain any targets that have selected models."
    )


########################################
# Keep Only Date, Return And Targets
########################################

market = (
    market[
        [
            "Date",
            "Return",
        ]
        + available_market_targets
    ]
    .copy()
)


market = market[
    market["Date"].between(
        BACKTEST_START,
        BACKTEST_END,
        inclusive="both",
    )
].copy()


########################################
# Convert Market Targets From Columns
# Into Rows
########################################

market_predictions_df = market.melt(
    id_vars=[
        "Date",
        "Return",
    ],
    value_vars=available_market_targets,
    var_name="Target",
    value_name="Prediction",
)


logger.info(
    "Market targets reshaped | rows=%d | targets=%d",
    len(market_predictions_df),
    market_predictions_df["Target"].nunique(),
)



market_predictions_df["Ticker"] = "^GSPC"


market_predictions_df = market_predictions_df.dropna(
    subset=[
        "Date",
        "Return"
    ]
).copy()


CONTINUOUS_PORTFOLIO_TARGET_TYPES = {
    "ALPHA",
    "RELATIVE_ALPHA",
    "RISK_ADJUSTED_ALPHA",
    "CROSS_SECTION_ALPHA",

    "VOLATILITY",
    "DOWNSIDE_VOLATILITY",
    "UPSIDE_VOLATILITY",
    "VOLATILITY_ASYMMETRY",
    "ABSOLUTE_MOVE",

    "DOWNSIDE",
    "TAIL_RISK",
    "DOWNSIDE_EXCURSION",
    "UPSIDE_EXCURSION",

    "TIME_TO_DOWNSIDE_EXCURSION",
    "TIME_TO_UPSIDE_EXCURSION",

    "RECOVERY",
    "REVERSAL",

    "EXECUTION",
    "LIQUIDITY",
    "MARKET_IMPACT",
    "CORRELATION",
    "COVARIANCE",
}


BINARY_PORTFOLIO_TARGET_TYPES = {
    "DIRECTION",
    "ALPHA_BINARY",
    "TAIL_EVENT",
    "UPSIDE_EVENT",
    "VOLATILITY_EVENT",
    "CROSS_SECTION_DOWNSIDE",
}


MULTICLASS_PORTFOLIO_TARGET_TYPES = {
    "DIRECTION_MULTICLASS",
    "BARRIER_ALPHA",
    "REGIME",
}


def prediction_to_signal(row):
    if row["Portfolio Target Type"] in CONTINUOUS_PORTFOLIO_TARGET_TYPES:

        metrics = target_values[row["Target"]]

        signal = metrics[2] * ((row["Prediction"] - metrics[0])/metrics[1] )

    elif row["Portfolio Target Type"] in BINARY_PORTFOLIO_TARGET_TYPES:

        target = row["Target"]

        metrics = target_values[target]

        prediction = pd.to_numeric(
            row["Prediction"],
            errors="coerce",
        )

        if pd.isna(prediction):
            signal = 0.0

        else:
            prediction = float(
                np.clip(
                    prediction,
                    0.0,
                    1.0,
                )
            )

            # The mean of a binary training target
            # is its positive-class base probability.
            p0 = metrics[0]

            if (
                not np.isfinite(p0)
                or p0 <= 0.0
                or p0 >= 1.0
            ):
                signal = 0.0

            else:

                if prediction >= p0:

                    signal = (
                        prediction
                        - p0
                    ) / (
                        1.0
                        - p0
                    )

                else:

                    signal = (
                        prediction
                        - p0
                    ) / p0

            signal = metrics[2] * (float(
                np.clip(
                    signal,
                    -1.0,
                    1.0,
                ))
            )

    elif row["Portfolio Target Type"] in MULTICLASS_PORTFOLIO_TARGET_TYPES:

        target = row["Target"]

        prediction = pd.to_numeric(
            row["Prediction"],
            errors="coerce",
        )

        if pd.isna(prediction):
            signal = 0.0

        else:
            class_values = (
                pd.to_numeric(
                    train_df[target],
                    errors="coerce",
                )
                .dropna()
                .unique()
            )

            if len(class_values) < 2:
                signal = 0.0

            else:
                lower_class = float(
                    np.min(class_values)
                )

                upper_class = float(
                    np.max(class_values)
                )

                signal = (
                    2.0
                    * (
                        float(prediction)
                        - lower_class
                    )
                    / (
                        upper_class
                        - lower_class
                    )
                    - 1.0
                )

                signal = target_values[target][2] * (float(
                    np.clip(
                        signal,
                        -1.0,
                        1.0,
                    ))
                )

    else:

        raise ValueError(
            "Unknown Portfolio Target Type: "
            f"{row['Portfolio Target Type']}"
        )

    return signal

########################################
# Stock Signals
########################################

predictions_df["Signal"] = (
    predictions_df.apply(
        prediction_to_signal,
        axis=1,
    )
)


logger.info(
    "Stock signals created | rows=%d | missing=%d",
    len(predictions_df),
    int(predictions_df["Signal"].isna().sum()),
)

predictions_df.dropna(
    subset=["Signal"],
    inplace=True,
)

predictions_df.reset_index(
    drop=True,
    inplace=True,
)

predictions_df["Adjusted Signal"] = (
    predictions_df["Signal"]
    * predictions_df["Quality Score"]
)


predictions_df = (
    predictions_df
    .sort_values(
        [
            "Date",
            "Ticker",
            "Target",
        ]
    )
    .reset_index(
        drop=True
    )
)


########################################
# Prediction Metadata By Target
########################################

prediction_target_metadata = (
    predictions_df[
        [
            "Target",
            "Portfolio Target Type",
            "Horizon Key",
            "Quality Score",
        ]
    ]
    .sort_values(
        [
            "Target",
            "Quality Score",
        ],
        ascending=[
            True,
            False,
        ],
    )
    .drop_duplicates(
        subset=[
            "Target",
        ],
        keep="first",
    )
)


########################################
# Add Metadata To Every Market Row
########################################

market_predictions_df = (
    market_predictions_df
    .drop(
        columns=[
            "Portfolio Target Type",
            "Horizon Key",
            "Quality Score",
        ],
        errors="ignore",
    )
    .merge(
        prediction_target_metadata,
        on="Target",
        how="inner",
        validate="many_to_one",
    )
)


logger.info(
    "Market metadata merged | rows=%d | targets=%d | portfolio types=%d",
    len(market_predictions_df),
    market_predictions_df["Target"].nunique(),
    market_predictions_df["Portfolio Target Type"].nunique(),
)


if market_predictions_df[
    "Portfolio Target Type"
].isna().any():

    missing_targets = (
        market_predictions_df.loc[
            market_predictions_df[
                "Portfolio Target Type"
            ].isna(),
            "Target",
        ]
        .unique()
        .tolist()
    )

    raise ValueError(
        "Missing Portfolio Target Type for "
        f"market targets: {missing_targets}"
    )


########################################
# Market Signals
########################################

market_predictions_df["Signal"] = (
    market_predictions_df.apply(
        prediction_to_signal,
        axis=1,
    )
)


logger.info(
    "Market signals created | rows=%d | missing=%d",
    len(market_predictions_df),
    int(market_predictions_df["Signal"].isna().sum()),
)

market_predictions_df.dropna(
    subset=["Signal"],
    inplace=True,
)

market_predictions_df.reset_index(
    drop=True,
    inplace=True,
)


market_predictions_df[
    "Adjusted Signal"
] = (
    market_predictions_df["Signal"]
    * market_predictions_df["Quality Score"]
)


market_predictions_df = (
    market_predictions_df[
        [
            "Date",
            "Ticker",
            "Return",
            "Target",
            "Portfolio Target Type",
            "Horizon Key",
            "Quality Score",
            "Signal",
            "Adjusted Signal",
        ]
    ]
    .sort_values(
        [
            "Date",
            "Target",
        ]
    )
    .reset_index(
        drop=True
    )
)

########################################
# Market Adjusted Signal
########################################

market_predictions_df["Adjusted Signal"] = (
    market_predictions_df["Signal"]
    * market_predictions_df["Quality Score"]
)


market_predictions_df = (
    market_predictions_df[
        [
            "Date",
            "Ticker",
            "Return",
            "Target",
            "Portfolio Target Type",
            "Horizon Key",
            "Quality Score",
            "Signal",
            "Adjusted Signal",
        ]
    ]
    .sort_values(
        [
            "Date",
            "Target",
        ]
    )
    .reset_index(drop=True)
)


########################################
# Group Market Like Stock Predictions
########################################

grouped_market_predictions_df = (
    market_predictions_df
    .groupby(
        [
            "Date",
            "Ticker",
            "Portfolio Target Type",
            "Horizon Key",
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


########################################
# Aggregate Target Models By Type/Horizon
########################################

grouped_predictions_df = (
    predictions_df[
        [
            "Date",
            "Ticker",
            "Return",
            "Portfolio Target Type",
            "Horizon Key",
            "Adjusted Signal",
        ]
    ]
    .groupby(
        [
            "Date",
            "Ticker",
            "Portfolio Target Type",
            "Horizon Key",
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


logger.info(
    "Prediction aggregation complete | stock rows=%d | market rows=%d",
    len(grouped_predictions_df),
    len(grouped_market_predictions_df),
)




########################################
# Matching Market Backtest Data
########################################

market_backtest = (
    market[
        market["Date"].between(
            BACKTEST_START,
            BACKTEST_END,
            inclusive="both",
        )
    ]
    .copy()
    .sort_values("Date")
    .reset_index(drop=True)
)


########################################
# Final In-Memory Outputs
########################################

logger.info(
    "Ready | predictions_df=%d rows | grouped_predictions_df=%d rows | "
    "market_predictions_df=%d rows | grouped_market_predictions_df=%d rows | "
    "market_backtest=%d rows",
    len(predictions_df),
    len(grouped_predictions_df),
    len(market_predictions_df),
    len(grouped_market_predictions_df),
    len(market_backtest),
)

########################################
# Save Market And Stock Predictions
########################################

BACKTEST_DATABASE = Path(
    "/Users/sam/Progressive-Projects/Projects/"
    "Equity Selector/data/Backtest_Database.db"
)


with sqlite3.connect(
    BACKTEST_DATABASE
) as connection:

    grouped_market_predictions_df.to_sql(
        "Market",
        connection,
        if_exists="replace",
        index=False,
    )

    grouped_predictions_df.to_sql(
        "Stocks",
        connection,
        if_exists="replace",
        index=False,
    )


logger.info(
    "Saved backtest data | database=%s | "
    "market rows=%d | stocks rows=%d",
    BACKTEST_DATABASE,
    len(grouped_market_predictions_df),
    len(grouped_predictions_df),
)

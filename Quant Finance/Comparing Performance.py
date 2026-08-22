import ast
import logging
import re
import sqlite3
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from lightgbm import LGBMClassifier, LGBMRegressor
from scipy.optimize import minimize
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import (
    ElasticNet,
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
)
from sklearn.preprocessing import StandardScaler

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

# SQL is now used ONLY for the previously selected model specification:
# target, model name, parameters, target type and predictability score.
FINAL_RESULTS_DB = (
    "/Users/sam/Progressive-Projects/Projects/Equity Selector/"
    "data/Final_Test_Results.db"
)

# This is used ONLY for the selected feature list belonging to each target.
SELECTED_FEATURES_FILE = (
    "/Users/sam/Progressive-Projects/Projects/Equity Selector/"
    "data/Selected_Features.txt"
)


########################################
# Settings
########################################

# Raw history before train/backtest start used only to warm up rolling features.
FEATURE_WARMUP_YEARS = 3

# Required gap between the end of model-training data and the start of the
# backtest. This is deliberately conservative and helps protect forward-label
# horizons from overlapping the backtest.
TRAINING_EMBARGO_YEARS = 1

# Portfolio objective.
ALPHA_IMPORTANCE = 0.50
VOLATILITY_IMPORTANCE = 0.25
DOWNSIDE_IMPORTANCE = 0.25
CONCENTRATION_PENALTY = 0.15

DEFAULT_REBALANCE_EVERY = 60


########################################
# Model Helpers
########################################

def parse_parameters(parameters):
    if parameters is None:
        return {}

    if isinstance(parameters, dict):
        return parameters.copy()

    try:
        if pd.isna(parameters):
            return {}
    except (TypeError, ValueError):
        pass

    parameters = str(parameters).strip()

    if parameters == "":
        return {}

    parameters = re.sub(
        r"\bnull\b",
        "None",
        parameters,
        flags=re.IGNORECASE,
    )
    parameters = re.sub(
        r"\btrue\b",
        "True",
        parameters,
        flags=re.IGNORECASE,
    )
    parameters = re.sub(
        r"\bfalse\b",
        "False",
        parameters,
        flags=re.IGNORECASE,
    )

    parsed = ast.literal_eval(parameters)

    return {
        key.replace("model__", "", 1): value
        for key, value in parsed.items()
    }


def build_model(model_name, target_type, parameters):
    """
    Rebuild the previously selected model specification using the exact
    hyperparameters stored in Final_Test_Results.db.

    The fitted object itself is NOT reused. It is refitted on the fresh
    yfinance training universe selected for this run.
    """
    params = parse_parameters(parameters)

    name = model_name.lower().strip()
    is_regression = target_type.lower() == "continuous"

    if name in {"linear regression", "ols"}:
        return LinearRegression(**params)

    if name == "ridge":
        return Ridge(**params)

    if name == "lasso":
        return Lasso(**params)

    if name in {"elastic net", "elasticnet"}:
        return ElasticNet(**params)

    if name in {"logistic regression", "logistic"}:
        params.setdefault("max_iter", 5000)
        return LogisticRegression(**params)

    if name in {"l2 logistic regression", "l2 logistic"}:
        params.setdefault("penalty", "l2")
        params.setdefault("solver", "lbfgs")
        params.setdefault("max_iter", 5000)
        return LogisticRegression(**params)

    if name in {"l1 logistic regression", "l1 logistic"}:
        params.setdefault("penalty", "l1")
        params.setdefault("solver", "liblinear")
        params.setdefault("max_iter", 5000)
        return LogisticRegression(**params)

    if name in {
        "elastic net logistic regression",
        "elasticnet logistic regression",
        "elastic net logistic",
    }:
        params.setdefault("penalty", "elasticnet")
        params.setdefault("solver", "saga")
        params.setdefault("max_iter", 5000)
        return LogisticRegression(**params)

    if name in {
        "random forest",
        "random forest regressor",
        "random forest classifier",
    }:
        if is_regression:
            return RandomForestRegressor(**params)
        return RandomForestClassifier(**params)

    if name in {
        "extra trees",
        "extra trees regressor",
        "extra trees classifier",
    }:
        if is_regression:
            return ExtraTreesRegressor(**params)
        return ExtraTreesClassifier(**params)

    if name in {
        "gradient boosting",
        "gradient boosting regressor",
        "gradient boosting classifier",
    }:
        if is_regression:
            return GradientBoostingRegressor(**params)
        return GradientBoostingClassifier(**params)

    if name in {
        "hist gradient boosting",
        "histogram gradient boosting",
        "histgradientboosting",
    }:
        if is_regression:
            return HistGradientBoostingRegressor(**params)
        return HistGradientBoostingClassifier(**params)

    if name in {"lightgbm", "lgbm"}:
        if is_regression:
            return LGBMRegressor(**params)
        return LGBMClassifier(**params)

    raise ValueError(
        f"Unknown model '{model_name}'. "
        "Add its constructor to build_model()."
    )


########################################
# Load Previously Selected Research Results
########################################

logger.info("Loading previously selected model specifications")

with sqlite3.connect(FINAL_RESULTS_DB) as connection:
    test_results = pd.read_sql_query(
        "SELECT * FROM 'Most Predictable Results'",
        connection,
    )

with open(SELECTED_FEATURES_FILE, "r") as file:
    selected_features = ast.literal_eval(file.read())


########################################
# Choose Alpha / Volatility / Downside Targets
########################################

def choose_target(prediction_type):
    available = (
        test_results[
            test_results["Prediction Type"].str.lower()
            == prediction_type.lower()
        ]
        .copy()
    )

    # Do not allow a baseline to become the live/backtest model.
    available = available[
        ~available["Model"].str.contains(
            "Baseline",
            case=False,
            na=False,
        )
    ]

    # The current portfolio optimiser needs one scalar per stock.
    available = available[
        available["Target Type"].isin(
            ["continuous", "binary"]
        )
    ]

    available = (
        available
        .sort_values(
            "Predictability Score",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    if available.empty:
        raise ValueError(
            f"No usable targets are available for {prediction_type}."
        )

    print(f"\n{'=' * 70}")
    print(f"{prediction_type.upper()} TARGETS")
    print("=" * 70)

    for i, row in available.iterrows():
        print(
            f"{i + 1}. {row['Target']} "
            f"| {row['Model']} "
            f"| Score: {row['Predictability Score']:.3f}"
        )

    while True:
        try:
            choice = int(
                input(
                    f"\nWhich target would you like to use for "
                    f"{prediction_type}? "
                )
            )

            if 1 <= choice <= len(available):
                return available.iloc[choice - 1]

        except ValueError:
            pass

        print("Please enter one of the numbers shown above.")


alpha_result = choose_target("alpha")
volatility_result = choose_target("volatility")
downside_result = choose_target("downside")


########################################
# Selected Feature Union
########################################

alpha_features = list(
    selected_features[
        alpha_result["Target"]
    ]
)

volatility_features = list(
    selected_features[
        volatility_result["Target"]
    ]
)

downside_features = list(
    selected_features[
        downside_result["Target"]
    ]
)

required_features = list(
    dict.fromkeys(
        alpha_features
        + volatility_features
        + downside_features
    )
)

selected_targets = list(
    dict.fromkeys(
        [
            alpha_result["Target"],
            volatility_result["Target"],
            downside_result["Target"],
        ]
    )
)

logger.info(
    "Selected-feature union contains %d unique features",
    len(required_features),
)


########################################
# Choose Training + Backtest Data
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


def get_tokens(prompt, default):
    print("\nDefault:")
    print(", ".join(default))

    value = input(
        f"\n{prompt} "
        "(comma separated, Enter for default): "
    ).strip()

    if not value:
        return default.copy()

    return list(
        dict.fromkeys(
            token.strip().upper()
            for token in value.split(",")
            if token.strip()
        )
    )


def get_date(prompt):
    while True:
        value = input(prompt).strip()

        try:
            return pd.Timestamp(value)
        except Exception:
            print("Please enter the date as YYYY-MM-DD.")


print("\n" + "=" * 70)
print("TRAINING DATA")
print("=" * 70)

train_tokens = get_tokens(
    "Training tickers:",
    DEFAULT_TRAIN_TOKENS,
)

train_start = get_date(
    "Training start date (YYYY-MM-DD): "
)

train_end = get_date(
    "Training end date   (YYYY-MM-DD): "
)

if train_end <= train_start:
    raise ValueError(
        "Training end date must be after training start date."
    )


print("\n" + "=" * 70)
print("BACKTEST DATA")
print("=" * 70)

backtest_tokens = get_tokens(
    "Backtest tickers:",
    DEFAULT_BACKTEST_TOKENS,
)

backtest_start = get_date(
    "Backtest start date (YYYY-MM-DD): "
)

backtest_end = get_date(
    "Backtest end date   (YYYY-MM-DD): "
)

if backtest_end <= backtest_start:
    raise ValueError(
        "Backtest end date must be after backtest start date."
    )


########################################
# Leakage Protection
########################################

minimum_backtest_start = train_end

if backtest_start < minimum_backtest_start:
    raise ValueError(
        "\nBacktest starts too close to training.\n"
        f"Training ends:          {train_end.date()}\n"
        f"Earliest backtest date: {minimum_backtest_start.date()}\n"
        f"Embargo:                 "
        f"{TRAINING_EMBARGO_YEARS} year(s)\n"
    )


rebalance_input = input(
    f"Rebalance every N trading days "
    f"[{DEFAULT_REBALANCE_EVERY}]: "
).strip()

REBALANCE_EVERY = (
    int(rebalance_input)
    if rebalance_input
    else DEFAULT_REBALANCE_EVERY
)

if REBALANCE_EVERY <= 0:
    raise ValueError(
        "Rebalance frequency must be greater than zero."
    )


########################################
# ONE yfinance Download
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
    + pd.Timedelta(days=1)
)

symbols = list(
    dict.fromkeys(
        all_tokens
        + ["^GSPC"]
    )
)

logger.info(
    "Downloading %d stocks + S&P 500 in one yfinance call",
    len(all_tokens),
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

if "^GSPC" not in raw_download.columns.get_level_values(0):
    raise ValueError(
        "S&P 500 (^GSPC) was not returned by yfinance."
    )

market_df = (
    raw_download["^GSPC"]
    .copy()
    .dropna(how="all")
)

market_df.index = pd.to_datetime(
    market_df.index
)


########################################
# Efficient Feature Generation
########################################

# These are the same feature-family dependencies as the original research
# pipeline, but we only run as far as necessary for the union of selected
# features. All unused engineered columns are discarded before model fitting.
#
# This is deliberately dependency-safe: later derived features may require
# columns produced by earlier families.

def selected_features_complete(dataframe):
    return set(
        required_features
    ).issubset(
        dataframe.columns
    )


def build_individual_features(
    stock_df,
    benchmark_df,
):
    """
    Build feature families in original research order, stopping as soon as
    every feature required by the three selected models exists.
    """
    stock_df = stock_df.copy()

    stock_df["Return"] = (
        stock_df["Close"]
        .pct_change()
    )

    feature_steps = [
        (
            "return",
            lambda x: all_return_features(x),
        ),
        (
            "momentum",
            lambda x: all_momentum_features(x),
        ),
        (
            "volatility",
            lambda x: all_volatility_features(x),
        ),
        (
            "range volatility",
            lambda x: all_range_volatility_features(x),
        ),
        (
            "trend",
            lambda x: all_trend_features(x),
        ),
        (
            "moving average",
            lambda x: all_moving_average_features(x),
        ),
        (
            "drawdown",
            lambda x: all_drawdown_features(x),
        ),
        (
            "distribution",
            lambda x: all_distribution_features(x),
        ),
        (
            "tail risk",
            lambda x: all_tail_risk_features(x),
        ),
        (
            "volume",
            lambda x: all_volume_features(x),
        ),
        (
            "liquidity",
            lambda x: all_liquidity_features(x),
        ),
        (
            "ohlc",
            lambda x: all_ohlc_features(x),
        ),
        (
            "market relative",
            lambda x: all_market_relative_features(
                x,
                market_df=benchmark_df,
            ),
        ),
        (
            "beta",
            lambda x: all_beta_features(
                x,
                market_df=benchmark_df,
            ),
        ),
        (
            "correlation",
            lambda x: all_correlation_features(
                x,
                market_df=benchmark_df,
            ),
        ),
        (
            "residual",
            lambda x: all_residual_features(
                x,
                market_df=benchmark_df,
            ),
        ),
        (
            "technical",
            lambda x: all_technical_features(x),
        ),
        (
            "regime",
            lambda x: all_regime_features(x),
        ),
        (
            "interaction",
            lambda x: all_interaction_features(x),
        ),
        (
            "composite",
            lambda x: all_composite_features(x),
        ),
        (
            "experimental",
            lambda x: all_experimental_features(x),
        ),
    ]

    for step_name, feature_function in feature_steps:
        if selected_features_complete(
            stock_df
        ):
            logger.debug(
                "Selected-feature union complete; "
                "stopping before %s features",
                step_name,
            )
            break

        stock_df = feature_function(
            stock_df
        )

    return stock_df


def add_cross_stock_features_if_needed(
    panel,
):
    """
    Build cross-sectional and breadth/dispersion features only when the
    selected-feature union still contains columns that have not yet been
    created by the individual-stock feature pipeline.
    """
    if set(
        required_features
    ).issubset(
        panel.columns
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

    if panel["Ticker"].nunique() > 1:
        logger.info(
            "Selected-feature union requires "
            "cross-sectional features"
        )

        individual_feature_columns = [
            column
            for column in panel.columns
            if column not in base_columns
        ]

        panel = all_cross_sectional_features(
            panel,
            columns=individual_feature_columns,
            date_col="Date",
        )

    if set(
        required_features
    ).issubset(
        panel.columns
    ):
        return panel

    if panel["Ticker"].nunique() <= 1:
        return panel

    logger.info(
        "Selected-feature union requires "
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
    """
    Build a feature panel for exactly one universe.

    Training and backtest universes are built separately so that
    cross-sectional/breadth features in training cannot accidentally use
    backtest-only stocks, and vice versa.
    """
    logger.info(
        "Building %s feature panel for %d requested stocks",
        panel_name,
        len(universe),
    )

    warmup_start = (
        start_date
        - pd.DateOffset(
            years=FEATURE_WARMUP_YEARS
        )
    )

    benchmark_slice = market_df[
        (market_df.index >= warmup_start)
        & (market_df.index <= end_date)
    ].copy()

    downloaded_symbols = set(
        raw_download.columns
        .get_level_values(0)
    )

    stock_dfs = {}

    for token in universe:
        if token not in downloaded_symbols:
            logger.warning(
                "%s skipped: not returned by yfinance",
                token,
            )
            continue

        stock_df = (
            raw_download[token]
            .copy()
            .dropna(how="all")
        )

        stock_df.index = pd.to_datetime(
            stock_df.index
        )

        stock_df = stock_df[
            (stock_df.index >= warmup_start)
            & (stock_df.index <= end_date)
        ].copy()

        if stock_df.empty:
            logger.warning(
                "%s skipped: no observations in requested range",
                token,
            )
            continue

        logger.info(
            "%s | building selected feature pipeline",
            token,
        )

        stock_df = build_individual_features(
            stock_df,
            benchmark_slice,
        )

        stock_dfs[token] = stock_df

    usable_tokens = [
        token
        for token in universe
        if token in stock_dfs
    ]

    if not usable_tokens:
        raise ValueError(
            f"No stocks remain in {panel_name} universe."
        )

    panel_parts = []

    for token in usable_tokens:
        stock_df = stock_dfs[
            token
        ].copy()

        stock_df["Ticker"] = token
        stock_df["Date"] = (
            stock_df.index
        )

        panel_parts.append(
            stock_df.reset_index(
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
        for feature in required_features
        if feature not in panel.columns
    ]

    if missing_features:
        raise ValueError(
            f"\n{panel_name} feature pipeline could not create "
            "the following selected features:\n"
            + "\n".join(
                missing_features
            )
        )

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

    panel = panel[
        columns_to_keep
    ].copy()

    panel["Date"] = pd.to_datetime(
        panel["Date"]
    )

    panel[
        required_features
    ] = (
        panel[
            required_features
        ]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
    )

    # Warm-up rows were needed to create rolling features, but they do not
    # belong in the requested train/backtest period itself.
    panel = panel[
        (panel["Date"] >= start_date)
        & (panel["Date"] <= end_date)
    ].copy()

    return panel, usable_tokens


########################################
# Build Fresh Yfinance Training Features
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
# Build Fresh Yfinance Backtest Features
########################################

backtest_df, backtest_tokens = (
    build_feature_panel(
        universe=backtest_tokens,
        start_date=backtest_start,
        end_date=backtest_end,
        panel_name="BACKTEST",
    )
)

tokens = backtest_tokens


########################################
# Build ONLY The Selected Training Targets
########################################

def build_training_targets(
    feature_df,
):
    """
    Recreate target values from the fresh yfinance training data.

    The target *definitions* are reused from targets.py, but no historical
    feature/target SQL dataframe is used.

    The feature panel ends at train_end before targets are built, so forward
    targets whose outcomes would extend beyond train_end naturally become NaN
    and are later excluded from model fitting.
    """
    logger.info(
        "Generating selected training targets from fresh yfinance data"
    )

    target_parts = []

    # All individual-stock target families are attempted in the same research
    # order. We stop for a stock as soon as all three selected targets have
    # appeared. Cross-sectional ranking targets are handled after panel combine.
    for token in train_tokens:
        token_df = (
            feature_df[
                feature_df["Ticker"]
                == token
            ]
            .sort_values("Date")
            .copy()
        )

        if token_df.empty:
            continue

        token_df = (
            token_df
            .set_index("Date")
        )

        # Keep Ticker out of target functions; retain all market/base columns
        # they may need.
        if "Ticker" in token_df.columns:
            token_df = token_df.drop(
                columns=["Ticker"]
            )

        target_steps = [
            (
                "return",
                lambda x: all_return_targets(
                    x,
                    benchmark_df=market_df[
                        market_df.index
                        <= train_end
                    ],
                ),
            ),
            (
                "volatility",
                lambda x: all_volatility_targets(x),
            ),
            (
                "direction",
                lambda x: all_direction_targets(x),
            ),
            (
                "barrier",
                lambda x: all_barrier_targets(x),
            ),
            (
                "excursion",
                lambda x: all_excursion_targets(x),
            ),
            (
                "drawdown",
                lambda x: all_drawdown_targets(x),
            ),
            (
                "risk adjusted",
                lambda x: all_risk_adjusted_targets(x),
            ),
        ]

        for step_name, target_function in target_steps:
            present = set(
                selected_targets
            ).intersection(
                token_df.columns
            )

            if len(present) == len(
                selected_targets
            ):
                break

            logger.debug(
                "%s | building %s targets",
                token,
                step_name,
            )

            token_df = target_function(
                token_df
            )

        token_df["Ticker"] = token
        token_df["Date"] = (
            token_df.index
        )

        available_selected = [
            target
            for target in selected_targets
            if target in token_df.columns
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
            .reset_index(drop=True)
        )

    if not target_parts:
        raise ValueError(
            "No training target data could be generated."
        )

    target_panel = pd.concat(
        target_parts,
        ignore_index=True,
    )

    missing_targets = [
        target
        for target in selected_targets
        if target not in target_panel.columns
    ]

    ####################################
    # Cross-Sectional Ranking Targets
    ####################################

    if missing_targets:
        logger.info(
            "Some selected targets were not individual-stock targets; "
            "attempting cross-sectional ranking targets"
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
            for target in missing_targets
            if target in ranking_input.columns
        ]

        if ranking_available:
            ranking_targets = ranking_input[
                [
                    "Date",
                    "Ticker",
                ]
                + ranking_available
            ].copy()

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
        for target in selected_targets
        if target not in target_panel.columns
    ]

    if missing_targets:
        raise ValueError(
            "\nCould not recreate these selected targets "
            "from targets.py:\n"
            + "\n".join(
                missing_targets
            )
        )

    target_panel = target_panel[
        [
            "Date",
            "Ticker",
        ]
        + selected_targets
    ].copy()

    training_df = (
        feature_df.merge(
            target_panel,
            on=[
                "Date",
                "Ticker",
            ],
            how="left",
        )
    )

    return training_df


training_df = build_training_targets(
    training_features_df
)


########################################
# Fit The Three Previously Selected Model Designs
# On The NEW Yfinance Training Data
########################################

def fit_selected_model(
    result,
    features,
):
    target = result["Target"]
    target_type = result["Target Type"]
    model_name = result["Model"]
    parameters = result["Parameters"]

    model_df = (
        training_df
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .dropna(
            subset=features
            + [target]
        )
        .copy()
    )

    if model_df.empty:
        raise ValueError(
            f"No valid training observations remain for {target}."
        )

    X_train = model_df[
        features
    ].copy()

    y_train = model_df[
        target
    ].copy()

    if (
        target_type != "continuous"
        and y_train.nunique() < 2
    ):
        raise ValueError(
            f"{target} has fewer than two classes in "
            "the selected training data."
        )

    model = build_model(
        model_name=model_name,
        target_type=target_type,
        parameters=parameters,
    )

    scale_models = {
        "ols",
        "linear regression",
        "ridge",
        "lasso",
        "elastic net",
        "elasticnet",
        "logistic regression",
        "logistic",
        "l1 logistic regression",
        "l1 logistic",
        "l2 logistic regression",
        "l2 logistic",
        "elastic net logistic regression",
        "elasticnet logistic regression",
        "elastic net logistic",
    }

    scaler = None

    if (
        model_name
        .lower()
        .strip()
        in scale_models
    ):
        scaler = StandardScaler()

        X_train_model = (
            scaler.fit_transform(
                X_train
            )
        )

    else:
        X_train_model = X_train

    logger.info(
        "Fitting %s | model=%s | rows=%d | features=%d",
        target,
        model_name,
        len(model_df),
        len(features),
    )

    logger.info(
        "%s parameters: %s",
        target,
        parse_parameters(
            parameters
        ),
    )

    model.fit(
        X_train_model,
        y_train,
    )

    return {
        "model": model,
        "scaler": scaler,
        "features": features,
        "target": target,
        "target_type": target_type,
        "model_name": model_name,
        "parameters": parse_parameters(
            parameters
        ),
    }


alpha_model = fit_selected_model(
    alpha_result,
    alpha_features,
)

volatility_model = fit_selected_model(
    volatility_result,
    volatility_features,
)

downside_model = fit_selected_model(
    downside_result,
    downside_features,
)


########################################
# Prediction Function
########################################

def predict_current(
    model_info,
    data,
):
    X = data[
        model_info["features"]
    ].copy()

    X = X.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    if X.isna().any().any():
        raise ValueError(
            f"Missing values found for {model_info['target']}."
        )

    if model_info["scaler"] is not None:
        X = (
            model_info["scaler"]
            .transform(
                X
            )
        )

    if model_info["target_type"] == "binary":
        probabilities = (
            model_info["model"]
            .predict_proba(
                X
            )
        )

        classes = list(
            model_info["model"]
            .classes_
        )

        if 1 not in classes:
            raise ValueError(
                f"Binary model for {model_info['target']} "
                "has no class 1."
            )

        return probabilities[
            :,
            classes.index(1)
        ]

    return model_info[
        "model"
    ].predict(
        X
    )


########################################
# Backtest Dates
########################################

backtest_df = backtest_df[
    (backtest_df["Date"] >= backtest_start)
    & (backtest_df["Date"] <= backtest_end)
].copy()

if backtest_df.empty:
    raise ValueError(
        "No backtest observations exist inside "
        "the requested date range."
    )

backtest_dates = (
    backtest_df["Date"]
    .drop_duplicates()
    .sort_values()
    .reset_index(drop=True)
)

rebalance_dates = (
    backtest_dates.iloc[
        ::REBALANCE_EVERY
    ]
)

logger.info(
    "Backtesting %d stocks from %s to %s | "
    "rebalance every %d trading days",
    len(tokens),
    backtest_dates.iloc[0].date(),
    backtest_dates.iloc[-1].date(),
    REBALANCE_EVERY,
)


########################################
# Portfolio Optimiser
########################################

def get_max_weight(
    universe_size,
    conviction_multiplier=3.0,
    min_weight=0.05,
    absolute_max_weight=0.30,
):
    if universe_size <= 0:
        raise ValueError(
            "Universe size must be positive."
        )

    equal_weight = (
        1.0
        / universe_size
    )

    max_weight = (
        conviction_multiplier
        * equal_weight
    )

    max_weight = min(
        max_weight,
        absolute_max_weight,
    )

    max_weight = max(
        max_weight,
        min_weight,
    )

    # Must always be mathematically possible to invest 100%.
    max_weight = max(
        max_weight,
        equal_weight,
    )

    return max_weight


def construct_portfolio(
    predictions_df,
    alpha_importance=ALPHA_IMPORTANCE,
    volatility_importance=VOLATILITY_IMPORTANCE,
    downside_importance=DOWNSIDE_IMPORTANCE,
    concentration_penalty=CONCENTRATION_PENALTY,
):
    portfolio_df = predictions_df.copy()

    def normalize(series):
        minimum = series.min()
        maximum = series.max()

        if maximum == minimum:
            return pd.Series(
                0.5,
                index=series.index,
            )

        return (
            (series - minimum)
            / (maximum - minimum)
        )

    # Higher alpha is better.
    portfolio_df[
        "Alpha Score"
    ] = normalize(
        portfolio_df["Alpha"]
    )

    # Lower volatility is better.
    portfolio_df[
        "Volatility Score"
    ] = (
        1
        - normalize(
            portfolio_df[
                "Volatility"
            ]
        )
    )

    # A less-negative downside forecast is better.
    portfolio_df[
        "Downside Score"
    ] = normalize(
        portfolio_df["Downside"]
    )

    alpha = (
        portfolio_df[
            "Alpha Score"
        ]
        .to_numpy()
    )

    volatility = (
        portfolio_df[
            "Volatility Score"
        ]
        .to_numpy()
    )

    downside = (
        portfolio_df[
            "Downside Score"
        ]
        .to_numpy()
    )

    n_stocks = len(
        portfolio_df
    )

    if n_stocks < 2:
        raise ValueError(
            "At least two valid stocks are required."
        )

    max_weight = get_max_weight(
        n_stocks
    )

    def objective(weights):
        total_score = (
            alpha_importance
            * np.dot(
                weights,
                alpha,
            )
            + volatility_importance
            * np.dot(
                weights,
                volatility,
            )
            + downside_importance
            * np.dot(
                weights,
                downside,
            )
            - concentration_penalty
            * np.sum(
                weights ** 2
            )
        )

        return -total_score

    constraints = {
        "type": "eq",
        "fun": lambda weights:
            np.sum(weights)
            - 1,
    }

    bounds = [
        (0, max_weight)
        for _ in range(
            n_stocks
        )
    ]

    starting_weights = (
        np.ones(
            n_stocks
        )
        / n_stocks
    )

    result = minimize(
        objective,
        starting_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    if not result.success:
        raise ValueError(
            "Portfolio optimisation failed: "
            f"{result.message}"
        )

    portfolio_df[
        "Weight"
    ] = result.x

    portfolio_df[
        "Weight %"
    ] = (
        portfolio_df[
            "Weight"
        ]
        * 100
    )

    return (
        portfolio_df
        .sort_values(
            "Weight",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )


########################################
# Generate Strict OOS Portfolio Weights
########################################

historical_weights = []

for date in rebalance_dates:
    date_df = (
        backtest_df[
            backtest_df["Date"]
            == date
        ]
        .copy()
    )

    date_df = (
        date_df
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .dropna(
            subset=required_features
        )
    )

    if len(date_df) < 2:
        logger.warning(
            "Skipping rebalance %s: fewer than two "
            "stocks have all selected features",
            date.date(),
        )
        continue

    alpha = predict_current(
        alpha_model,
        date_df,
    )

    volatility = predict_current(
        volatility_model,
        date_df,
    )

    downside = predict_current(
        downside_model,
        date_df,
    )

    date_predictions = pd.DataFrame(
        {
            "Ticker":
                date_df[
                    "Ticker"
                ].to_numpy(),

            "Date":
                date,

            "Alpha":
                alpha,

            "Volatility":
                volatility,

            "Downside":
                downside,
        }
    )

    portfolio = construct_portfolio(
        date_predictions
    )

    for _, row in portfolio.iterrows():
        historical_weights.append(
            {
                "Date":
                    date,

                "Ticker":
                    row["Ticker"],

                "Weight":
                    row["Weight"],
            }
        )


if not historical_weights:
    raise ValueError(
        "No valid rebalance portfolios were created."
    )


########################################
# Daily Held Weights
########################################

rebalance_weights = (
    pd.DataFrame(
        historical_weights
    )
    .pivot(
        index="Date",
        columns="Ticker",
        values="Weight",
    )
    .reindex(
        columns=tokens
    )
    .fillna(0)
)

weights_df = (
    rebalance_weights
    .reindex(
        backtest_dates
    )
    .ffill()
    .fillna(0)
)


########################################
# Daily Backtest Returns
########################################

returns_df = (
    backtest_df
    .pivot(
        index="Date",
        columns="Ticker",
        values="Return",
    )
    .reindex(
        index=backtest_dates,
        columns=tokens,
    )
)

# Signal at date t uses information through that day's close. It therefore
# begins earning returns on the next trading day, not on date t itself.
held_weights = (
    weights_df
    .shift(1)
    .fillna(0)
)

strategy_contributions = (
    held_weights
    * returns_df.fillna(0)
)

strategy_return = (
    strategy_contributions
    .sum(axis=1)
)

active = (
    held_weights
    .sum(axis=1)
    > 0
)

if not active.any():
    raise ValueError(
        "No portfolio ever became active in the backtest."
    )

strategy_start = (
    active[
        active
    ]
    .index[0]
)

backtest = pd.DataFrame(
    index=backtest_dates
)

backtest[
    "Strategy_Return"
] = strategy_return

backtest = backtest.loc[
    strategy_start:
].copy()

backtest[
    "Strategy"
] = (
    1
    + backtest[
        "Strategy_Return"
    ]
).cumprod()


########################################
# S&P 500 Buy-And-Hold
########################################

sp500_close = (
    market_df[
        "Close"
    ]
    .copy()
)

sp500_close.index = (
    pd.to_datetime(
        sp500_close.index
    )
)

sp500_close = (
    sp500_close
    .reindex(
        backtest.index
    )
    .ffill()
)

if (
    sp500_close.empty
    or sp500_close.isna().iloc[0]
):
    raise ValueError(
        "No S&P 500 price is available on "
        "the strategy start date."
    )

backtest[
    "S&P 500"
] = (
    sp500_close
    / sp500_close.iloc[0]
)

backtest[
    "Strategy_vs_S&P500"
] = (
    backtest[
        "Strategy"
    ]
    / backtest[
        "S&P 500"
    ]
    - 1
)


########################################
# Backtest Results
########################################

strategy_total_return = (
    backtest[
        "Strategy"
    ].iloc[-1]
    - 1
)

sp500_total_return = (
    backtest[
        "S&P 500"
    ].iloc[-1]
    - 1
)

relative_return = (
    backtest[
        "Strategy_vs_S&P500"
    ].iloc[-1]
)


print("\n")
print("=" * 70)
print("BACKTEST SETUP")
print("=" * 70)

print(
    "Training stocks:          "
    + ", ".join(
        train_tokens
    )
)

print(
    f"Training period:          "
    f"{train_start.date()} to "
    f"{train_end.date()}"
)

print(
    "Backtest stocks:          "
    + ", ".join(
        tokens
    )
)

print(
    f"Requested backtest:       "
    f"{backtest_start.date()} to "
    f"{backtest_end.date()}"
)

print(
    f"Actual strategy start:    "
    f"{strategy_start.date()}"
)

print(
    f"Training/backtest gap:    "
    f"{TRAINING_EMBARGO_YEARS} year(s) minimum"
)

print(
    f"Rebalance frequency:      "
    f"{REBALANCE_EVERY} trading days"
)

print(
    f"Selected-feature union:   "
    f"{len(required_features)} features"
)

print(
    f"Alpha:                    "
    f"{alpha_model['target']} "
    f"({alpha_model['model_name']})"
)

print(
    f"Volatility:               "
    f"{volatility_model['target']} "
    f"({volatility_model['model_name']})"
)

print(
    f"Downside:                 "
    f"{downside_model['target']} "
    f"({downside_model['model_name']})"
)


print("\n")
print("=" * 70)
print("BACKTEST RESULTS")
print("=" * 70)

print(
    f"Strategy return:          "
    f"{strategy_total_return:.2%}"
)

print(
    f"S&P 500 buy & hold:       "
    f"{sp500_total_return:.2%}"
)

print(
    f"Relative performance:     "
    f"{relative_return:.2%}"
)

print(
    "Transaction costs:        NOT included"
)


########################################
# Final Rebalance Weights
########################################

last_rebalance_date = (
    rebalance_weights
    .index
    .max()
)

last_weights = (
    rebalance_weights
    .loc[
        last_rebalance_date
    ]
    .sort_values(
        ascending=False
    )
)

last_weights = (
    last_weights[
        last_weights > 0
    ]
)

print("\n")
print("=" * 70)
print(
    f"LAST REBALANCE WEIGHTS "
    f"({last_rebalance_date.date()})"
)
print("=" * 70)

for ticker, weight in last_weights.items():
    print(
        f"{ticker:8s} "
        f"{weight:8.2%}"
    )


########################################
# Plot Growth Of £1
########################################

backtest[
    [
        "S&P 500",
        "Strategy",
    ]
].plot(
    figsize=(12, 6),
    title="Out-of-Sample Strategy vs S&P 500",
)

plt.xlabel(
    "Date"
)

plt.ylabel(
    "Growth of £1"
)

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.show()


########################################
# Risk / Performance Comparison
########################################

backtest["S&P500_Return"] = (
    backtest["S&P 500"]
    .pct_change()
)

comparison_returns = (
    backtest[
        [
            "S&P500_Return",
            "Strategy_Return",
        ]
    ]
    .dropna()
)

buy_and_hold_volatility = (
    comparison_returns["S&P500_Return"].std()
    * np.sqrt(252)
)

strategy_volatility = (
    comparison_returns["Strategy_Return"].std()
    * np.sqrt(252)
)

buy_and_hold_mean = (
    comparison_returns["S&P500_Return"].mean()
    * 252
)

strategy_mean = (
    comparison_returns["Strategy_Return"].mean()
    * 252
)

buy_and_hold_sharpe = (
    buy_and_hold_mean / buy_and_hold_volatility
    if buy_and_hold_volatility > 0
    else np.nan
)

strategy_sharpe = (
    strategy_mean / strategy_volatility
    if strategy_volatility > 0
    else np.nan
)

backtest["S&P500_Peak"] = (
    backtest["S&P 500"]
    .cummax()
)

backtest["Strategy_Peak"] = (
    backtest["Strategy"]
    .cummax()
)

backtest["S&P500_Drawdown"] = (
    backtest["S&P 500"]
    / backtest["S&P500_Peak"]
    - 1
)

backtest["Strategy_Drawdown"] = (
    backtest["Strategy"]
    / backtest["Strategy_Peak"]
    - 1
)

buy_and_hold_max_drawdown = (
    backtest["S&P500_Drawdown"]
    .min()
)

strategy_max_drawdown = (
    backtest["Strategy_Drawdown"]
    .min()
)

comparison = pd.DataFrame(
    {
        "S&P 500 Buy and Hold": [
            sp500_total_return,
            buy_and_hold_mean,
            buy_and_hold_volatility,
            buy_and_hold_sharpe,
            buy_and_hold_max_drawdown,
        ],
        "Strategy": [
            strategy_total_return,
            strategy_mean,
            strategy_volatility,
            strategy_sharpe,
            strategy_max_drawdown,
        ],
    },
    index=[
        "Total Return",
        "Annualised Mean Return",
        "Annualised Volatility",
        "Sharpe Ratio",
        "Maximum Drawdown",
    ],
)

print("\n")
print("=" * 70)
print("STRATEGY COMPARISON")
print("=" * 70)

comparison_display = comparison.copy()

for row in [
    "Total Return",
    "Annualised Mean Return",
    "Annualised Volatility",
    "Maximum Drawdown",
]:
    comparison_display.loc[row] = (
        comparison.loc[row]
        .map(
            lambda value:
                f"{value:.2%}"
                if pd.notna(value)
                else "NaN"
        )
    )

comparison_display.loc["Sharpe Ratio"] = (
    comparison.loc["Sharpe Ratio"]
    .map(
        lambda value:
            f"{value:.3f}"
            if pd.notna(value)
            else "NaN"
    )
)

print(comparison_display.to_string())

backtest[
    [
        "S&P500_Drawdown",
        "Strategy_Drawdown",
    ]
].rename(
    columns={
        "S&P500_Drawdown": "S&P 500 Drawdown",
        "Strategy_Drawdown": "Strategy Drawdown",
    }
).plot(
    figsize=(12, 6),
    title="Strategy vs S&P 500 Drawdown",
)

plt.xlabel("Date")
plt.ylabel("Drawdown")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

held_positions = (
    rebalance_weights
    > 0
).astype(int)

position_changes = (
    held_positions
    .diff()
    .abs()
    .fillna(held_positions.iloc[0])
    .sum()
    .sum()
)

print("\n")
print("=" * 70)
print("TRADING ACTIVITY")
print("=" * 70)
print(f"Number of position changes: {int(position_changes)}")
print("Trading fees:               NOT modelled")

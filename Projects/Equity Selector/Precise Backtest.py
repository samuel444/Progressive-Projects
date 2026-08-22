import ast
import json
import logging
import re
import sqlite3
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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

BACKTEST_DATA_DB = (
    DATA_DIR
    / "Backtest_Features_Targets.db"
)

########################################
# Single Backtest Settings
#
# This file runs ONE chosen configuration.
# It then performs institutional-style
# anomaly / risk diagnostics on that result.
########################################

SELECTED_UNIVERSE = "High Liquidity"

# Options:
#
# "Manual"
# "All Cached"
# "Low Liquidity"
# "Medium Liquidity"
# "High Liquidity"

ALPHA_IMPORTANCE = 0.60
VOLATILITY_IMPORTANCE = 0.20
DOWNSIDE_IMPORTANCE = 0.20

MAX_WEIGHT = 0.25
CONCENTRATION_PENALTY = 0.10
REBALANCE_EVERY = 60

# Trading 212 commission assumption.
TRADING_FEE = 0.00


########################################
# Manual Universe
########################################

MANUAL_TRAIN_TICKERS = [
    "AAPL",
    "MSFT",
]

MANUAL_TEST_TICKERS = [
    "NVDA",
    "GOOG",
    "META",
    "TSLA",
    "AVGO",
    "AMD",
    "NFLX",
    "ADBE",
    "TXN",
    "AMAT",
    "MA",
    "MS",
    "C",
    "SPGI",
    "LLY",
    "AMGN",
    "GILD",
    "MDT",
    "NKE",
    "SBUX",
    "TGT",
    "BKNG",
    "SLB",
    "BA",
    "DE",
    "UPS",
    "IBM",
    "INTU",
    "CRM",
    "PG",
]


########################################
# Liquidity Universe Settings
########################################

# For Low / Medium / High liquidity tests,
# the first N test trading days are used to
# classify stocks by median dollar volume.
# The actual strategy starts AFTER this
# classification window.
LIQUIDITY_CLASSIFICATION_DAYS = 60


########################################
# Anomaly Detection Settings
########################################

# Daily return shock relative to the PREVIOUS
# rolling distribution.
RETURN_Z_WINDOW = 60
RETURN_Z_THRESHOLD = 3.0
ABS_DAILY_RETURN_THRESHOLD = 0.05

# Strategy minus S&P 500 daily return.
ACTIVE_RETURN_Z_THRESHOLD = 3.0
ABS_ACTIVE_RETURN_THRESHOLD = 0.04

# Volatility regime detection.
SHORT_VOL_WINDOW = 20
VOL_BASELINE_WINDOW = 252
VOL_SPIKE_MULTIPLIER = 1.75

# Rolling market exposure.
BETA_WINDOW = 60
BETA_ABS_THRESHOLD = 1.50

CORRELATION_WINDOW = 60
CORRELATION_ABS_THRESHOLD = 0.90

# Rolling strategy failure.
ROLLING_SHARPE_WINDOW = 60
ROLLING_SHARPE_COLLAPSE = -1.00

# Drawdown episode reporting.
DRAWDOWN_REPORT_THRESHOLD = -0.10

# Turnover is sum(abs(change in weights)).
# A full switch from one 100% portfolio to
# another is turnover = 2.0 under this definition.
TURNOVER_ABS_THRESHOLD = 1.00
TURNOVER_Z_THRESHOLD = 3.0

# Portfolio concentration.
MIN_EFFECTIVE_HOLDINGS = 3.50
TOP3_WEIGHT_THRESHOLD = 0.80
WEIGHT_TOLERANCE = 1e-6

# Contribution concentration on material P&L days.
CONTRIBUTION_CONCENTRATION_THRESHOLD = 0.75
CONTRIBUTION_MIN_ABS_STRATEGY_RETURN = 0.01

# Consecutive losing days.
LOSS_STREAK_THRESHOLD = 5

# Capacity / liquidity checks.
PORTFOLIO_CAPITAL = 10_000_000
ADV_LOOKBACK = 20
POSITION_ADV_WARNING = 0.01
TRADE_ADV_WARNING = 0.05

# Keep original backtest behaviour if data is missing
# while also flagging it as a HIGH severity anomaly.
# Set True if you prefer the script to fail instead.
STRICT_MISSING_HELD_RETURNS = False

# Number of extreme days / episodes to print.
TOP_EXTREME_DAYS = 10
TOP_DRAWDOWN_EPISODES = 10
TOP_ANOMALIES_TO_PRINT = 50


########################################
# Output Settings
########################################

PLOT_RESULTS = True
PLOT_ANOMALIES = True

ANOMALY_RESULTS_DB = (
    DATA_DIR
    / "Single_Backtest_Anomaly_Report.db"
)

ANOMALY_RESULTS_CSV = (
    DATA_DIR
    / "Single_Backtest_Anomalies.csv"
)

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


def build_model(
    model_name,
    target_type,
    parameters,
):

    params = parse_parameters(parameters)

    name = (
        model_name
        .lower()
        .strip()
    )

    is_regression = (
        target_type.lower()
        == "continuous"
    )

    if name in {
        "linear regression",
        "ols",
    }:
        return LinearRegression(**params)

    if name == "ridge":
        return Ridge(**params)

    if name == "lasso":
        return Lasso(**params)

    if name in {
        "elastic net",
        "elasticnet",
    }:
        return ElasticNet(**params)

    if name in {
        "logistic regression",
        "logistic",
    }:
        params.setdefault("max_iter", 5000)
        return LogisticRegression(**params)

    if name in {
        "l2 logistic regression",
        "l2 logistic",
    }:
        params.setdefault("penalty", "l2")
        params.setdefault("solver", "lbfgs")
        params.setdefault("max_iter", 5000)
        return LogisticRegression(**params)

    if name in {
        "l1 logistic regression",
        "l1 logistic",
    }:
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

    if name in {
        "lightgbm",
        "lgbm",
    }:
        if is_regression:
            return LGBMRegressor(**params)
        return LGBMClassifier(**params)

    raise ValueError(
        f"Unknown model '{model_name}'."
    )


########################################
# Load Cached Database
########################################

if not BACKTEST_DATA_DB.exists():

    raise FileNotFoundError(
        f"\nCached database does not exist:\n"
        f"{BACKTEST_DATA_DB}\n\n"
        "Run 01_build_backtest_database.py first."
    )


logger.info(
    "Loading cached feature / target data"
)


with sqlite3.connect(
    BACKTEST_DATA_DB
) as connection:

    model_data = pd.read_sql_query(
        "SELECT * FROM Model_Data",
        connection,
    )

    benchmark_df = pd.read_sql_query(
        "SELECT * FROM Benchmark",
        connection,
    )

    selected_models_df = pd.read_sql_query(
        "SELECT * FROM Selected_Models",
        connection,
    )

    required_features_df = pd.read_sql_query(
        """
        SELECT *
        FROM Required_Features
        ORDER BY "Feature Order"
        """,
        connection,
    )

    config_df = pd.read_sql_query(
        "SELECT * FROM Config",
        connection,
    )


########################################
# Parse Cached Data
########################################

config = dict(
    zip(
        config_df["Key"],
        config_df["Value"],
    )
)


model_data["Date"] = pd.to_datetime(
    model_data["Date"]
)

benchmark_df["Date"] = pd.to_datetime(
    benchmark_df["Date"]
)


required_features = (
    required_features_df[
        "Feature"
    ]
    .tolist()
)


all_training_data = (
    model_data[
        model_data[
            "Split"
        ]
        == "TRAIN"
    ]
    .copy()
)


all_backtest_data = (
    model_data[
        model_data[
            "Split"
        ]
        == "BACKTEST"
    ]
    .copy()
)


if all_training_data.empty:
    raise ValueError(
        "No TRAIN rows are stored in the cached database."
    )


if all_backtest_data.empty:
    raise ValueError(
        "No BACKTEST rows are stored in the cached database."
    )


available_train_tickers = sorted(
    all_training_data[
        "Ticker"
    ]
    .dropna()
    .unique()
    .tolist()
)


available_test_tickers = sorted(
    all_backtest_data[
        "Ticker"
    ]
    .dropna()
    .unique()
    .tolist()
)


########################################
# Stored Model Specifications
########################################

def get_model_specification(
    prediction_type,
):

    rows = (
        selected_models_df[
            selected_models_df[
                "Prediction Type"
            ]
            .str.lower()
            == prediction_type.lower()
        ]
        .copy()
    )

    if len(rows) != 1:

        raise ValueError(
            f"Expected exactly one "
            f"{prediction_type} model "
            "in Selected_Models."
        )

    row = rows.iloc[0]

    return {
        "prediction_type":
            prediction_type,

        "target":
            row["Target"],

        "target_type":
            row["Target Type"],

        "model_name":
            row["Model"],

        "parameters":
            row["Parameters"],

        "features":
            json.loads(
                row["Features"]
            ),
    }


alpha_spec = get_model_specification(
    "Alpha"
)

volatility_spec = get_model_specification(
    "Volatility"
)

downside_spec = get_model_specification(
    "Downside"
)


########################################
# Liquidity Buckets
########################################

def liquidity_table(
    dataframe,
    first_n_dates=None,
):

    data = (
        dataframe[
            [
                "Date",
                "Ticker",
                "Close",
                "Volume",
            ]
        ]
        .dropna()
        .copy()
    )

    if first_n_dates is not None:

        dates = (
            data[
                "Date"
            ]
            .drop_duplicates()
            .sort_values()
            .iloc[
                :first_n_dates
            ]
        )

        data = data[
            data[
                "Date"
            ]
            .isin(
                dates
            )
        ].copy()

    data[
        "Dollar Volume"
    ] = (
        data[
            "Close"
        ]
        * data[
            "Volume"
        ]
    )

    liquidity = (
        data
        .groupby(
            "Ticker"
        )[
            "Dollar Volume"
        ]
        .median()
        .dropna()
        .sort_values()
    )

    return liquidity


def split_liquidity_buckets(
    liquidity,
):

    if len(liquidity) < 6:

        raise ValueError(
            "At least 6 tickers are needed "
            "to create useful Low / Medium / High "
            "liquidity buckets."
        )

    ordered_tickers = (
        liquidity
        .sort_values()
        .index
        .to_numpy()
    )

    low, medium, high = (
        np.array_split(
            ordered_tickers,
            3,
        )
    )

    return {
        "Low Liquidity":
            low.tolist(),

        "Medium Liquidity":
            medium.tolist(),

        "High Liquidity":
            high.tolist(),
    }


training_liquidity = liquidity_table(
    all_training_data
)


test_liquidity = liquidity_table(
    all_backtest_data,
    first_n_dates=(
        LIQUIDITY_CLASSIFICATION_DAYS
    ),
)


training_liquidity_buckets = (
    split_liquidity_buckets(
        training_liquidity
    )
)


test_liquidity_buckets = (
    split_liquidity_buckets(
        test_liquidity
    )
)


########################################
# Backtest Start
#
# Liquidity buckets use only the first N
# test dates to classify the test universe.
# For liquidity-mode tests, the strategy
# therefore begins AFTER that window.
########################################

all_test_dates = (
    all_backtest_data[
        "Date"
    ]
    .drop_duplicates()
    .sort_values()
    .reset_index(
        drop=True
    )
)


liquidity_mode_names = {
    "Low Liquidity",
    "Medium Liquidity",
    "High Liquidity",
}


if SELECTED_UNIVERSE in liquidity_mode_names:

    if (
        len(all_test_dates)
        <= LIQUIDITY_CLASSIFICATION_DAYS
    ):

        raise ValueError(
            "Backtest does not contain enough dates "
            "for the requested liquidity classification window."
        )

    common_backtest_start = (
        all_test_dates.iloc[
            LIQUIDITY_CLASSIFICATION_DAYS
        ]
    )

else:

    common_backtest_start = (
        all_test_dates.iloc[0]
    )


########################################
# Select ONE Universe Scenario
########################################

def existing_tickers(
    requested,
    available,
):

    available_set = set(
        available
    )

    return [
        ticker
        for ticker in requested
        if ticker in available_set
    ]


if SELECTED_UNIVERSE == "Manual":

    selected_scenario = {
        "Name":
            "Manual",

        "Train Tickers":
            existing_tickers(
                MANUAL_TRAIN_TICKERS,
                available_train_tickers,
            ),

        "Test Tickers":
            existing_tickers(
                MANUAL_TEST_TICKERS,
                available_test_tickers,
            ),
    }


elif SELECTED_UNIVERSE == "All Cached":

    selected_scenario = {
        "Name":
            "All Cached",

        "Train Tickers":
            available_train_tickers,

        "Test Tickers":
            available_test_tickers,
    }


elif SELECTED_UNIVERSE in {
    "Low Liquidity",
    "Medium Liquidity",
    "High Liquidity",
}:

    selected_scenario = {
        "Name":
            SELECTED_UNIVERSE,

        "Train Tickers":
            training_liquidity_buckets[
                SELECTED_UNIVERSE
            ],

        "Test Tickers":
            test_liquidity_buckets[
                SELECTED_UNIVERSE
            ],
    }


else:

    raise ValueError(
        "SELECTED_UNIVERSE must be one of: "
        "'Manual', 'All Cached', "
        "'Low Liquidity', 'Medium Liquidity', "
        "'High Liquidity'."
    )


########################################
# Liquidity Report
########################################

def format_dollar_volume(value):

    if value >= 1_000_000_000:
        return f"${value / 1_000_000_000:.2f}bn"

    if value >= 1_000_000:
        return f"${value / 1_000_000:.2f}m"

    return f"${value:,.0f}"


print("\n")
print("=" * 90)
print("LIQUIDITY UNIVERSES")
print("=" * 90)

print(
    f"Test liquidity classification window: "
    f"{LIQUIDITY_CLASSIFICATION_DAYS} trading days"
)

print(
    f"Common strategy start:                 "
    f"{common_backtest_start.date()}"
)

print(
    "\nLiquidity buckets are RELATIVE to the "
    "stocks contained in the cached database."
)


for bucket_name in [
    "Low Liquidity",
    "Medium Liquidity",
    "High Liquidity",
]:

    train_names = (
        training_liquidity_buckets[
            bucket_name
        ]
    )

    test_names = (
        test_liquidity_buckets[
            bucket_name
        ]
    )

    train_values = (
        training_liquidity
        .reindex(
            train_names
        )
        .dropna()
    )

    test_values = (
        test_liquidity
        .reindex(
            test_names
        )
        .dropna()
    )

    print("\n" + bucket_name.upper())

    print(
        "Train: "
        + ", ".join(
            train_names
        )
    )

    print(
        "Test:  "
        + ", ".join(
            test_names
        )
    )

    if not train_values.empty:

        print(
            "Train median dollar-volume range: "
            f"{format_dollar_volume(train_values.min())} "
            f"to "
            f"{format_dollar_volume(train_values.max())}"
        )

    if not test_values.empty:

        print(
            "Test median dollar-volume range:  "
            f"{format_dollar_volume(test_values.min())} "
            f"to "
            f"{format_dollar_volume(test_values.max())}"
        )


########################################
# Fit Model On One Training Universe
########################################

SCALE_MODELS = {
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


def fit_selected_model(
    training_df,
    specification,
):

    target = specification[
        "target"
    ]

    features = specification[
        "features"
    ]

    target_type = specification[
        "target_type"
    ]

    model_name = specification[
        "model_name"
    ]

    parameters = specification[
        "parameters"
    ]

    model_df = (
        training_df
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
        .dropna(
            subset=(
                features
                + [
                    target
                ]
            )
        )
        .copy()
    )

    if model_df.empty:

        raise ValueError(
            f"No valid training rows remain "
            f"for {target}."
        )

    X_train = (
        model_df[
            features
        ]
        .copy()
    )

    y_train = (
        model_df[
            target
        ]
        .copy()
    )

    if (
        target_type
        != "continuous"
        and y_train.nunique()
        < 2
    ):

        raise ValueError(
            f"{target} has fewer than two "
            "classes in this training universe."
        )

    model = build_model(
        model_name=model_name,
        target_type=target_type,
        parameters=parameters,
    )

    scaler = None

    if (
        model_name
        .lower()
        .strip()
        in SCALE_MODELS
    ):

        scaler = (
            StandardScaler()
        )

        X_train_model = (
            scaler
            .fit_transform(
                X_train
            )
        )

    else:

        X_train_model = (
            X_train
        )

    model.fit(
        X_train_model,
        y_train,
    )

    return {
        "model":
            model,

        "scaler":
            scaler,

        "features":
            features,

        "target":
            target,

        "target_type":
            target_type,

        "model_name":
            model_name,

        "training_rows":
            len(
                model_df
            ),
    }


########################################
# Prediction
########################################

def predict_model(
    model_info,
    dataframe,
):

    X = (
        dataframe[
            model_info[
                "features"
            ]
        ]
        .copy()
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
    )

    if X.isna().any().any():

        raise ValueError(
            f"Missing values found for "
            f"{model_info['target']}."
        )

    if (
        model_info[
            "scaler"
        ]
        is not None
    ):

        X = (
            model_info[
                "scaler"
            ]
            .transform(
                X
            )
        )

    if (
        model_info[
            "target_type"
        ]
        == "binary"
    ):

        probabilities = (
            model_info[
                "model"
            ]
            .predict_proba(
                X
            )
        )

        classes = list(
            model_info[
                "model"
            ]
            .classes_
        )

        if 1 not in classes:

            raise ValueError(
                f"Binary model for "
                f"{model_info['target']} "
                "has no class 1."
            )

        return probabilities[
            :,
            classes.index(1)
        ]

    return (
        model_info[
            "model"
        ]
        .predict(
            X
        )
    )


########################################
# Prepare One Universe Scenario
#
# Models are fitted ONCE per universe.
# Predictions are also generated ONCE.
#
# The parameter loops therefore only rerun
# portfolio construction / backtesting.
########################################

def prepare_universe_scenario(
    scenario,
):

    train_tickers = (
        scenario[
            "Train Tickers"
        ]
    )

    test_tickers = (
        scenario[
            "Test Tickers"
        ]
    )

    if len(train_tickers) < 2:

        raise ValueError(
            f"{scenario['Name']} has fewer "
            "than two training stocks."
        )

    if len(test_tickers) < 2:

        raise ValueError(
            f"{scenario['Name']} has fewer "
            "than two test stocks."
        )

    training_df = (
        all_training_data[
            all_training_data[
                "Ticker"
            ]
            .isin(
                train_tickers
            )
        ]
        .copy()
    )

    backtest_df = (
        all_backtest_data[
            (
                all_backtest_data[
                    "Ticker"
                ]
                .isin(
                    test_tickers
                )
            )
            &
            (
                all_backtest_data[
                    "Date"
                ]
                >= common_backtest_start
            )
        ]
        .copy()
    )

    if training_df.empty:
        raise ValueError(
            "Training dataframe is empty."
        )

    if backtest_df.empty:
        raise ValueError(
            "Backtest dataframe is empty."
        )

    print("\n")
    print("=" * 90)
    print(
        f"PREPARING UNIVERSE: "
        f"{scenario['Name']}"
    )
    print("=" * 90)

    print(
        f"Training tickers ({len(train_tickers)}): "
        + ", ".join(
            train_tickers
        )
    )

    print(
        f"Test tickers ({len(test_tickers)}):     "
        + ", ".join(
            test_tickers
        )
    )

    logger.info(
        "%s | fitting Alpha",
        scenario[
            "Name"
        ],
    )

    alpha_model = fit_selected_model(
        training_df,
        alpha_spec,
    )

    logger.info(
        "%s | fitting Volatility",
        scenario[
            "Name"
        ],
    )

    volatility_model = fit_selected_model(
        training_df,
        volatility_spec,
    )

    logger.info(
        "%s | fitting Downside",
        scenario[
            "Name"
        ],
    )

    downside_model = fit_selected_model(
        training_df,
        downside_spec,
    )

    valid_predictions = (
        backtest_df
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
        .dropna(
            subset=(
                required_features
            )
        )
        .copy()
    )

    if valid_predictions.empty:

        raise ValueError(
            "No valid backtest rows remain "
            "after feature filtering."
        )

    logger.info(
        "%s | precomputing predictions",
        scenario[
            "Name"
        ],
    )

    valid_predictions[
        "Alpha"
    ] = predict_model(
        alpha_model,
        valid_predictions,
    )

    valid_predictions[
        "Volatility"
    ] = predict_model(
        volatility_model,
        valid_predictions,
    )

    valid_predictions[
        "Downside"
    ] = predict_model(
        downside_model,
        valid_predictions,
    )

    backtest_dates = (
        backtest_df[
            "Date"
        ]
        .drop_duplicates()
        .sort_values()
        .reset_index(
            drop=True
        )
    )

    returns_df = (
        backtest_df
        .pivot(
            index="Date",
            columns="Ticker",
            values="Return",
        )
        .reindex(
            index=backtest_dates,
            columns=test_tickers,
        )
    )

    benchmark_series = (
        benchmark_df
        .set_index(
            "Date"
        )[
            "Close"
        ]
        .sort_index()
    )

    return {
        "name":
            scenario[
                "Name"
            ],

        "train_tickers":
            train_tickers,

        "test_tickers":
            test_tickers,

        "training_df":
            training_df,

        "backtest_df":
            backtest_df,

        "predictions":
            valid_predictions,

        "backtest_dates":
            backtest_dates,

        "returns_df":
            returns_df,

        "benchmark_series":
            benchmark_series,

        "alpha_model":
            alpha_model,

        "volatility_model":
            volatility_model,

        "downside_model":
            downside_model,
    }


########################################
# Portfolio Optimiser
########################################

def normalize(
    series,
):

    minimum = (
        series.min()
    )

    maximum = (
        series.max()
    )

    if maximum == minimum:

        return pd.Series(
            0.5,
            index=series.index,
        )

    return (
        (
            series
            - minimum
        )
        /
        (
            maximum
            - minimum
        )
    )


def construct_portfolio(
    predictions_df,
    alpha_importance,
    volatility_importance,
    downside_importance,
    max_weight,
    concentration_penalty,
):

    portfolio_df = (
        predictions_df.copy()
    )

    portfolio_df[
        "Alpha Score"
    ] = normalize(
        portfolio_df[
            "Alpha"
        ]
    )

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

    portfolio_df[
        "Downside Score"
    ] = normalize(
        portfolio_df[
            "Downside"
        ]
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
            "At least two valid stocks "
            "are required."
        )

    if (
        max_weight <= 0
        or max_weight > 1
    ):

        raise ValueError(
            "MAX_WEIGHT must be between "
            "0 and 1."
        )

    if (
        max_weight
        * n_stocks
        < 1
    ):

        raise ValueError(
            f"MAX_WEIGHT={max_weight:.2%} "
            f"is impossible with only "
            f"{n_stocks} valid stocks."
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
        "type":
            "eq",

        "fun":
            lambda weights:
                np.sum(
                    weights
                )
                - 1,
    }

    bounds = [
        (
            0,
            max_weight,
        )
        for _
        in range(
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
# Run One Backtest
########################################

def run_single_backtest(
    prepared,
    importance_preset,
    max_weight,
    concentration_penalty,
    rebalance_every,
):

    alpha_importance = (
        importance_preset[
            "Alpha"
        ]
    )

    volatility_importance = (
        importance_preset[
            "Volatility"
        ]
    )

    downside_importance = (
        importance_preset[
            "Downside"
        ]
    )

    predictions = (
        prepared[
            "predictions"
        ]
    )

    backtest_dates = (
        prepared[
            "backtest_dates"
        ]
    )

    test_tickers = (
        prepared[
            "test_tickers"
        ]
    )

    rebalance_dates = (
        backtest_dates.iloc[
            ::rebalance_every
        ]
    )

    historical_weights = []

    skipped_rebalances = 0

    for date in rebalance_dates:

        date_predictions = (
            predictions[
                predictions[
                    "Date"
                ]
                == date
            ][
                [
                    "Ticker",
                    "Date",
                    "Alpha",
                    "Volatility",
                    "Downside",
                ]
            ]
            .dropna()
            .copy()
        )

        if len(date_predictions) < 2:

            skipped_rebalances += 1
            continue

        if (
            max_weight
            * len(
                date_predictions
            )
            < 1
        ):

            skipped_rebalances += 1
            continue

        portfolio = construct_portfolio(
            date_predictions,
            alpha_importance=(
                alpha_importance
            ),
            volatility_importance=(
                volatility_importance
            ),
            downside_importance=(
                downside_importance
            ),
            max_weight=(
                max_weight
            ),
            concentration_penalty=(
                concentration_penalty
            ),
        )

        for _, row in portfolio.iterrows():

            historical_weights.append(
                {
                    "Date":
                        date,

                    "Ticker":
                        row[
                            "Ticker"
                        ],

                    "Weight":
                        row[
                            "Weight"
                        ],
                }
            )

    if not historical_weights:

        raise ValueError(
            "No valid rebalance portfolios "
            "were created."
        )

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
            columns=test_tickers
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

    held_weights = (
        weights_df
        .shift(1)
        .fillna(0)
    )

    returns_df = (
        prepared[
            "returns_df"
        ]
    )

    strategy_contributions = (
        held_weights
        * returns_df
        .fillna(0)
    )

    turnover = (
        held_weights
        .diff()
        .abs()
        .sum(
            axis=1
        )
        .fillna(0)
    )

    trading_cost = (
        turnover
        * TRADING_FEE
    )

    strategy_return = (
        strategy_contributions
        .sum(
            axis=1
        )
        - trading_cost
    )

    active = (
        held_weights
        .sum(
            axis=1
        )
        > 0
    )

    if not active.any():

        raise ValueError(
            "No portfolio became active."
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

    backtest = (
        backtest.loc[
            strategy_start:
        ]
        .copy()
    )

    backtest[
        "Strategy"
    ] = (
        1
        + backtest[
            "Strategy_Return"
        ]
    ).cumprod()

    sp500_close = (
        prepared[
            "benchmark_series"
        ]
        .reindex(
            backtest.index
        )
        .ffill()
    )

    if (
        sp500_close.empty
        or pd.isna(
            sp500_close.iloc[0]
        )
    ):

        raise ValueError(
            "No S&P 500 price is available "
            "on the strategy start date."
        )

    backtest[
        "S&P 500"
    ] = (
        sp500_close
        / sp500_close.iloc[0]
    )

    backtest[
        "S&P500_Return"
    ] = (
        backtest[
            "S&P 500"
        ]
        .pct_change()
    )

    strategy_total_return = (
        backtest[
            "Strategy"
        ]
        .iloc[-1]
        - 1
    )

    sp500_total_return = (
        backtest[
            "S&P 500"
        ]
        .iloc[-1]
        - 1
    )

    relative_return = (
        backtest[
            "Strategy"
        ]
        .iloc[-1]
        /
        backtest[
            "S&P 500"
        ]
        .iloc[-1]
        - 1
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

    strategy_volatility = (
        comparison_returns[
            "Strategy_Return"
        ]
        .std()
        * np.sqrt(252)
    )

    strategy_mean = (
        comparison_returns[
            "Strategy_Return"
        ]
        .mean()
        * 252
    )

    strategy_sharpe = (
        strategy_mean
        / strategy_volatility
        if strategy_volatility > 0
        else np.nan
    )

    sp500_volatility = (
        comparison_returns[
            "S&P500_Return"
        ]
        .std()
        * np.sqrt(252)
    )

    sp500_mean = (
        comparison_returns[
            "S&P500_Return"
        ]
        .mean()
        * 252
    )

    sp500_sharpe = (
        sp500_mean
        / sp500_volatility
        if sp500_volatility > 0
        else np.nan
    )

    backtest[
        "Strategy_Peak"
    ] = (
        backtest[
            "Strategy"
        ]
        .cummax()
    )

    backtest[
        "S&P500_Peak"
    ] = (
        backtest[
            "S&P 500"
        ]
        .cummax()
    )

    backtest[
        "Strategy_Drawdown"
    ] = (
        backtest[
            "Strategy"
        ]
        /
        backtest[
            "Strategy_Peak"
        ]
        - 1
    )

    backtest[
        "S&P500_Drawdown"
    ] = (
        backtest[
            "S&P 500"
        ]
        /
        backtest[
            "S&P500_Peak"
        ]
        - 1
    )

    strategy_max_drawdown = (
        backtest[
            "Strategy_Drawdown"
        ]
        .min()
    )

    sp500_max_drawdown = (
        backtest[
            "S&P500_Drawdown"
        ]
        .min()
    )

    held_positions = (
        rebalance_weights
        > 0.000001
    ).astype(int)

    position_changes = (
        held_positions
        .diff()
        .abs()
        .fillna(
            held_positions.iloc[0]
        )
        .sum()
        .sum()
    )

    average_holdings = (
        (
            held_weights
            > 0.000001
        )
        .sum(
            axis=1
        )[
            active
        ]
        .mean()
    )

    average_turnover = (
        turnover[
            active
        ]
        .mean()
    )

    last_rebalance_date = (
        rebalance_weights
        .index
        .max()
    )

    last_weights = (
        rebalance_weights.loc[
            last_rebalance_date
        ]
        .sort_values(
            ascending=False
        )
    )

    last_weights = (
        last_weights[
            last_weights
            > 0.000001
        ]
    )

    result_row = {
        "Universe":
            prepared[
                "name"
            ],

        "Train Stocks":
            len(
                prepared[
                    "train_tickers"
                ]
            ),

        "Test Stocks":
            len(
                prepared[
                    "test_tickers"
                ]
            ),

        "Importance Preset":
            importance_preset[
                "Name"
            ],

        "Alpha Importance":
            alpha_importance,

        "Volatility Importance":
            volatility_importance,

        "Downside Importance":
            downside_importance,

        "Max Weight":
            max_weight,

        "Concentration Penalty":
            concentration_penalty,

        "Rebalance Days":
            rebalance_every,

        "Strategy Return":
            strategy_total_return,

        "S&P 500 Return":
            sp500_total_return,

        "Relative Performance":
            relative_return,

        "Annualised Mean":
            strategy_mean,

        "Annualised Volatility":
            strategy_volatility,

        "Sharpe":
            strategy_sharpe,

        "S&P Sharpe":
            sp500_sharpe,

        "Max Drawdown":
            strategy_max_drawdown,

        "S&P Max Drawdown":
            sp500_max_drawdown,

        "Position Changes":
            int(
                position_changes
            ),

        "Average Holdings":
            average_holdings,

        "Average Daily Turnover":
            average_turnover,

        "Skipped Rebalances":
            skipped_rebalances,

        "Strategy Start":
            strategy_start,

        "Strategy End":
            backtest.index[-1],

        "Last Rebalance":
            last_rebalance_date,

        "Last Holdings":
            ", ".join(
                f"{ticker}:{weight:.1%}"
                for ticker, weight
                in last_weights.items()
            ),
    }

    return (
        result_row,
        backtest,
        rebalance_weights,
    )


########################################
# Per-Experiment Print
########################################

def print_single_result(
    experiment_number,
    total_experiments,
    row,
):

    print("\n")
    print("=" * 90)

    print(
        f"EXPERIMENT "
        f"{experiment_number:,} / "
        f"{total_experiments:,}"
    )

    print("=" * 90)

    print(
        f"Universe:                 "
        f"{row['Universe']}"
    )

    print(
        f"Train / Test stocks:      "
        f"{row['Train Stocks']} / "
        f"{row['Test Stocks']}"
    )

    print(
        f"Importance preset:        "
        f"{row['Importance Preset']}"
    )

    print(
        f"Alpha / Vol / Downside:   "
        f"{row['Alpha Importance']:.2f} / "
        f"{row['Volatility Importance']:.2f} / "
        f"{row['Downside Importance']:.2f}"
    )

    print(
        f"Max weight:               "
        f"{row['Max Weight']:.0%}"
    )

    print(
        f"Concentration penalty:    "
        f"{row['Concentration Penalty']:.2f}"
    )

    print(
        f"Rebalance frequency:      "
        f"{row['Rebalance Days']} trading days"
    )

    print("\n")
    print("-" * 90)
    print("BACKTEST RESULTS")
    print("-" * 90)

    print(
        f"Strategy return:          "
        f"{row['Strategy Return']:.2%}"
    )

    print(
        f"S&P 500 buy & hold:       "
        f"{row['S&P 500 Return']:.2%}"
    )

    print(
        f"Relative performance:     "
        f"{row['Relative Performance']:.2%}"
    )

    print(
        f"Annualised mean:          "
        f"{row['Annualised Mean']:.2%}"
    )

    print(
        f"Annualised volatility:    "
        f"{row['Annualised Volatility']:.2%}"
    )

    print(
        f"Sharpe ratio:             "
        f"{row['Sharpe']:.3f}"
    )

    print(
        f"Maximum drawdown:         "
        f"{row['Max Drawdown']:.2%}"
    )

    print(
        f"Average holdings:         "
        f"{row['Average Holdings']:.2f}"
    )

    print(
        f"Position changes:         "
        f"{row['Position Changes']}"
    )

    print(
        f"Skipped rebalances:       "
        f"{row['Skipped Rebalances']}"
    )


########################################
# Run ONE Fixed Backtest
########################################

fixed_importance_preset = {
    "Name":
        "Fixed Settings",

    "Alpha":
        ALPHA_IMPORTANCE,

    "Volatility":
        VOLATILITY_IMPORTANCE,

    "Downside":
        DOWNSIDE_IMPORTANCE,
}


prepared = (
    prepare_universe_scenario(
        selected_scenario
    )
)


(
    result_row,
    backtest,
    rebalance_weights,
) = run_single_backtest(
    prepared=prepared,
    importance_preset=fixed_importance_preset,
    max_weight=MAX_WEIGHT,
    concentration_penalty=(
        CONCENTRATION_PENALTY
    ),
    rebalance_every=(
        REBALANCE_EVERY
    ),
)


print_single_result(
    1,
    1,
    result_row,
)


########################################
# Reconstruct Daily Portfolio State
########################################

backtest_dates = (
    prepared[
        "backtest_dates"
    ]
)


test_tickers = (
    prepared[
        "test_tickers"
    ]
)


weights_df = (
    rebalance_weights
    .reindex(
        backtest_dates
    )
    .ffill()
    .fillna(0)
)


# The signal is generated at close t.
# These are the weights ACTUALLY held
# for the return earned on day t.
held_weights = (
    weights_df
    .shift(1)
    .fillna(0)
)


returns_df = (
    prepared[
        "returns_df"
    ]
    .reindex(
        index=backtest_dates,
        columns=test_tickers,
    )
)


strategy_contributions = (
    held_weights
    * returns_df
    .fillna(0)
)


turnover = (
    held_weights
    .diff()
    .abs()
    .sum(
        axis=1
    )
    .fillna(0)
)


# Restrict all diagnostics to the actual
# strategy period.
held_weights = (
    held_weights
    .reindex(
        backtest.index
    )
)


returns_df = (
    returns_df
    .reindex(
        backtest.index
    )
)


strategy_contributions = (
    strategy_contributions
    .reindex(
        backtest.index
    )
)


turnover = (
    turnover
    .reindex(
        backtest.index
    )
    .fillna(0)
)


########################################
# Data Integrity Checks
########################################

missing_held_returns = (
    (
        held_weights
        > WEIGHT_TOLERANCE
    )
    &
    returns_df.isna()
)


missing_held_return_count = int(
    missing_held_returns
    .sum()
    .sum()
)


if missing_held_return_count > 0:

    print(
        "\nWARNING: "
        f"{missing_held_return_count} "
        "held-position daily returns are missing. "
        "The original backtest treats these as 0% returns."
    )


    if STRICT_MISSING_HELD_RETURNS:

        raise ValueError(
            "Missing returns were found while positions "
            "were held. STRICT_MISSING_HELD_RETURNS=True."
        )


########################################
# Dollar Volume / Capacity
########################################

price_df = (
    prepared[
        "backtest_df"
    ]
    .pivot(
        index="Date",
        columns="Ticker",
        values="Close",
    )
    .reindex(
        index=backtest.index,
        columns=test_tickers,
    )
)


volume_df = (
    prepared[
        "backtest_df"
    ]
    .pivot(
        index="Date",
        columns="Ticker",
        values="Volume",
    )
    .reindex(
        index=backtest.index,
        columns=test_tickers,
    )
)


dollar_volume_df = (
    price_df
    * volume_df
)


adv_df = (
    dollar_volume_df
    .rolling(
        ADV_LOOKBACK,
        min_periods=min(
            5,
            ADV_LOOKBACK,
        ),
    )
    .mean()
)


position_notional_df = (
    held_weights
    * PORTFOLIO_CAPITAL
)


position_adv_fraction_df = (
    position_notional_df
    .div(
        adv_df
        .replace(
            0,
            np.nan,
        )
    )
)


trade_weight_df = (
    held_weights
    .diff()
    .abs()
    .fillna(0)
)


trade_notional_df = (
    trade_weight_df
    * PORTFOLIO_CAPITAL
)


trade_adv_fraction_df = (
    trade_notional_df
    .div(
        adv_df
        .replace(
            0,
            np.nan,
        )
    )
)


########################################
# Daily Risk Diagnostics
########################################

diagnostics = pd.DataFrame(
    index=backtest.index
)


diagnostics[
    "Strategy Return"
] = (
    backtest[
        "Strategy_Return"
    ]
)


diagnostics[
    "S&P 500 Return"
] = (
    backtest[
        "S&P500_Return"
    ]
)


diagnostics[
    "Active Return"
] = (
    diagnostics[
        "Strategy Return"
    ]
    - diagnostics[
        "S&P 500 Return"
    ]
)


####################################
# Return Z-Scores
#
# Current return is compared only
# against PREVIOUS observations.
####################################

past_return_mean = (
    diagnostics[
        "Strategy Return"
    ]
    .rolling(
        RETURN_Z_WINDOW,
        min_periods=20,
    )
    .mean()
    .shift(1)
)


past_return_std = (
    diagnostics[
        "Strategy Return"
    ]
    .rolling(
        RETURN_Z_WINDOW,
        min_periods=20,
    )
    .std()
    .shift(1)
)


diagnostics[
    "Return Z"
] = (
    (
        diagnostics[
            "Strategy Return"
        ]
        - past_return_mean
    )
    /
    past_return_std
    .replace(
        0,
        np.nan,
    )
)


past_active_mean = (
    diagnostics[
        "Active Return"
    ]
    .rolling(
        RETURN_Z_WINDOW,
        min_periods=20,
    )
    .mean()
    .shift(1)
)


past_active_std = (
    diagnostics[
        "Active Return"
    ]
    .rolling(
        RETURN_Z_WINDOW,
        min_periods=20,
    )
    .std()
    .shift(1)
)


diagnostics[
    "Active Return Z"
] = (
    (
        diagnostics[
            "Active Return"
        ]
        - past_active_mean
    )
    /
    past_active_std
    .replace(
        0,
        np.nan,
    )
)


####################################
# Rolling Volatility
####################################

diagnostics[
    "Rolling Volatility"
] = (
    diagnostics[
        "Strategy Return"
    ]
    .rolling(
        SHORT_VOL_WINDOW,
        min_periods=max(
            5,
            SHORT_VOL_WINDOW // 2,
        ),
    )
    .std()
    * np.sqrt(252)
)


vol_baseline = (
    diagnostics[
        "Rolling Volatility"
    ]
    .rolling(
        VOL_BASELINE_WINDOW,
        min_periods=min(
            60,
            VOL_BASELINE_WINDOW,
        ),
    )
    .median()
    .shift(1)
)


diagnostics[
    "Volatility Ratio"
] = (
    diagnostics[
        "Rolling Volatility"
    ]
    /
    vol_baseline
    .replace(
        0,
        np.nan,
    )
)


####################################
# Rolling Beta / Correlation
####################################

rolling_covariance = (
    diagnostics[
        "Strategy Return"
    ]
    .rolling(
        BETA_WINDOW,
        min_periods=max(
            20,
            BETA_WINDOW // 2,
        ),
    )
    .cov(
        diagnostics[
            "S&P 500 Return"
        ]
    )
)


rolling_market_variance = (
    diagnostics[
        "S&P 500 Return"
    ]
    .rolling(
        BETA_WINDOW,
        min_periods=max(
            20,
            BETA_WINDOW // 2,
        ),
    )
    .var()
)


diagnostics[
    "Rolling Beta"
] = (
    rolling_covariance
    /
    rolling_market_variance
    .replace(
        0,
        np.nan,
    )
)


diagnostics[
    "Rolling Correlation"
] = (
    diagnostics[
        "Strategy Return"
    ]
    .rolling(
        CORRELATION_WINDOW,
        min_periods=max(
            20,
            CORRELATION_WINDOW // 2,
        ),
    )
    .corr(
        diagnostics[
            "S&P 500 Return"
        ]
    )
)


####################################
# Rolling Sharpe
####################################

rolling_mean = (
    diagnostics[
        "Strategy Return"
    ]
    .rolling(
        ROLLING_SHARPE_WINDOW,
        min_periods=max(
            20,
            ROLLING_SHARPE_WINDOW // 2,
        ),
    )
    .mean()
    * 252
)


rolling_std = (
    diagnostics[
        "Strategy Return"
    ]
    .rolling(
        ROLLING_SHARPE_WINDOW,
        min_periods=max(
            20,
            ROLLING_SHARPE_WINDOW // 2,
        ),
    )
    .std()
    * np.sqrt(252)
)


diagnostics[
    "Rolling Sharpe"
] = (
    rolling_mean
    /
    rolling_std
    .replace(
        0,
        np.nan,
    )
)


####################################
# Portfolio Concentration
####################################

diagnostics[
    "Turnover"
] = turnover


diagnostics[
    "Weight Sum"
] = (
    held_weights
    .sum(
        axis=1
    )
)


diagnostics[
    "Top Weight"
] = (
    held_weights
    .max(
        axis=1
    )
)


diagnostics[
    "Top 3 Weight"
] = (
    held_weights
    .apply(
        lambda row:
            row
            .nlargest(3)
            .sum(),
        axis=1,
    )
)


diagnostics[
    "HHI"
] = (
    (
        held_weights
        ** 2
    )
    .sum(
        axis=1
    )
)


diagnostics[
    "Effective Holdings"
] = (
    1
    /
    diagnostics[
        "HHI"
    ]
    .replace(
        0,
        np.nan,
    )
)


diagnostics[
    "Number Holdings"
] = (
    (
        held_weights
        > WEIGHT_TOLERANCE
    )
    .sum(
        axis=1
    )
)


####################################
# Drawdown
####################################

diagnostics[
    "Drawdown"
] = (
    backtest[
        "Strategy_Drawdown"
    ]
)


####################################
# Capacity
####################################

diagnostics[
    "Max Position % ADV"
] = (
    position_adv_fraction_df
    .max(
        axis=1
    )
)


diagnostics[
    "Max Trade % ADV"
] = (
    trade_adv_fraction_df
    .max(
        axis=1
    )
)


####################################
# P&L Contribution Concentration
####################################

absolute_contribution_sum = (
    strategy_contributions
    .abs()
    .sum(
        axis=1
    )
)


diagnostics[
    "Largest Contribution Share"
] = (
    strategy_contributions
    .abs()
    .max(
        axis=1
    )
    /
    absolute_contribution_sum
    .replace(
        0,
        np.nan,
    )
)


########################################
# Helper: Actual Weights / Contributions
########################################

rebalance_index = (
    pd.DatetimeIndex(
        rebalance_weights.index
    )
    .sort_values()
)


def last_rebalance_on_or_before(
    date,
):

    eligible = (
        rebalance_index[
            rebalance_index
            <= date
        ]
    )

    if len(eligible) == 0:
        return pd.NaT

    return eligible[-1]


def held_weights_text(
    date,
    limit=None,
):

    if date not in held_weights.index:
        return ""

    row = (
        held_weights.loc[
            date
        ]
        .sort_values(
            ascending=False
        )
    )

    row = (
        row[
            row
            > WEIGHT_TOLERANCE
        ]
    )

    if limit is not None:
        row = row.head(limit)

    return ", ".join(
        f"{ticker}:{weight:.1%}"
        for ticker, weight
        in row.items()
    )


def contribution_text(
    date,
    limit=5,
):

    if date not in strategy_contributions.index:
        return ""

    contributions = (
        strategy_contributions.loc[
            date
        ]
        .dropna()
    )

    contributions = (
        contributions[
            contributions
            .abs()
            .sort_values(
                ascending=False
            )
            .index
        ]
        .head(limit)
    )

    parts = []

    for ticker, contribution in contributions.items():

        weight = (
            held_weights
            .loc[
                date,
                ticker,
            ]
            if ticker
            in held_weights.columns
            else np.nan
        )

        stock_return = (
            returns_df
            .loc[
                date,
                ticker,
            ]
            if ticker
            in returns_df.columns
            else np.nan
        )

        parts.append(
            f"{ticker}: "
            f"{contribution:+.2%} contribution "
            f"({weight:.1%} wt, "
            f"{stock_return:+.2%} stock)"
        )

    return " | ".join(parts)


########################################
# Drawdown Episodes
########################################

def extract_drawdown_episodes(
    wealth,
):

    wealth = (
        wealth
        .dropna()
        .copy()
    )

    if wealth.empty:
        return pd.DataFrame()

    index_positions = {
        date: i
        for i, date
        in enumerate(
            wealth.index
        )
    }

    episodes = []

    peak_value = (
        wealth.iloc[0]
    )

    peak_date = (
        wealth.index[0]
    )

    in_drawdown = False

    episode_peak_value = None
    episode_peak_date = None
    trough_value = None
    trough_date = None

    for date, value in wealth.iloc[1:].items():

        if not in_drawdown:

            if value >= peak_value:

                peak_value = value
                peak_date = date

            else:

                in_drawdown = True

                episode_peak_value = (
                    peak_value
                )

                episode_peak_date = (
                    peak_date
                )

                trough_value = value
                trough_date = date

        else:

            if value < trough_value:

                trough_value = value
                trough_date = date

            if value >= episode_peak_value:

                depth = (
                    trough_value
                    /
                    episode_peak_value
                    - 1
                )

                episodes.append(
                    {
                        "Peak Date":
                            episode_peak_date,

                        "Trough Date":
                            trough_date,

                        "Recovery Date":
                            date,

                        "Depth":
                            depth,

                        "Trading Days To Trough":
                            (
                                index_positions[
                                    trough_date
                                ]
                                -
                                index_positions[
                                    episode_peak_date
                                ]
                            ),

                        "Trading Days To Recovery":
                            (
                                index_positions[
                                    date
                                ]
                                -
                                index_positions[
                                    episode_peak_date
                                ]
                            ),

                        "Status":
                            "Recovered",
                    }
                )

                in_drawdown = False

                peak_value = value
                peak_date = date

    if in_drawdown:

        depth = (
            trough_value
            /
            episode_peak_value
            - 1
        )

        episodes.append(
            {
                "Peak Date":
                    episode_peak_date,

                "Trough Date":
                    trough_date,

                "Recovery Date":
                    pd.NaT,

                "Depth":
                    depth,

                "Trading Days To Trough":
                    (
                        index_positions[
                            trough_date
                        ]
                        -
                        index_positions[
                            episode_peak_date
                        ]
                    ),

                "Trading Days To Recovery":
                    (
                        len(
                            wealth
                        )
                        - 1
                        -
                        index_positions[
                            episode_peak_date
                        ]
                    ),

                "Status":
                    "Still Underwater",
            }
        )

    episodes_df = pd.DataFrame(
        episodes
    )

    if episodes_df.empty:
        return episodes_df

    episodes_df = (
        episodes_df[
            episodes_df[
                "Depth"
            ]
            <= DRAWDOWN_REPORT_THRESHOLD
        ]
        .copy()
    )

    if episodes_df.empty:
        return episodes_df

    episodes_df[
        "Weights At Trough"
    ] = (
        episodes_df[
            "Trough Date"
        ]
        .map(
            lambda date:
                held_weights_text(
                    date
                )
        )
    )

    episodes_df[
        "Top Contributions At Trough"
    ] = (
        episodes_df[
            "Trough Date"
        ]
        .map(
            lambda date:
                contribution_text(
                    date
                )
        )
    )

    return (
        episodes_df
        .sort_values(
            "Depth"
        )
        .reset_index(
            drop=True
        )
    )


drawdown_episodes = (
    extract_drawdown_episodes(
        backtest[
            "Strategy"
        ]
    )
)


########################################
# Loss Streaks
########################################

def find_loss_streaks(
    returns,
):

    streaks = []

    start_date = None
    dates = []

    for date, value in returns.items():

        if pd.notna(value) and value < 0:

            if start_date is None:
                start_date = date

            dates.append(date)

        else:

            if (
                start_date is not None
                and len(dates)
                >= LOSS_STREAK_THRESHOLD
            ):

                streak_returns = (
                    returns.loc[
                        dates
                    ]
                )

                streaks.append(
                    {
                        "Start Date":
                            start_date,

                        "End Date":
                            dates[-1],

                        "Trading Days":
                            len(dates),

                        "Cumulative Return":
                            (
                                (
                                    1
                                    + streak_returns
                                )
                                .prod()
                                - 1
                            ),

                        "Weights At End":
                            held_weights_text(
                                dates[-1]
                            ),
                    }
                )

            start_date = None
            dates = []

    if (
        start_date is not None
        and len(dates)
        >= LOSS_STREAK_THRESHOLD
    ):

        streak_returns = (
            returns.loc[
                dates
            ]
        )

        streaks.append(
            {
                "Start Date":
                    start_date,

                "End Date":
                    dates[-1],

                "Trading Days":
                    len(dates),

                "Cumulative Return":
                    (
                        (
                            1
                            + streak_returns
                        )
                        .prod()
                        - 1
                    ),

                "Weights At End":
                    held_weights_text(
                        dates[-1]
                    ),
            }
        )

    return (
        pd.DataFrame(
            streaks
        )
    )


loss_streaks = (
    find_loss_streaks(
        diagnostics[
            "Strategy Return"
        ]
    )
)


if drawdown_episodes.empty:

    drawdown_episodes = pd.DataFrame(
        columns=[
            "Peak Date",
            "Trough Date",
            "Recovery Date",
            "Depth",
            "Trading Days To Trough",
            "Trading Days To Recovery",
            "Status",
            "Weights At Trough",
            "Top Contributions At Trough",
        ]
    )


if loss_streaks.empty:

    loss_streaks = pd.DataFrame(
        columns=[
            "Start Date",
            "End Date",
            "Trading Days",
            "Cumulative Return",
            "Weights At End",
        ]
    )


########################################
# Anomaly Event Builder
########################################

anomaly_records = []


def severity_from_z(
    z_value,
):

    if pd.isna(z_value):
        return "MEDIUM"

    if abs(z_value) >= 4:
        return "HIGH"

    return "MEDIUM"


def add_anomaly(
    date,
    category,
    severity,
    metric,
    value,
    threshold,
    period_start=None,
    period_end=None,
    notes="",
):

    if date not in diagnostics.index:
        return

    last_rebalance = (
        last_rebalance_on_or_before(
            date
        )
    )

    anomaly_records.append(
        {
            "Date":
                date,

            "Category":
                category,

            "Severity":
                severity,

            "Metric":
                metric,

            "Value":
                value,

            "Threshold / Context":
                threshold,

            "Period Start":
                (
                    period_start
                    if period_start
                    is not None
                    else date
                ),

            "Period End":
                (
                    period_end
                    if period_end
                    is not None
                    else date
                ),

            "Strategy Return":
                diagnostics
                .loc[
                    date,
                    "Strategy Return",
                ],

            "S&P 500 Return":
                diagnostics
                .loc[
                    date,
                    "S&P 500 Return",
                ],

            "Active Return":
                diagnostics
                .loc[
                    date,
                    "Active Return",
                ],

            "Drawdown":
                diagnostics
                .loc[
                    date,
                    "Drawdown",
                ],

            "Rolling Volatility":
                diagnostics
                .loc[
                    date,
                    "Rolling Volatility",
                ],

            "Rolling Beta":
                diagnostics
                .loc[
                    date,
                    "Rolling Beta",
                ],

            "Rolling Correlation":
                diagnostics
                .loc[
                    date,
                    "Rolling Correlation",
                ],

            "Rolling Sharpe":
                diagnostics
                .loc[
                    date,
                    "Rolling Sharpe",
                ],

            "Turnover":
                diagnostics
                .loc[
                    date,
                    "Turnover",
                ],

            "Top Weight":
                diagnostics
                .loc[
                    date,
                    "Top Weight",
                ],

            "Top 3 Weight":
                diagnostics
                .loc[
                    date,
                    "Top 3 Weight",
                ],

            "Effective Holdings":
                diagnostics
                .loc[
                    date,
                    "Effective Holdings",
                ],

            "Max Position % ADV":
                diagnostics
                .loc[
                    date,
                    "Max Position % ADV",
                ],

            "Max Trade % ADV":
                diagnostics
                .loc[
                    date,
                    "Max Trade % ADV",
                ],

            "Last Rebalance":
                last_rebalance,

            "Weights Held":
                held_weights_text(
                    date
                ),

            "Top P&L Contributions":
                contribution_text(
                    date
                ),

            "Notes":
                notes,
        }
    )


def add_contiguous_regime_anomalies(
    mask,
    metric_series,
    category,
    threshold_text,
    choose="max",
    severity="MEDIUM",
):

    mask = (
        mask
        .fillna(False)
        .astype(bool)
    )

    group_id = (
        mask
        .ne(
            mask.shift(
                fill_value=False
            )
        )
        .cumsum()
    )

    true_groups = (
        group_id[
            mask
        ]
        .unique()
    )

    for current_group in true_groups:

        dates = (
            mask.index[
                (
                    group_id
                    == current_group
                )
                &
                mask
            ]
        )

        if len(dates) == 0:
            continue

        values = (
            metric_series
            .reindex(
                dates
            )
        )

        if choose == "min":

            event_date = (
                values.idxmin()
            )

        elif choose == "abs":

            event_date = (
                values
                .abs()
                .idxmax()
            )

        else:

            event_date = (
                values.idxmax()
            )

        add_anomaly(
            date=event_date,
            category=category,
            severity=severity,
            metric=(
                metric_series.name
                if metric_series.name
                else category
            ),
            value=(
                metric_series
                .loc[
                    event_date
                ]
            ),
            threshold=threshold_text,
            period_start=dates[0],
            period_end=dates[-1],
        )


########################################
# 1. Daily Return Shocks
########################################

return_shock_mask = (
    (
        diagnostics[
            "Return Z"
        ]
        .abs()
        >= RETURN_Z_THRESHOLD
    )
    |
    (
        diagnostics[
            "Strategy Return"
        ]
        .abs()
        >= ABS_DAILY_RETURN_THRESHOLD
    )
)


for date in diagnostics.index[
    return_shock_mask
]:

    add_anomaly(
        date=date,
        category="Daily Return Shock",
        severity=severity_from_z(
            diagnostics
            .loc[
                date,
                "Return Z",
            ]
        ),
        metric="Strategy Return",
        value=(
            diagnostics
            .loc[
                date,
                "Strategy Return",
            ]
        ),
        threshold=(
            f"|z| >= {RETURN_Z_THRESHOLD:.1f} "
            f"or |return| >= "
            f"{ABS_DAILY_RETURN_THRESHOLD:.1%}"
        ),
        notes=(
            f"Return z-score: "
            f"{diagnostics.loc[date, 'Return Z']:.2f}"
        ),
    )


########################################
# 2. Active Return Shocks
########################################

active_shock_mask = (
    (
        diagnostics[
            "Active Return Z"
        ]
        .abs()
        >= ACTIVE_RETURN_Z_THRESHOLD
    )
    |
    (
        diagnostics[
            "Active Return"
        ]
        .abs()
        >= ABS_ACTIVE_RETURN_THRESHOLD
    )
)


for date in diagnostics.index[
    active_shock_mask
]:

    add_anomaly(
        date=date,
        category="Market-Relative Return Shock",
        severity=severity_from_z(
            diagnostics
            .loc[
                date,
                "Active Return Z",
            ]
        ),
        metric="Active Return",
        value=(
            diagnostics
            .loc[
                date,
                "Active Return",
            ]
        ),
        threshold=(
            f"|z| >= {ACTIVE_RETURN_Z_THRESHOLD:.1f} "
            f"or |active return| >= "
            f"{ABS_ACTIVE_RETURN_THRESHOLD:.1%}"
        ),
        notes=(
            f"Active-return z-score: "
            f"{diagnostics.loc[date, 'Active Return Z']:.2f}"
        ),
    )


########################################
# 3. Drawdown Episodes
########################################

if not drawdown_episodes.empty:

    for _, episode in drawdown_episodes.iterrows():

        add_anomaly(
            date=episode[
                "Trough Date"
            ],
            category="Drawdown Trough",
            severity=(
                "HIGH"
                if episode[
                    "Depth"
                ]
                <= -0.20
                else "MEDIUM"
            ),
            metric="Drawdown",
            value=episode[
                "Depth"
            ],
            threshold=(
                f"Drawdown <= "
                f"{DRAWDOWN_REPORT_THRESHOLD:.0%}"
            ),
            period_start=episode[
                "Peak Date"
            ],
            period_end=(
                episode[
                    "Recovery Date"
                ]
                if pd.notna(
                    episode[
                        "Recovery Date"
                    ]
                )
                else backtest.index[-1]
            ),
            notes=(
                f"{episode['Status']}; "
                f"{episode['Trading Days To Trough']} "
                "trading days to trough; "
                f"{episode['Trading Days To Recovery']} "
                "trading days peak-to-end/recovery."
            ),
        )


########################################
# 4. Volatility Regime Spikes
########################################

volatility_spike_mask = (
    diagnostics[
        "Volatility Ratio"
    ]
    >= VOL_SPIKE_MULTIPLIER
)


add_contiguous_regime_anomalies(
    mask=volatility_spike_mask,
    metric_series=(
        diagnostics[
            "Volatility Ratio"
        ]
    ),
    category="Volatility Regime Spike",
    threshold_text=(
        f"{SHORT_VOL_WINDOW}d annualised volatility "
        f">= {VOL_SPIKE_MULTIPLIER:.2f}x "
        "rolling baseline"
    ),
    choose="max",
    severity="MEDIUM",
)


########################################
# 5. Beta Spikes
########################################

beta_spike_mask = (
    diagnostics[
        "Rolling Beta"
    ]
    .abs()
    >= BETA_ABS_THRESHOLD
)


add_contiguous_regime_anomalies(
    mask=beta_spike_mask,
    metric_series=(
        diagnostics[
            "Rolling Beta"
        ]
    ),
    category="Market Beta Spike",
    threshold_text=(
        f"|{BETA_WINDOW}d beta| >= "
        f"{BETA_ABS_THRESHOLD:.2f}"
    ),
    choose="abs",
    severity="MEDIUM",
)


########################################
# 6. Correlation Spikes
########################################

correlation_spike_mask = (
    diagnostics[
        "Rolling Correlation"
    ]
    .abs()
    >= CORRELATION_ABS_THRESHOLD
)


add_contiguous_regime_anomalies(
    mask=correlation_spike_mask,
    metric_series=(
        diagnostics[
            "Rolling Correlation"
        ]
    ),
    category="Market Correlation Spike",
    threshold_text=(
        f"|{CORRELATION_WINDOW}d correlation| >= "
        f"{CORRELATION_ABS_THRESHOLD:.2f}"
    ),
    choose="abs",
    severity="MEDIUM",
)


########################################
# 7. Rolling Sharpe Collapse
########################################

sharpe_collapse_mask = (
    diagnostics[
        "Rolling Sharpe"
    ]
    <= ROLLING_SHARPE_COLLAPSE
)


add_contiguous_regime_anomalies(
    mask=sharpe_collapse_mask,
    metric_series=(
        diagnostics[
            "Rolling Sharpe"
        ]
    ),
    category="Rolling Sharpe Collapse",
    threshold_text=(
        f"{ROLLING_SHARPE_WINDOW}d Sharpe <= "
        f"{ROLLING_SHARPE_COLLAPSE:.2f}"
    ),
    choose="min",
    severity="MEDIUM",
)


########################################
# 8. Turnover Spikes
########################################

turnover_nonzero = (
    diagnostics[
        "Turnover"
    ][
        diagnostics[
            "Turnover"
        ]
        > 0
    ]
)


if len(turnover_nonzero) >= 5:

    turnover_mean = (
        turnover_nonzero.mean()
    )

    turnover_std = (
        turnover_nonzero.std()
    )

    dynamic_turnover_threshold = (
        turnover_mean
        + TURNOVER_Z_THRESHOLD
        * turnover_std
    )

else:

    dynamic_turnover_threshold = np.inf


turnover_threshold = min(
    TURNOVER_ABS_THRESHOLD,
    dynamic_turnover_threshold,
)


if not np.isfinite(
    turnover_threshold
):

    turnover_threshold = (
        TURNOVER_ABS_THRESHOLD
    )


turnover_spike_mask = (
    diagnostics[
        "Turnover"
    ]
    >= turnover_threshold
)


for date in diagnostics.index[
    turnover_spike_mask
]:

    add_anomaly(
        date=date,
        category="Turnover Spike",
        severity=(
            "HIGH"
            if diagnostics
            .loc[
                date,
                "Turnover",
            ]
            >= 1.5
            else "MEDIUM"
        ),
        metric="Turnover",
        value=(
            diagnostics
            .loc[
                date,
                "Turnover",
            ]
        ),
        threshold=(
            f">= {turnover_threshold:.2f}"
        ),
    )


########################################
# 9. Concentration Anomalies
########################################

effective_holdings_mask = (
    diagnostics[
        "Effective Holdings"
    ]
    < MIN_EFFECTIVE_HOLDINGS
)


add_contiguous_regime_anomalies(
    mask=effective_holdings_mask,
    metric_series=(
        diagnostics[
            "Effective Holdings"
        ]
    ),
    category="Low Effective Diversification",
    threshold_text=(
        f"Effective holdings < "
        f"{MIN_EFFECTIVE_HOLDINGS:.2f}"
    ),
    choose="min",
    severity="MEDIUM",
)


top3_mask = (
    diagnostics[
        "Top 3 Weight"
    ]
    > TOP3_WEIGHT_THRESHOLD
)


add_contiguous_regime_anomalies(
    mask=top3_mask,
    metric_series=(
        diagnostics[
            "Top 3 Weight"
        ]
    ),
    category="Top-3 Concentration",
    threshold_text=(
        f"Top 3 weights > "
        f"{TOP3_WEIGHT_THRESHOLD:.0%}"
    ),
    choose="max",
    severity="MEDIUM",
)


####################################
# Hard Constraint / Weight Integrity
####################################

max_weight_breach_mask = (
    diagnostics[
        "Top Weight"
    ]
    >
    (
        MAX_WEIGHT
        + WEIGHT_TOLERANCE
    )
)


for date in diagnostics.index[
    max_weight_breach_mask
]:

    add_anomaly(
        date=date,
        category="MAX_WEIGHT Constraint Breach",
        severity="HIGH",
        metric="Top Weight",
        value=(
            diagnostics
            .loc[
                date,
                "Top Weight",
            ]
        ),
        threshold=(
            f"> {MAX_WEIGHT:.2%}"
        ),
    )


active_weight_mask = (
    diagnostics[
        "Weight Sum"
    ]
    > WEIGHT_TOLERANCE
)


weight_sum_breach_mask = (
    active_weight_mask
    &
    (
        (
            diagnostics[
                "Weight Sum"
            ]
            - 1
        )
        .abs()
        > 1e-5
    )
)


for date in diagnostics.index[
    weight_sum_breach_mask
]:

    add_anomaly(
        date=date,
        category="Portfolio Weight Integrity",
        severity="HIGH",
        metric="Weight Sum",
        value=(
            diagnostics
            .loc[
                date,
                "Weight Sum",
            ]
        ),
        threshold="Expected fully invested weight sum = 1.0",
    )


########################################
# 10. P&L Contribution Concentration
########################################

contribution_concentration_mask = (
    (
        diagnostics[
            "Largest Contribution Share"
        ]
        >= CONTRIBUTION_CONCENTRATION_THRESHOLD
    )
    &
    (
        diagnostics[
            "Strategy Return"
        ]
        .abs()
        >= CONTRIBUTION_MIN_ABS_STRATEGY_RETURN
    )
)


for date in diagnostics.index[
    contribution_concentration_mask
]:

    add_anomaly(
        date=date,
        category="Single-Name P&L Concentration",
        severity="MEDIUM",
        metric="Largest Contribution Share",
        value=(
            diagnostics
            .loc[
                date,
                "Largest Contribution Share",
            ]
        ),
        threshold=(
            f">= {CONTRIBUTION_CONCENTRATION_THRESHOLD:.0%} "
            "of gross absolute daily contributions "
            f"on |portfolio return| >= "
            f"{CONTRIBUTION_MIN_ABS_STRATEGY_RETURN:.1%}"
        ),
    )


########################################
# 11. Capacity / ADV Anomalies
########################################

position_adv_mask = (
    diagnostics[
        "Max Position % ADV"
    ]
    >= POSITION_ADV_WARNING
)


add_contiguous_regime_anomalies(
    mask=position_adv_mask,
    metric_series=(
        diagnostics[
            "Max Position % ADV"
        ]
    ),
    category="Position Capacity Warning",
    threshold_text=(
        f"Position notional >= "
        f"{POSITION_ADV_WARNING:.1%} "
        f"of {ADV_LOOKBACK}d ADV "
        f"at capital={PORTFOLIO_CAPITAL:,.0f}"
    ),
    choose="max",
    severity="MEDIUM",
)


trade_adv_mask = (
    diagnostics[
        "Max Trade % ADV"
    ]
    >= TRADE_ADV_WARNING
)


for date in diagnostics.index[
    trade_adv_mask
]:

    add_anomaly(
        date=date,
        category="Execution Capacity Warning",
        severity=(
            "HIGH"
            if diagnostics
            .loc[
                date,
                "Max Trade % ADV",
            ]
            >= 0.10
            else "MEDIUM"
        ),
        metric="Max Trade % ADV",
        value=(
            diagnostics
            .loc[
                date,
                "Max Trade % ADV",
            ]
        ),
        threshold=(
            f">= {TRADE_ADV_WARNING:.1%} "
            f"of {ADV_LOOKBACK}d ADV"
        ),
    )


########################################
# 12. Missing Held Returns
########################################

missing_dates = (
    missing_held_returns
    .any(
        axis=1
    )
)


for date in missing_dates.index[
    missing_dates
]:

    missing_names = (
        missing_held_returns
        .columns[
            missing_held_returns
            .loc[
                date
            ]
            .to_numpy()
        ]
        .tolist()
    )

    add_anomaly(
        date=date,
        category="Missing Held Return",
        severity="HIGH",
        metric="Missing Held Tickers",
        value=len(
            missing_names
        ),
        threshold="Expected no missing return while position is held",
        notes=(
            ", ".join(
                missing_names
            )
        ),
    )


########################################
# 13. Loss Streaks
########################################

if not loss_streaks.empty:

    for _, streak in loss_streaks.iterrows():

        add_anomaly(
            date=streak[
                "End Date"
            ],
            category="Loss Streak",
            severity=(
                "HIGH"
                if streak[
                    "Trading Days"
                ]
                >= 8
                else "MEDIUM"
            ),
            metric="Consecutive Losing Days",
            value=streak[
                "Trading Days"
            ],
            threshold=(
                f">= {LOSS_STREAK_THRESHOLD} "
                "consecutive negative days"
            ),
            period_start=streak[
                "Start Date"
            ],
            period_end=streak[
                "End Date"
            ],
            notes=(
                f"Cumulative streak return: "
                f"{streak['Cumulative Return']:.2%}"
            ),
        )


########################################
# Final Anomaly DataFrame
########################################

anomalies = pd.DataFrame(
    anomaly_records
)


if anomalies.empty:

    anomalies = pd.DataFrame(
        columns=[
            "Event ID",
            "Date",
            "Category",
            "Severity",
            "Metric",
            "Value",
            "Threshold / Context",
            "Period Start",
            "Period End",
            "Strategy Return",
            "S&P 500 Return",
            "Active Return",
            "Drawdown",
            "Rolling Volatility",
            "Rolling Beta",
            "Rolling Correlation",
            "Rolling Sharpe",
            "Turnover",
            "Top Weight",
            "Top 3 Weight",
            "Effective Holdings",
            "Max Position % ADV",
            "Max Trade % ADV",
            "Last Rebalance",
            "Weights Held",
            "Top P&L Contributions",
            "Notes",
        ]
    )


if not anomalies.empty:

    severity_order = {
        "HIGH": 0,
        "MEDIUM": 1,
        "LOW": 2,
    }

    anomalies[
        "_Severity Order"
    ] = (
        anomalies[
            "Severity"
        ]
        .map(
            severity_order
        )
        .fillna(9)
    )

    anomalies = (
        anomalies
        .sort_values(
            [
                "_Severity Order",
                "Date",
            ]
        )
        .drop(
            columns=[
                "_Severity Order"
            ]
        )
        .reset_index(
            drop=True
        )
    )

    anomalies.insert(
        0,
        "Event ID",
        range(
            1,
            len(
                anomalies
            )
            + 1
        ),
    )


########################################
# Anomaly Position-Level Detail
########################################

anomaly_weight_records = []


if not anomalies.empty:

    for _, event in anomalies.iterrows():

        date = event[
            "Date"
        ]

        if date not in held_weights.index:
            continue

        for ticker, weight in (
            held_weights
            .loc[
                date
            ]
            .sort_values(
                ascending=False
            )
            .items()
        ):

            if weight <= WEIGHT_TOLERANCE:
                continue

            anomaly_weight_records.append(
                {
                    "Event ID":
                        event[
                            "Event ID"
                        ],

                    "Date":
                        date,

                    "Category":
                        event[
                            "Category"
                        ],

                    "Severity":
                        event[
                            "Severity"
                        ],

                    "Ticker":
                        ticker,

                    "Weight":
                        weight,

                    "Stock Return":
                        (
                            returns_df
                            .loc[
                                date,
                                ticker,
                            ]
                            if ticker
                            in returns_df.columns
                            else np.nan
                        ),

                    "Portfolio Contribution":
                        (
                            strategy_contributions
                            .loc[
                                date,
                                ticker,
                            ]
                            if ticker
                            in strategy_contributions.columns
                            else np.nan
                        ),

                    "20d ADV":
                        (
                            adv_df
                            .loc[
                                date,
                                ticker,
                            ]
                            if ticker
                            in adv_df.columns
                            else np.nan
                        ),

                    "Position % ADV":
                        (
                            position_adv_fraction_df
                            .loc[
                                date,
                                ticker,
                            ]
                            if ticker
                            in position_adv_fraction_df.columns
                            else np.nan
                        ),
                }
            )


anomaly_weights = pd.DataFrame(
    anomaly_weight_records
)


if anomaly_weights.empty:

    anomaly_weights = pd.DataFrame(
        columns=[
            "Event ID",
            "Date",
            "Category",
            "Severity",
            "Ticker",
            "Weight",
            "Stock Return",
            "Portfolio Contribution",
            "20d ADV",
            "Position % ADV",
        ]
    )


########################################
# Rebalance Weights DURING Anomaly Periods
#
# This is particularly useful for drawdowns
# and volatility / beta regimes because it
# shows exactly how the portfolio changed
# while the abnormal period was underway.
########################################

anomaly_period_rebalance_records = []


if not anomalies.empty:

    for _, event in anomalies.iterrows():

        period_start = pd.Timestamp(
            event[
                "Period Start"
            ]
        )

        period_end = pd.Timestamp(
            event[
                "Period End"
            ]
        )

        period_rebalances = (
            rebalance_weights[
                (
                    rebalance_weights.index
                    >= period_start
                )
                &
                (
                    rebalance_weights.index
                    <= period_end
                )
            ]
        )


        # If there was no rebalance inside the
        # anomaly window, include the most recent
        # rebalance that established the holdings.
        if period_rebalances.empty:

            previous_dates = (
                rebalance_weights.index[
                    rebalance_weights.index
                    <= event[
                        "Date"
                    ]
                ]
            )

            if len(previous_dates) > 0:

                previous_date = (
                    previous_dates[-1]
                )

                period_rebalances = (
                    rebalance_weights.loc[
                        [
                            previous_date
                        ]
                    ]
                )


        for rebalance_date, weights in (
            period_rebalances.iterrows()
        ):

            positive = (
                weights[
                    weights
                    > WEIGHT_TOLERANCE
                ]
                .sort_values(
                    ascending=False
                )
            )

            for ticker, weight in positive.items():

                anomaly_period_rebalance_records.append(
                    {
                        "Event ID":
                            event[
                                "Event ID"
                            ],

                        "Category":
                            event[
                                "Category"
                            ],

                        "Severity":
                            event[
                                "Severity"
                            ],

                        "Period Start":
                            period_start,

                        "Period End":
                            period_end,

                        "Rebalance Date":
                            rebalance_date,

                        "Ticker":
                            ticker,

                        "Weight":
                            weight,
                    }
                )


anomaly_period_rebalances = pd.DataFrame(
    anomaly_period_rebalance_records
)


if anomaly_period_rebalances.empty:

    anomaly_period_rebalances = pd.DataFrame(
        columns=[
            "Event ID",
            "Category",
            "Severity",
            "Period Start",
            "Period End",
            "Rebalance Date",
            "Ticker",
            "Weight",
        ]
    )


########################################
# Extreme Daily P&L Tables
########################################

def build_extreme_day_table(
    dates,
):

    rows = []

    for date in dates:

        rows.append(
            {
                "Date":
                    date,

                "Strategy Return":
                    diagnostics
                    .loc[
                        date,
                        "Strategy Return",
                    ],

                "S&P 500 Return":
                    diagnostics
                    .loc[
                        date,
                        "S&P 500 Return",
                    ],

                "Active Return":
                    diagnostics
                    .loc[
                        date,
                        "Active Return",
                    ],

                "Drawdown":
                    diagnostics
                    .loc[
                        date,
                        "Drawdown",
                    ],

                "Return Z":
                    diagnostics
                    .loc[
                        date,
                        "Return Z",
                    ],

                "Weights Held":
                    held_weights_text(
                        date
                    ),

                "Top Contributions":
                    contribution_text(
                        date
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


worst_dates = (
    diagnostics[
        "Strategy Return"
    ]
    .nsmallest(
        TOP_EXTREME_DAYS
    )
    .index
)


best_dates = (
    diagnostics[
        "Strategy Return"
    ]
    .nlargest(
        TOP_EXTREME_DAYS
    )
    .index
)


worst_days = (
    build_extreme_day_table(
        worst_dates
    )
)


best_days = (
    build_extreme_day_table(
        best_dates
    )
)


########################################
# Rebalance Risk Table
########################################

rebalance_risk_records = []


for rebalance_date, row in (
    rebalance_weights
    .sort_index()
    .iterrows()
):

    positive = (
        row[
            row
            > WEIGHT_TOLERANCE
        ]
        .sort_values(
            ascending=False
        )
    )

    hhi = (
        (
            positive
            ** 2
        )
        .sum()
    )

    effective_n = (
        1 / hhi
        if hhi > 0
        else np.nan
    )

    rebalance_risk_records.append(
        {
            "Date":
                rebalance_date,

            "Number Holdings":
                len(
                    positive
                ),

            "Top Weight":
                (
                    positive.iloc[0]
                    if len(
                        positive
                    )
                    > 0
                    else np.nan
                ),

            "Top 3 Weight":
                (
                    positive.head(3).sum()
                ),

            "Effective Holdings":
                effective_n,

            "Weights":
                ", ".join(
                    f"{ticker}:{weight:.1%}"
                    for ticker, weight
                    in positive.items()
                ),
        }
    )


rebalance_risk = pd.DataFrame(
    rebalance_risk_records
)


########################################
# Institutional Risk Summary
########################################

strategy_returns = (
    diagnostics[
        "Strategy Return"
    ]
    .dropna()
)


market_returns = (
    diagnostics[
        "S&P 500 Return"
    ]
    .reindex(
        strategy_returns.index
    )
)


active_returns = (
    strategy_returns
    - market_returns
)


annualised_mean = (
    strategy_returns.mean()
    * 252
)


annualised_vol = (
    strategy_returns.std()
    * np.sqrt(252)
)


downside_deviation = (
    strategy_returns[
        strategy_returns
        < 0
    ]
    .std()
    * np.sqrt(252)
)


sharpe = (
    annualised_mean
    / annualised_vol
    if annualised_vol > 0
    else np.nan
)


sortino = (
    annualised_mean
    / downside_deviation
    if downside_deviation > 0
    else np.nan
)


max_drawdown = (
    diagnostics[
        "Drawdown"
    ]
    .min()
)


calmar = (
    annualised_mean
    / abs(
        max_drawdown
    )
    if max_drawdown < 0
    else np.nan
)


var_95 = (
    strategy_returns
    .quantile(
        0.05
    )
)


var_99 = (
    strategy_returns
    .quantile(
        0.01
    )
)


es_95 = (
    strategy_returns[
        strategy_returns
        <= var_95
    ]
    .mean()
)


es_99 = (
    strategy_returns[
        strategy_returns
        <= var_99
    ]
    .mean()
)


full_covariance = (
    strategy_returns
    .cov(
        market_returns
    )
)


full_market_variance = (
    market_returns.var()
)


full_beta = (
    full_covariance
    / full_market_variance
    if full_market_variance > 0
    else np.nan
)


full_correlation = (
    strategy_returns
    .corr(
        market_returns
    )
)


tracking_error = (
    active_returns.std()
    * np.sqrt(252)
)


information_ratio = (
    (
        active_returns.mean()
        * 252
    )
    / tracking_error
    if tracking_error > 0
    else np.nan
)


risk_summary = pd.DataFrame(
    {
        "Metric": [
            "Total Return",
            "Annualised Mean Return",
            "Annualised Volatility",
            "Sharpe Ratio",
            "Downside Deviation",
            "Sortino Ratio",
            "Maximum Drawdown",
            "Calmar Ratio",
            "Daily VaR 95%",
            "Daily Expected Shortfall 95%",
            "Daily VaR 99%",
            "Daily Expected Shortfall 99%",
            "Market Beta",
            "Market Correlation",
            "Tracking Error",
            "Information Ratio",
            "Daily Hit Rate",
            "Daily Return Skew",
            "Daily Return Excess Kurtosis",
            "Worst Day",
            "Best Day",
            "Average Effective Holdings",
            "Minimum Effective Holdings",
            "Maximum Single Position",
            "Average Daily Turnover",
            "Maximum Daily Turnover",
            "Maximum Position % ADV",
            "Maximum Trade % ADV",
            "Missing Held Returns",
        ],

        "Value": [
            (
                backtest[
                    "Strategy"
                ]
                .iloc[-1]
                - 1
            ),
            annualised_mean,
            annualised_vol,
            sharpe,
            downside_deviation,
            sortino,
            max_drawdown,
            calmar,
            var_95,
            es_95,
            var_99,
            es_99,
            full_beta,
            full_correlation,
            tracking_error,
            information_ratio,
            (
                strategy_returns
                > 0
            ).mean(),
            strategy_returns.skew(),
            strategy_returns.kurt(),
            strategy_returns.min(),
            strategy_returns.max(),
            diagnostics[
                "Effective Holdings"
            ].mean(),
            diagnostics[
                "Effective Holdings"
            ].min(),
            diagnostics[
                "Top Weight"
            ].max(),
            diagnostics[
                "Turnover"
            ].mean(),
            diagnostics[
                "Turnover"
            ].max(),
            diagnostics[
                "Max Position % ADV"
            ].max(),
            diagnostics[
                "Max Trade % ADV"
            ].max(),
            missing_held_return_count,
        ],
    }
)


########################################
# Year-by-Year Performance
########################################

yearly_records = []


for year in sorted(
    strategy_returns
    .index
    .year
    .unique()
):

    year_mask = (
        strategy_returns
        .index
        .year
        == year
    )

    year_strategy = (
        strategy_returns[
            year_mask
        ]
    )

    year_market = (
        market_returns[
            year_mask
        ]
    )

    if year_strategy.empty:
        continue

    year_total = (
        (
            1
            + year_strategy
        )
        .prod()
        - 1
    )

    market_total = (
        (
            1
            + year_market
            .fillna(0)
        )
        .prod()
        - 1
    )

    year_vol = (
        year_strategy.std()
        * np.sqrt(252)
    )

    year_sharpe = (
        (
            year_strategy.mean()
            * 252
        )
        / year_vol
        if year_vol > 0
        else np.nan
    )

    year_wealth = (
        (
            1
            + year_strategy
        )
        .cumprod()
    )

    year_dd = (
        year_wealth
        /
        year_wealth.cummax()
        - 1
    )

    yearly_records.append(
        {
            "Year":
                int(year),

            "Strategy Return":
                year_total,

            "S&P 500 Return":
                market_total,

            "Excess Return":
                year_total
                - market_total,

            "Annualised Volatility":
                year_vol,

            "Sharpe":
                year_sharpe,

            "Maximum Drawdown":
                year_dd.min(),

            "Worst Day":
                year_strategy.min(),

            "Best Day":
                year_strategy.max(),
        }
    )


yearly_performance = pd.DataFrame(
    yearly_records
)


########################################
# Print Risk Summary
########################################

def format_risk_value(
    metric,
    value,
):

    percentage_metrics = {
        "Total Return",
        "Annualised Mean Return",
        "Annualised Volatility",
        "Downside Deviation",
        "Maximum Drawdown",
        "Daily VaR 95%",
        "Daily Expected Shortfall 95%",
        "Daily VaR 99%",
        "Daily Expected Shortfall 99%",
        "Tracking Error",
        "Daily Hit Rate",
        "Worst Day",
        "Best Day",
        "Maximum Single Position",
        "Average Daily Turnover",
        "Maximum Daily Turnover",
        "Maximum Position % ADV",
        "Maximum Trade % ADV",
    }

    if pd.isna(value):
        return "NaN"

    if metric in percentage_metrics:
        return f"{value:.2%}"

    if metric == "Missing Held Returns":
        return f"{int(value)}"

    return f"{value:.3f}"


print("\n")
print("=" * 110)
print("HEDGE-FUND STYLE RISK SUMMARY")
print("=" * 110)


for _, row in risk_summary.iterrows():

    print(
        f"{row['Metric']:<35}"
        f"{format_risk_value(row['Metric'], row['Value']):>18}"
    )


########################################
# Print Yearly Performance
########################################

print("\n")
print("=" * 110)
print("YEAR-BY-YEAR PERFORMANCE")
print("=" * 110)


if yearly_performance.empty:

    print(
        "No yearly performance available."
    )

else:

    yearly_display = (
        yearly_performance
        .copy()
    )

    for column in [
        "Strategy Return",
        "S&P 500 Return",
        "Excess Return",
        "Annualised Volatility",
        "Maximum Drawdown",
        "Worst Day",
        "Best Day",
    ]:

        yearly_display[
            column
        ] = (
            yearly_display[
                column
            ]
            .map(
                lambda value:
                    f"{value:.2%}"
                    if pd.notna(
                        value
                    )
                    else "NaN"
            )
        )

    yearly_display[
        "Sharpe"
    ] = (
        yearly_display[
            "Sharpe"
        ]
        .map(
            lambda value:
                f"{value:.3f}"
                if pd.notna(
                    value
                )
                else "NaN"
        )
    )

    print(
        yearly_display
        .to_string(
            index=False
        )
    )


########################################
# Print Drawdown Episodes
########################################

print("\n")
print("=" * 110)
print(
    f"DRAWDOWN EPISODES <= "
    f"{DRAWDOWN_REPORT_THRESHOLD:.0%}"
)
print("=" * 110)


if drawdown_episodes.empty:

    print(
        "No drawdown episode crossed "
        "the configured threshold."
    )

else:

    drawdown_display = (
        drawdown_episodes
        .head(
            TOP_DRAWDOWN_EPISODES
        )
        .copy()
    )

    drawdown_display[
        "Depth"
    ] = (
        drawdown_display[
            "Depth"
        ]
        .map(
            lambda value:
                f"{value:.2%}"
        )
    )

    print(
        drawdown_display[
            [
                "Peak Date",
                "Trough Date",
                "Recovery Date",
                "Depth",
                "Trading Days To Trough",
                "Trading Days To Recovery",
                "Status",
                "Weights At Trough",
            ]
        ]
        .to_string(
            index=False
        )
    )


########################################
# Print Worst / Best Days With Weights
########################################

def format_extreme_days(
    dataframe,
):

    display = dataframe.copy()

    for column in [
        "Strategy Return",
        "S&P 500 Return",
        "Active Return",
        "Drawdown",
    ]:

        display[
            column
        ] = (
            display[
                column
            ]
            .map(
                lambda value:
                    f"{value:.2%}"
                    if pd.notna(
                        value
                    )
                    else "NaN"
            )
        )

    display[
        "Return Z"
    ] = (
        display[
            "Return Z"
        ]
        .map(
            lambda value:
                f"{value:.2f}"
                if pd.notna(
                    value
                )
                else "NaN"
        )
    )

    return display


print("\n")
print("=" * 110)
print(
    f"WORST {TOP_EXTREME_DAYS} DAYS "
    "- INCLUDING ACTUAL HELD WEIGHTS"
)
print("=" * 110)


print(
    format_extreme_days(
        worst_days
    )
    .to_string(
        index=False
    )
)


print("\n")
print("=" * 110)
print(
    f"BEST {TOP_EXTREME_DAYS} DAYS "
    "- INCLUDING ACTUAL HELD WEIGHTS"
)
print("=" * 110)


print(
    format_extreme_days(
        best_days
    )
    .to_string(
        index=False
    )
)


########################################
# Print Anomaly Summary
########################################

print("\n")
print("=" * 110)
print("ANOMALY DETECTION SUMMARY")
print("=" * 110)


if anomalies.empty:

    print(
        "No anomalies crossed the configured thresholds."
    )

else:

    anomaly_counts = (
        anomalies
        .groupby(
            [
                "Severity",
                "Category",
            ]
        )
        .size()
        .reset_index(
            name="Events"
        )
        .sort_values(
            [
                "Severity",
                "Events",
            ],
            ascending=[
                True,
                False,
            ],
        )
    )

    print(
        anomaly_counts
        .to_string(
            index=False
        )
    )


    print("\n")
    print("-" * 110)
    print(
        f"TOP {min(TOP_ANOMALIES_TO_PRINT, len(anomalies))} "
        "ANOMALY EVENTS"
    )
    print("-" * 110)


    anomaly_display = (
        anomalies
        .head(
            TOP_ANOMALIES_TO_PRINT
        )[
            [
                "Event ID",
                "Date",
                "Category",
                "Severity",
                "Value",
                "Strategy Return",
                "S&P 500 Return",
                "Drawdown",
                "Turnover",
                "Effective Holdings",
                "Weights Held",
                "Top P&L Contributions",
            ]
        ]
        .copy()
    )


    for column in [
        "Strategy Return",
        "S&P 500 Return",
        "Drawdown",
        "Turnover",
    ]:

        anomaly_display[
            column
        ] = (
            anomaly_display[
                column
            ]
            .map(
                lambda value:
                    f"{value:.2%}"
                    if pd.notna(
                        value
                    )
                    else "NaN"
            )
        )


    print(
        anomaly_display
        .to_string(
            index=False
        )
    )


########################################
# Save Institutional Diagnostics
########################################

DATA_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


if anomalies.empty:

    pd.DataFrame(
        columns=[
            "Event ID",
            "Date",
            "Category",
            "Severity",
        ]
    ).to_csv(
        ANOMALY_RESULTS_CSV,
        index=False,
    )

else:

    anomalies.to_csv(
        ANOMALY_RESULTS_CSV,
        index=False,
    )


with sqlite3.connect(
    ANOMALY_RESULTS_DB
) as connection:

    pd.DataFrame(
        [
            result_row
        ]
    ).to_sql(
        "Backtest_Summary",
        connection,
        if_exists="replace",
        index=False,
    )


    risk_summary.to_sql(
        "Risk_Summary",
        connection,
        if_exists="replace",
        index=False,
    )


    diagnostics.reset_index(
        names="Date"
    ).to_sql(
        "Daily_Diagnostics",
        connection,
        if_exists="replace",
        index=False,
    )


    held_weights.reset_index(
        names="Date"
    ).to_sql(
        "Daily_Held_Weights",
        connection,
        if_exists="replace",
        index=False,
    )


    strategy_contributions.reset_index(
        names="Date"
    ).to_sql(
        "Daily_Contributions",
        connection,
        if_exists="replace",
        index=False,
    )


    rebalance_risk.to_sql(
        "Rebalance_Risk",
        connection,
        if_exists="replace",
        index=False,
    )


    anomalies.to_sql(
        "Anomalies",
        connection,
        if_exists="replace",
        index=False,
    )


    anomaly_weights.to_sql(
        "Anomaly_Weights",
        connection,
        if_exists="replace",
        index=False,
    )


    anomaly_period_rebalances.to_sql(
        "Anomaly_Period_Rebalances",
        connection,
        if_exists="replace",
        index=False,
    )


    drawdown_episodes.to_sql(
        "Drawdown_Episodes",
        connection,
        if_exists="replace",
        index=False,
    )


    loss_streaks.to_sql(
        "Loss_Streaks",
        connection,
        if_exists="replace",
        index=False,
    )


    worst_days.to_sql(
        "Worst_Days",
        connection,
        if_exists="replace",
        index=False,
    )


    best_days.to_sql(
        "Best_Days",
        connection,
        if_exists="replace",
        index=False,
    )


    yearly_performance.to_sql(
        "Yearly_Performance",
        connection,
        if_exists="replace",
        index=False,
    )


print("\n")
print("=" * 110)
print("ANOMALY REPORT FILES")
print("=" * 110)

print(
    f"CSV: "
    f"{ANOMALY_RESULTS_CSV}"
)

print(
    f"DB:  "
    f"{ANOMALY_RESULTS_DB}"
)


########################################
# Plots
########################################

if PLOT_RESULTS:

    backtest[
        [
            "S&P 500",
            "Strategy",
        ]
    ].plot(
        figsize=(
            12,
            6,
        ),
        title=(
            "Single Backtest Strategy "
            "vs S&P 500"
        ),
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


    backtest[
        [
            "S&P500_Drawdown",
            "Strategy_Drawdown",
        ]
    ].rename(
        columns={
            "S&P500_Drawdown":
                "S&P 500 Drawdown",

            "Strategy_Drawdown":
                "Strategy Drawdown",
        }
    ).plot(
        figsize=(
            12,
            6,
        ),
        title=(
            "Single Backtest Drawdown"
        ),
    )

    plt.xlabel(
        "Date"
    )

    plt.ylabel(
        "Drawdown"
    )

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.show()


if (
    PLOT_ANOMALIES
    and not anomalies.empty
):

    figure, axis = plt.subplots(
        figsize=(
            12,
            6,
        )
    )

    axis.plot(
        backtest.index,
        backtest[
            "Strategy"
        ],
        label="Strategy",
    )

    return_shock_dates = (
        anomalies[
            anomalies[
                "Category"
            ]
            .isin(
                [
                    "Daily Return Shock",
                    "Drawdown Trough",
                ]
            )
        ][
            "Date"
        ]
        .drop_duplicates()
    )

    return_shock_dates = [
        date
        for date
        in return_shock_dates
        if date
        in backtest.index
    ]

    if return_shock_dates:

        axis.scatter(
            return_shock_dates,
            backtest.loc[
                return_shock_dates,
                "Strategy",
            ],
            label=(
                "Return Shock / Drawdown Trough"
            ),
        )

    axis.set_title(
        "Strategy With Major Anomaly Dates"
    )

    axis.set_xlabel(
        "Date"
    )

    axis.set_ylabel(
        "Growth of £1"
    )

    axis.grid(
        alpha=0.3
    )

    axis.legend()

    plt.tight_layout()

    plt.show()
import ast
import itertools
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
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.linear_model import (
    ElasticNet,
    HuberRegressor,
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import (
    KNeighborsClassifier,
    KNeighborsRegressor,
)
from sklearn.neural_network import (
    MLPClassifier,
    MLPRegressor,
)
from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler,
)
from sklearn.svm import (
    SVC,
    SVR,
)
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import (
    XGBClassifier,
    XGBRegressor,
)


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

RESULTS_CSV = (
    DATA_DIR
    / "Automated_Backtest_Results.csv"
)

RESULTS_DB = (
    DATA_DIR
    / "Automated_Backtest_Results.db"
)


########################################
# Parameter Sweep
#
# These are intentionally broad but still
# sensible ranges for this portfolio model.
########################################

IMPORTANCE_PRESETS = [
    {
        "Name": "Balanced",
        "Alpha": 0.40,
        "Volatility": 0.30,
        "Downside": 0.30,
    },
    {
        "Name": "Alpha 50",
        "Alpha": 0.50,
        "Volatility": 0.25,
        "Downside": 0.25,
    },
    {
        "Name": "Alpha 60",
        "Alpha": 0.60,
        "Volatility": 0.20,
        "Downside": 0.20,
    },
    {
        "Name": "Volatility Heavy",
        "Alpha": 0.40,
        "Volatility": 0.40,
        "Downside": 0.20,
    },
    {
        "Name": "Downside Heavy",
        "Alpha": 0.40,
        "Volatility": 0.20,
        "Downside": 0.40,
    },
]

MAX_WEIGHT_VALUES = [
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
]

CONCENTRATION_PENALTIES = [
    0.00,
    0.05,
    0.10,
    0.20,
]

REBALANCE_VALUES = [
    20,
    40,
    60,
    90,
    120,
]

# Trading 212 commission assumption.
TRADING_FEE = 0.00


########################################
# Which Universe Tests To Run
########################################

RUN_MANUAL_UNIVERSE = True
RUN_ALL_CACHED_UNIVERSE = True
RUN_LIQUIDITY_UNIVERSES = True

# Your original / specialised training experiment.
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
# Liquidity Test Settings
########################################

# Liquidity is measured with median:
#
#     Close * Volume
#
# Training liquidity is measured using the training data.
#
# Test liquidity is classified using only the FIRST N
# trading days of the test period. All experiments then
# begin AFTER this classification window so that the
# later test does not use future liquidity information.
LIQUIDITY_CLASSIFICATION_DAYS = 60

# The cached stocks are split into three equally sized
# groups after ranking by median dollar volume:
#
#     Low
#     Medium
#     High
#
# These are RELATIVE liquidity buckets within whatever
# universe was included when the database was built.
USE_COMMON_POST_LIQUIDITY_START = True


########################################
# Output Settings
########################################

# This can create a lot of terminal output.
PRINT_EACH_RESULT = True

# Do not open thousands of matplotlib windows.
PLOT_EACH_RESULT = False

# Plot only the best Sharpe result at the end.
PLOT_BEST_RESULT = True

# How many rows to print in the final overall ranking.
FINAL_TOP_N = 30


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
    num_classes=None,
):

    params = parse_parameters(parameters)

    name = (
        model_name
        .lower()
        .strip()
    )

    target_type = (
        target_type
        .lower()
        .strip()
    )

    is_regression = (
        target_type
        == "continuous"
    )


    ####################################
    # Existing Model Families
    ####################################

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

        params.setdefault(
            "max_iter",
            5000,
        )

        return LogisticRegression(
            **params
        )

    if name in {
        "l2 logistic regression",
        "l2 logistic",
    }:

        params.setdefault(
            "penalty",
            "l2",
        )

        params.setdefault(
            "solver",
            "lbfgs",
        )

        params.setdefault(
            "max_iter",
            5000,
        )

        return LogisticRegression(
            **params
        )

    if name in {
        "l1 logistic regression",
        "l1 logistic",
    }:

        params.setdefault(
            "penalty",
            "l1",
        )

        params.setdefault(
            "solver",
            "liblinear",
        )

        params.setdefault(
            "max_iter",
            5000,
        )

        return LogisticRegression(
            **params
        )

    if name in {
        "elastic net logistic regression",
        "elasticnet logistic regression",
        "elastic net logistic",
    }:

        params.setdefault(
            "penalty",
            "elasticnet",
        )

        params.setdefault(
            "solver",
            "saga",
        )

        params.setdefault(
            "max_iter",
            5000,
        )

        return LogisticRegression(
            **params
        )

    if name in {
        "random forest",
        "random forest regressor",
        "random forest classifier",
    }:

        if is_regression:

            return RandomForestRegressor(
                **params
            )

        return RandomForestClassifier(
            **params
        )

    if name in {
        "extra trees",
        "extra trees regressor",
        "extra trees classifier",
    }:

        if is_regression:

            return ExtraTreesRegressor(
                **params
            )

        return ExtraTreesClassifier(
            **params
        )

    if name in {
        "gradient boosting",
        "gradient boosting regressor",
        "gradient boosting classifier",
    }:

        if is_regression:

            return GradientBoostingRegressor(
                **params
            )

        return GradientBoostingClassifier(
            **params
        )

    if name in {
        "hist gradient boosting",
        "histogram gradient boosting",
        "histgradientboosting",
    }:

        if is_regression:

            return HistGradientBoostingRegressor(
                **params
            )

        return HistGradientBoostingClassifier(
            **params
        )

    if name in {
        "lightgbm",
        "lgbm",
    }:

        if is_regression:

            return LGBMRegressor(
                **params
            )

        return LGBMClassifier(
            **params
        )


    ####################################
    # Additional Continuous Models
    ####################################

    if is_regression:

        if name == "huber":

            return HuberRegressor(
                epsilon=params[
                    "epsilon"
                ],
                alpha=params[
                    "alpha"
                ],
                max_iter=1000,
            )

        if name == "xgboost":

            return XGBRegressor(
                **params,
                objective=(
                    "reg:squarederror"
                ),
                random_state=42,
                n_jobs=-1,
            )

        if name == "svr":

            return SVR(
                **params
            )

        if name in {
            "knn",
            "knn regressor",
        }:

            return KNeighborsRegressor(
                **params,
                n_jobs=-1,
            )

        if name in {
            "mlp",
            "mlp regressor",
        }:

            return MLPRegressor(
                **params,
                max_iter=1000,
                random_state=42,
            )


    ####################################
    # Multinomial Logistic Models
    ####################################

    if name == "multinomial logistic regression":

        return LogisticRegression(
            C=np.inf,
            class_weight=(
                params.get(
                    "class_weight"
                )
            ),
            solver="lbfgs",
            max_iter=5000,
            random_state=42,
        )

    if name == "l2 multinomial logistic regression":

        return LogisticRegression(
            C=params[
                "C"
            ],
            l1_ratio=0,
            class_weight=(
                params.get(
                    "class_weight"
                )
            ),
            solver="lbfgs",
            max_iter=5000,
            random_state=42,
        )

    if name == "l1 multinomial logistic regression":

        return LogisticRegression(
            C=params[
                "C"
            ],
            l1_ratio=1,
            class_weight=(
                params.get(
                    "class_weight"
                )
            ),
            solver="saga",
            max_iter=5000,
            random_state=42,
        )

    if name == "elastic net multinomial logistic regression":

        return LogisticRegression(
            C=params[
                "C"
            ],
            l1_ratio=params[
                "l1_ratio"
            ],
            class_weight=(
                params.get(
                    "class_weight"
                )
            ),
            solver="saga",
            max_iter=5000,
            random_state=42,
        )


    ####################################
    # Additional Classification Models
    ####################################

    if name == "lda":

        return LinearDiscriminantAnalysis(
            **params
        )

    if name == "qda":

        return QuadraticDiscriminantAnalysis(
            **params
        )

    if name == "xgboost":

        # Validation stores class_weight as a search parameter,
        # but XGBoost receives it through sample_weight at fit time.
        params.pop(
            "class_weight",
            None,
        )

        if target_type == "binary":

            return XGBClassifier(
                **params,
                objective=(
                    "binary:logistic"
                ),
                eval_metric="logloss",
                random_state=42,
                n_jobs=-1,
            )

        if num_classes is None:

            raise ValueError(
                "num_classes is required for "
                "multiclass XGBoost."
            )

        return XGBClassifier(
            **params,
            objective=(
                "multi:softprob"
            ),
            num_class=int(
                num_classes
            ),
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=-1,
        )

    if name == "svm":

        return SVC(
            **params,
            probability=True,
            random_state=42,
        )

    if name in {
        "knn",
        "knn classifier",
    }:

        return KNeighborsClassifier(
            **params,
            n_jobs=-1,
        )

    if name in {
        "naive bayes",
        "gaussian naive bayes",
    }:

        return GaussianNB(
            **params
        )

    if name in {
        "mlp",
        "mlp classifier",
    }:

        return MLPClassifier(
            **params,
            max_iter=1000,
            random_state=42,
        )


    raise ValueError(
        f"Unknown model '{model_name}' "
        f"for target type '{target_type}'."
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
# Common Backtest Start
#
# The test liquidity buckets use only the
# first N test dates. To avoid using those
# observations to classify the universe
# and then pretending they were untouched,
# all experiments can start AFTER them.
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


if (
    USE_COMMON_POST_LIQUIDITY_START
    and RUN_LIQUIDITY_UNIVERSES
):

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
# Universe Scenarios
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


universe_scenarios = []


if RUN_MANUAL_UNIVERSE:

    manual_train = existing_tickers(
        MANUAL_TRAIN_TICKERS,
        available_train_tickers,
    )

    manual_test = existing_tickers(
        MANUAL_TEST_TICKERS,
        available_test_tickers,
    )

    universe_scenarios.append(
        {
            "Name":
                "Manual AAPL-MSFT",

            "Train Tickers":
                manual_train,

            "Test Tickers":
                manual_test,
        }
    )


if RUN_ALL_CACHED_UNIVERSE:

    universe_scenarios.append(
        {
            "Name":
                "All Cached",

            "Train Tickers":
                available_train_tickers,

            "Test Tickers":
                available_test_tickers,
        }
    )


if RUN_LIQUIDITY_UNIVERSES:

    for bucket_name in [
        "Low Liquidity",
        "Medium Liquidity",
        "High Liquidity",
    ]:

        universe_scenarios.append(
            {
                "Name":
                    bucket_name,

                "Train Tickers":
                    training_liquidity_buckets[
                        bucket_name
                    ],

                "Test Tickers":
                    test_liquidity_buckets[
                        bucket_name
                    ],
            }
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
    "huber",
    "svr",
    "knn",
    "knn regressor",
    "knn classifier",
    "mlp",
    "mlp regressor",
    "mlp classifier",
    "logistic regression",
    "logistic",
    "l1 logistic regression",
    "l1 logistic",
    "l2 logistic regression",
    "l2 logistic",
    "elastic net logistic regression",
    "elasticnet logistic regression",
    "elastic net logistic",
    "multinomial logistic regression",
    "l2 multinomial logistic regression",
    "l1 multinomial logistic regression",
    "elastic net multinomial logistic regression",
    "lda",
    "qda",
    "svm",
}


def clean_binary_target(
    y,
    target,
):

    y = y.copy()

    ####################################
    # Future Direction
    #
    # Validation / final testing uses the
    # ordered classes -1 / +1 for this target.
    ####################################

    if target.startswith(
        "Future Direction"
    ):

        y = pd.Series(
            np.where(
                y.to_numpy() > 0,
                1,
                -1,
            ),
            index=y.index,
        )

    classes = np.sort(
        pd.Series(
            y
        )
        .dropna()
        .unique()
    )

    if len(classes) > 2:

        raise ValueError(
            f"{target} is defined as binary "
            f"but contains classes {classes}."
        )

    return y


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

    target_type = (
        specification[
            "target_type"
        ]
        .lower()
        .strip()
    )

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

    if target_type == "binary":

        y_train = clean_binary_target(
            y_train,
            target,
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

    num_classes = (
        int(
            y_train.nunique()
        )
        if target_type
        != "continuous"
        else None
    )

    model = build_model(
        model_name=model_name,
        target_type=target_type,
        parameters=parameters,
        num_classes=num_classes,
    )

    scaler = None

    if (
        model_name
        .lower()
        .strip()
        in SCALE_MODELS
    ):

        scaler = StandardScaler()

        X_train_model = (
            scaler
            .fit_transform(
                X_train
            )
        )

    else:

        X_train_model = X_train

    class_values = None
    label_encoder = None

    is_xgboost_classifier = (
        target_type
        != "continuous"
        and model_name
        .lower()
        .strip()
        == "xgboost"
    )

    if is_xgboost_classifier:

        label_encoder = LabelEncoder()

        y_fit = (
            label_encoder
            .fit_transform(
                y_train
            )
        )

        class_weight = (
            parse_parameters(
                parameters
            )
            .get(
                "class_weight"
            )
        )

        sample_weight = None

        if class_weight is not None:

            sample_weight = (
                compute_sample_weight(
                    class_weight=(
                        class_weight
                    ),
                    y=y_train,
                )
            )

        model.fit(
            X_train_model,
            y_fit,
            sample_weight=(
                sample_weight
            ),
        )

        class_values = (
            label_encoder
            .classes_
            .copy()
        )

    else:

        model.fit(
            X_train_model,
            y_train,
        )

        if target_type != "continuous":

            if not hasattr(
                model,
                "classes_",
            ):

                raise ValueError(
                    f"{model_name} does not expose "
                    f"classes_ for {target}."
                )

            class_values = np.asarray(
                model.classes_
            ).copy()

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

        "class_values":
            class_values,

        "label_encoder":
            label_encoder,
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
        == "continuous"
    ):

        return (
            model_info[
                "model"
            ]
            .predict(
                X
            )
        )

    ####################################
    # Binary / Multiclass -> One Scalar
    #
    # Use the probability-weighted expected
    # class so every selected target produces
    # one continuous portfolio signal.
    #
    # [0, 1]    -> P(1)
    # [-1, 1]   -> P(+1) - P(-1)
    # [-1,0,1]  -> expected class state
    ####################################

    if not hasattr(
        model_info[
            "model"
        ],
        "predict_proba",
    ):

        raise ValueError(
            f"{model_info['model_name']} does not "
            f"support predict_proba for "
            f"{model_info['target']}."
        )

    probabilities = (
        model_info[
            "model"
        ]
        .predict_proba(
            X
        )
    )

    class_values = np.asarray(
        model_info[
            "class_values"
        ]
    )

    try:

        numeric_classes = (
            class_values
            .astype(float)
        )

    except (
        TypeError,
        ValueError,
    ) as error:

        raise ValueError(
            f"Classes for {model_info['target']} "
            f"must be numeric to create a scalar "
            f"portfolio signal. Got "
            f"{class_values.tolist()}."
        ) from error

    if (
        probabilities.shape[1]
        != len(
            numeric_classes
        )
    ):

        raise ValueError(
            f"Probability columns for "
            f"{model_info['target']} do not match "
            "the stored class values."
        )

    return (
        probabilities
        @ numeric_classes
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
# Build Parameter Grid
########################################

parameter_grid = list(
    itertools.product(
        IMPORTANCE_PRESETS,
        MAX_WEIGHT_VALUES,
        CONCENTRATION_PENALTIES,
        REBALANCE_VALUES,
    )
)


total_possible_experiments = (
    len(
        universe_scenarios
    )
    * len(
        parameter_grid
    )
)


print("\n")
print("=" * 90)
print("AUTOMATED PARAMETER SWEEP")
print("=" * 90)

print(
    f"Universe scenarios:       "
    f"{len(universe_scenarios)}"
)

print(
    f"Importance presets:       "
    f"{len(IMPORTANCE_PRESETS)}"
)

print(
    f"Max-weight values:        "
    f"{len(MAX_WEIGHT_VALUES)}"
)

print(
    f"Concentration penalties: "
    f"{len(CONCENTRATION_PENALTIES)}"
)

print(
    f"Rebalance frequencies:   "
    f"{len(REBALANCE_VALUES)}"
)

print(
    f"Maximum experiments:     "
    f"{total_possible_experiments:,}"
)


########################################
# Run Sweep
########################################

all_results = []

best_backtest = None
best_rebalance_weights = None
best_result_row = None

experiment_number = 0


for scenario in universe_scenarios:

    try:

        prepared = (
            prepare_universe_scenario(
                scenario
            )
        )

    except Exception as error:

        logger.exception(
            "Skipping universe %s: %s",
            scenario[
                "Name"
            ],
            error,
        )

        continue


    for (
        importance_preset,
        max_weight,
        concentration_penalty,
        rebalance_every,
    ) in parameter_grid:

        experiment_number += 1

        try:

            (
                result_row,
                backtest,
                rebalance_weights,
            ) = run_single_backtest(
                prepared=prepared,
                importance_preset=importance_preset,
                max_weight=max_weight,
                concentration_penalty=(
                    concentration_penalty
                ),
                rebalance_every=(
                    rebalance_every
                ),
            )

            result_row[
                "Experiment"
            ] = experiment_number

            all_results.append(
                result_row
            )

            if PRINT_EACH_RESULT:

                print_single_result(
                    experiment_number,
                    total_possible_experiments,
                    result_row,
                )

            if PLOT_EACH_RESULT:

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
                        f"{result_row['Universe']} | "
                        f"Sharpe {result_row['Sharpe']:.3f}"
                    ),
                )

                plt.tight_layout()
                plt.show()

            if (
                best_result_row is None
                or (
                    pd.notna(
                        result_row[
                            "Sharpe"
                        ]
                    )
                    and result_row[
                        "Sharpe"
                    ]
                    > best_result_row[
                        "Sharpe"
                    ]
                )
            ):

                best_result_row = (
                    result_row.copy()
                )

                best_backtest = (
                    backtest.copy()
                )

                best_rebalance_weights = (
                    rebalance_weights.copy()
                )

        except Exception as error:

            logger.warning(
                "Experiment failed | "
                "Universe=%s | "
                "Preset=%s | "
                "MaxWeight=%.2f | "
                "Penalty=%.2f | "
                "Rebalance=%d | %s",
                scenario[
                    "Name"
                ],
                importance_preset[
                    "Name"
                ],
                max_weight,
                concentration_penalty,
                rebalance_every,
                error,
            )


########################################
# Results DataFrame
########################################

if not all_results:

    raise ValueError(
        "No successful experiments "
        "were completed."
    )


results_df = pd.DataFrame(
    all_results
)


results_df = (
    results_df
    .sort_values(
        [
            "Sharpe",
            "Strategy Return",
        ],
        ascending=[
            False,
            False,
        ],
    )
    .reset_index(
        drop=True
    )
)


########################################
# Summaries
########################################

best_by_universe = (
    results_df
    .sort_values(
        "Sharpe",
        ascending=False,
    )
    .groupby(
        "Universe",
        as_index=False,
    )
    .first()
)


mean_by_universe = (
    results_df
    .groupby(
        "Universe",
        as_index=False,
    )
    .agg(
        Experiments=(
            "Sharpe",
            "size",
        ),
        Mean_Sharpe=(
            "Sharpe",
            "mean",
        ),
        Median_Sharpe=(
            "Sharpe",
            "median",
        ),
        Mean_Return=(
            "Strategy Return",
            "mean",
        ),
        Mean_Relative=(
            "Relative Performance",
            "mean",
        ),
        Mean_Max_Drawdown=(
            "Max Drawdown",
            "mean",
        ),
    )
    .sort_values(
        "Mean_Sharpe",
        ascending=False,
    )
)


mean_by_max_weight = (
    results_df
    .groupby(
        "Max Weight",
        as_index=False,
    )
    .agg(
        Mean_Sharpe=(
            "Sharpe",
            "mean",
        ),
        Median_Sharpe=(
            "Sharpe",
            "median",
        ),
        Mean_Return=(
            "Strategy Return",
            "mean",
        ),
        Mean_Max_Drawdown=(
            "Max Drawdown",
            "mean",
        ),
    )
    .sort_values(
        "Max Weight"
    )
)


mean_by_rebalance = (
    results_df
    .groupby(
        "Rebalance Days",
        as_index=False,
    )
    .agg(
        Mean_Sharpe=(
            "Sharpe",
            "mean",
        ),
        Median_Sharpe=(
            "Sharpe",
            "median",
        ),
        Mean_Return=(
            "Strategy Return",
            "mean",
        ),
        Mean_Max_Drawdown=(
            "Max Drawdown",
            "mean",
        ),
    )
    .sort_values(
        "Rebalance Days"
    )
)


mean_by_penalty = (
    results_df
    .groupby(
        "Concentration Penalty",
        as_index=False,
    )
    .agg(
        Mean_Sharpe=(
            "Sharpe",
            "mean",
        ),
        Median_Sharpe=(
            "Sharpe",
            "median",
        ),
        Mean_Return=(
            "Strategy Return",
            "mean",
        ),
        Mean_Max_Drawdown=(
            "Max Drawdown",
            "mean",
        ),
    )
    .sort_values(
        "Concentration Penalty"
    )
)


mean_by_importance = (
    results_df
    .groupby(
        "Importance Preset",
        as_index=False,
    )
    .agg(
        Mean_Sharpe=(
            "Sharpe",
            "mean",
        ),
        Median_Sharpe=(
            "Sharpe",
            "median",
        ),
        Mean_Return=(
            "Strategy Return",
            "mean",
        ),
        Mean_Max_Drawdown=(
            "Max Drawdown",
            "mean",
        ),
    )
    .sort_values(
        "Mean_Sharpe",
        ascending=False,
    )
)


########################################
# Display Helpers
########################################

def display_result_table(
    dataframe,
):

    display_df = (
        dataframe.copy()
    )

    percentage_columns = [
        "Strategy Return",
        "S&P 500 Return",
        "Relative Performance",
        "Annualised Mean",
        "Annualised Volatility",
        "Max Drawdown",
        "S&P Max Drawdown",
        "Max Weight",
    ]

    for column in percentage_columns:

        if column in display_df.columns:

            display_df[
                column
            ] = (
                display_df[
                    column
                ]
                .map(
                    lambda value:
                        f"{value:.2%}"
                        if pd.notna(value)
                        else "NaN"
                )
            )

    if "Sharpe" in display_df.columns:

        display_df[
            "Sharpe"
        ] = (
            display_df[
                "Sharpe"
            ]
            .map(
                lambda value:
                    f"{value:.3f}"
                    if pd.notna(value)
                    else "NaN"
            )
        )

    return display_df


def display_summary_table(
    dataframe,
):

    display_df = (
        dataframe.copy()
    )

    for column in display_df.columns:

        if (
            "Return" in column
            or "Relative" in column
            or "Drawdown" in column
            or column
            in {
                "Max Weight",
            }
        ):

            display_df[
                column
            ] = (
                display_df[
                    column
                ]
                .map(
                    lambda value:
                        f"{value:.2%}"
                        if pd.notna(value)
                        else "NaN"
                )
            )

        elif (
            "Sharpe" in column
        ):

            display_df[
                column
            ] = (
                display_df[
                    column
                ]
                .map(
                    lambda value:
                        f"{value:.3f}"
                        if pd.notna(value)
                        else "NaN"
                )
            )

    return display_df


########################################
# Final Overall Summary
########################################

print("\n\n")
print("=" * 110)
print("FINAL PARAMETER SWEEP SUMMARY")
print("=" * 110)

print(
    f"Successful experiments: "
    f"{len(results_df):,}"
)

print(
    f"Failed / skipped experiments: "
    f"{total_possible_experiments - len(results_df):,}"
)


########################################
# Top Results
########################################

top_columns = [
    "Experiment",
    "Universe",
    "Importance Preset",
    "Max Weight",
    "Concentration Penalty",
    "Rebalance Days",
    "Strategy Return",
    "Relative Performance",
    "Annualised Volatility",
    "Sharpe",
    "Max Drawdown",
    "Average Holdings",
]


print("\n")
print("=" * 110)
print(
    f"TOP {min(FINAL_TOP_N, len(results_df))} "
    "RESULTS BY SHARPE"
)
print("=" * 110)

print(
    display_result_table(
        results_df[
            top_columns
        ]
        .head(
            FINAL_TOP_N
        )
    )
    .to_string(
        index=False
    )
)


########################################
# Best Per Universe
########################################

print("\n")
print("=" * 110)
print("BEST RESULT PER UNIVERSE")
print("=" * 110)

print(
    display_result_table(
        best_by_universe[
            [
                "Universe",
                "Importance Preset",
                "Max Weight",
                "Concentration Penalty",
                "Rebalance Days",
                "Strategy Return",
                "Relative Performance",
                "Sharpe",
                "Max Drawdown",
                "Average Holdings",
            ]
        ]
    )
    .to_string(
        index=False
    )
)


########################################
# Average Per Universe
########################################

print("\n")
print("=" * 110)
print("AVERAGE ROBUSTNESS BY UNIVERSE")
print("=" * 110)

print(
    display_summary_table(
        mean_by_universe
    )
    .to_string(
        index=False
    )
)


########################################
# Max Weight Robustness
########################################

print("\n")
print("=" * 110)
print("AVERAGE RESULT BY MAX WEIGHT")
print("=" * 110)

print(
    display_summary_table(
        mean_by_max_weight
    )
    .to_string(
        index=False
    )
)


########################################
# Rebalance Robustness
########################################

print("\n")
print("=" * 110)
print("AVERAGE RESULT BY REBALANCE FREQUENCY")
print("=" * 110)

print(
    display_summary_table(
        mean_by_rebalance
    )
    .to_string(
        index=False
    )
)


########################################
# Concentration Penalty Robustness
########################################

print("\n")
print("=" * 110)
print("AVERAGE RESULT BY CONCENTRATION PENALTY")
print("=" * 110)

print(
    display_summary_table(
        mean_by_penalty
    )
    .to_string(
        index=False
    )
)


########################################
# Importance Robustness
########################################

print("\n")
print("=" * 110)
print("AVERAGE RESULT BY IMPORTANCE PRESET")
print("=" * 110)

print(
    display_summary_table(
        mean_by_importance
    )
    .to_string(
        index=False
    )
)


########################################
# Save Results
########################################

DATA_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


results_df.to_csv(
    RESULTS_CSV,
    index=False,
)


with sqlite3.connect(
    RESULTS_DB
) as connection:

    results_df.to_sql(
        "All_Results",
        connection,
        if_exists="replace",
        index=False,
    )

    best_by_universe.to_sql(
        "Best_By_Universe",
        connection,
        if_exists="replace",
        index=False,
    )

    mean_by_universe.to_sql(
        "Mean_By_Universe",
        connection,
        if_exists="replace",
        index=False,
    )

    mean_by_max_weight.to_sql(
        "Mean_By_Max_Weight",
        connection,
        if_exists="replace",
        index=False,
    )

    mean_by_rebalance.to_sql(
        "Mean_By_Rebalance",
        connection,
        if_exists="replace",
        index=False,
    )

    mean_by_penalty.to_sql(
        "Mean_By_Concentration_Penalty",
        connection,
        if_exists="replace",
        index=False,
    )

    mean_by_importance.to_sql(
        "Mean_By_Importance",
        connection,
        if_exists="replace",
        index=False,
    )


print("\n")
print("=" * 110)
print("RESULT FILES")
print("=" * 110)

print(
    f"CSV: "
    f"{RESULTS_CSV}"
)

print(
    f"DB:  "
    f"{RESULTS_DB}"
)


########################################
# Best Overall Result
########################################

print("\n")
print("=" * 110)
print("BEST OVERALL RESULT")
print("=" * 110)

print(
    display_result_table(
        pd.DataFrame(
            [
                best_result_row
            ]
        )[
            [
                "Universe",
                "Importance Preset",
                "Alpha Importance",
                "Volatility Importance",
                "Downside Importance",
                "Max Weight",
                "Concentration Penalty",
                "Rebalance Days",
                "Strategy Return",
                "S&P 500 Return",
                "Relative Performance",
                "Annualised Mean",
                "Annualised Volatility",
                "Sharpe",
                "Max Drawdown",
                "Average Holdings",
                "Position Changes",
                "Last Holdings",
            ]
        ]
    )
    .to_string(
        index=False
    )
)


########################################
# Plot Best Overall Result
########################################

if (
    PLOT_BEST_RESULT
    and best_backtest is not None
):

    best_backtest[
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
            "Best Parameter-Sweep Result | "
            f"{best_result_row['Universe']} | "
            f"Sharpe "
            f"{best_result_row['Sharpe']:.3f}"
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


    best_backtest[
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
            "Best Parameter-Sweep Result | "
            "Drawdown"
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
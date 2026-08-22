import sqlite3
import pandas as pd

from models import *

from sklearn.preprocessing import StandardScaler

from main_package import *

import json
import numpy as np

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)

FEATURE_DATABASE_PATH = (
    "/Users/sam/Progressive-Projects/Projects/"
    "Equity Selector/data/Features_Targets_Data.db"
)

STOCK_TYPE = (
    "High Liquidity 30"
    #"Medium Liquidity 30"
    #"Lower Liquidity 30"
    #"Intraday Higher Liquidity 30"
    #"Intraday Medium Liquidity 30"
    #"Sector Spread 30"
    #"Liquidity Barbell 30"
    #"Institutional Liquidity 60"
    #"Medium Small Liquidity 60"
    #"Medium Large Liquidity 60"
    #"All Liquidity 90"
)

VALIDATION_DATABASE_PATH = (
    "/Users/sam/Progressive-Projects/Projects/"
    "Equity Selector/data/Validation_Model_Fits/"
    f"{STOCK_TYPE.replace(' ', '_')}.db"
)

SELECTED_FEATURES_PATH = (
    "/Users/sam/Progressive-Projects/Projects/"
    "Equity Selector/data/Selected_Features.txt"
)


def quote_sql_identifier(identifier):

    return (
        '"'
        + str(identifier).replace('"', '""')
        + '"'
    )


file = open(SELECTED_FEATURES_PATH, 'r')
selected_features = file.read()
file.close()

if STOCK_TYPE == "High Liquidity 30":
    stock_type_index = 0
elif STOCK_TYPE == "Medium Liquidity 30":
    stock_type_index = 1
elif STOCK_TYPE == "Lower Liquidity 30":
    stock_type_index = 2
elif STOCK_TYPE == "Sector Spread 30":
    stock_type_index = 3
elif STOCK_TYPE == "Intraday Higher Liquidity 30":
    stock_type_index = 4
elif STOCK_TYPE == "Intraday Medium Liquidity 30":
    stock_type_index = 5
elif STOCK_TYPE == "Liquidity Barbell 30":
    stock_type_index = 6
elif STOCK_TYPE == "Institutional Liquidity 60":
    stock_type_index = 7
elif STOCK_TYPE == "Medium Small Liquidity 60":
    stock_type_index = 8
elif STOCK_TYPE == "Medium Large Liquidity 60":
    stock_type_index = 9
elif STOCK_TYPE == "All Liquidity 90":
    stock_type_index = 10
else:
    raise ValueError(
        f"Unknown STOCK_TYPE: {STOCK_TYPE}"
    )

import ast

selected_features = selected_features.split('\n')
selected_features = selected_features[stock_type_index]
selected_features = ast.literal_eval(selected_features)


############################################################
# LOAD TEST-ELIGIBLE VALIDATION MODELS
############################################################

def testing_eligible_mask(series):

    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)

    if pd.api.types.is_numeric_dtype(series):
        return (
            pd.to_numeric(
                series,
                errors="coerce"
            )
            .fillna(0)
            .eq(1)
        )

    return (
        series
        .astype(str)
        .str.strip()
        .str.lower()
        .isin([
            "true",
            "1",
            "yes"
        ])
    )


def load_test_eligible_models(
    selected_features,
    validation_database_path
):

    eligible_by_target = {}

    with sqlite3.connect(
        validation_database_path
    ) as validation_connection:

        table_names = pd.read_sql_query(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
            """,
            validation_connection
        )["name"].tolist()

        table_names = set(
            table_names
        )

        for target in selected_features.keys():

            if target not in table_names:
                continue

            table_info = pd.read_sql_query(
                f"PRAGMA table_info({quote_sql_identifier(target)})",
                validation_connection
            )

            table_columns = set(
                table_info["name"].tolist()
            )

            if "Testing Eligible" not in table_columns:
                continue

            leaderboard = pd.read_sql_query(
                f"SELECT * FROM {quote_sql_identifier(target)}",
                validation_connection
            )

            if leaderboard.empty:
                continue

            eligible = leaderboard[
                testing_eligible_mask(
                    leaderboard[
                        "Testing Eligible"
                    ]
                )
            ].copy()

            if eligible.empty:
                continue

            if "Rank" in eligible.columns:
                eligible = eligible.sort_values(
                    "Rank",
                    ascending=True,
                    na_position="last"
                )

            duplicate_columns = [
                column
                for column in [
                    "Model",
                    "Parameters"
                ]
                if column in eligible.columns
            ]

            if len(duplicate_columns) > 0:
                eligible = eligible.drop_duplicates(
                    subset=duplicate_columns,
                    keep="first"
                )

            eligible_by_target[target] = (
                eligible.reset_index(
                    drop=True
                )
            )

    return eligible_by_target


eligible_models_by_target = (
    load_test_eligible_models(
        selected_features=selected_features,
        validation_database_path=(
            VALIDATION_DATABASE_PATH
        )
    )
)

targets = [
    target
    for target in selected_features.keys()
    if target in eligible_models_by_target
]

logger.info(
    "Validation database: %s",
    VALIDATION_DATABASE_PATH
)

logger.info(
    "Test-eligible targets: %d / %d",
    len(targets),
    len(selected_features)
)


############################################################
# MEMORY-SAFE SOURCE DATABASE
#
# Only the columns required for the current target are read.
# The full Features_Targets_Data table is never loaded.
############################################################

data_connection = sqlite3.connect(
    FEATURE_DATABASE_PATH
)

data_connection.execute(
    "PRAGMA query_only = ON"
)

source_table_info = pd.read_sql_query(
    f"PRAGMA table_info({quote_sql_identifier(STOCK_TYPE)})",
    data_connection
)

if source_table_info.empty:
    raise ValueError(
        f"Source table {STOCK_TYPE!r} does not exist "
        f"in {FEATURE_DATABASE_PATH}"
    )

SOURCE_TABLE_COLUMNS = set(
    source_table_info["name"].tolist()
)

logger.info(
    "%s | Source table ready | %d columns | full table will not be loaded",
    STOCK_TYPE,
    len(SOURCE_TABLE_COLUMNS)
)

RIDGE_ALPHAS = [
    1e-5,
    3e-5,
    1e-4,
    3e-4,
    1e-3,
    3e-3,
    1e-2,
    3e-2,
    0.1,
    0.3,
    1,
    3,
    10,
    30,
    100,
    300,
    1000,
    3000,
    10000,
]


SPARSE_ALPHAS = [
    1e-8,
    3e-8,
    1e-7,
    3e-7,
    1e-6,
    3e-6,
    1e-5,
    3e-5,
    1e-4,
    3e-4,
    1e-3,
    3e-3,
    1e-2,
    3e-2,
    0.1,
    0.3,
    1,
    3,
    10,
]


C_VALUES = [
    1e-5,
    3e-5,
    1e-4,
    3e-4,
    1e-3,
    3e-3,
    1e-2,
    3e-2,
    0.1,
    0.3,
    1,
    3,
    10,
    30,
    100,
    300,
    1000,
]


L1_RATIOS = [
    0.01,
    0.05,
    0.10,
    0.25,
    0.50,
    0.75,
    0.90,
    0.95,
    0.99,
]


LEARNING_RATES = [
    0.005,
    0.01,
    0.03,
    0.05,
    0.10,
    0.20,
]


CLASS_WEIGHTS = [
    None,
    "balanced",
]


########################################
# Hist Gradient Boosting
########################################

HIST_GRADIENT_PARAMS = {

    "learning_rate": LEARNING_RATES,

    "max_iter": [
        100,
        200,
        300,
        600,
        1000,
        1500,
    ],

    "max_leaf_nodes": [
        7,
        15,
        31,
        63,
        127,
    ],

    "max_depth": [
        None,
        3,
        5,
        7,
        10,
    ],

    "min_samples_leaf": [
        5,
        10,
        20,
        50,
        100,
        200,
    ],

    "l2_regularization": [
        0,
        1e-4,
        1e-3,
        1e-2,
        0.1,
        1,
        10,
        100,
    ],
}


########################################
# Gradient Boosting
########################################

GRADIENT_BOOSTING_PARAMS = {

    "learning_rate": LEARNING_RATES,

    "n_estimators": [
        100,
        200,
        300,
        600,
        1000,
    ],

    "max_depth": [
        2,
        3,
        4,
        5,
        8,
    ],

    "min_samples_leaf": [
        5,
        10,
        20,
        50,
        100,
    ],

    "subsample": [
        0.5,
        0.6,
        0.8,
        1.0,
    ],

    "max_features": [
        None,
        "sqrt",
        0.3,
        0.5,
        0.8,
    ],
}


########################################
# Random Forest
########################################

RANDOM_FOREST_PARAMS = {

    "n_estimators": [
        200,
        500,
        1000,
        1500,
    ],

    "max_depth": [
        None,
        5,
        10,
        15,
        20,
        30,
    ],

    "min_samples_leaf": [
        1,
        2,
        5,
        10,
        20,
        50,
        100,
    ],

    "min_samples_split": [
        2,
        5,
        10,
        20,
        50,
    ],

    "max_features": [
        "sqrt",
        0.2,
        0.3,
        0.5,
        0.75,
        1.0,
    ],

    "bootstrap": [
        True,
        False,
    ],
}


########################################
# XGBoost
########################################

XGBOOST_PARAMS = {

    "n_estimators": [
        200,
        300,
        500,
        750,
        1000,
        1500,
    ],

    "learning_rate": LEARNING_RATES,

    "max_depth": [
        2,
        3,
        4,
        5,
        7,
        10,
    ],

    "min_child_weight": [
        1,
        2,
        3,
        5,
        10,
        20,
        50,
    ],

    "subsample": [
        0.5,
        0.6,
        0.8,
        1.0,
    ],

    "colsample_bytree": [
        0.4,
        0.5,
        0.75,
        1.0,
    ],

    "gamma": [
        0,
        0.001,
        0.01,
        0.1,
        0.5,
        1,
        5,
    ],

    "reg_alpha": [
        0,
        1e-5,
        1e-4,
        1e-3,
        1e-2,
        0.1,
        1,
        10,
    ],

    "reg_lambda": [
        0,
        0.01,
        0.1,
        1,
        10,
        100,
    ],
}


########################################
# LightGBM
########################################

LIGHTGBM_PARAMS = {

    "n_estimators": [
        200,
        300,
        500,
        750,
        1000,
        1500,
    ],

    "learning_rate": LEARNING_RATES,

    "num_leaves": [
        7,
        15,
        31,
        63,
        127,
        255,
    ],

    "max_depth": [
        -1,
        3,
        5,
        8,
        12,
        16,
    ],

    "min_child_samples": [
        5,
        10,
        20,
        50,
        100,
        200,
    ],

    "subsample": [
        0.5,
        0.6,
        0.8,
        1.0,
    ],

    "colsample_bytree": [
        0.4,
        0.5,
        0.75,
        1.0,
    ],

    "reg_alpha": [
        0,
        1e-5,
        1e-4,
        1e-3,
        1e-2,
        0.1,
        1,
        10,
    ],

    "reg_lambda": [
        0,
        0.01,
        0.1,
        1,
        10,
        100,
    ],
}


########################################
# MLP
#
# Currently unused because MLP models
# themselves are commented out below.
########################################

MLP_PARAMS = {

    "hidden_layer_sizes": [
        (32,),
        (64,),
        (128,),
        (256,),
        (64, 32),
        (128, 64),
        (256, 128),
        (128, 64, 32),
        (256, 128, 64),
    ],

    "activation": [
        "relu",
        "tanh",
    ],

    "alpha": [
        1e-7,
        1e-6,
        1e-5,
        1e-4,
        1e-3,
        1e-2,
        0.1,
    ],

    "learning_rate_init": [
        1e-5,
        3e-5,
        1e-4,
        3e-4,
        1e-3,
        3e-3,
        1e-2,
    ],

    "batch_size": [
        32,
        64,
        128,
        256,
        512,
        "auto",
    ],
}


########################################
# kNN
########################################

KNN_PARAMS = {

    "n_neighbors": [
        3,
        5,
        10,
        20,
        40,
        80,
        150,
        250,
    ],

    "weights": [
        "uniform",
        "distance",
    ],

    "p": [
        1,
        2,
    ],
}


########################################
########################################
# CONTINUOUS MODELS
########################################
########################################

CONTINUOUS_MODELS = [

    ########################################
    # Baselines / Linear
    ########################################

    {
        "name": "Mean Baseline",
        "function": "fit_mean_baseline",
        "scaled": False,
        "search": "grid",
        "params": {},
    },

    {
        "name": "OLS",
        "function": "fit_ols",
        "scaled": True,
        "search": "grid",
        "params": {},
    },

    {
        "name": "Ridge",
        "function": "fit_ridge",
        "scaled": True,
        "search": "grid",
        "params": {
            "alpha": RIDGE_ALPHAS,
        },
    },

    {
        "name": "Lasso",
        "function": "fit_lasso",
        "scaled": True,
        "search": "grid",
        "params": {
            "alpha": SPARSE_ALPHAS,
        },
    },

    {
        "name": "Elastic Net",
        "function": "fit_elastic_net",
        "scaled": True,
        "search": "random",
        "n_iter": 20,
        # Later:
        "n_iter": 75,
        "params": {
            "alpha": SPARSE_ALPHAS,
            "l1_ratio": L1_RATIOS,
        },
    },


    ########################################
    # Huber
    # Uncomment later
    ########################################

    {
        "name": "Huber",
        "function": "fit_huber",
        "scaled": True,
        "search": "random",
        "n_iter": 60,
        "params": {
    
            "epsilon": [
                1.05,
                1.15,
                1.25,
                1.35,
                1.50,
                1.75,
                2.00,
                2.50,
            ],
    
            "alpha": [
                0,
                1e-7,
                1e-6,
                1e-5,
                1e-4,
                1e-3,
                1e-2,
                0.1,
                1,
            ],
        },
    },


    ########################################
    # Main Tree Models
    ########################################

    {
       "name": "Hist Gradient Boosting",
       "function": "fit_hist_gradient_boosting_regressor",
       "scaled": False,
       "search": "random",
       "n_iter": 20,
       # Later:
       "n_iter": 75,
       "params": HIST_GRADIENT_PARAMS,
    },


    ########################################
    # Standard Gradient Boosting
    # Uncomment later
    ########################################

    {
        "name": "Gradient Boosting",
        "function": "fit_gradient_boosting_regressor",
        "scaled": False,
        "search": "random",
        "n_iter": 60,
        "params": GRADIENT_BOOSTING_PARAMS,
    },


    {
        "name": "Random Forest",
        "function": "fit_random_forest_regressor",
        "scaled": False,
        "search": "random",
        "n_iter": 20,
        # Later:
        "n_iter": 75,
        "params": RANDOM_FOREST_PARAMS,
    },

    {
        "name": "XGBoost",
        "function": "fit_xgboost_regressor",
        "scaled": False,
        "search": "random",
        "n_iter": 25,
        # Later:
        "n_iter": 100,
        "params": XGBOOST_PARAMS,
    },

    {
        "name": "LightGBM",
        "function": "fit_lightgbm_regressor",
        "scaled": False,
        "search": "random",
        "n_iter": 25,
        # Later:
        "n_iter": 100,
        "params": LIGHTGBM_PARAMS,
    },


    ########################################
    # SVR
    # Uncomment later
    ########################################

    {
        "name": "SVR",
        "function": "fit_svr",
        "scaled": True,
        "search": "random",
        "n_iter": 75,
        "params": [
    
            {
                "kernel": ["linear"],
                "C": C_VALUES,
    
                "epsilon": [
                    1e-4,
                    1e-3,
                    1e-2,
                    0.05,
                    0.1,
                    0.25,
                    0.5,
                ],
            },
    
            {
                "kernel": ["rbf"],
                "C": C_VALUES,
    
                "epsilon": [
                    1e-4,
                    1e-3,
                    1e-2,
                    0.05,
                    0.1,
                    0.25,
                    0.5,
                ],
    
                "gamma": [
                    "scale",
                    "auto",
                    1e-4,
                    1e-3,
                    1e-2,
                    0.1,
                    1,
                ],
            },
        ],
    },


    ########################################
    # kNN
    # Uncomment later
    ########################################

    {
        "name": "kNN",
        "function": "fit_knn_regressor",
        "scaled": True,
        "search": "grid",
        "params": KNN_PARAMS,
    },


    ########################################
    # MLP
    # Uncomment much later
    ########################################

    {
        "name": "MLP",
        "function": "fit_mlp_regressor",
        "scaled": True,
        "search": "random",
        "n_iter": 75,
        "params": MLP_PARAMS,
    },
]


########################################
########################################
# BINARY MODELS
########################################
########################################

BINARY_MODELS = [

    ########################################
    # Baseline / Logistic Models
    ########################################

    {
        "name": "Binary Baseline",
        "function": "fit_binary_baseline",
        "scaled": False,
        "search": "grid",
        "params": {},
    },

    {
        "name": "Logistic Regression",
        "function": "fit_logistic_regression",
        "scaled": True,
        "search": "grid",
        "params": {
            "class_weight": CLASS_WEIGHTS,
        },
    },

    {
        "name": "L2 Logistic Regression",
        "function": "fit_l2_logistic_regression",
        "scaled": True,
        "search": "grid",
        "params": {
            "C": C_VALUES,
            "class_weight": CLASS_WEIGHTS,
        },
    },

    {
        "name": "L1 Logistic Regression",
        "function": "fit_l1_logistic_regression",
        "scaled": True,
        "search": "grid",
        "params": {
            "C": C_VALUES,
            "class_weight": CLASS_WEIGHTS,
        },
    },

    {
        "name": "Elastic Net Logistic Regression",
        "function": "fit_elastic_net_logistic_regression",
        "scaled": True,
        "search": "random",
        "n_iter": 25,
        # Later:
        "n_iter": 100,
        "params": {
            "C": C_VALUES,
            "l1_ratio": L1_RATIOS,
            "class_weight": CLASS_WEIGHTS,
        },
    },


    ########################################
    # Tree Models
    ########################################

    {
       "name": "Hist Gradient Boosting",
       "function": "fit_hist_gradient_boosting_classifier",
       "scaled": False,
       "search": "random",
       "n_iter": 20,
       # Later:
       "n_iter": 75,
       "params": {
           **HIST_GRADIENT_PARAMS,
           "class_weight": CLASS_WEIGHTS,
       },
    },


    ########################################
    # Standard Gradient Boosting
    # Uncomment later
    ########################################

    {
        "name": "Gradient Boosting",
        "function": "fit_gradient_boosting_classifier",
        "scaled": False,
        "search": "random",
        "n_iter": 60,
        "params": GRADIENT_BOOSTING_PARAMS,
    },


    {
        "name": "Random Forest",
        "function": "fit_random_forest_classifier",
        "scaled": False,
        "search": "random",
        "n_iter": 20,
        # Later:
        "n_iter": 75,
        "params": {
            **RANDOM_FOREST_PARAMS,
            "class_weight": CLASS_WEIGHTS,
        },
    },

    {
       "name": "XGBoost",
       "function": "fit_xgboost_classifier",
       "scaled": False,
       "search": "random",
       "n_iter": 25,
       # Later:
       "n_iter": 100,
       "params": {
           **XGBOOST_PARAMS,
           "class_weight": CLASS_WEIGHTS,
       },
    },

    {
        "name": "LightGBM",
        "function": "fit_lightgbm_classifier",
        "scaled": False,
        "search": "random",
        "n_iter": 25,
        # Later:
        "n_iter": 100,
        "params": {
            **LIGHTGBM_PARAMS,
            "class_weight": CLASS_WEIGHTS,
        },
    },


    ########################################
    # SVM
    # Uncomment later
    ########################################

    {
        "name": "SVM",
        "function": "fit_svm_classifier",
        "scaled": True,
        "search": "random",
        "n_iter": 75,
        "params": [
    
            {
                "kernel": ["linear"],
                "C": C_VALUES,
                "class_weight": CLASS_WEIGHTS,
            },
    
            {
                "kernel": ["rbf"],
                "C": C_VALUES,
    
                "gamma": [
                    "scale",
                    "auto",
                    1e-4,
                    1e-3,
                    1e-2,
                    0.1,
                    1,
                ],
    
                "class_weight": CLASS_WEIGHTS,
            },
        ],
    },


    ########################################
    # kNN
    # Uncomment later
    ########################################

    {
        "name": "kNN",
        "function": "fit_knn_classifier",
        "scaled": True,
        "search": "grid",
        "params": KNN_PARAMS,
    },


    ########################################
    # Naive Bayes
    #
    # Cheap enough that I would leave this
    # active initially.
    ########################################

    {
        "name": "Naive Bayes",
        "function": "fit_naive_bayes",
        "scaled": False,
        "search": "grid",
        "params": {

            "var_smoothing": [
                1e-13,
                1e-12,
                1e-11,
                1e-10,
                1e-9,
                1e-8,
                1e-7,
                1e-6,
                1e-5,
            ],
        },
    },


    ########################################
    # MLP
    # Uncomment much later
    ########################################

    {
        "name": "MLP",
        "function": "fit_mlp_classifier",
        "scaled": True,
        "search": "random",
        "n_iter": 75,
        "params": MLP_PARAMS,
    },
]


########################################
########################################
# MULTICLASS MODELS
########################################
########################################

MULTICLASS_MODELS = [

    ########################################
    # Baseline / Logistic Models
    ########################################

    {
        "name": "Multiclass Baseline",
        "function": "fit_multiclass_baseline",
        "scaled": False,
        "search": "grid",
        "params": {},
    },

    {
        "name": "Multinomial Logistic Regression",
        "function": "fit_multinomial_logistic_regression",
        "scaled": True,
        "search": "grid",
        "params": {
            "class_weight": CLASS_WEIGHTS,
        },
    },

    {
        "name": "L2 Multinomial Logistic Regression",
        "function": "fit_l2_multinomial_logistic_regression",
        "scaled": True,
        "search": "grid",
        "params": {
            "C": C_VALUES,
            "class_weight": CLASS_WEIGHTS,
        },
    },

    {
        "name": "L1 Multinomial Logistic Regression",
        "function": "fit_l1_multinomial_logistic_regression",
        "scaled": True,
        "search": "grid",
        "params": {
            "C": C_VALUES,
            "class_weight": CLASS_WEIGHTS,
        },
    },

    {
        "name": "Elastic Net Multinomial Logistic Regression",
        "function": "fit_elastic_net_multinomial_logistic_regression",
        "scaled": True,
        "search": "random",
        "n_iter": 25,
        # Later:
        "n_iter": 100,
        "params": {
            "C": C_VALUES,
            "l1_ratio": L1_RATIOS,
            "class_weight": CLASS_WEIGHTS,
        },
    },


    ########################################
    # LDA
    # Uncomment later
    ########################################

    {
        "name": "LDA",
        "function": "fit_lda",
        "scaled": True,
        "search": "grid",
        "params": [
    
            {
                "solver": ["svd"],
            },
    
            {
                "solver": [
                    "lsqr",
                    "eigen",
                ],
    
                "shrinkage": [
                    None,
                    "auto",
                    0.05,
                    0.1,
                    0.25,
                    0.5,
                    0.75,
                    0.9,
                    0.95,
                ],
            },
        ],
    },


    ########################################
    # QDA
    # Uncomment later
    ########################################

    {
        "name": "QDA",
        "function": "fit_qda",
        "scaled": True,
        "search": "grid",
        "params": {
    
            "reg_param": [
                0,
                0.0001,
                0.001,
                0.01,
                0.05,
                0.1,
                0.25,
                0.5,
                0.75,
                1.0,
            ],
        },
    },


    ########################################
    # Tree Models
    ########################################

    {
       "name": "Hist Gradient Boosting",
       "function": "fit_hist_gradient_boosting_multiclass",
       "scaled": False,
       "search": "random",
       "n_iter": 20,
       # Later:
       "n_iter": 75,
       "params": {
           **HIST_GRADIENT_PARAMS,
           "class_weight": CLASS_WEIGHTS,
       },
    },


    ########################################
    # Standard Gradient Boosting
    # Uncomment later
    ########################################

    {
        "name": "Gradient Boosting",
        "function": "fit_gradient_boosting_multiclass",
        "scaled": False,
        "search": "random",
        "n_iter": 60,
        "params": GRADIENT_BOOSTING_PARAMS,
    },


    {
        "name": "Random Forest",
        "function": "fit_random_forest_multiclass",
        "scaled": False,
        "search": "random",
        "n_iter": 20,
        # Later:
        "n_iter": 75,
        "params": {
            **RANDOM_FOREST_PARAMS,
            "class_weight": CLASS_WEIGHTS,
        },
    },

    {
        "name": "XGBoost",
        "function": "fit_xgboost_multiclass",
        "scaled": False,
        "search": "random",
        "n_iter": 25,
        # Later:
        "n_iter": 100,
        "params": {
            **XGBOOST_PARAMS,
            "class_weight": CLASS_WEIGHTS,
        },
    },

    {
        "name": "LightGBM",
        "function": "fit_lightgbm_multiclass",
        "scaled": False,
        "search": "random",
        "n_iter": 25,
        # Later:
        "n_iter": 100,
        "params": {
            **LIGHTGBM_PARAMS,
            "class_weight": CLASS_WEIGHTS,
        },
    },


    ########################################
    # SVM
    # Uncomment later
    ########################################

    {
        "name": "SVM",
        "function": "fit_svm_multiclass",
        "scaled": True,
        "search": "random",
        "n_iter": 75,
        "params": [
    
            {
                "kernel": ["linear"],
                "C": C_VALUES,
                "class_weight": CLASS_WEIGHTS,
            },
    
            {
                "kernel": ["rbf"],
                "C": C_VALUES,
    
                "gamma": [
                    "scale",
                    "auto",
                    1e-4,
                    1e-3,
                    1e-2,
                    0.1,
                    1,
                ],
    
                "class_weight": CLASS_WEIGHTS,
            },
        ],
    },


    ########################################
    # MLP
    # Uncomment much later
    ########################################

    {
        "name": "MLP",
        "function": "fit_mlp_multiclass",
        "scaled": True,
        "search": "random",
        "n_iter": 75,
        "params": MLP_PARAMS,
    },


    ########################################
    # Ordinal Regression
    #
    # Relatively cheap, so keep active.
    ########################################

    {
        "name": "Ordinal Regression",
        "function": "fit_ordinal_regression",
        "scaled": True,
        "search": "grid",
        "params": {

            "alpha": [
                1e-5,
                3e-5,
                1e-4,
                3e-4,
                1e-3,
                3e-3,
                1e-2,
                3e-2,
                0.1,
                0.3,
                1,
                3,
                10,
                30,
                100,
            ],
        },
    },
]

def get_model_function(function_name):

    function = globals().get(function_name)

    if function is None:

        raise NameError(
            f"{function_name} has not been imported yet."
        )

    return function


def purge_training_data(train_df, purge_days):

    if purge_days <= 0:
        return train_df.copy()


    dates = np.sort(
        train_df["Date"].unique()
    )


    ########################################
    # Not Enough Training History
    ########################################

    if len(dates) <= purge_days:
        return train_df.iloc[0:0].copy()


    ########################################
    # Remove Last purge_days Trading Dates
    ########################################

    purge_start_date = dates[-purge_days]

    return train_df[
        train_df["Date"] < purge_start_date
    ].copy()


def final_test(

    train_df,
    validation_df,
    test_df,

    features,

    target,
    model_name,
    parameters
):

    ########################################
    # Target Type
    ########################################

    _type = final_target_type(
        target
    )


    logger.info(
        "%s | Final Test | Type=%s | Model=%s",
        target,
        _type,
        model_name
    )


    ########################################
    # Find Existing Model Config
    ########################################

    model_config = get_selected_model_config(
        model_name=model_name,
        target_type=_type
    )


    ########################################
    # Get Existing Fit Function
    ########################################

    fit_function = get_model_function(
        model_config["function"]
    )


    logger.info(
        "%s | Function=%s | Scaled=%s",
        target,
        model_config["function"],
        model_config["scaled"]
    )


    ########################################
    # Final Training Data
    #
    # Train + Validation
    ########################################

    final_train_df = pd.concat(
        [
            train_df,
            validation_df
        ],
        ignore_index=True
    )


    ########################################
    # Purge
    ########################################

    purge_days = target_purge_days(
        target
    )

    rows_before_purge = len(
        final_train_df
    )

    final_train_df = purge_training_data(
        final_train_df,
        purge_days
    )


    logger.info(
        "%s | Train rows=%d -> %d after purge | "
        "Test rows=%d | Purge=%d",
        target,
        rows_before_purge,
        len(final_train_df),
        len(test_df),
        purge_days
    )


    ########################################
    # X / Y
    ########################################

    x_train = final_train_df[
        features
    ]

    y_train = final_train_df[
        target
    ]

    x_test = test_df[
        features
    ]

    y_test = test_df[
        target
    ]


    ########################################
    # Binary Target Cleaning
    ########################################

    if _type == "binary":

        y_train = clean_binary_target(
            y_train,
            target
        )

        y_test = clean_binary_target(
            y_test,
            target
        )


    ########################################
    # Log Classes
    ########################################

    if _type in (
        "binary",
        "multiclass"
    ):

        train_classes = np.sort(
            y_train.dropna().unique()
        )

        test_classes = np.sort(
            y_test.dropna().unique()
        )

        logger.info(
            "%s | Train classes=%s | Test classes=%s",
            target,
            train_classes,
            test_classes
        )


    ########################################
    # Scaling
    ########################################

    if model_config["scaled"]:

        scaler = StandardScaler()

        x_train = pd.DataFrame(

            scaler.fit_transform(
                x_train
            ),

            columns=features,
            index=x_train.index
        )

        x_test = pd.DataFrame(

            scaler.transform(
                x_test
            ),

            columns=features,
            index=x_test.index
        )


    ########################################
    # Fit + Test
    ########################################

    logger.info(
        "%s | Fitting final model",
        target
    )

    result = fit_function(

        x_train,
        y_train,

        x_test,
        y_test,

        **parameters
    )


    if result is None:
        result = {}


    ########################################
    # Clean Result
    ########################################

    clean_result = clean_final_result(

        result=result,

        target=target,
        target_type=_type,

        model_name=model_name,
        parameters=parameters
    )


    logger.info(
        "%s | Complete | %s",
        target,
        {
            key: value
            for key, value in clean_result.items()
            if key not in [
                "Parameters"
            ]
        }
    )


    return clean_result


def get_selected_model_config(
    model_name,
    target_type
):

    models = get_models(
        target_type
    )

    for model_config in models:

        if model_config["name"] == model_name:
            return model_config

    raise ValueError(
        f"{model_name} not found for "
        f"target type {target_type}"
    )


def json_default(value):

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    if isinstance(value, np.ndarray):
        return value.tolist()

    return str(value)


def parameters_to_json(parameters):

    return json.dumps(
        parameters,
        sort_keys=True,
        default=json_default
    )

def clean_binary_target(y, target):

    y = y.copy()


    ########################################
    # Future Direction
    #
    # Positive     ->  1
    # Zero/Negative -> -1
    ########################################

    if target.startswith("Future Direction"):

        y = y.where(
            y > 0,
            -1
        )

        y = y.where(
            y <= 0,
            1
        )


    ########################################
    # Safety Check
    ########################################

    classes = np.sort(
        pd.Series(y)
        .dropna()
        .unique()
    )

    if len(classes) > 2:

        raise ValueError(
            f"{target} is defined as binary "
            f"but contains classes {classes}"
        )


    return y

def final_target_type(target):

    ########################################
    # Multiclass
    ########################################

    if (
        target.startswith("Three Class Direction")
        or target.startswith("Barrier")
        or target.startswith("Volatility Barrier")
    ):
        return "multiclass"


    ########################################
    # Binary
    ########################################

    if (
        target.startswith("Future Direction")
        or target.startswith("Future Return Above")
        or target.startswith("Top 20 Percent Future Return")
        or target.startswith("Top 25 Percent Future Return")
    ):
        return "binary"


    ########################################
    # Continuous
    ########################################

    return "continuous"


def get_models(target_type):

    if target_type == "continuous":
        return CONTINUOUS_MODELS

    if target_type == "binary":
        return BINARY_MODELS

    if target_type == "multiclass":
        return MULTICLASS_MODELS

    raise ValueError(
        f"Unknown target type: {target_type}"
    )


def clean_final_result(
    result,
    target,
    target_type,
    model_name,
    parameters
):

    ########################################
    # Base Information
    ########################################

    clean = {
        "Target": target,
        "Target Type": target_type,
        "Model": model_name,
        "Parameters": parameters_to_json(parameters)
    }


    ########################################
    # Continuous
    ########################################

    if target_type == "continuous":

        metrics = [
            "RMSE",
            "MAE",
            "R2",
            "Rank IC"
        ]


    ########################################
    # Binary
    ########################################

    elif target_type == "binary":

        metrics = [
            "ROC AUC",
            "PR AUC",
            "Log Loss",
            "F1"
        ]


    ########################################
    # Multiclass
    ########################################

    elif target_type == "multiclass":

        metrics = [
            "Macro F1",
            "Balanced Accuracy",
            "Log Loss"
        ]


    else:
        raise ValueError(
            f"Unknown target type: {target_type}"
        )


    ########################################
    # Only Add Metrics That Exist
    ########################################

    for metric in metrics:

        if metric in result:
            clean[metric] = result[metric]


    return clean

def predictability_score(row):

    if row["Target Type"] == "continuous":
        rank_ic = abs(row["Rank IC"])
        r2 = max(row["R2"], 0)

        # Ranking is especially important for financial signals
        return 0.65 * rank_ic + 0.35 * r2

    elif row["Target Type"] == "binary":
        # Convert AUC so 0.5 = no predictive value
        auc_edge = max(row["ROC AUC"] - 0.5, 0) * 2

        return (
            0.50 * auc_edge +
            0.30 * row["PR AUC"] +
            0.20 * row["F1"]
        )

    elif row["Target Type"] == "multiclass":
        return row["Macro F1"]

    return 0


def target_category(target):
    target = target.lower()

    # Downside / loss-risk predictors
    downside_keywords = [
        "downside",
        "adverse",
        "minimum return",
        "drawdown",
        "time to maximum adverse"
    ]

    if any(word in target for word in downside_keywords):
        return "downside"

    # Volatility / magnitude-of-movement predictors
    volatility_keywords = [
        "volatility",
        "variance",
        "absolute return"
    ]

    if any(word in target for word in volatility_keywords):
        return "volatility"

    # Everything return/direction/opportunity related
    return "alpha"


############################################################
# TEST-ELIGIBLE MODEL SELECTION
############################################################

def parse_selected_parameters(value):

    if isinstance(value, dict):
        return value.copy()

    if value is None:
        return {}

    if isinstance(value, float) and np.isnan(value):
        return {}

    if isinstance(value, str):

        stripped = value.strip()

        if stripped == "":
            return {}

        try:
            parsed = json.loads(
                stripped
            )
        except Exception:
            parsed = ast.literal_eval(
                stripped
            )

        if not isinstance(parsed, dict):
            raise ValueError(
                "Stored model parameters must decode "
                "to a dictionary."
            )

        return parsed

    raise TypeError(
        "Unsupported stored parameter type: "
        f"{type(value).__name__}"
    )


def eligible_metric_columns(
    target_type,
    columns
):

    if target_type == "continuous":
        wanted = [
            "R2 Mean",
            "Rank IC Mean",
            "RMSE Mean",
            "Rank IC Std"
        ]

    elif target_type == "binary":
        wanted = [
            "ROC AUC Mean",
            "PR AUC Mean",
            "F1 Mean",
            "ROC AUC Std"
        ]

    elif target_type == "multiclass":
        wanted = [
            "Macro F1 Mean",
            "Balanced Accuracy Mean",
            "Log Loss Mean",
            "Macro F1 Std"
        ]

    else:
        wanted = []

    return [
        column
        for column in wanted
        if column in columns
    ]


def choose_test_eligible_model(
    target,
    eligible_models
):

    if eligible_models.empty:
        raise ValueError(
            f"{target} has no test-eligible models."
        )

    if len(eligible_models) == 1:

        selected = eligible_models.iloc[0]

        model_name = selected[
            "Model"
        ]

        parameters = parse_selected_parameters(
            selected[
                "Parameters"
            ]
        )

        logger.info(
            "%s | One test-eligible model | Selected automatically: %s | %s",
            target,
            model_name,
            parameters
        )

        return (
            model_name,
            parameters
        )

    target_type = final_target_type(
        target
    )

    metric_columns = eligible_metric_columns(
        target_type=target_type,
        columns=eligible_models.columns
    )

    print("\n" + "=" * 100)
    print(
        f"{target} | {len(eligible_models)} TEST-ELIGIBLE MODELS"
    )
    print("=" * 100)

    for option_number, (_, row) in enumerate(
        eligible_models.iterrows(),
        start=1
    ):

        parts = [
            f"[{option_number}]",
            str(row["Model"]),
            f"Parameters={row['Parameters']}"
        ]

        if (
            "Rank" in eligible_models.columns
            and pd.notna(row.get("Rank"))
        ):
            parts.insert(
                1,
                f"Rank={row['Rank']}"
            )

        for metric in metric_columns:

            value = row.get(
                metric
            )

            if pd.notna(value):
                parts.append(
                    f"{metric}={value:.6f}"
                )

        print(
            " | ".join(parts)
        )

    while True:

        choice = input(
            f"Select model for final test [1-{len(eligible_models)}]: "
        ).strip()

        try:
            choice_number = int(
                choice
            )
        except ValueError:
            print(
                "Please enter a number from "
                f"1 to {len(eligible_models)}."
            )
            continue

        if not (
            1
            <= choice_number
            <= len(eligible_models)
        ):
            print(
                "Please enter a number from "
                f"1 to {len(eligible_models)}."
            )
            continue

        selected = eligible_models.iloc[
            choice_number - 1
        ]

        model_name = selected[
            "Model"
        ]

        parameters = parse_selected_parameters(
            selected[
                "Parameters"
            ]
        )

        logger.info(
            "%s | User selected test-eligible model=%s | Parameters=%s",
            target,
            model_name,
            parameters
        )

        return (
            model_name,
            parameters
        )


final_results = []
final_errors = []


logger.info(
    "Starting final testing | %d test-eligible targets",
    len(targets)
)


for target_number, target in enumerate(
    targets,
    start=1
):

    try:

        features = selected_features[
            target
        ]


        logger.info(
            "[%d/%d] %s | Starting | %d features",
            target_number,
            len(targets),
            target,
            len(features)
        )


        ########################################
        # No Features
        ########################################

        if len(features) == 0:

            logger.warning(
                "%s | Skipped | No selected features",
                target
            )

            continue


        ########################################
        # Select Test-Eligible Model
        ########################################

        eligible_models = (
            eligible_models_by_target[
                target
            ]
        )

        model_name, parameters = (
            choose_test_eligible_model(
                target=target,
                eligible_models=eligible_models
            )
        )


        logger.info(
            "%s | Selected model=%s | Parameters=%s",
            target,
            model_name,
            parameters
        )


        ########################################
        # Data
        #
        # Read only Date, Ticker, this target,
        # and this target's selected features.
        ########################################

        columns = (
            [
                "Date",
                "Ticker",
                target
            ]
            + features
        )

        columns = list(
            dict.fromkeys(
                columns
            )
        )

        missing_columns = [
            column
            for column in columns
            if column not in SOURCE_TABLE_COLUMNS
        ]

        if len(missing_columns) > 0:
            raise KeyError(
                f"{target} is missing source columns: "
                f"{missing_columns}"
            )

        sql_columns = ", ".join(
            quote_sql_identifier(
                column
            )
            for column in columns
        )

        target_query = (
            f"SELECT {sql_columns} "
            f"FROM {quote_sql_identifier(STOCK_TYPE)}"
        )

        logger.info(
            "%s | Loading %d/%d source columns",
            target,
            len(columns),
            len(SOURCE_TABLE_COLUMNS)
        )

        current_df = pd.read_sql_query(
            target_query,
            data_connection
        )

        logger.info(
            "%s | Loaded %d rows x %d columns | %.2f MB",
            target,
            len(current_df),
            len(current_df.columns),
            current_df.memory_usage(
                deep=True
            ).sum() / (1024 ** 2)
        )


        rows_before_dropna = len(
            current_df
        )


        current_df = current_df.dropna(
            subset=[
                target,
                *features
            ]
        )


        logger.info(
            "%s | Valid rows=%d/%d",
            target,
            len(current_df),
            rows_before_dropna
        )


        current_df["Date"] = pd.to_datetime(
            current_df["Date"]
        )


        current_df = current_df.sort_values(
            [
                "Date",
                "Ticker"
            ]
        ).reset_index(
            drop=True
        )


        ########################################
        # Original Split
        ########################################

        train_df, validation_df, test_df = (
            train_validation_test_split(
                current_df
            )
        )


        logger.info(
            "%s | Split | Train=%d | Validation=%d | Test=%d",
            target,
            len(train_df),
            len(validation_df),
            len(test_df)
        )


        ########################################
        # Final Test
        ########################################

        result = final_test(

            train_df=train_df,
            validation_df=validation_df,
            test_df=test_df,

            features=features,

            target=target,

            model_name=model_name,
            parameters=parameters
        )


        final_results.append(
            result
        )


    except Exception as error:

        logger.exception(
            "%s | FAILED",
            target
        )

        final_errors.append(
            {
                "Target": target,
                "Error": str(error)
            }
        )


data_connection.close()

logger.info(
    "Source database connection closed"
)


final_results_df = pd.DataFrame(
    final_results
)

final_errors_df = pd.DataFrame(
    final_errors
)


FINAL_RESULTS_DATABASE = (
    "/Users/sam/Progressive-Projects/Projects/"
    "Equity Selector/data/Final_Test_Results.db"
)


with sqlite3.connect(
    FINAL_RESULTS_DATABASE
) as connection:

    final_results_df.to_sql(
        f"Final Test Results {STOCK_TYPE}",
        connection,
        if_exists="replace",
        index=False
    )


    if not final_errors_df.empty:

        final_errors_df.to_sql(
            "Errors",
            connection,
            if_exists="replace",
            index=False
        )


logger.info(
    "Final testing complete | "
    "Successful=%d | Failed=%d",
    len(final_results),
    len(final_errors)
)


conn = sqlite3.connect("/Users/sam/Progressive-Projects/Projects/"
    "Equity Selector/data/Final_Test_Results.db")

results = pd.read_sql_query(
    f"SELECT * FROM 'Final Test Results {STOCK_TYPE}'",
    conn
)

# Remove baseline models
results = results[
    ~results["Model"].str.contains("Baseline", case=False, na=False)
]

continuous = (
    results["Target Type"].eq("continuous")
    &
    (
        # Good magnitude + ranking prediction
        (
            (results["R2"] >= 0.05) &
            (results["Rank IC"].abs() >= 0.10)
        )
        |
        # Exceptionally strong ranking signal
        (results["Rank IC"].abs() >= 0.20)
    )
)

binary = (
    results["Target Type"].eq("binary")
    &
    (results["ROC AUC"] >= 0.60)
    &
    (results["PR AUC"] >= 0.20)
)

multiclass = (
    results["Target Type"].eq("multiclass")
    &
    (results["Macro F1"] >= 0.45)
)

useful = results[
    continuous | binary | multiclass
].copy()

useful["Prediction Type"] = useful["Target"].apply(target_category)


useful["Predictability Score"] = useful.apply(
    predictability_score,
    axis=1
)

useful = useful[
    (
        (useful["Target Type"] == "continuous")
        & (useful["Predictability Score"] >= 0.12)
    )
    |
    (
        (useful["Target Type"] == "binary")
        & (useful["Predictability Score"] >= 0.20)
    )
    |
    (
        (useful["Target Type"] == "multiclass")
        & (useful["Predictability Score"] >= 0.35)
    )
].sort_values(
    "Predictability Score",
    ascending=False
)

print(
    useful[
        [
            "Target",
            "Model",
            "Parameters"
            "Predictability Score",
        ]
    ].to_string(index=False)
)

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


useful = (
    useful
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

useful[
    "Portfolio Target Type"
] = useful.apply(
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

useful[
    "Horizon"
] = useful[
    "Target"
].map(
    target_horizon
)

useful[
    "Quality Score"
] = useful.apply(
    lambda row: calculate_quality_score(
        row,
        row[
            "Portfolio Target Type"
        ],
    ),
    axis=1,
)

missing_horizons = useful[
    useful[
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



with sqlite3.connect(
    FINAL_RESULTS_DATABASE
) as connection:

    useful.to_sql(
        f"Most Predictable Results {STOCK_TYPE}",
        connection,
        if_exists="replace",
        index=False
    )
import sqlite3
import pandas as pd

from models import *

from sklearn.preprocessing import StandardScaler

from main_package import *

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

file = open('/Users/sam/Progressive-Projects/Projects/Equity Selector/data/Selected_Features.txt', 'r')
selected_features = file.read()
file.close()

selected_features = selected_features.split('\n')

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



selected_features = selected_features[stock_type_index]

import ast

selected_features = ast.literal_eval(selected_features)

targets = list(selected_features.keys())


import json
import logging
import sqlite3
import time
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterGrid, ParameterSampler


############################################################
# SETTINGS
############################################################

DATABASE_PATH = (
    f"/Users/sam/Progressive-Projects/Projects/Equity Selector/data/Validation_Model_Fits/{STOCK_TYPE.replace(' ', '_')}.db"
)

VALIDATION_WINDOW = 20

RANDOM_STATE = 42


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)

def quote_sql_identifier(identifier):
    return '"' + str(identifier).replace('"', '""') + '"'


with sqlite3.connect(
    FEATURE_DATABASE_PATH
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


if len(SOURCE_TABLE_COLUMNS) == 0:
    raise ValueError(
        f"Source table does not exist or has no columns: {STOCK_TYPE}"
    )


logger.info(
    "Source database ready | table: %s | available columns: %d | targets: %d",
    STOCK_TYPE,
    len(SOURCE_TABLE_COLUMNS),
    len(targets)
)

logger.info(
    "Memory-safe SQL mode | full source table will not be loaded"
)

logger.info(
    "Validation database: %s | validation window: %d | random state: %d",
    DATABASE_PATH,
    VALIDATION_WINDOW,
    RANDOM_STATE
)


############################################################
# IMPORTANT
#
# When you create your model fitting package, import the
# fitting functions here.
#
# Example:
#
# from model_fits import (
#     fit_mean_baseline,
#     fit_ols,
#     fit_ridge,
#     ...
# )
#
############################################################


############################################################
# COMMON PARAMETER VALUES
############################################################

RIDGE_ALPHAS = [
    # 1e-5,
    # 3e-5,
    1e-4,
    3e-4,
    1e-3,
    # 3e-3,
    1e-2,
    # 3e-2,
    0.1,
    # 0.3,
    1,
    # 3,
    10,
    # 30,
    100,
    # 300,
    1000,
    # 3000,
    # 10000,
]


SPARSE_ALPHAS = [
    # 1e-8,
    # 3e-8,
    #1e-7,
    # 3e-7,
    #1e-6,
    # 3e-6,
    #1e-5,
    # 3e-5,
    #1e-4,
    # 3e-4,
    #1e-3,
    # 3e-3,
    1e-2,
    # 3e-2,
    0.1,
    # 0.3,
    1,
    # 3,
    10,
]


C_VALUES = [
    # 1e-5,
    # 3e-5,
    1e-4,
    3e-4,
    1e-3,
    # 3e-3,
    1e-2,
    # 3e-2,
    0.1,
    # 0.3,
    1,
    # 3,
    10,
    # 30,
    100,
    # 300,
    # 1000,
]


L1_RATIOS = [
    # 0.01,
    0.05,
    # 0.10,
    0.25,
    0.50,
    0.75,
    # 0.90,
    # 0.95,
    0.99,
]


LEARNING_RATES = [
    # 0.005,
    0.01,
    # 0.03,
    0.05,
    0.10,
    # 0.20,
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
        # 200,
        300,
        # 600,
        1000,
        # 1500,
    ],

    "max_leaf_nodes": [
        # 7,
        15,
        31,
        63,
        # 127,
    ],

    "max_depth": [
        None,
        3,
        # 5,
        7,
        # 10,
    ],

    "min_samples_leaf": [
        5,
        # 10,
        20,
        50,
        # 100,
        # 200,
    ],

    "l2_regularization": [
        0,
        # 1e-4,
        1e-3,
        # 1e-2,
        0.1,
        1,
        # 10,
        # 100,
    ],
}


########################################
# Gradient Boosting
########################################

GRADIENT_BOOSTING_PARAMS = {

    "learning_rate": LEARNING_RATES,

    "n_estimators": [
        100,
        # 200,
        300,
        # 600,
        # 1000,
    ],

    "max_depth": [
        2,
        3,
        # 4,
        5,
        # 8,
    ],

    "min_samples_leaf": [
        5,
        # 10,
        20,
        50,
        # 100,
    ],

    "subsample": [
        # 0.5,
        0.6,
        0.8,
        1.0,
    ],

    "max_features": [
        None,
        "sqrt",
        # 0.3,
        0.5,
        # 0.8,
    ],
}


########################################
# Random Forest
########################################

RANDOM_FOREST_PARAMS = {

    "n_estimators": [
        200,
        500,
        # 1000,
        # 1500,
    ],

    "max_depth": [
        None,
        5,
        10,
        # 15,
        20,
        # 30,
    ],

    "min_samples_leaf": [
        1,
        2,
        5,
        10,
        # 20,
        # 50,
        # 100,
    ],

    "min_samples_split": [
        2,
        5,
        10,
        # 20,
        # 50,
    ],

    "max_features": [
        "sqrt",
        # 0.2,
        0.3,
        0.5,
        # 0.75,
        # 1.0,
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
        # 300,
        500,
        # 750,
        # 1000,
        # 1500,
    ],

    "learning_rate": LEARNING_RATES,

    "max_depth": [
        2,
        3,
        # 4,
        5,
        # 7,
        # 10,
    ],

    "min_child_weight": [
        1,
        # 2,
        3,
        5,
        # 10,
        # 20,
        # 50,
    ],

    "subsample": [
        # 0.5,
        0.6,
        0.8,
        1.0,
    ],

    "colsample_bytree": [
        # 0.4,
        0.5,
        0.75,
        1.0,
    ],

    "gamma": [
        0,
        # 0.001,
        0.01,
        0.1,
        # 0.5,
        1,
        # 5,
    ],

    "reg_alpha": [
        0,
        # 1e-5,
        1e-4,
        # 1e-3,
        1e-2,
        0.1,
        # 1,
        # 10,
    ],

    "reg_lambda": [
        # 0,
        0.01,
        0.1,
        1,
        10,
        # 100,
    ],
}


########################################
# LightGBM
########################################

LIGHTGBM_PARAMS = {

    "n_estimators": [
        200,
        # 300,
        500,
        # 750,
        # 1000,
        # 1500,
    ],

    "learning_rate": LEARNING_RATES,

    "num_leaves": [
        # 7,
        15,
        31,
        63,
        # 127,
        # 255,
    ],

    "max_depth": [
        -1,
        3,
        5,
        # 8,
        # 12,
        # 16,
    ],

    "min_child_samples": [
        5,
        # 10,
        20,
        50,
        # 100,
        # 200,
    ],

    "subsample": [
        # 0.5,
        0.6,
        0.8,
        1.0,
    ],

    "colsample_bytree": [
        # 0.4,
        0.5,
        0.75,
        1.0,
    ],

    "reg_alpha": [
        0,
        # 1e-5,
        1e-4,
        # 1e-3,
        1e-2,
        0.1,
        # 1,
        # 10,
    ],

    "reg_lambda": [
        # 0,
        0.01,
        0.1,
        1,
        10,
        # 100,
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
        # (256,),
        # (64, 32),
        (128, 64),
        # (256, 128),
        # (128, 64, 32),
        # (256, 128, 64),
    ],

    "activation": [
        "relu",
        # "tanh",
    ],

    "alpha": [
        # 1e-7,
        1e-6,
        1e-5,
        1e-4,
        1e-3,
        # 1e-2,
        # 0.1,
    ],

    "learning_rate_init": [
        # 1e-5,
        # 3e-5,
        1e-4,
        3e-4,
        1e-3,
        # 3e-3,
        # 1e-2,
    ],

    "batch_size": [
        # 32,
        64,
        128,
        256,
        # 512,
        # "auto",
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
        # 40,
        # 80,
        # 150,
        # 250,
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

    #{
    #    "name": "Elastic Net",
    #    "function": "fit_elastic_net",
    #    "scaled": True,
    #    "search": "random",
    #    "n_iter": 10,
    #    # Later:
    #    # "n_iter": 75,
    #    "params": {
    #        "alpha": SPARSE_ALPHAS,
    #        "l1_ratio": L1_RATIOS,
    #    },
    #},


    ########################################
    # Huber
    # Uncomment later
    ########################################

    # {
    #     "name": "Huber",
    #     "function": "fit_huber",
    #     "scaled": True,
    #     "search": "random",
    #     "n_iter": 60,
    #     "params": {
    #
    #         "epsilon": [
    #             1.05,
    #             1.15,
    #             1.25,
    #             1.35,
    #             1.50,
    #             1.75,
    #             2.00,
    #             2.50,
    #         ],
    #
    #         "alpha": [
    #             0,
    #             1e-7,
    #             1e-6,
    #             1e-5,
    #             1e-4,
    #             1e-3,
    #             1e-2,
    #             0.1,
    #             1,
    #         ],
    #     },
    # },


    ########################################
    # Main Tree Models
    ########################################

    #{
    #    "name": "Hist Gradient Boosting",
    #    "function": "fit_hist_gradient_boosting_regressor",
    #    "scaled": False,
    #    "search": "random",
    #    "n_iter": 20,
    #    # Later:
    #    # "n_iter": 75,
    #    "params": HIST_GRADIENT_PARAMS,
    #},


    ########################################
    # Standard Gradient Boosting
    # Uncomment later
    ########################################

    # {
    #     "name": "Gradient Boosting",
    #     "function": "fit_gradient_boosting_regressor",
    #     "scaled": False,
    #     "search": "random",
    #     "n_iter": 60,
    #     "params": GRADIENT_BOOSTING_PARAMS,
    # },


    #{
    #    "name": "Random Forest",
    #    "function": "fit_random_forest_regressor",
    #    "scaled": False,
    #    "search": "random",
    #    "n_iter": 20,
    #    # Later:
    #    # "n_iter": 75,
    #    "params": RANDOM_FOREST_PARAMS,
    #},

    {
        "name": "XGBoost",
        "function": "fit_xgboost_regressor",
        "scaled": False,
        "search": "random",
        "n_iter": 5,
        # Later:
        # "n_iter": 100,
        "params": XGBOOST_PARAMS,
    },

    #{
    #    "name": "LightGBM",
    #    "function": "fit_lightgbm_regressor",
    #    "scaled": False,
    #    "search": "random",
    #    "n_iter": 25,
    #    # Later:
    #    # "n_iter": 100,
    #    "params": LIGHTGBM_PARAMS,
    #},


    ########################################
    # SVR
    # Uncomment later
    ########################################

    # {
    #     "name": "SVR",
    #     "function": "fit_svr",
    #     "scaled": True,
    #     "search": "random",
    #     "n_iter": 75,
    #     "params": [
    #
    #         {
    #             "kernel": ["linear"],
    #             "C": C_VALUES,
    #
    #             "epsilon": [
    #                 1e-4,
    #                 1e-3,
    #                 1e-2,
    #                 0.05,
    #                 0.1,
    #                 0.25,
    #                 0.5,
    #             ],
    #         },
    #
    #         {
    #             "kernel": ["rbf"],
    #             "C": C_VALUES,
    #
    #             "epsilon": [
    #                 1e-4,
    #                 1e-3,
    #                 1e-2,
    #                 0.05,
    #                 0.1,
    #                 0.25,
    #                 0.5,
    #             ],
    #
    #             "gamma": [
    #                 "scale",
    #                 "auto",
    #                 1e-4,
    #                 1e-3,
    #                 1e-2,
    #                 0.1,
    #                 1,
    #             ],
    #         },
    #     ],
    # },


    ########################################
    # kNN
    # Uncomment later
    ########################################

    # {
    #     "name": "kNN",
    #     "function": "fit_knn_regressor",
    #     "scaled": True,
    #     "search": "grid",
    #     "params": KNN_PARAMS,
    # },


    ########################################
    # MLP
    # Uncomment much later
    ########################################

    # {
    #     "name": "MLP",
    #     "function": "fit_mlp_regressor",
    #     "scaled": True,
    #     "search": "random",
    #     "n_iter": 75,
    #     "params": MLP_PARAMS,
    # },
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

    #{
    #    "name": "L1 Logistic Regression",
    #    "function": "fit_l1_logistic_regression",
    #    "scaled": True,
    #    "search": "grid",
    #    "params": {
    #        "C": C_VALUES,
    #        "class_weight": CLASS_WEIGHTS,
    #    },
    #},

    #{
    #    "name": "Elastic Net Logistic Regression",
    #    "function": "fit_elastic_net_logistic_regression",
    #    "scaled": True,
    #    "search": "random",
    #    "n_iter": 25,
    #    # Later:
    #    # "n_iter": 100,
    #    "params": {
    #        "C": C_VALUES,
    #        "l1_ratio": L1_RATIOS,
    #        "class_weight": CLASS_WEIGHTS,
    #    },
    #},


    ########################################
    # Tree Models
    ########################################

    #{
    #    "name": "Hist Gradient Boosting",
    #    "function": "fit_hist_gradient_boosting_classifier",
    #    "scaled": False,
    #    "search": "random",
    #    "n_iter": 20,
    #    # Later:
    #    # "n_iter": 75,
    #    "params": {
    #        **HIST_GRADIENT_PARAMS,
    #        "class_weight": CLASS_WEIGHTS,
    #    },
    #},


    ########################################
    # Standard Gradient Boosting
    # Uncomment later
    ########################################

    # {
    #     "name": "Gradient Boosting",
    #     "function": "fit_gradient_boosting_classifier",
    #     "scaled": False,
    #     "search": "random",
    #     "n_iter": 60,
    #     "params": GRADIENT_BOOSTING_PARAMS,
    # },


    #{
    #    "name": "Random Forest",
    #    "function": "fit_random_forest_classifier",
    #    "scaled": False,
    #    "search": "random",
    #    "n_iter": 20,
    #    # Later:
    #    # "n_iter": 75,
    #    "params": {
    #        **RANDOM_FOREST_PARAMS,
    #        "class_weight": CLASS_WEIGHTS,
    #    },
    #},

    {
        "name": "XGBoost",
        "function": "fit_xgboost_classifier",
        "scaled": False,
        "search": "random",
        "n_iter": 5,
        # Later:
        # "n_iter": 100,
        "params": {
            **XGBOOST_PARAMS,
            "class_weight": CLASS_WEIGHTS,
        },
    },

    #{
    #    "name": "LightGBM",
    #    "function": "fit_lightgbm_classifier",
    #    "scaled": False,
    #    "search": "random",
    #    "n_iter": 25,
    #    # Later:
    #    # "n_iter": 100,
    #    "params": {
    #        **LIGHTGBM_PARAMS,
    #        "class_weight": CLASS_WEIGHTS,
    #    },
    #},


    ########################################
    # SVM
    # Uncomment later
    ########################################

    # {
    #     "name": "SVM",
    #     "function": "fit_svm_classifier",
    #     "scaled": True,
    #     "search": "random",
    #     "n_iter": 75,
    #     "params": [
    #
    #         {
    #             "kernel": ["linear"],
    #             "C": C_VALUES,
    #             "class_weight": CLASS_WEIGHTS,
    #         },
    #
    #         {
    #             "kernel": ["rbf"],
    #             "C": C_VALUES,
    #
    #             "gamma": [
    #                 "scale",
    #                 "auto",
    #                 1e-4,
    #                 1e-3,
    #                 1e-2,
    #                 0.1,
    #                 1,
    #             ],
    #
    #             "class_weight": CLASS_WEIGHTS,
    #         },
    #     ],
    # },


    ########################################
    # kNN
    # Uncomment later
    ########################################

    # {
    #     "name": "kNN",
    #     "function": "fit_knn_classifier",
    #     "scaled": True,
    #     "search": "grid",
    #     "params": KNN_PARAMS,
    # },


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
                # 1e-13,
                1e-12,
                # 1e-11,
                1e-10,
                1e-9,
                1e-8,
                # 1e-7,
                1e-6,
                # 1e-5,
            ],
        },
    },


    ########################################
    # MLP
    # Uncomment much later
    ########################################

    # {
    #     "name": "MLP",
    #     "function": "fit_mlp_classifier",
    #     "scaled": True,
    #     "search": "random",
    #     "n_iter": 75,
    #     "params": MLP_PARAMS,
    # },
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

    #{
    #    "name": "L2 Multinomial Logistic Regression",
    #    "function": "fit_l2_multinomial_logistic_regression",
    #    "scaled": True,
    #    "search": "grid",
    #    "params": {
    #        "C": C_VALUES,
    #        "class_weight": CLASS_WEIGHTS,
    #    },
    #},

    #{
    #    "name": "L1 Multinomial Logistic Regression",
    #    "function": "fit_l1_multinomial_logistic_regression",
    #    "scaled": True,
    #    "search": "grid",
    #    "params": {
    #        "C": C_VALUES,
    #        "class_weight": CLASS_WEIGHTS,
    #    },
    #},

    #{
    #    "name": "Elastic Net Multinomial Logistic Regression",
    #    "function": "fit_elastic_net_multinomial_logistic_regression",
    #    "scaled": True,
    #    "search": "random",
    #    "n_iter": 25,
    #    # Later:
    #    # "n_iter": 100,
    #    "params": {
    #        "C": C_VALUES,
    #        "l1_ratio": L1_RATIOS,
    #        "class_weight": CLASS_WEIGHTS,
    #    },
    #},


    ########################################
    # LDA
    # Uncomment later
    ########################################

    # {
    #     "name": "LDA",
    #     "function": "fit_lda",
    #     "scaled": True,
    #     "search": "grid",
    #     "params": [
    #
    #         {
    #             "solver": ["svd"],
    #         },
    #
    #         {
    #             "solver": [
    #                 "lsqr",
    #                 "eigen",
    #             ],
    #
    #             "shrinkage": [
    #                 None,
    #                 "auto",
    #                 0.05,
    #                 0.1,
    #                 0.25,
    #                 0.5,
    #                 0.75,
    #                 0.9,
    #                 0.95,
    #             ],
    #         },
    #     ],
    # },


    ########################################
    # QDA
    # Uncomment later
    ########################################

    # {
    #     "name": "QDA",
    #     "function": "fit_qda",
    #     "scaled": True,
    #     "search": "grid",
    #     "params": {
    #
    #         "reg_param": [
    #             0,
    #             0.0001,
    #             0.001,
    #             0.01,
    #             0.05,
    #             0.1,
    #             0.25,
    #             0.5,
    #             0.75,
    #             1.0,
    #         ],
    #     },
    # },


    ########################################
    # Tree Models
    ########################################

    #{
    #    "name": "Hist Gradient Boosting",
    #    "function": "fit_hist_gradient_boosting_multiclass",
    #    "scaled": False,
    #    "search": "random",
    #    "n_iter": 20,
    #    # Later:
    #    # "n_iter": 75,
    #    "params": {
    #        **HIST_GRADIENT_PARAMS,
    #        "class_weight": CLASS_WEIGHTS,
    #    },
    #},


    ########################################
    # Standard Gradient Boosting
    # Uncomment later
    ########################################

    # {
    #     "name": "Gradient Boosting",
    #     "function": "fit_gradient_boosting_multiclass",
    #     "scaled": False,
    #     "search": "random",
    #     "n_iter": 60,
    #     "params": GRADIENT_BOOSTING_PARAMS,
    # },


    #{
    #    "name": "Random Forest",
    #    "function": "fit_random_forest_multiclass",
    #    "scaled": False,
    #    "search": "random",
    #    "n_iter": 20,
    #    # Later:
    #    # "n_iter": 75,
    #    "params": {
    #        **RANDOM_FOREST_PARAMS,
    #        "class_weight": CLASS_WEIGHTS,
    #    },
    #},

    {
        "name": "XGBoost",
        "function": "fit_xgboost_multiclass",
        "scaled": False,
        "search": "random",
        "n_iter": 5,
        # Later:
        # "n_iter": 100,
        "params": {
            **XGBOOST_PARAMS,
            "class_weight": CLASS_WEIGHTS,
        },
    },

    #{
    #    "name": "LightGBM",
    #    "function": "fit_lightgbm_multiclass",
    #    "scaled": False,
    #    "search": "random",
    #    "n_iter": 25,
    #    # Later:
    #    # "n_iter": 100,
    #    "params": {
    #        **LIGHTGBM_PARAMS,
    #        "class_weight": CLASS_WEIGHTS,
    #    },
    #},


    ########################################
    # SVM
    # Uncomment later
    ########################################

    # {
    #     "name": "SVM",
    #     "function": "fit_svm_multiclass",
    #     "scaled": True,
    #     "search": "random",
    #     "n_iter": 75,
    #     "params": [
    #
    #         {
    #             "kernel": ["linear"],
    #             "C": C_VALUES,
    #             "class_weight": CLASS_WEIGHTS,
    #         },
    #
    #         {
    #             "kernel": ["rbf"],
    #             "C": C_VALUES,
    #
    #             "gamma": [
    #                 "scale",
    #                 "auto",
    #                 1e-4,
    #                 1e-3,
    #                 1e-2,
    #                 0.1,
    #                 1,
    #             ],
    #
    #             "class_weight": CLASS_WEIGHTS,
    #         },
    #     ],
    # },


    ########################################
    # MLP
    # Uncomment much later
    ########################################

    # {
    #     "name": "MLP",
    #     "function": "fit_mlp_multiclass",
    #     "scaled": True,
    #     "search": "random",
    #     "n_iter": 75,
    #     "params": MLP_PARAMS,
    # },


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
                # 1e-5,
                # 3e-5,
                1e-4,
                # 3e-4,
                1e-3,
                # 3e-3,
                1e-2,
                # 3e-2,
                0.1,
                # 0.3,
                1,
                # 3,
                10,
                # 30,
                100,
            ],
        },
    },
]

connection = sqlite3.connect(DATABASE_PATH)


import json
import copy
from sklearn.model_selection import ParameterGrid


def parameter_key(params):
    return json.dumps(
        params,
        sort_keys=True,
        separators=(",", ":"),
        default=list
    )




def remove_completed_models(
    models,
    old_df,
    old_combinations
):

    logger.info(
        "Checking %d model families for previously completed configurations",
        len(models)
    )

    new_models = []

    for model in models:

        model = copy.deepcopy(model)

        name = model["name"]
        params = model["params"]

        # If this model has never been run, leave it completely unchanged
        if name not in old_df["Model"].values:
            new_models.append(model)
            continue

        # Generate every exact parameter combination
        combinations = list(ParameterGrid(params))

        remaining = [
            combination
            for combination in combinations
            if (
                name,
                parameter_key(combination)
            ) not in old_combinations
        ]

        # Entire model has already been completed
        if not remaining:
            continue

        # Convert back into sklearn-compatible parameter dictionaries
        model["params"] = [
            {
                key: [value]
                for key, value in combination.items()
            }
            for combination in remaining
        ]

        # Prevent RandomizedSearchCV asking for more
        # combinations than are actually left
        if model["search"] == "random":
            model["n_iter"] = min(
                model["n_iter"],
                len(remaining)
            )

        new_models.append(model)

    logger.info(
        "Model-family filtering complete | remaining: %d / %d",
        len(new_models),
        len(models)
    )

    return new_models


ALL_CONTINUOUS_MODELS = [
    model.copy()
    for model in CONTINUOUS_MODELS
]

ALL_BINARY_MODELS = [
    model.copy()
    for model in BINARY_MODELS
]

ALL_MULTICLASS_MODELS = [
    model.copy()
    for model in MULTICLASS_MODELS
]


############################################################
# PRUNING / FINAL VALIDATION SETTINGS
############################################################

# Do not performance-prune from a single fold.
# Each configuration gets at least 3 validation folds before
# it can be removed for weak performance.
MIN_FOLDS_BEFORE_PRUNING = 3

# Progressive racing schedule.
#
# fold : (fraction of each model family's parameter sets kept,
#         minimum number kept in that family)
#
# A family with only one configuration is cheap enough to keep.
PRUNING_STAGES = {
    3: (0.50, 5),
    5: (0.50, 3),
    7: (0.50, 2),
}

# Number of mature search configurations to promote across all
# model families for a complete all-fold validation.
FINALISTS_PER_TARGET = 8

# Old databases created with the previous aggressive pruning may
# contain many configurations that only have 1-2 folds.
# Keep the best few partial configurations from each old model
# family so they get a fair all-fold re-evaluation.
LEGACY_PARTIAL_PER_MODEL = 2

# Stability penalty used ONLY to choose the small finalist set.
# Final leaderboard ranking still reports the actual metrics.
FINALIST_STD_PENALTY = 0.25

# Final test eligibility stability limits.
MAX_CONTINUOUS_RANK_IC_STD = 0.10
MAX_BINARY_ROC_AUC_STD = 0.07
MAX_MULTICLASS_MACRO_F1_STD = 0.10

try:

    ############################################################
    # CONTINUOUS TARGET
    ############################################################

    continuous_df = pd.read_sql(
        """
        SELECT *
        FROM 'Future Return 20__folds'
        """,
        connection
    )

    continuous_combinations = {
        (
            row["Model"],
            parameter_key(json.loads(row["Parameters"]))
        )
        for _, row in continuous_df.iterrows()
    }


    ############################################################
    # BINARY TARGET
    ############################################################

    binary_df = pd.read_sql(
        """
        SELECT *
        FROM 'Future Direction 20__folds'
        """,
        connection
    )

    binary_combinations = {
        (
            row["Model"],
            parameter_key(json.loads(row["Parameters"]))
        )
        for _, row in binary_df.iterrows()
    }


    ############################################################
    # MULTICLASS TARGET
    ############################################################

    multiclass_df = pd.read_sql(
        """
        SELECT *
        FROM 'Future Regime 20__folds'
        """,
        connection
    )

    multiclass_combinations = {
        (
            row["Model"],
            parameter_key(json.loads(row["Parameters"]))
        )
        for _, row in multiclass_df.iterrows()
    }


    ############################################################
    # ACTIVE MODEL LISTS
    ############################################################

    CONTINUOUS_MODELS = remove_completed_models(
        CONTINUOUS_MODELS,
        continuous_df,
        continuous_combinations
    )

    BINARY_MODELS = remove_completed_models(
        BINARY_MODELS,
        binary_df,
        binary_combinations
    )

    MULTICLASS_MODELS = remove_completed_models(
        MULTICLASS_MODELS,
        multiclass_df,
        multiclass_combinations
    )


except Exception:
    old_df = {}

    logger.info(
        "No existing fold-result rows"
    )


logger.info(
    "Active model families | continuous: %d | binary: %d | multiclass: %d",
    len(CONTINUOUS_MODELS),
    len(BINARY_MODELS),
    len(MULTICLASS_MODELS)
)



############################################################
# PARAMETER SEARCH
############################################################

def get_parameter_sets(model_config):

    params = model_config["params"]


    ########################################
    # No Parameters
    ########################################

    if params == {}:
        return [{}]


    ########################################
    # Exhaustive
    ########################################

    if model_config["search"] == "grid":

        return list(
            ParameterGrid(params)
        )


    ########################################
    # Random Search
    ########################################

    if model_config["search"] == "random":

        return list(
            ParameterSampler(
                params,
                n_iter=model_config["n_iter"],
                random_state=RANDOM_STATE
            )
        )


    raise ValueError(
        f"Unknown search type: {model_config['search']}"
    )


############################################################
# CONVERT PARAMETERS TO SQLITE-SAFE JSON
############################################################

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


############################################################
# PURGE TRAINING DATA
############################################################

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


############################################################
# FIND MODEL FUNCTION
############################################################

def get_model_function(function_name):

    function = globals().get(function_name)

    if function is None:

        raise NameError(
            f"{function_name} has not been imported yet."
        )

    return function


############################################################
# RUN ONE MODEL OVER ALL PARAMETER COMBINATIONS
############################################################

############################################################
# RUN ONE MODEL'S CURRENT SURVIVING PARAMETERS
############################################################

def run_model_search(

    model_config,
    parameter_sets,

    x_train,
    y_train,

    x_validation,
    y_validation,

    x_train_scaled,
    x_validation_scaled
):

    results = []

    logger.info(
        "%s | fitting %d parameter configurations",
        model_config["name"],
        len(parameter_sets)
    )


    fit_function = get_model_function(
        model_config["function"]
    )


    ########################################################
    # Scaled / Unscaled
    ########################################################

    if model_config["scaled"]:

        current_x_train = x_train_scaled
        current_x_validation = x_validation_scaled

    else:

        current_x_train = x_train
        current_x_validation = x_validation


    ########################################################
    # Surviving Parameter Sets Only
    ########################################################

    for parameters in parameter_sets:

        try:

            result = fit_function(

                current_x_train,
                y_train,

                current_x_validation,
                y_validation,

                **parameters
            )


            if result is None:
                result = {}


            result = result.copy()

            result["Model"] = (
                model_config["name"]
            )

            result["Parameters"] = (
                parameters_to_json(
                    parameters
                )
            )

            result["Status"] = "OK"

            result["Error"] = None


        except Exception as error:

            result = {

                "Model": (
                    model_config["name"]
                ),

                "Parameters": (
                    parameters_to_json(
                        parameters
                    )
                ),

                "Status": "ERROR",

                "Error": str(error)
            }


        results.append(
            result
        )


    logger.info(
        "%s | completed %d parameter configurations",
        model_config["name"],
        len(results)
    )

    return results

############################################################
# GET MODELS FOR TARGET TYPE
############################################################

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


############################################################
# RUN EVERY MODEL FOR ONE FOLD
############################################################

############################################################
# RUN SURVIVING MODELS
############################################################

def run_models(

    candidates,

    x_train,
    y_train,

    x_validation,
    y_validation,

    x_train_scaled,
    x_validation_scaled
):

    all_results = []


    for model_name, candidate_data in candidates.items():

        model_config = (
            candidate_data["model_config"]
        )

        parameter_sets = (
            candidate_data["parameters"]
        )


        logger.info(
            "%s | Testing %d configurations",
            model_name,
            len(parameter_sets)
        )


        model_results = run_model_search(

            model_config=model_config,

            parameter_sets=parameter_sets,

            x_train=x_train,
            y_train=y_train,

            x_validation=x_validation,
            y_validation=y_validation,

            x_train_scaled=x_train_scaled,
            x_validation_scaled=x_validation_scaled
        )


        all_results.extend(
            model_results
        )


    return all_results

############################################################
# PRUNE MODEL CONFIGURATIONS
############################################################

def prune_model_candidates(
    candidates,
    all_results,
    target,
    target_type,
    completed_fold
):

    ########################################################
    # No pruning before the configured racing stage.
    #
    # This is the main change from the old version:
    # nothing is eliminated after fold 1 or fold 2.
    ########################################################

    if completed_fold not in PRUNING_STAGES:

        return candidates


    metric, lower_is_better = primary_metric(
        target,
        target_type
    )


    results_df = pd.DataFrame(
        all_results
    )


    if results_df.empty:

        return candidates


    keep_fraction, minimum_keep = (
        PRUNING_STAGES[
            completed_fold
        ]
    )


    new_candidates = {}


    ########################################################
    # Prune parameter sets WITHIN each model family.
    #
    # We do not compare a 3-fold LightGBM parameter set with
    # a 10-fold Ridge parameter set here. The final all-model
    # comparison happens only after finalists are rerun over
    # every fold.
    ########################################################

    for model_name, candidate_data in candidates.items():

        model_config = (
            candidate_data["model_config"]
        )

        parameters = (
            candidate_data["parameters"]
        )


        ####################################################
        # One configuration is cheap enough to keep.
        ####################################################

        if len(parameters) <= 1:

            new_candidates[
                model_name
            ] = candidate_data

            continue


        parameter_lookup = {

            parameters_to_json(params):
            params

            for params in parameters
        }


        model_results = results_df[
            (
                results_df["Model"]
                == model_name
            )
            &
            (
                results_df["Parameters"].isin(
                    parameter_lookup.keys()
                )
            )
        ].copy()


        if (
            model_results.empty
            or metric not in model_results.columns
        ):

            logger.warning(
                "%s | Cannot prune using %s",
                model_name,
                metric
            )

            new_candidates[
                model_name
            ] = candidate_data

            continue


        ####################################################
        # Only successful fits count toward the metric.
        ####################################################

        successful = model_results[
            model_results["Status"] == "OK"
        ].copy()


        if successful.empty:

            logger.warning(
                "%s | No successful fits available at fold %d",
                model_name,
                completed_fold
            )

            # All currently-active configurations have already
            # been attempted through this stage. If none worked,
            # there is no reason to keep rerunning the family.
            continue


        ####################################################
        # Require enough successful folds BEFORE using a
        # configuration in a performance-based race.
        ####################################################

        summary = (
            successful
            .groupby(
                "Parameters"
            )
            .agg(
                Metric_Mean=(
                    metric,
                    "mean"
                ),
                Metric_Std=(
                    metric,
                    "std"
                ),
                Fold_Count=(
                    "Fold",
                    "nunique"
                )
            )
        )


        summary = summary[
            summary["Fold_Count"]
            >= MIN_FOLDS_BEFORE_PRUNING
        ].copy()


        summary = summary.dropna(
            subset=[
                "Metric_Mean"
            ]
        )


        if summary.empty:

            # This should be unusual once completed_fold >= 3,
            # but keeping the family is safer than pruning using
            # insufficient evidence.
            new_candidates[
                model_name
            ] = candidate_data

            continue


        ####################################################
        # Rank using mean performance first.
        #
        # Fold-to-fold standard deviation is only a tie-break
        # here. The explicit stability penalty is used later
        # when selecting the small finalist set.
        ####################################################

        summary[
            "Metric_Std"
        ] = (
            summary[
                "Metric_Std"
            ]
            .fillna(
                np.inf
            )
        )


        if lower_is_better:

            summary = summary.sort_values(
                [
                    "Metric_Mean",
                    "Metric_Std"
                ],
                ascending=[
                    True,
                    True
                ]
            )

        else:

            summary = summary.sort_values(
                [
                    "Metric_Mean",
                    "Metric_Std"
                ],
                ascending=[
                    False,
                    True
                ]
            )


        current_count = len(
            parameters
        )


        keep_count = max(
            minimum_keep,
            int(
                np.ceil(
                    current_count
                    * keep_fraction
                )
            )
        )


        keep_count = min(
            keep_count,
            len(summary)
        )


        surviving_ids = list(
            summary.index[
                :keep_count
            ]
        )


        surviving_parameters = [

            parameter_lookup[
                parameter_id
            ]

            for parameter_id in surviving_ids

            if parameter_id
            in parameter_lookup
        ]


        if len(surviving_parameters) == 0:

            continue


        new_candidates[
            model_name
        ] = {

            "model_config":
                model_config,

            "parameters":
                surviving_parameters
        }


        logger.info(
            "%s | Fold %d pruning: %d -> %d configurations",
            model_name,
            completed_fold,
            current_count,
            len(surviving_parameters)
        )


    return new_candidates


############################################################
# PRIMARY VALIDATION METRIC
############################################################

def primary_metric(target, target_type):

    if target_type == "continuous":

        if target.startswith("Future Return Rank"):
            return "Rank IC", False

        return "RMSE", True


    if target_type == "binary":
        return "ROC AUC", False


    if target_type == "multiclass":
        return "Macro F1", False


    raise ValueError(
        f"Unknown target type: {target_type}"
    )

############################################################
# INITIAL MODEL CANDIDATES
############################################################

def initialise_model_candidates(target_type):

    models = get_models(
        target_type
    )

    candidates = {}


    for model_config in models:

        parameter_sets = get_parameter_sets(
            model_config
        )

        candidates[
            model_config["name"]
        ] = {

            "model_config": model_config,

            "parameters": parameter_sets
        }


    return candidates


############################################################
# MASTER MODEL CONFIGURATION LOOKUP
############################################################

def get_all_models(target_type):

    if target_type == "continuous":
        return ALL_CONTINUOUS_MODELS

    if target_type == "binary":
        return ALL_BINARY_MODELS

    if target_type == "multiclass":
        return ALL_MULTICLASS_MODELS

    raise ValueError(
        f"Unknown target type: {target_type}"
    )


def get_all_model_config_lookup(
    target_type
):

    return {

        model_config["name"]:
            model_config

        for model_config
        in get_all_models(
            target_type
        )
    }


############################################################
# RESULT / CANDIDATE HELPERS
############################################################

def result_key(
    model,
    parameters,
    fold
):

    return (
        str(model),
        str(parameters),
        int(fold)
    )


def completed_result_keys(
    fold_results
):

    if (
        fold_results is None
        or len(fold_results) == 0
    ):

        return set()


    required = {
        "Model",
        "Parameters",
        "Fold"
    }


    if not required.issubset(
        fold_results.columns
    ):

        return set()


    keys = set()


    for _, row in fold_results.iterrows():

        if pd.isna(
            row["Fold"]
        ):
            continue

        keys.add(
            result_key(
                row["Model"],
                row["Parameters"],
                row["Fold"]
            )
        )


    return keys


def pending_candidates_for_fold(
    candidates,
    fold,
    existing_keys
):

    pending = {}


    for (
        model_name,
        candidate_data
    ) in candidates.items():

        remaining_parameters = []


        for parameters in (
            candidate_data[
                "parameters"
            ]
        ):

            parameter_id = (
                parameters_to_json(
                    parameters
                )
            )


            key = result_key(
                model_name,
                parameter_id,
                fold
            )


            if key not in existing_keys:

                remaining_parameters.append(
                    parameters
                )


        if len(
            remaining_parameters
        ) > 0:

            pending[
                model_name
            ] = {

                "model_config":
                    candidate_data[
                        "model_config"
                    ],

                "parameters":
                    remaining_parameters
            }


    return pending


def append_fold_metadata(
    fold_results,
    fold,
    target,
    target_type,
    current_train,
    current_validation,
    validation_start,
    validation_end,
    purge_days,
    features
):

    enriched = []


    for result in fold_results:

        result = result.copy()

        result["Fold"] = fold

        result["Target"] = target

        result[
            "Target Type"
        ] = target_type

        result[
            "Train Start"
        ] = (
            current_train[
                "Date"
            ].min()
        )

        result[
            "Train End"
        ] = (
            current_train[
                "Date"
            ].max()
        )

        result[
            "Validation Start"
        ] = validation_start

        result[
            "Validation End"
        ] = validation_end

        result[
            "Purge Days"
        ] = purge_days

        result[
            "Train Rows"
        ] = len(
            current_train
        )

        result[
            "Validation Rows"
        ] = len(
            current_validation
        )

        result[
            "Number Features"
        ] = len(
            features
        )

        enriched.append(
            result
        )


    return enriched


############################################################
# BUILD WALK-FORWARD FOLD CONTEXTS
############################################################

def build_walk_forward_folds(
    train_df,
    validation_df,
    features,
    target,
    purge_days,
    validation_window
):

    train_df = train_df.copy()

    validation_df = (
        validation_df.copy()
    )


    train_df[
        "Date"
    ] = pd.to_datetime(
        train_df[
            "Date"
        ]
    )


    validation_df[
        "Date"
    ] = pd.to_datetime(
        validation_df[
            "Date"
        ]
    )


    ########################################################
    # Expanding development data.
    #
    # Previous validation periods become available for later
    # training folds exactly as in your original code.
    ########################################################

    development_df = pd.concat(
        [
            train_df,
            validation_df
        ],
        ignore_index=True
    )


    development_df = (
        development_df
        .sort_values(
            [
                "Date",
                "Ticker"
            ]
        )
        .reset_index(
            drop=True
        )
    )


    validation_dates = np.sort(
        validation_df[
            "Date"
        ].unique()
    )


    folds = []

    fold_number = 1


    for start in range(
        0,
        len(validation_dates),
        validation_window
    ):

        fold_dates = validation_dates[
            start:
            start + validation_window
        ]


        if len(
            fold_dates
        ) == 0:

            continue


        validation_start = (
            fold_dates[0]
        )

        validation_end = (
            fold_dates[-1]
        )


        current_train = (
            development_df[
                development_df[
                    "Date"
                ]
                < validation_start
            ]
            .copy()
        )


        current_train = (
            purge_training_data(
                current_train,
                purge_days
            )
        )


        current_validation = (
            validation_df[
                validation_df[
                    "Date"
                ].isin(
                    fold_dates
                )
            ]
            .copy()
        )


        if current_train.empty:
            continue

        if current_validation.empty:
            continue


        x_train = (
            current_train[
                features
            ]
        )

        y_train = (
            current_train[
                target
            ]
        )


        x_validation = (
            current_validation[
                features
            ]
        )

        y_validation = (
            current_validation[
                target
            ]
        )


        ####################################################
        # Scale inside each fold only.
        ####################################################

        scaler = (
            StandardScaler()
        )


        x_train_scaled = pd.DataFrame(

            scaler.fit_transform(
                x_train
            ),

            columns=features,

            index=x_train.index
        )


        x_validation_scaled = (
            pd.DataFrame(

                scaler.transform(
                    x_validation
                ),

                columns=features,

                index=x_validation.index
            )
        )


        folds.append(
            {
                "fold":
                    fold_number,

                "validation_start":
                    validation_start,

                "validation_end":
                    validation_end,

                "current_train":
                    current_train,

                "current_validation":
                    current_validation,

                "x_train":
                    x_train,

                "y_train":
                    y_train,

                "x_validation":
                    x_validation,

                "y_validation":
                    y_validation,

                "x_train_scaled":
                    x_train_scaled,

                "x_validation_scaled":
                    x_validation_scaled
            }
        )


        fold_number += 1


    return folds


############################################################
# WALK-FORWARD SEARCH WITH CONSERVATIVE PRUNING
############################################################

def walk_forward_validation(
    train_df,
    validation_df,
    features,
    target,
    target_type,
    purge_days,
    validation_window=20,
    existing_fold_results=None
):

    ########################################################
    # Existing SQLite history is part of the search history.
    #
    # This is important when the script is rerun with only a
    # few new models left in CONTINUOUS_MODELS /
    # BINARY_MODELS / MULTICLASS_MODELS.
    ########################################################

    if existing_fold_results is None:

        existing_fold_results = (
            pd.DataFrame()
        )

    else:

        existing_fold_results = (
            existing_fold_results
            .copy()
        )


    folds = build_walk_forward_folds(
        train_df=train_df,
        validation_df=validation_df,
        features=features,
        target=target,
        purge_days=purge_days,
        validation_window=(
            validation_window
        )
    )


    total_folds = len(
        folds
    )


    if total_folds == 0:

        return (
            pd.DataFrame(),
            initialise_model_candidates(
                target_type
            ),
            existing_fold_results.copy(),
            0
        )


    candidates = (
        initialise_model_candidates(
            target_type
        )
    )


    new_results = []


    history_records = (
        existing_fold_results
        .to_dict(
            "records"
        )
        if not existing_fold_results.empty
        else []
    )


    existing_keys = (
        completed_result_keys(
            existing_fold_results
        )
    )


    ########################################################
    # Search folds.
    ########################################################

    for fold_data in folds:

        fold = fold_data[
            "fold"
        ]


        logger.info(
            "%s | Search fold %d/%d | Train rows %d | Validation rows %d",
            target,
            fold,
            total_folds,
            len(
                fold_data[
                    "current_train"
                ]
            ),
            len(
                fold_data[
                    "current_validation"
                ]
            )
        )


        ####################################################
        # Only run model / parameter / fold combinations that
        # are not already in SQLite.
        ####################################################

        pending_candidates = (
            pending_candidates_for_fold(
                candidates=candidates,
                fold=fold,
                existing_keys=(
                    existing_keys
                )
            )
        )


        if len(
            pending_candidates
        ) > 0:

            fold_results = (
                run_models(

                    candidates=(
                        pending_candidates
                    ),

                    x_train=(
                        fold_data[
                            "x_train"
                        ]
                    ),

                    y_train=(
                        fold_data[
                            "y_train"
                        ]
                    ),

                    x_validation=(
                        fold_data[
                            "x_validation"
                        ]
                    ),

                    y_validation=(
                        fold_data[
                            "y_validation"
                        ]
                    ),

                    x_train_scaled=(
                        fold_data[
                            "x_train_scaled"
                        ]
                    ),

                    x_validation_scaled=(
                        fold_data[
                            "x_validation_scaled"
                        ]
                    )
                )
            )


            fold_results = (
                append_fold_metadata(

                    fold_results=(
                        fold_results
                    ),

                    fold=fold,

                    target=target,

                    target_type=(
                        target_type
                    ),

                    current_train=(
                        fold_data[
                            "current_train"
                        ]
                    ),

                    current_validation=(
                        fold_data[
                            "current_validation"
                        ]
                    ),

                    validation_start=(
                        fold_data[
                            "validation_start"
                        ]
                    ),

                    validation_end=(
                        fold_data[
                            "validation_end"
                        ]
                    ),

                    purge_days=(
                        purge_days
                    ),

                    features=features
                )
            )


            for result in fold_results:

                new_results.append(
                    result
                )

                history_records.append(
                    result
                )

                existing_keys.add(
                    result_key(
                        result["Model"],
                        result["Parameters"],
                        result["Fold"]
                    )
                )


        ####################################################
        # IMPORTANT:
        # prune ONCE after the entire fold has completed.
        #
        # The old implementation pruned from inside the
        # "for result in fold_results" loop, so candidates
        # could be repeatedly pruned while the same fold was
        # still being appended.
        ####################################################

        candidates = (
            prune_model_candidates(

                candidates=candidates,

                all_results=(
                    history_records
                ),

                target=target,

                target_type=(
                    target_type
                ),

                completed_fold=fold
            )
        )


        if len(
            candidates
        ) == 0:

            logger.info(
                "%s | No active search candidates remain after fold %d",
                target,
                fold
            )

            break


    history_df = pd.DataFrame(
        history_records
    )


    return (
        pd.DataFrame(
            new_results
        ),
        candidates,
        history_df,
        total_folds
    )


############################################################
# AGGREGATE SEARCH PERFORMANCE FOR FINALIST SELECTION
############################################################

def finalist_search_summary(
    fold_results,
    target,
    target_type
):

    if fold_results.empty:

        return pd.DataFrame()


    successful = (
        fold_results[
            fold_results[
                "Status"
            ]
            == "OK"
        ]
        .copy()
    )


    if successful.empty:

        return pd.DataFrame()


    metric, lower_is_better = (
        primary_metric(
            target,
            target_type
        )
    )


    if metric not in successful.columns:

        return pd.DataFrame()


    summary = (
        successful
        .groupby(
            [
                "Model",
                "Parameters"
            ],
            dropna=False
        )
        .agg(
            Metric_Mean=(
                metric,
                "mean"
            ),
            Metric_Std=(
                metric,
                "std"
            ),
            Fold_Count=(
                "Fold",
                "nunique"
            )
        )
        .reset_index()
    )


    summary = summary.dropna(
        subset=[
            "Metric_Mean"
        ]
    )


    summary[
        "Metric_Std_For_Selection"
    ] = (
        summary[
            "Metric_Std"
        ]
        .fillna(
            0.0
        )
    )


    ########################################################
    # A modest stability penalty helps avoid promoting a
    # configuration simply because one fold was spectacular.
    ########################################################

    if lower_is_better:

        summary[
            "Selection Score"
        ] = (
            summary[
                "Metric_Mean"
            ]
            + FINALIST_STD_PENALTY
            * summary[
                "Metric_Std_For_Selection"
            ]
        )

        summary = summary.sort_values(
            [
                "Selection Score",
                "Metric_Mean"
            ],
            ascending=[
                True,
                True
            ]
        )

    else:

        summary[
            "Selection Score"
        ] = (
            summary[
                "Metric_Mean"
            ]
            - FINALIST_STD_PENALTY
            * summary[
                "Metric_Std_For_Selection"
            ]
        )

        summary = summary.sort_values(
            [
                "Selection Score",
                "Metric_Mean"
            ],
            ascending=[
                False,
                False
            ]
        )


    return (
        summary
        .reset_index(
            drop=True
        )
    )


############################################################
# SELECT A SMALL SET TO RE-RUN ON EVERY FOLD
############################################################

def select_finalist_configurations(
    fold_results,
    target,
    target_type
):

    summary = (
        finalist_search_summary(
            fold_results=fold_results,
            target=target,
            target_type=target_type
        )
    )


    if summary.empty:

        return {}


    selected_rows = []


    ########################################################
    # Mature search configurations:
    # at least 3 successful folds.
    ########################################################

    mature = summary[
        summary[
            "Fold_Count"
        ]
        >= MIN_FOLDS_BEFORE_PRUNING
    ].copy()


    if not mature.empty:

        selected_rows.extend(
            mature
            .head(
                FINALISTS_PER_TARGET
            )
            .to_dict(
                "records"
            )
        )


    ########################################################
    # Legacy repair:
    #
    # Older database results produced by the previous
    # aggressive pruning may only contain 1-2 folds.
    #
    # Promote the best few PARTIAL configurations from each
    # model family so those old models are not permanently
    # disadvantaged by the old fold-1 pruning rule.
    ########################################################

    partial = summary[
        summary[
            "Fold_Count"
        ]
        < MIN_FOLDS_BEFORE_PRUNING
    ].copy()


    if not partial.empty:

        for _, group in (
            partial.groupby(
                "Model",
                sort=False
            )
        ):

            selected_rows.extend(
                group
                .head(
                    LEGACY_PARTIAL_PER_MODEL
                )
                .to_dict(
                    "records"
                )
            )


    if len(
        selected_rows
    ) == 0:

        return {}


    selected_df = (
        pd.DataFrame(
            selected_rows
        )
        .drop_duplicates(
            subset=[
                "Model",
                "Parameters"
            ],
            keep="first"
        )
        .reset_index(
            drop=True
        )
    )


    ########################################################
    # Keep the finalist set bounded.
    #
    # Mature candidates get priority. Legacy partial entries
    # are only extra repair candidates.
    ########################################################

    maximum_finalists = (
        FINALISTS_PER_TARGET
        + (
            LEGACY_PARTIAL_PER_MODEL
            * selected_df[
                "Model"
            ].nunique()
        )
    )


    selected_df = (
        selected_df
        .head(
            maximum_finalists
        )
    )


    model_lookup = (
        get_all_model_config_lookup(
            target_type
        )
    )


    finalists = {}


    for _, row in selected_df.iterrows():

        model_name = row[
            "Model"
        ]


        if model_name not in model_lookup:

            logger.warning(
                "%s | Cannot fully validate %s because its model config is no longer defined",
                target,
                model_name
            )

            continue


        try:

            parameters = json.loads(
                row[
                    "Parameters"
                ]
            )

        except Exception as error:

            logger.warning(
                "%s | Could not parse parameters for %s: %s",
                target,
                model_name,
                error
            )

            continue


        if model_name not in finalists:

            finalists[
                model_name
            ] = {

                "model_config":
                    model_lookup[
                        model_name
                    ],

                "parameters": []
            }


        existing_parameter_ids = {

            parameters_to_json(
                params
            )

            for params in finalists[
                model_name
            ][
                "parameters"
            ]
        }


        parameter_id = (
            parameters_to_json(
                parameters
            )
        )


        if (
            parameter_id
            not in existing_parameter_ids
        ):

            finalists[
                model_name
            ][
                "parameters"
            ].append(
                parameters
            )


    finalist_count = sum(
        len(
            data[
                "parameters"
            ]
        )
        for data
        in finalists.values()
    )


    logger.info(
        "%s | Promoted %d configurations to full-fold validation",
        target,
        finalist_count
    )


    return finalists


############################################################
# FULL-FOLD VALIDATION FOR FINALISTS
############################################################

def run_full_validation_for_finalists(
    train_df,
    validation_df,
    features,
    target,
    target_type,
    purge_days,
    finalists,
    existing_fold_results,
    validation_window=20
):

    if len(
        finalists
    ) == 0:

        return (
            pd.DataFrame(),
            existing_fold_results.copy(),
            0
        )


    folds = build_walk_forward_folds(
        train_df=train_df,
        validation_df=validation_df,
        features=features,
        target=target,
        purge_days=purge_days,
        validation_window=(
            validation_window
        )
    )


    total_folds = len(
        folds
    )


    history_records = (
        existing_fold_results
        .to_dict(
            "records"
        )
        if not existing_fold_results.empty
        else []
    )


    existing_keys = (
        completed_result_keys(
            existing_fold_results
        )
    )


    new_results = []


    for fold_data in folds:

        fold = fold_data[
            "fold"
        ]


        pending = (
            pending_candidates_for_fold(
                candidates=finalists,
                fold=fold,
                existing_keys=(
                    existing_keys
                )
            )
        )


        pending_count = sum(
            len(
                data[
                    "parameters"
                ]
            )
            for data
            in pending.values()
        )


        if pending_count == 0:
            continue


        logger.info(
            "%s | Final validation fold %d/%d | %d missing finalist configurations",
            target,
            fold,
            total_folds,
            pending_count
        )


        fold_results = run_models(

            candidates=pending,

            x_train=(
                fold_data[
                    "x_train"
                ]
            ),

            y_train=(
                fold_data[
                    "y_train"
                ]
            ),

            x_validation=(
                fold_data[
                    "x_validation"
                ]
            ),

            y_validation=(
                fold_data[
                    "y_validation"
                ]
            ),

            x_train_scaled=(
                fold_data[
                    "x_train_scaled"
                ]
            ),

            x_validation_scaled=(
                fold_data[
                    "x_validation_scaled"
                ]
            )
        )


        fold_results = (
            append_fold_metadata(

                fold_results=(
                    fold_results
                ),

                fold=fold,

                target=target,

                target_type=(
                    target_type
                ),

                current_train=(
                    fold_data[
                        "current_train"
                    ]
                ),

                current_validation=(
                    fold_data[
                        "current_validation"
                    ]
                ),

                validation_start=(
                    fold_data[
                        "validation_start"
                    ]
                ),

                validation_end=(
                    fold_data[
                        "validation_end"
                    ]
                ),

                purge_days=(
                    purge_days
                ),

                features=features
            )
        )


        for result in fold_results:

            new_results.append(
                result
            )

            history_records.append(
                result
            )

            existing_keys.add(
                result_key(
                    result["Model"],
                    result["Parameters"],
                    result["Fold"]
                )
            )


    return (
        pd.DataFrame(
            new_results
        ),
        pd.DataFrame(
            history_records
        ),
        total_folds
    )


############################################################
# KEEP ONLY FINALIST FOLD ROWS
############################################################

def finalist_fold_results(
    fold_results,
    finalists
):

    if fold_results.empty:
        return pd.DataFrame()


    allowed = set()


    for (
        model_name,
        candidate_data
    ) in finalists.items():

        for parameters in (
            candidate_data[
                "parameters"
            ]
        ):

            allowed.add(
                (
                    model_name,
                    parameters_to_json(
                        parameters
                    )
                )
            )


    mask = [

        (
            row["Model"],
            row["Parameters"]
        )
        in allowed

        for _, row
        in fold_results.iterrows()
    ]


    return (
        fold_results[
            mask
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )


############################################################
# LEADERBOARD SORTING RULES
############################################################

def leaderboard_sorting(
    target,
    target_type
):

    ########################################
    # Continuous
    ########################################

    if target_type == "continuous":

        # For future return ranks you may prefer
        # Rank IC as the primary criterion.

        if target.startswith(
            "Future Return Rank"
        ):

            return [
                ("Rank IC Mean", False),
                ("Rank IC Std", True),
                ("RMSE Mean", True),
                ("RMSE Std", True),
                ("MAE Mean", True),
            ]


        return [
            ("RMSE Mean", True),
            ("RMSE Std", True),
            ("MAE Mean", True),
            ("R2 Mean", False),
            ("Rank IC Mean", False),
            ("Rank IC Std", True),
        ]


    ########################################
    # Binary
    ########################################

    if target_type == "binary":

        return [
            ("ROC AUC Mean", False),
            ("ROC AUC Std", True),
            ("PR AUC Mean", False),
            ("PR AUC Std", True),
            ("Log Loss Mean", True),
            ("F1 Mean", False),
        ]


    ########################################
    # Multiclass
    ########################################

    if target_type == "multiclass":

        return [
            ("Macro F1 Mean", False),
            ("Macro F1 Std", True),
            ("Balanced Accuracy Mean", False),
            ("Balanced Accuracy Std", True),
            ("Log Loss Mean", True),
        ]


    raise ValueError(
        f"Unknown target type: {target_type}"
    )


############################################################
# AGGREGATE WALK-FORWARD RESULTS
############################################################

def create_leaderboard(
    fold_results,
    target,
    target_type
):

    if fold_results.empty:
        return pd.DataFrame()


    ########################################################
    # Only Successful Fits
    ########################################################

    successful = fold_results[
        fold_results["Status"] == "OK"
    ].copy()


    if successful.empty:

        return pd.DataFrame({
            "Message": [
                "No model configurations completed successfully"
            ]
        })


    ########################################################
    # Columns That Are Metadata Rather Than Metrics
    ########################################################

    metadata_columns = {

        "Fold",
        "Target",
        "Target Type",

        "Model",
        "Parameters",

        "Status",
        "Error",

        "Train Start",
        "Train End",

        "Validation Start",
        "Validation End",

        "Purge Days",

        "Train Rows",
        "Validation Rows",

        "Number Features",
    }


    ########################################################
    # Find Numeric Metric Columns
    ########################################################

    metric_columns = [

        column

        for column in successful.columns

        if (
            column not in metadata_columns
            and pd.api.types.is_numeric_dtype(
                successful[column]
            )
        )
    ]


    ########################################################
    # Group By Exact Model + Parameter Combination
    ########################################################

    grouped = successful.groupby(
        [
            "Model",
            "Parameters"
        ],
        dropna=False
    )


    leaderboard_parts = []


    ########################################################
    # Mean Metrics
    ########################################################

    means = grouped[
        metric_columns
    ].mean()

    means.columns = [
        f"{column} Mean"
        for column in means.columns
    ]

    leaderboard_parts.append(
        means
    )


    ########################################################
    # Standard Deviation Across Folds
    ########################################################

    stds = grouped[
        metric_columns
    ].std()

    stds.columns = [
        f"{column} Std"
        for column in stds.columns
    ]

    leaderboard_parts.append(
        stds
    )


    ########################################################
    # Number Of Folds Successfully Completed
    ########################################################

    fold_count = grouped[
        "Fold"
    ].nunique().rename(
        "Fold Count"
    )

    leaderboard_parts.append(
        fold_count
    )


    ########################################################
    # Combine
    ########################################################

    leaderboard = pd.concat(
        leaderboard_parts,
        axis=1
    ).reset_index()


    ########################################################
    # Sorting
    ########################################################

    sorting_rules = leaderboard_sorting(
        target,
        target_type
    )


    sort_columns = []

    ascending = []


    for column, direction in sorting_rules:

        if column in leaderboard.columns:

            sort_columns.append(
                column
            )

            ascending.append(
                direction
            )


    if len(sort_columns) > 0:

        leaderboard = leaderboard.sort_values(

            sort_columns,

            ascending=ascending,

            na_position="last"
        )


    ########################################################
    # Rank
    ########################################################

    leaderboard = leaderboard.reset_index(
        drop=True
    )


    leaderboard.insert(
        0,
        "Rank",
        np.arange(
            1,
            len(leaderboard) + 1
        )
    )


    leaderboard.insert(
        1,
        "Target",
        target
    )


    leaderboard.insert(
        2,
        "Target Type",
        target_type
    )


    return leaderboard



############################################################
# FINAL TEST ELIGIBILITY
############################################################

def add_testing_eligibility(
    leaderboard,
    target_type,
    total_folds
):

    if leaderboard.empty:

        return leaderboard


    leaderboard = (
        leaderboard.copy()
    )


    ########################################################
    # Every final-test candidate must have completed every
    # walk-forward validation fold successfully.
    ########################################################

    full_validation = (
        leaderboard[
            "Fold Count"
        ]
        == total_folds
    )


    eligible = pd.Series(
        False,
        index=leaderboard.index
    )


    reasons = pd.Series(
        "",
        index=leaderboard.index,
        dtype=object
    )


    ########################################################
    # Continuous
    #
    # Same core quality gate you are already using for
    # predictable continuous targets, plus fold stability.
    ########################################################

    if target_type == "continuous":

        if (
            "Rank IC Mean"
            not in leaderboard.columns
        ):

            reasons[:] = (
                "Rank IC Mean missing"
            )

        else:

            rank_ic = (
                leaderboard[
                    "Rank IC Mean"
                ]
                .abs()
            )


            r2 = (
                leaderboard[
                    "R2 Mean"
                ]
                if "R2 Mean"
                in leaderboard.columns
                else pd.Series(
                    np.nan,
                    index=leaderboard.index
                )
            )


            rank_ic_std = (
                leaderboard[
                    "Rank IC Std"
                ]
                if "Rank IC Std"
                in leaderboard.columns
                else pd.Series(
                    np.nan,
                    index=leaderboard.index
                )
            )


            predictive_gate = (
                (
                    (r2 >= 0.05)
                    &
                    (rank_ic >= 0.10)
                )
                |
                (
                    rank_ic >= 0.20
                )
            )


            stability_gate = (
                rank_ic_std
                <= MAX_CONTINUOUS_RANK_IC_STD
            )


            eligible = (
                full_validation
                &
                predictive_gate
                &
                stability_gate
            )


            reasons = np.where(
                ~full_validation,
                "Not all folds completed",
                np.where(
                    ~predictive_gate,
                    "Below continuous predictability gate",
                    np.where(
                        ~stability_gate,
                        "Rank IC too unstable across folds",
                        "Eligible"
                    )
                )
            )


    ########################################################
    # Binary
    ########################################################

    elif target_type == "binary":

        roc_auc = (
            leaderboard[
                "ROC AUC Mean"
            ]
            if "ROC AUC Mean"
            in leaderboard.columns
            else pd.Series(
                np.nan,
                index=leaderboard.index
            )
        )


        pr_auc = (
            leaderboard[
                "PR AUC Mean"
            ]
            if "PR AUC Mean"
            in leaderboard.columns
            else pd.Series(
                np.nan,
                index=leaderboard.index
            )
        )


        roc_auc_std = (
            leaderboard[
                "ROC AUC Std"
            ]
            if "ROC AUC Std"
            in leaderboard.columns
            else pd.Series(
                np.nan,
                index=leaderboard.index
            )
        )


        balanced_accuracy = (
            leaderboard[
                "Balanced Accuracy Mean"
            ]
            if "Balanced Accuracy Mean"
            in leaderboard.columns
            else pd.Series(
                1.0,
                index=leaderboard.index
            )
        )


        predictive_gate = (
            (roc_auc >= 0.60)
            &
            (pr_auc >= 0.20)
            &
            (balanced_accuracy > 0.50)
        )


        stability_gate = (
            roc_auc_std
            <= MAX_BINARY_ROC_AUC_STD
        )


        eligible = (
            full_validation
            &
            predictive_gate
            &
            stability_gate
        )


        reasons = np.where(
            ~full_validation,
            "Not all folds completed",
            np.where(
                ~predictive_gate,
                "Below binary predictability gate",
                np.where(
                    ~stability_gate,
                    "ROC AUC too unstable across folds",
                    "Eligible"
                )
            )
        )


    ########################################################
    # Multiclass
    ########################################################

    elif target_type == "multiclass":

        macro_f1 = (
            leaderboard[
                "Macro F1 Mean"
            ]
            if "Macro F1 Mean"
            in leaderboard.columns
            else pd.Series(
                np.nan,
                index=leaderboard.index
            )
        )


        macro_f1_std = (
            leaderboard[
                "Macro F1 Std"
            ]
            if "Macro F1 Std"
            in leaderboard.columns
            else pd.Series(
                np.nan,
                index=leaderboard.index
            )
        )


        predictive_gate = (
            macro_f1 >= 0.45
        )


        stability_gate = (
            macro_f1_std
            <= MAX_MULTICLASS_MACRO_F1_STD
        )


        eligible = (
            full_validation
            &
            predictive_gate
            &
            stability_gate
        )


        reasons = np.where(
            ~full_validation,
            "Not all folds completed",
            np.where(
                ~predictive_gate,
                "Below multiclass Macro F1 gate",
                np.where(
                    ~stability_gate,
                    "Macro F1 too unstable across folds",
                    "Eligible"
                )
            )
        )


    else:

        raise ValueError(
            f"Unknown target type: {target_type}"
        )


    leaderboard.insert(
        3,
        "Testing Eligible",
        eligible.astype(bool)
    )


    leaderboard.insert(
        4,
        "Testing Eligibility Reason",
        reasons
    )


    return leaderboard


############################################################
# DEDUPLICATE COMPLETE FOLD HISTORY
############################################################

def deduplicate_fold_results(
    fold_results
):

    if fold_results.empty:

        return fold_results


    duplicate_columns = [

        column

        for column in [
            "Target",
            "Model",
            "Parameters",
            "Fold"
        ]

        if column
        in fold_results.columns
    ]


    if len(
        duplicate_columns
    ) < 2:

        return fold_results


    return (
        fold_results
        .drop_duplicates(
            subset=duplicate_columns,
            keep="first"
        )
        .reset_index(
            drop=True
        )
    )


############################################################
# MAIN TARGET LOOP
############################################################

file = open(
    "/Users/sam/Progressive-Projects/Projects/"
    "Equity Selector/data/Target_Best_Model.txt",
    "a"
)

file.write("{")


data_connection = sqlite3.connect(
    FEATURE_DATABASE_PATH
)

data_connection.execute(
    "PRAGMA query_only = ON"
)


with sqlite3.connect(
    DATABASE_PATH
) as connection:


    ####################################################
    # CHECK IF TABLE EXISTS
    ####################################################

    def table_exists(
        table_name
    ):

        result = connection.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type='table'
            AND name=?
            """,
            (
                table_name,
            )
        ).fetchone()

        return result is not None


    ####################################################
    # TARGET LOOP
    ####################################################

    for target_number, target in enumerate(
        targets,
        start=1
    ):


        ################################################
        # FEATURES SELECTED FOR THIS TARGET
        ################################################

        features = (
            selected_features[
                target
            ]
        )


        logger.info(
            "[%d/%d] %s | Starting | %d features",
            target_number,
            len(targets),
            target,
            len(features)
        )


        ################################################
        # NO FEATURES
        ################################################

        if len(
            features
        ) == 0:

            logger.info(
                "%s | No features survived screening",
                target
            )


            if not table_exists(
                target
            ):

                empty_result = pd.DataFrame(
                    {
                        "Rank":
                            [1],

                        "Target":
                            [target],

                        "Message":
                            [
                                "No features survived screening"
                            ]
                    }
                )


                empty_result.to_sql(

                    target,

                    connection,

                    if_exists="replace",

                    index=False
                )


            continue


        ################################################
        # DATA REQUIRED FOR TARGET
        ################################################

        columns = (

            [
                "Date",
                "Ticker",

                "Open",
                "Close",
                "Low",
                "High",
                "Volume",

                target
            ]

            + features
        )


        columns = list(
            dict.fromkeys(
                columns
            )
        )


        ################################################
        # LOAD ONLY THE DATA REQUIRED FOR THIS TARGET
        ################################################

        missing_columns = [
            column
            for column in columns
            if column not in SOURCE_TABLE_COLUMN_SET
        ]


        if len(missing_columns) > 0:
            raise KeyError(
                f"{target} | Columns missing from {STOCK_TYPE}: "
                f"{missing_columns}"
            )


        sql_columns = ", ".join(
            quote_sql_identifier(column)
            for column in columns
        )


        target_query = (
            f"SELECT {sql_columns} "
            f"FROM {quote_sql_identifier(STOCK_TYPE)}"
        )


        logger.info(
            "[%d/%d] %s | Loading %d/%d source columns",
            target_number,
            len(targets),
            target,
            len(columns),
            len(SOURCE_TABLE_COLUMNS)
        )


        load_start = time.perf_counter()


        current_df = pd.read_sql_query(
            target_query,
            data_connection
        )


        load_seconds = (
            time.perf_counter()
            - load_start
        )


        memory_mb = (
            current_df
            .memory_usage(
                index=True,
                deep=True
            )
            .sum()
            / (1024 ** 2)
        )


        logger.info(
            "[%d/%d] %s | SQL load complete | %d rows x %d columns | %.1f MB | %.2fs",
            target_number,
            len(targets),
            target,
            len(current_df),
            len(current_df.columns),
            memory_mb,
            load_seconds
        )


        ################################################
        # DROP MISSING VALUES ONLY FOR TARGET +
        # SELECTED FEATURES
        ################################################

        rows_before_dropna = len(
            current_df
        )


        current_df = (
            current_df.dropna(
                subset=(
                    [
                        target
                    ]
                    + features
                )
            )
        )


        logger.info(
            "[%d/%d] %s | Rows after dropna: %d/%d",
            target_number,
            len(targets),
            target,
            len(current_df),
            rows_before_dropna
        )


        current_df[
            "Date"
        ] = pd.to_datetime(
            current_df[
                "Date"
            ]
        )


        current_df = (
            current_df
            .sort_values(
                [
                    "Date",
                    "Ticker"
                ]
            )
            .reset_index(
                drop=True
            )
        )


        ################################################
        # TRAIN / VALIDATION / TEST
        ################################################

        (
            train_df,
            validation_df,
            test_df
        ) = (
            train_validation_test_split(
                current_df
            )
        )


        logger.info(
            "[%d/%d] %s | Rows | train: %d | validation: %d | test: %d",
            target_number,
            len(targets),
            target,
            len(train_df),
            len(validation_df),
            len(test_df)
        )


        ################################################
        # TARGET TYPE
        ################################################

        _type = target_type(
            train_df,
            target
        )


        ################################################
        # PURGE HORIZON
        ################################################

        purge_days = (
            target_purge_days(
                target
            )
        )


        logger.info(
            "%s | Type: %s | Purge: %d days",
            target,
            _type,
            purge_days
        )


        ################################################
        # LOAD EXISTING FOLD HISTORY FIRST
        #
        # This is deliberately BEFORE the new search.
        #
        # It allows:
        #   - already-completed model/fold combinations
        #     to be skipped;
        #   - a run containing only a few newly-added
        #     models to coexist with all previous models;
        #   - old partial candidates to be repaired by
        #     the final all-fold validation pass.
        ################################################

        folds_table = (
            f"{target}__folds"
        )


        if table_exists(
            folds_table
        ):

            old_fold_results = (
                pd.read_sql_query(

                    f'SELECT * '
                    f'FROM "{folds_table}"',

                    connection
                )
            )


            logger.info(
                "%s | Loaded %d previous fold results",
                target,
                len(
                    old_fold_results
                )
            )


        else:

            old_fold_results = (
                pd.DataFrame()
            )


        ################################################
        # SEARCH / RACING STAGE
        #
        # New active model configurations are evaluated.
        #
        # No performance pruning occurs until fold 3.
        # Pruning occurs once per completed fold stage,
        # never once per result row.
        ################################################

        (
            search_new_results,
            final_search_candidates,
            search_history,
            total_folds
        ) = walk_forward_validation(

            train_df=train_df,

            validation_df=(
                validation_df
            ),

            features=features,

            target=target,

            target_type=_type,

            purge_days=(
                purge_days
            ),

            validation_window=(
                VALIDATION_WINDOW
            ),

            existing_fold_results=(
                old_fold_results
            )
        )


        if total_folds == 0:

            logger.warning(
                "%s | No valid walk-forward folds",
                target
            )

            continue


        ################################################
        # SEARCH LEADERBOARD
        #
        # This table may intentionally contain different
        # Fold Count values. It is diagnostic only.
        #
        # DO NOT use this table to choose the final test
        # model.
        ################################################

        search_history = (
            deduplicate_fold_results(
                search_history
            )
        )


        search_leaderboard = (
            create_leaderboard(

                fold_results=(
                    search_history
                ),

                target=target,

                target_type=_type
            )
        )


        search_table = (
            f"{target}__search"
        )


        if not search_leaderboard.empty:

            search_leaderboard.to_sql(

                search_table,

                connection,

                if_exists="replace",

                index=False
            )


        ################################################
        # SELECT FINALISTS
        #
        # Strong mature search configurations are taken
        # across ALL model families.
        #
        # A few legacy 1-2-fold candidates from each old
        # family are also promoted so the old aggressive
        # pruning does not permanently bias the database.
        ################################################

        finalists = (
            select_finalist_configurations(

                fold_results=(
                    search_history
                ),

                target=target,

                target_type=_type
            )
        )


        finalist_count = sum(
            len(
                candidate_data[
                    "parameters"
                ]
            )

            for candidate_data
            in finalists.values()
        )


        if finalist_count == 0:

            logger.warning(
                "%s | No configurations could be promoted to final validation",
                target
            )

            ################################################
            # Preserve the updated fold history even if no
            # final candidates can currently be constructed.
            ################################################

            search_history.to_sql(

                folds_table,

                connection,

                if_exists="replace",

                index=False
            )

            continue


        ################################################
        # FINAL ALL-FOLD VALIDATION
        #
        # Every finalist is evaluated on every missing
        # walk-forward fold.
        #
        # Existing matching fold rows are reused rather
        # than recomputed.
        ################################################

        (
            final_validation_new_results,
            complete_history,
            final_total_folds
        ) = (
            run_full_validation_for_finalists(

                train_df=train_df,

                validation_df=(
                    validation_df
                ),

                features=features,

                target=target,

                target_type=_type,

                purge_days=(
                    purge_days
                ),

                finalists=finalists,

                existing_fold_results=(
                    search_history
                ),

                validation_window=(
                    VALIDATION_WINDOW
                )
            )
        )


        complete_history = (
            deduplicate_fold_results(
                complete_history
            )
        )


        ################################################
        # FINALIST FOLD RESULTS ONLY
        ################################################

        final_fold_results = (
            finalist_fold_results(

                fold_results=(
                    complete_history
                ),

                finalists=finalists
            )
        )


        ################################################
        # FINAL VALIDATION LEADERBOARD
        #
        # Only configurations that successfully completed
        # EVERY validation fold are allowed here.
        ################################################

        final_leaderboard = (
            create_leaderboard(

                fold_results=(
                    final_fold_results
                ),

                target=target,

                target_type=_type
            )
        )


        if not final_leaderboard.empty:

            final_leaderboard = (
                final_leaderboard[
                    final_leaderboard[
                        "Fold Count"
                    ]
                    == final_total_folds
                ]
                .copy()
                .reset_index(
                    drop=True
                )
            )


        ################################################
        # RE-RANK AFTER REMOVING INCOMPLETE FINALISTS
        ################################################

        if not final_leaderboard.empty:

            final_leaderboard[
                "Rank"
            ] = range(
                1,
                len(
                    final_leaderboard
                )
                + 1
            )


            final_leaderboard = (
                add_testing_eligibility(

                    leaderboard=(
                        final_leaderboard
                    ),

                    target_type=_type,

                    total_folds=(
                        final_total_folds
                    )
                )
            )


        ################################################
        # SAVE FINAL LEADERBOARD
        #
        # The target-named table now means:
        #
        # "finalists evaluated over the same complete
        #  walk-forward validation folds"
        #
        # This is the table from which a test-set model
        # may be selected.
        ################################################

        if not final_leaderboard.empty:

            final_leaderboard.to_sql(

                target,

                connection,

                if_exists="replace",

                index=False
            )


        else:

            logger.warning(
                "%s | No finalist completed all %d folds",
                target,
                final_total_folds
            )


        ################################################
        # SAVE COMPLETE FOLD HISTORY
        ################################################


        date_columns = [
            "Train Start",
            "Train End",
            "Validation Start",
            "Validation End"
        ]

        for column in date_columns:

            if column in complete_history.columns:

                complete_history[column] = pd.to_datetime(
                    complete_history[column],
                    errors="coerce"
                )

        complete_history.to_sql(
            folds_table,
            connection,
            if_exists="replace",
            index=False
        )


        ################################################
        # LOG COUNTS
        ################################################

        logger.info(
            "%s | Added %d search fold results",
            target,
            len(
                search_new_results
            )
        )


        logger.info(
            "%s | Added %d final-validation fold results",
            target,
            len(
                final_validation_new_results
            )
        )


        logger.info(
            "%s | Complete fold-history rows: %d",
            target,
            len(
                complete_history
            )
        )


        logger.info(
            "%s | Full-fold finalists ranked: %d",
            target,
            len(
                final_leaderboard
            )
        )


        ################################################
        # WRITE ONLY A TEST-ELIGIBLE WINNER
        #
        # Rank 1 alone is no longer enough.
        ################################################

        if (
            not final_leaderboard.empty
            and "Testing Eligible"
            in final_leaderboard.columns
        ):

            eligible_models = (
                final_leaderboard[
                    final_leaderboard[
                        "Testing Eligible"
                    ]
                ]
                .copy()
            )


            if not eligible_models.empty:

                winner = (
                    eligible_models.iloc[0]
                )


                logger.info(
                    "%s | TEST ELIGIBLE | Best: %s | %s",
                    target,
                    winner["Model"],
                    winner["Parameters"]
                )


                file.write(
                    f"'{target}' : "
                    f"['{winner['Model']}', "
                    f"{winner['Parameters']}], "
                )


            else:

                logger.info(
                    "%s | No model currently meets the final test eligibility gate",
                    target
                )


        logger.info(
            "%s | Complete",
            target
        )


file.write("}\n")

file.close()

data_connection.close()

logger.info(
    "Complete | source SQLite connection closed"
)
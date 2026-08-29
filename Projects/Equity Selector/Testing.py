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

MAX_CONTINUOUS_RANK_IC_STD = 0.10
MAX_BINARY_ROC_AUC_STD = 0.07
MAX_MULTICLASS_MACRO_F1_STD = 0.10

MODELS_TO_DO = []

logger.info("Starting validation model fitting")
logger.info("Stock type: %s", STOCK_TYPE)

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

no_features_targets = [
    target 
    for target in targets 
    if not selected_features[target]
]

targets = [
    target 
    for target in targets 
    if target not in no_features_targets
]

logger.info("Loaded %d targets", len(targets))

VALIDATION_DATABASE_PATH = (
    "/Users/sam/Progressive-Projects/Projects/"
    "Equity Selector/data/Validation_Model_Fits/"
    f"{STOCK_TYPE.replace(' ', '_')}.db"
)


def combos_to_models(all_models):

    models = {}

    for model in all_models:

        name = model["name"]
        params = model["params"]

        if name not in models:
            models[name] = {}

        for key, value in params.items():

            if key not in models[name]:
                models[name][key] = []

            if value not in models[name][key]:
                models[name][key].append(value)

    return [
        {
            "name": name,
            "params": params
        }
        for name, params in models.items()
    ]


def intersection_of_used_models(used_models):

    if not used_models:
        return []

    values = list(used_models.values())

    intersection = [
        model
        for model in values[0]
        if all(
            model in target_models
            for target_models in values[1:]
        )
    ]

    return intersection


logger.info("Reading existing validation tables")

with sqlite3.connect(VALIDATION_DATABASE_PATH) as conn:
    table_names = [
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table';"
        )
    ]

logger.info("Found %d validation database tables", len(table_names))

for i in range(len(table_names)):
    table_names[i] = table_names[i].rstrip("__search")
    table_names[i] = table_names[i].rstrip("__folds")

table_targets = list(set(table_names))

table_targets = [
    target 
    for target in table_targets 
    if target not in no_features_targets
]

missing_targets = [
    target
    for target in targets
    if target not in table_targets
]

logger.info(
    "%d/%d targets already have validation tables",
    len(targets) - len(missing_targets),
    len(targets)
)

logger.info(
    "%d targets have no existing validation tables",
    len(missing_targets)
)

types = []

FEATURE_DATABASE_PATH = (
    "/Users/sam/Progressive-Projects/Projects/"
    "Equity Selector/data/Features_Targets_Data.db"
)

with sqlite3.connect(FEATURE_DATABASE_PATH) as conn:

    columns = ", ".join(
        f'"{column}"'
        for column in missing_targets
    )

    df = pd.read_sql_query(
        f'''
        SELECT {columns}
        FROM "{STOCK_TYPE}"
        ''',
        conn
    )

for target in missing_targets:
    types.append(target_type(df, target))

types = list(set(types))

logger.info(
    "Target types with new targets: %s",
    types
)

import ast


USED_CONTINUOUS_MODELS = {}
USED_BINARY_MODELS = {}
USED_MULTICLASS_MODELS = {}


logger.info("Loading previously tested model configurations")

with sqlite3.connect(VALIDATION_DATABASE_PATH) as conn:

    for target in targets:

        try:
            df = pd.read_sql_query(
                f'''
                SELECT "Target", "Target Type", "Model", "Parameters"
                FROM "{target}__search"
                ''',
                conn
            )
        except Exception:
            logger.debug(
                "No existing search table found for target: %s",
                target
            )
            continue


        if df.empty:
            logger.debug(
                "Search table is empty for target: %s",
                target
            )
            continue

        _type = df["Target Type"].iloc[0]

        if _type == "continuous":
            used_models = USED_CONTINUOUS_MODELS

        elif _type == "binary":
            used_models = USED_BINARY_MODELS

        elif _type == "multiclass":
            used_models = USED_MULTICLASS_MODELS

        used_models[target] = []

        for _, row in df.iterrows():

            parameters = ast.literal_eval(
                row["Parameters"].replace("null", "None")
            )

            used_models[target].append({
                "name": row["Model"],
                "params": parameters,
            })



RIDGE_ALPHAS = [
     1e-5,
     3e-5,
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
    # 300,
     1000,
     3000,
     10000,
]


SPARSE_ALPHAS = [
     1e-8,
    # 3e-8,
     1e-7,
    # 3e-7,
     1e-6,
    # 3e-6,
     1e-5,
    # 3e-5,
     1e-4,
    # 3e-4,
     1e-3,
    # 3e-3,
    1e-2,
    # 3e-2,
    0.1,
     0.3,
     1,
     3,
     10,
]


C_VALUES = [
    # 1e-5,
    # 3e-5,
    # 1e-4,
    # 3e-4,
    # 1e-3,
    # 3e-3,
    # 1e-2,
    # 3e-2,
    0.1,
    # 0.3,
    # 1,
    # 3,
    # 10,
    # 30,
    # 100,
    # 300,
    # 1000,
]


L1_RATIOS = [
    # 0.01,
    # 0.05,
    # 0.10,
    # 0.25,
    0.50,
    # 0.75,
    # 0.90,
    # 0.95,
    # 0.99,
]


LEARNING_RATES = [
    # 0.005,
    # 0.01,
    # 0.03,
    0.05,
    # 0.10,
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
        # 100,
        200,
        # 300,
        # 600,
        # 1000,
        # 1500,
    ],

    "max_leaf_nodes": [
        # 7,
        # 15,
        31,
        # 63,
        # 127,
    ],

    "max_depth": [
        # None,
        # 3,
        5,
        # 7,
        # 10,
    ],

    "min_samples_leaf": [
        # 5,
        # 10,
        20,
        # 50,
        # 100,
        # 200,
    ],

    "l2_regularization": [
        # 0,
        # 1e-4,
        # 1e-3,
        # 1e-2,
        0.1,
        # 1,
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
        # 100,
        200,
        # 300,
        # 600,
        # 1000,
    ],

    "max_depth": [
        # 2,
        3,
        # 4,
        # 5,
        # 8,
    ],

    "min_samples_leaf": [
        # 5,
        # 10,
        20,
        # 50,
        # 100,
    ],

    "subsample": [
        # 0.5,
        # 0.6,
        0.8,
        # 1.0,
    ],

    "max_features": [
        # None,
        "sqrt",
        # 0.3,
        # 0.5,
        # 0.8,
    ],
}


########################################
# Random Forest
########################################

RANDOM_FOREST_PARAMS = {

    "n_estimators": [
        200,
        # 500,
        # 1000,
        # 1500,
    ],

    "max_depth": [
        # None,
        # 5,
        10,
        # 15,
        # 20,
        # 30,
    ],

    "min_samples_leaf": [
        # 1,
        # 2,
        5,
        # 10,
        # 20,
        # 50,
        # 100,
    ],

    "min_samples_split": [
        # 2,
        5,
        # 10,
        # 20,
        # 50,
    ],

    "max_features": [
        "sqrt",
        # 0.2,
        # 0.3,
        # 0.5,
        # 0.75,
        # 1.0,
    ],

    "bootstrap": [
        True,
        # False,
    ],
}


########################################
# XGBoost
########################################

XGBOOST_PARAMS = {

    "n_estimators": [
        200,
        # 300,
        # 500,
        # 750,
        # 1000,
        # 1500,
    ],

    "learning_rate": LEARNING_RATES,

    "max_depth": [
        # 2,
        3,
        # 4,
        # 5,
        # 7,
        # 10,
    ],

    "min_child_weight": [
        1,
        # 2,
        # 3,
        # 5,
        # 10,
        # 20,
        # 50,
    ],

    "subsample": [
        # 0.5,
        # 0.6,
        0.8,
        # 1.0,
    ],

    "colsample_bytree": [
        # 0.4,
        # 0.5,
        0.75,
        # 1.0,
    ],

    "gamma": [
        0,
        # 0.001,
        # 0.01,
        # 0.1,
        # 0.5,
        # 1,
        # 5,
    ],

    "reg_alpha": [
        0,
        # 1e-5,
        # 1e-4,
        # 1e-3,
        # 1e-2,
        # 0.1,
        # 1,
        # 10,
    ],

    "reg_lambda": [
        # 0,
        # 0.01,
        # 0.1,
        1,
        # 10,
        # 100,
    ],
}


########################################
# LightGBM
########################################

LIGHTGBM_PARAMS = {

    "n_estimators": [
        200,
         300,
        # 500,
        # 750,
        # 1000,
        # 1500,
    ],

    "learning_rate": LEARNING_RATES,

    "num_leaves": [
         7,
         15,
        #31,
        # 63,
        # 127,
        # 255,
    ],

    "max_depth": [
        -1,
         3,
        # 5,
        # 8,
        # 12,
        # 16,
    ],

    "min_child_samples": [
        # 5,
        # 10,
        20,
        # 50,
        # 100,
        # 200,
    ],

    "subsample": [
        # 0.5,
        # 0.6,
        0.8,
        # 1.0,
    ],

    "colsample_bytree": [
        # 0.4,
        # 0.5,
        0.75,
        # 1.0,
    ],

    "reg_alpha": [
        #0,
         1e-5,
         1e-4,
        # 1e-3,
         1e-2,
        # 0.1,
         1,
        # 10,
    ],

    "reg_lambda": [
        # 0,
        # 0.01,
        # 0.1,
        1,
        # 10,
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
        # (32,),
        (64,),
        # (128,),
        # (256,),
        # (64, 32),
        # (128, 64),
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
        # 1e-6,
        # 1e-5,
        1e-4,
        # 1e-3,
        # 1e-2,
        # 0.1,
    ],

    "learning_rate_init": [
        # 1e-5,
        # 3e-5,
        # 1e-4,
        # 3e-4,
          1e-3,
        # 3e-3,
        # 1e-2,
    ],

    "batch_size": [
        # 32,
        # 64,
        128,
        # 256,
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
        "params": {},
    },

    {
        "name": "OLS",
        "function": "fit_ols",
        "scaled": True,
        "params": {},
    },

    {
        "name": "Ridge",
        "function": "fit_ridge",
        "scaled": True,
        "params": {
            "alpha": RIDGE_ALPHAS,
        },
    },

    #{
    #    "name": "Lasso",
    #    "function": "fit_lasso",
    #    "scaled": True,
    #    "params": {
    #        "alpha": SPARSE_ALPHAS,
    #    },
    #},

    #{
    #    "name": "Elastic Net",
    #    "function": "fit_elastic_net",
    #    "scaled": True,
    #    "params": {
    #        "alpha": SPARSE_ALPHAS,
    #        "l1_ratio": L1_RATIOS,
    #    },
    #},


    ########################################
    # Huber
    # Uncomment later
    ########################################

    #{
    #    "name": "Huber",
    #    "function": "fit_huber",
    #    "scaled": True,
    #    "params": {

    #        "epsilon": [
    #            # 1.05,
    #            # 1.15,
    #            # 1.25,
    #            1.35,
    #            # 1.50,
    #            # 1.75,
    #            # 2.00,
    #            # 2.50,
    #        ],

    #        "alpha": [
    #            # 0,
    #            # 1e-7,
    #            # 1e-6,
    #            # 1e-5,
    #            1e-4,
    #            # 1e-3,
    #            # 1e-2,
    #            # 0.1,
    #            # 1,
    #        ],
    #    },
    #},


    ########################################
    # Main Tree Models
    ########################################

    #{
    #   "name": "Hist Gradient Boosting",
    #   "function": "fit_hist_gradient_boosting_regressor",
    #   "scaled": False,
    #   "params": HIST_GRADIENT_PARAMS,
    #},


    ########################################
    # Standard Gradient Boosting
    # Uncomment later
    ########################################

    #{
    #    "name": "Gradient Boosting",
    #    "function": "fit_gradient_boosting_regressor",
    #    "scaled": False,
    #    "params": GRADIENT_BOOSTING_PARAMS,
    #},


    #{
    #    "name": "Random Forest",
    #    "function": "fit_random_forest_regressor",
    #    "scaled": False,
    #    "params": RANDOM_FOREST_PARAMS,
    #},

    {
        "name": "XGBoost",
        "function": "fit_xgboost_regressor",
        "scaled": False,
        "params": XGBOOST_PARAMS,
    },

    {
        "name": "LightGBM",
        "function": "fit_lightgbm_regressor",
        "scaled": False,
        "params": LIGHTGBM_PARAMS,
    },


    ########################################
    # SVR
    # Uncomment later
    ########################################

    #{
    #    "name": "SVR Linear",
    #    "function": "fit_svr",
    #    "scaled": True,
    #    "params": {

    #        "kernel": ["linear"],
    #        "C": C_VALUES,

    #        "epsilon": [
    #            # 1e-4,
    #            # 1e-3,
    #            # 1e-2,
    #            # 0.05,
    #            #0.1,
    #            # 0.25,
    #            # 0.5,
    #        ],
    #    },
    #},

    {
        "name": "SVR RBF",
        "function": "fit_svr",
        "scaled": True,
        "params": {

            "kernel": ["rbf"],
            "C": C_VALUES,

            "epsilon": [
                # 1e-4,
                # 1e-3,
                # 1e-2,
                # 0.05,
                #0.1,
                # 0.25,
                # 0.5,
            ],

            "gamma": [
                "scale",
                # "auto",
                # 1e-4,
                # 1e-3,
                # 1e-2,
                # 0.1,
                # 1,
            ],
        },
    },

    ########################################
    # kNN
    # Uncomment later
    ########################################

    {
        "name": "kNN",
        "function": "fit_knn_regressor",
        "scaled": True,
        "params": KNN_PARAMS,
    },


    ########################################
    # MLP
    # Uncomment much later
    ########################################

    #{
    #    "name": "MLP",
    #    "function": "fit_mlp_regressor",
    #    "scaled": True,
    #    "params": MLP_PARAMS,
    #},
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
        "params": {},
    },

    {
        "name": "Logistic Regression",
        "function": "fit_logistic_regression",
        "scaled": True,
        "params": {
            "class_weight": CLASS_WEIGHTS,
        },
    },

    #{
    #    "name": "L2 Logistic Regression",
    #    "function": "fit_l2_logistic_regression",
    #    "scaled": True,
    #    "params": {
    #        "C": C_VALUES,
    #        "class_weight": CLASS_WEIGHTS,
    #    },
    #},

    #{
    #    "name": "L1 Logistic Regression",
    #    "function": "fit_l1_logistic_regression",
    #    "scaled": True,
    #    "params": {
    #        "C": C_VALUES,
    #        "class_weight": CLASS_WEIGHTS,
    #    },
    #},

    #{
    #    "name": "Elastic Net Logistic Regression",
    #    "function": "fit_elastic_net_logistic_regression",
    #    "scaled": True,
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
    #   "name": "Hist Gradient Boosting",
    #   "function": "fit_hist_gradient_boosting_classifier",
    #   "scaled": False,
    #   "params": {
    #       **HIST_GRADIENT_PARAMS,
    #       "class_weight": CLASS_WEIGHTS,
    #   },
    #},


    ########################################
    # Standard Gradient Boosting
    # Uncomment later
    ########################################

    #{
    #    "name": "Gradient Boosting",
    #    "function": "fit_gradient_boosting_classifier",
    #    "scaled": False,
    #    "params": GRADIENT_BOOSTING_PARAMS,
    #},


    #{
    #    "name": "Random Forest",
    #    "function": "fit_random_forest_classifier",
    #    "scaled": False,
    #    "params": {
    #        **RANDOM_FOREST_PARAMS,
    #        "class_weight": CLASS_WEIGHTS,
    #    },
    #},

    {
       "name": "XGBoost",
       "function": "fit_xgboost_classifier",
       "scaled": False,
       "params": {
           **XGBOOST_PARAMS,
           "class_weight": CLASS_WEIGHTS,
       },
    },

    {
        "name": "LightGBM",
        "function": "fit_lightgbm_classifier",
        "scaled": False,
        "params": {
            **LIGHTGBM_PARAMS,
            "class_weight": CLASS_WEIGHTS,
        },
    },


    ########################################
    # SVM
    # Uncomment later
    ########################################

    #{
    #    "name": "SVM Linear",
    #    "function": "fit_svm_classifier",
    #    "scaled": True,
    #    "params": {

    #        "kernel": ["linear"],
    #        "C": C_VALUES,
    #        "class_weight": CLASS_WEIGHTS,
    #    },
    #},

    #{
    #    "name": "SVM RBF",
    #    "function": "fit_svm_classifier",
    #    "scaled": True,
    #    "params": {

    #        "kernel": ["rbf"],
    #        "C": C_VALUES,

    #        "gamma": [
    #            "scale",
    #            # "auto",
    #            # 1e-4,
    #            # 1e-3,
    #            # 1e-2,
    #            # 0.1,
    #            # 1,
    #        ],

    #        "class_weight": CLASS_WEIGHTS,
    #    },
    #},

    ########################################
    # kNN
    # Uncomment later
    ########################################

    {
        "name": "kNN",
        "function": "fit_knn_classifier",
        "scaled": True,
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
        "params": {

            "var_smoothing": [
                # 1e-13,
                # 1e-12,
                # 1e-11,
                # 1e-10,
                1e-9,
                # 1e-8,
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

    #{
    #    "name": "MLP",
    #    "function": "fit_mlp_classifier",
    #    "scaled": True,
    #    "params": MLP_PARAMS,
    #},
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
        "params": {},
    },

    {
        "name": "Multinomial Logistic Regression",
        "function": "fit_multinomial_logistic_regression",
        "scaled": True,
        "params": {
            "class_weight": CLASS_WEIGHTS,
        },
    },

    #{
    #    "name": "L2 Multinomial Logistic Regression",
    #    "function": "fit_l2_multinomial_logistic_regression",
    #    "scaled": True,
    #    "params": {
    #        "C": C_VALUES,
    #        "class_weight": CLASS_WEIGHTS,
    #    },
    #},

    #{
    #    "name": "L1 Multinomial Logistic Regression",
    #    "function": "fit_l1_multinomial_logistic_regression",
    #    "scaled": True,
    #    "params": {
    #        "C": C_VALUES,
    #        "class_weight": CLASS_WEIGHTS,
    #    },
    #},

    #{
    #    "name": "Elastic Net Multinomial Logistic Regression",
    #    "function": "fit_elastic_net_multinomial_logistic_regression",
    #    "scaled": True,
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

    {
        "name": "LDA SVD",
        "function": "fit_lda",
        "scaled": True,
        "params": {

            "solver": ["svd"],
        },
    },

    {
        "name": "LDA LSQR/Eigen",
        "function": "fit_lda",
        "scaled": True,
        "params": {

            "solver": [
                "lsqr",
                "eigen",
            ],

            "shrinkage": [
                # None,
                "auto",
                # 0.05,
                # 0.1,
                # 0.25,
                # 0.5,
                # 0.75,
                # 0.9,
                # 0.95,
            ],
        },
    },


    ########################################
    # QDA
    # Uncomment later
    ########################################

    {
        "name": "QDA",
        "function": "fit_qda",
        "scaled": True,
        "params": {

            "reg_param": [
                0,
                # 0.0001,
                # 0.001,
                # 0.01,
                # 0.05,
                # 0.1,
                # 0.25,
                0.5,
                # 0.75,
                # 1.0,
            ],
        },
    },


    ########################################
    # Tree Models
    ########################################

    #{
    #   "name": "Hist Gradient Boosting",
    #   "function": "fit_hist_gradient_boosting_multiclass",
    #   "scaled": False,
    #   "params": {
    #       **HIST_GRADIENT_PARAMS,
    #       "class_weight": CLASS_WEIGHTS,
    #   },
    #},


    ########################################
    # Standard Gradient Boosting
    # Uncomment later
    ########################################

    #{
    #    "name": "Gradient Boosting",
    #    "function": "fit_gradient_boosting_multiclass",
    #    "scaled": False,
    #    "params": GRADIENT_BOOSTING_PARAMS,
    #},


    #{
    #    "name": "Random Forest",
    #    "function": "fit_random_forest_multiclass",
    #    "scaled": False,
    #    "params": {
    #        **RANDOM_FOREST_PARAMS,
    #        "class_weight": CLASS_WEIGHTS,
    #    },
    #},

    {
        "name": "XGBoost",
        "function": "fit_xgboost_multiclass",
        "scaled": False,
        "params": {
            **XGBOOST_PARAMS,
            "class_weight": CLASS_WEIGHTS,
        },
    },

    {
        "name": "LightGBM",
        "function": "fit_lightgbm_multiclass",
        "scaled": False,
        "params": {
            **LIGHTGBM_PARAMS,
            "class_weight": CLASS_WEIGHTS,
        },
    },


    ########################################
    # SVM
    # Uncomment later
    ########################################

    #{
    #    "name": "SVM Linear",
    #    "function": "fit_svm_multiclass",
    #    "scaled": True,
    #    "params": {

    #        "kernel": ["linear"],
    #        "C": C_VALUES,
    #        "class_weight": CLASS_WEIGHTS,
    #    },
    #},

    #{
    #    "name": "SVM RBF",
    #    "function": "fit_svm_multiclass",
    #    "scaled": True,
    #    "params": {

    #        "kernel": ["rbf"],
    #        "C": C_VALUES,

    #        "gamma": [
    #            "scale",
    #            # "auto",
    #            # 1e-4,
    #            # 1e-3,
    #            # 1e-2,
    #            # 0.1,
    #            # 1,
    #        ],

    #        "class_weight": CLASS_WEIGHTS,
    #    },
    #},


    ########################################
    # MLP
    # Uncomment much later
    ########################################

    #{
    #    "name": "MLP",
    #    "function": "fit_mlp_multiclass",
    #    "scaled": True,
    #    "params": MLP_PARAMS,
    #},


    ########################################
    # Ordinal Regression
    #
    # Currently disabled because
    # fit_ordinal_regression raises
    # NotImplementedError.
    ########################################

    # {
    #     "name": "Ordinal Regression",
    #     "function": "fit_ordinal_regression",
    #     "scaled": True,
    #     "params": {
    #
    #         "alpha": [
    #             # 1e-5,
    #             # 3e-5,
    #             # 1e-4,
    #             # 3e-4,
    #             # 1e-3,
    #             # 3e-3,
    #             # 1e-2,
    #             # 3e-2,
    #             0.1,
    #             # 0.3,
    #             # 1,
    #             # 3,
    #             # 10,
    #             # 30,
    #             # 100,
    #         ],
    #     },
    # },
]


testing_recommendations = input("Would you like to test model recommendations? (Y for YES, anything else for NO)\n")
if testing_recommendations.lower() in ['y','yes']:
    testing_recommendations = True

    try:
        N_RECOMMENDATIONS = int(input("How many models per targets? (Default 3)\n"))
        if N_RECOMMENDATIONS <= 0:
            N_RECOMMENDATIONS = 3
    except ValueError:
        N_RECOMMENDATIONS = 3

else:
    testing_recommendations = False

from itertools import product

def list_of_all_combos(models):

    all_models = []

    for model in models:

        params = model["params"]

        keys = params.keys()

        configurations = [
            dict(zip(keys, values))
            for values in product(*params.values())
        ]

        for config in configurations:

            all_models.append({
                "name": model["name"],
                "params": config
            })


    return all_models


def model_is_fully_used(model, combined_used):

    for used_model in combined_used:

        if model["name"] != used_model["name"]:
            continue

        current_params = model["params"]
        used_params = used_model["params"]

        if all(
            key in used_params
            and set(current_params[key]).issubset(set(used_params[key]))
            for key in current_params
        ):
            return True

    return False


def run_single_fold(
    train_df,
    validation_df,
    features,
    target,
    type,
    fold,
    model
):

    x_train = train_df[features]
    y_train = train_df[target]

    x_validation = validation_df[features]
    y_validation = validation_df[target]


    if model["scaled"]:

        scaler = StandardScaler()

        x_train = scaler.fit_transform(
            x_train
        )

        x_validation = scaler.transform(
            x_validation
        )


    fit_function = globals()[
        model["function"]
    ]


    results = fit_function(
        x_train,
        y_train,
        x_validation,
        y_validation,
        **model["params"]
    )

    results["Model"] = model["name"]
    results["Parameters"] = model["params"]
    results["Target"] = target
    results["Fold"] = fold
    results["Target Type"] = type


    return results


def walk_forward(models_to_do,df,features,target,purge_days,type, previous_results, no_table, validation_window=20):

    logger.info(
        "%s | Starting walk-forward validation | %d models | validation window: %d | purge days: %d",
        target,
        len(models_to_do),
        validation_window,
        purge_days
    )

    pruning_rules = {}


    if not no_table:

        pd.set_option(
            "display.max_columns",
            None
        )

        target_type = (
            previous_results[
                "Target Type"
            ].iloc[0]
        )

        logger.info(
            "Generating pruning rules | "
            "Target type: %s | "
            "Previous configurations: %d",
            target_type,
            len(previous_results)
        )

        pruning_rules = prune_models(
            previous_results,
            target_type
        )


    if any(
        rules
        for rules in pruning_rules.values()
    ):

        rule_count = sum(
            len(rules)
            for rules in pruning_rules.values()
        )

        logger.info(
            "Pruning rules generated | "
            "%d rules across %d model families",
            rule_count,
            sum(
                bool(rules)
                for rules in pruning_rules.values()
            )
        )


        models_before_pruning = len(
            models_to_do
        )


        models_to_do = [
            model
            for model in models_to_do
            if not should_prune_model(
                model,
                pruning_rules
            )
        ]


        models_removed = (
            models_before_pruning
            - len(models_to_do)
        )


        logger.info(
            "Model configuration pruning complete | "
            "Before: %d | "
            "Removed: %d | "
            "Remaining: %d",
            models_before_pruning,
            models_removed,
            len(models_to_do)
        )


    else:

        logger.info(
            "No statistically supported "
            "pruning rules found"
        )


    original_previous_results = previous_results.copy()


    logger.info(
        "%s | Loaded %d previous model configurations",
        target,
        len(previous_results)
    )


    full_train_df, full_validation_df, test_df = train_validation_test_split(df, 0.2,0.2)

    logger.info(
        "%s | Split complete | train rows: %d | validation rows: %d | test rows: %d",
        target,
        len(full_train_df),
        len(full_validation_df),
        len(test_df)
    )

    validation_dates = sorted(
        full_validation_df["Date"].unique()
    )

    logger.info(
        "%s | Validation contains %d unique dates",
        target,
        len(validation_dates)
    )

    validation_results = []

    all_validation_results = []

    try:

        with sqlite3.connect(VALIDATION_DATABASE_PATH) as conn:

            fold_results = pd.read_sql_query(
                f'''
                SELECT *
                FROM "{target}__folds"
                ''',
                conn
            )

    except Exception:

        fold_results = pd.DataFrame()


    fold = 1
    start = 0

    while start < len(validation_dates):

        fold_dates = validation_dates[
            start:start + validation_window
        ]

        if len(fold_dates) == 0:
            break

        logger.info(
            "%s | Fold %d | Validation dates: %s -> %s | %d dates",
            target,
            fold,
            fold_dates[0],
            fold_dates[-1],
            len(fold_dates)
        )

        validation_df = full_validation_df[
            full_validation_df["Date"].isin(
                fold_dates
            )
        ].copy()

        validation_start = fold_dates[0]

        train_df = pd.concat(
            [
                full_train_df,
                full_validation_df[
                    full_validation_df["Date"] < validation_start
                ]
            ]
        ).copy()

        logger.info(
            "%s | Fold %d | Train rows before purge: %d | Validation rows: %d",
            target,
            fold,
            len(train_df),
            len(validation_df)
        )

        if purge_days <= 0:

            logger.info(
                "%s | Fold %d | No purge required",
                target,
                fold
            )

            return train_df

        train_dates = sorted(
            train_df["Date"].unique()
        )

        dates_to_keep = train_dates[
            :-purge_days
        ]

        train_df = train_df[
            train_df["Date"].isin(
                dates_to_keep
            )
        ].copy()

        logger.info(
            "%s | Fold %d | Train rows after purge: %d | Removed final %d training dates",
            target,
            fold,
            len(train_df),
            purge_days
        )

        logger.info(
            "%s | Fold %d | Running %d model configurations",
            target,
            fold,
            len(models_to_do)
        )

        for model_number, model in enumerate(
            models_to_do,
            start=1
        ):

            
            model_fold_result = run_single_fold(
                    train_df,
                    validation_df,
                    features,
                    target,
                    type,
                    fold,
                    model
                )
            

            validation_results.append(
                model_fold_result
            )

            all_validation_results.append(
                model_fold_result
            )

            logger.info(
                "%s | Fold %d | Model %d/%d | %s | Complete",
                target,
                fold,
                model_number,
                len(models_to_do),
                model["name"]
            )

        logger.info(
            "%s | Fold %d | Complete",
            target,
            fold
        )

        if fold in [3,5,7,9,11,14]:

            PRUNING_STAGES = {
                3:  (0.95, 20000),
                5:  (0.9, 5000),
                7:  (0.8, 1000),
                9: (0.65, 300),
                11: (0.5, 70),
                14: (0.5, 15)
            }

            models_to_do = prune(
                models_to_do,
                validation_results,
                fold_results,
                fold,
                target,
                type,
                PRUNING_STAGES[fold][0],
                PRUNING_STAGES[fold][1]
            )

            validation_results = [
                result
                for result in validation_results
                if any(
                    result["Model"] == model["name"]
                    and result["Parameters"] == model["params"]
                    for model in models_to_do
                )
            ]

            if not models_to_do:
                start = len(validation_dates)

        fold += 1
        start += validation_window

    new_results = pd.DataFrame(
        all_validation_results
    )

    new_results["Parameters"] = (
        new_results["Parameters"]
        .apply(lambda x: str(x))
    )

    with sqlite3.connect(
            VALIDATION_DATABASE_PATH
        ) as conn:

            new_results.to_sql(
                f"{target}__folds",
                conn,
                if_exists="append",
                index=False
            )    
    
    metric_columns = [
        column
        for column in new_results.select_dtypes(
            include="number"
        ).columns
        if column not in [
            "Fold"
        ]
    ]
    
    new_results = (
        new_results
        .groupby(
            ["Model", "Parameters"]
        )
        .agg(
            {
                **{
                    column: ["mean", "std"]
                    for column in metric_columns
                },
                "Fold": "max"
            }
        )
    )

    new_results.columns = [
        (
            f"{column} {stat.title()}"
            if column != "Fold"
            else "Fold"
        )
        for column, stat in new_results.columns
    ]

    new_results = new_results.reset_index()

    new_results["Target"] = target
    new_results["Target Type"] = type

    if not(original_previous_results.empty):
        common_columns = original_previous_results.columns.intersection(
            new_results.columns
        )

        new_summary = pd.concat(
                [
                    original_previous_results[common_columns],
                    new_results[common_columns]    
                ],
                ignore_index=True
            )
        
    else:
        new_summary = new_results

    return new_summary
    

def prune(
    models_to_do,
    validation_results,
    fold_results,
    fold,
    target,
    type,
    multiplier,
    maximum_left
):
    
    original_models_to_do = {
        (
            model["name"],
            str(model["params"])
        )
        for model in models_to_do
    }

    current_results = pd.DataFrame(
        validation_results
    )

    current_results["Parameters"] = (
        current_results["Parameters"]
        .apply(lambda x: str(x))
    )

    metric_columns = [
        column
        for column in current_results.select_dtypes(
            include="number"
        ).columns
        if column not in [
            "Fold"
        ]
    ]

    current_results = (
        current_results
        .groupby(
            ["Model", "Parameters"]
        )
        .agg(
            {
                **{
                    column: "mean"
                    for column in metric_columns
                },
                "Fold": "nunique"
            }
        )
        .rename(
            columns={
                **{
                    column: f"{column} Mean"
                    for column in metric_columns
                }
            }
        )
        .reset_index()
    )



    ########################################################
    # PREVIOUS RESULTS + CURRENT NEW RESULTS
    ########################################################

    if fold_results.empty:

        combined_results = (
            current_results.copy()
        )

    else:

        fold_results = fold_results[
            fold_results["Fold"] <= fold
        ].copy()

        fold_results["Parameters"] = (
            fold_results["Parameters"]
            .apply(lambda x: str(x))
        )

        fold_results = (
            fold_results
            .groupby(
                ["Model", "Parameters"]
            )
            .agg(
                {
                    **{
                        column: "mean"
                        for column in metric_columns
                    },
                    "Fold": "nunique"
                }
            )
            .rename(
                columns={
                    **{
                        column: f"{column} Mean"
                        for column in metric_columns
                    }
                }
            )
            .reset_index()
        )

        common_columns = (
            fold_results.columns.intersection(
                current_results.columns
            )
        )

        combined_results = pd.concat(
            [
                fold_results[
                    common_columns
                ],

                current_results[
                    common_columns
                ]
            ],
            ignore_index=True
        )


    if type == "continuous":

        if target.startswith("Future Return Rank"):

            mean_column = "Rank IC Mean"
            higher_is_better = True

        else:

            mean_column = "NRMSE Mean"
            higher_is_better = False


    elif type == "binary":

        mean_column = "ROC AUC Mean"
        higher_is_better = True


    elif type == "multiclass":

        mean_column = "Macro F1 Mean"
        higher_is_better = True


    if higher_is_better:

        combined_results = combined_results.sort_values(
            mean_column,
            ascending=False
        )

    else:

        combined_results = combined_results.sort_values(
            mean_column,
            ascending=True
        )

    keep_count = int(
        np.ceil(
            len(combined_results) * multiplier
        )
    )

    keep_count = min(keep_count, maximum_left)


    better_results = combined_results.head(
        keep_count
    )


    better_models = set(
        zip(
            better_results["Model"],
            better_results["Parameters"]
        )
    )


    ########################################################
    # ONLY FILTER THE CURRENT MODELS_TO_DO
    ########################################################

    new_models_to_do = [
        model
        for model in models_to_do
        if (
            model["name"],
            str(model["params"])
        ) in better_models
    ]


    logger.info(
        "%s | Fold %d pruning | %d -> %d active new configurations",
        target,
        fold,
        len(models_to_do),
        len(new_models_to_do)
    )

    better_results = better_results[
        ~better_results.apply(
            lambda row: (
                row["Model"],
                row["Parameters"]
            ) in original_models_to_do,
            axis=1
        )
    ].copy()

    return new_models_to_do


def rank_validation_results(
    results,
    target,
    target_type
):

    results = results.copy()


    if target_type == "continuous":

        sorting_rules = [
            ("Rank IC Mean", False),
            ("Rank IC Std", True),
            ("NRMSE Mean", True),
            ("RMSE Mean", True),
            ("MAE Mean", True),
            ("R2 Mean", False)
        ]

    elif target_type == "binary":

        sorting_rules = [
            ("ROC AUC Mean", False),
            ("ROC AUC Std", True),
            ("PR AUC Mean", False),
            ("PR AUC Std", True),
            ("Log Loss Mean", True),
            ("F1 Mean", False),
        ]


    elif target_type == "multiclass":

        sorting_rules = [
            ("Macro F1 Mean", False),
            ("Macro F1 Std", True),
            ("Balanced Accuracy Mean", False),
            ("Balanced Accuracy Std", True),
            ("Log Loss Mean", True),
        ]


    else:

        raise ValueError(
            f"Unknown target type: {target_type}"
        )


    sort_columns = []
    ascending = []


    for column, direction in sorting_rules:

        if column in results.columns:

            sort_columns.append(
                column
            )

            ascending.append(
                direction
            )


    results = results.sort_values(
        sort_columns,
        ascending=ascending,
        na_position="last"
    )


    results = results.reset_index(
        drop=True
    )


    if "Rank" in results.columns:

        results = results.drop(
            columns=["Rank"]
        )


    results.insert(
        0,
        "Rank",
        range(
            1,
            len(results) + 1
        )
    )


    return results


def add_testing_eligibility(
    results,
    target_type,
    min_folds
):

    results = results.copy()


    full_validation = (
        results["Fold"]
        >= min_folds
    )


    eligible = pd.Series(
        False,
        index=results.index
    )


    reasons = pd.Series(
        "",
        index=results.index,
        dtype=object
    )


    ########################################################
    # CONTINUOUS
    ########################################################

    if target_type == "continuous":

        rank_ic = (
            results["Rank IC Mean"].abs()
            if "Rank IC Mean" in results.columns
            else pd.Series(
                np.nan,
                index=results.index
            )
        )


        r2 = (
            results["R2 Mean"]
            if "R2 Mean" in results.columns
            else pd.Series(
                np.nan,
                index=results.index
            )
        )


        rank_ic_std = (
            results["Rank IC Std"]
            if "Rank IC Std" in results.columns
            else pd.Series(
                np.nan,
                index=results.index
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
    # BINARY
    ########################################################

    elif target_type == "binary":

        roc_auc = (
            results["ROC AUC Mean"]
            if "ROC AUC Mean" in results.columns
            else pd.Series(
                np.nan,
                index=results.index
            )
        )


        pr_auc = (
            results["PR AUC Mean"]
            if "PR AUC Mean" in results.columns
            else pd.Series(
                np.nan,
                index=results.index
            )
        )


        roc_auc_std = (
            results["ROC AUC Std"]
            if "ROC AUC Std" in results.columns
            else pd.Series(
                np.nan,
                index=results.index
            )
        )


        balanced_accuracy = (
            results["Balanced Accuracy Mean"]
            if "Balanced Accuracy Mean" in results.columns
            else pd.Series(
                1.0,
                index=results.index
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
    # MULTICLASS
    ########################################################

    elif target_type == "multiclass":

        macro_f1 = (
            results["Macro F1 Mean"]
            if "Macro F1 Mean" in results.columns
            else pd.Series(
                np.nan,
                index=results.index
            )
        )


        macro_f1_std = (
            results["Macro F1 Std"]
            if "Macro F1 Std" in results.columns
            else pd.Series(
                np.nan,
                index=results.index
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


    if "Testing Eligible" in results.columns:

        results = results.drop(
            columns=[
                "Testing Eligible"
            ]
        )


    if "Testing Eligibility Reason" in results.columns:

        results = results.drop(
            columns=[
                "Testing Eligibility Reason"
            ]
        )


    results.insert(
        3,
        "Testing Eligible",
        eligible.astype(bool)
    )


    results.insert(
        4,
        "Testing Eligibility Reason",
        reasons
    )


    return results


if 'continuous' not in types:

    logger.info(
        "No completely new continuous targets found; checking previously completed continuous configurations"
    )

    USED_CONTINUOUS_INTERSECTION = intersection_of_used_models(
        USED_CONTINUOUS_MODELS
    )

    COMBINED_USED_CONTINUOUS = combos_to_models(USED_CONTINUOUS_INTERSECTION)

    continuous_before = len(CONTINUOUS_MODELS)

    CONTINUOUS_MODELS = [
        model
        for model in CONTINUOUS_MODELS
        if not model_is_fully_used(
            model,
            COMBINED_USED_CONTINUOUS
        )
    ]

    logger.info(
        "Continuous model families remaining: %d/%d",
        len(CONTINUOUS_MODELS),
        continuous_before
    )


if 'binary' not in types:

    logger.info(
        "No completely new binary targets found; checking previously completed binary configurations"
    )

    USED_BINARY_INTERSECTION = intersection_of_used_models(
        USED_BINARY_MODELS
    )

    COMBINED_USED_BINARY = combos_to_models(USED_BINARY_INTERSECTION)

    binary_before = len(BINARY_MODELS)

    BINARY_MODELS = [
        model
        for model in BINARY_MODELS
        if not model_is_fully_used(
            model,
            COMBINED_USED_BINARY
        )
    ]

    logger.info(
        "Binary model families remaining: %d/%d",
        len(BINARY_MODELS),
        binary_before
    )


if 'multiclass' not in types:

    logger.info(
        "No completely new multiclass targets found; checking previously completed multiclass configurations"
    )

    USED_MULTICLASS_INTERSECTION = intersection_of_used_models(
        USED_MULTICLASS_MODELS
    )

    COMBINED_USED_MULTICLASS = combos_to_models(USED_MULTICLASS_INTERSECTION)

    multiclass_before = len(MULTICLASS_MODELS)

    MULTICLASS_MODELS = [
        model
        for model in MULTICLASS_MODELS
        if not model_is_fully_used(
            model,
            COMBINED_USED_MULTICLASS
        )
    ]

    logger.info(
        "Multiclass model families remaining: %d/%d",
        len(MULTICLASS_MODELS),
        multiclass_before
    )


logger.info("Generating exact parameter combinations")

ALL_BINARY_MODELS = list_of_all_combos(BINARY_MODELS)
ALL_CONTINOUS_MODELS = list_of_all_combos(CONTINUOUS_MODELS)
ALL_MULTICLASS_MODELS = list_of_all_combos(MULTICLASS_MODELS)

logger.info(
    "Configurations available | Continuous: %d | Binary: %d | Multiclass: %d",
    len(ALL_CONTINOUS_MODELS),
    len(ALL_BINARY_MODELS),
    len(ALL_MULTICLASS_MODELS)
)


def main():

    logger.info("Beginning target processing")

    for target_number, target in enumerate(
        targets,
        start=1
    ):

        new_target = False

        logger.info(
            "[%d/%d] Target: %s | Starting",
            target_number,
            len(targets),
            target
        )

        if target in USED_CONTINUOUS_MODELS.keys():

            _type = 'continuous'

            MODELS_TO_DO = [
                model
                for model in ALL_CONTINOUS_MODELS
                if model not in USED_CONTINUOUS_MODELS[target]
            ]

        elif target in USED_BINARY_MODELS.keys():

            _type = 'binary'

            MODELS_TO_DO = [
                model
                for model in ALL_BINARY_MODELS
                if model not in USED_BINARY_MODELS[target]
            ]

        elif target in USED_MULTICLASS_MODELS.keys():

            _type = 'multiclass'

            MODELS_TO_DO = [
                model
                for model in ALL_MULTICLASS_MODELS
                if model not in USED_MULTICLASS_MODELS[target]
            ]

        else:
            new_target = True

            logger.info(
                "Target: %s | No previous validation results found | New target",
                target
            )

        if not new_target:

            logger.info(
                "Target: %s | Existing %s target | %d model configurations remaining",
                target,
                _type,
                len(MODELS_TO_DO)
            )

            if not(MODELS_TO_DO) and not(testing_recommendations):

                logger.info(
                    "Target: %s | No new model configurations to test | Skipping",
                    target
                )

                continue

        ########################################################
        # LOAD PREVIOUS RESULTS ONCE
        ########################################################

        try:

            with sqlite3.connect(VALIDATION_DATABASE_PATH) as conn:

                previous_results = pd.read_sql_query(
                    f'''
                    SELECT *
                    FROM "{target}__search"
                    ''',
                    conn
                )

                no_table = False

        except Exception:

            previous_results = pd.DataFrame()

            no_table = True


        features = selected_features[target]

        columns = ["Date","Ticker", target]
        columns.extend(features)

        logger.info(
            "Target: %s | Loading %d selected features",
            target,
            len(features)
        )

        with sqlite3.connect(FEATURE_DATABASE_PATH) as conn:

            columns = ", ".join(
                f'"{column}"'
                for column in columns
            )

            df = pd.read_sql_query(
                f'''
                SELECT {columns}
                FROM "{STOCK_TYPE}"
                ''',
                conn
            )

        logger.info(
            "Target: %s | Data loaded | Rows: %d | Columns: %d",
            target,
            len(df),
            len(df.columns)
        )

        if new_target:

            _type = target_type(df, target)

            logger.info(
                "Target: %s | Detected target type: %s",
                target,
                _type
            )

        else:

            logger.info(
                "Target: %s | Target type: %s",
                target,
                _type
            )

        if new_target:
            if _type == "continuous":
                MODELS_TO_DO = ALL_CONTINOUS_MODELS
            elif _type == "binary":
                MODELS_TO_DO = ALL_BINARY_MODELS
            elif _type == "multiclass":
                MODELS_TO_DO = ALL_MULTICLASS_MODELS

        if testing_recommendations and not(no_table):
            recommended_models = recommend_models_to_fit(
                previous_results,
                _type,
                n=N_RECOMMENDATIONS
            )

            if not(MODELS_TO_DO):
                MODELS_TO_DO = recommended_models
            else:
                MODELS_TO_DO.extend(recommended_models)

        
        if (MODELS_TO_DO):
            logger.info(
                "Target: %s | Ready to test %d configurations",
                target,
                len(MODELS_TO_DO)
            )
        else:
            logger.info(
                "Target: %s | No configurations to test",
                target
            )

            continue

        purge_days = target_purge_days(
            target
        )

        model_source = full_model_source(_type)

        model_lookup = {
            model["name"]: model
            for model in model_source
        }


        for model in MODELS_TO_DO:

            original_model = model_lookup[
                model["name"]
            ]

            model["function"] = original_model[
                "function"
            ]

            model["scaled"] = original_model[
                "scaled"
            ]

        df = df.dropna()

        new_summary = (
            walk_forward(

                models_to_do=MODELS_TO_DO,

                df=df,

                features=features,

                target=target,

                purge_days=purge_days,

                type=_type,

                previous_results=previous_results, 

                no_table=no_table,

                validation_window=20
            )
        )

        new_summary = rank_validation_results(
            new_summary,
            target,
            _type
        )

        new_summary = add_testing_eligibility(
            new_summary,
            _type,
            15
        )

        with sqlite3.connect(
            VALIDATION_DATABASE_PATH
        ) as conn:

            new_summary.to_sql(
                f"{target}__search",
                conn,
                if_exists="replace",
                index=False
            )

        test_eligible_results = new_summary[
            new_summary["Testing Eligible"]
        ].copy()

        with sqlite3.connect(
            VALIDATION_DATABASE_PATH
        ) as conn:

            test_eligible_results.to_sql(
                f"{target}",
                conn,
                if_exists="replace",
                index=False
            )
 


if (not MULTICLASS_MODELS) and (not BINARY_MODELS) and (not CONTINUOUS_MODELS) and not(testing_recommendations):

    logger.info(
        "No new continuous, binary, or multiclass model families remain to be tested"
    )

else:

    if not MULTICLASS_MODELS:
        logger.info(
            "No new multiclass model families to test"
        )

    if not BINARY_MODELS:
        logger.info(
            "No new binary model families to test"
        )

    if not CONTINUOUS_MODELS:
        logger.info(
            "No new continuous model families to test"
        )

    logger.info("At least one model type has configurations remaining to test")

    main()
    


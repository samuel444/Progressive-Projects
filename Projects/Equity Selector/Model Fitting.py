"""Edit SETTINGS here, then run this script. CLI path/log options override these values.
None retains optional defaults; required cache dates must be set here. Packages never prompt.
"""

from equity_selector.cli import run_stage

SETTINGS = {
    # legacy preserves the original explicit lists; grid uses the *_PARAMS grids below.
    "MODEL_CATALOGUE_MODE": "grid",
    "DATA_DIR": "data/extensive_20260904/model_research",
    "LOG_LEVEL": "INFO",
    "STOCK_TYPE": "High Liquidity 30",
    "MAX_CONTINUOUS_RANK_IC_STD": 0.1,
    "MAX_BINARY_ROC_AUC_STD": 0.07,
    "MAX_MULTICLASS_MACRO_F1_STD": 0.1,
    "RIDGE_ALPHAS": [0.01, 0.1, 1, 10, 100],
    "SPARSE_ALPHAS": [1e-05, 0.0001, 0.001, 0.01],
    "C_VALUES": [0.1, 1, 10],
    "L1_RATIOS": [0.25, 0.75],
    "LEARNING_RATES": [0.03, 0.1],
    "CLASS_WEIGHTS": [None, "balanced"],
    "RANDOM_FOREST_PARAMS": {
        "n_estimators": [300],
        "max_depth": [3, 6],
        "min_samples_leaf": [20, 50],
        "min_samples_split": [40],
        "max_features": ["sqrt"],
        "bootstrap": [True],
    },
    "XGBOOST_PARAMS": {
        "n_estimators": [300],
        "learning_rate": [0.03, 0.1],
        "max_depth": [2, 4],
        "min_child_weight": [10],
        "subsample": [0.8],
        "colsample_bytree": [0.8],
        "gamma": [0],
        "reg_alpha": [0],
        "reg_lambda": [10],
    },
    "LIGHTGBM_PARAMS": {
        "n_estimators": [300],
        "learning_rate": [0.03, 0.1],
        "num_leaves": [7, 15],
        "max_depth": [-1],
        "min_child_samples": [50],
        "subsample": [0.8],
        "colsample_bytree": [0.8],
        "reg_alpha": [0],
        "reg_lambda": [10],
    },
    "MLP_PARAMS": {
        "hidden_layer_sizes": [(32,)],
        "activation": ["relu"],
        "alpha": [0.01],
        "learning_rate_init": [0.001],
        "batch_size": [128],
    },
    "KNN_PARAMS": {"n_neighbors": [20, 50, 100], "weights": ["distance"], "p": [2]},
    "ALL_CONTINUOUS_MODELS": None,
    "ALL_BINARY_MODELS": None,
    "ALL_MULTICLASS_MODELS": None,
    "TEST_RECOMMENDATIONS": False,
    "N_RECOMMENDATIONS": 5,
    "N_MODELS": 3,
    "PRUNING_STAGES": {
        3: (0.95, 20000),
        5: (0.90, 5000),
        7: (0.80, 1000),
        9: (0.65, 300),
        11: (0.50, 70),
        14: (0.50, 15),
    },
    "VALIDATION_WINDOW": 63,
    "MIN_VALIDATION_FOLDS": 15,
    "RESEARCH_START": "2000-01-01",
    "RESEARCH_END": "2015-12-31",
    "MODEL_TRAIN_END": "2006-12-31",
    "MODEL_VALIDATION_END": "2012-12-31",
}

# Optional named helper replacements with the same signature; normally leave empty.
CALLBACKS = {}

if __name__ == "__main__":
    run_stage("training", settings=SETTINGS, callbacks=CALLBACKS)

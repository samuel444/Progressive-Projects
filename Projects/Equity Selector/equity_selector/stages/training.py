from equity_selector.validation import research_rows, required_validation_folds
from equity_selector.settings import setting as get_setting, callback, choose_catalogue

"""training: original sequential research stage; shared logic is in equity_selector."""

from equity_selector.config import data_root
from models import *
from main_package import *


def run():
    global \
        ALL_BINARY_MODELS, \
        ALL_CONTINUOUS_MODELS, \
        ALL_MULTICLASS_MODELS, \
        BINARY_MODELS, \
        CLASS_WEIGHTS, \
        COMBINED_USED_BINARY, \
        COMBINED_USED_CONTINUOUS, \
        COMBINED_USED_MULTICLASS, \
        CONTINUOUS_MODELS, \
        C_VALUES, \
        FEATURE_DATABASE_PATH, \
        GRADIENT_BOOSTING_PARAMS, \
        HIST_GRADIENT_PARAMS, \
        KNN_PARAMS, \
        L1_RATIOS, \
        LEARNING_RATES, \
        LIGHTGBM_PARAMS, \
        MAX_BINARY_ROC_AUC_STD, \
        MAX_CONTINUOUS_RANK_IC_STD, \
        MAX_MULTICLASS_MACRO_F1_STD, \
        MLP_PARAMS, \
        MODELS_TO_DO, \
        MULTICLASS_MODELS, \
        N_MODELS, \
        N_RECOMMENDATIONS, \
        PORTFOLIO_DIRECTION_TYPES, \
        PORTFOLIO_OPPORTUNITY_TYPES, \
        PORTFOLIO_RANKING_TYPES, \
        PORTFOLIO_RISK_TYPES, \
        PORTFOLIO_SPECIAL_TYPES, \
        RANDOM_FOREST_PARAMS, \
        RIDGE_ALPHAS, \
        SPARSE_ALPHAS, \
        STOCK_TYPE, \
        StandardScaler, \
        TARGET_PORTFOLIO_TYPES, \
        USED_BINARY_INTERSECTION, \
        USED_BINARY_MODELS, \
        USED_CONTINUOUS_INTERSECTION, \
        USED_CONTINUOUS_MODELS, \
        USED_MULTICLASS_INTERSECTION, \
        USED_MULTICLASS_MODELS, \
        VALIDATION_DATABASE_PATH, \
        XGBOOST_PARAMS, \
        _, \
        _type, \
        add_model_selection_score, \
        add_testing_eligibility, \
        ast, \
        binary_before, \
        column, \
        columns, \
        combos_to_models, \
        conn, \
        continuous_before, \
        df, \
        file, \
        i, \
        intersection_of_used_models, \
        json, \
        list_of_all_combos, \
        logger, \
        logging, \
        main, \
        missing_targets, \
        model, \
        model_is_fully_used, \
        multiclass_before, \
        no_features_targets, \
        np, \
        parameters, \
        params_str, \
        pd, \
        portfolio_selection_role, \
        portfolio_target_type, \
        portfolio_type_counts, \
        product, \
        prune, \
        rank_validation_results, \
        re, \
        row, \
        run_single_fold, \
        selected_features, \
        sqlite3, \
        stock_type_index, \
        table_names, \
        table_targets, \
        target, \
        targets, \
        testing_recommendations, \
        types, \
        unique_models, \
        used_models, \
        walk_forward, \
        write_frame
    from equity_selector.database import write_frame
    from equity_selector.parameters import unique_models
    import sqlite3
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    import json
    import numpy as np
    import re
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logger = logging.getLogger(__name__)

    def portfolio_target_type(target, prediction_type=None):
        name = str(target).strip().lower()
        for suffix in ("__ranking", "__level", "__direction", "__risk", "__opportunity"):
            if name.endswith(suffix):
                name = name[: -len(suffix)].strip()
                break
        prediction_type = str(prediction_type or "").strip().lower()
        if "market impact" in name or "price impact" in name:
            return "MARKET_IMPACT"
        if (
            "execution" in name
            or "fill probability" in name
            or "fill rate" in name
            or ("slippage" in name)
        ):
            return "EXECUTION"
        if (
            "liquidity" in name
            or "bid ask spread" in name
            or "bid-ask spread" in name
            or ("order book depth" in name)
            or ("order-book depth" in name)
        ):
            return "LIQUIDITY"
        if "covariance" in name:
            return "COVARIANCE"
        if "correlation" in name:
            return "CORRELATION"
        if "regime" in name:
            return "REGIME"
        if "recovery" in name or "recover" in name or "bounce back" in name:
            return "RECOVERY"
        if (
            "reversal" in name
            or "reverse" in name
            or "mean reversion" in name
            or ("mean-reversion" in name)
        ):
            return "REVERSAL"
        if name.startswith("volatility barrier"):
            return "VOLATILITY_EVENT"
        if (
            "volatility event" in name
            or "volatility spike" in name
            or "volatility breakout" in name
        ):
            return "VOLATILITY_EVENT"
        if (
            "sudden drawdown" in name
            or "crash" in name
            or "tail event" in name
            or ("extreme downside" in name)
            or ("downside event" in name)
            or ("negative spike" in name)
        ):
            return "TAIL_EVENT"
        if (
            "upside spike" in name
            or "positive spike" in name
            or "upside event" in name
            or ("positive event" in name)
        ):
            return "UPSIDE_EVENT"
        if name.startswith("time to maximum favourable excursion"):
            return "TIME_TO_UPSIDE_EXCURSION"
        if name.startswith("time to maximum adverse excursion"):
            return "TIME_TO_DOWNSIDE_EXCURSION"
        if name.startswith("maximum favourable excursion"):
            return "UPSIDE_EXCURSION"
        if name.startswith("maximum adverse excursion"):
            return "DOWNSIDE_EXCURSION"
        if (
            name.startswith("future maximum drawdown")
            or "expected shortfall" in name
            or "conditional value at risk" in name
            or ("conditional var" in name)
            or ("cvar" in name)
            or ("value at risk" in name)
            or re.search("\\bvar\\b", name)
            or ("tail risk" in name)
        ):
            return "TAIL_RISK"
        if (
            name.startswith("future minimum return")
            or "minimum return" in name
            or "min return" in name
        ):
            return "DOWNSIDE"
        if "downside upside volatility ratio" in name:
            return "VOLATILITY_ASYMMETRY"
        if "downside volatility" in name or "downside deviation" in name:
            return "DOWNSIDE_VOLATILITY"
        if "upside volatility" in name or "positive volatility" in name:
            return "UPSIDE_VOLATILITY"
        if (
            "return volatility ratio" in name
            or "sortino ratio" in name
            or "sharpe" in name
            or ("calmar" in name)
            or ("return minus risk" in name)
            or ("return drawdown ratio" in name)
            or ("risk adjusted" in name)
            or ("risk-adjusted" in name)
        ):
            return "RISK_ADJUSTED_ALPHA"
        if (
            "mean absolute return" in name
            or "maximum absolute return" in name
            or "max absolute return" in name
            or ("absolute return" in name)
        ):
            return "ABSOLUTE_MOVE"
        if name.startswith("future variance") or "variance" in name:
            return "VOLATILITY"
        if "volatility" in name:
            return "VOLATILITY"
        if (
            "bottom 20 percent future return" in name
            or "bottom 25 percent future return" in name
            or "bottom quintile" in name
            or ("bottom quartile" in name)
        ):
            return "CROSS_SECTION_DOWNSIDE"
        if (
            "top 20 percent future return" in name
            or "top 25 percent future return" in name
            or "top 10 percent future return" in name
            or ("top quintile" in name)
            or ("top quartile" in name)
            or ("future return rank" in name)
            or ("return rank" in name)
            or ("return percentile" in name)
            or ("return quantile" in name)
            or ("cross sectional" in name)
            or ("cross-sectional" in name)
        ):
            return "CROSS_SECTION_ALPHA"
        if (
            "excess return" in name
            or "relative return" in name
            or "abnormal return" in name
            or ("residual return" in name)
            or ("benchmark return" in name)
        ):
            return "RELATIVE_ALPHA"
        if name.startswith("three class direction"):
            return "DIRECTION_MULTICLASS"
        if name.startswith("future direction"):
            return "DIRECTION"
        if name.startswith("barrier"):
            return "BARRIER_ALPHA"
        if name.startswith("future return above"):
            return "ALPHA_BINARY"
        if name.startswith("forward return") or name.startswith("forward log return"):
            return "ALPHA"
        if "alpha" in name or "momentum" in name:
            return "ALPHA"
        if prediction_type == "volatility":
            return "VOLATILITY"
        if prediction_type == "downside":
            return "DOWNSIDE"
        raise ValueError(f"Could not determine Portfolio Target Type for target: {target!r}")

    portfolio_target_type = callback("portfolio_target_type", portfolio_target_type)

    PORTFOLIO_RANKING_TYPES = {
        "ALPHA",
        "RELATIVE_ALPHA",
        "RISK_ADJUSTED_ALPHA",
        "CROSS_SECTION_ALPHA",
    }
    PORTFOLIO_DIRECTION_TYPES = {
        "DIRECTION",
        "DIRECTION_MULTICLASS",
        "ALPHA_BINARY",
        "BARRIER_ALPHA",
    }
    PORTFOLIO_RISK_TYPES = {
        "VOLATILITY",
        "DOWNSIDE_VOLATILITY",
        "VOLATILITY_ASYMMETRY",
        "DOWNSIDE",
        "TAIL_RISK",
        "TAIL_EVENT",
        "DOWNSIDE_EXCURSION",
        "VOLATILITY_EVENT",
        "CROSS_SECTION_DOWNSIDE",
    }
    PORTFOLIO_OPPORTUNITY_TYPES = {
        "ABSOLUTE_MOVE",
        "UPSIDE_VOLATILITY",
        "UPSIDE_EVENT",
        "UPSIDE_EXCURSION",
        "RECOVERY",
        "REVERSAL",
    }
    PORTFOLIO_SPECIAL_TYPES = {
        "TIME_TO_DOWNSIDE_EXCURSION",
        "TIME_TO_UPSIDE_EXCURSION",
        "EXECUTION",
        "LIQUIDITY",
        "MARKET_IMPACT",
        "CORRELATION",
        "COVARIANCE",
        "REGIME",
    }

    def portfolio_selection_role(portfolio_type):
        if portfolio_type in PORTFOLIO_RANKING_TYPES:
            return "RANKING"
        if portfolio_type in PORTFOLIO_DIRECTION_TYPES:
            return "DIRECTION"
        if portfolio_type in PORTFOLIO_RISK_TYPES:
            return "RISK"
        if portfolio_type in PORTFOLIO_OPPORTUNITY_TYPES:
            return "OPPORTUNITY"
        if portfolio_type in PORTFOLIO_SPECIAL_TYPES:
            return "SPECIAL"
        raise ValueError(f"Unknown Portfolio Target Type: {portfolio_type}")

    portfolio_selection_role = callback("portfolio_selection_role", portfolio_selection_role)

    STOCK_TYPE = get_setting("STOCK_TYPE", "High Liquidity 30")
    MAX_CONTINUOUS_RANK_IC_STD = get_setting("MAX_CONTINUOUS_RANK_IC_STD", 0.1)
    MAX_BINARY_ROC_AUC_STD = get_setting("MAX_BINARY_ROC_AUC_STD", 0.07)
    MAX_MULTICLASS_MACRO_F1_STD = get_setting("MAX_MULTICLASS_MACRO_F1_STD", 0.1)
    MODELS_TO_DO = []
    logger.info("Starting validation model fitting")
    logger.info("Stock type: %s", STOCK_TYPE)
    from equity_selector.feature_mapping import load_feature_mapping

    selected_features = load_feature_mapping(data_root() / "Selected_Features.txt", STOCK_TYPE)
    targets = list(selected_features.keys())
    no_features_targets = [target for target in targets if not selected_features[target]]
    targets = [target for target in targets if target not in no_features_targets]
    if not targets:
        logger.info("No targets with selected features remain")
        return
    TARGET_PORTFOLIO_TYPES = {target: portfolio_target_type(target) for target in targets}
    portfolio_type_counts = pd.Series(TARGET_PORTFOLIO_TYPES).value_counts().sort_index()
    logger.info(
        "Loaded %d usable targets | %d targets removed because they have no selected features",
        len(targets),
        len(no_features_targets),
    )
    logger.info("Portfolio target types assigned | %s", portfolio_type_counts.to_dict())
    VALIDATION_DATABASE_PATH = (
        f"{str(data_root()) + '/Validation_Model_Fits/'}{STOCK_TYPE.replace(' ', '_')}.db"
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
        return [{"name": name, "params": params} for name, params in models.items()]

    combos_to_models = callback("combos_to_models", combos_to_models)

    def intersection_of_used_models(used_models):
        if not used_models:
            return []
        values = list(used_models.values())
        intersection = [
            model
            for model in values[0]
            if all((model in target_models for target_models in values[1:]))
        ]
        return intersection

    intersection_of_used_models = callback(
        "intersection_of_used_models", intersection_of_used_models
    )

    logger.info("Reading existing validation tables")
    with sqlite3.connect(VALIDATION_DATABASE_PATH) as conn:
        table_names = [
            row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table';")
        ]
    logger.info("Found %d validation database tables", len(table_names))
    for i in range(len(table_names)):
        table_names[i] = table_names[i].removesuffix("__search")
        table_names[i] = table_names[i].removesuffix("__folds")
    table_targets = list(set(table_names))
    table_targets = [target for target in table_targets if target not in no_features_targets]
    missing_targets = [target for target in targets if target not in table_targets]
    logger.info(
        "%d/%d targets already have validation tables",
        len(targets) - len(missing_targets),
        len(targets),
    )
    logger.info("%d targets have no existing validation tables", len(missing_targets))
    types = []
    FEATURE_DATABASE_PATH = str(data_root()) + "/Features_Targets_Data.db"
    if missing_targets:
        with sqlite3.connect(FEATURE_DATABASE_PATH) as conn:
            columns = ", ".join((f'"{column}"' for column in ["Date", *missing_targets]))
            df = research_rows(
                pd.read_sql_query(
                    f'\n            SELECT {columns}\n            FROM "{STOCK_TYPE}"\n            ',
                    conn,
                )
            )
    for target in missing_targets:
        types.append(target_type(df, target))
    types = list(set(types))
    logger.info("Target types with new targets: %s", types)
    import ast

    USED_CONTINUOUS_MODELS = {}
    USED_BINARY_MODELS = {}
    USED_MULTICLASS_MODELS = {}
    logger.info("Loading previously tested model configurations")
    with sqlite3.connect(VALIDATION_DATABASE_PATH) as conn:
        for target in targets:
            try:
                df = research_rows(
                    pd.read_sql_query(
                        f'\n                SELECT "Target", "Target Type", "Model", "Parameters"\n                FROM "{target}__search"\n                ',
                        conn,
                    )
                )
            except Exception:
                logger.debug("No existing search table found for target: %s", target)
                continue
            if df.empty:
                logger.debug("Search table is empty for target: %s", target)
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
                from equity_selector.parameters import parse_parameters

                parameters = parse_parameters(row["Parameters"])
                used_models[target].append({"name": row["Model"], "params": parameters})
    RIDGE_ALPHAS = get_setting(
        "RIDGE_ALPHAS", [1e-05, 3e-05, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 3000, 10000]
    )
    SPARSE_ALPHAS = get_setting(
        "SPARSE_ALPHAS", [1e-08, 1e-07, 1e-06, 1e-05, 0.0001, 0.001, 0.01, 0.1, 0.3, 1, 3, 10]
    )
    C_VALUES = get_setting("C_VALUES", [0.1])
    L1_RATIOS = get_setting("L1_RATIOS", [0.5])
    LEARNING_RATES = get_setting("LEARNING_RATES", [0.05])
    CLASS_WEIGHTS = get_setting("CLASS_WEIGHTS", [None, "balanced"])
    HIST_GRADIENT_PARAMS = {
        "learning_rate": LEARNING_RATES,
        "max_iter": [200],
        "max_leaf_nodes": [31],
        "max_depth": [5],
        "min_samples_leaf": [20],
        "l2_regularization": [0.1],
    }
    GRADIENT_BOOSTING_PARAMS = {
        "learning_rate": LEARNING_RATES,
        "n_estimators": [200],
        "max_depth": [3],
        "min_samples_leaf": [20],
        "subsample": [0.8],
        "max_features": ["sqrt"],
    }
    RANDOM_FOREST_PARAMS = get_setting(
        "RANDOM_FOREST_PARAMS",
        {
            "n_estimators": [200],
            "max_depth": [10],
            "min_samples_leaf": [5],
            "min_samples_split": [5],
            "max_features": ["sqrt"],
            "bootstrap": [True],
        },
    )
    XGBOOST_PARAMS = get_setting(
        "XGBOOST_PARAMS",
        {
            "n_estimators": [200],
            "learning_rate": LEARNING_RATES,
            "max_depth": [3],
            "min_child_weight": [1],
            "subsample": [0.8],
            "colsample_bytree": [0.75],
            "gamma": [0],
            "reg_alpha": [0],
            "reg_lambda": [1],
        },
    )
    LIGHTGBM_PARAMS = get_setting(
        "LIGHTGBM_PARAMS",
        {
            "n_estimators": [200, 300],
            "learning_rate": LEARNING_RATES,
            "num_leaves": [7, 15],
            "max_depth": [-1],
            "min_child_samples": [10, 20],
            "subsample": [0.8],
            "colsample_bytree": [0.75],
            "reg_alpha": [0, 0.1],
            "reg_lambda": [1],
        },
    )
    MLP_PARAMS = get_setting(
        "MLP_PARAMS",
        {
            "hidden_layer_sizes": [(64,)],
            "activation": ["relu"],
            "alpha": [0.0001],
            "learning_rate_init": [0.001],
            "batch_size": [128],
        },
    )
    KNN_PARAMS = get_setting(
        "KNN_PARAMS",
        {
            "n_neighbors": [3, 5, 10, 20, 40, 80, 150, 250],
            "weights": ["uniform", "distance"],
            "p": [1, 2],
        },
    )
    CONTINUOUS_MODELS = [
        {"name": "Mean Baseline", "function": "fit_mean_baseline", "scaled": False, "params": {}},
        {"name": "OLS", "function": "fit_ols", "scaled": True, "params": {}},
        {
            "name": "Ridge",
            "function": "fit_ridge",
            "scaled": True,
            "params": {"alpha": RIDGE_ALPHAS},
        },
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
        {
            "name": "SVR RBF",
            "function": "fit_svr",
            "scaled": True,
            "params": {"kernel": ["rbf"], "C": C_VALUES, "epsilon": [], "gamma": ["scale"]},
        },
        {"name": "kNN", "function": "fit_knn_regressor", "scaled": True, "params": KNN_PARAMS},
    ]
    BINARY_MODELS = [
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
            "params": {"class_weight": CLASS_WEIGHTS},
        },
        {
            "name": "XGBoost",
            "function": "fit_xgboost_classifier",
            "scaled": False,
            "params": {**XGBOOST_PARAMS, "class_weight": CLASS_WEIGHTS},
        },
        {
            "name": "LightGBM",
            "function": "fit_lightgbm_classifier",
            "scaled": False,
            "params": {**LIGHTGBM_PARAMS, "class_weight": CLASS_WEIGHTS},
        },
        {"name": "kNN", "function": "fit_knn_classifier", "scaled": True, "params": KNN_PARAMS},
        {
            "name": "Naive Bayes",
            "function": "fit_naive_bayes",
            "scaled": False,
            "params": {"var_smoothing": [1e-09, 1e-06]},
        },
    ]
    MULTICLASS_MODELS = [
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
            "params": {"class_weight": CLASS_WEIGHTS},
        },
        {"name": "LDA SVD", "function": "fit_lda", "scaled": True, "params": {"solver": ["svd"]}},
        {
            "name": "LDA LSQR/Eigen",
            "function": "fit_lda",
            "scaled": True,
            "params": {"solver": ["lsqr", "eigen"], "shrinkage": ["auto"]},
        },
        {"name": "QDA", "function": "fit_qda", "scaled": True, "params": {"reg_param": [0, 0.5]}},
        {
            "name": "XGBoost",
            "function": "fit_xgboost_multiclass",
            "scaled": False,
            "params": {**XGBOOST_PARAMS, "class_weight": CLASS_WEIGHTS},
        },
        {
            "name": "LightGBM",
            "function": "fit_lightgbm_multiclass",
            "scaled": False,
            "params": {**LIGHTGBM_PARAMS, "class_weight": CLASS_WEIGHTS},
        },
    ]
    testing_recommendations = get_setting("TEST_RECOMMENDATIONS", False)
    N_RECOMMENDATIONS = get_setting("N_RECOMMENDATIONS", 3)
    N_MODELS = get_setting("N_MODELS", 5)
    if not isinstance(testing_recommendations, bool) or N_RECOMMENDATIONS <= 0 or N_MODELS <= 0:
        raise ValueError("Recommendation flag must be boolean and counts positive")
    from itertools import product

    def list_of_all_combos(models):
        all_models = []
        for model in models:
            params = model["params"]
            keys = params.keys()
            configurations = [dict(zip(keys, values)) for values in product(*params.values())]
            for config in configurations:
                all_models.append({"name": model["name"], "params": config})
        return all_models

    list_of_all_combos = callback("list_of_all_combos", list_of_all_combos)

    def model_is_fully_used(model, combined_used):
        for used_model in combined_used:
            if model["name"] != used_model["name"]:
                continue
            current_params = model["params"]
            used_params = used_model["params"]
            if all(
                (
                    key in used_params and set(current_params[key]).issubset(set(used_params[key]))
                    for key in current_params
                )
            ):
                return True
        return False

    model_is_fully_used = callback("model_is_fully_used", model_is_fully_used)

    def run_single_fold(train_df, validation_df, features, target, type, fold, model):
        train_df = train_df.replace([np.inf, -np.inf], np.nan).dropna(subset=features + [target])
        validation_df = validation_df.replace([np.inf, -np.inf], np.nan).dropna(
            subset=features + [target]
        )
        if train_df.empty or validation_df.empty:
            return None
        x_train = train_df[features]
        y_train = train_df[target]
        x_validation = validation_df[features]
        y_validation = validation_df[target]
        if model["scaled"]:
            scaler = StandardScaler()
            x_train = scaler.fit_transform(x_train)
            x_validation = scaler.transform(x_validation)
        fit_function = globals()[model["function"]]
        results = fit_function(x_train, y_train, x_validation, y_validation, **model["params"])
        if results != None:
            results["Model"] = model["name"]
            results["Parameters"] = model["params"]
            results["Target"] = target
            results["Fold"] = fold
            results["Target Type"] = type
        return results

    run_single_fold = callback("run_single_fold", run_single_fold)

    def walk_forward(
        models_to_do,
        df,
        features,
        target,
        purge_days,
        type,
        portfolio_type,
        previous_results,
        no_table,
        validation_window=20,
    ):
        from equity_selector.training import walk_forward as validate

        return validate(
            models_to_do,
            df,
            features,
            target,
            purge_days,
            type,
            portfolio_type,
            previous_results,
            no_table,
            validation_window,
            database_path=VALIDATION_DATABASE_PATH,
            fit_function=run_single_fold,
            prune_function=prune,
            role_function=portfolio_selection_role,
        )

    walk_forward = callback("walk_forward", walk_forward)

    def add_model_selection_score(results, statistical_type, portfolio_type, target=None):
        results = results.copy()
        role = portfolio_selection_role(portfolio_type)
        target_name = str(target or "").strip().lower()
        if target_name.endswith("__ranking"):
            selection_objective = "RANKING_ONLY"
        elif target_name.endswith("__level"):
            selection_objective = "LEVEL_ONLY"
        else:
            selection_objective = "PORTFOLIO_ROLE"
        if statistical_type == "continuous":
            if selection_objective == "RANKING_ONLY":
                metric_weights = [("Rank IC Mean", 1.0, True)]
            elif selection_objective == "LEVEL_ONLY":
                metric_weights = [("NRMSE Mean", 1.0, False)]
            elif role == "RANKING":
                metric_weights = [("Rank IC Mean", 0.85, True), ("NRMSE Mean", 0.15, False)]
            elif role in {"RISK", "OPPORTUNITY"}:
                metric_weights = [("Rank IC Mean", 0.7, True), ("NRMSE Mean", 0.3, False)]
            else:
                metric_weights = [("Rank IC Mean", 0.55, True), ("NRMSE Mean", 0.45, False)]
        elif statistical_type == "binary":
            if role == "DIRECTION":
                metric_weights = [
                    ("ROC AUC Mean", 0.35, True),
                    ("PR AUC Mean", 0.3, True),
                    ("Balanced Accuracy Mean", 0.2, True),
                    ("F1 Mean", 0.1, True),
                    ("Log Loss Mean", 0.05, False),
                ]
            else:
                metric_weights = [
                    ("ROC AUC Mean", 0.45, True),
                    ("PR AUC Mean", 0.25, True),
                    ("Balanced Accuracy Mean", 0.15, True),
                    ("F1 Mean", 0.1, True),
                    ("Log Loss Mean", 0.05, False),
                ]
        elif statistical_type == "multiclass":
            metric_weights = [
                ("Macro F1 Mean", 0.5, True),
                ("Balanced Accuracy Mean", 0.3, True),
                ("Log Loss Mean", 0.2, False),
            ]
        else:
            raise ValueError(f"Unknown target type: {statistical_type}")
        weighted_score = pd.Series(0.0, index=results.index, dtype=float)
        available_weight = pd.Series(0.0, index=results.index, dtype=float)
        metrics_used = []
        for column, weight, higher_is_better in metric_weights:
            if column not in results.columns:
                continue
            values = pd.to_numeric(results[column], errors="coerce")
            valid = values.notna()
            percentile_score = values.rank(method="average", pct=True, ascending=higher_is_better)
            weighted_score.loc[valid] += percentile_score.loc[valid] * weight
            available_weight.loc[valid] += weight
            metrics_used.append(column)
        results["Model Selection Score"] = weighted_score.div(available_weight.replace(0.0, np.nan))
        if not metrics_used:
            raise ValueError(
                f"No selection metrics were available for {statistical_type} / {portfolio_type}"
            )
        return (results, metrics_used, selection_objective)

    add_model_selection_score = callback("add_model_selection_score", add_model_selection_score)

    def prune(
        models_to_do,
        validation_results,
        fold_results,
        fold,
        target,
        type,
        portfolio_type,
        multiplier,
        maximum_left,
    ):
        original_models_to_do = {(model["name"], str(model["params"])) for model in models_to_do}
        current_results = pd.DataFrame(validation_results)
        current_results["Parameters"] = current_results["Parameters"].apply(lambda x: str(x))
        metric_columns = [
            column
            for column in current_results.select_dtypes(include="number").columns
            if column not in ["Fold"]
        ]
        current_results = (
            current_results.groupby(["Model", "Parameters"])
            .agg({**{column: "mean" for column in metric_columns}, "Fold": "nunique"})
            .rename(columns={**{column: f"{column} Mean" for column in metric_columns}})
            .reset_index()
        )
        if fold_results.empty:
            combined_results = current_results.copy()
        else:
            fold_results = fold_results[fold_results["Fold"] <= fold].copy()
            fold_results["Parameters"] = fold_results["Parameters"].apply(lambda x: str(x))
            fold_results = (
                fold_results.groupby(["Model", "Parameters"])
                .agg({**{column: "mean" for column in metric_columns}, "Fold": "nunique"})
                .rename(columns={**{column: f"{column} Mean" for column in metric_columns}})
                .reset_index()
            )
            common_columns = fold_results.columns.intersection(current_results.columns)
            combined_results = pd.concat(
                [fold_results[common_columns], current_results[common_columns]], ignore_index=True
            )
        combined_results, metrics_used, selection_objective = add_model_selection_score(
            combined_results, type, portfolio_type, target
        )
        combined_results = combined_results.sort_values(
            "Model Selection Score", ascending=False, na_position="last"
        )
        logger.info(
            "%s | Fold %d | Portfolio-aware pruning | Portfolio type: %s | Role: %s | Objective: %s | Metrics: %s",
            target,
            fold,
            portfolio_type,
            portfolio_selection_role(portfolio_type),
            selection_objective,
            metrics_used,
        )
        keep_count = int(np.ceil(len(combined_results) * multiplier))
        keep_count = min(keep_count, maximum_left)
        better_results = combined_results.head(keep_count)
        better_models = set(zip(better_results["Model"], better_results["Parameters"]))
        new_models_to_do = [
            model
            for model in models_to_do
            if (model["name"], str(model["params"])) in better_models
        ]
        logger.info(
            "%s | Fold %d pruning | %d -> %d active new configurations",
            target,
            fold,
            len(models_to_do),
            len(new_models_to_do),
        )
        better_results = better_results[
            ~better_results.apply(
                lambda row: (row["Model"], row["Parameters"]) in original_models_to_do, axis=1
            )
        ].copy()
        return new_models_to_do

    prune = callback("prune", prune)

    def rank_validation_results(results, target, target_type, portfolio_type):
        results = results.copy()
        results, metrics_used, selection_objective = add_model_selection_score(
            results, target_type, portfolio_type, target
        )
        sort_columns = ["Fold", "Model Selection Score"]
        ascending = [False, False]
        results = results.sort_values(sort_columns, ascending=ascending, na_position="last")
        results = results.reset_index(drop=True)
        if "Rank" in results.columns:
            results = results.drop(columns=["Rank"])
        results.insert(0, "Rank", range(1, len(results) + 1))
        logger.info(
            "%s | Final model ranking complete | Statistical type: %s | Portfolio type: %s | Role: %s | Objective: %s | Primary metrics: %s",
            target,
            target_type,
            portfolio_type,
            portfolio_selection_role(portfolio_type),
            selection_objective,
            metrics_used,
        )
        return results

    rank_validation_results = callback("rank_validation_results", rank_validation_results)

    def add_testing_eligibility(results, target_type, min_folds):
        results = results.copy()
        full_validation = results["Fold"] >= min_folds
        eligible = pd.Series(False, index=results.index)
        reasons = pd.Series("", index=results.index, dtype=object)
        if target_type == "continuous":
            rank_ic = (
                results["Rank IC Mean"].abs()
                if "Rank IC Mean" in results.columns
                else pd.Series(np.nan, index=results.index)
            )
            r2 = (
                results["R2 Mean"]
                if "R2 Mean" in results.columns
                else pd.Series(np.nan, index=results.index)
            )
            rank_ic_std = (
                results["Rank IC Std"]
                if "Rank IC Std" in results.columns
                else pd.Series(np.nan, index=results.index)
            )
            predictive_gate = (r2 >= 0.05) & (rank_ic >= 0.1) | (rank_ic >= 0.2)
            stability_gate = rank_ic_std <= MAX_CONTINUOUS_RANK_IC_STD
            eligible = full_validation & predictive_gate & stability_gate
            reasons = np.where(
                ~full_validation,
                "Not all folds completed",
                np.where(
                    ~predictive_gate,
                    "Below continuous predictability gate",
                    np.where(~stability_gate, "Rank IC too unstable across folds", "Eligible"),
                ),
            )
        elif target_type == "binary":
            roc_auc = (
                results["ROC AUC Mean"]
                if "ROC AUC Mean" in results.columns
                else pd.Series(np.nan, index=results.index)
            )
            pr_auc = (
                results["PR AUC Mean"]
                if "PR AUC Mean" in results.columns
                else pd.Series(np.nan, index=results.index)
            )
            roc_auc_std = (
                results["ROC AUC Std"]
                if "ROC AUC Std" in results.columns
                else pd.Series(np.nan, index=results.index)
            )
            balanced_accuracy = (
                results["Balanced Accuracy Mean"]
                if "Balanced Accuracy Mean" in results.columns
                else pd.Series(1.0, index=results.index)
            )
            predictive_gate = (roc_auc >= 0.6) & (pr_auc >= 0.2) & (balanced_accuracy > 0.5)
            stability_gate = roc_auc_std <= MAX_BINARY_ROC_AUC_STD
            eligible = full_validation & predictive_gate & stability_gate
            reasons = np.where(
                ~full_validation,
                "Not all folds completed",
                np.where(
                    ~predictive_gate,
                    "Below binary predictability gate",
                    np.where(~stability_gate, "ROC AUC too unstable across folds", "Eligible"),
                ),
            )
        elif target_type == "multiclass":
            macro_f1 = (
                results["Macro F1 Mean"]
                if "Macro F1 Mean" in results.columns
                else pd.Series(np.nan, index=results.index)
            )
            macro_f1_std = (
                results["Macro F1 Std"]
                if "Macro F1 Std" in results.columns
                else pd.Series(np.nan, index=results.index)
            )
            predictive_gate = macro_f1 >= 0.45
            stability_gate = macro_f1_std <= MAX_MULTICLASS_MACRO_F1_STD
            eligible = full_validation & predictive_gate & stability_gate
            reasons = np.where(
                ~full_validation,
                "Not all folds completed",
                np.where(
                    ~predictive_gate,
                    "Below multiclass Macro F1 gate",
                    np.where(~stability_gate, "Macro F1 too unstable across folds", "Eligible"),
                ),
            )
        else:
            raise ValueError(f"Unknown target type: {target_type}")
        if "Testing Eligible" in results.columns:
            results = results.drop(columns=["Testing Eligible"])
        if "Testing Eligibility Reason" in results.columns:
            results = results.drop(columns=["Testing Eligibility Reason"])
        results.insert(3, "Testing Eligible", eligible.astype(bool))
        results.insert(4, "Testing Eligibility Reason", reasons)
        return results

    add_testing_eligibility = callback("add_testing_eligibility", add_testing_eligibility)

    if "continuous" not in types:
        logger.info(
            "No completely new continuous targets found; checking previously completed continuous configurations"
        )
        USED_CONTINUOUS_INTERSECTION = intersection_of_used_models(USED_CONTINUOUS_MODELS)
        COMBINED_USED_CONTINUOUS = combos_to_models(USED_CONTINUOUS_INTERSECTION)
        continuous_before = len(CONTINUOUS_MODELS)
        CONTINUOUS_MODELS = [
            model
            for model in CONTINUOUS_MODELS
            if not model_is_fully_used(model, COMBINED_USED_CONTINUOUS)
        ]
        logger.info(
            "Continuous model families remaining: %d/%d", len(CONTINUOUS_MODELS), continuous_before
        )
    if "binary" not in types:
        logger.info(
            "No completely new binary targets found; checking previously completed binary configurations"
        )
        USED_BINARY_INTERSECTION = intersection_of_used_models(USED_BINARY_MODELS)
        COMBINED_USED_BINARY = combos_to_models(USED_BINARY_INTERSECTION)
        binary_before = len(BINARY_MODELS)
        BINARY_MODELS = [
            model for model in BINARY_MODELS if not model_is_fully_used(model, COMBINED_USED_BINARY)
        ]
        logger.info("Binary model families remaining: %d/%d", len(BINARY_MODELS), binary_before)
    if "multiclass" not in types:
        logger.info(
            "No completely new multiclass targets found; checking previously completed multiclass configurations"
        )
        USED_MULTICLASS_INTERSECTION = intersection_of_used_models(USED_MULTICLASS_MODELS)
        COMBINED_USED_MULTICLASS = combos_to_models(USED_MULTICLASS_INTERSECTION)
        multiclass_before = len(MULTICLASS_MODELS)
        MULTICLASS_MODELS = [
            model
            for model in MULTICLASS_MODELS
            if not model_is_fully_used(model, COMBINED_USED_MULTICLASS)
        ]
        logger.info(
            "Multiclass model families remaining: %d/%d", len(MULTICLASS_MODELS), multiclass_before
        )
    logger.info("Generating exact parameter combinations")
    ALL_BINARY_MODELS = list_of_all_combos(BINARY_MODELS)
    ALL_CONTINUOUS_MODELS = list_of_all_combos(CONTINUOUS_MODELS)
    ALL_MULTICLASS_MODELS = list_of_all_combos(MULTICLASS_MODELS)
    ALL_CONTINUOUS_MODELS = choose_catalogue(
        "ALL_CONTINUOUS_MODELS",
        ALL_CONTINUOUS_MODELS,
        [
            {"name": "Mean Baseline", "params": {}},
            {"name": "OLS", "params": {}},
            {"name": "Ridge", "params": {"alpha": 1e-05}},
            {"name": "Ridge", "params": {"alpha": 10000}},
            {"name": "Lasso", "params": {"alpha": 10}},
            {"name": "Elastic Net", "params": {"alpha": 10, "l1_ratio": 0.01}},
            {"name": "Elastic Net", "params": {"alpha": 10, "l1_ratio": 0.99}},
            {"name": "Huber", "params": {"epsilon": 1.05, "alpha": 0}},
            {"name": "Huber", "params": {"epsilon": 1.05, "alpha": 1}},
            {"name": "Huber", "params": {"epsilon": 2.5, "alpha": 0}},
            {"name": "Huber", "params": {"epsilon": 2.5, "alpha": 1}},
            {
                "name": "Hist Gradient Boosting",
                "params": {
                    "learning_rate": 0.005,
                    "max_iter": 100,
                    "max_leaf_nodes": 7,
                    "max_depth": None,
                    "min_samples_leaf": 5,
                    "l2_regularization": 0,
                },
            },
            {
                "name": "Hist Gradient Boosting",
                "params": {
                    "learning_rate": 0.005,
                    "max_iter": 100,
                    "max_leaf_nodes": 7,
                    "max_depth": 3,
                    "min_samples_leaf": 200,
                    "l2_regularization": 100,
                },
            },
            {
                "name": "Gradient Boosting",
                "params": {
                    "learning_rate": 0.005,
                    "n_estimators": 100,
                    "max_depth": 2,
                    "min_samples_leaf": 100,
                    "subsample": 1.0,
                    "max_features": "sqrt",
                },
            },
            {
                "name": "Random Forest",
                "params": {
                    "n_estimators": 200,
                    "max_depth": None,
                    "min_samples_leaf": 1,
                    "min_samples_split": 2,
                    "max_features": "sqrt",
                    "bootstrap": True,
                },
            },
            {
                "name": "XGBoost",
                "params": {
                    "n_estimators": 200,
                    "learning_rate": 0.005,
                    "max_depth": 2,
                    "min_child_weight": 1,
                    "subsample": 0.5,
                    "colsample_bytree": 0.4,
                    "gamma": 0,
                    "reg_alpha": 0,
                    "reg_lambda": 0,
                },
            },
            {
                "name": "XGBoost",
                "params": {
                    "n_estimators": 200,
                    "learning_rate": 0.005,
                    "max_depth": 2,
                    "min_child_weight": 1,
                    "subsample": 0.5,
                    "colsample_bytree": 1.0,
                    "gamma": 5,
                    "reg_alpha": 10,
                    "reg_lambda": 100,
                },
            },
            {
                "name": "XGBoost",
                "params": {
                    "n_estimators": 200,
                    "learning_rate": 0.005,
                    "max_depth": 2,
                    "min_child_weight": 50,
                    "subsample": 1.0,
                    "colsample_bytree": 0.4,
                    "gamma": 0,
                    "reg_alpha": 10,
                    "reg_lambda": 100,
                },
            },
            {
                "name": "XGBoost",
                "params": {
                    "n_estimators": 200,
                    "learning_rate": 0.005,
                    "max_depth": 2,
                    "min_child_weight": 50,
                    "subsample": 1.0,
                    "colsample_bytree": 1.0,
                    "gamma": 5,
                    "reg_alpha": 0,
                    "reg_lambda": 0,
                },
            },
            {
                "name": "XGBoost",
                "params": {
                    "n_estimators": 200,
                    "learning_rate": 0.005,
                    "max_depth": 10,
                    "min_child_weight": 1,
                    "subsample": 1.0,
                    "colsample_bytree": 0.4,
                    "gamma": 5,
                    "reg_alpha": 0,
                    "reg_lambda": 100,
                },
            },
            {
                "name": "XGBoost",
                "params": {
                    "n_estimators": 200,
                    "learning_rate": 0.005,
                    "max_depth": 10,
                    "min_child_weight": 1,
                    "subsample": 1.0,
                    "colsample_bytree": 1.0,
                    "gamma": 0,
                    "reg_alpha": 10,
                    "reg_lambda": 0,
                },
            },
            {
                "name": "XGBoost",
                "params": {
                    "n_estimators": 200,
                    "learning_rate": 0.005,
                    "max_depth": 10,
                    "min_child_weight": 50,
                    "subsample": 0.5,
                    "colsample_bytree": 0.4,
                    "gamma": 5,
                    "reg_alpha": 10,
                    "reg_lambda": 0,
                },
            },
            {
                "name": "XGBoost",
                "params": {
                    "n_estimators": 200,
                    "learning_rate": 0.005,
                    "max_depth": 10,
                    "min_child_weight": 50,
                    "subsample": 0.5,
                    "colsample_bytree": 1.0,
                    "gamma": 0,
                    "reg_alpha": 0,
                    "reg_lambda": 100,
                },
            },
            {
                "name": "XGBoost",
                "params": {
                    "n_estimators": 200,
                    "learning_rate": 0.005,
                    "max_depth": 2,
                    "min_child_weight": 1,
                    "subsample": 0.5,
                    "colsample_bytree": 0.4,
                    "gamma": 0,
                    "reg_alpha": 10,
                    "reg_lambda": 100,
                },
            },
            {
                "name": "LightGBM",
                "params": {
                    "n_estimators": 200,
                    "learning_rate": 0.005,
                    "num_leaves": 7,
                    "max_depth": -1,
                    "min_child_samples": 5,
                    "subsample": 0.5,
                    "colsample_bytree": 0.4,
                    "reg_alpha": 0,
                    "reg_lambda": 0,
                },
            },
            {"name": "SVR Linear", "params": {"kernel": "linear", "C": 1e-05, "epsilon": 0.5}},
            {"name": "SVR RBF", "params": {"kernel": "rbf", "C": 1000, "epsilon": 0.5, "gamma": 1}},
            {
                "name": "SVR RBF",
                "params": {"kernel": "rbf", "C": 1e-05, "epsilon": 0.5, "gamma": "auto"},
            },
            {
                "name": "SVR RBF",
                "params": {"kernel": "rbf", "C": 1000, "epsilon": 0.5, "gamma": "scale"},
            },
            {
                "name": "SVR RBF",
                "params": {"kernel": "rbf", "C": 1e-05, "epsilon": 0.5, "gamma": 1},
            },
            {"name": "kNN", "params": {"n_neighbors": 3, "weights": "uniform", "p": 1}},
            {"name": "kNN", "params": {"n_neighbors": 3, "weights": "uniform", "p": 2}},
            {"name": "kNN", "params": {"n_neighbors": 3, "weights": "distance", "p": 1}},
            {"name": "kNN", "params": {"n_neighbors": 3, "weights": "distance", "p": 2}},
            {"name": "kNN", "params": {"n_neighbors": 250, "weights": "uniform", "p": 1}},
            {"name": "kNN", "params": {"n_neighbors": 250, "weights": "uniform", "p": 2}},
            {"name": "kNN", "params": {"n_neighbors": 250, "weights": "distance", "p": 1}},
            {"name": "kNN", "params": {"n_neighbors": 250, "weights": "distance", "p": 2}},
            {
                "name": "MLP",
                "params": {
                    "hidden_layer_sizes": (64,),
                    "activation": "tanh",
                    "alpha": 0.1,
                    "learning_rate_init": 0.01,
                    "batch_size": "auto",
                },
            },
            {
                "name": "MLP",
                "params": {
                    "hidden_layer_sizes": (64,),
                    "activation": "relu",
                    "alpha": 1e-07,
                    "learning_rate_init": 0.01,
                    "batch_size": 512,
                },
            },
            {
                "name": "MLP",
                "params": {
                    "hidden_layer_sizes": (64,),
                    "activation": "relu",
                    "alpha": 0.1,
                    "learning_rate_init": 1e-05,
                    "batch_size": 512,
                },
            },
            {
                "name": "MLP",
                "params": {
                    "hidden_layer_sizes": (64,),
                    "activation": "tanh",
                    "alpha": 1e-07,
                    "learning_rate_init": 1e-05,
                    "batch_size": "auto",
                },
            },
            {
                "name": "MLP",
                "params": {
                    "hidden_layer_sizes": (64,),
                    "activation": "relu",
                    "alpha": 0.1,
                    "learning_rate_init": 0.01,
                    "batch_size": 32,
                },
            },
            {
                "name": "MLP",
                "params": {
                    "hidden_layer_sizes": (64,),
                    "activation": "tanh",
                    "alpha": 1e-07,
                    "learning_rate_init": 0.01,
                    "batch_size": 32,
                },
            },
            {
                "name": "MLP",
                "params": {
                    "hidden_layer_sizes": (64,),
                    "activation": "relu",
                    "alpha": 0.1,
                    "learning_rate_init": 0.01,
                    "batch_size": "auto",
                },
            },
            {
                "name": "MLP",
                "params": {
                    "hidden_layer_sizes": (64,),
                    "activation": "tanh",
                    "alpha": 1e-07,
                    "learning_rate_init": 1e-05,
                    "batch_size": 512,
                },
            },
            {
                "name": "MLP",
                "params": {
                    "hidden_layer_sizes": (64,),
                    "activation": "tanh",
                    "alpha": 1e-07,
                    "learning_rate_init": 0.01,
                    "batch_size": "auto",
                },
            },
        ],
    )
    ALL_BINARY_MODELS = choose_catalogue(
        "ALL_BINARY_MODELS",
        ALL_BINARY_MODELS,
        [
            {"name": "Binary Baseline", "params": {}},
            {"name": "Logistic Regression", "params": {"class_weight": None}},
            {"name": "Logistic Regression", "params": {"class_weight": "balanced"}},
            {"name": "L2 Logistic Regression", "params": {"C": 1e-05, "class_weight": None}},
            {"name": "L2 Logistic Regression", "params": {"C": 1e-05, "class_weight": "balanced"}},
            {"name": "L2 Logistic Regression", "params": {"C": 1000, "class_weight": None}},
            {"name": "L2 Logistic Regression", "params": {"C": 1000, "class_weight": "balanced"}},
            {"name": "L1 Logistic Regression", "params": {"C": 1e-05, "class_weight": None}},
            {"name": "L1 Logistic Regression", "params": {"C": 1e-05, "class_weight": "balanced"}},
            {"name": "L1 Logistic Regression", "params": {"C": 1000, "class_weight": None}},
            {"name": "L1 Logistic Regression", "params": {"C": 1000, "class_weight": "balanced"}},
            {
                "name": "Elastic Net Logistic Regression",
                "params": {"C": 1e-05, "l1_ratio": 0.01, "class_weight": None},
            },
            {
                "name": "Elastic Net Logistic Regression",
                "params": {"C": 1e-05, "l1_ratio": 0.01, "class_weight": "balanced"},
            },
            {
                "name": "Elastic Net Logistic Regression",
                "params": {"C": 1e-05, "l1_ratio": 0.99, "class_weight": None},
            },
            {
                "name": "Elastic Net Logistic Regression",
                "params": {"C": 1e-05, "l1_ratio": 0.99, "class_weight": "balanced"},
            },
            {
                "name": "Elastic Net Logistic Regression",
                "params": {"C": 1000, "l1_ratio": 0.01, "class_weight": None},
            },
            {
                "name": "Elastic Net Logistic Regression",
                "params": {"C": 1000, "l1_ratio": 0.01, "class_weight": "balanced"},
            },
            {
                "name": "Elastic Net Logistic Regression",
                "params": {"C": 1000, "l1_ratio": 0.99, "class_weight": None},
            },
            {
                "name": "Elastic Net Logistic Regression",
                "params": {"C": 1000, "l1_ratio": 0.99, "class_weight": "balanced"},
            },
            {
                "name": "Hist Gradient Boosting",
                "params": {
                    "learning_rate": 0.005,
                    "max_iter": 100,
                    "max_leaf_nodes": 7,
                    "max_depth": None,
                    "min_samples_leaf": 5,
                    "l2_regularization": 0,
                    "class_weight": None,
                },
            },
            {
                "name": "Hist Gradient Boosting",
                "params": {
                    "learning_rate": 0.2,
                    "max_iter": 1500,
                    "max_leaf_nodes": 127,
                    "max_depth": 10,
                    "min_samples_leaf": 200,
                    "l2_regularization": 100,
                    "class_weight": "balanced",
                },
            },
            {
                "name": "Hist Gradient Boosting",
                "params": {
                    "learning_rate": 0.005,
                    "max_iter": 100,
                    "max_leaf_nodes": 7,
                    "max_depth": 3,
                    "min_samples_leaf": 200,
                    "l2_regularization": 100,
                    "class_weight": "balanced",
                },
            },
            {
                "name": "Hist Gradient Boosting",
                "params": {
                    "learning_rate": 0.2,
                    "max_iter": 1500,
                    "max_leaf_nodes": 127,
                    "max_depth": 3,
                    "min_samples_leaf": 5,
                    "l2_regularization": 0,
                    "class_weight": None,
                },
            },
            {
                "name": "Hist Gradient Boosting",
                "params": {
                    "learning_rate": 0.005,
                    "max_iter": 100,
                    "max_leaf_nodes": 127,
                    "max_depth": None,
                    "min_samples_leaf": 5,
                    "l2_regularization": 100,
                    "class_weight": "balanced",
                },
            },
            {
                "name": "Hist Gradient Boosting",
                "params": {
                    "learning_rate": 0.2,
                    "max_iter": 1500,
                    "max_leaf_nodes": 7,
                    "max_depth": 10,
                    "min_samples_leaf": 200,
                    "l2_regularization": 0,
                    "class_weight": None,
                },
            },
            {
                "name": "Hist Gradient Boosting",
                "params": {
                    "learning_rate": 0.005,
                    "max_iter": 100,
                    "max_leaf_nodes": 127,
                    "max_depth": 3,
                    "min_samples_leaf": 200,
                    "l2_regularization": 0,
                    "class_weight": None,
                },
            },
            {
                "name": "Hist Gradient Boosting",
                "params": {
                    "learning_rate": 0.2,
                    "max_iter": 1500,
                    "max_leaf_nodes": 7,
                    "max_depth": None,
                    "min_samples_leaf": 5,
                    "l2_regularization": 100,
                    "class_weight": "balanced",
                },
            },
            {
                "name": "Hist Gradient Boosting",
                "params": {
                    "learning_rate": 0.005,
                    "max_iter": 1500,
                    "max_leaf_nodes": 7,
                    "max_depth": 10,
                    "min_samples_leaf": 5,
                    "l2_regularization": 0,
                    "class_weight": "balanced",
                },
            },
            {
                "name": "Hist Gradient Boosting",
                "params": {
                    "learning_rate": 0.2,
                    "max_iter": 100,
                    "max_leaf_nodes": 127,
                    "max_depth": None,
                    "min_samples_leaf": 200,
                    "l2_regularization": 100,
                    "class_weight": None,
                },
            },
            {
                "name": "Hist Gradient Boosting",
                "params": {
                    "learning_rate": 0.005,
                    "max_iter": 1500,
                    "max_leaf_nodes": 7,
                    "max_depth": 3,
                    "min_samples_leaf": 5,
                    "l2_regularization": 100,
                    "class_weight": None,
                },
            },
            {
                "name": "Hist Gradient Boosting",
                "params": {
                    "learning_rate": 0.2,
                    "max_iter": 100,
                    "max_leaf_nodes": 127,
                    "max_depth": 10,
                    "min_samples_leaf": 5,
                    "l2_regularization": 0,
                    "class_weight": "balanced",
                },
            },
            {
                "name": "Hist Gradient Boosting",
                "params": {
                    "learning_rate": 0.005,
                    "max_iter": 1500,
                    "max_leaf_nodes": 127,
                    "max_depth": None,
                    "min_samples_leaf": 200,
                    "l2_regularization": 0,
                    "class_weight": "balanced",
                },
            },
            {
                "name": "Hist Gradient Boosting",
                "params": {
                    "learning_rate": 0.2,
                    "max_iter": 100,
                    "max_leaf_nodes": 7,
                    "max_depth": 10,
                    "min_samples_leaf": 5,
                    "l2_regularization": 100,
                    "class_weight": None,
                },
            },
            {
                "name": "Gradient Boosting",
                "params": {
                    "learning_rate": 0.005,
                    "n_estimators": 100,
                    "max_depth": 2,
                    "min_samples_leaf": 5,
                    "subsample": 0.5,
                    "max_features": None,
                },
            },
            {
                "name": "Gradient Boosting",
                "params": {
                    "learning_rate": 0.2,
                    "n_estimators": 1000,
                    "max_depth": 8,
                    "min_samples_leaf": 100,
                    "subsample": 1.0,
                    "max_features": 0.8,
                },
            },
            {
                "name": "Gradient Boosting",
                "params": {
                    "learning_rate": 0.005,
                    "n_estimators": 100,
                    "max_depth": 2,
                    "min_samples_leaf": 100,
                    "subsample": 1.0,
                    "max_features": "sqrt",
                },
            },
            {
                "name": "Gradient Boosting",
                "params": {
                    "learning_rate": 0.2,
                    "n_estimators": 1000,
                    "max_depth": 8,
                    "min_samples_leaf": 5,
                    "subsample": 0.5,
                    "max_features": 0.3,
                },
            },
            {
                "name": "Gradient Boosting",
                "params": {
                    "learning_rate": 0.005,
                    "n_estimators": 100,
                    "max_depth": 8,
                    "min_samples_leaf": 5,
                    "subsample": 1.0,
                    "max_features": 0.3,
                },
            },
            {
                "name": "Gradient Boosting",
                "params": {
                    "learning_rate": 0.2,
                    "n_estimators": 1000,
                    "max_depth": 2,
                    "min_samples_leaf": 100,
                    "subsample": 0.5,
                    "max_features": None,
                },
            },
            {
                "name": "Gradient Boosting",
                "params": {
                    "learning_rate": 0.005,
                    "n_estimators": 100,
                    "max_depth": 8,
                    "min_samples_leaf": 100,
                    "subsample": 0.5,
                    "max_features": 0.8,
                },
            },
            {
                "name": "Gradient Boosting",
                "params": {
                    "learning_rate": 0.2,
                    "n_estimators": 1000,
                    "max_depth": 2,
                    "min_samples_leaf": 5,
                    "subsample": 1.0,
                    "max_features": "sqrt",
                },
            },
            {
                "name": "Gradient Boosting",
                "params": {
                    "learning_rate": 0.005,
                    "n_estimators": 100,
                    "max_depth": 2,
                    "min_samples_leaf": 5,
                    "subsample": 1.0,
                    "max_features": 0.8,
                },
            },
            {
                "name": "Gradient Boosting",
                "params": {
                    "learning_rate": 0.2,
                    "n_estimators": 1000,
                    "max_depth": 8,
                    "min_samples_leaf": 100,
                    "subsample": 0.5,
                    "max_features": "sqrt",
                },
            },
            {
                "name": "Gradient Boosting",
                "params": {
                    "learning_rate": 0.005,
                    "n_estimators": 100,
                    "max_depth": 2,
                    "min_samples_leaf": 100,
                    "subsample": 0.5,
                    "max_features": 0.3,
                },
            },
            {
                "name": "Gradient Boosting",
                "params": {
                    "learning_rate": 0.2,
                    "n_estimators": 1000,
                    "max_depth": 8,
                    "min_samples_leaf": 5,
                    "subsample": 1.0,
                    "max_features": None,
                },
            },
            {
                "name": "Random Forest",
                "params": {
                    "n_estimators": 200,
                    "max_depth": None,
                    "min_samples_leaf": 1,
                    "min_samples_split": 2,
                    "max_features": "sqrt",
                    "bootstrap": True,
                    "class_weight": None,
                },
            },
            {
                "name": "Random Forest",
                "params": {
                    "n_estimators": 1500,
                    "max_depth": 30,
                    "min_samples_leaf": 100,
                    "min_samples_split": 50,
                    "max_features": 1.0,
                    "bootstrap": False,
                    "class_weight": "balanced",
                },
            },
            {
                "name": "Random Forest",
                "params": {
                    "n_estimators": 200,
                    "max_depth": 5,
                    "min_samples_leaf": 1,
                    "min_samples_split": 2,
                    "max_features": 0.2,
                    "bootstrap": False,
                    "class_weight": "balanced",
                },
            },
            {
                "name": "Random Forest",
                "params": {
                    "n_estimators": 1500,
                    "max_depth": None,
                    "min_samples_leaf": 100,
                    "min_samples_split": 50,
                    "max_features": 0.2,
                    "bootstrap": True,
                    "class_weight": None,
                },
            },
            {
                "name": "Random Forest",
                "params": {
                    "n_estimators": 200,
                    "max_depth": 5,
                    "min_samples_leaf": 100,
                    "min_samples_split": 50,
                    "max_features": "sqrt",
                    "bootstrap": True,
                    "class_weight": "balanced",
                },
            },
            {
                "name": "Random Forest",
                "params": {
                    "n_estimators": 1500,
                    "max_depth": 5,
                    "min_samples_leaf": 1,
                    "min_samples_split": 50,
                    "max_features": "sqrt",
                    "bootstrap": False,
                    "class_weight": None,
                },
            },
            {
                "name": "Random Forest",
                "params": {
                    "n_estimators": 200,
                    "max_depth": 30,
                    "min_samples_leaf": 1,
                    "min_samples_split": 2,
                    "max_features": 1.0,
                    "bootstrap": True,
                    "class_weight": "balanced",
                },
            },
            {
                "name": "Random Forest",
                "params": {
                    "n_estimators": 1500,
                    "max_depth": None,
                    "min_samples_leaf": 100,
                    "min_samples_split": 2,
                    "max_features": 1.0,
                    "bootstrap": False,
                    "class_weight": None,
                },
            },
            {
                "name": "Random Forest",
                "params": {
                    "n_estimators": 200,
                    "max_depth": 30,
                    "min_samples_leaf": 1,
                    "min_samples_split": 50,
                    "max_features": 0.2,
                    "bootstrap": True,
                    "class_weight": None,
                },
            },
            {
                "name": "Random Forest",
                "params": {
                    "n_estimators": 1500,
                    "max_depth": 5,
                    "min_samples_leaf": 100,
                    "min_samples_split": 2,
                    "max_features": "sqrt",
                    "bootstrap": False,
                    "class_weight": "balanced",
                },
            },
            {
                "name": "Random Forest",
                "params": {
                    "n_estimators": 200,
                    "max_depth": None,
                    "min_samples_leaf": 1,
                    "min_samples_split": 50,
                    "max_features": 1.0,
                    "bootstrap": False,
                    "class_weight": None,
                },
            },
            {
                "name": "Random Forest",
                "params": {
                    "n_estimators": 1500,
                    "max_depth": 30,
                    "min_samples_leaf": 100,
                    "min_samples_split": 2,
                    "max_features": 0.2,
                    "bootstrap": True,
                    "class_weight": "balanced",
                },
            },
            {
                "name": "Random Forest",
                "params": {
                    "n_estimators": 200,
                    "max_depth": None,
                    "min_samples_leaf": 100,
                    "min_samples_split": 50,
                    "max_features": 0.2,
                    "bootstrap": False,
                    "class_weight": "balanced",
                },
            },
            {
                "name": "Random Forest",
                "params": {
                    "n_estimators": 1500,
                    "max_depth": 5,
                    "min_samples_leaf": 1,
                    "min_samples_split": 2,
                    "max_features": 1.0,
                    "bootstrap": True,
                    "class_weight": None,
                },
            },
            {
                "name": "XGBoost",
                "params": {
                    "n_estimators": 200,
                    "learning_rate": 0.005,
                    "max_depth": 2,
                    "min_child_weight": 1,
                    "subsample": 0.5,
                    "colsample_bytree": 0.4,
                    "gamma": 0,
                    "reg_alpha": 0,
                    "reg_lambda": 0,
                    "class_weight": None,
                },
            },
            {
                "name": "XGBoost",
                "params": {
                    "n_estimators": 1500,
                    "learning_rate": 0.2,
                    "max_depth": 10,
                    "min_child_weight": 50,
                    "subsample": 1.0,
                    "colsample_bytree": 1.0,
                    "gamma": 5,
                    "reg_alpha": 10,
                    "reg_lambda": 100,
                    "class_weight": "balanced",
                },
            },
            {
                "name": "XGBoost",
                "params": {
                    "n_estimators": 200,
                    "learning_rate": 0.005,
                    "max_depth": 2,
                    "min_child_weight": 1,
                    "subsample": 0.5,
                    "colsample_bytree": 1.0,
                    "gamma": 5,
                    "reg_alpha": 10,
                    "reg_lambda": 100,
                    "class_weight": "balanced",
                },
            },
            {
                "name": "XGBoost",
                "params": {
                    "n_estimators": 1500,
                    "learning_rate": 0.2,
                    "max_depth": 10,
                    "min_child_weight": 50,
                    "subsample": 1.0,
                    "colsample_bytree": 0.4,
                    "gamma": 0,
                    "reg_alpha": 0,
                    "reg_lambda": 0,
                    "class_weight": None,
                },
            },
            {
                "name": "XGBoost",
                "params": {
                    "n_estimators": 200,
                    "learning_rate": 0.005,
                    "max_depth": 2,
                    "min_child_weight": 50,
                    "subsample": 1.0,
                    "colsample_bytree": 0.4,
                    "gamma": 0,
                    "reg_alpha": 0,
                    "reg_lambda": 100,
                    "class_weight": "balanced",
                },
            },
            {
                "name": "XGBoost",
                "params": {
                    "n_estimators": 1500,
                    "learning_rate": 0.2,
                    "max_depth": 10,
                    "min_child_weight": 1,
                    "subsample": 0.5,
                    "colsample_bytree": 1.0,
                    "gamma": 5,
                    "reg_alpha": 10,
                    "reg_lambda": 0,
                    "class_weight": None,
                },
            },
            {
                "name": "XGBoost",
                "params": {
                    "n_estimators": 200,
                    "learning_rate": 0.005,
                    "max_depth": 2,
                    "min_child_weight": 50,
                    "subsample": 1.0,
                    "colsample_bytree": 0.4,
                    "gamma": 5,
                    "reg_alpha": 10,
                    "reg_lambda": 0,
                    "class_weight": None,
                },
            },
            {
                "name": "XGBoost",
                "params": {
                    "n_estimators": 1500,
                    "learning_rate": 0.2,
                    "max_depth": 10,
                    "min_child_weight": 1,
                    "subsample": 0.5,
                    "colsample_bytree": 1.0,
                    "gamma": 0,
                    "reg_alpha": 0,
                    "reg_lambda": 100,
                    "class_weight": "balanced",
                },
            },
            {
                "name": "XGBoost",
                "params": {
                    "n_estimators": 200,
                    "learning_rate": 0.005,
                    "max_depth": 10,
                    "min_child_weight": 1,
                    "subsample": 1.0,
                    "colsample_bytree": 0.4,
                    "gamma": 0,
                    "reg_alpha": 10,
                    "reg_lambda": 0,
                    "class_weight": "balanced",
                },
            },
            {
                "name": "XGBoost",
                "params": {
                    "n_estimators": 1500,
                    "learning_rate": 0.2,
                    "max_depth": 2,
                    "min_child_weight": 50,
                    "subsample": 0.5,
                    "colsample_bytree": 1.0,
                    "gamma": 5,
                    "reg_alpha": 0,
                    "reg_lambda": 100,
                    "class_weight": None,
                },
            },
            {
                "name": "XGBoost",
                "params": {
                    "n_estimators": 200,
                    "learning_rate": 0.005,
                    "max_depth": 10,
                    "min_child_weight": 1,
                    "subsample": 1.0,
                    "colsample_bytree": 0.4,
                    "gamma": 5,
                    "reg_alpha": 0,
                    "reg_lambda": 100,
                    "class_weight": None,
                },
            },
            {
                "name": "XGBoost",
                "params": {
                    "n_estimators": 1500,
                    "learning_rate": 0.2,
                    "max_depth": 2,
                    "min_child_weight": 50,
                    "subsample": 0.5,
                    "colsample_bytree": 1.0,
                    "gamma": 0,
                    "reg_alpha": 10,
                    "reg_lambda": 0,
                    "class_weight": "balanced",
                },
            },
            {
                "name": "XGBoost",
                "params": {
                    "n_estimators": 200,
                    "learning_rate": 0.005,
                    "max_depth": 10,
                    "min_child_weight": 50,
                    "subsample": 0.5,
                    "colsample_bytree": 0.4,
                    "gamma": 0,
                    "reg_alpha": 10,
                    "reg_lambda": 100,
                    "class_weight": None,
                },
            },
            {
                "name": "XGBoost",
                "params": {
                    "n_estimators": 1500,
                    "learning_rate": 0.2,
                    "max_depth": 2,
                    "min_child_weight": 1,
                    "subsample": 1.0,
                    "colsample_bytree": 1.0,
                    "gamma": 5,
                    "reg_alpha": 0,
                    "reg_lambda": 0,
                    "class_weight": "balanced",
                },
            },
            {
                "name": "XGBoost",
                "params": {
                    "n_estimators": 200,
                    "learning_rate": 0.005,
                    "max_depth": 10,
                    "min_child_weight": 50,
                    "subsample": 0.5,
                    "colsample_bytree": 0.4,
                    "gamma": 5,
                    "reg_alpha": 0,
                    "reg_lambda": 0,
                    "class_weight": "balanced",
                },
            },
            {
                "name": "XGBoost",
                "params": {
                    "n_estimators": 1500,
                    "learning_rate": 0.2,
                    "max_depth": 2,
                    "min_child_weight": 1,
                    "subsample": 1.0,
                    "colsample_bytree": 1.0,
                    "gamma": 0,
                    "reg_alpha": 10,
                    "reg_lambda": 100,
                    "class_weight": None,
                },
            },
            {
                "name": "XGBoost",
                "params": {
                    "n_estimators": 200,
                    "learning_rate": 0.005,
                    "max_depth": 2,
                    "min_child_weight": 1,
                    "subsample": 1.0,
                    "colsample_bytree": 0.4,
                    "gamma": 0,
                    "reg_alpha": 10,
                    "reg_lambda": 100,
                    "class_weight": None,
                },
            },
            {
                "name": "XGBoost",
                "params": {
                    "n_estimators": 1500,
                    "learning_rate": 0.2,
                    "max_depth": 10,
                    "min_child_weight": 50,
                    "subsample": 0.5,
                    "colsample_bytree": 1.0,
                    "gamma": 5,
                    "reg_alpha": 0,
                    "reg_lambda": 0,
                    "class_weight": "balanced",
                },
            },
            {
                "name": "XGBoost",
                "params": {
                    "n_estimators": 200,
                    "learning_rate": 0.005,
                    "max_depth": 2,
                    "min_child_weight": 1,
                    "subsample": 1.0,
                    "colsample_bytree": 0.4,
                    "gamma": 5,
                    "reg_alpha": 0,
                    "reg_lambda": 0,
                    "class_weight": "balanced",
                },
            },
            {
                "name": "XGBoost",
                "params": {
                    "n_estimators": 1500,
                    "learning_rate": 0.2,
                    "max_depth": 10,
                    "min_child_weight": 50,
                    "subsample": 0.5,
                    "colsample_bytree": 1.0,
                    "gamma": 0,
                    "reg_alpha": 10,
                    "reg_lambda": 100,
                    "class_weight": None,
                },
            },
            {
                "name": "LightGBM",
                "params": {
                    "n_estimators": 200,
                    "learning_rate": 0.005,
                    "num_leaves": 7,
                    "max_depth": -1,
                    "min_child_samples": 5,
                    "subsample": 0.5,
                    "colsample_bytree": 0.4,
                    "reg_alpha": 0,
                    "reg_lambda": 0,
                    "class_weight": None,
                },
            },
            {
                "name": "LightGBM",
                "params": {
                    "n_estimators": 1500,
                    "learning_rate": 0.2,
                    "num_leaves": 255,
                    "max_depth": 16,
                    "min_child_samples": 200,
                    "subsample": 1.0,
                    "colsample_bytree": 1.0,
                    "reg_alpha": 10,
                    "reg_lambda": 100,
                    "class_weight": "balanced",
                },
            },
            {
                "name": "LightGBM",
                "params": {
                    "n_estimators": 200,
                    "learning_rate": 0.005,
                    "num_leaves": 7,
                    "max_depth": -1,
                    "min_child_samples": 5,
                    "subsample": 1.0,
                    "colsample_bytree": 1.0,
                    "reg_alpha": 10,
                    "reg_lambda": 100,
                    "class_weight": "balanced",
                },
            },
            {
                "name": "LightGBM",
                "params": {
                    "n_estimators": 1500,
                    "learning_rate": 0.2,
                    "num_leaves": 255,
                    "max_depth": 16,
                    "min_child_samples": 200,
                    "subsample": 0.5,
                    "colsample_bytree": 0.4,
                    "reg_alpha": 0,
                    "reg_lambda": 0,
                    "class_weight": None,
                },
            },
            {
                "name": "LightGBM",
                "params": {
                    "n_estimators": 200,
                    "learning_rate": 0.005,
                    "num_leaves": 7,
                    "max_depth": 16,
                    "min_child_samples": 200,
                    "subsample": 0.5,
                    "colsample_bytree": 0.4,
                    "reg_alpha": 0,
                    "reg_lambda": 100,
                    "class_weight": "balanced",
                },
            },
            {
                "name": "LightGBM",
                "params": {
                    "n_estimators": 1500,
                    "learning_rate": 0.2,
                    "num_leaves": 255,
                    "max_depth": -1,
                    "min_child_samples": 5,
                    "subsample": 1.0,
                    "colsample_bytree": 1.0,
                    "reg_alpha": 10,
                    "reg_lambda": 0,
                    "class_weight": None,
                },
            },
            {
                "name": "LightGBM",
                "params": {
                    "n_estimators": 200,
                    "learning_rate": 0.005,
                    "num_leaves": 7,
                    "max_depth": 16,
                    "min_child_samples": 200,
                    "subsample": 0.5,
                    "colsample_bytree": 1.0,
                    "reg_alpha": 10,
                    "reg_lambda": 0,
                    "class_weight": None,
                },
            },
            {
                "name": "LightGBM",
                "params": {
                    "n_estimators": 1500,
                    "learning_rate": 0.2,
                    "num_leaves": 255,
                    "max_depth": -1,
                    "min_child_samples": 5,
                    "subsample": 1.0,
                    "colsample_bytree": 0.4,
                    "reg_alpha": 0,
                    "reg_lambda": 100,
                    "class_weight": "balanced",
                },
            },
            {
                "name": "LightGBM",
                "params": {
                    "n_estimators": 200,
                    "learning_rate": 0.005,
                    "num_leaves": 255,
                    "max_depth": -1,
                    "min_child_samples": 200,
                    "subsample": 0.5,
                    "colsample_bytree": 0.4,
                    "reg_alpha": 10,
                    "reg_lambda": 0,
                    "class_weight": "balanced",
                },
            },
            {
                "name": "LightGBM",
                "params": {
                    "n_estimators": 1500,
                    "learning_rate": 0.2,
                    "num_leaves": 7,
                    "max_depth": 16,
                    "min_child_samples": 5,
                    "subsample": 1.0,
                    "colsample_bytree": 1.0,
                    "reg_alpha": 0,
                    "reg_lambda": 100,
                    "class_weight": None,
                },
            },
            {
                "name": "LightGBM",
                "params": {
                    "n_estimators": 200,
                    "learning_rate": 0.005,
                    "num_leaves": 255,
                    "max_depth": -1,
                    "min_child_samples": 200,
                    "subsample": 0.5,
                    "colsample_bytree": 1.0,
                    "reg_alpha": 0,
                    "reg_lambda": 100,
                    "class_weight": None,
                },
            },
            {
                "name": "LightGBM",
                "params": {
                    "n_estimators": 1500,
                    "learning_rate": 0.2,
                    "num_leaves": 7,
                    "max_depth": 16,
                    "min_child_samples": 5,
                    "subsample": 1.0,
                    "colsample_bytree": 0.4,
                    "reg_alpha": 10,
                    "reg_lambda": 0,
                    "class_weight": "balanced",
                },
            },
            {
                "name": "LightGBM",
                "params": {
                    "n_estimators": 200,
                    "learning_rate": 0.005,
                    "num_leaves": 255,
                    "max_depth": 16,
                    "min_child_samples": 5,
                    "subsample": 0.5,
                    "colsample_bytree": 0.4,
                    "reg_alpha": 10,
                    "reg_lambda": 100,
                    "class_weight": None,
                },
            },
            {
                "name": "LightGBM",
                "params": {
                    "n_estimators": 1500,
                    "learning_rate": 0.2,
                    "num_leaves": 7,
                    "max_depth": -1,
                    "min_child_samples": 200,
                    "subsample": 1.0,
                    "colsample_bytree": 1.0,
                    "reg_alpha": 0,
                    "reg_lambda": 0,
                    "class_weight": "balanced",
                },
            },
            {
                "name": "LightGBM",
                "params": {
                    "n_estimators": 200,
                    "learning_rate": 0.005,
                    "num_leaves": 255,
                    "max_depth": 16,
                    "min_child_samples": 5,
                    "subsample": 0.5,
                    "colsample_bytree": 1.0,
                    "reg_alpha": 0,
                    "reg_lambda": 0,
                    "class_weight": "balanced",
                },
            },
            {
                "name": "LightGBM",
                "params": {
                    "n_estimators": 1500,
                    "learning_rate": 0.2,
                    "num_leaves": 7,
                    "max_depth": -1,
                    "min_child_samples": 200,
                    "subsample": 1.0,
                    "colsample_bytree": 0.4,
                    "reg_alpha": 10,
                    "reg_lambda": 100,
                    "class_weight": None,
                },
            },
            {
                "name": "LightGBM",
                "params": {
                    "n_estimators": 200,
                    "learning_rate": 0.005,
                    "num_leaves": 7,
                    "max_depth": -1,
                    "min_child_samples": 200,
                    "subsample": 0.5,
                    "colsample_bytree": 0.4,
                    "reg_alpha": 10,
                    "reg_lambda": 100,
                    "class_weight": None,
                },
            },
            {
                "name": "LightGBM",
                "params": {
                    "n_estimators": 1500,
                    "learning_rate": 0.2,
                    "num_leaves": 255,
                    "max_depth": 16,
                    "min_child_samples": 5,
                    "subsample": 1.0,
                    "colsample_bytree": 1.0,
                    "reg_alpha": 0,
                    "reg_lambda": 0,
                    "class_weight": "balanced",
                },
            },
            {
                "name": "LightGBM",
                "params": {
                    "n_estimators": 200,
                    "learning_rate": 0.005,
                    "num_leaves": 7,
                    "max_depth": -1,
                    "min_child_samples": 200,
                    "subsample": 0.5,
                    "colsample_bytree": 1.0,
                    "reg_alpha": 0,
                    "reg_lambda": 0,
                    "class_weight": "balanced",
                },
            },
            {
                "name": "LightGBM",
                "params": {
                    "n_estimators": 1500,
                    "learning_rate": 0.2,
                    "num_leaves": 255,
                    "max_depth": 16,
                    "min_child_samples": 5,
                    "subsample": 1.0,
                    "colsample_bytree": 0.4,
                    "reg_alpha": 10,
                    "reg_lambda": 100,
                    "class_weight": None,
                },
            },
            {
                "name": "SVM Linear",
                "params": {"kernel": "linear", "C": 1e-05, "class_weight": None},
            },
            {
                "name": "SVM Linear",
                "params": {"kernel": "linear", "C": 1e-05, "class_weight": "balanced"},
            },
            {"name": "SVM Linear", "params": {"kernel": "linear", "C": 1000, "class_weight": None}},
            {
                "name": "SVM Linear",
                "params": {"kernel": "linear", "C": 1000, "class_weight": "balanced"},
            },
            {
                "name": "SVM RBF",
                "params": {"kernel": "rbf", "C": 1e-05, "gamma": "scale", "class_weight": None},
            },
            {
                "name": "SVM RBF",
                "params": {"kernel": "rbf", "C": 1000, "gamma": 1, "class_weight": "balanced"},
            },
            {
                "name": "SVM RBF",
                "params": {
                    "kernel": "rbf",
                    "C": 1e-05,
                    "gamma": "auto",
                    "class_weight": "balanced",
                },
            },
            {
                "name": "SVM RBF",
                "params": {"kernel": "rbf", "C": 1000, "gamma": 0.0001, "class_weight": None},
            },
            {
                "name": "SVM RBF",
                "params": {
                    "kernel": "rbf",
                    "C": 1e-05,
                    "gamma": "scale",
                    "class_weight": "balanced",
                },
            },
            {
                "name": "SVM RBF",
                "params": {"kernel": "rbf", "C": 1000, "gamma": "auto", "class_weight": None},
            },
            {
                "name": "SVM RBF",
                "params": {"kernel": "rbf", "C": 1e-05, "gamma": 0.0001, "class_weight": None},
            },
            {
                "name": "SVM RBF",
                "params": {"kernel": "rbf", "C": 1e-05, "gamma": 1, "class_weight": "balanced"},
            },
            {"name": "kNN", "params": {"n_neighbors": 3, "weights": "uniform", "p": 1}},
            {"name": "kNN", "params": {"n_neighbors": 3, "weights": "uniform", "p": 2}},
            {"name": "kNN", "params": {"n_neighbors": 3, "weights": "distance", "p": 1}},
            {"name": "kNN", "params": {"n_neighbors": 3, "weights": "distance", "p": 2}},
            {"name": "kNN", "params": {"n_neighbors": 250, "weights": "uniform", "p": 1}},
            {"name": "kNN", "params": {"n_neighbors": 250, "weights": "uniform", "p": 2}},
            {"name": "kNN", "params": {"n_neighbors": 250, "weights": "distance", "p": 1}},
            {"name": "kNN", "params": {"n_neighbors": 250, "weights": "distance", "p": 2}},
            {"name": "Naive Bayes", "params": {"var_smoothing": 1e-13}},
            {"name": "Naive Bayes", "params": {"var_smoothing": 1e-05}},
            {
                "name": "MLP",
                "params": {
                    "hidden_layer_sizes": (64,),
                    "activation": "relu",
                    "alpha": 1e-07,
                    "learning_rate_init": 1e-05,
                    "batch_size": 32,
                },
            },
            {
                "name": "MLP",
                "params": {
                    "hidden_layer_sizes": (64,),
                    "activation": "tanh",
                    "alpha": 0.1,
                    "learning_rate_init": 0.01,
                    "batch_size": "auto",
                },
            },
            {
                "name": "MLP",
                "params": {
                    "hidden_layer_sizes": (64,),
                    "activation": "relu",
                    "alpha": 1e-07,
                    "learning_rate_init": 0.01,
                    "batch_size": 512,
                },
            },
            {
                "name": "MLP",
                "params": {
                    "hidden_layer_sizes": (64,),
                    "activation": "tanh",
                    "alpha": 0.1,
                    "learning_rate_init": 1e-05,
                    "batch_size": 32,
                },
            },
            {
                "name": "MLP",
                "params": {
                    "hidden_layer_sizes": (64,),
                    "activation": "relu",
                    "alpha": 0.1,
                    "learning_rate_init": 1e-05,
                    "batch_size": 512,
                },
            },
            {
                "name": "MLP",
                "params": {
                    "hidden_layer_sizes": (64,),
                    "activation": "tanh",
                    "alpha": 1e-07,
                    "learning_rate_init": 1e-05,
                    "batch_size": "auto",
                },
            },
            {
                "name": "MLP",
                "params": {
                    "hidden_layer_sizes": (64,),
                    "activation": "relu",
                    "alpha": 0.1,
                    "learning_rate_init": 0.01,
                    "batch_size": 32,
                },
            },
            {
                "name": "MLP",
                "params": {
                    "hidden_layer_sizes": (64,),
                    "activation": "tanh",
                    "alpha": 1e-07,
                    "learning_rate_init": 0.01,
                    "batch_size": 32,
                },
            },
            {
                "name": "MLP",
                "params": {
                    "hidden_layer_sizes": (64,),
                    "activation": "relu",
                    "alpha": 1e-07,
                    "learning_rate_init": 1e-05,
                    "batch_size": 512,
                },
            },
            {
                "name": "MLP",
                "params": {
                    "hidden_layer_sizes": (64,),
                    "activation": "relu",
                    "alpha": 0.1,
                    "learning_rate_init": 0.01,
                    "batch_size": "auto",
                },
            },
            {
                "name": "MLP",
                "params": {
                    "hidden_layer_sizes": (64,),
                    "activation": "tanh",
                    "alpha": 1e-07,
                    "learning_rate_init": 1e-05,
                    "batch_size": 512,
                },
            },
            {
                "name": "MLP",
                "params": {
                    "hidden_layer_sizes": (64,),
                    "activation": "tanh",
                    "alpha": 1e-07,
                    "learning_rate_init": 0.01,
                    "batch_size": "auto",
                },
            },
        ],
    )
    ALL_MULTICLASS_MODELS = choose_catalogue(
        "ALL_MULTICLASS_MODELS",
        ALL_MULTICLASS_MODELS,
        [
            {"name": "Multiclass Baseline", "params": {}},
            {"name": "Multinomial Logistic Regression", "params": {"class_weight": None}},
            {"name": "Multinomial Logistic Regression", "params": {"class_weight": "balanced"}},
            {
                "name": "L2 Multinomial Logistic Regression",
                "params": {"C": 1e-05, "class_weight": None},
            },
            {
                "name": "L2 Multinomial Logistic Regression",
                "params": {"C": 1e-05, "class_weight": "balanced"},
            },
            {
                "name": "L2 Multinomial Logistic Regression",
                "params": {"C": 1000, "class_weight": None},
            },
            {
                "name": "L2 Multinomial Logistic Regression",
                "params": {"C": 1000, "class_weight": "balanced"},
            },
            {
                "name": "L1 Multinomial Logistic Regression",
                "params": {"C": 1e-05, "class_weight": None},
            },
            {
                "name": "L1 Multinomial Logistic Regression",
                "params": {"C": 1e-05, "class_weight": "balanced"},
            },
            {
                "name": "L1 Multinomial Logistic Regression",
                "params": {"C": 1000, "class_weight": None},
            },
            {
                "name": "L1 Multinomial Logistic Regression",
                "params": {"C": 1000, "class_weight": "balanced"},
            },
            {
                "name": "Elastic Net Multinomial Logistic Regression",
                "params": {"C": 1e-05, "l1_ratio": 0.01, "class_weight": None},
            },
            {
                "name": "Elastic Net Multinomial Logistic Regression",
                "params": {"C": 1e-05, "l1_ratio": 0.01, "class_weight": "balanced"},
            },
            {
                "name": "Elastic Net Multinomial Logistic Regression",
                "params": {"C": 1e-05, "l1_ratio": 0.99, "class_weight": None},
            },
            {
                "name": "Elastic Net Multinomial Logistic Regression",
                "params": {"C": 1e-05, "l1_ratio": 0.99, "class_weight": "balanced"},
            },
            {
                "name": "Elastic Net Multinomial Logistic Regression",
                "params": {"C": 1000, "l1_ratio": 0.01, "class_weight": None},
            },
            {
                "name": "Elastic Net Multinomial Logistic Regression",
                "params": {"C": 1000, "l1_ratio": 0.01, "class_weight": "balanced"},
            },
            {
                "name": "Elastic Net Multinomial Logistic Regression",
                "params": {"C": 1000, "l1_ratio": 0.99, "class_weight": None},
            },
            {
                "name": "Elastic Net Multinomial Logistic Regression",
                "params": {"C": 1000, "l1_ratio": 0.99, "class_weight": "balanced"},
            },
            {"name": "LDA SVD", "params": {"solver": "svd"}},
            {"name": "LDA LSQR/Eigen", "params": {"solver": "lsqr", "shrinkage": None}},
            {"name": "LDA LSQR/Eigen", "params": {"solver": "lsqr", "shrinkage": "auto"}},
            {"name": "LDA LSQR/Eigen", "params": {"solver": "lsqr", "shrinkage": 0.05}},
            {"name": "LDA LSQR/Eigen", "params": {"solver": "lsqr", "shrinkage": 0.95}},
            {"name": "LDA LSQR/Eigen", "params": {"solver": "eigen", "shrinkage": None}},
            {"name": "LDA LSQR/Eigen", "params": {"solver": "eigen", "shrinkage": "auto"}},
            {"name": "LDA LSQR/Eigen", "params": {"solver": "eigen", "shrinkage": 0.05}},
            {"name": "LDA LSQR/Eigen", "params": {"solver": "eigen", "shrinkage": 0.95}},
            {"name": "QDA", "params": {"reg_param": 0}},
            {"name": "QDA", "params": {"reg_param": 1.0}},
            {
                "name": "Hist Gradient Boosting",
                "params": {
                    "learning_rate": 0.005,
                    "max_iter": 100,
                    "max_leaf_nodes": 7,
                    "max_depth": None,
                    "min_samples_leaf": 5,
                    "l2_regularization": 0,
                    "class_weight": None,
                },
            },
            {
                "name": "Hist Gradient Boosting",
                "params": {
                    "learning_rate": 0.2,
                    "max_iter": 1500,
                    "max_leaf_nodes": 127,
                    "max_depth": 10,
                    "min_samples_leaf": 200,
                    "l2_regularization": 100,
                    "class_weight": "balanced",
                },
            },
            {
                "name": "Hist Gradient Boosting",
                "params": {
                    "learning_rate": 0.005,
                    "max_iter": 100,
                    "max_leaf_nodes": 7,
                    "max_depth": 3,
                    "min_samples_leaf": 200,
                    "l2_regularization": 100,
                    "class_weight": "balanced",
                },
            },
            {
                "name": "Hist Gradient Boosting",
                "params": {
                    "learning_rate": 0.2,
                    "max_iter": 1500,
                    "max_leaf_nodes": 127,
                    "max_depth": 3,
                    "min_samples_leaf": 5,
                    "l2_regularization": 0,
                    "class_weight": None,
                },
            },
            {
                "name": "Hist Gradient Boosting",
                "params": {
                    "learning_rate": 0.005,
                    "max_iter": 100,
                    "max_leaf_nodes": 127,
                    "max_depth": None,
                    "min_samples_leaf": 5,
                    "l2_regularization": 100,
                    "class_weight": "balanced",
                },
            },
            {
                "name": "Hist Gradient Boosting",
                "params": {
                    "learning_rate": 0.2,
                    "max_iter": 1500,
                    "max_leaf_nodes": 7,
                    "max_depth": 10,
                    "min_samples_leaf": 200,
                    "l2_regularization": 0,
                    "class_weight": None,
                },
            },
            {
                "name": "Hist Gradient Boosting",
                "params": {
                    "learning_rate": 0.005,
                    "max_iter": 100,
                    "max_leaf_nodes": 127,
                    "max_depth": 3,
                    "min_samples_leaf": 200,
                    "l2_regularization": 0,
                    "class_weight": None,
                },
            },
            {
                "name": "Hist Gradient Boosting",
                "params": {
                    "learning_rate": 0.2,
                    "max_iter": 1500,
                    "max_leaf_nodes": 7,
                    "max_depth": None,
                    "min_samples_leaf": 5,
                    "l2_regularization": 100,
                    "class_weight": "balanced",
                },
            },
            {
                "name": "Hist Gradient Boosting",
                "params": {
                    "learning_rate": 0.005,
                    "max_iter": 1500,
                    "max_leaf_nodes": 7,
                    "max_depth": 10,
                    "min_samples_leaf": 5,
                    "l2_regularization": 0,
                    "class_weight": "balanced",
                },
            },
            {
                "name": "Hist Gradient Boosting",
                "params": {
                    "learning_rate": 0.2,
                    "max_iter": 100,
                    "max_leaf_nodes": 127,
                    "max_depth": None,
                    "min_samples_leaf": 200,
                    "l2_regularization": 100,
                    "class_weight": None,
                },
            },
            {
                "name": "Hist Gradient Boosting",
                "params": {
                    "learning_rate": 0.005,
                    "max_iter": 1500,
                    "max_leaf_nodes": 7,
                    "max_depth": 3,
                    "min_samples_leaf": 5,
                    "l2_regularization": 100,
                    "class_weight": None,
                },
            },
            {
                "name": "Hist Gradient Boosting",
                "params": {
                    "learning_rate": 0.2,
                    "max_iter": 100,
                    "max_leaf_nodes": 127,
                    "max_depth": 10,
                    "min_samples_leaf": 5,
                    "l2_regularization": 0,
                    "class_weight": "balanced",
                },
            },
            {
                "name": "Hist Gradient Boosting",
                "params": {
                    "learning_rate": 0.005,
                    "max_iter": 1500,
                    "max_leaf_nodes": 127,
                    "max_depth": None,
                    "min_samples_leaf": 200,
                    "l2_regularization": 0,
                    "class_weight": "balanced",
                },
            },
            {
                "name": "Hist Gradient Boosting",
                "params": {
                    "learning_rate": 0.2,
                    "max_iter": 100,
                    "max_leaf_nodes": 7,
                    "max_depth": 10,
                    "min_samples_leaf": 5,
                    "l2_regularization": 100,
                    "class_weight": None,
                },
            },
            {
                "name": "Gradient Boosting",
                "params": {
                    "learning_rate": 0.005,
                    "n_estimators": 100,
                    "max_depth": 2,
                    "min_samples_leaf": 5,
                    "subsample": 0.5,
                    "max_features": None,
                },
            },
            {
                "name": "Gradient Boosting",
                "params": {
                    "learning_rate": 0.2,
                    "n_estimators": 1000,
                    "max_depth": 8,
                    "min_samples_leaf": 100,
                    "subsample": 1.0,
                    "max_features": 0.8,
                },
            },
            {
                "name": "Gradient Boosting",
                "params": {
                    "learning_rate": 0.005,
                    "n_estimators": 100,
                    "max_depth": 2,
                    "min_samples_leaf": 100,
                    "subsample": 1.0,
                    "max_features": "sqrt",
                },
            },
            {
                "name": "Gradient Boosting",
                "params": {
                    "learning_rate": 0.2,
                    "n_estimators": 1000,
                    "max_depth": 8,
                    "min_samples_leaf": 5,
                    "subsample": 0.5,
                    "max_features": 0.3,
                },
            },
            {
                "name": "Gradient Boosting",
                "params": {
                    "learning_rate": 0.005,
                    "n_estimators": 100,
                    "max_depth": 8,
                    "min_samples_leaf": 5,
                    "subsample": 1.0,
                    "max_features": 0.3,
                },
            },
            {
                "name": "Gradient Boosting",
                "params": {
                    "learning_rate": 0.2,
                    "n_estimators": 1000,
                    "max_depth": 2,
                    "min_samples_leaf": 100,
                    "subsample": 0.5,
                    "max_features": None,
                },
            },
            {
                "name": "Gradient Boosting",
                "params": {
                    "learning_rate": 0.005,
                    "n_estimators": 100,
                    "max_depth": 8,
                    "min_samples_leaf": 100,
                    "subsample": 0.5,
                    "max_features": 0.8,
                },
            },
            {
                "name": "Gradient Boosting",
                "params": {
                    "learning_rate": 0.2,
                    "n_estimators": 1000,
                    "max_depth": 2,
                    "min_samples_leaf": 5,
                    "subsample": 1.0,
                    "max_features": "sqrt",
                },
            },
            {
                "name": "Gradient Boosting",
                "params": {
                    "learning_rate": 0.005,
                    "n_estimators": 100,
                    "max_depth": 2,
                    "min_samples_leaf": 5,
                    "subsample": 1.0,
                    "max_features": 0.8,
                },
            },
            {
                "name": "Gradient Boosting",
                "params": {
                    "learning_rate": 0.2,
                    "n_estimators": 1000,
                    "max_depth": 8,
                    "min_samples_leaf": 100,
                    "subsample": 0.5,
                    "max_features": "sqrt",
                },
            },
            {
                "name": "Gradient Boosting",
                "params": {
                    "learning_rate": 0.005,
                    "n_estimators": 100,
                    "max_depth": 2,
                    "min_samples_leaf": 100,
                    "subsample": 0.5,
                    "max_features": 0.3,
                },
            },
            {
                "name": "Gradient Boosting",
                "params": {
                    "learning_rate": 0.2,
                    "n_estimators": 1000,
                    "max_depth": 8,
                    "min_samples_leaf": 5,
                    "subsample": 1.0,
                    "max_features": None,
                },
            },
            {
                "name": "Random Forest",
                "params": {
                    "n_estimators": 200,
                    "max_depth": None,
                    "min_samples_leaf": 1,
                    "min_samples_split": 2,
                    "max_features": "sqrt",
                    "bootstrap": True,
                    "class_weight": None,
                },
            },
            {
                "name": "Random Forest",
                "params": {
                    "n_estimators": 1500,
                    "max_depth": 30,
                    "min_samples_leaf": 100,
                    "min_samples_split": 50,
                    "max_features": 1.0,
                    "bootstrap": False,
                    "class_weight": "balanced",
                },
            },
            {
                "name": "Random Forest",
                "params": {
                    "n_estimators": 200,
                    "max_depth": 5,
                    "min_samples_leaf": 1,
                    "min_samples_split": 2,
                    "max_features": 0.2,
                    "bootstrap": False,
                    "class_weight": "balanced",
                },
            },
            {
                "name": "Random Forest",
                "params": {
                    "n_estimators": 1500,
                    "max_depth": None,
                    "min_samples_leaf": 100,
                    "min_samples_split": 50,
                    "max_features": 0.2,
                    "bootstrap": True,
                    "class_weight": None,
                },
            },
            {
                "name": "Random Forest",
                "params": {
                    "n_estimators": 200,
                    "max_depth": 5,
                    "min_samples_leaf": 100,
                    "min_samples_split": 50,
                    "max_features": "sqrt",
                    "bootstrap": True,
                    "class_weight": "balanced",
                },
            },
            {
                "name": "Random Forest",
                "params": {
                    "n_estimators": 1500,
                    "max_depth": 5,
                    "min_samples_leaf": 1,
                    "min_samples_split": 50,
                    "max_features": "sqrt",
                    "bootstrap": False,
                    "class_weight": None,
                },
            },
            {
                "name": "Random Forest",
                "params": {
                    "n_estimators": 200,
                    "max_depth": 30,
                    "min_samples_leaf": 1,
                    "min_samples_split": 2,
                    "max_features": 1.0,
                    "bootstrap": True,
                    "class_weight": "balanced",
                },
            },
            {
                "name": "Random Forest",
                "params": {
                    "n_estimators": 1500,
                    "max_depth": None,
                    "min_samples_leaf": 100,
                    "min_samples_split": 2,
                    "max_features": 1.0,
                    "bootstrap": False,
                    "class_weight": None,
                },
            },
            {
                "name": "Random Forest",
                "params": {
                    "n_estimators": 200,
                    "max_depth": 30,
                    "min_samples_leaf": 1,
                    "min_samples_split": 50,
                    "max_features": 0.2,
                    "bootstrap": True,
                    "class_weight": None,
                },
            },
            {
                "name": "Random Forest",
                "params": {
                    "n_estimators": 1500,
                    "max_depth": 5,
                    "min_samples_leaf": 100,
                    "min_samples_split": 2,
                    "max_features": "sqrt",
                    "bootstrap": False,
                    "class_weight": "balanced",
                },
            },
            {
                "name": "Random Forest",
                "params": {
                    "n_estimators": 200,
                    "max_depth": None,
                    "min_samples_leaf": 1,
                    "min_samples_split": 50,
                    "max_features": 1.0,
                    "bootstrap": False,
                    "class_weight": None,
                },
            },
            {
                "name": "Random Forest",
                "params": {
                    "n_estimators": 1500,
                    "max_depth": 30,
                    "min_samples_leaf": 100,
                    "min_samples_split": 2,
                    "max_features": 0.2,
                    "bootstrap": True,
                    "class_weight": "balanced",
                },
            },
            {
                "name": "Random Forest",
                "params": {
                    "n_estimators": 200,
                    "max_depth": None,
                    "min_samples_leaf": 100,
                    "min_samples_split": 50,
                    "max_features": 0.2,
                    "bootstrap": False,
                    "class_weight": "balanced",
                },
            },
            {
                "name": "Random Forest",
                "params": {
                    "n_estimators": 1500,
                    "max_depth": 5,
                    "min_samples_leaf": 1,
                    "min_samples_split": 2,
                    "max_features": 1.0,
                    "bootstrap": True,
                    "class_weight": None,
                },
            },
            {
                "name": "XGBoost",
                "params": {
                    "n_estimators": 200,
                    "learning_rate": 0.005,
                    "max_depth": 2,
                    "min_child_weight": 1,
                    "subsample": 0.5,
                    "colsample_bytree": 0.4,
                    "gamma": 0,
                    "reg_alpha": 0,
                    "reg_lambda": 0,
                    "class_weight": None,
                },
            },
            {
                "name": "XGBoost",
                "params": {
                    "n_estimators": 1500,
                    "learning_rate": 0.2,
                    "max_depth": 10,
                    "min_child_weight": 50,
                    "subsample": 1.0,
                    "colsample_bytree": 1.0,
                    "gamma": 5,
                    "reg_alpha": 10,
                    "reg_lambda": 100,
                    "class_weight": "balanced",
                },
            },
            {
                "name": "XGBoost",
                "params": {
                    "n_estimators": 200,
                    "learning_rate": 0.005,
                    "max_depth": 2,
                    "min_child_weight": 1,
                    "subsample": 0.5,
                    "colsample_bytree": 1.0,
                    "gamma": 5,
                    "reg_alpha": 10,
                    "reg_lambda": 100,
                    "class_weight": "balanced",
                },
            },
            {
                "name": "XGBoost",
                "params": {
                    "n_estimators": 1500,
                    "learning_rate": 0.2,
                    "max_depth": 10,
                    "min_child_weight": 50,
                    "subsample": 1.0,
                    "colsample_bytree": 0.4,
                    "gamma": 0,
                    "reg_alpha": 0,
                    "reg_lambda": 0,
                    "class_weight": None,
                },
            },
            {
                "name": "XGBoost",
                "params": {
                    "n_estimators": 200,
                    "learning_rate": 0.005,
                    "max_depth": 2,
                    "min_child_weight": 50,
                    "subsample": 1.0,
                    "colsample_bytree": 0.4,
                    "gamma": 0,
                    "reg_alpha": 0,
                    "reg_lambda": 100,
                    "class_weight": "balanced",
                },
            },
            {
                "name": "XGBoost",
                "params": {
                    "n_estimators": 1500,
                    "learning_rate": 0.2,
                    "max_depth": 10,
                    "min_child_weight": 1,
                    "subsample": 0.5,
                    "colsample_bytree": 1.0,
                    "gamma": 5,
                    "reg_alpha": 10,
                    "reg_lambda": 0,
                    "class_weight": None,
                },
            },
            {
                "name": "XGBoost",
                "params": {
                    "n_estimators": 200,
                    "learning_rate": 0.005,
                    "max_depth": 2,
                    "min_child_weight": 50,
                    "subsample": 1.0,
                    "colsample_bytree": 0.4,
                    "gamma": 5,
                    "reg_alpha": 10,
                    "reg_lambda": 0,
                    "class_weight": None,
                },
            },
            {
                "name": "XGBoost",
                "params": {
                    "n_estimators": 1500,
                    "learning_rate": 0.2,
                    "max_depth": 10,
                    "min_child_weight": 1,
                    "subsample": 0.5,
                    "colsample_bytree": 1.0,
                    "gamma": 0,
                    "reg_alpha": 0,
                    "reg_lambda": 100,
                    "class_weight": "balanced",
                },
            },
            {
                "name": "XGBoost",
                "params": {
                    "n_estimators": 200,
                    "learning_rate": 0.005,
                    "max_depth": 10,
                    "min_child_weight": 1,
                    "subsample": 1.0,
                    "colsample_bytree": 0.4,
                    "gamma": 0,
                    "reg_alpha": 10,
                    "reg_lambda": 0,
                    "class_weight": "balanced",
                },
            },
            {
                "name": "XGBoost",
                "params": {
                    "n_estimators": 1500,
                    "learning_rate": 0.2,
                    "max_depth": 2,
                    "min_child_weight": 50,
                    "subsample": 0.5,
                    "colsample_bytree": 1.0,
                    "gamma": 5,
                    "reg_alpha": 0,
                    "reg_lambda": 100,
                    "class_weight": None,
                },
            },
            {
                "name": "XGBoost",
                "params": {
                    "n_estimators": 200,
                    "learning_rate": 0.005,
                    "max_depth": 10,
                    "min_child_weight": 1,
                    "subsample": 1.0,
                    "colsample_bytree": 0.4,
                    "gamma": 5,
                    "reg_alpha": 0,
                    "reg_lambda": 100,
                    "class_weight": None,
                },
            },
            {
                "name": "XGBoost",
                "params": {
                    "n_estimators": 1500,
                    "learning_rate": 0.2,
                    "max_depth": 2,
                    "min_child_weight": 50,
                    "subsample": 0.5,
                    "colsample_bytree": 1.0,
                    "gamma": 0,
                    "reg_alpha": 10,
                    "reg_lambda": 0,
                    "class_weight": "balanced",
                },
            },
            {
                "name": "XGBoost",
                "params": {
                    "n_estimators": 200,
                    "learning_rate": 0.005,
                    "max_depth": 10,
                    "min_child_weight": 50,
                    "subsample": 0.5,
                    "colsample_bytree": 0.4,
                    "gamma": 0,
                    "reg_alpha": 10,
                    "reg_lambda": 100,
                    "class_weight": None,
                },
            },
            {
                "name": "XGBoost",
                "params": {
                    "n_estimators": 1500,
                    "learning_rate": 0.2,
                    "max_depth": 2,
                    "min_child_weight": 1,
                    "subsample": 1.0,
                    "colsample_bytree": 1.0,
                    "gamma": 5,
                    "reg_alpha": 0,
                    "reg_lambda": 0,
                    "class_weight": "balanced",
                },
            },
            {
                "name": "XGBoost",
                "params": {
                    "n_estimators": 200,
                    "learning_rate": 0.005,
                    "max_depth": 10,
                    "min_child_weight": 50,
                    "subsample": 0.5,
                    "colsample_bytree": 0.4,
                    "gamma": 5,
                    "reg_alpha": 0,
                    "reg_lambda": 0,
                    "class_weight": "balanced",
                },
            },
            {
                "name": "XGBoost",
                "params": {
                    "n_estimators": 1500,
                    "learning_rate": 0.2,
                    "max_depth": 2,
                    "min_child_weight": 1,
                    "subsample": 1.0,
                    "colsample_bytree": 1.0,
                    "gamma": 0,
                    "reg_alpha": 10,
                    "reg_lambda": 100,
                    "class_weight": None,
                },
            },
            {
                "name": "XGBoost",
                "params": {
                    "n_estimators": 200,
                    "learning_rate": 0.005,
                    "max_depth": 2,
                    "min_child_weight": 1,
                    "subsample": 1.0,
                    "colsample_bytree": 0.4,
                    "gamma": 0,
                    "reg_alpha": 10,
                    "reg_lambda": 100,
                    "class_weight": None,
                },
            },
            {
                "name": "XGBoost",
                "params": {
                    "n_estimators": 1500,
                    "learning_rate": 0.2,
                    "max_depth": 10,
                    "min_child_weight": 50,
                    "subsample": 0.5,
                    "colsample_bytree": 1.0,
                    "gamma": 5,
                    "reg_alpha": 0,
                    "reg_lambda": 0,
                    "class_weight": "balanced",
                },
            },
            {
                "name": "XGBoost",
                "params": {
                    "n_estimators": 200,
                    "learning_rate": 0.005,
                    "max_depth": 2,
                    "min_child_weight": 1,
                    "subsample": 1.0,
                    "colsample_bytree": 0.4,
                    "gamma": 5,
                    "reg_alpha": 0,
                    "reg_lambda": 0,
                    "class_weight": "balanced",
                },
            },
            {
                "name": "XGBoost",
                "params": {
                    "n_estimators": 1500,
                    "learning_rate": 0.2,
                    "max_depth": 10,
                    "min_child_weight": 50,
                    "subsample": 0.5,
                    "colsample_bytree": 1.0,
                    "gamma": 0,
                    "reg_alpha": 10,
                    "reg_lambda": 100,
                    "class_weight": None,
                },
            },
            {
                "name": "LightGBM",
                "params": {
                    "n_estimators": 200,
                    "learning_rate": 0.005,
                    "num_leaves": 7,
                    "max_depth": -1,
                    "min_child_samples": 5,
                    "subsample": 0.5,
                    "colsample_bytree": 0.4,
                    "reg_alpha": 0,
                    "reg_lambda": 0,
                    "class_weight": None,
                },
            },
            {
                "name": "LightGBM",
                "params": {
                    "n_estimators": 1500,
                    "learning_rate": 0.2,
                    "num_leaves": 255,
                    "max_depth": 16,
                    "min_child_samples": 200,
                    "subsample": 1.0,
                    "colsample_bytree": 1.0,
                    "reg_alpha": 10,
                    "reg_lambda": 100,
                    "class_weight": "balanced",
                },
            },
            {
                "name": "LightGBM",
                "params": {
                    "n_estimators": 200,
                    "learning_rate": 0.005,
                    "num_leaves": 7,
                    "max_depth": -1,
                    "min_child_samples": 5,
                    "subsample": 1.0,
                    "colsample_bytree": 1.0,
                    "reg_alpha": 10,
                    "reg_lambda": 100,
                    "class_weight": "balanced",
                },
            },
            {
                "name": "LightGBM",
                "params": {
                    "n_estimators": 1500,
                    "learning_rate": 0.2,
                    "num_leaves": 255,
                    "max_depth": 16,
                    "min_child_samples": 200,
                    "subsample": 0.5,
                    "colsample_bytree": 0.4,
                    "reg_alpha": 0,
                    "reg_lambda": 0,
                    "class_weight": None,
                },
            },
            {
                "name": "LightGBM",
                "params": {
                    "n_estimators": 200,
                    "learning_rate": 0.005,
                    "num_leaves": 7,
                    "max_depth": 16,
                    "min_child_samples": 200,
                    "subsample": 0.5,
                    "colsample_bytree": 0.4,
                    "reg_alpha": 0,
                    "reg_lambda": 100,
                    "class_weight": "balanced",
                },
            },
            {
                "name": "LightGBM",
                "params": {
                    "n_estimators": 1500,
                    "learning_rate": 0.2,
                    "num_leaves": 255,
                    "max_depth": -1,
                    "min_child_samples": 5,
                    "subsample": 1.0,
                    "colsample_bytree": 1.0,
                    "reg_alpha": 10,
                    "reg_lambda": 0,
                    "class_weight": None,
                },
            },
            {
                "name": "LightGBM",
                "params": {
                    "n_estimators": 200,
                    "learning_rate": 0.005,
                    "num_leaves": 7,
                    "max_depth": 16,
                    "min_child_samples": 200,
                    "subsample": 0.5,
                    "colsample_bytree": 1.0,
                    "reg_alpha": 10,
                    "reg_lambda": 0,
                    "class_weight": None,
                },
            },
            {
                "name": "LightGBM",
                "params": {
                    "n_estimators": 1500,
                    "learning_rate": 0.2,
                    "num_leaves": 255,
                    "max_depth": -1,
                    "min_child_samples": 5,
                    "subsample": 1.0,
                    "colsample_bytree": 0.4,
                    "reg_alpha": 0,
                    "reg_lambda": 100,
                    "class_weight": "balanced",
                },
            },
            {
                "name": "LightGBM",
                "params": {
                    "n_estimators": 200,
                    "learning_rate": 0.005,
                    "num_leaves": 255,
                    "max_depth": -1,
                    "min_child_samples": 200,
                    "subsample": 0.5,
                    "colsample_bytree": 0.4,
                    "reg_alpha": 10,
                    "reg_lambda": 0,
                    "class_weight": "balanced",
                },
            },
            {
                "name": "LightGBM",
                "params": {
                    "n_estimators": 1500,
                    "learning_rate": 0.2,
                    "num_leaves": 7,
                    "max_depth": 16,
                    "min_child_samples": 5,
                    "subsample": 1.0,
                    "colsample_bytree": 1.0,
                    "reg_alpha": 0,
                    "reg_lambda": 100,
                    "class_weight": None,
                },
            },
            {
                "name": "LightGBM",
                "params": {
                    "n_estimators": 200,
                    "learning_rate": 0.005,
                    "num_leaves": 255,
                    "max_depth": -1,
                    "min_child_samples": 200,
                    "subsample": 0.5,
                    "colsample_bytree": 1.0,
                    "reg_alpha": 0,
                    "reg_lambda": 100,
                    "class_weight": None,
                },
            },
            {
                "name": "LightGBM",
                "params": {
                    "n_estimators": 1500,
                    "learning_rate": 0.2,
                    "num_leaves": 7,
                    "max_depth": 16,
                    "min_child_samples": 5,
                    "subsample": 1.0,
                    "colsample_bytree": 0.4,
                    "reg_alpha": 10,
                    "reg_lambda": 0,
                    "class_weight": "balanced",
                },
            },
            {
                "name": "LightGBM",
                "params": {
                    "n_estimators": 200,
                    "learning_rate": 0.005,
                    "num_leaves": 255,
                    "max_depth": 16,
                    "min_child_samples": 5,
                    "subsample": 0.5,
                    "colsample_bytree": 0.4,
                    "reg_alpha": 10,
                    "reg_lambda": 100,
                    "class_weight": None,
                },
            },
            {
                "name": "LightGBM",
                "params": {
                    "n_estimators": 1500,
                    "learning_rate": 0.2,
                    "num_leaves": 7,
                    "max_depth": -1,
                    "min_child_samples": 200,
                    "subsample": 1.0,
                    "colsample_bytree": 1.0,
                    "reg_alpha": 0,
                    "reg_lambda": 0,
                    "class_weight": "balanced",
                },
            },
            {
                "name": "LightGBM",
                "params": {
                    "n_estimators": 200,
                    "learning_rate": 0.005,
                    "num_leaves": 255,
                    "max_depth": 16,
                    "min_child_samples": 5,
                    "subsample": 0.5,
                    "colsample_bytree": 1.0,
                    "reg_alpha": 0,
                    "reg_lambda": 0,
                    "class_weight": "balanced",
                },
            },
            {
                "name": "LightGBM",
                "params": {
                    "n_estimators": 1500,
                    "learning_rate": 0.2,
                    "num_leaves": 7,
                    "max_depth": -1,
                    "min_child_samples": 200,
                    "subsample": 1.0,
                    "colsample_bytree": 0.4,
                    "reg_alpha": 10,
                    "reg_lambda": 100,
                    "class_weight": None,
                },
            },
            {
                "name": "LightGBM",
                "params": {
                    "n_estimators": 200,
                    "learning_rate": 0.005,
                    "num_leaves": 7,
                    "max_depth": -1,
                    "min_child_samples": 200,
                    "subsample": 0.5,
                    "colsample_bytree": 0.4,
                    "reg_alpha": 10,
                    "reg_lambda": 100,
                    "class_weight": None,
                },
            },
            {
                "name": "LightGBM",
                "params": {
                    "n_estimators": 1500,
                    "learning_rate": 0.2,
                    "num_leaves": 255,
                    "max_depth": 16,
                    "min_child_samples": 5,
                    "subsample": 1.0,
                    "colsample_bytree": 1.0,
                    "reg_alpha": 0,
                    "reg_lambda": 0,
                    "class_weight": "balanced",
                },
            },
            {
                "name": "LightGBM",
                "params": {
                    "n_estimators": 200,
                    "learning_rate": 0.005,
                    "num_leaves": 7,
                    "max_depth": -1,
                    "min_child_samples": 200,
                    "subsample": 0.5,
                    "colsample_bytree": 1.0,
                    "reg_alpha": 0,
                    "reg_lambda": 0,
                    "class_weight": "balanced",
                },
            },
            {
                "name": "LightGBM",
                "params": {
                    "n_estimators": 1500,
                    "learning_rate": 0.2,
                    "num_leaves": 255,
                    "max_depth": 16,
                    "min_child_samples": 5,
                    "subsample": 1.0,
                    "colsample_bytree": 0.4,
                    "reg_alpha": 10,
                    "reg_lambda": 100,
                    "class_weight": None,
                },
            },
            {
                "name": "SVM Linear",
                "params": {"kernel": "linear", "C": 1e-05, "class_weight": None},
            },
            {
                "name": "SVM Linear",
                "params": {"kernel": "linear", "C": 1e-05, "class_weight": "balanced"},
            },
            {"name": "SVM Linear", "params": {"kernel": "linear", "C": 1000, "class_weight": None}},
            {
                "name": "SVM Linear",
                "params": {"kernel": "linear", "C": 1000, "class_weight": "balanced"},
            },
            {
                "name": "SVM RBF",
                "params": {"kernel": "rbf", "C": 1e-05, "gamma": "scale", "class_weight": None},
            },
            {
                "name": "SVM RBF",
                "params": {"kernel": "rbf", "C": 1000, "gamma": 1, "class_weight": "balanced"},
            },
            {
                "name": "SVM RBF",
                "params": {
                    "kernel": "rbf",
                    "C": 1e-05,
                    "gamma": "auto",
                    "class_weight": "balanced",
                },
            },
            {
                "name": "SVM RBF",
                "params": {"kernel": "rbf", "C": 1000, "gamma": 0.0001, "class_weight": None},
            },
            {
                "name": "SVM RBF",
                "params": {
                    "kernel": "rbf",
                    "C": 1e-05,
                    "gamma": "scale",
                    "class_weight": "balanced",
                },
            },
            {
                "name": "SVM RBF",
                "params": {"kernel": "rbf", "C": 1000, "gamma": "auto", "class_weight": None},
            },
            {
                "name": "SVM RBF",
                "params": {"kernel": "rbf", "C": 1e-05, "gamma": 0.0001, "class_weight": None},
            },
            {
                "name": "SVM RBF",
                "params": {"kernel": "rbf", "C": 1e-05, "gamma": 1, "class_weight": "balanced"},
            },
            {
                "name": "MLP",
                "params": {
                    "hidden_layer_sizes": (64,),
                    "activation": "relu",
                    "alpha": 1e-07,
                    "learning_rate_init": 1e-05,
                    "batch_size": 32,
                },
            },
            {
                "name": "MLP",
                "params": {
                    "hidden_layer_sizes": (64,),
                    "activation": "tanh",
                    "alpha": 0.1,
                    "learning_rate_init": 0.01,
                    "batch_size": "auto",
                },
            },
            {
                "name": "MLP",
                "params": {
                    "hidden_layer_sizes": (64,),
                    "activation": "relu",
                    "alpha": 1e-07,
                    "learning_rate_init": 0.01,
                    "batch_size": 512,
                },
            },
            {
                "name": "MLP",
                "params": {
                    "hidden_layer_sizes": (64,),
                    "activation": "tanh",
                    "alpha": 0.1,
                    "learning_rate_init": 1e-05,
                    "batch_size": 32,
                },
            },
            {
                "name": "MLP",
                "params": {
                    "hidden_layer_sizes": (64,),
                    "activation": "relu",
                    "alpha": 0.1,
                    "learning_rate_init": 1e-05,
                    "batch_size": 512,
                },
            },
            {
                "name": "MLP",
                "params": {
                    "hidden_layer_sizes": (64,),
                    "activation": "tanh",
                    "alpha": 1e-07,
                    "learning_rate_init": 1e-05,
                    "batch_size": "auto",
                },
            },
            {
                "name": "MLP",
                "params": {
                    "hidden_layer_sizes": (64,),
                    "activation": "relu",
                    "alpha": 0.1,
                    "learning_rate_init": 0.01,
                    "batch_size": 32,
                },
            },
            {
                "name": "MLP",
                "params": {
                    "hidden_layer_sizes": (64,),
                    "activation": "tanh",
                    "alpha": 1e-07,
                    "learning_rate_init": 0.01,
                    "batch_size": 32,
                },
            },
            {
                "name": "MLP",
                "params": {
                    "hidden_layer_sizes": (64,),
                    "activation": "relu",
                    "alpha": 1e-07,
                    "learning_rate_init": 1e-05,
                    "batch_size": 512,
                },
            },
            {
                "name": "MLP",
                "params": {
                    "hidden_layer_sizes": (64,),
                    "activation": "relu",
                    "alpha": 0.1,
                    "learning_rate_init": 0.01,
                    "batch_size": "auto",
                },
            },
            {
                "name": "MLP",
                "params": {
                    "hidden_layer_sizes": (64,),
                    "activation": "tanh",
                    "alpha": 1e-07,
                    "learning_rate_init": 1e-05,
                    "batch_size": 512,
                },
            },
            {
                "name": "MLP",
                "params": {
                    "hidden_layer_sizes": (64,),
                    "activation": "tanh",
                    "alpha": 1e-07,
                    "learning_rate_init": 0.01,
                    "batch_size": "auto",
                },
            },
            {"name": "Ordinal Regression", "params": {"alpha": 1e-05}},
            {"name": "Ordinal Regression", "params": {"alpha": 100}},
        ],
    )
    logger.info(
        "Configurations available | Continuous: %d | Binary: %d | Multiclass: %d",
        len(ALL_CONTINUOUS_MODELS),
        len(ALL_BINARY_MODELS),
        len(ALL_MULTICLASS_MODELS),
    )

    def main():
        logger.info("Beginning target processing")
        for target_number, target in enumerate(targets, start=1):
            new_target = False
            _portfolio_type = TARGET_PORTFOLIO_TYPES[target]
            logger.info(
                "[%d/%d] Target: %s | Starting | Portfolio type: %s | Selection role: %s",
                target_number,
                len(targets),
                target,
                _portfolio_type,
                portfolio_selection_role(_portfolio_type),
            )
            if target in USED_CONTINUOUS_MODELS.keys():
                _type = "continuous"
                MODELS_TO_DO = [
                    model
                    for model in ALL_CONTINUOUS_MODELS
                    if model not in USED_CONTINUOUS_MODELS[target]
                ]
            elif target in USED_BINARY_MODELS.keys():
                _type = "binary"
                MODELS_TO_DO = [
                    model for model in ALL_BINARY_MODELS if model not in USED_BINARY_MODELS[target]
                ]
            elif target in USED_MULTICLASS_MODELS.keys():
                _type = "multiclass"
                MODELS_TO_DO = [
                    model
                    for model in ALL_MULTICLASS_MODELS
                    if model not in USED_MULTICLASS_MODELS[target]
                ]
            else:
                new_target = True
                logger.info(
                    "Target: %s | No previous validation results found | New target", target
                )
            if not new_target:
                logger.info(
                    "Target: %s | Existing %s target | %d model configurations remaining",
                    target,
                    _type,
                    len(MODELS_TO_DO),
                )
                if not MODELS_TO_DO and (not testing_recommendations):
                    logger.info(
                        "Target: %s | No new model configurations to test | Skipping", target
                    )
                    continue
            try:
                with sqlite3.connect(VALIDATION_DATABASE_PATH) as conn:
                    previous_results = pd.read_sql_query(
                        f'\n                    SELECT *\n                    FROM "{target}__search"\n                    ',
                        conn,
                    )
                    no_table = False
            except Exception:
                previous_results = pd.DataFrame()
                no_table = True
            features = selected_features[target]
            columns = ["Date", "Ticker", target]
            columns.extend(features)
            logger.info("Target: %s | Loading %d selected features", target, len(features))
            with sqlite3.connect(FEATURE_DATABASE_PATH) as conn:
                columns = ", ".join((f'"{column}"' for column in columns))
                df = research_rows(
                    pd.read_sql_query(
                        f'\n                SELECT {columns}\n                FROM "{STOCK_TYPE}"\n                ',
                        conn,
                    )
                )
            logger.info(
                "Target: %s | Data loaded | Rows: %d | Columns: %d",
                target,
                len(df),
                len(df.columns),
            )
            missing_by_column = df.isna().sum()
            missing_by_column = missing_by_column[missing_by_column > 0].sort_values(
                ascending=False
            )
            duplicate_feature_count = len(features) - len(set(features))
            constant_features = [
                feature for feature in features if df[feature].nunique(dropna=True) <= 1
            ]
            logger.info(
                "Target: %s | Feature checks | Duplicate feature names: %d | Constant/all-missing features: %d",
                target,
                duplicate_feature_count,
                len(constant_features),
            )
            if constant_features:
                logger.warning(
                    "Target: %s | Constant/all-missing selected features: %s",
                    target,
                    constant_features,
                )
            if new_target:
                _type = target_type(df, target)
                logger.info("Target: %s | Detected target type: %s", target, _type)
            else:
                logger.info("Target: %s | Target type: %s", target, _type)
            if new_target:
                if _type == "continuous":
                    MODELS_TO_DO = unique_models(ALL_CONTINUOUS_MODELS)
                elif _type == "binary":
                    MODELS_TO_DO = unique_models(ALL_BINARY_MODELS)
                elif _type == "multiclass":
                    MODELS_TO_DO = unique_models(ALL_MULTICLASS_MODELS)
            model_source = full_model_source(_type)
            model_lookup = {model["name"]: model for model in model_source}
            if testing_recommendations and (not no_table):
                recommended_models, recommendation_midpoint_column = recommend_models_to_fit(
                    previous_results,
                    _type,
                    n=N_RECOMMENDATIONS,
                    x=N_MODELS,
                    model_lookup=model_lookup,
                )
                if not MODELS_TO_DO:
                    MODELS_TO_DO = recommended_models
                else:
                    MODELS_TO_DO.extend(recommended_models)
            if MODELS_TO_DO:
                logger.info(
                    "Target: %s | Ready to test %d configurations", target, len(MODELS_TO_DO)
                )
            else:
                logger.info("Target: %s | No configurations to test", target)
                continue
            purge_days = target_purge_days(target)
            for model in MODELS_TO_DO:
                original_model = model_lookup[model["name"]]
                model["function"] = original_model["function"]
                model["scaled"] = original_model["scaled"]
            rows_before_dropna = len(df)
            logger.info(
                "Target: %s | Missing-value cleaning complete | Rows before: %d | Rows removed: %d | Rows remaining: %d",
                target,
                rows_before_dropna,
                rows_before_dropna - len(df),
                len(df),
            )
            new_summary = walk_forward(
                models_to_do=MODELS_TO_DO,
                df=df,
                features=features,
                target=target,
                purge_days=purge_days,
                type=_type,
                portfolio_type=_portfolio_type,
                previous_results=previous_results,
                no_table=no_table,
                validation_window=get_setting("VALIDATION_WINDOW", 20),
            )
            if new_summary.empty:
                logger.info("%s | No surviving model configurations", target)
                continue
            new_summary = rank_validation_results(new_summary, target, _type, _portfolio_type)
            new_summary = add_testing_eligibility(
                new_summary,
                _type,
                required_validation_folds(
                    df,
                    get_setting("VALIDATION_WINDOW", 20),
                    get_setting("MIN_VALIDATION_FOLDS", 15),
                ),
            )
            new_summary["TPE Score?"] = pd.to_numeric(
                new_summary["TPE Score?"], errors="coerce"
            ).fillna(0)
            new_summary["TPE Score?"] = new_summary.groupby("Model")["TPE Score?"].transform("max")
            if testing_recommendations and (not no_table):
                new_summary = add_recommendation_midpoints(
                    new_summary=new_summary,
                    recommended_models=recommended_models,
                    recommendation_midpoint_column=recommendation_midpoint_column,
                )
            with sqlite3.connect(VALIDATION_DATABASE_PATH) as conn:
                write_frame(
                    new_summary, f"{target}__search", conn, if_exists="replace", index=False
                )
            test_eligible_results = new_summary[new_summary["Testing Eligible"]].copy()
            test_eligible_results = test_eligible_results.drop(columns=["TPE Score?", "Midpoint?"])
            with sqlite3.connect(VALIDATION_DATABASE_PATH) as conn:
                write_frame(
                    test_eligible_results, f"{target}", conn, if_exists="replace", index=False
                )

    main = callback("main", main)

    if (
        not MULTICLASS_MODELS
        and (not BINARY_MODELS)
        and (not CONTINUOUS_MODELS)
        and (not testing_recommendations)
    ):
        logger.info("No new continuous, binary, or multiclass model families remain to be tested")
    else:
        if not MULTICLASS_MODELS:
            logger.info("No new multiclass model families to test")
        if not BINARY_MODELS:
            logger.info("No new binary model families to test")
        if not CONTINUOUS_MODELS:
            logger.info("No new continuous model families to test")
        logger.info("At least one model type has configurations remaining to test")
        main()

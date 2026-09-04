from equity_selector.settings import setting as get_setting, callback, choose_model_row

"""final_test: original sequential research stage; shared logic is in equity_selector."""

from equity_selector.config import data_root
from models import *
from main_package import *


def run():
    global \
        FEATURE_DATABASE_PATH, \
        FINAL_RESULTS_DATABASE, \
        SELECTED_FEATURES_PATH, \
        SOURCE_TABLE_COLUMNS, \
        STOCK_TYPE, \
        StandardScaler, \
        VALIDATION_DATABASE_PATH, \
        _clip_quality, \
        _normalise_column_name, \
        _row_metric, \
        _weighted_available_mean, \
        ast, \
        binary, \
        calculate_quality_score, \
        choose_test_eligible_model, \
        clean_binary_target, \
        clean_final_result, \
        column, \
        columns, \
        conn, \
        connection, \
        continuous, \
        current_df, \
        data_connection, \
        display_test_eligible_models, \
        eligible_metric_columns, \
        eligible_models_by_target, \
        error, \
        features, \
        file, \
        final_errors, \
        final_errors_df, \
        final_results, \
        final_results_df, \
        final_target_type, \
        final_test, \
        get_model_function, \
        get_models, \
        get_selected_model_config, \
        json, \
        json_default, \
        load_test_eligible_models, \
        logger, \
        logging, \
        missing_columns, \
        missing_horizons, \
        model_name, \
        model_ranking_rules, \
        multiclass, \
        np, \
        parameters, \
        parameters_to_json, \
        parse_selected_parameters, \
        pd, \
        portfolio_target_type, \
        predictability_score, \
        purge_training_data, \
        quote_sql_identifier, \
        rank_eligible_models, \
        relative_quality_score, \
        result, \
        results, \
        rows_before_dropna, \
        select_all_test_models, \
        selected_features, \
        selected_model_from_row, \
        selected_test_models, \
        selection, \
        source_table_info, \
        sql_columns, \
        sqlite3, \
        stock_type_index, \
        target, \
        target_category, \
        target_horizon, \
        target_number, \
        target_query, \
        target_type_from_name, \
        targets, \
        test_df, \
        testing_eligible_mask, \
        train_df, \
        useful, \
        validation_df, \
        write_frame
    from equity_selector.database import write_frame
    import sqlite3
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    import json
    import numpy as np
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logger = logging.getLogger(__name__)
    FEATURE_DATABASE_PATH = str(data_root()) + "/Features_Targets_Data.db"
    STOCK_TYPE = get_setting("STOCK_TYPE", "High Liquidity 30")
    VALIDATION_DATABASE_PATH = (
        f"{str(data_root()) + '/Validation_Model_Fits/'}{STOCK_TYPE.replace(' ', '_')}.db"
    )
    SELECTED_FEATURES_PATH = str(data_root()) + "/Selected_Features.txt"

    def quote_sql_identifier(identifier):
        return '"' + str(identifier).replace('"', '""') + '"'

    quote_sql_identifier = callback("quote_sql_identifier", quote_sql_identifier)

    from equity_selector.feature_mapping import load_feature_mapping

    selected_features = load_feature_mapping(data_root() / "Selected_Features.txt", STOCK_TYPE)

    def testing_eligible_mask(series):
        if pd.api.types.is_bool_dtype(series):
            return series.fillna(False)
        if pd.api.types.is_numeric_dtype(series):
            return pd.to_numeric(series, errors="coerce").fillna(0).eq(1)
        return series.astype(str).str.strip().str.lower().isin(["true", "1", "yes"])

    testing_eligible_mask = callback("testing_eligible_mask", testing_eligible_mask)

    def model_ranking_rules(target_type):
        if target_type == "continuous":
            return [
                ("Rank IC Mean", False),
                ("Rank IC Std", True),
                ("NRMSE Mean", True),
                ("RMSE Mean", True),
                ("MAE Mean", True),
                ("R2 Mean", False),
            ]
        if target_type == "binary":
            return [
                ("ROC AUC Mean", False),
                ("ROC AUC Std", True),
                ("PR AUC Mean", False),
                ("F1 Mean", False),
                ("Log Loss Mean", True),
            ]
        if target_type == "multiclass":
            return [
                ("Macro F1 Mean", False),
                ("Macro F1 Std", True),
                ("Balanced Accuracy Mean", False),
                ("Log Loss Mean", True),
            ]
        raise ValueError(f"Unknown target type: {target_type}")

    model_ranking_rules = callback("model_ranking_rules", model_ranking_rules)

    def target_type_from_name(target):
        if (
            target.startswith("Three Class Direction")
            or target.startswith("Barrier")
            or target.startswith("Volatility Barrier")
        ):
            return "multiclass"
        if (
            target.startswith("Future Direction")
            or target.startswith("Future Return Above")
            or target.startswith("Top 20 Percent Future Return")
            or target.startswith("Top 25 Percent Future Return")
        ):
            return "binary"
        return "continuous"

    target_type_from_name = callback("target_type_from_name", target_type_from_name)

    def rank_eligible_models(leaderboard, target_type):
        available_rules = [
            (column, ascending)
            for column, ascending in model_ranking_rules(target_type)
            if column in leaderboard.columns
        ]
        if not available_rules:
            return leaderboard.copy()
        ranked = leaderboard.sort_values(
            by=[column for column, _ in available_rules],
            ascending=[ascending for _, ascending in available_rules],
            na_position="last",
            kind="stable",
        ).copy()
        ranked["Test Selection Rank"] = np.arange(1, len(ranked) + 1)
        return ranked

    rank_eligible_models = callback("rank_eligible_models", rank_eligible_models)

    def load_test_eligible_models(selected_features, validation_database_path):
        eligible_by_target = {}
        with sqlite3.connect(validation_database_path) as validation_connection:
            table_names = pd.read_sql_query(
                "\n            SELECT name\n            FROM sqlite_master\n            WHERE type = 'table'\n            ",
                validation_connection,
            )["name"].tolist()
            table_names = set(table_names)
            for target in selected_features.keys():
                if target not in table_names:
                    continue
                leaderboard = pd.read_sql_query(
                    f"\n                SELECT *\n                FROM {quote_sql_identifier(target)}\n                ",
                    validation_connection,
                )
                if leaderboard.empty:
                    continue
                eligible = rank_eligible_models(
                    leaderboard=leaderboard, target_type=target_type_from_name(target)
                )
                eligible_by_target[target] = eligible.reset_index(drop=True)
        return eligible_by_target

    load_test_eligible_models = callback("load_test_eligible_models", load_test_eligible_models)

    eligible_models_by_target = load_test_eligible_models(
        selected_features=selected_features, validation_database_path=VALIDATION_DATABASE_PATH
    )
    targets = [target for target in selected_features.keys() if target in eligible_models_by_target]
    logger.info("Validation database: %s", VALIDATION_DATABASE_PATH)
    logger.info("Test-eligible targets: %d / %d", len(targets), len(selected_features))
    if not targets:
        from equity_selector.results import save_final_test_results

        save_final_test_results(data_root() / "Final_Test_Results.db", STOCK_TYPE, [], [])
        logger.info("No test-eligible targets remain")
        return
    data_connection = sqlite3.connect(FEATURE_DATABASE_PATH)
    data_connection.execute("PRAGMA query_only = ON")
    source_table_info = pd.read_sql_query(
        f"PRAGMA table_info({quote_sql_identifier(STOCK_TYPE)})", data_connection
    )
    if source_table_info.empty:
        raise ValueError(f"Source table {STOCK_TYPE!r} does not exist in {FEATURE_DATABASE_PATH}")
    SOURCE_TABLE_COLUMNS = set(source_table_info["name"].tolist())
    logger.info(
        "%s | Source table ready | %d columns | full table will not be loaded",
        STOCK_TYPE,
        len(SOURCE_TABLE_COLUMNS),
    )

    def get_model_function(function_name):
        function = globals().get(function_name)
        if function is None:
            raise NameError(f"{function_name} has not been imported yet.")
        return function

    get_model_function = callback("get_model_function", get_model_function)

    def purge_training_data(train_df, purge_days):
        if purge_days <= 0:
            return train_df.copy()
        dates = np.sort(train_df["Date"].unique())
        if len(dates) <= purge_days:
            return train_df.iloc[0:0].copy()
        purge_start_date = dates[-purge_days]
        return train_df[train_df["Date"] < purge_start_date].copy()

    purge_training_data = callback("purge_training_data", purge_training_data)

    def final_test(train_df, validation_df, test_df, features, target, model_name, parameters):
        _type = final_target_type(target)
        logger.info("%s | Final Test | Type=%s | Model=%s", target, _type, model_name)
        model_config = get_selected_model_config(model_name=model_name, target_type=_type)
        fit_function = get_model_function(model_config["function"])
        logger.info(
            "%s | Function=%s | Scaled=%s", target, model_config["function"], model_config["scaled"]
        )
        final_train_df = pd.concat([train_df, validation_df], ignore_index=True)
        purge_days = target_purge_days(target)
        rows_before_purge = len(final_train_df)
        final_train_df = purge_training_data(final_train_df, purge_days)
        logger.info(
            "%s | Train rows=%d -> %d after purge | Test rows=%d | Purge=%d",
            target,
            rows_before_purge,
            len(final_train_df),
            len(test_df),
            purge_days,
        )
        final_train_df = final_train_df.replace([np.inf, -np.inf], np.nan).dropna(
            subset=features + [target]
        )
        test_df = test_df.replace([np.inf, -np.inf], np.nan).dropna(subset=features + [target])
        if final_train_df.empty or test_df.empty:
            raise ValueError(f"{target}: no valid final training/test rows")
        x_train = final_train_df[features]
        y_train = final_train_df[target]
        x_test = test_df[features]
        y_test = test_df[target]
        if _type == "binary":
            y_train = clean_binary_target(y_train, target)
            y_test = clean_binary_target(y_test, target)
        if _type in ("binary", "multiclass"):
            train_classes = np.sort(y_train.dropna().unique())
            test_classes = np.sort(y_test.dropna().unique())
            logger.info(
                "%s | Train classes=%s | Test classes=%s", target, train_classes, test_classes
            )
        if model_config["scaled"]:
            scaler = StandardScaler()
            x_train = pd.DataFrame(
                scaler.fit_transform(x_train), columns=features, index=x_train.index
            )
            x_test = pd.DataFrame(scaler.transform(x_test), columns=features, index=x_test.index)
        logger.info("%s | Fitting final model", target)
        result = fit_function(x_train, y_train, x_test, y_test, **parameters)
        if result is None:
            result = {}
        clean_result = clean_final_result(
            result=result,
            target=target,
            target_type=_type,
            model_name=model_name,
            parameters=parameters,
        )
        logger.info(
            "%s | Complete | %s",
            target,
            {key: value for key, value in clean_result.items() if key not in ["Parameters"]},
        )
        return clean_result

    final_test = callback("final_test", final_test)

    def get_selected_model_config(model_name, target_type):
        models = get_models(target_type)
        for model_config in models:
            if model_config["name"] == model_name:
                return model_config
        raise ValueError(f"{model_name} not found for target type {target_type}")

    get_selected_model_config = callback("get_selected_model_config", get_selected_model_config)

    def json_default(value):
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        return str(value)

    json_default = callback("json_default", json_default)

    from equity_selector.parameters import parameters_to_json

    def clean_binary_target(y, target):
        y = y.copy()
        if target.startswith("Future Direction"):
            y = y.where(y > 0, -1)
            y = y.where(y <= 0, 1)
        classes = np.sort(pd.Series(y).dropna().unique())
        if len(classes) > 2:
            raise ValueError(f"{target} is defined as binary but contains classes {classes}")
        return y

    clean_binary_target = callback("clean_binary_target", clean_binary_target)

    def final_target_type(target):
        return target_type_from_name(target)

    final_target_type = callback("final_target_type", final_target_type)

    def get_models(target_type):
        model_source = full_model_source(target_type)
        return model_source

    get_models = callback("get_models", get_models)

    def clean_final_result(result, target, target_type, model_name, parameters):
        clean = {
            "Target": target,
            "Target Type": target_type,
            "Model": model_name,
            "Parameters": parameters_to_json(parameters),
        }
        if target_type == "continuous":
            metrics = ["RMSE", "MAE", "R2", "Rank IC"]
        elif target_type == "binary":
            metrics = ["ROC AUC", "PR AUC", "Log Loss", "F1"]
        elif target_type == "multiclass":
            metrics = ["Macro F1", "Balanced Accuracy", "Log Loss"]
        else:
            raise ValueError(f"Unknown target type: {target_type}")
        for metric in metrics:
            if metric in result:
                clean[metric] = result[metric]
        return clean

    clean_final_result = callback("clean_final_result", clean_final_result)

    def predictability_score(row):
        if row["Target Type"] == "continuous":
            return max(row["Rank IC"], 0)
        elif row["Target Type"] == "binary":
            auc_edge = max(row["ROC AUC"] - 0.5, 0) * 2
            return 0.5 * auc_edge + 0.3 * row["PR AUC"] + 0.2 * row["F1"]
        elif row["Target Type"] == "multiclass":
            return row["Macro F1"]
        return 0

    predictability_score = callback("predictability_score", predictability_score)

    def target_category(target):
        target = target.lower()
        downside_keywords = [
            "downside",
            "adverse",
            "minimum return",
            "drawdown",
            "time to maximum adverse",
        ]
        if any((word in target for word in downside_keywords)):
            return "downside"
        volatility_keywords = ["volatility", "variance", "absolute return"]
        if any((word in target for word in volatility_keywords)):
            return "volatility"
        return "alpha"

    target_category = callback("target_category", target_category)

    from equity_selector.parameters import parse_parameters as parse_selected_parameters

    def eligible_metric_columns(target_type, columns):
        if target_type == "continuous":
            wanted = [
                "Rank IC Mean",
                "Rank IC Std",
                "NRMSE Mean",
                "RMSE Mean",
                "MAE Mean",
                "R2 Mean",
            ]
        elif target_type == "binary":
            wanted = ["ROC AUC Mean", "PR AUC Mean", "F1 Mean", "ROC AUC Std"]
        elif target_type == "multiclass":
            wanted = ["Macro F1 Mean", "Balanced Accuracy Mean", "Log Loss Mean", "Macro F1 Std"]
        else:
            wanted = []
        return [column for column in wanted if column in columns]

    eligible_metric_columns = callback("eligible_metric_columns", eligible_metric_columns)

    def selected_model_from_row(row):
        model_name = row["Model"]
        parameters = parse_selected_parameters(row["Parameters"])
        return (model_name, parameters)

    selected_model_from_row = callback("selected_model_from_row", selected_model_from_row)

    def display_test_eligible_models(target, eligible_models):
        target_type = final_target_type(target)
        metric_columns = eligible_metric_columns(
            target_type=target_type, columns=eligible_models.columns
        )
        print("\n" + "=" * 100)
        print(f"{target} | {len(eligible_models)} TEST-ELIGIBLE MODELS")
        print("=" * 100)
        for option_number, (_, row) in enumerate(eligible_models.iterrows(), start=1):
            parts = [f"[{option_number}]", str(row["Model"])]
            rank = row.get("Test Selection Rank")
            if pd.notna(rank):
                parts.insert(1, f"Rank={int(rank)}")
            for metric in metric_columns:
                value = row.get(metric)
                if pd.notna(value):
                    parts.append(f"{metric}={value:.6f}")
            print(" | ".join(parts))

    display_test_eligible_models = callback(
        "display_test_eligible_models", display_test_eligible_models
    )

    def choose_test_eligible_model(target, eligible_models):
        return selected_model_from_row(choose_model_row(eligible_models, target))

    choose_test_eligible_model = callback("choose_test_eligible_model", choose_test_eligible_model)

    def select_all_test_models(targets, eligible_models_by_target):
        selections = {}
        print("\n" + "=" * 100)
        print("FINAL TEST MODEL SELECTION")
        print("=" * 100)
        mode = get_setting("MODEL_SELECTION_MODE", "rank_one")
        if mode not in {"rank_one", "explicit"}:
            raise ValueError("MODEL_SELECTION_MODE must be rank_one or explicit")
        auto_rank_one = mode == "rank_one"
        for target_number, target in enumerate(targets, start=1):
            eligible_models = eligible_models_by_target[target]
            print(f"\n\nTARGET [{target_number}/{len(targets)}]")
            if auto_rank_one:
                selected = eligible_models.sort_values("Test Selection Rank").iloc[0]
                model_name, parameters = selected_model_from_row(selected)
                print(f"{target}\nAUTO SELECTED: Rank 1 | {model_name}")
            else:
                model_name, parameters = choose_test_eligible_model(
                    target=target, eligible_models=eligible_models
                )
            selections[target] = {"Model": model_name, "Parameters": parameters}
        print("\n\n" + "=" * 100)
        print("ALL FINAL TEST MODELS SELECTED")
        print("=" * 100)
        for target_number, target in enumerate(targets, start=1):
            selection = selections[target]
            print(f"[{target_number}/{len(targets)}] {target} | {selection['Model']}")
        print("\nSelection complete. No further user input is required.")
        print("Starting all final tests...\n")
        return selections

    select_all_test_models = callback("select_all_test_models", select_all_test_models)

    selected_test_models = select_all_test_models(
        targets=targets, eligible_models_by_target=eligible_models_by_target
    )
    final_results = []
    final_errors = []
    logger.info("Starting final testing | %d test-eligible targets", len(targets))
    for target_number, target in enumerate(targets, start=1):
        try:
            features = selected_features[target]
            logger.info(
                "[%d/%d] %s | Starting | %d features",
                target_number,
                len(targets),
                target,
                len(features),
            )
            if len(features) == 0:
                logger.warning("%s | Skipped | No selected features", target)
                continue
            selection = selected_test_models[target]
            model_name = selection["Model"]
            parameters = selection["Parameters"]
            logger.info(
                "%s | Pre-selected model=%s | Parameters=%s", target, model_name, parameters
            )
            columns = ["Date", "Ticker", target] + features
            columns = list(dict.fromkeys(columns))
            missing_columns = [column for column in columns if column not in SOURCE_TABLE_COLUMNS]
            if len(missing_columns) > 0:
                raise KeyError(f"{target} is missing source columns: {missing_columns}")
            sql_columns = ", ".join((quote_sql_identifier(column) for column in columns))
            target_query = f"SELECT {sql_columns} FROM {quote_sql_identifier(STOCK_TYPE)}"
            logger.info(
                "%s | Loading %d/%d source columns", target, len(columns), len(SOURCE_TABLE_COLUMNS)
            )
            current_df = pd.read_sql_query(target_query, data_connection)
            logger.info(
                "%s | Loaded %d rows x %d columns | %.2f MB",
                target,
                len(current_df),
                len(current_df.columns),
                current_df.memory_usage(deep=True).sum() / 1024**2,
            )
            rows_before_dropna = len(current_df)
            logger.info("%s | Valid rows=%d/%d", target, len(current_df), rows_before_dropna)
            current_df["Date"] = pd.to_datetime(current_df["Date"])
            current_df = current_df.sort_values(["Date", "Ticker"]).reset_index(drop=True)
            train_df, validation_df, test_df = train_validation_test_split(current_df)
            logger.info(
                "%s | Split | Train=%d | Validation=%d | Test=%d",
                target,
                len(train_df),
                len(validation_df),
                len(test_df),
            )
            result = final_test(
                train_df=train_df,
                validation_df=validation_df,
                test_df=test_df,
                features=features,
                target=target,
                model_name=model_name,
                parameters=parameters,
            )
            final_results.append(result)
        except Exception as error:
            logger.exception("%s | FAILED", target)
            final_errors.append({"Target": target, "Error": str(error)})
    data_connection.close()
    logger.info("Source database connection closed")
    from equity_selector.results import save_final_test_results

    FINAL_RESULTS_DATABASE = str(data_root() / "Final_Test_Results.db")
    final_results_df = save_final_test_results(
        FINAL_RESULTS_DATABASE, STOCK_TYPE, final_results, final_errors
    )
    if final_results_df.empty:
        logger.info("No successful final-test models remain | failures=%d", len(final_errors))
        return
    logger.info(
        "Final testing complete | Successful=%d | Failed=%d", len(final_results), len(final_errors)
    )
    conn = sqlite3.connect(str(data_root()) + "/Final_Test_Results.db")
    results = pd.read_sql_query(f"SELECT * FROM 'Final Test Results {STOCK_TYPE}'", conn)
    results = results[~results["Model"].str.contains("Baseline", case=False, na=False)]
    try:
        continuous = results["Target Type"].eq("continuous") & (
            results["Rank IC"] >= get_setting("MIN_RANK_IC", 0.1)
        )
    except KeyError:
        continuous = pd.Series(False, index=results.index)
    try:
        binary = (
            results["Target Type"].eq("binary")
            & (results["ROC AUC"] >= get_setting("MIN_ROC_AUC", 0.6))
            & (results["PR AUC"] >= get_setting("MIN_PR_AUC", 0.2))
        )
    except KeyError:
        binary = pd.Series(False, index=results.index)
    try:
        multiclass = results["Target Type"].eq("multiclass") & (
            results["Macro F1"] >= get_setting("MIN_MACRO_F1", 0.45)
        )
    except KeyError:
        multiclass = pd.Series(False, index=results.index)
    useful = results[continuous | binary | multiclass].copy()
    useful["Prediction Type"] = useful["Target"].apply(target_category)
    useful["Predictability Score"] = useful.apply(predictability_score, axis=1)

    def calculate_quality_score(row, portfolio_type):
        statistical_type = str(row.get("Target Type", "")).strip().lower()
        existing_predictability = _row_metric(row, ["Predictability Score"])
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
            r2 = _row_metric(row, ["R2", "R^2", "R Squared", "R2 Score", "Test R2", "Test R^2"])
            q_ic = _clip_quality(abs(rank_ic) / 0.3) if rank_ic is not None else None
            q_r2 = _clip_quality(max(r2, 0.0) / 0.2) if r2 is not None else None
            quality = _weighted_available_mean([(0.7, q_ic), (0.3, q_r2)])
            if quality is not None:
                return _clip_quality(quality)
        if statistical_type == "binary":
            roc_auc = _row_metric(row, ["ROC AUC", "ROC-AUC", "AUC ROC", "AUC", "Test ROC AUC"])
            pr_auc = _row_metric(
                row,
                ["PR AUC", "PR-AUC", "Average Precision", "Average Precision Score", "Test PR AUC"],
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
            q_roc = _clip_quality((roc_auc - 0.5) / 0.25) if roc_auc is not None else None
            q_pr = None
            if pr_auc is not None and positive_rate is not None and (0.0 <= positive_rate < 1.0):
                excellent_pr = min(1.0, positive_rate + 0.3)
                denominator = max(excellent_pr - positive_rate, 1e-12)
                q_pr = _clip_quality((pr_auc - positive_rate) / denominator)
            event_types = {"TAIL_EVENT", "VOLATILITY_EVENT", "UPSIDE_EVENT"}
            if portfolio_type in event_types:
                weights = [(0.4, q_roc), (0.6, q_pr)]
            else:
                weights = [(0.6, q_roc), (0.4, q_pr)]
            quality = _weighted_available_mean(weights)
            if quality is not None:
                return _clip_quality(quality)
        if statistical_type == "multiclass":
            macro_f1 = _row_metric(
                row, ["Macro F1", "Macro-F1", "F1 Macro", "Macro F1 Score", "Test Macro F1"]
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
                row, ["Number Classes", "Number of Classes", "N Classes", "Num Classes"]
            )
            if number_classes is None or number_classes < 2:
                number_classes = 3.0
            chance_f1 = 1.0 / number_classes
            excellent_f1 = 0.7
            q_f1 = None
            if macro_f1 is not None:
                denominator = max(excellent_f1 - chance_f1, 1e-12)
                q_f1 = _clip_quality((macro_f1 - chance_f1) / denominator)
            q_auc = _clip_quality((macro_auc - 0.5) / 0.25) if macro_auc is not None else None
            quality = _weighted_available_mean([(0.7, q_f1), (0.3, q_auc)])
            if quality is not None:
                return _clip_quality(quality)
        if existing_predictability is not None:
            return _clip_quality(existing_predictability)
        return 0.0

    calculate_quality_score = callback("calculate_quality_score", calculate_quality_score)

    def target_horizon(target):
        name = str(target).strip().lower()
        explicit = re.findall(
            "(?<![a-z0-9])(\\d+(?:\\.\\d+)?)\\s*(m|min|mins|minute|minutes|h|hr|hrs|hour|hours|d|day|days|w|week|weeks)(?![a-z])",
            name,
        )
        if explicit:
            value = float(explicit[-1][0])
            return int(value) if value.is_integer() else value
        numbers = re.findall("(?<![a-z0-9.])(\\d+(?:\\.\\d+)?)(?![a-z0-9.%])", name)
        if not numbers:
            return np.nan
        value = float(numbers[-1])
        return int(value) if value.is_integer() else value

    target_horizon = callback("target_horizon", target_horizon)

    def portfolio_target_type(target, prediction_type=None):
        name = str(target).strip().lower()
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

    def _normalise_column_name(value):
        return re.sub("[^a-z0-9]+", "", str(value).lower())

    _normalise_column_name = callback("_normalise_column_name", _normalise_column_name)

    def _row_metric(row, aliases):
        column_lookup = {_normalise_column_name(column): column for column in row.index}
        for alias in aliases:
            key = _normalise_column_name(alias)
            if key not in column_lookup:
                continue
            value = pd.to_numeric(pd.Series([row[column_lookup[key]]]), errors="coerce").iloc[0]
            if pd.notna(value):
                return float(value)
        return None

    _row_metric = callback("_row_metric", _row_metric)

    def _clip_quality(value):
        return float(np.clip(value, 0.0, 1.0))

    _clip_quality = callback("_clip_quality", _clip_quality)

    def _weighted_available_mean(values):
        available = [
            (weight, value) for weight, value in values if value is not None and np.isfinite(value)
        ]
        if not available:
            return None
        total_weight = sum((weight for weight, _ in available))
        return sum((weight * value for weight, value in available)) / total_weight

    _weighted_available_mean = callback("_weighted_available_mean", _weighted_available_mean)

    useful["Portfolio Target Type"] = useful.apply(
        lambda row: portfolio_target_type(row["Target"], row.get("Prediction Type", "")), axis=1
    )
    useful["Horizon"] = useful["Target"].map(target_horizon)
    useful["Absolute Quality Score"] = useful.apply(
        lambda row: calculate_quality_score(row, row["Portfolio Target Type"]), axis=1
    )

    def relative_quality_score(group):
        number_targets = len(group)
        if number_targets == 1:
            return pd.Series(0.5, index=group.index, dtype=float)
        lower_bound = 0.5 / number_targets
        upper_bound = 1.0 - lower_bound
        absolute_quality = group["Absolute Quality Score"].astype(float)
        minimum_quality = absolute_quality.min()
        maximum_quality = absolute_quality.max()
        if np.isclose(maximum_quality, minimum_quality):
            return pd.Series(0.5, index=group.index, dtype=float)
        relative_position = (absolute_quality - minimum_quality) / (
            maximum_quality - minimum_quality
        )
        relative_quality = lower_bound + relative_position * (upper_bound - lower_bound)
        return relative_quality

    relative_quality_score = callback("relative_quality_score", relative_quality_score)

    useful["Quality Score"] = useful.groupby("Portfolio Target Type", group_keys=False)[
        "Absolute Quality Score"
    ].transform(lambda scores: relative_quality_score(useful.loc[scores.index]))
    missing_horizons = useful[useful["Horizon"].isna()]
    if not missing_horizons.empty:
        logger.warning(
            "%d selected targets do not contain a parseable numeric horizon. Their Horizon value will be NULL: %s",
            len(missing_horizons),
            ", ".join(missing_horizons["Target"].astype(str)),
        )
    useful = useful.sort_values(["Target Type", "Quality Score"], ascending=[True, False])
    with sqlite3.connect(FINAL_RESULTS_DATABASE) as connection:
        write_frame(
            useful,
            f"{STOCK_TYPE} Passed Test Results",
            connection,
            if_exists="replace",
            index=False,
        )

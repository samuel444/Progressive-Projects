from equity_selector.settings import setting as get_setting, callback, run_screen_schedule

"""horizons: original sequential research stage; shared logic is in equity_selector."""

from equity_selector.config import data_root


def run():
    global \
        ACTIVE_ANALYSIS_TYPES, \
        ACTIVE_HORIZON_SCORE_VALUES, \
        ANALYSIS_MODE, \
        BASELINE_CONCENTRATION_PENALTY, \
        BASELINE_MAX_WEIGHT, \
        BASELINE_TYPE_CONFIGURATION, \
        BINARY_PORTFOLIO_TARGET_TYPES, \
        BQ, \
        CONFIGURATION_CORRELATION_THRESHOLD, \
        CONTINUOUS_PORTFOLIO_TARGET_TYPES, \
        DAILY_HORIZONS, \
        DATA_DIR, \
        FEATURE_DATABASE, \
        FINAL_RESULTS_DB, \
        HORIZON_DIR, \
        HORIZON_FREEZE_BQ_RANGE, \
        HORIZON_INDEX, \
        HORIZON_SCORES, \
        HORIZON_SCORE_VALUES, \
        HORIZON_TEST_CONCENTRATION_PENALTIES, \
        HORIZON_TEST_MAX_WEIGHTS, \
        HORIZON_TEST_SETTINGS, \
        HORIZON_TEST_TYPE_CONFIGURATIONS, \
        HORIZON_VALIDATION_FRACTION, \
        INTRADAY_HORIZONS, \
        MULTICLASS_PORTFOLIO_TARGET_TYPES, \
        NEAR_BEST_BQ_TOLERANCE, \
        PORTFOLIO_TYPE_GROUPS, \
        Path, \
        REBALANCE_MULTIPLIER, \
        RESEARCH_END, \
        RESEARCH_SPLITS, \
        RESULT_TABLES, \
        SELECTED_FEATURES_FILE, \
        SOURCE_TABLE_COLUMNS, \
        SOURCE_TABLE_COLUMN_SET, \
        STOCK_TYPE, \
        STOCK_TYPE_INDICES, \
        TARGET_ORIENTATION, \
        USE_CARTESIAN_HORIZON_TEST_SETTINGS, \
        _, \
        active_parameters, \
        actual_train_end, \
        analysis_fit_end_index, \
        analysis_max_horizon, \
        analysis_models, \
        analysis_type, \
        apply_horizon_signal_refresh, \
        apply_type_configuration, \
        ascending, \
        ast, \
        available_tables, \
        backtest_quality, \
        benchmark_close, \
        benchmark_download, \
        benchmark_metrics, \
        best_setting, \
        bq_by_setting, \
        candidate_key, \
        candidate_standard_deviation, \
        candidate_vector, \
        choose_analysis_mode, \
        combination, \
        combination_number, \
        concentration_penalty, \
        configuration, \
        configuration_vector, \
        connection, \
        correlated_configurations_skipped, \
        correlation, \
        create_models_and_predictions, \
        dataframe_memory_mb, \
        direction_types, \
        exact_duplicates_skipped, \
        exhaustive_results, \
        exhaustive_total, \
        feature, \
        features, \
        file, \
        fit_end, \
        fit_end_index, \
        fit_start, \
        freeze_parameter, \
        frozen_df, \
        frozen_score, \
        gc, \
        get_horizon_score, \
        get_target_orientation, \
        highly_correlated, \
        horizon, \
        horizon_key, \
        horizons, \
        initial_predictions_df, \
        iters, \
        itertools, \
        load_target_period, \
        log_every, \
        logger, \
        logging, \
        market_df, \
        market_results, \
        math, \
        max_selected_horizon, \
        max_weight, \
        maximum_range_quality, \
        mean_quality, \
        metadata_connection, \
        metrics, \
        missing_base_columns, \
        missing_feature_targets, \
        missing_result_columns, \
        model_number, \
        model_row, \
        new_iters, \
        new_thres, \
        normalize_horizon_configuration, \
        normalize_horizon_scores_by_type_group, \
        normalized_configuration, \
        normalized_top_configurations, \
        np, \
        num_of_configs, \
        one_model_df, \
        parameters, \
        pd, \
        possible_values, \
        prediction_parts, \
        prediction_to_signal, \
        predictions_df, \
        prepared, \
        qualities, \
        qualities_by_setting, \
        quality, \
        quote_sql_identifier, \
        random_screen, \
        ranges_by_setting, \
        reference_horizon_score, \
        required_base_columns, \
        required_features, \
        required_result_columns, \
        research_dates, \
        research_dates_df, \
        research_end_sql, \
        research_split, \
        result, \
        result_part, \
        result_parts, \
        result_table, \
        results, \
        row, \
        run_portfolio_backtest_from_predictions, \
        score, \
        seen_normalized_configurations, \
        selected_feature_line, \
        selected_feature_lines, \
        selected_features, \
        selected_models_df, \
        selected_standard_deviation, \
        selected_targets, \
        selected_vector, \
        selected_vectors, \
        sensitivity_results, \
        setting, \
        setting_mode, \
        setting_number, \
        setting_qualities, \
        setting_range, \
        sort_columns, \
        sqlite3, \
        stock_type_index, \
        table_info, \
        target, \
        target_configurations, \
        target_features, \
        target_metrics, \
        target_model_data, \
        target_predictions, \
        target_sql_columns, \
        target_train_df, \
        target_type, \
        target_validation_df, \
        target_values, \
        test_df, \
        test_results, \
        thres, \
        total_configurations, \
        train_df, \
        train_sql_columns, \
        type_configuration, \
        unsupported_parameters, \
        valid_horizon, \
        validation_end, \
        validation_start, \
        validation_start_index, \
        values, \
        yf
    import ast
    import gc
    import itertools
    import logging
    import math
    import sqlite3
    from pathlib import Path
    import numpy as np
    import pandas as pd
    import yfinance as yf
    from main_package import (
        benchmark_metrics,
        create_models_and_predictions,
        run_portfolio_backtest_from_predictions,
    )

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S"
    )
    logger = logging.getLogger(__name__)
    DATA_DIR = Path(str(data_root()) + "/")
    FINAL_RESULTS_DB = DATA_DIR / "Final_Test_Results.db"
    SELECTED_FEATURES_FILE = DATA_DIR / "Selected_Features.txt"
    FEATURE_DATABASE = DATA_DIR / "Features_Targets_Data.db"
    RESEARCH_END = pd.Timestamp(get_setting("RESEARCH_END", "2023-09-30"))
    HORIZON_VALIDATION_FRACTION = get_setting("HORIZON_VALIDATION_FRACTION", 0.25)
    STOCK_TYPE = get_setting("STOCK_TYPE", "High Liquidity 30")
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
        raise ValueError(f"Unknown STOCK_TYPE: {STOCK_TYPE}")
    stock_type_index = STOCK_TYPE_INDICES[STOCK_TYPE]

    ANALYSIS_MODE = get_setting("ANALYSIS_MODE", "DAILY").upper()
    if ANALYSIS_MODE not in {"DAILY", "INTRADAY", "COMBINED"}:
        raise ValueError("ANALYSIS_MODE must be DAILY, INTRADAY or COMBINED")
    RESULT_TABLES = {
        "DAILY": f"{STOCK_TYPE} Passed Test Results",
        "INTRADAY": f"Intraday {STOCK_TYPE} Passed Test Results",
    }
    ACTIVE_ANALYSIS_TYPES = (
        ["DAILY", "INTRADAY"] if ANALYSIS_MODE == "COMBINED" else [ANALYSIS_MODE]
    )
    setting_mode = get_setting("HORIZON_SETTING_MODE", "representative")
    if setting_mode not in {"representative", "cartesian"}:
        raise ValueError("HORIZON_SETTING_MODE must be representative or cartesian")

    def quote_sql_identifier(identifier):
        return '"' + str(identifier).replace('"', '""') + '"'

    quote_sql_identifier = callback("quote_sql_identifier", quote_sql_identifier)

    def dataframe_memory_mb(dataframe):
        return dataframe.memory_usage(index=True, deep=True).sum() / 1024**2

    dataframe_memory_mb = callback("dataframe_memory_mb", dataframe_memory_mb)

    logger.info("Loading passed results | mode=%s | stock type=%s", ANALYSIS_MODE, STOCK_TYPE)
    result_parts = []
    with sqlite3.connect(FINAL_RESULTS_DB) as connection:
        available_tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        for analysis_type in ACTIVE_ANALYSIS_TYPES:
            result_table = RESULT_TABLES[analysis_type]
            if result_table not in available_tables:
                raise ValueError(f"Missing {analysis_type} results table: {result_table}")
            result_part = pd.read_sql_query(
                f"\n            SELECT *\n            FROM {quote_sql_identifier(result_table)}\n            ",
                connection,
            )
            if result_part.empty:
                raise ValueError(f"Results table is empty: {result_table}")
            result_part["Analysis Type"] = analysis_type
            result_parts.append(result_part)
    test_results = pd.concat(result_parts, ignore_index=True, sort=False)
    del result_parts
    required_result_columns = {
        "Target",
        "Model",
        "Parameters",
        "Target Type",
        "Portfolio Target Type",
        "Horizon",
        "Quality Score",
        "Analysis Type",
    }
    missing_result_columns = required_result_columns.difference(test_results.columns)
    if missing_result_columns:
        raise ValueError(
            "Most Predictable Results table is missing: "
            + ", ".join(sorted(missing_result_columns))
        )
    logger.info("Loading selected features from %s", SELECTED_FEATURES_FILE)
    from equity_selector.feature_mapping import load_feature_mapping

    selected_features = load_feature_mapping(SELECTED_FEATURES_FILE, STOCK_TYPE)
    selected_models_df = test_results[
        test_results["Target"].astype(str).isin(selected_features.keys())
    ].copy()
    selected_models_df = selected_models_df[
        ~selected_models_df["Model"].astype(str).str.contains("Baseline", case=False, na=False)
    ].copy()
    selected_models_df["Target Type"] = (
        selected_models_df["Target Type"].astype(str).str.upper().str.strip()
    )
    selected_models_df["Horizon"] = pd.to_numeric(selected_models_df["Horizon"], errors="coerce")
    selected_models_df["Quality Score"] = pd.to_numeric(
        selected_models_df["Quality Score"], errors="coerce"
    ).clip(lower=0.0, upper=1.0)
    selected_models_df["Parameters"] = (
        selected_models_df["Parameters"]
        .where(selected_models_df["Parameters"].notna(), "{}")
        .astype(str)
    )
    selected_models_df = selected_models_df.dropna(
        subset=["Target", "Model", "Horizon", "Quality Score"]
    ).copy()
    DAILY_HORIZONS = {1, 5, 20, 60, 120, 252}
    INTRADAY_HORIZONS = {1, 5, 15, 60}
    valid_horizon = selected_models_df["Analysis Type"].eq("DAILY") & selected_models_df[
        "Horizon"
    ].isin(DAILY_HORIZONS) | selected_models_df["Analysis Type"].eq(
        "INTRADAY"
    ) & selected_models_df["Horizon"].isin(INTRADAY_HORIZONS)
    selected_models_df = selected_models_df[valid_horizon].copy()
    if selected_models_df.empty:
        raise ValueError(f"No {ANALYSIS_MODE.lower()} selected models remain.")
    sort_columns = ["Quality Score"]
    ascending = [False]
    if "Predictability Score" in selected_models_df.columns:
        selected_models_df["Predictability Score"] = pd.to_numeric(
            selected_models_df["Predictability Score"], errors="coerce"
        )
        sort_columns.append("Predictability Score")
        ascending.append(False)
    selected_models_df = (
        selected_models_df.sort_values(sort_columns, ascending=ascending)
        .drop_duplicates(subset=["Analysis Type", "Target"], keep="first")
        .reset_index(drop=True)
    )
    selected_models_df = selected_models_df[
        [
            "Analysis Type",
            "Target",
            "Model",
            "Parameters",
            "Target Type",
            "Portfolio Target Type",
            "Horizon",
            "Quality Score",
        ]
    ].copy()
    selected_targets = selected_models_df["Target"].astype(str).tolist()
    missing_feature_targets = [
        target for target in selected_targets if target not in selected_features
    ]
    if missing_feature_targets:
        raise ValueError(
            "Selected_Features.txt does not contain feature definitions for:\n"
            + "\n".join(missing_feature_targets)
        )
    target_features = {target: list(selected_features[target]) for target in selected_targets}
    required_features = list(
        dict.fromkeys(
            (feature for target in selected_targets for feature in target_features[target])
        )
    )
    logger.info(
        "Selected models ready | targets=%d | target types=%d | required features=%d",
        len(selected_models_df),
        selected_models_df["Portfolio Target Type"].nunique(),
        len(required_features),
    )
    logger.info("Inspecting feature database | %s | table=%s", FEATURE_DATABASE, STOCK_TYPE)
    with sqlite3.connect(FEATURE_DATABASE) as metadata_connection:
        table_info = metadata_connection.execute(
            f"PRAGMA table_info({quote_sql_identifier(STOCK_TYPE)})"
        ).fetchall()
    SOURCE_TABLE_COLUMNS = [row[1] for row in table_info]
    SOURCE_TABLE_COLUMN_SET = set(SOURCE_TABLE_COLUMNS)
    if not SOURCE_TABLE_COLUMNS:
        raise ValueError(f"Source table does not exist or has no columns: {STOCK_TYPE}")
    required_base_columns = {"Date", "Ticker", "Close"}
    missing_base_columns = required_base_columns.difference(SOURCE_TABLE_COLUMN_SET)
    if missing_base_columns:
        raise KeyError(
            f"Base columns missing from {STOCK_TYPE}: " + ", ".join(sorted(missing_base_columns))
        )
    logger.info("Memory-safe SQL mode | full %s table will never be loaded", STOCK_TYPE)
    research_end_sql = RESEARCH_END.strftime("%Y-%m-%d")
    with sqlite3.connect(FEATURE_DATABASE) as connection:
        research_dates_df = pd.read_sql_query(
            f"\n        SELECT DISTINCT {quote_sql_identifier('Date')} AS {quote_sql_identifier('Date')}\n        FROM {quote_sql_identifier(STOCK_TYPE)}\n        WHERE date({quote_sql_identifier('Date')}) <= date(?)\n        ORDER BY {quote_sql_identifier('Date')}\n        ",
            connection,
            params=[research_end_sql],
        )
    research_dates = (
        pd.to_datetime(research_dates_df["Date"])
        .dropna()
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )
    del research_dates_df
    if research_dates.empty:
        raise ValueError("No research dates are available before RESEARCH_END.")
    if get_setting("RESEARCH_START") is not None:
        research_dates = research_dates.loc[
            research_dates >= pd.Timestamp(get_setting("RESEARCH_START"))
        ].reset_index(drop=True)
    validation_start_index = (
        int(research_dates.searchsorted(pd.Timestamp(get_setting("HORIZON_VALIDATION_START"))))
        if get_setting("HORIZON_VALIDATION_START") is not None
        else int(len(research_dates) * (1.0 - HORIZON_VALIDATION_FRACTION))
    )
    if validation_start_index <= 0 or validation_start_index >= len(research_dates):
        raise ValueError("Invalid Horizon validation split.")
    max_selected_horizon = int(selected_models_df["Horizon"].max())
    fit_end_index = validation_start_index - max_selected_horizon
    if fit_end_index <= 0:
        raise ValueError(
            f"Not enough research dates for a {HORIZON_VALIDATION_FRACTION:.0%} Horizon validation period plus a {max_selected_horizon}-trading-day purge."
        )
    fit_start = research_dates.iloc[0]
    fit_end = research_dates.iloc[fit_end_index - 1]
    validation_start = research_dates.iloc[validation_start_index]
    validation_end = research_dates.iloc[-1]
    logger.info("Research fit period | %s to %s", fit_start.date(), fit_end.date())
    logger.info("Purged %d trading dates before Horizon validation", max_selected_horizon)
    logger.info(
        "Horizon validation period | %s to %s", validation_start.date(), validation_end.date()
    )
    RESEARCH_SPLITS = {}
    for analysis_type in ACTIVE_ANALYSIS_TYPES:
        analysis_models = selected_models_df[selected_models_df["Analysis Type"].eq(analysis_type)]
        analysis_max_horizon = int(analysis_models["Horizon"].max())
        analysis_fit_end_index = validation_start_index - analysis_max_horizon
        if analysis_fit_end_index <= 0:
            raise ValueError(
                f"Not enough observations for the {analysis_type} validation period and a {analysis_max_horizon}-period purge."
            )
        RESEARCH_SPLITS[analysis_type] = {
            "fit_start": research_dates.iloc[0],
            "fit_end": research_dates.iloc[analysis_fit_end_index - 1],
            "validation_start": research_dates.iloc[validation_start_index],
            "validation_end": research_dates.iloc[-1],
            "max_horizon": analysis_max_horizon,
        }
        logger.info(
            "%s split | fit=%s to %s | purge=%d | validation=%s to %s",
            analysis_type,
            RESEARCH_SPLITS[analysis_type]["fit_start"],
            RESEARCH_SPLITS[analysis_type]["fit_end"],
            analysis_max_horizon,
            RESEARCH_SPLITS[analysis_type]["validation_start"],
            RESEARCH_SPLITS[analysis_type]["validation_end"],
        )

    def target_sql_columns(target, features):
        columns = ["Date", "Ticker", "Close"]
        if "Return" in SOURCE_TABLE_COLUMN_SET:
            columns.append("Return")
        columns.append(target)
        columns.extend(features)
        columns = list(dict.fromkeys(columns))
        missing_columns = [column for column in columns if column not in SOURCE_TABLE_COLUMN_SET]
        if missing_columns:
            raise KeyError(f"{target} | Columns missing from {STOCK_TYPE}: {missing_columns}")
        return columns

    target_sql_columns = callback("target_sql_columns", target_sql_columns)

    def load_target_period(target, features, start_date, end_date, split):
        columns = target_sql_columns(target, features)
        sql_columns = ", ".join((quote_sql_identifier(column) for column in columns))
        query = f"SELECT {sql_columns} FROM {quote_sql_identifier(STOCK_TYPE)} WHERE date({quote_sql_identifier('Date')}) >= date(?) AND date({quote_sql_identifier('Date')}) <= date(?)"
        start_value = pd.Timestamp(start_date).strftime("%Y-%m-%d")
        end_value = pd.Timestamp(end_date).strftime("%Y-%m-%d")
        logger.info(
            "%s | %s | loading %d/%d source columns",
            target,
            split,
            len(columns),
            len(SOURCE_TABLE_COLUMNS),
        )
        with sqlite3.connect(FEATURE_DATABASE) as connection:
            dataframe = pd.read_sql_query(query, connection, params=[start_value, end_value])
        if dataframe.empty:
            raise ValueError(f"{target} | No rows loaded for {split}.")
        dataframe["Date"] = pd.to_datetime(dataframe["Date"])
        dataframe = dataframe.sort_values(["Date", "Ticker"]).reset_index(drop=True)
        if "Return" not in dataframe.columns:
            dataframe["Return"] = dataframe.groupby("Ticker", sort=False)["Close"].pct_change()
        dataframe["Split"] = split
        logger.info(
            "%s | %s | loaded %d rows x %d columns | %.1f MB",
            target,
            split,
            len(dataframe),
            len(dataframe.columns),
            dataframe_memory_mb(dataframe),
        )
        return dataframe

    load_target_period = callback("load_target_period", load_target_period)

    logger.info(
        "Downloading S&P 500 benchmark only | %s to %s",
        validation_start.date(),
        validation_end.date(),
    )
    benchmark_download = yf.download(
        "^GSPC",
        start=validation_start.strftime("%Y-%m-%d"),
        end=(validation_end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        auto_adjust=True,
        progress=False,
        multi_level_index=False,
    )
    if benchmark_download.empty:
        raise ValueError("No S&P 500 benchmark data returned.")
    benchmark_close = benchmark_download["Close"]
    if isinstance(benchmark_close, pd.DataFrame):
        benchmark_close = benchmark_close.iloc[:, 0]
    market_df = pd.DataFrame(
        {
            "Date": pd.to_datetime(benchmark_close.index),
            "Close": pd.to_numeric(benchmark_close.to_numpy(), errors="coerce"),
        }
    )
    market_df["Return"] = market_df["Close"].pct_change()
    market_df = market_df.dropna(subset=["Close"]).sort_values("Date").reset_index(drop=True)
    market_results = benchmark_metrics(market_df)
    logger.info(
        "Benchmark ready | return=%.4f | sharpe=%.4f | max_dd=%.4f | avg_dd=%.4f",
        market_results["Return"],
        market_results["Sharpe Ratio"],
        market_results["Max Drawdown"],
        market_results["Average Drawdown"],
    )
    HORIZON_SCORE_VALUES = get_setting(
        "HORIZON_SCORE_VALUES",
        {
            "ALPHA": {
                "1m": [0.0, 0.05, 0.1, 0.15, 0.2],
                "5m": [0.0, 0.05, 0.1, 0.2, 0.25],
                "15m": [0.0, 0.1, 0.15, 0.2, 0.25, 0.3],
                "60m": [0.0, 0.15, 0.2, 0.3, 0.35, 0.4],
                "1d": [0.0, 0.2, 0.3, 0.4, 0.45, 0.55],
                "5d": [0.0, 0.45, 0.5, 0.6, 0.7, 0.75],
                "20d": [0.0, 0.65, 0.7, 0.8, 0.9, 0.95],
                "60d": [0.0, 0.6, 0.7, 0.8, 0.9, 1.0],
                "120d": [0.0, 0.3, 0.45, 0.6, 0.7, 0.85],
                "252d": [0.0, 0.15, 0.3, 0.45, 0.6],
            },
            "RELATIVE_ALPHA": {
                "1m": [0.0, 0.05, 0.1, 0.15],
                "5m": [0.0, 0.05, 0.1, 0.15, 0.2],
                "15m": [0.0, 0.05, 0.1, 0.2, 0.25],
                "60m": [0.0, 0.1, 0.15, 0.2, 0.3, 0.35],
                "1d": [0.0, 0.2, 0.3, 0.35, 0.4, 0.5],
                "5d": [0.0, 0.45, 0.5, 0.6, 0.7, 0.75],
                "20d": [0.0, 0.65, 0.75, 0.8, 0.9, 1.0],
                "60d": [0.0, 0.6, 0.7, 0.8, 0.9, 1.0],
                "120d": [0.0, 0.3, 0.45, 0.6, 0.7, 0.85],
                "252d": [0.0, 0.15, 0.3, 0.5, 0.65],
            },
            "RISK_ADJUSTED_ALPHA": {
                "1m": [0.0, 0.05, 0.1, 0.15, 0.2],
                "5m": [0.0, 0.05, 0.1, 0.2, 0.25],
                "15m": [0.0, 0.05, 0.1, 0.2, 0.25, 0.3],
                "60m": [0.0, 0.15, 0.2, 0.3, 0.4, 0.45],
                "1d": [0.0, 0.3, 0.4, 0.45, 0.5, 0.6],
                "5d": [0.0, 0.55, 0.6, 0.7, 0.8, 0.85],
                "20d": [0.0, 0.7, 0.8, 0.85, 0.9, 1.0],
                "60d": [0.0, 0.6, 0.7, 0.8, 0.9, 1.0],
                "120d": [0.0, 0.3, 0.45, 0.6, 0.7, 0.85],
                "252d": [0.0, 0.15, 0.3, 0.45, 0.6],
            },
            "CROSS_SECTION_ALPHA": {
                "1m": [0.0, 0.05, 0.1, 0.15],
                "5m": [0.0, 0.05, 0.1, 0.15, 0.2],
                "15m": [0.0, 0.05, 0.1, 0.2, 0.25],
                "60m": [0.0, 0.1, 0.15, 0.2, 0.3, 0.35],
                "1d": [0.0, 0.25, 0.3, 0.4, 0.5, 0.55],
                "5d": [0.0, 0.5, 0.6, 0.65, 0.7, 0.8],
                "20d": [0.0, 0.7, 0.8, 0.85, 0.9, 1.0],
                "60d": [0.0, 0.6, 0.7, 0.8, 0.9, 1.0],
                "120d": [0.0, 0.25, 0.4, 0.5, 0.65, 0.8],
                "252d": [0.0, 0.15, 0.3, 0.4, 0.55],
            },
            "CROSS_SECTION_DOWNSIDE": {
                "1m": [0.0, 0.05, 0.1, 0.15, 0.2],
                "5m": [0.0, 0.05, 0.1, 0.2, 0.25],
                "15m": [0.0, 0.1, 0.15, 0.2, 0.3, 0.35],
                "60m": [0.0, 0.2, 0.3, 0.35, 0.4, 0.5],
                "1d": [0.0, 0.4, 0.5, 0.55, 0.6, 0.7],
                "5d": [0.0, 0.65, 0.7, 0.8, 0.9, 0.95],
                "20d": [0.0, 0.85, 0.9, 0.95, 1.0],
                "60d": [0.0, 0.8, 0.85, 0.9, 0.95, 1.0],
                "120d": [0.0, 0.55, 0.6, 0.7, 0.8, 0.85],
                "252d": [0.0, 0.35, 0.4, 0.5, 0.55, 0.6],
            },
            "DIRECTION": {
                "1m": [0.0, 0.1, 0.2, 0.3, 0.4],
                "5m": [0.0, 0.1, 0.2, 0.3, 0.35, 0.45],
                "15m": [0.0, 0.2, 0.3, 0.4, 0.45, 0.55],
                "60m": [0.0, 0.4, 0.5, 0.55, 0.6, 0.7],
                "1d": [0.0, 0.6, 0.7, 0.75, 0.8, 0.9],
                "5d": [0.0, 0.85, 0.9, 0.95, 1.0],
                "20d": [0.0, 0.75, 0.8, 0.9, 0.95, 1.0],
                "60d": [0.0, 0.45, 0.55, 0.6, 0.7, 0.8],
                "120d": [0.0, 0.2, 0.3, 0.4, 0.45, 0.55],
                "252d": [0.0, 0.1, 0.2, 0.25, 0.35],
            },
            "DIRECTION_MULTICLASS": {
                "1m": [0.0, 0.1, 0.2, 0.25, 0.35],
                "5m": [0.0, 0.1, 0.2, 0.3, 0.35, 0.45],
                "15m": [0.0, 0.2, 0.3, 0.4, 0.45, 0.55],
                "60m": [0.0, 0.35, 0.45, 0.5, 0.6, 0.7],
                "1d": [0.0, 0.55, 0.65, 0.7, 0.8, 0.9],
                "5d": [0.0, 0.8, 0.85, 0.9, 0.95, 1.0],
                "20d": [0.0, 0.85, 0.9, 0.95, 1.0],
                "60d": [0.0, 0.65, 0.7, 0.8, 0.9, 0.95],
                "120d": [0.0, 0.25, 0.35, 0.4, 0.5, 0.6],
                "252d": [0.0, 0.1, 0.2, 0.25, 0.35],
            },
            "ALPHA_BINARY": {
                "1m": [0.0, 0.1, 0.25, 0.4, 0.5],
                "5m": [0.0, 0.2, 0.3, 0.4, 0.5, 0.6],
                "15m": [0.0, 0.35, 0.45, 0.5, 0.6, 0.7],
                "60m": [0.0, 0.5, 0.6, 0.7, 0.75, 0.85],
                "1d": [0.0, 0.65, 0.75, 0.8, 0.9, 1.0],
                "5d": [0.0, 0.85, 0.9, 0.95, 1.0],
                "20d": [0.0, 0.7, 0.8, 0.85, 0.9, 1.0],
                "60d": [0.0, 0.4, 0.5, 0.6, 0.65, 0.75],
                "120d": [0.0, 0.15, 0.25, 0.3, 0.4, 0.5],
                "252d": [0.0, 0.1, 0.2, 0.25, 0.35],
            },
            "BARRIER_ALPHA": {
                "1m": [0.0, 0.15, 0.3, 0.4, 0.55],
                "5m": [0.0, 0.25, 0.35, 0.45, 0.55, 0.65],
                "15m": [0.0, 0.4, 0.5, 0.6, 0.65, 0.75],
                "60m": [0.0, 0.5, 0.6, 0.7, 0.75, 0.85],
                "1d": [0.0, 0.65, 0.75, 0.8, 0.9, 1.0],
                "5d": [0.0, 0.85, 0.9, 0.95, 1.0],
                "20d": [0.0, 0.75, 0.8, 0.9, 0.95, 1.0],
                "60d": [0.0, 0.45, 0.55, 0.6, 0.7, 0.8],
                "120d": [0.0, 0.2, 0.3, 0.4, 0.45, 0.55],
                "252d": [0.0, 0.1, 0.2, 0.25, 0.35],
            },
            "VOLATILITY": {
                "1m": [0.0, 0.1, 0.2, 0.35, 0.45],
                "5m": [0.0, 0.2, 0.3, 0.4, 0.45, 0.55],
                "15m": [0.0, 0.3, 0.4, 0.5, 0.55, 0.65],
                "60m": [0.0, 0.45, 0.55, 0.6, 0.7, 0.8],
                "1d": [0.0, 0.6, 0.7, 0.8, 0.85, 0.95],
                "5d": [0.0, 0.85, 0.9, 0.95, 1.0],
                "20d": [0.0, 0.8, 0.85, 0.9, 0.95, 1.0],
                "60d": [0.0, 0.6, 0.7, 0.8, 0.85, 0.95],
                "120d": [0.0, 0.4, 0.5, 0.6, 0.65, 0.75],
                "252d": [0.0, 0.25, 0.3, 0.4, 0.5, 0.55],
            },
            "ABSOLUTE_MOVE": {
                "1m": [0.0, 0.2, 0.3, 0.4, 0.45, 0.55],
                "5m": [0.0, 0.3, 0.4, 0.5, 0.55, 0.65],
                "15m": [0.0, 0.4, 0.5, 0.6, 0.65, 0.75],
                "60m": [0.0, 0.55, 0.6, 0.7, 0.8, 0.85],
                "1d": [0.0, 0.65, 0.7, 0.8, 0.9, 0.95],
                "5d": [0.0, 0.85, 0.9, 0.95, 1.0],
                "20d": [0.0, 0.85, 0.9, 0.95, 1.0],
                "60d": [0.0, 0.65, 0.7, 0.8, 0.9, 0.95],
                "120d": [0.0, 0.35, 0.45, 0.5, 0.6, 0.7],
                "252d": [0.0, 0.2, 0.25, 0.3, 0.4, 0.45],
            },
            "UPSIDE_VOLATILITY": {
                "1m": [0.0, 0.1, 0.2, 0.3, 0.35, 0.45],
                "5m": [0.0, 0.2, 0.3, 0.4, 0.45, 0.55],
                "15m": [0.0, 0.3, 0.4, 0.5, 0.55, 0.65],
                "60m": [0.0, 0.45, 0.55, 0.6, 0.7, 0.8],
                "1d": [0.0, 0.6, 0.7, 0.8, 0.85, 0.95],
                "5d": [0.0, 0.8, 0.85, 0.9, 0.95, 1.0],
                "20d": [0.0, 0.85, 0.9, 0.95, 1.0],
                "60d": [0.0, 0.65, 0.7, 0.8, 0.9, 0.95],
                "120d": [0.0, 0.35, 0.45, 0.5, 0.6, 0.7],
                "252d": [0.0, 0.25, 0.3, 0.4, 0.45, 0.5],
            },
            "DOWNSIDE_VOLATILITY": {
                "1m": [0.0, 0.15, 0.25, 0.3, 0.4, 0.5],
                "5m": [0.0, 0.25, 0.35, 0.4, 0.5, 0.6],
                "15m": [0.0, 0.35, 0.45, 0.5, 0.6, 0.7],
                "60m": [0.0, 0.5, 0.6, 0.7, 0.75, 0.85],
                "1d": [0.0, 0.65, 0.75, 0.8, 0.9, 1.0],
                "5d": [0.0, 0.85, 0.9, 0.95, 1.0],
                "20d": [0.0, 0.85, 0.9, 0.95, 1.0],
                "60d": [0.0, 0.7, 0.8, 0.85, 0.9, 1.0],
                "120d": [0.0, 0.4, 0.5, 0.6, 0.65, 0.75],
                "252d": [0.0, 0.2, 0.3, 0.4, 0.45, 0.55],
            },
            "VOLATILITY_ASYMMETRY": {
                "1m": [0.0, 0.1, 0.2, 0.25, 0.35],
                "5m": [0.0, 0.1, 0.2, 0.3, 0.35, 0.45],
                "15m": [0.0, 0.2, 0.3, 0.4, 0.45, 0.55],
                "60m": [0.0, 0.35, 0.45, 0.5, 0.6, 0.7],
                "1d": [0.0, 0.5, 0.6, 0.7, 0.75, 0.85],
                "5d": [0.0, 0.7, 0.8, 0.85, 0.9, 1.0],
                "20d": [0.0, 0.85, 0.9, 0.95, 1.0],
                "60d": [0.0, 0.75, 0.8, 0.9, 0.95, 1.0],
                "120d": [0.0, 0.4, 0.5, 0.6, 0.65, 0.75],
                "252d": [0.0, 0.25, 0.3, 0.4, 0.45, 0.5],
            },
            "VOLATILITY_EVENT": {
                "1m": [0.0, 0.45, 0.55, 0.6, 0.7, 0.8],
                "5m": [0.0, 0.55, 0.65, 0.7, 0.8, 0.9],
                "15m": [0.0, 0.7, 0.8, 0.85, 0.9, 1.0],
                "60m": [0.0, 0.8, 0.85, 0.9, 0.95, 1.0],
                "1d": [0.0, 0.8, 0.85, 0.9, 0.95, 1.0],
                "5d": [0.0, 0.6, 0.7, 0.8, 0.85, 0.95],
                "20d": [0.0, 0.35, 0.45, 0.5, 0.6, 0.7],
                "60d": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                "120d": [0.0, 0.1, 0.2, 0.25, 0.35],
                "252d": [0.0, 0.05, 0.1, 0.15, 0.2],
            },
            "DOWNSIDE": {
                "1m": [0.0, 0.2, 0.3, 0.4, 0.45, 0.55],
                "5m": [0.0, 0.3, 0.4, 0.5, 0.55, 0.65],
                "15m": [0.0, 0.4, 0.5, 0.6, 0.65, 0.75],
                "60m": [0.0, 0.55, 0.65, 0.7, 0.8, 0.9],
                "1d": [0.0, 0.7, 0.8, 0.85, 0.9, 1.0],
                "5d": [0.0, 0.85, 0.9, 0.95, 1.0],
                "20d": [0.0, 0.75, 0.8, 0.9, 0.95, 1.0],
                "60d": [0.0, 0.5, 0.6, 0.7, 0.75, 0.85],
                "120d": [0.0, 0.3, 0.4, 0.5, 0.55, 0.65],
                "252d": [0.0, 0.15, 0.2, 0.3, 0.4, 0.45],
            },
            "TAIL_RISK": {
                "1m": [0.0, 0.35, 0.45, 0.5, 0.6, 0.7],
                "5m": [0.0, 0.45, 0.55, 0.6, 0.7, 0.8],
                "15m": [0.0, 0.55, 0.65, 0.7, 0.8, 0.9],
                "60m": [0.0, 0.65, 0.75, 0.8, 0.9, 1.0],
                "1d": [0.0, 0.85, 0.9, 0.95, 1.0],
                "5d": [0.0, 0.8, 0.85, 0.9, 0.95, 1.0],
                "20d": [0.0, 0.6, 0.7, 0.75, 0.8, 0.9],
                "60d": [0.0, 0.3, 0.4, 0.5, 0.55, 0.65],
                "120d": [0.0, 0.1, 0.2, 0.3, 0.35, 0.45],
                "252d": [0.0, 0.1, 0.15, 0.2, 0.3],
            },
            "TAIL_EVENT": {
                "1m": [0.0, 0.8, 0.85, 0.9, 0.95, 1.0],
                "5m": [0.0, 0.85, 0.9, 0.95, 1.0],
                "15m": [0.0, 0.8, 0.85, 0.9, 0.95, 1.0],
                "60m": [0.0, 0.65, 0.75, 0.8, 0.9, 1.0],
                "1d": [0.0, 0.55, 0.65, 0.7, 0.8, 0.9],
                "5d": [0.0, 0.3, 0.4, 0.5, 0.55, 0.65],
                "20d": [0.0, 0.1, 0.2, 0.3, 0.4],
                "60d": [0.0, 0.05, 0.1, 0.2, 0.25],
                "120d": [0.0, 0.05, 0.1, 0.15],
                "252d": [0.0, 0.05, 0.1],
            },
            "UPSIDE_EVENT": {
                "1m": [0.0, 0.8, 0.85, 0.9, 0.95, 1.0],
                "5m": [0.0, 0.85, 0.9, 0.95, 1.0],
                "15m": [0.0, 0.8, 0.85, 0.9, 0.95, 1.0],
                "60m": [0.0, 0.65, 0.75, 0.8, 0.9, 1.0],
                "1d": [0.0, 0.5, 0.6, 0.7, 0.75, 0.85],
                "5d": [0.0, 0.25, 0.35, 0.4, 0.5, 0.6],
                "20d": [0.0, 0.1, 0.2, 0.25, 0.35],
                "60d": [0.0, 0.05, 0.1, 0.15, 0.2],
                "120d": [0.0, 0.05, 0.1, 0.15],
                "252d": [0.0, 0.05, 0.1],
            },
            "UPSIDE_EXCURSION": {
                "1m": [0.0, 0.5, 0.6, 0.7, 0.75, 0.85],
                "5m": [0.0, 0.6, 0.7, 0.8, 0.85, 0.95],
                "15m": [0.0, 0.7, 0.8, 0.85, 0.9, 1.0],
                "60m": [0.0, 0.75, 0.8, 0.9, 0.95, 1.0],
                "1d": [0.0, 0.75, 0.8, 0.9, 0.95, 1.0],
                "5d": [0.0, 0.85, 0.9, 0.95, 1.0],
                "20d": [0.0, 0.7, 0.8, 0.85, 0.9, 1.0],
                "60d": [0.0, 0.45, 0.55, 0.6, 0.7, 0.8],
                "120d": [0.0, 0.15, 0.25, 0.3, 0.4, 0.5],
                "252d": [0.0, 0.1, 0.15, 0.2, 0.3],
            },
            "DOWNSIDE_EXCURSION": {
                "1m": [0.0, 0.55, 0.65, 0.7, 0.8, 0.9],
                "5m": [0.0, 0.65, 0.75, 0.8, 0.9, 1.0],
                "15m": [0.0, 0.75, 0.8, 0.9, 0.95, 1.0],
                "60m": [0.0, 0.8, 0.85, 0.9, 0.95, 1.0],
                "1d": [0.0, 0.8, 0.85, 0.9, 0.95, 1.0],
                "5d": [0.0, 0.85, 0.9, 0.95, 1.0],
                "20d": [0.0, 0.75, 0.8, 0.9, 0.95, 1.0],
                "60d": [0.0, 0.5, 0.6, 0.7, 0.75, 0.85],
                "120d": [0.0, 0.2, 0.3, 0.4, 0.45, 0.55],
                "252d": [0.0, 0.1, 0.2, 0.25, 0.35],
            },
            "TIME_TO_UPSIDE_EXCURSION": {
                "1m": [0.0, 0.1, 0.2, 0.3, 0.4],
                "5m": [0.0, 0.15, 0.25, 0.3, 0.4, 0.5],
                "15m": [0.0, 0.3, 0.4, 0.5, 0.55, 0.65],
                "60m": [0.0, 0.5, 0.6, 0.65, 0.7, 0.8],
                "1d": [0.0, 0.5, 0.6, 0.65, 0.7, 0.8],
                "5d": [0.0, 0.65, 0.7, 0.8, 0.9, 0.95],
                "20d": [0.0, 0.8, 0.85, 0.9, 0.95, 1.0],
                "60d": [0.0, 0.7, 0.8, 0.85, 0.9, 1.0],
                "120d": [0.0, 0.3, 0.4, 0.5, 0.55, 0.65],
                "252d": [0.0, 0.1, 0.2, 0.25, 0.35],
            },
            "TIME_TO_DOWNSIDE_EXCURSION": {
                "1m": [0.0, 0.1, 0.2, 0.35, 0.45],
                "5m": [0.0, 0.2, 0.3, 0.4, 0.45, 0.55],
                "15m": [0.0, 0.35, 0.45, 0.5, 0.6, 0.7],
                "60m": [0.0, 0.55, 0.6, 0.7, 0.8, 0.85],
                "1d": [0.0, 0.55, 0.6, 0.7, 0.8, 0.85],
                "5d": [0.0, 0.7, 0.8, 0.85, 0.9, 1.0],
                "20d": [0.0, 0.85, 0.9, 0.95, 1.0],
                "60d": [0.0, 0.75, 0.8, 0.9, 0.95, 1.0],
                "120d": [0.0, 0.35, 0.45, 0.5, 0.6, 0.7],
                "252d": [0.0, 0.1, 0.2, 0.25, 0.3, 0.4],
            },
            "RECOVERY": {
                "1m": [0.0, 0.8, 0.85, 0.9, 0.95, 1.0],
                "5m": [0.0, 0.85, 0.9, 0.95, 1.0],
                "15m": [0.0, 0.8, 0.85, 0.9, 0.95, 1.0],
                "60m": [0.0, 0.65, 0.75, 0.8, 0.9, 1.0],
                "1d": [0.0, 0.45, 0.55, 0.6, 0.7, 0.8],
                "5d": [0.0, 0.15, 0.25, 0.3, 0.4, 0.5],
                "20d": [0.0, 0.1, 0.15, 0.2, 0.3],
                "60d": [0.0, 0.05, 0.1, 0.15],
                "120d": [0.0, 0.05, 0.1],
                "252d": [0.0, 0.05],
            },
            "REVERSAL": {
                "1m": [0.0, 0.8, 0.85, 0.9, 0.95, 1.0],
                "5m": [0.0, 0.85, 0.9, 0.95, 1.0],
                "15m": [0.0, 0.8, 0.85, 0.9, 0.95, 1.0],
                "60m": [0.0, 0.65, 0.75, 0.8, 0.9, 1.0],
                "1d": [0.0, 0.45, 0.55, 0.6, 0.7, 0.8],
                "5d": [0.0, 0.15, 0.25, 0.3, 0.4, 0.5],
                "20d": [0.0, 0.1, 0.15, 0.2, 0.3],
                "60d": [0.0, 0.05, 0.1, 0.15],
                "120d": [0.0, 0.05, 0.1],
                "252d": [0.0, 0.05],
            },
            "REGIME": {
                "1m": [0.0, 0.05, 0.1, 0.2, 0.25],
                "5m": [0.0, 0.1, 0.15, 0.2, 0.3],
                "15m": [0.0, 0.1, 0.2, 0.25, 0.35],
                "60m": [0.0, 0.1, 0.2, 0.3, 0.35, 0.45],
                "1d": [0.0, 0.3, 0.4, 0.5, 0.55, 0.65],
                "5d": [0.0, 0.55, 0.65, 0.7, 0.8, 0.9],
                "20d": [0.0, 0.85, 0.9, 0.95, 1.0],
                "60d": [0.0, 0.8, 0.85, 0.9, 0.95, 1.0],
                "120d": [0.0, 0.1, 0.3, 0.45, 0.6, 0.8],
                "252d": [0.0, 0.15, 0.3, 0.5, 0.65],
            },
            "CORRELATION": {
                "1m": [0.0, 0.05, 0.1, 0.2, 0.25],
                "5m": [0.0, 0.1, 0.15, 0.2, 0.3],
                "15m": [0.0, 0.05, 0.15, 0.2, 0.3, 0.4],
                "60m": [0.0, 0.15, 0.25, 0.3, 0.4, 0.5],
                "1d": [0.0, 0.35, 0.45, 0.5, 0.6, 0.7],
                "5d": [0.0, 0.6, 0.7, 0.8, 0.85, 0.95],
                "20d": [0.0, 0.85, 0.9, 0.95, 1.0],
                "60d": [0.0, 0.8, 0.85, 0.9, 0.95, 1.0],
                "120d": [0.0, 0.1, 0.3, 0.45, 0.6, 0.8],
                "252d": [0.0, 0.15, 0.3, 0.5, 0.65],
            },
            "COVARIANCE": {
                "1m": [0.0, 0.05, 0.1, 0.2, 0.25],
                "5m": [0.0, 0.1, 0.15, 0.2, 0.3],
                "15m": [0.0, 0.05, 0.15, 0.2, 0.3, 0.4],
                "60m": [0.0, 0.15, 0.25, 0.3, 0.4, 0.5],
                "1d": [0.0, 0.35, 0.45, 0.5, 0.6, 0.7],
                "5d": [0.0, 0.6, 0.7, 0.8, 0.85, 0.95],
                "20d": [0.0, 0.85, 0.9, 0.95, 1.0],
                "60d": [0.0, 0.8, 0.85, 0.9, 0.95, 1.0],
                "120d": [0.0, 0.1, 0.3, 0.45, 0.6, 0.8],
                "252d": [0.0, 0.15, 0.3, 0.5, 0.65],
            },
            "LIQUIDITY": {
                "1m": [0.0, 0.6, 0.7, 0.8, 0.85, 0.95],
                "5m": [0.0, 0.75, 0.8, 0.9, 0.95, 1.0],
                "15m": [0.0, 0.85, 0.9, 0.95, 1.0],
                "60m": [0.0, 0.8, 0.85, 0.9, 0.95, 1.0],
                "1d": [0.0, 0.5, 0.6, 0.7, 0.75, 0.85],
                "5d": [0.0, 0.2, 0.3, 0.4, 0.45, 0.55],
                "20d": [0.0, 0.1, 0.2, 0.25, 0.35],
                "60d": [0.0, 0.05, 0.1, 0.15, 0.2],
                "120d": [0.0, 0.05, 0.1],
                "252d": [0.0, 0.05],
            },
            "EXECUTION": {
                "1m": [0.0, 0.85, 0.9, 0.95, 1.0],
                "5m": [0.0, 0.85, 0.9, 0.95, 1.0],
                "15m": [0.0, 0.8, 0.85, 0.9, 0.95, 1.0],
                "60m": [0.0, 0.6, 0.7, 0.8, 0.85, 0.95],
                "1d": [0.0, 0.2, 0.3, 0.4, 0.45, 0.55],
                "5d": [0.0, 0.1, 0.15, 0.2, 0.3],
                "20d": [0.0, 0.05, 0.1, 0.15],
                "60d": [0.0, 0.05, 0.1],
                "120d": [0.0, 0.05],
                "252d": [0.0],
            },
            "MARKET_IMPACT": {
                "1m": [0.0, 0.85, 0.9, 0.95, 1.0],
                "5m": [0.0, 0.85, 0.9, 0.95, 1.0],
                "15m": [0.0, 0.8, 0.85, 0.9, 0.95, 1.0],
                "60m": [0.0, 0.6, 0.7, 0.8, 0.85, 0.95],
                "1d": [0.0, 0.2, 0.3, 0.4, 0.45, 0.55],
                "5d": [0.0, 0.1, 0.15, 0.2, 0.3],
                "20d": [0.0, 0.05, 0.1, 0.15],
                "60d": [0.0, 0.05, 0.1],
                "120d": [0.0, 0.05],
                "252d": [0.0],
            },
        },
    )
    HORIZON_INDEX = 2

    def horizon_key(row):
        suffix = "m" if str(row.get("Analysis Type", "DAILY")).upper() == "INTRADAY" else "d"
        return f"{int(row['Horizon'])}{suffix}"

    horizon_key = callback("horizon_key", horizon_key)

    selected_models_df["Horizon Key"] = selected_models_df.apply(horizon_key, axis=1)
    unsupported_parameters = []
    for _, row in selected_models_df.iterrows():
        target_type = str(row["Portfolio Target Type"]).upper().strip()
        horizon = row["Horizon Key"]
        if (
            target_type not in HORIZON_SCORE_VALUES
            or horizon not in HORIZON_SCORE_VALUES[target_type]
        ):
            unsupported_parameters.append((str(row["Target"]), target_type, horizon))
    if unsupported_parameters:
        raise ValueError(
            "Selected models contain unsupported Target Type / Horizon pairs:\n"
            + "\n".join(
                (
                    f"{target}: {target_type} {horizon}"
                    for target, target_type, horizon in unsupported_parameters
                )
            )
        )
    active_parameters = (
        selected_models_df[["Portfolio Target Type", "Horizon Key"]]
        .drop_duplicates()
        .sort_values(["Portfolio Target Type", "Horizon Key"])
        .reset_index(drop=True)
    )
    ACTIVE_HORIZON_SCORE_VALUES = {}
    for _, row in active_parameters.iterrows():
        target_type = row["Portfolio Target Type"]
        horizon = row["Horizon Key"]
        ACTIVE_HORIZON_SCORE_VALUES.setdefault(target_type, {})[horizon] = HORIZON_SCORE_VALUES[
            target_type
        ][horizon]
    logger.info("Active Horizon Score parameters=%d", len(active_parameters))

    def reference_horizon_score(values):
        nonzero_values = [value for value in values if value != 0]
        reference_values = nonzero_values if nonzero_values else values
        index = min(HORIZON_INDEX, len(reference_values) - 1)
        return float(reference_values[index])

    reference_horizon_score = callback("reference_horizon_score", reference_horizon_score)

    def get_horizon_score(row):
        target_type = str(row["Portfolio Target Type"]).upper().strip()
        if "Horizon Key" in row.index:
            horizon = row["Horizon Key"]
        else:
            horizon = horizon_key(row)
        values = ACTIVE_HORIZON_SCORE_VALUES[target_type][horizon]
        return reference_horizon_score(values)

    get_horizon_score = callback("get_horizon_score", get_horizon_score)

    def apply_horizon_signal_refresh(predictions_df, rebalance_multiplier):
        if not 0 < rebalance_multiplier <= 1:
            raise ValueError("rebalance_multiplier must be greater than 0 and no greater than 1.")
        required_columns = {"Date", "Ticker", "Portfolio Target Type", "Horizon Key", "Signal"}
        missing_columns = required_columns - set(predictions_df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {sorted(missing_columns)}")
        refreshed = (
            predictions_df.copy()
            .sort_values(["Ticker", "Portfolio Target Type", "Horizon Key", "Date"])
            .reset_index(drop=True)
        )
        group_columns = ["Ticker", "Portfolio Target Type", "Horizon Key"]
        for group_values, group_indexes in refreshed.groupby(
            group_columns, sort=False
        ).groups.items():
            ticker, portfolio_type, horizon_key = group_values
            horizon_key = str(horizon_key).strip().lower()
            if not horizon_key.endswith("d"):
                raise ValueError(
                    f"Daily Horizon Key must end in 'd'. Received {horizon_key!r} for {ticker!r} / {portfolio_type!r}."
                )
            try:
                horizon_days = int(horizon_key[:-1])
            except ValueError as error:
                raise ValueError(
                    f"Could not extract the number of days from Horizon Key {horizon_key!r}."
                ) from error
            refresh_rows = max(1, int(np.ceil(rebalance_multiplier * horizon_days)))
            group_indexes = np.asarray(list(group_indexes))
            original_signals = refreshed.loc[group_indexes, "Signal"].to_numpy()
            row_positions = np.arange(len(original_signals))
            refresh_start_positions = row_positions // refresh_rows * refresh_rows
            refreshed.loc[group_indexes, "Signal"] = original_signals[refresh_start_positions]
        return refreshed.sort_values(
            ["Date", "Ticker", "Portfolio Target Type", "Horizon Key"]
        ).reset_index(drop=True)

    apply_horizon_signal_refresh = callback(
        "apply_horizon_signal_refresh", apply_horizon_signal_refresh
    )

    selected_models_df["Horizon Score"] = 1.0
    logger.info(
        "Fitting selected models on internal TRAIN and generating Horizon-validation predictions"
    )
    prediction_parts = []
    for model_number, (_, model_row) in enumerate(selected_models_df.iterrows(), start=1):
        target = str(model_row["Target"])
        analysis_type = str(model_row["Analysis Type"])
        research_split = RESEARCH_SPLITS[analysis_type]
        features = target_features[target]
        logger.info(
            "[%d/%d] %s | fitting with %d selected features",
            model_number,
            len(selected_models_df),
            target,
            len(features),
        )
        target_train_df = load_target_period(
            target=target,
            features=features,
            start_date=research_split["fit_start"],
            end_date=research_split["fit_end"],
            split="TRAIN",
        )
        target_validation_df = load_target_period(
            target=target,
            features=features,
            start_date=research_split["validation_start"],
            end_date=research_split["validation_end"],
            split="BACKTEST",
        )
        target_model_data = pd.concat([target_train_df, target_validation_df], ignore_index=True)
        one_model_df = pd.DataFrame([model_row.to_dict()])
        prepared = create_models_and_predictions(
            dataframe=target_model_data,
            selected_models_df=one_model_df,
            model_features={target: features},
            purge=False,
        )
        target_predictions = prepared["predictions"].copy()
        target_predictions["Analysis Type"] = model_row["Analysis Type"]
        target_predictions["Portfolio Target Type"] = model_row["Portfolio Target Type"]
        prediction_parts.append(target_predictions)
        logger.info(
            "[%d/%d] %s | predictions complete | rows=%d",
            model_number,
            len(selected_models_df),
            target,
            len(target_predictions),
        )
        del target_train_df
        del target_validation_df
        del target_model_data
        del one_model_df
        del prepared
        del target_predictions
        gc.collect()
    if not prediction_parts:
        raise ValueError("No selected target predictions were generated.")
    predictions_df = pd.concat(prediction_parts, ignore_index=True)
    del prediction_parts
    train_sql_columns = ", ".join((quote_sql_identifier(target) for target in selected_targets))
    actual_train_end = min(
        (RESEARCH_SPLITS[analysis_type]["fit_end"] for analysis_type in ACTIVE_ANALYSIS_TYPES)
    )
    with sqlite3.connect(FEATURE_DATABASE) as connection:
        train_df = pd.read_sql_query(
            f"\n        SELECT {train_sql_columns}\n        FROM {quote_sql_identifier(STOCK_TYPE)}\n        WHERE date({quote_sql_identifier('Date')})\n              <= date(?)\n        ",
            connection,
            params=[pd.Timestamp(actual_train_end).strftime("%Y-%m-%d")],
        )
    train_df[selected_targets] = train_df[selected_targets].apply(pd.to_numeric, errors="coerce")
    target_metrics = (
        train_df[selected_targets]
        .agg(["mean", "std"])
        .T.rename(columns={"mean": "Mean", "std": "Std"})
    )
    target_metrics.index.name = "Target"
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

    def get_target_orientation(target):
        target_tokens = str(target).lower().split()
        matches = []
        for base_target, orientation in TARGET_ORIENTATION.items():
            base_tokens = base_target.lower().split()
            target_iterator = iter(target_tokens)
            is_match = all(
                (
                    any((target_token == base_token for target_token in target_iterator))
                    for base_token in base_tokens
                )
            )
            if is_match:
                matches.append((len(base_tokens), orientation, base_target))
        if not matches:
            raise KeyError(f"No TARGET_ORIENTATION entry matched target: {target}")
        _, orientation, _ = max(matches, key=lambda value: value[0])
        return orientation

    get_target_orientation = callback("get_target_orientation", get_target_orientation)

    target_values = {}
    for target, metrics in target_metrics.iterrows():
        target_values[target] = (
            float(metrics["Mean"]),
            float(metrics["Std"]),
            get_target_orientation(target),
        )
    logger.info(
        "Prediction generation complete | rows=%d | targets=%d | %.1f MB",
        len(predictions_df),
        predictions_df["Target"].nunique(),
        dataframe_memory_mb(predictions_df),
    )
    predictions_df["Horizon Key"] = predictions_df.apply(horizon_key, axis=1)
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
    MULTICLASS_PORTFOLIO_TARGET_TYPES = {"DIRECTION_MULTICLASS", "BARRIER_ALPHA", "REGIME"}

    def prediction_to_signal(row):
        if row["Portfolio Target Type"] in CONTINUOUS_PORTFOLIO_TARGET_TYPES:
            metrics = target_values[row["Target"]]
            signal = metrics[2] * ((row["Prediction"] - metrics[0]) / metrics[1])
        elif row["Portfolio Target Type"] in BINARY_PORTFOLIO_TARGET_TYPES:
            target = row["Target"]
            metrics = target_values[target]
            prediction = pd.to_numeric(row["Prediction"], errors="coerce")
            if pd.isna(prediction):
                signal = 0.0
            else:
                prediction = float(np.clip(prediction, 0.0, 1.0))
                p0 = metrics[0]
                if not np.isfinite(p0) or p0 <= 0.0 or p0 >= 1.0:
                    signal = 0.0
                elif prediction >= p0:
                    signal = (prediction - p0) / (1.0 - p0)
                else:
                    signal = (prediction - p0) / p0
                signal = metrics[2] * float(np.clip(signal, -1.0, 1.0))
        elif row["Portfolio Target Type"] in MULTICLASS_PORTFOLIO_TARGET_TYPES:
            target = row["Target"]
            prediction = pd.to_numeric(row["Prediction"], errors="coerce")
            if pd.isna(prediction):
                signal = 0.0
            else:
                class_values = pd.to_numeric(train_df[target], errors="coerce").dropna().unique()
                if len(class_values) < 2:
                    signal = 0.0
                else:
                    lower_class = float(np.min(class_values))
                    upper_class = float(np.max(class_values))
                    signal = (
                        2.0 * (float(prediction) - lower_class) / (upper_class - lower_class) - 1.0
                    )
                    signal = target_values[target][2] * float(np.clip(signal, -1.0, 1.0))
        else:
            raise ValueError(f"Unknown Portfolio Target Type: {row['Portfolio Target Type']}")
        return signal

    prediction_to_signal = callback("prediction_to_signal", prediction_to_signal)

    predictions_df["Signal"] = predictions_df.apply(prediction_to_signal, axis=1)
    predictions_df.dropna(subset=["Signal"], inplace=True)
    predictions_df.reset_index(drop=True, inplace=True)
    predictions_df["Adjusted Signal"] = predictions_df["Signal"] * predictions_df["Quality Score"]
    predictions_df = predictions_df[
        ["Date", "Ticker", "Return", "Portfolio Target Type", "Horizon Key", "Adjusted Signal"]
    ]
    predictions_df = predictions_df.groupby(
        ["Date", "Ticker", "Portfolio Target Type", "Horizon Key"], as_index=False
    ).agg(Return=("Return", "first"), Signal=("Adjusted Signal", "mean"))
    REBALANCE_MULTIPLIER = get_setting("REBALANCE_MULTIPLIER", 1)
    predictions_df = apply_horizon_signal_refresh(
        predictions_df=predictions_df, rebalance_multiplier=REBALANCE_MULTIPLIER
    )
    predictions_df["Horizon Score"] = predictions_df.apply(get_horizon_score, axis=1)
    direction_types = ["DIRECTION", "DIRECTION_MULTICLASS", "ALPHA_BINARY", "BARRIER_ALPHA"]
    predictions_df["Direction Signal"] = (
        predictions_df["Signal"]
        .where(predictions_df["Portfolio Target Type"].isin(direction_types), 0.0)
        .fillna(0.0)
    )
    initial_predictions_df = predictions_df
    logger.info(
        "Initial Horizon Scores assigned | missing=%d", predictions_df["Horizon Score"].isna().sum()
    )
    PORTFOLIO_TYPE_GROUPS = {
        "Ranking": {"ALPHA", "RELATIVE_ALPHA", "RISK_ADJUSTED_ALPHA", "CROSS_SECTION_ALPHA"},
        "Direction": {"DIRECTION", "DIRECTION_MULTICLASS", "ALPHA_BINARY", "BARRIER_ALPHA"},
        "Risk": {
            "VOLATILITY",
            "DOWNSIDE_VOLATILITY",
            "VOLATILITY_ASYMMETRY",
            "DOWNSIDE",
            "TAIL_RISK",
            "TAIL_EVENT",
            "DOWNSIDE_EXCURSION",
            "VOLATILITY_EVENT",
            "CROSS_SECTION_DOWNSIDE",
        },
        "Opportunity": {
            "ABSOLUTE_MOVE",
            "UPSIDE_VOLATILITY",
            "UPSIDE_EVENT",
            "UPSIDE_EXCURSION",
            "RECOVERY",
            "REVERSAL",
        },
        "Special": {
            "TIME_TO_DOWNSIDE_EXCURSION",
            "TIME_TO_UPSIDE_EXCURSION",
            "EXECUTION",
            "LIQUIDITY",
            "MARKET_IMPACT",
            "CORRELATION",
            "COVARIANCE",
            "REGIME",
        },
    }
    HORIZON_TEST_TYPE_CONFIGURATIONS = get_setting(
        "HORIZON_TEST_TYPE_CONFIGURATIONS",
        [
            {
                "Name": "Equal Weight Baseline",
                "Ranking": 0.2,
                "Direction": 0.2,
                "Risk": 0.2,
                "Opportunity": 0.2,
                "Special": 0.2,
            },
            {
                "Name": "Signal Focused",
                "Ranking": 0.4,
                "Direction": 0.3,
                "Risk": 0.15,
                "Opportunity": 0.1,
                "Special": 0.05,
            },
            {
                "Name": "Defensive Opportunity",
                "Ranking": 0.2,
                "Direction": 0.15,
                "Risk": 0.35,
                "Opportunity": 0.2,
                "Special": 0.1,
            },
        ],
    )
    HORIZON_TEST_MAX_WEIGHTS = get_setting("HORIZON_TEST_MAX_WEIGHTS", [0.2, 0.1, 0.3])
    HORIZON_TEST_CONCENTRATION_PENALTIES = get_setting(
        "HORIZON_TEST_CONCENTRATION_PENALTIES", [0.1, 0.0, 0.3]
    )
    BASELINE_TYPE_CONFIGURATION = HORIZON_TEST_TYPE_CONFIGURATIONS[0]
    BASELINE_MAX_WEIGHT = HORIZON_TEST_MAX_WEIGHTS[0]
    BASELINE_CONCENTRATION_PENALTY = HORIZON_TEST_CONCENTRATION_PENALTIES[0]
    USE_CARTESIAN_HORIZON_TEST_SETTINGS = setting_mode == "cartesian"
    if USE_CARTESIAN_HORIZON_TEST_SETTINGS:
        HORIZON_TEST_SETTINGS = [
            {
                "Name": f"{type_configuration['Name']} | Max Weight {max_weight:.2f} | Penalty {concentration_penalty:.2f}",
                "Type Configuration": type_configuration,
                "Max Weight": max_weight,
                "Concentration Penalty": concentration_penalty,
            }
            for type_configuration, max_weight, concentration_penalty in itertools.product(
                HORIZON_TEST_TYPE_CONFIGURATIONS,
                HORIZON_TEST_MAX_WEIGHTS,
                HORIZON_TEST_CONCENTRATION_PENALTIES,
            )
        ]
    else:
        HORIZON_TEST_SETTINGS = [
            {
                "Name": "Baseline",
                "Type Configuration": BASELINE_TYPE_CONFIGURATION,
                "Max Weight": BASELINE_MAX_WEIGHT,
                "Concentration Penalty": BASELINE_CONCENTRATION_PENALTY,
            },
            *[
                {
                    "Name": type_configuration["Name"],
                    "Type Configuration": type_configuration,
                    "Max Weight": BASELINE_MAX_WEIGHT,
                    "Concentration Penalty": BASELINE_CONCENTRATION_PENALTY,
                }
                for type_configuration in HORIZON_TEST_TYPE_CONFIGURATIONS[1:]
            ],
            *[
                {
                    "Name": f"Max Weight {max_weight:.2f}",
                    "Type Configuration": BASELINE_TYPE_CONFIGURATION,
                    "Max Weight": max_weight,
                    "Concentration Penalty": BASELINE_CONCENTRATION_PENALTY,
                }
                for max_weight in HORIZON_TEST_MAX_WEIGHTS[1:]
            ],
            *[
                {
                    "Name": f"Concentration Penalty {concentration_penalty:.2f}",
                    "Type Configuration": BASELINE_TYPE_CONFIGURATION,
                    "Max Weight": BASELINE_MAX_WEIGHT,
                    "Concentration Penalty": concentration_penalty,
                }
                for concentration_penalty in HORIZON_TEST_CONCENTRATION_PENALTIES[1:]
            ],
        ]
    logger.info(
        "Horizon test settings ready | settings=%d | names=%s",
        len(HORIZON_TEST_SETTINGS),
        [setting["Name"] for setting in HORIZON_TEST_SETTINGS],
    )
    for setting in HORIZON_TEST_SETTINGS:
        logger.info(
            "Horizon test setting | %s | type=%s | max_weight=%.2f | concentration_penalty=%.2f",
            setting["Name"],
            setting["Type Configuration"]["Name"],
            setting["Max Weight"],
            setting["Concentration Penalty"],
        )
    HORIZON_FREEZE_BQ_RANGE = get_setting("HORIZON_FREEZE_BQ_RANGE", 0.003)

    def normalize_horizon_scores_by_type_group(dataframe):
        normalized = dataframe.copy()
        for target_types in PORTFOLIO_TYPE_GROUPS.values():
            group_mask = normalized["Portfolio Target Type"].isin(target_types)
            if not group_mask.any():
                continue
            active_horizon_scores = (
                normalized.loc[
                    group_mask, ["Portfolio Target Type", "Horizon Key", "Horizon Score"]
                ]
                .groupby(["Portfolio Target Type", "Horizon Key"], as_index=False)
                .agg(Horizon_Score=("Horizon Score", "first"))
            )
            total_horizon_score = active_horizon_scores["Horizon_Score"].sum()
            if not np.isfinite(total_horizon_score) or total_horizon_score <= 0:
                continue
            normalized.loc[group_mask, "Horizon Score"] = (
                normalized.loc[group_mask, "Horizon Score"] / total_horizon_score
            )
        return normalized

    normalize_horizon_scores_by_type_group = callback(
        "normalize_horizon_scores_by_type_group", normalize_horizon_scores_by_type_group
    )

    def apply_type_configuration(dataframe, type_configuration):
        weighted = dataframe.copy()
        for group_name, target_types in PORTFOLIO_TYPE_GROUPS.items():
            group_mask = weighted["Portfolio Target Type"].isin(target_types)
            weighted.loc[group_mask, "Horizon Score"] = (
                weighted.loc[group_mask, "Horizon Score"] * type_configuration[group_name]
            )
        return weighted

    apply_type_configuration = callback("apply_type_configuration", apply_type_configuration)

    def backtest_quality(df, test_setting=None):
        if test_setting is None:
            test_setting = HORIZON_TEST_SETTINGS[0]
        normalized_df = normalize_horizon_scores_by_type_group(df)
        weighted_df = apply_type_configuration(
            dataframe=normalized_df, type_configuration=test_setting["Type Configuration"]
        )
        results = run_portfolio_backtest_from_predictions(
            predictions_df=weighted_df,
            max_weight=test_setting["Max Weight"],
            concentration_penalty=test_setting["Concentration Penalty"],
            trading_fee=0.0,
        )
        epsilon = 1e-12
        strategy_return = float(results["Strategy Return"])
        market_return = float(market_results["Return"])
        relative_return = (strategy_return - market_return) / (
            abs(strategy_return) + abs(market_return) + epsilon
        )
        strategy_sharpe = float(results["Sharpe Ratio"])
        market_sharpe = float(market_results["Sharpe Ratio"])
        relative_sharpe = (strategy_sharpe - market_sharpe) / (
            abs(strategy_sharpe) + abs(market_sharpe) + epsilon
        )
        strategy_max_drawdown = abs(float(results["Max Drawdown"]))
        market_max_drawdown = abs(float(market_results["Max Drawdown"]))
        relative_max_drawdown = (market_max_drawdown - strategy_max_drawdown) / (
            market_max_drawdown + strategy_max_drawdown + epsilon
        )
        strategy_average_drawdown = abs(float(results["Average Drawdown"]))
        market_average_drawdown = abs(float(market_results["Average Drawdown"]))
        relative_average_drawdown = (market_average_drawdown - strategy_average_drawdown) / (
            market_average_drawdown + strategy_average_drawdown + epsilon
        )
        return (
            0.25 * relative_sharpe
            + 0.35 * relative_return
            + 0.25 * relative_max_drawdown
            + 0.15 * relative_average_drawdown
        )

    backtest_quality = callback("backtest_quality", backtest_quality)

    HORIZON_SCORES = {
        target_type: {horizon: values.copy() for horizon, values in horizons.items()}
        for target_type, horizons in ACTIVE_HORIZON_SCORE_VALUES.items()
    }
    logger.info(
        "Candidate Horizon Score grid built | parameters=%d | configurations=%d",
        sum((len(horizons) for horizons in HORIZON_SCORES.values())),
        math.prod(
            (len(values) for horizons in HORIZON_SCORES.values() for values in horizons.values())
        ),
    )
    frozen_df = predictions_df.copy()
    for target_type, horizons in HORIZON_SCORES.items():
        for horizon, values in horizons.items():
            frozen_df.loc[
                (frozen_df["Portfolio Target Type"] == target_type)
                & (frozen_df["Horizon Key"] == horizon),
                "Horizon Score",
            ] = reference_horizon_score(values)
    sensitivity_results = []
    for target_type, horizons in list(HORIZON_SCORES.items()):
        for horizon, values in list(horizons.items()):
            qualities_by_setting = {}
            ranges_by_setting = {}
            freeze_parameter = True
            logger.info(
                "Sensitivity screen | %s %s | candidates=%d", target_type, horizon, len(values)
            )
            for setting_number, setting in enumerate(HORIZON_TEST_SETTINGS, start=1):
                setting_qualities = []
                for score in values:
                    test_df = frozen_df.copy()
                    test_df.loc[
                        (test_df["Portfolio Target Type"] == target_type)
                        & (test_df["Horizon Key"] == horizon),
                        "Horizon Score",
                    ] = score
                    BQ = backtest_quality(test_df, test_setting=setting)
                    setting_qualities.append(BQ)
                qualities_by_setting[setting["Name"]] = setting_qualities
                setting_range = float(np.max(setting_qualities)) - float(np.min(setting_qualities))
                ranges_by_setting[setting["Name"]] = setting_range
                if setting_range >= HORIZON_FREEZE_BQ_RANGE:
                    freeze_parameter = False
                    logger.info(
                        "Sensitivity early exit | %s %s | setting=%s (%d/%d) | range=%.6f disqualifies freezing",
                        target_type,
                        horizon,
                        setting["Name"],
                        setting_number,
                        len(HORIZON_TEST_SETTINGS),
                        setting_range,
                    )
                    break
            mean_quality = float(
                np.mean(
                    [
                        quality
                        for qualities in qualities_by_setting.values()
                        for quality in qualities
                    ]
                )
            )
            maximum_range_quality = max(ranges_by_setting.values())
            sensitivity_results.append(
                {
                    "Portfolio Target Type": target_type,
                    "Horizon": horizon,
                    "Mean BQ": mean_quality,
                    "BQ Ranges By Setting": ranges_by_setting,
                    "Maximum BQ Range": maximum_range_quality,
                    "Frozen": freeze_parameter,
                }
            )
            logger.info(
                "Sensitivity result | %s %s | Mean BQ=%.6f | Max Range BQ=%.6f | Frozen=%s",
                target_type,
                horizon,
                mean_quality,
                maximum_range_quality,
                freeze_parameter,
            )
            if freeze_parameter:
                frozen_score = reference_horizon_score(values)
                HORIZON_SCORES[target_type][horizon] = [frozen_score]
                logger.info(
                    "Sensitivity freeze | %s %s -> %.2f | all %d setting ranges below %.6f",
                    target_type,
                    horizon,
                    frozen_score,
                    len(HORIZON_TEST_SETTINGS),
                    HORIZON_FREEZE_BQ_RANGE,
                )
    sensitivity_results = pd.DataFrame(sensitivity_results)
    logger.info(
        "First sensitivity screen complete | frozen=%d | variable=%d",
        sum(
            (
                len(values) == 1
                for horizons in HORIZON_SCORES.values()
                for values in horizons.values()
            )
        ),
        sum(
            (
                len(values) > 1
                for horizons in HORIZON_SCORES.values()
                for values in horizons.values()
            )
        ),
    )
    NEAR_BEST_BQ_TOLERANCE = get_setting("NEAR_BEST_BQ_TOLERANCE", 0.002)

    def random_screen(iterations, threshold):
        global HORIZON_SCORES
        results = []
        logger.info(
            "Random screen started | iterations=%d | threshold=%.3f | settings=%d",
            iterations,
            threshold,
            len(HORIZON_TEST_SETTINGS),
        )
        parameters = [
            (target_type, horizon)
            for target_type, horizons in HORIZON_SCORES.items()
            for horizon in horizons
        ]
        for target_type, horizon in parameters:
            values = HORIZON_SCORES[target_type][horizon].copy()
            if len(values) == 1:
                logger.debug(
                    "Skipping single-candidate parameter | %s %s -> %s",
                    target_type,
                    horizon,
                    values,
                )
                continue
            logger.info(
                "Random screen parameter | %s %s | candidates=%d | backgrounds=%d",
                target_type,
                horizon,
                len(values),
                iterations,
            )
            random_iterations = []
            for _ in range(iterations):
                configuration = {}
                for other_type, other_horizons in HORIZON_SCORES.items():
                    configuration[other_type] = {}
                    for other_horizon, other_values in other_horizons.items():
                        configuration[other_type][other_horizon] = float(
                            np.random.choice(other_values)
                        )
                random_iterations.append(configuration)
            quality_by_setting = {}
            near_best_rates_by_setting = {}
            protected_indices = set()
            best_mean_quality_by_index = {index: -np.inf for index in range(len(values))}
            for setting_number, setting in enumerate(HORIZON_TEST_SETTINGS, start=1):
                quality_by_index = {index: [] for index in range(len(values))}
                for configuration in random_iterations:
                    for index, score in enumerate(values):
                        test_df = predictions_df.copy()
                        for config_type, config_horizons in configuration.items():
                            for config_horizon, config_score in config_horizons.items():
                                test_df.loc[
                                    (test_df["Portfolio Target Type"] == config_type)
                                    & (test_df["Horizon Key"] == config_horizon),
                                    "Horizon Score",
                                ] = config_score
                        test_df.loc[
                            (test_df["Portfolio Target Type"] == target_type)
                            & (test_df["Horizon Key"] == horizon),
                            "Horizon Score",
                        ] = score
                        BQ = backtest_quality(test_df, test_setting=setting)
                        quality_by_index[index].append(BQ)
                quality_by_setting[setting["Name"]] = quality_by_index
                near_best_counts = {index: 0 for index in quality_by_index}
                for iteration in range(iterations):
                    iteration_scores = {
                        index: quality_by_index[index][iteration] for index in quality_by_index
                    }
                    best_iteration_quality = max(iteration_scores.values())
                    for index, quality in iteration_scores.items():
                        if quality >= best_iteration_quality - NEAR_BEST_BQ_TOLERANCE:
                            near_best_counts[index] += 1
                setting_rates = {
                    index: near_best_counts[index] / iterations for index in near_best_counts
                }
                near_best_rates_by_setting[setting["Name"]] = setting_rates
                for index in range(len(values)):
                    best_mean_quality_by_index[index] = max(
                        best_mean_quality_by_index[index], float(np.mean(quality_by_index[index]))
                    )
                    if setting_rates[index] > threshold:
                        protected_indices.add(index)
                if len(protected_indices) == len(values):
                    logger.info(
                        "Random screen early exit | %s %s | setting=%s (%d/%d) | every value is protected",
                        target_type,
                        horizon,
                        setting["Name"],
                        setting_number,
                        len(HORIZON_TEST_SETTINGS),
                    )
                    break
            maximum_near_best_rate = {
                index: max(
                    (setting_rates[index] for setting_rates in near_best_rates_by_setting.values())
                )
                for index in range(len(values))
            }
            best_index = max(best_mean_quality_by_index, key=best_mean_quality_by_index.get)
            remaining_values = [
                value
                for index, value in enumerate(values)
                if maximum_near_best_rate[index] > threshold or index == best_index
            ]
            HORIZON_SCORES[target_type][horizon] = remaining_values
            if len(remaining_values) == 1:
                logger.info(
                    "Random screen reduced parameter to one value | %s %s -> %.2f | no alternative passed in any setting",
                    target_type,
                    horizon,
                    remaining_values[0],
                )
            logger.info(
                "Random screen result | %s %s | max near-best rates=%s | passing settings=%s | best_index=%d | kept=%s",
                target_type,
                horizon,
                {index: round(rate, 3) for index, rate in maximum_near_best_rate.items()},
                {
                    values[index]: [
                        setting_name
                        for setting_name, setting_rates in near_best_rates_by_setting.items()
                        if setting_rates[index] > threshold
                    ]
                    for index in range(len(values))
                },
                best_index,
                remaining_values,
            )
            results.append(
                {
                    "Portfolio Target Type": target_type,
                    "Horizon": horizon,
                    "Original Values": values,
                    "Mean BQ By Setting": {
                        setting_name: {
                            index: np.mean(qualities)
                            for index, qualities in quality_by_index.items()
                        }
                        for setting_name, quality_by_index in quality_by_setting.items()
                    },
                    "Near-Best Rates By Setting": near_best_rates_by_setting,
                    "Maximum Near-Best Rate": maximum_near_best_rate,
                    "Best Index": best_index,
                    "Best Value": values[best_index],
                    "Remaining Values": remaining_values,
                    "Random Iterations": random_iterations,
                }
            )
        logger.info(
            "Random screen complete | iterations=%d | threshold=%.3f | remaining configurations=%d",
            iterations,
            threshold,
            math.prod(
                (
                    len(values)
                    for horizons in HORIZON_SCORES.values()
                    for values in horizons.values()
                )
            ),
        )
        return results

    random_screen = callback("random_screen", random_screen)

    def remaining_configurations():
        return math.prod(
            len(values) for horizons in HORIZON_SCORES.values() for values in horizons.values()
        )

    results = run_screen_schedule(
        random_screen,
        remaining_configurations,
        get_setting("RANDOM_SCREENS", [(20, 0.15), (50, 0.15), (100, 0.2)]),
        get_setting("EXTRA_RANDOM_SCREENS", []),
        get_setting("MAX_EXHAUSTIVE_CONFIGURATIONS", 1000),
    )
    total_configurations = remaining_configurations()
    parameters = [
        (target_type, horizon)
        for target_type, horizons in HORIZON_SCORES.items()
        for horizon in horizons
    ]
    possible_values = [HORIZON_SCORES[target_type][horizon] for target_type, horizon in parameters]
    exhaustive_results = []
    exhaustive_total = math.prod((len(values) for values in possible_values))
    logger.info(
        "Starting exhaustive search | configurations=%d | settings per configuration=%d | total backtests=%d",
        exhaustive_total,
        len(HORIZON_TEST_SETTINGS),
        exhaustive_total * len(HORIZON_TEST_SETTINGS),
    )
    log_every = max(1, exhaustive_total // 20)
    for combination_number, combination in enumerate(itertools.product(*possible_values), start=1):
        if (
            combination_number == 1
            or combination_number % log_every == 0
            or combination_number == exhaustive_total
        ):
            logger.info(
                "Exhaustive search progress | %d/%d (%.1f%%)",
                combination_number,
                exhaustive_total,
                100 * combination_number / exhaustive_total,
            )
        test_df = predictions_df.copy()
        configuration = {}
        for (target_type, horizon), score in zip(parameters, combination):
            test_df.loc[
                (test_df["Portfolio Target Type"] == target_type)
                & (test_df["Horizon Key"] == horizon),
                "Horizon Score",
            ] = score
            configuration.setdefault(target_type, {})[horizon] = float(score)
        bq_by_setting = {
            setting["Name"]: backtest_quality(test_df, test_setting=setting)
            for setting in HORIZON_TEST_SETTINGS
        }
        best_setting = max(bq_by_setting, key=bq_by_setting.get)
        BQ = bq_by_setting[best_setting]
        exhaustive_results.append(
            {
                "BQ": float(BQ),
                "BQ By Setting": bq_by_setting,
                "Best Setting": best_setting,
                "Horizon Scores": configuration,
            }
        )
    exhaustive_results.sort(key=lambda result: result["BQ"], reverse=True)
    logger.info(
        "Exhaustive search best-setting counts=%s",
        {
            setting["Name"]: sum(
                (result["Best Setting"] == setting["Name"] for result in exhaustive_results)
            )
            for setting in HORIZON_TEST_SETTINGS
        },
    )
    exhaustive_results = [result["Horizon Scores"] for result in exhaustive_results]
    logger.info(
        "Exhaustive search complete | Configurations=%d/%d",
        len(exhaustive_results),
        exhaustive_total,
    )
    CONFIGURATION_CORRELATION_THRESHOLD = get_setting("CONFIGURATION_CORRELATION_THRESHOLD", 0.95)

    def normalize_horizon_configuration(configuration):
        normalized_configuration = {
            target_type: {horizon: float(score) for horizon, score in horizon_scores.items()}
            for target_type, horizon_scores in configuration.items()
        }
        for target_types in PORTFOLIO_TYPE_GROUPS.values():
            group_total = sum(
                (
                    score
                    for target_type, horizon_scores in normalized_configuration.items()
                    if target_type in target_types
                    for score in horizon_scores.values()
                )
            )
            if group_total <= 0:
                continue
            for target_type, horizon_scores in normalized_configuration.items():
                if target_type not in target_types:
                    continue
                for horizon in horizon_scores:
                    horizon_scores[horizon] = round(horizon_scores[horizon] / group_total, 10)
        return normalized_configuration

    normalize_horizon_configuration = callback(
        "normalize_horizon_configuration", normalize_horizon_configuration
    )

    def configuration_vector(configuration):
        return np.array(
            [configuration[target_type][horizon] for target_type, horizon in parameters],
            dtype=float,
        )

    configuration_vector = callback("configuration_vector", configuration_vector)

    if exhaustive_results:
        num_of_configs = len(exhaustive_results)
        target_configurations = min(num_of_configs, max(5, int(np.sqrt(num_of_configs))))
        normalized_top_configurations = []
        selected_vectors = []
        seen_normalized_configurations = set()
        exact_duplicates_skipped = 0
        correlated_configurations_skipped = 0
        for configuration in exhaustive_results:
            normalized_configuration = normalize_horizon_configuration(configuration)
            candidate_vector = configuration_vector(normalized_configuration)
            candidate_key = tuple(np.round(candidate_vector, 10))
            if candidate_key in seen_normalized_configurations:
                exact_duplicates_skipped += 1
                continue
            highly_correlated = False
            for selected_vector in selected_vectors:
                candidate_standard_deviation = np.std(candidate_vector)
                selected_standard_deviation = np.std(selected_vector)
                if candidate_standard_deviation == 0 or selected_standard_deviation == 0:
                    continue
                correlation = float(np.corrcoef(candidate_vector, selected_vector)[0, 1])
                if np.isfinite(correlation) and correlation >= CONFIGURATION_CORRELATION_THRESHOLD:
                    highly_correlated = True
                    break
            if highly_correlated:
                correlated_configurations_skipped += 1
                continue
            normalized_top_configurations.append(normalized_configuration)
            selected_vectors.append(candidate_vector)
            seen_normalized_configurations.add(candidate_key)
            if len(normalized_top_configurations) >= target_configurations:
                break
        logger.info(
            "Diverse top configurations retained=%d/%d | exact duplicates skipped=%d | correlation >= %.2f skipped=%d",
            len(normalized_top_configurations),
            target_configurations,
            exact_duplicates_skipped,
            CONFIGURATION_CORRELATION_THRESHOLD,
            correlated_configurations_skipped,
        )
        if len(normalized_top_configurations) < target_configurations:
            logger.warning(
                "Only %d of %d requested configurations were sufficiently distinct.",
                len(normalized_top_configurations),
                target_configurations,
            )
    else:
        normalized_top_configurations = []
        logger.warning("No BQ configurations survived.")
    HORIZON_DIR = DATA_DIR / "Top_Horizon_Scores.txt"
    with open(HORIZON_DIR, "w") as file:
        file.write(str(normalized_top_configurations))
    logger.info(
        "Normalized horizon configurations saved | configurations=%d | file=%s",
        len(normalized_top_configurations),
        HORIZON_DIR,
    )

from equity_selector.settings import setting as get_setting, callback

"""simulations: original sequential research stage; shared logic is in equity_selector."""

from equity_selector.config import data_root
from main_package import *


def run():
    global \
        BACKTEST_DATABASE, \
        BASE_BQ_THRESHOLD, \
        CONCENTRATION_PENALTIES, \
        DATA_DIR, \
        HORIZON_SCORES_FILE, \
        MAXIMUM_STRESSED_BQ_THRESHOLD, \
        MAX_WEIGHTS, \
        MINIMUM_REMOVED_FRACTION, \
        MINIMUM_STRESSED_BQ_THRESHOLD, \
        PORTFOLIO_DIRECTION_TYPES, \
        PORTFOLIO_GROUP_CONFIGURATIONS, \
        PORTFOLIO_OPPORTUNITY_TYPES, \
        PORTFOLIO_RANKING_TYPES, \
        PORTFOLIO_RISK_TYPES, \
        PORTFOLIO_SPECIAL_TYPES, \
        PORTFOLIO_TYPE_REJECTION_THRESHOLDS, \
        PRIMARY_REJECTION_TYPES, \
        Path, \
        REBALANCE_MULTIPLIERS, \
        SECONDARY_REJECTION_TYPES, \
        SIMULATION_RESULTS_DATABASE, \
        TERTIARY_REJECTION_TYPES, \
        THRESHOLD_INCREASE_MULTIPLIER, \
        add_horizon_scores, \
        add_type_scores, \
        align_simulation_results, \
        apply_horizon_signal_refresh, \
        ascending, \
        ast, \
        average_stock_overlap, \
        backtest_standard_deviation, \
        benchmark_summary, \
        best_21_day_end, \
        best_21_day_end_position, \
        best_21_day_start_position, \
        best_252_day_end, \
        best_252_day_end_position, \
        best_252_day_start_position, \
        best_5_day_end, \
        best_5_day_end_position, \
        best_5_day_start_position, \
        best_day, \
        best_day_position, \
        build_score_stocks_with_direction, \
        column, \
        combinations, \
        comparison, \
        competitor, \
        competitor_id, \
        config, \
        configuration, \
        configuration_distance, \
        configuration_keys, \
        connection, \
        current_stocks, \
        deviations, \
        difference, \
        existing_tables, \
        extract_type_score_columns, \
        file, \
        final_evaluation_column, \
        final_evaluation_common, \
        final_evaluation_extra_columns, \
        final_evaluation_id, \
        final_evaluation_market_row, \
        final_evaluation_market_rows, \
        final_evaluation_market_values, \
        final_evaluation_passed_columns, \
        final_evaluation_period_name, \
        final_evaluation_relative, \
        final_evaluation_relative_mask, \
        final_evaluation_removals, \
        final_evaluation_removed_quality, \
        final_evaluation_row, \
        final_evaluation_stock_removals, \
        final_evaluation_storage, \
        final_evaluation_strategy_values, \
        final_evaluation_type_columns, \
        final_evaluation_type_sd, \
        final_evaluation_valid, \
        final_evaluation_worst, \
        gc, \
        get_two_closest_configurations, \
        group_deviations, \
        group_neighbourhood_score, \
        group_neighbourhood_scores, \
        group_score_deviations, \
        higher_is_better, \
        horizon_score_configurations, \
        is_redundant, \
        iteration, \
        itertools, \
        kept_simulation_id, \
        kept_simulations, \
        logger, \
        logging, \
        lower_is_better, \
        market, \
        market_average_drawdown, \
        market_backtest_quality, \
        market_max_drawdown, \
        market_results, \
        market_return, \
        market_sharpe, \
        market_simulations_results, \
        matched_returns, \
        math, \
        metric_name, \
        neighbour_configuration, \
        neighbour_configurations, \
        neighbour_qualities, \
        neighbour_stocks, \
        no_worse, \
        np, \
        number_of_backtest_rows, \
        number_of_iterations, \
        number_of_stocks, \
        number_to_select, \
        original_backtest_quality, \
        original_strategy_quality, \
        pair, \
        pd, \
        percentage_difference, \
        percentage_from_pair_median, \
        performance_differences, \
        performance_metric_names, \
        performance_metrics, \
        performance_percentage_threshold, \
        performance_similar, \
        perturbed_concentration_penalty, \
        perturbed_max_weight, \
        perturbed_rebalance_multiplier, \
        perturbed_stocks, \
        perturbed_strategy_qualities, \
        portfolio_group_robustness_threshold, \
        random_deviations, \
        random_generator, \
        random_neighbourhood_scalar, \
        random_neighbourhood_score, \
        random_neighbourhood_scores, \
        random_score_deviations, \
        ranked_simulations, \
        rebalanced_current_stocks, \
        rebalanced_neighbour_stocks, \
        remaining_simulations, \
        removed_fraction, \
        removed_results, \
        removed_score_stocks, \
        removed_ticker, \
        result_metrics, \
        results_a, \
        results_b, \
        return_correlation, \
        return_correlation_threshold, \
        returns_a, \
        returns_b, \
        rolling_period, \
        rolling_periods, \
        rolling_test_settings, \
        row, \
        rows_after, \
        rows_after_return_filter, \
        rows_before, \
        rows_before_return_filter, \
        run_simulations, \
        save_final_evaluation_values, \
        score_stocks, \
        select_and_remove, \
        selected_simulation_groups, \
        selected_simulations, \
        selection_column, \
        selection_steps, \
        settings_robustness_threshold, \
        similar_pairs, \
        simulation, \
        simulation_backtest_results, \
        simulation_id, \
        simulation_id_a, \
        simulation_id_b, \
        simulation_ids, \
        simulation_pair_comparisons, \
        simulation_standard_deviation, \
        simulations_to_remove, \
        sqlite3, \
        standard_deviation, \
        stock_overlap, \
        stock_overlap_threshold, \
        stock_removal_threshold, \
        stocks, \
        stocks_results, \
        stocks_results_without_best_21_days, \
        stocks_results_without_best_252_days, \
        stocks_results_without_best_5_days, \
        stocks_results_without_best_day, \
        stop_if_no_simulations, \
        strategy_average_drawdown, \
        strategy_max_drawdown, \
        strategy_quality, \
        strategy_relative_average_drawdown, \
        strategy_relative_max_drawdown, \
        strategy_relative_return, \
        strategy_relative_sharpe, \
        strategy_return, \
        strategy_sharpe, \
        strategy_simulations_results, \
        stressed_bq_threshold, \
        strictly_better, \
        sys, \
        ticker_columns, \
        tickers, \
        too_similar, \
        total_backtests, \
        type_score_columns, \
        type_score_rejection, \
        type_scores, \
        unseen_deviation, \
        unseen_quality_scores, \
        unseen_stock_robustness_threshold, \
        unseen_stocks, \
        used_stocks, \
        value_a, \
        value_b, \
        worst_deviation, \
        worst_group_deviation, \
        write_frame, \
        yf
    from equity_selector.database import write_frame
    import ast
    import gc
    import itertools
    import logging
    import math
    import sqlite3
    from pathlib import Path
    import sys
    import numpy as np
    import pandas as pd
    import yfinance as yf

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S"
    )
    logger = logging.getLogger(__name__)
    DATA_DIR = Path(str(data_root()) + "/")
    BACKTEST_DATABASE = DATA_DIR / "Backtest_Database.db"
    HORIZON_SCORES_FILE = DATA_DIR / "Top_Horizon_Scores.txt"
    SIMULATION_RESULTS_DATABASE = DATA_DIR / "Portfolio_Simulation_Results.db"
    with sqlite3.connect(BACKTEST_DATABASE) as connection:
        market = pd.read_sql_query('SELECT * FROM "Market"', connection)
        stocks = pd.read_sql_query('SELECT * FROM "Stocks"', connection)
    market["Date"] = pd.to_datetime(market["Date"])
    stocks["Date"] = pd.to_datetime(stocks["Date"])
    logger.info("Loaded backtest data: %d market rows and %d stock rows", len(market), len(stocks))
    with open(HORIZON_SCORES_FILE, "r") as file:
        horizon_score_configurations = ast.literal_eval(file.read())
    if not isinstance(horizon_score_configurations, list):
        raise TypeError("Top_Horizon_Scores.txt must contain a list of dictionaries.")
    if not all((isinstance(configuration, dict) for configuration in horizon_score_configurations)):
        raise TypeError("Every Horizon Score configuration must be a dictionary.")
    logger.info("Loaded %d horizon-score configurations", len(horizon_score_configurations))
    REBALANCE_MULTIPLIERS = get_setting("REBALANCE_MULTIPLIERS", [0, 0.25, 0.5, 0.75, 1.0])
    CONCENTRATION_PENALTIES = get_setting("CONCENTRATION_PENALTIES", [0.0, 0.05, 0.1, 0.2, 0.3])
    MAX_WEIGHTS = get_setting("MAX_WEIGHTS", [0.1, 0.15, 0.2, 0.25, 0.3])
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
    PORTFOLIO_GROUP_CONFIGURATIONS = get_setting(
        "PORTFOLIO_GROUP_CONFIGURATIONS",
        [
            {
                "Name": "Balanced",
                "Ranking": 0.3,
                "Direction": 0.25,
                "Risk": 0.25,
                "Opportunity": 0.15,
                "Special": 0.05,
            },
            {
                "Name": "Equal Weight",
                "Ranking": 0.2,
                "Direction": 0.2,
                "Risk": 0.2,
                "Opportunity": 0.2,
                "Special": 0.2,
            },
            {
                "Name": "Core Balanced",
                "Ranking": 0.35,
                "Direction": 0.3,
                "Risk": 0.25,
                "Opportunity": 0.1,
                "Special": 0.0,
            },
            {
                "Name": "Ranking Heavy",
                "Ranking": 0.5,
                "Direction": 0.2,
                "Risk": 0.2,
                "Opportunity": 0.1,
                "Special": 0.0,
            },
            {
                "Name": "Ranking And Risk",
                "Ranking": 0.45,
                "Direction": 0.15,
                "Risk": 0.3,
                "Opportunity": 0.1,
                "Special": 0.0,
            },
            {
                "Name": "Ranking And Direction",
                "Ranking": 0.45,
                "Direction": 0.3,
                "Risk": 0.15,
                "Opportunity": 0.1,
                "Special": 0.0,
            },
            {
                "Name": "Direction Heavy",
                "Ranking": 0.25,
                "Direction": 0.45,
                "Risk": 0.2,
                "Opportunity": 0.1,
                "Special": 0.0,
            },
            {
                "Name": "Direction And Risk",
                "Ranking": 0.25,
                "Direction": 0.4,
                "Risk": 0.3,
                "Opportunity": 0.05,
                "Special": 0.0,
            },
            {
                "Name": "Risk Heavy",
                "Ranking": 0.25,
                "Direction": 0.2,
                "Risk": 0.45,
                "Opportunity": 0.1,
                "Special": 0.0,
            },
            {
                "Name": "Conservative",
                "Ranking": 0.3,
                "Direction": 0.15,
                "Risk": 0.45,
                "Opportunity": 0.05,
                "Special": 0.05,
            },
            {
                "Name": "Opportunity Heavy",
                "Ranking": 0.25,
                "Direction": 0.2,
                "Risk": 0.2,
                "Opportunity": 0.35,
                "Special": 0.0,
            },
            {
                "Name": "Ranking And Special",
                "Ranking": 0.4,
                "Direction": 0.15,
                "Risk": 0.2,
                "Opportunity": 0.1,
                "Special": 0.15,
            },
        ],
    )
    total_backtests = (
        len(horizon_score_configurations)
        * len(PORTFOLIO_GROUP_CONFIGURATIONS)
        * len(REBALANCE_MULTIPLIERS)
        * len(MAX_WEIGHTS)
        * len(CONCENTRATION_PENALTIES)
    )
    from equity_selector.scoring import build_score_stocks_with_direction
    from equity_selector.scoring import apply_horizon_signal_refresh
    from equity_selector.scoring import add_horizon_scores
    from equity_selector.scoring import add_type_scores

    def extract_type_score_columns(type_score_dataframe):
        """
        Return one result column for every portfolio target
        type. Supports the type being either in the index or
        in a Portfolio Target Type column.
        """
        if "Portfolio Target Type" in type_score_dataframe.columns:
            type_scores_series = type_score_dataframe.set_index("Portfolio Target Type")[
                "Type_Score"
            ]
        else:
            type_scores_series = type_score_dataframe["Type_Score"]
        return {
            f"Type Score | {portfolio_type}": float(type_score) if pd.notna(type_score) else np.nan
            for portfolio_type, type_score in type_scores_series.items()
        }

    extract_type_score_columns = callback("extract_type_score_columns", extract_type_score_columns)

    market_results = market.groupby("Date", as_index=False).agg(Return=("Return", "first"))
    from equity_selector.metrics import performance_metrics

    benchmark_summary = performance_metrics(market_results["Return"].dropna())
    market_return = benchmark_summary["Return"]
    market_sharpe = benchmark_summary["Sharpe Ratio"]
    market_average_drawdown = benchmark_summary["Average Drawdown"]
    market_max_drawdown = benchmark_summary["Max Drawdown"]

    def result_metrics(results_dataframe):
        from equity_selector.metrics import relative_metrics

        return relative_metrics(
            results_dataframe,
            {
                "Return": market_return,
                "Sharpe Ratio": market_sharpe,
                "Max Drawdown": market_max_drawdown,
                "Average Drawdown": market_average_drawdown,
            },
        )

    result_metrics = callback("result_metrics", result_metrics)

    def run_simulations():
        logger.info("Starting simulation grid: %d backtests", total_backtests)
        stock_simulation_records = []
        market_simulation_records = []
        completed_backtests = 0
        for horizon_configuration_number, scores in enumerate(horizon_score_configurations):
            current_stocks = add_horizon_scores(
                dataframe=stocks, horizon_score_configuration=scores
            )
            current_market = add_horizon_scores(
                dataframe=market, horizon_score_configuration=scores
            )
            for type_scores in PORTFOLIO_GROUP_CONFIGURATIONS:
                current_stocks = add_type_scores(
                    dataframe=current_stocks, type_score_configuration=type_scores
                )
                current_market = add_type_scores(
                    dataframe=current_market, type_score_configuration=type_scores
                )
                current_market["Contribution"] = (
                    current_market["Horizon Score"]
                    * current_market["Signal"]
                    * current_market["Type Score"]
                )
                current_stocks["Contribution"] = (
                    current_stocks["Horizon Score"]
                    * current_stocks["Signal"]
                    * current_stocks["Type Score"]
                )
                market_scores = current_market.groupby(
                    ["Date", "Portfolio Target Type"], as_index=False
                ).agg(Contribution=("Contribution", "sum"))
                market_scores = market_scores.groupby("Portfolio Target Type", as_index=False).agg(
                    Type_Score=("Contribution", "mean")
                )
                for multiplier in REBALANCE_MULTIPLIERS:
                    rebalanced_current_stocks = apply_horizon_signal_refresh(
                        predictions_df=current_stocks, rebalance_multiplier=multiplier
                    )
                    stocks_scores = rebalanced_current_stocks.groupby(
                        ["Date", "Ticker", "Portfolio Target Type"], as_index=False
                    ).agg(Contribution=("Contribution", "sum"))
                    score_stocks = build_score_stocks_with_direction(rebalanced_current_stocks)
                    for max_weight in MAX_WEIGHTS:
                        for penalty in CONCENTRATION_PENALTIES:
                            stocks_results = portfolio_returns_from_scores(
                                score_stocks, max_weight=max_weight, concentration_penalty=penalty
                            )
                            (
                                strategy_return,
                                strategy_sharpe,
                                strategy_average_drawdown,
                                strategy_max_drawdown,
                                strategy_relative_return,
                                strategy_relative_sharpe,
                                strategy_relative_max_drawdown,
                                strategy_relative_average_drawdown,
                                strategy_quality,
                            ) = result_metrics(stocks_results)
                            market_relative_return = (
                                2 * market_return / (abs(market_return) + abs(strategy_return))
                            )
                            market_relative_sharpe = (
                                2 * market_sharpe / (abs(market_sharpe) + abs(strategy_sharpe))
                            )
                            market_relative_max_drawdown = (
                                2
                                * market_max_drawdown
                                / (abs(market_max_drawdown) + abs(strategy_max_drawdown))
                            )
                            market_relative_average_drawdown = (
                                2
                                * market_average_drawdown
                                / (abs(market_average_drawdown) + abs(strategy_average_drawdown))
                            )
                            market_quality = (
                                0.25 * market_relative_sharpe
                                + 0.35 * market_relative_return
                                + 0.25 * market_relative_max_drawdown
                                + 0.15 * market_relative_average_drawdown
                            )
                            market_metrics = {
                                "Strategy Return": market_return,
                                "Average Drawdown": market_average_drawdown,
                                "Max Drawdown": market_max_drawdown,
                                "Sharpe Ratio": market_sharpe,
                                "Relative Return": market_relative_return,
                                "Relative Average Drawdown": market_relative_average_drawdown,
                                "Relative Max Drawdown": market_relative_max_drawdown,
                                "Relative Sharpe Ratio": market_relative_sharpe,
                                "Backtest Quality": market_quality,
                            }
                            stock_metrics = {
                                "Strategy Return": strategy_return,
                                "Average Drawdown": strategy_average_drawdown,
                                "Max Drawdown": strategy_max_drawdown,
                                "Sharpe Ratio": strategy_sharpe,
                                "Relative Return": strategy_relative_return,
                                "Relative Average Drawdown": strategy_relative_average_drawdown,
                                "Relative Max Drawdown": strategy_relative_max_drawdown,
                                "Relative Sharpe Ratio": strategy_relative_sharpe,
                                "Backtest Quality": strategy_quality,
                            }
                            stocks_results["Date"] = pd.to_datetime(stocks_results["Date"])
                            current_score_stocks = stocks_scores.copy()
                            current_score_stocks["Date"] = pd.to_datetime(
                                current_score_stocks["Date"]
                            )
                            ticker_columns = [
                                column
                                for column in stocks_results.columns
                                if column not in {"Date", "Return"}
                            ]
                            weights_long = stocks_results.melt(
                                id_vars=["Date"],
                                value_vars=ticker_columns,
                                var_name="Ticker",
                                value_name="Weight",
                            )
                            current_score_stocks = current_score_stocks.merge(
                                weights_long,
                                on=["Date", "Ticker"],
                                how="left",
                                validate="many_to_one",
                            )
                            current_score_stocks["Weight"] = pd.to_numeric(
                                current_score_stocks["Weight"], errors="coerce"
                            ).fillna(0.0)
                            current_score_stocks["Weighted Score"] = (
                                current_score_stocks["Weight"]
                                * current_score_stocks["Contribution"]
                            )
                            current_score_stocks = current_score_stocks.groupby(
                                ["Date", "Portfolio Target Type"], as_index=False
                            ).agg(Daily_Score=("Weighted Score", "sum"))
                            current_score_stocks = current_score_stocks.groupby(
                                "Portfolio Target Type", as_index=False
                            ).agg(Type_Score=("Daily_Score", "mean"))
                            current_stocks_scores = rebalanced_current_stocks.groupby(
                                ["Ticker", "Portfolio Target Type"], as_index=False
                            ).agg(Type_Score=("Contribution", "mean"))
                            simulation_id = completed_backtests + 1
                            type_configuration_index = PORTFOLIO_GROUP_CONFIGURATIONS.index(
                                type_scores
                            )
                            simulation_settings = {
                                "Simulation ID": simulation_id,
                                "Horizon Score Index": horizon_configuration_number,
                                "Type Configuration": type_scores["Name"],
                                "Rebalance Multiplier": multiplier,
                                "Max Weight": max_weight,
                                "Concentration Penalty": penalty,
                            }
                            stock_simulation_records.append(
                                {
                                    **simulation_settings,
                                    **stock_metrics,
                                    **extract_type_score_columns(current_score_stocks),
                                }
                            )
                            market_simulation_records.append(
                                {
                                    **simulation_settings,
                                    **market_metrics,
                                    **extract_type_score_columns(market_scores),
                                }
                            )
                            completed_backtests += 1
                            progress_interval = max(1, total_backtests // 20)
                            if (
                                completed_backtests % progress_interval == 0
                                or completed_backtests == total_backtests
                                or completed_backtests == 1
                            ):
                                logger.info(
                                    "Simulation grid progress: %d/%d (%.0f%%)",
                                    completed_backtests,
                                    total_backtests,
                                    100 * completed_backtests / total_backtests,
                                )
        stock_simulation_results_df = pd.DataFrame(stock_simulation_records)
        market_simulation_results_df = pd.DataFrame(market_simulation_records)
        with sqlite3.connect(SIMULATION_RESULTS_DATABASE) as connection:
            write_frame(
                stock_simulation_results_df,
                "Stock Simulation Results",
                connection,
                if_exists="replace",
                index=False,
            )
            write_frame(
                market_simulation_results_df,
                "Market Simulation Results",
                connection,
                if_exists="replace",
                index=False,
            )
        logger.info("Simulation grid complete; saved %d results", completed_backtests)

    run_simulations = callback("run_simulations", run_simulations)

    with sqlite3.connect(SIMULATION_RESULTS_DATABASE) as connection:
        existing_tables = {
            row[0]
            for row in connection.execute(
                "\n            SELECT name\n            FROM sqlite_master\n            WHERE type = 'table'\n            "
            ).fetchall()
        }
    if {"Stock Simulation Results", "Market Simulation Results"}.issubset(existing_tables):
        logger.info("Using existing simulation-result tables")
    else:
        logger.info("Simulation-result tables not found")
        run_simulations()
    with sqlite3.connect(SIMULATION_RESULTS_DATABASE) as connection:
        market_simulations_results = pd.read_sql_query(
            'SELECT * FROM "Market Simulation Results"', connection
        )
        strategy_simulations_results = pd.read_sql_query(
            'SELECT * FROM "Stock Simulation Results"', connection
        )
    from equity_selector.simulations import align_simulation_results

    strategy_simulations_results, market_simulations_results = align_simulation_results(
        strategy_simulations_results, market_simulations_results
    )
    final_evaluation_storage = {}
    final_evaluation_stock_removals = {}

    def save_final_evaluation_values(simulation_id, **values):
        final_evaluation_storage.setdefault(simulation_id, {}).update(values)

    save_final_evaluation_values = callback(
        "save_final_evaluation_values", save_final_evaluation_values
    )

    logger.info("Loaded %d strategy simulations for filtering", len(strategy_simulations_results))
    backtest_standard_deviation = market_simulations_results["Backtest Quality"].std()

    def stop_if_no_simulations(dataframe, filter_name):
        if dataframe.empty:
            logger.info("No simulations remain after %s", filter_name)
            sys.exit(0)

    stop_if_no_simulations = callback("stop_if_no_simulations", stop_if_no_simulations)

    logger.info("Starting performance and portfolio-type filters")
    rows_before_return_filter = len(strategy_simulations_results)
    strategy_simulations_results = strategy_simulations_results[
        strategy_simulations_results["Strategy Return"]
        > market_simulations_results["Strategy Return"]
    ].copy()
    rows_after_return_filter = len(strategy_simulations_results)
    stop_if_no_simulations(strategy_simulations_results, "market return filter")
    PORTFOLIO_TYPE_REJECTION_THRESHOLDS = get_setting(
        "PORTFOLIO_TYPE_REJECTION_THRESHOLDS",
        {
            "Backtest Quality": 1.25,
            "ALPHA": 1.75,
            "RELATIVE_ALPHA": 1.75,
            "RISK_ADJUSTED_ALPHA": 1.75,
            "CROSS_SECTION_ALPHA": 1.75,
            "DIRECTION": 1.75,
            "DIRECTION_MULTICLASS": 2.0,
            "ALPHA_BINARY": 2.0,
            "BARRIER_ALPHA": 2.0,
            "VOLATILITY": 2.0,
            "DOWNSIDE_VOLATILITY": 1.75,
            "VOLATILITY_ASYMMETRY": 2.0,
            "DOWNSIDE": 1.75,
            "TAIL_RISK": 1.75,
            "TAIL_EVENT": 2.0,
            "DOWNSIDE_EXCURSION": 2.0,
            "VOLATILITY_EVENT": 2.0,
            "CROSS_SECTION_DOWNSIDE": 2.0,
            "ABSOLUTE_MOVE": 2.25,
            "UPSIDE_VOLATILITY": 2.25,
            "UPSIDE_EVENT": 2.25,
            "UPSIDE_EXCURSION": 2.25,
            "RECOVERY": 2.25,
            "REVERSAL": 2.25,
            "TIME_TO_DOWNSIDE_EXCURSION": 2.25,
            "TIME_TO_UPSIDE_EXCURSION": 2.5,
            "EXECUTION": 2.5,
            "LIQUIDITY": 2.25,
            "MARKET_IMPACT": 2.5,
            "CORRELATION": 2.5,
            "COVARIANCE": 2.5,
            "REGIME": 2.25,
        },
    )
    type_score_columns = [
        column[13:]
        for column in market_simulations_results.columns
        if column.startswith("Type Score | ")
    ]
    for column in type_score_columns:
        standard_deviation = market_simulations_results[f"Type Score | {column}"].std()
        strategy_simulations_results[f"STD from Benchmark {column}"] = (
            market_simulations_results[f"Type Score | {column}"]
            - strategy_simulations_results[f"Type Score | {column}"]
        ) / standard_deviation
        strategy_simulations_results[f"Type {column} Threshold"] = (
            PORTFOLIO_TYPE_REJECTION_THRESHOLDS[column]
        )
    PRIMARY_REJECTION_TYPES = get_setting(
        "PRIMARY_REJECTION_TYPES",
        {
            "ALPHA",
            "RELATIVE_ALPHA",
            "RISK_ADJUSTED_ALPHA",
            "CROSS_SECTION_ALPHA",
            "DIRECTION",
            "DIRECTION_MULTICLASS",
            "ALPHA_BINARY",
            "DOWNSIDE_VOLATILITY",
            "DOWNSIDE",
            "TAIL_RISK",
        },
    )
    SECONDARY_REJECTION_TYPES = get_setting(
        "SECONDARY_REJECTION_TYPES",
        {
            "BARRIER_ALPHA",
            "VOLATILITY",
            "VOLATILITY_ASYMMETRY",
            "TAIL_EVENT",
            "DOWNSIDE_EXCURSION",
            "VOLATILITY_EVENT",
            "CROSS_SECTION_DOWNSIDE",
            "REGIME",
        },
    )
    TERTIARY_REJECTION_TYPES = get_setting(
        "TERTIARY_REJECTION_TYPES",
        {
            "ABSOLUTE_MOVE",
            "UPSIDE_VOLATILITY",
            "UPSIDE_EVENT",
            "UPSIDE_EXCURSION",
            "RECOVERY",
            "REVERSAL",
            "TIME_TO_DOWNSIDE_EXCURSION",
            "TIME_TO_UPSIDE_EXCURSION",
            "EXECUTION",
            "LIQUIDITY",
            "MARKET_IMPACT",
            "CORRELATION",
            "COVARIANCE",
        },
    )

    def type_score_rejection(strategy_simulations_results, rejection_types):
        available_rejection_types = [
            column for column in rejection_types if column in type_score_columns
        ]
        for column in available_rejection_types:
            rows_before = len(strategy_simulations_results)
            standard_deviation = market_simulations_results[f"Type Score | {column}"].std()
            strategy_simulations_results = strategy_simulations_results[
                strategy_simulations_results[f"STD from Benchmark {column}"]
                < strategy_simulations_results[f"Type {column} Threshold"]
            ].copy()
            rows_after = len(strategy_simulations_results)
            stop_if_no_simulations(strategy_simulations_results, f"{column} type-score filter")
        return strategy_simulations_results

    type_score_rejection = callback("type_score_rejection", type_score_rejection)

    strategy_simulations_results = type_score_rejection(
        strategy_simulations_results, PRIMARY_REJECTION_TYPES
    )
    rows_before = len(strategy_simulations_results)
    backtest_standard_deviation = market_simulations_results["Backtest Quality"].std()
    strategy_simulations_results["Backtest Quality STD from Benchmark"] = (
        market_simulations_results["Backtest Quality"]
        - strategy_simulations_results["Backtest Quality"]
    ) / backtest_standard_deviation
    strategy_simulations_results["Backtest Threshold"] = PORTFOLIO_TYPE_REJECTION_THRESHOLDS[
        "Backtest Quality"
    ]
    strategy_simulations_results = strategy_simulations_results[
        strategy_simulations_results["Backtest Quality STD from Benchmark"]
        < strategy_simulations_results["Backtest Threshold"]
    ].copy()
    rows_after = len(strategy_simulations_results)
    stop_if_no_simulations(strategy_simulations_results, "Backtest Quality filter")
    simulations_to_remove = []
    for simulation in strategy_simulations_results.to_dict(orient="records"):
        current_stocks = add_horizon_scores(
            dataframe=stocks,
            horizon_score_configuration=horizon_score_configurations[
                simulation["Horizon Score Index"]
            ],
        )
        for config in PORTFOLIO_GROUP_CONFIGURATIONS:
            if config["Name"] == simulation["Type Configuration"]:
                type_scores = config
                break
        current_stocks = add_type_scores(
            dataframe=current_stocks, type_score_configuration=type_scores
        )
        current_stocks["Contribution"] = (
            current_stocks["Horizon Score"]
            * current_stocks["Signal"]
            * current_stocks["Type Score"]
        )
        rebalanced_current_stocks = apply_horizon_signal_refresh(
            predictions_df=current_stocks, rebalance_multiplier=simulation["Rebalance Multiplier"]
        )
        score_stocks = build_score_stocks_with_direction(rebalanced_current_stocks)
        stocks_results = portfolio_returns_from_scores(
            score_stocks,
            max_weight=simulation["Max Weight"],
            concentration_penalty=simulation["Concentration Penalty"],
        )[["Date", "Return"]]
        stocks_results["5 Day Rolling Return"] = (1.0 + stocks_results["Return"]).rolling(
            window=5, min_periods=5
        ).apply(np.prod, raw=True) - 1.0
        stocks_results["21 Day Rolling Return"] = (1.0 + stocks_results["Return"]).rolling(
            window=21, min_periods=21
        ).apply(np.prod, raw=True) - 1.0
        stocks_results["252 Day Rolling Return"] = (1.0 + stocks_results["Return"]).rolling(
            window=252, min_periods=252
        ).apply(np.prod, raw=True) - 1.0
        stocks_results_without_best_day = stocks_results.copy()
        best_day = stocks_results["Return"].idxmax()
        best_day_position = stocks_results.index.get_loc(best_day)
        stocks_results_without_best_day.loc[
            stocks_results_without_best_day.index[best_day_position], "Return"
        ] = 0.0
        stocks_results_without_best_5_days = stocks_results.copy()
        best_5_day_end = stocks_results["5 Day Rolling Return"].idxmax()
        best_5_day_end_position = stocks_results.index.get_loc(best_5_day_end)
        best_5_day_start_position = best_5_day_end_position - 5 + 1
        stocks_results_without_best_5_days.loc[
            stocks_results_without_best_5_days.index[
                best_5_day_start_position : best_5_day_end_position + 1
            ],
            "Return",
        ] = 0.0
        stocks_results_without_best_21_days = stocks_results.copy()
        best_21_day_end = stocks_results["21 Day Rolling Return"].idxmax()
        best_21_day_end_position = stocks_results.index.get_loc(best_21_day_end)
        best_21_day_start_position = best_21_day_end_position - 21 + 1
        stocks_results_without_best_21_days.loc[
            stocks_results_without_best_21_days.index[
                best_21_day_start_position : best_21_day_end_position + 1
            ],
            "Return",
        ] = 0.0
        stocks_results_without_best_252_days = stocks_results.copy()
        best_252_day_end = stocks_results["252 Day Rolling Return"].idxmax()
        best_252_day_end_position = stocks_results.index.get_loc(best_252_day_end)
        best_252_day_start_position = best_252_day_end_position - 252 + 1
        stocks_results_without_best_252_days.loc[
            stocks_results_without_best_252_days.index[
                best_252_day_start_position : best_252_day_end_position + 1
            ],
            "Return",
        ] = 0.0
        removed_results = {}
        removed_results["1"] = stocks_results_without_best_day
        removed_results["5"] = stocks_results_without_best_5_days
        removed_results["21"] = stocks_results_without_best_21_days
        removed_results["252"] = stocks_results_without_best_252_days
        number_of_backtest_rows = len(stocks_results)
        number_of_backtest_rows = len(stocks_results)
        BASE_BQ_THRESHOLD = 1.25
        MINIMUM_STRESSED_BQ_THRESHOLD = 1.5
        MAXIMUM_STRESSED_BQ_THRESHOLD = 2.5
        MINIMUM_REMOVED_FRACTION = 0.002
        THRESHOLD_INCREASE_MULTIPLIER = 4.0
        rolling_periods = [1, 5, 21, 252]
        rolling_test_settings = {}
        for rolling_period in rolling_periods:
            removed_fraction = rolling_period / number_of_backtest_rows
            stressed_bq_threshold = np.clip(
                BASE_BQ_THRESHOLD + THRESHOLD_INCREASE_MULTIPLIER * removed_fraction,
                MINIMUM_STRESSED_BQ_THRESHOLD,
                MAXIMUM_STRESSED_BQ_THRESHOLD,
            )
            if removed_fraction >= MINIMUM_REMOVED_FRACTION:
                (
                    strategy_return,
                    strategy_sharpe,
                    strategy_average_drawdown,
                    strategy_max_drawdown,
                    strategy_relative_return,
                    strategy_relative_sharpe,
                    strategy_relative_max_drawdown,
                    strategy_relative_average_drawdown,
                    strategy_quality,
                ) = result_metrics(removed_results[str(rolling_period)])
                simulation_id = simulation["Simulation ID"]
                market_backtest_quality = market_simulations_results.loc[
                    market_simulations_results["Simulation ID"].eq(simulation_id),
                    "Backtest Quality",
                ].iloc[0]
                deviations = (
                    market_backtest_quality - strategy_quality
                ) / backtest_standard_deviation
                if deviations >= stressed_bq_threshold:
                    simulations_to_remove.append(simulation["Simulation ID"])
            if removed_fraction < MINIMUM_REMOVED_FRACTION:
                final_evaluation_removed_quality = result_metrics(
                    removed_results[str(rolling_period)].copy()
                )[-1]
            else:
                final_evaluation_removed_quality = strategy_quality
            final_evaluation_period_name = {1: "Day", 5: "Week", 21: "Month", 252: "Year"}[
                rolling_period
            ]
            save_final_evaluation_values(
                simulation["Simulation ID"],
                **{
                    f"Best {final_evaluation_period_name} Removed Quality": final_evaluation_removed_quality
                },
            )
    strategy_simulations_results = strategy_simulations_results[
        ~strategy_simulations_results["Simulation ID"].isin(simulations_to_remove)
    ].copy()
    stop_if_no_simulations(strategy_simulations_results, "rolling-period robustness test")
    tickers = stocks["Ticker"].unique()
    number_of_stocks = len(tickers)
    stock_removal_threshold = np.clip(np.sqrt(2.0 * np.log(max(number_of_stocks, 2))), 2.0, 3.5)
    simulations_to_remove = []
    for simulation in strategy_simulations_results.to_dict(orient="records"):
        current_stocks = add_horizon_scores(
            dataframe=stocks,
            horizon_score_configuration=horizon_score_configurations[
                simulation["Horizon Score Index"]
            ],
        )
        for config in PORTFOLIO_GROUP_CONFIGURATIONS:
            if config["Name"] == simulation["Type Configuration"]:
                type_scores = config
                break
        current_stocks = add_type_scores(
            dataframe=current_stocks, type_score_configuration=type_scores
        )
        current_stocks["Contribution"] = (
            current_stocks["Horizon Score"]
            * current_stocks["Signal"]
            * current_stocks["Type Score"]
        )
        rebalanced_current_stocks = apply_horizon_signal_refresh(
            predictions_df=current_stocks, rebalance_multiplier=simulation["Rebalance Multiplier"]
        )
        score_stocks = build_score_stocks_with_direction(rebalanced_current_stocks)
        for removed_ticker in tickers:
            removed_score_stocks = score_stocks[score_stocks["Ticker"].ne(removed_ticker)].copy()
            stocks_results = portfolio_returns_from_scores(
                removed_score_stocks,
                max_weight=simulation["Max Weight"],
                concentration_penalty=simulation["Concentration Penalty"],
            )[["Date", "Return"]]
            (
                strategy_return,
                strategy_sharpe,
                strategy_average_drawdown,
                strategy_max_drawdown,
                strategy_relative_return,
                strategy_relative_sharpe,
                strategy_relative_max_drawdown,
                strategy_relative_average_drawdown,
                strategy_quality,
            ) = result_metrics(stocks_results)
            simulation_id = simulation["Simulation ID"]
            market_backtest_quality = market_simulations_results.loc[
                market_simulations_results["Simulation ID"].eq(simulation_id), "Backtest Quality"
            ].iloc[0]
            deviations = (market_backtest_quality - strategy_quality) / backtest_standard_deviation
            if deviations >= stock_removal_threshold:
                simulations_to_remove.append(simulation["Simulation ID"])
            final_evaluation_stock_removals.setdefault(simulation_id, []).append(
                {"Ticker": removed_ticker, "Backtest Quality": strategy_quality}
            )
    strategy_simulations_results = strategy_simulations_results[
        ~strategy_simulations_results["Simulation ID"].isin(simulations_to_remove)
    ].copy()
    stop_if_no_simulations(strategy_simulations_results, "stock-removal robustness test")
    logger.info(
        "Starting portfolio-group robustness tests for %d simulations",
        len(strategy_simulations_results),
    )
    portfolio_group_robustness_threshold = get_setting("portfolio_group_robustness_threshold", 1.75)
    simulations_to_remove = []
    group_neighbourhood_scores = {}
    configuration_keys = ["Ranking", "Direction", "Risk", "Opportunity", "Special"]

    def configuration_distance(config_a, config_b):
        return np.sqrt(sum(((config_a[key] - config_b[key]) ** 2 for key in configuration_keys)))

    configuration_distance = callback("configuration_distance", configuration_distance)

    def get_two_closest_configurations(configuration_name):
        current_configuration = next(
            (
                config
                for config in PORTFOLIO_GROUP_CONFIGURATIONS
                if config["Name"] == configuration_name
            )
        )
        distances = []
        for candidate_configuration in PORTFOLIO_GROUP_CONFIGURATIONS:
            if candidate_configuration["Name"] == configuration_name:
                continue
            distance = configuration_distance(current_configuration, candidate_configuration)
            distances.append((distance, candidate_configuration))
        distances.sort(key=lambda x: x[0])
        return [distances[0][1], distances[1][1]]

    get_two_closest_configurations = callback(
        "get_two_closest_configurations", get_two_closest_configurations
    )

    simulation_standard_deviation = strategy_simulations_results["Backtest Quality"].std()
    for simulation in strategy_simulations_results.to_dict(orient="records"):
        simulation_id = simulation["Simulation ID"]
        original_backtest_quality = strategy_simulations_results.loc[
            strategy_simulations_results["Simulation ID"].eq(simulation_id), "Backtest Quality"
        ].iloc[0]
        current_stocks = add_horizon_scores(
            dataframe=stocks,
            horizon_score_configuration=horizon_score_configurations[
                simulation["Horizon Score Index"]
            ],
        )
        neighbour_configurations = get_two_closest_configurations(simulation["Type Configuration"])
        neighbour_qualities = []
        for neighbour_configuration in neighbour_configurations:
            neighbour_stocks = current_stocks.copy()
            neighbour_stocks = add_type_scores(
                dataframe=neighbour_stocks, type_score_configuration=neighbour_configuration
            )
            neighbour_stocks["Contribution"] = (
                neighbour_stocks["Horizon Score"]
                * neighbour_stocks["Signal"]
                * neighbour_stocks["Type Score"]
            )
            rebalanced_neighbour_stocks = apply_horizon_signal_refresh(
                predictions_df=neighbour_stocks,
                rebalance_multiplier=simulation["Rebalance Multiplier"],
            )
            score_stocks = build_score_stocks_with_direction(rebalanced_neighbour_stocks)
            stocks_results = portfolio_returns_from_scores(
                score_stocks,
                max_weight=simulation["Max Weight"],
                concentration_penalty=simulation["Concentration Penalty"],
            )[["Date", "Return"]]
            (
                strategy_return,
                strategy_sharpe,
                strategy_average_drawdown,
                strategy_max_drawdown,
                strategy_relative_return,
                strategy_relative_sharpe,
                strategy_relative_max_drawdown,
                strategy_relative_average_drawdown,
                strategy_quality,
            ) = result_metrics(stocks_results)
            neighbour_qualities.append(strategy_quality)
        neighbour_qualities = np.asarray(neighbour_qualities, dtype=float)
        group_deviations = (
            original_backtest_quality - neighbour_qualities
        ) / simulation_standard_deviation
        group_score_deviations = np.maximum(group_deviations, 0.0)
        group_neighbourhood_score = np.mean(group_score_deviations)
        group_neighbourhood_scores[simulation_id] = group_neighbourhood_score
        worst_group_deviation = np.max(group_deviations)
        if worst_group_deviation >= portfolio_group_robustness_threshold:
            simulations_to_remove.append(simulation_id)
    strategy_simulations_results = strategy_simulations_results[
        ~strategy_simulations_results["Simulation ID"].isin(simulations_to_remove)
    ].copy()
    stop_if_no_simulations(strategy_simulations_results, "portfolio-group robustness test")
    logger.info(
        "Starting settings robustness tests for %d simulations", len(strategy_simulations_results)
    )
    number_of_iterations = get_setting("number_of_iterations", 30)
    settings_robustness_threshold = get_setting("settings_robustness_threshold", 2.75)
    random_neighbourhood_scalar = (
        portfolio_group_robustness_threshold / settings_robustness_threshold
    )
    random_generator = np.random.default_rng(42)
    simulations_to_remove = []
    random_neighbourhood_scores = {}
    simulation_standard_deviation = strategy_simulations_results["Backtest Quality"].std()
    for simulation in strategy_simulations_results.to_dict(orient="records"):
        simulation_id = simulation["Simulation ID"]
        original_strategy_quality = strategy_simulations_results.loc[
            strategy_simulations_results["Simulation ID"].eq(simulation_id), "Backtest Quality"
        ].iloc[0]
        current_stocks = add_horizon_scores(
            dataframe=stocks,
            horizon_score_configuration=horizon_score_configurations[
                simulation["Horizon Score Index"]
            ],
        )
        for config in PORTFOLIO_GROUP_CONFIGURATIONS:
            if config["Name"] == simulation["Type Configuration"]:
                type_scores = config
                break
        current_stocks = add_type_scores(
            dataframe=current_stocks, type_score_configuration=type_scores
        )
        current_stocks["Contribution"] = (
            current_stocks["Horizon Score"]
            * current_stocks["Signal"]
            * current_stocks["Type Score"]
        )
        perturbed_strategy_qualities = []
        for iteration in range(number_of_iterations):
            perturbed_rebalance_multiplier = random_generator.uniform(
                max(0.0, simulation["Rebalance Multiplier"] - 0.05),
                min(1.0, simulation["Rebalance Multiplier"] + 0.05),
            )
            perturbed_concentration_penalty = random_generator.uniform(
                max(0.0, simulation["Concentration Penalty"] - 0.05),
                min(1.0, simulation["Concentration Penalty"] + 0.05),
            )
            perturbed_max_weight = random_generator.uniform(
                max(0.0, simulation["Max Weight"] - 0.05), min(1.0, simulation["Max Weight"] + 0.05)
            )
            perturbed_stocks = apply_horizon_signal_refresh(
                predictions_df=current_stocks.copy(),
                rebalance_multiplier=perturbed_rebalance_multiplier,
            )
            score_stocks = build_score_stocks_with_direction(perturbed_stocks)
            stocks_results = portfolio_returns_from_scores(
                score_stocks,
                max_weight=perturbed_max_weight,
                concentration_penalty=perturbed_concentration_penalty,
            )[["Date", "Return"]]
            (
                strategy_return,
                strategy_sharpe,
                strategy_average_drawdown,
                strategy_max_drawdown,
                strategy_relative_return,
                strategy_relative_sharpe,
                strategy_relative_max_drawdown,
                strategy_relative_average_drawdown,
                strategy_quality,
            ) = result_metrics(stocks_results)
            perturbed_strategy_qualities.append(strategy_quality)
        perturbed_strategy_qualities = np.asarray(perturbed_strategy_qualities, dtype=float)
        random_deviations = (
            original_strategy_quality - perturbed_strategy_qualities
        ) / simulation_standard_deviation
        random_score_deviations = np.maximum(random_deviations, 0.0)
        random_neighbourhood_score = np.mean(random_score_deviations) * random_neighbourhood_scalar
        random_neighbourhood_scores[simulation_id] = random_neighbourhood_score
        save_final_evaluation_values(
            simulation_id,
            **{
                "Neighbourhood Pass Rate": float(
                    np.mean(random_deviations < settings_robustness_threshold)
                )
                if np.isfinite(random_deviations).all()
                else np.nan
            },
        )
        worst_deviation = np.max(random_deviations)
        if worst_deviation >= settings_robustness_threshold:
            simulations_to_remove.append(simulation_id)
    strategy_simulations_results = strategy_simulations_results[
        ~strategy_simulations_results["Simulation ID"].isin(simulations_to_remove)
    ].copy()
    stop_if_no_simulations(strategy_simulations_results, "portfolio-settings robustness test")
    strategy_simulations_results["Neighbourhood Score"] = (
        strategy_simulations_results["Simulation ID"].map(group_neighbourhood_scores)
        + strategy_simulations_results["Simulation ID"].map(random_neighbourhood_scores)
    ) / 2
    logger.info(
        "Starting unseen-stock robustness tests for %d simulations",
        len(strategy_simulations_results),
    )
    unseen_stock_robustness_threshold = get_setting("unseen_stock_robustness_threshold", 1.5)
    simulations_to_remove = []
    with sqlite3.connect(BACKTEST_DATABASE) as connection:
        unseen_stocks = pd.read_sql_query('SELECT * FROM "Unseen"', connection)
    unseen_quality_scores = {}
    for simulation in strategy_simulations_results.to_dict(orient="records"):
        simulation_id = simulation["Simulation ID"]
        market_backtest_quality = market_simulations_results.loc[
            market_simulations_results["Simulation ID"].eq(simulation_id), "Backtest Quality"
        ].iloc[0]
        current_stocks = add_horizon_scores(
            dataframe=unseen_stocks.copy(),
            horizon_score_configuration=horizon_score_configurations[
                simulation["Horizon Score Index"]
            ],
        )
        for config in PORTFOLIO_GROUP_CONFIGURATIONS:
            if config["Name"] == simulation["Type Configuration"]:
                type_scores = config
                break
        current_stocks = add_type_scores(
            dataframe=current_stocks, type_score_configuration=type_scores
        )
        current_stocks["Contribution"] = (
            current_stocks["Horizon Score"]
            * current_stocks["Signal"]
            * current_stocks["Type Score"]
        )
        rebalanced_current_stocks = apply_horizon_signal_refresh(
            predictions_df=current_stocks, rebalance_multiplier=simulation["Rebalance Multiplier"]
        )
        score_stocks = build_score_stocks_with_direction(rebalanced_current_stocks)
        stocks_results = portfolio_returns_from_scores(
            score_stocks,
            max_weight=simulation["Max Weight"],
            concentration_penalty=simulation["Concentration Penalty"],
        )[["Date", "Return"]]
        (
            strategy_return,
            strategy_sharpe,
            strategy_average_drawdown,
            strategy_max_drawdown,
            strategy_relative_return,
            strategy_relative_sharpe,
            strategy_relative_max_drawdown,
            strategy_relative_average_drawdown,
            strategy_quality,
        ) = result_metrics(stocks_results)
        unseen_deviation = (
            market_backtest_quality - strategy_quality
        ) / backtest_standard_deviation
        if unseen_deviation >= unseen_stock_robustness_threshold:
            simulations_to_remove.append(simulation_id)
        unseen_quality_scores[simulation_id] = unseen_deviation
        save_final_evaluation_values(
            simulation_id,
            **{
                "Unseen Backtest Quality": strategy_quality,
                "Unseen Gate Passed": bool(unseen_deviation < unseen_stock_robustness_threshold)
                if np.isfinite(unseen_deviation)
                else None,
            },
        )
    strategy_simulations_results["Unseen Stock Score"] = strategy_simulations_results[
        "Simulation ID"
    ].map(unseen_quality_scores)
    strategy_simulations_results = strategy_simulations_results[
        ~strategy_simulations_results["Simulation ID"].isin(simulations_to_remove)
    ].copy()
    stop_if_no_simulations(strategy_simulations_results, "unseen-stock robustness test")
    lower_is_better = ["Neighbourhood Score", "Unseen Stock Score"]
    higher_is_better = ["Strategy Return", "Sharpe Ratio", "Max Drawdown", "Average Drawdown"]
    simulations_to_remove = []
    for simulation in strategy_simulations_results.to_dict(orient="records"):
        simulation_id = simulation["Simulation ID"]
        for competitor in strategy_simulations_results.to_dict(orient="records"):
            competitor_id = competitor["Simulation ID"]
            if competitor_id == simulation_id:
                continue
            no_worse = all(
                (competitor[column] <= simulation[column] for column in lower_is_better)
            ) and all((competitor[column] >= simulation[column] for column in higher_is_better))
            strictly_better = any(
                (competitor[column] < simulation[column] for column in lower_is_better)
            ) or any((competitor[column] > simulation[column] for column in higher_is_better))
            if no_worse and strictly_better:
                simulations_to_remove.append(simulation_id)
                break
    strategy_simulations_results = strategy_simulations_results[
        ~strategy_simulations_results["Simulation ID"].isin(simulations_to_remove)
    ].copy()
    stop_if_no_simulations(strategy_simulations_results, "Pareto-dominance filter")
    logger.info(
        "Starting pairwise similarity checks for %d simulations", len(strategy_simulations_results)
    )
    from itertools import combinations

    return_correlation_threshold = get_setting("return_correlation_threshold", 0.97)
    performance_percentage_threshold = get_setting("performance_percentage_threshold", 0.1)
    stock_overlap_threshold = get_setting("stock_overlap_threshold", 0.85)
    performance_metric_names = [
        "Strategy Return",
        "Sharpe Ratio",
        "Average Drawdown",
        "Max Drawdown",
    ]
    simulation_backtest_results = {}
    for simulation in strategy_simulations_results.to_dict(orient="records"):
        simulation_id = simulation["Simulation ID"]
        current_stocks = add_horizon_scores(
            dataframe=stocks.copy(),
            horizon_score_configuration=horizon_score_configurations[
                simulation["Horizon Score Index"]
            ],
        )
        for config in PORTFOLIO_GROUP_CONFIGURATIONS:
            if config["Name"] == simulation["Type Configuration"]:
                type_scores = config
                break
        current_stocks = add_type_scores(
            dataframe=current_stocks, type_score_configuration=type_scores
        )
        current_stocks["Contribution"] = (
            current_stocks["Horizon Score"]
            * current_stocks["Signal"]
            * current_stocks["Type Score"]
        )
        rebalanced_current_stocks = apply_horizon_signal_refresh(
            predictions_df=current_stocks, rebalance_multiplier=simulation["Rebalance Multiplier"]
        )
        score_stocks = build_score_stocks_with_direction(rebalanced_current_stocks)
        stocks_results = portfolio_returns_from_scores(
            score_stocks,
            max_weight=simulation["Max Weight"],
            concentration_penalty=simulation["Concentration Penalty"],
        )
        (
            strategy_return,
            strategy_sharpe,
            strategy_average_drawdown,
            strategy_max_drawdown,
            strategy_relative_return,
            strategy_relative_sharpe,
            strategy_relative_max_drawdown,
            strategy_relative_average_drawdown,
            strategy_quality,
        ) = result_metrics(stocks_results[["Date", "Return"]])
        ticker_columns = [
            column for column in stocks_results.columns if column not in {"Date", "Return"}
        ]
        used_stocks = (
            stocks_results.melt(
                id_vars=["Date"], value_vars=ticker_columns, var_name="Ticker", value_name="Weight"
            )
            .loc[lambda dataframe: dataframe["Weight"].ne(0), ["Date", "Ticker"]]
            .copy()
        )
        simulation_backtest_results[simulation_id] = {
            "Metrics": {
                "Strategy Return": strategy_return,
                "Sharpe Ratio": strategy_sharpe,
                "Average Drawdown": strategy_average_drawdown,
                "Max Drawdown": strategy_max_drawdown,
                "Strategy Quality": strategy_quality,
            },
            "Stocks Results": stocks_results.copy(),
            "Score Stocks": score_stocks.copy(),
            "Used Stocks": used_stocks.copy(),
        }

    def percentage_from_pair_median(value_a, value_b):
        median_value = np.median([value_a, value_b])
        difference = abs(value_a - value_b)
        denominator = abs(median_value)
        if denominator == 0:
            if difference == 0:
                return 0.0
            return np.inf
        return difference / denominator

    percentage_from_pair_median = callback(
        "percentage_from_pair_median", percentage_from_pair_median
    )

    def average_stock_overlap(stocks_a, stocks_b):
        stocks_by_date_a = stocks_a.groupby("Date")["Ticker"].apply(set).to_dict()
        stocks_by_date_b = stocks_b.groupby("Date")["Ticker"].apply(set).to_dict()
        dates = set(stocks_by_date_a.keys()) | set(stocks_by_date_b.keys())
        daily_overlaps = []
        for date in dates:
            tickers_a = stocks_by_date_a.get(date, set())
            tickers_b = stocks_by_date_b.get(date, set())
            intersection = tickers_a & tickers_b
            union = tickers_a | tickers_b
            if len(union) == 0:
                daily_overlap = 1.0
            else:
                daily_overlap = len(intersection) / len(union)
            daily_overlaps.append(daily_overlap)
        if not daily_overlaps:
            return np.nan
        return np.mean(daily_overlaps)

    average_stock_overlap = callback("average_stock_overlap", average_stock_overlap)

    simulation_pair_comparisons = []
    simulation_ids = list(simulation_backtest_results.keys())
    for simulation_id_a, simulation_id_b in combinations(simulation_ids, 2):
        results_a = simulation_backtest_results[simulation_id_a]
        results_b = simulation_backtest_results[simulation_id_b]
        returns_a = results_a["Stocks Results"][["Date", "Return"]].rename(
            columns={"Return": "Return A"}
        )
        returns_b = results_b["Stocks Results"][["Date", "Return"]].rename(
            columns={"Return": "Return B"}
        )
        matched_returns = returns_a.merge(returns_b, on="Date", how="inner").dropna()
        if len(matched_returns) >= 2:
            return_correlation = matched_returns["Return A"].corr(matched_returns["Return B"])
        else:
            return_correlation = np.nan
        performance_differences = {}
        for metric_name in performance_metric_names:
            value_a = results_a["Metrics"][metric_name]
            value_b = results_b["Metrics"][metric_name]
            percentage_difference = percentage_from_pair_median(value_a, value_b)
            performance_differences[metric_name] = percentage_difference
        stock_overlap = average_stock_overlap(results_a["Used Stocks"], results_b["Used Stocks"])
        performance_similar = all(
            (
                difference <= performance_percentage_threshold
                for difference in performance_differences.values()
            )
        )
        too_similar = (
            return_correlation >= return_correlation_threshold
            and performance_similar
            and (stock_overlap >= stock_overlap_threshold)
        )
        simulation_pair_comparisons.append(
            {
                "Simulation ID A": simulation_id_a,
                "Simulation ID B": simulation_id_b,
                "Return Correlation": return_correlation,
                "Strategy Return Difference": performance_differences["Strategy Return"],
                "Sharpe Ratio Difference": performance_differences["Sharpe Ratio"],
                "Average Drawdown Difference": performance_differences["Average Drawdown"],
                "Max Drawdown Difference": performance_differences["Max Drawdown"],
                "Stock Overlap": stock_overlap,
                "Too Similar": too_similar,
            }
        )
    simulation_pair_comparisons = pd.DataFrame(simulation_pair_comparisons)
    ranked_simulations = strategy_simulations_results.sort_values(
        by=["Backtest Quality", "Neighbourhood Score", "Unseen Stock Score"],
        ascending=[False, True, True],
    ).copy()
    similar_pairs = set()
    for comparison in simulation_pair_comparisons.to_dict(orient="records"):
        if not comparison["Too Similar"]:
            continue
        simulation_id_a = comparison["Simulation ID A"]
        simulation_id_b = comparison["Simulation ID B"]
        similar_pairs.add(frozenset([simulation_id_a, simulation_id_b]))
    kept_simulations = []
    simulations_to_remove = []
    for simulation in ranked_simulations.to_dict(orient="records"):
        simulation_id = simulation["Simulation ID"]
        is_redundant = False
        for kept_simulation_id in kept_simulations:
            pair = frozenset([simulation_id, kept_simulation_id])
            if pair in similar_pairs:
                is_redundant = True
                break
        if is_redundant:
            simulations_to_remove.append(simulation_id)
        else:
            kept_simulations.append(simulation_id)
    strategy_simulations_results = strategy_simulations_results[
        ~strategy_simulations_results["Simulation ID"].isin(simulations_to_remove)
    ].copy()
    stop_if_no_simulations(strategy_simulations_results, "strategy-similarity filter")
    if len(strategy_simulations_results) > 10:
        logger.info(
            "Applying secondary type-score filters to %d simulations",
            len(strategy_simulations_results),
        )
        strategy_simulations_results = type_score_rejection(
            strategy_simulations_results, SECONDARY_REJECTION_TYPES
        )
        stop_if_no_simulations(strategy_simulations_results, "secondary type-score filters")
    if len(strategy_simulations_results) > 10:
        logger.info(
            "Applying tertiary type-score filters to %d simulations",
            len(strategy_simulations_results),
        )
        strategy_simulations_results = type_score_rejection(
            strategy_simulations_results, TERTIARY_REJECTION_TYPES
        )
        stop_if_no_simulations(strategy_simulations_results, "tertiary type-score filters")
    if len(strategy_simulations_results) > 10:
        remaining_simulations = strategy_simulations_results.copy()
        selected_simulation_groups = []

        def select_and_remove(dataframe, column, number_to_select, ascending):
            selected = (
                dataframe.sort_values(by=column, ascending=ascending, na_position="last")
                .head(number_to_select)
                .copy()
            )
            remaining = dataframe[
                ~dataframe["Simulation ID"].isin(selected["Simulation ID"])
            ].copy()
            return (selected, remaining)

        selection_steps = [
            ("Backtest Quality", 4, False),
            ("Strategy Return", 2, False),
            ("Average Drawdown", 2, False),
            ("Unseen Stock Score", 2, True),
        ]
        for selection_column, number_to_select, ascending in selection_steps:
            selected_simulations, remaining_simulations = select_and_remove(
                dataframe=remaining_simulations,
                column=selection_column,
                number_to_select=number_to_select,
                ascending=ascending,
            )
            selected_simulation_groups.append(selected_simulations)
        strategy_simulations_results = pd.concat(selected_simulation_groups, ignore_index=True)
        logger.info(
            "Final strategy selection reduced the set to %d simulations",
            len(strategy_simulations_results),
        )
    logger.info("Pipeline complete: %d simulations retained", len(strategy_simulations_results))
    for final_evaluation_id in strategy_simulations_results["Simulation ID"]:
        final_evaluation_removals = pd.DataFrame(
            final_evaluation_stock_removals.get(final_evaluation_id, []),
            columns=["Ticker", "Backtest Quality"],
        )
        final_evaluation_valid = final_evaluation_removals["Backtest Quality"].notna()
        final_evaluation_worst = (
            final_evaluation_removals.loc[final_evaluation_removals["Backtest Quality"].idxmin()]
            if final_evaluation_valid.any()
            else None
        )
        save_final_evaluation_values(
            final_evaluation_id,
            **{
                "Mean Stock Removal Quality": final_evaluation_removals["Backtest Quality"].mean()
                if len(final_evaluation_removals) and final_evaluation_valid.all()
                else np.nan,
                "Worst Stock Removal Quality": final_evaluation_worst["Backtest Quality"]
                if final_evaluation_worst is not None
                else np.nan,
                "Worst Removed Ticker": final_evaluation_worst["Ticker"]
                if final_evaluation_worst is not None
                else None,
            },
        )
    final_evaluation_market_rows = market_simulations_results.set_index("Simulation ID")
    if not final_evaluation_market_rows.index.is_unique:
        raise ValueError("Market Simulation Results must have unique Simulation IDs.")
    final_evaluation_type_columns = [
        column
        for column in strategy_simulations_results.columns
        if column.startswith("Type Score | ") and column in market_simulations_results.columns
    ]
    final_evaluation_type_sd = market_simulations_results[final_evaluation_type_columns].std()
    for final_evaluation_row in strategy_simulations_results.to_dict(orient="records"):
        final_evaluation_id = final_evaluation_row["Simulation ID"]
        final_evaluation_market_row = final_evaluation_market_rows.loc[final_evaluation_id]
        final_evaluation_strategy_values = pd.Series(
            {column: final_evaluation_row[column] for column in final_evaluation_type_columns},
            dtype=float,
        )
        final_evaluation_market_values = final_evaluation_market_row[
            final_evaluation_type_columns
        ].astype(float)
        final_evaluation_common = np.isfinite(final_evaluation_strategy_values) & np.isfinite(
            final_evaluation_market_values
        )
        final_evaluation_relative_mask = (
            final_evaluation_common
            & np.isfinite(final_evaluation_type_sd)
            & final_evaluation_type_sd.gt(0)
        )
        final_evaluation_relative = (
            final_evaluation_strategy_values - final_evaluation_market_values
        ).loc[final_evaluation_relative_mask] / final_evaluation_type_sd.loc[
            final_evaluation_relative_mask
        ]
        save_final_evaluation_values(
            final_evaluation_id,
            **{
                "Portfolio Target Score": final_evaluation_strategy_values[
                    final_evaluation_common
                ].mean(),
                "Market Target Score": final_evaluation_market_values[
                    final_evaluation_common
                ].mean(),
                "Relative Target Score": final_evaluation_relative.mean(),
            },
        )
    final_evaluation_extra_columns = pd.DataFrame.from_dict(
        final_evaluation_storage, orient="index"
    )
    for final_evaluation_column in final_evaluation_extra_columns.columns:
        strategy_simulations_results[final_evaluation_column] = strategy_simulations_results[
            "Simulation ID"
        ].map(final_evaluation_extra_columns[final_evaluation_column])
    final_evaluation_passed_columns = [
        "Simulation ID",
        "Horizon Score Index",
        "Type Configuration",
        "Rebalance Multiplier",
        "Max Weight",
        "Concentration Penalty",
        "Strategy Return",
        "Sharpe Ratio",
        "Average Drawdown",
        "Max Drawdown",
        "Relative Return",
        "Relative Sharpe Ratio",
        "Relative Max Drawdown",
        "Relative Average Drawdown",
        "Backtest Quality",
        "Best Day Removed Quality",
        "Best Week Removed Quality",
        "Best Month Removed Quality",
        "Best Year Removed Quality",
        "Mean Stock Removal Quality",
        "Worst Stock Removal Quality",
        "Worst Removed Ticker",
        "Neighbourhood Score",
        "Neighbourhood Pass Rate",
        "Unseen Stock Score",
        "Unseen Backtest Quality",
        "Unseen Gate Passed",
        "Portfolio Target Score",
        "Market Target Score",
        "Relative Target Score",
    ]
    strategy_simulations_results = strategy_simulations_results.loc[
        :, final_evaluation_passed_columns
    ].copy()
    with sqlite3.connect(BACKTEST_DATABASE) as connection:
        write_frame(
            strategy_simulations_results,
            "Passed Strategies",
            connection,
            if_exists="replace",
            index=False,
        )

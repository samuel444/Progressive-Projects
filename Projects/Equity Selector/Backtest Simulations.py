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

from main_package import *


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
    "/Users/sam/Progressive-Projects/Projects/"
    "Equity Selector/data/"
)

BACKTEST_DATABASE = (
    DATA_DIR
    / "Backtest_Database.db"
)

HORIZON_SCORES_FILE = (
    DATA_DIR
    / "Top_Horizon_Scores.txt"
)

SIMULATION_RESULTS_DATABASE = (
    DATA_DIR
    / "Portfolio_Simulation_Results.db"
)


########################################
# Load Market And Stock Dataframes
########################################

with sqlite3.connect(
    BACKTEST_DATABASE
) as connection:

    market = pd.read_sql_query(
        'SELECT * FROM "Market"',
        connection,
    )

    stocks = pd.read_sql_query(
        'SELECT * FROM "Stocks"',
        connection,
    )


market["Date"] = pd.to_datetime(
    market["Date"]
)

stocks["Date"] = pd.to_datetime(
    stocks["Date"]
)


logger.info(
    "Backtest data loaded | market rows=%d | stock rows=%d | stock tickers=%d",
    len(market),
    len(stocks),
    stocks["Ticker"].nunique(),
)


########################################
# Load Horizon Score Configurations
########################################

with open(
    HORIZON_SCORES_FILE,
    "r",
) as file:

    horizon_score_configurations = (
        ast.literal_eval(
            file.read()
        )
    )


if not isinstance(
    horizon_score_configurations,
    list,
):

    raise TypeError(
        "Top_Horizon_Scores.txt must contain "
        "a list of dictionaries."
    )


if not all(
    isinstance(
        configuration,
        dict,
    )
    for configuration
    in horizon_score_configurations
):

    raise TypeError(
        "Every Horizon Score configuration "
        "must be a dictionary."
    )


logger.info(
    "Horizon configurations loaded | configurations=%d | file=%s",
    len(horizon_score_configurations),
    HORIZON_SCORES_FILE,
)

    
REBALANCE_MULTIPLIERS = [
    0.25,
    0.50,
    0.75,
    1.00,
]

CONCENTRATION_PENALTIES = [
    0.00,
    0.05,
    0.10,
    0.20,
    0.30,
]

MAX_WEIGHTS = [
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
]

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

PORTFOLIO_GROUP_CONFIGURATIONS = [

    ####################################
    # Reference Configurations
    ####################################

    {
        "Name": "Balanced",
        "Ranking": 0.30,
        "Direction": 0.25,
        "Risk": 0.25,
        "Opportunity": 0.15,
        "Special": 0.05,
    },

    {
        "Name": "Equal Weight",
        "Ranking": 0.20,
        "Direction": 0.20,
        "Risk": 0.20,
        "Opportunity": 0.20,
        "Special": 0.20,
    },

    {
        "Name": "Core Balanced",
        "Ranking": 0.35,
        "Direction": 0.30,
        "Risk": 0.25,
        "Opportunity": 0.10,
        "Special": 0.00,
    },


    ####################################
    # Ranking Focused
    ####################################

    {
        "Name": "Ranking Heavy",
        "Ranking": 0.50,
        "Direction": 0.20,
        "Risk": 0.20,
        "Opportunity": 0.10,
        "Special": 0.00,
    },

    {
        "Name": "Ranking And Risk",
        "Ranking": 0.45,
        "Direction": 0.15,
        "Risk": 0.30,
        "Opportunity": 0.10,
        "Special": 0.00,
    },

    {
        "Name": "Ranking And Direction",
        "Ranking": 0.45,
        "Direction": 0.30,
        "Risk": 0.15,
        "Opportunity": 0.10,
        "Special": 0.00,
    },


    ####################################
    # Direction Focused
    ####################################

    {
        "Name": "Direction Heavy",
        "Ranking": 0.25,
        "Direction": 0.45,
        "Risk": 0.20,
        "Opportunity": 0.10,
        "Special": 0.00,
    },

    {
        "Name": "Direction And Risk",
        "Ranking": 0.25,
        "Direction": 0.40,
        "Risk": 0.30,
        "Opportunity": 0.05,
        "Special": 0.00,
    },


    ####################################
    # Risk Focused
    ####################################

    {
        "Name": "Risk Heavy",
        "Ranking": 0.25,
        "Direction": 0.20,
        "Risk": 0.45,
        "Opportunity": 0.10,
        "Special": 0.00,
    },

    {
        "Name": "Conservative",
        "Ranking": 0.30,
        "Direction": 0.15,
        "Risk": 0.45,
        "Opportunity": 0.05,
        "Special": 0.05,
    },


    ####################################
    # Opportunity Focused
    ####################################

    {
        "Name": "Opportunity Heavy",
        "Ranking": 0.25,
        "Direction": 0.20,
        "Risk": 0.20,
        "Opportunity": 0.35,
        "Special": 0.00,
    },


    ####################################
    # Special Information
    ####################################

    {
        "Name": "Ranking And Special",
        "Ranking": 0.40,
        "Direction": 0.15,
        "Risk": 0.20,
        "Opportunity": 0.10,
        "Special": 0.15,
    },
]


total_backtests = (
    len(horizon_score_configurations)
    * len(PORTFOLIO_GROUP_CONFIGURATIONS)
    * len(REBALANCE_MULTIPLIERS)
    * len(MAX_WEIGHTS)
    * len(CONCENTRATION_PENALTIES)
)

logger.info(
    "Grid ready | horizon=%d | type=%d | rebalance=%d | max weights=%d | "
    "penalties=%d | total backtests=%d",
    len(horizon_score_configurations),
    len(PORTFOLIO_GROUP_CONFIGURATIONS),
    len(REBALANCE_MULTIPLIERS),
    len(MAX_WEIGHTS),
    len(CONCENTRATION_PENALTIES),
    total_backtests,
)

def apply_horizon_signal_refresh(
    predictions_df,
    rebalance_multiplier,
):

    if not (
        0 < rebalance_multiplier <= 1
    ):

        raise ValueError(
            "rebalance_multiplier must be "
            "greater than 0 and no greater than 1."
        )


    required_columns = {
        "Date",
        "Ticker",
        "Portfolio Target Type",
        "Horizon Key",
        "Contribution",
    }

    missing_columns = (
        required_columns
        -
        set(predictions_df.columns)
    )

    if missing_columns:

        raise ValueError(
            "Missing required columns: "
            f"{sorted(missing_columns)}"
        )


    refreshed = (
        predictions_df
        .copy()
        .sort_values(
            [
                "Ticker",
                "Portfolio Target Type",
                "Horizon Key",
                "Date",
            ]
        )
        .reset_index(drop=True)
    )


    group_columns = [
        "Ticker",
        "Portfolio Target Type",
        "Horizon Key",
    ]


    for group_values, group_indexes in refreshed.groupby(
        group_columns,
        sort=False,
    ).groups.items():

        ticker, portfolio_type, horizon_key = (
            group_values
        )

        horizon_key = str(
            horizon_key
        ).strip().lower()


        ########################################
        # Horizon Key Validation
        #
        # Expected examples:
        # 1d, 5d, 20d, 60d
        ########################################

        if not horizon_key.endswith("d"):

            raise ValueError(
                "Daily Horizon Key must end in 'd'. "
                f"Received {horizon_key!r} for "
                f"{ticker!r} / {portfolio_type!r}."
            )


        try:

            horizon_days = int(
                horizon_key[:-1]
            )

        except ValueError as error:

            raise ValueError(
                "Could not extract the number of days "
                f"from Horizon Key {horizon_key!r}."
            ) from error


        ########################################
        # Number of Rows Before Refresh
        ########################################

        refresh_rows = max(
            1,
            int(
                np.ceil(
                    rebalance_multiplier
                    *
                    horizon_days
                )
            ),
        )


        ########################################
        # Hold Each Refresh Signal
        ########################################

        group_indexes = np.asarray(
            list(group_indexes)
        )

        original_signals = (
            refreshed.loc[
                group_indexes,
                "Contribution",
            ]
            .to_numpy()
        )

        row_positions = np.arange(
            len(original_signals)
        )

        refresh_start_positions = (
            row_positions
            //
            refresh_rows
        ) * refresh_rows

        refreshed.loc[
            group_indexes,
            "Contribution",
        ] = original_signals[
            refresh_start_positions
        ]


    return (
        refreshed
        .sort_values(
            [
                "Date",
                "Ticker",
                "Portfolio Target Type",
                "Horizon Key",
            ]
        )
        .reset_index(drop=True)
    )

def add_horizon_scores(
    dataframe,
    horizon_score_configuration,
):
    """
    Add the Horizon Score column for one horizon-score configuration.
    """

    horizon_score_rows = []

    for portfolio_type, horizon_values in (
        horizon_score_configuration.items()
    ):

        for horizon_key, horizon_score in (
            horizon_values.items()
        ):

            horizon_score_rows.append(
                {
                    "Portfolio Target Type":
                        portfolio_type,

                    "Horizon Key":
                        horizon_key,

                    "Horizon Score":
                        float(
                            horizon_score
                        ),
                }
            )

    horizon_score_df = pd.DataFrame(
        horizon_score_rows
    )

    result = (
        dataframe
        .drop(
            columns=[
                "Horizon Score",
            ],
            errors="ignore",
        )
        .merge(
            horizon_score_df,
            on=[
                "Portfolio Target Type",
                "Horizon Key",
            ],
            how="left",
            validate="many_to_one",
        )
    )

    logger.info(
        "Horizon scores added | rows=%d | missing=%d",
        len(result),
        int(
            result[
                "Horizon Score"
            ]
            .isna()
            .sum()
        ),
    )

    return result


def add_type_scores(
    dataframe,
    type_score_configuration,
):
    """
    Add the Type Score column for one portfolio-group configuration.
    """

    portfolio_type_score_map = {}

    for portfolio_type in (
        PORTFOLIO_RANKING_TYPES
    ):

        portfolio_type_score_map[
            portfolio_type
        ] = type_score_configuration[
            "Ranking"
        ]

    for portfolio_type in (
        PORTFOLIO_DIRECTION_TYPES
    ):

        portfolio_type_score_map[
            portfolio_type
        ] = type_score_configuration[
            "Direction"
        ]

    for portfolio_type in (
        PORTFOLIO_RISK_TYPES
    ):

        portfolio_type_score_map[
            portfolio_type
        ] = type_score_configuration[
            "Risk"
        ]

    for portfolio_type in (
        PORTFOLIO_OPPORTUNITY_TYPES
    ):

        portfolio_type_score_map[
            portfolio_type
        ] = type_score_configuration[
            "Opportunity"
        ]

    for portfolio_type in (
        PORTFOLIO_SPECIAL_TYPES
    ):

        portfolio_type_score_map[
            portfolio_type
        ] = type_score_configuration[
            "Special"
        ]

    result = dataframe.copy()

    result["Type Score"] = (
        result[
            "Portfolio Target Type"
        ]
        .map(
            portfolio_type_score_map
        )
    )

    logger.info(
        "Type scores added | configuration=%s | rows=%d | missing=%d",
        type_score_configuration.get(
            "Name",
            "Unnamed",
        ),
        len(result),
        int(
            result[
                "Type Score"
            ]
            .isna()
            .sum()
        ),
    )

    return result

def extract_type_score_columns(
    type_score_dataframe,
):
    """
    Return one result column for every portfolio target
    type. Supports the type being either in the index or
    in a Portfolio Target Type column.
    """

    if (
        "Portfolio Target Type"
        in type_score_dataframe.columns
    ):

        type_scores_series = (
            type_score_dataframe
            .set_index(
                "Portfolio Target Type"
            )[
                "Type_Score"
            ]
        )

    else:

        type_scores_series = (
            type_score_dataframe[
                "Type_Score"
            ]
        )

    return {
        f"Type Score | {portfolio_type}": (
            float(type_score)
            if pd.notna(type_score)
            else np.nan
        )
        for portfolio_type, type_score
        in type_scores_series.items()
    }

def run_simulations():

    ########################################
    # Simulation Result Storage
    ########################################

    stock_simulation_records = []
    market_simulation_records = []

    completed_backtests = 0
    horizon_configuration_number = 0

    for scores in horizon_score_configurations:

        current_stocks = add_horizon_scores(
            dataframe=stocks,
            horizon_score_configuration=scores,
        )

        current_market = add_horizon_scores(
            dataframe=market,
            horizon_score_configuration=scores,
        )

        for type_scores in PORTFOLIO_GROUP_CONFIGURATIONS:

            current_stocks = add_type_scores(
                dataframe=current_stocks,
                type_score_configuration=type_scores,
            )

            current_market = add_type_scores(
                dataframe=current_market,
                type_score_configuration=type_scores,
            )

            current_market["Contribution"] = current_market["Horizon Score"] * current_market["Signal"] * current_market["Type Score"] 

            current_stocks["Contribution"] = current_stocks["Horizon Score"] * current_stocks["Signal"] * current_stocks["Type Score"]

            market_scores = (
                current_market
                .groupby(["Date","Portfolio Target Type"], as_index=False)
                .agg(
                    Contribution=("Contribution", "sum"),
                )
            )

            market_scores = (
                market_scores
                .groupby("Portfolio Target Type", as_index=False)
                .agg(
                    Type_Score=("Contribution", "mean"),
                )
            )

            for multiplier in REBALANCE_MULTIPLIERS:

                logger.debug(
                    "Preparing scores | horizon=%d | type=%s | rebalance multiplier=%.2f",
                    horizon_configuration_number,
                    type_scores["Name"],
                    multiplier,
                )

                rebalanced_current_stocks = apply_horizon_signal_refresh(
                    predictions_df=current_stocks,
                    rebalance_multiplier=multiplier,
                )

                stocks_scores = (
                    current_stocks
                    .groupby(["Date","Ticker","Portfolio Target Type"], as_index=False)
                    .agg(
                        Contribution=("Contribution", "sum"),
                    )
                )

                backtest_stocks = rebalanced_current_stocks[["Date", "Ticker", "Return", "Contribution"]].dropna()

                score_stocks = (
                    backtest_stocks
                    .groupby(["Date", "Ticker"], as_index=False)
                    .agg(
                        Stock_Score=("Contribution", "sum"),
                        Return=("Return", "first")
                    )
                )

                #negative_direction = (
                #    score_stocks["Stock_Direction"]
                #    <
                #    0
                #)

                #score_stocks.loc[
                #    negative_direction,
                #    "Stock_Score",
                #] = -score_stocks.loc[
                #    negative_direction,
                #    "Stock_Score",
                #].abs()

                for max_weight in MAX_WEIGHTS:

                    for penalty in CONCENTRATION_PENALTIES:

                        stocks_results = portfolio_returns_from_scores(score_stocks, max_weight=max_weight, concentration_penalty=penalty)

                        # Calculate cumulative strategy returns
                        stocks_results["Strategy Return"] = (
                            1 + stocks_results["Return"]
                        ).cumprod()


                        strategy_return = (
                            stocks_results["Strategy Return"].iloc[-1] - 1
                        )

                        strategy_volatility = (
                            stocks_results["Strategy Return"].std()
                            * np.sqrt(252)
                        )


                        # Sharpe Ratio
                        strategy_sharpe = (
                            strategy_return
                            / strategy_volatility
                        )

                        stocks_results["Strategy Peak"] = (
                            stocks_results["Strategy Return"]
                            .cummax()
                        )

                        stocks_results["Strategy Drawdown"] = (
                            (stocks_results["Strategy Return"] - stocks_results["Strategy Peak"])
                            / stocks_results["Strategy Peak"]
                        )

                        strategy_average_drawdown = stocks_results["Strategy Drawdown"].mean()

                        strategy_max_drawdown = (
                            stocks_results["Strategy Drawdown"].min()
                        )

                        strategy_relative_return = (
                            2
                            * strategy_return
                            / (
                                abs(market_return)
                                + abs(strategy_return)
                            )
                        )

                        market_relative_return = (
                            2
                            * market_return
                            / (
                                abs(market_return)
                                + abs(strategy_return)
                            )
                        )


                        strategy_relative_sharpe = (
                            2
                            * strategy_sharpe
                            / (
                                abs(market_sharpe)
                                + abs(strategy_sharpe)
                            )
                        )

                        market_relative_sharpe = (
                            2
                            * market_sharpe
                            / (
                                abs(market_sharpe)
                                + abs(strategy_sharpe)
                            )
                        )


                        strategy_relative_max_drawdown = (
                            2
                            * strategy_max_drawdown
                            / (
                                abs(market_max_drawdown)
                                + abs(strategy_max_drawdown)
                            )
                        )

                        market_relative_max_drawdown = (
                            2
                            * market_max_drawdown
                            / (
                                abs(market_max_drawdown)
                                + abs(strategy_max_drawdown)
                            )
                        )


                        strategy_relative_average_drawdown = (
                            2
                            * strategy_average_drawdown
                            / (
                                abs(market_average_drawdown)
                                + abs(strategy_average_drawdown)
                            )
                        )

                        market_relative_average_drawdown = (
                            2
                            * market_average_drawdown
                            / (
                                abs(market_average_drawdown)
                                + abs(strategy_average_drawdown)
                            )
                        )

                        strategy_quality = (
                                    0.25 * strategy_relative_sharpe +
                                    0.35 * strategy_relative_return +
                                    0.25 * strategy_relative_max_drawdown +
                                    0.15 * strategy_relative_average_drawdown
                                )

                        market_quality = (
                                    0.25 * market_relative_sharpe +
                                    0.35 * market_relative_return +
                                    0.25 * market_relative_max_drawdown +
                                    0.15 * market_relative_average_drawdown
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

                        ########################################
                        # Convert Daily Weights To Long Format
                        ########################################

                        stocks_results["Date"] = pd.to_datetime(
                            stocks_results["Date"]
                        )

                        current_score_stocks = stocks_scores.copy()

                        current_score_stocks["Date"] = pd.to_datetime(
                            current_score_stocks["Date"]
                        )


                        ticker_columns = [
                            column
                            for column in stocks_results.columns
                            if column not in {
                                "Date",
                                "Return",
                            }
                        ]


                        weights_long = stocks_results.melt(
                            id_vars=[
                                "Date",
                            ],
                            value_vars=ticker_columns,
                            var_name="Ticker",
                            value_name="Weight",
                        )


                        ########################################
                        # Add Weight To Every Target-Type Row
                        ########################################

                        current_score_stocks = (
                            current_score_stocks
                            .merge(
                                weights_long,
                                on=[
                                    "Date",
                                    "Ticker",
                                ],
                                how="left",
                                validate="many_to_one",
                            )
                        )


                        current_score_stocks["Weight"] = (
                            pd.to_numeric(
                                current_score_stocks["Weight"],
                                errors="coerce",
                            )
                            .fillna(0.0)
                        )

                        current_score_stocks["Weighted Score"] = current_score_stocks["Weight"] * current_score_stocks["Contribution"]

                        current_score_stocks = (
                            current_score_stocks
                            .groupby(["Date","Portfolio Target Type"], as_index=False)
                            .agg(
                                Daily_Score=("Weighted Score", "sum"),
                            )
                        )

                        current_score_stocks = (
                            current_score_stocks
                            .groupby("Portfolio Target Type", as_index=False)
                            .agg(
                                Type_Score=("Daily_Score", "mean"),
                            )
                        )

                        current_stocks_scores = (
                            rebalanced_current_stocks
                            .groupby(["Ticker","Portfolio Target Type"], as_index=False)
                            .agg(
                                Type_Score=("Contribution", "mean"),
                            )
                        )


                        ########################################
                        # Record Completed Simulation
                        ########################################

                        simulation_id = (
                            completed_backtests
                            + 1
                        )

                        type_configuration_index = (
                            PORTFOLIO_GROUP_CONFIGURATIONS
                            .index(
                                type_scores
                            )
                        )

                        simulation_settings = {
                            "Simulation ID": simulation_id,
                            "Horizon Score Index": (
                                horizon_configuration_number
                                - 1
                            ),
                            "Type Configuration": type_scores[
                                "Name"
                            ],
                            "Rebalance Multiplier": multiplier,
                            "Max Weight": max_weight,
                            "Concentration Penalty": penalty,
                        }

                        stock_simulation_records.append(
                            {
                                **simulation_settings,
                                **stock_metrics,
                                **extract_type_score_columns(
                                    current_score_stocks
                                ),
                            }
                        )

                        market_simulation_records.append(
                            {
                                **simulation_settings,
                                **market_metrics,
                                **extract_type_score_columns(
                                    market_scores
                                ),
                            }
                        )


                        completed_backtests += 1

                        logger.info(
                            "Backtest progress | completed=%d/%d | %.1f%% | "
                            "horizon=%d | type=%s | multiplier=%.2f | "
                            "max weight=%.2f | penalty=%.2f",
                            completed_backtests,
                            total_backtests,
                            100.0 * completed_backtests / total_backtests,
                            horizon_configuration_number,
                            type_scores["Name"],
                            multiplier,
                            max_weight,
                            penalty,
                        )


    logger.info(
        "Grid search complete | backtests=%d",
        completed_backtests,
    )


    ########################################
    # Save Simulation Results
    ########################################

    stock_simulation_results_df = pd.DataFrame(
        stock_simulation_records
    )

    market_simulation_results_df = pd.DataFrame(
        market_simulation_records
    )


    with sqlite3.connect(
        SIMULATION_RESULTS_DATABASE
    ) as connection:

        stock_simulation_results_df.to_sql(
            "Stock Simulation Results",
            connection,
            if_exists="replace",
            index=False,
        )

        market_simulation_results_df.to_sql(
            "Market Simulation Results",
            connection,
            if_exists="replace",
            index=False,
        )


    logger.info(
        "Simulation results saved | database=%s | stock rows=%d | market rows=%d",
        SIMULATION_RESULTS_DATABASE,
        len(stock_simulation_results_df),
        len(market_simulation_results_df),
    )



with sqlite3.connect(
    SIMULATION_RESULTS_DATABASE
) as connection:

    existing_tables = {
        row[0]
        for row in connection.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
            """
        ).fetchall()
    }

if {
    "Stock Simulation Results",
    "Market Simulation Results",
}.issubset(
    existing_tables
):

    logger.info(
        "Existing stock and market simulation "
        "result tables found; skipping simulations."
    )

else:

    logger.info(
        "Simulation result tables are missing; "
        "running simulations."
    )

    run_simulations()

market_results = (
    market
    .groupby("Date", as_index=False)
    .agg(
        Return=("Return", "first")
    )
)

# Calculate cumulative strategy returns
market_results["Strategy Return"] = (
    1 + market_results["Return"]
).cumprod()


market_return = (
    market_results["Strategy Return"].iloc[-1] - 1
)

market_volatility = (
    market_results["Strategy Return"].std()
    * np.sqrt(252)
)


# Sharpe Ratio
market_sharpe = (
    market_return
    / market_volatility
)

market_results["Strategy Peak"] = (
    market_results["Strategy Return"]
    .cummax()
)

market_results["Strategy Drawdown"] = (
    (market_results["Strategy Return"] - market_results["Strategy Peak"])
    / market_results["Strategy Peak"]
)

market_average_drawdown = market_results["Strategy Drawdown"].mean()

market_max_drawdown = (
    market_results["Strategy Drawdown"].min()
)

logger.info(
    "Market benchmark ready | return=%.6f | sharpe=%.6f | "
    "max drawdown=%.6f | average drawdown=%.6f",
    market_return,
    market_sharpe,
    market_max_drawdown,
    market_average_drawdown,
)


with sqlite3.connect(
    SIMULATION_RESULTS_DATABASE
) as connection:

    market_simulations_results = (
        pd.read_sql_query(
            'SELECT * FROM "Market Simulation Results"',
            connection,
        )
    )

    strategy_simulations_results = (
        pd.read_sql_query(
            'SELECT * FROM "Stock Simulation Results"',
            connection,
        )
    )


logger.info(
    "Simulation results loaded | market rows=%d | "
    "strategy rows=%d",
    len(market_simulations_results),
    len(strategy_simulations_results),
)

backtest_standard_deviation = (
    market_simulations_results[
        "Backtest Quality"
    ]
    .std()
)

'''
def stop_if_no_simulations(
    dataframe,
    filter_name,
):

    if dataframe.empty:

        logger.info(
            "No simulations remain after %s | stopping",
            filter_name,
        )

        sys.exit(0)

########################################
# Market Return Filter
########################################

rows_before_return_filter = len(
    strategy_simulations_results
)

logger.info(
    "Market return filter started | strategies=%d",
    rows_before_return_filter,
)

strategy_simulations_results = (
    strategy_simulations_results[
        strategy_simulations_results[
            "Strategy Return"
        ]
        >
        market_simulations_results[
            "Strategy Return"
        ]
    ]
    .copy()
)

rows_after_return_filter = len(
    strategy_simulations_results
)

logger.info(
    "Market return filter complete | removed=%d | remaining=%d",
    rows_before_return_filter - rows_after_return_filter,
    rows_after_return_filter,
)

stop_if_no_simulations(
    strategy_simulations_results,
    "market return filter",
)


########################################
# Portfolio Type Rejection Thresholds
########################################

PORTFOLIO_TYPE_REJECTION_THRESHOLDS = {

    "Backtest Quality": 0.75,

    ####################################
    # Ranking
    ####################################

    "ALPHA": 1.00,
    "RELATIVE_ALPHA": 0.90,
    "RISK_ADJUSTED_ALPHA": 0.90,
    "CROSS_SECTION_ALPHA": 1.00,


    ####################################
    # Direction
    ####################################

    "DIRECTION": 0.90,
    "DIRECTION_MULTICLASS": 1.00,
    "ALPHA_BINARY": 1.00,
    "BARRIER_ALPHA": 1.25,


    ####################################
    # Risk
    ####################################

    "VOLATILITY": 1.25,
    "DOWNSIDE_VOLATILITY": 0.90,
    "VOLATILITY_ASYMMETRY": 1.25,
    "DOWNSIDE": 0.90,
    "TAIL_RISK": 0.85,
    "TAIL_EVENT": 1.00,
    "DOWNSIDE_EXCURSION": 1.00,
    "VOLATILITY_EVENT": 1.25,
    "CROSS_SECTION_DOWNSIDE": 1.00,


    ####################################
    # Opportunity
    ####################################

    "ABSOLUTE_MOVE": 1.75,
    "UPSIDE_VOLATILITY": 1.50,
    "UPSIDE_EVENT": 1.50,
    "UPSIDE_EXCURSION": 1.50,
    "RECOVERY": 1.75,
    "REVERSAL": 1.75,


    ####################################
    # Timing / Special Information
    ####################################

    "TIME_TO_DOWNSIDE_EXCURSION": 1.50,
    "TIME_TO_UPSIDE_EXCURSION": 2.00,

    "EXECUTION": 2.00,
    "LIQUIDITY": 1.75,
    "MARKET_IMPACT": 2.00,

    "CORRELATION": 2.25,
    "COVARIANCE": 2.25,
    "REGIME": 1.50,
}


########################################
# Identify Available Type Scores
########################################

type_score_columns = [
    column[13:]
    for column in market_simulations_results.columns
    if column.startswith("Type Score | ")
]

logger.info(
    "Portfolio type rejection preparation started | "
    "strategies=%d | target_types=%d",
    len(strategy_simulations_results),
    len(type_score_columns),
)


########################################
# Calculate Benchmark Differences
########################################

for column in type_score_columns:

    standard_deviation = (
        market_simulations_results[
            f"Type Score | {column}"
        ]
        .std()
    )

    strategy_simulations_results[
        f"STD from Benchmark {column}"
    ] = (
        market_simulations_results[
            f"Type Score | {column}"
        ]
        - strategy_simulations_results[
            f"Type Score | {column}"
        ]
    ) / standard_deviation

    strategy_simulations_results[
        f"Type {column} Threshold"
    ] = (
        PORTFOLIO_TYPE_REJECTION_THRESHOLDS[
            column
        ]
    )

    logger.info(
        "Type comparison prepared | %s | "
        "market_std=%.6f | threshold=%.2f",
        column,
        standard_deviation,
        PORTFOLIO_TYPE_REJECTION_THRESHOLDS[
            column
        ],
    )


########################################
# Rejection Groups
########################################

PRIMARY_REJECTION_TYPES = {
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
}

SECONDARY_REJECTION_TYPES = {
    "BARRIER_ALPHA",
    "VOLATILITY",
    "VOLATILITY_ASYMMETRY",
    "TAIL_EVENT",
    "DOWNSIDE_EXCURSION",
    "VOLATILITY_EVENT",
    "CROSS_SECTION_DOWNSIDE",
    "REGIME",
}

TERTIARY_REJECTION_TYPES = {
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
}


########################################
# Type Score Rejection
########################################

def type_score_rejection(
    strategy_simulations_results,
    rejection_types,
):

    available_rejection_types = [
        column
        for column in rejection_types
        if column in type_score_columns
    ]

    logger.info(
        "Type rejection group started | "
        "available_types=%d | strategies=%d",
        len(available_rejection_types),
        len(strategy_simulations_results),
    )

    for column in available_rejection_types:

        rows_before = len(
            strategy_simulations_results
        )

        standard_deviation = (
            market_simulations_results[
                f"Type Score | {column}"
            ]
            .std()
        )

        strategy_simulations_results = (
            strategy_simulations_results[
                strategy_simulations_results[
                    f"STD from Benchmark {column}"
                ]
                <
                strategy_simulations_results[
                    f"Type {column} Threshold"
                ]
            ]
            .copy()
        )

        rows_after = len(
            strategy_simulations_results
        )

        logger.info(
            "Type rejection | %s | "
            "market_std=%.6f | threshold=%.2f | "
            "removed=%d | remaining=%d",
            column,
            standard_deviation,
            PORTFOLIO_TYPE_REJECTION_THRESHOLDS[
                column
            ],
            rows_before - rows_after,
            rows_after,
        )

        stop_if_no_simulations(
            strategy_simulations_results,
            f"{column} type-score filter",
        )

    logger.info(
        "Type rejection group complete | remaining=%d",
        len(strategy_simulations_results),
    )

    return strategy_simulations_results


########################################
# Apply Type Score Rejection
########################################

strategy_simulations_results = type_score_rejection(
    strategy_simulations_results,
    PRIMARY_REJECTION_TYPES,
)

# strategy_simulations_results = type_score_rejection(
#     strategy_simulations_results,
#     SECONDARY_REJECTION_TYPES,
# )

# strategy_simulations_results = type_score_rejection(
#     strategy_simulations_results,
#     TERTIARY_REJECTION_TYPES,
# )

logger.info(
    "Portfolio type rejection complete | "
    "remaining strategies=%d",
    len(strategy_simulations_results),
)


########################################
# Backtest Quality Filter
########################################

rows_before = len(
    strategy_simulations_results
)

logger.info(
    "Backtest Quality filter started | strategies=%d",
    rows_before,
)

backtest_standard_deviation = (
    market_simulations_results[
        "Backtest Quality"
    ]
    .std()
)

strategy_simulations_results[
    "Backtest Quality STD from Benchmark"
] = (
    market_simulations_results[
        "Backtest Quality"
    ]
    - strategy_simulations_results[
        "Backtest Quality"
    ]
) / backtest_standard_deviation

strategy_simulations_results[
    "Backtest Threshold"
] = (
    PORTFOLIO_TYPE_REJECTION_THRESHOLDS[
        "Backtest Quality"
    ]
)

strategy_simulations_results = (
    strategy_simulations_results[
        strategy_simulations_results[
            "Backtest Quality STD from Benchmark"
        ]
        <
        strategy_simulations_results[
            "Backtest Threshold"
        ]
    ]
    .copy()
)

rows_after = len(
    strategy_simulations_results
)

logger.info(
    "Backtest Quality rejection complete | "
    "market_std=%.6f | threshold=%.2f | "
    "removed=%d | remaining=%d",
    backtest_standard_deviation,
    PORTFOLIO_TYPE_REJECTION_THRESHOLDS[
        "Backtest Quality"
    ],
    rows_before - rows_after,
    rows_after,
)

stop_if_no_simulations(
    strategy_simulations_results,
    "Backtest Quality filter",
)


simulations_to_remove = []

for simulation in strategy_simulations_results.to_dict(
    orient="records"
):

    current_stocks = add_horizon_scores(
        dataframe=stocks,
        horizon_score_configuration=horizon_score_configurations[simulation["Horizon Score Index"]],
    )

    for config in PORTFOLIO_GROUP_CONFIGURATIONS:
        if config['Name'] == simulation['Type Configuration']:
            type_scores = config
            break
    
    current_stocks = add_type_scores(
        dataframe=current_stocks,
        type_score_configuration=type_scores,
    )

    current_stocks["Contribution"] = current_stocks["Horizon Score"] * current_stocks["Signal"] * current_stocks["Type Score"]

    rebalanced_current_stocks = apply_horizon_signal_refresh(
        predictions_df=current_stocks,
        rebalance_multiplier=simulation["Rebalance Multiplier"],
    )
    
    backtest_stocks = rebalanced_current_stocks[["Date", "Ticker", "Return", "Contribution"]].dropna()
    
    score_stocks = (
        backtest_stocks
        .groupby(["Date", "Ticker"], as_index=False)
        .agg(
            Stock_Score=("Contribution", "sum"),
            Return=("Return", "first")
        )
    )

    stocks_results = portfolio_returns_from_scores(score_stocks, max_weight=simulation['Max Weight'], concentration_penalty=simulation["Concentration Penalty"])[['Date','Return']]

    stocks_results["5 Day Rolling Return"] = (
        (1.0 + stocks_results["Return"])
        .rolling(
            window=5,
            min_periods=5,
        )
        .apply(
            np.prod,
            raw=True,
        )
        - 1.0
    )

    stocks_results["21 Day Rolling Return"] = (
        (1.0 + stocks_results["Return"])
        .rolling(
            window=21,
            min_periods=21,
        )
        .apply(
            np.prod,
            raw=True,
        )
        - 1.0
    )

    stocks_results["252 Day Rolling Return"] = (
        (1.0 + stocks_results["Return"])
        .rolling(
            window=252,
            min_periods=252,
        )
        .apply(
            np.prod,
            raw=True,
        )
        - 1.0
    )

    ########################################
    # Remove Best 5-Day Period
    ########################################

    stocks_results_without_best_day = (
        stocks_results.copy()
    )

    best_day = (
        stocks_results[
            "Return"
        ]
        .idxmax()
    )

    best_day_position = (
        stocks_results.index.get_loc(
            best_day
        )
    )

    stocks_results_without_best_day.loc[
        stocks_results_without_best_day.index[
            best_day_position
        ],
        "Return",
    ] = 0.0
    

    ########################################
    # Remove Best 5-Day Period
    ########################################

    stocks_results_without_best_5_days = (
        stocks_results.copy()
    )

    best_5_day_end = (
        stocks_results[
            "5 Day Rolling Return"
        ]
        .idxmax()
    )

    best_5_day_end_position = (
        stocks_results.index.get_loc(
            best_5_day_end
        )
    )

    best_5_day_start_position = (
        best_5_day_end_position
        - 5
        + 1
    )

    stocks_results_without_best_5_days.loc[
        stocks_results_without_best_5_days.index[
            best_5_day_start_position:
            best_5_day_end_position + 1
        ],
        "Return",
    ] = 0.0


    ########################################
    # Remove Best 21-Day Period
    ########################################

    stocks_results_without_best_21_days = (
        stocks_results.copy()
    )

    best_21_day_end = (
        stocks_results[
            "21 Day Rolling Return"
        ]
        .idxmax()
    )

    best_21_day_end_position = (
        stocks_results.index.get_loc(
            best_21_day_end
        )
    )

    best_21_day_start_position = (
        best_21_day_end_position
        - 21
        + 1
    )

    stocks_results_without_best_21_days.loc[
        stocks_results_without_best_21_days.index[
            best_21_day_start_position:
            best_21_day_end_position + 1
        ],
        "Return",
    ] = 0.0


    ########################################
    # Remove Best 252-Day Period
    ########################################

    stocks_results_without_best_252_days = (
        stocks_results.copy()
    )

    best_252_day_end = (
        stocks_results[
            "252 Day Rolling Return"
        ]
        .idxmax()
    )

    best_252_day_end_position = (
        stocks_results.index.get_loc(
            best_252_day_end
        )
    )

    best_252_day_start_position = (
        best_252_day_end_position
        - 252
        + 1
    )

    stocks_results_without_best_252_days.loc[
        stocks_results_without_best_252_days.index[
            best_252_day_start_position:
            best_252_day_end_position + 1
        ],
        "Return",
    ] = 0.0

    removed_results = {}

    removed_results['1'] = stocks_results_without_best_day

    removed_results['5'] = stocks_results_without_best_5_days

    removed_results['21'] = stocks_results_without_best_21_days

    removed_results['252'] = stocks_results_without_best_252_days

    number_of_backtest_rows = len(
        stocks_results
    )

    number_of_backtest_rows = len(
        stocks_results
    )

    BASE_BQ_THRESHOLD = 0.75
    MINIMUM_STRESSED_BQ_THRESHOLD = 0.80
    MAXIMUM_STRESSED_BQ_THRESHOLD = 1.50

    MINIMUM_REMOVED_FRACTION = 0.002
    THRESHOLD_INCREASE_MULTIPLIER = 3.0


    rolling_periods = [
        1,
        5,
        21,
        252,
    ]


    rolling_test_settings = {}

    for rolling_period in rolling_periods:

        removed_fraction = (
            rolling_period
            / number_of_backtest_rows
        )

        stressed_bq_threshold = np.clip(
            BASE_BQ_THRESHOLD
            + (
                THRESHOLD_INCREASE_MULTIPLIER
                * removed_fraction
            ),
            MINIMUM_STRESSED_BQ_THRESHOLD,
            MAXIMUM_STRESSED_BQ_THRESHOLD,
        )

        if removed_fraction >= MINIMUM_REMOVED_FRACTION:

            # Calculate cumulative strategy returns
            removed_results[str(rolling_period)]["Strategy Return"] = (
                1 + removed_results[str(rolling_period)]["Return"]
            ).cumprod()


            strategy_return = (
                removed_results[str(rolling_period)]["Strategy Return"].iloc[-1] - 1
            )

            strategy_volatility = (
                removed_results[str(rolling_period)]["Strategy Return"].std()
                * np.sqrt(252)
            )


            # Sharpe Ratio
            strategy_sharpe = (
                strategy_return
                / strategy_volatility
            )

            removed_results[str(rolling_period)]["Strategy Peak"] = (
                removed_results[str(rolling_period)]["Strategy Return"]
                .cummax()
            )

            removed_results[str(rolling_period)]["Strategy Drawdown"] = (
                (removed_results[str(rolling_period)]["Strategy Return"] - removed_results[str(rolling_period)]["Strategy Peak"])
                / removed_results[str(rolling_period)]["Strategy Peak"]
            )

            strategy_average_drawdown = removed_results[str(rolling_period)]["Strategy Drawdown"].mean()

            strategy_max_drawdown = (
                removed_results[str(rolling_period)]["Strategy Drawdown"].min()
            )

            strategy_relative_return = (
                2
                * strategy_return
                / (
                    abs(market_return)
                    + abs(strategy_return)
                )
            )

            strategy_relative_sharpe = (
                2
                * strategy_sharpe
                / (
                    abs(market_sharpe)
                    + abs(strategy_sharpe)
                )
            )

            strategy_relative_max_drawdown = (
                2
                * strategy_max_drawdown
                / (
                    abs(market_max_drawdown)
                    + abs(strategy_max_drawdown)
                )
            )

            strategy_relative_average_drawdown = (
                2
                * strategy_average_drawdown
                / (
                    abs(market_average_drawdown)
                    + abs(strategy_average_drawdown)
                )
            )

            strategy_quality = (
                        0.25 * strategy_relative_sharpe +
                        0.35 * strategy_relative_return +
                        0.25 * strategy_relative_max_drawdown +
                        0.15 * strategy_relative_average_drawdown
                    )

            simulation_id = simulation[
                "Simulation ID"
            ]

            market_backtest_quality = (
                market_simulations_results.loc[
                    market_simulations_results[
                        "Simulation ID"
                    ].eq(
                        simulation_id
                    ),
                    "Backtest Quality",
                ]
                .iloc[0]
            )

            deviations = (market_backtest_quality - strategy_quality) / backtest_standard_deviation

            if deviations >= stressed_bq_threshold:
                simulations_to_remove.append(simulation["Simulation ID"])

strategy_simulations_results = (
    strategy_simulations_results[
        ~strategy_simulations_results[
            "Simulation ID"
        ]
        .isin(
            simulations_to_remove
        )
    ]
    .copy()
)
'''



tickers = (
    stocks[
        "Ticker"
    ]
    .unique()
)

number_of_stocks = len(tickers)

stock_removal_threshold = np.clip(
    0.75
    + (
        3.0
        / number_of_stocks
    ),
    0.80,
    1.25,
)

simulations_to_remove = []

for simulation in strategy_simulations_results.to_dict(
    orient="records"
):

    current_stocks = add_horizon_scores(
        dataframe=stocks,
        horizon_score_configuration=horizon_score_configurations[simulation["Horizon Score Index"]],
    )

    for config in PORTFOLIO_GROUP_CONFIGURATIONS:
        if config['Name'] == simulation['Type Configuration']:
            type_scores = config
            break
    
    current_stocks = add_type_scores(
        dataframe=current_stocks,
        type_score_configuration=type_scores,
    )

    current_stocks["Contribution"] = current_stocks["Horizon Score"] * current_stocks["Signal"] * current_stocks["Type Score"]

    rebalanced_current_stocks = apply_horizon_signal_refresh(
        predictions_df=current_stocks,
        rebalance_multiplier=simulation["Rebalance Multiplier"],
    )
    
    backtest_stocks = rebalanced_current_stocks[["Date", "Ticker", "Return", "Contribution"]].dropna()
    
    score_stocks = (
        backtest_stocks
        .groupby(["Date", "Ticker"], as_index=False)
        .agg(
            Stock_Score=("Contribution", "sum"),
            Return=("Return", "first")
        )
    )

    for removed_ticker in tickers:

        removed_score_stocks = (
            score_stocks[
                score_stocks[
                    "Ticker"
                ].ne(
                    removed_ticker
                )
            ]
            .copy()
        )
        
        stocks_results = portfolio_returns_from_scores(score_stocks, max_weight=simulation['Max Weight'], concentration_penalty=simulation["Concentration Penalty"])[['Date','Return']]

            # Calculate cumulative strategy returns
        stocks_results["Strategy Return"] = (
            1 + stocks_results["Return"]
        ).cumprod()


        strategy_return = (
            stocks_results["Strategy Return"].iloc[-1] - 1
        )

        strategy_volatility = (
            stocks_results["Strategy Return"].std()
            * np.sqrt(252)
        )


        # Sharpe Ratio
        strategy_sharpe = (
            strategy_return
            / strategy_volatility
        )

        stocks_results["Strategy Peak"] = (
            stocks_results["Strategy Return"]
            .cummax()
        )

        stocks_results["Strategy Drawdown"] = (
            (stocks_results["Strategy Return"] - stocks_results["Strategy Peak"])
            / stocks_results["Strategy Peak"]
        )

        strategy_average_drawdown = stocks_results["Strategy Drawdown"].mean()

        strategy_max_drawdown = (
            stocks_results["Strategy Drawdown"].min()
        )

        strategy_relative_return = (
            2
            * strategy_return
            / (
                abs(market_return)
                + abs(strategy_return)
            )
        )

        strategy_relative_sharpe = (
            2
            * strategy_sharpe
            / (
                abs(market_sharpe)
                + abs(strategy_sharpe)
            )
        )

        strategy_relative_max_drawdown = (
            2
            * strategy_max_drawdown
            / (
                abs(market_max_drawdown)
                + abs(strategy_max_drawdown)
            )
        )

        strategy_relative_average_drawdown = (
            2
            * strategy_average_drawdown
            / (
                abs(market_average_drawdown)
                + abs(strategy_average_drawdown)
            )
        )

        strategy_quality = (
                    0.25 * strategy_relative_sharpe +
                    0.35 * strategy_relative_return +
                    0.25 * strategy_relative_max_drawdown +
                    0.15 * strategy_relative_average_drawdown
                )

        simulation_id = simulation[
            "Simulation ID"
        ]

        market_backtest_quality = (
            market_simulations_results.loc[
                market_simulations_results[
                    "Simulation ID"
                ].eq(
                    simulation_id
                ),
                "Backtest Quality",
            ]
            .iloc[0]
        )

        deviations = (market_backtest_quality - strategy_quality) / backtest_standard_deviation

        if deviations >= stock_removal_threshold:
            simulations_to_remove.append(simulation["Simulation ID"])

strategy_simulations_results = (
    strategy_simulations_results[
        ~strategy_simulations_results[
            "Simulation ID"
        ]
        .isin(
            simulations_to_remove
        )
    ]
    .copy()
)
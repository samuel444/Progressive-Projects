"""Standalone Equity Selector final evaluation.

Run this file directly. It reads Passed Strategies and source data from SQLite.
The simulation script and main_package are never imported or executed.
Copied helper definitions preserve the supplied backtest conventions.
Dependencies: numpy, pandas. See --help for paths and evaluation assumptions.
"""
import argparse
import ast
import json
import copy
import logging
import sqlite3
from statistics import NormalDist
from pathlib import Path
import numpy as np
import pandas as pd

# Copied settings and helpers from the supplied pipeline.

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

def build_score_stocks_with_direction(
    dataframe,
):
    """
    Aggregate target contributions into one daily stock score
    and apply the direction gate.

    Negative direction forces Stock_Score to be non-positive,
    causing portfolio_returns_from_scores() to assign zero weight.
    """

    required_columns = {
        "Date",
        "Ticker",
        "Return",
        "Portfolio Target Type",
        "Contribution",
    }

    missing_columns = (
        required_columns
        - set(dataframe.columns)
    )

    if missing_columns:

        raise ValueError(
            "Missing required columns: "
            f"{sorted(missing_columns)}"
        )


    current_dataframe = (
        dataframe[
            [
                "Date",
                "Ticker",
                "Return",
                "Portfolio Target Type",
                "Contribution",
            ]
        ]
        .dropna(
            subset=[
                "Date",
                "Ticker",
                "Return",
                "Contribution",
            ]
        )
        .copy()
    )


    ########################################
    # Separate Direction Contribution
    ########################################

    current_dataframe[
        "Direction Contribution"
    ] = (
        current_dataframe[
            "Contribution"
        ]
        .where(
            current_dataframe[
                "Portfolio Target Type"
            ].isin(
                PORTFOLIO_DIRECTION_TYPES
            ),
            0.0,
        )
    )


    ########################################
    # Aggregate Each Portfolio Target Type
    ########################################

    type_score_stocks = (
        current_dataframe
        .groupby(
            [
                "Date",
                "Ticker",
                "Portfolio Target Type",
            ],
            as_index=False,
        )
        .agg(
            Contribution_Sum=(
                "Contribution",
                "sum",
            ),
            Direction_Sum=(
                "Direction Contribution",
                "sum",
            ),
            Return=(
                "Return",
                "first",
            ),
        )
    )


    ########################################
    # Aggregate Complete Stock Score
    ########################################

    score_stocks = (
        type_score_stocks
        .groupby(
            [
                "Date",
                "Ticker",
            ],
            as_index=False,
        )
        .agg(
            Stock_Score=(
                "Contribution_Sum",
                "sum",
            ),
            Stock_Direction=(
                "Direction_Sum",
                "mean",
            ),
            Return=(
                "Return",
                "first",
            ),
        )
    )


    ########################################
    # Apply Negative-Direction Gate
    ########################################

    negative_direction = (
        score_stocks[
            "Stock_Direction"
        ]
        <
        0
    )

    score_stocks.loc[
        negative_direction,
        "Stock_Score",
    ] = -score_stocks.loc[
        negative_direction,
        "Stock_Score",
    ].abs()


    return score_stocks

def apply_horizon_signal_refresh(
    predictions_df,
    rebalance_multiplier,
):

    if not (
        0 <= rebalance_multiplier <= 1
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


    return result

def result_metrics(
    results_dataframe,
):
    # Calculate cumulative strategy returns
    results_dataframe["Strategy Return"] = (
        1 + results_dataframe["Return"]
    ).cumprod()
    
    
    strategy_return = (
        results_dataframe["Strategy Return"].iloc[-1] - 1
    )
    
    strategy_volatility = (
        results_dataframe["Return"].std()
        * np.sqrt(252)
    )
    
    
    # Sharpe Ratio
    strategy_sharpe = (
        strategy_return
        / strategy_volatility
    )
    
    results_dataframe["Strategy Peak"] = (
        results_dataframe["Strategy Return"]
        .cummax()
    )
    
    results_dataframe["Strategy Drawdown"] = (
        (results_dataframe["Strategy Return"] - results_dataframe["Strategy Peak"])
        / results_dataframe["Strategy Peak"]
    )
    
    strategy_average_drawdown = results_dataframe["Strategy Drawdown"].mean()
    
    strategy_max_drawdown = (
        results_dataframe["Strategy Drawdown"].min()
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

    return strategy_return, strategy_sharpe, strategy_average_drawdown, strategy_max_drawdown, strategy_relative_return, strategy_relative_sharpe, strategy_relative_max_drawdown, strategy_relative_average_drawdown, strategy_quality

def portfolio_returns_from_scores(
    dataframe,
    max_weight = 0.30,
    concentration_penalty = 0.10,
    trading_fee = 0.00
):

    ########################################
    # Validate
    ########################################

    required_columns = {
        "Date",
        "Ticker",
        "Return",
        "Stock_Score",
    }

    missing_columns = (
        required_columns
        .difference(
            dataframe.columns
        )
    )

    if missing_columns:
        raise ValueError(
            "Missing columns: "
            + ", ".join(
                sorted(
                    missing_columns
                )
            )
        )


    ########################################
    # Clean
    ########################################

    data = dataframe[
        [
            "Date",
            "Ticker",
            "Return",
            "Stock_Score",
        ]
    ].copy()

    data[
        "Date"
    ] = pd.to_datetime(
        data[
            "Date"
        ],
        errors="coerce",
    )

    data[
        "Return"
    ] = (
        pd.to_numeric(
            data[
                "Return"
            ],
            errors="coerce",
        )
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
    )

    data[
        "Stock_Score"
    ] = (
        pd.to_numeric(
            data[
                "Stock_Score"
            ],
            errors="coerce",
        )
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
        .fillna(
            0.0
        )
        .clip(
            lower=0.0
        )
    )

    data = (
        data
        .dropna(
            subset=[
                "Date",
                "Ticker",
                "Return",
            ]
        )
        .sort_values(
            [
                "Date",
                "Ticker",
            ]
        )
        .reset_index(
            drop=True
        )
    )


    ########################################
    # Weight Calculation
    ########################################

    def calculate_weights(
        daily_data,
    ):

        scores = (
            daily_data
            .set_index(
                "Ticker"
            )[
                "Stock_Score"
            ]
        )

        active = (
            scores > 0
        )

        if not active.any():
            return pd.Series(
                dtype=float
            )


        score_weights = (
            scores
            / scores.sum()
        )

        equal_weights = (
            active.astype(float)
            / active.sum()
        )


        # Blend mostly score-proportional
        # weights with 10% equal weighting.
        desired_weights = (
            (
                1.0
                - concentration_penalty
            )
            * score_weights
            +
            concentration_penalty
            * equal_weights
        )


        ####################################
        # Enforce Maximum Weight
        ####################################

        final_weights = pd.Series(
            0.0,
            index=desired_weights.index,
        )

        remaining_tickers = list(
            desired_weights.index[
                active
            ]
        )

        remaining_capital = min(
            1.0,
            len(
                remaining_tickers
            )
            * max_weight,
        )


        while (
            remaining_tickers
            and remaining_capital > 1e-12
        ):

            remaining_scores = (
                desired_weights.loc[
                    remaining_tickers
                ]
            )

            proposed = (
                remaining_scores
                / remaining_scores.sum()
                * remaining_capital
            )

            over_cap = (
                proposed > max_weight
            )

            if not over_cap.any():

                final_weights.loc[
                    remaining_tickers
                ] = proposed

                break


            capped_tickers = list(
                proposed.index[
                    over_cap
                ]
            )

            final_weights.loc[
                capped_tickers
            ] = max_weight

            remaining_capital -= (
                max_weight
                * len(
                    capped_tickers
                )
            )

            remaining_tickers = [
                ticker
                for ticker in remaining_tickers
                if ticker not in capped_tickers
            ]


        return final_weights[
            final_weights > 0
        ]


    ########################################
    # Run Through Dates
    ########################################

    dates = (
        data[
            "Date"
        ]
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    current_weights = pd.Series(
        dtype=float
    )

    previous_weights = pd.Series(
        dtype=float
    )

    return_records = []


    for date_number, date in enumerate(
        dates[:-1]
    ):


        ########################################
        # Current Date
        #
        # Scores from this date determine the
        # portfolio weights.
        ########################################

        daily_data = (
            data[
                data[
                    "Date"
                ].eq(
                    date
                )
            ]
            .copy()
        )


        ########################################
        # Following Trading Date
        #
        # These are the returns earned by the
        # portfolio selected on the current date.
        ########################################

        tomorrow_date = dates[
            date_number + 1
        ]

        tomorrow_data = (
            data[
                data[
                    "Date"
                ].eq(
                    tomorrow_date
                )
            ]
            .copy()
        )


        current_weights = (
            calculate_weights(
                daily_data
            )
        )


        all_tickers = (
            previous_weights.index
            .union(
                current_weights.index
            )
        )

        old_weights = (
            previous_weights
            .reindex(
                all_tickers,
                fill_value=0.0,
            )
        )

        new_weights = (
            current_weights
            .reindex(
                all_tickers,
                fill_value=0.0,
            )
        )

        turnover = (
            0.5
            * (
                new_weights
                - old_weights
            )
            .abs()
            .sum()
        )

        previous_weights = (
            current_weights.copy()
        )



        ####################################
        # Apply Held Weights
        ####################################

        tomorrow_returns = (
            daily_data
            .set_index(
                "Ticker"
            )[
                "Return"
            ]
        )

        aligned_returns = (
            tomorrow_returns
            .reindex(
                current_weights.index
            )
            .fillna(
                0.0
            )
        )

        gross_return = float(
            (
                current_weights
                * aligned_returns
            ).sum()
        )

        net_return = (
            gross_return
            - turnover
            * trading_fee
        )

        return_record = {
            "Date": tomorrow_date,
            "Return": net_return,
        }

        all_tickers = sorted(
            data["Ticker"]
            .dropna()
            .unique()
        )

        for ticker in all_tickers:

            weight = 0

            if ticker in current_weights.index:
                weight = current_weights[ticker]

            return_record[ticker] = weight


        return_records.append(
            return_record
        )


    return pd.DataFrame(
        return_records
    )

# Explicit evaluation assumptions; these do not change earlier selection.
FE_COST_BPS = 10.0             # Illustrative all-in cost per unit BUY + SELL notional.
FE_RF_ANNUAL = 0.0
FE_DSR_SAMPLE_SIZE = 100       # Random UNFILTERED grid reruns to estimate Sharpe dispersion.
FE_DSR_TRIALS = None           # Override with total research trials, if larger than saved grid.
FE_SEED = 20260904
FE_DAYS = 252
FE_NEIGHBOURHOOD_SD = np.nan  # Optional ORIGINAL pre-pruning cohort SD.
FE_UNSEEN_GATE = 1.50        # Original unseen-stock rejection threshold.

# Final-table name -> your existing result-column name. Edit the right-hand names
# when you add these metrics upstream. Existing non-null values take precedence.
FE_EXISTING_COLUMNS = {
    'Best Day Removed Quality': 'Best Day Removed Quality',
    'Best Week Removed Quality': 'Best Week Removed Quality',
    'Best Month Removed Quality': 'Best Month Removed Quality',
    'Best Year Removed Quality': 'Best Year Removed Quality',
    'Mean Stock Removal Quality': 'Mean Stock Removal Quality',
    'Worst Stock Removal Quality': 'Worst Stock Removal Quality',
    'Worst Removed Ticker': 'Worst Removed Ticker',
    'Neighbourhood Score': 'Neighbourhood Score',
    'Neighbourhood Pass Rate': 'Neighbourhood Pass Rate',
    'Unseen Stock Score': 'Unseen Stock Score',
    'Unseen Backtest Quality': 'Unseen Backtest Quality',
    'Unseen Gate Passed': 'Unseen Gate Passed',
    'Portfolio Target Score': 'Portfolio Target Score',
    'Market Target Score': 'Market Target Score',
    'Relative Target Score': 'Relative Target Score',
}

FE_METRICS = [
    'Strategy Return', 'Sharpe Ratio', 'Average Drawdown', 'Max Drawdown',
    'Relative Return', 'Relative Sharpe Ratio', 'Relative Max Drawdown',
    'Relative Average Drawdown', 'Backtest Quality',
]
FE_SETTINGS = ['Horizon Score Index', 'Type Configuration',
               'Rebalance Multiplier', 'Max Weight', 'Concentration Penalty']


def fe_series(frame):
    s = frame.set_index('Date')['Return'].copy()
    s.index = pd.to_datetime(s.index)
    s = s.sort_index().astype(float)
    if s.index.has_duplicates or not np.isfinite(s).all() or (s < -1).any():
        raise ValueError('Invalid, duplicated or missing daily returns.')
    if len(s) < 3:
        raise ValueError('At least three daily observations required.')
    return s


def fe_sharpe(r, annual=True):
    x = np.asarray(r, dtype=float) - ((1 + FE_RF_ANNUAL)**(1 / FE_DAYS) - 1)
    sd = x.std(ddof=1)
    return x.mean() / sd * (np.sqrt(FE_DAYS) if annual else 1) if sd > 0 else np.nan


def fe_psr(r, benchmark=0.0):
    # Original PSR moment approximation: IID/stationary observations assumed.
    x = np.asarray(r, dtype=float) - ((1 + FE_RF_ANNUAL)**(1 / FE_DAYS) - 1)
    n = len(x)
    sd = x.std(ddof=1)
    if n < 30 or sd <= 0 or not np.isfinite(benchmark):
        return np.nan
    sr = x.mean() / sd
    centered = x - x.mean()
    m2 = np.mean(centered**2)
    skew = np.mean(centered**3) / m2**1.5
    kurtosis = np.mean(centered**4) / m2**2  # Pearson, not excess kurtosis.
    variance = 1 - skew * sr + (kurtosis - 1) * sr**2 / 4
    if variance <= 0:
        return np.nan
    return NormalDist().cdf((sr - benchmark) * np.sqrt(n - 1) / np.sqrt(variance))


def fe_quality(r):
    # Preserve result_metrics(), including its full-history benchmark globals.
    return dict(zip(FE_METRICS, result_metrics(
        pd.DataFrame({'Date': r.index, 'Return': r.to_numpy()})
    )))


def fe_scores(row, universe):
    configs = [c for c in PORTFOLIO_GROUP_CONFIGURATIONS
               if c['Name'] == row['Type Configuration']]
    if len(configs) != 1:
        raise ValueError('Type Configuration must identify exactly one configuration.')
    h = float(row['Horizon Score Index'])
    if not h.is_integer() or not 0 <= h < len(horizon_score_configurations):
        raise ValueError('Invalid Horizon Score Index.')
    d = add_horizon_scores(universe.copy(), horizon_score_configurations[int(h)])
    d = add_type_scores(d, copy.deepcopy(configs[0]))
    d['Contribution'] = d['Horizon Score'] * d['Signal'] * d['Type Score']
    d = apply_horizon_signal_refresh(d, float(row['Rebalance Multiplier']))
    return build_score_stocks_with_direction(d)


def fe_run_scores(row, scores):
    # Explicit zero fee reproduces the supplied pipeline's gross-return runs.
    frame = portfolio_returns_from_scores(
        scores.copy(), max_weight=float(row['Max Weight']),
        concentration_penalty=float(row['Concentration Penalty']), trading_fee=0.0,
    )
    frame['Date'] = pd.to_datetime(frame['Date'])
    frame = frame.sort_values('Date').reset_index(drop=True)
    fe_series(frame)
    return frame


def fe_best_removed(r, window):
    # Original convention: zero best 1/5/21/252 consecutive returns; retain dates.
    if len(r) <= window:
        return np.nan
    rolling = (1 + r).rolling(window).apply(np.prod, raw=True)
    end = int(np.nanargmax(rolling.to_numpy()))
    changed = r.copy()
    changed.iloc[end - window + 1:end + 1] = 0.0
    return fe_quality(changed)['Backtest Quality']


def fe_costs(frame):
    # Match the helper: changes in TARGET weights, starting from cash, no drift.
    # Helper turnover = 0.5 * sum(abs(delta weights)); costs here use FULL notional.
    w = frame.drop(columns=['Date', 'Return']).astype(float)
    if w.empty or not np.isfinite(w.to_numpy()).all():
        raise ValueError('Daily ticker weights missing or invalid.')
    delta = w.diff()
    delta.iloc[0] = w.iloc[0]
    traded = delta.abs().sum(axis=1).to_numpy()
    r = frame['Return'].to_numpy(dtype=float)
    c = FE_COST_BPS / 10000
    if np.any(1 + r - 2 * c * traded <= 0):
        raise ValueError('Cost stress causes insolvency; inspect the daily results.')
    # Break-even total cost: compounded terminal equity equals initial equity.
    if np.prod(1 + r) <= 1:
        breakeven = 0.0
    elif not np.any(traded > 0):
        breakeven = np.inf
    else:
        low = 0.0
        high = np.min((1 + r[traded > 0]) / traded[traded > 0])
        for _ in range(80):
            mid = (low + high) / 2
            if np.log1p(r - mid * traded).sum() > 0:
                low = mid
            else:
                high = mid
        breakeven = (low + high) / 2 * 10000
    return {
        'Annual Turnover': 0.5 * traded.mean() * FE_DAYS,
        'Sharpe at Realistic Costs': fe_sharpe(r - c * traded),
        'Sharpe at 2x Costs': fe_sharpe(r - 2 * c * traded),
        'Break-Even Cost (bps)': breakeven,
    }, traded


def fe_es(r):
    # Exact empirical lower 5% probability mass, including a fractional boundary.
    x = np.sort(np.asarray(r, dtype=float))
    mass = .05 * len(x)
    whole = int(np.floor(mass))
    return (x[:whole].sum() + (mass - whole) * x[whole]) / mass


def build_final_strategy_evaluation(finalists=None):
    log = logging.getLogger('final_evaluation')
    with sqlite3.connect(Path(SIMULATION_RESULTS_DATABASE).resolve().as_uri() + '?mode=ro',
                         uri=True) as connection:
        grid = pd.read_sql_query('SELECT * FROM "Stock Simulation Results"', connection)
        market_grid = pd.read_sql_query('SELECT * FROM "Market Simulation Results"', connection)
        if finalists is None:
            with sqlite3.connect(Path(BACKTEST_DATABASE).resolve().as_uri() + '?mode=ro',
                                 uri=True) as passed_connection:
                finalists = pd.read_sql_query('SELECT * FROM "Passed Strategies"', passed_connection)
    selected = finalists.copy(deep=True)
    if not 1 <= len(selected) <= 10:
        raise ValueError('Pass the already-selected 1–10 strategies; no automatic truncation.')
    for df in (selected, grid, market_grid):
        if df['Simulation ID'].duplicated().any():
            raise ValueError('Simulation ID must be unique.')
    if not set(FE_SETTINGS).issubset(selected.columns):
        raise ValueError('Selected rows are missing reconstruction settings.')
    if not set(selected['Simulation ID']).issubset(set(grid['Simulation ID'])):
        raise ValueError('Finalists do not belong to the saved simulation grid.')
    if FE_COST_BPS < 0 or FE_RF_ANNUAL <= -1 or FE_DSR_SAMPLE_SIZE < 2:
        raise ValueError('Invalid final evaluation assumptions.')

    unseen = None
    if unseen is None:
        with sqlite3.connect(Path(BACKTEST_DATABASE).resolve().as_uri() + '?mode=ro',
                             uri=True) as connection:
            unseen = pd.read_sql_query('SELECT * FROM "Unseen"', connection)
    unseen = unseen.copy()
    unseen['Date'] = pd.to_datetime(unseen['Date'])
    if unseen.empty or set(unseen['Ticker']) & set(stocks['Ticker']):
        raise ValueError('Unseen universe must be nonempty and disjoint from Stocks.')

    # Original neighbourhood cohort SD cannot be recovered from the final ten.
    # It is optional and used only for the detailed drop diagnostic.
    # Unseen SD is recoverable from the original full market-results table.
    neighbour_sd = FE_NEIGHBOURHOOD_SD
    unseen_sd = market_grid['Backtest Quality'].std()
    unseen_gate = FE_UNSEEN_GATE
    benchmark_rows = market_grid.set_index('Simulation ID')
    raw_grid = grid.set_index('Simulation ID')
    details, rows, streams = {}, [], {}

    for _, row in selected.iterrows():
        sid = row['Simulation ID']
        log.info('Final evaluation: simulation %s', sid)
        saved = raw_grid.loc[sid]
        # Use persisted calibration values from the simulation, when available.
        neighbour_sd = float(row.get('Neighbourhood SD', FE_NEIGHBOURHOOD_SD))
        unseen_sd = float(row.get('Unseen Quality SD', market_grid['Backtest Quality'].std()))
        unseen_gate = float(row.get('Unseen Threshold', FE_UNSEEN_GATE))
        for key in FE_SETTINGS:
            equal = (str(row[key]) == str(saved[key]) if key == 'Type Configuration'
                     else np.isclose(float(row[key]), float(saved[key])))
            if not equal:
                raise ValueError(f'{sid}: settings differ from saved grid: {key}')
        score = fe_scores(row, stocks)
        frame = fe_run_scores(row, score)
        r = fe_series(frame)
        metrics = fe_quality(r)
        mismatches = [k for k in FE_METRICS if k in row and
                      not np.isclose(float(row[k]), metrics[k], equal_nan=True,
                                     rtol=1e-7, atol=1e-10)]
        if mismatches:
            raise ValueError(f'{sid}: rerun differs from saved metrics: {mismatches}. '
                             'Check data, helpers and benchmark globals before continuing.')
        streams[sid] = r
        out = {'Simulation ID': sid, **metrics}
        for key in ['Neighbourhood Score', 'Unseen Stock Score']:
            out[key] = row.get(key, np.nan)
        for label, window in [('Day', 1), ('Week', 5), ('Month', 21), ('Year', 252)]:
            out[f'Best {label} Removed Quality'] = fe_best_removed(r, window)

        # Fixed 252-session windows, advanced monthly; no fresh signal warm-up/reset.
        # Use the same existing full-history-relative quality definition, labelled so.
        periods = []
        ends = sorted(set(range(FE_DAYS, len(r) + 1, 21)) | ({len(r)} if len(r) >= FE_DAYS else set()))
        for end in ends:
            part = r.iloc[end - FE_DAYS:end]
            periods.append({'Start': part.index[0], 'End': part.index[-1],
                            'Backtest Quality': fe_quality(part)['Backtest Quality']})
        out['Worst 252d Quality (fixed benchmark)'] = (
            min(p['Backtest Quality'] for p in periods) if periods else np.nan)

        removals = []
        for ticker in sorted(score['Ticker'].unique()):
            reduced = score.loc[score['Ticker'].ne(ticker)]
            rr = fe_series(fe_run_scores(row, reduced))
            if not rr.index.equals(r.index):
                raise ValueError(f'{sid}: removing {ticker} changes the date coverage.')
            removals.append({'Ticker': ticker, 'Backtest Quality': fe_quality(rr)['Backtest Quality']})
        removal_df = pd.DataFrame(removals)
        valid = removal_df['Backtest Quality'].notna()
        out['Mean Stock Removal Quality'] = removal_df['Backtest Quality'].mean() if valid.all() else np.nan
        worst = removal_df.loc[removal_df['Backtest Quality'].idxmin()] if valid.any() else None
        out['Worst Stock Removal Quality'] = worst['Backtest Quality'] if worst is not None else np.nan
        out['Worst Removed Ticker'] = worst['Ticker'] if worst is not None else None

        # Fresh reproducible diagnostic, same +/-0.05 bounds as existing pipeline.
        rng = np.random.default_rng(np.random.SeedSequence([FE_SEED, int(sid)]))
        neighbours = []
        for iteration in range(30):
            p = row.copy()
            for key in ['Rebalance Multiplier', 'Concentration Penalty', 'Max Weight']:
                p[key] = rng.uniform(max(0., float(row[key]) - .05),
                                     min(1., float(row[key]) + .05))
            neighbour_frame = fe_run_scores(p, fe_scores(p, stocks))
            nr = fe_series(neighbour_frame)
            if not nr.index.equals(r.index):
                raise ValueError('Neighbour changed date coverage.')
            q = fe_quality(nr)['Backtest Quality']
            drop = (metrics['Backtest Quality'] - q) / neighbour_sd if neighbour_sd > 0 else np.nan
            _, neighbour_traded = fe_costs(neighbour_frame)
            neighbour_net = nr.to_numpy() - FE_COST_BPS / 10000 * neighbour_traded
            passed = bool(np.prod(1 + neighbour_net) > 1)
            neighbours.append({**{k: p[k] for k in FE_SETTINGS},
                               'Backtest Quality': q, 'Drop (sigma)': drop, 'Passed': passed})
        neighbour_df = pd.DataFrame(neighbours)
        out['Neighbourhood Cost Survival Rate'] = (neighbour_df['Passed'].mean()
            if neighbour_df['Passed'].notna().all() else np.nan)

        ur = fe_series(fe_run_scores(row, fe_scores(row, unseen)))
        if not ur.index.equals(r.index):
            raise ValueError(f'{sid}: unseen history differs; align source universes explicitly.')
        uq = fe_quality(ur)['Backtest Quality']
        out['Unseen Backtest Quality'] = uq
        # Original code compares unseen quality with the matched MARKET quality.
        uscore = ((float(benchmark_rows.loc[sid, 'Backtest Quality']) - uq) / unseen_sd
                  if unseen_sd > 0 else np.nan)
        out['Unseen Gate Passed'] = (bool(uscore < unseen_gate)
            if np.isfinite(uscore) and np.isfinite(unseen_gate) else None)
        if np.isfinite(uscore) and pd.notna(out['Unseen Stock Score']):
            if not np.isclose(uscore, float(out['Unseen Stock Score']), rtol=1e-7, atol=1e-10):
                raise ValueError(f'{sid}: unseen score changed; inspect data/settings.')

        # Existing type contributions are signal scores, not predictive accuracy.
        # Standardize each stock-minus-market difference before aggregating types.
        type_details = []
        for key in market_grid.columns:
            if not key.startswith('Type Score | ') or key not in row:
                continue
            a, b = float(row[key]), float(benchmark_rows.loc[sid, key])
            sd = market_grid[key].std()
            if np.isfinite(a) and np.isfinite(b) and sd > 0:
                type_details.append({'Type': key[13:], 'Strategy': a, 'Market': b,
                                     'Relative (sigma)': (a - b) / sd})
        out['Mean Relative Target Score (sigma)'] = (
            np.mean([t['Relative (sigma)'] for t in type_details]) if type_details else np.nan)
        out['Standard Annual Sharpe'] = fe_sharpe(r)
        out['PSR (IID)'] = fe_psr(r)
        costs, traded = fe_costs(frame)
        out.update(costs)
        out['Daily Expected Shortfall 95%'] = fe_es(r)
        details[sid] = {'Saved Row': row.to_dict(), 'Daily Results': frame,
                        'Daily Traded Notional': traded, 'Stock Removals': removal_df,
                        'Rolling Periods': pd.DataFrame(periods), 'Neighbours': neighbour_df,
                        'Unseen Returns': ur, 'Target Scores': pd.DataFrame(type_details)}
        # Keep supplied metrics, including upstream calculations not present in the
        # reference script. Retain rerun versions separately for inspection.
        details[sid]['Recomputed Summary'] = out.copy()
        for destination, source in FE_EXISTING_COLUMNS.items():
            if source in row and pd.notna(row[source]):
                out[destination] = row[source]
        if 'Relative Target Score' in out:
            out.pop('Mean Relative Target Score (sigma)', None)
        rows.append(out)

    # Estimate daily Sharpe dispersion from an unbiased sample of the FULL saved grid.
    # Do not use the legacy Sharpe column or only the ten survivors for DSR.
    unique_grid = grid.drop_duplicates(FE_SETTINGS)
    trials = len(unique_grid) if FE_DSR_TRIALS is None else int(FE_DSR_TRIALS)
    if trials < len(unique_grid):
        raise ValueError('FE_DSR_TRIALS must cover at least the full saved unique grid.')
    sample = unique_grid.sample(min(FE_DSR_SAMPLE_SIZE, len(unique_grid)), random_state=FE_SEED)
    trial_rows = []
    reference_index = next(iter(streams.values())).index
    for number, (_, row) in enumerate(sample.iterrows(), 1):
        sid = row['Simulation ID']
        log.info('DSR calibration rerun %d/%d', number, len(sample))
        rr = streams[sid] if sid in streams else fe_series(fe_run_scores(row, fe_scores(row, stocks)))
        if not rr.index.equals(reference_index):
            raise ValueError('DSR calibration histories differ.')
        trial_rows.append({'Simulation ID': sid, 'Daily Sharpe': fe_sharpe(rr, annual=False)})
    trial_df = pd.DataFrame(trial_rows)
    sd = trial_df['Daily Sharpe'].std()
    threshold = np.nan
    if trials == 1:
        threshold = 0.0
    elif np.isfinite(sd) and sd > 0 and trial_df['Daily Sharpe'].notna().all():
        normal = NormalDist()
        gamma = 0.5772156649015329
        threshold = sd * ((1 - gamma) * normal.inv_cdf(1 - 1 / trials)
                          + gamma * normal.inv_cdf(1 - 1 / (trials * np.e)))
    result = pd.DataFrame(rows).set_index('Simulation ID')
    result['DSR (estimated, IID)'] = [fe_psr(streams[sid], threshold) for sid in result.index]
    returns = pd.concat(streams, axis=1)
    if returns.isna().any().any():
        raise ValueError('Finalist dates differ; correlations must use common coverage.')
    corr = returns.corr(min_periods=30)
    corr = corr.mask(np.eye(len(corr), dtype=bool))
    result['Average Finalist Correlation'] = corr.mean().reindex(result.index)
    result = result.reset_index()
    result.attrs.update({
        'cost_bps_per_buy_plus_sell_notional': FE_COST_BPS,
        'annual_risk_free_rate': FE_RF_ANNUAL,
        'DSR_trial_count': trials, 'DSR_calibration_sample_size': len(sample),
        'DSR_daily_Sharpe_dispersion': sd, 'DSR_daily_threshold': threshold,
        'DSR_assumption': 'Saved grid trials treated as independent; earlier research not inferred.',
        'PSR_DSR_assumption': 'IID moment approximation; serial dependence is not corrected.',
        'turnover_convention': 'Half absolute target-weight changes, initial entry included; no drift or final liquidation.',
        'cost_model': 'Linear target-weight trade cost; illustrative, no liquidity/size-dependent impact.',
        'legacy_conventions': 'Core Sharpe = total return / annual volatility; initial NAV excluded from DD peak; helper uses current-date Return with next-date label.',
        'rolling_quality': '252-row windows, stride 21; original full-history benchmark denominators.',
        'neighbourhood_pass': 'Fresh 30 draws; positive compounded return after configured 1x costs; original combined score preserved.',
        'unseen_pass': 'One portfolio-level original gate; not a per-stock success percentage.',
        'units': 'Returns, drawdowns, ES and pass rates are fractions; turnover is multiples/year.',
    })
    return result, details, returns, corr, trial_df



def main(argv=None):
    global BACKTEST_DATABASE, SIMULATION_RESULTS_DATABASE
    global stocks, market, horizon_score_configurations
    global market_return, market_sharpe, market_max_drawdown, market_average_drawdown
    global FE_COST_BPS, FE_DSR_SAMPLE_SIZE, FE_DSR_TRIALS, FE_RF_ANNUAL
    global FE_NEIGHBOURHOOD_SD, FE_UNSEEN_GATE

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path(
        '/Users/sam/Progressive-Projects/Projects/Equity Selector/data'))
    parser.add_argument('--output-dir', type=Path, default=None)
    parser.add_argument('--cost-bps', type=float, default=FE_COST_BPS)
    parser.add_argument('--risk-free-rate', type=float, default=FE_RF_ANNUAL)
    parser.add_argument('--dsr-sample-size', type=int, default=FE_DSR_SAMPLE_SIZE)
    parser.add_argument('--dsr-trials', type=int, default=FE_DSR_TRIALS)
    parser.add_argument('--neighbourhood-sd', type=float, default=FE_NEIGHBOURHOOD_SD,
                        help='Optional original random-neighbour cohort SD; never estimate from finalists.')
    parser.add_argument('--unseen-gate', type=float, default=FE_UNSEEN_GATE)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s')
    FE_COST_BPS, FE_RF_ANNUAL = args.cost_bps, args.risk_free_rate
    FE_DSR_SAMPLE_SIZE, FE_DSR_TRIALS = args.dsr_sample_size, args.dsr_trials
    FE_NEIGHBOURHOOD_SD, FE_UNSEEN_GATE = args.neighbourhood_sd, args.unseen_gate
    BACKTEST_DATABASE = args.data_dir / 'Backtest_Database.db'
    SIMULATION_RESULTS_DATABASE = args.data_dir / 'Portfolio_Simulation_Results.db'
    horizons_file = args.data_dir / 'Top_Horizon_Scores.txt'
    for path in (BACKTEST_DATABASE, SIMULATION_RESULTS_DATABASE, horizons_file):
        if not path.is_file():
            raise FileNotFoundError(path)
    horizon_score_configurations = ast.literal_eval(horizons_file.read_text())
    if not isinstance(horizon_score_configurations, list) or not all(
            isinstance(c, dict) for c in horizon_score_configurations):
        raise TypeError('Top_Horizon_Scores.txt must contain a list of dictionaries.')
    with sqlite3.connect(BACKTEST_DATABASE.resolve().as_uri() + '?mode=ro', uri=True) as connection:
        market = pd.read_sql_query('SELECT * FROM "Market"', connection)
        stocks = pd.read_sql_query('SELECT * FROM "Stocks"', connection)
    for frame in (market, stocks):
        frame['Date'] = pd.to_datetime(frame['Date'])
        if frame.empty:
            raise ValueError('Market and Stocks must contain data.')

    # Exact original benchmark setup, including its legacy Sharpe/DD definitions.
    benchmark = market.groupby('Date', as_index=False).agg(Return=('Return', 'first'))
    equity = (1 + benchmark['Return']).cumprod()
    market_return = equity.iloc[-1] - 1
    market_sharpe = market_return / (benchmark['Return'].std() * np.sqrt(252))
    drawdown = (equity - equity.cummax()) / equity.cummax()
    market_average_drawdown = drawdown.mean()
    market_max_drawdown = drawdown.min()

    evaluation, details, returns, correlations, calibration = build_final_strategy_evaluation()
    # Separate output database: input databases are opened read-only throughout.
    output_dir = args.output_dir or args.data_dir / 'Final Evaluation'
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluation.to_csv(output_dir / 'Final_Strategy_Evaluation.csv', index=False)
    evaluation.to_pickle(output_dir / 'Final_Strategy_Evaluation.pkl')
    returns.to_csv(output_dir / 'Final_Strategy_Daily_Returns.csv', index_label='Date')
    correlations.to_csv(output_dir / 'Final_Strategy_Correlations.csv')
    calibration.to_csv(output_dir / 'DSR_Calibration.csv', index=False)
    pd.to_pickle(details, output_dir / 'Final_Strategy_Evaluation_Details.pkl')
    with sqlite3.connect(output_dir / 'Final_Strategy_Evaluation.db') as connection:
        evaluation.to_sql('Final Strategy Evaluation', connection, if_exists='replace', index=False)
    # Convert nonfinite floats to JSON null; pickle retains the original attributes.
    metadata = {key: (None if isinstance(value, (float, np.floating)) and not np.isfinite(value)
                      else value.item() if isinstance(value, np.generic) else value)
                for key, value in evaluation.attrs.items()}
    (output_dir / 'Evaluation_Assumptions.json').write_text(json.dumps(metadata, indent=2))
    print(evaluation.to_string(index=False))
    print(f'\nSaved final evaluation to: {output_dir.resolve()}')
    return evaluation


if __name__ == '__main__':
    final_strategy_evaluation = main()

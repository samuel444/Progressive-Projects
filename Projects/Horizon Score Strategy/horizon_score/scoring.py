"""Shared daily target scoring and horizon refresh used by simulation stages."""

import numpy as np
import pandas as pd

PORTFOLIO_RANKING_TYPES = {"ALPHA", "RELATIVE_ALPHA", "RISK_ADJUSTED_ALPHA", "CROSS_SECTION_ALPHA"}

PORTFOLIO_DIRECTION_TYPES = {"DIRECTION", "DIRECTION_MULTICLASS", "ALPHA_BINARY", "BARRIER_ALPHA"}

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


def build_score_stocks_with_direction(dataframe):
    """
    Aggregate target contributions into one daily stock score
    and apply the direction gate.

    Negative direction forces Stock_Score to be non-positive,
    causing portfolio_returns_from_scores() to assign zero weight.
    """
    required_columns = {"Date", "Ticker", "Return", "Portfolio Target Type", "Contribution"}
    missing_columns = required_columns - set(dataframe.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")
    current_dataframe = (
        dataframe[["Date", "Ticker", "Return", "Portfolio Target Type", "Contribution"]]
        .dropna(subset=["Date", "Ticker", "Contribution"])
        .copy()
    )
    current_dataframe["Direction Contribution"] = current_dataframe["Contribution"].where(
        current_dataframe["Portfolio Target Type"].isin(PORTFOLIO_DIRECTION_TYPES), 0.0
    )
    type_score_stocks = current_dataframe.groupby(
        ["Date", "Ticker", "Portfolio Target Type"], as_index=False
    ).agg(
        Contribution_Sum=("Contribution", "sum"),
        Direction_Sum=("Direction Contribution", "sum"),
        Return=("Return", "first"),
    )
    score_stocks = type_score_stocks.groupby(["Date", "Ticker"], as_index=False).agg(
        Stock_Score=("Contribution_Sum", "sum"),
        Stock_Direction=("Direction_Sum", "mean"),
        Return=("Return", "first"),
    )
    negative_direction = score_stocks["Stock_Direction"] < 0
    score_stocks.loc[negative_direction, "Stock_Score"] = -score_stocks.loc[
        negative_direction, "Stock_Score"
    ].abs()
    return score_stocks


def apply_horizon_signal_refresh(predictions_df, rebalance_multiplier):
    if not 0 <= rebalance_multiplier <= 1:
        raise ValueError("rebalance_multiplier must be greater than 0 and no greater than 1.")
    required_columns = {"Date", "Ticker", "Portfolio Target Type", "Horizon Key", "Contribution"}
    missing_columns = required_columns - set(predictions_df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")
    refreshed = (
        predictions_df.copy()
        .sort_values(["Ticker", "Portfolio Target Type", "Horizon Key", "Date"])
        .reset_index(drop=True)
    )
    group_columns = ["Ticker", "Portfolio Target Type", "Horizon Key"]
    for group_values, group_indexes in refreshed.groupby(group_columns, sort=False).groups.items():
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
        if horizon_days <= 0:
            raise ValueError("Horizon days must be positive")
        refresh_rows = max(1, int(np.ceil(rebalance_multiplier * horizon_days)))
        group_indexes = np.asarray(list(group_indexes))
        original_signals = refreshed.loc[group_indexes, "Contribution"].to_numpy()
        row_positions = np.arange(len(original_signals))
        refresh_start_positions = row_positions // refresh_rows * refresh_rows
        refreshed.loc[group_indexes, "Contribution"] = original_signals[refresh_start_positions]
    return refreshed.sort_values(
        ["Date", "Ticker", "Portfolio Target Type", "Horizon Key"]
    ).reset_index(drop=True)


def add_horizon_scores(dataframe, horizon_score_configuration):
    """
    Add the Horizon Score column for one horizon-score configuration.
    """
    horizon_score_rows = []
    for portfolio_type, horizon_values in horizon_score_configuration.items():
        for horizon_key, horizon_score in horizon_values.items():
            horizon_score_rows.append(
                {
                    "Portfolio Target Type": portfolio_type,
                    "Horizon Key": horizon_key,
                    "Horizon Score": float(horizon_score),
                }
            )
    horizon_score_df = pd.DataFrame(horizon_score_rows)
    result = dataframe.drop(columns=["Horizon Score"], errors="ignore").merge(
        horizon_score_df,
        on=["Portfolio Target Type", "Horizon Key"],
        how="left",
        validate="many_to_one",
    )
    return result


def add_type_scores(dataframe, type_score_configuration):
    """
    Add the Type Score column for one portfolio-group configuration.
    """
    portfolio_type_score_map = {}
    for portfolio_type in PORTFOLIO_RANKING_TYPES:
        portfolio_type_score_map[portfolio_type] = type_score_configuration["Ranking"]
    for portfolio_type in PORTFOLIO_DIRECTION_TYPES:
        portfolio_type_score_map[portfolio_type] = type_score_configuration["Direction"]
    for portfolio_type in PORTFOLIO_RISK_TYPES:
        portfolio_type_score_map[portfolio_type] = type_score_configuration["Risk"]
    for portfolio_type in PORTFOLIO_OPPORTUNITY_TYPES:
        portfolio_type_score_map[portfolio_type] = type_score_configuration["Opportunity"]
    for portfolio_type in PORTFOLIO_SPECIAL_TYPES:
        portfolio_type_score_map[portfolio_type] = type_score_configuration["Special"]
    result = dataframe.copy()
    result["Type Score"] = result["Portfolio Target Type"].map(portfolio_type_score_map)
    return result

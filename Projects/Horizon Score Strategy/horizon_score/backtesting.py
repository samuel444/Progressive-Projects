import numpy as np
import pandas as pd
from horizon_score.portfolio import portfolio_returns_from_scores
def benchmark_metrics(df):
    """
    Calculate S&P 500 benchmark performance metrics.

    Required columns:
        Close
        Return

    Returns:
        total_return
        average_drawdown
        max_drawdown
        sharpe_ratio
    """

    close = df["Close"].dropna()
    returns = df["Return"].dropna()

    ########################################
    # Total Return
    ########################################

    total_return = (close.iloc[-1] / close.iloc[0]) - 1

    ########################################
    # Drawdown
    ########################################

    running_max = close.cummax()

    drawdown = (close / running_max) - 1

    average_drawdown = drawdown.mean()

    max_drawdown = drawdown.min()

    ########################################
    # Annualised Sharpe Ratio
    ########################################

    if returns.std() == 0:
        sharpe_ratio = 0.0

    else:
        sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252)

    ########################################
    # Results
    ########################################

    return {
        "Return": total_return,
        "Average Drawdown": average_drawdown,
        "Max Drawdown": max_drawdown,
        "Sharpe Ratio": sharpe_ratio,
    }


DEFAULT_TYPE_VALUES = {
    "ALPHA": 1.00,
    "RELATIVE_ALPHA": 1.15,
    "CROSS_SECTION_ALPHA": 0.80,
    "DIRECTION": 0.50,
    "ALPHA_BINARY": 0.70,
    "BARRIER_ALPHA": 0.80,
    "VOLATILITY": -0.70,
    "VOLATILITY_EVENT": -0.80,
    "DOWNSIDE": -1.00,
    "TAIL_RISK": -1.20,
    "TAIL_EVENT": -1.30,
    "UPSIDE_RISK": 0.20,
    "UPSIDE_EVENT": 0.80,
    "RECOVERY": 0.60,
    "REVERSAL": -0.50,
    "REGIME": 0.30,
    "CORRELATION": -0.50,
    "COVARIANCE": -0.60,
    "LIQUIDITY": 0.40,
    "EXECUTION": 0.30,
    "MARKET_IMPACT": -0.50,
}




def run_portfolio_backtest_from_predictions(
    predictions_df,
    type_values=None,
    max_weight=0.30,
    concentration_penalty=0.10,
    trading_fee=0.0,
    annualisation=252,
):
    """
    Run the portfolio backtest using cached predictions only.

    NO model fitting and NO prediction generation happen here.

    To test another horizon-score combination, change the
    ``Horizon Score`` column in predictions_df and call again.
    """
    required = {
        "Date",
        "Ticker",
        "Return",
        "Signal",
        "Direction Signal",
        "Horizon Score",
        "Portfolio Target Type",
    }
    missing = required.difference(predictions_df.columns)
    if missing:
        raise ValueError("predictions_df is missing columns: " + ", ".join(sorted(missing)))

    predictions = predictions_df.copy()
    predictions["Date"] = pd.to_datetime(predictions["Date"])
    predictions["Signal"] = pd.to_numeric(predictions["Signal"], errors="coerce")
    predictions["Direction Signal"] = pd.to_numeric(
        predictions["Direction Signal"], errors="coerce"
    )
    predictions["Horizon Score"] = pd.to_numeric(
        predictions["Horizon Score"], errors="coerce"
    ).clip(0.0, 1.0)

    predictions["Contribution"] = predictions["Signal"] * predictions["Horizon Score"]

    predictions["Direction Contribution"] = (
        predictions["Direction Signal"] * predictions["Horizon Score"]
    )

    type_value_map = DEFAULT_TYPE_VALUES.copy()
    if type_values is not None:
        type_value_map.update(
            {str(key).upper().strip(): float(value) for key, value in type_values.items()}
        )

    valid = predictions["Signal"].notna()

    predictions = (
        predictions.loc[valid]
        .groupby(["Date", "Ticker", "Portfolio Target Type"], as_index=False)
        .agg(
            Contribution_Sum=("Contribution", "sum"),
            Direction_Sum=("Direction Contribution", "sum"),
            Return=("Return", "first"),
        )
    )

    BASE_TYPE_SCORES = pd.Series(
        {
            "ALPHA": 0.55,
            "RELATIVE_ALPHA": 0.55,
            "RISK_ADJUSTED_ALPHA": 0.60,
            "CROSS_SECTION_ALPHA": 0.60,
            "CROSS_SECTION_DOWNSIDE": 0.55,
            "DIRECTION": 0.55,
            "DIRECTION_MULTICLASS": 0.50,
            "ALPHA_BINARY": 0.50,
            "BARRIER_ALPHA": 0.50,
            "VOLATILITY": 0.55,
            "ABSOLUTE_MOVE": 0.50,
            "UPSIDE_VOLATILITY": 0.45,
            "DOWNSIDE_VOLATILITY": 0.55,
            "VOLATILITY_ASYMMETRY": 0.50,
            "VOLATILITY_EVENT": 0.50,
            "DOWNSIDE": 0.60,
            "TAIL_RISK": 0.60,
            "TAIL_EVENT": 0.55,
            "UPSIDE_RISK": 0.45,
            "UPSIDE_EVENT": 0.45,
            "UPSIDE_EXCURSION": 0.50,
            "DOWNSIDE_EXCURSION": 0.55,
            "TIME_TO_UPSIDE_EXCURSION": 0.45,
            "TIME_TO_DOWNSIDE_EXCURSION": 0.50,
            "RECOVERY": 0.50,
            "REVERSAL": 0.50,
            "REGIME": 0.55,
            "CORRELATION": 0.50,
            "COVARIANCE": 0.50,
            "LIQUIDITY": 0.50,
            "EXECUTION": 0.50,
            "MARKET_IMPACT": 0.50,
        },
        name="Base Type Score",
        dtype=float,
    )

    BASE_TYPE_SCORES.index.name = "Portfolio Target Type"

    predictions = predictions.merge(
        BASE_TYPE_SCORES,
        how="left",
        left_on="Portfolio Target Type",
        right_index=True,
        validate="many_to_one",
    )

    if type_values is not None:
        overrides = {str(key).upper().strip(): float(value) for key, value in type_values.items()}
        predictions["Base Type Score"] = (
            predictions["Portfolio Target Type"]
            .map(overrides)
            .fillna(predictions["Base Type Score"])
        )
    predictions["Type Score"] = predictions["Contribution_Sum"] * predictions["Base Type Score"]

    predictions = predictions.groupby(["Date", "Ticker"], as_index=False).agg(
        Stock_Score=("Type Score", "sum"),
        Stock_Direction=("Direction_Sum", "mean"),
        Return=("Return", "first"),
    )

    negative_direction = predictions["Stock_Direction"] < 0

    predictions.loc[
        negative_direction,
        "Stock_Score",
    ] = -predictions.loc[
        negative_direction,
        "Stock_Score",
    ].abs()

    backtest = portfolio_returns_from_scores(
        predictions,
        max_weight=max_weight,
        concentration_penalty=concentration_penalty,
        trading_fee=trading_fee,
    )
    from equity_selector.metrics import performance_metrics

    metrics = performance_metrics(backtest["Return"], annualisation)
    return {
        "Strategy Return": metrics["Return"],
        "Average Drawdown": metrics["Average Drawdown"],
        "Max Drawdown": metrics["Max Drawdown"],
        "Sharpe Ratio": metrics["Sharpe Ratio"],
    }

"""Performance metrics on periodic simple returns, with initial wealth 1."""

import numpy as np
import pandas as pd


def performance_metrics(returns, annualisation=252):
    if not np.isfinite(annualisation) or annualisation <= 0:
        raise ValueError("annualisation must be positive")
    returns = pd.to_numeric(pd.Series(returns, copy=True), errors="raise")
    if returns.empty:
        return {
            "Return": 0.0,
            "Volatility": np.nan,
            "Sharpe Ratio": np.nan,
            "Average Drawdown": 0.0,
            "Max Drawdown": 0.0,
        }
    if not np.isfinite(returns).all() or (returns < -1).any():
        raise ValueError("Returns must be finite and at least -1")
    wealth = (1 + returns).cumprod()
    drawdown = wealth / wealth.cummax().clip(lower=1.0) - 1
    std = returns.std(ddof=1)
    sharpe = float(returns.mean() / std * np.sqrt(annualisation)) if std > 0 else np.nan
    return {
        "Return": float(wealth.iloc[-1] - 1),
        "Volatility": float(std * np.sqrt(annualisation)),
        "Sharpe Ratio": sharpe,
        "Average Drawdown": float(drawdown.mean()),
        "Max Drawdown": float(drawdown.min()),
    }


def relative_metrics(frame, benchmark, annualisation=252):
    """Legacy nine-field simulation result, without mutating caller data.

    Preserve the existing relative-score convention and coefficients; undefined
    zero-volatility Sharpe remains NaN instead of inventing a finite estimate.
    """
    metrics = performance_metrics(frame["Return"], annualisation)
    keys = ["Return", "Sharpe Ratio", "Max Drawdown", "Average Drawdown"]
    relative = []
    for key in keys:
        value, reference = metrics[key], benchmark[key]
        denominator = abs(value) + abs(reference)
        relative.append(2 * value / denominator if denominator else 0.0)
    quality = sum(w * value for w, value in zip([0.35, 0.25, 0.25, 0.15], relative))
    return (
        metrics["Return"],
        metrics["Sharpe Ratio"],
        metrics["Average Drawdown"],
        metrics["Max Drawdown"],
        *relative,
        quality,
    )

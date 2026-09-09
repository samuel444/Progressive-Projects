"""Capped, long-only allocation using scores known before the return period."""

import logging
from time import perf_counter

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def capped_weights(scores, max_weight=0.30, concentration_penalty=0.10):
    if not np.isfinite(max_weight) or not 0 < max_weight <= 1:
        raise ValueError("max_weight must be in (0, 1]")
    if not np.isfinite(concentration_penalty) or not 0 <= concentration_penalty <= 1:
        raise ValueError("concentration_penalty must be in [0, 1]")
    scores = (
        pd.to_numeric(scores, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
        .clip(lower=0)
    )
    if not scores.index.is_unique:
        raise ValueError("Ticker must be unique within each date")
    scores = scores[scores > 0]
    if scores.empty:
        return pd.Series(dtype=float)
    # Scale first to avoid overflow when adding large finite scores.
    scores = scores / scores.max()
    desired = (1 - concentration_penalty) * scores / scores.sum() + concentration_penalty / len(
        scores
    )
    weights = pd.Series(0.0, index=desired.index)
    remaining = desired.index
    capital = min(1.0, len(remaining) * max_weight)
    while len(remaining) and capital > 1e-12:
        proposed = desired.loc[remaining] / desired.loc[remaining].sum() * capital
        capped = proposed.index[proposed > max_weight]
        if capped.empty:
            weights.loc[remaining] = proposed
            break
        weights.loc[capped] = max_weight
        capital -= len(capped) * max_weight
        remaining = remaining.difference(capped, sort=False)
    return weights[weights > 0]


def portfolio_returns_from_scores(
    dataframe, max_weight=0.30, concentration_penalty=0.10, trading_fee=0.0, missing_return="zero"
):
    """Scores at t earn backward-looking Return at t+1.

    Cash earns zero. Fees use total buy-plus-sell target-weight notional,
    including initial entry. trading_fee is a fraction: 0.001 means 0.1%. `missing_return='raise'` supports strict
    audits; default zero preserves the existing missing-price convention.
    """
    started = perf_counter()
    required = {"Date", "Ticker", "Return", "Stock_Score"}
    if missing := required.difference(dataframe.columns):
        raise ValueError("Missing columns: " + ", ".join(sorted(missing)))
    capped_weights(pd.Series(dtype=float), max_weight, concentration_penalty)
    if not np.isfinite(trading_fee) or trading_fee < 0:
        raise ValueError("trading_fee must be finite and nonnegative")
    if missing_return not in {"zero", "raise"}:
        raise ValueError("missing_return must be 'zero' or 'raise'")
    data = dataframe[["Date", "Ticker", "Return", "Stock_Score"]].copy()
    data["Date"] = pd.to_datetime(data["Date"], errors="raise")
    if data[["Date", "Ticker"]].isna().any().any():
        raise ValueError("Date and Ticker must not be missing")
    if data.duplicated(["Date", "Ticker"]).any():
        raise ValueError("Duplicate Date/Ticker observations")
    data["Return"] = pd.to_numeric(data["Return"], errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    if data["Return"].lt(-1).any():
        raise ValueError("Simple returns must be at least -1")
    data = data.sort_values(["Date", "Ticker"])
    groups = list(data.groupby("Date", sort=True))
    tickers = sorted(data["Ticker"].unique())
    records = []
    previous = pd.Series(dtype=float)
    missing_count = 0
    for (_, today), (date, tomorrow) in zip(groups, groups[1:]):
        weights = capped_weights(
            today.set_index("Ticker")["Stock_Score"], max_weight, concentration_penalty
        )
        union = previous.index.union(weights.index)
        turnover = (
            (weights.reindex(union, fill_value=0) - previous.reindex(union, fill_value=0))
            .abs()
            .sum()
        )
        returns = tomorrow.set_index("Ticker")["Return"].reindex(weights.index)
        missing_count += int(returns.isna().sum())
        if returns.isna().any() and missing_return == "raise":
            raise ValueError(f"Missing held return on {date}")
        net_return = float((weights * returns.fillna(0)).sum() - turnover * trading_fee)
        records.append(
            {
                "Date": date,
                "Return": net_return,
                **weights.reindex(tickers, fill_value=0.0).to_dict(),
            }
        )
        previous = weights
    result = pd.DataFrame(records, columns=["Date", "Return", *tickers])
    logger.debug(
        "Portfolio complete | dates=%d securities=%d missing held returns=%d elapsed=%.3fs",
        len(result),
        len(tickers),
        missing_count,
        perf_counter() - started,
    )
    return result

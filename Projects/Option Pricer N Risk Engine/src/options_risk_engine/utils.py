
"""Small reusable helpers with no project-layer dependencies."""

from __future__ import annotations

import numpy as np
import pandas as pd

def safe_relative_edge(
    model_value: pd.Series,
    market_value: pd.Series,
) -> pd.Series:
    """Calculate (model - market) / market without creating infinities."""

    model_value = pd.to_numeric(model_value, errors="coerce")
    market_value = pd.to_numeric(market_value, errors="coerce")

    valid_market = market_value > 0

    edge = pd.Series(
        np.nan,
        index=model_value.index,
        dtype=float,
    )

    edge.loc[valid_market] = (
        model_value.loc[valid_market]
        - market_value.loc[valid_market]
    ) / market_value.loc[valid_market]

    return edge


def flatten_dataframe_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with SQL-safe, one-level string column names."""
    result = frame.copy()
    if isinstance(result.columns, pd.MultiIndex):
        result.columns = [
            "__".join(str(part) for part in column if str(part) != "")
            for column in result.columns
        ]
    else:
        result.columns = [str(column) for column in result.columns]
    return result


def require_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"{name} is missing columns: {sorted(missing)}")

import numpy as np


def future_maximum_drawdown(df, horizons=(20, 60, 120)):
    if isinstance(horizons, int):
        horizons = [horizons]

    prices = df["Close"].to_numpy(dtype=float)

    for horizon in horizons:
        values = np.full(len(df), np.nan)

        for i in range(len(df) - horizon):
            path = prices[i:i + horizon + 1]
            peak = np.maximum.accumulate(path)
            drawdown = path / peak - 1
            values[i] = np.min(drawdown)

        df[f"Future Maximum Drawdown {horizon}"] = values

    return df


def future_minimum_return(df, horizons=(5, 20, 60)):
    if isinstance(horizons, int):
        horizons = [horizons]

    prices = df["Close"].to_numpy(dtype=float)

    for horizon in horizons:
        values = np.full(len(df), np.nan)

        for i in range(len(df) - horizon):
            path = prices[i + 1:i + horizon + 1] / prices[i] - 1
            values[i] = np.min(path)

        df[f"Future Minimum Return {horizon}"] = values

    return df


def future_drawdown_at_horizon(df, horizons=(5, 20, 60)):
    if isinstance(horizons, int):
        horizons = [horizons]

    for horizon in horizons:
        future_price = df["Close"].shift(-horizon)
        current_to_future_high = df["Close"][::-1].rolling(horizon + 1).max()[::-1]
        df[f"Future Drawdown At Horizon {horizon}"] = future_price / current_to_future_high - 1

    return df


def all_drawdown_targets(df):
    df = future_maximum_drawdown(df)
    df = future_minimum_return(df)

    return df

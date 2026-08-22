import numpy as np


def future_volatility(df, horizons=(5, 10, 20, 60), annualize=False):
    if isinstance(horizons, int):
        horizons = [horizons]

    returns = df["Close"].pct_change()
    multiplier = np.sqrt(252) if annualize else 1

    for horizon in horizons:
        df[f"Future Volatility {horizon}"] = returns.shift(-1)[::-1].rolling(horizon).std()[::-1] * multiplier

    return df


def future_variance(df, horizons=(5, 20, 60)):
    if isinstance(horizons, int):
        horizons = [horizons]

    returns = df["Close"].pct_change()

    for horizon in horizons:
        df[f"Future Variance {horizon}"] = returns.shift(-1)[::-1].rolling(horizon).var()[::-1]

    return df


def future_upside_downside_volatility(df, horizons=(10, 20, 60)):
    if isinstance(horizons, int):
        horizons = [horizons]

    returns = df["Close"].pct_change()

    for horizon in horizons:
        shifted = returns.shift(-1)[::-1]
        minimum = max(2, horizon // 4)
        upside = shifted.where(shifted > 0).rolling(horizon, min_periods=minimum).std()[::-1]
        downside = shifted.where(shifted < 0).rolling(horizon, min_periods=minimum).std()[::-1]

        df[f"Future Upside Volatility {horizon}"] = upside
        df[f"Future Downside Volatility {horizon}"] = downside
        df[f"Future Downside Upside Volatility Ratio {horizon}"] = downside / upside

    return df


def future_absolute_return(df, horizons=(5, 20, 60)):
    if isinstance(horizons, int):
        horizons = [horizons]

    returns = df["Close"].pct_change().abs()

    for horizon in horizons:
        df[f"Future Mean Absolute Return {horizon}"] = returns.shift(-1)[::-1].rolling(horizon).mean()[::-1]
        df[f"Future Maximum Absolute Return {horizon}"] = returns.shift(-1)[::-1].rolling(horizon).max()[::-1]

    return df


def all_volatility_targets(df):
    df = future_volatility(df)
    df = future_variance(df)
    df = future_upside_downside_volatility(df)
    df = future_absolute_return(df)

    return df

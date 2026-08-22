import numpy as np


def _close_frame(df):
    close = df["Close"]

    if getattr(close, "ndim", 1) != 2:
        raise ValueError("Dispersion features require Close data for multiple stocks")

    return close


def return_dispersion(df, periods=(1, 5, 20)):
    if isinstance(periods, int):
        periods = [periods]

    close = _close_frame(df)

    for period in periods:
        returns = close.pct_change(period)
        df[f"Return Dispersion {period}"] = returns.std(axis=1)
        df[f"Return IQR Dispersion {period}"] = returns.quantile(0.75, axis=1) - returns.quantile(0.25, axis=1)

    return df


def momentum_dispersion(df, windows=(20, 60, 120, 252)):
    if isinstance(windows, int):
        windows = [windows]

    close = _close_frame(df)

    for window in windows:
        momentum = close.pct_change(window)
        df[f"Momentum Dispersion {window}"] = momentum.std(axis=1)

    return df


def volatility_dispersion(df, windows=(20, 60)):
    if isinstance(windows, int):
        windows = [windows]

    close = _close_frame(df)
    returns = close.pct_change()

    for window in windows:
        volatility = returns.rolling(window).std()
        df[f"Volatility Dispersion {window}"] = volatility.std(axis=1)

    return df


def drawdown_dispersion(df, windows=(20, 60, 252)):
    if isinstance(windows, int):
        windows = [windows]

    close = _close_frame(df)

    for window in windows:
        drawdown = close / close.rolling(window).max() - 1
        df[f"Drawdown Dispersion {window}"] = drawdown.std(axis=1)

    return df


def all_dispersion_features(df):
    df = return_dispersion(df)
    df = momentum_dispersion(df)
    df = volatility_dispersion(df)
    df = drawdown_dispersion(df)

    return df

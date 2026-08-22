import numpy as np


def current_drawdown(df, windows=(20, 60, 120, 252)):
    if isinstance(windows, int):
        windows = [windows]

    for window in windows:
        rolling_high = df["Close"].rolling(window).max()
        df[f"Drawdown {window}"] = df["Close"] / rolling_high - 1

    return df


def maximum_drawdown(df, windows=(20, 60, 120, 252)):
    if isinstance(windows, int):
        windows = [windows]

    def calculate(values):
        peak = np.maximum.accumulate(values)
        drawdown = values / peak - 1
        return np.nanmin(drawdown)

    for window in windows:
        df[f"Maximum Drawdown {window}"] = df["Close"].rolling(window).apply(calculate, raw=True)

    return df


def distance_from_high_low(df, windows=(5, 20, 60, 120, 252)):
    if isinstance(windows, int):
        windows = [windows]

    for window in windows:
        rolling_high = df["Close"].rolling(window).max()
        rolling_low = df["Close"].rolling(window).min()

        df[f"Distance From High {window}"] = df["Close"] / rolling_high - 1
        df[f"Distance From Low {window}"] = df["Close"] / rolling_low - 1

    return df


def days_since_high_low(df, windows=(20, 60, 252)):
    if isinstance(windows, int):
        windows = [windows]

    def since_high(values):
        return len(values) - 1 - np.argmax(values)

    def since_low(values):
        return len(values) - 1 - np.argmin(values)

    for window in windows:
        df[f"Days Since High {window}"] = df["Close"].rolling(window).apply(since_high, raw=True)
        df[f"Days Since Low {window}"] = df["Close"].rolling(window).apply(since_low, raw=True)

    return df


def drawdown_duration(df):
    running_high = df["Close"].cummax()
    in_drawdown = df["Close"] < running_high
    groups = (~in_drawdown).cumsum()
    duration = in_drawdown.groupby(groups).cumsum()

    df["Drawdown Duration"] = duration

    return df


def drawdown_change(df, windows=(20, 60, 252), periods=(1, 5, 20)):
    if isinstance(windows, int):
        windows = [windows]

    if isinstance(periods, int):
        periods = [periods]

    for window in windows:
        drawdown = df["Close"] / df["Close"].rolling(window).max() - 1

        for period in periods:
            df[f"Drawdown {window} Change {period}"] = drawdown.diff(period)

    return df


def all_drawdown_features(df):
    df = current_drawdown(df)
    df = maximum_drawdown(df)
    df = distance_from_high_low(df)
    df = days_since_high_low(df)
    df = drawdown_duration(df)
    df = drawdown_change(df)

    return df

import numpy as np


def _close_frame(df):
    close = df["Close"]

    if getattr(close, "ndim", 1) != 2:
        raise ValueError("Breadth features require Close data for multiple stocks")

    return close


def breadth_ma(df, windows=(20, 50, 200)):
    if isinstance(windows, int):
        windows = [windows]

    close = _close_frame(df)

    for window in windows:
        moving_average = close.rolling(window).mean()
        valid = close.rolling(window).count() == window
        above = (close > moving_average) & valid
        denominator = valid.sum(axis=1).replace(0, np.nan)

        df[f"Breadth MA {window}"] = above.sum(axis=1) / denominator

    return df


def advance_decline(df, periods=(1,)):
    if isinstance(periods, int):
        periods = [periods]

    close = _close_frame(df)

    for period in periods:
        returns = close.pct_change(period)
        advances = (returns > 0).sum(axis=1)
        declines = (returns < 0).sum(axis=1)
        valid = returns.notna().sum(axis=1).replace(0, np.nan)

        df[f"Advance Decline Difference {period}"] = (advances - declines) / valid
        df[f"Advance Decline Ratio {period}"] = advances / declines.replace(0, np.nan)

    return df


def new_high_low(df, windows=(20, 60, 252)):
    if isinstance(windows, int):
        windows = [windows]

    close = _close_frame(df)

    for window in windows:
        rolling_high = close.rolling(window).max()
        rolling_low = close.rolling(window).min()
        valid = close.rolling(window).count() == window

        new_highs = ((close == rolling_high) & valid).sum(axis=1)
        new_lows = ((close == rolling_low) & valid).sum(axis=1)
        denominator = valid.sum(axis=1).replace(0, np.nan)

        df[f"New High Breadth {window}"] = new_highs / denominator
        df[f"New Low Breadth {window}"] = new_lows / denominator
        df[f"High Low Breadth {window}"] = (new_highs - new_lows) / denominator

    return df


def positive_return_breadth(df, windows=(5, 20, 60)):
    if isinstance(windows, int):
        windows = [windows]

    close = _close_frame(df)

    for window in windows:
        returns = close.pct_change(window)
        valid = returns.notna()
        positive = (returns > 0) & valid
        denominator = valid.sum(axis=1).replace(0, np.nan)
        df[f"Positive Return Breadth {window}"] = positive.sum(axis=1) / denominator

    return df


def all_breadth_features(df):
    df = breadth_ma(df)
    df = advance_decline(df)
    df = new_high_low(df)
    df = positive_return_breadth(df)

    return df

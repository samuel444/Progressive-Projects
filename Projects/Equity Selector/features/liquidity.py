import numpy as np


def dollar_volume(df, windows=(1, 5, 20, 60)):
    if isinstance(windows, int):
        windows = [windows]

    values = df["Close"] * df["Volume"]
    df["Dollar Volume"] = values

    for window in windows:
        if window == 1:
            df["Average Dollar Volume 1"] = values
        else:
            df[f"Average Dollar Volume {window}"] = values.rolling(window).mean()

    return df


def amihud_illiquidity(df, windows=(20, 60, 252)):
    if isinstance(windows, int):
        windows = [windows]

    returns = df["Close"].pct_change().abs()
    value = df["Close"] * df["Volume"]
    daily_illiquidity = returns / value.replace(0, np.nan)

    for window in windows:
        df[f"Amihud Illiquidity {window}"] = daily_illiquidity.rolling(window).mean()

    return df


def turnover(df, shares_outstanding, windows=(1, 5, 20, 60)):
    if isinstance(windows, int):
        windows = [windows]

    daily_turnover = df["Volume"] / shares_outstanding

    for window in windows:
        if window == 1:
            df["Turnover 1"] = daily_turnover
        else:
            df[f"Turnover {window}"] = daily_turnover.rolling(window).mean()

    return df


def zero_return_frequency(df, windows=(20, 60, 252)):
    if isinstance(windows, int):
        windows = [windows]

    zero_return = (df["Close"].pct_change() == 0).astype(float)

    for window in windows:
        df[f"Zero Return Frequency {window}"] = zero_return.rolling(window).mean()

    return df


def all_liquidity_features(df, shares_outstanding=None):
    df = dollar_volume(df)
    df = amihud_illiquidity(df)
    df = zero_return_frequency(df)

    if shares_outstanding is not None:
        df = turnover(df, shares_outstanding)

    return df

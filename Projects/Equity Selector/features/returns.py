import numpy as np


def return_lags(df, periods=(1, 2, 3, 5, 10, 20)):
    if isinstance(periods, int):
        periods = [periods]

    returns = df["Close"].pct_change()

    for period in periods:
        df[f"Return Lag {period}"] = returns.shift(period - 1)

    return df


def log_return_lags(df, periods=(1, 2, 5, 10, 20)):
    if isinstance(periods, int):
        periods = [periods]

    log_returns = np.log(df["Close"] / df["Close"].shift(1))

    for period in periods:
        df[f"Log Return Lag {period}"] = log_returns.shift(period - 1)

    return df


def absolute_return_lags(df, periods=(1, 2, 5, 10, 20)):
    if isinstance(periods, int):
        periods = [periods]

    returns = df["Close"].pct_change().abs()

    for period in periods:
        df[f"Absolute Return Lag {period}"] = returns.shift(period - 1)

    return df


def squared_return_lags(df, periods=(1, 2, 5, 10, 20)):
    if isinstance(periods, int):
        periods = [periods]

    returns = df["Close"].pct_change() ** 2

    for period in periods:
        df[f"Squared Return Lag {period}"] = returns.shift(period - 1)

    return df


def rolling_mean_return(df, windows=(5, 10, 20, 60, 120, 252)):
    if isinstance(windows, int):
        windows = [windows]

    returns = df["Close"].pct_change()

    for window in windows:
        df[f"Mean Return {window}"] = returns.rolling(window).mean()
        df[f"Median Return {window}"] = returns.rolling(window).median()

    return df


def positive_negative_days(df, windows=(5, 10, 20, 60)):
    if isinstance(windows, int):
        windows = [windows]

    returns = df["Close"].pct_change()

    for window in windows:
        positive = (returns > 0).rolling(window).sum()
        negative = (returns < 0).rolling(window).sum()
        positive_return = returns.where(returns > 0)
        negative_return = returns.where(returns < 0)

        df[f"Positive Days {window}"] = positive
        df[f"Negative Days {window}"] = negative
        df[f"Positive Day Ratio {window}"] = positive / window
        df[f"Negative Day Ratio {window}"] = negative / window
        df[f"Mean Positive Return {window}"] = positive_return.rolling(window, min_periods=1).mean()
        df[f"Mean Negative Return {window}"] = negative_return.rolling(window, min_periods=1).mean()

    return df


def all_return_features(df):
    df = return_lags(df)
    df = log_return_lags(df)
    df = absolute_return_lags(df)
    df = squared_return_lags(df)
    df = rolling_mean_return(df)
    df = positive_negative_days(df)

    return df

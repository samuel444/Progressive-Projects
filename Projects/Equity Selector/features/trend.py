import numpy as np


def _slope(values):
    if np.isnan(values).any():
        return np.nan

    x = np.arange(len(values))
    return np.polyfit(x, values, 1)[0]


def _r_squared(values):
    if np.isnan(values).any():
        return np.nan

    x = np.arange(len(values))
    slope, intercept = np.polyfit(x, values, 1)
    fitted = slope * x + intercept
    total = ((values - values.mean()) ** 2).sum()

    if total == 0:
        return 0

    residual = ((values - fitted) ** 2).sum()
    return 1 - residual / total


def trend_slope(df, windows=(10, 20, 60, 120)):
    if isinstance(windows, int):
        windows = [windows]

    log_price = np.log(df["Close"])

    for window in windows:
        df[f"Trend Slope {window}"] = log_price.rolling(window).apply(_slope, raw=True)

    return df


def trend_r_squared(df, windows=(20, 60, 120)):
    if isinstance(windows, int):
        windows = [windows]

    log_price = np.log(df["Close"])

    for window in windows:
        df[f"Trend R Squared {window}"] = log_price.rolling(window).apply(_r_squared, raw=True)

    return df


def trend_strength(df, windows=(20, 60, 120)):
    if isinstance(windows, int):
        windows = [windows]

    log_price = np.log(df["Close"])

    for window in windows:
        slope = log_price.rolling(window).apply(_slope, raw=True)
        r_squared = log_price.rolling(window).apply(_r_squared, raw=True)
        df[f"Trend Strength {window}"] = slope * r_squared

    return df


def trend_efficiency(df, windows=(5, 10, 20, 60, 120)):
    if isinstance(windows, int):
        windows = [windows]

    for window in windows:
        direction = (df["Close"] - df["Close"].shift(window)).abs()
        path = df["Close"].diff().abs().rolling(window).sum()
        df[f"Trend Efficiency {window}"] = direction / path

    return df


def trend_change(df, windows=(20, 60), periods=(5, 20)):
    if isinstance(windows, int):
        windows = [windows]

    if isinstance(periods, int):
        periods = [periods]

    log_price = np.log(df["Close"])

    for window in windows:
        slope = log_price.rolling(window).apply(_slope, raw=True)

        for period in periods:
            df[f"Trend Slope {window} Change {period}"] = slope.diff(period)

    return df


def all_trend_features(df):
    df = trend_slope(df)
    df = trend_r_squared(df)
    df = trend_strength(df)
    df = trend_efficiency(df)
    df = trend_change(df)

    return df

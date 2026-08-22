import numpy as np


def high_low_range(df, windows=(5, 20, 60)):
    if isinstance(windows, int):
        windows = [windows]

    daily_range = (df["High"] - df["Low"]) / df["Close"]
    df["High Low Range"] = daily_range

    for window in windows:
        df[f"Mean High Low Range {window}"] = daily_range.rolling(window).mean()
        df[f"High Low Range Volatility {window}"] = daily_range.rolling(window).std()
        df[f"High Low Range Z Score {window}"] = (
            (daily_range - daily_range.rolling(window).mean()) / daily_range.rolling(window).std()
        )

    return df


def atr(df, windows=(5, 10, 14, 20, 60)):
    if isinstance(windows, int):
        windows = [windows]

    previous_close = df["Close"].shift(1)
    true_range = np.maximum(
        df["High"] - df["Low"],
        np.maximum((df["High"] - previous_close).abs(), (df["Low"] - previous_close).abs())
    )

    for window in windows:
        atr_value = true_range.rolling(window).mean()
        df[f"ATR {window}"] = atr_value
        df[f"ATR Percent {window}"] = atr_value / df["Close"]

    return df


def parkinson_volatility(df, windows=(20, 60, 120)):
    if isinstance(windows, int):
        windows = [windows]

    log_range = np.log(df["High"] / df["Low"]) ** 2

    for window in windows:
        df[f"Parkinson Volatility {window}"] = np.sqrt(
            log_range.rolling(window).mean() / (4 * np.log(2))
        )

    return df


def garman_klass_volatility(df, windows=(20, 60, 120)):
    if isinstance(windows, int):
        windows = [windows]

    log_hl = np.log(df["High"] / df["Low"])
    log_co = np.log(df["Close"] / df["Open"])
    variance = 0.5 * log_hl ** 2 - (2 * np.log(2) - 1) * log_co ** 2

    for window in windows:
        df[f"Garman Klass Volatility {window}"] = np.sqrt(variance.clip(lower=0).rolling(window).mean())

    return df


def rogers_satchell_volatility(df, windows=(20, 60, 120)):
    if isinstance(windows, int):
        windows = [windows]

    rs = (
        np.log(df["High"] / df["Close"]) * np.log(df["High"] / df["Open"]) +
        np.log(df["Low"] / df["Close"]) * np.log(df["Low"] / df["Open"])
    )

    for window in windows:
        df[f"Rogers Satchell Volatility {window}"] = np.sqrt(rs.clip(lower=0).rolling(window).mean())

    return df


def range_ratios(df, short_windows=(5, 20), long_windows=(20, 60)):
    if isinstance(short_windows, int):
        short_windows = [short_windows]

    if isinstance(long_windows, int):
        long_windows = [long_windows]

    daily_range = (df["High"] - df["Low"]) / df["Close"]

    for short_window in short_windows:
        for long_window in long_windows:
            if short_window >= long_window:
                continue

            short_range = daily_range.rolling(short_window).mean()
            long_range = daily_range.rolling(long_window).mean()
            df[f"Range Ratio {short_window} {long_window}"] = short_range / long_range

    return df


def all_range_volatility_features(df):
    df = high_low_range(df)
    df = atr(df)
    df = parkinson_volatility(df)
    df = garman_klass_volatility(df)
    df = rogers_satchell_volatility(df)
    df = range_ratios(df)

    return df

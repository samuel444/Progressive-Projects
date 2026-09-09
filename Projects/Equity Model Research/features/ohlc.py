import numpy as np


def gap_features(df, windows=(5, 20, 60)):
    if isinstance(windows, int):
        windows = [windows]

    gap = df["Open"] / df["Close"].shift(1) - 1
    df["Overnight Gap"] = gap
    df["Absolute Overnight Gap"] = gap.abs()

    for window in windows:
        df[f"Mean Gap {window}"] = gap.rolling(window).mean()
        df[f"Gap Volatility {window}"] = gap.rolling(window).std()
        df[f"Gap Z Score {window}"] = (gap - gap.rolling(window).mean()) / gap.rolling(window).std()

    return df


def intraday_features(df):
    daily_range = (df["High"] - df["Low"]).replace(0, np.nan)

    df["Intraday Return"] = df["Close"] / df["Open"] - 1
    df["Open To High"] = df["High"] / df["Open"] - 1
    df["Open To Low"] = df["Low"] / df["Open"] - 1
    df["High To Close"] = df["Close"] / df["High"] - 1
    df["Low To Close"] = df["Close"] / df["Low"] - 1
    df["Close Location Value"] = (df["Close"] - df["Low"]) / daily_range

    return df


def candle_structure(df):
    daily_range = (df["High"] - df["Low"]).replace(0, np.nan)
    body = df["Close"] - df["Open"]
    upper = df["High"] - np.maximum(df["Open"], df["Close"])
    lower = np.minimum(df["Open"], df["Close"]) - df["Low"]

    df["Candle Body Range Ratio"] = body.abs() / daily_range
    df["Candle Direction"] = np.sign(body)
    df["Upper Wick Range Ratio"] = upper / daily_range
    df["Lower Wick Range Ratio"] = lower / daily_range

    return df


def gap_volume_interaction(df, volume_windows=(20, 60)):
    if isinstance(volume_windows, int):
        volume_windows = [volume_windows]

    gap = df["Open"] / df["Close"].shift(1) - 1

    for window in volume_windows:
        relative_volume = df["Volume"] / df["Volume"].rolling(window).mean()
        df[f"Gap Volume Interaction {window}"] = gap * relative_volume

    return df


def all_ohlc_features(df):
    df = gap_features(df)
    df = intraday_features(df)
    df = candle_structure(df)
    df = gap_volume_interaction(df)

    return df

import math
import numpy as np


def _hurst(values):
    values = np.asarray(values)

    if len(values) < 20 or np.any(values <= 0):
        return np.nan

    lags = range(2, min(20, len(values) // 2))
    tau = [np.std(np.subtract(values[lag:], values[:-lag])) for lag in lags]

    if any(value <= 0 for value in tau):
        return np.nan

    return np.polyfit(np.log(list(lags)), np.log(tau), 1)[0]


def hurst_exponent(df, windows=(60, 120, 252)):
    if isinstance(windows, int):
        windows = [windows]

    for window in windows:
        df[f"Hurst Exponent {window}"] = df["Close"].rolling(window).apply(_hurst, raw=True)

    return df


def sign_entropy(df, windows=(20, 60, 252)):
    if isinstance(windows, int):
        windows = [windows]

    returns = df["Close"].pct_change()
    signs = (returns > 0).astype(float)

    def entropy(values):
        p = values.mean()

        if p == 0 or p == 1:
            return 0

        return -(p * math.log(p) + (1 - p) * math.log(1 - p))

    for window in windows:
        df[f"Sign Entropy {window}"] = signs.rolling(window).apply(entropy, raw=True)

    return df


def return_entropy(df, windows=(20, 60, 252), bins=10):
    if isinstance(windows, int):
        windows = [windows]

    returns = df["Close"].pct_change()

    def entropy(values):
        counts, _ = np.histogram(values, bins=bins)
        probabilities = counts[counts > 0] / counts.sum()
        return -(probabilities * np.log(probabilities)).sum()

    for window in windows:
        df[f"Return Entropy {window}"] = returns.rolling(window).apply(entropy, raw=True)

    return df


def variance_ratio(df, windows=(20, 60, 120), lags=(2, 5, 10)):
    if isinstance(windows, int):
        windows = [windows]

    if isinstance(lags, int):
        lags = [lags]

    returns = df["Close"].pct_change()

    for window in windows:
        one_period_variance = returns.rolling(window).var()

        for lag in lags:
            aggregated = returns.rolling(lag).sum()
            aggregated_variance = aggregated.rolling(window).var()
            df[f"Variance Ratio {window} Lag {lag}"] = aggregated_variance / (
                lag * one_period_variance
            )

    return df


def all_experimental_features(df):
    df = hurst_exponent(df)
    df = sign_entropy(df)
    df = return_entropy(df)
    df = variance_ratio(df)

    return df

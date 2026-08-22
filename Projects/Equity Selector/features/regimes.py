import numpy as np


def trend_regime(df, short_window=50, long_window=200):
    short_ma = df["Close"].rolling(short_window).mean()
    long_ma = df["Close"].rolling(long_window).mean()

    df[f"Trend Regime {short_window} {long_window}"] = np.where(short_ma > long_ma, 1, -1)

    return df


def volatility_regime(df, short_window=20, long_window=252):
    returns = df["Close"].pct_change()
    short_vol = returns.rolling(short_window).std()
    long_vol = returns.rolling(long_window).std()

    df[f"Volatility Regime {short_window} {long_window}"] = short_vol / long_vol
    df[f"High Volatility Regime {short_window} {long_window}"] = (short_vol > long_vol).astype(int)

    return df


def drawdown_regime(df, windows=(60, 252), thresholds=(-0.10, -0.20)):
    if isinstance(windows, int):
        windows = [windows]

    if isinstance(thresholds, (int, float)):
        thresholds = [thresholds]

    for window in windows:
        drawdown = df["Close"] / df["Close"].rolling(window).max() - 1

        for threshold in thresholds:
            label = int(abs(threshold) * 100)
            df[f"Drawdown Regime {window} Below {label} Percent"] = (drawdown < threshold).astype(int)

    return df


def momentum_regime(df, windows=(20, 60, 252)):
    if isinstance(windows, int):
        windows = [windows]

    for window in windows:
        momentum = df["Close"].pct_change(window)
        df[f"Positive Momentum Regime {window}"] = (momentum > 0).astype(int)

    return df


def all_regime_features(df):
    df = trend_regime(df)
    df = volatility_regime(df)
    df = drawdown_regime(df)
    df = momentum_regime(df)

    return df

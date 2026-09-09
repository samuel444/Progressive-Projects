import numpy as np


def rolling_volatility(df, windows=(3, 5, 10, 20, 40, 60, 90, 120, 252), annualize=False):
    if isinstance(windows, int):
        windows = [windows]

    returns = df["Close"].pct_change()
    multiplier = np.sqrt(252) if annualize else 1

    for window in windows:
        df[f"Volatility {window}"] = returns.rolling(window).std() * multiplier

    return df


def ewma_volatility(df, spans=(5, 10, 20, 60), annualize=False):
    if isinstance(spans, int):
        spans = [spans]

    returns = df["Close"].pct_change()
    multiplier = np.sqrt(252) if annualize else 1

    for span in spans:
        df[f"EWMA Volatility {span}"] = returns.ewm(span=span, adjust=False).std() * multiplier

    return df


def volatility_ratios(df, short_windows=(5, 10, 20), long_windows=(20, 60, 120, 252)):
    if isinstance(short_windows, int):
        short_windows = [short_windows]

    if isinstance(long_windows, int):
        long_windows = [long_windows]

    returns = df["Close"].pct_change()

    for short_window in short_windows:
        for long_window in long_windows:
            if short_window >= long_window:
                continue

            short_vol = returns.rolling(short_window).std()
            long_vol = returns.rolling(long_window).std()

            df[f"Volatility Ratio {short_window} {long_window}"] = short_vol / long_vol

    return df


def upside_downside_volatility(df, windows=(5, 20, 60, 120)):
    if isinstance(windows, int):
        windows = [windows]

    returns = df["Close"].pct_change()
    upside = returns.where(returns > 0)
    downside = returns.where(returns < 0)

    for window in windows:
        minimum = max(2, window // 4)
        upside_vol = upside.rolling(window, min_periods=minimum).std()
        downside_vol = downside.rolling(window, min_periods=minimum).std()

        df[f"Upside Volatility {window}"] = upside_vol
        df[f"Downside Volatility {window}"] = downside_vol
        df[f"Downside Upside Volatility Ratio {window}"] = downside_vol / upside_vol

    return df


def semivariance(df, windows=(20, 60, 120)):
    if isinstance(windows, int):
        windows = [windows]

    returns = df["Close"].pct_change()

    for window in windows:
        downside = (
            (returns.where(returns < 0) ** 2)
            .rolling(window, min_periods=max(2, window // 4))
            .mean()
        )
        upside = (
            (returns.where(returns > 0) ** 2)
            .rolling(window, min_periods=max(2, window // 4))
            .mean()
        )

        df[f"Downside Semivariance {window}"] = downside
        df[f"Upside Semivariance {window}"] = upside
        df[f"Semivariance Ratio {window}"] = downside / upside

    return df


def volatility_change(df, windows=(5, 20, 60, 252), periods=(1, 5, 20)):
    if isinstance(windows, int):
        windows = [windows]

    if isinstance(periods, int):
        periods = [periods]

    returns = df["Close"].pct_change()

    for window in windows:
        volatility = returns.rolling(window).std()

        for period in periods:
            df[f"Volatility {window} Change {period}"] = volatility.diff(period)

    return df


def volatility_persistence(df, windows=(20, 60, 252), lags=(1, 5)):
    if isinstance(windows, int):
        windows = [windows]

    if isinstance(lags, int):
        lags = [lags]

    returns = df["Close"].pct_change()

    for window in windows:
        for lag in lags:
            df[f"Absolute Return Autocorrelation {window} Lag {lag}"] = (
                returns.abs().rolling(window).corr(returns.abs().shift(lag))
            )
            df[f"Squared Return Autocorrelation {window} Lag {lag}"] = (
                (returns**2).rolling(window).corr((returns**2).shift(lag))
            )

    return df


def all_volatility_features(df):
    df = rolling_volatility(df)
    df = ewma_volatility(df)
    df = volatility_ratios(df)
    df = upside_downside_volatility(df)
    df = semivariance(df)
    df = volatility_change(df)
    df = volatility_persistence(df)

    return df

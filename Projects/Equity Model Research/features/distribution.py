def rolling_distribution(df, windows=(20, 60, 252)):
    if isinstance(windows, int):
        windows = [windows]

    returns = df["Close"].pct_change()

    for window in windows:
        rolling = returns.rolling(window)
        df[f"Return Skew {window}"] = rolling.skew()
        df[f"Return Kurtosis {window}"] = rolling.kurt()
        df[f"Return Median {window}"] = rolling.median()
        df[f"Return Minimum {window}"] = rolling.min()
        df[f"Return Maximum {window}"] = rolling.max()

    return df


def rolling_quantiles(df, windows=(20, 60, 252), quantiles=(0.10, 0.25, 0.75, 0.90)):
    if isinstance(windows, int):
        windows = [windows]

    if isinstance(quantiles, (int, float)):
        quantiles = [quantiles]

    returns = df["Close"].pct_change()

    for window in windows:
        for quantile in quantiles:
            label = int(round(quantile * 100))
            df[f"Return Percentile {label} {window}"] = returns.rolling(window).quantile(quantile)

    return df


def rolling_iqr(df, windows=(20, 60, 252)):
    if isinstance(windows, int):
        windows = [windows]

    returns = df["Close"].pct_change()

    for window in windows:
        q1 = returns.rolling(window).quantile(0.25)
        q3 = returns.rolling(window).quantile(0.75)
        df[f"Return IQR {window}"] = q3 - q1

    return df


def return_autocorrelation(df, windows=(20, 60, 252), lags=(1, 2, 5)):
    if isinstance(windows, int):
        windows = [windows]

    if isinstance(lags, int):
        lags = [lags]

    returns = df["Close"].pct_change()

    for window in windows:
        for lag in lags:
            df[f"Return Autocorrelation {window} Lag {lag}"] = returns.rolling(window).corr(
                returns.shift(lag)
            )

    return df


def all_distribution_features(df):
    df = rolling_distribution(df)
    df = rolling_quantiles(df)
    df = rolling_iqr(df)
    df = return_autocorrelation(df)

    return df

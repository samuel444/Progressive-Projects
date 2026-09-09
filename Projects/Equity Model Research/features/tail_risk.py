def historical_var(df, windows=(20, 60, 252), quantiles=(0.01, 0.05)):
    if isinstance(windows, int):
        windows = [windows]

    if isinstance(quantiles, (int, float)):
        quantiles = [quantiles]

    returns = df["Close"].pct_change()

    for window in windows:
        for quantile in quantiles:
            label = int(round(quantile * 100))
            df[f"VaR {label} Percent {window}"] = returns.rolling(window).quantile(quantile)

    return df


def expected_shortfall(df, windows=(20, 60, 252), quantiles=(0.01, 0.05)):
    if isinstance(windows, int):
        windows = [windows]

    if isinstance(quantiles, (int, float)):
        quantiles = [quantiles]

    returns = df["Close"].pct_change()

    for window in windows:
        for quantile in quantiles:
            label = int(round(quantile * 100))

            def calculate(values):
                cutoff = values.quantile(quantile)
                return values[values <= cutoff].mean()

            df[f"Expected Shortfall {label} Percent {window}"] = returns.rolling(window).apply(
                lambda values: values[values <= values.quantile(quantile)].mean(), raw=False
            )

    return df


def extreme_returns(df, windows=(20, 60, 252), standard_deviations=(2, 3)):
    if isinstance(windows, int):
        windows = [windows]

    if isinstance(standard_deviations, (int, float)):
        standard_deviations = [standard_deviations]

    returns = df["Close"].pct_change()

    for window in windows:
        mean = returns.rolling(window).mean()
        std = returns.rolling(window).std()

        for standard_deviation in standard_deviations:
            extreme = (returns - mean).abs() > standard_deviation * std
            df[f"Extreme Return Frequency {window} {standard_deviation} SD"] = extreme.rolling(
                window
            ).mean()

    return df


def best_worst_return(df, windows=(5, 20, 60, 252)):
    if isinstance(windows, int):
        windows = [windows]

    returns = df["Close"].pct_change()

    for window in windows:
        df[f"Best Return {window}"] = returns.rolling(window).max()
        df[f"Worst Return {window}"] = returns.rolling(window).min()

    return df


def tail_ratio(df, windows=(20, 60, 252), lower=0.05, upper=0.95):
    if isinstance(windows, int):
        windows = [windows]

    returns = df["Close"].pct_change()

    for window in windows:
        upper_tail = returns.rolling(window).quantile(upper).abs()
        lower_tail = returns.rolling(window).quantile(lower).abs()
        df[f"Tail Ratio {window}"] = upper_tail / lower_tail

    return df


def all_tail_risk_features(df):
    df = historical_var(df)
    df = expected_shortfall(df)
    df = extreme_returns(df)
    df = best_worst_return(df)
    df = tail_ratio(df)

    return df

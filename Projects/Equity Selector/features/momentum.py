
def momentum(df, windows=(2, 3, 5, 10, 20, 40, 60, 90, 120, 180, 252)):
    if isinstance(windows, int):
        windows = [windows]

    for window in windows:
        df[f"Momentum {window}"] = df["Close"].pct_change(window)

    return df


def skip_period_momentum(df, windows=(60, 120, 252), skip_periods=(5, 20)):
    if isinstance(windows, int):
        windows = [windows]

    if isinstance(skip_periods, int):
        skip_periods = [skip_periods]

    for window in windows:
        for skip_period in skip_periods:
            if skip_period >= window:
                continue

            df[f"Momentum {window} Skip {skip_period}"] = (
                df["Close"].shift(skip_period) / df["Close"].shift(window) - 1
            )

    return df


def momentum_change(df, windows=(5, 20, 60, 120), periods=(1, 5, 20)):
    if isinstance(windows, int):
        windows = [windows]

    if isinstance(periods, int):
        periods = [periods]

    for window in windows:
        values = df["Close"].pct_change(window)

        for period in periods:
            df[f"Momentum {window} Change {period}"] = values.diff(period)

    return df


def momentum_spread(df, short_windows=(5, 20, 60), long_windows=(20, 60, 120, 252)):
    if isinstance(short_windows, int):
        short_windows = [short_windows]

    if isinstance(long_windows, int):
        long_windows = [long_windows]

    for short_window in short_windows:
        for long_window in long_windows:
            if short_window >= long_window:
                continue

            short_momentum = df["Close"].pct_change(short_window)
            long_momentum = df["Close"].pct_change(long_window)

            df[f"Momentum Spread {short_window} {long_window}"] = short_momentum - long_momentum

    return df


def reversal(df, windows=(1, 2, 5, 10)):
    if isinstance(windows, int):
        windows = [windows]

    for window in windows:
        df[f"Reversal {window}"] = -df["Close"].pct_change(window)

    return df


def all_momentum_features(df):
    df = momentum(df)
    df = skip_period_momentum(df)
    df = momentum_change(df)
    df = momentum_spread(df)
    df = reversal(df)

    return df

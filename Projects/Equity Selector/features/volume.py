
def volume_changes(df, periods=(1, 5, 20)):
    if isinstance(periods, int):
        periods = [periods]

    for period in periods:
        df[f"Volume Change {period}"] = df["Volume"].pct_change(period)

    return df


def average_volume(df, windows=(5, 20, 60, 252)):
    if isinstance(windows, int):
        windows = [windows]

    for window in windows:
        df[f"Average Volume {window}"] = df["Volume"].rolling(window).mean()
        df[f"Volume Volatility {window}"] = df["Volume"].pct_change().rolling(window).std()

    return df


def relative_volume(df, windows=(5, 20, 60)):
    if isinstance(windows, int):
        windows = [windows]

    for window in windows:
        average = df["Volume"].rolling(window).mean()
        std = df["Volume"].rolling(window).std()

        df[f"Relative Volume {window}"] = df["Volume"] / average
        df[f"Volume Z Score {window}"] = (df["Volume"] - average) / std

    return df


def volume_momentum(df, short_windows=(5, 20), long_windows=(20, 60, 252)):
    if isinstance(short_windows, int):
        short_windows = [short_windows]

    if isinstance(long_windows, int):
        long_windows = [long_windows]

    for short_window in short_windows:
        for long_window in long_windows:
            if short_window >= long_window:
                continue

            short_average = df["Volume"].rolling(short_window).mean()
            long_average = df["Volume"].rolling(long_window).mean()
            df[f"Volume Momentum {short_window} {long_window}"] = short_average / long_average - 1

    return df


def return_volume_interactions(df, momentum_windows=(1, 5, 20, 60), volume_windows=(20, 60)):
    if isinstance(momentum_windows, int):
        momentum_windows = [momentum_windows]

    if isinstance(volume_windows, int):
        volume_windows = [volume_windows]

    for momentum_window in momentum_windows:
        stock_return = df["Close"].pct_change(momentum_window)

        for volume_window in volume_windows:
            relative = df["Volume"] / df["Volume"].rolling(volume_window).mean()
            df[f"Return Volume Interaction {momentum_window} {volume_window}"] = stock_return * relative

    return df


def all_volume_features(df):
    df = volume_changes(df)
    df = average_volume(df)
    df = relative_volume(df)
    df = volume_momentum(df)
    df = return_volume_interactions(df)

    return df

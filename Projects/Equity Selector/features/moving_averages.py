
def moving_average_distance(df, windows=(5, 10, 20, 50, 100, 200)):
    if isinstance(windows, int):
        windows = [windows]

    for window in windows:
        moving_average = df["Close"].rolling(window).mean()
        df[f"Distance From MA {window}"] = df["Close"] / moving_average - 1

    return df


def exponential_moving_average_distance(df, spans=(5, 10, 20, 50, 100, 200)):
    if isinstance(spans, int):
        spans = [spans]

    for span in spans:
        moving_average = df["Close"].ewm(span=span, adjust=False).mean()
        df[f"Distance From EMA {span}"] = df["Close"] / moving_average - 1

    return df


def moving_average_ratios(df, short_windows=(5, 10, 20, 50, 100), long_windows=(20, 50, 100, 200)):
    if isinstance(short_windows, int):
        short_windows = [short_windows]

    if isinstance(long_windows, int):
        long_windows = [long_windows]

    for short_window in short_windows:
        for long_window in long_windows:
            if short_window >= long_window:
                continue

            short_ma = df["Close"].rolling(short_window).mean()
            long_ma = df["Close"].rolling(long_window).mean()

            df[f"MA Ratio {short_window} {long_window}"] = short_ma / long_ma - 1

    return df


def exponential_moving_average_ratios(df, short_spans=(5, 10, 20, 50), long_spans=(20, 50, 100, 200)):
    if isinstance(short_spans, int):
        short_spans = [short_spans]

    if isinstance(long_spans, int):
        long_spans = [long_spans]

    for short_span in short_spans:
        for long_span in long_spans:
            if short_span >= long_span:
                continue

            short_ema = df["Close"].ewm(span=short_span, adjust=False).mean()
            long_ema = df["Close"].ewm(span=long_span, adjust=False).mean()

            df[f"EMA Ratio {short_span} {long_span}"] = short_ema / long_ema - 1

    return df


def moving_average_slope(df, windows=(20, 50, 100, 200), periods=(1, 5, 20)):
    if isinstance(windows, int):
        windows = [windows]

    if isinstance(periods, int):
        periods = [periods]

    for window in windows:
        moving_average = df["Close"].rolling(window).mean()

        for period in periods:
            df[f"MA {window} Slope {period}"] = moving_average.pct_change(period)

    return df


def exponential_moving_average_slope(df, spans=(20, 50, 100, 200), periods=(1, 5, 20)):
    if isinstance(spans, int):
        spans = [spans]

    if isinstance(periods, int):
        periods = [periods]

    for span in spans:
        moving_average = df["Close"].ewm(span=span, adjust=False).mean()

        for period in periods:
            df[f"EMA {span} Slope {period}"] = moving_average.pct_change(period)

    return df


def all_moving_average_features(df):
    df = moving_average_distance(df)
    df = exponential_moving_average_distance(df)
    df = moving_average_ratios(df)
    df = exponential_moving_average_ratios(df)
    df = moving_average_slope(df)
    df = exponential_moving_average_slope(df)

    return df

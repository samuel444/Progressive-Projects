import numpy as np


def rsi(df, windows=(5, 10, 14, 20, 60)):
    if isinstance(windows, int):
        windows = [windows]

    change = df["Close"].diff()
    gains = change.clip(lower=0)
    losses = -change.clip(upper=0)

    for window in windows:
        average_gain = gains.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
        average_loss = losses.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
        relative_strength = average_gain / average_loss.replace(0, np.nan)
        df[f"RSI {window}"] = 100 - 100 / (1 + relative_strength)

    return df


def macd(df, fast_spans=(12,), slow_spans=(26,), signal_spans=(9,)):
    if isinstance(fast_spans, int):
        fast_spans = [fast_spans]

    if isinstance(slow_spans, int):
        slow_spans = [slow_spans]

    if isinstance(signal_spans, int):
        signal_spans = [signal_spans]

    for fast_span in fast_spans:
        for slow_span in slow_spans:
            if fast_span >= slow_span:
                continue

            fast = df["Close"].ewm(span=fast_span, adjust=False).mean()
            slow = df["Close"].ewm(span=slow_span, adjust=False).mean()
            value = (fast - slow) / df["Close"]

            df[f"MACD {fast_span} {slow_span}"] = value

            for signal_span in signal_spans:
                signal = value.ewm(span=signal_span, adjust=False).mean()
                df[f"MACD Signal {fast_span} {slow_span} {signal_span}"] = signal
                df[f"MACD Histogram {fast_span} {slow_span} {signal_span}"] = value - signal

    return df


def bollinger(df, windows=(20, 60), standard_deviations=(1, 2)):
    if isinstance(windows, int):
        windows = [windows]

    if isinstance(standard_deviations, (int, float)):
        standard_deviations = [standard_deviations]

    for window in windows:
        mean = df["Close"].rolling(window).mean()
        std = df["Close"].rolling(window).std()
        z_score = (df["Close"] - mean) / std

        df[f"Price Z Score {window}"] = z_score

        for standard_deviation in standard_deviations:
            upper = mean + standard_deviation * std
            lower = mean - standard_deviation * std
            width = upper - lower

            df[f"Bollinger Position {window} {standard_deviation}"] = (df["Close"] - lower) / width
            df[f"Bollinger Width {window} {standard_deviation}"] = width / mean

    return df


def return_z_score(df, windows=(20, 60, 252)):
    if isinstance(windows, int):
        windows = [windows]

    returns = df["Close"].pct_change()

    for window in windows:
        mean = returns.rolling(window).mean()
        std = returns.rolling(window).std()
        df[f"Return Z Score {window}"] = (returns - mean) / std

    return df


def stochastic_oscillator(df, windows=(14, 20, 60)):
    if isinstance(windows, int):
        windows = [windows]

    for window in windows:
        low = df["Low"].rolling(window).min()
        high = df["High"].rolling(window).max()
        df[f"Stochastic Oscillator {window}"] = (df["Close"] - low) / (high - low) * 100

    return df


def all_technical_features(df):
    df = rsi(df)
    df = macd(df)
    df = bollinger(df)
    df = return_z_score(df)
    df = stochastic_oscillator(df)

    return df

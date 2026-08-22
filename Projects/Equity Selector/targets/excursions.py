import numpy as np


def maximum_favourable_excursion(df, horizons=(5, 20, 60)):
    if isinstance(horizons, int):
        horizons = [horizons]

    prices = df["Close"].to_numpy(dtype=float)

    for horizon in horizons:
        values = np.full(len(df), np.nan)

        for i in range(len(df) - horizon):
            path = prices[i + 1:i + horizon + 1] / prices[i] - 1
            values[i] = np.max(path)

        df[f"Maximum Favourable Excursion {horizon}"] = values

    return df


def maximum_adverse_excursion(df, horizons=(5, 20, 60)):
    if isinstance(horizons, int):
        horizons = [horizons]

    prices = df["Close"].to_numpy(dtype=float)

    for horizon in horizons:
        values = np.full(len(df), np.nan)

        for i in range(len(df) - horizon):
            path = prices[i + 1:i + horizon + 1] / prices[i] - 1
            values[i] = np.min(path)

        df[f"Maximum Adverse Excursion {horizon}"] = values

    return df


def time_to_favourable_excursion(df, horizons=(20, 60)):
    if isinstance(horizons, int):
        horizons = [horizons]

    prices = df["Close"].to_numpy(dtype=float)

    for horizon in horizons:
        values = np.full(len(df), np.nan)

        for i in range(len(df) - horizon):
            path = prices[i + 1:i + horizon + 1] / prices[i] - 1
            values[i] = np.argmax(path) + 1

        df[f"Time To Maximum Favourable Excursion {horizon}"] = values

    return df


def time_to_adverse_excursion(df, horizons=(20, 60)):
    if isinstance(horizons, int):
        horizons = [horizons]

    prices = df["Close"].to_numpy(dtype=float)

    for horizon in horizons:
        values = np.full(len(df), np.nan)

        for i in range(len(df) - horizon):
            path = prices[i + 1:i + horizon + 1] / prices[i] - 1
            values[i] = np.argmin(path) + 1

        df[f"Time To Maximum Adverse Excursion {horizon}"] = values

    return df


def all_excursion_targets(df):
    df = maximum_favourable_excursion(df)
    df = maximum_adverse_excursion(df)
    df = time_to_favourable_excursion(df)
    df = time_to_adverse_excursion(df)

    return df

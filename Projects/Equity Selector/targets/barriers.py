import numpy as np


def _as_list(value):
    if isinstance(value, (list, tuple)):
        return list(value)

    return [value]


def first_hit_barrier(df, upper=(0.02, 0.05), lower=(-0.02, -0.05), horizons=(10, 20, 60)):
    upper = _as_list(upper)
    lower = _as_list(lower)
    horizons = _as_list(horizons)
    prices = df["Close"].to_numpy(dtype=float)

    for horizon in horizons:
        for upper_barrier in upper:
            for lower_barrier in lower:
                labels = np.full(len(df), np.nan)

                for i in range(len(df) - horizon):
                    path = prices[i + 1:i + horizon + 1] / prices[i] - 1
                    upper_hits = np.where(path >= upper_barrier)[0]
                    lower_hits = np.where(path <= lower_barrier)[0]

                    if len(upper_hits) == 0 and len(lower_hits) == 0:
                        labels[i] = 0
                    elif len(lower_hits) == 0:
                        labels[i] = 1
                    elif len(upper_hits) == 0:
                        labels[i] = -1
                    elif upper_hits[0] < lower_hits[0]:
                        labels[i] = 1
                    elif lower_hits[0] < upper_hits[0]:
                        labels[i] = -1
                    else:
                        labels[i] = 0

                up_label = round(upper_barrier * 100, 2)
                down_label = round(lower_barrier * 100, 2)
                df[f"Barrier {up_label} {down_label} {horizon}"] = labels

    return df


def volatility_barrier(df, horizons=(20, 60), volatility_windows=(20, 60), upper_multiples=(1, 2), lower_multiples=(1, 2)):
    horizons = _as_list(horizons)
    volatility_windows = _as_list(volatility_windows)
    upper_multiples = _as_list(upper_multiples)
    lower_multiples = _as_list(lower_multiples)
    prices = df["Close"].to_numpy(dtype=float)
    returns = df["Close"].pct_change()

    for volatility_window in volatility_windows:
        volatility = returns.rolling(volatility_window).std().to_numpy(dtype=float)

        for horizon in horizons:
            for upper_multiple in upper_multiples:
                for lower_multiple in lower_multiples:
                    labels = np.full(len(df), np.nan)

                    for i in range(len(df) - horizon):
                        if np.isnan(volatility[i]):
                            continue

                        scale = volatility[i] * np.sqrt(horizon)
                        upper_barrier = upper_multiple * scale
                        lower_barrier = -lower_multiple * scale
                        path = prices[i + 1:i + horizon + 1] / prices[i] - 1
                        upper_hits = np.where(path >= upper_barrier)[0]
                        lower_hits = np.where(path <= lower_barrier)[0]

                        if len(upper_hits) == 0 and len(lower_hits) == 0:
                            labels[i] = 0
                        elif len(lower_hits) == 0:
                            labels[i] = 1
                        elif len(upper_hits) == 0:
                            labels[i] = -1
                        elif upper_hits[0] < lower_hits[0]:
                            labels[i] = 1
                        elif lower_hits[0] < upper_hits[0]:
                            labels[i] = -1
                        else:
                            labels[i] = 0

                    df[f"Volatility Barrier {volatility_window} {horizon} {upper_multiple} {lower_multiple}"] = labels

    return df


def all_barrier_targets(df):
    df = first_hit_barrier(df)
    df = volatility_barrier(df)

    return df

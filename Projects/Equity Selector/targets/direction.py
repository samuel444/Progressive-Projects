
def future_direction(df, horizons=(1, 5, 20, 60)):
    if isinstance(horizons, int):
        horizons = [horizons]

    for horizon in horizons:
        forward = df["Close"].shift(-horizon) / df["Close"] - 1
        df[f"Future Direction {horizon}"] = (forward > 0).astype(float)
        df.loc[forward.isna(), f"Future Direction {horizon}"] = float("nan")

    return df


def threshold_direction(df, horizons=(5, 20, 60), thresholds=(0.01, 0.02, 0.05, 0.10)):
    if isinstance(horizons, int):
        horizons = [horizons]

    if isinstance(thresholds, (int, float)):
        thresholds = [thresholds]

    for horizon in horizons:
        forward = df["Close"].shift(-horizon) / df["Close"] - 1

        for threshold in thresholds:
            label = int(round(threshold * 100))
            column = f"Future Return Above {label} Percent {horizon}"
            df[column] = (forward > threshold).astype(float)
            df.loc[forward.isna(), column] = float("nan")

    return df


def three_class_direction(df, horizons=(20, 60), thresholds=(0.02, 0.05)):
    if isinstance(horizons, int):
        horizons = [horizons]

    if isinstance(thresholds, (int, float)):
        thresholds = [thresholds]

    for horizon in horizons:
        forward = df["Close"].shift(-horizon) / df["Close"] - 1

        for threshold in thresholds:
            label = int(round(threshold * 100))
            column = f"Three Class Direction {label} Percent {horizon}"
            df[column] = 0.0
            df.loc[forward > threshold, column] = 1.0
            df.loc[forward < -threshold, column] = -1.0
            df.loc[forward.isna(), column] = float("nan")

    return df


def all_direction_targets(df):
    df = future_direction(df)
    df = threshold_direction(df)
    df = three_class_direction(df)

    return df

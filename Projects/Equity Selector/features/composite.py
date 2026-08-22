import numpy as np


def composite_score(df, columns, name, weights=None, invert=None):
    if isinstance(columns, str):
        columns = [columns]

    if weights is None:
        weights = [1] * len(columns)

    if invert is None:
        invert = []

    if len(weights) != len(columns):
        raise ValueError("weights must have the same length as columns")

    values = []

    for column, weight in zip(columns, weights):
        feature = df[column]

        if column in invert:
            feature = -feature

        values.append(feature * weight)

    df[name] = sum(values) / sum(abs(weight) for weight in weights)

    return df


def momentum_score(df, columns=("Momentum 20", "Momentum 60", "Momentum 120", "Momentum 252"), weights=None):
    return composite_score(df, columns, "Momentum Score", weights=weights)


def low_risk_score(df, columns=("Volatility 20", "Drawdown 60", "Rolling Beta 60"), weights=None):
    return composite_score(df, columns, "Low Risk Score", weights=weights, invert=list(columns))


def trend_score(df, columns=("Trend Strength 20", "Trend Strength 60", "Trend Efficiency 20"), weights=None):
    return composite_score(df, columns, "Trend Score", weights=weights)


def all_composite_features(df):
    available_momentum = [column for column in ("Momentum 20", "Momentum 60", "Momentum 120", "Momentum 252") if column in df.columns]
    available_risk = [column for column in ("Volatility 20", "Drawdown 60", "Rolling Beta 60") if column in df.columns]
    available_trend = [column for column in ("Trend Strength 20", "Trend Strength 60", "Trend Efficiency 20") if column in df.columns]

    if available_momentum:
        df = momentum_score(df, available_momentum)

    if available_risk:
        df = low_risk_score(df, available_risk)

    if available_trend:
        df = trend_score(df, available_trend)

    return df

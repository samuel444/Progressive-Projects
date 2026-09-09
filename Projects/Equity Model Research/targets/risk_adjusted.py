import numpy as np


def future_return_volatility_ratio(df, horizons=(5, 20, 60)):
    if isinstance(horizons, int):
        horizons = [horizons]

    returns = df["Close"].pct_change()

    for horizon in horizons:
        forward_return = df["Close"].shift(-horizon) / df["Close"] - 1
        future_volatility = returns.shift(-1)[::-1].rolling(horizon).std()[::-1]
        df[f"Future Return Volatility Ratio {horizon}"] = forward_return / future_volatility

    return df


def future_sortino_ratio(df, horizons=(20, 60)):
    if isinstance(horizons, int):
        horizons = [horizons]

    returns = df["Close"].pct_change()

    for horizon in horizons:
        forward_return = df["Close"].shift(-horizon) / df["Close"] - 1
        shifted = returns.shift(-1)[::-1]
        downside = (
            shifted.where(shifted < 0)
            .rolling(horizon, min_periods=max(2, horizon // 4))
            .std()[::-1]
        )
        df[f"Future Sortino Ratio {horizon}"] = forward_return / downside

    return df


def future_return_minus_risk(df, horizons=(20, 60), risk_weights=(0.5, 1, 2)):
    if isinstance(horizons, int):
        horizons = [horizons]

    if isinstance(risk_weights, (int, float)):
        risk_weights = [risk_weights]

    returns = df["Close"].pct_change()

    for horizon in horizons:
        forward_return = df["Close"].shift(-horizon) / df["Close"] - 1
        future_volatility = returns.shift(-1)[::-1].rolling(horizon).std()[::-1]

        for risk_weight in risk_weights:
            df[f"Future Return Minus Risk {horizon} {risk_weight}"] = (
                forward_return - risk_weight * future_volatility
            )

    return df


def future_return_drawdown_ratio(df, horizons=(20, 60)):
    if isinstance(horizons, int):
        horizons = [horizons]

    prices = df["Close"].to_numpy(dtype=float)

    for horizon in horizons:
        values = np.full(len(df), np.nan)

        for i in range(len(df) - horizon):
            path = prices[i : i + horizon + 1]
            peak = np.maximum.accumulate(path)
            max_drawdown = abs(np.min(path / peak - 1))
            forward_return = prices[i + horizon] / prices[i] - 1

            if max_drawdown != 0:
                values[i] = forward_return / max_drawdown

        df[f"Future Return Drawdown Ratio {horizon}"] = values

    return df


def all_risk_adjusted_targets(df):
    df = future_return_volatility_ratio(df)
    df = future_sortino_ratio(df)
    df = future_return_minus_risk(df)
    df = future_return_drawdown_ratio(df)

    return df

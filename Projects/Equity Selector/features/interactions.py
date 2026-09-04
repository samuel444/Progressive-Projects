def feature_interactions(df, left_features, right_features):
    if isinstance(left_features, str):
        left_features = [left_features]

    if isinstance(right_features, str):
        right_features = [right_features]

    for left_feature in left_features:
        for right_feature in right_features:
            if left_feature == right_feature:
                continue

            df[f"{left_feature} x {right_feature}"] = df[left_feature] * df[right_feature]

    return df


def momentum_volatility_interactions(
    df, momentum_windows=(20, 60, 120), volatility_windows=(20, 60)
):
    if isinstance(momentum_windows, int):
        momentum_windows = [momentum_windows]

    if isinstance(volatility_windows, int):
        volatility_windows = [volatility_windows]

    returns = df["Close"].pct_change()

    for momentum_window in momentum_windows:
        momentum = df["Close"].pct_change(momentum_window)

        for volatility_window in volatility_windows:
            volatility = returns.rolling(volatility_window).std()
            df[f"Momentum Volatility Interaction {momentum_window} {volatility_window}"] = (
                momentum * volatility
            )

    return df


def momentum_drawdown_interactions(
    df, momentum_windows=(20, 60, 120), drawdown_windows=(20, 60, 252)
):
    if isinstance(momentum_windows, int):
        momentum_windows = [momentum_windows]

    if isinstance(drawdown_windows, int):
        drawdown_windows = [drawdown_windows]

    for momentum_window in momentum_windows:
        momentum = df["Close"].pct_change(momentum_window)

        for drawdown_window in drawdown_windows:
            drawdown = df["Close"] / df["Close"].rolling(drawdown_window).max() - 1
            df[f"Momentum Drawdown Interaction {momentum_window} {drawdown_window}"] = (
                momentum * drawdown
            )

    return df


def momentum_volume_interactions(df, momentum_windows=(20, 60, 120), volume_windows=(20, 60)):
    if isinstance(momentum_windows, int):
        momentum_windows = [momentum_windows]

    if isinstance(volume_windows, int):
        volume_windows = [volume_windows]

    for momentum_window in momentum_windows:
        momentum = df["Close"].pct_change(momentum_window)

        for volume_window in volume_windows:
            relative_volume = df["Volume"] / df["Volume"].rolling(volume_window).mean()
            df[f"Momentum Volume Interaction {momentum_window} {volume_window}"] = (
                momentum * relative_volume
            )

    return df


def all_interaction_features(df):
    df = momentum_volatility_interactions(df)
    df = momentum_drawdown_interactions(df)
    df = momentum_volume_interactions(df)

    return df

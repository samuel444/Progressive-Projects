def future_return_rank(
    df, horizons=(5, 20, 60), ticker_col="Ticker", date_col="Date", price_col="Close"
):
    if isinstance(horizons, int):
        horizons = [horizons]

    for horizon in horizons:
        future_price = df.groupby(ticker_col)[price_col].shift(-horizon)
        forward_return = future_price / df[price_col] - 1
        df[f"Forward Return {horizon}"] = forward_return
        df[f"Future Return Rank {horizon}"] = (
            df.assign(__Target=forward_return).groupby(date_col)["__Target"].rank(pct=True)
        )

    return df


def top_quantile_label(
    df,
    horizons=(20, 60),
    quantiles=(0.20, 0.25),
    ticker_col="Ticker",
    date_col="Date",
    price_col="Close",
):
    if isinstance(horizons, int):
        horizons = [horizons]

    if isinstance(quantiles, (int, float)):
        quantiles = [quantiles]

    for horizon in horizons:
        future_price = df.groupby(ticker_col)[price_col].shift(-horizon)
        forward_return = future_price / df[price_col] - 1
        rank = df.assign(__Target=forward_return).groupby(date_col)["__Target"].rank(pct=True)

        for quantile in quantiles:
            label = int(round(quantile * 100))
            column = f"Top {label} Percent Future Return {horizon}"
            df[column] = (rank >= 1 - quantile).astype(float)
            df.loc[forward_return.isna(), column] = float("nan")

    return df


def bottom_quantile_label(
    df,
    horizons=(20, 60),
    quantiles=(0.20, 0.25),
    ticker_col="Ticker",
    date_col="Date",
    price_col="Close",
):
    if isinstance(horizons, int):
        horizons = [horizons]

    if isinstance(quantiles, (int, float)):
        quantiles = [quantiles]

    for horizon in horizons:
        future_price = df.groupby(ticker_col)[price_col].shift(-horizon)
        forward_return = future_price / df[price_col] - 1
        rank = df.assign(__Target=forward_return).groupby(date_col)["__Target"].rank(pct=True)

        for quantile in quantiles:
            label = int(round(quantile * 100))
            column = f"Bottom {label} Percent Future Return {horizon}"
            df[column] = (rank <= quantile).astype(float)
            df.loc[forward_return.isna(), column] = float("nan")

    return df


def all_ranking_targets(df, ticker_col="Ticker", date_col="Date", price_col="Close"):
    df = future_return_rank(df, ticker_col=ticker_col, date_col=date_col, price_col=price_col)
    df = top_quantile_label(df, ticker_col=ticker_col, date_col=date_col, price_col=price_col)
    df = bottom_quantile_label(df, ticker_col=ticker_col, date_col=date_col, price_col=price_col)

    return df

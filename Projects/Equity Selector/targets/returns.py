import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def forward_return(df, horizons=(1, 5, 10, 20, 60, 120, 252)):
    if isinstance(horizons, int):
        horizons = [horizons]

    for horizon in horizons:
        df[f"Forward Return {horizon}"] = df["Close"].shift(-horizon) / df["Close"] - 1

    return df


def forward_log_return(df, horizons=(1, 5, 20, 60, 120, 252)):
    if isinstance(horizons, int):
        horizons = [horizons]

    for horizon in horizons:
        df[f"Forward Log Return {horizon}"] = np.log(df["Close"].shift(-horizon) / df["Close"])

    return df


def forward_excess_return(df, horizons=(5, 20, 60), benchmark_df=None, benchmark_ticker="^GSPC"):
    if isinstance(horizons, int):
        horizons = [horizons]

    if benchmark_df is None:
        first_date = df.index[0]
        last_date = df.index[-1] + pd.Timedelta(days=1)

        logger.info("Downloading benchmark data for %s", benchmark_ticker)
        import yfinance as yf

        benchmark_df = yf.download(
            benchmark_ticker,
            start=first_date,
            end=last_date,
            auto_adjust=True,
            progress=False,
            multi_level_index=False
        )

    benchmark_close = benchmark_df["Close"].reindex(df.index)

    for horizon in horizons:
        stock_return = df["Close"].shift(-horizon) / df["Close"] - 1
        benchmark_return = benchmark_close.shift(-horizon) / benchmark_close - 1
        df[f"Forward Excess Return {horizon}"] = stock_return - benchmark_return

    return df


def all_return_targets(df, benchmark_df=None, benchmark_ticker="^GSPC"):
    df = forward_return(df)
    df = forward_log_return(df)
    df = forward_excess_return(df, benchmark_df=benchmark_df, benchmark_ticker=benchmark_ticker)

    return df

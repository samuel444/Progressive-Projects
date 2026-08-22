import logging
import pandas as pd

logger = logging.getLogger(__name__)


def _benchmark_return(df, benchmark_df=None, benchmark_ticker="^GSPC"):
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

    return benchmark_df["Close"].reindex(df.index).pct_change()


def rolling_correlation(df, windows=(20, 60, 120, 252), benchmark_df=None, benchmark_ticker="^GSPC", label="Market"):
    if isinstance(windows, int):
        windows = [windows]

    stock_return = df["Close"].pct_change()
    benchmark_return = _benchmark_return(df, benchmark_df, benchmark_ticker)

    for window in windows:
        df[f"{label} Correlation {window}"] = stock_return.rolling(window).corr(benchmark_return)

    return df


def upside_downside_correlation(df, windows=(60, 252), benchmark_df=None, benchmark_ticker="^GSPC", label="Market"):
    if isinstance(windows, int):
        windows = [windows]

    stock_return = df["Close"].pct_change()
    benchmark_return = _benchmark_return(df, benchmark_df, benchmark_ticker)

    upside_stock = stock_return.where(benchmark_return > 0)
    upside_benchmark = benchmark_return.where(benchmark_return > 0)
    downside_stock = stock_return.where(benchmark_return < 0)
    downside_benchmark = benchmark_return.where(benchmark_return < 0)

    for window in windows:
        minimum = max(3, window // 4)
        df[f"Upside {label} Correlation {window}"] = (
            upside_stock.rolling(window, min_periods=minimum).corr(upside_benchmark)
        )
        df[f"Downside {label} Correlation {window}"] = (
            downside_stock.rolling(window, min_periods=minimum).corr(downside_benchmark)
        )

    return df


def correlation_change(df, windows=(20, 60, 252), periods=(5, 20), benchmark_df=None, benchmark_ticker="^GSPC", label="Market"):
    if isinstance(windows, int):
        windows = [windows]

    if isinstance(periods, int):
        periods = [periods]

    stock_return = df["Close"].pct_change()
    benchmark_return = _benchmark_return(df, benchmark_df, benchmark_ticker)

    for window in windows:
        correlation = stock_return.rolling(window).corr(benchmark_return)

        for period in periods:
            df[f"{label} Correlation {window} Change {period}"] = correlation.diff(period)

    return df


def all_correlation_features(df, market_df=None, market_ticker="^GSPC"):
    df = rolling_correlation(df, benchmark_df=market_df, benchmark_ticker=market_ticker)
    df = upside_downside_correlation(df, benchmark_df=market_df, benchmark_ticker=market_ticker)
    df = correlation_change(df, benchmark_df=market_df, benchmark_ticker=market_ticker)

    return df

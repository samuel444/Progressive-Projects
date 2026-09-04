import logging
import pandas as pd

logger = logging.getLogger(__name__)


def _market_return(df, market_df=None, market_ticker="^GSPC"):
    if market_df is None:
        first_date = df.index[0]
        last_date = df.index[-1] + pd.Timedelta(days=1)

        logger.info("Downloading market data for %s", market_ticker)
        import yfinance as yf

        market_df = yf.download(
            market_ticker,
            start=first_date,
            end=last_date,
            auto_adjust=True,
            progress=False,
            multi_level_index=False,
        )

    return market_df["Close"].reindex(df.index).pct_change()


def rolling_beta(df, windows=(20, 60, 120, 252), market_df=None, market_ticker="^GSPC"):
    if isinstance(windows, int):
        windows = [windows]

    stock_return = df["Close"].pct_change()
    market_return = _market_return(df, market_df, market_ticker)

    for window in windows:
        covariance = stock_return.rolling(window).cov(market_return)
        variance = market_return.rolling(window).var()

        df[f"Rolling Beta {window}"] = covariance / variance

    return df


def downside_beta(df, windows=(20, 60, 120, 252), market_df=None, market_ticker="^GSPC"):
    if isinstance(windows, int):
        windows = [windows]

    stock_return = df["Close"].pct_change()
    market_return = _market_return(df, market_df, market_ticker)
    downside_stock = stock_return.where(market_return < 0)
    downside_market = market_return.where(market_return < 0)

    for window in windows:
        minimum = max(3, window // 4)
        covariance = downside_stock.rolling(window, min_periods=minimum).cov(downside_market)
        variance = downside_market.rolling(window, min_periods=minimum).var()

        df[f"Downside Beta {window}"] = covariance / variance

    return df


def upside_beta(df, windows=(20, 60, 120, 252), market_df=None, market_ticker="^GSPC"):
    if isinstance(windows, int):
        windows = [windows]

    stock_return = df["Close"].pct_change()
    market_return = _market_return(df, market_df, market_ticker)
    upside_stock = stock_return.where(market_return > 0)
    upside_market = market_return.where(market_return > 0)

    for window in windows:
        minimum = max(3, window // 4)
        covariance = upside_stock.rolling(window, min_periods=minimum).cov(upside_market)
        variance = upside_market.rolling(window, min_periods=minimum).var()

        df[f"Upside Beta {window}"] = covariance / variance

    return df


def beta_change(
    df, windows=(20, 60, 120, 252), periods=(1, 5, 20), market_df=None, market_ticker="^GSPC"
):
    if isinstance(windows, int):
        windows = [windows]

    if isinstance(periods, int):
        periods = [periods]

    missing = [window for window in windows if f"Rolling Beta {window}" not in df.columns]

    if missing:
        df = rolling_beta(df, missing, market_df, market_ticker)

    for window in windows:
        for period in periods:
            df[f"Rolling Beta {window} Change {period}"] = df[f"Rolling Beta {window}"].diff(period)

    return df


def beta_ratios(
    df, short_windows=(20, 60), long_windows=(120, 252), market_df=None, market_ticker="^GSPC"
):
    if isinstance(short_windows, int):
        short_windows = [short_windows]

    if isinstance(long_windows, int):
        long_windows = [long_windows]

    windows = sorted(set(list(short_windows) + list(long_windows)))
    missing = [window for window in windows if f"Rolling Beta {window}" not in df.columns]

    if missing:
        df = rolling_beta(df, missing, market_df, market_ticker)

    for short_window in short_windows:
        for long_window in long_windows:
            if short_window >= long_window:
                continue

            df[f"Beta Ratio {short_window} {long_window}"] = (
                df[f"Rolling Beta {short_window}"] / df[f"Rolling Beta {long_window}"]
            )

    return df


def all_beta_features(df, market_df=None, market_ticker="^GSPC"):
    df = rolling_beta(df, market_df=market_df, market_ticker=market_ticker)
    df = downside_beta(df, market_df=market_df, market_ticker=market_ticker)
    df = upside_beta(df, market_df=market_df, market_ticker=market_ticker)
    df = beta_change(df, market_df=market_df, market_ticker=market_ticker)
    df = beta_ratios(df, market_df=market_df, market_ticker=market_ticker)

    return df

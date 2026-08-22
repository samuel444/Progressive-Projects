import logging
import pandas as pd

logger = logging.getLogger(__name__)


def _market_close(df, market_df=None, market_ticker="^GSPC"):
    if market_df is not None:
        return market_df["Close"].reindex(df.index)

    first_date = df.index[0]
    last_date = df.index[-1] + pd.Timedelta(days=1)

    logger.info("Downloading market data for %s", market_ticker)
    import yfinance as yf

    market = yf.download(
        market_ticker,
        start=first_date,
        end=last_date,
        auto_adjust=True,
        progress=False,
        multi_level_index=False
    )

    return market["Close"].reindex(df.index)


def market_relative_returns(df, market_df=None, market_ticker="^GSPC", periods=(1, 5, 20, 60, 120, 252)):
    if isinstance(periods, int):
        periods = [periods]

    market_close = _market_close(df, market_df, market_ticker)

    for period in periods:
        stock_return = df["Close"].pct_change(period)
        market_return = market_close.pct_change(period)
        df[f"Market Relative Return {period}"] = stock_return - market_return

    return df


def market_relative_strength(df, market_df=None, market_ticker="^GSPC", windows=(20, 60, 120, 252)):
    if isinstance(windows, int):
        windows = [windows]

    market_close = _market_close(df, market_df, market_ticker)
    relative_price = df["Close"] / market_close

    for window in windows:
        df[f"Market Relative Strength {window}"] = relative_price.pct_change(window)

    return df


def market_relative_ma_distance(df, market_df=None, market_ticker="^GSPC", windows=(20, 50, 200)):
    if isinstance(windows, int):
        windows = [windows]

    market_close = _market_close(df, market_df, market_ticker)

    for window in windows:
        stock_distance = df["Close"] / df["Close"].rolling(window).mean() - 1
        market_distance = market_close / market_close.rolling(window).mean() - 1
        df[f"Market Relative MA Distance {window}"] = stock_distance - market_distance

    return df


def all_market_relative_features(df, market_df=None, market_ticker="^GSPC"):
    df = market_relative_returns(df, market_df, market_ticker)
    df = market_relative_strength(df, market_df, market_ticker)
    df = market_relative_ma_distance(df, market_df, market_ticker)

    return df

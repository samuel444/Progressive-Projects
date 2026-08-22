import logging
import pandas as pd

logger = logging.getLogger(__name__)


def _sector_close(df, sector_df=None, sector_ticker=None):
    if sector_df is not None:
        return sector_df["Close"].reindex(df.index)

    if sector_ticker is None:
        raise ValueError("sector_df or sector_ticker must be provided")

    first_date = df.index[0]
    last_date = df.index[-1] + pd.Timedelta(days=1)

    logger.info("Downloading sector data for %s", sector_ticker)
    import yfinance as yf

    sector = yf.download(
        sector_ticker,
        start=first_date,
        end=last_date,
        auto_adjust=True,
        progress=False,
        multi_level_index=False
    )

    return sector["Close"].reindex(df.index)


def sector_relative_returns(df, sector_df=None, sector_ticker=None, periods=(1, 5, 20, 60, 120, 252)):
    if isinstance(periods, int):
        periods = [periods]

    sector_close = _sector_close(df, sector_df, sector_ticker)

    for period in periods:
        stock_return = df["Close"].pct_change(period)
        sector_return = sector_close.pct_change(period)
        df[f"Sector Relative Return {period}"] = stock_return - sector_return

    return df


def sector_relative_strength(df, sector_df=None, sector_ticker=None, windows=(20, 60, 120, 252)):
    if isinstance(windows, int):
        windows = [windows]

    sector_close = _sector_close(df, sector_df, sector_ticker)
    relative_price = df["Close"] / sector_close

    for window in windows:
        df[f"Sector Relative Strength {window}"] = relative_price.pct_change(window)

    return df


def sector_relative_ma_distance(df, sector_df=None, sector_ticker=None, windows=(20, 50, 200)):
    if isinstance(windows, int):
        windows = [windows]

    sector_close = _sector_close(df, sector_df, sector_ticker)

    for window in windows:
        stock_distance = df["Close"] / df["Close"].rolling(window).mean() - 1
        sector_distance = sector_close / sector_close.rolling(window).mean() - 1
        df[f"Sector Relative MA Distance {window}"] = stock_distance - sector_distance

    return df


def all_sector_relative_features(df, sector_df=None, sector_ticker=None):
    df = sector_relative_returns(df, sector_df, sector_ticker)
    df = sector_relative_strength(df, sector_df, sector_ticker)
    df = sector_relative_ma_distance(df, sector_df, sector_ticker)

    return df

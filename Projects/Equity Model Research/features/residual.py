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


def rolling_alpha(df, windows=(20, 60, 120, 252), market_df=None, market_ticker="^GSPC"):
    if isinstance(windows, int):
        windows = [windows]

    stock_return = df["Close"].pct_change()
    market_return = _market_return(df, market_df, market_ticker)

    for window in windows:
        beta = stock_return.rolling(window).cov(market_return) / market_return.rolling(window).var()
        alpha = stock_return.rolling(window).mean() - beta * market_return.rolling(window).mean()
        df[f"Rolling Alpha {window}"] = alpha

    return df


def idiosyncratic_volatility(df, windows=(20, 60, 252), market_df=None, market_ticker="^GSPC"):
    if isinstance(windows, int):
        windows = [windows]

    stock_return = df["Close"].pct_change()
    market_return = _market_return(df, market_df, market_ticker)

    for window in windows:
        beta = stock_return.rolling(window).cov(market_return) / market_return.rolling(window).var()
        alpha = stock_return.rolling(window).mean() - beta * market_return.rolling(window).mean()
        residual = stock_return - alpha - beta * market_return
        df[f"Idiosyncratic Volatility {window}"] = residual.rolling(window).std()

    return df


def residual_momentum(df, windows=(20, 60, 120, 252), market_df=None, market_ticker="^GSPC"):
    if isinstance(windows, int):
        windows = [windows]

    stock_return = df["Close"].pct_change()
    market_return = _market_return(df, market_df, market_ticker)

    for window in windows:
        beta = stock_return.rolling(window).cov(market_return) / market_return.rolling(window).var()
        alpha = stock_return.rolling(window).mean() - beta * market_return.rolling(window).mean()
        residual = stock_return - alpha - beta * market_return
        df[f"Residual Momentum {window}"] = (1 + residual).rolling(window).apply(
            lambda values: values.prod(), raw=True
        ) - 1

    return df


def market_r_squared(df, windows=(20, 60, 120, 252), market_df=None, market_ticker="^GSPC"):
    if isinstance(windows, int):
        windows = [windows]

    stock_return = df["Close"].pct_change()
    market_return = _market_return(df, market_df, market_ticker)

    for window in windows:
        correlation = stock_return.rolling(window).corr(market_return)
        df[f"Market R Squared {window}"] = correlation**2

    return df


def all_residual_features(df, market_df=None, market_ticker="^GSPC"):
    df = rolling_alpha(df, market_df=market_df, market_ticker=market_ticker)
    df = idiosyncratic_volatility(df, market_df=market_df, market_ticker=market_ticker)
    df = residual_momentum(df, market_df=market_df, market_ticker=market_ticker)
    df = market_r_squared(df, market_df=market_df, market_ticker=market_ticker)

    return df


"""Historical equity data download and volatility feature engineering."""

from __future__ import annotations

import logging
from collections.abc import Iterable

import numpy as np
import pandas as pd
import yfinance as yf

from options_risk_engine.domain import OptionTicker

logger = logging.getLogger(__name__)


def download_historical_market_data(
    symbols: Iterable[str],
    period: str = "5y",
    interval: str = "1d",
) -> pd.DataFrame:
    """Download adjusted closes and validate the returned ticker universe."""
    symbols = [str(symbol).upper().strip() for symbol in symbols]
    if not symbols:
        raise ValueError("At least one symbol is required")

    logger.info("Downloading historical data for %d symbols", len(symbols))
    raw = yf.download(
        symbols,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False,
    )
    if raw.empty or "Close" not in raw.columns.get_level_values(0):
        raise RuntimeError("Historical download did not contain closing prices")

    closes = raw[["Close"]].copy().sort_index()
    close_symbols = set(closes["Close"].columns)
    missing = sorted(set(symbols).difference(close_symbols))
    if missing:
        logger.warning("No close series returned for: %s", missing)

    if (closes["Close"] <= 0).any().any():
        raise ValueError("Historical close data contains non-positive prices")

    logger.info("Historical data downloaded: %d rows", len(closes))
    return closes


def fetch_dividend_yields(tickers: list[OptionTicker]) -> dict[str, float]:
    """Fetch and attach ticker-level dividend yields with a zero fallback."""
    yields: dict[str, float] = {}
    for ticker in tickers:
        try:
            raw_value = yf.Ticker(ticker.symbol).info.get("dividendYield", 0)
            value = float(raw_value or 0)
            # Keep the original script's convention while protecting against
            # obviously percentage-form values.
            if value > 1:
                value /= 100
            ticker.dividend_yield = max(value, 0.0)
            ticker.refresh_derived_market_values()
        except Exception as error:
            logger.warning(
                "Failed to retrieve dividend yield for %s: %s",
                ticker.symbol,
                error,
            )
            ticker.dividend_yield = 0.0
        yields[ticker.symbol] = ticker.dividend_yield
    return yields


def build_volatility_feature_dataset(
    close_data: pd.DataFrame,
    tickers: list[OptionTicker],
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Create the original return, realised-volatility and lagged features."""
    if close_data.empty:
        raise ValueError("close_data cannot be empty")
    if "Close" not in close_data.columns.get_level_values(0):
        raise KeyError("close_data must contain a top-level 'Close' column")

    df = close_data[["Close"]].copy()
    returns = df["Close"].pct_change()
    returns.columns = pd.MultiIndex.from_product(
        [["Return"], returns.columns], names=df.columns.names
    )
    df = pd.concat([df, returns], axis=1)
    r = df["Return"]

    rv_20 = r.rolling(20).std() * np.sqrt(252)
    rv_60 = r.rolling(60).std() * np.sqrt(252)
    rv_252 = r.rolling(252).std() * np.sqrt(252)
    abs_return = r.abs()
    squared_return = r ** 2
    ewm_vol_20 = r.ewm(span=20, adjust=False).std() * np.sqrt(252)
    ewm_vol_60 = r.ewm(span=60, adjust=False).std() * np.sqrt(252)
    mean_abs_return_5 = abs_return.rolling(5).mean()
    mean_abs_return_20 = abs_return.rolling(20).mean()
    max_abs_return_20 = abs_return.rolling(20).max()
    rv_ratio_20_60 = rv_20 / rv_60
    rv_ratio_60_252 = rv_60 / rv_252
    vol_of_vol_20 = rv_20.rolling(20).std()
    rv_20_lag1 = rv_20.shift(1)
    rv_20_lag5 = rv_20.shift(5)
    rv_60_lag1 = rv_60.shift(1)
    return_lag1 = r.shift(1)
    return_lag2 = r.shift(2)
    return_lag5 = r.shift(5)
    abs_return_lag1 = abs_return.shift(1)
    squared_return_lag1 = squared_return.shift(1)

    features = {
        "RV20": rv_20,
        "RV60": rv_60,
        "RV252": rv_252,
        "AbsReturn": abs_return,
        "SquaredReturn": squared_return,
        "EWMVol20": ewm_vol_20,
        "EWMVol60": ewm_vol_60,
        "MeanAbsReturn5": mean_abs_return_5,
        "MeanAbsReturn20": mean_abs_return_20,
        "MaxAbsReturn20": max_abs_return_20,
        "RVRatio20_60": rv_ratio_20_60,
        "RVRatio60_252": rv_ratio_60_252,
        "VolOfVol20": vol_of_vol_20,
        "RV20Lag1": rv_20_lag1,
        "RV20Lag5": rv_20_lag5,
        "RV60Lag1": rv_60_lag1,
        "ReturnLag1": return_lag1,
        "ReturnLag2": return_lag2,
        "ReturnLag5": return_lag5,
        "AbsReturnLag1": abs_return_lag1,
        "SquaredReturnLag1": squared_return_lag1,
    }

    for name, feature in features.items():
        feature = feature.copy()
        feature.columns = pd.MultiIndex.from_product(
            [[name], feature.columns], names=df.columns.names
        )
        df = pd.concat([df, feature], axis=1)

    # Retain the original sequencing: drop rows required by historic features,
    # then append the forward target (which naturally leaves NaNs at the end).
    df = df.dropna()
    target_rv_by_symbol = {}
    for ticker in tickers:
        target_rv_by_symbol[ticker.symbol] = (
            r[ticker.symbol]
            .rolling(ticker.forecast_horizon)
            .std()
            .shift(-ticker.forecast_horizon)
            * np.sqrt(252)
        )

    target_rv = pd.DataFrame(target_rv_by_symbol)
    target_rv.columns = pd.MultiIndex.from_product(
        [["Target_RV"], target_rv.columns], names=df.columns.names
    )
    df = pd.concat([df, target_rv], axis=1)

    feature_names = list(features.keys())
    logger.info(
        "Feature dataset created: %d rows, %d engineered features per ticker",
        len(df),
        len(feature_names),
    )
    return df, r, feature_names


def equity_prices_to_long(close_data: pd.DataFrame) -> pd.DataFrame:
    prices = close_data["Close"].stack().rename("close").reset_index()
    prices.columns = ["date", "ticker", "close"]
    return prices


def features_to_long(feature_data: pd.DataFrame) -> pd.DataFrame:
    long = feature_data.copy()
    long.columns = [f"{a}__{b}" for a, b in long.columns]
    long = long.reset_index().rename(columns={long.index.name or "index": "date"})
    return long

def latest_feature_value(
    data: pd.DataFrame,
    feature: str,
    ticker: OptionTicker,
) -> float:
    """Return the latest non-missing feature value for one ticker object."""

    column = (feature, ticker.symbol)

    if column not in data.columns:
        raise KeyError(f"Missing column {column!r}")

    values = data[column].dropna()

    if values.empty:
        raise ValueError(
            f"No usable values found for {feature} and {ticker.symbol}"
        )

    value = float(values.iloc[-1])

    if not np.isfinite(value):
        raise ValueError(
            f"Latest {feature} value for {ticker.symbol} is not finite"
        )

    return value

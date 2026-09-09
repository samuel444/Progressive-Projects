import numpy as np
import pandas as pd

from features.beta import rolling_beta, downside_beta, upside_beta, beta_change
from features.breadth import breadth_ma, new_high_low, advance_decline
from features.builder import build_features
from features.cross_sectional import cross_sectional_rank, cross_sectional_z_score
from features.momentum import momentum, momentum_change
from features.volatility import rolling_volatility, volatility_ratios


def make_price_data(length=320, seed=10):
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2023-01-02", periods=length)
    returns = rng.normal(0.0005, 0.012, length)
    close = 100 * np.cumprod(1 + returns)
    open_price = close * (1 + rng.normal(0, 0.002, length))
    high = np.maximum(open_price, close) * (1 + rng.uniform(0, 0.01, length))
    low = np.minimum(open_price, close) * (1 - rng.uniform(0, 0.01, length))
    volume = rng.integers(1_000_000, 5_000_000, length)

    return pd.DataFrame(
        {
            "Open": open_price,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
        },
        index=index,
    )


def make_market_data(index, seed=11):
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0004, 0.009, len(index))
    close = 200 * np.cumprod(1 + returns)
    return pd.DataFrame({"Close": close}, index=index)


def test_single_value_and_list_inputs():
    df = make_price_data()
    df = momentum(df, windows=20)
    df = momentum_change(df, windows=[20, 60], periods=[1, 5])

    assert "Momentum 20" in df.columns
    assert "Momentum 20 Change 1" in df.columns
    assert "Momentum 20 Change 5" in df.columns
    assert "Momentum 60 Change 1" in df.columns
    assert "Momentum 60 Change 5" in df.columns


def test_volatility_combinations():
    df = make_price_data()
    df = rolling_volatility(df, windows=[5, 20, 60])
    df = volatility_ratios(df, short_windows=[5, 20], long_windows=[20, 60])

    assert "Volatility Ratio 5 20" in df.columns
    assert "Volatility Ratio 5 60" in df.columns
    assert "Volatility Ratio 20 60" in df.columns
    assert "Volatility Ratio 20 20" not in df.columns


def test_beta_functions_with_supplied_market_data():
    df = make_price_data()
    market = make_market_data(df.index)

    df = rolling_beta(df, windows=[20, 60], market_df=market)
    df = downside_beta(df, windows=20, market_df=market)
    df = upside_beta(df, windows=20, market_df=market)
    df = beta_change(df, windows=20, periods=[1, 5], market_df=market)

    assert "Rolling Beta 20" in df.columns
    assert "Rolling Beta 60" in df.columns
    assert "Downside Beta 20" in df.columns
    assert "Upside Beta 20" in df.columns
    assert "Rolling Beta 20 Change 1" in df.columns
    assert df["Rolling Beta 20"].notna().sum() > 0
    assert df["Downside Beta 20"].notna().sum() > 0
    assert df["Upside Beta 20"].notna().sum() > 0


def test_feature_builder():
    df = make_price_data()
    original_columns = len(df.columns)

    df = build_features(df, groups=["returns", "momentum"])

    assert len(df.columns) > original_columns
    assert "Return Lag 1" in df.columns
    assert "Momentum 20" in df.columns


def test_cross_sectional_rank_and_z_score():
    dates = pd.date_range("2024-01-01", periods=3)
    rows = []

    for date in dates:
        for ticker, value in zip(["A", "B", "C"], [1.0, 2.0, 3.0]):
            rows.append({"Date": date, "Ticker": ticker, "Signal": value})

    df = pd.DataFrame(rows)
    df = cross_sectional_rank(df, "Signal")
    df = cross_sectional_z_score(df, "Signal")

    assert df.loc[df["Ticker"] == "C", "Signal Rank"].eq(1.0).all()
    assert np.isclose(df.groupby("Date")["Signal Cross Sectional Z Score"].mean(), 0).all()


def test_breadth_uses_requested_window_and_real_lows():
    index = pd.bdate_range("2024-01-01", periods=260)
    close = pd.DataFrame(
        {
            "A": np.arange(1, 261, dtype=float),
            "B": np.arange(261, 1, -1, dtype=float),
            "C": np.linspace(100, 120, 260),
        },
        index=index,
    )
    df = pd.concat({"Close": close}, axis=1)

    df = breadth_ma(df, windows=20)
    df = new_high_low(df, windows=20)
    df = advance_decline(df)

    assert ("Breadth MA 20", "") in df.columns
    assert ("New High Breadth 20", "") in df.columns
    assert ("New Low Breadth 20", "") in df.columns
    assert ("Advance Decline Difference 1", "") in df.columns

    last_high = df[("New High Breadth 20", "")].iloc[-1]
    last_low = df[("New Low Breadth 20", "")].iloc[-1]

    assert last_high > 0
    assert last_low > 0

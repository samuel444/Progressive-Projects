import numpy as np
import pandas as pd

from targets.barriers import first_hit_barrier
from targets.builder import build_targets
from targets.excursions import maximum_adverse_excursion, maximum_favourable_excursion
from targets.ranking import future_return_rank, top_quantile_label
from targets.returns import forward_return
from targets.volatility import future_volatility


def make_price_data(length=120):
    index = pd.bdate_range("2024-01-01", periods=length)
    close = np.linspace(100, 130, length)

    return pd.DataFrame(
        {
            "Open": close,
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Volume": np.full(length, 1_000_000),
        },
        index=index,
    )


def test_forward_return_alignment():
    df = make_price_data()
    df = forward_return(df, horizons=5)

    expected = df["Close"].iloc[5] / df["Close"].iloc[0] - 1

    assert np.isclose(df["Forward Return 5"].iloc[0], expected)
    assert df["Forward Return 5"].tail(5).isna().all()


def test_future_volatility_uses_future_returns():
    index = pd.bdate_range("2024-01-01", periods=10)
    close = pd.Series([100, 101, 100, 103, 102, 106, 104, 108, 107, 110], index=index)
    df = pd.DataFrame({"Close": close})
    df = future_volatility(df, horizons=3)

    returns = close.pct_change()
    expected = returns.iloc[1:4].std()

    assert np.isclose(df["Future Volatility 3"].iloc[0], expected)


def test_first_hit_barrier_labels_up_move_first():
    index = pd.bdate_range("2024-01-01", periods=8)
    df = pd.DataFrame({"Close": [100, 103, 106, 104, 98, 97, 96, 95]}, index=index)
    df = first_hit_barrier(df, upper=0.05, lower=-0.05, horizons=5)

    assert df["Barrier 5.0 -5.0 5"].iloc[0] == 1


def test_excursions():
    index = pd.bdate_range("2024-01-01", periods=6)
    df = pd.DataFrame({"Close": [100, 102, 105, 99, 98, 101]}, index=index)
    df = maximum_favourable_excursion(df, horizons=5)
    df = maximum_adverse_excursion(df, horizons=5)

    assert np.isclose(df["Maximum Favourable Excursion 5"].iloc[0], 0.05)
    assert np.isclose(df["Maximum Adverse Excursion 5"].iloc[0], -0.02)


def test_target_builder():
    df = make_price_data()
    df = build_targets(df, groups=["volatility", "direction"])

    assert "Future Volatility 20" in df.columns
    assert "Future Direction 20" in df.columns


def test_cross_sectional_future_rank():
    dates = pd.bdate_range("2024-01-01", periods=8)
    rows = []

    paths = {
        "A": [100, 101, 102, 103, 104, 105, 106, 107],
        "B": [100, 102, 104, 106, 108, 110, 112, 114],
        "C": [100, 100, 100, 100, 100, 100, 100, 100],
    }

    for ticker, prices in paths.items():
        for date, price in zip(dates, prices):
            rows.append({"Date": date, "Ticker": ticker, "Close": price})

    df = pd.DataFrame(rows).sort_values(["Ticker", "Date"]).reset_index(drop=True)
    df = future_return_rank(df, horizons=2)
    df = top_quantile_label(df, horizons=2, quantiles=1 / 3)

    first_date = df[df["Date"] == dates[0]].sort_values("Future Return Rank 2")

    assert first_date.iloc[-1]["Ticker"] == "B"
    assert df[(df["Date"] == dates[0]) & (df["Ticker"] == "B")]["Top 33 Percent Future Return 2"].iloc[0] == 1

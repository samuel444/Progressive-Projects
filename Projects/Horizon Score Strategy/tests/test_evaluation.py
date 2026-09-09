import numpy as np
import pandas as pd
import pytest
from horizon_score.evaluation import evaluate_backtest, benchmark_returns


def fixture(n=300):
    dates = pd.bdate_range("2020-01-01", periods=n)
    r = np.random.default_rng(42).normal(0.0004, 0.01, n)
    frame = pd.DataFrame({"Date": dates, "Return": r, "A": 0.6, "B": 0.4})
    market = pd.DataFrame({"Date": dates, "Return": r * 0.7 + 0.0001})
    regimes = {
        "Growth": {
            "Expansion": [
                (str(dates[0].date()), str(dates[80].date())),
                (str(dates[170].date()), str(dates[-1].date())),
            ],
            "Neutral": [],
            "Contraction": [(str(dates[81].date()), str(dates[169].date()))],
            "Unknown": [],
        }
    }
    return frame, regimes, market


def test_metrics_costs_alignment_and_episodes(tmp_path):
    frame, regimes, market = fixture()
    result = evaluate_backtest(
        frame, regimes, market.sample(frac=1, random_state=3), output_dir=tmp_path
    )
    assert result["summary"]["Total Return"] == pytest.approx(
        (1 + frame.Return).prod() - 1
    )
    assert result["benchmark"]["Beta"] == pytest.approx(1 / 0.7)
    assert result["portfolio"]["Average HHI Concentration"] == pytest.approx(0.52)
    assert result["portfolio"]["Average Effective Holdings"] == pytest.approx(1 / 0.52)
    assert result["portfolio"]["Annualised Turnover"] == pytest.approx(0.5 / 300 * 252)
    net = frame.Return.copy()
    net.iloc[0] -= 0.001
    assert result["costs"]["Net Total Return"] == pytest.approx((1 + net).prod() - 1)
    assert len(result["regime_episodes"]) == 3
    assert result["regimes"]["Max Drawdown"].isna().all()
    assert result["rolling"]["63 Rolling Return"].iloc[:62].isna().all()
    assert len(list(tmp_path.glob("*.csv"))) == 11
    assert result["regimes"].Days.sum() == 300


def test_missing_market_dates_are_not_positionally_aligned():
    f, r, m = fixture()
    out = evaluate_backtest(f, r, m.iloc[::2])
    assert out["benchmark"]["Aligned Observations"] == 150
    assert out["daily"]["Market Return"].isna().sum() == 150
    assert out["rolling"]["63 Rolling Beta"].isna().all()


def test_multiindex_and_missing_close_no_fill():
    dates = pd.bdate_range("2020-01-01", periods=4)
    data = pd.DataFrame(
        [100, 110, np.nan, 121],
        index=dates,
        columns=pd.MultiIndex.from_tuples([("Close", "^GSPC")]),
    )
    r = benchmark_returns(data)
    assert r.iloc[1] == pytest.approx(0.1)
    assert r.iloc[2:].isna().all()


def test_invalid_inputs_and_cash():
    f, r, m = fixture()
    f[["A", "B"]] = 0.0
    out = evaluate_backtest(f, {}, m)
    assert np.isnan(out["portfolio"]["Average Effective Holdings"])
    f.loc[0, "A"] = -0.1
    with pytest.raises(ValueError, match="weights"):
        evaluate_backtest(f, {}, m)
    f.loc[0, "A"] = 0
    with pytest.raises(ValueError, match="Overlapping"):
        evaluate_backtest(
            f,
            {
                "Growth": {
                    "A": [("2020-01-01", "2020-02-01")],
                    "B": [("2020-01-02", "2020-03-01")],
                }
            },
            m,
        )

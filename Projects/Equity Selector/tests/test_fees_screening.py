import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from equity_selector.portfolio import portfolio_returns_from_scores
from equity_selector.stages import precise
from equity_selector.validation import screening_training_rows, train_validation_test_split
from screening.screening_features import missingness
from screening.screening_stocks import missingness_stocks


def test_fees_charge_entry_switch_and_exit_and_match_precise(monkeypatch):
    dates = pd.date_range("2024-01-01", periods=5)
    data = pd.DataFrame(
        [
            {
                "Date": date,
                "Ticker": ticker,
                "Return": 0.0,
                "Stock_Score": float(ticker == selected),
            }
            for date, selected in zip(dates, ["A", "A", "B", None, None])
            for ticker in ["A", "B"]
        ]
    )
    gross = portfolio_returns_from_scores(data, max_weight=1, concentration_penalty=0)
    net = portfolio_returns_from_scores(
        data, max_weight=1, concentration_penalty=0, trading_fee=0.001
    )
    # $1 entry, no trade, $1 sale + $1 purchase, $1 exit, on $1 target NAV.
    expected = np.array([-0.001, 0, -0.002, -0.001])
    np.testing.assert_allclose(net.Return, expected)
    monkeypatch.setattr(precise, "FE_TRADING_FEE", 0.001)
    monkeypatch.setattr(precise, "FE_RF_ANNUAL", 0)
    stats, traded = precise.fe_costs(gross)
    np.testing.assert_allclose(traded, [1, 0, 2, 1])
    np.testing.assert_allclose(gross.Return - 0.001 * traded, net.Return)
    assert stats["Sharpe at Realistic Costs"] == pytest.approx(
        expected.mean() / expected.std(ddof=1) * np.sqrt(252)
    )
    assert stats["Annual Turnover"] == 126


def test_break_even_percentage_is_economic_root(monkeypatch):
    monkeypatch.setattr(precise, "FE_TRADING_FEE", 0.001)
    frame = pd.DataFrame(
        {
            "Date": pd.date_range("2024-01-01", periods=2),
            "Return": [0.01, 0.02],
            "A": [1.0, 0.0],
            "B": [0.0, 1.0],
        }
    )
    stats, traded = precise.fe_costs(frame)
    fee = stats["Break-Even Cost (%)"] / 100
    assert np.prod(1 + frame.Return - fee * traded) == pytest.approx(1)
    assert stats["Break-Even Cost (bps)"] == pytest.approx(100 * stats["Break-Even Cost (%)"])


@pytest.mark.parametrize("value", ["nan", "inf", "-0.1"])
def test_invalid_cli_fee_fails_before_reading_databases(value, tmp_path):
    with pytest.raises(SystemExit) as error:
        precise.main(["--data-dir", str(tmp_path), "--trading-fee", value])
    assert error.value.code == 2
    assert list(tmp_path.iterdir()) == []


def test_screening_decisions_ignore_heldout_values():
    dates = pd.date_range("2024-01-01", periods=10)
    panel = pd.DataFrame({"Date": dates, "Ticker": "A", "feature": np.arange(10, dtype=float)})
    changed = panel.copy()
    changed.loc[6:8, "feature"] = np.nan
    # Full-sample missingness would reject this feature; training-only screening must not.
    assert missingness(changed[["feature"]])[1] == ["feature"]
    before = screening_training_rows(panel)
    after = screening_training_rows(changed)
    assert_frame_equal(before, after)
    assert_frame_equal(before.reset_index(drop=True), train_validation_test_split(panel)[0])
    assert missingness(after[["feature"]])[1] == []
    stock = changed.set_index("Date")[["feature"]]
    assert missingness_stocks(stock) == "drop"
    assert missingness_stocks(screening_training_rows(stock, dates)) == "keep"
    after.iloc[0, after.columns.get_loc("feature")] = -99
    assert changed.feature.iloc[0] == 0

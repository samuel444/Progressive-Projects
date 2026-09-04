import sqlite3

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from equity_selector.database import read_table, write_frame
from equity_selector.metrics import performance_metrics
from equity_selector.portfolio import portfolio_returns_from_scores
from equity_selector.simulations import align_simulation_results
from main_package.backtesting import (
    create_models_and_predictions,
    run_portfolio_backtest_from_predictions,
)


def test_cached_signals_aggregation_index_and_configuration(tmp_path):
    rows = []
    dates = pd.date_range("2024-01-01", periods=3)
    for day, date in enumerate(dates):
        for ticker, signal, returns in [
            ("A", 2.0, [0.9, -0.2, 0.3]),
            ("B", 1.0, [-0.8, 0.4, -0.1]),
        ]:
            for portfolio_type in ["ALPHA", "RELATIVE_ALPHA"]:
                rows.append(
                    {
                        "Date": date,
                        "Ticker": ticker,
                        "Return": returns[day],
                        "Portfolio Target Type": portfolio_type,
                        "Signal": signal,
                        "Direction Signal": 1.0,
                        "Horizon Score": 1.0,
                    }
                )
    rows.append({**rows[0], "Signal": np.nan})
    frame = pd.DataFrame(rows).sample(frac=1, random_state=4)
    frame.index = np.arange(len(frame)) + 100
    original = frame.copy(deep=True)
    actual = run_portfolio_backtest_from_predictions(
        frame, max_weight=1.0, concentration_penalty=0.0, trading_fee=0.02, annualisation=12
    )
    # Scores in ratio 2:1; next-day returns: 0, 1/6. Full entry fee=.02.
    expected = performance_metrics([-0.02, 1 / 6], annualisation=12)
    assert actual["Strategy Return"] == pytest.approx(expected["Return"])
    assert actual["Sharpe Ratio"] == pytest.approx(expected["Sharpe Ratio"])
    assert actual["Max Drawdown"] == pytest.approx(-0.02)
    assert_frame_equal(frame, original)
    # Setting all used type weights to zero deliberately holds cash.
    cash = run_portfolio_backtest_from_predictions(
        frame, type_values={"ALPHA": 0.0, "RELATIVE_ALPHA": 0.0}
    )
    assert cash["Strategy Return"] == 0
    path = tmp_path / "cache.db"
    with sqlite3.connect(path) as connection:
        write_frame(frame, "Signals", connection, if_exists="replace")
    restored = read_table(path, "Signals")
    assert (
        run_portfolio_backtest_from_predictions(
            restored, max_weight=1.0, concentration_penalty=0.0, trading_fee=0.02, annualisation=12
        )
        == actual
    )


def test_fit_predict_cache_score_portfolio_integration(tmp_path):
    target = "Forward Return 1"
    dates = pd.date_range("2024-01-01", periods=24)
    rows = []
    for i, date in enumerate(dates):
        for j, ticker in enumerate(["A", "B", "C"]):
            x = i + j / 10
            rows.append(
                {
                    "Date": date,
                    "Ticker": ticker,
                    "Close": 100 + x,
                    "Return": [0.01, -0.02, 0.03][j],
                    "x": x,
                    target: 2 * x + 1,
                    "Split": "TRAIN" if i < 20 else "BACKTEST",
                }
            )
    frame = pd.DataFrame(rows).sample(frac=1, random_state=7)
    original = frame.copy(deep=True)
    metadata = pd.DataFrame(
        [
            {
                "Target": target,
                "Model": "OLS",
                "Parameters": "{}",
                "Target Type": "ALPHA",
                "Statistical Type": "continuous",
                "Horizon": 1,
                "Horizon Score": 1.0,
                "Quality Score": 1.0,
            }
        ]
    )
    result = create_models_and_predictions(frame, metadata, {target: ["x"]}, strict=True)
    predictions = result["predictions"]
    # Training has 20 dates, purge removes the final label-bearing date: 19*3 rows.
    assert result["model_summary"]["Training Rows"].iloc[0] == 57
    expected = predictions.merge(
        frame[["Date", "Ticker", "x"]], on=["Date", "Ticker"], validate="one_to_one"
    )
    assert np.allclose(expected.Prediction, 2 * expected.x + 1)
    path = tmp_path / "pipeline.db"
    with sqlite3.connect(path) as connection:
        write_frame(predictions, "Predictions", connection, if_exists="replace")
    restored = read_table(path, "Predictions")
    restored["Stock_Score"] = restored.groupby("Date")["Prediction"].rank(pct=True)
    portfolio = portfolio_returns_from_scores(restored, max_weight=1.0, concentration_penalty=0)
    assert portfolio["Return"].tolist() == pytest.approx([0.01 / 6 - 0.02 / 3 + 0.03 / 2] * 3)
    assert performance_metrics(portfolio["Return"])["Return"] == pytest.approx(
        (1 + 0.01 / 6 - 0.02 / 3 + 0.03 / 2) ** 3 - 1
    )
    assert_frame_equal(frame, original)
    # Direct API must reject overlapping TRAIN/BACKTEST date sets.
    overlap = frame.copy()
    overlap.loc[overlap["Split"].eq("BACKTEST"), "Date"] = dates[0]
    with pytest.raises(ValueError, match="precede"):
        create_models_and_predictions(overlap, metadata, {target: ["x"]}, strict=True)


def test_simulation_alignment_by_id():
    strategies = pd.DataFrame({"Simulation ID": [2, 1], "Return": [0.2, 0.1]})
    benchmarks = pd.DataFrame({"Simulation ID": [1, 2], "Return": [0.01, 0.02]})
    left, right = align_simulation_results(strategies, benchmarks)
    assert (left["Return"] - right["Return"]).tolist() == pytest.approx([0.18, 0.09])
    with pytest.raises(ValueError, match="unique"):
        align_simulation_results(strategies, pd.concat([benchmarks, benchmarks]))

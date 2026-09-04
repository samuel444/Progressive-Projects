import json
import sqlite3

import numpy as np
import pandas as pd
import pytest

from equity_selector.database import write_frame
from equity_selector.stages import precise


@pytest.mark.parametrize(
    "fee_args", [["--cost-bps", "1"], ["--trading-fee", "0.0001"], ["--fee-percent", "0.01"]]
)
def test_precise_end_to_end_temporary_databases(tmp_path, fee_args):
    dates = pd.bdate_range("2024-01-01", periods=35)

    def universe(tickers):
        return pd.DataFrame(
            [
                {
                    "Date": date,
                    "Ticker": ticker,
                    "Portfolio Target Type": "ALPHA",
                    "Horizon Key": "1d",
                    "Signal": j + 1.0,
                    "Return": 0.001 + 0.004 * np.sin(i + j),
                }
                for i, date in enumerate(dates)
                for j, ticker in enumerate(tickers)
            ]
        )

    grid = pd.DataFrame(
        [
            {
                "Simulation ID": 1,
                "Horizon Score Index": 0,
                "Type Configuration": "Balanced",
                "Rebalance Multiplier": 0.5,
                "Max Weight": 0.4,
                "Concentration Penalty": 0.1,
            },
            {
                "Simulation ID": 2,
                "Horizon Score Index": 0,
                "Type Configuration": "Balanced",
                "Rebalance Multiplier": 0.5,
                "Max Weight": 0.5,
                "Concentration Penalty": 0.2,
            },
        ]
    )
    with sqlite3.connect(tmp_path / "Backtest_Database.db") as connection:
        for table, frame in [
            ("Stocks", universe(["A", "B", "C"])),
            ("Market", universe(["M"])),
            ("Unseen", universe(["X", "Y", "Z"])),
            ("Passed Strategies", grid.iloc[:1]),
        ]:
            write_frame(frame, table, connection, if_exists="replace")
    with sqlite3.connect(tmp_path / "Portfolio_Simulation_Results.db") as connection:
        write_frame(grid, "Stock Simulation Results", connection, if_exists="replace")
        benchmark = grid.copy()
        benchmark["Backtest Quality"] = [0.1, 0.2]
        write_frame(benchmark, "Market Simulation Results", connection, if_exists="replace")
    (tmp_path / "Top_Horizon_Scores.txt").write_text("[{'ALPHA': {'1d': 1.0}}]")
    inputs = {p.name: p.read_bytes() for p in tmp_path.glob("*.db")}
    result = precise.main(
        [
            "--data-dir",
            str(tmp_path),
            "--output-dir",
            str(tmp_path / "result"),
            "--dsr-sample-size",
            "2",
            *fee_args,
        ]
    )
    assert result["Simulation ID"].tolist() == [1]
    assert np.isfinite(result["Standard Annual Sharpe"].iloc[0])
    assert (tmp_path / "result/Final_Strategy_Evaluation.db").exists()
    returns = pd.read_csv(tmp_path / "result/Final_Strategy_Daily_Returns.csv")
    assert len(returns) == 34
    assumptions = json.loads((tmp_path / "result/Evaluation_Assumptions.json").read_text())
    assert assumptions["trading_fee_fraction"] == pytest.approx(0.0001)
    assert assumptions["trading_fee_percent"] == pytest.approx(0.01)
    assert "scores at t earn Return at t+1" in assumptions["core_conventions"]
    for name, before in inputs.items():
        assert (tmp_path / name).read_bytes() == before

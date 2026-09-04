import sqlite3
import json

import pandas as pd
import pytest

from equity_selector.database import write_frame, read_table
from equity_selector.database_audit import audit_databases
from equity_selector.frozen import evaluate_frozen


def fixture(tmp_path):
    selection = tmp_path / "selection.db"
    cache = tmp_path / "final.db"
    horizons = tmp_path / "horizons.txt"
    chosen = pd.DataFrame(
        [
            {
                "Simulation ID": i,
                "Horizon Score Index": 0,
                "Type Configuration": "Ranking",
                "Rebalance Multiplier": 0.0,
                "Max Weight": cap,
                "Concentration Penalty": 0.0,
            }
            for i, cap in [(1, 1.0), (2, 0.5)]
        ]
    )
    data = pd.DataFrame(
        [
            {
                "Date": date,
                "Ticker": ticker,
                "Portfolio Target Type": "ALPHA",
                "Horizon Key": "1d",
                "Signal": score,
                "Return": 0.01,
            }
            for date in pd.date_range("2023-01-02", periods=4)
            for ticker, score in [("A", 1.0), ("B", 2.0)]
        ]
    )
    with sqlite3.connect(selection) as c:
        write_frame(chosen, "Passed Strategies", c, if_exists="replace")
        write_frame(pd.DataFrame({"Date": ["2022-12-30"]}), "Stocks", c, if_exists="replace")
    with sqlite3.connect(cache) as c:
        write_frame(data, "Stocks", c, if_exists="replace")
    horizons.write_text("[{'ALPHA': {'1d': 1.0}}]")
    return dict(
        selection_database=selection,
        cache_database=cache,
        horizon_file=horizons,
        type_configurations=[
            {
                "Name": "Ranking",
                "Ranking": 1.0,
                "Direction": 0.0,
                "Risk": 0.0,
                "Opportunity": 0.0,
                "Special": 0.0,
            }
        ],
        start="2023-01-01",
        end="2023-12-31",
        output_dir=tmp_path / "output",
    )


def test_frozen_final_keeps_every_selection_and_replays_metrics(tmp_path):
    settings = fixture(tmp_path)
    originals = {
        name: settings[name].read_bytes()
        for name in ["selection_database", "cache_database", "horizon_file"]
    }
    summary = evaluate_frozen(**settings)
    assert summary["Simulation ID"].tolist() == [1, 2]
    daily = read_table(tmp_path / "output/Frozen_Final_Evaluation.db", "Daily Returns")
    assert daily.loc[daily["Simulation ID"].eq(1), "Return"].tolist() == pytest.approx(
        [0.009, 0.01, 0.01]
    )
    assert summary.Return.iloc[0] == pytest.approx(1.009 * 1.01 * 1.01 - 1)
    for name, before in originals.items():
        assert settings[name].read_bytes() == before
    contract = json.loads((tmp_path / "output/audit-contract.json").read_text())
    report = audit_databases([tmp_path / "output/Frozen_Final_Evaluation.db"], contract=contract)
    assert report["status"] == "PASS"
    with pytest.raises(FileExistsError):
        evaluate_frozen(**settings)


def test_final_overlap_and_wrong_dates_fail_before_output(tmp_path):
    settings = fixture(tmp_path)
    settings["start"] = "2022-01-01"
    with pytest.raises(ValueError, match="after all dates"):
        evaluate_frozen(**settings)
    assert not settings["output_dir"].exists()
    settings["start"] = "2023-01-04"
    with pytest.raises(ValueError, match="entirely within"):
        evaluate_frozen(**settings)
    assert not settings["output_dir"].exists()

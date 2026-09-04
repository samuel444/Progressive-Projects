import ast
import importlib
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from equity_selector.database import read_table, write_frame
from equity_selector.files import commit_with_text
from equity_selector.pruning import is_numeric_parameter, violates_rule


@pytest.mark.parametrize(
    "stage",
    ["data", "intraday", "training", "final_test", "horizons", "cache", "simulations", "precise"],
)
def test_stage_import_has_no_research_io(stage, monkeypatch):
    import yfinance

    def forbidden(*args, **kwargs):
        raise AssertionError("Import attempted database/network/prompt I/O")

    monkeypatch.setattr(sqlite3, "connect", forbidden)
    monkeypatch.setattr(yfinance, "download", forbidden)
    monkeypatch.setattr("builtins.input", forbidden)
    module = importlib.import_module("equity_selector.stages." + stage)
    assert callable(getattr(module, "run", getattr(module, "main", None)))


def test_legacy_cli_help_without_data(tmp_path):
    root = Path(__file__).parents[1]
    for path in root.glob("*.py"):
        result = subprocess.run(
            [sys.executable, str(path), "--help"],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=tmp_path,
        )
        assert result.returncode == 0, (path.name, result.stderr)
        assert "usage:" in result.stdout


def test_intraday_conversion_preserves_rows_and_updates_mapping(tmp_path, monkeypatch):
    module = importlib.import_module("equity_selector.stages.intraday")
    database = tmp_path / "intraday.db"
    mapping = tmp_path / "Selected_Features.txt"
    universe = "Fixture"
    mapping.write_text("{'Future Direction 1': ['Momentum 1', 'Remove Me']}")
    frame = pd.DataFrame(
        {
            "Date": pd.date_range("2024-01-01 09:30", periods=4, freq="min"),
            "Ticker": "A",
            "Momentum 1": [1.0, 2.0, 3.0, 4.0],
            "Future Direction 1": [1.0, 1.0, 0.0, 0.0],
            "Remove Me": [9.0] * 4,
        }
    )
    with sqlite3.connect(database) as connection:
        write_frame(frame, universe, connection, if_exists="replace")
    monkeypatch.setattr(module, "DATABASE", database)
    monkeypatch.setattr(module, "SELECTED_FEATURES_FILE", mapping)
    monkeypatch.setattr(module, "STOCK_TYPE", universe)
    monkeypatch.setattr(module, "STOCK_TYPE_INDICES", {universe: 0})
    # Explicit rules isolate transactional/session boundary behavior from naming heuristics.
    monkeypatch.setattr(
        module,
        "analyse_columns",
        lambda columns: (
            [column for column in columns if column != "Remove Me"],
            {
                "Momentum 1": {"first_rows": 1, "last_rows": 0},
                "Future Direction 1": {"first_rows": 0, "last_rows": 1},
            },
        ),
    )
    module.clean_intraday_table()
    result = read_table(database, universe)
    assert len(result) == 4
    assert pd.isna(result["Momentum 1"].iloc[0])
    assert pd.isna(result["Future Direction 1"].iloc[-1])
    assert result["Momentum 1"].iloc[1] == 2
    assert "Remove Me" not in result
    from equity_selector.feature_mapping import load_feature_mapping

    assert load_feature_mapping(mapping, universe) == {"Future Direction 1": ["Momentum 1"]}
    before = mapping.read_text()
    monkeypatch.setattr(module, "analyse_columns", lambda _: (["Date", "Ticker", "Missing"], {}))
    with pytest.raises((sqlite3.Error, ValueError)):
        module.clean_intraday_table()
    assert mapping.read_text() == before
    assert len(read_table(database, universe)) == 4


def test_file_failure_rolls_back_sql(tmp_path, monkeypatch):
    import equity_selector.files as files

    database = tmp_path / "db.sqlite"
    mapping = tmp_path / "mapping.txt"
    mapping.write_text("original")
    with sqlite3.connect(database) as connection:
        write_frame(pd.DataFrame({"Value": [1]}), "Table", connection, if_exists="replace")

    def fail(*args):
        raise OSError("Injected mapping write failure")

    monkeypatch.setattr(files, "atomic_write_text", fail)
    with sqlite3.connect(database) as connection:
        write_frame(pd.DataFrame({"Value": [2]}), "Table", connection, if_exists="replace")
        with pytest.raises(OSError):
            commit_with_text(connection, mapping, "changed")
    assert read_table(database, "Table").Value.tolist() == [1]
    assert mapping.read_text() == "original"


def test_bool_and_categorical_parameters_are_not_numeric_tails():
    assert not is_numeric_parameter(pd.Series([True, False]))
    assert not is_numeric_parameter(pd.Series(["1", "2"]))
    assert is_numeric_parameter(pd.Series([1, 2], dtype=object))
    rule = {"rule_type": "upper_tail", "parameter": "max_depth", "operator": ">", "value": 10}
    assert not violates_rule({"max_depth": None}, rule)
    assert not violates_rule({"max_depth": "unlimited"}, rule)
    assert violates_rule({"max_depth": 20}, rule)


def test_missing_credentials_fail_before_network(monkeypatch):
    import main_package.signals as signals

    monkeypatch.setattr(signals, "BOT_TOKEN", "")
    monkeypatch.setattr(signals, "API_KEY", "")
    monkeypatch.setattr(
        signals.requests, "post", lambda *args, **kwargs: pytest.fail("network call")
    )
    monkeypatch.setattr(
        signals.requests, "get", lambda *args, **kwargs: pytest.fail("network call")
    )
    with pytest.raises(ValueError, match="TELEGRAM"):
        signals.send_notification("test")
    with pytest.raises(ValueError, match="TRADING212"):
        signals._get_account()


def test_training_and_final_test_with_no_targets(tmp_path, monkeypatch):
    monkeypatch.setenv("EQUITY_SELECTOR_DATA_DIR", str(tmp_path))
    (tmp_path / "Selected_Features.txt").write_text("\n".join(["{}"] * 11))
    (tmp_path / "Validation_Model_Fits").mkdir()
    monkeypatch.setattr("builtins.input", lambda *args: pytest.fail("Empty run prompted for input"))
    training = importlib.import_module("equity_selector.stages.training")
    training.run()
    assert training.targets == []
    final = importlib.import_module("equity_selector.stages.final_test")
    final.run()
    result = read_table(tmp_path / "Final_Test_Results.db", "High Liquidity 30 Passed Test Results")
    assert result.empty
    assert {"Target", "Horizon", "Quality Score"}.issubset(result.columns)


def test_final_model_test_with_deterministic_database(tmp_path, monkeypatch):
    import numpy as np

    monkeypatch.setenv("EQUITY_SELECTOR_DATA_DIR", str(tmp_path))
    target = "Forward Return 1"
    mapping = {target: ["x"]}
    (tmp_path / "Selected_Features.txt").write_text(str(mapping))
    (tmp_path / "Validation_Model_Fits").mkdir()
    data = pd.DataFrame(
        {
            "Date": pd.date_range("2020-01-01", periods=50),
            "Ticker": "A",
            "x": np.arange(50.0),
            target: 2 * np.arange(50.0) + 1,
        }
    )
    with sqlite3.connect(tmp_path / "Features_Targets_Data.db") as connection:
        write_frame(data, "High Liquidity 30", connection, if_exists="replace")
    leaderboard = pd.DataFrame(
        [
            {
                "Model": "OLS",
                "Parameters": "{}",
                "Rank": 1,
                "Rank IC Mean": 1.0,
                "Testing Eligible": True,
            }
        ]
    )
    with sqlite3.connect(tmp_path / "Validation_Model_Fits/High_Liquidity_30.db") as connection:
        write_frame(leaderboard, target, connection, if_exists="replace")
    monkeypatch.setattr("builtins.input", lambda *args: "1")
    final = importlib.import_module("equity_selector.stages.final_test")
    final.run()
    result = read_table(tmp_path / "Final_Test_Results.db", "Final Test Results High Liquidity 30")
    assert result["R2"].tolist() == pytest.approx([1.0])
    assert result["Rank IC"].tolist() == pytest.approx([1.0])
    assert read_table(tmp_path / "Final_Test_Results.db", "Errors").empty

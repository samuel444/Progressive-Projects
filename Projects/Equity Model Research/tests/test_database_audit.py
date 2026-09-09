import sqlite3
import json
from pathlib import Path

import pandas as pd
import pytest

from equity_selector.database import write_frame
from equity_selector.database_audit import audit_databases, main


def database(path, tables):
    with sqlite3.connect(path) as connection:
        for name, frame in tables.items():
            write_frame(frame, name, connection, if_exists="replace")
    return path


def codes(report, status):
    return {c["code"] for c in report["checks"] if c["status"] == status}


def test_valid_signal_cache_readonly_and_contract(tmp_path):
    data = pd.DataFrame(
        {
            "Date": ["2024-01-01", "2024-01-02"],
            "Ticker": ["A", "A"],
            "Return": [0.0, 0.1],
            "Signal": [1.0, 2.0],
            "Portfolio Target Type": "ALPHA",
            "Horizon Key": "1d",
        }
    )
    path = database(tmp_path / "Backtest_Database.db", {"Stocks": data, "Market": data})
    before = path.read_bytes()
    contract = {
        path.name: {
            "Stocks": {
                "date_min": "2024-01-01",
                "date_max": "2024-01-02",
                "expected_rows": 2,
                "required_columns": ["Return"],
            }
        }
    }
    report = audit_databases([path], contract=contract)
    assert report["status"] == "PASS"
    assert path.read_bytes() == before
    assert main([str(path), "--report", str(tmp_path / "report.json")]) == 0
    assert json.loads((tmp_path / "report.json").read_text())["status"] == "PASS"


def test_duplicate_alignment_and_invalid_return_detected(tmp_path):
    frame = pd.DataFrame(
        {
            "Date": ["2024-01-01"] * 2,
            "Ticker": "A",
            "Return": [-2.0, 0.1],
            "Signal": [1.0, 1.0],
            "Portfolio Target Type": "ALPHA",
            "Horizon Key": "1d",
        }
    )
    path = database(tmp_path / "signals.db", {"Stocks": frame})
    report = audit_databases([path])
    assert {"row_identity", "return_alignment", "metric_domain"} <= codes(report, "FAIL")


def test_fold_summary_independent_replay(tmp_path):
    folds = pd.DataFrame(
        {
            "Model": ["Ridge"] * 2,
            "Parameters": ['{"alpha":1}', "{'alpha':1}"],
            "Fold": [1, 2],
            "Target": "T",
            "RMSE": [1.0, 3.0],
            "Train End": ["2020-01-01"] * 2,
            "Validation Start": ["2021-01-01"] * 2,
            "Validation End": ["2021-01-31"] * 2,
        }
    )
    summary = pd.DataFrame(
        {
            "Model": ["Ridge"],
            "Parameters": ['{"alpha":1}'],
            "Target": ["T"],
            "RMSE Mean": [2.0],
            "RMSE Std": [2**0.5],
            "Fold": [2],
        }
    )
    path = database(tmp_path / "fits.db", {"T__folds": folds, "T__search": summary})
    report = audit_databases([path])
    assert "fold_summary_replay" in codes(report, "PASS")
    summary["RMSE Mean"] = 20
    database(path, {"T__search": summary})
    assert "fold_summary_replay" in codes(audit_databases([path]), "FAIL")


def test_malformed_parameters_never_execute_and_duplicates_canonicalized(tmp_path):
    marker = tmp_path / "executed"
    frame = pd.DataFrame(
        {
            "Model": ["Ridge"] * 3,
            "Parameters": [
                '{"alpha":1}',
                "{'alpha': 1}",
                f"__import__('pathlib').Path('{marker}').touch()",
            ],
        }
    )
    path = database(tmp_path / "fits.db", {"Results": frame})
    assert {"parameters_parse", "model_identity"} <= codes(audit_databases([path]), "FAIL")
    assert not marker.exists()


def test_empty_missing_and_partial_are_not_reported_as_success(tmp_path):
    empty = tmp_path / "empty.db"
    empty.touch()
    missing = tmp_path / "missing.db"
    report = audit_databases([empty, missing])
    assert {"empty_database", "missing_database"} <= codes(report, "FAIL")
    assert not missing.exists()
    path = database(
        tmp_path / "large.db", {"Rows": pd.DataFrame({"Date": ["2024-01-01", "2024-01-02"]})}
    )
    assert audit_databases([path], max_rows=1)["status"] == "INCOMPLETE"


def test_simulation_rows_align_by_id_not_position(tmp_path):
    path = database(
        tmp_path / "Portfolio_Simulation_Results.db",
        {
            "Stock Simulation Results": pd.DataFrame({"Simulation ID": [1, 2]}),
            "Market Simulation Results": pd.DataFrame({"Simulation ID": [2, 1]}),
        },
    )
    assert "simulation_id_alignment" in codes(audit_databases([path]), "PASS")
    database(path, {"Market Simulation Results": pd.DataFrame({"Simulation ID": [2, 3]})})
    assert "simulation_id_alignment" in codes(audit_databases([path]), "FAIL")


def test_fold_dates_and_missing_evidence(tmp_path):
    frame = pd.DataFrame(
        {
            "Model": ["Ridge"],
            "Parameters": ["{}"],
            "Fold": [1],
            "Train End": ["2024-02-01"],
            "Validation Start": ["2024-01-01"],
            "Validation End": ["2024-01-31"],
        }
    )
    path = database(tmp_path / "fit.db", {"T__folds": frame})
    assert "fold_chronology" in codes(audit_databases([path]), "FAIL")
    database(path, {"T__folds": frame.drop(columns=["Train End"])})
    assert audit_databases([path])["status"] == "INCOMPLETE"


def test_report_cannot_overwrite_source(tmp_path):
    path = database(tmp_path / "input.db", {"Rows": pd.DataFrame({"Date": ["2024-01-01"]})})
    before = path.read_bytes()
    with pytest.raises(SystemExit):
        main([str(path), "--report", str(path)])
    assert path.read_bytes() == before


def test_actual_legacy_numpy_nan_parameters_are_nullable(tmp_path):
    frame = pd.DataFrame(
        {
            "Model": ["Hist Gradient Boosting"],
            "Parameters": ["{'max_depth': np.float64(nan), 'max_iter': np.int64(100)}"],
        }
    )
    path = database(tmp_path / "legacy.db", {"Results": frame})
    report = audit_databases([path])
    assert "parameters_parse" in codes(report, "PASS")
    from equity_selector.parameters import parse_parameters

    assert parse_parameters(frame.Parameters.iloc[0]) == {"max_depth": None, "max_iter": 100}


def test_daily_returns_replay_catches_incorrect_sharpe_and_initial_drawdown(tmp_path):
    daily = pd.DataFrame({"Date": ["2024-01-01", "2024-01-02"], "Return": [-0.1, 0.2]})
    summary = pd.DataFrame({"Return": [0.08], "Max Drawdown": [-0.1]})
    path = database(tmp_path / "portfolio.db", {"Daily": daily, "Summary": summary})
    contract = {
        path.name: {
            "_metric_replays": [
                {
                    "daily_table": "Daily",
                    "summary_table": "Summary",
                    "metrics": {"Return": "Return", "Max Drawdown": "Max Drawdown"},
                }
            ]
        }
    }
    assert "portfolio_metric_replay" in codes(audit_databases([path], contract=contract), "PASS")
    summary["Max Drawdown"] = 0
    database(path, {"Summary": summary})
    assert "portfolio_metric_replay" in codes(audit_databases([path], contract=contract), "FAIL")

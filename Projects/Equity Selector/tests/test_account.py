import sqlite3
import json

import pandas as pd
import pytest

from equity_selector.account import gbp_account_returns
from equity_selector.database import read_table, write_frame
from equity_selector.database_audit import audit_databases
from equity_selector.frozen import evaluate_frozen
from equity_selector.preparation import prepare_research
from test_frozen_final import fixture


def fx_file(tmp_path, rates=(0.8, 0.8, 1.0, 0.8)):
    path = tmp_path / "fx.csv"
    pd.DataFrame(
        {"Date": pd.date_range("2023-01-02", periods=len(rates)), "GBP_per_USD": rates}
    ).to_csv(path, index=False)
    return path


def test_currency_movement_and_conversion_fees_with_idle_usd(tmp_path):
    frame = pd.DataFrame(
        {"Date": pd.date_range("2023-01-03", periods=3), "Return": [0.0, 0.0, 0.0]}
    )
    before = frame.copy(deep=True)
    out = gbp_account_returns(frame, fx_file=fx_file(tmp_path), initial_date="2023-01-02")
    # GBP 4000 becomes GBP 3994 after entry; retained USD gains 25% in GBP,
    # then loses 20%; a single final conversion costs 0.15% again.
    assert out["GBP Equity"].tolist() == pytest.approx([3994.0, 4992.5, 4000 * 0.9985**2])
    assert out.Return.tolist() == pytest.approx([-0.0015, 0.25, 0.8 * 0.9985 - 1])
    pd.testing.assert_frame_equal(frame, before)


def test_fx_alignment_is_by_date_and_missing_rates_fail(tmp_path):
    path = fx_file(tmp_path)
    fx = pd.read_csv(path).iloc[::-1]
    fx.to_csv(path, index=False)
    frame = pd.DataFrame({"Date": ["2023-01-03", "2023-01-04"], "Return": [0.1, 0.0]})
    result = gbp_account_returns(
        frame, fx_file=path, initial_date="2023-01-02", convert_back_at_end=False
    )
    assert result["GBP Equity"].iloc[-1] == pytest.approx(4000 * 0.9985 * 1.1 * 1.25)
    fx.iloc[:1].to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing required dates"):
        gbp_account_returns(frame, fx_file=path, initial_date="2023-01-02")


def test_selection_gate_and_final_reports_failures_without_reselection(tmp_path):
    settings = fixture(tmp_path)
    path = fx_file(tmp_path, [0.8, 0.8, 0.8, 0.5])
    account = dict(fx_file=str(path), initial_capital_gbp=4000, max_drawdown=0.2)
    # Final reporting preserves both failing finalists.
    final = evaluate_frozen(**settings, account=account)
    assert len(final) == 2
    assert not final["GBP Drawdown Limit Passed"].any()
    db = tmp_path / "output/Frozen_Final_Evaluation.db"
    contract = json.loads((tmp_path / "output/audit-contract.json").read_text())
    assert audit_databases([db], contract=contract)["status"] == "PASS"
    # Earlier account-selection phase writes an empty, schema-preserving selection.
    chosen = read_table(settings["selection_database"], "Passed Strategies")
    with sqlite3.connect(settings["cache_database"]) as conn:
        write_frame(chosen, "Passed Strategies", conn, if_exists="replace")
    settings.update(
        selection_database=settings["cache_database"], output_dir=tmp_path / "selection_check"
    )
    selected = evaluate_frozen(**settings, account=account, evaluation_kind="selection")
    assert len(selected) == 2
    assert read_table(tmp_path / "selection_check/GBP_Selection.db", "Passed Strategies").empty
    assert len(read_table(tmp_path / "selection_check/GBP_Selection.db", "Summary")) == 2


def test_account_selection_keeps_passing_candidates_and_sources(tmp_path):
    settings = fixture(tmp_path)
    chosen = read_table(settings["selection_database"], "Passed Strategies")
    with sqlite3.connect(settings["cache_database"]) as conn:
        write_frame(chosen, "Passed Strategies", conn, if_exists="replace")
    settings.update(selection_database=settings["cache_database"])
    original = settings["cache_database"].read_bytes()
    evaluate_frozen(
        **settings,
        account=dict(fx_file=str(fx_file(tmp_path, [0.8] * 4)), max_drawdown=0.2),
        evaluation_kind="selection",
    )
    assert len(read_table(tmp_path / "output/GBP_Selection.db", "Passed Strategies")) == 2
    assert settings["cache_database"].read_bytes() == original


def test_preparation_never_overwrites_different_frozen_inputs(tmp_path):
    args = dict(
        model_dir=tmp_path / "model",
        selection_dir=tmp_path / "selection",
        final_dir=tmp_path / "final",
        download_fx=False,
    )
    prepare_research(phase="model", **args)
    for name in ["Final_Test_Results.db", "Selected_Features.txt", "Top_Horizon_Scores.txt"]:
        (args["model_dir"] / name).write_text("fixture")
    prepare_research(phase="selection", **args)
    prepare_research(phase="selection", **args)
    (args["model_dir"] / "Selected_Features.txt").write_text("different")
    with pytest.raises(FileExistsError, match="Conflicting"):
        prepare_research(phase="selection", **args)
    assert (args["selection_dir"] / "Selected_Features.txt").read_text() == "fixture"


def test_preparation_downloads_only_selected_phase_fx(tmp_path, monkeypatch):
    import yfinance

    calls = []

    def download(ticker, **kwargs):
        calls.append((ticker, kwargs))
        return pd.DataFrame({"Close": [1.25, 1.5]}, index=pd.date_range("2019-01-02", periods=2))

    monkeypatch.setattr(yfinance, "download", download)
    args = dict(
        model_dir=tmp_path / "model",
        selection_dir=tmp_path / "selection",
        final_dir=tmp_path / "final",
    )
    prepare_research(phase="model", **args)
    assert calls == []
    for name in ["Final_Test_Results.db", "Selected_Features.txt", "Top_Horizon_Scores.txt"]:
        (args["model_dir"] / name).write_text("fixture")
    prepare_research(phase="selection", **args)
    assert calls[0][0] == "GBPUSD=X"
    assert calls[0][1]["end"] == "2023-01-01"
    fx = pd.read_csv(args["selection_dir"] / "GBP_per_USD.csv")
    assert fx.GBP_per_USD.tolist() == pytest.approx([0.8, 2 / 3])


def test_m2_launcher_configures_threads_before_dispatch(monkeypatch):
    import runpy
    from pathlib import Path
    import os

    root = Path(__file__).resolve().parents[1]
    import sys

    monkeypatch.setattr(sys, "argv", ["Run Research.py"])
    namespace = runpy.run_path(str(root / "Run Research.py"))
    called = []
    monkeypatch.chdir(root)
    for name in [
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "LOKY_MAX_CPU_COUNT",
    ]:
        monkeypatch.setenv(name, "1")
    monkeypatch.setattr(
        runpy,
        "run_path",
        lambda path, **kwargs: called.append(
            (path, os.environ["OMP_NUM_THREADS"], os.environ["LOKY_MAX_CPU_COUNT"])
        ),
    )
    namespace.update(__name__="__main__", __file__=str(root / "Run Research.py"))
    exec(
        compile((root / "Run Research.py").read_text(), str(root / "Run Research.py"), "exec"),
        namespace,
    )
    assert called == [(str(root / "Prepare Research.py"), "2", "2")]

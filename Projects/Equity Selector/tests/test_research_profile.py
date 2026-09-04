"""Cross-launcher research contracts: chronology, frozen identity and search bounds."""

import math
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    return runpy.run_path(str(ROOT / name))


def test_research_periods_and_artifacts_are_separated():
    training = load("Model Fitting.py")["SETTINGS"]
    confirmation = load("Best_Model_Test.py")["SETTINGS"]
    horizon = load("Horizon Score Backtests.py")["SETTINGS"]
    cache = load("Backtest Database.py")
    final = load("Frozen Final Test.py")["SETTINGS"]
    assert training["MODEL_TRAIN_END"] < training["MODEL_VALIDATION_END"]
    assert training["MODEL_VALIDATION_END"] < training["RESEARCH_END"]
    assert training["RESEARCH_END"] < horizon["HORIZON_VALIDATION_START"]
    assert horizon["RESEARCH_END"] < cache["SETTINGS"]["BACKTEST_START"]
    assert cache["SETTINGS"]["BACKTEST_END"] < final["start"]
    for key in ("RESEARCH_START", "RESEARCH_END", "MODEL_TRAIN_END", "MODEL_VALIDATION_END"):
        assert training[key] == confirmation[key]
    assert (
        len(
            {
                training["DATA_DIR"],
                cache["SETTINGS"]["DATA_DIR"],
                cache["FINAL_CACHE_SETTINGS"]["DATA_DIR"],
            }
        )
        == 3
    )
    assert cache["FINAL_CACHE_SETTINGS"]["TRAIN_END"] < final["start"]
    assert cache["FINAL_CACHE_SETTINGS"]["BACKTEST_START"] == final["start"]
    assert cache["FINAL_CACHE_SETTINGS"]["BACKTEST_END"] == final["end"]
    assert (
        str(Path(cache["FINAL_CACHE_SETTINGS"]["DATA_DIR"]) / "Backtest_Database.db")
        == final["cache_database"]
    )


def test_frozen_signal_group_identity_and_bounded_exhaustive_search():
    horizon = load("Horizon Score Backtests.py")["SETTINGS"]
    simulations = load("Backtest Simulations.py")["SETTINGS"]
    precise = load("Precise Backtest.py")["SETTINGS"]
    final = load("Frozen Final Test.py")["SETTINGS"]
    groups = simulations["PORTFOLIO_GROUP_CONFIGURATIONS"]
    assert groups == precise["PORTFOLIO_GROUP_CONFIGURATIONS"] == final["type_configurations"]
    assert groups == horizon["HORIZON_TEST_TYPE_CONFIGURATIONS"]
    for group in groups:
        assert math.isclose(sum(value for key, value in group.items() if key != "Name"), 1.0)
    size = math.prod(
        len(values)
        for horizons in horizon["HORIZON_SCORE_VALUES"].values()
        for values in horizons.values()
    )
    assert size <= horizon["MAX_EXHAUSTIVE_CONFIGURATIONS"]
    assert size == 729
    assert horizon["RANDOM_SCREENS"] == []
    assert horizon["ANALYSIS_MODE"] == "DAILY"
    assert (
        precise["FE_TRADING_FEE"]
        == final["trading_fee"] + final["account"]["execution_cost_fraction"]
    )


def test_cache_launcher_dispatches_selection_dates_without_running_research(monkeypatch):
    import equity_selector.cli

    calls = []
    monkeypatch.setattr(
        equity_selector.cli, "run_stage", lambda stage, **kwargs: calls.append((stage, kwargs))
    )
    runpy.run_path(str(ROOT / "Backtest Database.py"), run_name="__main__")
    assert len(calls) == 1
    assert calls[0][0] == "cache"
    assert calls[0][1]["settings"]["BACKTEST_END"] == "2022-12-30"


def test_cache_launcher_dispatches_final_dates_without_mutating_selection(monkeypatch):
    import equity_selector.cli

    calls = []
    monkeypatch.setattr(
        equity_selector.cli, "run_stage", lambda stage, **kwargs: calls.append(kwargs["settings"])
    )
    source = (
        (ROOT / "Backtest Database.py")
        .read_text()
        .replace('CACHE_PHASE = "selection"', 'CACHE_PHASE = "final"')
    )
    namespace = {"__name__": "__main__"}
    exec(compile(source, str(ROOT / "Backtest Database.py"), "exec"), namespace)
    assert calls[0]["BACKTEST_START"] == "2023-01-01"
    assert calls[0]["TRAIN_END"] == "2022-12-31"
    assert namespace["SETTINGS"]["BACKTEST_END"] == "2022-12-30"

import importlib
import os
import runpy
from pathlib import Path

import pandas as pd
import pytest

from equity_selector.cli import run_stage
from equity_selector.settings import setting, callback, stage_settings
from equity_selector.settings_catalogue import STAGE_KEYS
from equity_selector.validation import train_validation_test_split, screening_training_rows


def test_settings_are_copied_scoped_and_restored_on_failure():
    original = {"VALUES": [1]}
    with pytest.raises(RuntimeError), stage_settings(original):
        value = setting("VALUES")
        value.append(2)
        assert setting("VALUES") == [1]
        raise RuntimeError()
    assert original == {"VALUES": [1]}
    assert setting("VALUES") is None


def test_launcher_override_reaches_stage_and_cli_path_wins(tmp_path, monkeypatch):
    module = importlib.import_module("equity_selector.stages.training")
    seen = {}

    def run():
        seen.update(
            STOCK_TYPE=setting("STOCK_TYPE"), directory=os.environ["EQUITY_SELECTOR_DATA_DIR"]
        )
        return 42

    monkeypatch.setattr(module, "run", run)
    monkeypatch.setenv("EQUITY_SELECTOR_DATA_DIR", "original")
    assert (
        run_stage(
            "training",
            ["--data-dir", str(tmp_path)],
            settings={"STOCK_TYPE": "Fixture", "DATA_DIR": "ignored"},
        )
        == 42
    )
    assert seen == {"STOCK_TYPE": "Fixture", "directory": str(tmp_path)}
    assert os.environ["EQUITY_SELECTOR_DATA_DIR"] == "original"
    with pytest.raises(ValueError, match="Unknown settings"):
        run_stage("training", [], settings={"STOCK_TYEP": "wrong"})


def test_intraday_paths_and_constants_reconfigured_after_prior_import(tmp_path, monkeypatch):
    module = importlib.import_module("equity_selector.stages.intraday")
    original = module.DATABASE
    seen = []
    monkeypatch.setattr(module, "run", lambda: seen.append((module.DATABASE, module.STOCK_TYPE)))
    run_stage("intraday", [], settings={"DATA_DIR": str(tmp_path), "STOCK_TYPE": "Fixture"})
    assert seen == [(tmp_path / "Features_Targets_Data.db", "Fixture")]
    assert module.DATABASE == original


def test_explicit_dates_ignore_later_data_and_match_screening():
    data = pd.DataFrame(
        {"Date": pd.date_range("2020-01-01", periods=10), "Ticker": "A", "x": range(10)}
    )
    with stage_settings(
        {
            "MODEL_TRAIN_END": "2020-01-03",
            "RESEARCH_START": "2020-01-01",
            "MODEL_VALIDATION_END": "2020-01-06",
            "RESEARCH_END": "2020-01-08",
        }
    ):
        train, validation, test = train_validation_test_split(data)
        assert train.x.tolist() == [0, 1, 2]
        assert validation.x.tolist() == [3, 4, 5]
        assert test.x.tolist() == [6, 7]
        pd.testing.assert_frame_equal(screening_training_rows(data), train)
    with stage_settings({"MODEL_TRAIN_END": "2020-01-03"}), pytest.raises(ValueError):
        train_validation_test_split(data)


def test_all_launcher_settings_are_known_and_import_safe():
    root = Path(__file__).parents[1]
    names = {
        "data": "Data_Creation_Screening.py",
        "intraday": "Intraday Conversion.py",
        "training": "Model Fitting.py",
        "final_test": "Best_Model_Test.py",
        "horizons": "Horizon Score Backtests.py",
        "cache": "Backtest Database.py",
        "simulations": "Backtest Simulations.py",
        "precise": "Precise Backtest.py",
    }
    for stage, name in names.items():
        namespace = runpy.run_path(str(root / name))
        assert set(namespace["SETTINGS"]) <= set(STAGE_KEYS[stage])


def test_callable_hook_is_explicit_and_scoped():
    base = lambda x: x + 1
    override = lambda x: x * 2
    with stage_settings(callbacks={"f": override}):
        assert callback("f", base)(3) == 6
    assert callback("f", base)(3) == 4
    with pytest.raises(TypeError), stage_settings(callbacks={"f": 1}):
        pass


def test_catalogue_grid_mode_and_explicit_overrides():
    from equity_selector.settings import choose_catalogue

    generated = [{"name": "Ridge", "params": {"alpha": 3}}]
    legacy = [{"name": "Ridge", "params": {"alpha": 1}}]
    assert choose_catalogue("ALL_CONTINUOUS_MODELS", generated, legacy) == legacy
    with stage_settings({"MODEL_CATALOGUE_MODE": "grid"}):
        assert choose_catalogue("ALL_CONTINUOUS_MODELS", generated, legacy) == generated
    with stage_settings({"ALL_CONTINUOUS_MODELS": []}):
        assert choose_catalogue("ALL_CONTINUOUS_MODELS", generated, legacy) == []


def test_function_kwargs_and_callback_reach_builder():
    from equity_selector.settings import configured

    def builder(frame, scale=1):
        return frame * scale

    with stage_settings({"FUNCTION_KWARGS": {"builder": {"scale": 3}}}):
        assert configured(builder, 2) == 6
        assert configured(builder, 2, scale=4) == 8


def test_all_folds_requirement_uses_complete_validation_calendar():
    from equity_selector.validation import required_validation_folds, research_rows

    data = pd.DataFrame({"Date": pd.date_range("2020-01-01", periods=20), "Ticker": "A"})
    with stage_settings(
        {
            "MODEL_TRAIN_END": "2020-01-04",
            "MODEL_VALIDATION_END": "2020-01-11",
            "RESEARCH_END": "2020-01-15",
        }
    ):
        assert required_validation_folds(data, 3, "all") == 3
        assert len(research_rows(data)) == 15


def test_packages_have_no_interactive_input_calls():
    import ast

    root = Path(__file__).parents[1]
    for package in [
        "equity_selector",
        "main_package",
        "features",
        "targets",
        "models",
        "screening",
    ]:
        for path in (root / package).rglob("*.py"):
            tree = ast.parse(path.read_text())
            calls = [
                node
                for node in ast.walk(tree)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "input"
            ]
            assert not calls, str(path)


def test_cache_missing_dates_raise_before_download_or_prompt(tmp_path, monkeypatch):
    import yfinance

    def forbidden(*args, **kwargs):
        raise AssertionError("Package tried to download or prompt before validating settings")

    monkeypatch.setattr(yfinance, "download", forbidden)
    monkeypatch.setattr("builtins.input", forbidden)
    with pytest.raises(ValueError, match="DOWNLOAD_START"):
        run_stage("cache", [], settings={"DATA_DIR": str(tmp_path)})


def test_explicit_model_selection_uses_identity_or_rank_not_row_order():
    from equity_selector.settings import choose_model_row

    eligible = pd.DataFrame(
        {
            "Test Selection Rank": [2, 1],
            "Model": ["Ridge", "OLS"],
            "Parameters": ['{"alpha":1}', "{}"],
        }
    )
    assert choose_model_row(eligible, "T").Model == "OLS"
    with stage_settings(
        {
            "MODEL_SELECTION_MODE": "explicit",
            "MODEL_SELECTIONS": {"T": {"Model": "Ridge", "Parameters": {"alpha": 1}}},
        }
    ):
        assert choose_model_row(eligible, "T").Model == "Ridge"
    with stage_settings({"MODEL_SELECTION_MODE": "explicit", "MODEL_SELECTIONS": {"T": 2}}):
        assert choose_model_row(eligible, "T").Model == "Ridge"
    with (
        stage_settings({"MODEL_SELECTION_MODE": "explicit"}),
        pytest.raises(ValueError, match="MODEL_SELECTIONS"),
    ):
        choose_model_row(eligible, "T")


def test_horizon_continuation_is_finite_and_configured():
    from equity_selector.settings import run_screen_schedule

    calls = []

    def screen(**kwargs):
        calls.append(kwargs)

    with pytest.raises(ValueError, match="EXTRA_RANDOM_SCREENS"):
        run_screen_schedule(screen, lambda: 2000, [(20, 0.15)], [(40, 0.3)], 1000)
    assert calls == [{"iterations": 20, "threshold": 0.15}, {"iterations": 40, "threshold": 0.3}]
    calls.clear()
    run_screen_schedule(screen, lambda: 500, [(20, 0.15)], [(40, 0.3)], 1000)
    assert calls == [{"iterations": 20, "threshold": 0.15}]


def test_horizon_settings_accessor_survives_portfolio_setting_loop(monkeypatch):
    module = importlib.import_module("equity_selector.stages.horizons")
    # The stage stores each portfolio setting in this module-global loop variable.
    monkeypatch.setattr(module, "setting", {"Name": "Fixture"}, raising=False)
    with stage_settings({"NEAR_BEST_BQ_TOLERANCE": 0.123}):
        assert module.get_setting("NEAR_BEST_BQ_TOLERANCE", 0.002) == 0.123

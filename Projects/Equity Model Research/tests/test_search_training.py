import sqlite3

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from equity_selector.parameters import configuration_key, parse_parameters
from equity_selector.search import add_recommendation_midpoints, recommend_models_to_fit
from equity_selector.training import walk_forward


def search_frame(parameters, midpoint=None):
    return pd.DataFrame(
        {
            "Model": ["Ridge"] * len(parameters),
            "Parameters": parameters,
            "Rank": np.arange(1, len(parameters) + 1),
            "Model Selection Score": np.linspace(1, 0.1, len(parameters)),
            "TPE Score?": 0.0,
            "Midpoint?": midpoint,
        },
        index=pd.Index(np.arange(10, 10 + len(parameters)) * 3, name="row_id"),
    )


def test_numeric_midpoints_exclude_tested_and_preserve_index_state():
    frame = search_frame([{"alpha": 1.0}, {"alpha": 9.0}])
    models, markers = recommend_models_to_fit(
        frame, "continuous", 1, 4, {"Ridge": {"params": {"alpha": [1, 9]}}}
    )
    assert models == [{"name": "Ridge", "params": {"alpha": 5.0}}]
    assert markers == ["alpha"]
    assert frame.index.tolist() == [30, 33]
    summary = pd.concat(
        [
            frame.reset_index(drop=True),
            pd.DataFrame(
                [
                    {
                        "Model": "Ridge",
                        "Parameters": '{"alpha":5}',
                        "TPE Score?": 0.0,
                        "Midpoint?": np.nan,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    result = add_recommendation_midpoints(summary, models, markers)
    assert result["Midpoint?"].iloc[-1] == "alpha"
    assert pd.isna(summary["Midpoint?"].iloc[-1])


def test_categorical_parameters_never_use_values_from_another_column():
    frame = search_frame([{"solver": "a", "alpha": 1}, {"solver": "b", "alpha": 9}])
    models, markers = recommend_models_to_fit(
        frame, "continuous", 1, 10, {"Ridge": {"params": {"solver": ["a", "b"], "alpha": [1, 9]}}}
    )
    assert models
    assert all(model["params"]["solver"] in {"a", "b"} for model in models)
    assert len(models) == len(markers)
    assert len({configuration_key(model) for model in models}) == len(models)


def test_tpe_ordering_is_independent_of_sql_row_order_and_returns_dicts():
    params = [{"alpha": i % 4, "depth": i // 4} for i in range(12)]
    # Observed grid has untested combinations: three diagonal depth groups.
    params = [{"alpha": i, "depth": i % 3} for i in range(12)]
    frame = search_frame(params)
    lookup = {"Ridge": {"params": {"alpha": list(range(12)), "depth": [0, 1, 2]}}}
    np.random.seed(5)
    first, tags = recommend_models_to_fit(frame.copy(), "continuous", 1, 8, lookup)
    np.random.seed(5)
    second, _ = recommend_models_to_fit(
        frame.sample(frac=1, random_state=9), "continuous", 1, 8, lookup
    )
    assert first == second
    assert 0 < len(first) <= 8
    assert len(tags) == len(first)
    assert all(isinstance(model, dict) for model in first)
    tested = {configuration_key({"name": "Ridge", "params": p}) for p in params}
    assert all(configuration_key(model) not in tested for model in first)
    # The top quartile used alpha 0,1,2; exclusively bad alpha values must rank below them.
    assert first[0]["params"]["alpha"] in {0, 1, 2}


def test_converged_midpoint_promotes_to_tpe_without_duplicate_fit():
    frame = search_frame([{"alpha": 1.0}, {"alpha": 2.0}], midpoint="alpha")
    frame["Model Selection Score"] = [0.8, 0.799]
    models, markers = recommend_models_to_fit(
        frame, "continuous", 1, 2, {"Ridge": {"params": {"alpha": [1, 2]}}}
    )
    assert models == markers == []
    assert frame["TPE Score?"].tolist() == [1.0, 1.0]


def test_exhausted_search_is_empty():
    frame = search_frame([{"alpha": 1}])
    models, markers = recommend_models_to_fit(
        frame, "continuous", 1, 8, {"Ridge": {"params": {"alpha": [1]}}}
    )
    assert models == markers == []


def test_nullable_and_tuple_parameters_are_preserved():
    frame = search_frame(
        [
            {"class_weight": None, "hidden_layer_sizes": (10,), "alpha": 1.0},
            {"class_weight": "balanced", "hidden_layer_sizes": (10,), "alpha": 9.0},
        ]
    )
    lookup = {
        "Ridge": {
            "params": {
                "class_weight": [None, "balanced"],
                "hidden_layer_sizes": [(10,)],
                "alpha": [1.0, 9.0],
            }
        }
    }
    models, _ = recommend_models_to_fit(frame, "continuous", 1, 10, lookup)
    assert any(
        model["params"]["class_weight"] is None and model["params"]["alpha"] != 1
        for model in models
    )
    assert all(model["params"]["hidden_layer_sizes"] == (10,) for model in models)


@pytest.mark.parametrize("purge", [0, 2])
def test_walk_forward_boundaries_and_database(purge, tmp_path):
    data = pd.DataFrame(
        {
            "Date": pd.date_range("2024-01-01", periods=30),
            "Ticker": "A",
            "x": np.arange(30.0),
            "target": np.arange(30.0),
        }
    )
    observed = []

    def fit(train, valid, features, target, kind, fold, model):
        assert train.Date.max() < valid.Date.min()
        assert (valid.Date.min() - train.Date.max()).days == purge + 1
        observed.append((train.Date.max(), valid.Date.min(), valid.Date.max()))
        return {
            "Model": model["name"],
            "Parameters": model["params"],
            "Fold": fold,
            "Target": target,
            "Target Type": kind,
            "Rank IC": 1.0,
            "NRMSE": 0.0,
            "R2": 1.0,
        }

    models = [{"name": "OLS", "params": {}}] * 2
    database = tmp_path / "folds.db"
    summary = walk_forward(
        models,
        data,
        ["x"],
        "target",
        purge,
        "continuous",
        "ALPHA",
        pd.DataFrame(),
        True,
        validation_window=2,
        database_path=database,
        fit_function=fit,
        prune_function=lambda models, *args: models,
        role_function=lambda _: "ranking",
    )
    assert len(observed) == 3  # duplicate configs fit only once per fold
    assert observed[-1][2] == pd.Timestamp("2024-01-24")
    assert summary["Fold"].tolist() == [3]
    assert summary["Rank IC Mean"].tolist() == [1.0]
    with sqlite3.connect(database) as connection:
        stored = pd.read_sql_query("SELECT * FROM target__folds", connection)
    assert len(stored) == 3
    assert stored["Parameters"].map(parse_parameters).tolist() == [{}, {}, {}]


def test_all_failed_fits_return_previous_results_without_broken_sql(tmp_path):
    data = pd.DataFrame({"Date": pd.date_range("2024-01-01", periods=30), "Ticker": "A"})
    previous = pd.DataFrame({"Model": ["Old"], "Rank": [1]})
    result = walk_forward(
        [{"name": "Fail", "params": {}}],
        data,
        [],
        "target",
        0,
        "continuous",
        "ALPHA",
        previous,
        False,
        database_path=tmp_path / "failed.db",
        fit_function=lambda *args: None,
        prune_function=lambda models, *args: models,
        role_function=lambda _: "ranking",
    )
    assert_frame_equal(result, previous)


def test_later_failed_model_is_disqualified_but_diagnostic_folds_remain(tmp_path):
    data = pd.DataFrame({"Date": pd.date_range("2024-01-01", periods=30), "Ticker": "A"})
    calls = []

    def fit(train, valid, features, target, kind, fold, model):
        calls.append(fold)
        if fold == 2:
            return None
        return {"Model": model["name"], "Parameters": model["params"], "Fold": fold, "Rank IC": 1.0}

    database = tmp_path / "later_failure.db"
    result = walk_forward(
        [{"name": "Fails Later", "params": {}}],
        data,
        [],
        "target",
        0,
        "continuous",
        "ALPHA",
        pd.DataFrame(),
        True,
        validation_window=2,
        database_path=database,
        fit_function=fit,
        prune_function=lambda models, *args: models,
        role_function=lambda _: "ranking",
    )
    assert calls == [1, 2]
    assert result.empty
    with sqlite3.connect(database) as connection:
        stored = pd.read_sql_query("SELECT * FROM target__folds", connection)
    assert stored["Fold"].tolist() == [1]
